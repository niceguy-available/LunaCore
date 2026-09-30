"""Phase 3 — deep feature matching with spatially uniform selection.

The default matcher is LoFTR (kornia): a detector-free transformer whose
self/cross attention layers compare every coarse patch of one image with every
patch of the other, ``softmax(QK^T / sqrt(d_k)) V``, before a fine stage
refines each match to sub-pixel precision. A SIFT matcher is provided as a
CPU/offline fallback behind the same interface.

VRAM is bounded by (a) never feeding the network more than ``max_side_px``
pixels per side, (b) tiling when the geo-prior has already put both images
in the same frame, (c) fp16 autocast on CUDA, and (d) ``torch.inference_mode``
with tensors released after every call.

After matching, :func:`quadtree_distribute` enforces an even spread: the image
is recursively split into quadrants while a node holds more than ``capacity``
matches, and each leaf keeps only its most confident matches, so a single
boulder field cannot dominate the estimate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple

import cv2
import numpy as np

from .config import MatchingConfig
from .errors import MatchingError

log = logging.getLogger(__name__)


@dataclass
class Matches:
    """Correspondences: ``pts0[i]`` in the source canvas matches ``pts1[i]`` in the reference."""

    pts0: np.ndarray        # (N, 2) float64 x, y
    pts1: np.ndarray        # (N, 2) float64
    confidence: np.ndarray  # (N,) float64 in [0, 1]

    def __len__(self) -> int:
        return len(self.pts0)

    def subset(self, idx: np.ndarray) -> "Matches":
        return Matches(self.pts0[idx], self.pts1[idx], self.confidence[idx])

    @staticmethod
    def empty() -> "Matches":
        return Matches(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0))

    @staticmethod
    def concat(parts: List["Matches"]) -> "Matches":
        parts = [p for p in parts if len(p)]
        if not parts:
            return Matches.empty()
        return Matches(np.vstack([p.pts0 for p in parts]), np.vstack([p.pts1 for p in parts]),
                       np.concatenate([p.confidence for p in parts]))


class Matcher(Protocol):
    def match(self, img0: np.ndarray, img1: np.ndarray) -> Matches:
        """Match two uint8 grayscale images; coordinates in each image's own pixels."""


# ---------------------------------------------------------------------------
# LoFTR
# ---------------------------------------------------------------------------

_MODEL_CACHE: Dict[Tuple[str, str], object] = {}


def _resolve_device(name: str) -> str:
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _fit_to_network(img: np.ndarray, max_side: int) -> Tuple[np.ndarray, float, float]:
    """Resize so the longest side <= max_side and both sides are multiples of 8."""
    h, w = img.shape
    s = min(1.0, max_side / max(h, w))
    nh, nw = max(8, int(round(h * s / 8)) * 8), max(8, int(round(w * s / 8)) * 8)
    if (nh, nw) == (h, w):
        return img, 1.0, 1.0
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, (nw, nh), interpolation=interp), nw / w, nh / h


def _unscale(pts: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Invert a pixel-centre-correct resize."""
    return np.column_stack([(pts[:, 0] + 0.5) / sx - 0.5, (pts[:, 1] + 0.5) / sy - 0.5])


class LoFTRMatcher:
    """kornia LoFTR wrapper with model caching and memory-conscious inference."""

    def __init__(self, weights: str = "outdoor", device: str = "auto",
                 half_precision: bool = True, max_side_px: int = 1024) -> None:
        self.weights = weights
        self.device = _resolve_device(device)
        self.half = half_precision and self.device == "cuda"
        self.max_side = max_side_px
        self._model = None

    def _load(self):
        key = (self.weights, self.device)
        if key not in _MODEL_CACHE:
            import torch
            import kornia.feature as KF

            if self.weights in ("outdoor", "indoor", "indoor_new"):
                model = KF.LoFTR(pretrained=self.weights)
            else:
                model = KF.LoFTR(pretrained=None)
                state = torch.load(self.weights, map_location="cpu", weights_only=True)
                model.load_state_dict(state.get("state_dict", state))
            _MODEL_CACHE[key] = model.eval().to(self.device)
            log.info("loaded LoFTR (%s) on %s", self.weights, self.device)
        return _MODEL_CACHE[key]

    def match(self, img0: np.ndarray, img1: np.ndarray) -> Matches:
        import torch

        model = self._model or self._load()
        self._model = model
        a, ax, ay = _fit_to_network(img0, self.max_side)
        b, bx, by = _fit_to_network(img1, self.max_side)
        t0 = torch.from_numpy(a).to(self.device, torch.float32)[None, None] / 255.0
        t1 = torch.from_numpy(b).to(self.device, torch.float32)[None, None] / 255.0
        try:
            amp_device = "cuda" if self.device == "cuda" else "cpu"
            with torch.inference_mode(), torch.autocast(
                    device_type=amp_device, dtype=torch.float16, enabled=self.half):
                out = model({"image0": t0, "image1": t1})
            k0 = out["keypoints0"].float().cpu().numpy().astype(np.float64)
            k1 = out["keypoints1"].float().cpu().numpy().astype(np.float64)
            conf = out["confidence"].float().cpu().numpy().astype(np.float64)
        except RuntimeError as exc:
            raise MatchingError(f"LoFTR inference failed: {exc}") from exc
        finally:
            del t0, t1
            if self.device == "cuda":
                torch.cuda.empty_cache()
        return Matches(_unscale(k0, ax, ay), _unscale(k1, bx, by), conf)


# ---------------------------------------------------------------------------
# SIFT fallback
# ---------------------------------------------------------------------------

class SIFTMatcher:
    """Classical fallback: SIFT + Lowe ratio test + mutual nearest neighbour."""

    def __init__(self, n_features: int = 8000, ratio: float = 0.8) -> None:
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        self.ratio = ratio

    def match(self, img0: np.ndarray, img1: np.ndarray) -> Matches:
        k0, d0 = self.sift.detectAndCompute(img0, None)
        k1, d1 = self.sift.detectAndCompute(img1, None)
        if d0 is None or d1 is None or len(k0) < 2 or len(k1) < 2:
            return Matches.empty()
        bf = cv2.BFMatcher(cv2.NORM_L2)
        fwd = bf.knnMatch(d0, d1, k=2)
        back = {m.queryIdx: m.trainIdx for m in bf.match(d1, d0)}
        keep = [(m.queryIdx, m.trainIdx, 1.0 - m.distance / max(n.distance, 1e-9))
                for m, n in (p for p in fwd if len(p) == 2)
                if m.distance < self.ratio * n.distance and back.get(m.trainIdx) == m.queryIdx]
        if not keep:
            return Matches.empty()
        q, t, c = map(np.asarray, zip(*keep))
        p0 = np.array([k0[i].pt for i in q], dtype=np.float64)
        p1 = np.array([k1[i].pt for i in t], dtype=np.float64)
        return Matches(p0, p1, np.clip(c.astype(np.float64), 0.0, 1.0))


def build_matcher(cfg: MatchingConfig) -> Matcher:
    if cfg.matcher == "loftr":
        return LoFTRMatcher(cfg.loftr_weights, cfg.device, cfg.half_precision, cfg.max_side_px)
    if cfg.matcher == "sift":
        return SIFTMatcher()
    raise MatchingError(f"unknown matcher {cfg.matcher!r}")


# ---------------------------------------------------------------------------
# Spatial distribution
# ---------------------------------------------------------------------------

def quadtree_distribute(matches: Matches, width: int, height: int, max_depth: int = 6,
                        capacity: int = 4, max_matches: Optional[int] = None) -> Matches:
    """Select an evenly spread subset of matches with an adaptive quad-tree.

    Nodes are split into four children while they hold more than ``capacity``
    matches and are shallower than ``max_depth``; each leaf then keeps its
    ``capacity`` most confident matches. Sparse regions keep everything, dense
    clusters are thinned. Finally, if ``max_matches`` is set, the most
    confident survivors are kept.
    """
    n = len(matches)
    if n == 0:
        return matches
    pts = matches.pts0
    conf = matches.confidence
    kept: List[np.ndarray] = []
    stack = [(np.arange(n), 0.0, 0.0, float(width), float(height), 0)]
    while stack:
        idx, x0, y0, x1, y1, depth = stack.pop()
        if len(idx) <= capacity:
            kept.append(idx)
            continue
        if depth >= max_depth:
            kept.append(idx[np.argsort(-conf[idx], kind="stable")[:capacity]])
            continue
        xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        right = pts[idx, 0] >= xm
        below = pts[idx, 1] >= ym
        for r, b, box in ((False, False, (x0, y0, xm, ym)), (True, False, (xm, y0, x1, ym)),
                          (False, True, (x0, ym, xm, y1)), (True, True, (xm, ym, x1, y1))):
            child = idx[(right == r) & (below == b)]
            if len(child):
                stack.append((child, *box, depth + 1))
    sel = np.sort(np.concatenate(kept))
    if max_matches is not None and len(sel) > max_matches:
        sel = np.sort(sel[np.argsort(-conf[sel], kind="stable")[:max_matches]])
    return matches.subset(sel)


def _mask_filter(matches: Matches, valid0: np.ndarray, valid1: np.ndarray,
                 erosion_px: int) -> Matches:
    """Drop matches on or near no-data (image borders, fill values)."""
    if len(matches) == 0:
        return matches
    if erosion_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_px + 1,) * 2)
        valid0 = cv2.erode(valid0.astype(np.uint8), kernel) > 0
        valid1 = cv2.erode(valid1.astype(np.uint8), kernel) > 0

    def inside(pts: np.ndarray, valid: np.ndarray) -> np.ndarray:
        h, w = valid.shape
        xi = np.rint(pts[:, 0]).astype(np.int64)
        yi = np.rint(pts[:, 1]).astype(np.int64)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        ok[ok] = valid[yi[ok], xi[ok]]
        return ok

    return matches.subset(np.flatnonzero(inside(matches.pts0, valid0)
                                         & inside(matches.pts1, valid1)))


def _tiles(h: int, w: int, tile: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    step = max(1, tile - overlap)
    ys = list(range(0, max(h - tile, 0) + 1, step)) or [0]
    xs = list(range(0, max(w - tile, 0) + 1, step)) or [0]
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    return [(y, min(y + tile, h), x, min(x + tile, w)) for y in ys for x in xs]


def match_images(img0: np.ndarray, valid0: np.ndarray, img1: np.ndarray, valid1: np.ndarray,
                 matcher: Matcher, cfg: MatchingConfig, same_frame: bool) -> Matches:
    """Match source canvas ``img0`` to reference ``img1`` and distribute the result.

    When ``same_frame`` is true (a geo-prior already aligned the canvases) and
    the images exceed the network size, co-located tiles are matched
    independently at full matching resolution; otherwise the matcher sees the
    whole (down-scaled) pair once.
    """
    h, w = img0.shape
    if same_frame and img0.shape == img1.shape and max(h, w) > cfg.max_side_px:
        tile = cfg.max_side_px
        parts = []
        for y0, y1, x0, x1 in _tiles(h, w, tile, overlap=tile // 8):
            if valid0[y0:y1, x0:x1].mean() < 0.1 or valid1[y0:y1, x0:x1].mean() < 0.1:
                continue
            m = matcher.match(img0[y0:y1, x0:x1], img1[y0:y1, x0:x1])
            off = np.array([x0, y0], dtype=np.float64)
            parts.append(Matches(m.pts0 + off, m.pts1 + off, m.confidence))
        raw = Matches.concat(parts)
    else:
        raw = matcher.match(img0, img1)

    raw = raw.subset(np.flatnonzero(raw.confidence >= cfg.min_confidence))
    raw = _mask_filter(raw, valid0, valid1, cfg.mask_erosion_px)
    out = quadtree_distribute(raw, w, h, cfg.quadtree_max_depth, cfg.quadtree_capacity,
                              cfg.max_matches)
    log.info("matching: %d raw -> %d after confidence/mask/quad-tree", len(raw), len(out))
    return out
