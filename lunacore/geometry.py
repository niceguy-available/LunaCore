"""Phases 4 & 5 — robust homography estimation and the final warp.

Phase 4 (outlier rejection)
    MAGSAC++ (OpenCV USAC) finds the consensus set without a hard inlier
    threshold by marginalising over noise scales. The consensus homography is
    then re-solved on all inliers with the Hartley-normalised DLT: stacking two
    rows per correspondence gives ``A h = 0`` and the least-squares solution is
    the right singular vector of ``A`` with the smallest singular value (the
    eigenvector of ``A^T A`` with the smallest eigenvalue). A Levenberg-
    Marquardt pass then minimises the geometric (pixel) reprojection error,
    which the algebraic DLT error only approximates.

Phase 5 (warp)
    ``[x' y' w']^T = H [x y 1]^T`` and ``X = x'/w'``, ``Y = y'/w'``. The
    warp is performed as an inverse mapping with bicubic interpolation (a
    weighted 4x4 = 16-pixel neighbourhood) from the anti-aliased, reduced
    source, so down-sampling never aliases.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from .config import GeometryConfig
from .errors import GeometryError
from .matching import Matches

log = logging.getLogger(__name__)

_CV_MAX_DIM = 32767   # cv2.remap/warp* coordinate limit (SHRT_MAX)


# ---------------------------------------------------------------------------
# Homography algebra
# ---------------------------------------------------------------------------

def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Project (N, 2) points: homogeneous multiply then divide by w'."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ph = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    w = ph[:, 2:3]
    if np.any(np.abs(w) < 1e-12):
        raise GeometryError("point mapped to infinity (w' = 0)")
    return ph[:, :2] / w


def _hartley_normalization(pts: np.ndarray) -> np.ndarray:
    """Similarity T so T·pts has zero mean and mean distance sqrt(2)."""
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2.0) / max(d, 1e-12)
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]], dtype=np.float64)


def normalized_dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares homography via normalised DLT and SVD (>= 4 correspondences)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if len(src) < 4 or len(src) != len(dst):
        raise GeometryError("DLT needs >= 4 matching point pairs")
    Ts, Td = _hartley_normalization(src), _hartley_normalization(dst)
    s = apply_homography(Ts, src)
    d = apply_homography(Td, dst)
    n = len(s)
    A = np.zeros((2 * n, 9), dtype=np.float64)
    x, y, u, v = s[:, 0], s[:, 1], d[:, 0], d[:, 1]
    A[0::2, 0:3] = np.column_stack([-x, -y, -np.ones(n)])
    A[0::2, 6:9] = np.column_stack([u * x, u * y, u])
    A[1::2, 3:6] = np.column_stack([-x, -y, -np.ones(n)])
    A[1::2, 6:9] = np.column_stack([v * x, v * y, v])
    _, sv, Vt = np.linalg.svd(A, full_matrices=False)
    if sv[-2] < 1e-12 * sv[0]:
        raise GeometryError("degenerate configuration (collinear points)")
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    if abs(H[2, 2]) < 1e-15:
        raise GeometryError("DLT produced a homography with H[2,2] = 0")
    return H / H[2, 2]


def refine_homography_lm(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Levenberg-Marquardt minimisation of the forward reprojection error."""
    from scipy.optimize import least_squares

    def residual(h: np.ndarray) -> np.ndarray:
        Hc = np.append(h, 1.0).reshape(3, 3)
        ph = np.column_stack([src, np.ones(len(src))]) @ Hc.T
        return (ph[:, :2] / ph[:, 2:3] - dst).ravel()

    h0 = (H / H[2, 2]).ravel()[:8]
    res = least_squares(residual, h0, method="lm", xtol=1e-12, ftol=1e-12)
    Hr = np.append(res.x, 1.0).reshape(3, 3)
    before = np.sqrt(np.mean(residual(h0) ** 2))
    after = np.sqrt(np.mean(res.fun ** 2))
    return Hr if np.all(np.isfinite(Hr)) and after <= before else H


def reprojection_errors(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    return np.linalg.norm(apply_homography(H, src) - dst, axis=1)


def validate_homography(H: np.ndarray, width: int, height: int, max_scale: float) -> None:
    """Reject non-finite, orientation-flipping or wildly distorting homographies."""
    if H is None or not np.all(np.isfinite(H)):
        raise GeometryError("homography is not finite")
    A = H[:2, :2] / H[2, 2]
    if np.linalg.det(A) <= 0:
        raise GeometryError("homography flips orientation")
    sv = np.linalg.svd(A, compute_uv=False)
    if sv.max() > max_scale or sv.min() < 1.0 / max_scale:
        raise GeometryError(f"implausible scale change {sv.min():.3f}..{sv.max():.3f}")
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
    w = np.column_stack([corners, np.ones(4)]) @ H[2]
    if np.any(w <= 0):
        raise GeometryError("homography maps part of the image through infinity")


# ---------------------------------------------------------------------------
# Phase 4: robust estimation
# ---------------------------------------------------------------------------

@dataclass
class HomographyEstimate:
    H: np.ndarray               # source canvas -> reference (matching resolution)
    inliers: np.ndarray         # boolean mask over the input matches
    rmse_px: float              # inlier reprojection RMSE (matching pixels)

    @property
    def n_inliers(self) -> int:
        return int(self.inliers.sum())


def estimate_homography(matches: Matches, cfg: GeometryConfig,
                        canvas_wh: Tuple[int, int]) -> HomographyEstimate:
    """MAGSAC++ consensus, SVD re-fit on inliers and LM polishing."""
    if len(matches) < max(4, cfg.min_inliers):
        raise GeometryError(f"only {len(matches)} matches; need >= {cfg.min_inliers}")
    src = matches.pts0.astype(np.float64)
    dst = matches.pts1.astype(np.float64)
    H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, cfg.magsac_threshold_px,
                                 maxIters=cfg.magsac_max_iters,
                                 confidence=cfg.magsac_confidence)
    if H is None or mask is None:
        raise GeometryError("MAGSAC++ found no consensus")
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < cfg.min_inliers:
        raise GeometryError(f"only {inliers.sum()} inliers; need >= {cfg.min_inliers}")

    H = normalized_dlt(src[inliers], dst[inliers])
    H = refine_homography_lm(H, src[inliers], dst[inliers])
    # One re-classification with the refined model recovers inliers MAGSAC's
    # early model rejected at the margin.
    err = reprojection_errors(H, src, dst)
    grown = err < cfg.magsac_threshold_px
    if grown.sum() >= inliers.sum():
        inliers = grown
        H = refine_homography_lm(normalized_dlt(src[inliers], dst[inliers]),
                                 src[inliers], dst[inliers])
    validate_homography(H, *canvas_wh, cfg.max_scale_change)
    rmse = float(np.sqrt(np.mean(reprojection_errors(H, src[inliers], dst[inliers]) ** 2)))
    log.info("MAGSAC++: %d/%d inliers, RMSE %.3f px", inliers.sum(), len(src), rmse)
    return HomographyEstimate(H=H / H[2, 2], inliers=inliers, rmse_px=rmse)


def refine_ecc(src_img: np.ndarray, src_valid: np.ndarray, ref_img: np.ndarray,
               ref_valid: np.ndarray, H: np.ndarray, cfg: GeometryConfig
               ) -> Tuple[np.ndarray, float, float]:
    """Dense sub-pixel polish with the Enhanced Correlation Coefficient.

    The source is first warped into the reference frame with ``H``; ECC then
    estimates the small residual ``W`` with ``warped(W x) ~ ref(x)`` over the
    common footprint only, and the refined model is ``H' = W^-1 H``.
    Returns ``(H', cc_before, cc_after)``; ``H`` is returned unchanged if ECC
    diverges, does not improve the correlation, or moves any corner by more
    than 3 px (a sign it locked onto the wrong structure).
    """
    h, w = ref_img.shape
    ref_f = ref_img.astype(np.float32)
    warped = cv2.warpPerspective(src_img.astype(np.float32), H, (w, h), flags=cv2.INTER_LINEAR)
    warped_valid = cv2.warpPerspective(src_valid.astype(np.uint8), H, (w, h),
                                       flags=cv2.INTER_NEAREST) > 0
    mask = (warped_valid & ref_valid).astype(np.uint8)
    mask = cv2.erode(mask, np.ones((7, 7), np.uint8))   # keep blur kernels off the edges
    if mask.sum() < 1024:
        log.info("ECC refinement skipped: overlap too small")
        return H, float("nan"), float("nan")
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, cfg.ecc_max_iters, cfg.ecc_eps)
    try:
        cc0 = cv2.computeECC(ref_f, warped, mask)
        cc1, W = cv2.findTransformECC(ref_f, warped, np.eye(3, dtype=np.float32),
                                      cv2.MOTION_HOMOGRAPHY, criteria, mask, 5)
    except cv2.error as exc:
        log.info("ECC refinement skipped: %s", str(exc).strip().splitlines()[-1][:160])
        return H, float("nan"), float("nan")
    W = W.astype(np.float64)
    probe = np.array([[0, 0], [w, 0], [w, h], [0, h], [w / 2, h / 2]], dtype=np.float64)
    shift = np.abs(apply_homography(W, probe) - probe).max()
    if not np.isfinite(cc1) or cc1 <= cc0 or shift > 3.0:
        log.info("ECC rejected (cc %.4f -> %.4f, max shift %.2f px)", cc0, cc1, shift)
        return H, float(cc0), float(cc0)
    H_new = np.linalg.inv(W) @ H
    H_new /= H_new[2, 2]
    log.info("ECC refined: cc %.4f -> %.4f (max shift %.3f px)", cc0, cc1, shift)
    return H_new, float(cc0), float(cc1)


# ---------------------------------------------------------------------------
# Phase 5: warp
# ---------------------------------------------------------------------------

def warp_to_reference(src: np.ndarray, M: np.ndarray, out_shape: Tuple[int, int]
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Bicubic warp of a float image with NaN no-data onto the reference grid.

    Args:
        src: float32 source raster (NaN = no data).
        M: 3x3 homography mapping ``src`` pixels to output pixels.
        out_shape: (rows, cols) of the reference grid.

    Returns:
        (warped float32 with NaN outside the footprint, boolean validity mask)
    """
    if max(src.shape) > _CV_MAX_DIM or max(out_shape) > _CV_MAX_DIM:
        raise GeometryError("raster exceeds OpenCV's 32767 px warp limit; tile the output")
    valid = np.isfinite(src)
    filled = np.where(valid, src, 0.0).astype(np.float32)
    dsize = (out_shape[1], out_shape[0])
    warped = cv2.warpPerspective(filled, M, dsize, flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # A pixel is valid only if its whole 4x4 bicubic support was valid: erode
    # the mask by one pixel, then require full bilinear coverage of the rest.
    core = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    support = cv2.warpPerspective(core, M, dsize, flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out_valid = support > 0.999
    warped[~out_valid] = np.nan
    return warped, out_valid
