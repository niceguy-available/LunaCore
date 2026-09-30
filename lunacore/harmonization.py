"""Phase 2 — radiometric and geometric harmonisation.

Brings the source and reference to the same ground resolution and a
comparable radiometry before the matcher sees them:

* **PCA** collapses a hyperspectral IIRS cube to its first principal
  component. The band covariance is accumulated strip by strip (two passes
  over the memory-mapped cube), so a cube larger than RAM never has to be
  loaded.
* **Scale ratio** ``S = GSD_ref / GSD_src`` drives an anti-aliased,
  NaN-aware area reduction (exact integer block means, then a fractional
  ``INTER_AREA`` step). The mapping is tracked as a 3x3 matrix so match
  coordinates can be lifted back to full resolution exactly.
* **CLAHE** on an 8x8 tile grid equalises local contrast (softening shadows)
  after a robust percentile stretch.
* **Geo-prior**: when the label carries corner coordinates, the reduced
  source is pre-warped into the reference frame so the matcher only has to
  resolve a small residual (and LoFTR, which is not rotation invariant,
  sees both images north-up).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np

from .config import HarmonizationConfig
from .errors import HarmonizationError
from .ingestion import Pds4Product, lonlat_to_pixels
from .wms import ReferenceMap

log = logging.getLogger(__name__)

RowReader = Callable[[int, int], np.ndarray]


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------

def compute_scale_ratio(gsd_reference_m: float, gsd_source_m: float) -> float:
    """``S = GSD_ref / GSD_src``: how many source pixels span one reference pixel."""
    if gsd_reference_m <= 0 or gsd_source_m <= 0:
        raise HarmonizationError("GSD values must be positive")
    return gsd_reference_m / gsd_source_m


def scale_matrix(sx: float, sy: float) -> np.ndarray:
    """Pixel-centre-correct resampling matrix: x' = (x + 0.5) * sx - 0.5."""
    return np.array([[sx, 0.0, 0.5 * sx - 0.5],
                     [0.0, sy, 0.5 * sy - 0.5],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _block_mean(strip: np.ndarray, k: int) -> np.ndarray:
    """NaN-aware k x k block mean of a 2D strip whose dims are multiples of k."""
    h, w = strip.shape
    blocks = strip.reshape(h // k, k, w // k, k)
    valid = np.isfinite(blocks)
    total = np.where(valid, blocks, 0.0).sum(axis=(1, 3), dtype=np.float64)
    count = valid.sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (total / count).astype(np.float32)
    out[count == 0] = np.nan
    return out


def reduce_image(reader: RowReader, n_rows: int, n_cols: int, factor: float,
                 strip_rows: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """Down-sample a 2D raster by ``factor`` (>= 1) while streaming rows.

    Returns:
        (reduced float32 image with NaN nodata, 3x3 matrix mapping input
        pixel coordinates to reduced pixel coordinates)
    """
    if factor < 1.0:
        raise HarmonizationError("reduce_image only down-samples (factor >= 1)")
    k = max(1, int(math.floor(factor)))
    rows_k, cols_k = n_rows // k, n_cols // k
    if rows_k == 0 or cols_k == 0:
        raise HarmonizationError(f"image {n_rows}x{n_cols} too small for factor {factor}")

    step = max(k, (strip_rows // k) * k)
    parts = []
    for r0 in range(0, rows_k * k, step):
        r1 = min(r0 + step, rows_k * k)
        strip = reader(r0, r1)[:, : cols_k * k]
        parts.append(_block_mean(strip, k) if k > 1 else strip.astype(np.float32, copy=False))
    blocked = np.vstack(parts)
    D = scale_matrix(1.0 / k, 1.0 / k)

    residual = factor / k
    if residual > 1.0 + 1e-6:
        new_w = max(1, int(round(cols_k / residual)))
        new_h = max(1, int(round(rows_k / residual)))
        valid = np.isfinite(blocked).astype(np.float32)
        filled = np.where(valid > 0, blocked, 0.0).astype(np.float32)
        area = cv2.resize(filled, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cover = cv2.resize(valid, (new_w, new_h), interpolation=cv2.INTER_AREA)
        with np.errstate(invalid="ignore", divide="ignore"):
            blocked = area / cover     # renormalise so edges are not darkened
        blocked[cover < 0.5] = np.nan
        D = scale_matrix(new_w / cols_k, new_h / rows_k) @ D
    return blocked.astype(np.float32, copy=False), D


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

@dataclass
class PCAResult:
    mean: np.ndarray
    component: np.ndarray            # unit eigenvector of the largest eigenvalue
    explained_variance_ratio: float
    scale: Optional[np.ndarray] = None   # per-band std when standardised


def fit_pca_streaming(product: Pds4Product, strip_rows: int = 256,
                      standardize: bool = False) -> PCAResult:
    """First principal component of a (bands, lines, samples) cube.

    Accumulates ``sum x`` and ``sum x x^T`` in float64 over strips, forms the
    covariance ``C = E[xx^T] - mu mu^T`` and takes the eigenvector of the
    largest eigenvalue (``numpy.linalg.eigh`` on the symmetric matrix). Pixels
    with any non-finite band are excluded.
    """
    if not product.is_multiband:
        raise HarmonizationError("PCA requires a multi-band product")
    bands = product.n_bands
    n_rows = product.shape_2d[0]
    s1 = np.zeros(bands, dtype=np.float64)
    s2 = np.zeros((bands, bands), dtype=np.float64)
    n = 0
    for r0 in range(0, n_rows, strip_rows):
        x = product.read_rows(r0, min(r0 + strip_rows, n_rows)).reshape(bands, -1)
        x = x[:, np.isfinite(x).all(axis=0)].astype(np.float64)
        if x.size == 0:
            continue
        s1 += x.sum(axis=1)
        s2 += x @ x.T
        n += x.shape[1]
    if n < bands + 1:
        raise HarmonizationError(f"only {n} fully-valid pixels for {bands}-band PCA")

    mean = s1 / n
    cov = s2 / n - np.outer(mean, mean)
    scale = None
    if standardize:
        scale = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        cov = cov / np.outer(scale, scale)
    evals, evecs = np.linalg.eigh(cov)          # ascending eigenvalues
    component = evecs[:, -1]
    # Eigenvectors are sign-ambiguous; orient PC1 to correlate positively with
    # mean brightness so the projected image "looks like" an albedo image.
    if component.sum() < 0:
        component = -component
    ratio = float(evals[-1] / max(evals.sum(), 1e-30))
    log.info("PCA: %d bands, %d pixels, PC1 explains %.1f%% of variance", bands, n, 100 * ratio)
    return PCAResult(mean=mean, component=component, explained_variance_ratio=ratio, scale=scale)


def pca_row_reader(product: Pds4Product, pca: PCAResult) -> RowReader:
    """Row reader yielding the PC1 projection ``(x - mu) . v`` of each strip."""
    weights = pca.component / pca.scale if pca.scale is not None else pca.component
    offset = float(pca.mean @ weights)
    w32 = weights.astype(np.float32)

    def read(r0: int, r1: int) -> np.ndarray:
        cube = product.read_rows(r0, r1)                      # (B, h, w)
        return (np.tensordot(w32, cube, axes=(0, 0)) - offset).astype(np.float32)

    return read


# ---------------------------------------------------------------------------
# Radiometric normalisation
# ---------------------------------------------------------------------------

def normalize_to_uint8(image: np.ndarray, valid: np.ndarray, cfg: HarmonizationConfig
                       ) -> np.ndarray:
    """Robust percentile stretch to uint8 followed by 8x8-tile CLAHE.

    CLAHE clips each tile's histogram and remaps intensities through its CDF
    so the local CDF approaches a straight line; bilinear blending between
    tiles avoids block artefacts.
    """
    out = np.zeros(image.shape, dtype=np.uint8)
    if not valid.any():
        raise HarmonizationError("image has no valid pixels")
    lo, hi = np.percentile(image[valid], [cfg.percentile_low, cfg.percentile_high])
    if hi <= lo:
        raise HarmonizationError("image has no dynamic range")
    stretched = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    stretched[~valid] = 0.0
    out[:] = (stretched * 255.0 + 0.5).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit,
                            tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
    out = clahe.apply(out)
    out[~valid] = 0
    return out


# ---------------------------------------------------------------------------
# Geo-prior
# ---------------------------------------------------------------------------

def geo_to_reference_pixels(lonlat: np.ndarray, reference: ReferenceMap) -> np.ndarray:
    """Map planetocentric (lon, lat) to reference pixel centres, reprojecting if needed."""
    if reference.crs is not None and reference.crs.is_projected:
        from rasterio.warp import transform as rio_transform
        from .geo import moon_crs

        xs, ys = rio_transform(moon_crs(), reference.crs, lonlat[:, 0].tolist(),
                               lonlat[:, 1].tolist())
        lonlat = np.column_stack([xs, ys])
    return lonlat_to_pixels(lonlat, reference.transform)


def compute_prior(product: Pds4Product, reference: ReferenceMap, mode: str
                  ) -> Optional[np.ndarray]:
    """Homography from full-resolution source pixels to reference pixels.

    ``corners``: 4-point homography from the label's corner lat/lon.
    ``bbox``: axis-aligned mapping of the image extent onto the bounding box
    (valid for north-up, map-projected products such as TMC-2 orthoimages).
    ``scale``: no positional prior. ``auto``: corners, else scale.
    """
    if mode == "scale":
        return None
    corners = product.corner_pixels()
    if mode in ("auto", "corners") and corners is not None:
        pix, ll = corners
        ref_pix = geo_to_reference_pixels(ll, reference)
        H, _ = cv2.findHomography(pix, ref_pix, 0)
        if H is None:
            raise HarmonizationError("corner coordinates are degenerate")
        return H
    if mode == "corners":
        raise HarmonizationError("prior='corners' requested but label has no corners")
    if mode == "bbox":
        if product.bbox is None:
            raise HarmonizationError("prior='bbox' requested but label has no bounding box")
        b = product.bbox
        h, w = product.shape_2d
        pix = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)
        ll = np.array([[b.min_lon, b.max_lat], [b.max_lon, b.max_lat],
                       [b.max_lon, b.min_lat], [b.min_lon, b.min_lat]])
        H, _ = cv2.findHomography(pix, geo_to_reference_pixels(ll, reference), 0)
        return H
    return None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class HarmonizedPair:
    """Everything the matcher and the warp need, at a common ground resolution.

    Coordinate frames (all OpenCV pixel-centre conventions):
      * ``src full``  —D_src→  ``src small``  —H0→  ``src canvas``
      * ``ref full``  —D_ref→  ``ref small`` (= reference matching canvas)
    """

    src_canvas_u8: np.ndarray
    src_canvas_valid: np.ndarray
    ref_u8: np.ndarray
    ref_valid: np.ndarray
    src_small: np.ndarray            # radiometric (float32, NaN) at matching GSD
    D_src: np.ndarray
    D_ref: np.ndarray
    H0: np.ndarray                   # src small -> src canvas
    matching_gsd_m: float
    scale_ratio: float
    prior_used: str
    info: Dict[str, object] = field(default_factory=dict)


def harmonize(product: Pds4Product, reference: ReferenceMap, cfg: HarmonizationConfig
              ) -> HarmonizedPair:
    """Run Phase 2 on a loaded source product and reference map."""
    if product.gsd_m is None:
        raise HarmonizationError("source GSD unknown; pass it explicitly")
    info: Dict[str, object] = {}

    if product.is_multiband:
        pca = fit_pca_streaming(product, cfg.strip_rows, cfg.pca_standardize)
        reader = pca_row_reader(product, pca)
        info["pca_explained_variance_ratio"] = pca.explained_variance_ratio
    else:
        reader = product.read_rows

    matching_gsd = max(product.gsd_m, reference.gsd_m)
    scale_ratio = compute_scale_ratio(reference.gsd_m, product.gsd_m)
    src_factor = matching_gsd / product.gsd_m
    ref_factor = matching_gsd / reference.gsd_m
    log.info("scale ratio S=%.3f; matching at %.2f m/px (src /%.2f, ref /%.2f)",
             scale_ratio, matching_gsd, src_factor, ref_factor)

    n_rows, n_cols = product.shape_2d
    src_small, D_src = reduce_image(reader, n_rows, n_cols, src_factor, cfg.strip_rows)
    ref_img = reference.image
    ref_small, D_ref = reduce_image(lambda a, b: ref_img[a:b], *ref_img.shape, ref_factor,
                                    cfg.strip_rows)

    prior_full = compute_prior(product, reference, cfg.prior)
    if prior_full is not None:
        # src small -> src full -> ref full -> ref small
        H0 = D_ref @ prior_full @ np.linalg.inv(D_src)
        H0 /= H0[2, 2]
        canvas_hw = ref_small.shape
        prior_used = cfg.prior if cfg.prior != "auto" else "corners"
    else:
        H0 = np.eye(3)
        canvas_hw = src_small.shape
        prior_used = "scale"

    src_valid_small = np.isfinite(src_small)
    filled = np.where(src_valid_small, src_small, 0.0).astype(np.float32)
    if prior_used == "scale":
        canvas, canvas_valid = filled, src_valid_small
    else:
        dsize = (canvas_hw[1], canvas_hw[0])
        canvas = cv2.warpPerspective(filled, H0, dsize, flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        canvas_valid = cv2.warpPerspective(src_valid_small.astype(np.uint8), H0, dsize,
                                           flags=cv2.INTER_NEAREST) > 0
    if canvas_valid.sum() < 64:
        raise HarmonizationError("source footprint does not overlap the reference")

    ref_valid = np.isfinite(ref_small)
    pair = HarmonizedPair(
        src_canvas_u8=normalize_to_uint8(canvas, canvas_valid, cfg),
        src_canvas_valid=canvas_valid,
        ref_u8=normalize_to_uint8(np.where(ref_valid, ref_small, 0.0), ref_valid, cfg),
        ref_valid=ref_valid,
        src_small=src_small, D_src=D_src, D_ref=D_ref, H0=H0,
        matching_gsd_m=matching_gsd, scale_ratio=scale_ratio, prior_used=prior_used,
        info=info,
    )
    return pair
