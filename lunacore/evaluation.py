"""Phase 6 — accuracy metrics and georeferenced output.

* **RMSE** of point residuals, ``sqrt(mean((dx)^2 + (dy)^2))``, reported both
  on the MAGSAC inliers (fit residual) and on held-out checkpoints that did
  not take part in the fit (an honest estimate of registration accuracy).
* **ZNCC** between the warped source and the reference over their common
  footprint, ``sum((R-R̄)(S-S̄)) / sqrt(sum(R-R̄)^2 · sum(S-S̄)^2)``, on raw
  intensities and on gradient magnitude (the latter is meaningful across
  modalities, e.g. infrared vs. optical, where intensities may invert).
* **Uniformity**: fraction of occupied cells and normalised entropy of the
  inlier distribution on an ``n x n`` grid.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .geometry import normalized_dlt, refine_homography_lm, reprojection_errors
from .errors import GeometryError
from .wms import ReferenceMap

log = logging.getLogger(__name__)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean square Euclidean distance between (N, 2) point sets."""
    actual = np.asarray(actual, dtype=np.float64).reshape(-1, 2)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1, 2)
    if len(actual) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum((actual - predicted) ** 2, axis=1))))


def zncc(reference: np.ndarray, source: np.ndarray, mask: Optional[np.ndarray] = None
         ) -> float:
    """Zero-normalised cross-correlation in [-1, 1] over ``mask`` (and finite pixels)."""
    r = np.asarray(reference, dtype=np.float64)
    s = np.asarray(source, dtype=np.float64)
    m = np.isfinite(r) & np.isfinite(s)
    if mask is not None:
        m &= mask
    if m.sum() < 2:
        return float("nan")
    rv = r[m] - r[m].mean()
    sv = s[m] - s[m].mean()
    denom = np.sqrt((rv ** 2).sum() * (sv ** 2).sum())
    return float((rv * sv).sum() / denom) if denom > 0 else float("nan")


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    img = np.nan_to_num(image.astype(np.float32), nan=0.0)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def uniformity(points: np.ndarray, width: int, height: int, grid: int = 8,
               footprint: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Grid coverage and normalised Shannon entropy of a point distribution.

    With ``footprint`` (a boolean mask of the image area that can hold
    matches), only grid cells at least half inside it count, so a source
    that covers part of the reference frame is not penalised for the rest.
    """
    cells = np.ones(grid * grid, dtype=bool)
    if footprint is not None:
        small = cv2.resize(footprint.astype(np.float32), (grid, grid),
                           interpolation=cv2.INTER_AREA)
        cells = small.ravel() >= 0.5
    n_cells = int(cells.sum())
    if len(points) == 0 or n_cells == 0:
        return {"coverage": 0.0, "entropy": 0.0}
    gx = np.clip((points[:, 0] / width * grid).astype(int), 0, grid - 1)
    gy = np.clip((points[:, 1] / height * grid).astype(int), 0, grid - 1)
    counts = np.bincount(gy * grid + gx, minlength=grid * grid).astype(np.float64)[cells]
    if counts.sum() == 0:
        return {"coverage": 0.0, "entropy": 0.0}
    p = counts[counts > 0] / counts.sum()
    entropy = float(-(p * np.log(p)).sum() / np.log(n_cells)) if n_cells > 1 else 1.0
    return {"coverage": float((counts > 0).mean()), "entropy": entropy}


def checkpoint_rmse(src: np.ndarray, dst: np.ndarray, fraction: float, seed: int
                    ) -> Tuple[float, int]:
    """Hold out ``fraction`` of the inliers, fit on the rest, RMSE on the held-out set."""
    n = len(src)
    n_test = int(round(n * fraction))
    if n_test < 3 or n - n_test < 8:
        return float("nan"), 0
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    test, train = perm[:n_test], perm[n_test:]
    try:
        H = refine_homography_lm(normalized_dlt(src[train], dst[train]), src[train], dst[train])
    except GeometryError:
        return float("nan"), 0
    err = reprojection_errors(H, src[test], dst[test])
    return float(np.sqrt(np.mean(err ** 2))), n_test


def write_geotiff(path: Path, image: np.ndarray, reference: ReferenceMap,
                  tags: Dict[str, str], compress: str = "deflate") -> None:
    """Write a float32 single-band GeoTIFF on the reference grid (NaN = no data)."""
    import rasterio

    h, w = image.shape
    profile: Dict[str, Any] = {
        "driver": "GTiff", "height": h, "width": w, "count": 1, "dtype": "float32",
        "crs": reference.crs, "transform": reference.transform, "nodata": np.nan,
        "compress": compress, "predictor": 3 if compress in ("deflate", "lzw") else 1,
    }
    if h >= 256 and w >= 256:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(image.astype(np.float32, copy=False), 1)
        ds.update_tags(**tags)
    log.info("wrote %s", path)


def write_report(path: Path, report: Dict[str, Any]) -> None:
    def default(o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"not JSON serialisable: {type(o)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=default, allow_nan=True))
