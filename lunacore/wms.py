"""Phase 1b — automatic reference base-map retrieval over OGC WMS.

Given the Chandrayaan-2 footprint, :func:`fetch_reference` sizes a GetMap
request so the returned LRO mosaic has (approximately) square ground pixels at
the requested resolution, downloads it with retries and streaming, validates
that the server returned an image rather than a ``ServiceExceptionReport``,
and attaches exact georeferencing derived from the request itself.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .config import WMSConfig
from .errors import WMSError
from .geo import METERS_PER_DEGREE, BoundingBox, moon_crs

log = logging.getLogger(__name__)

_IMAGE_MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+", b"\x89PNG", b"\xff\xd8\xff")
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


@dataclass
class ReferenceMap:
    """A single-channel reference raster with its georeferencing."""

    image: np.ndarray            # float32 (H, W), NaN where no data
    transform: "object"          # affine.Affine, pixel corner -> map coordinates
    crs: "object"                # rasterio.crs.CRS
    gsd_m: float                 # meridional ground sample distance
    path: Optional[Path] = None

    @property
    def shape(self) -> Tuple[int, int]:
        return self.image.shape  # type: ignore[return-value]


def plan_reference_grid(bbox: BoundingBox, source_gsd_m: Optional[float],
                        cfg: WMSConfig) -> Tuple[int, int, float]:
    """Choose the GetMap WIDTH/HEIGHT for ``bbox``.

    The target resolution is ``cfg.target_gsd_m`` or the source GSD, but never
    finer than the base map's native ``min_reference_gsd_m`` (up-sampling a
    100 m/px mosaic to 0.25 m/px only wastes bandwidth). Width uses the
    cos(latitude)-shrunk east-west extent so pixels are square on the ground.

    Returns:
        (width_px, height_px, achieved_gsd_m)
    """
    target = cfg.target_gsd_m or max(source_gsd_m or cfg.min_reference_gsd_m,
                                     cfg.min_reference_gsd_m)
    if target <= 0:
        raise WMSError(f"target GSD must be positive, got {target}")
    width = max(1, math.ceil(bbox.width_m / target))
    height = max(1, math.ceil(bbox.height_m / target))
    longest = max(width, height)
    if longest > cfg.max_size_px:
        shrink = cfg.max_size_px / longest
        width = max(1, int(width * shrink))
        height = max(1, int(height * shrink))
        log.warning("reference request capped at %d px; GSD coarsened", cfg.max_size_px)
    return width, height, bbox.height_m / height


def build_getmap_params(bbox: BoundingBox, width: int, height: int,
                        cfg: WMSConfig) -> Dict[str, str]:
    """Build OGC WMS GetMap query parameters.

    WMS 1.3.0 mandates latitude-first axis order for EPSG:4326 and renames
    ``SRS`` to ``CRS``; 1.1.1 is always longitude-first.
    """
    if cfg.version.startswith("1.3") and cfg.crs.upper() == "EPSG:4326":
        coords = (bbox.min_lat, bbox.min_lon, bbox.max_lat, bbox.max_lon)
    else:
        coords = bbox.as_tuple()
    params = {
        "SERVICE": "WMS",
        "VERSION": cfg.version,
        "REQUEST": "GetMap",
        "LAYERS": cfg.layer,
        "STYLES": cfg.styles,
        "BBOX": ",".join(f"{c:.10f}" for c in coords),
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "FORMAT": cfg.image_format,
        "CRS" if cfg.version.startswith("1.3") else "SRS": cfg.crs,
    }
    params.update(cfg.extra_params)
    return params


def _download(url: str, params: Dict[str, str], dest: Path, cfg: WMSConfig,
              session=None) -> None:
    """Stream a GetMap response to ``dest`` with exponential-backoff retries."""
    import requests

    sess = session or requests.Session()
    last_error: Optional[Exception] = None
    for attempt in range(cfg.retries + 1):
        if attempt:
            delay = cfg.backoff_s * (2 ** (attempt - 1))
            log.warning("WMS retry %d/%d in %.1fs (%s)", attempt, cfg.retries, delay, last_error)
            time.sleep(delay)
        try:
            with sess.get(url, params=params, stream=True, timeout=cfg.timeout_s) as resp:
                if resp.status_code in _RETRYABLE_STATUS:
                    last_error = WMSError(f"HTTP {resp.status_code}")
                    continue
                if resp.status_code != 200:
                    raise WMSError(f"WMS HTTP {resp.status_code}: {resp.text[:500]}")
                ctype = resp.headers.get("Content-Type", "").lower()
                if "xml" in ctype or ctype.startswith("text/"):
                    raise WMSError(f"WMS service exception: {resp.text[:1000]}")
                fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
                try:
                    with os.fdopen(fd, "wb") as fh:
                        for chunk in resp.iter_content(chunk_size=cfg.chunk_bytes):
                            fh.write(chunk)
                    os.replace(tmp, dest)
                finally:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                return
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = exc
    raise WMSError(f"WMS download failed after {cfg.retries + 1} attempts: {last_error}")


def _validate_image_file(path: Path) -> None:
    with open(path, "rb") as fh:
        head = fh.read(512)
    if not head:
        raise WMSError("WMS returned an empty body")
    if not any(head.startswith(m) for m in _IMAGE_MAGIC):
        if head.lstrip().startswith(b"<"):
            raise WMSError(f"WMS returned XML instead of an image: {head[:300]!r}")
        raise WMSError("WMS response is not a recognised image format")


def _read_single_channel(path: Path) -> Tuple[np.ndarray, "object", "object"]:
    """Read a raster as float32 luminance with NaN nodata; returns (img, transform, crs)."""
    import rasterio

    with rasterio.open(path) as ds:
        count = ds.count
        colour = [i for i in range(1, count + 1)
                  if ds.colorinterp[i - 1].name != "alpha"] or [1]
        alpha = [i for i in range(1, count + 1) if ds.colorinterp[i - 1].name == "alpha"]
        img = np.zeros((ds.height, ds.width), dtype=np.float32)
        for band in colour:          # accumulate band by band to bound memory
            img += ds.read(band, out_dtype=np.float32)
        img /= len(colour)
        invalid = np.zeros(img.shape, dtype=bool)
        if ds.nodata is not None:
            invalid |= ds.read(colour[0]) == ds.nodata
        if alpha:
            invalid |= ds.read(alpha[0]) == 0
        img[invalid] = np.nan
        return img, ds.transform, ds.crs


def fetch_reference(bbox: BoundingBox, source_gsd_m: Optional[float], cfg: WMSConfig,
                    out_dir: Path, session=None) -> ReferenceMap:
    """Download the LRO base map covering ``bbox`` (padded) and georeference it.

    Args:
        bbox: Source footprint in planetocentric degrees.
        source_gsd_m: Source GSD, used to pick the request resolution.
        cfg: WMS endpoint/layer settings.
        out_dir: Directory for the downloaded GeoTIFF.
        session: Optional ``requests.Session`` (injectable for testing/pooling).
    """
    from rasterio.transform import from_bounds

    padded = bbox.padded(cfg.bbox_padding)
    width, height, gsd = plan_reference_grid(padded, source_gsd_m, cfg)
    params = build_getmap_params(padded, width, height, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "reference_wms.tif"
    log.info("WMS GetMap %s layer=%s bbox=%s size=%dx%d (%.2f m/px)",
             cfg.url, cfg.layer, padded.as_tuple(), width, height, gsd)
    _download(cfg.url, params, dest, cfg, session=session)
    _validate_image_file(dest)

    img, _, _ = _read_single_channel(dest)
    if img.shape != (height, width):
        raise WMSError(f"WMS returned {img.shape[::-1]} px, requested {(width, height)}")
    if not np.isfinite(img).any() or np.nanstd(img) == 0:
        raise WMSError("WMS image is blank; check the layer name and footprint")
    # The request defines the georeferencing exactly; server-embedded tags
    # (often an Earth datum) are ignored.
    transform = from_bounds(*padded.as_tuple(), width, height)
    return ReferenceMap(image=img, transform=transform, crs=moon_crs(), gsd_m=gsd, path=dest)


def load_reference(path: str | Path) -> ReferenceMap:
    """Load a user-supplied reference GeoTIFF (bypasses the WMS fetch)."""
    path = Path(path)
    img, transform, crs = _read_single_channel(path)
    if crs is not None and crs.is_projected:
        gsd = abs(transform.e) * (crs.linear_units_factor[1] if crs.linear_units_factor else 1.0)
    else:
        gsd = abs(transform.e) * METERS_PER_DEGREE
    return ReferenceMap(image=img, transform=transform, crs=crs or moon_crs(),
                        gsd_m=float(gsd), path=path)
