"""Typed configuration for every phase of the registration pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class WMSConfig:
    """Phase 1 — where and how to fetch the reference base map."""

    url: str = "https://planetarymaps.usgs.gov/cgi-bin/mapserv"
    # USGS mapserv needs a mapfile; LROC's WMS ignores unknown params.
    extra_params: Dict[str, str] = field(
        default_factory=lambda: {"map": "/maps/earth/moon_simp_cyl.map"})
    layer: str = "LROC_WAC"
    version: str = "1.1.1"
    # Planetary WMS servers publish simple-cylindrical lunar lon/lat under the
    # EPSG:4326 code. Note that EPSG:32630 is a terrestrial UTM zone and must
    # not be used for lunar data.
    crs: str = "EPSG:4326"
    image_format: str = "image/tiff"
    styles: str = ""
    target_gsd_m: Optional[float] = None   # None -> use max(source GSD, min_reference_gsd_m)
    min_reference_gsd_m: float = 5.0
    max_size_px: int = 4096
    bbox_padding: float = 0.15             # grow footprint so pointing errors stay in frame
    timeout_s: float = 120.0
    retries: int = 4
    backoff_s: float = 2.0
    chunk_bytes: int = 1 << 20


@dataclass
class HarmonizationConfig:
    """Phase 2 — radiometric/geometric harmonisation."""

    clahe_clip_limit: float = 2.0
    clahe_grid: int = 8
    percentile_low: float = 0.5
    percentile_high: float = 99.5
    pca_standardize: bool = False
    strip_rows: int = 1024                 # rows per strip when streaming large arrays
    prior: str = "auto"                    # auto | corners | bbox | scale


@dataclass
class MatchingConfig:
    """Phase 3 — deep matcher and spatial distribution."""

    matcher: str = "loftr"                 # loftr | sift
    loftr_weights: str = "outdoor"         # 'outdoor' | 'indoor' | path to a .ckpt
    device: str = "auto"                   # auto | cpu | cuda | mps
    half_precision: bool = True            # fp16 autocast on CUDA
    max_side_px: int = 1024                # longest side fed to the network
    min_confidence: float = 0.2
    try_inverted_polarity: bool = True     # also match the negated source (inverted shadows)
    quadtree_max_depth: int = 6
    quadtree_capacity: int = 4
    max_matches: int = 2000
    mask_erosion_px: int = 8


@dataclass
class GeometryConfig:
    """Phases 4-5 — outlier rejection, homography, warp."""

    magsac_threshold_px: float = 3.0
    magsac_confidence: float = 0.9999
    magsac_max_iters: int = 20000
    min_inliers: int = 12
    # ECC maximises intensity correlation, which is biased when the two images
    # have different sun angles or modalities; enable it only for pairs with
    # similar illumination (it is still rejected automatically if it degrades).
    ecc_refine: bool = False
    ecc_max_iters: int = 200
    ecc_eps: float = 1e-7
    max_scale_change: float = 4.0          # sanity bound on the residual homography


@dataclass
class EvaluationConfig:
    """Phase 6 — accuracy metrics and output."""

    holdout_fraction: float = 0.2
    random_seed: int = 0
    uniformity_grid: int = 8
    geotiff_compress: str = "deflate"


@dataclass
class PipelineConfig:
    wms: WMSConfig = field(default_factory=WMSConfig)
    harmonization: HarmonizationConfig = field(default_factory=HarmonizationConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        sections = {
            "wms": WMSConfig, "harmonization": HarmonizationConfig,
            "matching": MatchingConfig, "geometry": GeometryConfig,
            "evaluation": EvaluationConfig,
        }
        unknown = set(data) - set(sections)
        if unknown:
            raise ValueError(f"unknown config sections: {sorted(unknown)}")
        return cls(**{name: klass(**data.get(name, {})) for name, klass in sections.items()})
