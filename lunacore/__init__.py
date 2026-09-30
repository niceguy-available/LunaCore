"""LunaCore — multi-modal, sun-angle and scale invariant lunar image registration.

Six phases, one module each:

1. :mod:`lunacore.ingestion` / :mod:`lunacore.wms` — PDS4 loading, WMS base map
2. :mod:`lunacore.harmonization` — PCA, GSD scaling, CLAHE, geo-prior
3. :mod:`lunacore.matching` — LoFTR (kornia) + quad-tree distribution
4. :mod:`lunacore.geometry` — MAGSAC++, SVD/DLT, LM, ECC
5. :mod:`lunacore.geometry` — bicubic projective warp
6. :mod:`lunacore.evaluation` — RMSE, ZNCC, uniformity, GeoTIFF output

Heavy dependencies (torch/kornia, rasterio, requests) are imported lazily.
"""

from .config import PipelineConfig
from .errors import (GeometryError, HarmonizationError, LabelParseError, LunaCoreError,
                     MatchingError, WMSError)

__version__ = "0.1.0"


def register(*args, **kwargs):
    """Proxy for :func:`lunacore.pipeline.register` (keeps package import light)."""
    from .pipeline import register as _register

    return _register(*args, **kwargs)


__all__ = [
    "PipelineConfig", "register", "LunaCoreError", "LabelParseError", "WMSError",
    "HarmonizationError", "MatchingError", "GeometryError", "__version__",
]
