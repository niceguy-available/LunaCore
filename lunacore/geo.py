"""Lunar geodesy helpers shared by the ingestion, WMS and output phases.

All geographic coordinates are planetocentric degrees on the IAU 2015 lunar
sphere (R = 1737.4 km), longitudes normalised to [-180, 180).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

MOON_RADIUS_M: float = 1_737_400.0
METERS_PER_DEGREE: float = math.pi * MOON_RADIUS_M / 180.0  # ~30 323.6 m

# Moon (2015) sphere, planetocentric lon/lat. rasterio/PROJ >= 8.2 understands
# the authority code; the WKT is the fallback for older PROJ builds.
MOON_CRS_CODE: str = "IAU_2015:30100"
MOON_CRS_WKT: str = (
    'GEOGCS["Moon (2015) - Sphere / Ocentric",'
    'DATUM["Moon (2015) - Sphere",SPHEROID["Moon (2015) - Sphere",1737400,0]],'
    'PRIMEM["Reference Meridian",0],'
    'UNIT["degree",0.0174532925199433]]'
)


def normalize_longitude(lon: float) -> float:
    """Map a longitude in any convention (e.g. 0..360 east) to [-180, 180)."""
    return ((float(lon) + 180.0) % 360.0) - 180.0


@dataclass(frozen=True)
class BoundingBox:
    """Geographic bounding box in planetocentric degrees."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def __post_init__(self) -> None:
        if not (-90.0 <= self.min_lat < self.max_lat <= 90.0):
            raise ValueError(f"invalid latitude range [{self.min_lat}, {self.max_lat}]")
        if not self.min_lon < self.max_lon:
            raise ValueError(
                f"invalid longitude range [{self.min_lon}, {self.max_lon}]; "
                "footprints crossing the 180° meridian are not supported"
            )

    @classmethod
    def from_points(cls, lons: Iterable[float], lats: Iterable[float]) -> "BoundingBox":
        lon_arr = np.array([normalize_longitude(v) for v in lons], dtype=np.float64)
        lat_arr = np.asarray(list(lats), dtype=np.float64)
        if lon_arr.size == 0 or lon_arr.size != lat_arr.size:
            raise ValueError("need matching, non-empty longitude/latitude lists")
        return cls(float(lon_arr.min()), float(lat_arr.min()),
                   float(lon_arr.max()), float(lat_arr.max()))

    @property
    def center(self) -> Tuple[float, float]:
        return (0.5 * (self.min_lon + self.max_lon), 0.5 * (self.min_lat + self.max_lat))

    @property
    def width_m(self) -> float:
        """East-west ground extent at the box's central latitude."""
        return (self.max_lon - self.min_lon) * METERS_PER_DEGREE * math.cos(
            math.radians(self.center[1]))

    @property
    def height_m(self) -> float:
        return (self.max_lat - self.min_lat) * METERS_PER_DEGREE

    def padded(self, fraction: float) -> "BoundingBox":
        """Grow the box by ``fraction`` of its size on every side (clamped at the poles)."""
        dlon = (self.max_lon - self.min_lon) * fraction
        dlat = (self.max_lat - self.min_lat) * fraction
        return BoundingBox(
            max(-180.0, self.min_lon - dlon), max(-90.0, self.min_lat - dlat),
            min(180.0, self.max_lon + dlon), min(90.0, self.max_lat + dlat),
        )

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)


def moon_crs():
    """Return a rasterio CRS for the lunar sphere (lazy import keeps rasterio optional)."""
    from rasterio.crs import CRS
    from rasterio.errors import CRSError

    try:
        return CRS.from_string(MOON_CRS_CODE)
    except CRSError:
        return CRS.from_wkt(MOON_CRS_WKT)
