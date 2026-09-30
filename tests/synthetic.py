"""Synthetic lunar scenes with exact ground truth for pipeline tests."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from lunacore.geo import METERS_PER_DEGREE

LABEL_NS = ('xmlns="http://pds.nasa.gov/pds4/pds/v1" '
            'xmlns:isda="http://pds.nasa.gov/pds4/isda/v1"')


def crater_heightmap(size: int, n_craters: int = 220, seed: int = 0) -> np.ndarray:
    """Height field with bowl-shaped, rimmed craters on multi-octave noise."""
    rng = np.random.default_rng(seed)
    z = np.zeros((size, size), dtype=np.float32)
    for sigma, amp in ((64, 6.0), (16, 2.0), (4, 0.6)):
        z += amp * cv2.GaussianBlur(rng.standard_normal((size, size)).astype(np.float32),
                                    (0, 0), sigma) * sigma
    radii = np.clip(rng.pareto(1.6, n_craters) * 6 + 5, 5, size / 6)
    for r in radii:
        cx, cy = rng.uniform(0, size, 2)
        x0, x1 = int(max(cx - 2 * r, 0)), int(min(cx + 2 * r + 1, size))
        y0, y1 = int(max(cy - 2 * r, 0)), int(min(cy + 2 * r + 1, size))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.hypot(xx - cx, yy - cy) / r
        depth = 0.2 * r
        bowl = np.where(d < 1, depth * (d ** 2 - 1), 0.0)
        rim = 0.25 * depth * np.exp(-((d - 1) / 0.25) ** 2)
        z[y0:y1, x0:x1] += (bowl + rim).astype(np.float32)
    return z


def shade(z: np.ndarray, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Lambertian hill-shade in [0, 1] with the sun at the given azimuth/elevation."""
    gy, gx = np.gradient(z.astype(np.float64))
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    sun = np.array([math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)])
    n = np.dstack([-gx, -gy, np.ones_like(gx)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return np.clip(n @ sun, 0.0, 1.0).astype(np.float32)


def write_reference_geotiff(path: Path, image: np.ndarray, bounds: Tuple[float, float, float, float]):
    import rasterio
    from rasterio.transform import from_bounds
    from lunacore.geo import moon_crs

    h, w = image.shape
    transform = from_bounds(*bounds, w, h)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1,
                       dtype="float32", crs=moon_crs(), transform=transform) as ds:
        ds.write(image.astype(np.float32), 1)
    return transform


def write_pds4(dir_: Path, data: np.ndarray, *, name: str = "src",
               data_type: str = "UnsignedMSB2", gsd: Optional[Tuple[float, str]] = (1.0, "m/pixel"),
               corners: Optional[Dict[str, Tuple[float, float]]] = None,
               bbox: Optional[Tuple[float, float, float, float]] = None,
               axis_names: Sequence[str] = ("Line", "Sample"),
               offset: int = 0, scaling: Optional[Tuple[float, float]] = None,
               special: Optional[Dict[str, str]] = None,
               instrument: str = "Orbiter High Resolution Camera") -> Path:
    """Write a minimal but schema-shaped PDS4 label + raw array file."""
    from lunacore.ingestion import _PDS4_DTYPES

    dtype = np.dtype(_PDS4_DTYPES[data_type])
    dat = dir_ / f"{name}.dat"
    with open(dat, "wb") as fh:
        fh.write(b"\0" * offset)
        fh.write(np.ascontiguousarray(data, dtype=dtype).tobytes())

    axes = "".join(
        f"<Axis_Array><axis_name>{n}</axis_name><elements>{e}</elements>"
        f"<sequence_number>{i + 1}</sequence_number></Axis_Array>"
        for i, (n, e) in enumerate(zip(axis_names, data.shape)))
    scale_xml = ""
    if scaling:
        scale_xml = (f"<scaling_factor>{scaling[0]}</scaling_factor>"
                     f"<value_offset>{scaling[1]}</value_offset>")
    special_xml = ""
    if special:
        special_xml = "<Special_Constants>" + "".join(
            f"<{k}>{v}</{k}>" for k, v in special.items()) + "</Special_Constants>"
    geo = ""
    if gsd:
        geo += f'<isda:pixel_resolution unit="{gsd[1]}">{gsd[0]}</isda:pixel_resolution>'
    if corners:
        for cname, (lon, lat) in corners.items():
            geo += (f"<isda:{cname}_latitude>{lat:.10f}</isda:{cname}_latitude>"
                    f"<isda:{cname}_longitude>{lon:.10f}</isda:{cname}_longitude>")
    if bbox:
        w, s, e, n = bbox
        geo += ("<cart:Bounding_Coordinates xmlns:cart=\"http://pds.nasa.gov/pds4/cart/v1\">"
                f"<cart:west_bounding_coordinate>{w}</cart:west_bounding_coordinate>"
                f"<cart:east_bounding_coordinate>{e}</cart:east_bounding_coordinate>"
                f"<cart:north_bounding_coordinate>{n}</cart:north_bounding_coordinate>"
                f"<cart:south_bounding_coordinate>{s}</cart:south_bounding_coordinate>"
                "</cart:Bounding_Coordinates>")
    array_tag = "Array_2D_Image" if data.ndim == 2 else "Array_3D_Spectrum"
    label = f"""<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational {LABEL_NS}>
  <Observation_Area>
    <Observing_System><Observing_System_Component>
      <name>{instrument}</name><type>Instrument</type>
    </Observing_System_Component></Observing_System>
    <Discipline_Area><isda:Geometry_Parameters>{geo}
      <isda:solar_incidence_angle>70.0</isda:solar_incidence_angle>
    </isda:Geometry_Parameters></Discipline_Area>
  </Observation_Area>
  <File_Area_Observational>
    <File><file_name>{dat.name}</file_name></File>
    <{array_tag}>
      <offset unit="byte">{offset}</offset>
      <axes>{data.ndim}</axes>
      <axis_index_order>Last Index Fastest</axis_index_order>
      <Element_Array><data_type>{data_type}</data_type>{scale_xml}</Element_Array>
      {axes}{special_xml}
    </{array_tag}>
  </File_Area_Observational>
</Product_Observational>
"""
    label_path = dir_ / f"{name}.xml"
    label_path.write_text(label)
    return label_path


class Scene:
    """A world at 1 m/px, a 4 m/px reference GeoTIFF and a rotated 1 m/px source."""

    WORLD = 1200
    REF_FACTOR = 4
    SRC = 640

    def __init__(self, tmp: Path, *, ref_sun=(135.0, 25.0), src_sun=(150.0, 30.0),
                 rotation_deg: float = 6.0, pointing_error_ref_px: float = 3.0,
                 with_corners: bool = True, seed: int = 3):
        z = crater_heightmap(self.WORLD, seed=seed)
        world_ref = shade(z, *ref_sun)
        world_src = shade(z, *src_sun)

        # Reference: area-reduce by 4 -> 300 px at 4 m/px.
        k = self.REF_FACTOR
        ref = cv2.resize(world_ref, (self.WORLD // k,) * 2, interpolation=cv2.INTER_AREA)
        deg = (self.WORLD * 1.0) / METERS_PER_DEGREE
        self.bounds = (20.0, 5.0, 20.0 + deg, 5.0 + deg)
        self.ref_path = tmp / "reference.tif"
        self.transform = write_reference_geotiff(self.ref_path, ref * 1000.0, self.bounds)

        # Source: world(G x) where G = source px -> world px.
        c = self.SRC / 2
        a = math.radians(rotation_deg)
        R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        T0 = np.array([[1, 0, -c], [0, 1, -c], [0, 0, 1]], dtype=np.float64)
        T1 = np.array([[1, 0, self.WORLD / 2 + 17.0], [0, 1, self.WORLD / 2 - 11.0], [0, 0, 1]])
        self.G = T1 @ R @ T0
        src = cv2.warpPerspective(world_src, self.G, (self.SRC, self.SRC),
                                  flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
        self.src = np.clip(src * 4000 + 100, 0, 65535).astype(np.uint16)

        # Ground truth: source full px -> reference px.
        from lunacore.harmonization import scale_matrix
        self.H_true = scale_matrix(1.0 / k, 1.0 / k) @ self.G

        corners = None
        if with_corners:
            from lunacore.geometry import apply_homography
            s = self.SRC - 1
            pix = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float64)
            ref_px = apply_homography(self.H_true, pix) + pointing_error_ref_px
            lon, lat = self.transform * (ref_px[:, 0] + 0.5, ref_px[:, 1] + 0.5)
            corners = {n: (float(lo), float(la)) for n, lo, la in zip(
                ("upper_left", "upper_right", "lower_right", "lower_left"), lon, lat)}
        self.label = write_pds4(tmp, self.src, corners=corners, gsd=(1.0, "m/pixel"))

    def error_ref_px(self, H: np.ndarray) -> float:
        """Max point error of H vs ground truth over a grid of source pixels."""
        from lunacore.geometry import apply_homography

        g = np.linspace(0, self.SRC - 1, 9)
        pts = np.array([(x, y) for x in g for y in g], dtype=np.float64)
        return float(np.abs(apply_homography(H, pts) - apply_homography(self.H_true, pts)).max())
