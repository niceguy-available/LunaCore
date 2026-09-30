"""Phase 1a — Chandrayaan-2 PDS4 product ingestion.

The label is parsed namespace-agnostically (Chandrayaan-2 products mix the
core ``pds:`` namespace with the ISRO ``isda:`` discipline dictionary and, for
map-projected products, ``cart:``). The binary array is memory-mapped straight
from the ``.dat``/``.img``/``.qub`` file described by the label, so an OHRC
strip of several gigabytes costs no RAM until it is streamed through
:meth:`Pds4Product.read_rows`. Labels whose array layout this reader does not
understand are handed to ``pds4_tools`` as a fallback.
"""

from __future__ import annotations

import logging
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .errors import LabelParseError
from .geo import BoundingBox, normalize_longitude

log = logging.getLogger(__name__)

# PDS4 Element_Array data_type -> numpy dtype string.
_PDS4_DTYPES: Dict[str, str] = {
    "SignedByte": "i1", "UnsignedByte": "u1",
    "SignedMSB2": ">i2", "UnsignedMSB2": ">u2", "SignedLSB2": "<i2", "UnsignedLSB2": "<u2",
    "SignedMSB4": ">i4", "UnsignedMSB4": ">u4", "SignedLSB4": "<i4", "UnsignedLSB4": "<u4",
    "SignedMSB8": ">i8", "UnsignedMSB8": ">u8", "SignedLSB8": "<i8", "UnsignedLSB8": "<u8",
    "IEEE754MSBSingle": ">f4", "IEEE754LSBSingle": "<f4",
    "IEEE754MSBDouble": ">f8", "IEEE754LSBDouble": "<f8",
}

_ARRAY_TAGS = ("Array_2D_Image", "Array_2D", "Array_3D_Spectrum", "Array_3D_Image", "Array_3D")

# Local tag names that carry a ground sample distance, in priority order.
_GSD_TAGS = (
    "pixel_resolution_x", "pixel_resolution_y", "ground_sample_distance",
    "pixel_resolution", "spatial_resolution", "sample_resolution", "line_resolution",
    "pixel_scale",
)

_LENGTH_UNITS_M = {"m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001}

_CORNER_RE = re.compile(r"^(upper|lower)_(left|right)_(latitude|longitude)$")

_SPECIAL_CONSTANTS = (
    "missing_constant", "invalid_constant", "saturated_constant",
    "high_instrument_saturation", "high_representation_saturation",
    "low_instrument_saturation", "low_representation_saturation", "unknown_constant",
)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(node: ET.Element, name: str) -> List[ET.Element]:
    return [c for c in node if _local(c.tag) == name]


def _child_text(node: ET.Element, name: str) -> Optional[str]:
    for c in node:
        if _local(c.tag) == name and c.text is not None:
            return c.text.strip()
    return None


def _iter_local(root: ET.Element, name: str):
    for el in root.iter():
        if _local(el.tag) == name:
            yield el


def _first_text(root: ET.Element, name: str) -> Optional[str]:
    for el in _iter_local(root, name):
        if el.text and el.text.strip():
            return el.text.strip()
    return None


def _to_float(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _length_to_m(value: float, unit: Optional[str]) -> float:
    """Convert '0.25' with unit 'm/pixel', 'km', 'm/pix' ... to metres."""
    if not unit:
        return value
    base = unit.strip().lower().split("/")[0].strip()
    return value * _LENGTH_UNITS_M.get(base, 1.0)


@dataclass
class ArrayLayout:
    """Byte layout of the image array inside the data file."""

    file_path: Path
    offset: int
    shape: Tuple[int, ...]          # in storage order, slowest axis first
    axis_names: Tuple[str, ...]
    dtype: np.dtype
    scaling_factor: float = 1.0
    value_offset: float = 0.0
    nodata: Tuple[float, ...] = ()


@dataclass
class Pds4Product:
    """A loaded Chandrayaan-2 product: lazily-mapped pixels plus geometry."""

    label_path: Path
    data: np.ndarray                          # (lines, samples) or (bands, lines, samples)
    layout: Optional[ArrayLayout]
    gsd_m: Optional[float]
    bbox: Optional[BoundingBox]
    corners: Optional[Dict[str, Tuple[float, float]]]   # name -> (lon, lat)
    instrument: Optional[str] = None
    metadata: Dict[str, float] = field(default_factory=dict)

    @property
    def is_multiband(self) -> bool:
        return self.data.ndim == 3

    @property
    def n_bands(self) -> int:
        return self.data.shape[0] if self.is_multiband else 1

    @property
    def shape_2d(self) -> Tuple[int, int]:
        return tuple(self.data.shape[-2:])  # type: ignore[return-value]

    def read_rows(self, r0: int, r1: int) -> np.ndarray:
        """Return rows [r0, r1) as float32 with scaling applied and nodata -> NaN.

        The result is ``(r1-r0, samples)`` for 2D products and
        ``(bands, r1-r0, samples)`` for cubes. Only the requested strip is paged
        in from disk.
        """
        out = np.array(self.data[..., r0:r1, :], dtype=np.float32)   # owned, writable copy
        if self.layout is not None:
            if self.layout.nodata:
                # Compare in float32: exact for every integer PDS4 type up to
                # 24 bits, and immune to sentinels unrepresentable in the dtype.
                out[np.isin(out, np.asarray(self.layout.nodata, dtype=np.float32))] = np.nan
            if self.layout.scaling_factor != 1.0 or self.layout.value_offset != 0.0:
                out *= np.float32(self.layout.scaling_factor)
                out += np.float32(self.layout.value_offset)
        out[~np.isfinite(out)] = np.nan
        return out

    def corner_pixels(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (pixel_xy, lonlat) 4x2 arrays for the four image corners, if known."""
        if not self.corners:
            return None
        h, w = self.shape_2d
        pix = {
            "upper_left": (0.0, 0.0), "upper_right": (w - 1.0, 0.0),
            "lower_right": (w - 1.0, h - 1.0), "lower_left": (0.0, h - 1.0),
        }
        names = [n for n in ("upper_left", "upper_right", "lower_right", "lower_left")
                 if n in self.corners]
        if len(names) < 4:
            return None
        return (np.array([pix[n] for n in names], dtype=np.float64),
                np.array([self.corners[n] for n in names], dtype=np.float64))


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def _parse_array_layout(root: ET.Element, label_path: Path) -> ArrayLayout:
    file_area = next(_iter_local(root, "File_Area_Observational"), None)
    if file_area is None:
        raise LabelParseError("label has no File_Area_Observational")

    file_name = None
    for f in _children(file_area, "File"):
        file_name = _child_text(f, "file_name")
    if not file_name:
        raise LabelParseError("File_Area_Observational/File/file_name missing")
    file_path = label_path.parent / file_name
    if not file_path.exists():
        # Archives sometimes differ only in case (e.g. .DAT vs .dat).
        matches = [p for p in label_path.parent.iterdir() if p.name.lower() == file_name.lower()]
        if not matches:
            raise LabelParseError(f"data file {file_path} not found")
        file_path = matches[0]

    array = next((c for c in file_area if _local(c.tag) in _ARRAY_TAGS), None)
    if array is None:
        raise LabelParseError(f"no supported array ({', '.join(_ARRAY_TAGS)}) in label")

    offset_text = _child_text(array, "offset")
    offset = int(float(offset_text)) if offset_text else 0

    order = _child_text(array, "axis_index_order") or "Last Index Fastest"
    if order != "Last Index Fastest":
        raise LabelParseError(f"unsupported axis_index_order {order!r}")

    axes = []
    for ax in _children(array, "Axis_Array"):
        name = _child_text(ax, "axis_name") or ""
        elements = _child_text(ax, "elements")
        seq = _child_text(ax, "sequence_number")
        if elements is None or seq is None:
            raise LabelParseError("Axis_Array missing elements/sequence_number")
        axes.append((int(seq), name, int(elements)))
    if len(axes) not in (2, 3):
        raise LabelParseError(f"expected a 2D or 3D array, found {len(axes)} axes")
    axes.sort()

    element = next(iter(_children(array, "Element_Array")), None)
    if element is None:
        raise LabelParseError("Element_Array missing")
    data_type = _child_text(element, "data_type")
    if data_type not in _PDS4_DTYPES:
        raise LabelParseError(f"unsupported data_type {data_type!r}")
    scale = _to_float(_child_text(element, "scaling_factor")) or 1.0
    value_offset = _to_float(_child_text(element, "value_offset")) or 0.0

    nodata: List[float] = []
    for sc in _iter_local(array, "Special_Constants"):
        for name in _SPECIAL_CONSTANTS:
            text = _child_text(sc, name)
            if text is None:
                continue
            try:
                nodata.append(float(int(text, 16)) if text.lower().startswith("0x") else float(text))
            except ValueError:
                log.warning("ignoring unparsable special constant %s=%r", name, text)

    return ArrayLayout(
        file_path=file_path, offset=offset,
        shape=tuple(a[2] for a in axes), axis_names=tuple(a[1] for a in axes),
        dtype=np.dtype(_PDS4_DTYPES[data_type]),
        scaling_factor=scale, value_offset=value_offset, nodata=tuple(nodata),
    )


def _parse_gsd(root: ET.Element) -> Optional[float]:
    for name in _GSD_TAGS:
        values = []
        for el in _iter_local(root, name):
            v = _to_float(el.text)
            if v is not None and v > 0:
                values.append(_length_to_m(v, el.get("unit")))
        if values:
            # Along/across-track resolutions differ slightly: geometric mean.
            return float(np.exp(np.mean(np.log(values))))
    return None


def _parse_corners(root: ET.Element) -> Optional[Dict[str, Tuple[float, float]]]:
    found: Dict[str, Dict[str, float]] = {}
    for el in root.iter():
        m = _CORNER_RE.match(_local(el.tag))
        v = _to_float(el.text)
        if m and v is not None:
            found.setdefault(f"{m.group(1)}_{m.group(2)}", {})[m.group(3)] = v
    corners = {k: (normalize_longitude(v["longitude"]), v["latitude"])
               for k, v in found.items() if {"latitude", "longitude"} <= v.keys()}
    return corners if len(corners) == 4 else None


def _parse_bbox(root: ET.Element,
                corners: Optional[Dict[str, Tuple[float, float]]]) -> Optional[BoundingBox]:
    west = _to_float(_first_text(root, "west_bounding_coordinate"))
    east = _to_float(_first_text(root, "east_bounding_coordinate"))
    north = _to_float(_first_text(root, "north_bounding_coordinate"))
    south = _to_float(_first_text(root, "south_bounding_coordinate"))
    if None not in (west, east, north, south):
        return BoundingBox.from_points([west, east], [south, north])
    if corners:
        lons, lats = zip(*corners.values())
        return BoundingBox.from_points(lons, lats)
    return None


def _parse_instrument(root: ET.Element) -> Optional[str]:
    for comp in _iter_local(root, "Observing_System_Component"):
        if (_child_text(comp, "type") or "").lower() == "instrument":
            return _child_text(comp, "name")
    return None


def _parse_illumination(root: ET.Element) -> Dict[str, float]:
    meta: Dict[str, float] = {}
    for el in root.iter():
        name = _local(el.tag)
        if ("sun" in name or "solar" in name) and len(el) == 0:
            v = _to_float(el.text)
            if v is not None:
                meta[name] = v
    return meta


def _memmap(layout: ArrayLayout) -> np.ndarray:
    expected = layout.offset + int(np.prod(layout.shape)) * layout.dtype.itemsize
    actual = layout.file_path.stat().st_size
    if actual < expected:
        raise LabelParseError(
            f"{layout.file_path.name} is {actual} bytes, label describes {expected}")
    arr = np.memmap(layout.file_path, dtype=layout.dtype, mode="r",
                    offset=layout.offset, shape=layout.shape)
    if arr.ndim == 3:
        # Bring the band axis first (BSQ view) without copying: IIRS is often BIL.
        names = [n.lower() for n in layout.axis_names]
        band_axis = next((i for i, n in enumerate(names) if "band" in n), 0)
        arr = np.moveaxis(arr, band_axis, 0)
    return arr


def _load_with_pds4_tools(label_path: Path) -> np.ndarray:
    try:
        import pds4_tools
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise LabelParseError(
            "label layout not supported natively and pds4_tools is not installed") from exc
    try:
        struct_list = pds4_tools.read(str(label_path), lazy_load=True, quiet=True)
        for struct in struct_list:
            if struct.is_array() and struct.data.ndim in (2, 3):
                return np.asarray(struct.data)
    except Exception as exc:  # pds4_tools raises a wide range of error types
        raise LabelParseError(f"pds4_tools could not read {label_path.name}: {exc}") from exc
    raise LabelParseError("pds4_tools found no 2D/3D array in the product")


def load_pds4(label_path: str | Path,
              gsd_override_m: Optional[float] = None) -> Pds4Product:
    """Load a PDS4 label and memory-map its image array.

    Args:
        label_path: Path to the ``.xml`` label.
        gsd_override_m: Ground sample distance to use when the label has none.

    Raises:
        LabelParseError: if the label is malformed or lacks the pixel array.
    """
    label_path = Path(label_path)
    if not label_path.is_file():
        raise LabelParseError(f"label {label_path} does not exist")
    try:
        # Labels are archive-controlled text; ElementTree does not resolve
        # external entities, so parsing is safe against XXE.
        root = ET.parse(label_path).getroot()
    except ET.ParseError as exc:
        raise LabelParseError(f"malformed XML in {label_path}: {exc}") from exc

    layout: Optional[ArrayLayout]
    try:
        layout = _parse_array_layout(root, label_path)
    except LabelParseError as exc:
        log.info("native PDS4 reader declined (%s); falling back to pds4_tools", exc)
        layout = None
        data = _load_with_pds4_tools(label_path)
    else:
        data = _memmap(layout)   # a size mismatch is a data error, not a layout one

    corners = _parse_corners(root)
    gsd = gsd_override_m if gsd_override_m is not None else _parse_gsd(root)
    product = Pds4Product(
        label_path=label_path, data=data, layout=layout, gsd_m=gsd,
        bbox=_parse_bbox(root, corners), corners=corners,
        instrument=_parse_instrument(root), metadata=_parse_illumination(root),
    )
    log.info("loaded %s: shape=%s gsd=%s m bbox=%s instrument=%s",
             label_path.name, data.shape, gsd, product.bbox, product.instrument)
    return product


def infer_gsd_from_footprint(product: Pds4Product) -> Optional[float]:
    """Estimate GSD (m/px) from the geographic footprint when the label has none."""
    if product.bbox is None:
        return None
    h, w = product.shape_2d
    corners = product.corner_pixels()
    if corners is not None:
        from .geo import METERS_PER_DEGREE
        pix, ll = corners
        lat0 = math.radians(float(ll[:, 1].mean()))
        xy_m = np.column_stack([ll[:, 0] * math.cos(lat0), ll[:, 1]]) * METERS_PER_DEGREE
        across = np.linalg.norm(xy_m[1] - xy_m[0]) / max(w - 1, 1)
        along = np.linalg.norm(xy_m[3] - xy_m[0]) / max(h - 1, 1)
        return float(math.sqrt(across * along))
    return float(math.sqrt(product.bbox.width_m * product.bbox.height_m / (w * h)))


def lonlat_to_pixels(lonlat: np.ndarray, transform) -> np.ndarray:
    """Map Nx2 (lon, lat) to Nx2 (col, row) pixel-centre coordinates of a raster."""
    inv = ~transform
    cols, rows = inv * (lonlat[:, 0], lonlat[:, 1])
    # rasterio's affine maps pixel *corners*; shift to OpenCV pixel centres.
    return np.column_stack([np.asarray(cols) - 0.5, np.asarray(rows) - 0.5])


__all__: Sequence[str] = (
    "ArrayLayout", "Pds4Product", "load_pds4", "infer_gsd_from_footprint", "lonlat_to_pixels",
)
