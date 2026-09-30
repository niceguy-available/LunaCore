import numpy as np
import pytest

from lunacore.errors import LabelParseError
from lunacore.ingestion import infer_gsd_from_footprint, load_pds4
from synthetic import write_pds4

CORNERS = {"upper_left": (10.0, 1.0), "upper_right": (10.1, 1.0),
           "lower_right": (10.1, 0.9), "lower_left": (10.0, 0.9)}


def test_2d_label_geometry_and_memmap(tmp_path):
    data = np.arange(12 * 7, dtype=np.uint16).reshape(12, 7)
    label = write_pds4(tmp_path, data, gsd=(0.25, "m/pixel"), corners=CORNERS, offset=64)
    p = load_pds4(label)
    assert isinstance(p.data, np.memmap)
    np.testing.assert_array_equal(np.asarray(p.data), data)
    assert p.gsd_m == pytest.approx(0.25)
    assert p.bbox.as_tuple() == pytest.approx((10.0, 0.9, 10.1, 1.0))
    assert p.instrument == "Orbiter High Resolution Camera"
    assert p.metadata["solar_incidence_angle"] == 70.0
    pix, ll = p.corner_pixels()
    assert pix[2].tolist() == [6.0, 11.0]


def test_units_scaling_and_special_constants(tmp_path):
    data = np.array([[1, 2], [0, 4]], dtype=np.int16)
    label = write_pds4(tmp_path, data, data_type="SignedLSB2", gsd=(0.005, "km/pixel"),
                       scaling=(0.5, 10.0), special={"missing_constant": "0"})
    p = load_pds4(label)
    assert p.gsd_m == pytest.approx(5.0)
    rows = p.read_rows(0, 2)
    assert rows[0].tolist() == [10.5, 11.0]
    assert np.isnan(rows[1, 0]) and rows[1, 1] == 12.0


def test_bil_cube_is_exposed_band_first(tmp_path):
    cube = np.random.default_rng(0).random((5, 3, 4)).astype(np.float32)  # line, band, sample
    label = write_pds4(tmp_path, cube, data_type="IEEE754MSBSingle",
                       axis_names=("Line", "Band", "Sample"))
    p = load_pds4(label)
    assert p.is_multiband and p.data.shape == (3, 5, 4)
    np.testing.assert_allclose(p.read_rows(1, 3), np.moveaxis(cube, 1, 0)[:, 1:3])


def test_longitudes_in_0_360_are_normalized(tmp_path):
    corners = {k: (lon + 350.0 - 10.0, lat) for k, (lon, lat) in CORNERS.items()}
    p = load_pds4(write_pds4(tmp_path, np.zeros((4, 4)), corners=corners))
    assert p.bbox.min_lon == pytest.approx(-10.0)


def test_bounding_coordinates_and_gsd_inference(tmp_path):
    p = load_pds4(write_pds4(tmp_path, np.zeros((100, 100)), gsd=None,
                             bbox=(0.0, 0.0, 0.1, 0.1)))
    assert p.gsd_m is None and p.bbox is not None
    assert infer_gsd_from_footprint(p) == pytest.approx(30.32, rel=1e-3)


def test_errors(tmp_path):
    with pytest.raises(LabelParseError):
        load_pds4(tmp_path / "missing.xml")
    bad = tmp_path / "bad.xml"
    bad.write_text("<Product_Observational><unclosed>")
    with pytest.raises(LabelParseError):
        load_pds4(bad)
    label = write_pds4(tmp_path, np.zeros((8, 8), dtype=np.uint16))
    (tmp_path / "src.dat").write_bytes(b"\0" * 10)      # truncated data file
    with pytest.raises(LabelParseError):
        load_pds4(label)
