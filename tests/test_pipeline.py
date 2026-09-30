import json

import numpy as np
import pytest
import rasterio

from lunacore.cli import main
from lunacore.config import PipelineConfig
from lunacore.pipeline import register
from synthetic import Scene


def _cfg():
    cfg = PipelineConfig()
    cfg.matching.matcher = "sift"   # LoFTR weights need network access
    return cfg


@pytest.mark.parametrize("kwargs", [
    {},                                              # geo-prior from label corners
    {"with_corners": False},                         # GSD-only scale prior
    {"src_sun": (315.0, 30.0)},                      # opposite sun: inverted shadows
], ids=["corners", "scale-only", "inverted-shadows"])
def test_end_to_end_subpixel(tmp_path, kwargs):
    scene = Scene(tmp_path, **kwargs)
    res = register(scene.label, tmp_path / "out", _cfg(), reference_path=scene.ref_path)
    assert scene.error_ref_px(res.H_source_to_reference) < 0.75
    m = res.metrics
    assert m["n_inliers"] >= 30 and m["scale_ratio_S"] == pytest.approx(4.0)
    assert m["rmse_checkpoint_ref_px"] < 1.0 and m["subpixel"]
    assert abs(m["zncc"]) > 0.9
    if "src_sun" in kwargs:
        assert m["polarity"] == "inverted"
    with rasterio.open(res.registered_path) as ds:
        assert ds.shape == (300, 300) and "Moon" in ds.crs.to_wkt()
        assert np.isfinite(ds.read(1)).mean() > 0.2
    report = json.loads(res.report_path.read_text())
    assert np.array(report["homography_source_to_reference"]).shape == (3, 3)


def test_cli(tmp_path, capsys):
    scene = Scene(tmp_path)
    rc = main(["register", str(scene.label), "-o", str(tmp_path / "cli"),
               "--reference", str(scene.ref_path), "--matcher", "sift"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["metrics"]["n_inliers"] > 0


def test_cli_reports_failure(tmp_path):
    assert main(["register", str(tmp_path / "none.xml"), "-o", str(tmp_path)]) == 2


def test_tiled_matching_path(tmp_path):
    scene = Scene(tmp_path)
    cfg = _cfg()
    cfg.matching.max_side_px = 128          # 300 px canvas -> co-located tiles
    res = register(scene.label, tmp_path / "out", cfg, reference_path=scene.ref_path)
    assert scene.error_ref_px(res.H_source_to_reference) < 0.75


def test_wms_driven_run(tmp_path):
    """Phase 1 end to end: the footprint drives a GetMap request whose image is used."""
    from test_wms import FakeResponse, FakeSession

    scene = Scene(tmp_path)
    cfg = _cfg()
    cfg.wms.bbox_padding = 0.0
    cfg.wms.min_reference_gsd_m = 4.0
    # Serve the scene's reference resampled onto exactly the grid the pipeline
    # will request for the label's footprint.
    from lunacore.ingestion import load_pds4
    from lunacore.wms import plan_reference_grid

    bbox = load_pds4(scene.label).bbox
    w, h, _ = plan_reference_grid(bbox, 1.0, cfg.wms)
    import cv2
    import rasterio
    from synthetic import write_reference_geotiff

    with rasterio.open(scene.ref_path) as ds:
        win = rasterio.windows.from_bounds(*bbox.as_tuple(), transform=ds.transform)
        crop = ds.read(1, window=win.round_offsets().round_lengths())
    served = tmp_path / "served.tif"
    write_reference_geotiff(served, cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA),
                            bbox.as_tuple())
    session = FakeSession([FakeResponse(200, served.read_bytes(), "image/tiff")])
    res = register(scene.label, tmp_path / "out", cfg, session=session)
    params = session.calls[0]
    assert params["REQUEST"] == "GetMap" and params["WIDTH"] == str(w)
    assert (tmp_path / "out" / "reference_wms.tif").exists()
    assert res.metrics["n_inliers"] >= 20
    assert res.metrics["rmse_checkpoint_ref_px"] < 1.0
