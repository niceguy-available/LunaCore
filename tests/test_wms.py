import numpy as np
import pytest
import requests

from lunacore.config import WMSConfig
from lunacore.errors import WMSError
from lunacore.geo import BoundingBox
from lunacore.wms import build_getmap_params, fetch_reference, plan_reference_grid

BBOX = BoundingBox(10.0, -5.0, 10.2, -4.9)


def _tiff_bytes(w, h):
    from rasterio.io import MemoryFile

    with MemoryFile() as mem:
        with mem.open(driver="GTiff", width=w, height=h, count=1, dtype="uint8") as ds:
            ds.write(np.random.default_rng(0).integers(0, 255, (h, w), dtype=np.uint8), 1)
        return mem.read()


class FakeResponse:
    def __init__(self, status, body, ctype):
        self.status_code, self._body, self.headers = status, body, {"Content-Type": ctype}
        self.text = body.decode(errors="replace")

    def iter_content(self, chunk_size):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, **kw):
        self.calls.append(params)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_plan_grid_square_ground_pixels_and_cap():
    cfg = WMSConfig(min_reference_gsd_m=100.0)
    w, h, gsd = plan_reference_grid(BBOX, 0.25, cfg)
    assert 95.0 < gsd <= 100.0
    assert w / h == pytest.approx(BBOX.width_m / BBOX.height_m, rel=0.02)
    cfg = WMSConfig(min_reference_gsd_m=1.0, max_size_px=512)
    w, h, gsd = plan_reference_grid(BBOX, 1.0, cfg)
    assert max(w, h) <= 512 and gsd > 1.0


def test_axis_order_by_version():
    p11 = build_getmap_params(BBOX, 10, 20, WMSConfig(version="1.1.1"))
    assert p11["SRS"] == "EPSG:4326" and p11["BBOX"].startswith("10.0")
    p13 = build_getmap_params(BBOX, 10, 20, WMSConfig(version="1.3.0"))
    assert p13["CRS"] == "EPSG:4326" and p13["BBOX"].startswith("-5.0")
    assert p13["FORMAT"] == "image/tiff" and p13["REQUEST"] == "GetMap"


def test_fetch_retries_then_georeferences(tmp_path):
    cfg = WMSConfig(min_reference_gsd_m=100.0, backoff_s=0.0, bbox_padding=0.0)
    w, h, _ = plan_reference_grid(BBOX, 1.0, cfg)
    sess = FakeSession([requests.ConnectionError("boom"), FakeResponse(503, b"", "text/plain"),
                        FakeResponse(200, _tiff_bytes(w, h), "image/tiff")])
    ref = fetch_reference(BBOX, 1.0, cfg, tmp_path, session=sess)
    assert len(sess.calls) == 3
    assert ref.shape == (h, w)
    assert ref.transform.c == pytest.approx(10.0) and ref.transform.f == pytest.approx(-4.9)
    assert "Moon" in ref.crs.to_wkt()


def test_service_exception_is_reported(tmp_path):
    xml = b'<?xml version="1.0"?><ServiceExceptionReport>LayerNotDefined</ServiceExceptionReport>'
    sess = FakeSession([FakeResponse(200, xml, "application/vnd.ogc.se_xml")])
    with pytest.raises(WMSError, match="LayerNotDefined"):
        fetch_reference(BBOX, 1.0, WMSConfig(backoff_s=0.0), tmp_path, session=sess)


def test_non_image_body_is_rejected(tmp_path):
    sess = FakeSession([FakeResponse(200, b"<html>login</html>", "image/tiff")])
    with pytest.raises(WMSError):
        fetch_reference(BBOX, 1.0, WMSConfig(backoff_s=0.0), tmp_path, session=sess)
