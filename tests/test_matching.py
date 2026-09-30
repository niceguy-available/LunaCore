import numpy as np
import pytest

from lunacore.matching import Matches, _tiles, quadtree_distribute


def _matches(pts, conf=None):
    pts = np.asarray(pts, dtype=np.float64)
    return Matches(pts, pts + 1.0, np.ones(len(pts)) if conf is None else np.asarray(conf))


def test_quadtree_thins_clusters_but_keeps_sparse_points():
    rng = np.random.default_rng(0)
    cluster = rng.uniform(0, 20, (500, 2))
    sparse = np.array([[900.0, 900.0], [100.0, 800.0], [800.0, 100.0]])
    m = _matches(np.vstack([cluster, sparse]), rng.random(503))
    out = quadtree_distribute(m, 1000, 1000, max_depth=6, capacity=4)
    assert len(out) < 60
    for p in sparse:
        assert (np.abs(out.pts0 - p).sum(axis=1) < 1e-9).any()
    # Correspondence pairing is preserved.
    np.testing.assert_allclose(out.pts1, out.pts0 + 1.0)


def test_quadtree_prefers_confident_matches_and_caps_total():
    pts = np.full((10, 2), 5.0)
    conf = np.arange(10) / 10
    out = quadtree_distribute(_matches(pts, conf), 10, 10, max_depth=2, capacity=3)
    assert sorted(out.confidence.tolist()) == pytest.approx([0.7, 0.8, 0.9])
    grid = np.stack(np.meshgrid(np.arange(0, 100, 5.0), np.arange(0, 100, 5.0)), -1).reshape(-1, 2)
    out = quadtree_distribute(_matches(grid), 100, 100, max_matches=50)
    assert len(out) == 50


def test_tiles_cover_image():
    tiles = _tiles(1000, 700, 512, 64)
    cover = np.zeros((1000, 700), bool)
    for y0, y1, x0, x1 in tiles:
        cover[y0:y1, x0:x1] = True
    assert cover.all()


def test_loftr_plumbing_with_random_weights():
    pytest.importorskip("torch")
    KF = pytest.importorskip("kornia.feature")
    from lunacore import matching

    matching._MODEL_CACHE[("random", "cpu")] = KF.LoFTR(pretrained=None).eval()
    m = matching.LoFTRMatcher(weights="random", device="cpu", max_side_px=160)
    img = (np.random.default_rng(0).random((250, 330)) * 255).astype(np.uint8)
    out = m.match(img, img[:200, :240].copy())
    assert out.pts0.shape == out.pts1.shape and out.pts0.shape[1] == 2
    if len(out):
        assert out.pts0[:, 0].max() < 330 and out.pts1[:, 0].max() < 240
