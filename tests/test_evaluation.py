import numpy as np
import pytest

from lunacore.evaluation import checkpoint_rmse, rmse, uniformity, zncc


def test_rmse_formula():
    a = np.array([[0.0, 0.0], [1.0, 1.0]])
    b = np.array([[3.0, 4.0], [1.0, 1.0]])
    assert rmse(a, b) == pytest.approx(np.sqrt(25 / 2))


def test_zncc_brightness_invariance_and_sign():
    x = np.random.default_rng(0).random((32, 32))
    assert zncc(x, 3 * x + 7) == pytest.approx(1.0)
    assert zncc(x, -x) == pytest.approx(-1.0)
    mask = np.zeros_like(x, bool)
    mask[:16] = True
    y = x.copy()
    y[16:] = 0
    assert zncc(x, y, mask) == pytest.approx(1.0)


def test_uniformity_with_footprint():
    pts = np.array([[5.0, 5.0], [15.0, 5.0], [5.0, 15.0], [15.0, 15.0]])
    u = uniformity(pts, 20, 20, grid=2)
    assert u == {"coverage": 1.0, "entropy": pytest.approx(1.0)}
    fp = np.zeros((20, 20), bool)
    fp[:10] = True
    u = uniformity(pts[:2], 20, 20, grid=2, footprint=fp)
    assert u["coverage"] == 1.0


def test_checkpoint_rmse_is_small_for_consistent_points():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 100, (60, 2))
    dst = src * 1.1 + 3 + rng.normal(0, 0.1, src.shape)
    err, n = checkpoint_rmse(src, dst, 0.2, 0)
    assert n == 12 and err < 0.3
