import numpy as np
import pytest

from lunacore.config import GeometryConfig
from lunacore.errors import GeometryError
from lunacore.geometry import (apply_homography, estimate_homography, normalized_dlt,
                               validate_homography, warp_to_reference)
from lunacore.matching import Matches

H_TRUE = np.array([[0.98, -0.12, 14.0], [0.11, 1.01, -7.5], [2e-5, -1e-5, 1.0]])


def test_normalized_dlt_recovers_exact_homography():
    pts = np.random.default_rng(0).uniform(0, 500, (30, 2))
    H = normalized_dlt(pts, apply_homography(H_TRUE, pts))
    np.testing.assert_allclose(H, H_TRUE, rtol=1e-8, atol=1e-10)


def test_dlt_rejects_collinear_points():
    pts = np.column_stack([np.arange(10.0), np.arange(10.0)])
    with pytest.raises(GeometryError):
        normalized_dlt(pts, pts)


def test_magsac_rejects_outliers_with_subpixel_accuracy():
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 500, (300, 2))
    dst = apply_homography(H_TRUE, src) + rng.normal(0, 0.3, (300, 2))
    outliers = rng.random(300) < 0.4
    dst[outliers] = rng.uniform(0, 500, (outliers.sum(), 2))
    est = estimate_homography(Matches(src, dst, np.ones(300)), GeometryConfig(), (500, 500))
    assert (est.inliers & outliers).sum() <= 2
    assert est.inliers[~outliers].mean() > 0.97
    grid = rng.uniform(0, 500, (50, 2))
    err = np.abs(apply_homography(est.H, grid) - apply_homography(H_TRUE, grid)).max()
    assert err < 0.3


def test_too_few_matches():
    m = Matches(np.zeros((5, 2)), np.zeros((5, 2)), np.ones(5))
    with pytest.raises(GeometryError):
        estimate_homography(m, GeometryConfig(), (100, 100))


def test_validate_rejects_mirror_and_extreme_scale():
    with pytest.raises(GeometryError):
        validate_homography(np.diag([-1.0, 1.0, 1.0]), 100, 100, 4.0)
    with pytest.raises(GeometryError):
        validate_homography(np.diag([10.0, 10.0, 1.0]), 100, 100, 4.0)


def test_warp_places_pixels_and_propagates_nodata():
    src = np.zeros((50, 50), np.float32)
    src[20, 30] = 1.0
    src[0:5, 0:5] = np.nan
    M = np.array([[1, 0, 7.0], [0, 1, 3.0], [0, 0, 1]])
    out, valid = warp_to_reference(src, M, (60, 60))
    assert np.unravel_index(np.nanargmax(out), out.shape) == (23, 37)
    assert not valid[3:8, 7:12].any()        # the NaN block moved with the image
    assert not valid[0, 0] and valid[30, 30]
