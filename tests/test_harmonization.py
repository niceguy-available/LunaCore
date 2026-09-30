import numpy as np
import pytest

from lunacore.config import HarmonizationConfig
from lunacore.geometry import apply_homography
from lunacore.harmonization import (compute_scale_ratio, fit_pca_streaming,
                                    normalize_to_uint8, pca_row_reader, reduce_image)
from lunacore.ingestion import load_pds4
from synthetic import write_pds4


def test_scale_ratio_definition():
    assert compute_scale_ratio(5.0, 0.25) == pytest.approx(20.0)


@pytest.mark.parametrize("factor", [1.0, 3.0, 4.5])
def test_reduce_image_coordinate_mapping(factor):
    # A linear ramp in x: the reduced value at each pixel must equal the
    # source x-coordinate that the returned matrix maps onto that pixel.
    h, w = 90, 120
    ramp = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    small, D = reduce_image(lambda a, b: ramp[a:b], h, w, factor, strip_rows=16)
    Dinv = np.linalg.inv(D)
    ys, xs = np.mgrid[2:small.shape[0] - 2, 2:small.shape[1] - 2]
    pts = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    err = small[ys, xs].ravel() - apply_homography(Dinv, pts)[:, 0]
    # Box-filtering pixelated data jitters samples for fractional factors, but
    # the mapping itself must be unbiased.
    assert abs(err.mean()) < 0.02
    assert np.abs(err).max() < (0.05 if factor == int(factor) else 0.25)


def test_reduce_image_ignores_nodata():
    img = np.ones((8, 8), np.float32)
    img[0, 0] = np.nan
    img[4:, 4:] = np.nan
    small, _ = reduce_image(lambda a, b: img[a:b], 8, 8, 4.0)
    assert small[0, 0] == 1.0 and np.isnan(small[1, 1])


def test_streaming_pca_matches_numpy(tmp_path):
    rng = np.random.default_rng(1)
    base = rng.random((40, 30)).astype(np.float32)
    loadings = np.linspace(1.0, 3.0, 6)
    cube = base[None] * loadings[:, None, None] + 0.01 * rng.standard_normal((6, 40, 30))
    cube[:, 5, 5] = -9999.0
    label = write_pds4(tmp_path, cube.astype(np.float32), data_type="IEEE754LSBSingle",
                       axis_names=("Band", "Line", "Sample"),
                       special={"missing_constant": "-9999.0"})
    prod = load_pds4(label)
    pca = fit_pca_streaming(prod, strip_rows=7)
    X = cube.reshape(6, -1)
    X = X[:, (X != -9999.0).all(axis=0)]
    evals, evecs = np.linalg.eigh(np.cov(X, bias=True))
    assert abs(pca.component @ evecs[:, -1]) == pytest.approx(1.0, abs=1e-6)
    assert pca.explained_variance_ratio > 0.99
    pc1 = pca_row_reader(prod, pca)(0, 40)
    assert np.isnan(pc1[5, 5])
    assert np.corrcoef(pc1[np.isfinite(pc1)], base[np.isfinite(pc1)])[0, 1] > 0.999


def test_clahe_normalization_masks_invalid():
    img = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    valid = np.ones_like(img, bool)
    valid[:8] = False
    out = normalize_to_uint8(img, valid, HarmonizationConfig())
    assert out.dtype == np.uint8 and (out[:8] == 0).all() and out[8:].std() > 30
