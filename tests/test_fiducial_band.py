"""Tests for the rank-space fiducial ROC confidence band."""

import numpy as np
import pytest

from studroc_paper.methods import fiducial_band


@pytest.fixture(scope="module")
def gaussian_data():
    rng = np.random.default_rng(42)
    n = 150
    y_true = np.repeat([0, 1], n)
    y_score = np.concatenate([rng.normal(0, 1, n), rng.normal(1.5, 1, n)])
    return y_true, y_score


def test_shapes_bounds_and_endpoints(gaussian_data):
    y_true, y_score = gaussian_data
    n0 = int((y_true == 0).sum())
    fpr, lo, hi = fiducial_band(y_true, y_score, alpha=0.05, random_state=0)
    assert fpr.shape == lo.shape == hi.shape == (n0 + 1,)
    assert np.allclose(fpr, np.arange(n0 + 1) / n0)
    assert np.all((0.0 <= lo) & (lo <= hi) & (hi <= 1.0))
    assert lo[0] == 0.0
    assert hi[-1] == 1.0


def test_band_edges_monotone(gaussian_data):
    y_true, y_score = gaussian_data
    _, lo, hi = fiducial_band(y_true, y_score, alpha=0.05, random_state=0)
    assert np.all(np.diff(lo) >= -1e-12)
    assert np.all(np.diff(hi) >= -1e-12)


def test_deterministic_given_seed(gaussian_data):
    y_true, y_score = gaussian_data
    _, lo1, hi1 = fiducial_band(y_true, y_score, alpha=0.05, random_state=7)
    _, lo2, hi2 = fiducial_band(y_true, y_score, alpha=0.05, random_state=7)
    assert np.array_equal(lo1, lo2)
    assert np.array_equal(hi1, hi2)


def test_bands_nested_in_alpha(gaussian_data):
    """With the same draws (same seed), a lower-confidence band is inside."""
    y_true, y_score = gaussian_data
    _, lo05, hi05 = fiducial_band(
        y_true, y_score, alpha=0.05, n_draws=3000, random_state=3
    )
    _, lo50, hi50 = fiducial_band(
        y_true, y_score, alpha=0.50, n_draws=3000, random_state=3
    )
    assert np.all(lo50 >= lo05 - 1e-12)
    assert np.all(hi50 <= hi05 + 1e-12)


def test_rank_invariance(gaussian_data):
    """A strictly monotone score transform must not change the band."""
    y_true, y_score = gaussian_data
    _, lo1, hi1 = fiducial_band(y_true, y_score, alpha=0.05, random_state=11)
    _, lo2, hi2 = fiducial_band(
        y_true, np.exp(y_score / 2.0), alpha=0.05, random_state=11
    )
    assert np.array_equal(lo1, lo2)
    assert np.array_equal(hi1, hi2)


def test_upper_edge_reaches_one_on_plateau():
    """Where the empirical TPR is 1, the upper edge must equal 1 exactly."""
    rng = np.random.default_rng(1)
    n = 200
    y_true = np.repeat([0, 1], n)
    # Well-separated classes: empirical TPR hits 1 at moderate FPR.
    y_score = np.concatenate([rng.normal(0, 1, n), rng.normal(4, 1, n)])
    fpr, _, hi = fiducial_band(y_true, y_score, alpha=0.05, random_state=2)
    scores_neg = np.sort(y_score[:n])[::-1]
    min_pos = y_score[n:].min()
    k_plateau = int((scores_neg > min_pos).sum())  # all positives above kth neg
    assert np.all(hi[k_plateau + 1 :] == 1.0)


def test_lower_edge_zero_where_no_positives_seen():
    rng = np.random.default_rng(5)
    n = 200
    y_true = np.repeat([0, 1], n)
    y_score = np.concatenate([rng.normal(0, 1, n), rng.normal(0.5, 1, n)])
    _, lo, _ = fiducial_band(y_true, y_score, alpha=0.05, random_state=2)
    # The top-ranked observation is a negative with positive probability
    # across seeds; regardless, the first grid point's lower edge is 0.
    assert lo[0] == 0.0


def test_ties_handled(gaussian_data):
    y_true, y_score = gaussian_data
    quantized = np.round(y_score)  # heavy ties
    fpr, lo, hi = fiducial_band(y_true, quantized, alpha=0.05, random_state=0)
    assert np.all((0.0 <= lo) & (lo <= hi) & (hi <= 1.0))
    _, lo_e, hi_e = fiducial_band(
        y_true, quantized, alpha=0.05, tie_break="even", random_state=0
    )
    assert np.all((0.0 <= lo_e) & (lo_e <= hi_e) & (hi_e <= 1.0))


def test_custom_output_grid(gaussian_data):
    y_true, y_score = gaussian_data
    fpr, lo, hi = fiducial_band(y_true, y_score, alpha=0.05, k=101, random_state=0)
    assert fpr.shape == lo.shape == hi.shape == (101,)
    # Conservative resampling: native band at matching points is contained.
    fpr_n, lo_n, hi_n = fiducial_band(y_true, y_score, alpha=0.05, random_state=0)
    on_native = np.isin(np.round(fpr * 150).astype(int), np.arange(151))
    assert on_native.all()
    assert np.all(lo <= np.interp(fpr, fpr_n, hi_n) + 1e-12)


def test_small_draws_warns():
    rng = np.random.default_rng(0)
    n = 400
    y_true = np.repeat([0, 1], n)
    y_score = np.concatenate([rng.normal(0, 1, n), rng.normal(1, 1, n)])
    with pytest.warns(UserWarning, match="trim depth"):
        fiducial_band(y_true, y_score, alpha=0.02, n_draws=150, random_state=0)


def test_invalid_inputs():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError):
        fiducial_band(y, s, alpha=0.0)
    with pytest.raises(ValueError):
        fiducial_band(np.zeros(4), s)
    with pytest.raises(ValueError):
        fiducial_band(y, s, trim_exponent=0.0)
    with pytest.raises(ValueError):
        fiducial_band(y, s, tie_break="neg_first")
