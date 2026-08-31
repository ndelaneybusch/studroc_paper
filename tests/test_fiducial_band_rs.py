"""Tests for the Rust-accelerated rank-space fiducial ROC band.

The reference implementation (`fiducial_band`) is the oracle: the Rust band
must satisfy every structural invariant of the method exactly and agree with
the reference within its own Monte Carlo variability (the two consume
independent RNG streams, so agreement is statistical, not bit-wise).
"""

import numpy as np
import pytest

pytest.importorskip("fiducial_core")

from studroc_paper.methods import (  # noqa: E402
    fiducial_band,
    fiducial_band_rs,
    production_trim_rows,
)


@pytest.fixture(scope="module")
def gaussian_data():
    rng = np.random.default_rng(42)
    n = 150
    y_true = np.repeat([0, 1], n)
    y_score = np.concatenate([rng.normal(0, 1, n), rng.normal(1.5, 1, n)])
    return y_true, y_score


def test_shapes_bounds_endpoints_and_monotone_edges(gaussian_data):
    y_true, y_score = gaussian_data
    n0 = int((y_true == 0).sum())
    fpr, lo, hi = fiducial_band_rs(y_true, y_score, alpha=0.05, random_state=0)
    assert fpr.shape == lo.shape == hi.shape == (n0 + 1,)
    assert np.allclose(fpr, np.arange(n0 + 1) / n0)
    assert np.all((0.0 <= lo) & (lo <= hi) & (hi <= 1.0))
    assert lo[0] == 0.0
    assert hi[-1] == 1.0
    assert np.all(np.diff(hi) >= -1e-12)


def test_agrees_with_reference_within_monte_carlo_noise(gaussian_data):
    """Cross-implementation spread must match the implementations' own MC spread.

    Both edges are extreme order statistics of independent Monte Carlo
    clouds, so the honest yardstick is seed-to-seed sup-norm variability —
    of whichever implementation is noisier (at the C = 1 default the trim
    is shallow and the Rust path's own spread is the larger one), not an
    absolute tolerance.
    """
    y_true, y_score = gaussian_data
    seeds = range(4)
    py = [fiducial_band(y_true, y_score, n_draws=4000, random_state=s) for s in seeds]
    rs = [
        fiducial_band_rs(y_true, y_score, n_draws=4000, random_state=s) for s in seeds
    ]

    def sup(a, b):
        return max(np.abs(a[1] - b[1]).max(), np.abs(a[2] - b[2]).max())

    def spread(bands):
        return max(sup(a, b) for i, a in enumerate(bands) for b in bands[i + 1 :])

    own_spread = max(spread(py), spread(rs))
    cross_spread = max(sup(a, b) for a in py for b in rs)
    assert cross_spread <= 2.0 * own_spread + 0.01, (
        f"rust-vs-python spread {cross_spread:.4f} exceeds twice the "
        f"implementations' own worst seed spread {own_spread:.4f}"
    )


def test_deterministic_given_seed_and_sensitive_to_seed(gaussian_data):
    y_true, y_score = gaussian_data
    a = fiducial_band_rs(y_true, y_score, alpha=0.05, random_state=7)
    b = fiducial_band_rs(y_true, y_score, alpha=0.05, random_state=7)
    c = fiducial_band_rs(y_true, y_score, alpha=0.05, random_state=8)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert not np.array_equal(a[1], c[1]) or not np.array_equal(a[2], c[2])


def test_thread_count_does_not_change_output(gaussian_data):
    y_true, y_score = gaussian_data
    a = fiducial_band_rs(y_true, y_score, n_threads=1, random_state=3)
    b = fiducial_band_rs(y_true, y_score, n_threads=4, random_state=3)
    c = fiducial_band_rs(y_true, y_score, n_threads=0, random_state=3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert all(np.array_equal(x, y) for x, y in zip(a, c, strict=True))


def test_bands_nested_in_alpha(gaussian_data):
    y_true, y_score = gaussian_data
    _, lo05, hi05 = fiducial_band_rs(
        y_true, y_score, alpha=0.05, n_draws=3000, random_state=3
    )
    _, lo50, hi50 = fiducial_band_rs(
        y_true, y_score, alpha=0.50, n_draws=3000, random_state=3
    )
    assert np.all(lo50 >= lo05 - 1e-12)
    assert np.all(hi50 <= hi05 + 1e-12)


def test_rank_invariance(gaussian_data):
    """A strictly monotone score transform must not change the band."""
    y_true, y_score = gaussian_data
    _, lo1, hi1 = fiducial_band_rs(y_true, y_score, alpha=0.05, random_state=11)
    _, lo2, hi2 = fiducial_band_rs(
        y_true, np.exp(y_score / 2.0), alpha=0.05, random_state=11
    )
    assert np.array_equal(lo1, lo2)
    assert np.array_equal(hi1, hi2)


def test_upper_edge_reaches_one_on_plateau():
    """Where the empirical TPR is 1, the upper edge must equal 1 exactly."""
    rng = np.random.default_rng(1)
    n = 200
    y_true = np.repeat([0, 1], n)
    y_score = np.concatenate([rng.normal(0, 1, n), rng.normal(4, 1, n)])
    fpr, _, hi = fiducial_band_rs(y_true, y_score, alpha=0.05, random_state=2)
    scores_neg = np.sort(y_score[:n])[::-1]
    min_pos = y_score[n:].min()
    k_plateau = int((scores_neg > min_pos).sum())
    assert np.all(hi[k_plateau + 1 :] == 1.0)


def test_cp_allowance_lifts_upper_edge_at_origin():
    """On separated supports the upper edge at t=0 must stay off 0.

    The raw fiducial cloud is identically 0 at t=0; only the Clopper-Pearson
    allowance keeps the band valid there (theory doc, Corollary 9.3).
    """
    n = 100
    y_true = np.repeat([0, 1], n)
    y_score = np.concatenate([np.linspace(0, 1, n), np.linspace(2, 3, n)])
    _, _, hi = fiducial_band_rs(y_true, y_score, alpha=0.05, random_state=0)
    assert hi[0] > 0.5  # empirical TPR at t=0 is 1; CP bound sits near 1


def test_ties_handled(gaussian_data):
    y_true, y_score = gaussian_data
    quantized = np.round(y_score)
    fpr, lo, hi = fiducial_band_rs(y_true, quantized, alpha=0.05, random_state=0)
    assert np.all((0.0 <= lo) & (lo <= hi) & (hi <= 1.0))
    _, lo_e, hi_e = fiducial_band_rs(
        y_true, quantized, alpha=0.05, tie_break="even", random_state=0
    )
    assert np.all((0.0 <= lo_e) & (lo_e <= hi_e) & (hi_e <= 1.0))


def test_custom_output_grid_is_conservative(gaussian_data):
    y_true, y_score = gaussian_data
    fpr, lo, hi = fiducial_band_rs(y_true, y_score, alpha=0.05, k=101, random_state=0)
    assert fpr.shape == lo.shape == hi.shape == (101,)
    fpr_n, lo_n, hi_n = fiducial_band_rs(y_true, y_score, alpha=0.05, random_state=0)
    n0 = int((y_true == 0).sum())
    up_idx = np.minimum(np.ceil(fpr * n0).astype(int), n0)
    lo_idx = np.floor(fpr * n0).astype(int)
    assert np.array_equal(hi, hi_n[up_idx])
    assert np.array_equal(lo, lo_n[lo_idx])


def test_small_draws_warns():
    rng = np.random.default_rng(0)
    n = 400
    y_true = np.repeat([0, 1], n)
    y_score = np.concatenate([rng.normal(0, 1, n), rng.normal(1, 1, n)])
    with pytest.warns(UserWarning, match="trim depth"):
        fiducial_band_rs(y_true, y_score, alpha=0.02, n_draws=150, random_state=0)


def test_production_trim_rows_rule():
    """The section-5.3 thinning rule: full grid through 2001 points, then
    every ceil(K/1000)-th point plus 50 edge points at each end."""
    assert production_trim_rows(101) is None
    assert production_trim_rows(2001) is None
    rows = production_trim_rows(5001)
    assert rows is not None
    assert rows[0] == 0 and rows[-1] == 5000
    assert np.all(np.diff(rows) > 0)
    step = int(np.ceil(5001 / 1000))
    expected = sorted(
        set(range(0, 5001, step)) | set(range(50)) | set(range(5001 - 50, 5001))
    )
    assert np.array_equal(rows, expected)
    # Edge blocks are contiguous single steps.
    assert np.all(np.diff(rows[:50]) == 1)
    assert np.all(np.diff(rows[-50:]) == 1)


def test_thinned_trim_grid_band_is_valid_and_deterministic():
    """K > 2001 routes the trim through the thinned grid; the band must keep
    every structural invariant and stay seed-deterministic."""
    rng = np.random.default_rng(5)
    n0, n1 = 2200, 300
    y_true = np.concatenate([np.zeros(n0), np.ones(n1)])
    y_score = np.concatenate([rng.normal(0, 1, n0), rng.normal(1.2, 1, n1)])
    fpr, lo, hi = fiducial_band_rs(
        y_true, y_score, alpha=0.05, n_draws=2000, random_state=1
    )
    assert fpr.shape == (n0 + 1,)
    assert np.all((0.0 <= lo) & (lo <= hi) & (hi <= 1.0))
    assert lo[0] == 0.0 and hi[-1] == 1.0
    assert np.all(np.diff(hi) >= -1e-12)
    _, lo2, hi2 = fiducial_band_rs(
        y_true, y_score, alpha=0.05, n_draws=2000, random_state=1
    )
    assert np.array_equal(lo, lo2) and np.array_equal(hi, hi2)


def test_invalid_inputs():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError):
        fiducial_band_rs(y, s, alpha=0.0)
    with pytest.raises(ValueError):
        fiducial_band_rs(np.zeros(4), s)
    with pytest.raises(ValueError):
        fiducial_band_rs(y, s, trim_exponent=0.0)
    with pytest.raises(ValueError):
        fiducial_band_rs(y, s, tie_break="neg_first")
    with pytest.raises(ValueError):
        fiducial_band_rs(y, s, n_draws=50)
    with pytest.raises(ValueError):
        fiducial_band_rs(y, s, k=1)
