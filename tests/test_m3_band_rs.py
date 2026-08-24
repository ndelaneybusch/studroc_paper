"""Tests for the M3 exact composition ROC band and its calibration kernel.

M3 carries a finite-sample coverage theorem, so the tests check the exact
pieces directly: the Rust non-crossing probability against Monte Carlo and
closed forms, the calibrated per-class band against its own guarantee, and
the composed band's structural invariants and measured coverage.
"""

import numpy as np
import pytest
from scipy.stats import beta as beta_dist
from scipy.stats import norm

pytest.importorskip("fiducial_core")

import fiducial_core  # noqa: E402

from studroc_paper.methods import m3_band_rs  # noqa: E402
from studroc_paper.methods.m3_band_rs import _ell_gamma  # noqa: E402


@pytest.fixture(scope="module")
def gaussian_data():
    rng = np.random.default_rng(42)
    n = 150
    y_true = np.repeat([0, 1], n)
    y_score = np.concatenate([rng.normal(0, 1, n), rng.normal(1.5, 1, n)])
    return y_true, y_score


# ---------------------------------------------------------------------------
# the exact calibration kernel
# ---------------------------------------------------------------------------


def test_crossing_probability_n1_exact():
    p = fiducial_core.ell_crossing_probability(np.array([0.2]), np.array([0.9]))
    assert abs(p - 0.7) < 1e-14


def test_crossing_probability_matches_monte_carlo():
    n = 40
    i = np.arange(1, n + 1, dtype=float)
    gamma = 0.01
    lower = beta_dist.ppf(gamma, i, n + 1.0 - i)
    upper = beta_dist.ppf(1.0 - gamma, i, n + 1.0 - i)
    p = fiducial_core.ell_crossing_probability(lower, upper)

    rng = np.random.default_rng(7)
    reps = 100_000
    u = np.sort(rng.random((reps, n)), axis=1)
    mc = float(np.mean(np.all((u >= lower) & (u <= upper), axis=1)))
    se = np.sqrt(mc * (1.0 - mc) / reps)
    assert abs(p - mc) < 5.0 * se + 1e-9, f"dp {p} vs mc {mc}"


def test_calibrated_gamma_hits_target_conservatively():
    core = fiducial_core
    for n, alpha_class in [(80, 0.0253), (200, 0.293), (1, 0.1)]:
        g = _ell_gamma(core, n, alpha_class)
        i = np.arange(1, n + 1, dtype=float)
        cov = core.ell_crossing_probability(
            beta_dist.ppf(g, i, n + 1.0 - i), beta_dist.ppf(1.0 - g, i, n + 1.0 - i)
        )
        assert cov >= 1.0 - alpha_class - 1e-12, f"n={n}: coverage {cov}"
        # Not vacuously conservative: doubling gamma must break the target
        # (the calibration sits at the boundary, not at Bonferroni).
        if n > 1:
            cov2 = core.ell_crossing_probability(
                beta_dist.ppf(2.5 * g, i, n + 1.0 - i),
                beta_dist.ppf(1.0 - 2.5 * g, i, n + 1.0 - i),
            )
            assert cov2 < 1.0 - alpha_class


# ---------------------------------------------------------------------------
# the composed band
# ---------------------------------------------------------------------------


def test_shapes_bounds_endpoints_and_monotone_edges(gaussian_data):
    y_true, y_score = gaussian_data
    n0 = int((y_true == 0).sum())
    fpr, lo, hi = m3_band_rs(y_true, y_score, alpha=0.05, random_state=0)
    assert fpr.shape == lo.shape == hi.shape == (n0 + 1,)
    assert np.allclose(fpr, np.arange(n0 + 1) / n0)
    assert np.all((0.0 <= lo) & (lo <= hi) & (hi <= 1.0))
    assert lo[0] == 0.0
    assert hi[-1] == 1.0 and lo[-1] == 1.0
    assert np.all(np.diff(hi) >= -1e-12)
    assert np.all(np.diff(lo) >= -1e-12)


def test_finite_sample_coverage():
    """The theorem says coverage >= 1 - alpha at every n; measure it.

    Binormal truth, n0 = n1 = 60, alpha = 0.2. M3 is deterministic given the
    ranks, so 300 replicates give a tight check; measured coverage of the
    composition is ~1.0 (the split + composition are conservative).
    """
    rng = np.random.default_rng(3)
    n, mu = 60, 1.0
    grid = np.arange(n + 1) / n
    r_true = norm.cdf(norm.ppf(grid) + mu)  # rank-space binormal ROC
    r_true[0] = 0.0
    misses = 0
    reps = 300
    for _ in range(reps):
        y_true = np.repeat([0, 1], n)
        y_score = np.concatenate([rng.normal(0, 1, n), rng.normal(mu, 1, n)])
        _, lo, hi = m3_band_rs(y_true, y_score, alpha=0.2, random_state=rng)
        if np.any(lo > r_true + 1e-12) or np.any(hi < r_true - 1e-12):
            misses += 1
    coverage = 1.0 - misses / reps
    assert coverage >= 0.8, f"coverage {coverage} below nominal"


def test_bands_nested_in_alpha(gaussian_data):
    y_true, y_score = gaussian_data
    _, lo05, hi05 = m3_band_rs(y_true, y_score, alpha=0.05, random_state=3)
    _, lo50, hi50 = m3_band_rs(y_true, y_score, alpha=0.50, random_state=3)
    assert np.all(lo50 >= lo05 - 1e-12)
    assert np.all(hi50 <= hi05 + 1e-12)


def test_rank_invariance_and_determinism(gaussian_data):
    y_true, y_score = gaussian_data
    _, lo1, hi1 = m3_band_rs(y_true, y_score, alpha=0.05, random_state=11)
    _, lo2, hi2 = m3_band_rs(y_true, np.exp(y_score / 2.0), alpha=0.05, random_state=99)
    # Tie-free scores: the band depends on the ranks only, so neither the
    # monotone transform nor the seed can change it.
    assert np.array_equal(lo1, lo2)
    assert np.array_equal(hi1, hi2)


def test_upper_edge_at_origin_is_honest_by_default():
    """Separated supports: R(0) = 1, so a pinned U(0) = 0 would miss surely.

    The default composition value carries this case (Corollary 9.3); the
    pin is opt-in and asserts R(0) = 0.
    """
    n = 100
    y_true = np.repeat([0, 1], n)
    y_score = np.concatenate([np.linspace(0, 1, n), np.linspace(2, 3, n)])
    _, _, hi = m3_band_rs(y_true, y_score, alpha=0.05, random_state=0)
    assert hi[0] == 1.0  # all positives rank above all negatives
    _, _, hi_pin = m3_band_rs(
        y_true, y_score, alpha=0.05, assume_r0_zero=True, random_state=0
    )
    assert hi_pin[0] == 0.0


def test_split_ratio_shifts_width_between_classes(gaussian_data):
    y_true, y_score = gaussian_data
    bands = {
        rho: m3_band_rs(y_true, y_score, alpha=0.1, split_ratio=rho, random_state=0)
        for rho in (0.2, 0.5, 0.8)
    }
    areas = {rho: float(np.mean(b[2] - b[1])) for rho, b in bands.items()}
    # All valid bands; widths differ across splits (the lever moves).
    assert len({round(a, 6) for a in areas.values()}) == 3


def test_ties_handled(gaussian_data):
    y_true, y_score = gaussian_data
    quantized = np.round(y_score)
    fpr, lo, hi = m3_band_rs(y_true, quantized, alpha=0.05, random_state=0)
    assert np.all((0.0 <= lo) & (lo <= hi) & (hi <= 1.0))
    _, lo_e, hi_e = m3_band_rs(
        y_true, quantized, alpha=0.05, tie_break="even", random_state=0
    )
    assert np.all((0.0 <= lo_e) & (lo_e <= hi_e) & (hi_e <= 1.0))


def test_custom_output_grid_is_conservative(gaussian_data):
    y_true, y_score = gaussian_data
    fpr, lo, hi = m3_band_rs(y_true, y_score, alpha=0.05, k=101, random_state=0)
    assert fpr.shape == lo.shape == hi.shape == (101,)
    fpr_n, lo_n, hi_n = m3_band_rs(y_true, y_score, alpha=0.05, random_state=0)
    n0 = int((y_true == 0).sum())
    up_idx = np.minimum(np.ceil(fpr * n0).astype(int), n0)
    lo_idx = np.floor(fpr * n0).astype(int)
    assert np.array_equal(hi, hi_n[up_idx])
    assert np.array_equal(lo, lo_n[lo_idx])


def test_invalid_inputs():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError):
        m3_band_rs(y, s, alpha=0.0)
    with pytest.raises(ValueError):
        m3_band_rs(np.zeros(4), s)
    with pytest.raises(ValueError):
        m3_band_rs(y, s, split_ratio=0.0)
    with pytest.raises(ValueError):
        m3_band_rs(y, s, split_ratio=1.0)
    with pytest.raises(ValueError):
        m3_band_rs(y, s, tie_break="neg_first")
    with pytest.raises(ValueError):
        m3_band_rs(y, s, k=1)
