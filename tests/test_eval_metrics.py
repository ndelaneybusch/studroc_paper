"""Tests for the coverage/width evaluation metric.

Threat model: ``evaluate_single_band`` and ``aggregate_band_results`` are the
lens through which *every* coverage number and figure in the paper is computed.
A silent bug here — a mis-placed boundary mask, a wrong tolerance, an integration
slip — would corrupt every headline claim while leaving the band code untouched.
These tests pin the metric to hand-computable truth.

Tolerances: coverage flags and counts are exact (``==``); violation magnitudes
and areas are reconstructed in float64 and checked at ``atol=1e-12`` (the metric
does only subtraction and trapezoid integration, both exact to rounding here).
The lone exception is the documented ``1e-6`` coverage tolerance, which is probed
from both sides rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from studroc_paper.eval.eval import aggregate_band_results, evaluate_single_band

# An 11-point grid keeps interior indices (1..9) easy to reason about; sqrt is a
# valid concave ROC pinned at (0,0) and (1,1) with strictly interior middle.
GRID = np.linspace(0.0, 1.0, 11)
TRUTH = np.sqrt(GRID)


def _covered_band(half_width: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """A symmetric band that strictly brackets ``TRUTH`` at every interior point."""
    lower = np.clip(TRUTH - half_width, 0.0, 1.0)
    upper = np.clip(TRUTH + half_width, 0.0, 1.0)
    return lower, upper


# ---------------------------------------------------------------------------
# evaluate_single_band: coverage detection and violation magnitudes
# ---------------------------------------------------------------------------


def test_bracketing_band_covers_with_zero_violation():
    lower, upper = _covered_band()
    result = evaluate_single_band(lower, upper, TRUTH, GRID)

    assert result.covers_entirely
    assert not result.violation_above
    assert not result.violation_below
    assert result.max_violation_above == 0.0
    assert result.max_violation_below == 0.0
    assert result.violation_area_above == 0.0
    assert result.violation_area_below == 0.0


def test_violation_above_flags_and_exact_magnitude():
    lower, upper = _covered_band()
    gap = 0.02
    i = 4  # interior
    upper[i] = TRUTH[i] - gap  # true ROC now pokes above the upper band

    result = evaluate_single_band(lower, upper, TRUTH, GRID)

    assert not result.covers_entirely
    assert result.violation_above
    assert not result.violation_below
    assert result.max_violation_above == pytest.approx(gap, abs=1e-12)
    assert result.max_violation_below == 0.0
    # Violation is located at exactly the tampered FPR and nowhere else.
    np.testing.assert_array_equal(result.violation_fpr_above, [GRID[i]])


def test_violation_below_flags_and_exact_magnitude():
    lower, upper = _covered_band()
    gap = 0.03
    i = 6  # interior
    lower[i] = TRUTH[i] + gap  # lower band now sits above the true ROC

    result = evaluate_single_band(lower, upper, TRUTH, GRID)

    assert not result.covers_entirely
    assert result.violation_below
    assert not result.violation_above
    assert result.max_violation_below == pytest.approx(gap, abs=1e-12)
    np.testing.assert_array_equal(result.violation_fpr_below, [GRID[i]])


# ---------------------------------------------------------------------------
# Boundary masking: FPR=0 and FPR=1 are pinned and must not count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("boundary_index", [0, -1], ids=["fpr_0", "fpr_1"])
def test_boundary_violation_is_excluded_from_coverage(boundary_index):
    lower, upper = _covered_band()
    # Force a gross below-violation exactly on a pinned boundary point.
    lower[boundary_index] = 0.5
    upper[boundary_index] = 0.6

    result = evaluate_single_band(lower, upper, TRUTH, GRID)

    assert result.covers_entirely
    assert not result.violation_below


def test_interior_violation_adjacent_to_boundary_is_counted():
    lower, upper = _covered_band()
    # Same tampering one step into the interior must now register.
    lower[1] = TRUTH[1] + 0.1

    result = evaluate_single_band(lower, upper, TRUTH, GRID)

    assert not result.covers_entirely
    assert result.violation_below


# ---------------------------------------------------------------------------
# The 1e-6 coverage tolerance: probe both sides rather than trust it
# ---------------------------------------------------------------------------
# Rationale (mirrors the metric's own docstring): float32 spacing near TPR=1 is
# ~6e-8, so crossings finer than ~1e-6 are representation noise, not statistical
# violations. We straddle the threshold well clear of float64 ambiguity.


def test_subtolerance_crossing_is_not_a_violation():
    lower = TRUTH - 0.1
    upper = TRUTH.copy()
    i = 5
    upper[i] = TRUTH[i] - 5e-7  # crossing below the 1e-6 floor

    result = evaluate_single_band(lower, upper, TRUTH, GRID)

    assert result.covers_entirely
    assert not result.violation_above


def test_supertolerance_crossing_is_a_violation():
    lower = TRUTH - 0.1
    upper = TRUTH.copy()
    i = 5
    upper[i] = TRUTH[i] - 2e-6  # crossing above the 1e-6 floor

    result = evaluate_single_band(lower, upper, TRUTH, GRID)

    assert not result.covers_entirely
    assert result.violation_above
    assert result.max_violation_above == pytest.approx(2e-6, abs=1e-12)


# ---------------------------------------------------------------------------
# band_area is the trapezoid integral of the width
# ---------------------------------------------------------------------------


def test_band_area_constant_width_equals_width():
    width = 0.3
    lower = np.zeros_like(GRID)
    upper = np.full_like(GRID, width)
    result = evaluate_single_band(lower, upper, TRUTH, GRID)
    # ∫_0^1 0.3 dFPR = 0.3, exact for a constant integrand.
    assert result.band_area == pytest.approx(width, abs=1e-12)


def test_band_area_triangular_width_equals_analytic_integral():
    # width(FPR) = FPR -> ∫_0^1 FPR dFPR = 1/2, exact for the trapezoid rule on
    # a linear integrand sampled on a uniform grid.
    lower = np.zeros_like(GRID)
    upper = GRID.copy()
    result = evaluate_single_band(lower, upper, TRUTH, GRID)
    assert result.band_area == pytest.approx(0.5, abs=1e-12)


# ---------------------------------------------------------------------------
# NaN handling: missing band values must not crash or manufacture violations
# ---------------------------------------------------------------------------


def test_nan_in_band_is_ignored_in_area_and_coverage():
    lower, upper = _covered_band()
    i = 5
    upper[i] = np.nan  # a missing band value at an interior point

    result = evaluate_single_band(lower, upper, TRUTH, GRID)

    # Area integrates over the finite points only and stays finite.
    valid = ~np.isnan(upper - lower)
    expected_area = np.trapezoid((upper - lower)[valid], GRID[valid])
    assert np.isfinite(result.band_area)
    assert result.band_area == pytest.approx(expected_area, abs=1e-12)
    # A NaN comparison is neither above nor below: the point is treated as covered.
    assert result.pointwise_covered[i]
    assert result.max_violation_above == 0.0
    assert result.max_violation_below == 0.0


# ---------------------------------------------------------------------------
# aggregate_band_results: coverage rate, CI, direction test, pointwise rates
# ---------------------------------------------------------------------------


def _make_results(n_cover: int, n_miss: int):
    """Build ``n_cover`` covering and ``n_miss`` below-missing band results."""
    results = []
    lower_ok, upper_ok = _covered_band()
    for _ in range(n_cover):
        results.append(evaluate_single_band(lower_ok, upper_ok, TRUTH, GRID))
    for _ in range(n_miss):
        lower = lower_ok.copy()
        lower[5] = TRUTH[5] + 0.05  # below-violation at one interior point
        results.append(evaluate_single_band(lower, upper_ok, TRUTH, GRID))
    return results


def test_coverage_rate_and_se_match_closed_form():
    n_cover, n_miss = 7, 3
    n = n_cover + n_miss
    evaluation = aggregate_band_results(_make_results(n_cover, n_miss), fpr_grid=GRID)

    p = n_cover / n
    assert evaluation.n_simulations == n
    assert evaluation.coverage_rate == pytest.approx(p, abs=1e-12)
    assert evaluation.coverage_se == pytest.approx(np.sqrt(p * (1 - p) / n), abs=1e-12)


def test_coverage_ci_matches_independent_wilson_recompute():
    n_cover, n_miss = 7, 3
    n = n_cover + n_miss
    evaluation = aggregate_band_results(_make_results(n_cover, n_miss), fpr_grid=GRID)

    p = n_cover / n
    z = stats.norm.ppf(0.975)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    assert evaluation.coverage_ci_lower == pytest.approx(center - margin, abs=1e-12)
    assert evaluation.coverage_ci_upper == pytest.approx(center + margin, abs=1e-12)


def test_pointwise_coverage_rate_is_column_mean():
    # One fully covering band and one that misses only at index 5 -> the column
    # mean at index 5 is 0.5 and 1.0 elsewhere over the interior.
    results = _make_results(n_cover=1, n_miss=1)
    evaluation = aggregate_band_results(results, fpr_grid=GRID)

    assert evaluation.pointwise_coverage_rates[5] == pytest.approx(0.5, abs=1e-12)
    interior = np.ones(len(GRID), dtype=bool)
    interior[[0, 5, -1]] = False
    assert np.allclose(evaluation.pointwise_coverage_rates[interior], 1.0, atol=1e-12)


def _directional_results(n_above: int, n_below: int):
    """Results carrying purely-above or purely-below violations."""
    lower_ok, upper_ok = _covered_band()
    results = []
    for _ in range(n_above):
        upper = upper_ok.copy()
        upper[5] = TRUTH[5] - 0.05  # true ROC above the upper band
        results.append(evaluate_single_band(lower_ok, upper, TRUTH, GRID))
    for _ in range(n_below):
        lower = lower_ok.copy()
        lower[5] = TRUTH[5] + 0.05  # lower band above the true ROC
        results.append(evaluate_single_band(lower, upper_ok, TRUTH, GRID))
    return results


def test_direction_test_pvalue_small_for_one_sided_violations():
    n_above = 8
    evaluation = aggregate_band_results(
        _directional_results(n_above=n_above, n_below=0), fpr_grid=GRID
    )
    expected = stats.binomtest(n_above, n_above, 0.5, alternative="two-sided").pvalue
    assert evaluation.direction_test_pvalue == pytest.approx(expected, abs=1e-12)
    assert evaluation.direction_test_pvalue < 0.05


def test_direction_test_pvalue_unity_for_balanced_violations():
    evaluation = aggregate_band_results(
        _directional_results(n_above=4, n_below=4), fpr_grid=GRID
    )
    expected = stats.binomtest(4, 8, 0.5, alternative="two-sided").pvalue
    assert evaluation.direction_test_pvalue == pytest.approx(expected, abs=1e-12)
    assert evaluation.direction_test_pvalue == pytest.approx(1.0, abs=1e-12)


def test_empty_results_raises():
    with pytest.raises(ValueError, match="No results"):
        aggregate_band_results([], fpr_grid=GRID)
