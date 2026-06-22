"""Tests for the studentized retention, the assembled envelope, and the suite.

Threat model: this is the band's interior machinery and final assembly. The
paper's informativeness claims ("tighter than KS", "adaptive width") depend on the
retention keeping exactly the right fraction of curves and the envelope being their
true pointwise min/max; the validity claims depend on the band being a monotone
ROC band pinned at the corners. The ablation chapter additionally relies on
``envelope_band_suite`` reproducing the canonical single-band function exactly.

Tolerances: the retention helpers operate on hand-built tensors and are checked
exactly. End-to-end bands round-trip through float32, so reconstructions and
equivalences use ``atol=1e-6``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from studroc_paper.methods.envelope_boot import (
    _ks_retention_envelope,
    _studentized_ks_statistics,
    envelope_band_suite,
    envelope_bootstrap_band,
    wilson_beta_band,
)
from studroc_paper.sampling import generate_bootstrap_grid


def _boot_for(y_true: np.ndarray, y_score: np.ndarray, grid: np.ndarray, seed: int = 0):
    """Reproducible bootstrap TPR matrix for given scores (CPU, fixed seed)."""
    torch.manual_seed(seed)
    matrix = generate_bootstrap_grid(
        y_true=torch.from_numpy(y_true),
        y_score=torch.from_numpy(y_score),
        B=200,
        grid=torch.from_numpy(grid),
        device=torch.device("cpu"),
        batch_size=500,
        tpr_method="empirical",
    )
    return matrix.cpu().numpy()


def _is_nondecreasing(x: np.ndarray, *, slack: float = 1e-6) -> bool:
    return bool(np.all(np.diff(x) >= -slack))


# ---------------------------------------------------------------------------
# Studentized KS statistic (per-curve sup deviation)
# ---------------------------------------------------------------------------


def test_studentized_ks_statistic_matches_manual_with_collapsed_column():
    # Column 3 has std below epsilon -> the collapsed-variance branch applies.
    deviations = torch.tensor([[0.2, -0.4, 0.1, 5e-7], [-0.1, 0.8, -0.3, 2e-3]])
    std_dev = torch.tensor([1.0, 2.0, 0.5, 1e-9])
    epsilon = 1e-6

    out = _studentized_ks_statistics(deviations, std_dev, epsilon)

    # Reconstruct the studentization rule: normal columns divide by std; the
    # collapsed column zeroes sub-epsilon deviations and divides the rest by eps.
    studentized = np.empty_like(deviations.numpy())
    for j, s in enumerate(std_dev.numpy()):
        col = deviations.numpy()[:, j]
        if s < epsilon:
            studentized[:, j] = np.where(np.abs(col) < epsilon, 0.0, col / epsilon)
        else:
            studentized[:, j] = col / s
    expected = np.abs(studentized).max(axis=1)
    np.testing.assert_allclose(out.numpy(), expected, atol=1e-6)


def test_ks_retention_keeps_ceil_fraction_and_correct_curves():
    # Ten curves with strictly ordered KS statistics; alpha=0.2 keeps ceil(8)=8.
    boot = torch.arange(50, dtype=torch.float32).reshape(10, 5) / 100.0
    ks = torch.tensor([9.0, 1, 8, 2, 7, 3, 6, 4, 5, 0])  # distinct, unsorted
    lower, upper = _ks_retention_envelope(boot, ks, alpha=0.2)

    kept = ks.numpy() <= np.sort(ks.numpy())[7]  # the 8 smallest
    assert kept.sum() == 8
    expected_lower = np.clip(boot.numpy()[kept].min(axis=0), 0, 1)
    expected_upper = np.clip(boot.numpy()[kept].max(axis=0), 0, 1)
    np.testing.assert_allclose(lower.numpy(), expected_lower, atol=1e-7)
    np.testing.assert_allclose(upper.numpy(), expected_upper, atol=1e-7)


# ---------------------------------------------------------------------------
# Envelope assembly (boundary_method="none")
# ---------------------------------------------------------------------------


def test_envelope_none_at_tiny_alpha_is_full_min_max(gaussian_scores, fpr_grid):
    # alpha small enough that ceil((1-alpha)*B) == B: every curve is retained,
    # so the band is exactly the clipped pointwise min/max of all bootstrap
    # curves, with the pinned corners. This pins steps 4-6 of the assembly.
    y_true, y_score = gaussian_scores
    boot = _boot_for(y_true, y_score, fpr_grid)
    alpha = 0.001  # ceil(0.999 * 200) = 200 = B

    _, lower, upper = envelope_bootstrap_band(
        boot_tpr_matrix=boot,
        fpr_grid=fpr_grid,
        y_true=y_true,
        y_score=y_score,
        alpha=alpha,
        boundary_method="none",
    )

    expected_lower = np.clip(boot.min(axis=0), 0, 1)
    expected_upper = np.clip(boot.max(axis=0), 0, 1)
    expected_lower[0] = 0.0
    expected_upper[-1] = 1.0
    np.testing.assert_allclose(lower, expected_lower, atol=1e-6)
    np.testing.assert_allclose(upper, expected_upper, atol=1e-6)


def test_identical_bootstrap_curves_collapse_the_band(gaussian_scores, fpr_grid):
    # Zero bootstrap variance -> the envelope collapses onto the single curve
    # in the interior rather than crashing on the studentization divide.
    y_true, y_score = gaussian_scores
    boot = _boot_for(y_true, y_score, fpr_grid)
    boot[:] = boot[0]

    _, lower, upper = envelope_bootstrap_band(
        boot_tpr_matrix=boot,
        fpr_grid=fpr_grid,
        y_true=y_true,
        y_score=y_score,
        alpha=0.05,
        boundary_method="none",
    )
    np.testing.assert_allclose(lower[1:-1], upper[1:-1], atol=1e-6)


# ---------------------------------------------------------------------------
# Structural contract across the configuration surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("boundary_method", ["none", "wilson", "ks"])
@pytest.mark.parametrize("retention_method", ["ks", "symmetric"])
@pytest.mark.parametrize("use_logit", [False, True])
@pytest.mark.parametrize("alpha", [0.05, 0.5])
def test_band_is_valid_monotone_and_pinned(
    gaussian_scores,
    fpr_grid,
    boundary_method,
    retention_method,
    use_logit,
    alpha,
    assert_valid_band,
):
    y_true, y_score = gaussian_scores
    boot = _boot_for(y_true, y_score, fpr_grid)

    fpr, lower, upper = envelope_bootstrap_band(
        boot_tpr_matrix=boot,
        fpr_grid=fpr_grid,
        y_true=y_true,
        y_score=y_score,
        alpha=alpha,
        boundary_method=boundary_method,
        retention_method=retention_method,
        use_logit=use_logit,
    )

    assert_valid_band(fpr, lower, upper, n_grid=len(fpr_grid))
    # The envelope is a genuine ROC band: monotone non-decreasing and pinned.
    assert _is_nondecreasing(lower)
    assert _is_nondecreasing(upper)
    assert lower[0] == 0.0
    assert upper[-1] == 1.0


def test_alpha_nesting_higher_confidence_contains_lower(gaussian_scores, fpr_grid):
    # A 95% band must contain the 50% band pointwise (more retention -> wider).
    y_true, y_score = gaussian_scores
    boot = _boot_for(y_true, y_score, fpr_grid)
    _, lo95, hi95 = envelope_bootstrap_band(
        boot_tpr_matrix=boot,
        fpr_grid=fpr_grid,
        y_true=y_true,
        y_score=y_score,
        alpha=0.05,
        boundary_method="none",
    )
    _, lo50, hi50 = envelope_bootstrap_band(
        boot_tpr_matrix=boot,
        fpr_grid=fpr_grid,
        y_true=y_true,
        y_score=y_score,
        alpha=0.50,
        boundary_method="none",
    )
    assert np.all(lo95 <= lo50 + 1e-6)
    assert np.all(hi95 >= hi50 - 1e-6)


@pytest.mark.parametrize(
    "transform",
    [np.exp, lambda x: 4.0 * x + 2.0, lambda x: x**3],
    ids=["exp", "affine", "cube"],
)
def test_full_band_is_distribution_free_under_monotone_transform(
    gaussian_scores, fpr_grid, transform
):
    # The decisive distribution-free check: with the bootstrap RNG fixed, a
    # strictly increasing reparametrization of the scores produces a byte-for-byte
    # identical band, because the entire pipeline (bootstrap TPRs, empirical ROC,
    # variances, floors) is a function of ranks alone.
    y_true, y_score = gaussian_scores
    boot_base = _boot_for(y_true, y_score, fpr_grid, seed=7)
    _, lo_base, hi_base = envelope_bootstrap_band(
        boot_tpr_matrix=boot_base,
        fpr_grid=fpr_grid,
        y_true=y_true,
        y_score=y_score,
        alpha=0.05,
        boundary_method="wilson",
    )

    y_score_t = transform(y_score)
    boot_t = _boot_for(y_true, y_score_t, fpr_grid, seed=7)
    _, lo_t, hi_t = envelope_bootstrap_band(
        boot_tpr_matrix=boot_t,
        fpr_grid=fpr_grid,
        y_true=y_true,
        y_score=y_score_t,
        alpha=0.05,
        boundary_method="wilson",
    )

    np.testing.assert_allclose(lo_t, lo_base, atol=1e-6)
    np.testing.assert_allclose(hi_t, hi_base, atol=1e-6)


# ---------------------------------------------------------------------------
# envelope_band_suite: routing, documented equivalences, floor attribution
# ---------------------------------------------------------------------------

_EXPECTED_VARIANTS = {
    "envelope",
    "envelope_no_beta_floor",
    "envelope_no_wilson_floor",
    "envelope_no_floors",
    "envelope_beta_both_tails",
    "envelope_wilson_both_tails",
}


def test_suite_returns_all_variants_for_each_alpha(gaussian_scores, fpr_grid):
    y_true, y_score = gaussian_scores
    boot = _boot_for(y_true, y_score, fpr_grid)
    alphas = [0.05, 0.5]
    suite = envelope_band_suite(
        boot_tpr_matrix=boot,
        fpr_grid=fpr_grid,
        y_true=y_true,
        y_score=y_score,
        alphas=alphas,
    )
    assert set(suite.keys()) == set(alphas)
    for alpha in alphas:
        assert set(suite[alpha].keys()) == _EXPECTED_VARIANTS


def test_suite_no_floors_matches_canonical_none_path(gaussian_scores, fpr_grid):
    # Documented equivalence: the shared-work suite reproduces the canonical
    # single-band function, so a refactor drift would be caught here.
    y_true, y_score = gaussian_scores
    boot = _boot_for(y_true, y_score, fpr_grid)
    suite = envelope_band_suite(
        boot_tpr_matrix=boot,
        fpr_grid=fpr_grid,
        y_true=y_true,
        y_score=y_score,
        alphas=[0.05],
    )
    lo_suite, hi_suite = suite[0.05]["envelope_no_floors"]
    _, lo_canon, hi_canon = envelope_bootstrap_band(
        boot_tpr_matrix=boot,
        fpr_grid=fpr_grid,
        y_true=y_true,
        y_score=y_score,
        alpha=0.05,
        boundary_method="none",
    )
    np.testing.assert_allclose(lo_suite, lo_canon, atol=1e-6)
    np.testing.assert_allclose(hi_suite, hi_canon, atol=1e-6)


def test_suite_envelope_matches_canonical_wilson_path(gaussian_scores, fpr_grid):
    y_true, y_score = gaussian_scores
    boot = _boot_for(y_true, y_score, fpr_grid)
    suite = envelope_band_suite(
        boot_tpr_matrix=boot,
        fpr_grid=fpr_grid,
        y_true=y_true,
        y_score=y_score,
        alphas=[0.05],
    )
    lo_suite, hi_suite = suite[0.05]["envelope"]
    _, lo_canon, hi_canon = envelope_bootstrap_band(
        boot_tpr_matrix=boot,
        fpr_grid=fpr_grid,
        y_true=y_true,
        y_score=y_score,
        alpha=0.05,
        boundary_method="wilson",
    )
    np.testing.assert_allclose(lo_suite, lo_canon, atol=1e-6)
    np.testing.assert_allclose(hi_suite, hi_canon, atol=1e-6)


def test_floors_only_lower_the_lower_band_and_beta_owns_low_fpr(
    gaussian_scores, fpr_grid
):
    # Floors are pointwise minima on the lower band: the full envelope's lower
    # bound never exceeds the no-floors lower bound. The Beta floor's effect is
    # confined to (and present in) the low-FPR corner.
    y_true, y_score = gaussian_scores
    boot = _boot_for(y_true, y_score, fpr_grid)
    suite = envelope_band_suite(
        boot_tpr_matrix=boot,
        fpr_grid=fpr_grid,
        y_true=y_true,
        y_score=y_score,
        alphas=[0.05],
    )[0.05]

    lo_full = suite["envelope"][0]
    lo_no_floors = suite["envelope_no_floors"][0]
    lo_no_beta = suite["envelope_no_beta_floor"][0]

    # Adding floors can only pull the lower band down (or leave it).
    assert np.all(lo_full <= lo_no_floors + 1e-6)
    # Dropping the Beta floor changes only the low-FPR region; the two agree in
    # the interior and differ somewhere within the Beta jurisdiction.
    low_fpr = fpr_grid <= 0.2
    interior = fpr_grid > 0.2
    assert np.allclose(lo_full[interior], lo_no_beta[interior], atol=1e-6)
    assert np.any(np.abs(lo_full[low_fpr] - lo_no_beta[low_fpr]) > 1e-6)


# ---------------------------------------------------------------------------
# wilson_beta_band (no-bootstrap ablation)
# ---------------------------------------------------------------------------


def test_wilson_beta_band_is_valid_monotone_and_pinned(
    gaussian_scores, assert_valid_band
):
    y_true, y_score = gaussian_scores
    fpr, lower, upper = wilson_beta_band(y_true, y_score, alpha=0.05)
    assert_valid_band(fpr, lower, upper, n_grid=len(fpr))
    assert _is_nondecreasing(lower)
    assert _is_nondecreasing(upper)
    assert lower[0] == 0.0
    assert upper[-1] == 1.0
