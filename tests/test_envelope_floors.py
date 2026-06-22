"""Tests for the two exact tail floors and the Wilson variance-ratio gate.

Threat model: the floors are the paper's *exact, finite-sample, distribution-free*
guarantees at the two corners where the bootstrap and asymptotic arguments break
down. The README's central repair claim ("high-AUC coverage 0.77 -> 0.95") and the
"each floor owns its corner" ablation both rest on these functions computing the
right order-statistic law and mapping it onto the FPR grid correctly.

Strategy:
  * Reconstruct each floor's construction independently from the documented math
    (Beta quantiles, the Wilson score formula, the searchsorted jurisdiction map)
    and assert the implementation matches.
  * Pin the *distribution-free* property directly: a strictly increasing transform
    of the scores leaves the floor unchanged, because every quantity it uses is a
    rank statistic. This is a property the reconstruction cannot fake.
  * Pin the honest-vacuity boundary: below the first order statistic's reach the
    lower floor is 0 (no certifiable bound), exactly as the paper claims.

Tolerances: the floors round-trip through float32 torch tensors, so reconstructed
values are compared at ``atol=1e-6`` (>> one float32 ulp, << any band width). The
Šidák scalar is pure-Python float and checked at ``atol=1e-12``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import beta as beta_dist
from scipy.stats import norm

from studroc_paper.methods.envelope_boot import (
    _apply_beta_orderstat_floor,
    _apply_beta_orderstat_floor_upper_tail,
    _apply_wilson_variance_ratio_floor,
    _compute_variance_ratio_alpha,
    _wilson_lower_one_sided,
    _wilson_upper_one_sided,
)

ALPHA = 0.05
J_MAX = 25


@pytest.fixture
def floor_data(rng):
    """Continuous, tie-free two-class scores plus the standard FPR grid.

    Tensors are float32 (the production dtype); the matching float64 arrays are
    read back *from* the tensors so reconstruction sees byte-identical inputs.
    """
    n_neg = n_pos = 150
    neg = rng.normal(0.0, 1.0, n_neg)
    pos = rng.normal(1.4, 1.0, n_pos)
    y_true = torch.tensor([1] * n_pos + [0] * n_neg, dtype=torch.int64)
    y_score = torch.tensor(np.concatenate([pos, neg]), dtype=torch.float32)
    fpr = torch.linspace(0.0, 1.0, 101, dtype=torch.float32)
    score_np = y_score.numpy().astype(np.float64)
    neg_np = score_np[150:]
    pos_np = score_np[:150]
    return {
        "fpr": fpr,
        "y_true": y_true,
        "y_score": y_score,
        "fpr_np": fpr.numpy().astype(np.float64),
        "neg": neg_np,
        "pos": pos_np,
    }


# ---------------------------------------------------------------------------
# Wilson score one-sided bounds (the floors' building block)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p_hat", [0.0, 0.1, 0.5, 0.83, 1.0])
def test_wilson_one_sided_bounds_match_closed_form(p_hat):
    n, alpha = 80, 0.01
    z = float(norm.ppf(1 - alpha))
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))

    lo = _wilson_lower_one_sided(np.array([p_hat]), n, alpha)[0]
    hi = _wilson_upper_one_sided(np.array([p_hat]), n, alpha)[0]

    assert lo == pytest.approx(np.clip(center - half, 0, 1), abs=1e-12)
    assert hi == pytest.approx(np.clip(center + half, 0, 1), abs=1e-12)
    # Bracketing and ordering must always hold.
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_lower_is_nondegenerate_at_p_equals_one():
    # At p=1 the binomial-variance term vanishes but the Wilson floor must still
    # leave a strictly positive gap below 1 (this nondegeneracy is *why* it is
    # used as a floor where the bootstrap collapses).
    lo = _wilson_lower_one_sided(np.array([1.0]), 100, 0.025)[0]
    assert 0.0 < lo < 1.0


# ---------------------------------------------------------------------------
# Beta order-statistic floor (low-FPR corner)
# ---------------------------------------------------------------------------


def _expected_low_floor(fpr_np, prior_np, neg, pos, alpha=ALPHA, j_max=J_MAX):
    """Independent reconstruction of the low-FPR Beta floor from documented math."""
    n_neg, n_pos = len(neg), len(pos)
    j_used = min(j_max, n_neg)
    alpha_event = alpha / (2 * j_max)
    js = np.arange(1, j_used + 1)
    q_j = beta_dist.ppf(1.0 - alpha_event, js, n_neg + 1 - js)
    neg_desc = np.sort(neg)[::-1]
    tpr_hat = (pos[None, :] > neg_desc[:j_used, None]).mean(axis=1)
    bounds = np.concatenate(
        ([0.0], _wilson_lower_one_sided(tpr_hat, n_pos, alpha_event))
    )
    floor = np.full_like(fpr_np, np.inf)
    zone = (fpr_np > 0.0) & (fpr_np <= q_j[-1])
    j_star = np.searchsorted(q_j, fpr_np[zone], side="right")
    floor[zone] = bounds[j_star]
    return np.minimum(prior_np, floor), q_j


def test_low_floor_matches_independent_reconstruction(floor_data):
    prior = torch.ones_like(floor_data["fpr"])
    out = _apply_beta_orderstat_floor(
        fpr_grid=floor_data["fpr"],
        lower_envelope=prior.clone(),
        y_true=floor_data["y_true"],
        y_score=floor_data["y_score"],
        alpha=ALPHA,
    )
    expected, _ = _expected_low_floor(
        floor_data["fpr_np"],
        prior.numpy().astype(np.float64),
        floor_data["neg"],
        floor_data["pos"],
    )
    np.testing.assert_allclose(out.numpy(), expected, atol=1e-6)


def test_low_floor_is_honestly_vacuous_below_first_order_statistic(floor_data):
    # Below q_1 no negative order statistic certifies a bound: the floor is 0.
    prior = torch.ones_like(floor_data["fpr"])
    out = _apply_beta_orderstat_floor(
        fpr_grid=floor_data["fpr"],
        lower_envelope=prior.clone(),
        y_true=floor_data["y_true"],
        y_score=floor_data["y_score"],
        alpha=ALPHA,
    ).numpy()

    n_neg = len(floor_data["neg"])
    q1 = beta_dist.ppf(1.0 - ALPHA / (2 * J_MAX), 1, n_neg)
    fpr = floor_data["fpr_np"]
    below = (fpr > 0.0) & (fpr < q1)
    # Vacuous zone is pulled to 0 (lower bound says nothing), not left at the prior.
    assert np.all(out[below] == 0.0)
    # And q_1 ~ 7/n_neg, matching the README's "FPR ~ 7/n0" honesty boundary.
    assert q1 == pytest.approx(7.0 / n_neg, rel=0.5)
    # Somewhere above q_1 the floor certifies a positive lower bound.
    assert np.any(out[fpr >= q1] > 0.0)


def test_low_floor_only_lowers_never_raises(floor_data):
    # The floor is a pointwise minimum: an already-low lower envelope is untouched
    # (it is already coverage-safe), so a zero prior stays zero everywhere.
    prior = torch.zeros_like(floor_data["fpr"])
    out = _apply_beta_orderstat_floor(
        fpr_grid=floor_data["fpr"],
        lower_envelope=prior.clone(),
        y_true=floor_data["y_true"],
        y_score=floor_data["y_score"],
        alpha=ALPHA,
    ).numpy()
    assert np.all(out == 0.0)


@pytest.mark.parametrize(
    "transform",
    [np.exp, lambda x: 3.0 * x + 1.0, lambda x: x**3],
    ids=["exp", "affine", "cube"],
)
def test_low_floor_is_invariant_to_monotone_score_transform(floor_data, transform):
    # The distribution-free claim, tested directly: every quantity the floor uses
    # is a rank statistic, so a strictly increasing reparametrization of the
    # scores must leave the floor identical.
    prior = torch.ones_like(floor_data["fpr"])
    base = _apply_beta_orderstat_floor(
        fpr_grid=floor_data["fpr"],
        lower_envelope=prior.clone(),
        y_true=floor_data["y_true"],
        y_score=floor_data["y_score"],
        alpha=ALPHA,
    ).numpy()

    transformed = transform(floor_data["y_score"].numpy().astype(np.float64))
    out = _apply_beta_orderstat_floor(
        fpr_grid=floor_data["fpr"],
        lower_envelope=prior.clone(),
        y_true=floor_data["y_true"],
        y_score=torch.tensor(transformed, dtype=torch.float32),
        alpha=ALPHA,
    ).numpy()
    np.testing.assert_allclose(out, base, atol=1e-6)


def test_low_floor_no_op_when_a_class_is_empty(floor_data):
    prior = torch.ones_like(floor_data["fpr"]) * 0.5
    all_pos = torch.ones_like(floor_data["y_true"])
    out = _apply_beta_orderstat_floor(
        fpr_grid=floor_data["fpr"],
        lower_envelope=prior.clone(),
        y_true=all_pos,  # no negatives -> floor cannot anchor
        y_score=floor_data["y_score"],
        alpha=ALPHA,
    )
    assert torch.equal(out, prior)


# ---------------------------------------------------------------------------
# Beta order-statistic floor (high-FPR / plateau corner, mirrored)
# ---------------------------------------------------------------------------


def _expected_upper_tail_floor(fpr_np, prior_np, neg, pos, alpha=ALPHA, j_max=J_MAX):
    """Independent reconstruction of the high-FPR Beta floor."""
    n_neg, n_pos = len(neg), len(pos)
    j_used = min(j_max, n_pos)
    alpha_event = alpha / (2 * j_max)
    js = np.arange(1, j_used + 1)
    rho_j = beta_dist.ppf(1.0 - alpha_event, js, n_pos + 1 - js)
    tpr_bounds = 1.0 - rho_j
    pos_asc = np.sort(pos)
    fpr_hat = (neg[None, :] > pos_asc[:j_used, None]).mean(axis=1)
    f_j = _wilson_upper_one_sided(fpr_hat, n_neg, alpha_event)
    floor = np.full_like(fpr_np, np.inf)
    zone = (fpr_np >= f_j[-1]) & (fpr_np < 1.0)
    f_ascending = f_j[::-1].copy()
    count_qual = np.searchsorted(f_ascending, fpr_np[zone], side="right")
    j_star_idx = j_used - count_qual
    floor[zone] = tpr_bounds[j_star_idx]
    return np.minimum(prior_np, floor), f_j


def test_upper_tail_floor_matches_independent_reconstruction(floor_data):
    prior = torch.ones_like(floor_data["fpr"])
    out = _apply_beta_orderstat_floor_upper_tail(
        fpr_grid=floor_data["fpr"],
        lower_envelope=prior.clone(),
        y_true=floor_data["y_true"],
        y_score=floor_data["y_score"],
        alpha=ALPHA,
    )
    expected, _ = _expected_upper_tail_floor(
        floor_data["fpr_np"],
        prior.numpy().astype(np.float64),
        floor_data["neg"],
        floor_data["pos"],
    )
    np.testing.assert_allclose(out.numpy(), expected, atol=1e-6)


def test_upper_tail_floor_unchanged_below_its_jurisdiction(floor_data):
    prior = torch.ones_like(floor_data["fpr"])
    out = _apply_beta_orderstat_floor_upper_tail(
        fpr_grid=floor_data["fpr"],
        lower_envelope=prior.clone(),
        y_true=floor_data["y_true"],
        y_score=floor_data["y_score"],
        alpha=ALPHA,
    ).numpy()
    _, f_j = _expected_upper_tail_floor(
        floor_data["fpr_np"],
        prior.numpy().astype(np.float64),
        floor_data["neg"],
        floor_data["pos"],
    )
    below_zone = floor_data["fpr_np"] < f_j[-1]
    assert np.all(out[below_zone] == 1.0)


@pytest.mark.parametrize(
    "transform",
    [np.exp, lambda x: 2.0 * x - 3.0, lambda x: x**3],
    ids=["exp", "affine", "cube"],
)
def test_upper_tail_floor_invariant_to_monotone_transform(floor_data, transform):
    prior = torch.ones_like(floor_data["fpr"])
    base = _apply_beta_orderstat_floor_upper_tail(
        fpr_grid=floor_data["fpr"],
        lower_envelope=prior.clone(),
        y_true=floor_data["y_true"],
        y_score=floor_data["y_score"],
        alpha=ALPHA,
    ).numpy()
    transformed = transform(floor_data["y_score"].numpy().astype(np.float64))
    out = _apply_beta_orderstat_floor_upper_tail(
        fpr_grid=floor_data["fpr"],
        lower_envelope=prior.clone(),
        y_true=floor_data["y_true"],
        y_score=torch.tensor(transformed, dtype=torch.float32),
        alpha=ALPHA,
    ).numpy()
    np.testing.assert_allclose(out, base, atol=1e-6)


# ---------------------------------------------------------------------------
# Wilson variance-ratio gate (the plateau floor's trigger and Šidák strength)
# ---------------------------------------------------------------------------


def test_variance_ratio_deficiency_and_sidak_are_exact():
    # r = [2, 0.5, 0, 1] -> deficiency = max(0, 1-r) = [0, 0.5, 1, 0], K_eff = 1.5.
    wilson_var = torch.ones(4)
    bootstrap_var = torch.tensor([2.0, 0.5, 0.0, 1.0])
    deficiency, alpha_wilson = _compute_variance_ratio_alpha(
        bootstrap_var, wilson_var, ALPHA
    )
    np.testing.assert_allclose(deficiency.numpy(), [0.0, 0.5, 1.0, 0.0], atol=1e-7)
    k_eff = 1.5
    assert alpha_wilson == pytest.approx(1 - (1 - ALPHA) ** (1 / k_eff), abs=1e-12)


def test_variance_ratio_no_sidak_when_k_eff_below_one():
    # A single half-deficient point (K_eff = 0.5 <= 1) gets no Šidák inflation.
    wilson_var = torch.ones(3)
    bootstrap_var = torch.tensor([1.0, 0.5, 2.0])  # deficiency = [0, 0.5, 0]
    _, alpha_wilson = _compute_variance_ratio_alpha(bootstrap_var, wilson_var, ALPHA)
    assert alpha_wilson == pytest.approx(ALPHA, abs=1e-12)


def test_wilson_floor_is_identity_when_no_deficiency(floor_data):
    lower = torch.linspace(0.0, 0.9, 101)
    upper = torch.linspace(0.1, 1.0, 101)
    out_lo, out_hi = _apply_wilson_variance_ratio_floor(
        floor_data["fpr"],
        lower.clone(),
        upper.clone(),
        floor_data["y_true"],
        floor_data["y_score"],
        deficiency=torch.zeros(101),
        alpha_wilson=ALPHA,
    )
    assert torch.equal(out_lo, lower)
    assert torch.equal(out_hi, upper)


def test_wilson_floor_only_widens_and_stays_monotone(floor_data):
    # A deliberately non-monotone, tight band with deficiency on the plateau.
    rng = np.random.default_rng(1)
    lower = torch.tensor(rng.uniform(0.3, 0.6, 101), dtype=torch.float32)
    upper = lower + 0.02
    deficiency = torch.zeros(101)
    deficiency[80:] = 1.0  # plateau collapse

    out_lo, out_hi = _apply_wilson_variance_ratio_floor(
        floor_data["fpr"],
        lower.clone(),
        upper.clone(),
        floor_data["y_true"],
        floor_data["y_score"],
        deficiency=deficiency,
        alpha_wilson=ALPHA,
    )
    out_lo_np, out_hi_np = out_lo.numpy(), out_hi.numpy()

    # The floor + monotonicity enforcement can only widen the band.
    assert np.all(out_lo_np <= lower.numpy() + 1e-6)
    assert np.all(out_hi_np >= upper.numpy() - 1e-6)
    # Output bands are monotone non-decreasing in FPR (the ROC-band contract).
    assert np.all(np.diff(out_lo_np) >= -1e-6)
    assert np.all(np.diff(out_hi_np) >= -1e-6)
    # The deficient plateau region is genuinely floored wider than the input.
    assert np.any(out_hi_np[80:] - out_lo_np[80:] > 0.02 + 1e-4)
