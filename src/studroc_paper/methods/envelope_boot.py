"""Envelope Bootstrap Confidence Bands using PyTorch.

This module provides methods for constructing simultaneous confidence bands
for ROC curves using studentized bootstrap envelope techniques. It implements
multiple boundary correction methods and curve retention strategies, with
optional GPU acceleration via PyTorch.

The main function, envelope_bootstrap_band, computes confidence bands by:
1. Studentizing bootstrap ROC curves relative to the empirical ROC
2. Selecting curves based on their consistency with the empirical ROC
3. Taking the pointwise envelope of retained curves
4. Applying boundary corrections where bootstrap variance collapses

Key features:
- Studentized bootstrap for improved finite-sample coverage
- Adaptive tail floor using Wilson Rectangle bounds with Šidák correction
- Exact Beta order-statistic floor carrying threshold-location uncertainty
  at extreme FPR
- KS-style boundary extension option
- Logit-space construction option for variance stabilization
- GPU acceleration for large bootstrap samples
- Diagnostic visualization integration
"""

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import beta as beta_dist
from scipy.stats import norm
from torch import Tensor

from studroc_paper.viz import plot_band_diagnostics

from .method_utils import (
    compute_empirical_roc_from_scores,
    compute_empirical_roc_from_scores_hd,
    numpy_to_torch,
    torch_to_numpy,
    wilson_halfwidth_squared_torch,
)
from .wilson_band import wilson_rectangle_band

# Type alias for boundary extension method selection
BoundaryMethod = Literal["none", "wilson", "ks"]

# Type alias for curve retention method selection
RetentionMethod = Literal["ks", "symmetric"]

# Type alias for TPR estimation method
TprMethod = Literal["empirical", "harrell_davis"]


def _compute_empirical_roc(
    y_true: Tensor, y_score: Tensor, fpr_grid: Tensor, method: TprMethod = "empirical"
) -> Tensor:
    """Compute empirical ROC curve and interpolate at fpr_grid points.

    Args:
        y_true: Tensor of true binary labels (0 or 1).
        y_score: Tensor of predicted scores.
        fpr_grid: FPR values at which to evaluate TPR.
        method: TPR estimation method. "empirical" uses standard step-function
            interpolation; "harrell_davis" uses beta-weighted quantile estimation
            for reduced finite-sample bias. Defaults to "empirical".

    Returns:
        TPR values at fpr_grid points.

    Examples:
        >>> y_true = torch.tensor([0, 0, 1, 1])
        >>> y_score = torch.tensor([0.1, 0.4, 0.35, 0.8])
        >>> fpr_grid = torch.linspace(0, 1, 11)
        >>> tpr = _compute_empirical_roc(y_true, y_score, fpr_grid)
        >>> tpr.shape
        torch.Size([11])
    """
    # Separate scores by class
    neg_scores = y_score[y_true == 0]
    pos_scores = y_score[y_true == 1]

    if method == "harrell_davis":
        return compute_empirical_roc_from_scores_hd(neg_scores, pos_scores, fpr_grid)
    return compute_empirical_roc_from_scores(neg_scores, pos_scores, fpr_grid)


def _haldane_logit(tpr: Tensor, n_pos: int) -> Tensor:
    """Apply Logit transform with Haldane-Anscombe correction (+0.5).

    The Haldane-Anscombe correction adds 0.5 to both the numerator and
    denominator before computing the logit, preventing infinities at the
    boundaries (TPR = 0 or TPR = 1).

    Args:
        tpr: Tensor of true positive rates in [0, 1].
        n_pos: Number of positive samples.

    Returns:
        Logit-transformed TPR values with Haldane-Anscombe correction.

    Examples:
        >>> tpr = torch.tensor([0.0, 0.5, 1.0])
        >>> _haldane_logit(tpr, n_pos=100)
        tensor([-5.2983,  0.0000,  5.2983])
    """
    k = tpr * n_pos
    return torch.log((k + 0.5) / (n_pos - k + 0.5))


def _extend_boundary_ks_style(
    fpr_grid: Tensor,
    lower_envelope: Tensor,
    upper_envelope: Tensor,
    empirical_tpr: Tensor,
    n_neg: int,
    n_pos: int,
    alpha: float,
) -> tuple[Tensor, Tensor]:
    """Extend confidence band at boundaries using KS-style margins.

    At the corners of ROC space where bootstrap variance collapses to zero,
    extend the confidence band using the same horizontal and vertical margins
    used in the fixed-width KS band (Campbell 1994).

    This ensures the band connects smoothly from the interior (where bootstrap
    provides genuine variance) to the corners (0,0) and (1,1), with statistical
    margins based on sample sizes.

    Args:
        fpr_grid: FPR values at which the envelope is evaluated.
        lower_envelope: Lower bound of the confidence band.
        upper_envelope: Upper bound of the confidence band.
        empirical_tpr: Empirical TPR values at each grid point.
        n_neg: Number of negative samples.
        n_pos: Number of positive samples.
        alpha: Significance level.

    Returns:
        Tuple of (extended_lower, extended_upper) envelopes.

    Examples:
        >>> fpr_grid = torch.linspace(0, 1, 101)
        >>> lower = torch.zeros(101)
        >>> upper = torch.ones(101)
        >>> empirical_tpr = fpr_grid  # Perfect diagonal
        >>> lower_ext, upper_ext = _extend_boundary_ks_style(
        ...     fpr_grid, lower, upper, empirical_tpr, n_neg=100, n_pos=100, alpha=0.05
        ... )
        >>> lower_ext.shape
        torch.Size([101])
    """
    # KS critical value (Smirnov approximation)
    # For two one-sided tests combined: alpha_adj = 1 - sqrt(1 - alpha)
    alpha_adj = 1.0 - math.sqrt(1.0 - alpha)
    c_alpha = math.sqrt(-0.5 * math.log(alpha_adj / 2))

    # Effective sample size for two-sample KS test
    n_eff = (n_pos * n_neg) / (n_pos + n_neg)
    d = c_alpha / math.sqrt(n_eff)  # Vertical margin for TPR

    # Find where the band is degenerate (bootstrap variance ~= 0)
    band_width = upper_envelope - lower_envelope
    meaningful_width = 1e-6

    # Clone to avoid in-place modification issues
    lower_ext = lower_envelope.clone()
    upper_ext = upper_envelope.clone()

    # === Lower bound extension at the start (near FPR=0) ===
    # Find first grid point with meaningful band width
    meaningful_mask = band_width > meaningful_width
    if meaningful_mask.any():
        first_meaningful_idx = meaningful_mask.nonzero(as_tuple=True)[0][0].item()

        if first_meaningful_idx > 0:
            # Anchor point: first meaningful point, extended down by d
            fpr_anchor = fpr_grid[first_meaningful_idx]
            lower_anchor = max(lower_envelope[first_meaningful_idx].item() - d, 0.0)

            # Linear interpolation from (0,0) to anchor
            for i in range(first_meaningful_idx):
                if fpr_anchor > 0:
                    t = fpr_grid[i].item() / fpr_anchor.item()
                else:
                    t = 0.0
                lower_ext[i] = t * lower_anchor

    # === Upper bound extension at the end (near FPR=1) ===
    # Find last grid point with meaningful band width
    if meaningful_mask.any():
        last_meaningful_idx = (
            len(fpr_grid)
            - 1
            - meaningful_mask.flip(0).nonzero(as_tuple=True)[0][0].item()
        )

        if last_meaningful_idx < len(fpr_grid) - 1:
            # Anchor point: last meaningful point, extended up by d
            fpr_anchor = fpr_grid[last_meaningful_idx]
            upper_anchor = min(upper_envelope[last_meaningful_idx].item() + d, 1.0)

            # Linear interpolation from anchor to (1,1)
            fpr_remaining = 1.0 - fpr_anchor.item()
            for i in range(last_meaningful_idx + 1, len(fpr_grid)):
                if fpr_remaining > 0:
                    t = (fpr_grid[i].item() - fpr_anchor.item()) / fpr_remaining
                else:
                    t = 1.0
                upper_ext[i] = upper_anchor + t * (1.0 - upper_anchor)

    # Ensure bounds stay valid
    lower_ext = torch.clamp(lower_ext, 0.0, 1.0)
    upper_ext = torch.clamp(upper_ext, 0.0, 1.0)

    return lower_ext, upper_ext


def _compute_variance_ratio_alpha(
    bootstrap_var: Tensor,
    wilson_var: Tensor,
    alpha: float,
) -> tuple[Tensor, float]:
    """Compute variance ratio and Šidák-corrected alpha for Wilson floor.

    Uses the ratio r(t) = bootstrap_var(t) / wilson_var(t) to determine where
    the bootstrap has collapsed and how strong the Šidák correction should be.
    Points with r < 1 are "Wilson-dependent" -- the bootstrap underestimates
    even the binomial variance component.

    The effective dimensionality K_eff = sum max(0, 1 - r(t)) is a continuous
    measure of how many grid points need Wilson correction. A fully collapsed
    point (r=0) contributes 1; a partially collapsed point (r=0.5) contributes
    0.5; a healthy point (r>=1) contributes 0.

    Args:
        bootstrap_var: Bootstrap variance at each grid point.
        wilson_var: Wilson score variance at each grid point.
        alpha: Significance level.

    Returns:
        Tuple of (deficiency_weights, alpha_wilson):
        - deficiency_weights: Tensor of max(0, 1 - r(t)) values at each grid point.
          Positive where the bootstrap is deficient, zero where healthy.
        - alpha_wilson: Šidák-corrected alpha based on K_eff.
    """
    # Avoid division by zero where Wilson variance is zero (shouldn't happen,
    # but guard against it)
    safe_wilson = torch.clamp(wilson_var, min=1e-30)
    r = bootstrap_var / safe_wilson

    # Deficiency weight: 1 where fully collapsed, 0 where healthy
    deficiency = torch.clamp(1.0 - r, min=0.0)

    # Effective dimensionality of Wilson-dependent region
    k_eff = float(deficiency.sum().item())

    # Šidák correction: alpha_wilson = 1 - (1 - alpha)^(1/K_eff)
    if k_eff > 1.0:
        alpha_wilson = 1.0 - (1.0 - alpha) ** (1.0 / k_eff)
    else:
        alpha_wilson = alpha

    return deficiency, alpha_wilson


def _apply_wilson_variance_ratio_floor(
    fpr_grid: Tensor,
    lower_envelope: Tensor,
    upper_envelope: Tensor,
    y_true: Tensor,
    y_score: Tensor,
    deficiency: Tensor,
    alpha_wilson: float,
) -> tuple[Tensor, Tensor]:
    """Apply Wilson Rectangle floor where bootstrap variance is deficient.

    Uses the variance ratio deficiency weights to determine where to apply
    Wilson Rectangle bounds. The floor is applied only at points where the
    bootstrap variance is below the Wilson variance (deficiency > 0).

    Args:
        fpr_grid: FPR grid values.
        lower_envelope: Current lower envelope bound.
        upper_envelope: Current upper envelope bound.
        y_true: True binary labels.
        y_score: Predicted scores.
        deficiency: Variance ratio deficiency weights from
            _compute_variance_ratio_alpha. Positive where Wilson is needed.
        alpha_wilson: Šidák-corrected alpha level for Wilson bounds.

    Returns:
        Tuple of (lower_envelope, upper_envelope) with Wilson floor applied.
    """
    needs_floor = deficiency > 0
    if not needs_floor.any():
        return lower_envelope, upper_envelope

    # Compute Wilson Rectangle bounds at the corrected alpha
    y_true_np = torch_to_numpy(y_true)
    y_score_np = torch_to_numpy(y_score)
    fpr_np = torch_to_numpy(fpr_grid)

    _, wilson_lower_np, wilson_upper_np = wilson_rectangle_band(
        y_true=y_true_np,
        y_score=y_score_np,
        k=len(fpr_np),
        alpha=alpha_wilson,
        correction="sidak",
        tpr_method="empirical",
    )

    device = fpr_grid.device
    wilson_lower = numpy_to_torch(wilson_lower_np, device).float()
    wilson_upper = numpy_to_torch(wilson_upper_np, device).float()

    # Apply Wilson floor where bootstrap is deficient
    lower_result = lower_envelope.clone()
    upper_result = upper_envelope.clone()

    lower_result[needs_floor] = torch.minimum(
        lower_result[needs_floor], wilson_lower[needs_floor]
    )
    upper_result[needs_floor] = torch.maximum(
        upper_result[needs_floor], wilson_upper[needs_floor]
    )

    # Enforce monotonicity: ROC bands must be non-decreasing.
    # cummax propagates the highest upper bound seen so far (left to right).
    # Flipped cummin propagates the lowest lower bound seen so far (right to left).
    lower_flipped = torch.flip(lower_result, dims=[0])
    lower_cummin, _ = torch.cummin(lower_flipped, dim=0)
    lower_result = torch.flip(lower_cummin, dims=[0])

    upper_result, _ = torch.cummax(upper_result, dim=0)

    return lower_result, upper_result


def _wilson_lower_one_sided(p_hat: NDArray, n: int, alpha: float) -> NDArray:
    """Compute the one-sided Wilson score lower bound at level alpha.

    Args:
        p_hat: Observed proportions.
        n: Number of trials.
        alpha: One-sided significance level.

    Returns:
        Lower confidence bounds, clipped to [0, 1].
    """
    z = float(norm.ppf(1.0 - alpha))
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    halfwidth = (z / denom) * np.sqrt(
        p_hat * (1.0 - p_hat) / n + z * z / (4 * n * n)
    )
    return np.clip(center - halfwidth, 0.0, 1.0)


def _apply_beta_orderstat_floor(
    *,
    fpr_grid: Tensor,
    lower_envelope: Tensor,
    y_true: Tensor,
    y_score: Tensor,
    alpha: float,
    j_max: int = 25,
) -> Tensor:
    """Apply the exact Beta order-statistic floor to the lower envelope.

    At extreme FPR the dominant uncertainty is horizontal -- the true FPR of
    the threshold at the j-th largest negative score -- not the binomial TPR
    uncertainty that variance-based corrections measure. For continuous
    scores that true FPR follows Beta(j, n_neg + 1 - j) exactly (probability
    integral transform), independent of the score distribution.

    Let q_j be the (1 - alpha_event) upper quantile of that law. On the
    event {true FPR of the j-th largest negative <= q_j}, monotonicity of
    the true ROC gives, for every evaluation point t >= q_j:

        R_true(t) >= R_true at that threshold >= WilsonLower(TPR_hat there)

    The floor at t is therefore the one-sided Wilson lower bound of the
    empirical TPR at the largest j with q_j <= t -- a backward-looking bound
    anchored at a smaller-FPR operating point whose true FPR provably (with
    high probability) sits at or below t. For t < q_1 no order statistic
    qualifies and the floor is vacuous (zero): no distribution-free lower
    bound exists there. The alpha budget is split Bonferroni-style across
    the 2 * j_max one-sided events (j_max Beta quantile events plus j_max
    Wilson bounds). With ties the true exceedance is stochastically smaller
    than the Beta law, so discrete scores err conservative.

    Args:
        fpr_grid: FPR grid values.
        lower_envelope: Current lower envelope bound.
        y_true: True binary labels.
        y_score: Predicted scores.
        alpha: Total alpha budget for the floor.
        j_max: Number of order statistics used. The floor's jurisdiction
            ends at the Beta upper quantile of the j_max-th order statistic
            (roughly 1.7 * j_max / n_neg grid points).

    Returns:
        Lower envelope with the floor applied (pointwise minimum within the
        floor's jurisdiction, unchanged elsewhere).
    """
    neg_scores = torch_to_numpy(y_score[y_true == 0]).astype(np.float64)
    pos_scores = torch_to_numpy(y_score[y_true == 1]).astype(np.float64)
    n_neg = len(neg_scores)
    n_pos = len(pos_scores)
    j_used = min(j_max, n_neg)
    if j_used == 0 or n_pos == 0:
        return lower_envelope

    alpha_event = alpha / (2 * j_max)
    js = np.arange(1, j_used + 1)
    q_j = beta_dist.ppf(1.0 - alpha_event, js, n_neg + 1 - js)

    # Wilson-lowered empirical TPR at each of the j largest negatives
    neg_desc = np.sort(neg_scores)[::-1]
    tpr_hat = (pos_scores[None, :] > neg_desc[:j_used, None]).mean(axis=1)
    bounds = np.concatenate(
        ([0.0], _wilson_lower_one_sided(tpr_hat, n_pos, alpha_event))
    )

    # Floor value at t: bound from the largest j with q_j <= t; +inf outside
    # the jurisdiction so the pointwise minimum is a no-op there
    fpr_np = torch_to_numpy(fpr_grid).astype(np.float64)
    zone = (fpr_np > 0.0) & (fpr_np <= q_j[-1])
    if not zone.any():
        return lower_envelope

    floor_np = np.full_like(fpr_np, np.inf)
    j_star = np.searchsorted(q_j, fpr_np[zone], side="right")
    floor_np[zone] = bounds[j_star]

    floor = numpy_to_torch(floor_np, lower_envelope.device).float()
    return torch.minimum(lower_envelope, floor)


def _wilson_upper_one_sided(p_hat: NDArray, n: int, alpha: float) -> NDArray:
    """Compute the one-sided Wilson score upper bound at level alpha.

    Args:
        p_hat: Observed proportions.
        n: Number of trials.
        alpha: One-sided significance level.

    Returns:
        Upper confidence bounds, clipped to [0, 1].
    """
    z = float(norm.ppf(1.0 - alpha))
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    halfwidth = (z / denom) * np.sqrt(
        p_hat * (1.0 - p_hat) / n + z * z / (4 * n * n)
    )
    return np.clip(center + halfwidth, 0.0, 1.0)


def _apply_beta_orderstat_floor_upper_tail(
    *,
    fpr_grid: Tensor,
    lower_envelope: Tensor,
    y_true: Tensor,
    y_score: Tensor,
    alpha: float,
    j_max: int = 25,
) -> Tensor:
    """Apply the mirrored Beta order-statistic floor at the high-FPR tail.

    Mirror image of _apply_beta_orderstat_floor under the class-swap
    symmetry of ROC space. Anchors are the j-th smallest positive scores
    Y_(j): the true positive-class CDF at Y_(j) follows Beta(j, n_pos+1-j)
    exactly, so with per-event confidence the true TPR of the operating
    point at Y_(j) is at least 1 - rho_j, where rho_j is the Beta upper
    quantile. The true FPR of that operating point is bounded above by a
    one-sided Wilson upper bound f_j on the empirical FPR there (n_neg
    Bernoulli trials independent of Y_(j)). By ROC monotonicity, for every
    evaluation point t >= f_j the true curve satisfies
    R_true(t) >= 1 - rho_{j*} with j* the smallest j such that f_j <= t.

    On the TPR plateau this lowers an over-tight lower envelope (collapsed
    bootstrap support) to an exact, distribution-free bound, mirroring what
    the low-FPR floor does at the steep corner. The alpha budget is split
    Bonferroni-style across the 2 * j_max one-sided events.

    Args:
        fpr_grid: FPR grid values.
        lower_envelope: Current lower envelope bound.
        y_true: True binary labels.
        y_score: Predicted scores.
        alpha: Total alpha budget for the floor.
        j_max: Number of positive order statistics used.

    Returns:
        Lower envelope with the floor applied (pointwise minimum within the
        floor's jurisdiction, unchanged elsewhere).
    """
    neg_scores = torch_to_numpy(y_score[y_true == 0]).astype(np.float64)
    pos_scores = torch_to_numpy(y_score[y_true == 1]).astype(np.float64)
    n_neg = len(neg_scores)
    n_pos = len(pos_scores)
    j_used = min(j_max, n_pos)
    if j_used == 0 or n_neg == 0:
        return lower_envelope

    alpha_event = alpha / (2 * j_max)
    js = np.arange(1, j_used + 1)
    # Certified true-TPR lower bound at the j-th smallest positive
    rho_j = beta_dist.ppf(1.0 - alpha_event, js, n_pos + 1 - js)
    tpr_bounds = 1.0 - rho_j

    # Wilson-raised empirical FPR at each of the j smallest positives marks
    # the start of that anchor's jurisdiction
    pos_asc = np.sort(pos_scores)
    fpr_hat = (neg_scores[None, :] > pos_asc[:j_used, None]).mean(axis=1)
    f_j = _wilson_upper_one_sided(fpr_hat, n_neg, alpha_event)

    fpr_np = torch_to_numpy(fpr_grid).astype(np.float64)
    zone = (fpr_np >= f_j[-1]) & (fpr_np < 1.0)
    if not zone.any():
        return lower_envelope

    # f_j is non-increasing in j; the best (largest) certified bound at t
    # comes from the smallest j with f_j <= t
    f_ascending = f_j[::-1].copy()
    count_qualifying = np.searchsorted(f_ascending, fpr_np[zone], side="right")
    j_star_idx = j_used - count_qualifying

    floor_np = np.full_like(fpr_np, np.inf)
    floor_np[zone] = tpr_bounds[j_star_idx]

    floor = numpy_to_torch(floor_np, lower_envelope.device).float()
    return torch.minimum(lower_envelope, floor)


def _beta_tail_mask(
    fpr_grid: Tensor, n_neg: int, alpha: float, j_max: int = 25
) -> Tensor:
    """Mark the FPR-space tail jurisdictions matching the Beta floor's reach.

    The low tail is (0, q_J] where q_J is the Beta(J, n_neg+1-J) upper
    quantile at the floor's per-event level; the high tail is its mirror
    [1 - q_J, 1). Used to force the Wilson Rectangle floor onto both tails
    in the symmetric-tail ablation.

    Args:
        fpr_grid: FPR grid values.
        n_neg: Number of negative samples.
        alpha: Total alpha budget (split as in the Beta floor).
        j_max: Number of order statistics defining the jurisdiction.

    Returns:
        Boolean tensor marking grid points inside either tail jurisdiction.
    """
    j_used = min(j_max, n_neg)
    alpha_event = alpha / (2 * j_max)
    q_j = float(beta_dist.ppf(1.0 - alpha_event, j_used, n_neg + 1 - j_used))
    low_tail = (fpr_grid > 0.0) & (fpr_grid <= q_j)
    high_tail = (fpr_grid < 1.0) & (fpr_grid >= 1.0 - q_j)
    return low_tail | high_tail


def _studentized_ks_statistics(
    deviations: Tensor, std_dev: Tensor, epsilon: float
) -> Tensor:
    """Compute per-curve maximum absolute studentized deviations.

    Args:
        deviations: (B, K) signed deviations of bootstrap curves from the
            empirical ROC.
        std_dev: (K,) standard deviation used for studentization.
        epsilon: Regularizer below which variance is treated as collapsed;
            deviations smaller than epsilon are treated as numerical noise.

    Returns:
        (B,) tensor of studentized KS statistics.
    """
    studentized = torch.zeros_like(deviations)
    low_var_mask = std_dev < epsilon
    normal_mask = ~low_var_mask
    if normal_mask.any():
        studentized[:, normal_mask] = (
            deviations[:, normal_mask] / std_dev[normal_mask]
        )
    if low_var_mask.any():
        low_devs = deviations[:, low_var_mask]
        studentized[:, low_var_mask] = torch.where(
            torch.abs(low_devs) < epsilon,
            torch.zeros_like(low_devs),
            low_devs / epsilon,
        )
    return torch.abs(studentized).max(dim=1).values


def _ks_retention_envelope(
    boot_tpr: Tensor, ks_statistics: Tensor, alpha: float
) -> tuple[Tensor, Tensor]:
    """Retain the (1-alpha) most typical curves and take their envelope.

    Args:
        boot_tpr: (B, K) bootstrap TPR curves.
        ks_statistics: (B,) studentized KS statistic per curve.
        alpha: Significance level.

    Returns:
        Tuple of (lower, upper) envelopes, clipped to [0, 1].
    """
    n_bootstrap = boot_tpr.shape[0]
    n_retain = int(np.ceil((1 - alpha) * n_bootstrap))
    ks_sorted = torch.sort(ks_statistics).values
    threshold = ks_sorted[n_retain - 1] if n_retain > 0 else float("inf")
    retained = boot_tpr[ks_statistics <= threshold]
    lower = torch.clamp(retained.min(dim=0).values, 0.0, 1.0)
    upper = torch.clamp(retained.max(dim=0).values, 0.0, 1.0)
    return lower, upper


def envelope_band_suite(
    *,
    boot_tpr_matrix: NDArray | Tensor,
    fpr_grid: NDArray | Tensor,
    y_true: NDArray | Tensor,
    y_score: NDArray | Tensor,
    alphas: Sequence[float],
    tpr_method: TprMethod = "empirical",
    include_pre_floor_arm: bool = False,
) -> dict[float, dict[str, tuple[NDArray, NDArray]]]:
    """Compute the envelope band and its ablation variants with shared work.

    Produces, for each alpha, the probability-space KS-retention envelope
    in six configurations that differ only in which tail repairs are applied:

    - "envelope": Wilson variance floor during studentization, variance-ratio
      gated Wilson Rectangle floor, and the exact Beta order-statistic floor
      on the lower band at low FPR. Identical to
      envelope_bootstrap_band(boundary_method="wilson").
    - "envelope_no_beta_floor": as "envelope" without the Beta floor.
    - "envelope_no_wilson_floor": raw bootstrap variance for studentization,
      no rectangle floor; the Beta floor alone repairs the low-FPR tail.
    - "envelope_no_floors": raw variance, no repairs. Identical to
      envelope_bootstrap_band(boundary_method="none").
    - "envelope_beta_both_tails": no Wilson machinery; Beta order-statistic
      floors applied at both tails of the lower band (negative order
      statistics at low FPR, mirrored positive order statistics at high FPR).
    - "envelope_wilson_both_tails": Wilson machinery only, with the
      rectangle floor forced onto both FPR tail jurisdictions in addition
      to the variance-ratio gate; no Beta floor.

    When include_pre_floor_arm is True, each alpha additionally carries
    "envelope_pre_floor": the variance-floored KS-retention envelope before
    the rectangle and Beta floors are applied. This is the bootstrap arm of
    the full method, against which floor attribution can be measured
    exactly (a final lower bound strictly below this arm was set by a
    floor, not by the bootstrap).

    The expensive shared quantities (bootstrap deviations, variances,
    studentized statistics) are computed once across all variants and alphas.

    Args:
        boot_tpr_matrix: (n_bootstrap, n_grid_points) array of TPR values.
        fpr_grid: (n_grid_points,) array of FPR values.
        y_true: Array of true binary labels (0 or 1) from original data.
        y_score: Array of predicted scores from original data.
        alphas: Significance levels to evaluate.
        tpr_method: Method for computing the empirical ROC curve.
        include_pre_floor_arm: Whether to include the "envelope_pre_floor"
            diagnostic entry in the results.

    Returns:
        Mapping alpha -> variant name -> (lower_envelope, upper_envelope)
        numpy arrays on the input FPR grid.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(y_score, np.ndarray):
        dtype = y_score.dtype
    elif isinstance(y_score, torch.Tensor):
        dtype = y_score.cpu().numpy().dtype
    else:
        dtype = np.asarray(y_score).dtype

    boot_tpr = numpy_to_torch(boot_tpr_matrix, device).float()
    fpr = numpy_to_torch(fpr_grid, device).float()
    y_true_t = numpy_to_torch(y_true, device)
    y_score_t = numpy_to_torch(y_score, device).float()

    n_neg = int((y_true_t == 0).sum().item())
    n_pos = int((y_true_t == 1).sum().item())
    n_total = n_neg + n_pos

    empirical_tpr = _compute_empirical_roc(y_true_t, y_score_t, fpr, method=tpr_method)

    deviations = boot_tpr - empirical_tpr.unsqueeze(0)
    var_raw = torch.var(boot_tpr, dim=0, correction=1)
    std_raw = torch.sqrt(var_raw)
    epsilon = min(1.0 / n_total, 1e-6)

    # Raw-variance studentization is alpha-independent; compute once
    ks_raw = _studentized_ks_statistics(deviations, std_raw, epsilon)

    results: dict[float, dict[str, tuple[NDArray, NDArray]]] = {}

    for alpha in alphas:
        z_alpha = (2.0**0.5) * torch.erfinv(torch.tensor(1.0 - alpha)).item()
        wilson_var = (
            wilson_halfwidth_squared_torch(empirical_tpr, n_pos, z_alpha) / z_alpha**2
        )
        std_floored = torch.sqrt(torch.maximum(var_raw, wilson_var))
        ks_floored = _studentized_ks_statistics(deviations, std_floored, epsilon)

        lower_raw, upper_raw = _ks_retention_envelope(boot_tpr, ks_raw, alpha)
        lower_flr, upper_flr = _ks_retention_envelope(boot_tpr, ks_floored, alpha)

        # Variance-ratio gated rectangle (shared by "envelope" and
        # "envelope_no_beta_floor")
        deficiency, alpha_wilson = _compute_variance_ratio_alpha(
            var_raw, wilson_var, alpha
        )
        lower_rect, upper_rect = _apply_wilson_variance_ratio_floor(
            fpr, lower_flr, upper_flr, y_true_t, y_score_t, deficiency, alpha_wilson
        )

        # Rectangle forced onto both FPR tails in addition to the gate
        tail_mask = _beta_tail_mask(fpr, n_neg, alpha)
        deficiency_tails = torch.maximum(deficiency, tail_mask.float())
        k_eff_tails = float(deficiency_tails.sum().item())
        alpha_wilson_tails = (
            1.0 - (1.0 - alpha) ** (1.0 / k_eff_tails) if k_eff_tails > 1.0 else alpha
        )
        lower_rect_tails, upper_rect_tails = _apply_wilson_variance_ratio_floor(
            fpr,
            lower_flr,
            upper_flr,
            y_true_t,
            y_score_t,
            deficiency_tails,
            alpha_wilson_tails,
        )

        beta_lower_kwargs = dict(
            fpr_grid=fpr, y_true=y_true_t, y_score=y_score_t, alpha=alpha
        )

        lower_beta_low_tail = _apply_beta_orderstat_floor(
            lower_envelope=lower_raw, **beta_lower_kwargs
        )
        variants = {
            "envelope": (
                _apply_beta_orderstat_floor(
                    lower_envelope=lower_rect, **beta_lower_kwargs
                ),
                upper_rect,
            ),
            "envelope_no_beta_floor": (lower_rect, upper_rect),
            "envelope_no_wilson_floor": (lower_beta_low_tail, upper_raw),
            "envelope_no_floors": (lower_raw, upper_raw),
            "envelope_beta_both_tails": (
                _apply_beta_orderstat_floor_upper_tail(
                    lower_envelope=lower_beta_low_tail, **beta_lower_kwargs
                ),
                upper_raw,
            ),
            "envelope_wilson_both_tails": (lower_rect_tails, upper_rect_tails),
        }
        if include_pre_floor_arm:
            variants["envelope_pre_floor"] = (lower_flr, upper_flr)

        results[alpha] = {}
        for name, (lower_t, upper_t) in variants.items():
            lower_np = np.array(torch_to_numpy(lower_t), dtype=dtype, copy=True)
            upper_np = np.array(torch_to_numpy(upper_t), dtype=dtype, copy=True)
            lower_np[0] = 0.0
            upper_np[-1] = 1.0
            results[alpha][name] = (lower_np, upper_np)

    return results


def wilson_beta_band(
    y_true: NDArray | Tensor,
    y_score: NDArray | Tensor,
    k: int | None = None,
    alpha: float = 0.05,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute the no-bootstrap ablation of the envelope band.

    Applies the envelope method's two tail repairs to the whole curve
    without any bootstrap component:

    1. Wilson Rectangle bounds at every grid point, with the per-point level
       Šidák-corrected across the interior grid points (the role the
       variance-ratio gate's K_eff plays when the bootstrap is present), and
       Šidák-corrected across each rectangle's two margins.
    2. The exact Beta order-statistic floor on the lower band, extended to
       all n_neg order statistics so its jurisdiction spans the whole curve
       (per-event level alpha / (2 * n_neg)).

    Args:
        y_true: True binary labels (0 or 1).
        y_score: Predicted scores.
        k: Number of points in the FPR grid. If None, uses n_neg + 1.
        alpha: Significance level.

    Returns:
        Tuple of (fpr_grid, lower_envelope, upper_envelope) as numpy arrays.
    """
    y_true_t = numpy_to_torch(y_true, torch.device("cpu"))
    y_score_t = numpy_to_torch(y_score, torch.device("cpu")).float()
    n_neg = int((y_true_t == 0).sum().item())

    if k is None:
        k = n_neg + 1

    m = max(k - 2, 1)
    alpha_grid = 1.0 - (1.0 - alpha) ** (1.0 / m)

    fpr_np, lower_np, upper_np = wilson_rectangle_band(
        y_true=y_true,
        y_score=y_score,
        k=k,
        alpha=alpha_grid,
        correction="sidak",
        tpr_method="empirical",
    )

    # Enforce band monotonicity
    upper_np = np.maximum.accumulate(upper_np)
    lower_np = np.minimum.accumulate(lower_np[::-1])[::-1].copy()

    dtype = lower_np.dtype
    lower_t = _apply_beta_orderstat_floor(
        fpr_grid=numpy_to_torch(fpr_np, torch.device("cpu")).float(),
        lower_envelope=numpy_to_torch(lower_np, torch.device("cpu")).float(),
        y_true=y_true_t,
        y_score=y_score_t,
        alpha=alpha,
        j_max=n_neg,
    )
    lower_np = torch_to_numpy(lower_t).astype(dtype)

    lower_np[0] = 0.0
    upper_np[-1] = 1.0

    return fpr_np, lower_np, upper_np


def envelope_bootstrap_band(
    boot_tpr_matrix: NDArray | Tensor,
    fpr_grid: NDArray | Tensor,
    y_true: NDArray | Tensor,
    y_score: NDArray | Tensor,
    alpha: float = 0.05,
    boundary_method: BoundaryMethod = "none",
    retention_method: RetentionMethod = "ks",
    use_logit: bool = False,
    tpr_method: TprMethod = "empirical",
    plot: bool = False,
    plot_title: str | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute Studentized Bootstrap Envelope Simultaneous Confidence Bands.

    Retains a subset of bootstrap curves based on their consistency with the
    empirical ROC and returns their pointwise envelope.

    Args:
        boot_tpr_matrix: (n_bootstrap, n_grid_points) array of TPR values.
        fpr_grid: (n_grid_points,) array of FPR values.
        y_true: Array of true binary labels (0 or 1) from original data.
        y_score: Array of predicted scores from original data.
        alpha: Significance level. Defaults to 0.05.
        boundary_method: Method for handling zero-variance boundaries where
            bootstrap collapses. Options:
            - "wilson": Adaptive floor using Wilson Rectangle bounds plus an
              exact Beta order-statistic floor. Uses the variance ratio
              r(t) = bootstrap_var / wilson_var to detect where the bootstrap
              has collapsed (r < 1) and applies Šidák-corrected Wilson
              Rectangle bounds as a floor at those points. The Šidák
              correction strength adapts to the effective number of deficient
              points. Interior points where bootstrap variance exceeds Wilson
              variance are left untouched. The lower envelope additionally
              receives a distribution-free Beta order-statistic floor at
              extreme FPR, where threshold-location uncertainty dominates and
              variance-based detection is blind. Also uses simple Wilson
              variance floor during studentization to prevent division by
              zero.
            - "ks": Use KS-style margin extension (Campbell 1994).
              Extends the band from interior points to corners using
              horizontal/vertical margins based on sample sizes.
            - "none": No boundary correction.
            Defaults to "none".
        retention_method: Method for selecting which bootstrap curves to retain.
            Options:
            - "ks": Retain (1-α) curves with smallest studentized KS statistic
              (maximum absolute deviation from empirical).
            - "symmetric": Trim α/2 from curves that deviate most upward and
              α/2 from curves that deviate most downward. This addresses
              asymmetric alpha mass at high AUC where positive deviations
              are bounded by 1 but negative deviations are not.
            Defaults to "ks".
        use_logit: If True, construct the bands in logit space to stabilize
            the variance of the ROC curve. Defaults to False.
        tpr_method: Method for computing the empirical ROC curve (band center).
            Options:
            - "empirical": Standard step-function interpolation (default).
            - "harrell_davis": Beta-weighted quantile estimation for reduced
              finite-sample bias.
            Defaults to "empirical".
        plot: If True, generate diagnostic plots using the viz module. Defaults to False.
        plot_title: Optional custom title for the diagnostic plots. If None, uses
            method description. Defaults to None.

    Returns:
        Tuple of (fpr_grid, lower_envelope, upper_envelope) as numpy arrays.

    Examples:
        >>> import numpy as np
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.linear_model import LogisticRegression
        >>> # Generate data
        >>> X, y = make_classification(n_samples=200, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        >>> # Fit model and get scores
        >>> model = LogisticRegression().fit(X_train, y_train)
        >>> y_score = model.predict_proba(X_test)[:, 1]
        >>> # Generate bootstrap samples (simplified)
        >>> fpr_grid = np.linspace(0, 1, 101)
        >>> boot_tpr = np.random.rand(1000, 101)  # Mock bootstrap samples
        >>> # Compute envelope band
        >>> fpr, lower, upper = envelope_bootstrap_band(
        ...     boot_tpr_matrix=boot_tpr,
        ...     fpr_grid=fpr_grid,
        ...     y_true=y_test,
        ...     y_score=y_score,
        ...     alpha=0.05,
        ...     boundary_method="wilson",
        ...     retention_method="ks",
        ... )
        >>> fpr.shape
        (101,)
        >>> lower.shape
        (101,)
        >>> upper.shape
        (101,)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine dtype from y_score (convert to numpy if needed to get dtype)
    if isinstance(y_score, np.ndarray):
        dtype = y_score.dtype
    elif isinstance(y_score, torch.Tensor):
        dtype = y_score.cpu().numpy().dtype
    else:
        dtype = np.asarray(y_score).dtype

    # Convert all inputs to tensors on the target device
    boot_tpr = numpy_to_torch(boot_tpr_matrix, device).float()
    fpr = numpy_to_torch(fpr_grid, device).float()
    y_true_t = numpy_to_torch(y_true, device)
    y_score_t = numpy_to_torch(y_score, device).float()

    n_bootstrap, n_grid_points = boot_tpr.shape

    # Compute sample sizes
    n_neg = int((y_true_t == 0).sum().item())
    n_pos = int((y_true_t == 1).sum().item())
    n_total = n_neg + n_pos

    # Step 0: Compute empirical ROC
    empirical_tpr = _compute_empirical_roc(y_true_t, y_score_t, fpr, method=tpr_method)

    z_alpha = (2.0**0.5) * torch.erfinv(torch.tensor(1.0 - alpha)).item()

    # Compute probability-space bootstrap variance (needed for variance ratio
    # in Wilson boundary correction regardless of whether logit path is used)
    bootstrap_var_prob = torch.var(boot_tpr, dim=0, correction=1)

    if use_logit:
        # --- PATH A: LOGIT SPACE ENVELOPE ---
        # 1. Transform everything to Haldane-corrected Logit Space
        logit_tpr_hat = _haldane_logit(empirical_tpr, n_pos)
        logit_boot_tpr = _haldane_logit(boot_tpr, n_pos)

        # 2. Compute Bootstrap Variance in Logit Space
        bootstrap_var_logit = torch.var(logit_boot_tpr, dim=0, correction=1)

        # 3. Apply Variance Floors (transform from probability to logit space)
        # Simple Wilson floor prevents division by zero during studentization
        if boundary_method == "wilson":
            variance_floor_prob = (
                wilson_halfwidth_squared_torch(empirical_tpr, n_pos, z_alpha)
                / z_alpha**2
            )
        else:
            variance_floor_prob = torch.zeros_like(empirical_tpr)

        if boundary_method == "wilson":
            # Transform variance floor to logit space using Jacobian
            # Jacobian of logit: d(logit(p))/dp = 1/(p(1-p))
            # Variance transforms as: var_logit = var_prob * jacobian^2
            p_safe = torch.clamp(empirical_tpr, 1e-6, 1.0 - 1e-6)
            jacobian = 1.0 / (p_safe * (1.0 - p_safe))
            variance_floor_logit = variance_floor_prob * jacobian.pow(2)

            # Apply floor in logit space
            bootstrap_var_logit = torch.maximum(
                bootstrap_var_logit, variance_floor_logit
            )

        std_dev = torch.sqrt(bootstrap_var_logit)

        # 4. Compute Signed Deviations in Logit Space
        signed_deviations = logit_boot_tpr - logit_tpr_hat.unsqueeze(0)

        # 5. Studentize
        epsilon = min(1.0 / n_total, 1e-6)
        low_var_mask = std_dev < epsilon

        studentized_signed = torch.zeros_like(signed_deviations)

        # Normal points
        normal_mask = ~low_var_mask
        if normal_mask.any():
            studentized_signed[:, normal_mask] = (
                signed_deviations[:, normal_mask] / std_dev[normal_mask]
            )

        # Low variance points
        if low_var_mask.any():
            low_devs = signed_deviations[:, low_var_mask]
            studentized_signed[:, low_var_mask] = torch.where(
                torch.abs(low_devs) < epsilon,
                torch.zeros_like(low_devs),
                low_devs / epsilon,
            )

        # Prepare for retention (absolute deviations for KS)
        studentized_abs = torch.abs(studentized_signed)
    else:
        # --- PATH B: PROBABILITY SPACE ENVELOPE ---
        # 1. Compute Bootstrap Std (Empirical)
        bootstrap_std = torch.std(boot_tpr, dim=0, correction=1)
        bootstrap_var = bootstrap_std.pow(2)

        # 2. Apply Variance Floors (Boundary Methods)
        # Simple Wilson floor prevents division by zero during studentization
        if boundary_method == "wilson":
            variance_floor = (
                wilson_halfwidth_squared_torch(empirical_tpr, n_pos, z_alpha)
                / z_alpha**2
            )
        else:
            variance_floor = torch.zeros_like(empirical_tpr)

        if boundary_method == "wilson":
            bootstrap_var = torch.maximum(bootstrap_var, variance_floor)
            bootstrap_std = torch.sqrt(bootstrap_var)

        # 3. Studentize
        epsilon = min(1.0 / n_total, 1e-6)
        signed_deviations = boot_tpr - empirical_tpr.unsqueeze(0)

        # Handle low variance points (avoid div by zero)
        low_var_mask = bootstrap_std < epsilon
        std_dev = bootstrap_std.clone()  # For unified reference later

        studentized_signed = torch.zeros_like(signed_deviations)

        # Normal points
        normal_mask = ~low_var_mask
        if normal_mask.any():
            studentized_signed[:, normal_mask] = (
                signed_deviations[:, normal_mask] / bootstrap_std[normal_mask]
            )

        # Low variance points
        if low_var_mask.any():
            low_devs = signed_deviations[:, low_var_mask]
            # If dev is tiny, 0; else scale by epsilon
            studentized_signed[:, low_var_mask] = torch.where(
                torch.abs(low_devs) < epsilon,
                torch.zeros_like(low_devs),
                low_devs / epsilon,
            )

        studentized_abs = torch.abs(studentized_signed)

    # Step 4: Curve Retention
    if retention_method == "symmetric":
        # Trim tails separately
        max_above = studentized_signed.max(dim=1).values
        max_below = studentized_signed.min(dim=1).values

        upper_thresh = torch.quantile(max_above, 1.0 - alpha / 2)
        lower_thresh = torch.quantile(max_below, alpha / 2)

        retained_mask = (max_above <= upper_thresh) & (max_below >= lower_thresh)

    else:  # "ks" (default)
        # Trim based on max absolute deviation
        ks_statistics = torch.max(studentized_abs, dim=1).values

        n_retain = int(np.ceil((1 - alpha) * n_bootstrap))
        ks_sorted = torch.sort(ks_statistics).values
        threshold = ks_sorted[n_retain - 1] if n_retain > 0 else float("inf")

        retained_mask = ks_statistics <= threshold

    # Step 5: Envelope Construction
    if use_logit:
        # Construct envelope in Logit Space
        retained_logits = logit_boot_tpr[retained_mask]
        lower_logit = torch.min(retained_logits, dim=0).values
        upper_logit = torch.max(retained_logits, dim=0).values

        # Back-transform to Probability Space
        lower_envelope = torch.sigmoid(lower_logit)
        upper_envelope = torch.sigmoid(upper_logit)

    else:
        # Construct envelope in Probability Space
        retained_curves = boot_tpr[retained_mask]
        lower_envelope = torch.min(retained_curves, dim=0).values
        upper_envelope = torch.max(retained_curves, dim=0).values

    # Step 6: Clip to [0, 1]
    lower_envelope = torch.clamp(lower_envelope, 0.0, 1.0)
    upper_envelope = torch.clamp(upper_envelope, 0.0, 1.0)

    # Step 6b: Apply boundary corrections
    if boundary_method == "ks":
        lower_envelope, upper_envelope = _extend_boundary_ks_style(
            fpr, lower_envelope, upper_envelope, empirical_tpr, n_neg, n_pos, alpha
        )
    elif boundary_method == "wilson":
        # Determine where bootstrap variance is deficient relative to Wilson
        # and apply Šidák-corrected Wilson Rectangle floor at those points.
        # Always compare in probability space for a consistent ratio.
        wilson_var = (
            wilson_halfwidth_squared_torch(empirical_tpr, n_pos, z_alpha)
            / z_alpha**2
        )
        deficiency, alpha_wilson = _compute_variance_ratio_alpha(
            bootstrap_var_prob, wilson_var, alpha,
        )
        lower_envelope, upper_envelope = _apply_wilson_variance_ratio_floor(
            fpr, lower_envelope, upper_envelope,
            y_true_t, y_score_t, deficiency, alpha_wilson,
        )
        # The variance-ratio floor measures vertical (binomial) uncertainty
        # and cannot see the horizontal threshold-location uncertainty that
        # dominates at extreme FPR; the exact Beta order-statistic floor
        # carries that channel for the lower band.
        lower_envelope = _apply_beta_orderstat_floor(
            fpr_grid=fpr,
            lower_envelope=lower_envelope,
            y_true=y_true_t,
            y_score=y_score_t,
            alpha=alpha,
        )

    # Enforce boundary conditions
    # FPR=0: [0, upper]
    # FPR=1: [lower, 1]
    lower_envelope[0] = 0.0
    upper_envelope[-1] = 1.0

    # Convert back to numpy with original dtype
    fpr_np = torch_to_numpy(fpr).astype(dtype)
    lower_np = torch_to_numpy(lower_envelope).astype(dtype)
    upper_np = torch_to_numpy(upper_envelope).astype(dtype)

    # Generate diagnostic plots if requested
    if plot:
        bootstrap_var_np = torch_to_numpy(bootstrap_var_prob).astype(dtype)
        if boundary_method == "wilson":
            variance_floor_np = torch_to_numpy(wilson_var).astype(dtype)
        else:
            variance_floor_np = None

        try:
            empirical_tpr_np = torch_to_numpy(empirical_tpr).astype(dtype)
            boot_tpr_np = torch_to_numpy(boot_tpr).astype(dtype)

            # Determine method name for title
            if plot_title is None:
                plot_title = f"Envelope Bootstrap ({retention_method} retention, {boundary_method} boundary)"

            fig = plot_band_diagnostics(
                fpr_grid=fpr_np,
                empirical_tpr=empirical_tpr_np,
                lower_envelope=lower_np,
                upper_envelope=upper_np,
                boot_tpr_matrix=boot_tpr_np,
                bootstrap_var=bootstrap_var_np,
                wilson_var=variance_floor_np,
                alpha=alpha,
                method_name=plot_title,
                layout="2x2",
            )
            fig.show()
        except ImportError:
            import warnings

            warnings.warn(
                "Visualization module not available. Install matplotlib to enable plotting.",
                stacklevel=2,
            )

    return (fpr_np, lower_np, upper_np)
