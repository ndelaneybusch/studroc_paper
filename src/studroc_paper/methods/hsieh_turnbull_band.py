"""Hsieh-Turnbull Simultaneous Confidence Bands for ROC Curves.

This module implements simultaneous confidence bands based on the
asymptotic variance formula derived by Hsieh & Turnbull (1996).
"""

import warnings
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import stats
from torch import Tensor

from ..sampling.bootstrap_grid import generate_bootstrap_grid
from ..viz.band_diagnostics import plot_band_diagnostics
from .method_utils import compute_hsieh_turnbull_variance, numpy_to_torch


def _check_log_concavity_assumptions(
    neg_scores: NDArray, pos_scores: NDArray, kurtosis_threshold: float = 2.0
) -> tuple[bool, str]:
    """Check if data grossly violates log-concave density assumptions.

    Log-concave densities include Normal, Gamma, Beta, Exponential, Logistic,
    and most unimodal symmetric distributions. Violations include:
    - Heavy-tailed distributions (high kurtosis)
    - Multimodal distributions
    - Highly skewed distributions with infinite variance

    This is a heuristic check based on excess kurtosis, which is a simple
    proxy for heavy-tailedness. Log-concave distributions have at most
    exponential tails, corresponding to excess kurtosis ≤ 0 for symmetric
    cases. Values significantly above 0 suggest potential violations.

    Args:
        neg_scores: Control/negative class scores.
        pos_scores: Case/positive class scores.
        kurtosis_threshold: Threshold for excess kurtosis above which a
            warning is issued. Default 2.0 is conservative (normal = 0,
            Laplace ≈ 3, t(5) ≈ 6, heavier tails → higher values).

    Returns:
        Tuple of (passes_check, message) where passes_check is False if
        the data appears to violate log-concavity assumptions.
    """
    neg_kurtosis = stats.kurtosis(neg_scores, fisher=True)  # Excess kurtosis
    pos_kurtosis = stats.kurtosis(pos_scores, fisher=True)

    violations = []

    if neg_kurtosis > kurtosis_threshold:
        violations.append(
            f"Negative class has high excess kurtosis ({neg_kurtosis:.2f} > {kurtosis_threshold}), "
            "suggesting heavy tails"
        )
    if pos_kurtosis > kurtosis_threshold:
        violations.append(
            f"Positive class has high excess kurtosis ({pos_kurtosis:.2f} > {kurtosis_threshold}), "
            "suggesting heavy tails"
        )

    # Check for potential bimodality via Hartigan's dip test proxy:
    # If the distribution has clear gaps, the density ratio g/f can be unstable
    # Here we use a simple heuristic: coefficient of variation combined with
    # range-to-IQR ratio. True bimodality tests require more complex methods.
    for name, scores in [("negative", neg_scores), ("positive", pos_scores)]:
        iqr = np.percentile(scores, 75) - np.percentile(scores, 25)
        full_range = np.max(scores) - np.min(scores)
        if iqr > 0 and full_range / iqr > 6:
            violations.append(
                f"{name.capitalize()} class has unusual range-to-IQR ratio "
                f"({full_range / iqr:.1f}), potentially indicating outliers or multimodality"
            )

    if violations:
        message = (
            "Data may violate log-concave assumptions required for Hsieh-Turnbull "
            "asymptotic variance. Issues detected:\n  - " + "\n  - ".join(violations)
        )
        return False, message

    return True, ""


def hsieh_turnbull_band(
    y_true: NDArray | Tensor,
    y_score: NDArray | Tensor,
    k: int = 1000,
    alpha: float = 0.05,
    use_logit_transform: bool = True,
    density_method: Literal["log_concave", "reflected_kde"] = "log_concave",
    n_bootstraps: int = 2000,
    check_assumptions: bool = True,
    plot: bool = False,
    plot_title: str | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute Hsieh-Turnbull simultaneous confidence bands for ROC curves.

    Constructs asymptotic confidence bands using the Hsieh-Turnbull (1996)
    variance formula, which accounts for both vertical (case) and horizontal
    (control) sources of variation in the empirical ROC curve.

    The asymptotic variance formula is:
        Var(R̂(t)) = R(t)(1-R(t))/n₁ + [R'(t)]² × t(1-t)/n₀

    where R(t) is the TPR at FPR=t, R'(t) = g(c)/f(c) is the ROC slope
    (likelihood ratio at threshold c), n₀ is the control sample size,
    and n₁ is the case sample size.

    **Calibration Method:**
    By default (`n_bootstraps > 0`), this function uses **bootstrap calibration**
    to determine the simultaneous critical value `z`. Instead of relying on strict
    independence (Bonferroni) or asymptotic Gaussian process theory (Brownian Bridge),
    it generates bootstrap replicates of the ROC process to empirically determine
    the critical value that ensures 1-alpha simultaneous coverage:

        z_cal = Quantile_{1-alpha} ( sup_t |(R*(t) - R(t)) / SE(t)| )

    This provides a tighter band than Bonferroni while correctly accounting for
    the correlation structure of the ROC curve.

    **When This Approach Is Valid:**
    - The underlying score densities are approximately log-concave (includes
      Normal, Gamma, Beta, Exponential, Logistic, and most unimodal
      symmetric/mildly skewed distributions)
    - Sample sizes are moderately large (asymptotic theory; typically n≥50
      per class recommended)
    - The ROC curve is smooth (no abrupt discontinuities in the population)
    - The densities f and g are absolutely continuous with bounded derivatives

    **When This Approach May Fail:**
    - Heavy-tailed distributions (e.g., Cauchy, Pareto, log-normal with
      high σ, Student-t with low degrees of freedom): The density ratio
      g(c)/f(c) becomes unstable and the asymptotic variance may be
      severely underestimated
    - Multimodal score distributions: The log-concave MLE imposes unimodality,
      which may produce biased density estimates
    - Very small sample sizes (n < 30 per class): Asymptotic approximations
      break down; consider bootstrap methods instead
    - Discrete or highly tied scores: The continuous density assumption fails
    - Strongly separated distributions (AUC ≈ 1): Numerical instability in
      density ratio estimation at extreme thresholds

    **Note on Logit Transform:**
    When `use_logit_transform=True` (default), confidence intervals are
    constructed on the logit(TPR) scale and back-transformed. This is
    recommended because:
    1. The logit scale has variance-stabilizing properties near boundaries
    2. Back-transformation automatically respects [0, 1] constraints
    3. It prevents the "pinching" artifact where bands collapse at corners

    Args:
        y_true: Array of binary class labels (0/1) (numpy array or torch tensor).
        y_score: Array of continuous scores (numpy array or torch tensor).
        k: Number of evaluation points on the FPR grid.
        alpha: Significance level (e.g., 0.05 for 95% confidence).
        use_logit_transform: If True (default), construct intervals on the logit
            scale for stable boundary behavior. If False, use direct Gaussian
            intervals on the probability scale.
        density_method: Method for estimating score densities. Options:
            - "log_concave": Log-concave MLE via convex optimization. More
              robust to outliers and provides smooth derivative estimates.
            - "reflected_kde": Reflected kernel density estimation with ISJ
              bandwidth. May exhibit instability in tails.
        n_bootstraps: Number of bootstrap replicates to use for calibrating the
            critical value. If > 0 (default 2000), uses bootstrap calibration
            which provides tighter, more accurate bands than the Bonferroni
            heuristic. If 0, uses the conservative `sqrt(k)` heuristic.
        check_assumptions: If True (default), run heuristic checks for gross
            violations of log-concavity assumptions and issue warnings.
        plot: If True, generate diagnostic plots using the viz module (default False).
        plot_title: Optional custom title for the diagnostic plots. If None, uses
            "Hsieh-Turnbull".

    Returns:
        Tuple of (fpr_grid, lower_envelope, upper_envelope) as numpy arrays.

    Raises:
        ValueError: If y_true contains values other than 0 and 1, or if there
            are fewer than 2 samples per class.

    References:
        Hsieh, F. and Turnbull, B.W. (1996). "Nonparametric and Semiparametric
        Estimation of the Receiver Operating Characteristic Curve."
        The Annals of Statistics, 24(1), 25-40.
    """
    # Keep original tensors if possible for bootstrap efficiency, but need numpy for basic stats
    if isinstance(y_score, Tensor):
        # Store original tensors for bootstrap
        y_score_torch = y_score
        y_true_torch = (
            y_true
            if isinstance(y_true, Tensor)
            else torch.tensor(y_true, device=y_score.device)
        )

        y_score = y_score.detach().cpu().numpy()
        dtype = y_score.dtype
    else:
        # User passed numpy, create torch tensors if needed later
        y_score_torch = None
        y_true_torch = None
        y_score = np.asarray(y_score)
        dtype = y_score.dtype

    if isinstance(y_true, Tensor):
        y_true = y_true.detach().cpu().numpy()
    else:
        y_true = np.asarray(y_true)

    # Validate inputs
    unique_labels = np.unique(y_true)
    if not np.array_equal(unique_labels, np.array([0, 1])):
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError(
                f"y_true must contain only 0 and 1, got unique values: {unique_labels}"
            )

    # Separate scores by class
    neg_scores = y_score[y_true == 0]
    pos_scores = y_score[y_true == 1]
    n0 = len(neg_scores)
    n1 = len(pos_scores)

    if n0 < 2 or n1 < 2:
        raise ValueError(f"Need at least 2 samples per class, got n0={n0}, n1={n1}")

    # Check assumptions if requested
    if check_assumptions:
        passes, message = _check_log_concavity_assumptions(neg_scores, pos_scores)
        if not passes:
            warnings.warn(
                f"{message}\n"
                "Consider using bootstrap-based methods (e.g., envelope_bootstrap_band) "
                "for more robust inference with heavy-tailed or non-standard distributions.",
                UserWarning,
                stacklevel=2,
            )

    # Create FPR grid
    fpr_grid = np.linspace(0.0, 1.0, k, dtype=dtype)

    # Compute empirical TPR at each grid point
    # Threshold c at FPR=t is the (1-t) quantile of negative scores
    thresholds = np.zeros_like(fpr_grid)
    valid_mask = (fpr_grid > 0) & (fpr_grid < 1)

    if np.any(valid_mask):
        thresholds[valid_mask] = np.quantile(neg_scores, 1 - fpr_grid[valid_mask])

    # Set boundary thresholds to ensure strict inequality checks work
    # We add a buffer because we use strict inequality > in searchsorted/comparisons.
    # Without buffer, max_score > max_score is False.
    min_score = min(np.min(neg_scores), np.min(pos_scores))
    max_score = max(np.max(neg_scores), np.max(pos_scores))
    thresholds[fpr_grid <= 0] = max_score + 0.1
    thresholds[fpr_grid >= 1] = min_score - 0.1

    # Compute empirical TPR: R(t) = P(Y > c)
    tpr_empirical = np.mean(pos_scores[:, None] > thresholds[None, :], axis=0)

    # Compute Hsieh-Turnbull variance
    ht_variance = compute_hsieh_turnbull_variance(
        neg_scores=neg_scores,
        pos_scores=pos_scores,
        fpr_grid=fpr_grid,
        method=density_method,
    )

    # --- Critical Value Determination ---

    if n_bootstraps > 0:
        # Use Bootstrap Calibration

        # Prepare Tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if y_score_torch is None or y_true_torch is None:
            # Conversion required
            neg_scores_t = numpy_to_torch(neg_scores, device)
            pos_scores_t = numpy_to_torch(pos_scores, device)
        else:
            neg_scores_t = y_score_torch[y_true_torch == 0].to(device)
            pos_scores_t = y_score_torch[y_true_torch == 1].to(device)

        fpr_grid_t = numpy_to_torch(fpr_grid, device)
        tpr_empirical_t = numpy_to_torch(tpr_empirical, device)
        ht_variance_t = numpy_to_torch(ht_variance, device)

        # Compute sorted empirical ROC for rank reconstruction
        # We need the full set of jump points, not just grid points
        all_scores = torch.cat([neg_scores_t, pos_scores_t])
        all_labels = torch.cat(
            [torch.zeros_like(neg_scores_t), torch.ones_like(pos_scores_t)]
        )

        # Sort scores descending
        perm = torch.argsort(all_scores, descending=True)
        all_labels_sorted = all_labels[perm]

        # Cumulative sums for TPR/FPR
        tps = torch.cumsum(all_labels_sorted, dim=0)
        fps = torch.cumsum(1 - all_labels_sorted, dim=0)

        # Prepend 0,0 and divide by N
        tpr_full = torch.cat([torch.tensor([0.0], device=device), tps / n1])
        fpr_full = torch.cat([torch.tensor([0.0], device=device), fps / n0])

        # Generate Bootstrap ROCs: (B, K)
        boot_tpr_matrix = generate_bootstrap_grid(
            fpr=fpr_full,
            tpr=tpr_full,
            n_negatives=n0,
            n_positives=n1,
            B=n_bootstraps,
            grid=fpr_grid_t,
            device=device,
        )

        # Compute Sup-Statistic
        # Stat = |R* - R| / SE

        if use_logit_transform:
            # Transform to logit scale
            eps = 1e-7
            tpr_clipped = torch.clamp(tpr_empirical_t, eps, 1 - eps)
            logit_tpr = torch.log(tpr_clipped / (1 - tpr_clipped))

            boot_clipped = torch.clamp(boot_tpr_matrix, eps, 1 - eps)
            logit_boot = torch.log(boot_clipped / (1 - boot_clipped))

            # SE on logit scale
            logit_var = ht_variance_t / (tpr_clipped * (1 - tpr_clipped)) ** 2
            logit_se = torch.sqrt(logit_var)

            # Handle SE=0 cases
            valid_se = logit_se > 1e-9

            # Exclude regions with low effective sample size for logit stability
            # Rule of thumb: n*p > 5 for normal approximation on logit scale
            min_tpr = 5.0 / n1
            stable_mask = (tpr_empirical_t > min_tpr) & (tpr_empirical_t < 1 - min_tpr)
            calib_mask = valid_se & stable_mask

            deviations = torch.abs(logit_boot - logit_tpr.unsqueeze(0))

            # Initialize with zeros
            stat_grid = torch.zeros_like(deviations)
            stat_grid[:, calib_mask] = deviations[:, calib_mask] / logit_se[calib_mask]

        else:
            se = torch.sqrt(ht_variance_t)
            valid_se = se > 1e-9

            # Apply same effective sample size filter for consistency
            min_tpr = 5.0 / n1
            stable_mask = (tpr_empirical_t > min_tpr) & (tpr_empirical_t < 1 - min_tpr)
            calib_mask = valid_se & stable_mask

            deviations = torch.abs(boot_tpr_matrix - tpr_empirical_t.unsqueeze(0))

            stat_grid = torch.zeros_like(deviations)
            stat_grid[:, calib_mask] = deviations[:, calib_mask] / se[calib_mask]

        # Sup over t (grid) for each bootstrap
        sup_stats = torch.max(stat_grid, dim=1).values

        # Quantile
        z_simultaneous = torch.quantile(sup_stats, 1 - alpha).item()

    else:
        # Effective number of independent comparisons (rule of thumb: sqrt(k))
        effective_comparisons = max(np.sqrt(k), 10)
        z_simultaneous = stats.norm.ppf(1 - alpha / (2 * effective_comparisons))

    if use_logit_transform:
        # Construct intervals on logit scale
        # logit(R) = log(R / (1-R))
        # Var(logit(R)) ≈ Var(R) / [R(1-R)]^2  (Delta method)

        # Clip TPR to avoid log(0) or log(inf)
        eps = 1e-7
        tpr_clipped = np.clip(tpr_empirical, eps, 1 - eps)

        logit_tpr = np.log(tpr_clipped / (1 - tpr_clipped))

        # Variance on logit scale
        logit_variance = ht_variance / (tpr_clipped * (1 - tpr_clipped)) ** 2

        # Standard error on logit scale
        logit_se = np.sqrt(logit_variance)

        # Confidence bounds on logit scale
        logit_lower = logit_tpr - z_simultaneous * logit_se
        logit_upper = logit_tpr + z_simultaneous * logit_se

        # Back-transform to probability scale
        lower_envelope = 1 / (1 + np.exp(-logit_lower))
        upper_envelope = 1 / (1 + np.exp(-logit_upper))

    else:
        # Direct Gaussian intervals on probability scale
        se = np.sqrt(ht_variance)
        lower_envelope = tpr_empirical - z_simultaneous * se
        upper_envelope = tpr_empirical + z_simultaneous * se

    # Enforce [0, 1] constraints
    lower_envelope = np.clip(lower_envelope, 0.0, 1.0)
    upper_envelope = np.clip(upper_envelope, 0.0, 1.0)

    # Fix endpoints: lower must pass through (0,0), upper through (1,1)
    lower_envelope[0] = 0.0
    upper_envelope[-1] = 1.0

    # Cast to original dtype
    lower_envelope = lower_envelope.astype(dtype)
    upper_envelope = upper_envelope.astype(dtype)

    # Generate diagnostic plots if requested
    if plot:
        try:
            # Determine method name for title
            if plot_title is None:
                plot_title = f"Hsieh-Turnbull estimated by {density_method} density"
                if use_logit_transform:
                    plot_title += " - Logit Transform"
                if n_bootstraps > 0:
                    plot_title += " (Bootstrap Calibrated)"

            # Convert bootstrap matrix to numpy if it was generated
            boot_tpr_matrix_np = None
            if n_bootstraps > 0:
                boot_tpr_matrix_np = (
                    boot_tpr_matrix.detach().cpu().numpy().astype(dtype)
                )

            fig = plot_band_diagnostics(
                fpr_grid=fpr_grid,
                empirical_tpr=tpr_empirical,
                lower_envelope=lower_envelope,
                upper_envelope=upper_envelope,
                boot_tpr_matrix=boot_tpr_matrix_np,
                bootstrap_var=ht_variance,
                wilson_var=None,  # H-T doesn't use Wilson variance floor
                alpha=alpha,
                method_name=plot_title,
                layout="2x2",
            )
            fig.show()
        except ImportError:
            warnings.warn(
                "Visualization module not available. Install matplotlib to enable plotting.",
                stacklevel=2,
            )

    return fpr_grid, lower_envelope, upper_envelope
