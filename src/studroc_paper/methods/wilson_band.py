from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import stats as sp_stats
from torch import Tensor

from studroc_paper.viz import plot_band_diagnostics

from .method_utils import (
    compute_empirical_roc_from_scores,
    compute_empirical_roc_from_scores_hd,
    numpy_to_torch,
    torch_to_numpy,
)


def _wilson_bounds_torch(p: Tensor, n: int, z: float) -> tuple[Tensor, Tensor]:
    """Compute Wilson score interval bounds for a binomial proportion.

    The Wilson score interval is:
        (p + z²/2n ± z√(p(1-p)/n + z²/4n²)) / (1 + z²/n)

    This interval has guaranteed non-zero width even at p=0 or p=1, making it
    suitable for ROC analysis where empirical rates hit boundaries.

    Args:
        p: Tensor of proportion estimates (e.g., TPR or FPR values).
        n: Number of Bernoulli trials (e.g., n_pos for TPR, n_neg for FPR).
        z: Critical value from standard normal (e.g., 1.96 for 95% CI).

    Returns:
        Tuple of (lower_bound, upper_bound) tensors, clamped to [0, 1].

    Examples:
        >>> p = torch.tensor([0.0, 0.5, 1.0])
        >>> lower, upper = _wilson_bounds_torch(p, n=100, z=1.96)
        >>> torch.all(upper > lower)
        tensor(True)
        >>> lower[0] > 0  # Non-zero even at p=0
        tensor(True)
    """
    if n <= 0:
        return torch.zeros_like(p), torch.ones_like(p)

    z_sq = z * z
    denom = 1.0 + z_sq / n

    # Wilson-adjusted center (shrinks toward 0.5)
    center = (p + z_sq / (2.0 * n)) / denom

    # Half-width term
    discriminant = p * (1.0 - p) / n + z_sq / (4.0 * n * n)
    halfwidth = (z / denom) * torch.sqrt(discriminant)

    lower = torch.clamp(center - halfwidth, 0.0, 1.0)
    upper = torch.clamp(center + halfwidth, 0.0, 1.0)

    return lower, upper


def wilson_band(
    y_true: NDArray | Tensor,
    y_score: NDArray | Tensor,
    k: int | None = None,
    alpha: float = 0.05,
    harrell_davis: bool = True,
    plot_diagnostics: bool = False,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute Wilson confidence band for ROC curve.

    Args:
        y_true: True binary labels (0 or 1).
        y_score: Predicted scores.
        k: Number of points in the FPR grid. If None, use n_neg + 1.
        alpha: Significance level (e.g., 0.05 for 95% confidence).
        harrell_davis: Whether to use Harrell-Davis smoothing for the center estimate.
        plot_diagnostics: Whether to generate diagnostic plots.

    Returns:
        Tuple of (fpr_grid, lower_envelope, upper_envelope).
    """
    y_score = numpy_to_torch(y_score)
    y_true = numpy_to_torch(y_true)
    dtype = y_score.cpu().numpy().dtype

    # Calculate N positive (for TPR variance) and N negative (for grid)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    if k is None:
        k = n_neg + 1

    fpr_grid = np.linspace(0, 1, k, dtype=dtype)
    fpr_grid = numpy_to_torch(fpr_grid)

    # 1. Estimate the center line (Point Estimate)
    if harrell_davis:
        # Smooth estimator: reduces stepping artifacts
        tpr_est = compute_empirical_roc_from_scores_hd(y_score, y_true, fpr_grid)
    else:
        # Standard empirical estimator
        tpr_est = compute_empirical_roc_from_scores(y_score, y_true, fpr_grid)

    # 2. Compute Asymmetric Wilson Bounds
    # Correct Z-score for two-tailed test
    # erfinv(1 - alpha) gives the point where CDF is 1 - alpha/2
    z_alpha = (2.0**0.5) * torch.erfinv(torch.tensor(1.0 - alpha)).item()

    # We pass the estimated TPR as 'p' to the Wilson formula.
    # We use n_pos because we are estimating uncertainty of TPR (Sensitivity).
    lower_envelope, upper_envelope = _wilson_bounds_torch(tpr_est, n_pos, z_alpha)

    # Convert to numpy
    fpr_grid_np = torch_to_numpy(fpr_grid)
    tpr_np = torch_to_numpy(tpr_est)
    lower_envelope_np = torch_to_numpy(lower_envelope)
    upper_envelope_np = torch_to_numpy(upper_envelope)

    if plot_diagnostics:
        # For visualization, we can infer an approximate "variance"
        # from the width, though Wilson doesn't assume constant variance.
        # This is strictly for the diagnostic plot function.
        approx_width = (upper_envelope - lower_envelope) / 2.0
        # Recover 'variance' by reversing Wald: width = z * sqrt(var) -> var = (width/z)^2
        wilson_var_proxy = (approx_width / z_alpha) ** 2
        wilson_var_np = torch_to_numpy(wilson_var_proxy)

        method_name = "Wilson Band"
        if harrell_davis:
            method_name += " (Harrell-Davis)"

        fig = plot_band_diagnostics(
            fpr_grid=fpr_grid_np,
            empirical_tpr=tpr_np,
            lower_envelope=lower_envelope_np,
            upper_envelope=upper_envelope_np,
            boot_tpr_matrix=None,
            bootstrap_var=None,
            wilson_var=wilson_var_np,
            alpha=alpha,
            method_name=method_name,
            layout="2x2",
        )
        fig.show()

    return (fpr_grid_np, lower_envelope_np, upper_envelope_np)


# Type aliases
TprMethod = Literal["empirical", "harrell_davis"]
CorrectionMethod = Literal["none", "sidak", "bonferroni"]


def _compute_corrected_z(alpha: float, correction: CorrectionMethod) -> float:
    """Compute z critical value with optional multiple comparisons correction.

    For joint 2D confidence rectangles with independent margins:
    - "none": Use α directly (joint coverage ≈ (1-α)², anti-conservative)
    - "sidak": Use α' = 1 - √(1-α) per margin (exact joint coverage)
    - "bonferroni": Use α/2 per margin (conservative)

    Args:
        alpha: Desired significance level for the joint region.
        correction: Correction method for two independent margins.

    Returns:
        z critical value for constructing each marginal interval.

    Examples:
        >>> _compute_corrected_z(0.05, "none")  # ~1.96
        1.959...
        >>> _compute_corrected_z(0.05, "sidak")  # ~2.24
        2.236...
    """
    if correction == "none":
        alpha_marginal = alpha
    elif correction == "sidak":
        # For 2 independent tests: (1-α')² = 1-α → α' = 1 - √(1-α)
        alpha_marginal = 1.0 - np.sqrt(1.0 - alpha)
    elif correction == "bonferroni":
        # Conservative: α' = α/2 for 2 tests
        alpha_marginal = alpha / 2.0
    else:
        raise ValueError(f"Unknown correction method: {correction}")

    return float(sp_stats.norm.ppf(1.0 - alpha_marginal / 2.0))


def _interpolate_envelope_to_grid(
    env_fpr: Tensor, env_tpr: Tensor, fpr_grid: Tensor, fill_lower: bool = True
) -> Tensor:
    """Interpolate envelope points onto a regular FPR grid.

    Handles potential non-monotonicity in envelope FPR values by sorting
    and using linear interpolation. Boundary behavior depends on whether
    this is a lower or upper envelope.

    Args:
        env_fpr: FPR coordinates of envelope points.
        env_tpr: TPR coordinates of envelope points.
        fpr_grid: Target FPR grid for interpolation.
        fill_lower: If True, use 0 for extrapolation below and 1 above
            (appropriate for lower envelope). If False, reverse.

    Returns:
        Interpolated TPR values at fpr_grid points.
    """
    # Sort by FPR for interpolation
    sort_idx = torch.argsort(env_fpr)
    fpr_sorted = env_fpr[sort_idx]
    tpr_sorted = env_tpr[sort_idx]

    # Handle duplicates by taking appropriate extreme
    unique_fpr, inverse_idx = torch.unique(fpr_sorted, return_inverse=True)
    unique_tpr = torch.zeros_like(unique_fpr)

    for i in range(len(unique_fpr)):
        mask = inverse_idx == i
        if fill_lower:
            unique_tpr[i] = tpr_sorted[mask].min()
        else:
            unique_tpr[i] = tpr_sorted[mask].max()

    # Linear interpolation to grid
    # Use numpy interp (torch doesn't have native 1D interp)
    fpr_np = unique_fpr.cpu().numpy()
    tpr_np = unique_tpr.cpu().numpy()
    grid_np = fpr_grid.cpu().numpy()

    tpr_interp = np.interp(grid_np, fpr_np, tpr_np)

    return torch.tensor(tpr_interp, dtype=fpr_grid.dtype, device=fpr_grid.device)


def wilson_rectangle_band(
    y_true: NDArray | Tensor,
    y_score: NDArray | Tensor,
    k: int | None = None,
    alpha: float = 0.05,
    correction: CorrectionMethod = "none",
    tpr_method: TprMethod = "empirical",
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute Wilson rectangle confidence bands for ROC curves.

    Constructs pointwise confidence bands by forming 2D confidence rectangles
    at each ROC operating point, using Wilson score intervals for both FPR
    (from n_neg samples) and TPR (from n_pos samples).

    The band envelope is formed by:
    - Upper envelope: upper-left corners (FPR_lower, TPR_upper) — optimistic
    - Lower envelope: lower-right corners (FPR_upper, TPR_lower) — pessimistic

    Unlike bootstrap methods, Wilson intervals provide guaranteed non-zero
    width at boundaries (TPR=0, TPR=1, FPR=0, FPR=1), avoiding the variance
    collapse problem when empirical rates hit extremes.

    Note: This produces pointwise (not simultaneous) coverage. For k grid
    points, joint coverage is less than 1-α. Use correction="sidak" to
    achieve proper joint coverage for each rectangle, though this still
    does not guarantee simultaneous coverage of the entire curve.

    Args:
        y_true: True binary labels (0 or 1). Shape (n_samples,).
        y_score: Predicted scores (higher indicates positive). Shape (n_samples,).
        k: Number of points in FPR grid. If None, uses n_neg + 1 for all
            unique empirical FPR values. Defaults to None.
        alpha: Significance level for confidence bands. Defaults to 0.05.
        correction: Multiple comparisons correction for 2D rectangles.
            - "none": No correction (default). Joint rectangle coverage ≈ (1-α)².
            - "sidak": Exact correction for independent margins. Each rectangle
              has exactly (1-α) coverage.
            - "bonferroni": Conservative correction. Each rectangle has at least
              (1-α) coverage.
            Defaults to "none".
        tpr_method: Method for computing empirical TPR at each FPR grid point.
            - "empirical": Standard step-function interpolation.
            - "harrell_davis": Beta-weighted quantile estimation for smoother
              curves with reduced finite-sample bias.
            Defaults to "empirical".
        monotonic: If True, enforce monotonically non-decreasing envelopes
            (required for valid ROC bands). Defaults to True.

    Returns:
        Tuple of three numpy arrays:
            - fpr_grid: FPR grid of shape (k,).
            - lower_envelope: Lower confidence band (TPR) of shape (k,).
            - upper_envelope: Upper confidence band (TPR) of shape (k,).

    Raises:
        ValueError: If y_true contains values other than 0 and 1.
        ValueError: If y_true and y_score have different lengths.

    Examples:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> y_true = np.array([0] * 100 + [1] * 100)
        >>> y_score = np.concatenate(
        ...     [np.random.normal(0, 1, 100), np.random.normal(1.5, 1, 100)]
        ... )
        >>> fpr, lower, upper = wilson_rectangle_band(y_true, y_score)
        >>> fpr.shape
        (101,)
        >>> np.all(upper >= lower)
        True
        >>> lower[0], upper[-1]  # Boundary anchors
        (0.0, 1.0)
    """
    # Input validation and conversion
    y_score_t = numpy_to_torch(y_score)
    y_true_t = numpy_to_torch(y_true)

    if y_score_t.shape != y_true_t.shape:
        raise ValueError(
            f"Shape mismatch: y_score {y_score_t.shape} vs y_true {y_true_t.shape}"
        )

    unique_labels = torch.unique(y_true_t)
    if not torch.all((unique_labels == 0) | (unique_labels == 1)):
        raise ValueError(f"y_true must contain only 0 and 1, got {unique_labels}")

    # Preserve input dtype for output
    dtype = y_score_t.cpu().numpy().dtype

    # Extract class counts
    n_pos = int((y_true_t == 1).sum().item())
    n_neg = int((y_true_t == 0).sum().item())

    if n_pos < 1 or n_neg < 1:
        raise ValueError(
            f"Need at least 1 sample per class, got n_pos={n_pos}, n_neg={n_neg}"
        )

    # Construct FPR grid
    if k is None:
        k = n_neg + 1
    fpr_grid = torch.linspace(
        0.0, 1.0, k, dtype=y_score_t.dtype, device=y_score_t.device
    )

    # Compute empirical TPR at grid points
    neg_scores = y_score_t[y_true_t == 0]
    pos_scores = y_score_t[y_true_t == 1]

    if tpr_method == "harrell_davis":
        tpr = compute_empirical_roc_from_scores_hd(neg_scores, pos_scores, fpr_grid)
    else:
        tpr = compute_empirical_roc_from_scores(neg_scores, pos_scores, fpr_grid)

    # Compute corrected z critical value
    z = _compute_corrected_z(alpha, correction)

    # Wilson bounds for both dimensions
    fpr_lower, fpr_upper = _wilson_bounds_torch(fpr_grid, n_neg, z)
    tpr_lower, tpr_upper = _wilson_bounds_torch(tpr, n_pos, z)

    # Form envelope from rectangle corners
    # Upper envelope: upper-left corners (optimistic: low FPR, high TPR)
    upper_env_fpr = fpr_lower
    upper_env_tpr = tpr_upper

    # Lower envelope: lower-right corners (pessimistic: high FPR, low TPR)
    lower_env_fpr = fpr_upper
    lower_env_tpr = tpr_lower

    # Interpolate envelopes onto the FPR grid
    lower_envelope = _interpolate_envelope_to_grid(
        lower_env_fpr, lower_env_tpr, fpr_grid, fill_lower=True
    )
    upper_envelope = _interpolate_envelope_to_grid(
        upper_env_fpr, upper_env_tpr, fpr_grid, fill_lower=False
    )

    # Enforce boundary constraints
    lower_envelope[0] = 0.0
    upper_envelope[-1] = 1.0

    # Convert to numpy with original dtype
    fpr_grid_np = torch_to_numpy(fpr_grid).astype(dtype)
    lower_envelope_np = torch_to_numpy(lower_envelope).astype(dtype)
    upper_envelope_np = torch_to_numpy(upper_envelope).astype(dtype)

    return fpr_grid_np, lower_envelope_np, upper_envelope_np
