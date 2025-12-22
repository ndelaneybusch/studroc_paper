"""Envelope Bootstrap Confidence Bands using PyTorch."""

import math
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .method_utils import numpy_to_torch, torch_step_interp, torch_to_numpy

# Type alias for boundary extension method selection
BoundaryMethod = Literal["none", "wilson", "ks"]

# Type alias for curve retention method selection
RetentionMethod = Literal["ks", "symmetric"]


def _compute_empirical_roc(y_true: Tensor, y_score: Tensor, fpr_grid: Tensor) -> Tensor:
    """Compute empirical ROC curve and interpolate at fpr_grid points.

    Args:
        y_true: Tensor of true binary labels (0 or 1).
        y_score: Tensor of predicted scores.
        fpr_grid: FPR values at which to evaluate TPR.

    Returns:
        TPR values at fpr_grid points.
    """
    device = y_score.device

    # Separate scores by class
    neg_scores = y_score[y_true == 0]
    pos_scores = y_score[y_true == 1]

    # Get thresholds from negative scores (sorted descending)
    thresholds = torch.sort(neg_scores, descending=True).values

    n_neg = len(neg_scores)
    n_pos = len(pos_scores)

    # Vectorized computation: for each threshold, compute FPR and TPR
    # Shape: (n_thresholds,)
    fpr_emp = (neg_scores.unsqueeze(0) >= thresholds.unsqueeze(1)).sum(
        dim=1
    ).float() / n_neg
    tpr_emp = (pos_scores.unsqueeze(0) >= thresholds.unsqueeze(1)).sum(
        dim=1
    ).float() / n_pos

    # Add boundary points (0,0) and (1,1)
    fpr_emp = torch.cat(
        [
            torch.tensor([0.0], device=device),
            fpr_emp,
            torch.tensor([1.0], device=device),
        ]
    )
    tpr_emp = torch.cat(
        [
            torch.tensor([0.0], device=device),
            tpr_emp,
            torch.tensor([1.0], device=device),
        ]
    )

    # Sort by fpr (should already be sorted, but ensure for interp)
    sort_idx = torch.argsort(fpr_emp)
    fpr_emp = fpr_emp[sort_idx]
    tpr_emp = tpr_emp[sort_idx]

    # Interpolate at fpr_grid points
    return torch_step_interp(fpr_grid, fpr_emp, tpr_emp)


def _wilson_score_variance(empirical_tpr: Tensor, n_pos: int, alpha: float) -> Tensor:
    """Compute Wilson-score-based minimum variance for TPR at each grid point.

    The Wilson score interval provides non-zero confidence interval width even
    when the observed proportion is exactly 0 or 1. This function returns the
    variance implied by the Wilson interval, which serves as a principled
    floor for bootstrap variance estimation at boundaries.

    The Wilson score interval for a proportion p with n observations is:
        center = (p + z²/(2n)) / (1 + z²/n)
        half_width = z * sqrt(p(1-p)/n + z²/(4n²)) / (1 + z²/n)

    We return half_width² as the minimum variance.

    Args:
        empirical_tpr: TPR values at each grid point.
        n_pos: Number of positive samples (denominator for TPR).
        alpha: Significance level for the interval.

    Returns:
        Tensor of minimum variance values for each grid point.
    """
    # Normal quantile for (1-alpha/2) confidence
    # Using torch to stay on device
    z = torch.tensor(
        2.0**0.5 * torch.erfinv(torch.tensor(1.0 - alpha)).item(),
        device=empirical_tpr.device,
    )
    z_sq = z * z

    # Wilson score half-width for each TPR value
    # half_width = z * sqrt(p(1-p)/n + z²/(4n²)) / (1 + z²/n)
    p = empirical_tpr
    n = float(n_pos)

    numerator = z * torch.sqrt(p * (1 - p) / n + z_sq / (4 * n * n))
    denominator = 1 + z_sq / n
    half_width = numerator / denominator

    # Return variance (half_width squared)
    return half_width * half_width


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
    extend the confidence band using the same horizontal (e) and vertical (d)
    margins used in the fixed-width KS band (Campbell 1994).

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
    """
    # KS critical value (Smirnov approximation)
    # For two one-sided tests combined: alpha_adj = 1 - sqrt(1 - alpha)
    alpha_adj = 1.0 - math.sqrt(1.0 - alpha)
    c_alpha = math.sqrt(-0.5 * math.log(alpha_adj / 2))

    d = c_alpha / math.sqrt(n_pos)  # Vertical margin for TPR
    # Note: horizontal margin e = c_alpha / sqrt(n_neg) is not used here
    # because we only extend vertically (TPR), not horizontally (FPR)

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


def envelope_bootstrap_band(
    boot_tpr_matrix: NDArray | Tensor,
    fpr_grid: NDArray | Tensor,
    y_true: NDArray | Tensor,
    y_score: NDArray | Tensor,
    alpha: float = 0.05,
    boundary_method: BoundaryMethod = "wilson",
    retention_method: RetentionMethod = "ks",
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute Studentized Bootstrap Envelope Simultaneous Confidence Bands.

    Retains a subset of bootstrap curves based on their consistency with the
    empirical ROC and returns their pointwise envelope.

    Args:
        boot_tpr_matrix: (n_bootstrap, n_grid_points) array of TPR values.
        fpr_grid: (n_grid_points,) array of FPR values.
        y_true: Array of true binary labels (0 or 1) from original data.
        y_score: Array of predicted scores from original data.
        alpha: Significance level (default 0.05).
        boundary_method: Method for handling zero-variance boundaries where
            bootstrap collapses. Options:
            - "wilson": Use Wilson-score-based variance floor (default).
              Provides a principled minimum variance based on binomial
              confidence intervals, ensuring non-degenerate bands at TPR=0/1.
            - "ks": Use KS-style margin extension (Campbell 1994).
              Extends the band from interior points to corners using
              horizontal/vertical margins based on sample sizes.
            - "none": No boundary correction (original behavior).
        retention_method: Method for selecting which bootstrap curves to retain.
            Options:
            - "ks": Retain (1-α) curves with smallest studentized KS statistic
              (maximum absolute deviation from empirical). Default.
            - "symmetric": Trim α/2 from curves that deviate most upward and
              α/2 from curves that deviate most downward. This addresses
              asymmetric alpha mass at high AUC where positive deviations
              are bounded by 1 but negative deviations are not.

    Returns:
        Tuple of (fpr_grid, lower_envelope, upper_envelope) as numpy arrays.
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
    empirical_tpr = _compute_empirical_roc(y_true_t, y_score_t, fpr)

    # Step 1: Variance Estimation - std across bootstrap dimension
    bootstrap_std = torch.std(boot_tpr, dim=0, correction=1)
    bootstrap_var = bootstrap_std * bootstrap_std

    # Step 1b: Apply Wilson-score variance floor if requested
    if boundary_method == "wilson":
        wilson_var = _wilson_score_variance(empirical_tpr, n_pos, alpha)
        bootstrap_var = torch.maximum(bootstrap_var, wilson_var)
        bootstrap_std = torch.sqrt(bootstrap_var)

    # Step 2: Regularization Parameter
    epsilon = min(1.0 / n_total, 1e-6)

    # Step 3: Studentized KS Statistics (FULLY VECTORIZED)
    # Compute absolute deviations: (n_bootstrap, n_grid_points)
    absolute_deviations = torch.abs(boot_tpr - empirical_tpr.unsqueeze(0))

    # Create regularized std for division
    low_var_mask = bootstrap_std < epsilon

    # For points with low variance: if deviation < epsilon, result is 0,
    # else deviation/epsilon
    studentized_deviations = torch.zeros_like(absolute_deviations)

    # Handle normal variance points
    normal_mask = ~low_var_mask
    if normal_mask.any():
        studentized_deviations[:, normal_mask] = (
            absolute_deviations[:, normal_mask] / bootstrap_std[normal_mask]
        )

    # Handle low variance points
    if low_var_mask.any():
        low_var_devs = absolute_deviations[:, low_var_mask]
        # Where deviation < epsilon, set to 0; otherwise divide by epsilon
        result = torch.where(
            low_var_devs < epsilon,
            torch.zeros_like(low_var_devs),
            low_var_devs / epsilon,
        )
        studentized_deviations[:, low_var_mask] = result

    # Max studentized deviation across grid for each bootstrap (KS statistic)
    ks_statistics = torch.max(studentized_deviations, dim=1).values

    # Step 4: Curve Retention
    if retention_method == "symmetric":
        # === Symmetric Tail Trimming ===
        # Instead of absolute deviations, use signed deviations to trim each tail
        # This fixes asymmetric alpha mass when TPR is near boundaries (e.g., high AUC)

        # Compute signed studentized deviations (not absolute)
        signed_deviations = boot_tpr - empirical_tpr.unsqueeze(0)  # (B, K)

        # Studentize the signed deviations
        signed_studentized = torch.zeros_like(signed_deviations)
        if normal_mask.any():
            signed_studentized[:, normal_mask] = (
                signed_deviations[:, normal_mask] / bootstrap_std[normal_mask]
            )
        if low_var_mask.any():
            # For low variance points, use epsilon as denominator
            signed_studentized[:, low_var_mask] = (
                signed_deviations[:, low_var_mask] / epsilon
            )

        # For each curve: max deviation upward and max deviation downward
        max_above = signed_studentized.max(dim=1).values  # Most positive
        max_below = signed_studentized.min(dim=1).values  # Most negative

        # Trim α/2 from curves that go too far up
        upper_threshold = torch.quantile(max_above, 1.0 - alpha / 2)
        # Trim α/2 from curves that go too far down
        lower_threshold = torch.quantile(max_below, alpha / 2)

        # Retain curves that don't exceed either threshold
        retained_mask = (max_above <= upper_threshold) & (max_below >= lower_threshold)

    else:
        # === Original KS-based Retention ===
        n_retain = int(np.floor((1 - alpha) * n_bootstrap))

        # Get threshold (n_retain-th order statistic)
        ks_sorted = torch.sort(ks_statistics).values
        threshold = ks_sorted[n_retain - 1] if n_retain > 0 else float("inf")

        # Identify retained curves
        retained_mask = ks_statistics <= threshold

    retained_curves = boot_tpr[retained_mask]

    # Step 5: Envelope Construction
    lower_envelope = torch.min(retained_curves, dim=0).values
    upper_envelope = torch.max(retained_curves, dim=0).values

    # Step 6: Clip to [0, 1]
    lower_envelope = torch.clamp(lower_envelope, 0.0, 1.0)
    upper_envelope = torch.clamp(upper_envelope, 0.0, 1.0)

    # Step 6b: Apply KS-style boundary extension if requested
    if boundary_method == "ks":
        lower_envelope, upper_envelope = _extend_boundary_ks_style(
            fpr, lower_envelope, upper_envelope, empirical_tpr, n_neg, n_pos, alpha
        )

    # Enforce boundary conditions
    # FPR=0: [0, upper]
    # FPR=1: [lower, 1]
    lower_envelope[0] = 0.0
    upper_envelope[-1] = 1.0

    # Convert back to numpy with original dtype
    return (
        torch_to_numpy(fpr).astype(dtype),
        torch_to_numpy(lower_envelope).astype(dtype),
        torch_to_numpy(upper_envelope).astype(dtype),
    )
