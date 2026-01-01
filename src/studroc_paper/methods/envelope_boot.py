"""Envelope Bootstrap Confidence Bands using PyTorch."""

import math
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from studroc_paper.viz import plot_band_diagnostics

from .method_utils import (
    compute_empirical_roc_from_scores,
    compute_hsieh_turnbull_variance,
    numpy_to_torch,
    torch_to_numpy,
    wilson_halfwidth_squared_torch,
)

# Type alias for boundary extension method selection
BoundaryMethod = Literal["none", "wilson", "reflected_kde", "log_concave", "ks"]

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
    # Separate scores by class
    neg_scores = y_score[y_true == 0]
    pos_scores = y_score[y_true == 1]

    return compute_empirical_roc_from_scores(neg_scores, pos_scores, fpr_grid)


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
    boundary_method: BoundaryMethod = "none",
    retention_method: RetentionMethod = "ks",
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
        alpha: Significance level (default 0.05).
        boundary_method: Method for handling zero-variance boundaries where
            bootstrap collapses. Options:
            - "wilson": Use Wilson-score-based variance floor (default).
              Provides a principled minimum variance based on binomial
              confidence intervals, ensuring non-degenerate bands at TPR=0/1.
            - "reflected_kde": Use Hsieh-Turnbull asymptotic variance with
              reflected KDE density estimation. Provides variance floor based
              on asymptotic ROC variance theory with ISJ bandwidth selection.
            - "log_concave": Use Hsieh-Turnbull asymptotic variance with
              log-concave MLE density estimation. Enforces shape constraints
              via convex optimization for robust density estimates.
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
        plot: If True, generate diagnostic plots using the viz module (default False).
        plot_title: Optional custom title for the diagnostic plots. If None, uses
            method description.

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

    # Step 1b: Compute variance floor based on boundary method
    z_alpha = (2.0**0.5) * torch.erfinv(torch.tensor(1.0 - alpha)).item()

    if boundary_method == "wilson":
        # Wilson-score variance floor
        variance_floor = wilson_halfwidth_squared_torch(
            empirical_tpr, n_pos, z_alpha
        ) / (z_alpha**2)  # this function returns the band - convert to variance.
    elif boundary_method in ("reflected_kde", "kde", "log_concave"):
        # Hsieh-Turnbull asymptotic variance floor
        # Convert to numpy for H-T computation
        neg_scores_np = torch_to_numpy(y_score_t[y_true_t == 0])
        pos_scores_np = torch_to_numpy(y_score_t[y_true_t == 1])
        fpr_np = torch_to_numpy(fpr)

        # Compute H-T variance with specified density estimation method
        ht_var_np = compute_hsieh_turnbull_variance(
            neg_scores_np, pos_scores_np, fpr_np, method=boundary_method
        )
        variance_floor = numpy_to_torch(ht_var_np, device).float()
    elif boundary_method == "ks":
        # KS method uses geometric extension later, no variance floor
        variance_floor = torch.zeros_like(empirical_tpr)
    else:  # "none"
        # No variance floor
        variance_floor = torch.zeros_like(empirical_tpr)

    # Apply variance floor to bootstrap variance
    if boundary_method not in ("none", "ks"):
        bootstrap_var = torch.maximum(bootstrap_var, variance_floor)
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

    # Step 5b: Apply variance floor to envelopes
    # Ensure minimum envelope width at boundaries where bootstrap collapses
    if boundary_method not in ("none", "ks"):
        sigma_floor = torch.sqrt(variance_floor)
        # Upper band should be at least center + floor half-width
        upper_envelope = torch.maximum(upper_envelope, empirical_tpr + sigma_floor)
        # Lower band should be at most center - floor half-width
        lower_envelope = torch.minimum(lower_envelope, empirical_tpr - sigma_floor)

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
    fpr_np = torch_to_numpy(fpr).astype(dtype)
    lower_np = torch_to_numpy(lower_envelope).astype(dtype)
    upper_np = torch_to_numpy(upper_envelope).astype(dtype)

    # Generate diagnostic plots if requested
    if plot:
        try:
            empirical_tpr_np = torch_to_numpy(empirical_tpr).astype(dtype)
            boot_tpr_np = torch_to_numpy(boot_tpr).astype(dtype)
            bootstrap_var_np = torch_to_numpy(bootstrap_var).astype(dtype)
            variance_floor_np = torch_to_numpy(variance_floor).astype(dtype)

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
