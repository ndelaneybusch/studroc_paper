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
- Multiple boundary correction methods (Wilson, KS-style, density-based)
- Logit-space construction option for variance stabilization
- GPU acceleration for large bootstrap samples
- Diagnostic visualization integration
"""

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


def envelope_bootstrap_band(
    boot_tpr_matrix: NDArray | Tensor,
    fpr_grid: NDArray | Tensor,
    y_true: NDArray | Tensor,
    y_score: NDArray | Tensor,
    alpha: float = 0.05,
    boundary_method: BoundaryMethod = "none",
    retention_method: RetentionMethod = "ks",
    use_logit: bool = False,
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
            - "wilson": Use Wilson-score-based variance floor.
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
    empirical_tpr = _compute_empirical_roc(y_true_t, y_score_t, fpr)

    z_alpha = (2.0**0.5) * torch.erfinv(torch.tensor(1.0 - alpha)).item()

    if use_logit:
        # --- PATH A: LOGIT SPACE ENVELOPE ---
        # 1. Transform everything to Haldane-corrected Logit Space
        logit_tpr_hat = _haldane_logit(empirical_tpr, n_pos)
        logit_boot_tpr = _haldane_logit(boot_tpr, n_pos)

        # 2. Compute Bootstrap Variance in Logit Space
        bootstrap_var_logit = torch.var(logit_boot_tpr, dim=0, correction=1)

        # 3. Apply Variance Floors (transform from probability to logit space)
        if boundary_method == "wilson":
            variance_floor_prob = (
                wilson_halfwidth_squared_torch(empirical_tpr, n_pos, z_alpha)
                / z_alpha**2
            )
        elif boundary_method in ("reflected_kde", "kde", "log_concave"):
            neg_np = torch_to_numpy(y_score_t[y_true_t == 0])
            pos_np = torch_to_numpy(y_score_t[y_true_t == 1])
            fpr_np = torch_to_numpy(fpr)
            ht_var = compute_hsieh_turnbull_variance(
                neg_np, pos_np, fpr_np, method=boundary_method
            )
            variance_floor_prob = numpy_to_torch(ht_var, device).float()
        else:
            variance_floor_prob = torch.zeros_like(empirical_tpr)

        if boundary_method not in ("none", "ks"):
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
        if boundary_method == "wilson":
            variance_floor = (
                wilson_halfwidth_squared_torch(empirical_tpr, n_pos, z_alpha)
                / z_alpha**2
            )
        elif boundary_method in ("reflected_kde", "kde", "log_concave"):
            neg_np = torch_to_numpy(y_score_t[y_true_t == 0])
            pos_np = torch_to_numpy(y_score_t[y_true_t == 1])
            fpr_np = torch_to_numpy(fpr)
            ht_var = compute_hsieh_turnbull_variance(
                neg_np, pos_np, fpr_np, method=boundary_method
            )
            variance_floor = numpy_to_torch(ht_var, device).float()
        else:
            variance_floor = torch.zeros_like(empirical_tpr)

        if boundary_method not in ("none", "ks"):
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

        # Apply variance floor to envelope widths (only for standard path)
        if boundary_method not in ("none", "ks"):
            sigma_floor = torch.sqrt(variance_floor)
            upper_envelope = torch.maximum(
                upper_envelope, empirical_tpr + sigma_floor * z_alpha
            )
            lower_envelope = torch.minimum(
                lower_envelope, empirical_tpr - sigma_floor * z_alpha
            )

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
        if use_logit:
            # Project logit-space variance back to probability space for visualization
            # Delta_p approx p(1-p) * Delta_logit
            p = empirical_tpr
            deriv = p * (1 - p)
            bootstrap_var_np = torch_to_numpy((std_dev * deriv) ** 2).astype(dtype)
            variance_floor_np = torch_to_numpy(variance_floor_prob).astype(dtype)
        else:
            bootstrap_var_np = torch_to_numpy(bootstrap_var).astype(dtype)
            variance_floor_np = torch_to_numpy(variance_floor).astype(dtype)

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
