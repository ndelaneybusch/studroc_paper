"""Envelope Bootstrap Confidence Bands using PyTorch."""

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .method_utils import numpy_to_torch, torch_step_interp, torch_to_numpy


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


def envelope_bootstrap_band(
    boot_tpr_matrix: NDArray | Tensor,
    fpr_grid: NDArray | Tensor,
    y_true: NDArray | Tensor,
    y_score: NDArray | Tensor,
    alpha: float = 0.05,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute Studentized Bootstrap Envelope Simultaneous Confidence Bands.

    Retains the (1-α) fraction of bootstrap curves most consistent with the
    empirical ROC (ranked by studentized Kolmogorov-Smirnov statistic) and
    returns their pointwise envelope.

    Args:
        boot_tpr_matrix: (n_bootstrap, n_grid_points) array of TPR values.
        fpr_grid: (n_grid_points,) array of FPR values.
        y_true: Array of true binary labels (0 or 1) from original data.
        y_score: Array of predicted scores from original data.
        alpha: Significance level (default 0.05).

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

    # Step 0: Compute empirical ROC and sample size
    n_total = len(y_true_t)
    empirical_tpr = _compute_empirical_roc(y_true_t, y_score_t, fpr)

    # Step 1: Variance Estimation - std across bootstrap dimension
    bootstrap_std = torch.std(boot_tpr, dim=0, correction=1)

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

    # Convert back to numpy with original dtype
    return (
        torch_to_numpy(fpr).astype(dtype),
        torch_to_numpy(lower_envelope).astype(dtype),
        torch_to_numpy(upper_envelope).astype(dtype),
    )
