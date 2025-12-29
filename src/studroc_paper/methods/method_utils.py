"""Shared utilities for PyTorch-based methods."""

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor


def numpy_to_torch(arr: NDArray | Tensor, device: torch.device | None = None) -> Tensor:
    """Convert numpy array or torch tensor to tensor on specified device.

    Args:
        arr: Input numpy array or torch tensor.
        device: Target device (defaults to CUDA if available).

    Returns:
        PyTorch tensor on the specified device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If already a tensor, just move to the target device
    if isinstance(arr, Tensor):
        return arr.to(device)

    # If numpy array, convert to tensor
    return torch.from_numpy(np.asarray(arr)).to(device)


def torch_to_numpy(tensor: Tensor | NDArray) -> NDArray:
    """Convert tensor or numpy array to numpy array.

    Args:
        tensor: Input PyTorch tensor or numpy array.

    Returns:
        Numpy array with preserved dtype.
    """
    # If already numpy array, return as-is
    if isinstance(tensor, np.ndarray):
        return tensor

    # If tensor, convert to numpy (moving to CPU if necessary)
    return tensor.detach().cpu().numpy()


def torch_step_interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """
    Step-function interpolation (right-continuous).

    For x between xp[j] and xp[j+1], returns fp[j].
    """
    # Find insertion indices
    indices = torch.searchsorted(xp, x, right=True) - 1
    indices = torch.clamp(indices, 0, len(fp) - 1)
    return fp[indices]


def torch_interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """Linear interpolation equivalent to np.interp.

    Args:
        x: X-coordinates at which to evaluate.
        xp: X-coordinates of data points (must be increasing).
        fp: Y-coordinates of data points.

    Returns:
        Interpolated values at x positions.
    """
    # Find indices where x would be inserted
    indices = torch.searchsorted(xp, x)

    # Clamp indices for boundary handling
    indices = torch.clamp(indices, 1, len(xp) - 1)

    # Get left and right points
    x0 = xp[indices - 1]
    x1 = xp[indices]
    y0 = fp[indices - 1]
    y1 = fp[indices]

    # Linear interpolation
    t = (x - x0) / (x1 - x0 + 1e-12)
    return y0 + t * (y1 - y0)


def compute_empirical_roc_from_scores(
    neg_scores: Tensor, pos_scores: Tensor, fpr_grid: Tensor
) -> Tensor:
    """Compute empirical ROC curve and interpolate at fpr_grid points.

    Vectorized computation using PyTorch.

    Args:
        neg_scores: Tensor of negative class scores.
        pos_scores: Tensor of positive class scores.
        fpr_grid: FPR values at which to evaluate TPR.

    Returns:
        TPR values at fpr_grid points.
    """
    device = neg_scores.device

    # Get thresholds from negative scores (sorted descending)
    # Note: We use all negative scores as candidate thresholds
    thresholds = torch.sort(neg_scores, descending=True).values

    n_neg = len(neg_scores)
    n_pos = len(pos_scores)

    # Vectorized computation: for each threshold, compute FPR and TPR
    # Shape: (n_thresholds,)
    # Uses broadcasting: (1, n_neg) >= (n_thresholds, 1) -> (n_thresholds, n_neg)
    # This is O(N^2) memory, which is fine for typical N (~1000s)
    # but be careful for very large N.

    # Expanding dimensions for broadcasting
    # neg_scores: (N_neg,) -> (1, N_neg)
    # thresholds: (N_neg,) -> (N_neg, 1)

    # Determine dtype from inputs
    dtype = neg_scores.dtype

    # Calculate FPR for each threshold
    # FPR = FP / N_neg = sum(neg_scores >= threshold) / N_neg
    # Note: sum() returns long for int/bool inputs, so we must cast to original dtype
    fpr_emp = (neg_scores.unsqueeze(0) >= thresholds.unsqueeze(1)).sum(dim=1).to(
        dtype=dtype
    ) / n_neg

    # Calculate TPR for each threshold
    # TPR = TP / N_pos = sum(pos_scores >= threshold) / N_pos
    tpr_emp = (pos_scores.unsqueeze(0) >= thresholds.unsqueeze(1)).sum(dim=1).to(
        dtype=dtype
    ) / n_pos

    # Add boundary points (0,0) and (1,1)
    fpr_emp = torch.cat(
        [
            torch.tensor([0.0], device=device, dtype=dtype),
            fpr_emp,
            torch.tensor([1.0], device=device, dtype=dtype),
        ]
    )
    tpr_emp = torch.cat(
        [
            torch.tensor([0.0], device=device, dtype=dtype),
            tpr_emp,
            torch.tensor([1.0], device=device, dtype=dtype),
        ]
    )

    # Interpolate at fpr_grid points (Step interpolation for empirical ROC)
    return torch_step_interp(fpr_grid, fpr_emp, tpr_emp)


# =============================================================================
# Wilson Score Interval Utilities
# =============================================================================


def wilson_halfwidth_squared_np(p: NDArray, n: int, z: float) -> NDArray:
    """Compute squared half-width of Wilson score interval (NumPy version).

    The Wilson score interval for a binomial proportion p is:
        p̃ ± (z / denom) * sqrt(p(1-p)/n + z²/(4n²))

    where denom = 1 + z²/n and p̃ is the Wilson-adjusted center.

    This function returns the squared half-width:
        (z / denom)² * (p(1-p)/n + z²/(4n²))

    This is used as a variance floor because:
    - It's always positive, even when p=0 or p=1 (where Wald variance is 0)
    - At boundaries: returns z⁴/(4n²(1 + z²/n)²) > 0
    - Provides a principled minimum uncertainty from exact binomial CI theory

    Args:
        p: Proportion estimates (e.g., TPR values).
        n: Sample size (e.g., number of positive samples).
        z: Critical value from normal distribution (e.g., z = Φ⁻¹(1 - α/2)).

    Returns:
        Squared half-width of Wilson interval at each p value (variance floor).
    """
    if n <= 0:
        return np.zeros_like(p)

    denom = 1 + z**2 / n
    return (z**2 / denom**2) * (p * (1 - p) / n + z**2 / (4 * n**2))


def wilson_halfwidth_squared_torch(p: Tensor, n: int, z: float) -> Tensor:
    """Compute squared half-width of Wilson score interval (PyTorch version).

    The Wilson score interval for a binomial proportion p is:
        p̃ ± (z / denom) * sqrt(p(1-p)/n + z²/(4n²))

    where denom = 1 + z²/n and p̃ is the Wilson-adjusted center.

    This function returns the squared half-width:
        (z / denom)² * (p(1-p)/n + z²/(4n²))

    This is used as a variance floor because:
    - It's always positive, even when p=0 or p=1 (where Wald variance is 0)
    - At boundaries: returns z⁴/(4n²(1 + z²/n)²) > 0
    - Provides a principled minimum uncertainty from exact binomial CI theory

    Args:
        p: Proportion estimates (e.g., TPR values) as tensor.
        n: Sample size (e.g., number of positive samples).
        z: Critical value from normal distribution (e.g., z = Φ⁻¹(1 - α/2)).

    Returns:
        Squared half-width of Wilson interval at each p value (variance floor).
    """
    if n <= 0:
        return torch.zeros_like(p)

    z_sq = z * z
    denom = 1 + z_sq / n
    return (z_sq / (denom * denom)) * (p * (1 - p) / n + z_sq / (4 * n * n))
