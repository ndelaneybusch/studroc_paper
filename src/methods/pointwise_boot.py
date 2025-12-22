"""Pointwise Bootstrap Confidence Bands using PyTorch."""

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .method_utils import numpy_to_torch, torch_to_numpy


def pointwise_bootstrap_band(
    boot_tpr_matrix: NDArray | Tensor, fpr_grid: NDArray | Tensor, alpha: float = 0.05
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute Pointwise Bootstrap Confidence Bands.

    Args:
        boot_tpr_matrix: (B, F) array where B is iterations, F is len(fpr_grid).
                         Contains TPR values interpolated at fpr_grid points.
        fpr_grid: (F,) array of FPR values corresponding to columns.
        alpha: Significance level.

    Returns:
        Tuple of (fpr_grid, lower_envelope, upper_envelope) as numpy arrays.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine dtype from boot_tpr_matrix (convert to numpy if needed to get dtype)
    if isinstance(boot_tpr_matrix, np.ndarray):
        dtype = boot_tpr_matrix.dtype
    elif isinstance(boot_tpr_matrix, torch.Tensor):
        dtype = boot_tpr_matrix.cpu().numpy().dtype
    else:
        dtype = np.asarray(boot_tpr_matrix).dtype

    # Convert inputs to tensors on the target device
    boot_tpr = numpy_to_torch(boot_tpr_matrix, device).float()
    fpr = numpy_to_torch(fpr_grid, device).float()

    # Calculate quantiles column-wise (dim 0)
    # torch.quantile uses [0, 1] range, not [0, 100]
    lower_q = alpha / 2.0
    upper_q = 1.0 - alpha / 2.0

    lower_envelope = torch.quantile(boot_tpr, lower_q, dim=0)
    upper_envelope = torch.quantile(boot_tpr, upper_q, dim=0)

    # Clip to [0, 1]
    lower_envelope = torch.clamp(lower_envelope, 0.0, 1.0)
    upper_envelope = torch.clamp(upper_envelope, 0.0, 1.0)

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
