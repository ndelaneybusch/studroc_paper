"""Fixed-Width KS Confidence Bands using PyTorch."""

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .method_utils import numpy_to_torch, torch_interp, torch_to_numpy


def fixed_width_ks_band(
    y_true: NDArray | Tensor,
    y_score: NDArray | Tensor,
    k: int = 1000,
    alpha: float = 0.05,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute Fixed-Width Simultaneous Confidence Bands (Campbell 1994).

    Args:
        y_true: Input binary labels.
        y_score: Input prediction scores.
        k: Grid resolution.
        alpha: Significance level.

    Returns:
        Tuple of (fpr_grid, lower_envelope, upper_envelope) as numpy arrays.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert inputs
    y_score_np = np.asarray(y_score)
    dtype = y_score_np.dtype
    y_true_np = np.asarray(y_true)

    y_score_t = numpy_to_torch(y_score_np, device).float()
    y_true_t = numpy_to_torch(y_true_np, device)

    n0 = (y_true_t == 0).sum().item()
    n1 = (y_true_t == 1).sum().item()

    # Compute Empirical ROC - sort descending
    desc_score_indices = torch.argsort(y_score_t, descending=True)
    y_score_desc = y_score_t[desc_score_indices]
    y_true_desc = y_true_t[desc_score_indices]

    # Vectorized accumulation of TP and FP
    # Find indices where values change
    diff = y_score_desc[1:] - y_score_desc[:-1]
    distinct_value_indices = torch.where(diff != 0)[0]

    # Include the last index
    threshold_idxs = torch.cat(
        [distinct_value_indices, torch.tensor([y_true_desc.size(0) - 1], device=device)]
    )

    # Cumulative sum of true labels
    tps = torch.cumsum(y_true_desc.float(), dim=0)[threshold_idxs]
    fps = (1 + threshold_idxs.float()) - tps

    # Add (0, 0) point
    tps = torch.cat([torch.tensor([0.0], device=device), tps])
    fps = torch.cat([torch.tensor([0.0], device=device), fps])

    emp_tpr = tps / tps[-1]
    emp_fpr = fps / fps[-1]

    # Critical Values (Kolmogorov-Smirnov with Bonferroni correction)
    alpha_adj = 1 - np.sqrt(1 - alpha)

    # Smirnov approximation for critical value
    c_alpha = np.sqrt(-0.5 * np.log(alpha_adj / 2))

    d = c_alpha / np.sqrt(n1)  # Vertical margin
    e = c_alpha / np.sqrt(n0)  # Horizontal margin

    # Evaluation grid
    fpr_grid = torch.linspace(0, 1, k, device=device)

    # Compute bands using linear interpolation
    upper_envelope = torch_interp(torch.clamp(fpr_grid + e, 0, 1), emp_fpr, emp_tpr) + d
    lower_envelope = torch_interp(torch.clamp(fpr_grid - e, 0, 1), emp_fpr, emp_tpr) - d

    # Clip to [0, 1]
    upper_envelope = torch.clamp(upper_envelope, 0, 1)
    lower_envelope = torch.clamp(lower_envelope, 0, 1)

    # Convert back to numpy with original dtype
    return (
        torch_to_numpy(fpr_grid).astype(dtype),
        torch_to_numpy(lower_envelope).astype(dtype),
        torch_to_numpy(upper_envelope).astype(dtype),
    )
