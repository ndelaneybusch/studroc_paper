"""Fixed-Width KS Confidence Bands."""

import numpy as np
from numpy.typing import NDArray
from torch import Tensor


def fixed_width_ks_band(
    y_true: NDArray | Tensor,
    y_score: NDArray | Tensor,
    k: int = 1000,
    alpha: float = 0.05,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute Fixed-Width Simultaneous Confidence Bands (Campbell 1994).

    Args:
        y_true: Input binary labels (numpy array or torch tensor).
        y_score: Input prediction scores (numpy array or torch tensor).
        k: Grid resolution.
        alpha: Significance level.

    Returns:
        Tuple of (fpr_grid, lower_envelope, upper_envelope) as numpy arrays.
    """
    # Convert to numpy arrays, handling torch tensors (including CUDA)
    if isinstance(y_score, Tensor):
        y_score = y_score.detach().cpu().numpy()
        dtype = y_score.dtype
    else:
        y_score = np.asarray(y_score)
        dtype = y_score.dtype

    if isinstance(y_true, Tensor):
        y_true = y_true.detach().cpu().numpy()
    else:
        y_true = np.asarray(y_true)

    n0 = (y_true == 0).sum()
    n1 = (y_true == 1).sum()

    # Compute Empirical ROC - sort descending
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score_desc = y_score[desc_score_indices]
    y_true_desc = y_true[desc_score_indices]

    # Vectorized accumulation of TP and FP
    # Find indices where values change
    diff = y_score_desc[1:] - y_score_desc[:-1]
    distinct_value_indices = np.where(diff != 0)[0]

    # Include the last index
    threshold_idxs = np.concatenate([distinct_value_indices, [len(y_true_desc) - 1]])

    # Cumulative sum of true labels
    tps = np.cumsum(y_true_desc.astype(float))[threshold_idxs]
    fps = (1 + threshold_idxs.astype(float)) - tps

    # Add (0, 0) point
    tps = np.concatenate([[0.0], tps])
    fps = np.concatenate([[0.0], fps])

    emp_tpr = tps / tps[-1]
    emp_fpr = fps / fps[-1]

    # Critical Values (Kolmogorov-Smirnov with Bonferroni correction)
    alpha_adj = 1 - np.sqrt(1 - alpha)

    # Smirnov approximation for critical value
    c_alpha = np.sqrt(-0.5 * np.log(alpha_adj / 2))

    d = c_alpha / np.sqrt(n1)  # Vertical margin
    e = c_alpha / np.sqrt(n0)  # Horizontal margin

    # Evaluation grid
    fpr_grid = np.linspace(0, 1, k, dtype=dtype)

    # Compute bands
    # For upper envelope, we want the maximum TPR at a given FPR
    # (top of vertical segments). np.interp returns the value at the last
    # occurrence of x, which corresponds to the top.
    upper_envelope = np.interp(np.clip(fpr_grid + e, 0, 1), emp_fpr, emp_tpr) + d

    # For lower envelope, we want the minimum TPR at a given FPR
    # (bottom of vertical segments). We can achieve this by interpolating
    # on negated, reversed arrays.
    # emp_fpr is sorted ascending. -emp_fpr[::-1] is also sorted ascending.
    # emp_tpr[::-1] corresponds to those points.
    # np.interp will pick the last occurrence in the reversed array, which is the
    # first occurrence in the original array (bottom of segment).

    # We interpolate at -clip(fpr_grid - e)
    neg_fpr_rev = -emp_fpr[::-1]
    tpr_rev = emp_tpr[::-1]

    lower_interp = np.interp(-np.clip(fpr_grid - e, 0, 1), neg_fpr_rev, tpr_rev)
    lower_envelope = lower_interp - d

    # Clip to [0, 1]
    upper_envelope = np.clip(upper_envelope, 0, 1).astype(dtype)
    lower_envelope = np.clip(lower_envelope, 0, 1).astype(dtype)

    # Enforce boundary conditions
    # FPR=0: [0, upper]
    # FPR=1: [lower, 1]
    lower_envelope[0] = 0.0
    upper_envelope[-1] = 1.0

    return fpr_grid, lower_envelope, upper_envelope
