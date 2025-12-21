"""Working-Hotelling Simultaneous Confidence Bands."""

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import chi2, norm
from torch import Tensor


def working_hotelling_band(
    y_true: NDArray | Tensor,
    y_score: NDArray | Tensor,
    k: int = 1000,
    alpha: float = 0.05,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute Working-Hotelling simultaneous confidence bands for ROC curves.

    Uses the binormal assumption.

    Args:
        y_true: Array of binary class labels (0/1) (numpy array or torch tensor).
        y_score: Array of continuous scores (numpy array or torch tensor).
        k: Number of evaluation points on the grid.
        alpha: Significance level (e.g., 0.05 for 95% confidence).

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

    # Separate scores by class
    neg = y_score[y_true == 0]
    pos = y_score[y_true == 1]
    n0 = len(neg)
    n1 = len(pos)

    # Method of Moments estimates for Binormal Parameters
    mu0, s0 = neg.mean(), neg.std(ddof=1)
    mu1, s1 = pos.mean(), pos.std(ddof=1)

    # ROC parameters: a = (mu1 - mu0) / s1, b = s0 / s1
    a_hat = (mu1 - mu0) / s1
    b_hat = s0 / s1

    # Covariance matrix for (a, b) via Delta Method
    var_a = (1 / n1) + (b_hat**2 / n0) + (a_hat**2 / (2 * n1))
    var_b = (b_hat**2 / (2 * n0)) + (b_hat**2 / (2 * n1))
    cov_ab = (a_hat * b_hat) / (2 * n1)

    # Evaluation grid (probit space)
    fpr_grid = np.linspace(0, 1, k, dtype=dtype)
    fpr_clipped = np.clip(fpr_grid, 1e-9, 1 - 1e-9)

    # Inverse CDF (probit transform)
    x_probit = norm.ppf(fpr_clipped)

    # Standard error of fitted line in probit space
    se_probit = np.sqrt(var_a + (x_probit**2 * var_b) + (2 * x_probit * cov_ab))

    # Working-Hotelling Critical Value
    W = np.sqrt(chi2.ppf(1 - alpha, df=2))

    # Band in probit space
    y_probit_center = a_hat + b_hat * x_probit
    y_probit_lower = y_probit_center - W * se_probit
    y_probit_upper = y_probit_center + W * se_probit

    # Transform back to ROC space (TPR)
    lower_envelope = norm.cdf(y_probit_lower).astype(dtype)
    upper_envelope = norm.cdf(y_probit_upper).astype(dtype)

    # Fix endpoints
    lower_envelope[0] = 0.0
    upper_envelope[-1] = 1.0

    return fpr_grid, lower_envelope, upper_envelope
