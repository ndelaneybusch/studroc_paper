"""Working-Hotelling Simultaneous Confidence Bands using PyTorch."""

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import chi2, norm
from torch import Tensor

from .method_utils import numpy_to_torch, torch_to_numpy


def working_hotelling_band(
    y_true: NDArray | Tensor,
    y_score: NDArray | Tensor,
    k: int = 1000,
    alpha: float = 0.05,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute Working-Hotelling simultaneous confidence bands for ROC curves.

    Uses the binormal assumption.

    Args:
        y_true: Array of binary class labels (0/1).
        y_score: Array of continuous scores.
        k: Number of evaluation points on the grid.
        alpha: Significance level (e.g., 0.05 for 95% confidence).

    Returns:
        Tuple of (fpr_grid, lower_envelope, upper_envelope) as numpy arrays.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to numpy first to get dtype, then to tensor
    y_score_np = np.asarray(y_score)
    dtype = y_score_np.dtype
    y_true_np = np.asarray(y_true)

    y_score_t = numpy_to_torch(y_score_np, device).float()
    y_true_t = numpy_to_torch(y_true_np, device)

    # Separate scores by class
    neg = y_score_t[y_true_t == 0]
    pos = y_score_t[y_true_t == 1]
    n0 = len(neg)
    n1 = len(pos)

    # Method of Moments estimates for Binormal Parameters
    mu0, s0 = neg.mean(), neg.std(correction=1)
    mu1, s1 = pos.mean(), pos.std(correction=1)

    # ROC parameters: a = (mu1 - mu0) / s1, b = s0 / s1
    a_hat = (mu1 - mu0) / s1
    b_hat = s0 / s1

    # Covariance matrix for (a, b) via Delta Method
    var_a = (1 / n1) + (b_hat**2 / n0) + (a_hat**2 / (2 * n1))
    var_b = (b_hat**2 / (2 * n0)) + (b_hat**2 / (2 * n1))
    cov_ab = (a_hat * b_hat) / (2 * n1)

    # Evaluation grid (probit space) - use scipy for ppf/cdf
    fpr_grid = torch.linspace(0, 1, k, device=device)
    fpr_clipped = torch.clamp(fpr_grid, 1e-9, 1 - 1e-9)

    # scipy.stats.norm for ppf (inverse CDF) - run on CPU
    x_probit = torch.tensor(
        norm.ppf(fpr_clipped.cpu().numpy()), device=device, dtype=torch.float32
    )

    # Standard error of fitted line in probit space
    se_probit = torch.sqrt(var_a + (x_probit**2 * var_b) + (2 * x_probit * cov_ab))

    # Working-Hotelling Critical Value (chi2 from scipy)
    W = float(np.sqrt(chi2.ppf(1 - alpha, df=2)))

    # Band in probit space
    y_probit_center = a_hat + b_hat * x_probit
    y_probit_lower = y_probit_center - W * se_probit
    y_probit_upper = y_probit_center + W * se_probit

    # Transform back to ROC space (TPR) using scipy.norm.cdf
    lower_envelope = torch.tensor(
        norm.cdf(y_probit_lower.cpu().numpy()), device=device, dtype=torch.float32
    )
    upper_envelope = torch.tensor(
        norm.cdf(y_probit_upper.cpu().numpy()), device=device, dtype=torch.float32
    )

    # Fix endpoints
    lower_envelope[0] = 0.0
    upper_envelope[-1] = 1.0

    # Convert back to numpy with original dtype
    return (
        torch_to_numpy(fpr_grid).astype(dtype),
        torch_to_numpy(lower_envelope).astype(dtype),
        torch_to_numpy(upper_envelope).astype(dtype),
    )
