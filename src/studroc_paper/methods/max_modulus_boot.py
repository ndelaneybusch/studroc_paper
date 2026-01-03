"""Max-Modulus Bootstrap Confidence Bands for ROC Curves.

This module implements simultaneous confidence bands using the max-modulus method
with logit-space studentization. The approach addresses boundary collapse issues
in standard bootstrap ROC bands by applying Haldane-Anscombe correction.
"""

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .method_utils import (
    compute_empirical_roc_from_scores,
    numpy_to_torch,
    torch_to_numpy,
)


def _compute_empirical_roc(y_true: Tensor, y_score: Tensor, fpr_grid: Tensor) -> Tensor:
    """Compute empirical ROC curve (TPR) at specified FPR grid points.

    Args:
        y_true: Tensor of true binary labels (0 or 1).
        y_score: Tensor of predicted scores.
        fpr_grid: FPR values at which to evaluate TPR.

    Returns:
        TPR values at fpr_grid points.

    Examples:
        >>> y_true = torch.tensor([0, 0, 1, 1])
        >>> y_score = torch.tensor([0.1, 0.4, 0.35, 0.8])
        >>> fpr_grid = torch.tensor([0.0, 0.5, 1.0])
        >>> _compute_empirical_roc(y_true, y_score, fpr_grid)
        tensor([0.0, 0.5, 1.0])
    """
    neg_scores = y_score[y_true == 0]
    pos_scores = y_score[y_true == 1]
    return compute_empirical_roc_from_scores(neg_scores, pos_scores, fpr_grid)


def _haldane_logit(tpr: Tensor, n_pos: int) -> Tensor:
    """Apply Logit transform with Haldane-Anscombe correction.

    The correction (+0.5) ensures finite values at TPR=0 and TPR=1.
    Formula: log((k + 0.5) / (n - k + 0.5)) where k = TPR * n.

    Args:
        tpr: TPR values (0 to 1).
        n_pos: Number of positive samples.

    Returns:
        Logit-transformed values.

    Examples:
        >>> tpr = torch.tensor([0.0, 0.5, 1.0])
        >>> _haldane_logit(tpr, n_pos=10)
        tensor([-2.9444, 0.0000, 2.9444])
    """
    k = tpr * n_pos
    numerator = k + 0.5
    denominator = n_pos - k + 0.5
    return torch.log(numerator / denominator)


def _logit_std_error(tpr: Tensor, n_pos: int) -> Tensor:
    """Compute asymptotic standard error in Haldane-corrected logit space.

    Formula: 1 / sqrt(n * p_hat * (1 - p_hat))
    Where p_hat is the continuity-corrected proportion: (k + 0.5) / (n + 1).

    Args:
        tpr: TPR values (0 to 1).
        n_pos: Number of positive samples.

    Returns:
        Estimated standard error tensor.

    Examples:
        >>> tpr = torch.tensor([0.5])
        >>> _logit_std_error(tpr, n_pos=100)
        tensor([0.2000])
    """
    # Continuity-corrected proportion
    k = tpr * n_pos
    p_hat = (k + 0.5) / (n_pos + 1.0)

    # Asymptotic variance for log odds
    # Clamping ensures numerical stability at extreme values
    p_hat = torch.clamp(p_hat, 1e-6, 1.0 - 1e-6)
    variance = 1.0 / (n_pos * p_hat * (1.0 - p_hat))

    return torch.sqrt(variance)


def logit_bootstrap_band(
    boot_tpr_matrix: NDArray | Tensor,
    fpr_grid: NDArray | Tensor,
    y_true: NDArray | Tensor,
    y_score: NDArray | Tensor,
    alpha: float = 0.05,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute simultaneous confidence bands using logit-space studentization.

    This method addresses the boundary collapse problem of standard bootstrap ROC
    bands by transforming the bootstrap distribution into logit space using the
    Haldane-Anscombe correction. It constructs the band using the max-modulus
    method, where a single critical value is derived from the supremum of the
    studentized deviations.

    The resulting band is:
        logit(TPR_hat) +/- c_alpha * SE_logit
    Transformed back to probability space via expit.

    Args:
        boot_tpr_matrix: (n_bootstrap, n_grid_points) array of bootstrap TPR values.
        fpr_grid: (n_grid_points,) array of FPR values.
        y_true: Array of true binary labels (0 or 1).
        y_score: Array of predicted scores.
        alpha: Significance level. Defaults to 0.05.

    Returns:
        Tuple of (fpr_grid, lower_envelope, upper_envelope) as numpy arrays.

    Examples:
        >>> import numpy as np
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_score = np.array([0.1, 0.4, 0.35, 0.8])
        >>> fpr_grid = np.array([0.0, 0.5, 1.0])
        >>> boot_tpr = np.array([[0.0, 0.5, 1.0], [0.0, 0.6, 1.0]])
        >>> fpr, lower, upper = logit_bootstrap_band(
        ...     boot_tpr, fpr_grid, y_true, y_score
        ... )
        >>> fpr.shape
        (3,)
        >>> lower.shape == upper.shape == (3,)
        True
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preserve dtype for output arrays
    if isinstance(y_score, np.ndarray):
        dtype = y_score.dtype
    elif isinstance(y_score, torch.Tensor):
        dtype = y_score.cpu().numpy().dtype
    else:
        dtype = np.asarray(y_score).dtype

    # Convert inputs to tensors
    boot_tpr = numpy_to_torch(boot_tpr_matrix, device).float()
    fpr = numpy_to_torch(fpr_grid, device).float()
    y_true_t = numpy_to_torch(y_true, device)
    y_score_t = numpy_to_torch(y_score, device).float()

    n_pos = int((y_true_t == 1).sum().item())

    # Compute empirical ROC curve (center of the band)
    neg_scores = y_score_t[y_true_t == 0]
    pos_scores = y_score_t[y_true_t == 1]
    tpr_hat = compute_empirical_roc_from_scores(neg_scores, pos_scores, fpr)

    # Transform to logit space with Haldane correction
    logit_tpr_hat = _haldane_logit(tpr_hat, n_pos)
    logit_boot_tpr = _haldane_logit(boot_tpr, n_pos)

    # Compute asymptotic standard error in logit space
    logit_se = _logit_std_error(tpr_hat, n_pos)

    # Compute studentized deviations (max-modulus statistic)
    # Z_b = (theta*_b - theta_hat) / sigma_hat
    deviations = (logit_boot_tpr - logit_tpr_hat.unsqueeze(0)) / logit_se.unsqueeze(0)

    # Compute supremum of absolute deviations for each bootstrap replicate
    # D_b = max_k |Z_{b,k}|
    max_abs_deviations = torch.max(torch.abs(deviations), dim=1).values

    # Determine critical value c_alpha
    # (1 - alpha) quantile ensures simultaneous coverage across the grid
    c_alpha = torch.quantile(max_abs_deviations, 1.0 - alpha)

    # Construct band in logit space: theta_hat +/- c_alpha * sigma_hat
    margin = c_alpha * logit_se
    logit_lower = logit_tpr_hat - margin
    logit_upper = logit_tpr_hat + margin

    # Transform back to probability space using sigmoid
    lower_envelope = torch.sigmoid(logit_lower)
    upper_envelope = torch.sigmoid(logit_upper)

    # Enforce boundary anchors at FPR endpoints
    lower_envelope[0] = 0.0
    upper_envelope[-1] = 1.0

    # Ensure logical consistency (prevent numerical precision errors)
    upper_envelope = torch.maximum(upper_envelope, lower_envelope)

    # Convert to numpy arrays with original dtype
    fpr_np = torch_to_numpy(fpr).astype(dtype)
    lower_np = torch_to_numpy(lower_envelope).astype(dtype)
    upper_np = torch_to_numpy(upper_envelope).astype(dtype)

    return fpr_np, lower_np, upper_np
