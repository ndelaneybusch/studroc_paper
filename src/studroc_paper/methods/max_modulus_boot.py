"""Logit-Space Studentized Bootstrap Confidence Bands for ROC Curves."""

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
    """
    # Continuity-corrected proportion
    k = tpr * n_pos
    p_hat = (k + 0.5) / (n_pos + 1.0)

    # Asymptotic variance for log odds
    # Clamp p_hat slightly to ensure numerical stability, though +0.5 helps
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
    """Compute Simultaneous Confidence Bands using Logit-Space Studentization.

    This method addresses the "boundary collapse" problem of standard bootstrap ROC
    bands by transforming the bootstrap distribution into logit space using the
    Haldane-Anscombe correction. It constructs the band using the "Max-Modulus"
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
        alpha: Significance level (default 0.05).

    Returns:
        Tuple of (fpr_grid, lower_envelope, upper_envelope) as numpy arrays.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preserve dtype
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

    # Get sample size
    n_pos = int((y_true_t == 1).sum().item())

    # 1. Compute Empirical ROC (Center of the band)
    neg_scores = y_score_t[y_true_t == 0]
    pos_scores = y_score_t[y_true_t == 1]
    tpr_hat = compute_empirical_roc_from_scores(neg_scores, pos_scores, fpr)

    # 2. Transform to Logit Space (Haldane Correction)
    # We transform the center estimate and the bootstrap replicates
    logit_tpr_hat = _haldane_logit(tpr_hat, n_pos)
    logit_boot_tpr = _haldane_logit(boot_tpr, n_pos)

    # 3. Compute Asymptotic Standard Error in Logit Space
    # We use the SE of the empirical curve to studentize
    logit_se = _logit_std_error(tpr_hat, n_pos)

    # 4. Compute Studentized Deviations (Max-Modulus Statistic)
    # Z_b = (theta*_b - theta_hat) / sigma_hat
    # shape: (n_bootstrap, n_grid)
    deviations = (logit_boot_tpr - logit_tpr_hat.unsqueeze(0)) / logit_se.unsqueeze(0)

    # Compute the supremum of absolute deviations for each bootstrap replicate
    # D_b = max_k |Z_{b,k}|
    # shape: (n_bootstrap,)
    max_abs_deviations = torch.max(torch.abs(deviations), dim=1).values

    # 5. Determine Critical Value c_alpha
    # We want the (1 - alpha) quantile of the max deviations
    # This critical value ensures simultaneous coverage across the grid
    c_alpha = torch.quantile(max_abs_deviations, 1.0 - alpha)

    # 6. Construct Band in Logit Space
    # Band = theta_hat +/- c_alpha * sigma_hat
    margin = c_alpha * logit_se
    logit_lower = logit_tpr_hat - margin
    logit_upper = logit_tpr_hat + margin

    # 7. Transform Back to Probability Space (Sigmoid/Expit)
    lower_envelope = torch.sigmoid(logit_lower)
    upper_envelope = torch.sigmoid(logit_upper)

    # 8. Boundary Enforcement
    # The logit transform naturally respects (0,1), but we enforce
    # hard anchors at FPR=0 and FPR=1 for cleanliness.
    lower_envelope[0] = 0.0
    upper_envelope[-1] = 1.0

    # Ensure logical consistency (precision errors could cause upper < lower rarely)
    # though unlikely with symmetric margin
    upper_envelope = torch.maximum(upper_envelope, lower_envelope)

    # Convert to numpy
    fpr_np = torch_to_numpy(fpr).astype(dtype)
    lower_np = torch_to_numpy(lower_envelope).astype(dtype)
    upper_np = torch_to_numpy(upper_envelope).astype(dtype)

    return fpr_np, lower_np, upper_np
