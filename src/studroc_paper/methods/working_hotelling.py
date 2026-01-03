"""Working-Hotelling simultaneous confidence bands for ROC curves.

This module implements the Working-Hotelling method for constructing simultaneous
confidence bands around ROC curves under the binormal assumption. The method
assumes that the scores for positive and negative classes follow normal
distributions and uses the chi-squared distribution to control the family-wise
error rate across the entire curve.
"""

import numpy as np
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

    This function constructs simultaneous confidence bands that control the
    family-wise error rate across the entire ROC curve. The method assumes
    scores follow binormal distributions and uses the delta method to estimate
    parameter uncertainty.

    Args:
        y_true: Binary class labels (0 for negative, 1 for positive). Accepts
            numpy arrays or torch tensors (including CUDA tensors).
        y_score: Continuous prediction scores, with higher values indicating
            stronger predictions for the positive class. Accepts numpy arrays
            or torch tensors (including CUDA tensors).
        k: Number of uniformly-spaced evaluation points along the FPR axis.
            Must be positive. Defaults to 1000.
        alpha: Significance level for the confidence band. The resulting band
            has coverage probability 1-alpha. Must be in (0, 1). Defaults to 0.05.

    Returns:
        A tuple containing three numpy arrays of length k:
            - fpr_grid: False positive rates uniformly spaced from 0 to 1.
            - lower_envelope: Lower confidence bound on TPR at each FPR.
            - upper_envelope: Upper confidence bound on TPR at each FPR.

    Notes:
        The method fits a binormal ROC model using method of moments estimation,
        then constructs confidence bands in probit space using the Working-Hotelling
        approach. The critical value is derived from the chi-squared distribution
        with 2 degrees of freedom to ensure simultaneous coverage.

        The implementation includes safeguards against degenerate cases where
        sample sizes are small or variances are near-zero, replacing problematic
        values with small constants to prevent numerical failures.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_score = np.array([0.1, 0.4, 0.35, 0.8])
        >>> fpr, lower, upper = working_hotelling_band(y_true, y_score, k=100)
        >>> fpr.shape
        (100,)
        >>> lower.shape
        (100,)
        >>> upper.shape
        (100,)
        >>> lower[0]  # Lower bound at FPR=0
        0.0
        >>> upper[-1]  # Upper bound at FPR=1
        1.0
        >>> np.all(lower <= upper)
        True
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

    # Handle degenerate cases (n < 2 or constant values)
    epsilon = 1e-8
    if np.isnan(s0) or s0 < epsilon:
        s0 = epsilon
    if np.isnan(s1) or s1 < epsilon:
        s1 = epsilon

    # ROC parameters: a = (mu1 - mu0) / s1, b = s0 / s1
    a_hat = (mu1 - mu0) / s1
    b_hat = s0 / s1

    # Covariance matrix for (a, b) via Delta Method
    # Ensure all variance components are finite
    var_a = (1 / n1) + (b_hat**2 / n0) + (a_hat**2 / (2 * n1))
    var_b = (b_hat**2 / (2 * n0)) + (b_hat**2 / (2 * n1))
    cov_ab = (a_hat * b_hat) / (2 * n1)

    # Replace any NaN/inf variance components with fallback values
    if not np.isfinite(var_a):
        var_a = 1.0 / n1
    if not np.isfinite(var_b):
        var_b = 1.0 / n1
    if not np.isfinite(cov_ab):
        cov_ab = 0.0

    # Evaluation grid (probit space)
    fpr_grid = np.linspace(0, 1, k, dtype=dtype)
    fpr_clipped = np.clip(fpr_grid, 1e-9, 1 - 1e-9)

    # Inverse CDF (probit transform)
    x_probit = norm.ppf(fpr_clipped)

    # Standard error of fitted line in probit space
    variance_probit = var_a + (x_probit**2 * var_b) + (2 * x_probit * cov_ab)

    # Replace invalid variance with minimum plausible value
    invalid_mask = ~np.isfinite(variance_probit) | (variance_probit < 0)
    if np.any(invalid_mask):
        variance_probit[invalid_mask] = 1.0 / n1

    se_probit = np.sqrt(variance_probit)

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
