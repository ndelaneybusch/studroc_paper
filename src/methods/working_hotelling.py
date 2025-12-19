import numpy as np
from scipy.stats import chi2, norm


def working_hotelling_band(y_true, y_score, k=1000, alpha=0.05):
    """
    Computes Working-Hotelling simultaneous confidence bands for ROC curves
    under the binormal assumption.

    Parameters:
    - y_true: array-like of binary class labels (0/1).
    - y_score: array-like of continuous scores.
    - k: int, number of evaluation points on the grid.
    - alpha: float, significance level (e.g., 0.05 for 95% confidence).

    Returns:
    - fpr_grid: array of k uniform FPR points.
    - lower_envelope: array of lower bound TPR values.
    - upper_envelope: array of upper bound TPR values.
    """
    y_score = np.asarray(y_score)
    dtype = y_score.dtype
    y_true = np.asarray(y_true)

    # Separate scores
    neg = y_score[y_true == 0]
    pos = y_score[y_true == 1]
    n0 = len(neg)
    n1 = len(pos)

    # Method of Moments estimates for Binormal Parameters
    # We assume scores X ~ N(mu0, s0^2) and Y ~ N(mu1, s1^2)
    mu0, s0 = np.mean(neg), np.std(neg, ddof=1)
    mu1, s1 = np.mean(pos), np.std(pos, ddof=1)

    # ROC parameters: a = (mu1 - mu0) / s1, b = s0 / s1
    a_hat = (mu1 - mu0) / s1
    b_hat = s0 / s1

    # Covariance matrix for (a, b) derived via Delta Method
    # Variances of sample moments: Var(mu) = s^2/n, Var(s) = s^2/(2n)
    # Derivatives:
    # da/dmu1 = 1/s1, da/dmu0 = -1/s1, da/ds1 = -(mu1-mu0)/s1^2 = -a/s1
    # db/ds0 = 1/s1,  db/ds1 = -s0/s1^2 = -b/s1

    var_a = (1 / n1) + (b_hat**2 / n0) + (a_hat**2 / (2 * n1))
    var_b = (b_hat**2 / (2 * n0)) + (b_hat**2 / (2 * n1))
    cov_ab = (a_hat * b_hat) / (2 * n1)

    # Evaluation grid (probit space)
    # Avoid 0 and 1 strictly to prevent infinity
    fpr_grid = np.linspace(0, 1, k, dtype=dtype)
    fpr_clipped = np.clip(fpr_grid, 1e-9, 1 - 1e-9)
    x_probit = norm.ppf(fpr_clipped).astype(dtype)

    # Compute standard error of the fitted line in probit space at each x
    # Var(a + bx) = Var(a) + x^2*Var(b) + 2x*Cov(a,b)
    se_probit = np.sqrt(
        var_a + (x_probit**2 * var_b) + (2 * x_probit * cov_ab)
    ).astype(dtype)

    # Working-Hotelling Critical Value
    # Using Chi-squared approximation for simultaneous region (Ma & Hall 1993)
    # W = sqrt(chi2_2, 1-alpha)
    W = np.sqrt(chi2.ppf(1 - alpha, df=2))

    # Band in probit space
    y_probit_center = a_hat + b_hat * x_probit
    y_probit_lower = y_probit_center - W * se_probit
    y_probit_upper = y_probit_center + W * se_probit

    # Transform back to ROC space (TPR)
    lower_envelope = norm.cdf(y_probit_lower).astype(dtype)
    upper_envelope = norm.cdf(y_probit_upper).astype(dtype)

    # Fix endpoints
    lower_envelope[0] = dtype.type(0.0)
    upper_envelope[-1] = dtype.type(1.0)

    return fpr_grid, lower_envelope, upper_envelope
