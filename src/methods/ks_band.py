import numpy as np


def fixed_width_ks_band(y_true, y_score, k=1000, alpha=0.05):
    """
    Computes Fixed-Width Simultaneous Confidence Bands (Campbell 1994).

    Parameters:
    - y_true, y_score: Input data.
    - k: grid resolution.
    - alpha: significance level.

    Returns:
    - fpr_grid, lower_envelope, upper_envelope
    """
    y_score = np.asarray(y_score)
    dtype = y_score.dtype
    y_true = np.asarray(y_true)
    n0 = np.sum(y_true == 0)
    n1 = np.sum(y_true == 1)

    # Compute Empirical ROC
    # Sort descending
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score_desc = y_score[desc_score_indices]
    y_true_desc = y_true[desc_score_indices]

    # Vectorized accumulation of TP and FP
    distinct_value_indices = np.where(np.diff(y_score_desc))
    threshold_idxs = np.r_[distinct_value_indices, y_true_desc.size - 1]

    tps = np.cumsum(y_true_desc)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps

    # Add (0,0) point
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    emp_tpr = tps / tps[-1]
    emp_fpr = fps / fps[-1]

    # Critical Values (Kolmogorov-Smirnov)
    # We apply Bonferroni correction for the two independent dimensions (FP and TP)
    # so we use (1 - sqrt(1-alpha)) for each dimension to maintain global alpha.
    alpha_adj = 1 - np.sqrt(1 - alpha)

    # Smirnov approximation for critical value D_n,alpha
    # D approx sqrt(-0.5 * ln(alpha/2)) * sqrt((n+m)/nm)?
    # Standard approximation: c(alpha) / sqrt(n)
    # c(0.05) ~ 1.36. We calculate c based on alpha_adj.
    c_alpha = np.sqrt(-0.5 * np.log(alpha_adj / 2))

    d = c_alpha / np.sqrt(n1)  # Vertical margin
    e = c_alpha / np.sqrt(n0)  # Horizontal margin

    # Evaluation Grid
    fpr_grid = np.linspace(0, 1, k, dtype=dtype)

    # The upper band U(t) is the maximum TPR reachable from any point (fpr, tpr)
    # such that the horizontal distance |t - fpr| <= e.
    # Because Empirical ROC is monotonic, this simplifies to:
    # U(t) = R_emp(min(1, t + e)) + d
    # L(t) = R_emp(max(0, t - e)) - d

    # We use linear interpolation for R_emp to handle the shifts
    # Note: interp needs sorted x, emp_fpr is sorted ascending
    upper_envelope = (
        np.interp(np.clip(fpr_grid + e, 0, 1), emp_fpr, emp_tpr) + d
    ).astype(dtype)
    lower_envelope = (
        np.interp(np.clip(fpr_grid - e, 0, 1), emp_fpr, emp_tpr) - d
    ).astype(dtype)

    # Clip to [0,1]
    upper_envelope = np.clip(upper_envelope, 0, 1).astype(dtype)
    lower_envelope = np.clip(lower_envelope, 0, 1).astype(dtype)

    return fpr_grid, lower_envelope, upper_envelope
