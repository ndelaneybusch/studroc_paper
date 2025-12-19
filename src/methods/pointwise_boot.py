import numpy as np


def pointwise_bootstrap_band(boot_tpr_matrix, fpr_grid, alpha=0.05):
    """
    Computes Pointwise Bootstrap Confidence Bands.

    Parameters:
    - boot_tpr_matrix: (B, F) numpy array where B is iterations, F is len(fpr_grid).
                       Contains TPR values interpolated at fpr_grid points.
    - fpr_grid: (F,) numpy array of FPR values corresponding to columns.
    - alpha: float, significance level.

    Returns:
    - fpr_grid: The input FPR vector.
    - lower_envelope: 2.5th percentile (for alpha=0.05) at each point.
    - upper_envelope: 97.5th percentile (for alpha=0.05) at each point.
    """
    # Ensure inputs are numpy arrays, using boot_tpr_matrix dtype
    boot_tpr_matrix = np.asarray(boot_tpr_matrix)
    dtype = boot_tpr_matrix.dtype
    fpr_grid = np.asarray(fpr_grid, dtype=dtype)

    # Calculate percentiles column-wise (axis 0)
    # For 95% interval, we want 2.5th and 97.5th percentiles
    lower_p = (alpha / 2.0) * 100
    upper_p = (1 - alpha / 2.0) * 100

    lower_envelope = np.percentile(boot_tpr_matrix, lower_p, axis=0).astype(dtype)
    upper_envelope = np.percentile(boot_tpr_matrix, upper_p, axis=0).astype(dtype)

    # Enforce basic constraints (monotonicity is not guaranteed by pointwise quantiles,
    # but clipping is safe)
    lower_envelope = np.clip(lower_envelope, 0, 1).astype(dtype)
    upper_envelope = np.clip(upper_envelope, 0, 1).astype(dtype)

    return fpr_grid, lower_envelope, upper_envelope
