import numpy as np


def _compute_empirical_roc(y_true, y_score, fpr_grid):
    """
    Compute empirical ROC curve and interpolate at fpr_grid points.

    Parameters:
    - y_true: Array of true binary labels (0 or 1)
    - y_score: Array of predicted scores
    - fpr_grid: FPR values at which to evaluate TPR

    Returns:
    - empirical_tpr: TPR values at fpr_grid points
    """
    y_score = np.asarray(y_score)
    dtype = y_score.dtype
    y_true = np.asarray(y_true)

    # Separate scores by class
    neg_scores = y_score[y_true == 0]
    pos_scores = y_score[y_true == 1]

    # Get unique thresholds from negative scores (defines FPR jumps)
    thresholds = np.sort(neg_scores)[::-1]  # Descending order

    # Compute empirical ROC curve
    n_neg = len(neg_scores)
    n_pos = len(pos_scores)

    # Add boundary points
    fpr_emp = [0.0]
    tpr_emp = [0.0]

    for threshold in thresholds:
        # FPR: fraction of negatives >= threshold
        fpr = np.sum(neg_scores >= threshold) / n_neg
        # TPR: fraction of positives >= threshold
        tpr = np.sum(pos_scores >= threshold) / n_pos
        fpr_emp.append(fpr)
        tpr_emp.append(tpr)

    # Add top-right corner
    fpr_emp.append(1.0)
    tpr_emp.append(1.0)

    fpr_emp = np.array(fpr_emp, dtype=dtype)
    tpr_emp = np.array(tpr_emp, dtype=dtype)

    # Interpolate at fpr_grid points (right-continuous step function)
    empirical_tpr = np.interp(fpr_grid, fpr_emp, tpr_emp).astype(dtype)

    return empirical_tpr


def envelope_bootstrap_band(boot_tpr_matrix, fpr_grid, y_true, y_score, alpha=0.05):
    """
    Computes Studentized Bootstrap Envelope Simultaneous Confidence Bands.

    This method retains the (1-α) fraction of bootstrap curves most consistent
    with the empirical ROC (ranked by studentized Kolmogorov-Smirnov statistic)
    and returns their pointwise envelope.

    Parameters:
    - boot_tpr_matrix: (n_bootstrap, n_grid_points) numpy array where n_bootstrap
                       is the number of bootstrap iterations and n_grid_points is
                       len(fpr_grid). Contains TPR values at fpr_grid points.
    - fpr_grid: (n_grid_points,) numpy array of FPR values.
    - y_true: Array of true binary labels (0 or 1) from original data.
    - y_score: Array of predicted scores from original data.
    - alpha: float, significance level (default 0.05).

    Returns:
    - fpr_grid: The input FPR vector.
    - lower_envelope: Lower envelope of retained bootstrap curves.
    - upper_envelope: Upper envelope of retained bootstrap curves.
    """
    # Ensure inputs are numpy arrays, prioritizing y_score dtype
    y_score = np.asarray(y_score)
    dtype = y_score.dtype
    boot_tpr_matrix = np.asarray(boot_tpr_matrix, dtype=dtype)
    fpr_grid = np.asarray(fpr_grid, dtype=dtype)
    y_true = np.asarray(y_true)

    n_bootstrap, n_grid_points = boot_tpr_matrix.shape

    # === Step 0: Compute empirical ROC and sample size ===
    n_total = len(y_true)
    empirical_tpr = _compute_empirical_roc(y_true, y_score, fpr_grid)

    # === Step 1: Variance Estimation ===
    # Compute standard deviation at each grid point
    bootstrap_std = np.std(boot_tpr_matrix, axis=0, ddof=1)  # ddof=1 for 1/(B-1)

    # === Step 2: Regularization Parameter ===
    # epsilon = min(1/n, 1e-6) where n = n0 + n1
    epsilon = min(1.0 / n_total, 1e-6)

    # === Step 3: Studentized KS Statistics ===
    ks_statistics = np.zeros(n_bootstrap, dtype=dtype)

    for boot_idx in range(n_bootstrap):
        max_studentized_dev = 0.0

        for grid_idx in range(n_grid_points):
            absolute_deviation = np.abs(
                boot_tpr_matrix[boot_idx, grid_idx] - empirical_tpr[grid_idx]
            )

            # Handle low variance cases with regularization
            if bootstrap_std[grid_idx] < epsilon:
                if absolute_deviation < epsilon:
                    studentized_deviation = 0.0
                else:
                    studentized_deviation = absolute_deviation / epsilon
            else:
                studentized_deviation = absolute_deviation / bootstrap_std[grid_idx]

            max_studentized_dev = max(max_studentized_dev, studentized_deviation)

        ks_statistics[boot_idx] = max_studentized_dev

    # === Step 4: Curve Retention ===
    # Retain the (1-alpha)*n_bootstrap curves with smallest KS statistics
    n_retain = int(np.floor((1 - alpha) * n_bootstrap))

    # Get the threshold (the n_retain-th order statistic)
    ks_sorted = np.sort(ks_statistics)
    threshold = ks_sorted[n_retain - 1] if n_retain > 0 else np.inf

    # Identify retained curves
    retained_indices = np.where(ks_statistics <= threshold)[0]

    # === Step 5: Envelope Construction ===
    # Compute pointwise min and max of retained curves
    retained_curves = boot_tpr_matrix[retained_indices, :]

    lower_envelope = np.min(retained_curves, axis=0)
    upper_envelope = np.max(retained_curves, axis=0)

    # === Step 6: Clip to [0,1] ===
    lower_envelope = np.clip(lower_envelope, 0.0, 1.0).astype(dtype)
    upper_envelope = np.clip(upper_envelope, 0.0, 1.0).astype(dtype)

    return fpr_grid, lower_envelope, upper_envelope
