"""
Bootstrap sampling for ROC curves with efficient grid evaluation.
"""

from typing import Literal

import numpy as np
import torch
from scipy.stats import beta as beta_dist
from torch import Tensor


def _harrell_davis_weights(
    n: int, p: Tensor, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """
    Compute Harrell-Davis weights for quantile estimation.

    Uses beta-weighted average of order statistics for reduced bias.
    Weight for order statistic i is:
        P(i/n < U < (i+1)/n) where U ~ Beta(a, b)
        a = p * (n + 1), b = (1 - p) * (n + 1)

    Args:
        n: Sample size.
        p: Quantile probabilities, shape (K,).
        device: Target device.
        dtype: Target dtype.

    Returns:
        Weight matrix of shape (K, n) where each row sums to 1.
    """
    # Convert to numpy for scipy computation
    p_np = p.cpu().numpy()
    K = len(p_np)
    weights_np = np.zeros((K, n), dtype=np.float64)

    # Bin edges: 0, 1/n, 2/n, ..., 1
    bin_edges = np.arange(0, n + 1) / n

    # Identify boundary cases (p=0 or p=1) where Beta parameters would be invalid
    eps = 1e-9

    for i, prob in enumerate(p_np):
        if prob <= eps:
            # p=0 quantile: minimum order statistic (index 0 in ascending sort)
            weights_np[i, 0] = 1.0
        elif prob >= 1 - eps:
            # p=1 quantile: maximum order statistic (index n-1 in ascending sort)
            weights_np[i, -1] = 1.0
        else:
            # Interior case: use Beta distribution
            a = prob * (n + 1)
            b = (1 - prob) * (n + 1)
            cdf_vals = beta_dist.cdf(bin_edges, a, b)
            weights_np[i] = cdf_vals[1:] - cdf_vals[:-1]

    # Convert back to torch tensor on target device
    return torch.from_numpy(weights_np).to(device=device, dtype=dtype)


def reconstruct_ranks(
    fpr: Tensor, tpr: Tensor, n_negatives: int, n_positives: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """
    Reconstruct implicit score ranks from an empirical ROC curve.

    Assumes the ROC curve is formed by a set of scores where ties (if any)
    are broken optimistically (Positives > Negatives).

    Args:
        fpr: Empirical FPR coordinates (sorted).
        tpr: Empirical TPR coordinates (sorted).
        n_negatives: Number of negative samples.
        n_positives: Number of positive samples.
        device: Device to create tensors on.

    Returns:
        tuple (pos_ranks, neg_ranks):
            pos_ranks: Tensor of shape (n_positives,) containing reconstructed ranks.
            neg_ranks: Tensor of shape (n_negatives,) containing reconstructed ranks.
            Higher rank indicates higher score.
    """
    fpr = fpr.to(device)
    tpr = tpr.to(device)

    # 1. Handle (0,0) start
    if fpr[0] != 0 or tpr[0] != 0:
        zero = torch.tensor([0.0], device=device, dtype=fpr.dtype)
        fpr = torch.cat((zero, fpr))
        tpr = torch.cat((zero, tpr))

    d_fpr = torch.diff(fpr)
    d_tpr = torch.diff(tpr)

    # 2. Robust Count Construction
    # We enforce that the sum of counts equals N exactly by dumping
    # rounding errors into the largest segment (or last).
    counts_neg = (d_fpr * n_negatives).round().long()
    counts_pos = (d_tpr * n_positives).round().long()

    # Force sum to match exactly
    if counts_neg.sum() != n_negatives:
        diff = n_negatives - counts_neg.sum()
        # Add difference to the bin with the most samples to minimize distortion
        counts_neg[counts_neg.argmax()] += diff

    if counts_pos.sum() != n_positives:
        diff = n_positives - counts_pos.sum()
        counts_pos[counts_pos.argmax()] += diff

    # 3. Vectorized Label Construction
    # Stack: (Pos, Neg) per segment implies Optimistic Tie Breaking (Pos > Neg)
    counts = torch.stack((counts_pos, counts_neg), dim=1).flatten()

    # Optimization: Create vals on CPU then move to device only if small,
    # but here repeating on device is fast.
    vals = torch.tensor([1, 0], device=device, dtype=torch.long).repeat(len(counts_pos))
    labels = torch.repeat_interleave(vals, counts)

    # 4. Extract Ranks
    # Since labels are constructed High-Score first (index 0),
    # Rank = (Total - 1) - index
    total_samples = n_negatives + n_positives

    # Use nonzero to find indices (already sorted by virtue of 'labels' structure)
    pos_indices = labels.nonzero(as_tuple=True)[0]
    neg_indices = (labels == 0).nonzero(as_tuple=True)[0]

    pos_ranks = (total_samples - 1) - pos_indices
    neg_ranks = (total_samples - 1) - neg_indices

    return pos_ranks, neg_ranks


def generate_bootstrap_grid(
    fpr: Tensor | None = None,
    tpr: Tensor | None = None,
    n_negatives: int | None = None,
    n_positives: int | None = None,
    y_true: Tensor | None = None,
    y_score: Tensor | None = None,
    B: int | None = None,
    grid: Tensor = None,
    device: torch.device | None = None,
    batch_size: int = 500,
    tpr_method: Literal["empirical", "harrell_davis"] = "empirical",
) -> Tensor:
    """
    Generate a tensor of bootstrapped ROC samples evaluated across a grid.

    Efficiently computes the ROC values at specified grid points without
    generating the full empirical ROC for each bootstrap sample.

    Supports two usage patterns:
    A) Provide fpr, tpr, n_negatives, n_positives (reconstructs ranks from ROC curve)
    B) Provide y_true, y_score (uses scores directly, bypassing reconstruction)

    Args:
        fpr: Empirical FPR coordinates (sorted). Required for path A.
        tpr: Empirical TPR coordinates (sorted). Required for path A.
        n_negatives: Number of negative samples (n0). Required for path A.
        n_positives: Number of positive samples (n1). Required for path A.
        y_true: True binary labels (0 or 1). Required for path B.
        y_score: Predicted scores or probabilities. Required for path B.
        B: Number of bootstrap replicates.
        grid: Uniform evaluation grid for FPR (1D tensor).
        device: Target device. If None, uses CUDA if available.
        batch_size: Batch size for memory safety.
        tpr_method: Method for computing TPR from bootstrap samples. Defaults to
            "empirical" (standard step-function interpolation). Use "harrell_davis"
            for bias-reduced quantile estimation using beta-weighted order statistics.

    Returns:
        Tensor of shape (B, len(grid)) containing TPR values.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validate inputs
    path_a = (
        fpr is not None
        and tpr is not None
        and n_negatives is not None
        and n_positives is not None
    )
    path_b = y_true is not None and y_score is not None

    if not (path_a or path_b):
        raise ValueError(
            "Must provide either (fpr, tpr, n_negatives, n_positives) or (y_true, y_score)"
        )
    if path_a and path_b:
        raise ValueError(
            "Cannot provide both (fpr, tpr, n_negatives, n_positives) and (y_true, y_score)"
        )

    grid = grid.to(device)

    # 1. Get positive and negative values
    if path_b:
        # Path B: Direct from scores
        y_true = y_true.to(device)
        y_score = y_score.to(device)

        pos_mask = y_true == 1
        neg_mask = y_true == 0

        pos_values = y_score[pos_mask]
        neg_values = y_score[neg_mask]

        n_positives = len(pos_values)
        n_negatives = len(neg_values)
    else:
        # Path A: Reconstruct ranks from ROC curve
        fpr = fpr.to(device)
        tpr = tpr.to(device)
        pos_values, neg_values = reconstruct_ranks(
            fpr, tpr, n_negatives, n_positives, device
        )

    # Prepare Output
    tpr_boot_list = []

    use_hd = tpr_method == "harrell_davis"

    if use_hd:
        # Precompute HD weights for threshold estimation
        # HD quantile at p uses order statistics weighted by Beta(p*(n+1), (1-p)*(n+1))
        # For FPR=t, we want the (1-t) quantile of negatives (high threshold = low FPR)
        assert n_negatives is not None  # Guaranteed by path A/B validation
        quantile_probs = 1.0 - grid
        hd_weights = _harrell_davis_weights(
            n_negatives, quantile_probs, device, grid.dtype
        )  # Shape: (K, n_negatives)
    else:
        # Pre-calculate grid indices for efficiency (empirical path)
        # k is the index in the sorted negatives corresponding to the FPR grid point
        k_indices = torch.floor(grid * n_negatives).long()
        k_indices = torch.clamp(k_indices, 0, n_negatives)

    # 2. Process in Batches
    # This prevents allocating (B, N) tensors which causes OOM on large datasets
    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)
        current_B = end - start

        # --- Bootstrap Negatives (Y_star) ---
        # Sample
        idx_neg = torch.randint(0, n_negatives, (current_B, n_negatives), device=device)
        Y_star = neg_values[idx_neg]

        if use_hd:
            # Harrell-Davis path: sort ascending for weighted combination
            Y_star_sorted, _ = torch.sort(Y_star, dim=1, descending=False)

            # Threshold = weighted sum of order statistics
            # Shape: (current_B, n_negatives) @ (n_negatives, K) -> (current_B, K)
            thresholds = Y_star_sorted @ hd_weights.T

            del Y_star_sorted, Y_star, idx_neg
        else:
            # Empirical path: sort descending and index directly
            Y_star, _ = torch.sort(Y_star, dim=1, descending=True)

            # Pad with sentinel for the FPR=1.0 case (accept all)
            # Use -inf for scores (path B) or -1 for ranks (path A)
            sentinel_value = float("-inf") if path_b else -1
            sentinel = torch.full(
                (current_B, 1), sentinel_value, device=device, dtype=Y_star.dtype
            )
            Y_star = torch.cat([Y_star, sentinel], dim=1)

            # Extract Thresholds
            thresholds = Y_star[:, k_indices]

            # Free memory immediately
            del Y_star, idx_neg

        # --- Bootstrap Positives (X_star) ---
        # Sample
        idx_pos = torch.randint(0, n_positives, (current_B, n_positives), device=device)
        X_star = pos_values[idx_pos]

        # Sort Ascending for searchsorted
        X_star, _ = torch.sort(X_star, dim=1, descending=False)

        # --- Compute TPR ---
        # Count X_star > Threshold
        # searchsorted returns count <= Threshold
        counts_le = torch.searchsorted(X_star, thresholds, right=True)
        counts_gt = n_positives - counts_le

        batch_tpr = counts_gt.float() / n_positives
        tpr_boot_list.append(batch_tpr)

        # Free memory
        del X_star, idx_pos

    # Concatenate all batches
    tpr_boot_matrix = torch.cat(tpr_boot_list, dim=0)

    return tpr_boot_matrix
