"""
Bootstrap sampling for ROC curves with efficient grid evaluation.
"""

import torch
from torch import Tensor


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
    fpr: Tensor,
    tpr: Tensor,
    n_negatives: int,
    n_positives: int,
    B: int,
    grid: Tensor,
    device: torch.device | None = None,
    batch_size: int = 500,  # New Argument for memory safety
) -> Tensor:
    """
    Generate a tensor of bootstrapped ROC samples evaluated across a grid.

    Efficiently computes the ROC values at specified grid points without
    generating the full empirical ROC for each bootstrap sample.

    Args:
        fpr: Empirical FPR coordinates (sorted).
        tpr: Empirical TPR coordinates (sorted).
        n_negatives: Number of negative samples (n0).
        n_positives: Number of positive samples (n1).
        B: Number of bootstrap replicates.
        grid: Uniform evaluation grid for FPR (1D tensor).
        device: Target device. If None, uses CUDA if available.

    Returns:
        Tensor of shape (B, len(grid)) containing TPR values.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fpr = fpr.to(device)
    tpr = tpr.to(device)
    grid = grid.to(device)

    # 1. Reconstruct Ranks (One-time cost)
    pos_ranks, neg_ranks = reconstruct_ranks(fpr, tpr, n_negatives, n_positives, device)

    # Prepare Output
    tpr_boot_list = []

    # Pre-calculate grid indices for efficiency
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
        Y_star = neg_ranks[idx_neg]

        # Sort Descending (Highest Rank/Score first)
        Y_star, _ = torch.sort(Y_star, dim=1, descending=True)

        # Pad with -1 sentinel for the FPR=1.0 case (accept all)
        sentinel = torch.full((current_B, 1), -1, device=device, dtype=Y_star.dtype)
        Y_star = torch.cat([Y_star, sentinel], dim=1)

        # Extract Thresholds
        thresholds = Y_star[:, k_indices]

        # Free memory immediately
        del Y_star, idx_neg

        # --- Bootstrap Positives (X_star) ---
        # Sample
        idx_pos = torch.randint(0, n_positives, (current_B, n_positives), device=device)
        X_star = pos_ranks[idx_pos]

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
    return torch.cat(tpr_boot_list, dim=0)
