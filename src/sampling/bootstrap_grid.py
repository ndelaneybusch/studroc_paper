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
    # Ensure inputs are on device
    fpr = fpr.to(device)
    tpr = tpr.to(device)

    # Ensure we start at 0
    if fpr[0] != 0 or tpr[0] != 0:
        zero = torch.tensor([0.0], device=device, dtype=fpr.dtype)
        fpr = torch.cat((zero, fpr))
        tpr = torch.cat((zero, tpr))

    d_fpr = torch.diff(fpr)
    d_tpr = torch.diff(tpr)

    # Counts of negatives and positives in each segment
    # Use round to handle floating point inaccuracies
    counts_neg = (d_fpr * n_negatives).round().long()
    counts_pos = (d_tpr * n_positives).round().long()

    # Total ranks
    total_samples = n_negatives + n_positives

    # Vectorized construction of label sequence
    # We interleave: Pos then Neg for each segment.
    # We construct a flat tensor of values to repeat and a flat tensor of counts.
    # vals: 1 (Pos), 0 (Neg), 1, 0, ...
    # counts: counts_pos[0], counts_neg[0], counts_pos[1], counts_neg[1], ...

    # Stack counts: (N_segments, 2) -> flatten -> (2 * N_segments,)
    counts = torch.stack((counts_pos, counts_neg), dim=1).flatten()

    # Create values pattern: 1, 0 repeated N_segments times
    vals = torch.tensor([1, 0], device=device, dtype=torch.long).repeat(len(counts_pos))

    # Repeat values according to counts
    labels = torch.repeat_interleave(vals, counts)

    # Ranks: 0 to total_samples - 1
    # The first elements in 'labels' correspond to the highest scores (start of ROC).
    # So index 0 has rank N-1.

    # Indices of positives in `labels`
    # nonzero(as_tuple=True) returns tuple of tensors
    pos_indices = labels.nonzero(as_tuple=True)[0]
    neg_indices = (labels == 0).nonzero(as_tuple=True)[0]

    # Map indices to ranks
    # rank = (total_samples - 1) - index
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

    # Ensure inputs are on device
    fpr = fpr.to(device)
    tpr = tpr.to(device)
    grid = grid.to(device)

    # 1. Reconstruct implicit ranks from the empirical ROC
    pos_ranks, neg_ranks = reconstruct_ranks(fpr, tpr, n_negatives, n_positives, device)

    # 2. Bootstrap the ranks
    # Positives: (B, n1)
    idx_pos = torch.randint(0, n_positives, (B, n_positives), device=device)
    X_star = pos_ranks[idx_pos]

    # Negatives: (B, n0)
    idx_neg = torch.randint(0, n_negatives, (B, n_negatives), device=device)
    Y_star = neg_ranks[idx_neg]

    # 3. Determine thresholds for the grid
    # For a uniform grid u, we want the TPR at FPR = u.
    # This corresponds to the threshold defined by the k-th largest negative,
    # where k = floor(u * n0).

    # Sort Y_star descending
    Y_star_desc, _ = torch.sort(Y_star, dim=1, descending=True)

    # Compute indices k for each grid point
    k_indices = torch.floor(grid * n_negatives).long()
    k_indices = torch.clamp(k_indices, 0, n_negatives)

    # Pad Y_star_desc with sentinel -1 to handle k=n0 case (accept all negatives)
    sentinel = torch.full((B, 1), -1, device=device, dtype=Y_star.dtype)
    Y_star_padded = torch.cat([Y_star_desc, sentinel], dim=1)

    # Gather thresholds: (B, K)
    thresholds = Y_star_padded[:, k_indices]

    # 4. Compute TPR
    # TPR = (Count of X_star > threshold) / n1

    # Sort X_star ascending for searchsorted
    X_star_asc, _ = torch.sort(X_star, dim=1, descending=False)

    # searchsorted returns count of elements <= threshold
    counts_le = torch.searchsorted(X_star_asc, thresholds, right=True)
    counts_gt = n_positives - counts_le

    tpr_boot = counts_gt.float() / n_positives

    return tpr_boot
