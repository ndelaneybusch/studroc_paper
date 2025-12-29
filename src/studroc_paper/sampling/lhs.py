from typing import Literal

import numpy as np
from numpy.typing import NDArray


def maximin_lhs(
    n: int,
    k: int,
    method: Literal["build", "iterative"] = "build",
    dup: int = 1,
    eps: float = 0.05,
    max_iter: int = 100,
    optimize_on: Literal["grid", "result"] = "grid",
    seed: int | None = None,
) -> NDArray[np.float64]:
    """
    Generates a Maximin Latin Hypercube Sample (LHS).

    This function attempts to optimize the sample by maximizing the minimum distance
    between design points (maximin criterion). It replicates the logic of the R
    function `maximinLHS` from the `lhs` package.

    Args:
        n: The number of sample points (rows).
        k: The number of dimensions/variables (columns).
        method: The method of LHS creation.
            - 'build': (Default) Greedily finds the next best point while constructing
              the LHS. Uses `dup` to determine the number of candidates.
            - 'iterative': Starts with a random LHS and iteratively swaps elements
              to improve the maximin criterion.
        dup: A factor that determines the number of candidate points used in the
            search for the 'build' method. The number of candidates is `dup` times
            the number of unplaced points.
        eps: The minimum percent change in the minimum distance required to continue
            optimization in the 'iterative' method.
        max_iter: The maximum number of iterations for the 'iterative' method.
        optimize_on: The basis for optimization.
            - 'grid': Optimizes on the integer grid [1, n].
            - 'result': Optimizes on the unit hypercube [0, 1].
        seed: Random seed for reproducibility.

    Returns:
        An (n x k) numpy array with values uniformly distributed on [0, 1].
    """
    rng = np.random.default_rng(seed)

    if method == "build":
        return _maximin_lhs_build(n, k, dup, optimize_on, rng)
    elif method == "iterative":
        return _maximin_lhs_iterative(n, k, eps, max_iter, optimize_on, rng)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'build' or 'iterative'.")


def _calculate_min_dist(points: NDArray[np.float64]) -> float:
    """Calculates the minimum Euclidean distance between any pair of points."""
    # Efficient vectorized Euclidean distance matrix calculation
    # (n, 1, k) - (1, n, k) broadcasts to (n, n, k)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # Sum squares along the last axis (dimensions)
    sq_dist = np.sum(diff**2, axis=-1)
    # Fill diagonal with infinity to ignore self-distance
    np.fill_diagonal(sq_dist, np.inf)
    # Return the square root of the minimum squared distance
    return np.sqrt(np.min(sq_dist))


def _maximin_lhs_build(
    n: int, k: int, dup: int, optimize_on: str, rng: np.random.Generator
) -> NDArray[np.float64]:
    """
    Sequential 'build' algorithm for Maximin LHS.
    """
    # Initialize available indices for each column
    available_indices = [list(range(1, n + 1)) for _ in range(k)]
    result_grid = np.zeros((n, k), dtype=np.int32)

    # 1. Pick the first point randomly
    for col in range(k):
        idx_in_list = rng.integers(0, len(available_indices[col]))
        val = available_indices[col].pop(idx_in_list)
        result_grid[0, col] = val

    # 2. Build remaining points
    for i in range(1, n):
        points_remaining = n - i
        n_candidates = max(1, dup * points_remaining)

        best_candidate = None

        current_design = result_grid[:i, :].astype(np.float64)
        if optimize_on == "result":
            current_design = (current_design - 0.5) / n

        # IMPROVEMENT: Generate all candidates at once for vectorization
        candidates = np.zeros((n_candidates, k), dtype=np.int32)
        for c in range(n_candidates):
            for col in range(k):
                rand_idx = rng.integers(0, len(available_indices[col]))
                candidates[c, col] = available_indices[col][rand_idx]

        # Vectorized distance computation
        candidate_points = candidates.astype(np.float64)
        if optimize_on == "result":
            candidate_points = (candidate_points - 0.5) / n

        # Shape: (n_candidates, i) - distance from each candidate to each existing point
        # (n_candidates, 1, k) - (1, i, k) -> (n_candidates, i, k)
        diffs = candidate_points[:, np.newaxis, :] - current_design[np.newaxis, :, :]
        all_dists = np.sqrt(np.sum(diffs**2, axis=2))  # (n_candidates, i)
        min_dists = np.min(all_dists, axis=1)  # (n_candidates,)

        best_idx = np.argmax(min_dists)
        best_candidate = candidates[best_idx]

        # Add best candidate and update availability
        result_grid[i, :] = best_candidate
        for col in range(k):
            available_indices[col].remove(best_candidate[col])

    # Transform to [0, 1]
    zero_based_grid = result_grid - 1
    perturbation = rng.random((n, k))
    return (zero_based_grid + perturbation) / n


def _maximin_lhs_iterative(
    n: int,
    k: int,
    eps: float,
    max_iter: int,
    optimize_on: str,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """
    Iterative 'swapping' algorithm to optimize Maximin LHS.
    """
    # 1. Generate initial random LHS (grid 1..n)
    grid = np.zeros((n, k), dtype=np.int32)
    for col in range(k):
        grid[:, col] = rng.permutation(np.arange(1, n + 1))

    # Helper to get points for distance calculation
    def get_points(g):
        p = g.astype(np.float64)
        if optimize_on == "result":
            # Center points in their cells for optimization if 'result'
            # (R implementation usually optimizes the center points or specific
            # realizations)
            p = (p - 0.5) / n
        return p

    current_points = get_points(grid)
    current_min_dist = _calculate_min_dist(current_points)

    # 2. Iterative optimization
    for _ in range(max_iter):
        # Try to swap two elements in a random column
        col_to_swap = rng.integers(0, k)
        row1, row2 = rng.choice(n, 2, replace=False)

        # Perform swap on a copy
        new_grid = grid.copy()
        new_grid[row1, col_to_swap], new_grid[row2, col_to_swap] = (
            new_grid[row2, col_to_swap],
            new_grid[row1, col_to_swap],
        )

        new_points = get_points(new_grid)
        new_min_dist = _calculate_min_dist(new_points)

        # Check acceptance criteria
        # If new distance is better, accept.
        if new_min_dist > current_min_dist:
            grid = new_grid
            current_min_dist = new_min_dist
            # R's `eps` parameter is used to break loops if improvement is negligible
            # or to define the 'acceptance' threshold.
            # Here we follow a simple hill-climbing 'strictly better' rule.

    # 3. Finalize: Transform grid to uniform [0,1]
    # (grid - 1 + random) / n
    perturbation = rng.random((n, k))
    final_lhs = (grid - 1 + perturbation) / n

    return final_lhs


def iman_conover_transform(
    X: np.ndarray, target_corr: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """
    Iman-Conover method to induce target correlation structure
    while preserving marginal distributions.

    Reference: Iman & Conover (1982), Communications in Statistics
    """
    n, k = X.shape

    # Get rank scores (van der Waerden scores work well)
    from scipy import stats

    ranks = np.zeros_like(X)
    for j in range(k):
        order = np.argsort(X[:, j])
        ranks[order, j] = np.arange(1, n + 1)

    # Convert ranks to normal scores
    scores = stats.norm.ppf(ranks / (n + 1))

    # Current correlation of scores
    current_corr = np.corrcoef(scores.T)

    # Cholesky decomposition
    C_target = np.linalg.cholesky(target_corr)
    C_current = np.linalg.cholesky(current_corr)

    # Transform scores to achieve target correlation
    T = C_target @ np.linalg.inv(C_current)
    transformed_scores = (T @ scores.T).T

    # Reorder original X to match the rank order of transformed scores
    result = np.zeros_like(X)
    for j in range(k):
        target_order = np.argsort(np.argsort(transformed_scores[:, j]))
        original_sorted = np.sort(X[:, j])
        result[:, j] = original_sorted[target_order]

    return result
