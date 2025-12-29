"""
Bootstrap resource allocation and grid construction for studentized ROC envelope.

This module implements optimal allocation of computational budget between bootstrap
replicates (B) and grid resolution (K) for computing simultaneous confidence bands
via the studentized bootstrap envelope method.
"""

from typing import Literal, TypedDict

import numpy as np
from numpy.typing import DTypeLike, NDArray
from scipy.stats import norm


# Type definitions for return values
class ComparisonMethod(TypedDict):
    """Details for a single allocation method."""

    feasible: bool
    B: int | None
    K: int | None
    error: float | None


class UniformComparisonMethod(ComparisonMethod):
    """Details for uniform grid allocation including regime information."""

    regime: Literal["unconstrained", "B-constrained", "K-constrained"] | None


class AllocationComparison(TypedDict):
    """Comparison between full and uniform grid methods."""

    full: ComparisonMethod
    uniform: UniformComparisonMethod


class AllocationParams(TypedDict):
    """Fundamental parameters used in budget allocation."""

    alpha: float
    beta: float
    D: float
    B_min: int
    K_min: int
    n0: int
    n1: int
    C: int


class AllocationResult(TypedDict):
    """Complete result of budget allocation optimization."""

    status: Literal["ok", "insufficient_budget"]
    method: Literal["full", "uniform"] | None
    B: int | None
    K: int | None
    grid_spec: str | None
    delta_B: float | None
    delta_K: float | None
    total_error: float | None
    error_ratio: float | None
    comparison: AllocationComparison
    params: AllocationParams
    message: str | None


def compute_beta(alpha: float) -> float:
    """
    Compute bootstrap error coefficient from quantile estimation theory.

    The coefficient β relates bootstrap sampling error to the number of replicates
    via δ_B = β / √B. It derives from the asymptotic variance of order statistics.

    Parameters
    ----------
    alpha : float
        Significance level (e.g., 0.05 for 95% confidence)

    Returns
    -------
    float
        Bootstrap error coefficient β

    Notes
    -----
    Formula: β = √(α(1-α)) / φ(Φ^{-1}(1-α))
    where φ is the standard normal PDF and Φ^{-1} is the inverse CDF.

    Typical values:
    - α = 0.10 → β ≈ 1.71
    - α = 0.05 → β ≈ 2.12
    - α = 0.01 → β ≈ 3.69

    Examples
    --------
    >>> compute_beta(0.05)
    2.12...
    """
    quantile = norm.ppf(1 - alpha)
    pdf_at_quantile = norm.pdf(quantile)
    beta = np.sqrt(alpha * (1 - alpha)) / pdf_at_quantile
    return beta


def estimate_D(
    fpr: NDArray[np.floating],
    tpr: NDArray[np.floating],
    n_negatives: int,
    n_positives: int,
) -> float:
    """
    Estimate discretization sensitivity from empirical ROC curve.

    D captures how much the supremum KS statistic can be underestimated when
    the evaluation grid misses jump points in the ROC curve. It combines
    the number of potential peaks (n_negatives) with the signal-to-noise ratio.

    Parameters
    ----------
    fpr : ndarray
        False positive rates (empirical ROC curve)
    tpr : ndarray
        True positive rates (empirical ROC curve)
    n_negatives : int
        Number of negative samples (controls FPR granularity)
    n_positives : int
        Number of positive samples (controls TPR granularity)

    Returns
    -------
    float
        Discretization sensitivity D

    Notes
    -----
    The estimation strategy:
    1. Find maximum TPR jump (where large deviations can occur)
    2. Estimate noise level σ in ROC interior using variance formula:
       Var[R(t)] ≈ R(1-R)/n₁ + R'²·t(1-t)/n₀
    3. Combine: D = n₀ × max_jump / σ

    For sparse or degenerate ROCs, falls back to theoretical estimate.
    """
    # Maximum TPR jump (where large deviations can occur)
    tpr_jumps = np.diff(tpr)
    max_jump = np.max(tpr_jumps) if len(tpr_jumps) > 0 else 1.0 / n_positives

    # Estimate noise level in ROC interior (avoid boundary effects)
    interior_mask = (fpr > 0.2) & (fpr < 0.8)

    if np.sum(interior_mask) >= 10:
        tpr_interior = tpr[interior_mask]
        fpr_interior = fpr[interior_mask]

        # Local slopes (ROC derivative estimate)
        with np.errstate(divide="ignore", invalid="ignore"):
            slopes = np.diff(tpr_interior) / np.diff(fpr_interior)
        slopes = slopes[np.isfinite(slopes) & (slopes > 0)]

        if len(slopes) > 0:
            roc_midpoint = np.median(tpr_interior)
            slope_midpoint = np.median(slopes)
            # Var[R(t)] ≈ R(1-R)/n₁ + R'²·t(1-t)/n₀
            variance_binomial = roc_midpoint * (1 - roc_midpoint) / n_positives
            variance_slope = slope_midpoint**2 * 0.25 / n_negatives  # t(1-t) ≤ 0.25
            noise_level = np.sqrt(variance_binomial + variance_slope)
        else:
            # Fallback: theoretical estimate at midpoint
            noise_level = np.sqrt(0.25 / n_positives + 0.25 / n_negatives)
    else:
        # Sparse interior: use harmonic mean of sample sizes
        effective_sample_size = (
            2 * n_negatives * n_positives / (n_negatives + n_positives)
        )
        noise_level = 0.5 / np.sqrt(effective_sample_size)

    # D ≈ (number of potential peaks) × (peak height / noise)
    discretization_sensitivity = n_negatives * max_jump / noise_level
    return discretization_sensitivity


def estimate_D_theoretical(n_negatives: int, n_positives: int) -> float:
    """
    Theoretical discretization sensitivity estimate.

    Use this when the ROC curve is not yet available, such as in planning
    or pre-computation stages. Makes conservative assumptions suitable for
    typical classifiers.

    Parameters
    ----------
    n_negatives : int
        Number of negative samples
    n_positives : int
        Number of positive samples

    Returns
    -------
    float
        Theoretical D estimate

    Notes
    -----
    Assumptions:
    1. Continuous scores (no ties): max_jump = 1/n₁
       If ties exist, actual max_jump may be larger, making this conservative.

    2. Worst-case variance: Evaluated at R = 0.5 where binomial variance
       R(1-R)/n₁ is maximized. Conservative for most classifiers (AUC > 0.5).

    3. Negligible slope contribution: Uses only R(1-R)/n₁ term, ignoring
       R'²·t(1-t)/n₀. Valid for moderate ROC slopes.

    4. Balanced effective sample size: n_eff = 2n₀n₁/(n₀+n₁) (harmonic mean).

    Formula:
        max_jump = 1/n₁
        σ = 0.5 / √n_eff = 0.5 × √((n₀+n₁)/(2n₀n₁))
        D = n₀ × max_jump / σ
          = 2n₀ × √(2n₀ / (n₁(n₀+n₁)))

    When to use empirical estimate_D() instead:
    - ROC curve is already computed
    - Known high AUC (> 0.9) where slope term matters
    - Known ties in positive scores
    - Class imbalance ratio exceeds 10:1

    Examples
    --------
    >>> estimate_D_theoretical(1000, 1000)
    44.72...
    """
    discriminant = 2 * n_negatives / (n_positives * (n_negatives + n_positives))
    discretization_sensitivity = 2 * n_negatives * np.sqrt(discriminant)
    return discretization_sensitivity


def _infer_dtype(
    dtype: DTypeLike | None,
    fpr: NDArray[np.floating] | None,
    tpr: NDArray[np.floating] | None,
) -> np.dtype:
    """
    Infer the appropriate dtype for memory calculations.

    Priority order:
    1. Explicit dtype parameter
    2. dtype from fpr array
    3. dtype from tpr array
    4. Default to float32

    Parameters
    ----------
    dtype : dtype-like or None
        Explicitly requested dtype
    fpr : ndarray or None
        FPR array to infer dtype from
    tpr : ndarray or None
        TPR array to infer dtype from

    Returns
    -------
    np.dtype
        The inferred dtype
    """
    if dtype is not None:
        return np.dtype(dtype)
    if fpr is not None:
        return fpr.dtype
    if tpr is not None:
        return tpr.dtype
    return np.dtype(np.float32)


def _gigabytes_to_elements(gigabytes: float, dtype: np.dtype) -> int:
    """
    Convert memory budget from gigabytes to number of array elements.

    Parameters
    ----------
    gigabytes : float
        Memory budget in GB
    dtype : np.dtype
        Data type of array elements

    Returns
    -------
    int
        Number of elements that fit in the budget
    """
    bytes_per_element = dtype.itemsize
    total_bytes = gigabytes * (1024**3)
    num_elements = int(total_bytes / bytes_per_element)
    return num_elements


def allocate_budget(
    n_negatives: int,
    n_positives: int,
    budget_gb: float,
    alpha: float = 0.05,
    fpr: NDArray[np.floating] | None = None,
    tpr: NDArray[np.floating] | None = None,
    dtype: DTypeLike | None = None,
) -> AllocationResult:
    """
    Allocate memory budget between bootstrap replicates and grid resolution.

    Determines the optimal number of bootstrap replicates B and grid points K
    to minimize total error E = √(δ_B² + δ_K²) subject to memory constraint
    B × K ≤ C, where C is derived from the budget in gigabytes.

    Parameters
    ----------
    n_negatives : int
        Number of negative samples (n₀)
    n_positives : int
        Number of positive samples (n₁)
    budget_gb : float
        Memory budget in gigabytes for storing B ROC curves with K points each
    alpha : float, default=0.05
        Significance level for confidence bands
    fpr : ndarray, optional
        Empirical ROC false positive rates for data-driven D estimation
    tpr : ndarray, optional
        Empirical ROC true positive rates for data-driven D estimation
    dtype : dtype-like, optional
        Data type for memory calculations. If None, inferred from fpr/tpr
        or defaults to float32.

    Returns
    -------
    AllocationResult
        Dictionary with keys:
        - status: "ok" or "insufficient_budget"
        - method: "full" or "uniform" (if status is "ok")
        - B: Number of bootstrap replicates
        - K: Number of grid points
        - grid_spec: Human-readable grid description
        - delta_B: Bootstrap sampling error component
        - delta_K: Discretization error component
        - total_error: Total error √(δ_B² + δ_K²)
        - error_ratio: δ_K / δ_B (measures balance)
        - comparison: Detailed comparison of full vs uniform methods
        - params: All fundamental parameters used
        - message: Error message if status is "insufficient_budget"

    Notes
    -----
    Decision rule: Full grid is optimal when (n₀+1)³ < 27D²C/(4β²)
    Equivalently: n₀+1 < 1.5·K_opt

    Full grid:
        - Uses all n₀+1 jump points (exact supremum, δ_K = 0)
        - B = C / (n₀+1)
        - Best for small datasets

    Uniform grid:
        - Optimizes B_opt = (β²C²/2D²)^(1/3), K_opt = (2D²C/β²)^(1/3)
        - At optimum: δ_B = √2·δ_K
        - Best for large datasets

    Constraints:
        - B ≥ B_min = max(500, ⌈100/α⌉)
        - K ≥ K_min = 100

    Examples
    --------
    Small dataset (full grid optimal):
    >>> result = allocate_budget(
    ...     n_negatives=1000, n_positives=1000, budget_gb=0.04
    ... )  # ~10M elements at float32
    >>> result["method"]
    'full'
    >>> result["K"]
    1001

    Large dataset (uniform grid optimal):
    >>> result = allocate_budget(
    ...     n_negatives=50000, n_positives=50000, budget_gb=0.4
    ... )  # ~100M elements
    >>> result["method"]
    'uniform'
    """
    # === Infer dtype and convert budget ===
    array_dtype = _infer_dtype(dtype, fpr, tpr)
    budget_elements = _gigabytes_to_elements(budget_gb, array_dtype)

    # === Fundamental parameters ===
    beta = compute_beta(alpha)

    if fpr is not None and tpr is not None:
        discretization_sensitivity = estimate_D(fpr, tpr, n_negatives, n_positives)
    else:
        discretization_sensitivity = estimate_D_theoretical(n_negatives, n_positives)

    min_bootstrap_replicates = max(500, int(np.ceil(100 / alpha)))
    min_grid_points = 100

    # === Full grid analysis ===
    full_grid_size = n_negatives + 1
    full_bootstrap_replicates = budget_elements // full_grid_size
    full_is_feasible = full_bootstrap_replicates >= min_bootstrap_replicates

    if full_is_feasible:
        full_error = beta / np.sqrt(full_bootstrap_replicates)
    else:
        full_error = np.inf

    # === Uniform grid optimization ===
    # Unconstrained optimum from Lagrange multipliers
    unconstrained_B = (
        beta**2 * budget_elements**2 / (2 * discretization_sensitivity**2)
    ) ** (1 / 3)
    unconstrained_K = budget_elements / unconstrained_B

    # Apply constraints
    if (
        unconstrained_B >= min_bootstrap_replicates
        and unconstrained_K >= min_grid_points
    ):
        uniform_bootstrap_replicates = unconstrained_B
        uniform_grid_points = unconstrained_K
        optimization_regime = "unconstrained"
    elif unconstrained_B < min_bootstrap_replicates:
        # B-constrained: hit minimum bootstrap replicates
        uniform_bootstrap_replicates = min_bootstrap_replicates
        uniform_grid_points = budget_elements / min_bootstrap_replicates
        optimization_regime = "B-constrained"
    else:
        # K-constrained: hit minimum grid points
        uniform_grid_points = min_grid_points
        uniform_bootstrap_replicates = budget_elements / min_grid_points
        optimization_regime = "K-constrained"

    # Check uniform feasibility
    uniform_is_feasible = (
        uniform_bootstrap_replicates >= min_bootstrap_replicates
        and uniform_grid_points >= min_grid_points
    )

    if uniform_is_feasible:
        uniform_bootstrap_error = beta / np.sqrt(uniform_bootstrap_replicates)
        uniform_discretization_error = discretization_sensitivity / uniform_grid_points
        uniform_total_error = np.sqrt(
            uniform_bootstrap_error**2 + uniform_discretization_error**2
        )
    else:
        uniform_total_error = np.inf
        optimization_regime = None

    # === Method selection ===
    if not full_is_feasible and not uniform_is_feasible:
        minimum_required_budget = min_bootstrap_replicates * min(
            min_grid_points, full_grid_size
        )
        return AllocationResult(
            status="insufficient_budget",
            method=None,
            B=None,
            K=None,
            grid_spec=None,
            delta_B=None,
            delta_K=None,
            total_error=None,
            error_ratio=None,
            comparison=AllocationComparison(
                full=ComparisonMethod(
                    feasible=full_is_feasible,
                    B=int(full_bootstrap_replicates) if full_is_feasible else None,
                    K=int(full_grid_size),
                    error=float(full_error) if full_is_feasible else None,
                ),
                uniform=UniformComparisonMethod(
                    feasible=uniform_is_feasible,
                    B=int(np.round(uniform_bootstrap_replicates))
                    if uniform_is_feasible
                    else None,
                    K=int(np.round(uniform_grid_points))
                    if uniform_is_feasible
                    else None,
                    error=float(uniform_total_error) if uniform_is_feasible else None,
                    regime=optimization_regime,
                ),
            ),
            params=AllocationParams(
                alpha=alpha,
                beta=beta,
                D=discretization_sensitivity,
                B_min=min_bootstrap_replicates,
                K_min=min_grid_points,
                n0=n_negatives,
                n1=n_positives,
                C=budget_elements,
            ),
            message=f"Budget {budget_gb:.3f} GB too small. Minimum required: "
            f"{minimum_required_budget * array_dtype.itemsize / (1024**3):.3f} GB",
        )

    # Decision rule: full grid wins if error is lower (or only feasible option)
    use_full_grid = full_is_feasible and (
        not uniform_is_feasible or full_error <= uniform_total_error
    )

    if use_full_grid:
        selected_method = "full"
        selected_B = int(full_bootstrap_replicates)
        selected_K = int(full_grid_size)
        bootstrap_error = beta / np.sqrt(selected_B)
        discretization_error = 0.0
        total_error = bootstrap_error
        grid_description = f"All {selected_K} jump points (exact evaluation)"
    else:
        selected_method = "uniform"
        selected_B = int(np.round(uniform_bootstrap_replicates))
        selected_K = int(np.round(uniform_grid_points))
        bootstrap_error = beta / np.sqrt(selected_B)
        discretization_error = discretization_sensitivity / selected_K
        total_error = np.sqrt(bootstrap_error**2 + discretization_error**2)
        grid_description = f"Uniform {selected_K} points ({optimization_regime})"

    error_balance_ratio = (
        discretization_error / bootstrap_error if bootstrap_error > 0 else 0
    )

    return AllocationResult(
        status="ok",
        method=selected_method,
        B=selected_B,
        K=selected_K,
        grid_spec=grid_description,
        delta_B=bootstrap_error,
        delta_K=discretization_error,
        total_error=total_error,
        error_ratio=error_balance_ratio,
        comparison=AllocationComparison(
            full=ComparisonMethod(
                feasible=full_is_feasible,
                B=int(full_bootstrap_replicates) if full_is_feasible else None,
                K=int(full_grid_size),
                error=float(full_error) if full_is_feasible else None,
            ),
            uniform=UniformComparisonMethod(
                feasible=uniform_is_feasible,
                B=int(np.round(uniform_bootstrap_replicates))
                if uniform_is_feasible
                else None,
                K=int(np.round(uniform_grid_points)) if uniform_is_feasible else None,
                error=float(uniform_total_error) if uniform_is_feasible else None,
                regime=optimization_regime,
            ),
        ),
        params=AllocationParams(
            alpha=alpha,
            beta=beta,
            D=discretization_sensitivity,
            B_min=min_bootstrap_replicates,
            K_min=min_grid_points,
            n0=n_negatives,
            n1=n_positives,
            C=budget_elements,
        ),
        message=None,
    )


def construct_grid(
    method: Literal["full", "uniform"],
    num_grid_points: int,
    n_negatives: int,
    dtype: DTypeLike | None = None,
) -> NDArray[np.floating]:
    """
    Construct FPR evaluation grid for ROC curve interpolation.

    Parameters
    ----------
    method : {"full", "uniform"}
        Grid construction method:
        - "full": All n₀+1 jump points (exact supremum with step interpolation)
        - "uniform": K evenly spaced points
    num_grid_points : int
        Number of grid points K (used for uniform method)
    n_negatives : int
        Number of negative samples n₀ (determines jump point locations)
    dtype : dtype-like, optional
        Data type for output array. Defaults to float32.

    Returns
    -------
    ndarray
        Grid of FPR values in [0, 1]
        - Full grid: shape (n₀+1,), values {0, 1/n₀, 2/n₀, ..., 1}
        - Uniform grid: shape (K,), linearly spaced

    Notes
    -----
    With step interpolation (zero-order hold), the ROC value at any t between
    grid points t_j and t_{j+1} equals the value at t_j. This correctly
    represents piecewise-constant ROC curves.

    Both the empirical ROC R̂(t) and bootstrap ROC R_b(t) have jumps only at
    FPR values {0, 1/n₀, 2/n₀, ..., 1}. Using the full grid with step
    interpolation captures the exact supremum of the KS statistic.

    Examples
    --------
    Full grid for n₀=10:
    >>> grid = construct_grid("full", 11, n_negatives=10)
    >>> grid
    array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])

    Uniform grid with K=5:
    >>> grid = construct_grid("uniform", 5, n_negatives=10)
    >>> grid
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    """
    output_dtype = np.dtype(dtype) if dtype is not None else np.dtype(np.float32)

    if method == "full":
        # All jump points: {0, 1/n₀, 2/n₀, ..., 1}
        grid = np.linspace(0, 1, n_negatives + 1, dtype=output_dtype)
    elif method == "uniform":
        # K evenly spaced points
        grid = np.linspace(0, 1, num_grid_points, dtype=output_dtype)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'full' or 'uniform'.")

    return grid
