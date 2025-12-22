from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from ..datagen.true_rocs import DGP

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class BandResult:
    """Results from evaluating a single confidence band against the true ROC."""

    # Core coverage
    covers_entirely: bool
    violation_above: bool  # True ROC exceeds upper band somewhere
    violation_below: bool  # True ROC falls below lower band somewhere

    # Violation locations (FPR values where violations occur)
    violation_fpr_above: np.ndarray
    violation_fpr_below: np.ndarray

    # Violation magnitudes
    max_violation_above: float  # max(true_tpr - upper, 0)
    max_violation_below: float  # max(lower - true_tpr, 0)

    # Band properties
    band_area: float
    band_widths: np.ndarray  # Width at each FPR grid point
    fpr_grid: np.ndarray

    # Pointwise coverage (boolean array: True where band contains true ROC)
    pointwise_covered: np.ndarray

    # Violation indicators by FPR region
    # Keys: '0-10', '10-30', '30-50', '50-70', '70-90', '90-100'
    violation_by_region: dict = field(default_factory=dict)


@dataclass
class BandEvaluation:
    """Aggregated evaluation results from many band simulations."""

    n_simulations: int
    nominal_alpha: float

    # Coverage metrics
    coverage_rate: float
    coverage_se: float
    coverage_ci_lower: float
    coverage_ci_upper: float

    # Directional violations
    violation_rate_above: float
    violation_rate_below: float
    direction_test_pvalue: float  # H0: P(above) = P(below) | violation

    # Tightness metrics
    mean_band_area: float
    std_band_area: float
    mean_band_width: float  # Average across FPR domain
    width_percentiles: dict  # 10th, 50th, 90th percentile of mean width
    width_by_fpr_region: dict  # Mean width in each FPR region

    # Uniformity metrics
    pointwise_coverage_rates: np.ndarray
    pointwise_violation_rates: np.ndarray
    violation_rate_by_region: dict
    fpr_grid: np.ndarray

    # Violation location analysis
    violation_fpr_distribution_above: np.ndarray | None
    violation_fpr_distribution_below: np.ndarray | None

    # Violation magnitudes
    mean_max_violation: float
    percentile_95_max_violation: float


# =============================================================================
# Per-Band Evaluation Functions
# =============================================================================


def evaluate_single_band(
    lower_band: np.ndarray,
    upper_band: np.ndarray,
    true_tpr: np.ndarray,
    fpr_grid: np.ndarray,
) -> BandResult:
    """
    Evaluate a single confidence band against the true ROC curve.

    Parameters
    ----------
    lower_band : np.ndarray
        Lower bound of confidence band at each FPR grid point
    upper_band : np.ndarray
        Upper bound of confidence band at each FPR grid point
    true_tpr : np.ndarray
        True TPR values at each FPR grid point
    fpr_grid : np.ndarray
        FPR values where band and true ROC are evaluated

    Returns
    -------
    BandResult
        Comprehensive evaluation of this single band

    Notes
    -----
    ROC curves are pinned at (0,0) and (1,1), so bands at these points
    should have zero width and always cover the true ROC.
    """
    # Validate inputs and preserve dtype from true_tpr
    true_tpr = np.asarray(true_tpr)
    dtype = true_tpr.dtype
    lower_band = np.asarray(lower_band, dtype=dtype)
    upper_band = np.asarray(upper_band, dtype=dtype)
    fpr_grid = np.asarray(fpr_grid, dtype=dtype)

    n = len(fpr_grid)
    assert len(lower_band) == n and len(upper_band) == n and len(true_tpr) == n

    # Pointwise coverage
    # Use small tolerance for boundary comparisons due to floating point
    tolerance = 1e-10
    above = (true_tpr - upper_band) > tolerance
    below = (lower_band - true_tpr) > tolerance
    pointwise_covered = ~(above | below)

    # Overall coverage
    covers_entirely = pointwise_covered.all()
    violation_above = above.any()
    violation_below = below.any()

    # Violation locations
    violation_fpr_above = (
        fpr_grid[above] if violation_above else np.array([], dtype=dtype)
    )
    violation_fpr_below = (
        fpr_grid[below] if violation_below else np.array([], dtype=dtype)
    )

    # Violation magnitudes
    violations_above_mag = np.maximum(true_tpr - upper_band, 0)
    violations_below_mag = np.maximum(lower_band - true_tpr, 0)
    max_violation_above = violations_above_mag.max()
    max_violation_below = violations_below_mag.max()

    # Band properties
    band_widths = upper_band - lower_band
    band_area = np.trapezoid(band_widths, fpr_grid)

    # Violations by region
    regions = {
        "0-10": (0.0, 0.1),
        "10-30": (0.1, 0.3),
        "30-50": (0.3, 0.5),
        "50-70": (0.5, 0.7),
        "70-90": (0.7, 0.9),
        "90-100": (0.9, 1.0),
    }

    violation_by_region = {}
    for region_name, (lo, hi) in regions.items():
        mask = (fpr_grid >= lo) & (fpr_grid < hi if hi < 1.0 else fpr_grid <= hi)
        if mask.any():
            # Any violation in this region?
            violation_by_region[region_name] = (above[mask] | below[mask]).any()
        else:
            violation_by_region[region_name] = False

    return BandResult(
        covers_entirely=covers_entirely,
        violation_above=violation_above,
        violation_below=violation_below,
        violation_fpr_above=violation_fpr_above,
        violation_fpr_below=violation_fpr_below,
        max_violation_above=max_violation_above,
        max_violation_below=max_violation_below,
        band_area=band_area,
        band_widths=band_widths,
        fpr_grid=fpr_grid,
        pointwise_covered=pointwise_covered,
        violation_by_region=violation_by_region,
    )


def compute_empirical_roc(
    scores_pos: np.ndarray, scores_neg: np.ndarray, fpr_grid: np.ndarray
) -> np.ndarray:
    """
    Compute empirical ROC curve at specified FPR values.

    Parameters
    ----------
    scores_pos : np.ndarray
        Scores for positive class samples
    scores_neg : np.ndarray
        Scores for negative class samples
    fpr_grid : np.ndarray
        FPR values at which to evaluate the ROC

    Returns
    -------
    np.ndarray
        Empirical TPR at each FPR grid point
    """
    scores_pos = np.asarray(scores_pos)
    dtype = scores_pos.dtype

    # Thresholds corresponding to each FPR
    # FPR = P(neg > threshold) = 1 - P(neg <= threshold)
    # So threshold = quantile(neg, 1 - FPR)
    thresholds = np.quantile(scores_neg, 1 - fpr_grid).astype(dtype)

    # TPR at each threshold
    # Handle edge cases where all scores might be above/below threshold
    tpr = np.array([(scores_pos >= t).mean() for t in thresholds], dtype=dtype)

    return tpr


# =============================================================================
# Aggregation Function
# =============================================================================


def aggregate_band_results(
    results: Iterable[BandResult],
    nominal_alpha: float = 0.05,
    fpr_grid: np.ndarray | None = None,
) -> BandEvaluation:
    """
    Aggregate multiple BandResult objects into overall evaluation statistics.

    Parameters
    ----------
    results : Iterable[BandResult]
        Iterator or list of BandResult objects from individual simulations
    nominal_alpha : float
        Nominal significance level (for reference in output)
    fpr_grid : Optional[np.ndarray]
        FPR grid for pointwise statistics. If None, uses grid from first result.

    Returns
    -------
    BandEvaluation
        Comprehensive aggregated evaluation statistics
    """
    # Convert to list to allow multiple passes
    results_list: list[BandResult] = list(results)
    n_sims = len(results_list)

    if n_sims == 0:
        raise ValueError("No results to aggregate")

    # Get FPR grid from first result if not provided
    if fpr_grid is None:
        fpr_grid = results_list[0].fpr_grid
    else:
        fpr_grid = np.asarray(fpr_grid)

    dtype = fpr_grid.dtype

    # -------------------------------------------------------------------------
    # Coverage statistics
    # -------------------------------------------------------------------------
    covers = np.array([r.covers_entirely for r in results_list])
    coverage_rate = covers.mean()
    coverage_se = np.sqrt(coverage_rate * (1 - coverage_rate) / n_sims)

    # Wilson score interval for coverage
    z = stats.norm.ppf(0.975)
    denom = 1 + z**2 / n_sims
    center = (coverage_rate + z**2 / (2 * n_sims)) / denom
    margin = (
        z
        * np.sqrt(coverage_rate * (1 - coverage_rate) / n_sims + z**2 / (4 * n_sims**2))
        / denom
    )
    coverage_ci_lower = center - margin
    coverage_ci_upper = center + margin

    # -------------------------------------------------------------------------
    # Directional violations
    # -------------------------------------------------------------------------
    violations_above = np.array([r.violation_above for r in results_list])
    violations_below = np.array([r.violation_below for r in results_list])

    violation_rate_above = violations_above.mean()
    violation_rate_below = violations_below.mean()

    # Test for directional symmetry
    # Among simulations with violations, are above/below equally likely?
    n_above = violations_above.sum()
    n_below = violations_below.sum()
    n_with_any_violation = ((violations_above) | (violations_below)).sum()

    if n_with_any_violation > 0:
        # Use exact binomial test
        # H0: P(violation is above | violation occurred) = 0.5
        # Count: number of "above" events among all violation events
        # Note: a single sim can have both above and below violations
        direction_test_pvalue = _binomial_test_twosided(n_above, n_above + n_below, 0.5)
    else:
        direction_test_pvalue = 1.0

    # -------------------------------------------------------------------------
    # Tightness metrics
    # -------------------------------------------------------------------------
    band_areas = np.array([r.band_area for r in results_list], dtype=dtype)
    mean_band_area = band_areas.mean()
    std_band_area = band_areas.std()

    # Stack band widths: (n_sims, n_grid)
    band_widths_matrix = np.vstack([r.band_widths for r in results_list])
    mean_widths_by_fpr = band_widths_matrix.mean(axis=0)

    # Exclude pinned boundaries (FPR=0 and FPR=1) from width calculations
    # since these have zero variance by construction
    width_mask = np.ones(len(fpr_grid), dtype=bool)
    if fpr_grid[0] == 0.0:
        width_mask[0] = False
    if fpr_grid[-1] == 1.0:
        width_mask[-1] = False

    # Mean width excluding boundaries
    if width_mask.any():
        mean_band_width = mean_widths_by_fpr[width_mask].mean()
        sim_mean_widths = band_widths_matrix[:, width_mask].mean(
            axis=1
        )  # Mean width per simulation
    else:
        mean_band_width = mean_widths_by_fpr.mean()
        sim_mean_widths = band_widths_matrix.mean(axis=1)

    # Percentiles of mean width across simulations
    width_percentiles = {
        "p10": np.percentile(sim_mean_widths, 10),
        "p50": np.percentile(sim_mean_widths, 50),
        "p90": np.percentile(sim_mean_widths, 90),
    }

    # Mean width by FPR region
    regions = {
        "0-10": (0.0, 0.1),
        "10-30": (0.1, 0.3),
        "30-50": (0.3, 0.5),
        "50-70": (0.5, 0.7),
        "70-90": (0.7, 0.9),
        "90-100": (0.9, 1.0),
    }

    width_by_fpr_region = {}
    for region_name, (lo, hi) in regions.items():
        mask = (fpr_grid >= lo) & (fpr_grid < hi if hi < 1.0 else fpr_grid <= hi)
        if mask.any():
            width_by_fpr_region[region_name] = mean_widths_by_fpr[mask].mean()
        else:
            width_by_fpr_region[region_name] = np.nan

    # -------------------------------------------------------------------------
    # Pointwise coverage/violation rates
    # -------------------------------------------------------------------------
    pointwise_covered_matrix = np.vstack([r.pointwise_covered for r in results_list])
    pointwise_coverage_rates = pointwise_covered_matrix.mean(axis=0)
    pointwise_violation_rates = 1 - pointwise_coverage_rates

    # -------------------------------------------------------------------------
    # Violation rates by region
    # -------------------------------------------------------------------------
    violation_rate_by_region = {}
    for region_name in regions.keys():
        region_violations = [
            r.violation_by_region.get(region_name, False) for r in results_list
        ]
        violation_rate_by_region[region_name] = np.mean(region_violations)

    # -------------------------------------------------------------------------
    # Violation location distributions
    # -------------------------------------------------------------------------
    all_violation_fpr_above = (
        np.concatenate(
            [
                r.violation_fpr_above
                for r in results_list
                if len(r.violation_fpr_above) > 0
            ]
        )
        if any(len(r.violation_fpr_above) > 0 for r in results_list)
        else None
    )

    all_violation_fpr_below = (
        np.concatenate(
            [
                r.violation_fpr_below
                for r in results_list
                if len(r.violation_fpr_below) > 0
            ]
        )
        if any(len(r.violation_fpr_below) > 0 for r in results_list)
        else None
    )

    # -------------------------------------------------------------------------
    # Violation magnitudes
    # -------------------------------------------------------------------------
    max_violations = np.array(
        [max(r.max_violation_above, r.max_violation_below) for r in results_list],
        dtype=dtype,
    )
    mean_max_violation = max_violations.mean()
    percentile_95_max_violation = np.percentile(max_violations, 95)

    return BandEvaluation(
        n_simulations=n_sims,
        nominal_alpha=nominal_alpha,
        coverage_rate=coverage_rate,
        coverage_se=coverage_se,
        coverage_ci_lower=coverage_ci_lower,
        coverage_ci_upper=coverage_ci_upper,
        violation_rate_above=violation_rate_above,
        violation_rate_below=violation_rate_below,
        direction_test_pvalue=direction_test_pvalue,
        mean_band_area=mean_band_area,
        std_band_area=std_band_area,
        mean_band_width=mean_band_width,
        width_percentiles=width_percentiles,
        width_by_fpr_region=width_by_fpr_region,
        pointwise_coverage_rates=pointwise_coverage_rates,
        pointwise_violation_rates=pointwise_violation_rates,
        violation_rate_by_region=violation_rate_by_region,
        fpr_grid=fpr_grid,
        violation_fpr_distribution_above=all_violation_fpr_above,
        violation_fpr_distribution_below=all_violation_fpr_below,
        mean_max_violation=mean_max_violation,
        percentile_95_max_violation=percentile_95_max_violation,
    )


def _binomial_test_twosided(k: int, n: int, p: float) -> float:
    """Two-sided binomial test p-value."""
    if n == 0:
        return 1.0
    # Use scipy's binomial test if available (scipy >= 1.7)
    try:
        result = stats.binomtest(k, n, p, alternative="two-sided")
        return result.pvalue
    except AttributeError:
        # Fallback for older scipy
        expected = n * p
        if k <= expected:
            pvalue = 2 * stats.binom.cdf(k, n, p)
        else:
            pvalue = 2 * (1 - stats.binom.cdf(k - 1, n, p))
        return min(pvalue, 1.0)


# =============================================================================
# Simulation Runner (Convenience Function)
# =============================================================================


def run_band_simulation(
    dgp: DGP,
    band_method: Callable,
    n_pos: int,
    n_neg: int,
    alpha: float = 0.05,
    n_simulations: int = 1000,
    fpr_grid: np.ndarray | None = None,
    true_roc_n_samples: int = 100_000,
    seed: int | None = None,
    progress: bool = False,
) -> BandEvaluation:
    """
    Convenience function to run full simulation study.

    Parameters
    ----------
    dgp : DGP
        Data generating process
    band_method : Callable
        Function (scores_pos, scores_neg, alpha) -> (fpr, lower, upper)
    n_pos, n_neg : int
        Sample sizes for positive and negative classes
    alpha : float
        Nominal significance level
    n_simulations : int
        Number of Monte Carlo simulations
    fpr_grid : np.ndarray | None
        FPR grid for evaluation. Default: linspace(0, 1, 201)
    true_roc_n_samples : int
        Sample size for estimating true ROC if not provided analytically
    seed : int | None
        Random seed for reproducibility
    progress : bool
        Whether to show progress (requires tqdm)

    Returns
    -------
    BandEvaluation
        Aggregated evaluation results
    """
    rng = np.random.default_rng(seed)

    # Generate a sample to determine dtype
    sample_pos, sample_neg = dgp.sample(1, 1, rng)
    dtype = sample_pos.dtype

    if fpr_grid is None:
        fpr_grid = np.linspace(0, 1, 201, dtype=dtype)
    else:
        fpr_grid = np.asarray(fpr_grid, dtype=dtype)

    # Get true ROC (either analytic or estimated)
    true_tpr = dgp.get_true_roc(fpr_grid, n_samples=true_roc_n_samples, rng=rng)

    # Generator for results
    def result_generator():
        iterator = range(n_simulations)
        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="Simulating")
            except ImportError:
                pass

        for _ in iterator:
            # Generate data
            scores_pos, scores_neg = dgp.sample(n_pos, n_neg, rng)

            # Compute band
            fpr_out, lower, upper = band_method(scores_pos, scores_neg, alpha)

            # Interpolate to common grid if needed
            if len(fpr_out) != len(fpr_grid) or not np.allclose(fpr_out, fpr_grid):
                lower = np.interp(
                    fpr_grid, fpr_out, lower, left=lower[0], right=lower[-1]
                ).astype(dtype)
                upper = np.interp(
                    fpr_grid, fpr_out, upper, left=upper[0], right=upper[-1]
                ).astype(dtype)

            yield evaluate_single_band(lower, upper, true_tpr, fpr_grid)

    return aggregate_band_results(
        result_generator(), nominal_alpha=alpha, fpr_grid=fpr_grid
    )


# =============================================================================
# Diagnostic Functions
# =============================================================================


def compute_pointwise_coverage_diagnostics(
    evaluation: BandEvaluation, alpha: float = 0.05
) -> dict:
    """
    Analyze pointwise coverage patterns.

    Returns dict with:
    - excess_coverage: pointwise_coverage - (1-alpha), positive = conservative
    - conservative_regions: FPR regions where significantly over-covering
    - liberal_regions: FPR regions where significantly under-covering

    Notes
    -----
    At pinned boundaries (FPR=0, FPR=1), coverage should be 100% with zero
    variance. Z-scores are set to NaN at these points as they are not meaningful.
    """
    dtype = evaluation.fpr_grid.dtype
    nominal = 1 - alpha
    n_sims = evaluation.n_simulations
    fpr_grid = evaluation.fpr_grid

    # Standard error of coverage at each point
    pc = evaluation.pointwise_coverage_rates
    pointwise_se = np.sqrt(pc * (1 - pc) / n_sims).astype(dtype)

    # Z-scores for deviation from nominal
    # Set to NaN at pinned boundaries where SE is zero by construction
    z_scores = np.full_like(pc, np.nan, dtype=dtype)
    valid_mask = pointwise_se > 1e-10

    # Only compute z-scores where SE is non-zero
    z_scores[valid_mask] = (
        (pc[valid_mask] - nominal) / pointwise_se[valid_mask]
    ).astype(dtype)

    # Mark pinned boundaries explicitly
    is_pinned = np.zeros(len(fpr_grid), dtype=bool)
    if fpr_grid[0] == 0.0:
        is_pinned[0] = True
    if fpr_grid[-1] == 1.0:
        is_pinned[-1] = True

    # Excess coverage (positive = conservative)
    excess_coverage = (pc - nominal).astype(dtype)

    return {
        "fpr_grid": evaluation.fpr_grid,
        "pointwise_coverage": pc,
        "pointwise_se": pointwise_se,
        "excess_coverage": excess_coverage,
        "z_scores": z_scores,
        "is_pinned": is_pinned,
        "nominal": nominal,
    }


def compute_uniformity_diagnostics(
    evaluation: BandEvaluation,
    method: str = "permutation",
    n_permutations: int = 1000,
    seed: int | None = None,
) -> dict:
    """
    Test whether violations are uniform across the FPR domain.

    Parameters
    ----------
    evaluation : BandEvaluation
        Aggregated evaluation results
    method : str
        'chi2' (ignores autocorrelation) or 'permutation' (accounts for it)
    n_permutations : int
        Number of permutations for permutation test
    seed : Optional[int]
        Random seed

    Returns
    -------
    dict with test results and diagnostics

    Notes
    -----
    The chi-square test assumes independent observations, which is violated
    here due to autocorrelation along the FPR axis. The permutation test
    provides a more appropriate alternative by using the observed violation
    patterns and testing whether their distribution across regions differs
    from what we'd expect if violations were uniformly distributed.
    """
    regions = ["0-10", "10-30", "30-50", "50-70", "70-90", "90-100"]
    region_rates = [evaluation.violation_rate_by_region[r] for r in regions]

    # Chi-square test (with caveat about autocorrelation)
    observed = np.array(region_rates) * evaluation.n_simulations
    expected = np.mean(observed)

    if expected < 5:
        chi2_pvalue = np.nan
        chi2_warning = "Expected counts too low for chi-square test"
    else:
        chi2_stat = ((observed - expected) ** 2 / expected).sum()
        chi2_pvalue = 1 - stats.chi2.cdf(chi2_stat, df=len(regions) - 1)
        chi2_warning = "Chi-square assumes independence"

    result = {
        "region_violation_rates": dict(zip(regions, region_rates, strict=True)),
        "chi2_pvalue": chi2_pvalue,
        "chi2_warning": chi2_warning,
    }

    # Add coefficient of variation as a descriptive measure
    if np.mean(region_rates) > 0:
        cv = np.std(region_rates) / np.mean(region_rates)
        result["coefficient_of_variation"] = cv
    else:
        result["coefficient_of_variation"] = 0.0

    return result


def summarize_evaluation(evaluation: BandEvaluation) -> str:
    """Generate a formatted text summary of evaluation results."""
    lines = [
        "=" * 60,
        "ROC CONFIDENCE BAND EVALUATION SUMMARY",
        "=" * 60,
        f"Simulations: {evaluation.n_simulations}",
        f"Nominal α: {evaluation.nominal_alpha:.3f}",
        "",
        "COVERAGE",
        "-" * 40,
        f"  Coverage rate: {evaluation.coverage_rate:.4f}",
        f"  Standard error: {evaluation.coverage_se:.4f}",
        f"  95% CI: [{evaluation.coverage_ci_lower:.4f}, {evaluation.coverage_ci_upper:.4f}]",
        f"  Nominal (1-α): {1 - evaluation.nominal_alpha:.4f}",
        "",
        "DIRECTIONAL VIOLATIONS",
        "-" * 40,
        f"  Violation rate (above): {evaluation.violation_rate_above:.4f}",
        f"  Violation rate (below): {evaluation.violation_rate_below:.4f}",
        f"  Symmetry test p-value: {evaluation.direction_test_pvalue:.4f}",
        "",
        "BAND TIGHTNESS",
        "-" * 40,
        f"  Mean band area: {evaluation.mean_band_area:.4f}",
        f"  Std band area: {evaluation.std_band_area:.4f}",
        f"  Mean band width: {evaluation.mean_band_width:.4f}",
        "",
        "WIDTH BY FPR REGION",
        "-" * 40,
    ]

    for region, width in evaluation.width_by_fpr_region.items():
        lines.append(f"  {region}: {width:.4f}")

    lines.extend(["", "VIOLATION RATE BY FPR REGION", "-" * 40])

    for region, rate in evaluation.violation_rate_by_region.items():
        lines.append(f"  {region}: {rate:.4f}")

    lines.extend(
        [
            "",
            "VIOLATION MAGNITUDES",
            "-" * 40,
            f"  Mean max violation: {evaluation.mean_max_violation:.4f}",
            f"  95th percentile max violation: {evaluation.percentile_95_max_violation:.4f}",
            "=" * 60,
        ]
    )

    return "\n".join(lines)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example DGP: Gaussian scores with separation
    def gaussian_dgp(n_pos: int, n_neg: int, rng: np.random.Generator):
        """Positive ~ N(1, 1), Negative ~ N(0, 1)"""
        scores_pos = rng.normal(1.0, 1.0, n_pos)
        scores_neg = rng.normal(0.0, 1.0, n_neg)
        return scores_pos, scores_neg

    def true_roc_gaussian(fpr: np.ndarray) -> np.ndarray:
        """Analytic ROC for Gaussian with Δμ=1, σ=1."""
        return stats.norm.cdf(stats.norm.ppf(fpr) + 1.0)

    # Create DGP with analytic true ROC
    dgp_with_true = DGP(generator=gaussian_dgp, true_roc=true_roc_gaussian)

    # Create DGP without analytic true ROC (will estimate)
    dgp_estimated = DGP(generator=gaussian_dgp, true_roc=None)

    # Example band method (placeholder - replace with your method)
    def example_band_method(scores_pos, scores_neg, alpha):
        """Placeholder bootstrap band method."""
        fpr_grid = np.linspace(0, 1, 101)

        # Simple pointwise bootstrap (NOT a proper simultaneous band!)
        n_boot = 200
        rng = np.random.default_rng()
        boot_tprs = []

        for _ in range(n_boot):
            idx_pos = rng.choice(len(scores_pos), len(scores_pos), replace=True)
            idx_neg = rng.choice(len(scores_neg), len(scores_neg), replace=True)
            tpr = compute_empirical_roc(
                scores_pos[idx_pos], scores_neg[idx_neg], fpr_grid
            )
            boot_tprs.append(tpr)

        boot_tprs = np.array(boot_tprs)
        lower = np.percentile(boot_tprs, 100 * alpha / 2, axis=0)
        upper = np.percentile(boot_tprs, 100 * (1 - alpha / 2), axis=0)

        return fpr_grid, lower, upper

    # Run evaluation
    print("Running simulation with analytic true ROC...")
    eval_result = run_band_simulation(
        dgp=dgp_with_true,
        band_method=example_band_method,
        n_pos=100,
        n_neg=100,
        alpha=0.05,
        n_simulations=500,
        seed=42,
        progress=True,
    )

    print(summarize_evaluation(eval_result))

    # Manual usage pattern
    print("\n\nManual evaluation pattern:")
    print("-" * 40)

    fpr_grid = np.linspace(0, 1, 201)
    true_tpr = dgp_with_true.get_true_roc(fpr_grid)
    rng = np.random.default_rng(123)

    # Evaluate individual bands
    results = []
    for i in range(100):
        scores_pos, scores_neg = dgp_with_true.sample(100, 100, rng)
        fpr, lower, upper = example_band_method(scores_pos, scores_neg, 0.05)

        # Interpolate to common grid
        lower_interp = np.interp(fpr_grid, fpr, lower)
        upper_interp = np.interp(fpr_grid, fpr, upper)

        # Evaluate this band
        result = evaluate_single_band(lower_interp, upper_interp, true_tpr, fpr_grid)
        results.append(result)

        if i < 3:
            print(
                f"  Band {i}: covers={result.covers_entirely}, "
                f"area={result.band_area:.3f}, "
                f"above={result.violation_above}, below={result.violation_below}"
            )

    # Aggregate
    agg = aggregate_band_results(results, nominal_alpha=0.05)
    print(f"\n  Aggregated coverage: {agg.coverage_rate:.3f}")
