"""Bernstein Polynomial Smoothed Bootstrap Confidence Bands.

This module implements simultaneous confidence bands for ROC curves using
Bernstein polynomial smoothing with exact numerical methods.

Key features:
- BP-smoothed CDF estimation with analytical derivatives
- Exact ROC computation via numerical CDF/quantile (no Monte Carlo)
- Smoothed bootstrap with studentized retention
- Wilson score variance floor for boundary stability
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import comb as scipy_comb
from scipy.stats import norm

# Type alias for curve retention method selection
RetentionMethod = Literal["ks", "symmetric"]
SamplingMethod = Literal["exact", "interpolate"]


# =============================================================================
# Bernstein Polynomial CDF
# =============================================================================


class BernsteinCDF:
    """Bernstein polynomial smoothed CDF estimator.

    The BP-smoothed CDF of degree m is:
        F̃_m(u) = Σ_{k=0}^m F̂(k/m) · B_{k,m}(u)

    where B_{k,m}(u) = C(m,k) · u^k · (1-u)^{m-k} is the Bernstein basis.

    The PDF (derivative) is:
        f̃_m(u) = m · Σ_{k=0}^{m-1} [F̂((k+1)/m) - F̂(k/m)] · B_{k,m-1}(u)
    """

    def __init__(
        self, data: NDArray, degree: int | None = None, support_extension: float = 0.02
    ) -> None:
        """Initialize Bernstein CDF estimator.

        Args:
            data: Original observations.
            degree: Polynomial degree. Default: max(10, sqrt(n)).
            support_extension: Fraction to extend support beyond observed range.
        """
        self.data_original = np.sort(np.asarray(data, dtype=np.float64))
        self.n = len(self.data_original)

        if self.n < 2:
            raise ValueError("Need at least 2 data points")

        # Set degree with bias-variance tradeoff
        if degree is None:
            self.degree = max(10, int(np.sqrt(self.n)))
        else:
            self.degree = max(2, degree)

        # Define support with extension
        self.data_min = self.data_original[0]
        self.data_max = self.data_original[-1]
        data_range = self.data_max - self.data_min

        if data_range < 1e-10:
            data_range = 1.0  # Handle constant data

        self.buffer = support_extension * data_range
        self.support_min = self.data_min - self.buffer
        self.support_max = self.data_max + self.buffer
        self.support_range = self.support_max - self.support_min

        # Precompute coefficients
        self._precompute_coefficients()

    def _precompute_coefficients(self) -> None:
        """Precompute all coefficients needed for CDF and PDF evaluation."""
        m = self.degree

        # CDF coefficients: F̂(k/m) for k = 0, ..., m
        self.cdf_coeffs = np.array(
            [self._empirical_cdf_at_node(k / m) for k in range(m + 1)]
        )

        # PDF coefficients: m · [F̂((k+1)/m) - F̂(k/m)] for k = 0, ..., m-1
        self.pdf_coeffs = m * np.diff(self.cdf_coeffs)

        # Precompute binomial coefficients
        self.binom_cdf = np.array([scipy_comb(m, k, exact=True) for k in range(m + 1)])
        self.binom_pdf = np.array([scipy_comb(m - 1, k, exact=True) for k in range(m)])

    def _empirical_cdf_at_node(self, u: float) -> float:
        """Compute empirical CDF at unit-space position u."""
        if u <= 0:
            return 0.0
        if u >= 1:
            return 1.0
        x = self._from_unit(u)
        return np.searchsorted(self.data_original, x, side="right") / self.n

    def _to_unit(self, x: NDArray) -> NDArray:
        """Transform from original support to [0, 1]."""
        return (x - self.support_min) / self.support_range

    def _from_unit(self, u: float | NDArray) -> NDArray:
        """Transform from [0, 1] to original support."""
        return np.asarray(u) * self.support_range + self.support_min

    def _eval_bernstein_poly(
        self, u: NDArray, coeffs: NDArray, binom: NDArray
    ) -> NDArray:
        """Evaluate Bernstein polynomial with given coefficients.

        Uses log-space evaluation for numerical stability.
        """
        u = np.atleast_1d(np.asarray(u, dtype=np.float64))
        m = len(coeffs) - 1

        # Clip to avoid numerical issues at boundaries
        u = np.clip(u, 1e-15, 1 - 1e-15)

        result = np.zeros_like(u)
        log_u = np.log(u)
        log_1mu = np.log(1 - u)

        for k in range(m + 1):
            if coeffs[k] == 0:
                continue
            # B_{k,m}(u) = binom(m,k) * u^k * (1-u)^(m-k)
            log_basis = np.log(binom[k]) + k * log_u + (m - k) * log_1mu
            result += coeffs[k] * np.exp(log_basis)

        return result

    def cdf_unit(self, u: NDArray) -> NDArray:
        """Evaluate BP-smoothed CDF at u ∈ [0, 1]."""
        u = np.atleast_1d(u)
        result = np.zeros_like(u, dtype=np.float64)

        # Handle boundaries
        mask_lo = u <= 0
        mask_hi = u >= 1
        mask_mid = ~(mask_lo | mask_hi)

        result[mask_lo] = 0.0
        result[mask_hi] = 1.0

        if np.any(mask_mid):
            result[mask_mid] = self._eval_bernstein_poly(
                u[mask_mid], self.cdf_coeffs, self.binom_cdf
            )

        return np.clip(result, 0, 1)

    def pdf_unit(self, u: NDArray) -> NDArray:
        """Evaluate BP-smoothed PDF at u ∈ [0, 1].

        This is the analytical derivative of the CDF.
        """
        u = np.atleast_1d(u)
        result = np.zeros_like(u, dtype=np.float64)

        # PDF is zero outside (0, 1)
        mask = (u > 0) & (u < 1)

        if np.any(mask):
            result[mask] = self._eval_bernstein_poly(
                u[mask], self.pdf_coeffs, self.binom_pdf
            )

        return np.maximum(result, 0)

    def cdf(self, x: NDArray) -> NDArray:
        """Evaluate BP-smoothed CDF at x in original support."""
        u = self._to_unit(np.atleast_1d(x))
        return self.cdf_unit(u)

    def pdf(self, x: NDArray) -> NDArray:
        """Evaluate BP-smoothed PDF at x in original support.

        Accounts for Jacobian of unit transformation.
        """
        u = self._to_unit(np.atleast_1d(x))
        return self.pdf_unit(u) / self.support_range

    def quantile_unit(self, p: float, tol: float = 1e-12) -> float:
        """Compute quantile F̃^{-1}(p) in unit space using Brent's method."""
        if p <= 0:
            return 0.0
        if p >= 1:
            return 1.0

        def objective(u: float) -> float:
            return float(self.cdf_unit(np.array([u]))[0]) - p

        try:
            return brentq(objective, 1e-15, 1 - 1e-15, xtol=tol)
        except ValueError:
            # Fallback: bisection
            lo, hi = 0.0, 1.0
            for _ in range(100):
                mid = (lo + hi) / 2
                if hi - lo < tol:
                    break
                if objective(mid) < 0:
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) / 2

    def quantile(self, p: float, tol: float = 1e-12) -> float:
        """Compute quantile F̃^{-1}(p) in original support."""
        return float(self._from_unit(self.quantile_unit(p, tol)))

    def sample(self, n_samples: int, rng: np.random.Generator | None = None) -> NDArray:
        """Generate samples using the Beta mixture property (Vectorized/Exact).

        This replaces numerical inversion (Brent's method) with a generative
        process. The Bernstein PDF is a mixture of Beta distributions:
            f(u) = Σ w_k * Beta(u | k+1, m-k)

        Complexity: O(N) instead of O(N * iterations).
        """
        if rng is None:
            rng = np.random.default_rng()

        m = self.degree

        # 1. Calculate mixture weights
        # These are just the probability mass increments between nodes
        weights = np.diff(self.cdf_coeffs)

        # Ensure weights sum to 1.0 (fix potential float precision issues)
        weights = weights / np.sum(weights)

        # 2. Select components
        # Sample 'n_samples' indices k based on weights w_k
        # This tells us which Beta component each sample comes from
        component_indices = rng.choice(m, size=n_samples, p=weights)

        # 3. Sample from Beta components
        # Params: alpha = k + 1, beta = m - k
        alpha = component_indices + 1
        beta = m - component_indices

        samples_unit = rng.beta(alpha, beta)

        # 4. Transform from unit space [0,1] to original support
        return self._from_unit(samples_unit)


# =============================================================================
# Exact ROC Computation
# =============================================================================


@dataclass
class ExactBPROC:
    """Container for exact BP-ROC computation results."""

    fpr: NDArray
    tpr: NDArray
    thresholds: NDArray
    roc_slope: NDArray
    pdf_neg_at_c: NDArray
    pdf_pos_at_c: NDArray


def _compute_bp_roc_exact(
    bp_neg: BernsteinCDF,
    bp_pos: BernsteinCDF,
    fpr_grid: NDArray,
    compute_derivatives: bool = True,
) -> ExactBPROC:
    """Compute ROC curve from BP-smoothed distributions exactly.

    The ROC is computed as:
        R̃(t) = 1 - G̃(c)  where c = F̃^{-1}(1-t)

    No Monte Carlo simulation—purely numerical CDF/quantile evaluation.

    Args:
        bp_neg: BernsteinCDF for negative class.
        bp_pos: BernsteinCDF for positive class.
        fpr_grid: Array of FPR values to evaluate.
        compute_derivatives: Whether to compute R'(t) = g(c)/f(c).

    Returns:
        ExactBPROC with all computed quantities.
    """
    fpr_grid = np.atleast_1d(fpr_grid)
    n_points = len(fpr_grid)

    tpr = np.zeros(n_points)
    thresholds = np.zeros(n_points)
    roc_slope = np.zeros(n_points)
    pdf_neg = np.zeros(n_points)
    pdf_pos = np.zeros(n_points)

    for i, t in enumerate(fpr_grid):
        if t <= 0:
            tpr[i] = 0.0
            thresholds[i] = np.inf
            roc_slope[i] = np.inf
            continue

        if t >= 1:
            tpr[i] = 1.0
            thresholds[i] = -np.inf
            roc_slope[i] = 0.0
            continue

        # Compute threshold c = F̃_neg^{-1}(1 - t)
        c = bp_neg.quantile(1 - t)
        thresholds[i] = c

        # Compute TPR = 1 - G̃_pos(c)
        G_at_c = bp_pos.cdf(np.array([c]))[0]
        tpr[i] = 1 - G_at_c

        # Compute derivatives if requested
        if compute_derivatives:
            f_c = bp_neg.pdf(np.array([c]))[0]
            g_c = bp_pos.pdf(np.array([c]))[0]

            pdf_neg[i] = f_c
            pdf_pos[i] = g_c

            # R'(t) = g(c) / f(c)
            if f_c > 1e-10:
                roc_slope[i] = g_c / f_c
            else:
                roc_slope[i] = np.nan

    return ExactBPROC(
        fpr=fpr_grid,
        tpr=tpr,
        thresholds=thresholds,
        roc_slope=roc_slope,
        pdf_neg_at_c=pdf_neg,
        pdf_pos_at_c=pdf_pos,
    )


def _compute_empirical_roc(
    scores_neg: NDArray, scores_pos: NDArray, fpr_grid: NDArray
) -> NDArray:
    """Compute empirical ROC (step function) at given FPR values."""
    tpr = np.zeros(len(fpr_grid))

    for i, t in enumerate(fpr_grid):
        if t <= 0:
            tpr[i] = 0
        elif t >= 1:
            tpr[i] = 1
        else:
            c = np.quantile(scores_neg, 1 - t)
            tpr[i] = np.mean(scores_pos > c)

    return tpr


# =============================================================================
# Wilson Floor
# =============================================================================


def _wilson_variance_floor(p: NDArray, n: int, z: float = 1.96) -> NDArray:
    """Compute Wilson score interval variance as floor."""
    if n <= 0:
        return np.zeros_like(p)

    denom = 1 + z**2 / n
    return (z**2 / denom**2) * (p * (1 - p) / n + z**2 / (4 * n**2))


# =============================================================================
# Main SCB Construction
# =============================================================================


def bp_smoothed_bootstrap_band(
    scores_neg: NDArray,
    scores_pos: NDArray,
    alpha: float = 0.05,
    n_bootstrap: int = 2000,
    bp_degree: int | None = None,
    grid_points: int = 201,
    retention_method: RetentionMethod = "ks",
    sampling_method: SamplingMethod = "interpolate",
    random_seed: int | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Construct SCB using BP-smoothed bootstrap with exact numerical methods.

    Key features:
    1. Center ROC computed exactly via numerical CDF/quantile
    2. ROC slope computed analytically from BP PDFs
    3. Bootstrap samples drawn from BP distributions
    4. Studentized retention with Wilson score floor

    Args:
        scores_neg: Negative class scores.
        scores_pos: Positive class scores.
        alpha: Significance level (default 0.05).
        n_bootstrap: Number of bootstrap replicates (default 2000).
        bp_degree: Bernstein polynomial degree. Default: max(10, sqrt(n)).
        grid_points: Number of FPR evaluation points (default 201).
        retention_method: 'ks' for KS-based retention or 'symmetric' for
            two-sided trimming (default 'ks').
        sampling_method: 'exact' for exact quantiles or 'interpolate' for
            faster grid-based sampling (default 'interpolate').
        random_seed: Random seed for reproducibility (default None).

    Returns:
        Tuple of (fpr_grid, lower_band, upper_band).
    """
    rng = np.random.default_rng(random_seed)

    scores_neg = np.asarray(scores_neg, dtype=np.float64)
    scores_pos = np.asarray(scores_pos, dtype=np.float64)
    n_neg, n_pos = len(scores_neg), len(scores_pos)

    # === Step 1: FPR grid ===
    fpr_grid = np.linspace(0, 1, grid_points)
    n_grid = len(fpr_grid)

    # === Step 2: Fit Bernstein polynomial CDFs ===
    bp_neg = BernsteinCDF(scores_neg, degree=bp_degree)
    bp_pos = BernsteinCDF(scores_pos, degree=bp_degree)

    # === Step 3: Exact BP-implied ROC (center curve) ===
    bp_roc = _compute_bp_roc_exact(bp_neg, bp_pos, fpr_grid, compute_derivatives=True)
    roc_center = bp_roc.tpr

    # === Step 4: Smoothed bootstrap ===
    roc_bootstrap = np.zeros((n_bootstrap, n_grid))

    for b in range(n_bootstrap):
        # Sample from BP-smoothed distributions
        boot_neg = bp_neg.sample(n_neg, method=sampling_method, rng=rng)
        boot_pos = bp_pos.sample(n_pos, method=sampling_method, rng=rng)

        # Compute empirical ROC from bootstrap samples
        roc_bootstrap[b, :] = _compute_empirical_roc(boot_neg, boot_pos, fpr_grid)

    # === Step 5: Variance estimation ===
    sigma_bootstrap = np.std(roc_bootstrap, axis=0, ddof=1)

    # Wilson floor
    z_alpha = norm.ppf(1 - alpha / 2)
    sigma_wilson = np.sqrt(_wilson_variance_floor(roc_center, n_pos, z_alpha))

    # Combine: max of bootstrap, Wilson
    sigma_final = np.maximum(sigma_bootstrap, sigma_wilson)

    # === Step 6: Studentized retention ===
    epsilon = min(1 / (n_neg + n_pos), 1e-6)

    if retention_method == "symmetric":
        # Symmetric tail trimming
        M_up = np.zeros(n_bootstrap)
        M_down = np.zeros(n_bootstrap)

        for b in range(n_bootstrap):
            diff = roc_bootstrap[b, :] - roc_center
            z_vals = np.where(
                sigma_final >= epsilon,
                diff / sigma_final,
                np.where(np.abs(diff) < epsilon, 0, diff / epsilon),
            )
            M_up[b] = np.max(z_vals)
            M_down[b] = np.min(z_vals)

        q_up = np.quantile(M_up, 1 - alpha / 2)
        q_down = np.quantile(M_down, alpha / 2)
        retained_mask = (M_down >= q_down) & (M_up <= q_up)

    else:  # 'ks'
        # KS-based retention
        Z = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            delta = np.abs(roc_bootstrap[b, :] - roc_center)
            z_vals = np.where(
                sigma_final >= epsilon,
                delta / sigma_final,
                np.where(delta < epsilon, 0, delta / epsilon),
            )
            Z[b] = np.max(z_vals)

        threshold = np.quantile(Z, 1 - alpha)
        retained_mask = Z <= threshold

    # === Step 7: Envelope construction ===
    retained_curves = roc_bootstrap[retained_mask, :]
    n_retained = int(retained_mask.sum())

    if n_retained == 0:
        # Fallback: use all curves
        retained_curves = roc_bootstrap
        n_retained = n_bootstrap

    lower_band = np.min(retained_curves, axis=0)
    upper_band = np.max(retained_curves, axis=0)

    # === Step 8: Boundary constraints ===
    lower_band = np.clip(lower_band, 0, 1)
    upper_band = np.clip(upper_band, 0, 1)
    lower_band[0] = 0.0
    upper_band[-1] = 1.0

    # Enforce monotonicity
    lower_band = np.maximum.accumulate(lower_band)
    upper_band = np.maximum.accumulate(upper_band)

    return (fpr_grid, lower_band, upper_band)
