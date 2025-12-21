"""
Data Generating Processes with Analytic ROC Curves

Each DGP includes:
- generator: sampling function (n_pos, n_neg, rng) -> (scores_pos, scores_neg)
- true_roc: analytic ROC function (fpr) -> tpr
- description: what makes this case interesting
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.optimize import brentq
from scipy.special import betainc, betaincinv

# =============================================================================
# Base DGP Class (from previous code)
# =============================================================================


@dataclass
class DGP:
    """
    Data generating process wrapper.

    Parameters
    ----------
    generator : Callable
        Function with signature (n_pos, n_neg, rng) -> (scores_pos, scores_neg)
    true_roc : Optional[Callable]
        Function with signature (fpr_array) -> tpr_array. If None, will be
        estimated via Monte Carlo when needed.
    name : str
        Human-readable name for this DGP
    description : str
        Description of what makes this DGP interesting
    """
    generator: Callable[[int, int, np.random.Generator], tuple[np.ndarray, np.ndarray]]
    true_roc: Callable[[np.ndarray], np.ndarray] | None = None
    name: str = ""
    description: str = ""
    _estimated_roc_cache: tuple[np.ndarray, np.ndarray] | None = field(
        default=None, repr=False, init=False
    )

    def sample(
        self, n_pos: int, n_neg: int, rng: np.random.Generator | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a sample from the DGP."""
        if rng is None:
            rng = np.random.default_rng()
        return self.generator(n_pos, n_neg, rng)

    def get_true_roc(
        self,
        fpr_grid: np.ndarray,
        n_samples: int = 100_000,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Get true ROC values at specified FPR points.

        If true_roc function is provided, uses that. Otherwise estimates
        via Monte Carlo with large sample size.

        Parameters
        ----------
        fpr_grid : np.ndarray
            FPR values at which to evaluate the ROC
        n_samples : int
            Sample size for Monte Carlo estimation (if no analytic ROC)
        rng : np.random.Generator, optional
            Random number generator for Monte Carlo estimation

        Returns
        -------
        np.ndarray
            TPR values at the specified FPR points
        """
        if self.true_roc is not None:
            return self.true_roc(fpr_grid)

        # Estimate from large sample
        return estimate_true_roc(self, fpr_grid, n_samples, rng)


def estimate_true_roc(
    dgp: DGP,
    fpr_grid: np.ndarray,
    n_samples: int = 100_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Estimate the true ROC curve via Monte Carlo with large sample.

    Uses a very large sample to approximate the population ROC.
    The standard error of the estimated TPR at each point is approximately
    sqrt(TPR * (1-TPR) / n_pos), which for n=100,000 is < 0.002.

    Parameters
    ----------
    dgp : DGP
        Data generating process
    fpr_grid : np.ndarray
        FPR values at which to evaluate the ROC
    n_samples : int
        Sample size for Monte Carlo estimation
    rng : np.random.Generator, optional
        Random number generator

    Returns
    -------
    np.ndarray
        Estimated TPR values at the specified FPR points
    """
    if rng is None:
        rng = np.random.default_rng()

    scores_pos, scores_neg = dgp.sample(n_samples, n_samples, rng)
    dtype = scores_pos.dtype

    # Compute thresholds at each FPR level
    # FPR = P(score_neg > threshold), so threshold = quantile(1 - FPR)
    thresholds = np.quantile(scores_neg, 1 - fpr_grid).astype(dtype)

    # Handle edge cases
    thresholds = np.clip(
        thresholds, scores_neg.min() - 1, scores_neg.max() + 1
    ).astype(dtype)

    # TPR at each threshold
    tpr = np.array([(scores_pos >= t).mean() for t in thresholds], dtype=dtype)

    # Ensure monotonicity (should be automatic but numerical issues possible)
    tpr = np.maximum.accumulate(tpr).astype(dtype)

    return tpr


# =============================================================================
# 1. BASELINE: Gaussian with Equal Variance
# =============================================================================


def make_gaussian_dgp(delta_mu: float = 1.0, sigma: float = 1.0) -> DGP:
    """
    Baseline Gaussian case.

    Neg ~ N(0, σ²), Pos ~ N(Δμ, σ²)

    ROC Derivation:
    ---------------
    FPR(t) = P(Neg > t) = 1 - Φ(t/σ) = Φ(-t/σ)
    TPR(t) = P(Pos > t) = 1 - Φ((t-Δμ)/σ) = Φ((Δμ-t)/σ)

    Inverting: t = -σ·Φ⁻¹(FPR)
    Therefore: TPR = Φ((Δμ + σ·Φ⁻¹(FPR))/σ) = Φ(Δμ/σ + Φ⁻¹(FPR))

    The ROC depends only on d' = Δμ/σ (discriminability).
    """
    d_prime = delta_mu / sigma

    def generator(n_pos, n_neg, rng):
        scores_neg = rng.normal(0, sigma, n_neg)
        scores_pos = rng.normal(delta_mu, sigma, n_pos)
        return scores_pos, scores_neg

    def true_roc(fpr):
        fpr = np.asarray(fpr)
        # Handle edge cases
        fpr_clipped = np.clip(fpr, 1e-10, 1 - 1e-10)
        return stats.norm.cdf(d_prime + stats.norm.ppf(fpr_clipped))

    return DGP(
        generator=generator,
        true_roc=true_roc,
        name=f"Gaussian(Δμ={delta_mu}, σ={sigma})",
        description="Baseline equal-variance Gaussian. ROC is symmetric about the minor diagonal.",
    )


# =============================================================================
# 2. SKEWED TOWARD EACH OTHER: Beta Distributions
# =============================================================================


def make_beta_opposing_skew_dgp(alpha: float = 2, beta: float = 5) -> DGP:
    """
    Distributions skewed toward each other.

    Neg ~ Beta(α, β) - right-skewed, mass near 0
    Pos ~ Beta(β, α) - left-skewed, mass near 1

    This creates good separation with heavy overlap in the tails.

    ROC Derivation:
    ---------------
    For X ~ Beta(a, b), CDF is F(x) = I_x(a, b) (regularized incomplete beta)

    Neg CDF: F_neg(t) = I_t(α, β)
    Pos CDF: F_pos(t) = I_t(β, α)

    FPR(t) = 1 - F_neg(t) = 1 - I_t(α, β)
    TPR(t) = 1 - F_pos(t) = 1 - I_t(β, α)

    Invert: t = I⁻¹_{1-FPR}(α, β)
    Then: TPR = 1 - I_t(β, α)
    """

    def generator(n_pos, n_neg, rng):
        scores_neg = rng.beta(alpha, beta, n_neg)
        scores_pos = rng.beta(beta, alpha, n_pos)
        return scores_pos, scores_neg

    def true_roc(fpr):
        fpr = np.asarray(fpr)
        tpr = np.zeros_like(fpr, dtype=float)

        for i, f in enumerate(fpr.flat):
            if f <= 0:
                tpr.flat[i] = 0.0
            elif f >= 1:
                tpr.flat[i] = 1.0
            else:
                # Threshold where FPR = f
                # 1 - I_t(α, β) = f => I_t(α, β) = 1 - f
                t = betaincinv(alpha, beta, 1 - f)
                # TPR at this threshold
                tpr.flat[i] = 1 - betainc(beta, alpha, t)

        return tpr

    return DGP(
        generator=generator,
        true_roc=true_roc,
        name=f"Beta_OpposingSkew(α={alpha}, β={beta})",
        description="Neg right-skewed toward Pos, Pos left-skewed toward Neg. Good separation, overlapping tails.",
    )


def make_beta_same_support_opposing_dgp(
    neg_alpha: float = 2, neg_beta: float = 3, pos_alpha: float = 3, pos_beta: float = 2
) -> DGP:
    """
    More general opposing skew with controllable parameters.

    Neg ~ Beta(neg_α, neg_β)
    Pos ~ Beta(pos_α, pos_β)

    Example: neg_alpha=2, neg_beta=3 gives slight right skew
             pos_alpha=3, pos_beta=2 gives slight left skew
    """

    def generator(n_pos, n_neg, rng):
        scores_neg = rng.beta(neg_alpha, neg_beta, n_neg)
        scores_pos = rng.beta(pos_alpha, pos_beta, n_pos)
        return scores_pos, scores_neg

    def true_roc(fpr):
        fpr = np.asarray(fpr)
        tpr = np.zeros_like(fpr, dtype=float)

        for i, f in enumerate(fpr.flat):
            if f <= 0:
                tpr.flat[i] = 0.0
            elif f >= 1:
                tpr.flat[i] = 1.0
            else:
                t = betaincinv(neg_alpha, neg_beta, 1 - f)
                tpr.flat[i] = 1 - betainc(pos_alpha, pos_beta, t)

        return tpr

    return DGP(
        generator=generator,
        true_roc=true_roc,
        name=f"Beta(neg={neg_alpha},{neg_beta}; pos={pos_alpha},{pos_beta})",
        description="General Beta with opposing skewness directions.",
    )


# =============================================================================
# 3. SKEWED IN SAME DIRECTION: Log-Normal
# =============================================================================


def make_lognormal_dgp(
    neg_mu: float = 0.0, pos_mu: float = 0.5, sigma: float = 0.5
) -> DGP:
    """
    Both classes right-skewed (log-normal).

    Neg ~ LogNormal(μ_neg, σ²)
    Pos ~ LogNormal(μ_pos, σ²)

    Both have long right tails. Discrimination depends on separation of log-means.

    ROC Derivation:
    ---------------
    If X ~ LogNormal(μ, σ²), then log(X) ~ N(μ, σ²)
    CDF: F(x) = Φ((log(x) - μ) / σ)

    FPR(t) = 1 - Φ((log(t) - μ_neg) / σ)
    TPR(t) = 1 - Φ((log(t) - μ_pos) / σ)

    From FPR: (log(t) - μ_neg) / σ = -Φ⁻¹(FPR)
              log(t) = μ_neg - σ·Φ⁻¹(FPR)

    TPR = 1 - Φ((μ_neg - σ·Φ⁻¹(FPR) - μ_pos) / σ)
        = 1 - Φ(-Φ⁻¹(FPR) - (μ_pos - μ_neg)/σ)
        = Φ(Φ⁻¹(FPR) + Δμ/σ)

    Interestingly, this is the SAME as the Gaussian ROC with d' = Δμ/σ!
    This is because log-transform makes it Gaussian.
    """
    d_prime = (pos_mu - neg_mu) / sigma

    def generator(n_pos, n_neg, rng):
        scores_neg = rng.lognormal(neg_mu, sigma, n_neg)
        scores_pos = rng.lognormal(pos_mu, sigma, n_pos)
        return scores_pos, scores_neg

    def true_roc(fpr):
        fpr = np.asarray(fpr)
        fpr_clipped = np.clip(fpr, 1e-10, 1 - 1e-10)
        return stats.norm.cdf(stats.norm.ppf(fpr_clipped) + d_prime)

    return DGP(
        generator=generator,
        true_roc=true_roc,
        name=f"LogNormal(Δμ={pos_mu - neg_mu}, σ={sigma})",
        description="Both classes right-skewed. ROC equivalent to Gaussian (log-transform).",
    )


def make_gamma_dgp(
    neg_shape: float = 2.0,
    pos_shape: float = 2.0,
    neg_scale: float = 1.0,
    pos_scale: float = 2.0,
) -> DGP:
    """
    Both classes right-skewed (Gamma).

    Neg ~ Gamma(k_neg, θ_neg)
    Pos ~ Gamma(k_pos, θ_pos)

    Unlike log-normal, no closed-form ROC. Uses numerical inversion.

    ROC Derivation:
    ---------------
    CDF: F(x; k, θ) = γ(k, x/θ) / Γ(k) (lower incomplete gamma ratio)

    FPR(t) = 1 - F_neg(t)
    TPR(t) = 1 - F_pos(t)

    Invert FPR numerically to find t, then compute TPR.
    """
    neg_dist = stats.gamma(neg_shape, scale=neg_scale)
    pos_dist = stats.gamma(pos_shape, scale=pos_scale)

    def generator(n_pos, n_neg, rng):
        scores_neg = rng.gamma(neg_shape, neg_scale, n_neg)
        scores_pos = rng.gamma(pos_shape, pos_scale, n_pos)
        return scores_pos, scores_neg

    def true_roc(fpr):
        fpr = np.asarray(fpr)
        tpr = np.zeros_like(fpr, dtype=float)

        # Numerical bounds for threshold search
        t_min = min(neg_dist.ppf(1e-10), pos_dist.ppf(1e-10))
        t_max = max(neg_dist.ppf(1 - 1e-10), pos_dist.ppf(1 - 1e-10))

        for i, f in enumerate(fpr.flat):
            if f <= 0:
                tpr.flat[i] = 0.0
            elif f >= 1:
                tpr.flat[i] = 1.0
            else:
                # Find t such that 1 - F_neg(t) = f
                try:
                    t = brentq(lambda t: (1 - neg_dist.cdf(t)) - f, t_min, t_max)
                    tpr.flat[i] = 1 - pos_dist.cdf(t)
                except ValueError:
                    tpr.flat[i] = np.nan

        return tpr

    return DGP(
        generator=generator,
        true_roc=true_roc,
        name=f"Gamma(neg_k={neg_shape}, pos_k={pos_shape})",
        description="Both classes right-skewed Gamma. Different shapes create asymmetric ROC.",
    )


# =============================================================================
# 4. HETEROSKEDASTIC: Unequal Variances
# =============================================================================


def make_heteroskedastic_gaussian_dgp(
    delta_mu: float = 1.0, sigma_neg: float = 1.0, sigma_pos: float = 2.0
) -> DGP:
    """
    Gaussian with different variances.

    Neg ~ N(0, σ_neg²)
    Pos ~ N(Δμ, σ_pos²)

    When σ_pos > σ_neg, the ROC curve bows differently than equal-variance case.
    Creates "improper" ROC if variances are very different (curve crosses diagonal).

    ROC Derivation:
    ---------------
    FPR(t) = 1 - Φ(t/σ_neg)
    TPR(t) = 1 - Φ((t - Δμ)/σ_pos)

    From FPR: t = -σ_neg · Φ⁻¹(FPR)
    TPR = 1 - Φ((-σ_neg · Φ⁻¹(FPR) - Δμ) / σ_pos)
        = Φ((Δμ + σ_neg · Φ⁻¹(FPR)) / σ_pos)
        = Φ(Δμ/σ_pos + (σ_neg/σ_pos) · Φ⁻¹(FPR))
    """
    ratio = sigma_neg / sigma_pos
    intercept = delta_mu / sigma_pos

    def generator(n_pos, n_neg, rng):
        scores_neg = rng.normal(0, sigma_neg, n_neg)
        scores_pos = rng.normal(delta_mu, sigma_pos, n_pos)
        return scores_pos, scores_neg

    def true_roc(fpr):
        fpr = np.asarray(fpr)
        fpr_clipped = np.clip(fpr, 1e-10, 1 - 1e-10)
        return stats.norm.cdf(intercept + ratio * stats.norm.ppf(fpr_clipped))

    return DGP(
        generator=generator,
        true_roc=true_roc,
        name=f"Hetero_Gaussian(Δμ={delta_mu}, σ_neg={sigma_neg}, σ_pos={sigma_pos})",
        description=f"Unequal variance Gaussian. σ_pos/σ_neg = {sigma_pos / sigma_neg:.2f} affects ROC curvature.",
    )


# =============================================================================
# 5. BIMODAL / MIXTURE OF GAUSSIANS
# =============================================================================


def make_gaussian_mixture_dgp(
    neg_means: list = [0.0],
    neg_stds: list = [1.0],
    neg_weights: list = [1.0],
    pos_means: list = [1.5],
    pos_stds: list = [1.0],
    pos_weights: list = [1.0],
) -> DGP:
    """
    General Gaussian mixture for both classes.

    Neg ~ Σᵢ wᵢ · N(μᵢ, σᵢ²)
    Pos ~ Σⱼ vⱼ · N(νⱼ, τⱼ²)

    Creates multi-modal distributions. Can model subpopulations with
    different characteristics.

    ROC Derivation:
    ---------------
    CDF of mixture: F(t) = Σᵢ wᵢ · Φ((t - μᵢ) / σᵢ)

    FPR(t) = 1 - F_neg(t)
    TPR(t) = 1 - F_pos(t)

    Invert FPR numerically (mixture CDF not analytically invertible).

    Example configurations:
    - Bimodal negative: neg_means=[0, 3], neg_weights=[0.7, 0.3]
    - Bimodal positive: pos_means=[2, 5], pos_weights=[0.5, 0.5]
    """
    # Normalize weights
    neg_weights = np.array(neg_weights) / np.sum(neg_weights)
    pos_weights = np.array(pos_weights) / np.sum(pos_weights)

    neg_means = np.array(neg_means)
    neg_stds = np.array(neg_stds)
    pos_means = np.array(pos_means)
    pos_stds = np.array(pos_stds)

    def mixture_cdf(t, means, stds, weights):
        """CDF of Gaussian mixture."""
        return np.sum(
            [
                w * stats.norm.cdf(t, loc=m, scale=s)
                for w, m, s in zip(weights, means, stds)
            ],
            axis=0,
        )

    def mixture_sf(t, means, stds, weights):
        """Survival function (1 - CDF) of Gaussian mixture."""
        return 1 - mixture_cdf(t, means, stds, weights)

    def generator(n_pos, n_neg, rng):
        # Sample component indices
        neg_components = rng.choice(len(neg_means), size=n_neg, p=neg_weights)
        pos_components = rng.choice(len(pos_means), size=n_pos, p=pos_weights)

        # Sample from selected components
        scores_neg = rng.normal(neg_means[neg_components], neg_stds[neg_components])
        scores_pos = rng.normal(pos_means[pos_components], pos_stds[pos_components])

        return scores_pos, scores_neg

    def true_roc(fpr):
        fpr = np.asarray(fpr)
        tpr = np.zeros_like(fpr, dtype=float)

        # Determine search bounds
        all_means = np.concatenate([neg_means, pos_means])
        all_stds = np.concatenate([neg_stds, pos_stds])
        t_min = all_means.min() - 5 * all_stds.max()
        t_max = all_means.max() + 5 * all_stds.max()

        for i, f in enumerate(fpr.flat):
            if f <= 0:
                tpr.flat[i] = 0.0
            elif f >= 1:
                tpr.flat[i] = 1.0
            else:
                # Find t such that SF_neg(t) = f
                try:
                    t = brentq(
                        lambda t: mixture_sf(t, neg_means, neg_stds, neg_weights) - f,
                        t_min,
                        t_max,
                    )
                    tpr.flat[i] = mixture_sf(t, pos_means, pos_stds, pos_weights)
                except ValueError:
                    tpr.flat[i] = np.nan

        return tpr

    desc = f"Neg: {len(neg_means)} components, Pos: {len(pos_means)} components"
    return DGP(
        generator=generator,
        true_roc=true_roc,
        name=f"GaussianMixture({desc})",
        description="Gaussian mixture. Can create bimodal/multimodal score distributions.",
    )


def make_bimodal_negative_dgp(
    neg_means: tuple = (0.0, 2.5),
    neg_stds: tuple = (0.8, 0.8),
    neg_weights: tuple = (0.7, 0.3),
    pos_mean: float = 1.5,
    pos_std: float = 1.0,
) -> DGP:
    """
    Bimodal negative class, unimodal positive class.

    Models scenario where negatives have two subpopulations
    (e.g., "easy" negatives far from positives and "hard" negatives
    that overlap with positives).

    This can create interesting ROC behavior with inflection points.
    """
    return make_gaussian_mixture_dgp(
        neg_means=list(neg_means),
        neg_stds=list(neg_stds),
        neg_weights=list(neg_weights),
        pos_means=[pos_mean],
        pos_stds=[pos_std],
        pos_weights=[1.0],
    )


def make_bimodal_both_dgp(
    neg_means: tuple = (0.0, 3.0),
    neg_stds: tuple = (0.7, 0.7),
    neg_weights: tuple = (0.6, 0.4),
    pos_means: tuple = (1.5, 4.5),
    pos_stds: tuple = (0.7, 0.7),
    pos_weights: tuple = (0.5, 0.5),
) -> DGP:
    """
    Both classes bimodal.

    Creates complex ROC with potentially multiple inflection points.
    Models scenarios with distinct subpopulations in both classes.
    """
    return make_gaussian_mixture_dgp(
        neg_means=list(neg_means),
        neg_stds=list(neg_stds),
        neg_weights=list(neg_weights),
        pos_means=list(pos_means),
        pos_stds=list(pos_stds),
        pos_weights=list(pos_weights),
    )


# =============================================================================
# 6. HEAVY TAILS: Student's t and Related
# =============================================================================


def make_student_t_dgp(
    df: float = 3.0, delta_loc: float = 1.0, scale: float = 1.0
) -> DGP:
    """
    Heavy-tailed Student's t distribution.

    Neg ~ t_ν(loc=0, scale=σ)
    Pos ~ t_ν(loc=Δ, scale=σ)

    Lower df = heavier tails. df=1 is Cauchy (very heavy).
    df→∞ converges to Gaussian.

    Heavy tails mean more extreme scores in both classes, affecting
    behavior at extreme FPR values.

    ROC Derivation:
    ---------------
    Uses scipy's t distribution CDF/PPF for numerical computation.
    No closed form due to t-distribution complexity.
    """
    neg_dist = stats.t(df=df, loc=0, scale=scale)
    pos_dist = stats.t(df=df, loc=delta_loc, scale=scale)

    def generator(n_pos, n_neg, rng):
        scores_neg = neg_dist.rvs(size=n_neg, random_state=rng)
        scores_pos = pos_dist.rvs(size=n_pos, random_state=rng)
        return scores_pos, scores_neg

    def true_roc(fpr):
        fpr = np.asarray(fpr)
        fpr_clipped = np.clip(fpr, 1e-10, 1 - 1e-10)

        # Threshold at each FPR: t such that P(Neg > t) = FPR
        # P(Neg > t) = 1 - F_neg(t) = FPR => F_neg(t) = 1 - FPR
        thresholds = neg_dist.ppf(1 - fpr_clipped)

        # TPR at each threshold
        tpr = 1 - pos_dist.cdf(thresholds)

        return tpr

    return DGP(
        generator=generator,
        true_roc=true_roc,
        name=f"StudentT(df={df}, Δ={delta_loc})",
        description=f"Heavy-tailed t-distribution. df={df} gives excess kurtosis = {6 / (df - 4) if df > 4 else '∞'}.",
    )


def make_cauchy_dgp(delta_loc: float = 1.0, scale: float = 1.0) -> DGP:
    """
    Cauchy distribution (t with df=1). Extremely heavy tails.

    Mean and variance are undefined. Median separation is delta_loc.

    This is a stress test for confidence band methods - extreme values
    are common and can cause issues with bootstrap methods.
    """
    return make_student_t_dgp(df=1.0, delta_loc=delta_loc, scale=scale)


def make_pareto_dgp(
    neg_alpha: float = 3.0,
    pos_alpha: float = 2.0,
    neg_scale: float = 1.0,
    pos_scale: float = 1.0,
) -> DGP:
    """
    Pareto distribution (power-law tails).

    Neg ~ Pareto(α_neg, scale_neg)  (minimum value = scale)
    Pos ~ Pareto(α_pos, scale_pos)

    PDF: f(x) = α·x_m^α / x^(α+1) for x ≥ x_m

    Lower α = heavier tail. Variance undefined for α ≤ 2, mean undefined for α ≤ 1.

    Common in modeling wealth, city sizes, word frequencies, etc.

    ROC Derivation:
    ---------------
    CDF: F(x) = 1 - (x_m / x)^α for x ≥ x_m
    SF: S(x) = (x_m / x)^α

    FPR(t) = (scale_neg / t)^α_neg
    t = scale_neg · FPR^(-1/α_neg)

    TPR(t) = (scale_pos / t)^α_pos if t ≥ scale_pos, else 1
    """

    def generator(n_pos, n_neg, rng):
        # Pareto: scipy uses shape=alpha, scale=x_m
        scores_neg = stats.pareto.rvs(
            neg_alpha, scale=neg_scale, size=n_neg, random_state=rng
        )
        scores_pos = stats.pareto.rvs(
            pos_alpha, scale=pos_scale, size=n_pos, random_state=rng
        )
        return scores_pos, scores_neg

    def true_roc(fpr):
        fpr = np.asarray(fpr)
        tpr = np.zeros_like(fpr, dtype=float)

        for i, f in enumerate(fpr.flat):
            if f <= 0:
                tpr.flat[i] = 0.0
            elif f >= 1:
                tpr.flat[i] = 1.0
            else:
                # Threshold where FPR = f
                # (neg_scale / t)^neg_alpha = f
                # t = neg_scale * f^(-1/neg_alpha)
                t = neg_scale * (f ** (-1.0 / neg_alpha))

                # TPR at this threshold
                if t < pos_scale:
                    tpr.flat[i] = 1.0
                else:
                    tpr.flat[i] = (pos_scale / t) ** pos_alpha

        return tpr

    return DGP(
        generator=generator,
        true_roc=true_roc,
        name=f"Pareto(α_neg={neg_alpha}, α_pos={pos_alpha})",
        description="Power-law tails. Tests behavior with extreme values.",
    )


# =============================================================================
# 7. UNIFORM DISTRIBUTIONS
# =============================================================================


def make_uniform_dgp(
    neg_low: float = 0.0,
    neg_high: float = 2.0,
    pos_low: float = 1.0,
    pos_high: float = 3.0,
) -> DGP:
    """
    Uniform distributions with possible overlap.

    Neg ~ Uniform(a, b)
    Pos ~ Uniform(c, d)

    The ROC is piecewise linear with at most 3 segments.

    ROC Derivation:
    ---------------
    CDF_neg(t) = (t - a) / (b - a) for a ≤ t ≤ b
    CDF_pos(t) = (t - c) / (d - c) for c ≤ t ≤ d

    FPR(t) = 1 - CDF_neg(t) = (b - t) / (b - a)
    TPR(t) = 1 - CDF_pos(t) = (d - t) / (d - c)

    Case analysis (assuming a < b, c < d):

    1. No overlap (b ≤ c): Perfect separation, ROC goes (0,0)→(0,1)→(1,1)
    2. Full overlap (a=c, b=d): ROC is diagonal
    3. Partial overlap: Piecewise linear

    For partial overlap with a < c < b < d:
    - t > b: FPR = 0, varies TPR
    - c ≤ t ≤ b: Both vary linearly, creating diagonal segment
    - t < c: TPR = 1, varies FPR
    """
    a, b = neg_low, neg_high
    c, d = pos_low, pos_high

    def generator(n_pos, n_neg, rng):
        scores_neg = rng.uniform(a, b, n_neg)
        scores_pos = rng.uniform(c, d, n_pos)
        return scores_pos, scores_neg

    def true_roc(fpr):
        fpr = np.asarray(fpr)
        tpr = np.zeros_like(fpr, dtype=float)

        for i, f in enumerate(fpr.flat):
            if f <= 0:
                tpr.flat[i] = 0.0
            elif f >= 1:
                tpr.flat[i] = 1.0
            else:
                # Threshold t such that FPR(t) = f
                # (b - t) / (b - a) = f => t = b - f * (b - a)
                t = b - f * (b - a)

                # TPR at this threshold
                if t >= d:
                    tpr.flat[i] = 0.0
                elif t <= c:
                    tpr.flat[i] = 1.0
                else:
                    tpr.flat[i] = (d - t) / (d - c)

        return tpr

    # Characterize overlap
    overlap = max(0, min(b, d) - max(a, c))
    total_range = max(b, d) - min(a, c)
    overlap_ratio = overlap / total_range if total_range > 0 else 0

    return DGP(
        generator=generator,
        true_roc=true_roc,
        name=f"Uniform(neg=[{a},{b}], pos=[{c},{d}])",
        description=f"Uniform distributions. Overlap ratio: {overlap_ratio:.2f}. ROC is piecewise linear.",
    )


def make_uniform_no_overlap_dgp(gap: float = 0.5) -> DGP:
    """Perfect separation: negatives and positives don't overlap."""
    return make_uniform_dgp(neg_low=0, neg_high=1, pos_low=1 + gap, pos_high=2 + gap)


def make_uniform_full_overlap_dgp() -> DGP:
    """Complete overlap: ROC is the diagonal (random classifier)."""
    return make_uniform_dgp(neg_low=0, neg_high=1, pos_low=0, pos_high=1)


def make_uniform_partial_overlap_dgp(overlap_fraction: float = 0.5) -> DGP:
    """Controlled overlap between uniform distributions."""
    # neg: [0, 1], pos: [1-overlap, 2-overlap]
    return make_uniform_dgp(
        neg_low=0,
        neg_high=1,
        pos_low=1 - overlap_fraction,
        pos_high=2 - overlap_fraction,
    )


# =============================================================================
# 8. ADDITIONAL INTERESTING CASES
# =============================================================================


def make_exponential_dgp(neg_rate: float = 1.0, pos_rate: float = 0.5) -> DGP:
    """
    Exponential distributions (memoryless, one-sided heavy tail).

    Neg ~ Exp(λ_neg)  (rate parameterization, mean = 1/λ)
    Pos ~ Exp(λ_pos)

    Lower rate = higher mean = scores shifted right.

    ROC Derivation:
    ---------------
    CDF: F(x) = 1 - exp(-λx)
    SF: S(x) = exp(-λx)

    FPR(t) = exp(-λ_neg · t)
    t = -ln(FPR) / λ_neg

    TPR(t) = exp(-λ_pos · t) = exp(λ_pos · ln(FPR) / λ_neg)
           = FPR^(λ_pos / λ_neg)

    The ROC is a power function!
    """
    power = pos_rate / neg_rate

    def generator(n_pos, n_neg, rng):
        scores_neg = rng.exponential(1.0 / neg_rate, n_neg)
        scores_pos = rng.exponential(1.0 / pos_rate, n_pos)
        return scores_pos, scores_neg

    def true_roc(fpr):
        fpr = np.asarray(fpr)
        return np.power(np.clip(fpr, 0, 1), power)

    return DGP(
        generator=generator,
        true_roc=true_roc,
        name=f"Exponential(λ_neg={neg_rate}, λ_pos={pos_rate})",
        description=f"Exponential. ROC is power function: TPR = FPR^{power:.2f}",
    )


def make_weibull_dgp(
    neg_shape: float = 1.5,
    pos_shape: float = 1.5,
    neg_scale: float = 1.0,
    pos_scale: float = 2.0,
) -> DGP:
    """
    Weibull distributions (flexible hazard rates).

    Shape < 1: decreasing hazard (heavy right tail)
    Shape = 1: exponential (constant hazard)
    Shape > 1: increasing hazard (light right tail)

    Commonly used in survival analysis and reliability.
    """
    neg_dist = stats.weibull_min(neg_shape, scale=neg_scale)
    pos_dist = stats.weibull_min(pos_shape, scale=pos_scale)

    def generator(n_pos, n_neg, rng):
        scores_neg = neg_dist.rvs(size=n_neg, random_state=rng)
        scores_pos = pos_dist.rvs(size=n_pos, random_state=rng)
        return scores_pos, scores_neg

    def true_roc(fpr):
        fpr = np.asarray(fpr)
        fpr_clipped = np.clip(fpr, 1e-10, 1 - 1e-10)
        thresholds = neg_dist.ppf(1 - fpr_clipped)
        return pos_dist.sf(thresholds)

    return DGP(
        generator=generator,
        true_roc=true_roc,
        name=f"Weibull(k_neg={neg_shape}, k_pos={pos_shape})",
        description="Weibull. Shape parameter controls tail behavior.",
    )


# =============================================================================
# CONVENIENCE: Get All DGPs for Testing
# =============================================================================


def get_standard_test_dgps() -> dict:
    """
    Returns a dictionary of standard test DGPs covering various scenarios.

    Useful for systematic evaluation of confidence band methods.
    """
    return {
        # Baseline
        "gaussian_d1": make_gaussian_dgp(delta_mu=1.0),
        "gaussian_d2": make_gaussian_dgp(delta_mu=2.0),
        # Skewed toward each other
        "beta_opposing": make_beta_opposing_skew_dgp(alpha=2, beta=5),
        "beta_mild_opposing": make_beta_same_support_opposing_dgp(2, 3, 3, 2),
        # Skewed same direction
        "lognormal": make_lognormal_dgp(neg_mu=0, pos_mu=0.5, sigma=0.5),
        "gamma": make_gamma_dgp(neg_shape=2, pos_shape=2, neg_scale=1, pos_scale=2),
        # Heteroskedastic
        "hetero_pos_wider": make_heteroskedastic_gaussian_dgp(1.0, 1.0, 2.0),
        "hetero_neg_wider": make_heteroskedastic_gaussian_dgp(1.0, 2.0, 1.0),
        # Bimodal
        "bimodal_neg": make_bimodal_negative_dgp(),
        "bimodal_both": make_bimodal_both_dgp(),
        # Heavy tails
        "student_t3": make_student_t_dgp(df=3),
        "student_t5": make_student_t_dgp(df=5),
        "cauchy": make_cauchy_dgp(delta_loc=1.0),
        "pareto": make_pareto_dgp(neg_alpha=3, pos_alpha=2),
        # Uniform
        "uniform_partial": make_uniform_partial_overlap_dgp(0.5),
        "uniform_high_overlap": make_uniform_partial_overlap_dgp(0.8),
        # Other
        "exponential": make_exponential_dgp(neg_rate=1.0, pos_rate=0.5),
        "weibull": make_weibull_dgp(
            neg_shape=1.5, pos_shape=1.5, neg_scale=1, pos_scale=2
        ),
    }


def visualize_dgp(dgp: DGP, n_samples: int = 5000, seed: int = 42):
    """
    Visualize a DGP: score distributions and true ROC curve.

    Returns matplotlib figure (requires matplotlib).
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    scores_pos, scores_neg = dgp.sample(n_samples, n_samples, rng)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Score distributions
    ax = axes[0]
    bins = np.linspace(
        min(scores_neg.min(), scores_pos.min()),
        max(scores_neg.max(), scores_pos.max()),
        50,
    )
    ax.hist(scores_neg, bins=bins, alpha=0.5, label="Negative", density=True)
    ax.hist(scores_pos, bins=bins, alpha=0.5, label="Positive", density=True)
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title(f"{dgp.name}\nScore Distributions")
    ax.legend()

    # ROC curve
    ax = axes[1]
    fpr = np.linspace(0, 1, 201)
    tpr = dgp.get_true_roc(fpr)
    ax.plot(fpr, tpr, "b-", linewidth=2, label="True ROC")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"True ROC Curve\nAUC = {np.trapezoid(tpr, fpr):.3f}")
    ax.legend()
    ax.set_aspect("equal")

    plt.tight_layout()
    return fig


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    # Test each DGP
    print("Testing DGPs...\n")

    fpr_grid = np.linspace(0, 1, 101)

    for name, dgp in get_standard_test_dgps().items():
        print(f"{name}:")
        print(f"  {dgp.description}")

        # Verify ROC is valid
        tpr = dgp.get_true_roc(fpr_grid)
        auc = np.trapezoid(tpr, fpr_grid)

        # Check monotonicity
        is_monotonic = np.all(np.diff(tpr) >= -1e-10)

        # Check bounds
        in_bounds = np.all((tpr >= 0) & (tpr <= 1))

        print(f"  AUC: {auc:.3f}, Monotonic: {is_monotonic}, Valid bounds: {in_bounds}")
        print()
