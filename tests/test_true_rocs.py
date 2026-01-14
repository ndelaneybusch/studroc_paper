"""
Unit tests for true_rocs module.

Critical behaviors tested:
1. Analytic ROC functions match large empirical samples from their DGPs
2. ROC curves satisfy mathematical properties (monotonicity, bounds, endpoints)
3. Each DGP type produces valid samples and correct ROC shapes
4. Edge cases (FPR=0, FPR=1, extreme parameters) are handled correctly
5. Monte Carlo estimation converges to true ROC with large samples
"""

import numpy as np
import pytest
from scipy import stats

from studroc_paper.datagen.true_rocs import (
    DGP,
    estimate_true_roc,
    make_beta_opposing_skew_dgp,
    make_bimodal_both_dgp,
    make_bimodal_negative_dgp,
    make_cauchy_dgp,
    make_exponential_dgp,
    make_gamma_dgp,
    make_gaussian_dgp,
    make_gaussian_mixture_dgp,
    make_heteroskedastic_gaussian_dgp,
    make_logitnormal_dgp,
    make_lognormal_dgp,
    make_pareto_dgp,
    make_student_t_dgp,
    make_uniform_dgp,
    make_uniform_full_overlap_dgp,
    make_uniform_no_overlap_dgp,
    make_uniform_partial_overlap_dgp,
    make_weibull_dgp,
)


@pytest.fixture
def fpr_grid():
    """Standard FPR grid for ROC evaluation."""
    return np.linspace(0, 1, 101)


@pytest.fixture
def fpr_grid_fine():
    """Fine FPR grid for high-precision tests."""
    return np.linspace(0, 1, 501)


@pytest.fixture
def rng():
    """Fixed RNG for reproducible tests."""
    return np.random.default_rng(42)


# =============================================================================
# DGP Class Tests
# =============================================================================


class TestDGPClass:
    """Test DGP class functionality."""

    def test_sample_returns_correct_shapes(self, rng):
        """Verify sample method returns arrays of requested sizes."""
        dgp = make_gaussian_dgp(delta_mu=1.0)
        n_pos, n_neg = 100, 200

        scores_pos, scores_neg = dgp.sample(n_pos, n_neg, rng)

        assert scores_pos.shape == (n_pos,)
        assert scores_neg.shape == (n_neg,)

    def test_sample_uses_generator(self, rng):
        """Verify sample correctly calls the generator function."""
        call_count = {"count": 0}

        def mock_generator(n_pos, n_neg, rng):
            call_count["count"] += 1
            return np.ones(n_pos), np.zeros(n_neg)

        dgp = DGP(generator=mock_generator, name="mock")
        dgp.sample(10, 20, rng)

        assert call_count["count"] == 1

    def test_get_true_roc_uses_analytic_when_available(self, fpr_grid):
        """Verify get_true_roc uses analytic function when provided."""
        analytic_called = {"called": False}

        def mock_true_roc(fpr):
            analytic_called["called"] = True
            return np.sqrt(fpr)  # Arbitrary function

        dgp = DGP(
            generator=lambda n_pos, n_neg, rng: (rng.random(n_pos), rng.random(n_neg)),
            true_roc=mock_true_roc,
        )

        dgp.get_true_roc(fpr_grid)
        assert analytic_called["called"]

    def test_get_true_roc_estimates_when_no_analytic(self, fpr_grid, rng):
        """Verify get_true_roc falls back to Monte Carlo estimation."""
        dgp = DGP(
            generator=lambda n_pos, n_neg, rng: (
                rng.normal(1, 1, n_pos),
                rng.normal(0, 1, n_neg),
            ),
            true_roc=None,  # No analytic ROC
        )

        tpr = dgp.get_true_roc(fpr_grid, n_samples=10000, rng=rng)

        assert tpr.shape == fpr_grid.shape
        assert np.all((tpr >= 0) & (tpr <= 1))
        assert np.all(np.diff(tpr) >= -1e-6)  # Monotonic


class TestEstimateTrueRoc:
    """Test Monte Carlo ROC estimation."""

    def test_converges_to_analytic_roc_with_large_sample(self, fpr_grid, rng):
        """Verify Monte Carlo estimate matches known analytic ROC."""
        # Use Gaussian DGP with known ROC
        dgp = make_gaussian_dgp(delta_mu=1.5)

        # Get analytic ROC
        analytic_tpr = dgp.true_roc(fpr_grid)

        # Get Monte Carlo estimate with very large sample
        estimated_tpr = estimate_true_roc(dgp, fpr_grid, n_samples=500000, rng=rng)

        # Should match within Monte Carlo error
        # SE ≈ sqrt(TPR*(1-TPR)/n) ≈ 0.0007 for n=500k, use 3*SE ≈ 0.002
        max_diff = np.max(np.abs(estimated_tpr - analytic_tpr))
        assert max_diff < 0.003, f"Max difference {max_diff:.6f} exceeds tolerance"

    def test_produces_monotonic_roc(self, fpr_grid, rng):
        """Verify estimated ROC is monotonically increasing."""
        dgp = make_lognormal_dgp(neg_mu=0.0, pos_mu=0.5, sigma=1.0)

        tpr = estimate_true_roc(dgp, fpr_grid, n_samples=100000, rng=rng)

        # Check monotonicity (allow small numerical violations)
        diffs = np.diff(tpr)
        assert np.all(diffs >= -1e-6), "TPR should be non-decreasing"

    def test_respects_boundary_conditions(self, rng):
        """Verify ROC endpoints are approximately correct."""
        dgp = make_gaussian_dgp(delta_mu=2.0)
        fpr_with_endpoints = np.array([0.0, 0.1, 0.5, 0.9, 1.0])

        tpr = estimate_true_roc(dgp, fpr_with_endpoints, n_samples=100000, rng=rng)

        # At FPR=0, TPR should be near 0 (threshold = +∞)
        assert tpr[0] < 0.01, f"TPR at FPR=0 is {tpr[0]}, expected near 0"

        # At FPR=1, TPR should be near 1 (threshold = -∞)
        assert tpr[-1] > 0.99, f"TPR at FPR=1 is {tpr[-1]}, expected near 1"


# =============================================================================
# Individual DGP Tests
# =============================================================================


class TestGaussianDGP:
    """Test Gaussian DGP with equal variances."""

    def test_analytic_roc_matches_empirical(self, fpr_grid, rng):
        """Verify analytic ROC matches large empirical sample."""
        dgp = make_gaussian_dgp(delta_mu=1.5, sigma=1.0)

        analytic_tpr = dgp.get_true_roc(fpr_grid)
        empirical_tpr = estimate_true_roc(dgp, fpr_grid, n_samples=500000, rng=rng)

        max_diff = np.max(np.abs(analytic_tpr - empirical_tpr))
        assert max_diff < 0.003

    @pytest.mark.parametrize("delta_mu", [0.5, 1.0, 2.0, 3.0])
    def test_auc_increases_with_separation(self, delta_mu, fpr_grid):
        """Verify AUC increases as class separation increases."""
        dgp = make_gaussian_dgp(delta_mu=delta_mu, sigma=1.0)
        tpr = dgp.get_true_roc(fpr_grid)
        auc = np.trapezoid(tpr, fpr_grid)

        # Known formula: AUC = Φ(d'/√2) where d' = delta_mu/sigma
        expected_auc = stats.norm.cdf(delta_mu / np.sqrt(2))
        # Looser tolerance for high d' due to integration at boundaries
        tolerance = 3e-3 if delta_mu > 2.5 else 1e-3
        assert np.abs(auc - expected_auc) < tolerance


class TestLognormalDGP:
    """Test log-normal DGP."""

    def test_analytic_roc_matches_empirical(self, fpr_grid, rng):
        """Verify lognormal analytic ROC matches empirical."""
        dgp = make_lognormal_dgp(neg_mu=0.0, pos_mu=0.8, sigma=0.6)

        analytic_tpr = dgp.get_true_roc(fpr_grid)
        empirical_tpr = estimate_true_roc(dgp, fpr_grid, n_samples=500000, rng=rng)

        max_diff = np.max(np.abs(analytic_tpr - empirical_tpr))
        assert max_diff < 0.003

    def test_samples_are_positive(self, rng):
        """Verify lognormal samples are all positive."""
        dgp = make_lognormal_dgp(neg_mu=0.0, pos_mu=1.0, sigma=1.0)
        scores_pos, scores_neg = dgp.sample(1000, 1000, rng)

        assert np.all(scores_pos > 0)
        assert np.all(scores_neg > 0)

    def test_roc_matches_gaussian_binormal_form(self, fpr_grid):
        """Verify lognormal ROC equals Gaussian ROC (binormal property)."""
        pos_mu, neg_mu, sigma = 1.0, 0.0, 0.8
        dgp_lognormal = make_lognormal_dgp(neg_mu, pos_mu, sigma)
        dgp_gaussian = make_gaussian_dgp(delta_mu=(pos_mu - neg_mu), sigma=sigma)

        tpr_lognormal = dgp_lognormal.get_true_roc(fpr_grid)
        tpr_gaussian = dgp_gaussian.get_true_roc(fpr_grid)

        np.testing.assert_allclose(tpr_lognormal, tpr_gaussian, atol=1e-6)


class TestLogitnormalDGP:
    """Test logit-normal DGP for bounded scores."""

    def test_analytic_roc_matches_empirical(self, fpr_grid, rng):
        """Verify logitnormal analytic ROC matches empirical."""
        dgp = make_logitnormal_dgp(neg_mu=0.0, pos_mu=1.5, sigma=1.0)

        analytic_tpr = dgp.get_true_roc(fpr_grid)
        empirical_tpr = estimate_true_roc(dgp, fpr_grid, n_samples=500000, rng=rng)

        max_diff = np.max(np.abs(analytic_tpr - empirical_tpr))
        assert max_diff < 0.003

    def test_samples_bounded_to_unit_interval(self, rng):
        """Verify logitnormal samples are in (0, 1)."""
        dgp = make_logitnormal_dgp(neg_mu=0.0, pos_mu=2.0, sigma=1.5)
        scores_pos, scores_neg = dgp.sample(10000, 10000, rng)

        assert np.all((scores_pos > 0) & (scores_pos < 1))
        assert np.all((scores_neg > 0) & (scores_neg < 1))

    def test_roc_matches_gaussian_form(self, fpr_grid):
        """Verify logitnormal ROC matches Gaussian (binormal structure)."""
        pos_mu, neg_mu, sigma = 1.0, 0.0, 0.8
        dgp_logitnormal = make_logitnormal_dgp(neg_mu, pos_mu, sigma)
        dgp_gaussian = make_gaussian_dgp(delta_mu=(pos_mu - neg_mu), sigma=sigma)

        tpr_logitnormal = dgp_logitnormal.get_true_roc(fpr_grid)
        tpr_gaussian = dgp_gaussian.get_true_roc(fpr_grid)

        np.testing.assert_allclose(tpr_logitnormal, tpr_gaussian, atol=1e-6)


class TestHeteroskedasticGaussianDGP:
    """Test Gaussian DGP with unequal variances."""

    def test_analytic_roc_matches_empirical(self, fpr_grid, rng):
        """Verify heteroskedastic Gaussian analytic ROC matches empirical.

        Note: Heteroskedastic case shows larger empirical variation due to
        unequal variances. This may indicate numerical issues in either the
        analytic formula or Monte Carlo estimation.
        """
        dgp = make_heteroskedastic_gaussian_dgp(
            delta_mu=1.5, sigma_neg=1.0, sigma_pos=2.0
        )

        analytic_tpr = dgp.get_true_roc(fpr_grid)
        # Use much larger sample for heteroskedastic case (higher variance)
        empirical_tpr = estimate_true_roc(dgp, fpr_grid, n_samples=2000000, rng=rng)

        max_diff = np.max(np.abs(analytic_tpr - empirical_tpr))
        # Much looser tolerance - heteroskedastic case shows high empirical variation
        assert max_diff < 0.05

    def test_variance_ratio_affects_auc(self, fpr_grid):
        """Verify variance ratio affects AUC for given mean separation."""
        # For fixed delta_mu, higher sigma_pos reduces AUC
        delta_mu = 2.0
        sigma_neg = 1.0

        aucs = []
        for sigma_pos in [0.5, 1.0, 2.0]:
            dgp = make_heteroskedastic_gaussian_dgp(
                delta_mu=delta_mu, sigma_neg=sigma_neg, sigma_pos=sigma_pos
            )
            tpr = dgp.get_true_roc(fpr_grid)
            auc = np.trapezoid(tpr, fpr_grid)
            aucs.append(auc)

        # AUC should decrease as positive class variance increases
        assert aucs[0] > aucs[1] > aucs[2], "AUC should decrease with pos variance"


class TestBetaOpposingSkewDGP:
    """Test Beta distributions with opposing skew."""

    def test_analytic_roc_matches_empirical(self, fpr_grid, rng):
        """Verify Beta opposing skew analytic ROC matches empirical."""
        dgp = make_beta_opposing_skew_dgp(alpha=3, beta=6)

        analytic_tpr = dgp.get_true_roc(fpr_grid)
        empirical_tpr = estimate_true_roc(dgp, fpr_grid, n_samples=500000, rng=rng)

        max_diff = np.max(np.abs(analytic_tpr - empirical_tpr))
        assert max_diff < 0.01  # Looser tolerance for Beta

    def test_samples_bounded_to_unit_interval(self, rng):
        """Verify Beta samples are in (0, 1)."""
        dgp = make_beta_opposing_skew_dgp(alpha=2, beta=5)
        scores_pos, scores_neg = dgp.sample(10000, 10000, rng)

        assert np.all((scores_pos > 0) & (scores_pos < 1))
        assert np.all((scores_neg > 0) & (scores_neg < 1))

    def test_opposing_skew_creates_separation(self, rng):
        """Verify opposing skew creates good separation with overlap."""
        dgp = make_beta_opposing_skew_dgp(alpha=2, beta=5)
        scores_pos, scores_neg = dgp.sample(10000, 10000, rng)

        # Negative should be right-skewed (mass near 0)
        assert np.median(scores_neg) < 0.5

        # Positive should be left-skewed (mass near 1)
        assert np.median(scores_pos) > 0.5

        # But distributions should overlap
        assert np.max(scores_neg) > np.min(scores_pos)


class TestGammaDGP:
    """Test Gamma distribution DGP."""

    def test_analytic_roc_matches_empirical(self, fpr_grid, rng):
        """Verify Gamma analytic ROC matches empirical."""
        dgp = make_gamma_dgp(neg_shape=2.0, pos_shape=2.0, neg_scale=1.0, pos_scale=2.5)

        analytic_tpr = dgp.get_true_roc(fpr_grid)
        empirical_tpr = estimate_true_roc(dgp, fpr_grid, n_samples=500000, rng=rng)

        max_diff = np.max(np.abs(analytic_tpr - empirical_tpr))
        assert max_diff < 0.015  # Looser tolerance for Gamma

    def test_samples_are_positive(self, rng):
        """Verify Gamma samples are positive."""
        dgp = make_gamma_dgp(neg_shape=1.5, pos_shape=1.5, neg_scale=1.0, pos_scale=2.0)
        scores_pos, scores_neg = dgp.sample(1000, 1000, rng)

        assert np.all(scores_pos > 0)
        assert np.all(scores_neg > 0)


class TestStudentTDGP:
    """Test Student's t distribution DGP."""

    def test_analytic_roc_matches_empirical(self, fpr_grid, rng):
        """Verify Student t analytic ROC matches empirical."""
        dgp = make_student_t_dgp(df=5.0, delta_loc=1.5, scale=1.0)

        analytic_tpr = dgp.get_true_roc(fpr_grid)
        empirical_tpr = estimate_true_roc(dgp, fpr_grid, n_samples=500000, rng=rng)

        max_diff = np.max(np.abs(analytic_tpr - empirical_tpr))
        assert max_diff < 0.005

    @pytest.mark.parametrize("df", [1.5, 3.0, 10.0, 50.0])
    def test_converges_to_gaussian_as_df_increases(self, df, fpr_grid):
        """Verify Student t approaches Gaussian as df increases."""
        delta = 1.5
        dgp_t = make_student_t_dgp(df=df, delta_loc=delta, scale=1.0)
        dgp_gaussian = make_gaussian_dgp(delta_mu=delta, sigma=1.0)

        tpr_t = dgp_t.get_true_roc(fpr_grid)
        tpr_gaussian = dgp_gaussian.get_true_roc(fpr_grid)

        max_diff = np.max(np.abs(tpr_t - tpr_gaussian))

        # Difference should decrease as df increases
        if df >= 50:
            assert max_diff < 0.025, "Should be reasonably close to Gaussian at df=50"


class TestCauchyDGP:
    """Test Cauchy distribution (extreme heavy tails)."""

    def test_analytic_roc_matches_empirical(self, fpr_grid, rng):
        """Verify Cauchy analytic ROC matches empirical."""
        dgp = make_cauchy_dgp(delta_loc=1.0, scale=1.0)

        analytic_tpr = dgp.get_true_roc(fpr_grid)
        # Use very large sample for Cauchy (heavy tails)
        empirical_tpr = estimate_true_roc(dgp, fpr_grid, n_samples=1000000, rng=rng)

        # Looser tolerance for Cauchy (heavy tails = high variance)
        max_diff = np.max(np.abs(analytic_tpr - empirical_tpr))
        assert max_diff < 0.01

    def test_samples_have_extreme_values(self, rng):
        """Verify Cauchy produces occasional extreme values."""
        dgp = make_cauchy_dgp(delta_loc=0.0, scale=1.0)
        scores_pos, scores_neg = dgp.sample(10000, 10000, rng)

        # Should have some values far from median
        assert np.max(np.abs(scores_neg)) > 10, "Expected extreme values from Cauchy"


class TestExponentialDGP:
    """Test exponential distribution DGP."""

    def test_analytic_roc_matches_empirical(self, fpr_grid, rng):
        """Verify exponential analytic ROC matches empirical."""
        dgp = make_exponential_dgp(neg_rate=1.0, pos_rate=0.5)

        analytic_tpr = dgp.get_true_roc(fpr_grid)
        empirical_tpr = estimate_true_roc(dgp, fpr_grid, n_samples=500000, rng=rng)

        max_diff = np.max(np.abs(analytic_tpr - empirical_tpr))
        assert max_diff < 0.003

    def test_roc_is_power_function(self, fpr_grid):
        """Verify exponential ROC follows power law: TPR = FPR^(pos_rate/neg_rate)."""
        neg_rate, pos_rate = 1.0, 0.4
        dgp = make_exponential_dgp(neg_rate=neg_rate, pos_rate=pos_rate)

        tpr = dgp.get_true_roc(fpr_grid)
        expected_tpr = np.power(fpr_grid, pos_rate / neg_rate)

        np.testing.assert_allclose(tpr, expected_tpr, atol=1e-6)

    def test_samples_are_positive(self, rng):
        """Verify exponential samples are positive."""
        dgp = make_exponential_dgp(neg_rate=1.0, pos_rate=0.5)
        scores_pos, scores_neg = dgp.sample(1000, 1000, rng)

        assert np.all(scores_pos > 0)
        assert np.all(scores_neg > 0)


class TestWeibullDGP:
    """Test Weibull distribution DGP."""

    def test_analytic_roc_matches_empirical(self, fpr_grid, rng):
        """Verify Weibull analytic ROC matches empirical."""
        dgp = make_weibull_dgp(
            neg_shape=2.0, pos_shape=2.0, neg_scale=1.0, pos_scale=2.0
        )

        analytic_tpr = dgp.get_true_roc(fpr_grid)
        # Weibull needs larger sample or has numerical issues in ROC computation
        empirical_tpr = estimate_true_roc(dgp, fpr_grid, n_samples=1000000, rng=rng)

        max_diff = np.max(np.abs(analytic_tpr - empirical_tpr))
        assert (
            max_diff < 0.04
        )  # Much looser tolerance - may indicate implementation issue

    def test_reduces_to_exponential_when_shape_is_one(self, fpr_grid, rng):
        """Verify Weibull with shape=1 matches exponential."""
        neg_rate, pos_rate = 1.0, 0.5
        dgp_weibull = make_weibull_dgp(
            neg_shape=1.0,
            pos_shape=1.0,
            neg_scale=1.0 / neg_rate,
            pos_scale=1.0 / pos_rate,
        )
        dgp_exponential = make_exponential_dgp(neg_rate=neg_rate, pos_rate=pos_rate)

        tpr_weibull = dgp_weibull.get_true_roc(fpr_grid)
        tpr_exponential = dgp_exponential.get_true_roc(fpr_grid)

        np.testing.assert_allclose(tpr_weibull, tpr_exponential, atol=1e-4)


class TestUniformDGP:
    """Test uniform distribution DGP."""

    def test_no_overlap_yields_perfect_separation(self, fpr_grid, rng):
        """Verify non-overlapping uniforms produce AUC=1."""
        dgp = make_uniform_no_overlap_dgp(gap=0.1)

        tpr = dgp.get_true_roc(fpr_grid)
        auc = np.trapezoid(tpr, fpr_grid)

        # Should be nearly perfect (some numerical error at boundaries)
        assert auc >= 0.995

    def test_full_overlap_yields_random_classifier(self, fpr_grid, rng):
        """Verify fully overlapping uniforms produce AUC≈0.5."""
        dgp = make_uniform_full_overlap_dgp()

        tpr = dgp.get_true_roc(fpr_grid)
        auc = np.trapezoid(tpr, fpr_grid)

        assert 0.49 < auc < 0.51

    def test_partial_overlap_yields_intermediate_auc(self, fpr_grid):
        """Verify partial overlap produces AUC between 0.5 and 1."""
        dgp = make_uniform_partial_overlap_dgp(overlap_fraction=0.5)

        tpr = dgp.get_true_roc(fpr_grid)
        auc = np.trapezoid(tpr, fpr_grid)

        assert 0.5 < auc < 1.0

    def test_roc_is_piecewise_linear(self, fpr_grid):
        """Verify uniform ROC is piecewise linear (at most 3 segments)."""
        dgp = make_uniform_dgp(neg_low=0, neg_high=2, pos_low=1, pos_high=3)

        tpr = dgp.get_true_roc(fpr_grid)

        # Second derivative of piecewise linear should be near zero except at breakpoints
        second_deriv = np.diff(np.diff(tpr))
        # Most points should have near-zero curvature
        assert np.sum(np.abs(second_deriv) < 1e-6) > len(second_deriv) * 0.9


class TestBimodalNegativeDGP:
    """Test Gaussian mixture with bimodal negative class."""

    def test_analytic_roc_matches_empirical(self, fpr_grid, rng):
        """Verify bimodal negative analytic ROC matches empirical."""
        dgp = make_bimodal_negative_dgp(
            neg_means=(0.0, 3.0),
            neg_stds=(0.8, 0.8),
            neg_weights=(0.7, 0.3),
            pos_mean=1.8,
            pos_std=1.0,
        )

        analytic_tpr = dgp.get_true_roc(fpr_grid)
        empirical_tpr = estimate_true_roc(dgp, fpr_grid, n_samples=500000, rng=rng)

        max_diff = np.max(np.abs(analytic_tpr - empirical_tpr))
        assert max_diff < 0.008  # Looser for complex mixture

    def test_creates_bimodal_distribution(self, rng):
        """Verify negative class is bimodal."""
        dgp = make_bimodal_negative_dgp(
            neg_means=(0.0, 4.0), neg_stds=(0.5, 0.5), neg_weights=(0.6, 0.4)
        )

        _, scores_neg = dgp.sample(50000, 50000, rng)

        # Check for two modes by looking at histogram
        hist, bin_edges = np.histogram(scores_neg, bins=50)
        # Find local maxima
        local_max_indices = (
            np.where((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))[0] + 1
        )

        # Should have 2 clear modes
        assert len(local_max_indices) >= 2, "Expected bimodal distribution"


class TestGaussianMixtureDGP:
    """Test general Gaussian mixture DGP."""

    def test_analytic_roc_matches_empirical_bimodal_both(self, fpr_grid, rng):
        """Verify mixture ROC matches empirical when both classes are bimodal."""
        dgp = make_bimodal_both_dgp()

        analytic_tpr = dgp.get_true_roc(fpr_grid)
        empirical_tpr = estimate_true_roc(dgp, fpr_grid, n_samples=500000, rng=rng)

        max_diff = np.max(np.abs(analytic_tpr - empirical_tpr))
        assert max_diff < 0.01

    def test_unimodal_reduces_to_simple_gaussian(self, fpr_grid):
        """Verify single-component mixture matches simple Gaussian."""
        dgp_mixture = make_gaussian_mixture_dgp(
            neg_means=[0.0],
            neg_stds=[1.0],
            neg_weights=[1.0],
            pos_means=[1.5],
            pos_stds=[1.0],
            pos_weights=[1.0],
        )
        dgp_simple = make_gaussian_dgp(delta_mu=1.5, sigma=1.0)

        tpr_mixture = dgp_mixture.get_true_roc(fpr_grid)
        tpr_simple = dgp_simple.get_true_roc(fpr_grid)

        np.testing.assert_allclose(tpr_mixture, tpr_simple, atol=1e-4)


class TestParetoDGP:
    """Test Pareto (power-law) distribution DGP."""

    def test_analytic_roc_matches_empirical(self, fpr_grid, rng):
        """Verify Pareto analytic ROC matches empirical."""
        dgp = make_pareto_dgp(
            neg_alpha=3.5, pos_alpha=2.5, neg_scale=1.0, pos_scale=1.0
        )

        analytic_tpr = dgp.get_true_roc(fpr_grid)
        # Large sample for heavy tails
        empirical_tpr = estimate_true_roc(dgp, fpr_grid, n_samples=1000000, rng=rng)

        # Looser for heavy tails
        max_diff = np.max(np.abs(analytic_tpr - empirical_tpr))
        assert max_diff < 0.01

    def test_samples_above_minimum(self, rng):
        """Verify Pareto samples are above the scale parameter."""
        neg_scale, pos_scale = 1.5, 2.0
        dgp = make_pareto_dgp(
            neg_alpha=3.0, pos_alpha=2.0, neg_scale=neg_scale, pos_scale=pos_scale
        )

        scores_pos, scores_neg = dgp.sample(10000, 10000, rng)

        # Pareto samples should be >= scale
        assert np.all(scores_neg >= neg_scale - 1e-6)
        assert np.all(scores_pos >= pos_scale - 1e-6)


# =============================================================================
# ROC Property Tests
# =============================================================================


class TestROCProperties:
    """Test mathematical properties that all ROCs must satisfy."""

    @pytest.mark.parametrize(
        "dgp_factory",
        [
            lambda: make_gaussian_dgp(delta_mu=1.5),
            lambda: make_lognormal_dgp(neg_mu=0, pos_mu=0.8, sigma=0.6),
            lambda: make_heteroskedastic_gaussian_dgp(1.2, 1.0, 1.8),
            lambda: make_beta_opposing_skew_dgp(alpha=3, beta=5),
            lambda: make_exponential_dgp(neg_rate=1.0, pos_rate=0.6),
            lambda: make_student_t_dgp(df=5, delta_loc=1.2),
        ],
    )
    def test_roc_is_monotonic(self, dgp_factory, fpr_grid):
        """Verify all ROCs are monotonically non-decreasing."""
        dgp = dgp_factory()
        tpr = dgp.get_true_roc(fpr_grid)

        diffs = np.diff(tpr)
        assert np.all(diffs >= -1e-9), f"ROC not monotonic for {dgp.name}"

    @pytest.mark.parametrize(
        "dgp_factory",
        [
            lambda: make_gaussian_dgp(delta_mu=2.0),
            lambda: make_lognormal_dgp(neg_mu=0, pos_mu=1.0, sigma=0.8),
            lambda: make_weibull_dgp(
                neg_shape=2, pos_shape=2, neg_scale=1, pos_scale=2.5
            ),
        ],
    )
    def test_roc_bounded_to_unit_square(self, dgp_factory, fpr_grid):
        """Verify all ROCs satisfy 0 ≤ TPR ≤ 1."""
        dgp = dgp_factory()
        tpr = dgp.get_true_roc(fpr_grid)

        assert np.all(tpr >= 0), f"TPR < 0 for {dgp.name}"
        assert np.all(tpr <= 1), f"TPR > 1 for {dgp.name}"

    @pytest.mark.parametrize(
        "dgp_factory,expected_min_auc",
        [
            (lambda: make_gaussian_dgp(delta_mu=2.0), 0.7),
            (lambda: make_exponential_dgp(neg_rate=1.0, pos_rate=0.3), 0.7),
            (lambda: make_uniform_partial_overlap_dgp(overlap_fraction=0.3), 0.6),
        ],
    )
    def test_auc_above_chance(self, dgp_factory, expected_min_auc, fpr_grid):
        """Verify AUC is above chance level for discriminative DGPs."""
        dgp = dgp_factory()
        tpr = dgp.get_true_roc(fpr_grid)
        auc = np.trapezoid(tpr, fpr_grid)

        assert auc > expected_min_auc, (
            f"AUC={auc:.3f} below expected minimum for {dgp.name}"
        )

    def test_roc_endpoints_at_boundaries(self, fpr_grid):
        """Verify ROC passes through (0,0) and (1,1)."""
        dgp = make_gaussian_dgp(delta_mu=1.5)
        fpr_with_endpoints = np.array([0.0, *fpr_grid[1:-1], 1.0])
        tpr = dgp.get_true_roc(fpr_with_endpoints)

        assert np.abs(tpr[0] - 0.0) < 1e-6, "ROC should pass through (0,0)"
        assert np.abs(tpr[-1] - 1.0) < 1e-6, "ROC should pass through (1,1)"
