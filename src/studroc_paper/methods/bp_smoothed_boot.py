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
import torch
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import gammaln
from scipy.stats import norm
from studroc_paper.viz import plot_band_diagnostics
from torch.distributions import Beta

from .method_utils import compute_empirical_roc_from_scores, wilson_halfwidth_squared_np

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
            self.degree = max(10, int(self.n**0.4))  # MISE-optimal rate
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

    def _eval_bernstein_poly(self, u: NDArray, coeffs: NDArray) -> NDArray:
        """Evaluate Bernstein polynomial with given coefficients.

        Uses fully vectorized log-space evaluation for numerical stability.
        Computes: sum_{k=0}^{m} coeffs[k] * B_{k,m}(u)

        where B_{k,m}(u) = C(m,k) * u^k * (1-u)^(m-k)

        Args:
            u: Points at which to evaluate, shape (N,).
            coeffs: Bernstein coefficients, shape (m+1,).

        Returns:
            Evaluated polynomial values, shape (N,).
        """
        u = np.atleast_1d(np.asarray(u, dtype=np.float64))
        m = len(coeffs) - 1

        # Safety clip for log domain to avoid -inf
        u = np.clip(u, 1e-15, 1 - 1e-15)

        # Create range of k values: [0, 1, ..., m]
        k = np.arange(m + 1)

        # Log-binomial coefficients using gammaln for numerical stability:
        # log(C(m,k)) = log(m!) - log(k!) - log((m-k)!)
        # This prevents overflow for large m that would occur with direct computation.
        log_binom = gammaln(m + 1) - gammaln(k + 1) - gammaln(m - k + 1)

        # Compute log-basis for all k and all u using broadcasting.
        # k_col: shape (m+1, 1), u_row: shape (1, N) -> result: (m+1, N)
        k_col = k[:, np.newaxis]
        u_row = u[np.newaxis, :]

        # log(u^k * (1-u)^(m-k)) = k*log(u) + (m-k)*log(1-u)
        log_powers = k_col * np.log(u_row) + (m - k_col) * np.log(1 - u_row)

        # Combine: log_basis[k, :] = log_binom[k] + log_powers[k, :]
        log_basis = log_binom[:, np.newaxis] + log_powers

        # Compute weighted sum: sum_k coeffs[k] * exp(log_basis[k, :])
        # Use coeffs[:, np.newaxis] for broadcasting: (m+1, 1) * (m+1, N) -> (m+1, N)
        weighted_basis = coeffs[:, np.newaxis] * np.exp(log_basis)

        return np.sum(weighted_basis, axis=0)

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
            result[mask_mid] = self._eval_bernstein_poly(u[mask_mid], self.cdf_coeffs)

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
            result[mask] = self._eval_bernstein_poly(u[mask], self.pdf_coeffs)

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

        This uses a generative process. The Bernstein PDF is a mixture of Beta
        distributions:
            f(u) = Σ w_k * Beta(u | k+1, m-k)

        Complexity: O(N).
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

    def sample_batched(
        self,
        n_samples: int,
        n_bootstraps: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Generate batched samples using PyTorch (Vectorized/Parallel).

        Generates B bootstrap replicates of n samples each in a single pass.

        Args:
            n_samples: Number of samples per bootstrap (n).
            n_bootstraps: Number of bootstrap replicates (B).
            device: Torch device.
            dtype: Target dtype for the output samples.

        Returns:
            Tensor of shape (B, n) containing sampled values.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        m = self.degree

        # 1. Calculate mixture weights (CPU)
        weights = np.diff(self.cdf_coeffs)
        weights = weights / np.sum(weights)

        # Move to device
        weights_t = torch.tensor(weights, device=device, dtype=dtype)

        # 2. Select components
        # We need (B, n) indices
        total_samples = n_samples * n_bootstraps

        # Multinomial requires 1D or 2D input.
        # For 1D input of weights, it performs n_samples samples.
        indices = torch.multinomial(weights_t, total_samples, replacement=True).view(
            n_bootstraps, n_samples
        )

        # 3. Sample from Beta components
        alpha = (indices + 1).to(dtype=dtype)
        beta_param = (m - indices).to(dtype=dtype)

        dist = Beta(alpha, beta_param)
        samples_unit = dist.sample()

        # 4. Transform to original support
        support_min = torch.tensor(self.support_min, device=device, dtype=dtype)
        support_range = torch.tensor(self.support_range, device=device, dtype=dtype)

        return samples_unit * support_range + support_min


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
# Main SCB Construction
# =============================================================================


def bp_smoothed_bootstrap_band(
    y_true: NDArray | torch.Tensor,
    y_score: NDArray | torch.Tensor,
    fpr_grid: NDArray | torch.Tensor,
    alpha: float = 0.05,
    n_bootstrap: int = 2000,
    bp_degree: int | None = None,
    retention_method: RetentionMethod = "ks",
    random_seed: int | None = None,
    plot: bool = False,
    plot_title: str | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Construct SCB using BP-smoothed bootstrap with exact numerical methods.

    Key features:
    1. Center ROC computed exactly via numerical CDF/quantile
    2. ROC slope computed analytically from BP PDFs
    3. Bootstrap samples drawn from BP distributions
    4. Studentized retention with Wilson score floor

    Args:
        y_true: True binary labels (numpy array or torch tensor).
        y_score: Predicted scores (numpy array or torch tensor).
        fpr_grid: FPR evaluation points (numpy array or torch tensor).
        alpha: Significance level (default 0.05).
        n_bootstrap: Number of bootstrap replicates (default 2000).
        bp_degree: Bernstein polynomial degree. Default: max(10, sqrt(n)).
        retention_method: 'ks' for KS-based retention or 'symmetric' for
            two-sided trimming (default 'ks').
        random_seed: Random seed for reproducibility (default None).
        plot: If True, generate diagnostic plots using the viz module (default False).
        plot_title: Optional custom title for the diagnostic plots. If None, uses
            method description.

    Returns:
        Tuple of (fpr_grid, lower_band, upper_band).
    """
    rng = np.random.default_rng(random_seed)

    # Determine native dtype and device from inputs
    if isinstance(y_score, torch.Tensor):
        dtype = y_score.dtype
        device = y_score.device
    else:
        # Default for numpy inputs
        # Check if numpy array is float64, otherwise float32
        if np.asarray(y_score).dtype == np.float64:
            dtype = torch.float64
        else:
            dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split scores into pos/neg
    # Iterate to numpy for splitting/Bernstein fitting (requires CPU numpy)
    if isinstance(y_true, torch.Tensor):
        y_true_np = y_true.detach().cpu().numpy()
    else:
        y_true_np = np.asarray(y_true)

    if isinstance(y_score, torch.Tensor):
        y_score_np = y_score.detach().cpu().numpy()
    else:
        y_score_np = np.asarray(y_score)

    scores_neg_np = y_score_np[y_true_np == 0].astype(np.float64)
    scores_pos_np = y_score_np[y_true_np == 1].astype(np.float64)

    n_neg = len(scores_neg_np)
    n_pos = len(scores_pos_np)

    # === Step 1: FPR grid ===
    # Convert fpr_grid to numpy for exact ROC calc (BernsteinCDF is numpy/CPU)
    if isinstance(fpr_grid, torch.Tensor):
        fpr_grid_np = fpr_grid.detach().cpu().numpy()
    else:
        fpr_grid_np = np.asarray(fpr_grid)

    n_grid = len(fpr_grid_np)

    # === Step 2: Fit Bernstein polynomial CDFs ===
    bp_neg = BernsteinCDF(scores_neg_np, degree=bp_degree)
    bp_pos = BernsteinCDF(scores_pos_np, degree=bp_degree)

    # === Step 3: Exact BP-implied ROC (center curve) ===
    bp_roc = _compute_bp_roc_exact(
        bp_neg, bp_pos, fpr_grid_np, compute_derivatives=True
    )
    roc_center = bp_roc.tpr

    # === Step 4: Smoothed bootstrap ===
    roc_bootstrap = np.zeros((n_bootstrap, n_grid))

    # Use optimized batched sampling if possible
    try:
        # Move grid to device
        # Use provided fpr_grid if tensor, else make tensor
        if isinstance(fpr_grid, torch.Tensor):
            fpr_grid_t = fpr_grid.to(device=device, dtype=dtype)
        else:
            fpr_grid_t = torch.tensor(fpr_grid_np, device=device, dtype=dtype)

        # Pre-allocate output tensor
        roc_tensor = torch.zeros((n_bootstrap, n_grid), device=device, dtype=dtype)

        # Process in batches for memory efficiency
        BATCH_SIZE = 500

        for start_idx in range(0, n_bootstrap, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, n_bootstrap)
            current_batch_size = end_idx - start_idx

            # Generate batch of samples
            # Shape: (current_batch_size, n_neg)
            boot_neg_batch = bp_neg.sample_batched(
                n_neg, current_batch_size, device=device, dtype=dtype
            )
            # Shape: (current_batch_size, n_pos)
            boot_pos_batch = bp_pos.sample_batched(
                n_pos, current_batch_size, device=device, dtype=dtype
            )

            # Compute ROCs for this batch
            for i in range(current_batch_size):
                roc_tensor[start_idx + i] = compute_empirical_roc_from_scores(
                    boot_neg_batch[i], boot_pos_batch[i], fpr_grid_t
                )

            # Free batch memory
            del boot_neg_batch
            del boot_pos_batch

        # Move result to CPU numpy
        roc_bootstrap = roc_tensor.cpu().numpy()

    except (ImportError, RuntimeError) as e:
        # Fallback to original loop if torch fails or not available
        print(f"Sampling/computation failed on torch, falling back to numpy: {e}")
        for b in range(n_bootstrap):
            # Sample from BP-smoothed distributions
            boot_neg = bp_neg.sample(n_neg, rng=rng)
            boot_pos = bp_pos.sample(n_pos, rng=rng)

            # Compute empirical ROC from bootstrap samples
            # Note: _compute_empirical_roc expects 1d arrays, loop works on 1d arrays
            roc_bootstrap[b, :] = _compute_empirical_roc(
                boot_neg, boot_pos, fpr_grid_np
            )

    # === Step 5: Variance estimation ===
    sigma_bootstrap = np.std(roc_bootstrap, axis=0, ddof=1)

    # Wilson floor
    z_alpha = norm.ppf(1 - alpha / 2)
    sigma_wilson = np.sqrt(wilson_halfwidth_squared_np(roc_center, n_pos, z_alpha))

    # Combine: max of bootstrap, Wilson
    sigma_final = np.maximum(sigma_bootstrap, sigma_wilson)

    # === Step 6: Studentized retention ===
    epsilon = min(1 / (n_neg + n_pos), 1e-6)

    # Pre-compute valid locations to avoid divide-by-zero
    # Mask of points with meaningful variance
    valid_sigma = sigma_final >= epsilon

    # Denominator: use sigma_final if valid, else epsilon (to avoid div/0 temporarily)
    denom = np.where(valid_sigma, sigma_final, epsilon)

    # Calculate signed differences for all bootstraps at once
    # Shape: (n_bootstrap, n_grid)
    diff = roc_bootstrap - roc_center

    # Initial normalization
    z_scores_raw = diff / denom

    # We only need to zero out cases where not-valid-sigma AND diff is small
    low_sigma_mask = ~valid_sigma
    small_diff_mask = np.abs(diff) < epsilon
    zero_out_mask = low_sigma_mask & small_diff_mask

    z_scores = z_scores_raw.copy()
    z_scores[zero_out_mask] = 0.0

    if retention_method == "symmetric":
        # Symmetric tail trimming
        M_up = np.max(z_scores, axis=1)
        M_down = np.min(z_scores, axis=1)

        q_up = np.quantile(M_up, 1 - alpha / 2)
        q_down = np.quantile(M_down, alpha / 2)
        retained_mask = (M_down >= q_down) & (M_up <= q_up)

    elif retention_method == "ks":
        # KS-based retention
        # Max absolute studentized deviation
        Z = np.max(np.abs(z_scores), axis=1)

        threshold = np.quantile(Z, 1 - alpha)
        retained_mask = Z <= threshold

    else:
        raise ValueError(f"Invalid retention method: {retention_method}")

    # === Step 7: Envelope construction ===
    retained_curves = roc_bootstrap[retained_mask, :]
    n_retained = int(retained_mask.sum())

    if n_retained == 0:
        # Fallback: use all curves
        retained_curves = roc_bootstrap
        n_retained = n_bootstrap

    lower_band = np.min(retained_curves, axis=0)
    upper_band = np.max(retained_curves, axis=0)

    # === Step 7b: Apply Wilson floor to envelopes ===
    # Ensure minimum envelope width at boundaries where bootstrap collapses
    # Upper band should be at least center + Wilson half-width
    upper_band = np.maximum(upper_band, roc_center + sigma_wilson)
    # Lower band should be at most center - Wilson half-width
    lower_band = np.minimum(lower_band, roc_center - sigma_wilson)

    # === Step 8: Boundary constraints ===
    # Enforce monotonicity
    lower_band = np.maximum.accumulate(lower_band)
    upper_band = np.maximum.accumulate(upper_band)

    # Enforce bounds
    lower_band = np.clip(lower_band, 0, 1)
    upper_band = np.clip(upper_band, 0, 1)
    lower_band[0] = 0.0
    upper_band[-1] = 1.0

    # Generate diagnostic plots if requested
    if plot:
        try:
            # Compute empirical ROC for comparison
            empirical_tpr = _compute_empirical_roc(
                scores_neg_np, scores_pos_np, fpr_grid_np
            )

            # Determine method name for title
            if plot_title is None:
                plot_title = f"BP Smoothed Bootstrap (degree={bp_degree or 'auto'}, {retention_method} retention)"

            fig = plot_band_diagnostics(
                fpr_grid=fpr_grid_np,
                empirical_tpr=empirical_tpr,
                lower_envelope=lower_band,
                upper_envelope=upper_band,
                boot_tpr_matrix=roc_bootstrap,
                bootstrap_var=sigma_bootstrap**2,
                wilson_var=sigma_wilson**2,
                alpha=alpha,
                method_name=plot_title,
                additional_curves={"BP-smoothed": roc_center},
                layout="2x2",
            )
            fig.show()
        except ImportError:
            import warnings

            warnings.warn(
                "Visualization module not available. Install matplotlib to enable plotting.",
                stacklevel=2,
            )

    return (fpr_grid_np, lower_band, upper_band)
