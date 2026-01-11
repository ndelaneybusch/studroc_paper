"""Shared utilities for PyTorch-based ROC analysis methods.

This module provides core utilities for computing empirical ROC curves,
variance estimates, and density-based calculations. Key functionality includes:

- Tensor/array conversion utilities for PyTorch and NumPy interoperability
- Interpolation functions for ROC curve evaluation
- Empirical ROC computation from score distributions
- Wilson score interval calculations for binomial confidence
- Hsieh-Turnbull variance estimation using KDE and log-concave MLE
- Density and derivative estimation with reflection and adaptive bandwidth
"""

from typing import Literal

import cvxpy as cp
import numpy as np
import torch
from KDEpy import FFTKDE
from numpy.typing import NDArray
from scipy import interpolate
from scipy.stats import beta as beta_dist
from torch import Tensor


def numpy_to_torch(arr: NDArray | Tensor, device: torch.device | None = None) -> Tensor:
    """Convert numpy array or torch tensor to tensor on specified device.

    Handles both numpy arrays and existing tensors, moving them to the
    target device. If no device is specified, automatically selects CUDA
    if available, otherwise CPU.

    Args:
        arr: Input numpy array or torch tensor.
        device: Target device. Defaults to None (auto-select CUDA or CPU).

    Returns:
        PyTorch tensor on the specified device.

    Examples:
        >>> import numpy as np
        >>> arr = np.array([1.0, 2.0, 3.0])
        >>> tensor = numpy_to_torch(arr)
        >>> tensor.device.type in ["cuda", "cpu"]
        True
        >>> existing_tensor = torch.tensor([1.0, 2.0])
        >>> result = numpy_to_torch(existing_tensor, device=torch.device("cpu"))
        >>> result.device.type
        'cpu'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If already a tensor, just move to the target device
    if isinstance(arr, Tensor):
        return arr.to(device)

    # If numpy array, convert to tensor
    return torch.from_numpy(np.asarray(arr)).to(device)


def torch_to_numpy(tensor: Tensor | NDArray) -> NDArray:
    """Convert tensor or numpy array to numpy array.

    Handles conversion from PyTorch tensors to numpy arrays, automatically
    detaching from computational graph and moving to CPU if necessary.
    If input is already a numpy array, returns it unchanged.

    Args:
        tensor: Input PyTorch tensor or numpy array.

    Returns:
        Numpy array with preserved dtype.

    Examples:
        >>> tensor = torch.tensor([1.0, 2.0, 3.0])
        >>> arr = torch_to_numpy(tensor)
        >>> isinstance(arr, np.ndarray)
        True
        >>> existing_arr = np.array([1.0, 2.0])
        >>> result = torch_to_numpy(existing_arr)
        >>> result is existing_arr
        True
    """
    # If already numpy array, return as-is
    if isinstance(tensor, np.ndarray):
        return tensor

    # If tensor, convert to numpy (moving to CPU if necessary)
    return tensor.detach().cpu().numpy()


def torch_step_interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """Step-function interpolation (right-continuous).

    Implements a right-continuous step function where for each query point x
    between xp[j] and xp[j+1], the function returns fp[j]. This is the standard
    convention for empirical cumulative distribution functions.

    Args:
        x: Query points at which to evaluate the step function.
        xp: X-coordinates of step discontinuities (must be increasing).
        fp: Y-values corresponding to each step level.

    Returns:
        Interpolated step function values at x positions.

    Examples:
        >>> xp = torch.tensor([0.0, 0.5, 1.0])
        >>> fp = torch.tensor([0.0, 0.7, 1.0])
        >>> x = torch.tensor([0.25, 0.75])
        >>> torch_step_interp(x, xp, fp)
        tensor([0.0000, 0.7000])
    """
    # Find insertion indices
    indices = torch.searchsorted(xp, x, right=True) - 1
    indices = torch.clamp(indices, 0, len(fp) - 1)
    return fp[indices]


def torch_interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """Linear interpolation equivalent to np.interp.

    Performs piecewise linear interpolation between data points.
    Values outside the range of xp are clamped to boundary values.

    Args:
        x: X-coordinates at which to evaluate.
        xp: X-coordinates of data points (must be increasing).
        fp: Y-coordinates of data points.

    Returns:
        Interpolated values at x positions.

    Examples:
        >>> xp = torch.tensor([0.0, 1.0, 2.0])
        >>> fp = torch.tensor([0.0, 1.0, 0.5])
        >>> x = torch.tensor([0.5, 1.5])
        >>> torch_interp(x, xp, fp)
        tensor([0.5000, 0.7500])
    """
    # Find indices where x would be inserted
    indices = torch.searchsorted(xp, x)

    # Clamp indices for boundary handling
    indices = torch.clamp(indices, 1, len(xp) - 1)

    # Get left and right points
    x0 = xp[indices - 1]
    x1 = xp[indices]
    y0 = fp[indices - 1]
    y1 = fp[indices]

    # Linear interpolation
    t = (x - x0) / (x1 - x0 + 1e-12)
    return y0 + t * (y1 - y0)


def compute_empirical_roc_from_scores(
    neg_scores: Tensor, pos_scores: Tensor, fpr_grid: Tensor
) -> Tensor:
    """Compute empirical ROC curve and interpolate at fpr_grid points.

    Calculates the empirical receiver operating characteristic (ROC) curve
    by evaluating true positive rate (TPR) at all possible thresholds from
    the negative class scores, then interpolates to the requested FPR grid
    using step interpolation (right-continuous).

    Args:
        neg_scores: Tensor of negative class scores.
        pos_scores: Tensor of positive class scores.
        fpr_grid: FPR values at which to evaluate TPR.

    Returns:
        TPR values at fpr_grid points.

    Examples:
        >>> neg = torch.tensor([0.2, 0.4, 0.5, 0.6])
        >>> pos = torch.tensor([0.6, 0.7, 0.8, 0.9])
        >>> fpr = torch.tensor([0.0, 0.25, 0.5, 1.0])
        >>> tpr = compute_empirical_roc_from_scores(neg, pos, fpr)
        >>> tpr.shape
        torch.Size([4])
    """
    device = neg_scores.device

    # Use all negative scores as candidate thresholds (sorted descending)
    thresholds = torch.sort(neg_scores, descending=True).values

    n_neg = len(neg_scores)
    n_pos = len(pos_scores)

    # Vectorized computation using broadcasting
    # Shape: (n_thresholds,) computed via (1, n_neg) >= (n_thresholds, 1)
    # Memory is O(N^2), acceptable for typical sample sizes
    dtype = neg_scores.dtype

    # Calculate FPR: FP / N_neg = sum(neg_scores >= threshold) / N_neg
    # sum() returns long for bool inputs, so cast to original dtype
    fpr_emp = (neg_scores.unsqueeze(0) >= thresholds.unsqueeze(1)).sum(dim=1).to(
        dtype=dtype
    ) / n_neg

    # Calculate TPR: TP / N_pos = sum(pos_scores >= threshold) / N_pos
    tpr_emp = (pos_scores.unsqueeze(0) >= thresholds.unsqueeze(1)).sum(dim=1).to(
        dtype=dtype
    ) / n_pos

    # Add boundary points (0,0) and (1,1)
    fpr_emp = torch.cat(
        [
            torch.tensor([0.0], device=device, dtype=dtype),
            fpr_emp,
            torch.tensor([1.0], device=device, dtype=dtype),
        ]
    )
    tpr_emp = torch.cat(
        [
            torch.tensor([0.0], device=device, dtype=dtype),
            tpr_emp,
            torch.tensor([1.0], device=device, dtype=dtype),
        ]
    )

    # Interpolate at fpr_grid points using step interpolation
    return torch_step_interp(fpr_grid, fpr_emp, tpr_emp)


def harrell_davis_quantile(x_sorted: NDArray, p: float) -> float:
    """
    Harrell-Davis quantile estimator from pre-sorted array.

    Uses beta-weighted average of order statistics for reduced bias.

    Args:
        x_sorted: Pre-sorted (ascending) array of values.
        p: Quantile probability in [0, 1].

    Returns:
        Estimated quantile value.
    """
    n = len(x_sorted)
    if p <= 0:
        return float(x_sorted[0])
    if p >= 1:
        return float(x_sorted[-1])

    a = p * (n + 1)
    b = (1 - p) * (n + 1)

    bin_edges = np.arange(0, n + 1) / n
    weights = beta_dist.cdf(bin_edges[1:], a, b) - beta_dist.cdf(bin_edges[:-1], a, b)

    return float(np.dot(weights, x_sorted))


def compute_empirical_roc_hd(
    neg_scores: NDArray, pos_scores: NDArray, fpr_grid: NDArray
) -> NDArray:
    """Compute empirical ROC using Harrell-Davis threshold estimation (NumPy).

    Uses beta-weighted quantile estimation for determining thresholds, which
    reduces finite-sample bias compared to standard empirical quantiles.

    Args:
        neg_scores: Negative class scores.
        pos_scores: Positive class scores.
        fpr_grid: FPR values at which to evaluate TPR.

    Returns:
        TPR values at fpr_grid points.
    """
    neg_sorted = np.sort(neg_scores)
    tpr = np.zeros_like(fpr_grid)

    for i, target_fpr in enumerate(fpr_grid):
        if target_fpr <= 0:
            tpr[i] = 0.0
        elif target_fpr >= 1:
            tpr[i] = 1.0
        else:
            threshold = harrell_davis_quantile(neg_sorted, 1 - target_fpr)
            tpr[i] = np.mean(pos_scores > threshold)

    return tpr


def compute_empirical_roc_from_scores_hd(
    neg_scores: Tensor, pos_scores: Tensor, fpr_grid: Tensor
) -> Tensor:
    """Compute empirical ROC using Harrell-Davis threshold estimation (PyTorch).

    Uses beta-weighted quantile estimation for determining thresholds, which
    reduces finite-sample bias compared to standard empirical quantiles.

    The Harrell-Davis estimator computes quantiles as weighted averages of
    order statistics, with weights from a Beta distribution. For quantile p,
    the weight for order statistic i is:
        P((i-1)/n < U < i/n) where U ~ Beta(p*(n+1), (1-p)*(n+1))

    Args:
        neg_scores: Tensor of negative class scores.
        pos_scores: Tensor of positive class scores.
        fpr_grid: FPR values at which to evaluate TPR.

    Returns:
        TPR values at fpr_grid points.

    Examples:
        >>> neg = torch.tensor([0.2, 0.4, 0.5, 0.6])
        >>> pos = torch.tensor([0.6, 0.7, 0.8, 0.9])
        >>> fpr = torch.tensor([0.0, 0.25, 0.5, 1.0])
        >>> tpr = compute_empirical_roc_from_scores_hd(neg, pos, fpr)
        >>> tpr.shape
        torch.Size([4])
    """
    device = neg_scores.device
    dtype = neg_scores.dtype

    # Convert to numpy for HD quantile computation (scipy-based)
    neg_np = neg_scores.cpu().numpy()
    pos_np = pos_scores.cpu().numpy()
    fpr_np = fpr_grid.cpu().numpy()

    # Compute HD-based ROC in NumPy
    tpr_np = compute_empirical_roc_hd(neg_np, pos_np, fpr_np)

    # Convert back to torch
    return torch.from_numpy(tpr_np).to(device=device, dtype=dtype)


# =============================================================================
# Wilson Score Interval Utilities
# =============================================================================


def wilson_halfwidth_squared_np(p: NDArray, n: int, z: float) -> NDArray:
    """Compute squared half-width of Wilson score interval (NumPy version).

    The Wilson score interval for a binomial proportion p is:
        p̃ ± (z / denom) * sqrt(p(1-p)/n + z²/(4n²))

    where denom = 1 + z²/n and p̃ is the Wilson-adjusted center.

    This function returns the squared half-width:
        (z / denom)² * (p(1-p)/n + z²/(4n²))

    This is used as a variance floor because:
    - It's always positive, even when p=0 or p=1 (where Wald variance is 0)
    - At boundaries: returns z⁴/(4n²(1 + z²/n)²) > 0
    - Provides a principled minimum uncertainty from exact binomial CI theory

    Args:
        p: Proportion estimates (e.g., TPR values).
        n: Sample size (e.g., number of positive samples).
        z: Critical value from normal distribution (e.g., z = Φ⁻¹(1 - α/2)).

    Returns:
        Squared half-width of Wilson interval at each p value (variance floor).

    Examples:
        >>> p = np.array([0.0, 0.5, 1.0])
        >>> n = 100
        >>> z = 1.96  # 95% confidence
        >>> var_floor = wilson_halfwidth_squared_np(p, n, z)
        >>> np.all(var_floor > 0)
        True
    """
    if n <= 0:
        return np.zeros_like(p)

    denom = 1 + z**2 / n
    return (z**2 / denom**2) * (p * (1 - p) / n + z**2 / (4 * n**2))


def wilson_halfwidth_squared_torch(p: Tensor, n: int, z: float) -> Tensor:
    """Compute squared half-width of Wilson score interval (PyTorch version).

    The Wilson score interval for a binomial proportion p is:
        p̃ ± (z / denom) * sqrt(p(1-p)/n + z²/(4n²))

    where denom = 1 + z²/n and p̃ is the Wilson-adjusted center.

    This function returns the squared half-width:
        (z / denom)² * (p(1-p)/n + z²/(4n²))

    This is used as a variance floor because:
    - It's always positive, even when p=0 or p=1 (where Wald variance is 0)
    - At boundaries: returns z⁴/(4n²(1 + z²/n)²) > 0
    - Provides a principled minimum uncertainty from exact binomial CI theory

    Args:
        p: Proportion estimates (e.g., TPR values) as tensor.
        n: Sample size (e.g., number of positive samples).
        z: Critical value from normal distribution (e.g., z = Φ⁻¹(1 - α/2)).

    Returns:
        Squared half-width of Wilson interval at each p value (variance floor).

    Examples:
        >>> p = torch.tensor([0.0, 0.5, 1.0])
        >>> n = 100
        >>> z = 1.96  # 95% confidence
        >>> var_floor = wilson_halfwidth_squared_torch(p, n, z)
        >>> torch.all(var_floor > 0)
        tensor(True)
    """
    if n <= 0:
        return torch.zeros_like(p)

    z_sq = z * z
    denom = 1 + z_sq / n
    return (z_sq / (denom * denom)) * (p * (1 - p) / n + z_sq / (4 * n * n))


# =============================================================================
# Hsieh-Turnbull Variance Utilities
# =============================================================================


def _sheather_jones_bandwidth(data: NDArray) -> float:
    """Compute Improved Sheather-Jones bandwidth using KDEpy.

    Uses the Improved Sheather-Jones (ISJ) method for optimal bandwidth
    selection in kernel density estimation. This method minimizes
    asymptotic mean integrated squared error.

    Args:
        data: Input data array for bandwidth estimation.

    Returns:
        Optimal bandwidth value.

    Examples:
        >>> data = np.random.randn(100)
        >>> bw = _sheather_jones_bandwidth(data)
        >>> bw > 0
        True
    """
    # KDEpy handles ISJ bandwidth selection efficiently
    # We fit on the data to extract the computed bandwidth.
    kde = FFTKDE(kernel="gaussian", bw="ISJ").fit(data)
    return float(kde.bw)


def _kde_density_derivative(
    data: NDArray,
    eval_points: NDArray,
    bw_method: Literal["silverman", "ISJ"] = "ISJ",
    reflected: bool = False,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> tuple[NDArray, NDArray]:
    """Compute density and derivative using KDE.

    Uses KDEpy for optimal bandwidth selection, then computes exact
    Gaussian mixture density and derivatives manually. Supports boundary
    reflection to reduce edge bias near data boundaries.

    Args:
        data: Input data array.
        eval_points: Points at which to evaluate density and derivative.
        bw_method: Bandwidth selection method ('ISJ' or 'silverman'). Defaults to 'ISJ'.
        reflected: If True, use boundary reflection to reduce edge bias. Defaults to False.
        lower_bound: Lower boundary for reflection. Defaults to None (uses np.min(data)).
        upper_bound: Upper boundary for reflection. Defaults to None (uses np.max(data)).

    Returns:
        Tuple of (density, derivative) arrays at eval_points.

    Examples:
        >>> data = np.random.randn(100)
        >>> eval_pts = np.linspace(-3, 3, 50)
        >>> density, deriv = _kde_density_derivative(data, eval_pts)
        >>> density.shape == eval_pts.shape
        True
        >>> deriv.shape == eval_pts.shape
        True
    """
    # Determine bandwidth using Improved Sheather-Jones (ISJ)
    if bw_method == "ISJ":
        try:
            h = _sheather_jones_bandwidth(data)
        except Exception:
            bw_method = "silverman"

    if bw_method == "silverman":
        try:
            h = float(FFTKDE(kernel="gaussian", bw="silverman").fit(data).bw)
        except Exception:
            h = 1.06 * np.std(data) * (len(data) ** (-0.2))

    if h <= 1e-9:
        h = 1e-6

    if reflected:
        # Augment data using boundary reflection
        L = lower_bound if lower_bound is not None else np.min(data)
        U = upper_bound if upper_bound is not None else np.max(data)
        # Reflect around the boundaries: D_aug = [D, 2*L - D, 2*U - D]
        data_aug = np.concatenate([data, 2 * L - data, 2 * U - data])
        n_aug = len(data_aug)
    else:
        n_aug = len(data)
        data_aug = data

    # Vectorized Gaussian sum computation
    eval_points = np.asarray(eval_points)
    pdf = np.zeros_like(eval_points)
    deriv = np.zeros_like(eval_points)

    chunk_size = 100
    const_norm = 1.0 / (np.sqrt(2 * np.pi))
    factor_pdf = 1.0 / (n_aug * h)
    factor_deriv = 1.0 / (n_aug * h**2)

    # Normalize result by 3.0 because we want density on original support [L, U]
    # and we reflected twice (tripling mass).
    correction = 3.0 if reflected else 1.0

    for i in range(0, len(eval_points), chunk_size):
        x_chunk = eval_points[i : i + chunk_size]

        # Compute standardized distances: u = (x - xi) / h
        u = (x_chunk[:, None] - data_aug[None, :]) / h

        # Gaussian kernel: K(u) = (1/sqrt(2π)) * exp(-u²/2)
        k_u = const_norm * np.exp(-0.5 * u**2)

        # Density: sum of kernels
        sum_k = np.sum(k_u, axis=1)
        pdf[i : i + chunk_size] = sum_k * factor_pdf * correction

        # Derivative: sum of kernel derivatives
        sum_k_prime = np.sum(-u * k_u, axis=1)
        deriv[i : i + chunk_size] = sum_k_prime * factor_deriv * correction

    return pdf, deriv


def _log_concave_mle_density_derivative(
    data: NDArray, eval_points: NDArray
) -> tuple[NDArray, NDArray]:
    """Estimate density and derivative using Log-Concave MLE.

    Solves the binned log-concave maximum likelihood estimation problem
    using convex optimization (cvxpy). The density is constrained to have
    a concave logarithm, which is a common shape constraint in statistics.

    Args:
        data: Input data array for density estimation.
        eval_points: Points at which to evaluate the estimated density and derivative.

    Returns:
        Tuple of (density, derivative) arrays at eval_points.

    Examples:
        >>> data = np.random.randn(100)
        >>> eval_pts = np.linspace(-3, 3, 50)
        >>> density, deriv = _log_concave_mle_density_derivative(data, eval_pts)
        >>> density.shape == eval_pts.shape
        True
        >>> np.all(density >= 0)
        True
    """
    # Bin data to a fine grid
    n_bins = 200
    # Include some buffer for tails
    data_min, data_max = np.min(data), np.max(data)
    pad = 0.1 * (data_max - data_min)
    grid_min, grid_max = data_min - pad, data_max + pad

    # Uniform binning is robust enough for high n_bins.
    grid_edges = np.linspace(grid_min, grid_max, n_bins + 1)
    grid_centers = 0.5 * (grid_edges[:-1] + grid_edges[1:])
    widths = np.diff(grid_edges)

    counts, _ = np.histogram(data, bins=grid_edges)

    # Setup convex optimization problem
    # Variable: phi = log(f) at grid_centers
    phi = cp.Variable(n_bins)

    # Objective: Maximize log-likelihood ~ sum(counts * phi) - n * integral(exp(phi))
    # Integral approximation: sum(width * exp(phi))
    # This corresponds to Poisson regression with concavity constraint
    likelihood_term = counts @ phi
    integral_term = cp.sum(cp.multiply(widths, cp.exp(phi)))

    objective = cp.Maximize(likelihood_term - integral_term)

    # Constraints: Concavity of log-density
    # Discrete 2nd derivative <= 0
    # For uniform grid: phi[i-1] - 2*phi[i] + phi[i+1] <= 0
    constraints = [cp.diff(phi, k=2) <= 0]

    # Solve the optimization problem
    prob = cp.Problem(objective, constraints)

    try:
        # SCS is a robust first-order solver
        prob.solve(solver=cp.SCS, eps=1e-4)
    except cp.SolverError:
        try:
            prob.solve(solver=cp.ECOS)
        except Exception:
            pass

    if phi.value is None or np.all(np.isnan(phi.value)):
        # Solver failed, fallback to normal approximation
        mu, std = np.mean(data), np.std(data)
        phi_res = -0.5 * ((grid_centers - mu) / std) ** 2 - np.log(
            std * np.sqrt(2 * np.pi)
        )
    else:
        phi_res = phi.value

    # Interpolate result using cubic spline on log-density
    try:
        tck = interpolate.splrep(grid_centers, phi_res, k=3, s=0)
    except Exception:
        # Fallback if spline fails
        tck = interpolate.splrep(grid_centers, np.nan_to_num(phi_res), k=3, s=0)

    phi_eval = interpolate.splev(eval_points, tck)
    phi_prime_eval = interpolate.splev(eval_points, tck, der=1)

    # Compute density and derivative
    # f = exp(phi), f' = f * phi'
    # Clip phi to prevent overflow (exp(700) is near float64 max)
    phi_eval = np.clip(phi_eval, -100, 100)
    f_eval = np.exp(phi_eval)
    f_prime_eval = f_eval * phi_prime_eval

    # Normalize density (integral = 1)
    # The optimization forces sum(widths*exp(phi)) ~= N_total
    # So we divide by N to normalize
    n_total = np.sum(counts)
    if n_total > 0:
        f_eval /= n_total
        f_prime_eval /= n_total

    return f_eval, f_prime_eval


def compute_hsieh_turnbull_variance(
    neg_scores: NDArray,
    pos_scores: NDArray,
    fpr_grid: NDArray,
    method: str = "reflected_kde",
    data_floor: float | None = None,
    data_ceil: float | None = None,
) -> NDArray:
    """Compute asymptotic variance of ROC curve using Hsieh-Turnbull formula.

    Uses the Hsieh-Turnbull asymptotic variance formula for empirical ROC curves:
        Var(R(t)) = R(t)(1-R(t))/n1 + (g(c)/f(c))^2 * t(1-t)/n0

    where R(t) is the TPR at FPR=t, f is the negative class density,
    g is the positive class density, and c is the threshold at FPR=t.

    Args:
        neg_scores: Negative class (control) scores.
        pos_scores: Positive class (case) scores.
        fpr_grid: FPR values at which to evaluate variance.
        method: Density estimation method. Defaults to 'reflected_kde'.
            Options: 'reflected_kde', 'log_concave', 'kde'.
        data_floor: Optional lower boundary for reflected KDE. Defaults to None (uses data minimum).
        data_ceil: Optional upper boundary for reflected KDE. Defaults to None (uses data maximum).

    Returns:
        Variance array matching fpr_grid shape.

    Examples:
        >>> neg = np.random.randn(100)
        >>> pos = np.random.randn(100) + 1.0
        >>> fpr = np.linspace(0.01, 0.99, 50)
        >>> var = compute_hsieh_turnbull_variance(neg, pos, fpr)
        >>> var.shape == fpr.shape
        True
        >>> np.all(var > 0)
        True
    """
    n0 = len(neg_scores)
    n1 = len(pos_scores)

    # Compute thresholds and R(t)
    # Threshold c corresponds to F_neg(c) = 1 - t
    # For t=0 (c=inf), t=1 (c=-inf)
    thresholds = np.zeros_like(fpr_grid)
    valid_mask = (fpr_grid > 0) & (fpr_grid < 1)

    if np.any(valid_mask):
        thresholds[valid_mask] = np.quantile(neg_scores, 1 - fpr_grid[valid_mask])

    # Set boundary thresholds with buffer for density evaluation
    min_score = min(np.min(neg_scores), np.min(pos_scores))
    max_score = max(np.max(neg_scores), np.max(pos_scores))

    thresholds[fpr_grid <= 0] = max_score + 0.1
    thresholds[fpr_grid >= 1] = min_score - 0.1

    # Compute R(t) = P(Y > c) using empirical probabilities
    tpr_empirical = np.mean(pos_scores[:, None] > thresholds[None, :], axis=0)

    # Estimate densities and ROC slopes
    if method == "log_concave":
        f_vals, _ = _log_concave_mle_density_derivative(neg_scores, thresholds)
        g_vals, _ = _log_concave_mle_density_derivative(pos_scores, thresholds)
    elif method == "kde":
        f_vals, _ = _kde_density_derivative(neg_scores, thresholds, reflected=False)
        g_vals, _ = _kde_density_derivative(pos_scores, thresholds, reflected=False)
    elif method == "reflected_kde":
        f_vals, _ = _kde_density_derivative(
            neg_scores,
            thresholds,
            reflected=True,
            lower_bound=data_floor,
            upper_bound=data_ceil,
        )
        g_vals, _ = _kde_density_derivative(
            pos_scores,
            thresholds,
            reflected=True,
            lower_bound=data_floor,
            upper_bound=data_ceil,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Protect against division by zero and extreme ratios
    f_vals = np.maximum(f_vals, 1e-12)
    g_vals = np.maximum(g_vals, 1e-12)
    roc_slope = g_vals / f_vals

    # Clip ROC slope to prevent overflow
    # Extreme slopes (>1000) are unrealistic and cause numerical issues
    roc_slope = np.clip(roc_slope, 0.0, 1000.0)

    # Assemble variance components
    # Var = Term1 + Term2
    term1 = tpr_empirical * (1 - tpr_empirical) / n1
    term2 = (roc_slope**2) * fpr_grid * (1 - fpr_grid) / n0

    variance = term1 + term2

    # Replace any remaining inf/nan with binomial variance fallback
    invalid_mask = ~np.isfinite(variance)
    if np.any(invalid_mask):
        variance[invalid_mask] = (
            tpr_empirical[invalid_mask] * (1 - tpr_empirical[invalid_mask]) / n1
        )

    return variance
