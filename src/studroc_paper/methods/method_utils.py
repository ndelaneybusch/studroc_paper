"""Shared utilities for PyTorch-based methods."""

from typing import Literal

import cvxpy as cp
import numpy as np
import torch
from KDEpy import FFTKDE
from numpy.typing import NDArray
from scipy import interpolate
from torch import Tensor


def numpy_to_torch(arr: NDArray | Tensor, device: torch.device | None = None) -> Tensor:
    """Convert numpy array or torch tensor to tensor on specified device.

    Args:
        arr: Input numpy array or torch tensor.
        device: Target device (defaults to CUDA if available).

    Returns:
        PyTorch tensor on the specified device.
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

    Args:
        tensor: Input PyTorch tensor or numpy array.

    Returns:
        Numpy array with preserved dtype.
    """
    # If already numpy array, return as-is
    if isinstance(tensor, np.ndarray):
        return tensor

    # If tensor, convert to numpy (moving to CPU if necessary)
    return tensor.detach().cpu().numpy()


def torch_step_interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """
    Step-function interpolation (right-continuous).

    For x between xp[j] and xp[j+1], returns fp[j].
    """
    # Find insertion indices
    indices = torch.searchsorted(xp, x, right=True) - 1
    indices = torch.clamp(indices, 0, len(fp) - 1)
    return fp[indices]


def torch_interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """Linear interpolation equivalent to np.interp.

    Args:
        x: X-coordinates at which to evaluate.
        xp: X-coordinates of data points (must be increasing).
        fp: Y-coordinates of data points.

    Returns:
        Interpolated values at x positions.
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

    Vectorized computation using PyTorch.

    Args:
        neg_scores: Tensor of negative class scores.
        pos_scores: Tensor of positive class scores.
        fpr_grid: FPR values at which to evaluate TPR.

    Returns:
        TPR values at fpr_grid points.
    """
    device = neg_scores.device

    # Get thresholds from negative scores (sorted descending)
    # Note: We use all negative scores as candidate thresholds
    thresholds = torch.sort(neg_scores, descending=True).values

    n_neg = len(neg_scores)
    n_pos = len(pos_scores)

    # Vectorized computation: for each threshold, compute FPR and TPR
    # Shape: (n_thresholds,)
    # Uses broadcasting: (1, n_neg) >= (n_thresholds, 1) -> (n_thresholds, n_neg)
    # This is O(N^2) memory, which is fine for typical N (~1000s)
    # but be careful for very large N.

    # Expanding dimensions for broadcasting
    # neg_scores: (N_neg,) -> (1, N_neg)
    # thresholds: (N_neg,) -> (N_neg, 1)

    # Determine dtype from inputs
    dtype = neg_scores.dtype

    # Calculate FPR for each threshold
    # FPR = FP / N_neg = sum(neg_scores >= threshold) / N_neg
    # Note: sum() returns long for int/bool inputs, so we must cast to original dtype
    fpr_emp = (neg_scores.unsqueeze(0) >= thresholds.unsqueeze(1)).sum(dim=1).to(
        dtype=dtype
    ) / n_neg

    # Calculate TPR for each threshold
    # TPR = TP / N_pos = sum(pos_scores >= threshold) / N_pos
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

    # Interpolate at fpr_grid points (Step interpolation for empirical ROC)
    return torch_step_interp(fpr_grid, fpr_emp, tpr_emp)


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
    """
    Compute Improved Sheather-Jones bandwidth using KDEpy.
    """
    # KDEpy handles ISJ bandwidth selection efficiently
    # We fit on the data to extract the computed bandwidth.
    kde = FFTKDE(kernel="gaussian", bw="ISJ").fit(data)
    return float(kde.bw)


def _kde_density_derivative(
    data: NDArray,
    eval_points: NDArray,
    bw_method: Literal["silverman", "ISJ"] = "ISJ",
    reflected=False,
) -> tuple[NDArray, NDArray]:
    """
    Compute density and derivative using KDE.

    Uses KDEpy for optimal bandwidth selection, then computes exact
    Gaussian mixture density and derivatives manually (handling reflection).
    """
    # 1. Determine bandwidth using Improved Sheather-Jones (ISJ)
    try:
        h = _sheather_jones_bandwidth(data)
    except Exception:
        # Fallback if ISJ fails (e.g., too few unique points)
        try:
            h = float(FFTKDE(kernel="gaussian", bw="silverman").fit(data).bw)
        except Exception:
            h = 1.06 * np.std(data) * (len(data) ** (-0.2))

    if h <= 1e-9:
        h = 1e-6

    if reflected:
        # 2. Augment data (Reflection)
        lower, upper = np.min(data), np.max(data)
        # D_aug = [D, 2*L - D, 2*U - D]
        data_aug = np.concatenate([data, 2 * lower - data, 2 * upper - data])
        n_aug = len(data_aug)  # = 3 * n
    else:
        n_aug = len(data)
        data_aug = data

    # 3. Vectorized Gaussian Sum computation
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

        # u = (x - xi) / h
        u = (x_chunk[:, None] - data_aug[None, :]) / h

        # K(u)
        k_u = const_norm * np.exp(-0.5 * u**2)

        # sum K(u)
        sum_k = np.sum(k_u, axis=1)
        pdf[i : i + chunk_size] = sum_k * factor_pdf * correction

        # sum -u * K(u)
        sum_k_prime = np.sum(-u * k_u, axis=1)
        deriv[i : i + chunk_size] = sum_k_prime * factor_deriv * correction

    return pdf, deriv


def _log_concave_mle_density_derivative(
    data: NDArray, eval_points: NDArray
) -> tuple[NDArray, NDArray]:
    """
    Estimate density and derivative using Log-Concave MLE via Convex Optimization.

    Solves the binned Log-Concave Maximum Likelihood problem using cvxpy.
    """
    # 1. Bin data to a fine grid
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

    # 2. Setup Convex Problem
    # Variable: phi = log(f) at grid_centers
    phi = cp.Variable(n_bins)

    # Objective: Maximize Log-Likelihood ~ sum(counts * phi) - n * integral(exp(phi))
    # Approximation: integral approx sum(width * exp(phi))
    # Obj = sum(counts * phi) - sum(widths * exp(phi))
    # This corresponds to a Poisson regression with convexity constraint

    likelihood_term = counts @ phi
    integral_term = cp.sum(cp.multiply(widths, cp.exp(phi)))

    objective = cp.Maximize(likelihood_term - integral_term)

    # Constraints: Concavity
    # Discrete 2nd derivative <= 0
    # For uniform grid: phi[i-1] - 2*phi[i] + phi[i+1] <= 0
    # Vectorized: cp.diff(phi, k=2) <= 0
    constraints = [cp.diff(phi, k=2) <= 0]

    # 3. Solve
    prob = cp.Problem(objective, constraints)

    try:
        # SCS is a robust first-order solver, often default
        prob.solve(solver=cp.SCS, eps=1e-4)
    except cp.SolverError:
        try:
            prob.solve(solver=cp.ECOS)
        except Exception:
            pass

    if phi.value is None or np.all(np.isnan(phi.value)):
        # Solver failed -> Fallback to normal approximation
        mu, std = np.mean(data), np.std(data)
        phi_res = -0.5 * ((grid_centers - mu) / std) ** 2 - np.log(
            std * np.sqrt(2 * np.pi)
        )
    else:
        phi_res = phi.value

    # 4. Interpolate result
    # Fit cubic spline to the log-density phi
    try:
        tck = interpolate.splrep(grid_centers, phi_res, k=3, s=0)
    except Exception:
        # Fallback if spline fails
        tck = interpolate.splrep(grid_centers, np.nan_to_num(phi_res), k=3, s=0)

    phi_eval = interpolate.splev(eval_points, tck)
    phi_prime_eval = interpolate.splev(eval_points, tck, der=1)

    # f = exp(phi)
    # f' = f * phi'
    f_eval = np.exp(phi_eval)
    f_prime_eval = f_eval * phi_prime_eval

    # Normalize density (integral = 1)
    # The term - integral(exp(phi)) forces sum(widths*exp(phi)) ~= N_total
    # So f integrates to N. We must divide by N.
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
) -> NDArray:
    """
    Compute Asymptotic Variance of ROC curve using Hsieh-Turnbull formula.

    Var(R(t)) = R(t)(1-R(t))/n1 + (g(c)/f(c))^2 * t(1-t)/n0

    Args:
        neg_scores: Control scores
        pos_scores: Case scores
        fpr_grid: FPR values t over which to evaluate.
        method: 'reflected_kde' or 'log_concave' or 'kde'.

    Returns:
        Variance array matching fpr_grid.
    """
    n0 = len(neg_scores)
    n1 = len(pos_scores)

    # 1. Compute Thresholds and R(t)
    # Note: For the 'center' R(t) term, we use empirical probabilities
    # as they are unbiased and standard.
    # Threshold c corresponds to F_neg(c) = 1 - t
    # For t=0 (c=inf), t=1 (c=-inf).

    # Using numpy quantile for consistency
    # Handle boundaries carefully
    thresholds = np.zeros_like(fpr_grid)
    valid_mask = (fpr_grid > 0) & (fpr_grid < 1)

    if np.any(valid_mask):
        thresholds[valid_mask] = np.quantile(neg_scores, 1 - fpr_grid[valid_mask])

    # Set boundary thresholds to min/max with buffer for density eval
    min_score = min(np.min(neg_scores), np.min(pos_scores))
    max_score = max(np.max(neg_scores), np.max(pos_scores))

    thresholds[fpr_grid <= 0] = max_score + 0.1
    thresholds[fpr_grid >= 1] = min_score - 0.1

    # Compute R(t) = P(Y > c)
    tpr_empirical = np.mean(pos_scores[:, None] > thresholds[None, :], axis=0)

    # 2. Estimate Densities and Slopes
    if method == "log_concave":
        f_vals, _ = _log_concave_mle_density_derivative(neg_scores, thresholds)
        g_vals, _ = _log_concave_mle_density_derivative(pos_scores, thresholds)
    elif method == "kde":
        f_vals, _ = _kde_density_derivative(neg_scores, thresholds, reflected=False)
        g_vals, _ = _kde_density_derivative(pos_scores, thresholds, reflected=False)
    elif method == "reflected_kde":
        f_vals, _ = _kde_density_derivative(neg_scores, thresholds, reflected=True)
        g_vals, _ = _kde_density_derivative(pos_scores, thresholds, reflected=True)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Protect against division by zero
    f_vals = np.maximum(f_vals, 1e-12)
    roc_slope = g_vals / f_vals

    # 3. Assemble Variance
    # Var = Term1 + Term2

    term1 = tpr_empirical * (1 - tpr_empirical) / n1
    term2 = (roc_slope**2) * fpr_grid * (1 - fpr_grid) / n0

    variance = term1 + term2

    return variance
