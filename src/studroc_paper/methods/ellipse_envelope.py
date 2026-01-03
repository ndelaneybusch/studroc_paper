"""Ellipse-Envelope Simultaneous Confidence Bands for ROC Curves.

Implementation based on Demidenko (2012): "Confidence intervals and bands
for the binormal ROC curve revisited", Journal of Applied Statistics, 39(1), 67-79.

The ellipse-envelope method provides improved coverage probability over the
classic Working-Hotelling approach by accounting for the estimation of variances
rather than treating them as fixed and known.
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, norm

try:
    from torch import Tensor

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Tensor = None  # type: ignore[misc, assignment]

from studroc_paper.viz import plot_band_diagnostics

from .method_utils import compute_empirical_roc_from_scores


def _solve_quartic_numpy(coefficients: NDArray) -> NDArray:
    """Solve quartic polynomial using numpy's companion matrix method.

    Args:
        coefficients: Array of shape (5,) with [p4, p3, p2, p1, p0].

    Returns:
        Array of real roots (imaginary parts < 1e-10 are filtered out).

    Examples:
        >>> coeffs = np.array([1, 0, -5, 0, 4])  # x^4 - 5x^2 + 4
        >>> roots = _solve_quartic_numpy(coeffs)
    """
    roots = np.roots(coefficients)
    # Extract real roots (with small imaginary tolerance)
    real_mask = np.abs(roots.imag) < 1e-10
    return roots[real_mask].real


def _convert_to_numpy(arr: "NDArray | Tensor") -> tuple[NDArray, "np.dtype"]:
    """Convert input array to numpy, handling torch tensors.

    Args:
        arr: Input array (numpy or torch tensor).

    Returns:
        Tuple of (numpy array, original dtype).

    Examples:
        >>> arr_np = np.array([1, 2, 3], dtype=np.float32)
        >>> result, dtype = _convert_to_numpy(arr_np)
        >>> dtype
        dtype('float32')
    """
    if TORCH_AVAILABLE and isinstance(arr, Tensor):
        numpy_arr = arr.detach().cpu().numpy()
        return numpy_arr, numpy_arr.dtype
    numpy_arr = np.asarray(arr)
    return numpy_arr, numpy_arr.dtype


def _estimate_binormal_parameters(
    y_true: NDArray, y_score: NDArray, minimum_std: float = 1e-8
) -> tuple[float, float, float, float, int, int]:
    """Estimate binormal model parameters from data.

    Computes sample means and standard deviations for each class, applying
    a minimum standard deviation floor to prevent numerical degeneracy.

    Args:
        y_true: Binary class labels (0/1).
        y_score: Continuous unbounded prediction scores.
        minimum_std: Minimum allowed standard deviation to prevent degeneracy. Defaults to 1e-8.

    Returns:
        Tuple of (mu0, std0, mu1, std1, n0, n1) where mu and std are the mean
        and standard deviation for each class, and n is the sample count.

    Examples:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_score = np.array([0.2, 0.4, 0.7, 0.9])
        >>> mu0, std0, mu1, std1, n0, n1 = _estimate_binormal_parameters(
        ...     y_true, y_score
        ... )
    """
    negative_scores = y_score[y_true == 0]
    positive_scores = y_score[y_true == 1]

    num_negative = len(negative_scores)
    num_positive = len(positive_scores)

    mean_negative = negative_scores.mean()
    mean_positive = positive_scores.mean()

    # Use unbiased sample standard deviation (ddof=1)
    std_negative = negative_scores.std(ddof=1)
    std_positive = positive_scores.std(ddof=1)

    # Handle degenerate cases
    if np.isnan(std_negative) or std_negative < minimum_std:
        std_negative = minimum_std
    if np.isnan(std_positive) or std_positive < minimum_std:
        std_positive = minimum_std

    return (
        mean_negative,
        std_negative,
        mean_positive,
        std_positive,
        num_negative,
        num_positive,
    )


def _compute_ellipse_coefficients(
    cutoff: float,
    mean_negative: float,
    std_negative: float,
    mean_positive: float,
    std_positive: float,
    num_negative: int,
    num_positive: int,
    chi2_critical: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Compute the ellipse equation coefficients for a given cutoff.

    Following Demidenko (2012) notation, computes standardized statistics and
    their variance-adjusted coefficients for the confidence ellipse.

    Notation:
        - γ̂_k(c) = (x̄_k - c) / s_k (standardized location)
        - A_k = (1/n_k + γ̂_k²(c) / (2(n_k-1)))^(-1) (inverse variance weight)
        - B_k = 2*A_k / σ_k (gradient coefficient)
        - D_k = γ̂_k(c) * A_k² / (σ_k * (n_k-1)) (curvature coefficient)

    Args:
        cutoff: Decision threshold c.
        mean_negative: Sample mean of negative class.
        std_negative: Sample std of negative class.
        mean_positive: Sample mean of positive class.
        std_positive: Sample std of positive class.
        num_negative: Number of negative samples.
        num_positive: Number of positive samples.
        chi2_critical: Chi-squared critical value (q_{λ,2}).

    Returns:
        Tuple of (gamma0, gamma1, A0, A1, B0, B1, D0, D1).
    """
    # γ̂_k(c) = (x̄_k - c) / s_k
    # Note: For ROC curve, we use the convention that higher scores indicate positive class
    gamma_negative = (mean_negative - cutoff) / std_negative
    gamma_positive = (mean_positive - cutoff) / std_positive

    # A_k = (1/n_k + γ̂_k²(c) / (2(n_k-1)))^(-1)
    variance_gamma_negative = 1.0 / num_negative + gamma_negative**2 / (
        2 * (num_negative - 1)
    )
    variance_gamma_positive = 1.0 / num_positive + gamma_positive**2 / (
        2 * (num_positive - 1)
    )

    a_negative = 1.0 / variance_gamma_negative
    a_positive = 1.0 / variance_gamma_positive

    # The derivative involves ∂γ/∂c = -1/σ, so Bk = Ak * (1/σk)
    b_negative = 2 * a_negative / std_negative
    b_positive = 2 * a_positive / std_positive

    # D_k = γ̂_k(c) * A_k² / (σ_k * (n_k-1))
    d_negative = gamma_negative * a_negative**2 / (std_negative * (num_negative - 1))
    d_positive = gamma_positive * a_positive**2 / (std_positive * (num_positive - 1))

    return (
        gamma_negative,
        gamma_positive,
        a_negative,
        a_positive,
        b_negative,
        b_positive,
        d_negative,
        d_positive,
    )


def _solve_envelope_quartic(
    a_negative: float,
    a_positive: float,
    b_negative: float,
    b_positive: float,
    d_negative: float,
    d_positive: float,
    chi2_critical: float,
) -> tuple[NDArray, NDArray]:
    """Solve the quartic equation for the ellipse envelope.

    Uses the Demidenko (2012) appendix quartic formulation to find envelope
    points in the (u, v) offset space.

    The quartic coefficients satisfy:
        p4*η^4 + p3*η^3 + p2*η^2 + p1*η + p0 = 0

    Args:
        a_negative: A coefficient for negative class.
        a_positive: A coefficient for positive class.
        b_negative: B coefficient for negative class.
        b_positive: B coefficient for positive class.
        d_negative: D coefficient for negative class.
        d_positive: D coefficient for positive class.
        chi2_critical: Chi-squared critical value.

    Returns:
        Tuple of (u_values, v_values) for envelope points in offset space.
    """
    q = chi2_critical

    # Quartic coefficients from Demidenko appendix
    p4 = (
        a_positive**2 * d_negative**2
        - 2 * d_negative * d_positive * a_negative * a_positive
        + a_negative**2 * d_positive**2
    )
    p3 = (
        2 * d_negative * a_negative * a_positive * b_positive
        - 2 * a_negative**2 * d_positive * b_positive
    )
    p2 = (
        a_negative * a_positive * b_positive**2
        + 2 * d_negative * a_positive * d_positive * q
        + b_positive**2 * a_negative**2
        - 2 * d_negative**2 * a_positive * q
    )
    p1 = -2 * a_negative * d_negative * b_positive * q
    p0 = -(b_negative**2) * a_negative * q + d_negative**2 * q**2

    coefficients = np.array([p4, p3, p2, p1, p0])

    # Solve quartic using numpy
    eta_roots = _solve_quartic_numpy(coefficients)

    if len(eta_roots) == 0:
        return np.array([]), np.array([])

    # v = η
    v_values = eta_roots

    # u = ((A0*D1 - D0*A1)*η² - A0*B1*η + D0*q) / (B0*A0)
    denominator = b_negative * a_negative
    if np.abs(denominator) < 1e-15:
        return np.array([]), np.array([])

    u_values = (
        (a_negative * d_positive - d_negative * a_positive) * eta_roots**2
        - a_negative * b_positive * eta_roots
        + d_negative * q
    ) / denominator

    return u_values, v_values


def ellipse_envelope_band(
    y_true: "NDArray | Tensor",
    y_score: "NDArray | Tensor",
    num_grid_points: int = 1000,
    alpha: float = 0.05,
    minimum_std: float = 1e-8,
    probit_clip: float = 1e-9,
    envelope_method: Literal["sweep", "quartic"] = "sweep",
    num_cutoffs: int = 1000,
    plot: bool = False,
    plot_title: str | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute ellipse-envelope simultaneous confidence bands for ROC curves.

    This implements the ellipse-envelope method from Demidenko (2012), which
    provides improved coverage probability over the classic Working-Hotelling
    approach by properly accounting for variance estimation uncertainty.

    The method assumes a binormal model where scores in both classes follow
    normal distributions (after possible transformation).

    Args:
        y_true: Array of binary class labels (0/1). Can be numpy array or torch tensor.
        y_score: Array of continuous prediction scores. Can be numpy array or torch tensor.
            Higher scores should indicate higher probability of positive class. Assumed to be
            binormal (i.e. pos and neg scores follow normal distributions). If bounded by [0, 1],
            it is probit-transformed.
        num_grid_points: Number of evaluation points on the FPR grid.
            Higher values give smoother bands but increase computation time.
        alpha: Significance level for the confidence band.
            E.g., 0.05 gives a 95% simultaneous confidence band.
        minimum_std: Minimum allowed standard deviation to prevent numerical issues
            with near-constant data. Values below this are clamped.
        probit_clip: Clipping value for FPR grid to avoid infinite probit values at 0 and 1.
        envelope_method: Method for computing the envelope.
            - "sweep": Sweep through cutoffs and take min/max of ellipse boundaries (robust).
            - "quartic": Use quartic polynomial solution (as in Demidenko paper).
        num_cutoffs: Number of cutoff values to sweep through (for "sweep" method).
        plot: If True, generate diagnostic plots using the viz module (default False).
        plot_title: Optional custom title for the diagnostic plots. If None, uses
            method description.

    Returns:
        Tuple of three numpy arrays:
            - fpr_grid: False positive rate grid of shape (num_grid_points,).
            - lower_envelope: Lower confidence band (TPR) of shape (num_grid_points,).
            - upper_envelope: Upper confidence band (TPR) of shape (num_grid_points,).

    Raises:
        ValueError: If inputs have incompatible shapes or invalid values.

    References:
        Demidenko, E. (2012). "Confidence intervals and bands for the binormal ROC
        curve revisited". Journal of Applied Statistics, 39(1), 67-79.
        https://doi.org/10.1080/02664763.2011.578616

    Examples:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> y_true = np.array([0] * 100 + [1] * 100)
        >>> y_score = np.concatenate(
        ...     [np.random.normal(0, 1, 100), np.random.normal(1.5, 1.2, 100)]
        ... )
        >>> fpr, lower, upper = ellipse_envelope_band(y_true, y_score)
    """
    # Convert to numpy
    y_score_np, dtype = _convert_to_numpy(y_score)
    y_true_np, _ = _convert_to_numpy(y_true)

    # Validate inputs
    if y_score_np.shape != y_true_np.shape:
        raise ValueError(
            f"Shape mismatch: y_score {y_score_np.shape} vs y_true {y_true_np.shape}"
        )

    unique_labels = np.unique(y_true_np)
    if not np.array_equal(unique_labels, np.array([0, 1])):
        raise ValueError(
            f"y_true must contain exactly classes 0 and 1, got {unique_labels}"
        )

    # Check if scores are bounded on (0,1) and apply probit transformation if so
    if np.all((y_score_np > 0) & (y_score_np < 1)):
        # Scores appear to be probabilities - apply probit transformation
        # Clip to avoid infinite values at boundaries
        y_score_clipped = np.clip(y_score_np, probit_clip, 1 - probit_clip)
        y_score_np = norm.ppf(y_score_clipped)

    # Estimate binormal parameters
    mean_neg, std_neg, mean_pos, std_pos, n_neg, n_pos = _estimate_binormal_parameters(
        y_true_np, y_score_np, minimum_std
    )

    # Validate sample sizes
    if n_neg < 2 or n_pos < 2:
        raise ValueError(
            f"Need at least 2 samples per class, got n_neg={n_neg}, n_pos={n_pos}"
        )

    # Chi-squared critical value
    chi2_critical = chi2.ppf(1 - alpha, df=2)

    # Create FPR grid
    fpr_grid = np.linspace(0, 1, num_grid_points, dtype=dtype)
    fpr_clipped = np.clip(fpr_grid, probit_clip, 1 - probit_clip)
    probit_fpr = norm.ppf(fpr_clipped)

    # Initialize envelope arrays with extreme values
    lower_envelope = np.ones(num_grid_points, dtype=dtype)
    upper_envelope = np.zeros(num_grid_points, dtype=dtype)

    if envelope_method == "sweep":
        # Generate cutoffs (increase count to reduce discretization bias)
        # +/- 5 sigma covers FPR from ~3e-7 to 1-3e-7
        probit_range = np.linspace(-5, 5, num_cutoffs)
        cutoffs = mean_neg - std_neg * probit_range

        # Pre-allocate arrays for vectorization
        # shape: (num_cutoffs, num_grid_points)
        lower_candidates = np.ones((len(cutoffs), num_grid_points), dtype=dtype)
        upper_candidates = np.zeros((len(cutoffs), num_grid_points), dtype=dtype)

        for i, cutoff in enumerate(cutoffs):
            gamma_neg = (mean_neg - cutoff) / std_neg
            gamma_pos = (mean_pos - cutoff) / std_pos

            var_gamma_neg = 1.0 / n_neg + gamma_neg**2 / (2 * (n_neg - 1))
            var_gamma_pos = 1.0 / n_pos + gamma_pos**2 / (2 * (n_pos - 1))

            a_neg = 1.0 / var_gamma_neg
            a_pos = 1.0 / var_gamma_pos

            # VECTORIZED: Compute intersection for all grid points at once
            # Equation: A_neg * (x - gamma_neg)^2 + A_pos * (y - gamma_pos)^2 = chi2

            # 1. Calculate the 'cost' of the x-deviation
            x_cost = a_neg * (probit_fpr - gamma_neg) ** 2

            # 2. Find valid indices where the ellipse exists (cost < critical val)
            remaining = chi2_critical - x_cost
            valid_mask = remaining >= 0

            # 3. Compute y boundaries where valid
            if np.any(valid_mask):
                y_offset = np.sqrt(remaining[valid_mask] / a_pos)

                # Probit space boundaries
                y_lower = gamma_pos - y_offset
                y_upper = gamma_pos + y_offset

                # Transform to TPR (probability) space
                tpr_lower = norm.cdf(y_lower)
                tpr_upper = norm.cdf(y_upper)

                # Store in candidate arrays
                # Initialize invalid points to 1.0 (lower) and 0.0 (upper) so they don't affect min/max
                lower_candidates[i, valid_mask] = tpr_lower
                upper_candidates[i, valid_mask] = tpr_upper

        # Collapse candidates to finding the envelope
        # min over cutoffs for lower band, max over cutoffs for upper band
        lower_envelope = np.min(lower_candidates, axis=0)
        upper_envelope = np.max(upper_candidates, axis=0)

    else:  # quartic method
        # Use the quartic polynomial approach from Demidenko's paper
        #
        # The key insight: for each cutoff c, the quartic gives envelope points
        # with coordinates (γ̂₀(c) + u, γ̂₁(c) + v) in probit space.
        # These points do NOT align with the FPR grid, so we must:
        # 1. Collect all envelope points across many cutoffs
        # 2. Convert to ROC space (FPR, TPR)
        # 3. Interpolate onto the desired FPR grid

        # Generate dense cutoffs for envelope point collection
        probit_cutoff_range = np.linspace(-5, 5, num_cutoffs)
        cutoff_values = mean_neg - std_neg * probit_cutoff_range

        # Collect all envelope points
        envelope_fpr_lower: list[float] = []
        envelope_tpr_lower: list[float] = []
        envelope_fpr_upper: list[float] = []
        envelope_tpr_upper: list[float] = []

        for cutoff in cutoff_values:
            gamma_neg, gamma_pos, a_neg, a_pos, b_neg, b_pos, d_neg, d_pos = (
                _compute_ellipse_coefficients(
                    cutoff,
                    mean_neg,
                    std_neg,
                    mean_pos,
                    std_pos,
                    n_neg,
                    n_pos,
                    chi2_critical,
                )
            )

            u_vals, v_vals = _solve_envelope_quartic(
                a_neg, a_pos, b_neg, b_pos, d_neg, d_pos, chi2_critical
            )

            if len(v_vals) > 0:
                # Compute actual envelope point coordinates in probit space
                # x = γ̂₀(c) + u, y = γ̂₁(c) + v
                probit_fpr_points = gamma_neg + u_vals
                probit_tpr_points = gamma_pos + v_vals

                # Convert to ROC space
                fpr_points = norm.cdf(probit_fpr_points)
                tpr_points = norm.cdf(probit_tpr_points)

                # Filter valid points
                valid_mask = (
                    np.isfinite(fpr_points)
                    & np.isfinite(tpr_points)
                    & (fpr_points >= 0)
                    & (fpr_points <= 1)
                    & (tpr_points >= 0)
                    & (tpr_points <= 1)
                )

                if np.any(valid_mask):
                    valid_fpr = fpr_points[valid_mask]
                    valid_tpr = tpr_points[valid_mask]

                    # Separate into lower and upper envelope points
                    # Points below the ROC curve center go to lower envelope
                    # Points above go to upper envelope
                    tpr_center = norm.cdf(gamma_pos)

                    for f, t in zip(valid_fpr, valid_tpr):
                        if t <= tpr_center:
                            envelope_fpr_lower.append(f)
                            envelope_tpr_lower.append(t)
                        else:
                            envelope_fpr_upper.append(f)
                            envelope_tpr_upper.append(t)

        # Convert to arrays and sort by FPR
        if envelope_fpr_lower:
            lower_points = np.array(list(zip(envelope_fpr_lower, envelope_tpr_lower)))
            lower_points = lower_points[lower_points[:, 0].argsort()]

            # Interpolate onto the FPR grid
            # Use minimum TPR at each FPR for lower envelope
            from scipy.interpolate import interp1d

            if len(lower_points) >= 2:
                interp_lower = interp1d(
                    lower_points[:, 0],
                    lower_points[:, 1],
                    kind="linear",
                    bounds_error=False,
                    fill_value=(0.0, 1.0),
                )
                lower_envelope = np.clip(interp_lower(fpr_grid), 0.0, 1.0).astype(dtype)

        if envelope_fpr_upper:
            upper_points = np.array(list(zip(envelope_fpr_upper, envelope_tpr_upper)))
            upper_points = upper_points[upper_points[:, 0].argsort()]

            # Interpolate onto the FPR grid
            # Use maximum TPR at each FPR for upper envelope
            if len(upper_points) >= 2:
                interp_upper = interp1d(
                    upper_points[:, 0],
                    upper_points[:, 1],
                    kind="linear",
                    bounds_error=False,
                    fill_value=(0.0, 1.0),
                )
                upper_envelope = np.clip(interp_upper(fpr_grid), 0.0, 1.0).astype(dtype)

        # If interpolation failed, fall back to sweep method bounds
        if np.all(lower_envelope == 1.0) or np.all(upper_envelope == 0.0):
            # Recompute using sweep as fallback
            probit_range = np.linspace(-4, 4, num_cutoffs)
            cutoffs_sweep = mean_neg - std_neg * probit_range
            lower_envelope = np.ones(num_grid_points, dtype=dtype)
            upper_envelope = np.zeros(num_grid_points, dtype=dtype)

            for cutoff in cutoffs_sweep:
                gamma_neg = (mean_neg - cutoff) / std_neg
                gamma_pos = (mean_pos - cutoff) / std_pos
                var_gamma_neg = 1.0 / n_neg + gamma_neg**2 / (2 * (n_neg - 1))
                var_gamma_pos = 1.0 / n_pos + gamma_pos**2 / (2 * (n_pos - 1))
                a_neg = 1.0 / var_gamma_neg
                a_pos = 1.0 / var_gamma_pos

                for i, x in enumerate(probit_fpr):
                    x_deviation_sq = (x - gamma_neg) ** 2
                    remaining = chi2_critical - a_neg * x_deviation_sq
                    if remaining >= 0:
                        y_offset = np.sqrt(remaining / a_pos)
                        tpr_lower = norm.cdf(gamma_pos - y_offset)
                        tpr_upper = norm.cdf(gamma_pos + y_offset)
                        lower_envelope[i] = min(lower_envelope[i], tpr_lower)
                        upper_envelope[i] = max(upper_envelope[i], tpr_upper)

    # Handle any remaining extreme values
    lower_envelope = np.clip(lower_envelope, 0.0, 1.0)
    upper_envelope = np.clip(upper_envelope, 0.0, 1.0)

    # Fix endpoints
    lower_envelope[0] = 0.0
    upper_envelope[-1] = 1.0

    # Ensure lower <= upper
    lower_envelope = np.minimum(lower_envelope, upper_envelope)

    # Apply monotonicity constraint (ROC bands should be non-decreasing)
    for i in range(1, num_grid_points):
        lower_envelope[i] = max(lower_envelope[i], lower_envelope[i - 1])
        upper_envelope[i] = max(upper_envelope[i], upper_envelope[i - 1])

    # Convert to final dtype before plotting/returning
    fpr_grid_final = fpr_grid.astype(dtype)
    lower_envelope_final = lower_envelope.astype(dtype)
    upper_envelope_final = upper_envelope.astype(dtype)

    # Generate diagnostic plots if requested
    if plot:
        try:
            # Compute empirical ROC curve for visualization
            neg_scores = y_score_np[y_true_np == 0]
            pos_scores = y_score_np[y_true_np == 1]

            # Convert to torch tensors for compute_empirical_roc_from_scores
            import torch

            neg_scores_t = torch.from_numpy(neg_scores).float()
            pos_scores_t = torch.from_numpy(pos_scores).float()
            fpr_grid_t = torch.from_numpy(fpr_grid_final).float()

            empirical_tpr_t = compute_empirical_roc_from_scores(
                neg_scores_t, pos_scores_t, fpr_grid_t
            )
            empirical_tpr_np = empirical_tpr_t.cpu().numpy().astype(dtype)

            # Determine method name for title
            if plot_title is None:
                plot_title = f"Ellipse-Envelope ({envelope_method} method)"

            fig = plot_band_diagnostics(
                fpr_grid=fpr_grid_final,
                empirical_tpr=empirical_tpr_np,
                lower_envelope=lower_envelope_final,
                upper_envelope=upper_envelope_final,
                boot_tpr_matrix=None,
                bootstrap_var=None,
                wilson_var=None,
                alpha=alpha,
                method_name=plot_title,
                layout="2x2",
            )
            fig.show()
        except ImportError:
            import warnings

            warnings.warn(
                "Visualization module not available. Install matplotlib to enable plotting.",
                stacklevel=2,
            )

    return fpr_grid_final, lower_envelope_final, upper_envelope_final
