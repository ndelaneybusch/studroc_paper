"""Variance-model bootstrap confidence bands for ROC curves.

This module implements simultaneous confidence bands using a variance-model
approach with bootstrap-calibrated critical values. Unlike the envelope method
(which retains bootstrap curves and takes their pointwise min/max), this method
constructs a band: R_hat(t) +/- c_alpha * sigma_hat(t), where c_alpha is the
(1-alpha)-quantile of the bootstrap studentized supremum statistic and sigma_hat
is the bootstrap standard deviation with a Wilson variance floor at boundary
points.

The primary entry point is :func:`variance_model_band`, which accepts a
pre-computed bootstrap TPR matrix and returns lower/upper confidence bands.

This construction combines:

- Bootstrap variance (nonparametric, adaptive) for local uncertainty.
- Bootstrap calibration of the critical value (respects the correlation
  structure of the ROC process) from HT-autocalib.
- Wilson variance-ratio floor (boundary correction) from the envelope approach.
- Smooth tunability across confidence levels (band width proportional to c,
  which varies continuously with alpha).

The key advantage over the envelope: the band width scales smoothly with the
confidence level, so the 50% band is genuinely narrower than the 95% band.
The envelope method has weak sensitivity to alpha because the sup-norm critical
value ratio c_{0.95}/c_{0.50} is modest for correlated processes.
"""

from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from studroc_paper.viz import plot_band_diagnostics

from .envelope_boot import (
    _compute_empirical_roc,
    _compute_variance_ratio_alpha,
    _haldane_logit,
)
from .method_utils import numpy_to_torch, torch_to_numpy, wilson_halfwidth_squared_torch

#: Method for TPR estimation: ``"empirical"`` for standard step-function
#: interpolation, or ``"harrell_davis"`` for beta-weighted quantile estimation.
TprMethod = Literal["empirical", "harrell_davis"]


def variance_model_band(
    boot_tpr_matrix: NDArray | Tensor,
    fpr_grid: NDArray | Tensor,
    y_true: NDArray | Tensor,
    y_score: NDArray | Tensor,
    alpha: float = 0.05,
    use_logit: bool = False,
    use_wilson_floor: bool = True,
    tpr_method: TprMethod = "empirical",
    plot: bool = False,
    plot_title: str | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute variance-model bootstrap simultaneous confidence bands for ROC curves.

    Constructs a band R_hat(t) +/- c_alpha * sigma_hat(t) where sigma_hat is the
    bootstrap standard deviation (with optional Wilson variance floor) and c_alpha
    is the (1-alpha)-quantile of the bootstrap studentized supremum statistic.

    The ``boot_tpr_matrix`` is typically pre-computed via
    :func:`studroc_paper.sampling.bootstrap_grid.generate_bootstrap_grid`.

    Args:
        boot_tpr_matrix: (n_bootstrap, n_grid_points) array of bootstrap TPR
            values, pre-computed via
            :func:`~studroc_paper.sampling.bootstrap_grid.generate_bootstrap_grid`.
        fpr_grid: (n_grid_points,) array of FPR evaluation points.
        y_true: Array of true binary labels (0 or 1) from the original sample.
        y_score: Array of predicted scores from the original sample.
        alpha: Significance level. Defaults to 0.05.
        use_logit: If True, construct the band in logit space using the
            Haldane-Anscombe correction and back-transform via sigmoid. This
            produces asymmetric bands in probability space that naturally
            respect [0, 1] boundaries. Defaults to False.
        use_wilson_floor: If True, apply a Wilson variance-ratio floor where
            bootstrap variance has collapsed below the binomial minimum. Uses
            the variance ratio r(t) = bootstrap_var / wilson_var to detect
            deficient points and applies Sidak-corrected Wilson variance as
            a floor. Defaults to True.
        tpr_method: Method for computing the empirical ROC curve (band center).
            ``"empirical"`` uses standard step-function interpolation;
            ``"harrell_davis"`` uses beta-weighted quantile estimation for
            reduced finite-sample bias. Defaults to ``"empirical"``.
        plot: If True, generate diagnostic plots via
            :func:`~studroc_paper.viz.plot_band_diagnostics`. Defaults to False.
        plot_title: Custom title for diagnostic plots. If None, a descriptive
            title is generated from method parameters. Defaults to None.

    Returns:
        Tuple of ``(fpr_grid, lower_band, upper_band)`` as
        :py:class:`numpy.ndarray` arrays, each of shape ``(n_grid_points,)``,
        with the same dtype as the input ``y_score``. The lower and upper bands
        are clipped to [0, 1] with boundary conditions enforced at the
        endpoints (lower[0] = 0, upper[-1] = 1).

    Notes:
        When ``use_logit=True``, the band is constructed in logit space via the
        Haldane-Anscombe correction and back-transformed through the sigmoid.
        This yields asymmetric bands in probability space that naturally respect
        [0, 1] boundaries without clipping artifacts.

        The Wilson variance floor (when enabled) uses a Sidak-corrected alpha
        derived from the variance-ratio profile
        (:func:`~studroc_paper.methods.envelope_boot._compute_variance_ratio_alpha`)
        to avoid over-correction at interior points while providing adequate
        coverage at boundary points where bootstrap variance collapses.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from studroc_paper.sampling.bootstrap_grid import generate_bootstrap_grid
        >>> rng = np.random.default_rng(42)
        >>> y_true = np.concatenate([np.zeros(100), np.ones(100)])
        >>> y_score = np.concatenate([rng.normal(0, 1, 100), rng.normal(1.5, 1, 100)])
        >>> fpr_grid = np.linspace(0, 1, 101)
        >>> boot_tpr = generate_bootstrap_grid(
        ...     y_true=torch.from_numpy(y_true),
        ...     y_score=torch.from_numpy(y_score),
        ...     B=2000,
        ...     grid=torch.from_numpy(fpr_grid),
        ... )
        >>> fpr, lower, upper = variance_model_band(
        ...     boot_tpr_matrix=boot_tpr,
        ...     fpr_grid=fpr_grid,
        ...     y_true=y_true,
        ...     y_score=y_score,
        ...     alpha=0.05,
        ...     use_wilson_floor=True,
        ... )
        >>> fpr.shape
        (101,)
        >>> np.all((lower >= 0) & (upper <= 1))
        True
        >>> lower[0] == 0.0 and upper[-1] == 1.0
        True
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine output dtype from y_score
    if isinstance(y_score, np.ndarray):
        dtype = y_score.dtype
    elif isinstance(y_score, Tensor):
        dtype = y_score.cpu().numpy().dtype
    else:
        dtype = np.asarray(y_score).dtype

    # Convert inputs to tensors
    boot_tpr = numpy_to_torch(boot_tpr_matrix, device).float()
    fpr = numpy_to_torch(fpr_grid, device).float()
    y_true_t = numpy_to_torch(y_true, device)
    y_score_t = numpy_to_torch(y_score, device).float()

    n_neg = int((y_true_t == 0).sum().item())
    n_pos = int((y_true_t == 1).sum().item())
    n_total = n_neg + n_pos

    # Empirical ROC (band center)
    tpr_hat = _compute_empirical_roc(y_true_t, y_score_t, fpr, method=tpr_method)

    # z-value for Wilson CI scaling
    z_alpha = (2.0**0.5) * torch.erfinv(torch.tensor(1.0 - alpha)).item()

    # Bootstrap variance in probability space (needed for variance-ratio
    # detection regardless of which space the band is constructed in)
    bootstrap_var_prob = torch.var(boot_tpr, dim=0, correction=1)

    # Wilson variance in probability space (for ratio detection)
    wilson_var_prob = (
        wilson_halfwidth_squared_torch(tpr_hat, n_pos, z_alpha) / z_alpha**2
    )

    # Detect deficient points and compute Sidak-corrected alpha
    if use_wilson_floor:
        deficiency, alpha_wilson = _compute_variance_ratio_alpha(
            bootstrap_var_prob, wilson_var_prob, alpha
        )
        needs_floor = deficiency > 0

        # Wilson variance at the Sidak-corrected alpha level
        z_wilson = (2.0**0.5) * torch.erfinv(torch.tensor(1.0 - alpha_wilson)).item()
        wilson_var_corrected = (
            wilson_halfwidth_squared_torch(tpr_hat, n_pos, z_wilson) / z_wilson**2
        )
    else:
        needs_floor = torch.zeros(len(fpr), dtype=torch.bool, device=device)

    # Epsilon floor for numerical stability during studentization
    epsilon = min(1.0 / n_total, 1e-6)

    if use_logit:
        # ---- Logit-space path ----
        # Transform to logit space via Haldane-Anscombe correction
        logit_tpr_hat = _haldane_logit(tpr_hat, n_pos)
        logit_boot_tpr = _haldane_logit(boot_tpr, n_pos)

        # Bootstrap variance in logit space
        logit_bootstrap_var = torch.var(logit_boot_tpr, dim=0, correction=1)

        # Wilson floor in logit space (delta-method Jacobian transform)
        if use_wilson_floor and needs_floor.any():
            p_safe = torch.clamp(tpr_hat, 1e-6, 1.0 - 1e-6)
            jacobian_sq = 1.0 / (p_safe * (1.0 - p_safe)) ** 2
            wilson_var_logit = wilson_var_corrected * jacobian_sq

            floored_logit_var = logit_bootstrap_var.clone()
            floored_logit_var[needs_floor] = torch.maximum(
                logit_bootstrap_var[needs_floor], wilson_var_logit[needs_floor]
            )
        else:
            floored_logit_var = logit_bootstrap_var

        sigma_hat = torch.sqrt(floored_logit_var)
        safe_sigma = torch.clamp(sigma_hat, min=epsilon)

        # Studentized supremum statistic in logit space
        logit_deviations = torch.abs(logit_boot_tpr - logit_tpr_hat.unsqueeze(0))
        studentized = logit_deviations / safe_sigma.unsqueeze(0)
        sup_stats = torch.max(studentized, dim=1).values

        # Bootstrap-calibrated critical value
        c_alpha = torch.quantile(sup_stats, 1.0 - alpha).item()

        # Construct band in logit space and back-transform
        logit_margin = c_alpha * sigma_hat
        logit_lower = logit_tpr_hat - logit_margin
        logit_upper = logit_tpr_hat + logit_margin

        lower = torch.sigmoid(logit_lower)
        upper = torch.sigmoid(logit_upper)

    else:
        # ---- Probability-space path ----
        # Apply Wilson floor to bootstrap variance
        if use_wilson_floor and needs_floor.any():
            floored_var = bootstrap_var_prob.clone()
            floored_var[needs_floor] = torch.maximum(
                bootstrap_var_prob[needs_floor], wilson_var_corrected[needs_floor]
            )
        else:
            floored_var = bootstrap_var_prob

        sigma_hat = torch.sqrt(floored_var)
        safe_sigma = torch.clamp(sigma_hat, min=epsilon)

        # Studentized supremum statistic
        deviations = torch.abs(boot_tpr - tpr_hat.unsqueeze(0))
        studentized = deviations / safe_sigma.unsqueeze(0)
        sup_stats = torch.max(studentized, dim=1).values

        # Bootstrap-calibrated critical value
        c_alpha = torch.quantile(sup_stats, 1.0 - alpha).item()

        # Construct band
        margin = c_alpha * sigma_hat
        lower = tpr_hat - margin
        upper = tpr_hat + margin

        # Clip to [0, 1]
        lower = torch.clamp(lower, 0.0, 1.0)
        upper = torch.clamp(upper, 0.0, 1.0)

    # Enforce boundary conditions and logical consistency
    lower[0] = 0.0
    upper[-1] = 1.0
    upper = torch.maximum(upper, lower)

    # Convert to numpy with original dtype
    fpr_np = torch_to_numpy(fpr).astype(dtype)
    lower_np = torch_to_numpy(lower).astype(dtype)
    upper_np = torch_to_numpy(upper).astype(dtype)

    # Diagnostic plots
    if plot:
        bootstrap_var_np = torch_to_numpy(bootstrap_var_prob).astype(dtype)
        wilson_var_np = (
            torch_to_numpy(wilson_var_prob).astype(dtype) if use_wilson_floor else None
        )

        try:
            empirical_tpr_np = torch_to_numpy(tpr_hat).astype(dtype)
            boot_tpr_np = torch_to_numpy(boot_tpr).astype(dtype)

            if plot_title is None:
                space_label = "logit" if use_logit else "probability"
                floor_label = "Wilson floor" if use_wilson_floor else "no floor"
                plot_title = f"Variance-Model Band ({space_label}, {floor_label})"

            fig = plot_band_diagnostics(
                fpr_grid=fpr_np,
                empirical_tpr=empirical_tpr_np,
                lower_envelope=lower_np,
                upper_envelope=upper_np,
                boot_tpr_matrix=boot_tpr_np,
                bootstrap_var=bootstrap_var_np,
                wilson_var=wilson_var_np,
                alpha=alpha,
                method_name=plot_title,
                layout="2x2",
            )
            fig.show()
        except ImportError:
            import warnings

            warnings.warn(
                "Visualization module not available. "
                "Install matplotlib to enable plotting.",
                stacklevel=2,
            )

    return (fpr_np, lower_np, upper_np)
