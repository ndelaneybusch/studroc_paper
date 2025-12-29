"""Publication-grade diagnostic visualizations for simultaneous confidence bands.

This module provides a suite of diagnostic plots for ROC confidence band methods,
designed to be publication-ready with consistent styling and flexible composition.

Example usage:
    ```python
    from src.viz.band_diagnostics import plot_band_diagnostics

    # After computing confidence bands, create all diagnostics
    fig = plot_band_diagnostics(
        fpr_grid=fpr,
        empirical_tpr=empirical_tpr,
        lower_envelope=lower_envelope,
        upper_envelope=upper_envelope,
        boot_tpr_matrix=boot_tpr,
        bootstrap_var=bootstrap_var,
        wilson_var=wilson_var,
        alpha=0.05,
        method_name="Envelope Bootstrap",
    )
    fig.savefig("diagnostics.png", dpi=300, bbox_inches="tight")
    ```
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray


def _configure_publication_style() -> dict[str, Any]:
    """Return matplotlib rcParams for publication-quality plots."""
    return {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "patch.linewidth": 0.5,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "figure.dpi": 150,
    }


def _compute_band_area(
    fpr_grid: NDArray, lower_envelope: NDArray, upper_envelope: NDArray
) -> float:
    """Compute the area between upper and lower confidence bands.

    Uses trapezoidal integration.

    Args:
        fpr_grid: FPR values at which bands are evaluated.
        lower_envelope: Lower confidence bound.
        upper_envelope: Upper confidence bound.

    Returns:
        Integrated area between bands.
    """
    band_width = upper_envelope - lower_envelope
    return float(np.trapz(band_width, fpr_grid))


def plot_roc_with_band(
    fpr_grid: NDArray,
    empirical_tpr: NDArray,
    lower_envelope: NDArray,
    upper_envelope: NDArray,
    alpha: float = 0.05,
    ax: Axes | None = None,
    show_band_area: bool = True,
    additional_curves: dict[str, NDArray] | None = None,
    method_name: str = "Confidence Band",
) -> Axes:
    """Plot empirical ROC curve with simultaneous confidence band.

    Args:
        fpr_grid: FPR values at which ROC is evaluated (shape: n_grid).
        empirical_tpr: Empirical TPR values (shape: n_grid).
        lower_envelope: Lower confidence bound (shape: n_grid).
        upper_envelope: Upper confidence bound (shape: n_grid).
        alpha: Significance level (default: 0.05).
        ax: Matplotlib axes object. If None, creates new figure.
        show_band_area: Whether to display band area in title (default: True).
        additional_curves: Optional dict of {label: tpr_values} for additional curves
            to overlay (e.g., {"Smoothed ROC": smoothed_tpr}).
        method_name: Name of the method for the title (default: "Confidence Band").

    Returns:
        Matplotlib Axes object containing the plot.

    Example:
        ```python
        ax = plot_roc_with_band(
            fpr_grid, empirical_tpr, lower_envelope, upper_envelope,
            alpha=0.05,
            additional_curves={"BP-smoothed": roc_center},
            method_name="BP Bootstrap"
        )
        ```
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Confidence band (filled region)
    ax.fill_between(
        fpr_grid,
        lower_envelope,
        upper_envelope,
        alpha=0.25,
        color="steelblue",
        label=f"{int((1-alpha)*100)}% Simultaneous Band",
    )

    # Plot band boundaries
    ax.plot(fpr_grid, lower_envelope, "--", color="steelblue", linewidth=1.0, alpha=0.7)
    ax.plot(fpr_grid, upper_envelope, "--", color="steelblue", linewidth=1.0, alpha=0.7)

    # Plot additional curves (e.g., smoothed ROC) before empirical so they appear below
    if additional_curves is not None:
        for label, tpr_values in additional_curves.items():
            ax.plot(
                fpr_grid,
                tpr_values,
                "-",
                linewidth=2.0,
                alpha=0.8,
                label=label,
            )

    # Empirical ROC curve
    ax.plot(
        fpr_grid,
        empirical_tpr,
        "k-",
        linewidth=1.5,
        label="Empirical ROC",
        zorder=10,
    )

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.3, label="Chance")

    # Styling
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Title with band area if requested
    if show_band_area:
        band_area = _compute_band_area(fpr_grid, lower_envelope, upper_envelope)
        title = f"{method_name}\n(Band Area = {band_area:.4f})"
    else:
        title = method_name

    ax.set_title(title, fontweight="bold")

    # Legend
    ax.legend(loc="lower right", framealpha=0.95)

    return ax


def plot_variance_comparison(
    fpr_grid: NDArray,
    bootstrap_var: NDArray,
    wilson_var: NDArray | None = None,
    ax: Axes | None = None,
    log_scale: bool = False,
    show_std: bool = False,
) -> Axes:
    """Plot variance estimates as a function of FPR.

    Visualizes how variance changes across the ROC curve, comparing bootstrap
    variance to theoretical Wilson score variance floor (if provided).

    Args:
        fpr_grid: FPR values at which variance is evaluated (shape: n_grid).
        bootstrap_var: Bootstrap variance estimates (shape: n_grid).
        wilson_var: Wilson score variance floor (shape: n_grid), optional.
        ax: Matplotlib axes object. If None, creates new figure.
        log_scale: Whether to use log scale for y-axis (default: False).
        show_std: Whether to plot standard deviation instead of variance (default: False).

    Returns:
        Matplotlib Axes object containing the plot.

    Example:
        ```python
        ax = plot_variance_comparison(
            fpr_grid, bootstrap_var, wilson_var,
            log_scale=True, show_std=True
        )
        ```
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # Convert to std if requested
    if show_std:
        bootstrap_y = np.sqrt(bootstrap_var)
        wilson_y = np.sqrt(wilson_var) if wilson_var is not None else None
        ylabel = "Standard Deviation"
        title = "Variance Estimation (Std Dev)"
    else:
        bootstrap_y = bootstrap_var
        wilson_y = wilson_var
        ylabel = "Variance"
        title = "Variance Estimation"

    # Plot bootstrap variance
    ax.plot(
        fpr_grid,
        bootstrap_y,
        "-",
        color="steelblue",
        linewidth=2.0,
        label="Bootstrap",
    )

    # Plot Wilson variance if provided
    if wilson_y is not None:
        ax.plot(
            fpr_grid,
            wilson_y,
            "--",
            color="darkorange",
            linewidth=2.0,
            label="Wilson Floor",
        )

        # Highlight regions where Wilson floor is active
        active_floor = bootstrap_var <= wilson_var
        if np.any(active_floor):
            ax.fill_between(
                fpr_grid,
                0,
                bootstrap_y,
                where=active_floor,
                alpha=0.15,
                color="darkorange",
                label="Wilson Active",
            )

    # Styling
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(1e-10, np.min(bootstrap_y[bootstrap_y > 0]) * 0.5))
    else:
        ax.set_ylim(bottom=0)

    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="best", framealpha=0.95)

    return ax


def plot_bootstrap_vs_empirical(
    fpr_grid: NDArray,
    empirical_tpr: NDArray,
    boot_tpr_matrix: NDArray,
    ax: Axes | None = None,
    show_individual_curves: bool = False,
    n_curves_to_show: int = 50,
    alpha_curves: float = 0.1,
) -> Axes:
    """Plot mean of bootstrap ROC curves vs empirical ROC.

    Visualizes the agreement between the empirical ROC and the mean of bootstrap
    replicates, which is a diagnostic for bootstrap bias.

    Args:
        fpr_grid: FPR values at which ROC is evaluated (shape: n_grid).
        empirical_tpr: Empirical TPR values (shape: n_grid).
        boot_tpr_matrix: Bootstrap TPR matrix (shape: n_bootstrap, n_grid).
        ax: Matplotlib axes object. If None, creates new figure.
        show_individual_curves: Whether to show individual bootstrap curves (default: False).
        n_curves_to_show: Number of individual curves to plot if show_individual_curves=True.
        alpha_curves: Alpha transparency for individual curves.

    Returns:
        Matplotlib Axes object containing the plot.

    Example:
        ```python
        ax = plot_bootstrap_vs_empirical(
            fpr_grid, empirical_tpr, boot_tpr_matrix,
            show_individual_curves=True, n_curves_to_show=100
        )
        ```
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Compute bootstrap mean
    boot_mean = np.mean(boot_tpr_matrix, axis=0)

    # Optionally show individual bootstrap curves
    if show_individual_curves:
        n_bootstrap = boot_tpr_matrix.shape[0]
        indices = np.linspace(0, n_bootstrap - 1, n_curves_to_show, dtype=int)
        for i in indices:
            ax.plot(
                fpr_grid,
                boot_tpr_matrix[i],
                "-",
                color="gray",
                linewidth=0.5,
                alpha=alpha_curves,
            )

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.3, label="Chance")

    # Bootstrap mean
    ax.plot(
        fpr_grid,
        boot_mean,
        "-",
        color="steelblue",
        linewidth=2.5,
        label="Bootstrap Mean",
        zorder=5,
    )

    # Empirical ROC
    ax.plot(
        fpr_grid,
        empirical_tpr,
        "-",
        color="darkred",
        linewidth=2.0,
        label="Empirical ROC",
        zorder=10,
    )

    # Compute and display bias
    bias = boot_mean - empirical_tpr
    max_abs_bias = np.max(np.abs(bias))
    mean_abs_bias = np.mean(np.abs(bias))

    # Styling
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    title = f"Bootstrap vs Empirical ROC\n(Max |Bias| = {max_abs_bias:.4f}, Mean |Bias| = {mean_abs_bias:.4f})"
    ax.set_title(title, fontweight="bold")

    # Legend
    if show_individual_curves:
        legend_label = f"Individual Curves (n={n_curves_to_show})"
        handles, labels = ax.get_legend_handles_labels()
        # Add proxy artist for individual curves
        from matplotlib.lines import Line2D

        handles.insert(
            0, Line2D([0], [0], color="gray", linewidth=0.5, alpha=alpha_curves)
        )
        labels.insert(0, legend_label)
        ax.legend(handles, labels, loc="lower right", framealpha=0.95)
    else:
        ax.legend(loc="lower right", framealpha=0.95)

    return ax


def plot_bias_profile(
    fpr_grid: NDArray,
    empirical_tpr: NDArray,
    boot_tpr_matrix: NDArray,
    ax: Axes | None = None,
) -> Axes:
    """Plot bootstrap bias as a function of FPR.

    Shows the pointwise difference between bootstrap mean and empirical ROC,
    along with bias confidence intervals.

    Args:
        fpr_grid: FPR values at which ROC is evaluated (shape: n_grid).
        empirical_tpr: Empirical TPR values (shape: n_grid).
        boot_tpr_matrix: Bootstrap TPR matrix (shape: n_bootstrap, n_grid).
        ax: Matplotlib axes object. If None, creates new figure.

    Returns:
        Matplotlib Axes object containing the plot.

    Example:
        ```python
        ax = plot_bias_profile(fpr_grid, empirical_tpr, boot_tpr_matrix)
        ```
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # Compute bias
    boot_mean = np.mean(boot_tpr_matrix, axis=0)
    bias = boot_mean - empirical_tpr

    # Compute bias standard error (std of the bias estimate)
    boot_std = np.std(boot_tpr_matrix, axis=0, ddof=1)
    n_bootstrap = boot_tpr_matrix.shape[0]
    bias_se = boot_std / np.sqrt(n_bootstrap)

    # Plot bias
    ax.plot(fpr_grid, bias, "-", color="steelblue", linewidth=2.0, label="Bias")

    # Plot ±2 SE bands
    ax.fill_between(
        fpr_grid,
        bias - 2 * bias_se,
        bias + 2 * bias_se,
        alpha=0.25,
        color="steelblue",
        label="±2 SE",
    )

    # Zero reference line
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)

    # Styling
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("Bias (Bootstrap Mean - Empirical)")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_title("Bootstrap Bias Profile", fontweight="bold")
    ax.legend(loc="best", framealpha=0.95)

    return ax


def plot_band_diagnostics(
    fpr_grid: NDArray,
    empirical_tpr: NDArray,
    lower_envelope: NDArray,
    upper_envelope: NDArray,
    boot_tpr_matrix: NDArray | None = None,
    bootstrap_var: NDArray | None = None,
    wilson_var: NDArray | None = None,
    alpha: float = 0.05,
    method_name: str = "Confidence Band",
    additional_curves: dict[str, NDArray] | None = None,
    layout: str = "2x2",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Create a comprehensive diagnostic plot panel for confidence bands.

    Combines multiple diagnostic visualizations into a single figure with
    publication-quality styling.

    Args:
        fpr_grid: FPR values at which ROC is evaluated (shape: n_grid).
        empirical_tpr: Empirical TPR values (shape: n_grid).
        lower_envelope: Lower confidence bound (shape: n_grid).
        upper_envelope: Upper confidence bound (shape: n_grid).
        boot_tpr_matrix: Bootstrap TPR matrix (shape: n_bootstrap, n_grid), optional.
            Required for bootstrap comparison plots.
        bootstrap_var: Bootstrap variance estimates (shape: n_grid), optional.
            Required for variance comparison plot.
        wilson_var: Wilson score variance floor (shape: n_grid), optional.
        alpha: Significance level (default: 0.05).
        method_name: Name of the method for titles (default: "Confidence Band").
        additional_curves: Optional dict of {label: tpr_values} for ROC plot.
        layout: Layout style - "2x2" (default), "1x3", or "1x4".
        figsize: Figure size as (width, height). If None, uses default based on layout.

    Returns:
        Matplotlib Figure object containing all diagnostic plots.

    Example:
        ```python
        fig = plot_band_diagnostics(
            fpr_grid, empirical_tpr, lower_envelope, upper_envelope,
            boot_tpr_matrix=boot_tpr,
            bootstrap_var=bootstrap_var,
            wilson_var=wilson_var,
            alpha=0.05,
            method_name="Envelope Bootstrap",
            additional_curves={"Smoothed": roc_smoothed},
            layout="2x2"
        )
        fig.savefig("diagnostics.png", dpi=300, bbox_inches="tight")
        ```
    """
    # Apply publication styling
    with plt.rc_context(_configure_publication_style()):
        # Determine layout
        if layout == "2x2":
            if figsize is None:
                figsize = (10, 9)
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            axes = axes.flatten()
        elif layout == "1x3":
            if figsize is None:
                figsize = (15, 4.5)
            fig, axes = plt.subplots(1, 3, figsize=figsize)
        elif layout == "1x4":
            if figsize is None:
                figsize = (18, 4)
            fig, axes = plt.subplots(1, 4, figsize=figsize)
        else:
            raise ValueError(f"Invalid layout: {layout}. Choose '2x2', '1x3', or '1x4'.")

        # Plot 1: ROC with confidence band
        plot_roc_with_band(
            fpr_grid,
            empirical_tpr,
            lower_envelope,
            upper_envelope,
            alpha=alpha,
            ax=axes[0],
            show_band_area=True,
            additional_curves=additional_curves,
            method_name=method_name,
        )

        # Plot 2: Variance comparison (if variance data provided)
        if bootstrap_var is not None:
            plot_variance_comparison(
                fpr_grid,
                bootstrap_var,
                wilson_var=wilson_var,
                ax=axes[1],
                log_scale=False,
                show_std=True,
            )
        else:
            axes[1].text(
                0.5,
                0.5,
                "Variance data not provided",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )
            axes[1].set_title("Variance Estimation", fontweight="bold")

        # Plot 3: Bootstrap vs Empirical (if bootstrap matrix provided)
        if boot_tpr_matrix is not None:
            plot_bootstrap_vs_empirical(
                fpr_grid,
                empirical_tpr,
                boot_tpr_matrix,
                ax=axes[2],
                show_individual_curves=True,
                n_curves_to_show=50,
            )
        else:
            axes[2].text(
                0.5,
                0.5,
                "Bootstrap matrix not provided",
                ha="center",
                va="center",
                transform=axes[2].transAxes,
            )
            axes[2].set_title("Bootstrap vs Empirical ROC", fontweight="bold")

        # Plot 4: Bias profile (if bootstrap matrix provided and layout has 4 panels)
        if len(axes) > 3:
            if boot_tpr_matrix is not None:
                plot_bias_profile(
                    fpr_grid,
                    empirical_tpr,
                    boot_tpr_matrix,
                    ax=axes[3],
                )
            else:
                axes[3].text(
                    0.5,
                    0.5,
                    "Bootstrap matrix not provided",
                    ha="center",
                    va="center",
                    transform=axes[3].transAxes,
                )
                axes[3].set_title("Bootstrap Bias Profile", fontweight="bold")

        # Adjust layout
        fig.tight_layout()

        return fig
