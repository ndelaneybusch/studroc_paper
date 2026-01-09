"""
ROC Confidence Band Visualization Functions

Professional, publication-quality visualizations for comparing ROC confidence band methods
across different synthetic datasets.
"""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

# =============================================================================
# Global Style Configuration
# =============================================================================


def set_publication_style() -> None:
    """Set global matplotlib/seaborn style for publication-quality figures.

    Configures font families, sizes, axes styling, tick parameters, and
    output settings optimized for academic publications. Uses colorblind-safe
    palette from seaborn.

    Examples:
        >>> set_publication_style()
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 9])
        >>> plt.savefig("publication_plot.pdf")
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )
    sns.set_palette("colorblind")


# =============================================================================
# Color Mapper Utility
# =============================================================================


def get_method_color(method: str) -> str:
    """Map a method name to a consistent color.

    Color scheme:
    - Envelope methods: shades of red, orange, and pink
    - HT methods: shades of blue and purple
    - pointwise: black
    - ks: dark brown
    - working_hotelling: light brown
    - All other methods: shades of yellow and green

    Args:
        method: The method name to map to a color.

    Returns:
        Hex color code for the method.

    Examples:
        >>> get_method_color("pointwise")
        '#000000'
        >>> get_method_color("envelope_standard")
        '#E53935'
        >>> get_method_color("ht_log_concave")
        '#1565C0'
    """
    method_lower = method.lower()

    # Special cases first
    if method_lower == "pointwise":
        return "#000000"  # Black
    if method_lower == "ks":
        return "#5D4037"  # Dark brown
    if method_lower == "working_hotelling":
        return "#A1887F"  # Light brown

    # Envelope methods - reds, oranges, pinks
    envelope_colors = {
        "envelope_standard": "#E53935",  # Red
        "envelope_symmetric": "#FF7043",  # Deep orange
        "envelope_kde": "#EC407A",  # Pink
        "envelope_wilson": "#C62828",  # Dark red
        "envelope_wilson_symmetric": "#FF5722",  # Orange
        "envelope_logit": "#F48FB1",  # Light pink
        "envelope_symmetric_logit": "#FFAB91",  # Light orange
        "envelope_wilson_logit": "#D32F2F",  # Medium red
        "envelope_wilson_symmetric_logit": "#FF8A65",  # Coral
        "envelope_kde_logit": "#F06292",  # Medium pink
        "envelope_kde_symmetric_logit": "#FFCCBC",  # Pale orange
    }

    # HT methods - blues and purples
    ht_colors = {
        "ht_log_concave": "#1565C0",  # Blue
        "ht_log_concave_calib": "#7B1FA2",  # Purple
        "ht_reflected_kde": "#0288D1",  # Light blue
        "ht_kde": "#303F9F",  # Indigo
        "ht_kde_calib": "#9C27B0",  # Purple
        "ht_kde_wilson": "#1976D2",  # Medium blue
        "ht_kde_calib_wilson": "#8E24AA",  # Medium purple
    }

    # Other methods - yellows and greens
    other_colors = {
        "ellipse_envelope_sweep": "#388E3C",  # Green
        "ellipse_envelope_quartic": "#689F38",  # Light green
        "logit_max_modulus": "#FBC02D",  # Yellow
        "bootstrap_percentile": "#AFB42B",  # Lime
        "bootstrap_bca": "#8BC34A",  # Light lime
    }

    # Check envelope methods
    if "envelope" in method_lower and "ellipse" not in method_lower:
        if method_lower in envelope_colors:
            return envelope_colors[method_lower]
        # Fallback for unknown envelope methods
        return "#FF6B6B"  # Generic red

    # Check HT methods
    if method_lower.startswith("ht_"):
        if method_lower in ht_colors:
            return ht_colors[method_lower]
        # Fallback for unknown HT methods
        return "#5C6BC0"  # Generic blue

    # Check other known methods
    if method_lower in other_colors:
        return other_colors[method_lower]

    # Fallback for any other unknown methods - cycle through greens/yellows
    hash_val = hash(method_lower) % 6
    fallback_colors = ["#4CAF50", "#CDDC39", "#8BC34A", "#FFC107", "#009688", "#00BCD4"]
    return fallback_colors[hash_val]


def get_method_colors_dict(methods: list[str]) -> dict[str, str]:
    """Get a dictionary mapping method names to colors.

    Args:
        methods: List of method names.

    Returns:
        Dictionary mapping method names to hex color codes.

    Examples:
        >>> get_method_colors_dict(["pointwise", "ks"])
        {'pointwise': '#000000', 'ks': '#5D4037'}
    """
    return {method: get_method_color(method) for method in methods}


# =============================================================================
# Pareto Frontier Plot
# =============================================================================


def plot_pareto_frontier(
    df: pd.DataFrame,
    mode: Literal["mean", "n_path"] = "mean",
    nominal_alpha: float = 0.05,
    figsize: tuple[float, float] = (8, 6),
    title: str | None = None,
    ax: Axes | None = None,
    show_legend: bool = True,
) -> Axes:
    """Plot Pareto frontier of mean_band_area vs coverage_rate.

    Visualizes the trade-off between band area (efficiency) and coverage rate
    (reliability) across different ROC confidence band methods. Can show either
    overall mean performance or trajectories across sample sizes.

    Args:
        df: DataFrame containing columns: model, mean_band_area, coverage_rate,
            n_total, nominal_alpha.
        mode: Visualization mode. "mean" shows overall mean for each method;
            "n_path" shows trajectory across n_total values connected by lines.
        nominal_alpha: Nominal alpha level for horizontal reference line.
        figsize: Figure size in inches (width, height).
        title: Plot title. If None, uses default based on mode.
        ax: Matplotlib axes to plot on. If None, creates new figure.

    Returns:
        The matplotlib axes object with the plot.

    Examples:
        >>> df = pd.read_feather("results.feather")
        >>> ax = plot_pareto_frontier(df, mode="mean")
        >>> plt.savefig("pareto.pdf")
        >>>
        >>> fig, ax = plt.subplots()
        >>> plot_pareto_frontier(df, mode="n_path", ax=ax)
    """
    set_publication_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    methods = df["model"].unique()
    colors = get_method_colors_dict(methods)

    if mode == "mean":
        # Calculate mean for each method
        grouped = (
            df.groupby("model")
            .agg({"mean_band_area": "mean", "coverage_rate": "mean"})
            .reset_index()
        )

        for _, row in grouped.iterrows():
            ax.scatter(
                row["mean_band_area"],
                row["coverage_rate"],
                c=colors[row["model"]],
                s=80,
                label=row["model"],
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )

    elif mode == "n_path":
        # Show path through n_total values
        for method in methods:
            method_df = df[df["model"] == method].copy()
            method_grouped = (
                method_df.groupby("n_total")
                .agg({"mean_band_area": "mean", "coverage_rate": "mean"})
                .reset_index()
                .sort_values("n_total")
            )

            ax.plot(
                method_grouped["mean_band_area"],
                method_grouped["coverage_rate"],
                c=colors[method],
                linewidth=1.5,
                alpha=0.7,
                zorder=2,
            )
            ax.scatter(
                method_grouped["mean_band_area"],
                method_grouped["coverage_rate"],
                c=colors[method],
                s=50,
                label=method,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )

    # Add nominal coverage line
    target_coverage = 1 - nominal_alpha
    ax.axhline(
        y=target_coverage,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Nominal ({target_coverage:.0%})",
        zorder=1,
    )

    ax.set_xlabel("Mean Band Area")
    ax.set_ylabel("Coverage Rate")
    if title is None:
        title = f"Pareto Frontier: Band Area vs Coverage ({mode} mode)"
    ax.set_title(title)

    # Create legend with smaller markers
    if show_legend:
        ax.legend(
            bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, markerscale=0.8
        )

    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    plt.tight_layout()
    return ax


# =============================================================================
# Violation Proximity Plot
# =============================================================================


def plot_violation_proximity(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (8, 6),
    title: str | None = None,
    ax: Axes | None = None,
    show_legend: bool = True,
) -> Axes:
    """Plot violation proximity: mean_band_width vs mean_max_violation.

    Visualizes the relationship between band width and maximum violation depth
    to assess how close confidence bands come to violating coverage guarantees.

    Args:
        df: DataFrame containing columns: model, mean_band_width, mean_max_violation.
        figsize: Figure size in inches (width, height).
        title: Plot title. If None, uses default.
        ax: Matplotlib axes to plot on. If None, creates new figure.

    Returns:
        The matplotlib axes object with the plot.

    Examples:
        >>> df = pd.read_feather("results.feather")
        >>> ax = plot_violation_proximity(df)
        >>> plt.savefig("violations.pdf")
    """
    set_publication_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    methods = df["model"].unique()
    colors = get_method_colors_dict(methods)

    # Calculate mean for each method
    grouped = (
        df.groupby("model")
        .agg({"mean_band_width": "mean", "mean_max_violation": "mean"})
        .reset_index()
    )

    for _, row in grouped.iterrows():
        ax.scatter(
            row["mean_band_width"],
            row["mean_max_violation"],
            c=colors[row["model"]],
            s=80,
            label=row["model"],
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    ax.set_xlabel("Mean Band Width")
    ax.set_ylabel("Mean Max Violation")
    if title is None:
        title = "Violation Proximity: Band Width vs Max Violation"
    ax.set_title(title)

    if show_legend:
        ax.legend(
            bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, markerscale=0.8
        )

    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    plt.tight_layout()
    return ax


# =============================================================================
# Coverage by n_total Plot
# =============================================================================


def plot_coverage_by_n_total(
    df: pd.DataFrame,
    nominal_alpha: float = 0.05,
    figsize: tuple[float, float] = (10, 6),
    title: str | None = None,
    ax: Axes | None = None,
    show_legend: bool = True,
) -> Axes:
    """Plot coverage rate by sample size (n_total) on log10 scale.

    Shows how coverage rates vary with sample size for different methods,
    helping identify which methods maintain nominal coverage across scales.

    Args:
        df: DataFrame containing columns: model, n_total, coverage_rate, nominal_alpha.
        nominal_alpha: Nominal alpha level for horizontal reference line.
        figsize: Figure size in inches (width, height).
        title: Plot title. If None, uses default.
        ax: Matplotlib axes to plot on. If None, creates new figure.

    Returns:
        The matplotlib axes object with the plot.

    Examples:
        >>> df = pd.read_feather("results.feather")
        >>> ax = plot_coverage_by_n_total(df, nominal_alpha=0.05)
        >>> plt.savefig("coverage_by_n.pdf")
    """
    set_publication_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    methods = df["model"].unique()
    colors = get_method_colors_dict(methods)

    for method in methods:
        method_df = df[df["model"] == method].copy()
        method_grouped = (
            method_df.groupby("n_total")
            .agg({"coverage_rate": "mean"})
            .reset_index()
            .sort_values("n_total")
        )

        ax.plot(
            method_grouped["n_total"],
            method_grouped["coverage_rate"],
            c=colors[method],
            linewidth=1.5,
            alpha=0.8,
            zorder=2,
        )
        ax.scatter(
            method_grouped["n_total"],
            method_grouped["coverage_rate"],
            c=colors[method],
            s=40,
            label=method,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    # Add nominal coverage line
    target_coverage = 1 - nominal_alpha
    ax.axhline(
        y=target_coverage,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Nominal ({target_coverage:.0%})",
        zorder=1,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Sample Size (n_total, log₁₀ scale)")
    ax.set_ylabel("Mean Coverage Rate")
    if title is None:
        title = "Coverage Rate by Sample Size"
    ax.set_title(title)

    if show_legend:
        ax.legend(
            bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, markerscale=0.8
        )

    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    plt.tight_layout()
    return ax


# =============================================================================
# Coverage by Prevalence Plot
# =============================================================================


def plot_coverage_by_prevalence(
    df: pd.DataFrame,
    nominal_alpha: float = 0.05,
    figsize: tuple[float, float] = (10, 6),
    title: str | None = None,
    ax: Axes | None = None,
    show_legend: bool = True,
) -> Axes:
    """Plot coverage rate by prevalence (categorical x-axis).

    Shows how coverage rates vary with class prevalence for different methods.
    Uses only n_totals shared by all prevalences to ensure fair comparison.

    Args:
        df: DataFrame containing columns: model, prevalence, n_total, coverage_rate,
            nominal_alpha.
        nominal_alpha: Nominal alpha level for horizontal reference line.
        figsize: Figure size in inches (width, height).
        title: Plot title. If None, uses default.
        ax: Matplotlib axes to plot on. If None, creates new figure.

    Returns:
        The matplotlib axes object with the plot.

    Examples:
        >>> df = pd.read_feather("results.feather")
        >>> ax = plot_coverage_by_prevalence(df)
        >>> plt.savefig("coverage_by_prevalence.pdf")
    """
    set_publication_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Find n_totals shared by all prevalences
    prevalences = df["prevalence"].unique()
    shared_n_totals = None
    for prev in prevalences:
        prev_n_totals = set(df[df["prevalence"] == prev]["n_total"].unique())
        if shared_n_totals is None:
            shared_n_totals = prev_n_totals
        else:
            shared_n_totals = shared_n_totals.intersection(prev_n_totals)

    # Filter to shared n_totals
    df_filtered = df[df["n_total"].isin(shared_n_totals)].copy()

    methods = df_filtered["model"].unique()
    colors = get_method_colors_dict(methods)

    # Sort prevalences for x-axis
    sorted_prevalences = sorted(df_filtered["prevalence"].unique())
    prev_to_idx = {p: i for i, p in enumerate(sorted_prevalences)}

    for method in methods:
        method_df = df_filtered[df_filtered["model"] == method].copy()
        method_grouped = (
            method_df.groupby("prevalence").agg({"coverage_rate": "mean"}).reset_index()
        )
        method_grouped = method_grouped.sort_values("prevalence")

        x_positions = [prev_to_idx[p] for p in method_grouped["prevalence"]]

        ax.plot(
            x_positions,
            method_grouped["coverage_rate"],
            c=colors[method],
            linewidth=1.5,
            alpha=0.8,
            zorder=2,
        )
        ax.scatter(
            x_positions,
            method_grouped["coverage_rate"],
            c=colors[method],
            s=40,
            label=method,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    # Add nominal coverage line
    target_coverage = 1 - nominal_alpha
    ax.axhline(
        y=target_coverage,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Nominal ({target_coverage:.0%})",
        zorder=1,
    )

    ax.set_xticks(range(len(sorted_prevalences)))
    ax.set_xticklabels(
        [f"{p:.1%}" if p < 1 else f"{p:.0%}" for p in sorted_prevalences]
    )
    ax.set_xlabel("Prevalence")
    ax.set_ylabel("Mean Coverage Rate")
    if title is None:
        title = "Coverage Rate by Prevalence"
    ax.set_title(title)

    if show_legend:
        ax.legend(
            bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, markerscale=0.8
        )

    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, axis="y")

    plt.tight_layout()
    return ax


# =============================================================================
# Violation Direction Plot
# =============================================================================


def plot_violation_direction(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (10, 6),
    title: str | None = None,
    ax: Axes | None = None,
    show_legend: bool = True,
) -> Axes:
    """Plot violation direction as colored vertical barplot.

    Shows the ratio of above vs below violations for each method, helping
    identify systematic biases in confidence band construction. A ratio of 0.5
    indicates balanced violations; values far from 0.5 indicate asymmetric
    behavior.

    Args:
        df: DataFrame containing columns: model, violation_rate_above, violation_rate_below.
        figsize: Figure size in inches (width, height).
        title: Plot title. If None, uses default.
        ax: Matplotlib axes to plot on. If None, creates new figure.

    Returns:
        The matplotlib axes object with the plot.

    Examples:
        >>> df = pd.read_feather("results.feather")
        >>> ax = plot_violation_direction(df)
        >>> plt.savefig("violation_direction.pdf")
    """
    set_publication_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Calculate mean violation rates and ratio for each method
    grouped = (
        df.groupby("model")
        .agg({"violation_rate_above": "mean", "violation_rate_below": "mean"})
        .reset_index()
    )

    # Calculate ratio (above / total violations)
    grouped["total_violations"] = (
        grouped["violation_rate_above"] + grouped["violation_rate_below"]
    )
    grouped["violation_ratio"] = np.where(
        grouped["total_violations"] > 0,
        grouped["violation_rate_above"] / grouped["total_violations"],
        0.5,  # Default to 0.5 if no violations
    )

    # Sort by violation ratio (highest to lowest)
    grouped = grouped.sort_values("violation_ratio", ascending=False).reset_index(
        drop=True
    )

    methods = grouped["model"].tolist()
    colors = get_method_colors_dict(methods)

    x_positions = range(len(methods))
    bar_colors = [colors[m] for m in methods]

    bars = ax.bar(
        x_positions,
        grouped["violation_ratio"],
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    # Add reference line at 0.5 (equal violations)
    ax.axhline(
        y=0.5,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Equal (0.5)",
        zorder=1,
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Method")
    ax.set_ylabel("Violation Ratio (Above / Total)")
    if title is None:
        title = "Violation Direction by Method"
    ax.set_title(title)

    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, axis="y")

    # Add text annotations for extreme values
    for i, (ratio, method) in enumerate(zip(grouped["violation_ratio"], methods)):
        if ratio > 0.8 or ratio < 0.2:
            ax.annotate(
                f"{ratio:.2f}",
                (i, ratio),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=7,
            )

    plt.tight_layout()
    return ax
