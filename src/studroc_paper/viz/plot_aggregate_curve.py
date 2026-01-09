"""Visualization functions for ROC confidence band method analysis.

Provides publication-quality plots for analyzing the aggregate behavior of
different ROC confidence band methods across synthetic datasets. This module
focuses on regionwise analysis, where the false positive rate (FPR) range
[0, 100] is divided into regions (e.g., 0-10%, 10-30%) and coverage
statistics are computed separately for each region, called the "curve" data
frame.

Key functionality:
    - Coverage by region plots showing violation rates across FPR regions
    - Pareto frontier plots showing the tradeoff between region width and
      violation rate
    - Publication-ready styling for matplotlib/seaborn figures
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from .plot_aggregate import get_method_colors_dict

# =============================================================================
# Style Configuration
# =============================================================================


def set_publication_style() -> None:
    """Set matplotlib/seaborn style for publication-quality figures.

    Configures matplotlib rcParams and seaborn settings for consistent,
    publication-ready plots with appropriate font sizes, line widths,
    and color palettes.

    Examples:
        >>> set_publication_style()
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 9])
        >>> plt.savefig("publication_plot.pdf")
    """
    plt.rcParams.update(
        {
            # Font settings
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            # Figure settings
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            # Axes settings
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.axisbelow": True,
            # Tick settings
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.direction": "out",
            "ytick.direction": "out",
            # Line settings
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            # Legend settings
            "legend.frameon": False,
            "legend.borderpad": 0.4,
            "legend.handletextpad": 0.5,
        }
    )
    sns.set_palette("colorblind")


# Initialize style on module import
set_publication_style()


# =============================================================================
# Helper Functions
# =============================================================================


def _parse_fpr_region(region_str: str) -> tuple[int, int]:
    """Parse FPR region string into numeric range bounds.

    Args:
        region_str: String in format 'start-end' (e.g., '0-10', '30-50').

    Returns:
        Tuple of (start, end) as integers representing the region bounds.

    Examples:
        >>> _parse_fpr_region("0-10")
        (0, 10)
        >>> _parse_fpr_region("70-90")
        (70, 90)
    """
    start, end = region_str.split("-")
    return int(start), int(end)


def _get_region_midpoint(region_str: str) -> float:
    """Get the midpoint of an FPR region for plotting.

    Args:
        region_str: String in format 'start-end' (e.g., '0-10', '30-50').

    Returns:
        Midpoint of the region as a float.

    Examples:
        >>> _get_region_midpoint("0-10")
        5.0
        >>> _get_region_midpoint("70-90")
        80.0
    """
    start, end = _parse_fpr_region(region_str)
    return (start + end) / 2


def _get_region_order(regions: list[str]) -> list[str]:
    """Sort FPR regions in ascending order by start value.

    Args:
        regions: List of region strings (e.g., ['30-50', '0-10', '10-30']).

    Returns:
        List of region strings sorted by their start values.

    Examples:
        >>> _get_region_order(["30-50", "0-10", "10-30"])
        ['0-10', '10-30', '30-50']
    """
    return sorted(regions, key=lambda r: _parse_fpr_region(r)[0])


def get_fpr_region_markers() -> dict[str, str]:
    """Get marker shapes for different FPR regions.

    Returns:
        Dictionary mapping region strings to matplotlib marker codes.
        Provides distinct shapes for visual differentiation in plots.

    Examples:
        >>> markers = get_fpr_region_markers()
        >>> markers["0-10"]
        'o'
        >>> markers["50-70"]
        'D'
    """
    return {
        "0-10": "o",
        "10-30": "s",
        "30-50": "^",
        "50-70": "D",
        "70-90": "v",
        "90-100": "P",
    }


# =============================================================================
# Coverage by Region Plot
# =============================================================================


def plot_coverage_by_region(
    df: pd.DataFrame,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (6, 4),
    nominal_alpha: float | None = None,
    title: str | None = None,
    xlabel: str = "FPR Region",
    ylabel: str = "Region Violation Rate",
    legend_loc: str = "best",
    legend_ncol: int = 1,
    marker_size: float = 7,
    line_alpha: float = 0.8,
    show_legend: bool = True,
) -> Axes:
    """Plot coverage (violation rate) by FPR region for each method.

    Creates a line plot showing how violation rates vary across FPR regions
    for different confidence band methods. Each method is represented by a
    colored line with markers, with regions ordered left-to-right by their
    start values.

    Args:
        df: DataFrame containing columns 'model', 'fpr_region', and
            'region_violation_rate'.
        ax: Matplotlib axes to plot on. If None, creates new figure. Defaults
            to None.
        figsize: Figure size as (width, height) in inches. Defaults to (6, 4).
        nominal_alpha: If provided, draws a horizontal reference line at this
            value (e.g., 0.05 for 95% confidence level). Defaults to None.
        title: Plot title. If None, no title is added. Defaults to None.
        xlabel: Label for x-axis. Defaults to "FPR Region".
        ylabel: Label for y-axis. Defaults to "Region Violation Rate".
        legend_loc: Legend location string. Defaults to "best".
        legend_ncol: Number of columns in legend. Defaults to 1.
        marker_size: Size of markers. Defaults to 7.
        line_alpha: Alpha transparency for lines. Defaults to 0.8.

    Returns:
        The matplotlib Axes object containing the plot.

    Examples:
        >>> import pandas as pd
        >>> import matplotlib.pyplot as plt
        >>> df = pd.DataFrame(
        ...     {
        ...         "model": ["Method A"] * 3 + ["Method B"] * 3,
        ...         "fpr_region": ["0-10", "10-30", "30-50"] * 2,
        ...         "region_violation_rate": [0.03, 0.04, 0.06, 0.05, 0.05, 0.07],
        ...     }
        ... )
        >>> ax = plot_coverage_by_region(df, nominal_alpha=0.05)
        >>> plt.savefig("coverage_by_region.pdf")
        >>> plt.close()
    """
    set_publication_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Aggregate by taking mean if multiple values exist for same method/region
    df_agg = df.groupby(["model", "fpr_region"], as_index=False)[
        "region_violation_rate"
    ].mean()

    # Get unique methods and regions
    methods = df_agg["model"].unique().tolist()
    regions = _get_region_order(df_agg["fpr_region"].unique().tolist())

    # Get colors for methods
    color_dict = get_method_colors_dict(methods)

    # Create region midpoints for x-axis
    region_midpoints = {r: _get_region_midpoint(r) for r in regions}

    # Plot each method
    for method in methods:
        method_df = df_agg[df_agg["model"] == method].copy()

        # Sort by region order
        method_df["_region_order"] = method_df["fpr_region"].apply(
            lambda r: _parse_fpr_region(r)[0]
        )
        method_df = method_df.sort_values("_region_order")

        x_vals = [region_midpoints[r] for r in method_df["fpr_region"]]
        y_vals = method_df["region_violation_rate"].values

        ax.plot(
            x_vals,
            y_vals,
            marker="o",
            markersize=marker_size,
            color=color_dict[method],
            label=method,
            alpha=line_alpha,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )

    # Add nominal alpha reference line if provided
    if nominal_alpha is not None:
        ax.axhline(
            nominal_alpha,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label=f"Nominal α = {nominal_alpha}",
            zorder=0,
        )

    # Configure x-axis with region labels
    ax.set_xticks([region_midpoints[r] for r in regions])
    ax.set_xticklabels(regions, rotation=45, ha="right")

    # Labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Legend
    if show_legend:
        ax.legend(loc=legend_loc, ncol=legend_ncol)

    # Ensure y-axis starts at 0
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    return ax


# =============================================================================
# Regionwise Pareto Frontier Plot
# =============================================================================


def plot_regionwise_pareto_frontier(
    df: pd.DataFrame,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (6, 5),
    title: str | None = None,
    xlabel: str = "Region Width",
    ylabel: str = "Region Violation Rate",
    legend_loc: str = "best",
    legend_ncol: int = 2,
    marker_size: float = 8,
    line_alpha: float = 0.6,
    show_region_legend: bool = True,
    show_legend: bool = True,
) -> Axes:
    """Plot regionwise Pareto frontier: region_width vs region_violation_rate.

    Creates a scatter plot showing the tradeoff between confidence band width
    and violation rate across different FPR regions. Each method is represented
    by a colored line connecting markers of different shapes (one per region).
    Ideally, methods should lie in the lower-left (narrow bands with low
    violation rates).

    Args:
        df: DataFrame containing columns 'model', 'fpr_region', 'region_width',
            and 'region_violation_rate'.
        ax: Matplotlib axes to plot on. If None, creates new figure. Defaults
            to None.
        figsize: Figure size as (width, height) in inches. Defaults to (6, 5).
        title: Plot title. If None, no title is added. Defaults to None.
        xlabel: Label for x-axis. Defaults to "Region Width".
        ylabel: Label for y-axis. Defaults to "Region Violation Rate".
        legend_loc: Legend location string. Defaults to "best".
        legend_ncol: Number of columns in legend. Defaults to 2.
        marker_size: Size of markers. Defaults to 8.
        line_alpha: Alpha transparency for connecting lines. Defaults to 0.6.
        show_region_legend: Whether to show a legend mapping marker shapes to
            FPR regions. Defaults to True.

    Returns:
        The matplotlib Axes object containing the plot.

    Examples:
        >>> import pandas as pd
        >>> import matplotlib.pyplot as plt
        >>> df = pd.DataFrame(
        ...     {
        ...         "model": ["Method A"] * 3 + ["Method B"] * 3,
        ...         "fpr_region": ["0-10", "10-30", "30-50"] * 2,
        ...         "region_width": [0.12, 0.15, 0.18, 0.10, 0.13, 0.16],
        ...         "region_violation_rate": [0.03, 0.04, 0.06, 0.05, 0.05, 0.07],
        ...     }
        ... )
        >>> ax = plot_regionwise_pareto_frontier(df)
        >>> plt.savefig("pareto_frontier.pdf")
        >>> plt.close()
    """
    set_publication_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Aggregate by taking mean if multiple values exist for same method/region
    df_agg = df.groupby(["model", "fpr_region"], as_index=False)[
        ["region_width", "region_violation_rate"]
    ].mean()

    # Get unique methods and regions
    methods = df_agg["model"].unique().tolist()
    regions = _get_region_order(df_agg["fpr_region"].unique().tolist())

    # Get colors and markers
    color_dict = get_method_colors_dict(methods)
    marker_dict = get_fpr_region_markers()

    # Plot each method
    for method in methods:
        method_df = df_agg[df_agg["model"] == method].copy()

        # Sort by region order
        method_df["_region_order"] = method_df["fpr_region"].apply(
            lambda r: _parse_fpr_region(r)[0]
        )
        method_df = method_df.sort_values("_region_order")

        x_vals = method_df["region_width"].values
        y_vals = method_df["region_violation_rate"].values
        region_labels = method_df["fpr_region"].values

        # Draw connecting line first
        ax.plot(
            x_vals,
            y_vals,
            color=color_dict[method],
            alpha=line_alpha,
            linewidth=1.2,
            zorder=1,
        )

        # Draw markers for each region
        for x, y, region in zip(x_vals, y_vals, region_labels):
            marker = marker_dict.get(region, "o")
            ax.scatter(
                x,
                y,
                marker=marker,
                s=marker_size**2,
                color=color_dict[method],
                edgecolor="white",
                linewidth=0.5,
                zorder=2,
            )

    # Create method legend handles
    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=color_dict[m],
            linestyle="-",
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=m,
        )
        for m in methods
    ]

    # Create region shape legend handles
    region_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_dict.get(r, "o"),
            color="gray",
            linestyle="None",
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=f"FPR {r}%",
        )
        for r in regions
    ]

    # Add legends
    if show_legend:
        if show_region_legend:
            # Create combined legend with both methods and regions
            all_handles = (
                method_handles + [Line2D([], [], linestyle="None")] + region_handles
            )
            all_labels = (
                [m for m in methods]
                + [""]  # Spacer
                + [f"FPR {r}%" for r in regions]
            )
            ax.legend(
                handles=all_handles,
                labels=all_labels,
                loc=legend_loc,
                ncol=legend_ncol,
                columnspacing=1.0,
                handletextpad=0.5,
            )
        else:
            ax.legend(
                handles=method_handles, loc=legend_loc, ncol=legend_ncol, title="Method"
            )

    # Labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Ensure axes start at 0
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    return ax
