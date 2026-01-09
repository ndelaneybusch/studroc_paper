"""Visualization functions for ROC confidence band simulation results.

This module provides publication-quality visualization functions for analyzing
confidence interval method performance in ROC curve simulations. It includes:

- Heatmaps for visualizing coverage rates across data properties
- GAM-smoothed line plots for coverage trends
- Violation location analysis across FPR regions
- Perceptually uniform colormaps using HCL color space
- Publication-ready styling and formatting utilities

The module automatically applies publication-quality styling on import and
provides consistent color schemes for different CI methods.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from colorspacious import cspace_convert
from matplotlib.axes import Axes
from pygam import LogisticGAM, s

from .plot_aggregate import get_method_colors_dict

# =============================================================================
# Style Configuration
# =============================================================================


def set_publication_style() -> None:
    """Set matplotlib/seaborn style for publication-quality figures.

    Configures matplotlib and seaborn with settings optimized for
    academic publications, including font sizes, line widths, and
    clean aesthetics with removed top and right spines.

    Examples:
        >>> set_publication_style()
        >>> fig, ax = plt.subplots()
        >>> # Figure will use publication-quality styling
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
            "legend.title_fontsize": 10,
            # Line and marker settings
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            # Axes settings
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.axisbelow": True,
            # Grid settings
            "grid.linewidth": 0.5,
            "grid.alpha": 0.4,
            # Figure settings
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            # Legend settings
            "legend.frameon": False,
            "legend.borderpad": 0.4,
        }
    )
    sns.set_palette("deep")


# Initialize style on module import
set_publication_style()


def create_hcl_colormap(
    h_range: tuple[float, float] = (260, 10),
    c: float = 70,
    l_range: tuple[float, float] = (30, 90),
    n_colors: int = 256,
    name: str = "hcl_sequential",
) -> mcolors.LinearSegmentedColormap:
    """Create a perceptually uniform colormap using HCL color space.

    Generates a sequential colormap with perceptually uniform color
    transitions using the CAM02-UCS color space. Colors are specified
    in JCh (similar to HCL) and converted to sRGB.

    Args:
        h_range: Hue range (start, end) in degrees [0, 360). Defaults to (260, 10).
        c: Chroma (saturation) value. Defaults to 70.
        l_range: Luminance range (start, end) in [0, 100]. Defaults to (30, 90).
        n_colors: Number of colors in the colormap. Defaults to 256.
        name: Name for the colormap. Defaults to "hcl_sequential".

    Returns:
        A matplotlib LinearSegmentedColormap with perceptually uniform
        color transitions.

    Examples:
        >>> cmap = create_hcl_colormap()
        >>> cmap.N
        256
        >>> cmap = create_hcl_colormap(h_range=(0, 120), c=50, n_colors=128)
        >>> cmap.name
        'hcl_sequential'
    """
    # Generate HCL colors
    h_vals = np.linspace(h_range[0], h_range[1], n_colors)
    # Handle hue wrapping
    h_vals = h_vals % 360
    l_vals = np.linspace(l_range[0], l_range[1], n_colors)

    # Convert JCh (similar to HCL) to sRGB
    jch_colors = np.column_stack(
        [
            l_vals,  # J (lightness)
            np.full(n_colors, c),  # C (chroma)
            h_vals,  # h (hue)
        ]
    )

    # Convert to sRGB via CAM02-UCS
    rgb_colors = cspace_convert(jch_colors, "JCh", "sRGB1")

    # Clip to valid range
    rgb_colors = np.clip(rgb_colors, 0, 1)

    return mcolors.LinearSegmentedColormap.from_list(name, rgb_colors)


def create_diverging_hcl_colormap(
    h_neg: float = 260,
    h_pos: float = 10,
    c: float = 70,
    l_range: tuple[float, float] = (30, 95),
    n_colors: int = 256,
    name: str = "hcl_diverging",
) -> mcolors.LinearSegmentedColormap:
    """Create a perceptually uniform diverging colormap.

    Generates a diverging colormap with two distinct hues that transition
    through a neutral center. Uses CAM02-UCS color space for perceptual
    uniformity. The colormap goes from dark to light on one side and
    light to dark on the other.

    Args:
        h_neg: Hue for negative/low end in degrees [0, 360). Defaults to 260.
        h_pos: Hue for positive/high end in degrees [0, 360). Defaults to 10.
        c: Maximum chroma (saturation) value at endpoints. Defaults to 70.
        l_range: Luminance range (dark, light) in [0, 100]. Defaults to (30, 95).
        n_colors: Total number of colors in the colormap. Defaults to 256.
        name: Name for the colormap. Defaults to "hcl_diverging".

    Returns:
        A diverging LinearSegmentedColormap with perceptually uniform
        transitions between the two hues.

    Examples:
        >>> cmap = create_diverging_hcl_colormap()
        >>> cmap.N
        256
        >>> cmap = create_diverging_hcl_colormap(h_neg=240, h_pos=20, n_colors=128)
        >>> cmap.name
        'hcl_diverging'
    """
    half = n_colors // 2

    # Negative side (dark to light)
    l_neg = np.linspace(l_range[0], l_range[1], half)
    c_neg = np.linspace(c, 0, half)

    # Positive side (light to dark)
    l_pos = np.linspace(l_range[1], l_range[0], n_colors - half)
    c_pos = np.linspace(0, c, n_colors - half)

    jch_neg = np.column_stack([l_neg, c_neg, np.full(half, h_neg)])
    jch_pos = np.column_stack([l_pos, c_pos, np.full(n_colors - half, h_pos)])

    jch_colors = np.vstack([jch_neg, jch_pos])
    rgb_colors = cspace_convert(jch_colors, "JCh", "sRGB1")
    rgb_colors = np.clip(rgb_colors, 0, 1)

    return mcolors.LinearSegmentedColormap.from_list(name, rgb_colors)


# =============================================================================
# Visualization Functions
# =============================================================================


def plot_violation_heatmap(
    df: pd.DataFrame,
    x_col: str = "lhs_auc",
    y_col: str = "n_total",
    value_col: str = "covers_entirely",
    method: str | None = None,
    n_bins_x: int = 10,
    n_bins_y: int = 10,
    cmap: str | mcolors.Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    ax: Axes | None = None,
    cbar_label: str | None = None,
    title: str | None = None,
) -> Axes:
    """Create a heatmap of mean values binned by two variables.

    Bins data by two continuous variables and displays the mean of a third
    variable as a heatmap. Useful for visualizing how coverage rates or other
    metrics vary across combinations of data properties.

    Args:
        df: DataFrame containing simulation results with columns for binning and aggregation.
        x_col: Column name for x-axis binning. Defaults to "lhs_auc".
        y_col: Column name for y-axis binning. Defaults to "n_total".
        value_col: Column name to aggregate (mean) for heatmap values. Defaults to "covers_entirely".
        method: If provided, filter to this method only. Defaults to None (use all methods).
        n_bins_x: Number of bins for x-axis. Defaults to 10.
        n_bins_y: Number of bins for y-axis. Defaults to 10.
        cmap: Colormap name or object. Defaults to None (uses HCL sequential).
        vmin: Minimum value for color scale. Defaults to None (auto).
        vmax: Maximum value for color scale. Defaults to None (auto).
        ax: Matplotlib Axes object to plot on. Defaults to None (creates new figure).
        cbar_label: Label for colorbar. Defaults to None (uses formatted column name).
        title: Plot title. Defaults to None (auto-generates from method name).

    Returns:
        The matplotlib Axes object containing the heatmap.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "lhs_auc": [0.7, 0.8, 0.9],
        ...         "n_total": [100, 200, 300],
        ...         "covers_entirely": [0.9, 0.95, 0.92],
        ...         "method": ["pointwise", "pointwise", "pointwise"],
        ...     }
        ... )
        >>> ax = plot_violation_heatmap(df, method="pointwise")
        >>> ax.get_xlabel()
        'AUC'
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    # Filter by method if specified
    plot_df = df.copy()
    if method is not None:
        plot_df = plot_df[plot_df["method"] == method]

    # Create bins
    x_bins = pd.cut(plot_df[x_col], bins=n_bins_x)
    y_bins = pd.cut(plot_df[y_col], bins=n_bins_y)

    # Aggregate
    agg_df = (
        plot_df.groupby([y_bins, x_bins], observed=True)[value_col].mean().unstack()
    )

    # Format bin labels
    x_labels = [f"{interval.mid:.2f}" for interval in agg_df.columns]
    y_labels = [
        f"{interval.mid:.0f}" if interval.right > 10 else f"{interval.mid:.2f}"
        for interval in agg_df.index
    ]

    # Default colormap
    if cmap is None:
        cmap = create_hcl_colormap(h_range=(260, 40), c=65, l_range=(25, 95))

    # Set default vmin/vmax for coverage
    if value_col == "covers_entirely":
        vmin = vmin or 0.0
        vmax = vmax or 1.0

    # Create heatmap
    sns.heatmap(
        agg_df,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cbar_kws={
            "label": cbar_label or _format_column_label(value_col),
            "shrink": 0.8,
        },
        linewidths=0.5,
        linecolor="white",
    )

    # Labels and title
    ax.set_xlabel(_format_column_label(x_col))
    ax.set_ylabel(_format_column_label(y_col))

    if title:
        ax.set_title(title, fontweight="medium", pad=10)
    elif method:
        ax.set_title(
            f"Coverage Rate: {_format_method_name(method)}", fontweight="medium", pad=10
        )

    # Rotate x-axis labels
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    return ax


def plot_data_property_lines(
    df: pd.DataFrame,
    x_col: str = "lhs_auc",
    y_col: str = "covers_entirely",
    methods: list[str] | None = None,
    n_splines: int = 10,
    confidence_level: float | None = None,
    show_points: bool = False,
    point_alpha: float = 0.1,
    ax: Axes | None = None,
    title: str | None = None,
    show_legend: bool = True,
    legend_loc: str = "best",
    nominal_coverage: float | None = None,
) -> Axes:
    """Create a line plot with GAM-smoothed coverage by a data property.

    Fits a binomial GAM (logit link) to binary coverage outcomes for each
    method and plots the smoothed probability curves. This visualization
    reveals how coverage rates vary as a function of continuous data
    properties like AUC or sample size.

    Args:
        df: DataFrame containing simulation results with binary outcomes.
        x_col: Column name for x-axis (continuous predictor). Defaults to "lhs_auc".
        y_col: Column name for y-axis (binary outcome). Defaults to "covers_entirely".
        methods: List of method names to include. Defaults to None (all methods).
        n_splines: Number of splines for GAM smoothing. Defaults to 10.
        confidence_level: If provided, filter to this confidence level. Defaults to None.
        show_points: Whether to show raw data points with jitter. Defaults to False.
        point_alpha: Transparency for raw data points. Defaults to 0.1.
        ax: Matplotlib Axes object to plot on. Defaults to None (creates new figure).
        title: Plot title. Defaults to None (no title).
        show_legend: Whether to show legend. Defaults to True.
        legend_loc: Legend location string. Defaults to "best".
        nominal_coverage: If provided, draw horizontal reference line at this level. Defaults to None.

    Returns:
        The matplotlib Axes object containing the line plot.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "lhs_auc": [0.7, 0.75, 0.8, 0.85, 0.9] * 20,
        ...         "covers_entirely": [1, 1, 0, 1, 1] * 20,
        ...         "method": ["pointwise"] * 100,
        ...         "confidence_level": [0.95] * 100,
        ...     }
        ... )
        >>> ax = plot_data_property_lines(
        ...     df, methods=["pointwise"], nominal_coverage=0.95
        ... )
        >>> ax.get_ylabel()
        'Coverage Rate'
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    plot_df = df.copy()

    # Filter by confidence level if specified
    if confidence_level is not None:
        plot_df = plot_df[plot_df["confidence_level"] == confidence_level]

    # Get methods
    if methods is None:
        methods = sorted(plot_df["method"].unique())

    # Get colors
    colors = get_method_colors_dict(methods)

    # Create smooth x values for prediction
    x_min, x_max = plot_df[x_col].min(), plot_df[x_col].max()
    x_smooth = np.linspace(x_min, x_max, 200)

    for method in methods:
        method_df = plot_df[plot_df["method"] == method]

        if len(method_df) < 20:
            continue

        X = method_df[x_col].values.reshape(-1, 1)
        y = method_df[y_col].astype(int).values

        # Fit GAM with logit link
        try:
            gam = LogisticGAM(s(0, n_splines=n_splines)).fit(X, y)
            y_pred = gam.predict_proba(x_smooth.reshape(-1, 1))

            # Plot smooth line
            ax.plot(
                x_smooth,
                y_pred,
                color=colors[method],
                label=_format_method_name(method),
                linewidth=1.8,
            )

            # Optionally show raw points
            if show_points:
                # Jitter y for visibility
                y_jitter = y + np.random.normal(0, 0.02, len(y))
                ax.scatter(
                    X.ravel(),
                    y_jitter,
                    color=colors[method],
                    alpha=point_alpha,
                    s=10,
                    linewidths=0,
                )
        except Exception:
            # Fallback: plot binned means
            bins = pd.cut(method_df[x_col], bins=20)
            binned = method_df.groupby(bins, observed=True)[y_col].mean()
            bin_centers = [interval.mid for interval in binned.index]
            ax.plot(
                bin_centers,
                binned.values,
                color=colors[method],
                label=_format_method_name(method),
                linewidth=1.8,
                linestyle="--",
            )

    # Reference line for nominal coverage
    if nominal_coverage is not None:
        ax.axhline(
            nominal_coverage,
            color="#666666",
            linestyle=":",
            linewidth=1.2,
            label=f"Nominal ({nominal_coverage:.0%})",
            zorder=0,
        )

    # Labels and formatting
    ax.set_xlabel(_format_column_label(x_col))
    ax.set_ylabel(_format_column_label(y_col))
    ax.set_ylim(-0.02, 1.02)

    if title:
        ax.set_title(title, fontweight="medium", pad=10)

    if show_legend:
        ax.legend(loc=legend_loc, frameon=True, fancybox=False, edgecolor="#cccccc")

    # Light grid
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)

    return ax


def plot_violation_location_gradient(
    df: pd.DataFrame,
    bin_col: str = "lhs_auc",
    n_bins: int | None = 8,
    method: str | None = None,
    cmap: str = "viridis",
    ax: Axes | None = None,
    title: str | None = None,
    show_legend: bool = True,
) -> Axes:
    """Plot violation rates across FPR regions, colored by a binned variable.

    Visualizes how violation rates vary across different false positive rate
    (FPR) regions of the ROC curve. Each line represents a bin of the color
    variable, allowing comparison of violation patterns across data properties.

    Args:
        df: DataFrame containing simulation results with violation_* columns.
        bin_col: Column name to bin for color gradient. Defaults to "lhs_auc".
        n_bins: Number of bins for color gradient. If None, uses unique values
            sorted alphanumerically. Defaults to 8.
        method: If provided, filter to this method only. Defaults to None.
        cmap: Colormap name for gradient. Defaults to "viridis".
        ax: Matplotlib Axes object to plot on. Defaults to None (creates new figure).
        title: Plot title. Defaults to None (auto-generates from method name).
        show_legend: Whether to show legend. Defaults to True.

    Returns:
        The matplotlib Axes object containing the line plot.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "lhs_auc": [0.7, 0.8, 0.9] * 10,
        ...         "method": ["pointwise"] * 30,
        ...         "violation_0-10": [0.05, 0.03, 0.02] * 10,
        ...         "violation_10-20": [0.04, 0.02, 0.01] * 10,
        ...     }
        ... )
        >>> ax = plot_violation_location_gradient(df, method="pointwise", n_bins=3)
        >>> ax.get_xlabel()
        'FPR Region'
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Filter by method if specified
    plot_df = df.copy()
    if method is not None:
        plot_df = plot_df[plot_df["method"] == method]

    # Identify violation location columns
    violation_cols = [
        col for col in plot_df.columns if col.startswith("violation_") and "-" in col
    ]

    # Sort by FPR range
    def sort_key(col: str) -> int:
        try:
            return int(col.split("_")[1].split("-")[0])
        except (IndexError, ValueError):
            return 999

    violation_cols = sorted(violation_cols, key=sort_key)

    # Create x-axis positions and labels
    x_positions = np.arange(len(violation_cols))
    x_labels = [
        col.replace("violation_", "").replace("-", "–") + "%" for col in violation_cols
    ]

    # Create bins or use unique values for color variable
    if n_bins is None:
        # Use unique values sorted alphanumerically
        unique_values = sorted(
            plot_df[bin_col].unique(),
            key=lambda x: (
                [int(c) if c.isdigit() else c.lower() for c in str(x).split()]
                if isinstance(x, str)
                else x
            ),
        )
        n_groups = len(unique_values)
        value_to_idx = {val: idx for idx, val in enumerate(unique_values)}
        plot_df["_bin"] = plot_df[bin_col].map(value_to_idx)
        group_labels = [str(val) for val in unique_values]
    else:
        # Create bins using pd.cut
        plot_df["_bin"] = pd.cut(plot_df[bin_col], bins=n_bins, labels=False)
        bin_edges = pd.cut(plot_df[bin_col], bins=n_bins).cat.categories
        n_groups = n_bins
        group_labels = [
            f"{interval.left:.2f}–{interval.right:.2f}" for interval in bin_edges
        ]

    # Get colormap
    cmap_obj = plt.get_cmap(cmap)
    colors = [
        cmap_obj(i / (n_groups - 1)) if n_groups > 1 else cmap_obj(0.5)
        for i in range(n_groups)
    ]

    # Plot lines for each group
    for group_idx in range(n_groups):
        group_df = plot_df[plot_df["_bin"] == group_idx]

        if len(group_df) == 0:
            continue

        # Calculate mean violation rate for each region
        means = [group_df[col].mean() for col in violation_cols]

        ax.plot(
            x_positions,
            means,
            color=colors[group_idx],
            linewidth=1.8,
            marker="o",
            markersize=5,
            label=group_labels[group_idx],
        )

    # Formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_xlabel("FPR Region")
    ax.set_ylabel("Violation Rate")
    ax.set_ylim(bottom=0)

    if title:
        ax.set_title(title, fontweight="medium", pad=10)
    elif method:
        ax.set_title(
            f"Violation Location: {_format_method_name(method)}",
            fontweight="medium",
            pad=10,
        )

    if show_legend:
        # Create a more compact legend or colorbar
        legend = ax.legend(
            title=_format_column_label(bin_col),
            loc="upper right",
            fontsize=8,
            title_fontsize=9,
            frameon=True,
            fancybox=False,
            edgecolor="#cccccc",
            ncol=2 if n_groups > 5 else 1,
        )

    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)

    return ax


# =============================================================================
# Helper Functions
# =============================================================================


def _format_column_label(col: str) -> str:
    """Format column name for display in plots.

    Converts internal column names to human-readable labels with proper
    formatting, mathematical symbols, and capitalization.

    Args:
        col: Column name to format.

    Returns:
        Formatted label string suitable for plot axes and legends.

    Examples:
        >>> _format_column_label("lhs_auc")
        'AUC'
        >>> _format_column_label("covers_entirely")
        'Coverage Rate'
        >>> _format_column_label("unknown_column")
        'Unknown Column'
    """
    label_map = {
        "lhs_auc": "AUC",
        "lhs_sigma": "σ",
        "lhs_sigma_ratio": "σ Ratio (σ₊/σ₋)",
        "lhs_alpha": "α",
        "n_total": "Sample Size (n)",
        "n_pos": "n Positive",
        "n_neg": "n Negative",
        "prevalence": "Prevalence",
        "covers_entirely": "Coverage Rate",
        "violation_above": "Violation Above",
        "violation_below": "Violation Below",
        "max_violation_above": "Max Violation Above",
        "max_violation_below": "Max Violation Below",
        "band_area": "Band Area",
        "mean_band_width": "Mean Band Width",
        "empirical_auc": "Empirical AUC",
        "confidence_level": "Confidence Level",
    }
    return label_map.get(col, col.replace("_", " ").title())


def _format_method_name(method: str) -> str:
    """Format method name for display in plots.

    Converts internal method names to readable labels with proper
    capitalization, abbreviations, and spacing.

    Args:
        method: Method name to format (e.g., "envelope_standard", "HT_kde_calib").

    Returns:
        Formatted method name suitable for plot legends and titles.

    Examples:
        >>> _format_method_name("envelope_standard")
        'Envelope Standard'
        >>> _format_method_name("HT_kde_calib")
        'HT KDE calibrated'
        >>> _format_method_name("envelope_wilson_logit")
        'Envelope Wilson (logit)'
    """
    parts = method.split("_")
    formatted_parts = []

    for part in parts:
        if part.upper() in ("HT", "KDE", "KS"):
            formatted_parts.append(part.upper())
        elif part == "logit":
            formatted_parts.append("(logit)")
        elif part == "calib":
            formatted_parts.append("calibrated")
        elif part == "symmetric":
            formatted_parts.append("sym.")
        else:
            formatted_parts.append(part.capitalize())

    return " ".join(formatted_parts)
