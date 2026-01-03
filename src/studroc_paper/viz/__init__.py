"""Visualization utilities for ROC confidence band diagnostics."""

from .band_diagnostics import (
    plot_band_diagnostics,
    plot_bias_profile,
    plot_bootstrap_vs_empirical,
    plot_roc_with_band,
    plot_variance_comparison,
)
from .plot_aggregate import (
    get_method_color,
    get_method_colors_dict,
    plot_coverage_by_n_total,
    plot_coverage_by_prevalence,
    plot_pareto_frontier,
    plot_violation_direction,
    plot_violation_proximity,
    set_publication_style,
)
from .plot_aggregate_curve import (
    plot_coverage_by_region,
    plot_regionwise_pareto_frontier,
)

__all__ = [
    # Band diagnostics
    "plot_band_diagnostics",
    "plot_roc_with_band",
    "plot_variance_comparison",
    "plot_bootstrap_vs_empirical",
    "plot_bias_profile",
    # Aggregate analysis plots
    "plot_pareto_frontier",
    "plot_violation_proximity",
    "plot_coverage_by_n_total",
    "plot_coverage_by_prevalence",
    "plot_violation_direction",
    # Curve/region analysis plots
    "plot_coverage_by_region",
    "plot_regionwise_pareto_frontier",
    # Utilities
    "get_method_color",
    "get_method_colors_dict",
    "set_publication_style",
]
