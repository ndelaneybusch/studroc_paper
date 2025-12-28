"""Visualization utilities for ROC confidence band diagnostics."""

from .band_diagnostics import (
    plot_band_diagnostics,
    plot_bias_profile,
    plot_bootstrap_vs_empirical,
    plot_roc_with_band,
    plot_variance_comparison,
)

__all__ = [
    "plot_band_diagnostics",
    "plot_roc_with_band",
    "plot_variance_comparison",
    "plot_bootstrap_vs_empirical",
    "plot_bias_profile",
]
