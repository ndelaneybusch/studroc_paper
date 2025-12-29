"""Evaluation utilities for ROC confidence band diagnostics."""

from .eval import (
    BandEvaluation,
    BandResult,
    aggregate_band_results,
    compute_empirical_roc,
    compute_pointwise_coverage_diagnostics,
    compute_uniformity_diagnostics,
    evaluate_single_band,
    run_band_simulation,
    summarize_evaluation,
)

__all__ = [
    "BandEvaluation",
    "BandResult",
    "aggregate_band_results",
    "compute_empirical_roc",
    "compute_pointwise_coverage_diagnostics",
    "compute_uniformity_diagnostics",
    "evaluate_single_band",
    "run_band_simulation",
    "summarize_evaluation",
]
