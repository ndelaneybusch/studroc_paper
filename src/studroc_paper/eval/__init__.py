"""Evaluation utilities for ROC confidence band diagnostics.

This module provides tools for evaluating the quality of ROC confidence bands through
simulation studies and diagnostic metrics. It includes:

- **Simulation framework**: Run Monte Carlo simulations to assess coverage rates,
  band widths, and uniformity properties across different data-generating processes
- **Diagnostic metrics**: Compute pointwise coverage, violation rates, directional
  bias tests, and uniformity statistics for confidence bands
- **Result aggregation**: Combine simulation results across multiple runs and compute
  summary statistics with confidence intervals
- **Data ingestion**: Load simulation results from JSON files into structured pandas
  DataFrames for downstream analysis and visualization

The module supports evaluation of various band construction methods (envelope-based,
Kolmogorov-Smirnov, bootstrap) across different confidence levels and sample sizes.

Example:
    Run a simulation study and aggregate results::

        from studroc_paper.eval import run_band_simulation, aggregate_band_results

        # Run simulation for a specific DGP
        results = run_band_simulation(
            dgp_fn=my_dgp_function,
            n_simulations=1000,
            alpha=0.05
        )

        # Aggregate and summarize
        summary = aggregate_band_results(results)

    Load pre-computed results from JSON files::

        from studroc_paper.eval import process_folder

        # Load all aggregated results from a folder
        dfs = process_folder("data/results", pattern="*_aggregated.json")

        # Access standard and curve DataFrames by alpha level
        df_standard = dfs["alpha_0.05_standard"]
        df_curve = dfs["alpha_0.05_curve"]
"""

from .build_data_from_jsons import (
    get_available_keys,
    process_folder,
    process_json_file,
)
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
    "get_available_keys",
    "process_folder",
    "process_json_file",
    "run_band_simulation",
    "summarize_evaluation",
]
