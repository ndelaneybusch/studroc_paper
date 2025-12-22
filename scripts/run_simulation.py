#!/usr/bin/env python
"""
ROC Confidence Band Simulation Study

This script runs the complete simulation experiment comparing confidence band methods
across various DGPs, sample sizes, and prevalence scenarios.

Usage:
    python run_simulation.py                          # Run with defaults
    python run_simulation.py --n-lhs 500 --n-sim 10  # Custom parameters
    python run_simulation.py --dgps lognormal         # Run specific DGP only
    python run_simulation.py --help                   # Show all options
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datagen.roc_to_dgp import map_lhs_to_dgp
from datagen.true_rocs import (
    make_beta_opposing_skew_dgp,
    make_bimodal_negative_dgp,
    make_exponential_dgp,
    make_heteroskedastic_gaussian_dgp,
    make_lognormal_dgp,
    make_student_t_dgp,
)
from eval.eval import aggregate_band_results, evaluate_single_band
from methods.envelope_boot import envelope_bootstrap_band
from methods.ks_band import fixed_width_ks_band
from methods.pointwise_boot import pointwise_bootstrap_band
from methods.working_hotelling import working_hotelling_band
from sampling.bootstrap_grid import generate_bootstrap_grid
from sampling.lhs import maximin_lhs

# =============================================================================
# Configuration
# =============================================================================


def get_dgp_specs() -> dict[str, dict[str, NDArray]]:
    """Define DGP specifications for LHS sampling."""
    return {
        "lognormal": {
            "make_dgp": make_lognormal_dgp,
            "lhs_params": ["auc", "sigma"],
            "lhs_bounds": [(0.55, 0.99), (0.1, 3.0)],
        },
        "hetero_gaussian": {
            "make_dgp": make_heteroskedastic_gaussian_dgp,
            "lhs_params": ["auc", "sigma_ratio"],
            "lhs_bounds": [(0.55, 0.99), (0.2, 5.0)],
        },
        "beta_opposing": {
            "make_dgp": make_beta_opposing_skew_dgp,
            "lhs_params": ["auc", "alpha"],
            "lhs_bounds": [(0.55, 0.99), (0.5, 10.0)],
        },
        "student_t": {
            "make_dgp": make_student_t_dgp,
            "lhs_params": ["auc", "df"],
            "lhs_bounds": [(0.55, 0.99), (1.1, 30.0)],
        },
        "bimodal_negative": {
            "make_dgp": make_bimodal_negative_dgp,
            "lhs_params": ["auc", "mixture_weight", "mode_separation"],
            "lhs_bounds": [(0.55, 0.99), (0.1, 0.9), (0.1, 4.0)],
        },
        "exponential": {
            "make_dgp": make_exponential_dgp,
            "lhs_params": ["auc", "neg_rate"],
            "lhs_bounds": [(0.55, 0.99), (0.1, 10.0)],
        },
    }


def get_sample_size_configs():
    """Define sample size configurations (n0, n1 pairs)."""
    configs = []

    # Balanced samples for most sizes
    for n_total in [10, 30, 100, 300, 1000, 10000]:
        if n_total != 1000:
            configs.append(
                {
                    "n_total": n_total,
                    "n_pos": n_total // 2,
                    "n_neg": n_total // 2,
                    "prevalence": 0.5,
                }
            )

    # Special prevalence scenarios for n=1000
    configs.extend(
        [
            {"n_total": 1000, "n_pos": 10, "n_neg": 990, "prevalence": 0.01},
            {"n_total": 1000, "n_pos": 100, "n_neg": 900, "prevalence": 0.10},
            {"n_total": 1000, "n_pos": 500, "n_neg": 500, "prevalence": 0.50},
        ]
    )

    return configs


# =============================================================================
# CI Method Wrappers
# =============================================================================


def compute_all_bands(
    y_true: NDArray,
    y_score: NDArray,
    boot_tpr_matrix: NDArray,
    fpr_grid: NDArray,
    true_tpr: NDArray,
    alpha: float,
) -> dict[str, tuple[NDArray, NDArray]]:
    """Compute all confidence bands."""

    dtype = y_score.dtype
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fpr = fpr.astype(dtype)
    tpr = tpr.astype(dtype)

    results = {"fpr_grid": fpr_grid, "tpr": tpr}

    # Compute envelope band
    fpr_out, lower, upper = envelope_bootstrap_band(
        boot_tpr_matrix=boot_tpr_matrix,
        fpr_grid=fpr_grid,
        y_true=y_true,
        y_score=y_score,
        alpha=alpha,
    )
    results["envelope"] = evaluate_single_band(
        lower_band=lower, upper_band=upper, true_tpr=true_tpr, fpr_grid=fpr_grid
    )

    # Compute pointwise band
    fpr_out, lower, upper = pointwise_bootstrap_band(
        boot_tpr_matrix=boot_tpr_matrix, fpr_grid=fpr_grid, alpha=alpha
    )
    results["pointwise"] = evaluate_single_band(
        lower_band=lower, upper_band=upper, true_tpr=true_tpr, fpr_grid=fpr_grid
    )

    # Compute ks band
    fpr_out, lower, upper = fixed_width_ks_band(
        y_true=y_true, y_score=y_score, k=len(fpr_grid), alpha=alpha
    )
    results["ks"] = evaluate_single_band(
        lower_band=lower, upper_band=upper, true_tpr=true_tpr, fpr_grid=fpr_grid
    )

    # Compute Working-Hotelling band
    fpr_out, lower, upper = working_hotelling_band(
        y_true=y_true, y_score=y_score, k=len(fpr_grid), alpha=alpha
    )
    results["working_hotelling"] = evaluate_single_band(
        lower_band=lower, upper_band=upper, true_tpr=true_tpr, fpr_grid=fpr_grid
    )

    return results


# =============================================================================
# Loop Level Functions
# =============================================================================


def run_single_simulation(
    dgp, n_pos, n_neg, confidence_levels, fpr_grid, rng, dtype, B
):
    """
    Run a single simulation: generate data, compute CIs, evaluate.

    Returns:
        dict: Results keyed by (method_name, confidence_level)
    """
    # Generate data
    scores_pos, scores_neg = dgp.sample(n_pos, n_neg, rng)
    true_tpr = dgp.get_true_roc(fpr_grid)

    # Create labels and scores for sklearn-style interface
    y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    y_score = np.concatenate([scores_pos, scores_neg])
    y_score = y_score.astype(dtype)

    # generate bootstraps
    fpr, tpr, _ = roc_curve(y_true, y_score)
    tpr = tpr.astype(dtype)

    # convert to torch
    fpr = torch.from_numpy(fpr)
    tpr = torch.from_numpy(tpr)
    fpr_grid = torch.from_numpy(fpr_grid)

    boot_tpr_matrix = generate_bootstrap_grid(
        fpr=fpr,
        tpr=tpr,
        n_negatives=n_neg,
        n_positives=n_pos,
        B=B,
        grid=fpr_grid,
        device=None,
        batch_size=500,
    )

    # Evaluate each method at each confidence level
    results = {
        "fpr": fpr,
        "tpr": tpr,
        "true_tpr": true_tpr,
        "auc": roc_auc_score(y_true, y_score),
    }

    for alpha in confidence_levels:
        # Compute band
        results[alpha] = compute_all_bands(
            y_true=y_true,
            y_score=y_score,
            boot_tpr_matrix=boot_tpr_matrix,
            fpr_grid=fpr_grid,
            true_tpr=true_tpr,
            alpha=alpha,
        )

    return results


def run_lhs_combination(
    lhs_idx,
    dgp_params,
    dgp_type,
    dgp_spec,
    sample_config,
    n_sim,
    confidence_levels,
    fpr_grid,
    lhs_params_dict,
    rng,
    dtype,
    B,
):
    """
    Run all simulations for a single LHS parameter combination.

    Returns:
        list: List of result dictionaries, one per simulation repeat
    """
    # Create DGP instance with these parameters
    # Extract scalar parameters for this LHS index
    params_for_dgp = {}
    for key, value in dgp_params.items():
        if isinstance(value, np.ndarray):
            params_for_dgp[key] = (
                float(value[lhs_idx]) if value.ndim > 0 else float(value)
            )
        elif isinstance(value, list):
            # Handle lists (for bimodal_negative)
            params_for_dgp[key] = value[lhs_idx] if len(value) > lhs_idx else value[0]
        else:
            params_for_dgp[key] = value

    dgp = dgp_spec["make_dgp"](**params_for_dgp)

    # Run n_sim simulations
    simulation_results = []

    for sim_idx in range(n_sim):
        sim_results = run_single_simulation(
            dgp=dgp,
            n_pos=sample_config["n_pos"],
            n_neg=sample_config["n_neg"],
            confidence_levels=confidence_levels,
            fpr_grid=fpr_grid,
            rng=rng,
            B=B,
            dtype=dtype,
        )
        # Collect metadata for this simulation
        metadata = {
            "lhs_idx": lhs_idx,
            "sim_idx": sim_idx,
            "dgp_type": dgp_type,
            "n_pos": sample_config["n_pos"],
            "n_neg": sample_config["n_neg"],
            "n_total": sample_config["n_total"],
            "prevalence": sample_config["prevalence"],
        }

        # Add LHS parameters
        for param_name in dgp_spec["lhs_params"]:
            metadata[f"lhs_{param_name}"] = lhs_params_dict[param_name][lhs_idx]

        # Add DGP parameters
        for key, value in params_for_dgp.items():
            metadata[f"dgp_{key}"] = value

        simulation_results.append({"metadata": metadata, "ci_results": sim_results})

    return simulation_results


def run_sample_size_config(
    sample_config,
    dgp_type,
    dgp_spec,
    lhs_params_dict,
    dgp_params,
    n_lhs,
    n_sim,
    confidence_levels,
    output_dir,
    seed,
    dtype,
    B,
):
    """
    Run all LHS combinations for a single sample size configuration.

    Saves results to disk after completion.
    """
    rng = np.random.default_rng(seed)

    # Determine K (eval grid size)
    K = sample_config["n_neg"] + 1
    fpr_grid = np.linspace(0, 1, K)

    # Storage for results
    all_simulation_results = []

    # Progress bar over LHS combinations
    desc = f"{dgp_type} n={sample_config['n_total']} prev={sample_config['prevalence']:.2f}"

    for lhs_idx in tqdm(range(n_lhs), desc=desc, leave=False):
        lhs_results = run_lhs_combination(
            lhs_idx=lhs_idx,
            dgp_params=dgp_params,
            dgp_type=dgp_type,
            dgp_spec=dgp_spec,
            sample_config=sample_config,
            n_sim=n_sim,
            confidence_levels=confidence_levels,
            fpr_grid=fpr_grid,
            lhs_params_dict=lhs_params_dict,
            rng=rng,
            dtype=dtype,
            B=B,
        )

        all_simulation_results.extend(lhs_results)

    # Save results
    save_results(
        simulation_results=all_simulation_results,
        dgp_type=dgp_type,
        sample_config=sample_config,
        confidence_levels=confidence_levels,
        output_dir=output_dir,
    )


def run_dgp(
    dgp_type,
    dgp_spec,
    sample_configs,
    n_lhs,
    n_sim,
    confidence_levels,
    output_dir,
    seed,
    dtype,
    B,
):
    """
    Run all sample size configurations for a single DGP.
    """
    print(f"\n{'=' * 60}")
    print(f"DGP: {dgp_type}")
    print(f"{'=' * 60}")

    rng = np.random.default_rng(seed)

    # Generate LHS samples
    print(f"Generating {n_lhs} LHS samples...")
    n_dims = len(dgp_spec["lhs_params"])

    lhs_unit = maximin_lhs(
        n=n_lhs, k=n_dims, method="build", dup=5, seed=rng.integers(0, 2**31)
    )

    # Scale to parameter bounds
    lower = np.array([b[0] for b in dgp_spec["lhs_bounds"]])
    upper = np.array([b[1] for b in dgp_spec["lhs_bounds"]])
    lhs_scaled = lower + lhs_unit * (upper - lower)

    # Create parameter dictionary
    lhs_params_dict = {
        name: lhs_scaled[:, i] for i, name in enumerate(dgp_spec["lhs_params"])
    }

    # Map to DGP parameters
    dgp_params = map_lhs_to_dgp(dgp_type, lhs_params_dict)

    # Run each sample size configuration
    for sample_config in tqdm(sample_configs, desc="Sample sizes"):
        run_sample_size_config(
            sample_config=sample_config,
            dgp_type=dgp_type,
            dgp_spec=dgp_spec,
            lhs_params_dict=lhs_params_dict,
            dgp_params=dgp_params,
            n_lhs=n_lhs,
            n_sim=n_sim,
            confidence_levels=confidence_levels,
            output_dir=output_dir,
            seed=rng.integers(0, 2**31),
            dtype=dtype,
            B=B,
        )


# =============================================================================
# Result Saving
# =============================================================================


def save_results(
    simulation_results, dgp_type, sample_config, confidence_levels, output_dir
):
    """
    Save simulation results to disk.

    Creates:
    - Individual CI evaluations as feather (long format)
    - Aggregated metrics as JSON (per method, per confidence level)
    """
    timestamp = datetime.now().strftime("%Y%m%d")
    n_total = sample_config["n_total"]
    prev = int(sample_config["prevalence"] * 100)

    base_filename = f"{dgp_type}_n{n_total}_prev{prev}_{timestamp}"

    # Hardcoded method names
    method_names = ["envelope", "pointwise", "ks", "working_hotelling"]

    # Prepare individual results (long format)
    individual_records = []

    # Organize results by (method, confidence_level)
    results_by_method_alpha = {}

    for sim_result in simulation_results:
        metadata = sim_result["metadata"]
        ci_results = sim_result["ci_results"]

        # Add empirical AUC to metadata
        metadata["empirical_auc"] = ci_results.get("auc", np.nan)

        # Iterate over confidence levels
        for alpha in confidence_levels:
            if alpha not in ci_results:
                continue

            alpha_results = ci_results[alpha]

            # Iterate over methods
            for method_name in method_names:
                if method_name not in alpha_results:
                    continue

                band_result = alpha_results[method_name]

                # Add to aggregation dict
                key = (method_name, alpha)
                if key not in results_by_method_alpha:
                    results_by_method_alpha[key] = []
                results_by_method_alpha[key].append(band_result)

                # Create individual record
                # Start with all metadata to ensure unique identification
                record = {}

                # Core identifiers
                record["dgp_type"] = metadata["dgp_type"]
                record["method"] = method_name
                record["alpha"] = alpha
                record["confidence_level"] = 1 - alpha

                # Sample configuration
                record["n_pos"] = metadata["n_pos"]
                record["n_neg"] = metadata["n_neg"]
                record["n_total"] = metadata["n_total"]
                record["prevalence"] = metadata["prevalence"]

                # Extract LHS parameters (columns starting with "lhs_")
                for key, value in metadata.items():
                    if key.startswith("lhs_"):
                        record[key] = value

                # Extract DGP parameters (columns starting with "dgp_")
                for key, value in metadata.items():
                    if key.startswith("dgp_"):
                        record[key] = value

                record["lhs_idx"] = metadata["lhs_idx"]
                record["sim_idx"] = metadata["sim_idx"]

                # Empirical AUC
                record["empirical_auc"] = metadata["empirical_auc"]

                # Band evaluation results
                record["covers_entirely"] = band_result.covers_entirely
                record["violation_above"] = band_result.violation_above
                record["violation_below"] = band_result.violation_below
                record["max_violation_above"] = float(band_result.max_violation_above)
                record["max_violation_below"] = float(band_result.max_violation_below)
                record["band_area"] = float(band_result.band_area)
                record["mean_band_width"] = float(band_result.band_widths.mean())

                # Add regional violations
                for region, violated in band_result.violation_by_region.items():
                    record[f"violation_{region}"] = violated

                individual_records.append(record)

    # Save individual results as feather
    df_individual = pd.DataFrame(individual_records)
    feather_path = output_dir / f"{base_filename}_individual.feather"
    df_individual.to_feather(feather_path)

    # Aggregate and save summary statistics
    aggregated_results = {}

    for (method_name, alpha), band_results in results_by_method_alpha.items():
        conf_level = 1 - alpha
        aggregated = aggregate_band_results(band_results, nominal_alpha=alpha)

        # Convert to serializable dict
        agg_dict = {
            "n_simulations": aggregated.n_simulations,
            "nominal_alpha": aggregated.nominal_alpha,
            "confidence_level": conf_level,
            "coverage_rate": float(aggregated.coverage_rate),
            "coverage_se": float(aggregated.coverage_se),
            "coverage_ci_lower": float(aggregated.coverage_ci_lower),
            "coverage_ci_upper": float(aggregated.coverage_ci_upper),
            "violation_rate_above": float(aggregated.violation_rate_above),
            "violation_rate_below": float(aggregated.violation_rate_below),
            "direction_test_pvalue": float(aggregated.direction_test_pvalue),
            "mean_band_area": float(aggregated.mean_band_area),
            "std_band_area": float(aggregated.std_band_area),
            "mean_band_width": float(aggregated.mean_band_width),
            "width_percentiles": {
                k: float(v) for k, v in aggregated.width_percentiles.items()
            },
            "width_by_fpr_region": {
                k: float(v) for k, v in aggregated.width_by_fpr_region.items()
            },
            "violation_rate_by_region": {
                k: float(v) for k, v in aggregated.violation_rate_by_region.items()
            },
            "mean_max_violation": float(aggregated.mean_max_violation),
            "percentile_95_max_violation": float(
                aggregated.percentile_95_max_violation
            ),
        }

        if method_name not in aggregated_results:
            aggregated_results[method_name] = {}
        aggregated_results[method_name][f"alpha_{alpha}"] = agg_dict

    # Add metadata
    aggregated_results["metadata"] = {
        "dgp_type": dgp_type,
        "n_total": sample_config["n_total"],
        "n_pos": sample_config["n_pos"],
        "n_neg": sample_config["n_neg"],
        "prevalence": sample_config["prevalence"],
        "timestamp": timestamp,
        "n_lhs_combinations": len(
            set(r["metadata"]["lhs_idx"] for r in simulation_results)
        ),
        "n_simulations_per_lhs": len(
            set(r["metadata"]["sim_idx"] for r in simulation_results)
        ),
    }

    # Save aggregated results as JSON
    json_path = output_dir / f"{base_filename}_aggregated.json"
    with open(json_path, "w") as f:
        json.dump(aggregated_results, f, indent=2)

    print(f"  Saved: {feather_path.name}")
    print(f"  Saved: {json_path.name}")


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Run ROC confidence band simulation study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Global parameters
    parser.add_argument(
        "--n-lhs", type=int, default=1000, help="Number of LHS parameter combinations"
    )
    parser.add_argument(
        "--n-sim",
        type=int,
        default=1,
        help="Number of simulation repeats per configuration",
    )
    parser.add_argument(
        "--bootstrap-size",
        "-B",
        type=int,
        default=4000,
        help="Number of bootstrap replicates for envelope method",
    )
    parser.add_argument(
        "--confidence-levels",
        type=float,
        nargs="+",
        default=[0.5, 0.05],
        help="Confidence levels (as alpha values, e.g., 0.05 for 95%% CI)",
    )

    # DGP selection
    parser.add_argument(
        "--dgps",
        nargs="+",
        choices=list(get_dgp_specs().keys()) + ["all"],
        default=["all"],
        help="Which DGPs to run",
    )

    # Sample size selection
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        help="Specific total sample sizes to run (overrides default configs)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results"),
        help="Output directory for results",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["envelope_boot", "working_hotelling", "all"],
        default=["all"],
        help="Which CI methods to run",
    )

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get DGP specs
    dgp_specs = get_dgp_specs()

    # Get sample size configs
    if args.sample_sizes:
        sample_configs = []
        for n_total in args.sample_sizes:
            if n_total == 1000:
                # Add all prevalence scenarios
                sample_configs.extend(
                    [
                        {
                            "n_total": 1000,
                            "n_pos": 10,
                            "n_neg": 990,
                            "prevalence": 0.01,
                        },
                        {
                            "n_total": 1000,
                            "n_pos": 100,
                            "n_neg": 900,
                            "prevalence": 0.10,
                        },
                        {
                            "n_total": 1000,
                            "n_pos": 500,
                            "n_neg": 500,
                            "prevalence": 0.50,
                        },
                    ]
                )
            else:
                sample_configs.append(
                    {
                        "n_total": n_total,
                        "n_pos": n_total // 2,
                        "n_neg": n_total // 2,
                        "prevalence": 0.5,
                    }
                )
    else:
        sample_configs = get_sample_size_configs()

    # Print configuration
    print("\n" + "=" * 60)
    print("SIMULATION CONFIGURATION")
    print("=" * 60)
    print(f"Sample size configs: {len(sample_configs)}")
    print(f"LHS combinations per DGP: {args.n_lhs}")
    print(f"Simulation repeats per combination: {args.n_sim}")
    print(f"Bootstrap replicates (envelope): {args.bootstrap_size}")
    print(f"Confidence levels (alpha): {args.confidence_levels}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {args.seed}")
    print("=" * 60)

    # Check for GPU
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU (consider using GPU for faster bootstrap)")

    # Run simulations
    rng = np.random.default_rng(args.seed)

    for dgp_type in [
        "lognormal",
        "hetero_gaussian",
        "beta_opposing",
        "student_t",
        "bimodal_negative",
        "exponential",
    ]:
        dgp_spec = dgp_specs[dgp_type]
        run_dgp(
            dgp_type=dgp_type,
            dgp_spec=dgp_spec,
            sample_configs=sample_configs,
            n_lhs=args.n_lhs,
            n_sim=args.n_sim,
            confidence_levels=args.confidence_levels,
            output_dir=output_dir,
            seed=rng.integers(0, 2**31),
            dtype=np.float32,
            B=args.bootstrap_size,
        )

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
