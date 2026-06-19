#!/usr/bin/env python
"""
ROC Confidence Band Simulation Study

This script runs the complete simulation experiment comparing confidence band methods
across various DGPs, sample sizes, and prevalence scenarios.

Method roster:
    Baselines: fixed-width KS (Campbell), Working-Hotelling (parametric binormal),
    pointwise bootstrap (uncorrected and Sidak-corrected), Wilson rectangles
    (Sidak and Bonferroni margin corrections).

    Envelope family: the final studentized bootstrap envelope (Wilson variance
    floor + gated Wilson Rectangle floor + Beta order-statistic floor), three
    ablations (no Beta floor, no Wilson floor, no bootstrap), two symmetric-tail
    variants (Beta floors on both tails, Wilson rectangle on both tails), and the
    bare envelope with no floors.

AUC is sampled on the probit scale (uniform in z = Phi^-1(AUC), i.e. uniform in
binormal d'), which concentrates sampling in the high-AUC regime where tail
behavior matters most.

Usage:
    python run_simulation.py                          # Run with defaults
    python run_simulation.py --n-lhs 500 --n-sim 10  # Custom parameters
    python run_simulation.py --dgps binormal          # Run specific DGP only
    python run_simulation.py --help                   # Show all options
"""

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from numpy.typing import NDArray
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from studroc_paper.datagen.roc_to_dgp import map_lhs_to_dgp
from studroc_paper.datagen.true_rocs import (
    make_beta_opposing_skew_dgp,
    make_bimodal_negative_dgp,
    make_heteroskedastic_gaussian_dgp,
    make_logitnormal_dgp,
    make_student_t_dgp,
    make_weibull_dgp,
)
from studroc_paper.eval.eval import aggregate_band_results, evaluate_single_band
from studroc_paper.methods.envelope_boot import envelope_band_suite, wilson_beta_band
from studroc_paper.methods.ks_band import fixed_width_ks_band
from studroc_paper.methods.pointwise_boot import pointwise_bootstrap_band
from studroc_paper.methods.wilson_band import wilson_rectangle_band
from studroc_paper.methods.working_hotelling import working_hotelling_band
from studroc_paper.sampling.bootstrap_grid import generate_bootstrap_grid
from studroc_paper.sampling.lhs import iman_conover_transform, maximin_lhs

# =============================================================================
# Configuration
# =============================================================================

# Envelope-family variants computed jointly by envelope_band_suite
ENVELOPE_SUITE_METHODS = [
    "envelope",
    "envelope_no_beta_floor",
    "envelope_no_wilson_floor",
    "envelope_no_floors",
    "envelope_beta_both_tails",
    "envelope_wilson_both_tails",
]

METHOD_NAMES = [
    # Baselines
    "ks",
    "working_hotelling",
    "pointwise",
    "pointwise_sidak",
    "wilson_rectangle_sidak",
    "wilson_rectangle_bonferroni",
    # Envelope family (final method, ablations, symmetric-tail variants)
    *ENVELOPE_SUITE_METHODS,
    "envelope_no_bootstrap",
]

# Band width is recorded at these FPR landmarks (must match eval.py landmarks)
WIDTH_LANDMARKS = ("0.01", "0.05", "0.1", "0.25", "0.5", "0.75", "0.9")

FPR_REGIONS = ("0-10", "10-30", "30-50", "50-70", "70-90", "90-100")

# Dense FPR grid for measuring the true AUC, log-refined toward both corners
# where high-AUC curves are steep; independent of the per-config eval grid
TRUE_AUC_GRID = np.unique(
    np.concatenate(
        [
            [0.0, 1.0],
            np.logspace(-6, np.log10(0.5), 1500),
            1.0 - np.logspace(-6, np.log10(0.5), 1500),
        ]
    )
)


def get_dgp_specs() -> dict:
    """Define DGP specifications for LHS sampling.

    The "auc" dimension is sampled uniformly on the probit scale (see
    scale_lhs_samples); bounds are stated in AUC units.
    """
    return {
        "binormal": {
            "make_dgp": make_heteroskedastic_gaussian_dgp,
            "lhs_params": ["auc"],
            "lhs_bounds": [(0.55, 0.99)],
        },
        "logitnormal": {
            "make_dgp": make_logitnormal_dgp,
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
        "weibull": {
            "make_dgp": make_weibull_dgp,
            "lhs_params": ["auc", "shape"],
            "lhs_bounds": [(0.55, 0.99), (0.5, 5.0)],
        },
    }


def scale_lhs_samples(lhs_unit: NDArray, dgp_spec: dict) -> dict[str, NDArray]:
    """Scale unit-hypercube LHS samples to parameter bounds.

    The "auc" dimension is transformed through the probit scale: the unit
    sample is mapped uniformly onto [Phi^-1(lo), Phi^-1(hi)] and pushed back
    through Phi. Uniform sampling in z = Phi^-1(AUC) is uniform in binormal
    d' = sqrt(2) * z, concentrating the design in the high-AUC regime (the
    density in AUC units is proportional to 1/phi(z), about 15x higher at
    AUC = 0.99 than at 0.55). All other dimensions are scaled linearly.

    Args:
        lhs_unit: (n, k) LHS samples on the unit hypercube.
        dgp_spec: DGP specification with "lhs_params" and "lhs_bounds".

    Returns:
        Mapping of parameter name to scaled sample column.
    """
    columns = {}
    for i, name in enumerate(dgp_spec["lhs_params"]):
        lo, hi = dgp_spec["lhs_bounds"][i]
        u = lhs_unit[:, i]
        if name == "auc":
            z_lo, z_hi = norm.ppf(lo), norm.ppf(hi)
            columns[name] = norm.cdf(z_lo + u * (z_hi - z_lo))
        else:
            columns[name] = lo + u * (hi - lo)
    return columns


def get_sample_size_configs() -> list[dict]:
    """Define sample size configurations (n0, n1 pairs)."""
    configs = []

    # Balanced samples for most sizes
    for n_total in [10, 30, 100, 300, 1000, 3000, 10000]:
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
            {"n_total": 1000, "n_pos": 100, "n_neg": 900, "prevalence": 0.10},
            {"n_total": 1000, "n_pos": 500, "n_neg": 500, "prevalence": 0.50},
        ]
    )

    return configs


# =============================================================================
# CI Method Wrappers
# =============================================================================


def compute_bands_without_bootstrap(
    y_true: NDArray,
    y_score: NDArray,
    fpr_grid: NDArray,
    true_tpr: NDArray,
    alpha: float,
) -> dict[str, dict]:
    """Compute and evaluate confidence bands that do not require bootstrap samples."""
    method_calls = {
        "ks": (
            fixed_width_ks_band,
            {"y_true": y_true, "y_score": y_score, "k": len(fpr_grid), "alpha": alpha},
        ),
        "working_hotelling": (
            working_hotelling_band,
            {"y_true": y_true, "y_score": y_score, "k": len(fpr_grid), "alpha": alpha},
        ),
        "wilson_rectangle_sidak": (
            wilson_rectangle_band,
            {
                "y_true": y_true,
                "y_score": y_score,
                "k": len(fpr_grid),
                "alpha": alpha,
                "correction": "sidak",
                "tpr_method": "empirical",
            },
        ),
        "wilson_rectangle_bonferroni": (
            wilson_rectangle_band,
            {
                "y_true": y_true,
                "y_score": y_score,
                "k": len(fpr_grid),
                "alpha": alpha,
                "correction": "bonferroni",
                "tpr_method": "empirical",
            },
        ),
        "envelope_no_bootstrap": (
            wilson_beta_band,
            {"y_true": y_true, "y_score": y_score, "k": len(fpr_grid), "alpha": alpha},
        ),
    }

    results = {}
    for name, (band_fn, kwargs) in method_calls.items():
        t_start = time.perf_counter()
        _, lower, upper = band_fn(**kwargs)
        runtime = time.perf_counter() - t_start
        results[name] = {
            "band": evaluate_single_band(
                lower_band=lower, upper_band=upper, true_tpr=true_tpr, fpr_grid=fpr_grid
            ),
            "runtime_seconds": runtime,
            "runtime_is_shared": False,
        }
    return results


def compute_bands_with_bootstrap(
    y_true: NDArray,
    y_score: NDArray,
    boot_tpr_matrix,
    fpr_grid: NDArray,
    true_tpr: NDArray,
    confidence_levels: list[float],
) -> dict[float, dict[str, dict]]:
    """Compute and evaluate bands that consume the shared bootstrap matrix.

    The envelope-family variants are computed jointly by envelope_band_suite,
    sharing the expensive studentization work across variants and alphas;
    their recorded runtime is the suite total (flagged runtime_is_shared).
    """
    results: dict[float, dict[str, dict]] = {alpha: {} for alpha in confidence_levels}

    for alpha in confidence_levels:
        for name, correction in [("pointwise", "none"), ("pointwise_sidak", "sidak")]:
            t_start = time.perf_counter()
            _, lower, upper = pointwise_bootstrap_band(
                boot_tpr_matrix=boot_tpr_matrix,
                fpr_grid=fpr_grid,
                alpha=alpha,
                correction=correction,
            )
            runtime = time.perf_counter() - t_start
            results[alpha][name] = {
                "band": evaluate_single_band(
                    lower_band=lower,
                    upper_band=upper,
                    true_tpr=true_tpr,
                    fpr_grid=fpr_grid,
                ),
                "runtime_seconds": runtime,
                "runtime_is_shared": False,
            }

    t_start = time.perf_counter()
    suite = envelope_band_suite(
        boot_tpr_matrix=boot_tpr_matrix,
        fpr_grid=fpr_grid,
        y_true=y_true,
        y_score=y_score,
        alphas=confidence_levels,
    )
    suite_runtime = time.perf_counter() - t_start

    for alpha, variants in suite.items():
        for name, (lower, upper) in variants.items():
            results[alpha][name] = {
                "band": evaluate_single_band(
                    lower_band=lower,
                    upper_band=upper,
                    true_tpr=true_tpr,
                    fpr_grid=fpr_grid,
                ),
                "runtime_seconds": suite_runtime,
                "runtime_is_shared": True,
            }

    return results


# =============================================================================
# Loop Level Functions
# =============================================================================


def run_single_simulation(
    dgp,
    n_pos: int,
    n_neg: int,
    confidence_levels: list[float],
    fpr_grid: NDArray,
    true_tpr: NDArray,
    rng: np.random.Generator,
    dtype,
    B: int,
) -> dict:
    """
    Run a single simulation: generate data, compute CIs, evaluate.

    Returns:
        dict: Results keyed by confidence level, plus "auc" and
        "bootstrap_gen_seconds".
    """
    # Seed torch from the numpy stream so bootstrap resampling is reproducible
    torch.manual_seed(int(rng.integers(0, 2**63 - 1)))

    # Generate data
    scores_pos, scores_neg = dgp.sample(n_pos, n_neg, rng)

    # Create labels and scores for sklearn-style interface
    y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(dtype)
    y_score = np.concatenate([scores_pos, scores_neg]).astype(dtype)
    fpr_grid = fpr_grid.astype(dtype)

    results = {"auc": roc_auc_score(y_true, y_score)}

    # Generate shared bootstrap matrix (stays on the compute device; all
    # consumers move/keep it there without copies)
    t_start = time.perf_counter()
    boot_tpr_matrix = generate_bootstrap_grid(
        y_true=torch.from_numpy(y_true),
        y_score=torch.from_numpy(y_score),
        B=B,
        grid=torch.from_numpy(fpr_grid),
        device=None,
        batch_size=500,
        tpr_method="empirical",
    )
    results["bootstrap_gen_seconds"] = time.perf_counter() - t_start

    boot_results = compute_bands_with_bootstrap(
        y_true=y_true,
        y_score=y_score,
        boot_tpr_matrix=boot_tpr_matrix,
        fpr_grid=fpr_grid,
        true_tpr=true_tpr,
        confidence_levels=confidence_levels,
    )
    del boot_tpr_matrix

    for alpha in confidence_levels:
        results[alpha] = compute_bands_without_bootstrap(
            y_true=y_true,
            y_score=y_score,
            fpr_grid=fpr_grid,
            true_tpr=true_tpr,
            alpha=alpha,
        )
        results[alpha].update(boot_results[alpha])

    return results


def run_lhs_combination(
    lhs_idx: int,
    dgp_params: dict,
    dgp_type: str,
    dgp_spec: dict,
    sample_config: dict,
    n_sim: int,
    confidence_levels: list[float],
    fpr_grid: NDArray,
    lhs_params_dict: dict,
    rng: np.random.Generator,
    dtype,
    B: int,
) -> list[dict]:
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

    # The true ROC is a property of the DGP instance, not of any sample;
    # compute it once per (LHS combination, sample size) rather than per repeat
    true_tpr = dgp.get_true_roc(fpr_grid)
    true_auc = float(np.trapezoid(dgp.get_true_roc(TRUE_AUC_GRID), TRUE_AUC_GRID))

    # Run n_sim simulations
    simulation_results = []

    for sim_idx in range(n_sim):
        sim_results = run_single_simulation(
            dgp=dgp,
            n_pos=sample_config["n_pos"],
            n_neg=sample_config["n_neg"],
            confidence_levels=confidence_levels,
            fpr_grid=fpr_grid,
            true_tpr=true_tpr,
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
            "true_auc": true_auc,
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
    sample_config: dict,
    dgp_type: str,
    dgp_spec: dict,
    lhs_params_dict: dict,
    dgp_params: dict,
    n_lhs: int,
    n_sim: int,
    confidence_levels: list[float],
    output_dir: Path,
    seed: int,
    dtype,
    B: int,
) -> None:
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
        B=B,
    )


def run_dgp(
    dgp_type: str,
    dgp_spec: dict,
    sample_configs: list[dict],
    n_lhs: int,
    n_sim: int,
    confidence_levels: list[float],
    output_dir: Path,
    seed: int,
    dtype,
    B: int,
) -> None:
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

    # Decorrelate columns; the sample correlation matrix can be singular when
    # the number of samples is not comfortably larger than the dimension
    if n_dims > 1:
        try:
            lhs_unit = iman_conover_transform(
                lhs_unit, target_corr=np.eye(n_dims), rng=rng
            )
        except np.linalg.LinAlgError:
            print(
                "  WARNING: Iman-Conover decorrelation skipped "
                f"(singular correlation with n_lhs={n_lhs}, n_dims={n_dims})"
            )

    # Scale to parameter bounds (AUC dimension via the probit transform)
    lhs_params_dict = scale_lhs_samples(lhs_unit, dgp_spec)

    # Map to DGP parameters
    dgp_params = map_lhs_to_dgp(dgp_type, lhs_params_dict)

    # Filter out LHS samples where DGP parameters contain NaN (unachievable AUC)
    valid_mask = np.ones(n_lhs, dtype=bool)
    for key, value in dgp_params.items():
        if isinstance(value, np.ndarray) and value.ndim > 0:
            valid_mask &= ~np.isnan(value)

    n_valid = int(np.sum(valid_mask))
    if n_valid < n_lhs:
        n_invalid = n_lhs - n_valid
        print(f"  Filtered {n_invalid} samples with unachievable AUC ({n_valid} remaining)")

        # Filter lhs_params_dict
        lhs_params_dict = {
            name: arr[valid_mask] for name, arr in lhs_params_dict.items()
        }
        # Filter dgp_params
        for key, value in dgp_params.items():
            if isinstance(value, np.ndarray) and value.ndim > 0:
                dgp_params[key] = value[valid_mask]
            elif isinstance(value, list) and len(value) == n_lhs:
                dgp_params[key] = [v for v, m in zip(value, valid_mask) if m]
        n_lhs = n_valid

    if n_lhs == 0:
        print(f"  WARNING: No valid LHS samples for {dgp_type}, skipping...")
        return

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
    simulation_results: list[dict],
    dgp_type: str,
    sample_config: dict,
    confidence_levels: list[float],
    output_dir: Path,
    B: int,
) -> None:
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

    # Prepare individual results (long format)
    individual_records = []

    # Organize results by (method, confidence_level)
    results_by_method_alpha = {}

    for sim_result in simulation_results:
        metadata = sim_result["metadata"]
        ci_results = sim_result["ci_results"]

        # Iterate over confidence levels
        for alpha in confidence_levels:
            if alpha not in ci_results:
                continue

            alpha_results = ci_results[alpha]

            # Iterate over methods
            for method_name in METHOD_NAMES:
                if method_name not in alpha_results:
                    continue

                method_result = alpha_results[method_name]
                band_result = method_result["band"]

                # Add to aggregation dict
                key = (method_name, alpha)
                if key not in results_by_method_alpha:
                    results_by_method_alpha[key] = []
                results_by_method_alpha[key].append(band_result)

                # Create individual record
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
                for key_, value in metadata.items():
                    if key_.startswith("lhs_"):
                        record[key_] = value

                # Extract DGP parameters (columns starting with "dgp_")
                for key_, value in metadata.items():
                    if key_.startswith("dgp_"):
                        record[key_] = value

                record["lhs_idx"] = metadata["lhs_idx"]
                record["sim_idx"] = metadata["sim_idx"]

                # AUC: design target / analytic (true) and realized (empirical)
                record["true_auc"] = metadata["true_auc"]
                record["empirical_auc"] = ci_results.get("auc", np.nan)

                # Timing
                record["runtime_seconds"] = method_result["runtime_seconds"]
                record["runtime_is_shared"] = method_result["runtime_is_shared"]
                record["bootstrap_gen_seconds"] = ci_results.get(
                    "bootstrap_gen_seconds", np.nan
                )

                # Band evaluation results
                record["covers_entirely"] = band_result.covers_entirely
                record["violation_above"] = band_result.violation_above
                record["violation_below"] = band_result.violation_below
                record["max_violation_above"] = float(band_result.max_violation_above)
                record["max_violation_below"] = float(band_result.max_violation_below)
                record["violation_area_above"] = float(band_result.violation_area_above)
                record["violation_area_below"] = float(band_result.violation_area_below)
                record["band_area"] = float(band_result.band_area)
                record["mean_band_width"] = float(band_result.band_widths.mean())
                record["proportion_grid_points_violated"] = float(
                    band_result.proportion_grid_points_violated
                )

                # Direction-specific violation FPR extents (None -> NaN)
                for field_name in (
                    "violation_fpr_above_min",
                    "violation_fpr_above_max",
                    "violation_fpr_below_min",
                    "violation_fpr_below_max",
                ):
                    value = getattr(band_result, field_name)
                    record[field_name] = np.nan if value is None else float(value)

                # Band width at landmark FPRs
                for landmark in WIDTH_LANDMARKS:
                    record[f"width_at_fpr_{landmark}"] = band_result.width_at_landmarks.get(
                        landmark, np.nan
                    )

                # Mean band width by FPR region
                for region in FPR_REGIONS:
                    record[f"width_region_{region}"] = band_result.width_by_region.get(
                        region, np.nan
                    )

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
            "mean_violation_area_above": float(aggregated.mean_violation_area_above),
            "mean_violation_area_below": float(aggregated.mean_violation_area_below),
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
        "bootstrap_replicates": B,
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


def write_run_metadata(args: argparse.Namespace, output_dir: Path) -> None:
    """Write run-level reproducibility metadata alongside the results."""
    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = None

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "git_hash": git_hash,
        "methods": METHOD_NAMES,
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "pandas": pd.__version__,
        },
        "cuda_available": torch.cuda.is_available(),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
    }

    path = output_dir / f"run_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Run metadata saved to: {path}")


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
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
        default=[0.5, 0.2, 0.05],
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
        default=Path("data/results/final_run/"),
        help="Output directory for results",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get DGP specs
    dgp_specs = get_dgp_specs()
    if "all" in args.dgps:
        dgp_types = list(dgp_specs.keys())
    else:
        dgp_types = args.dgps

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
    print(f"DGPs: {dgp_types}")
    print(f"Sample size configs: {len(sample_configs)}")
    print(f"LHS combinations per DGP: {args.n_lhs}")
    print(f"Simulation repeats per combination: {args.n_sim}")
    print(f"Bootstrap replicates (envelope): {args.bootstrap_size}")
    print(f"Confidence levels (alpha): {args.confidence_levels}")
    print(f"Methods ({len(METHOD_NAMES)}): {METHOD_NAMES}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {args.seed}")
    print("=" * 60)

    # Check for GPU
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU (consider using GPU for faster bootstrap)")

    write_run_metadata(args, output_dir)

    # Run simulations
    rng = np.random.default_rng(args.seed)

    for dgp_type in dgp_types:
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
