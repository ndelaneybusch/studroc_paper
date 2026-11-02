#!/usr/bin/env python
"""
ROC Confidence Band Simulation Study

This script runs the complete simulation experiment comparing confidence band methods
across various DGPs, sample sizes, and prevalence scenarios.

Method roster:
    Baselines: fixed-width KS (Campbell) and Working-Hotelling (parametric
    binormal).

    Studentized bootstrap envelope: Wilson variance floor, gated Wilson
    Rectangle floor, and Beta order-statistic floor.

    Exact and fiducial bands: M3 (exact composition band, sample-size class
    split), the raw rank-space fiducial band at C = 1, and the fiducial hybrid
    (C = 1 band with the localized exact M3 floor: tails-recalibrated trim,
    tightened closure, sample-size split). Both fiducial arms are trimmed from
    one shared cloud per replicate, so the comparison between them is paired.

AUC is sampled on the probit scale (uniform in z = Phi^-1(AUC), i.e. uniform in
binormal d'), which concentrates sampling in the high-AUC regime where tail
behavior matters most.

Execution:
    The main process draws every dataset and seed in a fixed order, so
    results do not depend on scheduling or worker count. Each replicate's
    CPU work (rank-space methods and evaluation) runs in a pool of
    single-threaded worker processes. The bootstrap envelope runs on the GPU
    in the main process for configurations with at least --gpu-min-neg
    negatives, overlapping with the pool; below that, GPU launch overhead
    outweighs the work and the envelope runs on the CPU workers too.
    Runtimes are recorded under this concurrent load and are not
    single-method benchmarks.

Usage:
    python run_simulation.py                          # Run with defaults
    python run_simulation.py --n-lhs 500 --n-sim 10  # Custom parameters
    python run_simulation.py --dgps binormal          # Run specific DGP only
    python run_simulation.py --help                   # Show all options
"""

import argparse
import dataclasses
import json
import os
import subprocess
import time
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ProcessPoolExecutor,
    wait,
)
from dataclasses import dataclass, field
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
from threadpoolctl import threadpool_limits
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
from studroc_paper.eval.eval import (
    BandResult,
    aggregate_band_results,
    evaluate_single_band,
)
from studroc_paper.methods.envelope_boot import envelope_bootstrap_band
from studroc_paper.methods.fiducial_band_rs import fiducial_band_suite_rs
from studroc_paper.methods.hybrid_floor import M3Floor
from studroc_paper.methods.ks_band import fixed_width_ks_band
from studroc_paper.methods.m3_band_rs import (
    load_m3_calibration_cache,
    m3_band_rs,
    m3_calibration_cache,
    warm_m3_calibration,
)
from studroc_paper.methods.working_hotelling import working_hotelling_band
from studroc_paper.sampling.bootstrap_grid import generate_bootstrap_grid
from studroc_paper.sampling.lhs import iman_conover_transform, maximin_lhs

# =============================================================================
# Configuration
# =============================================================================

# Floor settings of the fiducial hybrid; recorded in the run metadata
HYBRID_FLOOR = M3Floor()

METHOD_NAMES = (
    "ks",
    "working_hotelling",
    "envelope",
    "m3",
    "fiducial",
    "fiducial_hybrid",
)

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

DEFAULT_SAMPLE_SIZES = (10, 30, 100, 300, 1000, 3000, 10000)

# Labels, scores, and grid are generated in this precision
DTYPE = np.float32

BOOTSTRAP_BATCH_SIZE = 500


def get_dgp_specs() -> dict:
    """Define DGP specifications for LHS sampling.

    The "auc" dimension is sampled uniformly on the probit scale (see
    scale_lhs_samples); bounds are stated in AUC units.

    Returns:
        Mapping of DGP name to its constructor, LHS parameter names, and
        LHS parameter bounds.
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


def sample_configs_for(n_total: int) -> list[dict]:
    """Return the (n_pos, n_neg) configurations run at one total sample size.

    Every size runs balanced; n = 1000 additionally runs 10% prevalence.

    Args:
        n_total: Total sample size.

    Returns:
        Sample configurations with "n_total", "n_pos", "n_neg", and
        "prevalence".
    """
    configs = []
    if n_total == 1000:
        configs.append(
            {"n_total": 1000, "n_pos": 100, "n_neg": 900, "prevalence": 0.10}
        )
    configs.append(
        {
            "n_total": n_total,
            "n_pos": n_total // 2,
            "n_neg": n_total // 2,
            "prevalence": 0.5,
        }
    )
    return configs


# =============================================================================
# Band Methods
# =============================================================================


@dataclass(frozen=True)
class Replicate:
    """One simulated dataset and everything its bands consume.

    Attributes:
        y_true: Binary labels (1 = positive).
        y_score: Scores, higher indicating the positive class.
        fpr_grid: Evaluation grid, the native grid ``linspace(0, 1, n_neg + 1)``.
        true_tpr: True ROC on ``fpr_grid``.
        method_seed: Seed for the rank-space methods' tie-breaking and the
            fiducial cloud.
        boot_seed: Seed of the bootstrap's own torch generator.
    """

    y_true: NDArray
    y_score: NDArray
    fpr_grid: NDArray
    true_tpr: NDArray
    method_seed: int
    boot_seed: int


@dataclass(frozen=True)
class MethodResult:
    """One band's evaluation and cost.

    Attributes:
        band: Evaluation of the band against the true ROC.
        runtime_seconds: Wall time of the call that produced the band.
        runtime_is_shared: Whether that call produced several methods' bands
            (the fiducial suite), so the runtime covers all of them.
        device: Device the band was computed on.
    """

    band: BandResult
    runtime_seconds: float
    runtime_is_shared: bool
    device: str


# alpha -> method name -> result
BandSet = dict[float, dict[str, MethodResult]]


@dataclass(frozen=True)
class EnvelopeResult:
    """Bootstrap envelope bands and the cost of their bootstrap matrix.

    Attributes:
        bands: Envelope results by alpha.
        bootstrap_gen_seconds: Wall time to generate the bootstrap matrix.
    """

    bands: BandSet
    bootstrap_gen_seconds: float


def _evaluate(rep: Replicate, lower: NDArray, upper: NDArray) -> BandResult:
    return evaluate_single_band(
        lower_band=lower, upper_band=upper, true_tpr=rep.true_tpr, fpr_grid=rep.fpr_grid
    )


def envelope_bands(
    *, rep: Replicate, alphas: list[float], B: int, device: torch.device
) -> EnvelopeResult:
    """Generate the bootstrap matrix and compute the envelope at every alpha.

    Args:
        rep: Replicate to compute bands for.
        alphas: Significance levels.
        B: Number of bootstrap replicates.
        device: Device for the bootstrap and envelope.

    Returns:
        Envelope evaluations by alpha and the bootstrap generation time.
    """
    generator = torch.Generator(device=device).manual_seed(rep.boot_seed)
    t_start = time.perf_counter()
    boot_tpr_matrix = generate_bootstrap_grid(
        y_true=torch.from_numpy(rep.y_true),
        y_score=torch.from_numpy(rep.y_score),
        B=B,
        grid=torch.from_numpy(rep.fpr_grid),
        device=device,
        batch_size=BOOTSTRAP_BATCH_SIZE,
        tpr_method="empirical",
        generator=generator,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    boot_seconds = time.perf_counter() - t_start

    bands: BandSet = {}
    for alpha in alphas:
        t_start = time.perf_counter()
        _, lower, upper = envelope_bootstrap_band(
            boot_tpr_matrix=boot_tpr_matrix,
            fpr_grid=rep.fpr_grid,
            y_true=rep.y_true,
            y_score=rep.y_score,
            alpha=alpha,
            boundary_method="wilson",
            device=device,
        )
        bands[alpha] = {
            "envelope": MethodResult(
                band=_evaluate(rep, lower, upper),
                runtime_seconds=time.perf_counter() - t_start,
                runtime_is_shared=False,
                device=device.type,
            )
        }
    return EnvelopeResult(bands=bands, bootstrap_gen_seconds=boot_seconds)


def rank_space_bands(*, rep: Replicate, alphas: list[float]) -> BandSet:
    """Compute KS, Working-Hotelling, M3, and both fiducial arms.

    The rank-space methods return their native grid, which is the evaluation
    grid; resampling onto an identical linspace could round a grid point up
    one step.

    Args:
        rep: Replicate to compute bands for.
        alphas: Significance levels.

    Returns:
        Evaluations by alpha and method.
    """
    bands: BandSet = {alpha: {} for alpha in alphas}
    k = len(rep.fpr_grid)

    def record(alpha, name, lower, upper, runtime, shared=False):
        bands[alpha][name] = MethodResult(
            band=_evaluate(rep, lower, upper),
            runtime_seconds=runtime,
            runtime_is_shared=shared,
            device="cpu",
        )

    for alpha in alphas:
        for name, band_fn, kwargs in (
            ("ks", fixed_width_ks_band, {"k": k}),
            ("working_hotelling", working_hotelling_band, {"k": k}),
            ("m3", m3_band_rs, {"random_state": rep.method_seed}),
        ):
            t_start = time.perf_counter()
            _, lower, upper = band_fn(
                y_true=rep.y_true, y_score=rep.y_score, alpha=alpha, **kwargs
            )
            record(alpha, name, lower, upper, time.perf_counter() - t_start)

    t_start = time.perf_counter()
    _, suite = fiducial_band_suite_rs(
        y_true=rep.y_true,
        y_score=rep.y_score,
        alphas=alphas,
        m3_floors={"fiducial": False, "fiducial_hybrid": HYBRID_FLOOR},
        random_state=rep.method_seed,
    )
    runtime = time.perf_counter() - t_start
    for alpha, variants in suite.items():
        for name, (lower, upper) in variants.items():
            record(alpha, name, lower, upper, runtime, shared=True)
    return bands


def init_worker(m3_cache: dict[tuple[int, float], float]) -> None:
    """Pin a worker process to one CPU thread and load M3 levels.

    Runs before the worker touches the Rust thread pool, which reads its
    configuration on first use. Workers never initialize CUDA: every CPU-lane
    call passes its device, so the GPU stays the main process's alone. The
    GPU is not hidden through ``CUDA_VISIBLE_DEVICES``, because initializing
    the driver with no visible device destabilizes the process.

    Args:
        m3_cache: Calibrated M3 local levels from the parent process.
    """
    os.environ["RAYON_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    threadpool_limits(limits=1)
    load_m3_calibration_cache(m3_cache)


# Native thread pools size themselves when their library loads, which in a
# spawned worker happens while the task module is imported, before any
# initializer runs. Unpinned, OpenBLAS reserves buffers for every core: about
# 4 GiB of committed memory per worker on a 24-thread machine, enough for
# Windows to refuse allocations once two dozen workers are running.
WORKER_THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def worker_pool(
    *, workers: int, m3_cache: dict[tuple[int, float], float]
) -> ProcessPoolExecutor:
    """Create a pool of single-threaded CPU workers.

    Sets :data:`WORKER_THREAD_ENV` in this process's environment, which the
    workers inherit; libraries this process has already loaded keep their
    thread pools.

    Args:
        workers: Number of worker processes.
        m3_cache: Calibrated M3 local levels for :func:`init_worker`.

    Returns:
        The pool.
    """
    os.environ.update(WORKER_THREAD_ENV)
    return ProcessPoolExecutor(
        max_workers=workers, initializer=init_worker, initargs=(m3_cache,)
    )


def _calibrate_m3_point(n0: int, n1: int, alpha: float) -> dict:
    warm_m3_calibration(n0=n0, n1=n1, alpha=alpha)
    return m3_calibration_cache()


def calibrate_m3(
    *, sample_configs: list[dict], alphas: list[float], workers: int
) -> dict[tuple[int, float], float]:
    """Calibrate M3 for every design point in parallel, once per run.

    Args:
        sample_configs: Sample configurations of the run.
        alphas: Significance levels.
        workers: Number of processes.

    Returns:
        Calibrated local levels for :func:`init_worker`.
    """
    points = {(c["n_neg"], c["n_pos"], a) for c in sample_configs for a in alphas}
    # Largest first, so the slowest calibrations start immediately
    ordered = sorted(points, key=lambda p: -max(p[0], p[1]))
    cache: dict[tuple[int, float], float] = {}
    with worker_pool(workers=min(workers, len(ordered)), m3_cache={}) as pool:
        for part in pool.map(_calibrate_m3_point, *zip(*ordered, strict=True)):
            cache.update(part)
    return cache


# =============================================================================
# Study Design
# =============================================================================


@dataclass(frozen=True)
class DgpDesign:
    """The LHS design of one DGP after filtering unachievable points.

    Attributes:
        dgp_type: DGP name.
        dgp_spec: DGP specification from ``get_dgp_specs``.
        lhs_params: Scaled LHS parameter columns.
        dgp_params: DGP constructor parameter columns.
        n_lhs: Number of retained LHS combinations.
    """

    dgp_type: str
    dgp_spec: dict
    lhs_params: dict[str, NDArray]
    dgp_params: dict
    n_lhs: int

    def constructor_params(self, lhs_idx: int) -> dict:
        """Return the DGP constructor arguments of one LHS combination."""
        params = {}
        for key, value in self.dgp_params.items():
            if isinstance(value, np.ndarray):
                params[key] = float(value[lhs_idx]) if value.ndim > 0 else float(value)
            elif isinstance(value, list):
                params[key] = value[lhs_idx] if len(value) > lhs_idx else value[0]
            else:
                params[key] = value
        return params


def design_dgp(
    *, dgp_type: str, dgp_spec: dict, n_lhs: int, rng: np.random.Generator
) -> DgpDesign:
    """Draw the LHS design for one DGP and drop unachievable points.

    Args:
        dgp_type: DGP name.
        dgp_spec: DGP specification from ``get_dgp_specs``.
        n_lhs: Number of LHS combinations to draw.
        rng: The DGP's random stream.

    Returns:
        The filtered design.
    """
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
                f"  WARNING ({dgp_type}): Iman-Conover decorrelation skipped "
                f"(singular correlation with n_lhs={n_lhs}, n_dims={n_dims})"
            )

    # Scale to parameter bounds (AUC dimension via the probit transform)
    lhs_params = scale_lhs_samples(lhs_unit, dgp_spec)
    dgp_params = map_lhs_to_dgp(dgp_type, lhs_params)

    # Drop LHS samples whose DGP parameters are NaN (unachievable AUC)
    valid_mask = np.ones(n_lhs, dtype=bool)
    for value in dgp_params.values():
        if isinstance(value, np.ndarray) and value.ndim > 0:
            valid_mask &= ~np.isnan(value)

    n_valid = int(valid_mask.sum())
    if n_valid < n_lhs:
        print(
            f"  {dgp_type}: filtered {n_lhs - n_valid} samples with unachievable "
            f"AUC ({n_valid} remaining)"
        )
        lhs_params = {name: arr[valid_mask] for name, arr in lhs_params.items()}
        for key, value in dgp_params.items():
            if isinstance(value, np.ndarray) and value.ndim > 0:
                dgp_params[key] = value[valid_mask]
            elif isinstance(value, list) and len(value) == n_lhs:
                dgp_params[key] = [
                    v for v, m in zip(value, valid_mask, strict=True) if m
                ]

    return DgpDesign(
        dgp_type=dgp_type,
        dgp_spec=dgp_spec,
        lhs_params=lhs_params,
        dgp_params=dgp_params,
        n_lhs=n_valid,
    )


@dataclass(frozen=True)
class ConfigUnit:
    """One (DGP, sample configuration) cell of the study, saved as one file.

    Attributes:
        design: The DGP's LHS design.
        sample_config: Sample configuration from ``sample_configs_for``.
        seed: Root seed of the cell; each LHS combination draws from its own
            spawned child stream, so any process can regenerate any
            combination independently.
    """

    design: DgpDesign
    sample_config: dict
    seed: int

    def tasks(self, n_sim: int) -> list["LhsTask"]:
        """Return one task per LHS combination, in LHS order."""
        children = np.random.SeedSequence(self.seed).spawn(self.design.n_lhs)
        return [
            LhsTask(
                dgp_type=self.design.dgp_type,
                params=self.design.constructor_params(lhs_idx),
                lhs_values={
                    name: float(self.design.lhs_params[name][lhs_idx])
                    for name in self.design.dgp_spec["lhs_params"]
                },
                sample_config=self.sample_config,
                lhs_idx=lhs_idx,
                seed=child,
                n_sim=n_sim,
            )
            for lhs_idx, child in enumerate(children)
        ]


def plan_units(
    *, dgp_types: list[str], sample_configs: list[dict], n_lhs: int, seed: int
) -> list[ConfigUnit]:
    """Draw every DGP design and assign each study cell its seed.

    Args:
        dgp_types: DGPs to run.
        sample_configs: Sample configurations to run.
        n_lhs: LHS combinations per DGP before filtering.
        seed: Global seed.

    Returns:
        Study cells in run order.
    """
    dgp_specs = get_dgp_specs()
    rng = np.random.default_rng(seed)
    units = []
    for dgp_type in dgp_types:
        dgp_rng = np.random.default_rng(rng.integers(0, 2**31))
        design = design_dgp(
            dgp_type=dgp_type, dgp_spec=dgp_specs[dgp_type], n_lhs=n_lhs, rng=dgp_rng
        )
        if design.n_lhs == 0:
            print(f"  WARNING: No valid LHS samples for {dgp_type}, skipping...")
            continue
        units.extend(
            ConfigUnit(
                design=design,
                sample_config=sample_config,
                seed=int(dgp_rng.integers(0, 2**31)),
            )
            for sample_config in sample_configs
        )
    return units


@dataclass(frozen=True)
class LhsTask:
    """One LHS combination of a study cell, self-contained for any process.

    Attributes:
        dgp_type: DGP name.
        params: DGP constructor arguments.
        lhs_values: LHS parameter values.
        sample_config: Sample configuration.
        lhs_idx: Index of the combination within its DGP design.
        seed: The combination's seed stream.
        n_sim: Repeats of the combination.
    """

    dgp_type: str
    params: dict
    lhs_values: dict[str, float]
    sample_config: dict
    lhs_idx: int
    seed: np.random.SeedSequence
    n_sim: int


@dataclass(frozen=True)
class SimulatedLhs:
    """The datasets of one LHS combination.

    Attributes:
        dgp: The DGP instance.
        replicates: One replicate per repeat, in order.
        empirical_aucs: Empirical AUC of each replicate.
    """

    dgp: object
    replicates: list[Replicate]
    empirical_aucs: list[float]


def simulate_lhs(task: LhsTask) -> SimulatedLhs:
    """Generate every replicate of one LHS combination from its seed alone.

    Args:
        task: The combination.

    Returns:
        Its DGP and replicates; identical in every process.
    """
    config = task.sample_config
    rng = np.random.default_rng(task.seed)
    dgp = get_dgp_specs()[task.dgp_type]["make_dgp"](**task.params)
    fpr_grid = np.linspace(0, 1, config["n_neg"] + 1)
    # The true ROC is a property of the DGP instance, not of any sample
    true_tpr = dgp.get_true_roc(fpr_grid)
    y_true = np.concatenate(
        [np.ones(config["n_pos"]), np.zeros(config["n_neg"])]
    ).astype(DTYPE)

    replicates, aucs = [], []
    for _ in range(task.n_sim):
        boot_seed = int(rng.integers(0, 2**63 - 1))
        scores_pos, scores_neg = dgp.sample(config["n_pos"], config["n_neg"], rng)
        y_score = np.concatenate([scores_pos, scores_neg]).astype(DTYPE)
        replicates.append(
            Replicate(
                y_true=y_true,
                y_score=y_score,
                fpr_grid=fpr_grid.astype(DTYPE),
                true_tpr=true_tpr,
                method_seed=int(rng.integers(0, 2**63 - 1)),
                boot_seed=boot_seed,
            )
        )
        aucs.append(float(roc_auc_score(y_true, y_score)))
    return SimulatedLhs(dgp=dgp, replicates=replicates, empirical_aucs=aucs)


@dataclass(frozen=True)
class LhsOutcome:
    """Everything a worker computes for one LHS combination.

    Attributes:
        metadata: Per-repeat metadata records.
        empirical_aucs: Per-repeat empirical AUC.
        bands: Per-repeat evaluations by alpha and method.
        envelopes: Per-repeat envelope results when the worker computed
            them, else None.
    """

    metadata: list[dict]
    empirical_aucs: list[float]
    bands: list[BandSet]
    envelopes: list[EnvelopeResult] | None


def run_lhs_task(*, task: LhsTask, alphas: list[float], B: int | None) -> LhsOutcome:
    """Worker entry point: simulate one LHS combination and compute its bands.

    Args:
        task: The combination.
        alphas: Significance levels.
        B: Bootstrap replicates for a CPU envelope, or None when the envelope
            runs on the GPU lane.

    Returns:
        The combination's metadata and evaluations.
    """
    sim = simulate_lhs(task)
    config = task.sample_config
    base_metadata = {
        "lhs_idx": task.lhs_idx,
        "dgp_type": task.dgp_type,
        "n_pos": config["n_pos"],
        "n_neg": config["n_neg"],
        "n_total": config["n_total"],
        "prevalence": config["prevalence"],
        "true_auc": float(
            np.trapezoid(sim.dgp.get_true_roc(TRUE_AUC_GRID), TRUE_AUC_GRID)
        ),
        **{f"lhs_{name}": value for name, value in task.lhs_values.items()},
        **{f"dgp_{key}": value for key, value in task.params.items()},
    }
    cpu = torch.device("cpu")
    return LhsOutcome(
        metadata=[{**base_metadata, "sim_idx": i} for i in range(task.n_sim)],
        empirical_aucs=sim.empirical_aucs,
        bands=[rank_space_bands(rep=rep, alphas=alphas) for rep in sim.replicates],
        envelopes=None
        if B is None
        else [
            envelope_bands(rep=rep, alphas=alphas, B=B, device=cpu)
            for rep in sim.replicates
        ],
    )


# =============================================================================
# Orchestration
# =============================================================================


@dataclass
class UnitResults:
    """Results of one study cell as they arrive from both lanes.

    Attributes:
        unit: Study cell.
        n_sim: Repeats per LHS combination.
        outcomes: Worker outcome per LHS combination.
        gpu_envelopes: GPU-lane envelope results per LHS combination.
        gpu_aucs: Empirical AUCs of the data the GPU lane regenerated, per
            LHS combination; they must match the worker's.
    """

    unit: ConfigUnit
    n_sim: int
    outcomes: dict[int, LhsOutcome] = field(default_factory=dict)
    gpu_envelopes: dict[int, list[EnvelopeResult]] = field(default_factory=dict)
    gpu_aucs: dict[int, list[float]] = field(default_factory=dict)

    def add_gpu(
        self, lhs_idx: int, sim: SimulatedLhs, envelopes: list[EnvelopeResult]
    ) -> None:
        """Store the GPU lane's envelopes for one LHS combination."""
        self.gpu_envelopes[lhs_idx] = envelopes
        self.gpu_aucs[lhs_idx] = sim.empirical_aucs

    def add_outcome(self, lhs_idx: int, outcome: LhsOutcome) -> None:
        """Store a worker outcome, checking it saw the GPU lane's data.

        Raises:
            RuntimeError: If the worker and the GPU lane regenerated
                different data for the combination.
        """
        if (
            lhs_idx in self.gpu_aucs
            and self.gpu_aucs[lhs_idx] != outcome.empirical_aucs
        ):
            raise RuntimeError(
                f"{self.unit.design.dgp_type} LHS {lhs_idx}: the worker and the "
                "GPU lane generated different data from the same seed"
            )
        self.outcomes[lhs_idx] = outcome

    @property
    def complete(self) -> bool:
        """Whether every LHS combination's worker outcome has arrived."""
        return len(self.outcomes) == self.unit.design.n_lhs

    def slots(self) -> Iterator[dict]:
        """Yield each replicate's merged results, in LHS and repeat order."""
        for lhs_idx in range(self.unit.design.n_lhs):
            outcome = self.outcomes[lhs_idx]
            envelopes = outcome.envelopes or self.gpu_envelopes[lhs_idx]
            for i in range(self.n_sim):
                bands = {
                    alpha: {**by_method, **envelopes[i].bands[alpha]}
                    for alpha, by_method in outcome.bands[i].items()
                }
                yield {
                    "metadata": outcome.metadata[i],
                    "empirical_auc": outcome.empirical_aucs[i],
                    "bands": bands,
                    "bootstrap_gen_seconds": envelopes[i].bootstrap_gen_seconds,
                }


def run_study(
    *,
    units: list[ConfigUnit],
    n_sim: int,
    alphas: list[float],
    B: int,
    pool: Executor,
    gpu: torch.device | None,
    gpu_min_neg: int,
    max_in_flight: int,
    output_dir: Path,
) -> None:
    """Run every study cell across the worker pool and the GPU lane.

    LHS combinations are submitted in a fixed order across cell boundaries,
    so the pool never drains between cells; a cell is saved as soon as its
    last worker outcome arrives. For GPU-lane cells the main process
    regenerates each combination's data from its seed and computes the
    envelopes while the workers handle the rest.

    Args:
        units: Study cells in run order.
        n_sim: Repeats per LHS combination.
        alphas: Significance levels.
        B: Bootstrap replicates.
        pool: Executor whose workers ran :func:`init_worker`.
        gpu: CUDA device for the envelope lane, or None to run every
            envelope on the workers.
        gpu_min_neg: Smallest negative count routed to the GPU lane.
        max_in_flight: Cap on submitted but unfinished LHS combinations.
        output_dir: Directory for result files.
    """
    pending: dict[Future, tuple[UnitResults, int]] = {}
    progress = tqdm(
        total=sum(u.design.n_lhs for u in units) * n_sim,
        desc="Replicates",
        smoothing=0.05,
    )

    def collect(limit: int) -> None:
        while len(pending) > limit:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                results, lhs_idx = pending.pop(future)
                results.add_outcome(lhs_idx, future.result())
                progress.update(n_sim)
                if results.complete:
                    save_results(
                        results=results, alphas=alphas, output_dir=output_dir, B=B
                    )

    for unit in units:
        results = UnitResults(unit=unit, n_sim=n_sim)
        on_gpu = gpu is not None and unit.sample_config["n_neg"] >= gpu_min_neg
        for task in unit.tasks(n_sim):
            future = pool.submit(
                run_lhs_task, task=task, alphas=alphas, B=None if on_gpu else B
            )
            pending[future] = (results, task.lhs_idx)
            if on_gpu:
                sim = simulate_lhs(task)
                envelopes = [
                    envelope_bands(rep=rep, alphas=alphas, B=B, device=gpu)
                    for rep in sim.replicates
                ]
                results.add_gpu(task.lhs_idx, sim, envelopes)
            collect(max_in_flight)
    collect(0)
    progress.close()


# =============================================================================
# Result Saving
# =============================================================================


def _band_record(band: BandResult) -> dict:
    """Flatten one band evaluation into feather columns.

    Args:
        band: Evaluation of one band against the true ROC.

    Returns:
        Column name to scalar value; absent violation extents become NaN.
    """
    record = {
        "covers_entirely": band.covers_entirely,
        "violation_above": band.violation_above,
        "violation_below": band.violation_below,
        "max_violation_above": float(band.max_violation_above),
        "max_violation_below": float(band.max_violation_below),
        "violation_area_above": float(band.violation_area_above),
        "violation_area_below": float(band.violation_area_below),
        "band_area": float(band.band_area),
        "mean_band_width": float(band.band_widths.mean()),
        "proportion_grid_points_violated": float(band.proportion_grid_points_violated),
    }
    for field_name in (
        "violation_fpr_above_min",
        "violation_fpr_above_max",
        "violation_fpr_below_min",
        "violation_fpr_below_max",
    ):
        value = getattr(band, field_name)
        record[field_name] = np.nan if value is None else float(value)
    for landmark in WIDTH_LANDMARKS:
        record[f"width_at_fpr_{landmark}"] = band.width_at_landmarks.get(
            landmark, np.nan
        )
    for region in FPR_REGIONS:
        record[f"width_region_{region}"] = band.width_by_region.get(region, np.nan)
    for region, violated in band.violation_by_region.items():
        record[f"violation_{region}"] = violated
    return record


def _aggregate_record(band_results: list[BandResult], alpha: float) -> dict:
    """Summarize one method's evaluations at one alpha as a JSON-ready dict.

    Args:
        band_results: Evaluations across all replicates.
        alpha: Nominal alpha.

    Returns:
        Coverage, violation, and width summaries.
    """
    agg = aggregate_band_results(band_results, nominal_alpha=alpha)
    return {
        "n_simulations": agg.n_simulations,
        "nominal_alpha": agg.nominal_alpha,
        "confidence_level": 1 - alpha,
        "coverage_rate": float(agg.coverage_rate),
        "coverage_se": float(agg.coverage_se),
        "coverage_ci_lower": float(agg.coverage_ci_lower),
        "coverage_ci_upper": float(agg.coverage_ci_upper),
        "violation_rate_above": float(agg.violation_rate_above),
        "violation_rate_below": float(agg.violation_rate_below),
        "direction_test_pvalue": float(agg.direction_test_pvalue),
        "mean_band_area": float(agg.mean_band_area),
        "std_band_area": float(agg.std_band_area),
        "mean_band_width": float(agg.mean_band_width),
        "width_percentiles": {k: float(v) for k, v in agg.width_percentiles.items()},
        "width_by_fpr_region": {
            k: float(v) for k, v in agg.width_by_fpr_region.items()
        },
        "violation_rate_by_region": {
            k: float(v) for k, v in agg.violation_rate_by_region.items()
        },
        "mean_max_violation": float(agg.mean_max_violation),
        "percentile_95_max_violation": float(agg.percentile_95_max_violation),
        "mean_violation_area_above": float(agg.mean_violation_area_above),
        "mean_violation_area_below": float(agg.mean_violation_area_below),
    }


def save_results(
    *, results: UnitResults, alphas: list[float], output_dir: Path, B: int
) -> None:
    """Save one study cell's results to disk.

    Writes per-replicate band evaluations as feather (long format, one row
    per replicate x method x alpha) and per-method aggregates as JSON.

    Args:
        results: The cell's completed results.
        alphas: Significance levels evaluated.
        output_dir: Directory for result files.
        B: Number of bootstrap replicates.
    """
    dgp_type = results.unit.design.dgp_type
    sample_config = results.unit.sample_config
    timestamp = datetime.now().strftime("%Y%m%d")
    prev = int(sample_config["prevalence"] * 100)
    base_filename = f"{dgp_type}_n{sample_config['n_total']}_prev{prev}_{timestamp}"

    records = []
    bands_by_method_alpha: dict[tuple[str, float], list[BandResult]] = defaultdict(list)
    for slot in results.slots():
        for alpha in alphas:
            for method_name in METHOD_NAMES:
                result = slot["bands"][alpha][method_name]
                bands_by_method_alpha[(method_name, alpha)].append(result.band)
                records.append(
                    {
                        "method": method_name,
                        "alpha": alpha,
                        "confidence_level": 1 - alpha,
                        **slot["metadata"],
                        "empirical_auc": slot["empirical_auc"],
                        "runtime_seconds": result.runtime_seconds,
                        "runtime_is_shared": result.runtime_is_shared,
                        "device": result.device,
                        "bootstrap_gen_seconds": slot["bootstrap_gen_seconds"],
                        **_band_record(result.band),
                    }
                )

    feather_path = output_dir / f"{base_filename}_individual.feather"
    pd.DataFrame(records).to_feather(feather_path)

    aggregated: dict[str, dict] = defaultdict(dict)
    for (method_name, alpha), bands in bands_by_method_alpha.items():
        aggregated[method_name][f"alpha_{alpha}"] = _aggregate_record(bands, alpha)
    aggregated["metadata"] = {
        "dgp_type": dgp_type,
        "n_total": sample_config["n_total"],
        "n_pos": sample_config["n_pos"],
        "n_neg": sample_config["n_neg"],
        "prevalence": sample_config["prevalence"],
        "bootstrap_replicates": B,
        "timestamp": timestamp,
        "n_lhs_combinations": results.unit.design.n_lhs,
        "n_simulations_per_lhs": results.n_sim,
    }

    json_path = output_dir / f"{base_filename}_aggregated.json"
    json_path.write_text(json.dumps(aggregated, indent=2))
    tqdm.write(f"  Saved: {feather_path.name}, {json_path.name}")


def write_run_metadata(
    *, args: argparse.Namespace, output_dir: Path, gpu: torch.device | None
) -> None:
    """Write run-level reproducibility metadata alongside the results.

    Args:
        args: Parsed command-line arguments.
        output_dir: Directory for result files.
        gpu: CUDA device of the envelope lane, or None.
    """
    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = None

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "git_hash": git_hash,
        "methods": list(METHOD_NAMES),
        "hybrid_floor": dataclasses.asdict(HYBRID_FLOOR),
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "pandas": pd.__version__,
        },
        "gpu_lane_device": None if gpu is None else torch.cuda.get_device_name(gpu),
    }

    path = output_dir / f"run_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(metadata, indent=2))
    print(f"Run metadata saved to: {path}")


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Parse arguments and run the simulation study."""
    dgp_names = list(get_dgp_specs())

    parser = argparse.ArgumentParser(
        description="Run ROC confidence band simulation study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
        help="Number of bootstrap replicates for the envelope method",
    )
    parser.add_argument(
        "--confidence-levels",
        type=float,
        nargs="+",
        default=[0.5, 0.2, 0.05],
        help="Confidence levels (as alpha values, e.g., 0.05 for 95%% CI)",
    )
    parser.add_argument(
        "--dgps",
        nargs="+",
        choices=[*dgp_names, "all"],
        default=["all"],
        help="Which DGPs to run",
    )
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SAMPLE_SIZES),
        help="Total sample sizes to run (n=1000 also runs 10%% prevalence)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/final_run/"),
        help="Output directory for results",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Single-threaded CPU worker processes",
    )
    parser.add_argument(
        "--gpu-min-neg",
        type=int,
        default=500,
        help="Smallest negative count whose envelope runs on the GPU",
    )
    parser.add_argument(
        "--no-gpu", action="store_true", help="Run every envelope on the CPU workers"
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu = (
        torch.device("cuda", 0)
        if torch.cuda.is_available() and not args.no_gpu
        else None
    )

    dgp_types = dgp_names if "all" in args.dgps else args.dgps
    sample_configs = [
        config
        for n_total in args.sample_sizes
        for config in sample_configs_for(n_total)
    ]

    print("\n" + "=" * 60)
    print("SIMULATION CONFIGURATION")
    print("=" * 60)
    print(f"DGPs: {dgp_types}")
    print(f"Sample size configs: {len(sample_configs)}")
    print(f"LHS combinations per DGP: {args.n_lhs}")
    print(f"Simulation repeats per combination: {args.n_sim}")
    print(f"Bootstrap replicates (envelope): {args.bootstrap_size}")
    print(f"Confidence levels (alpha): {args.confidence_levels}")
    print(f"Methods ({len(METHOD_NAMES)}): {list(METHOD_NAMES)}")
    print(f"Hybrid floor: {HYBRID_FLOOR}")
    print(f"CPU workers: {args.workers}")
    if gpu is None:
        print("GPU lane: off (every envelope on the CPU workers)")
    else:
        lane = f"n_neg >= {args.gpu_min_neg}"
        print(f"GPU lane: {torch.cuda.get_device_name(gpu)}, envelopes with {lane}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {args.seed}")
    print("=" * 60)

    write_run_metadata(args=args, output_dir=output_dir, gpu=gpu)

    units = plan_units(
        dgp_types=dgp_types,
        sample_configs=sample_configs,
        n_lhs=args.n_lhs,
        seed=args.seed,
    )

    t_start = time.perf_counter()
    m3_cache = calibrate_m3(
        sample_configs=sample_configs,
        alphas=args.confidence_levels,
        workers=args.workers,
    )
    print(f"M3 calibration: {time.perf_counter() - t_start:.1f}s")

    with worker_pool(workers=args.workers, m3_cache=m3_cache) as pool:
        run_study(
            units=units,
            n_sim=args.n_sim,
            alphas=args.confidence_levels,
            B=args.bootstrap_size,
            pool=pool,
            gpu=gpu,
            gpu_min_neg=args.gpu_min_neg,
            max_in_flight=4 * args.workers,
            output_dir=output_dir,
        )

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
