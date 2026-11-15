#!/usr/bin/env python
"""Stratified coverage study by shape-parameter and AUC bin.

Collects the data behind the shape × AUC coverage grid (paper Fig 25 and the
cross-prevalence Fig P27) with an equal number of replicates in every point,
for the hybrid fiducial band and its two ablations: the raw fiducial band
(same cloud, no floor) and the M3 exact band (the floor alone).

The main suite's maximin LHS spreads draws uniformly over each parameter, so
the bins that matter most (Student-t with df < 2, for example) receive only
30-50 draws per point there. Here every (DGP, shape bin, AUC bin, sample size,
prevalence) point receives exactly ``--reps`` replicates. Each replicate draws
fresh parameters uniformly within its shape bin and uniformly on the probit
scale within its AUC bin (the main suite's AUC measure, restricted to the
bin), then one dataset. Parameter draws whose AUC the family cannot reach
are redrawn, so a bin that is only partly achievable is sampled on its
achievable part; rejection counts are recorded.

Replicates per point follow from the target precision: the half-width of a
95% interval for coverage p after R replicates is 1.96 * sqrt(p(1-p)/R), so
``--precision-pp`` sets R = ceil(1.96^2 p(1-p) / h^2) at the planning
coverage ``--planning-coverage``. ``--plan`` prints that table for 0.2, 0.5,
1, and 2 pp and a wall-time estimate from timing the methods on this machine,
then exits.

Results are saved per (DGP, sample configuration) in the main suite's
long format, with ``shape_bin`` and ``auc_bin`` columns. Finished chunks are
checkpointed under ``<output-dir>/partial/`` every few minutes and when the
run is interrupted (Ctrl-C); a cell's checkpoints are merged into its result
file once its last chunk finishes. Rerunning the same command skips finished
cells and checkpointed chunks, so a run can be stopped and resumed at any
time. Every chunk's random stream depends only on the seed and the chunk's
position in the design, so a resumed run produces exactly the results of an
uninterrupted one; ``design.json`` records the design, and a run whose
replicates, seed, levels, or strata differ from it is refused.

Usage:
    uv run python scripts/run_shape_auc_study.py --plan
    uv run python scripts/run_shape_auc_study.py --precision-pp 1 \\
        --output-dir data/results/shape_auc_1pp
    uv run python scripts/run_shape_auc_study.py --reps 500 --prevalences 0.5 0.1
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, Executor, Future, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from run_simulation import (
    DTYPE,
    HYBRID_FLOOR,
    TRUE_AUC_GRID,
    WORKER_THREAD_ENV,
    Replicate,
    _band_record,
    _evaluate,
    _floor_region_record,
    _floor_split_record,
    calibrate_m3,
    get_dgp_specs,
    sample_configs_for,
    worker_pool,
)
from scipy.stats import norm
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from studroc_paper.datagen.roc_to_dgp import map_lhs_to_dgp
from studroc_paper.methods.fiducial_band_rs import fiducial_band_suite_rs
from studroc_paper.methods.m3_band_rs import m3_band_rs

# =============================================================================
# Design
# =============================================================================

METHOD_NAMES = ("m3", "fiducial", "fiducial_hybrid")

# Bin edges match the paper's shape × AUC coverage grid
# (``SHAPE_BINS`` and ``AUC_BINS`` in stats/paper_figures_nov11.py); draws are
# confined to the intersection of each bin with the main suite's LHS bounds.
SHAPE_EDGES: dict[str, tuple[str, list[float]]] = {
    "student_t": ("df", [1.0, 2.0, 4.0, 10.0, 30.0]),
    "hetero_gaussian": ("sigma_ratio", [0.2, 0.5, 1.0, 2.0, 5.0]),
}
AUC_EDGES = [0.5, 0.75, 0.9, 0.95, 1.0]

DEFAULT_SAMPLE_SIZES = (30, 100, 300, 1000, 3000, 10000)
PRECISION_TABLE_PP = (0.2, 0.5, 1.0, 2.0)
Z_95 = float(norm.ppf(0.975))

# Replicates simulated per worker task; small enough that the slowest
# (n = 10,000) tasks finish in a few minutes
CHUNK_SIZE = 10

# Seconds between checkpoints of finished chunks; a run killed without the
# chance to checkpoint loses at most this much work plus the chunks in flight
CHECKPOINT_SECONDS = 120


def reps_for_precision(*, half_width_pp: float, coverage: float) -> int:
    """Replicates per point for a 95% interval of the given half-width.

    Args:
        half_width_pp: Target half-width in percentage points.
        coverage: Planning coverage.

    Returns:
        The replicate count, rounded up.
    """
    h = half_width_pp / 100
    return math.ceil(Z_95**2 * coverage * (1 - coverage) / h**2)


@dataclass(frozen=True)
class Bin:
    """One (shape bin, AUC bin) stratum of a DGP, clipped to the LHS bounds.

    Attributes:
        shape_bin: Index of the shape bin in ``SHAPE_EDGES``.
        auc_bin: Index of the AUC bin in ``AUC_EDGES``.
        shape_range: Sampling range of the shape parameter.
        auc_range: Sampling range of the target AUC.
    """

    shape_bin: int
    auc_bin: int
    shape_range: tuple[float, float]
    auc_range: tuple[float, float]


def dgp_bins(dgp_type: str) -> list[Bin]:
    """Return a DGP's strata, each clipped to its LHS parameter bounds.

    Args:
        dgp_type: Key of ``SHAPE_EDGES``.

    Returns:
        Strata in (shape bin, AUC bin) order.
    """
    spec = get_dgp_specs()[dgp_type]
    bounds = dict(zip(spec["lhs_params"], spec["lhs_bounds"], strict=True))
    shape_name, shape_edges = SHAPE_EDGES[dgp_type]
    shape_lo, shape_hi = bounds[shape_name]
    auc_lo, auc_hi = bounds["auc"]
    return [
        Bin(
            shape_bin=s,
            auc_bin=a,
            shape_range=(
                max(shape_edges[s], shape_lo),
                min(shape_edges[s + 1], shape_hi),
            ),
            auc_range=(max(AUC_EDGES[a], auc_lo), min(AUC_EDGES[a + 1], auc_hi)),
        )
        for s in range(len(shape_edges) - 1)
        for a in range(len(AUC_EDGES) - 1)
    ]


@dataclass(frozen=True)
class StudyCell:
    """One (DGP, sample configuration) cell, saved as one file.

    Attributes:
        dgp_type: DGP name.
        sample_config: Sample configuration from ``sample_configs_for``.
        cell_key: Integers identifying the cell in the seed tree.
    """

    dgp_type: str
    sample_config: dict
    cell_key: tuple[int, ...]

    @property
    def stem(self) -> str:
        """File stem of the cell's results."""
        prev = round(self.sample_config["prevalence"] * 100)
        return f"{self.dgp_type}_n{self.sample_config['n_total']}_prev{prev}"


@dataclass(frozen=True)
class ChunkTask:
    """Consecutive replicates of one stratum of one cell.

    Attributes:
        cell: The study cell.
        stratum: The stratum.
        first_rep: Index of the chunk's first replicate within its point.
        n_reps: Replicates in the chunk.
        seed: Root seed of the study.
        alphas: Significance levels.
    """

    cell: StudyCell
    stratum: Bin
    first_rep: int
    n_reps: int
    seed: int
    alphas: tuple[float, ...]


def plan_tasks(
    *, cells: list[StudyCell], reps: int, seed: int, alphas: tuple[float, ...]
) -> dict[str, list[ChunkTask]]:
    """Split every cell's points into chunk tasks.

    Args:
        cells: Study cells.
        reps: Replicates per point.
        seed: Root seed.
        alphas: Significance levels.

    Returns:
        Tasks per cell, keyed by the cell's file stem.
    """
    return {
        cell.stem: [
            ChunkTask(
                cell=cell,
                stratum=stratum,
                first_rep=first,
                n_reps=min(CHUNK_SIZE, reps - first),
                seed=seed,
                alphas=alphas,
            )
            for stratum in dgp_bins(cell.dgp_type)
            for first in range(0, reps, CHUNK_SIZE)
        ]
        for cell in cells
    }


# =============================================================================
# Simulation
# =============================================================================


def draw_parameters(
    *, dgp_type: str, stratum: Bin, count: int, rng: np.random.Generator
) -> tuple[list[dict], dict[str, NDArray], int]:
    """Draw achievable DGP parameters uniformly within a stratum.

    Args:
        dgp_type: DGP name.
        stratum: The stratum.
        count: Parameter sets needed.
        rng: Random stream.

    Returns:
        Constructor arguments per draw, the drawn LHS-scale values, and the
        number of draws rejected as unachievable.
    """
    shape_name, _ = SHAPE_EDGES[dgp_type]
    z_lo, z_hi = norm.ppf(stratum.auc_range[0]), norm.ppf(stratum.auc_range[1])
    accepted: dict[str, list[float]] = {"auc": [], shape_name: []}
    params: list[dict] = []
    rejected = 0
    while len(params) < count:
        batch = 4 * (count - len(params))
        lhs = {
            "auc": norm.cdf(rng.uniform(z_lo, z_hi, batch)),
            shape_name: rng.uniform(*stratum.shape_range, batch),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            mapped = map_lhs_to_dgp(dgp_type, lhs)
        valid = np.ones(batch, dtype=bool)
        for value in mapped.values():
            if isinstance(value, np.ndarray) and value.ndim > 0:
                valid &= ~np.isnan(value)
        rejected += int((~valid).sum())
        for i in np.flatnonzero(valid)[: count - len(params)]:
            params.append(
                {
                    key: float(value[i])
                    if isinstance(value, np.ndarray) and value.ndim > 0
                    else value
                    for key, value in mapped.items()
                }
            )
            for name in accepted:
                accepted[name].append(float(lhs[name][i]))
    return params, {k: np.asarray(v) for k, v in accepted.items()}, rejected


def run_chunk(task: ChunkTask) -> list[dict]:
    """Worker entry point: simulate one chunk and evaluate its bands.

    Args:
        task: The chunk.

    Returns:
        One record per replicate, method, and alpha.
    """
    cell, stratum = task.cell, task.stratum
    config = cell.sample_config
    rng = np.random.default_rng(
        np.random.SeedSequence(
            task.seed,
            spawn_key=(
                *cell.cell_key,
                stratum.shape_bin,
                stratum.auc_bin,
                task.first_rep,
            ),
        )
    )
    params, lhs_values, rejected = draw_parameters(
        dgp_type=cell.dgp_type, stratum=stratum, count=task.n_reps, rng=rng
    )
    make_dgp = get_dgp_specs()[cell.dgp_type]["make_dgp"]
    fpr_grid = np.linspace(0, 1, config["n_neg"] + 1)
    y_true = np.concatenate([np.ones(config["n_pos"]), np.zeros(config["n_neg"])])
    records = []
    for i, dgp_params in enumerate(params):
        dgp = make_dgp(**dgp_params)
        scores_pos, scores_neg = dgp.sample(config["n_pos"], config["n_neg"], rng)
        method_seed = int(rng.integers(0, 2**63 - 1))
        rep_idx = task.first_rep + i
        rep = Replicate(
            y_true=y_true.astype(DTYPE),
            y_score=np.concatenate([scores_pos, scores_neg]).astype(DTYPE),
            fpr_grid=fpr_grid.astype(DTYPE),
            true_tpr=dgp.get_true_roc(fpr_grid),
            method_seed=method_seed,
            boot_seed=0,
            provenance=(
                f"dgp={cell.dgp_type}, params={dgp_params}, n_pos={config['n_pos']}, "
                f"n_neg={config['n_neg']}, shape_bin={stratum.shape_bin}, "
                f"auc_bin={stratum.auc_bin}, rep_idx={rep_idx}, seed={task.seed}, "
                f"method_seed={method_seed}"
            ),
        )
        base = {
            "dgp_type": cell.dgp_type,
            "n_pos": config["n_pos"],
            "n_neg": config["n_neg"],
            "n_total": config["n_total"],
            "prevalence": config["prevalence"],
            "shape_bin": stratum.shape_bin,
            "auc_bin": stratum.auc_bin,
            "rep_idx": rep_idx,
            # The chunk's rejections are booked on its first replicate, so the
            # column sums to the total over any set of whole chunks
            "rejected_draws": rejected if i == 0 else 0,
            "true_auc": float(
                np.trapezoid(dgp.get_true_roc(TRUE_AUC_GRID), TRUE_AUC_GRID)
            ),
            **{f"lhs_{name}": float(v[i]) for name, v in lhs_values.items()},
            **{f"dgp_{key}": value for key, value in dgp_params.items()},
            "device": "cpu",
        }
        records.extend(evaluate_replicate(rep=rep, alphas=task.alphas, base=base))
    return records


def evaluate_replicate(
    *, rep: Replicate, alphas: tuple[float, ...], base: dict
) -> list[dict]:
    """Compute M3 and both fiducial arms for one replicate and flatten them.

    Args:
        rep: The replicate.
        alphas: Significance levels.
        base: Replicate-level columns shared by every record.

    Returns:
        One record per method and alpha.
    """
    records = []
    t_start = time.perf_counter()
    suite = fiducial_band_suite_rs(
        y_true=rep.y_true,
        y_score=rep.y_score,
        alphas=list(alphas),
        m3_floors={"fiducial": False, "fiducial_hybrid": HYBRID_FLOOR},
        random_state=rep.method_seed,
    )
    suite_seconds = time.perf_counter() - t_start
    for alpha in alphas:
        region = suite.floor_regions[alpha]["fiducial_hybrid"]
        t_start = time.perf_counter()
        _, m3_lower, m3_upper = m3_band_rs(
            y_true=rep.y_true,
            y_score=rep.y_score,
            alpha=alpha,
            random_state=rep.method_seed,
        )
        bands = {
            "m3": (m3_lower, m3_upper, time.perf_counter() - t_start, False),
            **{
                name: (lower, upper, suite_seconds, True)
                for name, (lower, upper) in suite.bands[alpha].items()
            },
        }
        for method in METHOD_NAMES:
            lower, upper, seconds, shared = bands[method]
            band = _evaluate(
                rep=rep, method=method, alpha=alpha, lower=lower, upper=upper
            )
            records.append(
                {
                    **base,
                    "method": method,
                    "alpha": alpha,
                    "confidence_level": 1 - alpha,
                    "runtime_seconds": seconds,
                    "runtime_is_shared": shared,
                    **_floor_region_record(region),
                    **_band_record(band),
                    **_floor_split_record(band, region),
                }
            )
    return records


# =============================================================================
# Planning
# =============================================================================


def time_configs(
    *, sample_configs: list[dict], alphas: tuple[float, ...], trials: int = 3
) -> dict[int, float]:
    """Time one replicate of the three methods per sample configuration.

    Runs single-threaded in this process, as the workers do, on the
    heteroscedastic Gaussian at a high-AUC, wide-positives stratum.

    Args:
        sample_configs: Sample configurations.
        alphas: Significance levels.
        trials: Replicates timed per configuration; the median is used.

    Returns:
        Seconds per replicate by configuration index.
    """
    os.environ["RAYON_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    stratum = dgp_bins("hetero_gaussian")[-1]
    seconds = {}
    with threadpool_limits(limits=1):
        for k, config in enumerate(sample_configs):
            cell = StudyCell(
                dgp_type="hetero_gaussian", sample_config=config, cell_key=(k,)
            )
            times = []
            for trial in range(trials):
                start = time.perf_counter()
                run_chunk(
                    ChunkTask(
                        cell=cell,
                        stratum=stratum,
                        first_rep=trial,
                        n_reps=1,
                        seed=0,
                        alphas=alphas,
                    )
                )
                times.append(time.perf_counter() - start)
            seconds[k] = float(np.median(times))
    return seconds


def print_plan(
    *,
    dgp_types: list[str],
    sample_configs: list[dict],
    alphas: tuple[float, ...],
    coverage: float,
    workers: int,
) -> None:
    """Print replicates and estimated wall time for each target precision.

    Args:
        dgp_types: DGPs to run.
        sample_configs: Sample configurations.
        alphas: Significance levels.
        coverage: Planning coverage.
        workers: Worker processes.
    """
    print("Timing one replicate per sample configuration (single-threaded)...")
    seconds = time_configs(sample_configs=sample_configs, alphas=alphas)
    for k, config in enumerate(sample_configs):
        print(
            f"  n = {config['n_total']:>6,}, prevalence {config['prevalence']:.2f}: "
            f"{seconds[k]:.3f} s"
        )
    strata = sum(len(dgp_bins(d)) for d in dgp_types)
    per_rep = strata * sum(seconds.values())
    points = strata * len(sample_configs)
    print(
        f"\n{points} points ({strata} strata × {len(sample_configs)} sample "
        f"configurations), {workers} workers, planning coverage {coverage:.3f}"
    )
    print(f"{'±pp':>6} {'reps/point':>11} {'replicates':>12} {'est. hours':>11}")
    for pp in PRECISION_TABLE_PP:
        reps = reps_for_precision(half_width_pp=pp, coverage=coverage)
        hours = reps * per_rep / workers / 3600
        print(f"{pp:>6.1f} {reps:>11,} {reps * points:>12,} {hours:>11.1f}")
    print(
        "\nHalf-widths are for a 95% interval at the planning coverage; "
        "a point covering less (e.g. 0.80) is wider for the same replicates."
    )


# =============================================================================
# Orchestration and saving
# =============================================================================


def study_design(*, reps: int, seed: int, alphas: tuple[float, ...]) -> dict:
    """Return everything a cell's results depend on besides the cell itself.

    Args:
        reps: Replicates per point.
        seed: Root seed.
        alphas: Significance levels.

    Returns:
        The design, in the JSON form stored as ``design.json``.
    """
    design = {
        "reps": reps,
        "seed": seed,
        "alphas": list(alphas),
        "chunk_size": CHUNK_SIZE,
        "strata": {d: [vars(b) for b in dgp_bins(d)] for d in SHAPE_EDGES},
    }
    return json.loads(json.dumps(design))


def claim_output_dir(*, output_dir: Path, design: dict) -> None:
    """Record the design in an output directory, or check it matches.

    Args:
        output_dir: Output directory.
        design: Design from ``study_design``.

    Raises:
        ValueError: If the directory holds results of a different design, or
            results without a recorded design.
    """
    path = output_dir / "design.json"
    if path.exists():
        stored = json.loads(path.read_text())
        differing = sorted(
            k for k in design.keys() | stored.keys() if stored.get(k) != design.get(k)
        )
        if differing:
            raise ValueError(
                f"{output_dir} holds a run with a different design "
                f"(differs in: {', '.join(differing)}); use another --output-dir "
                "or rerun with the original arguments"
            )
        return
    if (
        any(output_dir.glob("*_individual.feather"))
        or (output_dir / "partial").exists()
    ):
        raise ValueError(
            f"{output_dir} holds results without a design.json, so they cannot "
            "be resumed safely; use another --output-dir"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(design, indent=2))


def partial_dir(*, output_dir: Path, stem: str) -> Path:
    """Directory of a cell's checkpointed chunks."""
    return output_dir / "partial" / stem


def checkpointed_chunks(*, output_dir: Path, stem: str) -> set[tuple[int, int, int]]:
    """Return the chunks of a cell already checkpointed.

    Args:
        output_dir: Output directory.
        stem: The cell's file stem.

    Returns:
        (shape bin, AUC bin, first replicate) of every checkpointed chunk.
    """
    keys: set[tuple[int, int, int]] = set()
    for part in partial_dir(output_dir=output_dir, stem=stem).glob("part_*.feather"):
        frame = pd.read_feather(part, columns=["shape_bin", "auc_bin", "rep_idx"])
        first = frame["rep_idx"] - frame["rep_idx"] % CHUNK_SIZE
        keys.update(
            zip(
                frame["shape_bin"].tolist(),
                frame["auc_bin"].tolist(),
                first.tolist(),
                strict=True,
            )
        )
    return keys


def chunk_key(task: ChunkTask) -> tuple[int, int, int]:
    """(shape bin, AUC bin, first replicate) identifying a chunk in its cell."""
    return (task.stratum.shape_bin, task.stratum.auc_bin, task.first_rep)


def write_atomic(*, frame: pd.DataFrame, path: Path) -> None:
    """Write a feather through a temporary file so readers never see a partial one.

    Args:
        frame: Frame to write.
        path: Destination.
    """
    tmp = path.with_suffix(".tmp")
    frame.to_feather(tmp)
    tmp.replace(path)


def checkpoint(*, stem: str, records: list[dict], output_dir: Path) -> None:
    """Append finished chunks of a cell to its checkpoints.

    Args:
        stem: The cell's file stem.
        records: Records of whole chunks.
        output_dir: Output directory.
    """
    directory = partial_dir(output_dir=output_dir, stem=stem)
    directory.mkdir(parents=True, exist_ok=True)
    write_atomic(
        frame=pd.DataFrame.from_records(records),
        path=directory / f"part_{time.time_ns()}.feather",
    )


def finalize_cell(*, stem: str, output_dir: Path) -> None:
    """Merge a complete cell's checkpoints into its long-format result file.

    Args:
        stem: The cell's file stem.
        output_dir: Output directory.
    """
    directory = partial_dir(output_dir=output_dir, stem=stem)
    frame = pd.concat(
        (pd.read_feather(p) for p in sorted(directory.glob("part_*.feather"))),
        ignore_index=True,
    ).sort_values(
        ["shape_bin", "auc_bin", "rep_idx", "method", "alpha"], ignore_index=True
    )
    write_atomic(frame=frame, path=output_dir / f"{stem}_individual.feather")
    shutil.rmtree(directory)
    if not any(directory.parent.iterdir()):
        directory.parent.rmdir()


def pending_cells(*, cells: list[StudyCell], output_dir: Path) -> list[StudyCell]:
    """Return the cells without a result file, clearing stale checkpoints.

    Args:
        cells: Study cells.
        output_dir: Output directory.

    Returns:
        Cells still to run, in the given order.
    """
    pending = []
    for cell in cells:
        if (output_dir / f"{cell.stem}_individual.feather").exists():
            # Checkpoints left by an exit between saving a cell and clearing them
            shutil.rmtree(
                partial_dir(output_dir=output_dir, stem=cell.stem), ignore_errors=True
            )
        else:
            pending.append(cell)
    return pending


def run_study(
    *, tasks: dict[str, list[ChunkTask]], pool: Executor, output_dir: Path, workers: int
) -> None:
    """Run every chunk not yet checkpointed, saving each cell once complete.

    Chunks are submitted largest sample size first, so the slowest work
    starts immediately, with a bounded number in flight. Finished chunks are
    checkpointed every ``CHECKPOINT_SECONDS`` and whenever the loop exits,
    including on Ctrl-C; queued chunks are cancelled on the way out.

    Args:
        tasks: Chunk tasks per cell, keyed by the cell's file stem.
        pool: Worker pool.
        output_dir: Output directory.
        workers: Worker processes, which bounds the in-flight queue.
    """
    pending: dict[str, list[ChunkTask]] = {}
    for stem, cell_tasks in tasks.items():
        saved = checkpointed_chunks(output_dir=output_dir, stem=stem)
        pending[stem] = [t for t in cell_tasks if chunk_key(t) not in saved]
        if not pending[stem]:
            finalize_cell(stem=stem, output_dir=output_dir)
            tqdm.write(f"  Saved: {stem}")
    queue = sorted(
        (task for cell_tasks in pending.values() for task in cell_tasks),
        key=lambda t: -t.cell.sample_config["n_neg"],
    )
    total = sum(len(cell_tasks) for cell_tasks in tasks.values())
    remaining = {stem: len(cell_tasks) for stem, cell_tasks in pending.items()}
    buffers: dict[str, list[dict]] = {stem: [] for stem in pending}
    in_flight: dict[Future, ChunkTask] = {}
    progress = tqdm(total=total, initial=total - len(queue), desc="Chunks")

    def flush() -> None:
        for stem, records in buffers.items():
            if records:
                checkpoint(stem=stem, records=records, output_dir=output_dir)
                buffers[stem] = []

    last_checkpoint = time.monotonic()
    position = 0
    try:
        while position < len(queue) or in_flight:
            while position < len(queue) and len(in_flight) < 4 * workers:
                task = queue[position]
                in_flight[pool.submit(run_chunk, task)] = task
                position += 1
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                stem = in_flight.pop(future).cell.stem
                buffers[stem].extend(future.result())
                remaining[stem] -= 1
                progress.update()
                if remaining[stem] == 0:
                    flush()
                    finalize_cell(stem=stem, output_dir=output_dir)
                    tqdm.write(f"  Saved: {stem}")
            if time.monotonic() - last_checkpoint >= CHECKPOINT_SECONDS:
                flush()
                last_checkpoint = time.monotonic()
    except BaseException:
        for future in in_flight:
            future.cancel()
        raise
    finally:
        flush()
        progress.close()


def write_metadata(*, args: argparse.Namespace, reps: int, output_dir: Path) -> None:
    """Write the run's design and provenance next to its results.

    Args:
        args: Parsed command-line arguments.
        reps: Replicates per point.
        output_dir: Output directory.
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
        "reps_per_point": reps,
        "methods": list(METHOD_NAMES),
        "shape_edges": {d: edges for d, (_, edges) in SHAPE_EDGES.items()},
        "auc_edges": AUC_EDGES,
        "strata": {d: [vars(b) for b in dgp_bins(d)] for d in args.dgps},
        "git_hash": git_hash,
    }
    path = output_dir / f"run_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(metadata, indent=2))


def main() -> None:
    """Parse arguments and run or plan the stratified study."""
    parser = argparse.ArgumentParser(
        description="Stratified shape × AUC coverage study of the fiducial bands",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--reps", type=int, help="Replicates per point")
    size.add_argument(
        "--precision-pp",
        type=float,
        default=1.0,
        help="Target 95%% half-width in percentage points; sets --reps",
    )
    parser.add_argument(
        "--planning-coverage",
        type=float,
        default=0.95,
        help="Coverage at which --precision-pp is computed",
    )
    parser.add_argument(
        "--plan", action="store_true", help="Print the precision plan and exit"
    )
    parser.add_argument(
        "--dgps", nargs="+", choices=list(SHAPE_EDGES), default=list(SHAPE_EDGES)
    )
    parser.add_argument(
        "--sample-sizes", type=int, nargs="+", default=list(DEFAULT_SAMPLE_SIZES)
    )
    parser.add_argument("--prevalences", type=float, nargs="+", default=[0.5])
    parser.add_argument(
        "--confidence-levels",
        type=float,
        nargs="+",
        default=[0.05],
        help="Alpha values",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/results/shape_auc_study")
    )
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    if not all(0 < p < 1 for p in args.prevalences):
        parser.error("--prevalences must lie strictly between 0 and 1")
    try:
        sample_configs = [
            config
            for n_total in args.sample_sizes
            for config in sample_configs_for(
                n_total=n_total, prevalences=args.prevalences
            )
        ]
    except ValueError as err:
        parser.error(str(err))
    alphas = tuple(args.confidence_levels)

    if args.plan:
        print_plan(
            dgp_types=args.dgps,
            sample_configs=sample_configs,
            alphas=alphas,
            coverage=args.planning_coverage,
            workers=args.workers,
        )
        return

    reps = args.reps or reps_for_precision(
        half_width_pp=args.precision_pp, coverage=args.planning_coverage
    )
    output_dir = args.output_dir
    try:
        claim_output_dir(
            output_dir=output_dir,
            design=study_design(reps=reps, seed=args.seed, alphas=alphas),
        )
    except ValueError as err:
        parser.error(str(err))
    cells = [
        StudyCell(
            dgp_type=dgp,
            sample_config=config,
            cell_key=(d, config["n_total"], round(config["prevalence"] * 10000)),
        )
        for d, dgp in enumerate(SHAPE_EDGES)
        if dgp in args.dgps
        for config in sample_configs
    ]
    pending = pending_cells(cells=cells, output_dir=output_dir)
    resumed = sum(
        partial_dir(output_dir=output_dir, stem=c.stem).exists() for c in pending
    )
    print(
        f"{len(cells)} cells × {len(dgp_bins(args.dgps[0]))} strata, {reps:,} "
        f"replicates per point; {len(cells) - len(pending)} cells already saved, "
        f"{resumed} resuming from checkpoints"
    )
    if not pending:
        return
    write_metadata(args=args, reps=reps, output_dir=output_dir)
    os.environ.update(WORKER_THREAD_ENV)
    m3_cache = calibrate_m3(
        sample_configs=sample_configs, alphas=list(alphas), workers=args.workers
    )
    tasks = plan_tasks(cells=pending, reps=reps, seed=args.seed, alphas=alphas)
    try:
        with worker_pool(workers=args.workers, m3_cache=m3_cache) as pool:
            run_study(
                tasks=tasks, pool=pool, output_dir=output_dir, workers=args.workers
            )
    except KeyboardInterrupt:
        print(
            "\nInterrupted; finished chunks are checkpointed. "
            "Rerun the same command to resume."
        )
        sys.exit(130)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
