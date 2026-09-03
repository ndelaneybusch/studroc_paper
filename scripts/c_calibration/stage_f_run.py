"""Resumable paired execution engine for Stage F.

The runner consumes a study manifest. Each replicate draws one
tie-resolved rank ordering and one fiducial cloud; all fiducial levels are
read from that cloud, and all M3/hybrid arms are paired to the same ordering.
Hybrid rules are scored offline from lossless parent-band records.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage_f_analysis import (  # noqa: E402
    load_complete_study,
    score_record,
    wilson_interval,
    write_study_summary,
)
from stage_f_core import (  # noqa: E402
    ObservableSummary,
    build_record,
    decode_violations,
    empirical_observables,
)
from stage_f_design import (  # noqa: E402
    DEFAULT_OUT,
    PRIMARY_ALPHA,
    StageFCell,
    cell_curve,
    load_manifest,
    sample_labels,
    write_manifests,
)

from studroc_paper.methods.fiducial_band_rs import (  # noqa: E402
    _apply_corner_allowances,
)
from studroc_paper.methods.fiducial_ladder import ladder_profile  # noqa: E402
from studroc_paper.methods.m3_band_rs import _m3_band_from_labels_rs  # noqa: E402

CELL_SCHEMA = "stage-f-cell/v2"
TOPUP_BATCH = 400
CHECKPOINT_BATCH = 100
BAR = 0.94
COMPOSITE_EXPONENT = 2.5


def _level_key(value: float) -> str:
    """Return the canonical JSON key for a nominal level."""
    return f"{value:g}"


def _fiducial_bands(
    labels: np.ndarray,
    *,
    cell: StageFCell,
    truth: np.ndarray,
    observables: ObservableSummary,
    khat: np.ndarray,
    seed: int,
    n_threads: int,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray] | None
]:
    """Evaluate every requested fiducial level from a single Rust cloud."""
    specifications = [(alpha, 1.0) for alpha in cell.alphas]
    if cell.study == "B":
        specifications.append((PRIMARY_ALPHA, COMPOSITE_EXPONENT))
    alpha_effs = tuple(
        1.0 - (1.0 - alpha) ** exponent for alpha, exponent in specifications
    )
    profile = ladder_profile(
        labels,
        rtrue=truth,
        n_draws=cell.m_draws,
        seed=seed,
        ladder=np.array([1], dtype=np.int64),
        alpha_effs=alpha_effs,
        trim_rows="production",
        return_edges=True,
        n_threads=n_threads,
    )
    if profile.edges is None:
        raise RuntimeError(
            "Stage F requested edges but the ladder kernel returned none"
        )
    depths, raw_lower, raw_upper = profile.edges
    depth_indices = {int(depth): index for index, depth in enumerate(depths)}
    bands = {}
    composite = None
    for index, (alpha, exponent) in enumerate(specifications):
        depth = int(profile.ref_j[index])
        edge_index = depth_indices[depth]
        band = _apply_corner_allowances(
            lower=raw_lower[edge_index],
            upper=raw_upper[edge_index],
            khat=khat,
            n1=observables.n1,
            trim_depth=depth,
            n_draws=cell.m_draws,
        )
        if exponent == 1.0:
            bands[_level_key(alpha)] = band
        else:
            composite = band
    return bands, composite


def run_replicate(cell: StageFCell, rep: int, *, n_threads: int) -> dict:
    """Generate one lossless paired Stage F replicate record."""
    sample = sample_labels(cell, rep)
    labels = sample.labels
    grid = np.arange(cell.n_grid, dtype=np.float64) / cell.n0
    truth = np.clip(cell_curve(cell).eval(grid), 0.0, 1.0)
    observables, khat = empirical_observables(labels)
    fiducial, composite = _fiducial_bands(
        labels,
        cell=cell,
        truth=truth,
        observables=observables,
        khat=khat,
        seed=sample.cloud_seed,
        n_threads=n_threads,
    )
    m3_levels = sorted(
        {alpha for nominal in cell.alphas for alpha in (nominal, nominal / 2.0)}
    )
    m3 = {
        _level_key(alpha): _m3_band_from_labels_rs(
            labels, alpha=alpha, split_ratio=0.5, assume_r0_zero=False
        )[1:]
        for alpha in m3_levels
    }
    record = build_record(
        observables=observables, khat=khat, truth=truth, fiducial=fiducial, m3=m3
    )
    record["rep"] = rep
    record["simulation_diagnostics"] = sample.diagnostics
    if composite is not None:
        from stage_f_core import encode_array

        record["composite_c2.5"] = {
            "lower": encode_array(composite[0]),
            "upper": encode_array(composite[1]),
        }
    return record


def cell_path(root: Path, cell: StageFCell) -> Path:
    """Return a cell's compressed record path."""
    return root / cell.study / f"{cell.name}.json.gz"


def cell_metadata(cell: StageFCell) -> dict:
    """Return a cell definition in the form stored by JSON."""
    return json.loads(json.dumps(asdict(cell)))


def load_existing(
    path: Path, *, expected_cell: StageFCell
) -> tuple[list[dict], dict | None]:
    """Load resumable records for the requested cell."""
    if not path.exists():
        return [], None
    with gzip.open(path, "rt") as handle:
        payload = json.load(handle)
    if payload.get("schema") != CELL_SCHEMA:
        raise RuntimeError(f"Unknown Stage F cell schema in {path}")
    if payload["meta"].get("cell") != cell_metadata(expected_cell):
        raise RuntimeError(f"Existing output does not match cell {expected_cell.name}")
    records = payload["records"]
    if [row.get("rep") for row in records] != list(range(len(records))):
        raise RuntimeError(f"Non-contiguous replicate sequence in {path}")
    return records, payload["meta"]


def _write_cell(path: Path, *, meta: dict, records: list[dict]) -> None:
    """Atomically replace one compressed cell checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", compresslevel=6) as handle:
        json.dump({"schema": CELL_SCHEMA, "meta": meta, "records": records}, handle)
    temporary.replace(path)


def check_replay_parity(cell: StageFCell, records: list[dict]) -> None:
    """Refuse a replay whose C=1 coverage differs by over three combined SEs."""
    if cell.prior_coverage is None or cell.prior_reps is None:
        return
    actual = np.mean(
        [
            not decode_violations(record["fiducial_violations"]["0.05"]).any()
            for record in records
        ]
    )
    previous = cell.prior_coverage
    standard_error = np.sqrt(
        actual * (1.0 - actual) / len(records)
        + previous * (1.0 - previous) / cell.prior_reps
    )
    tolerance = 3.0 * max(float(standard_error), 1.0 / max(len(records), 1))
    if abs(float(actual) - previous) > tolerance:
        raise RuntimeError(
            f"Replay parity failed for {cell.name}: prior={previous:.4f}, "
            f"replay={actual:.4f}, tolerance={tolerance:.4f}"
        )


def needs_topup(records: list[dict], *, m_draws: int) -> bool:
    """Return whether any prespecified alpha=.05 floor arm straddles .94."""
    for rule in (
        "probe_fpr",
        "count5",
        "frontier_run0",
        "frontier_j1",
        "frontier_floor_v1",
    ):
        for alpha2_key in ("0.05", "0.025"):
            successes = sum(
                score_record(
                    record,
                    rule=rule,
                    alpha=PRIMARY_ALPHA,
                    alpha2_key=alpha2_key,
                    m_draws=m_draws,
                )["covered"]
                for record in records
            )
            lower, upper = wilson_interval(successes, len(records))
            if lower < BAR < upper:
                return True
    return False


def run_cell(
    cell: StageFCell,
    *,
    root: Path,
    workers: int,
    threads_per_call: int,
    mem_gb: float,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    """Run or resume one cell, including the external-study top-up rule."""
    path = cell_path(root, cell)
    if force:
        records, previous = [], None
    else:
        records, previous = load_existing(path, expected_cell=cell)
    cloud_gb = 4 * cell.m_draws * cell.n_grid / 2**30
    effective_workers = max(1, min(workers, int(mem_gb / max(cloud_gb, 1e-12))))
    started = time.time()
    previous_runtime = float((previous or {}).get("runtime_s", 0.0))

    def checkpoint() -> dict:
        """Persist all completed records as an atomic resumable checkpoint."""
        meta = {
            "cell": cell_metadata(cell),
            "runtime_s": round(time.time() - started, 1) + previous_runtime,
        }
        _write_cell(path, meta=meta, records=records)
        return meta

    def extend(target: int) -> None:
        """Generate and checkpoint records through an exclusive target."""
        while len(records) < target:
            stop = min(len(records) + CHECKPOINT_BATCH, target)
            pending = list(range(len(records), stop))
            with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                futures = [
                    pool.submit(run_replicate, cell, rep, n_threads=threads_per_call)
                    for rep in pending
                ]
                records.extend(future.result() for future in futures)
            checkpoint()
            if verbose and len(records) % 50 == 0:
                print(f"  [{cell.name}] {len(records)}/{target} replicates", flush=True)

    extend(cell.reps)
    if cell.study in {"B", "C"}:
        while len(records) < cell.reps_max and needs_topup(
            records, m_draws=cell.m_draws
        ):
            extend(min(len(records) + TOPUP_BATCH, cell.reps_max))

    check_replay_parity(cell, records)
    meta = checkpoint()
    if verbose:
        print(f"[{cell.name}] stored {len(records)} paired replicates", flush=True)
    return meta


def run_study(
    *,
    manifest_path: Path,
    root: Path,
    workers: int,
    threads_per_call: int,
    mem_gb: float,
    select: str | None = None,
    force: bool = False,
) -> None:
    """Execute every selected cell in a study manifest."""
    _, cells = load_manifest(manifest_path)
    if select:
        cells = [cell for cell in cells if select in cell.name]
    for index, cell in enumerate(cells, start=1):
        print(f"[{index}/{len(cells)}] {cell.name}", flush=True)
        run_cell(
            cell,
            root=root,
            workers=workers,
            threads_per_call=threads_per_call,
            mem_gb=mem_gb,
            force=force,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the Stage F command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=("design", "run", "summarize"))
    parser.add_argument("--study", choices=("A", "B", "C"))
    parser.add_argument("--root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 8) // 4)
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--mem-gb", type=float, default=40.0)
    parser.add_argument("--select")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Stage F CLI entry point."""
    args = parse_args(argv)
    if args.action == "design":
        kwargs = {"out_dir": args.root}
        if args.source_root is not None:
            kwargs["source_root"] = args.source_root
        paths = write_manifests(**kwargs)
        for item, path in paths.items():
            print(f"{item}: {path}")
        return 0
    if args.action == "summarize":
        if args.study not in {"A", "B", "C"}:
            raise ValueError("summarize requires --study A|B|C")
        cells = load_complete_study(args.root, study=args.study)
        write_study_summary(
            cells,
            path=args.root / "analysis" / f"study_{args.study.lower()}_summary.json",
        )
        return 0
    if args.manifest is None:
        raise ValueError("run requires --manifest")
    _, cells = load_manifest(args.manifest)
    if args.dry_run:
        for cell in cells:
            if args.select is None or args.select in cell.name:
                print(
                    f"{cell.name}: n={cell.n0}x{cell.n1}, M={cell.m_draws}, "
                    f"reps={cell.reps}-{cell.reps_max}"
                )
        return 0
    run_study(
        manifest_path=args.manifest,
        root=args.root,
        workers=args.workers,
        threads_per_call=args.threads,
        mem_gb=args.mem_gb,
        select=args.select,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
