"""Resumable paired execution engine for Stage F.

The runner consumes previously frozen manifests.  Each replicate draws one
tie-resolved rank ordering and one fiducial cloud; all fiducial levels are
read from that cloud, and all M3/hybrid arms are paired to the same ordering.
Hybrid rules are scored offline from lossless parent-band records.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
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
    write_external_summary,
)
from stage_f_core import (  # noqa: E402
    ObservableSummary,
    RegionArtifact,
    build_record,
    decode_violations,
    empirical_observables,
    load_artifact,
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

CELL_SCHEMA = "stage-f-cell/v1"
TOPUP_BATCH = 400
CHECKPOINT_BATCH = 100
BAR = 0.94
COMPOSITE_EXPONENT = 2.5
CODE_FILES = (
    "scripts/c_calibration/stage_f_core.py",
    "scripts/c_calibration/stage_f_design.py",
    "scripts/c_calibration/stage_f_analysis.py",
    "scripts/c_calibration/stage_f_run.py",
    "src/studroc_paper/methods/fiducial_band_rs.py",
    "src/studroc_paper/methods/fiducial_ladder.py",
    "src/studroc_paper/methods/m3_band_rs.py",
)


def code_fingerprint(root: Path | None = None) -> str:
    """Hash every implementation file that can change stored Stage F values."""
    root = root or Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for relative in CODE_FILES:
        path = root / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_hash(root: Path | None = None) -> str:
    """Return the checked-out commit hash for output provenance."""
    root = root or Path(__file__).resolve().parents[2]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


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
    labels, seed = sample_labels(cell, rep)
    grid = np.arange(cell.n_grid, dtype=np.float64) / cell.n0
    truth = np.clip(cell_curve(cell).eval(grid), 0.0, 1.0)
    observables, khat = empirical_observables(labels)
    fiducial, composite = _fiducial_bands(
        labels,
        cell=cell,
        truth=truth,
        observables=observables,
        khat=khat,
        seed=seed,
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
    if composite is not None:
        from stage_f_core import encode_array

        record["composite_c2.5"] = {
            "lower": encode_array(composite[0]),
            "upper": encode_array(composite[1]),
        }
    return record


def compatibility_payload(
    *, cell: StageFCell, manifest: dict, artifact: RegionArtifact | None
) -> dict:
    """Return all constants that must match before records may be resumed."""
    return {
        "manifest_hash": manifest["content_hash"],
        "rule_artifact_hash": artifact.content_hash if artifact else None,
        "code_fingerprint": code_fingerprint(),
        "cell": asdict(cell),
        "m3_split_ratio": 0.5,
        "tie_break": "random",
        "trim_grid": "production",
        "alpha_grid": list(cell.alphas),
        "auc_delta": 0.05,
        "composite_exponent": COMPOSITE_EXPONENT if cell.study == "B" else None,
    }


def cell_path(root: Path, cell: StageFCell) -> Path:
    """Return a cell's compressed record path."""
    return root / cell.study / f"{cell.name}.json.gz"


def load_existing(
    path: Path, *, expected_compatibility: dict
) -> tuple[list[dict], dict | None]:
    """Load resumable records, refusing every provenance mismatch."""
    if not path.exists():
        return [], None
    with gzip.open(path, "rt") as handle:
        payload = json.load(handle)
    if payload.get("schema") != CELL_SCHEMA:
        raise RuntimeError(f"Unknown Stage F cell schema in {path}")
    actual = payload["meta"].get("compatibility")
    if actual != expected_compatibility:
        raise RuntimeError(
            f"Refusing to mix {path} with different manifest, rule, code, "
            "or design constants"
        )
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


def needs_topup(records: list[dict], *, artifact: RegionArtifact) -> bool:
    """Return whether any prespecified alpha=.05 floor arm straddles .94."""
    for rule in ("probe_fpr", "count5", "stage_f_v1"):
        for alpha2_key in ("0.05", "0.025"):
            successes = sum(
                score_record(
                    record,
                    rule=rule,
                    alpha=PRIMARY_ALPHA,
                    alpha2_key=alpha2_key,
                    artifact=artifact,
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
    manifest: dict,
    root: Path,
    artifact: RegionArtifact | None,
    workers: int,
    threads_per_call: int,
    mem_gb: float,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    """Run or resume one cell, including the external-study top-up rule."""
    compatibility = compatibility_payload(
        cell=cell, manifest=manifest, artifact=artifact
    )
    path = cell_path(root, cell)
    if force:
        records, previous = [], None
    else:
        records, previous = load_existing(path, expected_compatibility=compatibility)
    cloud_gb = 4 * cell.m_draws * cell.n_grid / 2**30
    effective_workers = max(1, min(workers, int(mem_gb / max(cloud_gb, 1e-12))))
    started = time.time()
    previous_runtime = float((previous or {}).get("runtime_s", 0.0))

    def checkpoint() -> dict:
        """Persist all completed records as an atomic resumable checkpoint."""
        meta = {
            "cell": asdict(cell),
            "compatibility": compatibility,
            "reps_done": len(records),
            "runtime_s": round(time.time() - started, 1) + previous_runtime,
            "git_hash": git_hash(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
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
        if artifact is None:
            raise ValueError("Studies B/C require the frozen stage_f_v1 artifact")
        while len(records) < cell.reps_max and needs_topup(records, artifact=artifact):
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
    artifact_path: Path | None,
    workers: int,
    threads_per_call: int,
    mem_gb: float,
    select: str | None = None,
    force: bool = False,
) -> None:
    """Execute every selected cell in one already-frozen manifest."""
    manifest, cells = load_manifest(manifest_path)
    artifact = load_artifact(artifact_path) if artifact_path else None
    if manifest["study"] in {"B", "C"} and artifact is None:
        raise ValueError("Studies B/C require --artifact rules/stage_f_v1.json")
    if (
        manifest["study"] in {"B", "C"}
        and artifact is not None
        and artifact.training.get("phase") != "refit"
    ):
        raise ValueError("Studies B/C require the final refit Stage F artifact")
    if select:
        cells = [cell for cell in cells if select in cell.name]
    for index, cell in enumerate(cells, start=1):
        print(f"[{index}/{len(cells)}] {cell.name}", flush=True)
        run_cell(
            cell,
            manifest=manifest,
            root=root,
            artifact=artifact,
            workers=workers,
            threads_per_call=threads_per_call,
            mem_gb=mem_gb,
            force=force,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the Stage F command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=("manifests", "run", "fit", "summarize"))
    parser.add_argument("--study", choices=("A", "B", "C"))
    parser.add_argument("--root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--artifact", type=Path)
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
    if args.action == "manifests":
        kwargs = {"out_dir": args.root}
        if args.source_root is not None:
            kwargs["source_root"] = args.source_root
        paths = write_manifests(**kwargs)
        for study, path in paths.items():
            print(f"Study {study}: {path}")
        return 0
    if args.action == "fit":
        from stage_f_analysis import default_stage_a_fit

        artifact = default_stage_a_fit(args.root)
        print(f"froze {artifact.rule_id}: {artifact.content_hash}")
        return 0
    if args.action == "summarize":
        if args.study not in {"B", "C"} or args.artifact is None:
            parser_error = "summarize requires --study B|C and --artifact"
            raise ValueError(parser_error)
        artifact = load_artifact(args.artifact)
        if artifact.training.get("phase") != "refit":
            raise ValueError("summarize requires the final refit Stage F artifact")
        cells = load_complete_study(args.root, study=args.study)
        write_external_summary(
            cells,
            artifact=artifact,
            path=args.root / "analysis" / f"study_{args.study.lower()}_summary.json",
        )
        return 0
    if args.study is None or args.manifest is None:
        raise ValueError("run requires --study and --manifest")
    manifest, cells = load_manifest(args.manifest)
    if manifest["study"] != args.study:
        raise ValueError("--study does not match the supplied manifest")
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
        artifact_path=args.artifact,
        workers=args.workers,
        threads_per_call=args.threads,
        mem_gb=args.mem_gb,
        select=args.select,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
