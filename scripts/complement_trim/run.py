"""Resumable runner for the complement-trim experiment.

Usage, from the repository root::

    uv run --no-sync python -m scripts.complement_trim.run design  --out DIR
    uv run --no-sync python -m scripts.complement_trim.run dry-run --out DIR
    uv run --no-sync python -m scripts.complement_trim.run run     --out DIR \
        [--cells SUBSTR ...] [--workers 4] [--threads 4] [--reps-scale 1.0]
    uv run --no-sync python -m scripts.complement_trim.run summarize --out DIR

Spec: ``stats/experiments/complement_trim_spec.md``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from scripts.complement_trim.core import ARMS, REFERENCES, run_replicate
from scripts.complement_trim.design import (
    ALPHAS,
    TOPUP_BATCH,
    Cell,
    all_cells,
    sample_replicate,
    select_cells,
    with_reps,
)
from studroc_paper.methods.fiducial_band import _auto_n_draws

SCHEMA = "complement-trim-cell/v1"
CHECKPOINT = 50
BAR = 0.94
TOPUP_ARMS = ("comp", "comp_budget")
DEFAULT_OUT = Path("data/results/complement_trim_20260926")


def wilson(successes: int, count: int, z: float = 1.959964) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial proportion."""
    if count == 0:
        return (0.0, 1.0)
    p = successes / count
    centre = (p + z * z / (2 * count)) / (1 + z * z / count)
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count))
    return (centre - half / (1 + z * z / count), centre + half / (1 + z * z / count))


def _cell_path(root: Path, cell: Cell) -> Path:
    """Return the record file of one cell."""
    return root / "cells" / f"{cell.name}.json.gz"


def _load(path: Path, meta: dict) -> list[dict]:
    """Load existing records, refusing to mix a changed cell definition."""
    if not path.exists():
        return []
    with gzip.open(path, "rt") as handle:
        payload = json.load(handle)
    if payload["meta"] != meta:
        raise RuntimeError(f"{path} was written for a different cell definition")
    records = payload["records"]
    if [r["rep"] for r in records] != list(range(len(records))):
        raise RuntimeError(f"{path} has a non-contiguous replicate sequence")
    return records


def _save(path: Path, meta: dict, records: list[dict]) -> None:
    """Write records atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as handle:
        json.dump({"meta": meta, "records": records}, handle)
    os.replace(tmp, path)


def _meta(cell: Cell) -> dict:
    """Return the cell definition stored with its records."""
    described = {
        k: v for k, v in cell.describe().items() if k not in ("reps", "reps_max")
    }
    return {"schema": SCHEMA, "alphas": list(ALPHAS), **described}


def replicate_record(cell: Cell, rep: int, *, n_threads: int) -> dict:
    """Simulate and score one replicate of a cell."""
    labels, seed, diagnostics = sample_replicate(cell, rep)
    curve, _ = cell.truth()
    truth = np.clip(curve.eval(np.arange(cell.n0 + 1) / cell.n0), 0.0, 1.0)
    result = run_replicate(
        labels=labels, seed=seed, truth=truth, alphas=ALPHAS, n_threads=n_threads
    )
    return {"rep": rep, "diagnostics": diagnostics, **result}


def needs_topup(records: list[dict]) -> bool:
    """Whether any complement arm's alpha=.05 Wilson interval straddles the bar."""
    key = f"{ALPHAS[0]:g}"
    for arm in TOPUP_ARMS:
        hits = sum(r["levels"][key]["scores"][arm]["covered"] for r in records)
        low, high = wilson(hits, len(records))
        if low < BAR < high:
            return True
    return False


def run_cell(cell: Cell, root: Path, *, workers: int, n_threads: int) -> None:
    """Run a cell to its base count, then top up while the bar is straddled."""
    meta = _meta(cell)
    path = _cell_path(root, cell)
    records = _load(path, meta)

    def extend(target: int) -> None:
        start = time.time()
        while len(records) < target:
            batch = range(len(records), min(target, len(records) + CHECKPOINT))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                records.extend(
                    pool.map(
                        lambda r: replicate_record(cell, r, n_threads=n_threads), batch
                    )
                )
            _save(path, meta, records)
            rate = (time.time() - start) / max(1, len(batch))
            print(
                f"  {cell.name}: {len(records)}/{target} ({rate:.2f} s/rep)", flush=True
            )

    extend(cell.reps)
    while len(records) < cell.reps_max and needs_topup(records):
        extend(min(cell.reps_max, len(records) + TOPUP_BATCH))


def dry_run(cells: list[Cell], *, n_threads: int) -> None:
    """Time a small reference kernel call and print a cost estimate per cell.

    Cost is modelled as proportional to ``n0 * M`` summed over the eight
    cloud builds of a replicate (two trims at each of two levels per alpha).
    """
    from scripts.complement_trim.core import build_level
    from studroc_paper.methods.fiducial_ladder import khat_from_labels

    probe = next(c for c in cells if c.n0 >= 1_000)
    labels, seed, _ = sample_replicate(probe, 0)
    khat = khat_from_labels(lab_s=labels)
    start = time.time()
    build_level(labels=labels, khat=khat, level=0.05, seed=seed, n_threads=n_threads)
    unit = (time.time() - start) / (2 * probe.n0 * _auto_n_draws(probe.n0 + 1, 0.05))
    total = 0.0
    for cell in cells:
        work = sum(
            2 * cell.n0 * _auto_n_draws(cell.n0 + 1, level)
            for alpha in ALPHAS
            for level in (alpha, alpha / 2)
        )
        seconds = unit * work * cell.reps
        total += seconds
        print(f"{cell.name:55s} {seconds / 60:8.1f} min at {cell.reps} reps")
    print(
        f"total (base reps, no top-up): {total / 3600:.1f} h wall at this thread count"
    )


def summarize(root: Path) -> dict:
    """Aggregate every stored cell into ``summary.json`` and ``report.md``."""
    cells = []
    for path in sorted((root / "cells").glob("*.json.gz")):
        with gzip.open(path, "rt") as handle:
            payload = json.load(handle)
        meta, records = payload["meta"], payload["records"]
        entry = {"cell": meta, "reps": len(records), "levels": {}}
        for key in (f"{a:g}" for a in meta["alphas"]):
            level = {}
            ref = np.array(
                [r["levels"][key]["scores"]["hybrid"]["area"] for r in records]
            )
            m3 = np.array([r["levels"][key]["scores"]["m3"]["area"] for r in records])
            for arm in ARMS + REFERENCES:
                scores = [r["levels"][key]["scores"][arm] for r in records]
                hits = sum(s["covered"] for s in scores)
                area = np.array([s["area"] for s in scores])
                ratio = area / ref
                level[arm] = {
                    "coverage": hits / len(scores),
                    "wilson": wilson(hits, len(scores)),
                    "below": float(np.mean([s["below"] for s in scores])),
                    "above": float(np.mean([s["above"] for s in scores])),
                    "miss_in": float(np.mean([s["miss_in"] for s in scores])),
                    "miss_out": float(np.mean([s["miss_out"] for s in scores])),
                    "area": float(area.mean()),
                    "area_out": float(np.mean([s["area_out"] for s in scores])),
                    "ratio_hybrid": float(ratio.mean()),
                    "ratio_hybrid_se": float(ratio.std(ddof=1) / np.sqrt(len(ratio)))
                    if len(ratio) > 1
                    else None,
                    "ratio_m3": float(np.mean(area / m3)),
                }
            for tag in ("diag", "diag_half"):
                diags = [r["levels"][key][tag] for r in records]
                level[tag] = {
                    "n_draws": diags[0]["n_draws"],
                    "j_full": float(np.mean([d["j_full"] for d in diags])),
                    "j_comp": float(np.mean([d["j_comp"] for d in diags])),
                    "enforced": int(sum(d["enforced"] for d in diags)),
                    "fallback": float(np.mean([d["fallback"] for d in diags])),
                    "region_frac": float(np.mean([d["region_frac"] for d in diags])),
                }
            sampled = [r["diagnostics"].get("feature_sampled") for r in records]
            if any(s is not None for s in sampled):
                for flag in (True, False):
                    subset = [
                        r for r, s in zip(records, sampled, strict=True) if s is flag
                    ]
                    level[f"feature_sampled_{flag}"] = {
                        "reps": len(subset),
                        **{
                            arm: float(
                                np.mean(
                                    [
                                        r["levels"][key]["scores"][arm]["covered"]
                                        for r in subset
                                    ]
                                )
                            )
                            if subset
                            else None
                            for arm in ARMS + REFERENCES
                        },
                    }
            entry["levels"][key] = level
        cells.append(entry)
    summary = {"schema": "complement-trim-summary/v1", "cells": cells}
    (root / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (root / "report.md").write_text(_report(cells), encoding="utf-8")
    return summary


def _report(cells: list[dict]) -> str:
    """Render per-cell coverage and paired width tables (no verdicts)."""
    lines = ["# Complement-trim experiment: per-cell measurements", ""]
    for key in (f"{a:g}" for a in ALPHAS):
        lines += [
            f"## alpha = {key}",
            "",
            "| cell | reps | raw | hybrid | comp | comp_budget | hybrid_half | m3 |"
            " comp/hyb | budget/hyb | j full→comp | enforced | fallback |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
        ]
        for entry in cells:
            level = entry["levels"][key]
            cov = [
                f"{level[a]['coverage']:.3f}"
                for a in ("raw", "hybrid", "comp", "comp_budget", "hybrid_half", "m3")
            ]
            diag = level["diag"]
            lines.append(
                f"| {entry['cell']['name']} | {entry['reps']} | "
                + " | ".join(cov)
                + f" | {level['comp']['ratio_hybrid']:.3f}"
                f" | {level['comp_budget']['ratio_hybrid']:.3f}"
                f" | {diag['j_full']:.1f}→{diag['j_comp']:.1f}"
                f" | {diag['enforced'] + level['diag_half']['enforced']}"
                f" | {diag['fallback']:.2f} |"
            )
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["design", "dry-run", "run", "summarize"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cells", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--reps-scale", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    cells = [with_reps(c, scale=args.reps_scale) for c in select_cells(args.cells)]
    if args.command == "design":
        args.out.mkdir(parents=True, exist_ok=True)
        manifest = {"schema": SCHEMA, "cells": [c.describe() for c in all_cells()]}
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print(f"wrote {len(manifest['cells'])} cells to {args.out / 'manifest.json'}")
    elif args.command == "dry-run":
        dry_run(
            sorted(cells, key=lambda c: (c.n0, c.n1, c.name)), n_threads=args.threads
        )
    elif args.command == "run":
        for cell in sorted(cells, key=lambda c: (c.n0, c.n1, c.name)):
            print(f"cell {cell.name}", flush=True)
            run_cell(cell, args.out, workers=args.workers, n_threads=args.threads)
        summarize(args.out)
    else:
        summarize(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
