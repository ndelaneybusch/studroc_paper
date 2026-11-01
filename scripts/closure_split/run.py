"""Resumable runner for the closure/split experiment.

Usage, from the repository root::

    uv run --no-sync python -m scripts.closure_split.run design    [--out DIR]
    uv run --no-sync python -m scripts.closure_split.run calibrate [--workers N]
    uv run --no-sync python -m scripts.closure_split.run dry-run   [--workers N]
    uv run --no-sync python -m scripts.closure_split.run run       [--study T|S] \
        [--cells SUBSTR ...] [--workers N] [--threads 1] [--reps-scale 1.0]
    uv run --no-sync python -m scripts.closure_split.run summarize [--out DIR]

Spec: ``stats/experiments/closure_split_spec.md``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from pathlib import Path

from scripts.closure_split import core, split
from scripts.closure_split.cells import (
    ALPHAS_S,
    ALPHAS_T,
    Cell,
    all_cells,
    sample_replicate,
    select_cells,
    true_roc_on_grid,
)

SCHEMA = "closure-split-cell/v1"
DEFAULT_OUT = Path("data/results/closure_split_20261027")
TOPUP_ARM = "rec_T"
TOPUP_BAR = 0.96
GAMMAS = "gammas.json"

_WORKER_STATE: dict = {}


def chunk_size(cell: Cell) -> int:
    """Replicates per task: small enough to balance, large enough to amortize."""
    if cell.study == "S":
        return 100
    if cell.n0 >= 5_000:
        return 2
    if cell.n0 >= 1_000:
        return 5
    return 20


def topup_batch(cell: Cell) -> int:
    """Replicates added per top-up round: the cell's base count."""
    return cell.reps


def needs_topup(cell: Cell, records: list[dict]) -> bool:
    """Study T only: add replicates while ``rec_T`` covers below .96 at .05.

    This allocates precision to the cells where absolute coverage near
    nominal needs resolving; it does not judge the cell.
    """
    if cell.study != "T" or len(records) >= cell.reps_max:
        return False
    key = f"{ALPHAS_T[0]:g}"
    hits = sum(r["levels"][key]["scores"][TOPUP_ARM]["covered"] for r in records)
    return hits / len(records) < TOPUP_BAR


def replicate_record(cell: Cell, rep: int, *, n_threads: int) -> dict:
    """Simulate and score one replicate of a cell."""
    sample = sample_replicate(cell, rep)
    truth = true_roc_on_grid(cell)
    if cell.study == "T":
        result = core.run_replicate(
            labels=sample.labels,
            seed=sample.seed,
            truth=truth,
            alphas=ALPHAS_T,
            n_threads=n_threads,
        )
    else:
        result = split.run_replicate(
            labels=sample.labels, truth=truth, n0=cell.n0, n1=cell.n1
        )
    return {"rep": rep, "diagnostics": sample.diagnostics, **result}


def _work(task: tuple[str, int, int, int, str]) -> tuple[str, list[dict], float]:
    """Run a contiguous block of replicates of one cell in a worker process."""
    name, start, stop, n_threads, gamma_path = task
    if _WORKER_STATE.get("gammas") != gamma_path:
        split.load_gammas(Path(gamma_path))
        _WORKER_STATE["gammas"] = gamma_path
    cell = next(c for c in all_cells() if c.name == name)
    began = time.perf_counter()
    records = [
        replicate_record(cell, r, n_threads=n_threads) for r in range(start, stop)
    ]
    return name, records, time.perf_counter() - began


def _cell_path(root: Path, cell: Cell) -> Path:
    return root / "cells" / f"{cell.name}.json.gz"


def _meta(cell: Cell) -> dict:
    described = {
        k: v for k, v in cell.describe().items() if k not in ("reps", "reps_max")
    }
    alphas = ALPHAS_T if cell.study == "T" else ALPHAS_S
    return {"schema": SCHEMA, "alphas": list(alphas), **described}


def load_records(path: Path, meta: dict | None = None) -> list[dict]:
    """Load a cell's records, refusing a changed cell definition."""
    if not path.exists():
        return []
    with gzip.open(path, "rt") as handle:
        payload = json.load(handle)
    if meta is not None and payload["meta"] != meta:
        raise RuntimeError(f"{path} was written for a different cell definition")
    records = payload["records"]
    if [r["rep"] for r in records] != list(range(len(records))):
        raise RuntimeError(f"{path} has a non-contiguous replicate sequence")
    return records


def _save(path: Path, meta: dict, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as handle:
        json.dump({"meta": meta, "records": records}, handle)
    os.replace(tmp, path)


def run(cells: list[Cell], root: Path, *, workers: int, n_threads: int) -> None:
    """Run every cell to its base count, then top up cells below the bar.

    Cells are scheduled largest first so the slow cells start early; each
    cell's records are saved whenever its outstanding tasks complete.
    """
    gamma_path = str(root / GAMMAS)
    state = {}
    for cell in cells:
        meta = _meta(cell)
        records = load_records(_cell_path(root, cell), meta)
        state[cell.name] = {
            "cell": cell,
            "meta": meta,
            "done": {r["rep"]: r for r in records},
            "target": max(cell.reps, len(records)),
            "pending": 0,
        }
    order = sorted(
        cells, key=lambda c: -(c.n0 + c.n1) * (2 if c.study == "T" else 0.01)
    )
    queue: list[tuple[str, int, int]] = []

    def schedule(name: str) -> None:
        entry = state[name]
        have = len(entry["done"])
        size = chunk_size(entry["cell"])
        for start in range(have, entry["target"], size):
            queue.append((name, start, min(start + size, entry["target"])))

    for cell in order:
        schedule(cell.name)
    began = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        running = {}
        while queue or running:
            while queue and len(running) < 2 * workers:
                name, start, stop = queue.pop(0)
                future = pool.submit(_work, (name, start, stop, n_threads, gamma_path))
                running[future] = name
                state[name]["pending"] += 1
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                name = running.pop(future)
                _, records, seconds = future.result()
                entry = state[name]
                entry["pending"] -= 1
                entry["done"].update({r["rep"]: r for r in records})
                if entry["pending"]:
                    continue
                ordered = [entry["done"][r] for r in sorted(entry["done"])]
                _save(_cell_path(root, entry["cell"]), entry["meta"], ordered)
                minutes = (time.time() - began) / 60
                print(f"[{minutes:6.1f} min] {name}: {len(ordered)} reps", flush=True)
                if needs_topup(entry["cell"], ordered):
                    entry["target"] = min(
                        entry["cell"].reps_max,
                        len(ordered) + topup_batch(entry["cell"]),
                    )
                    schedule(name)


def dry_run(cells: list[Cell], root: Path, *, workers: int, n_threads: int) -> None:
    """Time a few replicates of every cell and extrapolate to base replication.

    The estimate divides total single-process seconds at base replication by
    the worker count, so it assumes the kernel threads and workers do not
    contend; expect the real wall time to run longer when they do.
    """
    split.load_gammas(root / GAMMAS)
    total_t = total_s = 0.0
    for cell in cells:
        reps = 2 if cell.study == "T" else 3
        start = time.perf_counter()
        for rep in range(reps):
            replicate_record(cell, 10_000 + rep, n_threads=n_threads)
        per_rep = (time.perf_counter() - start) / reps
        cost = per_rep * cell.reps
        if cell.study == "T":
            total_t += cost
        else:
            total_s += cost
        print(
            f"{cell.name:58s} {per_rep:7.3f} s/rep  {cost / 60:7.1f} core-min",
            flush=True,
        )
    wall = (total_t + total_s) / 3600 / workers
    print(
        f"Study T {total_t / 3600:.2f} core-h, Study S {total_s / 3600:.2f} core-h "
        f"at base reps; about {wall:.2f} h wall on {workers} workers"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command", choices=["design", "calibrate", "dry-run", "run", "summarize"]
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--study", choices=["T", "S"], default=None)
    parser.add_argument("--cells", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--reps-scale", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    cells = [
        replace(
            c,
            reps=max(2, round(c.reps * args.reps_scale)),
            reps_max=max(2, round(c.reps_max * args.reps_scale)),
        )
        for c in select_cells(study=args.study, patterns=args.cells)
    ]
    if args.command == "design":
        args.out.mkdir(parents=True, exist_ok=True)
        manifest = {"schema": SCHEMA, "cells": [c.describe() for c in all_cells()]}
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print(f"wrote {len(manifest['cells'])} cells to {args.out / 'manifest.json'}")
    elif args.command == "calibrate":
        count = split.precompute_gammas(cells, args.out / GAMMAS, workers=args.workers)
        print(f"{count} class calibrations in {args.out / GAMMAS}")
    elif args.command == "dry-run":
        dry_run(cells, args.out, workers=args.workers, n_threads=args.threads)
    elif args.command == "run":
        run(cells, args.out, workers=args.workers, n_threads=args.threads)
        from scripts.closure_split.analyze import summarize

        summarize(args.out)
    else:
        from scripts.closure_split.analyze import summarize

        summarize(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
