"""CLI driver of the trim-exponent calibration study.

Stage S is a low-cost stop/go screen. Stage A (fitting arms) should run only
if that screen shows a useful calibration margin; Stage B (confirmation)
requires a frozen map artifact produced by ``fit_stage_a.py`` and reviewed by
a human first. Cells run sequentially; replicates within a cell run
concurrently.

Examples (from the project root)::

    # Inspect the decision-first screen, run the parity gate, then run it.
    uv run python scripts/c_calibration/run.py --stage S --dry-run
    uv run python scripts/c_calibration/parity_gate.py
    uv run python scripts/c_calibration/run.py --stage S
    uv run python scripts/c_calibration/check_screen.py

    # After a positive screen, run the selected Stage A arms.
    uv run python scripts/c_calibration/run.py --stage A

    # Shard: only the large-n arm, or a name filter.
    uv run python scripts/c_calibration/run.py --stage A --arms large_n
    uv run python scripts/c_calibration/run.py --stage A --select 't2_95'

    # Re-run cells flagged saturated at 2x the Monte Carlo budget.
    uv run python scripts/c_calibration/run.py --stage A --rerun-saturated

    # Stage B against the frozen map.
    uv run python scripts/c_calibration/run.py --stage B \
        --map data/results/c_calibration/frozen_map.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from design import (  # noqa: E402
    Cell,
    screening_cells,
    stage_a_cells,
    stage_b_cells,
    summarize,
)
from runner import cell_paths, run_cell, se_gate_needs_topup  # noqa: E402

DEFAULT_OUT = Path("data/results/c_calibration")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stage", choices=("S", "A", "B"), required=True)
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="Output root directory"
    )
    parser.add_argument(
        "--arms", nargs="*", default=None, help="Restrict to these study arms"
    )
    parser.add_argument(
        "--select", default=None, help="Substring filter on cell names"
    )
    parser.add_argument(
        "--map",
        type=Path,
        default=None,
        help="Frozen map artifact (required for Stage B)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Concurrent replicates per cell (default: cores // threads)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Rayon threads per kernel call (default 4)",
    )
    parser.add_argument(
        "--mem-gb",
        type=float,
        default=40.0,
        help="Memory budget for concurrent fiducial clouds (default 40)",
    )
    parser.add_argument(
        "--reps-scale",
        type=float,
        default=1.0,
        help="Multiplier on every cell's replicate counts",
    )
    parser.add_argument(
        "--m-scale",
        type=float,
        default=1.0,
        help="Multiplier on every cell's Monte Carlo budget",
    )
    parser.add_argument(
        "--rerun-saturated",
        action="store_true",
        help="Re-run cells whose summaries flag ladder saturation, at 2x M",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def select_cells(args: argparse.Namespace) -> list[Cell]:
    if args.stage == "S":
        cells = screening_cells()
    elif args.stage == "A":
        cells = stage_a_cells()
    else:
        cells = stage_b_cells()
    if args.arms:
        cells = [c for c in cells if c.arm in set(args.arms)]
    if args.select:
        cells = [c for c in cells if args.select in c.name]
    if args.reps_scale != 1.0:
        cells = [
            Cell(
                **{
                    **c.__dict__,
                    "reps": max(1, int(c.reps * args.reps_scale)),
                    "reps_max": max(1, int(c.reps_max * args.reps_scale)),
                }
            )
            for c in cells
        ]
    return cells


def cell_is_complete(cell: Cell, out_dir: Path) -> bool:
    """A cell is complete when its summary matches the design and either the
    SE gate passes or the top-up ceiling was reached."""
    _, summary_path = cell_paths(out_dir, cell)
    if not summary_path.exists():
        return False
    summary = json.loads(summary_path.read_text())
    meta = summary["meta"]
    if meta["m_draws"] != cell.m_draws or meta["reps_done"] < cell.reps:
        return False
    if cell.stage == "B":
        return True
    return (
        not se_gate_needs_topup(cell, summary["aggregate"])
        or meta["reps_done"] >= cell.reps_max
    )


def saturated_cells(cells: list[Cell], out_dir: Path) -> list[Cell]:
    flagged = []
    for cell in cells:
        _, summary_path = cell_paths(out_dir, cell)
        if not summary_path.exists():
            continue
        agg = json.loads(summary_path.read_text())["aggregate"]
        if any(v["saturated"] for v in agg["per_alpha"].values()):
            flagged.append(cell)
    return flagged


def main(argv=None) -> int:
    args = parse_args(argv)
    cells = select_cells(args)
    out_dir = args.out / f"stage{args.stage}"

    if args.dry_run:
        summary = summarize(cells)
        for row in summary.rows:
            print(
                f"{row['name']:<55} n={row['n0']}x{row['n1']:<7} "
                f"M={row['M']:<7} reps={row['reps']:<5} "
                f"cloud={row['cloud_gb']:.2f}GB est={row['est_hours']:.2f}h"
            )
        print(
            f"\n{summary.n_cells} cells, {summary.total_reps} baseline reps, "
            f"~{summary.total_hours:.0f} core-saturated hours, "
            f"max cloud {summary.max_cloud_gb:.2f} GB"
        )
        return 0

    auto_fn = None
    if args.stage == "B":
        if args.map is None:
            print("Stage B requires --map <frozen_map.json>", file=sys.stderr)
            return 2
        from map_eval import load_artifact, require_confirmation_ready, resolve_exponent

        artifact = load_artifact(args.map)
        require_confirmation_ready(artifact)

        def auto_fn(n0, n1, alpha):
            return resolve_exponent(artifact, n0=n0, n1=n1, alpha=alpha)

    if args.rerun_saturated:
        cells = saturated_cells(cells, out_dir)
        print(f"Re-running {len(cells)} saturated cells at 2x M")
        m_scale = 2.0 * args.m_scale
    else:
        m_scale = args.m_scale

    cores = os.cpu_count() or 8
    workers = args.workers or max(1, cores // args.threads)

    todo = [
        c
        for c in cells
        if m_scale != 1.0 or args.force or not cell_is_complete(c, out_dir)
    ]
    print(
        f"Stage {args.stage}: {len(todo)}/{len(cells)} cells to run "
        f"({workers} workers x {args.threads} threads, {args.mem_gb:.0f} GB budget)"
    )
    failures = []
    for i, cell in enumerate(todo):
        print(f"[{i + 1}/{len(todo)}] {cell.name} (M={cell.m_draws}, reps={cell.reps})")
        try:
            run_cell(
                cell,
                out_dir,
                workers=workers,
                threads_per_call=args.threads,
                mem_gb=args.mem_gb,
                auto_exponent_fn=auto_fn,
                m_scale=m_scale,
                force=args.force,
            )
        except Exception as err:  # noqa: BLE001 - keep the campaign alive
            failures.append((cell.name, repr(err)))
            print(f"[{cell.name}] FAILED: {err!r}", file=sys.stderr, flush=True)

    remaining_saturated = saturated_cells(cells, out_dir)
    if remaining_saturated:
        print(
            "\nSaturated cells (excluded from fitting; re-run with "
            "--rerun-saturated):"
        )
        for cell in remaining_saturated:
            print(f"  {cell.name}")
    if failures:
        print("\nFailed cells:")
        for name, err in failures:
            print(f"  {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
