"""Fixed-shape diagnostic ladders for the C = 1 coverage boundary.

Two targeted probes that isolate effects the greedy design surfaced but could
not attribute, because its cells vary shape and sample size together.

``nladder``
    Sweeps n at a single shape where the infill batch measured coverage
    *falling* as n grows. Holding the shape fixed is what makes the
    sample-size effect attributable, and this probe is the direct falsification
    of the pre-registered surface's monotone-in-n sign constraint.

``thinprobe``
    Brackets the production trim-grid rule's thinning threshold. The min-p trim
    switches to a thinned grid at K = n0 + 1 > 2001 while the band is still
    evaluated on the full grid; if that switch drives the large-n degradation,
    coverage jumps between n0 = 1900 and n0 = 2100. It does not — the probe is
    what rules the artifact out.

Both write cells the ordinary loaders pick up, so their results join the rest of
the boundary study.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from followup_runs import DEFAULT_OUT, _cell, register_followup_shapes  # noqa: E402
from runner import run_cell  # noqa: E402
from shapes import ShapeSpec, make_t_shape, shape_registry  # noqa: E402


@dataclass(frozen=True)
class Ladder:
    """A fixed shape swept over sample size."""

    key: str
    shape_name: str
    df: float
    auc: float
    reps: int
    ns: tuple[int, ...]
    subdir: str
    arm: str
    notes: str


LADDERS = {
    "nladder": Ladder(
        key="nladder",
        shape_name="tladder_df47_a986",
        df=4.69,
        auc=0.986,
        reps=300,
        ns=(150, 400, 800, 1200, 2000),
        subdir="boundary_nladder",
        arm="followup_boundary_nladder",
        notes="fixed-shape n ladder testing monotonicity in n",
    ),
    "thinprobe": Ladder(
        key="thinprobe",
        shape_name="tthin_df66_a988",
        df=6.62,
        auc=0.9883,
        reps=400,
        ns=(1500, 1900, 2100, 2600, 4000),
        subdir="boundary_thinprobe",
        arm="followup_boundary_thinprobe",
        notes="trim-grid thinning discontinuity probe at K=2001",
    ),
}


def run_ladder(
    ladder: Ladder,
    *,
    out_root: Path,
    workers: int,
    threads_per_call: int,
    mem_gb: float,
) -> None:
    """Run every sample size of one ladder, resuming through the runner."""
    shape_registry().setdefault(
        ladder.shape_name,
        ShapeSpec(
            name=ladder.shape_name,
            role="followup",
            build=lambda: make_t_shape(ladder.auc, df=ladder.df),
            meta={"family": "student_t", "auc": ladder.auc, "df": ladder.df},
        ),
    )
    out_dir = out_root / ladder.subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "nladder" if ladder.key == "nladder" else "thin"
    for n in ladder.ns:
        cell = _cell(
            name=f"{prefix}--{ladder.shape_name}--n{n}",
            stage="S",
            arm=ladder.arm,
            shape=ladder.shape_name,
            n0=n,
            n1=n,
            reps=ladder.reps,
            reps_max=ladder.reps,
            notes=ladder.notes,
        )
        print(f"running {cell.name}", flush=True)
        run_cell(
            cell,
            out_dir,
            workers=workers,
            threads_per_call=threads_per_call,
            mem_gb=mem_gb,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes", nargs="+", choices=[*LADDERS, "all"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--threads-per-call", type=int, default=4)
    parser.add_argument("--mem-gb", type=float, default=40.0)
    args = parser.parse_args()

    register_followup_shapes()
    keys = list(LADDERS) if "all" in args.probes else args.probes
    for key in keys:
        run_ladder(
            LADDERS[key],
            out_root=args.out,
            workers=args.workers,
            threads_per_call=args.threads_per_call,
            mem_gb=args.mem_gb,
        )


if __name__ == "__main__":
    main()
