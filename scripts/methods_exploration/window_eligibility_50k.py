"""Window eligibility at n = 50,000, computed without building the fiducial cloud.

At these class sizes the exact left cutoff is at most 16 native columns against a
guard column at 200-1,800, for every realizable trim depth, so the left mask can
never reach the window. Eligibility is therefore decided entirely by the right
start, which is a deterministic function of the trailing all-negative run. That
makes it computable at high replication, and independent of the cloud budget.

The `--validate` mode recomputes the screen's balanced n = 50,000 cells and
checks them against the stored records unit by unit.
"""

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from .common import curve, interval, rng_for, sample
from .hybrid import floor_components, window_status

SIZE = 50_000
DRAWS = 4_000
SHAPES = ["normal_0.95", "t2_0.95", "kink", "sliver", "interior_sliver"]


def eligibility(*, labels: np.ndarray, depth: int) -> dict:
    """Return the window verdict and the geometry that decides it."""
    n0 = len(labels) - int(labels.sum())
    grid = np.arange(n0 + 1) / n0
    left, right = floor_components(labels=labels, draws=DRAWS, depth=depth)
    status = window_status(grid=grid, region=left | right)
    return {
        "eligible": status["eligible"],
        "reason": status["reason"],
        "left_end_index": int(np.flatnonzero(left)[-1]),
        "right_start_index": int(np.flatnonzero(right)[0]),
        "right_start_fpr": float(np.flatnonzero(right)[0] / n0),
    }


def cell(*, ratio: float, shape: str, reps: int, depth: int) -> dict:
    """Measure the eligibility rate on the screen's own datasets for one cell."""
    n0, n1 = int(2 * SIZE * ratio), 2 * SIZE - int(2 * SIZE * ratio)
    truth = curve(name=shape, n0=n0, n1=n1)
    flags, rights = [], []
    for rep in range(reps):
        rng = rng_for("interior", f"{SIZE}/{ratio}/{shape}/{rep}")
        labels = sample(truth=truth, n0=n0, n1=n1, rng=rng)
        out = eligibility(labels=labels, depth=depth)
        flags.append(out["eligible"])
        rights.append(out["right_start_fpr"])
    successes = int(np.sum(flags))
    return {
        "ratio": ratio,
        "shape": shape,
        "n0": n0,
        "n1": n1,
        "reps": reps,
        "eligible": successes,
        "rate": successes / reps,
        "ci": interval(successes=successes, count=reps),
        "right_start_fpr_mean": float(np.mean(rights)),
        "right_start_fpr_q05": float(np.quantile(rights, 0.05)),
    }


def validate(*, records: Path) -> None:
    """Check the cloud-free verdict against every stored balanced n=50,000 unit."""
    stored = collections.defaultdict(dict)
    for line in records.open():
        if '"n": 50000' not in line:
            continue
        row = json.loads(line)
        if row.get("audit"):
            continue
        d = row["alphas"]["0.05"]
        stored[(row["ratio"], row["shape"])][row["rep"]] = (
            d["window"]["eligible"],
            d["geometry"]["right_start_index"],
            d["depth"],
        )
    checked = mismatch = 0
    for (ratio, shape), reps in sorted(stored.items()):
        n0, n1 = int(2 * SIZE * ratio), 2 * SIZE - int(2 * SIZE * ratio)
        truth = curve(name=shape, n0=n0, n1=n1)
        for rep, (was_eligible, right_index, depth) in sorted(reps.items()):
            rng = rng_for("interior", f"{SIZE}/{ratio}/{shape}/{rep}")
            labels = sample(truth=truth, n0=n0, n1=n1, rng=rng)
            out = eligibility(labels=labels, depth=depth)
            checked += 1
            agrees = (
                out["eligible"] == was_eligible
                and out["right_start_index"] == right_index
            )
            if not agrees:
                mismatch += 1
                print(
                    f"  MISMATCH {ratio}/{shape}/{rep}: "
                    f"{out} vs {(was_eligible, right_index)}"
                )
    print(f"validated {checked} stored units, {mismatch} mismatches")


def main() -> None:
    """Validate against stored units, then measure the two missing directions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Realized trim depth; the verdict is invariant to it at these sizes.",
    )
    parser.add_argument("--records", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.records:
        print("=== validation against stored balanced n=50,000 units ===")
        validate(records=args.records)

    print(f"\n=== eligibility, n = 50,000, {args.reps} replicates per cell ===")
    header = (
        f"{'ratio':>6}{'n0':>8}{'n1':>8}{'shape':<17}{'elig':>7}{'rate':>7}"
        f"  {'95% CI':<16}{'rightFPR':>10}{'q05':>9}"
    )
    print(header)
    results = []
    for ratio in (0.1, 0.5, 0.9):
        for shape in SHAPES:
            r = cell(ratio=ratio, shape=shape, reps=args.reps, depth=args.depth)
            results.append(r)
            ci = f"[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]"
            print(
                f"{ratio:>6}{r['n0']:>8}{r['n1']:>8}{shape:<17}{r['eligible']:>7}"
                f"{r['rate']:>7.3f}  {ci:<16}{r['right_start_fpr_mean']:>10.4f}"
                f"{r['right_start_fpr_q05']:>9.4f}"
            )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"draws": DRAWS, "reps": args.reps, "cells": results}
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
