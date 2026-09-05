"""Grid-point-level mechanism analysis for Stage F.

The per-cell summaries report whole-band coverage and mean width. This module
resolves both to the grid: pointwise miss rates by distance from each end, the
distance from residual violations to the floored region, a within-cell
per-replicate test of the saturated-run trigger, an effective-looks
decomposition of the residual, and a post-hoc sweep of the left cutoff.

Every quantity is recomputed from stored paired parent bands. Nothing here
re-simulates, and nothing here modifies ``frontier_floor_v1``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage_f_analysis import load_study_records  # noqa: E402
from stage_f_core import (  # noqa: E402
    decode_array,
    decode_band,
    frontier_region_masks,
    record_observables,
    stitch_hybrid,
)

STUDIES = ("A", "B", "C")
# Roughly geometric bin edges: unit resolution over the corner region the
# theory implicates, widening into the interior where the rate is flat.
# fmt: off
LOG_EDGES = np.array([
    1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 17, 23, 30, 40, 55, 75, 100, 140, 190,
    260, 350, 480, 650, 900, 1250, 1700, 2400,
])
# fmt: on
K_LEFT_SWEEP = (0, 3, 5, 6, 7, 8, 10)
MISS_TOLERANCE = 1e-12


def _replicate(record: dict, *, m_draws: int) -> dict:
    """Rebuild one replicate's parents, frontier region, and floored band.

    Args:
        record: Lossless paired-parent replicate record.
        m_draws: Fiducial cloud size fixing the left cutoff.

    Returns:
        Parent edges, region components, and per-edge miss masks.
    """
    observables = record_observables(record)
    khat = decode_array(record["khat"]).astype(np.int64)
    truth = decode_array(record["truth"]).astype(np.float64)
    fid_lower, fid_upper = decode_band(record["fiducial"]["0.05"])
    m3_lower, m3_upper = decode_band(record["m3"]["0.05"])
    left, right = frontier_region_masks(
        "frontier_floor_v1", observables=observables, khat=khat, m_draws=m_draws
    )
    lower, upper = stitch_hybrid(
        fid_lower, fid_upper, m3_lower, m3_upper, left | right, closure="widening"
    )
    return {
        "observables": observables,
        "khat": khat,
        "truth": truth,
        "fid": (fid_lower, fid_upper),
        "m3": (m3_lower, m3_upper),
        "left": left,
        "right": right,
        "c1_low": fid_lower > truth + MISS_TOLERANCE,
        "c1_high": truth > fid_upper + MISS_TOLERANCE,
        "floor_low": lower > truth + MISS_TOLERANCE,
        "floor_high": truth > upper + MISS_TOLERANCE,
    }


def pointwise_profiles(root: Path, *, study: str, n0_min: int = 0) -> dict:
    """Return pointwise miss rates in log-spaced bins measured from the left.

    Args:
        root: Stage F output root.
        study: Study identifier.
        n0_min: Restrict to cells at or above this negative-class size.

    Returns:
        Per-bin miss counts for both bands and both edges, plus bin exposure.
    """
    bins = len(LOG_EDGES) - 1
    keys = ("c1_low", "c1_high", "floor_low", "floor_high")
    accumulator = {key: np.zeros(bins) for key in (*keys, "n")}
    for cell in load_study_records(root, study=study):
        meta = cell.meta["cell"]
        n0, m_draws = int(meta["n0"]), int(meta["m_draws"])
        if n0 < n0_min:
            continue
        index = np.digitize(np.arange(n0 + 1), LOG_EDGES) - 1
        valid = (index >= 0) & (index < bins)
        for record in cell.records:
            replicate = _replicate(record, m_draws=m_draws)
            for key in keys:
                accumulator[key] += np.bincount(
                    index[valid & replicate[key]], minlength=bins
                )[:bins]
            accumulator["n"] += np.bincount(index[valid], minlength=bins)[:bins]
    return accumulator


def residual_distances(root: Path, *, study: str) -> np.ndarray:
    """Return distances in grid points from floored violations to the region.

    Args:
        root: Stage F output root.
        study: Study identifier.

    Returns:
        Array of ``(distance, n0)`` rows, one per violated grid point.
    """
    rows = []
    for cell in load_study_records(root, study=study):
        meta = cell.meta["cell"]
        n0, m_draws = int(meta["n0"]), int(meta["m_draws"])
        for record in cell.records:
            replicate = _replicate(record, m_draws=m_draws)
            miss = replicate["floor_low"] | replicate["floor_high"]
            if not miss.any():
                continue
            region_index = np.flatnonzero(replicate["left"] | replicate["right"])
            miss_index = np.flatnonzero(miss)
            distance = np.min(
                np.abs(miss_index[:, None] - region_index[None, :]), axis=1
            )
            rows.append(np.column_stack([distance, np.full(distance.shape, n0)]))
    return np.vstack(rows) if rows else np.empty((0, 2))


def multiplicity(root: Path, *, study: str) -> list[dict]:
    """Return per-cell pointwise rate, whole-band miscoverage, effective looks.

    Args:
        root: Stage F output root.
        study: Study identifier.

    Returns:
        One mapping per cell.
    """
    output = []
    for cell in load_study_records(root, study=study):
        meta = cell.meta["cell"]
        n0, m_draws = int(meta["n0"]), int(meta["m_draws"])
        points = missed = failures = excursions = 0
        for record in cell.records:
            replicate = _replicate(record, m_draws=m_draws)
            miss = replicate["floor_low"] | replicate["floor_high"]
            points += miss.size
            missed += int(miss.sum())
            if miss.any():
                failures += 1
                excursions += int(
                    np.count_nonzero(np.diff(np.r_[0, miss.astype(np.int8), 0]) == 1)
                )
        reps = len(cell.records)
        pointwise = missed / points
        whole = failures / reps
        looks = (
            math.log1p(-whole) / math.log1p(-pointwise)
            if 0.0 < pointwise < 1.0 and whole < 1.0
            else float("nan")
        )
        output.append(
            {
                "cell": cell.name,
                "n0": n0,
                "pointwise": pointwise,
                "whole": whole,
                "effective_looks": looks,
                "excursions_per_failure": excursions / failures if failures else 0.0,
            }
        )
    return output


def trigger_association(root: Path, *, study: str) -> dict:
    """Return within-cell standardized run-length differences by failure status.

    Holding the cell fixed removes shape, sample size, and AUC as explanations,
    so the remaining variation is the rank realization alone.

    Args:
        root: Stage F output root.
        study: Study identifier.

    Returns:
        Mean standardized difference, count positive, and cells compared, for
        the C = 1 band and the floored band.
    """
    per_cell: dict[str, list[tuple[float, int, int]]] = defaultdict(list)
    for cell in load_study_records(root, study=study):
        meta = cell.meta["cell"]
        n0, m_draws = int(meta["n0"]), int(meta["m_draws"])
        for record in cell.records:
            replicate = _replicate(record, m_draws=m_draws)
            observables = replicate["observables"]
            saturated = observables.n1 - replicate["khat"]
            k_sat = int(np.flatnonzero(saturated == 0)[0])
            per_cell[cell.name].append(
                (
                    float(n0 - k_sat),
                    int(replicate["c1_low"].any() or replicate["c1_high"].any()),
                    int(replicate["floor_low"].any() or replicate["floor_high"].any()),
                )
            )
    c1_diffs: list[float] = []
    floor_diffs: list[float] = []
    for rows in per_cell.values():
        run = np.array([row[0] for row in rows])
        if run.std() == 0:
            continue
        for column, sink in ((1, c1_diffs), (2, floor_diffs)):
            flag = np.array([row[column] for row in rows], dtype=bool)
            if flag.sum() >= 5 and (~flag).sum() >= 5:
                sink.append(float((run[flag].mean() - run[~flag].mean()) / run.std()))
    return {
        "c1": (float(np.mean(c1_diffs)), sum(d > 0 for d in c1_diffs), len(c1_diffs)),
        "floor": (
            float(np.mean(floor_diffs)),
            sum(d > 0 for d in floor_diffs),
            len(floor_diffs),
        ),
    }


def k_left_sweep(root: Path, *, study: str) -> dict[int, dict[str, list[float]]]:
    """Sweep the left cutoff with the right region held fixed.

    This variant family was chosen after seeing Stage F outcomes. It is
    exploratory and is not a Stage F arm; any cutoff it favours needs its own
    confirmation data.

    Args:
        root: Stage F output root.
        study: Study identifier.

    Returns:
        Per-cutoff lists of per-cell coverage and paired width ratio.
    """
    output: dict[int, dict[str, list[float]]] = {
        cutoff: {"coverage": [], "width": []} for cutoff in K_LEFT_SWEEP
    }
    for cell in load_study_records(root, study=study):
        meta = cell.meta["cell"]
        n0, m_draws = int(meta["n0"]), int(meta["m_draws"])
        index = np.arange(n0 + 1)
        totals = {cutoff: [0, 0.0] for cutoff in K_LEFT_SWEEP}
        for record in cell.records:
            replicate = _replicate(record, m_draws=m_draws)
            fid_lower, fid_upper = replicate["fid"]
            m3_lower, m3_upper = replicate["m3"]
            truth = replicate["truth"]
            base = float(np.mean(fid_upper - fid_lower))
            for cutoff in K_LEFT_SWEEP:
                region = replicate["right"]
                if cutoff:
                    region = region | (index <= min(n0, cutoff))
                lower, upper = stitch_hybrid(
                    fid_lower, fid_upper, m3_lower, m3_upper, region, closure="widening"
                )
                missed = (lower > truth + MISS_TOLERANCE) | (
                    truth > upper + MISS_TOLERANCE
                )
                totals[cutoff][0] += int(not missed.any())
                totals[cutoff][1] += float(np.mean(upper - lower)) / base - 1.0
        reps = len(cell.records)
        for cutoff in K_LEFT_SWEEP:
            output[cutoff]["coverage"].append(totals[cutoff][0] / reps)
            output[cutoff]["width"].append(totals[cutoff][1] / reps)
    return output


def main(argv: list[str] | None = None) -> int:
    """Write the grid-level analysis for every study."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--root", type=Path, default=Path("data/results/hybrid_floor_20260902")
    )
    args = parser.parse_args(argv)
    payload: dict[str, dict] = {}
    for study in STUDIES:
        profiles = pointwise_profiles(args.root, study=study)
        distances = residual_distances(args.root, study=study)
        payload[study] = {
            "profile_bin_edges": LOG_EDGES.tolist(),
            "profiles": {key: value.tolist() for key, value in profiles.items()},
            "distance_quantiles": {
                str(q): float(np.percentile(distances[:, 0], q))
                for q in (10, 25, 50, 75, 90)
            }
            if len(distances)
            else {},
            "distance_within_10_points": float(np.mean(distances[:, 0] <= 10))
            if len(distances)
            else None,
            "multiplicity": multiplicity(args.root, study=study),
            "trigger_association": trigger_association(args.root, study=study),
            "k_left_sweep": {
                str(cutoff): value
                for cutoff, value in k_left_sweep(args.root, study=study).items()
            },
        }
    destination = args.root / "analysis" / "deep_analysis.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
