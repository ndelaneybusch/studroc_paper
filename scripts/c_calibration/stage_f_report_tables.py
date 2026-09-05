"""Report-facing tables for Stage F: capture, price, and violation geometry.

The study summaries written by :mod:`stage_f_analysis` aggregate coverage and
width but discard where a violation sat. This module re-scores stored records
to recover the direction, FPR location, and TPR level of every miss, and
renders the per-study tables quoted in ``stats/hybrid_floor_report.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage_f_analysis import (  # noqa: E402
    RULE_NAMES,
    CellRecords,
    load_study_records,
    score_record,
    wilson_interval,
)
from stage_f_core import decode_array, decode_band, decode_violations  # noqa: E402


def violation_geometry(
    cell: CellRecords, *, rule: str | None, alpha: float, alpha2_key: str
) -> dict:
    """Summarize where and in which direction a procedure's misses occur.

    Args:
        cell: Stored cell records.
        rule: Prespecified region rule, or ``None`` for the bare C = 1 band.
        alpha: Nominal fiducial level.
        alpha2_key: Stored M3 level key.

    Returns:
        Miss counts by direction plus FPR and true-TPR location summaries.
    """
    m_draws = int(cell.meta["cell"]["m_draws"])
    n0 = int(cell.meta["cell"]["n0"])
    fprs: list[float] = []
    tprs: list[float] = []
    depths: list[float] = []
    low = high = failing = 0
    for record in cell.records:
        truth = decode_array(record["truth"]).astype(np.float64)
        if rule is None:
            mask = decode_violations(record["fiducial_violations"][f"{alpha:g}"])
            lower, upper = decode_band(record["fiducial"][f"{alpha:g}"])
            covered = not bool(mask.any())
            viol_low = bool((lower > truth + 1e-12).any())
            viol_high = bool((truth > upper + 1e-12).any())
            depth = float(np.maximum(lower - truth, truth - upper).max(initial=0.0))
        else:
            score = score_record(
                record, rule=rule, alpha=alpha, alpha2_key=alpha2_key, m_draws=m_draws
            )
            mask = decode_violations(score["violations"])
            covered = score["covered"]
            viol_low, viol_high = score["viol_low"], score["viol_high"]
            depth = score["max_depth"]
        if covered:
            continue
        failing += 1
        low += viol_low
        high += viol_high
        indices = np.flatnonzero(mask)
        fprs.extend((indices / n0).tolist())
        tprs.extend(truth[indices].tolist())
        depths.append(max(0.0, depth))
    reps = len(cell.records)
    quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    return {
        "cell": cell.name,
        "reps": reps,
        "failing_reps": failing,
        "coverage": 1.0 - failing / reps,
        "viol_low": low / reps,
        "viol_high": high / reps,
        "miss_points": len(fprs),
        "fpr_quantiles": np.quantile(fprs, quantiles).tolist() if fprs else None,
        "tpr_quantiles": np.quantile(tprs, quantiles).tolist() if tprs else None,
        "max_depth": max(depths) if depths else 0.0,
        "median_depth": float(np.median(depths)) if depths else 0.0,
    }


def geometry_table(
    root: Path,
    *,
    study: str,
    rules: tuple[str | None, ...] = (None,) + RULE_NAMES,
    alpha: float = 0.05,
    alpha2_key: str = "0.05",
) -> list[dict]:
    """Return violation-geometry rows for every cell of one study.

    A ``None`` entry in ``rules`` scores the unmodified C = 1 parent band.
    """
    rows = []
    for cell in load_study_records(root, study=study):
        for rule in rules:
            rows.append(
                {
                    "rule": rule or "c1",
                    **violation_geometry(
                        cell, rule=rule, alpha=alpha, alpha2_key=alpha2_key
                    ),
                }
            )
    return rows


def baseline_table(root: Path, *, study: str, alpha: float = 0.05) -> list[dict]:
    """Return per-cell C=1 and M3 reference coverage for one study."""
    rows = []
    for cell in load_study_records(root, study=study):
        m_draws = int(cell.meta["cell"]["m_draws"])
        scores = [
            score_record(
                record,
                rule="frontier_floor_v1",
                alpha=alpha,
                alpha2_key=f"{alpha:g}",
                m_draws=m_draws,
            )
            for record in cell.records
        ]
        reps = len(scores)
        successes = sum(row["covered"] for row in scores)
        rows.append(
            {
                "cell": cell.name,
                "source": cell.source,
                "n0": int(cell.meta["cell"]["n0"]),
                "n1": int(cell.meta["cell"]["n1"]),
                "auc": float(cell.meta["cell"]["shape_meta"].get("auc", float("nan"))),
                "family": str(cell.meta["cell"]["shape_meta"].get("family", "")),
                "geometry": str(
                    cell.meta["cell"]["shape_meta"].get("corner_geometry", "")
                ),
                "reps": reps,
                "c1_coverage": sum(row["fiducial_covered"] for row in scores) / reps,
                "m3_coverage": sum(row["m3_covered"] for row in scores) / reps,
                "floor_coverage": successes / reps,
                "floor_wilson": wilson_interval(successes, reps),
                "area_ratio": float(
                    np.mean([row["area_ratio_vs_fiducial"] for row in scores])
                ),
                "m3_ratio": float(
                    np.mean([row["m3_area"] / row["fiducial_area"] for row in scores])
                ),
                "region_fraction": float(
                    np.mean([row["region_fraction"] for row in scores])
                ),
                "left_fraction": float(
                    np.mean([row["left_fraction"] for row in scores])
                ),
                "right_fraction": float(
                    np.mean([row["right_fraction"] for row in scores])
                ),
                "run_length": float(np.mean([row["run_length"] for row in scores])),
                "right_margin": float(np.mean([row["right_margin"] for row in scores])),
                "exterior_escape": sum(row["exterior_escape"] for row in scores) / reps,
                "edge_escape": sum(row["edge_escape"] for row in scores) / reps,
                "far_escape": sum(row["far_escape"] for row in scores) / reps,
                "floor_region_failure": (
                    sum(row["floor_region_failure"] for row in scores) / reps
                ),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    """Write the report tables for one study."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--root", type=Path, default=Path("data/results/hybrid_floor_20260902")
    )
    parser.add_argument("--study", required=True, choices=("A", "B", "C"))
    args = parser.parse_args(argv)
    out = args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": baseline_table(args.root, study=args.study),
        "geometry": geometry_table(args.root, study=args.study),
    }
    path = out / f"study_{args.study.lower()}_tables.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
