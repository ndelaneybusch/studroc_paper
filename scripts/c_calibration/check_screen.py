"""Stop/go analysis for the decision-first calibration screen.

The screen does not fit or freeze an auto map. It asks whether a larger
campaign is justified by checking the alpha=.05 lower envelope across
shapes, approximate available width gain, the large-n taper, and directional
imbalance at fixed minority-class size.
"""

import argparse
import json
from pathlib import Path

import numpy as np

PRIMARY_ALPHA = 0.05
MIN_USEFUL_C_MARGIN = 0.15
MIN_USEFUL_AREA_GAIN = 0.04
MAX_SIMPLE_MAP_C_SPREAD = 0.15
# Bonferroni-rounded normal cutoff for 12 two-sided within-shape comparisons.
IMBALANCE_SIMULTANEOUS_Z = 3.1
EXPECTED_SHAPE_CELLS = 10
EXPECTED_TAPER_CELLS = 12
EXPECTED_IMBALANCE_CELLS = 8


def load_summaries(path: Path) -> list[dict]:
    """Load all screening summaries.

    Args:
        path: Directory containing summary JSON files.

    Returns:
        Parsed summary payloads.

    Raises:
        FileNotFoundError: If the directory contains no summaries.
    """
    summaries = [json.loads(p.read_text()) for p in sorted(path.glob("*.summary.json"))]
    if not summaries:
        raise FileNotFoundError(f"No screening summaries under {path}")
    return summaries


def alpha_estimate(summary: dict, alpha: float = PRIMARY_ALPHA) -> dict | None:
    """Return one cell's usable calibration estimate at alpha.

    Args:
        summary: Cell summary payload.
        alpha: Nominal significance level.

    Returns:
        Compact estimate, or None when the ladder estimate is excluded.
    """
    est = summary["aggregate"]["per_alpha"].get(f"{alpha:g}")
    if est is None or est["infeasible"] or est["saturated"] or est["unconstrained"]:
        return None
    if est["c_star"] is None or est["c_star_ci"]["se"] is None:
        return None
    se = float(est["c_star_ci"]["se"])
    ladder = np.asarray(summary["aggregate"]["ladder"])
    idx = int(np.flatnonzero(ladder == est["j_star"])[0])
    oracle_area = float(summary["aggregate"]["area_by_j"][idx])
    c1 = next(
        row
        for row in summary["aggregate"]["ref_maps"]
        if row["label"] == "c1" and row["alpha"] == alpha
    )
    return {
        "cell": summary["meta"]["cell"]["name"],
        "arm": summary["meta"]["cell"]["arm"],
        "shape": summary["meta"]["cell"]["shape"],
        "n0": summary["meta"]["cell"]["n0"],
        "n1": summary["meta"]["cell"]["n1"],
        "c_star": float(est["c_star"]),
        "c_se": se,
        "c_lower_1se": float(est["c_star"] - se),
        "area_gain_vs_c1": float(1.0 - oracle_area / c1["area"]),
    }


def taper_diagnostics(rows: list[dict]) -> dict[str, dict]:
    """Summarize whether C* decreases from the first to last screened n.

    Args:
        rows: Usable alpha=.05 taper rows.

    Returns:
        Endpoint contrasts and uncertainty by shape.
    """
    out = {}
    for shape in sorted({row["shape"] for row in rows}):
        group = sorted(
            (row for row in rows if row["shape"] == shape),
            key=lambda row: row["n0"],
        )
        if len(group) < 2:
            continue
        first, last = group[0], group[-1]
        decrease = first["c_star"] - last["c_star"]
        decrease_se = float(np.hypot(first["c_se"], last["c_se"]))
        out[shape] = {
            "n_first": first["n0"],
            "n_last": last["n0"],
            "c_star_decrease": decrease,
            "decrease_lower_95": decrease - 1.96 * decrease_se,
            "resolved_decrease": decrease - 1.96 * decrease_se > 0.0,
            "high_n_c_lower_1se": last["c_lower_1se"],
        }
    return out


def evaluate(summaries: list[dict]) -> dict:
    """Evaluate whether the full calibration campaign is warranted.

    Args:
        summaries: Screening cell summaries.

    Returns:
        Evidence tables and a conservative stop/go verdict.
    """
    rows = [
        row for summary in summaries if (row := alpha_estimate(summary)) is not None
    ]
    shape_rows = [
        r
        for r in rows
        if r["arm"] == "screen_shape"
        or (r["arm"] == "screen_taper" and r["n0"] == r["n1"] == 500)
    ]
    taper_rows = sorted(
        (r for r in rows if r["arm"] == "screen_taper"),
        key=lambda r: (r["shape"], r["n0"]),
    )
    imbalance_rows = [r for r in rows if r["arm"] == "screen_imbalance"]
    taper_by_shape = taper_diagnostics(taper_rows)

    shape_lower = min((r["c_lower_1se"] for r in shape_rows), default=float("nan"))
    gains = [r["area_gain_vs_c1"] for r in shape_rows]
    mean_gain = float(np.mean(gains)) if gains else float("nan")
    floor_flags = [r for r in rows if r["c_star"] + 1.96 * r["c_se"] < 1.0]

    imbalance_by_shape = {}
    for shape in sorted({r["shape"] for r in imbalance_rows}):
        group = [r for r in imbalance_rows if r["shape"] == shape]
        high = max(group, key=lambda r: r["c_star"])
        low = min(group, key=lambda r: r["c_star"])
        spread = high["c_star"] - low["c_star"]
        spread_se = float(np.hypot(high["c_se"], low["c_se"]))
        orientation = []
        for majority in sorted({max(r["n0"], r["n1"]) for r in group}):
            negative_majority = next(
                (
                    r
                    for r in group
                    if r["n0"] == majority and r["n1"] < r["n0"]
                ),
                None,
            )
            positive_majority = next(
                (
                    r
                    for r in group
                    if r["n1"] == majority and r["n0"] < r["n1"]
                ),
                None,
            )
            if negative_majority is None or positive_majority is None:
                continue
            contrast = negative_majority["c_star"] - positive_majority["c_star"]
            contrast_se = float(
                np.hypot(negative_majority["c_se"], positive_majority["c_se"])
            )
            orientation.append(
                {
                    "majority_n": majority,
                    "negative_minus_positive_c_star": contrast,
                    "ci95": [
                        contrast - 1.96 * contrast_se,
                        contrast + 1.96 * contrast_se,
                    ],
                }
            )
        imbalance_by_shape[shape] = {
            "min_c_lower_1se": min(r["c_lower_1se"] for r in group),
            "max_c_star_spread": spread,
            "spread_lower_simultaneous": (
                spread - IMBALANCE_SIMULTANEOUS_Z * spread_se
            ),
            "orientation_contrasts": orientation,
            "cells": group,
        }

    complete = {
        "shape": len(shape_rows) == EXPECTED_SHAPE_CELLS,
        "taper": len(taper_rows) == EXPECTED_TAPER_CELLS,
        "imbalance": len(imbalance_rows) == EXPECTED_IMBALANCE_CELLS,
    }
    useful = bool(
        complete["shape"]
        and shape_lower >= 1.0 + MIN_USEFUL_C_MARGIN
        and mean_gain >= MIN_USEFUL_AREA_GAIN
    )
    proceed = all(complete.values()) and useful and not floor_flags
    large_n_margin = max(
        (r["c_lower_1se"] - 1.0 for r in taper_rows if r["n0"] == 50_000),
        default=float("nan"),
    )
    strong_imbalance = any(
        value["spread_lower_simultaneous"] > MAX_SIMPLE_MAP_C_SPREAD
        for value in imbalance_by_shape.values()
    )
    if not all(complete.values()):
        verdict = (
            "INCONCLUSIVE: repair or rerun missing screen cells; do not launch Stage A"
        )
    elif proceed:
        verdict = "PROCEED to a reduced Stage A map fit"
    else:
        verdict = (
            "STOP: retain a documented fixed/default rule; full auto-map "
            "study is not justified"
        )
    return {
        "primary_alpha": PRIMARY_ALPHA,
        "n_usable_cells": len(rows),
        "shape_screen": {
            "lower_envelope_c_minus_1se": shape_lower,
            "mean_oracle_area_gain_vs_c1": mean_gain,
            "cells": shape_rows,
        },
        "taper_screen": {"by_shape": taper_by_shape, "cells": taper_rows},
        "imbalance_screen": imbalance_by_shape,
        "completeness": complete,
        "strong_c_below_1_flags": floor_flags,
        "thresholds": {
            "minimum_c_margin": MIN_USEFUL_C_MARGIN,
            "minimum_mean_area_gain": MIN_USEFUL_AREA_GAIN,
            "maximum_simple_map_c_spread": MAX_SIMPLE_MAP_C_SPREAD,
        },
        "recommendations": {
            "large_n": (
                "retain a focused large-n alpha=.05 fitting arm"
                if large_n_margin >= MIN_USEFUL_C_MARGIN
                else (
                    "do not run a dense large-n arm; validate a C=1 clamp "
                    "above the measured range"
                )
            ),
            "imbalance": (
                "retain the imbalance arm and plan for a directional or 2-D rule"
                if strong_imbalance
                else "test min(n0,n1) first; omit a broad 2-D sweep"
            ),
            "alpha": (
                "defer the full alpha grid until the alpha=.05 usefulness gate passes"
            ),
        },
        "verdict": verdict,
        "interpretation": (
            "This is a resource-allocation gate, not a coverage guarantee. "
            "A positive verdict still requires a constrained fit and fresh "
            "confirmation."
        ),
    }


def write_markdown(path: Path, result: dict) -> None:
    """Write a compact human-readable screening report.

    Args:
        path: Output Markdown path.
        result: Result from evaluate.
    """
    shape = result["shape_screen"]
    lines = [
        "# C-calibration screening verdict",
        "",
        f"**{result['verdict']}**",
        "",
        f"- Usable alpha=.05 cells: {result['n_usable_cells']}",
        (
            "- Shape lower envelope, C* minus one SE: "
            f"{shape['lower_envelope_c_minus_1se']:.3f}"
        ),
        (
            "- Mean oracle area gain versus C=1 on the shape screen: "
            f"{100 * shape['mean_oracle_area_gain_vs_c1']:.1f}%"
        ),
        f"- Strong C* < 1 flags: {len(result['strong_c_below_1_flags'])}",
        f"- Complete evidence arms: {result['completeness']}",
        "",
        result["interpretation"],
        "",
        "## Stage A routing",
        "",
        f"- Large n: {result['recommendations']['large_n']}",
        f"- Imbalance: {result['recommendations']['imbalance']}",
        f"- Alpha: {result['recommendations']['alpha']}",
        "",
        "The JSON companion contains the taper and directional-imbalance tables.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    """Run the screening analysis CLI."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--in",
        dest="input_dir",
        type=Path,
        default=Path("data/results/c_calibration/stageS"),
    )
    parser.add_argument("--out", type=Path, default=Path("data/results/c_calibration"))
    args = parser.parse_args(argv)
    result = evaluate(load_summaries(args.input_dir))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "screening_check.json").write_text(json.dumps(result, indent=1))
    write_markdown(args.out / "screening_report.md", result)
    print(result["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
