"""Pre-registered acceptance criteria A1-A3 against the Stage B results.

Spec section 8: auto mode ships as the library default iff, on the
confirmation runs,

- **A1 (validity):** auto coverage >= 1 - alpha - 1.0pp point estimate AND
  >= 1 - alpha - 2.5pp at the lower 95% CI bound, for every confirmation
  cell at alpha <= .10;
- **A2 (efficiency):** auto mean area <= the C = 1 arm's area in every
  cell, and >= 4% below it averaged over cells with n <= 1000;
- **A3 (no regret):** no confirmation cell where auto is both wider and
  lower-coverage than C = 2.

A4 (internal consistency of the D1-D6 decisions) is a human judgment over
the Stage A fit report and is not automated. Writes
``acceptance_check.json`` and prints a verdict.
"""

import argparse
import json
from pathlib import Path

import numpy as np

A1_POINT_TOL_PP = 1.0
A1_CI_TOL_PP = 2.5
A1_ALPHA_MAX = 0.10
A2_GAP_TARGET = 0.04
A2_SMALL_N = 1_000


def load_stage_b(stage_b_dir: Path) -> list[dict]:
    out = [
        json.loads(p.read_text())
        for p in sorted(stage_b_dir.glob("*.summary.json"))
    ]
    if not out:
        raise FileNotFoundError(f"No Stage B summaries under {stage_b_dir}")
    return out


def arm_table(summary: dict) -> dict[tuple[str, float], dict]:
    return {
        (ref["label"], ref["alpha"]): ref for ref in summary["aggregate"]["ref_maps"]
    }


def check(summaries: list[dict]) -> dict:
    a1_failures, a2_failures, a3_failures = [], [], []
    small_n_gaps = []
    for summary in summaries:
        cell = summary["meta"]["cell"]
        table = arm_table(summary)
        alphas = sorted({alpha for (_, alpha) in table})
        for alpha in alphas:
            auto = table.get(("auto", alpha))
            c1 = table.get(("c1", alpha))
            c2 = table.get(("c2", alpha))
            if auto is None or c1 is None or c2 is None:
                continue
            target = 1.0 - alpha
            if alpha <= A1_ALPHA_MAX:
                point_ok = auto["coverage"] >= target - A1_POINT_TOL_PP / 100.0
                ci_lo = auto["coverage"] - 1.96 * auto["coverage_se"]
                ci_ok = ci_lo >= target - A1_CI_TOL_PP / 100.0
                if not (point_ok and ci_ok):
                    a1_failures.append(
                        {
                            "cell": cell["name"],
                            "alpha": alpha,
                            "coverage": auto["coverage"],
                            "ci_lo": round(ci_lo, 4),
                            "target": target,
                        }
                    )
            if auto["area"] > c1["area"] + 1e-12:
                a2_failures.append(
                    {
                        "cell": cell["name"],
                        "alpha": alpha,
                        "auto_area": auto["area"],
                        "c1_area": c1["area"],
                    }
                )
            if alpha == 0.05 and min(cell["n0"], cell["n1"]) <= A2_SMALL_N:
                small_n_gaps.append(1.0 - auto["area"] / c1["area"])
            if auto["area"] > c2["area"] + 1e-12 and auto["coverage"] < c2["coverage"]:
                a3_failures.append(
                    {
                        "cell": cell["name"],
                        "alpha": alpha,
                        "auto": {"area": auto["area"], "coverage": auto["coverage"]},
                        "c2": {"area": c2["area"], "coverage": c2["coverage"]},
                    }
                )
    mean_gap = float(np.mean(small_n_gaps)) if small_n_gaps else None
    a2_gap_ok = mean_gap is not None and mean_gap >= A2_GAP_TARGET
    return {
        "A1": {"passed": not a1_failures, "failures": a1_failures},
        "A2": {
            "passed": not a2_failures and a2_gap_ok,
            "never_wider_failures": a2_failures,
            "mean_small_n_area_gap": mean_gap,
            "gap_target": A2_GAP_TARGET,
        },
        "A3": {"passed": not a3_failures, "failures": a3_failures},
        "verdict": (
            "SHIP auto as default"
            if not a1_failures and not a2_failures and a2_gap_ok and not a3_failures
            else "DO NOT ship as default (see failures; spec section 8 fallbacks)"
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="A1-A3 acceptance check")
    parser.add_argument(
        "--in",
        dest="stage_b_dir",
        type=Path,
        default=Path("data/results/c_calibration/stageB"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/results/c_calibration")
    )
    args = parser.parse_args(argv)
    result = check(load_stage_b(args.stage_b_dir))
    (args.out / "acceptance_check.json").write_text(json.dumps(result, indent=1))
    for crit in ("A1", "A2", "A3"):
        print(f"{crit}: {'PASSED' if result[crit]['passed'] else 'FAILED'}")
    print(result["verdict"])
    return 0 if result["verdict"].startswith("SHIP") else 1


if __name__ == "__main__":
    raise SystemExit(main())
