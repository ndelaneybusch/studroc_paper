"""Information tables for the closure/split experiment.

Usage, from the repository root::

    uv run --no-sync python -m scripts.closure_split.analyze [--out DIR]

Writes ``t_reps.feather``, ``t_cells.csv``, ``s_cells.csv``,
``summary.json``, and ``report.md`` next to the records. The report is
organized by the questions in spec section 8. Each section is a contrast
table with the numbers needed to judge it; nothing is scored against a
threshold.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.closure_split.core import ARMS, NESTINGS

DEFAULT_OUT = Path("data/results/closure_split_20261027")
Z = 1.959964
CLOSURE_PAIRS = (("rec_T", "rec_W"), ("hyb_T", "hyb_W"))


def wilson(successes: int, count: int) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if count == 0:
        return (0.0, 1.0)
    p = successes / count
    centre = (p + Z * Z / (2 * count)) / (1 + Z * Z / count)
    half = Z * math.sqrt(p * (1 - p) / count + Z * Z / (4 * count * count))
    scale = 1 + Z * Z / count
    return (centre - half / scale, centre + half / scale)


def load(root: Path) -> dict[str, tuple[dict, list[dict]]]:
    """Return ``{cell name: (meta, records)}`` for every stored cell."""
    out = {}
    for path in sorted((root / "cells").glob("*.json.gz")):
        with gzip.open(path, "rt") as handle:
            payload = json.load(handle)
        out[payload["meta"]["name"]] = (payload["meta"], payload["records"])
    return out


def red_group(cell: str) -> str:
    """Red-team group tag (``R1`` ... ``R10``) of a cell name, else the block."""
    parts = cell.split("-")
    return parts[2] if len(parts) > 2 and parts[1] == "red" else parts[1]


def flatten_t(cells: dict) -> pd.DataFrame:
    """One row per (cell, rep, alpha) with every arm's scores as columns."""
    rows = []
    for name, (meta, records) in cells.items():
        if meta["study"] != "T":
            continue
        for record in records:
            for alpha, level in record["levels"].items():
                row = {
                    "cell": name,
                    "block": meta["block"],
                    "group": red_group(name),
                    "n0": meta["n0"],
                    "n1": meta["n1"],
                    "rep": record["rep"],
                    "alpha": float(alpha),
                    "K": record["K"],
                    "feature_count": record["diagnostics"].get("feature_count"),
                    **{f"d_{k}": v for k, v in level["diag"].items()},
                    **{f"c_{k}": v for k, v in level["checks"].items()},
                }
                for arm in ARMS:
                    s = level["scores"][arm]
                    for key in (
                        "covered",
                        "below",
                        "above",
                        "miss_in",
                        "miss_near",
                        "miss_far",
                        "depth",
                        "area",
                        "area_left",
                        "area_right",
                        "area_near",
                        "area_far",
                    ):
                        row[f"{arm}.{key}"] = s[key]
                rows.append(row)
    return pd.DataFrame(rows)


def _paired_ratio(frame: pd.DataFrame, arm: str, ref: str) -> tuple[float, float]:
    ratio = frame[f"{arm}.area"] / frame[f"{ref}.area"]
    return float(ratio.mean()), float(ratio.std(ddof=1) / math.sqrt(len(ratio)))


def t_cell_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per (cell, alpha) coverage, paired width, paired flips, and diagnostics."""
    out = []
    for (cell, alpha), g in df.groupby(["cell", "alpha"], sort=False):
        n = len(g)
        row = {
            "cell": cell,
            "alpha": alpha,
            "block": g["block"].iloc[0],
            "group": g["group"].iloc[0],
            "n0": g["n0"].iloc[0],
            "n1": g["n1"].iloc[0],
            "reps": n,
        }
        for arm in ARMS:
            hits = int(g[f"{arm}.covered"].sum())
            row[f"cov_{arm}"] = hits / n
            row[f"cov_{arm}_lo"], row[f"cov_{arm}_hi"] = wilson(hits, n)
            row[f"below_{arm}"] = float(g[f"{arm}.below"].mean())
            row[f"above_{arm}"] = float(g[f"{arm}.above"].mean())
            row[f"area_{arm}"] = float(g[f"{arm}.area"].mean())
        for arm, ref in (
            ("rec_T", "rec_W"),
            ("hyb_T", "hyb_W"),
            ("rec_W", "hyb_W"),
            ("rec_T", "hyb_W"),
            ("rec_T_ss", "rec_T"),
            ("m3_ss", "m3"),
            ("rec_T", "raw"),
            ("rec_T", "m3"),
            ("rec_T", "ks"),
            ("m3_ss", "ks"),
        ):
            row[f"r_{arm}/{ref}"], row[f"se_{arm}/{ref}"] = _paired_ratio(g, arm, ref)
        for arm, ref in (("rec_T", "rec_W"), ("hyb_T", "hyb_W"), ("rec_T_ss", "rec_T")):
            lost = g[f"{ref}.covered"] & ~g[f"{arm}.covered"]
            row[f"lost_{arm}/{ref}"] = float(lost.mean())
            row[f"nlost_{arm}/{ref}"] = int(lost.sum())
            row[f"gained_{arm}/{ref}"] = float(
                (~g[f"{ref}.covered"] & g[f"{arm}.covered"]).mean()
            )
            flips = g[lost]
            for where in ("miss_in", "miss_near", "miss_far", "below", "above"):
                row[f"flip_{arm}_{where}"] = (
                    float(flips[f"{arm}.{where}"].mean()) if len(flips) else np.nan
                )
        row["j"] = float(g["d_j"].mean())
        row["j_rec"] = float(g["d_j_rec"].mean())
        row["region_frac"] = float(g["d_region_frac"].mean())
        row["fallback_frac"] = float((g["d_comp_rows"] == 0).mean())
        row["K"] = float(g["K"].mean())
        nest_cols = [f"c_nest_{a}_in_{b}" for a, b in NESTINGS]
        row["max_nest_violation"] = float(g[nest_cols].to_numpy().max())
        row["tight_hull_mismatch"] = int(
            (~g["c_tight_equals_hull_hyb"]).sum()
            + (~g["c_tight_equals_hull_rec"]).sum()
        )
        if g["feature_count"].notna().any():
            for label, mask in (
                ("unsampled", g["feature_count"] == 0),
                ("sampled", g["feature_count"] > 0),
            ):
                row[f"n_{label}"] = int(mask.sum())
                for arm in ("raw", "hyb_W", "rec_W", "rec_T", "hyb_T"):
                    row[f"cov_{arm}_{label}"] = (
                        float(g.loc[mask, f"{arm}.covered"].mean())
                        if mask.any()
                        else np.nan
                    )
        out.append(row)
    return pd.DataFrame(out)


def integrity(t_cells: pd.DataFrame) -> pd.DataFrame:
    """Implementation status: the exact properties of spec section 3.2."""
    return pd.DataFrame(
        [
            {
                "max nesting violation": float(t_cells["max_nest_violation"].max()),
                "tightened/unclosed coverage mismatches": int(
                    t_cells["tight_hull_mismatch"].sum()
                ),
                "cell-alpha pairs with recalibration fallback": int(
                    (t_cells["fallback_frac"] > 0).sum()
                ),
            }
        ]
    )


def closure_by_block(t_cells: pd.DataFrame) -> pd.DataFrame:
    """Q1: what each tightened arm buys and costs, by block and alpha."""
    rows = []
    for (alpha, block), g in t_cells.groupby(["alpha", "block"]):
        for arm, ref in CLOSURE_PAIRS:
            ratio = g[f"r_{arm}/{ref}"]
            lost = g[f"lost_{arm}/{ref}"]
            rows.append(
                {
                    "alpha": alpha,
                    "block": block,
                    "contrast": f"{arm} vs {ref}",
                    "cells": len(g),
                    "width ratio (macro)": float(ratio.mean()),
                    "width ratio (min)": float(ratio.min()),
                    "width ratio (max)": float(ratio.max()),
                    "lost cov (macro)": float(lost.mean()),
                    "lost cov (max cell)": float(lost.max()),
                    "gained cov (macro)": float(g[f"gained_{arm}/{ref}"].mean()),
                    f"cov {ref} (macro)": float(g[f"cov_{ref}"].mean()),
                    f"cov {arm} (macro)": float(g[f"cov_{arm}"].mean()),
                    f"cov {arm} (min cell)": float(g[f"cov_{arm}"].min()),
                    "net width vs hyb_W": float(g[f"r_{arm}/hyb_W"].mean())
                    if f"r_{arm}/hyb_W" in g
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def flip_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Q2: where the misses that tightening adds land, by block and alpha.

    A flip is a replicate the widened band covers and the tightened band
    misses. Location shares can sum above one when a replicate misses in
    several places.
    """
    rows = []
    for (alpha, block), g in df.groupby(["alpha", "block"]):
        for arm, ref in CLOSURE_PAIRS:
            flips = g[g[f"{ref}.covered"] & ~g[f"{arm}.covered"]]
            if flips.empty:
                rows.append(
                    {
                        "alpha": alpha,
                        "block": block,
                        "contrast": f"{arm} vs {ref}",
                        "flips": 0,
                    }
                )
                continue
            rows.append(
                {
                    "alpha": alpha,
                    "block": block,
                    "contrast": f"{arm} vs {ref}",
                    "flips": len(flips),
                    "of reps": len(g),
                    "in region": float(flips[f"{arm}.miss_in"].mean()),
                    "adjacent strip": float(flips[f"{arm}.miss_near"].mean()),
                    "elsewhere": float(flips[f"{arm}.miss_far"].mean()),
                    "lower edge": float(flips[f"{arm}.below"].mean()),
                    "upper edge": float(flips[f"{arm}.above"].mean()),
                    "median depth": float(flips[f"{arm}.depth"].median()),
                }
            )
    return pd.DataFrame(rows)


def red_team(t_cells: pd.DataFrame) -> pd.DataFrame:
    """Q3: every red cell side by side, with feature-conditional coverage."""
    red = t_cells[t_cells["block"] == "red"]
    cols = [
        "group",
        "cell",
        "alpha",
        "reps",
        "region_frac",
        "cov_raw",
        "cov_hyb_W",
        "cov_rec_W",
        "cov_rec_T",
        "cov_hyb_T",
        "lost_rec_T/rec_W",
        "flip_rec_T_miss_near",
        "flip_rec_T_below",
        "r_rec_T/rec_W",
        "r_hyb_T/hyb_W",
    ]
    extra = [
        c
        for c in (
            "n_unsampled",
            "cov_rec_W_unsampled",
            "cov_rec_T_unsampled",
            "n_sampled",
            "cov_rec_W_sampled",
            "cov_rec_T_sampled",
        )
        if c in red
    ]
    return red[cols + extra].sort_values(["alpha", "group", "cell"])


def lowest_cells(t_cells: pd.DataFrame, *, count: int = 15) -> pd.DataFrame:
    """Q4: the lowest-covering cells for the tightened arms, beside incumbents."""
    a05 = t_cells[t_cells["alpha"] == 0.05].copy()
    a05["min tightened cov"] = a05[["cov_rec_T", "cov_hyb_T"]].min(axis=1)
    cols = [
        "cell",
        "reps",
        "cov_hyb_W",
        "cov_rec_W",
        "cov_rec_T",
        "cov_rec_T_lo",
        "cov_rec_T_hi",
        "cov_hyb_T",
        "cov_raw",
        "lost_rec_T/rec_W",
        "region_frac",
    ]
    return a05.nsmallest(count, "min tightened cov")[cols]


def typical_by_size(t_cells: pd.DataFrame) -> pd.DataFrame:
    """Q5: the paper's population, by class sizes."""
    typ = t_cells[t_cells["block"] == "typical"].copy()
    typ["sizes"] = typ["n0"].astype(str) + "x" + typ["n1"].astype(str)
    cols = [
        "region_frac",
        "cov_hyb_W",
        "cov_rec_W",
        "cov_rec_T",
        "cov_hyb_T",
        "r_rec_T/rec_W",
        "r_hyb_T/hyb_W",
        "r_rec_T/hyb_W",
        "lost_rec_T/rec_W",
        "lost_hyb_T/hyb_W",
        "r_rec_T/m3",
        "r_rec_T/ks",
    ]
    return typ.groupby(["alpha", "sizes"])[cols].mean()


def calibration(t_cells: pd.DataFrame) -> pd.DataFrame:
    """Q6: macro coverage and width against production, alpha = .5."""
    half = t_cells[t_cells["alpha"] == 0.5]
    cols = [
        f"cov_{a}" for a in ("raw", "hyb_W", "hyb_T", "rec_W", "rec_T", "m3", "m3_ss")
    ]
    widths = ["r_hyb_T/hyb_W", "r_rec_W/hyb_W", "r_rec_T/hyb_W"]
    return half.groupby("block")[cols + widths].mean()


def split_inside_t(t_cells: pd.DataFrame) -> pd.DataFrame:
    """Q7: the sample-size split on the Study T cells, standalone and in the floor."""
    imb = t_cells[t_cells["n0"] != t_cells["n1"]]
    cols = [
        "cell",
        "alpha",
        "r_m3_ss/m3",
        "r_rec_T_ss/rec_T",
        "lost_rec_T_ss/rec_T",
        "gained_rec_T_ss/rec_T",
        "cov_m3",
        "cov_m3_ss",
    ]
    return imb[cols].sort_values(["alpha", "cell"])


def s_cell_table(cells: dict) -> pd.DataFrame:
    """Per (cell, alpha) split comparison against Sidak, with the oracle split."""
    out = []
    for name, (meta, records) in cells.items():
        if meta["study"] != "S":
            continue
        for alpha in records[0]["levels"]:
            arms = list(records[0]["levels"][alpha])
            area = {
                a: np.array([r["levels"][alpha][a]["area"] for r in records])
                for a in arms
            }
            row = {
                "cell": name,
                "alpha": float(alpha),
                "block": meta["block"],
                "shape": name.split("--")[0][2:],
                "n0": meta["n0"],
                "n1": meta["n1"],
                "sizes": f"{meta['n0']}x{meta['n1']}",
                "reps": len(records),
            }
            for arm in arms:
                cov = np.array([r["levels"][alpha][arm]["covered"] for r in records])
                row[f"cov_{arm}"] = float(cov.mean())
                row[f"cov_{arm}_hi"] = wilson(int(cov.sum()), len(cov))[1]
                row[f"below_{arm}"] = float(
                    np.mean([r["levels"][alpha][arm]["below"] for r in records])
                )
                row[f"above_{arm}"] = float(
                    np.mean([r["levels"][alpha][arm]["above"] for r in records])
                )
                ratio = area[arm] / area["sidak"]
                row[f"r_{arm}"] = float(ratio.mean())
                row[f"se_{arm}"] = float(ratio.std(ddof=1) / math.sqrt(len(ratio)))
            sweep = {
                a: row[f"r_{a}"] for a in arms if a.startswith("rho") or a == "sidak"
            }
            best = min(sweep, key=sweep.get)
            row["oracle_arm"] = best
            row["oracle_ratio"] = sweep[best]
            gain = 1.0 - row["oracle_ratio"]
            row["sqrt_capture"] = (
                (1.0 - row["r_sqrt"]) / gain if gain > 1e-9 else np.nan
            )
            for band in ("0", "0.1", "0.5", "0.9"):
                s = np.array(
                    [r["levels"][alpha]["sqrt"][f"area_{band}"] for r in records]
                )
                b = np.array(
                    [r["levels"][alpha]["sidak"][f"area_{band}"] for r in records]
                )
                row[f"dwidth_sqrt_{band}"] = float((s - b).mean())
            out.append(row)
    return pd.DataFrame(out)


def split_by_sizes(s_cells: pd.DataFrame) -> pd.DataFrame:
    """Q8: each rule against Sidak by class-size configuration, over shapes."""
    rows = []
    for (alpha, sizes), g in s_cells.groupby(["alpha", "sizes"], sort=False):
        rows.append(
            {
                "alpha": alpha,
                "sizes": sizes,
                "sqrt (macro)": float(g["r_sqrt"].mean()),
                "sqrt (min)": float(g["r_sqrt"].min()),
                "sqrt (max)": float(g["r_sqrt"].max()),
                "linear (macro)": float(g["r_linear"].mean()),
                "linear (max)": float(g["r_linear"].max()),
                "oracle (macro)": float(g["oracle_ratio"].mean()),
                "sqrt capture (median)": float(g["sqrt_capture"].median()),
            }
        )
    return pd.DataFrame(rows)


def split_by_shape(s_cells: pd.DataFrame) -> pd.DataFrame:
    """Q9: the sqrt rule by shape and configuration at alpha = .05."""
    a05 = s_cells[(s_cells["alpha"] == 0.05) & (s_cells["block"] != "balanced")]
    return a05.pivot_table(index="shape", columns="sizes", values="r_sqrt")


def split_width_location(s_cells: pd.DataFrame) -> pd.DataFrame:
    """Q10: where along FPR the sqrt rule moves width, by configuration."""
    cols = [f"dwidth_sqrt_{b}" for b in ("0", "0.1", "0.5", "0.9")]
    a05 = s_cells[(s_cells["alpha"] == 0.05) & (s_cells["block"] != "balanced")]
    return a05.groupby("sizes")[cols].mean()


def split_calibration(s_cells: pd.DataFrame) -> pd.DataFrame:
    """Q11: coverage and direction by rule at each alpha."""
    cols = []
    for arm in ("sidak", "sqrt", "linear", "ks"):
        cols += [f"cov_{arm}", f"below_{arm}", f"above_{arm}"]
    return s_cells.groupby(["alpha", "block"])[cols].mean()


def _fmt(value) -> str:
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.4f}"
    if isinstance(value, tuple | list):
        return "(" + ", ".join(_fmt(v) for v in value) + ")"
    return str(value)


def _md(frame: pd.DataFrame, *, index: bool = True) -> str:
    """Render a small frame as a Markdown table without optional dependencies."""
    table = frame.reset_index() if index else frame
    header = "| " + " | ".join(str(c) for c in table.columns) + " |"
    rule = "|" + "---|" * len(table.columns)
    body = [
        "| " + " | ".join(_fmt(v) for v in row) + " |"
        for row in table.itertuples(index=False)
    ]
    return "\n".join([header, rule, *body])


def build_tables(
    t_df: pd.DataFrame, t_cells: pd.DataFrame, s_cells: pd.DataFrame
) -> dict[str, tuple[str, pd.DataFrame, bool]]:
    """Every report table as ``{key: (heading, frame, keep index)}``."""
    tables: dict[str, tuple[str, pd.DataFrame, bool]] = {}
    if len(t_cells):
        tables |= {
            "integrity": (
                "Implementation checks (spec 3.2)",
                integrity(t_cells),
                False,
            ),
            "closure": (
                "Q1. What tightening buys and costs",
                closure_by_block(t_cells),
                False,
            ),
            "flips": ("Q2. Where the added misses land", flip_locations(t_df), False),
            "red": ("Q3. Red-team cells", red_team(t_cells), False),
            "lowest": (
                "Q4. Lowest-covering cells at alpha = .05",
                lowest_cells(t_cells),
                False,
            ),
            "typical": (
                "Q5. The paper's population by class sizes",
                typical_by_size(t_cells),
                True,
            ),
            "calibration": (
                "Q6. Coverage and width at alpha = .5",
                calibration(t_cells),
                True,
            ),
            "split_t": (
                "Q7. The split on Study T's imbalanced cells",
                split_inside_t(t_cells),
                False,
            ),
        }
    if len(s_cells):
        tables |= {
            "split_sizes": (
                "Q8. Split rules against Sidak by class sizes",
                split_by_sizes(s_cells),
                False,
            ),
            "split_shape": (
                "Q9. Sqrt rule by shape, alpha = .05",
                split_by_shape(s_cells),
                True,
            ),
            "split_where": (
                "Q10. Where the sqrt rule moves width (FPR bands)",
                split_width_location(s_cells),
                True,
            ),
            "split_cov": (
                "Q11. Coverage and direction by split rule",
                split_calibration(s_cells),
                True,
            ),
        }
    return tables


def report(tables: dict[str, tuple[str, pd.DataFrame, bool]]) -> str:
    """Render the tables as Markdown, in question order."""
    lines = [
        "# Closure/split experiment: generated tables",
        "",
        "Questions and reading notes are in "
        "`stats/experiments/closure_split_spec.md` section 8. Width ratios are "
        "paired per-replicate means; coverage changes are paired flip rates.",
        "",
    ]
    for heading, frame, keep_index in tables.values():
        lines += [f"## {heading}", "", _md(frame.round(4), index=keep_index), ""]
    return "\n".join(lines)


def summarize(root: Path) -> dict[str, pd.DataFrame]:
    """Write every table, the per-cell CSVs, and the report."""
    cells = load(root)
    t_df = flatten_t(cells)
    t_cells = t_cell_table(t_df) if len(t_df) else pd.DataFrame()
    s_cells = s_cell_table(cells)
    if len(t_df):
        t_df.to_feather(root / "t_reps.feather")
        t_cells.to_csv(root / "t_cells.csv", index=False)
    if len(s_cells):
        s_cells.to_csv(root / "s_cells.csv", index=False)
    tables = build_tables(t_df, t_cells, s_cells)
    (root / "report.md").write_text(report(tables), encoding="utf-8")
    summary = {
        key: json.loads(frame.reset_index().to_json(orient="records"))
        for key, (_, frame, _) in tables.items()
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=1))
    return {key: frame for key, (_, frame, _) in tables.items()}


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    summarize(parser.parse_args(argv).out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
