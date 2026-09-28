"""Analysis tables for the complement-trim experiment.

Usage, from the repository root::

    uv run --no-sync python -m scripts.complement_trim.analyze [--out DIR]

Reads the per-cell record files, flattens them to one row per
(cell, replicate, alpha, arm), and prints the tables used in the report:
per-cell coverage and paired width ratios, width ratios by block and size,
the location of the misses ``comp`` adds over ``hybrid`` (reconstructing the
floor region from the stored depth), budget-split decomposition, and
feature-conditional coverage. Also writes ``reps.feather`` and ``cells.csv``
next to the records.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.complement_trim.core import ARMS, REFERENCES
from scripts.complement_trim.design import all_cells, sample_replicate
from scripts.complement_trim.run import DEFAULT_OUT, wilson
from studroc_paper.methods.fiducial_ladder import khat_from_labels
from studroc_paper.methods.hybrid_floor import M3Floor, floor_region

BANDS = ARMS + REFERENCES


def load_records(root: Path) -> dict[str, tuple[dict, list[dict]]]:
    """Return ``{cell name: (meta, records)}`` for every stored cell."""
    out = {}
    for path in sorted((root / "cells").glob("*.json.gz")):
        with gzip.open(path, "rt") as handle:
            payload = json.load(handle)
        out[payload["meta"]["name"]] = (payload["meta"], payload["records"])
    return out


def flatten(cells: dict[str, tuple[dict, list[dict]]]) -> pd.DataFrame:
    """Flatten records to one row per (cell, rep, alpha, band)."""
    rows = []
    for name, (meta, records) in cells.items():
        for record in records:
            sampled = record["diagnostics"].get("feature_sampled")
            for key, level in record["levels"].items():
                for band in BANDS:
                    s = level["scores"][band]
                    diag = level[
                        "diag_half"
                        if band in ("comp_budget", "hybrid_half")
                        else "diag"
                    ]
                    rows.append(
                        {
                            "cell": name,
                            "block": meta["block"],
                            "n0": meta["n0"],
                            "n1": meta["n1"],
                            "rep": record["rep"],
                            "K": record["K"],
                            "feature_sampled": sampled,
                            "alpha": float(key),
                            "band": band,
                            "covered": s["covered"],
                            "below": s["below"],
                            "above": s["above"],
                            "miss_in": s["miss_in"],
                            "miss_out": s["miss_out"],
                            "depth": s["depth"],
                            "area": s["area"],
                            "area_in": s["area_in"],
                            "area_out": s["area_out"],
                            "low_runs": s["low_runs"],
                            "high_runs": s["high_runs"],
                            "j_full": diag["j_full"],
                            "j_raw": diag["j_raw"],
                            "j_comp": diag["j_comp"],
                            "enforced": diag["enforced"],
                            "fallback": diag["fallback"],
                            "region_frac": diag["region_frac"],
                            "n_draws": diag["n_draws"],
                            "trim_cols_full": diag["trim_cols_full"],
                            "trim_cols_comp": diag["trim_cols_comp"],
                        }
                    )
    return pd.DataFrame(rows)


def cell_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per (cell, alpha) coverage, paired width ratios, and depth summaries."""
    wide = df.pivot_table(
        index=["cell", "block", "n0", "n1", "alpha", "rep"],
        columns="band",
        values=["covered", "area", "area_out", "area_in"],
    )
    out = []
    for (cell, block, n0, n1, alpha), g in wide.groupby(
        level=["cell", "block", "n0", "n1", "alpha"]
    ):
        row = {"cell": cell, "block": block, "n0": n0, "n1": n1, "alpha": alpha}
        row["reps"] = len(g)
        for band in BANDS:
            cov = g[("covered", band)]
            row[f"cov_{band}"] = cov.mean()
            lo, hi = wilson(int(cov.sum()), len(cov))
            row[f"lo_{band}"], row[f"hi_{band}"] = lo, hi
            row[f"area_{band}"] = g[("area", band)].mean()
        for num, den in [
            ("comp", "hybrid"),
            ("comp_budget", "hybrid"),
            ("comp_budget", "hybrid_half"),
            ("hybrid_half", "hybrid"),
            ("hybrid", "raw"),
            ("comp", "raw"),
            ("m3", "hybrid"),
        ]:
            r = g[("area", num)] / g[("area", den)]
            row[f"r_{num}/{den}"] = r.mean()
            row[f"se_{num}/{den}"] = r.std(ddof=1) / np.sqrt(len(r))
        row["r_out_comp/hybrid"] = (
            g[("area_out", "comp")] / g[("area_out", "hybrid")]
        ).mean()
        out.append(row)
    table = pd.DataFrame(out)
    diag = (
        df[df.band.isin(["hybrid", "hybrid_half"])]
        .assign(level=lambda d: np.where(d.band == "hybrid", "full", "half"))
        .groupby(["cell", "alpha", "level"])
        .agg(
            j_full=("j_full", "mean"),
            j_comp=("j_comp", "mean"),
            j_ratio=("j_comp", lambda s: np.nan),
            enforced=("enforced", "sum"),
            fallback=("fallback", "mean"),
            region_frac=("region_frac", "mean"),
            M=("n_draws", "first"),
            K=("K", "mean"),
        )
        .drop(columns="j_ratio")
        .unstack("level")
    )
    diag.columns = [f"{a}_{b}" for a, b in diag.columns]
    return table.merge(diag.reset_index(), on=["cell", "alpha"])


def region_mask(meta: dict, rep: int, *, j_full: int, n_draws: int, level: float):
    """Reconstruct the floor region of one replicate from its stored depth."""
    cell = next(c for c in all_cells() if c.name == meta["name"])
    labels, _, _ = sample_replicate(cell, rep)
    khat = khat_from_labels(lab_s=labels)
    return floor_region(
        khat=khat,
        n_draws=n_draws,
        trim_depth=j_full,
        floor=M3Floor(rule="exact", alpha=level),
    )


def added_misses(
    df: pd.DataFrame, cells: dict[str, tuple[dict, list[dict]]]
) -> pd.DataFrame:
    """Describe each replicate where a complement arm misses and its parent covers.

    ``comp`` is paired with ``hybrid`` and ``comp_budget`` with
    ``hybrid_half``. For each such replicate, report the direction, the FPR
    location of the first violating grid point, and its grid distance to the
    nearest floor-region column (reconstructed from the stored ``j_full``).
    """
    idx = df.set_index(["cell", "rep", "alpha", "band"])
    rows = []
    for child, parent in [("comp", "hybrid"), ("comp_budget", "hybrid_half")]:
        c = idx.xs(child, level="band")
        p = idx.xs(parent, level="band")
        flip = c.index[(~c.covered) & (p.covered.reindex(c.index))]
        for cell, rep, alpha in flip:
            r = c.loc[(cell, rep, alpha)]
            meta = cells[cell][0]
            level = alpha if child == "comp" else alpha / 2
            region = region_mask(
                meta, rep, j_full=int(r.j_full), n_draws=int(r.n_draws), level=level
            )
            region_cols = np.flatnonzero(region)
            for direction, runs in (("low", r.low_runs), ("high", r.high_runs)):
                for start, stop in runs:
                    cols = np.arange(start, stop)
                    dist = (
                        np.min(np.abs(cols[:, None] - region_cols[None, :]))
                        if len(region_cols)
                        else np.inf
                    )
                    rows.append(
                        {
                            "pair": f"{child}/{parent}",
                            "cell": cell,
                            "block": meta["block"],
                            "n0": meta["n0"],
                            "alpha": alpha,
                            "rep": rep,
                            "direction": direction,
                            "fpr_start": start / meta["n0"],
                            "fpr_stop": (stop - 1) / meta["n0"],
                            "run_len": stop - start,
                            "grid_dist_to_region": dist,
                            "in_region": bool(region[cols].any()),
                            "left_cut": int(np.flatnonzero(~region)[0]) - 1
                            if (~region).any()
                            else meta["n0"],
                            "right_start": int(np.flatnonzero(~region)[-1]) + 1
                            if (~region).any()
                            else 0,
                            "depth": r.depth,
                        }
                    )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    """Entry point: build and write the flattened tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    cells = load_records(args.out)
    df = flatten(cells)
    table = cell_table(df)
    flips = added_misses(df, cells)
    df.drop(columns=["low_runs", "high_runs"]).to_feather(args.out / "reps.feather")
    table.to_csv(args.out / "cells.csv", index=False)
    flips.to_csv(args.out / "added_misses.csv", index=False)
    print(f"{len(cells)} cells, {len(df)} rows, {len(flips)} added-miss runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
