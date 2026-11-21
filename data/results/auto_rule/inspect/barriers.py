"""Locate the unsafe grid points that keep the routing rule from growing.

Usage: barriers.py GRID_FEATHER OUT_FEATHER
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path("stats/experiments").resolve()))
import auto_rule_surface as ars  # noqa: E402

grid = pd.read_feather(sys.argv[1])
routed = (grid["n_min"] * grid["ratio"] <= ars.ROUTED_N_MAX).to_numpy()
unsafe = ars.routed_unsafe(grid=grid, assurance=0.9)
mean_bad = (grid["p_mean"] > ars.ALPHA).to_numpy() & routed
print(f"routed points {routed.sum()}, unsafe {unsafe.sum()}, mean>5% {mean_bad.sum()}")


def best(unsafe_mask, label):
    ladder = ars.recommend(search=ars.search_rules(grid=grid, unsafe=unsafe_mask))
    print(f"\n== {label}")
    for _, r in ladder.iterrows():
        rounded = r["rounded"]
        args = {c: grid[c].to_numpy(float) for c in ("n_min", "auc", "ratio")}
        print(
            f"  {r['checks']} checks share {r['admitted_share']:.3f} "
            f"(rounded {rounded.admits(**args).mean():.3f})"
            f"{' *' if r['recommended'] else ''}: {rounded.describe()}"
        )
    return ladder


current = best(unsafe, "90% test")
limit = best(mean_bad, "mean-only (infinite data limit)")
for cut in (0.045, 0.04):
    best(unsafe & (grid["p_mean"] > cut).to_numpy(), f"unsafe and mean > {cut}")

rule = current.loc[current["recommended"], "fit"].item().rule
lim_rule = limit.loc[limit["recommended"], "fit"].item().rule
args = {c: grid[c].to_numpy(float) for c in ("n_min", "auc", "ratio")}
gain = lim_rule.admits(**args) & ~rule.admits(**args)
barrier = gain & unsafe
print(f"\nPoints the limit rule adds: {gain.sum()}, of which unsafe now {barrier.sum()}")
b = grid[barrier].copy()
pd.set_option("display.width", 200)
print(
    b.groupby(["worst_family", "worst_minority", "ratio"])
    .agg(
        points=("auc", "size"),
        n_lo=("n_min", "min"),
        n_hi=("n_min", "max"),
        auc_lo=("auc", "min"),
        auc_hi=("auc", "max"),
        mean_med=("p_mean", "median"),
        mean_hi=("p_mean", "max"),
        exc_med=("exceedance", "median"),
        shape_med=("worst_shape", "median"),
    )
    .to_string()
)
b.reset_index(drop=True).to_feather(sys.argv[2])
