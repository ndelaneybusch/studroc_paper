"""Write a design of draws around the barrier points of the routing rule.

Usage: target_design.py BARRIER_FEATHER ROUND_INDEX CPU_HOURS SEED [MAX_MEAN [SHAPE_JITTER]]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import norm

sys.path.insert(0, str(Path("stats/experiments").resolve()))
import auto_rule_surface as ars  # noqa: E402

barrier = pd.read_feather(sys.argv[1])
round_index = int(sys.argv[2])
cpu_hours = float(sys.argv[3])
rng = np.random.default_rng(int(sys.argv[4]))
max_mean = float(sys.argv[5]) if len(sys.argv) > 5 else 0.048
shape_jitter = float(sys.argv[6]) if len(sys.argv) > 6 else 0.3
out = Path("data/results/auto_rule") / f"design_round{round_index}.feather"
if out.exists():
    raise SystemExit(f"{out} exists")

points = barrier[barrier["p_mean"] <= max_mean].reset_index(drop=True)
# Effective draws for the posterior to clear 90% assurance at the current mean
gap = logit(ars.ALPHA) - logit(points["p_mean"])
need = (norm.ppf(0.9) / gap) ** 2 / (points["p_mean"] * (1 - points["p_mean"]))
points["weight"] = need.clip(300, 3000)

log_n_step = np.diff(np.log(ars.N_MIN_BOUNDS))[0] / (ars.GRID_N - 1)
probit_step = np.diff(norm.ppf(ars.AUC_BOUNDS))[0] / (ars.GRID_AUC - 1)
negative_minority = (points["worst_minority"] == "negative").to_numpy()


def draws_for(point: pd.Series, count: int) -> pd.DataFrame:
    family = ars.FAMILIES[point["worst_family"]]
    n_min = np.exp(np.log(point["n_min"]) + rng.uniform(-0.5, 0.5, count) * log_n_step)
    probit = norm.ppf(point["auc"]) + rng.uniform(-0.5, 0.5, count) * probit_step
    auc = norm.cdf(np.clip(probit, *norm.ppf(ars.AUC_BOUNDS)))
    ratio = np.clip(
        point["ratio"] * np.exp(rng.uniform(-0.15, 0.15, count)), 1, ars.RATIO_MAX
    )
    lo, hi = np.log(family.shape_bounds)
    shape = np.exp(
        np.clip(np.log(point["worst_shape"]) + rng.uniform(-shape_jitter, shape_jitter, count), lo, hi)
    )
    n_min = np.rint(n_min).astype(int)
    n_max = np.rint(n_min * ratio).astype(int)
    minority_negative = point["worst_minority"] == "negative"
    return pd.DataFrame(
        {
            "family": family.name,
            "shape": shape,
            "auc": auc,
            "n_pos": n_max if minority_negative else n_min,
            "n_neg": n_min if minority_negative else n_max,
        }
    )


n_major = points["n_min"] * points["ratio"]
per_point_cost = ars.cost_seconds(
    pd.DataFrame(
        {
            "n_pos": np.where(negative_minority, n_major, points["n_min"]),
            "n_neg": np.where(negative_minority, points["n_min"], n_major),
        }
    )
)
scale = cpu_hours * 3600 / float((points["weight"] * per_point_cost).sum())
counts = np.maximum(np.rint(points["weight"] * scale).astype(int), 1)
design = pd.concat(
    [draws_for(p, c) for (_, p), c in zip(points.iterrows(), counts, strict=True)],
    ignore_index=True,
)
design["criterion"] = "target"
inside = ars.in_design_space(frame=design.assign(source="target", miss=False))
if len(inside) != len(design):
    raise SystemExit(f"{len(design) - len(inside)} draws outside the design space")
order = np.argsort(-ars.cost_seconds(design), kind="stable")
design = design.iloc[order].reset_index(drop=True)
design.insert(0, "draw", np.arange(len(design)))
print(
    f"{len(points)} points, {len(design):,} draws, "
    f"{ars.cost_seconds(design).sum() / 3600:.1f} modelled CPU-hours"
)
print(
    points.assign(draws=counts)
    .groupby(["worst_family", "worst_minority", "ratio"])["draws"]
    .sum()
    .to_string()
)
design.to_feather(out)
print(f"Saved {out}")
