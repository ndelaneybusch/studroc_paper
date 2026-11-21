"""Hybrid / unfloored band area in the settings only one auto rule routes unfloored.

Pairs the two bands on the same datasets of the earlier studies, for every
score distribution they ran, and classifies each dataset by which of the
recommended and alternative rules send it to the unfloored band.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("stats/experiments").resolve()))
import auto_rule_surface as ars  # noqa: E402


def check(kind, threshold, a=0.0, b=0.0):
    return ars.Check(kind=kind, threshold=threshold, auc_exponent=a, ratio_exponent=b)


RULES = {
    "recommended": ars.Rule((check("floor", 160), check("corner", 18, 2.0))),
    "alternative": ars.Rule(
        (
            check("floor", 210),
            check("auc_cap", 0.94),
            check("imbalance_corner", 8, 2.0, 1 / 3),
        )
    ),
}
DIRECTORIES = [
    Path("data/results") / n
    for n in ("shape_auc_study", "shape_auc_study_p10", "nov11", "nov11_p10", "nov11_p30")
]

parts = []
for directory in DIRECTORIES:
    for path in sorted(directory.glob("*_individual.feather")):
        frame = pd.read_feather(
            path, columns=["method", "alpha", "true_auc", "n_pos", "n_neg", "band_area"]
        )
        frame = frame[np.isclose(frame["alpha"], ars.ALPHA)]
        plain = frame[frame["method"] == "fiducial"].reset_index(drop=True)
        hybrid = frame[frame["method"] == "fiducial_hybrid"].reset_index(drop=True)
        if hybrid.empty or len(plain) != len(hybrid):
            continue
        if not np.allclose(plain["true_auc"], hybrid["true_auc"]):
            raise ValueError(f"{path} does not pair by position")
        family = next((f for f in ars.FAMILIES if path.name.startswith(f)), "other")
        parts.append(
            pd.DataFrame(
                {
                    "family": family,
                    "auc": plain["true_auc"].to_numpy(float),
                    "n_min": np.minimum(plain["n_pos"], plain["n_neg"]),
                    "n_max": np.maximum(plain["n_pos"], plain["n_neg"]),
                    "area_plain": plain["band_area"].to_numpy(float),
                    "area_hybrid": hybrid["band_area"].to_numpy(float),
                }
            )
        )
pairs = pd.concat(parts, ignore_index=True)
pairs = pairs[np.isfinite(pairs["area_plain"]) & (pairs["area_plain"] > 0)]
args = {
    "n_min": pairs["n_min"].to_numpy(float),
    "auc": pairs["auc"].to_numpy(float),
    "ratio": (pairs["n_max"] / pairs["n_min"]).to_numpy(float),
}
rec = RULES["recommended"].admits(**args)
alt = RULES["alternative"].admits(**args)
pairs["domain"] = np.select(
    [rec & ~alt, alt & ~rec, rec & alt], ["recommended only", "alternative only", "both"], "neither"
)
pairs["penalty"] = pairs["area_hybrid"] / pairs["area_plain"] - 1
pairs["scope"] = np.where(pairs["family"] == "other", "other distributions", "two modelled families")

pd.set_option("display.width", 220)
summary = pairs.groupby(["domain", "scope"]).agg(
    datasets=("penalty", "size"),
    mean_penalty=("penalty", "mean"),
    median_penalty=("penalty", "median"),
    p90_penalty=("penalty", lambda x: x.quantile(0.9)),
    auc_lo=("auc", "min"),
    auc_hi=("auc", "max"),
)
print(summary.to_string(float_format=lambda v: f"{v:.3f}"))
for domain in ("recommended only", "alternative only"):
    part = pairs[pairs["domain"] == domain]
    print(f"\n== {domain}, by size and AUC band")
    part = part.assign(auc_band=pd.cut(part["auc"], [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]))
    table = part.groupby(["n_min", "n_max", "auc_band"], observed=True).agg(
        datasets=("penalty", "size"),
        mean_penalty=("penalty", "mean"),
        p90_penalty=("penalty", lambda x: x.quantile(0.9)),
    )
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))

print("\n== nearest paired sizes to the recommended-only domain")
near = pairs[
    pairs.set_index(["n_min", "n_max"]).index.isin(
        [(150, 150), (90, 210), (30, 270), (300, 2700), (100, 900), (5000, 5000), (3000, 7000), (1000, 9000)]
    )
]
near = near.assign(auc_band=pd.cut(near["auc"], [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.94, 0.96, 1.0]))
print(
    near.groupby(["n_min", "n_max", "auc_band"], observed=True)
    .agg(datasets=("penalty", "size"), mean_penalty=("penalty", "mean"), p90_penalty=("penalty", lambda x: x.quantile(0.9)))
    .to_string(float_format=lambda v: f"{v:.3f}")
)
