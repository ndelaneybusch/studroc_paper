"""Small-sample hybrid, tail-overlap, and genuinely unprotected-interior studies."""

import time
from itertools import product

import numpy as np

from studroc_paper.methods.fiducial_band import _auto_n_draws, production_trim_rows

from .common import (
    Curve,
    Store,
    curve,
    fiducial_edges,
    interval,
    m3_edges,
    metrics,
    paired_summary,
    rng_for,
    runs,
)
from .hybrid import floor_components, geometry, hull

SIZES = [
    (10, 10),
    (15, 15),
    (20, 20),
    (30, 30),
    (50, 50),
    (10, 50),
    (50, 10),
    (20, 50),
    (50, 20),
]
SHAPES = [
    "diagonal",
    "normal_0.6",
    "normal_0.95",
    "t2_0.95",
    "kink",
    "sliver",
    "interior_sliver",
    "jump",
    "endpoint_atoms",
]
REPLICATES = 500


def study_curve(*, name: str, n0: int, n1: int) -> Curve:
    """Include support-gap endpoint mass as well as smooth and rare-mass truths."""
    if name == "endpoint_atoms":
        return Curve(
            name=name,
            x=np.array([0.0, 0.0, 0.5, 1.0, 1.0]),
            y=np.array([0.0, 0.15, 0.75, 0.9, 1.0]),
        )
    return curve(name=name, n0=n0, n1=n1)


def sample_details(
    *, truth: Curve, n0: int, n1: int, rng: np.random.Generator
) -> tuple:
    """Sample ranks while retaining latent rare-mass observations for diagnosis."""
    negatives = rng.random(n0)
    positives = truth.inverse(uniforms=rng.random(n1))
    labels = np.r_[np.zeros(n0, dtype=np.uint8), np.ones(n1, dtype=np.uint8)][
        np.argsort(np.r_[negatives, positives], kind="stable")
    ]
    rare_count = None
    if truth.name in {"sliver", "interior_sliver"}:
        end = 1.0 if truth.name == "sliver" else 0.6
        start = end - min(0.05, 1 / n0)
        rare_count = int(np.count_nonzero((positives > start) & (positives <= end)))
    positive_before = np.cumsum(labels)[labels == 0]
    return labels, {
        "rare_count": rare_count,
        "no_rare_observation": rare_count == 0 if rare_count is not None else None,
        "empirical_auc": float(positive_before.sum() / (n0 * n1)),
        "complete_separation": bool(np.all(labels[:n1] == 1)),
    }


def regional_metrics(
    *, mask: np.ndarray, lower: np.ndarray, upper: np.ndarray, values: np.ndarray
) -> dict:
    """Report empty regions as absent and integrate only wholly contained cells."""
    miss_low, miss_high = values < lower - 1e-12, values > upper + 1e-12
    cells = mask[:-1] & mask[1:]
    width = upper[1:] - lower[:-1]
    return {
        "points": int(mask.sum()),
        "cell_fraction": float(cells.mean()),
        "below": bool(np.any(miss_low & mask)) if mask.any() else None,
        "above": bool(np.any(miss_high & mask)) if mask.any() else None,
        "miss": bool(np.any((miss_low | miss_high) & mask)) if mask.any() else None,
        "area": float(width[cells].sum() / len(cells)),
        "mean_width": float(width[cells].mean()) if cells.any() else None,
    }


def measure(
    *,
    labels: np.ndarray,
    truth: Curve,
    seed: int,
    threads: int,
    draw_multiplier: int = 1,
) -> dict:
    """Compare exact hybrid, each tail ablation, Stage F, raw C=1 and full M3."""
    n0 = len(labels) - int(labels.sum())
    grid = np.arange(n0 + 1) / n0
    values = truth.evaluate(grid=grid)
    result = {}
    for alpha in [0.05, 0.5]:
        draws = draw_multiplier * _auto_n_draws(n_grid=len(grid), alpha_eff=alpha)
        raw, depths = fiducial_edges(
            labels=labels,
            truth=truth,
            draws=draws,
            seed=seed,
            alphas=[alpha],
            trim_rows=production_trim_rows(len(grid)),
            threads=threads,
        )
        depth = int(depths[0])
        _, lo3, hi3 = m3_edges(labels=labels, alpha=alpha)
        m3 = (lo3, hi3)
        left, right = floor_components(labels=labels, draws=draws, depth=depth)
        old_left, old_right = floor_components(
            labels=labels, draws=draws, depth=depth, rule="stage_f"
        )
        masks = {
            "left_floor": left,
            "right_floor": right,
            "overlap": left & right,
            "protected": left | right,
            "unfloored": ~(left | right),
        }
        arms = {
            "raw": raw[0],
            "m3": m3,
            "hybrid": hull(raw=raw[0], m3=m3, region=left | right),
            "left_only": hull(raw=raw[0], m3=m3, region=left),
            "right_only": hull(raw=raw[0], m3=m3, region=right),
            "stage_f": hull(raw=raw[0], m3=m3, region=old_left | old_right),
        }
        scores = {}
        for name, (lo, hi) in arms.items():
            scores[name] = {
                **metrics(grid=grid, lower=lo, upper=hi, truth=truth),
                "regions": {
                    region: regional_metrics(
                        mask=mask, lower=lo, upper=hi, values=values
                    )
                    for region, mask in masks.items()
                },
            }
        changed = (arms["hybrid"][0] < raw[0][0] - 1e-12) | (
            arms["hybrid"][1] > raw[0][1] + 1e-12
        )
        result[str(alpha)] = {
            "draws": draws,
            "depth": depth,
            "geometry": geometry(labels=labels, left=left, right=right),
            "stage_f_geometry": geometry(labels=labels, left=old_left, right=old_right),
            "left_runs": runs(mask=left),
            "right_runs": runs(mask=right),
            "changed_unfloored_points": int(
                np.count_nonzero(changed & masks["unfloored"])
            ),
            "raw_failure_repaired": not scores["raw"]["covered"]
            and scores["hybrid"]["covered"],
            "arms": scores,
        }
    return result


def summarize(*, store: Store, expected: int, pilot: bool) -> None:
    """Describe paired improvements, tail geometry and conditional failure rates."""
    groups = {}
    for row in store.records:
        if not row.get("audit"):
            for alpha, data in row["alphas"].items():
                groups.setdefault(
                    (row["n0"], row["n1"], row["shape"], alpha), []
                ).append((row, data))
    cells = []
    for key, records in groups.items():
        data = [d for _, d in records]
        count = len(data)
        arms = {}
        for name in data[0]["arms"]:
            covered = sum(d["arms"][name]["covered"] for d in data)
            regional = {}
            for region in data[0]["arms"][name]["regions"]:
                present = [
                    d["arms"][name]["regions"][region]
                    for d in data
                    if d["arms"][name]["regions"][region]["miss"] is not None
                ]
                covered_here = sum(not r["miss"] for r in present)
                regional[region] = {
                    "exposed_replicates": len(present),
                    "coverage": covered_here / len(present) if present else None,
                    "coverage_ci": interval(successes=covered_here, count=len(present))
                    if present
                    else None,
                }
            pointwise_low, pointwise_high = np.zeros(key[0] + 1), np.zeros(key[0] + 1)
            for d in data:
                for side, accumulator in [
                    ("low_runs", pointwise_low),
                    ("high_runs", pointwise_high),
                ]:
                    for first, last in d["arms"][name][side]:
                        accumulator[first:last] += 1
            arms[name] = {
                "coverage": covered / count,
                "coverage_ci": interval(successes=covered, count=count),
                "area_to_raw": paired_summary(
                    values=[
                        d["arms"][name]["area"] / d["arms"]["raw"]["area"] for d in data
                    ]
                ),
                "area_to_m3": paired_summary(
                    values=[
                        d["arms"][name]["area"] / d["arms"]["m3"]["area"] for d in data
                    ]
                ),
                "pointwise_below": (pointwise_low / count).tolist(),
                "pointwise_above": (pointwise_high / count).tolist(),
                "regions": regional,
            }
        conditions = {}
        for condition in [
            "no_rare_observation",
            "complete_separation",
            "fully_floored",
        ]:
            selected = [
                d
                for r, d in records
                if (
                    d["geometry"][condition]
                    if condition == "fully_floored"
                    else r["observables"][condition]
                )
            ]
            conditions[condition] = {
                "replicates": len(selected),
                "coverage": {
                    name: {
                        "estimate": sum(d["arms"][name]["covered"] for d in selected)
                        / len(selected),
                        "ci": interval(
                            successes=sum(d["arms"][name]["covered"] for d in selected),
                            count=len(selected),
                        ),
                    }
                    for name in arms
                }
                if selected
                else None,
            }
        geo = {
            field: paired_summary(values=[float(d["geometry"][field]) for d in data])
            for field in [
                "left_end_fpr",
                "right_start_fpr",
                "unfloored_points",
                "unfloored_cell_fraction",
            ]
        }
        events = {
            field: {
                "rate": sum(bool(d["geometry"][field]) for d in data) / count,
                "ci": interval(
                    successes=sum(bool(d["geometry"][field]) for d in data), count=count
                ),
            }
            for field in ["fully_floored", "no_unfloored_cells"]
        }
        cells.append(
            {
                "cell": key,
                "replicates": count,
                "arms": arms,
                "geometry": geo,
                "geometry_events": events,
                "conditional": conditions,
                "window_eligibility_rate": np.mean(
                    [d["geometry"]["window"]["eligible"] for d in data]
                ).item(),
                "repair_probability": np.mean(
                    [d["raw_failure_repaired"] for d in data]
                ).item(),
            }
        )
    done = sum(not row.get("audit", False) for row in store.records)
    store.save(
        name="summary.json",
        payload={
            "complete": done == expected,
            "pilot": pilot,
            "expected_units": expected,
            "completed_units": done,
            "eligible": False,
            "scope": (
                "Descriptive small-n hybrid study; conditional regional rates "
                "are not honesty guarantees."
            ),
            "cells": cells,
        },
    )


def run(*, store: Store, pilot: bool, threads: int) -> None:
    """Run native-grid small-sample studies with production budgets and tail rules."""
    sizes = [(10, 10), (50, 50), (10, 50)] if pilot else SIZES
    shapes = ["normal_0.95", "interior_sliver", "endpoint_atoms"] if pilot else SHAPES
    reps = 3 if pilot else REPLICATES
    expected = len(sizes) * len(shapes) * reps
    try:
        for (n0, n1), shape in product(sizes, shapes):
            truth = study_curve(name=shape, n0=n0, n1=n1)
            for rep in range(reps):
                key = f"{n0}/{n1}/{shape}/{rep}"
                if key in store.keys:
                    continue
                rng = rng_for("small_n", key)
                labels, observables = sample_details(truth=truth, n0=n0, n1=n1, rng=rng)
                seed = int(rng.integers(2**63))
                start = time.perf_counter()
                data = measure(labels=labels, truth=truth, seed=seed, threads=threads)
                if (
                    rep == 0
                    and n0 == n1
                    and shape in {"normal_0.95", "interior_sliver"}
                ):
                    audit = measure(
                        labels=labels,
                        truth=truth,
                        seed=seed,
                        threads=threads,
                        draw_multiplier=2,
                    )
                    store.add(
                        key=f"audit/{key}",
                        audit=True,
                        n0=n0,
                        n1=n1,
                        shape=shape,
                        rep=rep,
                        observables=observables,
                        alphas=audit,
                    )
                store.add(
                    key=key,
                    n0=n0,
                    n1=n1,
                    shape=shape,
                    rep=rep,
                    observables=observables,
                    seconds=time.perf_counter() - start,
                    alphas=data,
                )
            print(f"small_n {n0}/{n1}/{shape}", flush=True)
    finally:
        summarize(store=store, expected=expected, pilot=pilot)
