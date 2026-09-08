"""Fixed-interior exponent ladders with a frozen exact corner region."""

import time
from itertools import product

import numpy as np
from scipy.stats import beta, binom

from studroc_paper.methods.fiducial_ladder import khat_from_labels

from .common import (
    Store,
    curve,
    fiducial_edges,
    interval,
    m3_edges,
    metrics,
    paired_summary,
    rng_for,
    runs,
    sample,
)

EXPONENTS = np.array([1.0, 1.25, 1.5, 2.0, 2.5, 3.5, 5.0, 8.0])


def floor_mask(*, labels: np.ndarray, draws: int, depth: int) -> np.ndarray:
    """Compute the exact inclusive left cutoff and beta-inverted right margin."""
    counts = khat_from_labels(lab_s=labels)
    n0 = len(counts) - 1
    left = n0
    for k in range(n0 + 1):
        probability = 0.0 if k == n0 else np.exp(n0 * np.log1p(-k / n0))
        if binom.sf(depth - 1, draws, probability) <= 0.001:
            left = k
            break
    run = n0 - int(np.flatnonzero(counts == counts[-1])[0])
    right = (
        0
        if run == n0
        else max(0, n0 - int(np.ceil(n0 * beta.ppf(0.975, run + 1, n0 - run) - 1e-9)))
    )
    indices = np.arange(n0 + 1)
    return (indices <= left) | (indices >= right)


def stitch(
    *, base: tuple, interior: tuple, m3: tuple, grid: np.ndarray, region: np.ndarray
) -> tuple:
    """Switch on a fixed window, take the floor hull, then widen to monotonicity."""
    window = (grid >= 0.02) & (grid <= 0.95)
    lower = np.where(window, interior[0], base[0])
    upper = np.where(window, interior[1], base[1])
    lower = np.where(region, np.minimum(lower, m3[0]), lower)
    upper = np.where(region, np.maximum(upper, m3[1]), upper)
    return np.minimum.accumulate(lower[::-1])[::-1], np.maximum.accumulate(upper)


def crossing(*, coverage: np.ndarray, target: float) -> dict:
    """Return an interval-valued nested ladder crossing without extrapolation."""
    if np.any(np.diff(coverage) > 1e-12):
        raise ValueError("Non-nested coverage ladder")
    good = np.flatnonzero(coverage >= target)
    if not len(good):
        return {"lower": None, "upper": 1.0, "censor": "left"}
    index = int(good[-1])
    return {
        "lower": float(EXPONENTS[index]),
        "upper": float(EXPONENTS[index + 1]) if index + 1 < len(EXPONENTS) else None,
        "censor": "right" if index + 1 == len(EXPONENTS) else None,
    }


def design(*, pilot: bool) -> list[tuple]:
    """Declare size, direction and shape cells with balanced-equivalent n."""
    sizes = [100] if pilot else [100, 500, 5_000, 50_000]
    shapes = (
        ["normal_0.95", "interior_sliver"]
        if pilot
        else ["normal_0.95", "t2_0.95", "kink", "sliver", "interior_sliver"]
    )
    ratios = [0.5] if pilot else [0.5, 0.1, 0.9]
    return list(product(sizes, ratios, shapes))


def summarize(*, store: Store, pilot: bool, expected: int) -> None:
    """Invert coverage intervals and freeze only fully supported schedules."""
    groups = {}
    width_groups = {}
    for row in store.records:
        if row.get("audit"):
            continue
        for alpha in [0.05, 0.5]:
            key = (row["n"], row["ratio"], row["shape"], alpha)
            arms = row["alphas"][str(alpha)]["arms"]
            width_groups.setdefault(key, []).append(arms)
            groups.setdefault(key, []).append(
                [
                    row["alphas"][str(alpha)]["arms"][f"C{c:g}"]["covered"]
                    for c in EXPONENTS
                ]
            )
    cells = []
    conservative = {}
    grouped_shapes = {}
    for key, flags in groups.items():
        data = np.asarray(flags, dtype=float)
        means = data.mean(axis=0)
        intervals = np.array(
            [interval(successes=int(s), count=len(data)) for s in data.sum(axis=0)]
        )
        target = 1 - key[-1]
        observed = crossing(coverage=means, target=target)
        lower = crossing(coverage=intervals[:, 0], target=target)
        upper = crossing(coverage=intervals[:, 1], target=target)
        cells.append(
            {
                "cell": key,
                "replicates": len(data),
                "coverage": means.tolist(),
                "coverage_ci": intervals.tolist(),
                "crossing": observed,
                "crossing_ci": [lower, upper],
                "paired_to_floor": {
                    name: {
                        metric: paired_summary(
                            values=[
                                arms[name][metric] / arms["floor"][metric]
                                for arms in width_groups[key]
                            ]
                        )
                        for metric in ["area", "left_width", "right_width"]
                    }
                    for name in width_groups[key][0]
                },
            }
        )
        conservative[key] = lower["lower"]
        grouped_shapes.setdefault((key[0], key[1], key[-1]), []).append(data)
    spreads = []
    all_collapsed = True
    rng = rng_for("interior", "spread-bootstrap")
    for key, shapes in grouped_shapes.items():
        draws = []
        censored = False
        for _ in range(500):
            values = []
            for data in shapes:
                draw = data[rng.integers(len(data), size=len(data))].mean(axis=0)
                estimate = crossing(coverage=draw, target=1 - key[-1])
                if estimate["censor"]:
                    censored = True
                values.append((estimate["lower"] or 1.0, estimate["upper"] or 8.0))
            draws.append(max(v[1] for v in values) - min(v[0] for v in values))
        ci = np.quantile(draws, [0.025, 0.975]).tolist()
        spreads.append(
            {
                "cell": key,
                "spread_upper_bound_bootstrap_ci": ci,
                "censored_bootstraps": censored,
            }
        )
        all_collapsed &= not censored and ci[1] <= 0.5 and len(shapes) == 5
    complete = len([r for r in store.records if not r.get("audit")]) == expected
    resolution_failures = sum(
        data["depth"] < 3 or min(data["interior_depths"]) < 3
        for row in store.records
        for data in row["alphas"].values()
    )
    frozen = {"eligible": False, "kind": "unsupported", "coefficients": {}}
    if complete and not pilot and not resolution_failures:
        for alpha in [0.05, 0.5]:
            available = [
                (2 * (2 * k[0] * k[1]) * (2 * k[0] * (1 - k[1])) / (2 * k[0]), c)
                for k, c in conservative.items()
                if k[-1] == alpha
            ]
            if all_collapsed and all(c is not None for _, c in available):
                choices = []
                for exponent in np.linspace(0.1, 1.0, 91):
                    amplitude = min(
                        (c - 1) * (n / 100) ** exponent for n, c in available
                    )
                    objective = np.mean(
                        [1 + amplitude * (n / 100) ** (-exponent) for n, _ in available]
                    )
                    choices.append((objective, amplitude, exponent))
                _, amplitude, exponent = max(choices)
                frozen["coefficients"][str(alpha)] = {"a": amplitude, "b": exponent}
                frozen["kind"] = "decaying"
            else:
                usable = [c or 1.0 for n, c in available if 500 <= n <= 5000]
                frozen["coefficients"][str(alpha)] = {
                    "C": min(usable, default=1.0),
                    "n_eff_range": [500, 5000],
                    "outside_C": 1.0,
                }
                frozen["kind"] = "finite_range"
        frozen["eligible"] = any(
            v.get("a", v.get("C", 1) - 1) > 0 for v in frozen["coefficients"].values()
        )
    store.save(
        name="summary.json",
        payload={
            "complete": complete,
            "pilot": pilot,
            "expected_units": expected,
            "completed_units": len([r for r in store.records if not r.get("audit")]),
            "cells": cells,
            "shape_spreads": spreads,
            "candidate": frozen,
            "unresolved_cloud_depth_cells": resolution_failures,
        },
    )
    store.save(name="candidate.json", payload=frozen)


def run(*, store: Store, pilot: bool, threads: int) -> None:
    """Run paired raw, floor, and interior arms, checkpointing each observation."""
    cells = design(pilot=pilot)
    reps = {n: (2 if pilot else 24 if n >= 5000 else 64) for n, _, _ in cells}
    expected = sum(reps[n] for n, _, _ in cells)
    try:
        for rep in range(max(reps.values())):
            for n, ratio, shape in cells:
                if rep >= reps[n]:
                    continue
                key = f"{n}/{ratio}/{shape}/{rep}"
                if key in store.keys:
                    continue
                n0, n1 = int(2 * n * ratio), 2 * n - int(2 * n * ratio)
                truth = curve(name=shape, n0=n0, n1=n1)
                rng = rng_for("interior", key)
                labels = sample(truth=truth, n0=n0, n1=n1, rng=rng)
                seed = int(rng.integers(2**63))
                draws = 256 if pilot else 4000 if n >= 5000 else 2000
                started = time.perf_counter()
                alphas = measure(
                    labels=labels, truth=truth, draws=draws, seed=seed, threads=threads
                )
                if rep == 0 and ratio == 0.5 and shape == "normal_0.95":
                    audit = measure(
                        labels=labels,
                        truth=truth,
                        draws=2 * draws,
                        seed=seed,
                        threads=threads,
                    )
                    store.add(
                        key=f"audit/{key}",
                        audit=True,
                        n=n,
                        ratio=ratio,
                        shape=shape,
                        rep=rep,
                        draws=2 * draws,
                        alphas=audit,
                    )
                store.add(
                    key=key,
                    n=n,
                    ratio=ratio,
                    shape=shape,
                    rep=rep,
                    draws=draws,
                    seconds=time.perf_counter() - started,
                    alphas=alphas,
                )
                print(f"interior {key}", flush=True)
    finally:
        summarize(store=store, pilot=pilot, expected=expected)


def measure(*, labels: np.ndarray, truth, draws: int, seed: int, threads: int) -> dict:
    """Measure a complete paired ladder with one frozen floor per alpha."""
    n0 = len(labels) - int(labels.sum())
    grid = np.arange(n0 + 1) / n0
    raw, depths = fiducial_edges(
        labels=labels,
        truth=truth,
        draws=draws,
        seed=seed,
        alphas=[0.05, 0.5],
        trim_rows=None,
        threads=threads,
    )
    levels = [1 - (1 - a) ** c for a in [0.05, 0.5] for c in EXPONENTS]
    interior, inner_depths = fiducial_edges(
        labels=labels,
        truth=truth,
        draws=draws,
        seed=seed,
        alphas=levels,
        trim_rows=np.flatnonzero((grid >= 0.02) & (grid <= 0.95)),
        threads=threads,
    )
    alphas = {}
    for index, alpha in enumerate([0.05, 0.5]):
        _, lo3, hi3 = m3_edges(labels=labels, alpha=alpha)
        region = floor_mask(labels=labels, draws=draws, depth=int(depths[index]))
        floor = stitch(
            base=raw[index],
            interior=raw[index],
            m3=(lo3, hi3),
            grid=grid,
            region=region,
        )
        arms = {"raw": raw[index], "floor": floor, "m3": (lo3, hi3)}
        for k, c in enumerate(EXPONENTS):
            arms[f"C{c:g}"] = stitch(
                base=floor,
                interior=interior[index * len(EXPONENTS) + k],
                m3=(lo3, hi3),
                grid=grid,
                region=region,
            )
        truth_values = truth.evaluate(grid=grid)
        arm_metrics = {}
        for name, (lo, hi) in arms.items():
            score = metrics(grid=grid, lower=lo, upper=hi, truth=truth)
            misses = (truth_values < lo - 1e-12) | (truth_values > hi + 1e-12)
            score["floor_miss"] = bool(np.any(misses & region))
            score["unfloored_miss"] = bool(np.any(misses & ~region))
            arm_metrics[name] = score
        alphas[str(alpha)] = {
            "floor_runs": runs(mask=region),
            "depth": int(depths[index]),
            "interior_depths": inner_depths[index * 8 : (index + 1) * 8].tolist(),
            "arms": arm_metrics,
        }
    return alphas


def schedule_exponent(*, n0: int, n1: int, alpha: float, candidate: dict) -> float:
    """Read a frozen proposal with its declared sample-size range and C=1 clamp."""
    if n0 < 1 or n1 < 1:
        raise ValueError("Both class sizes must be positive")
    settings = candidate["coefficients"][str(alpha)]
    n_eff = 2 * n0 * n1 / (n0 + n1)
    if "a" in settings:
        return max(1.0, 1 + settings["a"] * (n_eff / 100) ** (-settings["b"]))
    left, right = settings["n_eff_range"]
    return max(1.0, settings["C"]) if left <= n_eff <= right else 1.0
