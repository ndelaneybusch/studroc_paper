"""Fixed-interior exponent ladders with a frozen exact corner region."""

import time
from itertools import product

import numpy as np

from studroc_paper.methods.fiducial_band import production_trim_rows

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
from .hybrid import WINDOW, floor_components, geometry, hull, window_status

MIN_ACTIVE = 400
SMALL_REPLICATES = 2000
LARGE_REPLICATES = 1000

EXPONENTS = np.array([1.0, 1.25, 1.5, 2.0, 2.5, 3.5, 5.0, 8.0])


def stitch(
    *, base: tuple, interior: tuple, m3: tuple, grid: np.ndarray, region: np.ndarray
) -> tuple:
    """Switch on a fixed window, take the floor hull, then widen to monotonicity."""
    if not window_status(grid=grid, region=region)["eligible"]:
        return base[0].copy(), base[1].copy()
    window = (grid >= WINDOW[0]) & (grid <= WINDOW[1])
    combined = (
        np.where(window, interior[0], base[0]),
        np.where(window, interior[1], base[1]),
    )
    return hull(raw=combined, m3=m3, region=region)


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
    sizes = [100, 1000] if pilot else [100, 500, 5_000, 50_000]
    shapes = (
        ["normal_0.95", "interior_sliver"]
        if pilot
        else ["normal_0.95", "t2_0.95", "kink", "sliver", "interior_sliver"]
    )
    ratios = [0.5] if pilot else [0.5, 0.1, 0.9]
    return list(product(sizes, ratios, shapes))


def coverage_ladder(*, rows: list[dict], alpha: float) -> dict | None:
    """Estimate one coverage ladder without treating unexposed data as evidence."""
    if not rows:
        return None
    flags = np.array(
        [[row["arms"][f"C{c:g}"]["covered"] for c in EXPONENTS] for row in rows],
        dtype=float,
    )
    ci = np.array(
        [interval(successes=int(s), count=len(rows)) for s in flags.sum(axis=0)]
    )
    return {
        "replicates": len(rows),
        "coverage": flags.mean(axis=0).tolist(),
        "coverage_ci": ci.tolist(),
        "crossing": crossing(coverage=flags.mean(axis=0), target=1 - alpha),
        "crossing_ci": [
            crossing(coverage=ci[:, 0], target=1 - alpha),
            crossing(coverage=ci[:, 1], target=1 - alpha),
        ],
    }


def spread_summary(*, groups: list[list[dict]], alpha: float) -> dict:
    """Bootstrap shape spread only when every shape has enough eligible samples."""
    if len(groups) != 5 or any(len(group) < MIN_ACTIVE for group in groups):
        return {
            "status": "insufficient_eligible_data",
            "collapsed": False,
            "eligible_counts": [len(group) for group in groups],
        }
    rng = rng_for("interior", "conditional-spread", alpha)
    bounds = []
    censored = False
    samples = [
        np.array(
            [[r["arms"][f"C{c:g}"]["covered"] for c in EXPONENTS] for r in group],
            dtype=float,
        )
        for group in groups
    ]
    for _ in range(500):
        intervals = []
        for flags in samples:
            means = flags[rng.integers(len(flags), size=len(flags))].mean(axis=0)
            value = crossing(coverage=means, target=1 - alpha)
            censored |= value["censor"] is not None
            intervals.append((value["lower"] or 1.0, value["upper"] or 8.0))
        bounds.append(max(v[1] for v in intervals) - min(v[0] for v in intervals))
    ci = np.quantile(bounds, [0.025, 0.975]).tolist()
    return {
        "status": "censored" if censored else "resolved",
        "spread_upper_bound_bootstrap_ci": ci,
        "collapsed": not censored and ci[1] <= 0.5,
        "eligible_counts": [len(group) for group in groups],
    }


def summarize(*, store: Store, pilot: bool, expected: int) -> None:
    """Separate deployed coverage from eligible-only calibration and selection."""
    groups = {}
    for row in store.records:
        if not row.get("audit"):
            for alpha in [0.05, 0.5]:
                groups.setdefault(
                    (row["n"], row["ratio"], row["shape"], alpha), []
                ).append(row["alphas"][str(alpha)])
    cells, spread_groups = [], {}
    for key, rows in groups.items():
        active = [r for r in rows if r["window"]["eligible"]]
        inactive = [r for r in rows if not r["window"]["eligible"]]
        active_summary = coverage_ladder(rows=active, alpha=key[-1])
        entry = {
            "cell": key,
            "replicates": len(rows),
            "eligible_replicates": len(active),
            "eligibility_rate": len(active) / len(rows),
            "eligibility_ci": interval(successes=len(active), count=len(rows)),
            "operational": coverage_ladder(rows=rows, alpha=key[-1]),
            "eligible_only": active_summary,
            "fallback_only": coverage_ladder(rows=inactive, alpha=key[-1]),
            "calibration_status": "no_exposure"
            if not active
            else "insufficient_eligible_data"
            if len(active) < MIN_ACTIVE
            else "estimable",
            "paired_to_floor": {
                name: {
                    metric: paired_summary(
                        values=[
                            r["arms"][name][metric] / r["arms"]["floor"][metric]
                            for r in rows
                        ]
                    )
                    for metric in ["area", "left_width", "right_width"]
                }
                for name in rows[0]["arms"]
            },
            "eligible_area_ratios": {
                name: paired_summary(
                    values=[
                        r["arms"][name]["area"] / r["arms"]["floor"]["area"]
                        for r in active
                    ]
                )
                for name in rows[0]["arms"]
            }
            if active
            else None,
        }
        cells.append(entry)
        spread_groups.setdefault((key[0], key[1], key[-1]), []).append(active)
    spreads = [
        {"cell": key, **spread_summary(groups=value, alpha=key[-1])}
        for key, value in spread_groups.items()
    ]
    complete = sum(not row.get("audit", False) for row in store.records) == expected
    frozen = {
        "eligible": False,
        "kind": "gated",
        "window": list(WINDOW),
        "requires_window_eligibility": True,
        "coefficients": {},
        "reasons": {},
    }
    for alpha in [0.05, 0.5]:
        applicable = [c for c in cells if c["cell"][-1] == alpha]
        active = [c for c in applicable if c["eligible_replicates"]]
        adequate = active and all(
            c["eligible_replicates"] >= MIN_ACTIVE for c in active
        )
        resolution = any(
            r["depth"] < 3 or min(r["interior_depths"]) < 3
            for k, rows in groups.items()
            if k[-1] == alpha
            for r in rows
            if r["window"]["eligible"]
        )
        operational = all(
            c["operational"]["crossing_ci"][0]["lower"] is not None for c in applicable
        )
        fallback = {"C": 1.0, "n_eff_range": [500, 5000], "outside_C": 1.0}
        frozen["coefficients"][str(alpha)] = fallback
        if not complete or pilot or not adequate or resolution or not operational:
            frozen["reasons"][str(alpha)] = (
                "incomplete_or_insufficient_eligible_coverage_evidence"
            )
            continue
        constraints = []
        for c in active:
            n, ratio, _, _ = c["cell"]
            limit = min(
                c["eligible_only"]["crossing_ci"][0]["lower"] or 1.0,
                c["operational"]["crossing_ci"][0]["lower"] or 1.0,
            )
            constraints.append((4 * n * ratio * (1 - ratio), limit))
        informative = [
            s
            for s in spreads
            if s["cell"][-1] == alpha and sum(s["eligible_counts"]) > 0
        ]
        collapsed = informative and all(s["collapsed"] for s in informative)
        if collapsed:
            choices = []
            for b in np.linspace(0.1, 1.0, 91):
                a = min((limit - 1) * (n / 100) ** b for n, limit in constraints)
                choices.append(
                    (np.mean([1 + a * (n / 100) ** (-b) for n, _ in constraints]), a, b)
                )
            _, a, b = max(choices)
            frozen["coefficients"][str(alpha)] = {"a": a, "b": b}
        else:
            fallback["C"] = min(
                (limit for n, limit in constraints if 500 <= n <= 5000), default=1.0
            )
        frozen["reasons"][str(alpha)] = "decaying" if collapsed else "finite_range"
    frozen["eligible"] = any(
        v.get("a", v.get("C", 1) - 1) > 0 for v in frozen["coefficients"].values()
    )
    store.save(name="candidate.json", payload=frozen)
    store.save(
        name="summary.json",
        payload={
            "complete": complete,
            "pilot": pilot,
            "expected_units": expected,
            "completed_units": sum(
                not row.get("audit", False) for row in store.records
            ),
            "minimum_eligible_replicates": MIN_ACTIVE,
            "cells": cells,
            "shape_spreads": spreads,
            "candidate": frozen,
        },
    )


def run(*, store: Store, pilot: bool, threads: int) -> None:
    """Run paired raw, floor, and interior arms, checkpointing each observation."""
    cells = design(pilot=pilot)
    reps = {
        n: (3 if pilot else LARGE_REPLICATES if n >= 5000 else SMALL_REPLICATES)
        for n, _, _ in cells
    }
    expected = sum(reps[n] for n, _, _ in cells)
    try:
        for n, ratio in dict.fromkeys((n, ratio) for n, ratio, _ in cells):
            block_shapes = [
                shape for size, share, shape in cells if (size, share) == (n, ratio)
            ]
            for rep, shape in product(range(reps[n]), block_shapes):
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
    """Trim only eligible datasets, retaining the unchanged hybrid elsewhere."""
    n0 = len(labels) - int(labels.sum())
    grid = np.arange(n0 + 1) / n0
    raw, depths = fiducial_edges(
        labels=labels,
        truth=truth,
        draws=draws,
        seed=seed,
        alphas=[0.05, 0.5],
        trim_rows=production_trim_rows(len(grid)),
        threads=threads,
    )
    components = [
        floor_components(labels=labels, draws=draws, depth=int(depth))
        for depth in depths
    ]
    eligibility = [
        window_status(grid=grid, region=left | right) for left, right in components
    ]
    active_indices = [i for i, status in enumerate(eligibility) if status["eligible"]]
    interior, inner_depths = [], np.array([], dtype=int)
    if active_indices:
        levels = [
            1 - (1 - [0.05, 0.5][i]) ** c for i in active_indices for c in EXPONENTS
        ]
        interior, inner_depths = fiducial_edges(
            labels=labels,
            truth=truth,
            draws=draws,
            seed=seed,
            alphas=levels,
            trim_rows=np.flatnonzero((grid >= WINDOW[0]) & (grid <= WINDOW[1])),
            threads=threads,
        )
    alphas = {}
    for index, alpha in enumerate([0.05, 0.5]):
        _, lo3, hi3 = m3_edges(labels=labels, alpha=alpha)
        left, right = components[index]
        region = left | right
        floor = hull(raw=raw[index], m3=(lo3, hi3), region=region)
        arms = {"raw": raw[index], "floor": floor, "m3": (lo3, hi3)}
        used_depths = None
        if eligibility[index]["eligible"]:
            offset = active_indices.index(index) * len(EXPONENTS)
            used_depths = inner_depths[offset : offset + len(EXPONENTS)].tolist()
            for k, c in enumerate(EXPONENTS):
                arms[f"C{c:g}"] = stitch(
                    base=floor,
                    interior=interior[offset + k],
                    m3=(lo3, hi3),
                    grid=grid,
                    region=region,
                )
        else:
            arms.update({f"C{c:g}": floor for c in EXPONENTS})
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
            "geometry": geometry(labels=labels, left=left, right=right),
            "window": eligibility[index],
            "depth": int(depths[index]),
            "interior_depths": used_depths,
            "arms": arm_metrics,
        }
    return alphas


def schedule_exponent(
    *, n0: int, n1: int, alpha: float, candidate: dict, window_eligible: bool
) -> float:
    """Read a frozen proposal with its declared sample-size range and C=1 clamp."""
    if n0 < 1 or n1 < 1:
        raise ValueError("Both class sizes must be positive")
    if not window_eligible:
        return 1.0
    settings = candidate["coefficients"][str(alpha)]
    n_eff = 2 * n0 * n1 / (n0 + n1)
    if "a" in settings:
        return max(1.0, 1 + settings["a"] * (n_eff / 100) ** (-settings["b"]))
    left, right = settings["n_eff_range"]
    return max(1.0, settings["C"]) if left <= n_eff <= right else 1.0
