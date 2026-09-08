"""Offline ROC-width optimization of independently certified marginal events."""

import time
from itertools import product

import fiducial_core
import numpy as np
from scipy.stats import kstwo

from studroc_paper.methods.m3_band_rs import _ell_bounds

from .common import Store, curve, metrics, paired_summary, rng_for, sample


def compose(*, labels: np.ndarray, bounds0: tuple, bounds1: tuple) -> tuple:
    """Project arbitrary monotone marginal order-statistic boundaries."""
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    grid = np.arange(n0 + 1) / n0
    lo0, hi0 = bounds0
    lo1, hi1 = bounds1
    counts = np.r_[0, np.cumsum(labels)[labels == 0]]
    iup = np.searchsorted(lo0, grid, side="left") + 1
    ilo = np.minimum(np.searchsorted(hi0, grid, side="left") + 1, n0 + 1)
    upper = np.r_[0.0, hi1, 1.0][counts[np.minimum(iup, n0)] + 1]
    upper[iup > n0] = 1
    lower = np.r_[0.0, lo1][counts[ilo - 1]]
    lower[0] = 0
    lower[-1] = upper[-1] = 1
    return grid, lower, upper


def calibrated(*, n: int, alpha: float, eta: float, theta: float) -> tuple:
    """Calibrate the final blended, tail-modified event by conservative bisection."""
    ell = _ell_bounds(core=fiducial_core, n=n, alpha_class=alpha)
    center = np.arange(1, n + 1) / (n + 1)
    radius = kstwo.ppf(1 - alpha, n)
    kslo = np.maximum(0, np.arange(1, n + 1) / n - radius)
    kshi = np.minimum(1, np.arange(n) / n + radius)
    modifier = (1 + np.log1p(np.log(1 / (4 * center * (1 - center))))) ** theta
    left = ((1 - eta) * (center - ell[0]) + eta * (center - kslo)) * modifier
    right = ((1 - eta) * (ell[1] - center) + eta * (kshi - center)) * modifier
    left, right = np.maximum(left, 1e-12), np.maximum(right, 1e-12)

    def event(*, scale: float) -> tuple:
        """Apply monotone tightening before evaluating the actual pivot event."""
        lower = np.maximum.accumulate(np.clip(center - scale * left, 0, 1))
        upper = np.minimum.accumulate(np.clip(center + scale * right, 0, 1)[::-1])[
            ::-1
        ].copy()
        probability = float(fiducial_core.ell_crossing_probability(lower, upper))
        if not np.isfinite(probability):
            raise ArithmeticError("Nonfinite non-crossing probability")
        return lower, upper, probability

    low, high = 0.0, 1.0
    target = 1 - alpha + 1e-9
    while event(scale=high)[2] < target:
        high *= 2
        if high > 1e6:
            raise ArithmeticError("Unable to bracket the pivot target")
    for _ in range(30):
        midpoint = (low + high) / 2
        if event(scale=midpoint)[2] >= target:
            high = midpoint
        else:
            low = midpoint
    result = event(scale=high)
    if result[2] < 1 - alpha:
        raise ArithmeticError("Boundary failed its final coverage certificate")
    return result


def plans(*, n0: int, n1: int, alpha: float, pilot: bool) -> dict:
    """Construct a fixed family indexed only by counts and the declared level."""
    class_alpha = 1 - np.sqrt(1 - alpha)
    baseline = (
        _ell_bounds(core=fiducial_core, n=n0, alpha_class=class_alpha),
        _ell_bounds(core=fiducial_core, n=n1, alpha_class=class_alpha),
    )
    result = {
        "m3": {
            "bounds": baseline,
            "parameters": [0.0, 0.0, 0.0],
            "coverage": [
                float(fiducial_core.ell_crossing_probability(*b)) for b in baseline
            ],
        }
    }
    family = product(
        [0.0, 1.0] if pilot else [0.0, 0.5, 1.0],
        [0.0] if pilot else [-0.5, 0.0, 0.5],
        [0.0, 0.5, 1.0],
    )
    cache = {}
    for eta, theta, split in family:
        rho = n0 ** (-split) / (n0 ** (-split) + n1 ** (-split))
        bounds = []
        probabilities = []
        for n, share in [(n0, rho), (n1, 1 - rho)]:
            level = 1 - (1 - alpha) ** share
            key = (n, level, eta, theta)
            if key not in cache:
                cache[key] = calibrated(n=n, alpha=level, eta=eta, theta=theta)
            lower, upper, probability = cache[key]
            bounds.append((lower, upper))
            probabilities.append(probability)
        if np.prod(probabilities) < 1 - alpha - 1e-12:
            raise ArithmeticError("Joint class probability below nominal")
        result[f"eta{eta}_theta{theta}_s{split}"] = {
            "bounds": bounds,
            "coverage": probabilities,
            "parameters": [eta, theta, split],
        }
    return result


def run(*, store: Store, pilot: bool, threads: int) -> None:
    """Freeze constrained training optima before drawing held-out evaluations."""
    sizes = (
        [(20, 20)]
        if pilot
        else [(50, 50), (100, 100), (50, 450), (450, 50), (500, 500)]
    )
    alphas = [0.05, 0.5] if pilot else [0.05, 0.2, 0.5]
    training = ["normal_0.55", "normal_0.65", "normal_0.8", "normal_0.95"]
    evaluation = [
        "diagonal",
        "normal_0.6",
        "normal_0.7",
        "normal_0.9",
        "hetero_0.7",
        "t2_0.7",
        "t2_0.95",
        "sliver",
        "interior_sliver",
    ]
    reps = 2 if pilot else 128
    expected = len(sizes) * len(alphas) * len(evaluation) * reps
    try:
        for (n0, n1), alpha in product(sizes, alphas):
            unit = f"{n0}-{n1}-{alpha}"
            frozen_path = store.directory / f"frozen-{unit}.json"
            if frozen_path.exists():
                import json

                frozen = json.loads(frozen_path.read_text())
                selected = tuple(
                    tuple(np.asarray(edge) for edge in band)
                    for band in frozen["selected_bounds"]
                )
                baseline = tuple(
                    tuple(np.asarray(edge) for edge in band)
                    for band in frozen["baseline_bounds"]
                )
            else:
                start = time.perf_counter()
                candidates = plans(n0=n0, n1=n1, alpha=alpha, pilot=pilot)
                calibration_seconds = time.perf_counter() - start
                training_values = {name: [] for name in candidates}
                for shape in training:
                    truth = curve(name=shape, n0=n0, n1=n1)
                    for rep in range(reps):
                        labels = sample(
                            truth=truth,
                            n0=n0,
                            n1=n1,
                            rng=rng_for("m3", "train", unit, shape, rep),
                        )
                        for name, candidate in candidates.items():
                            grid, lower, upper = compose(
                                labels=labels,
                                bounds0=candidate["bounds"][0],
                                bounds1=candidate["bounds"][1],
                            )
                            score = metrics(
                                grid=grid, lower=lower, upper=upper, truth=truth
                            )
                            training_values[name].append(
                                [
                                    score[k]
                                    for k in ["area", "left_width", "right_width"]
                                ]
                            )
                base = np.asarray(training_values["m3"]).reshape(len(training), reps, 3)
                eligible = []
                summaries = {}
                for name, values in training_values.items():
                    ratios = np.asarray(values).reshape(base.shape) / base
                    means = ratios.mean(axis=1)
                    summaries[name] = means.tolist()
                    if np.all(means[:, 1:] <= 1 + 1e-12):
                        eligible.append((float(means[:, 0].mean()), name))
                _, winner = min(eligible)
                selected = candidates[winner]["bounds"]
                baseline = candidates["m3"]["bounds"]
                frozen = {
                    "winner": winner,
                    "parameters": candidates[winner]["parameters"],
                    "selected_bounds": [
                        [edge.tolist() for edge in b] for b in selected
                    ],
                    "baseline_bounds": [
                        [edge.tolist() for edge in b] for b in baseline
                    ],
                    "certificates": {
                        name: p["coverage"] for name, p in candidates.items()
                    },
                    "training_ratios_by_shape": summaries,
                    "calibration_seconds": calibration_seconds,
                    "pilot": pilot,
                }
                store.save(name=f"frozen-{unit}.json", payload=frozen)
            for shape in evaluation:
                truth = curve(name=shape, n0=n0, n1=n1)
                for rep in range(reps):
                    key = f"{unit}/{shape}/{rep}"
                    if key in store.keys:
                        continue
                    labels = sample(
                        truth=truth, n0=n0, n1=n1, rng=rng_for("m3", "evaluation", key)
                    )
                    arms = {}
                    for name, bounds in [("m3", baseline), ("optimized", selected)]:
                        start = time.perf_counter()
                        grid, lower, upper = compose(
                            labels=labels, bounds0=bounds[0], bounds1=bounds[1]
                        )
                        score = metrics(
                            grid=grid, lower=lower, upper=upper, truth=truth
                        )
                        score["cached_seconds"] = time.perf_counter() - start
                        arms[name] = score
                    store.add(
                        key=key,
                        n0=n0,
                        n1=n1,
                        alpha=alpha,
                        shape=shape,
                        rep=rep,
                        winner=frozen["winner"],
                        arms=arms,
                    )
            print(f"m3 {unit}: {frozen['winner']}", flush=True)
    finally:
        groups = {}
        for row in store.records:
            groups.setdefault(
                (row["n0"], row["n1"], row["alpha"], row["shape"]), []
            ).append(row)
        summaries = []
        for cell, rows in groups.items():
            entry = {"cell": cell, "reps": len(rows)}
            for metric in ["area", "left_width", "right_width"]:
                summary = paired_summary(
                    values=[
                        r["arms"]["optimized"][metric] / r["arms"]["m3"][metric]
                        for r in rows
                    ]
                )
                entry[metric] = summary
            summaries.append(entry)
        complete = len(store.records) == expected
        promotion = promotion_gate(
            rows=store.records, cells=summaries, complete=complete, pilot=pilot
        )
        store.save(name="candidate.json", payload=promotion)
        store.save(
            name="summary.json",
            payload={
                "complete": complete,
                "pilot": pilot,
                "expected_units": expected,
                "completed_units": len(store.records),
                "eligible": promotion["eligible"],
                "design_gates": promotion["designs"],
                "cells": summaries,
            },
        )


def promotion_gate(
    *, rows: list[dict], cells: list[dict], complete: bool, pilot: bool
) -> dict:
    """Retain useful count/alpha regimes while vetoing each held-out tail loss."""
    groups = {}
    for row in rows:
        groups.setdefault((row["n0"], row["n1"], row["alpha"]), []).append(row)
    designs = []
    for key, records in groups.items():
        paired_blocks = {}
        for row in records:
            paired_blocks.setdefault(row["rep"], []).append(
                row["arms"]["optimized"]["area"] / row["arms"]["m3"]["area"]
            )
        area = paired_summary(
            values=[float(np.mean(block)) for block in paired_blocks.values()]
        )
        shape_cells = [cell for cell in cells if tuple(cell["cell"][:3]) == key]
        tails_pass = bool(shape_cells) and all(
            cell[tail]["ci"] is not None and cell[tail]["ci"][1] <= 1.01
            for cell in shape_cells
            for tail in ["left_width", "right_width"]
        )
        eligible = (
            complete
            and not pilot
            and tails_pass
            and area["ci"] is not None
            and area["ci"][1] <= 0.97
        )
        designs.append(
            {
                "design": key,
                "mean_area_ratio": area,
                "tail_gate": tails_pass,
                "eligible": bool(eligible),
                "boundary": records[0]["winner"] if eligible else "m3",
            }
        )
    return {
        "eligible": any(row["eligible"] for row in designs),
        "designs": designs,
        "unlisted_counts": "m3",
        "scope": "Frozen before independent simulation",
    }
