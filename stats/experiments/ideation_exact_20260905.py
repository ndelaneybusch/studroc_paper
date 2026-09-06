"""Paired screening of exact composition geometries for ROC bands.

Run with ``uv run --no-sync python stats/experiments/ideation_exact_20260905.py``.
The JSON on stdout records calibration, paired widths, and native-grid coverage.
This is a small design screen, not a coverage certification study.
"""

import json
import sys
import time

import fiducial_core
import numpy as np
from scipy.optimize import brentq
from scipy.stats import beta, norm, t

from studroc_paper.methods.m3_band_rs import _ell_bounds


def weighted_bounds(*, n: int, alpha: float, power: float) -> tuple:
    """Calibrate deterministic rank weights using the exact crossing kernel.

    Args:
        n: Class sample size.
        alpha: Class error probability.
        power: Exponent of the symmetric central-rank weight.

    Returns:
        Lower and upper pivot bounds and their computed coverage.
    """
    ranks = np.arange(1, n + 1)
    fractions = ranks / (n + 1)
    weights = (4 * fractions * (1 - fractions)) ** power

    def bounds(*, gamma: float) -> tuple:
        """Return monotone bounds at a common weighted local level."""
        lower = beta.ppf(q=gamma * weights, a=ranks, b=n + 1 - ranks)
        upper = beta.ppf(q=1 - gamma * weights, a=ranks, b=n + 1 - ranks)
        lower = np.maximum.accumulate(lower)
        upper = np.ascontiguousarray(np.minimum.accumulate(upper[::-1])[::-1])
        return lower, upper

    low, high = alpha / (2 * n), 0.499
    for _ in range(35):
        midpoint = np.sqrt(low * high)
        lower, upper = bounds(gamma=midpoint)
        coverage = fiducial_core.ell_crossing_probability(lower, upper)
        if coverage >= 1 - alpha + 1e-9:
            low = midpoint
        else:
            high = midpoint
    lower, upper = bounds(gamma=low)
    return lower, upper, fiducial_core.ell_crossing_probability(lower, upper)


def compose(*, labels: np.ndarray, bounds0: tuple, bounds1: tuple) -> tuple:
    """Project class pivot rectangles through the observed label ordering.

    Args:
        labels: Class labels in ascending placement order.
        bounds0: Negative-class lower and upper pivot bounds.
        bounds1: Positive-class lower and upper pivot bounds.

    Returns:
        Native-grid lower and upper ROC edges.
    """
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    grid = np.arange(n0 + 1) / n0
    lo0, hi0 = bounds0
    lo1, hi1 = bounds1
    counts = np.concatenate([[0], np.cumsum(labels)[labels == 0]])
    iup = np.searchsorted(lo0, grid, side="left") + 1
    ilo = np.minimum(np.searchsorted(hi0, grid, side="left") + 1, n0 + 1)
    upper = np.concatenate([[0], hi1, [1]])[counts[np.minimum(iup, n0)] + 1]
    upper[iup > n0] = 1
    lower = np.concatenate([[0], lo1])[counts[ilo - 1]]
    lower[-1] = upper[-1] = 1
    return lower, upper


def joint_levels(*, alpha: float, segments: int = 20) -> tuple:
    """Build an outer staircase for Fisher's two-pivot acceptance region.

    Args:
        alpha: Overall error probability.
        segments: Number of rectangles covering the curved boundary.

    Returns:
        Class-level pairs and the exact probability of their union for
        independent uniform class p-values.
    """
    cutoff = brentq(
        f=lambda value: value * (1 - np.log(value)) - alpha, a=1e-12, b=1 - 1e-12
    )
    edges = np.geomspace(cutoff, 1, segments + 1)
    pairs = list(zip(edges[:-1], cutoff / edges[1:], strict=True))
    coverage = sum(
        (right - left) * (1 - level1)
        for (left, level1), right in zip(pairs, edges[1:], strict=True)
    )
    return pairs, float(coverage)


def run() -> None:
    """Print reproducible paired results for the prespecified screening cells."""
    started = time.perf_counter()
    replicates = 400
    rng = np.random.default_rng(seed=20260905)
    results = {"seed": 20260905, "replicates": replicates, "cells": []}
    for n0, n1 in [(500, 500), (100, 900), (900, 100)]:
        plans = {}
        calibration = {}
        for alpha in [0.05, 0.5]:
            class_alpha = 1 - np.sqrt(1 - alpha)
            base0 = _ell_bounds(core=fiducial_core, n=n0, alpha_class=class_alpha)
            base1 = _ell_bounds(core=fiducial_core, n=n1, alpha_class=class_alpha)
            plans[alpha] = {"m3": [(base0, base1)]}
            weight0 = weighted_bounds(n=n0, alpha=class_alpha, power=0.25)
            weight1 = weighted_bounds(n=n1, alpha=class_alpha, power=0.25)
            plans[alpha]["weighted_025"] = [(weight0[:2], weight1[:2])]
            calibration[str(alpha)] = {
                "weighted_joint_coverage": weight0[2] * weight1[2]
            }
            for rho in [0.25, 0.75]:
                plans[alpha][f"split_{rho}"] = [
                    (
                        _ell_bounds(
                            core=fiducial_core, n=n0, alpha_class=1 - (1 - alpha) ** rho
                        ),
                        _ell_bounds(
                            core=fiducial_core,
                            n=n1,
                            alpha_class=1 - (1 - alpha) ** (1 - rho),
                        ),
                    )
                ]
            levels, joint_coverage = joint_levels(alpha=alpha)
            plans[alpha]["joint_fisher"] = [
                (
                    _ell_bounds(core=fiducial_core, n=n0, alpha_class=level0),
                    _ell_bounds(core=fiducial_core, n=n1, alpha_class=level1),
                )
                for level0, level1 in levels
            ]
            calibration[str(alpha)]["joint_fisher_pivot_coverage"] = joint_coverage
        grid = np.arange(n0 + 1) / n0
        for shape in ["diagonal", "normal_095", "t2_shift10", "sliver"]:
            if shape == "diagonal":
                truth = grid.copy()
            elif shape == "normal_095":
                truth = norm.sf(norm.isf(grid) - np.sqrt(2) * norm.ppf(0.95))
            elif shape == "t2_shift10":
                truth = t.sf(t.isf(grid, df=2) - 10, df=2)
            else:
                knots = np.array([0, 0.1, 1 - 1 / n0, 1])
                values = np.array([0, 1 - 0.8 / n1, 1 - 0.8 / n1, 1])
                truth = np.interp(x=grid, xp=knots, fp=values)
            records = {
                (alpha, name): [] for alpha, arms in plans.items() for name in arms
            }
            for _ in range(replicates):
                negatives = rng.uniform(size=n0)
                if shape == "diagonal":
                    positives = rng.uniform(size=n1)
                elif shape == "normal_095":
                    positives = norm.sf(
                        rng.normal(loc=np.sqrt(2) * norm.ppf(0.95), size=n1)
                    )
                elif shape == "t2_shift10":
                    positives = t.sf(rng.standard_t(df=2, size=n1) + 10, df=2)
                else:
                    positives = np.interp(x=rng.uniform(size=n1), xp=values, fp=knots)
                labels = np.concatenate(
                    [np.zeros(n0, dtype=int), np.ones(n1, dtype=int)]
                )[np.argsort(np.concatenate([negatives, positives]))]
                for alpha, arms in plans.items():
                    for name, rectangles in arms.items():
                        bands = [
                            compose(labels=labels, bounds0=b0, bounds1=b1)
                            for b0, b1 in rectangles
                        ]
                        lower = np.min([band[0] for band in bands], axis=0)
                        upper = np.max([band[1] for band in bands], axis=0)
                        below = bool(np.any(truth < lower - 1e-12))
                        above = bool(np.any(truth > upper + 1e-12))
                        records[alpha, name].append(
                            (
                                not (below or above),
                                below,
                                above,
                                np.trapezoid(y=upper - lower, x=grid),
                            )
                        )
            for alpha, arms in plans.items():
                baseline = np.asarray(records[alpha, "m3"])
                summaries = {}
                for name in arms:
                    samples = np.asarray(records[alpha, name])
                    relative = samples[:, 3] / baseline[:, 3]
                    summaries[name] = {
                        "coverage": float(samples[:, 0].mean()),
                        "below_band": float(samples[:, 1].mean()),
                        "above_band": float(samples[:, 2].mean()),
                        "mean_area": float(samples[:, 3].mean()),
                        "paired_area_ratio": float(relative.mean()),
                        "paired_area_ratio_se": float(
                            relative.std(ddof=1) / np.sqrt(replicates)
                        ),
                    }
                results["cells"].append(
                    {
                        "n0": n0,
                        "n1": n1,
                        "shape": shape,
                        "alpha": alpha,
                        "arms": summaries,
                        "calibration": calibration[str(alpha)],
                    }
                )
        print(
            f"Finished n0={n0}, n1={n1} in {time.perf_counter() - started:.1f}s",
            file=sys.stderr,
            flush=True,
        )
    results["elapsed_seconds"] = time.perf_counter() - started
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run()
