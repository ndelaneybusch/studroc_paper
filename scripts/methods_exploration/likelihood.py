"""Predictive, cell-coarsening and coordinate-hull losses in likelihood inversion."""

import time
from fractions import Fraction
from itertools import combinations_with_replacement

import numpy as np

from .common import (
    Curve,
    Store,
    curve,
    fiducial_edges,
    m3_edges,
    metrics,
    paired_summary,
    resample,
    rng_for,
    sample,
)
from .projection import KNOTS, outer
from .rank import (
    exact_probability,
    likelihoods,
    partition,
    sequential_mass,
    train_predictor,
    uniform_mass,
)

Q = Fraction


def smooth_model(*, auc: float) -> Curve:
    """Freeze a rational eight-cell PL approximation to a smooth binormal ROC."""
    source = curve(name=f"normal_{auc}", n0=20, n1=20)
    grid = np.linspace(0, 1, 9)
    values = np.rint(source.evaluate(grid=grid) * 4096) / 4096
    return Curve(name=f"smooth_{auc}", x=grid, y=values)


def library() -> tuple[list[Curve], list[Curve]]:
    """Build the fixed efficiency lattice and prespecified predictive mixture."""
    grid = np.linspace(0, 1, 5)
    candidates = [
        Curve(name=f"lattice_{k}", x=grid, y=np.array([0.0, *values, 1.0]))
        for k, values in enumerate(
            combinations_with_replacement(np.linspace(0, 1, 5), 3)
        )
    ]
    mixture = [smooth_model(auc=a) for a in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]]
    candidates.extend(mixture)
    candidates.extend(
        [curve(name="jump", n0=20, n1=20), curve(name="sliver", n0=20, n1=20)]
    )
    return candidates, mixture


def hull(*, values: np.ndarray, accepted: np.ndarray, grid: np.ndarray) -> dict:
    """Measure an inner library hull and rejected curves hidden by its projection."""
    if not np.any(accepted):
        return {
            "empty": True,
            "area": 0.0,
            "accepted": 0,
            "rejected_inside_hull": 0,
            "projection_slack_fraction": 0.0,
        }
    lower, upper = values[accepted].min(axis=0), values[accepted].max(axis=0)
    inside = np.all((values >= lower) & (values <= upper), axis=1)
    hidden = int(np.count_nonzero(inside & ~accepted))
    return {
        "empty": False,
        "area": float(np.dot(np.diff(grid), upper[1:] - lower[:-1])),
        "accepted": int(accepted.sum()),
        "rejected_inside_hull": hidden,
        "projection_slack_fraction": hidden / max(1, int((~accepted).sum())),
    }


def exact_predictor(
    *, labels: np.ndarray, name: str, mixture: list[Curve], table: np.ndarray
) -> Fraction:
    """Evaluate a normalized rational numerator for certified box exclusions."""
    from math import comb

    n1 = int(labels.sum())
    n0 = len(labels) - n1
    if name == "uniform":
        return Q(1, comb(n0 + n1, n1))
    if name == "sequential":
        result = Q(1)
        i = j = 0
        for label in labels:
            probability = Q(float(table[i, j]))
            result *= probability if label else 1 - probability
            i += 1 - int(label)
            j += int(label)
        return result
    masses = []
    for model in mixture:
        xs, ys = tuple(map(Q, model.x)), tuple(map(Q, model.y))
        cells = tuple(
            (b - a, d - c)
            for a, b, c, d in zip(xs[:-1], xs[1:], ys[:-1], ys[1:], strict=True)
        )
        masses.append(exact_probability(labels=tuple(map(int, labels)), cells=cells))
    return sum(masses, start=Q(0)) / len(masses)


def certified_likelihood_outer(
    *, labels: np.ndarray, cutoff: Fraction, boxes: int
) -> dict:
    """Bound all curve completions using positive-mass caps on the full knot cube."""
    a = tuple(b - a for a, b in zip((Q(0),) + KNOTS, KNOTS + (Q(1),), strict=True))

    def reject(*, lower: tuple, upper: tuple) -> bool:
        """Reject only when the exact rational upper likelihood is below cutoff."""
        caps = tuple(
            b - a for a, b in zip((Q(0),) + lower, upper + (Q(1),), strict=True)
        )
        bound = exact_probability(
            labels=tuple(map(int, labels)),
            cells=tuple(zip(a, caps, strict=True)),
            mode="upper",
        )
        return min(Q(1), bound) <= cutoff

    return outer(reject=reject, boxes=boxes)


def summarize(*, store: Store, expected: int, pilot: bool) -> dict:
    """Gate solver work using completed, nonempty, paired smooth n=20 diagnostics."""
    complete = len([r for r in store.records if r.get("kind") == "screen"]) == expected
    groups = {}
    for row in store.records:
        if row.get("kind") != "screen":
            continue
        for alpha, predictors in row["alphas"].items():
            for name, data in predictors.items():
                groups.setdefault((row["n"], row["shape"], alpha, name), []).append(
                    (row, data)
                )
    summaries = []
    eligible = []
    for key, rows in groups.items():
        ratios = [
            data["exact_hull"]["area"] / row["comparators"][key[2]]["m3"]["area"]
            for row, data in rows
        ]
        upper_ratios = [
            data["cell_hulls"]["32"]["area"] / row["comparators"][key[2]]["m3"]["area"]
            for row, data in rows
        ]
        exact = paired_summary(values=ratios)
        upper = paired_summary(values=upper_ratios)
        nonempty = np.mean([not data["exact_hull"]["empty"] for _, data in rows])
        entry = {
            "cell": key,
            "ratio": exact,
            "cell32_ratio": upper,
            "nonempty_fraction": float(nonempty),
        }
        summaries.append(entry)
    for alpha in ["0.05", "0.5"]:
        for name in ["uniform", "mixture", "sequential"]:
            smooth = [
                r
                for r in summaries
                if r["cell"][0] == 20
                and r["cell"][1].startswith("smooth")
                and r["cell"][2:] == (alpha, name)
            ]
            passes = len(smooth) == 4 and all(
                r["nonempty_fraction"] >= 0.99
                and r["ratio"]["ci"] is not None
                and r["ratio"]["ci"][1] < 0.97
                and r["cell32_ratio"]["ci"][1] < 1
                for r in smooth
            )
            if complete and not pilot and passes:
                eligible.append({"alpha": float(alpha), "predictor": name})
    result = {
        "complete": complete,
        "pilot": pilot,
        "expected_units": expected,
        "completed_units": len([r for r in store.records if r.get("kind") == "screen"]),
        "solver_eligible": eligible,
        "cells": summaries,
        "scope": "Finite-library inner hulls are optimistic, not certified bands.",
    }
    store.save(name="summary.json", payload=result)
    return result


def run(*, store: Store, pilot: bool, threads: int) -> None:
    """Separate predictive and geometric losses on paired small rank experiments."""
    candidates, mixture = library()
    truths = [smooth_model(auc=a) for a in [0.5, 0.6, 0.8, 0.95]]
    truths.extend(
        [curve(name="jump", n0=20, n1=20), curve(name="sliver", n0=20, n1=20)]
    )
    if pilot:
        truths = [truths[0], truths[-1]]
    sizes = [5] if pilot else [5, 10, 20]
    reps = 2 if pilot else 64
    expected = len(sizes) * len(truths) * reps
    grid = np.unique(np.r_[np.linspace(0, 1, 33), *[c.x for c in candidates]])
    base_grid = grid
    tables = {}
    try:
        for n in sizes:
            grid = np.unique(np.r_[base_grid, np.arange(n + 1) / n])
            candidate_values = np.array([c.evaluate(grid=grid) for c in candidates])
            training_rng = rng_for("likelihood", "training", n)
            training = [
                sample(
                    truth=mixture[int(training_rng.integers(len(mixture)))],
                    n0=n,
                    n1=n,
                    rng=training_rng,
                )
                for _ in range(128 if pilot else 4096)
            ]
            table = train_predictor(samples=training, n0=n, n1=n)
            tables[n] = table
            store.save(
                name=f"predictor-n{n}.json",
                payload={
                    "table": table.tolist(),
                    "training_samples": len(training),
                    "mixture_aucs": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
                },
            )
            for truth in truths:
                models = candidates + [truth]
                partitions = {
                    m: partition(curves=models, cells=m) for m in [4, 8, 16, 32]
                }
                mix_a, mix_b = partition(curves=mixture, cells=8)
                for rep in range(reps):
                    key = f"{n}/{truth.name}/{rep}"
                    if key in store.keys:
                        continue
                    rng = rng_for("likelihood", "evaluation", key)
                    labels = sample(truth=truth, n0=n, n1=n, rng=rng)
                    start = time.perf_counter()
                    a, b = partitions[4]
                    exact = likelihoods(labels=labels, negative=a, positive=b)
                    if exact[-1] <= 0:
                        raise ArithmeticError("Sampled path has zero truth likelihood")
                    q = {
                        "uniform": uniform_mass(n0=n, n1=n),
                        "mixture": float(
                            likelihoods(
                                labels=labels, negative=mix_a, positive=mix_b
                            ).mean()
                        ),
                        "sequential": sequential_mass(labels=labels, table=table),
                        "oracle": float(exact[-1]),
                    }
                    bounds = {}
                    for m, (a, b) in partitions.items():
                        low = likelihoods(
                            labels=labels, negative=a, positive=b, mode="lower"
                        )
                        high = likelihoods(
                            labels=labels, negative=a, positive=b, mode="upper"
                        )
                        tolerance = np.maximum(exact * 1e-10, 1e-300)
                        if np.any(low > exact + tolerance) or np.any(
                            exact > high + tolerance
                        ):
                            raise ArithmeticError(
                                "Cell bounds failed independent likelihood comparison"
                            )
                        bounds[m] = (
                            low,
                            high,
                            min(1.0, float(n * n * np.dot(a, b[-1]))),
                        )
                    dp_seconds = time.perf_counter() - start
                    bands, _ = fiducial_edges(
                        labels=labels,
                        truth=truth,
                        draws=256 if pilot else 2000,
                        seed=int(rng.integers(2**63)),
                        alphas=[0.05, 0.5],
                        trim_rows=None,
                        threads=threads,
                    )
                    comparators = {}
                    alphas = {}
                    for k, alpha in enumerate([0.05, 0.5]):
                        grid3, lo3, hi3 = m3_edges(labels=labels, alpha=alpha)
                        native = np.arange(n + 1) / n
                        lo3, hi3 = resample(
                            source=grid3, lower=lo3, upper=hi3, target=grid
                        )
                        loc, hic = resample(
                            source=native,
                            lower=bands[k][0],
                            upper=bands[k][1],
                            target=grid,
                        )
                        comparators[str(alpha)] = {
                            "m3": metrics(grid=grid, lower=lo3, upper=hi3, truth=truth),
                            "c1": metrics(grid=grid, lower=loc, upper=hic, truth=truth),
                        }
                        alphas[str(alpha)] = {}
                        for name, probability in q.items():
                            cutoff = alpha * probability
                            exact_accepted = exact[:-1] > cutoff
                            cell_hulls = {}
                            gaps = {}
                            for m, (low, high, union) in bounds.items():
                                accepted = high[:-1] > cutoff
                                cell_hulls[str(m)] = hull(
                                    values=candidate_values,
                                    accepted=accepted,
                                    grid=grid,
                                )
                                gaps[str(m)] = {
                                    "actual_cells": len(partitions[m][0]),
                                    "actual": float(high[-1] - low[-1]),
                                    "cutoff_ratio": float(
                                        (high[-1] - low[-1]) / cutoff
                                    ),
                                    "union_bound": union,
                                    "extra_candidates": int(
                                        np.count_nonzero(accepted & ~exact_accepted)
                                    ),
                                }
                            alphas[str(alpha)][name] = {
                                "q": probability,
                                "log_p_true_over_q": float(
                                    np.log(exact[-1] / probability)
                                ),
                                "truth_rejected": bool(exact[-1] <= cutoff),
                                "exact_hull": hull(
                                    values=candidate_values,
                                    accepted=exact_accepted,
                                    grid=grid,
                                ),
                                "cell_hulls": cell_hulls,
                                "cell_gaps": gaps,
                            }
                    store.add(
                        key=key,
                        kind="screen",
                        n=n,
                        shape=truth.name,
                        rep=rep,
                        labels=labels.tolist(),
                        alphas=alphas,
                        comparators=comparators,
                        dp_seconds=dp_seconds,
                        exact_likelihoods=exact.tolist(),
                    )
                print(f"likelihood n={n}, shape={truth.name}", flush=True)
    finally:
        result = summarize(store=store, expected=expected, pilot=pilot)
    for gate in result["solver_eligible"]:
        for row in list(store.records):
            if row.get("kind") != "screen" or row["n"] != 20 or row["rep"] >= 8:
                continue
            key = f"solver/{gate['alpha']}/{gate['predictor']}/{row['key']}"
            if key in store.keys:
                continue
            labels = np.asarray(row["labels"])
            q = exact_predictor(
                labels=labels, name=gate["predictor"], mixture=mixture, table=tables[20]
            )
            band = certified_likelihood_outer(
                labels=labels, cutoff=Q(str(gate["alpha"])) * q, boxes=63
            )
            truth = next(t for t in truths if t.name == row["shape"])
            store.add(
                key=key,
                kind="solver",
                gate=gate,
                metrics=metrics(
                    grid=band["grid"],
                    lower=band["lower"],
                    upper=band["upper"],
                    truth=truth,
                ),
                boxes_visited=band["boxes_visited"],
                unresolved_boxes=band["unresolved_boxes"],
            )
