"""Outer inversion using genuinely monotone rank statistics and full-domain boxes."""

import heapq
import time
from fractions import Fraction
from itertools import combinations_with_replacement

import numpy as np

from .common import Curve, Store, curve, m3_edges, metrics, rng_for, sample
from .rank import exact_probability, paths, step_law

Q = Fraction
KNOTS = (Q(1, 4), Q(1, 2), Q(3, 4))


def statistics(*, labels: np.ndarray, regions: bool = True) -> np.ndarray:
    """Compute nonnegative early/all/late weighted positive-before-negative counts."""
    counts = np.cumsum(labels)[labels == 0]
    cut = max(1, len(counts) // 2)
    if not regions:
        return np.array([counts.sum()], dtype=int)
    return np.array([counts[:cut].sum(), counts.sum(), counts[cut:].sum()], dtype=int)


def endpoint_steps(*, lower: tuple, upper: tuple) -> tuple:
    """Enclose every monotone completion, allowing both endpoint atoms."""
    return ((KNOTS + (Q(1),), lower + (Q(1),)), ((Q(0),) + KNOTS, upper + (Q(1),)))


def split_box(*, lower: tuple, upper: tuple) -> list[tuple]:
    """Bisect the widest knot interval and propagate monotonicity constraints."""
    index = max(range(len(lower)), key=lambda k: upper[k] - lower[k])
    midpoint = (lower[index] + upper[index]) / 2
    children = []
    for half in [0, 1]:
        lo, hi = list(lower), list(upper)
        if half:
            lo[index] = midpoint
        else:
            hi[index] = midpoint
        for k in range(1, len(lo)):
            lo[k] = max(lo[k], lo[k - 1])
        for k in reversed(range(len(hi) - 1)):
            hi[k] = min(hi[k], hi[k + 1])
        if all(a <= b for a, b in zip(lo, hi, strict=True)):
            children.append((tuple(lo), tuple(hi)))
    return children


def outer(*, reject, boxes: int) -> dict:
    """Retain every unresolved box after a bounded search of the full ordered cube."""
    if boxes < 0:
        raise ValueError("Box budget must be nonnegative")
    pending = [(-3.0, 0, (Q(0),) * 3, (Q(1),) * 3)]
    visited = rejected = serial = 0
    while pending and visited < boxes:
        _, _, lo, hi = heapq.heappop(pending)
        visited += 1
        if reject(lower=lo, upper=hi):
            rejected += 1
            continue
        for child_lo, child_hi in split_box(lower=lo, upper=hi):
            serial += 1
            priority = -float(
                sum(b - a for a, b in zip(child_lo, child_hi, strict=True))
            )
            heapq.heappush(pending, (priority, serial, child_lo, child_hi))
    if pending:
        lo = [min(box[2][k] for box in pending) for k in range(3)]
        hi = [max(box[3][k] for box in pending) for k in range(3)]
    else:
        lo, hi = [Q(0)] * 3, [Q(1)] * 3
    return {
        "grid": np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        "lower": np.array([0.0, *map(float, lo), 1.0]),
        "upper": np.array([float(hi[0]), *map(float, hi), 1.0]),
        "boxes_visited": visited,
        "boxes_rejected": rejected,
        "unresolved_boxes": len(pending),
        "empty_set_fallback": not bool(pending),
        "volume_proxy": float(
            sum(
                np.prod([float(b - a) for a, b in zip(box[2], box[3], strict=True)])
                for box in pending
            )
        ),
    }


def exact_rejector(
    *, observed: np.ndarray, n0: int, n1: int, alpha: Fraction, regions: bool
):
    """Construct a rational box certificate using stochastic endpoint ordering."""
    all_paths = paths(n0=n0, n1=n1)
    scores = np.array(
        [statistics(labels=np.asarray(p), regions=regions) for p in all_paths]
    )
    value = statistics(labels=observed, regions=regions)
    level = alpha / (2 * len(value))

    def reject(*, lower: tuple, upper: tuple) -> bool:
        """Bound each inclusive tail by its least favorable step completion."""
        lower_step, upper_step = endpoint_steps(lower=lower, upper=upper)
        low_law = step_law(n0=n0, n1=n1, knots=lower_step[0], values=lower_step[1])
        high_law = step_law(n0=n0, n1=n1, knots=upper_step[0], values=upper_step[1])
        for k in range(len(value)):
            low_tail = sum(
                (
                    p
                    for p, score in zip(low_law, scores[:, k], strict=True)
                    if score <= value[k]
                ),
                start=Q(0),
            )
            high_tail = sum(
                (
                    p
                    for p, score in zip(high_law, scores[:, k], strict=True)
                    if score >= value[k]
                ),
                start=Q(0),
            )
            if min(low_tail, high_tail) <= level:
                return True
        return False

    return reject


def coupled_scores(
    *,
    uniforms0: np.ndarray,
    uniforms1: np.ndarray,
    knots: tuple,
    values: tuple,
    regions: bool,
) -> np.ndarray:
    """Generate endpoint statistics with a shared bank of quantile uniforms."""
    x, y = np.asarray(knots, dtype=float), np.asarray(values, dtype=float)
    positives = x[np.searchsorted(y, uniforms1, side="left")]
    results = []
    for negative, positive in zip(uniforms0, positives, strict=True):
        labels = np.r_[
            np.zeros(len(negative), dtype=int), np.ones(len(positive), dtype=int)
        ][np.argsort(np.r_[negative, positive], kind="stable")]
        results.append(statistics(labels=labels, regions=regions))
    return np.asarray(results)


def mc_rejector(
    *,
    observed: np.ndarray,
    uniforms0: np.ndarray,
    uniforms1: np.ndarray,
    alpha: Fraction,
    regions: bool,
):
    """Bound the plus-one Monte Carlo p-values pathwise over each knot box."""
    observed_scores = statistics(labels=observed, regions=regions)
    level = alpha / (2 * len(observed_scores))
    cache = {}

    def reject(*, lower: tuple, upper: tuple) -> bool:
        """Use the same random bank at all candidates and subdivision depths."""
        low, high = endpoint_steps(lower=lower, upper=upper)
        for step in [low, high]:
            if step not in cache:
                cache[step] = coupled_scores(
                    uniforms0=uniforms0,
                    uniforms1=uniforms1,
                    knots=step[0],
                    values=step[1],
                    regions=regions,
                )
        lower_counts = (cache[low] <= observed_scores).sum(axis=0)
        upper_counts = (cache[high] >= observed_scores).sum(axis=0)
        minimum = int(min(lower_counts.min(), upper_counts.min()))
        return Q(1 + minimum, len(uniforms0) + 1) <= level

    return reject


def rational_models() -> list[tuple]:
    """Define a complete coarse lattice plus smooth and nonsmooth test truths."""
    models = []
    xs = (Q(0),) + KNOTS + (Q(1),)
    for values in combinations_with_replacement([Q(i, 4) for i in range(5)], 3):
        ys = (Q(0),) + values + (Q(1),)
        models.append((f"lattice_{values}", xs, ys))
    normal = curve(name="normal_0.8", n0=4, n1=4)
    ys = tuple(
        Q(round(v * 4096), 4096)
        for v in normal.evaluate(grid=np.asarray(xs, dtype=float))
    )
    models.extend(
        [
            ("binormal", xs, ys),
            ("jump", (Q(0), Q(1, 2), Q(1, 2), Q(1)), (Q(0), Q(0), Q(1), Q(1))),
            (
                "sliver",
                (Q(0), Q(1, 4), Q(15, 16), Q(1)),
                (Q(0), Q(15, 16), Q(15, 16), Q(1)),
            ),
            ("diagonal", (Q(0), Q(1)), (Q(0), Q(1))),
        ]
    )
    return models


def model_law(*, n0: int, n1: int, xs: tuple, ys: tuple) -> tuple:
    """Calculate an exact PL/atom rank law independently of the box endpoint laws."""
    cells = tuple(
        (b - a, d - c)
        for a, b, c, d in zip(xs[:-1], xs[1:], ys[:-1], ys[1:], strict=True)
    )
    return tuple(
        exact_probability(labels=labels, cells=cells) for labels in paths(n0=n0, n1=n1)
    )


def run(*, store: Store, pilot: bool, threads: int) -> None:
    """Enumerate confidence-set containment before the n=50 CRN probe."""
    sizes = [2] if pilot else [2, 3, 4]
    models = rational_models()
    completed_exact = True
    for n in sizes:
        all_paths = paths(n0=n, n1=n)
        laws = [model_law(n0=n, n1=n, xs=xs, ys=ys) for _, xs, ys in models]
        for law in laws:
            if sum(law) != 1:
                raise ArithmeticError("Unnormalized exact rank law")
        for regions in [False, True]:
            scores = np.array(
                [statistics(labels=np.asarray(p), regions=regions) for p in all_paths]
            )
            for alpha in [Q(1, 20), Q(1, 2)]:
                for index, path in enumerate(all_paths):
                    key = f"exact/{n}/{regions}/{alpha}/{index}"
                    if key in store.keys:
                        continue
                    labels = np.asarray(path)
                    start = time.perf_counter()
                    band = outer(
                        reject=exact_rejector(
                            observed=labels, n0=n, n1=n, alpha=alpha, regions=regions
                        ),
                        boxes=7 if pilot else 63,
                    )
                    seconds = time.perf_counter() - start
                    accepted = []
                    named = {}
                    for (name, xs, ys), law in zip(models, laws, strict=True):
                        level = alpha / (2 * scores.shape[1])
                        rejected = any(
                            min(
                                sum(
                                    (
                                        p
                                        for p, s in zip(law, scores[:, k], strict=True)
                                        if s <= scores[index, k]
                                    ),
                                    start=Q(0),
                                ),
                                sum(
                                    (
                                        p
                                        for p, s in zip(law, scores[:, k], strict=True)
                                        if s >= scores[index, k]
                                    ),
                                    start=Q(0),
                                ),
                            )
                            <= level
                            for k in range(scores.shape[1])
                        )
                        truth = Curve(
                            name=name,
                            x=np.asarray(xs, dtype=float),
                            y=np.asarray(ys, dtype=float),
                        )
                        score = metrics(
                            grid=band["grid"],
                            lower=band["lower"],
                            upper=band["upper"],
                            truth=truth,
                        )
                        if not rejected and not score["covered"]:
                            raise AssertionError(
                                "Outer envelope omitted an exact-test accepted curve"
                            )
                        accepted.append(not rejected)
                        if not name.startswith("lattice"):
                            grid3, lo3, hi3 = m3_edges(
                                labels=labels, alpha=float(alpha)
                            )
                            named[name] = {
                                "probability": float(law[index]),
                                "outer": score,
                                "m3": metrics(
                                    grid=grid3, lower=lo3, upper=hi3, truth=truth
                                ),
                                "exact_test_rejected": bool(rejected),
                            }
                    store.add(
                        key=key,
                        kind="exact",
                        n=n,
                        alpha=float(alpha),
                        regions=regions,
                        accepted_count=sum(accepted),
                        checked_candidates=len(models),
                        seconds=seconds,
                        named=named,
                        **{
                            k: v
                            for k, v in band.items()
                            if k not in {"grid", "lower", "upper"}
                        },
                    )
            print(f"projection exact n={n}, regions={regions}", flush=True)
    for shape in ["diagonal", "normal_0.8", "jump", "interior_sliver"]:
        n = 25
        truth = curve(name=shape, n0=n, n1=n)
        for rep in range(2 if pilot else 32):
            rng = rng_for("projection", "mc", shape, rep)
            labels = sample(truth=truth, n0=n, n1=n, rng=rng)
            u0, u1 = rng.random((199, n)), rng.random((199, n))
            for alpha in [Q(1, 20), Q(1, 2)]:
                for regions in [False, True]:
                    key = f"mc/{shape}/{rep}/{alpha}/{regions}"
                    if key in store.keys:
                        continue
                    start = time.perf_counter()
                    band = outer(
                        reject=mc_rejector(
                            observed=labels,
                            uniforms0=u0,
                            uniforms1=u1,
                            alpha=alpha,
                            regions=regions,
                        ),
                        boxes=7 if pilot else 127,
                    )
                    score = metrics(
                        grid=band["grid"],
                        lower=band["lower"],
                        upper=band["upper"],
                        truth=truth,
                    )
                    grid3, lo3, hi3 = m3_edges(labels=labels, alpha=float(alpha))
                    store.add(
                        key=key,
                        kind="mc",
                        shape=shape,
                        rep=rep,
                        alpha=float(alpha),
                        regions=regions,
                        B=199,
                        outer=score,
                        m3=metrics(grid=grid3, lower=lo3, upper=hi3, truth=truth),
                        seconds=time.perf_counter() - start,
                        **{
                            k: v
                            for k, v in band.items()
                            if k not in {"grid", "lower", "upper"}
                        },
                    )
    summaries = []
    for n, alpha, regions, name in (
        (n, a, r, name)
        for n in sizes
        for a in [0.05, 0.5]
        for r in [False, True]
        for name in ["diagonal", "binormal", "jump", "sliver"]
    ):
        rows = [
            row["named"][name]
            for row in store.records
            if row["kind"] == "exact"
            and row["n"] == n
            and row["alpha"] == alpha
            and row["regions"] == regions
        ]
        mass = sum(row["probability"] for row in rows)
        coverage = sum(row["probability"] * row["outer"]["covered"] for row in rows)
        exact_error = sum(
            row["probability"] * row["exact_test_rejected"] for row in rows
        )
        if (
            abs(mass - 1) > 1e-10
            or coverage < 1 - alpha - 1e-10
            or exact_error > alpha + 1e-10
        ):
            raise AssertionError("Enumerated coverage or probability mass failed")
        summaries.append(
            {
                "n": n,
                "alpha": alpha,
                "regions": regions,
                "shape": name,
                "coverage": coverage,
                "test_error": exact_error,
                "area": sum(row["probability"] * row["outer"]["area"] for row in rows),
                "m3_area": sum(row["probability"] * row["m3"]["area"] for row in rows),
            }
        )
    store.save(
        name="summary.json",
        payload={
            "complete": completed_exact,
            "pilot": pilot,
            "enumerated": summaries,
            "records": len(store.records),
            "certificate": "rational exact tails; pathwise plus-one CRN tails",
            "eligible": False,
            "reason": "Prototype measures outer-width and compute; no promotion gate.",
        },
    )
