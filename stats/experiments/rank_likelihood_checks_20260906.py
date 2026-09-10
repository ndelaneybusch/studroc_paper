"""Exact-arithmetic checks of bracket likelihood and conservative ROC inversion.

Run from this directory with
``uv run --no-sync python rank_likelihood_checks_20260906.py``.
All probability inequalities and exclusion decisions use rational arithmetic.
Floating point is used only to summarize results in JSON.
"""

import itertools
import json
from fractions import Fraction
from math import comb, factorial

Q = Fraction
LabelPath = tuple[int, ...]
Cell = tuple[Fraction, Fraction]


def paths(*, n0: int, n1: int) -> list[LabelPath]:
    """Enumerate label paths with the specified class counts."""
    return [
        tuple(int(k in positions) for k in range(n0 + n1))
        for positions in itertools.combinations(range(n0 + n1), n1)
    ]


def vertices(*, labels: LabelPath) -> list[tuple[int, int]]:
    """Return all prefix counts, including both terminal vertices."""
    result = [(0, 0)]
    for label in labels:
        i, j = result[-1]
        result.append((i + 1 - label, j + label))
    return result


def cell_probability(*, labels: LabelPath, cells: list[Cell], mode: str) -> Fraction:
    """Integrate a path over ordered cells by a prefix dynamic program.

    Args:
        labels: Labels in increasing placement order.
        cells: Negative and positive cell probabilities, each summing to one.
        mode: ``upper`` ignores within-cell order, ``lower`` allows only pure
            cells, and ``uniform`` uses uniform conditional locations in each
            cell. A cell with zero negative mass can also represent an atom.

    Returns:
        An exact likelihood or its conservative lower or upper bound.
    """
    assert mode in {"upper", "lower", "uniform"}
    assert sum(q for q, _ in cells) == sum(p for _, p in cells) == 1
    counts = vertices(labels=labels)
    n0, n1 = counts[-1]
    state = [Q(1)] + [Q(0)] * len(labels)
    for q, p in cells:
        next_state = [Q(0)] * len(state)
        for end, (i_end, j_end) in enumerate(counts):
            for start in range(end + 1):
                a = i_end - counts[start][0]
                b = j_end - counts[start][1]
                if mode == "lower" and a and b:
                    continue
                denominator = (
                    factorial(a + b)
                    if mode == "uniform"
                    else factorial(a) * factorial(b)
                )
                next_state[end] += state[start] * q**a * p**b / denominator
        state = next_state
    return factorial(n0) * factorial(n1) * state[-1]


def binomial_mass(*, n: int, k: int, p: Fraction) -> Fraction:
    """Evaluate a binomial probability exactly, including the endpoints."""
    return comb(n, k) * p**k * (1 - p) ** (n - k)


def cut_coefficients(*, labels: LabelPath, t: Fraction) -> tuple[Fraction, ...]:
    """Return Bernstein coefficients for the one-cut likelihood upper bound."""
    counts = vertices(labels=labels)
    n0, n1 = counts[-1]
    coefficients = [Q(0)] * (n1 + 1)
    for i, j in counts:
        coefficients[j] += binomial_mass(n=n0, k=i, p=t)
    return tuple(coefficients)


def evaluate(*, coefficients: tuple[Fraction, ...], r: Fraction) -> Fraction:
    """Evaluate a Bernstein polynomial using its probability basis."""
    n = len(coefficients) - 1
    return sum(
        (c * binomial_mass(n=n, k=j, p=r) for j, c in enumerate(coefficients)),
        start=Q(0),
    )


def subdivide(
    *, coefficients: tuple[Fraction, ...]
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    """Subdivide a Bernstein polynomial at its interval midpoint exactly."""
    row = coefficients
    left, right = [row[0]], [row[-1]]
    while len(row) > 1:
        row = tuple((a + b) / 2 for a, b in itertools.pairwise(row))
        left.append(row[0])
        right.append(row[-1])
    return tuple(left), tuple(reversed(right))


def outer_interval(
    *, coefficients: tuple[Fraction, ...], cutoff: Fraction, depth: int
) -> tuple[Fraction, Fraction]:
    """Enclose a polynomial's strict superlevel set by exact subdivision.

    Unresolved intervals are retained. An empty set is reported as [0, 1],
    a conservative fallback sufficient for this verification experiment.
    """
    pending = [(Q(0), Q(1), coefficients, 0)]
    retained = []
    while pending:
        lo, hi, current, level = pending.pop()
        if max(current) <= cutoff:
            continue
        if min(current) > cutoff or level == depth:
            retained.append((lo, hi))
            continue
        left, right = subdivide(coefficients=current)
        mid = (lo + hi) / 2
        pending.extend([(lo, mid, left, level + 1), (mid, hi, right, level + 1)])
    if not retained:
        return Q(0), Q(1)
    return min(lo for lo, _ in retained), max(hi for _, hi in retained)


def placement_cdf(*, name: str, t: Fraction, left_at_one: bool = False) -> Fraction:
    """Evaluate CDFs whose quantiles can also be computed exactly."""
    if name == "diagonal":
        return t
    if name == "jump":
        return Q(t >= Q(1, 2))
    if name == "endpoint_atoms":
        return Q(1, 4) + t / 2 if t < 1 or left_at_one else Q(1)
    raise ValueError(name)


def placement_quantile(*, name: str, v: Fraction) -> Fraction:
    """Invert the verification CDFs, including their placement atoms."""
    if name == "diagonal":
        return v
    if name == "jump":
        return Q(1, 2)
    if name == "endpoint_atoms":
        return max(Q(0), min(Q(1), 2 * (v - Q(1, 4))))
    raise ValueError(name)


def check_bracket_compatibility() -> dict:
    """Compare full-gap containment with independently merged quantile draws."""
    pool = [Q(k, 6) for k in range(1, 6)]
    cases = 0
    for n0, n1 in [(1, 2), (2, 1), (2, 2), (3, 2)]:
        for own_u, own_v, name in itertools.product(
            itertools.combinations(pool, n0),
            itertools.combinations(pool, n1),
            ["diagonal", "jump", "endpoint_atoms"],
        ):
            if name == "jump" and Q(1, 2) in own_u:
                continue
            placed_v = [placement_quantile(name=name, v=v) for v in own_v]
            if any(u == w for u in own_u for w in placed_v):
                continue
            observed = tuple(
                label
                for _, label in sorted(
                    [(u, 0) for u in own_u] + [(w, 1) for w in placed_v]
                )
            )
            u, v = (Q(0), *own_u, Q(1)), (Q(0), *own_v, Q(1))
            for labels in paths(n0=n0, n1=n1):
                counts = [0]
                positives = 0
                for label in labels:
                    positives += label
                    if not label:
                        counts.append(positives)
                counts.append(n1)
                contained = all(
                    v[counts[i]] <= placement_cdf(name=name, t=u[i])
                    and placement_cdf(name=name, t=u[i + 1], left_at_one=True)
                    <= v[counts[i + 1] + 1]
                    for i in range(n0 + 1)
                )
                assert contained == (labels == observed)
                cases += 1
    return {"exact_anchor_path_checks": cases}


def check_likelihood_bounds() -> dict:
    """Stress exact cell bounds with imbalance, zero masses, and atoms."""
    models = {
        "diagonal": [(Q(1, 2), Q(1, 2)), (Q(1, 2), Q(1, 2))],
        "steep_left": [(Q(1, 4), Q(9, 10)), (Q(3, 4), Q(1, 10))],
        "interior_gap": [(Q(1, 3), Q(1, 2)), (Q(1, 3), Q(0)), (Q(1, 3), Q(1, 2))],
        "endpoint_atoms": [(Q(0), Q(1, 5)), (Q(1), Q(1, 2)), (Q(0), Q(3, 10))],
        "interior_atom": [(Q(1, 3), Q(0)), (Q(0), Q(1)), (Q(2, 3), Q(0))],
    }
    checks = 0
    normalization = []
    for n0, n1 in [(1, 1), (1, 4), (4, 1), (2, 3), (3, 3)]:
        all_paths = paths(n0=n0, n1=n1)
        for name, cells in models.items():
            probabilities = []
            for labels in all_paths:
                exact = cell_probability(labels=labels, cells=cells, mode="uniform")
                lower = cell_probability(labels=labels, cells=cells, mode="lower")
                upper = cell_probability(labels=labels, cells=cells, mode="upper")
                assert 0 <= lower <= exact <= upper <= 1
                assert upper - lower <= n0 * n1 * sum(q * p for q, p in cells)
                if name == "diagonal":
                    assert exact == Q(1, comb(n0 + n1, n0))
                if name == "interior_atom":
                    positive_start = labels.index(1)
                    contiguous = (
                        labels[positive_start : positive_start + n1] == (1,) * n1
                    )
                    expected = (
                        binomial_mass(n=n0, k=positive_start, p=Q(1, 3))
                        if contiguous
                        else Q(0)
                    )
                    assert exact == expected
                probabilities.append(exact)
                checks += 1
            assert sum(probabilities) == 1
            normalization.append({"n0": n0, "n1": n1, "model": name})
    return {"path_model_checks": checks, "normalized_laws": len(normalization)}


def check_refinement() -> list[dict]:
    """Verify nested bounds and their error rate under repeated cell splitting."""
    labels = (0, 1, 1, 0, 1)
    cells = [(Q(1, 4), Q(3, 4)), (Q(3, 4), Q(1, 4))]
    exact = cell_probability(labels=labels, cells=cells, mode="uniform")
    previous_lower, previous_upper = Q(0), Q(1)
    result = []
    for _ in range(7):
        lower = cell_probability(labels=labels, cells=cells, mode="lower")
        upper = cell_probability(labels=labels, cells=cells, mode="upper")
        assert previous_lower <= lower <= exact <= upper <= previous_upper
        assert upper - lower <= 6 * max(q for q, _ in cells)
        result.append(
            {
                "cells": len(cells),
                "lower": float(lower),
                "exact": float(exact),
                "upper": float(upper),
                "gap": float(upper - lower),
            }
        )
        previous_lower, previous_upper = lower, upper
        cells = [half for q, p in cells for half in [(q / 2, p / 2)] * 2]
    return result


def check_cut_inversion() -> dict:
    """Enumerate sampling error and certify polynomial projection with rationals."""
    formula_checks = 0
    coverage = []
    grid = [Q(k, 16) for k in range(17)]
    for n0, n1 in [(1, 3), (3, 1), (2, 2), (3, 3)]:
        all_paths = paths(n0=n0, n1=n1)
        predictor = {labels: Q(1, len(all_paths)) for labels in all_paths}
        for labels in all_paths:
            for t in [Q(0), Q(1, 7), Q(1, 2), Q(1)]:
                coefficients = cut_coefficients(labels=labels, t=t)
                assert all(0 <= c <= 1 for c in coefficients)
                for r in [Q(0), Q(1, 11), Q(1, 2), Q(1)]:
                    bound = evaluate(coefficients=coefficients, r=r)
                    dp = cell_probability(
                        labels=labels, cells=[(t, r), (1 - t, 1 - r)], mode="upper"
                    )
                    assert bound == dp
                    left, right = subdivide(coefficients=coefficients)
                    assert evaluate(coefficients=left, r=r) == evaluate(
                        coefficients=coefficients, r=r / 2
                    )
                    assert evaluate(coefficients=right, r=r) == evaluate(
                        coefficients=coefficients, r=(1 + r) / 2
                    )
                    formula_checks += 1
        for alpha in [Q(1, 20), Q(1, 2)]:
            bands = {
                labels: [
                    outer_interval(
                        coefficients=cut_coefficients(labels=labels, t=t),
                        cutoff=alpha * predictor[labels],
                        depth=8,
                    )
                    for t in grid
                ]
                for labels in all_paths
            }
            for name, cells, truth in [
                ("diagonal", [(Q(1), Q(1))], grid),
                (
                    "jump_at_half",
                    [(Q(1, 2), Q(0)), (Q(0), Q(1)), (Q(1, 2), Q(0))],
                    [Q(t >= Q(1, 2)) for t in grid],
                ),
                (
                    "jump_at_fifteen_sixteenths",
                    [(Q(15, 16), Q(0)), (Q(0), Q(1)), (Q(1, 16), Q(0))],
                    [Q(t >= Q(15, 16)) for t in grid],
                ),
                (
                    "endpoint_atoms",
                    [(Q(0), Q(1, 4)), (Q(1), Q(1, 2)), (Q(0), Q(1, 4))],
                    [Q(1, 4) + t / 2 if t < 1 else Q(1) for t in grid],
                ),
            ]:
                grid_error = Q(0)
                likelihood_error = Q(0)
                expected_width = Q(0)
                for labels in all_paths:
                    probability = cell_probability(
                        labels=labels, cells=cells, mode="uniform"
                    )
                    cutoff = alpha * predictor[labels]
                    if probability <= cutoff:
                        likelihood_error += probability
                    bounds = bands[labels]
                    for t, r, (lo, hi) in zip(grid, truth, bounds, strict=True):
                        value = evaluate(
                            coefficients=cut_coefficients(labels=labels, t=t), r=r
                        )
                        assert probability <= value
                        if value > cutoff:
                            assert lo <= r <= hi
                    miss = any(
                        not lo <= r <= hi
                        for r, (lo, hi) in zip(truth, bounds, strict=True)
                    )
                    if miss:
                        assert probability <= cutoff
                        grid_error += probability
                    expected_width += (
                        probability * sum(hi - lo for lo, hi in bounds) / len(grid)
                    )
                assert grid_error <= likelihood_error <= alpha
                if (n0, n1, alpha, name) == (
                    3,
                    3,
                    Q(1, 20),
                    "jump_at_fifteen_sixteenths",
                ):
                    assert grid_error == likelihood_error == Q(1, 4096)
                coverage.append(
                    {
                        "n0": n0,
                        "n1": n1,
                        "alpha": float(alpha),
                        "model": name,
                        "grid_error": float(grid_error),
                        "likelihood_set_error": float(likelihood_error),
                        "mean_grid_width": float(expected_width),
                    }
                )
    return {"formula_checks": formula_checks, "enumerated_coverage": coverage}


def main() -> None:
    """Print independently reproducible verification results as JSON."""
    print(
        json.dumps(
            {
                "arithmetic": (
                    "fractions.Fraction; no simulation or floating-point exclusions"
                ),
                "bracket_compatibility": check_bracket_compatibility(),
                "likelihood_bounds": check_likelihood_bounds(),
                "refinement": check_refinement(),
                "cut_inversion": check_cut_inversion(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
