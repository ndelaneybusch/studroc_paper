"""Independent cell likelihood DPs and normalized rank predictors."""

from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from math import comb, factorial

import numpy as np

from .common import Curve

Q = Fraction


def paths(*, n0: int, n1: int) -> list[tuple[int, ...]]:
    """Enumerate the rank experiment with both fixed class counts."""
    return [
        tuple(int(i in positions) for i in range(n0 + n1))
        for positions in combinations(range(n0 + n1), n1)
    ]


def exact_probability(
    *, labels: tuple[int, ...], cells: tuple, mode: str = "exact"
) -> Fraction:
    """Integrate cell blocks with rational arithmetic, including atom cells.

    Args:
        labels: Merged path in ascending placement order.
        cells: Pairs of rational negative masses and positive masses or caps.
        mode: Exact uniform-within-cell likelihood, upper bound, or lower bound.

    Returns:
        A rational probability or upper bound (caps need not sum to one).
    """
    if mode not in {"exact", "upper", "lower"}:
        raise ValueError(mode)
    pos = [0, *np.cumsum(labels).tolist()]
    neg = [k - j for k, j in enumerate(pos)]
    state = [Q(1)] + [Q(0)] * len(labels)
    for a, b in cells:
        if a < 0 or b < 0:
            raise ValueError("Negative cell mass")
        new = [Q(0)] * len(state)
        for end in range(len(state)):
            for start in range(end + 1):
                i, j = neg[end] - neg[start], pos[end] - pos[start]
                if mode == "lower" and i and j:
                    continue
                denominator = (
                    factorial(i + j) if mode == "exact" else factorial(i) * factorial(j)
                )
                new[end] += state[start] * a**i * b**j / denominator
        state = new
    return state[-1] * factorial(neg[-1]) * factorial(pos[-1])


def likelihoods(
    *,
    labels: np.ndarray,
    negative: np.ndarray,
    positive: np.ndarray,
    mode: str = "exact",
) -> np.ndarray:
    """Vectorize the cell DP across models for a bounded small-n diagnostic."""
    if mode not in {"exact", "upper", "lower"}:
        raise ValueError(mode)
    positives = np.r_[0, np.cumsum(labels, dtype=np.int64)]
    negatives = np.arange(len(labels) + 1) - positives
    i = negatives[:, None] - negatives[None, :]
    j = positives[:, None] - positives[None, :]
    valid = (i >= 0) & (j >= 0) & np.tri(len(i), dtype=bool)
    if mode == "lower":
        valid &= (i == 0) | (j == 0)
    i, j = np.maximum(i, 0), np.maximum(j, 0)
    facts = np.array([float(factorial(k)) for k in range(len(labels) + 1)])
    denominator = facts[i + j] if mode == "exact" else facts[i] * facts[j]
    state = np.zeros((len(positive), len(i)))
    state[:, 0] = 1
    for h, a in enumerate(negative):
        transitions = (a**i)[None, :, :] * positive[:, h, None, None] ** j
        transitions *= valid / denominator
        state = np.einsum("mes,ms->me", transitions, state, optimize=False)
    return state[:, -1] * facts[negatives[-1]] * facts[positives[-1]]


def partition(*, curves: list[Curve], cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Partition at every model knot, preserving atoms as zero-width cells."""
    knots = sorted(
        set(np.linspace(0, 1, cells + 1)) | {float(x) for c in curves for x in c.x}
    )
    events = []
    for x in knots:
        repeated = any(np.count_nonzero(c.x == x) > 1 for c in curves)
        events.extend([(x, "left"), (x, "right")] if repeated else [(x, "right")])
    xs = np.array([x for x, _ in events])
    values = []
    for candidate in curves:
        row = []
        for x, side in events:
            indices = np.flatnonzero(candidate.x == x)
            row.append(
                candidate.y[indices[0]]
                if side == "left" and len(indices)
                else np.interp(x, candidate.x, candidate.y)
            )
        values.append(row)
    return np.diff(xs), np.diff(values, axis=1)


def sequential_mass(*, labels: np.ndarray, table: np.ndarray) -> float:
    """Multiply prefix-only probabilities, respecting exhausted class counts."""
    i = j = 0
    mass = 1.0
    for label in labels:
        probability = table[i, j]
        mass *= probability if label else 1 - probability
        i += 1 - int(label)
        j += int(label)
    return float(mass)


def train_predictor(*, samples: list[np.ndarray], n0: int, n1: int) -> np.ndarray:
    """Fit a Beta-smoothed prefix table from independent training paths."""
    ones = np.ones((n0 + 1, n1 + 1))
    visits = np.full_like(ones, 2.0)
    for labels in samples:
        i = j = 0
        for label in labels:
            ones[i, j] += label
            visits[i, j] += 1
            i += 1 - int(label)
            j += int(label)
    table = ones / visits
    table[n0, :] = 1
    table[:, n1] = 0
    return table


@lru_cache(maxsize=4096)
def step_law(*, n0: int, n1: int, knots: tuple, values: tuple) -> tuple:
    """Enumerate a step CDF's exact rank law using atoms at rational knots."""
    cells = []
    previous_x = previous_y = Q(0)
    for x, y in zip(knots, values, strict=True):
        if x > previous_x:
            cells.append((x - previous_x, Q(0)))
        if y > previous_y:
            cells.append((Q(0), y - previous_y))
        previous_x, previous_y = x, y
    if previous_x < 1:
        cells.append((1 - previous_x, Q(0)))
    return tuple(
        exact_probability(labels=labels, cells=tuple(cells))
        for labels in paths(n0=n0, n1=n1)
    )


def uniform_mass(*, n0: int, n1: int) -> float:
    """Return the exchangeable-label predictive probability."""
    return 1 / comb(n0 + n1, n1)
