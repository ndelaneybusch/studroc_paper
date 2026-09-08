"""Production-compatible floor geometry and conservative hybrid composition."""

from typing import Literal

import numpy as np
from scipy.stats import beta, binom

from studroc_paper.methods.fiducial_ladder import khat_from_labels

WINDOW = (0.02, 0.95)


def floor_components(
    *,
    labels: np.ndarray,
    draws: int,
    depth: int,
    rule: Literal["exact", "stage_f"] = "exact",
) -> tuple:
    """Return inclusive production left and right masks on the native grid.

    Args:
        labels: Merged labels in descending score order.
        draws: Realized cloud budget, including production's automatic budget.
        depth: Realized C=1 trim depth.
        rule: Declared-budget exact rule or frozen Stage F comparison.

    Returns:
        Left and right masks, with any overlap preserved.
    """
    counts = khat_from_labels(lab_s=labels)
    n0 = len(counts) - 1
    run = n0 - int(np.flatnonzero(counts == counts[-1])[0])
    if rule == "exact":
        left = n0
        for k in range(n0 + 1):
            probability = 0.0 if k == n0 else np.exp(n0 * np.log1p(-k / n0))
            if binom.sf(depth - 1, draws, probability) <= 0.001:
                left = k
                break
        right = (
            0
            if run == n0
            else max(
                0, n0 - int(np.ceil(n0 * beta.ppf(0.975, run + 1, n0 - run) - 1e-9))
            )
        )
    elif rule == "stage_f":
        left = min(n0, int(np.ceil(np.log(draws + 1))))
        right = max(0, n0 - run - int(np.ceil(2 * np.sqrt(max(run, 1)))))
    else:
        raise ValueError(f"Unknown floor rule: {rule}")
    index = np.arange(n0 + 1)
    return index <= left, index >= right


def floor_mask(*, labels: np.ndarray, draws: int, depth: int) -> np.ndarray:
    """Return the union of the exact production floor components."""
    left, right = floor_components(labels=labels, draws=draws, depth=depth)
    return left | right


def hull(*, raw: tuple, m3: tuple, region: np.ndarray) -> tuple:
    """Take the regional M3 hull and restore monotonicity by widening only."""
    lower = np.where(region, np.minimum(raw[0], m3[0]), raw[0])
    upper = np.where(region, np.maximum(raw[1], m3[1]), raw[1])
    return np.minimum.accumulate(lower[::-1])[::-1], np.maximum.accumulate(upper)


def window_status(*, grid: np.ndarray, region: np.ndarray) -> dict:
    """Require the whole fixed window and its interpolation guards to be free.

    Args:
        grid: Full native reporting grid, spanning zero to one.
        region: Union of both inclusive floor masks.

    Returns:
        Eligibility, reason, and bracketing/trim indices. Requiring the guard
        columns also protects non-grid-aligned endpoints at small sample sizes.
    """
    first = int(np.searchsorted(grid, WINDOW[0], side="right") - 1)
    last = int(np.searchsorted(grid, WINDOW[1], side="left"))
    rows = np.flatnonzero((grid >= WINDOW[0]) & (grid <= WINDOW[1]))
    overlap = bool(np.any(region[first : last + 1]))
    eligible = len(rows) >= 2 and not overlap
    return {
        "eligible": eligible,
        "reason": "eligible"
        if eligible
        else "tail_overlap"
        if overlap
        else "too_few_columns",
        "guard_indices": [first, last],
        "trim_columns": len(rows),
    }


def geometry(*, labels: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict:
    """Describe both tail components and the grid cells genuinely left interior."""
    n0 = len(left) - 1
    n1 = int(labels.sum())
    k_left = int(np.flatnonzero(left)[-1])
    k_right = int(np.flatnonzero(right)[0])
    free = ~(left | right)
    both_free = free[:-1] & free[1:]
    return {
        "n0": n0,
        "n1": n1,
        "left_end_index": k_left,
        "right_start_index": k_right,
        "left_end_fpr": k_left / n0,
        "right_start_fpr": k_right / n0,
        "trailing_negatives": n0
        - int(np.flatnonzero(khat_from_labels(lab_s=labels) == n1)[0]),
        "overlap_points": int(np.count_nonzero(left & right)),
        "unfloored_points": int(free.sum()),
        "unfloored_cell_fraction": float(both_free.mean()),
        "fully_floored": bool(np.all(left | right)),
        "no_unfloored_cells": not bool(np.any(both_free)),
        "window": window_status(grid=np.arange(n0 + 1) / n0, region=~free),
    }
