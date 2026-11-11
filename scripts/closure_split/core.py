"""Per-replicate construction and scoring for Study T (floor closure).

At each nominal alpha, one production-budget cloud (fixed seed and draw
count) is evaluated by two kernel calls, so every arm at that alpha sees
the identical cloud:

1. the full production trim rows at level alpha (depth ``j``);
2. the production trim rows outside the floor region at level alpha
   (depth ``j_rec >= j``, theory doc Proposition 10b).

Arms (spec section 4):

=============  ============  =======  ===============  =========================
arm            trim depth    closure  floor M3 split   role
=============  ============  =======  ===============  =========================
``raw``        ``j``         --       --               C = 1 parent
``hyb_W``      ``j``         widen    Sidak            production hybrid
``hyb_T``      ``j``         tighten  Sidak            tightened, contains raw
``rec_W``      ``j_rec``     widen    Sidak            current preferred variant
``rec_T``      ``j_rec``     tighten  Sidak            tightened recalibrated
``rec_T_ss``   ``j_rec``     tighten  sample size      tightening plus split
``m3``         --            --       Sidak            exact reference
``m3_ss``      --            --       sample size      M3 with the split
``ks``         --            --       --               fixed-width comparator
=============  ============  =======  ===============  =========================

Every fiducial arm is composed from the production functions that
``fiducial_band_rs(m3_floor=M3Floor(...))`` calls, and a test checks
bit-for-bit parity with that entry point.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from studroc_paper.methods.fiducial_band import (
    DEFAULT_N_DRAWS,
    _apply_corner_allowances,
    production_trim_rows,
)
from studroc_paper.methods.fiducial_band_rs import _require_fiducial_core
from studroc_paper.methods.fiducial_ladder import khat_from_labels
from studroc_paper.methods.hybrid_floor import (
    M3Floor,
    complement_trim_rows,
    floor_region,
    stitch_m3_floor,
)
from studroc_paper.methods.ks_band import fixed_width_ks_band
from studroc_paper.methods.m3_band_rs import _m3_band_from_labels_rs

TOL = 1e-12
MAX_RUNS = 8
FIDUCIAL_ARMS = ("raw", "hyb_W", "hyb_T", "rec_W", "rec_T", "rec_T_ss")
EXACT_ARMS = ("m3", "m3_ss", "ks")
ARMS = FIDUCIAL_ARMS + EXACT_ARMS
FLOOR = M3Floor(rule="exact")

# (inner, outer) pairs that are nested on one cloud by construction.
NESTINGS = (
    ("rec_T", "rec_W"),
    ("hyb_T", "hyb_W"),
    ("rec_T", "hyb_T"),
    ("rec_W", "hyb_W"),
    ("raw", "hyb_T"),
    ("raw", "hyb_W"),
)


def near_width(n0: int) -> int:
    """Grid points on each side of a region boundary counted as adjacent."""
    return max(10, math.ceil(0.02 * n0))


def region_bounds(region: NDArray[np.bool_]) -> tuple[int, int]:
    """Return the inclusive left cut and the first right-region index.

    Args:
        region: Floor mask ``(idx <= k_left) | (idx >= right_start)``.

    Returns:
        ``(k_left, right_start)``. A region covering the whole grid returns
        ``(n0, n0 + 1)``.
    """
    free = np.flatnonzero(~region)
    if len(free) == 0:
        return len(region) - 1, len(region)
    return int(free[0]) - 1, int(free[-1]) + 1


def location_masks(region: NDArray[np.bool_]) -> dict[str, NDArray[np.bool_]]:
    """Partition the grid into floor ends, the adjacent strips, and the rest."""
    n_grid = len(region)
    k_left, right_start = region_bounds(region)
    index = np.arange(n_grid)
    width = near_width(n_grid - 1)
    left = index <= k_left
    right = index >= right_start
    near = ~region & (
        ((index > k_left) & (index <= k_left + width))
        | ((index < right_start) & (index >= right_start - width))
    )
    return {"left": left, "right": right, "near": near, "far": ~(region | near)}


def _runs(mask: NDArray[np.bool_]) -> list[list[int]]:
    """Half-open index intervals of a boolean mask (at most ``MAX_RUNS``)."""
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)
    return edges[:MAX_RUNS].tolist()


def score(
    *, lower: NDArray, upper: NDArray, truth: NDArray, masks: dict[str, NDArray]
) -> dict:
    """Score one native-grid band against the truth.

    Args:
        lower: Lower edge.
        upper: Upper edge.
        truth: True ROC on the same grid.
        masks: Location masks from :func:`location_masks` (the arm-independent
            region at the full-grid depth).

    Returns:
        Coverage, direction, location, depth, and width summaries.
    """
    below = truth < lower - TOL
    above = truth > upper + TOL
    miss = below | above
    width = upper - lower
    n_grid = len(width)
    inside = masks["left"] | masks["right"]
    return {
        "covered": not bool(miss.any()),
        "below": bool(below.any()),
        "above": bool(above.any()),
        "miss_in": bool((miss & inside).any()),
        "miss_near": bool((miss & masks["near"]).any()),
        "miss_far": bool((miss & masks["far"]).any()),
        "depth": float(max(0.0, np.max(lower - truth), np.max(truth - upper))),
        "area": float(width.mean()),
        **{
            f"area_{key}": float(width[mask].sum() / n_grid)
            for key, mask in masks.items()
        },
        "low_runs": _runs(below),
        "high_runs": _runs(above),
    }


def _tube(
    core,
    labels: NDArray,
    n_draws: int,
    level: float,
    seed: int,
    n_threads: int,
    rows: NDArray | None,
) -> tuple[NDArray, NDArray, int]:
    """One kernel call on fixed (labels, draw count, seed)."""
    lower, upper, depth = core.fiducial_trimmed_tube(
        labels,
        n_draws,
        level,
        seed,
        n_threads,
        None if rows is None else rows.astype(np.uint64),
    )
    return np.asarray(lower, np.float64), np.asarray(upper, np.float64), int(depth)


def _unclosed_hull(lower, upper, m3_lower, m3_upper, region):
    """The region hull before any closure (for the exact-equivalence check)."""
    return (
        np.where(region, np.minimum(lower, m3_lower), lower),
        np.where(region, np.maximum(upper, m3_upper), upper),
    )


def build_alpha(
    *, labels: NDArray, khat: NDArray, alpha: float, seed: int, n_threads: int
) -> dict:
    """Build every Study T arm at one nominal alpha on one cloud.

    Args:
        labels: Tie-resolved labels in descending score order.
        khat: Empirical positive counts on the native grid.
        alpha: Nominal level.
        seed: Cloud seed shared by the three kernel calls.
        n_threads: Rayon threads per kernel call (0 = global pool).

    Returns:
        ``{"bands": {arm: (lower, upper)}, "region": mask, "diag": {...}}``.

    Raises:
        RuntimeError: If a restricted or deeper trim returns a smaller depth,
            which Proposition 10b rules out on a common cloud.
    """
    core = _require_fiducial_core()
    lab8 = np.ascontiguousarray(labels, dtype=np.uint8)
    n_grid = len(khat)
    n1 = int(khat[-1])
    n_draws = DEFAULT_N_DRAWS
    rows = production_trim_rows(n_grid)

    full_lo, full_up, j = _tube(core, lab8, n_draws, alpha, seed, n_threads, rows)
    raw = _apply_corner_allowances(
        lower=full_lo, upper=full_up, khat=khat, n1=n1, trim_depth=j, n_draws=n_draws
    )
    region = floor_region(khat=khat, n_draws=n_draws, trim_depth=j, floor=FLOOR)

    comp_rows = complement_trim_rows(n_grid=n_grid, region=region, trim_rows=rows)
    if len(comp_rows):
        rec_lo, rec_up, j_rec = _tube(
            core, lab8, n_draws, alpha, seed, n_threads, comp_rows
        )
        if j_rec < j:
            raise RuntimeError(f"recalibrated depth {j_rec} < full depth {j}")
        rec = _apply_corner_allowances(
            lower=rec_lo,
            upper=rec_up,
            khat=khat,
            n1=n1,
            trim_depth=j_rec,
            n_draws=n_draws,
        )
    else:
        j_rec, rec = j, raw

    _, m3_lo, m3_up = _m3_band_from_labels_rs(labels, alpha=alpha, split_ratio=0.5)
    _, ss_lo, ss_up = _m3_band_from_labels_rs(
        labels, alpha=alpha, split_ratio="sample_size"
    )

    def floored(parent, m3, closure):
        return stitch_m3_floor(
            lower=parent[0],
            upper=parent[1],
            m3_lower=m3[0],
            m3_upper=m3[1],
            region=region,
            closure=closure,
        )

    m3 = (np.asarray(m3_lo, float), np.asarray(m3_up, float))
    m3_ss = (np.asarray(ss_lo, float), np.asarray(ss_up, float))
    n0 = n_grid - 1
    scores = -np.arange(len(labels), dtype=np.float64)
    _, ks_lo, ks_up = fixed_width_ks_band(labels, scores, k=n_grid, alpha=alpha)
    bands = {
        "raw": raw,
        "hyb_W": floored(raw, m3, "widen"),
        "hyb_T": floored(raw, m3, "tighten"),
        "rec_W": floored(rec, m3, "widen"),
        "rec_T": floored(rec, m3, "tighten"),
        "rec_T_ss": floored(rec, m3_ss, "tighten"),
        "m3": m3,
        "m3_ss": m3_ss,
        "ks": (np.asarray(ks_lo, float), np.asarray(ks_up, float)),
    }
    unclosed = {
        "hyb": _unclosed_hull(*raw, *m3, region),
        "rec": _unclosed_hull(*rec, *m3, region),
    }
    k_left, right_start = region_bounds(region)
    return {
        "bands": bands,
        "unclosed": unclosed,
        "region": region,
        "diag": {
            "n_draws": n_draws,
            "j": j,
            "j_rec": j_rec,
            "k_left": k_left,
            "right_start": right_start,
            "region_frac": float(region.mean()),
            "trim_rows": n_grid if rows is None else len(rows),
            "comp_rows": len(comp_rows),
            "n0": n0,
        },
    }


def exact_checks(built: dict, truth: NDArray) -> dict:
    """Evaluate the construction's exact properties on one replicate.

    Returns:
        The largest violation of each nesting (``<= TOL`` when it holds) and
        whether each tightened arm's coverage equals its unclosed hull's.
    """
    bands = built["bands"]
    out = {}
    for inner, outer in NESTINGS:
        lo_i, up_i = bands[inner]
        lo_o, up_o = bands[outer]
        out[f"nest_{inner}_in_{outer}"] = float(
            max(0.0, np.max(lo_o - lo_i), np.max(up_i - up_o))
        )

    def covers(lower, upper):
        return not bool(np.any((truth < lower - TOL) | (truth > upper + TOL)))

    for name, arm in (("hyb", "hyb_T"), ("rec", "rec_T")):
        out[f"tight_equals_hull_{name}"] = covers(*bands[arm]) == covers(
            *built["unclosed"][name]
        )
    return out


def run_replicate(
    *,
    labels: NDArray,
    seed: int,
    truth: NDArray,
    alphas: tuple[float, ...],
    n_threads: int = 0,
) -> dict:
    """Build, check, and score every Study T arm at each nominal alpha.

    Returns:
        ``{"K": ..., "khat0": ..., "levels": {alpha: {"diag", "checks",
        "scores"}}}``.
    """
    khat = khat_from_labels(lab_s=labels)
    n1 = int(khat[-1])
    run_length = len(khat) - 1 - int(np.flatnonzero(khat == n1)[0])
    out: dict = {"K": run_length, "khat0": int(khat[0]), "levels": {}}
    seeds = np.random.SeedSequence(entropy=seed).generate_state(len(alphas), np.uint64)
    for index, alpha in enumerate(alphas):
        built = build_alpha(
            labels=labels,
            khat=khat,
            alpha=alpha,
            seed=int(seeds[index]),
            n_threads=n_threads,
        )
        masks = location_masks(built["region"])
        out["levels"][f"{alpha:g}"] = {
            "diag": built["diag"],
            "checks": exact_checks(built, truth),
            "scores": {
                arm: score(lower=lo, upper=up, truth=truth, masks=masks)
                for arm, (lo, up) in built["bands"].items()
            },
        }
    return out
