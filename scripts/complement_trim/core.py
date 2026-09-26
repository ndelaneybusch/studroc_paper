"""Per-replicate construction and scoring of the complement-trim arms.

For each nominal alpha two trim levels are built, each from its own
production-budget cloud:

* level ``alpha`` (floor M3 at ``alpha``): the raw C = 1 band (reference),
  the production hybrid (arm ``hybrid``), and the complement-trimmed hybrid
  (arm ``comp``), plus full M3 at ``alpha`` (arm ``m3``);
* level ``alpha / 2`` (floor M3 at ``alpha / 2``): the full-grid hybrid at
  that level (reference ``hybrid_half``) and the complement-trimmed hybrid
  (arm ``comp_budget``).

Within a level, the floor region is computed once from the full-grid trim
depth ``j_full`` and the complement tube reuses the same cloud (same seed and
draw count). The complement depth is enforced to be at least ``j_full``; when
the kernel's depth is smaller (impossible in exact arithmetic, since a minimum
over fewer columns cannot fall), the full-grid tube, which is exactly the tube
at depth ``j_full``, is used and the event is recorded.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from studroc_paper.methods.fiducial_band import _auto_n_draws, production_trim_rows
from studroc_paper.methods.fiducial_band_rs import (
    _apply_corner_allowances,
    _require_fiducial_core,
)
from studroc_paper.methods.fiducial_ladder import khat_from_labels
from studroc_paper.methods.hybrid_floor import M3Floor, floor_region, stitch_m3_floor
from studroc_paper.methods.m3_band_rs import _m3_band_from_labels_rs

TOL = 1e-12
MAX_RUNS = 16
ARMS = ("hybrid", "comp", "comp_budget", "m3")
REFERENCES = ("raw", "hybrid_half")


def full_trim_rows(n_grid: int) -> NDArray[np.int64]:
    """Return the production trim rows as an explicit index array."""
    rows = production_trim_rows(n_grid)
    return np.arange(n_grid, dtype=np.int64) if rows is None else rows


def complement_rows(n_grid: int, region: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Return the production trim rows that lie outside the floor region."""
    rows = full_trim_rows(n_grid)
    return rows[~np.asarray(region, bool)[rows]]


def enforce_depth(*, j_full: int, j_raw: int) -> tuple[int, bool]:
    """Return ``max(j_full, j_raw)`` and whether the maximum was binding."""
    return max(j_full, j_raw), j_raw < j_full


def _runs(mask: NDArray[np.bool_]) -> list[list[int]]:
    """Half-open index intervals of a boolean mask (at most ``MAX_RUNS``)."""
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)
    return edges[:MAX_RUNS].tolist()


def score(
    *, lower: NDArray, upper: NDArray, truth: NDArray, region: NDArray[np.bool_]
) -> dict:
    """Score one band on the native grid, split by the floor region."""
    below = truth < lower - TOL
    above = truth > upper + TOL
    miss = below | above
    width = upper - lower
    inside = np.asarray(region, bool)
    return {
        "covered": not bool(miss.any()),
        "below": bool(below.any()),
        "above": bool(above.any()),
        "miss_in": bool((miss & inside).any()),
        "miss_out": bool((miss & ~inside).any()),
        "depth": float(max(0.0, np.max(lower - truth), np.max(truth - upper))),
        "area": float(width.mean()),
        "area_in": float(width[inside].sum() / len(width)),
        "area_out": float(width[~inside].sum() / len(width)),
        "low_runs": _runs(below),
        "high_runs": _runs(above),
    }


def build_level(
    *, labels: NDArray, khat: NDArray, level: float, seed: int, n_threads: int
) -> dict:
    """Build the full-grid and complement-trimmed hybrids at one trim level.

    Args:
        labels: Tie-resolved labels in descending score order.
        khat: Empirical positive counts on the native grid.
        level: Trim level ``alpha_eff`` (C = 1) and M3 floor level.
        seed: Cloud seed shared by both trims at this level.
        n_threads: Rayon threads per kernel call (0 = global pool).

    Returns:
        Edges of the raw, floored, and complement-floored bands, the region,
        M3 edges at ``level``, and trim diagnostics.
    """
    core = _require_fiducial_core()
    n1 = int(khat[-1])
    n_grid = len(khat)
    draws = _auto_n_draws(n_grid, level)
    lab8 = labels.astype(np.uint8)

    rows = full_trim_rows(n_grid)
    full_lo, full_up, j_full = core.fiducial_trimmed_tube(
        lab8, draws, level, seed, n_threads, rows.astype(np.uint64)
    )
    raw = _apply_corner_allowances(
        lower=full_lo, upper=full_up, khat=khat, n1=n1, trim_depth=j_full, n_draws=draws
    )
    floor = M3Floor(rule="exact", alpha=level)
    region = floor_region(khat=khat, n_draws=draws, trim_depth=j_full, floor=floor)
    _, m3_lo, m3_up = _m3_band_from_labels_rs(labels, alpha=level)
    hybrid = stitch_m3_floor(
        lower=raw[0], upper=raw[1], m3_lower=m3_lo, m3_upper=m3_up, region=region
    )

    comp_rows = complement_rows(n_grid, region)
    fallback = len(comp_rows) == 0
    if fallback:
        j_raw, j_comp, enforced = j_full, j_full, False
        comp = hybrid
    else:
        comp_lo, comp_up, j_raw = core.fiducial_trimmed_tube(
            lab8, draws, level, seed, n_threads, comp_rows.astype(np.uint64)
        )
        j_comp, enforced = enforce_depth(j_full=j_full, j_raw=j_raw)
        if enforced:
            comp_lo, comp_up = full_lo, full_up
        comp_raw = _apply_corner_allowances(
            lower=comp_lo,
            upper=comp_up,
            khat=khat,
            n1=n1,
            trim_depth=j_comp,
            n_draws=draws,
        )
        comp = stitch_m3_floor(
            lower=comp_raw[0],
            upper=comp_raw[1],
            m3_lower=m3_lo,
            m3_upper=m3_up,
            region=region,
        )
    return {
        "raw": raw,
        "hybrid": hybrid,
        "comp": comp,
        "m3": (np.asarray(m3_lo, float), np.asarray(m3_up, float)),
        "region": region,
        "diag": {
            "n_draws": draws,
            "j_full": int(j_full),
            "j_raw": int(j_raw),
            "j_comp": int(j_comp),
            "enforced": bool(enforced),
            "fallback": bool(fallback),
            "trim_cols_full": len(rows),
            "trim_cols_comp": len(comp_rows),
            "region_frac": float(np.mean(region)),
        },
    }


def run_replicate(
    *,
    labels: NDArray,
    seed: int,
    truth: NDArray,
    alphas: tuple[float, ...],
    n_threads: int = 0,
) -> dict:
    """Build and score every arm and reference at each nominal alpha.

    The level-``alpha`` and level-``alpha/2`` clouds use seeds derived from
    ``seed`` by alpha index and level, so arms at one level share a cloud.

    Returns:
        ``{"K": ..., "levels": {alpha: {"diag": ..., "diag_half": ...,
        "scores": {arm: score}}}}``.
    """
    khat = khat_from_labels(lab_s=labels)
    run_length = len(khat) - 1 - int(np.flatnonzero(khat == khat[-1])[0])
    out: dict = {"K": run_length, "levels": {}}
    seeds = np.random.SeedSequence(entropy=seed).generate_state(
        2 * len(alphas), np.uint64
    )
    for index, alpha in enumerate(alphas):
        full = build_level(
            labels=labels,
            khat=khat,
            level=alpha,
            seed=int(seeds[2 * index]),
            n_threads=n_threads,
        )
        half = build_level(
            labels=labels,
            khat=khat,
            level=alpha / 2,
            seed=int(seeds[2 * index + 1]),
            n_threads=n_threads,
        )
        bands = {
            "raw": (full["raw"], full["region"]),
            "hybrid": (full["hybrid"], full["region"]),
            "comp": (full["comp"], full["region"]),
            "m3": (full["m3"], full["region"]),
            "hybrid_half": (half["hybrid"], half["region"]),
            "comp_budget": (half["comp"], half["region"]),
        }
        out["levels"][f"{alpha:g}"] = {
            "diag": full["diag"],
            "diag_half": half["diag"],
            "scores": {
                name: score(lower=edges[0], upper=edges[1], truth=truth, region=region)
                for name, (edges, region) in bands.items()
            },
        }
    return out
