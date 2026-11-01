"""Study S: M3 class-split rules, and the shared M3 calibration cache.

Every arm is the exact M3 band; arms differ only in the split ratio
``rho``, the negative class's share of the log confidence budget. All arms
share the replicate's ranks, so width differences are paired.

Rules are the family ``rho_p = n0**-p / (n0**-p + n1**-p)``:
``p = 0`` is Sidak (``"sidak"``), ``p = 1/2`` the production
``"sample_size"`` rule (``"sqrt"``), and ``p = 1`` gives ``rho = n1 / n``
(``"linear"``). A fixed sweep of ``rho`` locates each cell's oracle split,
which measures the headroom left by the rules. The sweep is diagnostic: a
split chosen per cell from outcomes would be a library fit, while a rule
that reads only the class sizes keeps the theorem.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from studroc_paper.methods.class_split import resolve_split_ratio
from studroc_paper.methods.fiducial_band_rs import _require_fiducial_core
from studroc_paper.methods.ks_band import fixed_width_ks_band
from studroc_paper.methods.m3_band_rs import (
    _GAMMA_CACHE,
    _ell_gamma,
    _m3_band_from_labels_rs,
)

from .cells import ALPHAS_S, ALPHAS_T, Cell

TOL = 1e-12
SWEEP = (0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9)
FPR_BANDS = ((0.0, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.0 + 1e-12))


def split_arms(n0: int, n1: int) -> dict[str, float]:
    """Return every Study S arm's split ratio for one class-size pair."""
    arms = {
        "sidak": 0.5,
        "sqrt": resolve_split_ratio(split_ratio="sample_size", n0=n0, n1=n1),
        "linear": n1 / (n0 + n1),
    }
    arms |= {f"rho{rho:.1f}": rho for rho in SWEEP}
    return arms


def class_levels(
    *, alpha: float, rho: float, n0: int, n1: int
) -> list[tuple[int, float]]:
    """The two ``(n, alpha_class)`` calibrations an M3 band at ``rho`` needs."""
    return [
        (n0, round(1.0 - (1.0 - alpha) ** rho, 12)),
        (n1, round(1.0 - (1.0 - alpha) ** (1.0 - rho), 12)),
    ]


def required_levels(cells: Iterable[Cell]) -> set[tuple[int, float]]:
    """Every class calibration either study will request."""
    needed: set[tuple[int, float]] = set()
    for cell in cells:
        if cell.study == "S":
            rhos = split_arms(cell.n0, cell.n1).values()
            alphas = ALPHAS_S
        else:
            rhos = (
                0.5,
                resolve_split_ratio(split_ratio="sample_size", n0=cell.n0, n1=cell.n1),
            )
            alphas = ALPHAS_T
        for alpha in alphas:
            for rho in rhos:
                needed.update(
                    class_levels(alpha=alpha, rho=rho, n0=cell.n0, n1=cell.n1)
                )
    return needed


def _calibrate(key: tuple[int, float]) -> tuple[int, float, float]:
    n, alpha_class = key
    core = _require_fiducial_core()
    return n, alpha_class, _ell_gamma(core, n, alpha_class)


def precompute_gammas(cells: Iterable[Cell], path: Path, *, workers: int) -> int:
    """Calibrate every needed class level once and store them as JSON.

    Args:
        cells: Cells of either study.
        path: Output JSON path; existing entries are kept and reused.
        workers: Worker processes.

    Returns:
        Number of entries in the stored cache.
    """
    stored = json.loads(path.read_text()) if path.exists() else []
    have = {(int(n), float(a)) for n, a, _ in stored}
    todo = sorted(required_levels(cells) - have, key=lambda k: -k[0])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        stored.extend(pool.map(_calibrate, todo))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stored))
    return len(stored)


def load_gammas(path: Path) -> None:
    """Seed the production M3 calibration cache from a stored JSON file."""
    if not path.exists():
        return
    for n, alpha_class, gamma in json.loads(path.read_text()):
        _GAMMA_CACHE[(int(n), round(float(alpha_class), 12))] = float(gamma)


def score(*, lower: NDArray, upper: NDArray, truth: NDArray, grid: NDArray) -> dict:
    """Coverage, direction, and width by FPR band for one M3-type band."""
    below = truth < lower - TOL
    above = truth > upper + TOL
    width = upper - lower
    out = {
        "covered": not bool((below | above).any()),
        "below": bool(below.any()),
        "above": bool(above.any()),
        "area": float(width.mean()),
    }
    for lo, hi in FPR_BANDS:
        mask = (grid >= lo) & (grid < hi)
        out[f"area_{lo:g}"] = float(width[mask].sum() / len(width))
    return out


def run_replicate(*, labels: NDArray, truth: NDArray, n0: int, n1: int) -> dict:
    """Score every split arm and KS at each Study S alpha on one replicate."""
    grid = np.arange(n0 + 1) / n0
    scores_desc = -np.arange(len(labels), dtype=np.float64)
    levels = {}
    for alpha in ALPHAS_S:
        arm_scores = {}
        for arm, rho in split_arms(n0, n1).items():
            _, lower, upper = _m3_band_from_labels_rs(
                labels, alpha=alpha, split_ratio=rho
            )
            arm_scores[arm] = score(lower=lower, upper=upper, truth=truth, grid=grid)
        _, ks_lo, ks_up = fixed_width_ks_band(
            labels, scores_desc, k=n0 + 1, alpha=alpha
        )
        arm_scores["ks"] = score(
            lower=np.asarray(ks_lo, float),
            upper=np.asarray(ks_up, float),
            truth=truth,
            grid=grid,
        )
        levels[f"{alpha:g}"] = arm_scores
    return {"levels": levels}
