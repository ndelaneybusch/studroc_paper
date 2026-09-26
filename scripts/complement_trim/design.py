"""Cells, truths, and seed streams for the complement-trim experiment.

The design reuses the Stage F Study B (30) and Study C (14) cells with fresh
names and seed streams, and adds six interior-feature cells. See
``stats/experiments/complement_trim_spec.md``.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "c_calibration"))

from shapes import get_curve, quantize_jitter  # noqa: E402
from stage_f_design import (  # noqa: E402
    StageFCell,
    cell_curve,
    register_cell_shape,
    study_b_cells,
    study_c_cells,
)

STUDY_SEED = 20260926
ALPHAS = (0.05, 0.5)
BASE_REPS = 400
MAX_REPS = 1_200
LARGE_N0 = 5_000
LARGE_BASE_REPS = 200
LARGE_MAX_REPS = 600
TOPUP_BATCH = 400


@dataclass(frozen=True)
class PLCurve:
    """Piecewise-linear placement CDF whose inverse never bridges plateaus.

    Attributes:
        x: Ordered FPR knots from 0 to 1 (repeated knots allowed).
        y: Nondecreasing TPR values from 0 to 1 at the knots.
    """

    x: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        """Reject knots that do not describe a CDF from (0, 0) to (1, 1)."""
        x, y = np.asarray(self.x, float), np.asarray(self.y, float)
        if (
            x.shape != y.shape
            or x[0] != 0.0
            or x[-1] != 1.0
            or y[0] != 0.0
            or y[-1] != 1.0
            or np.any(np.diff(x) < 0)
            or np.any(np.diff(y) < 0)
        ):
            raise ValueError("expected an ordered CDF from (0, 0) to (1, 1)")

    def eval(self, grid: np.ndarray) -> np.ndarray:
        """Return the curve at the requested FPR values."""
        return np.interp(grid, self.x, self.y)

    def inv(self, uniforms: np.ndarray) -> np.ndarray:
        """Invert each positive-mass segment separately (exact PL sampling)."""
        index = np.clip(
            np.searchsorted(self.y, uniforms, side="left"), 1, len(self.y) - 1
        )
        left = index - 1
        mass = self.y[index] - self.y[left]
        fraction = np.divide(
            uniforms - self.y[left], mass, out=np.zeros_like(uniforms), where=mass > 0
        )
        return self.x[left] + fraction * (self.x[index] - self.x[left])

    def auc(self) -> float:
        """Trapezoidal area under the curve."""
        return float(np.trapezoid(self.y, self.x))


def interior_sliver_curve(*, n0: int, n1: int) -> tuple[PLCurve, tuple[float, float]]:
    """Return the methods-exploration interior sliver and its feature interval.

    Mass ``min(.1, .8 / n1)`` sits in a width-``1/n0`` step at FPR .6, after a
    plateau, with 20% positive mass beyond FPR .85 so that the saturated-run
    floor does not reach the feature.
    """
    mass, width = min(0.1, 0.8 / n1), min(0.05, 1.0 / n0)
    curve = PLCurve(
        x=np.array([0.0, 0.1, 0.6 - width, 0.6, 0.85, 1.0]),
        y=np.array([0.0, 0.8 - mass, 0.8 - mass, 0.8, 0.8, 1.0]),
    )
    return curve, (0.6 - width, 0.6)


def interior_jump_curve(
    *, n0: int, height: float = 0.2, location: float = 0.3, base_auc: float = 0.8
) -> tuple[PLCurve, tuple[float, float]]:
    """Return a binormal body with a sub-grid interior jump and its interval.

    ``R(t) = (1 - h) B(t) + h * clip((t - t0) / w, 0, 1)`` with ``B`` the
    equal-variance binormal ROC and ``w = 1 / (4 n0)``: a fraction ``h`` of
    the positives sits inside a quarter of one negative gap at FPR ``t0``.
    """
    width = 1.0 / (4.0 * n0)
    grid = np.unique(
        np.r_[
            0.0,
            np.geomspace(1e-7, 0.01, 48),
            np.linspace(0.01, 0.99, 513),
            1.0 - np.geomspace(1e-7, 0.01, 48),
            location,
            location + width,
            1.0,
        ]
    )
    shift = np.sqrt(2.0) * norm.ppf(base_auc)
    body = norm.cdf(shift + norm.ppf(np.clip(grid, 1e-15, 1 - 1e-15)))
    body[0], body[-1] = 0.0, 1.0
    values = (1.0 - height) * body + height * np.clip((grid - location) / width, 0, 1)
    return PLCurve(x=grid, y=values), (location, location + width)


@dataclass(frozen=True)
class Cell:
    """One experiment cell.

    Attributes:
        name: Unique cell name; also keys the seed stream.
        block: ``"B"``, ``"C"``, or ``"interior"``.
        n0: Negative-class size.
        n1: Positive-class size.
        stage_f: Source Stage F cell (``None`` for interior cells).
        interior: Interior-feature specification (``None`` otherwise).
        reps: Initial replicate count.
        reps_max: Replicate cap for the sequential top-up.
    """

    name: str
    block: str
    n0: int
    n1: int
    stage_f: StageFCell | None = None
    interior: dict | None = None
    reps: int = BASE_REPS
    reps_max: int = MAX_REPS

    def truth(self) -> tuple[object, tuple[float, float] | None]:
        """Return the exact simulation truth and any interior feature interval."""
        if self.stage_f is not None:
            return cell_curve(self.stage_f), None
        spec = dict(self.interior or {})
        kind = spec.pop("kind")
        if kind == "interior_sliver":
            return interior_sliver_curve(n0=self.n0, n1=self.n1)
        if kind == "interior_jump":
            return interior_jump_curve(n0=self.n0, **spec)
        raise ValueError(f"unknown interior kind {kind!r}")

    def describe(self) -> dict:
        """Return a JSON-native description for manifests and summaries."""
        out = {"name": self.name, "block": self.block, "n0": self.n0, "n1": self.n1}
        if self.stage_f is not None:
            out |= {
                "stage_f_name": self.stage_f.name,
                "source": self.stage_f.source,
                "shape_meta": self.stage_f.shape_meta,
                "quantize": self.stage_f.quantize,
            }
        else:
            out["interior"] = self.interior
        out["reps"], out["reps_max"] = self.reps, self.reps_max
        return out


def _stable_hash(text: str) -> int:
    """Map text to a stable unsigned 64-bit integer."""
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def _replication(n0: int) -> dict[str, int]:
    """Base and maximum replicate counts; halved-cost policy at large n0."""
    if n0 >= LARGE_N0:
        return {"reps": LARGE_BASE_REPS, "reps_max": LARGE_MAX_REPS}
    return {"reps": BASE_REPS, "reps_max": MAX_REPS}


def all_cells() -> list[Cell]:
    """Return the full design: Stage F B and C cells plus the interior block."""
    cells = []
    for source in study_b_cells() + study_c_cells():
        source = source.with_budget()
        cells.append(
            Cell(
                name=f"ct-{source.name}",
                block=source.study,
                n0=source.n0,
                n1=source.n1,
                stage_f=source,
                **_replication(source.n0),
            )
        )
    interior = [
        ({"kind": "interior_sliver"}, 500, 500),
        ({"kind": "interior_sliver"}, 2_000, 2_000),
        ({"kind": "interior_jump", "height": 0.2}, 500, 500),
        ({"kind": "interior_jump", "height": 0.2}, 2_000, 2_000),
        ({"kind": "interior_jump", "height": 0.2}, 2_000, 500),
        ({"kind": "interior_jump", "height": 0.2}, 500, 2_000),
    ]
    for spec, n0, n1 in interior:
        cells.append(
            Cell(
                name=f"ct-i-{spec['kind']}--n{n0}x{n1}",
                block="interior",
                n0=n0,
                n1=n1,
                interior=spec,
            )
        )
    names = [cell.name for cell in cells]
    if len(set(names)) != len(names):
        raise RuntimeError("duplicate cell names")
    return cells


def select_cells(patterns: list[str] | None) -> list[Cell]:
    """Return cells whose names contain any of the given substrings."""
    cells = all_cells()
    if not patterns:
        return cells
    return [cell for cell in cells if any(p in cell.name for p in patterns)]


def sample_replicate(cell: Cell, rep: int) -> tuple[np.ndarray, int, dict]:
    """Draw one tie-resolved label order, its cloud seed, and diagnostics.

    Returns:
        Labels in descending score order, the 64-bit cloud seed, and
        pre-outcome diagnostics (sliver/feature sample counts).
    """
    rng = np.random.default_rng(
        np.random.SeedSequence(entropy=(STUDY_SEED, _stable_hash(cell.name), rep))
    )
    negative = rng.random(cell.n0)
    diagnostics: dict[str, int | bool] = {}
    if cell.stage_f is not None:
        register_cell_shape(cell.stage_f)
        positive = get_curve(cell.stage_f.shape).inv(rng.random(cell.n1))
        if cell.stage_f.shape_meta["family"] == "sliver":
            count = int(np.count_nonzero(positive >= 1.0 - 1.0 / cell.n0))
            diagnostics = {"feature_count": count, "feature_sampled": count > 0}
        if cell.stage_f.quantize is not None:
            negative, positive = quantize_jitter(
                negative, positive, cell.stage_f.quantize, rng
            )
    else:
        curve, (lo, hi) = cell.truth()
        positive = curve.inv(rng.random(cell.n1))
        count = int(np.count_nonzero((positive >= lo) & (positive <= hi)))
        diagnostics = {"feature_count": count, "feature_sampled": count > 0}
    labels = np.r_[np.zeros(cell.n0, np.uint8), np.ones(cell.n1, np.uint8)]
    order = np.argsort(np.r_[negative, positive], kind="stable")
    seed = int(rng.integers(0, 2**64, dtype=np.uint64))
    return labels[order], seed, diagnostics


def with_reps(cell: Cell, *, scale: float) -> Cell:
    """Scale a cell's replicate counts (for pilots), keeping at least 10."""
    return replace(
        cell,
        reps=max(10, round(cell.reps * scale)),
        reps_max=max(10, round(cell.reps_max * scale)),
    )
