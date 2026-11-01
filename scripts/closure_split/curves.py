"""Truth constructions for the closure/split experiment.

Every truth is a rank-space ROC curve: negatives are Uniform(0, 1) and
positives have CDF equal to the curve (fiducial theory Proposition 2), so a
curve fully specifies the sampling law of every rank-based band. The
constructions below model mechanisms seen in real classifier scores, each
aimed at a specific weakness of the tightened floor closure:

* **Hard positives** (label noise or undetectable cases): a fraction of the
  positives score like negatives, ``R = (1 - pi) B + pi t``. The saturated
  run is broken by those positives, so the floor's right region is short,
  and the body's own convex corner becomes an interior hook just outside it.
* **Hard negatives** (mislabelled or confusable negatives): a fraction of the
  negatives score like positives. The convex rise at the low-FPR corner is
  pushed to FPR of about ``pi``, past the floor's left cut.
* **Obvious-positive subgroup**: a fraction of the positives lies among the
  top ``c`` negatives, ``c`` slightly beyond the left cut, so a near-vertical
  segment straddles the left region boundary.
* **Separated top subgroup**: a fraction of the positives lies above all
  negatives (a bounded negative support), so ``R(0) > 0`` and the upper edge
  must be high from the first grid point.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "c_calibration"))

from shapes import Curve, fine_grid  # noqa: E402


def _body_values(body: Curve, grid: np.ndarray) -> np.ndarray:
    """Body ROC on a grid with the endpoints pinned to 0 and 1."""
    values = body.eval(grid)
    values[grid == 0.0] = 0.0
    values[grid == 1.0] = 1.0
    return values


def hard_positive_curve(*, body: Curve, fraction: float) -> Curve:
    """Mix the body ROC with a fraction of positives scored like negatives.

    Args:
        body: ROC of the separable positives.
        fraction: Share ``pi`` of positives whose placement is Uniform(0, 1).

    Returns:
        ``R(t) = (1 - pi) B(t) + pi t``.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    grid = np.unique(np.r_[fine_grid(), body.t])
    values = (1.0 - fraction) * _body_values(body, grid) + fraction * grid
    return Curve(grid, values)


def hard_negative_curve(*, body: Curve, fraction: float) -> Curve:
    """ROC when a fraction of the negatives score like positives.

    With the body's negative survival ``s`` and positive survival ``B(s)``, a
    negative mixture puts the total negative survival at
    ``t(s) = (1 - pi) s + pi B(s)``, so the curve is ``{(t(s), B(s))}``.

    Args:
        body: ROC against the clean negatives.
        fraction: Share ``pi`` of negatives drawn from the positive law.

    Returns:
        The contaminated ROC.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    s = np.unique(np.r_[fine_grid(), body.t])
    tpr = _body_values(body, s)
    fpr = (1.0 - fraction) * s + fraction * tpr
    grid = fine_grid()
    return Curve(grid, np.interp(grid, fpr, tpr))


def obvious_subgroup_curve(
    *, body: Curve, fraction: float, top_count: float, n0: int
) -> Curve:
    """Put a fraction of the positives among the top ``top_count`` negatives.

    Args:
        body: ROC of the remaining positives.
        fraction: Share ``tau`` of positives in the subgroup.
        top_count: Width of the subgroup's FPR range in negatives, ``c``.
        n0: Negative sample size defining the width ``c / n0``.

    Returns:
        ``R(t) = tau min(1, t n0 / c) + (1 - tau) B(t)``.
    """
    if not 0.0 < fraction < 1.0 or top_count <= 0 or n0 < 1:
        raise ValueError("fraction must be in (0, 1); top_count and n0 positive")
    width = top_count / n0
    if width >= 1.0:
        raise ValueError("the subgroup must occupy less than the whole FPR range")
    grid = np.unique(np.r_[fine_grid(), body.t, np.linspace(0.0, width, 257)])
    values = fraction * np.minimum(1.0, grid / width) + (1.0 - fraction) * (
        _body_values(body, grid)
    )
    return Curve(grid, values)


def separated_top_curve(*, body: Curve, fraction: float) -> Curve:
    """Put a fraction of the positives above every negative.

    Args:
        body: ROC of the remaining positives.
        fraction: Share ``pi`` of positives with placement 0, so ``R(0) = pi``.

    Returns:
        ``R(t) = pi + (1 - pi) B(t)`` including ``t = 0``.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    grid = np.unique(np.r_[fine_grid(), body.t])
    values = fraction + (1.0 - fraction) * _body_values(body, grid)
    return Curve(grid, values)


def diagonal_curve() -> Curve:
    """The chance ROC ``R(t) = t``."""
    grid = np.array([0.0, 1.0])
    return Curve(grid, grid.copy())


def interior_jump_curve(
    *, n0: int, height: float, location: float = 0.3, base_auc: float = 0.8
) -> Curve:
    """Binormal body with a jump inside a quarter of one negative gap.

    ``R(t) = (1 - h) B(t) + h clip((t - t0) / w, 0, 1)`` with
    ``w = 1 / (4 n0)``: a positive subgroup sitting where the negatives have
    almost no mass (a support gap of the negative scores).

    Args:
        n0: Negative sample size defining the jump width.
        height: Jump height ``h``.
        location: Jump FPR ``t0``.
        base_auc: AUC of the equal-variance binormal body.

    Returns:
        The jump ROC.
    """
    width = 1.0 / (4.0 * n0)
    grid = np.unique(np.r_[fine_grid(), location, location + width])
    shift = np.sqrt(2.0) * norm.ppf(base_auc)
    body = norm.cdf(shift + norm.ppf(np.clip(grid, 1e-15, 1.0 - 1e-15)))
    body[grid == 0.0], body[grid == 1.0] = 0.0, 1.0
    values = (1.0 - height) * body + height * np.clip((grid - location) / width, 0, 1)
    return Curve(grid, values)
