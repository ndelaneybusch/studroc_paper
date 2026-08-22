"""Rank-Space Fiducial Confidence Bands for ROC Curves.

This module implements a simultaneous confidence band for the true population
ROC curve built from the exact rank-space structure of the two-sample problem.

The construction rests on one reduction: the ROC curve is invariant to
strictly increasing transforms of the score, and under the transform
``s -> 1 - F(s)`` (``F`` = negative-class CDF) the negatives become iid
Uniform(0,1) while the positives become iid with CDF exactly the true ROC.
Everything the data reveal about the ROC is therefore contained in the
interleaving ranks of the two samples, and the exact finite-sample law of
each class's CDF at its own order statistics is Dirichlet(1, ..., 1)
(uniform spacings), at every n, for every continuous score distribution.

The band is assembled in four steps:

1. **Fiducial cloud.** Draw ``n_draws`` pairs of class CDFs from their
   Dirichlet spacings laws, placing the other class's within-gap elements at
   sorted-uniform fractions of the gap, and compose each pair into a fiducial
   ROC curve on the grid ``t_k = k / n0``.
2. **Equal-local-levels trim.** Score each draw by its min-p depth (the
   minimum over grid points of its rank from either end of the cloud) and
   trim to depth ``j`` = the ``alpha_eff``-quantile of the depths, where
   ``alpha_eff = 1 - (1 - alpha)**trim_exponent``. The default exponent 2 is
   the empirically calibrated level remap (one simultaneity budget per
   class, a Sidak-like correction); exponent 1 gives the raw fiducial
   credible band, which is conservative at central alpha levels.
3. **Band.** Lower/upper edges are the pointwise j-th smallest / j-th
   largest draws.
4. **Exact binomial allowances at degenerate corners.** The upper edge is
   unioned with the exact Clopper-Pearson upper bound at the band's own
   local level ``j / (n_draws + 1)`` (essential: a credible upper edge
   cannot touch 1, but the frequentist bound at an empirical TPR of 1 must
   equal 1), and the lower edge is set to 0 wherever no positive exceeds
   the operating threshold (the free mirror of the same fact).

The method is fully rank-based: coverage depends on the data-generating
process only through the true ROC curve and the two sample sizes. Ties are
broken at random by default, which makes the estimand the trapezoidal
(Mann-Whitney) ROC of the observed score distribution exactly.

Tuning inputs are Monte Carlo budgets, not shape parameters: ``n_draws``
controls resolution (self-diagnosed via the realized trim depth; a warning
is raised when it is too small) and ``trim_exponent`` selects the level
remap.
"""

import math
import warnings

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import beta as beta_dist
from torch import Tensor

from .method_utils import torch_to_numpy

TieBreak = str  # "random" | "even"

_CHUNK_DRAWS = 1024  # draw-generation chunk (bounds peak numpy memory)
_CHUNK_COLS = 512  # grid-column chunk for the rank passes


def _merged_labels(
    y_true: NDArray, y_score: NDArray, tie_break: TieBreak, rng: np.random.Generator
) -> NDArray:
    """Return class labels sorted by descending score (ascending rank space).

    Ties are broken at random (``"random"``, the default estimand-exact
    convention) or by even within-block spreading of each class
    (``"even"``, deterministic and slightly conservative). Ranking one
    class systematically above the other inside tied blocks is invalid and
    deliberately not offered.
    """
    if tie_break == "random":
        sub_key = rng.random(len(y_score))
    elif tie_break == "even":
        # Position the i-th tied member of each class at (i - 1/2) / count
        # within its block, per class, then interleave by that fraction.
        order = np.lexsort((y_true, -y_score))
        sub_key = np.empty(len(y_score))
        scores_sorted = y_score[order]
        labels_sorted = y_true[order]
        block_starts = np.flatnonzero(
            np.concatenate([[True], scores_sorted[1:] != scores_sorted[:-1]])
        )
        block_ids = np.cumsum(
            np.concatenate([[True], scores_sorted[1:] != scores_sorted[:-1]])
        )
        for start in block_starts:
            end = start + np.searchsorted(
                block_ids[start:], block_ids[start], side="right"
            )
            for cls in (0, 1):
                members = np.flatnonzero(labels_sorted[start:end] == cls) + start
                if len(members):
                    sub_key[order[members]] = (np.arange(len(members)) + 0.5) / len(
                        members
                    )
    else:
        raise ValueError(f"Unknown tie_break: {tie_break!r}")
    order = np.lexsort((sub_key, -y_score))
    return y_true[order].astype(np.int8)


def _run_ids(mask: NDArray) -> NDArray:
    """Run ids of consecutive-True runs, aligned to the True elements."""
    starts = mask & ~np.concatenate([[False], mask[:-1]])
    return np.cumsum(starts)[mask] - 1


def _axis_coords(
    count_own: NDArray,
    is_other: NDArray,
    spacings_cum: NDArray,
    rid_other: NDArray,
    rng: np.random.Generator,
) -> NDArray:
    """Fiducial coordinates on one axis for all merged elements.

    Elements of the axis-owning class sit at their Dirichlet cumulative
    masses; elements of the other class are spread at sorted-uniform
    fractions of the gap they fall in.
    """
    m, _ = spacings_cum.shape
    n_elems = len(count_own)
    out = np.empty((m, n_elems))
    own = ~is_other
    out[:, own] = spacings_cum[:, count_own[own] - 1]
    j = count_own[is_other]
    base = np.where(j > 0, spacings_cum[:, np.maximum(j - 1, 0)], 0.0)
    mass = spacings_cum[:, j] - base
    u = rng.random((m, int(is_other.sum())))
    key = np.sort(rid_other[None, :] + u, axis=1)
    frac = key - rid_other[None, :]
    out[:, is_other] = base + frac * mass
    return out


def _fiducial_cloud(
    lab_s: NDArray,
    n0: int,
    n1: int,
    n_draws: int,
    grid: NDArray,
    rng: np.random.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Draw the fiducial ROC cloud, evaluated on ``grid``: (n_draws, K)."""
    is_pos = lab_s.astype(bool)
    ncnt = np.cumsum(~is_pos).astype(np.int64)
    pcnt = np.cumsum(is_pos).astype(np.int64)
    rid_pos = _run_ids(is_pos)
    rid_neg = _run_ids(~is_pos)
    n_elems = len(lab_s)
    grid_t = torch.as_tensor(grid, device=device, dtype=torch.float64)

    out = torch.empty((n_draws, len(grid)), device=device, dtype=dtype)
    for lo in range(0, n_draws, _CHUNK_DRAWS):
        m = min(_CHUNK_DRAWS, n_draws - lo)
        p = rng.standard_exponential((m, n0 + 1))
        q = rng.standard_exponential((m, n1 + 1))
        pc = np.cumsum(p, axis=1) / p.sum(axis=1, keepdims=True)
        qc = np.cumsum(q, axis=1) / q.sum(axis=1, keepdims=True)
        xv = np.zeros((m, n_elems + 2))
        yv = np.zeros((m, n_elems + 2))
        xv[:, 1:-1] = _axis_coords(ncnt, is_pos, pc, rid_pos, rng)
        yv[:, 1:-1] = _axis_coords(pcnt, ~is_pos, qc, rid_neg, rng)
        xv[:, -1] = 1.0
        yv[:, -1] = 1.0

        xt = torch.as_tensor(xv, device=device)
        yt = torch.as_tensor(yv, device=device)
        tq = grid_t.expand(m, -1)
        idx = torch.searchsorted(xt.contiguous(), tq.contiguous(), right=True)
        idx = torch.clamp(idx, 1, n_elems + 1)
        x1 = torch.gather(xt, 1, idx - 1)
        x2 = torch.gather(xt, 1, idx)
        y1 = torch.gather(yt, 1, idx - 1)
        y2 = torch.gather(yt, 1, idx)
        frac = torch.clamp((tq - x1) / torch.clamp(x2 - x1, min=1e-300), 0.0, 1.0)
        out[lo : lo + m] = (y1 + frac * (y2 - y1)).to(dtype)
    return out


def _minp_depths(cloud: Tensor) -> Tensor:
    """Min-p depth of each draw: min over grid of rank from either cloud end."""
    n_draws, n_cols = cloud.shape
    s = torch.full((n_draws,), n_draws, device=cloud.device, dtype=torch.int64)
    for lo in range(0, n_cols, _CHUNK_COLS):
        cols = cloud[:, lo : lo + _CHUNK_COLS].T.contiguous()
        csort, _ = torch.sort(cols, dim=1)
        rank_le = torch.searchsorted(csort, cols, right=True)
        rank_ge = n_draws - torch.searchsorted(csort, cols, right=False)
        s = torch.minimum(s, torch.minimum(rank_le, rank_ge).min(dim=0).values)
    return s


def _pointwise_order_stats(cloud: Tensor, j: int) -> tuple[Tensor, Tensor]:
    """Per-column j-th smallest and j-th largest values of the cloud."""
    n_draws, n_cols = cloud.shape
    lower = torch.empty(n_cols, device=cloud.device, dtype=cloud.dtype)
    upper = torch.empty(n_cols, device=cloud.device, dtype=cloud.dtype)
    for lo in range(0, n_cols, _CHUNK_COLS):
        cols = cloud[:, lo : lo + _CHUNK_COLS].T.contiguous()
        lower[lo : lo + cols.shape[0]] = torch.kthvalue(cols, j, dim=1).values
        upper[lo : lo + cols.shape[0]] = torch.kthvalue(
            cols, n_draws - j + 1, dim=1
        ).values
    return lower, upper


def _auto_n_draws(n_grid: int, alpha_eff: float) -> int:
    """Monte Carlo budget from the empirical local-level law.

    The realized local level of the trimmed band is approximately
    ``ell(K, a) = 9.7e-4 * (a / 0.05)**1.2 * (K / 500)**-0.27`` (fitted in
    ``stats/experiments/m2_report.md`` P4); requiring a trim depth of at
    least ~5 for alpha-resolution gives ``n_draws ~ 5 / ell``.
    """
    ell = 9.7e-4 * (alpha_eff / 0.05) ** 1.2 * (n_grid / 500.0) ** (-0.27)
    return int(np.clip(math.ceil(5.0 / ell), 2000, 20000))


def fiducial_band(
    y_true: NDArray | Tensor,
    y_score: NDArray | Tensor,
    alpha: float = 0.05,
    n_draws: int | None = None,
    trim_exponent: float = 2.0,
    k: int | None = None,
    tie_break: TieBreak = "random",
    random_state: int | np.random.Generator | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute a rank-space fiducial simultaneous confidence band for the ROC.

    Draws a fiducial cloud of ROC curves from the exact Dirichlet-spacings
    law of each class's CDF at its order statistics, trims the cloud to its
    most central curves by equal-local-levels (min-p) depth at the remapped
    level ``1 - (1 - alpha)**trim_exponent``, takes the pointwise envelope of
    the retained depth, and applies two exact binomial corner allowances:
    the Clopper-Pearson upper bound at the band's own local level (the upper
    edge must reach 1 wherever the empirical TPR is 1) and a zero lower
    bound wherever the empirical TPR is 0. See the module docstring for the
    construction and ``stats/experiments/m2_report.md`` for its measured
    behavior.

    Args:
        y_true: Binary class labels (0 = negative, 1 = positive). Accepts
            numpy arrays or torch tensors.
        y_score: Prediction scores, higher indicating the positive class.
            Ties are handled by ``tie_break``; scores are otherwise used
            only through their ranks.
        alpha: Significance level; the band targets simultaneous coverage
            ``1 - alpha``. Defaults to 0.05.
        n_draws: Number of fiducial draws. ``None`` selects a budget from
            the grid size and level (2,000-20,000; ~10,000 at
            ``n0 = 5000``, ``alpha = 0.05``). The realized trim depth is
            self-diagnosing: a warning is raised if it falls below 3,
            meaning ``n_draws`` is too small for the requested level and
            the band falls back toward the (conservative) full envelope of
            the cloud.
        trim_exponent: Exponent ``C`` of the level remap
            ``alpha_eff = 1 - (1 - alpha)**C``. ``2.0`` (default) is the
            empirically centred choice, read as one simultaneity budget per
            class; ``1.0`` gives the raw fiducial credible band, valid but
            over-conservative at central alpha levels.
        k: Optional output grid size. ``None`` (default) returns the band on
            its native grid of ``n0 + 1`` points ``t = i / n0``. Otherwise
            the band is step-resampled onto ``linspace(0, 1, k)``
            conservatively (upper edge from the next native point, lower
            edge from the previous one).
        tie_break: ``"random"`` (default) breaks tied scores uniformly at
            random, making the estimand exactly the trapezoidal
            (Mann-Whitney) ROC of the score distribution; ``"even"``
            spreads each class evenly within tied blocks (deterministic,
            slightly conservative).
        random_state: Seed or ``numpy.random.Generator`` for the fiducial
            draws (and random tie-breaking). ``None`` draws fresh entropy.

    Returns:
        Tuple of ``(fpr_grid, lower_envelope, upper_envelope)`` numpy
        arrays, with ``lower[0] = 0`` and ``upper[-1] = 1``.

    Raises:
        ValueError: If either class is empty or arguments are out of range.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y_true = np.repeat([0, 1], 100)
        >>> y_score = np.concatenate([rng.normal(0, 1, 100), rng.normal(1.5, 1, 100)])
        >>> fpr, lo, hi = fiducial_band(y_true, y_score, alpha=0.05, random_state=1)
        >>> fpr.shape, lo.shape, hi.shape
        ((101,), (101,), (101,))
        >>> bool(np.all(lo <= hi)) and lo[0] == 0.0 and hi[-1] == 1.0
        True
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if trim_exponent <= 0.0:
        raise ValueError(f"trim_exponent must be positive, got {trim_exponent}")

    y_true_np = (
        torch_to_numpy(y_true) if isinstance(y_true, Tensor) else np.asarray(y_true)
    )
    y_score_np = (
        torch_to_numpy(y_score) if isinstance(y_score, Tensor) else np.asarray(y_score)
    )
    y_true_np = y_true_np.astype(np.int64)
    n0 = int((y_true_np == 0).sum())
    n1 = int((y_true_np == 1).sum())
    if n0 == 0 or n1 == 0:
        raise ValueError(f"Both classes must be present (n0={n0}, n1={n1})")

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    alpha_eff = 1.0 - (1.0 - alpha) ** trim_exponent
    grid = np.arange(n0 + 1) / n0
    if n_draws is None:
        n_draws = _auto_n_draws(len(grid), alpha_eff)
    if n_draws < 100:
        raise ValueError(f"n_draws must be at least 100, got {n_draws}")

    # Descending-score label sequence (= ascending rank space), ties broken.
    lab_s = _merged_labels(y_true_np, y_score_np, tie_break, rng)

    # Empirical TPR counts at each grid point (staircase-upper convention):
    # khat[i] = #positives ranked above the (i+1)-th ranked negative.
    cpos = np.cumsum(lab_s)
    neg_idx = np.flatnonzero(lab_s == 0)
    khat = np.concatenate([cpos[neg_idx], [n1]]).astype(np.int64)

    dtype = torch.float64 if n_draws * len(grid) <= 40_000_000 else torch.float32
    cloud = _fiducial_cloud(lab_s, n0, n1, n_draws, grid, rng, device, dtype)

    # Trim depth: alpha_eff-quantile of the min-p depths, clipped to [1, M/2].
    depths = _minp_depths(cloud)
    depth_sorted, _ = torch.sort(depths)
    j = int(depth_sorted[int(math.floor(alpha_eff * n_draws))].item())
    j = max(1, min(j, n_draws // 2))
    if j < 3:
        warnings.warn(
            f"Realized trim depth j={j} < 3: n_draws={n_draws} is too small "
            f"for alpha={alpha} on a {len(grid)}-point grid; the band falls "
            "back toward the conservative full envelope of the cloud and "
            "nearby alpha levels become indistinguishable. Increase n_draws.",
            stacklevel=2,
        )

    lower_t, upper_t = _pointwise_order_stats(cloud, j)
    lower = np.clip(torch_to_numpy(lower_t).astype(np.float64), 0.0, 1.0)
    upper = np.clip(torch_to_numpy(upper_t).astype(np.float64), 0.0, 1.0)

    # Exact binomial allowances at the band's own local level.
    local_level = j / (n_draws + 1)
    cp_upper = np.ones(len(grid))
    interior = khat < n1
    cp_upper[interior] = beta_dist.ppf(
        1.0 - local_level, khat[interior] + 1, n1 - khat[interior]
    )
    upper = np.maximum.accumulate(np.maximum(upper, cp_upper))
    lower[khat == 0] = 0.0

    upper = np.clip(upper, 0.0, 1.0)
    lower[0] = 0.0
    upper[-1] = 1.0

    if k is not None:
        if k < 2:
            raise ValueError(f"k must be at least 2, got {k}")
        out_grid = np.linspace(0.0, 1.0, k)
        upper = upper[np.minimum(np.ceil(out_grid * n0).astype(int), n0)]
        lower = lower[np.floor(out_grid * n0).astype(int)]
        grid = out_grid

    return grid, lower, upper
