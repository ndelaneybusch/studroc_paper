"""M3: the exact composition confidence band for ROC curves (Rust-accelerated).

M3 is the *provable* distribution-free band of this project — the
certification layer next to the tighter, Monte-Carlo-validated fiducial band
(`fiducial_band` / `fiducial_band_rs`). Its finite-sample coverage theorem
(`stats/fiducial_band_theory.md` §12, Proposition 12) holds at every ``n``
for every continuous score distribution, with ties handled exactly via
random tie-breaking (trapezoidal / Mann-Whitney estimand).

Construction
------------
For one sample from a continuous CDF ``H``, the vector ``H(Z_(i))`` is
distributed as uniform order statistics (exactly, at every ``n``). A
two-sided equal-local-levels (ELL) band uses the same local level ``gamma``
at every order statistic::

    BetaInv(gamma; i, n+1-i) <= H(Z_(i)) <= BetaInv(1-gamma; i, n+1-i)

and its simultaneous coverage is a non-crossing probability of uniform order
statistics, computed exactly (to floating-point roundoff) by the
``fiducial_core`` Rust kernel (a counting-process dynamic program over the
sorted bounds). ``gamma`` is calibrated by bisection, returning the
conservative bracket endpoint, so each class band has simultaneous level at
least ``1 - alpha_class`` and within the bisection tolerance of it; no
Monte Carlo and no safety shading enters.

The two class bands compose into an ROC band by monotonicity: with ``F`` the
negative and ``G`` the positive class CDF (rank-space convention),
``R(t) <= G_hi((F_lo)^{-1}(t))`` and ``R(t) >= G_lo((F_hi)^{-1}(t))``.
Written on the merged label sequence, both edges are pure index gathers, so
a band costs O(n) once the two ``gamma`` values are calibrated (cached per
``(n, alpha_class)``).

The class levels multiply (independent samples): ``alpha_class`` per class is
set by ``(1 - alpha_F)(1 - alpha_G) = 1 - alpha`` with the split ratio
``rho``: ``1 - alpha_F = (1-alpha)^rho``. The default ``rho = 0.5`` is the
Sidak split; the ratio is data-independent, so any fixed choice preserves
the theorem (a lever for heavy class imbalance).

Corner honesty: the upper edge at ``t = 0`` is the composition's own value
(the exact Beta bound at the count of positives ranked above the top
negative) — pinning ``U(0) = 0`` is *invalid* distribution-free
(theory doc, Corollary 9.3: continuous scores can have ``R(0) > 0``). Users
who can assert ``R(0) = 0`` (overlapping supports) may opt into the pin via
``assume_r0_zero=True``. ``R(1) = 1`` holds for every continuous DGP, so
both edges are pinned to 1 at ``t = 1``.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta as beta_dist
from torch import Tensor

from .fiducial_band import TieBreak, _merged_labels
from .fiducial_band_rs import _require_fiducial_core
from .method_utils import torch_to_numpy

# Calibrated ELL local levels, keyed by (n, alpha_class rounded to 12 places).
_GAMMA_CACHE: dict[tuple[int, float], float] = {}

# The exact DP is correct to f64 roundoff; aim this far above the target so
# rounding can never tip realized coverage below nominal.
_ROUNDOFF_MARGIN = 1e-9


def _ell_coverage(core, n: int, gamma: float) -> float:
    """Exact simultaneous coverage of the two-sided ELL band at ``gamma``."""
    i = np.arange(1, n + 1, dtype=np.float64)
    lower = beta_dist.ppf(gamma, i, n + 1.0 - i)
    upper = beta_dist.ppf(1.0 - gamma, i, n + 1.0 - i)
    return float(core.ell_crossing_probability(lower, upper))


def _ell_gamma(core, n: int, alpha_class: float) -> float:
    """Largest local level whose exact two-sided ELL coverage >= 1 - alpha_class.

    Bisection on ``log gamma`` between the Bonferroni level
    ``alpha_class / (2n)`` (guaranteed conservative by the union bound) and
    the single-point necessary bound ``alpha_class / 2`` (any larger gamma
    is defeated by one order statistic alone). Returns the conservative
    bracket endpoint, so realized class coverage is >= the target by
    construction.
    """
    key = (n, round(alpha_class, 12))
    cached = _GAMMA_CACHE.get(key)
    if cached is not None:
        return cached
    target = 1.0 - alpha_class + _ROUNDOFF_MARGIN
    lo = alpha_class / (2.0 * n)
    hi = alpha_class / 2.0
    if _ell_coverage(core, n, hi) >= target:
        _GAMMA_CACHE[key] = hi
        return hi
    if _ell_coverage(core, n, lo) < target:
        # The margin pushed the target above the Bonferroni coverage (only
        # possible when the union bound is essentially tight, e.g. n = 1);
        # the Bonferroni level itself is provably >= 1 - alpha_class.
        _GAMMA_CACHE[key] = lo
        return lo
    # Invariant: coverage(lo) >= target > coverage(hi).
    for _ in range(40):
        mid = float(np.sqrt(lo * hi))
        if _ell_coverage(core, n, mid) >= target:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1.0 + 1e-4:
            break
    _GAMMA_CACHE[key] = lo
    return lo


def m3_band_rs(
    y_true: NDArray | Tensor,
    y_score: NDArray | Tensor,
    alpha: float = 0.05,
    k: int | None = None,
    tie_break: TieBreak = "random",
    split_ratio: float = 0.5,
    assume_r0_zero: bool = False,
    random_state: int | np.random.Generator | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute the M3 exact composition ROC confidence band.

    Distribution-free simultaneous confidence band with a finite-sample
    coverage theorem: ``P(forall t: L(t) <= R(t) <= U(t)) >= 1 - alpha`` for
    every continuous score distribution, at every sample size (theory doc
    Proposition 12). Wider than the fiducial band (measured 1.45-2.19x its
    area at alpha=0.05) but provable, deterministic given the ranks, and
    orders of magnitude cheaper: one O(n) pass after a cached per-(n, level)
    calibration.

    Args:
        y_true: Binary class labels (0 = negative, 1 = positive). Accepts
            numpy arrays or torch tensors.
        y_score: Prediction scores, higher indicating the positive class.
            Used only through their ranks; ties handled by ``tie_break``.
        alpha: Significance level; the band guarantees simultaneous coverage
            at least ``1 - alpha``. Defaults to 0.05.
        k: Optional output grid size. ``None`` (default) returns the band on
            its native grid of ``n0 + 1`` points; otherwise the band is
            step-resampled conservatively onto ``linspace(0, 1, k)``.
        tie_break: ``"random"`` (default, estimand-exact for the trapezoidal
            ROC) or ``"even"`` (deterministic, slightly conservative).
        split_ratio: Fraction ``rho`` of the log confidence budget spent on
            the negative-class band: ``1 - alpha_F = (1 - alpha)**rho`` and
            ``1 - alpha_G = (1 - alpha)**(1 - rho)``. The default 0.5 is the
            Sidak split; any fixed value in (0, 1) preserves the theorem.
        assume_r0_zero: If True, pin ``U(0) = 0``. Valid only under the
            *user-asserted* assumption ``R(0) = 0`` (class supports overlap
            at the top); the distribution-free default is the composition's
            own exact bound at ``t = 0`` (Corollary 9.3 of the theory doc).
        random_state: Seed or ``numpy.random.Generator`` for tie-breaking
            only — the band is otherwise deterministic given the ranks.

    Returns:
        Tuple of ``(fpr_grid, lower_envelope, upper_envelope)`` numpy
        arrays, with ``lower[0] = 0`` and ``upper[-1] = lower[-1] = 1``.

    Raises:
        ImportError: If the ``fiducial_core`` extension is not built.
        ValueError: If either class is empty or arguments are out of range.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y_true = np.repeat([0, 1], 100)
        >>> y_score = np.concatenate([rng.normal(0, 1, 100), rng.normal(1.5, 1, 100)])
        >>> fpr, lo, hi = m3_band_rs(y_true, y_score, alpha=0.05, random_state=1)
        >>> fpr.shape, lo.shape, hi.shape
        ((101,), (101,), (101,))
        >>> bool(np.all(lo <= hi)) and lo[0] == 0.0 and hi[-1] == 1.0
        True
    """
    core = _require_fiducial_core()

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not 0.0 < split_ratio < 1.0:
        raise ValueError(f"split_ratio must be in (0, 1), got {split_ratio}")

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

    # Exact per-class levels: (1 - alpha_F)(1 - alpha_G) = 1 - alpha.
    alpha_f = 1.0 - (1.0 - alpha) ** split_ratio
    alpha_g = 1.0 - (1.0 - alpha) ** (1.0 - split_ratio)
    g0 = _ell_gamma(core, n0, alpha_f)
    g1 = _ell_gamma(core, n1, alpha_g)

    i0 = np.arange(1, n0 + 1, dtype=np.float64)
    b0lo = beta_dist.ppf(g0, i0, n0 + 1.0 - i0)
    b0hi = beta_dist.ppf(1.0 - g0, i0, n0 + 1.0 - i0)
    i1 = np.arange(1, n1 + 1, dtype=np.float64)
    b1lo = beta_dist.ppf(g1, i1, n1 + 1.0 - i1)
    b1hi = beta_dist.ppf(1.0 - g1, i1, n1 + 1.0 - i1)

    # Composition index maps on the native grid t_j = j / n0 (1-based order
    # statistics of the negatives): iup(t) = min{i : b0lo[i] >= t} composes
    # the G-upper band with the F-lower band; ilo(t) = min{i : b0hi[i] >= t}
    # the G-lower band with the F-upper band.
    grid = np.arange(n0 + 1) / n0
    iup = np.searchsorted(b0lo, grid, side="left") + 1
    sent = iup > n0  # no such order statistic: the upper edge is vacuous (1)
    iup = np.clip(iup, 1, n0)
    ilo = np.minimum(np.searchsorted(b0hi, grid, side="left") + 1, n0 + 1)

    # pcnt[i] = number of positives ranked above the i-th ranked negative
    # (descending scores), pcnt[0] = 0.
    lab_s = _merged_labels(y_true_np, y_score_np, tie_break, rng)
    cpos = np.cumsum(lab_s)
    neg_idx = np.flatnonzero(lab_s == 0)
    pcnt = np.concatenate([[0], cpos[neg_idx]]).astype(np.int64)

    # Degenerate-index extensions: G-band bound at 0 positives is 0; the
    # bound "above the (n1+1)-th positive" is 1.
    b1hi_ext = np.concatenate([[0.0], b1hi, [1.0]])
    b1lo_ext = np.concatenate([[0.0], b1lo])

    upper = b1hi_ext[pcnt[iup] + 1]
    upper[sent] = 1.0
    lower = b1lo_ext[pcnt[ilo - 1]]

    if assume_r0_zero:
        upper[0] = 0.0  # admissible only under the asserted R(0) = 0
    lower[0] = 0.0
    # R(1) = 1 exactly for every continuous DGP (trapezoidal estimand).
    lower[-1] = 1.0
    upper[-1] = 1.0

    if k is not None:
        if k < 2:
            raise ValueError(f"k must be at least 2, got {k}")
        out_grid = np.linspace(0.0, 1.0, k)
        upper = upper[np.minimum(np.ceil(out_grid * n0).astype(int), n0)]
        lower = lower[np.floor(out_grid * n0).astype(int)]
        grid = out_grid

    return grid, lower, upper
