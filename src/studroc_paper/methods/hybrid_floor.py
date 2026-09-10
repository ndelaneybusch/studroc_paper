"""Localized exact M3 floor for the rank-space fiducial ROC band.

The fiducial cloud completes each class's CDF inside the other class's gaps
with sorted uniforms, which amounts to assuming the ROC is linear across
every gap. That assumption is harmless in the interior, where gaps are
``O(log n / n)`` wide, but it fails at the two ends of the curve: the gap
before the first negative anchor and the run of negatives below the smallest
positive can each carry order-one relative uncertainty that a straight chord
misrepresents. Convex corners (heavy-tailed positives at high AUC) and
unsampled slivers of positive mass turn that misrepresentation into a
lower-edge miss (theory doc section 7.4; Stage F report sections 4 and 7).

The floor repairs exactly those two regions by taking the pointwise hull of
the fiducial band and the exact M3 band there, then closing the result by
widening only. Three statements are exact for every ROC curve and every
region rule, including rules that read the data or the cloud:

* the floored band contains the raw fiducial band pointwise;
* a miss inside the region implies a full-curve M3 miss, so the in-region
  miss probability is at most the M3 level;
* the whole-band miss probability is at most the M3 level plus the
  probability that the raw band misses somewhere outside the region.

The exterior term has no distribution-free bound. The region rule therefore
changes width and the unproved term only, never the proved ones.

Region rules
------------
``"exact"`` derives both cut points from exact laws of the cloud and the
sample, each with a declared budget:

* **Left prefix.** At native grid point ``k`` the number of draws whose
  first negative anchor lies beyond ``t_k`` is ``Binomial(M, p_k)`` with
  ``p_k = (1 - k / n0) ** n0`` exactly, because that anchor is
  ``Beta(1, n0)`` whatever the truth is. Those draws contribute chord
  values, which are always the lowest values at ``t_k``, so the band's
  lower edge (the ``j``-th smallest draw) is a chord whenever at least
  ``j`` such draws exist. The cutoff is the smallest ``k`` at which that
  event has probability at most ``eps``; the cutoff point is included.
  The rule is insensitive to ``eps`` (two orders of magnitude move it by
  at most one grid point) and uses the exact chord probability, which
  matters at small ``n0`` where ``exp(-k)`` overstates it.
* **Right run.** Let ``K`` be the number of negatives ranked below the
  smallest positive. Conditional on the positives, ``K`` is
  ``Binomial(n0, S)`` with ``S`` the unknown negative mass below that
  positive, so ``S <= BetaInv(1 - delta; K + 1, n0 - K)`` with probability
  at least ``1 - delta``. The region starts at the grid point where that
  upper limit lands, rounded outward. The same limit bounds the auxiliary
  coordinate of the smallest positive under every completion (theory doc
  Proposition 10a), and it is pointwise optimal among monotone bounds for a
  single boundary.

``"stage_f"`` is the frozen rule measured in the Stage F study: the first
``ceil(log(M + 1))`` grid points and the saturated run extended inward by
``ceil(2 sqrt(K))``. On the left it replaces the declared budget by the
condition that the expected chord count be at most one, which at a trim
depth of 5 sits between ``eps`` of about 4e-3 and 5e-5 depending on
rounding, so the two rules agree within one grid point in either direction
at ``alpha = .05`` and the frozen rule is longer at ``alpha = .5``. On the
right it is a normal approximation to the Beta inversion, shorter for runs
under about ten points and longer for runs in the hundreds. It is retained
as a comparison arm.
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta as beta_dist
from scipy.stats import binom

FloorRule = Literal["exact", "stage_f"]


@dataclass(frozen=True)
class M3Floor:
    """Settings for the localized M3 floor.

    Attributes:
        rule: Region rule, ``"exact"`` (declared-budget cut points) or
            ``"stage_f"`` (the frozen Stage F frontier rule).
        eps: Left-cutoff budget for the exact rule: the largest tolerated
            probability that the raw lower edge at the first unfloored grid
            point is a chord draw.
        delta: Right-margin budget for the exact rule: the largest tolerated
            probability that the true saturated-run boundary lies inside the
            unfloored region.
        alpha: Level of the M3 component. ``None`` uses the band's own
            ``alpha``, the Stage F choice; smaller values widen the floor.
        split_ratio: M3 log-confidence share assigned to the negative class.
    """

    rule: FloorRule = "exact"
    eps: float = 1e-3
    delta: float = 0.025
    alpha: float | None = None
    split_ratio: float = 0.5

    def __post_init__(self) -> None:
        """Validate the region rule and its probability budgets."""
        if self.rule not in ("exact", "stage_f"):
            raise ValueError(f"rule must be 'exact' or 'stage_f', got {self.rule!r}")
        if not 0.0 < self.eps < 1.0:
            raise ValueError(f"eps must be in (0, 1), got {self.eps}")
        if not 0.0 < self.delta < 0.5:
            raise ValueError(f"delta must be in (0, 0.5), got {self.delta}")
        if self.alpha is not None and not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1) or None, got {self.alpha}")
        if not 0.0 < self.split_ratio < 1.0:
            raise ValueError(f"split_ratio must be in (0, 1), got {self.split_ratio}")


def resolve_floor(m3_floor: bool | M3Floor) -> M3Floor | None:
    """Turn the public toggle into floor settings or ``None``.

    Args:
        m3_floor: ``False`` disables the floor, ``True`` selects the default
            exact rule, and an :class:`M3Floor` supplies explicit settings.

    Returns:
        Floor settings, or ``None`` when the floor is disabled.
    """
    if isinstance(m3_floor, M3Floor):
        return m3_floor
    return M3Floor() if m3_floor else None


def chord_probability(*, n0: int, k: int) -> float:
    """Exact probability that a draw's first negative anchor lies beyond ``k/n0``.

    Args:
        n0: Negative-class sample size.
        k: Native grid index in ``0..n0``.

    Returns:
        ``(1 - k / n0) ** n0``, evaluated in log space.
    """
    if k >= n0:
        return 0.0
    return math.exp(n0 * math.log1p(-k / n0))


def exact_left_cutoff(*, n0: int, n_draws: int, trim_depth: int, eps: float) -> int:
    """Return the inclusive left cutoff of the exact floor rule.

    The cutoff is the smallest grid index at which fewer than ``trim_depth``
    chord draws are present with probability at least ``1 - eps``, so that
    the raw lower edge beyond it is set by anchored draws.

    Args:
        n0: Negative-class sample size.
        n_draws: Fiducial cloud size ``M``.
        trim_depth: Realized one-indexed trim depth ``j``.
        eps: Tolerated probability that the edge at the cutoff is a chord.

    Returns:
        Inclusive native-grid index in ``0..n0``.

    Raises:
        ValueError: If any argument is out of range.
    """
    if n0 < 1 or n_draws < 1 or trim_depth < 1:
        raise ValueError("n0, n_draws and trim_depth must be positive")
    if not 0.0 < eps < 1.0:
        raise ValueError(f"eps must be in (0, 1), got {eps}")
    for k in range(n0 + 1):
        p_k = chord_probability(n0=n0, k=k)
        if binom.sf(trim_depth - 1, n_draws, p_k) <= eps:
            return k
    return n0


def exact_right_start(*, n0: int, run_length: int, delta: float) -> int:
    """Return the first grid index of the exact right floor region.

    Args:
        n0: Negative-class sample size.
        run_length: Number ``K`` of negatives ranked below the smallest
            positive, equivalently the length of the empirical-TPR-1 run.
        delta: Tolerated probability that the true boundary lies left of the
            returned index.

    Returns:
        Native-grid index in ``0..n0 - K``; the region is every index at or
        beyond it.

    Raises:
        ValueError: If any argument is out of range.
    """
    if n0 < 1 or not 0 <= run_length <= n0:
        raise ValueError("run_length must lie in 0..n0")
    if not 0.0 < delta < 0.5:
        raise ValueError(f"delta must be in (0, 0.5), got {delta}")
    if run_length == n0:
        return 0
    s_upper = float(beta_dist.ppf(1.0 - delta, run_length + 1, n0 - run_length))
    expanded = math.ceil(n0 * s_upper - 1e-9)
    return max(0, n0 - expanded)


def stage_f_left_cutoff(*, n0: int, n_draws: int) -> int:
    """Return the frozen Stage F inclusive left cutoff ``min(n0, ceil(log(M+1)))``."""
    if n0 < 1 or n_draws < 1:
        raise ValueError("n0 and n_draws must be positive")
    return min(n0, math.ceil(math.log(n_draws + 1)))


def stage_f_right_start(*, n0: int, run_length: int) -> int:
    """Return the frozen Stage F right start: the run extended by ``ceil(2 sqrt K)``."""
    if n0 < 1 or not 0 <= run_length <= n0:
        raise ValueError("run_length must lie in 0..n0")
    margin = math.ceil(2.0 * math.sqrt(max(run_length, 1)))
    return max(0, n0 - run_length - margin)


def floor_region(
    *, khat: NDArray, n_draws: int, trim_depth: int, floor: M3Floor
) -> NDArray[np.bool_]:
    """Return the native-grid mask on which the M3 hull is taken.

    Args:
        khat: Empirical positive counts on the native grid, ``khat[k]`` being
            the number of positives ranked above the ``(k + 1)``-th negative
            and ``khat[n0] = n1``.
        n_draws: Fiducial cloud size.
        trim_depth: Realized one-indexed trim depth.
        floor: Floor settings selecting the rule and its budgets.

    Returns:
        Boolean mask of length ``n0 + 1``.

    Raises:
        ValueError: If ``khat`` is not a valid empirical count map.
    """
    counts = np.asarray(khat, dtype=np.int64)
    if counts.ndim != 1 or len(counts) < 2:
        raise ValueError("khat must be a one-dimensional array of length n0 + 1 >= 2")
    n1 = int(counts[-1])
    if counts[0] < 0 or n1 < 1 or np.any(np.diff(counts) < 0):
        raise ValueError("khat must be nondecreasing, nonnegative, and end at n1 >= 1")
    n0 = len(counts) - 1
    k_sat = int(np.flatnonzero(counts == n1)[0])
    run_length = n0 - k_sat
    if floor.rule == "exact":
        k_left = exact_left_cutoff(
            n0=n0, n_draws=n_draws, trim_depth=trim_depth, eps=floor.eps
        )
        right_start = exact_right_start(n0=n0, run_length=run_length, delta=floor.delta)
    else:
        k_left = stage_f_left_cutoff(n0=n0, n_draws=n_draws)
        right_start = stage_f_right_start(n0=n0, run_length=run_length)
    index = np.arange(n0 + 1)
    return (index <= k_left) | (index >= right_start)


def stitch_m3_floor(
    *,
    lower: NDArray,
    upper: NDArray,
    m3_lower: NDArray,
    m3_upper: NDArray,
    region: NDArray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Take the M3 hull on the region and close the result by widening only.

    The lower edge becomes its reverse cumulative minimum and the upper edge
    its forward cumulative maximum, so every containment of the inputs is
    preserved and the output is monotone.

    Args:
        lower: Raw fiducial lower edge on the native grid.
        upper: Raw fiducial upper edge on the native grid.
        m3_lower: M3 lower edge on the same grid.
        m3_upper: M3 upper edge on the same grid.
        region: Boolean mask selecting the floored grid points.

    Returns:
        Floored lower and upper edges, clipped to ``[0, 1]``.

    Raises:
        ValueError: If the inputs do not share one shape.
    """
    arrays = [
        np.asarray(a, dtype=np.float64) for a in (lower, upper, m3_lower, m3_upper)
    ]
    mask = np.asarray(region, dtype=bool)
    if len({a.shape for a in arrays} | {mask.shape}) != 1:
        raise ValueError("band edges and region must have the same shape")
    fid_lower, fid_upper, exact_lower, exact_upper = arrays
    out_lower = np.where(mask, np.minimum(fid_lower, exact_lower), fid_lower)
    out_upper = np.where(mask, np.maximum(fid_upper, exact_upper), fid_upper)
    out_lower = np.minimum.accumulate(out_lower[::-1])[::-1]
    out_upper = np.maximum.accumulate(out_upper)
    return np.clip(out_lower, 0.0, 1.0), np.clip(out_upper, 0.0, 1.0)


def apply_m3_floor(
    *,
    lab_s: NDArray,
    khat: NDArray,
    lower: NDArray,
    upper: NDArray,
    alpha: float,
    n_draws: int,
    trim_depth: int,
    floor: M3Floor,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Floor a native-grid fiducial band with the exact M3 band.

    Args:
        lab_s: Tie-resolved 0/1 labels in descending score order.
        khat: Empirical positive counts on the native grid.
        lower: Raw fiducial lower edge after its corner allowances.
        upper: Raw fiducial upper edge after its corner allowances.
        alpha: Level of the fiducial band; the M3 level when
            ``floor.alpha`` is ``None``.
        n_draws: Fiducial cloud size.
        trim_depth: Realized one-indexed trim depth.
        floor: Floor settings.

    Returns:
        Floored lower and upper edges on the native grid.
    """
    # Imported here because m3_band_rs depends on the fiducial modules that
    # call this function.
    from .m3_band_rs import _m3_band_from_labels_rs

    _, m3_lower, m3_upper = _m3_band_from_labels_rs(
        lab_s,
        alpha=alpha if floor.alpha is None else floor.alpha,
        split_ratio=floor.split_ratio,
    )
    region = floor_region(
        khat=khat, n_draws=n_draws, trim_depth=trim_depth, floor=floor
    )
    return stitch_m3_floor(
        lower=lower, upper=upper, m3_lower=m3_lower, m3_upper=m3_upper, region=region
    )
