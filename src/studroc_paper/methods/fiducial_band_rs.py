"""Rust-accelerated rank-space fiducial confidence band for ROC curves.

Statistically identical construction to :mod:`.fiducial_band` (see that
module's docstring for the method), with the Monte Carlo core — fiducial
cloud generation, min-p depth trim, and pointwise order statistics — running
in the ``fiducial_core`` Rust extension (rayon-parallel, xoshiro256++ seeded
per draw). The two implementations agree statistically, not bit-wise: they
consume different RNG streams, and the Rust cloud is stored in float32
(granularity three orders below the ``1 / n_draws`` Monte Carlo resolution
of the band edges).

Build the extension once per environment from the project root::

    uv run --with maturin maturin develop --release -m rust/Cargo.toml

Reproducibility contract for the Rust core: the fiducial draws are a pure
function of ``(seed, draw_index)``, so output is bit-identical for the same
seed regardless of thread count.
"""

import warnings

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from .fiducial_band import (
    TieBreak,
    _apply_corner_allowances,
    _auto_n_draws,
    _merged_labels,
    production_trim_rows,
)
from .hybrid_floor import (
    M3Floor,
    apply_m3_floor,
    complement_trim_rows,
    floor_region,
    resolve_floor,
)
from .method_utils import torch_to_numpy


def _require_fiducial_core():
    """Import the Rust extension, with a build hint on failure."""
    try:
        import fiducial_core
    except ImportError as err:
        raise ImportError(
            "The fiducial_core Rust extension is not installed. Build it "
            "from the project root with: "
            "uv run --with maturin maturin develop --release -m rust/Cargo.toml"
        ) from err
    return fiducial_core


def fiducial_band_rs(
    y_true: NDArray | Tensor,
    y_score: NDArray | Tensor,
    alpha: float = 0.05,
    n_draws: int | None = None,
    trim_exponent: float = 1.0,
    k: int | None = None,
    tie_break: TieBreak = "random",
    n_threads: int = 0,
    random_state: int | np.random.Generator | None = None,
    m3_floor: bool | M3Floor = False,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute the rank-space fiducial ROC band using the Rust kernel.

    Drop-in accelerated equivalent of
    :func:`studroc_paper.methods.fiducial_band`: same construction (fiducial
    cloud from Dirichlet spacings, equal-local-levels min-p trim at
    ``1 - (1 - alpha)**trim_exponent`` — over the production thinned
    trim-grid on grids larger than 2001 points — pointwise envelope of the
    retained depth, Clopper-Pearson-form upper allowance and zero lower
    allowance at the degenerate corners), same defaults, same output
    convention. Results match the reference implementation in distribution
    but not bit-wise (independent RNG streams).

    Args:
        y_true: Binary class labels (0 = negative, 1 = positive). Accepts
            numpy arrays or torch tensors.
        y_score: Prediction scores, higher indicating the positive class.
            Used only through their ranks; ties handled by ``tie_break``.
        alpha: Significance level; the band targets simultaneous coverage
            ``1 - alpha``. Defaults to 0.05.
        n_draws: Number of fiducial draws. ``None`` selects the same budget
            rule as the reference implementation (2,000-20,000). A warning
            is raised when the realized trim depth falls below 3.
        trim_exponent: Exponent ``C`` of the level remap
            ``alpha_eff = 1 - (1 - alpha)**C``; ``1.0`` (default) is the raw
            fiducial credible band. Validity caveat (2026-09-01): at C = 1
            the band under-covers inside a curved (AUC, n) wedge — heavy
            tails x high AUC, failures measured n ~ 100 to beyond 6,000,
            coverage not monotone in n (theory doc section 7.3); enable
            ``m3_floor`` or use :func:`m3_band_rs` there. Values above 1
            trim deeper and are
            anti-conservative on heavy-tailed shapes (the former default
            ``2.0`` measured 92-94% at ``alpha = .05`` on t(2) cells at
            ``n >= 500``, and 75% at ``n = 100``).
        k: Optional output grid size. ``None`` (default) returns the band on
            its native grid of ``n0 + 1`` points; otherwise the band is
            step-resampled conservatively onto ``linspace(0, 1, k)``.
        tie_break: ``"random"`` (default) or ``"even"``, as in the reference
            implementation.
        n_threads: Rayon thread count for the kernel; ``0`` (default) uses
            the global pool (all cores).
        random_state: Seed or ``numpy.random.Generator`` for tie-breaking
            and the kernel seed. ``None`` draws fresh entropy.
        m3_floor: Localized exact M3 floor at the two ends of the curve
            (:mod:`.hybrid_floor`). ``False`` (default) returns the raw
            fiducial band. ``True`` applies the default :class:`M3Floor`: the
            trim depth is recalibrated on the rows outside the exact-rule
            region, the hull with the M3 band (at this ``alpha``, with the
            sample-size class split) is taken on the region, and the result
            is closed by tightening. An :class:`M3Floor` sets the rule, its
            budgets, the closure (``"widen"`` or ``"tighten"``), the M3 class
            split, and whether the trim depth is recalibrated. Without
            recalibration the floored band contains the raw band pointwise;
            under either closure the in-region miss probability is capped at
            the M3 level, and misses outside the region have no
            distribution-free bound.

    Returns:
        Tuple of ``(fpr_grid, lower_envelope, upper_envelope)`` numpy
        arrays, with ``lower[0] = 0`` and ``upper[-1] = 1``.

    Raises:
        ImportError: If the ``fiducial_core`` extension is not built.
        ValueError: If either class is empty or arguments are out of range.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y_true = np.repeat([0, 1], 100)
        >>> y_score = np.concatenate([rng.normal(0, 1, 100), rng.normal(1.5, 1, 100)])
        >>> fpr, lo, hi = fiducial_band_rs(y_true, y_score, alpha=0.05, random_state=1)
        >>> fpr.shape, lo.shape, hi.shape
        ((101,), (101,), (101,))
        >>> bool(np.all(lo <= hi)) and lo[0] == 0.0 and hi[-1] == 1.0
        True
    """
    core = _require_fiducial_core()

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

    seed = int(rng.integers(0, 2**64, dtype=np.uint64))
    trim_rows = production_trim_rows(len(grid))
    lower, upper, j = core.fiducial_trimmed_tube(
        lab_s.astype(np.uint8),
        n_draws,
        alpha_eff,
        seed,
        n_threads,
        None if trim_rows is None else trim_rows.astype(np.uint64),
    )
    if j < 3:
        warnings.warn(
            f"Realized trim depth j={j} < 3: n_draws={n_draws} is too small "
            f"for alpha={alpha} on a {len(grid)}-point grid; the band falls "
            "back toward the conservative full envelope of the cloud and "
            "nearby alpha levels become indistinguishable. Increase n_draws.",
            stacklevel=2,
        )

    lower, upper = _apply_corner_allowances(
        lower=lower, upper=upper, khat=khat, n1=n1, trim_depth=j, n_draws=n_draws
    )

    floor = resolve_floor(m3_floor)
    if floor is not None:
        region = floor_region(khat=khat, n_draws=n_draws, trim_depth=j, floor=floor)
        if floor.recalibrate_trim:
            rows = complement_trim_rows(
                n_grid=len(grid), region=region, trim_rows=trim_rows
            )
            if len(rows):
                # Same seed and draw count: the kernel regenerates the
                # identical cloud, so the depth can only rise (Prop. 10b).
                rec_lower, rec_upper, j_rec = core.fiducial_trimmed_tube(
                    lab_s.astype(np.uint8),
                    n_draws,
                    alpha_eff,
                    seed,
                    n_threads,
                    rows.astype(np.uint64),
                )
                if j_rec < j:
                    raise RuntimeError(
                        f"Recalibrated depth {j_rec} fell below the full-grid "
                        f"depth {j}; the kernel did not regenerate the same cloud."
                    )
                lower, upper = _apply_corner_allowances(
                    lower=rec_lower,
                    upper=rec_upper,
                    khat=khat,
                    n1=n1,
                    trim_depth=j_rec,
                    n_draws=n_draws,
                )
        lower, upper = apply_m3_floor(
            lab_s=lab_s,
            lower=lower,
            upper=upper,
            region=region,
            alpha=alpha,
            floor=floor,
        )

    if k is not None:
        if k < 2:
            raise ValueError(f"k must be at least 2, got {k}")
        out_grid = np.linspace(0.0, 1.0, k)
        upper = upper[np.minimum(np.ceil(out_grid * n0).astype(int), n0)]
        lower = lower[np.floor(out_grid * n0).astype(int)]
        grid = out_grid

    return grid, lower, upper
