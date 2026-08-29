"""Shape library for the trim-exponent calibration study.

Implements section 5.1 of ``stats/c_calibration_spec.md``: ten fitting
shapes and six held-out shapes, each a true ROC curve represented as a dense
piecewise-linear ``Curve``. By rank invariance (``fiducial_band_theory.md``
Proposition 2) a cell of the study is fully specified by (curve shape, n0,
n1): replicates are simulated directly in rank space — negatives iid
Uniform(0,1), positives iid with CDF equal to the curve.

Shapes are deterministic: hand-picked members are closed-form or solved by
bisection against fixed targets, and the two LHS-sampled held-out members
are drawn from the paper's DGP mapper with the frozen seed below.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm, weibull_min
from scipy.stats import t as tdist

from studroc_paper.datagen.roc_to_dgp import (
    BetaOpposingSolver,
    BimodalNegativeSolver,
    GammaSolver,
    StudentTSolver,
    hetero_gaussian_params,
    weibull_params,
)

# Frozen seed for the two LHS-sampled held-out shapes (guards against
# designer bias in the hand-picked library; spec section 5.1).
LHS_SHAPE_SEED = 20260824

# Parameter bounds of the paper's LHS design (roc_to_dgp
# ``generate_simulation_design``), restricted to families whose rank-space
# shape is not already spanned by a fitting member: lognormal / logitnormal /
# binormal collapse to the binormal shape under monotone transforms, and
# exponential is the Weibull ROC family.
LHS_FAMILIES: dict[str, dict[str, tuple[float, float]]] = {
    "student_t": {"df": (1.1, 30.0)},
    "gamma": {"shape": (0.5, 10.0)},
    "weibull": {"shape": (0.5, 5.0)},
    "beta_opposing": {"alpha": (0.5, 10.0)},
    "bimodal_negative": {"mixture_weight": (0.1, 0.9), "mode_separation": (0.1, 4.0)},
    "hetero_gaussian": {"sigma_ratio": (0.2, 5.0)},
}
LHS_AUC_BOUNDS = (0.55, 0.99)


def fine_grid() -> np.ndarray:
    """Dense FPR grid, geometric near both corners, for curve construction."""
    a = np.geomspace(1e-9, 0.05, 4000)
    b = np.linspace(0.05, 0.95, 4000)
    c = 1.0 - np.geomspace(1e-9, 0.05, 4000)[::-1]
    return np.unique(np.clip(np.concatenate([[0.0], a, b, c, [1.0]]), 0, 1))


class Curve:
    """A true ROC curve as a monotone piecewise-linear function on [0, 1].

    Attributes:
        t: FPR knots (non-decreasing, spanning [0, 1]).
        r: TPR values at the knots (clipped to [0, 1], monotonized).
    """

    def __init__(self, t: np.ndarray, r: np.ndarray) -> None:
        r = np.maximum.accumulate(np.clip(r, 0, 1))
        self.t = np.asarray(t, dtype=np.float64)
        self.r = r
        ru, idx = np.unique(r, return_index=True)
        self._ri, self._ti = ru, self.t[idx]

    def eval(self, tq: np.ndarray) -> np.ndarray:
        """TPR at the requested FPR values (linear interpolation)."""
        return np.interp(tq, self.t, self.r)

    def inv(self, v: np.ndarray) -> np.ndarray:
        """Generalized inverse: placement values with CDF equal to the curve."""
        return np.interp(v, self._ri, self._ti)

    def auc(self) -> float:
        """Area under the curve (trapezoidal)."""
        return float(np.trapezoid(self.r, self.t))


def _curve_from_cdfs(
    cdf_neg: Callable[[np.ndarray], np.ndarray],
    cdf_pos: Callable[[np.ndarray], np.ndarray],
    thresholds: np.ndarray,
) -> Curve:
    """ROC curve from class CDFs on a dense threshold grid (predict + if
    score > threshold)."""
    tt = 1.0 - cdf_neg(thresholds)
    rr = 1.0 - cdf_pos(thresholds)
    order = np.argsort(tt)
    tt, rr = tt[order], rr[order]
    t = fine_grid()
    r = np.interp(t, tt, rr, left=0.0, right=1.0)
    r[t == 0] = 0.0
    r[t == 1] = 1.0
    return Curve(t, r)


def _threshold_grid(components: list) -> np.ndarray:
    """Dense threshold grid covering the supports of all component
    distributions (union of their quantiles, tails geometrically refined)."""
    qs = np.unique(
        np.concatenate(
            [np.geomspace(1e-10, 0.5, 4000), 1.0 - np.geomspace(1e-10, 0.5, 4000)]
        )
    )
    pieces = [dist.ppf(qs) for dist in components]
    c = np.unique(np.concatenate(pieces))
    return c[np.isfinite(c)]


def make_binormal(auc: float) -> Curve:
    """Equal-variance binormal shape at the given AUC."""
    mu = np.sqrt(2.0) * norm.ppf(auc)
    t = fine_grid()
    tt = np.clip(t, 1e-15, 1 - 1e-15)
    r = norm.cdf(mu + norm.ppf(tt))
    r[t == 0] = 0.0
    r[t == 1] = 1.0
    return Curve(t, r)


def make_hetero_gaussian(auc: float, sigma_ratio: float) -> Curve:
    """Heteroscedastic Gaussian shape (asymmetric curve), closed form."""
    mu = float(hetero_gaussian_params(np.asarray(auc), np.asarray(sigma_ratio)))
    t = fine_grid()
    tt = np.clip(t, 1e-15, 1 - 1e-15)
    r = norm.cdf((mu + norm.ppf(tt)) / sigma_ratio)
    r[t == 0] = 0.0
    r[t == 1] = 1.0
    return Curve(t, r)


def make_t_shape(auc: float, df: float) -> Curve:
    """Student-t location-shift shape at the given AUC (heavy tails)."""
    delta = StudentTSolver().solve(df, auc)
    comps = [tdist(df=df), tdist(df=df, loc=delta)]
    c = _threshold_grid(comps)
    return _curve_from_cdfs(comps[0].cdf, comps[1].cdf, c)


def make_bimodal_negative(auc: float, sep: float, weight: float = 0.5) -> Curve:
    """Bimodal-negative shape: Gaussian mixture negatives vs Gaussian
    positives (mid-curve inflection and plateau)."""
    pos_mean = BimodalNegativeSolver().solve(weight, sep, auc)
    neg_comps = [norm(loc=0.0), norm(loc=sep)]
    pos = norm(loc=pos_mean)

    def cdf_neg(x: np.ndarray) -> np.ndarray:
        return weight * neg_comps[0].cdf(x) + (1 - weight) * neg_comps[1].cdf(x)

    c = _threshold_grid([*neg_comps, pos])
    return _curve_from_cdfs(cdf_neg, pos.cdf, c)


def make_kink(t_kink: float = 0.004, tpr_kink: float = 0.6) -> Curve:
    """Adversarial piecewise-linear truth: near-vertical to ``tpr_kink`` by
    FPR ``t_kink``, then a shallow straight line to (1, 1)."""
    t = fine_grid()
    r = np.where(
        t <= t_kink,
        tpr_kink * t / t_kink,
        tpr_kink + (1 - tpr_kink) * (t - t_kink) / (1 - t_kink),
    )
    r[t == 0] = 0.0
    r[t == 1] = 1.0
    return Curve(t, r)


def make_trapezoid(base: Curve, q: int) -> Curve:
    """ROC of the base shape's scores quantized to ``q`` equal-negative-mass
    levels under random tie-breaking: the trapezoid through the bin corners.

    A legitimately rough (piecewise-linear, coarsely-kinked) truth — the
    exact estimand of quantized scores, used to test the C* >= 1 floor
    conjecture (spec D5).
    """
    edges = np.arange(q + 1) / q
    return Curve(edges, base.eval(edges))


def _lhs_curve(family: str, auc: float, params: dict[str, float]) -> Curve:
    """Build the true ROC curve of an LHS-sampled DGP from its solved
    parameters (scipy CDF composition on a dense threshold grid)."""
    if family == "student_t":
        df = params["df"]
        delta = StudentTSolver().solve(df, auc)
        comps = [tdist(df=df), tdist(df=df, loc=delta)]
        return _curve_from_cdfs(comps[0].cdf, comps[1].cdf, _threshold_grid(comps))
    if family == "gamma":
        shape = params["shape"]
        ratio = GammaSolver().solve(shape, auc)
        comps = [gamma_dist(a=shape, scale=1.0), gamma_dist(a=shape, scale=ratio)]
        return _curve_from_cdfs(comps[0].cdf, comps[1].cdf, _threshold_grid(comps))
    if family == "weibull":
        shape = params["shape"]
        pos_scale = float(weibull_params(np.asarray(auc), np.asarray(shape)))
        comps = [weibull_min(c=shape, scale=1.0), weibull_min(c=shape, scale=pos_scale)]
        return _curve_from_cdfs(comps[0].cdf, comps[1].cdf, _threshold_grid(comps))
    if family == "beta_opposing":
        a = params["alpha"]
        b = BetaOpposingSolver().solve(a, auc)
        comps = [beta_dist(a=a, b=b), beta_dist(a=b, b=a)]
        return _curve_from_cdfs(comps[0].cdf, comps[1].cdf, _threshold_grid(comps))
    if family == "bimodal_negative":
        return make_bimodal_negative(
            auc, sep=params["mode_separation"], weight=params["mixture_weight"]
        )
    if family == "hetero_gaussian":
        return make_hetero_gaussian(auc, sigma_ratio=params["sigma_ratio"])
    raise ValueError(f"Unknown LHS family: {family!r}")


def lhs_heldout_specs() -> list[dict]:
    """The two LHS-sampled held-out shape specifications (frozen seed).

    Two distinct families are drawn without replacement; the AUC and each
    family's shape parameters are sampled uniformly within the paper's LHS
    design bounds. Deterministic: same seed, same specs.
    """
    rng = np.random.default_rng(LHS_SHAPE_SEED)
    families = sorted(LHS_FAMILIES)
    picks = rng.choice(len(families), size=2, replace=False)
    specs = []
    for i, pick in enumerate(picks):
        family = families[int(pick)]
        auc = float(rng.uniform(*LHS_AUC_BOUNDS))
        params = {
            name: float(rng.uniform(*bounds))
            for name, bounds in sorted(LHS_FAMILIES[family].items())
        }
        specs.append(
            {
                "name": f"lhs{i + 1}_{family}",
                "family": family,
                "auc": auc,
                "params": params,
            }
        )
    return specs


@dataclass(frozen=True)
class ShapeSpec:
    """A named shape with a lazy curve builder.

    Attributes:
        name: Unique shape identifier used in cell names and outputs.
        role: ``"fitting"`` or ``"heldout"`` (spec section 5.1 split).
        build: Zero-argument curve constructor.
        meta: Human-readable provenance of the shape.
    """

    name: str
    role: str
    build: Callable[[], Curve]
    meta: dict = field(default_factory=dict)


def _fitting_specs() -> list[ShapeSpec]:
    return [
        ShapeSpec(
            name=f"binormal_{int(round(a * 100))}",
            role="fitting",
            build=(lambda a=a: make_binormal(a)),
            meta={"family": "binormal", "auc": a},
        )
        for a in (0.60, 0.75, 0.90, 0.95, 0.99)
    ] + [
        ShapeSpec(
            name="hetero_90_r3",
            role="fitting",
            build=lambda: make_hetero_gaussian(0.90, sigma_ratio=3.0),
            meta={"family": "hetero_gaussian", "auc": 0.90, "sigma_ratio": 3.0},
        ),
        ShapeSpec(
            name="t2_95",
            role="fitting",
            build=lambda: make_t_shape(0.95, df=2.0),
            meta={"family": "student_t", "auc": 0.95, "df": 2.0},
        ),
        ShapeSpec(
            name="kink_80",
            role="fitting",
            build=lambda: make_kink(t_kink=0.004, tpr_kink=0.6),
            meta={
                "family": "kink",
                "auc": 0.798,
                "t_kink": 0.004,
                "tpr_kink": 0.6,
                "note": "fixed shape (t_kink = 2/500), not n-dependent",
            },
        ),
        ShapeSpec(
            name="bimodal_90",
            role="fitting",
            build=lambda: make_bimodal_negative(0.90, sep=3.0, weight=0.5),
            meta={"family": "bimodal_negative", "auc": 0.90, "sep": 3.0, "weight": 0.5},
        ),
        ShapeSpec(
            name="trapezoid_q10_90",
            role="fitting",
            build=lambda: make_trapezoid(make_binormal(0.90), q=10),
            meta={
                "family": "trapezoid",
                "base": "binormal_90",
                "q": 10,
                "note": "legitimately rough truth for the D5 floor conjecture",
            },
        ),
    ]


def _heldout_specs() -> list[ShapeSpec]:
    hand_picked = [
        ShapeSpec(
            name="binormal_85",
            role="heldout",
            build=lambda: make_binormal(0.85),
            meta={"family": "binormal", "auc": 0.85},
        ),
        ShapeSpec(
            name="t3_90",
            role="heldout",
            build=lambda: make_t_shape(0.90, df=3.0),
            meta={"family": "student_t", "auc": 0.90, "df": 3.0},
        ),
        ShapeSpec(
            name="bimodal_80_sep15",
            role="heldout",
            build=lambda: make_bimodal_negative(0.80, sep=1.5, weight=0.5),
            meta={"family": "bimodal_negative", "auc": 0.80, "sep": 1.5, "weight": 0.5},
        ),
        ShapeSpec(
            name="heterologit_88_r2",
            role="heldout",
            build=lambda: make_hetero_gaussian(0.88, sigma_ratio=2.0),
            meta={
                "family": "hetero_gaussian",
                "auc": 0.88,
                "sigma_ratio": 2.0,
                "note": (
                    "the spec's 'logit-normal-type shape': a hetero logit-normal "
                    "pair is rank-equivalent to this hetero-Gaussian shape "
                    "(logit is monotone), sigma-ratio distinct from the fitting "
                    "member's 3"
                ),
            },
        ),
    ]
    lhs = [
        ShapeSpec(
            name=spec["name"],
            role="heldout",
            build=(
                lambda family=spec["family"], auc=spec["auc"], params=spec["params"]:
                _lhs_curve(family, auc, params)
            ),
            meta={
                "family": spec["family"],
                "auc": spec["auc"],
                **spec["params"],
                "lhs_seed": LHS_SHAPE_SEED,
            },
        )
        for spec in lhs_heldout_specs()
    ]
    return hand_picked + lhs


@cache
def shape_registry() -> dict[str, ShapeSpec]:
    """All study shapes by name (10 fitting + 6 held-out)."""
    specs = _fitting_specs() + _heldout_specs()
    registry = {s.name: s for s in specs}
    if len(registry) != len(specs):
        raise RuntimeError("Duplicate shape names in the library")
    return registry


@cache
def get_curve(name: str) -> Curve:
    """Build (and cache) the curve of a registered shape."""
    return shape_registry()[name].build()


def quantize_jitter(
    u: np.ndarray, w: np.ndarray, q: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize rank-space values to ``q`` equal-negative-mass bins and break
    ties uniformly at random (the ties red-team cell).

    Random tie-breaking makes the estimand exactly the trapezoid ROC of the
    quantized score (:func:`make_trapezoid` at the same ``q``).

    Args:
        u: Negative-class rank-space values in [0, 1).
        w: Positive-class rank-space values in [0, 1).
        q: Number of quantization levels.
        rng: Source of the tie-breaking jitter.

    Returns:
        Jittered ``(u, w)`` pairs, again in [0, 1).
    """
    bu = np.minimum((u * q).astype(int), q - 1)
    bw = np.minimum((w * q).astype(int), q - 1)
    return (bu + rng.random(len(bu))) / q, (bw + rng.random(len(bw))) / q
