"""Probe of Frey-type optimal class bands for the ROC composition band.

Checks the derivations in ``stats/frey_ell_plan.md`` numerically at small
class sizes:

1. The block-factorized gradient of the non-crossing probability against
   finite differences.
2. The closed-form expected area of a composition band under a reference ROC
   against Monte Carlo over rank sequences.
3. The uniform-weight minimum-width band against the values digitized from
   Frey's figures.
4. The linear objective for which the ELL band is optimal (its coverage
   gradient), compared with uniform and standardized width weights.
5. Bands that minimize expected ROC area under a reference ROC, cross-evaluated
   against the exact KS band, ELL, and the uniform-weight band.

Run with ``uv run python stats/experiments/frey_roc_probe.py``.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

from studroc_paper.methods.fiducial_band_rs import _require_fiducial_core

CORE = _require_fiducial_core()
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ClassBand:
    """Bounds ``lower[i] <= U_(i+1) <= upper[i]`` on uniform order statistics."""

    lower: FloatArray
    upper: FloatArray

    @property
    def n(self) -> int:
        """Class size."""
        return len(self.lower)


def crossing_prob(*, lower: FloatArray, upper: FloatArray) -> float:
    """Exact non-crossing probability, with the empty band counted as certain.

    Args:
        lower: Nondecreasing lower bounds in [0, 1].
        upper: Nondecreasing upper bounds in [0, 1].

    Returns:
        ``P(lower[i] <= U_(i+1) <= upper[i] for all i)``.
    """
    if len(lower) == 0:
        return 1.0
    lo = np.maximum.accumulate(np.clip(lower, 0.0, 1.0))
    hi = np.maximum.accumulate(np.clip(upper, 0.0, 1.0))
    return float(CORE.ell_crossing_probability(lo, hi))


def coverage(band: ClassBand) -> float:
    """Exact simultaneous coverage of a class band."""
    return crossing_prob(lower=band.lower, upper=band.upper)


def coverage_gradient(band: ClassBand) -> tuple[FloatArray, FloatArray]:
    """Gradient of the coverage in every bound, by block factorization.

    Conditional on ``U_(i) = c`` the smaller order statistics are those of
    ``i - 1`` uniforms on ``(0, c)`` and the larger ones those of ``n - i``
    uniforms on ``(c, 1)``, independently. The derivative in a bound placed at
    ``c`` is the Beta density of ``U_(i)`` at ``c`` times the two conditional
    non-crossing probabilities.

    Args:
        band: The class band.

    Returns:
        ``(d coverage / d lower, d coverage / d upper)``; the first is
        nonpositive and the second nonnegative.
    """
    n = band.n
    lower, upper = band.lower, band.upper
    d_lower = np.zeros(n)
    d_upper = np.zeros(n)
    for idx in range(n):
        rank = idx + 1
        for which, c in (("lower", lower[idx]), ("upper", upper[idx])):
            if not 0.0 < c < 1.0:
                density = float(stats.beta.pdf(c, rank, n + 1 - rank))
                if density == 0.0 or not np.isfinite(density):
                    continue
            density = float(stats.beta.pdf(c, rank, n + 1 - rank))
            left = 1.0
            if idx > 0:
                left = (
                    crossing_prob(
                        lower=lower[:idx] / c, upper=np.minimum(upper[:idx], c) / c
                    )
                    if c > 0.0
                    else 0.0
                )
            right = 1.0
            if idx < n - 1:
                right = (
                    crossing_prob(
                        lower=(np.maximum(lower[idx + 1 :], c) - c) / (1.0 - c),
                        upper=(upper[idx + 1 :] - c) / (1.0 - c),
                    )
                    if c < 1.0
                    else 0.0
                )
            value = density * left * right
            if which == "lower":
                d_lower[idx] = -value
            else:
                d_upper[idx] = value
    return d_lower, d_upper


def ell_band(*, n: int, alpha_c: float) -> ClassBand:
    """Equal-local-levels band calibrated to exact coverage ``1 - alpha_c``."""
    i = np.arange(1, n + 1, dtype=np.float64)

    def at(gamma: float) -> ClassBand:
        return ClassBand(
            lower=stats.beta.ppf(gamma, i, n + 1 - i),
            upper=stats.beta.ppf(1 - gamma, i, n + 1 - i),
        )

    lo, hi = alpha_c / (2 * n), alpha_c / 2
    for _ in range(60):
        mid = float(np.sqrt(lo * hi))
        if coverage(at(mid)) >= 1 - alpha_c:
            lo = mid
        else:
            hi = mid
    return at(lo)


def trimmed_ell_band(*, n: int, alpha_c: float, trim: int) -> ClassBand:
    """ELL band without the ``trim`` innermost bounds at each end, recalibrated.

    The lower bounds of the ``trim`` smallest order statistics are set to 0 and
    the upper bounds of the ``trim`` largest to 1. Those bounds carry almost no
    ROC area, and dropping them lets the common local level rise.
    """
    i = np.arange(1, n + 1, dtype=np.float64)

    def at(gamma: float) -> ClassBand:
        lower = stats.beta.ppf(gamma, i, n + 1 - i)
        upper = stats.beta.ppf(1 - gamma, i, n + 1 - i)
        lower[:trim] = 0.0
        upper[n - trim :] = 1.0
        return ClassBand(lower=lower, upper=upper)

    lo, hi = alpha_c / (2 * n), alpha_c / 2
    for _ in range(60):
        mid = float(np.sqrt(lo * hi))
        if coverage(at(mid)) >= 1 - alpha_c:
            lo = mid
        else:
            hi = mid
    return at(lo)


def asymmetric_ell_band(*, n: int, alpha_c: float, skew: float) -> ClassBand:
    """ELL band with different local levels below and above, calibrated exactly.

    The lower bounds use local level ``g * exp(skew)`` and the upper bounds
    ``g * exp(-skew)``; ``g`` is the largest value with coverage at least
    ``1 - alpha_c``. Positive ``skew`` tightens the lower bounds.
    """
    i = np.arange(1, n + 1, dtype=np.float64)

    def at(g: float) -> ClassBand:
        return ClassBand(
            lower=stats.beta.ppf(g * np.exp(skew), i, n + 1 - i),
            upper=stats.beta.ppf(1 - g * np.exp(-skew), i, n + 1 - i),
        )

    lo = alpha_c / (2 * n * np.cosh(skew))
    hi = 0.5 * np.exp(-abs(skew))
    for _ in range(50):
        mid = float(np.sqrt(lo * hi))
        if coverage(at(mid)) >= 1 - alpha_c:
            lo = mid
        else:
            hi = mid
    return at(lo)


def asymmetric_ell_pair(
    *, n0: int, n1: int, alpha: float, law: FloatArray
) -> tuple[ClassBand, ClassBand, dict[str, float]]:
    """Three-parameter ELL variant minimizing expected ROC area.

    The parameters are one skew per class and the class split of the level.
    """

    def build(x: FloatArray) -> tuple[ClassBand, ClassBand]:
        rho = 1.0 / (1.0 + np.exp(-x[2]))
        neg = asymmetric_ell_band(n=n0, alpha_c=1 - (1 - alpha) ** rho, skew=x[0])
        pos = asymmetric_ell_band(n=n1, alpha_c=1 - (1 - alpha) ** (1 - rho), skew=x[1])
        return neg, pos

    def objective(x: FloatArray) -> float:
        if np.abs(x).max() > 6:
            return 2.0
        neg, pos = build(x)
        return expected_area(neg=neg, pos=pos, law=law)

    result = optimize.minimize(
        fun=objective,
        x0=np.zeros(3),
        method="Nelder-Mead",
        options={"maxiter": 200, "xatol": 1e-2, "fatol": 1e-6},
    )
    neg, pos = build(result.x)
    params = {
        "skew_neg": float(result.x[0]),
        "skew_pos": float(result.x[1]),
        "rho": float(1.0 / (1.0 + np.exp(-result.x[2]))),
    }
    return neg, pos, params


def local_levels(band: ClassBand) -> tuple[FloatArray, FloatArray]:
    """Marginal tail probabilities cut off by each lower and upper bound."""
    i = np.arange(1, band.n + 1, dtype=np.float64)
    below = stats.beta.cdf(band.lower, i, band.n + 1 - i)
    above = stats.beta.sf(band.upper, i, band.n + 1 - i)
    return below, above


def ks_band(*, n: int, alpha_c: float) -> ClassBand:
    """Constant-margin band calibrated to exact coverage ``1 - alpha_c``."""
    i = np.arange(1, n + 1, dtype=np.float64)

    def at(margin: float) -> ClassBand:
        return ClassBand(
            lower=np.clip(i / n - margin, 0.0, 1.0),
            upper=np.clip((i - 1) / n + margin, 0.0, 1.0),
        )

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if coverage(at(mid)) >= 1 - alpha_c:
            hi = mid
        else:
            lo = mid
    return at(hi)


def restore_coverage(*, band: ClassBand, target: float) -> ClassBand:
    """Widen a band by the smallest common relaxation that reaches ``target``."""
    if coverage(band) >= target:
        return band

    def relaxed(eps: float) -> ClassBand:
        return ClassBand(
            lower=band.lower * (1 - eps), upper=1 - (1 - band.upper) * (1 - eps)
        )

    lo, hi = 0.0, 1e-6
    while coverage(relaxed(hi)) < target:
        hi *= 4
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if coverage(relaxed(mid)) >= target:
            hi = mid
        else:
            lo = mid
    return relaxed(hi)


def optimize_class_band(
    *,
    start: ClassBand,
    weight_lower: FloatArray,
    weight_upper: FloatArray,
    target: float,
) -> ClassBand:
    """Minimize ``weight_upper @ upper - weight_lower @ lower`` at exact coverage.

    The feasible set is convex (Brunn-Minkowski), so a local solution of this
    linear program over it is global.

    Args:
        start: Feasible starting band.
        weight_lower: Nonnegative weights on raising the lower bounds.
        weight_upper: Nonnegative weights on lowering the upper bounds.
        target: Required exact coverage.

    Returns:
        The optimized band, widened if needed so its coverage is at least
        ``target``.
    """
    n = start.n
    scale = max(float(weight_lower.sum() + weight_upper.sum()), 1e-300)
    w_lo, w_hi = weight_lower / scale, weight_upper / scale

    def split(x: FloatArray) -> ClassBand:
        return ClassBand(lower=x[:n], upper=x[n:])

    def log_cov(x: FloatArray) -> float:
        return float(np.log(max(coverage(split(x)), 1e-300)) - np.log(target))

    def log_cov_grad(x: FloatArray) -> FloatArray:
        band = split(x)
        d_lower, d_upper = coverage_gradient(band)
        return np.concatenate([d_lower, d_upper]) / max(coverage(band), 1e-300)

    difference = np.eye(n, k=1)[: n - 1] - np.eye(n)[: n - 1]
    zeros = np.zeros((n - 1, n))
    linear = np.vstack(
        [
            np.hstack([difference, zeros]),
            np.hstack([zeros, difference]),
            np.hstack([-np.eye(n), np.eye(n)]),
        ]
    )
    result = optimize.minimize(
        fun=lambda x: float(w_hi @ x[n:] - w_lo @ x[:n]),
        jac=lambda x: np.concatenate([-w_lo, w_hi]),
        x0=np.concatenate([start.lower, start.upper]),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * (2 * n),
        constraints=[
            {"type": "ineq", "fun": log_cov, "jac": log_cov_grad},
            {"type": "ineq", "fun": lambda x: linear @ x, "jac": lambda x: linear},
        ],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    x = np.clip(result.x, 0.0, 1.0)
    band = ClassBand(
        lower=np.maximum.accumulate(x[:n]), upper=np.maximum.accumulate(x[n:])
    )
    return restore_coverage(band=band, target=target)


def binormal_roc(*, auc: float):
    """Equal-variance binormal ROC and its inverse at the given AUC."""
    shift = np.sqrt(2.0) * stats.norm.ppf(auc)

    def roc(t: FloatArray) -> FloatArray:
        return stats.norm.cdf(shift + stats.norm.ppf(t))

    def inverse(v: FloatArray) -> FloatArray:
        return stats.norm.cdf(stats.norm.ppf(v) - shift)

    return roc, inverse


def rank_count_law(*, n0: int, n1: int, auc: float, nodes: int = 3000) -> FloatArray:
    """Law of the number of positives above each negative order statistic.

    Args:
        n0: Number of negatives.
        n1: Number of positives.
        auc: AUC of the binormal reference ROC.
        nodes: Gauss-Legendre nodes for the integral over the true FPR.

    Returns:
        Array ``M`` of shape ``(n0, n1 + 1)`` with ``M[i - 1, j] = P(p_i = j)``.
    """
    roc, _ = binormal_roc(auc=auc)
    x, w = np.polynomial.legendre.leggauss(nodes)
    u = 0.5 * (x + 1.0)
    w = 0.5 * w
    tpr = roc(u)
    counts = np.arange(n1 + 1)
    pmf = stats.binom.pmf(counts[None, :], n1, tpr[:, None])
    i = np.arange(1, n0 + 1)
    density = stats.beta.pdf(u[None, :], i[:, None], (n0 + 1 - i)[:, None])
    law = (density * w[None, :]) @ pmf
    return law / law.sum(axis=1, keepdims=True)


def expected_area(*, neg: ClassBand, pos: ClassBand, law: FloatArray) -> float:
    """Expected continuum area of the composition band under a rank-count law."""
    pos_upper = np.append(pos.upper, 1.0)
    pos_lower = np.insert(pos.lower, 0, 0.0)
    q_upper = law @ pos_upper
    q_lower = law @ pos_lower
    upper_area = np.diff(np.insert(neg.lower, 0, 0.0)) @ q_upper + 1.0 - neg.lower[-1]
    lower_area = np.diff(np.append(neg.upper, 1.0)) @ q_lower
    return float(upper_area - lower_area)


def area_weights_neg(
    *, pos: ClassBand, law: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Weights the expected area puts on the negative-class bounds."""
    q_upper = np.append(law @ np.append(pos.upper, 1.0), 1.0)
    q_lower = np.insert(law @ np.insert(pos.lower, 0, 0.0), 0, 0.0)
    return np.diff(q_upper), np.diff(q_lower)


def area_weights_pos(
    *, neg: ClassBand, law: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Weights the expected area puts on the positive-class bounds."""
    weight_upper = np.diff(np.insert(neg.lower, 0, 0.0)) @ law[:, :-1]
    weight_lower = np.diff(np.append(neg.upper, 1.0)) @ law[:, 1:]
    return weight_lower, weight_upper


def roc_optimal_pair(
    *, n0: int, n1: int, alpha: float, law: FloatArray, sweeps: int = 12
) -> tuple[ClassBand, ClassBand]:
    """Alternate exact class optimizations of the expected ROC area.

    Uses an equal split of the level, so each step is a convex program.
    """
    target = np.sqrt(1.0 - alpha)
    neg = ell_band(n=n0, alpha_c=1 - target)
    pos = ell_band(n=n1, alpha_c=1 - target)
    previous = expected_area(neg=neg, pos=pos, law=law)
    for _ in range(sweeps):
        w_lo, w_hi = area_weights_neg(pos=pos, law=law)
        neg = optimize_class_band(
            start=neg, weight_lower=w_lo, weight_upper=w_hi, target=target
        )
        w_lo, w_hi = area_weights_pos(neg=neg, law=law)
        pos = optimize_class_band(
            start=pos, weight_lower=w_lo, weight_upper=w_hi, target=target
        )
        current = expected_area(neg=neg, pos=pos, law=law)
        if previous - current < 1e-7:
            break
        previous = current
    return neg, pos


def sample_rank_counts(
    *, n0: int, n1: int, auc: float, draws: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    """Sample the vectors ``(p_1, ..., p_n0)`` under a binormal reference."""
    _, inverse = binormal_roc(auc=auc)
    fpr = np.sort(rng.uniform(size=(draws, n0)), axis=1)
    pos_fpr = np.sort(inverse(rng.uniform(size=(draws, n1))), axis=1)
    return np.stack(
        [np.searchsorted(pos_fpr[d], fpr[d], side="right") for d in range(draws)]
    )


def band_area(
    *, neg: ClassBand, pos: ClassBand, counts: NDArray[np.int64], limit: float
) -> FloatArray:
    """Continuum area of the composition band on ``[0, limit]`` per rank vector."""
    pos_upper = np.append(pos.upper, 1.0)
    pos_lower = np.insert(pos.lower, 0, 0.0)
    cut_lower = np.minimum(np.insert(neg.lower, 0, 0.0), limit)
    cut_upper = np.minimum(np.append(neg.upper, 1.0), limit)
    upper_area = pos_upper[counts] @ np.diff(cut_lower) + limit - cut_lower[-1]
    lower_area = pos_lower[counts] @ np.diff(cut_upper)
    return upper_area - lower_area


def check_gradient(*, n: int) -> float:
    """Largest gap between the factorized gradient and central differences."""
    band = ell_band(n=n, alpha_c=0.05)
    d_lower, d_upper = coverage_gradient(band)
    step = 1e-6
    worst = 0.0
    for idx in range(n):
        for name, analytic in (("lower", d_lower[idx]), ("upper", d_upper[idx])):
            plus = {"lower": band.lower.copy(), "upper": band.upper.copy()}
            minus = {"lower": band.lower.copy(), "upper": band.upper.copy()}
            plus[name][idx] += step
            minus[name][idx] -= step
            numeric = (crossing_prob(**plus) - crossing_prob(**minus)) / (2 * step)
            worst = max(worst, abs(numeric - analytic) / max(abs(analytic), 1e-12))
    return worst


FREY_N20_LOWER = np.array(
    [0, 0.00009, 0.00469, 0.01790, 0.03883, 0.06554, 0.09665, 0.13145, 0.16919,
     0.20935, 0.25178, 0.29610, 0.34222, 0.39024, 0.44000, 0.49204, 0.54636,
     0.60418, 0.66692, 0.73904]
)  # fmt: skip


def main() -> None:
    """Run the probe and write a JSON summary next to this script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=str, default="30x30,100x100,150x30")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--draws", type=int, default=4000)
    args = parser.parse_args()
    rng = np.random.default_rng(20261120)
    out: dict = {}

    out["gradient_max_rel_error_n25"] = check_gradient(n=25)
    print("1. gradient vs finite differences, max relative error:",
          f"{out['gradient_max_rel_error_n25']:.2e}")  # fmt: skip

    start = ks_band(n=20, alpha_c=0.05)
    frey20 = optimize_class_band(
        start=start, weight_lower=np.ones(20), weight_upper=np.ones(20), target=0.95
    )
    out["frey_n20"] = {
        "sum_width": float((frey20.upper - frey20.lower).sum()),
        "ks_sum_width": float((start.upper - start.lower).sum()),
        "max_abs_dev_from_digitized_lower": float(
            np.abs(frey20.lower - FREY_N20_LOWER).max()
        ),
        "coverage": coverage(frey20),
    }
    print("3. uniform-weight optimum, n=20:", out["frey_n20"])

    n = 100
    ell = ell_band(n=n, alpha_c=0.05)
    d_lower, d_upper = coverage_gradient(ell)
    i = np.arange(1, n + 1)
    sd = np.sqrt(i * (n + 1 - i) / ((n + 1) ** 2 * (n + 2)))
    spread = lambda v: float(v.max() / v.min())  # noqa: E731
    out["ell_implied_weights_n100"] = {
        "max_over_min_gradient_lower": spread(-d_lower),
        "max_over_min_gradient_times_sd_lower": spread(-d_lower * sd),
        "max_over_min_gradient_times_sd_upper": spread(d_upper * sd),
        "gradient_times_sd_lower_at_ranks_1_2_5_10_25_50_75_90_99_100": [
            float(-d_lower[k - 1] * sd[k - 1]) for k in (1, 2, 5, 10, 25, 50, 75, 90, 99, 100)
        ],
    }  # fmt: skip
    print(
        "4. ELL implied weights, n=100:",
        json.dumps(out["ell_implied_weights_n100"], indent=1),
    )

    aucs = (0.5, 0.8, 0.95)
    out["cells"] = {}
    for size in args.sizes.split(","):
        n0, n1 = (int(v) for v in size.split("x"))
        target = np.sqrt(1 - args.alpha)
        laws = {auc: rank_count_law(n0=n0, n1=n1, auc=auc) for auc in aucs}
        counts = {
            auc: sample_rank_counts(n0=n0, n1=n1, auc=auc, draws=args.draws, rng=rng)
            for auc in aucs
        }
        ones0, ones1 = np.ones(n0), np.ones(n1)
        bands: dict[str, tuple[ClassBand, ClassBand]] = {
            "ks_exact": (ks_band(n=n0, alpha_c=1 - target), ks_band(n=n1, alpha_c=1 - target)),
            "ell": (ell_band(n=n0, alpha_c=1 - target), ell_band(n=n1, alpha_c=1 - target)),
        }  # fmt: skip
        bands["frey_uniform"] = (
            optimize_class_band(start=bands["ks_exact"][0], weight_lower=ones0, weight_upper=ones0, target=target),
            optimize_class_band(start=bands["ks_exact"][1], weight_lower=ones1, weight_upper=ones1, target=target),
        )  # fmt: skip
        for trim in (1, 3):
            bands[f"ell_trim_{trim}"] = (
                trimmed_ell_band(n=n0, alpha_c=1 - target, trim=trim),
                trimmed_ell_band(n=n1, alpha_c=1 - target, trim=trim),
            )
        for auc in aucs:
            bands[f"roc_opt_{auc}"] = roc_optimal_pair(
                n0=n0, n1=n1, alpha=args.alpha, law=laws[auc]
            )
        mix_law = sum(laws.values()) / len(laws)
        bands["roc_opt_mix"] = roc_optimal_pair(
            n0=n0, n1=n1, alpha=args.alpha, law=mix_law
        )
        cell: dict = {}
        for label, law in (("0.8", laws[0.8]), ("mix", mix_law)):
            neg, pos, params = asymmetric_ell_pair(
                n0=n0, n1=n1, alpha=args.alpha, law=law
            )
            bands[f"asym_ell_{label}"] = (neg, pos)
            cell[f"asym_ell_{label}_params"] = params
            print(f"asymmetric ELL fitted to {label}: {params}")
        ranks0 = sorted(
            {1, 2, 3, 5, n0 // 4, n0 // 2, 3 * n0 // 4, n0 - 4, n0 - 2, n0 - 1, n0}
        )
        ranks1 = sorted(
            {1, 2, 3, 5, n1 // 4, n1 // 2, 3 * n1 // 4, n1 - 4, n1 - 2, n1 - 1, n1}
        )
        profile = {}
        for name in (
            "ell",
            "frey_uniform",
            "roc_opt_0.8",
            "roc_opt_0.95",
            "roc_opt_mix",
        ):
            neg, pos = bands[name]
            nb, na = local_levels(neg)
            pb, pa = local_levels(pos)
            profile[name] = {
                "neg_ranks": ranks0,
                "neg_level_below": [float(nb[r - 1]) for r in ranks0],
                "neg_level_above": [float(na[r - 1]) for r in ranks0],
                "pos_ranks": ranks1,
                "pos_level_below": [float(pb[r - 1]) for r in ranks1],
                "pos_level_above": [float(pa[r - 1]) for r in ranks1],
            }
        cell["local_level_profiles"] = profile
        print(
            f"\n5. n0={n0} n1={n1} alpha={args.alpha}: expected area | low-FPR area [0, 0.1]"
        )
        print(
            f"{'band':>14}"
            + "".join(f"  AUC {a:<4}        " for a in aucs)
            + " joint coverage"
        )
        for name, (neg, pos) in bands.items():
            row: dict = {}
            text = ""
            for auc in aucs:
                exact = expected_area(neg=neg, pos=pos, law=laws[auc])
                mc = float(
                    band_area(neg=neg, pos=pos, counts=counts[auc], limit=1.0).mean()
                )
                low = float(
                    band_area(neg=neg, pos=pos, counts=counts[auc], limit=0.1).mean()
                )
                row[str(auc)] = {"area": exact, "area_mc": mc, "low_fpr_area": low}
                text += f"  {exact:.4f} | {low:.4f}"
            row["joint_coverage"] = coverage(neg) * coverage(pos)
            cell[name] = row
            print(f"{name:>14}{text}   {row['joint_coverage']:.6f}")
        worst_mc = max(
            abs(cell[name][str(a)]["area"] - cell[name][str(a)]["area_mc"])
            for name in bands
            for a in aucs
        )
        cell["max_abs_exact_minus_mc_area"] = worst_mc
        print(
            f"2. closed-form expected area vs Monte Carlo, max abs difference: {worst_mc:.4f}"
        )
        neg, pos = bands["roc_opt_0.95"]
        cell["roc_opt_0.95_shape"] = {
            "neg_lower_first5": neg.lower[:5].tolist(),
            "neg_upper_last5": neg.upper[-5:].tolist(),
            "pos_lower_first5": pos.lower[:5].tolist(),
            "pos_upper_last5": pos.upper[-5:].tolist(),
            "ell_neg_lower_first5": bands["ell"][0].lower[:5].tolist(),
            "ell_pos_upper_last5": bands["ell"][1].upper[-5:].tolist(),
        }
        out["cells"][size] = cell

    path = Path(__file__).with_name("res_frey_roc_probe.json")
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
