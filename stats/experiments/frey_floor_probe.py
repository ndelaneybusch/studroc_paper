"""Probe: an exact floor optimized for the hybrid band's floor regions.

For one class-size design, builds class bounds that minimize the expected
width of the exact composition band on the grid points the hybrid floors,
under a mix of reference ROC curves, at the same exact joint coverage as the
ELL floor. It then swaps that floor into the production hybrid on the same
fiducial cloud and compares width and coverage with the default hybrid on
fresh datasets, including score families not used as references.

The floor regions and reference curves come from training draws that are
independent of the test datasets, so the optimized bounds are fixed before
any test ranks are seen.

Run with ``uv run python stats/experiments/frey_floor_probe.py --n 100 --prev 0.5``.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from frey_roc_probe import (  # noqa: E402
    ClassBand,
    coverage,
    ell_band,
    optimize_class_band,
)

from studroc_paper.datagen.roc_to_dgp import map_lhs_to_dgp  # noqa: E402
from studroc_paper.datagen.true_rocs import (  # noqa: E402
    DGP,
    make_beta_opposing_skew_dgp,
    make_bimodal_negative_dgp,
    make_heteroskedastic_gaussian_dgp,
    make_student_t_dgp,
    make_weibull_dgp,
)
from studroc_paper.methods import M3Floor, fiducial_band_suite_rs  # noqa: E402
from studroc_paper.methods.class_split import resolve_split_ratio  # noqa: E402

m3_module = importlib.import_module("studroc_paper.methods.m3_band_rs")

FloatArray = NDArray[np.float64]
ALPHA = 0.05
AUC_BOUNDS = (0.55, 0.99)
FAMILIES = {
    "binormal": (make_heteroskedastic_gaussian_dgp, {}),
    "hetero_gaussian": (make_heteroskedastic_gaussian_dgp, {"sigma_ratio": (0.2, 5.0)}),
    "student_t": (make_student_t_dgp, {"df": (1.1, 30.0)}),
    "weibull": (make_weibull_dgp, {"shape": (0.5, 5.0)}),
    "bimodal_negative": (
        make_bimodal_negative_dgp,
        {"mixture_weight": (0.1, 0.9), "mode_separation": (0.1, 4.0)},
    ),
    "beta_opposing": (make_beta_opposing_skew_dgp, {"alpha": (0.5, 10.0)}),
}
REFERENCE_FAMILIES = ("binormal", "hetero_gaussian", "student_t")
AUC_BANDS = ((0.55, 0.8), (0.8, 0.95), (0.95, 0.99))


def draw_dgp(*, family: str, rng: np.random.Generator) -> tuple[DGP, float] | None:
    """Draw one DGP of a family with AUC uniform on the probit scale.

    Args:
        family: Key of ``FAMILIES``.
        rng: Source of the parameter draw.

    Returns:
        The DGP and its design AUC, or ``None`` when the parameters are not
        achievable or the family has no analytic ROC.
    """
    make_dgp, shape_bounds = FAMILIES[family]
    z = rng.uniform(*stats.norm.ppf(AUC_BOUNDS))
    auc = float(stats.norm.cdf(z))
    params = {"auc": np.array([auc])} | {
        name: np.array([rng.uniform(lo, hi)]) for name, (lo, hi) in shape_bounds.items()
    }
    try:
        mapped = map_lhs_to_dgp(family, params)
        first = {
            k: v[0] if isinstance(v, list) else float(np.ravel(v)[0])
            for k, v in mapped.items()
        }
        dgp = make_dgp(**first)
    except (ValueError, KeyError, FloatingPointError):
        return None
    if dgp.true_roc is None:
        return None
    return dgp, auc


def rank_count_law(*, n0: int, n1: int, dgp: DGP, nodes: int = 3000) -> FloatArray:
    """Law of the number of positives above each negative order statistic."""
    x, w = np.polynomial.legendre.leggauss(nodes)
    u = 0.5 * (x + 1.0)
    tpr = np.clip(dgp.true_roc(u), 0.0, 1.0)
    pmf = stats.binom.pmf(np.arange(n1 + 1)[None, :], n1, tpr[:, None])
    i = np.arange(1, n0 + 1)
    density = stats.beta.pdf(u[None, :], i[:, None], (n0 + 1 - i)[:, None])
    law = (density * (0.5 * w)[None, :]) @ pmf
    return law / law.sum(axis=1, keepdims=True)


def labels_by_score(*, scores_pos: FloatArray, scores_neg: FloatArray) -> NDArray:
    """Class labels ordered by descending score."""
    scores = np.concatenate([scores_neg, scores_pos])
    labels = np.concatenate([np.zeros(len(scores_neg)), np.ones(len(scores_pos))])
    return labels[np.argsort(-scores, kind="stable")].astype(np.uint8)


def compose(
    *, labels: NDArray, neg: ClassBand, pos: ClassBand
) -> tuple[FloatArray, FloatArray]:
    """Native-grid composition band from class bounds and ordered labels."""
    n0 = neg.n
    grid = np.arange(n0 + 1) / n0
    iup = np.searchsorted(neg.lower, grid, side="left") + 1
    sentinel = iup > n0
    iup = np.clip(iup, 1, n0)
    ilo = np.minimum(np.searchsorted(neg.upper, grid, side="left") + 1, n0 + 1)
    counts = np.concatenate([[0], np.cumsum(labels)[labels == 0]]).astype(np.int64)
    upper = np.concatenate([[0.0], pos.upper, [1.0]])[counts[iup] + 1]
    upper[sentinel] = 1.0
    lower = np.concatenate([[0.0], pos.lower])[counts[ilo - 1]]
    lower[0], lower[-1], upper[-1] = 0.0, 1.0, 1.0
    return lower, upper


class Window:
    """A weight function on FPR given by its values on the native grid."""

    def __init__(self, *, weights: FloatArray) -> None:
        """Store the weights and their running integral.

        Args:
            weights: Weight at each native grid point ``k / n0``.
        """
        self.grid = np.linspace(0.0, 1.0, len(weights))
        self.weights = weights
        steps = 0.5 * (weights[1:] + weights[:-1]) * np.diff(self.grid)
        self.cumulative = np.concatenate([[0.0], np.cumsum(steps)])

    def density(self, t: FloatArray) -> FloatArray:
        """Weight at ``t``."""
        return np.interp(t, self.grid, self.weights)

    def mass(self, t: FloatArray) -> FloatArray:
        """Integral of the weight over ``[0, t]``."""
        return np.interp(t, self.grid, self.cumulative)


def windowed_area(
    *, neg: ClassBand, pos: ClassBand, law: FloatArray, window: Window
) -> float:
    """Expected weighted area of the composition band."""
    q_upper = law @ np.append(pos.upper, 1.0)
    q_lower = law @ np.insert(pos.lower, 0, 0.0)
    mass_lower = window.mass(np.insert(neg.lower, 0, 0.0))
    mass_upper = window.mass(np.append(neg.upper, 1.0))
    upper_area = np.diff(mass_lower) @ q_upper + window.mass(1.0) - mass_lower[-1]
    return float(upper_area - np.diff(mass_upper) @ q_lower)


def optimize_floor(
    *,
    n0: int,
    n1: int,
    targets: tuple[float, float],
    law: FloatArray,
    window: Window,
    sweeps: int = 6,
) -> tuple[ClassBand, ClassBand]:
    """Alternate class optimizations of the expected weighted area.

    The negative-class step linearizes the weight function at the current
    bounds, so the best iterate is kept instead of the last.
    """
    neg = ell_band(n=n0, alpha_c=1 - targets[0])
    pos = ell_band(n=n1, alpha_c=1 - targets[1])
    best = (windowed_area(neg=neg, pos=pos, law=law, window=window), neg, pos)
    for _ in range(sweeps):
        q_upper = np.append(law @ np.append(pos.upper, 1.0), 1.0)
        q_lower = np.insert(law @ np.insert(pos.lower, 0, 0.0), 0, 0.0)
        neg = optimize_class_band(
            start=neg,
            weight_lower=window.density(neg.lower) * np.diff(q_upper),
            weight_upper=window.density(neg.upper) * np.diff(q_lower),
            target=targets[0],
        )
        mass_lower = window.mass(np.insert(neg.lower, 0, 0.0))
        mass_upper = window.mass(np.append(neg.upper, 1.0))
        pos = optimize_class_band(
            start=pos,
            weight_lower=np.diff(mass_upper) @ law[:, 1:],
            weight_upper=np.diff(mass_lower) @ law[:, :-1],
            target=targets[1],
        )
        value = windowed_area(neg=neg, pos=pos, law=law, window=window)
        if value < best[0] - 1e-7:
            best = (value, neg, pos)
        else:
            break
    return best[1], best[2]


def hybrid_band(
    *, scores_pos: FloatArray, scores_neg: FloatArray, seed: int
) -> tuple[FloatArray, FloatArray, NDArray]:
    """Production hybrid band and its native floor mask."""
    y_true = np.concatenate([np.zeros(len(scores_neg)), np.ones(len(scores_pos))])
    suite = fiducial_band_suite_rs(
        y_true=y_true,
        y_score=np.concatenate([scores_neg, scores_pos]),
        alphas=[ALPHA],
        m3_floors={"hybrid": M3Floor()},
        random_state=seed,
    )
    lower, upper = suite.bands[ALPHA]["hybrid"]
    return lower, upper, np.asarray(suite.floor_regions[ALPHA]["hybrid"], dtype=bool)


def band_metrics(
    *, lower: FloatArray, upper: FloatArray, truth: FloatArray, region: NDArray
) -> dict[str, float]:
    """Area, floor-region width, and coverage of one native-grid band."""
    width = upper - lower
    grid = np.linspace(0.0, 1.0, len(width))
    inner = slice(1, -1)
    miss = (truth[inner] < lower[inner] - 1e-6) | (truth[inner] > upper[inner] + 1e-6)
    return {
        "area": float(np.trapezoid(width, grid)),
        "region_width": float(width[region].mean()) if region.any() else np.nan,
        "covers": float(not miss.any()),
    }


def main() -> None:
    """Run one design and write a JSON summary next to this script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--prev", type=float, required=True)
    parser.add_argument("--references", type=int, default=30)
    parser.add_argument("--tests-per-family", type=int, default=60)
    args = parser.parse_args()
    n1 = max(2, round(args.prev * args.n))
    n0 = args.n - n1
    rng = np.random.default_rng([20261121, args.n, round(100 * args.prev)])
    rho = resolve_split_ratio(split_ratio="sample_size", n0=n0, n1=n1)
    targets = ((1 - ALPHA) ** rho, (1 - ALPHA) ** (1 - rho))

    # Training: reference curves and the floor regions the hybrid selects.
    references = []
    while len(references) < args.references:
        family = REFERENCE_FAMILIES[len(references) % len(REFERENCE_FAMILIES)]
        drawn = draw_dgp(family=family, rng=rng)
        if drawn is None:
            continue
        dgp, auc = drawn
        masks = []
        for _ in range(4):
            scores_pos, scores_neg = dgp.sample(n1, n0, rng)
            masks.append(
                hybrid_band(
                    scores_pos=scores_pos,
                    scores_neg=scores_neg,
                    seed=int(rng.integers(2**31)),
                )[2]
            )
        references.append(
            {
                "auc": auc,
                "law": rank_count_law(n0=n0, n1=n1, dgp=dgp),
                "mask": np.mean(masks, axis=0),
            }
        )

    floors: dict[str, tuple[ClassBand, ClassBand]] = {
        "ell": (
            ell_band(n=n0, alpha_c=1 - targets[0]),
            ell_band(n=n1, alpha_c=1 - targets[1]),
        )
    }
    floor_info = {}
    for name, keep in (("opt_all", 0.0), ("opt_high", 0.8)):
        chosen = [r for r in references if r["auc"] >= keep]
        law = np.mean([r["law"] for r in chosen], axis=0)
        inclusion = np.mean([r["mask"] for r in chosen], axis=0)
        window = Window(weights=np.maximum(inclusion, 0.02))
        neg, pos = optimize_floor(n0=n0, n1=n1, targets=targets, law=law, window=window)
        floors[name] = (neg, pos)
        floor_info[name] = {
            "references": len(chosen),
            "mean_floored_fraction": float(inclusion.mean()),
            "joint_coverage": coverage(neg) * coverage(pos),
            "expected_window_area_ratio_to_ell": windowed_area(
                neg=neg, pos=pos, law=law, window=window
            )
            / windowed_area(
                neg=floors["ell"][0], pos=floors["ell"][1], law=law, window=window
            ),
        }
        print(name, floor_info[name], flush=True)

    # Test: fresh datasets, including families that supplied no references.
    original = m3_module._m3_band_from_labels_rs
    rows = []
    for family in FAMILIES:
        done = 0
        while done < args.tests_per_family:
            drawn = draw_dgp(family=family, rng=rng)
            if drawn is None:
                continue
            dgp, auc = drawn
            scores_pos, scores_neg = dgp.sample(n1, n0, rng)
            labels = labels_by_score(scores_pos=scores_pos, scores_neg=scores_neg)
            truth = dgp.get_true_roc(np.arange(n0 + 1) / n0)
            seed = int(rng.integers(2**31))
            lower, upper, region = hybrid_band(
                scores_pos=scores_pos, scores_neg=scores_neg, seed=seed
            )
            row = {
                "family": family,
                "auc": auc,
                "floored_fraction": float(region.mean()),
                "hybrid_ell": band_metrics(
                    lower=lower, upper=upper, truth=truth, region=region
                ),
            }
            for name, (neg, pos) in floors.items():
                exact_lower, exact_upper = compose(labels=labels, neg=neg, pos=pos)
                row[f"exact_{name}"] = band_metrics(
                    lower=exact_lower, upper=exact_upper, truth=truth, region=region
                )
                if name == "ell":
                    continue

                def patched(lab_s, *, neg=neg, pos=pos, **_):
                    lo, up = compose(labels=np.asarray(lab_s), neg=neg, pos=pos)
                    return np.arange(len(lo)) / (len(lo) - 1), lo, up

                m3_module._m3_band_from_labels_rs = patched
                try:
                    lower, upper, _ = hybrid_band(
                        scores_pos=scores_pos, scores_neg=scores_neg, seed=seed
                    )
                finally:
                    m3_module._m3_band_from_labels_rs = original
                row[f"hybrid_{name}"] = band_metrics(
                    lower=lower, upper=upper, truth=truth, region=region
                )
            rows.append(row)
            done += 1

    def summarize(subset: list[dict]) -> dict:
        out = {"datasets": len(subset)}
        if not subset:
            return out
        out["floored_fraction"] = float(
            np.mean([r["floored_fraction"] for r in subset])
        )
        for arm in ("hybrid_ell", "hybrid_opt_all", "hybrid_opt_high",
                    "exact_ell", "exact_opt_all", "exact_opt_high"):  # fmt: skip
            base = "hybrid_ell" if arm.startswith("hybrid") else "exact_ell"
            area = [np.log(r[arm]["area"] / r[base]["area"]) for r in subset]
            region = [
                np.log(r[arm]["region_width"] / r[base]["region_width"]) for r in subset
            ]
            out[arm] = {
                "coverage": float(np.mean([r[arm]["covers"] for r in subset])),
                "area_ratio": float(np.exp(np.mean(area))),
                "region_width_ratio": float(np.exp(np.nanmean(region))),
            }
        for arm in ("hybrid_opt_all", "hybrid_opt_high"):
            base = np.array([r["hybrid_ell"]["covers"] for r in subset], dtype=bool)
            new = np.array([r[arm]["covers"] for r in subset], dtype=bool)
            out[arm]["paired_coverage"] = {
                "both_cover": int((base & new).sum()),
                "only_default_covers": int((base & ~new).sum()),
                "only_optimized_covers": int((~base & new).sum()),
                "both_miss": int((~base & ~new).sum()),
            }
        return out

    summary = {
        "n": args.n, "prev": args.prev, "n0": n0, "n1": n1, "floors": floor_info,
        "all": summarize(rows),
        "by_auc": {
            f"{lo}-{hi}": summarize([r for r in rows if lo <= r["auc"] < hi])
            for lo, hi in AUC_BANDS
        },
        "held_out_families": summarize(
            [r for r in rows if r["family"] not in REFERENCE_FAMILIES]
        ),
    }  # fmt: skip
    path = Path(__file__).with_name(
        f"res_frey_floor_probe_n{args.n}_p{round(100 * args.prev)}.json"
    )
    summary["datasets"] = rows
    path.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["all"], indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
