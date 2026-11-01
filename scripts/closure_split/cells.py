"""Cells, truths, and seed streams for the closure/split experiment.

Study T compares floor closures (and the floor's M3 class split) on the
production fiducial path. Study S compares M3 class-split rules, alone, on a
denser grid of shapes and class-size ratios. See
``stats/experiments/closure_split_spec.md``.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "c_calibration"))

from shapes import (  # noqa: E402
    Curve,
    _lhs_curve,
    lhs_heldout_specs,
    make_bimodal_negative,
    make_binormal,
    make_hetero_gaussian,
    make_t_shape,
    make_trapezoid,
    quantize_jitter,
)
from stage_f_design import make_sliver_curve  # noqa: E402

from .curves import (  # noqa: E402
    diagonal_curve,
    hard_negative_curve,
    hard_positive_curve,
    interior_jump_curve,
    obvious_subgroup_curve,
    separated_top_curve,
)

STUDY_SEED = 20261027
TYPICAL_SEED = 20261028
ALPHAS_T = (0.05, 0.5)
ALPHAS_S = (0.05, 0.2, 0.5)

# Main-suite LHS bounds (simulation_spec.md section 2) for the six families
# that are distinct under rank invariance (dgp_redundancy.md): logit-normal
# is binormal, and every Weibull shape gives the same power ROC.
TYPICAL_FAMILIES: dict[str, dict[str, tuple[float, float]]] = {
    "binormal": {},
    "hetero_gaussian": {"sigma_ratio": (0.2, 5.0)},
    "beta_opposing": {"alpha": (0.5, 10.0)},
    "student_t": {"df": (1.1, 30.0)},
    "bimodal_negative": {"mixture_weight": (0.1, 0.9), "mode_separation": (0.1, 4.0)},
    "weibull": {},
}
AUC_BOUNDS = (0.55, 0.99)

# Main-suite class sizes at which the floor leaves an unfloored interior
# (at total n of 10 and 30 the hybrid is M3), with the typical-cell count.
TYPICAL_CONFIGS: tuple[tuple[int, int, int], ...] = (
    (50, 50, 6),
    (150, 150, 12),
    (500, 500, 12),
    (900, 100, 12),
    (5_000, 5_000, 4),
)


@dataclass(frozen=True)
class Cell:
    """One experiment cell.

    Attributes:
        name: Unique cell name; also keys the seed stream.
        study: ``"T"`` (closure, fiducial path) or ``"S"`` (M3 class split).
        block: Design block (``"typical"``, ``"favored"``, ``"red"``, or a
            Study S block).
        n0: Negative-class size.
        n1: Positive-class size.
        truth: JSON-native truth specification for :func:`build_truth`.
        quantize: Number of equal-negative-mass score levels, or ``None``.
        reps: Initial replicate count.
        reps_max: Replicate cap for the top-up rule.
        mechanism: Why the cell is in the design (red and favored cells).
    """

    name: str
    study: str
    block: str
    n0: int
    n1: int
    truth: dict
    quantize: int | None = None
    reps: int = 1_000
    reps_max: int = 3_000
    mechanism: str = ""
    meta: dict = field(default_factory=dict)

    def describe(self) -> dict:
        """Return a JSON-native description for manifests and summaries."""
        return {
            "name": self.name,
            "study": self.study,
            "block": self.block,
            "n0": self.n0,
            "n1": self.n1,
            "truth": self.truth,
            "quantize": self.quantize,
            "reps": self.reps,
            "reps_max": self.reps_max,
            "mechanism": self.mechanism,
            "meta": self.meta,
        }


def build_truth(spec: dict, *, n0: int, n1: int) -> Curve:
    """Build a rank-space ROC curve from its JSON-native specification.

    Args:
        spec: Truth specification with a ``"kind"`` key.
        n0: Negative size (for n-dependent constructions).
        n1: Positive size (for n-dependent constructions).

    Returns:
        The true ROC curve.

    Raises:
        ValueError: If the kind is unknown or its parameters are invalid.
    """
    kind = spec["kind"]
    if kind == "binormal":
        return make_binormal(spec["auc"])
    if kind == "hetero_gaussian":
        return make_hetero_gaussian(spec["auc"], sigma_ratio=spec["sigma_ratio"])
    if kind == "student_t":
        return make_t_shape(spec["auc"], df=spec["df"])
    if kind == "bimodal_negative":
        return make_bimodal_negative(
            spec["auc"], sep=spec["mode_separation"], weight=spec["mixture_weight"]
        )
    if kind == "lhs":
        return _lhs_curve(spec["family"], spec["auc"], dict(spec["params"]))
    if kind == "diagonal":
        return diagonal_curve()
    if kind == "sliver":
        return make_sliver_curve(
            auc=spec["auc"],
            n0=n0,
            n1=n1,
            expected_sliver_count=spec["expected_sliver_count"],
            tail_extent=spec["tail_extent"],
        )
    if kind == "interior_jump":
        return interior_jump_curve(n0=n0, height=spec["height"])
    body = build_truth(spec["body"], n0=n0, n1=n1) if "body" in spec else None
    if kind == "hard_positive":
        return hard_positive_curve(body=body, fraction=spec["fraction"])
    if kind == "hard_negative":
        return hard_negative_curve(body=body, fraction=spec["fraction"])
    if kind == "obvious_subgroup":
        return obvious_subgroup_curve(
            body=body, fraction=spec["fraction"], top_count=spec["top_count"], n0=n0
        )
    if kind == "separated_top":
        return separated_top_curve(body=body, fraction=spec["fraction"])
    raise ValueError(f"unknown truth kind {kind!r}")


_BASE_CACHE: dict[str, Curve] = {}
_ESTIMAND_CACHE: dict[str, Curve] = {}


def cell_base_curve(cell: Cell) -> Curve:
    """Return (and cache) the curve the positives are sampled from."""
    if cell.name not in _BASE_CACHE:
        _BASE_CACHE[cell.name] = build_truth(cell.truth, n0=cell.n0, n1=cell.n1)
    return _BASE_CACHE[cell.name]


def cell_truth(cell: Cell) -> Curve:
    """Return (and cache) the exact estimand of a cell.

    Quantized cells use the trapezoid ROC of the quantized score, which is
    the estimand under random tie-breaking.
    """
    if cell.name not in _ESTIMAND_CACHE:
        curve = cell_base_curve(cell)
        if cell.quantize is not None:
            curve = make_trapezoid(curve, q=cell.quantize)
        _ESTIMAND_CACHE[cell.name] = curve
    return _ESTIMAND_CACHE[cell.name]


def _lhs_spec(family: str, auc: float, params: dict[str, float]) -> dict:
    """Truth specification for one main-suite family draw."""
    if family == "binormal":
        return {"kind": "binormal", "auc": auc}
    if family == "weibull":
        return {
            "kind": "lhs",
            "family": "weibull",
            "auc": auc,
            "params": {"shape": 1.0},
        }
    return {"kind": "lhs", "family": family, "auc": auc, "params": params}


def _achievable_draw(
    *,
    rng: np.random.Generator,
    family: str,
    z_bounds: tuple[float, float],
    n0: int,
    n1: int,
) -> tuple[dict, float, int]:
    """Draw one family member until its AUC is achievable.

    Args:
        rng: Design generator (consumed in a fixed order).
        family: Main-suite family name.
        z_bounds: Probit-scale AUC bounds.
        n0: Negative size passed to the truth builder.
        n1: Positive size passed to the truth builder.

    Returns:
        The truth specification, its AUC, and the number of redraws.

    Raises:
        RuntimeError: If 50 draws are all unachievable.
    """
    for redraws in range(50):
        auc = float(norm.cdf(rng.uniform(*z_bounds)))
        params = {
            key: float(rng.uniform(*bounds))
            for key, bounds in sorted(TYPICAL_FAMILIES[family].items())
        }
        spec = _lhs_spec(family, auc, params)
        try:
            build_truth(spec, n0=n0, n1=n1)
        except ValueError:
            continue
        return spec, auc, redraws
    raise RuntimeError(f"no achievable draw for {family}")


def typical_cells() -> list[Cell]:
    """Frozen-seed draws from the main suite's design space.

    Each configuration takes the same number of draws from every family, and
    any remainder from families chosen without replacement. AUC is uniform
    on the probit scale over the suite's bounds and shape parameters are
    uniform within theirs, as in the suite's LHS. Draws whose AUC is
    unachievable for the family are redrawn, as the suite filters them.
    """
    rng = np.random.default_rng(TYPICAL_SEED)
    families = sorted(TYPICAL_FAMILIES)
    z_lo, z_hi = (norm.ppf(b) for b in AUC_BOUNDS)
    cells = []
    for n0, n1, count in TYPICAL_CONFIGS:
        whole, rest = divmod(count, len(families))
        picks = families * whole + [
            families[i] for i in sorted(rng.choice(len(families), rest, replace=False))
        ]
        for family in picks:
            spec, auc, redraws = _achievable_draw(
                rng=rng, family=family, z_bounds=(z_lo, z_hi), n0=n0, n1=n1
            )
            cells.append(
                Cell(
                    name=f"t-typ-{family}-a{auc:.3f}--n{n0}x{n1}",
                    study="T",
                    block="typical",
                    n0=n0,
                    n1=n1,
                    truth=spec,
                    meta={"redraws": redraws},
                )
            )
    return cells


def _t(auc: float, df: float) -> dict:
    return {"kind": "student_t", "auc": auc, "df": df}


def _bn(auc: float) -> dict:
    return {"kind": "binormal", "auc": auc}


def favored_cells() -> list[Cell]:
    """Cells where the widening closure is expected to cost the most.

    The closure's charge grows with the floor's extent relative to the grid:
    a long saturated run at moderate n0. Four fresh mechanism-defined shapes
    and two anchors that carried the largest probe gains.
    """
    specs = [
        ("bn99", _bn(0.99), 300, 300, "very long run, smooth concave truth"),
        (
            "hetero0.5-97",
            {"kind": "hetero_gaussian", "auc": 0.97, "sigma_ratio": 0.5},
            600,
            600,
            "narrow positives: long run at n = 600",
        ),
        (
            "beta-a0.7-98",
            {
                "kind": "lhs",
                "family": "beta_opposing",
                "auc": 0.98,
                "params": {"alpha": 0.7},
            },
            1_000,
            1_000,
            "U-shaped scores, long run at n = 1,000",
        ),
        (
            "bimodal-97",
            {
                "kind": "bimodal_negative",
                "auc": 0.97,
                "mode_separation": 3.5,
                "mixture_weight": 0.7,
            },
            400,
            400,
            "mid-curve plateau with a long run",
        ),
        ("anchor-wedge01-t2-99", _t(0.99, 2.0), 500, 500, "probe anchor (wedge-01)"),
        (
            "anchor-imb17-t2-98",
            _t(0.98, 2.0),
            1_500,
            300,
            "probe anchor (imbalance-17), scarce positives",
        ),
    ]
    return [
        Cell(
            name=f"t-fav-{tag}--n{n0}x{n1}",
            study="T",
            block="favored",
            n0=n0,
            n1=n1,
            truth=truth,
            mechanism=mechanism,
        )
        for tag, truth, n0, n1, mechanism in specs
    ]


def red_cells() -> list[Cell]:
    """Mechanism-targeted cells where the tightened closure may be weak.

    Tightening gives up exactly the room that widening spreads from the
    region into the adjacent unfloored interior. Each group places a real
    feature in that adjacent strip or stresses the region boundaries.
    """
    lhs = lhs_heldout_specs()
    specs: list[tuple[str, dict, int, int, int | None, str]] = [
        # R1: convex approach to (1, 1) that extends beyond run + margin.
        (
            "R1-hetero3-97",
            {"kind": "hetero_gaussian", "auc": 0.97, "sigma_ratio": 3.0},
            300,
            300,
            None,
            "upper-corner hook wider than the right floor",
        ),
        (
            "R1-hetero3-97",
            {"kind": "hetero_gaussian", "auc": 0.97, "sigma_ratio": 3.0},
            2_000,
            2_000,
            None,
            "upper-corner hook wider than the right floor",
        ),
        (
            "R1-t1.3-95",
            _t(0.95, 1.3),
            1_000,
            1_000,
            None,
            "near-Cauchy hooks at both corners",
        ),
        # R2: hard positives break the run; the body corner becomes an interior hook.
        (
            "R2-hardpos2-t2-99",
            {"kind": "hard_positive", "body": _t(0.99, 2.0), "fraction": 0.02},
            500,
            500,
            None,
            "hard positives: interior hook just outside a short right region",
        ),
        (
            "R2-hardpos1-t2-99",
            {"kind": "hard_positive", "body": _t(0.99, 2.0), "fraction": 0.01},
            2_000,
            2_000,
            None,
            "hard positives: interior hook just outside a short right region",
        ),
        (
            "R2-hardpos4-bn97",
            {"kind": "hard_positive", "body": _bn(0.97), "fraction": 0.04},
            300,
            300,
            None,
            "hard positives on a smooth body",
        ),
        # R3: hard negatives push the low-FPR hook past the left cut.
        (
            "R3-hardneg1-t2-99",
            {"kind": "hard_negative", "body": _t(0.99, 2.0), "fraction": 0.01},
            1_000,
            1_000,
            None,
            "hard negatives: low-FPR hook at about 10 grid points",
        ),
        (
            "R3-hardneg2-t2-99",
            {"kind": "hard_negative", "body": _t(0.99, 2.0), "fraction": 0.02},
            300,
            300,
            None,
            "hard negatives: low-FPR hook at about 6 grid points",
        ),
        # R4: obvious-positive subgroup straddling the left cut.
        (
            "R4-obvious-c12-bn75",
            {
                "kind": "obvious_subgroup",
                "body": _bn(0.75),
                "fraction": 0.5,
                "top_count": 12,
            },
            500,
            500,
            None,
            "near-vertical rise over the first 12 negatives",
        ),
        (
            "R4-obvious-c12-bn75",
            {
                "kind": "obvious_subgroup",
                "body": _bn(0.75),
                "fraction": 0.5,
                "top_count": 12,
            },
            2_000,
            500,
            None,
            "near-vertical rise over the first 12 negatives, scarce positives",
        ),
        # R5: separated top subgroup, R(0) > 0: the upper edge at the left end.
        (
            "R5-septop25-bn80",
            {"kind": "separated_top", "body": _bn(0.80), "fraction": 0.25},
            300,
            300,
            None,
            "R(0) = .25: upper edge at the left cut",
        ),
        (
            "R5-septop25-bn80",
            {"kind": "separated_top", "body": _bn(0.80), "fraction": 0.25},
            2_000,
            2_000,
            None,
            "R(0) = .25: upper edge at the left cut",
        ),
        # R6: scarce positives at scale (research record section 7.4).
        (
            "R6-scarce-t3-95",
            _t(0.95, 3.0),
            4_500,
            500,
            None,
            "negative majority: long run governed by n1",
        ),
        (
            "R6-scarce-bn95",
            _bn(0.95),
            9_000,
            1_000,
            None,
            "rare positives at the main suite's largest n",
        ),
        (
            "R6-mirror-t3-95",
            _t(0.95, 3.0),
            500,
            4_500,
            None,
            "positive-majority mirror of R6",
        ),
        # R7: interior support gap at small n, where the unfloored gap is short.
        (
            "R7-jump3",
            {"kind": "interior_jump", "height": 0.3},
            50,
            50,
            None,
            "interior jump in the short unfloored gap",
        ),
        (
            "R7-jump3",
            {"kind": "interior_jump", "height": 0.3},
            150,
            150,
            None,
            "interior jump in the short unfloored gap",
        ),
        # R8: ties from coarse scores.
        (
            "R8-ties-q10-bn90",
            _bn(0.90),
            500,
            500,
            10,
            "ten score levels, trapezoid estimand",
        ),
        # R9: large-n cells where the production hybrid sat near .955.
        (
            "R9-weibull-large13",
            {
                "kind": "lhs",
                "family": lhs[1]["family"],
                "auc": max(0.95, float(lhs[1]["auc"])),
                "params": dict(lhs[1]["params"]),
            },
            8_000,
            8_000,
            None,
            "complement study's lowest cell",
        ),
        (
            "R9-gamma-large03",
            {"kind": "lhs", "family": "gamma", "auc": 0.925, "params": {"shape": 0.5}},
            8_000,
            8_000,
            None,
            "complement study near-binding cell",
        ),
        # R10: a fresh corner sliver, the floor's core mechanism.
        (
            "R10-sliver-80",
            {
                "kind": "sliver",
                "auc": 0.80,
                "expected_sliver_count": 0.8,
                "tail_extent": 0.25,
            },
            1_000,
            1_000,
            None,
            "unsampled sliver at the upper corner, new n",
        ),
    ]
    return [
        Cell(
            name=f"t-red-{tag}--n{n0}x{n1}",
            study="T",
            block="red",
            n0=n0,
            n1=n1,
            truth=truth,
            quantize=quantize,
            mechanism=mechanism,
        )
        for tag, truth, n0, n1, quantize, mechanism in specs
    ]


def _replication_t(cell: Cell) -> Cell:
    """Base and maximum replicates by size: more cells, fewer reps at large n."""
    if cell.n0 >= 5_000:
        return replace(cell, reps=500, reps_max=1_500)
    if cell.n0 >= 2_000:
        return replace(cell, reps=750, reps_max=2_250)
    return replace(cell, reps=1_000, reps_max=3_000)


SPLIT_SHAPES: dict[str, dict] = {
    "diagonal": {"kind": "diagonal"},
    "bn65": _bn(0.65),
    "bn80": _bn(0.80),
    "bn95": _bn(0.95),
    "bn99": _bn(0.99),
    "hetero0.5-90": {"kind": "hetero_gaussian", "auc": 0.90, "sigma_ratio": 0.5},
    "hetero2-90": {"kind": "hetero_gaussian", "auc": 0.90, "sigma_ratio": 2.0},
    "t2-95": _t(0.95, 2.0),
    "beta-a0.7-98": {
        "kind": "lhs",
        "family": "beta_opposing",
        "auc": 0.98,
        "params": {"alpha": 0.7},
    },
    "bimodal-90": {
        "kind": "bimodal_negative",
        "auc": 0.90,
        "mode_separation": 3.0,
        "mixture_weight": 0.5,
    },
    "sliver-80": {
        "kind": "sliver",
        "auc": 0.80,
        "expected_sliver_count": 0.8,
        "tail_extent": 0.25,
    },
    "jump-80": {"kind": "interior_jump", "height": 0.2},
}

SPLIT_COUNTS: tuple[tuple[int, int], ...] = (
    (500, 500),
    (250, 750),
    (750, 250),
    (100, 900),
    (900, 100),
    (500, 4_500),
    (4_500, 500),
    (150, 4_500),
    (4_500, 150),
    (100, 10_000),
    (10_000, 100),
    (20, 180),
    (180, 20),
)


def split_cells() -> list[Cell]:
    """Study S: every split shape at every class-size configuration."""
    cells = []
    for tag, truth in SPLIT_SHAPES.items():
        for n0, n1 in SPLIT_COUNTS:
            block = (
                "balanced" if n0 == n1 else ("neg_major" if n0 > n1 else "pos_major")
            )
            cells.append(
                Cell(
                    name=f"s-{tag}--n{n0}x{n1}",
                    study="S",
                    block=block,
                    n0=n0,
                    n1=n1,
                    truth=truth,
                    reps=1_000,
                    reps_max=1_000,
                )
            )
    return cells


@cache
def all_cells() -> tuple[Cell, ...]:
    """Return the full design: Study T then Study S."""
    study_t = [
        _replication_t(cell) for cell in typical_cells() + favored_cells() + red_cells()
    ]
    cells = (*study_t, *split_cells())
    names = [cell.name for cell in cells]
    if len(set(names)) != len(names):
        raise RuntimeError("duplicate cell names")
    return cells


def select_cells(*, study: str | None, patterns: list[str] | None) -> list[Cell]:
    """Return cells of a study whose names contain any of the substrings."""
    cells = [c for c in all_cells() if study is None or c.study == study]
    if not patterns:
        return cells
    return [c for c in cells if any(p in c.name for p in patterns)]


def _stable_hash(text: str) -> int:
    """Map text to a stable unsigned 64-bit integer."""
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


@dataclass(frozen=True)
class Replicate:
    """One simulated rank order and its pre-outcome diagnostics.

    Attributes:
        labels: Tie-resolved 0/1 labels in descending score order.
        seed: 64-bit cloud seed for this replicate.
        diagnostics: Feature counts that do not depend on any band.
    """

    labels: np.ndarray
    seed: int
    diagnostics: dict


def sample_replicate(cell: Cell, rep: int) -> Replicate:
    """Draw one replicate of a cell in rank space.

    Args:
        cell: The cell.
        rep: Replicate index; with the cell name it keys the seed stream.

    Returns:
        Labels, the cloud seed, and feature diagnostics.
    """
    rng = np.random.default_rng(
        np.random.SeedSequence(entropy=(STUDY_SEED, _stable_hash(cell.name), rep))
    )
    curve = cell_base_curve(cell)
    negative = rng.random(cell.n0)
    positive = curve.inv(rng.random(cell.n1))
    diagnostics = feature_diagnostics(cell, negative=negative, positive=positive)
    if cell.quantize is not None:
        negative, positive = quantize_jitter(negative, positive, cell.quantize, rng)
    labels = np.r_[np.zeros(cell.n0, np.uint8), np.ones(cell.n1, np.uint8)]
    order = np.argsort(np.r_[negative, positive], kind="stable")
    seed = int(rng.integers(0, 2**64, dtype=np.uint64))
    return Replicate(labels=labels[order], seed=seed, diagnostics=diagnostics)


def feature_diagnostics(
    cell: Cell, *, negative: np.ndarray, positive: np.ndarray
) -> dict[str, int]:
    """Count sampled positives in a cell's feature, before any band is built.

    Args:
        cell: The cell.
        negative: Rank-space negative values.
        positive: Rank-space positive placements.

    Returns:
        ``{"feature_count": ...}`` for cells with a localized feature, else
        an empty dict.
    """
    kind = cell.truth["kind"]
    if kind == "sliver":
        return {"feature_count": int(np.count_nonzero(positive >= 1.0 - 1.0 / cell.n0))}
    if kind == "interior_jump":
        lo, hi = 0.3, 0.3 + 1.0 / (4.0 * cell.n0)
        return {
            "feature_count": int(np.count_nonzero((positive >= lo) & (positive <= hi)))
        }
    if kind == "separated_top":
        return {"feature_count": int(np.count_nonzero(positive == 0.0))}
    if kind == "obvious_subgroup":
        width = cell.truth["top_count"] / cell.n0
        return {"feature_count": int(np.count_nonzero(positive < width))}
    return {}


def true_roc_on_grid(cell: Cell) -> np.ndarray:
    """Return the estimand on the native grid ``k / n0``."""
    grid = np.arange(cell.n0 + 1) / cell.n0
    return np.clip(cell_truth(cell).eval(grid), 0.0, 1.0)
