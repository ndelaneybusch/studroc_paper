"""Checks for the closure/split experiment (scripts/closure_split)."""

import gzip
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.closure_split import analyze, core, split  # noqa: E402
from scripts.closure_split import run as runner  # noqa: E402
from scripts.closure_split.cells import (  # noqa: E402
    AUC_BOUNDS,
    TYPICAL_CONFIGS,
    TYPICAL_FAMILIES,
    all_cells,
    build_truth,
    cell_truth,
    sample_replicate,
    true_roc_on_grid,
    typical_cells,
)
from scripts.closure_split.curves import (  # noqa: E402
    hard_negative_curve,
    hard_positive_curve,
    obvious_subgroup_curve,
    separated_top_curve,
)
from studroc_paper.methods import M3Floor, fiducial_band_rs  # noqa: E402
from studroc_paper.methods.fiducial_band import (  # noqa: E402
    _auto_n_draws,
    _merged_labels,
)
from studroc_paper.methods.fiducial_ladder import khat_from_labels  # noqa: E402

fiducial_core = pytest.importorskip("fiducial_core")
m3_module = importlib.import_module("studroc_paper.methods.m3_band_rs")


def _cell(fragment: str):
    return next(c for c in all_cells() if fragment in c.name)


def _empirical_roc(neg: np.ndarray, pos: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Exact population-style ROC estimate at the requested FPR values."""
    thresholds = np.quantile(neg, 1.0 - grid)
    return np.array([np.mean(pos > c) for c in thresholds])


# ---------------------------------------------------------------------------
# truth constructions
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def binormal_body():
    return build_truth({"kind": "binormal", "auc": 0.9}, n0=100, n1=100)


def test_hard_negative_curve_matches_score_space_contamination(binormal_body):
    """The parametric curve is the ROC of a negative mixture in score space."""
    rng = np.random.default_rng(1)
    mu = np.sqrt(2.0) * norm.ppf(0.9)
    fraction, n = 0.05, 400_000
    clean = rng.normal(0, 1, n)
    hard = rng.normal(mu, 1, n)
    neg = np.where(rng.random(n) < fraction, hard, clean)
    pos = rng.normal(mu, 1, n)
    grid = np.linspace(0.005, 0.995, 60)
    curve = hard_negative_curve(body=binormal_body, fraction=fraction)
    assert np.max(np.abs(curve.eval(grid) - _empirical_roc(neg, pos, grid))) < 0.006


def test_hard_positive_curve_matches_score_space_contamination(binormal_body):
    rng = np.random.default_rng(2)
    mu = np.sqrt(2.0) * norm.ppf(0.9)
    fraction, n = 0.04, 400_000
    neg = rng.normal(0, 1, n)
    pos = np.where(rng.random(n) < fraction, rng.normal(0, 1, n), rng.normal(mu, 1, n))
    grid = np.linspace(0.005, 0.995, 60)
    curve = hard_positive_curve(body=binormal_body, fraction=fraction)
    assert np.max(np.abs(curve.eval(grid) - _empirical_roc(neg, pos, grid))) < 0.006


def test_separated_top_curve_puts_an_atom_above_every_negative(binormal_body):
    """R(0) equals the subgroup share, and rank-space sampling reproduces it."""
    curve = separated_top_curve(body=binormal_body, fraction=0.25)
    assert curve.eval(np.array([0.0]))[0] == pytest.approx(0.25)
    rng = np.random.default_rng(3)
    placements = curve.inv(rng.random(200_000))
    assert np.mean(placements == 0.0) == pytest.approx(0.25, abs=0.004)
    grid = np.linspace(0.01, 0.99, 50)
    ecdf = np.searchsorted(np.sort(placements), grid, side="right") / len(placements)
    assert np.max(np.abs(ecdf - curve.eval(grid))) < 0.005


def test_obvious_subgroup_occupies_exactly_the_top_negatives(binormal_body):
    n0, top = 500, 12
    curve = obvious_subgroup_curve(
        body=binormal_body, fraction=0.5, top_count=top, n0=n0
    )
    width = top / n0
    edge = np.array([0.0, width / 2, width, 0.5])
    body = binormal_body.eval(edge)
    expected = 0.5 * np.minimum(1.0, edge / width) + 0.5 * body
    expected[0] = 0.0
    np.testing.assert_allclose(curve.eval(edge), expected, atol=2e-6)


@pytest.mark.parametrize(
    "fragment",
    [
        "R2-hardpos2",
        "R3-hardneg1",
        "R4-obvious-c12-bn75--n500",
        "R5-septop25-bn80--n300",
        "R7-jump3--n50",
        "R10-sliver",
        "R8-ties",
    ],
)
def test_red_cell_truths_are_valid_rocs_with_their_declared_auc_scale(fragment):
    cell = _cell(fragment)
    truth = true_roc_on_grid(cell)
    assert truth.shape == (cell.n0 + 1,)
    assert np.all(np.diff(truth) >= -1e-12) and truth[-1] == pytest.approx(1.0)
    assert 0.5 < cell_truth(cell).auc() < 1.0


def test_quantized_cell_scores_against_the_trapezoid_estimand():
    cell = _cell("R8-ties")
    truth = cell_truth(cell)
    knots = np.arange(cell.quantize + 1) / cell.quantize
    mids = 0.5 * (knots[:-1] + knots[1:])
    np.testing.assert_allclose(
        truth.eval(mids),
        0.5 * (truth.eval(knots[:-1]) + truth.eval(knots[1:])),
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# design and seeds
# ---------------------------------------------------------------------------


def test_typical_block_is_a_frozen_stratified_draw_from_the_suite_space():
    first, second = typical_cells(), typical_cells()
    assert [c.name for c in first] == [c.name for c in second]
    for n0, n1, count in TYPICAL_CONFIGS:
        members = [c for c in first if (c.n0, c.n1) == (n0, n1)]
        assert len(members) == count
        families = [c.name.split("-")[2] for c in members]
        whole, rest = divmod(count, len(TYPICAL_FAMILIES))
        tallies = {f: families.count(f) for f in TYPICAL_FAMILIES}
        assert sorted(tallies.values()) == sorted(
            [whole] * (len(TYPICAL_FAMILIES) - rest) + [whole + 1] * rest
        )
    aucs = [c.truth["auc"] for c in first]
    assert min(aucs) >= AUC_BOUNDS[0] and max(aucs) <= AUC_BOUNDS[1]


def test_design_has_unique_names_and_the_declared_blocks():
    cells = all_cells()
    assert len({c.name for c in cells}) == len(cells)
    study_t = [c for c in cells if c.study == "T"]
    assert {c.block for c in study_t} == {"typical", "favored", "red"}
    assert all(c.reps <= c.reps_max for c in cells)
    assert all(c.reps == 500 for c in study_t if c.n0 >= 5_000)


def test_replicates_are_reproducible_and_distinct():
    cell = _cell("R5-septop25-bn80--n300")
    a, b = sample_replicate(cell, 3), sample_replicate(cell, 3)
    assert np.array_equal(a.labels, b.labels) and a.seed == b.seed
    c = sample_replicate(cell, 4)
    assert not np.array_equal(a.labels, c.labels) and a.seed != c.seed
    assert a.labels.sum() == cell.n1 and len(a.labels) == cell.n0 + cell.n1
    # Separated-top positives rank above every negative.
    count = a.diagnostics["feature_count"]
    assert count > 0 and np.all(a.labels[:count] == 1)


# ---------------------------------------------------------------------------
# Study T construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.05, 0.5])
def test_core_arms_equal_the_production_entry_point(alpha):
    """Every fiducial arm is bit-for-bit what fiducial_band_rs returns."""
    rng = np.random.default_rng(4)
    n0, n1 = 300, 120
    y = np.r_[np.zeros(n0, int), np.ones(n1, int)]
    scores = np.r_[rng.standard_t(2, n0), rng.standard_t(2, n1) + 4.5]
    generator = np.random.default_rng(9)
    labels = _merged_labels(y, scores, "random", generator)
    seed = int(generator.integers(0, 2**64, dtype=np.uint64))
    built = core.build_alpha(
        labels=labels,
        khat=khat_from_labels(lab_s=labels),
        alpha=alpha,
        seed=seed,
        n_threads=1,
    )
    floors = {
        "hyb_W": M3Floor(split_ratio=0.5, closure="widen", recalibrate_trim=False),
        "hyb_T": M3Floor(split_ratio=0.5, closure="tighten", recalibrate_trim=False),
        "rec_W": M3Floor(split_ratio=0.5, closure="widen", recalibrate_trim=True),
        "rec_T": M3Floor(split_ratio=0.5, closure="tighten", recalibrate_trim=True),
        "rec_T_ss": M3Floor(),
    }
    for arm, floor in floors.items():
        _, lo, up = fiducial_band_rs(
            y,
            scores,
            alpha=alpha,
            random_state=9,
            n_threads=1,
            n_draws=_auto_n_draws(n0 + 1, alpha),
            m3_floor=floor,
        )
        assert np.array_equal(lo, built["bands"][arm][0]), arm
        assert np.array_equal(up, built["bands"][arm][1]), arm


@pytest.mark.parametrize(
    "fragment",
    [
        "t-typ-binormal-a0.603--n50x50",
        "R3-hardneg2",
        "R5-septop25-bn80--n300",
        "R7-jump3--n150",
        "fav-bn99",
        "t-typ-student_t-a0.966--n900x100",
    ],
)
def test_exact_properties_hold_on_real_replicates(fragment):
    """Nestings and the tightened/unclosed coverage identity, on design cells."""
    cell = _cell(fragment)
    truth = true_roc_on_grid(cell)
    for rep in range(2):
        sample = sample_replicate(cell, 90_000 + rep)
        result = core.run_replicate(
            labels=sample.labels,
            seed=sample.seed,
            truth=truth,
            alphas=(0.05, 0.5),
            n_threads=1,
        )
        for level in result["levels"].values():
            checks = level["checks"]
            for key, value in checks.items():
                if key.startswith("nest_"):
                    assert value <= core.TOL, (key, value)
                else:
                    assert value is True, key
            diag = level["diag"]
            assert diag["j_rec"] >= diag["j"]
            scores = level["scores"]
            # Domination of the raw band transfers to coverage.
            assert scores["hyb_T"]["covered"] >= scores["raw"]["covered"]
            assert scores["rec_W"]["covered"] >= scores["rec_T"]["covered"]


def test_location_masks_partition_the_grid():
    region = np.zeros(301, dtype=bool)
    region[:9] = True
    region[250:] = True
    masks = core.location_masks(region)
    stacked = np.vstack(list(masks.values())).astype(int)
    assert np.all(stacked.sum(axis=0) == 1)
    assert masks["near"].sum() == 2 * core.near_width(300)
    assert np.array_equal(masks["left"] | masks["right"], region)
    whole = core.location_masks(np.ones(11, dtype=bool))
    assert whole["left"].all() and not whole["near"].any()


# ---------------------------------------------------------------------------
# Study S construction
# ---------------------------------------------------------------------------


def test_split_rules_are_the_declared_family():
    arms = split.split_arms(100, 900)
    assert arms["sidak"] == 0.5
    assert arms["sqrt"] == pytest.approx(0.75)
    assert arms["linear"] == pytest.approx(0.9)
    assert sorted(v for k, v in arms.items() if k.startswith("rho")) == list(
        split.SWEEP
    )
    balanced = split.split_arms(500, 500)
    assert balanced["sqrt"] == balanced["linear"] == 0.5


def test_precomputed_calibrations_cover_every_request(tmp_path, monkeypatch):
    """A replicate run after loading the cache never triggers a calibration."""
    cells = [_cell("s-bn80--n20x180"), _cell("t-typ-binormal-a0.603--n50x50")]
    path = tmp_path / "gammas.json"
    split.precompute_gammas(cells, path, workers=2)
    m3_module._GAMMA_CACHE.clear()
    m3_module._ell_bounds.cache_clear()
    split.load_gammas(path)

    def forbidden(*args, **kwargs):
        raise AssertionError("calibration requested outside the precomputed cache")

    monkeypatch.setattr(m3_module, "_ell_coverage", forbidden)
    for cell in cells:
        runner.replicate_record(cell, 0, n_threads=1)


# ---------------------------------------------------------------------------
# runner and analysis
# ---------------------------------------------------------------------------


def _fake_records(covered: list[bool]) -> list[dict]:
    return [
        {"levels": {"0.05": {"scores": {"rec_T": {"covered": c}}}}} for c in covered
    ]


def test_topup_triggers_only_below_the_bar_and_below_the_cap():
    cell = replace(_cell("fav-bn99"), reps=10, reps_max=20)
    assert runner.needs_topup(cell, _fake_records([True] * 9 + [False]))
    assert not runner.needs_topup(cell, _fake_records([True] * 10))
    assert not runner.needs_topup(cell, _fake_records([False] * 20))
    s_cell = _cell("s-bn80--n20x180")
    assert not runner.needs_topup(s_cell, _fake_records([False] * 5))


def test_records_from_a_changed_cell_definition_are_refused(tmp_path):
    cell = _cell("s-bn80--n20x180")
    meta = runner._meta(cell)
    path = tmp_path / "cell.json.gz"
    runner._save(path, meta, [{"rep": 0}, {"rep": 1}])
    assert len(runner.load_records(path, meta)) == 2
    with pytest.raises(RuntimeError):
        runner.load_records(path, {**meta, "n0": cell.n0 + 1})
    runner._save(path, meta, [{"rep": 0}, {"rep": 2}])
    with pytest.raises(RuntimeError):
        runner.load_records(path, meta)


def test_end_to_end_run_resumes_and_writes_the_report(tmp_path):
    cells = [
        replace(_cell("t-typ-binormal-a0.603--n50x50"), reps=3, reps_max=3),
        replace(_cell("s-bn80--n20x180"), reps=4, reps_max=4),
    ]
    split.precompute_gammas(cells, tmp_path / runner.GAMMAS, workers=2)
    runner.run(cells, tmp_path, workers=2, n_threads=1)
    paths = sorted((tmp_path / "cells").glob("*.json.gz"))
    assert len(paths) == 2
    with gzip.open(paths[0], "rt") as handle:
        first = json.load(handle)
    runner.run(cells, tmp_path, workers=2, n_threads=1)
    with gzip.open(paths[0], "rt") as handle:
        assert json.load(handle) == first
    summary = analyze.summarize(tmp_path)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    for heading in ("Q1.", "Q2.", "Q5.", "Q8.", "Q11."):
        assert heading in report
    assert "pass" not in report.lower()
    t_cells = pd.read_csv(tmp_path / "t_cells.csv")
    assert set(t_cells["alpha"]) == {0.05, 0.5}
    assert (t_cells["max_nest_violation"] <= core.TOL).all()
    assert set(summary) >= {"closure", "flips", "split_sizes"}
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert len(saved["closure"]) == len(summary["closure"])


def _flip_frame() -> pd.DataFrame:
    """Ten replicates: the widened band covers all; tightening loses three.

    Two of the lost replicates miss in the adjacent strip on the lower edge,
    one misses far from the region on the upper edge.
    """
    rows = []
    for rep in range(10):
        lost = rep < 3
        row = {"alpha": 0.05, "block": "red"}
        for arm in ARMS_FOR_TEST:
            row[f"{arm}.covered"] = True
            for key in ("miss_in", "miss_near", "miss_far", "below", "above"):
                row[f"{arm}.{key}"] = False
            row[f"{arm}.depth"] = 0.0
        if lost:
            row["rec_T.covered"] = False
            row["rec_T.depth"] = 0.01 * (rep + 1)
            if rep < 2:
                row["rec_T.miss_near"] = row["rec_T.below"] = True
            else:
                row["rec_T.miss_far"] = row["rec_T.above"] = True
        rows.append(row)
    return pd.DataFrame(rows)


ARMS_FOR_TEST = ("rec_T", "rec_W", "hyb_T", "hyb_W")


def test_flip_locations_count_the_added_misses_and_where_they_land():
    table = analyze.flip_locations(_flip_frame())
    rec = table[table["contrast"] == "rec_T vs rec_W"].iloc[0]
    assert rec["flips"] == 3 and rec["of reps"] == 10
    assert rec["adjacent strip"] == pytest.approx(2 / 3)
    assert rec["elsewhere"] == pytest.approx(1 / 3)
    assert rec["in region"] == 0.0
    assert rec["lower edge"] == pytest.approx(2 / 3)
    assert rec["upper edge"] == pytest.approx(1 / 3)
    assert rec["median depth"] == pytest.approx(0.02)
    hyb = table[table["contrast"] == "hyb_T vs hyb_W"].iloc[0]
    assert hyb["flips"] == 0
