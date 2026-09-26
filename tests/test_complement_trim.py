"""Checks for the complement-trim experiment (scripts/complement_trim)."""

import gzip
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.complement_trim import run as runner  # noqa: E402
from scripts.complement_trim.core import (  # noqa: E402
    build_level,
    complement_rows,
    enforce_depth,
    full_trim_rows,
)
from scripts.complement_trim.design import (  # noqa: E402
    Cell,
    PLCurve,
    all_cells,
    interior_jump_curve,
    interior_sliver_curve,
    sample_replicate,
)
from studroc_paper.methods.fiducial_band_rs import (
    _apply_corner_allowances,  # noqa: E402
)
from studroc_paper.methods.fiducial_ladder import khat_from_labels  # noqa: E402
from studroc_paper.methods.hybrid_floor import M3Floor, apply_m3_floor  # noqa: E402

pytest.importorskip("fiducial_core")


def _labels(n0: int, n1: int, shift: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scores = np.r_[rng.normal(0, 1, n0), rng.normal(shift, 1, n1)]
    labels = np.r_[np.zeros(n0, np.uint8), np.ones(n1, np.uint8)]
    return labels[np.argsort(-scores, kind="stable")]


def test_enforce_depth_takes_the_maximum() -> None:
    assert enforce_depth(j_full=5, j_raw=9) == (9, False)
    assert enforce_depth(j_full=5, j_raw=5) == (5, False)
    assert enforce_depth(j_full=5, j_raw=3) == (5, True)


def test_complement_rows_are_production_rows_outside_the_region() -> None:
    for n_grid in (101, 2001, 5001):
        region = np.zeros(n_grid, bool)
        region[:7] = True
        region[-40:] = True
        rows = complement_rows(n_grid, region)
        assert set(rows) <= set(full_trim_rows(n_grid))
        assert not region[rows].any()
        assert set(rows) | set(np.flatnonzero(region)) >= set(full_trim_rows(n_grid))


@pytest.mark.parametrize(("n0", "n1", "shift"), [(200, 200, 2.5), (300, 100, 1.5)])
def test_level_nesting_and_production_parity(n0: int, n1: int, shift: float) -> None:
    labels = _labels(n0, n1, shift, seed=n0 + n1)
    khat = khat_from_labels(lab_s=labels)
    level = build_level(labels=labels, khat=khat, level=0.05, seed=11, n_threads=1)
    diag = level["diag"]
    assert diag["j_comp"] >= diag["j_full"]
    # Deeper trim on the same cloud gives a pointwise-nested band.
    raw_lo, raw_up = level["raw"]
    assert level["region"].any()
    # The complement tube (pre-floor) sits inside the full tube; after the
    # common floor the complement hybrid sits inside the production hybrid.
    comp_lo, comp_up = level["comp"]
    hyb_lo, hyb_up = level["hybrid"]
    assert np.all(comp_lo >= hyb_lo - 1e-12)
    assert np.all(comp_up <= hyb_up + 1e-12)
    # The hybrid arm equals the production floor applied to the raw band.
    ref = apply_m3_floor(
        lab_s=labels,
        khat=khat,
        lower=raw_lo,
        upper=raw_up,
        alpha=0.05,
        n_draws=diag["n_draws"],
        trim_depth=diag["j_full"],
        floor=M3Floor(rule="exact", alpha=0.05),
    )
    np.testing.assert_array_equal(ref[0], hyb_lo)
    np.testing.assert_array_equal(ref[1], hyb_up)
    # Inside the region both floored bands contain M3.
    m3_lo, m3_up = level["m3"]
    region = level["region"]
    for lo, up in (level["hybrid"], level["comp"]):
        assert np.all(lo[region] <= m3_lo[region] + 1e-12)
        assert np.all(up[region] >= m3_up[region] - 1e-12)


def test_raw_band_matches_allowance_path() -> None:
    labels = _labels(150, 150, 2.0, seed=3)
    khat = khat_from_labels(lab_s=labels)
    level = build_level(labels=labels, khat=khat, level=0.05, seed=5, n_threads=1)
    lo, up = level["raw"]
    assert lo[0] == 0.0 and up[-1] == 1.0
    assert np.all(np.diff(up) >= 0)
    again = _apply_corner_allowances(
        lower=lo,
        upper=up,
        khat=khat,
        n1=150,
        trim_depth=level["diag"]["j_full"],
        n_draws=level["diag"]["n_draws"],
    )
    np.testing.assert_array_equal(again[1], up)


def test_fully_floored_grid_falls_back_to_the_hybrid() -> None:
    labels = _labels(10, 10, 1.0, seed=1)
    khat = khat_from_labels(lab_s=labels)
    level = build_level(labels=labels, khat=khat, level=0.05, seed=2, n_threads=1)
    assert level["diag"]["fallback"]
    for a, b in zip(level["comp"], level["hybrid"], strict=True):
        np.testing.assert_array_equal(a, b)


def test_interior_curves_are_exact_cdfs() -> None:
    sliver, (lo, hi) = interior_sliver_curve(n0=500, n1=500)
    jump, (jlo, jhi) = interior_jump_curve(n0=500)
    rng = np.random.default_rng(0)
    for curve in (sliver, jump):
        draws = curve.inv(rng.random(200_000))
        for t in (0.05, 0.3, 0.3 + 1 / 2000, 0.59, 0.6, 0.85, 0.95):
            assert abs(np.mean(draws <= t) - curve.eval(np.array([t]))[0]) < 5e-3
    mass = sliver.eval(np.array([hi]))[0] - sliver.eval(np.array([lo]))[0]
    assert mass == pytest.approx(0.8 / 500)
    step = jump.eval(np.array([jhi]))[0] - jump.eval(np.array([jlo]))[0]
    assert step == pytest.approx(0.2, abs=2e-3)
    with pytest.raises(ValueError):
        PLCurve(x=np.array([0.0, 0.5]), y=np.array([0.0, 1.0]))


def test_design_is_unique_and_sampling_is_deterministic() -> None:
    cells = all_cells()
    assert len(cells) == 50
    assert len({c.name for c in cells}) == 50
    cell = next(c for c in cells if c.block == "interior")
    a, seed_a, diag_a = sample_replicate(cell, 3)
    b, seed_b, diag_b = sample_replicate(cell, 3)
    np.testing.assert_array_equal(a, b)
    assert seed_a == seed_b and diag_a == diag_b
    c, seed_c, _ = sample_replicate(cell, 4)
    assert seed_c != seed_a


def test_runner_resumes_and_refuses_changed_definitions(tmp_path: Path) -> None:
    cell = Cell(
        name="ct-test-small",
        block="interior",
        n0=60,
        n1=60,
        interior={"kind": "interior_jump", "height": 0.2},
        reps=4,
        reps_max=4,
    )
    runner.run_cell(cell, tmp_path, workers=2, n_threads=1)
    path = tmp_path / "cells" / "ct-test-small.json.gz"
    with gzip.open(path, "rt") as handle:
        first = json.load(handle)["records"]
    assert [r["rep"] for r in first] == [0, 1, 2, 3]
    runner.run_cell(replace(cell, reps=6, reps_max=6), tmp_path, workers=2, n_threads=1)
    with gzip.open(path, "rt") as handle:
        second = json.load(handle)["records"]
    assert second[:4] == first and len(second) == 6
    with pytest.raises(RuntimeError):
        runner.run_cell(replace(cell, n1=61), tmp_path, workers=1, n_threads=1)
    summary = runner.summarize(tmp_path)
    level = summary["cells"][0]["levels"]["0.05"]
    assert level["hybrid"]["ratio_hybrid"] == pytest.approx(1.0)
    assert (tmp_path / "report.md").exists()
