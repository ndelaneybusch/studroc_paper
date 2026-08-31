"""Focused tests for scripts/c_calibration/followup_runs.py.

Covers the decision-changing mechanics flagged in review: the stitched
band's monotone closure, the Wilson-interval classification rule, the
sentinel config restriction, and the refuse-to-mix reuse validation.
No kernel calls — everything here is pure Python.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/c_calibration"))

from followup_runs import (  # noqa: E402
    BAR_POINT,
    CORNER_EXPONENT,
    LHS_AUC_BOUNDS,
    LHS_DF_BOUNDS,
    LHS_N_BOUNDS,
    LHS_N_CELLS,
    SENTINEL_INTERIOR_C,
    _composite_constants,
    _load_composite,
    _stitch,
    boundary_lhs_points,
    classify,
    composite_cells,
    composite_configs,
    fit_boundary_surface,
    needs_topup,
    replay_empirical_aucs,
    surface_n_star,
    surface_predict,
    wilson_ci,
)


class TestStitch:
    def _reviewer_probe(self):
        """The review's synthetic case: wide lower re-entering at b_hi used
        to produce a decreasing lower edge [0, .2, .4, .3, 1]."""
        grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        lo_wide = np.array([0.0, 0.1, 0.2, 0.3, 1.0])
        hi_wide = np.array([0.5, 0.6, 0.9, 0.95, 1.0])
        lo_int = np.array([0.0, 0.2, 0.4, 0.5, 1.0])
        hi_int = np.array([0.3, 0.5, 0.7, 0.8, 1.0])
        return grid, lo_wide, hi_wide, lo_int, hi_int

    def test_monotone_edges_and_invariants(self):
        grid, lo_wide, hi_wide, lo_int, hi_int = self._reviewer_probe()
        lo, hi = _stitch(lo_wide, hi_wide, lo_int, hi_int, grid, 0.1, 0.7)
        assert np.all(np.diff(lo) >= 0)
        assert np.all(np.diff(hi) >= 0)
        assert np.all(lo <= hi)
        assert lo[0] == 0.0 and hi[-1] == 1.0

    def test_lower_seam_repaired_by_tightening(self):
        # At b_hi the raw stitch would drop from the interior lower (.5 at
        # t=.75 is interior? no: t=.75 >= b_hi=.7 -> wide .3): the closure
        # must carry the interior maximum forward, not fall back to .3.
        grid, lo_wide, hi_wide, lo_int, hi_int = self._reviewer_probe()
        lo, _ = _stitch(lo_wide, hi_wide, lo_int, hi_int, grid, 0.1, 0.7)
        assert lo[3] >= lo[2]

    def test_coverage_event_unchanged_by_lower_closure(self):
        # The running-max lower is a valid tightening: any truth covered by
        # the raw stitched band is covered by the closed band and vice versa.
        grid, lo_wide, hi_wide, lo_int, hi_int = self._reviewer_probe()
        b_lo, b_hi = 0.1, 0.7
        corner = (grid <= b_lo) | (grid >= b_hi)
        raw_lo = np.where(corner, lo_wide, lo_int)
        raw_hi = np.where(corner, hi_wide, hi_int)
        lo, hi = _stitch(lo_wide, hi_wide, lo_int, hi_int, grid, b_lo, b_hi)
        rng = np.random.default_rng(0)
        for _ in range(200):
            # random monotone truth
            r = np.sort(rng.random(len(grid)))
            r[0], r[-1] = 0.0, 1.0
            hi_closed_raw = np.maximum.accumulate(raw_hi)
            raw_ok = bool(np.all(raw_lo <= r) and np.all(r <= hi_closed_raw))
            closed_ok = bool(np.all(lo <= r) and np.all(r <= hi))
            assert raw_ok == closed_ok

    def test_all_interior_reduces_to_interior_band(self):
        grid, lo_wide, hi_wide, lo_int, hi_int = self._reviewer_probe()
        lo, hi = _stitch(lo_wide, hi_wide, lo_int, hi_int, grid, -1.0, 2.0)
        assert np.array_equal(lo, np.maximum.accumulate(lo_int))
        assert np.array_equal(hi, np.maximum.accumulate(hi_int))


class TestClassification:
    def test_wilson_basic(self):
        lo, hi = wilson_ci(0.95, 1000)
        assert 0.93 < lo < 0.95 < hi < 0.97

    def test_wilson_edge_cases(self):
        assert wilson_ci(1.0, 100)[1] == 1.0
        assert wilson_ci(0.0, 100)[0] == 0.0
        assert wilson_ci(0.5, 0) == (0.0, 1.0)

    def test_classify_pass(self):
        # High coverage with tight CI clears the A1-letter bar.
        assert classify(0.96, 2000) == "PASS"

    def test_classify_fail(self):
        assert classify(0.80, 1000) == "FAIL"

    def test_classify_marginal_straddles_bar(self):
        # Point just under the bar with a CI containing it.
        p = BAR_POINT - 0.002
        lo, hi = wilson_ci(p, 400)
        assert lo < BAR_POINT <= hi
        assert classify(p, 400) == "MARGINAL"
        assert needs_topup(p, 400)

    def test_no_topup_when_decisive(self):
        assert not needs_topup(0.99, 2000)
        assert not needs_topup(0.80, 2000)


class TestCompositeDesign:
    def test_sentinel_configs_are_restricted(self):
        cells = {c.name: c for c in composite_cells()}
        sentinel = cells["composite--t2_95--n20000x20000"]
        regular = cells["composite--t2_95--n500x500"]
        s_cfgs = composite_configs(sentinel)
        r_cfgs = composite_configs(regular)
        assert {c for _, _, c in s_cfgs} == {1.0, *SENTINEL_INTERIOR_C}
        assert len(s_cfgs) < len(r_cfgs)
        assert s_cfgs[0][0] == "full" and r_cfgs[0][0] == "full"

    def test_reuse_refuses_mismatched_constants(self, tmp_path):
        cell = composite_cells()[1]  # t2_95 n500
        path = tmp_path / f"{cell.name}.composite.json"
        good = _composite_constants(cell)
        bad = dict(good, corner_exponent=CORNER_EXPONENT * 10)
        path.write_text(
            '{"constants": ' + __import__("json").dumps(bad) + ', "records": {}}'
        )
        with pytest.raises(RuntimeError, match="different design constants"):
            _load_composite(tmp_path, cell)

    def test_reuse_accepts_matching_constants(self, tmp_path):
        import json

        cell = composite_cells()[1]
        path = tmp_path / f"{cell.name}.composite.json"
        payload = {"constants": _composite_constants(cell), "records": {}}
        path.write_text(json.dumps(payload))
        # round-trips through JSON (tuples become lists) and still matches
        loaded = _load_composite(tmp_path, cell)
        assert loaded is not None


class TestBoundarySurface:
    def test_lhs_points_deterministic_and_in_bounds(self):
        a = boundary_lhs_points()
        b = boundary_lhs_points()
        assert a == b
        # the achievability filter may drop a few sampled points (the
        # paper's LHS pipeline drops the same combinations)
        assert 0.8 * LHS_N_CELLS <= len(a) <= LHS_N_CELLS
        for pt in a:
            assert LHS_DF_BOUNDS[0] <= pt["df"] <= LHS_DF_BOUNDS[1]
            assert LHS_AUC_BOUNDS[0] <= pt["auc"] <= LHS_AUC_BOUNDS[1]
            assert LHS_N_BOUNDS[0] <= pt["n"] <= LHS_N_BOUNDS[1]
        assert len({pt["index"] for pt in a}) == len(a)

    def test_lhs_points_all_achievable(self):
        from studroc_paper.datagen.roc_to_dgp import StudentTSolver

        solver = StudentTSolver()
        for pt in boundary_lhs_points():
            assert pt["auc"] <= solver._compute_auc(pt["df"], 20.0)

    def _synthetic_rows(self, beta_true, n_cells=80, reps=125, seed=1):
        from scipy.special import expit
        from scipy.stats import norm

        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(n_cells):
            df = float(np.exp(rng.uniform(np.log(1.1), np.log(30))))
            auc = float(norm.cdf(rng.uniform(norm.ppf(0.55), norm.ppf(0.99))))
            n = int(np.exp(rng.uniform(np.log(100), np.log(2500))))
            p = expit(
                beta_true[0]
                + beta_true[1] * np.log(n)
                + beta_true[2] * np.log(df)
                + beta_true[3] * norm.ppf(auc)
            )
            rows.append(
                {
                    "df": df,
                    "auc": auc,
                    "n": n,
                    "cov": rng.binomial(reps, p) / reps,
                    "reps": reps,
                }
            )
        return rows

    def test_fit_recovers_monotone_surface(self):
        beta_true = np.array([-2.0, 0.9, 0.6, -0.8])
        rows = self._synthetic_rows(beta_true)
        fit = fit_boundary_surface(rows, n_boot=10, seed=0)
        beta = fit["beta"]
        # sign constraints respected and slope in log n roughly recovered
        assert beta[1] >= 0 and beta[2] >= 0 and beta[3] <= 0
        assert abs(beta[1] - beta_true[1]) < 0.3
        # fitted coverage close to truth at a mid-design point
        from scipy.special import expit
        from scipy.stats import norm

        p_true = expit(
            beta_true[0]
            + beta_true[1] * np.log(500)
            + beta_true[2] * np.log(3.0)
            + beta_true[3] * norm.ppf(0.95)
        )
        assert abs(surface_predict(beta, 3.0, 0.95, 500) - p_true) < 0.03

    def test_n_star_monotone_in_df_and_auc(self):
        beta = np.array([-2.0, 0.9, 0.6, -0.8])
        # heavier tail (smaller df) or higher AUC needs a larger n
        assert surface_n_star(beta, 1.1, 0.95) > surface_n_star(beta, 5.0, 0.95)
        assert surface_n_star(beta, 2.0, 0.99) > surface_n_star(beta, 2.0, 0.90)
        # crossing consistency: fitted coverage at n* equals the bar
        n_star = surface_n_star(beta, 2.0, 0.95)
        assert abs(surface_predict(beta, 2.0, 0.95, n_star) - BAR_POINT) < 1e-9

    def test_replay_empirical_auc_matches_direct_and_is_deterministic(self):
        from followup_runs import (
            boundary_cells,
            register_followup_shapes,
            sample_scores,
        )

        register_followup_shapes()
        cell = [c for c in boundary_cells() if "t2_95--n150" in c.name][0]
        aucs = replay_empirical_aucs(cell, 5)
        assert np.array_equal(aucs, replay_empirical_aucs(cell, 5))
        # direct pairwise Mann-Whitney on the same replayed data
        y_true, y_score, _ = sample_scores(cell, 3)
        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]
        direct = float(np.mean(pos[:, None] > neg[None, :]))
        assert abs(aucs[3] - direct) < 1e-12
        # in the right neighborhood of the cell's true AUC (.95)
        assert 0.85 < aucs.mean() < 1.0

    def test_n_star_flat_slope_guard(self):
        below = np.array([-10.0, 0.0, 0.0, 0.0])
        above = np.array([10.0, 0.0, 0.0, 0.0])
        assert surface_n_star(below, 2.0, 0.95) == float("inf")
        assert surface_n_star(above, 2.0, 0.95) == 0.0
