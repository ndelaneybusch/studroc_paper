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
    SENTINEL_INTERIOR_C,
    _composite_constants,
    _load_composite,
    _stitch,
    classify,
    composite_cells,
    composite_configs,
    needs_topup,
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
