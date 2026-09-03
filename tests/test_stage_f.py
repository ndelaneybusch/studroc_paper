"""Red-team tests for the fixed Stage F frontier-floor infrastructure."""

import gzip
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/c_calibration"))

import stage_f_run  # noqa: E402
from shapes import get_curve, shape_registry  # noqa: E402
from stage_f_analysis import load_complete_study, score_record  # noqa: E402
from stage_f_core import (  # noqa: E402
    ObservableSummary,
    build_record,
    component_masks,
    conservative_resample,
    decode_array,
    decode_violations,
    empirical_observables,
    encode_array,
    encode_violations,
    frontier_left_cutoff,
    frontier_region_masks,
    stitch_hybrid,
)
from stage_f_design import (  # noqa: E402
    RankSample,
    StageFCell,
    cell_curve,
    imbalance_lhs_cells,
    make_sliver_curve,
    manifest_payload,
    sample_labels,
    study_b_cells,
    study_c_cells,
)
from stage_f_run import CELL_SCHEMA, load_existing  # noqa: E402

M_DRAWS = 2_000


@pytest.fixture
def observables() -> ObservableSummary:
    """Return representative rank-derived diagnostics on an 11-point grid."""
    return ObservableSummary(
        n0=10, n1=5, auc_hat=0.8, auc_ub=0.95, auc_delong_ub=0.91, m30=1, m50=2, m70=3
    )


@pytest.fixture(autouse=True)
def restore_shape_registry():
    """Remove curves registered by each Stage F test."""
    registry = shape_registry()
    original_names = set(registry)
    yield
    for name in set(registry) - original_names:
        del registry[name]
    get_curve.cache_clear()


def test_widening_closure_contains_both_required_parents_on_their_domains():
    """Guard the regional M3 containment argument after monotone closure."""
    fid_lower = np.array([0.0, 0.6, 0.6, 0.7, 1.0])
    fid_upper = np.array([0.3, 0.7, 0.8, 0.9, 1.0])
    m3_lower = np.array([0.0, 0.1, 0.2, 0.3, 1.0])
    m3_upper = np.array([0.4, 0.75, 0.85, 0.95, 1.0])
    region = np.array([False, False, True, True, True])
    lower, upper = stitch_hybrid(
        fid_lower, fid_upper, m3_lower, m3_upper, region, closure="widening"
    )
    assert np.all(lower <= fid_lower) and np.all(upper >= fid_upper)
    assert np.all(lower[region] <= m3_lower[region])
    assert np.all(upper[region] >= m3_upper[region])
    assert np.all(np.diff(lower) >= 0.0) and np.all(np.diff(upper) >= 0.0)


def test_legacy_lower_closure_has_regional_m3_counterexample():
    """Keep the known legacy closure defect visible as a benchmark warning."""
    fid_lower = np.array([0.0, 0.6, 0.6, 0.7, 1.0])
    upper = np.array([0.4, 0.7, 0.8, 0.9, 1.0])
    m3_lower = np.array([0.0, 0.1, 0.2, 0.3, 1.0])
    region = np.array([False, False, True, True, True])
    legacy_lower, _ = stitch_hybrid(
        fid_lower, upper, m3_lower, upper, region, closure="legacy"
    )
    assert legacy_lower[2] == 0.6
    assert legacy_lower[2] > m3_lower[2]


def test_empty_and_full_regions_reduce_to_expected_parent_relations():
    """Prevent closure from changing the fiducial-only or full-floor limits."""
    fid_lower = np.array([0.0, 0.2, 0.5, 1.0])
    fid_upper = np.array([0.3, 0.6, 0.9, 1.0])
    m3_lower = np.array([0.0, 0.1, 0.4, 1.0])
    m3_upper = np.array([0.4, 0.7, 0.95, 1.0])
    empty_band = stitch_hybrid(
        fid_lower, fid_upper, m3_lower, m3_upper, np.zeros(4, dtype=bool)
    )
    np.testing.assert_array_equal(empty_band[0], fid_lower)
    np.testing.assert_array_equal(empty_band[1], fid_upper)
    full_band = stitch_hybrid(
        fid_lower, fid_upper, m3_lower, m3_upper, np.ones(4, dtype=bool)
    )
    assert np.all(full_band[0] <= m3_lower)
    assert np.all(full_band[1] >= m3_upper)


def test_grid_resampling_uses_conservative_step_indices():
    """Prevent plotting-grid conversion from silently narrowing a band."""
    lower = np.array([0.0, 0.2, 0.7, 1.0])
    upper = np.array([0.1, 0.5, 0.9, 1.0])
    lower_out, upper_out = conservative_resample(lower, upper, size=5)
    np.testing.assert_array_equal(lower_out, [0.0, 0.0, 0.2, 0.7, 1.0])
    np.testing.assert_array_equal(upper_out, [0.1, 0.5, 0.9, 1.0, 1.0])


def test_frontier_components_match_all_three_prespecified_rules(
    observables: ObservableSummary,
):
    """Pin the count frontier, complete preimages, and square-root margin."""
    khat = np.array([0, 1, 1, 2, 2, 3, 4, 5, 5, 5, 5])
    floor_left, floor_right = component_masks(
        observables=observables, khat=khat, m_draws=M_DRAWS
    )
    run_left, run_right = frontier_region_masks(
        "frontier_run0", observables=observables, khat=khat, m_draws=M_DRAWS
    )
    j1_left, j1_right = frontier_region_masks(
        "frontier_j1", observables=observables, khat=khat, m_draws=M_DRAWS
    )
    np.testing.assert_array_equal(floor_left, np.arange(11) <= 8)
    np.testing.assert_array_equal(run_left, floor_left)
    np.testing.assert_array_equal(j1_left, floor_left)
    np.testing.assert_array_equal(run_right, np.arange(11) >= 7)
    np.testing.assert_array_equal(j1_right, np.arange(11) >= 6)
    np.testing.assert_array_equal(floor_right, np.arange(11) >= 3)


def test_unsampled_sliver_rank_pattern_expands_right_region_leftward(
    observables: ObservableSummary,
):
    """Check the proposed adaptive response to a longer saturated run."""
    sampled = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    unsampled = np.array([0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5])
    _, sampled_right = component_masks(
        observables=observables, khat=sampled, m_draws=M_DRAWS
    )
    _, unsampled_right = component_masks(
        observables=observables, khat=unsampled, m_draws=M_DRAWS
    )
    assert np.flatnonzero(unsampled_right)[0] < np.flatnonzero(sampled_right)[0]
    assert np.all(unsampled_right[sampled_right])


def test_frontier_rule_has_no_auc_or_true_roc_channel(observables: ObservableSummary):
    """Enforce rank-only routing independently of every AUC diagnostic."""
    khat = np.array([0, 1, 1, 2, 2, 3, 4, 5, 5, 5, 5])
    first = component_masks(observables=observables, khat=khat, m_draws=M_DRAWS)
    changed = replace(
        observables, auc_hat=0.01, auc_ub=0.02, auc_delong_ub=0.03, m30=9, m50=9
    )
    second = component_masks(observables=changed, khat=khat, m_draws=M_DRAWS)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_frontier_left_cutoff_uses_each_cells_actual_cloud_budget():
    """Keep the left frontier responsive to the simulation's cloud size."""
    assert frontier_left_cutoff(n0=10, m_draws=20) == 4
    assert frontier_left_cutoff(n0=10, m_draws=2_000) == 8


def test_array_and_violation_encodings_are_lossless_with_overflow_fallback():
    """Protect exact offline reconstruction for irregular violation sets."""
    values = np.array([0.0, np.nan, np.inf, -3.25], dtype=np.float64)
    decoded = decode_array(encode_array(values))
    assert decoded.tobytes() == values.tobytes()
    mask = np.zeros(300, dtype=bool)
    mask[::2] = True
    encoded = encode_violations(mask, max_intervals=4)
    assert encoded["encoding"] == "bitset" and encoded["overflow"] is True
    np.testing.assert_array_equal(decode_violations(encoded), mask)


def test_empirical_observables_match_pairwise_auc_and_count_map():
    """Validate retained diagnostics without granting them routing authority."""
    labels = np.array([1, 0, 1, 0, 0, 1], dtype=np.uint8)
    summary, khat = empirical_observables(labels, delta=0.05)
    positive_positions = np.flatnonzero(labels == 1)
    negative_positions = np.flatnonzero(labels == 0)
    direct = np.mean(positive_positions[:, None] < negative_positions[None, :])
    assert summary.auc_hat == pytest.approx(direct, abs=1e-15)
    np.testing.assert_array_equal(khat, [1, 2, 2, 3])


@pytest.mark.parametrize(
    ("n0", "n1", "expected_count"),
    [(500, 500, 0.8), (2_000, 500, 0.8), (500, 2_000, 0.8)],
    ids=("balanced", "negative-heavy", "positive-heavy"),
)
def test_sliver_curve_has_requested_area_mass_and_continuity(
    n0: int, n1: int, expected_count: float
):
    """Validate the generalized construction in both imbalance directions."""
    curve = make_sliver_curve(
        auc=0.8, n0=n0, n1=n1, expected_sliver_count=expected_count, tail_extent=0.25
    )
    hinge = 1.0 - 1.0 / n0
    values = curve.eval(np.array([hinge - 1e-10, hinge, hinge + 1e-10]))
    assert curve.auc() == pytest.approx(0.8, abs=3e-5)
    assert 1.0 - values[1] == pytest.approx(expected_count / n1, abs=1e-12)
    assert np.max(np.diff(values)) < 1e-6
    plateau = curve.eval(np.array([0.75, (0.75 + hinge) / 2.0, hinge]))
    np.testing.assert_allclose(plateau, 1.0 - expected_count / n1, atol=1e-12)


def test_sliver_sampler_records_the_actual_sampled_event():
    """Ensure conditional summaries use realized sliver membership."""
    cell = StageFCell(
        name="test-sliver-sample",
        study="B",
        source="sliver_fresh",
        shape="test_sliver_sample_shape",
        shape_meta={
            "family": "sliver",
            "auc": 0.8,
            "n0": 50,
            "n1": 80,
            "expected_sliver_count": 0.8,
            "tail_extent": 0.25,
        },
        n0=50,
        n1=80,
        reps=1,
        reps_max=1,
        m_draws=2_000,
    )
    sample = sample_labels(cell, 0)
    assert sample.diagnostics["sliver_sampled"] == (
        sample.diagnostics["sliver_count"] > 0
    )
    assert sample.labels.sum() == cell.n1


def test_replicate_routes_one_shared_tie_order_to_every_parent(monkeypatch):
    """Prevent accidental independent sampling across paired procedures."""
    labels = np.array([1, 0, 1, 0, 0, 1], dtype=np.uint8)
    seen = []
    lower = np.array([0.0, 0.1, 0.4, 1.0])
    upper = np.array([0.3, 0.7, 0.9, 1.0])

    class FakeCurve:
        """Minimal exact truth used by the runner routing test."""

        def eval(self, grid):
            """Return a valid truth on the requested four-point grid."""
            assert len(grid) == 4
            return np.array([0.0, 0.4, 0.8, 1.0])

    def fake_fiducial(shared, **kwargs):
        """Capture the shared labels passed to the fiducial parent."""
        seen.append(shared)
        return {"0.05": (lower, upper), "0.5": (lower, upper)}, None

    def fake_m3(shared, **kwargs):
        """Capture the shared labels passed to each M3 parent level."""
        seen.append(shared)
        return np.arange(4) / 3, lower, upper

    monkeypatch.setattr(
        stage_f_run,
        "sample_labels",
        lambda cell, rep: RankSample(labels=labels, cloud_seed=7, diagnostics={}),
    )
    monkeypatch.setattr(stage_f_run, "cell_curve", lambda cell: FakeCurve())
    monkeypatch.setattr(stage_f_run, "_fiducial_bands", fake_fiducial)
    monkeypatch.setattr(stage_f_run, "_m3_band_from_labels_rs", fake_m3)
    cell = StageFCell(
        name="ties",
        study="A",
        source="test",
        shape="unused",
        shape_meta={"family": "binormal", "auc": 0.9},
        n0=3,
        n1=3,
        reps=1,
        reps_max=1,
        m_draws=100,
    )
    record = stage_f_run.run_replicate(cell, 0, n_threads=1)
    assert len(seen) == 5
    assert all(shared is labels for shared in seen)
    assert record["observables"] == asdict(empirical_observables(labels)[0])


def test_offline_frontier_score_equals_direct_band_reconstruction():
    """Ensure stored-parent analysis exactly reproduces direct stitching."""
    labels = np.array([1, 0, 1, 0, 0, 1], dtype=np.uint8)
    observables, khat = empirical_observables(labels)
    truth = np.array([0.0, 0.7, 0.75, 1.0])
    fid = (np.array([0.0, 0.2, 0.6, 1.0]), np.array([0.2, 0.6, 0.9, 1.0]))
    m3 = (np.array([0.0, 0.1, 0.5, 1.0]), np.array([0.3, 0.7, 0.95, 1.0]))
    record = build_record(
        observables=observables,
        khat=khat,
        truth=truth,
        fiducial={"0.05": fid, "0.5": fid},
        m3={"0.05": m3, "0.025": m3, "0.5": m3, "0.25": m3},
    )
    score = score_record(
        record, rule="frontier_floor_v1", alpha=0.05, alpha2_key="0.05", m_draws=M_DRAWS
    )
    left, right = component_masks(observables=observables, khat=khat, m_draws=M_DRAWS)
    direct = stitch_hybrid(fid[0], fid[1], m3[0], m3[1], left | right)
    assert score["covered"] == bool(
        np.all(direct[0] <= truth) and np.all(truth <= direct[1])
    )
    assert score["area"] == float(np.mean(direct[1] - direct[0]))


def test_resume_refuses_output_from_a_different_cell(tmp_path: Path):
    """Prevent an edited cell from appending to incompatible simulations."""
    cell = StageFCell(
        name="resume",
        study="A",
        source="test",
        shape="binormal_90",
        shape_meta={"family": "binormal", "auc": 0.9},
        n0=10,
        n1=10,
        reps=2,
        reps_max=2,
        m_draws=M_DRAWS,
    )
    path = tmp_path / "cell.json.gz"
    payload = {"schema": CELL_SCHEMA, "meta": {"cell": asdict(cell)}, "records": []}
    with gzip.open(path, "wt") as handle:
        json.dump(payload, handle)
    with pytest.raises(RuntimeError, match="does not match cell resume"):
        load_existing(path, expected_cell=replace(cell, n1=11))


def test_design_sizes_slivers_and_geometry_labels():
    """Pin every new prospective design requirement before outcomes exist."""
    study_b = study_b_cells()
    study_c = study_c_cells()
    assert len(study_b) == 30
    assert len(study_c) == 14
    slivers = [cell for cell in study_b if cell.shape_meta["family"] == "sliver"]
    assert len(slivers) == 6
    assert {cell.shape_meta["auc"] for cell in slivers} == {0.6, 0.8, 0.95}
    assert any(cell.n0 > cell.n1 for cell in slivers)
    assert any(cell.n1 > cell.n0 for cell in slivers)
    assert {cell.n0 for cell in slivers} >= {250, 2_000}
    for cell in slivers:
        assert cell_curve(cell).auc() == pytest.approx(cell.shape_meta["auc"], abs=3e-5)
    assert all(
        cell.shape_meta["corner_geometry"]
        in {"corner-concave", "corner-convex", "ambiguous"}
        for cell in study_c
    )


def test_imbalance_lhs_retains_auc_and_orientation_strata():
    """Protect Study A mechanism coverage after removing its partitions."""
    imbalance = imbalance_lhs_cells()
    assert len(imbalance) == 24
    for lower, upper in ((0.85, 0.90), (0.90, 0.95), (0.95, 1.0)):
        cells = [cell for cell in imbalance if lower <= cell.shape_meta["auc"] < upper]
        assert any(cell.n0 > cell.n1 for cell in cells)
        assert any(cell.n1 > cell.n0 for cell in cells)


def test_manifest_is_a_plain_readable_design_snapshot():
    """Keep manifests inspectable without a parallel artifact protocol."""
    cell = StageFCell(
        name="test",
        study="B",
        source="test",
        shape="binormal_90",
        shape_meta={"family": "binormal", "auc": 0.9},
        n0=10,
        n1=10,
        reps=400,
        reps_max=1_200,
        m_draws=2_000,
    )
    original = manifest_payload(study="B", cells=[cell])
    changed = manifest_payload(study="B", cells=[replace(cell, n1=11)])
    assert set(original) == {"schema", "study", "cells"}
    assert original["cells"] == [asdict(cell)]
    assert original["cells"] != changed["cells"]


def test_complete_study_loader_requires_every_manifest_cell(tmp_path: Path):
    """Prevent summaries from silently accepting a partial study."""
    cell = StageFCell(
        name="required",
        study="A",
        source="test",
        shape="binormal_90",
        shape_meta={"family": "binormal", "auc": 0.9},
        n0=10,
        n1=10,
        reps=2,
        reps_max=2,
        m_draws=2_000,
    )
    manifest = manifest_payload(study="A", cells=[cell])
    manifest_path = tmp_path / "manifests/study_a.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="Missing Stage F A cell"):
        load_complete_study(tmp_path, study="A")

    output_path = tmp_path / "A/required.json.gz"
    output_path.parent.mkdir()
    payload = {
        "schema": CELL_SCHEMA,
        "meta": {"cell": manifest["cells"][0]},
        "records": [{"rep": 0}],
    }
    with gzip.open(output_path, "wt") as handle:
        json.dump(payload, handle)
    with pytest.raises(RuntimeError, match="at least 2 are required"):
        load_complete_study(tmp_path, study="A")


def test_cli_has_no_post_outcome_fit_action():
    """Ensure the command surface cannot resurrect the learned-region path."""
    with pytest.raises(SystemExit):
        stage_f_run.parse_args(["fit"])
