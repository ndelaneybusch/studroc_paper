"""Red-team tests for the Stage F localized-M3-floor infrastructure."""

import gzip
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/c_calibration"))

import stage_f_run  # noqa: E402
from stage_f_analysis import (  # noqa: E402
    CellRecords,
    fit_stage_a,
    load_complete_study,
    score_record,
)
from stage_f_core import (  # noqa: E402
    EdgeRule,
    ObservableSummary,
    RegionArtifact,
    SupportBox,
    build_record,
    component_masks,
    conservative_resample,
    decode_array,
    decode_violations,
    empirical_observables,
    encode_array,
    encode_violations,
    freeze_artifact,
    load_artifact,
    stitch_hybrid,
    write_artifact,
)
from stage_f_design import (  # noqa: E402
    StageFCell,
    imbalance_lhs_cells,
    manifest_payload,
    study_b_cells,
    study_c_cells,
)
from stage_f_run import CELL_SCHEMA, load_existing, run_study  # noqa: E402


@pytest.fixture
def observables() -> ObservableSummary:
    """Representative observable-only rule inputs."""
    return ObservableSummary(
        n0=4, n1=5, auc_hat=0.8, auc_ub=0.95, auc_delong_ub=0.91, m30=1, m50=2, m70=3
    )


@pytest.fixture
def artifact() -> RegionArtifact:
    """Small frozen artifact exercising count and flat-ROC coordinates."""
    return freeze_artifact(
        RegionArtifact(
            rule_id="stage_f_v1",
            left=EdgeRule(
                coordinate="negative_count", family="constant", intercept=1.0
            ),
            right=EdgeRule(
                coordinate="positive_tail", family="constant", intercept=3.0
            ),
            support=SupportBox(n0=(2, 10), n1=(2, 10), auc_ub=(0.5, 1.0)),
        )
    )


def test_widening_closure_contains_both_required_parents_on_their_domains():
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
    fid_lower = np.array([0.0, 0.6, 0.6, 0.7, 1.0])
    upper = np.array([0.4, 0.7, 0.8, 0.9, 1.0])
    m3_lower = np.array([0.0, 0.1, 0.2, 0.3, 1.0])
    region = np.array([False, False, True, True, True])
    legacy_lower, _ = stitch_hybrid(
        fid_lower, upper, m3_lower, upper, region, closure="legacy"
    )
    assert legacy_lower[2] == 0.6
    assert legacy_lower[2] > m3_lower[2]


def test_empty_and_full_regions_reduce_to_the_expected_parent_relations():
    fid_lower = np.array([0.0, 0.2, 0.5, 1.0])
    fid_upper = np.array([0.3, 0.6, 0.9, 1.0])
    m3_lower = np.array([0.0, 0.1, 0.4, 1.0])
    m3_upper = np.array([0.4, 0.7, 0.95, 1.0])
    empty = np.zeros(4, dtype=bool)
    full = np.ones(4, dtype=bool)
    empty_band = stitch_hybrid(fid_lower, fid_upper, m3_lower, m3_upper, empty)
    np.testing.assert_array_equal(empty_band[0], fid_lower)
    np.testing.assert_array_equal(empty_band[1], fid_upper)
    full_band = stitch_hybrid(fid_lower, fid_upper, m3_lower, m3_upper, full)
    assert np.all(full_band[0] <= m3_lower)
    assert np.all(full_band[1] >= m3_upper)


def test_grid_resampling_uses_conservative_step_indices():
    lower = np.array([0.0, 0.2, 0.7, 1.0])
    upper = np.array([0.1, 0.5, 0.9, 1.0])
    lower_out, upper_out = conservative_resample(lower, upper, size=5)
    np.testing.assert_array_equal(lower_out, [0.0, 0.0, 0.2, 0.7, 1.0])
    np.testing.assert_array_equal(upper_out, [0.1, 0.5, 0.9, 1.0, 1.0])


def test_positive_tail_region_includes_complete_flat_preimages(
    artifact: RegionArtifact, observables: ObservableSummary
):
    khat = np.array([0, 2, 2, 4, 5])
    left, right = component_masks(artifact, observables=observables, khat=khat)
    np.testing.assert_array_equal(left, [True, True, False, False, False])
    np.testing.assert_array_equal(right, [False, True, True, True, True])


def test_out_of_support_rule_uses_full_region(
    artifact: RegionArtifact, observables: ObservableSummary
):
    outside = replace(observables, n0=100)
    left, right = component_masks(
        artifact, observables=outside, khat=np.zeros(101, dtype=int)
    )
    assert left.all() and right.all()


def test_artifact_round_trip_and_hash_tamper_detection(
    tmp_path: Path, artifact: RegionArtifact
):
    path = tmp_path / "stage_f_v1.json"
    write_artifact(artifact, path)
    assert load_artifact(path) == artifact
    payload = json.loads(path.read_text())
    payload["left"]["intercept"] += 1.0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="content hash"):
        load_artifact(path)


def test_array_and_violation_encodings_are_lossless_with_overflow_fallback():
    values = np.array([0.0, np.nan, np.inf, -3.25], dtype=np.float64)
    decoded = decode_array(encode_array(values))
    assert decoded.tobytes() == values.tobytes()
    mask = np.zeros(300, dtype=bool)
    mask[::2] = True
    encoded = encode_violations(mask, max_intervals=4)
    assert encoded["encoding"] == "bitset" and encoded["overflow"] is True
    np.testing.assert_array_equal(decode_violations(encoded), mask)


def test_empirical_observables_match_pairwise_auc_and_primary_bound():
    labels = np.array([1, 0, 1, 0, 0, 1], dtype=np.uint8)
    summary, khat = empirical_observables(labels, delta=0.05)
    positive_positions = np.flatnonzero(labels == 1)
    negative_positions = np.flatnonzero(labels == 0)
    direct = np.mean(positive_positions[:, None] < negative_positions[None, :])
    assert summary.auc_hat == pytest.approx(direct, abs=1e-15)
    expected = min(
        1.0,
        direct
        + np.sqrt(0.5 * (1.0 / summary.n0 + 1.0 / summary.n1) * np.log(1.0 / 0.05)),
    )
    assert summary.auc_ub == pytest.approx(expected, abs=1e-15)
    np.testing.assert_array_equal(khat, [1, 2, 2, 3])


def test_replicate_routes_one_shared_tie_order_to_every_arm(monkeypatch):
    labels = np.array([1, 0, 1, 0, 0, 1], dtype=np.uint8)
    seen = []
    lower = np.array([0.0, 0.1, 0.4, 1.0])
    upper = np.array([0.3, 0.7, 0.9, 1.0])

    class FakeCurve:
        def eval(self, grid):
            return np.array([0.0, 0.4, 0.8, 1.0])

    def fake_fiducial(shared, **kwargs):
        seen.append(shared)
        return {"0.05": (lower, upper), "0.5": (lower, upper)}, None

    def fake_m3(shared, **kwargs):
        seen.append(shared)
        return np.arange(4) / 3, lower, upper

    monkeypatch.setattr(stage_f_run, "sample_labels", lambda cell, rep: (labels, 7))
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
        partition="selection",
        m_draws=100,
    )
    record = stage_f_run.run_replicate(cell, 0, n_threads=1)
    assert len(seen) == 5
    assert all(shared is labels for shared in seen)
    assert record["observables"] == empirical_observables(labels)[0].__dict__


def test_rule_has_no_channel_for_true_auc_or_true_roc(
    artifact: RegionArtifact, observables: ObservableSummary
):
    khat = np.array([0, 2, 2, 4, 5])
    first = component_masks(artifact, observables=observables, khat=khat)
    radically_different_truth = np.array([0.0, 0.01, 0.02, 0.03, 1.0])
    second = component_masks(artifact, observables=observables, khat=khat)
    assert radically_different_truth.shape == first[0].shape
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_offline_score_equals_direct_band_reconstruction():
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
    score = score_record(record, rule="count5", alpha=0.05, alpha2_key="0.05")
    direct = stitch_hybrid(fid[0], fid[1], m3[0], m3[1], np.ones(4, dtype=bool))
    assert score["covered"] == bool(
        np.all(direct[0] <= truth) and np.all(truth <= direct[1])
    )
    assert score["fiducial_failed"] is True
    assert score["conditional_capture"] is True
    assert score["area"] == float(np.mean(direct[1] - direct[0]))


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        (
            {"manifest_hash": "first", "rule_artifact_hash": "same"},
            {"manifest_hash": "second", "rule_artifact_hash": "same"},
        ),
        (
            {"manifest_hash": "same", "rule_artifact_hash": "first"},
            {"manifest_hash": "same", "rule_artifact_hash": "second"},
        ),
    ],
    ids=("design", "rule"),
)
def test_resume_refuses_changed_design_or_rule_hash(
    tmp_path: Path, stored: dict, expected: dict
):
    """Prevent resuming across either design or frozen-rule boundaries."""
    path = tmp_path / "cell.json.gz"
    payload = {"schema": CELL_SCHEMA, "meta": {"compatibility": stored}, "records": []}
    with gzip.open(path, "wt") as handle:
        json.dump(payload, handle)
    with pytest.raises(RuntimeError, match="Refusing to mix"):
        load_existing(path, expected_compatibility=expected)


def test_frozen_design_sizes_and_imbalance_orientation_strata():
    assert len(study_b_cells()) == 24
    assert len(study_c_cells()) == 14
    imbalance = imbalance_lhs_cells()
    assert len(imbalance) == 24
    for lower, upper in ((0.85, 0.90), (0.90, 0.95), (0.95, 1.0)):
        cells = [cell for cell in imbalance if lower <= cell.shape_meta["auc"] < upper]
        assert any(cell.n0 > cell.n1 for cell in cells)
        assert any(cell.n1 > cell.n0 for cell in cells)


def test_manifest_hash_is_sensitive_to_every_cell_constant():
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
        partition="external",
        m_draws=2_000,
    )
    original = manifest_payload(study="B", cells=[cell])
    changed = manifest_payload(study="B", cells=[replace(cell, n0=11)])
    assert original["content_hash"] != changed["content_hash"]


def test_complete_study_loader_requires_every_frozen_cell(tmp_path: Path):
    """Prevent fitting from a silent partial Study A output directory."""
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
        partition="selection",
        m_draws=100,
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
        "meta": {
            "cell": manifest["cells"][0],
            "compatibility": {
                "manifest_hash": manifest["content_hash"],
                "cell": manifest["cells"][0],
            },
        },
        "records": [{"rep": 0}],
    }
    with gzip.open(output_path, "wt") as handle:
        json.dump(payload, handle)
    with pytest.raises(RuntimeError, match="at least 2 are required"):
        load_complete_study(tmp_path, study="A")


def test_external_run_refuses_non_refit_artifact(
    tmp_path: Path, artifact: RegionArtifact
):
    """Prevent external data generation with a selection-phase candidate."""
    candidate = freeze_artifact(replace(artifact, training={"phase": "selection"}))
    artifact_path = tmp_path / "candidate.json"
    write_artifact(candidate, artifact_path)
    cell = StageFCell(
        name="must-not-run",
        study="B",
        source="test",
        shape="binormal_90",
        shape_meta={"family": "binormal", "auc": 0.9},
        n0=10,
        n1=10,
        reps=1,
        reps_max=1,
        partition="external",
        m_draws=100,
    )
    manifest_path = tmp_path / "study_b.json"
    manifest_path.write_text(json.dumps(manifest_payload(study="B", cells=[cell])))
    with pytest.raises(ValueError, match="final refit"):
        run_study(
            manifest_path=manifest_path,
            root=tmp_path,
            artifact_path=artifact_path,
            workers=1,
            threads_per_call=1,
            mem_gb=1.0,
        )


def test_stage_a_pipeline_selects_refits_and_freezes_from_stored_parents(
    tmp_path: Path,
):
    """Exercise the complete offline selector without generating any cloud."""
    labels = np.array([1, 0, 1, 0, 0, 1], dtype=np.uint8)
    observables, khat = empirical_observables(labels)
    truth = np.array([0.0, 0.7, 0.75, 1.0])
    fiducial = (np.array([0.0, 0.2, 0.6, 1.0]), np.array([0.2, 0.6, 0.9, 1.0]))
    m3 = (np.array([0.0, 0.1, 0.5, 1.0]), np.array([0.3, 0.7, 0.95, 1.0]))
    record = build_record(
        observables=observables,
        khat=khat,
        truth=truth,
        fiducial={"0.05": fiducial, "0.5": fiducial},
        m3={"0.05": m3, "0.025": m3, "0.5": m3, "0.25": m3},
    )

    def cell(name: str, partition: str) -> CellRecords:
        """Build one synthetic stored-parent cell for an offline partition."""
        return CellRecords(
            meta={
                "cell": {"name": name, "partition": partition},
                "compatibility": {"manifest_hash": "synthetic"},
            },
            records=[record] * 10,
        )

    artifact_path = tmp_path / "rules/stage_f_v1.json"
    report_path = tmp_path / "analysis/study_a_selection.json"
    artifact = fit_stage_a(
        [cell("select", "selection"), cell("validate", "internal_validation")],
        artifact_path=artifact_path,
        report_path=report_path,
    )
    assert artifact.training["phase"] == "refit"
    assert load_artifact(artifact_path) == artifact
    assert json.loads(report_path.read_text())["artifact_hash"] == artifact.content_hash
    assert report_path.with_name("stage_f_v1_spec_amendment.md").exists()
