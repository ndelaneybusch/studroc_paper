"""Tests for the trim-exponent calibration study's design, shapes, and map.

The study runs once on another machine, so the risks these tests retire are
design-time, not runtime: a shape whose AUC misses its spec target, a
non-deterministic seed path (which would break resume/extend), an M budget
that under-resolves the deepest fitted trim, or a frozen-map resolver whose
clamps/fallbacks disagree with the spec's behavior contracts.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts" / "c_calibration")
)

import design  # noqa: E402
import map_eval  # noqa: E402
import shapes  # noqa: E402

# ---------------------------------------------------------------------------
# shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    sorted(set(shapes.shape_registry()) - {"trapezoid_q10_90", "kink_80"}),
)
def test_shape_auc_hits_target(name):
    spec = shapes.shape_registry()[name]
    curve = shapes.get_curve(name)
    assert curve.auc() == pytest.approx(spec.meta["auc"], abs=2e-3), name


def test_every_curve_is_a_valid_roc():
    for name in shapes.shape_registry():
        curve = shapes.get_curve(name)
        t = np.linspace(0, 1, 2001)
        r = curve.eval(t)
        assert r[0] == 0.0 and r[-1] == 1.0, name
        assert np.all(np.diff(r) >= -1e-12), name
        assert np.all((r >= 0) & (r <= 1)), name


def test_curve_inverse_samples_have_curve_as_cdf():
    """Sampling positives via curve.inv(U) must reproduce the curve as the
    placement-value CDF — the exactness the whole rank-space design rests on."""
    curve = shapes.get_curve("hetero_90_r3")
    rng = np.random.default_rng(0)
    w = curve.inv(rng.random(200_000))
    for t in (0.01, 0.05, 0.2, 0.5, 0.9):
        frac = float((w <= t).mean())
        assert frac == pytest.approx(curve.eval(t), abs=0.005), f"t={t}"


def test_trapezoid_is_the_quantized_estimand():
    base = shapes.get_curve("binormal_90")
    trap = shapes.make_trapezoid(base, q=10)
    edges = np.arange(11) / 10
    assert np.allclose(trap.eval(edges), base.eval(edges))
    mids = edges[:-1] + 0.05
    assert np.allclose(
        trap.eval(mids), 0.5 * (base.eval(edges[:-1]) + base.eval(edges[1:]))
    )
    assert trap.auc() < base.auc()  # strictly inside a concave curve


def test_quantize_jitter_matches_trapezoid_estimand():
    """After quantize+jitter, positives' CDF at the bin edges must equal the
    trapezoid truth — random tie-breaking is estimand-exact."""
    curve = shapes.get_curve("binormal_90")
    rng = np.random.default_rng(4)
    q = 20
    w = curve.inv(rng.random(150_000))
    _, wj = shapes.quantize_jitter(np.array([0.5]), w, q, rng)
    trap = shapes.make_trapezoid(curve, q)
    for t in np.arange(1, q) / q:
        assert float((wj <= t).mean()) == pytest.approx(trap.eval(t), abs=0.006)


def test_lhs_heldout_shapes_are_deterministic_and_disjoint():
    a = shapes.lhs_heldout_specs()
    b = shapes.lhs_heldout_specs()
    assert a == b
    assert a[0]["family"] != a[1]["family"]
    for spec in a:
        lo, hi = shapes.LHS_AUC_BOUNDS
        assert lo <= spec["auc"] <= hi
    roles = {s.role for s in shapes.shape_registry().values()}
    assert roles == {"fitting", "heldout"}
    fitting = {n for n, s in shapes.shape_registry().items() if s.role == "fitting"}
    heldout = {n for n, s in shapes.shape_registry().items() if s.role == "heldout"}
    assert len(fitting) == 10 and len(heldout) == 6
    assert not fitting & heldout


# ---------------------------------------------------------------------------
# design
# ---------------------------------------------------------------------------


def test_cell_tables_are_unique_and_stage_consistent():
    a_cells = design.stage_a_cells()
    b_cells = design.stage_b_cells()
    names = [c.name for c in a_cells + b_cells]
    assert len(names) == len(set(names))
    assert all(c.stage == "A" for c in a_cells)
    assert all(c.stage == "B" for c in b_cells)
    registry = shapes.shape_registry()
    for cell in a_cells + b_cells:
        assert cell.shape in registry, cell.name
        assert cell.m_draws >= 2000
        assert cell.reps_max >= cell.reps
    # Stage A fits only fitting shapes; held-out shapes appear only in B.
    for cell in a_cells:
        assert registry[cell.shape].role == "fitting", cell.name


def test_m_budget_resolves_the_deepest_fitted_trim():
    """The x2 safety rule must leave the C=1 arm's expected trim depth
    >= 10 at the cell's smallest alpha (spec sections 4/5.3)."""
    for cell in design.stage_a_cells():
        ell = design.local_level_law(design.k_trim_of(cell.n0), cell.alpha_min)
        assert cell.m_draws * ell >= 10 - 1e-9, cell.name


def test_m_budget_uses_thinned_grid_above_threshold():
    assert design.k_trim_of(2000) == 2001
    k_thinned = design.k_trim_of(50_000)
    assert k_thinned < 2001
    # Thinning caps the budget: large-n cells stay near M ~ 12-13k.
    assert design.m_budget(50_000, 0.05) < 15_000
    assert design.m_budget(5_000, 0.01) > 50_000


def test_seed_sequences_are_deterministic_and_distinct():
    cells = design.stage_a_cells()[:2]
    s1 = design.rep_seed_sequence(cells[0], 7)
    s2 = design.rep_seed_sequence(cells[0], 7)
    s3 = design.rep_seed_sequence(cells[0], 8)
    s4 = design.rep_seed_sequence(cells[1], 7)
    r1 = np.random.default_rng(s1).random(4)
    assert np.array_equal(r1, np.random.default_rng(s2).random(4))
    assert not np.array_equal(r1, np.random.default_rng(s3).random(4))
    assert not np.array_equal(r1, np.random.default_rng(s4).random(4))


def test_reference_arms_cover_all_alphas_with_three_maps():
    arms = design.reference_arms((0.5, 0.05), 100, 100)
    labels = {(a.label, a.alpha) for a in arms}
    assert labels == {
        ("c1", 0.5),
        ("c2", 0.5),
        ("auto_prov", 0.5),
        ("c1", 0.05),
        ("c2", 0.05),
        ("auto_prov", 0.05),
    }
    for arm in arms:
        assert arm.alpha_eff == pytest.approx(1 - (1 - arm.alpha) ** arm.exponent)
    # Stage B: a frozen-map resolver replaces the provisional formula.
    frozen = design.reference_arms(
        (0.05,), 100, 100, auto_exponent_fn=lambda n0, n1, a: 1.7
    )
    auto = [a for a in frozen if a.label == "auto"]
    assert len(auto) == 1 and auto[0].exponent == 1.7


def test_provisional_auto_exponent_tapers_and_floors():
    c_small = design.provisional_auto_exponent(25)
    c_mid = design.provisional_auto_exponent(500)
    c_big = design.provisional_auto_exponent(50_000)
    assert c_small > c_mid > c_big > 1.0
    assert c_mid == pytest.approx(1.8)


# ---------------------------------------------------------------------------
# frozen-map resolver
# ---------------------------------------------------------------------------


def test_map_resolver_contracts():
    artifact = map_eval.placeholder_artifact()
    map_eval.validate_artifact(artifact)
    # Monotone taper toward 1 in n.
    cs = [
        map_eval.resolve_exponent(artifact, n0=n, n1=n, alpha=0.05)
        for n in (25, 100, 1000, 10_000, 200_000)
    ]
    assert all(a >= b for a, b in zip(cs[:-1], cs[1:], strict=True))
    assert all(c >= 1.0 for c in cs)
    # n below the calibrated range clamps to the small-n end (no upward
    # extrapolation).
    assert map_eval.resolve_exponent(
        artifact, n0=5, n1=5, alpha=0.05
    ) == pytest.approx(map_eval.resolve_exponent(artifact, n0=25, n1=25, alpha=0.05))
    # Imbalance uses the declared reduction (min).
    assert map_eval.resolve_exponent(
        artifact, n0=9000, n1=100, alpha=0.05
    ) == pytest.approx(map_eval.resolve_exponent(artifact, n0=100, n1=100, alpha=0.05))
    # Alpha outside the calibrated range: C = 1 with a warning.
    with pytest.warns(UserWarning, match="calibrated range"):
        assert map_eval.resolve_exponent(artifact, n0=500, n1=500, alpha=0.005) == 1.0


def test_map_validation_rejects_malformed_artifacts():
    good = map_eval.placeholder_artifact()
    for mutate in (
        {"schema": "nope"},
        {"coordinate": "banana"},
        {"alpha_range": [0.5, 0.01]},
        {"n_range": [100, 50]},
    ):
        bad = {**good, **mutate}
        with pytest.raises(ValueError):
            map_eval.validate_artifact(bad)
    bad_taper = {**good, "taper": {**good["taper"], "family": "spline"}}
    with pytest.raises(ValueError):
        map_eval.validate_artifact(bad_taper)
