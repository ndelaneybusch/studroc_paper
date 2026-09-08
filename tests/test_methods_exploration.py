"""Adversarial checks for experiment validity, numerical bounds and resumability."""

import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.methods_exploration.common import Curve, Store, metrics  # noqa: E402
from scripts.methods_exploration.interior import crossing, stitch  # noqa: E402
from scripts.methods_exploration.likelihood import (  # noqa: E402
    certified_likelihood_outer,
    exact_predictor,
    library,
)
from scripts.methods_exploration.m3 import calibrated, compose  # noqa: E402
from scripts.methods_exploration.projection import (  # noqa: E402
    coupled_scores,
    endpoint_steps,
    exact_rejector,
    mc_rejector,
    model_law,
    outer,
    rational_models,
    split_box,
    statistics,
)
from scripts.methods_exploration.rank import (  # noqa: E402
    exact_probability,
    likelihoods,
    partition,
    paths,
    sequential_mass,
    train_predictor,
)
from studroc_paper.methods.m3_band_rs import (  # noqa: E402
    _ell_bounds,
    _m3_band_from_labels_rs,
)

Q = Fraction


@pytest.mark.parametrize("mode", ["exact", "upper", "lower"])
@pytest.mark.parametrize(
    "labels", [(0, 1, 1, 0, 1), (1, 1, 1, 0, 0)], ids=["interlaced", "separated"]
)
def test_vectorized_dp_matches_rational_bounds_with_atoms_and_empty_cells(mode, labels):
    """Floating diagnostic probabilities must respect exact atom and gap semantics."""
    cells = ((Q(1, 4), Q(1, 2)), (Q(0), Q(1, 4)), (Q(1, 4), Q(0)), (Q(1, 2), Q(1, 4)))
    exact = exact_probability(labels=labels, cells=cells, mode=mode)
    measured = likelihoods(
        labels=np.asarray(labels, dtype=np.uint8),
        negative=np.array([float(a) for a, _ in cells]),
        positive=np.array([[float(b) for _, b in cells]]),
        mode=mode,
    )[0]
    assert measured == pytest.approx(float(exact), rel=2e-14, abs=1e-16)


def test_partition_refinement_preserves_atoms_and_nested_bounds():
    """Refining a common partition cannot silently smooth a jump or change its law."""
    truth = Curve(
        name="atom-gap",
        x=np.array([0.0, 0.25, 0.25, 0.75, 1.0]),
        y=np.array([0.0, 0.2, 0.7, 0.7, 1.0]),
    )
    labels = np.array([0, 1, 1, 0, 1])
    values = []
    for count in [4, 8, 16, 32]:
        a, b = partition(curves=[truth], cells=count)
        np.testing.assert_allclose([a.sum(), b.sum()], [1, 1], atol=1e-15)
        assert b[0, a == 0].sum() == pytest.approx(0.5)
        values.append(
            [
                likelihoods(labels=labels, negative=a, positive=b, mode=mode)[0]
                for mode in ["lower", "exact", "upper"]
            ]
        )
    values = np.array(values)
    np.testing.assert_allclose(values[:, 1], values[0, 1], rtol=2e-14)
    assert np.all(np.diff(values[:, 0]) >= -1e-15)
    assert np.all(np.diff(values[:, 2]) <= 1e-15)
    assert values[-1, 2] - values[-1, 0] < values[0, 2] - values[0, 0]


def test_plateau_inverse_places_rare_mass_only_in_its_declared_sliver():
    """Generalized inversion must not fill the long zero-density interval."""
    truth = Curve(
        name="sliver",
        x=np.array([0.0, 0.1, 0.59, 0.6, 1.0]),
        y=np.array([0.0, 0.99, 0.99, 1.0, 1.0]),
    )
    uniforms = np.array([0.0, 0.495, 0.99, 0.995, 1.0])
    np.testing.assert_allclose(
        truth.inverse(uniforms=uniforms), [0.0, 0.05, 0.1, 0.595, 0.6], atol=1e-14
    )
    np.testing.assert_allclose(
        truth.evaluate(grid=truth.inverse(uniforms=uniforms)), uniforms, atol=1e-14
    )


def test_predictors_normalize_even_with_adversarial_training():
    """A prefix learner cannot spend more than one unit of predictive mass."""
    all_paths = paths(n0=2, n1=3)
    training = [np.array([1, 1, 1, 0, 0])] * 50
    table = train_predictor(samples=training, n0=2, n1=3)
    q = [sequential_mass(labels=np.asarray(p), table=table) for p in all_paths]
    assert sum(q) == pytest.approx(1.0, abs=1e-14)
    assert q[0] > q[-1] * 20
    _, mixture = library()
    for name in ["uniform", "mixture", "sequential"]:
        mass = sum(
            (
                exact_predictor(
                    labels=np.asarray(p), name=name, mixture=mixture, table=table
                )
                for p in all_paths
            ),
            start=Q(0),
        )
        assert mass == 1


def test_exact_outer_contains_accepted_curves_for_every_small_rank_path():
    """All-completion boxes must enclose the exact confidence set, including jumps."""
    n = 2
    all_paths = paths(n0=n, n1=n)
    models = rational_models()[-4:]
    scores = np.array([statistics(labels=np.asarray(p)) for p in all_paths])
    laws = [model_law(n0=n, n1=n, xs=xs, ys=ys) for _, xs, ys in models]
    certified_rejections = 0
    for index, labels in enumerate(all_paths):
        band = outer(
            reject=exact_rejector(
                observed=np.asarray(labels), n0=n, n1=n, alpha=Q(1, 2), regions=True
            ),
            boxes=31,
        )
        reject = exact_rejector(
            observed=np.asarray(labels), n0=n, n1=n, alpha=Q(1, 2), regions=True
        )
        certified_rejections += reject(lower=(Q(0),) * 3, upper=(Q(0),) * 3)
        certified_rejections += reject(lower=(Q(1),) * 3, upper=(Q(1),) * 3)
        for (name, xs, ys), law in zip(models, laws, strict=True):
            accepted = all(
                min(
                    sum(
                        (
                            p
                            for p, s in zip(law, scores[:, k], strict=True)
                            if s <= scores[index, k]
                        ),
                        start=Q(0),
                    ),
                    sum(
                        (
                            p
                            for p, s in zip(law, scores[:, k], strict=True)
                            if s >= scores[index, k]
                        ),
                        start=Q(0),
                    ),
                )
                > Q(1, 12)
                for k in range(3)
            )
            if accepted:
                truth = Curve(
                    name=name,
                    x=np.asarray(xs, dtype=float),
                    y=np.asarray(ys, dtype=float),
                )
                assert metrics(
                    grid=band["grid"],
                    lower=band["lower"],
                    upper=band["upper"],
                    truth=truth,
                )["covered"]
    assert certified_rejections >= 2


def test_common_random_numbers_bound_each_path_and_candidate_p_value():
    """Stochastic ordering must hold per draw, not merely in a simulated average."""
    rng = np.random.default_rng(17)
    u0, u1 = rng.random((199, 5)), rng.random((199, 3))
    lo, hi = (Q(1, 8), Q(1, 4), Q(1, 2)), (Q(1, 2), Q(3, 4), Q(7, 8))
    low_step, high_step = endpoint_steps(lower=lo, upper=hi)
    lower = coupled_scores(
        uniforms0=u0, uniforms1=u1, knots=low_step[0], values=low_step[1], regions=True
    )
    upper = coupled_scores(
        uniforms0=u0,
        uniforms1=u1,
        knots=high_step[0],
        values=high_step[1],
        regions=True,
    )
    candidates = [
        Curve(
            name="center",
            x=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
            y=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        ),
        Curve(
            name="jump",
            x=np.array([0.0, 0.4, 0.4, 1.0]),
            y=np.array([0.0, 0.2, 0.5, 1.0]),
        ),
    ]
    for candidate in candidates:
        middle = []
        for negatives, uniforms in zip(u0, u1, strict=True):
            positive = candidate.inverse(uniforms=uniforms)
            labels = np.r_[np.zeros(5, dtype=int), np.ones(3, dtype=int)][
                np.argsort(np.r_[negatives, positive])
            ]
            middle.append(statistics(labels=labels))
        assert np.all(lower <= middle)
        assert np.all(middle <= upper)
    observed = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    reject = mc_rejector(
        observed=observed, uniforms0=u0, uniforms1=u1, alpha=Q(1, 2), regions=True
    )
    assert not reject(lower=(Q(0),) * 3, upper=(Q(1),) * 3)
    assert reject(lower=(Q(0),) * 3, upper=(Q(0),) * 3)


def test_anytime_outer_and_likelihood_caps_keep_unresolved_domain():
    """Zero work must be vacuous and extra refinement must never enlarge the hull."""
    labels = np.array([1, 1, 0, 0])
    previous = None
    for budget in [0, 1, 31, 511]:
        band = certified_likelihood_outer(labels=labels, cutoff=Q(2, 5), boxes=budget)
        if previous is not None:
            assert np.all(band["lower"] >= previous["lower"])
            assert np.all(band["upper"] <= previous["upper"])
        previous = band
    assert previous["boxes_rejected"] > 0
    children = split_box(lower=(Q(0),) * 3, upper=(Q(1),) * 3)
    for coordinates in combinations_with_replacement_for_test():
        assert any(
            all(
                lo <= v <= hi
                for lo, v, hi in zip(lower, coordinates, upper, strict=True)
            )
            for lower, upper in children
        )


def combinations_with_replacement_for_test():
    """Enumerate monotone stress points on the complete box, including its faces."""
    from itertools import combinations_with_replacement

    return combinations_with_replacement([Q(i, 4) for i in range(5)], 3)


def test_stitch_preserves_floor_and_exponent_nesting_even_at_seams():
    """The floor guarantee cannot be removed by lower-edge monotonic tightening."""
    grid = np.linspace(0, 1, 6)
    base = (
        np.array([0.0, 0.1, 0.2, 0.3, 0.5, 1.0]),
        np.array([0.2, 0.4, 0.6, 0.8, 0.9, 1.0]),
    )
    narrow = (
        np.array([0.0, 0.3, 0.4, 0.6, 0.8, 1.0]),
        np.array([0.1, 0.35, 0.5, 0.7, 0.85, 1.0]),
    )
    region = np.array([True, True, False, False, True, True])
    bands = [
        stitch(base=base, interior=interior, m3=base, grid=grid, region=region)
        for interior in [base, narrow]
    ]
    assert np.all(bands[1][0][region] <= base[0][region])
    assert np.all(bands[1][1][region] >= base[1][region])
    assert np.all(bands[0][0] <= bands[1][0])
    assert np.all(bands[0][1] >= bands[1][1])
    assert bands[1][0][2] > bands[0][0][2]
    np.testing.assert_allclose(
        metrics(
            grid=np.array([0.0, 0.5, 1.0]),
            lower=np.array([0.0, 0.2, 1.0]),
            upper=np.array([0.3, 0.7, 1.0]),
            truth=Curve(
                name="diagonal", x=np.array([0.0, 1.0]), y=np.array([0.0, 1.0])
            ),
        )["area"],
        0.75,
    )


def test_censored_crossings_are_not_fitted_endpoint_estimates():
    """An unobserved crossing must remain distinguishable from a measured bracket."""
    assert crossing(coverage=np.full(8, 0.99), target=0.95) == {
        "lower": 8.0,
        "upper": None,
        "censor": "right",
    }
    assert crossing(coverage=np.full(8, 0.94), target=0.95) == {
        "lower": None,
        "upper": 1.0,
        "censor": "left",
    }
    with pytest.raises(ValueError, match="Non-nested"):
        crossing(coverage=np.array([0.9, 0.95]), target=0.95)


@pytest.mark.parametrize("n0,n1", [(3, 7), (7, 3)])
def test_general_boundary_composition_matches_production_in_both_directions(n0, n1):
    """Marginal optimization must preserve the production order and endpoint map."""
    import fiducial_core

    alpha = 0.2
    bounds = [
        _ell_bounds(core=fiducial_core, n=n, alpha_class=1 - np.sqrt(1 - alpha))
        for n in [n0, n1]
    ]
    for path in paths(n0=n0, n1=n1):
        labels = np.asarray(path)
        actual = compose(labels=labels, bounds0=bounds[0], bounds1=bounds[1])
        expected = _m3_band_from_labels_rs(lab_s=labels, alpha=alpha)
        np.testing.assert_array_equal(actual, expected)
    lower, upper, probability = calibrated(n=n0, alpha=0.1, eta=0.5, theta=0.5)
    assert 0.9 <= probability <= 0.900001
    assert np.all(np.diff(lower) >= 0) and np.all(np.diff(upper) >= 0)


def test_checkpoint_discards_torn_tail_and_rejects_changed_configuration(tmp_path):
    """A killed unit may be repeated, but completed observations cannot be mixed."""
    store = Store(directory=tmp_path, config={"seed": 1})
    store.add(key="first", value=3)
    with store.path.open("ab") as output:
        output.write(b'{"key":"torn"')
    recovered = Store(directory=tmp_path, config={"seed": 1})
    recovered.add(key="first", value=99)
    recovered.add(key="second", value=4)
    assert [json.loads(line) for line in store.path.read_text().splitlines()] == [
        {"key": "first", "value": 3},
        {"key": "second", "value": 4},
    ]
    with pytest.raises(ValueError, match="incompatible resume"):
        Store(directory=tmp_path, config={"seed": 2})


def test_refined_reporting_grid_preserves_step_area_and_jump_coverage():
    """A finer comparator grid must not create an artificial width advantage."""
    from scripts.methods_exploration.common import resample

    source = np.array([0.0, 0.5, 1.0])
    lower, upper = np.array([0.0, 0.2, 1.0]), np.array([0.3, 0.7, 1.0])
    target = np.array([0.0, 0.01, 0.49, 0.5, 0.51, 0.99, 1.0])
    lo, hi = resample(source=source, lower=lower, upper=upper, target=target)
    np.testing.assert_array_equal(lo, [0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 1.0])
    np.testing.assert_array_equal(hi, [0.3, 0.7, 0.7, 0.7, 1.0, 1.0, 1.0])
    truth = Curve(
        name="atom",
        x=np.array([0.0, 0.49, 0.49, 1.0]),
        y=np.array([0.0, 0.0, 0.6, 1.0]),
    )
    coarse = metrics(grid=source, lower=lower, upper=upper, truth=truth)
    fine = metrics(grid=target, lower=lo, upper=hi, truth=truth)
    assert coarse["covered"] == fine["covered"]
    assert coarse["area"] == pytest.approx(fine["area"], abs=1e-15)


def test_frozen_schedule_uses_both_counts_and_clamps_finite_range():
    """Imbalance and the range boundaries must affect the proposed exponent."""
    from scripts.methods_exploration.interior import schedule_exponent

    candidate = {"coefficients": {"0.05": {"C": 2.5, "n_eff_range": [500, 5000]}}}
    assert schedule_exponent(n0=500, n1=500, alpha=0.05, candidate=candidate) == 2.5
    assert schedule_exponent(n0=100, n1=900, alpha=0.05, candidate=candidate) == 1
    assert schedule_exponent(n0=5001, n1=5001, alpha=0.05, candidate=candidate) == 1
    candidate = {"coefficients": {"0.05": {"a": 2.0, "b": 0.5}}}
    assert schedule_exponent(n0=400, n1=400, alpha=0.05, candidate=candidate) == 2
    assert schedule_exponent(
        n0=1_000_000, n1=1_000_000, alpha=0.05, candidate=candidate
    ) == pytest.approx(1.02)


def test_supervisor_deadline_preserves_incomplete_status_and_blocks_promotion(
    tmp_path, monkeypatch
):
    """A timed-out stage cannot become eligible through a preexisting summary."""
    import subprocess

    from scripts.methods_exploration import run as runner

    calls = []

    class Process:
        """Simulate a worker that requires termination to enforce its deadline."""

        returncode = -15

        def __init__(self, command, *, stdout, stderr):
            """Record routing and leave a tempting intermediate stage summary."""
            calls.append(command)
            directory = tmp_path / "interior"
            (directory / "summary.json").write_text(
                json.dumps({"complete": True, "candidate": {"eligible": True}})
            )
            self.terminated = False

        def wait(self, *, timeout=None):
            """Time out once, then acknowledge termination."""
            if not self.terminated:
                raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout)
            return -15

        def terminate(self):
            """Record that the supervisor actually stopped the worker."""
            self.terminated = True
            calls.append("terminated")

    monkeypatch.setattr(runner, "fingerprint", lambda: {"source": "fixed"})
    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--profile",
            "screen",
            "--track",
            "interior",
            "--minutes",
            ".001",
            "--out",
            str(tmp_path),
        ],
    )
    runner.main()
    assert calls[-1] == "terminated"
    assert calls[0][1:4] == ["-m", "scripts.methods_exploration.run", "--worker"]
    assert (
        json.loads((tmp_path / "interior/status.json").read_text())["status"]
        == "budget_exhausted"
    )
    assert "complete=False; eligible=False" in (tmp_path / "report.md").read_text()


@pytest.mark.parametrize(
    "tail_upper,complete,expected",
    [(1.0, True, True), (1.02, True, False), (1.0, False, False)],
    ids=["retain-useful-size", "veto-local-tail-loss", "incomplete-cannot-promote"],
)
def test_m3_promotion_is_per_design_and_cannot_hide_a_tail_penalty(
    tail_upper, complete, expected
):
    """Retain useful sizes subject to every local tail gate."""
    from scripts.methods_exploration.m3 import promotion_gate

    rows, cells = [], []
    for n, ratio in [(50, 0.94), (500, 1.0)]:
        for shape in ["a", "b"]:
            cells.append(
                {
                    "cell": (n, n, 0.05, shape),
                    "left_width": {"ci": [0.99, tail_upper if shape == "b" else 1.0]},
                    "right_width": {"ci": [0.99, 1.0]},
                }
            )
            for rep in range(4):
                rows.append(
                    {
                        "n0": n,
                        "n1": n,
                        "alpha": 0.05,
                        "shape": shape,
                        "rep": rep,
                        "winner": "optimized-shape",
                        "arms": {
                            "m3": {"area": 0.2},
                            "optimized": {"area": 0.2 * ratio},
                        },
                    }
                )
    result = promotion_gate(rows=rows, cells=cells, complete=complete, pilot=False)
    assert result["eligible"] is expected
    assert result["designs"][0]["eligible"] is expected
    assert result["designs"][1]["boundary"] == "m3"
    assert not promotion_gate(rows=rows, cells=cells, complete=True, pilot=True)[
        "eligible"
    ]
