"""Window routing, calibration exposure, and small-n production geometry checks."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.methods_exploration import interior, small_n  # noqa: E402
from scripts.methods_exploration.common import Curve, Store, interval  # noqa: E402
from scripts.methods_exploration.hybrid import (  # noqa: E402
    floor_components,
    geometry,
    hull,
    window_status,
)
from studroc_paper.methods.fiducial_band import _auto_n_draws  # noqa: E402


@pytest.mark.parametrize(
    "n,left_end,right_start,expected",
    [
        (1000, 19, 951, True),
        (1000, 20, 951, False),
        (1000, 19, 950, False),
        (10, 0, 10, False),
        (100, 1, 96, True),
    ],
    ids=["strict-interior", "touch-left", "touch-right", "off-grid-guards", "aligned"],
)
def test_window_requires_unprotected_endpoint_guards(
    n, left_end, right_start, expected
):
    """Tails touching the window or its bracketing columns disable the trim."""
    index = np.arange(n + 1)
    status = window_status(
        grid=index / n, region=(index <= left_end) | (index >= right_start)
    )
    assert status["eligible"] is expected


def test_small_n_floor_uses_production_mass_laws_and_can_fill_the_grid():
    """Even no trailing negatives can leave no usable interior at small n."""
    labels = np.r_[np.zeros(10, dtype=int), np.ones(10, dtype=int)]
    draws = _auto_n_draws(n_grid=11, alpha_eff=0.05)
    left, right = floor_components(labels=labels, draws=draws, depth=5)
    result = geometry(labels=labels, left=left, right=right)
    assert result["left_end_fpr"] == result["right_start_fpr"] == 0.6
    assert result["fully_floored"] and result["unfloored_points"] == 0
    left, right = floor_components(labels=labels, draws=draws, depth=86)
    result = geometry(labels=labels, left=left, right=right)
    assert result["left_end_fpr"] == 0.3
    assert result["right_start_fpr"] == 0.6
    assert result["unfloored_points"] == 2
    assert result["unfloored_cell_fraction"] == 0.1
    assert not result["window"]["eligible"]


def test_eligible_stitch_is_nested_and_preserves_floor_containment():
    """Actual interior trimming narrows the band without withdrawing tail protection."""
    grid = np.arange(1001) / 1000
    base = (np.maximum(0, grid - 0.2), np.minimum(1, grid + 0.2))
    narrow = (np.maximum(0, grid - 0.1), np.minimum(1, grid + 0.1))
    region = (grid <= 0.01) | (grid >= 0.99)
    result = interior.stitch(
        base=base, interior=narrow, m3=base, grid=grid, region=region
    )
    assert np.all(result[0] >= base[0]) and np.all(result[1] <= base[1])
    np.testing.assert_array_equal(result[0][region], base[0][region])
    np.testing.assert_array_equal(result[1][region], base[1][region])
    assert result[0][500] == 0.4 and result[1][500] == 0.6


def test_only_eligible_alphas_reach_the_interior_kernel(monkeypatch):
    """Fallback samples must skip interior computation and depth estimation."""
    labels = np.tile([0, 1], 1000)
    grid = np.arange(1001) / 1000
    base = (np.maximum(0, grid - 0.2), np.minimum(1, grid + 0.2))
    inner = (np.maximum(0, grid - 0.1), np.minimum(1, grid + 0.1))
    calls = []

    def profile(**kwargs):
        """Return distinctive full-grid and interior depths and capture routing."""
        calls.append(kwargs)
        if len(calls) == 1:
            return [base, base], np.array([5, 10])
        return [inner] * len(interior.EXPONENTS), np.full(len(interior.EXPONENTS), 50)

    def components(*, labels, draws, depth):
        """Make only alpha .5 eligible, independently of every candidate C."""
        return grid <= (0.03 if depth == 5 else 0.005), grid >= 0.99

    monkeypatch.setattr(interior, "fiducial_edges", profile)
    monkeypatch.setattr(interior, "floor_components", components)
    monkeypatch.setattr(interior, "m3_edges", lambda **kw: (grid, *base))
    result = interior.measure(
        labels=labels,
        truth=Curve(name="diagonal", x=np.array([0.0, 1.0]), y=np.array([0.0, 1.0])),
        draws=2000,
        seed=3,
        threads=1,
    )
    assert len(calls) == 2
    np.testing.assert_allclose(calls[1]["alphas"], 1 - 0.5**interior.EXPONENTS)
    assert result["0.05"]["interior_depths"] is None
    assert result["0.5"]["interior_depths"] == [50] * len(interior.EXPONENTS)
    for c in interior.EXPONENTS:
        assert result["0.05"]["arms"][f"C{c:g}"] == result["0.05"]["arms"]["floor"]
    assert result["0.5"]["arms"]["C1"]["area"] < result["0.5"]["arms"]["floor"]["area"]


def test_no_eligible_samples_do_not_create_an_infinite_cstar(tmp_path):
    """Identical hybrid fallbacks are no exposure, not evidence of a common law."""
    store = Store(directory=tmp_path, config={})
    metric = {"covered": True, "area": 0.2, "left_width": 0.3, "right_width": 0.1}
    arms = {
        name: metric
        for name in ["raw", "floor", "m3", *[f"C{c:g}" for c in interior.EXPONENTS]]
    }
    for rep in range(3):
        store.add(
            key=str(rep),
            n=100,
            ratio=0.5,
            shape="diagonal",
            alphas={
                str(alpha): {
                    "arms": arms,
                    "window": {"eligible": False},
                    "depth": 10,
                    "interior_depths": None,
                }
                for alpha in [0.05, 0.5]
            },
        )
    interior.summarize(store=store, pilot=False, expected=3)
    import json

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert all(
        c["eligible_only"] is None and c["calibration_status"] == "no_exposure"
        for c in summary["cells"]
    )
    assert not summary["candidate"]["eligible"]
    assert all(
        s["status"] == "insufficient_eligible_data" for s in summary["shape_spreads"]
    )


def test_replication_can_support_nominal_with_observed_failures():
    """The precision plan must permit promotion without requiring zero misses."""
    for count in [interior.LARGE_REPLICATES, interior.SMALL_REPLICATES]:
        assert interval(successes=round(0.98 * count), count=count)[0] > 0.95
    assert interval(successes=520, count=1000)[0] < 0.5


def test_absent_small_n_interior_is_recorded_as_absent():
    """No interior must not be summarized as perfectly covered at zero width."""
    score = small_n.regional_metrics(
        mask=np.zeros(11, dtype=bool),
        lower=np.zeros(11),
        upper=np.ones(11),
        values=np.linspace(0, 1, 11),
    )
    assert score == {
        "points": 0,
        "cell_fraction": 0.0,
        "below": None,
        "above": None,
        "miss": None,
        "area": 0.0,
        "mean_width": None,
    }
    raw = (np.zeros(11), np.full(11, 0.8))
    m3 = (np.full(11, 0.1), np.ones(11))
    lo, hi = hull(raw=raw, m3=m3, region=np.ones(11, dtype=bool))
    np.testing.assert_array_equal(lo, np.zeros(11))
    np.testing.assert_array_equal(hi, np.ones(11))


@pytest.mark.parametrize("n0,n1", [(10, 10), (10, 50), (50, 10)])
def test_small_n_cloud_budgets_match_production_and_endpoint_atoms_survive(n0, n1):
    """Small-n exploration must not shrink the floor by substituting a tiny cloud."""
    truth = small_n.study_curve(name="endpoint_atoms", n0=n0, n1=n1)
    labels, _ = small_n.sample_details(
        truth=truth, n0=n0, n1=n1, rng=np.random.default_rng(63)
    )
    result = small_n.measure(labels=labels, truth=truth, seed=42, threads=1)
    assert truth.evaluate(grid=np.array([0.0]))[0] == 0.15
    for alpha in [0.05, 0.5]:
        data = result[str(alpha)]
        assert data["draws"] == _auto_n_draws(n_grid=n0 + 1, alpha_eff=alpha)
        assert data["arms"]["hybrid"]["area"] >= data["arms"]["raw"]["area"]
        assert not data["geometry"]["window"]["eligible"]


def test_schedule_geometry_fallback_precedes_coefficient_lookup():
    """An overlapping window cannot activate a frozen exponent or require a fit."""
    assert (
        interior.schedule_exponent(
            n0=10, n1=50, alpha=0.05, candidate={}, window_eligible=False
        )
        == 1.0
    )


def test_small_n_sampling_reproduces_missing_sliver_probability():
    """Conditional diagnostics must retain the latent rare-mass sampling event."""
    n0, n1 = 30, 20
    truth = small_n.study_curve(name="interior_sliver", n0=n0, n1=n1)
    rng = np.random.default_rng(824)
    absent = sum(
        small_n.sample_details(truth=truth, n0=n0, n1=n1, rng=rng)[1][
            "no_rare_observation"
        ]
        for _ in range(3000)
    )
    expected = (1 - 0.8 / n1) ** n1
    lo, hi = interval(successes=absent, count=3000, confidence=0.999)
    assert lo < expected < hi
    assert 0.3 < absent / 3000 < 0.6
