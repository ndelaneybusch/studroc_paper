"""Tests for the localized exact M3 floor on the fiducial ROC band."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom

from studroc_paper.methods import (
    M3Floor,
    exact_left_cutoff,
    exact_right_start,
    floor_region,
    stitch_m3_floor,
)
from studroc_paper.methods.fiducial_band import _merged_labels
from studroc_paper.methods.hybrid_floor import (
    chord_probability,
    resolve_floor,
    stage_f_left_cutoff,
    stage_f_right_start,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/c_calibration"))

from stage_f_core import (  # noqa: E402
    empirical_observables,
    frontier_region_masks,
    stitch_hybrid,
)

# (n0, n_draws, trim_depth): production budgets at alpha .05 and .5, plus the
# saturated depth j = 1 that the Stage F rule implicitly assumes.
CLOUD_CASES = [
    pytest.param(10, 2000, 5, id="n10-a05"),
    pytest.param(10, 2000, 86, id="n10-a50"),
    pytest.param(30, 2434, 5, id="n30-a05"),
    pytest.param(100, 3347, 5, id="n100-a05"),
    pytest.param(500, 5158, 5, id="n500-a05"),
    pytest.param(500, 2000, 31, id="n500-a50"),
    pytest.param(10000, 11575, 5, id="n10k-a05"),
    pytest.param(10000, 2000, 14, id="n10k-a50"),
    pytest.param(500, 5158, 1, id="n500-saturated"),
    pytest.param(500, 20000, 1, id="n500-saturated-maxM"),
]


def _exact_reach(*, n0: int, n_draws: int, trim_depth: int) -> int:
    """Smallest grid index at which the expected chord count drops to ``j``."""
    return next(
        k
        for k in range(n0 + 1)
        if n_draws * chord_probability(n0=n0, k=k) <= trim_depth
    )


def _descending_labels(
    *, n0: int, n1: int, shift: float, df: float | None, seed: int
) -> np.ndarray:
    """Tie-free 0/1 labels in descending score order from a two-sample draw."""
    rng = np.random.default_rng(seed)
    if df is None:
        neg, pos = rng.normal(0, 1, n0), rng.normal(shift, 1, n1)
    else:
        neg, pos = rng.standard_t(df, n0), rng.standard_t(df, n1) + shift
    y_true = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    return _merged_labels(y_true, np.concatenate([neg, pos]), "random", rng)


@pytest.mark.parametrize(("n0", "n_draws", "trim_depth"), CLOUD_CASES)
@pytest.mark.parametrize("eps", [1e-2, 1e-3, 1e-4], ids=["e-2", "e-3", "e-4"])
def test_exact_left_cutoff_is_minimal_and_never_below_the_exact_reach(
    n0, n_draws, trim_depth, eps
):
    k = exact_left_cutoff(n0=n0, n_draws=n_draws, trim_depth=trim_depth, eps=eps)
    tail_at = binom.sf(trim_depth - 1, n_draws, chord_probability(n0=n0, k=k))
    assert tail_at <= eps
    if k > 0:
        tail_before = binom.sf(
            trim_depth - 1, n_draws, chord_probability(n0=n0, k=k - 1)
        )
        assert tail_before > eps
    assert k >= _exact_reach(n0=n0, n_draws=n_draws, trim_depth=trim_depth)
    if n0 >= 100:
        assert k >= math.ceil(math.log((n_draws + 1) / trim_depth))


@pytest.mark.parametrize(
    ("n0", "n_draws", "trim_depth", "expected"),
    [
        pytest.param(500, 5158, 5, 9, id="n500-a05"),
        pytest.param(10000, 11575, 5, 10, id="n10k-a05"),
        pytest.param(500, 2000, 31, 5, id="n500-a50"),
        pytest.param(10000, 2000, 14, 6, id="n10k-a50"),
        pytest.param(30, 2434, 5, 8, id="n30-a05"),
        pytest.param(10, 2000, 5, 6, id="n10-a05"),
        pytest.param(10, 2000, 86, 3, id="n10-a50"),
    ],
)
def test_exact_left_cutoff_reproduces_the_declared_values(
    n0, n_draws, trim_depth, expected
):
    assert (
        exact_left_cutoff(n0=n0, n_draws=n_draws, trim_depth=trim_depth, eps=1e-3)
        == expected
    )


def test_exact_left_cutoff_undercuts_the_log_form_only_where_exp_overstates_reach():
    """At n0 = 10 the log approximation e^-k exceeds (1 - k/n0)^n0 materially."""
    n0, n_draws, trim_depth = 10, 2000, 86
    exact = exact_left_cutoff(n0=n0, n_draws=n_draws, trim_depth=trim_depth, eps=1e-3)
    log_form = math.ceil(math.log((n_draws + 1) / trim_depth))
    assert exact == 3 and log_form == 4
    assert chord_probability(n0=n0, k=3) < 0.6 * math.exp(-3)


def test_exact_left_cutoff_is_monotone_in_budget_depth_and_cloud_size():
    n0 = 500
    by_eps = [
        exact_left_cutoff(n0=n0, n_draws=5158, trim_depth=5, eps=eps)
        for eps in (0.3, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6)
    ]
    assert by_eps == sorted(by_eps) and by_eps[0] < by_eps[-1]
    by_depth = [
        exact_left_cutoff(n0=n0, n_draws=5158, trim_depth=j, eps=1e-3)
        for j in (1, 2, 5, 10, 31, 100)
    ]
    assert by_depth == sorted(by_depth, reverse=True) and by_depth[0] > by_depth[-1]
    by_cloud = [
        exact_left_cutoff(n0=n0, n_draws=m, trim_depth=5, eps=1e-3)
        for m in (2000, 5158, 11575, 20000)
    ]
    assert by_cloud == sorted(by_cloud) and by_cloud[0] < by_cloud[-1]


def test_exact_left_cutoff_terminates_at_n0_when_no_budget_is_reachable():
    assert exact_left_cutoff(n0=3, n_draws=20000, trim_depth=1, eps=1e-4) == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"n0": 0, "n_draws": 100, "trim_depth": 1, "eps": 0.1}, id="n0"),
        pytest.param({"n0": 5, "n_draws": 0, "trim_depth": 1, "eps": 0.1}, id="M"),
        pytest.param({"n0": 5, "n_draws": 100, "trim_depth": 0, "eps": 0.1}, id="j"),
        pytest.param({"n0": 5, "n_draws": 100, "trim_depth": 1, "eps": 1.0}, id="eps"),
    ],
)
def test_exact_left_cutoff_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        exact_left_cutoff(**kwargs)


@pytest.mark.parametrize(
    ("run_length", "extra_ranks"),
    [(0, 4), (1, 5), (5, 7), (25, 12), (100, 19), (250, 23)],
    ids=lambda v: str(v),
)
def test_exact_right_margin_reproduces_the_theory_table(run_length, extra_ranks):
    """Theory doc section 10.2, n0 = 500 and delta = .025."""
    n0 = 500
    start = exact_right_start(n0=n0, run_length=run_length, delta=0.025)
    assert (n0 - run_length) - start == extra_ranks


@pytest.mark.parametrize("run_length", [500, 499], ids=["K=n0", "K=n0-1"])
def test_exact_right_start_covers_the_whole_grid_on_complete_separation(run_length):
    assert exact_right_start(n0=500, run_length=run_length, delta=0.025) == 0


def test_exact_right_start_is_monotone_in_delta_and_run_length():
    n0 = 2000
    by_delta = [
        exact_right_start(n0=n0, run_length=20, delta=d)
        for d in (0.4, 0.1, 0.025, 1e-3, 1e-5)
    ]
    assert by_delta == sorted(by_delta, reverse=True) and by_delta[0] > by_delta[-1]
    by_run = [
        exact_right_start(n0=n0, run_length=k, delta=0.025)
        for k in (0, 1, 10, 100, 1000, 1999, 2000)
    ]
    assert by_run == sorted(by_run, reverse=True) and by_run[-1] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"n0": 100, "run_length": 101, "delta": 0.025}, id="K>n0"),
        pytest.param({"n0": 100, "run_length": -1, "delta": 0.025}, id="K<0"),
        pytest.param({"n0": 100, "run_length": 5, "delta": 0.5}, id="delta"),
    ],
)
def test_exact_right_start_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        exact_right_start(**kwargs)


@pytest.mark.parametrize(
    ("n0", "n1", "shift", "df"),
    [
        pytest.param(50, 40, 1.0, None, id="small-gaussian"),
        pytest.param(300, 120, 3.0, None, id="separated-gaussian"),
        pytest.param(1000, 1000, 0.0, None, id="diagonal"),
        pytest.param(80, 900, 2.5, 2.0, id="positive-majority-t2"),
        pytest.param(900, 80, 6.0, 3.0, id="negative-majority-t3"),
    ],
)
@pytest.mark.parametrize("n_draws", [6730, 14992], ids=["M6730", "M14992"])
def test_stage_f_rule_reproduces_the_frozen_frontier_masks(n0, n1, shift, df, n_draws):
    labels = _descending_labels(n0=n0, n1=n1, shift=shift, df=df, seed=n0 + n1)
    observables, khat = empirical_observables(labels)
    left, right = frontier_region_masks(
        "frontier_floor_v1", observables=observables, khat=khat, m_draws=n_draws
    )
    region = floor_region(
        khat=khat, n_draws=n_draws, trim_depth=7, floor=M3Floor(rule="stage_f")
    )
    assert np.array_equal(region, left | right)
    assert stage_f_left_cutoff(n0=n0, n_draws=n_draws) == min(
        n0, math.ceil(math.log(n_draws + 1))
    )
    k_sat = int(np.flatnonzero(khat == n1)[0])
    assert stage_f_right_start(n0=n0, run_length=n0 - k_sat) == int(
        np.flatnonzero(right)[0]
    )


def test_exact_and_stage_f_rules_agree_within_one_point_at_alpha_05():
    """The frozen left cutoff caps the expected chord count at one, in e^-k form."""
    for n0, n_draws in ((100, 3347), (500, 5158), (2000, 7496), (10000, 11575)):
        exact = exact_left_cutoff(n0=n0, n_draws=n_draws, trim_depth=5, eps=1e-3)
        frozen = stage_f_left_cutoff(n0=n0, n_draws=n_draws)
        assert abs(frozen - exact) <= 1
        assert n_draws * chord_probability(n0=n0, k=frozen) <= 1.0


def test_stitch_reproduces_the_stage_f_widening_closure_and_contains_both_parents():
    rng = np.random.default_rng(3)
    for _ in range(20):
        n = int(rng.integers(5, 400))
        raw_lower = np.sort(rng.uniform(0, 0.9, n))
        raw_upper = np.clip(raw_lower + rng.uniform(0, 0.3, n), 0, 1)
        m3_lower = np.clip(raw_lower - rng.uniform(0, 0.2, n), 0, 1)
        m3_upper = np.clip(raw_upper + rng.uniform(0, 0.2, n), 0, 1)
        region = rng.random(n) < 0.3
        region[[0, -1]] = True
        lower, upper = stitch_m3_floor(
            lower=raw_lower,
            upper=raw_upper,
            m3_lower=m3_lower,
            m3_upper=m3_upper,
            region=region,
        )
        ref_lower, ref_upper = stitch_hybrid(
            raw_lower, raw_upper, m3_lower, m3_upper, region, closure="widening"
        )
        assert np.array_equal(lower, ref_lower) and np.array_equal(upper, ref_upper)
        assert np.all(lower <= raw_lower) and np.all(upper >= raw_upper)
        assert np.all(lower[region] <= m3_lower[region])
        assert np.all(upper[region] >= m3_upper[region])
        assert np.all(np.diff(lower) >= 0) and np.all(np.diff(upper) >= 0)


def test_stitch_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        stitch_m3_floor(
            lower=np.zeros(4),
            upper=np.ones(4),
            m3_lower=np.zeros(5),
            m3_upper=np.ones(5),
            region=np.ones(4, dtype=bool),
        )


@pytest.mark.parametrize("rule", ["exact", "stage_f"])
def test_floor_region_is_the_whole_grid_under_complete_separation(rule):
    khat = np.full(201, 60, dtype=np.int64)
    region = floor_region(
        khat=khat, n_draws=5000, trim_depth=5, floor=M3Floor(rule=rule)
    )
    assert region.all()


def test_floor_region_protects_both_ends_and_leaves_the_interior_alone():
    n0, n1 = 2000, 300
    khat = np.minimum(n1, (np.arange(n0 + 1) * n1 * 1.3 / n0).astype(np.int64))
    khat[-3:] = n1
    region = floor_region(khat=khat, n_draws=10000, trim_depth=5, floor=M3Floor())
    assert region[0] and region[-1]
    left_end = exact_left_cutoff(n0=n0, n_draws=10000, trim_depth=5, eps=1e-3)
    assert region[: left_end + 1].all() and not region[left_end + 1]
    k_sat = int(np.flatnonzero(khat == n1)[0])
    assert region[k_sat:].all()
    right_start = exact_right_start(n0=n0, run_length=n0 - k_sat, delta=0.025)
    assert region[right_start:].all() and not region[right_start - 1]
    assert not region[n0 // 2]


@pytest.mark.parametrize(
    "khat",
    [
        pytest.param(np.array([0, 3, 2, 5]), id="nonmonotone"),
        pytest.param(np.array([-1, 0, 5]), id="negative"),
        pytest.param(np.array([0, 0, 0]), id="no-positives"),
        pytest.param(np.array([4]), id="too-short"),
    ],
)
def test_floor_region_rejects_invalid_count_maps(khat):
    with pytest.raises(ValueError):
        floor_region(khat=khat, n_draws=2000, trim_depth=5, floor=M3Floor())


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"rule": "frontier"}, id="rule"),
        pytest.param({"eps": 0.0}, id="eps-zero"),
        pytest.param({"eps": 1.0}, id="eps-one"),
        pytest.param({"delta": 0.5}, id="delta"),
        pytest.param({"alpha": 1.0}, id="alpha"),
        pytest.param({"split_ratio": 1.0}, id="split"),
    ],
)
def test_m3_floor_settings_are_validated(kwargs):
    with pytest.raises(ValueError):
        M3Floor(**kwargs)


def test_resolve_floor_maps_the_public_toggle():
    assert resolve_floor(False) is None
    assert resolve_floor(True) == M3Floor()
    custom = M3Floor(rule="stage_f", alpha=0.01)
    assert resolve_floor(custom) is custom


# ---------------------------------------------------------------------------
# Production integration
# ---------------------------------------------------------------------------

_core = pytest.importorskip("fiducial_core")

from studroc_paper.methods import (  # noqa: E402
    fiducial_band,
    fiducial_band_rs,
    khat_from_labels,
    m3_band_rs,
)


@pytest.fixture(
    scope="module",
    params=[
        pytest.param((150, 150, 1.5, None), id="gaussian"),
        pytest.param((400, 100, 6.0, 2.0), id="convex-corner-t2"),
    ],
)
def dataset(request):
    """Class-labelled scores; the t(2) case is inside the C = 1 failure wedge."""
    n0, n1, shift, df = request.param
    rng = np.random.default_rng(2026)
    if df is None:
        neg, pos = rng.normal(0, 1, n0), rng.normal(shift, 1, n1)
    else:
        neg, pos = rng.standard_t(df, n0), rng.standard_t(df, n1) + shift
    y_true = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    return y_true, np.concatenate([neg, pos])


def _saturation_index(y_true, y_score, seed):
    labels = _merged_labels(y_true, y_score, "random", np.random.default_rng(seed))
    khat = khat_from_labels(labels)
    return int(np.flatnonzero(khat == khat[-1])[0])


def test_floored_band_contains_the_raw_band_and_m3_at_both_ends(dataset):
    y_true, y_score = dataset
    seed = 11
    _, raw_lo, raw_hi = fiducial_band_rs(y_true, y_score, random_state=seed)
    fpr, lo, hi = fiducial_band_rs(y_true, y_score, random_state=seed, m3_floor=True)
    _, m3_lo, m3_hi = m3_band_rs(y_true, y_score, random_state=seed)
    assert fpr.shape == lo.shape == hi.shape == raw_lo.shape
    assert np.all(lo <= raw_lo) and np.all(hi >= raw_hi)
    assert np.all(np.diff(lo) >= 0) and np.all(np.diff(hi) >= 0)
    assert lo[0] == 0.0 and hi[-1] == 1.0
    k_sat = _saturation_index(y_true, y_score, seed)
    ends = np.r_[np.arange(4), np.arange(k_sat, len(fpr))]
    assert np.all(lo[ends] <= m3_lo[ends]) and np.all(hi[ends] >= m3_hi[ends])
    assert np.any(lo < raw_lo)


def test_floor_widens_monotonically_in_the_m3_level(dataset):
    y_true, y_score = dataset
    seed = 5
    _, lo_a, hi_a = fiducial_band_rs(y_true, y_score, random_state=seed, m3_floor=True)
    _, lo_b, hi_b = fiducial_band_rs(
        y_true, y_score, random_state=seed, m3_floor=M3Floor(alpha=0.01)
    )
    assert np.all(lo_b <= lo_a) and np.all(hi_b >= hi_a)
    assert np.any(lo_b < lo_a) or np.any(hi_b > hi_a)


@pytest.mark.parametrize("alpha", [0.05, 0.5], ids=["a05", "a50"])
def test_stage_f_toggle_reproduces_the_frozen_construction_bit_for_bit(dataset, alpha):
    """Production parents composed by the frozen Stage F code equal the toggle."""
    y_true, y_score = dataset
    seed, n_draws = 8, 6730
    labels = _merged_labels(y_true, y_score, "random", np.random.default_rng(seed))
    observables, khat = empirical_observables(labels)
    _, raw_lo, raw_hi = fiducial_band_rs(
        y_true, y_score, alpha=alpha, n_draws=n_draws, random_state=seed
    )
    _, m3_lo, m3_hi = m3_band_rs(y_true, y_score, alpha=alpha, random_state=seed)
    left, right = frontier_region_masks(
        "frontier_floor_v1", observables=observables, khat=khat, m_draws=n_draws
    )
    want_lo, want_hi = stitch_hybrid(
        raw_lo, raw_hi, m3_lo, m3_hi, left | right, closure="widening"
    )
    _, got_lo, got_hi = fiducial_band_rs(
        y_true,
        y_score,
        alpha=alpha,
        n_draws=n_draws,
        random_state=seed,
        m3_floor=M3Floor(rule="stage_f"),
    )
    assert np.array_equal(got_lo, want_lo) and np.array_equal(got_hi, want_hi)
    assert np.any(got_lo < raw_lo)


def test_reference_implementation_supports_the_same_floor():
    rng = np.random.default_rng(9)
    n0, n1 = 60, 60
    y_true = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    y_score = np.concatenate([rng.standard_t(2, n0), rng.standard_t(2, n1) + 5.0])
    raw = fiducial_band(y_true, y_score, n_draws=2000, random_state=4)
    floored = fiducial_band(
        y_true, y_score, n_draws=2000, random_state=4, m3_floor=True
    )
    assert np.all(floored[1] <= raw[1]) and np.all(floored[2] >= raw[2])
    assert np.any(floored[1] < raw[1])
    assert floored[1][0] == 0.0 and floored[2][-1] == 1.0
    assert np.all(np.diff(floored[1]) >= 0) and np.all(np.diff(floored[2]) >= 0)


def test_output_grid_resampling_preserves_the_floor_containment(dataset):
    y_true, y_score = dataset
    seed = 13
    grid, raw_lo, raw_hi = fiducial_band_rs(y_true, y_score, k=41, random_state=seed)
    grid_f, lo, hi = fiducial_band_rs(
        y_true, y_score, k=41, random_state=seed, m3_floor=True
    )
    assert np.array_equal(grid, grid_f) and len(grid) == 41
    assert np.all(lo <= raw_lo) and np.all(hi >= raw_hi)
    assert lo[0] == 0.0 and hi[-1] == 1.0
