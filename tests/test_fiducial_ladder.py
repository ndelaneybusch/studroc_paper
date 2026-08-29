"""Tests for the ladder-profile study kernel and its Python wrapper.

Three oracles: scipy (for the in-Rust Beta quantile), the production band
``fiducial_band_rs`` (same-seed exact parity of the reference-map path —
the "calibrate what ships" guarantee), and an independent numpy
reconstruction of the band assembly from the kernel's own raw tube edges.
"""

import numpy as np
import pytest

pytest.importorskip("fiducial_core")

import fiducial_core  # noqa: E402
from scipy.stats import beta as beta_dist  # noqa: E402

from studroc_paper.methods.fiducial_band import (  # noqa: E402
    _merged_labels,
    production_trim_rows,
)
from studroc_paper.methods.fiducial_band_rs import fiducial_band_rs  # noqa: E402
from studroc_paper.methods.fiducial_ladder import (  # noqa: E402
    khat_from_labels,
    ladder_profile,
    make_ladder,
)


def _labels_and_truth(n0: int, n1: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A rank-space replicate from a binormal-ish truth plus the truth on
    the native grid."""
    rng = np.random.default_rng(seed)
    grid = np.arange(n0 + 1) / n0
    rtrue = np.clip(grid**0.4, 0.0, 1.0)
    u = rng.random(n0)
    w = np.interp(rng.random(n1), np.linspace(0, 1, 512) ** 0.4, np.linspace(0, 1, 512))
    lab = np.concatenate([np.zeros(n0, np.uint8), np.ones(n1, np.uint8)])
    order = np.argsort(np.concatenate([u, w]), kind="stable")
    return lab[order], rtrue


@pytest.mark.parametrize(
    ("p", "a", "b"),
    [
        (0.95, 8, 493),
        (0.999, 1, 5000),
        (1 - 1 / 10001, 17, 4984),
        (0.5, 2500, 2500),
        (0.05, 40, 3),
        (1e-6, 3, 40),
        (0.9, 1, 1),
    ],
    ids=lambda v: f"{v:g}",
)
def test_inverse_beta_matches_scipy(p, a, b):
    """The in-Rust CP quantile must match scipy's beta.ppf to ~1e-11 — this
    is what keeps the study's allowance identical to production's."""
    ours = fiducial_core.inv_reg_incomplete_beta(p, a, b)
    ref = float(beta_dist.ppf(p, a, b))
    assert abs(ours - ref) < 1e-11, f"{ours} vs scipy {ref}"


@pytest.mark.parametrize(
    ("n0", "n1", "alpha", "c_exp"),
    [(120, 90, 0.05, 2.0), (120, 90, 0.5, 1.0), (2100, 300, 0.05, 2.0)],
    ids=["balanced-a05-c2", "balanced-a50-c1", "thinned-grid"],
)
def test_reference_path_matches_production_band_exactly(n0, n1, alpha, c_exp):
    """Same seed, same alpha_eff: the ladder kernel's reference-map stats
    must reproduce the production band verbatim (coverage indicator, area,
    and the raw tube edges at the realized depth), including on the thinned
    trim-grid path (K > 2001)."""
    lab_s, rtrue = _labels_and_truth(n0, n1, seed=5)
    m_draws = 1500

    # Drive the production wrapper deterministically: strictly descending
    # scores make _merged_labels the identity, so the kernel seed is the
    # wrapper rng's next integer draw.
    probe = np.random.default_rng(99)
    assert np.array_equal(
        _merged_labels(lab_s.astype(np.int64), -np.arange(len(lab_s), dtype=float),
                       "random", probe),
        lab_s.astype(np.int8),
    )
    seed = int(probe.integers(0, 2**64, dtype=np.uint64))

    _, lo, hi = fiducial_band_rs(
        lab_s.astype(np.int64),
        -np.arange(len(lab_s), dtype=np.float64),
        alpha=alpha,
        n_draws=m_draws,
        trim_exponent=c_exp,
        random_state=np.random.default_rng(99),
    )
    alpha_eff = 1.0 - (1.0 - alpha) ** c_exp
    prof = ladder_profile(
        lab_s,
        rtrue=rtrue,
        n_draws=m_draws,
        seed=seed,
        ladder=np.array([1, 3, 9]),
        alpha_effs=[alpha_eff],
        return_edges=True,
    )

    band_covered = bool(np.all(rtrue >= lo - 1e-12) and np.all(rtrue <= hi + 1e-12))
    assert bool(prof.ref_covered[0]) == band_covered
    assert abs(float(np.mean(hi - lo)) - prof.ref_area[0]) < 1e-9

    depths, edge_lo, edge_hi = prof.edges
    di = int(np.flatnonzero(depths == prof.ref_j[0])[0])
    trim_rows = production_trim_rows(n0 + 1)
    raw_lo, raw_hi, j_tube = fiducial_core.fiducial_trimmed_tube(
        lab_s,
        m_draws,
        alpha_eff,
        seed,
        0,
        None if trim_rows is None else trim_rows.astype(np.uint64),
    )
    assert j_tube == prof.ref_j[0]
    assert np.array_equal(edge_lo[di].astype(np.float64), raw_lo)
    assert np.array_equal(edge_hi[di].astype(np.float64), raw_hi)


def test_ladder_stats_match_independent_reconstruction():
    """Rebuild the allowance-augmented band in numpy from the kernel's own
    raw tube edges plus scipy's CP quantile, following the production
    operation order, and compare every per-depth statistic."""
    lab_s, rtrue = _labels_and_truth(60, 45, seed=3)
    n1 = int(lab_s.sum())
    n_grid = len(lab_s) - n1 + 1
    m_draws = 400
    ladder = np.array([1, 2, 5, 13, 40, 120, 200])
    prof = ladder_profile(
        lab_s,
        rtrue=rtrue,
        n_draws=m_draws,
        seed=17,
        ladder=ladder,
        alpha_effs=[0.0975],
        return_edges=True,
    )
    khat = khat_from_labels(lab_s)
    depths, edge_lo, edge_hi = prof.edges
    for li, j in enumerate(ladder):
        di = int(np.flatnonzero(depths == j)[0])
        lo = np.clip(edge_lo[di].astype(np.float64), 0.0, 1.0)
        hi = np.clip(edge_hi[di].astype(np.float64), 0.0, 1.0)
        level = j / (m_draws + 1.0)
        cp = np.ones(n_grid)
        interior = khat < n1
        cp[interior] = beta_dist.ppf(
            1.0 - level, khat[interior] + 1, n1 - khat[interior]
        )
        upper = np.maximum.accumulate(np.maximum(hi, cp))
        lower = lo.copy()
        lower[khat == 0] = 0.0
        upper = np.clip(upper, 0.0, 1.0)
        lower[0] = 0.0
        upper[-1] = 1.0

        d_lo = lower - rtrue
        d_hi = rtrue - upper
        viol = np.maximum(np.maximum(d_lo, d_hi), 0.0)
        covered = not ((d_lo > 1e-12).any() or (d_hi > 1e-12).any())
        assert bool(prof.covered[li]) == covered, f"coverage at j={j}"
        assert bool(prof.viol_low[li]) == bool((d_lo > 1e-12).any())
        assert bool(prof.viol_high[li]) == bool((d_hi > 1e-12).any())
        assert abs(prof.area[li] - float(np.mean(upper - lower))) < 1e-9
        assert abs(prof.area_raw[li] - float(np.mean(hi - lo))) < 1e-9
        if viol.max() > 1e-12:
            assert prof.miss_depth[li] == pytest.approx(viol.max(), abs=1e-10)
            assert prof.worst_k[li] == int(np.argmax(viol))
        else:
            assert prof.worst_k[li] == -1


def test_depth_distribution_and_cdf_are_consistent():
    lab_s, rtrue = _labels_and_truth(50, 40, seed=11)
    prof = ladder_profile(lab_s, rtrue=rtrue, n_draws=600, seed=1, alpha_effs=[0.1])
    assert len(prof.depths_sorted) == 600
    assert np.all(np.diff(prof.depths_sorted.astype(np.int64)) >= 0)
    cdf = prof.depth_cdf_at(prof.ladder)
    manual = np.array([(prof.depths_sorted < j).sum() for j in prof.ladder])
    assert np.array_equal(cdf, manual)
    # Production depth rule: floor-quantile of the sorted depths, clamped.
    q = prof.depths_sorted[int(np.floor(0.1 * 600))]
    assert prof.ref_j[0] == int(np.clip(q, 1, 300))


def test_trim_rows_weakly_deepen_and_leave_truth_depth_alone():
    lab_s, rtrue = _labels_and_truth(90, 70, seed=2)
    full = ladder_profile(
        lab_s, rtrue=rtrue, n_draws=500, seed=9, alpha_effs=[0.1], trim_rows=None
    )
    rows = np.arange(0, 91, 5)
    thin = ladder_profile(
        lab_s, rtrue=rtrue, n_draws=500, seed=9, alpha_effs=[0.1], trim_rows=rows
    )
    assert np.all(thin.depths_sorted >= full.depths_sorted)
    assert thin.ref_j[0] >= full.ref_j[0]
    assert thin.truth_depth_low == full.truth_depth_low
    assert thin.truth_depth_high == full.truth_depth_high


def test_make_ladder_spans_and_orders():
    for m in (150, 3000, 84_000):
        ladder = make_ladder(m)
        assert ladder[0] == 1
        assert ladder[-1] == m // 2
        assert np.all(np.diff(ladder) > 0)
        # Dense head: every depth 1..40 present (alpha-resolution at deep trims).
        assert np.array_equal(ladder[:40], np.arange(1, 41))


def test_khat_from_labels_matches_direct_count():
    rng = np.random.default_rng(0)
    lab_s = (rng.random(300) < 0.4).astype(np.uint8)
    lab_s[0], lab_s[-1] = 0, 1  # both classes present
    n1 = int(lab_s.sum())
    khat = khat_from_labels(lab_s)
    neg_positions = np.flatnonzero(lab_s == 0)
    expected = [int(lab_s[:pos].sum()) for pos in neg_positions] + [n1]
    assert np.array_equal(khat, expected)
    assert np.all(np.diff(khat) >= 0)


def test_wrapper_rejects_malformed_inputs():
    lab_s, rtrue = _labels_and_truth(30, 20, seed=1)
    with pytest.raises(ValueError, match="rtrue"):
        ladder_profile(lab_s, rtrue=rtrue[:-1], n_draws=200, seed=0)
    with pytest.raises(ValueError, match="ladder"):
        ladder_profile(
            lab_s, rtrue=rtrue, n_draws=200, seed=0, ladder=np.array([5, 5])
        )
    with pytest.raises(ValueError, match="alpha_eff"):
        ladder_profile(lab_s, rtrue=rtrue, n_draws=200, seed=0, alpha_effs=[1.5])
    with pytest.raises(ValueError, match="trim_rows"):
        ladder_profile(lab_s, rtrue=rtrue, n_draws=200, seed=0, trim_rows="bogus")
