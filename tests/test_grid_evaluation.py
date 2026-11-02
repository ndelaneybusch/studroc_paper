"""Exactness of ROC evaluation on the native FPR grid, on every device.

Threat model: the native grid ``linspace(0, 1, n0 + 1)`` puts every point on a
jump of the empirical ROC, so a point's value depends on which side of the jump
it is read. Floating-point ``k / n0`` and device-specific division rounding can
flip that side point by point: the center curve then disagrees with itself
across devices, and bootstrap curves are read at the wrong order statistic.
The envelope's studentized retention amplifies a one-step error at a single
point into a different retained set, so the simulation's CPU and GPU lanes
would report different bands for the same data.

Both the center and each bootstrap curve read the top of every jump (the
right-continuous empirical ROC); references are computed independently in
float64 numpy from sorted scores.
Sizes include the negative counts where float32 ``k / n0`` rounds below the
jump (150, 500, 900, 5000).
"""

import numpy as np
import pytest
import torch

from studroc_paper.methods.envelope_boot import envelope_bootstrap_band
from studroc_paper.methods.method_utils import compute_empirical_roc_from_scores
from studroc_paper.sampling.bootstrap_grid import (
    fpr_grid_fp_counts,
    generate_bootstrap_grid,
)

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA unavailable"
        ),
    ),
]


CASES = [
    "balanced_150_ties",
    "balanced_500",
    "imbalanced_900_heavy_tail",
    "balanced_5000",
]


def _scores(case: str) -> tuple[np.ndarray, np.ndarray]:
    """Float32 (neg, pos) scores for one realistic case."""
    rng = np.random.default_rng(CASES.index(case))
    match case:
        case "balanced_150_ties":
            neg = np.round(rng.normal(0.0, 1.0, 150), 1)
            pos = np.round(rng.normal(0.8, 1.0, 150), 1)
        case "balanced_500":
            neg, pos = rng.normal(0.0, 1.0, 500), rng.normal(1.0, 1.3, 500)
        case "imbalanced_900_heavy_tail":
            neg, pos = rng.standard_t(3, 900), rng.standard_t(3, 100) + 1.5
        case "balanced_5000":
            neg, pos = rng.normal(0.0, 1.0, 5000), rng.normal(0.5, 1.0, 5000)
    return neg.astype(np.float32), pos.astype(np.float32)


def _top_of_jump_roc(neg: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """TPR at FPR k/n0: positives strictly above the (k+1)-th largest negative."""
    neg_desc = np.append(np.sort(neg.astype(np.float64))[::-1], -np.inf)
    pos_sorted = np.sort(pos.astype(np.float64))
    return (len(pos) - np.searchsorted(pos_sorted, neg_desc, side="right")) / len(pos)


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["f32", "f64"])
def test_native_grid_maps_to_every_false_positive_count(dtype):
    """Every native grid point reads as exactly its own count, for all n0."""
    for n_neg in range(1, 6001):
        grid = torch.from_numpy(np.linspace(0, 1, n_neg + 1).astype(dtype))
        counts = fpr_grid_fp_counts(fpr_grid=grid, n_neg=n_neg)
        assert torch.equal(counts, torch.arange(n_neg + 1)), n_neg


def test_off_grid_points_floor_to_the_admitted_count():
    """Points off the native grid admit only the negatives they strictly pass."""
    n_neg = 900
    k = torch.arange(1, n_neg, dtype=torch.float64)
    mid = fpr_grid_fp_counts(fpr_grid=(k + 0.5) / n_neg, n_neg=n_neg)
    below = fpr_grid_fp_counts(fpr_grid=(k - 1e-3) / n_neg, n_neg=n_neg)
    assert torch.equal(mid, k.long())
    assert torch.equal(below, k.long() - 1)
    edges = torch.tensor([-0.1, 0.0, 1.0, 1.1], dtype=torch.float64)
    assert fpr_grid_fp_counts(fpr_grid=edges, n_neg=n_neg).tolist() == [
        0,
        0,
        n_neg,
        n_neg,
    ]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", CASES)
def test_center_and_identity_resample_read_the_top_of_every_jump(
    case, device, monkeypatch
):
    """The center and its bootstrap image are the same curve, exactly.

    A resample that draws every point once must reproduce the center: if the
    two read different sides of a jump, every bootstrap deviation carries that
    jump as bias, and the retention ranks curves by it.
    """
    neg, pos = _scores(case)
    grid = torch.from_numpy(np.linspace(0, 1, len(neg) + 1).astype(np.float32))
    expected = _top_of_jump_roc(neg, pos)

    center = compute_empirical_roc_from_scores(
        torch.from_numpy(neg).to(device),
        torch.from_numpy(pos).to(device),
        grid.to(device),
    )
    np.testing.assert_allclose(center.double().cpu().numpy(), expected, atol=1e-6)

    def identity(low, high, size, *, device, generator):
        return torch.arange(high, device=device).expand(size)

    monkeypatch.setattr(torch, "randint", identity)
    y_true = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    boot = generate_bootstrap_grid(
        y_true=torch.from_numpy(y_true),
        y_score=torch.from_numpy(np.concatenate([neg, pos])),
        B=2,
        grid=grid,
        device=torch.device(device),
    )
    for row in boot.double().cpu().numpy():
        np.testing.assert_allclose(row, expected, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("alpha", [0.5, 0.05])
def test_cpu_and_cuda_envelopes_agree_on_identical_resamples(case, alpha):
    """The simulation's two lanes produce the same band from the same matrix.

    Only float32 rounding differs between devices, so the bands may differ by
    at most one TPR step where curves tie at the retention threshold within an
    ulp, and must agree closely on average.
    """
    neg, pos = _scores(case)
    y_true = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))]).astype(np.float32)
    y_score = np.concatenate([neg, pos])
    grid = np.linspace(0, 1, len(neg) + 1).astype(np.float32)
    boot = generate_bootstrap_grid(
        y_true=torch.from_numpy(y_true),
        y_score=torch.from_numpy(y_score),
        B=1000,
        grid=torch.from_numpy(grid),
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(7),
    )
    bands = {
        device: envelope_bootstrap_band(
            boot_tpr_matrix=boot.to(device),
            fpr_grid=grid,
            y_true=y_true,
            y_score=y_score,
            alpha=alpha,
            boundary_method="wilson",
            device=torch.device(device),
        )
        for device in ("cpu", "cuda")
    }
    for side in (1, 2):
        diff = np.abs(bands["cpu"][side] - bands["cuda"][side])
        assert diff.max() <= 1 / len(pos) + 1e-6
        assert diff.mean() <= 1e-4
