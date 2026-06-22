"""Shared fixtures and band-contract helpers for the method/eval test suite.

These fixtures supply small, fully reproducible inputs (fixed numpy *and* torch
seeds) so that band methods can be exercised deterministically. Sample sizes and
the bootstrap count ``B`` are kept deliberately small: every test here is a unit
test that pins a construction or an invariant, not a Monte Carlo coverage study,
so nothing depends on ``B`` or ``n`` being large.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch

from studroc_paper.datagen import (
    make_exponential_dgp,
    make_gaussian_dgp,
    make_uniform_dgp,
)
from studroc_paper.datagen.true_rocs import DGP
from studroc_paper.sampling import generate_bootstrap_grid

_CPU = torch.device("cpu")


def labels_and_scores(
    scores_pos: np.ndarray, scores_neg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble sklearn-style ``(y_true, y_score)`` from class-split scores.

    Args:
        scores_pos: Scores for the positive class.
        scores_neg: Scores for the negative class.

    Returns:
        Tuple ``(y_true, y_score)`` with positives stacked before negatives.
    """
    y_true = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
    y_score = np.concatenate([scores_pos, scores_neg])
    return y_true, y_score


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded NumPy generator for reproducible sampling."""
    return np.random.default_rng(0)


@pytest.fixture
def fpr_grid() -> np.ndarray:
    """Uniform FPR grid pinned at both endpoints (matches the simulation grid)."""
    return np.linspace(0.0, 1.0, 101)


@pytest.fixture
def gaussian_scores(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Balanced Gaussian sample (d' = 1.5, AUC ~ 0.86), the home turf DGP.

    Returns:
        Tuple ``(y_true, y_score)`` with 200 positives and 200 negatives.
    """
    dgp = make_gaussian_dgp(delta_mu=1.5, sigma=1.0)
    scores_pos, scores_neg = dgp.sample(n_pos=200, n_neg=200, rng=rng)
    return labels_and_scores(scores_pos, scores_neg)


@pytest.fixture
def boot_tpr_matrix(
    gaussian_scores: tuple[np.ndarray, np.ndarray], fpr_grid: np.ndarray
) -> np.ndarray:
    """Reproducible ``(B, K)`` bootstrap TPR matrix for ``gaussian_scores``.

    The torch RNG is seeded so the same matrix is produced on every run, which
    is what lets the band tests assert exact equalities.
    """
    y_true, y_score = gaussian_scores
    torch.manual_seed(0)
    matrix = generate_bootstrap_grid(
        y_true=torch.from_numpy(y_true),
        y_score=torch.from_numpy(y_score),
        B=200,
        grid=torch.from_numpy(fpr_grid),
        device=_CPU,
        batch_size=500,
        tpr_method="empirical",
    )
    return matrix.cpu().numpy()


@pytest.fixture(params=["gaussian", "exponential", "uniform"])
def closed_form_dgp(request: pytest.FixtureRequest) -> DGP:
    """A DGP with an exact, numerically well-behaved analytic true ROC.

    Gaussian (binormal), exponential (power-law ROC), and uniform (piecewise
    linear ROC) all have closed-form ``true_roc`` callables, so tests can use
    known truth without Monte Carlo. Delicate DGPs (Weibull, Cauchy) are
    deliberately excluded.
    """
    if request.param == "gaussian":
        return make_gaussian_dgp(delta_mu=1.5, sigma=1.0)
    if request.param == "exponential":
        return make_exponential_dgp(neg_rate=1.0, pos_rate=0.5)
    return make_uniform_dgp()


@pytest.fixture
def make_known_band() -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    """Factory: a symmetric band of fixed half-width around a true ROC.

    Returns:
        ``_make(true_tpr, half_width=0.05)`` producing ``(lower, upper)`` clipped
        to ``[0, 1]``.
    """

    def _make(
        true_tpr: np.ndarray, *, half_width: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        true_tpr = np.asarray(true_tpr, dtype=float)
        lower = np.clip(true_tpr - half_width, 0.0, 1.0)
        upper = np.clip(true_tpr + half_width, 0.0, 1.0)
        return lower, upper

    return _make


@pytest.fixture
def assert_valid_band() -> Callable[..., None]:
    """Factory: assert the universal ROC-band contract.

    The returned checker enforces what *any* confidence band must satisfy
    regardless of method: aligned finite shapes, containment in the unit
    interval, and a non-crossing ``lower <= upper``. Monotonicity is *not*
    asserted here because some pointwise methods produce jagged bands; the
    envelope's monotonicity guarantee is checked separately where it is a
    genuine contract.
    """

    def _check(
        fpr: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        *,
        n_grid: int | None = None,
    ) -> None:
        fpr = np.asarray(fpr)
        lower = np.asarray(lower)
        upper = np.asarray(upper)
        if n_grid is not None:
            assert fpr.shape == (n_grid,)
        assert lower.shape == fpr.shape == upper.shape
        assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
        # One float32 ulp of slack: clipping to [0, 1] can land a hair outside.
        assert lower.min() >= -1e-6 and upper.max() <= 1.0 + 1e-6
        assert np.all(upper - lower >= -1e-6), "band must not cross (upper < lower)"

    return _check
