"""
Unit tests for the maximin Latin hypercube 'build' method.

Critical behaviors tested:
1. Every column places exactly one point in each of the n strata
2. Designs are reproducible from the seed and differ across seeds
3. The greedy maximin criterion spreads points well beyond a random LHS
"""

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from studroc_paper.sampling.lhs import maximin_lhs


def _random_lhs_grid(*, n: int, k: int, seed: int) -> np.ndarray:
    """Return the cell indices of an unoptimized random LHS."""
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.permutation(n) for _ in range(k)])


@pytest.mark.parametrize(
    ("n", "k", "dup"),
    [
        pytest.param(1, 1, 5, id="single-point"),
        pytest.param(2, 3, 5, id="two-points"),
        pytest.param(97, 1, 5, id="one-dim"),
        pytest.param(250, 3, 5, id="three-dim"),
        pytest.param(120, 2, 1, id="one-candidate-per-free-level"),
        pytest.param(40, 6, 3, id="six-dim-small-n"),
    ],
)
def test_build_places_one_point_per_stratum_in_every_column(
    n: int, k: int, dup: int
) -> None:
    x = maximin_lhs(n=n, k=k, method="build", dup=dup, seed=7)

    assert x.shape == (n, k)
    assert np.all((x >= 0) & (x < 1))
    cells = np.floor(x * n).astype(int)
    for col in range(k):
        np.testing.assert_array_equal(np.sort(cells[:, col]), np.arange(n))


def test_build_is_reproducible_from_seed_and_sensitive_to_it() -> None:
    a = maximin_lhs(n=150, k=3, method="build", dup=5, seed=11)
    b = maximin_lhs(n=150, k=3, method="build", dup=5, seed=11)
    c = maximin_lhs(n=150, k=3, method="build", dup=5, seed=12)

    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(np.floor(a * 150), np.floor(c * 150))


@pytest.mark.parametrize(
    ("n", "k", "min_ratio"),
    [
        pytest.param(200, 2, 2.0, id="two-dim"),
        pytest.param(300, 3, 2.0, id="three-dim"),
    ],
)
def test_build_separates_points_far_more_than_random_lhs(
    n: int, k: int, min_ratio: float
) -> None:
    seeds = range(8)
    maximin = [
        pdist(np.floor(maximin_lhs(n=n, k=k, method="build", dup=5, seed=s) * n)).min()
        for s in seeds
    ]
    random = [pdist(_random_lhs_grid(n=n, k=k, seed=s)).min() for s in seeds]

    # Even the worst maximin design beats a typical random design
    assert min(maximin) > np.mean(random)
    assert np.mean(maximin) >= min_ratio * np.mean(random)
