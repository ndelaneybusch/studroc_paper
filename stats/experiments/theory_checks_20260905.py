"""Stress checks for the fiducial theory's finite-sample inequalities.

Run with ``uv run --no-sync python stats/experiments/theory_checks_20260905.py``.
Results are printed as JSON. These checks exercise formulas and adversarial
rank paths; they do not certify population coverage of a fiducial band.
"""

import itertools
import json

import numpy as np
from scipy.stats import beta, binom


def bracket_check(*, labels: np.ndarray, du: np.ndarray, dv: np.ndarray) -> dict:
    """Check a completion bracket by two independent exact area calculations.

    Args:
        labels: Merged labels in ascending placement order.
        du: Negative-class probability spacings, including both end gaps.
        dv: Positive-class probability spacings, including both end gaps.

    Returns:
        Integrated bracket width and its ratio to the largest-spacing bound.
    """
    n1 = int(labels.sum())
    v = np.concatenate(([0.0], np.cumsum(dv)))
    v[-1] = 1.0
    counts = np.concatenate(([0], np.cumsum(labels)[labels == 0], [n1]))
    area_by_gaps = float(np.dot(du, v[counts[1:] + 1] - v[counts[:-1]]))
    i = j = 0
    area_by_path = float(du[0] * dv[0])
    for label in labels:
        i += int(label == 0)
        j += int(label == 1)
        area_by_path += float(du[i] * dv[j])
    np.testing.assert_allclose(actual=area_by_gaps, desired=area_by_path, atol=1e-14)
    bound = float(du.max() + dv.max())
    assert area_by_gaps <= bound + 1e-14

    u = np.concatenate(([0.0], np.cumsum(du)))
    u[-1] = 1.0
    lower = v[counts[:-1]]
    upper = v[counts[1:] + 1]
    points = (u[:-1] + u[1:]) / 2
    shifted = points + du.max()
    indices = np.searchsorted(a=u, v=shifted, side="right") - 1
    shifted_lower = np.where(shifted >= 1, 1.0, lower[np.minimum(indices, len(du) - 1)])
    assert np.all(upper <= shifted_lower + dv.max() + 1e-13)
    return {"area": area_by_gaps, "bound": bound, "ratio": area_by_gaps / bound}


def check_brackets(*, rng: np.random.Generator) -> dict:
    """Exercise every small rank path and large, highly imbalanced paths."""
    count = 0
    largest_ratio = 0.0
    for n0, n1 in itertools.product(range(1, 5), repeat=2):
        for positions in itertools.combinations(range(n0 + n1), n1):
            labels = np.zeros(n0 + n1, dtype=int)
            labels[list(positions)] = 1
            for concentration in [0.15, 1.0, 8.0]:
                result = bracket_check(
                    labels=labels,
                    du=rng.dirichlet(alpha=np.full(n0 + 1, concentration)),
                    dv=rng.dirichlet(alpha=np.full(n1 + 1, concentration)),
                )
                count += 1
                largest_ratio = max(largest_ratio, result["ratio"])
    for n0, n1 in [(2, 100), (100, 2), (50, 50), (100, 100)]:
        for _ in range(2500):
            labels = rng.permutation(
                np.concatenate((np.zeros(n0, dtype=int), np.ones(n1, dtype=int)))
            )
            result = bracket_check(
                labels=labels,
                du=rng.dirichlet(alpha=np.ones(n0 + 1)),
                dv=rng.dirichlet(alpha=np.ones(n1 + 1)),
            )
            count += 1
            largest_ratio = max(largest_ratio, result["ratio"])
    uniform = bracket_check(
        labels=np.tile([0, 1], reps=100),
        du=np.full(101, 1 / 101),
        dv=np.full(101, 1 / 101),
    )
    np.testing.assert_allclose(actual=uniform["ratio"], desired=201 / 202, atol=1e-13)
    return {
        "random_cases": count,
        "largest_random_ratio": largest_ratio,
        "near_sharp": uniform,
    }


def check_cutpoints() -> dict:
    """Verify exact end-gap cuts and binomial upper-limit coverage near jumps."""
    coverage_checks = []
    left_cuts = []
    for n in [5, 20, 100, 500]:
        ranks = np.arange(n + 1)
        for delta in [0.005, 0.025, 0.05]:
            upper = np.concatenate(
                (beta.ppf(q=1 - delta, a=ranks[:-1] + 1, b=n - ranks[:-1]), [1.0])
            )
            parameters = np.unique(
                np.concatenate((np.linspace(0, 1, 1001), np.minimum(upper + 1e-10, 1)))
            )
            last_rejecting_rank = (
                np.searchsorted(a=upper, v=parameters, side="left") - 1
            )
            errors = binom.cdf(k=last_rejecting_rank, n=n, p=parameters)
            worst = float(np.max(errors))
            assert worst <= delta + 1e-10
            assert worst >= delta - 1e-6
            coverage_checks.append({"n": n, "delta": delta, "largest_error": worst})
            cut = int(np.ceil(-n * np.expm1(np.log(delta) / n)))
            reach = float((1 - cut / n) ** n)
            predecessor_reach = float((1 - (cut - 1) / n) ** n)
            assert reach <= delta < predecessor_reach
            left_cuts.append(
                {"n": n, "epsilon": delta, "cut": cut, "reach_probability": reach}
            )
    margins = []
    for k in [0, 1, 5, 25, 100, 250]:
        upper = float(beta.ppf(q=0.975, a=k + 1, b=500 - k))
        margins.append(
            {
                "n0": 500,
                "K": k,
                "delta": 0.025,
                "s_upper": upper,
                "exact_extra_ranks": int(np.ceil(500 * upper)) - k,
                "two_sqrt_extra_ranks": int(np.ceil(2 * np.sqrt(k))),
            }
        )
    return {
        "binomial_checks": coverage_checks,
        "left_cuts": left_cuts,
        "right_margin_comparison": margins,
    }


def check_gaussian_direction() -> list[dict]:
    """Check covariance signs and report the proved directional probability bounds."""
    grid = np.linspace(0.01, 0.99, 101)
    minima = []
    for power, ratio in itertools.product([0.25, 1.0, 3.0], [0.1, 1.0, 10.0]):
        curve = grid**power
        derivative = power * grid ** (power - 1)
        covariance = np.minimum.outer(curve, curve) - np.outer(curve, curve)
        covariance += (
            ratio
            * np.outer(derivative, derivative)
            * (np.minimum.outer(grid, grid) - np.outer(grid, grid))
        )
        assert np.min(covariance) >= 0
        assert np.linalg.eigvalsh(covariance)[0] > 0
        minima.append(float(np.min(covariance)))
    return [
        {
            "alpha": alpha,
            "one_sided_lower": alpha / 2,
            "one_sided_upper": float(1 - np.sqrt(1 - alpha)),
            "both_sides_upper": float((1 - np.sqrt(1 - alpha)) ** 2),
            "smallest_checked_covariance": min(minima),
        }
        for alpha in [0.05, 0.5]
    ]


def main() -> None:
    """Print reproducible verification results without mutating repository data."""
    rng = np.random.default_rng(seed=20260905)
    result = {
        "seed": 20260905,
        "brackets": check_brackets(rng=rng),
        "cutpoints": check_cutpoints(),
        "gaussian_direction": check_gaussian_direction(),
    }
    print(json.dumps(obj=result, indent=2))


if __name__ == "__main__":
    main()
