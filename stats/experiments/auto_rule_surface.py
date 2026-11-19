#!/usr/bin/env python
"""Coverage surface of the unfloored fiducial band, and the auto rule it implies.

The question
------------
A user-facing ``auto`` mode must choose between the unfloored fiducial band
and the hybrid from quantities it can see: the class counts and the AUC. The
unfloored band under-covers when the ROC has a convex hook in a corner that
holds about one expected observation of one class, so the unsafe region is a
ridge in (sample size, AUC) whose position depends on the score distributions.
This study estimates the probability that the unfloored band misses the true
ROC as a continuous function of

- the smaller class count ``n_min`` (log scale),
- the signed class log-ratio ``log(n_pos / n_neg)``,
- the AUC (probit scale), and
- the family's shape parameter (log scale),

separately for the two families whose unfloored band is known to fail
(Student-t location shift, by degrees of freedom; heteroscedastic Gaussian, by
sigma ratio). The rule is then read off the worst case over shape and family.

Design
------
Every draw is one dataset at its own point of the covariate space: there are
no cells and no replicates, because a binary-response surface fit borrows
strength across neighbouring points far more efficiently than per-cell
precision would.

Round 0 fills the space. Candidates come from a scrambled Sobol sequence and
are subsampled with probability proportional to the local sparsity of the
existing trial-level data (the distance to the k-th nearest existing draw,
raised to the dimension) divided by a power of the draw's cost. The existing
data are the main suite and the shape x AUC study (``--hot-start``), which sit
on a few sample sizes and prevalences, so new draws go between them.

Later rounds refine the decision boundary. The surface is refitted to all
draws and candidates are weighted by the posterior probability that the latent
surface lies within ``CONTOUR_BAND`` of the target miss rate, again divided by
cost. A fixed fraction of every refinement round is still spent on the round-0
criterion, which keeps evidence arriving in regions the surface calls safe.

Surface
-------
A Gaussian process with a Bernoulli likelihood, linear mean and anisotropic
Matern-5/2 kernel, in its low-rank form (subset of regressors on a fixed
space-filling set of inducing points). In weight space this is a penalised
logistic regression on radial basis functions, so it is fitted by penalised
IRLS on every draw at once; the length scales and amplitude maximise the
Laplace approximation to the marginal likelihood. Nothing constrains the
surface to be monotone in any covariate: coverage is known to dip and recover
as n grows, and the fit has to be able to show it.

The fit is checked by K-fold cross-validation (a reliability table of
predicted against observed miss rates) and, free of any model, by the observed
miss rate of the draws that a rule admits.

Rule
----
For every (AUC, class ratio) on a grid, the frontier ``n*`` is the smallest
``n_min`` from which the worst-case miss probability stays at or below the
target for every larger ``n_min`` in the design, whichever class is the
minority. The worst case is over shape and family. The rule

    n_min >= max(floor, C * ratio^b / (1 - AUC)^a)

is the least-slack upper envelope of that frontier (a linear program on the
log scale), fitted once to the posterior mean surface and once to its upper
credible bound. Where the surface is still unsafe at the largest sample size
in the design, the envelope is required to exclude the whole tested range.

Usage
-----
    uv run python stats/experiments/auto_rule_surface.py run --batch 40000 --dry-run
    uv run python stats/experiments/auto_rule_surface.py run --batch 40000
    uv run python stats/experiments/auto_rule_surface.py fit
    uv run python stats/experiments/auto_rule_surface.py run --batch 20000

``run`` resumes an unfinished round; otherwise it designs and runs the next
one. ``fit`` writes ``report.md``, ``frontier.csv`` and ``surface.png`` to the
output directory.
"""

import argparse
import json
import multiprocessing
import os
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.optimize import linprog, minimize
from scipy.spatial import cKDTree
from scipy.special import expit, logit
from scipy.stats import norm, qmc, rankdata
from tqdm import tqdm

# =============================================================================
# Covariate space
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class Family:
    """One score-distribution family and the range of its shape parameter.

    Attributes:
        name: DGP name understood by ``map_lhs_to_dgp``.
        shape_name: Name of the shape parameter in the LHS parametrisation.
        shape_bounds: Range of the shape parameter.
        label: Axis label of the shape parameter.
    """

    name: str
    shape_name: str
    shape_bounds: tuple[float, float]
    label: str


FAMILIES = {
    family.name: family
    for family in (
        Family(
            name="student_t",
            shape_name="df",
            shape_bounds=(1.1, 30.0),
            label="degrees of freedom",
        ),
        Family(
            name="hetero_gaussian",
            shape_name="sigma_ratio",
            shape_bounds=(0.2, 5.0),
            label="sigma ratio (positive / negative)",
        ),
    )
}

ALPHA = 0.05
N_MIN_BOUNDS = (10, 20_000)
# Largest class ratio in the design; 19 is 5% prevalence
RATIO_MAX = 19.0
# Draws whose larger class would exceed this are not run
N_CLASS_MAX = 100_000
AUC_BOUNDS = (0.55, 0.99)
DIM = 4

# Seconds per observation of each class for one unfloored band, measured
# single-threaded; only the ratio between draws matters to the design
COST_PER_NEGATIVE = 6.0e-4
COST_PER_POSITIVE = 3.2e-4
COST_FIXED = 0.02

# Unit-cube distance beyond which existing trial-level rows are not
# considered part of the design space
HOT_START_MARGIN = 0.02

DRAW_COLUMNS = [
    "source",
    "family",
    "shape",
    "auc",
    "n_pos",
    "n_neg",
    "miss",
    "miss_below",
    "miss_above",
]

# Dense FPR grid for the true AUC, log-refined toward both corners
TRUE_AUC_GRID = np.unique(
    np.concatenate(
        [
            [0.0, 1.0],
            np.logspace(-6, np.log10(0.5), 1500),
            1.0 - np.logspace(-6, np.log10(0.5), 1500),
        ]
    )
)


def _scale(values: NDArray, *, bounds: tuple[float, float]) -> NDArray:
    return (values - bounds[0]) / (bounds[1] - bounds[0])


def _unscale(unit: NDArray, *, bounds: tuple[float, float]) -> NDArray:
    return bounds[0] + unit * (bounds[1] - bounds[0])


def unit_coordinates(*, frame: pd.DataFrame, family: Family) -> NDArray:
    """Map draws of one family to the unit cube the surface is fitted on.

    Args:
        frame: Draws with ``n_pos``, ``n_neg``, ``auc`` and ``shape``.
        family: The draws' family.

    Returns:
        (rows, 4) coordinates: log smaller class count, signed class
        log-ratio, probit AUC, log shape.
    """
    n_pos = frame["n_pos"].to_numpy(float)
    n_neg = frame["n_neg"].to_numpy(float)
    log_ratio_max = np.log(RATIO_MAX)
    return np.column_stack(
        [
            _scale(
                np.log(np.minimum(n_pos, n_neg)), bounds=tuple(np.log(N_MIN_BOUNDS))
            ),
            _scale(np.log(n_pos / n_neg), bounds=(-log_ratio_max, log_ratio_max)),
            _scale(
                norm.ppf(frame["auc"].to_numpy(float)),
                bounds=tuple(norm.ppf(AUC_BOUNDS)),
            ),
            _scale(
                np.log(frame["shape"].to_numpy(float)),
                bounds=tuple(np.log(family.shape_bounds)),
            ),
        ]
    )


def design_rows(*, unit: NDArray, family: Family) -> pd.DataFrame:
    """Turn unit-cube points into runnable draws.

    Class counts are rounded to integers, so a draw's exact coordinates are
    those of :func:`unit_coordinates` applied to the result.

    Args:
        unit: (rows, 4) points of the unit cube.
        family: Family of the draws.

    Returns:
        Draws with ``family``, ``shape``, ``auc``, ``n_pos`` and ``n_neg``.
    """
    n_min = np.rint(np.exp(_unscale(unit[:, 0], bounds=tuple(np.log(N_MIN_BOUNDS)))))
    log_ratio = _unscale(unit[:, 1], bounds=(-np.log(RATIO_MAX), np.log(RATIO_MAX)))
    n_max = np.rint(n_min * np.exp(np.abs(log_ratio)))
    positives_larger = log_ratio >= 0
    return pd.DataFrame(
        {
            "family": family.name,
            "shape": np.exp(
                _unscale(unit[:, 3], bounds=tuple(np.log(family.shape_bounds)))
            ),
            "auc": norm.cdf(_unscale(unit[:, 2], bounds=tuple(norm.ppf(AUC_BOUNDS)))),
            "n_pos": np.where(positives_larger, n_max, n_min).astype(int),
            "n_neg": np.where(positives_larger, n_min, n_max).astype(int),
        }
    )


def cost_seconds(frame: pd.DataFrame) -> NDArray:
    """Single-threaded seconds to run each draw.

    Args:
        frame: Draws with ``n_pos`` and ``n_neg``.

    Returns:
        Estimated seconds per draw.
    """
    return (
        COST_FIXED
        + COST_PER_NEGATIVE * frame["n_neg"].to_numpy(float)
        + COST_PER_POSITIVE * frame["n_pos"].to_numpy(float)
    )


# =============================================================================
# Data
# =============================================================================


def load_hot_start(directories: list[Path]) -> pd.DataFrame:
    """Read the unfloored band's trial-level results from earlier studies.

    Reads every ``*_individual.feather`` of the main suite or the shape x AUC
    study and keeps the unfloored fiducial band at the study's alpha for the
    families in :data:`FAMILIES`.

    Args:
        directories: Result directories; missing ones are skipped.

    Returns:
        Draws in :data:`DRAW_COLUMNS`, one row per replicate.
    """
    parts = []
    for directory in directories:
        for path in sorted(directory.glob("*_individual.feather")):
            family = next(
                (f for f in FAMILIES.values() if path.name.startswith(f.name)), None
            )
            if family is None:
                continue
            frame = pd.read_feather(
                path,
                columns=[
                    "method",
                    "alpha",
                    f"lhs_{family.shape_name}",
                    "true_auc",
                    "n_pos",
                    "n_neg",
                    "covers_entirely",
                    "violation_below",
                    "violation_above",
                ],
            )
            frame = frame[
                (frame["method"] == "fiducial") & np.isclose(frame["alpha"], ALPHA)
            ]
            parts.append(
                pd.DataFrame(
                    {
                        "source": directory.name,
                        "family": family.name,
                        "shape": frame[f"lhs_{family.shape_name}"].to_numpy(float),
                        "auc": frame["true_auc"].to_numpy(float),
                        "n_pos": frame["n_pos"].to_numpy(int),
                        "n_neg": frame["n_neg"].to_numpy(int),
                        "miss": ~frame["covers_entirely"].to_numpy(bool),
                        "miss_below": frame["violation_below"].to_numpy(bool),
                        "miss_above": frame["violation_above"].to_numpy(bool),
                    }
                )
            )
    if not parts:
        return pd.DataFrame(columns=DRAW_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def load_rounds(out_dir: Path) -> pd.DataFrame:
    """Read the finished draws of every round of this study.

    Args:
        out_dir: Output directory.

    Returns:
        Draws in :data:`DRAW_COLUMNS` plus the per-draw diagnostics.
    """
    parts = [pd.read_feather(p) for p in sorted(out_dir.glob("draws_round*.feather"))]
    if not parts:
        return pd.DataFrame(columns=DRAW_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def load_draws(*, hot_start: list[Path], out_dir: Path) -> pd.DataFrame:
    """Read every draw available to the study: earlier results and own rounds.

    Args:
        hot_start: Result directories of earlier studies.
        out_dir: Output directory of this study.

    Returns:
        Draws in :data:`DRAW_COLUMNS`.
    """
    parts = [
        frame[DRAW_COLUMNS]
        for frame in (load_hot_start(hot_start), load_rounds(out_dir))
        if not frame.empty
    ]
    if not parts:
        return pd.DataFrame(columns=DRAW_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def in_design_space(*, frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the draws that lie inside the covariate space, with coordinates.

    Args:
        frame: Draws in :data:`DRAW_COLUMNS`.

    Returns:
        The retained draws with unit coordinates in columns ``u0`` to ``u3``.
    """
    parts = []
    for family in FAMILIES.values():
        sub = frame[frame["family"] == family.name].reset_index(drop=True)
        if sub.empty:
            continue
        unit = unit_coordinates(frame=sub, family=family)
        keep = np.all(
            (unit >= -HOT_START_MARGIN) & (unit <= 1 + HOT_START_MARGIN), axis=1
        )
        coords = pd.DataFrame(unit, columns=[f"u{d}" for d in range(DIM)])
        parts.append(pd.concat([sub, coords], axis=1)[keep])
    if not parts:
        return pd.DataFrame(columns=[*DRAW_COLUMNS, *(f"u{d}" for d in range(DIM))])
    return pd.concat(parts, ignore_index=True)


def unit_matrix(frame: pd.DataFrame) -> NDArray:
    """Unit coordinates of draws returned by :func:`in_design_space`."""
    return frame[[f"u{d}" for d in range(DIM)]].to_numpy(float)


# =============================================================================
# Surface: low-rank Gaussian process with a Bernoulli likelihood
# =============================================================================

# Inducing points number 2 ** INDUCING_LOG2
INDUCING_LOG2 = 8
# Rows used to choose the kernel hyperparameters; the coefficients are then
# fitted to every row
HYPER_ROWS = 15_000
# Prior standard deviation of the intercept and linear-mean coefficients
MEAN_PRIOR_SD = 10.0
LENGTH_BOUNDS = (0.05, 3.0)
AMPLITUDE_BOUNDS = (0.1, 10.0)
JITTER = 1e-6
# Rows whose kernel features are held in memory at once while fitting
FIT_BLOCK = 50_000


def matern52(*, a: NDArray, b: NDArray, lengthscales: NDArray) -> NDArray:
    """Anisotropic Matern-5/2 kernel with unit variance.

    Args:
        a: (n, d) points.
        b: (m, d) points.
        lengthscales: (d,) length scales.

    Returns:
        (n, m) kernel matrix.
    """
    a_scaled, b_scaled = a / lengthscales, b / lengthscales
    squared = (
        np.sum(a_scaled**2, axis=1)[:, None]
        + np.sum(b_scaled**2, axis=1)[None, :]
        - 2 * a_scaled @ b_scaled.T
    )
    r = np.sqrt(5 * np.maximum(squared, 0.0))
    return (1 + r + r**2 / 3) * np.exp(-r)


class Surface:
    """Fitted miss-probability surface of one family.

    The latent logit is ``[1, x] @ beta + amplitude * phi(x) @ w`` with
    ``w ~ N(0, I)``, where ``phi`` are the whitened kernel features on the
    inducing points, so ``amplitude^2 * phi(x) @ phi(x')`` is the
    subset-of-regressors approximation to the Matern kernel.

    Attributes:
        inducing: (m, d) inducing points.
        lengthscales: (d,) kernel length scales.
        amplitude: Kernel standard deviation.
        coefficients: Posterior mode of ``[beta, w]``.
        covariance: Laplace posterior covariance of the coefficients.
        evidence: Laplace log marginal likelihood of the fitted rows.
    """

    def __init__(
        self, *, inducing: NDArray, lengthscales: NDArray, amplitude: float
    ) -> None:
        """Set the kernel; :meth:`fit` estimates the coefficients.

        Args:
            inducing: (m, d) inducing points.
            lengthscales: (d,) kernel length scales.
            amplitude: Kernel standard deviation.
        """
        self.inducing = inducing
        self.lengthscales = np.asarray(lengthscales, float)
        self.amplitude = float(amplitude)
        gram = matern52(a=inducing, b=inducing, lengthscales=self.lengthscales)
        self._chol = cholesky(gram + JITTER * np.eye(len(inducing)), lower=True)
        self.coefficients: NDArray | None = None
        self.covariance: NDArray | None = None
        self.evidence = -np.inf

    def features(self, x: NDArray) -> NDArray:
        """Design matrix of the latent logit at ``x``.

        Args:
            x: (n, d) unit coordinates.

        Returns:
            (n, 1 + d + m) matrix of intercept, linear mean and scaled kernel
            features.
        """
        cross = matern52(a=self.inducing, b=x, lengthscales=self.lengthscales)
        phi = solve_triangular(self._chol, cross, lower=True).T
        return np.hstack([np.ones((len(x), 1)), x, self.amplitude * phi])

    def _penalised_likelihood(
        self, *, x: NDArray, y: NDArray, theta: NDArray, penalty: NDArray
    ) -> tuple[float, NDArray, NDArray]:
        """Penalised log-likelihood with its gradient and negative Hessian.

        Rows are processed in blocks of :data:`FIT_BLOCK`, so the design
        matrix of the whole data set is never held in memory.

        Args:
            x: (n, d) unit coordinates.
            y: (n,) miss indicators.
            theta: Coefficients at which to evaluate.
            penalty: Prior precision of each coefficient.

        Returns:
            The value, its gradient, and the negative Hessian.
        """
        value = -0.5 * float(theta @ (penalty * theta))
        gradient = -penalty * theta
        hessian = np.diag(penalty)
        for start in range(0, len(x), FIT_BLOCK):
            design = self.features(x[start : start + FIT_BLOCK])
            y_block = y[start : start + FIT_BLOCK]
            eta = design @ theta
            p = expit(eta)
            weights = np.maximum(p * (1 - p), 1e-10)
            value += float(np.sum(y_block * eta - np.logaddexp(0, eta)))
            gradient += design.T @ (y_block - p)
            hessian += (design * weights[:, None]).T @ design
        return value, gradient, hessian

    def fit(self, *, x: NDArray, y: NDArray, max_iter: int = 60) -> "Surface":
        """Find the posterior mode by Newton's method and its Laplace covariance.

        Args:
            x: (n, d) unit coordinates.
            y: (n,) miss indicators.
            max_iter: Newton iteration limit.

        Returns:
            This surface, fitted.
        """
        n_mean = 1 + x.shape[1]
        penalty = np.concatenate(
            [np.full(n_mean, MEAN_PRIOR_SD**-2), np.ones(len(self.inducing))]
        )
        theta = np.zeros(len(penalty))
        theta[0] = logit(np.clip(y.mean(), 1e-4, 1 - 1e-4))
        current, gradient, hessian = self._penalised_likelihood(
            x=x, y=y, theta=theta, penalty=penalty
        )
        for _ in range(max_iter):
            step = cho_solve(cho_factor(hessian, lower=True), gradient)
            scale = 1.0
            while True:
                proposal = self._penalised_likelihood(
                    x=x, y=y, theta=theta + scale * step, penalty=penalty
                )
                if proposal[0] >= current or scale < 1e-4:
                    break
                scale /= 2
            if proposal[0] < current:
                # No step along the Newton direction improves the objective
                break
            gain = proposal[0] - current
            theta = theta + scale * step
            current, gradient, hessian = proposal
            if gain < 1e-7 * max(1.0, abs(current)):
                break
        factor = cho_factor(hessian, lower=True)
        self.coefficients = theta
        self.covariance = cho_solve(factor, np.eye(len(theta)))
        self.evidence = (
            current + 0.5 * np.sum(np.log(penalty)) - np.sum(np.log(np.diag(factor[0])))
        )
        return self

    def predict(self, x: NDArray, *, block: int = 20_000) -> tuple[NDArray, NDArray]:
        """Posterior mean and standard deviation of the latent logit.

        Args:
            x: (n, d) unit coordinates.
            block: Rows evaluated at a time.

        Returns:
            Mean and standard deviation, each (n,).
        """
        means, sds = [], []
        for start in range(0, len(x), block):
            design = self.features(x[start : start + block])
            means.append(design @ self.coefficients)
            sds.append(
                np.sqrt(
                    np.maximum(np.sum((design @ self.covariance) * design, axis=1), 0)
                )
            )
        return np.concatenate(means), np.concatenate(sds)


def fit_surface(
    *,
    x: NDArray,
    y: NDArray,
    seed: int,
    groups: NDArray | None = None,
    hyperparameters: tuple[NDArray, float] | None = None,
) -> Surface:
    """Fit one family's surface, choosing the kernel by marginal likelihood.

    Args:
        x: (n, d) unit coordinates.
        y: (n,) miss indicators.
        seed: Seed of the inducing points and the hyperparameter subsample.
        groups: (n,) label of the study each row came from. The
            hyperparameter subsample takes equal numbers of rows from every
            study, so that a large study confined to a few sample sizes and
            prevalences does not decide the length scales alone. None treats
            the rows as one study.
        hyperparameters: Length scales and amplitude to use as given; None
            maximises the Laplace marginal likelihood on a subsample of at
            most :data:`HYPER_ROWS` rows.

    Returns:
        The surface fitted to every row.
    """
    y = np.asarray(y, float)
    inducing = qmc.Sobol(d=x.shape[1], seed=seed).random_base2(INDUCING_LOG2)
    if hyperparameters is None:
        rng = np.random.default_rng(seed)
        labels = np.zeros(len(x), int) if groups is None else np.asarray(groups)
        members = [np.flatnonzero(labels == g) for g in np.unique(labels)]
        quota = HYPER_ROWS // len(members)
        rows = np.concatenate(
            [rng.choice(m, size=min(len(m), quota), replace=False) for m in members]
        )
        spare = np.setdiff1d(np.arange(len(x)), rows)
        rows = np.concatenate(
            [
                rows,
                rng.choice(
                    spare, size=min(len(spare), HYPER_ROWS - len(rows)), replace=False
                ),
            ]
        )

        def negative_evidence(log_params: NDArray) -> float:
            surface = Surface(
                inducing=inducing,
                lengthscales=np.exp(log_params[:-1]),
                amplitude=float(np.exp(log_params[-1])),
            )
            return -surface.fit(x=x[rows], y=y[rows]).evidence

        result = minimize(
            negative_evidence,
            x0=np.log([*([0.4] * x.shape[1]), 2.0]),
            method="L-BFGS-B",
            bounds=[
                *([tuple(np.log(LENGTH_BOUNDS))] * x.shape[1]),
                tuple(np.log(AMPLITUDE_BOUNDS)),
            ],
            options={"maxiter": 25, "eps": 0.05},
        )
        hyperparameters = (np.exp(result.x[:-1]), float(np.exp(result.x[-1])))
    lengthscales, amplitude = hyperparameters
    return Surface(
        inducing=inducing, lengthscales=lengthscales, amplitude=amplitude
    ).fit(x=x, y=y)


def fit_surfaces(*, draws: pd.DataFrame, seed: int) -> dict[str, Surface]:
    """Fit a surface to every family that has draws.

    Args:
        draws: Draws returned by :func:`in_design_space`.
        seed: Seed passed to :func:`fit_surface`.

    Returns:
        Fitted surface by family name.
    """
    surfaces = {}
    for name in FAMILIES:
        sub = draws[draws["family"] == name]
        if len(sub):
            surfaces[name] = fit_surface(
                x=unit_matrix(sub),
                y=sub["miss"].to_numpy(float),
                seed=seed,
                groups=sub["source"].to_numpy(),
            )
    return surfaces


# =============================================================================
# Design
# =============================================================================

CANDIDATES_PER_DRAW = 20
SPARSITY_NEIGHBOUR = 5
# Share of every refinement round spent on the space-filling criterion
FILL_FRACTION = 0.25
# Half-width, on the logit scale, of the band around the target miss rate
# that refinement draws aim for; 0.35 is about 3.6% to 6.9% at a 5% target
CONTOUR_BAND = 0.35


def candidate_points(*, count: int, seed: int) -> NDArray:
    """Scrambled Sobol points of the unit cube whose draws are runnable.

    Args:
        count: Points to generate before filtering, rounded up to a power
            of two.
        seed: Scrambling seed.

    Returns:
        (rows, 4) unit-cube points whose larger class stays within
        :data:`N_CLASS_MAX`.
    """
    unit = qmc.Sobol(d=DIM, seed=seed).random_base2(int(np.ceil(np.log2(count))))
    rows = design_rows(unit=unit, family=next(iter(FAMILIES.values())))
    return unit[rows[["n_pos", "n_neg"]].max(axis=1).to_numpy() <= N_CLASS_MAX]


def sparsity_weights(*, candidates: NDArray, existing: NDArray) -> NDArray:
    """Weight candidates by how little existing data surrounds them.

    The distance to the k-th nearest existing draw, raised to the dimension,
    is proportional to the reciprocal of a k-nearest-neighbour density
    estimate, so sampling by it evens out the combined design.

    Args:
        candidates: (c, 4) candidate points.
        existing: (e, 4) coordinates of existing draws of the same family.

    Returns:
        (c,) weights with mean 1, capped at their 95th percentile.
    """
    if len(existing) < SPARSITY_NEIGHBOUR:
        return np.ones(len(candidates))
    distance, _ = cKDTree(existing).query(candidates, k=SPARSITY_NEIGHBOUR)
    weights = distance[:, -1] ** DIM
    weights = np.minimum(weights, np.quantile(weights, 0.95))
    return weights / weights.mean()


def contour_weights(*, candidates: NDArray, surface: Surface, target: float) -> NDArray:
    """Posterior probability that the latent surface is near the target.

    Args:
        candidates: (c, 4) candidate points.
        surface: Fitted surface of the candidates' family.
        target: Target miss rate.

    Returns:
        (c,) probabilities that the latent logit lies within
        :data:`CONTOUR_BAND` of ``logit(target)``.
    """
    mean, sd = surface.predict(candidates)
    sd = np.maximum(sd, 1e-6)
    centre = logit(target)
    return norm.cdf((centre + CONTOUR_BAND - mean) / sd) - norm.cdf(
        (centre - CONTOUR_BAND - mean) / sd
    )


def next_design(
    *,
    batch: int,
    existing: pd.DataFrame,
    surfaces: dict[str, Surface] | None,
    cost_power: float,
    seed: int,
) -> pd.DataFrame:
    """Choose the draws of the next round.

    Args:
        batch: Draws to choose.
        existing: Every draw so far, from :func:`in_design_space`.
        surfaces: Fitted surfaces for a refinement round, or None for a
            space-filling round.
        cost_power: Exponent of the cost in the sampling weights; 0 ignores
            cost and 1 chooses per CPU-second.
        seed: Seed of the candidates and the selection.

    Returns:
        The chosen draws with a ``criterion`` column (``fill`` or
        ``contour``), in run order.
    """
    rng = np.random.default_rng(seed)
    pools = []
    for index, family in enumerate(FAMILIES.values()):
        candidates = candidate_points(
            count=CANDIDATES_PER_DRAW * batch, seed=seed + index
        )
        rows = design_rows(unit=candidates, family=family)
        deflator = cost_seconds(rows) ** cost_power
        fill = sparsity_weights(
            candidates=candidates,
            existing=unit_matrix(existing[existing["family"] == family.name]),
        )
        rows["w_fill"] = fill / deflator
        rows["w_contour"] = (
            contour_weights(
                candidates=candidates, surface=surfaces[family.name], target=ALPHA
            )
            / deflator
            if surfaces and family.name in surfaces
            else 0.0
        )
        pools.append(rows)
    pool = pd.concat(pools, ignore_index=True)
    n_contour = round((1 - FILL_FRACTION) * batch) if pool["w_contour"].sum() > 0 else 0
    chosen = []
    available = np.ones(len(pool), dtype=bool)
    for criterion, count in (("contour", n_contour), ("fill", batch - n_contour)):
        if count == 0:
            continue
        weights = np.where(available, pool[f"w_{criterion}"].to_numpy(float), 0.0)
        picked = rng.choice(
            len(pool),
            size=min(count, int((weights > 0).sum())),
            replace=False,
            p=weights / weights.sum(),
        )
        available[picked] = False
        chosen.append(pool.iloc[picked].assign(criterion=criterion))
    design = pd.concat(chosen, ignore_index=True).drop(columns=["w_fill", "w_contour"])
    # Slowest first, so the long draws start at once and the pool drains evenly
    order = np.argsort(-cost_seconds(design), kind="stable")
    design = design.iloc[order].reset_index(drop=True)
    design.insert(0, "draw", np.arange(len(design)))
    return design


# =============================================================================
# Simulation
# =============================================================================

WORKER_THREAD_ENV = {
    "RAYON_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
# A chunk closes once it holds this many draws or this much estimated work
CHUNK_DRAWS = 64
CHUNK_SECONDS = 20.0
CHECKPOINT_SECONDS = 120
# Peak memory of one unfloored band beyond the worker's baseline, per negative:
# the fiducial cloud holds 20,000 float32 draws at every grid point
BAND_BYTES_PER_NEGATIVE = 8.2e4


def band_memory_gb(frame: pd.DataFrame) -> NDArray:
    """Peak memory of each draw's band, beyond the worker's baseline.

    Args:
        frame: Draws with ``n_neg``.

    Returns:
        Estimated gigabytes per draw.
    """
    return BAND_BYTES_PER_NEGATIVE * (frame["n_neg"].to_numpy(float) + 1) / 1e9


def simulate_draw(*, row: dict, seed: int, round_index: int) -> dict | None:
    """Simulate one dataset and evaluate the unfloored band on it.

    Scores, grid and truth follow the main suite: scores and the evaluation
    grid in single precision, the true ROC evaluated on the band's native
    grid in double precision.

    Args:
        row: One row of a round's design.
        seed: Root seed of the study.
        round_index: Index of the round, which keys the random stream.

    Returns:
        The draw's record, or None if the family cannot reach the draw's AUC
        at its shape.
    """
    from studroc_paper.datagen.roc_to_dgp import map_lhs_to_dgp
    from studroc_paper.datagen.true_rocs import (
        make_heteroskedastic_gaussian_dgp,
        make_student_t_dgp,
    )
    from studroc_paper.eval.eval import evaluate_single_band
    from studroc_paper.methods.fiducial_band_rs import fiducial_band_rs

    makers = {
        "student_t": make_student_t_dgp,
        "hetero_gaussian": make_heteroskedastic_gaussian_dgp,
    }
    family = FAMILIES[row["family"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mapped = map_lhs_to_dgp(
            family.name,
            {
                "auc": np.array([row["auc"]]),
                family.shape_name: np.array([row["shape"]]),
            },
        )
    params = {
        key: float(value[0]) if isinstance(value, np.ndarray) else value
        for key, value in mapped.items()
    }
    if any(isinstance(v, float) and np.isnan(v) for v in params.values()):
        return None
    dgp = makers[family.name](**params)
    n_pos, n_neg = int(row["n_pos"]), int(row["n_neg"])
    rng = np.random.default_rng(
        np.random.SeedSequence(seed, spawn_key=(round_index, int(row["draw"])))
    )
    scores_pos, scores_neg = dgp.sample(n_pos, n_neg, rng)
    y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.float32)
    y_score = np.concatenate([scores_pos, scores_neg]).astype(np.float32)
    grid = np.linspace(0, 1, n_neg + 1)
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, lower, upper = fiducial_band_rs(
            y_true=y_true,
            y_score=y_score,
            alpha=ALPHA,
            n_threads=1,
            random_state=int(rng.integers(0, 2**63 - 1)),
        )
    seconds = time.perf_counter() - start
    band = evaluate_single_band(
        lower_band=lower,
        upper_band=upper,
        true_tpr=dgp.get_true_roc(grid),
        fpr_grid=grid.astype(np.float32),
    )
    ranks = rankdata(y_score)
    return {
        "source": f"round{round_index}",
        "draw": int(row["draw"]),
        "criterion": row["criterion"],
        "family": family.name,
        "shape": float(row["shape"]),
        "auc": float(np.trapezoid(dgp.get_true_roc(TRUE_AUC_GRID), TRUE_AUC_GRID)),
        "auc_target": float(row["auc"]),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "miss": not band.covers_entirely,
        "miss_below": bool(band.violation_below),
        "miss_above": bool(band.violation_above),
        "max_violation_below": float(band.max_violation_below),
        "max_violation_above": float(band.max_violation_above),
        "mean_band_width": float(band.mean_band_width),
        "empirical_auc": float(
            (ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        ),
        "seconds": seconds,
    }


def run_chunk(*, rows: list[dict], seed: int, round_index: int) -> list[dict]:
    """Worker entry point: simulate consecutive draws of a round.

    Args:
        rows: Design rows.
        seed: Root seed of the study.
        round_index: Index of the round.

    Returns:
        Records of the achievable draws, with unachievable ones marked by a
        record holding only ``draw`` and ``unachievable``.
    """
    records = []
    for row in rows:
        record = simulate_draw(row=row, seed=seed, round_index=round_index)
        records.append(record or {"draw": int(row["draw"]), "unachievable": True})
    return records


def run_round(
    *,
    design: pd.DataFrame,
    round_index: int,
    out_dir: Path,
    seed: int,
    workers: int,
    memory_gb: float,
) -> None:
    """Run a round's remaining draws, checkpointing as chunks finish.

    Chunks start in design order, except that a chunk whose band would take
    the running chunks' estimated memory past ``memory_gb`` waits, and the
    next chunk that fits starts in its place. A chunk larger than the whole
    budget runs alone.

    Args:
        design: The round's design.
        round_index: Index of the round.
        out_dir: Output directory.
        seed: Root seed of the study.
        workers: Worker processes.
        memory_gb: Budget for the bands of the running chunks, beyond the
            workers' baseline memory.
    """
    partial = out_dir / "partial" / f"round{round_index}"
    partial.mkdir(parents=True, exist_ok=True)
    # A temporary left by an interrupted write holds draws that are rerun
    for stale in partial.glob("part_*.tmp"):
        stale.unlink()
    done = {
        int(draw)
        for path in partial.glob("part_*.feather")
        for draw in pd.read_feather(path, columns=["draw"])["draw"]
    }
    rows = [r for r in design.to_dict("records") if r["draw"] not in done]
    chunks: list[list[dict]] = [[]]
    chunk_cost = 0.0
    costs = cost_seconds(pd.DataFrame(rows)) if rows else []
    for row, cost in zip(rows, costs, strict=True):
        if chunks[-1] and (
            len(chunks[-1]) >= CHUNK_DRAWS or chunk_cost + cost > CHUNK_SECONDS
        ):
            chunks.append([])
            chunk_cost = 0.0
        chunks[-1].append(row)
        chunk_cost += cost
    buffer: list[dict] = []

    def flush() -> None:
        if buffer:
            tmp = partial / f"part_{time.time_ns()}.tmp"
            pd.DataFrame.from_records(buffer).to_feather(tmp)
            tmp.replace(tmp.with_suffix(".feather"))
            buffer.clear()

    # Draws of a chunk run one after another, so its peak is its largest band
    pending = [
        (chunk, float(band_memory_gb(pd.DataFrame(chunk)).max()))
        for chunk in chunks
        if chunk
    ]
    running: dict[Future, float] = {}
    os.environ.update(WORKER_THREAD_ENV)
    last = time.monotonic()
    progress = tqdm(total=len(design), initial=len(done), desc=f"Round {round_index}")
    try:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            while pending or running:
                in_use = sum(running.values())
                index = 0
                while index < len(pending) and len(running) < workers:
                    chunk, memory = pending[index]
                    if running and in_use + memory > memory_gb:
                        index += 1
                        continue
                    future = pool.submit(
                        run_chunk, rows=chunk, seed=seed, round_index=round_index
                    )
                    running[future] = memory
                    in_use += memory
                    pending.pop(index)
                finished, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in finished:
                    del running[future]
                    records = future.result()
                    buffer.extend(records)
                    progress.update(len(records))
                if time.monotonic() - last >= CHECKPOINT_SECONDS:
                    flush()
                    last = time.monotonic()
    finally:
        flush()
        progress.close()
    frame = pd.concat(
        (pd.read_feather(p) for p in sorted(partial.glob("part_*.feather"))),
        ignore_index=True,
    )
    if "unachievable" in frame:
        frame = frame[frame["unachievable"].isna()].drop(columns="unachievable")
    frame = frame.sort_values("draw", ignore_index=True)
    frame.to_feather(out_dir / f"draws_round{round_index}.feather")
    for path in partial.glob("part_*.feather"):
        path.unlink()
    partial.rmdir()


def command_run(args: argparse.Namespace) -> None:
    """Design and run the next round, or resume an unfinished one."""
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    designs = sorted(out_dir.glob("design_round*.feather"))
    finished = {
        p.stem.removeprefix("draws_") for p in out_dir.glob("draws_round*.feather")
    }
    unfinished = [p for p in designs if p.stem.removeprefix("design_") not in finished]
    if unfinished:
        round_index = int(unfinished[0].stem.removeprefix("design_round"))
        design = pd.read_feather(unfinished[0])
        partial = out_dir / "partial" / f"round{round_index}"
        done = {
            int(draw)
            for path in partial.glob("part_*.feather")
            for draw in pd.read_feather(path, columns=["draw"])["draw"]
        }
        remaining = design[~design["draw"].isin(done)]
        cost = cost_seconds(remaining)
        print(
            f"Resuming round {round_index}: {len(remaining):,} of {len(design):,} "
            f"draws left, about {cost.sum() / 3600:.1f} CPU-hours, "
            f"{cost.sum() / 3600 / args.workers:.1f} h on {args.workers} workers"
        )
        if args.dry_run:
            return
    else:
        round_index = len(designs)
        existing = in_design_space(
            frame=load_draws(hot_start=args.hot_start, out_dir=out_dir)
        )
        by_source = existing.groupby(["source", "family"]).size()
        print(f"Existing draws in the design space: {len(existing):,}")
        if len(by_source):
            print(by_source.to_string())
        surfaces = None
        if round_index > 0 or args.refine_from_hot_start:
            print("Fitting surfaces for the refinement criterion...")
            surfaces = fit_surfaces(draws=existing, seed=args.seed)
        design = next_design(
            batch=args.batch,
            existing=existing,
            surfaces=surfaces,
            cost_power=args.cost_power,
            seed=args.seed + 1000 * round_index,
        )
        cost = cost_seconds(design)
        n_min = design[["n_pos", "n_neg"]].min(axis=1)
        print(
            f"Round {round_index}: {len(design):,} draws "
            f"({(design['criterion'] == 'contour').sum():,} contour), "
            f"about {cost.sum() / 3600:.1f} CPU-hours, "
            f"{cost.sum() / 3600 / args.workers:.1f} h on {args.workers} workers"
        )
        print(
            "  n_min quantiles (5/25/50/75/95%): "
            + ", ".join(
                f"{q:,.0f}" for q in n_min.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            )
        )
        if args.dry_run:
            return
        design.to_feather(out_dir / f"design_round{round_index}.feather")
    run_round(
        design=design,
        round_index=round_index,
        out_dir=out_dir,
        seed=args.seed,
        workers=args.workers,
        memory_gb=args.memory_gb,
    )
    print(f"Saved {out_dir / f'draws_round{round_index}.feather'}")


# =============================================================================
# Frontier and rule
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class Rule:
    """Admission rule ``n_min >= max(floor, constant * ratio^b / (1 - AUC)^a)``.

    Attributes:
        floor: Smallest admitted minority-class count.
        constant: Multiplier ``C``.
        auc_exponent: Exponent ``a`` of ``1 / (1 - AUC)``.
        ratio_exponent: Exponent ``b`` of the class ratio ``n_max / n_min``.
    """

    floor: float
    constant: float
    auc_exponent: float
    ratio_exponent: float

    def threshold(self, *, auc: NDArray, ratio: NDArray) -> NDArray:
        """Smallest admitted ``n_min`` at each (AUC, class ratio)."""
        return np.maximum(
            self.floor,
            self.constant
            * np.asarray(ratio, float) ** self.ratio_exponent
            / (1 - np.asarray(auc, float)) ** self.auc_exponent,
        )

    def admits(self, *, n_min: NDArray, auc: NDArray, ratio: NDArray) -> NDArray:
        """Whether the rule sends each case to the unfloored band."""
        return np.asarray(n_min, float) >= self.threshold(auc=auc, ratio=ratio)

    def describe(self) -> str:
        """The rule as a formula."""
        return (
            f"n_min >= max({self.floor:,.0f}, {self.constant:.3g} * "
            f"ratio^{self.ratio_exponent:.2f} / (1 - AUC)^{self.auc_exponent:.2f})"
        )


# Hand-derived rule from the closed-form corner screen, kept as the reference
# that fitted rules are compared against
REFERENCE_RULE = Rule(floor=100, constant=12, auc_exponent=2.5, ratio_exponent=2 / 3)

GRID_N = 56
GRID_AUC = 23
GRID_SHAPE = 17
GRID_RATIOS = (1.0, 1.5, 7 / 3, 4.0, 9.0, 19.0)
# Standard normal quantile of the upper credible bound used for the
# conservative frontier
UPPER_Z = float(norm.ppf(0.9))


def worst_case_grid(*, surfaces: dict[str, Surface]) -> pd.DataFrame:
    """Evaluate the worst case over shape, family and minority class on a grid.

    Args:
        surfaces: Fitted surface by family name.

    Returns:
        One row per (class ratio, AUC, n_min) with the worst-case miss
        probability under the posterior mean (``p_mean``) and under the upper
        credible bound (``p_upper``), the family and shape attaining the
        former, and whether the point is inside the design (``runnable``).
    """
    n_min = np.exp(np.linspace(*np.log(N_MIN_BOUNDS), GRID_N))
    auc = norm.cdf(np.linspace(*norm.ppf(AUC_BOUNDS), GRID_AUC))
    shape_unit = np.linspace(0, 1, GRID_SHAPE)
    blocks = []
    for ratio in GRID_RATIOS:
        mesh_n, mesh_a = np.meshgrid(n_min, auc, indexing="ij")
        best = {key: np.full(mesh_n.shape, -np.inf) for key in ("p_mean", "p_upper")}
        worst_family = np.full(mesh_n.shape, "", dtype=object)
        worst_shape = np.full(mesh_n.shape, np.nan)
        for name, surface in surfaces.items():
            family = FAMILIES[name]
            for sign in (-1.0, 1.0) if ratio > 1 else (1.0,):
                for u_shape in shape_unit:
                    unit = np.column_stack(
                        [
                            _scale(
                                np.log(mesh_n.ravel()),
                                bounds=tuple(np.log(N_MIN_BOUNDS)),
                            ),
                            np.full(
                                mesh_n.size,
                                0.5 + sign * np.log(ratio) / (2 * np.log(RATIO_MAX)),
                            ),
                            _scale(
                                norm.ppf(mesh_a.ravel()),
                                bounds=tuple(norm.ppf(AUC_BOUNDS)),
                            ),
                            np.full(mesh_n.size, u_shape),
                        ]
                    )
                    mean, sd = surface.predict(unit)
                    p_mean = expit(mean).reshape(mesh_n.shape)
                    p_upper = expit(mean + UPPER_Z * sd).reshape(mesh_n.shape)
                    improved = p_mean > best["p_mean"]
                    worst_family[improved] = name
                    worst_shape[improved] = np.exp(
                        _unscale(u_shape, bounds=tuple(np.log(family.shape_bounds)))
                    )
                    best["p_mean"] = np.maximum(best["p_mean"], p_mean)
                    best["p_upper"] = np.maximum(best["p_upper"], p_upper)
        blocks.append(
            pd.DataFrame(
                {
                    "ratio": ratio,
                    "auc": mesh_a.ravel(),
                    "n_min": mesh_n.ravel(),
                    "p_mean": best["p_mean"].ravel(),
                    "p_upper": best["p_upper"].ravel(),
                    "worst_family": worst_family.ravel(),
                    "worst_shape": worst_shape.ravel(),
                    "runnable": mesh_n.ravel() * ratio <= N_CLASS_MAX,
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)


def frontier(*, grid: pd.DataFrame, column: str, target: float) -> pd.DataFrame:
    """Find where the worst case becomes and stays safe as ``n_min`` grows.

    Args:
        grid: Output of :func:`worst_case_grid`.
        column: ``p_mean`` or ``p_upper``.
        target: Target miss rate.

    Returns:
        One row per (class ratio, AUC) with ``n_star``, the smallest grid
        ``n_min`` from which the worst case is at or below the target at
        every larger runnable ``n_min``; ``n_tested``, the largest runnable
        grid ``n_min``; and ``censored``, true when the worst case is still
        above the target at ``n_tested``.
    """
    rows = []
    for (ratio, auc), cell in grid[grid["runnable"]].groupby(["ratio", "auc"]):
        cell = cell.sort_values("n_min")
        unsafe = cell[column].to_numpy() > target
        n = cell["n_min"].to_numpy()
        # Index of the last unsafe grid point; the frontier is the next one
        last_unsafe = np.flatnonzero(unsafe).max(initial=-1)
        censored = last_unsafe == len(n) - 1
        rows.append(
            {
                "ratio": ratio,
                "auc": auc,
                "n_star": np.nan if censored else n[last_unsafe + 1],
                "n_tested": n[-1],
                "censored": censored,
            }
        )
    return pd.DataFrame(rows)


def fit_rule(*, frontier_table: pd.DataFrame) -> Rule:
    """Fit the least-slack power-law envelope of a frontier.

    For each candidate floor, a linear program on the log scale finds the
    power law that lies on or above every frontier point above the floor,
    and above the whole tested range where the frontier is censored, with
    the smallest total slack. The floor with the smallest total slack of
    ``max(floor, power law)`` over the uncensored frontier is kept.

    Args:
        frontier_table: Output of :func:`frontier`.

    Returns:
        The fitted rule.

    Raises:
        ValueError: If the frontier is censored everywhere.
    """
    required = np.log(
        np.where(
            frontier_table["censored"],
            frontier_table["n_tested"] * 1.0001,
            frontier_table["n_star"],
        ).astype(float)
    )
    x_auc = -np.log(1 - frontier_table["auc"].to_numpy(float))
    x_ratio = np.log(frontier_table["ratio"].to_numpy(float))
    observed = ~frontier_table["censored"].to_numpy(bool)
    if not observed.any():
        raise ValueError("The frontier is censored at every (AUC, class ratio)")
    best: tuple[float, Rule] | None = None
    for log_floor in np.unique(required[observed]):
        active = required > log_floor
        if active.any():
            design = np.column_stack(
                [np.ones(active.sum()), x_auc[active], x_ratio[active]]
            )
            solution = linprog(
                c=design.sum(axis=0),
                A_ub=-design,
                b_ub=-required[active],
                bounds=[(None, None), (0, None), (0, None)],
                method="highs",
            )
            if not solution.success:
                continue
            intercept, auc_exponent, ratio_exponent = solution.x
        else:
            intercept, auc_exponent, ratio_exponent = -np.inf, 0.0, 0.0
        predicted = np.maximum(
            log_floor, intercept + auc_exponent * x_auc + ratio_exponent * x_ratio
        )
        slack = float(np.sum(predicted[observed] - required[observed]))
        rule = Rule(
            floor=float(np.exp(log_floor)),
            constant=float(np.exp(intercept)),
            auc_exponent=float(auc_exponent),
            ratio_exponent=float(ratio_exponent),
        )
        if best is None or slack < best[0]:
            best = (slack, rule)
    return best[1]


# =============================================================================
# Checks and report
# =============================================================================


def markdown_table(frame: pd.DataFrame, *, floatfmt: str) -> str:
    """Render a frame, index included, as a Markdown table.

    Args:
        frame: Table to render.
        floatfmt: Format specification applied to floating-point cells.

    Returns:
        The table as Markdown text.
    """
    flat = frame.reset_index()

    def cell(value: object) -> str:
        if isinstance(value, float | np.floating):
            return format(value, floatfmt)
        return str(value)

    rows = [[str(column) for column in flat.columns]]
    rows += [[cell(value) for value in row] for row in flat.itertuples(index=False)]
    lines = ["| " + " | ".join(row) + " |" for row in rows]
    lines.insert(1, "|" + "|".join(" --- " for _ in flat.columns) + "|")
    return "\n".join(lines)


def cross_validate(
    *, draws: pd.DataFrame, surfaces: dict[str, Surface], folds: int, seed: int
) -> pd.DataFrame:
    """Predict every draw from a fit that excludes its fold.

    The kernel hyperparameters are those of the full fit; only the
    coefficients are refitted per fold.

    Args:
        draws: Draws returned by :func:`in_design_space`.
        surfaces: Full-data surfaces, which supply the hyperparameters.
        folds: Number of folds.
        seed: Seed of the fold assignment.

    Returns:
        The draws with an out-of-fold predicted miss probability ``p_cv``.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for name, surface in surfaces.items():
        sub = draws[draws["family"] == name].reset_index(drop=True)
        x, y = unit_matrix(sub), sub["miss"].to_numpy(float)
        fold = rng.integers(0, folds, len(sub))
        predicted = np.empty(len(sub))
        for k in range(folds):
            held = fold == k
            fitted = fit_surface(
                x=x[~held],
                y=y[~held],
                seed=seed,
                hyperparameters=(surface.lengthscales, surface.amplitude),
            )
            predicted[held] = expit(fitted.predict(x[held])[0])
        parts.append(sub.assign(p_cv=predicted))
    return pd.concat(parts, ignore_index=True)


def reliability_table(*, predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare out-of-fold predictions with observed miss rates.

    Args:
        predictions: Output of :func:`cross_validate`.

    Returns:
        Draw count, mean prediction, observed miss rate and its binomial
        standard error by predicted-probability bin.
    """
    edges = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.12, 0.2, 0.4, 1.0]
    binned = predictions.assign(
        bin=pd.cut(predictions["p_cv"], edges, include_lowest=True)
    )
    table = binned.groupby("bin", observed=True).agg(
        draws=("miss", "size"), predicted=("p_cv", "mean"), observed=("miss", "mean")
    )
    table["se"] = np.sqrt(table["observed"] * (1 - table["observed"]) / table["draws"])
    return table


def admitted_miss_rates(*, draws: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    """Observed miss rate of the draws a rule admits, by family and shape third.

    This uses no fitted surface: it is the direct evidence on whether the
    admitted region covers.

    Args:
        draws: Draws returned by :func:`in_design_space`.
        rule: The rule.

    Returns:
        Draw count, observed miss rate and the upper end of its 95% Wilson
        interval for each family and third of the shape range.
    """
    n_pos, n_neg = draws["n_pos"].to_numpy(float), draws["n_neg"].to_numpy(float)
    n_min, n_max = np.minimum(n_pos, n_neg), np.maximum(n_pos, n_neg)
    admitted = draws[
        rule.admits(n_min=n_min, auc=draws["auc"].to_numpy(float), ratio=n_max / n_min)
    ]
    table = (
        admitted.assign(shape_third=np.minimum((admitted["u3"] * 3).astype(int), 2))
        .groupby(["family", "shape_third"])
        .agg(draws=("miss", "size"), missed=("miss", "sum"))
    )
    z = norm.ppf(0.975)
    rate = table["missed"] / table["draws"]
    centre = (rate + z**2 / (2 * table["draws"])) / (1 + z**2 / table["draws"])
    half = (
        z
        * np.sqrt(rate * (1 - rate) / table["draws"] + z**2 / (4 * table["draws"] ** 2))
        / (1 + z**2 / table["draws"])
    )
    return table.assign(miss_rate=rate, wilson_upper=centre + half)


def rule_summary(*, rule: Rule, grid: pd.DataFrame) -> dict[str, float]:
    """Score a rule against the worst-case grid.

    Args:
        rule: The rule.
        grid: Output of :func:`worst_case_grid`.

    Returns:
        The share of runnable grid points admitted, and the largest
        worst-case miss probability among them under the posterior mean and
        under the upper credible bound.
    """
    runnable = grid[grid["runnable"]]
    admitted = runnable[
        rule.admits(
            n_min=runnable["n_min"], auc=runnable["auc"], ratio=runnable["ratio"]
        )
    ]
    return {
        "admitted_share": len(admitted) / len(runnable),
        "max_p_mean": float(admitted["p_mean"].max()) if len(admitted) else np.nan,
        "max_p_upper": float(admitted["p_upper"].max()) if len(admitted) else np.nan,
    }


def plot_surface(*, grid: pd.DataFrame, rules: dict[str, Rule], path: Path) -> None:
    """Draw the worst-case miss probability with each rule's threshold.

    Args:
        grid: Output of :func:`worst_case_grid`.
        rules: Rules to overlay, by label.
        path: Destination image.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    ratios = sorted(grid["ratio"].unique())
    fig, axes = plt.subplots(
        2, 3, figsize=(15, 8.5), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, ratio in zip(axes.ravel(), ratios, strict=True):
        cell = grid[grid["ratio"] == ratio]
        table = cell.pivot(index="auc", columns="n_min", values="p_mean")
        z_auc = norm.ppf(table.index.to_numpy())
        mesh = ax.pcolormesh(
            table.columns.to_numpy(),
            z_auc,
            table.to_numpy(),
            norm=TwoSlopeNorm(vmin=0.0, vcenter=ALPHA, vmax=0.30),
            cmap="RdBu_r",
            shading="nearest",
        )
        ax.contour(
            table.columns.to_numpy(),
            z_auc,
            table.to_numpy(),
            levels=[ALPHA],
            colors="black",
            linewidths=1.2,
        )
        for (label, rule), style in zip(rules.items(), ("-", "--", ":"), strict=False):
            ax.plot(
                rule.threshold(auc=table.index.to_numpy(), ratio=ratio),
                z_auc,
                style,
                color="tab:green",
                linewidth=2,
                label=label,
            )
        ax.axvline(N_CLASS_MAX / ratio, color="grey", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlim(*N_MIN_BOUNDS)
        ticks = [0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
        ax.set_yticks(norm.ppf(ticks), [f"{t:g}" for t in ticks])
        ax.set_title(f"class ratio {ratio:.3g} : 1 (prevalence {1 / (1 + ratio):.0%})")
    for ax in axes[-1]:
        ax.set_xlabel("smaller class count")
    for ax in axes[:, 0]:
        ax.set_ylabel("AUC")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.colorbar(mesh, ax=axes, label="worst-case miss probability (posterior mean)")
    fig.suptitle(
        "Unfloored band: worst case over shape, family and minority class; "
        "black = 5% contour, green = rules (admit to the right)"
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)


def command_fit(args: argparse.Namespace) -> None:
    """Fit the surfaces, derive the rules and write the report."""
    out_dir: Path = args.out_dir
    draws = in_design_space(frame=load_draws(hot_start=args.hot_start, out_dir=out_dir))
    draws["miss"] = draws["miss"].astype(bool)
    surfaces = fit_surfaces(draws=draws, seed=args.seed)
    grid = worst_case_grid(surfaces=surfaces)
    lines = ["# Auto rule: coverage surface of the unfloored band", ""]

    lines += ["## Data", ""]
    counts = draws.groupby(["source", "family"]).agg(
        draws=("miss", "size"), miss_rate=("miss", "mean")
    )
    lines += [markdown_table(counts, floatfmt=".4f"), ""]

    lines += ["## Surfaces", ""]
    for name, surface in surfaces.items():
        lines.append(
            f"- `{name}`: length scales (n_min, class ratio, AUC, shape) = "
            f"{np.round(surface.lengthscales, 3).tolist()}, amplitude "
            f"{surface.amplitude:.2f}"
        )
    predictions = cross_validate(
        draws=draws, surfaces=surfaces, folds=args.folds, seed=args.seed
    )
    clipped = np.clip(predictions["p_cv"], 1e-6, 1 - 1e-6)
    y = predictions["miss"].to_numpy(float)
    base = np.clip(predictions.groupby("family")["miss"].transform("mean"), 1e-6, 1)
    log_loss = -np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))
    log_loss_base = -np.mean(y * np.log(base) + (1 - y) * np.log(1 - base))
    lines += [
        "",
        f"{args.folds}-fold cross-validated log loss {log_loss:.4f} "
        f"(family miss rate alone: {log_loss_base:.4f}).",
        "",
        "Out-of-fold reliability:",
        "",
        markdown_table(reliability_table(predictions=predictions), floatfmt=".4f"),
        "",
    ]

    rules = {"reference": REFERENCE_RULE}
    frontiers = {}
    lines += ["## Frontier and rules", ""]
    for label, column in (("mean", "p_mean"), ("upper", "p_upper")):
        table = frontier(grid=grid, column=column, target=ALPHA)
        frontiers[label] = table
        try:
            rules[label] = fit_rule(frontier_table=table)
        except ValueError as err:
            lines.append(f"- `{label}`: no rule ({err}).")
    summary = pd.DataFrame(
        {
            label: {"rule": rule.describe(), **rule_summary(rule=rule, grid=grid)}
            for label, rule in rules.items()
        }
    ).T
    lines += [markdown_table(summary, floatfmt=".4f"), ""]
    merged = frontiers["mean"].merge(
        frontiers["upper"], on=["ratio", "auc"], suffixes=("_mean", "_upper")
    )
    merged.to_csv(out_dir / "frontier.csv", index=False)
    pivot = frontiers["mean"].assign(
        n_star=lambda t: np.where(t["censored"], np.inf, t["n_star"]),
        auc=lambda t: t["auc"].map("{:.3f}".format),
        ratio=lambda t: t["ratio"].map("{:.3g}".format),
    )
    pivot = pivot.pivot(index="auc", columns="ratio", values="n_star")[
        [f"{ratio:.3g}" for ratio in GRID_RATIOS]
    ]
    lines += [
        "Frontier `n*` under the posterior mean, by AUC (rows) and class ratio "
        "(columns); `inf` means still unsafe at the largest tested size:",
        "",
        markdown_table(pivot, floatfmt=",.0f"),
        "",
    ]

    lines += ["## Observed miss rate of admitted draws (no model)", ""]
    for label, rule in rules.items():
        lines += [
            f"`{label}`: {rule.describe()}",
            "",
            markdown_table(admitted_miss_rates(draws=draws, rule=rule), floatfmt=".4f"),
            "",
        ]

    plot_surface(grid=grid, rules=rules, path=out_dir / "surface.png")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "rules.json").write_text(
        json.dumps({k: asdict(r) for k, r in rules.items()}, indent=2), encoding="utf-8"
    )
    print("\n".join(lines))


def main() -> None:
    """Parse arguments and dispatch to a subcommand."""
    formatter = argparse.ArgumentDefaultsHelpFormatter
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--out-dir", type=Path, default=Path("data/results/auto_rule"))
    shared.add_argument(
        "--hot-start",
        type=Path,
        nargs="*",
        default=[
            Path("data/results") / name
            for name in (
                "shape_auc_study",
                "shape_auc_study_p10",
                "nov11",
                "nov11_p10",
                "nov11_p30",
            )
        ],
        help="Directories of earlier trial-level results",
    )
    shared.add_argument("--seed", type=int, default=20261117)
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=formatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser(
        "run",
        parents=[shared],
        formatter_class=formatter,
        help="Design and run the next round",
    )
    run.add_argument("--batch", type=int, default=40_000, help="Draws in a new round")
    run.add_argument(
        "--cost-power",
        type=float,
        default=0.25,
        help="Exponent of a draw's cost in the sampling weights",
    )
    run.add_argument(
        "--refine-from-hot-start",
        action="store_true",
        help="Use the contour criterion in round 0, from a fit to the hot start",
    )
    run.add_argument("--workers", type=int, default=os.cpu_count())
    run.add_argument(
        "--memory-gb",
        type=float,
        default=30.0,
        help="Memory budget for the running bands, beyond each worker's ~0.7 GB "
        "baseline; a 100,000-negative band takes about 8 GB",
    )
    run.add_argument(
        "--dry-run", action="store_true", help="Print the next design's size and cost"
    )
    run.set_defaults(handler=command_run)

    fit = commands.add_parser(
        "fit",
        parents=[shared],
        formatter_class=formatter,
        help="Fit the surfaces and derive the rule",
    )
    fit.add_argument("--folds", type=int, default=5)
    fit.set_defaults(handler=command_fit)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
