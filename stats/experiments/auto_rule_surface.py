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
ROC as a continuous function of the class counts, the AUC and the family's
shape parameter, separately for the two families whose unfloored band is
known to fail (Student-t location shift, by degrees of freedom;
heteroscedastic Gaussian, by sigma ratio). The surface is meant as a reference
that runtime calibration starts from, so it has to be accurate wherever users
land, not only near one contour. The rule is read off the worst case over
shape and family.

Draws are placed in a design space of

- the smaller class count ``n_min`` (log scale),
- the signed class log-ratio ``log(n_pos / n_neg)``,
- the AUC (probit scale), and
- the family's shape parameter (log scale).

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

Later rounds sample where data are worth most. A committee of surfaces is
refitted to all draws, and each candidate is weighted by the uncertainty of
the latent logit there: the reference surface's posterior standard deviation
combined with the committee's disagreement, which captures the structural
uncertainty a single model cannot see in itself. The uncertainty is scaled by
the candidate's relevance, highest where the predicted miss rate is near the
target and never below ``RELEVANCE_FLOOR``, so the whole surface keeps
improving, and divided by a power of the cost. A fixed fraction of every round
is still spent on the round-0 criterion.

Surface
-------
The reference surface is fitted on the expected corner counts
``log(n_pos (1 - AUC))`` and ``log(n_neg (1 - AUC))``, the probit AUC and the
log shape (:func:`surface_coordinates`); among the covariates and model
classes compared by ``surface_model_comparison.py`` this generalised best to
untested sample sizes and class ratios. The model is a Gaussian process with
a Bernoulli likelihood, constant mean and anisotropic Matern-5/2 kernel, so
that away from the data it reverts to a flat level rather than extrapolating
a trend, in its low-rank form (subset of regressors on a fixed space-filling
set of inducing points). In weight space this is a penalised
logistic regression on radial basis functions, so it is fitted by penalised
IRLS on every draw at once; the length scales and amplitude maximise the
Laplace approximation to the marginal likelihood. Nothing constrains the
surface to be monotone in any covariate: coverage is known to dip and recover
as n grows, and the fit has to be able to show it.

The fit is checked by K-fold cross-validation (a reliability table of
predicted against observed miss rates) and, free of any model, by the observed
miss rate of the draws that a rule admits.

A round can instead replicate a list of fixed cells (``run --cells``), to
settle a contested region of the surface without leaning on the model: each
cell's observed miss rate is compared with the surface in the report.

Rule
----
The rule is calibrated to an assurance, not to the central estimate. At every
point of a grid over ``n_min``, AUC and class ratio, the exceedance
probability is the largest posterior probability, over shape, family and
minority class, that the miss rate exceeds the target. A point is unsafe when
its exceedance probability is above ``1 - assurance``, and the rule may admit
no unsafe point with a majority class of at most :data:`ROUTED_N_MAX`, the
extent of the dense paired data. Beyond it the surface rests on sparse draws
near a 5% miss rate, so the rule is checked there against replicated cells
instead (see :data:`ROUTED_N_MAX`).

The rule is a conjunction of boundary checks, each a one- or two-parameter
bound that ropes off one slice of the unsafe region (:class:`Check`): a floor
on ``n_min``, a bound on the expected minority count beyond the corner
``n_min (1 - AUC)^a``, optionally deflated by the class ratio, and caps on the
AUC and the class ratio. For every combination of up to
:data:`MAX_CHECKS` kinds, the closing check's threshold is solved exactly for
every candidate setting of the others, and the combination that admits the
largest share of the grid wins. The report gives the best rule for each number
of checks and recommends the fewest checks within
:data:`PARSIMONY_TOLERANCE` of the best, with thresholds rounded to round
numbers on the strict side. It attributes to each check the unsafe slice that
it alone excludes, so that a check can be read as one failure mechanism.

Usage
-----
    uv run python stats/experiments/auto_rule_surface.py run --batch 40000 --dry-run
    uv run python stats/experiments/auto_rule_surface.py run --batch 40000
    uv run python stats/experiments/auto_rule_surface.py fit
    uv run python stats/experiments/auto_rule_surface.py report
    uv run python stats/experiments/auto_rule_surface.py run --cells c.csv --reps 2000

``run`` resumes an unfinished round; otherwise it designs and runs the next
one. ``fit`` writes ``report.md``, ``grid.feather``, ``rules.json`` and
``surface.png`` to the output directory, and caches the surfaces, the
cross-validation and the band timings in ``fit_cache.pkl``. ``report`` rewrites
those outputs from the cache in seconds, and refuses if the draws have changed
since the fit.
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import pickle
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import ndimage
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
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
        title: Readable name of the family.
    """

    name: str
    shape_name: str
    shape_bounds: tuple[float, float]
    label: str
    title: str


FAMILIES = {
    family.name: family
    for family in (
        Family(
            name="student_t",
            shape_name="df",
            shape_bounds=(1.1, 30.0),
            label="degrees of freedom",
            title="Student-t location shift",
        ),
        Family(
            name="hetero_gaussian",
            shape_name="sigma_ratio",
            shape_bounds=(0.2, 5.0),
            label="sigma ratio (positive / negative)",
            title="heteroscedastic Gaussian",
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


# Ranges of the surface covariates over the design space
LOG_COUNT_BOUNDS = (np.log(N_MIN_BOUNDS[0]), np.log(N_CLASS_MAX))
LOG_CORNER_BOUNDS = (
    np.log(N_MIN_BOUNDS[0] * (1 - AUC_BOUNDS[1])),
    np.log(N_CLASS_MAX * (1 - AUC_BOUNDS[0])),
)
Covariates = Literal["corner", "classes"]


def surface_coordinates(
    *, frame: pd.DataFrame, family: Family, covariates: Covariates = "corner"
) -> NDArray:
    """Covariates a surface is fitted on, scaled to the unit cube.

    ``corner`` uses the expected corner counts ``log(n (1 - AUC))`` of each
    class, roughly how many observations of that class fall on the wrong side
    of a random member of the other and so populate the ROC's corners;
    ``classes`` uses the log class counts. Both add the probit AUC and the log
    shape. The scaling is fixed by the design space, so draws slightly
    outside it map slightly outside the cube.

    Args:
        frame: Draws with ``n_pos``, ``n_neg``, ``auc`` and ``shape``.
        family: The draws' family.
        covariates: Which class-count covariates to use.

    Returns:
        (rows, 4) coordinates.
    """
    n_pos = frame["n_pos"].to_numpy(float)
    n_neg = frame["n_neg"].to_numpy(float)
    auc = frame["auc"].to_numpy(float)
    if covariates == "corner":
        counts, bounds = [n_pos * (1 - auc), n_neg * (1 - auc)], LOG_CORNER_BOUNDS
    else:
        counts, bounds = [n_pos, n_neg], LOG_COUNT_BOUNDS
    return np.column_stack(
        [
            *(_scale(np.log(c), bounds=bounds) for c in counts),
            _scale(norm.ppf(auc), bounds=tuple(norm.ppf(AUC_BOUNDS))),
            _scale(
                np.log(frame["shape"].to_numpy(float)),
                bounds=tuple(np.log(family.shape_bounds)),
            ),
        ]
    )


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
    subset-of-regressors approximation to the Matern kernel. Away from the
    data the kernel term fades and the logit reverts to the mean
    ``[1, x] @ beta``, or to the intercept alone without a linear mean.

    Attributes:
        inducing: (m, d) inducing points.
        lengthscales: (d,) kernel length scales.
        amplitude: Kernel standard deviation.
        linear_mean: Whether the mean is linear in ``x`` or constant.
        coefficients: Posterior mode of ``[beta, w]``.
        covariance: Laplace posterior covariance of the coefficients.
        evidence: Laplace log marginal likelihood of the fitted rows.
    """

    def __init__(
        self,
        *,
        inducing: NDArray,
        lengthscales: NDArray,
        amplitude: float,
        linear_mean: bool = True,
    ) -> None:
        """Set the kernel; :meth:`fit` estimates the coefficients.

        Args:
            inducing: (m, d) inducing points.
            lengthscales: (d,) kernel length scales.
            amplitude: Kernel standard deviation.
            linear_mean: Whether the mean is linear in ``x`` or constant.
        """
        self.inducing = inducing
        self.linear_mean = linear_mean
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
            features, without the d linear columns for a constant mean.
        """
        cross = matern52(a=self.inducing, b=x, lengthscales=self.lengthscales)
        phi = solve_triangular(self._chol, cross, lower=True).T
        mean = [np.ones((len(x), 1)), x] if self.linear_mean else [np.ones((len(x), 1))]
        return np.hstack([*mean, self.amplitude * phi])

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
        n_mean = 1 + (x.shape[1] if self.linear_mean else 0)
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
    linear_mean: bool = True,
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
        linear_mean: Whether the mean is linear in ``x`` or constant.

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
                linear_mean=linear_mean,
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
        inducing=inducing,
        lengthscales=lengthscales,
        amplitude=amplitude,
        linear_mean=linear_mean,
    ).fit(x=x, y=y)


def fit_surfaces(
    *, draws: pd.DataFrame, seed: int, covariates: Covariates = "corner"
) -> dict[str, Surface]:
    """Fit a constant-mean surface to every family that has draws.

    Args:
        draws: Draws returned by :func:`in_design_space`.
        seed: Seed passed to :func:`fit_surface`.
        covariates: Covariates of :func:`surface_coordinates`.

    Returns:
        Fitted surface by family name.
    """
    surfaces = {}
    for name, family in FAMILIES.items():
        sub = draws[draws["family"] == name]
        if len(sub):
            surfaces[name] = fit_surface(
                x=surface_coordinates(frame=sub, family=family, covariates=covariates),
                y=sub["miss"].to_numpy(float),
                seed=seed,
                groups=sub["source"].to_numpy(),
                linear_mean=False,
            )
    return surfaces


@dataclass(frozen=True, kw_only=True)
class Committee:
    """Surfaces of one family whose disagreement measures model uncertainty.

    Attributes:
        family: The family.
        reference: Gaussian process on the corner covariates, the surface the
            study reports.
        alternative: Gaussian process on the log class counts.
        boosted: Gradient-boosted trees on the corner covariates.
    """

    family: Family
    reference: Surface
    alternative: Surface
    boosted: object

    def predict(self, frame: pd.DataFrame) -> tuple[NDArray, NDArray, NDArray]:
        """Reference logit, its posterior SD, and the committee's spread.

        Args:
            frame: Draws with ``n_pos``, ``n_neg``, ``auc`` and ``shape``.

        Returns:
            The reference surface's latent mean and standard deviation, and
            the standard deviation of the three members' latent logits.
        """
        corner = surface_coordinates(frame=frame, family=self.family)
        mean, sd = self.reference.predict(corner)
        alternative = self.alternative.predict(
            surface_coordinates(frame=frame, family=self.family, covariates="classes")
        )[0]
        boosted = logit(
            np.clip(self.boosted.predict_proba(corner)[:, 1], 1e-6, 1 - 1e-6)
        )
        spread = np.std(np.column_stack([mean, alternative, boosted]), axis=1)
        return mean, sd, spread


def fit_committees(*, draws: pd.DataFrame, seed: int) -> dict[str, Committee]:
    """Fit the surface committee of every family that has draws.

    Args:
        draws: Draws returned by :func:`in_design_space`.
        seed: Seed of the fits.

    Returns:
        Committee by family name.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    references = fit_surfaces(draws=draws, seed=seed)
    alternatives = fit_surfaces(draws=draws, seed=seed, covariates="classes")
    committees = {}
    for name, reference in references.items():
        family = FAMILIES[name]
        sub = draws[draws["family"] == name]
        boosted = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=1000,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed,
        ).fit(surface_coordinates(frame=sub, family=family), sub["miss"].to_numpy(int))
        committees[name] = Committee(
            family=family,
            reference=reference,
            alternative=alternatives[name],
            boosted=boosted,
        )
    return committees


# =============================================================================
# Design
# =============================================================================

CANDIDATES_PER_DRAW = 20
SPARSITY_NEIGHBOUR = 5
# Share of every refinement round spent on the space-filling criterion
FILL_FRACTION = 0.25
# Relevance of a candidate falls off as a Gaussian in the distance of its
# predicted logit from the target's, with this scale; 1 spans about 2% to 12.5%
# at a 5% target. It never falls below RELEVANCE_FLOOR
RELEVANCE_SCALE = 1.0
RELEVANCE_FLOOR = 0.25


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


def value_weights(
    *, rows: pd.DataFrame, committee: Committee, target: float
) -> NDArray:
    """Value of a draw at each candidate: relevant uncertainty of the surface.

    The uncertainty is the root sum of squares of the reference surface's
    posterior SD and the committee's spread, both on the logit scale. It is
    scaled by a relevance that peaks where the predicted miss rate equals the
    target and never falls below :data:`RELEVANCE_FLOOR`.

    Args:
        rows: Candidate draws of the committee's family.
        committee: Fitted committee of the family.
        target: Target miss rate.

    Returns:
        (c,) non-negative weights.
    """
    mean, sd, spread = committee.predict(rows)
    relevance = RELEVANCE_FLOOR + (1 - RELEVANCE_FLOOR) * np.exp(
        -0.5 * ((mean - logit(target)) / RELEVANCE_SCALE) ** 2
    )
    return relevance * np.sqrt(sd**2 + spread**2)


def next_design(
    *,
    batch: int,
    existing: pd.DataFrame,
    committees: dict[str, Committee] | None,
    cost_power: float,
    seed: int,
) -> pd.DataFrame:
    """Choose the draws of the next round.

    Args:
        batch: Draws to choose.
        existing: Every draw so far, from :func:`in_design_space`.
        committees: Fitted committees for a value-driven round, or None for a
            space-filling round.
        cost_power: Exponent of the cost in the sampling weights; 0 ignores
            cost and 1 chooses per CPU-second.
        seed: Seed of the candidates and the selection.

    Returns:
        The chosen draws with a ``criterion`` column (``fill`` or
        ``value``), in run order.
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
        rows["w_value"] = (
            value_weights(rows=rows, committee=committees[family.name], target=ALPHA)
            / deflator
            if committees and family.name in committees
            else 0.0
        )
        pools.append(rows)
    pool = pd.concat(pools, ignore_index=True)
    n_value = round((1 - FILL_FRACTION) * batch) if pool["w_value"].sum() > 0 else 0
    chosen = []
    available = np.ones(len(pool), dtype=bool)
    for criterion, count in (("value", n_value), ("fill", batch - n_value)):
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
    design = pd.concat(chosen, ignore_index=True).drop(columns=["w_fill", "w_value"])
    # Slowest first, so the long draws start at once and the pool drains evenly
    order = np.argsort(-cost_seconds(design), kind="stable")
    design = design.iloc[order].reset_index(drop=True)
    design.insert(0, "draw", np.arange(len(design)))
    return design


CELL_COLUMNS = ["family", "shape", "auc", "n_pos", "n_neg"]


def cell_design(*, cells: pd.DataFrame, reps: int) -> pd.DataFrame:
    """Replicate fixed cells into a round's design.

    Args:
        cells: One row per cell with :data:`CELL_COLUMNS`, and optionally
            ``reps`` to override the number of draws of that cell.
        reps: Draws per cell without its own ``reps``.

    Returns:
        The design, with a ``cell`` column indexing the rows of ``cells``,
        ``criterion`` set to ``cell``, slowest draws first.

    Raises:
        ValueError: If a column is missing, a family is unknown, or a cell
            lies outside the design space.
    """
    missing = set(CELL_COLUMNS) - set(cells.columns)
    if missing:
        raise ValueError(f"Cells lack columns {sorted(missing)}")
    unknown = set(cells["family"]) - set(FAMILIES)
    if unknown:
        raise ValueError(f"Unknown families {sorted(unknown)}")
    counts = (
        cells["reps"].fillna(reps).astype(int).to_numpy()
        if "reps" in cells
        else np.full(len(cells), reps)
    )
    cells = cells[CELL_COLUMNS].reset_index(drop=True)
    outside = in_design_space(frame=cells.assign(source="cells", miss=False))
    if len(outside) != len(cells):
        raise ValueError("Some cells lie outside the design space")
    design = cells.loc[np.repeat(cells.index.to_numpy(), counts)].reset_index(
        names="cell"
    )
    design["criterion"] = "cell"
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


def empirical_binormal_slope(*, scores_pos: NDArray, scores_neg: NDArray) -> float:
    """Slope of the empirical ROC on the binormal (probit-probit) scale.

    The slope is the least-squares fit of ``probit(TPR)`` on ``probit(FPR)``
    over the empirical ROC's interior points; under a binormal model it
    estimates the ratio of the negative to the positive score spread.

    Args:
        scores_pos: Scores of the positives.
        scores_neg: Scores of the negatives.

    Returns:
        The slope, or NaN when fewer than three interior points exist.
    """
    thresholds = np.unique(np.concatenate([scores_pos, scores_neg]))
    tpr = 1 - np.searchsorted(np.sort(scores_pos), thresholds) / len(scores_pos)
    fpr = 1 - np.searchsorted(np.sort(scores_neg), thresholds) / len(scores_neg)
    inside = (tpr > 0) & (tpr < 1) & (fpr > 0) & (fpr < 1)
    if inside.sum() < 3:
        return float("nan")
    return float(np.polyfit(norm.ppf(fpr[inside]), norm.ppf(tpr[inside]), 1)[0])


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
        "cell": int(row.get("cell", -1)),
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
        "pos_above_all_neg": int(np.sum(scores_pos > scores_neg.max())),
        "neg_below_all_pos": int(np.sum(scores_neg < scores_pos.min())),
        "binormal_slope": empirical_binormal_slope(
            scores_pos=scores_pos, scores_neg=scores_neg
        ),
        "seconds": seconds,
    }


def completed_records(partial: Path) -> pd.DataFrame:
    """Records a round has checkpointed so far.

    Args:
        partial: The round's checkpoint directory.

    Returns:
        Every checkpointed record, achievable or not; an empty frame with a
        ``draw`` column if there are none.
    """
    parts = [pd.read_feather(path) for path in sorted(partial.glob("part_*.feather"))]
    if not parts:
        return pd.DataFrame({"draw": pd.Series(dtype=int)})
    return pd.concat(parts, ignore_index=True)


def finalize_round(*, round_index: int, out_dir: Path) -> None:
    """Write a round's checkpointed draws as its draws file and clear them.

    Args:
        round_index: Index of the round.
        out_dir: Output directory.

    Raises:
        ValueError: If the round has no checkpointed draws.
    """
    partial = out_dir / "partial" / f"round{round_index}"
    frame = completed_records(partial)
    if frame.empty:
        raise ValueError(f"Round {round_index} has no checkpointed draws")
    if "unachievable" in frame:
        frame = frame[frame["unachievable"].isna()].drop(columns="unachievable")
    frame = frame.sort_values("draw", ignore_index=True)
    frame.to_feather(out_dir / f"draws_round{round_index}.feather")
    for path in partial.glob("part_*"):
        path.unlink()
    partial.rmdir()


# Edges of the negative-count strata that share a trimmed round's budget
TRIM_STRATA = (0, 1_000, 10_000, 20_000, 50_000, np.inf)


def measured_cost_factor(records: pd.DataFrame) -> float:
    """Ratio of measured band seconds to :func:`cost_seconds` over draws run.

    Args:
        records: Checkpointed records of a round.

    Returns:
        The ratio, or 1 when no draw has a measured time.
    """
    if "seconds" not in records:
        return 1.0
    timed = records.dropna(subset=["seconds"])
    if timed.empty:
        return 1.0
    return float(timed["seconds"].sum() / cost_seconds(timed).sum())


def trim_design(
    *, design: pd.DataFrame, done: set[int], cpu_hours: float, seed: int
) -> pd.DataFrame:
    """Keep a round's completed draws and a stratified sample of the rest.

    The remaining draws are stratified by negative count (:data:`TRIM_STRATA`)
    and every stratum gets an equal share of the budget; a stratum whose
    draws cost less than its share passes the surplus to the others. Within a
    stratum the draws are taken in random order until the share is spent.
    The sample thins the design without reweighting it, so the round remains
    a draw from its own design criterion.

    Args:
        design: The round's design.
        done: Draw indices already completed.
        cpu_hours: Budget for the sampled draws in hours of
            :func:`cost_seconds`.
        seed: Seed of the random order.

    Returns:
        The completed and sampled draws, in design order.
    """
    completed = design["draw"].isin(done).to_numpy()
    remaining = design[~completed]
    seconds = cost_seconds(remaining)
    stratum = pd.cut(remaining["n_neg"], TRIM_STRATA, labels=False).to_numpy()
    totals = {k: float(seconds[stratum == k].sum()) for k in np.unique(stratum)}
    share = np.inf
    left = cpu_hours * 3600
    for count, total in enumerate(sorted(totals.values())):
        even = left / (len(totals) - count)
        if total > even:
            share = even
            break
        left -= total
    order = np.random.default_rng(seed).permutation(len(remaining))
    keep = np.zeros(len(remaining), dtype=bool)
    for k in totals:
        members = order[stratum[order] == k]
        keep[members[np.cumsum(seconds[members]) <= share]] = True
    kept = np.concatenate([np.flatnonzero(completed), np.flatnonzero(~completed)[keep]])
    return design.iloc[np.sort(kept)].reset_index(drop=True)


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
    done = set(completed_records(partial)["draw"].astype(int))
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
    finalize_round(round_index=round_index, out_dir=out_dir)


def command_run(args: argparse.Namespace) -> None:
    """Design and run the next round, or resume or close an unfinished one."""
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
        records = completed_records(out_dir / "partial" / f"round{round_index}")
        done = set(records["draw"].astype(int))
        factor = measured_cost_factor(records)
        closing = args.close_round is not None
        if closing:
            design = trim_design(
                design=design,
                done=done,
                cpu_hours=args.close_round / factor,
                seed=args.seed + 1000 * round_index + 1,
            )
        remaining = design[~design["draw"].isin(done)]
        seconds = cost_seconds(remaining) * factor
        memory_hours = float(
            (band_memory_gb(remaining) * seconds).sum() / args.memory_gb / 3600
        )
        print(
            f"{'Closing' if closing else 'Resuming'} round {round_index}: "
            f"{len(done):,} draws done, {len(remaining):,} to run, about "
            f"{seconds.sum() / 3600:.1f} CPU-hours at {factor:.2f}x the cost "
            f"model; at least {seconds.sum() / 3600 / args.workers:.1f} h on "
            f"{args.workers} workers and {memory_hours:.1f} h within "
            f"{args.memory_gb:g} GB"
        )
        if closing:
            strata = pd.cut(remaining["n_neg"], TRIM_STRATA)
            print(remaining.groupby(strata, observed=True).size().to_string())
        if args.dry_run:
            return
        if closing:
            untrimmed = out_dir / f"untrimmed_design_round{round_index}.feather"
            if not untrimmed.exists():
                unfinished[0].replace(untrimmed)
            design.to_feather(unfinished[0])
    elif args.close_round is not None:
        raise SystemExit("--close-round: there is no unfinished round to close")
    elif args.cells is not None:
        round_index = len(designs)
        design = cell_design(cells=pd.read_csv(args.cells), reps=args.reps)
    else:
        round_index = len(designs)
        existing = in_design_space(
            frame=load_draws(hot_start=args.hot_start, out_dir=out_dir)
        )
        by_source = existing.groupby(["source", "family"]).size()
        print(f"Existing draws in the design space: {len(existing):,}")
        if len(by_source):
            print(by_source.to_string())
        committees = None
        if round_index > 0 or args.refine_from_hot_start:
            print("Fitting the surface committees for the value criterion...")
            committees = fit_committees(draws=existing, seed=args.seed)
        design = next_design(
            batch=args.batch,
            existing=existing,
            committees=committees,
            cost_power=args.cost_power,
            seed=args.seed + 1000 * round_index,
        )
    if not unfinished:
        cost = cost_seconds(design)
        n_min = design[["n_pos", "n_neg"]].min(axis=1)
        print(
            f"Round {round_index}: {len(design):,} draws "
            f"({(design['criterion'] == 'value').sum():,} value), "
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
# Assurance grid
# =============================================================================

GRID_N = 56
GRID_AUC = 33
GRID_SHAPE = 17
GRID_RATIOS = (1.0, 1.5, 2.0, 7 / 3, 3.0, 4.0, 6.0, 9.0, 13.0, 19.0)
# Gauss-Hermite nodes for the expected excess of the miss rate over the target
EXCESS_NODES = 24


def expected_excess(*, mean: NDArray, sd: NDArray, target: float) -> NDArray:
    """Posterior expectation of ``max(p - target, 0)`` for a normal latent logit.

    Args:
        mean: Latent posterior means.
        sd: Latent posterior standard deviations.
        target: Target miss rate.

    Returns:
        The expected excess at each point.
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(EXCESS_NODES)
    latent = mean[:, None] + sd[:, None] * nodes[None, :]
    excess = np.maximum(expit(latent) - target, 0.0)
    return excess @ weights / weights.sum()


def worst_case_grid(*, surfaces: dict[str, Surface], target: float) -> pd.DataFrame:
    """Evaluate the worst case over shape, family and minority class on a grid.

    Args:
        surfaces: Fitted surface by family name.
        target: Target miss rate.

    Returns:
        One row per runnable (class ratio, AUC, n_min) with the worst-case
        posterior-mean miss probability (``p_mean``), the worst-case
        posterior probability that the miss rate exceeds the target
        (``exceedance``), the worst-case expected excess over the target
        (``excess``), and the family, shape and minority class attaining the
        worst exceedance.
    """
    n_min = np.exp(np.linspace(*np.log(N_MIN_BOUNDS), GRID_N))
    auc = norm.cdf(np.linspace(*norm.ppf(AUC_BOUNDS), GRID_AUC))
    shape_unit = np.linspace(0, 1, GRID_SHAPE)
    blocks = []
    for ratio in GRID_RATIOS:
        mesh_n, mesh_a = np.meshgrid(n_min, auc, indexing="ij")
        mesh_n, mesh_a = mesh_n.ravel(), mesh_a.ravel()
        runnable = mesh_n * ratio <= N_CLASS_MAX
        mesh_n, mesh_a = mesh_n[runnable], mesh_a[runnable]
        p_mean = np.full(len(mesh_n), -np.inf)
        exceedance = np.full(len(mesh_n), -np.inf)
        excess = np.full(len(mesh_n), -np.inf)
        worst = {
            "worst_family": np.full(len(mesh_n), "", dtype=object),
            "worst_shape": np.full(len(mesh_n), np.nan),
            "worst_minority": np.full(len(mesh_n), "", dtype=object),
        }
        for name, surface in surfaces.items():
            family = FAMILIES[name]
            log_bounds = tuple(np.log(family.shape_bounds))
            for minority in ("positive", "negative") if ratio > 1 else ("negative",):
                n_larger = mesh_n * ratio
                for u_shape in shape_unit:
                    shape = float(np.exp(_unscale(u_shape, bounds=log_bounds)))
                    points = pd.DataFrame(
                        {
                            "n_pos": mesh_n if minority == "positive" else n_larger,
                            "n_neg": n_larger if minority == "positive" else mesh_n,
                            "auc": mesh_a,
                            "shape": shape,
                        }
                    )
                    mean, sd = surface.predict(
                        surface_coordinates(frame=points, family=family)
                    )
                    exceed = norm.sf((logit(target) - mean) / np.maximum(sd, 1e-12))
                    worse = exceed > exceedance
                    worst["worst_family"][worse] = name
                    worst["worst_shape"][worse] = shape
                    worst["worst_minority"][worse] = minority
                    exceedance = np.maximum(exceedance, exceed)
                    p_mean = np.maximum(p_mean, expit(mean))
                    excess = np.maximum(
                        excess, expected_excess(mean=mean, sd=sd, target=target)
                    )
        blocks.append(
            pd.DataFrame(
                {
                    "ratio": ratio,
                    "auc": mesh_a,
                    "n_min": mesh_n,
                    "p_mean": p_mean,
                    "exceedance": exceedance,
                    "excess": excess,
                    **worst,
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)


# =============================================================================
# Rule
# =============================================================================

CheckKind = Literal["floor", "corner", "imbalance_corner", "auc_cap", "ratio_cap"]
# Kinds whose check admits large values of the statistic; the caps admit
# small values
LOWER_BOUNDS: frozenset[str] = frozenset({"floor", "corner", "imbalance_corner"})


@dataclass(frozen=True, kw_only=True)
class Check:
    """One boundary of the admitted region.

    The kinds and the cases each admits:

    - ``floor``: ``n_min >= threshold``
    - ``corner``: ``n_min (1 - AUC)^a >= threshold``; at ``a = 1`` the
      expected count of the minority class beyond the ROC's corner
    - ``imbalance_corner``: ``n_min (1 - AUC)^a / ratio^b >= threshold``
    - ``auc_cap``: ``AUC <= threshold``
    - ``ratio_cap``: ``n_max / n_min <= threshold``

    Attributes:
        kind: Kind of the check.
        threshold: Threshold on the natural scale of the kind's statistic.
        auc_exponent: Exponent ``a`` of the corner kinds.
        ratio_exponent: Exponent ``b`` of ``imbalance_corner``.
    """

    kind: CheckKind
    threshold: float
    auc_exponent: float = 0.0
    ratio_exponent: float = 0.0

    def score(self, *, n_min: NDArray, auc: NDArray, ratio: NDArray) -> NDArray:
        """Statistic on the scale of :meth:`bound`, admitted when at least it."""
        return check_score(
            kind=self.kind,
            auc_exponent=self.auc_exponent,
            ratio_exponent=self.ratio_exponent,
            n_min=np.asarray(n_min, float),
            auc=np.asarray(auc, float),
            ratio=np.asarray(ratio, float),
        )

    def bound(self) -> float:
        """The threshold on the scale of :meth:`score`."""
        if self.kind == "auc_cap":
            return -self.threshold
        if self.kind in LOWER_BOUNDS:
            return float(np.log(self.threshold))
        return float(-np.log(self.threshold))

    def admits(self, *, n_min: NDArray, auc: NDArray, ratio: NDArray) -> NDArray:
        """Whether the check lets each case through."""
        return self.score(n_min=n_min, auc=auc, ratio=ratio) >= self.bound()

    def describe(self) -> str:
        """The check as an inequality."""
        a, b, t = self.auc_exponent, self.ratio_exponent, self.threshold
        match self.kind:
            case "floor":
                return f"n_min >= {t:,.4g}"
            case "corner":
                return f"n_min * (1 - AUC)^{a:g} >= {t:,.4g}"
            case "imbalance_corner":
                return f"n_min * (1 - AUC)^{a:g} / ratio^{b:.3g} >= {t:,.4g}"
            case "auc_cap":
                return f"AUC <= {t:.4g}"
            case "ratio_cap":
                return f"n_max / n_min <= {t:.4g}"

    def explain(self) -> str:
        """The check in words, for a reader who does not know the study."""
        match self.kind:
            case "floor":
                return f"the smaller class has at least {self.threshold:,.4g} cases"
            case "corner" | "imbalance_corner":
                return (
                    "the AUC is low enough for the sample size: "
                    f"`{self.describe()}`. The left side grows with the expected "
                    "number of minority cases beyond the ROC's corner, so the "
                    "largest AUC allowed rises with the smaller class"
                )
            case "auc_cap":
                return f"the AUC is at most {self.threshold:.4g}"
            case "ratio_cap":
                return (
                    f"the larger class is at most {self.threshold:.4g} times the "
                    f"smaller (prevalence of the minority class at least "
                    f"{1 / (1 + self.threshold):.0%})"
                )


def check_score(
    *,
    kind: CheckKind,
    auc_exponent: float,
    ratio_exponent: float,
    n_min: NDArray,
    auc: NDArray,
    ratio: NDArray,
) -> NDArray:
    """Statistic of a check, transformed so that every check admits large values.

    Args:
        kind: Kind of the check.
        auc_exponent: Exponent ``a`` of the corner kinds.
        ratio_exponent: Exponent ``b`` of ``imbalance_corner``.
        n_min: Smaller class counts.
        auc: AUCs.
        ratio: Class ratios ``n_max / n_min``.

    Returns:
        The log statistic for the lower bounds, its negative for the ratio
        cap, and the negative AUC for ``auc_cap``.
    """
    match kind:
        case "floor":
            return np.log(n_min)
        case "corner":
            return np.log(n_min) + auc_exponent * np.log1p(-auc)
        case "imbalance_corner":
            return (
                np.log(n_min)
                + auc_exponent * np.log1p(-auc)
                - ratio_exponent * np.log(ratio)
            )
        case "auc_cap":
            return -auc
        case "ratio_cap":
            return -np.log(ratio)


def check_from_bound(
    *, kind: CheckKind, bound: float, auc_exponent: float, ratio_exponent: float
) -> Check:
    """Build a check from a threshold on the scale of :func:`check_score`."""
    if kind == "auc_cap":
        threshold = -bound
    elif kind in LOWER_BOUNDS:
        threshold = float(np.exp(bound))
    else:
        threshold = float(np.exp(-bound))
    return Check(
        kind=kind,
        threshold=threshold,
        auc_exponent=auc_exponent,
        ratio_exponent=ratio_exponent,
    )


@dataclass(frozen=True)
class Rule:
    """Conjunction of checks: a case goes to the unfloored band if all pass.

    Attributes:
        checks: The checks.
    """

    checks: tuple[Check, ...]

    def admits(self, *, n_min: NDArray, auc: NDArray, ratio: NDArray) -> NDArray:
        """Whether the rule sends each case to the unfloored band."""
        admitted = np.ones(np.broadcast(n_min, auc, ratio).shape, dtype=bool)
        for check in self.checks:
            admitted &= check.admits(n_min=n_min, auc=auc, ratio=ratio)
        return admitted

    def describe(self) -> str:
        """The rule as a conjunction of inequalities."""
        return " and ".join(check.describe() for check in self.checks)


# Rule fitted before the rounds placed around the settings that bound it, kept
# as the reference that fitted rules are compared against
REFERENCE_RULE = Rule(
    (
        Check(kind="floor", threshold=250),
        Check(kind="ratio_cap", threshold=6),
        Check(kind="corner", threshold=3.7, auc_exponent=2.5),
    )
)

AUC_EXPONENTS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5)
RATIO_EXPONENTS = (1 / 3, 2 / 3, 1.0)
EXPONENTS: dict[CheckKind, list[tuple[float, float]]] = {
    "floor": [(0.0, 0.0)],
    "corner": [(a, 0.0) for a in AUC_EXPONENTS],
    "imbalance_corner": [(a, b) for a in AUC_EXPONENTS for b in RATIO_EXPONENTS],
    "auc_cap": [(0.0, 0.0)],
    "ratio_cap": [(0.0, 0.0)],
}
# The first kind of a combination in this order closes it: its threshold is
# solved exactly, the others' are searched
CLOSING_ORDER: tuple[CheckKind, ...] = (
    "imbalance_corner",
    "corner",
    "floor",
    "ratio_cap",
    "auc_cap",
)
# Kinds that never appear together because one generalises the other
REDUNDANT_PAIRS = (frozenset({"corner", "imbalance_corner"}),)
MAX_CHECKS = 4
# Candidate thresholds per searched check, at quantiles of its statistic over
# the unsafe points
THRESHOLD_CANDIDATES = 16
# A rule with fewer checks is recommended if it admits at most this much less
# of the grid than the best rule
PARSIMONY_TOLERANCE = 0.01
# Largest majority class over which unsafe points constrain the rule. The
# earlier studies, which compute the unfloored band and the hybrid on the same
# datasets, reach a majority class of 9,000. At their largest sizes within the
# class ratios the rule admits, both bands miss less often than the target,
# their areas agree to about 1%, and the hybrid's ELL floor costs far more
# compute. Beyond this size the surface rests on sparse draws near a 5% miss
# rate, so its exceedance probabilities mostly measure that sparsity; the rule
# is not constrained there and the report checks it against replicated cells
ROUTED_N_MAX = 10_000


def routed_unsafe(*, grid: pd.DataFrame, assurance: float) -> NDArray:
    """Grid points that are unsafe at an assurance level and constrain the rule.

    Args:
        grid: Output of :func:`worst_case_grid`.
        assurance: Required posterior probability that the miss rate is at
            most the target.

    Returns:
        Whether each grid point is unsafe and has a majority class of at most
        :data:`ROUTED_N_MAX`.
    """
    routed = grid["n_min"] * grid["ratio"] <= ROUTED_N_MAX
    return ((grid["exceedance"] > 1 - assurance) & routed).to_numpy()


@dataclass(frozen=True, kw_only=True)
class RuleFit:
    """A rule found by :func:`search_rules` and its share of the grid.

    Attributes:
        rule: The rule, with only its active checks.
        admitted_share: Share of the runnable grid the rule admits.
    """

    rule: Rule
    admitted_share: float


def solve_combination(
    *,
    kinds: tuple[CheckKind, ...],
    exponents: tuple[tuple[float, float], ...],
    grid: pd.DataFrame,
    unsafe: NDArray,
) -> RuleFit | None:
    """Best rule with the given kinds and exponents.

    The last kind closes the rule. Every other kind is searched over
    :data:`THRESHOLD_CANDIDATES` thresholds, including none; for each setting
    the closing threshold is the smallest that excludes every unsafe point
    the other checks let through. Each active threshold of the winner is then
    moved onto the nearest admitted grid point, so that the rule claims
    nothing between the last admitted and the first excluded point.

    Args:
        kinds: Kinds of the checks, the closing kind last.
        exponents: ``(a, b)`` of each kind.
        grid: Output of :func:`worst_case_grid`.
        unsafe: Whether each grid point is unsafe.

    Returns:
        The rule admitting the largest share, or None if no unsafe point
        exists to bound.
    """
    if not unsafe.any():
        return None
    n_min, auc, ratio = (grid[c].to_numpy(float) for c in ("n_min", "auc", "ratio"))
    scores = [
        check_score(
            kind=kind,
            auc_exponent=a,
            ratio_exponent=b,
            n_min=n_min,
            auc=auc,
            ratio=ratio,
        )
        for kind, (a, b) in zip(kinds, exponents, strict=True)
    ]
    closing = scores[-1]
    options = []
    for score in scores[:-1]:
        bounds = np.unique(
            np.quantile(score[unsafe], np.linspace(0, 1, THRESHOLD_CANDIDATES - 1))
        )
        # A bound just above an unsafe value excludes that point
        options.append([-np.inf, *np.nextafter(bounds, np.inf)])
    # Every setting admits a share of at least zero, so the first replaces this
    best: tuple[float, list[float]] = (-1.0, [])
    for chosen in product(*options):
        passing = np.ones(len(grid), dtype=bool)
        for score, bound in zip(scores[:-1], chosen, strict=True):
            passing &= score >= bound
        leaking = passing & unsafe
        closing_bound = (
            float(np.nextafter(closing[leaking].max(), np.inf))
            if leaking.any()
            else -np.inf
        )
        share = float(np.mean(passing & (closing >= closing_bound)))
        if share > best[0]:
            best = (share, [*chosen, closing_bound])
    share, bounds = best
    admitted = np.ones(len(grid), dtype=bool)
    for score, bound in zip(scores, bounds, strict=True):
        admitted &= score >= bound
    if admitted.any():
        bounds = [
            float(score[admitted].min()) if np.isfinite(bound) else bound
            for score, bound in zip(scores, bounds, strict=True)
        ]
    checks = tuple(
        check_from_bound(kind=kind, bound=bound, auc_exponent=a, ratio_exponent=b)
        for kind, (a, b), bound in zip(kinds, exponents, bounds, strict=True)
        if np.isfinite(bound)
    )
    return RuleFit(rule=Rule(checks), admitted_share=share)


def search_rules(*, grid: pd.DataFrame, unsafe: NDArray) -> pd.DataFrame:
    """Find the best rule for every combination of check kinds.

    Args:
        grid: Output of :func:`worst_case_grid`.
        unsafe: Whether each grid point is unsafe.

    Returns:
        One row per combination of kinds with the best rule found
        (``fit``), its number of active checks (``checks``) and its admitted
        share, sorted by share.
    """
    rows = []
    for size in range(1, MAX_CHECKS + 1):
        for kinds in combinations(CLOSING_ORDER, size):
            if any(pair <= set(kinds) for pair in REDUNDANT_PAIRS):
                continue
            # combinations preserves CLOSING_ORDER, so the closing kind is first
            ordered = (*kinds[1:], kinds[0])
            best: RuleFit | None = None
            for exponents in product(*(EXPONENTS[k] for k in ordered)):
                fit = solve_combination(
                    kinds=ordered, exponents=exponents, grid=grid, unsafe=unsafe
                )
                if fit is not None and (
                    best is None or fit.admitted_share > best.admitted_share
                ):
                    best = fit
            if best is not None:
                rows.append(
                    {
                        "kinds": " + ".join(kinds),
                        "fit": best,
                        "checks": len(best.rule.checks),
                        "admitted_share": best.admitted_share,
                    }
                )
    return pd.DataFrame(rows).sort_values("admitted_share", ascending=False)


def round_strict(check: Check) -> Check:
    """Round a check's threshold to a round number on its strict side.

    Lower bounds round up and caps round down, to two significant figures; an
    AUC cap rounds down to a multiple of 0.005.

    Args:
        check: The check.

    Returns:
        The check with the rounded threshold; it admits a subset of what the
        original admits.
    """
    t = check.threshold
    # The tolerance keeps a threshold that is already round from moving
    if check.kind == "auc_cap":
        rounded = np.floor(t / 0.005 + 1e-9) * 0.005
    elif check.kind in LOWER_BOUNDS:
        scale = 10.0 ** (np.floor(np.log10(t)) - 1)
        rounded = np.ceil(t / scale - 1e-9) * scale
    else:
        scale = 10.0 ** (np.floor(np.log10(t)) - 1)
        rounded = np.floor(t / scale + 1e-9) * scale
    return Check(
        kind=check.kind,
        threshold=float(rounded),
        auc_exponent=check.auc_exponent,
        ratio_exponent=check.ratio_exponent,
    )


def recommend(*, search: pd.DataFrame) -> pd.DataFrame:
    """The best rule for each number of checks, and the recommended one.

    Args:
        search: Output of :func:`search_rules`.

    Returns:
        One row per number of checks with the best rule, its share before
        and after rounding, and whether it is recommended: the fewest checks
        within :data:`PARSIMONY_TOLERANCE` of the best share.
    """
    ladder = search.loc[search.groupby("checks")["admitted_share"].idxmax()]
    ladder = ladder.sort_values("checks").reset_index(drop=True)
    best = ladder["admitted_share"].max()
    eligible = ladder["admitted_share"] >= best - PARSIMONY_TOLERANCE
    ladder["recommended"] = False
    ladder.loc[eligible.idxmax(), "recommended"] = True
    ladder["rounded"] = [
        Rule(tuple(round_strict(c) for c in fit.rule.checks)) for fit in ladder["fit"]
    ]
    return ladder


def rule_summary(
    *, rule: Rule, grid: pd.DataFrame, unsafe: NDArray
) -> dict[str, float]:
    """Score a rule against the worst-case grid.

    Args:
        rule: The rule.
        grid: Output of :func:`worst_case_grid`.
        unsafe: Whether each grid point is unsafe.

    Returns:
        The share of the runnable grid admitted, the number of unsafe points
        admitted, the largest worst-case posterior-mean miss probability,
        exceedance probability and expected excess among the admitted points
        with a majority class of at most :data:`ROUTED_N_MAX`, and the largest
        posterior-mean miss probability among the admitted points beyond it.
    """
    admitted = rule.admits(n_min=grid["n_min"], auc=grid["auc"], ratio=grid["ratio"])
    routed = (grid["n_min"] * grid["ratio"] <= ROUTED_N_MAX).to_numpy()
    inside = grid[admitted & routed]
    beyond = grid[admitted & ~routed]
    return {
        "admitted_share": float(admitted.mean()),
        "unsafe_admitted": int((admitted & unsafe).sum()),
        "max_p_mean": float(inside["p_mean"].max()) if len(inside) else np.nan,
        "max_exceedance": float(inside["exceedance"].max()) if len(inside) else np.nan,
        "max_excess": float(inside["excess"].max()) if len(inside) else np.nan,
        "max_p_mean_beyond": float(beyond["p_mean"].max()) if len(beyond) else np.nan,
    }


def attribute_checks(
    *, rule: Rule, grid: pd.DataFrame, unsafe: NDArray
) -> pd.DataFrame:
    """Describe the unsafe slice each check of a rule is responsible for.

    A check's slice is the unsafe points that it alone excludes; points that
    several checks exclude are reported together.

    Args:
        rule: The rule.
        grid: Output of :func:`worst_case_grid`.
        unsafe: Whether each grid point is unsafe.

    Returns:
        One row per check, plus one for the shared points, with the slice's
        size, its ranges of ``n_min``, AUC and class ratio, the family and
        minority class that most often attain its worst case, and the median
        worst shape within that family.
    """
    n_min, auc, ratio = grid["n_min"], grid["auc"], grid["ratio"]
    failing = np.column_stack(
        [~c.admits(n_min=n_min, auc=auc, ratio=ratio) for c in rule.checks]
    )
    count = failing.sum(axis=1)
    slices = {
        check.describe(): unsafe & failing[:, k] & (count == 1)
        for k, check in enumerate(rule.checks)
    }
    slices["several checks"] = unsafe & (count > 1)
    rows = {}
    for label, members in slices.items():
        part = grid[members]
        if part.empty:
            rows[label] = {"unsafe_points": 0}
            continue
        family = part["worst_family"].mode().iloc[0]
        shape = part.loc[part["worst_family"] == family, "worst_shape"].median()
        rows[label] = {
            "unsafe_points": len(part),
            "n_min": f"{part['n_min'].min():,.0f}-{part['n_min'].max():,.0f}",
            "auc": f"{part['auc'].min():.3f}-{part['auc'].max():.3f}",
            "ratio": f"{part['ratio'].min():.3g}-{part['ratio'].max():.3g}",
            "worst_family": FAMILIES[family].title,
            "worst_minority": part["worst_minority"].mode().iloc[0],
            "worst_shape": f"{FAMILIES[family].label} {shape:.2g}",
        }
    return pd.DataFrame(rows).T


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
        x = surface_coordinates(frame=sub, family=FAMILIES[name])
        y = sub["miss"].to_numpy(float)
        fold = rng.integers(0, folds, len(sub))
        predicted = np.empty(len(sub))
        for k in range(folds):
            held = fold == k
            fitted = fit_surface(
                x=x[~held],
                y=y[~held],
                seed=seed,
                hyperparameters=(surface.lengthscales, surface.amplitude),
                linear_mean=surface.linear_mean,
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


def wilson_interval(*, missed: NDArray, draws: NDArray) -> tuple[NDArray, NDArray]:
    """95% Wilson interval of a binomial proportion.

    Args:
        missed: Successes.
        draws: Trials.

    Returns:
        Lower and upper ends.
    """
    missed, draws = np.asarray(missed, float), np.asarray(draws, float)
    z = norm.ppf(0.975)
    rate = missed / draws
    centre = (rate + z**2 / (2 * draws)) / (1 + z**2 / draws)
    half = (
        z
        * np.sqrt(rate * (1 - rate) / draws + z**2 / (4 * draws**2))
        / (1 + z**2 / draws)
    )
    return centre - half, centre + half


RATIO_BANDS = (1.0, 7 / 3, 6.0, RATIO_MAX)


def admitted_miss_rates(*, draws: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    """Observed miss rate of the draws a rule admits, by family, ratio and shape.

    This uses no fitted surface: it is the direct evidence on whether the
    admitted region covers.

    Args:
        draws: Draws returned by :func:`in_design_space`.
        rule: The rule.

    Returns:
        Draw count, observed miss rate and the upper end of its 95% Wilson
        interval for each family, class-ratio band and third of the shape
        range.
    """
    n_pos, n_neg = draws["n_pos"].to_numpy(float), draws["n_neg"].to_numpy(float)
    n_min, n_max = np.minimum(n_pos, n_neg), np.maximum(n_pos, n_neg)
    admitted = draws[
        rule.admits(n_min=n_min, auc=draws["auc"].to_numpy(float), ratio=n_max / n_min)
    ]
    ratio = np.maximum(admitted["n_pos"], admitted["n_neg"]) / np.minimum(
        admitted["n_pos"], admitted["n_neg"]
    )
    table = (
        admitted.assign(
            ratio_band=pd.cut(ratio, RATIO_BANDS, include_lowest=True),
            shape_third=np.minimum((admitted["u3"] * 3).astype(int), 2),
        )
        .groupby(["family", "ratio_band", "shape_third"], observed=True)
        .agg(draws=("miss", "size"), missed=("miss", "sum"))
    )
    _, upper = wilson_interval(missed=table["missed"], draws=table["draws"])
    return table.assign(miss_rate=table["missed"] / table["draws"], wilson_upper=upper)


def cell_check(
    *, cells: pd.DataFrame, surfaces: dict[str, Surface], target: float
) -> pd.DataFrame:
    """Compare the replicated cells' observed miss rates with the surface.

    Args:
        cells: Draws of cell rounds, with ``source`` and ``cell``.
        surfaces: Fitted surface by family name.
        target: Target miss rate.

    Returns:
        One row per cell with its settings, draw count, observed miss rate
        and 95% Wilson interval, the surface's posterior-mean miss
        probability, and its probability that the miss rate exceeds the
        target.
    """
    table = (
        cells.groupby(["source", "cell"])
        .agg(
            family=("family", "first"),
            shape=("shape", "first"),
            auc=("auc", "mean"),
            n_pos=("n_pos", "first"),
            n_neg=("n_neg", "first"),
            draws=("miss", "size"),
            missed=("miss", "sum"),
        )
        .reset_index()
    )
    lower, upper = wilson_interval(missed=table["missed"], draws=table["draws"])
    table["miss_rate"] = table["missed"] / table["draws"]
    table["wilson_lower"], table["wilson_upper"] = lower, upper
    table[["cell", "n_pos", "n_neg"]] = table[["cell", "n_pos", "n_neg"]].astype(int)
    table["p_surface"] = np.nan
    table["exceedance"] = np.nan
    for name, surface in surfaces.items():
        rows = table["family"] == name
        if rows.any():
            mean, sd = surface.predict(
                surface_coordinates(frame=table[rows], family=FAMILIES[name])
            )
            table.loc[rows, "p_surface"] = expit(mean)
            table.loc[rows, "exceedance"] = norm.sf(
                (logit(target) - mean) / np.maximum(sd, 1e-12)
            )
    return table.drop(columns="missed").set_index(["source", "cell"])


PLOT_RATIOS = (1.0, 2.0, 4.0, 9.0, 19.0)


def plot_surface(
    *, grid: pd.DataFrame, rules: dict[str, Rule], assurance: float, path: Path
) -> None:
    """Draw the worst-case miss probability, the assurance boundary and rules.

    Args:
        grid: Output of :func:`worst_case_grid`.
        rules: Rules to overlay, by label; each is drawn as the boundary of
            its admitted region.
        assurance: Posterior probability of safety the rules were fitted to.
        path: Destination image.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D

    n_axis = np.exp(np.linspace(*np.log(N_MIN_BOUNDS), GRID_N))
    fig, axes = plt.subplots(
        2,
        len(PLOT_RATIOS),
        figsize=(4.2 * len(PLOT_RATIOS), 8.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    styles = ("-", "--", "-.")
    meshes = {}
    for column, ratio in enumerate(PLOT_RATIOS):
        cell = grid[np.isclose(grid["ratio"], ratio)]
        for row, (value, label) in enumerate(
            (("p_mean", "posterior mean"), ("exceedance", "exceedance probability"))
        ):
            ax = axes[row, column]
            table = cell.pivot(index="auc", columns="n_min", values=value).reindex(
                columns=n_axis
            )
            z_auc = norm.ppf(table.index.to_numpy())
            x = table.columns.to_numpy()
            mesh = ax.pcolormesh(
                x,
                z_auc,
                table.to_numpy(),
                norm=(
                    TwoSlopeNorm(vmin=0.0, vcenter=ALPHA, vmax=0.30)
                    if value == "p_mean"
                    else TwoSlopeNorm(vmin=0.0, vcenter=1 - assurance, vmax=1.0)
                ),
                cmap="RdBu_r",
                shading="nearest",
            )
            exceed = cell.pivot(index="auc", columns="n_min", values="exceedance")
            exceed = exceed.reindex(columns=n_axis).to_numpy()
            ax.contour(
                x,
                z_auc,
                np.nan_to_num(exceed, nan=1.0),
                levels=[1 - assurance],
                colors="black",
                linewidths=1.2,
            )
            mesh_n, mesh_a = np.meshgrid(x, table.index.to_numpy())
            for rule, style in zip(rules.values(), styles, strict=False):
                admitted = rule.admits(n_min=mesh_n, auc=mesh_a, ratio=ratio)
                ax.contour(
                    x,
                    z_auc,
                    admitted.astype(float),
                    levels=[0.5],
                    colors="tab:green",
                    linestyles=style,
                    linewidths=2,
                )
            ax.axvline(N_CLASS_MAX / ratio, color="grey", linewidth=0.8)
            ax.axvline(
                ROUTED_N_MAX / ratio, color="black", linestyle=":", linewidth=1.2
            )
            ax.set_xscale("log")
            ax.set_xlim(*N_MIN_BOUNDS)
            ticks = [0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
            ax.set_yticks(norm.ppf(ticks), [f"{t:g}" for t in ticks])
            if row == 0:
                ax.set_title(
                    f"class ratio {ratio:.3g} : 1 (prevalence {1 / (1 + ratio):.0%})"
                )
            else:
                ax.set_xlabel("smaller class count")
            if column == 0:
                ax.set_ylabel(f"AUC\n({label})")
            meshes[row] = (mesh, label)
    for row, (mesh, label) in meshes.items():
        fig.colorbar(mesh, ax=axes[row], label=label)
    handles = [
        Line2D([], [], color="black", label=f"{assurance:.0%} assurance boundary"),
        Line2D(
            [],
            [],
            color="black",
            linestyle=":",
            label=f"larger class {ROUTED_N_MAX:,}: limit of calibration",
        ),
        *(
            Line2D([], [], color="tab:green", linestyle=style, label=name)
            for name, style in zip(rules, styles, strict=False)
        ),
    ]
    axes[0, 0].legend(handles=handles, loc="lower left", fontsize=8)
    fig.suptitle(
        "Unfloored band: worst case over shape, family and minority class "
        "(top: posterior-mean miss probability; bottom: posterior probability "
        f"of exceeding {ALPHA:g}); rules admit inside the green boundary"
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)


ASSURANCE_LEVELS = (0.8, 0.9, 0.95)
# Studies whose trial-level results seed the surface; each sits on a few fixed
# sample sizes at one prevalence
SOURCE_DESCRIPTIONS = {
    "nov11": "Main simulation suite, 50% prevalence",
    "nov11_p30": "Main simulation suite, 30% prevalence",
    "nov11_p10": "Main simulation suite, 10% prevalence",
    "shape_auc_study": "Shape x AUC study, 50% prevalence",
    "shape_auc_study_p10": "Shape x AUC study, 10% prevalence",
}
CRITERION_DESCRIPTIONS = {
    "fill": "spread over the design space",
    "contour": "near the 5% contour",
    "value": "where the surface was least certain near 5%",
    "cell": "replicated fixed cells",
    "target": "around grid points that fail the assurance test only through "
    "uncertainty and bound the rule",
}
# Smaller-class sizes and class ratios of the report's table of the largest
# AUC the rule routes
SUMMARY_N_MIN = (100, 250, 500, 1_000, 3_000, 10_000)
SUMMARY_RATIOS = (1.0, 2.0, 4.0, 6.0, 9.0, 19.0)
# Class sizes (positives, negatives) at which the report times both bands
TIMING_SIZES = ((1_000, 1_000), (1_000, 9_000), (10_000, 10_000))
# Smallest group of admitted draws whose miss rate the report quotes
MIN_GROUP_DRAWS = 200


def percent(value: float, *, digits: int = 1) -> str:
    """Format a proportion as a percentage, or a dash if it is missing."""
    return "—" if pd.isna(value) else f"{value:.{digits}%}"


def describe_sources(*, draws: pd.DataFrame, rounds: pd.DataFrame) -> pd.DataFrame:
    """Summarise where the draws come from.

    Args:
        draws: Draws returned by :func:`in_design_space`.
        rounds: Output of :func:`load_rounds`, which records how each of this
            study's draws was placed.

    Returns:
        One row per source with a description, the draw count, the range of
        the larger class and the observed miss rate.
    """
    rows = {}
    for source, part in draws.groupby("source", sort=False):
        if source in SOURCE_DESCRIPTIONS:
            description = f"{SOURCE_DESCRIPTIONS[source]}, fixed sample sizes"
        else:
            placed = rounds.loc[rounds["source"] == source, "criterion"]
            description = "This study, " + ", ".join(
                f"{share:.0%} {CRITERION_DESCRIPTIONS[criterion]}"
                for criterion, share in placed.value_counts(normalize=True).items()
            )
        n_max = np.maximum(part["n_pos"], part["n_neg"]).astype(int)
        rows[source] = {
            "description": description,
            "draws": f"{len(part):,}",
            "larger class": f"{n_max.min():,}-{n_max.max():,}",
            "miss rate": percent(part["miss"].mean()),
        }
    table = pd.DataFrame(rows).T
    table.index.name = "source"
    return table


def paired_floor_effect(*, directories: list[Path]) -> pd.DataFrame:
    """Compare the unfloored band with the hybrid on the same datasets.

    The earlier studies computed both bands on every dataset; this reads their
    trial-level results for the families in :data:`FAMILIES` at the study's
    alpha.

    Args:
        directories: Result directories of the earlier studies.

    Returns:
        One row per pair of class sizes with the number of datasets, the mean
        ratio of the hybrid's area to the unfloored band's, and each band's
        miss rate in the family where that rate is higher.
    """
    parts = []
    for directory in directories:
        for path in sorted(directory.glob("*_individual.feather")):
            family = next(
                (name for name in FAMILIES if path.name.startswith(name)), None
            )
            if family is None:
                continue
            frame = pd.read_feather(
                path,
                columns=[
                    "method",
                    "alpha",
                    "true_auc",
                    "n_pos",
                    "n_neg",
                    "covers_entirely",
                    "band_area",
                ],
            )
            frame = frame[np.isclose(frame["alpha"], ALPHA)]
            plain = frame[frame["method"] == "fiducial"].reset_index(drop=True)
            hybrid = frame[frame["method"] == "fiducial_hybrid"].reset_index(drop=True)
            if hybrid.empty:
                continue
            # The studies write every method of a dataset in the same order, so
            # the two bands pair by position
            if len(plain) != len(hybrid) or not np.allclose(
                plain["true_auc"], hybrid["true_auc"]
            ):
                raise ValueError(f"The bands in {path} do not pair by position")
            parts.append(
                pd.DataFrame(
                    {
                        "family": family,
                        "n_min": np.minimum(plain["n_pos"], plain["n_neg"]),
                        "n_max": np.maximum(plain["n_pos"], plain["n_neg"]),
                        "miss_unfloored": ~plain["covers_entirely"].to_numpy(bool),
                        "miss_hybrid": ~hybrid["covers_entirely"].to_numpy(bool),
                        "area_ratio": hybrid["band_area"].to_numpy(float)
                        / plain["band_area"].to_numpy(float),
                    }
                )
            )
    if not parts:
        return pd.DataFrame(
            columns=pd.Index(
                ["datasets", "area_ratio", "miss_unfloored", "miss_hybrid"]
            ),
            index=pd.MultiIndex.from_tuples([], names=["n_min", "n_max"]),
        )
    pairs = pd.concat(parts, ignore_index=True)
    sizes = ["n_min", "n_max"]
    worst = (
        pairs.groupby([*sizes, "family"])[["miss_unfloored", "miss_hybrid"]]
        .mean()
        .groupby(level=sizes)
        .max()
    )
    pooled = pairs.groupby(sizes).agg(
        datasets=("area_ratio", "size"), area_ratio=("area_ratio", "mean")
    )
    table = pooled.join(worst)
    return table.iloc[np.argsort(table.index.get_level_values("n_max"), kind="stable")]


def time_bands(*, sizes: tuple[tuple[int, int], ...]) -> pd.DataFrame:
    """Time the unfloored band and the hybrid on one dataset per class size.

    Every size is new to the process, so the hybrid pays for its ELL
    calibration as a user's first call at that size does.

    Args:
        sizes: Class sizes as (positives, negatives).

    Returns:
        One row per size with both bands' wall times in seconds.
    """
    from studroc_paper.methods.fiducial_band_rs import fiducial_band_rs

    rng = np.random.default_rng(0)
    rows = {}
    for n_pos, n_neg in sizes:
        y_true = np.r_[np.ones(n_pos), np.zeros(n_neg)].astype(np.float32)
        y_score = np.r_[
            rng.normal(1.0, 1.0, n_pos), rng.normal(0.0, 1.0, n_neg)
        ].astype(np.float32)
        seconds = []
        for floor in (False, True):
            start = time.perf_counter()
            fiducial_band_rs(
                y_true=y_true, y_score=y_score, alpha=ALPHA, m3_floor=floor
            )
            seconds.append(time.perf_counter() - start)
        rows[f"{n_pos:,} / {n_neg:,}"] = {
            "unfloored (s)": seconds[0],
            "hybrid (s)": seconds[1],
        }
    table = pd.DataFrame(rows).T
    table.index.name = "positives / negatives"
    return table


def largest_routed_auc(
    *, rule: Rule, n_mins: tuple[int, ...], ratios: tuple[float, ...]
) -> pd.DataFrame:
    """The largest AUC a rule sends to the unfloored band, by size and ratio.

    Args:
        rule: The rule.
        n_mins: Smaller-class sizes.
        ratios: Class ratios ``n_max / n_min``.

    Returns:
        The largest admitted AUC within :data:`AUC_BOUNDS` for each smaller
        class (rows) and class ratio (columns), or ``"hybrid"`` where the rule
        admits no AUC.
    """
    auc = np.linspace(*AUC_BOUNDS, 4401)
    table = {}
    for ratio in ratios:
        column = {}
        for n_min in n_mins:
            admitted = rule.admits(
                n_min=np.full_like(auc, n_min), auc=auc, ratio=np.full_like(auc, ratio)
            )
            column[f"{n_min:,}"] = (
                f"{auc[admitted].max():.3f}" if admitted.any() else "hybrid"
            )
        table[f"{ratio:g} : 1"] = column
    frame = pd.DataFrame(table)
    frame.index.name = "smaller class"
    return frame


def summary_columns(*, summary: dict[str, float]) -> dict[str, str]:
    """Format :func:`rule_summary` for a reader.

    Args:
        summary: Output of :func:`rule_summary`.

    Returns:
        The summary with readable names and formatted values.
    """
    return {
        "grid share": percent(summary["admitted_share"]),
        "unsafe settings admitted": f"{summary['unsafe_admitted']:,}",
        f"worst mean miss rate, n_max <= {ROUTED_N_MAX:,}": percent(
            summary["max_p_mean"], digits=2
        ),
        f"worst P(miss rate > {ALPHA:.0%}), n_max <= {ROUTED_N_MAX:,}": percent(
            summary["max_exceedance"]
        ),
        f"worst mean miss rate, n_max > {ROUTED_N_MAX:,}": percent(
            summary["max_p_mean_beyond"], digits=2
        ),
    }


def shape_third_label(*, family: str, third: int) -> str:
    """The range of a family's shape parameter in one third of its log range.

    Args:
        family: Family name.
        third: 0, 1 or 2, from the lowest third of the range up.

    Returns:
        The shape parameter's name and range.
    """
    low, high = np.log(FAMILIES[family].shape_bounds)
    edges = np.exp(low + (high - low) * np.array([third, third + 1]) / 3)
    return f"{FAMILIES[family].label} {edges[0]:.2g}-{edges[1]:.2g}"


def admits_cells(*, rule: Rule, cells: pd.DataFrame) -> NDArray:
    """Whether a rule admits each replicated cell.

    Args:
        rule: The rule.
        cells: Output of :func:`cell_check`.

    Returns:
        One flag per cell.
    """
    n_min = np.minimum(cells["n_pos"], cells["n_neg"]).to_numpy(float)
    n_max = np.maximum(cells["n_pos"], cells["n_neg"]).to_numpy(float)
    return rule.admits(
        n_min=n_min, auc=cells["auc"].to_numpy(float), ratio=n_max / n_min
    )


# Confidence of safety at which the report counts a setting the rule sends to
# the hybrid as left on the table, below the assurance the rule is held to
SAFE_CONFIDENCE = 0.8
# Largest regions of such settings the report describes, per rule
TOP_REGIONS = 6
# Bands of the smaller class and the class ratio that split a connected
# region, which otherwise follows a rule's boundary across the whole grid
REGION_N_EDGES = (300.0, 1_000.0, 3_000.0, 10_000.0)
REGION_RATIO_EDGES = (2.5, 6.5)


def grid_admits(*, rule: Rule, grid: pd.DataFrame) -> NDArray:
    """Whether a rule admits each grid point."""
    return rule.admits(
        n_min=grid["n_min"].to_numpy(float),
        auc=grid["auc"].to_numpy(float),
        ratio=grid["ratio"].to_numpy(float),
    )


def hybrid_breakdown(
    *, rules: dict[str, Rule], grid: pd.DataFrame, assurance: float
) -> pd.DataFrame:
    """How safe the settings each rule sends to the hybrid are.

    Args:
        rules: Rules by label.
        grid: Output of :func:`worst_case_grid`.
        assurance: Posterior probability of safety the rules are held to.

    Returns:
        One row per rule: the share of the grid sent to the hybrid, and how
        that part splits by the best estimate and by the posterior
        probability of safety, as shares of the part.
    """
    safe_at = {
        level: (grid["exceedance"] <= 1 - level).to_numpy()
        for level in (SAFE_CONFIDENCE, assurance)
    }
    below = (grid["p_mean"] <= ALPHA).to_numpy()
    beyond = (grid["n_min"] * grid["ratio"] > ROUTED_N_MAX).to_numpy()
    rows = {}
    for label, rule in rules.items():
        hybrid = ~grid_admits(rule=rule, grid=grid)
        loose = hybrid & safe_at[SAFE_CONFIDENCE]
        rows[label] = {
            "grid share sent to the hybrid": percent(hybrid.mean()),
            f"best estimate above {ALPHA:.0%}": percent(
                (hybrid & ~below).sum() / hybrid.sum()
            ),
            f"best estimate at most {ALPHA:.0%}": percent(
                (hybrid & below).sum() / hybrid.sum()
            ),
            f"safe with {SAFE_CONFIDENCE:.0%} confidence": percent(
                loose.sum() / hybrid.sum()
            ),
            f"safe with {assurance:.0%} confidence": percent(
                (hybrid & safe_at[assurance]).sum() / hybrid.sum()
            ),
            f"of the {SAFE_CONFIDENCE:.0%}-safe, larger class above "
            f"{ROUTED_N_MAX:,}": percent((loose & beyond).sum() / max(loose.sum(), 1)),
        }
    table = pd.DataFrame(rows).T
    table.index.name = "rule"
    return table


def grid_lattice(grid: pd.DataFrame) -> NDArray:
    """Lattice indices (ratio, AUC, n_min) of each grid point.

    Args:
        grid: Output of :func:`worst_case_grid`.

    Returns:
        (rows, 3) integer indices into the axes of :func:`worst_case_grid`.
    """
    log_n = np.log(N_MIN_BOUNDS)
    probit = norm.ppf(AUC_BOUNDS)
    return np.column_stack(
        [
            np.abs(grid["ratio"].to_numpy()[:, None] - np.array(GRID_RATIOS)).argmin(1),
            np.rint(
                (norm.ppf(grid["auc"].to_numpy()) - probit[0])
                / (probit[1] - probit[0])
                * (GRID_AUC - 1)
            ).astype(int),
            np.rint(
                (np.log(grid["n_min"].to_numpy()) - log_n[0])
                / (log_n[1] - log_n[0])
                * (GRID_N - 1)
            ).astype(int),
        ]
    )


def draws_to_assure(*, grid: pd.DataFrame, assurance: float) -> NDArray:
    """Rough effective draws still needed at each grid point to reach assurance.

    The worst-case posterior is treated as a binomial logit at the
    best-estimate miss rate: the posterior SD implied by the exceedance gives
    the effective draws behind it, and the SD that would put the target at the
    assurance quantile gives the draws needed. Draws near a point inform its
    neighbours within the surface's length scales, so the figure is a guide to
    the local information required, not a design.

    Args:
        grid: Output of :func:`worst_case_grid`.
        assurance: Posterior probability of safety to reach.

    Returns:
        Extra effective draws per point; zero where the point already meets
        the assurance and infinite where its best estimate is above the target.
    """
    p = grid["p_mean"].to_numpy(float)
    exceed = grid["exceedance"].to_numpy(float)
    gap = logit(ALPHA) - logit(np.clip(p, 1e-6, 1 - 1e-6))
    information = 1 / (p * (1 - p) * np.maximum(gap, 1e-12) ** 2)
    now = norm.isf(np.clip(exceed, 1e-12, 0.5)) ** 2 * information
    need = norm.ppf(assurance) ** 2 * information
    extra = np.where(gap > 0, np.maximum(need - now, 0.0), np.inf)
    return np.where(exceed <= 1 - assurance, 0.0, extra)


def point_cost(grid: pd.DataFrame) -> NDArray:
    """Modelled seconds per draw at each grid point's worst-case class sizes."""
    larger = grid["n_min"] * grid["ratio"]
    negative_minority = grid["worst_minority"] == "negative"
    return cost_seconds(
        pd.DataFrame(
            {
                "n_pos": np.where(negative_minority, larger, grid["n_min"]),
                "n_neg": np.where(negative_minority, grid["n_min"], larger),
            }
        )
    )


def span(values: pd.Series, *, fmt: str) -> str:
    """A range of values as text, or the single value if they agree."""
    low, high = values.min(), values.max()
    return (
        f"{low:{fmt}}"
        if f"{low:{fmt}}" == f"{high:{fmt}}"
        else f"{low:{fmt}}-{high:{fmt}}"
    )


def safe_hybrid_regions(
    *, rule: Rule, grid: pd.DataFrame, assurance: float
) -> pd.DataFrame:
    """Connected regions the rule sends to the hybrid that are probably safe.

    A grid point belongs to a region if the rule sends it to the hybrid and
    the posterior probability that its worst-case miss rate is at most the
    target is at least :data:`SAFE_CONFIDENCE`. Regions are the connected
    components of these points on the grid's lattice, neighbours sharing a
    face (one step in the smaller class, the AUC or the class ratio), split
    by the bands :data:`REGION_N_EDGES` and :data:`REGION_RATIO_EDGES` so
    that each describes one stretch of the rule's boundary.

    Args:
        rule: The rule.
        grid: Output of :func:`worst_case_grid`.
        assurance: Posterior probability of safety the rule is held to.

    Returns:
        One row per region, largest first: its size, where it lies, how
        close to the target it is, the checks that send it to the hybrid,
        and the draws its hardest setting still needs to reach
        ``assurance``.
    """
    mask = (
        ~grid_admits(rule=rule, grid=grid)
        & (grid["exceedance"] <= 1 - SAFE_CONFIDENCE).to_numpy()
    )
    lattice = grid_lattice(grid)
    volume = np.zeros((len(GRID_RATIOS), GRID_AUC, GRID_N), dtype=bool)
    volume[tuple(lattice[mask].T)] = True
    labels, _ = ndimage.label(volume)
    component = np.where(mask, labels[tuple(lattice.T)], 0)
    band = np.digitize(grid["n_min"], REGION_N_EDGES) * (
        len(REGION_RATIO_EDGES) + 1
    ) + np.digitize(grid["ratio"], REGION_RATIO_EDGES)
    keys = component * (len(REGION_N_EDGES) + 1) * (len(REGION_RATIO_EDGES) + 1) + band
    _, inverse = np.unique(keys[mask], return_inverse=True)
    region = np.zeros(len(grid), dtype=int)
    region[mask] = inverse + 1
    count = int(region.max())
    extra = draws_to_assure(grid=grid, assurance=assurance)
    cost = point_cost(grid)
    args = {c: grid[c].to_numpy(float) for c in ("n_min", "auc", "ratio")}
    failing = {check.describe(): ~check.admits(**args) for check in rule.checks}
    rows = []
    for k in range(1, count + 1):
        members = region == k
        part = grid[members]
        hardest = int(np.argmax(np.where(members, extra, -1.0)))
        worst = part.groupby(["worst_family", "worst_minority"]).size().idxmax()
        shapes = part.loc[
            (part["worst_family"] == worst[0]) & (part["worst_minority"] == worst[1]),
            "worst_shape",
        ]
        rows.append(
            {
                "settings": int(members.sum()),
                "grid share": percent(members.mean()),
                "smaller class": span(part["n_min"], fmt=",.0f"),
                "larger class": span(part["n_min"] * part["ratio"], fmt=",.0f"),
                "AUC": span(part["auc"], fmt=".3f"),
                "class ratio": span(part["ratio"], fmt=".3g"),
                "worst family": FAMILIES[worst[0]].title,
                "minority class": worst[1],
                "typical worst shape": f"{shapes.median():.3g}",
                "best estimate": span(part["p_mean"], fmt=".2%"),
                f"safe at {assurance:.0%}": percent(
                    (part["exceedance"] <= 1 - assurance).mean()
                ),
                "sent to the hybrid by": ", ".join(
                    f"`{name}` ({fails[members].mean():.0%})"
                    for name, fails in failing.items()
                    if fails[members].mean() >= 0.5
                ),
                "draws at the hardest setting": f"{extra[hardest]:,.0f}",
                "CPU-hours there": f"{extra[hardest] * cost[hardest] / 3600:,.1f}",
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.sort_values("settings", ascending=False).head(TOP_REGIONS)
    table.index = pd.RangeIndex(1, len(table) + 1, name="region")
    return table


def best_extra_clause(
    *, rule: Rule, grid: pd.DataFrame, unsafe: NDArray
) -> tuple[Rule | None, NDArray]:
    """Best second conjunction to admit, as an alternative, beside a rule.

    The routing becomes ``rule or clause``. The clause is searched like a
    rule (:func:`search_rules`, :func:`recommend`, rounded on the strict
    side) over the grid points the rule sends to the hybrid, so it must
    exclude every unsafe point among them.

    Args:
        rule: The rule.
        grid: Output of :func:`worst_case_grid`.
        unsafe: Points the clause must exclude.

    Returns:
        The clause, or None if no clause admits anything, and the points it
        adds to the rule.
    """
    hybrid = ~grid_admits(rule=rule, grid=grid)
    sub = grid[hybrid].reset_index(drop=True)
    if not unsafe[hybrid].any():
        return None, np.zeros(len(grid), dtype=bool)
    ladder = recommend(search=search_rules(grid=sub, unsafe=unsafe[hybrid]))
    clause = ladder.loc[ladder["recommended"], "rounded"].iloc[0]
    added = hybrid & grid_admits(rule=clause, grid=grid)
    if not added.any():
        return None, added
    return clause, added


def clause_options(*, rule: Rule, grid: pd.DataFrame, assurance: float) -> pd.DataFrame:
    """Second clauses for a rule, now and if more draws tightened the surface.

    Args:
        rule: The rule.
        grid: Output of :func:`worst_case_grid`.
        assurance: Posterior probability of safety the rule is held to.

    Returns:
        Two rows: the best clause that every point it adds already passes at
        ``assurance``, and the best clause whose points pass at
        :data:`SAFE_CONFIDENCE`, which needs draws at those that do not yet
        pass at ``assurance``. Each gives the clause, the grid share it adds,
        the points needing draws, and the draws and modelled CPU-hours at the
        hardest of them.
    """
    extra = draws_to_assure(grid=grid, assurance=assurance)
    cost = point_cost(grid)
    rows = {}
    for scenario, level in (
        (f"now, at {assurance:.0%}", assurance),
        (
            f"after draws lift {SAFE_CONFIDENCE:.0%}-safe settings to {assurance:.0%}",
            SAFE_CONFIDENCE,
        ),
    ):
        unsafe = (grid["exceedance"] > 1 - level).to_numpy()
        clause, added = best_extra_clause(rule=rule, grid=grid, unsafe=unsafe)
        pending = added & (grid["exceedance"] > 1 - assurance).to_numpy()
        hardest = (
            int(np.argmax(np.where(pending, extra, -1.0))) if pending.any() else None
        )
        rows[scenario] = {
            "clause": clause.describe() if clause else "none",
            "grid share added": percent(added.mean()),
            "settings needing draws": f"{int(pending.sum()):,}",
            "draws at the hardest setting": (
                f"{extra[hardest]:,.0f}" if hardest is not None else "—"
            ),
            "CPU-hours there": (
                f"{extra[hardest] * cost[hardest] / 3600:,.1f}"
                if hardest is not None
                else "—"
            ),
        }
    table = pd.DataFrame(rows).T
    table.index.name = "when"
    return table


def auc_headroom(
    *,
    rules: dict[str, Rule],
    grid: pd.DataFrame,
    assurance: float,
    n_mins: tuple[int, ...],
    ratios: tuple[float, ...],
) -> pd.DataFrame:
    """Largest AUC each rule admits beside the largest that is safe.

    The safe limit at a smaller class and class ratio is the largest grid AUC
    up to which every grid AUC passes the test at ``assurance``, at the grid
    point nearest that smaller class.

    Args:
        rules: Rules by label.
        grid: Output of :func:`worst_case_grid`.
        assurance: Posterior probability of safety of the test.
        n_mins: Smaller-class sizes of the rows.
        ratios: Class ratios of the columns.

    Returns:
        One row per smaller class and one column per ratio, each cell giving
        the rules' limits and the safe limit, ``hybrid`` where a rule admits
        nothing and ``none`` where no AUC is safe.
    """
    limits = {
        label: largest_routed_auc(rule=rule, n_mins=n_mins, ratios=ratios)
        for label, rule in rules.items()
    }
    grid_n = np.unique(grid["n_min"])
    table = pd.DataFrame(
        index=pd.Index([f"{n:,.0f}" for n in n_mins], name="smaller class"),
        columns=pd.Index([f"{r:g} : 1" for r in ratios], name="class ratio"),
        dtype=object,
    )
    for i, n in enumerate(n_mins):
        nearest = grid_n[np.abs(np.log(grid_n / n)).argmin()]
        for j, ratio in enumerate(ratios):
            column = grid[
                np.isclose(grid["n_min"], nearest) & np.isclose(grid["ratio"], ratio)
            ].sort_values("auc")
            safe = (column["exceedance"] <= 1 - assurance).to_numpy()
            run = int(np.argmin(safe)) if not safe.all() else len(safe)
            safe_text = f"{column['auc'].iloc[run - 1]:.3f}" if run else "none"
            table.iloc[i, j] = " / ".join(
                [*(str(limits[label].iloc[i, j]) for label in rules), safe_text]
            ) + (" †" if n * ratio > ROUTED_N_MAX else "")
    return table


# Exponents (AUC, ratio) of the two corner checks of the bent-corner form: a
# steep check that binds at small samples and a flat one that binds at large
STEEP_CORNERS = tuple(
    (a, b) for a in (0.5, 0.75, 1.0, 1.25, 1.5) for b in (0.0, 1 / 3, 2 / 3)
)
FLAT_CORNERS = tuple((a, b) for a in (2.0, 2.5, 3.0, 4.0, 6.0) for b in (0.0, 1 / 3))


def bent_corner_rule(*, grid: pd.DataFrame, unsafe: NDArray) -> tuple[Rule, float]:
    """Best rule with two corner checks, optionally with a floor.

    The rule search allows one corner check, whose limit on ``1 - AUC`` is a
    power of the smaller class. Two corner checks of different exponents
    bound ``1 - AUC`` by the larger of two powers, a frontier that bends
    from steep at small samples to flat at large ones.

    Args:
        grid: Output of :func:`worst_case_grid`.
        unsafe: Whether each grid point is unsafe.

    Returns:
        The best such rule, rounded on the strict side, and its grid share.
    """
    best: tuple[float, Rule] | None = None
    for floor in ((), (("floor", (0.0, 0.0)),)):
        for steep, flat in product(STEEP_CORNERS, FLAT_CORNERS):
            parts = (*floor, ("imbalance_corner", steep), ("imbalance_corner", flat))
            fit = solve_combination(
                kinds=tuple(kind for kind, _ in parts),
                exponents=tuple(exponent for _, exponent in parts),
                grid=grid,
                unsafe=unsafe,
            )
            if fit is None:
                continue
            rounded = Rule(
                tuple(
                    Check(
                        kind="corner",
                        threshold=c.threshold,
                        auc_exponent=c.auc_exponent,
                    )
                    if c.kind == "imbalance_corner" and c.ratio_exponent == 0
                    else c
                    for c in (round_strict(check) for check in fit.rule.checks)
                )
            )
            share = float(grid_admits(rule=rounded, grid=grid).mean())
            if best is None or share > best[0]:
                best = (share, rounded)
    if best is None:
        raise ValueError("No unsafe point to bound")
    return best[1], best[0]


def build_report(
    *,
    draws: pd.DataFrame,
    rounds: pd.DataFrame,
    surfaces: dict[str, Surface],
    predictions: pd.DataFrame,
    grid: pd.DataFrame,
    paired: pd.DataFrame,
    timing: pd.DataFrame,
    assurance: float,
    folds: int,
) -> tuple[dict[str, Rule], str]:
    """Derive the rules and write the report for a reader new to the study.

    Args:
        draws: Draws returned by :func:`in_design_space`.
        rounds: Output of :func:`load_rounds`.
        surfaces: Fitted surface by family name.
        predictions: Output of :func:`cross_validate`.
        grid: Output of :func:`worst_case_grid`.
        paired: Output of :func:`paired_floor_effect`.
        timing: Output of :func:`time_bands`.
        assurance: Required posterior probability that the miss rate is at
            most the target.
        folds: Number of cross-validation folds behind ``predictions``.

    Returns:
        The recommended rule, the previous rule and, if one is within the
        parsimony tolerance with more checks, the alternative rule, by label;
        and the report as Markdown.
    """
    families = " and ".join(
        f"{f.title} ({f.label} from {f.shape_bounds[0]:g} to {f.shape_bounds[1]:g})"
        for f in FAMILIES.values()
    )
    unsafe = routed_unsafe(grid=grid, assurance=assurance)
    unsafe_anywhere = (grid["exceedance"] > 1 - assurance).to_numpy()
    routed = (grid["n_min"] * grid["ratio"] <= ROUTED_N_MAX).to_numpy()
    search = search_rules(grid=grid, unsafe=unsafe)
    ladder = recommend(search=search)
    recommended = ladder.loc[ladder["recommended"], "rounded"].iloc[0]
    rules = {"recommended": recommended, "previous": REFERENCE_RULE}
    # The rule with the most checks among those within the parsimony tolerance
    # trades simplicity for a different shape of the admitted region
    eligible = ladder[
        ladder["admitted_share"] >= ladder["admitted_share"].max() - PARSIMONY_TOLERANCE
    ]
    if eligible["checks"].max() > ladder.loc[ladder["recommended"], "checks"].iloc[0]:
        rules["alternative"] = eligible.loc[eligible["checks"].idxmax(), "rounded"]
    summaries = {
        label: rule_summary(rule=rule, grid=grid, unsafe=unsafe)
        for label, rule in rules.items()
    }
    admitted_grid = recommended.admits(
        n_min=grid["n_min"], auc=grid["auc"], ratio=grid["ratio"]
    )
    breakdown = hybrid_breakdown(rules=rules, grid=grid, assurance=assurance)
    candidates = {
        label: rules[label]
        for label in ("recommended", "alternative")
        if label in rules
    }
    regions = {
        label: safe_hybrid_regions(rule=rule, grid=grid, assurance=assurance)
        for label, rule in candidates.items()
    }
    clauses = {
        label: clause_options(rule=rule, grid=grid, assurance=assurance)
        for label, rule in candidates.items()
    }
    loose_column = f"safe with {SAFE_CONFIDENCE:.0%} confidence"
    headroom = auc_headroom(
        rules=candidates,
        grid=grid,
        assurance=assurance,
        n_mins=SUMMARY_N_MIN,
        ratios=SUMMARY_RATIOS,
    )
    bent, bent_share = bent_corner_rule(grid=grid, unsafe=unsafe)

    reliability = reliability_table(predictions=predictions)
    near = reliability[reliability["predicted"].between(0.02, 0.12)]
    largest_deviation = (near["predicted"] - near["observed"]).abs().max()
    around_target = reliability[reliability["predicted"].between(0.03, 0.08)]
    bias = (around_target["predicted"] - around_target["observed"]).mean()

    cells = rounds[rounds["cell"] >= 0] if "cell" in rounds else rounds.iloc[0:0]
    cell_table = (
        cell_check(cells=cells, surfaces=surfaces, target=ALPHA)
        if len(cells)
        else pd.DataFrame(
            columns=pd.Index(
                [
                    "n_pos",
                    "n_neg",
                    "auc",
                    "miss_rate",
                    "wilson_lower",
                    "wilson_upper",
                    "p_surface",
                ]
            )
        )
    )
    # Replicated cells the rule admits beyond the range it is calibrated on
    far = (
        np.maximum(cell_table["n_pos"], cell_table["n_neg"]).to_numpy() > ROUTED_N_MAX
    ) & admits_cells(rule=recommended, cells=cell_table)
    far_cells = cell_table[far]

    admitted = admitted_miss_rates(draws=draws, rule=recommended)
    lower, _ = wilson_interval(missed=admitted["missed"], draws=admitted["draws"])
    admitted = admitted.assign(wilson_lower=lower)
    admitted_draws = int(admitted["draws"].sum())
    admitted_rate = admitted["missed"].sum() / admitted_draws
    significant = admitted[admitted["wilson_lower"] > ALPHA]
    large_groups = admitted[admitted["draws"] >= MIN_GROUP_DRAWS]

    lines = [
        "# Routing rule for `auto`: when the unfloored fiducial band is safe",
        "",
        "## Summary",
        "",
        "**The decision.** The `auto` mode chooses between two ROC confidence "
        "bands using only what it can see: the two class counts and the AUC. "
        "The *unfloored fiducial band* is narrow and fast. The *hybrid* adds an "
        "exact ELL floor at the two ends of the curve, which protects against "
        "the unfloored band's known failure: when the ROC has a convex hook in "
        "a corner that few observations reach, the unfloored band misses the "
        "true curve more often than its nominal 5%. Where the unfloored band is "
        "safe it is preferred, because it is narrower at small samples, much "
        "faster at large ones, and closer to nominal (Section 1). This report "
        "finds where it is safe.",
        "",
        "**Recommended rule.** Use the unfloored band when all of the following "
        "hold, and the hybrid otherwise (`n_min` and `n_max` are the smaller "
        "and larger class counts):",
        "",
        *(f"- {check.explain()}." for check in recommended.checks),
        "",
        f"In one line: `{recommended.describe()}`.",
        "",
        f"**What safe means.** A *miss* is a dataset whose band fails to contain "
        f"the true ROC somewhere; at 95% confidence the target miss rate is "
        f"{ALPHA:.0%}. At every setting the rule admits with a larger class of "
        f"at most {ROUTED_N_MAX:,}, the fitted model gives at least "
        f"{assurance:.0%} probability that the miss rate is at most "
        f"{ALPHA:.0%}. This holds for the worst score distribution in the two "
        f"families known to break the unfloored band, {families}, with either "
        f"class as the minority. The largest best-estimate miss rate the rule "
        f"admits there is "
        f"{percent(summaries['recommended']['max_p_mean'], digits=2)}.",
        "",
        "**Where it routes.** The largest AUC sent to the unfloored band, by the "
        "size of the smaller class and the class ratio. Above it, and wherever "
        "the table says hybrid, the hybrid is used:",
        "",
        markdown_table(
            largest_routed_auc(
                rule=recommended, n_mins=SUMMARY_N_MIN, ratios=SUMMARY_RATIOS
            ),
            floatfmt=".3f",
        ),
        "",
        "**Evidence.**",
        "",
        f"- The model of the miss rate predicts held-out data well: between "
        f"predicted rates of 2% and 12%, predictions are within "
        f"{largest_deviation * 100:.1f} percentage points of observed rates "
        f"(Section 3).",
    ]
    if len(cell_table):
        within = int(
            (
                (cell_table["p_surface"] >= cell_table["wilson_lower"])
                & (cell_table["p_surface"] <= cell_table["wilson_upper"])
            ).sum()
        )
        lines.append(
            f"- At {len(cell_table)} fixed settings, each simulated up to 1,000 "
            f"times, {within} observed miss rates have 95% intervals that contain "
            "the model's prediction (Section 4)."
        )
    lines += [
        f"- Without any model: of the {admitted_draws:,} simulated datasets the "
        f"rule admits, {percent(admitted_rate, digits=2)} missed, and "
        + (
            f"no group by family, class ratio and shape is significantly above "
            f"{ALPHA:.0%}"
            if significant.empty
            else f"{len(significant)} groups by family, class ratio and shape are "
            f"significantly above {ALPHA:.0%}"
        )
        + " (Section 6).",
        f"- The previous rule, fitted before the rounds placed around the "
        f"settings that bound it (`{REFERENCE_RULE.describe()}`), admits "
        f"{summaries['previous']['unsafe_admitted']:,} "
        f"setting{'' if summaries['previous']['unsafe_admitted'] == 1 else 's'} "
        f"that fail{'s' if summaries['previous']['unsafe_admitted'] == 1 else ''} the "
        f"same test; the recommended rule admits none (Section 5.4).",
        "",
        "**Limits.**",
        "",
        "- Only the two families above are modelled. They are the known failure "
        "modes, but other score distributions are not covered.",
        f"- Above a larger class of {ROUTED_N_MAX:,} the simulated data are "
        f"sparse and the rule is not held to the {assurance:.0%} test; there "
        f"the floor costs far more compute and buys no width (Section 1). "
        + (
            f"The {len(far_cells)} replicated settings measured there that the "
            f"rule admits missed at "
            f"{percent(far_cells['miss_rate'].min())}-"
            f"{percent(far_cells['miss_rate'].max())}, "
            + (
                f"none significantly above {ALPHA:.0%}"
                if (far_cells["wilson_lower"] <= ALPHA).all()
                else f"{int((far_cells['wilson_lower'] > ALPHA).sum())} "
                f"significantly above {ALPHA:.0%}"
            )
            + "; the model's best estimate, extended past its data, reaches "
            if len(far_cells)
            else "The model's best estimate, extended past its data, reaches "
        )
        + f"{percent(summaries['recommended']['max_p_mean_beyond'], digits=2)} at "
        f"the edge of the design (Section 5.5).",
        "- The study evaluates the rule at the true AUC; at runtime the "
        "empirical AUC stands in for it.",
        f"- The rule sends "
        f"{breakdown.loc['recommended', 'grid share sent to the hybrid']} of the "
        f"grid to the hybrid; "
        f"{breakdown.loc['recommended', loose_column]} of that is safe with "
        f"{SAFE_CONFIDENCE:.0%} confidence, mostly in slivers along the rule's "
        f"boundary"
        + (
            f". The largest such region has a smaller class of "
            f"{regions['recommended'].iloc[0]['smaller class']}, AUC "
            f"{regions['recommended'].iloc[0]['AUC']} and class ratio "
            f"{regions['recommended'].iloc[0]['class ratio']}"
            if len(regions["recommended"])
            else ""
        )
        + f". A second clause beside the rule would route a further "
        f"{clauses['recommended'].iloc[0]['grid share added']} of the grid "
        f"without new draws, and a rule with two corner checks "
        f"{(bent_share - summaries['recommended']['admitted_share']) * 100:.1f} "
        f"percentage points (Section 5.6).",
        f"- Shares of the grid quoted below count points of an evaluation grid "
        f"(log class size, probit AUC, ten class ratios) without weights. They "
        f"compare candidate rules and do not estimate how often users are "
        f"routed. The recommended rule admits "
        f"{percent(summaries['recommended']['admitted_share'])} of the grid.",
        "",
        "## 1. What routing buys",
        "",
        "The earlier studies computed both bands on the same datasets, so they "
        "can be compared directly. The table pairs them for the two families of "
        "this study at the study's alpha; each miss rate is from the family in "
        "which it is higher.",
        "",
    ]
    paired_display = pd.DataFrame(
        {
            "class ratio": [f"{n_max / n_min:.3g}" for n_min, n_max in paired.index],
            "datasets": [f"{n:,}" for n in paired["datasets"]],
            "hybrid area / unfloored area": [f"{r:.2f}" for r in paired["area_ratio"]],
            "unfloored miss rate": [percent(v) for v in paired["miss_unfloored"]],
            "hybrid miss rate": [percent(v) for v in paired["miss_hybrid"]],
        },
        index=paired.index,
    )
    n_max_paired = paired.index.get_level_values("n_max")
    small = paired[n_max_paired <= 50]
    large = paired[n_max_paired >= 5_000]
    failing = paired[paired["miss_unfloored"] > ALPHA]
    imbalanced_failures = int(
        (
            failing.index.get_level_values("n_max")
            > failing.index.get_level_values("n_min")
        ).sum()
    )
    timing_display = timing.map(lambda seconds: f"{seconds:.1f}")
    lines += [
        markdown_table(paired_display, floatfmt=".3f"),
        "",
        f"- **The unfloored band can exceed the target.** Its miss rate is "
        f"above {ALPHA:.0%} at {len(failing)} of these sizes, up to "
        f"{percent(failing['miss_unfloored'].max())}, all with a smaller class "
        f"of {failing.index.get_level_values('n_min').max():,} or less; "
        f"{imbalanced_failures} of them have unequal classes. The hybrid's miss "
        f"rate there is at most {percent(failing['miss_hybrid'].max())}. These "
        f"are averages over shapes, so the worst shapes fail by more. This is "
        f"what the rule must screen out."
        if len(failing)
        else f"- **On average over shapes the unfloored band stays below "
        f"{ALPHA:.0%} at these sizes**; the worst shapes need not, which is "
        f"what the rule must screen out.",
        f"- **At small samples the floor is costly.** With at most 50 cases in "
        f"the larger class the hybrid is up to "
        f"{small['area_ratio'].max():.1f} times as wide as the unfloored band.",
        f"- **At large samples it buys nothing a user needs.** With "
        f"{n_max_paired[n_max_paired >= 5_000].min():,} or more in the larger "
        f"class the two areas agree within "
        f"{(large['area_ratio'] - 1).abs().max():.1%}. The floor still lowers "
        f"the miss rate under imbalance, but further below the target, which "
        f"is no gain when closer to nominal is better.",
        "- **And it is slow there.** The hybrid's ELL floor needs a calibration "
        "for every new pair of class sizes. Timed once per size on this "
        "machine:",
        "",
        markdown_table(timing_display, floatfmt=".1f"),
        "",
        "So the rule's job is to find where the unfloored band is safe and to "
        "send everything else to the hybrid.",
        "",
        "## 2. Data",
        "",
        "Each draw is one simulated dataset: scores from one of the two families "
        "at a chosen shape and AUC, at chosen class counts, with a record of "
        "whether the unfloored band missed the true ROC. The earlier studies "
        "sit on a few fixed sample sizes; this study's rounds placed draws "
        "between them and where the answer was least certain.",
        "",
        markdown_table(describe_sources(draws=draws, rounds=rounds), floatfmt=".4f"),
        "",
        "## 3. The model of the miss rate, and how well it predicts",
        "",
        "For each family the probability that the band misses is modelled as a "
        "smooth function of four quantities: the expected number of each class "
        "beyond the ROC's corners, `n_pos (1 - AUC)` and `n_neg (1 - AUC)`; the "
        "AUC; and the family's shape parameter. The model (a Gaussian process "
        "classifier) gives, at any setting, a best estimate of the miss rate "
        "and how uncertain that estimate is. The rule uses both.",
        "",
        f"Held-out check: the data were split into {folds} parts, and each part "
        f"was predicted from a fit to the others. By predicted miss rate:",
        "",
    ]
    reliability_display = pd.DataFrame(
        {
            "datasets": [f"{n:,}" for n in reliability["draws"]],
            "predicted": [percent(v, digits=2) for v in reliability["predicted"]],
            "observed": [percent(v, digits=2) for v in reliability["observed"]],
            "standard error": [percent(v, digits=2) for v in reliability["se"]],
        },
        index=pd.Index(
            [
                f"{max(interval.left, 0.0):.0%}-{interval.right:.0%}"
                for interval in reliability.index
            ],
            name="predicted miss rate",
        ),
    )
    lines += [
        markdown_table(reliability_display, floatfmt=".4f"),
        "",
        f"Between 2% and 12%, the range the rule depends on, predictions are "
        f"within {largest_deviation * 100:.1f} percentage points of the observed "
        f"rates. Around the {ALPHA:.0%} target the model "
        + (
            "predicts slightly more misses than occur, which errs toward caution."
            if bias > 0
            else "predicts slightly fewer misses than occur, which the assurance "
            "requirement has to absorb."
        ),
        "",
    ]

    cell_display = pd.DataFrame()
    if len(cell_table):
        cell_display = pd.DataFrame(
            {
                "family": [FAMILIES[f].title for f in cell_table["family"]],
                "shape": [f"{v:.3g}" for v in cell_table["shape"]],
                "AUC": [f"{v:.3f}" for v in cell_table["auc"]],
                "positives": [f"{v:,}" for v in cell_table["n_pos"]],
                "negatives": [f"{v:,}" for v in cell_table["n_neg"]],
                "datasets": [f"{v:,}" for v in cell_table["draws"]],
                "observed miss rate (95% interval)": [
                    f"{percent(r)} ({percent(lo)}-{percent(hi)})"
                    for r, lo, hi in zip(
                        cell_table["miss_rate"],
                        cell_table["wilson_lower"],
                        cell_table["wilson_upper"],
                        strict=True,
                    )
                ],
                "model's best estimate": [percent(v) for v in cell_table["p_surface"]],
                f"model's P(miss rate > {ALPHA:.0%})": [
                    percent(v) for v in cell_table["exceedance"]
                ],
                "routed to": [
                    "unfloored" if admit else "hybrid"
                    for admit in admits_cells(rule=recommended, cells=cell_table)
                ],
            },
            index=pd.Index(cell_table.index.get_level_values("cell"), name="cell"),
        )
        above = cell_table[cell_table["wilson_lower"] > ALPHA]
        above_admitted = admits_cells(rule=recommended, cells=above)
        outside = cell_table[
            (cell_table["p_surface"] < cell_table["wilson_lower"])
            | (cell_table["p_surface"] > cell_table["wilson_upper"])
        ]
        disagreements = [
            f"cell {cell} (observed "
            f"{'fewer' if row['miss_rate'] < row['p_surface'] else 'more'} misses "
            f"than predicted)"
            for (_, cell), row in outside.iterrows()
        ]
        above_ids = above.index.get_level_values("cell")
        cell_findings = [
            f"{within} of {len(cell_table)} cells' 95% intervals contain the "
            "model's best estimate"
            + (
                f"; the exception is {', '.join(disagreements)}"
                if disagreements
                else ""
            )
            + "."
        ]
        if len(above):
            cell_findings.append(
                f"Significantly above {ALPHA:.0%}: "
                + ", ".join(f"cell {c}" for c in above_ids)
                + (
                    ", which the rule sends to the hybrid."
                    if not above_admitted.any()
                    else ", of which the rule admits "
                    + ", ".join(f"cell {c}" for c in above_ids[above_admitted])
                    + "."
                )
            )
        else:
            cell_findings.append(f"No cell is significantly above {ALPHA:.0%}.")
        predicted_high = cell_table[cell_table["p_surface"] > ALPHA]
        if len(predicted_high):
            high_admitted = admits_cells(rule=recommended, cells=predicted_high)
            high_ids = predicted_high.index.get_level_values("cell")
            high_far = (
                predicted_high[["n_pos", "n_neg"]].max(axis=1) > ROUTED_N_MAX
            ).to_numpy()
            cell_findings.append(
                f"All {len(predicted_high)} cells whose best estimate exceeds "
                f"{ALPHA:.0%} are routed to the hybrid."
                if not high_admitted.any()
                else f"Of the {len(predicted_high)} cells whose best estimate "
                f"exceeds {ALPHA:.0%}, the rule admits "
                + ", ".join(
                    f"cell {c}"
                    + (f" (larger class above {ROUTED_N_MAX:,})" if far else "")
                    for c, far in zip(
                        high_ids[high_admitted], high_far[high_admitted], strict=True
                    )
                )
                + "."
            )
        lines += [
            "## 4. Replicated cells: a direct check where it matters",
            "",
            "Most draws sit at their own setting, which lets the model borrow "
            "strength across settings but leaves it to the model to say what "
            "happens at any one of them. To check the model where the rule's "
            "boundaries are decided, fixed settings were each simulated up to "
            "1,000 times: worst-case shapes at the edges of the admitted region, "
            "at large samples and at high imbalance.",
            "",
            markdown_table(cell_display, floatfmt=".4f"),
            "",
            " ".join(cell_findings),
            "",
        ]

    ladder_display = pd.DataFrame(
        {
            "rule": [r.describe() for r in ladder["rounded"]],
            "grid share": [
                percent(
                    rule_summary(rule=r, grid=grid, unsafe=unsafe)["admitted_share"]
                )
                for r in ladder["rounded"]
            ],
            "recommended": ["yes" if r else "" for r in ladder["recommended"]],
        },
        index=pd.Index(ladder["checks"], name="checks"),
    )
    lines += [
        "## 5. Deriving the rule",
        "",
        "### 5.1 Which settings are unsafe",
        "",
        f"The model is evaluated on a grid of settings: smaller class "
        f"{N_MIN_BOUNDS[0]:,}-{N_MIN_BOUNDS[1]:,} (log scale), AUC "
        f"{AUC_BOUNDS[0]:g}-{AUC_BOUNDS[1]:g} (probit scale) and class ratios "
        f"{', '.join(f'{r:.3g}' for r in GRID_RATIOS)}. At each setting it takes "
        f"the worst case over both families, their shapes and which class is "
        f"the minority. A setting is *unsafe* if the probability that this "
        f"worst-case miss rate exceeds {ALPHA:.0%} is above "
        f"{1 - assurance:.0%}. Requiring a probability, not only a best "
        f"estimate, below the target keeps the rule out of regions the data "
        f"cannot yet vouch for.",
        "",
        f"{percent(unsafe_anywhere.mean())} of the grid is unsafe. The rule must "
        f"exclude every unsafe setting with a larger class of at most "
        f"{ROUTED_N_MAX:,} ({percent(unsafe.mean())} of the grid). Beyond that "
        f"size the simulated data are sparse, Section 1 shows the floor buys no "
        f"width there and costs far more compute, and the unsafe settings mostly "
        f"reflect uncertainty about miss rates just below {ALPHA:.0%}. Section "
        f"5.5 checks the rule there against the replicated cells.",
        "",
        "### 5.2 Choosing the rule",
        "",
        "Candidate rules are conjunctions of simple checks: a minimum on the "
        "smaller class, a cap on the class ratio, a cap on the AUC, and a "
        "*corner* check that the AUC is low enough for the sample size. For "
        "every combination the thresholds are set to admit as much of the grid "
        "as possible without admitting an unsafe setting, then rounded to round "
        "numbers on the safe side. The best rule for each number of checks:",
        "",
        markdown_table(ladder_display, floatfmt=".4f"),
        "",
        f"The recommended rule is the one with the fewest checks that admits "
        f"within {PARSIMONY_TOLERANCE * 100:g} percentage point of the best "
        f"rule's grid share.",
        "",
        "### 5.3 What each check guards against",
        "",
        "Each check is credited with the unsafe settings that it alone "
        "excludes. Each turns out to bound one failure mechanism, identified by "
        "the family, minority class and shape at which the worst case occurs:",
        "",
    ]
    attribution = attribute_checks(rule=recommended, grid=grid, unsafe=unsafe)
    attribution = attribution.rename(
        columns={
            "unsafe_points": "unsafe settings",
            "n_min": "smaller class",
            "auc": "AUC",
            "ratio": "class ratio",
            "worst_family": "worst family",
            "worst_minority": "minority class",
            "worst_shape": "typical worst shape",
        }
    )
    attribution.index.name = "check"
    lines += [
        markdown_table(attribution, floatfmt=".3g"),
        "",
        "The last row counts settings that more than one check excludes; they "
        "lie deep inside the unsafe region and do not bear on any one check.",
        "",
        "### 5.4 Sensitivity, and the previous rule",
        "",
        "The recommended rule at other assurance levels, and the previous rule "
        "judged by the same test:",
        "",
    ]
    by_level = {}
    for level in ASSURANCE_LEVELS:
        level_unsafe = routed_unsafe(grid=grid, assurance=level)
        level_ladder = recommend(search=search_rules(grid=grid, unsafe=level_unsafe))
        rule = level_ladder.loc[level_ladder["recommended"], "rounded"].iloc[0]
        by_level[f"recommended at {level:.0%} assurance"] = {
            "rule": rule.describe(),
            **summary_columns(
                summary=rule_summary(rule=rule, grid=grid, unsafe=level_unsafe)
            ),
        }
    by_level[f"previous, judged at {assurance:.0%}"] = {
        "rule": REFERENCE_RULE.describe(),
        **summary_columns(summary=summaries["previous"]),
    }
    if "alternative" in rules:
        by_level[f"alternative at {assurance:.0%} assurance"] = {
            "rule": rules["alternative"].describe(),
            **summary_columns(summary=summaries["alternative"]),
        }
    beyond = grid[admitted_grid & ~routed]
    beyond_high = beyond[beyond["p_mean"] > ALPHA]
    beyond_unsafe = int(unsafe_anywhere[admitted_grid & ~routed].sum())
    sensitivity = pd.DataFrame(by_level).T
    sensitivity.index.name = "rule"
    sensitivity = sensitivity.rename(columns={"rule": "checks"})
    lines += [
        markdown_table(sensitivity, floatfmt=".4f"),
        "",
        "The checks keep their form across assurance levels; the thresholds move.",
        "",
        f"### 5.5 Beyond a larger class of {ROUTED_N_MAX:,}",
        "",
        f"The rule admits {len(beyond):,} grid settings there, "
        f"{beyond_unsafe:,} of them unsafe by the test above. At "
        f"{beyond_unsafe - len(beyond_high):,} of those the best estimate is "
        f"below {ALPHA:.0%} and only its uncertainty fails the test. The best "
        f"estimate exceeds {ALPHA:.0%} at {len(beyond_high):,}"
        + (
            f", all with a larger class of "
            f"{(beyond_high['n_min'] * beyond_high['ratio']).min():,.0f} or more, "
            f"where it extends a rising trend beyond the data"
            if len(beyond_high)
            else ""
        )
        + ". The replicated cells in that range that the rule admits measure "
        "the band directly:",
        "",
    ]
    lines += [
        markdown_table(cell_display[far], floatfmt=".4f")
        if far.any()
        else "No replicated cell lies there.",
        "",
    ]
    lines += [
        "### 5.6 Safe settings the rule sends to the hybrid",
        "",
        "Sending a safe setting to the hybrid costs width at small samples and "
        "compute at large ones (Section 1). A setting goes to the hybrid either "
        "because it is unsafe or because a few simple checks cannot carve it "
        "out from unsafe neighbours. The table splits the part of the grid each "
        "rule sends to the hybrid by its best estimate and by the probability "
        "that it is safe:",
        "",
        markdown_table(breakdown, floatfmt=".4f"),
        "",
        "**Where the slack is.** At each smaller class and class ratio, the "
        "largest AUC the "
        + " and the ".join(f"{label} rule" for label in candidates)
        + f" admit, then the largest AUC up to which every setting is safe at "
        f"{assurance:.0%} (`none` where the lowest AUC already fails). A † marks "
        f"a larger class above {ROUTED_N_MAX:,}, where rules are not held to "
        "the test:",
        "",
        markdown_table(headroom, floatfmt=".3f"),
        "",
        "A single corner check bounds `1 - AUC` by one power of the smaller "
        "class, a straight line on log scales. The safe limit is not straight: "
        "it rises quickly with the smaller class at small samples and levels "
        "off at large ones. Two corner checks of different exponents follow "
        f"such a bend. The best rule of that form, `{bent.describe()}`, admits "
        f"{percent(bent_share)} of the grid against "
        f"{percent(summaries['recommended']['admitted_share'])} for the "
        "recommended rule; the rest of the slack lies in thin slivers beside "
        "unsafe settings, which no few checks can follow.",
        "",
    ]
    for label, rule in candidates.items():
        table = regions[label]
        lines += [
            f"**{label.capitalize()} rule** (`{rule.describe()}`). "
            + (
                f"The largest connected regions it sends to the hybrid that are "
                f"safe with {SAFE_CONFIDENCE:.0%} confidence (settings are "
                f"connected if they differ by one step of the grid in the "
                f"smaller class, the AUC or the class ratio):"
                if len(table)
                else f"It sends no setting that is safe with "
                f"{SAFE_CONFIDENCE:.0%} confidence to the hybrid."
            ),
            "",
        ]
        if len(table):
            lines += [markdown_table(table, floatfmt=".4f"), ""]
        lines += [
            "A second clause admitted beside the rule (`rule or clause`) can "
            "route part of this without loosening the rule's checks. Searched "
            "like the rule over the settings it sends to the hybrid, and held "
            "to the test at every class size:",
            "",
            markdown_table(clauses[label], floatfmt=".4f"),
            "",
        ]
    lines += [
        "*Sent to the hybrid by* lists the checks that at least half of a "
        "region fails, with the share failing. *Draws at the hardest setting* "
        f"is a rough count of the effective draws the region's least certain "
        f"setting still needs for {assurance:.0%} assurance, treating its "
        "posterior as binomial at the best estimate; draws nearby inform it "
        "too, so this is a guide to the information required rather than a "
        "design. Regions whose settings are mostly safe at "
        f"{assurance:.0%} already need only a different rule; the rest need "
        "draws first.",
        "",
    ]
    lines += [
        "## 6. Model-free check: the simulated datasets the rule admits",
        "",
        "The simulated datasets inside the recommended rule, grouped by family, "
        "class ratio and third of the family's shape range (on a log scale). "
        "Each group pools easier and harder shapes, so this confirms the rule "
        "rather than testing its worst case, which is what the model is for.",
        "",
    ]
    admitted_display = pd.DataFrame(
        {
            "datasets": [f"{n:,}" for n in admitted["draws"]],
            "miss rate": [percent(v, digits=2) for v in admitted["miss_rate"]],
            "95% upper bound": [percent(v, digits=2) for v in admitted["wilson_upper"]],
        },
        index=pd.MultiIndex.from_tuples(
            [
                (
                    FAMILIES[family].title,
                    f"{max(band.left, 1.0):.3g}-{band.right:.3g}",
                    shape_third_label(family=family, third=third),
                )
                for family, band, third in admitted.index
            ],
            names=["family", "class ratio", "shape"],
        ),
    )
    lines += [markdown_table(admitted_display, floatfmt=".4f"), ""]
    if len(large_groups):
        worst = large_groups["miss_rate"].idxmax()
        lines += [
            f"Over all {admitted_draws:,} datasets the miss rate is "
            f"{percent(admitted_rate, digits=2)}. The highest among groups of at "
            f"least {MIN_GROUP_DRAWS} datasets is "
            f"{percent(large_groups.loc[worst, 'miss_rate'], digits=2)} "
            f"({FAMILIES[worst[0]].title}, class ratio "
            f"{max(worst[1].left, 1.0):.3g}-{worst[1].right:.3g}, "
            f"{shape_third_label(family=worst[0], third=worst[2])}).",
            "",
        ]
    lines += [
        "## 7. Figure",
        "",
        "`surface.png` shows, for five class ratios, the worst-case best "
        "estimate of the miss rate (top row; blue below 5%, red above) and the "
        f"probability that it exceeds {ALPHA:.0%} (bottom row). The black "
        f"contour marks the {assurance:.0%} assurance boundary; the solid green "
        "line encloses the settings the recommended rule admits, the dashed "
        "line those of the previous rule and the dash-dot line those of the "
        "alternative rule, where there is one (Section 5.6). The dotted black "
        "line marks a larger "
        f"class of {ROUTED_N_MAX:,}, beyond which the rule is not held to the "
        "assurance test (Section 5.5), so the green line may cross the black "
        "contour to its right. The grey line marks the largest smaller class "
        "that can be simulated at that ratio.",
        "",
        "## Appendix: best rule for every combination of checks",
        "",
        markdown_table(
            pd.DataFrame(
                {
                    "rule (exact thresholds)": [
                        fit.rule.describe() for fit in search["fit"]
                    ],
                    "grid share": [percent(v) for v in search["admitted_share"]],
                },
                index=pd.Index(search["kinds"], name="checks"),
            ),
            floatfmt=".4f",
        ),
        "",
        "Model length scales (positive corner, negative corner, AUC, shape): "
        + "; ".join(
            f"{FAMILIES[name].title} {np.round(surface.lengthscales, 3).tolist()}, "
            f"amplitude {surface.amplitude:.2f}"
            for name, surface in surfaces.items()
        )
        + ".",
        "",
    ]
    return rules, "\n".join(lines)


# Results of a fit that the report reuses, so that it can be rewritten without
# refitting
FIT_CACHE = "fit_cache.pkl"


@dataclass(frozen=True, kw_only=True)
class FitArtifacts:
    """The slow parts of a fit, cached for :func:`command_report`.

    Attributes:
        fingerprint: :func:`draws_fingerprint` of the draws fitted.
        seed: Seed of the fit and the cross-validation.
        folds: Number of cross-validation folds.
        surfaces: Fitted surface by family name.
        predictions: Output of :func:`cross_validate`.
        grid: Output of :func:`worst_case_grid`.
        paired: Output of :func:`paired_floor_effect`.
        timing: Output of :func:`time_bands`.
    """

    fingerprint: str
    seed: int
    folds: int
    surfaces: dict[str, Surface]
    predictions: pd.DataFrame
    grid: pd.DataFrame
    paired: pd.DataFrame
    timing: pd.DataFrame


def draws_fingerprint(draws: pd.DataFrame) -> str:
    """Hash of the draws' contents, to tell whether a cached fit is current."""
    hashed = pd.util.hash_pandas_object(draws, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def load_fit_draws(args: argparse.Namespace) -> pd.DataFrame:
    """The draws a fit uses: every draw in the design space, misses as bool."""
    draws = in_design_space(
        frame=load_draws(hot_start=args.hot_start, out_dir=args.out_dir)
    )
    draws["miss"] = draws["miss"].astype(bool)
    return draws


def write_outputs(
    *, artifacts: FitArtifacts, draws: pd.DataFrame, out_dir: Path, assurance: float
) -> None:
    """Derive the rules and write the report, figure, grid and rules.

    Args:
        artifacts: The fit.
        draws: The draws fitted.
        out_dir: Directory of the outputs.
        assurance: Posterior probability of safety the rules are held to.
    """
    artifacts.grid.to_feather(out_dir / "grid.feather")
    rules, report = build_report(
        draws=draws,
        rounds=load_rounds(out_dir),
        surfaces=artifacts.surfaces,
        predictions=artifacts.predictions,
        grid=artifacts.grid,
        paired=artifacts.paired,
        timing=artifacts.timing,
        assurance=assurance,
        folds=artifacts.folds,
    )
    plot_surface(
        grid=artifacts.grid,
        rules=rules,
        assurance=assurance,
        path=out_dir / "surface.png",
    )
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "rules.json").write_text(
        json.dumps(
            {
                "assurance": assurance,
                **{
                    label: [asdict(check) for check in rule.checks]
                    for label, rule in rules.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(report)


def command_fit(args: argparse.Namespace) -> None:
    """Fit the surfaces, cache the fit, derive the rules and write the report."""
    draws = load_fit_draws(args)
    surfaces = fit_surfaces(draws=draws, seed=args.seed)
    artifacts = FitArtifacts(
        fingerprint=draws_fingerprint(draws),
        seed=args.seed,
        folds=args.folds,
        surfaces=surfaces,
        predictions=cross_validate(
            draws=draws, surfaces=surfaces, folds=args.folds, seed=args.seed
        ),
        grid=worst_case_grid(surfaces=surfaces, target=ALPHA),
        paired=paired_floor_effect(directories=args.hot_start),
        timing=time_bands(sizes=TIMING_SIZES),
    )
    with (args.out_dir / FIT_CACHE).open("wb") as handle:
        pickle.dump(artifacts, handle)
    write_outputs(
        artifacts=artifacts, draws=draws, out_dir=args.out_dir, assurance=args.assurance
    )


def command_report(args: argparse.Namespace) -> None:
    """Rewrite the report from the cached fit, without refitting."""
    cache = args.out_dir / FIT_CACHE
    if not cache.exists():
        raise SystemExit(f"No cached fit at {cache}; run fit first")
    with cache.open("rb") as handle:
        artifacts: FitArtifacts = pickle.load(handle)
    draws = load_fit_draws(args)
    if draws_fingerprint(draws) != artifacts.fingerprint:
        raise SystemExit("The draws have changed since the cached fit; run fit")
    write_outputs(
        artifacts=artifacts, draws=draws, out_dir=args.out_dir, assurance=args.assurance
    )


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
        help="Use the value criterion in round 0, from a fit to the hot start",
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
        "--close-round",
        type=float,
        nargs="?",
        const=0.0,
        default=None,
        metavar="CPU_HOURS",
        help="End the unfinished round after a stratified sample of its "
        "remaining draws worth CPU_HOURS of measured single-threaded time "
        "(none if omitted); the full design is kept as "
        "untrimmed_design_round<k>.feather",
    )
    run.add_argument(
        "--cells",
        type=Path,
        default=None,
        help="CSV of fixed cells (family, shape, auc, n_pos, n_neg, and "
        "optionally reps) to replicate as the next round, instead of a "
        "designed round",
    )
    run.add_argument(
        "--reps", type=int, default=1000, help="Draws per cell without its own reps"
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
    fit.add_argument(
        "--assurance",
        type=float,
        default=0.9,
        help="Posterior probability, at every admitted point, shape and "
        "family, that the miss rate is at most the target",
    )
    fit.set_defaults(handler=command_fit)

    report = commands.add_parser(
        "report",
        parents=[shared],
        formatter_class=formatter,
        help="Rewrite the report, figure and rules from the cached fit",
    )
    report.add_argument(
        "--assurance",
        type=float,
        default=0.9,
        help="Posterior probability, at every admitted point, shape and "
        "family, that the miss rate is at most the target",
    )
    report.set_defaults(handler=command_report)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
