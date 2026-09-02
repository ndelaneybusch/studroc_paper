"""Flexible refits of the C = 1 boundary surface (spec follow-up item 1b).

The pre-registered surface is a sign-constrained logistic *linear* smooth in
(log n, log df, probit AUC). Its holdout check against the classification-grade
anchors fails: 8 of 11 anchors land outside their measured Wilson intervals, and
the miss is anti-conservative in the heavy-tail/high-AUC corner (fitted .921
against a measured .690 at t(2)/AUC .99/n = 500). A form linear in the
covariates cannot track a coverage cliff.

This module refits the same 95-cell LHS sweep with two flexible alternatives and
scores all three against the untouched anchors:

- ``tprs`` — a thin plate regression spline (Wood 2003): the isotropic thin
  plate basis on standardized covariates, rank-reduced by eigen-truncation of
  the constrained penalty, fitted by penalized IRLS under a binomial likelihood
  with the smoothing parameter chosen by UBRE.
- ``gp`` — a Gaussian process with a binomial likelihood, a linear mean
  function and an ARD Matern-5/2 kernel, fitted by Laplace approximation with
  hyperparameters set by maximum approximate marginal likelihood.

Neither imposes the monotone tail-mass mechanism that the pre-registered fit
builds in by construction, so monotonicity becomes a testable diagnostic rather
than an assumption.

The LHS cells are the training set and the student-t anchors are the holdout;
the anchors enter no fit at any point.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.linalg import cho_solve, cholesky, qr
from scipy.optimize import minimize
from scipy.special import expit, gammaln
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from followup_runs import (  # noqa: E402
    BAR_POINT,
    DEFAULT_OUT,
    LHS_N_BOUNDS,
    fit_boundary_surface,
    register_followup_shapes,
    surface_predict,
    wilson_ci,
)

EPS = 1e-9
TPS_RANK = 45  # eigen-truncation rank of the thin plate basis
CONS_Q = 0.10  # lower-quantile level for the conservative GP contour


@dataclass(frozen=True)
class Cell:
    """One measured cell of the boundary study.

    Attributes:
        df: Student-t degrees of freedom of the shape.
        auc: True AUC of the shape.
        n: Per-class sample size.
        cov: Measured C = 1 coverage at alpha = .05.
        reps: Replicates behind ``cov``.
        name: Cell name, for reporting.
    """

    df: float
    auc: float
    n: int
    cov: float
    reps: int
    name: str = ""

    @property
    def successes(self) -> int:
        """Covered replicates, rounded to the nearest integer."""
        return int(round(self.cov * self.reps))


def design_matrix(cells: list[Cell]) -> np.ndarray:
    """The (log n, log df, probit AUC) covariates of ``cells``."""
    return np.column_stack(
        [
            np.log([c.n for c in cells]),
            np.log([c.df for c in cells]),
            norm.ppf([c.auc for c in cells]),
        ]
    )


def binomial_deviance(
    *, successes: np.ndarray, trials: np.ndarray, p: np.ndarray
) -> float:
    """Binomial deviance of ``p`` against observed counts."""
    p = np.clip(p, EPS, 1 - EPS)
    k, m = np.asarray(successes, float), np.asarray(trials, float)
    t1 = np.where(k > 0, k * np.log(np.maximum(k, EPS) / (m * p)), 0.0)
    t2 = np.where(k < m, (m - k) * np.log(np.maximum(m - k, EPS) / (m * (1 - p))), 0.0)
    return float(2.0 * (t1 + t2).sum())


# ---------------------------------------------------------------------------
# thin plate regression spline
# ---------------------------------------------------------------------------


def _tps_kernel(r: np.ndarray, *, d: int = 3, m: int = 2) -> np.ndarray:
    """The thin plate spline radial basis for dimension ``d`` and order ``m``.

    For odd ``d`` with ``2m > d`` this is a positive multiple of
    ``r**(2m - d)``; the shared constant is absorbed by the smoothing
    parameter, so only its sign matters to the fit.
    """
    if (2 * m - d) % 2 == 0:
        raise ValueError("even 2m-d requires the log-form kernel; unused here")
    c = np.exp(gammaln(m - d / 2) - (2 * m) * np.log(2) - (d / 2) * np.log(np.pi))
    return c * r ** (2 * m - d)


@dataclass
class TprsBasis:
    """A rank-reduced thin plate basis anchored at the training covariates."""

    knots: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    u: np.ndarray  # eigen-truncation map of the constrained penalty
    penalty: np.ndarray  # per-column penalty weights (the kept eigenvalues)
    sign: float

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.center) / self.scale

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Model matrix at ``x``: the null-space polynomials then the wiggly part."""
        xs = self._standardize(np.atleast_2d(x))
        r = np.linalg.norm(xs[:, None, :] - self.knots[None, :, :], axis=2)
        e = self.sign * _tps_kernel(r)
        return np.column_stack([np.ones(len(xs)), xs, e @ self.u])

    @property
    def n_null(self) -> int:
        """Number of unpenalized null-space columns."""
        return self.knots.shape[1] + 1


def build_tprs_basis(x: np.ndarray, *, rank: int = TPS_RANK) -> TprsBasis:
    """Construct the thin plate regression basis of Wood (2003).

    The isotropic thin plate penalty acts on distances, so the covariates are
    standardized first; the wiggly part is reparametrized onto the null space of
    the polynomial terms and truncated to the ``rank`` leading eigenvectors of
    the constrained penalty.

    Args:
        x: Training covariates, shape (n, d).
        rank: Number of retained eigenvectors.

    Returns:
        The basis, callable on new covariate matrices.
    """
    center, scale = x.mean(axis=0), x.std(axis=0)
    xs = (x - center) / scale
    n, d = xs.shape
    r = np.linalg.norm(xs[:, None, :] - xs[None, :, :], axis=2)
    e = _tps_kernel(r, d=d)

    t = np.column_stack([np.ones(n), xs])
    q, _ = qr(t, mode="full")
    z = q[:, t.shape[1] :]  # orthonormal basis of null(T^T)

    # The kernel's leading constant fixes only a scale, which the smoothing
    # parameter absorbs; its sign is what makes the penalty positive
    # semi-definite on the null space of the polynomial terms. For odd d the
    # conditionally positive-definite orientation is the negative of the
    # gamma-function constant, so it is selected by an explicit definiteness
    # test rather than assumed.
    s = z.T @ e @ z
    sign = 1.0 if np.linalg.eigvalsh(s).min() >= -1e-12 else -1.0
    s = sign * s
    vals, vecs = np.linalg.eigh(s)
    if vals.min() < -1e-8 * vals.max():
        raise ValueError("thin plate penalty is indefinite on the null space")

    keep = np.argsort(vals)[::-1][: min(rank, (vals > 1e-10 * vals.max()).sum())]
    return TprsBasis(
        knots=xs,
        center=center,
        scale=scale,
        u=z @ vecs[:, keep],
        penalty=vals[keep],
        sign=sign,
    )


@dataclass
class PirlsFit:
    """A penalized IRLS fit at one smoothing parameter."""

    beta: np.ndarray
    edf: float
    deviance: float
    converged: bool


def penalized_irls(
    *,
    model: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
    penalty: np.ndarray,
    lam: float,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> PirlsFit:
    """Fit a penalized binomial GLM by iteratively reweighted least squares.

    Args:
        model: Model matrix.
        successes: Covered replicates per cell.
        trials: Replicates per cell.
        penalty: Diagonal of the penalty matrix (zero on unpenalized columns).
        lam: Smoothing parameter.
        max_iter: Maximum IRLS iterations.
        tol: Convergence tolerance on the deviance.

    Returns:
        The fit, with effective degrees of freedom and deviance.
    """
    s = np.diag(lam * penalty)
    beta = np.zeros(model.shape[1])
    beta[0] = np.log((successes.sum() + 0.5) / (trials.sum() - successes.sum() + 0.5))
    dev_old = np.inf
    converged = False
    for _ in range(max_iter):
        eta = model @ beta
        mu = np.clip(expit(eta), EPS, 1 - EPS)
        w = trials * mu * (1 - mu)
        z = eta + (successes - trials * mu) / np.maximum(w, EPS)
        lhs = model.T @ (w[:, None] * model) + s
        rhs = model.T @ (w * z)
        beta = np.linalg.solve(lhs + 1e-10 * np.eye(len(beta)), rhs)
        dev = binomial_deviance(
            successes=successes, trials=trials, p=expit(model @ beta)
        )
        if abs(dev_old - dev) < tol * (abs(dev) + 0.1):
            converged = True
            break
        dev_old = dev
    eta = model @ beta
    mu = np.clip(expit(eta), EPS, 1 - EPS)
    w = trials * mu * (1 - mu)
    xtwx = model.T @ (w[:, None] * model)
    edf = float(np.trace(np.linalg.solve(xtwx + s + 1e-10 * np.eye(len(beta)), xtwx)))
    return PirlsFit(
        beta=beta,
        edf=edf,
        deviance=binomial_deviance(successes=successes, trials=trials, p=mu),
        converged=converged,
    )


@dataclass
class TprsModel:
    """A fitted thin plate regression spline surface."""

    basis: TprsBasis
    beta: np.ndarray
    lam: float
    edf: float
    label: str = "tprs"
    extras: dict = field(default_factory=dict)

    def predict(self, *, df: float, auc: float, n: float) -> float:
        """Fitted coverage at one (df, AUC, n) point."""
        x = np.array([[np.log(n), np.log(df), norm.ppf(auc)]])
        return float(expit(self.basis(x) @ self.beta)[0])


def fit_tprs(cells: list[Cell], *, rank: int = TPS_RANK) -> TprsModel:
    """Fit a thin plate regression spline with UBRE-selected smoothing.

    The scale parameter of a binomial likelihood is known, so UBRE (equivalently
    an AIC on the deviance scale) is the natural smoothing-parameter criterion.

    Args:
        cells: Training cells.
        rank: Basis rank passed to :func:`build_tprs_basis`.

    Returns:
        The fitted surface.
    """
    x = design_matrix(cells)
    basis = build_tprs_basis(x, rank=rank)
    model = basis(x)
    penalty = np.concatenate([np.zeros(basis.n_null), basis.penalty])
    k = np.array([c.successes for c in cells], float)
    m = np.array([c.reps for c in cells], float)

    best, best_score, best_lam = None, np.inf, None
    for lam in np.logspace(-6, 8, 141):
        fit = penalized_irls(
            model=model, successes=k, trials=m, penalty=penalty, lam=lam
        )
        ubre = fit.deviance / len(cells) + 2.0 * fit.edf / len(cells) - 1.0
        if ubre < best_score:
            best, best_score, best_lam = fit, ubre, lam
    return TprsModel(
        basis=basis,
        beta=best.beta,
        lam=best_lam,
        edf=best.edf,
        extras={"ubre": best_score, "deviance": best.deviance},
    )


# ---------------------------------------------------------------------------
# Gaussian process with a binomial likelihood
# ---------------------------------------------------------------------------


def matern52(*, a: np.ndarray, b: np.ndarray, lengthscales: np.ndarray) -> np.ndarray:
    """ARD Matern-5/2 correlation between two covariate sets."""
    d = (a[:, None, :] - b[None, :, :]) / lengthscales
    r = np.sqrt(np.maximum((d**2).sum(axis=2), 0.0))
    s = np.sqrt(5.0) * r
    return (1.0 + s + s**2 / 3.0) * np.exp(-s)


@dataclass
class LaplaceState:
    """The Laplace mode of the latent field and the pieces needed to predict."""

    f: np.ndarray
    w: np.ndarray
    grad: np.ndarray
    log_marginal: float


def _laplace_mode(
    *,
    kernel: np.ndarray,
    mean: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-9,
) -> LaplaceState:
    """Newton iteration to the Laplace mode of a binomial-likelihood GP."""
    n = len(successes)
    f = mean.copy()
    a = np.zeros(n)
    obj_old = -np.inf
    for _ in range(max_iter):
        p = np.clip(expit(f), EPS, 1 - EPS)
        grad = successes - trials * p
        w = np.maximum(trials * p * (1 - p), 1e-12)
        sw = np.sqrt(w)
        chol = cholesky(np.eye(n) + sw[:, None] * kernel * sw[None, :], lower=True)
        rhs = w * (f - mean) + grad
        a = rhs - sw * cho_solve((chol, True), sw * (kernel @ rhs))
        f = mean + kernel @ a
        p = np.clip(expit(f), EPS, 1 - EPS)
        loglik = float(
            (successes * np.log(p) + (trials - successes) * np.log(1 - p)).sum()
        )
        # a is K^-1 (f - mean) by construction, so this is the Laplace objective
        # log p(y|f) - (f-mean)' K^-1 (f-mean) / 2.
        obj = loglik - 0.5 * float(a @ (f - mean))
        if abs(obj - obj_old) < tol * (abs(obj) + 1.0):
            break
        obj_old = obj
    p = np.clip(expit(f), EPS, 1 - EPS)
    grad = successes - trials * p
    w = np.maximum(trials * p * (1 - p), 1e-12)
    sw = np.sqrt(w)
    chol = cholesky(np.eye(n) + sw[:, None] * kernel * sw[None, :], lower=True)
    loglik = float((successes * np.log(p) + (trials - successes) * np.log(1 - p)).sum())
    log_marginal = (
        loglik - 0.5 * float(a @ (f - mean)) - float(np.log(np.diag(chol)).sum())
    )
    return LaplaceState(f=f, w=w, grad=grad, log_marginal=log_marginal)


@dataclass
class GpModel:
    """A fitted binomial-likelihood GP surface with a linear mean function."""

    x_train: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    lengthscales: np.ndarray
    amplitude: float
    mean_beta: np.ndarray
    state: LaplaceState
    kernel: np.ndarray
    label: str = "gp"
    extras: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        sw = np.sqrt(self.state.w)
        self._sqrt_w = sw
        self._chol = cholesky(
            np.eye(len(self.x_train)) + sw[:, None] * self.kernel * sw[None, :],
            lower=True,
        )

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.center) / self.scale

    def _mean(self, xs: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(xs)), xs]) @ self.mean_beta

    def latent(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation of the latent field at ``x``."""
        xs = self._standardize(np.atleast_2d(x))
        ks = self.amplitude * matern52(
            a=xs, b=self.x_train, lengthscales=self.lengthscales
        )
        mu = self._mean(xs) + ks @ self.state.grad
        scaled = self._sqrt_w[:, None] * ks.T
        v = cho_solve((self._chol, True), scaled)
        var = self.amplitude - np.einsum("ij,ij->j", scaled, v)
        return mu, np.sqrt(np.maximum(var, 0.0))

    def latent_cov(self, x: np.ndarray) -> np.ndarray:
        """Joint posterior covariance of the latent field across the rows of ``x``.

        Batch design needs the joint law, not the marginals: conditioning on a
        new observation reduces the variance at every correlated point, and for
        a GP that reduction is known before the value is seen.
        """
        xs = self._standardize(np.atleast_2d(x))
        kxx = self.amplitude * matern52(
            a=xs, b=xs, lengthscales=self.lengthscales
        )
        ks = self.amplitude * matern52(
            a=xs, b=self.x_train, lengthscales=self.lengthscales
        )
        scaled = self._sqrt_w[:, None] * ks.T
        return kxx - scaled.T @ cho_solve((self._chol, True), scaled)

    def predict(self, *, df: float, auc: float, n: float) -> float:
        """Posterior mean coverage, averaging the logit link over the latent."""
        x = np.array([[np.log(n), np.log(df), norm.ppf(auc)]])
        mu, sd = self.latent(x)
        return float(expit(mu / np.sqrt(1.0 + np.pi * sd**2 / 8.0))[0])

    def predict_quantile(self, *, df: float, auc: float, n: float, q: float) -> float:
        """Coverage at posterior quantile ``q`` of the latent field."""
        x = np.array([[np.log(n), np.log(df), norm.ppf(auc)]])
        mu, sd = self.latent(x)
        return float(expit(mu + norm.ppf(q) * sd)[0])


def fit_gp(cells: list[Cell], *, restarts: int = 4, seed: int = 20260831) -> GpModel:
    """Fit a binomial GP by Laplace approximation and ML-II hyperparameters.

    The mean function is linear in the covariates, so the GP nests the
    pre-registered logistic-linear form and departures from it are carried by
    the kernel rather than by an arbitrary constant-mean reversion.

    Args:
        cells: Training cells.
        restarts: Random restarts of the hyperparameter optimizer.
        seed: Seed for the restart draws.

    Returns:
        The fitted surface.
    """
    x = design_matrix(cells)
    center, scale = x.mean(axis=0), x.std(axis=0)
    xs = (x - center) / scale
    k = np.array([c.successes for c in cells], float)
    m = np.array([c.reps for c in cells], float)
    t = np.column_stack([np.ones(len(xs)), xs])

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        return np.exp(theta[:3]), float(np.exp(theta[3])), theta[4:]

    def nll(theta: np.ndarray) -> float:
        ls, amp, mb = unpack(theta)
        kern = amp * matern52(a=xs, b=xs, lengthscales=ls) + 1e-6 * np.eye(len(xs))
        if not np.all(np.isfinite(kern)):
            return 1e12
        try:
            state = _laplace_mode(kernel=kern, mean=t @ mb, successes=k, trials=m)
        except (np.linalg.LinAlgError, ValueError):
            return 1e12
        if not np.isfinite(state.log_marginal):
            return 1e12
        return -state.log_marginal

    # Inputs are standardized, so a lengthscale far outside [e^-3, e^4] SD units
    # is either an interpolation spike or a flat mean; both are excluded to keep
    # the Laplace inner solve conditioned.
    bounds = [(-3.0, 4.0)] * 3 + [(-4.0, 5.0)] + [(-25.0, 25.0)] * 4
    p0 = np.log(np.clip(k.sum() / m.sum(), EPS, 1 - EPS) / (1 - k.sum() / m.sum()))
    rng = np.random.default_rng(seed)
    starts = [np.array([0.0, 0.0, 0.0, np.log(2.0), p0, 0.0, 0.0, 0.0])]
    for _ in range(restarts):
        starts.append(
            np.concatenate(
                [
                    rng.normal(0.0, 0.7, 3),
                    [rng.normal(0.7, 0.5)],
                    [p0],
                    rng.normal(0, 0.3, 3),
                ]
            )
        )
    best, best_val = None, np.inf
    for s0 in starts:
        s0 = np.clip(s0, [b[0] for b in bounds], [b[1] for b in bounds])
        res = minimize(
            nll, s0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300}
        )
        if res.fun < best_val:
            best, best_val = res.x, res.fun
    if best is None:
        raise RuntimeError("GP hyperparameter optimization failed from every start")
    ls, amp, mb = unpack(best)
    kern = amp * matern52(a=xs, b=xs, lengthscales=ls) + 1e-6 * np.eye(len(xs))
    state = _laplace_mode(kernel=kern, mean=t @ mb, successes=k, trials=m)
    return GpModel(
        x_train=xs,
        center=center,
        scale=scale,
        lengthscales=ls,
        amplitude=amp,
        mean_beta=mb,
        state=state,
        kernel=kern,
        extras={"log_marginal": state.log_marginal},
    )


# ---------------------------------------------------------------------------
# the pre-registered baseline, wrapped to a common interface
# ---------------------------------------------------------------------------


@dataclass
class LinearModel:
    """The pre-registered sign-constrained logistic-linear surface."""

    beta: np.ndarray
    label: str = "logistic-linear"
    extras: dict = field(default_factory=dict)

    def predict(self, *, df: float, auc: float, n: float) -> float:
        """Fitted coverage at one (df, AUC, n) point."""
        return surface_predict(self.beta, df, auc, n)


def fit_linear(cells: list[Cell]) -> LinearModel:
    """Refit the pre-registered surface on the same training cells."""
    rows = [
        {"df": c.df, "auc": c.auc, "n": c.n, "cov": c.cov, "reps": c.reps}
        for c in cells
    ]
    return LinearModel(beta=fit_boundary_surface(rows)["beta"])


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def n_star(
    model,
    *,
    df: float,
    auc: float,
    bar: float = BAR_POINT,
    quantile: float | None = None,
) -> float:
    """Smallest n in the sampled range whose fitted coverage clears ``bar``.

    Args:
        model: A fitted surface exposing ``predict`` (and ``predict_quantile``
            when ``quantile`` is given).
        df: Degrees of freedom.
        auc: True AUC.
        bar: Coverage bar to cross.
        quantile: Posterior quantile to read instead of the mean, for models
            that expose one.

    Returns:
        The crossing n, ``0`` if the surface clears the bar throughout the
        sampled range, or ``inf`` if it never does.
    """

    def cov_at(n: float) -> float:
        if quantile is not None:
            return model.predict_quantile(df=df, auc=auc, n=n, q=quantile)
        return model.predict(df=df, auc=auc, n=n)

    lo_n, hi_n = float(LHS_N_BOUNDS[0]), float(LHS_N_BOUNDS[1])
    if cov_at(lo_n) >= bar:
        return 0.0
    if cov_at(hi_n) < bar:
        return float("inf")
    lo, hi = np.log(lo_n), np.log(hi_n)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if cov_at(float(np.exp(mid))) >= bar:
            hi = mid
        else:
            lo = mid
    return float(np.exp(0.5 * (lo + hi)))


@dataclass
class Scores:
    """Holdout or cross-validation scores for one surface."""

    label: str
    dev_per_rep: float
    rmse: float
    mae: float
    bias: float
    worst_optimistic: float
    inside_ci: int
    n_cells: int


def score(model, cells: list[Cell], *, label: str) -> Scores:
    """Score a fitted surface against measured cells.

    Args:
        model: A fitted surface exposing ``predict``.
        cells: Cells to score against; never used in fitting.
        label: Name for the report.

    Returns:
        Deviance per replicate (the proper scoring rule), plus probability-scale
        error summaries. ``worst_optimistic`` is the largest amount by which the
        surface overstates measured coverage — the anti-conservative direction
        that matters for routing.
    """
    pred = np.array([model.predict(df=c.df, auc=c.auc, n=c.n) for c in cells])
    obs = np.array([c.cov for c in cells])
    k = np.array([c.successes for c in cells], float)
    m = np.array([c.reps for c in cells], float)
    inside = sum(
        wilson_ci(c.cov, c.reps)[0] <= p <= wilson_ci(c.cov, c.reps)[1]
        for c, p in zip(cells, pred, strict=True)
    )
    return Scores(
        label=label,
        dev_per_rep=binomial_deviance(successes=k, trials=m, p=pred) / m.sum(),
        rmse=float(np.sqrt(np.mean((pred - obs) ** 2))),
        mae=float(np.mean(np.abs(pred - obs))),
        bias=float(np.mean(pred - obs)),
        worst_optimistic=float(np.max(pred - obs)),
        inside_ci=int(inside),
        n_cells=len(cells),
    )


def loo_scores(fitter, cells: list[Cell], *, label: str) -> Scores:
    """Leave-one-cell-out scores, refitting the surface in every fold."""
    preds = []
    for i in range(len(cells)):
        train = [c for j, c in enumerate(cells) if j != i]
        preds.append(
            fitter(train).predict(df=cells[i].df, auc=cells[i].auc, n=cells[i].n)
        )
    pred = np.array(preds)
    obs = np.array([c.cov for c in cells])
    k = np.array([c.successes for c in cells], float)
    m = np.array([c.reps for c in cells], float)
    inside = sum(
        wilson_ci(c.cov, c.reps)[0] <= p <= wilson_ci(c.cov, c.reps)[1]
        for c, p in zip(cells, pred, strict=True)
    )
    return Scores(
        label=label,
        dev_per_rep=binomial_deviance(successes=k, trials=m, p=pred) / m.sum(),
        rmse=float(np.sqrt(np.mean((pred - obs) ** 2))),
        mae=float(np.mean(np.abs(pred - obs))),
        bias=float(np.mean(pred - obs)),
        worst_optimistic=float(np.max(pred - obs)),
        inside_ci=int(inside),
        n_cells=len(cells),
    )


def monotonicity_report(model, *, grid: int = 12) -> dict[str, float]:
    """Fraction of a covariate grid where the fitted surface is monotone.

    The tail-mass mechanism implies coverage nondecreasing in n and df and
    nonincreasing in AUC. The pre-registered fit imposes this; the flexible fits
    do not, so the realized fractions are a diagnostic on them.
    """
    dfs = np.exp(np.linspace(np.log(1.1), np.log(30.0), grid))
    aucs = norm.cdf(np.linspace(norm.ppf(0.55), norm.ppf(0.99), grid))
    ns = np.exp(np.linspace(np.log(LHS_N_BOUNDS[0]), np.log(LHS_N_BOUNDS[1]), grid))
    cube = np.array(
        [[[model.predict(df=d, auc=a, n=n) for n in ns] for a in aucs] for d in dfs]
    )
    return {
        "mono_n": float(np.mean(np.diff(cube, axis=2) >= -1e-9)),
        "mono_df": float(np.mean(np.diff(cube, axis=0) >= -1e-9)),
        "mono_auc": float(np.mean(np.diff(cube, axis=1) <= 1e-9)),
    }


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------


def load_cells(out_root: Path) -> tuple[list[Cell], list[Cell]]:
    """Load the LHS training cells and the student-t anchor holdout."""
    from followup_runs import _cell_row, _load_summaries

    sub = out_root / "boundary"
    rows = [_cell_row(s, sub) for s in _load_summaries(sub)]
    lhs, anchors = [], []
    for w in rows:
        meta = w["shape_meta"]
        if meta.get("family") != "student_t":
            continue
        cell = Cell(
            df=meta["df"],
            auc=meta["auc"],
            n=w["n0"],
            cov=w["cov"],
            reps=w["reps"],
            name=w["name"],
        )
        (lhs if w["arm"] == "followup_boundary_lhs" else anchors).append(cell)
    return lhs, sorted(anchors, key=lambda c: (c.df, c.auc, c.n))


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _fmt_n(value: float) -> str:
    if value <= 0:
        return f"<{LHS_N_BOUNDS[0]}"
    if not np.isfinite(value) or value > LHS_N_BOUNDS[1]:
        return f">{LHS_N_BOUNDS[1]}"
    return f"{value:.0f}"


def build_report(*, out_root: Path, gp_restarts: int) -> str:
    """Fit all three surfaces and render the comparison as markdown."""
    lhs, anchors = load_cells(out_root)
    lines = [
        "# Boundary surface — thin plate and Gaussian process refits",
        "",
        f"*Training: {len(lhs)} LHS cells x {lhs[0].reps} reps. "
        f"Holdout: {len(anchors)} classification-grade student-t anchors "
        f"({sum(c.reps for c in anchors):,} reps), used in no fit. "
        "Estimand: C = 1 coverage at alpha = .05; bar = .94.*",
        "",
    ]

    models = {
        "logistic-linear": fit_linear(lhs),
        "tprs": fit_tprs(lhs),
        "gp": fit_gp(lhs, restarts=gp_restarts),
    }
    fitters = {
        "logistic-linear": fit_linear,
        "tprs": fit_tprs,
        "gp": lambda cs: fit_gp(cs, restarts=1),
    }

    lines.extend(["## Fitted models", ""])
    lin = models["logistic-linear"].beta
    lines.append(
        f"- **logistic-linear**: b = ({lin[0]:.2f}, {lin[1]:.3f}, {lin[2]:.3f}, "
        f"{lin[3]:.3f}) on (1, log n, log df, probit AUC), sign-constrained."
    )
    t = models["tprs"]
    lines.append(
        f"- **tprs**: rank {len(t.basis.penalty) + t.basis.n_null} thin plate basis, "
        f"lambda = {t.lam:.3g} by UBRE, effective df = {t.edf:.1f}."
    )
    g = models["gp"]
    lines.append(
        f"- **gp**: ARD Matern-5/2, lengthscales (log n, log df, probit AUC) = "
        f"({g.lengthscales[0]:.2f}, {g.lengthscales[1]:.2f}, {g.lengthscales[2]:.2f}) "
        f"in SD units, amplitude {g.amplitude:.2f}, log marginal likelihood "
        f"{g.extras['log_marginal']:.1f}."
    )
    lines.append("")

    lines.extend(
        [
            "## Holdout: the 11 anchors",
            "",
            "Deviance per replicate is the proper score (lower is better). "
            "`worst opt.` is the largest overstatement of measured coverage — "
            "the anti-conservative direction that matters for routing.",
            "",
            "| model | dev/rep | RMSE | MAE | bias | worst opt. | in CI |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    holdout = {k: score(m, anchors, label=k) for k, m in models.items()}
    for s in holdout.values():
        lines.append(
            f"| {s.label} | {s.dev_per_rep:.4f} | {s.rmse:.3f} | {s.mae:.3f} | "
            f"{s.bias:+.3f} | {s.worst_optimistic:+.3f} | "
            f"{s.inside_ci}/{s.n_cells} |"
        )
    lines.append("")

    lines.extend(
        [
            "### Per-anchor predictions",
            "",
            "| anchor | measured [Wilson 95%] | " + " | ".join(models) + " |",
            "|---|---|" + "---|" * len(models),
        ]
    )
    for c in anchors:
        lo, hi = wilson_ci(c.cov, c.reps)
        cols = []
        for m in models.values():
            p = m.predict(df=c.df, auc=c.auc, n=c.n)
            mark = "" if lo <= p <= hi else ("!" if p > hi else "~")
            cols.append(f"{p:.3f}{mark}")
        lines.append(
            f"| df {c.df:g} / AUC {c.auc:g} / n {c.n} | "
            f"{c.cov:.3f} [{lo:.3f}, {hi:.3f}] | " + " | ".join(cols) + " |"
        )
    lines.extend(
        [
            "",
            "`!` = optimistic and outside the interval (unsafe direction); "
            "`~` = pessimistic and outside.",
            "",
        ]
    )

    lines.extend(
        [
            "## Leave-one-cell-out on the LHS sweep",
            "",
            "Interpolation quality inside the design, refitting each fold.",
            "",
            "| model | dev/rep | RMSE | MAE | bias | worst opt. | in CI |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for key, fitter in fitters.items():
        s = loo_scores(fitter, lhs, label=key)
        lines.append(
            f"| {s.label} | {s.dev_per_rep:.4f} | {s.rmse:.3f} | {s.mae:.3f} | "
            f"{s.bias:+.3f} | {s.worst_optimistic:+.3f} | "
            f"{s.inside_ci}/{s.n_cells} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Monotonicity (imposed on the baseline, diagnostic on the rest)",
            "",
            "| model | in n | in df | in AUC |",
            "|---|---|---|---|",
        ]
    )
    for key, m in models.items():
        mono = monotonicity_report(m)
        lines.append(
            f"| {key} | {mono['mono_n']:.0%} | {mono['mono_df']:.0%} | "
            f"{mono['mono_auc']:.0%} |"
        )
    lines.append("")

    aucs = (0.90, 0.95, 0.99)
    lines.extend(
        [
            "## Boundary contour n*(df, AUC) at the .94 bar",
            "",
            "GP columns give the posterior mean and, in brackets, the "
            f"{CONS_Q:.0%} posterior quantile — the conservative read the "
            "routing decision would use.",
            "",
            "| df | "
            + " | ".join(f"lin .{int(a * 100)}" for a in aucs)
            + " | "
            + " | ".join(f"tprs .{int(a * 100)}" for a in aucs)
            + " | "
            + " | ".join(f"gp .{int(a * 100)}" for a in aucs)
            + " |",
            "|---" * (1 + 3 * len(aucs)) + "|",
        ]
    )
    for df in (1.1, 1.5, 2.0, 3.0, 30.0):
        cells_out = [
            _fmt_n(n_star(models["logistic-linear"], df=df, auc=a)) for a in aucs
        ]
        cells_out += [_fmt_n(n_star(models["tprs"], df=df, auc=a)) for a in aucs]
        cells_out += [
            f"{_fmt_n(n_star(models['gp'], df=df, auc=a))} "
            f"[{_fmt_n(n_star(models['gp'], df=df, auc=a, quantile=CONS_Q))}]"
            for a in aucs
        ]
        lines.append(f"| {df:g} | " + " | ".join(cells_out) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gp-restarts", type=int, default=4)
    parser.add_argument(
        "--write", type=Path, default=None, help="also write the report to this path"
    )
    args = parser.parse_args()

    register_followup_shapes()
    report = build_report(out_root=args.out, gp_restarts=args.gp_restarts)
    print(report)
    if args.write is not None:
        args.write.write_text(report)
        print(f"\n[report] written to {args.write}")


if __name__ == "__main__":
    main()
