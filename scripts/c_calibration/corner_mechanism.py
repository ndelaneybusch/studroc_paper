"""Corner mechanism of the C = 1 fiducial band.

Theory: ``stats/fiducial_band_theory.md`` section 7.4 (2026-09-02). The cloud's
sorted-uniform within-gap spreading is an implicit *linear-ROC* assumption across
each gap between consecutive same-class observations. At a convex (heavy-tail)
corner that assumption fails in the anti-conservative direction. This module
implements

* the left-corner approximation and the large-k right-corner approximation;
* a resolution correction that retains the finite-k within-gap order-statistic
  law and removes the large-k approximation at the first grid points;
* a Poissonized simulation of the fiducial polyline restricted to the
  outermost grid points at each FPR end, with no cloud interior
  (``end_sim`` / ``run_cell``);
* a ground-truth check that runs the production band (``real_band_check``);
* the worst-case-over-df router table on an (AUC, n) grid (``router_table``);
* a comparison of both predictors with the 257 follow-up cells (``compare``).

Usage from the repository root::

    uv run python scripts/c_calibration/corner_mechanism.py <command>

Commands are ``cells``, ``closed``, ``simulate``, ``real``, ``sliver``, ``router``, and
``compare``. Run the module without a command to see their argument syntax.

All three predictors take the true ROC as input; they are tools for predicting
untested cells and for understanding the wedge, not runtime routers.
"""

import json
import sys
from collections.abc import Callable
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats
from scipy.optimize import brentq
from scipy.special import gammaln, kve

sys.path.insert(0, "src")
from studroc_paper.datagen.roc_to_dgp import StudentTSolver

_solver = StudentTSolver()


@cache
def t_shape(
    *, df: float, auc: float
) -> tuple[float, Callable, Callable, Callable, Callable]:
    """Construct a shifted Student-t ROC and its endpoint transforms.

    Args:
        df: Student-t degrees of freedom.
        auc: Target population AUC.

    Returns:
        Shift, ROC, inverse ROC, reflected endpoint ROC, and its inverse.

    Raises:
        ValueError: If the numerical AUC solver does not return a finite shift.
    """
    delta = _solver.solve(df=df, target_auc=auc)
    if not np.isfinite(delta):
        raise ValueError(f"No finite shift for df={df} and auc={auc}.")
    distribution = stats.t(df)

    def roc(t: np.ndarray | float) -> np.ndarray | float:
        """Evaluate the population ROC."""
        return distribution.sf(distribution.isf(np.clip(t, 1e-300, 1.0)) - delta)

    def inverse_roc(y: np.ndarray | float) -> np.ndarray | float:
        """Evaluate the inverse population ROC."""
        return distribution.sf(distribution.isf(np.clip(y, 1e-300, 1.0)) + delta)

    def reflected_roc(s: np.ndarray | float) -> np.ndarray | float:
        """Evaluate one minus the ROC at one minus an endpoint depth."""
        return distribution.cdf(distribution.ppf(np.clip(s, 1e-300, 1.0)) - delta)

    def inverse_reflected_roc(y: np.ndarray | float) -> np.ndarray | float:
        """Evaluate the inverse reflected endpoint ROC."""
        return distribution.cdf(distribution.ppf(np.clip(y, 1e-300, 1.0)) + delta)

    return delta, roc, inverse_roc, reflected_roc, inverse_reflected_roc


def _within_gap(
    *,
    base: np.ndarray,
    mass: np.ndarray,
    count_in_gap: int,
    rng: np.random.Generator,
    draws: int,
) -> np.ndarray:
    """Draw sorted-uniform coordinates within one same-class gap.

    Args:
        base: Per-draw lower endpoints.
        mass: Per-draw gap masses.
        count_in_gap: Number of other-class observations in the gap.
        rng: Random number generator.
        draws: Number of fiducial draws.

    Returns:
        Array with shape ``(draws, count_in_gap)``.
    """
    if count_in_gap == 0:
        return np.empty((draws, 0))
    u = np.sort(rng.random((draws, count_in_gap)), axis=1)
    return base[:, None] + u * mass[:, None]


def end_sim(
    *,
    end: str,
    roc: Callable,
    inverse_roc: Callable,
    reflected_roc: Callable,
    inverse_reflected_roc: Callable,
    n0: int,
    n1: int,
    ell: float,
    grid_size: int,
    repetitions: int,
    draws: int,
    rng: np.random.Generator,
    device: str = "cpu",
) -> dict[str, Any]:
    """Simulate fiducial failures near one ROC endpoint.

    Args:
        end: Endpoint to simulate, either ``"left"`` or ``"right"``.
        roc: Population ROC callable.
        inverse_roc: Inverse population ROC callable.
        reflected_roc: Reflected endpoint ROC callable.
        inverse_reflected_roc: Inverse reflected endpoint ROC callable.
        n0: Negative-class sample size.
        n1: Positive-class sample size.
        ell: Pointwise fiducial tail probability.
        grid_size: Number of endpoint-adjacent grid points to retain.
        repetitions: Number of independent datasets.
        draws: Number of fiducial draws per dataset.
        rng: Random number generator.
        device: Torch device used for interpolation.

    Returns:
        Endpoint miss rates, miss depths, and per-grid lower-miss rates.

    Raises:
        ValueError: If ``end`` is not ``"left"`` or ``"right"``.
    """
    if end not in {"left", "right"}:
        raise ValueError(f"Unknown endpoint: {end!r}.")
    if end == "right":
        pos_depth_of = inverse_reflected_roc
        truth_y = reflected_roc(np.arange(1, grid_size + 1) / n0)
    else:
        pos_depth_of = inverse_roc
        truth_y = roc(np.arange(1, grid_size + 1) / n0)
    grid = np.arange(1, grid_size + 1) / n0
    low_miss = 0
    up_miss = 0
    depths = []
    grid_low = np.zeros(grid_size)
    n_eval = 0
    for _ in range(repetitions):
        # --- data: outermost K+6 negatives and the positives beyond the last of them
        negative_count = grid_size + 6
        neg_d = np.cumsum(rng.exponential(size=negative_count)) / (n0 + 1)
        pos_cum = np.cumsum(rng.exponential(size=n1)) / (n1 + 1)
        pos_d = pos_depth_of(pos_cum)
        J = int(
            np.searchsorted(pos_d, neg_d[-1])
        )  # positives with depth < last negative
        pos_d = pos_d[:J]
        # merged sequence by depth: labels 0=neg, 1=pos
        d_all = np.concatenate([neg_d, pos_d])
        lab = np.concatenate([np.zeros(negative_count, int), np.ones(J, int)])
        order = np.argsort(d_all, kind="stable")
        lab = lab[order]
        n_nodes = negative_count + J
        # counts: for each node, number of own-class elements up to and including it
        cneg = np.cumsum(lab == 0)
        cpos = np.cumsum(lab == 1)
        # --- fiducial draws
        # Negatives have Gamma coordinates; positives spread within F-gaps.
        Fneg = np.cumsum(rng.exponential(size=(draws, negative_count)), axis=1) / (
            n0 + 1
        )
        Gpos = (
            np.cumsum(rng.exponential(size=(draws, J)), axis=1) / (n1 + 1)
            if J > 0
            else np.empty((draws, 0))
        )
        X = np.empty((draws, n_nodes))
        Y = np.empty((draws, n_nodes))
        is_neg = lab == 0
        X[:, is_neg] = Fneg
        Y[:, ~is_neg] = Gpos
        # A positive's F-gap index is the number of preceding negatives.
        # group positives by gap
        pos_idx = np.where(~is_neg)[0]
        gaps = cneg[pos_idx]  # 0..Kn
        for g in np.unique(gaps):
            sel = pos_idx[gaps == g]
            base = Fneg[:, g - 1] if g > 0 else np.zeros(draws)
            top = (
                Fneg[:, g]
                if g < negative_count
                else base + rng.exponential(size=draws) / (n0 + 1)
            )
            X[:, sel] = _within_gap(
                base=base, mass=top - base, count_in_gap=len(sel), rng=rng, draws=draws
            )
        # negatives' Y: within G-gaps. gap index = cpos at that node
        neg_idx = np.where(is_neg)[0]
        ggaps = cpos[neg_idx]  # 0..J
        # Negatives beyond the zone share the open final G-gap, which runs to
        # the next positive outside the zone.
        next_pos_depth = pos_depth_of(pos_cum[J]) if J < n1 else 1.0
        k_extra = int(rng.poisson(max(n0 * (next_pos_depth - neg_d[-1]), 0.0)))
        for g in np.unique(ggaps):
            sel = neg_idx[ggaps == g]
            base = Gpos[:, g - 1] if g > 0 else np.zeros(draws)
            top = Gpos[:, g] if g < J else base + rng.exponential(size=draws) / (n1 + 1)
            if g == J and k_extra > 0:
                full = _within_gap(
                    base=base,
                    mass=top - base,
                    count_in_gap=len(sel) + k_extra,
                    rng=rng,
                    draws=draws,
                )
                Y[:, sel] = full[:, : len(sel)]
            else:
                Y[:, sel] = _within_gap(
                    base=base,
                    mass=top - base,
                    count_in_gap=len(sel),
                    rng=rng,
                    draws=draws,
                )
        # prepend the corner node (0,0)
        X = np.concatenate([np.zeros((draws, 1)), X], axis=1)
        Y = np.concatenate([np.zeros((draws, 1)), Y], axis=1)
        # interpolate at grid depths (only where grid < last node)
        xt = torch.as_tensor(X, device=device)
        yt = torch.as_tensor(Y, device=device)
        tq = torch.as_tensor(grid, device=device).expand(draws, -1).contiguous()
        idx = torch.searchsorted(xt.contiguous(), tq, right=True).clamp(1, n_nodes)
        x1 = torch.gather(xt, 1, idx - 1)
        x2 = torch.gather(xt, 1, idx)
        y1 = torch.gather(yt, 1, idx - 1)
        y2 = torch.gather(yt, 1, idx)
        frac = ((tq - x1) / (x2 - x1).clamp(min=1e-300)).clamp(0, 1)
        yf = (y1 + frac * (y2 - y1)).cpu().numpy()  # (D, K)
        valid = grid < np.min(X[:, -1])  # grid points inside every draw's node range
        if end == "right":
            p_low = np.mean(
                yf[:, valid] > truth_y[valid], axis=0
            )  # fiducial residual larger => TPR lower
            p_up = np.mean(yf[:, valid] < truth_y[valid], axis=0)
            # CP upper allowance: count khat = n1 - (#positives with depth < k/n0)
            npos_below = np.searchsorted(pos_d, grid[valid])
            khat = n1 - npos_below
            cp_up = np.where(
                khat < n1,
                stats.beta.ppf(
                    1 - ell, np.minimum(khat + 1, n1), np.maximum(n1 - khat, 1e-12)
                ),
                1.0,
            )
            truth_tpr = 1 - truth_y[valid]
            up_hit = (p_up < ell) & (truth_tpr > cp_up)
            lo_hit = p_low < ell
            # Lower miss depth on the reflected TPR scale.
            if lo_hit.any():
                q = np.quantile(yf[:, valid][:, lo_hit], 1 - ell, axis=0)
                depths.append(float(np.max(truth_y[valid][lo_hit] - q)))
        else:
            p_low = np.mean(
                yf[:, valid] < truth_y[valid], axis=0
            )  # fiducial TPR lower than truth
            p_up = np.mean(yf[:, valid] > truth_y[valid], axis=0)
            npos_above = np.searchsorted(pos_d, grid[valid])
            khat = npos_above
            cp_up = np.where(
                khat < n1,
                stats.beta.ppf(
                    1 - ell, np.minimum(khat + 1, n1), np.maximum(n1 - khat, 1e-12)
                ),
                1.0,
            )
            up_hit = (p_up < ell) & (truth_y[valid] > cp_up)
            lo_hit = p_low < ell
            if lo_hit.any():
                q = np.quantile(yf[:, valid][:, lo_hit], ell, axis=0)
                depths.append(float(np.max(q - truth_y[valid][lo_hit])))
        grid_low[np.where(valid)[0][lo_hit]] += 1
        n_eval += 1
        low_miss += int(lo_hit.any())
        up_miss += int(up_hit.any())
    return dict(
        low=low_miss / repetitions,
        up=up_miss / repetitions,
        depth_mean=float(np.mean(depths)) if depths else 0.0,
        depth_max=float(np.max(depths)) if depths else 0.0,
        grid_low=(grid_low / max(n_eval, 1)).tolist(),
    )


def run_cell(
    *,
    df: float,
    auc: float,
    n0: int,
    n1: int,
    ell: float,
    repetitions: int = 200,
    draws: int = 4000,
    right_grid_size: int | None = None,
    left_grid_size: int = 25,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the endpoint simulator for one shifted Student-t cell.

    Args:
        df: Student-t degrees of freedom.
        auc: Population AUC.
        n0: Negative-class sample size.
        n1: Positive-class sample size.
        ell: Pointwise fiducial tail probability.
        repetitions: Number of independent datasets.
        draws: Number of fiducial draws per dataset.
        right_grid_size: Number of right-end grid points, selected from ``n0``
            when omitted.
        left_grid_size: Number of left-end grid points.
        seed: Random seed.

    Returns:
        Shift and simulated left- and right-end failure summaries.
    """
    rng = np.random.default_rng(seed)
    delta, roc, inverse_roc, reflected_roc, inverse_reflected_roc = t_shape(
        df=df, auc=auc
    )
    right_grid_size = right_grid_size or int(min(n0 // 2, 250))
    common = {
        "roc": roc,
        "inverse_roc": inverse_roc,
        "reflected_roc": reflected_roc,
        "inverse_reflected_roc": inverse_reflected_roc,
        "n0": n0,
        "n1": n1,
        "ell": ell,
        "repetitions": repetitions,
        "draws": draws,
        "rng": rng,
    }
    right = end_sim(end="right", grid_size=right_grid_size, **common)
    left = end_sim(end="left", grid_size=min(left_grid_size, n0 // 4), **common)
    return dict(delta=delta, left=left, right=right)


DATA = Path("data/results/c_calibration_followup_20260830")


def extract_cells(*, out_path: Path) -> list[dict[str, Any]]:
    """Extract follow-up Student-t cells and their C=1 coverage records.

    Args:
        out_path: Destination JSON path.

    Returns:
        Extracted cell records.
    """
    rows = []
    for summary_path in sorted(DATA.glob("boundary*/*.summary.json")):
        if summary_path.name.endswith(".m3.summary.json"):
            continue
        with summary_path.open(encoding="utf-8") as file:
            d = json.load(file)
        c = d["meta"]["cell"]
        sm = c["shape_meta"]
        if sm.get("family") != "student_t":
            continue
        ref = [
            r
            for r in d["aggregate"]["ref_maps"]
            if r["label"] == "c1" and abs(r["alpha"] - 0.05) < 1e-9
        ]
        if not ref:
            continue
        r = ref[0]
        rows.append(
            dict(
                name=c["name"],
                df=sm["df"],
                auc=sm["auc"],
                true_auc=d["meta"].get("true_auc"),
                n0=c["n0"],
                n1=c["n1"],
                reps=d["aggregate"]["reps"],
                cov=r["coverage"],
                vlow=r.get("viol_low"),
                vhigh=r.get("viol_high"),
                depth=r.get("mean_miss_depth_missers"),
                jstar_c1=r.get("mean_j"),
                M=d["meta"]["m_draws"],
            )
        )
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=0)
    return rows


def ell_law(*, grid_points: int, alpha: float = 0.05) -> float:
    """Approximate the section-9 pointwise fiducial tail probability.

    Args:
        grid_points: Number of ROC grid points.
        alpha: Simultaneous noncoverage level.

    Returns:
        Approximate local tail probability.
    """
    return 9.7e-4 * (alpha / 0.05) ** 1.2 * (grid_points / 500) ** -0.27


def closed_forms(
    *, df: float, auc: float, n0: int, n1: int, ell: float
) -> dict[str, float]:
    """Compute the left and large-k right corner approximations.

    Args:
        df: Student-t degrees of freedom.
        auc: Population area under the ROC curve.
        n0: Negative-class sample size.
        n1: Positive-class sample size.
        ell: Realized pointwise tail probability of the fiducial tube.

    Returns:
        Corner scales and the two large-count approximations. ``right`` and
        ``total`` are retained as aliases for historical result files;
        ``right_large_k`` names the approximation explicitly.
    """
    delta, roc, inverse_roc, reflected_roc, inverse_reflected_roc = t_shape(
        df=df, auc=auc
    )
    log_tail = np.log(1 / ell)
    first_grid_tpr = roc(1.0 / n0)
    target_tpr = log_tail * first_grid_tpr
    left = float(np.exp(-n0 * inverse_roc(target_tpr))) if target_tpr < 1 else 0.0
    h0 = (
        float((log_tail / n0) / inverse_roc(target_tpr))
        if target_tpr < 1
        else float("nan")
    )
    p_star = float(n1 * reflected_roc(1.0 / n0))
    k_star = float(n0 * inverse_reflected_roc(1.0 / n1))
    k_crit = log_tail / p_star
    right = (
        float(np.exp(-n1 * reflected_roc(min(k_crit / n0, 1.0))))
        if k_crit < n0
        else 0.0
    )
    total = 1 - (1 - left) * (1 - right)
    return dict(
        left=left,
        right=right,
        right_large_k=right,
        total=total,
        total_large_k=total,
        h0=h0,
        h=1 / (k_star * p_star),
        k_star=k_star,
        p_star=p_star,
        k_crit=k_crit,
        N0=n0 * (1 - auc) / 2,
        N1=n1 * (1 - auc) / 2,
        delta=delta,
        Q=log_tail,
    )


def _log_product_survival(value: float, k: int) -> float:
    """Return log P(E * Z_k > value) for the finite-k correction.

    Here ``E`` is standard exponential and ``Z_k`` is Gamma with shape and
    rate ``k``. The exponentially scaled Bessel function keeps the expression
    stable for moderately large ``k``.

    Args:
        value: Positive product threshold.
        k: Positive integer grid distance from the endpoint.

    Returns:
        Log survival probability.
    """
    argument = 2.0 * np.sqrt(k * value)
    return float(
        np.log(2.0)
        + (k / 2.0) * np.log(k * value)
        - gammaln(k)
        + np.log(kve(k, argument))
        - argument
    )


@lru_cache(maxsize=32_768)
def product_quantile(k: int, ell_rounded: float) -> float:
    """Return q such that P(E * Z_k > q) equals ``ell_rounded``.

    Args:
        k: Positive integer grid distance from the endpoint.
        ell_rounded: Tail probability, rounded before calling to enable
            caching across cells with the same Monte Carlo design.

    Returns:
        Upper-tail product quantile. It is larger than ``log(1 / ell)`` for
        finite ``k`` and converges to that value as ``k`` increases.
    """
    target = np.log(ell_rounded)
    upper = max(10.0, 4.0 * np.log(1.0 / ell_rounded))
    while _log_product_survival(upper, k) > target:
        upper *= 2.0
    return float(
        brentq(lambda value: _log_product_survival(value, k) - target, 1e-10, upper)
    )


def resolution_corrected_right_risk(
    *, tau, n0: int, n1: int, ell: float, exact_k_max: int = 256
) -> dict[str, float | int]:
    """Approximate right-corner miss risk while retaining finite grid resolution.

    For the grid point ``k`` steps left of FPR one, the within-gap fraction
    contributes ``Z_k = Gamma(k, rate=k)`` at leading order. Replacing this
    factor by one gives the older large-k criterion and is badly optimistic
    about the cloud's uncertainty at the first few grid points. This routine
    uses the exact product quantile through ``exact_k_max`` and its limit
    ``Q = log(1 / ell)`` afterwards.

    Args:
        tau: Callable returning ``1 - R(1 - s)``.
        n0: Negative-class sample size.
        n1: Positive-class sample size.
        ell: Realized pointwise tail probability of the fiducial tube.
        exact_k_max: Largest grid distance receiving the finite-k product
            quantile. The asymptotic value is used beyond it.

    Returns:
        Mapping containing the right-corner risk, critical saturated-zone
        count, and the grid point that minimizes that critical count.
    """
    grid_indices = np.arange(1, n0 + 1)
    q_values = np.full(n0, np.log(1.0 / ell), dtype=float)
    exact_count = min(n0, exact_k_max)
    ell_key = round(float(ell), 12)
    q_values[:exact_count] = [
        product_quantile(k=k, ell_rounded=ell_key) for k in range(1, exact_count + 1)
    ]
    truth_depth = n1 * tau(grid_indices / n0)
    required_zone = np.maximum(
        grid_indices, q_values * grid_indices / np.maximum(truth_depth, 1e-300)
    )
    minimizing_index = int(np.argmin(required_zone))
    critical_count = min(int(np.ceil(required_zone[minimizing_index])), n0)
    risk = float(np.exp(-n1 * tau(critical_count / n0))) if critical_count < n0 else 0.0
    return {
        "right_finite_k": risk,
        "k_crit_finite": critical_count,
        "k_witness": int(grid_indices[minimizing_index]),
        "q_witness": float(q_values[minimizing_index]),
    }


def corner_risk(
    *, df: float, auc: float, n0: int, n1: int, ell: float
) -> dict[str, float | int]:
    """Compute the resolution-corrected corner risk score for one t-family cell.

    Args:
        df: Student-t degrees of freedom.
        auc: Population area under the ROC curve.
        n0: Negative-class sample size.
        n1: Positive-class sample size.
        ell: Realized pointwise tail probability of the fiducial tube.

    Returns:
        Large-k diagnostics augmented with the resolution-corrected right risk
        and the union risk score ``corner_risk``.
    """
    _, _, _, tau, _ = t_shape(df=df, auc=auc)
    result = closed_forms(df=df, auc=auc, n0=n0, n1=n1, ell=ell)
    finite = resolution_corrected_right_risk(tau=tau, n0=n0, n1=n1, ell=ell)
    result.update(finite)
    result["corner_risk"] = 1 - (1 - result["left"]) * (1 - result["right_finite_k"])
    return result


def real_band_check(
    *,
    df: float,
    auc: float,
    sample_size: int,
    draws: int,
    repetitions: int,
    seed: int = 12345,
) -> list[dict[str, Any]]:
    """Run the production band and record endpoint miss geometry.

    Args:
        df: Student-t degrees of freedom.
        auc: Population AUC.
        sample_size: Per-class sample size.
        draws: Fiducial draws per dataset.
        repetitions: Number of independent datasets.
        seed: Random seed.

    Returns:
        Per-dataset saturation counts and miss diagnostics.
    """
    from studroc_paper.methods.fiducial_band_rs import fiducial_band_rs

    _, roc, inverse_roc, _, _ = t_shape(df=df, auc=auc)
    rng = np.random.default_rng(seed)
    grid = np.arange(sample_size + 1) / sample_size
    truth = roc(grid)
    truth[0] = 0
    truth[-1] = 1
    rows = []
    for rep in range(repetitions):
        u = rng.random(sample_size)
        w = inverse_roc(rng.random(sample_size))
        y = np.concatenate([np.zeros(sample_size, int), np.ones(sample_size, int)])
        score = -np.concatenate([u, w])
        k_sat = int(np.sum(u > w.max()))
        p_below1 = int(np.sum(w > u.max()))
        _, lo, hi = fiducial_band_rs(
            y_true=y,
            y_score=score,
            alpha=0.05,
            n_draws=draws,
            trim_exponent=1.0,
            random_state=int(rng.integers(2**31)),
        )
        low_viol = truth < lo - 1e-12
        up_viol = truth > hi + 1e-12
        kk = np.where(low_viol)[0]
        rows.append(
            dict(
                rep=rep,
                k_sat=k_sat,
                p_below1=p_below1,
                miss_low=bool(low_viol.any()),
                miss_up=bool(up_viol.any()),
                low_k_from_top=(sample_size - kk).tolist(),
                max_low_depth=float(np.max(lo - truth)) if low_viol.any() else 0.0,
                up_k=np.where(up_viol)[0].tolist(),
            )
        )
        print(
            f"rep {rep:3d} k_sat={k_sat:5d} "
            f"miss_low={low_viol.any()} miss_up={up_viol.any()} "
            f"k_from_top={(sample_size - kk)[:8].tolist()} "
            f"n_viol={len(kk)} maxdepth={rows[-1]['max_low_depth']:.5f}",
            flush=True,
        )
    return rows


def router_table() -> None:
    """Print the worst-case resolution-corrected risk on an AUC-by-n grid."""
    import warnings

    warnings.filterwarnings("ignore")

    def _cf(*, df: float, auc: float, n0: int, n1: int, ell: float) -> tuple:
        """Return left, right, and combined corner risks for one cell."""
        c = corner_risk(df=df, auc=auc, n0=n0, n1=n1, ell=ell)
        return c["left"], c["right_finite_k"], c["corner_risk"]

    aucs = [
        0.80,
        0.85,
        0.88,
        0.90,
        0.92,
        0.94,
        0.95,
        0.96,
        0.97,
        0.975,
        0.98,
        0.985,
        0.99,
    ]
    ns = [
        100,
        200,
        300,
        500,
        700,
        1000,
        1500,
        2000,
        3000,
        5000,
        8000,
        12000,
        20000,
        50000,
    ]
    dfs = [1.1, 1.3, 1.6, 2, 2.5, 3, 4, 5, 6, 8, 10, 15, 20, 30]

    def worst(*, auc: float, n0: int, n1: int) -> tuple:
        """Maximize the corner-risk score over the configured shape grid."""
        best = (0, None, 0, 0)
        for df in dfs:
            try:
                left, right, total = _cf(
                    df=df, auc=auc, n0=n0, n1=n1, ell=ell_law(grid_points=n0 + 1)
                )
            except ValueError:
                continue
            if total > best[0]:
                best = (total, df, left, right)
        return best

    print(
        "Worst-case (over df in [1.1,30]) resolution-corrected corner risk, "
        "balanced n0=n1=n"
    )
    print("AUC\\n  " + "".join(f"{n:>9d}" for n in ns))
    table = {}
    for auc in aucs:
        row = f"{auc:.3f}  "
        for n in ns:
            total, df, left, right = worst(auc=auc, n0=n, n1=n)
            table[(auc, n)] = (total, df, left, right)
            row += f"{total:8.3f}{'*' if total > 0.05 else ' '}"
        print(row)
    print("\nworst df per cell:")
    print("AUC\\n  " + "".join(f"{n:>9d}" for n in ns))
    for auc in aucs:
        print(
            f"{auc:.3f}  "
            + "".join(
                f"{table[(auc, n)][1] if table[(auc, n)][1] else 0:9.1f}" for n in ns
            )
        )
    print("\nleft/right split at the worst df (L:R):")
    for auc in [0.90, 0.95, 0.975, 0.99]:
        print(
            f"{auc:.3f}  "
            + "".join(f"{table[(auc, n)][2]:.2f}:{table[(auc, n)][3]:.2f} " for n in ns)
        )
    # imbalance: fixed n_total? fixed minority? show n0 != n1 at AUC .975
    print("\nImbalance at AUC .975 (worst df): rows n0, cols n1")
    nn = [200, 500, 1000, 2000, 5000]
    print("n0\\n1 " + "".join(f"{n:>9d}" for n in nn))
    for n0 in nn:
        print(
            f"{n0:5d} "
            + "".join(f"{worst(auc=0.975, n0=n0, n1=n1)[0]:9.3f}" for n1 in nn)
        )
    print("\nImbalance at AUC .95 (worst df): rows n0, cols n1")
    for n0 in nn:
        print(
            f"{n0:5d} "
            + "".join(f"{worst(auc=0.95, n0=n0, n1=n1)[0]:9.3f}" for n1 in nn)
        )


def compare(*, closed_path: Path, simulation_paths: list[Path]) -> None:
    """Compare corner predictors with measured follow-up coverage.

    Args:
        closed_path: JSON file containing analytical scores.
        simulation_paths: JSON files containing endpoint simulation scores.
    """
    with closed_path.open(encoding="utf-8") as file:
        cells = json.load(file)
    cov = np.array([c["cov"] for c in cells])
    vlow = np.array([c["vlow"] for c in cells])
    fail = cov < 0.94
    tot = np.array([c.get("corner_risk", c["total"]) for c in cells])
    safe = tot <= 0.05
    print(
        f"corner score: corr(risk, vlow)={np.corrcoef(tot, vlow)[0, 1]:.3f}; "
        f"cells with risk<=.05: {np.sum(safe)} (fails {np.sum(fail & safe)}); "
        f">.05: {np.sum(tot > 0.05)} (fails {np.sum(fail & (tot > 0.05))})"
    )
    sim = []
    for path in simulation_paths:
        with path.open(encoding="utf-8") as file:
            sim += json.load(file)
    if sim:
        s_cov = np.array([s["cov"] for s in sim])
        s_vlow = np.array([s["vlow"] for s in sim])
        exc = np.array([1 - (1 - s["left_low"]) * (1 - s["right_low"]) for s in sim])
        base = 0.975
        pred = base * (1 - exc)
        res = s_cov - pred
        print(
            f"tail simulator ({len(sim)} cells): "
            f"corr(pred cov, cov)={np.corrcoef(pred, s_cov)[0, 1]:.3f}; "
            f"corr(excess, vlow)={np.corrcoef(exc, s_vlow)[0, 1]:.3f}; "
            f"RMSE={np.sqrt(np.mean(res**2)):.3f}; mean resid={res.mean():+.3f}; "
            f"|resid|>.05 in {np.sum(np.abs(res) > 0.05)} cells"
        )
        sf = s_cov < 0.94
        for thr in (0.02, 0.03, 0.05):
            m = exc > thr
            print(
                f"  sim excess > {thr}: {m.sum()} cells, "
                f"{np.sum(sf & m)} fails | <= : {(~m).sum()} cells, "
                f"{np.sum(sf & ~m)} fails"
            )



# ---------------------------------------------------------------------------
# Proposition 14 made concrete: a continuous DGP with any prescribed AUC on
# which the C = 1 band fails at the right end and M3 does not.
# ---------------------------------------------------------------------------


def make_sliver(auc: float, n0: int, c: float, s1: float):
    """A "sliver" ROC: tau(s) = 1 - R(1-s) equals c*s on [0, 1/n0] (positive mass c/n0
    in the extreme lower tail, likelihood ratio c there), is flat on [1/n0, s1] (no
    positive mass), and a concave binormal body on [0, 1 - s1] is scaled so that the
    total area is ``auc``. Returns (R, Rinv, body_auc)."""
    s0 = 1.0 / n0
    h = 1 - c * s0
    tail_area = (s1 - s0) * h + s0 * (1 - c * s0 / 2)
    auc_b = (auc - tail_area) / (h * (1 - s1))
    if not 0.5 < auc_b < 1:
        raise ValueError(f"body AUC {auc_b:.3f} not attainable for auc={auc}, s1={s1}")
    d = np.sqrt(2) * stats.norm.ppf(auc_b)

    def body(u):
        return stats.norm.cdf(d + stats.norm.ppf(np.clip(u, 1e-300, 1 - 1e-16)))

    def R(t):
        t = np.asarray(t, float); out = np.empty_like(t)
        m1 = t <= 1 - s1; m3 = t >= 1 - s0; m2 = ~m1 & ~m3
        out[m1] = h * body(t[m1] / (1 - s1)); out[m2] = h; out[m3] = 1 - c * (1 - t[m3])
        return out

    tt = np.unique(np.concatenate([np.linspace(0, 1, 200001), 1 - np.geomspace(1e-9, 2 * s1, 20000),
                                   np.geomspace(1e-9, .1, 20000)]))
    rr = np.maximum.accumulate(R(tt))
    return R, (lambda y: np.interp(y, rr, tt)), float(auc_b)


def sliver_check(auc: float, n: int, c: float, s1: float, reps: int, M: int, seed: int = 7) -> dict:
    """Production C = 1 band and M3 on ``reps`` datasets from the sliver DGP (balanced n)."""
    from studroc_paper.methods.fiducial_band_rs import fiducial_band_rs
    from studroc_paper.methods.m3_band_rs import m3_band_rs
    R, Rinv, auc_b = make_sliver(auc, n, c, s1)
    rng = np.random.default_rng(seed)
    grid = np.arange(n + 1) / n; truth = R(grid); truth[0] = 0; truth[-1] = 1
    tg = np.linspace(0, 1, 100001); auc_num = float(np.trapezoid(R(tg), tg))
    miss_f = miss_m = 0; ksat = []; miss_ksat = []
    for _ in range(reps):
        u = rng.random(n); w = Rinv(rng.random(n))
        y = np.concatenate([np.zeros(n, int), np.ones(n, int)]); score = -np.concatenate([u, w])
        k = int(np.sum(u > w.max())); ksat.append(k)
        _, lo, hi = fiducial_band_rs(y, score, alpha=0.05, n_draws=M, trim_exponent=1.0,
                                     random_state=int(rng.integers(2 ** 31)))
        mf = bool(np.any(truth < lo - 1e-12) or np.any(truth > hi + 1e-12)); miss_f += mf
        if mf:
            miss_ksat.append(k)
        _, lo2, hi2 = m3_band_rs(y, score, alpha=0.05)
        miss_m += bool(np.any(truth < lo2 - 1e-12) or np.any(truth > hi2 + 1e-12))
    out = dict(auc=auc, auc_numerical=auc_num, body_auc=auc_b, n0=n, n1=n, c=c, s1=s1, reps=reps, M=M,
               cov_c1=1 - miss_f / reps, cov_m3=1 - miss_m / reps, k_sat_median=float(np.median(ksat)),
               k_sat_of_missing=sorted(miss_ksat))
    print(f"AUC {auc} (num {auc_num:.4f}, body {auc_b:.3f}) n={n} c={c} s1={s1}: C=1 cov {out['cov_c1']:.3f}, "
          f"M3 cov {out['cov_m3']:.3f}, k_sat median {out['k_sat_median']:.0f}", flush=True)
    return out

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cmd = sys.argv[1]
    if cmd == "cells":
        extract_cells(out_path=Path(sys.argv[2]))
    elif cmd == "closed":
        with Path(sys.argv[2]).open(encoding="utf-8") as file:
            cells = json.load(file)
        out = []
        for c in cells:
            ell = c["jstar_c1"] / (c["M"] + 1)
            out.append(
                dict(
                    c,
                    ell=ell,
                    **corner_risk(
                        df=c["df"], auc=c["auc"], n0=c["n0"], n1=c["n1"], ell=ell
                    ),
                )
            )
        output_path = Path(sys.argv[3])
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(out, file, indent=2)
        compare(closed_path=output_path, simulation_paths=[])
    elif cmd == "simulate":
        with Path(sys.argv[2]).open(encoding="utf-8") as file:
            cells = json.load(file)
        shard, nsh = (
            (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 1)
        )
        reps, D = (
            (int(sys.argv[6]), int(sys.argv[7])) if len(sys.argv) > 7 else (150, 2500)
        )
        out = []
        for c in cells[shard::nsh]:
            ell = c["jstar_c1"] / (c["M"] + 1)
            r = run_cell(
                df=c["df"],
                auc=c["auc"],
                n0=c["n0"],
                n1=c["n1"],
                ell=ell,
                repetitions=reps,
                draws=D,
                right_grid_size=int(min(c["n0"] // 2, 200)),
                left_grid_size=20,
                seed=shard,
            )
            out.append(
                dict(
                    c,
                    ell=ell,
                    delta=r["delta"],
                    left_low=r["left"]["low"],
                    left_up=r["left"]["up"],
                    right_low=r["right"]["low"],
                    right_up=r["right"]["up"],
                    right_depth=r["right"]["depth_mean"],
                    left_depth=r["left"]["depth_mean"],
                    right_grid=r["right"]["grid_low"],
                    left_grid=r["left"]["grid_low"],
                )
            )
            with Path(sys.argv[3]).open("w", encoding="utf-8") as file:
                json.dump(out, file, indent=2)
            print(c["name"], c["cov"], r["left"]["low"], r["right"]["low"], flush=True)
    elif cmd == "real":
        rows = real_band_check(
            df=float(sys.argv[2]),
            auc=float(sys.argv[3]),
            sample_size=int(sys.argv[4]),
            draws=int(sys.argv[5]),
            repetitions=int(sys.argv[6]),
        )
        output_path = Path(f"real_band_{sys.argv[2]}_{sys.argv[3]}_{sys.argv[4]}.json")
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(rows, file, indent=2)
    elif cmd == "sliver":
        # sliver <n> <reps> <M> [out.json]: AUC in {.6,.8,.95}; see theory doc 7.4(f)
        n, reps, M = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        res = [sliver_check(a, n, c, s1, reps, M) for a, c, s1 in ((0.60, 1.0, 0.12), (0.80, 0.8, 0.25), (0.95, 0.8, 0.25))]
        if len(sys.argv) > 5:
            json.dump(res, open(sys.argv[5], "w"), indent=1)
    elif cmd == "router":
        router_table()
    elif cmd == "compare":
        compare(
            closed_path=Path(sys.argv[2]),
            simulation_paths=[Path(path) for path in sys.argv[3:]],
        )
