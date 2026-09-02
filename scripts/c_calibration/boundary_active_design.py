"""Greedy level-set design for the C = 1 boundary surface (spec follow-up item 1b).

The 95-cell LHS sweep spends most of its budget where the answer was never in
doubt: 90 of its cells sit comfortably above the .94 bar, and the coverage cliff
in the heavy-tail/high-AUC corner rests on a single cell (df 2.62, AUC .987,
n = 160, coverage .696). Leaving that one cell out moves the GP's prediction
there from .70 to .96, so no smooth fitted to this design can resolve the
boundary — the binding constraint is where the cells are, not which smoother is
used.

This module places a new batch where the boundary actually is. Candidates are
scored by the *straddle* acquisition of Bryan et al. (2005), the standard
criterion for level-set estimation:

    a(x) = z * sd(x) - |mean(x) - logit(bar)|

on the latent (logit) scale, so a cell is attractive when the surface is
uncertain there *and* plausibly near the .94 decision boundary. Points deep in
the safe interior score badly however uncertain they are, which is exactly the
waste the LHS design incurred.

Selection is greedy and batch-aware: after each pick the joint posterior
covariance is downdated by that cell's planned Laplace weight, so the batch
spreads along the contour instead of collapsing onto one point. Scores are
divided by a calibrated cost so the batch is chosen per CPU-second rather than
per cell — replicate cost rises roughly fifteenfold from n = 100 to n = 2500,
and an uncost-aware batch would spend most of the budget on a handful of large-n
cells.

The design model is fitted to the LHS sweep *and* the classification-grade
anchors, since the anchors carry the strongest evidence about the cliff. That
choice spends the anchors as a design input: surfaces fitted after this batch
should be scored on the new cells (and on item 5's confirmation cells), not on
the anchors, which are no longer an untouched holdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm, qmc

sys.path.insert(0, str(Path(__file__).resolve().parent))

from boundary_surface_fits import Cell as MeasuredCell  # noqa: E402
from boundary_surface_fits import fit_gp, load_cells  # noqa: E402
from design import Cell  # noqa: E402
from followup_runs import (  # noqa: E402
    BAR_POINT,
    DEFAULT_OUT,
    LHS_AUC_BOUNDS,
    LHS_DF_BOUNDS,
    _cell,
    register_followup_shapes,
)
from runner import run_cell  # noqa: E402
from shapes import ShapeSpec, make_t_shape, shape_registry  # noqa: E402

ACTIVE_SEED = 20260901
ACTIVE_REPS = 250
N_BOUNDS = (100, 3_000)  # mild upward extension of the LHS box's 2,500
N_CANDIDATES = 3_000
STRADDLE_Z = 1.96
# Scores are divided by cost**COST_EXPONENT: 0 ignores cost, 1 makes the
# criterion purely per-core-second. Under a fixed budget 1 is the coherent
# choice, and it measurably buys more cells on the contour than 0.5 does.
COST_EXPONENT = 1.0
BUDGET_CPU_HOURS = 2.0

# Replicate cost in core-seconds, least-squares fitted to the measured runtimes
# of the 95 LHS cells (12 concurrent cores). Wall cost per replicate is
# essentially affine in n at fixed alpha grid.
COST_INTERCEPT = 0.0117
COST_SLOPE = 1.9416e-3


def rep_cost_core_seconds(n: int) -> float:
    """Calibrated core-seconds per replicate at per-class size ``n``."""
    return COST_INTERCEPT + COST_SLOPE * n


@dataclass(frozen=True)
class Candidate:
    """One proposed cell in the design space."""

    df: float
    auc: float
    n: int

    @property
    def cost_core_seconds(self) -> float:
        """Cost of running this candidate at :data:`ACTIVE_REPS` replicates."""
        return ACTIVE_REPS * rep_cost_core_seconds(self.n)


def auc_cap(df: float) -> float:
    """Largest AUC the DGP mapper can reach at ``df`` (its shift cap is 20)."""
    from studroc_paper.datagen.roc_to_dgp import StudentTSolver

    return float(StudentTSolver()._compute_auc(df, 20.0))


def candidate_pool(
    *, seed: int = ACTIVE_SEED, size: int = N_CANDIDATES
) -> list[Candidate]:
    """A Sobol pool over (log df, probit AUC, log n), achievable points only.

    Unachievable (df, AUC) combinations are dropped exactly as the paper's LHS
    pipeline drops them, so the pool stays inside the suite's design space.
    """
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    u = sampler.random(size)
    log_df = np.log(LHS_DF_BOUNDS[0]) + u[:, 0] * (
        np.log(LHS_DF_BOUNDS[1]) - np.log(LHS_DF_BOUNDS[0])
    )
    z_lo, z_hi = norm.ppf(LHS_AUC_BOUNDS[0]), norm.ppf(LHS_AUC_BOUNDS[1])
    auc = norm.cdf(z_lo + u[:, 1] * (z_hi - z_lo))
    log_n = np.log(N_BOUNDS[0]) + u[:, 2] * (np.log(N_BOUNDS[1]) - np.log(N_BOUNDS[0]))

    caps: dict[float, float] = {}
    out = []
    for i in range(size):
        df = round(float(np.exp(log_df[i])), 4)
        a = round(float(auc[i]), 4)
        if df not in caps:
            caps[df] = auc_cap(df)
        if a > caps[df] - 0.002:
            continue
        out.append(Candidate(df=df, auc=a, n=int(round(np.exp(log_n[i])))))
    return out


def _covariates(cands: list[Candidate]) -> np.ndarray:
    return np.column_stack(
        [
            np.log([c.n for c in cands]),
            np.log([c.df for c in cands]),
            norm.ppf([c.auc for c in cands]),
        ]
    )


@dataclass
class Selection:
    """The chosen batch and its accounting."""

    chosen: list[Candidate]
    scores: list[float]
    cost_core_hours: float
    n_pool: int


def select_batch(
    model,
    pool: list[Candidate],
    *,
    budget_cpu_hours: float = BUDGET_CPU_HOURS,
    bar: float = BAR_POINT,
    z: float = STRADDLE_Z,
    cost_exponent: float = COST_EXPONENT,
) -> Selection:
    """Greedily pick cells by cost-weighted straddle under a CPU budget.

    Args:
        model: A fitted GP exposing ``latent`` and ``latent_cov``.
        pool: Candidate cells to choose from.
        budget_cpu_hours: Total core-hour budget for the batch.
        bar: Coverage level whose contour is being located.
        z: Width multiplier in the straddle acquisition.
        cost_exponent: Scores are divided by ``cost ** cost_exponent``; 0
            ignores cost, 1 makes the criterion purely per-core-second.

    Returns:
        The selected batch, in selection order.
    """
    x = _covariates(pool)
    mu, _ = model.latent(x)
    cov = model.latent_cov(x)
    target = logit(bar)
    cost = np.array([c.cost_core_seconds for c in pool])
    budget = budget_cpu_hours * 3600.0

    chosen: list[int] = []
    scores: list[float] = []
    spent = 0.0
    taken = np.zeros(len(pool), dtype=bool)
    while True:
        sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
        straddle = z * sd - np.abs(mu - target)
        score = straddle / cost**cost_exponent
        score[taken] = -np.inf
        score[cost > budget - spent] = -np.inf
        j = int(np.argmax(score))
        if not np.isfinite(score[j]):
            break
        chosen.append(j)
        scores.append(float(score[j]))
        spent += cost[j]
        taken[j] = True
        # Downdate the joint posterior by this cell's planned Laplace weight.
        # The reduction is value-independent for a GP, which is what lets the
        # batch be chosen before any of it is run.
        p = float(expit(mu[j]))
        noise = 1.0 / max(ACTIVE_REPS * p * (1.0 - p), 1e-6)
        col = cov[:, j]
        cov = cov - np.outer(col, col) / (cov[j, j] + noise)
    return Selection(
        chosen=[pool[j] for j in chosen],
        scores=scores,
        cost_core_hours=spent / 3600.0,
        n_pool=len(pool),
    )


def register_active_shapes(chosen: list[Candidate]) -> list[str]:
    """Register a student-t shape per selected cell; returns the shape names."""
    registry = shape_registry()
    names = []
    for i, c in enumerate(chosen):
        name = f"tact{i:03d}"
        registry.setdefault(
            name,
            ShapeSpec(
                name=name,
                role="followup",
                build=(lambda auc=c.auc, df=c.df: make_t_shape(auc, df=df)),
                meta={
                    "family": "student_t",
                    "auc": c.auc,
                    "df": c.df,
                    "active_seed": ACTIVE_SEED,
                    "active_index": i,
                },
            ),
        )
        names.append(name)
    return names


def build_cells(chosen: list[Candidate], names: list[str]) -> list[Cell]:
    """Runner cells for the selected batch (fixed replicates, no top-up)."""
    return [
        _cell(
            name=f"boundary_act--{name}--n{c.n}",
            stage="S",
            arm="followup_boundary_active",
            shape=name,
            n0=c.n,
            n1=c.n,
            reps=ACTIVE_REPS,
            reps_max=ACTIVE_REPS,
            notes="greedy level-set infill of the C=1 boundary surface",
        )
        for c, name in zip(chosen, names, strict=True)
    ]


def summarize_selection(model, sel: Selection) -> str:
    """A human-readable description of what the batch targets."""
    x = _covariates(sel.chosen)
    mu, sd = model.latent(x)
    pred = expit(mu / np.sqrt(1.0 + np.pi * sd**2 / 8.0))
    lines = [
        f"pool {sel.n_pool} achievable candidates -> {len(sel.chosen)} cells, "
        f"{sel.cost_core_hours:.2f} CPU-hours at {ACTIVE_REPS} reps each",
        "",
        f"{'#':>3} {'df':>6} {'AUC':>6} {'n':>5} {'pred cov':>9} {'lat sd':>7} "
        f"{'cost(s)':>8}",
    ]
    for i, (c, p, s) in enumerate(zip(sel.chosen, pred, sd, strict=True)):
        lines.append(
            f"{i:>3} {c.df:>6.2f} {c.auc:>6.3f} {c.n:>5d} {p:>9.3f} {s:>7.3f} "
            f"{c.cost_core_seconds:>8.0f}"
        )
    dfs = np.array([c.df for c in sel.chosen])
    aucs = np.array([c.auc for c in sel.chosen])
    ns = np.array([c.n for c in sel.chosen])
    lines.extend(
        [
            "",
            f"df    : min {dfs.min():.2f} med {np.median(dfs):.2f} max {dfs.max():.2f}",
            f"AUC   : min {aucs.min():.3f} med {np.median(aucs):.3f} "
            f"max {aucs.max():.3f}",
            f"n     : min {ns.min()} med {np.median(ns):.0f} max {ns.max()}",
            f"pred  : {(pred < BAR_POINT).sum()} of {len(pred)} cells predicted "
            f"below the {BAR_POINT} bar",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--budget-cpu-hours", type=float, default=BUDGET_CPU_HOURS)
    parser.add_argument("--cost-exponent", type=float, default=COST_EXPONENT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--threads-per-call", type=int, default=4)
    parser.add_argument("--mem-gb", type=float, default=40.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    register_followup_shapes()
    lhs, anchors = load_cells(args.out)
    train: list[MeasuredCell] = lhs + anchors
    print(f"[design] fitting GP on {len(lhs)} LHS + {len(anchors)} anchor cells")
    model = fit_gp(train, restarts=6)
    pool = candidate_pool()
    sel = select_batch(
        model,
        pool,
        budget_cpu_hours=args.budget_cpu_hours,
        cost_exponent=args.cost_exponent,
    )
    print(summarize_selection(model, sel))
    if args.dry_run:
        return

    names = register_active_shapes(sel.chosen)
    cells = build_cells(sel.chosen, names)
    out_dir = args.out / "boundary_active"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "design_manifest.json").write_text(
        json.dumps(
            {
                "seed": ACTIVE_SEED,
                "reps": ACTIVE_REPS,
                "budget_cpu_hours": args.budget_cpu_hours,
                "cost_exponent": args.cost_exponent,
                "straddle_z": STRADDLE_Z,
                "bar": BAR_POINT,
                "n_bounds": list(N_BOUNDS),
                "planned_cpu_hours": sel.cost_core_hours,
                "cells": [
                    {"cell": c.name, "shape": s, **asdict(cand)}
                    for c, s, cand in zip(cells, names, sel.chosen, strict=True)
                ],
            },
            indent=1,
        )
    )
    t0 = time.time()
    for i, cell in enumerate(cells, 1):
        print(f"[{i}/{len(cells)}] {cell.name}", flush=True)
        run_cell(
            cell,
            out_dir,
            workers=args.workers,
            threads_per_call=args.threads_per_call,
            mem_gb=args.mem_gb,
        )
    print(f"[design] batch complete in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
