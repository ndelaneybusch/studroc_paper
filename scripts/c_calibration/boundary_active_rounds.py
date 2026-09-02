"""Iterative greedy level-set learning of the C = 1 coverage boundary.

The single infill batch of :mod:`boundary_active_design` overturned the
mechanism the pre-registered surface assumed: at high AUC, coverage *falls* as n
grows (fixed shape t(4.69)/AUC .986 measures .993, .947, .903, .823, .847 at
n = 150, 400, 800, 1200, 2000). The unsafe set is therefore not a half-space
``n < n*(df, AUC)`` but a curved region interior in n, and locating it needs
sampling on both of its sides.

This module runs the design loop to convergence rather than for one batch. Each
round fits a GP to everything measured so far, selects a cost-weighted straddle
batch, runs it, and — because the round's predictions are recorded before its
cells are run — scores the previous surface prospectively on genuinely unseen
cells. That prospective score is the learning trajectory: when a fresh round
stops being surprising, and the estimated boundary stops moving between rounds,
the design has converged and further cells buy little.

Two changes from the single-batch design. The n range is extended upward,
because the failure region reaches past the old 2,500 ceiling and the boundary's
far side has to be bracketed rather than extrapolated to. And the cost exponent
is softened, since the informative cells now live at large n where an
aggressively per-second criterion would refuse to look.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))

import boundary_active_design as bad  # noqa: E402
from boundary_active_design import (  # noqa: E402
    Candidate,
    _covariates,
    candidate_pool,
    select_batch,
)
from boundary_surface_fits import Cell as MeasuredCell  # noqa: E402
from boundary_surface_fits import binomial_deviance, fit_gp  # noqa: E402
from design import Cell  # noqa: E402
from followup_runs import (  # noqa: E402
    BAR_POINT,
    DEFAULT_OUT,
    _cell,
    _cell_row,
    _load_summaries,
    register_followup_shapes,
)
from runner import run_cell  # noqa: E402
from shapes import ShapeSpec, make_t_shape, shape_registry  # noqa: E402

ROUND_REPS = 250
ROUND_N_BOUNDS = (100, 8_000)
ROUND_COST_EXPONENT = 0.5
DATA_SUBDIRS = ("boundary", "boundary_active", "boundary_nladder")

# AUC grid on which the estimated unsafe interval in n is compared across
# rounds; the boundary's movement on this grid is the convergence signal.
TRACK_AUCS = (0.95, 0.97, 0.98, 0.985, 0.99)
TRACK_DF = 4.0


def load_all(out_root: Path, *, extra: list[str] = ()) -> list[MeasuredCell]:
    """Every measured student-t cell across the study's output subdirectories."""
    cells: list[MeasuredCell] = []
    for sub in list(DATA_SUBDIRS) + list(extra):
        d = out_root / sub
        if not d.exists():
            continue
        for w in (_cell_row(s, d) for s in _load_summaries(d)):
            meta = w["shape_meta"]
            if meta.get("family") != "student_t":
                continue
            cells.append(
                MeasuredCell(
                    df=meta["df"],
                    auc=meta["auc"],
                    n=w["n0"],
                    cov=w["cov"],
                    reps=w["reps"],
                    name=w["name"],
                )
            )
    return cells


def unsafe_interval(
    model, *, auc: float, df: float = TRACK_DF, bar: float = BAR_POINT, q: float = 0.5
) -> tuple[float, float] | None:
    """The interval of n over which the surface sits below ``bar``.

    Args:
        model: Fitted GP.
        auc: AUC at which to trace the boundary.
        df: Degrees of freedom at which to trace it.
        bar: Coverage bar.
        q: Latent quantile to read; .5 is the median surface, lower values give
            the conservative read.

    Returns:
        ``(n_lo, n_hi)`` of the unsafe stretch, or None if the surface never
        drops below the bar across the sampled range.
    """
    grid = np.exp(
        np.linspace(np.log(ROUND_N_BOUNDS[0]), np.log(ROUND_N_BOUNDS[1]), 160)
    )
    cov = np.array(
        [model.predict_quantile(df=df, auc=auc, n=float(n), q=q) for n in grid]
    )
    below = cov < bar
    if not below.any():
        return None
    idx = np.where(below)[0]
    return float(grid[idx[0]]), float(grid[idx[-1]])


@dataclass
class RoundRecord:
    """One round's design, cost, and prospective score of the prior surface."""

    index: int
    n_cells: int
    cpu_hours: float
    prospective_dev_per_rep: float
    prospective_rmse: float
    prospective_bias: float
    n_below_bar: int
    boundary: dict[str, list[float] | None]
    boundary_shift: float | None

    def as_dict(self) -> dict:
        """JSON-serializable form."""
        return {
            "round": self.index,
            "n_cells": self.n_cells,
            "cpu_hours": round(self.cpu_hours, 3),
            "prospective_dev_per_rep": round(self.prospective_dev_per_rep, 5),
            "prospective_rmse": round(self.prospective_rmse, 4),
            "prospective_bias": round(self.prospective_bias, 4),
            "n_below_bar": self.n_below_bar,
            "boundary": self.boundary,
            "boundary_shift_log_n": (
                None if self.boundary_shift is None else round(self.boundary_shift, 4)
            ),
        }


def _register(cands: list[Candidate], *, rnd: int) -> list[str]:
    registry = shape_registry()
    names = []
    for i, c in enumerate(cands):
        name = f"tr{rnd}s{i:03d}"
        registry.setdefault(
            name,
            ShapeSpec(
                name=name,
                role="followup",
                build=(lambda auc=c.auc, df=c.df: make_t_shape(auc, df=df)),
                meta={"family": "student_t", "auc": c.auc, "df": c.df, "round": rnd},
            ),
        )
        names.append(name)
    return names


def _cells(cands: list[Candidate], names: list[str], *, rnd: int) -> list[Cell]:
    return [
        _cell(
            name=f"bround{rnd}--{name}--n{c.n}",
            stage="S",
            arm="followup_boundary_round",
            shape=name,
            n0=c.n,
            n1=c.n,
            reps=ROUND_REPS,
            reps_max=ROUND_REPS,
            notes="iterative level-set learning of the C=1 boundary",
        )
        for c, name in zip(cands, names, strict=True)
    ]


def _boundary_snapshot(model) -> dict[str, list[float] | None]:
    out = {}
    for a in TRACK_AUCS:
        iv = unsafe_interval(model, auc=a)
        out[f"{a:g}"] = None if iv is None else [round(iv[0], 1), round(iv[1], 1)]
    return out


def _boundary_shift(prev: dict, cur: dict) -> float | None:
    """Mean absolute movement of the unsafe interval's endpoints, in log n."""
    diffs = []
    for key, cur_iv in cur.items():
        prev_iv = prev.get(key)
        if cur_iv is None and prev_iv is None:
            diffs.append(0.0)
        elif cur_iv is None or prev_iv is None:
            diffs.append(np.log(ROUND_N_BOUNDS[1] / ROUND_N_BOUNDS[0]))
        else:
            diffs.extend(
                [
                    abs(np.log(cur_iv[0]) - np.log(prev_iv[0])),
                    abs(np.log(cur_iv[1]) - np.log(prev_iv[1])),
                ]
            )
    return float(np.mean(diffs)) if diffs else None


def run_rounds(
    *,
    out_root: Path,
    n_rounds: int,
    budget_per_round: float,
    workers: int,
    threads_per_call: int,
    mem_gb: float,
) -> list[RoundRecord]:
    """Run the design loop, returning one record per round."""
    bad.N_BOUNDS = ROUND_N_BOUNDS
    bad.ACTIVE_REPS = ROUND_REPS
    records: list[RoundRecord] = []
    prev_boundary: dict | None = None
    extra: list[str] = []

    for rnd in range(1, n_rounds + 1):
        train = load_all(out_root, extra=extra)
        print(f"\n=== round {rnd}: fitting GP on {len(train)} cells ===", flush=True)
        model = fit_gp(train, restarts=6)
        pool = candidate_pool(seed=bad.ACTIVE_SEED + rnd, size=bad.N_CANDIDATES)
        sel = select_batch(
            model,
            pool,
            budget_cpu_hours=budget_per_round,
            cost_exponent=ROUND_COST_EXPONENT,
        )
        x = _covariates(sel.chosen)
        mu, sd = model.latent(x)
        pred = expit(mu / np.sqrt(1.0 + np.pi * sd**2 / 8.0))
        boundary = _boundary_snapshot(model)

        names = _register(sel.chosen, rnd=rnd)
        cells = _cells(sel.chosen, names, rnd=rnd)
        sub = f"boundary_round{rnd}"
        out_dir = out_root / sub
        out_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"round {rnd}: {len(cells)} cells, {sel.cost_core_hours:.2f} CPU-h, "
            f"n {min(c.n for c in sel.chosen)}-{max(c.n for c in sel.chosen)}, "
            f"{(pred < BAR_POINT).sum()} predicted below bar",
            flush=True,
        )
        t0 = time.time()
        for i, cell in enumerate(cells, 1):
            if i % 10 == 0 or i == 1:
                print(f"  [{i}/{len(cells)}] {cell.name}", flush=True)
            run_cell(
                cell,
                out_dir,
                workers=workers,
                threads_per_call=threads_per_call,
                mem_gb=mem_gb,
            )
        extra.append(sub)

        prefix = f"bround{rnd}--"
        obs_cells = [
            c for c in load_all(out_root, extra=[sub]) if c.name.startswith(prefix)
        ]
        by_name = {c.name: c for c in obs_cells}
        obs = np.array([by_name[c.name].cov for c in cells if c.name in by_name])
        reps = np.array(
            [by_name[c.name].reps for c in cells if c.name in by_name], dtype=float
        )
        keep = [i for i, c in enumerate(cells) if c.name in by_name]
        pk = pred[keep]
        dev = binomial_deviance(successes=np.round(obs * reps), trials=reps, p=pk)
        shift = (
            None if prev_boundary is None else _boundary_shift(prev_boundary, boundary)
        )
        rec = RoundRecord(
            index=rnd,
            n_cells=len(obs),
            cpu_hours=sel.cost_core_hours,
            prospective_dev_per_rep=dev / reps.sum(),
            prospective_rmse=float(np.sqrt(np.mean((pk - obs) ** 2))),
            prospective_bias=float(np.mean(pk - obs)),
            n_below_bar=int((obs < BAR_POINT).sum()),
            boundary=boundary,
            boundary_shift=shift,
        )
        records.append(rec)
        print(
            f"round {rnd} done in {(time.time() - t0) / 60:.1f} min: "
            f"prospective dev/rep {rec.prospective_dev_per_rep:.4f}, "
            f"RMSE {rec.prospective_rmse:.3f}, bias {rec.prospective_bias:+.3f}, "
            f"{rec.n_below_bar}/{rec.n_cells} below bar"
            + ("" if shift is None else f", boundary shift {shift:.3f} log-n"),
            flush=True,
        )
        prev_boundary = boundary
        (out_root / "learning_trajectory.json").write_text(
            json.dumps([r.as_dict() for r in records], indent=1)
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--budget-per-round", type=float, default=2.5)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--threads-per-call", type=int, default=4)
    parser.add_argument("--mem-gb", type=float, default=40.0)
    args = parser.parse_args()

    register_followup_shapes()
    records = run_rounds(
        out_root=args.out,
        n_rounds=args.rounds,
        budget_per_round=args.budget_per_round,
        workers=args.workers,
        threads_per_call=args.threads_per_call,
        mem_gb=args.mem_gb,
    )
    print("\n=== learning trajectory ===")
    print(
        f"{'round':>6} {'cells':>6} {'CPU-h':>7} {'dev/rep':>9} "
        f"{'RMSE':>7} {'bias':>7} {'shift':>7}"
    )
    for r in records:
        s = "-" if r.boundary_shift is None else f"{r.boundary_shift:.3f}"
        print(
            f"{r.index:>6} {r.n_cells:>6} {r.cpu_hours:>7.2f} "
            f"{r.prospective_dev_per_rep:>9.4f} {r.prospective_rmse:>7.3f} "
            f"{r.prospective_bias:>+7.3f} {s:>7}"
        )


if __name__ == "__main__":
    main()
