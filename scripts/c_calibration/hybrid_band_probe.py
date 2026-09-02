"""Where the C = 1 fiducial band misses, and whether an M3 floor repairs it.

Beyond the routing boundary the safe play is to hand the whole curve to M3,
which costs 28-46% width everywhere. This module asks whether the loss is
localized instead: if the fiducial band's misses concentrate in small FPR
regions, widening it to M3 *only there* would restore coverage at a fraction of
the width.

Answering that needs miss locations, which the stored cell summaries do not
carry — they record only direction (`viol_low`/`viol_high`) and depth. Every
replicate is deterministically seeded, so the locations are recovered by replay:
for each rep this rebuilds the same rank data the runner used, the C = 1
fiducial band and the M3 band on it, and records

- the FPR profile of fiducial misses, pooled over reps;
- per replicate, how far down the FPR axis its misses reach, which is what
  determines how wide a repair region has to be;
- whether M3 covers at exactly those points (an M3 floor can only help where
  M3 itself is right);
- coverage and width of hybrid bands taking the pointwise union with M3 on
  ``[0, tau_lo] u [1 - tau_hi, 1]``, swept over both ends.

The misses are two-sided — a small cluster at the left corner and the dominant
mass approaching FPR = 1 — so a one-sided sweep understates what a hybrid can
do. The hybrid's edges get the same monotone closure as the composite band of
spec item 3: a running max on both, a valid tightening of the lower edge that
leaves the per-rep coverage event unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from followup_runs import (  # noqa: E402
    COV_TOL,
    DEFAULT_OUT,
    _cell,
    register_followup_shapes,
    sample_scores,
    wilson_ci,
)
from runner import truth_curve  # noqa: E402
from shapes import ShapeSpec, make_t_shape, shape_registry  # noqa: E402

from studroc_paper.methods.fiducial_band_rs import fiducial_band_rs  # noqa: E402
from studroc_paper.methods.m3_band_rs import m3_band_rs  # noqa: E402

ALPHA = 0.05
BAR = 0.94
TAU_LO = (0.0, 0.002, 0.005, 0.01, 0.02)
TAU_HI = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50)

# Failing cells spanning the (AUC, n) wedge, each with its measured C = 1
# coverage from the main study for cross-checking the replay.
PROBE_CELLS = (
    {"df": 2.00, "auc": 0.9900, "n": 500, "reps": 200, "measured": 0.690},
    {"df": 4.69, "auc": 0.9860, "n": 1200, "reps": 200, "measured": 0.823},
    {"df": 6.62, "auc": 0.9883, "n": 2600, "reps": 200, "measured": 0.877},
    {"df": 1.13, "auc": 0.9260, "n": 130, "reps": 200, "measured": 0.916},
    {"df": 3.29, "auc": 0.9844, "n": 5131, "reps": 100, "measured": 0.936},
)


def _monotone_close(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Running-max closure of both edges (the composite band's convention)."""
    return np.maximum.accumulate(lo), np.maximum.accumulate(hi)


@dataclass
class ProbeResult:
    """Replay diagnostics and hybrid sweep for one cell."""

    df: float
    auc: float
    n: int
    reps: int
    measured: float
    fiducial_cov: float
    m3_cov: float
    m3_area_ratio: float
    miss_profile: np.ndarray
    m3_covers_at_miss: float
    miss_mass_quantiles: dict[str, float]
    miss_span: dict[str, float]
    hybrid: list[dict] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Compact cell identifier for tables."""
        return f"t({self.df:g})/{self.auc:.3f}/n{self.n}"

    def best_repair(self) -> dict | None:
        """Cheapest swept region reaching the bar below the full-M3 width."""
        ok = [
            h
            for h in self.hybrid
            if h["coverage"] >= BAR and h["area_vs_fiducial"] < self.m3_area_ratio - 1.0
        ]
        return min(ok, key=lambda h: h["area_vs_fiducial"]) if ok else None


def probe_cell(
    *,
    df: float,
    auc: float,
    n: int,
    reps: int,
    measured: float,
    n_threads: int,
    verbose: bool = True,
) -> ProbeResult:
    """Replay one cell and sweep the hybrid's two-sided FPR region.

    Args:
        df: Student-t degrees of freedom.
        auc: True AUC.
        n: Per-class sample size.
        reps: Replicates to replay.
        measured: The cell's C = 1 coverage from the main study, for a
            consistency check on the replay.
        n_threads: Rayon threads for the fiducial kernel.
        verbose: Print progress.

    Returns:
        The diagnostics and hybrid sweep for this cell.
    """
    shape = f"probe_df{df:g}_a{auc:.4f}".replace(".", "")
    shape_registry().setdefault(
        shape,
        ShapeSpec(
            name=shape,
            role="followup",
            build=(lambda a=auc, d=df: make_t_shape(a, df=d)),
            meta={"family": "student_t", "auc": auc, "df": df},
        ),
    )
    cell = _cell(
        name=f"hybridprobe--{shape}--n{n}",
        stage="S",
        arm="followup_hybrid_probe",
        shape=shape,
        n0=n,
        n1=n,
        alphas=(ALPHA,),
        reps=reps,
        reps_max=reps,
        notes="hybrid-band FPR localization probe",
    )
    curve = truth_curve(cell)
    grid = np.arange(cell.n_grid) / cell.n0
    rtrue = np.clip(curve.eval(grid), 0.0, 1.0)

    combos = [(a, b) for a in TAU_LO for b in TAU_HI]
    miss_profile = np.zeros(len(grid))
    m3_ok_num = m3_ok_den = 0
    fid_cov = m3_cov = 0
    fid_area = np.empty(reps)
    m3_area = np.empty(reps)
    hyb_cov = dict.fromkeys(combos, 0)
    hyb_area = {c: np.empty(reps) for c in combos}
    span_lo: list[float] = []
    span_hi: list[float] = []

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for rep in range(reps):
            y_true, y_score, seed = sample_scores(cell, rep)
            _, flo, fhi = fiducial_band_rs(
                y_true,
                y_score,
                alpha=ALPHA,
                n_draws=cell.m_draws,
                trim_exponent=1.0,
                n_threads=n_threads,
                random_state=seed,
            )
            _, mlo, mhi = m3_band_rs(y_true, y_score, alpha=ALPHA, random_state=rep)

            fid_miss = (flo > rtrue + COV_TOL) | (rtrue > fhi + COV_TOL)
            m3_miss = (mlo > rtrue + COV_TOL) | (rtrue > mhi + COV_TOL)
            fid_cov += not fid_miss.any()
            m3_cov += not m3_miss.any()
            fid_area[rep] = float(np.mean(fhi - flo))
            m3_area[rep] = float(np.mean(mhi - mlo))
            miss_profile += fid_miss
            if fid_miss.any():
                m3_ok_num += int((~m3_miss[fid_miss]).sum())
                m3_ok_den += int(fid_miss.sum())
                where = grid[fid_miss]
                span_lo.append(float(where.min()))
                span_hi.append(float(where.max()))

            union_lo = np.minimum(flo, mlo)
            union_hi = np.maximum(fhi, mhi)
            for tau_lo, tau_hi in combos:
                region = (grid <= tau_lo) | (grid >= 1.0 - tau_hi)
                lo, hi = _monotone_close(
                    np.where(region, union_lo, flo), np.where(region, union_hi, fhi)
                )
                hyb_cov[(tau_lo, tau_hi)] += bool(
                    np.all(lo <= rtrue + COV_TOL) and np.all(rtrue <= hi + COV_TOL)
                )
                hyb_area[(tau_lo, tau_hi)][rep] = float(np.mean(hi - lo))
            if verbose and (rep + 1) % 50 == 0:
                print(
                    f"    rep {rep + 1}/{reps} "
                    f"({(rep + 1) / (time.time() - t0):.1f}/s)",
                    flush=True,
                )

    total = miss_profile.sum()
    qs = {}
    if total > 0:
        cum = np.cumsum(miss_profile) / total
        for frac in (0.5, 0.9, 0.99, 1.0):
            idx = int(np.searchsorted(cum, frac - 1e-12))
            qs[f"{frac:g}"] = float(grid[min(idx, len(grid) - 1)])
    span = {}
    if span_lo:
        arr = np.asarray(span_lo)
        span = {
            "lowest_missed_fpr_q10": float(np.quantile(arr, 0.10)),
            "lowest_missed_fpr_median": float(np.median(arr)),
            "highest_missed_fpr_median": float(np.median(span_hi)),
            "frac_above_0.5": float(np.mean(arr > 0.5)),
            "frac_above_0.9": float(np.mean(arr > 0.9)),
        }

    base = float(fid_area.mean())
    hybrid = [
        {
            "tau_lo": a,
            "tau_hi": b,
            "coverage": hyb_cov[(a, b)] / reps,
            "coverage_wilson95": wilson_ci(hyb_cov[(a, b)] / reps, reps),
            "area": float(hyb_area[(a, b)].mean()),
            "area_vs_fiducial": float(hyb_area[(a, b)].mean() / base - 1.0),
        }
        for a, b in combos
    ]
    return ProbeResult(
        df=df,
        auc=auc,
        n=n,
        reps=reps,
        measured=measured,
        fiducial_cov=fid_cov / reps,
        m3_cov=m3_cov / reps,
        m3_area_ratio=float(m3_area.mean() / base),
        miss_profile=miss_profile,
        m3_covers_at_miss=(m3_ok_num / m3_ok_den if m3_ok_den else float("nan")),
        miss_mass_quantiles=qs,
        miss_span=span,
        hybrid=hybrid,
    )


def render(results: list[ProbeResult]) -> str:
    """Format the probe as a markdown report."""
    lines = [
        "# Hybrid band: where the fiducial misses, and whether an M3 floor repairs it",
        "",
        f"*Replayed cells, alpha = {ALPHA}. The hybrid takes the pointwise union "
        "with M3 on `[0, tau_lo] u [1 - tau_hi, 1]` and keeps the C = 1 fiducial "
        "band elsewhere, with running-max closure on both edges.*",
        "",
        "## Replay consistency and the M3 arm",
        "",
        "| cell | reps | C=1 replay | C=1 study | M3 cov | M3/C1 area |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.label} | {r.reps} | {r.fiducial_cov:.3f} | {r.measured:.3f} | "
            f"{r.m3_cov:.3f} | {r.m3_area_ratio:.2f}x |"
        )
    lines.extend(
        [
            "",
            "## Where the misses are",
            "",
            "FPR below which the stated share of pooled pointwise miss mass falls, "
            "and whether M3 covers at the points the fiducial band misses.",
            "",
            "| cell | 50% | 90% | 99% | 100% | M3 correct at miss points |",
            "|---|---|---|---|---|---|",
        ]
    )
    for r in results:
        q = r.miss_mass_quantiles
        vals = " | ".join(
            f"{q.get(k, float('nan')):.4f}" for k in ("0.5", "0.9", "0.99", "1")
        )
        lines.append(f"| {r.label} | {vals} | {r.m3_covers_at_miss:.3f} |")
    lines.extend(
        [
            "",
            "## How far down a repair must reach",
            "",
            "Per replicate that misses, the lowest FPR at which it misses. A "
            "region `[1 - tau_hi, 1]` can only fix reps whose misses lie "
            "entirely above `1 - tau_hi`.",
            "",
            "| cell | q10 | median | share confined to FPR>.5 | share >.9 |",
            "|---|---|---|---|---|",
        ]
    )
    for r in results:
        s = r.miss_span
        if not s:
            continue
        lines.append(
            f"| {r.label} | {s['lowest_missed_fpr_q10']:.3f} | "
            f"{s['lowest_missed_fpr_median']:.3f} | {s['frac_above_0.5']:.2f} | "
            f"{s['frac_above_0.9']:.2f} |"
        )
    lines.extend(["", "## Hybrid sweep: coverage (width change vs the C = 1 band)", ""])
    for r in results:
        lines.extend(
            [
                f"**{r.label}** — fiducial covers {r.fiducial_cov:.3f}; full M3 "
                f"covers {r.m3_cov:.3f} at {(r.m3_area_ratio - 1) * 100:+.0f}% width.",
                "",
                "| tau_lo \\ tau_hi | " + " | ".join(f"{b:g}" for b in TAU_HI) + " |",
                "|---" * (1 + len(TAU_HI)) + "|",
            ]
        )
        by = {(h["tau_lo"], h["tau_hi"]): h for h in r.hybrid}
        for a in TAU_LO:
            cells = [
                f"{by[(a, b)]['coverage']:.3f} "
                f"({by[(a, b)]['area_vs_fiducial'] * 100:+.0f}%)"
                for b in TAU_HI
            ]
            lines.append(f"| {a:g} | " + " | ".join(cells) + " |")
        best = r.best_repair()
        if best is not None:
            lines.append(
                f"\nCheapest region reaching {BAR}: tau_lo={best['tau_lo']:g}, "
                f"tau_hi={best['tau_hi']:g} gives coverage {best['coverage']:.3f} "
                f"at {best['area_vs_fiducial'] * 100:+.1f}% width, against "
                f"{(r.m3_area_ratio - 1) * 100:+.0f}% for full M3."
            )
        else:
            lines.append(
                f"\nNo swept region reaches {BAR} more cheaply than routing the "
                "whole curve to M3."
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT / "hybrid_probe")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--reps-scale", type=float, default=1.0)
    args = parser.parse_args()

    register_followup_shapes()
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in PROBE_CELLS:
        reps = max(10, int(spec["reps"] * args.reps_scale))
        print(
            f"[probe] t({spec['df']})/{spec['auc']}/n{spec['n']} x {reps}", flush=True
        )
        results.append(
            probe_cell(
                df=spec["df"],
                auc=spec["auc"],
                n=spec["n"],
                reps=reps,
                measured=spec["measured"],
                n_threads=args.threads,
            )
        )
    report = render(results)
    (args.out / "hybrid_probe.md").write_text(report)
    (args.out / "hybrid_probe.json").write_text(
        json.dumps(
            [
                {
                    "df": r.df,
                    "auc": r.auc,
                    "n": r.n,
                    "reps": r.reps,
                    "fiducial_cov": r.fiducial_cov,
                    "m3_cov": r.m3_cov,
                    "m3_area_ratio": r.m3_area_ratio,
                    "m3_covers_at_miss": r.m3_covers_at_miss,
                    "miss_mass_quantiles": r.miss_mass_quantiles,
                    "miss_span": r.miss_span,
                    "hybrid": r.hybrid,
                    "miss_profile": r.miss_profile.tolist(),
                }
                for r in results
            ],
            indent=1,
        )
    )
    print(report)


if __name__ == "__main__":
    main()
