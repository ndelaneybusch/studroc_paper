"""Parity gate for the Rust study path (spec section 5.6.2).

Must pass before any Stage A cell runs. Two parts:

1. **Statistical parity.** Reproduce the round-2 validation cell
   (binormal .95, n = 500/500, M = 3000) with the study's own replicate
   path at both trim exponents and both alpha in {.05, .5}; coverage and
   area must agree with the published Python-path numbers
   (``stats/next_method_ideas.md``, implementation-validation note) within
   Monte Carlo error. The RNG streams differ, so agreement is statistical.

2. **Exact same-seed parity.** The ladder-profile reference path must agree
   with the production band ``fiducial_band_rs`` exactly (same seed, same
   depth selection, same band): identical coverage indicator and area to
   1e-9 (the only permitted difference is the CP quantile source, Rust
   invbetai vs scipy, which agree to ~1e-12), on a balanced cell and on a
   thinned trim-grid cell (K > 2001).

Writes ``parity_gate.json`` into the output directory and exits non-zero on
failure.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from design import Cell  # noqa: E402
from runner import provenance, run_rep, sample_labels  # noqa: E402
from shapes import get_curve  # noqa: E402

from studroc_paper.methods.fiducial_band_rs import fiducial_band_rs  # noqa: E402
from studroc_paper.methods.fiducial_ladder import (  # noqa: E402
    ladder_profile,
    make_ladder,
)

# Published round-2 validation numbers (production fiducial_band, 150 reps,
# M = 3000; next_method_ideas.md implementation-validation note). SEs:
# ~1.6pp on coverage at the 95% level, ~4pp at the 50% level.
PUBLISHED = {
    ("c1", 0.05): {"coverage": 0.973, "area": 0.0640, "cov_se": 0.016},
    ("c1", 0.50): {"coverage": 0.760, "area": 0.0420, "cov_se": 0.040},
    ("c2", 0.05): {"coverage": 0.960, "area": 0.0584, "cov_se": 0.016},
    ("c2", 0.50): {"coverage": 0.553, "area": 0.0350, "cov_se": 0.040},
}
AREA_TOL = 0.004  # absolute; area SEs are ~0.0005, this allows drift 8x that
GATE_REPS = 400
GATE_M = 3000


def statistical_parity(threads: int) -> tuple[bool, dict]:
    """Part 1: the round-2 validation cell through the study's rep path."""
    cell = Cell(
        name="parity--binormal_95--n500",
        stage="A",
        arm="parity",
        shape="binormal_95",
        n0=500,
        n1=500,
        alphas=(0.50, 0.05),
        reps=GATE_REPS,
        reps_max=GATE_REPS,
        m_draws=GATE_M,
    )
    curve = get_curve(cell.shape)
    rtrue = np.clip(curve.eval(np.arange(cell.n_grid) / cell.n0), 0.0, 1.0)
    ladder = make_ladder(cell.m_draws)

    from design import RefArm

    arms = [
        RefArm(label=label, alpha=alpha, exponent=c)
        for alpha in (0.05, 0.50)
        for label, c in (("c1", 1.0), ("c2", 2.0))
    ]
    t0 = time.time()
    records = [
        run_rep(
            cell,
            rep,
            rtrue=rtrue,
            ladder=ladder,
            arms=arms,
            m_draws=cell.m_draws,
            n_threads=threads,
        )
        for rep in range(cell.reps)
    ]
    runtime = time.time() - t0

    ok = True
    rows = []
    for a, arm in enumerate(arms):
        cov = float(np.mean([r["ref_covered"][a] for r in records]))
        area = float(np.mean([r["ref_area"][a] for r in records]))
        ref = PUBLISHED[(arm.label, arm.alpha)]
        se_ours = float(np.sqrt(max(cov * (1 - cov), 1e-9) / cell.reps))
        se_comb = float(np.hypot(se_ours, ref["cov_se"]))
        cov_ok = abs(cov - ref["coverage"]) <= 3.0 * se_comb
        area_ok = abs(area - ref["area"]) <= AREA_TOL
        ok &= cov_ok and area_ok
        rows.append(
            {
                "arm": arm.label,
                "alpha": arm.alpha,
                "coverage": round(cov, 4),
                "coverage_published": ref["coverage"],
                "coverage_tol_3se": round(3.0 * se_comb, 4),
                "coverage_ok": bool(cov_ok),
                "area": round(area, 5),
                "area_published": ref["area"],
                "area_ok": bool(area_ok),
            }
        )
    return ok, {
        "reps": cell.reps,
        "m_draws": cell.m_draws,
        "runtime_s": round(runtime, 1),
        "rows": rows,
    }


def exact_parity(
    threads: int, *, n0: int, n1: int, m_draws: int, n_reps: int
) -> tuple[bool, dict]:
    """Part 2: same-seed agreement of the ladder reference path with the
    production band, including the thinned trim-grid path when K > 2001."""
    cell = Cell(
        name=f"parity_exact--binormal_90--n{n0}x{n1}",
        stage="A",
        arm="parity",
        shape="binormal_90",
        n0=n0,
        n1=n1,
        alphas=(0.05,),
        reps=n_reps,
        reps_max=n_reps,
        m_draws=m_draws,
    )
    curve = get_curve(cell.shape)
    grid = np.arange(cell.n_grid) / cell.n0
    rtrue = np.clip(curve.eval(grid), 0.0, 1.0)

    ok = True
    worst = {"area_diff": 0.0, "cov_mismatches": 0, "j_checked": []}
    for rep in range(n_reps):
        lab_s, _ = sample_labels(cell, rep)
        # Route the production wrapper deterministically: strictly
        # descending scores make _merged_labels the identity on lab_s, and
        # the kernel seed becomes the wrapper rng's next draw.
        probe = np.random.default_rng((77, rep))
        probe.random(len(lab_s))  # tie-break subkeys, consumed identically
        seed = int(probe.integers(0, 2**64, dtype=np.uint64))

        for c_exp, alpha in ((1.0, 0.05), (2.0, 0.05), (2.0, 0.5)):
            band_rng = np.random.default_rng((77, rep))
            _, lo, hi = fiducial_band_rs(
                lab_s.astype(np.int64),
                -np.arange(len(lab_s), dtype=np.float64),
                alpha=alpha,
                n_draws=m_draws,
                trim_exponent=c_exp,
                random_state=band_rng,
                n_threads=threads,
            )
            ae = 1.0 - (1.0 - alpha) ** c_exp
            prof = ladder_profile(
                lab_s,
                rtrue=rtrue,
                n_draws=m_draws,
                seed=seed,
                ladder=np.array([1, 2, 5]),
                alpha_effs=[ae],
                trim_rows="production",
                n_threads=threads,
            )
            band_cov = bool(np.all(rtrue >= lo - 1e-12) and np.all(rtrue <= hi + 1e-12))
            band_area = float(np.mean(hi - lo))
            area_diff = abs(band_area - float(prof.ref_area[0]))
            worst["area_diff"] = max(worst["area_diff"], area_diff)
            worst["j_checked"].append(int(prof.ref_j[0]))
            if band_cov != bool(prof.ref_covered[0]):
                worst["cov_mismatches"] += 1
                ok = False
            if area_diff > 1e-9:
                ok = False
    worst["j_checked"] = sorted(set(worst["j_checked"]))
    return ok, {"n0": n0, "n1": n1, "m_draws": m_draws, "reps": n_reps, **worst}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Rust-path parity gate")
    parser.add_argument("--out", type=Path, default=Path("data/results/c_calibration"))
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args(argv)

    print("Part 1: statistical parity against the round-2 validation cell...")
    ok1, part1 = statistical_parity(args.threads)
    for row in part1["rows"]:
        print(
            f"  {row['arm']} alpha={row['alpha']}: cov {row['coverage']} "
            f"(published {row['coverage_published']}, tol {row['coverage_tol_3se']}) "
            f"{'OK' if row['coverage_ok'] else 'FAIL'}; area {row['area']} "
            f"(published {row['area_published']}) "
            f"{'OK' if row['area_ok'] else 'FAIL'}"
        )

    print("Part 2: exact same-seed parity (full grid, n=300/200)...")
    ok2a, part2a = exact_parity(args.threads, n0=300, n1=200, m_draws=2000, n_reps=10)
    print(
        f"  max area diff {part2a['area_diff']:.2e}, "
        f"cov mismatches {part2a['cov_mismatches']} "
        f"{'OK' if ok2a else 'FAIL'}"
    )
    print("Part 3: exact same-seed parity (thinned trim grid, n=2500/500)...")
    ok2b, part2b = exact_parity(args.threads, n0=2500, n1=500, m_draws=2000, n_reps=5)
    print(
        f"  max area diff {part2b['area_diff']:.2e}, "
        f"cov mismatches {part2b['cov_mismatches']} "
        f"{'OK' if ok2b else 'FAIL'}"
    )

    passed = ok1 and ok2a and ok2b
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "parity_gate.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "statistical": {"passed": ok1, **part1},
                "exact_full_grid": {"passed": ok2a, **part2a},
                "exact_thinned_grid": {"passed": ok2b, **part2b},
                "provenance": provenance(),
            },
            indent=1,
        )
    )
    print(f"\nParity gate: {'PASSED' if passed else 'FAILED'} "
          f"(written to {args.out / 'parity_gate.json'})")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
