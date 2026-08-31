# Trim-exponent calibration study — runbook

Infrastructure for deciding whether, then calibrating, the fiducial band's
`trim_exponent="auto"` map. Spec: `stats/c_calibration_spec.md`. Theory:
`stats/fiducial_band_theory.md` §7/§7.1.

The full factorial campaign is deliberately conditional. Run the 27-cell
Stage S screen first. It tests whether a useful alpha=.05 margin survives the
shape envelope, whether the large-n taper is visible, and whether imbalance
is plausibly reducible. It does not fit a map or make a coverage claim.

Design decisions fixed at kickoff (2026-08-24): the screen starts at 500
reps and tops up only cells failing the alpha=.05 precision gate; a justified
full campaign uses **2× the original spec baseline** (2,000 fitting / 4,000
confirmation); the kink truth is the
**fixed shape** `t_kink = 0.004` (the m2 choice), not the n-dependent
`2/n0`; the pipeline **pauses between stages** for human review of the fit;
production `trim_exponent="auto"` wiring lands only after the acceptance
criteria pass (the artifact schema is already frozen in `map_eval.py`).

## One-time setup on the run machine

```bash
git clone <repo> && cd studroc_paper
uv sync                      # builds the fiducial_core Rust extension
uv run pytest tests/ -q      # includes the ladder/inv-beta parity tests
cargo test --release --manifest-path rust/Cargo.toml
```

## Run order

```bash
# 0. Inspect the stop/go screen (no computation).
uv run python scripts/c_calibration/run.py --stage S --dry-run

# 1. Validate the implementation before any simulation cell.
uv run python scripts/c_calibration/parity_gate.py

# 2. Run the screen, then read screening_report.md.
uv run python scripts/c_calibration/run.py --stage S
uv run python scripts/c_calibration/check_screen.py

# 3. Stop unless the report says the useful margin warrants map fitting.

# 4. Stage A (fitting arms: core grid, large-n, imbalance), reduced according
#    to the screen report.
#    Resumable: completed cells are skipped, partial cells extended.
uv run python scripts/c_calibration/run.py --stage A

# 4b. If any cells are flagged saturated at the end, re-run them at 2x M:
uv run python scripts/c_calibration/run.py --stage A --rerun-saturated

# 5. Fit: mechanical D1-D6 decisions, proposed frozen map + report.
uv run python scripts/c_calibration/fit_stage_a.py

# 6. HUMAN REVIEW of data/results/c_calibration/stage_a_fit_report.md.
#    If blockers exist, the fitter writes candidate_map.json rather than
#    frozen_map.json. Resolve and justify them in stats/c_calibration_report.md.

# 7. Stage B (confirmation: held-out shapes, large-n, imbalance, ties),
#    against the frozen map, fresh seeds.
uv run python scripts/c_calibration/run.py --stage B \
    --map data/results/c_calibration/frozen_map.json

# 8. Acceptance criteria A1-A3 (A4 is human judgment).
uv run python scripts/c_calibration/acceptance_check.py
```

## Tuning for the machine

Defaults assume ~16 cores / 64 GB. Per-cell parallelism is
`--workers` concurrent replicates × `--threads` rayon threads per kernel
call (default `cores // 4` × 4); `--mem-gb` (default 40) caps the number of
concurrent fiducial clouds — the n = 50,000 cells hold ~2.5 GB each, the
n = 5,000 / α_min = .01 cells ~1.6 GB (M ≈ 84k by the budget rule). For the
large-n arm prefer fewer workers with more threads, e.g.
`--workers 2 --threads 8`.

Sharding across machines: `--arms core`, `--arms large_n imbalance`, or
`--select <substring>` partition the cell list; outputs merge by simply
copying the `stageA/` JSONs into one directory.

Do not begin Stage A merely because compute is available. If Stage S says to
proceed, use its taper, shape, and imbalance tables to remove cells that no
longer answer a live decision. The checked-in Stage A remains the conservative
full design, not the default execution plan.

## Outputs (all under `data/results/c_calibration/`)

- `parity_gate.json` — gate results (record in the study report).
- `stageS/<cell>.json.gz` and `.summary.json` — raw and summarized stop/go
  evidence.
- `screening_check.json` / `screening_report.md` — the resource-allocation
  verdict; explicitly not a coverage guarantee.
- `stageA/<cell>.json.gz` — raw per-rep ladder profiles (refit-able under
  any aggregation without re-simulation).
- `stageA/<cell>.summary.json` — per-cell aggregates: cov(j), per-α
  j*/C*/α_eff*/ℓ* with bootstrap CIs, saturation/D6 flags, allowance
  attribution, reference-map (C=1 / C=2 / provisional-auto) coverage+area.
- `frozen_map.json` — the proposed map artifact
  (`c-calibration-map/v1`, see `map_eval.py`).
- `candidate_map.json` — written instead of `frozen_map.json` when D2, D4,
  or the floor check leaves a blocker; Stage B rejects it.
- `stage_a_fit_report.md` — mechanical D1-D6 evidence.
- `stageB/…` — confirmation cells (same formats; auto arm = frozen map).
- `acceptance_check.json` — A1-A3 verdict.

## What is deliberately NOT here yet

- `trim_exponent="auto"` in `fiducial_band` / `fiducial_band_rs`: lands
  with the calibrated constants after acceptance (schema and resolver are
  frozen in `map_eval.py`).
- The final `stats/c_calibration_report.md`: written by a human from the
  outputs above.

## Follow-up runs after the Stage S STOP (2026-08-30)

Stage S returned its pre-registered STOP (see the OUTCOME entry in
`stats/c_calibration_spec.md`): Stage A/B above will not run, and the
sections describing them are historical. What remains is
`followup_runs.py` (revised 2026-08-31 after external review; unit tests
in `tests/test_followup_runs.py`), detailed in the spec's dated
FOLLOW-UP RUN PLAN entry:

```
uv run --project . python scripts/c_calibration/followup_runs.py all
#   = boundary, heldout, composite, report (imbalance is deferred;
#     request it explicitly if the final-run guidance needs it)
uv run --project . python scripts/c_calibration/followup_runs.py boundary heldout
uv run --project . python scripts/c_calibration/followup_runs.py composite report
# knobs: --workers 4 --threads-per-call 4 --reps-scale --select <substr> --dry-run
```

Outputs land in `data/results/c_calibration_followup_20260830/<item>/`.
Runner cells resume/extend and top up on the coverage classification
(Wilson CI vs the .94 noninferiority bar — NOT the retired SE(C*) gate);
the boundary item also runs an ~95-cell LHS surface sweep (fixed 125
reps, no top-up) whose sign-constrained logistic smooth the report fits
and renders as a provisional n*(df, AUC) contour; composite cells store
per-rep records, resume, and refuse to mix with outputs from different
design constants. `report` writes `followup_report.md` with
PASS/MARGINAL/FAIL verdicts, the library-relative routing threshold,
the boundary surface + holdout diagnostics, and the composite candidate
table. Dry-run budget: ~8 idealized core-saturated hours plus top-ups
(composite includes two n=20,000 Theorem-7 erosion sentinels; cutoffs
proposed by the boundary smooth are confirmed later by spec follow-up
item 5).
