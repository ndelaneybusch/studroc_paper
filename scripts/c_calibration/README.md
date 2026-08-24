# Trim-exponent calibration study — runbook

Infrastructure for the one-shot offline calibration of the fiducial band's
`trim_exponent="auto"` map. Spec: `stats/c_calibration_spec.md`. Theory:
`stats/fiducial_band_theory.md` §7/§7.1.

Design decisions fixed at kickoff (2026-08-24): reps are **2× the spec
baseline** (2,000 fitting / 4,000 confirmation); the kink truth is the
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
# 0. Inspect the design + cost/memory estimates (no computation).
uv run python scripts/c_calibration/run.py --stage A --dry-run

# 1. Parity gate (REQUIRED before any Stage A cell; spec §5.6.2).
uv run python scripts/c_calibration/parity_gate.py

# 2. Stage A (fitting arms: core grid, large-n, imbalance).
#    Resumable: completed cells are skipped, partial cells extended.
uv run python scripts/c_calibration/run.py --stage A

# 2b. If any cells are flagged saturated at the end, re-run them at 2x M:
uv run python scripts/c_calibration/run.py --stage A --rerun-saturated

# 3. Fit: mechanical D1-D6 decisions, proposed frozen map + report.
uv run python scripts/c_calibration/fit_stage_a.py

# 4. HUMAN REVIEW of data/results/c_calibration/stage_a_fit_report.md.
#    Adjust/bless frozen_map.json; deviations from the mechanical rules
#    must be justified in stats/c_calibration_report.md.

# 5. Stage B (confirmation: held-out shapes, large-n, imbalance, ties),
#    against the frozen map, fresh seeds.
uv run python scripts/c_calibration/run.py --stage B \
    --map data/results/c_calibration/frozen_map.json

# 6. Acceptance criteria A1-A3 (A4 is human judgment).
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

Priority order if compute runs short (spec §9): core grid at n ≤ 5000 →
α = .05 large-n arm → confirmation arm → imbalance arm → central-α large-n
confirmation rows → α = .01 rows (drop by `--select`/`--arms`).

## Outputs (all under `data/results/c_calibration/`)

- `parity_gate.json` — gate results (record in the study report).
- `stageA/<cell>.json.gz` — raw per-rep ladder profiles (refit-able under
  any aggregation without re-simulation).
- `stageA/<cell>.summary.json` — per-cell aggregates: cov(j), per-α
  j*/C*/α_eff*/ℓ* with bootstrap CIs, saturation/D6 flags, allowance
  attribution, reference-map (C=1 / C=2 / provisional-auto) coverage+area.
- `frozen_map.json` — the proposed map artifact
  (`c-calibration-map/v1`, see `map_eval.py`).
- `stage_a_fit_report.md` — mechanical D1-D6 evidence.
- `stageB/…` — confirmation cells (same formats; auto arm = frozen map).
- `acceptance_check.json` — A1-A3 verdict.

## What is deliberately NOT here yet

- `trim_exponent="auto"` in `fiducial_band` / `fiducial_band_rs`: lands
  with the calibrated constants after acceptance (schema and resolver are
  frozen in `map_eval.py`).
- The final `stats/c_calibration_report.md`: written by a human from the
  outputs above.
