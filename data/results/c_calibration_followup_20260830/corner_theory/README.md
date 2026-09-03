# Corner mechanism: theory-side predictions for the C = 1 wedge (2026-09-02)

Companion to `stats/fiducial_band_theory.md` §7.4 and
`scripts/c_calibration/corner_mechanism.py`. Every predictor here is derived
from the true ROC; the risk scores are not fitted to measured coverage.

| file | produced by | content |
|---|---|---|
| `cells.json` | `corner_mechanism.py cells` | the 257 student-t follow-up cells with their measured C = 1, alpha = .05 record (coverage, viol_low/high, realized mean depth j*, M) |
| `closed_forms.json` | `corner_mechanism.py closed` | Lemma 13 large-count approximations and the resolution-corrected finite-grid risk score |
| `tail_sim_predictions.json` | `corner_mechanism.py simulate` (6 shards, 150 reps x 2,500 draws) | Poissonized endpoint-simulator miss rates, depths, and per-grid-point profiles |
| `real_band_6.62_0.988_6656.json` | `corner_mechanism.py real 6.62 0.988 6656 12000 40` | the production band (`fiducial_band_rs`, C = 1) on 40 simulated datasets: per replicate `k_sat`, miss flags, grid indices (from the top) of lower-edge violations, max depth |
| `router_table.txt` | historical `corner_mechanism.py router` output | superseded large-k screen; use the finite-grid table in theory §7.4(d) |
| `sliver_check.json` | `corner_mechanism.py sliver 500 100 8000 <out>` | Corollary 14.1: the production C = 1 band and M3 on the constructed sliver DGP at AUC .60/.80/.95, n = 500/500 (C = 1 covers .64/.54/.56, M3 1.000) |
| `pointwise_check.txt` | `corner_mechanism.py pointwise` | the *pointwise* 95% fiducial interval (no trim) at fixed FPR grid points: covers .59-.61 (sliver) and .78-.83 (t(2)/.99) at FPR >= .98, .96-.995 on a concave-corner shape — the defect is pointwise, theory doc 7.4(i) |
| `rocnreg_bb/` | `scripts/c_calibration/rocnreg_bb_check/` (R, ROCnReg 1.0.9) | the published pointwise Bayesian-bootstrap band `pooledROC.BB` at fixed corner FPRs: covers .46-.72 (t(2)/.99, FPR >= .98), .50 (sliver, FPR >= .80), and .00-.11 (t(30)/.95, FPR >= .95 — degenerate point interval, microscopic depth); theory doc 7.4(i) |

Headline numbers (all 257 cells, failure = coverage < .94, 65 cells):

- finite-grid analytic score: correlation .867 with lower-edge violation;
  score <= .05 selects 122 cells with no sub-.94 coverage.
- Poissonized endpoint simulator: predicted coverage correlates .90 with measured (RMSE .025,
  mean residual +.014, none off by more than .10); predicted lower-edge miss rate vs
  measured: slope .82, intercept .005, correlation .905; predicted excess <= .01 selects
  103 cells with 0 failures (min coverage .944), > .01 contains all 65 failures.
- sliver DGP (Prop. 14 / Cor. 14.1): at every AUC tested the C = 1 band can be forced to
  cover ~.55-.65 at n = 500 while M3 covers 1.000; AUC and sample sizes cannot bound corner risk.
- production band on t(6.62)/.988, n = 6656: 3/40 lower-edge misses, at k_sat = 94, 142,
  1274 (the three regimes of Lemma 13); covering replicates had k_sat up to 281.

Reproduce (repo root):

```bash
uv run python scripts/c_calibration/corner_mechanism.py cells  data/results/c_calibration_followup_20260830/corner_theory/cells.json
uv run python scripts/c_calibration/corner_mechanism.py closed data/results/c_calibration_followup_20260830/corner_theory/cells.json data/results/c_calibration_followup_20260830/corner_theory/closed_forms.json
for i in 0 1 2 3 4 5; do uv run python scripts/c_calibration/corner_mechanism.py simulate data/results/c_calibration_followup_20260830/corner_theory/cells.json sim_$i.json $i 6 150 2500 & done; wait   # ~1 h on 24 cores
uv run python scripts/c_calibration/corner_mechanism.py compare data/results/c_calibration_followup_20260830/corner_theory/closed_forms.json sim_*.json
uv run python scripts/c_calibration/corner_mechanism.py router > data/results/c_calibration_followup_20260830/corner_theory/router_table.txt
uv run python scripts/c_calibration/corner_mechanism.py real 6.62 0.988 6656 12000 40
```
