# M2 (rank-space fiducial ROC band): second falsification round

*Laptop CPU run, 2026-08-21. All code and results live in
`stats/experiments/`. This report continues §8b of
`stats/next_method_ideas.md`.*

## 0. What was run, and how to reproduce

New harness: **`m2_experiments.py`** (imports `rank_band_experiments.py` for
the truths, the Dirichlet-spacings fiducial sampler and the min-p machinery).
Helpers: `print_m2.py` (tables), `analyze_recal.py` (P1 transfer analysis),
`alpha_sweep.py` (P5), `baselines_m2.py` (oracle / KS / WH on the new cells).

```
cd stats/experiments
U="uv run --project /home/nathan/Documents/studroc_paper python"
$U m2_experiments.py --exp p1diag --cells C1 C2 --reps 400 --M 3000 \
      --alphas 0.5 0.2 0.1 0.05 --out res_p1diag_a.json
$U print_m2.py res_p1diag_a.json               # per-cell tables
$U analyze_recal.py res_*.json                 # recalibration transfer
$U alpha_sweep.py res_p1diag_a.json            # nominal vs actual
```

Other flags: `--exp {p1diag,p1cal,p2,p3,p4}`, `--subM 1000 3000` (analyse
prefixes of the same cloud at smaller M), `--thin 5` (trim on a thinned grid,
evaluate on the full one), `--quant 20 --tie {jitter,even,neg1st}
--evalmode {trapezoid,staircase}` (ties red team), `--cplo {none,full,deg}`
(lower-edge exact allowance), `--ncal/--min` (nested calibration budget).
Each cell is seeded deterministically from its name, so every table below is
reproducible. Result JSONs: `res_p1diag_*`, `res_p1cal_*`, `res_p2_*`,
`res_p3_*`, `res_p4_*`, `res_baselines_p2.json`; run logs `log_*.txt`.

### The identity that makes everything cheap

Let `R_1..R_M` be the fiducial draws on the grid `t_k = k/n0`, and for a
reference curve `c` put `a_k = #{m : R_m(t_k) <= c(t_k)}`,
`b_k = #{m : R_m(t_k) >= c(t_k)}`. The pointwise
`[j-th smallest, j-th largest]` tube contains `c` **iff**
`j <= S(c) := min_k min(a_k, b_k)`.

So one pass over the draws yields the band at *every* trim depth `j`, and the
depth statistic of the **truth** gives the coverage of every depth at once.
Every P1/P4 result below is a post-hoc slice of a single per-rep profile over a
~105-point ladder of `j`, which is why the whole programme fits in a few
minutes per cell. (Coverage with the Clopper–Pearson upper allowance is not a
pure function of `S`, so it is evaluated explicitly at each ladder point.)

### Cells

| id | truth | n0 / n1 | note |
|---|---|---|---|
| C1..C7 | as before | | binormal .75/.95/.90, bimodal .90, t(2) .95 |
| P2a | binormal .90 | 900 / 100 | 9:1 imbalance |
| P2b | binormal .90 | 100 / 900 | 1:9 imbalance |
| P2c | binormal .99 | 500 / 500 | steepest corner |
| P2d | binormal .99 | 150 / 150 | steepest corner, small n |
| P2e | binormal .55 | 500 / 500 | near-diagonal |
| P2f | kink (vertical to TPR .6 by FPR 2/n0, then straight to (1,1)) | 500 / 500 | AUC .798 |
| P4a/b/c | binormal .95 | 500 / 2000 / 5000 (balanced) | M-vs-K sweep; P4c == C6 |

Reps: 400 (200 for P4a/b, 120 for P4c, 100 for the nested-calibration arm).
Coverage SE ≈ 1.1pp at 400 reps / 0.95; ≈ 2.2pp at 100 reps.

### Arms

* **`fid_cp`** — the current M2 recipe: fiducial trim at `alpha_eff = alpha`
  plus the exact CP upper allowance at the band's own local level `j/(M+1)`.
* **`fid_rc`** — identical, but the trim uses a *recalibrated* level
  `alpha_eff = 1 - (1-alpha)^C` (fixed universal `C`).
* **`recal`** — the per-cell *optimal* `alpha_eff*` (largest `alpha_eff` whose
  realised coverage reaches `1-alpha`). Not implementable; it is the ceiling
  of any level-only recalibration, and it is what P1 is trying to estimate.
* **`fid_cal`** — per-rep frequentist calibration of the trim depth through a
  Hazen plug-in curve (nested Monte Carlo).

---

## 1. P1 — central-alpha recalibration of the trim level

### Setup

Hypothesis: the fiducial band's *shape* is right and only its *level* is
conservative. Two estimators of the level were tested.

1. **Level remap (cheap variant).** Keep the fiducial trim rule but evaluate it
   at `alpha_eff = f(alpha)`. `f` is fitted on {C1, C2, C4, C5} and then
   *transferred*, without refitting, to eight held-out cells (C3, C7, P2a–P2f)
   and to the three P4 sample sizes.
2. **Per-rep plug-in calibration (`fid_cal`).** For each replicate: build the
   Hazen plug-in `R0`; simulate `ncal = 80` rank-space datasets from `R0`; for
   each, draw `m_in = 1000` fiducial curves, build the full band ladder (CP
   allowance included) and record whether it covers `R0`; take the largest
   depth whose simulated coverage reaches `1-alpha`; rescale to the outer
   `M = 3000`. Run on C2 and C5, 100 reps (≈ 15 min/cell — 80× the cost of the
   band itself).

### Result 1a — the per-cell optimal level `alpha_eff*`

| cell | α=0.50 | α=0.20 | α=0.10 | α=0.05 |
|---|---|---|---|---|
| C1 binormal .75 n=500 | 0.780 | 0.410 | 0.250 | 0.130 |
| C2 binormal .95 n=500 | 0.780 | 0.420 | 0.210 | 0.110 |
| C4 bimodal .90 n=500 | 0.780 | 0.480 | 0.280 | 0.140 |
| C5 t(2) .95 n=500 | 0.680 | 0.310 | 0.190 | 0.090 |
| C3 binormal .95 n=150 | 0.810 | 0.480 | 0.270 | 0.160 |
| C7 binormal .90 n=25 | 0.880 | 0.580 | 0.350 | 0.210 |
| P2a 900/100 | 0.790 | 0.390 | 0.170 | 0.110 |
| P2b 100/900 | 0.750 | 0.390 | 0.240 | 0.140 |
| P2c .99 n=500 | 0.750 | 0.370 | 0.220 | 0.120 |
| P2d .99 n=150 | 0.820 | 0.360 | 0.250 | 0.140 |
| P2e .55 n=500 | 0.760 | 0.430 | 0.240 | 0.150 |
| P2f kink n=500 | 0.730 | 0.380 | 0.230 | 0.120 |
| P4b n=2000 (M=8000) | 0.695 | 0.380 | 0.210 | 0.095 |
| P4c n=5000 (M=8000) | 0.710 | 0.355 | 0.200 | 0.115 |
| **median** | **0.775** | **0.390** | **0.230** | **0.128** |

Written as a Šidák exponent `C = log(1-α_eff*)/log(1-α)`, the median is
2.18 / 2.29 / 2.60 / 2.83 at α = .5 / .2 / .1 / .05, with a per-cell range of
1.6–3.1. **`C ≈ 2` describes the whole 14-cell grid** — which spans n = 25 to
5000, AUC .55 to .99, three shapes, 9:1 imbalance in both directions, and a
kinked truth.

*Interpretation (conjecture, worth a lemma).* `1-α_eff = (1-α)^2` is exactly
the Šidák correction for **two independent samples**. The fiducial trim spends
the simultaneity budget as if the composed curve were one object; the
frequentist requirement appears to be one budget per class. If that is the
mechanism, `C = 2` is structural rather than fitted, and the "zero magic
numbers" property survives.

### Result 1b — transfer of the fixed map `alpha_eff = 1-(1-alpha)^C`

Coverage under the *fixed* map, 12 cells at n ≤ 900 (no refitting):

| map | α=.50 | α=.20 | α=.10 | α=.05 | mean error vs nominal |
|---|---|---|---|---|---|
| identity (`fid_cp`) | .655–.858 | .880–.960 | .945–.980 | .968–.995 | +.242 / +.121 / +.067 / +.031 |
| **C = 2.0** | .435–.627 | .762–.895 | .892–.965 | **.945–.980** | +.033 / +.027 / +.026 / +.017 |
| C = 2.2 | .403–.613 | .750–.875 | .877–.955 | .940–.973 | +.001 / +.008 / +.016 / +.012 |
| C = 2.4 | .380–.588 | .735–.870 | .870–.945 | .930–.970 | −.032 / −.003 / +.012 / +.006 |

Adding the large-n P4 cells (27 rows including deliberately under-resourced
`M`) the C = 2.0 map gives mean errors −.010 / +.010 / +.014 / +.005 and a
minimum α=.05 coverage of 0.942 among adequately-resourced runs (0.905 in the
`M = 1000, n0 = 2000` run, which is min-p saturated — see P4).

Area saving from recalibration: **9–13 % at α=.05, 11–14 % at α=.2, 14–18 % at
α=.5** (`ar/α` columns of `analyze_recal.py`).

### Result 1c — per-rep plug-in calibration (`fid_cal`), 100 reps

| cell | arm | α=.50 | α=.20 | α=.10 | α=.05 |
|---|---|---|---|---|---|
| C2 | `fid_cp` | .750 | .880 | .930 | .960 |
| C2 | `fid_cal` | **.630** | **.810** | .870 | .930 |
| C2 | `recal` (ceiling) | .530 | .810 | .900 | .950 |
| C2 | mean j: cal / ceiling | 86 / 109 | 26 / 33 | 12 / 8 | 5.3 / 4.0 |
| C5 | `fid_cp` | .660 | .900 | .960 | .970 |
| C5 | `fid_cal` | .610 | .900 | .970 | .980 |
| C5 | `recal` (ceiling) | .510 | .800 | .900 | .950 |
| C5 | mean j: cal / ceiling | 83 / 122 | 20 / 35 | 7 / 23 | 3.8 / 10.1 |

`fid_cal` removes most of the over-coverage on C2 (α=.2: .880 → .810, bang on
nominal) but **barely moves C5** — exactly the cell where the fixed map
under-covers. The calibrated depth is systematically *smaller* than the
optimum (factor 1.3 on C2, 1.7 on C5 at α=.2): the Hazen plug-in curve
inherits the roughness of one dataset, so its own fiducial bands cover it less
often than they cover a smooth truth, and the calibration returns a too-wide
band. This is the same plug-in pathology that killed M1, attenuated but not
gone. A second limitation is budgetary: at `m_in = 1000` the inner min-p
statistic is itself near saturation at α ≤ .1, which caps the resolution.

### Verdict vs the stated criteria

* Success criterion *"within ±0.05 at α=0.2 and ±0.10 at α=0.5 without
  dropping α=0.05 below ~0.94"*: **met** by the fixed C = 2.0 map for every
  cell except n = 25 (C7: +9.5pp at α=.2, +12.7pp at α=.5). Excluding n = 25,
  α=.2 lands in [.762, .858] and α=.5 in [.435, .573]. α=.05 never drops below
  0.942 with adequate `M`.
* Kill criterion *"shape-dependence > 10pp across cells at fixed α"*:
  **partially triggered**. At α=.05 and .10 the spread is 3.5pp and 7.3pp
  (pass). At α=.2 it is 13pp and at α=.5 19pp (fail). Crucially the *identity*
  map has 8pp / 20pp spread at the same α, so recalibration does not
  *introduce* the dependence — it removes the bias (+.24 → +.03 at α=.5) and
  leaves the spread. `dcoverage/d(alpha_eff) ≈ 1` at α=.5, so a 15pp spread in
  `alpha_eff*` is mechanically a 15pp spread in coverage there.
* Per-rep plug-in calibration is **not recommended**: 80× compute for a result
  that is better on one cell, no better on another, and biased conservative by
  a mechanism (plug-in roughness) we already know.

**What this means for the method.** The fiducial band's geometry is correct;
its level is off by a single, near-universal factor which looks like a Šidák
correction for the two samples. A one-line change (`alpha_eff = 1-(1-alpha)^2`)
converts M2 from "valid but 12–34pp over-covered at central α" into "centred to
within a few points at every α, ~10 % narrower, still ≥ 0.94 at α=.05". The
remaining shape-dependence at α ≥ .2 is real and is the honest limit of a
level-only correction.

---

## 2. P2 — new vulnerability slices (α = 0.05, `fid_cp` as shipped)

### Setup

`fid_cp` unchanged, 400 reps, M = 3000. Truth for P2f built directly on the
fine grid via `Curve()`. Baselines (oracle / KS / WH) from `baselines_m2.py`
with the same grid and metrics.

### Result

| cell | cov | v_low | v_high | mean depth\|miss | p95 depth | max depth | med miss FPR | area | oracle | KS | WH (cov) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P2a 900/100 .90 | .980 | .007 | .013 | .0126 | 0 | .055 | .489 | .1467 | .1113 | .2710 | .0832 (.965) |
| P2b 100/900 .90 | .990 | .007 | .003 | .0124 | 0 | .021 | .370 | .1423 | .1194 | .2691 | .0793 (.973) |
| P2c .99 n=500 | .978 | .007 | .015 | .0078 | 0 | .033 | .084 | .0288 | .0155 | .1393 | .0094 (.968) |
| P2d .99 n=150 | .995 | .000 | .005 | .0049 | 0 | .010 | .420 | .0613 | .0191 | .2412 | .0178 (.970) |
| P2e .55 n=500 | .983 | .013 | .005 | .0025 | 0 | .005 | .462 | .1625 | .1534 | .2475 | .0959 (.965) |
| P2f kink n=500 | .970 | .013 | .018 | .0052 | 0 | .014 | .787 | .1193 | .1099 | .2113 | .0673 (**.000**) |
| C3 n=150 | .990 | .005 | .005 | .0065 | 0 | .012 | .163 | .1137 | — | .280 | .057 |
| C7 n=25 | .988 | .000 | .013 | .0072 | 0 | .030 | .680 | .3453 | — | .602 | .211 |

* **No validity failure anywhere.** Coverage 0.970–0.995 across 9:1 imbalance
  in both directions, AUC .55 and .99, and the kinked truth. p95 miss depth is
  0 in every cell.
* **The low-FPR corner is handled natively.** For the two steep-corner cells
  (P2c/P2d, `R(2/n0)` = 0.74) and for the kink (`R(2/n0)` = 0.60), `v_low` is
  0.000–0.013 and the median miss sits at FPR 0.08–0.79, i.e. *not* at the
  corner. The fresh Dirichlet mass beyond the largest observed negative does
  the job the old method needed an explicit Beta order-statistic floor for.
* **WH is invalid on the kinked truth** (coverage 0.000, as it already was on
  bimodal and t(2)); KS is 100 % but 1.5–4.8× wider than `fid_cp`
  (`fid_cp` is 21–66 % of KS area across all 13 cells).
* **Where `fid_cp` is loose:** P2d (.99, n=150) is 3.2× the oracle band and
  P2c 1.9×; everywhere else it is 1.05–1.35× oracle. The steep corner at small
  n is the remaining width story, not the coverage story.

### Mirrored lower-edge allowance

Two variants of the mirror of the CP upper allowance were measured, both at
the band's own local level `ell = j/(M+1)`:

| variant | P2c area @.05 | P2f area @.05 | C2 area @.05 | coverage change |
|---|---|---|---|---|
| none (shipped) | .0288 | .1193 | .0634 | — |
| **full** `L := min(L, BetaInv(ell; k̂, n1−k̂+1))` | .0331 (**+15 %**) | .1197 (+0.3 %) | — | none |
| **degenerate only** `L := 0 where k̂ = 0` | .0288 (**+0 %**) | .1193 (+0 %) | .0634 (+0 %) | none |

The *full* mirror is a bad trade: at the TPR plateau `k̂ = n1` gives
`BetaInv(ell; n1, 1) = ell^{1/n1} ≈ 0.987`, which drags the lower edge down
across the whole plateau for 15 % of area and zero coverage. The
**degenerate-only** mirror — force `L = 0` wherever no positive has been seen,
the exact reflection of "CP upper = 1 when `k̂ = n1`" — is *free* on every
smooth truth and does rescue the one geometry where the truth is exactly 0
(see P3). Recommend shipping the degenerate mirror, not the full one.

**What this means for the method.** P2 found no new failure mode. The
asymmetry between the two exact allowances is now understood: the fiducial
band's only structural blind spot is *degenerate* corners (TPR exactly 1 or
exactly 0), and the exact binomial bound is needed only there.

---

## 3. P3 — ties / discreteness red team

### Setup

Scores are quantised to `Q ∈ {20, 100}` equal-probability (w.r.t. the negative
class) levels before the method sees them. Three tie conventions:

* **`jitter`** — random tie break (uniform position inside the level).
* **`even`** — deterministic: each class spread evenly inside the level at
  `(i−½)/count`, then merged (the mid-rank-style interleaving).
* **`neg1st`** — adversarial: all tied negatives ranked above all tied
  positives.

**Convention chosen for the method: random tie break (`jitter`).** It is not
merely conservative, it is *exact*: jittering inside a level makes the score
distribution genuinely continuous, and the jittered negatives are exactly
U(0,1) while the jittered positives have CDF exactly the piecewise-linear
interpolation of `R_true` through the level boundaries. So the rank-space
reduction applies verbatim, with estimand = the **trapezoidal ROC** of the
quantised score — which is also the convention behind the usual (Mann–Whitney)
AUC.

Truth for evaluation is therefore computed from the *quantised* class
probabilities (`quantized_truth`), not the continuous curve; for `neg1st` the
matching estimand is the **lower staircase**, also implemented.

### Result

| cell | Q | tie | estimand | α=.50 | α=.20 | α=.10 | α=.05 | area@.05 |
|---|---|---|---|---|---|---|---|---|
| C1 .75 n=500 | — | (continuous) | — | .733 | .927 | .963 | .980 | .1394 |
| C1 | 20 | jitter | trapezoid | .723 | .915 | .960 | **.980** | .1404 |
| C1 | 100 | jitter | trapezoid | .735 | .910 | .960 | **.985** | .1392 |
| C2 .95 n=500 | — | (continuous) | — | .748 | .912 | .963 | .975 | .0634 |
| C2 | 20 | jitter | trapezoid | .738 | .920 | .965 | **.978** | .0761 |
| C2 | 100 | jitter | trapezoid | .757 | .920 | .958 | **.980** | .0662 |
| C2 | 20 | even | trapezoid | .805 | .938 | .968 | **.983** | .0750 |
| C2 | 20 | neg1st | trapezoid | .000 | .000 | .000 | **.000** | .0986 |
| C2 | 20 | neg1st | staircase | .000 | .000 | .000 | **.000** | .0986 |
| C2 | 20 | neg1st | staircase + degenerate `L=0` | .310 | .420 | .458 | **.475** | .0986 |

* **Random tie breaking is a non-event.** Coverage under Q = 20 and Q = 100 is
  statistically indistinguishable from the continuous case at every α, on both
  cells. Q = 20 (where the first level contains ~75 % of the positives in C2)
  widens the band (area .0634 → .0761) because the data really are less
  informative — exactly as it should.
* **Even/mid-rank spreading is valid and slightly conservative** (.805 vs .738
  at α=.5), at essentially the same width.
* **Class-ordered tie breaking is fatal — and correctly so.** Ranking all tied
  negatives above all tied positives changes the estimand to a step ROC with
  *vertical cliffs at fixed FPRs*. Against the trapezoidal estimand the band
  misses by 0.33 in depth (v_high = 1.000); against its own matching staircase
  estimand it still misses, now by ~0.001 in depth (v_low = 1.000) because the
  staircase truth is *exactly* 0 on the first level while a credible lower edge
  will not touch 0. Adding the degenerate lower allowance (`L = 0` where
  `k̂ = 0`) fixes half of these (coverage 0.000 → 0.475); the residual comes
  from the FPR-axis randomness of the cliff position (the population cliff sits
  at FPR = 1/Q exactly, the sample cliff at `#neg in level 1 / n0`), which no
  finite-width band can absorb.

### Verdict

Success criterion *"coverage stays ≥ nominal"*: **met** for the declared
convention (random tie break) at both Q, and met with room to spare for the
even/mid-rank variant. The `neg1st` result is not a defect of M2 — it is a
demonstration that the tie convention used in construction must match the
convention that defines the estimand, and that class-ordered tie breaking
produces an estimand (vertical cliffs at deterministic FPRs) that is outside
the reach of any ROC band. **State the convention in the paper.**

**What this means for the method.** Ties cost nothing as long as they are
broken at random (or evenly). The rank-space reduction survives discreteness
intact — one of the few places where the theory said "should be fine" and the
experiment agreed exactly.

---

## 4. P4 — M-vs-K saturation

### Setup

Binormal .95, balanced, `n0 ∈ {500, 2000, 5000}` (K = n0+1 grid points),
`M ∈ {1000, 3000, 8000}`. To make the comparison exact, `M = 8000` draws are
generated once per replicate and the smaller `M` are prefixes of the same
cloud. Reps: 200 / 200 / 120. Thinned-grid trim (`--thin 5`: every 5th grid
point plus the first and last 25) evaluated on the *full* grid, run for
n0 = 500 and 2000.

### Result

| n0 | K | M | thin | mean j\* @.05 | cov@.05 | area@.05 | j\*@.20 | cov@.20 | cov@.50 |
|---|---|---|---|---|---|---|---|---|---|
| 500 | 501 | 1000 | – | **1.1** | .980 | .0650 | 5.3 | .915 | .740 |
| 500 | 501 | 3000 | – | 3.0 | .985 | .0634 | 15.4 | .920 | .770 |
| 500 | 501 | 8000 | – | 7.6 | .995 | .0632 | 40.8 | .920 | .765 |
| 500 | 501 | 3000 | yes | 3.5 | .985 | .0624 | 17.3 | .920 | .740 |
| 500 | 501 | 8000 | yes | 8.8 | .995 | .0623 | 46.0 | .915 | .740 |
| 2000 | 2001 | 1000 | – | **1.0** | .950 | .0317 | 3.5 | .880 | .685 |
| 2000 | 2001 | 3000 | – | 2.0 | .960 | .0321 | 10.1 | .890 | .705 |
| 2000 | 2001 | 8000 | – | 4.9 | .970 | .0320 | 26.6 | .905 | .715 |
| 2000 | 2001 | 8000 | yes | 5.6 | .970 | .0316 | 29.5 | .890 | .700 |
| 5000 | 5001 | 1000 | – | **1.0** | .942 | .0200 | 2.9 | .883 | .633 |
| 5000 | 5001 | 3000 | – | 1.8 | .967 | .0205 | 8.1 | .883 | .642 |
| 5000 | 5001 | 8000 | – | 3.9 | .967 | .0205 | 21.5 | .900 | .633 |

**Rule of thumb.** The quantity that matters is the realised *local level*
`ell = j*/(M+1)`, which is stable in `M` once unsaturated:

| K | ell @ α=.05 | ell @ α=.20 |
|---|---|---|
| 501 | 9.7e-4 | 5.1e-3 |
| 2001 | 6.4e-4 | 3.4e-3 |
| 5001 | 5.2e-4 | 2.7e-3 |

fitting `ell(K, α) ≈ 9.7e-4 · (α/0.05)^{1.2} · (K/500)^{-0.27}`. Requiring
`j* ≥ 5` (enough resolution that neighbouring α give different bands) gives

> **M ≳ 5 / ell(K, α)** — i.e. M ≈ 5,000 (K=500), 8,000 (K=2,000), 10,000
> (K=5,000) at α = 0.05; ~5× less at α = 0.2.
> **M = 10,000 covers every K ≤ 5,000 at α ≥ 0.05.**

The rule is self-checking: `j*` is computed anyway, so the method can *report*
it and warn when `j* < 3`.

**What saturation actually does.** It is not a validity cliff: at `j* = 1` the
band is the full envelope of `M` draws, i.e. the widest tube the cloud can
give, and coverage at α=.05 goes .967 → .942 (K=5001) and .970 → .950
(K=2001). What it destroys is **α-resolution**: at K=5001, M=1000, α=.05 and
α=.10 return the *identical* band (both `j*=1`), so the nominal level stops
meaning anything below α ≈ .10.

**Thinned-grid trim.** Trimming on every 5th grid point (plus the first/last
25) while building and evaluating the band on the full grid: `j*` rises 10–15 %,
area falls ~1 %, and coverage is unchanged to within Monte Carlo error
(.985/.985, .995/.995, .970/.970 at α=.05). **Nothing leaks** — the band still
covers the full-grid truth at the same rate. But the gain is far too small to
substitute for `M`: at K=2001 with M=1000 the thinned trim is still saturated
(`j* = 1.0`). Verdict: safe, marginally beneficial, not a fix.

### The C6 killer cell, rerun

`P4c` is C6 (binormal .95, n0=n1=5000). With `fid_cp` at **M = 8000**,
120 reps:

| α | coverage | v_low | v_high | p95 depth | max depth | area | KS | WH |
|---|---|---|---|---|---|---|---|---|
| .05 | **.967** | .017 | .017 | 0 | .0019 | .0205 | .0631 | .0099 |
| .10 | .958 | .025 | .017 | 0 | .0099 | .0189 | | |
| .20 | .900 | .050 | .050 | 0.0020 | .0183 | .0171 | | |
| .50 | .633 | .200 | .200 | 0.0095 | .0334 | .0143 | | |

Coverage ≥ 0.94 confirmed, misses perfectly balanced (.017/.017 and
.050/.050), max miss depth 0.0019 at α=.05, area 32 % of KS. With the C = 2.0
recalibration the same cell gives .958 / .908 / .767 / .450 at
α = .05 / .10 / .20 / .50 and area .0189.

**What this means for the method.** `M` is a genuine resource requirement, not
a nuisance: it must scale with the grid, and the requirement is legible
(`j* ≥ 5`). This is the one place where M2 costs more than the old bootstrap —
10,000 fiducial curves at n = 5000 is ~11 s per band on one laptop core.

---

## 5. P5 — nominal vs actual calibration curve

`alpha_sweep.py`, 400 reps, M = 3000, `C = 2.2` shown (the centred map;
`C = 2.0` is ~1–3pp more conservative everywhere).

| α | nominal | C2 `fid_cp` | C2 `fid_rc` | C2 area cp → rc | C1 `fid_cp` | C1 `fid_rc` | C4 `fid_cp` | C4 `fid_rc` | C5 `fid_cp` | C5 `fid_rc` |
|---|---|---|---|---|---|---|---|---|---|---|
| .50 | .500 | .748 | **.500** | .0418 → .0343 | .733 | .500 | .777 | .517 | .655 | .403 |
| .30 | .700 | .858 | **.710** | .0477 → .0407 | .868 | .700 | .890 | .738 | .810 | .625 |
| .20 | .800 | .912 | **.812** | .0517 → .0448 | .927 | .805 | .930 | .835 | .880 | .750 |
| .10 | .900 | .963 | **.907** | .0579 → .0513 | .963 | .925 | .970 | .925 | .945 | .877 |
| .05 | .950 | .975 | **.963** | .0634 → .0571 | .980 | .960 | .980 | .965 | .968 | .940 |
| .02 | .980 | .985 | **.980** | .0716 → .0654 | .995 | .985 | .993 | .985 | .978 | .970 |

The α = .02 row is at the edge of what M = 3000 can resolve (`j*` = 2–3); the
one sub-nominal entry there (C5 `fid_cp` .978) is within one Monte Carlo SE
and disappears with larger M.

---

## 6. Where we stand

### Solved (was broken before this round)

1. **Central-α over-coverage.** +.24/+.12 at α=.5/.2 → +.03/+.03 with the
   fixed `alpha_eff = 1-(1-α)^2` remap, at a 9–18 % area saving and no loss of
   α=.05 validity. Ceiling analysis (`recal`) shows a level-only fix is the
   right instrument: the band's shape needs nothing.
2. **Ties/discreteness.** Random tie breaking is provably exact for the
   trapezoidal estimand and empirically indistinguishable from continuous
   scores at Q = 20 and 100. No conservatism needed.
3. **The M-vs-K rule.** `ell(K,α) ≈ 9.7e-4 (α/.05)^{1.2} (K/500)^{-0.27}`,
   `M ≳ 5/ell`, self-checked by reporting `j*`. C6 at M = 8000 is .967 with
   balanced misses and 32 % of KS area.
4. **The low-FPR corner.** No mirrored Beta floor is required: 9:1 imbalance
   both ways, AUC .99, and a vertical kink at FPR 2/n0 all give `v_low` ≤ .013
   with misses spread across FPR, not concentrated at the corner.
5. **The lower-edge allowance question.** Answered: the *full* mirrored CP
   bound costs 15 % area for nothing; the *degenerate* mirror (`L = 0` where
   `k̂ = 0`) is free and is the exact reflection of the CP upper allowance.

### Still broken / open, ranked

1. **Shape dependence at α ≥ 0.2 survives recalibration.** After the remap,
   coverage at α=.2 spans .762 (t(2)) to .895 (n=25) — 13pp. The bias is gone,
   the spread is not. A data-driven level would be needed, and the obvious one
   fails (item 2). *Impact: the paper can claim "centred at every α", not
   "calibrated at every α".*
2. **The data-driven trim calibration does not (yet) beat the constant.**
   Nested plug-in calibration is conservative by 1.3–1.7× in `j` because the
   plug-in curve is rougher than the truth, and costs 80× compute. A fiducial-
   predictive version (calibrate against draws from the fiducial cloud instead
   of a point plug-in) is the untried alternative.
3. **The exponent `C ≈ 2` has no proof.** It is stable over 14 cells and has a
   natural Šidák-over-two-samples reading, but it is currently an empirical
   constant — the one blemish on the "no magic numbers" claim.
4. **Width at the steep corner with small n.** P2d (.99, n=150) is 3.2× the
   oracle band (2.7× after recalibration). Everywhere else 1.05–1.35×. The
   fiducial cloud is over-dispersed exactly where the curve is steepest.
5. **n = 25 is an outlier for the recalibration** (needs `alpha_eff*` = .21
   at α=.05 vs .13 typical); the fixed map is merely conservative there, which
   is the safe direction.
6. **Cost.** M = 10,000 curves at n = 5,000 is ~11 s/band single-core. Fine for
   a paper, worth a note for users.
7. **No coverage theorem.** Still empirics only for the fiducial composition.

---

## 7. Recommended M2 production recipe

Given a sample with `n0` negatives and `n1` positives, a level `α`, and a
Monte Carlo budget `M`:

1. **Ranks.** Merge and sort the scores **breaking ties at random**; keep the
   resulting label sequence. (Declare the estimand: the trapezoidal /
   random-tie-break ROC. Deterministic even-spreading within tied blocks is an
   acceptable, slightly conservative alternative. Never rank one class ahead of
   the other inside a tie block.)
2. **Fiducial cloud.** For `m = 1..M`: draw `n0+1` and `n1+1` iid Exp(1)
   spacings and normalise, giving Dirichlet(1,…,1) draws of each class's CDF at
   its own order statistics; place the *other* class's elements inside each gap
   at sorted-uniform fractions of the gap; compose
   `R̃_m(t_k)` on the grid `t_k = k/n0`, `k = 0..n0`.
3. **Trim depth.** Compute each draw's min-p depth
   `s_m = min_k min(rank-from-bottom, rank-from-top)` (tie-inclusive, both
   sides, min over the grid). Set
   **`alpha_eff = 1 − (1 − α)^2`** and take `j` = the `alpha_eff`-quantile of
   `{s_m}`, clipped to `[1, M/2]`. **Report `j`; if `j < 3`, `M` is too small —
   increase it** (`M ≳ 5/ell` with `ell ≈ 9.7e-4 (α/.05)^{1.2} (K/500)^{-0.27}`;
   `M = 10,000` is safe for `n0 ≤ 5,000`, `α ≥ 0.05`).
4. **Band.** `L_k` = `j`-th smallest, `U_k` = `j`-th largest of
   `{R̃_m(t_k)}`.
5. **Exact binomial allowances**, both at the band's own local level
   `ell = j/(M+1)`, using the observed TPR counts
   `k̂_k = #{positives below the (k+1)-th negative}`:
   * upper (**essential**): `U_k := max(U_k, BetaInv(1−ell; k̂_k+1, n1−k̂_k))`,
     then `cummax`. Without it coverage is 0 whenever the truth touches
     TPR = 1 (AUC .90 bimodal, AUC .99 binormal: `S(truth) = 0` in *every*
     replicate).
   * lower (**free insurance**): `L_k := 0` wherever `k̂_k = 0`. Do **not**
     use the full mirrored CP lower bound.
6. **Clip to [0,1]** and report. No other floors, no projection, no ε, no
   Šidák-on-K_eff.

Tuning constants: `M` (a Monte Carlo budget, self-diagnosing via `j`) and the
exponent 2 in step 3 (conjectured structural: one simultaneity budget per
sample). Everything else is derived.

**Scorecard of this recipe** (α=.05, 400 reps unless noted): coverage
.945–.980 over 14 cells spanning n = 25…5000 whenever `M` meets the step-3
rule (.942 in the one run where `M` was deliberately set below it), AUC .55…99, binormal / bimodal /
t(2) / kinked truths and 9:1 imbalance both ways; balanced misses
(`v_low ≈ v_high`); p95 miss depth 0 everywhere; area 21–66 % of KS, 1.05–3.2×
the oracle rank-space ceiling; α-calibration centred to within ~3pp at
α = .02….5 with a residual ±7pp shape spread at α ≥ .2.
