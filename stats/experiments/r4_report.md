# Round 4: the last calibration idea, a rank-computable roughness functional, exact tests at named curves, and two round-3 loose ends

*Laptop CPU run, 2026-08-23. Code: `r4_experiments.py` (all six
sub-experiments), `r4_analyze.py` (tables). Results: `res_r4_*.json`; logs
`log_r4_*.txt`. This report continues `m2_report.md` (round 2) and
`m3m4_report.md` (round 3) and uses their published numbers as the comparison
columns. Nothing here touches the scope of `stats/c_calibration_spec.md` (the
C(n,α)/taper map, its D1–D6 decisions, the α=.05 arm of the C\*(n) ladder at
n ≥ 10⁴, the trapezoid floor conjecture, the shape-library envelope); where a
result bears on one of those decisions it is flagged and left there.*

## 0. What was run

```
cd stats/experiments
U="uv run --project /home/nathan/Documents/studroc_paper python"

# P1  fiducial-predictive trim calibration (2 processes, 6 threads each)
$U r4_experiments.py --exp fpcal --cells C2 --reps 100 --M 3000 --ncal 60 \
      --min 1000 --arms raw sm --alphas 0.5 0.2 0.05 --threads 6 --seed 41 \
      --out res_r4_fpcal_C2.json
$U r4_experiments.py --exp fpcal --cells C5 ... --out res_r4_fpcal_C5.json

# P2  rank-computable roughness functionals, 14 cells
$U r4_experiments.py --exp rough --cells F_b70 F_b80 F_b90 F_b95 F_b99 \
      F_t295 F_bim90 H_b55 H_kink H_imb91 H_imb19 N_b95_150 N_b95_2000 \
      N_b90_25 --reps 200 --M 3000 --alphas 0.5 0.2 0.1 0.05 --threads 6 \
      --seed 41 --out res_r4_rough.json

# P3  exact Monte Carlo test at a named curve
$U r4_experiments.py --exp exact --cells b95_n150 b95_n500 t295_n500 \
      bim90_n500 b99_n500 --reps 20000 --BA 4000 --BN 8000 \
      --alphas 0.2 0.05 --aucs 0.93 0.94 0.96 0.97 \
      --kappas 0.6 0.8 1.25 1.6 --t1 0.05 --threads 6 --seed 41 \
      --out res_r4_exact.json

# P4  M3 nominal->actual map, all 14 cells on a refined level ladder
$U r4_experiments.py --exp m3grid --cells C1 C2 C3 C4 C5 C7 P2a P2b P2c P2d \
      P2e P2f P4b P4c --reps 400 --B 100000 --threads 6 --seed 7 \
      --out res_r4_m3grid.json

# P5  steep-corner pointwise repair, and why it does not bite
$U r4_experiments.py --exp repair --cells P2c P2d C2 --reps 400 --M 3000 \
      --alphas 0.05 0.2 --a2frac 0.1 0.25 --kc 10 25 0.02 --threads 6 \
      --seed 41 --out res_r4_repair.json
$U r4_experiments.py --exp corner --cells P2c P2d C2 --reps 100 --M 3000 \
      --alphas 0.05 0.2 --kc 25 --threads 6 --seed 41 \
      --out res_r4_corner.json

$U r4_analyze.py fpcal  res_r4_fpcal_C2.json res_r4_fpcal_C5.json
$U r4_analyze.py rough  res_r4_rough.json
$U r4_analyze.py m3grid res_r4_m3grid.json
$U r4_analyze.py repair res_r4_repair.json
```

Cells are those of `m2_report.md` §0. Every cell is seeded by the harness
convention `seed + sum(ord(c) for c in cellname)`; P4 uses `--seed 7` so that
the eight round-3 cells reproduce `res_m3_p1grid.json` exactly (they do — see
§4). P1/P2/P3/P5 use `--seed 41`, i.e. fresh data independent of every
published round.

**Monte Carlo error, stated once.** P1: 100 reps → coverage SE 2.2pp at the
95% level, 4.0pp at 80%, 5.0pp at 50%. P2: 200 reps → 1.5 / 2.8 / 3.5pp, and
the published `ae*` targets carry ≈2.3–4pp of their own. P3: 20,000 test
datasets against a *shared* 8,000-draw null calibration sample, so the
rejection-rate SE is `sqrt(α(1−α)(1/20000 + 1/8000))` = 0.53pp at α=.2 and
0.29pp at α=.05 (the shared sample dominates). P4: 400 reps → 1.1pp at 95%,
2.5pp at 50%, plus the granularity of the nominal-level ladder. P5: 400 reps
for the coverage/area probe, 100 for the corner diagnostic. Differences inside
2–3pp are not to be read anywhere below.

---

## 1. P1 — fiducial-predictive trim calibration

### Setup

The last live calibration idea of `next_method_ideas.md` §7. Per replicate:

1. draw `ncal = 60` candidate curves `R̃` from a *fresh* fiducial cloud built
   from the same label sequence (the fiducial predictive law; a fresh cloud
   rather than rows of the band's own cloud, so nothing is reused);
2. for each candidate, simulate **one** rank-space dataset from it
   (`n0` uniforms, `n1` draws with CDF `R̃` via the exact generalized inverse
   `m4.sample_curve`), build an inner cloud of `m_in = 1000` fiducial draws
   and the *whole* production band ladder (CP upper allowance included), and
   record whether the band at each depth covers `R̃`;
3. average the coverage indicator over candidates and take the largest depth
   whose averaged coverage reaches `1−α` (`j_thresh`); a second read-out
   `j_quant` is the α-quantile of the candidates' own min-p depths;
4. rescale to the outer `M = 3000` by the local level and evaluate the
   resulting band against the *true* curve.

Two arms: `raw` (candidates as drawn) and `sm` (each candidate monotone
moving-average smoothed, window `sqrt(n0)` = 23 — **this arm carries a tuning
constant** and is reported as a diagnostic, not a proposal). Cells C2
(binormal .95) and C5 (t(2) .95), both n=500/500, 100 reps, α ∈ {.5, .2, .05}.
Runtime 1251 s (C2) and 1245 s (C5) for both arms, i.e. ≈6.3 s per replicate
per arm against ≈0.3 s for the band itself — a ≈20× wall-time multiplier per
arm. The inner draw budget (`ncal × m_in` = 60,000 draws) is 0.75× the plug-in
arm's in `m2_report.md` §1c (80 × 1000), so the two are directly comparable in
cost.

### Result — C2 (binormal .95, n=500), 100 reps

| arm | cov@.5 | area | mean j | cov@.2 | area | mean j | cov@.05 | area | mean j |
|---|---|---|---|---|---|---|---|---|---|
| `fid_cp` (C=1) | .770 | .0420 | 56.4 | .930 | .0520 | 15.4 | .980 | .0638 | 3.0 |
| `fid_rc` (C=2, production) | .530 | .0346 | 132.1 | .850 | .0450 | 38.5 | .970 | .0578 | 7.0 |
| **`fid_pred_raw`** | **.690** | .0399 | **75.0** | **.920** | .0504 | **20.6** | **.970** | .0618 | **4.5** |
| `fid_pred_raw_q` | .710 | .0411 | 64.8 | .940 | .0519 | 16.6 | .980 | .0628 | 3.6 |
| `fid_pred_sm` | .690 | .0389 | 83.9 | .930 | .0492 | 24.5 | .980 | .0604 | 5.8 |
| `fid_pred_sm_q` | .730 | .0400 | 73.9 | .930 | .0504 | 21.0 | .980 | .0614 | 5.0 |
| `recal` (ceiling, this run) | .500 | .0335 | 148.1 | .800 | .0435 | 46.8 | .950 | .0522 | 14.9 |
| `ae*` (this run) | .815 | | | .445 | | | .195 | | |

Conservatism in the trim depth, as ceiling/calibrated:

| α | `fid_pred_raw` | `fid_pred_sm` | `fid_cp` (C=1) | `fid_rc` (C=2) | published plug-in `fid_cal` |
|---|---|---|---|---|---|
| .50 | **1.98×** | 1.76× | 2.62× | 1.12× | 1.27× |
| .20 | **2.28×** | 1.91× | 3.04× | 1.22× | 1.27× |
| .05 | 3.34× | 2.55× | 4.93× | 2.15× | not resolvable (0.75× on C2, 2.7× on C5) |

### Result — C5 (t(2) .95, n=500), 100 reps

| arm | cov@.5 | area | j | cov@.2 | area | j | cov@.05 | area | j |
|---|---|---|---|---|---|---|---|---|---|
| `fid_cp` (C=1) | .680 | .0506 | 68.4 | .860 | .0630 | 19.0 | .940 | .0772 | 3.8 |
| `fid_rc` (C=2) | .420 | .0412 | 158.6 | .730 | .0545 | 46.8 | .920 | .0700 | 8.7 |
| **`fid_pred_raw`** | **.670** | .0511 | **67.2** | **.860** | .0671 | **14.2** | **.940** | .0783 | **3.5** |
| `fid_pred_sm` | .600 | .0472 | 95.4 | .870 | .0619 | 23.8 | .920 | .0770 | 4.4 |
| `recal` (ceiling) | .500 | .0448 | 116.2 | .810 | .0588 | 29.4 | .950 | .0822 | 2.2 |
| `ae*` (this run) | .675 | | | .280 | | | .030 | | |

Conservatism: `fid_pred_raw` 1.73× / 2.07× and `fid_pred_sm` 1.22× / 1.24× at
α=.5/.2, against `fid_cp` 1.70× / 1.55× and the published plug-in 1.27×/1.27×.

### The mechanism, measured

Depth contrast (a curve's own min-p depth against the depth distribution of
the cloud it is compared to; ratios, so the different M scales cancel):

| quantile | C2 truth vs outer cloud | C2 raw candidate vs its inner cloud | C2 smoothed candidate | C5 truth | C5 raw candidate | C5 smoothed |
|---|---|---|---|---|---|---|
| q05 | **2.94×** | **0.00×** | 0.92× | **0.51×** | **0.00×** | 0.00× |
| q50 | 1.01× | 1.19× | 1.26× | 1.58× | 0.92× | 1.26× |

The lower tail is where the trim lives, and it is exactly where the calibration
target is wrong: at the 5% quantile the *truth* sits about 3× deeper in its
cloud than a draw does (C2), while a *candidate* sits at depth 0 — i.e. at
least 5% of candidates fall entirely outside their own inner cloud at some grid
point. Fiducial draws carry the interpolation/spacings roughness at scale 1/n
that the smooth truth does not, and they carry *more* of it than the Hazen
plug-in curve does (which is why the predictive calibration is worse than the
plug-in, not better). At the median the contrast is fine — which is why the
α=.5 damage is smaller than the α=.05 damage.

### Verdict vs the stated bar

> *"A clean negative here CLOSES the last calibration idea in the backlog."*

**Clean negative.**

- The raw fiducial-predictive calibration is **1.7–2.3× more conservative in
  the trim depth than the ceiling at central α**, against the plug-in's
  1.27× — i.e. it is *worse than the plug-in it was designed to replace*, at
  comparable inner compute. Its coverage removes only about a quarter of C=1's
  over-coverage on C2 (.770 → .690 against nominal .500) and none of it on C5
  (.680 → .670).
- The smoothed-candidate variant lands at 1.22–1.91×, i.e. **the plug-in's
  performance and no better** — the identical outcome round 3 got for the
  smoothed M4b bracket (1.25–1.79×) — and it needs a window constant.
- Fixed C=2 remains far closer to the ceiling (1.12–1.22× in j at central α)
  than any data-driven arm measured in rounds 2, 3 or 4.
- The mechanism is not a budget artifact: it is the roughness contrast, now
  measured directly, and it is *larger* for fiducial draws than for plug-in
  curves.

Caveats. (i) The α=.05 column is doubly resolution-limited — the optimal inner
depth is ≈5 on the `m_in = 1000` scale, and the averaged coverage has
granularity 1/60 = 1.7pp, so "cov ≥ .95" means "≤3 failures out of 60". Read
α=.2 and α=.5 as load-bearing, exactly as in round 3. (ii) Taking the *largest*
depth whose noisy averaged coverage clears the target biases `j` **upward**
(anti-conservatively), so the measured conservatism is if anything
understated. (iii) The comparison to the published plug-in numbers crosses
runs (different seeds, `ncal = 80` there); the within-run `recal` ceiling used
for the ratios above is itself a 100-rep estimate (this run's C2 ae\*@.5 = .815
vs the published .780), so treat the ratios as ±15% and the *ordering*
(predictive worse than plug-in worse than fixed C=2) as the finding.

---

## 2. P2 — a rank-computable roughness functional

### Setup

Targets are the published per-cell calibration ceilings `ae*` — for the seven
fitting shapes from `res_m4_family.json` (300 reps), for the held-out and
n-axis cells from `res_p2_*.json` / `res_p4_ab.json` (400 reps) — read from
the JSONs, not retyped. Functionals and the evaluation come from **fresh**
200-rep runs (seed 41, M=3000), so the scoring is not coupled to the target
noise. Fourteen cells, split *before* any fitting:

* **fitting (7)**, all n=500/500: binormal .70/.80/.90/.95/.99, t(2) .95,
  bimodal .90;
* **held out (4)**: binormal .55 (P2e), kink (P2f), 900/100 (P2a),
  100/900 (P2b);
* **n axis (3)**: binormal .95 at n=150 (C3) and n=2000 (P4b), binormal .90
  at n=25 (C7).

Consistency check on the fresh runs: own `ae*` vs published, α=.5, over the 14
cells — max discrepancy 8pp (H_imb91 .710 vs .790), typical 3–5pp, against a
combined MC error of ≈5pp. Nothing is out of line.

Candidate functionals (all computed from the merged label sequence plus the
band's own cloud): label-run statistics (`runs_z`, run-length entropy
`run_ent`, `run_maxlen`); windowed local-slope statistics on windows of
`sqrt(n0)` grid points, computed on the raw staircase and on the Hazen plug-in
(`slope_sd`, `slope_logsd`, `slope_max`, `slope_curv`, and the *upward* slope
variation `slope_up`, `slope_upmag`, `slope_nup`); the concavity defect against
the curve's own least concave majorant (`lcm_gap`, `lcm_max`); the plug-in
depth contrast (`dc05_*`, `dc50_*`, `S_*`); and rank-path crossing counts
through the cloud (`xing_plug`, `xing_lcm`, `xing_draw`, and the ratios) — the
last group being the direct geometric proxy for the "effective independent
looks" ratio that the erosion law of `fiducial_band_theory.md` §7.1 says *is*
C\*.

### Result 2a — correlations and honest model selection

Best correlates of `C* = log(1−ae*)/log(1−α)` over the seven fitting cells,
with leave-one-cell-out RMSE of the one-predictor linear fit (null model =
predict the fitting-set mean; `C*` range 1.67–2.22 at α=.5):

| functional | r@.5 | r@.2 | r@.05 | LOO@.5 | LOO@.2 |
|---|---|---|---|---|---|
| `dc05_lcm` | +0.82 | +0.74 | +0.79 | **0.127** | 0.327 |
| `run_ent` | −0.77 | −0.87 | −0.85 | 0.169 | **0.126** |
| `lcm_max_plug` | −0.79 | −0.83 | −0.90 | 0.163 | 0.513 |
| `xing_draw` | +0.76 | +0.69 | +0.62 | 0.168 | 0.204 |
| `slope_sd_emp` | −0.73 | −0.71 | −0.65 | 0.172 | 0.196 |
| `slope_max_emp` | −0.72 | −0.70 | −0.65 | 0.174 | 0.198 |
| *null (constant)* | — | — | — | *0.193* | *0.199* |

So the information is real but thin: the best single functional cuts the LOO
RMSE of `C*` from 0.193 to 0.127–0.17 at α=.5 and from 0.199 to 0.126 at
α=.2, and `dc05_lcm`'s advantage is carried almost entirely by one cell (t(2),
where the plug-in's least concave majorant sits at min-p depth 16 against
194–1050 everywhere else — it is a t(2) detector, not a gradient).

### Result 2b — out-of-sample scoring of a functional-driven level rule

Rule: `C* = a + b·f` fitted on the seven fitting cells, then applied (i) with
the cell-mean functional and (ii) **per replicate**, with `C` clamped to
[1, 5], and scored against the fresh coverage tables. Coverage spread
(max − min) over the four held-out cells and over all fourteen:

| rule | α | held-out: fixed C=2 | rule (cell mean) | rule (per rep) | all 14: C=2 | rule (cell mean) | rule (per rep) |
|---|---|---|---|---|---|---|---|
| `run_ent` | .50 | 8.5pp | 10.5pp | 8.5pp | 20.0pp | 15.0pp | 13.0pp |
| `run_ent` | .20 | 4.5pp | 4.5pp | 4.5pp | 14.0pp | 11.5pp | 10.5pp |
| `run_ent` | .05 | 4.5pp | 6.5pp | 6.5pp | 5.5pp | 6.5pp | 6.5pp |
| `slope_sd_emp` | .50 | 8.5pp | 7.5pp | 7.5pp | 20.0pp | 16.5pp | 17.0pp |
| `slope_sd_emp` | .20 | 4.5pp | 3.5pp | 3.0pp | 14.0pp | 12.5pp | 13.0pp |
| `slope_sd_emp` | .05 | 4.5pp | 5.5pp | 5.5pp | 5.5pp | 6.0pp | 6.0pp |
| `xing_draw` | .50 | 8.5pp | 6.5pp | 7.0pp | 20.0pp | **30.5pp** | 31.0pp |
| `slope_max_emp` | .50 | 8.5pp | 8.0pp | 7.5pp | 20.0pp | 16.0pp | 15.5pp |
| `slope_max_emp` | .20 | 4.5pp | 3.0pp | 3.0pp | 14.0pp | 11.5pp | 12.0pp |

Two-predictor rules (the "corner steepness + concavity defect" hypothesis)
fit the seven cells much better and generalize much worse. The best LOO pair
(`slope_upmag_emp` + `xing_ratio_lcm`, LOO@.5 = 0.086 against the null's
0.193) produces, on held-out cells at α=.5, a spread of **44.5pp** (0.140 to
0.585) because it extrapolates `Ĉ = 5.0` on the 9:1 imbalance cell. The
second-best pair (`run_ent` + `slope_curv_emp`) is neutral: held-out 10.0pp vs
C=2's 8.5pp.

### Verdict vs the stated bar

> *"Reduces the coverage spread at α=.2/.5 below the fixed-map's ±13–19pp on
> held-out cells without dropping α=.05 coverage below ~0.94."*

**Not met.** Three separate reasons, each worth recording:

1. **On held-out cells no rule beats the fixed map beyond noise.** The best
   held-out improvements are 8.5 → 6.5pp at α=.5 and 4.5 → 3.0pp at α=.2, both
   far inside the ±3.5pp/±2.8pp per-cell MC error (the spread of four cells has
   an MC component of roughly ±5pp at α=.5). The apparent all-14 improvements
   (20 → 13pp) are half in-sample.
2. **Every rule that moves anything at α=.05 costs validity.** Under the
   `run_ent` rule, α=.05 coverage lands at .910 (H_b55), .915 (F_b80, H_kink),
   .920 (F_b70), .930 (F_bim90), .935 (F_t295, N_b95_2000) — **seven** of the
   14 cells below the 0.94 floor, against **three** under C=2 (minimum .925,
   on t(2)). The α=.05 fit is dominated by
   the noise in the published `ae*@.05` targets (±4pp in `ae` is ±0.5–1.0 in
   `C`), which the fitted slope (−3.74 per unit of entropy, `Ĉ` spanning
   1.65–4.24) transcribes directly into the band.
3. **The held-out set as drawn does not even exhibit the ±13–19pp spread the
   bar refers to.** Under C=2 these four cells span 8.5pp at α=.5 and 4.5pp at
   α=.2; the ±13–19pp figure is a property of the *whole* 14-cell grid (here
   20pp / 14pp, consistent with `m2_report.md`). So the honest statement is
   that the bar is untestable on this held-out split and is failed on the split
   that does show the spread.

### Result 2c — three findings that are not about the rule

1. **The co-movement pathology does *not* bite here.** Within-cell Spearman
   correlation between the per-replicate functional and that replicate's
   realized truth depth is |ρ| ≤ 0.25 for essentially every functional × cell
   (two exceptions, `slope_logsd_plug` at 0.52 on C3 and 0.40 on C7). Per-rep
   and cell-mean versions of every rule give coverage agreeing to ≤1.5pp. So
   unlike plug-in *depth* calibration, a functional-driven *level* rule does
   not reintroduce the Wald-type bias — the obstruction is purely that the
   functionals do not carry enough shape information. This is a useful
   negative-of-a-negative: the failure mode that killed M1/`fid_cal`/M4b is not
   what stops this route.

2. **The "roughness axis" is not a concavity defect.** The obvious
   identification of round 3's axis — an inflection or hook in the truth —
   is falsified. Measured on the *true* curves: only t(2) .95 is
   non-concave (`lcm_gap` 0.0053, `lcm_max` **0.179**, upward slope variation
   0.0027). Bimodal .90 — the other shape sitting below the binormal ladder in
   `ae*` — is **exactly concave**, as are the kink and the whole binormal
   .55–.99 ladder. So the two off-family shapes that move `ae*` by 9–13pp do
   *not* share a concavity defect; whatever axis they share is still
   unidentified. (This kills the natural hypothesis that the operative feature
   is an inflection or hook in the truth.)

3. **The erosion law's "effective independent looks" ratio is not recovered by
   rank-path crossing counts — even in oracle form.** Counting median crossings
   of a curve's local-rank path through the cloud gives `K̂_d/K̂_t` (oracle
   numerator and denominator, using the *true* curve) with Pearson +0.69
   against `C*` over the seven fitting shapes at fixed n, but **−0.31 over all
   14** — it moves the wrong way along n (binormal .95: ratio 2.14 → 2.70 →
   2.95 at n = 150/500/2000 while `C*` falls 2.40 → 2.06 → 1.71), and the
   rank-form (plug-in) version has the *opposite sign* to the oracle version on
   the fitting cells (−0.54 vs +0.69). Two readings: the crossing count
   conflates decorrelation with grid size (`K = n0+1`), and min-p is governed
   by tail excursions rather than median crossings. Either way, the specific
   geometric proxy for assumption (A2) of §7.1 fails, and it fails *before*
   estimability is at issue.

Also recorded, across all 14 cells: the H2 depth contrast `S(truth)/q05(S(draw))`
runs 14–67× on plateau-free truths and collapses to 0.0–0.14 on the two
plateau-touching ones (binormal .99, bimodal .90), where the raw cloud misses
the truth in essentially every replicate and the CP allowance is what covers —
the `m2_report.md` §7 finding, reproduced as a by-product on a fresh 14-cell
grid. Those two cells' contrast numbers must not be read as roughness.

*Connection to `c_calibration_spec.md`, not pre-empting it:* D1 (the transfer
coordinate) and D5 (shape aggregation) are unaffected — this experiment says
only that no per-dataset shape functional in this battery earns a place in the
map, so the frozen map stays a function of (n0, n1, α) as specified. The
imbalance cell (H_imb91) being where the two-predictor rule explodes is a hint
for D2, no more.

---

## 3. P3 — exact Monte Carlo test at a named curve

### Setup

`H0: R = R0` is simple in rank space, so its whole law is exactly simulable
(`fiducial_band_theory.md` Prop. 2, consequence 2 — simulability). Statistic: the min-p depth `T` of the
observed empirical ROC in a **fixed, independent** cloud of `BA = 4,000`
empirical ROCs simulated from `R0`. Because that cloud is independent of the
data, `T` is a fixed measurable functional of the curve, and the null
distribution of `T` is obtained from a second independent sample of
`BN = 8,000` null draws. The Monte Carlo p-value
`p = (1 + #{T_null ≤ T_obs})/(BN+1)` is then exactly valid; the tie-randomized
version `p_rand = (#{T_null < T_obs} + U·(1 + #{T_null = T_obs}))/(BN+1)` is
exactly uniform under H0. Both are reported. 20,000 test datasets per row.

Local alternatives are confined to `t ≤ t1 = 0.05`:
`R_alt(t) = R0(t1)·(R0(t)/R0(t1))^κ` for `t ≤ t1`, identity above — monotone by
construction, exactly equal to the null at and above `t1`, so the deviation is
genuinely local; κ > 1 pushes the corner down.

### Result (i) — type-I error

| null curve | n0/n1 | T_null q05/q20/q50 | size@.2 (cons / rand) | size@.05 (cons / rand) |
|---|---|---|---|---|
| binormal .95 | 150/150 | 9 / 45 / 152 | .200 / **.202** | .046 / **.049** |
| binormal .95 | 500/500 | 5 / 28 / 98 | .194 / **.196** | .045 / **.048** |
| t(2) .95 | 500/500 | 5 / 32 / 115 | .188 / **.197** | .041 / **.048** |
| bimodal .90 | 500/500 | 5 / 28 / 98 | .200 / **.201** | .049 / **.051** |
| binormal .99 | 500/500 | 7 / 38 / 135 | .187 / **.190** | .045 / **.046** |

Exact within Monte Carlo error at every cell (SE 0.53pp at α=.2, 0.29pp at
α=.05; the largest deviation, binormal .99 at α=.2, is 1.9 SE). The
non-randomized version is conservative by 0–1pp, as the atoms of an
integer-valued depth statistic require. A supplementary size check at
n=150 with three independent seeds and 20,000 reps each gave .200–.204 at
α=.2 and .049–.055 at α=.05.

### Result (ii) — power

Rejection rate of the randomized test; `sup dev` is the sup-norm distance
from the null curve.

| null | n | alternative | AUC | sup dev | power@.2 | power@.05 |
|---|---|---|---|---|---|---|
| binormal .95 | 500 | AUC .93 | .930 | .095 | .891 | .710 |
| | | AUC .94 | .940 | .051 | .519 | .262 |
| | | AUC .96 | .960 | .060 | .529 | .230 |
| | | AUC .97 | .970 | .133 | .970 | .854 |
| | | local κ=1.6 | .947 | .128 | .526 | .268 |
| | | local κ=1.25 | .949 | .062 | .299 | .093 |
| | | local κ=0.8 | .951 | .061 | .210 | .047 |
| | | local κ=0.6 | .953 | .134 | .353 | .100 |
| binormal .95 | 150 | AUC .93 | .930 | .095 | .580 | .321 |
| | | AUC .94 | .940 | .051 | .341 | .132 |
| | | AUC .96 | .960 | .060 | .283 | .089 |
| | | AUC .97 | .970 | .133 | .622 | .324 |
| | | local κ=1.6 | .947 | .121 | .351 | .135 |
| | | local κ=0.8 | .951 | .050 | .184 | .044 |
| binormal .99 | 500 | AUC .93–.97 | — | .246–.452 | 1.000 | 1.000 |
| | | local κ=1.6 | .988 | .130 | .468 | .217 |
| | | local κ=0.8 | .991 | .050 | .195 | .048 |

Readings. (a) Power against a global AUC shift at n=500 is 0.23–0.26 at
|ΔAUC| = .01 and 0.71–0.85 at |ΔAUC| = .02, halving at n=150 — a usable
non-inferiority-test operating characteristic. (b) The test is markedly
**less** powerful against a *localized* early-FPR deviation of the same
sup-norm size: at n=500 a local perturbation with sup dev .128 gives power
.268, the same as a global AUC = .94 alternative whose sup dev is only .051 —
a 2.5× worse exchange rate. That is the min-p / equal-local-levels structure
showing through: the budget is spread across the grid, so power concentrates
on deviations that are themselves spread. (c) The direction is asymmetric:
pushing the corner *down* (κ=1.6, power .268) is easier to detect than pushing
it *up* (κ=0.6, sup dev .134, power .100). (d) On binormal .99 all global AUC
alternatives are detected with probability 1 because their sup deviations are
0.25–0.45; local alternatives behave as at .95.

### Verdict

**Delivered, positive.** The named-curve test is exact (within 2 SE at every
cell tested, including the two shapes on which Working–Hotelling has 0.000
coverage), costs two batched cloud simulations, and has a clean power profile.
It is a publishable deliverable in its own right and is the one piece of the
M4 test-inversion frame that is fully tractable: inverting it over a *family*
of named curves would give an exact confidence set — the cost is one null
simulation per hypothesized curve, which is why the band (a projection of the
set over all curves) remains out of reach.

Caveats: the p-value is exactly valid conditionally on cloud A (and hence
unconditionally), but the rejection *rates* tabulated above inherit the
`BN = 8,000` null sample's own fluctuation — that is what the SE formula in §0
accounts for, and it is the dominant term. The statistic is one choice among
many; no attempt was made to optimize power, and the `t1 = 0.05`
κ-parametrization of the local alternatives was chosen while looking at these
cells.

---

## 4. P4 — M3's worst-case-level probe

### Setup

`m3_experiments.run_m3grid` (unmodified, called with a refined level ladder:
round 3's 20 levels plus {.975, .925, .85, .75, .65, .55, .45, .35}), `sidak`
split, 400 reps, all 14 cells including the six round 3 did not cover (P2e
binormal .55, P2f kink, P2a 900/100, P2b 100/900, C7 n=25, P4c n=5000). Seed 7,
so the eight round-3 cells reproduce `res_m3_p1grid.json`: they do, exactly
(C1 .820 at α′=.95, C2 .843, C3 .877, C4 .875, C5 .755, P2c .860, P2d .877,
P4b .748 — identical to `m3m4_report.md` §2).

### Result — the nominal→actual map, extended

Largest nominal α′ whose realized coverage still reaches a target:

| cell | truth | n0/n1 | α′ for cov ≥ .95 | ≥ .80 | ≥ .50 |
|---|---|---|---|---|---|
| C7 | binormal .90 | 25/25 | 0.850 | 0.990 | 0.999 |
| P2d | binormal .99 | 150/150 | 0.850 | 0.990 | 0.999 |
| C3 | binormal .95 | 150/150 | 0.800 | 0.975 | 0.999 |
| C1 | binormal .75 | 500/500 | 0.800 | 0.950 | 0.990 |
| C2 | binormal .95 | 500/500 | 0.750 | 0.950 | 0.999 |
| C4 | bimodal .90 | 500/500 | 0.800 | 0.975 | 0.999 |
| C5 | t(2) .95 | 500/500 | 0.650 | 0.925 | 0.990 |
| **P2e** | binormal .55 | 500/500 | **0.800** | 0.950 | 0.990 |
| **P2f** | kink | 500/500 | **0.800** | 0.950 | 0.999 |
| P2c | binormal .99 | 500/500 | 0.800 | 0.975 | 0.999 |
| **P2a** | binormal .90 | 900/100 | **0.500** | **0.850** | 0.990 |
| **P2b** | binormal .90 | 100/900 | **0.750** | 0.925 | 0.999 |
| P4b | binormal .95 | 2000/2000 | 0.650 | 0.925 | 0.990 |
| **P4c** | binormal .95 | 5000/5000 | **0.600** | 0.925 | 0.990 |

(P4c's ELL calibration uses `B = 25,000` order-statistic draws at n=5000 — the
harness's automatic reduction for n0 > 700 — so it is the coarsest calibration
in the table, and coarse in the conservative direction.)

**Infimum over shapes, by sample size** (min(n0,n1) as the size coordinate):

| min(n0,n1) | cells | inf α′ for .95 | inf α′ for .80 |
|---|---|---|---|
| 25 | C7 | 0.850 | 0.990 |
| 100 | P2a, P2b | **0.500** | **0.850** |
| 150 | P2d, C3 | 0.800 | 0.975 |
| 500 | C1,C2,C4,C5,P2e,P2f,P2c | 0.650 | 0.925 |
| 2000 | P4b | 0.650 | 0.925 |
| 5000 | P4c | 0.600 | 0.925 |
| **all measured cells** | | **0.500** | **0.850** |

At *fixed shape* the drift with n is clean and monotone over a 33× range:
binormal .95 needs α′ = 0.800 (n=150) → 0.750 (500) → 0.650 (2000) → 0.600
(5000); binormal .99 needs 0.850 (n=150) → 0.800 (500). Round 3's coarse
ladder read this as 0.7–0.8 at n ≤ 500 → 0.6 at n = 2000; the refined ladder
confirms it and extends it. The decline is close to linear in log n at
≈ 0.13 per decade.

**A fixed nominal α′ applied to every cell** (worst case over the measured
library; area ratios against the published `fid_cp` (C=1) and `fid_rc` (C=2)
areas at α=.05, recomputed from the round-2 `by_ae` tables):

| α′ | min coverage | cells below .95 | max area / `fid_rc` | max area / `fid_cp` | mean area / `fid_rc` |
|---|---|---|---|---|---|
| 0.85 | .835 | 12 of 14 | 1.13 | 1.00 | 0.99 |
| 0.75 | .870 | C5, P2a, P4b, P4c | 1.19 | 1.06 | 1.06 |
| 0.65 | .912 | P2a, P4c | 1.34 | 1.19 | 1.13 |
| 0.60 | .932 | P2a | 1.37 | 1.22 | 1.15 |
| 0.55 | .940 | P2a | 1.40 | 1.24 | 1.18 |
| **0.50** | **.950** | **none** | **1.43** | **1.27** | **1.21** |
| 0.45 | .965 | none | 1.46 | 1.29 | 1.24 |
| 0.40 | .975 | none | 1.57 | 1.40 | 1.28 |
| 0.35 | .978 | none | 1.61 | 1.43 | 1.31 |
| 0.20 | .990 | none | 1.76 | 1.56 | 1.43 |

Per-cell at the inf-over-shapes level α′ = 0.5:

| cell | realized cov | M3 area | `fid_cp` | `fid_rc` | ×cp | ×rc |
|---|---|---|---|---|---|---|
| C7 | .993 | .3811 | .3453 | .3125 | 1.10 | 1.22 |
| P2d | .995 | .0777 | .0613 | .0544 | 1.27 | **1.43** |
| C3 | .998 | .1302 | .1137 | .1029 | 1.14 | 1.26 |
| C1 | .993 | .1508 | .1394 | .1272 | 1.08 | 1.19 |
| C2 | .988 | .0705 | .0634 | .0579 | 1.11 | 1.22 |
| C4 | .993 | .0962 | .0856 | .0787 | 1.12 | 1.22 |
| C5 | .980 | .0758 | .0775 | .0709 | 0.98 | 1.07 |
| P2e | .993 | .1771 | .1625 | .1498 | 1.09 | 1.18 |
| P2f | .975 | .1270 | .1193 | .1090 | 1.06 | 1.16 |
| P2c | .980 | .0351 | .0288 | .0259 | 1.22 | 1.36 |
| P2a | **.950** | .1500 | .1467 | .1339 | 1.02 | 1.12 |
| P2b | .990 | .1472 | .1423 | .1299 | 1.03 | 1.13 |
| P4b | .978 | .0351 | .0320 | .0294 | 1.10 | 1.19 |
| P4c | .965 | .0224 | .0205 | .0189 | 1.09 | 1.18 |

### Verdict — could a worst-case-over-shapes remapped M3 pass the ~1.5× bar?

**On the measured library, yes — with no margin, and only as a measurement.**

- A single fixed nominal α′ = 0.5 delivers realized coverage ≥ .95 on all 14
  cells (min exactly .950, on P2a) at **1.07–1.43× the production band's area**
  (mean 1.21×) and 0.98–1.27× the C=1 band's. That clears the ~1.5× criterion
  that M3 at its honest level failed by (round 3: 1.45–2.19×).
- The margin is one ladder step wide. α′ = 0.4 — a single step of safety —
  already reaches 1.57× and fails the bar; α′ = 0.55 already leaves P2a at
  .940.
- The binding cell is the **9:1 imbalance** cell (P2a, 900/100), not a shape:
  it needs α′ = 0.5 where every balanced cell is satisfied at 0.65–0.85, and
  its 1:9 mirror (P2b) needs only 0.75. So the infimum over this library is set
  by a dimension (class imbalance) that a shape library does not span — and
  the round-3 infimum of 0.6 was an artifact of not having run the imbalance
  cells.
- The n-drift undermines a fixed remap exactly as round 3 suspected: at fixed
  shape the required α′ falls 0.800 → 0.750 → 0.650 → 0.600 over
  n = 150 → 500 → 2000 → 5000, so any constant chosen on this library will
  eventually under-cover. The measured drift is ≈ 0.13 per decade of n at fixed
  shape; extrapolated linearly in log n, a constant α′ = 0.5 is exhausted near
  n ≈ 3×10⁴ for balanced binormal .95 — a crude extrapolation (the drift need
  not stay log-linear), and the imbalance direction has never been swept in n
  at all.
- **This is a measurement of the ceiling of any such scheme, not a method
  proposal.** A distribution-free version needs the worst case over *all*
  shapes and all (n0, n1); a finite library cannot establish that infimum, and
  a fitted α′ forfeits precisely the finite-sample theorem that is M3's only
  reason to exist. The honest statement remains round 3's: the 1.5–2× penalty
  M3 pays at its provable level is level accounting, and the accounting cannot
  be recovered without giving up the guarantee.

Caveats: the ELL calibration is Monte Carlo (`B` = 100k for n ≤ 700, 25k
above, with the quantile index shaded down 2 binomial SE), so the α′ column is
conservative by construction and coarse at n = 2000–5000; the level ladder's
granularity (0.05 steps in the relevant range) bounds the resolution of every
α′ above; and P2a's .950 at α′=0.5 sits exactly on the target with a 1.1pp SE,
so the "no cell below .95" row is a coin-flip away from being false.

*Connection to `c_calibration_spec.md`:* none of this is the C(n,α) map — it
is M3's own level accounting. The observation that class imbalance, not shape,
sets the infimum is however a data point for D2 (whether a 1-D `n_eff`
reduction suffices), in the direction of *not* assuming it does.

---

## 5. P5 — steep-corner pointwise repair probe

### Setup

Final band = fiducial(α) ∩ [M3(α₂) restricted to grid points
`1 ≤ k ≤ kc`, identity elsewhere], with α₂ spent by union bound (so the
certified level becomes 1−α−α₂). `k = 0` is **never** included: pinning
`U(0) = 0` is forbidden distribution-free (`fiducial_band_theory.md`
Cor. 9.3), and M3 as implemented pins it. α₂ ∈ {α/10, α/4} for α ∈ {.05, .2}
(i.e. α₂ ∈ {.005, .0125, .02, .05}), kc ∈ {10, 25, ⌈0.02·n0⌉}, two variants
(strictly local; and with the free monotone tightening a monotone estimand
permits). Cells P2c (binormal .99, n=500), P2d (binormal .99, n=150) and C2
(binormal .95, n=500) as control, 400 reps, M=3000, both the C=1 and C=2 trim
levels. The whole trim ladder is scored for every configuration, so the
matched-realized-coverage width comparison is available.

### Result

**Every configuration changes nothing** — 16 configurations at n=500 (4 α₂ × 2
kc × 2 variants) and 24 at n=150 (kc ∈ {3,10,25}), × 4 arms × 3 cells.
Maximum |Δarea| across all of them is 4.2×10⁻⁶ relative (one grid point in one
replicate, C2 at α₂=.05, kc=25 — so the intersection code does bite when it
can) and |Δcoverage| is exactly 0. Baselines for reference:

| cell | arm | coverage | area | w(.01) | w(.05) | mean j |
|---|---|---|---|---|---|---|
| P2c | α=.05, C=1 | .985 | .02924 | .4720 | .0932 | 4.3 |
| P2c | α=.05, C=2 | .965 | .02630 | .4038 | .0838 | 9.4 |
| P2d | α=.05, C=1 | .993 | .06099 | .7325 | .1896 | 7.7 |
| P2d | α=.05, C=2 | .988 | .05423 | .6848 | .1597 | 16.8 |
| C2 | α=.05, C=1 | .978 | .06320 | .4856 | .2389 | 3.0 |
| C2 | α=.05, C=2 | .953 | .05772 | .4474 | .2184 | 6.5 |

### Why — the corner diagnostic

`--exp corner` (100 reps) measures, on the window `k = 1..25`, the mean width
ratio M3(α₂)/fiducial and the fraction of replicates in which M3 is the
tighter band *anywhere* in the window, as a function of the nominal M3 level:

| cell / arm | α₂=.005 | .0125 | .05 | .2 | .5 | .7 | .9 | .99 |
|---|---|---|---|---|---|---|---|---|
| P2c, α=.05 C=2 | 2.88 / 0.00 | 2.59 / 0.00 | 2.20 / 0.00 | 1.71 / 0.00 | 1.39 / 0.08 | 1.21 / 0.99 | 1.01 / 1.00 | 0.82 / 1.00 |
| P2d, α=.05 C=2 | 3.93 / 0.00 | 3.46 / 0.00 | 2.63 / 0.00 | 1.96 / 0.00 | 1.49 / 0.61 | 1.34 / 0.86 | 1.08 / 1.00 | 0.84 / 1.00 |
| C2, α=.05 C=2 | 2.00 / 0.00 | 1.87 / 0.00 | 1.68 / 0.00 | 1.43 / 0.00 | 1.23 / 0.53 | 1.11 / 0.99 | 0.96 / 1.00 | 0.80 / 1.00 |
| P2c, α=.05 C=1 | 2.53 / 0.00 | 2.28 / 0.00 | 1.93 / 0.00 | 1.51 / 0.03 | 1.24 / 0.92 | 1.08 / 1.00 | 0.91 / 1.00 | 0.73 / 1.00 |

(entries: mean width ratio / fraction of replicates where M3 is tighter
somewhere in `k = 1..25`.)

Across the eight (cell, α, trim-exponent) arms actually used — α₂ ∈ {.005,
.0125} at α=.05 and {.02, .05} at α=.2 — **M3's corner edges are 1.7–4.6×
wider than the fiducial band's**, and tighter at not a single grid point in
100 replicates. M3 becomes the tighter band somewhere in the corner window in
the majority of replicates only at nominal α₂ ≈ 0.3–0.6 (α=.05 arms; 0.8–0.95
at α=.2), and reaches parity in *mean* window width only at α₂ ≈ 0.9 — i.e. it
would cost 30–90 percentage points of union-bound budget to buy anything at
all. (Consistent with round 3's containment table (b), which found
M3(α′ ≈ 0.15) already contains the production band everywhere.)

### Verdict

> *"Does this close a material part of the 2–3× gap to the oracle at steep
> corners, or is the fiducial cloud's corner over-dispersion elsewhere?"*

**Negative, and the round-3 premise behind it holds only at levels that carry
no guarantee.** The fiducial band is not over-dispersed *relative to M3* at the
first interior grid points — it is 2–4× tighter there at any usable α₂, and M3
is the narrower band only at nominal levels ≳0.9 (a ≤10%-confidence band),
which is where round-3's containment ladder was reading when it recorded the
opposite impression. The 2–3× gap to the oracle ceiling at
steep corners is therefore not reachable through M3's exact Beta corner bounds:
whatever the fiducial cloud is wasting at the corner, the composed-ELL band
wastes more. Both candidate mechanisms named in `fiducial_band_theory.md` §10 —
the within-gap interpolation convention, and the global equal-local-levels
budget interacting with the cloud's large local dispersion where the curve is
nearly vertical — survive untouched; this round does not discriminate between
them, it only removes an external repair. Note also what the probe *does*
establish about the certification story: the miss cap of round 3 (fiducial ∩
M3(α/10)) is inert not only globally but pointwise in the corner, which is
where one might have hoped a cap would be informative.

---

## 6. Where we stand — ranked

### Solved / answered by this round

1. **The last calibration idea in the backlog is closed.** Fiducial-predictive
   trim calibration is 1.7–2.3× more conservative in the trim depth than the
   ceiling at central α — *worse* than the plug-in's 1.27× that it was designed
   to beat — at ~20× the band's compute, and its smoothed variant lands exactly
   on plug-in performance (1.22–1.91×) while adding a window constant. The
   mechanism is measured, not inferred: at the 5% quantile of the depth
   distribution the truth sits ≈3× deeper in its cloud than a draw does, while
   a fiducial candidate sits at depth **0** — fiducial draws are rougher than
   plug-in curves, so integrating the calibration over the predictive law
   amplifies the roughness pathology instead of dodging it. With M1, `fid_cal`,
   M4b and now the predictive version all falsified by the same mechanism, the
   residual central-α shape spread has no remaining data-driven candidate on the
   list.
2. **An exact test at a named curve, delivered.** Size .190–.202 at α=.2 and
   .046–.051 at α=.05 across five (shape, n) cells — exact within 2 SE,
   including on t(2) and bimodal truths where Working–Hotelling has 0.000
   coverage. Power at n=500: .23–.26 at |ΔAUC|=.01, .71–.85 at |ΔAUC|=.02,
   halving at n=150. New characterization: the min-p statistic pays a ~2.5× worse
   sup-norm exchange rate against *localized* early-FPR alternatives than
   against global ones, and detects a corner pushed down more easily than one
   pushed up. This is a paper deliverable and the only fully tractable fragment
   of the M4 test-inversion frame.
3. **The ceiling of a worst-case-over-shapes remapped M3 is now measured.** A
   single fixed nominal α′ = 0.5 gives realized coverage ≥ .95 on all 14
   cells (min .950, on the 9:1 cell) at 1.07–1.43× the production band's
   area (mean 1.21×) — it would clear the ~1.5× bar that M3 at its honest level
   fails. But the margin is one ladder step (α′=0.4 → 1.57×), the binding cell
   is 9:1 **class imbalance** rather than any shape, and the required α′ drifts
   down monotonically with n at fixed shape (0.800 → 0.750 → 0.650 → 0.600 over
   n = 150 → 500 → 2000 → 5000, ≈0.13 per decade). A distribution-free version needs an infimum over all shapes
   *and* all (n0,n1) that no finite library can establish, and a fitted α′
   forfeits the theorem that is M3's reason to exist.
4. **The steep-corner pointwise repair is dead, and its premise holds only
   where M3 carries no guarantee.**
   At α₂ ∈ {α/10, α/4} M3's edges on `k = 1..25` are 1.7–4.6× *wider* than the
   fiducial band's, never tighter at any grid point in 100 replicates; M3 first
   bites there in most replicates only at nominal α₂ ≈ 0.3–0.6. The
   400-replicate intersection probe accordingly changes nothing (max |Δarea| 4×10⁻⁶, ΔCoverage exactly 0)
   on the two steepest-corner cells and the control.
5. **The "roughness axis" is not a concavity defect.** Measured on the true
   curves: t(2) .95 is the only non-concave shape in the library
   (`lcm_max` 0.179); bimodal .90 — the other shape sitting below the binormal
   ladder in `ae*` — is exactly concave, as are the kink and the whole binormal
   .55–.99 ladder. The natural "inflection/hook" hypothesis for the round-3
   axis is therefore falsified, and the axis remains unidentified.
6. **A level rule driven by a shape functional does *not* reintroduce the
   Wald-type bias.** Within-cell Spearman correlation between the per-replicate
   functional and that replicate's realized truth depth is |ρ| ≤ 0.25 almost
   everywhere, and per-rep rules match cell-mean rules to ≤1.5pp in coverage.
   The failure mode that killed every previous data-driven attempt is not what
   stops this route — the functionals simply do not carry the information.
7. **The erosion law's effective-looks ratio is not a rank-path crossing
   count** — not even in oracle form. `K̂_d/K̂_t` from median crossings
   correlates +0.69 with C\* across shapes at fixed n but −0.31 across all 14
   cells (wrong sign along n), and its plug-in version has the opposite sign to
   its oracle version. The specific geometric proxy for assumption (A2) of
   `fiducial_band_theory.md` §7.1 fails before estimability is even at issue.
8. **Reproduction check.** The refined-ladder M3 run reproduces
   `res_m3_p1grid.json` exactly on all eight round-3 cells, and the fresh
   14-cell `ae*` estimates agree with the published ones within combined MC
   error (max 8pp, typical 3–5pp).

### Still open, ranked

1. **The residual central-α shape spread has no candidate fix left.** 20pp at
   α=.5 and 14pp at α=.2 over the 14-cell grid under fixed C=2 (consistent
   with `m2_report.md`'s ±13–19pp). Every data-driven route measured — plug-in
   depth calibration, worst-case bracketing, fiducial-predictive calibration,
   and now a functional-driven level rule over five families / 32 candidate
   functionals — fails, three of them by the same roughness mechanism and the
   fourth for lack of signal. The honest claim stays "centred at every α, not
   calibrated at every α". What would change this: an identification of the
   shape axis itself (see 2), or a construction change rather than a level
   change.
2. **What *is* the shape axis?** The round-3 finding (flat along a five-fold
   early-slope ladder, 9–13pp along something else) survives, and this round
   removes two candidate identifications: concavity defect (bimodal is
   concave) and effective-looks-by-crossings (wrong sign along n). The best
   correlates found here — run-length entropy, windowed slope dispersion, the
   plug-in's depth contrast — reduce the leave-one-cell-out RMSE of C\* from
   0.193 to 0.13–0.17 at fixed n and then fail out-of-sample. A second-order
   analysis of the min-p functional under a rough-vs-smooth contrast (theory
   doc §12.1) is still the route; the tail-excursion structure, not the
   median-crossing structure, is what it has to characterize.
3. **Steep-corner width** (2–3× the oracle at AUC .99, small n) is untouched
   and now has one fewer candidate repair: the two mechanisms named in theory
   doc §10 (within-gap interpolation; equal-local-levels budget allocation
   against the cloud's corner dispersion) both remain live, and this round does
   not separate them. What it does close off is the idea that an *external*
   exact-Beta corner bound can be intersected in to recover the width.
4. **Whether a *provable* tightening of the M3 composition exists.** The level
   accounting is where all of M3's width goes (α′ = 0.5 buys .950 coverage at
   1.21× mean area), so the question is entirely about recovering a factor of
   ~12–16 in α inside a theorem. Not attempted here; the Šidák split plus
   worst-case composition is the obvious target.
5. **Class imbalance as a coverage axis in its own right.** P2a (900/100)
   needs a 1.5–1.7× smaller M3 nominal level than any balanced cell and is
   where the only functional rule that fit well explodes out-of-sample
   (Ĉ = 5.0, coverage .140). Imbalance was tested for validity in round 2 and
   passed, but as a *calibration* coordinate it has been swept at exactly two
   points (9:1 and 1:9 at n_total = 1000) and never in n. This is the
   C-calibration study's D2, and this round says do not assume a 1-D
   reduction.
6. **The α=.05 arm at n ≥ 10⁴, and the C\*(n) taper** — unchanged, and owned
   by `c_calibration_spec.md`. Nothing in this round touches it.
7. **A finite-sample coverage theorem** — unchanged. The domination route died
   in round 3; the exchangeability/conformal embedding route is still
   untouched. The one new adjacent asset is P3: an exact test at a *named*
   curve exists and is cheap, so the gap between "exact test per curve" and
   "band" is now the precisely stated obstacle (the projection over all
   curves), not a vague one.

## 7. Caveats specific to this round

* **P1's α=.05 column is resolution-limited twice over** (inner depth ≈5 on the
  `m_in = 1000` scale; averaged coverage granularity 1/60). Read α=.2/.5.
* **P1's ratio comparisons cross runs.** The within-run `recal` ceiling is a
  100-rep estimate and differs from the published one by up to 15% in `j`
  (this run's C2 `ae*@.5` = .815 vs the published .780). The ordering
  (predictive worse than plug-in worse than fixed C=2) is robust to that; the
  exact multipliers are not.
* **P2's held-out split does not exhibit the spread the bar was written
  against** (8.5pp at α=.5 under C=2, against the 14-cell grid's 20pp). The
  bar is failed on the grid that shows the spread and untestable on the
  held-outs; both readings are reported rather than the more flattering one.
* **P2's α=.05 targets are noisy** (`ae*@.05` carries ≈4pp, i.e. ±0.5–1.0 in
  C\*), which is why every α=.05 rule fitted on them misbehaves. A cleaner
  α=.05 fit needs the C-calibration study's precision machinery, which is
  exactly what that study is for.
* **Plateau-touching truths corrupt depth diagnostics.** On binormal .99 and
  bimodal .90 the raw cloud's depth of the truth is 0 in nearly every
  replicate (the CP allowance is what covers), so `S(truth)`-based statistics
  for those two cells measure the plateau, not roughness. They are excluded
  from every roughness reading above and reported separately.
* **P4's ELL calibration is Monte Carlo and deliberately conservative**
  (index shaded 2 binomial SE) — `B` = 25,000 rather than 100,000 at
  n0 > 700, so the n = 2000 and 5000 rows rest on the coarsest calibration in
  the table, in the safe direction. The level ladder's 0.05 granularity in the
  relevant range bounds every α′ figure.
* **P5's corner diagnostic is 100 reps**, adequate for width ratios (which are
  per-replicate deterministic given the bands) but coarse for the
  "fraction tighter anywhere" column near its transition.
* **In-sample choices in this round:** the candidate-functional battery, the
  smoothing window `sqrt(n0)`, the local-alternative family
  (`κ`-reparametrization on `t ≤ .05`), the refined level ladder, and the
  corner window `k ≤ 25` were all chosen while looking at these cells. None
  affects P3's exactness (a theorem plus MC error) or P4's validity (M3's
  theorem); the P2 fitting/held-out split was fixed before any fitting, but
  the *battery* was not.

---

*Result-file map: `res_r4_fpcal_C2.json` / `res_r4_fpcal_C5.json` (P1),
`res_r4_rough.json` (P2; contains per-replicate functionals and per-replicate
coverage/area over the α_eff grid, stored as a bit-string and a comma-joined
list respectively so that a level rule can be re-fitted and re-scored without
re-simulating), `res_r4_exact.json` (P3), `res_r4_m3grid.json` (P4),
`res_r4_repair.json` + `res_r4_corner.json` (P5).*
