# M3 / M4b: the guarantee layer and bracketed calibration

*Laptop CPU run, 2026-08-22. Code: `m3_experiments.py` (M3 core + P1/P2/P3),
`m4_experiments.py` (M4b bracket + the ordered-family premise check),
`analyze_m3.py` (tables). Results: `res_m3_*.json`, `res_m4_*.json`; logs
`log_m3_*.txt`, `log_m4_*.txt`. This report tests the two backlog ideas of
`stats/next_method_ideas.md` §7 against the benchmarks of
`stats/experiments/m2_report.md`.*

## 0. What was run

```
cd stats/experiments
U="uv run --project /home/nathan/Documents/studroc_paper python"

# P1  M3 alone over a ladder of nominal levels
$U m3_experiments.py --exp m3grid --cells C1 C2 C3 C4 C5 P2c P2d P4b \
      --reps 400 --B 100000 --threads 6 --out res_m3_p1grid.json

# P2 + P3  M3 and the fiducial band in the same replicate
$U m3_experiments.py --exp joint --cells C1 C2 C4 C5 P2c P2d --reps 400 \
      --M 3000 --B 100000 --alphas 0.05 0.2 --out res_m3_joint3.json
$U m3_experiments.py --exp joint --cells P4b --reps 200 --M 8000 \
      --B 100000 --alphas 0.05 0.2 --out res_m3_joint_p4b.json

# P4  bracketed calibration, and the ordered-family premise check
$U m4_experiments.py --exp bracket --cells C2 --reps 100 --M 3000 \
      --ncal 60 --min 1000 --ncalor 400 --alphas 0.5 0.2 0.05 \
      --out res_m4_brkC2.json
$U m4_experiments.py --exp bracket --cells C5 --reps 60 --M 3000 \
      --ncal 60 --min 1000 --ncalor 400 --alphas 0.5 0.2 0.05 \
      --out res_m4_brkC5.json
$U m4_experiments.py --exp family --specs binormal:0.70 binormal:0.80 \
      binormal:0.90 binormal:0.95 binormal:0.99 t2:0.95 bimodal:0.90 \
      --n0 500 --n1 500 --reps 300 --M 3000 --alphas 0.5 0.2 0.05 \
      --out res_m4_family.json

# P5  the C*(n) ladder at alpha=.5 / .2
$U m4_experiments.py --exp family --specs binormal:0.95 --n0 10000 --n1 10000 \
      --reps 150 --M 2500 --alphas 0.5 0.2 --out res_cstar_n10000.json
$U m4_experiments.py --exp family --specs binormal:0.95 --n0 20000 --n1 20000 \
      --reps 120 --M 2500 --alphas 0.5 0.2 --out res_cstar_n20000.json

$U analyze_m3.py res_m3_*.json res_m4_*.json res_cstar_*.json
```

Cells are those of `m2_report.md` §0 (C1 binormal .75 n=500; C2 binormal .95
n=500; C3 binormal .95 n=150; C4 bimodal .90 n=500; C5 t(2) .95 n=500; P2c
binormal .99 n=500; P2d binormal .99 n=150; P4b binormal .95 n=2000, all
balanced). 400 replicates per cell unless stated (coverage SE ≈ 1.1pp at the
95% level, ≈ 2.5pp at the 50% level); every cell seeded deterministically from
its name by the harness convention `seed + sum(ord(c) for c in cellname)`.
Ties are moot (all truths continuous); the declared convention is still random
tie-breaking, as in M2. Comparison columns (`fid_cp` = fiducial + CP upper
allowance at trim exponent C=1, `fid_rc` = the production recipe at C=2,
oracle, KS, WH) are the published `m2_report.md` numbers, with the `fid_*`
areas recomputed from the `by_ae` tables of `res_p1diag_*` / `res_p2_*` /
`res_p4_ab.json` at `ae = 1-(1-alpha)^C` so that C=2.0 (production), not the
C=2.2 of the report's §5 table, is what M3 is compared against.

## 1. The M3 construction as implemented (and why it is valid)

Rank space, harness convention: negatives `u ~ U(0,1)` iid, positives `w` with
CDF `R_true`, classify positive below the threshold, so `FPR = F0`, `TPR = F1`
and `R(t) = F1(F0^{-1}(t))` with both class CDFs increasing. `F0` is the
identity in rank space and `F1 = R_true`, but the band is built from the merged
label sequence only — the same input the fiducial arm gets.

**One-sample ELL bands.** `H(Z_(i)) ~ Beta(i, n+1-i)` exactly, and the vector
is distributed as uniform order statistics, so a band with a common local level
`gamma` at every order statistic has simultaneous coverage
`P(min_i BetaCDF(U_(i); i, n+1-i) >= gamma)` on the lower side; the upper
statistic has the same law under `u -> 1-u`, and the two-sided statistic is
`min_i min(BetaCDF, 1-BetaCDF)`. Both are functions of `(n, level)` only, so
calibration is done **once per sample size** by Monte Carlo (`B = 100,000`
order-statistic samples for n ≤ 700, `B = 25,000` at n = 2000; the empirical
quantile index is shaded down by 2 binomial SE so that the calibration's own
Monte Carlo error cannot eat into the guarantee). Verified against 40,000 fresh
samples at n = 500, `alpha_class = 0.0253`: realised one-sided lower 0.9887,
upper 0.9875, two-sided 0.9762 (target 0.9747) — exact, on the conservative
side as designed.

**Composition.** `F0 >= F0^lo` gives `F0^{-1}(t) <= (F0^lo)^{-1}(t)`, hence
`R(t) <= F1^hi((F0^lo)^{-1}(t))`; mirrored for the lower edge. With `p_i` the
number of positives ranked below the `i`-th negative, and each one-sided band
extended monotonically between order statistics,

```
U(t) = b1_hi[ p_{iup(t)} + 1 ],   iup(t) = min{ i : b0_lo[i] >= t }   (U=1 if none)
L(t) = b1_lo[ p_{ilo(t)-1} ],     ilo(t) = min{ i : b0_hi[i] >= t }   (b1_lo[0]=0)
```

`iup`/`ilo` depend only on `(n0, gamma)`, so they are tabulated once per
(cell, level) and each replicate costs two gathers. Note `b1_hi[k+1]` at
`k = khat*n1` is exactly the Clopper–Pearson upper bound that M2 uses as its
corner allowance: M3's upper edge is that same bound evaluated at an
FPR-shifted count, which is the precise sense in which M3 generalises the old
Beta/Wilson floors into one object.

**Alpha split.** Coverage is at least `P(E1∩E2)·P(E3∩E4)` (the class samples
are independent), where E1/E2 are the F0 lower/upper events and E3/E4 the F1
ones. Two splits were measured:

* **`sidak` (primary)** — one *two-sided* ELL band per class at
  `alpha_class = 1-sqrt(1-alpha)`, so the product is exactly `1-alpha`.
* **`bonf`** — the literal four-one-sided-component split at `alpha/4` each.

They are indistinguishable in area at `alpha <= 0.1` (e.g. C2: 0.09708 vs
0.09714) because the two `gamma`s nearly coincide there; `bonf` is strictly
worse at large `alpha` (its per-component level is capped at `alpha/4 <= 0.25`),
so all tables below use `sidak`.

**Endpoint pins.** `R(0)=0` and `R(1)=1` hold for every continuous DGP, so the
band is pinned (`U(0)=L(0)=0`, `L(1)=U(1)=1`). This is free validity and makes
areas comparable with the fiducial band, which pins them implicitly.

**Sanity gate.** Coverage must be ≥ nominal in every cell. It is, everywhere,
with a large margin (§2). Per-replicate checks in the runner also confirm
monotonicity of both edges and `L <= U` at every level and every replicate; the
four one-sided component events were logged separately at `alpha=.05` and land
at 0.965–0.9975 against a per-side target of ≈ 0.987 (the single 0.965 is
C1's F0-lower event; a direct re-run of the same seed reproduces it, and 120
independent 400-replicate blocks at the same `gamma` give mean 0.9878, min
0.9725, so it is an unlucky block, and the failure locations are spread over
the whole index range rather than clustered).

## 2. P1 — M3 baseline

400 reps/cell, `sidak` split, both `alpha = .05` and `alpha = .5` from the same
pass (the level ladder is evaluated in every replicate, so all levels cost
nothing extra).

| cell | truth | n0/n1 | α | M3 cov | v_lo | v_hi | p95 d | max d | M3 area | fid_cp | fid_rc | oracle | KS | ×cp | ×rc | ×orc | ×KS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | binorm .75 | 500/500 | .05 | 1.000 | .000 | .000 | 0 | 0 | .2040 | .1394 | .1272 | .1277 | .2328 | 1.46 | 1.60 | 1.60 | 0.88 |
| C2 | binorm .95 | 500/500 | .05 | 1.000 | .000 | .000 | 0 | 0 | .0971 | .0634 | .0579 | .0515 | .1718 | 1.53 | 1.68 | 1.88 | 0.57 |
| C3 | binorm .95 | 150/150 | .05 | 1.000 | .000 | .000 | 0 | 0 | .1840 | .1137 | .1029 | .0740 | .2796 | 1.62 | 1.79 | 2.49 | 0.66 |
| C4 | bimodal .90 | 500/500 | .05 | 1.000 | .000 | .000 | 0 | 0 | .1316 | .0856 | .0787 | .0737 | .1929 | 1.54 | 1.67 | 1.79 | 0.68 |
| C5 | t(2) .95 | 500/500 | .05 | 1.000 | .000 | .000 | 0 | 0 | .1029 | .0775 | .0709 | .0547 | .1777 | 1.33 | 1.45 | 1.88 | 0.58 |
| P2c | binorm .99 | 500/500 | .05 | 1.000 | .000 | .000 | 0 | 0 | .0510 | .0288 | .0259 | .0155 | .1393 | 1.77 | 1.97 | 3.29 | 0.37 |
| P2d | binorm .99 | 150/150 | .05 | 1.000 | .000 | .000 | 0 | 0 | .1190 | .0613 | .0544 | .0191 | .2412 | 1.94 | 2.19 | 6.23 | 0.49 |
| P4b | binorm .95 | 2000/2000 | .05 | 1.000 | .000 | .000 | 0 | 0 | .0472 | .0320 | .0294 | — | — | 1.48 | 1.61 | — | — |
| C1 | binorm .75 | 500/500 | .50 | 0.993 | .000 | .007 | 0 | .0128 | .1508 | .0947 | .0811 | .0863 | — | 1.59 | 1.86 | 1.75 | — |
| C2 | binorm .95 | 500/500 | .50 | 0.988 | .005 | .007 | 0 | .0043 | .0705 | .0418 | .0352 | .0352 | — | 1.69 | 2.00 | 2.00 | — |
| C3 | binorm .95 | 150/150 | .50 | 0.998 | .003 | .000 | 0 | .0020 | .1302 | .0719 | .0596 | .0518 | — | 1.81 | 2.18 | 2.51 | — |
| C4 | bimodal .90 | 500/500 | .50 | 0.993 | .003 | .005 | 0 | .0179 | .0962 | .0571 | .0482 | .0499 | — | 1.68 | 2.00 | 1.93 | — |
| C5 | t(2) .95 | 500/500 | .50 | 0.980 | .020 | .000 | 0 | .0850 | .0758 | .0509 | .0426 | .0410 | — | 1.49 | 1.78 | 1.85 | — |
| P2c | binorm .99 | 500/500 | .50 | 0.980 | .007 | .013 | 0 | .0105 | .0351 | .0179 | .0148 | .0109 | — | 1.96 | 2.37 | 3.22 | — |
| P2d | binorm .99 | 150/150 | .50 | 0.995 | .000 | .005 | 0 | .0063 | .0777 | .0361 | .0295 | .0135 | — | 2.15 | 2.63 | 5.75 | — |
| P4b | binorm .95 | 2000/2000 | .50 | 0.978 | .007 | .015 | 0 | .0085 | .0351 | .0219 | .0188 | — | — | 1.60 | 1.87 | — | — |

**Coverage.** 1.000 in all 8 cells at `alpha = .05` (0 misses in 3,200
replicate-cells) and 0.978–0.998 at `alpha = .5`. The theorem holds and the
sanity gate passes with room to spare.

**Where the width goes.** The penalty is spread across FPR rather than
localised (M3/`fid_cp` width ratio at FPR = .01/.05/.10/.50):

| cell | .01 | .05 | .10 | .50 |
|---|---|---|---|---|
| C1 | 1.36 | 1.42 | 1.47 | 1.48 |
| C2 | 1.48 | 1.52 | 1.58 | 1.48 |
| C3 | 1.28 | 1.57 | 1.69 | 1.65 |
| C4 | 1.36 | 1.45 | 1.51 | 1.84 |
| C5 | 1.24 | 1.46 | 1.73 | 1.26 |
| P2c | 2.01 | 1.72 | 1.70 | 1.66 |
| P2d | 1.34 | 2.12 | 2.12 | 1.92 |
| P4b | 1.42 | 1.51 | 1.55 | 1.41 |

**How conservative — the effective level.** The level ladder gives M3's
nominal→actual map directly. The largest nominal `alpha'` whose realised
coverage still reaches a target:

| cell | cov ≥ .95 | cov ≥ .80 | cov ≥ .50 |
|---|---|---|---|
| C1 | 0.80 | 0.95 | 0.99 |
| C2 | 0.70 | 0.95 | 0.999 |
| C3 | 0.80 | 0.95 | 0.999 |
| C4 | 0.80 | 0.95 | 0.999 |
| C5 | 0.60 | 0.90 | 0.99 |
| P2c | 0.80 | 0.95 | 0.999 |
| P2d | 0.80 | 0.99 | 0.999 |
| P4b | 0.60 | 0.90 | 0.99 |

M3 run at a *nominal* 20–40% confidence level already delivers 95% actual
coverage: the level accounting is conservative by a factor of 12–16 in `alpha`.
M3 at nominal 0.1% confidence still covers ~50–60% of the time.

**The consequence worth flagging.** M3 evaluated at its own effective level
has essentially the same area as the fiducial band:

| cell | α′ for cov ≈ .95 | realised cov | M3 area | fid_cp (.05) | fid_rc (.05) | ×cp | ×rc |
|---|---|---|---|---|---|---|---|
| C1 | 0.80 | 0.950 | .1297 | .1394 | .1272 | 0.93 | 1.02 |
| C2 | 0.70 | 0.965 | .0638 | .0634 | .0579 | 1.01 | 1.10 |
| C3 | 0.80 | 0.950 | .1091 | .1137 | .1029 | 0.96 | 1.06 |
| C4 | 0.80 | 0.960 | .0825 | .0856 | .0787 | 0.96 | 1.05 |
| C5 | 0.60 | 0.955 | .0725 | .0775 | .0709 | 0.93 | 1.02 |
| P2c | 0.80 | 0.955 | .0295 | .0288 | .0259 | 1.02 | 1.14 |
| P2d | 0.80 | 0.968 | .0630 | .0613 | .0544 | 1.03 | 1.16 |
| P4b | 0.60 | 0.958 | .0336 | .0320 | .0294 | 1.05 | 1.14 |

The composed-ELL band's *geometry* is as efficient as the fiducial cloud's;
the entire 1.5–2× area penalty is the price of the *provable* level accounting
(Šidák split plus worst-case composition), not of the shape. This is the same
diagnosis the `recal` analysis produced for M2 (§1a of `m2_report.md`), one
level down the stack — and the required `alpha'` is nearly as stable across
shapes (0.60–0.80) as M2's `alpha_eff*` was. Selecting `alpha'` this way is of
course not distribution-free (it uses the true curve), so it is a measurement
of where the slack lives, not a method.

### Verdict vs the stated criterion

> *"M3 is a shippable guarantee layer if its area is within ~1.5× the
> fiducial band's at α=.05."*

**Not met against the production band (`fid_rc`, C=2): 1.45–2.19×, median
1.68×; only C5 (t(2)) passes.** Against the C=1 band (`fid_cp`) the ratios are
1.33–1.94×, median 1.54×: three cells pass (C5 1.33, C1 1.46, P4b 1.48), two
are borderline (C2 1.53, C4 1.54) and three fail (C3 1.62, P2c 1.77, P2d 1.94).
The failure is systematic in the steep-corner direction — the two AUC .99
cells are the worst — which is the same regime where the fiducial band is
itself loosest relative to the oracle.

Two things the criterion does not capture and that are worth carrying forward:
M3 is **narrower than KS everywhere** (0.37–0.88× KS area, and only 0.37–0.49×
on the AUC .99 cells), so as a *provable* band it strictly dominates the
provable baseline; and its miss profile at `alpha=.5` is clean (p95 depth 0 in
every cell, max depth ≤ .0179 except C5's .0850).

## 3. P2 — the miss cap

Final band = fiducial band ∩ M3(α/10), i.e. `L = max`, `U = min` (both inputs
are monotone, so the intersection is monotone and no re-monotonisation is
needed; an empty intersection was never observed — `frac_empty = 0` in every
cell at every α).

### A convention artifact that has to be removed first

Measured against the *published* M2 recipe the cap has **negative** cost — it
narrows the band — by 0.06% (C5) to 12.9% (P2d), with coverage and every miss
statistic unchanged to the last digit. Localising it shows the cap binds at
**exactly one grid point in every replicate**: `bind_frac = 1/(n0+1)` in each of
the six n ≤ 500 cells (0.0020 at n0=500, 0.0066 at n0=150), and the point is
`k = 0`. The saving shrinks with n (0.3% at n = 2000) because it is governed by
`khat[0]/n1 ≈ R(1/n0)`.

The reason is a convention detail of the published recipe, not a property of
M3. The CP upper allowance is applied at every grid point including `t = 0`,
using `khat[0] = #{w < u_(1)}` — the staircase-*upper* empirical TPR at
FPR = 0, which is far from zero (≈ 0.29·n1 for binormal .95, ≈ 0.74·n1 for
binormal .99). So the published band has `U(0) ≈ 0.4`–`0.99` while `R(0) = 0`
exactly, and M3 — which pins `U(0) = 0` — clips it. Six-replicate spot checks:
C2 `fid_cp` gives `[L,U](0)` widths 0.43–0.59 at `k=0`; P2d gives 0.82–0.99.
Since `R(0)=0` for every continuous DGP, pinning `U(0)=0` is free validity, so
the honest reading is: **the published recipe leaves a free 0.1–13% of area on
the table at the single grid point t = 0, and should pin `U(0) = 0` (equally,
not apply the corner allowance at k=0).** That is a real, if cosmetic, finding
about M2, independent of M3.

All P2/P3 statistics below therefore use the **pinned** fiducial band
(`fp_cp`, `fp_rc` = the published recipe with `U(0) := 0`) as the reference, so
that the comparison is M3-body against fiducial-body.

### Result (400 reps; P4b 200 reps at M=8000; cap level `alpha/10`)

| cell | α | ref | fid area (published) | fid area (pinned) | cap area | cap cost | cov fid → cap | max depth fid → cap | bind frac |
|---|---|---|---|---|---|---|---|---|---|
| C1 | .05 | cp | .1394 | .1393 | .1393 | **0.00%** | .980 → .980 | .0120 → .0120 | 0.0000 |
| C1 | .05 | rc | .1278 | .1277 | .1277 | **0.00%** | .965 → .965 | .0142 → .0142 | 0.0000 |
| C2 | .05 | cp | .0634 | .0628 | .0628 | **0.00%** | .975 → .975 | .0237 → .0237 | 0.0000 |
| C2 | .05 | rc | .0581 | .0575 | .0575 | **0.00%** | .963 → .963 | .0278 → .0278 | 0.0000 |
| C4 | .05 | cp | .0856 | .0854 | .0854 | **0.00%** | .980 → .980 | .0195 → .0195 | 0.0000 |
| C4 | .05 | rc | .0789 | .0787 | .0787 | **0.00%** | .970 → .970 | .0268 → .0268 | 0.0000 |
| C5 | .05 | cp | .0775 | .0775 | .0775 | **0.00%** | .968 → .968 | .0376 → .0376 | 0.0000 |
| C5 | .05 | rc | .0712 | .0712 | .0712 | **0.00%** | .948 → .948 | .0610 → .0610 | 0.0000 |
| P2c | .05 | cp | .0288 | .0275 | .0275 | **0.00%** | .978 → .978 | .0331 → .0331 | 0.0000 |
| P2c | .05 | rc | .0260 | .0248 | .0248 | **0.00%** | .958 → .958 | .0462 → .0462 | 0.0000 |
| P2d | .05 | cp | .0613 | .0559 | .0559 | **0.00%** | .995 → .995 | .0098 → .0098 | 0.0000 |
| P2d | .05 | rc | .0547 | .0494 | .0494 | **0.00%** | .983 → .983 | .0289 → .0289 | 0.0000 |
| P4b | .05 | cp | .0320 | .0319 | .0319 | **0.00%** | .970 → .970 | .0179 → .0179 | 0.0000 |
| P4b | .05 | rc | .0295 | .0294 | .0294 | **0.00%** | .945 → .945 | .0243 → .0243 | 0.0000 |

The α = .2 rows are identical in structure (cap cost 0.00%, `bind_frac` 0.0000,
every statistic unchanged) and are in the JSONs.

**The cap never binds.** `bind_frac = 0.0000` in every cell at every α: M3 at
level α/10 contains the (pinned) fiducial band at *every* grid point in all
400 replicates, so the intersection is the fiducial band itself. `frac_empty`
is 0 everywhere, as it must be.

### Verdict vs the stated criterion

> *"Worth shipping if the width cost is ≤ ~3%."*

**Met, at a width cost of exactly 0.00%.** But the reason it is free is that it
is also inert: it changes no band, no coverage and no miss depth in any of the
2,600 replicates (6 cells x 400 + P4b x 200) at either α and against either
reference band — 10,400 band-level comparisons. What it buys is therefore purely the *certificate*: with
probability ≥ 1−α/10 the truth lies in M3(α/10), and then any miss of the
capped band has depth at most the local overhang of M3 beyond the fiducial
edge. That certified bound is measured next, and it is weak.

### How strong is the certificate?

`sup_t` overhang of M3(.005) outside the fiducial band at α=.05
(mean / p95 / max over replicates), against the *observed* max miss depth of
the fiducial band in the same run:

| cell | ref | whole grid | trim 25 pts each end | observed max miss |
|---|---|---|---|---|
| C1 | cp | .132 / .154 / .184 | .127 / .145 / .173 | .0120 |
| C1 | rc | .141 / .164 / .187 | .136 / .155 / .178 | .0142 |
| C2 | cp | .380 / .501 / .561 | .169 / .226 / .314 | .0237 |
| C2 | rc | .404 / .519 / .584 | .180 / .235 / .321 | .0278 |
| C4 | cp | .190 / .237 / .277 | .176 / .218 / .277 | .0195 |
| C5 | cp | .519 / .691 / .783 | .464 / .645 / .781 | .0376 |
| C5 | rc | .560 / .724 / .802 | .499 / .688 / .801 | .0610 |
| P2c | cp | .734 / .836 / .878 | .103 / .157 / .250 | .0331 |
| P2d | cp | .847 / .923 / .945 | .098 / .148 / .188 | .0098 |
| P4b | cp | .239 / .323 / .343 | .127 / .173 / .203 (.066 on t∈[.05,.95]) | .0179 |

The certified cap is **10–90× larger than the miss depth actually observed**,
and 4–25× larger even after discarding 25 grid points at each end. So
"the cap makes *misses are small* provable" is not delivered at any useful
constant: it makes "misses are at most ~0.1–0.9 deep" provable, where the
measured worst case over 400 replicates is 0.01–0.06. On the corner-dominated
whole-grid version the bound is close to vacuous, which is Lemma 9.2 biting —
M3's lower edge is 0 there, so no non-trivial depth bound exists.

## 4. P3 — containment probe (theory doc §12 open problem 2)

Both directions were measured in every replicate over a 20-point ladder of M3
levels `alpha'` from 0.999 down to 0.001, against the fiducial band at
`alpha = .05` in both its C=1 (`fp_cp`) and C=2 (`fp_rc`) forms. Because the
ELL bands are nested in `gamma`, the whole ladder is one tabulation per (cell,
level) and costs nothing per replicate. Containment was also evaluated on
nested interiors (the first and last `k0` grid points dropped, `k0 ∈ {0, 2, 5,
10, 25, 0.05·n0}`), because a single grid point can veto containment on the
whole grid.

### (a) The theorem-relevant direction: is M3(α′) ⊆ fiducial(.05)?

If it were — pathwise, in every replicate — the fiducial band would inherit
coverage ≥ 1−α′ by domination, which is the route to a finite-sample theorem
named in `fiducial_band_theory.md` §12.2.

**It essentially never happens, at any level, on any cell.**

| cell | ref | whole grid | k0 = 2 | k0 = 10 | k0 = 25 |
|---|---|---|---|---|---|
| C1 | fp_cp | never | never | .19 @ α′=.999, .08 @ .99 | .57 @ .999, .30 @ .99, .02 @ .95 |
| C1 | fp_rc | never | never | .12 @ .999 | .37 @ .999, .08 @ .99 |
| C2 | either | never | never | never | never |
| C4 | either | never | never | never | never |
| C5 | fp_cp | never | never | .67 @ .999, .32 @ .99, .01 @ .95 | .91 @ .999, .71 @ .99, .08 @ .95 |
| C5 | fp_rc | never | never | .50 @ .999, .05 @ .99 | .81 @ .999, .25 @ .99 |
| P2c | either | never | never | never | never |
| P2d | either | never | never | never | never |

(Entries are the fraction of 400 replicates in which containment holds; blank
levels are 0.00. The α′ = .999 M3 band is a *0.1%-confidence* band, so even the
non-zero entries certify nothing.) The pathwise domination route is therefore
**dead**: the fiducial band at α=.05 does not contain any M3 band that carries
a non-trivial guarantee, not even after discarding 25 grid points at each end,
and not even on the two cells (C1 at AUC .75, C5 at t(2) .95) where the
fiducial band is relatively widest. On the four remaining cells containment
fails on the interior at *every* level tested.

Mechanically there are two distinct obstructions, localised by a
three-replicate spot check on C1 and C2 (25 grid points trimmed at each end,
`alpha' ∈ {.999, .95, .5}`):

* **On the whole grid**, M3's lower edge is exactly 0 over the first grid
  points at any level (the `ilo(t) = 1` regime, i.e. the Lemma 9.2 honesty
  frontier) while the fiducial floor there is strictly positive. That alone
  vetoes containment at `k0 = 0` and `k0 = 2` in every replicate on every cell.
* **In the interior**, the residual failures sit near the **upper right**, not
  at the steep corner: M3's lower edge at the plateau is
  `b1_lo(n1) = gamma^(1/n1)`, which dips just under the fiducial floor. At
  `alpha' = 0.999` on C2 these are 43–62 of the 451 interior points with a
  maximum shortfall of 0.003–0.004, plus 21–23 points where the upper edges
  tie to within tolerance; on C1 the lower-edge set is *empty* and only 6–12
  upper-edge ties remain, which is why C1 reaches 57% containment at
  `k0 = 25`, `alpha' = .999`.
* **At any level carrying a real guarantee** the question is moot: at
  `alpha' = 0.5` M3 is wider than the fiducial band over most of the interior
  (295–397 of 451 points below the floor, 195–400 above the roof, magnitudes up
  to 0.04), so no amount of endpoint trimming helps.

So the obstruction is not a corner artifact that a definitional tweak could
remove, and it does not shrink with n (P4b, n = 2000, is one of the four cells
where interior containment never occurs at any level).

### (b) The miss-cap direction: is M3(α′) ⊇ fiducial(.05)?

Largest nominal `alpha'` at which M3(α′) contains the (pinned) fiducial band at
every grid point in **100%** of replicates:

| cell | n per class | vs fid_cp (C=1) | vs fid_rc (C=2, production) |
|---|---|---|---|
| C1 binormal .75 | 500 | 0.005 | 0.10 |
| C2 binormal .95 | 500 | 0.02 | 0.15 |
| C4 bimodal .90 | 500 | 0.01 | 0.15 |
| C5 t(2) .95 | 500 | 0.02 | 0.10 |
| P2c binormal .99 | 500 | 0.10 | 0.15 |
| P2d binormal .99 | 150 | 0.15 | 0.30 |
| P4b binormal .95 | 2000 | 0.05 | 0.15 |

This direction works, and comfortably: against the production band an M3 band
at nominal 85–90% confidence already contains it everywhere, so the α/10 cap of
P2 is far inside the regime where containment is automatic — which is exactly
why it never binds. Scaling: at fixed shape (binormal .95) the required level
is 0.02 at n=500 and 0.05 at n=2000 against C=1, and 0.15 at both against C=2
— i.e. flat-to-slightly-easier in n, no adverse trend. Across shape at fixed n
the required level moves by a factor of ~10 against C=1 (0.005 at AUC .75 to
0.10 at AUC .99) but only ~1.5 against the production band (0.10–0.15), the
high-AUC/small-n cells being the loosest.

## 5. P4 — M4b bracketed (worst-case) calibration

### Setup

Per replicate: build the M3 band at overall level **50%** (`sidak` split) as a
cheap confidence set of curves; take three members — the lower edge `lo`, the
midline `mid`, the upper edge `hi` — plus their monotone moving-average
smoothings (`s_lo`, `s_mid`, `s_hi`, window `sqrt(n0)` rounded to an odd
integer, re-monotonised, endpoints pinned), which separate the *shape* of a
member from its *roughness*; frequentistically calibrate the fiducial trim
depth against each member; and take the most conservative (smallest) depth over
each trio. Members are piecewise-linear on the FPR grid, hence continuous CDFs;
they are sampled with an explicit generalized inverse, because
`rbe.Curve.inv` dedupes ties keeping the smallest FPR and is therefore wrong on
the long flat stretches that the M3 edges have.

Calibration against a member `R0`: `ncal = 60` simulated rank-space datasets
from `R0`, `m_in = 1000` fiducial draws each, the full production band ladder
(CP upper allowance included), two read-outs —

* `j_thresh` — the largest ladder depth whose simulated coverage of `R0`
  reaches `1-alpha`. This is exactly the `calibrate_j` convention of
  `m2_report.md` §1c, so the bracket is directly comparable to the `fid_cal`
  plug-in arm.
* `j_quant` — the `alpha`-quantile of `R0`'s own min-p depth `S` over the same
  simulations. Same target computed as an order statistic; lower variance, but
  biased low (hence conservative) because it ignores the CP allowance.

Depths are rescaled to the outer `M = 3000` by the local level
`ell = j/(m_in+1)`. **Oracle calibration** — the identical procedure run
against the *true* curve — is computed once per cell at both `ncal = 60` (the
budget the bracket gets) and `ncal = 400`, which separates the bracket's own
bias from the bias a small inner budget induces. Reps: 100 (C2), 60 (C5);
coverage SE ≈ 2.2 / 2.8pp at the 95% level, ≈ 5.0 / 6.5pp at the 50% level.

**A resolution limit to state up front.** The coverage-optimal local level at
n = 500 is `ell* ≈ 2e-3` at α=.05, so the optimal depth on the `m_in = 1000`
scale is ≈ 2 — integer resolution is the binding constraint there, and the
α=.05 column of every bracket table below is resolution-limited (the
`ncal = 400` oracle returns `j = 6` on the M=3000 scale against a published
`recal` ceiling of 4.0, i.e. one ladder step of slack). At α=.2 and α=.5 the
optimal depths on the inner scale are ≈ 5 and ≈ 40 and the read-out is sound:
the `ncal = 400` oracle reproduces the published per-cell `recal` ceilings for
C2 essentially exactly (j = 33 vs the published 33 at α=.2, j = 111 vs 109 at
α=.5). Those two α are therefore the load-bearing columns.

### 5a. The premise: is the calibration functional monotone in early slope?

Measured directly, before any bracketing: the exact per-curve calibration
ceiling `ae*` (largest trim level whose realised coverage reaches `1-alpha`)
for an ordered family of truths at n = 500, 300 reps each, `M = 3000`
(`res_m4_family.json`). `R(.01)`/`R(.05)` are the truth's early slope
read-outs; `C* = log(1-ae*)/log(1-alpha)`.

| truth | AUC | R(.01) | R(.05) | ae*@.5 | C* | ae*@.2 | C* | ae*@.05 | C* |
|---|---|---|---|---|---|---|---|---|---|
| binormal .70 | .700 | .057 | .183 | .775 | 2.15 | .435 | 2.56 | .170 | 3.63 |
| binormal .80 | .800 | .128 | .325 | .785 | 2.22 | .430 | 2.52 | .155 | 3.28 |
| binormal .90 | .900 | .304 | .567 | .775 | 2.15 | .390 | 2.22 | .115 | 2.38 |
| binormal .95 | .950 | .500 | .752 | .760 | 2.06 | .410 | 2.36 | .140 | 2.94 |
| binormal .99 | .990 | .832 | .950 | .740 | 1.94 | .410 | 2.36 | .145 | 3.05 |
| bimodal .90 | .900 | .196 | .467 | .730 | 1.89 | .400 | 2.29 | .130 | 2.72 |
| t(2) .95 | .950 | .069 | .879 | .685 | 1.67 | .355 | 1.97 | .060 | 1.21 |

(MC error on `ae*`: ≈ 2.9pp at α=.5, ≈ 2.3pp at α=.2, ≈ 4pp at α=.05, since
`d(coverage)/d(ae)` ≈ 1 at α=.5 and smaller in the tail. The binormal .95 row
independently replicates the published `recal` ceiling for C2 — .760/.410/.140
here vs .780/.420/.110 in `m2_report.md` §1a.)

**The premise fails.** Along the binormal ladder the early slope `R(.05)`
ranges over a factor of five (.183 → .950) while `ae*@.5` moves by 4.5pp
(.785 → .740) — monotone-looking but barely more than one MC SE, and
`ae*@.2` / `ae*@.05` are flat within noise. Meanwhile the two off-family
shapes move `ae*` by far more than the whole early-slope ladder does, in the
*wrong* direction for an early-slope ordering:

* t(2) .95 has the *same* AUC as binormal .95 and a *steeper* `R(.05)` (.879 vs
  .752), yet `ae*@.5` = .685 vs .760 and `ae*@.05` = .060 vs .140 — it sits
  below the entire binormal ladder, whose steep end (.99) only reaches .740.
* bimodal .90 has a *shallower* `R(.05)` than binormal .90 (.467 vs .567) and
  also a *lower* `ae*` (.730 vs .775) — the opposite sign to the binormal
  ladder's trend.

So `ae*` is essentially constant along early slope and varies by 9–13pp along
some other shape axis — plausibly corner sharpness / local roughness, since
t(2) is the truth with a near-corner (`R(.01)` = .069 but `R(.05)` = .879) and
the theory doc's H2 predicts exactly that a rougher truth needs a smaller `ae`.
**Bracketing over early-slope extremes therefore cannot bracket the quantity
that actually varies**, which undercuts M4b before the bracket is even run.

### 5b. Result — the bracket on C2 (binormal .95, n = 500, 100 reps)

Oracle (calibrating against the true curve) trim depth on the M = 3000 scale,
`(j_thresh, j_quant)`:

| inner budget | α=.5 | α=.2 | α=.05 |
|---|---|---|---|
| `ncal = 400` | (111, 51) | (33, 24) | (6, 3) |
| `ncal = 60` (what the bracket gets) | (111, 54) | (24, 18) | (3, 3) |

The `ncal = 400` row reproduces the published per-cell `recal` ceiling for C2
(mean j = 109 at α=.5, 33 at α=.2, 4.0 at α=.05), so the read-out is sound;
the `ncal = 60` row shows the small-budget bias is a factor of ≈ 1.4 at α=.2
and unresolvable at α=.05.

Mean calibrated depth per member, and the worst-case over each trio:

| α | `lo` | `mid` | `hi` | `s_lo` | `s_mid` | `s_hi` | **brk_raw** | **brk_sm** | fid_cp | fid_rc | oracle |
|---|---|---|---|---|---|---|---|---|---|---|---|
| .50 | 3.0 | 75.0 | 93.6 | 94.8 | 97.4 | 122.8 | **3.0** | **85.7** | 56.4 | 120.9 | 111 |
| .20 | 3.0 | 17.5 | 23.9 | 26.5 | 22.0 | 33.8 | **3.0** | **18.4** | 15.4 | 34.1 | 33 |
| .05 | 3.0 | 3.1 | 5.7 | 5.9 | 3.1 | 7.6 | **3.0** | **3.0** | 3.0 | 6.3 | 6 |

Realised behaviour of the resulting bands:

| arm | α | coverage | v_lo | v_hi | p95 depth | max depth | area |
|---|---|---|---|---|---|---|---|
| `brk_raw` (M4b as specified) | .50 | **0.970** | .000 | .030 | 0 | .0058 | .0640 |
| `brk_sm` (smoothed members) | .50 | 0.570 | .130 | .320 | .0343 | .1029 | .0385 |
| `fid_cp` (C=1) | .50 | 0.670 | .110 | .230 | .0253 | .0884 | .0419 |
| `fid_rc` (C=2, production) | .50 | 0.500 | .160 | .380 | .0540 | .1408 | .0352 |
| `orc_j` (oracle depth) | .50 | 0.510 | .150 | .380 | .0492 | .1380 | .0360 |
| `brk_raw` | .20 | **0.970** | .000 | .030 | 0 | .0058 | .0640 |
| `brk_sm` | .20 | 0.870 | .020 | .110 | .0089 | .0220 | .0515 |
| `fid_cp` | .20 | 0.890 | .020 | .090 | .0050 | .0259 | .0521 |
| `fid_rc` | .20 | 0.800 | .050 | .160 | .0102 | .0530 | .0460 |
| `orc_j` | .20 | 0.800 | .050 | .160 | .0104 | .0525 | .0463 |
| `brk_raw` / `brk_sm` | .05 | 0.970 | .000 | .030 | 0 | .0058 | .0640 |
| `fid_cp` | .05 | 0.970 | .000 | .030 | 0 | .0058 | .0639 |
| `fid_rc` | .05 | 0.900 | .020 | .080 | .0010 | .0184 | .0586 |
| `orc_j` | .05 | 0.910 | .010 | .080 | .0010 | .0159 | .0589 |

**The bracket as specified collapses onto the floor.** The worst-case over the
three raw members is always the *lower edge* `lo`, whose calibrated depth is
pinned at the smallest resolvable value (`j_in = 1`, i.e. the widest tube the
cloud supports) at every α. So `brk_raw` returns **one band, independent of α**
— area .0640 at α = .5, .2 and .05 alike — with coverage 0.970 against nominal
0.50 and 0.80. Against the oracle depth it is 111/3 = **37× too conservative
in j at α=.5** and 33/3 = **11× at α=.2**, versus the 1.3–1.7× of the plug-in
calibration it was meant to improve on, at 4.5× the plug-in's inner compute
(6 members × 60 sims vs 1 × 80). α-resolution is destroyed entirely.

**Roughness, not shape, is what pins it.** Smoothing the same three members
changes nothing about their early slopes (`s_lo` slope .583 vs `lo` .599) but
moves `lo`'s calibrated depth from 3.0 to 94.8 — a factor of 30 at essentially
the same shape. The smoothed bracket is therefore the informative arm, and it
lands at 85.7 vs the oracle's 111 (**1.30×** conservative) at α=.5 and 18.4 vs
33 (**1.79×**) at α=.2 — i.e. *exactly the plug-in conservatism*, 1.27×/1.27×
for `fid_cal` on the same cell in `m2_report.md` §1c, at α=.5 and slightly
worse at α=.2. Its coverage (0.570 at nominal .500, 0.870 at nominal .800) sits
between `fid_cp` and `fid_rc`, and its area is 7–11% above the oracle band's.

Spearman correlation between a member's early slope `R(.05)` and its calibrated
depth, over 600 member-replicates: +0.475 (α=.5), +0.391 (α=.2), +0.241
(α=.05) — positive but modest, and confounded: within either trio the depth is
monotone in slope, but the raw-vs-smoothed contrast at *fixed* slope is larger
than the whole within-trio slope effect.

### 5c. Result — the bracket on C5 (t(2) .95, n = 500, 60 reps)

C5 is the cell where the fixed C = 2 remap under-covers most at central α, so
it is the hardest test for any data-driven level. Oracle depth on the M = 3000
scale: `ncal = 400` → 111 (α=.5), 27 (α=.2), 3 (α=.05); `ncal = 60` → 144, 24,
3. (60 reps: coverage SE ≈ 6.5pp at α=.5, 5.2pp at α=.2, 2.8pp at α=.05.)

| α | `lo` | `mid` | `hi` | `s_lo` | `s_mid` | `s_hi` | **brk_raw** | **brk_sm** | fid_cp | fid_rc | oracle |
|---|---|---|---|---|---|---|---|---|---|---|---|
| .50 | 3.0 | 86.7 | 82.7 | 99.8 | 110.8 | 117.7 | **3.0** | **83.5** | 68.6 | 144.6 | 111 |
| .20 | 3.0 | 23.9 | 18.2 | 28.8 | 31.7 | 34.5 | **3.0** | **21.6** | 19.1 | 41.7 | 27 |
| .05 | 3.0 | 4.8 | 3.7 | 6.6 | 7.3 | 6.8 | **3.0** | **6.6** | 3.9 | 8.0 | 3 |

| arm | cov @ α=.50 | area | cov @ α=.20 | area |
|---|---|---|---|---|
| `brk_raw` (M4b as specified) | **0.950** | .0790 | **0.950** | .0790 |
| `brk_sm` (smoothed members) | 0.533 | .0496 | 0.783 | .0629 |
| `fid_cp` (C=1) | 0.517 | .0503 | 0.850 | .0628 |
| `fid_rc` (C=2, production) | 0.367 | .0420 | 0.683 | .0554 |
| `orc_j` (oracle depth) | 0.400 | .0450 | 0.800 | .0595 |

At α=.05: `brk_raw` = `brk_sm` = `orc_j` = 0.950 coverage (area .0790/.0780/.0790),
`fid_cp` 0.933, `fid_rc` 0.917 — all inside 1 SE of each other and
resolution-limited.

Same picture as C2, with one nuance:

* `brk_raw` again pins at the floor (`lo` = 3.0 at every α), giving a single
  α-independent band: 37× the oracle depth at α=.5, 9× at α=.2, coverage 0.950
  against nominal 0.500 and 0.800, and an area (.0790) *larger* than the
  fiducial band's own α=.05 band (.0768).
* `brk_sm` is 111/83.5 = **1.33×** conservative at α=.5 and 27/21.6 = **1.25×**
  at α=.2 — the same 1.25–1.79× range as on C2 and as the plug-in
  (`fid_cal`, 1.27×/1.27×).
* The nuance: on *this* cell `brk_sm`'s coverage (0.533 at nominal .500, 0.783
  at nominal .800) is better calibrated than the fixed C = 2 map (0.367, 0.683)
  and about the same as C = 1 (0.517, 0.850). That is the one place M4b looks
  useful — but the effect (a 1.25–1.33× conservative depth) is what C = 1 gives
  for free on this cell, at 1/360 of the compute, so nothing is bought.

### Verdict vs the stated criteria

* **M4b as specified (worst-case over the raw M3-50% members) is a clear
  negative.** It is 11–37× more conservative in the trim depth than the oracle,
  far worse than the plug-in calibration it was designed to replace, and it
  returns an α-independent band. The mechanism is the one already known from
  M1/`fid_cal`: the members are *rougher* than the truth, so their own fiducial
  bands cover them less often, so calibration returns a too-wide band. Taking a
  worst case over a set does not fix that — it *selects for* it, because the
  roughest, most pathological member of the set is exactly the one that
  minimises the calibrated depth.
* **Smoothing the members recovers plug-in-level performance and no better**
  (1.30× / 1.79× on C2, 1.33× / 1.25× on C5, versus the plug-in's 1.27× /
  1.27×), so the bracket buys nothing over plug-in even after the roughness is
  removed by hand — and the smoothing window is a new tuning constant. On C5,
  `brk_sm` does beat the fixed C = 2 map at central α (0.533/0.783 vs
  0.367/0.683 at nominal .500/.800), but it does not beat C = 1 (0.517/0.850)
  on the same cell, and C = 1 costs nothing.
* Combined with §5a (the calibration functional is flat in early slope and
  varies along a different, roughness-like axis), the whole M4b route is
  **falsified as a practical calibration**: the axis it brackets is not the axis
  that matters, and the members it brackets over are contaminated by exactly
  the nuisance that the calibration is sensitive to.

## 6. P5 — the C*(n) ladder at α = .5

`fiducial_band_theory.md` §7 names this as the discriminating experiment
between H1 ("one simultaneity budget per class", predicting `C* → 2`) and H2
("roughness mismatch, vanishing", predicting `C* → 1`). Existing ladder, from
the `recal` ceilings of `m2_report.md` §1a at α = .5:

| n per class | 25 | 150 | 500 | 2000 | 5000 |
|---|---|---|---|---|---|
| `ae*` | .880 | .810 | .780 | .695 | .710 |
| `C* = log(1-ae*)/log(1-α)` | 3.06 | 2.40 | 2.18 | 1.71 | 1.79 |

### Result: the ladder extended to n = 10,000 and 20,000

Balanced binormal .95, `M = 2500`, 150 reps at n = 10,000 and 120 at
n = 20,000, α ∈ {.5, .2} (`res_cstar_n10000.json`, `res_cstar_n20000.json`).
Saturation was checked: the realised trim depth at the ceiling is
`j* = 36.1` (n = 10,000) and 27.7 (n = 20,000), and `j*` at `ae = .5` is 22.4
and 19.8 — all far above the `j* >= 5` resolution rule, so `M = 2500` is
adequate at these α.

| n per class | 25 | 150 | 500 | 2000 | 5000 | **10000** | **20000** |
|---|---|---|---|---|---|---|---|
| `ae*` @ α=.5 | .880 | .810 | .780 | .695 | .710 | **.645** | **.600** |
| `C*` @ α=.5 | 3.06 | 2.40 | 2.18 | 1.71 | 1.79 | **1.49** | **1.32** |
| `ae*` @ α=.2 | .580 | .480 | .420 | .380 | .355 | **.300** | **.240** |
| `C*` @ α=.2 | 3.89 | 2.93 | 2.44 | 2.14 | 1.97 | **1.60** | **1.23** |

(As in `m2_report.md` §1a the ladder mixes cells at the small end: n = 25 is
binormal .90 (C7) and n = 150 is binormal .95 (C3); n >= 500 is binormal .95
throughout, so the n >= 500 trend is at fixed shape.)

MC error: coverage SE ≈ 4.1pp at n = 10,000 (150 reps) and ≈ 4.6pp at
n = 20,000 (120 reps) at α=.5, and `dC*/d(ae*)` ≈ 3.6–4.1, so
`C* = 1.49 ± 0.17` and `1.32 ± 0.16`.

**H2 wins.** `C*` at α=.5 falls monotonically 3.06 → 1.32 from n = 25 to
n = 20,000 (the 2000→5000 wobble, 1.71→1.79, is well inside noise), and at
n = 20,000 it is **4.2 SE below 2**. H1 predicted a plateau near 2; it is
falsified over the tested range. The α=.2 ladder decays in step (3.89 → 1.23).
A power fit over the fixed-shape points n = 500…20,000 at α=.5 gives

```
C*(n) - 1  ~=  1.26 * (n/500)^(-0.32)
```

which extrapolates to `C* ≈ 1.23` at n = 10^5 and `≈ 1.11` at n = 10^6: the
taper toward 1 is real but slow, roughly `n^{-1/3}`.

**Mechanism.** The roughness-contrast diagnostic H2 predicts — the truth's
min-p depth stochastically dominating a draw's, because the draws carry `1/n`
interpolation roughness the smooth truth does not — has shrunk by roughly the
right amount. At n = 150 it reads S(truth) 5%-quantile 17 vs S(draw) 4.9, a
3.5× contrast (`fiducial_band_theory.md` §7); at n = 10,000 the two agree at
the 5%, 10% and 50% quantiles (1.0/1.03, 2.0/2.44, 22.5/22.4) and at
n = 20,000 the contrast is ≈ 1.1–1.9× (5%: 1.95 vs 1.00; 20%: 6.0 vs 5.1;
50%: 22.0 vs 19.8). Read this as order-of-magnitude only: at n ≥ 10^4 the
depths are small integers and their low quantiles are a coarse, noisy
statistic. The direction — a large contrast at small n collapsing toward
parity at large n — is what the ladder needs and what it shows.

**Consequence for the production recipe.** Fixed `C = 2` is over-trimming at
these sizes at central α: coverage at α=.5 with C = 2.2 is 0.407 at
n = 10,000 and 0.383 at n = 20,000 (nominal .500), while `fid_cp` (C=1) gives
0.633 and 0.583. At α=.2 with C = 2.2: 0.707 and 0.708 against nominal 0.800.
The α=.05 arm was not run at these n (it needs `M` ≈ 10–12k by the §9 budget
rule), and `d(coverage)/d(ae)` is small in that tail, so nothing here shows
C = 2 breaking α=.05 validity — but the n-taper flagged in
`next_method_ideas.md` §5.2 is now supported by data over a 40× range of n
rather than by theory alone.

## 7. Reproduction and validation notes

* The fiducial arm was **reimplemented** in `m3_experiments.py`
  (`fid_sorted_and_depths`, `fid_band_at`, `trim_depth`) rather than reused
  through `m2.rep_profile`, so that the exact realised trim depth (not the
  nearest ladder value) is used and so that the band arrays themselves are
  available for intersection. It reproduces the published `fid_cp` numbers
  exactly on every cell at α=.05 — area .1394/.0634/.0856/.0775/.0288/.0613
  and coverage .980/.975/.980/.968/.978/.995 on C1/C2/C4/C5/P2c/P2d, matching
  `m2_report.md` §2 and §5 to four decimals — which validates both
  implementations against each other.
* All M3 statistics use `cp_lo_mode = "none"` (no degenerate lower allowance on
  the fiducial band), matching the published `fid_cp`/`fid_rc` numbers rather
  than the §7 recommended recipe; the degenerate mirror is free on every
  continuous truth, so this does not affect any comparison here.
* Cells are seeded by the harness convention, so every table is reproducible
  from the commands in §0. The ELL calibration uses its own fixed seed
  (`987654321 + n`) so that the same `(n, B)` always produces the same
  `gamma`, independent of cell.

### Appendix to P1 — a level remap for M3, and why it is not a fix

Because M3's nominal→actual map is so compressed, a *fixed* nominal level well
above 0.05 delivers 95% actual coverage at near-fiducial width. Realised
coverage / area at fixed nominal `alpha'` (400 reps, `sidak`):

| cell | α′=.9 | α′=.8 | α′=.7 | α′=.6 | α′=.5 |
|---|---|---|---|---|---|
| C1 | .887 / .1199 | .950 / .1297 | .970 / .1372 | .985 / .1441 | .993 / .1508 |
| C2 | .887 / .0554 | .945 / .0603 | .965 / .0638 | .975 / .0668 | .988 / .0705 |
| C3 | .922 / .1012 | .950 / .1091 | .975 / .1185 | .990 / .1243 | .998 / .1302 |
| C4 | .912 / .0762 | .960 / .0825 | .975 / .0874 | .988 / .0917 | .993 / .0962 |
| C5 | .843 / .0606 | .905 / .0654 | .943 / .0693 | .955 / .0725 | .980 / .0758 |
| P2c | .900 / .0263 | .955 / .0295 | .965 / .0311 | .975 / .0324 | .980 / .0351 |
| P2d | .930 / .0589 | .968 / .0630 | .988 / .0713 | .993 / .0745 | .995 / .0777 |
| P4b | .850 / .0283 | .900 / .0304 | .935 / .0320 | .958 / .0336 | .978 / .0351 |

A single fixed `alpha' = 0.6` gives 0.955–0.993 coverage on all eight cells at
0.93–1.05× the `fid_cp` area, i.e. the composed-ELL band with a fixed 12× level
inflation is width-competitive with the fiducial band. Three reasons this is a
measurement and not a method: (i) it is an empirical constant fitted on these
eight cells, so it forfeits exactly the finite-sample theorem that is M3's
entire reason to exist; (ii) the composed band's coverage depends on the true
shape through where the composition binds, so no distribution-free joint
calibration of `alpha'` exists — only a worst-case-over-shapes one, which would
have to be at least as conservative as the Šidák split somewhere; (iii) the
required `alpha'` **drifts down with n** (0.7–0.8 at n = 150–500, 0.6 at
n = 2000, with C5's t(2) shape already at 0.6 at n = 500), so a fixed remap
would under-cover at large n — the same liability the C=2 remap carries for M2.

## 8. Where we stand — ranked

### Solved / answered by this round

1. **M3 is valid, cheap, and its width is now quantified.** Coverage 1.000 at
   α=.05 in all 8 cells (0 misses in 3,200 replicate-cells) and 0.978–0.998 at
   α=.5; area 1.33–1.94× `fid_cp` and 1.45–2.19× the production `fid_rc`;
   0.37–0.88× KS. Cost: one Monte Carlo calibration per sample size
   (`B = 100k` order-statistic draws, ≈ 11 s at n = 500) plus two gathers per
   band — cheaper than the fiducial cloud by orders of magnitude.
2. **The 1.5–2× penalty is level accounting, not geometry.** M3 evaluated at
   the nominal level that makes its realised coverage 0.95 has 0.93–1.05× the
   `fid_cp` area on all 8 cells; a single fixed nominal `alpha' = 0.6` gives
   0.955–0.993 coverage everywhere. The composed-ELL *shape* is as efficient as
   the fiducial cloud's; the Šidák split plus worst-case composition costs a
   factor of 12–16 in α. (Not a method — see §2 appendix (i)–(iii).)
3. **The miss cap is free and inert.** Fiducial ∩ M3(α/10) costs 0.00% area and
   never binds (`bind_frac` = 0.0000 across 10,400 band-level checks): M3 at α/10
   already contains the fiducial band at every grid point in every replicate.
   The certificate it provides is weak — a bound of 0.10–0.90 on the miss depth
   where the observed worst case is 0.01–0.06.
4. **The domination route to a finite-sample theorem is dead** (theory doc
   §12.2). M3(α′) ⊆ fiducial(.05) essentially never holds — not at any α′ up to
   0.999 (a 0.1%-confidence band), not on any cell, and not after discarding 25
   grid points at each end. Two separate obstructions: M3's lower edge is
   identically 0 over the first grid points (the Lemma 9.2 frontier) and dips
   just under the fiducial floor again near the plateau; and at any α′ carrying
   a real guarantee M3 is simply wider over most of the interior.
5. **M4b (bracketed worst-case calibration) is falsified.** The worst case over
   the raw M3-50% members is set by the lower edge, which pins the trim depth at
   the floor: 9–37× more conservative than the oracle on both cells,
   α-independent (one band at α = .5, .2 and .05 alike), and far worse than the
   plug-in it was meant to replace. Smoothing the members recovers exactly
   plug-in performance (1.30×/1.79× on C2, 1.33×/1.25× on C5, vs plug-in's
   1.27×/1.27×) and no better — and on C5, where it does beat the fixed C = 2
   map at central α, it does not beat the free C = 1 fallback.
6. **The M4b premise is false independently of the bracket.** The calibration
   ceiling `ae*` is flat along a five-fold early-slope ladder (±4.5pp at α=.5,
   ≈ 1 MC SE) and moves 9–13pp along a different, roughness-like axis (t(2) .95
   sits below the entire binormal .70–.99 ladder despite a mid-ladder early
   slope). Early slope is not the axis to bracket.
7. **H2 beats H1 on the C\*(n) question** (theory doc §7, §12.1). `C*` at α=.5:
   3.06 (n=25) → 2.18 (500) → 1.71/1.79 (2000/5000) → 1.49 ± 0.17 (10,000) →
   **1.32 ± 0.16 (20,000)**, i.e. 4.2 SE below H1's predicted plateau of 2, with
   the same decay at α=.2 (3.89 → 1.23) and with the roughness-contrast
   diagnostic H2 predicts shrinking from 3.5× at n = 150 to ≈ 1–2× at
   n ≥ 10^4. Fit: `C*(n) - 1 ≈ 1.26 (n/500)^{-0.32}`. A fixed C = 2 needs an
   n-taper, and the taper is slow (~`n^{-1/3}`).
8. **A free width saving in the published M2 recipe.** The CP upper allowance is
   applied at `t = 0`, where the staircase-upper count `khat[0]` is large but
   `R(0) = 0` exactly, so `U(0)` runs 0.4–0.99. Pinning `U(0) = 0` is free
   validity and saves 0.0% (C1) to 8.8% (P2d, binormal .99 n=150) of band area
   — the saving is largest exactly where the band is loosest.

### Still open

1. **M3 as a shippable guarantee layer fails the stated width criterion**
   (1.45–2.19× the production band vs the ~1.5× bar), worst at steep corners
   (AUC .99: 1.97× at n=500, 2.19× at n=150). Whether a *provable* tightening
   of the composition exists — rather than the empirical level remap of the §2
   appendix — is the open question, and it is the only route by which M3 becomes
   more than a KS replacement.
2. **A finite-sample coverage theorem for the fiducial band.** Domination by M3
   is ruled out (item 4 above). The remaining candidates named in the theory doc
   (an exchangeability/conformal embedding) are untouched by this round.
3. **The residual central-α shape spread** (`m2_report.md` §6.1) is unchanged:
   this round removes one candidate fix (M4b) and identifies the axis that
   matters (roughness, not early slope), but produces no estimator for it. A
   rank-computable roughness functional is now the specific target.
4. **The n-taper rate for C.** The ladder gives
   `C*(n) - 1 ≈ 1.18 (n/500)^{-0.29}` over n = 500…10,000 at α=.5, but that is a
   two-parameter fit to five points with `C*` SE ≈ 0.17 at the top end, and the
   α=.05 arm of the ladder was not run (it needs `M` ≈ 10–12k at these grid
   sizes).
5. **Steep-corner width** is untouched. M3 is *narrower* than the fiducial band
   at the first interior grid points (which is why containment fails in that
   direction), so the over-dispersion of the fiducial cloud there is real and
   M3's edge is a candidate pointwise repair — but a pointwise intersection at a
   handful of grid points would need its own validity accounting, and was not
   measured.
6. **Whether a distribution-free joint calibration of the composed band
   exists.** The composed band's coverage depends on the true shape through
   where the composition binds, so the required `alpha'` varies (0.6–0.8 over
   these cells) and drifts down with n. A worst-case-over-shapes calibration is
   the only distribution-free option and would have to be at least as
   conservative as Šidák somewhere; not attempted.

## 9. Caveats

* **Monte Carlo error.** 400 reps/cell for P1–P3 (coverage SE 1.1pp at the 95%
  level, ≈ 2.5pp at 50%); 200 for P4b joint; 100/60 for the P4 brackets (2.2 /
  2.8pp at 95%, 5.0 / 6.5pp at 50%); 300 for the P4 family; 150/120 for the P5
  ladder. Differences under 2–3pp in coverage are not to be read. The P4
  bracket's headline effects (9–37× in trim depth) are far outside noise; its
  C5-versus-C2 differences (1.25× vs 1.79× at α=.2) are not.
* **The α=.05 column of every P4 table is resolution-limited.** At n = 500 the
  coverage-optimal trim depth on the `m_in = 1000` inner scale is ≈ 2, so
  integer resolution, not statistics, sets the answer; the `ncal = 400` oracle
  itself lands one ladder step from the published `recal` ceiling there. Read
  α=.2 and α=.5 as the load-bearing columns.
* **ELL calibration is Monte Carlo, not exact.** `gamma` comes from `B` = 100k
  (n ≤ 700) or 25k (n = 2000) uniform-order-statistic samples with the quantile
  index shaded down by 2 binomial SE. At the smallest levels in the P3 ladder
  (`alpha' = .001`, `alpha_class = 5e-4`) the effective sample behind `gamma` is
  15–50 order statistics, so those rows are coarse; every level used for a
  headline claim (α ≥ .005) has ≥ 100. Exact Noé-type crossing recursions would
  remove this, and would also let the ladder extend below .001.
* **In-sample choices.** The `sidak` alpha split, the endpoint pinning, the
  interior masks `k0`, and the smoothing window `sqrt(n0)` for the M4b members
  were all chosen while looking at these cells. None of them can affect M3's
  validity (a theorem), but the *width* comparisons and the `brk_sm` numbers are
  to that extent in-sample. The `alpha' ≈ 0.6` remap of the §2 appendix is
  explicitly a fit to these eight cells and is reported as a measurement, not a
  method.
* **The t = 0 convention finding is about the harness's grid, not only about the
  band.** `rhat_batch` uses the staircase-upper convention, so `khat[0]` is the
  empirical TPR *just before* the first negative while the evaluation truth at
  `t_0 = 0` is exactly 0. Any implementation that applies a corner allowance at
  `k = 0` inherits the same slack; whether the production `fiducial_band` does
  was not checked here (this report only touches the harness).
* **P5 does not test α=.05 at large n.** The n = 10,000 and 20,000 runs used
  `M = 2500`, adequate for α ∈ {.5, .2} by the `j* >= 5` rule but not for
  α=.05, which needs `M` ≈ 10–12k at those grid sizes. So the ladder establishes
  the `C*` trend at central α only; the claim that fixed C = 2 remains safe at
  α=.05 for n ≥ 10^4 is an extrapolation from `d(coverage)/d(ae)` being small in
  the tail, not a measurement.
* **One unlucky calibration block.** C1's F0-lower component event realised
  0.965 against ≈ 0.987 expected (14 failures in 400 where ~5 are expected).
  Reproduced from the same seed, but 120 independent 400-replicate blocks at the
  same `gamma` gave mean 0.9878 and min 0.9725, and the failing order-statistic
  indices are spread over the whole range, so this is a block-level fluke rather
  than a construction error. C1's composed coverage was 1.000 regardless.

---

*Result-file map: `res_m3_p1grid.json` (P1), `res_m3_joint3.json` +
`res_m3_joint_p4b.json` (P2/P3; `res_m3_joint2.json` is the earlier pass of the same
experiment, run against the un-pinned reference band and with the sup-overhang
statistic not yet split by interior mask; it is kept because §3's convention
finding and the §4(a) `k0` table are read off it), `res_m4_brkC2.json` /
`res_m4_brkC5.json` (P4 bracket), `res_m4_family.json` (P4 premise check),
`res_cstar_n10000.json` / `res_cstar_n20000.json` (P5).*
