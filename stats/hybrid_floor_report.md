# Stage F: the frontier M3 floor — measurements

*2026-09-04. Runs the study specified in `stats/hybrid_floor_spec.md`.
Data: `data/results/hybrid_floor_20260902/`. 160 cells, 42,000 paired
replicates, 2.5 h of cell time. Primary estimand: simultaneous coverage on
the native grid at alpha = .05; alpha = .5 is a transfer diagnostic. The
rule was frozen before any cell ran and was not modified afterwards.*

*Companions: `stats/fiducial_band_theory.md` §7.3-7.4 (the mechanism),
`stats/c_calibration_followup_report.md` (what motivated this).*

---

## 0. Scope, and what this report deliberately does not do

Per the spec, Stage F is an information-gathering study, not a method
arbiter: "the final suite and the authors choose the paper's method", and
the .94 A1-letter bar and the strict .95 point bar are "descriptive
yardsticks, not method-selection gates". This report therefore states
measurements and the inferences that follow directly from them. It does not
declare arms to pass or fail, does not recommend what to ship, and does not
revise `frontier_floor_v1`.

Where a number invites an interpretation that the data do not compel, the
interpretation is deferred to §11, which is explicitly speculative.

**Reading guide.** §2 lists the design properties that bound every number in
the report; several headline figures are not what they would appear to be
without it. §3-§5 answer the questions the spec asked, in order. §6 collects
the results that were *not* anticipated. **§7 resolves coverage to individual
grid points and is where the strongest mechanism evidence sits**; §8 is a
post-hoc variant sweep, flagged as such. §9 lists what remains unmeasured.

---

## 1. The rule as run

`frontier_floor_v1` unions the C = 1 fiducial band with M3 on

- the first `min(n0, ceil(log(M+1)))` native grid points — 9 or 10 points at
  the budgets used here (M ranged 6,730-14,992), against the theory's
  `ceil(Q)` ~ 7; and
- the empirical-TPR-1 run extended inward by `ceil(2 sqrt(K))` grid points,

closed by widening only. Its inputs are class sizes, the cloud budget, and
the merged label sequence. No AUC, no fitted surface, no declared shape
class. The region is a function of `M` and the ranks and is therefore
**identical at alpha = .05 and alpha = .5** — this is a property of the
implementation, and it matters for §6(e).

Exact statements carried by the construction (not measured, proved): the
floored band contains C = 1 pointwise, so its coverage is never lower for any
DGP or replicate; and a miss inside the region implies a full-curve M3 miss,
so that component is capped at alpha2 whatever the selection rule. The
whole-band statement decomposes as `alpha2 + P(exterior miss)` and the
exterior term is what this study measures.

**Measured in-region failure rate: .0002 (A), .0003 (B), .0000 (C)**, worst
cell .005, against an exact cap of .05. The proved component is not binding
anywhere in the study.

---

## 2. Design properties that bound every number below

These are not caveats added after the fact; they are consequences of the
frozen design, and several of the report's headline numbers are
uninterpretable without them.

**(a) Study A's corpus is enriched for failure and its macro statistics are
not population estimates.** The replay selection takes *every* prior cell
below .94 and samples only 15 from [.94, .97) and 8 from >= .97. Of 88 replay
cells, **65 (74%) come from the previously-failing stratum**:

| prior C = 1 coverage | cells | C = 1 coverage now | floored | floored min |
|---|---:|---:|---:|---:|
| < .94 | 65 | .899 | .9864 | .945 |
| .94 - .97 | 15 | .959 | .9820 | .960 |
| >= .97 | 8 | .986 | .9888 | .975 |

Study A's C = 1 macro coverage of .926 is therefore a statistic about a
deliberately adversarial corpus, not an estimate of how the C = 1 band
behaves on a representative shape library. The same applies to every A macro
number, including the width comparisons. Cross-stratum, the floored band's
coverage is roughly flat (.982-.989), which the enrichment does not explain
away but also does not license extrapolating from.

The corpus is also larger than the spec anticipated — 116 cells against ~68 —
because the "take all failing cells" rule met more of them than the spec's
estimate assumed.

**(b) Study B is adversarial by construction too.** Sixteen of its 30 cells
are student-t wedge cells or fresh sliver cells. Its C = 1 macro of .829 is a
statement about that design, not about ROC curves in general.

**(c) Study C is the only roughly shape-neutral sample, and it is small.**
Fourteen cells, seven shapes at two sizes. Two of the fourteen are
student-t: the spec asked for seven non-student-t shapes, but one of the two
fixed-seed LHS mapper draws landed in the t family, so the non-t evidence
rests on six shapes, not seven.

**(d) Replication.** Study A cells carry 200 replicates each — Wilson
half-widths of roughly ±.03 at coverage .96. Individual A cell coverages
should not be read to three digits. Study B/C cells carry 400, with top-up to
1,200 while the Wilson interval straddles .94; the rule fired on two B cells
(both to 800) and no C cells.

**(e) Corner-geometry labels in Study C were computed from the exact true
curve.** They are a pre-outcome stratification, not a statistic any analyst
could compute from data.

---

## 3. Study A: the five questions the spec asked

### A-Q1. Does the region capture the observed corner mechanisms?

The spec's estimand is `q_c(R) = P(V nonempty and V not a subset of R)` —
the probability that a C = 1 violation set is not wholly contained in the
region.

| | A | B | C |
|---|---:|---:|---:|
| C = 1 miscoverage (macro) | .0744 | .1706 | .0321 |
| exterior escape `q_c(R)` (macro) | .0172 | .0182 | .0211 |
| max over cells | .0650 | .0350 | .0300 |
| floored miscoverage (macro) | .0157 | .0177 | .0196 |
| conditional capture (macro over cells) | .597 | .670 | .201 |

**Answer.** Partially, and the two summaries say different things. About 77%
of C = 1's miscoverage *mass* in A is removed (.0744 → .0172 exterior
escape). But conditional capture — the per-cell fraction of C = 1-failing
replicates whose violation set lies *entirely* inside the region — is .597 in
A and .670 in B. Roughly a third of failing replicates still have some
violation outside the region, and those replicates remain uncovered.

So the region does not capture every dangerous corner miss. What it does is
remove the deep ones; §6(a) shows the residual differs in direction, depth,
and location from what was removed.

**A mechanism worth noting: the rule's reach exceeds its region.** Floored
miscoverage is below the exterior-escape rate in all three studies (.0157 vs
.0172; .0177 vs .0182; .0196 vs .0211). This is a consequence of the
widening closure rather than noise: `L_closed[k] = min(L_raw[j] for j >= k)`
propagates any lower-edge reduction *leftward* across the whole grid, so an
M3 union applied only on the saturated run also lowers the band everywhere
below it. The nominal region fraction therefore understates the rule's
effective support on the lower edge, and "region fraction" and "capture"
should not be read as the same quantity.

### A-Q2. What does the margin buy, and how do the margins compare?

| arm | region frac | macro cov | min | width vs C = 1 |
|---|---:|---:|---:|---:|
| C = 1 | — | .9256 | .570 | — |
| `probe_legacy` (fixed, legacy closure) | .507 | .9763 | .915 | +7.3% |
| `probe_fpr` (fixed, widening) | .507 | .9776 | .915 | +12.6% |
| `count5` | .519 | .9858 | .945 | +14.1% |
| `frontier_run0` | .109 | .9843 | .940 | +10.2% |
| `frontier_floor_v1` | .129 | .9843 | .940 | +11.3% |
| `frontier_j1` | .187 | .9846 | .940 | +12.0% |
| full M3 | 1.000 | .9991 | .985 | +41.4% |

**Answer.** Across 116 cells x 200 replicates, `run0` and `floor_v1` differ in
the coverage outcome of **one replicate in one cell** (`tsurf73--n582`, .970
vs .975). `j1` saves five further single replicates. The `2 sqrt(K)` margin
costs +1.04pp of C = 1 width; `j1` costs a further +0.75pp.

For scale: the margin's measured effect is 4.3e-5 of replicates against a
residual exterior-escape rate of .0169 — a ratio of about 390. **This is a
measurement of effect size, not a demonstration that the margin never
matters**; a mechanism occurring at rate ~1e-4 would need on the order of
10^5 replicates to characterize, and the study has 2.3e4.

### A-Q3. Does the predicted right-channel imbalance direction appear?

§7.4(c)3 predicted negative-majority is the dangerous orientation at high
AUC. On the 24-cell imbalance LHS, restricted to AUC >= .95:

| orientation | cells | C = 1 mean | C = 1 min | floored mean |
|---|---:|---:|---:|---:|
| negative-majority (n0 > n1) | 4 | .870 | .570 | .983 |
| positive-majority (n1 > n0) | 6 | .977 | .955 | .984 |

**Answer.** Yes, in the predicted direction. The worst cell in the entire
study is `n0 x n1 = 1676 x 391` at AUC .989, covering .570 at C = 1.
**Caveat: this rests on 4 versus 6 cells at 200 replicates each**, and the
LHS did not control AUC or df across the orientation split — the two groups'
mean AUCs are .917 and .935 overall. The direction agrees with the
prediction; the magnitude should not be quoted as an effect size.

### A-Q4. What is the alpha2 = alpha versus alpha/2 frontier?

| alpha2 | macro cov | min | cells < .94 | cells < .95 | width |
|---|---:|---:|---:|---:|---:|
| .05 | .9843 | .940 | 0 | 2 | +11.3% |
| .025 | .9848 | .940 | 0 | 2 | +13.5% |

**Answer.** Halving alpha2 moves macro coverage by +0.05pp, leaves the
minimum and both bar counts unchanged, and costs +2.2pp of width. The
in-region failure rate is .0002 at alpha2 = .05, so the additional regional
budget is not being consumed.

### A-Q5. Do AUC, m_q, and curvature summaries explain residuals?

Correlations across the 116 A cells:

| covariate | with C = 1 miscoverage | with floored miscoverage |
|---|---:|---:|
| AUC | +.36 | −.00 |
| log n0 | +.01 | +.66 |
| log N0 = log(n0(1−AUC)/2) | −.39 | +.53 |
| saturated run length | +.34 | +.10 |
| log(n0/n1) | +.16 | +.01 |
| C = 1 miscoverage | — | +.03 |

**Answer.** No. AUC predicts C = 1's miscoverage (+.36) and none of the
floored band's (−.00). The floored residual's only strong correlate is
log n0. Critically, it is **uncorrelated with the quantity it repaired**
(r = +.03 with C = 1's own miscoverage), which says the residual is not
leftover corner mechanism.

**The n dependence needs care, and the obvious reading of it is wrong.** The
n0-bucketed table below is confounded with shape: A's corpus takes all
failing cells, and five of the eight cells in the last row are the
deliberately adversarial extent-stress and near-.99-AUC cells.

| n0 | cells | C = 1 | floored | floored min |
|---|---:|---:|---:|---:|
| <= 200 | 29 | .922 | .9921 | .975 |
| 201-500 | 31 | .920 | .9892 | .965 |
| 501-1500 | 33 | .940 | .9814 | .945 |
| 1501-4000 | 15 | .919 | .9787 | .960 |
| > 4000 | 8 | .914 | .9600 | .940 |

The unconfounded comparison is Study C's paired design, the same seven
shapes at two sizes:

| | n = 500 | n = 8,000 |
|---|---:|---:|
| floored coverage | .9854 | .9754 |
| C = 1 coverage | .9646 | .9711 |
| floor width cost | +26.6% | +4.4% |
| region fraction | .278 | .141 |

**Across all 160 cells, no cell's floored coverage is significantly below
nominal**: every Wilson-95% upper bound exceeds .95. The floored band's
coverage declines toward the nominal level as n grows while its width charge
falls in step, which is the direction a widening correction should move as
the correction shrinks. Whether the limit is .95 or below it is not
determined by these data (§7).

---

## 4. Study B: prospective external behaviour

30 cells, 400-1,200 replicates. Floored coverage macro .9823, minimum .965,
zero cells below .94; C = 1 macro .8294, minimum .505, 18 cells below .94.
Width +9.1% against full M3's +42.9%.

| arm | macro cov | min | cells < .94 | width |
|---|---:|---:|---:|---:|
| `probe_legacy` | .9820 | .911 | 1 | +9.2% |
| `probe_fpr` | .9824 | .912 | 1 | +13.5% |
| `count5` | .9853 | .966 | 0 | +13.7% |
| `frontier_run0` | .9822 | .965 | 0 | +8.1% |
| `frontier_floor_v1` | .9823 | .965 | 0 | +9.1% |
| `frontier_j1` | .9825 | .965 | 0 | +9.6% |

The A ordering reproduces: the frontier arms are indistinguishable from each
other in coverage, and the two fixed-FPR probes cover a region four times
larger while dropping below the bar on one cell that the frontier arms carry
(`b-imbalance-16`, n0 x n1 = 300 x 1500: .912 against .983).

### 4.1 The sliver block

Six fresh cells, new names and seed streams, defined before Study A ran.
Corollary 14.1 predicts that C = 1 failure tracks the *unsampled-sliver
event*, that the adaptive region expands on those realizations, and that the
floored band covers.

| cell | AUC | n0 x n1 | pred. unsampled | obs. | C = 1 all | C = 1 \| unsampled | C = 1 \| sampled | floored |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 24 | .60 | 250x250 | .367 | .413 | .613 | .097 | .974 | .980 |
| 25 | .60 | 2000x2000 | .368 | .393 | .590 | .000 | .971 | .978 |
| 26 | .80 | 250x250 | .449 | .473 | .520 | .000 | .986 | .988 |
| 27 | .95 | 2000x2000 | .449 | .485 | .505 | .000 | .981 | .983 |
| 28 | .80 | 2000x500 | .449 | .408 | .580 | .000 | .979 | .978 |
| 29 | .80 | 500x2000 | .449 | .455 | .525 | .000 | .963 | .978 |

**Prediction 1 — failure tracks the event.** Confirmed. Conditional on an
unsampled sliver, **no covering replicate was observed** in five of six cells
(0/157, 0/189, 0/194, 0/163, 0/182; one-sided Clopper-Pearson 95% upper
bounds 1.5-1.9%, so "zero observed", not "zero"). Cell 24 covers .097 — it is the
smallest and lowest-AUC cell, where the truth's deficit `d/n1 = 1/250` is
largest relative to the band's width. Observed unsampled fractions match the
predicted e^(−d) within Monte Carlo error, and the two-stratum decomposition
reproduces the overall coverage in every cell, so the saturation event
accounts for the deficit with no unexplained remainder beyond the .963-.986
coverage of the sampled stratum.

**Prediction 2 — the region expands on those realizations.** Confirmed, and
sharply. Conditional means:

| cell | run length \| sampled | \| unsampled | region frac \| sampled | \| unsampled |
|---|---:|---:|---:|---:|
| 24 (250x250) | 0.4 | 31.6 | .058 | .220 |
| 25 (2000x2000) | 0.4 | 241.8 | .007 | .143 |
| 27 (2000x2000) | 0.4 | 659.5 | .007 | .361 |
| 28 (2000x500) | 0.5 | 531.7 | .007 | .295 |

The empirical-TPR-1 run is the trigger and it moves by two to three orders of
magnitude on exactly the event the theory names.

**Prediction 3 — n-independence at fixed d.** Cells 24 and 25 hold d = 1.0
while n moves 8x: C = 1 covers .613 and .590. Consistent with the
construction not decaying in n.

**Prediction 4 — both orientations.** Cells 28 and 29 cover .580 and .525.

Floored coverage conditional on an unsampled sliver is .975-.995; M3 covers
.995-1.000 throughout.

---

## 5. Study C: geometry-class transfer

14 cells. Floored macro .9804, minimum .970. C = 1 macro .9679, minimum .865,
one cell below .94.

| geometry | cells | C = 1 mean | C = 1 min | cells < .94 | floored mean | floored min |
|---|---:|---:|---:|---:|---:|---:|
| corner-concave | 10 | .9772 | .968 | 0 | .9805 | .970 |
| ambiguous | 4 | .9444 | .865 | 1 | .9800 | .975 |

The single C = 1 failure is `hetero_gaussian` at AUC .98, n = 500, an
*ambiguous*-labelled cell whose misses carry the corner signature: FPR
.62-.998, true TPR .988-.999, lower-edge rate .115 against upper-edge .020.
The floored band covers .978 and its residual sits at FPR .10-.58.

**Answer to the Corollary 13.1 check.** Consistent with the prediction: no
corner-concave cell fell below the bar at C = 1, and the one cell that did
was not labelled concave. This is **not the first such evidence** — the
follow-up's Weibull/gamma/beta-opposing corner spot checks (.977-.992) were
already consistent with it — and what Stage F adds is that the labels were
fixed before outcomes. With 10 concave and 4 ambiguous cells and one failure
among them, the design cannot estimate how strongly the label separates; it
can only report that the separation was not contradicted.

**Transfer of the floor.** All 14 cells sit at .970-.995 floored.

**Price on cells that did not need repair.** On the four concave cells at
n = 500 the floor moves coverage by +0.2 to +0.7pp at +12% to +49% width
(beta-opposing: .993 → .995 for +48.8%, against M3's +79.3%).

Study C's arm ordering inverts A's on width: the fixed regions are cheaper
here (+11.3% and +14.5%) than the frontier rule (+15.5%), because on
high-AUC concave shapes at n = 500 the empirical-TPR-1 run is long. All arms
cover; this is a width observation only.

---

## 6. Results that were not anticipated

**(a) The residual's direction and location differ from what was removed.**

| | C = 1 | floored |
|---|---:|---:|
| A: lower-edge miss reps | 1,413 | 130 |
| A: upper-edge miss reps | 318 | 234 |
| B: lower-edge miss reps | 1,990 | 98 |
| B: upper-edge miss reps | 179 | 137 |
| A: max miss depth over cells | .641 | .118 |
| B: max miss depth over cells | .302 | .233 |
| A: cells with median miss FPR > .5 | 85 / 116 | 44 / 102 |
| B: cells with median miss FPR > .5 | 23 / 30 | 8 / 30 |

C = 1's misses in the worst A cells sit at FPR .68-.99 where the true TPR is
.996-1.000 — the empirical-TPR-1 run, at the depth scale Lemma 13 predicts.
After flooring, lower-edge failing replicates fall 91% (A) and 95% (B) while
upper-edge counts move much less, so the residual is roughly balanced in
direction and sits at interior FPR. Residual escapes are overwhelmingly
*far* from the region rather than at its edge (A: .0169 vs .0021; B: .0182 vs
.0016; C: .0209 vs .0020).

*Maximum depth is a max over cells and is a noisy statistic; the direction
counts and location distributions are the load-bearing evidence here.*

**(b) The margin's measured effect is ~1e-4 of replicates.** §3 A-Q2. The
spec treated the margin as mandatory on theoretical grounds; the study can
resolve effects of order 1e-3 and this one is smaller.

**(c) The left region carries most of the width cost at small n.**
Decomposing the paired width charge over A's 60 cells with n0 <= 500: left
component .00866, saturated-run component .00731, total .01576, against a
C = 1 area of .1207. **55% of the charge comes from the ~5% of the grid in
the left region**, while the run — 13% of the grid — carries the other 45%.
This is consistent with §7.4(h)5's claim that the run is cheap because both
bands are within O(Q/n1) of 1 there.

**(d) The two fixed-FPR probes cover four times more grid and cover no
better.** In A they reach .978 against the frontier arms' .984 at comparable
or higher width. On `b-wedge-08` (n = 130) `probe_fpr`'s residual misses sit
at FPR .008-.047 with true TPR .011-.79 — grid points 1 through 6. At
n0 = 130, `.005 x n0 = 0.65`, so the fixed rule floors one grid point where
the frontier rule floors nine. `count5`, which uses a count rather than a
fraction, matches the frontier arms' coverage (.9858) at +14.1%.

**(e) The alpha = .5 transfer is strongly conservative.**

| | C = 1 | floored (alpha2 = .5) | floored (alpha2 = .25) |
|---|---:|---:|---:|
| macro coverage (nominal .50) | .584 | .781 | .795 |
| width vs C = 1 | — | +19.7% | +26.9% |

The C = 1 band already over-covers at alpha = .5, and the region — identical
at both levels by construction (§1) — raises it a further 20pp.

---

## 7. Mechanism resolved to the grid

Everything above treats coverage as a whole-band event. Resolving it to
individual grid points changes several conclusions and is where the study's
sharpest evidence sits. Reproduced by
`scripts/c_calibration/stage_f_deep_analysis.py`.

### 7.1 The two corner channels, measured pointwise

Pointwise miss rate per grid point, pooled over all 160 cells and 42,000
replicates, by index `k` from the left:

| k | C = 1 low | C = 1 high | floored low | floored high |
|---|---:|---:|---:|---:|
| 1 | .01543 | .00048 | .00000 | .00000 |
| 2 | .00929 | .00055 | .00000 | .00002 |
| 3 | .00436 | .00081 | .00000 | .00000 |
| 4 | .00240 | .00090 | .00000 | .00000 |
| 5 | .00112 | .00076 | .00000 | .00002 |
| 6 | .00083 | .00067 | .00000 | .00005 |
| 7 | .00074 | .00071 | .00000 | .00010 |
| 8-9 | .00065 | .00080 | .00000 | .00005 |
| 10-12 | .00055 | .00085 | .00037 | .00006 |
| 13-16 | .00055 | .00092 | .00051 | .00040 |
| 30-39 | .00060 | .00063 | .00036 | .00056 |
| 100-139 | .00187 | .00074 | .00043 | .00071 |

And by index `j` from the right (`j = 1` is the last grid point before
FPR = 1):

| j | A: C = 1 low | A: floored low | B: C = 1 low | B: floored low |
|---|---:|---:|---:|---:|
| 1 | .0319 | .00000 | .1167 | .00000 |
| 3 | .0379 | .00000 | .1106 | .00000 |
| 5 | .0353 | .00000 | .1002 | .00000 |
| 10 | .0270 | .00000 | .0850 | .00000 |
| 20 | .0191 | .00000 | .0710 | .00000 |
| 40 | .0101 | .00009 | .0452 | .00015 |
| 63 | .0069 | .00004 | .0221 | .00000 |

Four things follow.

1. **Both corner channels are purely lower-edge.** The C = 1 upper-edge rate
   is flat at ~.0005-.0009 everywhere including k = 1, exactly as Lemma 13
   requires.
2. **The theory's `ceil(Q)` ~ 7 is empirically right.** The left excess is
   30x background at k = 1 and decays to background by k ~ 6-7. Restricted to
   cells with n0 >= 2000 the same profile reads .0079, .0037, .0019, .0012,
   .0007, .0006 at k = 1..6 against a .0005 background.
3. **The right channel is 20-60x larger than the left and decays far more
   slowly.** It is still 2-4x background 63 grid points in. This is the
   quantitative form of §7.4(c)5's "why the right end dominates", and it is
   why a long right region is necessary while a short left one suffices.
4. **The interior background is ~.0005 low plus ~.0006 high per grid point**,
   flat along the curve and — see §7.2 — flat in n.

### 7.2 The residual is a multiplicity effect, not a local defect

Writing `p` for the floored band's pointwise miss rate and `q` for its
whole-band miscoverage, define effective looks
`m_eff = log(1-q) / log(1-p)`.

| n0 | cells | pointwise p | whole q | m_eff | excursions per failure |
|---|---:|---:|---:|---:|---:|
| <= 200 | 30 | .00087 | .0078 | 28.6 | 1.32 |
| 201-500 | 48 | .00082 | .0117 | 60.6 | 1.39 |
| 501-1500 | 42 | .00089 | .0187 | 81.6 | 1.77 |
| 1501-4000 | 21 | .00104 | .0210 | 85.3 | 2.14 |
| > 4000 | 19 | .00078 | .0321 | 83.3 | 2.77 |

The pointwise rate is **uncorrelated with sample size** (r = −.000 against
log n0). In Study C's controlled paired design — same shape, 16x in n —
`p` goes .00080 → .00074 while `m_eff` goes 30.5 → 49.6 and excursions per
failure go 1.54 → 3.29. The implied scaling is `m_eff ~ n^0.18`.

**Inference.** The floored band's residual does not grow because any point
gets worse; it grows because a longer grid supplies more nearly independent
opportunities to fail, and failing replicates increasingly fail in several
separate excursions rather than one. This is simultaneity/multiplicity — the
Theorem 7 territory — and it is not addressable by any change to the region.

Related: **the floored band consumes 33% of its .05 error budget** on average
(mean q = .0164). The n-dependence is the band spending more of a budget it
is not exhausting.

### 7.3 No enlargement of the margin can reach the residual

Distance in grid points from each floored violation to the nearest point of
the region:

| | A | B | C |
|---|---:|---:|---:|
| median distance | 436 | 1,670 | 1,240 |
| median as a fraction of n0 | .246 | .309 | .163 |
| within 10 grid points | 1.5% | 0.9% | 0.4% |
| within 50 grid points | 7.7% | 4.1% | 2.2% |

The margin is `2 sqrt(K)` — 9 grid points on average. **Multiplying it
tenfold would recover under 2% of residual violation points.** This closes
the spec's risk row "square-root margin misses exterior violations": it does
not miss them by a little, it is not in their vicinity at all.

### 7.4 The saturated run is the trigger at the replicate level, and the floor severs it

Within each cell — so shape, n0, n1, and the true AUC are all held fixed and
only the rank realization varies — compare the empirical-TPR-1 run length K
between failing and covering replicates, in within-cell standard deviations:

| | mean std. difference | cells with positive difference |
|---|---:|---|
| A, C = 1 failing vs covering | **+1.58** | 88 / 94 |
| A, floored failing vs covering | +0.14 | 12 / 28 |
| B, C = 1 failing vs covering | **+1.27** | 26 / 29 |
| B, floored failing vs covering | −0.10 | 10 / 22 |
| C, C = 1 failing vs covering | +0.51 | 9 / 13 |
| C, floored failing vs covering | −0.28 | 3 / 12 |

Study C's association is much weaker, as it should be: its shapes are mostly
corner-concave, C = 1 fails on only 180 of 5,600 replicates there, and those
failures are largely not the run mechanism (§5).

**Inference.** At fixed DGP and fixed sample sizes, the realizations on which
C = 1 fails are the ones with long saturated runs, by more than 1.5 standard
deviations. This is the Lemma 13 mechanism observed at the level of the
individual dataset rather than inferred from cell averages. Under the floored
band the association is indistinguishable from zero: the rule does not merely
lower the failure rate, it removes the dependence on the trigger variable.

Two secondary readings from the same comparison: within-cell, `auc_hat` is
+0.46 SD (A) higher on C = 1-failing replicates, so even at fixed truth the
more separable-looking realizations are the dangerous ones; and `m50` is
**not** a replicate-level predictor (−0.17 SD, positive in 35/94 cells),
which locates §7.3(b)'s m-window as a cell-level coordinate only.

### 7.5 Where the width actually goes

Decomposition of the realized width charge, by grid-point set:

| component | n0 <= 500 | 500 < n0 <= 2000 | n0 > 2000 |
|---|---:|---:|---:|
| left points 0-6 | 29.3% | 28.9% | 31.8% |
| left points 7-`k_left` | 22.1% | 25.1% | 25.1% |
| saturated run + margin | 23.4% | 22.4% | 24.7% |
| widening closure outside the region | 25.2% | 23.6% | 18.4% |

Two consequences. First, a quarter of the charge is incurred *outside* the
region by the closure, which is the cost side of the reach noted in §3.
Second, the marginal cost of a left grid point **rises** with k — at
n0 <= 500 it is +.0085 at k = 0, +.174 at k = 1, +.251 at k = 7, then flat —
so the four points 7-10 cost about as much as the six points 0-6 that carry
essentially all the corner excess (§7.1).

---

## 8. Post-hoc: the left-cutoff frontier

**This section is exploratory and was selected after seeing outcomes.** It is
not a Stage F arm and cannot be treated as validated. It is included because
the offline scoring architecture makes the counterfactual exact — the same
stored parents, the same replicates, only the region rule changed — and
because §7.1 and §7.5 together make the question unavoidable.

| k_left | A cov | A min | A <.94 | A width | B cov | B min | B width | C cov | C width |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | .9658 | .840 | 12 | +5.5% | .9605 | .828 | +4.7% | .9777 | +5.1% |
| 3 | .9787 | .935 | 1 | +7.2% | .9789 | .958 | +5.7% | .9787 | +7.1% |
| 5 | .9818 | .940 | 0 | +8.2% | .9806 | .964 | +6.4% | .9793 | +9.6% |
| 6 | .9824 | .940 | 0 | +8.8% | .9812 | .964 | +6.8% | .9793 | +11.2% |
| 7 | .9832 | .940 | 0 | +9.5% | .9817 | .965 | +7.3% | .9795 | +13.0% |
| 8 | .9837 | .940 | 0 | +10.1% | .9820 | .965 | +7.9% | .9796 | +14.8% |
| 10 (as run) | .9844 | .940 | 0 | +11.5% | .9823 | .965 | +9.1% | .9804 | +15.5% |

Read directly: dropping the left region entirely is not viable — 12 A cells
and 4 B cells fall below .94 and the minima collapse to .84 and .83, so the
left region is load-bearing despite being the smaller channel. `k_left = 3` is
insufficient. From `k_left = 5` upward, **the minimum coverage and the
sub-.94 count are identical in all three studies**; only macro coverage moves,
by .0026 between 5 and 10 in A, while width moves 3.3pp.

Interpretation is deferred to §11.

---

## 9. What is unmeasured or underpowered

1. **Nothing is measured past n = 12,000.** Whether floored coverage
   converges to .95 or below it is not determined. Corollary 14.1's corner
   mechanism does not decay in n while the region does; the sliver cells hold
   .980 at n = 250 and .978 at n = 2,000, which is evidence against a problem
   at those scales but says nothing about n >> 10^4.
2. **The worst cell is underpowered.** `a-stress-t6.62-a0.9883-n12000` at
   .940 has a Wilson interval of (.898, .965) on 200 replicates.
3. **The margin's true effect rate.** Bounded above at roughly 1e-4 by this
   study; not estimated. §7.3 shows separately that no *enlargement* of the
   margin is worth testing, since the residual is nowhere near it; the open
   question is only whether the margin could be removed.
4. **The imbalance direction rests on 4 vs 6 cells** with uncontrolled AUC
   and df.
5. **The concavity label's discriminating power** cannot be estimated from 10
   concave / 4 ambiguous cells with one failure.
6. **The exterior term has no bound**, only measurements over three
   non-representative cell samples (§2).
7. **`alpha2` was examined only at alpha and alpha/2**, and only two levels
   of alpha were run.
8. **The §8 left-cutoff sweep is post-hoc.** It reuses the replicates that
   motivated it, so its coverage figures are optimistic for the cutoffs it
   favours by an unknown amount. No confirmation data exist for any cutoff
   other than the frozen one.
9. **The multiplicity model of §7.2 is descriptive.** `m_eff` is a derived
   quantity under an independence assumption that the excursion counts show
   is only approximate; its `n^0.18` scaling is measured over a 16x range in
   one paired design and should not be extrapolated far.

---

## 10. Composite piggyback (exploratory, quarantined by the spec)

`frontier_floor_v1` over the declared finite-range `b0.02-0.95_C2.5`
interior, Study B only. Macro coverage .9631, minimum .943, zero cells below
.94. Absolute paired widths: .934x the floored band, 1.013x plain C = 1.

The trade is about 1.9pp of macro coverage for 6.6% width against the
floor-only arm. Per the spec this arm cannot revise conclusions about the
floor-only arms and needs its own confirmation data; the interior exponent is
a fixed C > 1, which Theorem 7 excludes from an unrestricted method.

---

## 11. Speculative — qualitative impressions, not results

*Everything above is measurement. Everything here is the runner's opinion and
should not be cited as a finding.*

- **On the margin.** My impression is that the margin is close to
  free-but-useless at these sample sizes and that `run0` would behave
  identically in practice — but I would not act on it. The theoretical
  argument for a margin is a statement about the fiducial run's endpoint
  jitter, and a study powered to 1e-3 cannot refute a 1e-4 mechanism. If the
  margin is dropped, I would want a targeted high-replicate probe on cells
  with large K first.
- **On the left cutoff.** §7.1 and §8 together are, to me, the most
  actionable pair of results in the study: the corner excess is gone by
  k ~ 6-7, the theory's own `ceil(Q)` ~ 7 sits right there, and the four
  points 7-10 that the budget-derived substitute adds cost about as much as
  the six that do the work. My expectation is that `k_left = 7` is very close
  to free. But §8 is post-hoc on the same replicates, the frozen rule's
  minimum coverage was already attained at every cutoff from 5 up, and I
  would want a confirmation run on fresh seeds before anyone quoted it. I
  would also want the alpha = .5 case re-derived properly rather than patched:
  I did not measure the realized Q at alpha = .5 and cannot say the region is
  "too wide" there rather than the C = 1 band being conservative at central
  alpha for unrelated reasons.
- **On the closure.** A quarter of the width charge is incurred outside the
  region by the monotone closure (§7.5), and the closure also does part of
  the protecting (§3). I find it slightly uncomfortable that the object being
  reasoned about — "the region" — is not the object doing the work. If the
  rule is revised, I would want the closure's contribution accounted
  explicitly rather than left as a side effect.
- **On the residual.** §7.2 moved this from a guess to something closer to a
  measurement: the pointwise rate is flat in n and the whole-band rate is not,
  which is what a multiplicity explanation predicts and what a local-defect
  explanation does not. I still have not shown it *is* the Theorem 7 erosion
  rather than some other source of uniform background risk, and the
  independence model behind `m_eff` is crude. What I would now say with
  confidence is only the negative: it is not leftover corner mechanism, and no
  region change will touch it.
- **On what would move the coverage number.** If the residual is multiplicity
  against an unexhausted budget (33% used), then the lever is the trim level,
  not the floor — and that reopens the C-calibration question the Stage S
  screen closed, but now on top of a band whose corner defect is repaired.
  That strikes me as the most interesting direction the study opens, and it is
  entirely outside what Stage F was designed to answer.
- **On alpha2.** The economics at alpha = .05 favour alpha2 = alpha on these
  data. But the union-bound budget that alpha/2 buys exists to cover the
  exterior term, and the exterior term is the unbounded part of the
  decomposition. I would not read a +0.05pp coverage difference as evidence
  that the budget is unnecessary.
- **On the sliver result.** This is the result I find most compelling,
  because it is a prospective test of a construction that was designed to
  break the method, the failure is near-total on precisely the predicted
  event, and the trigger's response is two to three orders of magnitude.
  I would weight it well above the aggregate coverage numbers.
- **On what the aggregate coverage numbers are worth.** Less than they look.
  Given §2(a)-(c), "macro coverage .984" is a statement about three
  non-representative cell collections. The per-question answers in §3-§5 and
  the mechanism results in §6 are what I would carry forward.

---

## 12. Reproduction

```bash
uv run python scripts/c_calibration/stage_f_run.py design
uv run python scripts/c_calibration/stage_f_run.py run \
  --manifest data/results/hybrid_floor_20260902/manifests/study_{a,b,c}.json
uv run python scripts/c_calibration/stage_f_run.py summarize --study {A,B,C}
uv run python scripts/c_calibration/stage_f_report_tables.py --study {A,B,C}
```

```bash
uv run python scripts/c_calibration/stage_f_deep_analysis.py
```

Two analysis modules were added for this report, both re-scoring stored
records only, with no re-simulation. `stage_f_report_tables.py` recovers
*where* a violation sat, which the study summaries discard and §5-§6 need.
`stage_f_deep_analysis.py` produces §7 and §8: pointwise profiles,
residual-to-region distances, the effective-looks decomposition, the
within-cell trigger test, and the post-hoc left-cutoff sweep.

The `A/`, `B/`, and `C/` raw record directories total ~7 GB and are not in
version control; `manifests/` and `analysis/` are. Seeds are deterministic per
(study, cell, replicate), so re-running the study reproduces the records
exactly.

Study A's replay parity gate (three combined SEs against each replayed cell's
prior C = 1 coverage) passed on all 88 replay cells. The adaptive top-up rule
fired on two Study B cells (`b-wedge-08`, `b-large_n-20`, both to 800
replicates) and on no C cells.
