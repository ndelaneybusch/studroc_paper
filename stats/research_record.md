# ROC band research record, August–September 2026

*Consolidated 2026-09-26. This file replaces the eleven study specs and
reports listed in §10, and everything that was in `stats/experiments/`. They
were written round by round, mostly by earlier model generations, and each
one's conclusions were drawn from the evidence available at the time.
This record keeps the measurements, rechecks the interpretations against
later evidence, and states conclusions only as far as the data support
them. Where an earlier report overreached, §8 says so explicitly. The
originals can be recovered from git (§10).*

*Still-current companions: [`fiducial_band_theory.md`](fiducial_band_theory.md)
(proofs and mechanisms), [`next_method_ideas.md`](next_method_ideas.md)
(candidate methods), [`simulation_spec.md`](simulation_spec.md) (the paper's
main suite).*

**Evidence labels.** **[Exact]**: a proof or a deterministic property of the
construction. **[Measured]**: a simulation result, which holds only on its
design; most designs here were deliberately adversarial, so a pooled number
describes that design, not ROC curves in general. **[Inferred]**: a
conclusion that follows from measurements with a stated argument.
**[Hypothesis]**: a plausible explanation that has not been tested directly.
Coverage means simultaneous coverage of the true ROC on the native grid
`t_k = k/n0`. Area means mean band width over grid points. `n0/n1` are
negative/positive counts; a bare `n` means per class, balanced.

---

## 0. The state of things in one page

**What exists.**

| Band | Guarantee | Width at α = .05 | Calibration at α = .5 |
|---|---|---|---|
| **M3** (composed exact order-statistic bands) | **[Exact]** finite-sample, every continuous score law | 1.3–2.0× the C = 1 fiducial band; 0.37–0.88× KS | Covers .92–1.00 against nominal .50 |
| **C = 1 fiducial band** (production default) | None. Asymptotic interior theorem only (Theorem 7) | The narrowest candidate we have | Covers ~.58–.86 |
| **Hybrid**: C = 1 with an M3 floor at both ends (optional `m3_floor`) | **[Exact]** never covers less than C = 1; misses inside the floored region are capped at α₂. **No bound** on misses outside it | C = 1 + 9–16% macro on the Stage F designs (+4% at n = 8,000; +12–49% on shapes that needed no repair at n = 500); equal to M3 at n ≤ 15 | Covers ~.78–.90 |
| **Named-curve exact test** | **[Exact]** size, measured .046–.051 at α = .05 | — | — |

**What was learned.**

1. The rank-space reduction works as intended. Oracle calibration is exact at every α, the Dirichlet
   fiducial cloud has the right first-order geometry, and random tie-breaking
   is exact for the trapezoidal estimand. **[Exact/Measured]**
2. The C = 1 band is **not honest**. It fails when the ROC has features at
   the 1/n scale at its ends: convex heavy-tail hooks, unsampled slivers of
   rare mass, and jumps. Coverage then falls to .5–.7 in constructed cases
   and to .11–.25 for a step ROC. These failures are not confined to high AUC or
   small n, and coverage is not monotone in n. **[Measured]** No summary made of
   AUC and class counts can certify safety. **[Exact]**
3. The mechanism is understood. Sorted-uniform completion inside the end
   gaps amounts to assuming the ROC is locally linear there, and the lower edge
   inherits that assumption. Almost all corner misses are lower-edge.
   **[Exact geometry, heuristic risk model, Measured confirmation]**
4. A rank-only M3 floor on the two end regions repairs that mechanism on
   every construction tested, including prospective slivers. It removes the
   dependence of failure on the saturated-run length within cells. **[Measured]**
5. **Every attempt to calibrate the trim level failed.** That covers a fixed
   C = 2, an n-taper, plug-in calibration, bracketed calibration,
   fiducial-predictive calibration, shape-functional rules, a finite-range
   composite, and an interior exponent after the floor. Fixed C > 1 is
   asymptotically anti-conservative **[Exact, Theorem 7]**. The
   finite-sample surplus it exploited varies by shape in a way the data
   cannot reveal. **[Measured]**
6. M3's conservatism comes from its level accounting, not its geometry: M3
   at nominal α′ ≈ .5–.8 already covers .95. None of the attempts to
   recover that slack inside a theorem has worked yet: boundary
   optimization, joint class regions, likelihood inversion, and certified
   test inversion. **[Measured]**
7. At ≤ 50 per class the hybrid is M3 to a close approximation. **[Measured]**

**What is open.** A whole-band guarantee for the hybrid (the exterior
term). Central-α calibration of any honest band. A finite-sample
construction that beats M3's width. And the most basic practical gap: **no
fiducial variant has been run through the paper's main LHS suite**
(`simulation_spec.md` still specifies the envelope method). Every number
above comes from bespoke harnesses on hand-built or adversarial designs.

---

## 1. The problem, the vocabulary, and the cells

### 1.1 What we set out to do

The goal was to replace the studentized-bootstrap envelope with its Wilson and Beta floors
(`project_evaluation_report.md`). Its defects were that it was inelegant,
relied on magic numbers, had poor asymptotics, was miscalibrated for α > .1,
and was much wider than Working–Hotelling (WH) under binormal truth. The
desiderata were:
honest coverage on non-normal DGPs; calibration across α; tightness
between WH and KS; invariance to AUC, shape, and n; and small misses that
are balanced by direction and location.

The organizing insight was **rank invariance**. Put negatives through their own CDF and
they become Uniform(0,1), and the positives then have CDF exactly R, the
true ROC. Any rank-based band's coverage therefore depends only on
`(R, n0, n1, α)`, and it can be simulated exactly for any hypothesized
curve. **[Exact]** This makes the research program unusually clean: cells
are ROC *shapes*, parameter sweeps within a score family are redundant, and
"works on heavy tails" means "works on the ROC shapes heavy tails produce".

### 1.2 Vocabulary

| Term | Meaning |
|---|---|
| Fiducial cloud | M ROC draws. Each class's CDF gets Dirichlet(1,…,1) spacings; the other class's points are placed at sorted-uniform fractions inside each gap. |
| Min-p depth, `D_b` | A draw's minimum over trim columns of its two-sided pointwise rank in the cloud |
| `j`, `ℓ` | Trim depth (the α_eff-quantile of the depths) and realized local level `ℓ = j/(M+1)` |
| `C`, `α_eff` | Trim exponent: `α_eff = 1 − (1−α)^C`. C = 1 is the identity map. |
| `C*` | Per-cell calibration ceiling: the largest C whose realized coverage still reaches 1−α |
| CP allowance | Upper edge unioned with a Clopper–Pearson-form bound at level ℓ, plus a lower edge of 0 where the empirical TPR count is 0 |
| Saturated run, `K` | Negatives scoring below the lowest positive, i.e. the grid points where empirical TPR = 1 |
| M3 | Exact composition of two one-sample equal-local-level order-statistic bands, one per class (theory Prop. 12) |
| Hybrid / floored band | Pointwise hull of C = 1 and M3 on a rank-selected end region, closed by widening |
| Oracle | The fiducial trim calibrated against the true curve: a benchmark, not a method |

### 1.3 The early-round cells

Tables in §3 use these names from rounds 2–4.

| Cell | Truth | n0/n1 |
|---|---|---|
| C1 / C2 / C3 | binormal AUC .75 / .95 / .95 | 500 / 500 / 150 (balanced) |
| C4 | bimodal-negative AUC .90 (touches TPR = 1) | 500 |
| C5 | t(2) AUC .95 (convex hooks at both ends) | 500 |
| C6 / C7 | binormal .95 / .90 | 5,000 / 25 |
| P2a / P2b | binormal .90 | 900/100 and 100/900 |
| P2c / P2d | binormal .99 | 500 / 150 |
| P2e / P2f | binormal .55; kink (vertical to TPR .6 by FPR 2/n0, then linear; AUC .80) | 500 |
| P4a/b/c | binormal .95 | 500 / 2,000 / 5,000 |

---

## 2. Chronology

| Date | Study | Question | Answer at the time | What later evidence did to it |
|---|---|---|---|---|
| Aug 21 | Round 1 (`rank_band_experiments.py`) | Does the rank-space machinery work, and which band construction? | Oracle exact; plug-in calibration dead; fiducial + CP allowance works | Stands |
| Aug 21 | Round 2, M2 (`m2_report`) | Recalibrate the level; probe imbalance, AUC, kink, ties, M | C = 2 "centres" coverage; no validity failure anywhere | C = 2 refuted (Stage S); "no failure" was library-limited |
| Aug 22 | Round 3, M3/M4 (`m3m4_report`) | Is M3 a usable guarantee layer? Bracketed calibration? What is C\*(n)? | M3 valid but 1.5–2× wider; M4b falsified; C\* → 1 | Stands; "pin U(0)=0" corrected same day |
| Aug 23 | Round 4 (`r4_report`) | Last calibration ideas; exact named-curve test; M3 remap ceiling; corner repair | All calibration routes fail; exact test delivered; corner repair via M3 dead at usable levels | Stands; the "roughness axis" is still unidentified (§7.1) |
| Aug 29 | Stage S (`c_calibration_screening_report_stage_s`) | Is an auto C-map worth fitting? | STOP. First validity failure found (t(2), n = 100); C = 1 becomes the default | Validity failure turned out to be far larger in scope |
| Sep 1 | Boundary follow-up (`c_calibration_followup_report`) | Where does C = 1 fail? | A curved (AUC, n) wedge, non-monotone in n; localized M3 floor proposed | Wedge framing superseded: slivers fail at any AUC |
| Sep 2–4 | Stage F (`hybrid_floor_spec`/`report`) | Frozen rank-only floor: capture, price, transfer, slivers | Floor repairs every corner mechanism tested; residual lies deep in the interior | Stands; small-n and interior questions followed |
| Sep 5–6 | Theory checks and ideation screen (in `experiments/`, reported in the theory doc and `next_method_ideas.md` §4.5) | Verify new formulas; cheap M3 variants | Formulas verified; sample-size split helps under imbalance | Stands |
| Sep 9–10 | Methods exploration (`methods_exploration_*`) | Small n; interior exponent after the floor; likelihood and test inversion; M3 boundaries | Small n ≈ M3; no interior C > 1; exact routes not yet competitive | Stands, with caveats (§8) |

---

## 3. The research questions and what was learned

### Q1. Is the rank-space fiducial construction a sound basis?

**Yes, as a first-order construction.** Evidence:

- **Oracle equal-local-level (ELL) machinery is exact.** Calibrated against the true curve, it hits
  nominal at every α and shape from n = 25 to 5,000. It gives the width
  ceiling for rank-based bands, about 1.6× WH's area under binormal truth
  (KS is about 3×). **[Measured, round 1]**
- **Plug-in calibration (M1) fails.** It covers 0.28–0.43 at α = .05 with raw or
  Hazen-smoothed plug-ins. The mechanism is Wald-type: the plug-in curve
  co-moves with the data, shrinking the simulated dispersion exactly when
  the realized deviation is large. **[Measured, round 1]**
- **The fiducial cloud with a CP allowance works on smooth shapes.** It covered .968–.995 at
  α = .05 on 12–14 cells spanning n = 25–5,000, AUC .55–.99, 9:1 imbalance
  both ways, bimodal, t(2), and kink truths. Misses were balanced and p95
  miss depth was 0. Area was 21–66% of KS and 1.05–1.35× the oracle, except
  1.9–3.2× at AUC .99 with small n. WH covered 0.000 on bimodal, t(2), and kink.
  **[Measured, rounds 1–2]** Theorem 7 gives the matching asymptotics: on a
  regular interior the cloud converges to the right Gaussian process, and
  interior-calibrated trimming covers at `1 − α_eff`.
  **[Asymptotic, proof outline]**
- **The CP allowance is load-bearing, and it was designed in-sample.** The raw band
  missed the bimodal cell in 100% of reps, because the truth touches TPR = 1 and no
  credible edge reaches 1. The allowance was added after that was seen.
  Wherever the truth hits the plateau (bimodal .90, binormal .99), the
  truth's cloud depth is 0 in essentially every replicate and the allowance
  carries all the coverage. **[Measured]** A full mirrored lower CP bound costs
  +15% area for nothing; the degenerate mirror (L = 0 where k̂ = 0) is
  free. **[Measured]** Pinning U(0) = 0 is **not** free validity: separated
  supports give R(0) > 0 (theory §1, §9). **[Exact]** Round 3's contrary
  recommendation was wrong and was corrected at the time.
- **Ties.** Random tie-breaking is exact for the trapezoidal (Mann–Whitney)
  ROC (theory Prop. 2a), and at Q = 20 and 100 levels it was
  indistinguishable from continuous scores. Class-ordered tie-breaking
  changes the estimand and gives coverage 0.000. **[Exact/Measured]**
- **Monte Carlo budget.** M must scale with the grid: the needed local level
  falls as K grows (`ℓ ≈ 9.7e−4 (α/.05)^1.2 (K/500)^−0.27`), so `M ≳ 5/ℓ`.
  Saturation (j = 1) does not break validity, but it destroys α-resolution.
  Trimming on a thinned grid is leak-free and gains little. **[Measured]** The
  production rule is self-diagnosing: it warns when j < 3.

### Q2. Is the C = 1 fiducial band honest? The answer changed three times

This is the program's central empirical story, and how it unfolded is instructive.

**Round 2: "no validity failure anywhere."** The library did hold kinks, 9:1
imbalance, AUC .55–.99, and n = 25. But its only convex-hook shape, t(2)/.95,
appeared only at n = 500. There it was consistently the worst cell: lowest C\*,
lowest M3 α′, and the cell where C = 2 undercovered (.948). That was the warning sign.
At the time it was read as a "roughness" axis, not an endpoint phenomenon.

**Stage S (Aug 29): the first failure.** At t(2)/.95, n = 100, C = 1 covered
**.802**. Ninety-five percent of misses were lower-edge, bimodal in location (k ≤ 3 and
k ∈ [94, 99]), and the truth left the whole untrimmed cloud in 1–2% of
reps. **[Measured, 500 reps]** At the upper corner the truth's deficit,
1 − R(.96) = .0079, sits below the 1/n1 = .01 resolution. At the lower
corner the truth climbs .07 → .71 across three grid points. The report
concluded "C = 1 is safe for min(n0, n1) ≥ 500". That was an extrapolation.

**Boundary follow-up (Sep 1): a wedge, non-monotone in n.** Across 257
t-family cells and 64,625 reps:

| Shape | n = 250 | 500 | 1,000 | 2,500 |
|---|---|---|---|---|
| t(2)/.99 | .645 | **.690** | .842 | .951 |
| t(1.1)/.97 | .903 | .956 | .975 | .980 |

At fixed shape t(4.69)/.986, coverage runs .993, .947, .903, .823, .847 over
n = 150 → 2,000. So coverage is **not** monotone in n. **[Measured, 300 reps]**
Failures worst-cased over df by AUC band:

| AUC | .50–.90 | .90–.94 | .94–.96 | .96–.975 | .975–.985 | .985–1 |
|---|---|---|---|---|---|---|
| Failing cells / cells | 1/65 | 15/53 | 10/35 | 14/36 | 5/26 | 20/42 |
| n range of failures | 102 | 103–248 | 110–452 | 124–1,051 | 199–5,131 | 160–6,656 |

A held-out library passed 10/10 cells (worst .967), but it never went above
AUC .90 or included a heavy tail at high AUC, so it cannot speak to this region.
Cross-family spot checks at achievable corners (Weibull, gamma, beta-opposing) passed at .977–.992.

**Slivers and jumps: not a wedge at all.** Theory (theory §9, §12.3) showed
that AUC and class counts cannot identify endpoint risk. Put a sliver of
positive mass π = d/n1 where it goes unsampled with probability `e^−d`, at any AUC and any n.
Stage F's prospective sliver block confirmed it **[Measured, 400–1,200 reps]**:

| Cell | AUC | n0×n1 | P(unsampled) predicted / observed | C = 1 overall | C = 1 given unsampled | Hybrid |
|---|---|---|---|---|---|---|
| 24 | .60 | 250×250 | .367 / .413 | .613 | .097 | .980 |
| 25 | .60 | 2000×2000 | .368 / .393 | .590 | .000 | .978 |
| 26 | .80 | 250×250 | .449 / .473 | .520 | .000 | .988 |
| 27 | .95 | 2000×2000 | .449 / .485 | .505 | .000 | .983 |
| 28 | .80 | 2000×500 | .449 / .408 | .580 | .000 | .978 |
| 29 | .80 | 500×2000 | .449 / .455 | .525 | .000 | .978 |

"Zero" means zero observed in 157–194 conditioning reps. The one-sided 95%
upper bounds are 1.5–1.9%. On a step ROC (a jump at FPR .5), C = 1 covered .238,
.184, .136, .108 at 10, 20, 30, and 50 per class. It gets **worse** with n.
**[Measured, methods exploration]**

**What is established.**

- The mechanism **[Exact geometry; heuristic risk model; Measured confirmation]**:
  the cloud's within-gap completion spreads the end spacing uniformly,
  which acts like local linearity in the end gaps. At a convex hook or an
  unsampled sliver, the lower edge claims a TPR deficit of order `ln(1/ℓ)/(n1·K)`
  where the truth's is order 1/n1. So the lower edge is anti-conservative
  (theory §7.4). Stage F's pointwise profile shows both corner channels are
  purely lower-edge. The upper-edge miss rate is flat at ~.0005–.0009 per grid
  point everywhere, including k = 1. The right channel is 20–60× the left
  channel and decays slowly: it is still 2–4× background 63 points in. **[Measured]**
- At the replicate level, within a cell (same shape and n), failing
  replicates have saturated runs 1.58 SD longer than covering ones (A: 88/94
  cells positive; B: +1.27 SD, 26/29). **[Measured]** This is the trigger,
  observed directly rather than inferred from cell averages.

**What is not established.** How often these features occur in real
classifier score distributions. Everything here is a stress design.
Corollary-level theory says corner-*concave* ROCs should be safe at leading
order. In Stage C, 10 of 10 pre-labelled concave cells passed and the one
failure was labelled "ambiguous". That is consistent with the theory, but
it cannot measure how well the label separates safe from unsafe shapes.
**[Measured, small]**

### Q3. Can the corner defect be repaired locally?

**Yes, as far as we have measured. There is no whole-band theorem.**

**The construction.** Inside a rank-selected end region, take the hull of
C = 1 and M3. Outside it, keep C = 1. Close by widening only. Three exact properties hold for any region selection
**[Exact]**:
(i) the hybrid contains C = 1 pointwise, so its coverage is never lower for any DGP or
replicate; (ii) a hybrid miss inside the region implies a full M3 miss, so
misses in the region are ≤ α₂; (iii) `P(miss) ≤ α₂ + P(C = 1 misses outside the region)`.
Nothing bounds the last term.

**Evolution of the region rule.**

1. Follow-up probe: FPR ∈ [0, .005] ∪ [.5, 1], chosen in-sample on 5 cells.
   It lifted failing cells from .72–.94 to .955–.990 at +6.4% mean width, against
   +28–46% for full M3. **[Measured, 100–200 reps]**
2. Stage F `frontier_floor_v1`, frozen before any outcome: the left region is the first
   `⌈log(M+1)⌉` grid points (9–10 here); the right region is the saturated run
   extended inward by `⌈2√K⌉`. It uses only class sizes, M, and ranks.
3. The current production option (`m3_floor`, "exact" rule, 2026-09-07). The left cut is the smallest k with
   `P{Binom(M, (1−k/n0)^n0) ≥ j} ≤ .001`; the right margin comes from inverting a Beta
   bound on the saturated-run boundary (δ = .025; theory §10.1–10.2). It
   adapts to α through j. Stage F's rule is still available as `"stage_f"`.

**Stage F results** (160 cells, 42,000 paired reps; rule frozen in advance) **[Measured]**:

| Design, α = .05 | Cells | C = 1 macro (min) | Hybrid macro (min) | Width vs C = 1 | Full M3 width |
|---|---|---|---|---|---|
| A: enriched replay (74% previously failing) + imbalance LHS + large-n stress | 116 | .926 (.570) | .984 (.940) | +11.3% | +41.4% |
| B: prospective wedge, safe, imbalance, large-n, and sliver cells | 30 | .829 (.505) | .982 (.965) | +9.1% | +42.9% |
| C: six non-t shapes plus one t draw, each at n = 500 and 8,000 | 14 | .968 (.865) | .980 (.970) | +15.5% | — |

- The measured in-region failure rate is .0002–.0003, against a cap of .05. So the exact
  component is not what binds.
- Lower-edge failing replicates fell 1,413 → 130 (A) and 1,990 → 98 (B).
  Upper-edge counts moved less (318 → 234, 179 → 137). The residual is
  roughly balanced, with an upper tilt.
- **Residual violations are far from the region.** The median distance is .16–.31 of n0,
  and under 2% of residual points lie within 10 grid points. Enlarging the margin
  cannot reach them.
- The `2√K` margin changed one replicate outcome in 23,200 relative to no margin. The study can resolve
  effects of order 1e−3, not 1e−4. **[Measured]**
- Post hoc on the same reps (so optimistic): a left cut of 5 or more grid points gives
  the same minimum and the same sub-.94 count as 10. Dropping the left region entirely
  leaves 12 A cells below .94. The pointwise left excess decays to background by
  k ≈ 6–7, and the exact rule lands at 5–7 at α = .05. **[Measured]**
- Width decomposition: about 55% of the charge comes from the left region, about 45% from the run
  plus margin, and **a quarter of the total is incurred outside the region by
  the widening closure**. The closure also accounts for part of the protection:
  floored miscoverage is below the exterior-escape rate in all three studies.
  **[Measured]**

**Price where no repair was needed.** At n = 500 on concave Study C cells, the
floor added +0.2 to +0.7 pp of coverage at +12% to +49% width. The worst case was
beta-opposing: .993 → .995 for +48.8%, against +79.3% for M3. At n = 8,000 the average
charge was +4.4%. **[Measured]** The floor is cheap at large n and expensive
at moderate n on high-AUC concave shapes, where the saturated run is long.

**Small samples** (methods exploration, 10–50 per class, 9 shapes, 40,500
datasets) **[Measured]**: at n0 = 10 the floor covers the whole grid in 100% of reps, and
the hybrid equals M3 to machine precision. A real unfloored interior appears only
around n0 ≈ 50, and even there only .16 of cells are unfloored for binormal .95. The hybrid
covered ≥ .984 in all 81 cells at α = .05 and was never wider than M3 (0 of 162 cells).
Doubling M changed areas by ≤ 3%. The right-floor component does almost all
the repairing (jump at 50/50: right-region coverage .108 raw; the other regions ≥ .992).
Mean interior width at 10/10 is .49–.95 of the unit interval. Small-sample width is
M3's width.

**At α = .5** the region does not shrink with α in Stage F's rule. The hybrid
moved macro coverage from .584 to .781 (nominal .50) at +19.7% width.
The exact rule is α-adaptive and 4.5% narrower at α = .5 in the small-n study,
but its coverage was still .888. **[Measured]** The floor does not address
central-α calibration, and it makes C = 1's over-coverage there worse.

### Q4. Can the trim level be calibrated? (The C question)

**No. Nothing tried survives, and the reasons are now well understood.**

**Round 2: fixed C = 2.** Fitted on 4 cells, it transferred to 10 held-out
cells. It centred coverage (bias +.24/+.12 at α = .5/.2 → +.03/+.03) for 9–18% less area,
and coverage stayed ≥ .942 at α = .05. **[Measured]** The kill criterion
(a shape spread over 10 pp at fixed α) was *partially triggered*: 13 pp at α = .2
and 19 pp at α = .5. A level-only remap removes bias, not spread.

The "Šidák per class" reading (C = 2 as structural) was **falsified** in round 3.
At α = .5, C\*(n) = 3.06, 2.40, 2.18, 1.71, 1.79, 1.49 ± .17, 1.32 ± .16 for
n = 25 … 20,000. That is 4.2 SE below 2 at the top. **[Measured]**
Theorem 7 explains why: interior-calibrated fixed C has limiting coverage
(1−α)^C. **[Asymptotic]**

**Stage S** (27 cells, 52,000 reps, α = .05) **[Measured]**:

| Shape (n = 500) | C\* ± SE | Oracle area gain vs C = 1 |
|---|---|---|
| t2_95 | 1.17 ± .21 | 2.2% |
| kink_80 | 1.56 ± .22 | 5.4% |
| trapezoid_q10_90 | 2.01 ± .24 | 8.0% |
| binormal_95 / _90 | 2.23 / 2.28 | 9.8 / 9.7% |
| hetero_90_r3 | 2.36 | 10.8% |
| binormal_99 | 2.60 | 13.4% |
| bimodal_90 | 2.66 | 11.7% |
| binormal_75 / _60 | 2.81 / 2.99 | 11.6 / 12.1% |

| C\* at α = .05 | n = 100 | 500 | 5,000 | 50,000 |
|---|---|---|---|---|
| binormal_95 | 3.05 | 2.23 | 1.78 | 0.87 ± .12 |
| kink_80 | 2.45 | 1.56 | 1.51 | 1.03 ± .15 |
| t2_95 | (pinned) | 1.17 | 1.49 | 1.07 ± .15 |

- The shape envelope at n = 500 is 0.967 (t2), against a pre-registered
  requirement of 1.15, so the verdict was STOP. The mean oracle gain of 9.5% is real, but
  it can only be reached per shape.
- C\* reaches ≈ 1 on all three shapes by n = 50,000, where C = 1 covers
  .951, .954, and .960. **This directly confirms Theorem 7's limit at α = .05.**
  C = 2 covers .914–.925 there, so it undercovers every shape at large n.
- The approach is not a shared law. t2's C\* is non-monotone, and kink is flat over
  500 → 5,000.
- Imbalance at a fixed minority of 500 moves C\* by 1.0 (3.5 SE) on binormal_90,
  so a min(n0, n1) reduction is rejected there.

C = 1 became the production default on 2026-08-30. That was the user's
decision, made on this evidence.

**Data-driven levels, all failed** **[Measured]**:

| Route | Result | Why |
|---|---|---|
| Per-rep plug-in calibration (round 2) | Depth 1.3–1.7× too conservative; 80× compute | Plug-in inherits one dataset's roughness |
| M4b bracket over M3-50% members (round 3) | 9–37× too conservative; returns one α-independent band | The worst case selects the roughest member |
| Smoothed bracket / smoothed predictive | ≈ plug-in (1.2–1.9×) | Adds a smoothing constant and buys nothing |
| Fiducial-predictive calibration (round 4) | 1.7–2.3× conservative, *worse* than plug-in | At q05 the truth sits ≈3× deeper in its cloud than a draw; candidates sit at depth 0 |
| Shape-functional level rules, 32 functionals (round 4) | No held-out gain beyond noise; 7/14 cells fall below .94 at α = .05 | Functionals carry too little shape information; the co-movement bias is **not** the problem here (\|ρ\| ≤ .25) |

**Composite band (follow-up, item 3).** A stitched band took the nearly
untrimmed cloud envelope (plus allowances) on FPR < .02 and > .95, and
trimmed the interior at C = 2.5. It passed every n ≥ 500 core cell
at −6.8% pooled width. The saving inverted at n = 20,000, as Theorem 7
requires. On Stage F's B cells, piggybacked on the floor, it covered .963 (min
.943) at .934× the floored width. **[Measured, finite range only]** A fixed
C > 1 is excluded from any unrestricted method.

**Interior exponent after the exact floor (methods exploration).** A fixed
[.02, .95] window was re-trimmed only when it cleared both floor masks. **[Measured]**

- Eligibility is decided by geometry. The window was never eligible at n = 100 (0 of 30,000)
  and rarely at n = 500. The left cut misses the window by one or two grid columns there.
  On the right, binormal .95's saturated run blocks the window until n0 ≈ 50,000, and
  under 9:1 negative-majority it is eligible only 30% of the time even at n0 = 90,000.
- On eligible datasets, the windowed C = 1 band already sits at or near nominal on the
  interior. C\* brackets are [1, 1.25] or censored below 1 in 13 of 27 estimable cells.
  Binormal .95 at n = 5,000 remains at [1.5, 2]. The windowed band was .92–1.00 of the floor's area
  and covered .902–.975, against the floor band's .936–.990.
- **Eligibility is a strongly informative event.** For binormal .95 at
  n = 500 (n0 = 900), the floor band covers .983 on ineligible datasets and
  .400 on the 5 eligible ones; at α = .5 it is .774 against .000 (20 eligible). The window
  clears only when the saturated run comes out anomalously short, which is
  exactly when the band is in trouble.

Reading (§8 revises the original report's framing): trimming on the interior at
C = 1 does recover some width, and it lands near nominal on the interior, as
Theorem 7 predicts for interior-calibrated trimming. There is no room left for
C > 1. The eligibility gate makes the conditional law unrepresentative, and a
fixed-FPR window is inert on most realistic datasets.

### Q5. Can M3's conservatism be recovered?

**Its size is well measured. Nothing yet recovers it inside a theorem.**

- **Baseline** (round 3, 8 cells, 400 reps): coverage 1.000 at α = .05
  (0 misses in 3,200) and .978–.998 at α = .5. Area was 1.33–1.94× C = 1 and 0.37–0.88× KS.
  The ratio to C = 1 is roughly uniform across FPR; the worst cells are the AUC .99 cells.
  In Stage F full M3 cost +41–43% over C = 1; in the held-out cells, +39–49%. **[Measured]**
- **Where the slack lives.** The nominal α′ at which M3 realizes .95 is 0.50–0.85
  across 14 cells. That is a factor of about 10–17 in α. At that level M3's area is 0.93–1.05×
  C = 1's. **[Measured]** The matched-coverage comparison uses the truth. It
  shows M3's shape is not the problem on these smooth cells; it is not a method.
- **Remap ceiling** (round 4, 14 cells): α′ = .5 covers ≥ .950 on all 14 (min
  exactly .950, at 900/100) at 1.07–1.43× the C = 2 band's area. The margin is one ladder
  step. The binding cell is **imbalance** (900/100), not shape, and the needed α′
  falls about .13 per decade of n. **[Measured]** A library-fitted remap would
  forfeit the theorem, which is M3's only reason to exist.
- **Miss cap** (fiducial ∩ M3(α/10)) never binds (0 of 10,400 band checks). The
  certified depth bound is 0.10–0.90, against observed misses of 0.01–0.06. So it is inert.
  **[Measured]**
- **Domination route to a fiducial theorem** (does M3(α′) ⊆ fiducial?): it essentially
  never holds, at any α′ up to .999, or even after trimming 25 end points. **[Measured]**
- **M3 as a steep-corner repair** (round 4): at usable α₂, M3's corner edges are
  1.7–4.6× wider than the fiducial band's. It becomes tighter only at nominal
  α₂ ≳ .3–.9. **[Measured]** The fiducial band is *narrower* than M3 at
  the corner. Its failure there is in *location* (the lower edge sits too high), not in excess width.
  That is the same fact that makes the Q3 floor cheap relative to full M3.
- **Boundary optimization** (methods exploration): a 27-member family
  (ELL↔KS interpolation, an iterated-log tail modifier, a class-split exponent), each
  calibrated to exact non-crossing probability. The selection rule required every
  training shape's tail width to be ≤ 1.00× M3, with zero tolerance. Unmodified M3
  won all 15 count/α designs. Alternatives with smaller area all lost a tail.
  One exception is the class-split exponent s = .5 at 450/50, α = .05: it saved 1.2–2.4% area on every
  training shape and was vetoed by a left-tail ratio of 1.0009 on one shape.
  **[Measured]**
- **Ideation screen** (Sep 5, `next_method_ideas.md` §4.5, 400 reps, 12 combinations):
  the sample-size split `ρ = n0^−½/(n0^−½ + n1^−½)` gave 0.9–5.8% lower paired area
  on 8 imbalanced cells. A fixed rank weighting was mixed (.98–1.05×). The joint
  class-region (`p0·p1 ≥ k`) was mixed (.93–1.12×): it helped near the
  diagonal and hurt on heavy tails. At α = .5, every M3 variant covered .95–1.00. **[Measured]**

The class split is the one M3 lever that has survived two independent screens
(the ideation screen and the boundary optimization). It depends only on
`(n0, n1)`, so the theorem is unaffected. The gains are a few percent.

### Q6. Exact alternatives beyond M3

- **Named-curve test** (round 4): H0: R = R0 is simple in rank space, so a
  Monte Carlo min-p depth test is exact. Size was .190–.202 at α = .2 and
  .046–.051 at α = .05 on five cells, including t(2) and bimodal. Power at
  n = 500 was .23–.26 at ΔAUC = .01 and .71–.85 at .02, halving at n = 150.
  Min-p is about 2.5× less sensitive to a localized early-FPR deviation
  than to a global one of the same sup-norm size. It detects a corner pushed down more easily than one pushed up.
  **[Exact size; Measured power]** This is a usable deliverable
  (e.g. non-inferiority to a named curve) independent of the band question.
- **Rank-likelihood (e-value) inversion** (theory §§12.5–12.7; methods
  exploration at n = 5–20) **[Exact validity; Measured losses]**:
  - Predictive loss is essentially solved: a fixed seven-component binormal mixture
    costs about 1 nat on smooth truths. A fitted sequential predictor never beat it.
    The uniform predictor costs 14 nats at AUC .95.
  - Cell loss is not solved. With 12–36 fixed cells at n = 20, the
    likelihood bracket is 10⁴–10⁹ times the rejection cutoff.
  - The inner (finite-library) hull looks like 0.39–0.50× M3. A certified outer box
    search at the same budget returned essentially the unit square: 1.34× M3 on the
    same data, with 62 of 63 boxes unresolved.
- **Certified outer projection of monotone rank tests** (methods exploration)
  **[Exact validity; Measured width]**: exact containment checks never failed.
  With 3 interior knots, the outer band was .97–1.05× M3 at total n = 8 and
  1.19–1.67× at 25/25. Region-specific statistics paid for their Bonferroni cost only
  on the step ROC.
- **Exact formula checks** (Sep 5–6): the bracket-area formula, left-cut
  minimality, the right-margin table, binomial-limit coverage, the rank-likelihood
  identities (2,358 anchor/path cases; 544 one-cut identities), and enumerated
  small-n coverage. All passed. These verify formulas and code. They are not coverage evidence for any method.

The inversion screens show that exactness is cheap and **width is the whole
problem**. Both inversion routes were tested only with coarse,
non-adaptive resolution: fixed cells, and three knots. §8 explains why that limits
what the negatives mean.

### Q7. Imbalance

Imbalance was first tested at 9:1 in both directions in round 2. It was
valid at α = .05, but not probed as a calibration axis. The later picture is §7.4.

---

## 4. How the protections are built, and why they work

In one sentence per component:

- **Interior:** fiducial composition reproduces the two-sample Gaussian
  error process, including the cancellation between classes that M3's rectangle
  discards (theory §6). This is why C = 1 is narrow.
- **Ends:** in the end gaps, nothing in the data determines where the other class's mass
  lies. Honest bands must leave O(log(1/α)/n) room there (theory §9's
  missing-mass bounds). M3 leaves that room. Sorted-uniform completion does not.
- **Floor:** a rank-measurable region (the first few grid points and the saturated
  run) covers exactly the gaps where the completion assumption acts. Hulling with M3 there
  restores the missing room. Monotone closure spreads the lowered lower edge leftward.
- **What is left:** outside the region, C = 1's own interior behaviour. At
  moderate n its interior is conservative (§7.1); as n grows it approaches nominal
  (Theorem 7, Stage S).

---

## 5. Width, in one place

| Comparison (α = .05) | Ratio | Source |
|---|---|---|
| C = 1 vs oracle rank band | 1.05–1.35× (smooth, n ≥ 150); 1.9–3.2× at AUC .99, small n | Round 2 |
| Oracle rank band vs WH (binormal) | ≈ 1.6× | Round 1 |
| C = 1 vs KS | 0.21–0.66× | Round 2 |
| C = 2 vs C = 1 | 0.87–0.91× | Round 2 |
| M3 vs C = 1 | 1.33–1.94× (round 3); +28–49% (follow-up, held-out, Stage F) | — |
| M3 vs KS | 0.37–0.88× | Round 3 |
| Hybrid vs C = 1 | +9–16% macro (Stage F); Study C pairs +26.6% at n = 500, +4.4% at n = 8,000; +12–49% on safe concave shapes at n = 500; ≡ M3 at n ≤ 15 | Stage F, exploration |
| Hybrid vs M3 | ≤ 1 always observed; median .967 at 50/50 | Exploration |
| Certified test / likelihood inversion vs M3 | 0.97–1.67× | Exploration |

WH comparisons stopped after round 2. The paper's comparison against WH on
the LHS suite has not been run for any fiducial variant.

---

## 6. Monte Carlo and design caveats that affect many numbers

- **Adversarial designs.** Stage F A is 74% previously failing cells, and B is 16/30
  wedge or sliver cells. The exploration's shape libraries are mostly stress shapes. Macro
  coverages are statements about those designs.
- **Replication.** Coverage SE is ≈ 1.1 pp at 400 reps and ≈ 1.5 pp at 200
  near .95. Most Stage S C\* values have SE .15–.33, above the target. Many
  per-cell differences quoted in the originals are within noise. The
  load-bearing results are the large contrasts: sliver conditionals, the
  replicate-level trigger, the wedge's existence, and C\* → 1 at n = 50k.
- **Cloud budget.** The methods-exploration interior track ran at M = 2,000–4,000,
  below production's 5,158–17,874. Its full-grid parent band was trimmed 1.3–2.1× harder (narrower)
  than a deployed band. Its absolute floor-band coverages at n ≥ 5,000 are not production numbers.
  The raw records of that screen are not in this checkout.
- **In-sample choices.** The CP allowance (bimodal), C = 2 (4 cells), the follow-up
  floor region (5 cells), and Stage F's post-hoc left-cut sweep were all chosen on the data
  that scored them. Stage F's frozen rule and the sliver block are the
  cleanest prospective tests in the program.
- **Harness vs production.** Rounds 1–4 used a Python harness. Stage S onward used
  the Rust production path, after a parity gate (statistical, plus exact same-seed checks).
  Round 3–4's M3 used Monte Carlo ELL calibration; production M3 uses an exact DP.

---

## 7. Through-lines across the studies

These are patterns that no single report could see.

### 7.1 The ends explain the validity failures, but not most of the conservatism (tested)

The ends of the grid have been the most consequential variable for
**validity**: every failure mechanism in Q2 sits there. The obvious extension
is that they also explain the **calibration** puzzle, i.e. C = 1's
over-coverage at moderate n and the unidentified shape axis of rounds 3–4.
This consolidation proposed that. Four observations pointed that way:

- Stage S's lowest C\* at n = 500 were the shapes with end features:
  t2_95 at 1.17 and kink_80 at 1.56. The interior-rough trapezoid sat
  mid-library at 2.01.
- Round 4 found the truth deeper in its cloud than a draw at the 5% depth quantile.
- The exploration's full-grid trim depth had a median of 3, against 13 for an
  interior-only window.
- Interior C\* brackets were low after flooring.

The hypothesis was **tested directly on 2026-09-26 and mostly failed**
**[Measured]**. The design was:
- five Stage S shapes, n = 500 per class, M = 5,000, 400 reps;
- the production Python cloud and allowances;
- end columns FPR ≤ .02 and ≥ .95, interior (.02, .95).

| α = .05 | bn95 | bn60 | t2_95 | kink_80 | trapezoid |
|---|---|---|---|---|---|
| Draws with min-p depth attained at an end (7% of columns) | .27 | .31 | .42 | .39 | .25 |
| Mean j, full grid → interior only | 4.8 → 5.9 | 3.1 → 4.2 | 6.0 → 9.4 | 5.3 → 7.6 | 4.0 → 4.9 |
| Interior coverage, full-grid trim | .990 | .963 | .978 | .980 | .970 |
| Interior coverage, interior-only trim | .985 | .958 | .975 | .978 | .970 |
| Interior area saved by interior-only trim | 2% | 3% | 5% | 4% | 2% |
| C\* full curve → interior (SE .4–.8) | 2.18 → 2.21 | 1.36 → 1.45 | 1.52 → 1.99 | 1.55 → 1.96 | 1.79 → 2.15 |

At α = .5 the interior-only trim saved 5–11% of interior area. Interior
coverage was still .655–.728 against .50. The C\* range across shapes narrowed
only from 1.51–2.25 to 1.45–2.05, with t2 still lowest; SEs there are ≈ .1–.2.

What this establishes:

- **The ends pull more than their share, but they don't set the depth.** The
  end columns take 4–8× their per-column share of draw minima, and removing
  them raises j by 1.2–1.6×. That buys only 2–5% of interior width at α = .05.
- **At n = 500 the interior itself is conservative.** The truth is deeper
  than the draws on the interior alone (bn95 q5: 15 against 6; t2: 21 against 9).
  The interior-trimmed C = 1 band still covers .96–.99 at α = .05 and
  .66–.73 at α = .5.
- **The shape spread survives removal of the ends.** For t2 and kink,
  part of their low whole-curve C\* at α = .05 does come from end misses
  (≈ 1.5 → 2.0), but that change is within noise.

Reconciling with the exploration: there, interior-trimmed C = 1 landed near
nominal at n = 5,000. Here it is well above nominal at n = 500. **[Inferred]**
The surplus is a finite-n property of the interior cloud, decaying with n as
Theorem 7 and Stage S's C\* → 1 require. It is not an artifact of the ends.
Its shape dependence is still unexplained. The remaining candidates are the
finite-n discrepancy between how rough the cloud's draws are and how rough
the truth is, everywhere along the curve (round 4's depth contrast measured
this on the full curve), and correlation-length differences between the cloud
and the sampling process. The diagnostic script and JSON were one-off
scratch artifacts. Their numbers are recorded only here.

Practical consequence: excluding the floored ranges from the trim is
coherent, because the floor's regional cap is unaffected. But at moderate n
it is worth only a few percent of width at α = .05 and roughly 5–10% at α = .5.

### 7.2 Library-calibrated rules keep failing in the same way

Each round expanded the library after the previous one's rule broke:

| Rule | Calibrated on | Broken by |
|---|---|---|
| C = 2 | 4 cells at n = 500 | t(2) at n ≥ 500; every shape at n = 50k |
| "C = 1 safe at min(n) ≥ 500" | 27 Stage S cells | t(2)/.99 at n = 500 (.690) |
| AUC_ub/n routing heuristic | 257 t-cells, zero failures | Slivers at AUC .60/.80 |
| "C = 1 solid at AUC ≤ .90" (in the `fiducial_band` docstring) | t-family | Slivers (.52–.61) and the jump (.11–.25) |
| M3 remap α′ ≈ .6 | 8 cells | 900/100 needs .5 |

The protections that survived are the ones that are **rank-measurable** or
**exact**: the CP allowance, the domination and regional-cap properties of
the floor, and M3. The program's own methodological conclusion
(`next_method_ideas.md` §10: no global remaps learned from a finite library)
follows directly from this record. A library can falsify a rule. It cannot
certify one, because a feature at the 1/n scale can evade every smooth cell.

### 7.3 Data-adaptive relaxation co-moves with error

Several different mechanisms turned out to be the same one. When a procedure
uses the data to decide how much protection to spend, the datasets that look
most favourable are disproportionately the ones where the band is wrong:

- M1 plug-in calibration: the plug-in's dispersion shrinks when the realized deviation is large.
- M4b bracket: the worst case over a set selects its roughest member.
- Within a Stage F cell (fixed truth), failing replicates have AUC-hat 0.46 SD higher. The more
  separable-looking realizations are the dangerous ones.
- The exploration's interior eligibility: a high-AUC binormal clears the window only when its saturated
  run is anomalously short (.400 vs .983 coverage).
- Unseen slivers: the data law equals the no-sliver law on that event (theory §12.3).

The one clean exception is round 4's shape-functional rules. They failed
for lack of signal, not co-movement (|ρ| ≤ .25). Practical implication: any
router, gate, or adaptive relaxation should be judged by its **selected**
(conditional) failure rate, which theory §12.3 already makes formal. It should
by default be expected to relax protection on the wrong datasets.

### 7.4 Scarce positives are consistently the hard direction

Four independent studies point the same way:

- Round 4: M3's level map is bound by 900/100 (few positives): α′ = .5, against .75 for 100/900.
- Stage S: binormal_90 at 4,500×500 has the smallest C\* (1.69, against 2.2–2.7). The
  directional contrast is resolved at 95%.
- Stage F: C = 1 at AUC ≥ .95, negative-majority, mean .870 (the worst cell in the study is 1676×391 at .570),
  against positive-majority .977. That is only 4 vs 6 cells.
- Exploration: the right floor's reach is set by n1. With binormal .95 at total n = 100,000, the window is
  eligible .926 of the time at 10k/90k and .298 at 90k/10k.

**[Inferred]** The right-end resolution is 1/n1. With few positives the
saturated run is long, and the upper-FPR corner, where heavy-tailed truths
approach 1 slowly, is poorly resolved. This is also the most common regime in
practice (rare positives). Yet it has been swept in n at only a handful of
points, and never for the hybrid at production M.

### 7.5 Central-α over-coverage is universal and has not been touched

At nominal .50: C = 1 covers .58–.86, the hybrid .78–.90, M3 .92–1.00, and
M3 variants .95–1.00. The only things that ever moved it were level remaps (C > 1, and
M3 at α′), which are not honest. The two sources differ. In M3 it is projection:
two rectangles covering CDFs, when only their ROC composition matters. In the fiducial band it is
mostly the interior cloud's finite-n conservatism (§7.1); trimming against the
ends adds a little. The floor makes it worse.
"Calibration across α" was an original desideratum. The program has made no
progress on it for honest bands, and the reason is that it is a
construction problem, not a tuning problem.

### 7.6 The Stage F residual is C = 1's finite-n conservatism wearing off

Stage F found that the floored band's pointwise interior miss rate is flat in n (≈ .0005 low +
.0006 high per grid point), while its whole-band miscoverage rises from .008 at n0 ≤ 200 to .032
above 4,000. It read this as multiplicity. That is correct, but there is a simpler
description: Theorem 7 says interior coverage tends to 1 − α. Stage S measured
C = 1 at .951–.960 at n = 50,000 on binormal_95, kink_80, and t2_95. By
domination, the floored band covers at least that on those shapes.
**[Inferred]** So on those three regular shapes, the floored band's large-n limit
is already known to be at least nominal. The unanswered large-n
question is about the *wedge* shapes, where C = 1's interior may behave differently. Stage F's
speculation that "the lever is the trim level" was then tested by the
exploration, which found no room for C > 1 on the interior.

### 7.7 The Monte Carlo budget is entangled with statistical protection

Several quantities meant to be statistical depend on M:

- Stage F's left cut, `⌈log(M+1)⌉`, gave 8–10 grid points at every α.
- j saturates when M is small, destroying α-resolution.
- The exploration's parent band changed width with M.
- Stage S's C\* near the ladder boundary (j = 2) produced a spurious C\* = 0.084.

The production exact rule partly separates these: it keys the left cut to the realized
depth. The general lesson is to report `ℓ` and `j` alongside any comparison,
and to compare arms at matched production M.

---

## 8. Corrections to the earlier reports

Claims that should not be carried forward, or that need their scope stated:

| Original claim | Status |
|---|---|
| R2: C = 2 is "structural", a Šidák correction per sample | **Falsified.** C\* → 1 (R3 ladder; Stage S at α = .05) |
| R2: "No validity failure anywhere; the low-FPR corner is handled natively" | **Library-limited.** Convex hooks, slivers, and jumps fail |
| R2/R3: C = 1 "never measured below .967" | **False** after Stage S |
| R3: pin U(0) = 0, "R(0) = 0 for every continuous DGP" | **False.** Support gaps give R(0) > 0 (corrected at the time) |
| R3: "M3's geometry is as efficient as the fiducial cloud's" | True only at matched *realized* coverage on smooth cells, using the truth. Not a statement about any usable M3 |
| R3/R4: the calibration axis is "roughness" and "remains unidentified" | Still unidentified. End geometry was tested as the explanation and accounts for only a small part (§7.1). "Roughness" must mean a cloud-versus-truth discrepancy along the whole curve, not an interior feature of the truth (the trapezoid is mid-library) |
| R4: fixed C = 2 is "far closer to the ceiling than any data-driven arm" | True on those cells at those n; C = 2 was later shown invalid at large n and on t(2) |
| Stage S: C = 1 is safe for min(n0, n1) ≥ 500 | **False.** t(2)/.99 covers .690 at n = 500 |
| Follow-up: the unsafe set is an (AUC, n) wedge; the AUC_ub routing heuristic | Describes the t-family only. Not distribution-free (slivers fail at AUC .6–.8) |
| Follow-up: the m = n0·t_q window as a routing coordinate | Not a replicate-level predictor (Stage F: −0.17 SD, positive in 35/94 cells) |
| Stage F: "floor residual is multiplicity; the lever is the trim level" | Multiplicity is a fair description (§7.6). The trim-level lever was later tested and found no room for C > 1 |
| Exploration: "Stage S's surplus was mostly tail surplus"; "windowing costs coverage rather than harvesting surplus" | Not supported at n = 500. A direct test (§7.1) found the surplus mostly in the interior. At n = 5,000, windowed C = 1 recovered up to 8% width and landed near nominal. What the data support there is **no room for C > 1**, on selected (non-representative) eligible datasets, with binormal .95 unchanged at [1.5, 2] |
| Exploration: "M3's own boundary is the constrained optimum" | Within one 27-member family, 4 training shapes, and a zero-tolerance tail constraint. The class-split veto was a margin of 9e−4 |
| Exploration: "cell loss is catastrophic"; "test inversion's width is entirely a knot-resolution question" | Both measured only at fixed, coarse, non-adaptive resolution (≤ 36 cells, 3 knots). They say the naive versions fail, not that the routes do |
| Exploration: "the hybrid is never wider than M3" | Measured in 162 small-n cells. Not a theorem: the hull can exceed M3 in principle |
| `fiducial_band` docstring: "coverage is solid at AUC ≤ ~.90 at every tested n" | True of the t-family, contradicted by the sliver and jump constructions |

The earlier reports also contain several verdicts ("PASS", "the pick is…",
"recommend shipping…") issued against gates the reports set themselves. Only
two production decisions were actually taken: the C = 1 default (2026-08-30)
and the optional, default-off `m3_floor` (2026-09-07). Everything else is
still open.

---

## 9. Gaps: what has not been measured

In rough order of how much each could change the picture:

1. **No fiducial variant in the paper's main suite.** None of C = 1, the hybrid, or
   M3 has been run through `scripts/run_simulation.py` on the LHS DGP design.
   Every current claim comes from bespoke or adversarial cells. The
   representative-population behaviour, and the WH/KS comparison the paper
   needs, are unknown.
2. **The hybrid at production M on non-adversarial cells, at n = 1,000–10,000**,
   especially negative-majority (§7.4). The exploration's proposed n = 5,000
   production-budget check (about 1.5 h) has not been run.
3. **Interior features at moderate n.** Interior jumps and translated slivers were tested
   only at n ≤ 50 (where the floor covers nearly everything), plus one
   partially interior sliver in the interior track. The floor is a tails-only
   argument, and theory says interior unseen mass can defeat it.
4. **Full-bracket completion** (theory §3.1, §13.2 #1), the construction aimed
   at the mechanism's source, has never been run.
5. **Whole-band theory for the hybrid.** The exterior term has no bound, only
   measurements on non-representative designs.
6. **α other than .05 and .5** has barely been looked at since round 2. Stage S's
   α = .1/.2 columns are on disk and unanalyzed.
7. **Wedge shapes above n = 12,000.** Not measured.
8. **Ties with the floor.** One Q = 20 regression cell.
9. **Balance metrics** as the program defined them: intervalwise miss
   rates by FPR region, and pointwise miss distributions for M3. These were reported
   only piecemeal.
10. **What drives the interior's finite-n conservatism and its shape spread** (§7.1). The end-geometry explanation was tested and mostly failed.
11. **Class-split M3 follow-up** with a stated tolerance and more training shapes.
12. **Adaptive refinement** for likelihood and test inversion. Only fixed coarse
    resolutions were tried.

---

## 10. State of the research program

**Where it has converged.** The diagnosis is solid. The fiducial band's
failures have one mechanism, which is derived, located pointwise, and
confirmed prospectively on constructions built to exploit it. The repair
targets that mechanism with exact local properties. Level calibration has
been explored thoroughly enough that its failure is understood, not just
observed. M3's cost is quantified, and its source (projection and level accounting) is identified.
The infrastructure is strong: exact rank-space simulation, a shared-cloud
ladder, paired and replayable seeds, lossless records, and a parity-gated
Rust path. It makes paired comparisons of new constructions cheap.

**Where it has not.** Three original desiderata remain unmet by any
honest band: calibration across α, width near WH under binormal truth, and
a finite-sample guarantee narrower than M3. The hybrid is an empirical
compromise. On every stress test it is clearly better than C = 1, and it is
clearly narrower than M3 at moderate and large n. But its honesty rests
on measurements, and its α = .5 behaviour is worse than C = 1's.

**The shape of the effort.** Roughly: one round established the
construction, four rounds and two staged studies pursued level
calibration and located failures, one study built and tested a repair, and
one screen sampled five further directions. The level-calibration line consumed the
most effort and returned mainly negatives. Its lasting outputs are the
confirmation of Theorem 7 and the discovery of the corner failures that came
out of trying to calibrate. The later direction is construction-level: the floor, brackets,
inversion. That is where the remaining problems (§7.5, §9) point.

**The decisions ahead are the user's.** The evidence bears on them as follows:

- *What the paper's method roster is.* The strongest current evidence
  supports M3 as the exact reference and the hybrid as the empirical band,
  with C = 1 as an ablation. The missing input is gap 1: the main suite has never run any of them.
- *Whether to pursue a narrower exact band.* The inversion screens say exactness is easy and
  width is hard. The negatives so far come from coarse, non-adaptive versions, so
  they neither justify nor rule out a serious attempt.
- *Whether to pursue central-α calibration.* §7.5 suggests it needs a construction change
  (addressing projection in M3, or the interior cloud's finite-n conservatism in the fiducial band), not another
  level map.

---

## 11. Archive: what this file replaces, and where the data are

Deleted from the working tree on 2026-09-26. All are recoverable at commit
`8f904e6`, e.g. `git show 8f904e6:stats/hybrid_floor_report.md`, or
`git checkout 8f904e6 -- stats/experiments/`.

| Former file | Content | Raw data |
|---|---|---|
| `stats/experiments/m2_report.md` | Round 2 (Q1, Q4) | `stats/experiments/res_p*.json` (in git) |
| `stats/experiments/m3m4_report.md` | Round 3 (Q4, Q5) | `res_m3_*`, `res_m4_*`, `res_cstar_*` (in git) |
| `stats/experiments/r4_report.md` | Round 4 (Q4–Q6) | `res_r4_*` (in git) |
| `stats/experiments/*.py`, `res_*.json`, `log_*.txt` | Round 1–4 harnesses and results; the Sep 5–6 theory-check, ideation, and rank-likelihood verification scripts | (in git) |
| `stats/c_calibration_spec.md` | Auto-map spec, Stage S amendment, follow-up run plan | — |
| `stats/c_calibration_screening_report_stage_s.md` | Stage S | `data/results/c_calibration_20260829/` |
| `stats/c_calibration_followup_report.md` | Boundary follow-up | `data/results/c_calibration_followup_20260830/` |
| `stats/hybrid_floor_spec.md`, `stats/hybrid_floor_report.md` | Stage F | `data/results/hybrid_floor_20260902/` (manifests and analysis only; ~7 GB of records are not in this checkout) |
| `stats/methods_exploration_spec.md`, `_report.md`, `_validation.md` | Methods exploration | `data/results/methods_exploration_50k_eligibility/`, pilots; the screen's raw records are not in this checkout |

Runners remain in `scripts/c_calibration/` (Stage S, follow-up, Stage F;
see its README) and `scripts/methods_exploration/`.
