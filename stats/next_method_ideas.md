# The Rank-Space Fiducial ROC Band: Working Model and Evidence

*Status (2026-09-01): consolidated after four laptop experiment rounds, the
Stage S calibration screen (2026-08-29; 27 cells,
`data/results/c_calibration_20260829/`, theory-doc §7.2), and the
follow-up boundary study (2026-09-01; 257 student-t cells / 64,625 reps,
`data/results/c_calibration_followup_20260830/`, report
`stats/c_calibration_followup_report.md`). Stage S changed the production
default to **C = 1**, returned STOP on the shape-blind auto-C map, and
relocated the surviving width opportunity to the band's interior. The
follow-up then **enlarged the validity failure from a small-n hole to a
curved (AUC, n) wedge and falsified monotonicity of coverage in n**, which
retires the "safe above min(n₀,n₁) = 500" framing throughout this document
(§1, §3 P-A, §5 items 1 and 6) and promotes a localized M3 floor over the
composite band as the lead fix (§7). This document is a working model of
what the full simulation suite should show, based on what has actually been
measured — with the uncertainties stated. The full suite is the arbiter;
nothing here is a result of that suite yet.*

*Theory companion: `stats/fiducial_band_theory.md` — guarantees, rates,
corner impossibility results, and the analysis of the C-remap.*

*Method implementation: `src/studroc_paper/methods/fiducial_band.py`
(`fiducial_band`, exported from `studroc_paper.methods`; unit tests in
`tests/test_fiducial_band.py`). Evidence: `stats/experiments/` — harnesses
`rank_band_experiments.py`, `m2_experiments.py`, `m3_experiments.py`,
`m4_experiments.py`, and `r4_experiments.py`, ~40 result JSONs, and the
detailed round reports `m2_report.md` (round 2), `m3m4_report.md` (round 3:
the M3/M4 backlog and the C*(n) ladder), and `r4_report.md` (round 4: the
last calibration ideas, the roughness functional, the named-curve exact test,
and two round-3 loose ends). Earlier ideation that this work descends from is
summarized in §7–§9; superseded content has been cut.*

---

## 1. The method

**Reduction.** The ROC is invariant to strictly increasing score transforms.
Under s ↦ 1 − F(s) (F = negative-class CDF), negatives become iid
Uniform(0,1) and positives become iid with CDF exactly R_true. All information
about R_true is in the interleaving ranks, and the exact finite-sample law of
each class's CDF at its own order statistics is Dirichlet(1,…,1) — for every
continuous score distribution, at every n. Consequences: (i) any rank-based
band's coverage depends on the DGP only through (R_true, n₀, n₁, α) —
distribution-parameter invariance is structural; (ii) the sampling experiment
can be simulated exactly for any hypothesized curve.

**Construction** (the recipe validated in `m2_report.md` §7):

1. Sort scores; break ties at random (estimand = the trapezoidal /
   Mann–Whitney ROC); keep the merged label sequence.
2. Draw M fiducial ROC curves: per draw, Dirichlet(1,…,1) spacings for each
   class's CDF at its order statistics; the other class's within-gap elements
   at sorted-uniform fractions of the gap; compose on the grid t_k = k/n₀.
3. Trim by equal-local-levels min-p depth (each draw's minimum, over grid
   points, of its rank from either end of the cloud) at the remapped level
   **α_eff = 1 − (1−α)^C with C = 1** (the identity map — the default since
   2026-08-30; the former C = 2 was refuted by Stage S, which measured it
   at .917–.940 realized coverage at α=.05 on heavy-tail and large-n
   cells). Band = pointwise [j-th smallest,
   j-th largest] of the draws, j = the α_eff-quantile of the depths.
4. Two binomial (Clopper–Pearson-form) corner allowances at the band's own
   local level ℓ = j/(M+1): upper edge ∪ CP upper bound (essential — the
   upper edge must equal 1 wherever empirical TPR = 1); lower edge = 0
   wherever empirical TPR = 0 (free). The level is data-selected, so these
   are pure widenings at the corner-forced scale, not standalone exact
   devices (theory doc §8).

Tuning inputs: M (a Monte Carlo budget, self-diagnosing — the method warns
when the realized trim depth j < 3; rule of thumb M ≳ 5/ℓ(K,α), ≈10,000 at
n₀ = 5,000) and the exponent C = 1 (the identity map; values above 1 are a
manual option, anti-conservative on heavy-tailed shapes — see §5 item 2).
There are no variance floors, gates, jurisdiction constants, or ε
regularizers. **Known validity boundary (revised 2026-09-01 — it is not a
sample-size threshold).** Heavy-tailed high-AUC shapes under-cover at C = 1
inside a *curved wedge in (AUC, n)* that widens with AUC and reaches past
n = 6,000: worst measured .645 (t(2)/.99, n = 250) and .690 at n = 500, and
at AUC ≥ .975 no tested n up to 6,656 is safe. Coverage is **not monotone
in n** — at fixed shape t(4.69)/.986 it falls .993 → .823 from n = 150 to
1,200 — so the former "below min(n₀,n₁) ≈ 500" framing is wrong in kind,
not just in its constant. The measured routing rule and the localized M3
floor that repairs the failures are in
`stats/c_calibration_followup_report.md` §5 and §7; the exact M3 band
remains the indicated fallback inside the wedge.

---

## 2. Evidence base — what was actually measured

Four rounds, ~20 cells, 60–400 replicates per cell (coverage SE ≈ 1.1pp at
400 reps for the 95% level; ≈ 2.2pp at 120 reps; several pp at the 50%
level). Cells span n per class 25–5,000; AUC 0.55–0.99; binormal, bimodal-
negative, t(2), and kinked truths; 9:1 class imbalance in both directions;
score quantization to 20 and 100 levels. By rank invariance, cells are curve
*shapes*; parameter sweeps within a family (e.g. t's df at fixed shape) are
provably redundant. The third round (2026-08-22, `m3m4_report.md`) tested the
§7 backlog (M3 guarantee layer, M4b bracketed calibration) and extended the
C*(n) ladder to n = 10,000 and 20,000 at central α; the fourth (2026-08-23,
`r4_report.md`) closed the remaining calibration ideas, searched for a
rank-computable roughness functional, delivered the exact named-curve test,
and settled two round-3 loose ends. Deltas from both are folded into §3, §5,
§6, §7, and §8 below.

Development history matters for reading the numbers:

- The **CP upper allowance was designed after observing** the raw fiducial
  band fail on the bimodal cell (100% miss rate at depth ~1e-4). The bimodal
  cell's post-fix coverage is therefore in-sample. The six P2 cells were run
  after the fix without further changes.
- The **exponent C = 2 was fitted on 4 cells** (binormal .75/.95, bimodal,
  t(2), all n=500) and **transferred without refitting** to 10 held-out
  cells. But all cells were chosen by the same designers; the LHS parameter
  sweep of the full suite will visit shapes no one picked by hand.
- Everything else (fiducial construction, trim rule, M rule) was fixed before
  the cells that test it, or is measured on held-out cells.

Headline measurements (details and tables in `m2_report.md`):

| Quantity | Measured |
|---|---|
| Coverage @ α=.05, C=2 map, adequate M | 0.942–0.980 over 14 cells |
| Coverage @ α=.05, identity map (C=1) | 0.967–0.995 (more conservative) |
| Coverage @ α=.20 / .50, C=2 map (n ≤ 900, excl. n=25) | 0.762–0.858 / 0.435–0.573 |
| Coverage @ α=.20 / .50, identity map | 0.880–0.960 / 0.655–0.858 |
| n-trend @ α=.05 (n = 25→5000) | flat within noise (.967–.993) |
| Miss direction (v_low : v_high) | ≈ balanced (e.g. .017/.017 at n=5000) |
| Miss depth @ α=.05 | p95 = 0 in every cell; max ≈ 0.02–0.055 |
| Area vs KS | 21–66% of KS across all cells |
| Area vs oracle rank-band ceiling | 1.05–1.35×, except AUC .99 small-n: 1.9–3.2× |
| Area vs WH where WH is valid (binormal) | ≈ 1.7–2.1× (WH invalid on 3 shapes: 0.000 coverage) |
| Ties (random break, Q=20/100) | indistinguishable from continuous at every α |
| Runtime | ~11 s/band at n₀=5,000, M=10,000, one laptop core |

Calibration reference points also established: an **oracle arm** (calibrate
against the true curve) is exact at every α and shape and provides the width
ceiling for any rank-based band (~1.6× WH area under binormal truth); the
harness reproduces the known baselines (KS 100% everywhere; WH ≈ .95–.97 on
binormal cells and 0.000 on bimodal, t(2), and kinked truths).

---

## 3. Working model: what the full simulation suite should show

Predictions, each with a confidence tag and what would falsify it. "Suite"
means the existing 7-DGP × 6-n × LHS framework at α ∈ {0.05, 0.5}.

**P-A. Coverage at α=.05 lands in ~0.94–0.99 in every (DGP, n) stratum
*outside the (AUC, n) wedge*; inside it, expect failures at any n up to at
least 6,000.** *(Confidence: high on the wedge's existence, moderate on its
edges.)* **Revised 2026-09-01 — the previous form of this prediction ("flat
in n, safe above min class size 500") is falsified**, not merely
qualified: 257 student-t cells / 64,625 reps locate failures from n = 102 to
n = 6,656, and coverage falls with n at high AUC. Basis for the revision:
`stats/c_calibration_followup_report.md` §2, §4. The suite's high-AUC
student_t strata should be *expected* below nominal at every n the suite
visits; its AUC ≤ .90 strata should be comfortably above it (65 cells, one
failure). A stratum below ~0.93 at AUC ≤ .88 would contradict the current
model; so would a *safe* stratum at AUC ≥ .985 at any n, which no cell has
yet produced.

**P-B. With the shipped default C=1, coverage at α=.5 is conservative and
dispersed: expect 0.65–0.86 (valid everywhere, centred nowhere). Under the
retired C=2 it would be centred but dispersed: stratum values
roughly 0.40–0.60, mean near 0.50.** *(Confidence: moderate.)* The measured
per-cell optimal exponent ranges 1.6–3.1, and dcoverage/dα_eff ≈ 1 at α=.5,
so the ±15pp shape spread seen in the cells should reappear as ±10–15pp
stratum spread. This is the model's honest claim: *centred at every α, not
calibrated at every α*. Falsifier: mean far from 0.50, or spread much larger
than the cells showed. (If the suite is run with the identity map instead,
expect 0.65–0.86 at α=.5 — valid, conservative.) Round-3 update: the C*(n)
ladder now reaches n = 20,000 (§5 item 2) and fixed C ≈ 2 measurably over-trims at
central α for n ≥ 10⁴ (coverage .41 at n = 10,000 and .38 at n = 20,000
against nominal .50, at C = 2.2) — expect the suite's largest-n strata to land
at the bottom edge of, or just below, the 0.40–0.60 range.

**P-C. Directional and spatial miss balance.** *(Confidence: moderate-high.)*
v_low ≈ v_high within a factor ~2–3 in most strata (vs ~10:1 for the
envelope), median miss location spread across FPR rather than pinned at the
corner. Basis: consistent across all cells including AUC .99 and the kink.

**P-D. Misses are small.** *(Confidence: high at α=.05.)* p95 miss depth ≈ 0;
max depth a few pp of TPR. The catastrophic tail (>5pp misses at high AUC)
that afflicted the envelope pre-Beta-floor should be absent — the fiducial
F-side natively carries the extreme-order-statistic channel that required
the explicit Beta floor before. Falsifier: any stratum with >5pp misses at
a rate above ~1%.

**P-E. Width sits between WH and KS: roughly 1/3 to 2/3 of KS area,
approaching ~1/3 at large n; ~2× WH on binormal-compatible strata.**
*(Confidence: high, ±.)* Exception to model explicitly: the steep-corner
small-n strata (AUC ≳ .97, n ≲ 300) run 2–3× the oracle ceiling — the
fiducial cloud is over-dispersed exactly where the curve is steepest. Expect
the suite's high-AUC small-n cells to show the least width advantage.

**P-F. The suite's tie-free DGPs make the tie convention moot there**, but
the paper must state it: random tie-breaking, trapezoidal estimand. Basis:
quantization red-team passed exactly (random break is estimand-exact, even
spreading slightly conservative, class-ordered breaking invalid — coverage
0.000, correctly, because it changes the estimand to one no band can track).

**P-G. Integration risks (not statistical).** The eval framework's grid and
empirical-ROC conventions must match the band's native staircase-upper
convention on t_k = k/n₀; mismatches would show up as spurious sub-pp
"violations" at plateau edges. Runtime at n=10,000 × 4,000-LHS scale is
nontrivial (~tens of seconds per band × suite size) and may need the GPU
path or a reduced-M sensitivity check.

---

## 4. Comparison target

For reference, the incumbent (`envelope_wilson` + Beta floor; 2.25M-eval
suite, so not directly comparable to laptop cells): pooled 0.950 at α=.05
with drift 1.000→0.830 across n at prevalence 50% pre-floor and 0.95–0.99 on
problem strata with the floor; 0.85 actual at the 50% level; ~10:1 downward
miss imbalance; three stacked mechanisms (studentized envelope, gated Wilson
rectangles, Beta floor) with J=25, ε, K_eff-Šidák constants. The fiducial
band's laptop profile dominates this on every axis except possibly width in
some strata (envelope pooled area 0.397 vs KS 0.469; the fiducial cells ran
21–66% of KS — overlapping ranges, different cells; the suite will settle
it). This comparison is the bar the suite run has to confirm, not a
conclusion.

---

## 5. Known weaknesses and open uncertainties (ranked)

1. **The high-AUC validity wedge (revised 2026-09-01 — still the most
   serious finding, and larger than it looked).** What Stage S saw as a
   *small-n* heavy-tail hole is one edge of a curved unsafe region in
   (AUC, n). Measured over 257 student-t cells / 64,625 reps
   (`stats/c_calibration_followup_report.md`): failures span n = 102 to
   n = 6,656, the n-range of failure widens monotonically with AUC, and at
   AUC ≥ .975 **no tested n is safe**. Worst cells: t(2)/.99 covers .645 at
   n = 250 and .690 at n = 500; t(6.62)/.988 covers .852 at n = 6,656.
   **Coverage is not monotone in n** — at fixed shape t(4.69)/.986 it runs
   .993, .947, .903, .823, .847 across n = 150…2,000 — which falsifies the
   sign constraint the offline calibration surface was built on and means
   no `n ≥ threshold` rule can express the boundary. Mechanism (theory doc
   §7.2a) is unchanged in kind — *unseen tail mass*, misses lower-edge —
   but replay locates them mostly at the **upper** FPR end (peak at
   1−FPR ≈ .002–.04) plus a small left-corner cluster, not symmetrically at
   both corners. A partial first-principles coordinate exists: failures
   concentrate in a window of `m = n₀·t₅₀` (negatives above the median
   positive), which explains the non-monotonicity — n carries a shape
   *through* the window — but the window's upper edge grows with AUC, so m
   compresses the boundary without linearizing it. Two live fixes, in
   preference order: the **localized M3 floor** (§7, +6.4% width vs +28–46%
   for routing, and provably never worse than C = 1) and the conservative
   (AUC, n) **routing rule** (report §5; zero failures over all 257 cells).
   Both are validated in-sample only; spec follow-up item 5 still gates
   guidance.
2. **No shape-blind level map is worth shipping — the Stage S screen
   returned its pre-registered STOP.** Per-shape C*(.05) at n = 500 spans
   1.17 (t(2), 2,000 reps) to 3.0; the library lower envelope minus one SE
   is 0.97, i.e. at the n where the 9.5% mean oracle gain lives there is
   *no* safe C > 1. The former default C = 2 under-covers at α=.05 on t(2)
   at every n (.918–.940) and on binormal .95 at n = 50,000 (.917) —
   production default is now **C = 1** (measured ≥ .950 on every min-500+
   Stage S cell; margin ~0 at n = 50,000, consistent with Theorem 7's
   approach-from-above — but note the 2026-09-01 follow-up found min-500+
   cells that fail, so that Stage S summary describes its own library, not
   a safety guarantee; see item 1). The round-3 taper story survives only per-shape:
   the *envelope* is non-monotone in n (t(2): 0.08 → 1.17 → 1.49 → 1.07),
   so the tapered-C(n) family is withdrawn; re-evaluating the stored
   profiles, the largest safe shape-blind C is ≈1.5 at n = 5,000 (~4%
   area) and 1.0 at n = 50,000 and at every minority-500 imbalance cell.
3. **Shape dependence of calibration at central α survives any level
   remap — and after round 4 there is no data-driven candidate fix left**
   (±13–19pp spread at α=.2/.5; bias removed, spread remains). A level-only
   correction cannot fix this, and every per-dataset route tested is dead by
   the same roughness mechanism: plug-in calibration (1.3–1.7× conservative
   in j at 80× compute, round 2), worst-case bracketing (9–37×, round 3),
   fiducial-predictive calibration (1.7–2.3×, *worse* than plug-in, round 4),
   and functional-driven level rules (nothing among 32 rank functionals
   survives out-of-sample, round 4 — see §8.4). The offline shape-library
   route was then tried (Stage S) and STOPped by item 2. What remains:
   accept the spread at C = 1 (conservative everywhere at central α), or
   change the construction (composite band / depth functional — §7).
4. **No coverage theorem.** All validity evidence is Monte Carlo. The
   fiducial composition is not automatically a confidence procedure; the
   oracle/test-inversion framing (§7) is the likely proof route, and the
   degenerate-corner allowances are exactly the places where the naive
   "fiducial = confidence" heuristic measurably failed before repair.
5. **Steep-corner width at small n** (2–3× oracle at AUC .99, n=150). Valid
   but loose; the one width regime where the method leaves real money on
   the table. Round 4 removed one candidate repair: intersecting with M3's
   edges restricted to the first grid points (union-bound accounting) never
   binds — at any α₂ carrying a guarantee, M3 is 1.7–4.6× *wider* than the
   fiducial band on k = 1..25, reaching parity only near α₂ ≈ 0.9. The
   corner slack is not reachable through exact-Beta bounds; both §10
   mechanisms of the theory doc remain live and unseparated.
6. **Untested configurations (revised after Stage S):** large n at α=.05
   is now measured (C=1 covers .951–.960 at n = 50,000; the former
   n = 10,000 gap is closed by bracketing); the imbalance arm ran at
   minority 500 in both directions × two shapes (C=1 fine at .950–.973;
   a real directional C* effect on binormal .90 — majority-negative
   4500×500 cuts C* to 1.69 vs 2.2–2.7 elsewhere — confirming round 4's
   D2 concern). The 2026-09-01 follow-up closed the largest gaps: the
   failure boundary is mapped over AUC ∈ [.55, .99] × n ∈ [100, 7700] in
   the student-t family, and heavy-tail shapes above AUC .95 — flagged
   here as the likely worst case — are indeed where the failures
   concentrate. Still untested: **n above ~8,000 at AUC ≥ .985**, where no
   safe cell exists yet and the wedge may not close; the wedge outside the
   student-t family (the cross-family spot checks cover only the corners);
   imbalance with min(n₀,n₁) > 500; the full LHS shape sweep.
7. **Compute scales with M·K.** ~11 s/band at n₀=5,000 single-core; the
   suite multiplies this by ~10⁴ bands. GPU batching of draws (already
   chunked in the implementation) or per-n M tuning will matter.
8. **Estimand under ties** must be declared (trapezoidal, random break).
   Deterministic even-spreading is a valid conservative alternative;
   class-ordered tie-breaking is invalid and the implementation refuses it.
9. **The t = 0 width finding (round 3, `m3m4_report.md` §3) — measured
   correctly, but the recommended pin is NOT valid distribution-free.**
   The recipe's CP allowance at k = 0 does make U(0) run 0.4–0.99, worth
   0.0–8.8% of area on the tested cells. But the premise "R(0) = 0 for
   every continuous DGP" is false: continuous scores with bounded negative
   support and positive mass above it have R(0) > 0, and the corner
   impossibility argument applies at t = 0 (relocating the top c/n₀ of
   negative mass is a TV-c change that sets R(0) = R(c/n₀)), so any valid
   rank-based band must keep U(0) at exactly the scale the allowance
   produces — see `fiducial_band_theory.md` Corollary 9.3. The fiducial
   cloud alone has R̃(0) = 0 identically, so the k = 0 allowance is
   load-bearing on separated-support truths (without it: certain miss at
   t = 0 there). Production `fiducial_band` applies the allowance at k = 0
   (checked) and keeps it. The pin is admissible only under an explicit,
   user-asserted support-overlap assumption (true for all 7 suite DGPs) —
   a possible documented option, not a default, and not currently
   implemented. The production M3 (`m3_band_rs`) handles this correctly:
   its default U(0) is the composition's own exact Beta bound (theory doc
   Prop. 12), with the pin available only via `assume_r0_zero=True`; the
   round-3 experimental harness still pins by convention (harmless on the
   suite DGPs, all of which satisfy R(0) = 0).

---

## 6. Falsified approaches (dead ends — do not revisit without new ideas)

- **Plug-in Monte Carlo calibration (was "M1"/SIMCAL), as band
  construction.** 0.28–0.43 coverage at α=.05 with either raw-polyline or
  Hazen-smoothed plug-in curves. Mechanism: the plug-in curve co-moves with
  the data draw, shrinking simulated dispersion exactly when the realized
  deviation is large (the Wald-interval pathology, in curve space). Not a
  smoothing problem. The same mechanism, attenuated, biases even scalar
  plug-in calibration of the trim depth.
- **Full mirrored CP lower allowance:** +15% area at AUC .99 for zero
  coverage change (it drags the lower edge to ℓ^(1/n₁) across the plateau).
  Only the degenerate part (L=0 where k̂=0) is worth having.
- **Thinned-grid trimming as an M substitute:** safe (nothing leaks,
  j* +10–15%, area −1%) but far too weak; saturation persists.
- **Class-ordered tie-breaking:** produces an estimand with vertical cliffs
  at deterministic FPRs that no band can cover (0.000 even against its own
  staircase estimand).
- **Bracketed worst-case calibration over an M3-50% confidence set ("M4b").**
  Falsified in round 3 (`m3m4_report.md` §5). The worst case over the set's
  raw members is set by its roughest member (the M3 lower edge), which pins
  the calibrated trim depth at the floor: 9–37× more conservative than the
  oracle, α-independent (one identical band at α = .5/.2/.05), and far worse
  than the plug-in it was meant to replace. Worst-casing does not dodge the
  plug-in roughness pathology — it *selects for* it. Smoothing the members
  recovers exactly plug-in performance (1.25–1.79× vs plug-in's 1.27×) and no
  better, at 4.5× the compute plus a new tuning constant (the window).
- **Early slope as the bracketing axis for calibration.** The exact
  calibration ceiling ae* is flat along a five-fold early-slope ladder
  (binormal .70→.99: 4.5pp at α=.5, ≈1 MC SE) while two off-family shapes
  move it 9–13pp in directions inconsistent with any early-slope ordering
  (t(2) .95 sits below the entire binormal ladder despite a mid-ladder early
  slope; bimodal .90 moves opposite in sign). The axis that matters is
  roughness-like, not slope.
- **Fiducial-predictive trim calibration.** Falsified in round 4
  (`r4_report.md` §1) — the last live data-driven calibration idea.
  Calibrating the trim against candidate curves drawn from the fiducial
  cloud (coverage averaged over the predictive law) is 1.7–2.3× conservative
  in depth, *worse* than the plug-in it was meant to replace (1.27×), at
  comparable inner compute. Mechanism measured: fiducial candidates are
  rougher than Hazen plug-in curves — ≥5% of candidates fall outside their
  own inner cloud entirely (depth 0), while the smooth truth sits ≈2.9×
  deeper than a draw — so integrating over the predictive law *amplifies*
  the roughness pathology rather than averaging it out. Smoothing the
  candidates recovers exactly plug-in performance and no better, with a
  window constant. Coverage gains over C=1 at central α: none.
- **Steep-corner repair by intersecting with corner-restricted M3.** Never
  binds at any guarantee-carrying level; see §5 item 5.
- From the earlier envelope-era experiments (see
  `project_evaluation_report.md`): logit-space construction; Wilson-gate
  redesigns; the variance-model band (noisy variance × supremum).

---

## 7. Backlog: ideas retained but not currently needed

- **The localized M3 floor — the lead idea after the 2026-09-01 follow-up,
  and it displaces the composite band below.** Replay of failing cells
  (`c_calibration_followup_report.md` §7) shows the C = 1 band's misses are
  localized in FPR, and that **M3 covers at 100% of the miss points**.
  Taking the pointwise union with M3 on `FPR ∈ [0, .005] ∪ [.5, 1]` and
  keeping C = 1 elsewhere lifts coverage from .720–.940 to .955–.990 on
  five failing cells at **+6.4% mean width, against +28–46% for routing the
  whole curve to M3**. Two structural properties make this stronger than
  the composite band: the upper region is nearly free despite spanning half
  the curve (both bands are compressed against TPR = 1 there), and the
  hybrid **provably cannot do worse than C = 1** — the union is pointwise
  wider and the running-max closure preserves that ordering, so its
  coverage dominates identically rather than on average. That removes the
  usual composite-band risk and makes unconditional application
  defensible rather than routing-gated. Unlike the corner treatment below
  it is also theorem-capable in principle, since M3 carries Prop. 12.
  What it needs before shipping: the region was chosen on the five cells
  that score it (100–200 reps each, student-t only); `tau_lo = .005` should
  be re-expressed in grid points (at n = 130 it spans 0.65 of one); and the
  width cost where C = 1 was already valid is unpriced. The validation run
  is specified in the report §10 item 2 (~2–3 CPU-hours).
- **The composite band — corner-patched interior trim over a declared
  finite range (after Stage S; now the width play rather than the validity
  play, since the M3 floor above addresses validity more cheaply and with a
  domination guarantee). Derisk ran 2026-09-01: parity held on all 9 core
  cells and `b0.02-0.95_C2.5` PASSes every cell on the declared range
  n ≥ 500 at −6.8% pooled width, with the n = 20,000 sentinels showing the
  saving invert as Theorem 7 requires. It is a genuine finite-range
  candidate; the width surplus it harvests is real and distinct from
  anything the M3 floor does.**
  Stage S's miss-location analysis found that everything blocking a deeper
  trim is corner-local: the small-n heavy-tail failure (§5 item 1) misses
  at FPR ≤ .02 / ≥ .90, the shape spread that pinned the level-map
  envelope at C ≈ 1 collapses on the interior, and restricting to
  FPR ∈ (.05, .90) the worst-miss rate under C = 2 is ≤ 3.5% at *every*
  screen cell (nominal 5%) — including 1.0% at the t(2) n = 100 cell
  where the full-curve band covers .802, and flat at every measured n
  from 100 to 50,000. **That flatness is finite-range, not asymptotic:**
  Theorem 7's erosion applies to a fixed interior C > 1 exactly as it did
  to the full-curve C = 2 (interior coverage → (1−α)^C: .926/.903/.880 at
  C = 1.5/2/2.5, α = .05, at ~1pp/decade of n), so no fixed-C composite
  can be an unrestricted method — the shipping form must taper
  C_int(n) → 1 or clamp to 1 above a declared range. Proposed
  construction: the band is widened at the two corner regions to the
  untrimmed cloud envelope + allowances (an *empirical* widening — not an
  exact bound; see (iii)) and min-p-trimmed at C > 1 on the interior,
  calibrated *as one object*. A crude union bound (untrimmed corners +
  C=2 interior) already clears .95 at every screen cell (worst ≈ .953).
  Why this is the right shape of fix: the corners are where the width
  price is provably unavoidable for any rank-based band (theory doc
  §8–9) and where 100 samples of a heavy tail genuinely know nothing —
  honest width there is correct, not conservative; the interior is where
  the measured 7–8% (C=2 vs C=1) to 9.5% (per-shape oracle) area surplus
  actually lives. Known unknowns before it can ship: (i) the stored
  profiles log only worst-miss location, so interior rates are optimistic
  by an estimated ~0.1–0.3pp — the follow-up derisk
  (`scripts/c_calibration/followup_runs.py` item 3, revised 2026-08-31)
  removes this by building the actual stitched band per rep, with a
  predeclared noninferiority rule, paired width inference, and 20k
  sentinels for the erosion direction; (ii) corner/interior boundary
  placement is a real design knob (corner misses smear inward
  continuously — max low-corner miss at FPR .049 even at n = 50,000 — so
  boundary and interior-C must be chosen jointly, and a graded allowance
  may beat a hard mask); (iii) the composite's guarantee statement is
  library-relative (Monte Carlo), unless the corner treatment is made
  exact (an M3-style Beta bound on the corner regions only — cheap
  there — would make the corner piece theorem-carrying and confine the
  empirical claim to the interior); (iv) the taper-vs-range decision for
  C_int above the calibrated range. Companion decisions settled by the
  same evidence: default C = 1 shipped; the shape-blind auto-C(n) map is
  dead (Stage S STOP); an M3-below-n≤500 / auto-above hybrid was tested
  and rejected on economics (M3 costs 1.26–1.69× the C=1 band at n ≤ 500
  where C=1 is measured-valid, while the safe shape-blind gain above 500
  is ~2–6% in a mid-n window only); M3 remains the router target for the
  small-n_eff hole once its boundary is located.
- **M3 — composition of two exact one-sample (Berk–Jones/equal-local-level)
  bands. Now a production method** —
  `src/studroc_paper/methods/m3_band_rs.py` (tests
  `tests/test_m3_band_rs.py`), with the coverage theorem stated and proved
  as Prop. 12 of the theory doc. The production version upgrades the
  round-3 harness in three ways: local levels are calibrated *exactly*
  (non-crossing-probability DP in the `fiducial-core` Rust crate, replacing
  the B=100k Monte Carlo + 2-SE shading), the invalid U(0) = 0 pin is
  replaced by the composition's own exact bound (opt-in pin via
  `assume_r0_zero`), and the class-level split is exposed as
  `split_ratio` ((1−α_F) = (1−α)^ρ — a theorem-preserving lever for the
  9:1-imbalance liability below, unswept so far). Cost: one cached
  calibration per (n, level) (0.06s at n=500, ~47s at n=10⁴), ~1.5ms per
  band thereafter. Measured in round 3 (`m3m4_report.md` §1–4:
  Šidák split across the four one-sided components, MC-calibrated local
  levels, endpoint pins). The theorem holds with a large margin — coverage
  1.000 at α=.05 in all 8 cells (0 misses in 3,200 replicate-cells),
  0.978–0.998 at α=.5 — and the band is cheap (one calibration per sample
  size plus two gathers per band; orders of magnitude cheaper than the
  fiducial cloud). It does generalize the old Beta/Wilson floors into one
  object (its upper edge is the CP bound at an FPR-shifted count). Its three
  hoped-for roles, updated:
  - *(i) Guarantee layer:* fails the ~1.5× width criterion against the
    production band — area 1.45–2.19× `fid_rc` (median 1.68×; only t(2)
    passes), worst at steep corners (AUC .99: 1.97–2.19×) — but is narrower
    than KS everywhere (0.37–0.88×), so as a provable band it strictly
    dominates the provable baseline. The whole penalty is level accounting,
    not geometry: at the nominal level whose realized coverage is .95
    (α′ ≈ 0.6–0.8, shape- and n-dependent, drifting down with n), M3's area
    is 0.93–1.05× `fid_cp`. That remap is a measurement of where the slack
    lives, not a method — a fixed α′ forfeits exactly the theorem that is
    M3's reason to exist. Round 4 measured the worst-case-remap ceiling over
    all 14 cells (`r4_report.md` §4): the infimum-over-cells α′ attaining
    ≥.95 coverage is 0.500 — set by the 9:1 imbalance cell, not by any
    shape — and M3 at a fixed α′ = 0.5 has min coverage exactly .950 (±1.1pp
    SE) with area 1.07–1.43× `fid_rc` (mean 1.21×): it clears the ~1.5× bar
    on this library, but with zero margin (one ladder step of safety,
    α′ = 0.4, gives 1.57× and fails), and the required α′ drifts down with n
    (0.80 → 0.60 over n = 150 → 5000, ≈0.13/decade — a fixed 0.5 is
    exhausted near n ≈ 3×10⁴). A finite library cannot establish the
    distribution-free infimum, so this remains a ceiling measurement.
  - *(ii) Outer miss cap:* fiducial ∩ M3(α/10) is free (0.00% width cost)
    but inert — the cap never bound in 10,400 band-level checks — and the
    certificate is weak: it proves miss depth ≤ 0.10–0.90 where the observed
    worst case is 0.01–0.06. "Misses are small" is not made provable at a
    useful constant.
  - *(iii) Bridge to a coverage theorem via domination:* **ruled out
    empirically.** M3(α′) ⊆ fiducial(.05) essentially never holds — at any
    α′ up to .999, on any cell, even after discarding 25 grid points per end
    (M3's lower edge is identically 0 over the first grid points and dips
    under the fiducial floor near the plateau; at any α′ carrying a real
    guarantee M3 is wider over most of the interior). The reverse
    containment — the cap direction — holds comfortably at α′ ≈ 0.10–0.30
    against the production band.
- **M4 — exact rank-test inversion.** The theoretical ideal: H₀: R = R₀ is
  simple in rank space, so an exact confidence set exists by test inversion;
  bands are its projections. Intractable directly; remains relevant as the
  frame for proving what the fiducial band approximates. Its practical
  relaxation — bracketed worst-case calibration over an M3-50% set (M4b) —
  was tested in round 3 and falsified (see §6). Its tractable fragment was
  delivered in round 4 (`r4_report.md` §3): the **exact Monte Carlo test at
  a named curve** (min-p depth of R₀ in a cloud simulated from R₀) is exact
  within 2 SE at α ∈ {.05, .2} across five (shape, n) cells including the
  two where WH has 0.000 coverage; power at n = 500 is .23–.26 at
  |ΔAUC| = .01 and .71–.85 at .02 (halving at n = 150), with a ~2.5× worse
  sup-norm exchange rate against localized early-FPR alternatives than
  global ones, and easier detection of a corner pushed down than up. A clean
  paper deliverable (non-inferiority against a named benchmark curve).
- **Fiducial-predictive trim calibration** — falsified in round 4; moved to
  §6. With it, plug-in, worst-case bracketing, and functional-driven rules
  all dead, the central-α shape spread has no per-dataset candidate left.
  The offline library calibration (`c_calibration_spec.md`) was then run
  (Stage S) and STOPped by its own pre-registered gate — see §5 item 2;
  the composite band above is what replaced it as the live instrument.
  One narrow adaptive idea survives on validity grounds: a
  *conservative-only* data-driven guard (e.g. a tail-heaviness trigger at
  small n that can only lower C / widen the band) is immune to the plug-in
  co-movement pathology in the invalid direction — bias can cost width,
  never coverage. It cannot harvest the interior surplus, only soften the
  small-n routing cliff; unprioritized.
- **Change the depth functional (the one live construction idea for the
  central-α spread).** Every *level*-side fix is now dead; the measured
  mechanism (draws rougher than truth, contrast concentrated in the lower
  depth tail) suggests attacking the depth functional itself. Two
  candidates, both with content control proved (any per-draw trim score
  qualifies — `fiducial_band_theory.md` Lemma 6b): (a) **smoothed-depth
  trimming** — rank each draw by the min-p depth of its *smoothed* version,
  trim by that score, band = envelope of the retained raw draws;
  depth–tube duality is lost and a smoothing scale enters. (b) **ERL
  trimming** (extreme rank length, the standard tie-breaking refinement
  from the global-envelope literature — theory doc §5.1): parameter-free,
  re-weights exactly the deep-tail rank excursions that drive the
  roughness contrast, and as a bonus removes j*-saturation so M can shrink.
  Untried; a 3-cell derisk (C2/C5/C4 at α ∈ {.5,.2,.05}) covering both is
  cheap on the Rust core and would show whether either equalizes the
  truth-vs-draw depth laws. See `fiducial_band_theory.md` §12 open
  problem 3.
- **Exact-test spinoffs (NEW after round 4).** The named-curve test
  (`r4_report.md` §3, theory doc Prop. 11) opens three cheap deliverables:
  (i) non-inferiority testing against a named benchmark curve (exact,
  distribution-free — directly publishable); (ii) an exact
  goodness-of-fit test of the binormal assumption via a split-sample
  fitted null (fit the binormal curve on one half, test exactly on the
  other) — the diagnostic WH users never had, though the split-sample
  power cost needs measuring and the composite-null version needs care;
  (iii) fiducial confidence intervals for scalar summaries (AUC, partial
  AUC, TPR-at-fixed-FPR) by projecting the existing cloud — near-zero
  marginal code, benchmarked against DeLong. None affect the band; all
  reuse its machinery.
- **Oracle band as a published benchmark:** the exact rank-space width
  ceiling is computable for any (R_true, n₀, n₁) and makes a clean yardstick
  figure for the paper regardless of method.

---

## 8. Open theory questions

1. ~~Prove (or refute) that C = 2 is the correct level remap
   asymptotically.~~ Refuted empirically in round 3: C*(n) decays toward 1
   (1.32 ± 0.16 at n = 20,000), so the "one simultaneity budget per
   class-CDF" account (H1) is out and the roughness-mismatch account (H2)
   stands. What remains open: a proof of the H2 mechanism and of the taper
   rate (empirically ~n^{−1/3}; second-order analysis of the min-p
   functional under a rough-vs-smooth contrast). The direct α=.05
   measurement at large n (formerly owned by `c_calibration_spec.md` D3)
   is now done: Stage S measured C*(.05) to n = 50,000 on three shapes;
   the smooth-shape taper reaches ~1 and crosses it (binormal .95:
   3.05 → 2.23 → 1.78 → 0.87), while t(2)'s is non-monotone
   (0.08 → 1.17 → 1.49 → 1.07) — the *envelope* over shapes is therefore
   not a taper at all, and H2's proof target should be per-shape, with the
   small-n tail channel (theory doc §7.2a) treated as a separate mechanism
   outside the roughness account. Round 4
   removed one candidate proxy for that analysis: the effective-looks ratio
   is *not* a rank-path crossing count even in oracle form (+0.69 with C*
   across shapes at fixed n but −0.31 across all 14 cells — wrong sign
   along n — and the plug-in version flips sign against the oracle
   version); tail-excursion structure is what the second-order analysis
   must characterize instead.
2. A coverage theorem for the fiducial composition + degenerate-corner
   allowances, plausibly via the Dirichlet process / Bayesian bootstrap
   literature for a single CDF, extended to the two-sample composition. The
   domination-by-M3 route was ruled out empirically in round 3 (no M3 band
   carrying a non-trivial guarantee fits inside the fiducial band, interior
   or otherwise); the exchangeability/conformal embedding route is untouched.
   Stage S sharpened the target and the 2026-09-01 follow-up sharpened it
   again: no theorem can hold for the current construction inside the
   (AUC, n) wedge (the truth measurably exits the cloud's support — theory
   doc §7.2a), and since that region is *not* a small-n half-space, "the
   current band above some n" is not a provable object either. The
   candidates are the composite band (§7, with exact corner treatment) or —
   more promising, because M3 already carries Prop. 12 — the localized M3
   floor, whose union construction dominates C = 1 pointwise by
   construction.
3. The identifiability frontier at the corner, restated in rank space: no
   band can certify a nonvacuous lower bound below ~c/n₀ — connects to the
   old Beta-floor honesty result and bounds the achievable width in the
   steep-corner strata.
4. Characterize the shape functional driving the per-shape optimal exponent
   (range 1.6–3.1). Round 3 narrowed the search: early slope is excluded
   (the ceiling is flat along a five-fold early-slope ladder) and the
   operative axis is roughness-like (t(2) sits below the entire binormal
   ladder; smoothing a calibration target at fixed shape moves its
   calibrated depth 30×). Round 4 (`r4_report.md` §2) went further and came
   back mostly empty-handed: over 32 rank-computable candidates in 5
   families across 14 cells, in-sample correlations reach |r| 0.7–0.9 and
   LOO RMSE of C* falls 0.19 → 0.13, but **nothing survives out of
   sample** — held-out spread reductions sit inside MC noise, α=.05
   coverage *degrades* under the rules (7 of 14 cells below 0.94, min .910,
   vs 3 under fixed C=2), and the best-LOO two-predictor rule blows up on
   the 9:1 cell (Ĉ = 5.0). Two eliminations and one surprise: the axis is
   *not* a concavity defect (bimodal .90 is exactly concave yet sits below
   the ladder); and the co-movement/Wald pathology does **not** bite for
   functional-driven levels (within-cell |ρ| ≤ 0.25) — the failure is pure
   lack of signal, not the mechanism that killed the calibration routes.
   The axis remains unidentified; this is now a theory question (item 1's
   second-order analysis), not a search problem.
5. Literature check before claiming novelty — **first web pass done
   2026-08-23; see `fiducial_band_theory.md` §14 for the full accounting.**
   Headlines: the min-p trim + tube is the *global rank envelope*
   (Myllymäki et al. 2017, JRSS-B; = extremal depth, Narisetty & Nair 2016
   JASA; tube-from-draws back to Besag et al. 1995) and the named-curve
   exact test is their rank envelope test — cite, don't re-derive; the
   *nearest existing* ROC cloud is the Bayesian bootstrap of Gu–Ghosal–Roy
   (2008, Stat. Med.), pointwise only and not identical to ours (the BB
   puts n Dirichlet weights on the observed atoms and pins the extremes;
   our cloud is the (n+1)-spacings GFD, which carries the corner mass —
   an O(1/n) difference that is exactly the corner channel); the
   one-sample fiducial band with a
   functional Bernstein–von Mises theorem is Cui & Hannig (2019,
   Biometrika) — the proof template for Theorem 7. Plausibly novel
   (assessed honestly as combination-driven): the
   two-sample composition of spacings-GFDs as a *band*, the
   C-remap/roughness calibration study (an empirical phenomenon plus a toy
   model until open problem 1 is solved), the corner-necessity sketches. Practical imports: ERL
   tie-breaking as a saturation fix (theory doc §9), exact one-sample ELL
   levels via `qqconf` for M3, and Cui–Hannig's interval-valued treatment
   as an alternative to within-gap spreading.

---

## 9. Reproduction

- Round 1 (oracle/plug-in/fiducial falsification, cells C1–C7):
  `stats/experiments/rank_band_experiments.py`, results `res_C*.json`,
  `res_fidcp.json`; summary in git history of this file (§8b of the
  2026-08-21 version).
- Round 2 (P1–P5: recalibration, new slices, ties, M-vs-K, α-sweep):
  `stats/experiments/m2_experiments.py` + helpers; full report
  `stats/experiments/m2_report.md`; results `res_p*.json`,
  `res_baselines_p2.json`.
- Round 3 (M3 guarantee layer, miss cap, containment probe, M4b bracket,
  C*(n) ladder to n = 20,000): `stats/experiments/m3_experiments.py`,
  `m4_experiments.py`, `analyze_m3.py`; results `res_m3_*.json`,
  `res_m4_*.json`, `res_cstar_*.json`; full report
  `stats/experiments/m3m4_report.md`.
- Round 4 (fiducial-predictive calibration, roughness-functional search,
  exact named-curve test, M3 worst-case-level map, steep-corner repair
  probe): `stats/experiments/r4_experiments.py`, `r4_analyze.py`; results
  `res_r4_*.json`; full report `stats/experiments/r4_report.md`.
- Production implementation: `src/studroc_paper/methods/fiducial_band.py`;
  tests `tests/test_fiducial_band.py`; validation of the implementation
  against harness numbers: `validate_production.py` (scratch; figures quoted
  in the implementation-validation note below).

### Implementation-validation note

The production `fiducial_band` was validated against the harness on the
binormal .95, n=500/500 cell (150 fresh reps, M=3000 to match the harness;
coverage SE ≈ 1.6pp at the 95% level, ≈ 4pp at 50%):

| arm | α | production | harness (400 reps) |
|---|---|---|---|
| trim_exponent=1 (`fid_cp`) | .05 | .973 (area .0640) | .975 (area .0634) |
| trim_exponent=1 | .50 | .760 (area .0420) | .748 (area .0409–.0418) |
| trim_exponent=2 (`fid_rc`) | .05 | .960 (area .0584) | .963 @C=2.2 (area .0571); C=2.0 expected slightly higher |
| trim_exponent=2 | .50 | .553 (area .0350) | .500 @C=2.2; C=2.0 expected ~.52–.57 |

All within Monte Carlo error of the harness. The low-M warning (realized
trim depth j < 3) fired as designed at M=3000, α=.05 — the auto budget
would choose M ≈ 5,200 for this cell. Runtime ≈ 0.31 s/band at n=500,
M=3000, CPU.
