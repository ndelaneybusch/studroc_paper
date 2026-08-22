# The Rank-Space Fiducial ROC Band: Working Model and Evidence

*Status (2026-08-21): consolidated after two laptop experiment rounds. This
document is a working model of what the full simulation suite should show,
based on what has actually been measured — with the uncertainties stated. The
full suite is the arbiter; nothing here is a result of that suite yet.*

*Method implementation: `src/studroc_paper/methods/fiducial_band.py`
(`fiducial_band`, exported from `studroc_paper.methods`; unit tests in
`tests/test_fiducial_band.py`). Evidence: `stats/experiments/` — harnesses
`rank_band_experiments.py` and `m2_experiments.py`, ~20 result JSONs, and the
detailed second-round report `m2_report.md`. Earlier ideation that this work
descends from is summarized in §7–§9; superseded content has been cut.*

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
   **α_eff = 1 − (1−α)^C with C = 2**. Band = pointwise [j-th smallest,
   j-th largest] of the draws, j = the α_eff-quantile of the depths.
4. Two exact binomial corner allowances at the band's own local level
   ℓ = j/(M+1): upper edge ∪ Clopper–Pearson upper bound (essential — the
   upper edge must equal 1 wherever empirical TPR = 1); lower edge = 0
   wherever empirical TPR = 0 (free).

Tuning inputs: M (a Monte Carlo budget, self-diagnosing — the method warns
when the realized trim depth j < 3; rule of thumb M ≳ 5/ℓ(K,α), ≈10,000 at
n₀ = 5,000) and the exponent C = 2 (empirically fitted; conjectured
structural — see §5.2). There are no variance floors, gates, jurisdiction
constants, or ε regularizers.

---

## 2. Evidence base — what was actually measured

Two rounds, ~20 cells, 120–400 replicates per cell (coverage SE ≈ 1.1pp at
400 reps for the 95% level; ≈ 2.2pp at 120 reps; several pp at the 50%
level). Cells span n per class 25–5,000; AUC 0.55–0.99; binormal, bimodal-
negative, t(2), and kinked truths; 9:1 class imbalance in both directions;
score quantization to 20 and 100 levels. By rank invariance, cells are curve
*shapes*; parameter sweeps within a family (e.g. t's df at fixed shape) are
provably redundant.

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

**P-A. Coverage at α=.05 lands in ~0.94–0.99 in every (DGP, n) stratum, flat
in n.** *(Confidence: moderate-high.)* Basis: 14-cell flatness and the
structural rank invariance; the suite's DGP families reduce to curve shapes
similar to those tested. Falsifiers to watch: LHS shapes with steeper corners
or plateau structures not represented in the hand-picked cells; the n=10,000
configuration (K=10,001), which was **never run** — the M rule extrapolates
to M ≈ 11–12k there. A stratum below ~0.93 at adequate M would contradict
the model; a stratum at 1.00 would mean the map is more conservative there
than any tested shape.

**P-B. With C=2, coverage at α=.5 is centred but dispersed: stratum values
roughly 0.40–0.60, mean near 0.50.** *(Confidence: moderate.)* The measured
per-cell optimal exponent ranges 1.6–3.1, and dcoverage/dα_eff ≈ 1 at α=.5,
so the ±15pp shape spread seen in the cells should reappear as ±10–15pp
stratum spread. This is the model's honest claim: *centred at every α, not
calibrated at every α*. Falsifier: mean far from 0.50, or spread much larger
than the cells showed. (If the suite is run with the identity map instead,
expect 0.65–0.86 at α=.5 — valid, conservative.)

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

1. **Shape dependence of calibration at central α survives the C=2 remap**
   (±13–19pp spread at α=.2/.5; bias removed, spread remains). A level-only
   correction cannot fix this; per-shape calibration would be needed, and
   the obvious data-driven route (per-rep plug-in calibration of the trim
   depth) is measured to inherit the plug-in bias (1.3–1.7× conservative in
   j) at 80× compute — not recommended. Untried alternative:
   fiducial-predictive calibration (calibrate the trim against draws from
   the fiducial cloud itself).
2. **C = 2 is empirical.** It has a natural reading (one Šidák budget per
   sample class) and held on all 14 cells (per-cell fitted range 1.6–3.1),
   but there is no proof. If the suite's LHS shapes produce strata where the
   effective exponent is far from 2, α=.05 coverage could dip below 0.94 —
   the identity map (C=1) is the conservative fallback and never dropped
   below 0.967.
3. **No coverage theorem.** All validity evidence is Monte Carlo. The
   fiducial composition is not automatically a confidence procedure; the
   oracle/test-inversion framing (§7) is the likely proof route, and the
   degenerate-corner allowances are exactly the places where the naive
   "fiducial = confidence" heuristic measurably failed before repair.
4. **Steep-corner width at small n** (2–3× oracle at AUC .99, n=150). Valid
   but loose; the one width regime where the method leaves real money on
   the table.
5. **Untested configurations:** n = 10,000 (largest suite size); prevalence
   10% at n=1,000 (nearest tested: 900/100 at n=1,000 total); the full LHS
   shape sweep; α = 0.5 on the two largest n. Also the n=25 cell needed a
   larger effective level than the C=2 map gives (map is conservative
   there — safe direction, but +9–13pp over at central α).
6. **Compute scales with M·K.** ~11 s/band at n₀=5,000 single-core; the
   suite multiplies this by ~10⁴ bands. GPU batching of draws (already
   chunked in the implementation) or per-n M tuning will matter.
7. **Estimand under ties** must be declared (trapezoidal, random break).
   Deterministic even-spreading is a valid conservative alternative;
   class-ordered tie-breaking is invalid and the implementation refuses it.

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
- From the earlier envelope-era experiments (see
  `project_evaluation_report.md`): logit-space construction; Wilson-gate
  redesigns; the variance-model band (noisy variance × supremum).

---

## 7. Backlog: ideas retained but not currently needed

- **M3 — composition of two exact one-sample (Berk–Jones/equal-local-level)
  bands.** A provable finite-sample distribution-free band; generalizes the
  old Beta and Wilson floors into one object. Superseded as the primary
  method by the fiducial band's empirical performance, but remains (i) the
  guarantee layer if a formal validity claim is required, (ii) a possible
  outer cap making "misses are small" provable, (iii) the likely bridge to
  a coverage theorem.
- **M4 — exact rank-test inversion.** The theoretical ideal: H₀: R = R₀ is
  simple in rank space, so an exact confidence set exists by test inversion;
  bands are its projections. Intractable directly; relevant as the frame for
  proving what the fiducial band approximates, and for bracketed worst-case
  calibration if plug-in-free guarantees are ever needed.
- **Fiducial-predictive trim calibration** (untried): the one live idea for
  removing the residual central-α shape spread.
- **Oracle band as a published benchmark:** the exact rank-space width
  ceiling is computable for any (R_true, n₀, n₁) and makes a clean yardstick
  figure for the paper regardless of method.

---

## 8. Open theory questions

1. Prove (or refute) that C = 2 is the correct level remap asymptotically —
   the conjecture is that the fiducial trim spends one simultaneity budget
   for the composed curve where the frequentist requirement is one per
   class-CDF.
2. A coverage theorem for the fiducial composition + degenerate-corner
   allowances, plausibly via the Dirichlet process / Bayesian bootstrap
   literature for a single CDF, extended to the two-sample composition.
3. The identifiability frontier at the corner, restated in rank space: no
   band can certify a nonvacuous lower bound below ~c/n₀ — connects to the
   old Beta-floor honesty result and bounds the achievable width in the
   steep-corner strata.
4. Characterize the shape functional driving the per-shape optimal exponent
   (range 1.6–3.1) — if it is estimable from ranks, a data-driven level
   without the plug-in bias may exist.
5. Literature check before claiming novelty: fiducial/Bayesian-bootstrap
   simultaneous bands for a single CDF; two-sample confidence bands via
   Dirichlet spacings; equal-local-levels (Berk–Jones, Nair's equal-precision
   bands) applied to ROC; Campbell (1994); Claeskens et al. (2003);
   Macskassy–Provost–Rosset (2005); Hall–Hyndman–Fan smoothed ROC bands.

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
