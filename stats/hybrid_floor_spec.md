# Study Spec: The Localized M3 Floor — Region Geometry, Out-of-Sample Behavior, and Transfer ("Stage F")

*2026-09-02 (reframed same day: this study gathers information and improves
the techniques — it does not arbitrate the final method; that happens
later, with the full suite in hand). Companions:
`stats/c_calibration_followup_report.md` (the boundary study this builds
on), `stats/fiducial_band_theory.md` §7.3 (the wedge, the miss geometry,
and the floor's domination property), `stats/next_method_ideas.md` §7
(the method roster).*

---

## 0. Purpose: what we want to learn

The boundary study left a validated failure and a promising, barely
understood repair. The failure: at C = 1 the fiducial band under-covers
inside a curved (AUC, n) wedge — worst .645, failures to n = 6,656,
coverage non-monotone in n. The repair: misses are FPR-localized (upper
end plus a small left corner), M3 covers 100% of the observed miss
points, and a pointwise union with M3 on a fixed FPR region lifted five
failing cells to .955–.990 at +6.4% mean width, while **provably never
covering worse than C = 1**.

Everything we know about that repair comes from five cells whose region
was chosen on themselves. We do not know *why the region is where it is*
in any coordinate that generalizes, how its edges move with n₀, n₁, and
AUC separately, what it costs where C = 1 was already fine, what it fails
to repair and why, or whether any of it survives outside the student-t
family. Those are the gaps between "a probe that worked" and "the best
version of this technique," and closing them is the entire purpose here.

The improvement levers this study informs, and the information each
needs:

| lever | what would improve it | which study supplies that |
|---|---|---|
| Region coordinate | Knowing in which coordinate the miss set is invariant across (n₀, n₁, AUC) — the current FPR parametrization is provably wrong at the left end | A-Q1 |
| Edge placement | Conservative edge surfaces in that coordinate, with separate n₀/n₁ scaling if the mechanism's prediction holds | A-Q2, A-Q5 |
| Width efficiency | Price curves per cutoff — where the "upper half is nearly free" claim holds, where it breaks, where the left-edge knee sits | A-Q3 |
| Floor level α₂ | Coverage/width trade at α vs α/2 — a stronger exact regional cap may be nearly free | A-Q4 |
| Residual repair | Where misses remain *after* flooring, and whether they are region-insufficiency (edges too tight), coordinate error, or a different channel entirely | B, C |
| Mechanism / future statistics | The m-window's drift with AUC; whether it predicts off-family failures — feeds both a better region rule and any later routing statistic | A, C-Q2/Q3 |
| Compatibility with the interior trim | Whether floor and composite trim address disjoint width/validity budgets (they act on different parts of the curve; nothing yet measures them together) | B (piggyback arm) |

Measurement standards: cell results are reported against the familiar
yardsticks (the A1-letter bar — point ≥ .94 and Wilson-95 lower ≥ .925 —
and the strict .95 point bar) so numbers are comparable across the
project's studies. These are *reporting conventions for interpreting
results, not accept/reject gates*: a cell below the bar is a finding to
be diagnosed, and the diagnosis is often the most valuable output.

**Why this technique can be optimized aggressively while staying honest —
the asymmetry the design exploits.** Enlarging or moving the floor region
can only widen the band, so region choices *cannot break validity
relative to C = 1* (the domination property, exact). In-sample
optimization can only mislead about two things: how much of the wedge the
floor repairs, and what it costs. Study A therefore optimizes freely on
training cells, and Studies B/C measure exactly those two quantities on
fresh seeds and unseen shapes — with the residual-miss records that turn
any shortfall into the next design iteration rather than a dead end.

---

## 1. The object

For a dataset with n₀ negatives, n₁ positives, level α, and a region
R ⊂ [0, 1] of the FPR axis:

```
hybrid(t) = [ min(L_fid(t), L_M3(t)), max(U_fid(t), U_M3(t)) ]   for t in R
          = [ L_fid(t), U_fid(t) ]                               elsewhere
```

with monotone closure of both edges (running max — coverage-event-
preserving on the lower edge, production convention on the upper). `fid`
is the shipped C = 1 band; `M3` is `m3_band_rs` at level α₂.

Two exact facts frame everything:

- **[Exact] Domination.** hybrid coverage ≥ C = 1 coverage, identically
  per replicate, for any R.
- **[Exact] Regional miss cap.** P(M3 misses anywhere in R) ≤
  P(M3 misses anywhere) ≤ α₂: the floored region carries an exact bound,
  and only the un-floored interior claim is empirical — the two-piece
  structure the composite band lacked.

What is *not* guaranteed, and is exactly what the studies measure: that R
captures enough of the C = 1 miss set to restore ~.95 inside the wedge
(sufficiency), and that the union costs little where C = 1 was already
fine (price).

The region rule must be a function of observables only:
`R = R(n₀, n₁, AUC_ub)` with `AUC_ub` an upper confidence bound on the
empirical AUC (estimation noise then errs toward more floor, which the
domination property makes the safe direction).

---

## 2. Study A — region geometry (the exploratory core)

### The questions, and what each one buys

- **A-Q1 (coordinate).** In which coordinate is the miss region invariant
  across (n₀, n₁, AUC)? Candidates, each encoding a different mechanism:
  - raw FPR `t` (the probe's choice — known wrong at the left end: .005
    spans 0.65 grid points at n = 130);
  - left-edge negative count `k = n₀·t` (resolution of the negative
    tail);
  - right-edge negative count `n₀·(1 − t)`;
  - right-edge **positive** count `n₁·(1 − TPR_hat(t))` (resolution of
    the positive tail — the mechanism's favorite for the upper end, since
    the true ROC's slow approach to 1 under heavy positive tails is a
    positive-tail phenomenon);
  - the m-family, `m_q = n₀·t_q_hat` (exceedance counts at empirical TPR
    quantiles — the report §4 coordinate).
  Verdict criterion: the coordinate minimizing cross-cell dispersion of
  the miss-mass 1%/99% quantiles relative to its own scale, judged on a
  held-out cell split. *What it buys:* the right parametrization is the
  single biggest improvement available — it is the difference between a
  rule that transfers and one that must be re-fit per regime. Coordinate
  selection is a fitting step and gets fitting discipline, even though a
  wrong choice costs width, not coverage.
- **A-Q2 (edge surfaces).** Per cell, the minimal region in the winning
  coordinate capturing ≥ 99.5% of miss mass; then conservative
  (outer-quantile, margin-added) surfaces for the two edges over
  (n₀, n₁, AUC). Left and right edges modeled separately. *What it buys:*
  the floor stops being one hard-coded pair of numbers and becomes a rule
  that adapts to the dataset — narrower where the data say the danger is
  narrow.
- **A-Q3 (price curves).** Width cost of the floor as a function of each
  cutoff, per cell, computed offline from stored mean band/M3 width
  profiles for *any* candidate region without re-simulation. *What it
  buys:* quantifies where the upper region is genuinely free and where it
  is not (low AUC? large n? imbalance?), locating the efficient frontier
  between repair and width instead of guessing at it.
- **A-Q4 (floor level).** Coverage/width at α₂ ∈ {α, α/2}, offline from
  stored M3 edges at both levels. *What it buys:* if α/2 is nearly free,
  the technique acquires a strictly stronger exact statement at no
  material cost.
- **A-Q5 (imbalance).** First data on the floor under n₀ ≠ n₁. The
  mechanism predicts the left edge scales with n₀ and the right with n₁;
  only imbalanced cells can distinguish that. *What it buys:* either a
  two-argument edge rule with mechanistic support, or the discovery that
  one effective size suffices — both simplify what follows.

### Design

Two data sources, exploiting deterministic seeding:

1. **Replay corpus (no new randomness):** ~40 cells selected from the
   existing 257 — all cells with measured coverage < .94, ~15
   near-boundary cells (.94–.97), ~8 comfortably-safe cells for price
   curves — re-run at 200 reps with the *same seeds*, now recording per
   rep: the C = 1 band's pointwise miss intervals (sparse, grid indices)
   and observables (empirical AUC, m_q at q ∈ {.3, .5, .7}, exceedance
   counts at candidate edges); per cell: mean fiducial and M3(α), M3(α/2)
   width profiles. One band build per rep; everything downstream is
   offline re-analysis.
2. **New imbalance LHS (the unexplored axis):** 24 cells, maximin-free
   LHS (frozen seed) over probit-AUC ∈ [.85, .99] × log df ∈ [1.1, 30] ×
   log n_total ∈ [400, 10,000] × log(n₀/n₁) ∈ [1/5, 5],
   achievability-filtered; 200 reps, same records. Concentrated at high
   AUC because that is where the geometry questions live; per the
   standing design preference, cells over replicates — the geometry pools
   miss events across cells.
3. **Wedge-extent piggyback:** 4 cells at AUC ≥ .985, n ∈ {8,000, 12,000}
   balanced — geometry at the largest n, and whether the wedge closes as
   a by-product.

**Splits, pre-committed:** cells split 60/40 (selection vs. holdout)
before analysis, stratified by (AUC band, imbalance). A-Q1's winner and
A-Q2's surfaces are chosen on the 60 and checked on the 40. The resulting
region rule (coordinate, edge surfaces, margins, α₂) is recorded in this
spec as a dated amendment before Studies B/C run — not as a gate, but so
that B/C measure a *fixed* object and their numbers mean what they say.

### What Study A deliberately avoids

Certifying coverage (that is B/C's measurement), and fitting anything a
smoother must extrapolate at a cliff: the edge surfaces are conservative
outer envelopes with margins in a coordinate chosen for invariance — the
opposite of the failed boundary smooths, which had to predict a
discontinuity in a coordinate chosen for convenience.

---

## 3. Study B — out-of-sample behavior of the fixed rule

Fresh cell names (⇒ fresh seed streams), the recorded rule from A applied
verbatim, ~24 cells at 400–500 reps, both α ∈ {.05, .5}:

- **10 wedge cells:** t2_99 at n ∈ {250, 500, 1000} (including the .645
  configuration), the non-monotone traversal shape t(4.69)/.986 at
  n ∈ {400, 1200, 2000}, and four more spanning the m-window in both
  traversal directions.
- **6 safe cells:** binormal .75/.90, bimodal .90, kink, hetero,
  t(3)/.90 at n ∈ {250, 1000} mix — the price of unconditional use where
  C = 1 never needed help.
- **4 imbalance cells** in the wedge-adjacent region (min class 300–1500,
  both orientations).
- **2 large-n cells** (n = 8,000–12,000, AUC ≥ .985).
- **2 regression cells:** the ties cell (Q = 20) and one held-out-library
  member.

Arms per rep, all paired: C = 1, M3(α), M3(α/2), hybrid at the fixed rule
(both α₂ variants), and — as a piggyback costing one more stitch per
rep — **hybrid + composite interior trim** (the item-3 candidate
`b0.02-0.95_C2.5` inside the un-floored interior), giving the first
measurement of whether the validity repair and the width play compose.

**What Study B produces** (information, not verdicts):

- Repair sufficiency out-of-sample: hybrid coverage per wedge cell
  against the reporting bars, with Wilson intervals; sequential top-up
  (400 → 1,200) where the CI straddles .94, so borderline cells resolve
  instead of dangling.
- Price out-of-sample: paired width vs C = 1 on the safe cells, and vs
  full M3 on the wedge cells — the two comparisons that locate the
  technique between its parents.
- **Residual-miss geometry — the improvement loop's raw material.** For
  every rep the hybrid misses, the miss intervals in the winning
  coordinate, classified: inside R (the floor itself missed — bounded by
  the exact regional cap, so this should be rare and its rate is a check
  on the α₂ accounting), at the edges of R (edges too tight → margin or
  surface revision), or far from R (a different channel → new mechanism
  work). This classification, not any pass/fail, is what makes a
  disappointing cell useful.
- α = .5 behavior of the union (domination makes this a free sanity
  record, and the suite runs this level).
- Composite-stack behavior: does the interior trim's −6.8% survive under
  the floor, and does the floor's repair survive under the trim?

---

## 4. Study C — transfer beyond student-t

Everything upstream is student-t plus corner spot checks. ~14 fresh
cells, placed *by the mechanism, not by convenience*: for Weibull
(shape ≤ 1), gamma (shape ≤ 1), beta-opposing (α ≤ 1), bimodal-negative
(high separation), and two frozen-seed LHS draws from the paper's mapper,
choose (AUC, n) to put the cell's predicted m₅₀ inside the t-family
failure window, plus one cell per family outside it as a control. Arms:
C = 1, M3, hybrid at the fixed rule. 400 reps, α = .05 primary.

Three questions, each an independent piece of information:

- **C-Q1 (repair transfer):** does the hybrid restore coverage on any
  off-family cell where C = 1 fails? A shortfall here, run through the
  same residual-miss classification as Study B, says whether off-family
  miss geometry differs in location (fixable by edges) or in kind (new
  mechanism).
- **C-Q2 (mechanism transfer):** does C = 1 fail off-family where the
  m-coordinate predicts? A yes upgrades the m-window from a t-family
  observation toward a general statistic — valuable for any future
  region or routing rule regardless of what ships.
- **C-Q3 (geometry transfer):** do off-family miss locations fall inside
  the fixed region in the winning coordinate? The direct check on
  A-Q1's out-of-family validity.

---

## 5. Budget, order, and infrastructure

| study | cells | reps | band builds/rep | est. CPU-h |
|---|---|---|---|---|
| A: replay corpus | ~40 | 200 | 1 (+ M3, ~free) | 2–3 |
| A: imbalance LHS | 24 | 200 | 1 | 1.5–2 |
| A: extent piggyback | 4 | 200 | 1 | 1.5 |
| B: fixed-rule behavior | ~24 | 400–1200 | 1 (+ arms, ~free) | 3–5 |
| C: cross-family | ~14 | 400 | 1 | 2–3 |
| **total** | ~106 | | | **~10–14** |

Order: A → (record the rule as a dated amendment here) → B and C in
parallel. All infrastructure exists: deterministic per-(cell, rep)
seeding, `sample_scores`/`replay_empirical_aucs` for observables,
`fiducial_band_rs`/`m3_band_rs` for arms, the Wilson
classification/sequential-replication machinery of `followup_runs.py`,
and its refuse-to-mix output validation. New code: the per-rep
miss-interval recorder, the offline geometry/price analysis, and the
fixed-rule evaluator — pure Python around existing entry points, each
with focused tests in the `test_followup_runs.py` style.

## 6. Deliverables

1. `data/results/hybrid_floor_<date>/` — per-cell records (miss
   intervals, width profiles, observables) and per-study summaries.
2. The dated **region-rule amendment** to this spec between A and B/C.
3. `stats/hybrid_floor_report.md` — A's geometry findings (coordinate
   verdict, edge surfaces, price curves, α₂ trade), B's out-of-sample
   coverage/width/residual-miss picture, C's transfer findings, and an
   honest statement of what the improved technique now is, what it still
   fails to do, and what its numbers are — the inputs to the roster
   choice, which is made later and elsewhere.
4. Theory items riding along: the domination lemma and regional miss cap
   stated formally (one-paragraph proofs each); the m-window's AUC drift
   characterized empirically as input to open problem 1's second-order
   analysis.
5. Doc updates: theory doc §7.3 extended with A's geometry and B/C's
   measurements; `next_method_ideas.md` roster statuses refreshed.

## 7. Risks

| risk | mitigation |
|---|---|
| No coordinate makes the region invariant | Fall back to the union of per-coordinate conservative regions — wider, still dominated-safe; B prices it, and the failure itself localizes where the geometry is irreducibly multi-scale |
| Region rule sufficient on t, insufficient off-family | C's residual-miss classification distinguishes edge error from new mechanism; one documented refit iteration through A's pipeline |
| In-sample optimism in A | 60/40 split for selection; B/C are fresh seeds and shapes; the only quantities A can corrupt are the ones B/C measure |
| Miss-interval storage at n = 12k | Intervals are sparse (misses rare, localized); cap 64 intervals/rep with an overflow flag |
| The wedge is unbounded in n at AUC ≥ .985 | Does not undermine the floor (domination and the regional cap are n-free); it changes only how C = 1 alone is described |
