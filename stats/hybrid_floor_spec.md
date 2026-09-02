# Study Spec: The Localized M3 Floor — Region Geometry, Derisk, and Cross-Family Transfer ("Stage F")

*2026-09-02. Successor to the follow-up run plan of `c_calibration_spec.md`
(items E1–E3 of the 2026-09-02 planning discussion, re-cooked). Companions:
`stats/c_calibration_followup_report.md` (the boundary study this builds
on), `stats/fiducial_band_theory.md` §7.3 (the wedge, the miss geometry,
and the floor's domination property), `stats/next_method_ideas.md` §7
(roster #4). Purpose: decide the paper's fiducial roster entry before the
final `simulation_spec.md` run.*

---

## 0. Motivation and the decision this study serves

The boundary study left the project with a validated failure and a
promising, under-validated repair. The failure: at C = 1 the fiducial band
under-covers inside a curved (AUC, n) wedge — worst .645, failures to
n = 6,656, coverage non-monotone in n — and the suite's probit-weighted
AUC sampling concentrates cells exactly there. The repair: misses are
FPR-localized (upper end plus a small left corner), M3 covers 100% of the
observed miss points, and a pointwise union with M3 on a fixed FPR region
lifted five failing cells to .955–.990 at +6.4% mean width — one fifth of
routing's cost — while **provably never covering worse than C = 1** (the
union only widens; monotone closure preserves the ordering).

That probe is not a method yet. Its region was chosen on the five cells
that score it, its left cutoff is mis-parameterized (FPR .005 spans 0.65
grid points at n = 130), its width where C = 1 was already fine is
unpriced, imbalance has never touched it, and everything is student-t.
The three studies here take it from probe to decision:

- **Study A (exploratory)** learns the *geometry*: in what coordinate the
  miss region is invariant, and how the optimal cutoffs move with
  (n_pos, n_neg, AUC). Its output is a frozen region rule.
- **Study B (derisk)** validates the frozen rule out-of-sample at small
  scale — the E1 role, now with a rule worth testing rather than a guess.
- **Study C (cross-family)** tests transfer beyond student-t — the E3
  role, placed where the t-family evidence says the danger should be.

**The decision at the end:** the roster's fiducial entry is (a) the
hybrid floor applied unconditionally, (b) the floor applied
routing-gated, or (c) plain C = 1 with the (AUC, n) router — in that
preference order, each falling back to the next on a failed gate.

**Why the floor can be optimized aggressively and validated cheaply — the
asymmetry this design exploits.** Enlarging or moving the floor region
can only widen the band, so region selection *cannot break validity
relative to C = 1* (the domination property, exact). What in-sample
optimization CAN do is (i) overstate how much of the wedge the floor
repairs — domination gives ≥ C = 1, not ≥ .95 — and (ii) understate
width. So Study A optimizes freely on training cells; the two quantities
optimization can corrupt — repair sufficiency and width — are exactly
what Studies B/C measure on fresh seeds and held-out shapes. This is a
materially safer position than any previous calibration effort in this
project, all of which tuned quantities that could silently destroy
coverage.

---

## 1. The object

For a dataset with n₀ negatives, n₁ positives, level α, and a region
R ⊂ [0, 1] of the FPR axis:

```
hybrid(t) = [ min(L_fid(t), L_M3(t)), max(U_fid(t), U_M3(t)) ]   for t in R
          = [ L_fid(t), U_fid(t) ]                               elsewhere
```

with monotone closure of both edges (running max, as validated for the
composite stitch — coverage-event-preserving on the lower edge,
production convention on the upper). `fid` is the shipped C = 1 band;
`M3` is `m3_band_rs` at level α₂ (α₂ = α is the probe's choice; a
smaller α₂ is the theorem-friendly variant — see A-Q4).

Two facts frame everything:

- **[Exact] Domination.** hybrid coverage ≥ C = 1 coverage, identically
  per replicate, for any R.
- **[Exact] Regional miss cap.** P(M3 misses anywhere in R) ≤
  P(M3 misses anywhere) ≤ α₂, so the floored region carries an exact
  bound and only the un-floored interior claim is empirical — the
  two-piece guarantee the composite band lacked.

What is *not* guaranteed and must be measured: that R captures enough of
the C = 1 miss set to reach .95 inside the wedge (sufficiency), and that
the union costs little where C = 1 was already fine (price).

The rule must be a function of observables only:
`R = R(n₀, n₁, AUC_ub)` with `AUC_ub` an upper confidence bound on the
empirical AUC (noise must err toward more floor, which is the safe
direction here — unlike routing, where it errs toward more M3).

---

## 2. Study A — region geometry (exploratory; the questions come first)

### The questions

- **A-Q1 (coordinate).** In which coordinate is the miss region invariant
  across (n₀, n₁, AUC)? Candidates, chosen because each encodes a
  different mechanism:
  - raw FPR `t` (the probe's choice — known bad at the left end);
  - left-edge grid count `k = n₀·t` (negatives above threshold —
    resolution of the *negative* tail);
  - right-edge negative count `n₀·(1 − t)`;
  - right-edge **positive** count `n₁·(1 − TPR_hat(t))` (positives below
    threshold — resolution of the *positive* tail, which the mechanism
    says drives the upper-end misses: the true ROC's slow approach to 1
    is a positive-tail phenomenon);
  - the m-family, `m_q = n₀·t_q_hat` (exceedance counts at empirical TPR
    quantiles — the report §4 coordinate).
  The winner is the coordinate minimizing the cross-cell dispersion of
  the miss-mass quantiles (1% and 99%) relative to its own scale,
  *judged on a held-out cell split* — coordinate selection is a fitting
  step and gets fitting discipline, even though the domination property
  would forgive a wrong choice with width rather than coverage.
- **A-Q2 (sufficiency surfaces).** Per cell, the minimal region in the
  winning coordinate capturing ≥ 99.5% of miss mass; then conservative
  (outer-quantile, margin-added) surfaces for the two edges as functions
  of (n₀, n₁, AUC). Left and right edges are modeled separately — the
  mechanism predicts the left edge scales with n₀ and the right with n₁,
  and only the imbalance cells can distinguish that (A-Q5).
- **A-Q3 (price curves).** Width cost of the floor as a function of each
  cutoff, per cell, from stored mean band/M3 width profiles — computable
  offline for *any* candidate region without re-simulation. Deliverables:
  where the "upper half is nearly free" claim quantitatively holds and
  where it breaks (low AUC? large n? imbalance?), and the knee of the
  cost curve at the left edge. This prices the *unconditional* form of
  the method on cells that never needed repair.
- **A-Q4 (floor level).** Coverage/width at α₂ ∈ {α, α/2} — offline from
  stored M3 edges at both levels. If α/2 costs little width, the shipped
  form takes the stronger two-piece statement for free.
- **A-Q5 (imbalance).** First data ever on the floor under n₀ ≠ n₁; the
  edge-scaling question above, plus whether the wedge itself moves.

### Design

Two data sources, exploiting deterministic seeding:

1. **Replay corpus (no new randomness):** ~40 cells selected from the 257
   — all cells with measured coverage < .94, plus ~15 near-boundary cells
   (.94–.97) and ~8 comfortably-safe cells for price curves — re-running
   200 reps each with the *same seeds* but now storing, per rep: the
   C = 1 band's pointwise miss intervals (sparse, in grid indices), and
   per cell: mean fiducial and M3(α), M3(α/2) width profiles, plus the
   per-rep observables (empirical AUC, khat-derived m_q at q ∈
   {.3, .5, .7}, positive/negative exceedance counts at candidate edges).
   One band build per rep is the cost; everything downstream is offline.
2. **New imbalance LHS (fresh cells, the unexplored axis):** 24 cells,
   maximin-free LHS (frozen seed) over probit-AUC ∈ [.85, .99] ×
   log df ∈ [1.1, 30] × log n_total ∈ [400, 10,000] × log(n₀/n₁) ∈
   [1/5, 5] (achievability-filtered as always), 200 reps, same records.
   Concentrated at AUC ≥ .85 because that is where both the wedge and the
   region-geometry questions live; the safe-region price curves come from
   source 1. Per the standing design preference: cells over replicates —
   the geometry pools miss events across cells.
3. **Wedge-extent piggyback:** 4 cells at AUC ≥ .985, n ∈ {8,000, 12,000}
   balanced (the old E2, folded in): geometry at the largest n, and the
   does-the-wedge-close question as a by-product.

**Splits, pre-committed:** cells are split 60/40 into
coordinate-selection/edge-fitting vs. held-out *before analysis*, split
stratified by (AUC band, imbalance). A-Q1's winner and A-Q2's surfaces
are chosen on the 60; their invariance and sufficiency are checked on
the 40. The frozen rule (coordinate, edge surfaces, margins, α₂) is then
written into this spec as a dated amendment **before Study B runs**.

### What Study A does *not* do

It does not certify coverage (that is B/C), does not fit anything a
smoother must extrapolate at a cliff (the surfaces are conservative
outer envelopes with margins, in a coordinate chosen for invariance —
the opposite of the failed boundary smooths, which had to predict a
discontinuity in a coordinate chosen for convenience), and does not
touch the composite's interior trim (that remains a separate,
compatible width play once the floor is settled).

---

## 3. Study B — the derisk (E1's role, on a frozen rule)

Fresh cell names (⇒ fresh seed streams), the frozen rule from A applied
verbatim, ~24 cells at 400–500 reps, both α ∈ {.05, .5}:

- **10 wedge cells:** t2_99 at n ∈ {250, 500, 1000} (the worst known
  family, including the .645 cell's configuration), the non-monotone
  traversal shape t(4.69)/.986 at n ∈ {400, 1200, 2000}, and four more
  spanning the m-window in both traversal directions.
- **6 safe cells:** binormal .75/.90, bimodal .90, kink, hetero, t(3)/.90
  at n ∈ {250, 1000} mix — the unconditional-use price where C = 1 never
  needed help.
- **4 imbalance cells** in the wedge-adjacent region (min class 300–1500,
  both orientations).
- **2 large-n cells** (n = 8,000–12,000, AUC ≥ .985).
- **2 regression cells:** the ties cell (Q = 20) and one held-out-library
  member, both with the floor applied.

Arms per rep: C = 1, M3(α), M3(α/2), hybrid at the frozen rule (both α₂
variants — nearly free since the bands are already built). Paired per-rep
records throughout.

**Pre-registered gates (the A1-letter bar per cell — point ≥ .94 AND
Wilson-95 lower ≥ .925 — with sequential top-up 400 → 1200 while the CI
straddles .94):**

- **G-B1 (repair):** every wedge/imbalance/large-n cell PASSes at α=.05.
- **G-B2 (price):** on the safe cells, hybrid width ≤ C = 1 width + 8%
  (paired), and no safe cell's coverage moves below its C = 1 value
  (domination makes this a harness check, not a risk).
- **G-B3 (level sanity):** at α = .5 the hybrid stays at or above C = 1's
  conservative coverage (again domination; reported, not risked).

**Decisions:** G-B1 ∧ G-B2 ⇒ the hybrid ships *unconditionally* as the
roster's fiducial entry, with the router demoted to documentation.
G-B1 ∧ ¬G-B2 ⇒ ships *routing-gated* (floor applied when AUC_ub ≥ .88).
¬G-B1 ⇒ the region rule is insufficient: fall back to roster #3 (the
router, which then must clear its own fresh confirmation per spec item
5), and the failed cells' miss geometry goes back to Study A's records
to diagnose whether the failure is coordinate, margin, or mechanism.

---

## 4. Study C — cross-family transfer (E3's role)

Everything above is student-t plus corner spot checks. ~14 fresh cells,
placed *by the mechanism, not by convenience*: for each of Weibull
(shape ≤ 1), gamma (shape ≤ 1), beta-opposing (α ≤ 1), bimodal-negative
(high separation), and two frozen-seed LHS draws from the paper's mapper,
choose (AUC, n) to put the cell's predicted m₅₀ inside the t-family
failure window (and one per family outside it as a control). Arms: C = 1,
M3, hybrid at the frozen rule. 400 reps, α = .05 primary.

Three questions, in decreasing order of consequence:

- **C-Q1:** does the hybrid PASS everywhere — including on any
  off-family cell where C = 1 fails? (The shipping gate: the floor must
  repair wedge-analogues it has never seen.)
- **C-Q2:** does C = 1 fail off-family where the m-coordinate predicts it
  should? (Mechanism transfer — upgrades the m-window from a t-family
  observation toward a general routing/region statistic.)
- **C-Q3:** do the C = 1 failures' miss locations fall inside the frozen
  region in the winning coordinate? (Geometry transfer — the direct check
  on A-Q1's out-of-family validity.)

A C-Q1 failure blocks unconditional shipping regardless of Study B and
sends the failing family's geometry back through Study A's pipeline (one
documented iteration, as with the Stage S library rule).

---

## 5. Budget, order, and infrastructure

| study | cells | reps | band builds/rep | est. CPU-h |
|---|---|---|---|---|
| A: replay corpus | ~40 | 200 | 1 (+ M3, ~free) | 2–3 |
| A: imbalance LHS | 24 | 200 | 1 | 1.5–2 |
| A: extent piggyback | 4 | 200 | 1 | 1.5 |
| B: derisk | ~24 | 400–1200 | 1 (+ arms, ~free) | 3–5 |
| C: cross-family | ~14 | 400 | 1 | 2–3 |
| **total** | ~106 | | | **~10–14** |

Order: A → (freeze the rule as a dated amendment here) → B → C, with B
and C parallelizable after the freeze. All infrastructure exists:
deterministic per-(cell, rep) seeding, `sample_scores`/
`replay_empirical_aucs` for observables, `fiducial_band_rs`/`m3_band_rs`
for arms, the Wilson classification and sequential-replication machinery
of `followup_runs.py`, and its refuse-to-mix output validation. New code:
the per-rep miss-interval recorder, the offline geometry/price analysis,
and the frozen-rule evaluator — all pure Python around existing entry
points, each with focused tests in the `test_followup_runs.py` style.

## 6. Deliverables

1. `data/results/hybrid_floor_<date>/` — per-cell records (miss
   intervals, width profiles, observables) and per-study summaries.
2. A dated **frozen-rule amendment** to this spec between A and B.
3. `stats/hybrid_floor_report.md` — A's geometry findings (coordinate
   verdict, edge surfaces, price curves, α₂ choice), B's gate outcomes,
   C's transfer verdicts, and the roster decision with its evidence.
4. On a shipped floor: the production implementation (a `floor="m3"`
   option or wrapper on both band entry points), unit tests, docstring
   and theory-doc §7.3 updates, and the two-piece guarantee written as a
   lemma; `simulation_spec.md` §5 roster updated to the decided entry.
5. Theory items riding along: the domination lemma and regional miss cap
   stated formally (they are one-paragraph proofs); the m-window's AUC
   drift characterized empirically as input to open problem 1's
   second-order analysis.

## 7. Risks

| risk | mitigation |
|---|---|
| No coordinate makes the region invariant | The rule falls back to the union of per-coordinate conservative regions — wider, still dominated-safe; B prices it |
| Region rule sufficient on t, insufficient off-family | C-Q1 gate; one documented refit iteration |
| Width price on safe cells exceeds the bar | Routing-gated fallback is pre-declared, not improvised |
| In-sample optimism in A | 60/40 split for selection; B/C are fresh seeds and shapes; the only quantities A can corrupt are ones B/C measure |
| Miss-interval storage blows up at n = 12k | Intervals are sparse (misses are rare and localized); cap at 64 intervals/rep with overflow flag |
| The wedge is unbounded in n at AUC ≥ .985 | Does not block the floor (domination + regional cap are n-free); affects only the paper's description of C = 1 alone |
