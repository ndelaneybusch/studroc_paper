# Study Spec: The Localized M3 Floor — Geometry, External Behavior, and Transfer ("Stage F")

*2026-09-02. Companion to `stats/c_calibration_followup_report.md` (the
boundary study), `stats/fiducial_band_theory.md` §7.3 (the wedge, miss
geometry, and exact statements), `stats/next_method_ideas.md` §7 (the
candidate roster), and `stats/c_calibration_spec.md` (study discipline and
reporting conventions). Stage F is an information-gathering and
technique-improvement study. It does not choose the paper's final method;
that decision belongs to the full suite and the authors.*

---

## 0. Purpose, scope, and guardrails

The follow-up study established a serious failure and a promising repair.
At `C = 1`, the fiducial band under-covers inside a curved (AUC, n) wedge:
the worst measured coverage is .645, failures extend to n = 6,656, and
coverage is not monotone in n. Its misses are strongly FPR-localized: mostly
near the upper FPR endpoint, with a smaller left-corner cluster. On five
student-t cells, M3 covered every observed fiducial miss point, and unioning
the two bands on `FPR in [0, .005] union [.5, 1]` raised coverage to
.955-.990 at +6.4% mean width, versus +28-46% for full M3.

That probe does not identify a production rule. Its region was selected and
scored on the same five cells, its left cutoff is below one native grid
interval at the smallest n, and its price outside the wedge is unknown.
Stage F therefore has three jobs:

1. learn a simple, observable region rule and understand its mechanism;
2. measure fixed rules on fresh data, including the original probe as an
   untuned benchmark; and
3. determine whether the miss geometry and repair transfer beyond the
   student-t family.

The following guardrails apply throughout:

- Every empirical statement is library- and range-relative. Only the M3
  theorem and the set-containment statements in §1 are exact.
- True AUC, true ROC values, and true miss locations may select simulation
  cells and score a rule, but may never be inputs to the evaluated rule.
- A data-adaptive rule is evaluated as one procedure, replicate by
  replicate. Marginal band coverage and marginal routing/region frequencies
  are not combined as though selection were independent.
- The simple fixed-region benchmarks remain in Studies B/C even if Study A
  finds a more elaborate rule. This separates validation of the observed
  lead from validation of a fitted successor.
- Study A may select and refit a rule. Studies B/C do not alter it. Any rule
  revised after inspecting B or C is a new version and requires new external
  data before receiving out-of-sample language.

The familiar A1-letter bar (coverage point estimate at least .94 and
Wilson-95 lower bound at least .925) and the strict .95 point bar are
reported for comparability with earlier studies. They are descriptive
yardsticks, not method-selection gates.

---

## 1. Statistical object and exact statements

### 1.1 Construction

Work on the native grid `t_k = k / n0`, `k = 0, ..., n0`. For the same
rank realization and tie break, construct:

- `fid`: the production fiducial band with `C = 1`;
- `M3(alpha2)`: `m3_band_rs` with its distribution-free defaults, including
  `assume_r0_zero=False`; and
- a possibly data-adaptive region `R(D)` computed only from the observed
  dataset `D`.

Construct both parent bands with their production closures first, then
stitch them pointwise:

```text
L_hyb(t) = min(L_fid(t), L_M3(t))   and
U_hyb(t) = max(U_fid(t), U_M3(t))   for t in R(D),

L_hyb(t) = L_fid(t) and U_hyb(t) = U_fid(t) otherwise.
```

The stitch need not be monotone at a region boundary. Close it using only
widening operations:

```text
L_hyb_closed[k] = min(L_hyb[j] for j >= k)   # reverse cumulative minimum
U_hyb_closed[k] = max(U_hyb[j] for j <= k)   # cumulative maximum
```

then clip to `[0, 1]`. This produces nondecreasing edges without raising the
lower edge or lowering the upper edge. The distinction is essential: the
earlier probe's running-maximum lower closure preserves the whole-curve
coverage event and `C = 1` domination, but an unfloored lower-edge miss can
propagate across a region boundary, so it does not by itself preserve the
regional M3 cap. The widening-only closure preserves both parent
containments pointwise. Tests must verify those containments directly for
every arm and tie convention.

The learned region has two components, fitted separately because their
mechanisms and resolutions differ:

```text
R(D) = R_left(D) union R_right(D).
```

Each component is an endpoint-connected set after mapping back to the FPR
grid. Flat empirical-ROC preimages are included in full, so inversion of a
count coordinate cannot create data-dependent holes.

The primary level is `alpha = .05`. Both `alpha2 = alpha` and
`alpha2 = alpha / 2` are retained as named variants. The same region learned
at `.05` is applied unchanged at `alpha = .5` as a transfer diagnostic; no
alpha-dependent edge model is fitted from one level and described as
general.

### 1.2 Observable rule contract

The evaluated rule has the form

```text
R = R(n0, n1, AUC_ub, empirical ROC summaries; frozen parameters).
```

Its primary AUC input is the following one-sided, distribution-free
bounded-differences upper bound, clipped to one:

```text
AUC_ub = min(1, AUC_hat
                + sqrt(0.5 * (1 / n0 + 1 / n1) * log(1 / delta))),
delta = .05.
```

This bound is a conservative design covariate, not an additional coverage
guarantee and not part of the `alpha` budget. A DeLong upper bound and the
point estimate are reported as sensitivity analyses but cannot silently
replace the primary input. For tied scores, `AUC_hat`, the empirical ROC,
and both bands use the same random tie-break realization.

The fitted region must be nested outward in `AUC_ub`: increasing the upper
bound may leave the region unchanged or enlarge it, but may not shrink it.
Outside the sampled `(n0, n1, AUC_ub)` support, the frozen rule uses
`R = [0, 1]`. That fallback is potentially expensive but contains the full
M3 band and avoids unsupported extrapolation.

### 1.3 What is exact

For any fixed or data-adaptive `R(D)` and any `alpha2`:

1. **Domination.** The final hybrid contains the final `C = 1` band
   pointwise. Consequently, its simultaneous coverage is at least the
   `C = 1` coverage for every DGP and every replicate.
2. **Regional miss cap.** If the hybrid misses the true ROC anywhere in
   `R(D)`, M3 must miss there too. Because the event that M3 covers the
   entire curve implies coverage on every random subset,
   `P(hybrid misses somewhere in R(D)) <= alpha2`. No independence between
   `R(D)` and M3 is required.
3. **Two-piece decomposition.** Let `E_out` be the event that the `C = 1`
   band misses somewhere outside `R(D)`. Then

   ```text
   P(hybrid misses) <= alpha2 + P(E_out).
   ```

   Stage F estimates `P(E_out)`; it does not turn that term into a theorem.
   The hybrid has a full finite-sample guarantee only in the degenerate
   full-region case (or after a future argument controls the exterior
   term).

These statements distinguish the localized floor from the earlier
composite band: the floor never narrows `C = 1`, and the floored piece has
an exact cap, while the unfloored piece remains empirically justified.

---

## 2. Rules carried into external evaluation

Studies B/C evaluate all of the following; none is dropped after seeing
Study A:

1. **Legacy replication comparator (`probe_legacy`).**
   `R = [0, .005] union [.5, 1]`, followed by the probe's original running
   maximum of both edges. This is the exact construction behind the
   five-cell +6.4% result. It is retained only to test replication; it has
   `C = 1` domination but not the regional-cap claim in §1.3.
2. **Theorem-preserving probe (`probe_fpr`).**
   The same region with the widening-only closure in §1.1. This isolates
   the price of preserving M3 containment from the price of changing the
   region.
3. **Count-normalized benchmark (`count5`).**
   `R_left = {t_k: k = 0, ..., 5}` and `R_right = {t >= .5}`, with the §1.1
   closure. This implements the follow-up report's pre-data recommendation
   to express the left piece in grid points; it is a benchmark, not a fitted
   optimum.
4. **Learned rule (`stage_f_v1`).**
   Study A selects its coordinates and model form, refits numeric parameters
   on all Study A cells, and serializes the result before B/C begin.

Each rule is evaluated with both M3 levels from §1.1. Full `C = 1` and full
M3 are the parent references. Exact regional-cap language is restricted to
the three widening-closure rules.

---

## 3. Estimands and records

For replicate `r`, let

```text
V_r = {t_k: the true ROC is below L_fid or above U_fid at t_k}
```

after the production closure. `V_r` is simulation-only scoring information.
The primary region-sufficiency estimand in cell `c` is the **exterior escape
rate**

```text
q_c(R) = P(V_r is nonempty and V_r is not a subset of R(D_r)).
```

This is the part of the fiducial failure probability that a perfect floor
inside `R` could not repair. It is preferable to pooled "miss mass": a
region that captures 99.5% of pointwise violations can still miss many
replicates if each has one violation just outside the edge. The conditional
capture rate `P(V_r subset R | V_r nonempty)` and pointwise miss-intensity
quantiles are secondary geometry summaries.

For every rule and level, report:

- realized simultaneous coverage of the complete data-adaptive procedure,
  cell by cell, with Wilson intervals;
- `q_c(R)`, conditional capture, and the fraction of floor-region failures;
- mean area (project convention: mean grid-point width), paired area
  difference and ratio versus `C = 1`, and paired difference versus full M3;
- width cost split into the left and right components, plus overlap;
- miss direction, maximum depth, and violation intervals; and
- region size and selection summaries versus `(n0, n1, AUC_hat, AUC_ub,
  m_30, m_50, m_70)`.

Macro summaries weight cells equally. Pooled replicate summaries are shown
only alongside them, never as substitutes. Width uncertainty uses paired
cell-cluster bootstrap intervals; coverage intervals remain cellwise.

For offline evaluation of an adaptive rule, cell-mean width profiles are
insufficient. Store per-replicate union-width increments and lossless
cumulative summaries for every candidate left/right coordinate (or the full
compressed profile), together with the empirical coordinate maps. Store
miss sets as run-length intervals; if a replicate exceeds 64 intervals,
fall back to a packed bitset and set an overflow flag rather than truncating
the truth.

---

## 4. Study A — geometry and rule learning

### 4.1 Questions

- **A-Q1: endpoint coordinates.** Which coordinates transfer the left and
  right edge with the smallest exterior escape rate at a given width price?
  The two endpoints are selected independently.
- **A-Q2: edge dependence.** How do the selected cutoffs vary with `n0`,
  `n1`, and `AUC_ub`, and does imbalance support separate class-size
  scaling?
- **A-Q3: price curves.** What width is paid for moving either edge, and
  where does the observed "upper half is nearly free" behavior fail?
- **A-Q4: floor level.** What extra width and reduction in floor-region
  misses result from `alpha2 = alpha / 2` versus `alpha`?
- **A-Q5: mechanistic covariates.** Do empirical `m_q` summaries improve
  external prediction after `AUC_ub` and class sizes, or repeat the failed
  history of fitted rank functionals?

### 4.2 Candidate coordinates and model forms

Left-edge coordinates:

- raw FPR `t`;
- negative-grid count `k_left = n0 * t`.

Right-edge coordinates:

- raw distance `1 - t`;
- negative-grid distance `k_right = n0 * (1 - t)`;
- empirical positive-tail count
  `p_right = n1 * (1 - TPR_hat(t))`.

The `m_q = n0 * t_q_hat` quantities are cell/replicate-level covariates,
not pointwise coordinates; they may condition an edge cutoff but cannot be
used as though they locate an arbitrary `t`. The primary model excludes
them. An `m_q`-augmented model is promoted only if it improves the internal
validation objective beyond its cell-bootstrap uncertainty.

Candidate edge models are deliberately low-complexity: constant cutoffs;
piecewise-linear surfaces in `(log n0, log n1, AUC_ub)` with extra knots
near one; and conservative binned outer envelopes. The untransformed upper
bound avoids an infinite covariate when the distribution-free bound clips
to one. Models must be nested outward in
`AUC_ub`, may use separate `n0` and `n1` terms, and may not extrapolate
outside Study A support. Flexible GP/thin-plate fits are diagnostics only;
the boundary study showed that good average deviance does not protect a
cliff.

### 4.3 Cells and replication

Study A uses three sources:

1. **Replay corpus:** about 40 cells from the existing 257: every cell with
   measured `C = 1` coverage below .94, about 15 cells in .94-.97, and
   about 8 comfortably safe cells for price. Re-run 200 replicates with the
   original seeds and refuse the replay if parity with the stored `C = 1`
   result fails beyond three combined Monte Carlo SEs.
2. **Imbalance LHS:** 24 new cells from a frozen, achievability-filtered LHS
   over probit-AUC in `[.85, .99]`, log df in `[log(1.1), log(30)]`, log
   total sample size in `[log(400), log(10000)]`, and
   `log(n0 / n1)` in `[log(1/5), log(5)]`; 200 replicates. Both imbalance
   orientations must occur in every coarse AUC band.
3. **Extent stress cells:** four balanced cells at AUC at least .985 and
   `n per class in {8000, 12000}`; 200 replicates. These are reserved for
   range stress and are not used to fit edge parameters.

The exact cell manifest, LHS seed, tie policy, and split assignment are
written before outcomes are generated. In balanced-cell notation, `n`
always means observations per class; otherwise `n0`, `n1`, or `n_total` is
written explicitly.

### 4.4 Selection discipline and frozen artifact

Non-stress cells are assigned by a deterministic hash to 60% model-selection
and 40% internal-validation partitions, stratified by AUC band, coverage
band, data source, and imbalance orientation. The replay corpus was itself
chosen using earlier results, so the 40% partition is internal validation,
not new external evidence.

Coordinate, model-form, and complexity choices use the 60/40 split. Among
candidate rules, first retain those with cell-macro exterior escape at most
.005 among evaluable geometry cells and no such cell above .02 on the
internal-validation partition. A cell is evaluable for this criterion when
it has at least ten fiducial failures; other cells still contribute to the
width objective. Choose the retained rule with the lowest macro width cost;
paired-cell bootstrap uncertainty breaks statistical ties toward the
simpler coordinate and model. If no rule meets the escape target, choose
lexicographically by worst-cell escape, macro escape, width, and then
simplicity. These are fitting targets, not coverage guarantees.

After model form and complexity are selected, refit only its numeric edge
parameters on all non-stress Study A cells. Freeze a machine-readable
`stage_f_v1` artifact containing formulas, coefficients/cutoffs, training
support, out-of-support behavior, M3 split ratio, tie semantics, code commit,
study seed, and a content hash. A dated amendment to this spec records the
same information before any Study B/C outcome is inspected.

`alpha2 = alpha` and `alpha / 2` remain separate frozen variants; Study A
reports their frontier but does not select one by an invented utility
threshold.

---

## 5. Study B — external behavior of fixed rules

Use fresh cell names and seed streams. The §2 rules are applied verbatim to
24 cells at 400 replicates, with `alpha = .05` primary and `.5`
secondary:

- **10 wedge cells:** t(2)/.99 at `n = 250, 500, 1000`; the
  t(4.69)/.986 traversal at `n = 400, 1200, 2000`; and four frozen cells
  spanning both directions through the empirical m-window;
- **6 safe cells:** mechanism-diverse binormal, bimodal, kink,
  heteroscedastic, and t-family shapes at a mix of `n = 250, 1000`;
- **4 imbalanced cells:** both orientations with minority class size
  300-1500, placed in the wedge-adjacent regime;
- **2 large-n cells:** AUC at least .985 and `n per class = 8000-12000`;
  and
- **2 regression cells:** the `Q = 20` random-tie cell and one frozen
  held-out-library shape.

The complete B manifest is frozen before Study A outcomes are inspected.
Known cells above are mandatory; the remaining parameter values are chosen
from the pre-Stage-F boundary report and feasibility mapper, not from the
learned region.

Paired arms use the same rank/tie realization:

- `C = 1`;
- full M3 at `alpha` and `alpha / 2`;
- `probe_legacy`, `probe_fpr`, `count5`, and `stage_f_v1`, each at both M3
  levels; and
- one explicitly exploratory piggyback arm: `stage_f_v1` plus composite
  interior trim `b0.02-0.95_C2.5` on its declared finite range only.

The composite arm cannot affect the floor rule or its conclusions. It
answers the roster's separate question of whether the +floor and -trim
width effects compose without losing the floor's repair.

At `.05`, top up 400 to 1,200 replicates while the Wilson interval for any
prespecified floor-only hybrid straddles .94. A cell still unresolved at the
cap is reported as such; no pooled result overwrites it. The `.5` arm is not
topped up solely to meet a `.05` reporting convention.

For every widening-closure hybrid failure, classify violations as:

- **inside R:** the union, and therefore M3, missed inside the floored
  region;
- **edge escape:** an exterior violation is within one native grid step of
  a region edge; or
- **far escape:** an exterior violation lies farther away, indicating a
  coordinate or channel failure rather than a one-step margin error.

Classify `probe_legacy` separately, including whether a failure inside its
nominal region was propagated there by the running-maximum lower closure;
do not interpret its inside-region rate through the M3 cap.

Report the classification before proposing any revision. Selection effects
are assessed directly from realized hybrid coverage per cell and from
coverage conditional on region-size bins; unconditional parent-band
coverage is never substituted for the adaptive procedure's coverage.

---

## 6. Study C — transfer beyond student-t

Use 14 fresh cells: paired inside-window/control placements for seven frozen
families or shapes — Weibull with shape at most one, gamma with shape at
most one, beta-opposing with parameter at most one, high-separation
bimodal-negative, heteroscedastic Gaussian, and two frozen-seed LHS draws
from the paper's DGP mapper. The inside-window member is chosen using the
true DGP only to place the simulation cell's `(AUC, n)`; the evaluated region
still receives observables only. The control is matched as closely as
feasible in AUC while placing predicted true `m_50` outside the t-family
failure window. Freeze the full manifest before Study A outcomes.

Run 400 replicates with `.05` primary, using the same paired arms as Study B
except the composite piggyback. Top up by the same rule where needed.
Answer separately:

- **C-Q1, repair transfer:** where `C = 1` fails, how much do the fixed
  hybrids repair, and at what width?
- **C-Q2, mechanism transfer:** do failures occur where the m-window
  predicts, including traversal in both directions?
- **C-Q3, geometry transfer:** do violation sets fall inside the fixed
  regions, and are residuals inside, edge, or far escapes?

A rule that fails C is not refitted and then called externally validated.
The residual classifications may define a `stage_f_v2` development study,
with new confirmation data.

---

## 7. Reproducibility, implementation, and checks

All arms share deterministic per-`(study, cell, replicate)` data seeds,
tie-break seeds, and (where applicable) fiducial-cloud seeds. Results refuse
to mix unless the cell manifest, rule artifact hash, M3 parameters, trim-grid
rule, alpha grid, and code version agree.

Required implementation work:

1. a per-replicate violation-set and cumulative-width recorder;
2. offline coordinate, capture, and price analysis using lossless
   per-replicate sufficient records;
3. a serialized observable-only region evaluator; and
4. a paired multi-arm runner with resume/refuse-to-mix behavior.

Focused tests must cover at least:

- final hybrid pointwise containment of `C = 1`, and of M3 on `R`, after
  widening-only closure;
- replication of the legacy running-maximum stitch and a counterexample
  demonstrating why it does not receive the regional-cap claim;
- equality with `C = 1` for an empty region and containment of M3 for a
  full region;
- endpoint inclusion, flat empirical-ROC preimages, and grid resampling;
- invariance of a frozen rule to true AUC/ROC metadata;
- shared random tie-breaking across AUC, region, and band arms;
- offline price/capture equality to direct reconstruction;
- overflow fallback without miss-set truncation;
- artifact round-trip and out-of-support full-region behavior; and
- resume/refuse-to-mix failures for changed rule hashes or design constants.

The M3 arm's coverage is a regression check on implementation, not a new
test of Proposition 12. Any material discrepancy triggers parity debugging
before scientific interpretation.

---

## 8. Budget and order

| study | cells | reps | expensive fiducial builds/rep | estimated CPU-h |
|---|---:|---:|---:|---:|
| A: replay corpus | ~40 | 200 | 1 | 2-3 |
| A: imbalance LHS | 24 | 200 | 1 | 1.5-2 |
| A: extent stress | 4 | 200 | 1 | 1.5 |
| B: fixed-rule external behavior | ~24 | 400-1200 | 1 | 3-5 |
| C: cross-family transfer | 14 | 400-1200 | 1 | 2-3 |
| **total** | **~106** | | | **~10-15** |

Order:

1. freeze all cell manifests and split assignments;
2. run A, select/refit the rule, and record the dated artifact amendment;
3. run B and C (they may run in parallel because neither updates the rule);
4. write the report before any `stage_f_v2` work begins.

M3 computation and offline stitches are cheap relative to the one fiducial
cloud per replicate. Dry runs must replace these estimates with measured
budgets before launching the full study.

---

## 9. Deliverables

1. `data/results/hybrid_floor_<date>/`: manifests, design constants,
   per-replicate records, rule artifacts, and per-study summaries.
2. A dated amendment here containing the complete `stage_f_v1` rule and
   hash before B/C outcomes are read.
3. `stats/hybrid_floor_report.md`: A's coordinate/edge/price findings,
   direct external results for both simple benchmarks and the learned rule,
   alpha2 frontier, residual-miss classifications, cross-family transfer,
   and explicit empirical/exact labels.
4. Formal one-paragraph proofs of domination, the adaptive regional cap,
   and the two-piece decomposition in the theory document, including the
   widening-only closure condition that corrects the current §7.3 wording.
5. Updates to the §7 roster in `next_method_ideas.md`, including whether the
   original fixed-region lead replicated independently and whether the
   fitted rule added enough width efficiency to justify its complexity.

The report supplies evidence to the later roster decision; it does not make
that decision by relabeling the reporting bars as acceptance criteria.

---

## 10. Risks and planned interpretations

| risk | response |
|---|---|
| No coordinate transfers | Prefer the fixed count benchmark or a conservative union of endpoint regions; report the price. Do not hide the failure in a flexible surface. |
| `m_q` helps in-sample only | Keep it diagnostic unless its internal-validation gain clears cell-bootstrap uncertainty and B/C confirm the frozen rule. |
| Learned rule beats benchmarks only trivially | Prefer the simpler benchmark in the later roster discussion; Stage F still provides the geometry and price curves. |
| Region repairs t but not other families | Use C's inside/edge/far classification to distinguish margin error from a new miss channel; any refit becomes `stage_f_v2`. |
| A replay selection creates optimism | Describe A's split as internal validation; reserve out-of-sample language for fresh B/C seed streams and cells. |
| Data-adaptive region invalidates naive coverage arithmetic | Score the full procedure per replicate. The exact regional cap remains valid because full-curve M3 coverage implies coverage on every random subset. |
| Sparse interval encoding overflows | Store a packed bitset fallback and an overflow flag; never truncate a miss set. |
| Wedge persists beyond n = 12,000 | This does not affect domination or the adaptive regional cap; it limits the empirical exterior claim and favors the full-region fallback outside support. |
| Composite piggyback under-covers | Quarantine that result to the separate finite-range width candidate; it cannot weaken the floor-only arms or revise `stage_f_v1`. |
