# Study Spec: The Frontier M3 Floor — Geometry, External Behavior, and Transfer ("Stage F")

*Theory-driven revision 2026-09-02. Companion to
`stats/fiducial_band_theory.md` §7.3–7.4,
`stats/c_calibration_followup_report.md`, `stats/next_method_ideas.md` §7,
and `stats/c_calibration_spec.md`. Stage F remains an
information-gathering and method-improvement study; the final suite and the
authors choose the paper's method.*

> **Pre-run status.** No Stage F manifest or result directory has been
> created. The current implementation evaluates the rank-only frontier rule
> directly. It supersedes the older `(AUC_ub, n0,n1)`-conditioned path.

---

## 0. Decision revision

The follow-up established a curved student-t failure wedge and a promising
localized M3 repair. The newer corner theory changes what should be learned
from those observations:

- Proposition 14 and the sliver construction show that AUC and sample sizes
  cannot bound endpoint risk over continuous ROCs. An AUC-conditioned region
  can be a declared-class rule, but not a distribution-free floor.
- Lemma 13 identifies a rank-observable frontier region before Stage F data:
  the first `ceil(Q)` negative-grid points and the empirical-TPR-1 run.
  Fiducial re-randomization requires a predeclared inward margin at the right
  boundary.
- Corollary 13.1 supplies a separate, class-relative prediction: corner
  concavity should make the two end-gap channels conservative. It is a
  sketch/leading-order claim, not yet a finite-sample coverage theorem.

Accordingly:

| previous plan | revised disposition | reason |
|---|---|---|
| Fit `stage_f_v1` from `(AUC_ub,n0,n1,m_q)` | Retire as the primary floor | It encodes the development library's shape class |
| Use a 60/40 split to select coordinates and surfaces | Remove | The primary region is theory-fixed before outcomes |
| Choose the rule after Study A | Define the rule before Study A | B/C no longer depend on an A fit |
| Treat AUC upper bounds as conservative routing inputs | Use AUC only for design and reporting | Risk is nonmonotone and not identified by AUC |
| Use cross-family `m_50` windows | Classify cells by corner geometry | Curvature/gap structure is the theory's operative distinction |
| Validate only ordinary library cells | Add fresh sliver confirmations | The constructed failure is the relevant distribution-free stress |

Stage F now has three jobs:

1. test the fixed frontier floor's capture and width, including the required
   right-edge margin;
2. measure it prospectively on wedge, safe, sliver, imbalance, large-n, and
   cross-family cells; and
3. test the corner-concavity prediction separately from the
   distribution-free floor.

The familiar A1-letter bar and strict .95 point bar remain descriptive
yardsticks, not method-selection gates.

---

## 1. Statistical object and exact scope

### 1.1 Parent bands and widening closure

On the native grid `t_k = k/n0`, construct from one shared rank/tie
realization:

- production fiducial `C = 1`;
- `M3(alpha2)` with `assume_r0_zero=False`; and
- a rank-measurable endpoint region `R(D)`.

Inside `R(D)`, take the pointwise union of the parent intervals; outside,
retain `C = 1`. Close only by widening:

```text
L_closed[k] = min(L_raw[j] for j >= k)
U_closed[k] = max(U_raw[j] for j <= k)
```

and clip to `[0,1]`. The legacy running-maximum lower closure remains a
replication comparator only.

### 1.2 Frontier region

Let `M` be the fiducial cloud budget for the cell. The realized
local level is `ell=j/(M+1)` with `j>=1`, so deterministically
`log(1/ell)<=log(M+1)`. Define

```text
k_left(n0, M) = min(n0, ceil(log(M + 1))).
```

The typical theory value remains `ceil(Q)≈7`. This bound deliberately uses
the cell's budget rather than reading a realized `Q` from the cloud, so the
region depends only on the planned budget, class sizes, and ranks.

For each dataset define

```text
j_k = n1 - khat_k
k_sat = min{k: j_k = 0}
K = n0 - k_sat
m(K) = ceil(2 * sqrt(max(K, 1))).
```

Here `j_k` is the empirical number of positives below the threshold and
`K` is the number of negative-grid intervals in the empirical-TPR-1 run.
The primary region is

```text
R_left  = {t_k: k <= k_left}
R_right = {t_k: k >= max(0, k_sat - m(K))}
R_frontier = R_left union R_right.
```

This is `frontier_floor_v1`. It uses only class sizes and the merged label
sequence. It does not use `AUC_hat`, `AUC_ub`, a fitted surface, a true
curve, or a declared DGP class. Flat empirical-ROC preimages are included
in full.

Two predeclared ablations identify the cost of the margin:

- `frontier_run0`: the same left region and exactly `j_k=0` on the
  right;
- `frontier_j1`: the same left region and the complete `j_k<=1`
  preimage on the right.

`frontier_run0` is not a shipping candidate because the theory says a
margin is required. `frontier_j1` is a conservative alternative whose
random length may be much larger than the square-root margin. Outcomes may
motivate a future version, but they do not revise `frontier_floor_v1`.

The same .05-derived region is applied unchanged at `alpha=.5` as a
transfer diagnostic. Both `alpha2=alpha` and `alpha2=alpha/2` remain
separate reported variants.

### 1.3 Exact statements and labels

For any fixed or data-adaptive region and either M3 level:

1. **Domination.** The widening hybrid contains `C = 1` pointwise, so its
   coverage is never lower than `C = 1` for any DGP or replicate.
2. **Regional cap.** A hybrid miss inside the random region implies a
   full-curve M3 miss. Therefore
   `P(miss somewhere inside R(D)) <= alpha2` without independence.
3. **Two-piece decomposition.** If `E_out` is a `C = 1` miss outside
   the region, then

   ```text
   P(hybrid miss) <= alpha2 + P(E_out).
   ```

These are exact. The claim that the frontier region removes every dangerous
corner miss is theory-motivated but not yet a theorem because Lemma 9,
Lemma 13, and Corollary 13.1 retain sketch/leading-order steps. In this
document, **distribution-free heuristic** means that the rule itself is
rank-only and library-independent and carries the exact domination and
regional-cap statements. It does not mean that the whole hybrid already
has a `1-alpha` finite-sample theorem.

The unmargined theory base has provisional area-price scale
`Q(1/n0+1/n1)`. The operational left envelope replaces its `Q/n0` term by
the conservative `log(M+1)/n0` term. The mandatory square-root margin adds
a worst-case grid fraction at most `m(K)/(n0+1)`; Stage F reports its actual
union-width cost separately. No stronger deterministic price bound is
claimed until the margin calculation is completed.

---

## 2. Frozen arms

Every B/C arm below is retained regardless of Study A:

1. `C = 1`;
2. full M3 at `alpha2=alpha` and `alpha/2`;
3. `probe_legacy`: `[0,.005] union [.5,1]` with the historical
   running-maximum closure;
4. `probe_fpr`: the same fixed region with widening closure;
5. `count5`: first six grid points plus `t>=.5`, widening closure;
6. `frontier_run0`, mechanism ablation;
7. `frontier_j1`, conservative-margin comparator; and
8. `frontier_floor_v1`, the primary fixed frontier rule.

Study B also keeps the explicitly exploratory composite piggyback:
`frontier_floor_v1` applied over the declared finite-range
`b0.02-0.95_C2.5` interior construction. It cannot revise conclusions
about the floor-only arms.

No AUC-conditioned learned rule is an arm of the primary Stage F
comparison. Offline AUC, `m_q`, finite-grid risk scores, and curvature
summaries remain mechanism diagnostics. A later declared-class router must
have its own name, class declaration, specification, and confirmation data.

M3's class split remains `rho=.5` in Stage F. The theory's proposed
size-only split optimization is a useful separate width study, but adding
it now would confound region and M3-level economics. Any future split must
be chosen independently of observed ranks to retain Proposition 12.

---

## 3. Estimands and records

For replicate `r`, let `V_r` be the complete post-closure `C = 1`
violation set. The primary region-sufficiency estimand in cell `c` is

```text
q_c(R) = P(V_r is nonempty and V_r is not a subset of R(D_r)).
```

Report, per cell and complete procedure:

- simultaneous coverage and Wilson interval;
- exterior escape, conditional capture, and floor-region failure;
- edge versus far exterior escape;
- mean area and paired differences versus `C = 1` and full M3;
- left, saturated-run, and square-root-margin width contributions;
- miss direction, depth, and complete violation intervals;
- `K`, margin length, `j_k` boundary, and realized region fraction; and
- diagnostic associations with class sizes, AUC summaries, `m_30,m_50,m_70`,
  finite-grid risk, and precomputed corner-geometry labels.

AUC and true geometry may stratify or explain results but may not determine
the frontier rule. Macro summaries weight cells equally. Width uncertainty
uses paired cell-cluster bootstrap intervals.

Store lossless parent bands, truth, `khat`, violation sets, coordinate maps,
and cumulative union-width increments. Violation encoding falls back from
run-length intervals to a packed bitset rather than truncating.

---

## 4. Study A — mechanism and price audit

Study A no longer selects or refits a rule. It asks:

- **A-Q1:** Does the fixed left frontier plus saturated-run region capture
  the observed corner mechanisms?
- **A-Q2:** How much capture is lost by `run0`, and how do the square-root
  and `j<=1` margins compare in capture and width?
- **A-Q3:** Does the predicted right-channel imbalance direction appear?
- **A-Q4:** What is the `alpha2=alpha` versus `alpha/2` frontier?
- **A-Q5:** Do AUC, `m_q`, and curvature summaries explain residuals?
  This is diagnosis, not rule fitting.

Retain the previously designed sources:

1. about 40 mechanically selected replay cells, 200 replicates, with the
   original-seed three-combined-SE parity gate;
2. the 24-cell achievable imbalance LHS, 200 replicates, with both
   orientations in each coarse AUC band; and
3. four balanced high-AUC extent-stress cells at `n=8000,12000`, 200
   replicates.

All are one analysis partition. The archived `n=500` sliver runs informed
the theory and are labeled development evidence; they are not counted as
prospective Stage F validation.

Study A can falsify the proposed margin or show it is too expensive. It
cannot tune `frontier_floor_v1` and pass the tuned version to B/C under
the same name.

---

## 5. Study B — prospective external behavior and sliver stress

Freeze Study B before Study A outcomes. Retain the original 24 cells:

- 10 student-t wedge/traversal cells;
- 6 mechanism-diverse safe cells;
- 4 imbalance cells in both orientations;
- 2 large-n high-AUC cells; and
- 2 regression cells, including `Q=20` ties and a held-out-library shape.

Add a six-cell **fresh sliver confirmation block**. Its exact continuous
DGP formulas, AUCs, `(n0,n1)`, `d=n1*pi`, gap widths, names, and
predicted saturation probabilities are defined before Study A. It must:

- use new names and seed streams, not the 100-replicate development runs;
- include AUC .60, .80, and .95;
- include at least two sample-size scales to test the predicted
  n-independence at fixed `d`; and
- include both imbalance orientations with sliver mass parameterized as
  `pi=d/n1`.

The implemented unequal-size construction preserves the theory's wide
zero-mass stretch. For `s0=1/n0`, `pi=d/n1`, `h=1-pi`, `c=pi/s0`, and
the stated tail extent `s1`, use

```text
R(t) = h Phi(mu + Phi^-1(t/(1-s1))),   0 <= t <= 1-s1,
R(t) = h,                              1-s1 < t < 1-s0,
R(t) = 1-c(1-t),                       1-s0 <= t <= 1,
A_tail = (s1-s0)h + s0(1-pi/2),
A_body = (AUC-A_tail)/((1-s1)h),
mu = sqrt(2) Phi^-1(A_body).
```

This parameterization keeps sliver width `s0`, sliver mass `pi`, expected
sampled count `d`, and total AUC distinct under imbalance. The cells
are:

| name suffix | AUC | n0 | n1 | d | s1 | predicted no-sliver probability |
|---|---:|---:|---:|---:|---:|---:|
| `24--n250x250` | .60 | 250 | 250 | 1.0 | .12 | `(1-1/250)^250` |
| `25--n2000x2000` | .60 | 2000 | 2000 | 1.0 | .12 | `(1-1/2000)^2000` |
| `26--n250x250` | .80 | 250 | 250 | .8 | .25 | `(1-.8/250)^250` |
| `27--n2000x2000` | .95 | 2000 | 2000 | .8 | .25 | `(1-.8/2000)^2000` |
| `28--n2000x500` | .80 | 2000 | 500 | .8 | .25 | `(1-.8/500)^500` |
| `29--n500x2000` | .80 | 500 | 2000 | .8 | .25 | `(1-.8/2000)^2000` |

The sliver block is a theory stress test, not evidence that arbitrary
continuous ROCs are represented by six cells. Its central prediction is
that `C = 1` failure tracks the unsampled-sliver event, the adaptive
frontier region expands on those same realizations, and M3/floored bands
cover.

Run 400 replicates initially. At `alpha=.05`, top up a cell to 1,200 while
the Wilson interval of any prespecified floor-only hybrid straddles .94.
The .5 arm is not topped up solely for a .05 reporting convention.

For every failure, classify inside-region, one-grid-step edge escape, and
far escape. Report the legacy closure's propagated lower misses separately.
For sliver cells additionally report coverage conditional on whether the
sliver was sampled and conditional on `K`/margin bins.

---

## 6. Study C — geometry-class transfer

Retain a 14-cell budget: seven predetermined non-student-t shapes, each at a
smaller and larger sample size. The five named families remain Weibull,
gamma, beta-opposing, high-separation bimodal-negative, and
heteroscedastic Gaussian, plus two fixed-seed mapper draws.

Their role is:

- remove the t-family `m_50` “inside-window/control” labels;
- match AUC within each size pair as closely as feasible;
- precompute, without coverage outcomes, the sign of corner curvature on
  the Lemma 13 intervals and label each shape `corner-concave`,
  `corner-convex`, or `ambiguous`; and
- preserve both `n=500` and `n=8000` scales unless feasibility requires
  a documented change.

Study C asks:

1. whether the frontier floor's repair and price transfer;
2. whether corner-concave cells avoid the two end-gap failure channels, as
   Corollary 13.1 predicts at leading order; and
3. whether residuals are inside, edge, far, or an interior mechanism.

The concavity result is reported as a class-relative theory check. Passing
these cells does not promote Corollary 13.1 to a finite-sample theorem or
create a data-driven class test.

Use the Study B arms except the composite piggyback, with the same 400 to
1,200 top-up rule.

---

## 7. Implementation and tests

**Implementation status (2026-09-02): complete; no Stage F run has been
started.** The `stage_f_*.py` files now:

1. evaluate `frontier_floor_v1` directly from `(n0, n1, M, khat)`, with no
   fitted router or separate rule artifact;
2. provide the `run0`, `j1`, and square-root-margin comparisons with full
   flat preimages;
3. include the unequal-size sliver cells and geometry-labeled Study C; and
4. retain the useful execution machinery: one shared cloud per replicate,
   deterministic seeds, compact lossless records, atomic checkpoints,
   offline scoring, and resumability.

Manifests are ordinary readable design snapshots. They can be regenerated
while the study is exploratory. The runner only checks that a checkpoint
belongs to the same cell and has a contiguous replicate sequence; the
summary checks that the requested study is complete. There are no artifact
hashes, Git fingerprints, or compatibility ledgers.

Focused tests cover containment, closure, encoding, tie-sharing, offline
reconstruction, and checkpoint correctness, plus:

- exact budget-derived `k_left`, `K`, and square-root-margin endpoint
  inclusion;
- invariance of `frontier_floor_v1` to every AUC field and true metadata;
- flat `j=0` and `j<=1` preimages;
- balanced and imbalanced sliver construction at the requested numerical
  AUC;
- conditional expansion of the right region on unsampled-sliver rank
  paths; and
- equality between direct and offline frontier-region scoring.

Before a simulation, generate the manifests and inspect the dry-run cell
list and budgets. If the design changes after results exist, rerun the
affected cells instead of appending to them.

---

## 8. Budget and order

| study | cells | reps | fiducial clouds/rep | estimated CPU-h |
|---|---:|---:|---:|---:|
| A: replay corpus | ~40 | 200 | 1 | 2–3 |
| A: imbalance LHS | 24 | 200 | 1 | 1.5–2 |
| A: extent stress | 4 | 200 | 1 | 1.5 |
| B: ordinary external cells | 24 | 400–1200 | 1 | 3–5 |
| B: fresh sliver block | 6 | 400–1200 | 1 | ~1 |
| C: geometry-class transfer | 14 | 400–1200 | 1 | 2–3 |
| **total** | **~112** |  |  | **~11–16** |

Order:

1. revise and test the implementation;
2. generate and inspect the A/B/C manifests;
3. inspect dry-run cost;
4. run A, B, and C; B/C may run in parallel and never update the rule;
5. write the report before any successor design.

Because the rule is no longer learned from A, B/C need not wait for an A
fit. Running A first is still operationally useful for parity and storage
checks, not a condition of external validity.

---

## 9. Deliverables

1. The rule definition in source/spec and the A/B/C design manifests.
2. Lossless paired records and per-study summaries.
3. `stats/hybrid_floor_report.md`, including margin capture/price,
   alpha2 frontier, sliver conditional results, imbalance, residual
   classification, and geometry-class transfer.
4. A theory amendment that clearly separates exact domination/regional-cap
   statements, the sketch-level frontier argument, and empirical exterior
   control.
5. A roadmap update that retires the shape-blind AUC router as a
   distribution-free candidate and keeps any declared-class router
   separate.

---

## 10. Risks and planned interpretations

| risk | response |
|---|---|
| Square-root margin misses exterior violations | Report edge/far geometry; any enlarged successor is a new version requiring new confirmation data |
| Square-root margin costs too much | Compare `run0` and `j1`; do not tune the primary rule on B/C |
| Sliver failure persists outside the floor | Treat as a falsification of the proposed frontier trigger, not as an AUC-surface fitting problem |
| Corner-concave cells fail | Downgrade Corollary 13.1 and investigate interior versus finite-grid causes; do not invent a class test |
| AUC or `m_q` predicts residuals | Report class-relative diagnostic value only |
| M3 floor misses inside its region | Trigger implementation/parity debugging before scientific interpretation |
| Composite piggyback under-covers | Quarantine it to the finite-range width candidate |
| Sparse miss encoding overflows | Use packed-bitset fallback; never truncate |
| Large-n wedge or sliver persists | This is compatible with Proposition 14 and strengthens the case for a rank-adaptive floor |
