# Complement trimming: results

*Report, 2026-09-27. Spec: [`complement_trim_spec.md`](complement_trim_spec.md).
Runner and analysis: `scripts/complement_trim/` (`run.py`, `analyze.py`).
Data: `data/results/complement_trim_20260926/` (`cells/` records,
`cells.csv` per-cell table, `reps.feather` per-replicate scores,
`added_misses.csv` and `all_miss_runs.csv` miss locations). All results are
**[Measured]** on this design unless labelled otherwise. The design is
deliberately adversarial, so macro averages describe these 50 cells, not ROC
curves in general.*

## 1. Motivation

The hybrid band takes the C = 1 fiducial band and replaces its two ends with
the hull of that band and exact M3 (the floor). The trim depth `j` is still
computed over the whole grid, including the floored ends. The end columns
take 4–8× their share of the draws' minimum depths (research record §7.1),
so the interior may be paying for protection the floor already provides.
The complement arm (`comp`) computes `j` only on the columns the floor does
not protect. The region is fixed from the full-grid depth, and
`j_comp = max(j_full, j_raw)` is enforced. The questions were how much width
this saves, what it costs in coverage, and where that cost lands. The two
arms differ in their guarantees. `hybrid` contains the raw C = 1 band.
`comp` contains neither the raw band nor the hybrid, and nothing bounds its
misses outside the region.

Arms (spec §4): `hybrid` (production), `comp` (complement trim at α, same
cloud), `comp_budget` (complement trim and floor at α/2), `m3` (full M3 at α).
There are two references: `raw` (the C = 1 parent) and `hybrid_half` (the
full-grid hybrid at α/2, the parent of `comp_budget`).

## 2. What was run

- 50 cells: 30 Stage F Study B, 14 Study C, and 6 new interior-feature cells.
  Seeds are fresh (study seed 20260926). Nominal α ∈ {.05, .5}.
- 22,200 paired replicates per α. Top-ups were triggered in 11 cells, at the
  .94 bar for `comp` or `comp_budget`: `imbalance-18`, `hetero_gaussian_small`,
  and `interior_jump` 500×500 went to 800 reps; `wedge-09`, `large_n-21`, and six
  of the seven n = 8,000 Study C cells went to 600. The run took about 5 h
  wall (6 workers × 4 threads), against a dry-run estimate of 21 h.
- One code fix before the run: `summarize` wrote `report.md` in the platform
  default encoding and crashed on Windows at the `→` character. It now writes
  UTF-8. The spec's tests all pass.
- The analysis script rebuilds each replicate's floor region from the stored
  `j_full`. It reproduced the stored region fraction exactly on every checked
  replicate.

**Implementation checks (spec Q6).** The depth maximum bound **0 times** in
44,400 level builds at each trim level, so `j_raw ≥ j_full` held throughout.
A fallback (the floor covering the grid) occurred only on `wedge-00` 250×250
(9.8% of reps at α = .05, 7.5% at α = .5) and `wedge-01` 500×500 (0.5% / 0.2%).
`comp` is nested in `hybrid` in every replicate: 0 replicates where `comp`
covers and `hybrid` does not, at either α. The same holds for `comp_budget`
inside `hybrid_half`.

## 3. Results

### 3.1 Headline (macro over 50 cells, equal weight)

| α | raw | hybrid | comp | comp_budget | hybrid_half | m3 | min comp | comp/hybrid width | comp_budget/hybrid | comp_budget/hybrid_half | hybrid_half/hybrid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| .05 | .887 | .980 | .976 | .988 | .991 | .999 | .948 | .981 | 1.054 | .983 | 1.072 |
| .50 | .589 | .729 | .680 | .859 | .885 | .974 | .550 | .965 | 1.152 | .972 | 1.185 |

Width ratios are means of paired per-replicate area ratios. Every per-cell SE
is ≤ .001.

Paired flips, `hybrid` covers and `comp` misses: **92 of 22,200** replicates
at α = .05 (0.41 pp), and **1,056 of 22,200** at α = .5 (4.8 pp).

### 3.2 Width (spec Q1)

`comp`/`hybrid` is tightly clustered: **.964–.990 at α = .05** and **.938–.979
at α = .5**. Every cell gains something, and no cell gains much.

| n0 band | cells | j_comp / j_full (α = .05) | out-of-region width ratio | whole-band comp/hybrid | coverage loss (pp) |
|---|---|---|---|---|---|
| ≤ 800 | 20 | 1.47 | .965 | .977 | 0.5 |
| 1,000–3,000 | 17 | 1.25 | .979 | .982 | 0.4 |
| ≥ 5,000 | 13 | 1.17 | .985 | .986 | 0.3 |

At α = .5 the same rows are .959, .967, and .973 for width, and 5.6, 4.5, and
4.0 pp for coverage loss.

- **The gain shrinks with n, the opposite of the recorded prediction.** The
  prediction was .95–.98 at n ≤ 1,000 and lower at large n. The mechanism is
  visible in the depths. At large n the floor region is a tiny fraction of the
  grid (0.1–1% on most n = 8,000 cells), so excluding it raises `j` by only
  15–20%. The correlation of `j_comp/j_full` with log n0 is −.60 at α = .05.
  Across cells, the `j` ratio predicts the out-of-region width ratio well
  (r = −.84 at both α).
- **α = .5 gains more than α = .05, but only by about 1.6 pp of width** (.965
  against .981). At α = .5, M is the 2,000-draw floor at both levels, so `j`
  is well resolved (20–200).
- **The largest `j` moves bought little width.** On `wedge-00`, `j` went from
  14.7 to 33.8 at α = .05 and width fell only 1.5%, because 33% of its grid is
  floored and the floor is untouched. The largest whole-band gains (.964–.971)
  were at small n on wedge, imbalance, and `safe-12` cells.
- **`comp` is narrower than the unfloored raw band on 15 cells at α = .05**
  (21 at α = .5), and covers at least as often as raw on 11 of those 15. These
  are mostly cells with a short saturated run, where the floor adds almost no
  width (`hybrid`/`raw` ≈ 1.00–1.02: all large-n B cells, `reg-23`, the
  interior cells, and several slivers). On them the complement trim removes
  more width than the floor adds.

### 3.3 Coverage (spec Q2)

**α = .05.** No cell's `comp` coverage falls below .948. The per-cell loss
against `hybrid` is 0–1.2 pp.

| Lowest `comp` cells, α = .05 | reps | raw | hybrid | comp (Wilson 95%) | comp_budget |
|---|---|---|---|---|---|
| `weibull_large-13` 8,000 | 600 | .953 | .955 | .948 (.928, .963) | .973 |
| `large_n-21` 12,000 | 600 | .942 | .962 | .958 (.939, .972) | .980 |
| `student_t_large-11` 8,000 | 600 | .952 | .962 | .958 (.939, .972) | .982 |
| `wedge-09` 5,131 | 600 | .950 | .965 | .958 (.939, .972) | .983 |
| `gamma_large-03` 8,000 | 600 | .962 | .962 | .960 (.941, .973) | .970 |

The cells nearest the bar are the large-n cells, where `hybrid` itself sits at
.955–.965 and `comp` is 0–0.7 pp below it. The cells with the largest
α = .05 loss are at moderate n: `bimodal_negative_small` .995 → .982,
`wedge-03` .988 → .978, `reg-22` .980 → .970, `safe-12` .990 → .982, and
`weibull_small-12` .988 → .980. The loss is spread across blocks, with no
concentration in wedge, sliver, interior, or imbalance cells. Macro losses by
block and size are 0.2–0.6 pp.

`comp` does not dominate `raw`. At α = .05 there are 77 replicates where `raw`
covers and `comp` does not, against 1,908 the other way.

**α = .5.** `comp` moves macro coverage from .729 to .680, toward .50, closing
about a fifth of the gap. The per-cell range is .550–.930. By size:
- small: .788 → .736
- mid: .721 → .675
- large: .643 → .589 (B) and .651 → .619 (C)

The largest single drops are on `imbalance-16` 300×1,500 (.722 → .630),
`wedge-02` (.738 → .648), `wedge-08` (.830 → .742), and `safe-12`
(.822 → .750). `comp` coverage is close to `raw` coverage at α = .5 on many
cells, because the width saved roughly offsets what the floor added.
Replicate by replicate the two differ substantially: 789 raw-only against
2,735 comp-only.

A rough comparison, which interpolates between only two levels: moving
`hybrid` from α to α/2 buys about 0.15 pp of α = .05 coverage per 1% of width
(.84 pp at α = .5). Moving `hybrid` to `comp` gives up about 0.21 pp per 1%
(1.4 pp at α = .5). At α = .5, the complement trim costs coverage somewhat
faster per unit width than a level change does.

### 3.4 Where the added misses land (spec Q3)

The 92 (α = .05) and 1,056 (α = .5) added-miss replicates are compared below
with all of `hybrid`'s own misses (467 and 6,102 replicates).

| | hybrid misses, α = .05 | comp added, α = .05 | hybrid misses, α = .5 | comp added, α = .5 |
|---|---|---|---|---|
| Violating columns inside the region | 1 run | **0** | 79 runs | **0** |
| Replicates with a lower-edge run | 44% | 43% | 47% | 47% |
| Replicates with a run ≤ 1 / ≤ 5 / ≤ 10 columns from the region | 2.8 / 8.1 / 13.1% | 5.4 / 16.3 / 21.7% | 10.3 / 21.5 / 29.2% | 6.0 / 16.9 / 25.2% |
| Median distance to region, as a fraction of n0 | .074 | .071 | .032 | .056 |
| Lower-edge runs within 3 columns past the left cut | 1.4% | 5.4% | 3.1% | 7.0% |
| Run midpoints at FPR < .02 / .02–.1 / .1–.3 / .3–.5 / .5–.9 / > .9 | 8 / 20 / 26 / 15 / 23 / 8% | 5 / 23 / 25 / 12 / 23 / 12% | 12 / 20 / 23 / 15 / 21 / 10% | 13 / 24 / 19 / 16 / 19 / 9% |

- **None of the new misses is inside the region.** They are split evenly
  between the lower and upper edges, and they are spread along the curve in
  about the same proportions as `hybrid`'s own misses.
- **At α = .05 there is a modest excess close to the region boundary.**
  15 of the 92 added replicates have a run within 5 columns of the region,
  against 8% of hybrid's. These are small counts. Of those 15, 5 are
  lower-edge runs just past the left cut (1–3 columns): `wedge-01` (×2),
  `wedge-03`, `safe-14`, and `hetero_gaussian_small`. The rest are upper-edge
  runs near the right region, or lower-edge runs 3–5 columns in. So in
  22,200 replicates, about 5 are the specific pattern spec §8 Q3 named as a
  sign the `j_full` region is too short for the deeper band.
- **At α = .5 the added misses are, if anything, farther from the region than
  hybrid's.** Hybrid's α = .5 misses cluster near the boundary more than its
  α = .05 misses do.

### 3.5 Budget split (spec Q4)

`comp_budget` does not recover `hybrid`'s width. It is **wider than `hybrid` in
every cell**: 1.036–1.073 at α = .05 and 1.138–1.213 at α = .5. It covers more
than `hybrid` on average (.988 against .980 at α = .05). It falls below
`hybrid`'s coverage in 5 replicates at α = .05 and covers where `hybrid`
misses in 199.

Decomposition, as macro ratios:

| α | Halving the budget (hybrid_half/hybrid) | Domain effect at α/2 (comp_budget/hybrid_half) | Net (comp_budget/hybrid) |
|---|---|---|---|
| .05 | 1.072 | .983 | 1.054 |
| .5 | 1.185 | .972 | 1.152 |

The domain saves about as much at α/2 as at α (.983 against .981). The
halved budget costs about four times as much as the domain saves. This
matches the recorded prediction.

`comp_budget` loses coverage against its own parent in 57 of 22,200
replicates at α = .05. Its minimum cell is .970, on `gamma_large-03`, where
`hybrid_half` is .972. At α = .5 its coverage (.793–.962) is near
`hybrid_half`'s, far above nominal, because its trim level is α/2 = .25.

### 3.6 Interior features (spec Q5)

The interior-feature cells did not stress any arm.

- `raw` covers .968–.982 at α = .05 on all six cells. `hybrid` adds 0–0.4 pp
  and `comp` subtracts 0–0.7 pp from `hybrid`, the same as on ordinary cells.
- `interior_jump` (h = .2) is sampled in every replicate. `interior_sliver`
  is sampled in 56% of replicates. Conditional on the feature being sampled,
  `comp` at α = .05 covers .960 (500) and .969 (2,000), against `hybrid`'s
  .969 and .973.
- When the sliver is unsampled, `comp` covers .971 and .966, against
  `hybrid`'s .977 and .977.

Neither construction produces the interior-specific failure the block was
designed to probe. At these sizes the C = 1 cloud absorbs a sub-grid jump of
height .2, and a 1/n0-wide sliver of mass .8/n1, without a visible coverage
cost. So this block measures `comp` on benign interior structure. It does not
bound its risk on harsher interior structure.

For contrast, the corner slivers (Study B `sliver_fresh`) behave as before.
`raw` falls to 0 on unsampled-sliver replicates (0.10 on `sliver_fresh-24`),
and `hybrid` and `comp` both repair it (.955–.994 at α = .05, sampled or not). `comp` is within
0.7 pp of `hybrid` in every sliver cell, sampled or not.

## 4. Inferences

- **The complement trim is a small, uniform width gain, and it shrinks with n**
  **[Measured]**. That is 1–4% at α = .05 and 2–6% at α = .5, largest at
  small n and smallest at n ≥ 5,000. This extends §7.1's n = 500 diagnostic
  (2–5% interior width at α = .05) to the production floor region. The
  exploration's gain of up to 8% at n = 5,000 does not reappear, because that window
  excluded FPR < .02 and > .95 wholesale, while the exact floor region at
  large n is a handful of columns. **[Inferred]** The quantity that governs
  the gain is how much of the grid the floor covers, and that falls with n.
- **Its coverage cost at α = .05 is small, and it is mostly not
  boundary-located** **[Measured]**. It is 0.4 pp macro, at most 1.2 pp per
  cell, with a minimum cell of .948. The added misses look like `hybrid`'s own
  interior misses in direction and location. There is a slight excess of
  runs adjacent to the region, including about 5 replicates of lower-edge
  misses just past the left cut. That count is too small to confirm or rule
  out the §3.1 concern.
- **The cells nearest .95 are large-n cells, where `hybrid` is already
  .955–.965** **[Measured]**. There, `comp` takes 0–0.7 pp more. This is the
  regime where §7.6 attributes the residual to C = 1's finite-n surplus
  wearing off. The complement trim removes part of what is left of it.
- **At α = .5 it moves calibration in the right direction but barely**
  **[Measured]**. It closes about a fifth of the hybrid's gap to nominal
  (.729 → .680 against .50). Per unit width it costs coverage somewhat faster
  than a level change (a rough two-level comparison). The central-α
  over-coverage in §7.5 is essentially untouched.
- **The declared α/2 + α/2 split is strictly wider than production**
  **[Measured]**. It is +5.4% at α = .05 and +15% at α = .5, because halving
  the budget costs about 4× what the domain saves.
- **The depth monotonicity in spec §3.1 held in every build** **[Measured, 0
  of 88,800 builds]**, as the argument requires.

## 5. Caveats

- The interior-feature block turned out not to be adversarial even to `raw`,
  so this run gives no evidence on `comp` under interior structure that
  actually hurts C = 1.
- Large-n cells have 200–600 reps, so their Wilson intervals are about
  ±2 pp. The per-cell ranking near .95 is not resolved.
- The top-up rule targeted `comp` and `comp_budget` intervals that straddled
  .94. That adds replicates preferentially where coverage is near the bar.
  It does not bias the estimates, but it makes precision uneven across cells.
- The added-miss location statistics at α = .05 rest on 92 replicates, and
  the near-boundary counts on 5–15 of them.
