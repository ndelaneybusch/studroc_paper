# Tightened floor closure and the sample-size M3 split: results

*Report, 2026-10-31. Spec: [`closure_split_spec.md`](closure_split_spec.md).
Runner and analysis: `scripts/closure_split/`. Data:
`data/results/closure_split_20261027/` (`cells/` records, `t_reps.feather`
per-replicate scores, `t_cells.csv` and `s_cells.csv` per-cell tables,
`report.md` generated tables, `summary.json`). Results are **[Measured]** on
this design unless labelled **[Exact]** (follows from a proved property) or
**[Interpretation]**. The typical block samples the paper's design space; the
favored and red blocks are built to show the best and worst cases, and their
averages describe those cells, not ROC curves in general.*

## 1. What the study asked

The paper's methods are fixed after this study. It was set up to inform two
choices, and it makes neither.

**Closure.** The hybrid band takes the C = 1 fiducial band, replaces its two
ends (the floor region) with its hull with exact M3, and then has to restore
monotonicity. Production does this by *widening*. That carries the floor's
extra room out of the region into the interior next to it. The alternative
is *tightening*. Theory §10.7 (Prop. 10e) shows the tightened band covers a
monotone truth exactly when the unclosed hull does, is never wider, and keeps
every exact statement the hybrid has. What it gives up is exactly the
replicates where widening's carried room was what saved coverage. Nothing
bounds that term, so its size had to be measured.

**Split.** M3 combines one band per class, and has to divide its error
budget between them. Production splits evenly (Šidák, ρ = .5). The
alternative, `sqrt` (`split_ratio="sample_size"`), gives the smaller class
more of the budget, ρ = n0^−½/(n0^−½ + n1^−½). Both are exact, because ρ
depends only on class sizes. The questions were how much width it saves,
where it loses, and whether it matters inside the hybrid's floor.

The two closure options are applied to two parents:
- `hyb`: the production hybrid, with trim depth `j` from the whole grid.
- `rec`: the tails-recalibrated hybrid, with depth `j_rec ≥ j` recomputed on
  the columns outside the region. It was the preferred variant after the
  complement-trim study.

The suffix gives the closure (`_W` widen, `_T` tighten). `raw` is the C = 1
parent without a floor. All arms in a replicate share one fiducial cloud, so
every comparison between them is paired.

## 2. What was run

- **Study T (closure): 73 cells, α ∈ {.05, .5}.**
  - 46 typical cells, drawn with a frozen seed from the main suite's design
    space: six families, sizes 50², 150², 500², 900×100 and 5,000².
  - 6 favored cells with long saturated runs, including two anchors from the
    earlier probe (wedge-01 and imbalance-17).
  - 21 red-team cells in 10 groups (R1–R10). Each puts a realistic feature in
    the strip next to the floor region.
  - Replication was 1,000 / 750 / 500 per cell by n0, giving 68,250 paired
    replicates per α. No cell triggered a top-up: the lowest `rec_T`
    coverage at α = .05 was .960, at the bar but not below it. Wall time was
    93 min on 20 workers.
- **Study S (split): 156 cells, α ∈ {.05, .2, .5}.** That is 12 shapes × 13
  class-size configurations, from balanced 500² to 1:100 and 100:1, plus a
  small 20×180 configuration. Each cell ran 1,000 replicates. Rules compared:
  Šidák, `sqrt`, `linear` (ρ ∝ 1/n), and a diagnostic sweep ρ ∈ {.1, …, .9}.
  The best ρ per cell after the fact (the "oracle") shows how much headroom
  is left; it is not a usable rule. Wall time was 36 min.
- **Changes after the spec was first written, all before any result was
  seen:**
  - The decision thresholds were removed; the study reports contrasts.
  - A C = 1.5 level-change comparator was dropped.
  - Study T's replication was raised twice: first by 1.25×, using the compute
    freed by the comparator, and then doubled.
- **Implementation checks [Measured, all clean].**
  - No nesting violation in any replicate.
  - No replicate where a tightened band's coverage differs from its unclosed
    hull's, so Prop. 10e held.
  - No replicate where a tightened arm covers and its widened counterpart
    misses.
  - The recalibrated trim fell back to the plain trim in 338 replicates
    across 9 cells (16 cell×α pairs). In every one of them the floor covered
    the whole grid, so the band was M3 and `rec` = `hyb` identically. Most
    were high-AUC cells at 50² and 150²; the largest share was 16% of
    replicates (bimodal .94 at 50², α = .05).

---

## 3. Tightening

### 3.1 Where we stood before the run

- **The probe.** A scratch probe on 18 Stage F cells found tightening saved
  about 4% width at α = .05 and lost 7 of 3,600 replicates. But those were
  the cells the floor was designed on, mostly adversarial, and the probe was
  not prospective.
- **What was not known:**
  - whether the gain would persist on cells like the paper's;
  - whether the unbounded term (misses in the strip next to the region) is
    small in general, or only on the probe's cells;
  - whether realistic features placed in that strip would make it large;
  - whether tightening would push any cell below nominal.

The prediction recorded before the run was a typical gain of 2–4% and a loss
of at most 0.2 pp.

### 3.2 Headline

Macro over cells (equal weight within block). Width ratios are paired
per-replicate area ratios, and "lost" is the paired rate of replicates the
widened arm covers and the tightened arm misses. Per-cell width SEs are
around .001 or below.

| α | Block | `hyb_W` | `hyb_T` | `rec_W` | `rec_T` (min cell) | `rec_T/rec_W` width | `hyb_T/hyb_W` width | `rec_T/hyb_W` width | lost, rec / hyb |
|---|---|---|---|---|---|---|---|---|---|
| .05 | typical (46) | .986 | .985 | .980 | .978 (.960) | .944 | .957 | .926 | 0.23 / 0.12 pp |
| .05 | favored (6) | .988 | .986 | .981 | .978 (.976) | .930 | .946 | .910 | 0.33 / 0.15 pp |
| .05 | red (21) | .981 | .979 | .976 | .973 (.964) | .975 | .981 | .954 | 0.24 / 0.17 pp |
| .5 | typical | .782 | .766 | .716 | .686 (.597) | .925 | .945 | .890 | 3.0 / 1.6 pp |
| .5 | favored | .784 | .770 | .713 | .682 (.651) | .905 | .927 | .868 | 3.0 / 1.5 pp |
| .5 | red | .729 | .719 | .672 | .656 (.580) | .966 | .975 | .928 | 1.7 / 1.1 pp |

On typical cells, `rec_T` is .73 the width of M3 at α = .05 (.68–.75 across
sizes) and .63 at α = .5. Against fixed-width KS it is .52 at α = .05,
ranging from .67 at 50² to .37 at 5,000².

### 3.3 Coverage at α = .05: nominal holds everywhere tested

- **Every cell stays above nominal.** No arm has any cell with a point
  estimate below .95, or a Wilson upper bound below .95, in any block. The
  lowest `rec_T` cell is .960 (hetero-Gaussian .97 at 5,000², interval
  [.939, .974]). The lowest Wilson upper bound for `rec_T` is .971.
- **The low cells were already low before tightening.** In the ten lowest
  cells, tightening's own loss is 0–6 replicates per 1,000 (or per 750 or
  500 at large n). In most of them, recalibration costs more than
  tightening. The three lowest are large-n cells (two at 5,000², R1 at
  2,000²) where raw C = 1 is already at .960–.966 and tightening took
  nothing. This is the same large-n regime the complement-trim study
  identified, where C = 1's finite-sample surplus has worn off.
- **The shift is small and even across the distribution.** On the typical
  block, the cell distribution moves down by 0.2–0.3 pp at every quartile
  from `rec_W` to `rec_T`. The worst single cell loses 1.0 pp
  (hetero-Gaussian .99 at 150², 10 of 1,000).
- **Losses are concentrated at small n.** By size, the typical loss is
  0.42 pp at 50², 0.31 pp at 150², and 0.19 pp or less at every other
  size. 10 of the 46 typical cells lost no replicates at all.

### 3.4 Width: how much, where, and what governs it

- **The gain is larger than predicted on typical cells.** `rec_T/rec_W` is
  .944 against a predicted .96–.98, so close to 6% rather than 2–4%. It is
  .930 on the favored block, inside the predicted .85–.93.
- **It follows the size of the floor region.** The Spearman correlation
  with region fraction is .81. By region fraction (α = .05):

  | Region fraction | under 5% | 5–15% | 15–30% | 30–50% | over 50% |
  |---|---|---|---|---|---|
  | Gain | 0.5% | 1.9% | 4.6% | 9.0% | 8.6% |

  By size, the gain is 11% at 50², 7% at 150², 6% at 900×100, 3% at 500²
  and 0.4% at 5,000². The spec predicted 900×100 would gain the most of the
  typical sizes, but the two small sizes beat it. Region fraction falls with
  n, so size is a proxy for it.
- **Almost half the saving is inside the region, where it costs nothing
  [Measured; Exact that it costs no coverage].** On typical cells at
  α = .05, the width removed by tightening splits 46% in the region (36% in
  the right region, 10% in the left), 37% in the adjacent strip, and 17% in
  the far interior. The in-region part comes mostly from the right floor:
  the lower edge's forward cummax lifts M3's dip up to the interior's lower
  edge. Because tightened coverage is identical to the hull's, narrowing
  inside the region cannot create a miss. On the favored cells, and
  especially on the two probe anchors, 46% (75–82% for the anchors) of the
  saving is in the far interior. On long-run cells, widening carries room
  far past the near strip (10 grid points or 2% of n0).
- **The probe's anchor gains were somewhat optimistic.** Wedge-01 came out
  at .871 against the probe's .847, and imbalance-17 at .915 against .894.
  The typical block, which the probe did not cover, gained more than the
  probe's average (5.6% against 4.0%).

### 3.5 How tightening's misses happen

There were 170 added misses ("flips") in 68,250 replicates at α = .05, and
1,888 at α = .5.

- **The flips sit at the region boundary [Measured].** In every block and at
  both α, 67–78% of flips have their nearest miss within the near strip,
  against 8–22% for the widened arm's ordinary misses. Within two grid points
  of the boundary, the figures are 45–63% of flips and 0–3% of ordinary
  misses. The median distance is 1–3 grid points for flips and 80–250 for
  ordinary misses. This is the mechanism §10.7 describes, and the spread is
  narrow.
- **The dominant site is the upper edge just right of the left region.** By
  miss run, upper-edge-near-left is the most common site (748 runs). The
  next are lower edge in the right region (425), lower edge next to the
  right region (358) and lower edge far (355). The spec's account had
  emphasised the lower edge left of the right region. Both sites occur, but
  the left cut matters more than expected. At α = .05, 59% of typical flips
  are upper-edge.
- **Flips are not reliably shallower at α = .05.** The median depth of a flip
  is about the same as an ordinary miss: .0033 against .0035 on typical
  cells, and .0019 against .0018 on red. Within cells, the median ratio is
  about 1.4, but on small counts. At α = .5 flips are shallower (within-cell
  ratio about 0.8). At both levels the depths are fractions of a percent of
  TPR.
- **Flips track the region more than the run length.** Replicates that flip
  have larger regions than their cell's average, by about half a within-cell
  SD at α = .05. They also have somewhat longer saturated runs. Flip rates
  are elevated mainly in the top region-fraction bin (over .4). Run length
  alone shows no clean trend.
- **"In region" flips are expected, and none occur alone [Exact, confirmed].**
  Tightening narrows the band inside the region, so a tightened miss can show
  up there. By monotonicity, any such miss implies the hull also misses at an
  interior point. All 2,058 flips with an in-region miss also miss outside
  the region. The location shares in `report.md` Q2 add up to more than 1,
  because a single flip can miss in several places (718 of 2,058 do).

### 3.6 Red team

Paired flips from tightening (`rec_T` vs `rec_W`), out of each group's
replicates.

| Group | Mechanism | Reps | `rec_T` cov (.05) | Flips at .05 | Loss at .5 | Width (.05) | Where the flips landed |
|---|---|---|---|---|---|---|---|
| R3 | hard negatives | 2,000 | .976 | 15 (0.75 pp) | 3.9 pp | .917 | scattered: upper edge near and far, lower far |
| R2 | hard positives | 2,750 | .973 | 12 (0.44 pp) | 2.9 pp | .972 | half upper edge near the left cut, half lower edge |
| R1 | convex approach to (1, 1) | 2,750 | .973 | 8 | 2.1 pp | .982 | lower edge left of the right region, mostly beyond the near strip |
| R4 | obvious-positive subgroup | 1,750 | .970 | 7 | 0.7 pp | .990 | upper edge just right of the left cut, as targeted |
| R7 | interior jump, small n | 2,000 | .974 | 3 | 2.6 pp | .934 | both strips at .5, as targeted |
| R5 | separated top | 1,750 | .981 | 1 | 0.4 pp | .986 | upper edge near left, as targeted |
| R6 | scarce positives at scale | 2,250 | .970 | 1 | 0.9 pp | .993 | — |
| R10 | fresh corner sliver | 1,000 | .964 | 1 | 0.8 pp | .991 | — |
| R8 | coarse ties | 1,000 | .976 | 0 | 0.4 pp | .985 | — |
| R9 | large-n residual | 1,000 | .974 | 0 | 0.6 pp | .999 | upper edge, 1 point past the left cut (at .5) |

- **At α = .05 the red team found no mechanism that does real damage.** Every
  red cell stays at or above .964 for `rec_T`. Counts are 0–15 flips per
  group, so no group's loss can be told apart from about 0.5 pp. The largest
  is R3 (0.75 pp, upper bound about 1.2 pp). R3 is also the group where
  tightening saves the most width.
- **The groups predicted to hurt most did not.** R2 is second, but R6 is
  close to inert: one flip in 2,250. The mechanisms that bite at α = .5 are
  R3, R2, R7 and R1. R7's 50² cell loses 4.3 pp there.
- **Targeting.** R4, R5 and R7 put their flips where the spec aimed them.
  R1's feature is wider than the 10-point near strip at 300², so most of its
  flips count as "far" even though they are boundary-driven. On cells like
  this, the near/far split understates how much of the loss is tied to the
  boundary.
- **Whether the feature was sampled makes no difference.** R10 covers the
  same whether or not the sliver was sampled. R4, R5 and R7 always sample
  their feature, so they have no unsampled comparison. A separate R10
  observation that is not about tightening: replicates with two or more
  sliver positives cover worse (`hyb_W` .946, n = 202) than those with at
  most one (.979), and the shortfall is in every arm.

### 3.7 Tightening and recalibration together

The two narrowings stack, and their costs are quite different.

| Typical, macro | α = .05 | α = .5 |
|---|---|---|
| Coverage cost of recalibration (`hyb_W` − `rec_W`) | 0.56 pp | 6.6 pp |
| Coverage cost of tightening after recalibration (`rec_W` − `rec_T`) | 0.23 pp | 3.0 pp |
| Coverage cost of tightening on the plain hybrid (`hyb_W` − `hyb_T`) | 0.12 pp | 1.6 pp |
| pp lost per 1% width: recalibration | 0.30 | 1.8 |
| pp lost per 1% width: tightening after recalibration | 0.04 | 0.4 |

- **Tightening is much cheaper per unit width than recalibration.** At
  α = .05 it costs about a seventh as much coverage per 1% of width. Over
  all blocks, tightening is about 30% of the total `hyb_W` → `rec_T`
  coverage cost (170 of 565 lost replicates), while supplying more of the
  width (5.6% against 1.9% on typical cells).
- **The width ratios multiply almost exactly, and the two interact.**
  Tightening saves more on top of recalibration (5.5% of `hyb_W`'s area)
  than on the plain hybrid (4.3%), and this holds in all 46 typical cells.
  Equivalently, recalibration saves more once the band is tightened.
- **The recalibrated band covers like C = 1 wherever C = 1 is healthy.** On
  the typical block, `rec_W`'s coverage matches `raw`'s: .980 against .981 at
  α = .05 and .716 against .716 at α = .5, with a median cell gap of 0.1 pp.
  Where raw collapses at the corners (the favored anchors, R2, R3 and R10,
  with raw at .54–.89), `rec_W` keeps the floor's repair.
  *[Interpretation]:* recalibrating the trim on the unfloored columns hands
  back the surplus coverage the floor added on healthy cells, and keeps the
  floor where it is needed. Tightening then removes a further 0.2–0.3 pp at
  α = .05 for about 6% width.

### 3.8 α = .5

- **Tightening moves every cell toward nominal.** On the typical block it
  lowers coverage toward .5 in all 46 cells, for both parents. No floored arm
  goes below .5 in any cell, and every `rec_T` Wilson lower bound is above
  .5. Only raw drops below, in one cell (.31, hetero-Gaussian .99 at
  900×100).
- **All arms are still well above nominal.** Typical macro coverage is .686
  for `rec_T`, .716 for `rec_W` and .782 for `hyb_W`. Against production's
  gap to .5, recalibration closes about a quarter, and tightening a further
  tenth. Calibration across α, a desideratum nothing has met, stays unmet.
- **The trade is about ten times steeper than at α = .05.** Tightening costs
  0.4 pp per 1% of width at α = .5, against 0.04 pp at .05. At α = .5 the
  losses are a move toward nominal, not a deficit.
- **The extremes are at small n.** The largest single move is
  hetero-Gaussian .95 at 50², where `rec_W` .87 drops to `rec_T` .68 for
  23% less width.

### 3.9 What changed, and what remains uncertain

**What changed.**
- **On every cell tested, tightening keeps α = .05 coverage above nominal.**
  It costs about 0.2 pp on typical cells, and at most 1 pp in any cell
  (worst red cell 0.9 pp).
- **The width gain is real on the paper's kind of cells, not only on the
  probe's,** and it is larger than predicted: about 6% over `rec_W`, and
  7–8% net of production on typical cells.
- **The unbounded term turned out to be a boundary effect.** Added misses
  sit at the region edge, mainly the upper edge right of the left cut.
- **The ten red-team mechanisms tested do not make it large.**
- **Most of the coverage the `rec_T` package gives up comes from
  recalibration, not tightening.**

**What remains uncertain.**
- **Large n near nominal.** The lowest cells (.960–.966) are large-n cells
  where C = 1 itself is near nominal. Their intervals span about ±2 pp at
  500–750 replicates, so the per-cell ranking there is not resolved.
  Tightening took nothing in them; recalibration took a little.
- **Features in the strip that were not modelled.** The red team covers ten
  mechanisms. It shows that realistic features near the left cut (R3, R4)
  and convex hooks near the right region (R1, R2) cost under 1 pp at
  α = .05. A feature longer than the near strip at small n (as in R1 at
  300²) is where flips spread furthest.
- **Small n.** 50² and 150² carry the largest gains and the largest losses.
  They are also where the fallback (region = whole grid) occurs. Every typical
  cell at these sizes stays at or above .96, but this is the edge of the
  tested domain for losses.
- **Beyond n0 = 9,000.** Not tested. The region shrinks as a fraction of the
  grid, and with it both the gain and the loss, but that is extrapolation.
- **No new theory.** Tightening keeps the hybrid's existing exact statements
  and adds no whole-band guarantee.

---

## 4. The sample-size split

### 4.1 Where we stood before the run

- **Earlier evidence.** The Sep 5 screen found 0.9–5.8% less area on 8
  imbalanced cells, and no change at balance. A later boundary-family search
  vetoed the split on one training shape, by a width ratio of 1.0009 under a
  zero-tolerance rule.
- **The theoretical concern.** The first-order width argument says the rule
  equalises √n scaling but ignores the curve's slope. That suggested it
  would lose with few negatives at high AUC, where most of the curve is flat
  and the positive-class term should dominate.
- **What was not known:**
  - how general the gain is;
  - whether the predicted losing quadrant is real;
  - how large the losses get;
  - whether the rule matters inside the hybrid's floor.

### 4.2 Width

`sqrt`/Šidák macro over 12 shapes. ρ is the negative class's share of the
error budget, and `sqrt` gives the smaller class the larger share.

| Configuration | α = .05 | α = .2 | α = .5 |
|---|---|---|---|
| 100:1 and 1:100 | .94–.95 | .94–.96 | .90–.92 |
| 30:1 and 1:30 | .96–.97 | .96 | .94–.96 |
| 9:1 and 1:9 (incl. small 20×180) | .97–.99 | .97–.99 | .97–.98 |
| 3:1 and 1:3 | .995 | .985–.995 | .99 |
| balanced | 1 (identical) | 1 | 1 |

- **The gain grows with imbalance and with α.** The single best cell is .84
  (bn99 at 100×10,000, α = .5). Per-cell paired SEs are 10⁻⁴ or smaller, so
  every ratio shown is well resolved.
- **The split is as close to free as it gets.** 12 of 468 cell×α pairs are
  wider than Šidák, by at most 0.85% (1.0085, hetero σ ratio 2 at 250×750,
  α = .5). All 12 are statistically real, but small:
  - 20×180 at α = .05: bn99 1.005, t2 1.003, beta 1.001;
  - 100×900 at α = .2: bn99 1.0035, hetero-2 1.002;
  - 250×750: hetero-2 (.05 and .5), bn99 and beta (.5);
  - 750×250: hetero σ ratio .5, at all three α, up to 1.003.

  The losses are high-AUC or heteroscedastic shapes at moderate imbalance or
  very small n0. They are not monotone in α: 100×900 bn99 gives .93, 1.0035
  and .99 at the three levels.

### 4.3 The predicted losing quadrant gains the most

The spec named few negatives at high AUC (bn99, beta .98 and t2 .95 at
100×900, 150×4,500 and 100×10,000) as where `sqrt` would lose.
- **The prediction failed.** Those 27 cells have a macro ratio of .953,
  better than the other shapes at the same sizes (.958). Only one lost
  (100×900 bn99, α = .2). Across imbalanced cells, bn99 ranks first in gain
  and the diagonal last. The spec predicted the opposite ordering.
- **Where the width changed.** At few negatives, almost all the reduction is
  at FPR below .1, with some at .1–.5. The flat part of the curve widens
  slightly: about +.0001 to +.0008.
- *[Interpretation]:* the theory pictured the flat region as dominated by the
  positive-class term. But there TPR ≈ 1, so the positive-class variance is
  near zero, and that region contributes little width under either split.
  The width sits in the steep low-FPR corner, where the negative-class
  (horizontal) term dominates. Giving the scarce negatives more budget
  attacks exactly that corner.
- **Small n0 behaves differently.** At 20×180, the low-FPR bands get wider
  and the oracle is Šidák. *[Interpretation]:* with only 20 negatives, extra
  budget buys no further whole-column step on the horizontal band, while the
  positive band still pays. The mirror configuration (180×20) has no losses
  (worst .989). Absolute n0 appears to matter more than the ratio here.

### 4.4 Headroom and the oracle

- **`sqrt` is essentially at the grid oracle.** Its macro ratio is .972 and
  the oracle's is .970. The median headroom per cell is .0006, and `sqrt`
  matches or beats the oracle in a third of cells. At 1:100 the oracle is
  limited by the grid: its best value is ρ = .9, and `sqrt`'s ρ is .91.
- **Material headroom is confined to α = .5.** Thirteen cells leave more than
  .01, nearly all at α = .5:
  - 100×900, high-AUC shapes: bn99 .087, beta .065, bn95 .027;
  - 150×4,500 bn99 and 180×20 bn99: about .01 each;
  - four balanced 500² shapes: bn99 .027, beta .017, hetero-2 .011.
- **The optimum depends on shape even at balance [Measured].** At 500², the
  best ρ is .6–.7 for high-AUC shapes and hetero σ ratio .5, and .3 for
  hetero σ ratio 2, worth up to 2.7% at α = .5. No rule based on class sizes
  alone can capture this.
- **Width moves in steps at small n0.** The sweep often jumps between
  neighbouring ρ values (at 100×900 bn99, α = .05: .9994 at ρ = .6, .934 at
  .7). *[Interpretation]:* ranks move the horizontal band in whole-column
  steps. That would explain why individual losses and oracle choices flip
  with small changes in α. Read single losing cells as quantisation, not a
  trend.

### 4.5 `linear` (ρ ∝ 1/n)

- **`linear` is better than `sqrt` in only 22% of cells.** Its overall macro
  is .975, against `sqrt`'s .972.
- **Where it wins, it can win big.** Mostly at 100×900, and at 150×4,500 at
  α = .5: up to .10 better (bn99 at 150×4,500, α = .5).
- **It is worse at 1:100, at 20×180 and on the heteroscedastic shapes.** It
  is wider than Šidák in 47 cell×α pairs, reaching 1.027.
- **The spec predicted `linear` would win at 1:100. It loses there.** Neither
  rule dominates. `sqrt`'s worst case is about a third of `linear`'s.

### 4.6 Coverage

- **No exactness concern at α = .05.** Every rule covers .985 or more there.
- **The split lowers coverage toward nominal at the higher α levels.** Mean
  coverage at α = .2 / .5:

  | Rule | α = .2 | α = .5 |
  |---|---|---|
  | Šidák | .992 | .962 |
  | `sqrt` | .988 | .944 |
  | `linear` | .988 | .946 |
  | KS | .994 | .969 |

- **The drop tracks the width gain.** It is largest at 1:100, where it is
  6 pp at α = .5. The within-α correlation between the two is about −.7.
  Misses rise on both edges, slightly more on the upper.
- **M3 stays far above nominal at α = .5 under any rule.** The split is a
  small move toward calibration, not a fix.

### 4.7 Inside the hybrid's floor

- **Standalone, Study T matches Study S.** On Study T's 17 imbalanced cells,
  `m3_ss/m3` is .983 at α = .05 (.968–.991) and .970 at α = .5.
- **Inside the floor, almost all of that disappears.** `rec_T_ss/rec_T` is
  .998 at α = .05 (.990–1.002) and .997 at α = .5. The hybrid keeps a median
  of about 7% of M3's gain, because the floor uses M3 only on the region.
  The retained gain tracks M3's gain × region fraction (r = .59).
- **Small and harmless either way.**
  - Three cell×α pairs are slightly wider (imbalance-17 1.0006 and 1.0021;
    R4 at 2,000×500 1.0004).
  - Coverage changes by at most 2 replicates in 1,000.
  - The prediction of ±1% held.

### 4.8 What changed, and what remains uncertain

**What changed.**
- **The gain is general.** `sqrt` narrows M3 in nearly every imbalanced
  cell: 1–5% at 1:9 to 1:30, and 5–10% at 1:100.
- **The losses are rare and small.** They come to at most 0.85% across 468
  cell×α pairs.
- **The theoretical concern pointed the wrong way.** The predicted losing
  quadrant is where the rule gains most. The earlier veto margin (1.0009) is
  about the size of the losses seen here.
- **`sqrt` is near the best size-only split this design can find.** `linear`
  is less robust.
- **In the hybrid, the choice barely matters.**

**What remains uncertain.**
- **Very small n0.** Losses concentrate at n0 = 20 and at moderate imbalance
  with high AUC. Configurations with n0 < 20 were not tested.
- **Step behaviour.** At small n0, both the losses and the headroom move in
  discrete steps with α and ρ, so individual cells can flip sign.
- **Shape-dependent headroom.** Up to 2.7% at balance and up to 9% at
  100×900 (α = .5) is available in principle to a shape-aware split. That is
  a different kind of rule, and this design only shows that the headroom
  exists.

---

## 5. Connections

- **One quantity governs most of what happened: how much of the grid the
  floor covers.**
  - Tightening's gain and loss both scale with region fraction.
  - Recalibration's leverage does too, which is why tightening saves more on
    top of it.
  - The split's effect inside the hybrid is M3's gain times region fraction.
  - Region fraction falls with n, so every effect shrinks at 5,000² and
    peaks at 50²–150².
- **Small n is where both questions are most active.**
  - The largest closure gains and losses are at 50²–150².
  - The fallback to pure M3 happens there.
  - The split's only systematic losses are at n0 = 20.
  - Typical cells at these sizes still stay at or above .96 at α = .05.
- **Large n near nominal is a separate issue.** The cells closest to .95 at
  α = .05 are large-n cells where C = 1 itself has lost its surplus
  (5,000², R1 at 2,000², R6 at 4,500×500). Neither option studied here moves
  them much. That residual belongs to the interior trim, as the
  complement-trim study also found.
- **At α = .5 everything moves toward nominal, at a much steeper price.**
  Tightening, recalibration and the split all lower α = .5 coverage. None
  overshoots, and none comes close to .5. The price in coverage per unit
  width is about ten times its α = .05 value for tightening. Calibration
  across α is not addressed by any of these options.
- **The coverage cost of the `rec_T` package is mostly recalibration's.** On
  healthy cells, recalibration returns the hybrid to C = 1's own coverage.
  Tightening then costs a further 0.2 pp for about 6% width, and the
  two-step band ends up .89–.93 the width of production on typical cells.
- **Two pre-run accounts about where loss would occur were wrong in the same
  way.** For tightening, the spec emphasised the lower edge left of the right
  region; the upper edge right of the left cut dominates. For the split, the
  theory emphasised the flat high-TPR stretch; the steep low-FPR corner
  dominates. In both cases the low-FPR end of the curve carried more than
  expected.

## 6. Predictions recorded before the run

| Prediction (spec §9) | Observed | Held? |
|---|---|---|
| Typical `rec_T/rec_W` .96–.98 | .944 | No: larger gain |
| Favored .85–.93 | .930 | Yes, at the edge |
| Typical gain smaller than favored | 5.6% vs 7.0% | Yes; small gap |
| 900×100 largest typical gain | 50² (11%) and 150² (7%) exceed 900×100 (6%) | No |
| 5,000² smallest, at most 1% | 0.4% | Yes |
| α = .5 gain about 1.3× α = .05 | 1.3× | Yes |
| Typical lost coverage at most 0.2 pp | 0.23 pp at α = .05 (interval about .19–.28) | Marginally no |
| R2 and R6 largest red losses, lower edge | R3 largest, R2 second, R6 near inert; R2 flips half upper-edge | No |
| R5 upper-edge flips if any | 1 flip, upper edge | Yes |
| R9 moves at most 0.5 pp | 0 at .05; 0.6 pp at .5 | Yes at .05 |
| `sqrt/sidak` .96–.99 at 1:9 and steeper | 62% in range, 32% below .96 | Partly; larger gains |
| Largest split gains on diagonal shapes | Diagonal smallest, bn99 largest | No |
| Split above 1.00 on at least one few-negatives high-AUC cell, at most 1.01 | Yes (20×180; 100×900 at .2); max 1.0085 overall; 1 of 27 named cells lost | Literally yes, mechanism no |
| `linear` better at 1:100 | Worse at 1:100 | No |
| Floor split within ±1% | .990–1.002 | Yes |

## 7. Scope and caveats

- **Typical block.** 46 draws from the suite's design space represent its
  range of shapes and sizes, not its pooled averages.
- **Red block.** Ten mechanisms in 21 cells. Features in the strip that are
  not of these kinds are untested.
- **Size range.** No Study T cell has n0 above 9,000, and no Study S cell has
  n0 below 20.
- **Small counts.** At α = .05 the per-cell and per-group loss counts are
  small, typically 0–15 flips. Group-level and cell-level differences
  smaller than about 0.5–1 pp are not resolved; macro numbers and pooled
  flip counts are.
- **Near-strip width.** The near strip is fixed (10 points, or 2% of n0). At
  50² it is about 40% of the grid, and at 300² it is narrower than some
  features. Its near/far split is a coarse proxy. The distance-to-boundary
  figures in §3.5 are the sharper evidence.
- **The oracle.** It is chosen on the same replicates and limited to a grid
  of .1 steps. With SEs around 10⁻⁴, the selection bias is small next to the
  headroom gaps reported.
