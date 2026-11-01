# Tightened floor closure and the sample-size M3 split

*Spec, 2026-10-27. Runner: `scripts/closure_split/`; tests:
`tests/test_closure_split.py` (experiment), `tests/test_hybrid_floor.py` and
`tests/test_m3_band_rs.py` (production options). Background:
`stats/research_record.md` (Q3, Q5, §7.3–7.6), the complement-trim
[report](complement_trim_report.md). Theory: `stats/fiducial_band_theory.md`
§8, §10.6, **§10.7** (new), §12.1.*

*This is an information-gathering study, run before the paper's methods
are fixed. It sets no thresholds and makes no choices. Its products are
paired contrasts, organized by question (§8), that show what each option
buys, what it costs, and where.*

## 1. The two questions

**Q-closure. How the hybrid's floor is closed.** Today the hybrid takes the hull of
the fiducial band and M3 on the floor region, then closes it to a monotone
band by *widening*. That carries the floor's room into the unfloored
interior next to the region. The alternative closes by *tightening* instead.
Theory §10.7 shows the tightened band misses exactly when the unclosed hull
does. So it keeps every exact statement the hybrid has, and it is never
wider. What we need to know is whether it holds nominal coverage, what it
saves in width, and where any added misses fall.

- `rec_T`: the tails-recalibrated hybrid with tightening.
- `hyb_T`: the full-grid hybrid with tightening. It still contains the raw
  C = 1 band.
- Their widened counterparts: `rec_W` (the currently preferred variant) and
  `hyb_W` (the production default).

**Q-split. How M3 splits its confidence budget between the classes.** Today M3
uses the Šidák split ρ = .5. The alternative is ρ = n0^−½ / (n0^−½ + n1^−½),
which hands more error allowance to the smaller class. Both are exact,
because ρ depends only on the class sizes, which the design fixes. So this
is purely a width question, plus a calibration side effect at large α. The
rule matters twice: for standalone M3, and for the M3 inside the floor.

## 2. What we already know

**Tightening (probe of 2026-10-27, 18 Stage F B/C cells, 200 fresh reps,
production M, scratch harness).** Compared with `rec_W`, tightening:
- saved 4.0% macro width at α = .05 (range 1–15%), and 5.1% at α = .5;
- lost 7 of 3,600 replicates at α = .05 and 88 of 3,600 at α = .5;
- gained most on long-run cells (wedge-01 .847×, imbalance-17 .894×,
  wedge-08 .878×). On wedge-01 the tightened hybrid was within 0.6% of raw
  C = 1's width while covering .985 against raw's .675.

That probe ran on the cells the floor was designed on, so it is not
prospective, and its cells were mostly adversarial.

**Why widening can matter (theory §10.7).** For every replicate,

`P(miss, tightened) − P(miss, widened) = P(widened covers, unclosed hull misses)`.

These are exterior misses that fall within the room widening carries out of
the region: lower-edge misses left of the right region, and upper-edge
misses right of the left region. Stage F found that floored miscoverage sat
below the exterior-escape rate, which is exactly this protection at work.
The complement-trim study found a slight excess of added misses within 5
columns of the region. Nothing bounds this term. Whether it is small is the
empirical question.

**The split.** The Sep 5 ideation screen found 0.9–5.8% lower paired area on
8 imbalanced cells, and identity at balance. The boundary-family search
vetoed the class split on one training shape, by a tail-width ratio of
1.0009, under a zero-tolerance rule. Theory: the first-order ROC half-width
is about `c(α_G)·sd_vertical + R′(t)·c(α_F)·sd_horizontal`. The rule
equalizes the √n scaling but ignores the slope R′. **It should lose where
the scarce class's contribution is small anyway:** few negatives at high
AUC, where most of the curve is flat (R′ ≈ 0), so the vertical,
positive-class term dominates even though n1 is large. Study S places cells
there deliberately.

## 3. Constructions

### 3.1 Study T: one cloud, two kernel calls, nine arms

Per replicate and nominal α ∈ {.05, .5}, one production-budget cloud
`M = _auto_n_draws(n0 + 1, α)` with one seed feeds two kernel calls. The
Rust cloud is a pure function of (seed, draw index), so every call sees the
same draws:

1. Full production trim rows at level α. This gives depth `j`, the raw band,
   and the exact floor region `A` (rule `"exact"`, ε = .001, δ = .025).
2. Production trim rows outside `A`, at level α. This gives
   `j_rec ≥ j` (Prop. 10b, asserted).

| Arm | Depth | Closure | Floor M3 split | Role |
|---|---|---|---|---|
| `raw` | j | — | — | C = 1 parent (reference) |
| `hyb_W` | j | widen | Šidák | production `m3_floor=True` |
| `hyb_T` | j | tighten | Šidák | tightened, contains `raw` |
| `rec_W` | j_rec | widen | Šidák | current preferred variant |
| `rec_T` | j_rec | tighten | Šidák | tightened recalibrated hybrid |
| `rec_T_ss` | j_rec | tighten | sample size | tightening and the split together |
| `m3` | — | — | Šidák | exact reference |
| `m3_ss` | — | — | sample size | M3 with the split |
| `ks` | — | — | — | fixed-width KS (Campbell), width context |

All fiducial arms are composed from the production functions, and a test
checks them **bit for bit** against
`fiducial_band_rs(m3_floor=M3Floor(closure=…, recalibrate_trim=…, split_ratio=…))`.
The options are now production code: `M3Floor.closure`,
`M3Floor.recalibrate_trim`, and `split_ratio="sample_size"` in both M3 and the
floor.

### 3.2 Exact properties checked on every replicate

These are implementation checks. Any failure means the results reflect a
bug, not the constructions, and should be investigated before reading them.

- Nestings, from Prop. 10e and Cor. 10c:
  - `rec_T ⊆ rec_W`, `hyb_T ⊆ hyb_W`, `rec_T ⊆ hyb_T`, `rec_W ⊆ hyb_W`;
  - `raw ⊆ hyb_T`, `raw ⊆ hyb_W`.
- `covers(rec_T) = covers(unclosed recalibrated hull)`, and the same for
  `hyb_T` (Prop. 10e(1)).
- `j_rec ≥ j`.

### 3.3 Study S: M3 alone

Per replicate and α ∈ {.05, .2, .5}, M3 is built with split ratios from the
family `ρ_p = n0^−p / (n0^−p + n1^−p)`:
- `sidak` (p = 0);
- `sqrt` (p = ½, the production `"sample_size"`);
- `linear` (p = 1, ρ = n1/n).

A fixed sweep ρ ∈ {.1, …, .9} also runs. It locates each cell's oracle
split, which tells us how much headroom the rules leave. Choosing ρ per
cell from outcomes would be a library fit, so the sweep describes headroom;
the rules that read only class sizes are `sidak`, `sqrt`, and `linear`. KS gives width context. M3 is deterministic given the ranks and costs
O(n) once calibrated. Every calibration is precomputed once with the exact
DP (`run calibrate`), and a test confirms no replicate requests one outside
that cache.

## 4. Where each option could fail, and the cells that test it

**Tightening fails only by the event in §2.** That needs two things:
widening must carry room out of the region, and the truth must have a
feature in the strip next to it. The red-team block puts realistic features
in that strip on both sides and on both edges.

| Group | Mechanism (realistic origin) | Strip it targets | Cells |
|---|---|---|---|
| R1 | Convex approach to (1, 1) wider than the right floor: heteroscedastic positives (σ ratio 3), near-Cauchy tails | lower edge, left of the right region | hetero3 .97 at 300² and 2,000²; t(1.3) .95 at 1,000² |
| R2 | **Hard positives** (label noise / undetectable cases): 1–4% of positives score like negatives. The run is broken, so the right floor is short, and the body's convex corner becomes an interior hook just outside it | lower edge, left of the right region | t2 .99 body at 500² (2%) and 2,000² (1%); binormal .97 body at 300² (4%) |
| R3 | **Hard negatives** (mislabelled or confusable negatives): 1–2% of negatives score like positives, pushing the low-FPR hook to about π·n0 ≈ 6–10 grid points, just past the left cut | lower edge, right of the left region | t2 .99 body at 1,000² (1%) and 300² (2%) |
| R4 | **Obvious-positive subgroup**: half the positives lie among the top 12 negatives, a near-vertical rise straddling the left cut (9–10 points) | both edges at the left cut | binormal .75 body at 500² and 2,000×500 |
| R5 | **Separated top subgroup** (bounded negative support): 25% of positives above every negative, R(0) = .25 | upper edge, right of the left region | binormal .80 body at 300² and 2,000² |
| R6 | **Scarce positives at scale** (record §7.4): long runs governed by n1, where widening spreads furthest. Mirror included | lower edge, left of the right region | t3 .95 at 4,500×500 and 500×4,500; binormal .95 at 9,000×1,000 |
| R7 | Interior support gap where the unfloored gap is short (small n) | interior next to both regions | jump h = .3 at 50² and 150² |
| R8 | Coarse scores (10 levels, trapezoid estimand) | runs of ties at both ends | binormal .90 at 500² |
| R9 | Large-n cells where the production hybrid sat at .955–.962 | the residual near nominal | complement study's weibull-13 and gamma-03 shapes at 8,000² |
| R10 | Fresh corner sliver at a new n | the floor's core mechanism | AUC .80 at 1,000² |

I checked the geometry before freezing. On 5 pilot replicates per cell:
- R2's right region is short (1–6% of the grid);
- R4's subgroup ends at 12 against a left cut of 9–10;
- R1 and R5 leave their features outside the region at both sizes.

**Where tightening should win (favored block).** The closure's charge grows
with the floor's extent. This block has four fresh, mechanism-defined
long-run cells:
- binormal .99 at 300²;
- σ-ratio .5 hetero .97 at 600² (narrow positives; region ≈ .8 of the grid);
- U-shaped beta-opposing .98 at 1,000²;
- bimodal .97 at 400².

It also keeps two probe anchors: wedge-01, and imbalance-17 (1,500×300). The
anchors let the prospective result be read against the probe.

**What the paper will actually report (typical block, 46 cells).** These are
frozen-seed draws from the main suite's own design space. AUC is uniform on
the probit scale over [.55, .99], and parameters are uniform within the
suite's bounds, over the six families distinct under rank invariance:
binormal, hetero-Gaussian, beta-opposing, Student t, bimodal-negative, and
power (Weibull). The class sizes are the suite's own, wherever the floor
leaves an unfloored interior:

| Sizes | Cells |
|---|---|
| 50² | 6 |
| 150² | 12 |
| 500² | 12 |
| 900×100 | 12 |
| 5,000² | 4 |

Each configuration takes the same number of draws from every family (two
at the three middle sizes), plus a remainder from families chosen without
replacement. At total n of 10 and 30 the hybrid *is* M3, so those sizes are
omitted. The draws span AUC .56–.99 (median .93, reflecting the suite's
probit-scale emphasis on high AUC), with floor regions from 0.4% to 86% of
the grid. The block's size was set after the dry-run showed budget to spare:
it carries the numbers closest to what the paper will report, and more cells
there buy more than more replicates per cell.

**Where the split should lose (Study S).** There are 12 shapes at 13
class-size configurations:
- balanced control (500²);
- 1:3 and 3:1;
- 1:9 and 9:1 at two scales (100/900, 500/4,500);
- 1:30 and 30:1 (150/4,500);
- 1:100 and 100:1 (100/10,000);
- small 1:9 (20/180).

The shapes are:
- diagonal (AUC .5), where M3's advantage over KS is weakest;
- binormal .65, .80, .95, .99;
- hetero σ ratio .5 and 2 at .90 (opposite asymmetries);
- t2 .95;
- U-shaped beta .98;
- bimodal .90;
- an upper-corner sliver;
- an interior jump.

The predicted losing quadrant is few negatives at high AUC: bn99, beta .98
and t2 .95 at 100/900, 150/4,500 and 100/10,000.

## 5. Replication, seeds, and budget

- **Study T.** Base replication is 1,000 reps for n0 < 2,000, 750 for
  2,000 ≤ n0 < 5,000, and 500 for n0 ≥ 5,000. While `rec_T`'s α = .05
  coverage is below .96, a cell is topped up by its base count, to 3× base
  (3,000 / 2,250 / 1,500). At 1,000 reps a cell covering .95 has a coverage
  SE of about 0.7 pp. The main contrasts are paired, and they are precise at
  base replication: a paired flip rate of 0.5% at 1,000 reps has an SE of
  about 0.22 pp. The top-up allocates precision to cells near nominal, where
  absolute coverage needs resolving. It makes precision uneven across cells
  but does not bias the estimates.
- **Study S.** 1,000 reps per cell. M3 at α = .05 covers ≈ 1, so .05 informs
  width only. α = .2 and .5 carry the calibration reading.
- **Seeds.** Per (cell, rep): `SeedSequence((20261027, sha256(name), rep))`
  draws the labels, then one 64-bit seed. One cloud seed per α is derived
  from it. All arms at an α share that cloud. The typical block's shapes come
  from a separate frozen seed (20261028).
- **Budget.** The `dry-run` estimate, measured on this machine (24 logical
  cores, 20 workers × 1 kernel thread) by timing replicates outside the
  experiment's index range:

  | | Cells | Base core-hours | Worst-case top-up |
  |---|---|---|---|
  | Study T, n0 < 5,000 | 66 | 6.9 | +3.5 |
  | Study T, n0 ≥ 5,000 | 7 | 5.6 | +4.5 |
  | Study S | 156 | 8.5 | — |
  | **Total** | 229 | **21** | **≤ 29** |

  At 20 workers this is about 1.1 h of wall time nominally. Allowing for
  memory-bandwidth contention (14 physical cores) and the worst-case top-up,
  expect **2–3.5 h**, inside the 5 h budget. Run Study S first (about
  30 min), then Study T. The dominant costs are the two 8,000² R9 cells
  (19 s per replicate each) and R6's 9,000×1,000 cell (16 s). They start
  first, because the scheduler orders cells largest first. The 5 h ceiling
  left room to enlarge the typical block. Dropping a C = 1.5 comparator
  arm later saved about 20% per replicate (one of three kernel calls, timed
  on five cells from 500² to 8,000²), and that saving went into 1.25× base
  replication, so the core-hour totals above are unchanged. Study T's
  replication was then doubled when more compute became available, which
  doubles both Study T rows of the table.

## 6. Records

Per replicate, α, and arm, the runner records:
- simultaneous coverage, and its lower- and upper-edge components;
- misses split by location: in the region, in the strip next to it
  (`near`: `max(10, ⌈.02·n0⌉)` grid points on the unfloored side of each
  boundary), and elsewhere (`far`);
- maximum miss depth;
- mean width, split the same four ways (left region, right region, near,
  far);
- up to 8 violation runs per direction.

Per α it also records:
- `j`, `j_rec`, M;
- the left cut and right start;
- the region fraction;
- the trim-row counts;
- the exact checks of §3.2.

Per replicate it records the saturated-run length, the count of positives
above the top negative, and the feature counts. These are:
- for the sliver and jump cells, the number of positives in the feature;
- for R5, the number of separated positives;
- for R4, the number of positives inside the top-12 window.

The location split is what lets the report say whether `rec_T`'s added
misses are the §2 mechanism (concentrated `near`, lower edge left of the
right region, upper edge right of the left region) or ordinary interior
noise.

## 7. Estimands

- Coverage with Wilson intervals; lower- and upper-edge miss rates.
- **Paired** per-replicate width ratios with SE:
  - `rec_T/rec_W`, `hyb_T/hyb_W`, `rec_T/hyb_W` (the net change against
    production);
  - `rec_T_ss/rec_T`, `m3_ss/m3`;
  - `rec_T` against `raw`, `m3`, and `ks`.
- **Paired flip rates**: the tightened (or split) arm misses where its
  counterpart covers, and
  the reverse. The location and direction of those flips are reported
  separately.
- Conditional coverage, given the feature is sampled or unsampled.
- Study S: paired ratios to Šidák for every rule and sweep value, the oracle
  ρ and its ratio, the fraction of the oracle gain the `sqrt` rule captures,
  how the width change distributes over FPR bands ([0, .1), [.1, .5),
  [.5, .9), [.9, 1]), and coverage by direction at .2 and .5.

Macro summaries weight cells equally within a block and are reported by
block. The typical block stands in for the paper. The red and favored
blocks are stress designs and are never pooled with it.

## 8. Questions, and the tables that answer them

`summarize` writes one table per question into `report.md` (and every table
into `summary.json`), so the report can be read top to bottom. Per-cell
detail is in `t_cells.csv` and `s_cells.csv`, and per-replicate records in
`t_reps.feather`. Each entry below says what the table shows and what the
main patterns would mean.

**Implementation checks.** The largest nesting violation, the number of
replicates where the tightened band's coverage differs from its unclosed
hull's, and the cells where the recalibrated trim had no rows left. The
first two should be zero (§3.2).

**Q1. What does tightening buy, and what does it cost?** By block and α:
- `rec_T` against `rec_W`, and `hyb_T` against `hyb_W`: paired width
  ratios (macro, min, max over cells), and paired lost and gained coverage
  (macro and worst cell);
- macro coverage of each arm and the lowest cell;
- the net width against production `hyb_W`.

The typical block shows the scale of the effect on the paper's population.
The favored block shows the ceiling, and the red block the exposure.

**Q2. Where do the added misses land?** For replicates that the widened
band covers and the tightened band misses: the count, the shares in the
region, in the adjacent strip, and elsewhere, the shares on each edge, and
the median depth, by block and α.
- Misses concentrated in the adjacent strip, on the lower edge left of the
  right region or the upper edge right of the left region, are the §2
  mechanism: widening's room was doing work there.
- Misses spread through the interior look like ordinary interior noise.

**Q3. What happens on each red-team mechanism?** Every red cell side by
side: the region fraction, coverage of raw, `hyb_W`, `rec_W`, `rec_T`, and
`hyb_T`, lost coverage, the share of flips in the adjacent strip and on the
lower edge, both width ratios, and coverage conditional on the feature
being sampled or not.

**Q4. Which cells cover least?** The 15 lowest cells at α = .05 by the
lower of `rec_T` and `hyb_T`, beside the widened arms and raw C = 1, with
`rec_T`'s Wilson interval. This separates cells that were already near
nominal from cells the closure moved there.

**Q5. How does it scale across the paper's sizes?** The typical block
grouped by class sizes:
- the region fraction;
- coverage of the four floored arms;
- width against the widened arms, production, M3, and KS;
- lost coverage.

The prediction is that the effect follows the region fraction, so it is
largest at 900×100 and smallest at 5,000².

**Q6. What happens to central-α calibration?** Macro coverage of every arm
at α = .5, by block, with width ratios against `hyb_W`. Calibration across α
is the desideratum nothing has met; this shows how far each arm moves
toward .50 and at what width.

**Q7. What does the split do on Study T's imbalanced cells?** `m3_ss/m3`,
`rec_T_ss/rec_T`, and the floor split's lost and gained coverage per cell.
The floor uses M3 only at the ends, so this answers a different question
from Study S.

**Q8. How do the split rules compare across class sizes?** For each
configuration and α, over the 12 shapes:
- `sqrt/sidak` (macro, min, max) and `linear/sidak` (macro, max);
- the oracle sweep's ratio;
- the median share of the oracle gain that `sqrt` captures.

**Q9. Where does the sqrt rule lose?** `sqrt/sidak` at α = .05 by shape
(rows) and imbalanced configuration (columns). The prediction places losses
at few negatives and high AUC.

**Q10. Where along the curve does the split move width?** The width change
of `sqrt` against Šidák in FPR bands [0, .1), [.1, .5), [.5, .9), [.9, 1],
by configuration. It shows whether the gain comes from the tails or the body.

**Q11. Does the split change coverage or its direction?** Macro coverage,
lower- and upper-edge miss rates for each rule and KS, at each α and block.
At α = .05 this is a sanity check: M3's theorem guarantees coverage. At .2
and .5 it is the calibration reading.

## 9. Predictions, recorded before the run

- **Width.**
  - `rec_T/rec_W` will be .96–.98 macro on the typical block. That is less
    than on the favored block (.85–.93), because most typical cells have
    short runs.
  - At 900×100 the gain will be the largest of the typical sizes; at 5,000²
    the smallest (≤ 1%).
  - At α = .5 the gain is about 1.3× the α = .05 gain.
- **Coverage.**
  - Lost-coverage rates will be ≤ 0.2 pp on the typical block.
  - The red cells most likely to show the largest losses are R2 (hard
    positives) and R6 (scarce positives at scale), with flips located `near`
    and on the lower edge left of the right region.
  - R5 (separated top) will show upper-edge flips right of the left region,
    if it shows anything.
  - R9 will move by ≤ 0.5 pp, because its region is tiny.
- **Split.**
  - `sqrt/sidak` will be .96–.99 on the 1:9 and steeper cells, with the
    largest gains on the diagonal and near-diagonal shapes.
  - It will exceed 1.00 on at least one few-negatives, high-AUC cell, but
    stay ≤ 1.01.
  - `linear` will be better at 1:100 and worse on the predicted losing
    quadrant.
  - The floor split (Q7) will be within ±1% of Šidák in width.

## 10. Limitations

- **Coverage and scope.**
  - The typical block is 46 draws, not the 7,000-point suite. It represents
    the suite's design space, not its pooled numbers.
  - The red block is 21 cells of 10 mechanisms. A feature not modelled
    there, in the strip next to the region, is untested.
  - No cell exceeds n0 = 9,000. At larger n the region is a tiny fraction of
    the grid, so tightening's effect shrinks, but it is not measured there.
- **Resolution.**
  - At 1,000 base replicates, per-cell coverage differences of about 1.5 pp are
    resolvable. Smaller per-cell effects are visible only in the macro
    numbers and the paired flip counts.
- **What the study does not change.**
  - Neither option changes the hybrid's missing whole-band theorem.
    Tightening keeps the existing exact statements; it adds no new ones.

## 11. Running it

From the repository root:

```sh
uv run --no-sync python -m scripts.closure_split.run design
uv run --no-sync python -m scripts.closure_split.run calibrate --workers 20
uv run --no-sync python -m scripts.closure_split.run dry-run   --workers 20
uv run --no-sync python -m scripts.closure_split.run run --study S --workers 20
uv run --no-sync python -m scripts.closure_split.run run --study T --workers 20 --threads 1
uv run --no-sync python -m scripts.closure_split.run summarize
```

Records go to `data/results/closure_split_20261027/cells/`, one file per
cell, and runs resume from them. The runner refuses records written for a
changed cell definition. `summarize` writes `t_cells.csv`, `s_cells.csv`,
`t_reps.feather`, `summary.json`, and `report.md`. The report contains the
§8 tables in question order.
