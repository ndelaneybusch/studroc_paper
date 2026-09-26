# Complement trimming: stop charging the interior for the floored ends

*Spec, 2026-09-26. Runner: `scripts/complement_trim/`; tests:
`tests/test_complement_trim.py`. Background: `stats/research_record.md`
(Q3, Q4, §7.1, §7.6). Theory: `stats/fiducial_band_theory.md` §§5, 6, 10.
This is an information-gathering study. It does not select a method, and the
.94 and .95 bars below are yardsticks, not gates.*

## 1. The question

The hybrid band builds a C = 1 fiducial band, then replaces its two ends with
the hull of that band and the exact M3 band (the floor). But the fiducial trim
depth `j` is still computed over the whole grid, ends included. So wherever the
end columns lower the depth, the interior pays for ends that the floor has
already taken over.

**If the trim depth is computed only on the columns the floor does not
protect, how much narrower does the hybrid get, at what cost in coverage, and
where does that cost land?**

## 2. Why run it, and what we already expect

Reasons it could help:

- The end columns take 4–8× their per-column share of the draws' minimum
  depths (research record §7.1, diagnostic of 2026-09-26). Removing them
  raises `j` by 1.2–1.6× at n = 500, and in the pilot below by about 2× on a
  heavy-tailed wedge cell.
- At n = 5,000 an interior-only trim at C = 1 landed near nominal on the
  interior and was .92–1.00× the floored band's area (methods exploration,
  research record Q4). Its eligibility gate made that sample unrepresentative.
  This design has no gate.
- The floored band uses only about a third of its .05 error budget
  (Stage F §7.2), and it over-covers badly at α = .5 (.78–.90 against .50).
  Central-α calibration is the one desideratum nothing has touched (record §7.5).

Reasons it may not:

- At n = 500 the diagnostic found that interior-only trimming narrows the
  interior by only 2–5% at α = .05 and 5–11% at α = .5. The interior itself is
  conservative at that size, and the ends are not the main cause.
- The gain goes straight out of the margin that currently absorbs misses
  outside the floor region. Stage F measured those misses only on adversarial
  designs, and no theorem bounds them.

So the realistic hope is a few percent of width at α = .05, more at α = .5 and
at large n, with coverage moving toward nominal. The point of the study is to
measure that trade across sample size, shape, and imbalance, including on
constructions built to hurt an interior-trimmed band.

## 3. Construction

On one tie-resolved label order, at trim level `a` (C = 1, so `α_eff = a`),
with the production cloud budget `M = _auto_n_draws(n0 + 1, a)`:

1. **Full trim.** Build the production tube on the production trim rows `J`
   (the full grid, thinned when n0 + 1 > 2001). Call its depth `j_full`, and
   apply the production corner allowances at `j_full`.
2. **Region.** Compute the production exact floor region
   (`hybrid_floor.floor_region`, rule `"exact"`, ε = .001, δ = .025) at
   `j_full`.
3. **Complement trim.** On **the same cloud** (same seed and M), recompute the
   depth on `J' = J \ region`. Call it `j_raw`.
4. **Enforced direction.** Set `j_comp = max(j_full, j_raw)`. If the maximum
   binds, use the full-grid tube, which is exactly the tube at depth `j_full`.
   Every such event is counted. Apply the corner allowances at `j_comp`.
5. **Floor.** Take the hull with M3 at level `a` on the region from step 2,
   and close by widening. If `J'` is empty (the floor covers the grid), the
   complement arm is the production hybrid, flagged as a fallback.

### 3.1 Why computing the region from `j_full` is principled

The floor's left cut depends on `j`, and the complement trim changes `j`. Two
monotonicity facts make the dependence one-directional:

- **The complement depth cannot fall.** A draw's depth is its minimum rank over
  the trim columns. A minimum over a subset of columns is at least the minimum
  over all of them. So every draw's depth weakly rises, and so does the same
  quantile: `j_raw ≥ j_full` exactly. Step 4 enforces this rather than relying
  on it, so a numerical or indexing fault cannot silently reverse it.
- **The left cut only gets safer as j grows.** The exact cut is the first
  column k at which `P{Binom(M, (1 − k/n0)^n0) ≥ j} ≤ ε`, i.e. where the
  lower edge is a chord draw with probability at most ε. That tail falls as
  `j` grows. So a cut that satisfies the condition at `j_full` still satisfies
  it at `j_comp`.

The right start depends only on the saturated-run length and δ, not on `j`.
So the region computed at `j_full` is the most conservative region any trim
domain would produce, and its defining property holds for the deeper band.
Its cost is at most a grid point or so of extra left region. Iterating to a
fixed point instead is not principled: the map from j to region to j is
non-increasing and can cycle.

The exact left cut already treats the realized `j` as fixed, although `j`
comes from the same cloud. That is an approximation production makes today,
not one this design adds.

### 3.2 What is exact, and what is lost

- **Kept [Exact]:** misses inside the region imply a full M3 miss, so their
  probability is ≤ the floor's M3 level, for any data-dependent region.
- **Lost:** the production hybrid contains the raw C = 1 band pointwise, and
  so never covers less. The complement arm contains **neither**. It sits inside
  the production hybrid (tested), so it can cover less than both.
- **Budget:** `P(miss) ≤ α₂ + P(complement band misses outside the region)`.
  The second term has no bound. If the interior-calibrated asymptotics of
  Theorem 7 applied to the (random) complement domain, it would tend to the
  trim level. That motivates the two budget variants below, and nothing more.

## 4. Arms

Every arm is built on the same label order at each nominal α ∈ {.05, .5}.
All arms at one level share one cloud.

| Arm | Trim | Floor M3 level | Cloud | Role |
|---|---|---|---|---|
| `hybrid` | Full grid at α | α | M(α) | The production `m3_floor=True` band |
| `comp` | Complement at α | α | Same as `hybrid` | The direct change; worst-case budget about 2α |
| `comp_budget` | Complement at α/2 (region from its own `j_full`) | α/2 | M(α/2) | A declared α/2 + α/2 split |
| `m3` | — | — | — | Full M3 at α (exact reference) |

Two references come for free from the same clouds:

- `raw`: the C = 1 band at α, the parent of `hybrid` and `comp`.
- `hybrid_half`: the full-grid hybrid at α/2, the parent of `comp_budget`.

`hybrid_half` separates the cost of halving the budget from the effect of the
trim domain. Comparing `comp_budget` with `hybrid_half` isolates the domain;
comparing it with `hybrid` gives the net trade.

## 5. Cells

50 cells.

- **Stage F Study B (30):** 10 student-t wedge cells (n = 130–6,656), 6
  mechanism-diverse safe cells, 4 imbalance cells in both orientations, 2
  large-n cells (8,000, 12,000), 2 regression cells (Q = 20 ties and a held-out
  shape), and the 6 prospective sliver cells.
- **Stage F Study C (14):** seven non-wedge shapes at n = 500 and 8,000
  (six named families plus one LHS draw that is student-t).
- **Interior features (6, new).** These target the arm's specific risk: an
  interior-trimmed band is narrower exactly where the floor gives no
  protection.
  - `interior_sliver` at 500 and 2,000 per class: the methods-exploration
    shape, with mass min(.1, .8/n1) in a width-1/n0 step at FPR .6 and 20% of
    the positive mass beyond FPR .85, so the saturated run cannot reach it.
  - `interior_jump` (a binormal .80 body plus a jump of height .2 at FPR .3,
    width 1/(4 n0), i.e. inside a single negative gap) at 500×500,
    2,000×2,000, 2,000×500, and 500×2,000.

  The B/C truths and their construction are those of the Stage F design.
  Names and seed streams are new (`ct-` prefix, study seed 20260926), so
  nothing is replayed.

## 6. Replication, budget, and seeds

- **Replication.** Cells with n0 < 5,000 get 400 base reps, topped up in
  batches of 400 to 1,200 while the α = .05 Wilson interval of `comp` or
  `comp_budget` straddles .94. Cells with n0 ≥ 5,000 get 200 base reps and a
  cap of 600. That follows the program's preference for more cells over more
  replicates, and at large n the paired width ratios, which carry most of the
  information there, are precise at 200.
- **Cloud budget.** Production M at each trim level (so larger at α/2).
- **Seeds.** Per (cell, rep): `SeedSequence((20260926, hash(name), rep))` draws
  the labels, then one 64-bit seed, from which a separate seed is derived for
  each (α, level) cloud. The Rust cloud is a pure function of (seed, draw
  index), so the two trims at a level see the identical cloud.
- **Cost.** `dry-run` on this 14-core machine estimates 21 h of wall time at
  base replication: about 8 h for the 39 cells with n0 < 5,000 and 13 h for
  the 11 large cells. Cells run cheapest first. `--cells` and `--reps-scale`
  shard or shrink the run.

## 7. Records and estimands

Per replicate, α, and arm the runner records:
- simultaneous coverage and its lower- and upper-edge components;
- misses inside and outside the arm's floor region;
- maximum miss depth;
- mean width over grid points, split into the in-region and out-of-region parts;
- up to 16 violation intervals per direction.

Per level it records `j_full`, `j_raw`, `j_comp`, whether the max bound,
fallbacks, trim-column counts, the region fraction, M, and the saturated-run
length. Sliver and interior cells also record whether their feature was sampled.

Per cell the summary reports:
- coverage with Wilson intervals;
- directional and regional miss rates;
- mean width, and the paired per-replicate width ratio to `hybrid` (with SE) and to M3;
- mean depths;
- coverage conditional on whether the feature was sampled.

Macro summaries weight cells equally, and they describe these designs, not
ROC curves in general (research record §6).

## 8. Questions, and how each will be read

1. **Width.** The paired ratio of `comp` to `hybrid` by n, α, block, and
   imbalance. Is the gain larger at large n and at α = .5, as the diagnostic
   and the exploration suggest?
2. **Coverage.** How far do `comp` and `comp_budget` move toward nominal, and
   do any cells fall materially below it? Is the loss concentrated in
   particular blocks (wedge, sliver, interior features, imbalance)?
3. **Where the new misses land.** In region or out, lower or upper edge,
   near or far from the floor boundary. A rise in lower-edge misses just
   outside the region would mean the region computed at `j_full` is too short
   for the deeper band, contrary to §3.1. Misses spread through the interior
   would mean ordinary interior calibration.
4. **Budget split.** Whether `comp_budget` recovers `hybrid`'s coverage, and
   at what width; and how much of any difference is the halved budget
   (compare it with `hybrid_half`).
5. **Interior features.** Coverage of `comp` versus `hybrid` on the interior
   cells, conditional on the feature being sampled. This is the arm's
   distinctive risk.
6. **Depth direction.** The count of binding max events. The expectation is
   zero. Any nonzero count is an implementation finding, to investigate before
   interpreting the results.

**Predictions, recorded before the run:**
- `comp`/`hybrid` width will be about .95–.98 at α = .05 for n ≤ 1,000 and
  lower at large n and at α = .5.
- `comp` coverage will fall toward, but mostly stay at or above, .95 on
  regular cells, and will drop on at least some wedge or interior-feature cells.
- `comp_budget` will be wider than `hybrid` at moderate n, because the
  wider α/2 M3 and the deeper α/2 trim cost more than the domain saves.

## 9. Pilot (engineering check only)

This is 20–60 reps on three cells, and its numbers are not evidence. On
sliver-24, wedge-01, and interior-jump-500, `comp`/`hybrid` width was
.97–.98 at α = .05 and .96 at α = .5. `comp_budget` was 5–15% wider than
`hybrid`. On wedge-01, `j` moved from 12 to 25 at α = .05. The max never
bound; fallbacks occurred only when the floor covered the grid.

## 10. Limitations

- There is no fixed-shape large-n ladder beyond 12,000. Nothing here
  addresses n ≫ 10⁴.
- The interior-feature block is six cells of two constructions, and it cannot
  represent arbitrary interior structure.
- The complement domain depends on the ranks. No theorem, Theorem 7
  included, covers coverage on a random trim domain, so asymptotic readings
  are heuristic.
- The floor is the production exact rule only; the Stage F rule is not re-run.

## 11. Running it

From the repository root:

```sh
uv run --no-sync python -m scripts.complement_trim.run design  --out data/results/complement_trim_20260926
uv run --no-sync python -m scripts.complement_trim.run dry-run
uv run --no-sync python -m scripts.complement_trim.run run --workers 4 --threads 4
uv run --no-sync python -m scripts.complement_trim.run summarize
```

Runs resume from the per-cell files in `data/results/complement_trim_20260926/cells/`.
The runner refuses to mix records from a changed cell definition.
