# Bounded method exploration: what the screen found

*Run 2026-09-09/10. Executes `stats/methods_exploration_spec.md` at
`--profile screen`, 12 threads. Data: `data/results/methods_exploration_screen/`.
3.5 hours of track time, 135,448 stored units.
Companions: `fiducial_band_theory.md` (Theorem 7, §§12.1–12.2, 12.6–12.7),
`hybrid_floor_report.md` (Stage F), `c_calibration_screening_report_stage_s.md`
(Stage S), `c_calibration_followup_report.md`, `next_method_ideas.md`.*

This is an information-gathering screen. Nothing here promotes a method, and
no candidate was frozen from an incomplete design.

---

## 0. What ran

| Track | Status | Time | Budget | Units |
|---|---|---:|---:|---|
| small_n | complete | 9.0 min | 60 min | 40,500 of 40,500 |
| interior | budget exhausted | 194.9 min | 195 min | 75,510 of 90,000 |
| likelihood | complete | 2.9 min | 45 min | 1,152 of 1,152 |
| projection | complete | 4.6 min | 30 min | 896 (384 exact, 512 Monte Carlo) |
| m3 | complete | 1.2 min | 30 min | 17,280 of 17,280 |

Four of five tracks finished their whole design, three of them in a small
fraction of the time allotted. The interior track did not: it completed every
cell at n = 100, 500 and 5,000 (75,000 units) and then got 510 units into the
balanced n = 50,000 block before the deadline. The n = 50,000 numbers below are
reported as the partial evidence they are — about 100 replicates per shape in
the balanced direction only, and nothing at all in the two imbalanced
directions. Finishing that size at this per-unit cost would take roughly a day,
not the hour that was left.

One operational note that matters for reproducing this. The supervisor kills an
over-budget worker with `TerminateProcess` on Windows, which does not run
Python's `finally` blocks, so the interior track never wrote its own summary.
Every completed record was on disk, and the summary is a pure function of those
records, so it was rebuilt with the track's own `summarize`. The rebuild was
verified against the pilot: it reproduces the track's output byte-identically
apart from the `pilot` flag.

---

## 1. Small samples: is there an interior at all?

### The uncertainty

Stage F established that a localized M3 floor repairs the fiducial band's
corner failures at moderate and large samples, at roughly a fifth of the width
cost of routing. It said almost nothing about ten-to-fifty observations per
class, which is where a lot of real evaluation sets live. Three things were
open. Whether the floor's two tail regions leave any unprotected interior at
all at those sizes — if they meet in the middle, the "hybrid" is just M3
wearing a different name. Whether the new exact tail rules (a binomial left
cutoff keyed to the realized trim depth, a Beta-inverted right margin) beat
Stage F's frozen `ceil(log(M+1))` and `ceil(2√K)` heuristics. And whether
small-sample trouble is a problem of tail geometry, of cloud resolution, or of
the construction itself — because those three point at completely different
repairs.

### How it was measured, and what that costs

Nine class-size designs (10/10 through 50/50 balanced, plus 10/50, 50/10,
20/50, 50/20), nine shapes, 500 paired datasets each, at α = .05 and α = .5:
40,500 units. Every arm sees the same sampled labels and the same cloud seed.
Six arms: raw C = 1, the exact hybrid, full M3, and — as mechanism probes
rather than candidate bands — left-floor-only, right-floor-only, and the frozen
Stage F hybrid. Production's automatic cloud budget was used separately at each
α (M = 2,000–2,784 here), with production trim rows and the realized C = 1
depth, so the floor sees the budgets a deployed band would actually have.

Three costs are worth stating up front, because several headline numbers are
uninterpretable without them.

The shape library is a deliberate stress set. Two of its nine shapes (a
step-function `jump` at FPR .5, and a boundary `sliver` carrying rare positive
mass scaled to `n1`) are constructed to break the fiducial completion, and a
third puts positive atoms at both endpoints. Any number pooled across the nine
shapes describes that adversarial mixture, not ROC curves in general.

At 500 replicates, a cell coverage of .99 carries an exact two-sided interval
of about ±.01. Cell coverages should not be read to three digits.

Most importantly, the tail windows lose their meaning at these sizes. The
reporting windows [0, .02] and [.95, 1] span *at most one native grid cell*
when n₀ = 50, and none at all when n₀ < 50. Tail-width diagnostics carry almost
no information here, and the unfloored region must not be identified with
[.02, .95]. The study records realized floor-region metrics separately for
exactly that reason.

### What was found

**There is no interior, and that is arithmetic rather than statistics.** The
fixed [.02, .95] window was eligible in **0 of 40,500 datasets**, at every
size, shape and α. This is not a small-sample sampling accident. The window's
left guard column sits at grid index 0 when n₀ < 50 and index 1 when n₀ = 50,
and the exact left mask always covers index 0 and always extends to at least
index 1 — a cutoff of k = 0 gives inclusion probability 1, whose binomial
survival is 1, never ≤ .001. So at n₀ ≤ 50 the window cannot clear the floor
under any realization whatsoever. Eligibility needs n₀ ≳ 50·k*, and k* runs
about 7 grid points at α = .05 and 3–4 at α = .5, which puts the earliest
possible eligibility somewhere north of n₀ ≈ 350.

The unprotected region that does exist shrinks fast as sizes fall. At n₀ = 10
the floor covers the entire grid in 100% of replicates for all nine shapes. At
n₀ = 15 the fully-floored rate ranges from .09 (endpoint atoms) to .99 (`jump`,
`normal_0.95`). Only by n₀ = 50 does a genuine interior appear for most
shapes — .65 to .73 of native cells unprotected for the low-AUC shapes, but
still only .16 for `normal_0.95` and .41 for `sliver`.

**Raw C = 1 does not merely lose calibration at these sizes; on two shapes it
collapses.** At α = .05, 20 of 81 cells fall below nominal. The worst is
`jump`, where raw coverage runs .108–.252 across every size — the band misses
the truth in five replicates out of six. `sliver` runs .508–.612 once n₀ ≥ 20.
Conditional on the sliver's rare mass never being sampled, raw coverage is
**exactly zero** in every design with n₀ ≥ 30, on 205–235 conditioning
replicates per cell.

**The floor repairs essentially all of it.** Hybrid coverage is at or above
nominal in all 81 cells at α = .05 (minimum .984, at 50/50 endpoint atoms;
lowest exact-interval lower bound across all cells .969) and in all 81 cells at
α = .5 (minimum .790 against a nominal .5). The per-replicate repair
probability — raw fails, hybrid covers — reaches .88 on `jump`.

**The repair is a right-tail repair.** The regional decomposition is
unambiguous. At 50/50 on `jump` at α = .05, raw coverage inside the left-floor
region is 1.000, inside the right-floor region .108, and inside the genuinely
unfloored region .992. The saturated-run component does all of the work. The
ablations confirm it from the other side: left-floor-only still leaves 15 of 81
cells below nominal at α = .05 and does not move `jump` at all beyond n₀ = 30,
while right-floor-only leaves 2. Neither alone suffices, and their failure
rates are not additive because the masks overlap — the overlap rate runs from
.09 at 50/50 to .66 at 10/10.

**The hybrid is never wider than full M3.** The spec anticipated that a
regional hull could exceed M3 even with no usable interior. Empirically it
never did: P(hybrid area > M3 area) = 0.000 in all 162 cells, maximum observed
ratio exactly 1.0000. At n₀ ≤ 15 the hybrid *is* M3, to machine precision,
because the floor covers the grid. The saving only becomes material once an
interior exists — median hybrid/M3 area .967 at α = .05, .799 in the best cell.

**Cloud resolution is not the binding constraint.** Doubling M on the same
labels moved raw band area by at most 3.0% and hybrid area by at most 1.2%
across all twenty audited size/shape/α combinations, while the realized trim
depth doubled exactly, as it should. Realized depths at α = .05 ran 4–25, never
below the resolution floor of three.

**The exact tail rules beat Stage F's, mostly at large α.** The exact left
cutoff lands at a mean of 5.0–7.2 grid points at α = .05 and 3.0–3.8 at α = .5,
rising with n₀ and adapting to the realized depth; Stage F's `ceil(log(M+1))`
gives **exactly 8, at every size and both α**, because it is blind to α by
construction. The exact right start is 0.1–1.3 points earlier, so slightly more
right-side protection. Netting out, at α = .05 the exact hybrid is 0.5%
narrower at 50/50 with indistinguishable coverage (.9949 vs .9951 pooled); at
α = .5 it is 4.5% narrower, with pooled coverage .888 vs .901 against a nominal
.5. Trading surplus coverage for width at α = .5 is the right direction; the
α = .05 gain is real but small.

**The bands are very wide in absolute terms.** At 10/10, mean interior width
runs .49–.95 of the unit interval. Even at 50/50 it is .21–.48. The right-tail
width ratio of hybrid to raw reaches 8× at α = .05 and 13× at α = .5, because
raw collapses to near-zero width inside the saturated run and M3 restores an
honest width there.

### The responsible reading

At ten to fifty observations per class the hybrid is, to a very good
approximation, M3 with a fiducial interior that either does not exist or is too
small to matter. That reframes the tuning question rather than answering it.
Cloud resolution is ruled out as the lever — doubling the budget changes
nothing. Tail geometry is a real but small lever: the exact rules genuinely
improve on Stage F's heuristics and are worth carrying forward for their
α-adaptivity, but they buy half a percent of area at α = .05. If small-sample
width is something we care about, it needs a different construction, not a
better floor.

The second reading is about the fixed window itself. A [.02, .95] interior is
not a mild design choice at these sizes — it is unreachable by construction,
and the zero eligibility rate is that arithmetic showing up, not evidence about
any schedule. A future small-sample interior would have to be defined relative
to the realized floor, not to fixed FPR coordinates.

What this study cannot say: `raw` here is the C = 1 band on a stress library,
so its .108 on `jump` is a statement about a step-function ROC and the
sorted-uniform completion, not an average failure rate. What generalizes is the
mechanism and the conditional-on-unseen-mass zeros, not the pooled number.

---

## 2. The interior exponent after the exact floor

### The uncertainty

Stage S measured the whole-curve trim exponent and found C\* ≈ 3.1, 2.2, 1.8,
0.87 for binormal .95 across n = 100 to 50,000, converging to 1 at the top —
confirming Theorem 7's asymptotic prediction while showing that the approach to
that limit is not a law shared across shapes. Since then the exact floor has
taken ownership of both tails. The obvious question is where Stage S's surplus
conservatism actually lived. If it lived in the tails, the floor has now
absorbed it and there is nothing left in the interior to spend. If it lived in
the interior, an exponent schedule confined to a fixed interior window is a
free width win at moderate n.

### How it was measured, and what that costs

Balanced-equivalent sizes 100, 500, 5,000, 50,000; class fractions 1:1, 1:9,
9:1 at total size 2n; five shapes; α = .05 and .5; a nested C ladder of 1,
1.25, 1.5, 2, 2.5, 3.5, 5, 8; 2,000 replicates per cell at the two smaller
sizes and 1,000 at the two larger. On each dataset the study first builds the
production-compatible hybrid, then asks whether the whole fixed [.02, .95]
window — plus the two native columns bracketing its endpoints — is clear of
both floor masks. Only then does it re-trim on the window's columns with the
same cloud seed, keep the parent band outside, reapply the frozen M3 hull and
widen to monotonicity. When ineligible it returns the unchanged hybrid for
every C, including C = 1.

Three costs, and the third is the important one.

The design is 84% complete. Sizes 100, 500 and 5,000 are whole; n = 50,000 has
about 100 replicates per shape in the balanced direction only.

Conditioning on eligibility is not a neutral act. Eligibility is a function of
the observed ranks, so the eligible subsample is not a random subsample, and
Theorem 7 supplies no theorem for eligible-only coverage. The study therefore
reports operational coverage over all datasets, eligible-only coverage, and
fallback-only coverage separately. §6 shows how far from neutral the
conditioning turned out to be.

The cloud budget is below production. Per the spec the study used M = 2,000 at
n ≤ 500 and 4,000 above; production's automatic rule would use 5,158 at
n₀ = 500, 9,599 at n₀ = 5,000 and 17,874 at n₀ = 50,000. The consequence is
visible in the realized trim depth on eligible datasets at α = .05: cell means
run 1.99–6.42, and sixteen of the twenty-three cells sit below 3.2, with every
n = 50,000 cell at 2.0. That is the regime where the band falls back toward the
conservative full envelope of its own cloud. It makes the α = .05 arm *more*
conservative than production would be, not less, which matters for how the
result below should be read.

### What was found

**Eligibility is governed by the ROC's own geometry, not by sample size.** The
window's left guard clears once n₀ exceeds roughly 50·k*, and that happens
between n₀ = 100 and n₀ = 500. After that, eligibility is decided almost
entirely on the right, by the length of the trailing all-negative run — which
is a property of the truth, not of n.

| eligibility rate, α = .05 | n=100 (all ratios) | n=500, n₀=500 | n=5,000, n₀=5,000 | n=50,000, n₀=50,000 |
|---|---:|---:|---:|---:|
| `interior_sliver` | .000 | .000 | 1.000 | 1.000 |
| `kink` | .000 | .000 | 1.000 | 1.000 |
| `t2_0.95` | .000 | .004 | 1.000 | 1.000 |
| `sliver` | .000 | .418 | .579 | .480 |
| `normal_0.95` | .000 | .000 | .130 | .843 |

Zero in all 30,000 datasets at n = 100, at both α, in all three class ratios.
At n = 500 the α = .05 left cutoff lands at FPR .0215–.0219 against a guard at
.0200 — the window misses by one or two grid columns. The three shapes whose
right side is already clear at that size (`interior_sliver`, `kink`, `t2_0.95`)
are therefore eligible at α = .5, where the cutoff falls to .0100, and
ineligible at α = .05. On the right, a binormal at AUC .95 has a trailing
all-negative run long enough to block the window until roughly n₀ = 50,000. A
high-AUC ROC simply does not have an unprotected [.02, .95] interior at any
sample size a practitioner is likely to have.

**On eligible datasets, the first rung of the ladder is already at or below
nominal.** This is the central result.

| eligible-only coverage, α = .05, C = 1 | n=500, n₀=900 | n=5,000 (3 directions) | n=50,000, n₀=50,000 (partial) |
|---|---:|---:|---:|
| `interior_sliver` | .974 | .950–.959 | .951 |
| `kink` | .974 | .952–.967 | .902 |
| `t2_0.95` | .971 | .957–.961 | .922 |
| `sliver` | .955 | .902–.953 | .918 |
| `normal_0.95` | .400 (5 eligible) | .840–.962 | .977 |

(At n = 500 the only other estimable cell is `sliver` in the balanced
direction, at .952 on 837 eligible datasets.) At n = 5,000 the C = 1 windowed
band falls below nominal in 4 of 15 cells; in the partial n = 50,000 block, in
3 of 5. C\* is left-censored below the bottom of the ladder in those cells.
Where a crossing does resolve it is small — [1.0, 1.25] or [1.25, 1.5] in most
n = 5,000 cells, against Stage S's whole-curve 1.78 for binormal .95 and 1.51
for kink at the same size.

**Restricting the trim domain to the window costs coverage rather than
harvesting surplus.** On the same eligible datasets, the full-grid floor band
covers .936–.990 at α = .05 while the windowed C = 1 band covers .902–.975 —
lower in 21 of the 23 cells with at least 50 eligible datasets, tied in the
other two, by up to 3.5 points. The misses are essentially all outside the
floor region (in-region miss rate .000–.011), split roughly evenly between the
lower and upper edges. The width does drop: at α = .05 windowed C = 1 runs
.920–.999 of the floor's area and C = 2 runs .843–.960. But that width is
coming straight out of the coverage margin.

**No candidate was frozen, and the gate that blocked it is the right one.**
`normal_0.95` never reaches the 400-eligible threshold in any group — 0, 5, 20,
25, 130, 164, 86 eligible across the groups — so no group has all five shapes
adequately represented, no shape-spread bootstrap is estimable anywhere, and
the candidate file records `eligible: false` with C = 1 at both α. The realized
depth below three at α = .05 blocks it independently.

### The responsible reading

Stage S's surplus was mostly tail surplus. Once the exact floor owns the tails
and the trim is confined to a fixed interior, what remains is not conservative
enough to spend: on three of five shapes at n = 5,000 the ladder's first rung
is already at or under nominal. And the cloud-budget caveat pushes the same
way — the α = .05 arm here was *more* conservative than a production-budget
band would be, so a production band would show even less interior surplus, not
more. The honest inference is that an interior exponent schedule is not the
place to look for width after flooring. That is a genuine update: it does not
contradict Stage S, it relocates Stage S's finding.

The second inference is about the window as an idea. A fixed [.02, .95]
interior is unreachable below n₀ ≈ 350 and, for high-AUC ROCs, unreachable
until n₀ ≈ 50,000. A schedule gated on that window would be inert on most real
datasets and would fire exactly where the data are unusual. Any future interior
construction should be defined against the realized floor.

What this cannot say: the n = 50,000 evidence is ~100 replicates per shape in
one direction, wide intervals, balanced only. The trend it shows is consistent
with the n = 5,000 result, but it is not on its own decisive, and the two
imbalanced directions at that size were never run.

---

## 3. Likelihood inversion: measuring the losses before building a solver

### The uncertainty

Rank-likelihood inversion is a second exact route to an honest band, and the
question was never whether it is valid — it is — but whether the losses you
have to pay to compute it leave anything worth having. Three losses stack: the
predictive penalty from using a normalized predictor `q` instead of the
unknowable truth, the cell loss from bounding a within-cell likelihood on a
finite partition, and the projection slack from reporting a coordinate-wise
hull instead of the confidence set itself. The point of this track was to
measure all three before anyone writes a solver.

### How it was measured, and what that costs

At n₀ = n₁ ∈ {5, 10, 20}, 64 replicates on each of six truths, comparing three
normalized predictors — uniform over paths, a fixed equal-weight mixture of
piecewise-linear binormal rank laws at seven AUCs, and a sequential
count-respecting label predictor with a Beta-smoothed prefix table fitted on
4,096 independent mixture draws — plus the oracle `q = p_true` as a diagnostic
numerator that nobody could deploy. Exact within-cell likelihoods use
denominator (a+b)!; the upper and lower bounds use a!b! and pure transitions.
Every float64 computation is checked against an independent `Fraction` DP, and
the run asserts the bounds bracket the exact value.

The essential cost, which the spec is emphatic about: the candidate library is
48 fixed curves, so its hull is an **inner** object. It is optimistic by
construction, it can omit an accepted curve, and it never certifies a band. The
three losses also do not add in area; there is no canonical decomposition.

### What was found

**The predictive penalty is the one loss that a good predictor essentially
removes.** At n = 20, α = .05, log(p_true/q) on the four smooth truths:

| predictor | AUC .5 | .6 | .8 | .95 | `jump` |
|---|---:|---:|---:|---:|---:|
| uniform | 0.00 | 0.63 | 6.29 | 14.00 | 23.35 |
| mixture | 1.10 | 0.79 | 0.81 | 0.86 | 22.92 |
| sequential | 1.22 | 0.92 | 0.93 | 0.90 | 21.92 |

The fixed mixture costs about one nat, flat across the smooth family, where the
uniform predictor costs fourteen at AUC .95. Notably, the fitted sequential
predictor does not beat the fixed mixture anywhere — it is a shade worse
everywhere. And on `jump`, which is outside the mixture, everything costs 22–23
nats, so the predictive loss is really a statement about how far the truth sits
from your predictor's support.

**The cell loss is catastrophic at every resolution tried.** The actual
upper-minus-lower likelihood gap, divided by the acceptance cutoff, at n = 20,
α = .05, mixture predictor. (The requested 4/8/16/32-cell refinements become
12/12/20/36 actual cells once every candidate breakpoint and atom is added, so
the two coarsest requests coincide.)

| truth | 12 cells | 20 cells | 36 cells |
|---|---:|---:|---:|
| smooth .5 | 8.9e8 | 1.4e7 | 2.3e5 |
| smooth .8 | 4.3e7 | 1.3e6 | 5.8e4 |
| smooth .95 | 1.1e6 | 1.5e5 | 2.2e4 |

Five to nine orders of magnitude larger than the quantity it must resolve. The
capped union bound is 1.0000 everywhere, and the coarsened rule accepts 3.8 to
26 extra library curves beyond the exact-likelihood accepted set. The only
truth where the gap vanishes is `jump`, whose cells are degenerate.

**The inner hull looks excellent, and the certified outer version is
vacuous.** With the mixture predictor at n = 20, α = .05, the inner hull's area
on the four smooth truths is .392–.503 of M3, nonempty on 100% of replicates,
with 95% upper ratios of .42–.53 and 32-cell upper hulls of .66–.87. That
passes the spec's gate. The
prototype the gate enables then ran a certified rational box enclosure over the
whole ordered knot cube on 48 datasets, and returned a mean area of **0.986** —
the unit square, with 62.4 of 63 boxes unresolved.

Two smaller things. The projection slack — rejected library curves lying
entirely inside the hull of accepted ones — is .006–.133 on smooth truths but
.62–.70 on `jump`, so on a step ROC the coordinate hull throws away two thirds
of the rejection information. And with the uniform predictor the *inner* hull
on `sliver` and `smooth_.95` is 1.04–1.34× M3, i.e. wider than an honest exact
band, which is what a predictor that accepts nearly everything looks like.

### The responsible reading

The predictive loss is solved: a fixed mixture over seven binormal AUCs costs
about one nat and a fitted sequential predictor adds nothing. The cell loss is
not close to solved, and refinement is not the answer — going from 12 to 36
cells moved the gap by three orders of magnitude and left five to go. The gate
passed on the optimistic diagnostic and the prototype it authorized
immediately produced the whole unit square, which is the cleanest possible
demonstration that the finite-library hull and the certifiable object are not
the same thing. The factor-of-two-versus-M3 that the inner hull advertises is
the prize; the gap between it and 0.986 is the entire research problem, and it
is a cell-bounding problem, not a search problem.

---

## 4. Certified outer projection of rank tests

### The uncertainty

Direct rank-test inversion is exact in principle and the main research
candidate; what nobody had measured is what a *certified outer* enclosure
actually costs in width. The trick that makes it tractable is to use statistics
that really are monotone under quantile coupling, so that a knot box's
worst-case p-value is attained at one of its two extreme step completions. The
question is whether the staircase you are forced to enclose leaves a band
anybody would use.

### How it was measured, and what that costs

Three fixed nonnegative weight vectors on the positives-before-negative counts,
measuring early, whole-path and late mass, each with both inclusive tails, on a
total Bonferroni budget of α/6, plus a whole-path-only α/2 ablation. At total
sizes 4, 6 and 8 every rank path is enumerated, step-law probabilities are
computed in exact rational arithmetic, and the outer search starts from the
entire ordered cube on knots .25/.5/.75, keeping every unresolved box. The run
asserts, for every observed path and every candidate in a dense rational
lattice plus diagonal, binormal, jump and sliver, that the outer envelope
contains each curve the exact test accepts — and it never fired. At
n₀ = n₁ = 25 the same bounding argument runs with common random numbers and
plus-one Monte Carlo p-values at B = 199, fixed through each box search.

The cost is stated in the spec and confirmed here: three interior knots is very
coarse, and this outer band contains the Monte Carlo confidence set, which is
not the exact-null confidence set.

### What was found

**The certificate holds and the width is not there.** Probability-weighted
coverage is at or above nominal for every enumerated case, and the exact test's
own error never exceeds α. But at total size 8, α = .05, the outer band's
probability-weighted area is .97–1.00 of M3's, and at α = .5 with the
three-statistic budget it is .99–1.05 — sometimes *wider* than M3. Of the 63
boxes the search opens, a mean of 0.0 to 2.2 are ever rejected, leaving 59.6 to
64.0 on the pending list. The retained volume proxy runs .203–.234, against the
.2344 that the same search returns when nothing is rejected at all.

At n₀ = n₁ = 25 with B = 199 the picture is worse in absolute terms:

| shape | α | outer area | M3 area | ratio | unresolved of 127 |
|---|---:|---:|---:|---:|---:|
| diagonal | .05 | .976 | .807 | 1.21 | 126 |
| normal_0.8 | .05 | .838 | .705 | 1.19 | 112 |
| jump | .05 | .963 | .710 | 1.36 | 124 |
| diagonal | .5 | .905 | .643 | 1.41 | 118 |
| jump | .5 | .899 | .538 | 1.67 | 115 |

Nineteen to sixty-seven percent wider than M3, at about half a second per band.

**The three-statistic budget usually loses to the ablation, with one telling
exception.** At α = .5 the whole-path-only α/2 version is narrower nearly
everywhere — the Bonferroni cost of the extra two statistics exceeds their
geometric information. The exception is `jump` at n = 25, α = .5, where the
region-sensitive version gives both higher coverage (.969 vs .812) and much
less width (.714 vs .899). Region statistics pay exactly where the ROC has
local structure the whole-path statistic averages away.

### The responsible reading

Soundness was never the doubt and the run confirms it constructively. What the
screen adds is a number for the projection-resolution cost, and it is the whole
story: a three-knot staircase enclosure cannot be narrower than M3, because the
staircase completion of a coarse box is itself as wide as the band you are
trying to beat. This does not condemn rank-test inversion; it says the width
question is entirely about knot resolution and box-search efficiency, and that
any future experiment reporting outer widths at three knots is measuring the
grid, not the method. The one substantive design finding is that region
statistics earn their multiplicity only on shapes with local structure, which
argues for choosing the statistic family from the data-generating regime rather
than fixing it globally.

---

## 5. M3 deterministic boundary optimization

### The uncertainty

M3 is the exact reference and will stay in the paper regardless. Its marginal
order-statistic boundaries, though, are a free choice: nothing forces the ELL
boundary, an equal class split, or the absence of a tail modifier. Frey's
criterion is to optimize width directly; the Dümbgen–Wellner tail/centre
tradeoff suggests an iterated-log family worth trying. The question is whether
any of that buys ROC-projection width without paying for it in a tail.

### How it was measured, and what that costs

A three-parameter family: interpolation η between the ELL and KS boundaries, a
symmetric iterated-log tail modifier θ, and a class allocation exponent s giving
ρ = n₀^(−s)/(n₀^(−s) + n₁^(−s)). 27 members plus the unmodified M3 anchor. For
each member a common width scale is calibrated by bisection so the *actual*
joint non-crossing probability meets its class target — interpolation happens
in boundary values before monotone tightening, and calibration applies to the
final event, so every candidate carries its own numerical certificate. Training
on four smooth binormal shapes, boundaries frozen to JSON before any held-out
observation is drawn, then nine held-out shapes × 128 replicates at five count
designs and three α. 17,280 held-out units, all complete.

The selection rule is the cost worth naming: a candidate is eligible only if
*each* training shape's mean left and right tail width is ≤ 1.00 × M3. That is
a hard constraint with no tolerance, and it turns out to be what decides the
whole experiment.

### What was found

**In all fifteen count/α designs the winner is η = 0, θ = 0, s = 0 — that is,
M3 itself.** Across all 135 held-out shape/design cells the paired area ratio
lies in [0.999998, 1.000000] and both tail ratios in [0.999997, 1.000000]: the
bisection recovers the ELL boundary to within floating-point noise. The
calibrated winner's joint non-crossing probability matches the baseline's to
six digits (.950000 vs .950003 at 100/100, α = .05).

That is not because the family contains nothing better on area. It contains
several things better on area, and every one of them loses a tail:

| design | best candidate by area | area | left tail | right tail | eligible |
|---|---|---:|---:|---:|---|
| 50/450, α = .5 | η0 θ.5 s.5 | .938 | 1.014 | 1.242 | no |
| 500/500, α = .5 | η0 θ.5 s0 | .961 | 1.129 | 1.216 | no |
| 450/50, α = .05 | η0 θ0 s.5 | .984 | .990 | .954 | **no** |
| 500/500, α = .05 | (none below 1) | 1.000 | 1.000 | 1.000 | — |

The 450/50 row is the interesting one. The class-split exponent s = 0.5 gives a
1.2–2.4% area saving on *every* training shape, with both tails below M3 on
three of the four — and it is vetoed because on the AUC .95 training shape its
left tail ratio is **1.0009**. Nine parts in ten thousand. So the negative
result at that design is a knife-edge, not a rout, whereas at 500/500 the
alternatives are genuinely worse (1.023 area for the best θ variant).

The track also produced a clean measurement of what everyone already believed
about M3's calibration. Held-out coverage across the 45 shape/design cells runs
.984–1.000 at α = .05, .977–1.000 at α = .2, and **.922–1.000 at α = .5**
against a nominal .5.

### The responsible reading

Within this family, M3's own boundary is the constrained optimum for ROC
projection width, at every size and every α tested. The iterated-log tail
modifier does what its name suggests — it moves width out of the centre and
into the tails — which is the wrong direction for a criterion that refuses to
pay in either tail. That lowers the priority of the deterministic-boundary
route in `next_method_ideas.md` §4.2 fairly decisively.

The class-split exponent is the one thread that should not be cut. It is the
only lever that reduced area *and* both tails on most shapes, at the design
where imbalance is severe and in the direction (majority negatives) where you
would expect it to help. It failed a hard constraint by less than a tenth of a
percent on one of four training shapes. That is worth one targeted re-run with
a larger training set and a stated tolerance, not a family-wide search.

And the α = .5 numbers should be read as a reminder rather than a result: M3
delivering .92–1.00 coverage at a nominal .5 is a width tax that no boundary
tweak within this family touches. Central-α calibration is not a
boundary-shape problem.

---

## 6. Things we did not anticipate

**Eligibility for the interior window is a strongly informative event, and it
points in opposite directions on different shapes.** This is the most important
unanticipated result. For `normal_0.95` at n = 500, n₀ = 900, the floor band
covers .983 on the 1,995 ineligible datasets and **.400 on the 5 eligible
ones**; at n = 500 balanced, α = .5, it is .774 ineligible against **.000 on 20
eligible**. A binormal at AUC .95 has a long trailing all-negative run, so the
window clears only when that run comes out anomalously short — which is exactly
the sample where the band is in trouble. But the sign is not universal: for
`kink` and `t2_0.95` at α = .5 the same conditioning selects *better* datasets
(+.20 and +.22). Any construction that switches behaviour on a rank-selected
event inherits a conditional law that can be badly worse or modestly better
than the marginal one, and we now have both signs measured in the same study.

**A flat coverage ladder reads as "C = 8 is safe" and means "nothing
happened."** When the window is ineligible the interior kernel never fires and
every rung returns the same floor band, so the operational C\* is reported as
right-censored at the top of the ladder with a conservative lower bound of 8.0.
That is the artifact the spec warned about, and it appears in **26 of the 50
operational cells** at α = .05 — every cell at n = 100, most of n = 500, and
`normal_0.95` at n = 5,000. It is honest as recorded, but it would be very easy
to read a table of C\* values and conclude the opposite of what happened.

**The hybrid was never wider than M3, in 162 cells.** The spec explicitly
anticipated the opposite — a regional hull can in principle exceed the band it
hulls with. It never did, not once, at any size, shape or α. Worth knowing
before anyone spends effort guarding against it.

**Stage F's left rule is α-blind, and it costs about four grid points at
α = .5.** `ceil(log(M+1))` depends only on the cloud budget, so it protected
exactly 8 columns in every single one of the 40,500 small-sample datasets,
whether we asked for 95% or 50% coverage. The exact rule, keyed to the realized
trim depth, protects a mean of 3.0–3.8 at α = .5 — and the α = .5
hybrid is 4.5% narrower for it, at coverage .888 against a nominal .5. The
alpha-independence noted in the Stage F report as an implementation property
turns out to have a measurable width cost.

**The fitted sequential predictor never beat the fixed mixture.** Fitted on
4,096 independent draws with a Beta-smoothed prefix table, it was a shade worse
than the fixed seven-component mixture on every truth, size and α. The obvious
next step in that direction — a better-fitted predictor — is not where the gain
is.

**At n = 500, α = .05, the interior window misses eligibility by one grid
column.** The exact left cutoff lands at FPR .0215–.0219 against a guard at
.0200. In the balanced direction at that size, **6,183 datasets have a
completely clear right side and are held out by the left guard alone** — the
whole of `interior_sliver`, 99% of `kink`, 97% of `t2_0.95`. Shift either the
window or the cutoff rule by one column and all of them flip from "no exposure"
to "estimable". Worth knowing not because .02 is wrong, but because a threshold
this tight means the eligibility rate at moderate n is not a stable property of
the method.

**The left cutoff depends on the shape, through the realized depth.** At
n = 500 balanced, α = .05, `sliver` has a mean realized trim depth of 3.75
while the other four shapes sit at 2.07–2.67, and a higher depth drives the
binomial cutoff *down* — to FPR .0185, inside the guard. So `sliver` is the one
shape substantially eligible at that size and α (41.8%), for a reason that has
nothing to do with its ROC geometry and everything to do with how its cloud
trims. The floor's left edge is data-adaptive in a way that is easy to forget.

**The study's cloud budget put the α = .05 band into its fallback regime.** At
M = 2,000–4,000 the realized trim depth on eligible datasets has cell means of
1.99–6.42, with sixteen of twenty-three cells below 3.2 and every n = 50,000
cell at 2.0 — at or under the resolution floor of three, where the band sits
near the conservative full envelope of its own cloud. Production's automatic
rule would have used 2.5–4.5× that budget (5,158 at n₀ = 500 against 2,000;
17,874 at n₀ = 50,000 against 4,000). This does not overturn the interior
result — it strengthens it,
since a tighter production band would show even less interior surplus — but it
does mean the α = .05 interior coverage numbers are systematically conservative
and should not be quoted as production behaviour.

**`jump` is not a small-sample problem.** Raw C = 1 coverage on the
step-function ROC gets *worse* with sample size: .238 at 10/10, .184 at 20/20,
.136 at 30/30, .108 at 50/50. The whole failure sits in the right-floor region
(left-region coverage 1.000 at 50/50), and it repairs completely under the
hybrid. A discontinuous ROC is not something more data fixes.

**Three of the five tracks finished in under five minutes.** The likelihood,
projection and m3 tracks used 2.9, 4.6 and 1.2 minutes against allocations of
45, 30 and 30. Nearly all of the six-hour budget was consumed by one track, and
the runner does not reallocate unused time. At the observed 9.2 units/min for
n = 50,000, the 96 minutes the other three left on the table would have bought
about 900 more units — not the block, but nearly three times what the screen
actually got there. The allocation is worth revisiting before the
decision-gating simulation.

---

## 7. Where this leaves the shortlist

No method here is promoted. The interior and M3 tracks both wrote
`eligible: false` and fell back to their baselines. The likelihood track's gate
did pass — for the mixture and sequential predictors at α = .05 — and the only
thing it authorized was the rational box prototype, which then returned the
unit square. Read against `next_method_ideas.md` §3, the update is:

- **Fiducial band with localized exact protection** remains the empirical
  incumbent, and the exact tail rules are a small, real improvement on Stage F's
  heuristics — carry them forward, mostly for α-adaptivity. The interior
  exponent is not a source of width after flooring, and the fixed-window gate
  should be replaced by something defined against the realized floor.
- **M3 with optimized deterministic boundaries** drops in priority. Within a
  27-member family at five count designs and three α, M3 is the constrained
  optimum. The class-split exponent deserves one targeted follow-up; the
  iterated-log tail modifier does not.
- **Rank-likelihood inversion** has a solved predictive loss and an unsolved
  cell-bounding loss five orders of magnitude from usable. The inner hull's
  factor-of-two versus M3 is real as a target and not as a result.
- **Direct rank-test inversion** is confirmed sound and confirmed expensive at
  coarse knots. Its width question is a knot-resolution question, and future
  experiments should be designed to measure that rather than to re-measure
  soundness.
- **Small samples** (n ≤ 50 per class) are effectively an M3 regime today. If
  that matters, it is a new-construction problem, not a floor-tuning problem.

The interior design should be finished at n = 50,000 in both imbalanced
directions before anything from track 4 enters the decision-gating simulation,
and it should be re-run at production cloud budgets.
