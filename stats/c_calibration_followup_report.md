# C-calibration follow-up: the C = 1 coverage boundary

*2026-09-01. Reports the follow-up run plan of `stats/c_calibration_spec.md`
(items 1-3, plus work the results forced beyond it). Data:
`data/results/c_calibration_followup_20260830/`. 257 student-t cells,
64,625 replicates, ~14 CPU-hours. Estimand throughout: C = 1 coverage at
alpha = .05. Verdicts use the A1-letter bar — PASS iff point >= .94 and
Wilson-95% lower >= .925. All claims are library-relative.*

*Companions: `fiducial_band_theory.md` §7.2 (the mechanism),
`next_method_ideas.md` §5 item 1 and §7 (working model and backlog).*

---

## 0. Summary

The spec's item 1 fired the escalation clause it defined in advance: an
anchor fails at n = 500, so the shipped `min(n0, n1) >= 500` safety claim is
library-limited. Pursuing that finding falsified the mechanism the whole
pre-registered surface was built on — **coverage is not monotone in n** — and
relocated the unsafe set from a small-n half-space to a curved wedge in
(AUC, n) reaching past n = 6,000.

Four results carry forward:

1. **The routing boundary is a wedge, not a threshold** (§2, §4). A
   conservative rule with zero failures over all 257 cells is given in §5.
2. **Held-out validation passes but proves less than it appears to** (§3):
   the held-out library never reaches the failing region.
3. **The composite band has a candidate on a declared finite range** (§6).
4. **A localized M3 floor repairs the failures at ~1/5 the width cost of
   routing** (§7) — the most promising lead, and the least validated.

---

## 1. What was falsified

The pre-registered surface (spec item 1b) is a logistic smooth in
(log n, log df, probit AUC) with sign constraints encoding a monotone
tail-mass mechanism: coverage nondecreasing in n and df, nonincreasing in
AUC. **The constraint in n is false.** At a fixed shape, t(4.69)/AUC .986:

| n | 150 | 400 | 800 | 1200 | 2000 |
|---|---|---|---|---|---|
| coverage | .993 | .947 | .903 | .823 | .847 |

300 reps each; the Wilson intervals at the ends are disjoint. The model is
therefore misspecified in a way refitting cannot repair, and the
monotonicity violations of the flexible fits in §8 are signal, not defect.

A trim-grid artifact was considered and **rejected**. The production rule
thins the min-p grid at K = n0 + 1 > 2001 while the band is evaluated on the
full grid, which would predict a discontinuity there. A dedicated probe at
fixed shape (t(6.62)/.9883, 400 reps) measured .930 at n = 1900 against .907
at n = 2100 — 1.2 SE, no discontinuity — and the same shape already fails at
n = 1500 where no thinning occurs.

## 2. Item 1: the validity boundary

**Anchors (classification-grade).** `t2_99` fails badly and does not recover
until n = 2500; `t11_97` fails only at n = 250.

| shape | n=250 | n=500 | n=1000 | n=2500 |
|---|---|---|---|---|
| t2_99 | .645 | **.690** | **.842** | .951 PASS |
| t11_97 | .903 | .956 | .975 | .980 |

This is not a harness artifact. M3 covers .998-1.000 on the identical seeds
and truth curve; the ladder is nowhere near pinned (`min_j` 16-17,
`frac_j_below_3` = 0, unlike the Stage S t(2)/n=100 cell that was discounted
as boundary-pinned); and misses are almost entirely lower-edge
(`viol_low` .301 vs `viol_high` .013), the unseen-tail-mass mechanism of
theory doc §7.2a.

**Cross-family spot checks all pass** (.977-.992 at Weibull/gamma/
beta-opposing achievable corners), confirming the t-family is binding.

**Consequence.** The spec's trigger is met verbatim: theory doc §7.2(c) and
the `fiducial_band` docstring language need amending. The anchor-based
conservative threshold is `min(n0, n1) >= 2500` — but §4 supersedes it, since
an n-only threshold cannot express a wedge.

## 3. Item 2: held-out validation

All 10 held-out cells PASS, all clear the strict >= .95 bar, worst coverage
.967. No designer bias in the shipped default.

**The scope caveat matters more than the result.** The held-out library tops
out at AUC .90, and its heavy-tail members are mild — t(3)/.90 and the LHS
member at t(8.1)/.797. Nothing in it probes heavy tail x high AUC, which is
where the failures live. Quoted without that qualifier the sweep reads as
contradicting §2. It validates C = 1 **for AUC <= .90 at n = 500**, not at
n = 500 generally.

Side observations: alpha = .5 coverage runs .66-.77 against nominal .50
(conservative, as documented); M3 costs 39-49% width on these cells.

## 4. The boundary in (AUC, n)

Worst-cased over df across all 257 cells, the unsafe set is a wedge that
opens as AUC rises:

| AUC band | cells | failures | n-range of failures |
|---|---|---|---|
| .50-.90 | 65 | 1 | 102 |
| .90-.94 | 53 | 15 | 103 - 248 |
| .94-.96 | 35 | 10 | 110 - 452 |
| .96-.975 | 36 | 14 | 124 - 1051 |
| .975-.985 | 26 | 5 | 199 - 5131 |
| .985-1.0 | 42 | 20 | 160 - 6656 |

At AUC >= .975 no tested sample size up to 6,656 is safe.

**A partial mechanism.** Let `t_q` be the FPR at which the true ROC reaches
TPR = q, and `m = n0 * t_q` — at q = .5, the expected count of negatives
scoring above the median positive. Failures concentrate in a *window* of m
rather than below a threshold in n, which is what produces the apparent
non-monotonicity: n carries a shape through the window, and different shapes
traverse it in opposite directions.

    t(2)/.99      m = 1.2, 2.4, 4.8, 12.0        (n = 250..2500)   cov .645 -> .951  exiting
    t(4.69)/.986  m = 0.6, 1.7, 3.4, 5.1, 8.5    (n = 150..2000)   cov .993 -> .847  entering

On the first 194 cells every failure fell in m in [0.89, 11.2], and the
window held on 28 out-of-sample cells. **Round 2 of the design falsified it
as a universal rule**: extending n to 5,000-7,000 produced failures at
m = 17.6 (n = 6656, cov .852) and m = 30.2 (n = 5131, cov .936). The
window's upper edge grows with AUC. So m compresses the boundary but does
not linearize it — a real mechanism, an incomplete coordinate. Across all
cells the tightest variant is q = .3, whose failures span m in [0.71, 21.8].

The coordinate is estimable at runtime without knowing the truth (count
negatives scoring above the median positive), which makes it the more
mechanistic routing statistic once its AUC dependence is characterized.

**Superseded 2026-09-02 by a derived mechanism** (`fiducial_band_theory.md`
§7.4): the cloud's within-gap spreading is calibrated to a locally linear
ROC and fails at convex heavy-tail corners. A resolution-corrected endpoint
risk score screens 122 of the 257 cells with no observed sub-.94 coverage,
and for heavy tails `m ≈ n0 (1 − AUC) / 2`, the coordinate in which the
t-family's failure set is a tail-index-dependent window.

## 5. A conservative routing heuristic

Routing is admissible per-dataset adaptivity because mis-routing to M3 costs
width, never coverage, so the rule reads an *upper* confidence bound on AUC
and errs toward M3.

    AUC_ub <  .88   -> fiducial band at any n
    AUC_ub <  .96   -> fiducial band iff n > 600
    AUC_ub <  .975  -> fiducial band iff n > 1500
    AUC_ub >= .975  -> route to M3

Validated on all 257 cells: **72 routed to the fiducial band, zero failures,
minimum coverage .944.** Cost: 65% of the cells sent to M3 would have been
fine.

Two limits. The 28% retention figure is pessimistic about real workloads —
active learning deliberately oversampled the boundary, so a realistic AUC
distribution keeps the narrow band far more often. And the thresholds are
read off the same data that validates them; these cells stress the rule but
do not confirm it. Item 5's fresh confirmation cells remain required.

## 6. Item 3: the composite band

Parity holds on all 9 core cells (max 1.1 SE against Stage S), so the item is
not void. The generated report says "no survivor", but that verdict is driven
entirely by the n = 100 cell, and the spec frames item 3 as a *finite-range*
question. On the declared range n >= 500, four candidates survive:

| config | worst verdict | min cov | pooled dArea |
|---|---|---|---|
| **b0.02-0.95_C2.5** | PASS | .940 | **-6.8%** |
| b0.02-0.95_C2 | PASS | .949 | -4.5% |
| b0.02-0.95_C1.5 | PASS | .956 | -1.6% |
| b0.05-0.9_C2.5 | PASS | .950 | -1.2% |

By the spec's tie-break (largest saving) the pick is `b0.02-0.95_C2.5`. The
n = 20,000 sentinels show the saving inverting (+0.2% to +6.5% wider), which
supports clamping above the calibrated range as Theorem 7 requires.

Note the decision rule as implemented pools all core cells including n = 100,
so it can only ever report the unrestricted answer; the range's lower bound
is a judgment the code does not currently express.

## 7. A localized M3 floor (the strongest lead)

Beyond the routing boundary the safe play is full M3 at 28-46% width. That
loss is localized, and can be bought back.

**Where the misses are.** Replayed from the seeds (stored summaries record
only direction, not location). The misses are overwhelmingly at the *upper*
FPR end — peak pointwise miss rate at 1 - FPR ~ .002-.04, with ~70% of
missing replicates in the large-n cells having all their misses above
FPR = .9 — plus a secondary cluster at the extreme left corner (FPR <~ .005).
Mechanically: with heavy-tailed positives the true ROC approaches 1 slowly
while the band's lower edge, monotone and pinned to reach 1, overshoots it.
(Derived 2026-09-02, theory doc §7.4: the cloud spreads the positive mass
below the lowest *observed* positive uniformly over the negatives below it,
so its lower edge claims a TPR deficit of order `ln(1/ell) / (n1 k_sat)`
where the truth's is of order `1 / n1`; M3 uses the corresponding
no-interpolation bracketing principle and adds simultaneous control.)

**M3 covers at 100.0% of the fiducial's miss points** in all five probed
cells.

**The hybrid.** Taking the pointwise union with M3 on
`FPR in [0, 0.005] u [0.5, 1]` and keeping the C = 1 band elsewhere:

| cell | fiducial | hybrid | hybrid width | full M3 width |
|---|---|---|---|---|
| t(2)/.990/n500 | .720 | .985 | +4.9% | +46% |
| t(4.69)/.986/n1200 | .845 | .990 | +7.2% | +38% |
| t(6.62)/.988/n2600 | .890 | .980 | +9.0% | +38% |
| t(1.13)/.926/n130 | .915 | .955 | +4.4% | +36% |
| t(3.29)/.984/n5131 | .940 | .980 | +6.4% | +28% |

Worst-case coverage .955 at mean +6.4% width — roughly **one fifth the width
cost of routing to M3**.

Two structural points. The upper region is nearly free despite being half the
curve, because both bands are compressed against TPR = 1 there. And the
hybrid can never do worse than the fiducial band: the union is pointwise
wider, and since fiducial edges are already monotone the running-max closure
preserves that ordering, so hybrid coverage >= fiducial coverage
identically. That makes unconditional application defensible, not just
routing-gated use.

**Least validated result here.** Five cells at 100-200 reps (Wilson
half-widths ~ +/-.03), student-t only, and the region was selected on the
same cells that score it. `tau_lo = 0.005` is also a poor parametrization: at
n = 130 it spans 0.65 grid points and does nothing, so the left component
should be expressed in grid points (first 3-5).

## 8. Surface fits (item 1b, reworked)

The pre-registered logistic-linear fit was scored against the 11
classification-grade anchors alongside a thin-plate regression spline and a
binomial-likelihood GP, both fitted to the same 95-cell LHS sweep:

| model | holdout dev/rep | RMSE | worst optimistic | in CI |
|---|---|---|---|---|
| logistic-linear | .0910 | .115 | +.265 | 3/11 |
| thin plate (edf 24.8) | .0488 | .089 | +.210 | 4/11 |
| GP (Matern-5/2 ARD) | **.0370** | **.076** | +.195 | 3/11 |

Both flexible fits substantially improve, and the GP's hand-rolled Laplace
implementation was cross-checked against `sklearn`'s GPR on empirical logits
(mean absolute difference .015 on the anchors, same conclusions).

**None of them is usable for routing.** All three remain optimistic by 15-27
points at the t2_99 anchors — the cells that set the threshold. The binding
constraint was never the smoother: of the 95 LHS cells exactly one sat in the
cliff region, and leave-one-out at that cell moves the GP's prediction from
.70 to .96 under any number of restarts. No smoother recovers a
discontinuity from one noisy point.

## 9. Design method and convergence

The infill used the straddle acquisition of Bryan et al. (2005),
`a(x) = 1.96*sd(x) - |mean(x) - logit(.94)|` on the latent scale, with
batch-aware greedy selection (joint posterior downdated by each pick's
planned Laplace weight) and cost weighting per core-second against a runtime
model calibrated on the LHS sweep.

| round | cells | CPU-h | prospective dev/rep | RMSE | boundary shift (log n) |
|---|---|---|---|---|---|
| 1 | 28 | 2.49 | .0085 | .022 | - |
| 2 | 22 | 2.49 | .0169 | .030 | .245 |
| 3 | 19 | 2.50 | .0052 | .017 | .067 |
| 4 | 17 | 2.50 | .0144 | .027 | .015 |

**Prospective error is not a learning curve here.** Each round's test set is
chosen adversarially against the current model, so difficulty rises with
competence and the sequence is non-monotone by construction. The
interpretable signal is movement of the estimated boundary on a *fixed*
grid: .245 -> .067 -> .015 log-n. By round 4 the boundary had effectively
stopped moving.

The caveat is that round 2's counterexample arrived only because that round
extended the n range. Convergence is convergence *within the sampled box*.

## 10. What to run next

1. **Confirmation cells** at the §5 cutoffs (AUC .88/.96/.975, n 600/1500),
   fresh names and seeds, sequential replication as in item 1a. This is spec
   item 5 and it is still required; §5's thresholds are stress-tested, not
   certified.
2. **Validate the §7 hybrid properly** — fix the region, evaluate across a
   sample of the wedge including the worst known cell (t2_99 at n = 250,
   covering .645), plus safe-region cells to price the width cost where the
   fiducial band was already fine. ~2-3 CPU-hours for ~20 cells at 500 reps.
   If it holds, it is a better answer than routing and changes what the
   roster's fiducial entry should be.
3. **Extend n above 8,000 at AUC >= .985** to find whether the wedge closes
   or is unbounded in practice.
4. **Cross-family re-derivation.** Everything in §1, §4, §7 is student-t; the
   spot checks in §2 only establish that t-tails are binding at the corners
   tested.

## 11. Reproduction

| Artifact | Produces |
|---|---|
| `scripts/c_calibration/followup_runs.py` | items 1-3, `followup_report.md` |
| `scripts/c_calibration/boundary_surface_fits.py` | §8 |
| `scripts/c_calibration/boundary_active_design.py` | the first infill batch |
| `scripts/c_calibration/boundary_active_rounds.py` | §9 rounds, `learning_trajectory.json` |
| `scripts/c_calibration/boundary_diagnostics.py` | §1 ladders |
| `scripts/c_calibration/hybrid_band_probe.py` | §7, `hybrid_probe/` |

Cells resume and extend; seeding is deterministic in (study seed, stage, cell
name, rep), so every per-rep statistic in this report is recomputable by
replay without re-simulation.
