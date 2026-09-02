# The C = 1 coverage boundary: active-learning study

*Generated 2026-09-01 from `data/results/c_calibration_followup_20260830/`.
257 student-t cells, 64,625 replicates, ~14 CPU-hours. Estimand throughout:
C = 1 coverage at alpha = .05; bar = .94.*

## 1. What this overturned

The pre-registered surface (spec follow-up item 1b) is a logistic smooth in
(log n, log df, probit AUC) with sign constraints encoding a monotone tail-mass
mechanism: coverage nondecreasing in n and df, nonincreasing in AUC. **The
constraint in n is false.** At a fixed shape, t(4.69)/AUC .986 measures

| n | 150 | 400 | 800 | 1200 | 2000 |
|---|---|---|---|---|---|
| coverage | .993 | .947 | .903 | .823 | .847 |

Coverage *falls* with sample size, with disjoint Wilson intervals at the ends.
The pre-registered model is therefore misspecified in a way refitting cannot
repair, and the monotonicity violations of the thin-plate and GP fits
(51-55% monotone in n) were signal rather than defect.

## 2. The governing coordinate, and where it stops working

Let `t_q` be the FPR at which the true ROC reaches TPR = q, and

    m = n0 * t_q

the expected number of negatives scoring above the q-th quantile of the
positives — at q = .5, the count of negatives above the median positive.
Coverage failures concentrate in a *window* of m rather than below a threshold
in n, which is what produces the apparent non-monotonicity: n moves a shape
through the window, and different shapes traverse it in opposite directions.

    t(2)/.99      m = 1.2, 2.4, 4.8, 12.0   at n = 250..2500   cov .645 -> .951  (exiting)
    t(4.69)/.986  m = 0.6, 1.7, 3.4,  5.1,  8.5 at n = 150..2000  cov .993 -> .847  (entering)

On the first 194 cells every failure fell in m in [0.89, 11.2], and the window
held on 28 out-of-sample cells from round 1. **Round 2 falsified it**: extending
n to 5,000-7,000 produced failures at m = 17.6 (n = 6656, cov .852) and
m = 30.2 (n = 5131, cov .936). The window's upper edge grows with AUC, so m
compresses the boundary but does not linearize it. Across all 257 cells the
tightest variant is q = .3, whose failures span m in [0.71, 21.8].

A trim-grid artifact was considered and **rejected**: the production rule thins
the min-p grid at K = n0 + 1 > 2001, but a dedicated probe at fixed shape
(t(6.62)/.9883, 400 reps) found coverage .930 at n = 1900 vs .907 at n = 2100 —
1.2 SE, no discontinuity — and the same shape already fails at n = 1500 where no
thinning occurs.

## 3. The boundary in (AUC, n)

The unsafe set is a wedge that opens as AUC rises, worst-cased over df:

| AUC band | cells | failures | n-range of failures |
|---|---|---|---|
| .50-.90 | 65 | 1 | 102 |
| .90-.94 | 53 | 15 | 103 - 248 |
| .94-.96 | 35 | 10 | 110 - 452 |
| .96-.975 | 36 | 14 | 124 - 1051 |
| .975-.985 | 26 | 5 | 199 - 5131 |
| .985-1.0 | 42 | 20 | 160 - 6656 |

At AUC >= .975 no tested sample size up to 6,656 is safe.

## 4. A conservative routing heuristic

Routing is admissible adaptivity because mis-routing to M3 costs width, never
coverage, so the rule reads an *upper* confidence bound on AUC and errs toward
M3.

    AUC_ub <  .88             -> fiducial band at any n
    AUC_ub <  .96             -> fiducial band iff n > 600
    AUC_ub <  .975            -> fiducial band iff n > 1500
    AUC_ub >= .975            -> route to M3

Validated on all 257 cells: **72 routed to the fiducial band, zero failures,
minimum coverage .944**. The cost is over-routing — 65% of the cells sent to M3
would have been fine, and M3 is 39-49% wider.

Two limits on that 28% retention figure. The cell library is not a realistic
workload: active learning deliberately oversampled the boundary, so a real
AUC distribution would keep the narrow band far more often. And the thresholds
were read off the same data they are validated on; these cells stress the rule
but do not confirm it. Fresh classification-grade confirmation cells (spec
item 5) remain necessary before the rule enters guidance.

## 5. Learning trajectory

Four greedy straddle rounds, 2.5 CPU-hours each, each scored prospectively on
cells selected before they were run:

| round | cells | CPU-h | prospective dev/rep | RMSE | bias | boundary shift (log n) |
|---|---|---|---|---|---|---|
| 1 | 28 | 2.49 | .0085 | .022 | -.005 | - |
| 2 | 22 | 2.49 | .0169 | .030 | -.003 | .245 |
| 3 | 19 | 2.50 | .0052 | .017 | +.002 | .067 |
| 4 | 17 | 2.50 | .0144 | .027 | -.000 | .015 |

**Prospective error is not a learning curve here.** Each round's test set is
chosen adversarially against the current model, so difficulty rises with
competence and the sequence is non-monotone by construction. The interpretable
convergence signal is the movement of the estimated boundary on a *fixed*
grid, which fell .245 -> .067 -> .015 log-n: by round 4 the boundary had
effectively stopped moving, and further rounds at this budget would buy little.

The one caveat is that round 2's counterexample arrived only when the n range
was extended. Convergence is convergence *within the sampled box*; it says
nothing about regions the design has never been allowed to enter.

## 6. What to run next

1. Confirmation cells at the three proposed cutoffs (AUC .88/.96/.975 and
   n 600/1500), fresh names and seeds, sequential replication as in item 1a.
   The current thresholds are read off stress-test data and are not certified.
2. Extend n above 8,000 at AUC >= .985 to find whether the wedge ever closes,
   or establish that it does not within any practical n.
3. Cross-family checks: everything here is student-t. The wedge should be
   re-derived on Weibull/gamma/beta-opposing before being quoted suite-wide.
4. Replace `AUC_ub` with a direct estimate of m (count negatives above the
   median positive) once the m-window's AUC dependence is characterized; it is
   the more mechanistic statistic and needs no shape model.
