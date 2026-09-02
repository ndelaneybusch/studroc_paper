# C-calibration follow-up runs — report

*Generated 2026-09-01 12:11 from `data\results\c_calibration_followup_20260830`. Design: the dated follow-up entry in `stats/c_calibration_spec.md` (revised 2026-08-31). Cell verdicts use the A1-letter bar: PASS = point >= 0.94 AND Wilson-95 lower >= 0.925 at alpha = .05 (C = 1 arm); MARGINAL = CI still straddles the bar at the replication cap. All claims are library-relative.*

## 1. Validity boundary

**Anchors and spot checks (classification-grade):**

| cell | reps | C=1 cov @ .05 [Wilson 95%] | verdict | >= .95 | M3 cov | M3/C1 area |
|---|---|---|---|---|---|---|
| boundary--bo05_99--n100 | 1000 | 0.992 [0.984, 0.996] | PASS | yes | 1.000 | 2.18x |
| boundary--bo05_99--n250 | 1000 | 0.983 [0.973, 0.989] | PASS | yes | 0.999 | 1.99x |
| boundary--gm05_93--n100 | 1000 | 0.977 [0.966, 0.985] | PASS | yes | 1.000 | 1.59x |
| boundary--gm05_93--n250 | 1000 | 0.985 [0.975, 0.991] | PASS | yes | 1.000 | 1.52x |
| boundary--t11_97--n1000 | 1000 | 0.975 [0.963, 0.983] | PASS | yes | 1.000 | 1.26x |
| boundary--t11_97--n250 | 1000 | 0.903 [0.883, 0.920] | FAIL | no | 1.000 | 1.38x |
| boundary--t11_97--n2500 | 1000 | 0.980 [0.969, 0.987] | PASS | yes | 0.999 | 1.22x |
| boundary--t11_97--n500 | 1000 | 0.956 [0.941, 0.967] | PASS | yes | 0.998 | 1.30x |
| boundary--t2_95--n150 | 1000 | 0.846 [0.822, 0.867] | FAIL | no | 0.998 | 1.42x |
| boundary--t2_95--n250 | 1000 | 0.916 [0.897, 0.932] | FAIL | no | 0.999 | 1.38x |
| boundary--t2_95--n350 | 3000 | 0.941 [0.932, 0.949] | PASS | no | 0.998 | 1.35x |
| boundary--t2_99--n1000 | 1000 | 0.842 [0.818, 0.863] | FAIL | no | 1.000 | 1.34x |
| boundary--t2_99--n250 | 1000 | 0.645 [0.615, 0.674] | FAIL | no | 1.000 | 1.68x |
| boundary--t2_99--n2500 | 2000 | 0.951 [0.941, 0.960] | PASS | yes | 0.998 | 1.27x |
| boundary--t2_99--n500 | 1000 | 0.690 [0.661, 0.718] | FAIL | no | 0.999 | 1.47x |
| boundary--wb05_99--n100 | 1000 | 0.988 [0.979, 0.993] | PASS | yes | 0.999 | 2.06x |
| boundary--wb05_99--n250 | 1000 | 0.987 [0.978, 0.992] | PASS | yes | 0.999 | 1.80x |

**Routing threshold (library-relative).** Per shape, the smallest tested n whose cell and all larger tested n PASS:

- bo05_99: passes from n = 100 up.
- gm05_93: passes from n = 100 up.
- t11_97: passes from n = 500 up.
- t2_95: passes from n = 350 up.
- t2_99: passes from n = 2500 up.
- wb05_99: passes from n = 100 up.

**Global routing threshold (worst tested shape): min(n0, n1) >= 2500.** This is library-relative — valid over the tested shapes, not distribution-free; shapes outside the library can move it.


**Boundary surface (LHS sweep, 95 cells × ~125 reps).** Sign-constrained logistic smooth logit(cov) = b0 + b1·log n + b2·log df + b3·probit(AUC): b = (2.58, 0.211, 0.446, -0.752). The n*(df, AUC) contour at the .94 bar — point fit, with the conservative 90% bootstrap quantile (over cells) in brackets:

| df \ AUC | 0.9 | 0.95 | 0.99 |
|---|---|---|---|
| 1.1 | 173 [577] | 630 [1578] | >2500 (extrap.) [>2500 (extrap.)] |
| 1.5 | <100 [362] | 327 [845] | >2500 (extrap.) [>2500 (extrap.)] |
| 2 | <100 [225] | 178 [581] | 2020 [>2500 (extrap.)] |
| 3 | <100 [115] | <100 [294] | 858 [2150] |
| 30 | <100 [<100] | <100 [<100] | <100 [<100] |

**Holdout check (anchors vs fitted surface):**
- boundary--t11_97--n250: measured 0.903 [0.883, 0.920], fitted 0.915 — consistent
- boundary--t11_97--n500: measured 0.956 [0.941, 0.967], fitted 0.926 — OUTSIDE CI
- boundary--t11_97--n1000: measured 0.975 [0.963, 0.983], fitted 0.935 — OUTSIDE CI
- boundary--t11_97--n2500: measured 0.980 [0.969, 0.987], fitted 0.946 — OUTSIDE CI
- boundary--t2_95--n150: measured 0.846 [0.822, 0.867], fitted 0.938 — OUTSIDE CI
- boundary--t2_95--n250: measured 0.916 [0.897, 0.932], fitted 0.944 — OUTSIDE CI
- boundary--t2_95--n350: measured 0.941 [0.932, 0.949], fitted 0.948 — consistent
- boundary--t2_99--n250: measured 0.645 [0.615, 0.674], fitted 0.910 — OUTSIDE CI
- boundary--t2_99--n500: measured 0.690 [0.661, 0.718], fitted 0.921 — OUTSIDE CI
- boundary--t2_99--n1000: measured 0.842 [0.818, 0.863], fitted 0.931 — OUTSIDE CI
- boundary--t2_99--n2500: measured 0.951 [0.941, 0.960], fitted 0.942 — consistent

*The contour is PROVISIONAL (a smooth over a 2-D family slice; smoothing bias is largest exactly at the contour). Candidate routing cutoffs read from it are confirmed classification-grade by follow-up item 5 (cutoff confirmation) before any routing guidance is frozen; the surface's other product — per-stratum coverage predictions for the final suite's student_t strata — needs no confirmation to be useful as predictions.*

## 2. Held-out validation of C = 1

| cell | reps | C=1 cov @ .05 [Wilson 95%] | verdict | >= .95 | M3 cov | M3/C1 area |
|---|---|---|---|---|---|---|
| heldout--bimodal_80_sep15--n500 | 2000 | 0.982 [0.975, 0.987] | PASS | yes | 1.000 | 1.48x |
| heldout--bimodal_80_sep15--n5000 | 2000 | 0.967 [0.958, 0.974] | PASS | yes | — | — |
| heldout--binormal_85--n500 | 2000 | 0.982 [0.975, 0.987] | PASS | yes | 1.000 | 1.49x |
| heldout--binormal_90_q20--n1000 | 2000 | 0.984 [0.977, 0.988] | PASS | yes | 1.000 | 1.44x |
| heldout--heterologit_88_r2--n500 | 2000 | 0.987 [0.980, 0.991] | PASS | yes | 1.000 | 1.48x |
| heldout--lhs1_student_t--n500 | 2000 | 0.978 [0.971, 0.984] | PASS | yes | 1.000 | 1.46x |
| heldout--lhs2_weibull--n500 | 2000 | 0.982 [0.975, 0.987] | PASS | yes | 1.000 | 1.48x |
| heldout--lhs2_weibull--n5000 | 2000 | 0.969 [0.960, 0.976] | PASS | yes | — | — |
| heldout--t3_90--n500 | 2000 | 0.973 [0.964, 0.979] | PASS | yes | 1.000 | 1.39x |
| heldout--t3_90--n5000 | 2000 | 0.970 [0.961, 0.976] | PASS | yes | — | — |

Worst held-out C=1 coverage at alpha=.05: 0.967. All cells PASS.

## 3. Composite-band derisk (finite-range)

**Parity (full-curve C=1 arm vs Stage S):**
- composite--binormal_60--n500x500: 0.983 vs Stage S 0.987 — ok
- composite--binormal_95--n5000x5000: 0.968 vs Stage S 0.972 — ok
- composite--binormal_95--n500x500: 0.976 vs Stage S 0.982 — ok
- composite--kink_80--n500x500: 0.967 vs Stage S 0.973 — ok
- composite--t2_95--n100x100: 0.804 vs Stage S 0.802 — ok
- composite--t2_95--n5000x5000: 0.963 vs Stage S 0.968 — ok
- composite--t2_95--n500x4500: 0.947 vs Stage S 0.950 — ok
- composite--t2_95--n500x500: 0.963 vs Stage S 0.958 — ok
- composite--trapezoid_q10_90--n500x500: 0.976 vs Stage S 0.981 — ok

Candidate rule: every cell PASSes the coverage bar AND the pooled paired width change vs the full-curve C=1 band is negative. 'No survivor' is evidence against this coarse (cut x C) family only.

| config | worst cell verdict | min cov | pooled dArea (paired SE) | candidate |
|---|---|---|---|---|
| b0.02-0.95_C1.5 | FAIL | 0.886 | -1.8% (0.01pp) | no |
| b0.02-0.95_C2 | FAIL | 0.864 | -4.8% (0.01pp) | no |
| b0.02-0.95_C2.5 | FAIL | 0.856 | -7.2% (0.01pp) | no |
| b0.05-0.9_C1.5 | FAIL | 0.918 | +2.4% (0.02pp) | no |
| b0.05-0.9_C2 | FAIL | 0.906 | -0.1% (0.02pp) | no |
| b0.05-0.9_C2.5 | FAIL | 0.892 | -2.0% (0.02pp) | no |
| b0.1-0.85_C1.5 | MARGINAL | 0.940 | +7.4% (0.03pp) | no |
| b0.1-0.85_C2 | MARGINAL | 0.926 | +5.5% (0.03pp) | no |
| b0.1-0.85_C2.5 | FAIL | 0.918 | +4.0% (0.03pp) | no |

**Large-n sentinels (n = 20,000; Theorem-7 erosion direction — outside the candidate range):**
- composite--binormal_95--n20000x20000 b0.05-0.9_C1.5: cov 0.976 [0.949, 0.989], dArea +2.2%
- composite--binormal_95--n20000x20000 b0.05-0.9_C2: cov 0.968 [0.938, 0.984], dArea +0.2%
- composite--t2_95--n20000x20000 b0.05-0.9_C1.5: cov 0.960 [0.928, 0.978], dArea +6.5%
- composite--t2_95--n20000x20000 b0.05-0.9_C2: cov 0.956 [0.923, 0.975], dArea +4.9%

Any surviving candidate is a *finite-range* result: the production form must clamp the interior exponent to 1 (or taper it) above the calibrated range — Theorem 7 forces interior coverage toward (1-alpha)^C for fixed C > 1. Per-cell / per-rep detail is in the `*.composite.json` files.
