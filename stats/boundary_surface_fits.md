# Boundary surface — thin plate and Gaussian process refits

*Training: 95 LHS cells x 125 reps. Holdout: 11 classification-grade student-t anchors (14,000 reps), used in no fit. Estimand: C = 1 coverage at alpha = .05; bar = .94.*

## Fitted models

- **logistic-linear**: b = (2.58, 0.211, 0.446, -0.752) on (1, log n, log df, probit AUC), sign-constrained.
- **tprs**: rank 49 thin plate basis, lambda = 0.0794 by UBRE, effective df = 24.8.
- **gp**: ARD Matern-5/2, lengthscales (log n, log df, probit AUC) = (1.05, 0.67, 1.41) in SD units, amplitude 0.38, log marginal likelihood -1367.4.

## Holdout: the 11 anchors

Deviance per replicate is the proper score (lower is better). `worst opt.` is the largest overstatement of measured coverage — the anti-conservative direction that matters for routing.

| model | dev/rep | RMSE | MAE | bias | worst opt. | in CI |
|---|---|---|---|---|---|---|
| logistic-linear | 0.0910 | 0.115 | 0.076 | +0.056 | +0.265 | 3/11 |
| tprs | 0.0488 | 0.089 | 0.058 | +0.039 | +0.210 | 4/11 |
| gp | 0.0370 | 0.076 | 0.049 | +0.025 | +0.195 | 3/11 |

### Per-anchor predictions

| anchor | measured [Wilson 95%] | logistic-linear | tprs | gp |
|---|---|---|---|---|
| df 1.1 / AUC 0.97 / n 250 | 0.903 [0.883, 0.920] | 0.915 | 0.886 | 0.901 |
| df 1.1 / AUC 0.97 / n 500 | 0.956 [0.941, 0.967] | 0.926~ | 0.944 | 0.950 |
| df 1.1 / AUC 0.97 / n 1000 | 0.975 [0.963, 0.983] | 0.935~ | 0.958~ | 0.960~ |
| df 1.1 / AUC 0.97 / n 2500 | 0.980 [0.969, 0.987] | 0.946~ | 0.942~ | 0.939~ |
| df 2 / AUC 0.95 / n 150 | 0.846 [0.822, 0.867] | 0.938! | 0.900! | 0.854 |
| df 2 / AUC 0.95 / n 250 | 0.916 [0.897, 0.932] | 0.944! | 0.925 | 0.895~ |
| df 2 / AUC 0.95 / n 350 | 0.941 [0.932, 0.949] | 0.948 | 0.945 | 0.930~ |
| df 2 / AUC 0.99 / n 250 | 0.645 [0.615, 0.674] | 0.910! | 0.820! | 0.766! |
| df 2 / AUC 0.99 / n 500 | 0.690 [0.661, 0.718] | 0.921! | 0.900! | 0.885! |
| df 2 / AUC 0.99 / n 1000 | 0.842 [0.818, 0.863] | 0.931! | 0.925! | 0.924! |
| df 2 / AUC 0.99 / n 2500 | 0.951 [0.941, 0.960] | 0.942 | 0.928~ | 0.914~ |

`!` = optimistic and outside the interval (unsafe direction); `~` = pessimistic and outside.

## Leave-one-cell-out on the LHS sweep

Interpolation quality inside the design, refitting each fold.

| model | dev/rep | RMSE | MAE | bias | worst opt. | in CI |
|---|---|---|---|---|---|---|
| logistic-linear | 0.0206 | 0.032 | 0.016 | +0.000 | +0.259 | 83/95 |
| tprs | 0.0177 | 0.030 | 0.015 | +0.002 | +0.254 | 86/95 |
| gp | 0.0489 | 0.032 | 0.016 | +0.000 | +0.261 | 83/95 |

## Monotonicity (imposed on the baseline, diagnostic on the rest)

| model | in n | in df | in AUC |
|---|---|---|---|
| logistic-linear | 100% | 100% | 100% |
| tprs | 55% | 82% | 88% |
| gp | 51% | 63% | 91% |

## Boundary contour n*(df, AUC) at the .94 bar

GP columns give the posterior mean and, in brackets, the 10% posterior quantile — the conservative read the routing decision would use.

| df | lin .90 | lin .95 | lin .99 | tprs .90 | tprs .95 | tprs .99 | gp .90 | gp .95 | gp .99 |
|---|---|---|---|---|---|---|---|---|---|
| 1.1 | 173 | 630 | >2500 | 147 | 338 | >2500 | <100 [247] | 304 [>2500] | >2500 [>2500] |
| 1.5 | <100 | 327 | >2500 | 157 | 331 | >2500 | 153 [254] | 340 [>2500] | >2500 [>2500] |
| 2 | <100 | 178 | 2020 | 154 | 322 | >2500 | 210 [300] | 394 [>2500] | >2500 [>2500] |
| 3 | <100 | <100 | 858 | <100 | 250 | >2500 | 213 [359] | 404 [>2500] | >2500 [>2500] |
| 30 | <100 | <100 | <100 | <100 | <100 | <100 | <100 [<100] | <100 [<100] | <100 [<100] |
