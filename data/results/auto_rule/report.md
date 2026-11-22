# Routing rule for `auto`: when the unfloored fiducial band is safe

## Summary

**The decision.** The `auto` mode chooses between two ROC confidence bands using only what it can see: the two class counts and the AUC. The *unfloored fiducial band* is narrow and fast. The *hybrid* adds an exact ELL floor at the two ends of the curve, which protects against the unfloored band's known failure: when the ROC has a convex hook in a corner that few observations reach, the unfloored band misses the true curve more often than its nominal 5%. Where the unfloored band is safe it is preferred, because it is narrower at small samples, much faster at large ones, and closer to nominal (Section 1). This report finds where it is safe.

**Recommended rule.** Use the unfloored band when all of the following hold, and the hybrid otherwise (`n_min` and `n_max` are the smaller and larger class counts):

- the smaller class has at least 160 cases.
- the AUC is low enough for the sample size: `n_min * (1 - AUC)^2 >= 18`. The left side grows with the expected number of minority cases beyond the ROC's corner, so the largest AUC allowed rises with the smaller class.

In one line: `n_min >= 160 and n_min * (1 - AUC)^2 >= 18`.

**What safe means.** A *miss* is a dataset whose band fails to contain the true ROC somewhere; at 95% confidence the target miss rate is 5%. At every setting the rule admits with a larger class of at most 10,000, the fitted model gives at least 90% probability that the miss rate is at most 5%. This holds for the worst score distribution in the two families known to break the unfloored band, Student-t location shift (degrees of freedom from 1.1 to 30) and heteroscedastic Gaussian (sigma ratio (positive / negative) from 0.2 to 5), with either class as the minority. The largest best-estimate miss rate the rule admits there is 4.47%.

**Where it routes.** The largest AUC sent to the unfloored band, by the size of the smaller class and the class ratio. Above it, and wherever the table says hybrid, the hybrid is used:

| smaller class | 1 : 1 | 2 : 1 | 4 : 1 | 6 : 1 | 9 : 1 | 19 : 1 |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | hybrid | hybrid | hybrid | hybrid | hybrid | hybrid |
| 250 | 0.732 | 0.732 | 0.732 | 0.732 | 0.732 | 0.732 |
| 500 | 0.810 | 0.810 | 0.810 | 0.810 | 0.810 | 0.810 |
| 1,000 | 0.866 | 0.866 | 0.866 | 0.866 | 0.866 | 0.866 |
| 3,000 | 0.922 | 0.922 | 0.922 | 0.922 | 0.922 | 0.922 |
| 10,000 | 0.958 | 0.958 | 0.958 | 0.958 | 0.958 | 0.958 |

**Evidence.**

- The model of the miss rate predicts held-out data well: between predicted rates of 2% and 12%, predictions are within 0.5 percentage points of observed rates (Section 3).
- At 26 fixed settings, each simulated up to 1,000 times, 25 observed miss rates have 95% intervals that contain the model's prediction (Section 4).
- Without any model: of the 552,014 simulated datasets the rule admits, 2.41% missed, and no group by family, class ratio and shape is significantly above 5% (Section 6).
- The previous rule, fitted before the rounds placed around the settings that bound it (`n_min >= 250 and n_max / n_min <= 6 and n_min * (1 - AUC)^2.5 >= 3.7`), admits 1 setting that fails the same test; the recommended rule admits none (Section 5.4).

**Limits.**

- Only the two families above are modelled. They are the known failure modes, but other score distributions are not covered.
- Above a larger class of 10,000 the simulated data are sparse and the rule is not held to the 90% test; there the floor costs far more compute and buys no width (Section 1). The 12 replicated settings measured there that the rule admits missed at 2.5%-5.1%, none significantly above 5%; the model's best estimate, extended past its data, reaches 6.59% at the edge of the design (Section 5.5).
- The study evaluates the rule at the true AUC; at runtime the empirical AUC stands in for it.
- The rule sends 69.6% of the grid to the hybrid; 8.9% of that is safe with 80% confidence, mostly in slivers along the rule's boundary. The largest such region has a smaller class of 317-956, AUC 0.772-0.942 and class ratio 1-2.33. A second clause beside the rule would route a further 1.9% of the grid without new draws, and a rule with two corner checks 1.7 percentage points (Section 5.6).
- Shares of the grid quoted below count points of an evaluation grid (log class size, probit AUC, ten class ratios) without weights. They compare candidate rules and do not estimate how often users are routed. The recommended rule admits 30.4% of the grid.

## 1. What routing buys

The earlier studies computed both bands on the same datasets, so they can be compared directly. The table pairs them for the two families of this study at the study's alpha; each miss rate is from the family in which it is higher.

| n_min | n_max | class ratio | datasets | hybrid area / unfloored area | unfloored miss rate | hybrid miss rate |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 5 | 1 | 11,995 | 1.43 | 1.3% | 0.0% |
| 15 | 15 | 1 | 203,995 | 1.57 | 2.5% | 0.9% |
| 9 | 21 | 2.33 | 11,995 | 1.51 | 2.6% | 0.9% |
| 3 | 27 | 9 | 203,995 | 1.45 | 2.9% | 0.8% |
| 50 | 50 | 1 | 203,995 | 1.32 | 2.8% | 1.8% |
| 30 | 70 | 2.33 | 11,995 | 1.32 | 4.1% | 1.9% |
| 10 | 90 | 9 | 203,995 | 1.41 | 4.7% | 1.7% |
| 150 | 150 | 1 | 203,995 | 1.13 | 4.6% | 2.0% |
| 90 | 210 | 2.33 | 11,995 | 1.13 | 6.1% | 1.9% |
| 30 | 270 | 9 | 203,995 | 1.23 | 5.7% | 2.0% |
| 500 | 500 | 1 | 203,995 | 1.04 | 3.4% | 2.2% |
| 300 | 700 | 2.33 | 11,995 | 1.04 | 5.0% | 2.5% |
| 100 | 900 | 9 | 215,990 | 1.09 | 6.6% | 2.3% |
| 1500 | 1500 | 1 | 203,995 | 1.01 | 2.8% | 2.4% |
| 900 | 2100 | 2.33 | 11,995 | 1.01 | 3.9% | 2.8% |
| 300 | 2700 | 9 | 203,995 | 1.03 | 5.1% | 2.5% |
| 5000 | 5000 | 1 | 203,995 | 1.00 | 3.1% | 3.0% |
| 3000 | 7000 | 2.33 | 11,995 | 1.00 | 3.3% | 2.9% |
| 1000 | 9000 | 9 | 203,995 | 1.01 | 4.2% | 3.1% |

- **The unfloored band can exceed the target.** Its miss rate is above 5% at 4 of these sizes, up to 6.6%, all with a smaller class of 300 or less; 4 of them have unequal classes. The hybrid's miss rate there is at most 2.5%. These are averages over shapes, so the worst shapes fail by more. This is what the rule must screen out.
- **At small samples the floor is costly.** With at most 50 cases in the larger class the hybrid is up to 1.6 times as wide as the unfloored band.
- **At large samples it buys nothing a user needs.** With 5,000 or more in the larger class the two areas agree within 0.9%. The floor still lowers the miss rate under imbalance, but further below the target, which is no gain when closer to nominal is better.
- **And it is slow there.** The hybrid's ELL floor needs a calibration for every new pair of class sizes. Timed once per size on this machine:

| positives / negatives | unfloored (s) | hybrid (s) |
| --- | --- | --- |
| 1,000 / 1,000 | 0.2 | 0.5 |
| 1,000 / 9,000 | 0.5 | 22.8 |
| 10,000 / 10,000 | 0.8 | 25.4 |

So the rule's job is to find where the unfloored band is safe and to send everything else to the hybrid.

## 2. Data

Each draw is one simulated dataset: scores from one of the two families at a chosen shape and AUC, at chosen class counts, with a record of whether the unfloored band missed the true ROC. The earlier studies sit on a few fixed sample sizes; this study's rounds placed draws between them and where the answer was least certain.

| source | description | draws | larger class | miss rate |
| --- | --- | --- | --- | --- |
| shape_auc_study | Shape x AUC study, 50% prevalence, fixed sample sizes | 1,151,814 | 15-5,000 | 2.8% |
| shape_auc_study_p10 | Shape x AUC study, 10% prevalence, fixed sample sizes | 959,828 | 90-9,000 | 4.6% |
| nov11 | Main simulation suite, 50% prevalence, fixed sample sizes | 83,923 | 15-5,000 | 3.0% |
| nov11_p10 | Main simulation suite, 10% prevalence, fixed sample sizes | 59,945 | 90-9,000 | 5.3% |
| nov11_p30 | Main simulation suite, 30% prevalence, fixed sample sizes | 71,934 | 21-7,000 | 3.1% |
| round0 | This study, 69% near the 5% contour, 31% spread over the design space | 67,842 | 10-99,997 | 4.2% |
| round1 | This study, 75% where the surface was least certain near 5%, 25% spread over the design space | 14,914 | 10-99,993 | 3.0% |
| round2 | This study, 75% where the surface was least certain near 5%, 25% spread over the design space | 14,886 | 10-99,925 | 3.6% |
| round3 | This study, 75% where the surface was least certain near 5%, 25% spread over the design space | 19,892 | 10-99,875 | 3.4% |
| round4 | This study, 100% replicated fixed cells | 21,000 | 120-80,000 | 4.5% |
| round5 | This study, 100% around grid points that fail the assurance test only through uncertainty and bound the rule | 48,620 | 224-12,189 | 2.8% |
| round6 | This study, 100% around grid points that fail the assurance test only through uncertainty and bound the rule | 47,555 | 2,132-11,699 | 2.9% |
| round7 | This study, 100% replicated fixed cells | 8,000 | 210-1,000 | 4.5% |

## 3. The model of the miss rate, and how well it predicts

For each family the probability that the band misses is modelled as a smooth function of four quantities: the expected number of each class beyond the ROC's corners, `n_pos (1 - AUC)` and `n_neg (1 - AUC)`; the AUC; and the family's shape parameter. The model (a Gaussian process classifier) gives, at any setting, a best estimate of the miss rate and how uncertain that estimate is. The rule uses both.

Held-out check: the data were split into 5 parts, and each part was predicted from a fit to the others. By predicted miss rate:

| predicted miss rate | datasets | predicted | observed | standard error |
| --- | --- | --- | --- | --- |
| 0%-1% | 352,209 | 0.73% | 0.91% | 0.02% |
| 1%-2% | 807,644 | 1.48% | 1.51% | 0.01% |
| 2%-3% | 744,752 | 2.47% | 2.41% | 0.02% |
| 3%-4% | 213,304 | 3.40% | 3.15% | 0.04% |
| 4%-5% | 105,858 | 4.46% | 4.16% | 0.06% |
| 5%-6% | 69,518 | 5.46% | 4.99% | 0.08% |
| 6%-8% | 74,627 | 6.89% | 6.40% | 0.09% |
| 8%-12% | 71,483 | 9.73% | 9.27% | 0.11% |
| 12%-20% | 61,109 | 15.40% | 16.17% | 0.15% |
| 20%-40% | 53,031 | 28.13% | 30.26% | 0.20% |
| 40%-100% | 16,618 | 49.11% | 48.10% | 0.39% |

Between 2% and 12%, the range the rule depends on, predictions are within 0.5 percentage points of the observed rates. Around the 5% target the model predicts slightly more misses than occur, which errs toward caution.

## 4. Replicated cells: a direct check where it matters

Most draws sit at their own setting, which lets the model borrow strength across settings but leaves it to the model to say what happens at any one of them. To check the model where the rule's boundaries are decided, fixed settings were each simulated up to 1,000 times: worst-case shapes at the edges of the admitted region, at large samples and at high imbalance.

| cell | family | shape | AUC | positives | negatives | datasets | observed miss rate (95% interval) | model's best estimate | model's P(miss rate > 5%) | routed to |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | heteroscedastic Gaussian | 0.2 | 0.750 | 120 | 120 | 1,000 | 6.1% (4.8%-7.8%) | 6.3% | 100.0% | hybrid |
| 1 | heteroscedastic Gaussian | 0.2 | 0.924 | 420 | 420 | 1,000 | 5.5% (4.2%-7.1%) | 6.0% | 100.0% | hybrid |
| 2 | heteroscedastic Gaussian | 0.2 | 0.830 | 320 | 160 | 1,000 | 7.3% (5.8%-9.1%) | 7.0% | 100.0% | hybrid |
| 3 | heteroscedastic Gaussian | 0.3 | 0.850 | 640 | 160 | 1,000 | 5.2% (4.0%-6.8%) | 5.1% | 62.8% | hybrid |
| 4 | heteroscedastic Gaussian | 2.24 | 0.914 | 480 | 1,920 | 1,000 | 4.3% (3.2%-5.7%) | 5.3% | 91.0% | hybrid |
| 5 | heteroscedastic Gaussian | 2.24 | 0.862 | 420 | 3,780 | 1,000 | 5.6% (4.3%-7.2%) | 5.4% | 98.3% | hybrid |
| 6 | heteroscedastic Gaussian | 2.24 | 0.902 | 1,100 | 9,900 | 1,000 | 5.4% (4.2%-7.0%) | 5.4% | 97.2% | hybrid |
| 7 | heteroscedastic Gaussian | 2.24 | 0.862 | 550 | 10,450 | 1,000 | 5.2% (4.0%-6.8%) | 5.8% | 98.7% | hybrid |
| 8 | heteroscedastic Gaussian | 2.24 | 0.924 | 3,300 | 62,700 | 500 | 5.0% (3.4%-7.3%) | 5.3% | 66.2% | unfloored |
| 9 | heteroscedastic Gaussian | 0.3 | 0.772 | 945 | 105 | 1,000 | 3.6% (2.6%-4.9%) | 4.5% | 4.9% | hybrid |
| 10 | heteroscedastic Gaussian | 5 | 0.792 | 6,080 | 320 | 1,000 | 2.6% (1.8%-3.8%) | 3.0% | 0.0% | hybrid |
| 11 | Student-t location shift | 1.1 | 0.705 | 27,000 | 3,000 | 1,000 | 4.4% (3.3%-5.9%) | 3.9% | 0.3% | unfloored |
| 12 | Student-t location shift | 1.1 | 0.705 | 36,100 | 1,900 | 1,000 | 2.5% (1.7%-3.7%) | 3.7% | 0.2% | unfloored |
| 13 | Student-t location shift | 1.1 | 0.750 | 62,700 | 3,300 | 1,000 | 3.9% (2.9%-5.3%) | 3.5% | 0.3% | unfloored |
| 14 | Student-t location shift | 4.7 | 0.924 | 34,800 | 8,700 | 1,000 | 3.1% (2.2%-4.4%) | 3.8% | 0.4% | unfloored |
| 15 | Student-t location shift | 3.8 | 0.924 | 60,000 | 15,000 | 1,000 | 3.7% (2.7%-5.1%) | 4.3% | 7.9% | unfloored |
| 16 | Student-t location shift | 3.8 | 0.924 | 40,000 | 20,000 | 1,000 | 4.6% (3.5%-6.1%) | 4.4% | 7.8% | unfloored |
| 17 | Student-t location shift | 3.8 | 0.914 | 20,000 | 20,000 | 1,000 | 5.1% (3.9%-6.6%) | 4.2% | 2.3% | unfloored |
| 18 | heteroscedastic Gaussian | 5 | 0.550 | 20,000 | 80,000 | 500 | 4.2% (2.8%-6.3%) | 4.8% | 36.6% | unfloored |
| 19 | heteroscedastic Gaussian | 5 | 0.550 | 5,000 | 45,000 | 1,000 | 4.3% (3.2%-5.7%) | 4.0% | 2.1% | unfloored |
| 20 | heteroscedastic Gaussian | 5 | 0.550 | 20,000 | 40,000 | 1,000 | 3.8% (2.8%-5.2%) | 4.0% | 0.9% | unfloored |
| 21 | heteroscedastic Gaussian | 5 | 0.550 | 20,000 | 20,000 | 1,000 | 3.2% (2.3%-4.5%) | 3.3% | 0.0% | unfloored |
| 22 | heteroscedastic Gaussian | 0.2 | 0.825 | 210 | 210 | 2,000 | 5.6% (4.7%-6.7%) | 5.3% | 96.3% | hybrid |
| 23 | heteroscedastic Gaussian | 0.2 | 0.800 | 630 | 210 | 2,000 | 4.9% (4.0%-5.9%) | 4.9% | 26.4% | hybrid |
| 24 | heteroscedastic Gaussian | 0.2 | 0.800 | 1,000 | 250 | 2,000 | 4.8% (3.9%-5.8%) | 4.1% | 0.0% | hybrid |
| 25 | heteroscedastic Gaussian | 0.3 | 0.810 | 420 | 210 | 2,000 | 2.6% (2.0%-3.4%) | 3.2% | 0.0% | hybrid |

25 of 26 cells' 95% intervals contain the model's best estimate; the exception is cell 12 (observed fewer misses than predicted). Significantly above 5%: cell 2, which the rule sends to the hybrid. Of the 10 cells whose best estimate exceeds 5%, the rule admits cell 8 (larger class above 10,000).

## 5. Deriving the rule

### 5.1 Which settings are unsafe

The model is evaluated on a grid of settings: smaller class 10-20,000 (log scale), AUC 0.55-0.99 (probit scale) and class ratios 1, 1.5, 2, 2.33, 3, 4, 6, 9, 13, 19. At each setting it takes the worst case over both families, their shapes and which class is the minority. A setting is *unsafe* if the probability that this worst-case miss rate exceeds 5% is above 10%. Requiring a probability, not only a best estimate, below the target keeps the rule out of regions the data cannot yet vouch for.

73.1% of the grid is unsafe. The rule must exclude every unsafe setting with a larger class of at most 10,000 (55.3% of the grid). Beyond that size the simulated data are sparse, Section 1 shows the floor buys no width there and costs far more compute, and the unsafe settings mostly reflect uncertainty about miss rates just below 5%. Section 5.5 checks the rule there against the replicated cells.

### 5.2 Choosing the rule

Candidate rules are conjunctions of simple checks: a minimum on the smaller class, a cap on the class ratio, a cap on the AUC, and a *corner* check that the AUC is low enough for the sample size. For every combination the thresholds are set to admit as much of the grid as possible without admitting an unsafe setting, then rounded to round numbers on the safe side. The best rule for each number of checks:

| checks | rule | grid share | recommended |
| --- | --- | --- | --- |
| 1 | n_min * (1 - AUC)^2 >= 25 | 27.6% |  |
| 2 | n_min >= 160 and n_min * (1 - AUC)^2 >= 18 | 30.4% | yes |
| 3 | n_min >= 210 and AUC <= 0.94 and n_min * (1 - AUC)^2 / ratio^0.333 >= 8 | 30.1% |  |

The recommended rule is the one with the fewest checks that admits within 1 percentage point of the best rule's grid share.

### 5.3 What each check guards against

Each check is credited with the unsafe settings that it alone excludes. Each turns out to bound one failure mechanism, identified by the family, minority class and shape at which the worst case occurs:

| check | unsafe settings | smaller class | AUC | class ratio | worst family | minority class | typical worst shape |
| --- | --- | --- | --- | --- | --- | --- | --- |
| n_min >= 160 | 33 | 91-120 | 0.550-0.604 | 1-19 | heteroscedastic Gaussian | negative | sigma ratio (positive / negative) 0.2 |
| n_min * (1 - AUC)^2 >= 18 | 3037 | 182-8,728 | 0.728-0.990 | 1-19 | Student-t location shift | positive | degrees of freedom 2 |
| several checks | 6720 | 10-159 | 0.550-0.990 | 1-19 | heteroscedastic Gaussian | positive | sigma ratio (positive / negative) 0.2 |

The last row counts settings that more than one check excludes; they lie deep inside the unsafe region and do not bear on any one check.

### 5.4 Sensitivity, and the previous rule

The recommended rule at other assurance levels, and the previous rule judged by the same test:

| rule | checks | grid share | unsafe settings admitted | worst mean miss rate, n_max <= 10,000 | worst P(miss rate > 5%), n_max <= 10,000 | worst mean miss rate, n_max > 10,000 |
| --- | --- | --- | --- | --- | --- | --- |
| recommended at 80% assurance | n_min >= 160 and n_min * (1 - AUC)^2 >= 15 | 31.6% | 0 | 4.63% | 12.7% | 6.59% |
| recommended at 90% assurance | n_min >= 160 and n_min * (1 - AUC)^2 >= 18 | 30.4% | 0 | 4.47% | 6.0% | 6.59% |
| recommended at 95% assurance | n_min >= 370 and n_min * (1 - AUC)^2.5 / ratio^0.333 >= 2.8 | 28.7% | 0 | 4.51% | 4.9% | 6.58% |
| previous, judged at 90% | n_min >= 250 and n_max / n_min <= 6 and n_min * (1 - AUC)^2.5 >= 3.7 | 24.2% | 1 | 4.73% | 13.9% | 6.30% |
| alternative at 90% assurance | n_min >= 210 and AUC <= 0.94 and n_min * (1 - AUC)^2 / ratio^0.333 >= 8 | 30.1% | 0 | 4.60% | 5.6% | 6.58% |

The checks keep their form across assurance levels; the thresholds move.

### 5.5 Beyond a larger class of 10,000

The rule admits 2,817 grid settings there, 1,614 of them unsafe by the test above. At 1,093 of those the best estimate is below 5% and only its uncertainty fails the test. The best estimate exceeds 5% at 521, all with a larger class of 36,263 or more, where it extends a rising trend beyond the data. The replicated cells in that range that the rule admits measure the band directly:

| cell | family | shape | AUC | positives | negatives | datasets | observed miss rate (95% interval) | model's best estimate | model's P(miss rate > 5%) | routed to |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | heteroscedastic Gaussian | 2.24 | 0.924 | 3,300 | 62,700 | 500 | 5.0% (3.4%-7.3%) | 5.3% | 66.2% | unfloored |
| 11 | Student-t location shift | 1.1 | 0.705 | 27,000 | 3,000 | 1,000 | 4.4% (3.3%-5.9%) | 3.9% | 0.3% | unfloored |
| 12 | Student-t location shift | 1.1 | 0.705 | 36,100 | 1,900 | 1,000 | 2.5% (1.7%-3.7%) | 3.7% | 0.2% | unfloored |
| 13 | Student-t location shift | 1.1 | 0.750 | 62,700 | 3,300 | 1,000 | 3.9% (2.9%-5.3%) | 3.5% | 0.3% | unfloored |
| 14 | Student-t location shift | 4.7 | 0.924 | 34,800 | 8,700 | 1,000 | 3.1% (2.2%-4.4%) | 3.8% | 0.4% | unfloored |
| 15 | Student-t location shift | 3.8 | 0.924 | 60,000 | 15,000 | 1,000 | 3.7% (2.7%-5.1%) | 4.3% | 7.9% | unfloored |
| 16 | Student-t location shift | 3.8 | 0.924 | 40,000 | 20,000 | 1,000 | 4.6% (3.5%-6.1%) | 4.4% | 7.8% | unfloored |
| 17 | Student-t location shift | 3.8 | 0.914 | 20,000 | 20,000 | 1,000 | 5.1% (3.9%-6.6%) | 4.2% | 2.3% | unfloored |
| 18 | heteroscedastic Gaussian | 5 | 0.550 | 20,000 | 80,000 | 500 | 4.2% (2.8%-6.3%) | 4.8% | 36.6% | unfloored |
| 19 | heteroscedastic Gaussian | 5 | 0.550 | 5,000 | 45,000 | 1,000 | 4.3% (3.2%-5.7%) | 4.0% | 2.1% | unfloored |
| 20 | heteroscedastic Gaussian | 5 | 0.550 | 20,000 | 40,000 | 1,000 | 3.8% (2.8%-5.2%) | 4.0% | 0.9% | unfloored |
| 21 | heteroscedastic Gaussian | 5 | 0.550 | 20,000 | 20,000 | 1,000 | 3.2% (2.3%-4.5%) | 3.3% | 0.0% | unfloored |

### 5.6 Safe settings the rule sends to the hybrid

Sending a safe setting to the hybrid costs width at small samples and compute at large ones (Section 1). A setting goes to the hybrid either because it is unsafe or because a few simple checks cannot carve it out from unsafe neighbours. The table splits the part of the grid each rule sends to the hybrid by its best estimate and by the probability that it is safe:

| rule | grid share sent to the hybrid | best estimate above 5% | best estimate at most 5% | safe with 80% confidence | safe with 90% confidence | of the 80%-safe, larger class above 10,000 |
| --- | --- | --- | --- | --- | --- | --- |
| recommended | 69.6% | 89.5% | 10.5% | 8.9% | 8.2% | 3.4% |
| previous | 75.8% | 84.5% | 15.5% | 12.5% | 11.2% | 26.0% |
| alternative | 69.9% | 89.7% | 10.3% | 8.2% | 7.3% | 11.3% |

**Where the slack is.** At each smaller class and class ratio, the largest AUC the recommended rule and the alternative rule admit, then the largest AUC up to which every setting is safe at 90% (`none` where the lowest AUC already fails). A † marks a larger class above 10,000, where rules are not held to the test:

| smaller class | 1 : 1 | 2 : 1 | 4 : 1 | 6 : 1 | 9 : 1 | 19 : 1 |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | hybrid / hybrid / none | hybrid / hybrid / none | hybrid / hybrid / none | hybrid / hybrid / 0.550 | hybrid / hybrid / 0.604 | hybrid / hybrid / none |
| 250 | 0.732 / 0.821 / 0.811 | 0.732 / 0.799 / 0.811 | 0.732 / 0.775 / 0.811 | 0.732 / 0.759 / 0.829 | 0.732 / 0.742 / 0.811 | 0.732 / 0.708 / 0.750 |
| 500 | 0.810 / 0.873 / 0.902 | 0.810 / 0.858 / 0.902 | 0.810 / 0.841 / 0.890 | 0.810 / 0.829 / 0.862 | 0.810 / 0.818 / 0.846 | 0.810 / 0.793 / 0.792 |
| 1,000 | 0.866 / 0.911 / 0.942 | 0.866 / 0.900 / 0.942 | 0.866 / 0.887 / 0.902 | 0.866 / 0.879 / 0.890 | 0.866 / 0.871 / 0.876 | 0.866 / 0.854 / 0.846 † |
| 3,000 | 0.922 / 0.940 / 0.956 | 0.922 / 0.940 / 0.942 | 0.922 / 0.935 / 0.924 † | 0.922 / 0.930 / 0.914 † | 0.922 / 0.925 / none † | 0.922 / 0.916 / none † |
| 10,000 | 0.958 / 0.940 / 0.956 | 0.958 / 0.940 / none † | 0.958 / 0.940 / none † | 0.958 / 0.940 / none † | 0.958 / 0.940 / none † | 0.958 / 0.940 / none † |

A single corner check bounds `1 - AUC` by one power of the smaller class, a straight line on log scales. The safe limit is not straight: it rises quickly with the smaller class at small samples and levels off at large ones. Two corner checks of different exponents follow such a bend. The best rule of that form, `n_min >= 160 and n_min * (1 - AUC)^1.25 / ratio^0.667 >= 8.5 and n_min * (1 - AUC)^2 >= 14`, admits 32.1% of the grid against 30.4% for the recommended rule; the rest of the slack lies in thin slivers beside unsafe settings, which no few checks can follow.

**Recommended rule** (`n_min >= 160 and n_min * (1 - AUC)^2 >= 18`). The largest connected regions it sends to the hybrid that are safe with 80% confidence (settings are connected if they differ by one step of the grid in the smaller class, the AUC or the class ratio):

| region | settings | grid share | smaller class | larger class | AUC | class ratio | worst family | minority class | typical worst shape | best estimate | safe at 90% | sent to the hybrid by | draws at the hardest setting | CPU-hours there |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 245 | 1.4% | 317-956 | 317-2,231 | 0.772-0.942 | 1-2.33 | heteroscedastic Gaussian | negative | 0.2 | 2.82%-4.84% | 95.9% | `n_min * (1 - AUC)^2 >= 18` (100%) | 17,632 | 1.8 |
| 2 | 179 | 1.0% | 1,098-2,889 | 1,098-6,741 | 0.876-0.956 | 1-2.33 | Student-t location shift | negative | 5.74 | 2.64%-4.78% | 96.1% | `n_min * (1 - AUC)^2 >= 18` (100%) | 6,262 | 2.4 |
| 3 | 140 | 0.8% | 317-956 | 950-5,738 | 0.772-0.924 | 3-6 | Student-t location shift | negative | 5.74 | 3.00%-4.83% | 95.7% | `n_min * (1 - AUC)^2 >= 18` (100%) | 15,535 | 5.4 |
| 4 | 130 | 0.7% | 79-276 | 821-5,238 | 0.550-0.811 | 9-19 | heteroscedastic Gaussian | negative | 1.22 | 3.44%-4.71% | 90.8% | `n_min >= 160` (72%), `n_min * (1 - AUC)^2 >= 18` (65%) | 4,029 | 0.4 |
| 5 | 122 | 0.7% | 120-276 | 120-643 | 0.550-0.846 | 1-2.33 | heteroscedastic Gaussian | negative | 0.2 | 3.58%-4.81% | 88.5% | `n_min * (1 - AUC)^2 >= 18` (64%) | 9,415 | 0.4 |
| 6 | 113 | 0.6% | 105-276 | 361-1,654 | 0.550-0.846 | 3-6 | heteroscedastic Gaussian | negative | 0.2 | 3.33%-4.81% | 91.2% | `n_min * (1 - AUC)^2 >= 18` (67%) | 11,377 | 1.4 |

A second clause admitted beside the rule (`rule or clause`) can route part of this without loosening the rule's checks. Searched like the rule over the settings it sends to the hybrid, and held to the test at every class size:

| when | clause | grid share added | settings needing draws | draws at the hardest setting | CPU-hours there |
| --- | --- | --- | --- | --- | --- |
| now, at 90% | n_max / n_min <= 3 and AUC <= 0.92 and n_min * (1 - AUC)^1 >= 55 | 1.9% | 0 | — | — |
| after draws lift 80%-safe settings to 90% | n_max / n_min <= 3 and AUC <= 0.92 and n_min * (1 - AUC)^1 >= 48 | 2.4% | 5 | 9,415 | 0.4 |

**Alternative rule** (`n_min >= 210 and AUC <= 0.94 and n_min * (1 - AUC)^2 / ratio^0.333 >= 8`). The largest connected regions it sends to the hybrid that are safe with 80% confidence (settings are connected if they differ by one step of the grid in the smaller class, the AUC or the class ratio):

| region | settings | grid share | smaller class | larger class | AUC | class ratio | worst family | minority class | typical worst shape | best estimate | safe at 90% | sent to the hybrid by | draws at the hardest setting | CPU-hours there |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 169 | 1.0% | 79-276 | 821-5,238 | 0.550-0.811 | 9-19 | heteroscedastic Gaussian | negative | 1.22 | 3.26%-4.71% | 92.9% | `n_min >= 210` (89%), `n_min * (1 - AUC)^2 / ratio^0.333 >= 8` (59%) | 4,029 | 0.4 |
| 2 | 143 | 0.8% | 120-276 | 120-643 | 0.550-0.846 | 1-2.33 | heteroscedastic Gaussian | negative | 0.2 | 3.43%-4.81% | 90.2% | `n_min >= 210` (93%) | 9,415 | 0.4 |
| 3 | 141 | 0.8% | 317-956 | 317-2,231 | 0.829-0.942 | 1-2.33 | heteroscedastic Gaussian | negative | 0.2 | 3.19%-4.84% | 92.9% | `n_min * (1 - AUC)^2 / ratio^0.333 >= 8` (100%) | 17,632 | 1.8 |
| 4 | 139 | 0.8% | 105-276 | 361-1,654 | 0.550-0.846 | 3-6 | heteroscedastic Gaussian | negative | 0.2 | 3.22%-4.81% | 92.8% | `n_min >= 210` (84%) | 11,377 | 1.4 |
| 5 | 99 | 0.6% | 1,098-2,889 | 1,098-6,741 | 0.902-0.956 | 1-2.33 | Student-t location shift | positive | 5.74 | 3.22%-4.78% | 92.9% | `AUC <= 0.94` (60%), `n_min * (1 - AUC)^2 / ratio^0.333 >= 8` (97%) | 6,262 | 2.4 |
| 6 | 98 | 0.6% | 317-956 | 950-5,738 | 0.792-0.924 | 3-6 | heteroscedastic Gaussian | positive | 2.24 | 3.34%-4.83% | 93.9% | `n_min * (1 - AUC)^2 / ratio^0.333 >= 8` (100%) | 15,535 | 5.4 |

A second clause admitted beside the rule (`rule or clause`) can route part of this without loosening the rule's checks. Searched like the rule over the settings it sends to the hybrid, and held to the test at every class size:

| when | clause | grid share added | settings needing draws | draws at the hardest setting | CPU-hours there |
| --- | --- | --- | --- | --- | --- |
| now, at 90% | AUC <= 0.725 and n_min * (1 - AUC)^0.75 >= 69 | 1.2% | 0 | — | — |
| after draws lift 80%-safe settings to 90% | AUC <= 0.725 and n_min * (1 - AUC)^0.75 >= 63 | 1.5% | 8 | 9,415 | 0.4 |

*Sent to the hybrid by* lists the checks that at least half of a region fails, with the share failing. *Draws at the hardest setting* is a rough count of the effective draws the region's least certain setting still needs for 90% assurance, treating its posterior as binomial at the best estimate; draws nearby inform it too, so this is a guide to the information required rather than a design. Regions whose settings are mostly safe at 90% already need only a different rule; the rest need draws first.

## 6. Model-free check: the simulated datasets the rule admits

The simulated datasets inside the recommended rule, grouped by family, class ratio and third of the family's shape range (on a log scale). Each group pools easier and harder shapes, so this confirms the rule rather than testing its worst case, which is what the model is for.

| family | class ratio | shape | datasets | miss rate | 95% upper bound |
| --- | --- | --- | --- | --- | --- |
| heteroscedastic Gaussian | 1-2.33 | sigma ratio (positive / negative) 0.2-0.58 | 44,155 | 2.08% | 2.22% |
| heteroscedastic Gaussian | 1-2.33 | sigma ratio (positive / negative) 0.58-1.7 | 59,833 | 2.22% | 2.35% |
| heteroscedastic Gaussian | 1-2.33 | sigma ratio (positive / negative) 1.7-5 | 60,285 | 2.41% | 2.54% |
| heteroscedastic Gaussian | 2.33-6 | sigma ratio (positive / negative) 0.2-0.58 | 1,034 | 2.51% | 3.66% |
| heteroscedastic Gaussian | 2.33-6 | sigma ratio (positive / negative) 0.58-1.7 | 876 | 2.97% | 4.31% |
| heteroscedastic Gaussian | 2.33-6 | sigma ratio (positive / negative) 1.7-5 | 1,417 | 3.67% | 4.78% |
| heteroscedastic Gaussian | 6-19 | sigma ratio (positive / negative) 0.2-0.58 | 73,469 | 2.45% | 2.56% |
| heteroscedastic Gaussian | 6-19 | sigma ratio (positive / negative) 0.58-1.7 | 36,341 | 2.58% | 2.74% |
| heteroscedastic Gaussian | 6-19 | sigma ratio (positive / negative) 1.7-5 | 27,958 | 3.08% | 3.29% |
| Student-t location shift | 1-2.33 | degrees of freedom 1.1-3.3 | 61,593 | 2.27% | 2.39% |
| Student-t location shift | 1-2.33 | degrees of freedom 3.3-10 | 54,631 | 2.40% | 2.54% |
| Student-t location shift | 1-2.33 | degrees of freedom 10-30 | 48,286 | 2.26% | 2.40% |
| Student-t location shift | 2.33-6 | degrees of freedom 1.1-3.3 | 937 | 2.99% | 4.28% |
| Student-t location shift | 2.33-6 | degrees of freedom 3.3-10 | 2,961 | 3.51% | 4.24% |
| Student-t location shift | 2.33-6 | degrees of freedom 10-30 | 868 | 2.65% | 3.94% |
| Student-t location shift | 6-19 | degrees of freedom 1.1-3.3 | 32,798 | 2.59% | 2.76% |
| Student-t location shift | 6-19 | degrees of freedom 3.3-10 | 24,088 | 2.49% | 2.70% |
| Student-t location shift | 6-19 | degrees of freedom 10-30 | 20,484 | 2.47% | 2.69% |

Over all 552,014 datasets the miss rate is 2.41%. The highest among groups of at least 200 datasets is 3.67% (heteroscedastic Gaussian, class ratio 2.33-6, sigma ratio (positive / negative) 1.7-5).

## 7. Figure

`surface.png` shows, for five class ratios, the worst-case best estimate of the miss rate (top row; blue below 5%, red above) and the probability that it exceeds 5% (bottom row). The black contour marks the 90% assurance boundary; the solid green line encloses the settings the recommended rule admits, the dashed line those of the previous rule and the dash-dot line those of the alternative rule, where there is one (Section 5.6). The dotted black line marks a larger class of 10,000, beyond which the rule is not held to the assurance test (Section 5.5), so the green line may cross the black contour to its right. The grey line marks the largest smaller class that can be simulated at that ratio.

## 8. Validation plan for the frontier-shaped rule

[`hybrid_band.md` §4.4](../../../stats/hybrid_band.md#44-auto-routing-between-the-unfloored-and-hybrid-bands) recommends `n_min >= 150 and n_min * (1 - AUC) >= 50 * ratio^(1/6) and 1 - AUC >= 0.05 * sqrt(ratio)`: a corner count for small samples and an AUC ceiling, falling with imbalance, for large ones. It was chosen from this study's fitted grid. Four questions remain before it can be relied on, and replicated cells answer each one directly. The plan costs about 41 CPU-hours, plus the hybrid's ELL calibrations in step 3.

Sections 1–7 above are generated by `auto_rule_surface.py`. If they recommend a different rule, they come from a fit that predates the current rule search, which scores rules by the safe settings they capture; `report` rewrites them.

**How the cells are run.** Steps 1–3 are paired (`--paired`). Each draw computes the unfloored band and the hybrid on one fiducial cloud at α of 0.01, 0.05, 0.10 and 0.20. It records each band's miss indicator, its area, and the FPR range of its violations. The unfloored band at α = 0.05 is bit-identical to an unpaired draw, so the draws also join the surface fit. One round therefore answers four questions:

- Is the unfloored band safe where the rule admits it?
- Does the hybrid fix the misses where the rule refuses?
- What does the floor cost in width?
- How does the frontier move with α?

Cells reuse a few class sizes so that ELL calibrations are shared. Shapes are each family's worst case at that size on the fitted grid, with the minority class set accordingly.

```
uv run python stats/experiments/auto_rule_surface.py run --cells data/results/auto_rule/cells_validation_paired.csv --paired
uv run python stats/experiments/auto_rule_surface.py run --cells data/results/auto_rule/cells_validation_large.csv
uv run python stats/experiments/auto_rule_surface.py fit
```

**Step 1: the rule's new admits (10 cells, about 5 CPU-hours, 1,000 draws each).**

- **What it tests.** The rule admits settings that the earlier two-check rule sent to the hybrid: a minority class of 280–1,900 and AUC 0.76–0.93, mostly near balance. These settings rest on the model and pooled draws alone.
- **The cells.** Each sits 0.01–0.02 in AUC inside the rule's limit:
  - Heteroscedastic σ₊/σ₋ of 0.2 (balanced, and with the negatives as the minority), 0.5 (balanced), and 1.8 (balanced, and with the positives as the minority at ratios 2 and 4).
  - Student-t with 2, 6 and 9 degrees of freedom, up to 1,900/1,900.
- **Pass.** No cell's 95% interval lies above 5%.
- **Fail.** A failing cell tightens the corner constant or the ratio exponent.

**Step 2: just past the frontier (6 cells, about 2 CPU-hours, 1,000 draws each).**

- **What it tests.** The rule sends these cells to the hybrid, so the question is whether the hybrid actually lowers the miss rate there. This is criterion (b): route to the hybrid only where it makes a difference.
- **The cells.** Five mirror step 1, 0.03–0.04 in AUC past the limit. The sixth is the floor region: σ₊/σ₋ = 0.2 at 120/120 and AUC 0.60.
- **What to read.** Paired miss rates of the two bands, and the hybrid's area penalty.
  - Where the hybrid misses about as often as the unfloored band, routing there buys nothing.
  - If a cell shows that, the region needs a different remedy, not a stricter rule.

**Step 3: the ceiling (8 cells, about 24 CPU-hours, 300–500 draws each).**

- **What it tests.** The ceiling is measured at balance up to 10,000 per class. Its fall with imbalance, and its level above a larger class of 10,000, rest on sparse data.
- **The cells.** Four pairs, each straddling the ceiling, one cell inside and one outside:
  - Student-t, 5 degrees of freedom, 10,000/10,000.
  - Student-t, 5 degrees of freedom, positives as the minority, 4,000/16,000.
  - Heteroscedastic σ₊/σ₋ = 2.24, positives as the minority, 2,000/18,000.
  - Student-t, 3.8 degrees of freedom, 20,000/20,000.
- **Why these cells.** The outside cells at ratios 4 and 9 are ones the earlier two-check rule admitted. They show whether the rule's caution there is warranted.
- **Two outcomes.**
  - The outside cells pass and the hybrid does not change their miss rate: the ceiling can be relaxed beyond a larger class of 10,000.
  - They fail and the hybrid fixes them: the ceiling stays.
- **Calibration cost.** The hybrid's ELL calibration at 10,000–20,000 per class takes minutes per class size and level in each worker. Cells reuse four class sizes to bound this. The table of precomputed levels in [`ell_band.md` §4.4](../../../stats/ell_band.md#44-warm-started-calibration-from-a-table-of-levels-proposed) would remove most of it.

**Step 4: beyond a larger class of 50,000 (4 cells, about 11 CPU-hours, 300 draws each, unpaired).**

- **What it tests.** The model's best estimate exceeds 5% at a larger class of 47,800 or more and class ratios of 3 or more, even at AUC 0.55. Its draws there are sparse. The replicated cells measured so far in that range missed at 2.5–5.1%. The likelier cause is the band's thinning at large samples rather than corner hooks.
- **The cells.**
  - Heteroscedastic σ₊/σ₋ = 5 at AUC 0.60, 3,000/57,000.
  - Student-t with 2 degrees of freedom at AUC 0.70, 5,000/50,000.
  - Heteroscedastic σ₊/σ₋ = 0.2 at AUC 0.65, 60,000/6,000.
  - Heteroscedastic σ₊/σ₋ = 2.24 at AUC 0.80, 10,000/60,000.
- **Why unpaired.** An ELL calibration at these sizes takes too long.
- **Outcome.** A failure here limits the unfloored band at very large samples, whatever the AUC. That would be a separate check on `n_max` in the rule.

**Without new draws.**

- **Routing on the empirical AUC, end to end.** Use the step 1–3 cells: route each draw by its empirical AUC, then score the band it reaches. This bounds the selection effect at the worst cells, not only in aggregate.
- **A slope-gated variant.** At a minority class of 150 or more, the observed binormal slope separates the extreme heteroscedastic shapes perfectly. No draw with σ₊/σ₋ of at most 0.5 or at least 2 has a slope in [0.6, 1.67]. Those shapes bind the frontier up to a minority class of about 1,000. Re-running `fit` with the heteroscedastic shape range narrowed to 0.6–1.67 gives the rule that would apply when the observed slope is in range.

## Appendix: best rule for every combination of checks

| checks | rule (exact thresholds) | grid share |
| --- | --- | --- |
| imbalance_corner + floor + ratio_cap + auc_cap | n_min >= 209.1 and AUC <= 0.9418 and n_min * (1 - AUC)^2 / ratio^0.333 >= 7.905 | 31.2% |
| imbalance_corner + floor + auc_cap | n_min >= 209.1 and AUC <= 0.9418 and n_min * (1 - AUC)^2 / ratio^0.333 >= 7.905 | 31.2% |
| corner + floor | n_min >= 158.6 and n_min * (1 - AUC)^2 >= 17.17 | 30.9% |
| corner + floor + auc_cap | n_min >= 158.6 and n_min * (1 - AUC)^2 >= 17.17 | 30.9% |
| corner + floor + ratio_cap | n_min >= 158.6 and n_min * (1 - AUC)^2 >= 17.17 | 30.9% |
| corner + floor + ratio_cap + auc_cap | n_min >= 158.6 and n_min * (1 - AUC)^2 >= 17.17 | 30.9% |
| imbalance_corner + floor | n_min >= 209.1 and n_min * (1 - AUC)^2.5 / ratio^0.333 >= 3.609 | 30.4% |
| imbalance_corner + floor + ratio_cap | n_min >= 209.1 and n_min * (1 - AUC)^2.5 / ratio^0.333 >= 3.609 | 30.4% |
| corner + auc_cap | AUC <= 0.9418 and n_min * (1 - AUC)^1.5 >= 47.06 | 29.2% |
| corner + ratio_cap + auc_cap | AUC <= 0.9418 and n_min * (1 - AUC)^1.5 >= 47.06 | 29.2% |
| imbalance_corner + auc_cap | AUC <= 0.9418 and n_min * (1 - AUC)^1.5 / ratio^0.333 >= 33.1 | 28.2% |
| imbalance_corner + ratio_cap + auc_cap | AUC <= 0.9418 and n_min * (1 - AUC)^1.5 / ratio^0.333 >= 33.1 | 28.2% |
| corner | n_min * (1 - AUC)^2 >= 24.49 | 28.1% |
| corner + ratio_cap | n_min * (1 - AUC)^2 >= 24.49 | 28.1% |
| imbalance_corner | n_min * (1 - AUC)^2 / ratio^0.333 >= 21.54 | 25.0% |
| imbalance_corner + ratio_cap | n_min * (1 - AUC)^2 / ratio^0.333 >= 21.54 | 25.0% |
| floor + auc_cap | AUC <= 0.8764 and n_min >= 956.4 | 18.6% |
| floor + ratio_cap + auc_cap | AUC <= 0.8764 and n_min >= 956.4 | 18.6% |
| floor | n_min >= 1.002e+04 | 7.6% |
| floor + ratio_cap | n_min >= 1.002e+04 | 7.6% |
| auc_cap | AUC <= 0.55 | 0.0% |
| ratio_cap | n_max / n_min <= 1 | 0.0% |
| ratio_cap + auc_cap | n_max / n_min <= 1 | 0.0% |

Model length scales (positive corner, negative corner, AUC, shape): Student-t location shift [0.172, 0.465, 0.613, 0.344], amplitude 1.38; heteroscedastic Gaussian [0.198, 0.22, 0.851, 0.13], amplitude 1.36.
