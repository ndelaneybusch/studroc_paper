# Routing rule for `auto`: when the unfloored fiducial band is safe

## Summary

**The decision.** The `auto` mode chooses between two ROC confidence bands using only what it can see: the two class counts and the AUC. The *unfloored fiducial band* is narrow and fast. The *hybrid* adds an exact ELL floor at the two ends of the curve, which protects against the unfloored band's known failure: when the ROC has a convex hook in a corner that few observations reach, the unfloored band misses the true curve more often than its nominal 5%. Where the unfloored band is safe it is preferred, because it is narrower at small samples, much faster at large ones, and closer to nominal (Section 1). This report finds where it is safe.

**Recommended rule.** Use the unfloored band when all of the following hold, and the hybrid otherwise (`n_min` and `n_max` are the smaller and larger class counts):

- the smaller class has at least 250 cases.
- the larger class is at most 6 times the smaller (prevalence of the minority class at least 14%).
- the AUC is low enough for the sample size: `n_min * (1 - AUC)^2.5 >= 3.7`. The left side grows with the expected number of minority cases beyond the ROC's corner, so the largest AUC allowed rises with the smaller class.

In one line: `n_min >= 250 and n_max / n_min <= 6 and n_min * (1 - AUC)^2.5 >= 3.7`.

**What safe means.** A *miss* is a dataset whose band fails to contain the true ROC somewhere; at 95% confidence the target miss rate is 5%. At every setting the rule admits with a larger class of at most 10,000, the fitted model gives at least 90% probability that the miss rate is at most 5%. This holds for the worst score distribution in the two families known to break the unfloored band, Student-t location shift (degrees of freedom from 1.1 to 30) and heteroscedastic Gaussian (sigma ratio (positive / negative) from 0.2 to 5), with either class as the minority. The largest best-estimate miss rate the rule admits there is 4.67%.

**Where it routes.** The largest AUC sent to the unfloored band, by the size of the smaller class and the class ratio. Above it, and wherever the table says hybrid, the hybrid is used:

| smaller class | 1 : 1 | 2 : 1 | 4 : 1 | 6 : 1 | 9 : 1 | 19 : 1 |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | hybrid | hybrid | hybrid | hybrid | hybrid | hybrid |
| 250 | 0.815 | 0.815 | 0.815 | 0.815 | hybrid | hybrid |
| 500 | 0.859 | 0.859 | 0.859 | 0.859 | hybrid | hybrid |
| 1,000 | 0.893 | 0.893 | 0.893 | 0.893 | hybrid | hybrid |
| 3,000 | 0.931 | 0.931 | 0.931 | 0.931 | hybrid | hybrid |
| 10,000 | 0.958 | 0.958 | 0.958 | 0.958 | hybrid | hybrid |

**Evidence.**

- The model of the miss rate predicts held-out data well: between predicted rates of 2% and 12%, predictions are within 0.6 percentage points of observed rates (Section 3).
- At 22 fixed settings, each simulated up to 1,000 times, 21 observed miss rates have 95% intervals that contain the model's prediction (Section 4).
- Without any model: of the 373,263 simulated datasets the rule admits, 2.30% missed, and no group by family, class ratio and shape is significantly above 5% (Section 6).
- The rule currently in `hybrid_band.md` §4.4 (`n_min >= 100 and n_min * (1 - AUC)^2.5 / ratio^0.667 >= 12`) admits 30 settings that fail the same test; the recommended rule admits none (Section 5.4).

**Limits.**

- Only the two families above are modelled. They are the known failure modes, but other score distributions are not covered.
- Above a larger class of 10,000 the simulated data are sparse and the rule is not held to the 90% test; there the floor costs far more compute and buys no width (Section 1). The 7 replicated settings measured there that the rule admits missed at 3.1%-5.1%, none significantly above 5%; the model's best estimate, extended past its data, reaches 6.46% at the edge of the design (Section 5.5).
- The study evaluates the rule at the true AUC; at runtime the empirical AUC stands in for it.
- Shares of the grid quoted below count points of an evaluation grid (log class size, probit AUC, ten class ratios) without weights. They compare candidate rules and do not estimate how often users are routed. The recommended rule admits 24.2% of the grid.

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
| 1,000 / 9,000 | 0.5 | 23.7 |
| 10,000 / 10,000 | 0.8 | 26.4 |

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

## 3. The model of the miss rate, and how well it predicts

For each family the probability that the band misses is modelled as a smooth function of four quantities: the expected number of each class beyond the ROC's corners, `n_pos (1 - AUC)` and `n_neg (1 - AUC)`; the AUC; and the family's shape parameter. The model (a Gaussian process classifier) gives, at any setting, a best estimate of the miss rate and how uncertain that estimate is. The rule uses both.

Held-out check: the data were split into 5 parts, and each part was predicted from a fit to the others. By predicted miss rate:

| predicted miss rate | datasets | predicted | observed | standard error |
| --- | --- | --- | --- | --- |
| 0%-1% | 351,478 | 0.72% | 0.91% | 0.02% |
| 1%-2% | 809,099 | 1.48% | 1.50% | 0.01% |
| 2%-3% | 676,244 | 2.46% | 2.42% | 0.02% |
| 3%-4% | 190,069 | 3.41% | 3.17% | 0.04% |
| 4%-5% | 95,773 | 4.46% | 4.07% | 0.06% |
| 5%-6% | 66,826 | 5.47% | 4.90% | 0.08% |
| 6%-8% | 73,424 | 6.89% | 6.41% | 0.09% |
| 8%-12% | 71,133 | 9.73% | 9.11% | 0.11% |
| 12%-20% | 61,673 | 15.43% | 16.14% | 0.15% |
| 20%-40% | 53,510 | 27.99% | 29.97% | 0.20% |
| 40%-100% | 16,749 | 48.69% | 48.45% | 0.39% |

Between 2% and 12%, the range the rule depends on, predictions are within 0.6 percentage points of the observed rates. Around the 5% target the model predicts slightly more misses than occur, which errs toward caution.

## 4. Replicated cells: a direct check where it matters

Most draws sit at their own setting, which lets the model borrow strength across settings but leaves it to the model to say what happens at any one of them. To check the model where the rule's boundaries are decided, fixed settings were each simulated up to 1,000 times: worst-case shapes at the edges of the admitted region, at large samples and at high imbalance.

| cell | family | shape | AUC | positives | negatives | datasets | observed miss rate (95% interval) | model's best estimate | model's P(miss rate > 5%) | routed to |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | heteroscedastic Gaussian | 0.2 | 0.750 | 120 | 120 | 1,000 | 6.1% (4.8%-7.8%) | 5.8% | 99.9% | hybrid |
| 1 | heteroscedastic Gaussian | 0.2 | 0.924 | 420 | 420 | 1,000 | 5.5% (4.2%-7.1%) | 5.6% | 98.0% | hybrid |
| 2 | heteroscedastic Gaussian | 0.2 | 0.830 | 320 | 160 | 1,000 | 7.3% (5.8%-9.1%) | 6.4% | 100.0% | hybrid |
| 3 | heteroscedastic Gaussian | 0.3 | 0.850 | 640 | 160 | 1,000 | 5.2% (4.0%-6.8%) | 5.3% | 87.3% | hybrid |
| 4 | heteroscedastic Gaussian | 2.24 | 0.914 | 480 | 1,920 | 1,000 | 4.3% (3.2%-5.7%) | 5.2% | 77.5% | hybrid |
| 5 | heteroscedastic Gaussian | 2.24 | 0.862 | 420 | 3,780 | 1,000 | 5.6% (4.3%-7.2%) | 5.3% | 93.2% | hybrid |
| 6 | heteroscedastic Gaussian | 2.24 | 0.902 | 1,100 | 9,900 | 1,000 | 5.4% (4.2%-7.0%) | 5.4% | 96.2% | hybrid |
| 7 | heteroscedastic Gaussian | 2.24 | 0.862 | 550 | 10,450 | 1,000 | 5.2% (4.0%-6.8%) | 5.6% | 94.0% | hybrid |
| 8 | heteroscedastic Gaussian | 2.24 | 0.924 | 3,300 | 62,700 | 500 | 5.0% (3.4%-7.3%) | 5.6% | 81.5% | hybrid |
| 9 | heteroscedastic Gaussian | 0.3 | 0.772 | 945 | 105 | 1,000 | 3.6% (2.6%-4.9%) | 4.7% | 23.3% | hybrid |
| 10 | heteroscedastic Gaussian | 5 | 0.792 | 6,080 | 320 | 1,000 | 2.6% (1.8%-3.8%) | 3.2% | 0.1% | hybrid |
| 11 | Student-t location shift | 1.1 | 0.705 | 27,000 | 3,000 | 1,000 | 4.4% (3.3%-5.9%) | 3.9% | 0.2% | hybrid |
| 12 | Student-t location shift | 1.1 | 0.705 | 36,100 | 1,900 | 1,000 | 2.5% (1.7%-3.7%) | 3.7% | 0.4% | hybrid |
| 13 | Student-t location shift | 1.1 | 0.750 | 62,700 | 3,300 | 1,000 | 3.9% (2.9%-5.3%) | 3.6% | 0.4% | hybrid |
| 14 | Student-t location shift | 4.7 | 0.924 | 34,800 | 8,700 | 1,000 | 3.1% (2.2%-4.4%) | 3.9% | 0.7% | unfloored |
| 15 | Student-t location shift | 3.8 | 0.924 | 60,000 | 15,000 | 1,000 | 3.7% (2.7%-5.1%) | 4.3% | 7.1% | unfloored |
| 16 | Student-t location shift | 3.8 | 0.924 | 40,000 | 20,000 | 1,000 | 4.6% (3.5%-6.1%) | 4.5% | 11.6% | unfloored |
| 17 | Student-t location shift | 3.8 | 0.914 | 20,000 | 20,000 | 1,000 | 5.1% (3.9%-6.6%) | 4.3% | 4.2% | unfloored |
| 18 | heteroscedastic Gaussian | 5 | 0.550 | 20,000 | 80,000 | 500 | 4.2% (2.8%-6.3%) | 4.8% | 36.3% | unfloored |
| 19 | heteroscedastic Gaussian | 5 | 0.550 | 5,000 | 45,000 | 1,000 | 4.3% (3.2%-5.7%) | 4.1% | 5.2% | hybrid |
| 20 | heteroscedastic Gaussian | 5 | 0.550 | 20,000 | 40,000 | 1,000 | 3.8% (2.8%-5.2%) | 4.0% | 0.7% | unfloored |
| 21 | heteroscedastic Gaussian | 5 | 0.550 | 20,000 | 20,000 | 1,000 | 3.2% (2.3%-4.5%) | 3.2% | 0.0% | unfloored |

21 of 22 cells' 95% intervals contain the model's best estimate; the exception is cell 12 (observed fewer misses than predicted). Significantly above 5%: cell 2, which the rule sends to the hybrid. All 9 cells whose best estimate exceeds 5% are routed to the hybrid.

## 5. Deriving the rule

### 5.1 Which settings are unsafe

The model is evaluated on a grid of settings: smaller class 10-20,000 (log scale), AUC 0.55-0.99 (probit scale) and class ratios 1, 1.5, 2, 2.33, 3, 4, 6, 9, 13, 19. At each setting it takes the worst case over both families, their shapes and which class is the minority. A setting is *unsafe* if the probability that this worst-case miss rate exceeds 5% is above 10%. Requiring a probability, not only a best estimate, below the target keeps the rule out of regions the data cannot yet vouch for.

75.8% of the grid is unsafe. The rule must exclude every unsafe setting with a larger class of at most 10,000 (56.5% of the grid). Beyond that size the simulated data are sparse, Section 1 shows the floor buys no width there and costs far more compute, and the unsafe settings mostly reflect uncertainty about miss rates just below 5%. Section 5.5 checks the rule there against the replicated cells.

### 5.2 Choosing the rule

Candidate rules are conjunctions of simple checks: a minimum on the smaller class, a cap on the class ratio, a cap on the AUC, and a *corner* check that the AUC is low enough for the sample size. For every combination the thresholds are set to admit as much of the grid as possible without admitting an unsafe setting, then rounded to round numbers on the safe side. The best rule for each number of checks:

| checks | rule | grid share | recommended |
| --- | --- | --- | --- |
| 1 | n_min * (1 - AUC)^1.5 / ratio^0.667 >= 81 | 17.2% |  |
| 2 | AUC <= 0.94 and n_min * (1 - AUC)^1 / ratio^1 >= 55 | 22.6% |  |
| 3 | n_min >= 250 and n_max / n_min <= 6 and n_min * (1 - AUC)^2.5 >= 3.7 | 24.2% | yes |
| 4 | n_min >= 160 and n_max / n_min <= 6 and AUC <= 0.94 and n_min * (1 - AUC)^2 / ratio^0.333 >= 8.7 | 23.9% |  |

The recommended rule is the one with the fewest checks that admits within 1 percentage point of the best rule's grid share.

### 5.3 What each check guards against

Each check is credited with the unsafe settings that it alone excludes. Each turns out to bound one failure mechanism, identified by the family, minority class and shape at which the worst case occurs:

| check | unsafe settings | smaller class | AUC | class ratio | worst family | minority class | typical worst shape |
| --- | --- | --- | --- | --- | --- | --- | --- |
| n_min >= 250 | 406 | 30-182 | 0.550-0.772 | 1-6 | heteroscedastic Gaussian | negative | sigma ratio (positive / negative) 0.2 |
| n_max / n_min <= 6 | 181 | 276-1,098 | 0.550-0.890 | 9-19 | heteroscedastic Gaussian | negative | sigma ratio (positive / negative) 0.2 |
| n_min * (1 - AUC)^2.5 >= 3.7 | 1821 | 276-8,728 | 0.846-0.990 | 1-6 | Student-t location shift | positive | degrees of freedom 2.5 |
| several checks | 7581 | 10-1,098 | 0.550-0.990 | 1-19 | Student-t location shift | positive | degrees of freedom 1.1 |

The last row counts settings that more than one check excludes; they lie deep inside the unsafe region and do not bear on any one check.

### 5.4 Sensitivity, and the current rule

The recommended rule at other assurance levels, and the rule currently in `hybrid_band.md` §4.4 judged by the same test:

| rule | checks | grid share | unsafe settings admitted | worst mean miss rate, n_max <= 10,000 | worst P(miss rate > 5%), n_max <= 10,000 | worst mean miss rate, n_max > 10,000 |
| --- | --- | --- | --- | --- | --- | --- |
| recommended at 80% assurance | n_min >= 210 and n_max / n_min <= 9 and n_min * (1 - AUC)^2.5 / ratio^0.333 >= 2.6 | 27.5% | 0 | 4.66% | 14.8% | 6.46% |
| recommended at 90% assurance | n_min >= 250 and n_max / n_min <= 6 and n_min * (1 - AUC)^2.5 >= 3.7 | 24.2% | 0 | 4.67% | 7.4% | 6.46% |
| recommended at 95% assurance | n_min >= 250 and n_max / n_min <= 4 and n_min * (1 - AUC)^2.5 >= 3.9 | 20.7% | 0 | 4.39% | 4.5% | 6.20% |
| current, judged at 90% | n_min >= 100 and n_min * (1 - AUC)^2.5 / ratio^0.667 >= 12 | 19.4% | 30 | 4.67% | 23.5% | 6.34% |

The checks keep their form across assurance levels; the thresholds move.

### 5.5 Beyond a larger class of 10,000

The rule admits 1,903 grid settings there, 1,028 of them unsafe by the test above. At 847 of those the best estimate is below 5% and only its uncertainty fails the test. The best estimate exceeds 5% at 181, all with a larger class of 19,904 or more, where it extends a rising trend beyond the data. The replicated cells in that range that the rule admits measure the band directly:

| cell | family | shape | AUC | positives | negatives | datasets | observed miss rate (95% interval) | model's best estimate | model's P(miss rate > 5%) | routed to |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14 | Student-t location shift | 4.7 | 0.924 | 34,800 | 8,700 | 1,000 | 3.1% (2.2%-4.4%) | 3.9% | 0.7% | unfloored |
| 15 | Student-t location shift | 3.8 | 0.924 | 60,000 | 15,000 | 1,000 | 3.7% (2.7%-5.1%) | 4.3% | 7.1% | unfloored |
| 16 | Student-t location shift | 3.8 | 0.924 | 40,000 | 20,000 | 1,000 | 4.6% (3.5%-6.1%) | 4.5% | 11.6% | unfloored |
| 17 | Student-t location shift | 3.8 | 0.914 | 20,000 | 20,000 | 1,000 | 5.1% (3.9%-6.6%) | 4.3% | 4.2% | unfloored |
| 18 | heteroscedastic Gaussian | 5 | 0.550 | 20,000 | 80,000 | 500 | 4.2% (2.8%-6.3%) | 4.8% | 36.3% | unfloored |
| 20 | heteroscedastic Gaussian | 5 | 0.550 | 20,000 | 40,000 | 1,000 | 3.8% (2.8%-5.2%) | 4.0% | 0.7% | unfloored |
| 21 | heteroscedastic Gaussian | 5 | 0.550 | 20,000 | 20,000 | 1,000 | 3.2% (2.3%-4.5%) | 3.2% | 0.0% | unfloored |

## 6. Model-free check: the simulated datasets the rule admits

The simulated datasets inside the recommended rule, grouped by family, class ratio and third of the family's shape range (on a log scale). Each group pools easier and harder shapes, so this confirms the rule rather than testing its worst case, which is what the model is for.

| family | class ratio | shape | datasets | miss rate | 95% upper bound |
| --- | --- | --- | --- | --- | --- |
| heteroscedastic Gaussian | 1-2.33 | sigma ratio (positive / negative) 0.2-0.58 | 49,016 | 2.10% | 2.23% |
| heteroscedastic Gaussian | 1-2.33 | sigma ratio (positive / negative) 0.58-1.7 | 66,426 | 2.23% | 2.34% |
| heteroscedastic Gaussian | 1-2.33 | sigma ratio (positive / negative) 1.7-5 | 67,039 | 2.38% | 2.50% |
| heteroscedastic Gaussian | 2.33-6 | sigma ratio (positive / negative) 0.2-0.58 | 1,026 | 3.02% | 4.26% |
| heteroscedastic Gaussian | 2.33-6 | sigma ratio (positive / negative) 0.58-1.7 | 895 | 2.79% | 4.09% |
| heteroscedastic Gaussian | 2.33-6 | sigma ratio (positive / negative) 1.7-5 | 1,436 | 3.55% | 4.64% |
| Student-t location shift | 1-2.33 | degrees of freedom 1.1-3.3 | 68,349 | 2.28% | 2.39% |
| Student-t location shift | 1-2.33 | degrees of freedom 3.3-10 | 60,327 | 2.40% | 2.53% |
| Student-t location shift | 1-2.33 | degrees of freedom 10-30 | 53,892 | 2.23% | 2.36% |
| Student-t location shift | 2.33-6 | degrees of freedom 1.1-3.3 | 962 | 2.91% | 4.17% |
| Student-t location shift | 2.33-6 | degrees of freedom 3.3-10 | 2,985 | 3.52% | 4.24% |
| Student-t location shift | 2.33-6 | degrees of freedom 10-30 | 910 | 2.53% | 3.76% |

Over all 373,263 datasets the miss rate is 2.30%. The highest among groups of at least 200 datasets is 3.55% (heteroscedastic Gaussian, class ratio 2.33-6, sigma ratio (positive / negative) 1.7-5).

## 7. Figure

`surface.png` shows, for five class ratios, the worst-case best estimate of the miss rate (top row; blue below 5%, red above) and the probability that it exceeds 5% (bottom row). The black contour marks the 90% assurance boundary; the solid green line encloses the settings the recommended rule admits, the dashed line those of the current rule. The dotted black line marks a larger class of 10,000, beyond which the rule is not held to the assurance test (Section 5.5), so the green line may cross the black contour to its right. The grey line marks the largest smaller class that can be simulated at that ratio.

## Appendix: best rule for every combination of checks

| checks | rule (exact thresholds) | grid share |
| --- | --- | --- |
| corner + floor + ratio_cap | n_min >= 240.1 and n_max / n_min <= 6 and n_min * (1 - AUC)^2.5 >= 3.683 | 24.7% |
| imbalance_corner + floor + ratio_cap + auc_cap | n_min >= 158.6 and n_max / n_min <= 6 and AUC <= 0.9418 and n_min * (1 - AUC)^2 / ratio^0.333 >= 8.64 | 24.7% |
| corner + floor + ratio_cap + auc_cap | n_min >= 240.1 and n_max / n_min <= 6 and n_min * (1 - AUC)^2.5 >= 3.683 | 24.7% |
| imbalance_corner + floor + ratio_cap | n_min >= 158.6 and n_max / n_min <= 6 and n_min * (1 - AUC)^2.5 / ratio^0.333 >= 4.318 | 23.7% |
| imbalance_corner + ratio_cap + auc_cap | n_max / n_min <= 6 and AUC <= 0.9418 and n_min * (1 - AUC)^1.5 / ratio^0.333 >= 28.83 | 23.5% |
| imbalance_corner + auc_cap | AUC <= 0.9418 and n_min * (1 - AUC)^1 / ratio^1 >= 54.91 | 23.2% |
| imbalance_corner + floor + auc_cap | AUC <= 0.9418 and n_min * (1 - AUC)^1 / ratio^1 >= 54.91 | 23.2% |
| corner + ratio_cap + auc_cap | n_max / n_min <= 6 and AUC <= 0.9418 and n_min * (1 - AUC)^1.5 >= 42.25 | 23.0% |
| corner + ratio_cap | n_max / n_min <= 6 and n_min * (1 - AUC)^2 >= 21.33 | 22.6% |
| corner + floor + auc_cap | n_min >= 1,261 and n_min * (1 - AUC)^2.5 >= 3.683 | 22.5% |
| corner + floor | n_min >= 1,261 and n_min * (1 - AUC)^2.5 >= 3.683 | 22.5% |
| imbalance_corner + ratio_cap | n_max / n_min <= 6 and n_min * (1 - AUC)^2 / ratio^0.333 >= 18.76 | 21.4% |
| imbalance_corner + floor | n_min >= 1,261 and n_min * (1 - AUC)^2.5 / ratio^0.333 >= 3.53 | 21.3% |
| corner + auc_cap | AUC <= 0.9418 and n_min * (1 - AUC)^0.5 >= 739.4 | 19.7% |
| floor + ratio_cap + auc_cap | n_max / n_min <= 6 and AUC <= 0.8618 and n_min >= 316.6 | 18.2% |
| floor + auc_cap | AUC <= 0.8899 and n_min >= 1,261 | 17.9% |
| imbalance_corner | n_min * (1 - AUC)^1.5 / ratio^0.667 >= 80.23 | 17.3% |
| corner | n_min * (1 - AUC)^1 >= 497.9 | 16.4% |
| floor | n_min >= 1.002e+04 | 7.6% |
| floor + ratio_cap | n_min >= 1.002e+04 | 7.6% |
| auc_cap | AUC <= 0.55 | 0.0% |
| ratio_cap | n_max / n_min <= 1 | 0.0% |
| ratio_cap + auc_cap | n_max / n_min <= 1 | 0.0% |

Model length scales (positive corner, negative corner, AUC, shape): Student-t location shift [0.172, 0.346, 0.51, 0.321], amplitude 1.36; heteroscedastic Gaussian [0.201, 0.209, 0.848, 0.172], amplitude 1.45.
