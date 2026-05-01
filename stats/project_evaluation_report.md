# Project Evaluation: Simultaneous Confidence Bands for ROC Curves

*Assessment of the studentized bootstrap envelope method and its competitors, based on code review, theoretical analysis, and 2.25M simulation evaluations.*

---

**Implementation provenance note:** The large simulation suite summarized here was run against the earlier `envelope_wilson` implementation that used hard effective-count cutoffs for Wilson tail support (`tail_k_min=(15, 10)`, `tail_m_min=10`). The current code has since moved to a variance-ratio Wilson floor (`bootstrap_var / wilson_var`) with continuous effective dimensionality for the Sidak correction. The hard-cutoff mechanism is therefore the right explanation for the stored 2.25M-evaluation results; the variance-ratio implementation should be treated as a promising follow-up that still needs the same full evaluation pass.

## A. Motivation: The Gap Between Working-Hotelling and KS

### The practical problem

A practitioner computes an ROC curve from a finite sample and wants to know: where could the *true* ROC curve plausibly lie? This requires a **simultaneous** confidence band -- one that covers the entire curve at once, not just individual points. Two classical answers exist:

**Working-Hotelling (WH):** Assumes scores in both classes follow normal distributions (the binormal model), estimates two parameters (a, b) relating the probit-transformed ROC to a linear function, and constructs a chi-squared-based band in probit space. When the binormal assumption holds, this is efficient. When it fails -- heavy tails, multimodality, skew -- coverage collapses catastrophically (simulation evidence: <20% coverage for Student-t(3) at n=1000).

**Kolmogorov-Smirnov (KS):** Uses the DKW inequality to construct a fixed-width band around the empirical CDF of each class, then propagates to the ROC. Makes no distributional assumptions and has a finite-sample distribution-free guarantee. In this simulation suite it achieves 100% empirical coverage at every sample size and DGP. The price: bands are so wide that the 50% confidence band still achieves 98% coverage -- the method communicates little about *where* within the band the true ROC lies.

### The gap

The gap between WH and KS is the gap between *assuming everything* and *assuming nothing*. WH assumes a parametric model and gets tight bands when right, garbage when wrong. KS assumes nothing and gets valid but often uninformative bands.

The practical need is a method that:
1. Makes no distributional assumptions (or very weak ones)
2. Achieves approximately nominal coverage (~95%) across a wide range of DGPs
3. Produces bands tight enough to be informative
4. Degrades gracefully in typical failures -- small, localized violations rather than frequent catastrophic misses

This is a nonparametric inference problem with a simultaneity constraint, and it is genuinely hard.

### The most fruitful framing

The core difficulty is not "how to build a confidence band" -- there are many ways. The core difficulty is **the boundary problem**: the ROC curve is a function on [0,1] whose values are bounded in [0,1], and the most informative parts of the curve (where it rises steeply, near FPR=0) are precisely where nonparametric methods have the least information.

Any method for ROC confidence bands must solve three sub-problems:
1. **Interior estimation**: How to quantify uncertainty at FPR values where both classes have adequate representation above/below threshold
2. **Boundary correction**: How to handle the corners (0,0) and (1,1) where bootstrap variance collapses and parametric assumptions are least testable
3. **Simultaneity control**: How to ensure the band covers the *entire* curve, not just individual points

Different methods make different trade-offs across these three sub-problems. The bootstrap envelope excels at (1), struggles with (2), and has an unusual approach to (3) via the envelope operator. Understanding where each method wins requires understanding which sub-problem dominates in a given regime.

---

## B. Theory of the Studentized Bootstrap Envelope

### B.1 The construction

The method proceeds in five stages:

**Stage 1 -- Bootstrap.** Generate B stratified bootstrap ROC curves by resampling n_0 negatives and n_1 positives with replacement. Each bootstrap curve R_b(t) is a step function on the same FPR grid as the empirical ROC.

**Stage 2 -- Studentize.** At each FPR grid point t, compute the bootstrap standard deviation sigma_hat(t) and form the studentized deviation:

    z_b(t) = (R_b(t) - R_hat(t)) / sigma_hat(t)

This normalization makes deviations comparable across the FPR grid, accounting for the fact that variance is heteroscedastic (higher in the middle of the ROC, lower at boundaries).

**Stage 3 -- Retain.** Compute the supremum studentized statistic Z_b = sup_t |z_b(t)| for each bootstrap curve. Retain the (1-alpha) fraction with smallest Z_b. This is analogous to inverting a bootstrap KS test: we keep curves that are "consistent" with the empirical ROC.

**Stage 4 -- Envelope.** Take the pointwise minimum and maximum of retained curves:

    L(t) = min{R_b(t) : b retained}
    U(t) = max{R_b(t) : b retained}

**Stage 5 -- Boundary correction.** Apply Wilson Rectangle or KS-style extension in tail regions.

### B.2 Why studentization helps

Without studentization, the retention criterion is max_t |R_b(t) - R_hat(t)|. This treats a 0.01 deviation at FPR=0.5 (where sigma ~ 0.03) the same as a 0.01 deviation at FPR=0.02 (where sigma ~ 0.005). The result: interior deviations dominate the KS statistic, and boundary curves are retained even when they deviate far from the truth relative to local uncertainty. The band is too tight at boundaries, too wide in the interior.

Studentization reweights by local precision: a 0.01 deviation at a low-variance point is "stranger" than a 0.01 deviation at a high-variance point. This produces bands that adapt their width to local uncertainty -- the defining advantage over fixed-width methods like KS.

### B.3 Theoretical coverage guarantee (asymptotic)

The studentized bootstrap for the supremum functional is well-studied (e.g., Horowitz 2001, Hall & Horowitz 1993). Under regularity conditions (continuous score distributions, finite variance of R(t) at all t, smooth population ROC), the bootstrap distribution of sup_t |z_b(t)| converges to the true distribution of the studentized supremum process.

The coverage argument proceeds by inverting the bootstrap KS test. Define the true studentized supremum:

    Z = sup_t |(R_hat(t) - R_true(t)) / sigma(t)|

Bootstrap consistency means the bootstrap Z_b distribution converges to the distribution of Z, so the bootstrap (1-alpha)-quantile c_alpha converges to the true quantile. The event {Z <= c_alpha} is equivalent to R_true(t) lying within [R_hat(t) - c*sigma(t), R_hat(t) + c*sigma(t)] at every grid point t. Therefore:

    P(R_true(t) in [L(t), U(t)] for all t) -> 1 - alpha as n -> infinity

This is an asymptotic guarantee. The rate of convergence depends on the smoothness of the score distributions and the quality of the variance estimate.

### B.4 The envelope operator is a central-region projection

The retained set is defined by a KS-type event, but the final object is not exactly the rectangular tube `R_hat(t) +/- c*sigma(t)`. Every retained curve satisfies `Z_b <= c`, so each retained curve lies inside that tube. The envelope of retained curves is therefore bounded by the tube, but it need not reach the tube at every FPR.

This distinction matters. The bootstrap curves are not arbitrary functions inside a rectangle; they are monotone ROC step functions generated by resampling the observed score distributions. Their feasible deviations are correlated across FPR, asymmetric near 0 and 1, and constrained by the empirical support. The envelope is a projection of this central cloud of feasible ROC curves onto pointwise lower and upper bounds.

That projection has two consequences:

1. **Shape robustness:** The band inherits monotonicity, boundedness, and the empirical dependence structure instead of pretending that each FPR can move independently.
2. **Calibration opacity:** Coverage is no longer a direct inversion of a rectangular studentized band. The bootstrap KS statistic controls which curves enter the central region, while the pointwise min/max projection determines the reported band.

This is a better explanation for the simulation behavior than saying the envelope "equals" the theoretical band. The variance-model experiment in G.1 is useful here: replacing the envelope by `R_hat +/- c*sigma_hat` did not solve the 50% calibration problem and often made calibration worse, so the issue is not only the min/max projection. But the envelope operator still contributes finite-sample behavior that is not captured by the clean asymptotic tube argument.

### B.4a Finite-sample conservatism and the 50% CI problem

The simulation shows massive over-coverage at the 50% CI level (85% actual vs 50% nominal), decreasing with n but persisting even at n=10,000 (64%). This is not a mystery once the method is viewed as a sup-norm-calibrated central region.

**Effect 1: The Wilson floor.** At small n, the Wilson floor widens the band far beyond what the bootstrap alone would produce, dominating overall coverage. At n=10, the floor covers the entire grid. This is the primary source of over-coverage for n <= 100.

**Effect 2: Bootstrap discreteness and support limitation.** Bootstrap ROC curves are step functions with jumps at multiples of 1/n_0 (FPR) and 1/n_1 (TPR), and they can only reuse observed score values. This can make the finite-sample bootstrap process rougher than the smooth target process in the interior while simultaneously too optimistic at the extreme tails. The net effect at 50% confidence is excess width in most settings, especially when the Wilson floor is active.

The 50% CI is more sensitive to this bias than the 95% CI because the theoretical 50% band is narrow -- small overestimation of c produces proportionally large excess width. The bias diminishes with n as the step-function discreteness becomes finer.

**The sup-norm's weak sensitivity to alpha.** An additional factor is that `c_0.95/c_0.50` for a supremum statistic over many correlated grid points is modest -- typically 1.3-1.5x depending on K and the correlation structure. So the 95% band is only moderately wider than the 50% band. Combined with finite-sample conservatism, this means coverage at 50% can sit far above nominal even when coverage at 95% is near-exact. This is a property of sup-norm-calibrated simultaneous bands, not just this envelope implementation.

### B.5 The boundary problem

This is the method's fundamental weakness. At FPR = k/n_0 for small k:

1. The empirical ROC is a step function with jump height determined by few observations
2. Bootstrap variance is driven by small-count combinatorics of resampled negatives
3. The bootstrap can only explore score values present in the observed data

The bootstrap variance at these points is not wrong -- it correctly represents the variability of the *bootstrap mechanism*. But the bootstrap mechanism itself cannot represent uncertainty about population probability mass in unobserved regions of score space. If the true negative score distribution has probability mass beyond the most extreme observed negative score, the bootstrap has no way to know this.

**Result:** At boundary points, all bootstrap curves agree on approximately the same TPR. The envelope has near-zero width. If R_true deviates from R_hat at any such point, coverage fails.

This is not a bug in the implementation -- it is a structural limitation of the nonparametric bootstrap for extreme quantiles. No modification to the studentization, variance floor, or retention criterion changes this. The bootstrap tail problem is *why* the Wilson floor exists.

### B.6 The Wilson floor as a hybrid correction

The Wilson floor addresses the boundary problem by importing a parametric assumption: at each grid point, TPR is treated as a binomial proportion with n_1 trials. The Wilson score interval for this proportion provides a minimum uncertainty that is always positive, even at p=0 or p=1.

The Wilson floor is applied in two places:

**During studentization (Stage 2):** The bootstrap variance is floored to at least the Wilson variance. This prevents studentized statistics from exploding at zero-variance points, keeping the retention criterion well-behaved.

**After envelope construction (Stage 5):** In tail regions (defined by effective count thresholds k_min, m_min), Sidak-corrected Wilson Rectangle bounds are applied as a floor on the envelope. This directly widens the band at boundary points.

**What the Wilson floor captures:** The binomial component of ROC uncertainty: R(t)(1-R(t))/n_1. This is the variance of TPR *given a fixed threshold*. It is always present and does not depend on the score distribution.

**What the Wilson floor misses:** The threshold-uncertainty component: [g(c_t)/f(c_t)]^2 * t(1-t)/n_0, from the Hsieh-Turnbull formula. This is the additional variance from not knowing exactly which score threshold corresponds to FPR=t. In the interior, the bootstrap captures both components. At the boundary, the Wilson floor captures only the first.

**Why restrict to tails:** In the interior, the bootstrap variance is strictly more informative than the Wilson model because it captures both variance components and adapts to the actual shape of the ROC curve. Applying the Wilson floor everywhere would replace good estimates with worse ones. Restricting to tails (k < k_min or m < m_min) limits the parametric assumption to where it is needed.

### B.7 Scaling and the coverage trajectory

The tail region is defined by fixed count thresholds (k_min=15, m_min=10). As n grows, the fraction of the FPR grid in the tail region shrinks:

| n_0  | Approximate tail fraction | Wilson's role |
|------|--------------------------|---------------|
| 15   | 100%                     | Drives everything |
| 50   | ~50%                     | Major contributor |
| 150  | ~17%                     | Tail correction only |
| 500  | ~5%                      | Minor correction |
| 5000 | ~0.5%                    | Negligible |

This explains the observed coverage trajectory:
- **n <= 30:** Wilson dominates, coverage ~ 100% (over-conservative)
- **n ~ 100-300:** Wilson covers tails, bootstrap conservatism (step-function bias) provides additional over-coverage, near-nominal balance at 95%
- **n ~ 1000:** Sweet spot -- small tail correction, bootstrap bias diminishing, near-exact at 95%
- **n >= 10000:** Wilson negligible, bootstrap bias negligible, bare bootstrap boundary problem exposed, coverage drops to ~83%

This fixed-threshold scaling is the mechanism behind the stored simulation suite. The current variance-ratio implementation is a cleaner attempt to make the tail/interior split data-adaptive, but it should not be used retroactively to explain the large simulation results until it has been rerun at comparable scale.

### B.8 The near-boundary gap

Between "tail" (Wilson active) and "interior" (bootstrap reliable) there exists a transition zone: grid points with effective counts just above the threshold (k = 15-50). Here:
- Bootstrap variance is non-zero but unreliable (small-count effects)
- Wilson floor is not applied (k >= k_min)
- True ROC has genuine uncertainty that bootstrap underestimates

The near-boundary problem is not just "too few observations." It is specifically a **threshold-location problem**. At fixed FPR t, the operating threshold is an estimated negative-class quantile. Error in that threshold is converted into TPR error by the ROC slope, `R'(t) = g(c_t) / f(c_t)`. When AUC is high, the low-FPR ROC slope can be large, so small threshold errors create large downward misses. Wilson ignores this channel entirely; the bootstrap captures it in the interior but underrepresents it near the empirical support boundary.

As n grows, more grid points enter this gap (the tail shrinks, the gap persists), accumulating opportunities for small violations. This is the proximate cause of large-n coverage degradation.

### B.9 Expected failure modes

| Failure mode | Severity | When it occurs | Mechanism |
|---|---|---|---|
| Over-conservative at 50% CI | Moderate | Always, diminishing with n | Wilson floor (small n) + bootstrap step-function conservatism + weak sup-norm sensitivity to alpha |
| Over-conservative at small n | Mild | n <= 30 | Wilson floor dominates |
| Under-coverage at large n | Notable | n >= 10000 | Wilson floor vanishes, bootstrap boundary problem exposed |
| Under-coverage at high AUC | Notable | AUC > 0.9, n >= 1000 | Steep ROC concentrates information in boundary region |
| Lower-bound optimism | Common among failures | Most visible at high AUC and large n | Empirical ROC is upward-biased; true ROC falls below the lower band |
| Rare large misses | Rare but real | High AUC, low FPR, usually lower-bound failures | Tail support miss; P99 is small, but maxima can be large |

The method's typical failure mode is benign: at n=10,000 the mean max violation is ~0.002 and P99 is ~0.046, with violations concentrated in the first 10% of FPR. But the absolute maximum is not benign; rare low-FPR lower-bound failures exceed 50pp of TPR in the stored simulations. The accurate claim is therefore: violations are usually small and spatially concentrated, while rare high-AUC tail realizations can still be large.

---

## C. Overview of Alternative Methods

### C.1 Kolmogorov-Smirnov (KS) Band

**Approach:** Fixed-width band based on the DKW inequality. The band width d = c_alpha / sqrt(n_eff) is uniform across the entire ROC curve. This is a distribution-free simultaneous confidence band with guaranteed finite-sample coverage.

**Strengths:**
- Distribution-free finite-sample coverage guarantee, with 100% empirical coverage in this simulation suite
- No tuning parameters beyond alpha
- No distributional assumptions
- Simple to implement and understand

**Weaknesses:**
- Fixed width does not adapt to local variance
- Bands are uninformatively wide (50% band achieves 98% coverage)
- Width scales as O(1/sqrt(n)), same as any nonparametric method, but with a large constant

**Role in the landscape:** The KS band is the gold standard for safety. Any practical method should be judged by how much tighter it is than KS while maintaining acceptable coverage. It is not literally information-free, but it is so conservative here that even the 50% band covers 98% of the time.

### C.2 Working-Hotelling (Binormal)

**Approach:** Assumes scores follow normal distributions in both classes. Fits the binormal ROC model R(t) = Phi(a + b * Phi^{-1}(t)), constructs a band in probit space using the chi-squared critical value for 2 degrees of freedom.

**Strengths:**
- Tight bands when the binormal assumption holds
- Closed-form: no bootstrap, no iteration
- Well-studied asymptotic theory

**Weaknesses:**
- Catastrophic failure under misspecification (coverage < 20% for heavy-tailed data)
- Not suitable without prior distributional knowledge
- The binormal assumption is untestable in practice (you can test normality of scores, but not whether the *ROC* is binormal -- the ROC can be approximately binormal even when scores are non-normal)

**Role in the landscape:** A benchmark for what tight bands look like under ideal conditions. Not a practical recommendation for general use.

### C.3 Ellipse-Envelope (Demidenko 2012)

**Approach:** An improvement on Working-Hotelling that accounts for the estimation of variances rather than treating them as known. Constructs confidence ellipses at each threshold and takes their envelope.

**Strengths:**
- Tighter than WH for moderate sample sizes
- Proper accounting for parameter estimation uncertainty

**Weaknesses:**
- Same binormal assumption as WH -- same catastrophic failure modes
- More complex implementation (quartic polynomial solver or sweep)
- Numerical instability at extreme thresholds

**Role in the landscape:** A refinement of WH, not a fundamentally different approach. Shares WH's fatal flaw of parametric dependence.

### C.4 Hsieh-Turnbull with Density Estimation

**Approach:** Uses the asymptotic variance formula Var(R(t)) = R(t)(1-R(t))/n_1 + [g(c)/f(c)]^2 * t(1-t)/n_0, where f,g are score densities estimated via log-concave MLE or reflected KDE. Optional bootstrap calibration of the critical value.

**Strengths:**
- Captures both components of ROC variance (binomial + threshold uncertainty)
- Best calibrated at both 95% and 50% levels among all non-bootstrap methods
- With bootstrap calibration (autocalib), respects the actual correlation structure of the ROC curve
- Log-concave density estimation is semiparametric -- weaker assumption than binormality

**Weaknesses:**
- Log-concavity assumption excludes multimodal and heavy-tailed distributions
- Coverage is inconsistent across sample sizes (0.746 at n=300, 0.967 at n=1000 with autocalib)
- Density ratio g(c)/f(c) estimation is inherently unstable in the tails
- Reflected KDE has bandwidth sensitivity and boundary artifacts

**Role in the landscape:** The most principled analytical approach. If you could trust the density estimates, this would be the best method. The density estimation problem *is* the obstacle.

### C.5 Wilson Rectangle Band

**Approach:** Constructs 2D Wilson score confidence rectangles at each operating point (FPR, TPR). The band envelope comes from the upper-left corners (optimistic: low FPR, high TPR) and lower-right corners (pessimistic: high FPR, low TPR).

**Strengths:**
- Always-positive width, even at boundaries (Wilson's defining property)
- No distributional assumptions
- Tightest bands among methods with >= 90% coverage (area = 0.331)
- Simple closed-form computation

**Weaknesses:**
- Pointwise method with Sidak/Bonferroni correction -- not truly simultaneous
- Coverage degrades at large n (too many test points for the correction to handle)
- No adaptation to local ROC shape
- The 2D rectangle model treats FPR and TPR uncertainty as independent, which they are not (both depend on the same threshold)

**Role in the landscape:** The practical "quick and dirty" method. Good coverage at moderate n, tight bands, simple to implement. The lack of true simultaneity correction is its fundamental limitation. However, Wilson's always-positive-width property makes it an excellent building block for other methods (as the envelope_wilson method demonstrates).

---

## D. Empirical Findings and Implications

### D.1 The main result

At the standard reporting level (95% CI), `envelope_wilson` achieves essentially exact coverage (0.950) across all 7 DGPs tested, for sample sizes n=30 to n=1000. This is the headline finding: a nonparametric method that achieves nominal coverage without distributional assumptions, at the confidence level practitioners actually use.

No other nonparametric method achieves this. The KS band achieves 100% but is uninformative. The Wilson Rectangle with Sidak correction achieves 91% -- close but not quite. The Hsieh-Turnbull methods achieve 89-90% overall but with highly variable coverage across sample sizes and DGPs.

### D.2 The 50% CI problem

All envelope methods show massive over-coverage at the 50% level (85% actual vs 50% nominal at moderate n, decreasing to 64% at n=10,000). This comes from two finite-sample effects, not from a structural defect in the envelope operator:

1. **Wilson floor** (dominant at small n): directly widens the band beyond what the bootstrap produces.
2. **Bootstrap step-function conservatism**: the discrete jumps of bootstrap ROC curves inflate the supremum statistic relative to the smooth population process, biasing the critical value upward and making the band too wide.

Both effects diminish with n. The 50% CI is disproportionately affected because the theoretical 50% band is narrow, so small upward biases in the critical value produce proportionally large excess coverage. Additionally, the sup-norm's critical value ratio c_{0.95}/c_{0.50} is modest (typically 1.3-1.5x for correlated processes), meaning the 50% band is only modestly narrower than the 95% band to begin with.

**Implication:** The envelope method is best used for high-confidence bands (90-99%). The 50% over-coverage is not inherent to the envelope operator -- it is a finite-sample artifact that diminishes with n -- but in practice it means the method communicates limited information at lower confidence levels for the sample sizes where it is most useful (n <= 1000).

**Contrast with other simultaneous methods:** Any sup-norm-based simultaneous band (not just the envelope) would exhibit weak sensitivity of width to alpha. The HT-autocalib method avoids this because it uses a *pointwise* variance model scaled by a *single* critical value, giving it continuous tunability across confidence levels. This is a genuine structural advantage of variance-model-based approaches over envelope approaches.

### D.3 The large-n problem

Coverage drops to 0.830 at n=10,000. The theoretical analysis identifies the mechanism: the Wilson tail correction vanishes as n grows (covering only ~0.5% of the FPR grid at n=5000), exposing the bare bootstrap's boundary problem.

**Key finding:** Violation magnitudes are usually small even when coverage is lost. Mean max violation at n=10,000 is ~0.002 (0.2pp of TPR). The P99 max violation is ~0.046. Only 0.84% of n=10,000 simulations have any violation exceeding 5pp. Across all sample sizes, however, rare high-AUC tail cases can be much larger: the overall max is 0.668, the P99.9 is 0.141, and 0.69% of 95% CI simulations exceed 5pp.

**Direction and geography:** At 95% confidence, failures are predominantly lower-bound failures: `violation_below` occurs in 4.65% of simulations versus `violation_above` in 0.39%. At n=10,000 this imbalance widens to 15.69% below versus 1.49% above. Regionally, the first 10% of FPR dominates: at n=10,000, the 0-10% FPR region violates in 15.09% of simulations, versus 0.77% in 10-30% and below 0.65% in each higher-FPR region.

**Implication:** The coverage drop at large n is mostly a *technical* failure (the true ROC escapes the band by tiny amounts), but not always. Whether this matters depends on the use case:
- For regulatory submissions where a stated 95% guarantee must hold formally: this is a problem.
- For exploratory analysis where the question is "roughly where is the true ROC?": this is fine.
- For high-AUC applications where decisions depend on very low FPR, the rare large lower-tail misses are directly relevant and should not be averaged away.

### D.4 DGP robustness is the standout property, conditional on AUC

Coverage is much more stable across DGP families than across AUC and sample size. At 95% confidence, coverage ranges are about 1.1pp at n=30, 3.5pp at n=100, 6.3pp at n=300, 3.9pp at n=1000, and 5.2pp at n=10000. That is not literally "the same coverage," but it is far more robust than the parametric competitors.

The sharper abstraction is: **distribution family is second-order; ROC geometry is first-order.** High AUC creates a steep low-FPR segment, and that is where the bootstrap tail/support problem lives. Within each DGP, the high-AUC subset is consistently worse than the rest. For example, at n=10,000, `envelope_wilson` coverage is 0.848 for AUC < 0.9 but 0.752 for AUC >= 0.9. In the high-AUC Student-t subset, overall coverage drops to 0.875 and P99 max violation rises to 0.163.

Compare Working-Hotelling: coverage ranges from <20% (Student-t with low df) to >95% (heteroscedastic Gaussian). The parametric methods live or die by their assumptions; the bootstrap doesn't care.

**Implication:** This is the strongest argument for the bootstrap approach, but it should be framed correctly. In practice, the data scientist often does not know the DGP, but they can estimate the empirical ROC geometry. A method whose main risk is tied to observable geometry (high AUC, steep low-FPR slope) is easier to diagnose than a method whose risk is tied to unverified distributional assumptions.

### D.5 The Wilson floor ablation

The simulation report includes `envelope_standard` (no Wilson floor), which achieves ~50% coverage. This confirms the theoretical prediction: the bare bootstrap envelope is not a valid confidence band, at any sample size, because boundary variance collapse causes systematic under-coverage.

The Wilson floor is not a "band-aid" -- it is a *necessary correction* for a structural limitation of the nonparametric bootstrap. Without it, the method doesn't work. With the hard-cutoff version used for the big simulations, the method works well for n <= 1000 and usually degrades gracefully beyond, but it still has rare large low-FPR lower-bound failures.

### D.6 Logit-space construction hurts

All logit-transformed envelope methods show dramatically worse coverage (35-40% at 95% CI). This is surprising given that logit transforms are standard variance-stabilizing tools.

**Likely explanation:** The Haldane-Anscombe correction maps TPR=0 and TPR=1 to finite values rather than +/- infinity, but the logit transform still stretches the boundaries. Curves that are tightly clustered near TPR=0 in probability space become spread out in logit space, but in a way that doesn't improve the boundary problem. The logit transform was designed for pointwise intervals (where it prevents the band from escaping [0,1]); for the envelope operator, the [0,1] constraint is already enforced by clipping, and the logit distortion just makes the envelope wider in the interior (where it was already over-conservative) without helping at the boundary (where the bootstrap has zero variance regardless of the transform).

### D.7 Hsieh-Turnbull's calibration advantage

The `HT_log_concave_logit_autocalib_wilson` method has the best overall calibration (smallest total deviation from nominal at both 95% and 50% levels). Its 50% CI coverage is 0.611 -- much closer to the 0.50 target than any envelope method's 0.85.

**Why:** HT uses pointwise variance estimates scaled by a single critical value, not an envelope operator. The critical value can be smoothly adjusted by the bootstrap calibration step. This gives HT *continuous tunability* across confidence levels. Any sup-norm-based simultaneous method (including the envelope) has inherently weak sensitivity to alpha because the critical value ratio c_{0.95}/c_{0.50} is modest for correlated processes. HT avoids this because its band width is directly proportional to z * SE(t), where z is the single critical value.

**The trade-off:** HT requires density estimation. When the log-concavity assumption fails, coverage collapses. The method is best-calibrated *conditional on the assumption holding*, but fragile to violations.

---

## E. Progress, Walls, and Remaining Uncertainties

### E.1 Important progress

1. **The Wilson floor is a genuine contribution.** It transforms a broken method (50% coverage) into a near-exact one (95% coverage for n <= 1000). The insight that the bootstrap boundary problem can be patched with a simple binomial correction -- and that this correction should be restricted to the tails to preserve the bootstrap's advantages in the interior -- is the core intellectual contribution of the project.

2. **The simulation study is comprehensive and well-designed.** 7 DGPs, 6 sample sizes, 23 methods, 1000 LHS combinations per DGP, multiple confidence levels. The Latin Hypercube sampling over DGP parameters is a particularly good choice: it ensures coverage of the parameter space without exponential blowup. The evaluation framework (BandResult, BandEvaluation) is clean and the metrics are well-chosen.

3. **The theoretical analysis correctly identifies the three-region model** (tail / near-boundary / interior) and explains the coverage trajectory. This is the right framework for understanding the method's behavior.

4. **Typical graceful degradation is real, but the tail risk needs clearer language.** At n=10,000, mean max violation is ~0.002 and P99 is ~0.046, so most failures are tiny. But rare high-AUC, low-FPR lower-bound failures are large in absolute TPR. The qualitative difference from parametric methods is not "this method never misses badly"; it is "most misses are small and localized, and the large misses occur in a diagnosable low-FPR/high-AUC regime."

### E.2 Walls

1. **Sup-norm-based simultaneous bands have weak sensitivity to alpha.** The critical value ratio c_{0.95}/c_{0.50} for a supremum statistic over many correlated grid points is inherently modest (~1.3-1.5x). This means the 50% band is only modestly narrower than the 95% band, regardless of the construction method (envelope, studentized band, etc.). Combined with finite-sample bootstrap conservatism, this makes the 50% band substantially over-conservative in practice. The over-coverage diminishes with n but remains noticeable for the sample sizes where the method is most useful. This is not specific to the envelope operator -- it affects any method that determines simultaneous coverage via a sup-norm critical value. **Confirmed empirically (G.1):** A variance-model band (R_hat ± c*sigma_hat, no envelope operator) was implemented and tested. It does not improve 50% CI calibration relative to the envelope, and in most scenarios makes it worse, because the same sup-norm mechanism governs c for both methods. The weak sensitivity to alpha is a property of the supremum statistic itself, not of the envelope construction.

2. **The large-n coverage gap has no clean fix within the hard-cutoff framework.** The near-boundary zone (k = 15-50) is where the bootstrap has *some* variance but not enough. Options:
   - **Raise k_min:** Shifts the wall but makes the Wilson floor cover more of the curve, reducing the bootstrap's contribution. At k_min = 50, the Wilson floor covers ~10% of the grid at n=500 -- this starts to defeat the purpose.
   - **Use an adaptive threshold:** Make k_min scale with n (e.g., k_min = c*sqrt(n_0)). This would keep the tail fraction constant across sample sizes. But the choice of c is arbitrary and the theory doesn't tell you the right value.
   - **Smooth the transition:** Instead of a hard cutoff between "Wilson active" and "Wilson off," blend the Wilson floor with the bootstrap variance using a weight that decreases with k. This is more principled but adds complexity and tuning parameters.
   - **Use a variance-ratio gate:** This is now implemented in current code and removes the magic count thresholds, but it has not yet been validated against the full simulation grid that produced this report.

3. **The bootstrap can't help at the boundary.** This is the deepest wall. The nonparametric bootstrap resamples from the empirical distribution and therefore cannot represent uncertainty about probability mass beyond the observed data. At FPR=0, this means the bootstrap has exactly zero information about R_true(0). No clever resampling scheme fixes this within the nonparametric framework.

### E.3 Remaining uncertainties

1. **Is the large-n coverage gap practically important?** For most ROC analyses in biomedical and ML applications, n < 1000. The method works well in this range. Whether n=10000 coverage of 83% matters depends on the field and the stakes. This is an empirical question about use cases, not a technical question.

2. **Can the near-boundary zone be better served?** The three-region model suggests a principled intervention: use the Hsieh-Turnbull variance (which captures both variance components) as a floor in the near-boundary zone, falling back to Wilson only in the true tails where density estimation is impossible. This would require density or slope estimation at near-boundary thresholds, which may be feasible when there are roughly 15-50 negatives above threshold. This idea has not been tested.

3. **What is the right comparison for "stapling on fixes"?** The concern that other methods would be equally good with similar engineering effort is legitimate. The HT method with bootstrap calibration and Wilson floor already achieves 0.895 coverage with better 50% CI calibration. If the density estimation step were made more robust (e.g., using a more forgiving nonparametric estimator, or using the bootstrap to select among estimators), HT might close the gap to the envelope at 95% while maintaining its advantage at 50%.

4. **Is symmetric retention worth the complexity?** `envelope_wilson_symmetric` achieves nearly identical performance to `envelope_wilson` (0.946 vs 0.950 coverage). The theoretical motivation (addressing asymmetric alpha mass at high AUC) is sound, but the empirical improvement is negligible. This suggests the standard KS retention already handles the asymmetry adequately, and the symmetric correction is redundant in practice.

5. **The interaction between Harrell-Davis smoothing and the envelope.** HD smoothing reduces the discreteness of individual bootstrap ROC curves, potentially allowing finer-grained retention. The simulation includes HD variants but the report doesn't analyze them in detail. If HD smoothing doesn't help, it suggests the discreteness of the bootstrap ROC is not the binding constraint.

---

## F. Best Ideas Across All Methods

### F.1 Ideas that work well

**Wilson's always-positive width (from Wilson Rectangle).** The Wilson score interval guarantees non-zero width at p=0 and p=1. This is the key property that makes the boundary correction work. Any future method should use Wilson (not Wald) intervals whenever binomial proportions appear.

**Bootstrap calibration of the critical value (from HT-autocalib).** Instead of relying on asymptotic theory or Bonferroni correction to determine the simultaneous critical value, generate bootstrap replicates of the test process and find the empirical quantile. This respects the actual correlation structure of the ROC curve. The implementation in `hsieh_turnbull_band.py` is clean: generate bootstrap ROCs, compute the sup-statistic, take the (1-alpha)-quantile. **Important caveat (from G.1 findings):** This idea is *not* straightforwardly portable to any variance-based method. It works well with HT because the HT variance is a smooth analytical function of t -- every bootstrap replicate is studentized against the same stable denominator. When studentizing against pointwise bootstrap variance instead (which is noisy, varying across bootstrap replicates and grid points), the supremum selects grid points with accidentally low variance, inflating c_alpha and producing over-conservative bands. Bootstrap calibration requires a smooth variance estimate to be effective.

**Adaptive variance from the bootstrap (from envelope_boot).** The bootstrap variance at each FPR grid point captures both the binomial and threshold-uncertainty components of ROC variance, adapts to the actual shape of the score distributions, and makes no parametric assumptions. In the interior of the ROC, this is the best variance estimate available. However, it is noisy at each grid point (sampling variance from B replicates), and this noise interacts badly with supremum-based calibration (see caveat under bootstrap calibration above). Its strength is as an input to the *envelope operator*, which tolerates pointwise noise because it takes min/max of retained curves rather than studentizing against variance.

**Sidak correction for tail points (from envelope_boot).** When applying separate corrections at K_tail points, using alpha_tail = 1 - (1-alpha)^{1/K_tail} is exact for independent tests and usually less conservative than Bonferroni. It is a reasonable engineering choice for the hard-cutoff tail region, but it is not a universal dependence-robust guarantee; arbitrary dependence requires Bonferroni or an empirically calibrated correction.

**Restricting parametric corrections to where they're needed (from envelope_boot).** The three-region model -- parametric floor in tails, bootstrap in interior, nothing in between -- is the right architecture. The insight that the Wilson floor should *not* be applied everywhere is important: it preserves the bootstrap's advantages in the interior while patching its weakness at the boundary.

### F.2 Ideas that could be developed further

**Hsieh-Turnbull variance as a variance floor in the near-boundary zone.** The current method uses Wilson variance (binomial component only) as a floor. The HT variance captures both components but requires estimating the density ratio, equivalently the ROC slope. In the near-boundary zone (k = 15-50), there may be enough local information for a rough slope estimate, especially if the estimate is regularized, monotone-smoothed, or pooled across neighboring FPRs. Using HT-style variance as a floor in this zone -- falling back to Wilson in the true tails -- could close the large-n coverage gap. This is the most promising unexplored direction.

**Bootstrap-calibrated Wilson bands.** The Wilson Rectangle method with Sidak correction achieves 0.911 coverage and is the tightest method with >= 90% coverage. Its main weakness is the lack of true simultaneity control. What if you used bootstrap calibration (as in HT-autocalib) to determine the critical value z, but applied it to Wilson intervals instead of HT intervals? This would combine Wilson's always-positive-width property with bootstrap-calibrated simultaneity. **Caveat from G.1:** The Wilson variance is a smooth function of t (it depends only on the empirical TPR and n_pos, not on pointwise bootstrap samples), so it would *not* suffer from the noisy-variance problem that sank the variance-model band. This idea remains viable and may be the most promising path to improved alpha tunability, since it pairs a smooth variance with bootstrap-calibrated simultaneity.

**Smooth variance blending across regions.** Instead of a hard cutoff between Wilson-corrected and uncorrected regions, define a weight function w(k) that transitions smoothly from 1 (Wilson dominates) to 0 (bootstrap dominates) as the effective count k increases past k_min. The blended variance sigma^2_blend(t) = w(k) * sigma^2_Wilson(t) + (1-w(k)) * sigma^2_boot(t) would eliminate the hard boundary between regions and might close the near-boundary gap. The current variance-ratio gate is a first move in this direction: it is data-adaptive and threshold-free, but it is still a gate rather than a smooth blend and needs full empirical validation.

**Envelope with HT as the base instead of the bootstrap.** Rather than enveloping bootstrap curves (which inherit the boundary problem), envelope curves generated from the HT variance model. At each bootstrap replicate, compute R_b(t) = R_hat(t) + sigma_HT(t) * z_b(t) where z_b(t) is drawn from the bootstrap distribution of the studentized process. This would combine the envelope's adaptation properties with HT's variance model, potentially getting the best of both worlds. The risk is that HT's density estimation errors would propagate into the envelope.

### F.3 The hybrid insight

The most important lesson from this project is that **no single uncertainty quantification strategy works everywhere on the ROC curve**. The bootstrap works in the interior but fails at the boundary. Wilson works at the boundary but is too simple for the interior. HT captures both variance components but requires density estimation that fails in the tails and under model misspecification.

The envelope_wilson method succeeds because it is a hybrid: bootstrap in the interior, Wilson at the boundary. The next generation of improvement should extend this hybrid architecture to three regions (Wilson at the boundary, HT in the near-boundary zone, bootstrap in the interior).

The G.1 experiment (variance-model band) tested whether the envelope operator could be replaced with a pointwise variance-model approach using bootstrap-calibrated simultaneity. The answer is no -- not with pointwise bootstrap variance, which is too noisy for supremum-based calibration. The envelope operator has an underappreciated robustness property: it tolerates noisy variance estimates because it operates on retained curves directly rather than studentizing against a variance function. Replacing it requires a *smooth* variance estimate (like HT's analytical variance), not just a different band construction.

The "ugly band-aids" feeling is real but misleading. The Wilson floor is not a band-aid -- it is the correct response to a structural limitation of the nonparametric bootstrap. The method is a hybrid *by necessity*, not by accident. The question is not whether to hybrid, but how to do it more gracefully.

---

## Summary Table

| Property | envelope_wilson | KS | HT-autocalib | Wilson Rect (Sidak) | WH |
|---|---|---|---|---|---|
| Coverage at 95%, n=300 | 0.953 | 1.000 | 0.746 | 0.941 | ~0.80* |
| Coverage at 50% | 0.851 | 0.982 | 0.611 | 0.247 | -- |
| Mean band area | 0.397 | 0.469 | 0.536 | 0.331 | -- |
| DGP robustness | Excellent | Perfect | Moderate | Good | Poor |
| Large-n (10k) coverage | 0.830 | 1.000 | 0.926 | 0.839 | -- |
| Max violation (P99) | 0.037 | 0.000 | 0.067 | 0.267 | -- |
| Tunability (50% CI) | Poor | Poor | Good | Fair | -- |
| Distributional assumptions | None | None | Log-concave | None | Binormal |

*WH coverage varies wildly by DGP, from <20% to >95%.

---

## G. Future Directions

### G.1 Variance-model band with bootstrap calibration — implemented and tested

**Idea:** Replace the envelope operator with a variance-model band, keeping
the bootstrap for both variance estimation and critical value calibration.

**Construction:**
1. Compute bootstrap variance sigma²_boot(t) at each grid point (nonparametric,
   captures both binomial and threshold-uncertainty variance components)
2. Apply the Wilson variance-ratio floor where bootstrap has collapsed
3. Generate bootstrap replicates of the studentized supremum statistic, as in
   HT-autocalib: Z_b = sup_t |(R_b(t) - R_hat(t)) / sigma_hat(t)|
4. Find c_alpha = (1-alpha)-quantile of {Z_b}
5. Construct band: R_hat(t) ± c_alpha * sigma_hat(t)

Implementation: `variance_model_band.py`, with both probability-space and
logit-space variants, Wilson floor toggle, and the same signature conventions
as `envelope_bootstrap_band`.

**Result: negative.** The variance-model band is consistently more
over-conservative than `envelope_wilson`, with wider bands and calibration
further from nominal at both 95% and 50% levels. Selected results from 500
simulations per scenario (gap = coverage - nominal; closer to 0 is better):

| Scenario            | envelope_wilson gap@95 | varmodel gap@95 | envelope_wilson gap@50 | varmodel gap@50 |
|---------------------|------------------------|-----------------|------------------------|-----------------|
| Gaussian, n=10k     | +0.022                 | +0.028          | +0.278                 | +0.238          |
| High AUC, n=2k      | +0.032                 | +0.048          | +0.446                 | +0.378          |
| Student-t(3), n=600 | +0.012                 | +0.036          | +0.106                 | +0.186          |
| Gaussian, n=300     | -0.060                 | +0.030          | +0.182                 | +0.218          |

The varmodel is further from nominal in 6 of 8 cells. The one scenario where
the envelope is anti-conservative (n=300, -0.060 gap at 95%), the varmodel
overcorrects to +0.030 rather than landing near zero. Its bands are 20-50%
wider (higher mean area) with no compensating calibration benefit.

The logit-space variant (`use_logit=True`) is far worse: massively
over-conservative, with band areas 3-20x larger than the probability-space
version. The Haldane-Anscombe correction inflates logit-space variance at
boundary points so aggressively that the entire construction is dominated by
the floor.

**Root cause: noisy variance + supremum interaction.** The bootstrap variance
sigma²_boot(t) is estimated pointwise from B replicates and is therefore noisy
across the FPR grid. The studentized supremum Z_b = sup_t |dev_b(t)/sigma(t)|
is dominated by whichever grid point has the worst ratio. Points with
accidentally low variance estimates produce spuriously large studentized
deviations, and the supremum selects these. Over many bootstrap replicates,
this max-over-noise effect systematically inflates c_alpha.

This explains why HT-autocalib succeeds where the variance-model band fails.
HT uses a smooth, analytical variance function -- the same sigma(t) for every
bootstrap replicate, with no pointwise sampling noise. The supremum of the
studentized process has the correct distribution because the denominator is
stable. The success of bootstrap calibration depends not on the calibration
mechanism itself (both methods use sup-norm quantiles) but on the *smoothness*
of the variance estimate being studentized against.

**Violation geography shifts.** Spatial analysis of violations reveals that
the two methods fail in different regions:

- `envelope_wilson` violations concentrate in the **low-FPR tail**
  (FPR < 0.05), where the Wilson floor is *not* active -- the near-boundary
  gap identified in Section B.8.
- `varmodel_wilson` violations concentrate in the **mid-to-high FPR
  interior** (FPR 0.30-0.90), in the transition zone where the Wilson floor
  fires intermittently (active in 15-50% of simulations at a given point).
  The intermittent flooring makes the studentized statistic distribution
  inconsistent across simulations, further degrading calibration.

**Implication for future directions.** The negative result is informative. It
rules out the simplest version of the "replace envelope with variance model"
idea and identifies the binding constraint: the variance estimate must be
smooth for supremum-based calibration to work. This points toward either (a)
smoothing the bootstrap variance before studentizing (e.g., local polynomial
smoothing, or using HT variance as a smooth scaffold with bootstrap as a
correction), or (b) abandoning pointwise bootstrap variance entirely in favor
of a smooth parametric or semiparametric variance model, accepting the
distributional assumptions that entails.

The envelope operator, despite its weak alpha sensitivity, has an advantage
that was not previously appreciated: it is *robust to noisy variance
estimates* because it operates on retained curves directly (min/max) rather
than studentizing against a variance function. The noise in pointwise
bootstrap variance does not propagate into the envelope width in the same way
-- it affects which curves are retained (via the KS statistic during
retention), but the envelope itself is determined by the curves, not by
variance.

### G.1a Bootstrap calibration threshold for HT-autocalib

The current `n_bootstraps="auto"` threshold (`n > 300`) is a magic number.
A principled alternative emerges from treating calibration as a bias-variance
tradeoff.

The analytical critical value (Bonferroni-style z_bonf) is deterministic but
biased (conservative, ignores correlation). The bootstrap c_alpha is
asymptotically unbiased but has sampling variance from B replicates. Calibration
helps when the bias of the analytical value exceeds the sampling noise of the
bootstrap estimate.

Measuring |bias|/SE(c_alpha) across 50 independent datasets per sample size
(Gaussian DGP, B=4000, K=201):

| n per class | mean c_alpha | SE(c_alpha) | z_bonf | |bias|/SE |
|-------------|-------------|-------------|--------|----------|
| 30          | 3.86        | 0.99        | 2.92   | 1.0      |
| 50          | 4.38        | 1.15        | 2.92   | 1.3      |
| 100         | 4.62        | 1.19        | 2.92   | 1.4      |
| 150         | 4.14        | 0.55        | 2.92   | 2.2      |
| 300         | 3.94        | 0.41        | 2.92   | 2.5      |
| 1000        | 3.66        | 0.23        | 2.92   | 3.2      |
| 5000        | 3.41        | 0.10        | 2.92   | 4.7      |

At n=30, |bias|/SE ~ 1: calibration adds as much noise as it corrects, so it
is approximately a wash. By n=150, |bias|/SE ~ 2: calibration reliably
improves on Bonferroni. By n=1000+, the signal clearly dominates.

Note: c_alpha is consistently *above* z_bonf (the bias is negative), meaning
the analytical value is anti-conservative for the HT studentized process. This
is expected: HT variance has estimation error, so the studentized process is
heavier-tailed than Gaussian, and the bootstrap correctly captures this while
Bonferroni assumes Gaussian.

The non-monotonicity at n=50-100 (higher and more variable c_alpha than at
n=150+) likely reflects log-concave density estimation instability at small n:
the HT variance estimate itself becomes unreliable, inflating the studentized
supremum.

**A principled threshold** could be implemented as a split-half stability
check: divide the B bootstrap replicates into two halves, compute c_alpha on
each, and compare the half-sample difference to the gap between c_alpha and
z_bonf. If |c_alpha - z_bonf| >> |c_half1 - c_half2|, calibration is adding
signal; otherwise, fall back to the analytical value. This is data-adaptive
and avoids a fixed sample-size cutoff. It has not yet been implemented or
tested.

### G.2 Multiplier bootstrap (Priority 2)

**Idea:** Replace the standard resampling bootstrap with the multiplier (wild)
bootstrap to reduce the finite-sample calibration bias caused by step-function
discreteness of bootstrap ROC curves.

**Mechanism:** The standard bootstrap resamples n_0 negatives with replacement,
producing ROC curves that are step functions with jumps at the same set of
observed score values. The discrete multiplicities (0, 1, 2, ...) create
artificial variability in the supremum statistic that isn't present in the
smooth population process. This inflates the bootstrap critical value,
producing over-conservative bands (the "bootstrap conservatism" effect
identified in Section B.4a).

The multiplier bootstrap instead perturbs the empirical process directly:
for each replicate b, generate weights w_i ~ N(0,1) (or Rademacher: ±1 with
equal probability) and compute the weighted empirical CDFs:

    F_b(x) = (1/n_0) * sum_{i: y_i=0} w_i * I(s_i <= x)
    G_b(x) = (1/n_1) * sum_{j: y_j=1} w_j * I(s_j <= x)

The resulting process is still a step process on the observed thresholds, but the perturbations are continuous rather than multinomial counts. That can reduce resampling discreteness in the empirical-process approximation. It does not solve the unobserved-tail support problem by itself, so it should be evaluated as an interior calibration improvement, not as a replacement for boundary correction.

**Theoretical backing:** Kosorok (2008), *Introduction to Empirical Processes
and Semiparametric Inference*, Chapter 2.9, provides conditions under which the
multiplier bootstrap is consistent for the supremum functional of empirical
processes. The rate of convergence can be faster than the standard bootstrap
for processes with limited smoothness.

**Expected payoff:** Better interior calibration, especially at the 50% level where bootstrap discreteness is most visible. This is mostly orthogonal to the band construction (envelope vs variance-model) and the boundary correction (Wilson floor), but it should not be expected to fix high-AUC low-FPR misses without an additional tail model.

**Implementation effort:** Moderate. Requires a new bootstrap grid generator
that operates on weighted empirical CDFs rather than resampled scores. The
downstream code (studentization, retention, envelope/band construction) is
unchanged.

### G.3 Functional depth ranking (Priority 3)

**Idea:** Replace the KS statistic (supremum of |z_b(t)|) as the curve ranking
criterion with **band depth** from functional data analysis.

**Background:** The KS statistic measures "how extreme is this curve's worst
point?" It is dominated by a single grid point. A curve that deviates
moderately at every grid point (globally atypical) gets a better KS rank than
a curve that deviates strongly at one point but is typical everywhere else
(locally extreme). For the envelope, this means KS retention keeps some
globally-atypical curves while discarding locally-extreme ones.

Band depth (López-Pintado & Romo 2009) measures global typicality: for a curve
R_b, its band depth is the proportion of curve pairs (R_i, R_j) such that R_b
lies entirely within the band [min(R_i, R_j), max(R_i, R_j)]. A curve with
high band depth is "enclosed" by many pairs -- it is consistently central.
A curve with low band depth lies outside many pairs -- it is globally unusual.

Sun & Genton (2011), "Functional Boxplots," use band depth to construct central
regions for functional data. Their "50% central region" is the envelope of the
deepest 50% of curves -- exactly the construction used in the envelope
bootstrap, but with band depth instead of KS.

**Expected payoff:** Potentially tighter bands. Band depth retention discards
curves that are globally atypical (contributing width everywhere) rather than
curves that are locally extreme (contributing width at one point). The envelope
of band-depth-retained curves should be tighter because the retained set is
more compact in function space.

**Risk:** Full band depth is expensive. Checking whether one curve lies inside every pairwise band across K grid points is O(B^2 K), not just O(B^2). For B=4000 and K=1000, that is on the order of 16 billion grid comparisons versus 4 million for KS ranking. Modified band depth or random pair subsampling is probably required for the existing B and K.

**Implementation effort:** Moderate. Requires implementing band depth
computation (or using an existing implementation) and plugging it into the
retention step. The envelope construction is unchanged.

### G.4 Split-sample calibration against held-out empirical ROC curves

**Idea:** Use split-sample calibration to construct bands that cover a held-out empirical ROC curve. This is closer to conformal prediction than to classical confidence-band inference, and the distinction matters: exchangeability can calibrate coverage for a future empirical curve generated like the calibration curve, but it does not automatically give finite-sample coverage for the fixed population ROC `R_true`.

**Construction:**
1. Split data into a "fit" set and a "calibration" set
2. Compute the empirical ROC R_hat from the fit set
3. Compute the empirical ROC R_cal from the calibration set
4. Compute the studentized deviation: Z_cal = sup_t |(R_cal(t) - R_hat(t)) / sigma_hat(t)|
5. Set the band width such that it would cover R_cal

With many splits or cross-conformal aggregation, this could calibrate the distribution of empirical-ROC discrepancies without relying on nonparametric bootstrap consistency. The target would need to be stated carefully: prediction coverage for a future empirical ROC is not the same as confidence coverage for the population ROC.

**Advantage:** It is a useful stress test of the bootstrap calibration story and may expose whether the current bands are too narrow for future-sample variability. It also avoids some bootstrap discreteness artifacts because calibration uses genuinely held-out samples.

**Disadvantage:** Splitting the data reduces effective sample size. With n=100,
a 50/50 split gives n_fit=50 and n_cal=50, each with wider ROC uncertainty
than the full sample. Cross-conformal (Vovk 2015) or jackknife+ (Barber et al.
2021) can mitigate this. The larger disadvantage is conceptual: additional work is needed to translate held-out empirical-curve calibration into population-ROC confidence coverage.

**Reference:** Lei & Wasserman (2014), "Distribution-Free Prediction Bands for
Non-parametric Regression," provides the closest existing framework.

**Implementation effort:** Moderate-high. Requires new infrastructure for data
splitting and calibration, but the ROC computation and evaluation code can be
reused.

### G.5 Near-boundary improvement via finite-difference slopes

**Idea:** Address the near-boundary zone (where the bootstrap has some variance
but underestimates the threshold-uncertainty component) without density
estimation.

The Hsieh-Turnbull variance has two components:

    Var(R(t)) = R(t)(1-R(t))/n_1 + [g(c)/f(c)]² * t(1-t)/n_0

Wilson captures only the first component. The second requires the density
ratio g(c)/f(c), which equals the ROC slope R'(t). Instead of estimating
densities, estimate the slope directly from finite differences of the
empirical ROC:

    slope(t) ≈ (R_hat(t + delta) - R_hat(t - delta)) / (2 * delta)

Then construct an approximate HT variance:

    var_approx(t) = R(t)(1-R(t))/n_1 + slope(t)² * t(1-t)/n_0

This is crude but captures the right scaling: the threshold-uncertainty
component is large where the ROC is steep (high AUC, low FPR) and small
where it's flat. The resulting variance could serve as a second floor --
above Wilson, below the bootstrap in the interior -- specifically targeting
the near-boundary zone where the hard-cutoff approach has a gap.

**Advantage:** No density estimation, no log-concavity assumption, no
hyperparameters. Just finite differences of the empirical ROC.

**Risk:** Finite differences of a step function are noisy. Smoothing (e.g.,
Harrell-Davis estimation of the ROC, or local polynomial smoothing) would be
needed to get stable slope estimates, reintroducing a bandwidth parameter.
Alternatively, the Harrell-Davis ROC estimator already implemented in the
codebase provides smooth ROC curves whose derivatives are analytically
available.

### G.6 Priority and independence

These directions are largely independent and can be explored in parallel:

| Direction | Addresses | Effort | Status | Independent of |
|---|---|---|---|---|
| G.1 Variance-model band | 50% CI, tunability | Low | **Done — negative result** | G.2, G.3 |
| G.1a Calibration threshold | HT-autocalib robustness | Low | Characterized, not implemented | G.2, G.3 |
| Current variance-ratio Wilson floor | Hard cutoff artifact | Low-moderate | Implemented, not fully reevaluated | G.2, G.3, G.5 |
| G.2 Multiplier bootstrap | Calibration bias | Moderate | Open | G.1, G.3, G.5 |
| G.3 Band depth ranking | Band tightness | Moderate | Open | G.1, G.2 |
| G.4 Split-sample calibration | Future empirical ROC calibration | Moderate-high | Open | All others |
| G.5 Finite-difference slopes | Near-boundary zone | Low | Open | G.1, G.2, G.3 |

G.1's negative result eliminates the simplest path to improved alpha
tunability and narrows the remaining options. The key lesson -- that bootstrap
calibration requires a smooth variance estimate -- constrains G.2 and G.5:
any variance improvement must produce a smooth function of t, not a noisy
pointwise estimate.

The immediate priority is to rerun a targeted subset with the current variance-ratio Wilson floor against the hard-cutoff implementation: n=300, 1000, and 10000; AUC bins above and below 0.9; and the high-risk Student-t/logitnormal/hetero-Gaussian cases. That directly tests whether the newer gate fixes the near-boundary wall or only moves it.

After that, G.5 (finite-difference slopes) and G.2 (multiplier bootstrap) are the highest-priority open directions. G.5 could provide the smooth variance estimate that G.1 lacked, and G.2 addresses interior bootstrap discreteness orthogonally. G.1a is a small, self-contained improvement to the existing HT-autocalib method.

### Key references

- Kosorok (2008), *Introduction to Empirical Processes and Semiparametric
  Inference* -- multiplier bootstrap theory for empirical processes
- Sun & Genton (2011), "Functional Boxplots" -- band depth applied to
  functional central regions, closest existing work to the envelope bootstrap
- López-Pintado & Romo (2009), "On the Concept of Depth for Functional Data"
  -- band depth theory and computation
- Hall & Horowitz (2013), "A Simple Bootstrap Method for Constructing
  Nonparametric Confidence Bands for Functions" -- conditions for bootstrap
  band coverage, convergence rates
- Lei & Wasserman (2014), "Distribution-Free Prediction Bands for
  Non-parametric Regression" -- conformal approach to function-valued bands
- Barber et al. (2021), "Predictive Inference with the Jackknife+" --
  improved conformal prediction without data splitting

---

*Report generated 2026-04-28, updated 2026-05-01. Based on code review of 5 method implementations, simulation specification, 2,254,000 evaluations across 7 DGPs, 6 sample sizes, and 23 method variants, plus targeted simulations of the variance-model band (G.1) across 4 scenarios at 500 simulations each.*
