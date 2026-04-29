# Project Evaluation: Simultaneous Confidence Bands for ROC Curves

*Assessment of the studentized bootstrap envelope method and its competitors, based on code review, theoretical analysis, and 2.25M simulation evaluations.*

---

## A. Motivation: The Gap Between Working-Hotelling and KS

### The practical problem

A practitioner computes an ROC curve from a finite sample and wants to know: where could the *true* ROC curve plausibly lie? This requires a **simultaneous** confidence band -- one that covers the entire curve at once, not just individual points. Two classical answers exist:

**Working-Hotelling (WH):** Assumes scores in both classes follow normal distributions (the binormal model), estimates two parameters (a, b) relating the probit-transformed ROC to a linear function, and constructs a chi-squared-based band in probit space. When the binormal assumption holds, this is efficient. When it fails -- heavy tails, multimodality, skew -- coverage collapses catastrophically (simulation evidence: <20% coverage for Student-t(3) at n=1000).

**Kolmogorov-Smirnov (KS):** Uses the DKW inequality to construct a fixed-width band around the empirical CDF of each class, then propagates to the ROC. Makes no distributional assumptions. Achieves 100% coverage always, everywhere, for all sample sizes. The price: bands are so wide that the 50% confidence band still achieves 98% coverage -- the method communicates almost no information about *where* within the band the true ROC lies.

### The gap

The gap between WH and KS is the gap between *assuming everything* and *assuming nothing*. WH assumes a parametric model and gets tight bands when right, garbage when wrong. KS assumes nothing and gets valid-but-useless bands always.

The practical need is a method that:
1. Makes no distributional assumptions (or very weak ones)
2. Achieves approximately nominal coverage (~95%) across a wide range of DGPs
3. Produces bands tight enough to be informative
4. Degrades gracefully when it does fail -- small violations, not catastrophic misses

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

### B.4 The envelope operator equals the theoretical band

For the KS-type construction, the envelope of the retained set converges to the theoretical confidence band. This is worth spelling out because it is not obvious.

Any retained curve R_b satisfies Z_b <= c, meaning |z_b(t)| <= c at every grid point, meaning R_b(t) is in [R_hat(t) - c*sigma(t), R_hat(t) + c*sigma(t)]. So the envelope can be at most as wide as the theoretical band.

Does the envelope *reach* the theoretical band? At any grid point t, the question is: among the m retained curves, does any curve achieve z_b(t) near +c? Among thousands of curves, some will have their maximum studentized deviation concentrated at or near point t, achieving z_b(t) close to c while staying below c elsewhere. For large m, the envelope converges to the theoretical band at every grid point.

Different curves contribute the envelope's extremes at different grid points -- curve A may set the upper envelope at FPR=0.3, curve B at FPR=0.7. This is not a distortion; it is the *mechanism* by which the finite-sample envelope correctly converges to the theoretical band. The theoretical band allows R_true to deviate by up to c*sigma(t) at every point, and the envelope of retained curves recovers this same width.

**Consequence:** There is no gap between "retained set covers R_true" and "envelope of retained set covers R_true" for the KS-type construction. The coverage guarantee transfers directly from the bootstrap KS test to the envelope band.

### B.4a Finite-sample conservatism and the 50% CI problem

The asymptotic story is clean: the envelope equals the theoretical band, and coverage is 1-alpha. But the simulation shows massive over-coverage at the 50% CI level (85% actual vs 50% nominal), decreasing with n but persisting even at n=10,000 (64%). This over-coverage comes from two finite-sample effects, not from the envelope operator itself.

**Effect 1: The Wilson floor.** At small n, the Wilson floor widens the band far beyond what the bootstrap alone would produce, dominating overall coverage. At n=10, the floor covers the entire grid. This is the primary source of over-coverage for n <= 100.

**Effect 2: Bootstrap conservatism for the supremum functional.** Bootstrap ROC curves are step functions with jumps at multiples of 1/n_0 (FPR) and 1/n_1 (TPR). The true ROC is smooth. Step functions are intrinsically more variable than smooth curves -- discrete jumps create fluctuation in the supremum statistic that isn't present in the smooth population process. This means the bootstrap Z_b distribution is stochastically *larger* than the true Z distribution, biasing the critical value c_alpha upward and making the band wider than necessary. This is a well-known issue with the nonparametric bootstrap for step-function processes: bootstrap critical values for sup-norm statistics are biased upward in finite samples.

The 50% CI is more sensitive to this bias than the 95% CI because the theoretical 50% band is narrow -- small overestimation of c produces proportionally large excess width. The bias diminishes with n as the step-function discreteness becomes finer.

**The sup-norm's weak sensitivity to alpha.** An additional factor is that c_{0.95}/c_{0.50} for a supremum statistic over many correlated grid points is modest -- typically 1.3-1.5x depending on K and the correlation structure. So the 95% band is only 30-50% wider than the 50% band. Combined with the upward bias in c, this means coverage at 50% is substantially above nominal even when coverage at 95% is near-exact. This is a property of all sup-norm-based simultaneous bands, not specific to the envelope construction.

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

### B.8 The near-boundary gap

Between "tail" (Wilson active) and "interior" (bootstrap reliable) there exists a transition zone: grid points with effective counts just above the threshold (k = 15-50). Here:
- Bootstrap variance is non-zero but unreliable (small-count effects)
- Wilson floor is not applied (k >= k_min)
- True ROC has genuine uncertainty that bootstrap underestimates

As n grows, more grid points enter this gap (the tail shrinks, the gap persists), accumulating opportunities for small violations. This is the proximate cause of large-n coverage degradation.

### B.9 Expected failure modes

| Failure mode | Severity | When it occurs | Mechanism |
|---|---|---|---|
| Over-conservative at 50% CI | Moderate | Always, diminishing with n | Wilson floor (small n) + bootstrap step-function conservatism + weak sup-norm sensitivity to alpha |
| Over-conservative at small n | Mild | n <= 30 | Wilson floor dominates |
| Under-coverage at large n | Notable | n >= 10000 | Wilson floor vanishes, bootstrap boundary problem exposed |
| Under-coverage at high AUC | Notable | AUC > 0.9, n >= 1000 | Steep ROC concentrates information in boundary region |
| Large violation magnitude | Rare | Never observed > 5pp at P95 | Violations are always small in absolute terms |

The method's failure modes are *benign*: when coverage is lost, violations are small (~0.2pp of TPR) and concentrated in a narrow region of the ROC. This is qualitatively different from parametric methods, which can miss the true ROC by tens of percentage points.

---

## C. Overview of Alternative Methods

### C.1 Kolmogorov-Smirnov (KS) Band

**Approach:** Fixed-width band based on the DKW inequality. The band width d = c_alpha / sqrt(n_eff) is uniform across the entire ROC curve. This is a distribution-free simultaneous confidence band with guaranteed finite-sample coverage.

**Strengths:**
- Perfect coverage at all sample sizes, all DGPs, always
- No tuning parameters beyond alpha
- No distributional assumptions
- Simple to implement and understand

**Weaknesses:**
- Fixed width does not adapt to local variance
- Bands are uninformatively wide (50% band achieves 98% coverage)
- Width scales as O(1/sqrt(n)), same as any nonparametric method, but with a large constant

**Role in the landscape:** The KS band is the gold standard for safety. Any practical method should be judged by how much tighter it is than KS while maintaining acceptable coverage. It is the "free lunch" baseline: guaranteed correctness, zero information content.

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

**Key finding:** Violation magnitudes remain small even when coverage is lost. Mean max violation at n=10,000 is ~0.002 (0.2pp of TPR). The P99 max violation is ~0.046. Only 0.84% of simulations have any violation exceeding 5pp.

**Implication:** The coverage drop at large n is a *technical* failure (the true ROC escapes the band) but not a *practical* failure (it escapes by amounts far below any decision threshold). Whether this matters depends on the use case:
- For regulatory submissions where a stated 95% guarantee must hold formally: this is a problem.
- For exploratory analysis where the question is "roughly where is the true ROC?": this is fine.

### D.4 DGP robustness is the standout property

Coverage varies by at most 5pp across all 7 DGPs at any fixed sample size. This is remarkable: beta (opposing skew), bimodal negative (mixture), Student-t (heavy tails), Weibull, Gamma, logit-normal, and heteroscedastic Gaussian all produce essentially the same coverage.

Compare Working-Hotelling: coverage ranges from <20% (Student-t with low df) to >95% (heteroscedastic Gaussian). The parametric methods live or die by their assumptions; the bootstrap doesn't care.

**Implication:** This is the strongest argument for the bootstrap approach. In practice, the data scientist doesn't know the DGP. A method that works equally well for all DGPs is worth more than a method that works brilliantly for some and catastrophically for others.

### D.5 The Wilson floor ablation

The simulation report includes `envelope_standard` (no Wilson floor), which achieves ~50% coverage. This confirms the theoretical prediction: the bare bootstrap envelope is not a valid confidence band, at any sample size, because boundary variance collapse causes systematic under-coverage.

The Wilson floor is not a "band-aid" -- it is a *necessary correction* for a structural limitation of the nonparametric bootstrap. Without it, the method doesn't work. With it, the method works well for n <= 1000 and degrades gracefully beyond.

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

4. **Graceful degradation is real.** Even in the worst case (n=10000, high AUC), violation magnitudes are tiny. This is not a property that was designed in -- it emerges from the method's nonparametric structure. Parametric methods can miss the true ROC by 30pp; this method never misses by more than a few pp. This qualitative difference is undersold in the current reports.

### E.2 Walls

1. **Sup-norm-based simultaneous bands have weak sensitivity to alpha.** The critical value ratio c_{0.95}/c_{0.50} for a supremum statistic over many correlated grid points is inherently modest (~1.3-1.5x). This means the 50% band is only modestly narrower than the 95% band, regardless of the construction method (envelope, studentized band, etc.). Combined with finite-sample bootstrap conservatism, this makes the 50% band substantially over-conservative in practice. The over-coverage diminishes with n but remains noticeable for the sample sizes where the method is most useful. This is not specific to the envelope operator -- it affects any method that determines simultaneous coverage via a sup-norm critical value.

2. **The large-n coverage gap has no clean fix within the current framework.** The near-boundary zone (k = 15-50) is where the bootstrap has *some* variance but not enough. Options:
   - **Raise k_min:** Shifts the wall but makes the Wilson floor cover more of the curve, reducing the bootstrap's contribution. At k_min = 50, the Wilson floor covers ~10% of the grid at n=500 -- this starts to defeat the purpose.
   - **Use an adaptive threshold:** Make k_min scale with n (e.g., k_min = c*sqrt(n_0)). This would keep the tail fraction constant across sample sizes. But the choice of c is arbitrary and the theory doesn't tell you the right value.
   - **Smooth the transition:** Instead of a hard cutoff between "Wilson active" and "Wilson off," blend the Wilson floor with the bootstrap variance using a weight that decreases with k. This is more principled but adds complexity and tuning parameters.

3. **The bootstrap can't help at the boundary.** This is the deepest wall. The nonparametric bootstrap resamples from the empirical distribution and therefore cannot represent uncertainty about probability mass beyond the observed data. At FPR=0, this means the bootstrap has exactly zero information about R_true(0). No clever resampling scheme fixes this within the nonparametric framework.

### E.3 Remaining uncertainties

1. **Is the large-n coverage gap practically important?** For most ROC analyses in biomedical and ML applications, n < 1000. The method works well in this range. Whether n=10000 coverage of 83% matters depends on the field and the stakes. This is an empirical question about use cases, not a technical question.

2. **Can the near-boundary zone be better served?** The three-region model suggests a principled intervention: use the Hsieh-Turnbull variance (which captures both variance components) as a floor in the near-boundary zone, falling back to Wilson only in the true tails where density estimation is impossible. This would require density estimation at near-boundary thresholds, which is feasible (these points have k=15-50 negatives above threshold, enough for a rough density estimate). This idea has not been tested.

3. **What is the right comparison for "stapling on fixes"?** The concern that other methods would be equally good with similar engineering effort is legitimate. The HT method with bootstrap calibration and Wilson floor already achieves 0.895 coverage with better 50% CI calibration. If the density estimation step were made more robust (e.g., using a more forgiving nonparametric estimator, or using the bootstrap to select among estimators), HT might close the gap to the envelope at 95% while maintaining its advantage at 50%.

4. **Is symmetric retention worth the complexity?** `envelope_wilson_symmetric` achieves nearly identical performance to `envelope_wilson` (0.946 vs 0.950 coverage). The theoretical motivation (addressing asymmetric alpha mass at high AUC) is sound, but the empirical improvement is negligible. This suggests the standard KS retention already handles the asymmetry adequately, and the symmetric correction is redundant in practice.

5. **The interaction between Harrell-Davis smoothing and the envelope.** HD smoothing reduces the discreteness of individual bootstrap ROC curves, potentially allowing finer-grained retention. The simulation includes HD variants but the report doesn't analyze them in detail. If HD smoothing doesn't help, it suggests the discreteness of the bootstrap ROC is not the binding constraint.

---

## F. Best Ideas Across All Methods

### F.1 Ideas that work well

**Wilson's always-positive width (from Wilson Rectangle).** The Wilson score interval guarantees non-zero width at p=0 and p=1. This is the key property that makes the boundary correction work. Any future method should use Wilson (not Wald) intervals whenever binomial proportions appear.

**Bootstrap calibration of the critical value (from HT-autocalib).** Instead of relying on asymptotic theory or Bonferroni correction to determine the simultaneous critical value, generate bootstrap replicates of the test process and find the empirical quantile. This is strictly better than analytical corrections because it respects the actual correlation structure of the ROC curve. The implementation in `hsieh_turnbull_band.py` is clean: generate bootstrap ROCs, compute the sup-statistic, take the (1-alpha)-quantile. This idea is portable to any variance-based method.

**Adaptive variance from the bootstrap (from envelope_boot).** The bootstrap variance at each FPR grid point captures both the binomial and threshold-uncertainty components of ROC variance, adapts to the actual shape of the score distributions, and makes no parametric assumptions. In the interior of the ROC, this is the best variance estimate available.

**Sidak correction for tail points (from envelope_boot).** When applying separate corrections at K_tail points, using alpha_tail = 1 - (1-alpha)^{1/K_tail} is exact for independent tests and conservative for dependent ones. This is better than Bonferroni (alpha/K) and appropriate for the tail region where tests are approximately independent (they depend on disjoint sets of observations).

**Restricting parametric corrections to where they're needed (from envelope_boot).** The three-region model -- parametric floor in tails, bootstrap in interior, nothing in between -- is the right architecture. The insight that the Wilson floor should *not* be applied everywhere is important: it preserves the bootstrap's advantages in the interior while patching its weakness at the boundary.

### F.2 Ideas that could be developed further

**Hsieh-Turnbull variance as a variance floor in the near-boundary zone.** The current method uses Wilson variance (binomial component only) as a floor. The HT variance captures both components but requires density estimation. In the near-boundary zone (k = 15-50), there are enough observations for rough density estimation. Using HT variance as a floor in this zone -- falling back to Wilson in the true tails -- could close the large-n coverage gap. This is the most promising unexplored direction.

**Bootstrap-calibrated Wilson bands.** The Wilson Rectangle method with Sidak correction achieves 0.911 coverage and is the tightest method with >= 90% coverage. Its main weakness is the lack of true simultaneity control. What if you used bootstrap calibration (as in HT-autocalib) to determine the critical value z, but applied it to Wilson intervals instead of HT intervals? This would combine Wilson's always-positive-width property with bootstrap-calibrated simultaneity. Unlike the envelope approach, this would be smoothly tunable across confidence levels because it uses a single critical value, not an envelope operator.

**Smooth variance blending across regions.** Instead of a hard cutoff between Wilson-corrected and uncorrected regions, define a weight function w(k) that transitions smoothly from 1 (Wilson dominates) to 0 (bootstrap dominates) as the effective count k increases past k_min. The blended variance sigma^2_blend(t) = w(k) * sigma^2_Wilson(t) + (1-w(k)) * sigma^2_boot(t) would eliminate the hard boundary between regions and might close the near-boundary gap. The weight function could be data-driven (e.g., based on the ratio of Wilson to bootstrap variance) or parametric (e.g., logistic in k with inflection at k_min).

**Envelope with HT as the base instead of the bootstrap.** Rather than enveloping bootstrap curves (which inherit the boundary problem), envelope curves generated from the HT variance model. At each bootstrap replicate, compute R_b(t) = R_hat(t) + sigma_HT(t) * z_b(t) where z_b(t) is drawn from the bootstrap distribution of the studentized process. This would combine the envelope's adaptation properties with HT's variance model, potentially getting the best of both worlds. The risk is that HT's density estimation errors would propagate into the envelope.

### F.3 The hybrid insight

The most important lesson from this project is that **no single uncertainty quantification strategy works everywhere on the ROC curve**. The bootstrap works in the interior but fails at the boundary. Wilson works at the boundary but is too simple for the interior. HT captures both variance components but requires density estimation that fails in the tails and under model misspecification.

The envelope_wilson method succeeds because it is a hybrid: bootstrap in the interior, Wilson at the boundary. The next generation of improvement should extend this hybrid architecture to three regions (Wilson at the boundary, HT in the near-boundary zone, bootstrap in the interior) and consider whether the envelope operator is the right way to combine retained curves, or whether a pointwise quantile approach with bootstrap-calibrated simultaneity correction would be better.

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

*Report generated 2026-04-28. Based on code review of 5 method implementations, simulation specification, and 2,254,000 evaluations across 7 DGPs, 6 sample sizes, and 23 method variants.*
