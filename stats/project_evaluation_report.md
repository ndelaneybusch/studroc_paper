# Project Evaluation: Simultaneous Confidence Bands for ROC Curves

*Assessment of the studentized bootstrap envelope method and its competitors, based on code review, theoretical analysis, 2.25M simulation evaluations, and targeted follow-up experiments (paired implementation comparison, variance decomposition, and failure-point microscopy; June 2026).*

---

**Implementation provenance note:** The large simulation suite summarized here was run against the earlier `envelope_wilson` implementation that used hard effective-count cutoffs for Wilson tail support (`tail_k_min=(15, 10)`, `tail_m_min=10`). The current code has since moved to a variance-ratio Wilson floor (`bootstrap_var / wilson_var`) with continuous effective dimensionality for the Sidak correction. **A 1,400-case paired comparison (same datasets, same bootstrap matrices; `scratch_paired_comparison.py`) shows the two implementations are behaviorally equivalent**: coverage agrees within 0.3pp in every n × AUC stratum, only 4 of 1,400 cases have discordant outcomes, violation geography and magnitudes are identical, and band areas agree within ±2%. The stored 2.25M-evaluation results therefore describe the current variance-ratio implementation as well. Section D.8 explains why the gate redesign could not change behavior, and D.9 documents where the method actually fails.

**Update (2026-06-10):** the G.5a exact Beta order-statistic floor is now **integrated into `envelope_boot.py`** as part of `boundary_method="wilson"` (lower band only, per-event alpha = α/(2·25)). All coverage figures in sections B–E describe the band *without* that floor unless explicitly noted; the integrated band's measured behavior — coverage 0.95–0.99 on the previously failing problem-domain strata, zero >5pp misses, and the mechanism attribution of the final band — is reported in G.5a.

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

**Result at the literal corners (and at small n):** all bootstrap curves agree on approximately the same TPR, the envelope has near-zero width, and any deviation of R_true breaks coverage.

**Result at large n (measured, see D.9):** the collapse is *one-sided*, not total. At the first few grid points (k = 1-3 negatives above threshold), the pointwise bootstrap sd is roughly right (0.6-1.0x the true Monte Carlo sd), but the *lower* envelope arm reaches only ~0.8-1.4 sd below the empirical ROC while the upper arm stays at ~2.6-2.8 sd. The bootstrap maximum negative score can never exceed the observed maximum, so resampled curves can deviate far upward (when extreme negatives are dropped) but barely downward. A symmetric variance summary cannot detect this: the variance looks healthy while the lower tail of the deviation distribution is missing.

This is not a bug in the implementation -- it is a structural limitation of the nonparametric bootstrap for extreme quantiles. No modification to the studentization, variance floor, or retention criterion changes this. The bootstrap tail problem is *why* the Wilson floor exists -- though D.9 shows the Wilson floor repairs a different corner than the one that ultimately fails.

### B.6 The Wilson floor as a hybrid correction

The Wilson floor addresses the boundary problem by importing a parametric assumption: at each grid point, TPR is treated as a binomial proportion with n_1 trials. The Wilson score interval for this proportion provides a minimum uncertainty that is always positive, even at p=0 or p=1.

The Wilson floor is applied in two places:

**During studentization (Stage 2):** The bootstrap variance is floored to at least the Wilson variance. This prevents studentized statistics from exploding at zero-variance points, keeping the retention criterion well-behaved.

**After envelope construction (Stage 5):** In tail regions, Sidak-corrected Wilson Rectangle bounds are applied as a floor on the envelope. This directly widens the band at boundary points. The two implementations differ in both *where* and *on which side* the floor acts. The stored-suite (hard-cutoff) code applied it **one-sided per tail**: in the lower tail (k < 15) it raised only the *upper* bound, and in the upper tail it lowered only the *lower* bound -- a mirror-image design that assumes the corner pin (TPR=0 at FPR=0) protects the lower band near the origin. The current code applies the floor to *both* sides, but only where the variance-ratio deficiency `max(0, 1 - bootstrap_var/wilson_var)` is positive, which in practice is only the TPR-plateau region (measured: every gate-fired point sits at FPR > 0.59 in the probed violating cases). Net effect: in both versions the lower bound at the low-FPR tail is the bare envelope arm -- verified bit-identical between implementations at k = 1-25 on violating cases -- which is the deeper reason for the behavioral equivalence in D.8.

**What the Wilson floor captures:** The binomial component of ROC uncertainty: R(t)(1-R(t))/n_1. This is the variance of TPR *given a fixed threshold*. It is always present and does not depend on the score distribution.

**What the Wilson floor misses:** The threshold-uncertainty component: [g(c_t)/f(c_t)]^2 * t(1-t)/n_0, from the Hsieh-Turnbull formula. This is the additional variance from not knowing exactly which score threshold corresponds to FPR=t. In the interior, the bootstrap captures both components. At the boundary, the Wilson floor captures only the first.

**Why restrict to tails:** In the interior, the bootstrap variance is strictly more informative than the Wilson model because it captures both variance components and adapts to the actual shape of the ROC curve. Applying the Wilson floor everywhere would replace good estimates with worse ones. Restricting to tails limits the parametric assumption to where it is needed -- with the caveat (D.9) that "where the binomial model undercuts the bootstrap" and "where coverage actually fails" turn out to be different places at large n.

### B.7 Scaling and the coverage trajectory

In the stored-suite implementation the tail region is defined by fixed count thresholds (k_min=15, m_min=10); the current variance-ratio gate fires on a different set of points (the TPR plateau) but produces the same band on the failing side (B.6, D.8). As n grows, the fraction of the FPR grid in the tail region shrinks:

| n_0  | Approximate tail fraction | Wilson's role |
|------|--------------------------|---------------|
| 15   | 100%                     | Drives everything |
| 50   | ~50%                     | Major contributor |
| 150  | ~17%                     | Tail correction only |
| 500  | ~5%                      | Minor correction |
| 5000 | ~0.5%                    | Negligible |

This explains the observed coverage trajectory at the standard prevalence (50%, balanced classes):
- **n <= 30:** Wilson dominates, coverage ~ 100% (over-conservative)
- **n ~ 100-300:** Wilson covers tails, bootstrap conservatism (step-function bias) provides additional over-coverage; coverage crosses nominal here (0.976 at n=100, 0.953 at n=300)
- **n ~ 1000:** Already below nominal: **0.915 at prevalence 50%**. The previously reported "0.950 at n=1000" pools two prevalence configurations -- 0.985 at prevalence 10% (n_pos=100, wide Wilson floor) and 0.915 at prevalence 50% -- and the pooled average happens to land on nominal. There is no n=1000 sweet spot.
- **n >= 10000:** Wilson floor disengaged at the failure points, one-sided bootstrap tail collapse exposed, coverage drops to 0.830.

Two refinements to this story emerged from the follow-up experiments (D.8, D.9). First, the variance-ratio gate is behaviorally equivalent to the hard cutoffs, so neither gate design is what limits coverage. Second, the gate-active region and the failure region are *disjoint* at large n: the Wilson floor fires where empirical TPR is near 0 or 1 (where binomial variance is the dominant uncertainty), while coverage fails at the first few grid points where TPR is mid-range and the slope term dominates -- there the bootstrap variance exceeds the Wilson variance, so no Wilson-referenced gate can ever fire.

### B.8 The fragile zone is the first handful of grid points (k = 1-10), not k = 15-50

At n=10,000, violations concentrate at **k = 1-3 negatives above threshold**: the median violation FPR in re-evaluated violation cases is ~0.00015, and 87.5% of violations sit below FPR 0.001 (k <= 5 at n_neg = 5000). By k = 10 the lower envelope arm has recovered to ~3.2 sd and by k = 25 the band behaves like the interior.

The failure at these first points is a compound of three measured effects (medians across violating high-AUC cases at n=10,000; `scratch_width_check.py`, `scratch_bias_check.py`):

| k (negs above threshold) | Upward bias of R_hat (sd units) | Lower envelope arm (sd units) | Upper envelope arm (sd units) |
|---|---|---|---|
| 1 | +0.66 | 0.80 | 2.78 |
| 2 | +0.46 | 1.11 | 2.65 |
| 3 | +0.38 | 1.40 | 2.60 |
| 5 | +0.28 | 2.45 | 3.07 |
| 10 | +0.18 | 3.16 | 3.25 |
| 25 | +0.10 | 3.50 | 3.40 |

1. **One-sided support collapse.** Bootstrap thresholds at k=1-3 are extreme order statistics of the resample, bounded by the observed extremes. Curves can jump far upward (when extreme negatives are not redrawn) but barely downward, so the lower envelope arm is ~0.8-1.4 sd instead of the ~3.5 sd it reaches in the interior. Pointwise variance is approximately correct (0.6-1.0x truth) -- the deficit is in the lower tail of the deviation distribution, invisible to any variance summary.
2. **Upward bias of the empirical ROC.** At k=1-3 the empirical ROC sits ~0.4-0.7 sd above the true ROC (extreme-quantile geometry; decaying to ~0.1 sd by FPR=0.005 and ~0 by FPR=0.05). The bootstrap is centered on the empirical curve and cannot see this bias. A short lower arm anchored on an upward-biased center is doubly exposed.
3. **The Wilson floor is inert exactly here.** At these points the empirical TPR is mid-range (0.1-0.5 for high-AUC curves), so the binomial Wilson sd is only 0.1-0.3 of the true sd -- the true variance is dominated by the threshold-uncertainty (slope) term `[R'(t)]^2 t(1-t)/n_0`, which Wilson omits. Meanwhile the bootstrap sd is 0.5-1.3x truth, so `bootstrap_var >> wilson_var` and neither the count gate nor the variance-ratio gate ever fires at the failure points. The Hsieh-Turnbull formula evaluated with the *true* slope matches the Monte Carlo truth to within 1-3% at every probed FPR (`scratch_variance_decomposition.py`) -- the variance theory is exactly right; what is missing at the tail is the slope component, which no binomial-variance yardstick can detect.

The threshold-location framing from the earlier analysis remains correct -- error in the estimated negative-class quantile converts to TPR error through the ROC slope -- but it bites at k=1-10, where it manifests as bias and one-sided support collapse rather than as an understated symmetric variance.

**Why coverage degrades with n.** The fragile set (k=1-10) exists at every n, but at small n it is protected: the first grid point sits at FPR = 1/n_0 where the slope is less extreme, n_pos is small so the Wilson floor is wide relative to local sd, and the floor engages there. As n grows, the first grid points move deeper into the tail (FPR = k/n_0) where, for high-AUC curves, the slope -- and hence the gap between true variance and binomial variance -- grows; the Wilson floor disengages (bootstrap variance exceeds Wilson variance), while the bias and arm-collapse effects are order-statistic geometry at fixed k and do not shrink in sd units. Violations therefore become *more frequent but smaller in absolute TPR* as n grows, exactly the observed pattern (violation rate in FPR 0-10%: 0.8% at n=30 rising to 15.1% at n=10,000, while the median violation magnitude falls from ~5pp to ~0.5pp).

### B.9 Failure modes (measured)

| Failure mode | Severity | When it occurs | Mechanism |
|---|---|---|---|
| Over-conservative at 50% CI | Moderate | Always, diminishing with n | Wilson floor (small n) + bootstrap step-function conservatism + weak sup-norm sensitivity to alpha |
| Over-conservative at small n | Mild | n <= 30 | Wilson floor dominates |
| Under-coverage at moderate-to-large n | Notable | n >= 1000 (0.915 at n=1000 prev 50%; 0.830 at n=10000) | First-k fragile zone: one-sided bootstrap support collapse + upward bias of the empirical ROC; Wilson floor inert there (B.8) |
| Under-coverage at high AUC | Notable, monotone in AUC | Visible from n=100 up; 0.742 for AUC > 0.95 at n=10000 | Steep low-FPR slope amplifies the first-k effects |
| Lower-bound optimism | Dominant failure direction | All n; below:above ratio ~10:1 at 95% | Empirical ROC upward-biased at k=1-3; short lower envelope arm |
| Large misses (>5pp) | Rare overall, *not* rare at high AUC | 4-7% of AUC > 0.95 cases at n >= 300 | Same first-k mechanism with large local sd (steep curve) |

The method's typical failure mode is benign: at n=10,000 the median violation among violators is ~0.5pp of TPR, 66% of violations are under 1pp, and violations concentrate below FPR=0.001. But "benign on average" must be qualified by AUC: in the AUC > 0.95 stratum the rate of >5pp misses is 4.0% at n=300, 5.3% at n=1000, and 7.1% at n=10,000, and the worst stored miss is 0.668 of TPR. The accurate claim: violations are usually small, spatially concentrated, and lower-bound; in the high-AUC regime they are frequent enough and large enough to matter.

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

### D.1 The main result, stated precisely

At the standard reporting level (95% CI), `envelope_wilson` achieves 0.950 coverage *pooled over the whole suite*. That pooled figure must be decomposed, because it averages over-coverage at small n against under-coverage at large n. At prevalence 50%, coverage by n is: 1.000 (n=10), 0.991 (n=30), 0.976 (n=100), 0.953 (n=300), **0.915 (n=1000)**, 0.830 (n=10000). Coverage crosses nominal between n=300 and n=1000, not at n=1000: the previously reported "0.950 at n=1000" pools the prevalence-10% configuration (0.985 -- only 100 positives, so a wide Wilson floor) with the prevalence-50% configuration (0.915).

The honest headline: `envelope_wilson` is near-exact for n in roughly 100-500 per class, mildly conservative below that, and degrades steadily above it. This is still the best calibration profile of any nonparametric method tested. The KS band achieves 100% but is uninformative. The Wilson Rectangle with Sidak correction achieves 91% pooled with much worse tail risk (P99 max violation 0.267 vs 0.037). The Hsieh-Turnbull methods achieve 89-90% overall but with highly variable coverage across sample sizes and DGPs.

A useful side-finding: coverage improves as prevalence moves away from 50% because the Wilson floor scales with 1/n_pos -- fewer positives mean a wider floor and a more conservative band. Class imbalance is not a risk factor for this method; balanced classes at large n are.

(These figures describe the band before the integrated Beta order-statistic floor; on the problem-domain strata the floored band reaches 0.95--0.99 -- see G.5a.)

### D.2 The 50% CI problem

All envelope methods show massive over-coverage at the 50% level (85% actual vs 50% nominal at moderate n, decreasing to 64% at n=10,000). This comes from two finite-sample effects, not from a structural defect in the envelope operator:

1. **Wilson floor** (dominant at small n): directly widens the band beyond what the bootstrap produces.
2. **Bootstrap step-function conservatism**: the discrete jumps of bootstrap ROC curves inflate the supremum statistic relative to the smooth population process, biasing the critical value upward and making the band too wide.

Both effects diminish with n. The 50% CI is disproportionately affected because the theoretical 50% band is narrow, so small upward biases in the critical value produce proportionally large excess coverage. Additionally, the sup-norm's critical value ratio c_{0.95}/c_{0.50} is modest (typically 1.3-1.5x for correlated processes), meaning the 50% band is only modestly narrower than the 95% band to begin with.

**Implication:** The envelope method is best used for high-confidence bands (90-99%). The 50% over-coverage is not inherent to the envelope operator -- it is a finite-sample artifact that diminishes with n -- but in practice it means the method communicates limited information at lower confidence levels for the sample sizes where it is most useful (n <= 1000).

**The 50% failures are a different phenomenon from the 95% failures.** At alpha=0.5, violations spread across the whole curve (region rates at n=10,000: 24% in FPR 0-10%, but also 4-6% in every other region) and the AUC gradient *reverses* at moderate n (high-AUC cases have higher 50% coverage up to n=1000). This is the signature of a global calibration problem -- the sup-norm critical value is too large -- not of the boundary mechanism that drives the 95% failures, which are tightly localized at the first grid points (D.9).

**Contrast with other simultaneous methods:** Any sup-norm-based simultaneous band (not just the envelope) would exhibit weak sensitivity of width to alpha. The HT-autocalib method avoids this because it uses a *pointwise* variance model scaled by a *single* critical value, giving it continuous tunability across confidence levels. This is a genuine structural advantage of variance-model-based approaches over envelope approaches.

### D.3 The large-n problem

Coverage at prevalence 50% is 0.915 at n=1,000 and 0.830 at n=10,000 -- the degradation is already material at n=1,000 (see D.1). The mechanism is the first-k fragile zone of B.8/D.9: one-sided bootstrap support collapse plus upward bias of the empirical ROC at the first few grid points, where the Wilson floor cannot fire.

**Key finding:** Violation magnitudes are usually small even when coverage is lost, and they shrink with n while frequency grows. Among violators, the median max violation falls from 5.0pp (n=30) to 0.5pp (n=10,000), and the fraction of violations under 1pp rises from 9% to 66%. Mean max violation at n=10,000 is ~0.002 (0.2pp of TPR); the P99 is ~0.046. Only 0.84% of n=10,000 simulations have any violation exceeding 5pp -- but this pools all AUC levels; see D.4 for the high-AUC stratum, where the >5pp rate reaches 7%.

**Direction and geography:** At 95% confidence, failures are predominantly lower-bound failures: `violation_below` occurs in 4.65% of simulations versus `violation_above` in 0.39%. At n=10,000 this imbalance widens to 15.69% below versus 1.49% above. Regionally, the first 10% of FPR dominates: at n=10,000, the 0-10% FPR region violates in 15.09% of simulations, versus 0.77% in 10-30% and below 0.65% in each higher-FPR region. Within the 0-10% region, the localization is far tighter than the binning suggests: re-evaluation of violating cases shows a median violation FPR of ~0.00015, with 87.5% of violations below FPR 0.001 -- the first one to five grid points.

**The optimism signature.** Violations are not random draws -- they are *optimistic* draws. Among violated-below cases, the realized empirical AUC exceeds the true AUC by +0.16 on average at n=30, +0.024 at n=300, and +0.003 at n=10,000, versus ~+0.001 for covered cases. The few violated-above cases show the mirror image (empirical AUC 0.036 *below* truth). The band is anchored on the empirical curve; when the draw is lucky in the extreme negative tail (thin observed tail, high apparent early TPR), the bootstrap -- which resamples that same lucky tail -- cannot widen the band downward enough, and the truth escapes below.

**Implication:** The coverage drop at large n is mostly a *technical* failure (the true ROC escapes the band by tiny amounts), but not always. Whether this matters depends on the use case:
- For regulatory submissions where a stated 95% guarantee must hold formally: this is a problem.
- For exploratory analysis where the question is "roughly where is the true ROC?": this is fine.
- For high-AUC applications where decisions depend on very low FPR, the rare large lower-tail misses are directly relevant and should not be averaged away.

### D.4 DGP robustness is the standout property, conditional on AUC

Coverage is much more stable across DGP families than across AUC and sample size. At 95% confidence, coverage ranges are about 1.1pp at n=30, 3.5pp at n=100, 6.3pp at n=300, 3.9pp at n=1000, and 5.2pp at n=10000. That is not literally "the same coverage," but it is far more robust than the parametric competitors.

The sharper abstraction is: **distribution family is second-order; ROC geometry is first-order.** High AUC creates a steep low-FPR segment, and that is where the bootstrap tail/support problem lives. Within each DGP, the high-AUC subset is consistently worse than the rest. For example, at n=10,000, `envelope_wilson` coverage is 0.848 for AUC < 0.9 but 0.752 for AUC >= 0.9, and the gradient is monotone in AUC: 0.907 / 0.838 / 0.772 / 0.759 / 0.742 across AUC bins (0.5-0.7 / 0.7-0.8 / 0.8-0.9 / 0.9-0.95 / 0.95-1.0). Within-DGP point-biserial correlations of violation with true AUC are +0.19 to +0.27 at n=10,000 for five of seven DGPs; the exceptions (hetero_gaussian +0.02, weibull +0.05) are families whose high-AUC parameterizations produce shallower early ROC slopes, reinforcing that the slope, not the AUC number itself, is the risk factor. In the high-AUC Student-t subset, overall coverage drops to 0.875 and P99 max violation rises to 0.163 (a small curiosity: within student_t, *heavier* tails -- lower df -- are slightly safer, plausibly because heavy-tailed negatives place observations deeper into the extreme tail, extending the bootstrap's support exactly where it matters).

Compare Working-Hotelling: coverage ranges from <20% (Student-t with low df) to >95% (heteroscedastic Gaussian). The parametric methods live or die by their assumptions; the bootstrap doesn't care.

**Implication:** This is the strongest argument for the bootstrap approach, but it should be framed correctly. In practice, the data scientist often does not know the DGP, but they can estimate the empirical ROC geometry. A method whose main risk is tied to observable geometry (high AUC, steep low-FPR slope) is easier to diagnose than a method whose risk is tied to unverified distributional assumptions.

### D.5 The Wilson floor ablation

The simulation report includes `envelope_standard` (no Wilson floor), which achieves ~50% coverage. This confirms the theoretical prediction: the bare bootstrap envelope is not a valid confidence band, at any sample size, because boundary variance collapse causes systematic under-coverage.

The ablation also reveals *where* the floor earns its keep, and it is not where the method ultimately fails. `envelope_standard`'s violations concentrate at **high** FPR: at n=10,000, 55% of simulations violate in the FPR 90-100% region and 9% in 70-90%, versus 17% in 0-10%. The Wilson floor essentially eliminates the upper-right corner (90-100% rate drops from 55% to 0.6%) while only modestly improving the low-FPR corner (17% to 15%). The asymmetry is exactly the Hsieh-Turnbull variance decomposition made visible: near (1,1) the ROC is flat, the slope term vanishes, and binomial TPR variance -- which Wilson models correctly -- is the whole uncertainty; near (0,0) on a steep curve the slope term dominates and a binomial floor is the wrong yardstick.

The Wilson floor is therefore not a "band-aid" -- it is the *correct and complete* fix for the TPR-plateau corner, and the reason the method works at all. But it is structurally incapable of fixing the steep corner, which is where the residual failures live. Without the floor the method doesn't work; with it, the method works well for n <= 300-500 per class and degrades steadily beyond, with the residual failures confined to the first few grid points.

### D.6 Logit-space construction hurts

All logit-transformed envelope methods show dramatically worse coverage (35-40% at 95% CI). This is surprising given that logit transforms are standard variance-stabilizing tools.

**Likely explanation:** The Haldane-Anscombe correction maps TPR=0 and TPR=1 to finite values rather than +/- infinity, but the logit transform still stretches the boundaries. Curves that are tightly clustered near TPR=0 in probability space become spread out in logit space, but in a way that doesn't improve the boundary problem. The logit transform was designed for pointwise intervals (where it prevents the band from escaping [0,1]); for the envelope operator, the [0,1] constraint is already enforced by clipping, and the logit distortion just makes the envelope wider in the interior (where it was already over-conservative) without helping at the boundary (where the bootstrap has zero variance regardless of the transform).

### D.7 Hsieh-Turnbull's calibration advantage

The `HT_log_concave_logit_autocalib_wilson` method has the best overall calibration (smallest total deviation from nominal at both 95% and 50% levels). Its 50% CI coverage is 0.611 -- much closer to the 0.50 target than any envelope method's 0.85.

**Why:** HT uses pointwise variance estimates scaled by a single critical value, not an envelope operator. The critical value can be smoothly adjusted by the bootstrap calibration step. This gives HT *continuous tunability* across confidence levels. Any sup-norm-based simultaneous method (including the envelope) has inherently weak sensitivity to alpha because the critical value ratio c_{0.95}/c_{0.50} is modest for correlated processes. HT avoids this because its band width is directly proportional to z * SE(t), where z is the single critical value.

**The trade-off:** HT requires density estimation. When the log-concavity assumption fails, coverage collapses. The method is best-calibrated *conditional on the assumption holding*, but fragile to violations.

### D.8 The variance-ratio floor is behaviorally equivalent to the hard cutoffs (paired validation)

The April 2026 rewrite replaced the hard count cutoffs (`tail_k_min=(15,10)`, `tail_m_min=10`) with a variance-ratio gate: deficiency weights `max(0, 1 - bootstrap_var/wilson_var)`, Sidak correction over the continuous effective dimensionality `K_eff = sum(deficiency)`. Two validation efforts now exist:

**The selection-biased rerun (`scripts/validate_wilson_update.py`).** This script re-evaluates only the cases where the old method violated at n=10,000, and it has two design limitations that must be kept in mind when reading its output. First, it is structurally blind to regressions: it can never observe a case the old method covered but the new method misses. Second, its seeding (`default_rng(seed=lhs_idx)`) does *not* reproduce the original simulation's datasets (those came from a shared RNG stream advanced across LHS combinations), so it tests the new method on *fresh draws* from violation-prone parameter settings -- the comparison is contaminated by regression to the mean. It also uses a denser FPR grid (10,001 points vs the original n_neg+1 = 5,001), which inflates K_eff and makes the Wilson floor more conservative than it would be in the original setup. The stored checkpoint covers only beta_opposing (the run was interrupted after the first DGP; the other DGPs' PNGs are from an earlier run). Its result: of 121 old violations, 40 (33%) still violate on a fresh draw, with median violation FPR ~0.00015 and median magnitude ~1.3pp.

**The paired comparison (`scratch_paired_comparison.py`).** To remove both biases, 1,400 cases (4 DGPs x {n=1000, n=10000} x AUC-stratified LHS draws) were run through *both* implementations on identical datasets and identical bootstrap matrices. Result: the implementations are behaviorally equivalent.

| Stratum | n cases | Old coverage | New coverage |
|---|---|---|---|
| n=1000, AUC <= 0.9 | 400 | 0.9425 | 0.9425 |
| n=1000, AUC > 0.9 | 400 | 0.8450 | 0.8475 |
| n=10000, AUC <= 0.9 | 300 | 0.8200 | 0.8200 |
| n=10000, AUC > 0.9 | 300 | 0.7467 | 0.7500 |

Only 4 of 1,400 paired outcomes are discordant (2 fixed, 1 broken, 1 at n=1000). Violation geography, direction, and magnitudes are indistinguishable; band areas agree within ±2% (the new gate is ~2% tighter at n=1000 for low AUC, ~2% wider for high AUC).

**Why equivalence was inevitable.** The two implementations reach the same band on the failing (lower, low-FPR) side by different routes. The new gate asks "is the bootstrap variance below the Wilson variance?" -- but at the failure points (k=1-3, mid-range TPR on a steep curve) the bootstrap variance is roughly *correct* and the Wilson variance is 3-6x too *small* in sd (it omits the dominant slope term), so the measured ratio is 10-60, the deficiency is zero, and the gate never fires there (every fired point in the probed violating cases sits at FPR > 0.59). The old count gate *did* fire at k < 15 -- but its floor was one-sided by design, raising only the *upper* bound in the lower tail (on the mirror-image assumption that the (0,0) corner pin protects the lower band; see B.6). Either way, the lower bound at the failure points is the bare envelope arm: the two implementations' lower bands are bit-identical at k=1-25 on violating cases. The gate redesign improved the aesthetics (no magic counts, grid-adaptive correction strength) without touching the failure mechanism. Two practical corollaries: the stored 2.25M-evaluation results transfer to the current code, and further iteration on *gate design* is a dead end -- a fix must put a calibrated correction on the *lower* side of the first grid points, which neither version ever did.

### D.9 Failure-point microscopy: what is actually wrong at the first grid points

For violating high-AUC cases at n=10,000, three quantities were compared at low-FPR grid points against the *true* sampling distribution of the empirical ROC (2,000-replicate Monte Carlo per case): the bootstrap sd, the Wilson sd, and the Hsieh-Turnbull asymptotic sd evaluated with the true slope (`scratch_variance_decomposition.py`). Median ratios to the true MC sd:

| FPR | bootstrap sd / truth | Wilson sd / truth | HT(true slope) sd / truth |
|---|---|---|---|
| 0.001 | 1.16 | 0.18 | 1.03 |
| 0.005 | 1.12 | 0.27 | 1.01 |
| 0.02 | 1.09 | 0.41 | 1.00 |
| 0.10 | 1.00 | 0.64 | 1.00 |
| 0.30 | 1.01 | 0.82 | 0.99 |

Three conclusions overturn earlier assumptions:

1. **The bootstrap variance is not the problem at these FPRs.** It is approximately unbiased (though noisy, 0.5-1.9x across cases) down to FPR=0.001. The earlier claim that "bootstrap variance collapses" at the boundary is accurate only at the very first grid points (k=1-3, where the median ratio falls to ~0.57) and at literal corners.
2. **The HT variance formula is essentially exact.** With the true slope plugged in, it matches Monte Carlo truth to 1-3% everywhere probed. The obstacle to HT-style corrections is purely the slope/density estimation, not the theory.
3. **The Wilson variance is the wrong yardstick at low FPR by a factor of 3-6 in sd.** This is what makes every Wilson-referenced gate inert at the failure points (D.8).

The remaining question -- if pointwise variance is right, why does the band fail? -- is answered by the arm-width and bias measurements in B.8: at k=1-3 the envelope's lower arm reaches only 0.8-1.4 sd below an empirical curve that is itself biased 0.4-0.7 sd upward. The failure is a *one-sided, bias-compounded support problem at a handful of identifiable grid points*, not a variance-estimation problem. This is a much more hopeful diagnosis than "the bootstrap fails at the boundary": the fragile set is small (k <= ~10), its location is known a priori from n_0 alone, and the geometry of the failure (exceedance of the k-th order statistic) has an exact, distribution-free characterization (see G.5).

---

## E. Progress, Walls, and Remaining Uncertainties

### E.1 Important progress

1. **The Wilson floor is a genuine contribution.** It transforms a broken method (50% coverage) into one that is near-exact for n up to ~300-500 per class and 0.915 at n=1000 (prevalence 50%). The insight that the bootstrap boundary problem can be patched with a simple binomial correction -- and that this correction should be restricted to the tails to preserve the bootstrap's advantages in the interior -- is the core intellectual contribution of the project. The June 2026 microscopy sharpened where the credit belongs: the floor *completely* repairs the TPR-plateau corner that destroys the bare envelope (90-100% FPR violation rate: 55% without the floor, 0.6% with it).

2. **The simulation study is comprehensive and well-designed.** 7 DGPs, 6 sample sizes, 23 methods, 1000 LHS combinations per DGP, multiple confidence levels. The Latin Hypercube sampling over DGP parameters is a particularly good choice: it ensures coverage of the parameter space without exponential blowup. The evaluation framework (BandResult, BandEvaluation) is clean and the metrics are well-chosen.

3. **The failure set is now precisely localized and mechanistically understood** (D.8, D.9). Failures live at k=1-10, not k=15-50, and the binding constraints there are one-sided bootstrap support collapse and upward bias of the empirical ROC -- not variance underestimation. The right two-corner model: the TPR-plateau corner (high FPR, TPR near 1) is fully repaired by the Wilson floor because binomial variance is the whole story there; the steep corner (low FPR, high AUC) cannot be repaired by any binomial-variance floor because the slope term dominates.

4. **Typical graceful degradation is real, but the tail risk needs clearer language.** At n=10,000, mean max violation is ~0.002 and P99 is ~0.046, so most failures are tiny. But high-AUC, low-FPR lower-bound failures are large in absolute TPR and not rare within that stratum (4-7% of AUC > 0.95 cases miss by >5pp at n >= 300). The qualitative difference from parametric methods is not "this method never misses badly"; it is "most misses are small and localized, and the large misses occur in a diagnosable low-FPR/high-AUC regime."

5. **The variance-ratio gate question is closed.** The paired comparison establishes equivalence with the hard cutoffs at negligible compute cost, recovering the full value of the stored simulation suite for the current implementation and ruling out the entire family of Wilson-referenced gate refinements in one experiment.

### E.2 Walls

1. **Sup-norm-based simultaneous bands have weak sensitivity to alpha.** The critical value ratio c_{0.95}/c_{0.50} for a supremum statistic over many correlated grid points is inherently modest (~1.3-1.5x). This means the 50% band is only modestly narrower than the 95% band, regardless of the construction method (envelope, studentized band, etc.). Combined with finite-sample bootstrap conservatism, this makes the 50% band substantially over-conservative in practice. The over-coverage diminishes with n but remains noticeable for the sample sizes where the method is most useful. This is not specific to the envelope operator -- it affects any method that determines simultaneous coverage via a sup-norm critical value. **Confirmed empirically (G.1):** A variance-model band (R_hat ± c*sigma_hat, no envelope operator) was implemented and tested. It does not improve 50% CI calibration relative to the envelope, and in most scenarios makes it worse, because the same sup-norm mechanism governs c for both methods. The weak sensitivity to alpha is a property of the supremum statistic itself, not of the envelope construction.

2. **No Wilson-referenced gate can fix the large-n coverage gap -- the gate family is exhausted.** Possible Wilson gate tweaks (raise k_min, adaptive thresholds, smooth blending, the variance-ratio gate) all decide *where* to apply a binomial-variance floor. The paired comparison (D.8) shows the variance-ratio gate is equivalent to the hard cutoffs, and the microscopy (D.9) shows why every member of this family must be: at the failure points the bootstrap variance *exceeds* the Wilson variance, so any criterion of the form "apply the floor where the bootstrap is deficient relative to Wilson" evaluates to "do nothing" exactly where help is needed. What is wrong at those points is the floor's *yardstick* (binomial-only variance, missing the dominant slope term) and the band's *center* (upward-biased empirical ROC), not the gate. Fixes must change what is applied, not where.

3. **The bootstrap can't help at the boundary -- and the deficit is one-sided.** The nonparametric bootstrap resamples from the empirical distribution and therefore cannot represent uncertainty about probability mass beyond the observed data. At the first grid points this manifests asymmetrically: resampled curves deviate freely upward but barely downward (lower envelope arm 0.8-1.4 sd vs upper arm 2.6-2.8 sd at k=1-3), while the empirical center is biased upward. No clever resampling scheme fixes this within the nonparametric framework -- including the multiplier bootstrap, which reweights the same observed support. The integrated Beta floor (G.5a) circumvents the wall rather than breaching it: at those points it abandons resampling entirely in favor of the exact order-statistic law.

### E.3 Remaining uncertainties

1. **Is the large-n coverage gap practically important?** For most ROC analyses in biomedical and ML applications, n < 1000, and the method is near-exact up to n ~ 300-500 per class. But the gap is not confined to n=10,000: it is 3.5pp at n=1000 with balanced classes, and within the high-AUC stratum -- where ROC analysis is most often deployed in production -- it appears by n=300. Whether this matters depends on the field and the stakes; it is an empirical question about use cases, not a technical question. The earlier framing of "works well for n <= 1000" was too generous.

2. **Can the fragile first-k points be better served?** Now substantially answered in the negative for variance-based interventions and in the affirmative for geometric ones. The failure at k=1-10 is bias plus one-sided support collapse, so a wider *symmetric* floor (HT-variance or otherwise) is a blunt instrument: it would have to be slope-aware precisely where slope estimation is hardest, and it widens the upper arm needlessly. The geometry of the failure has an exact distribution-free characterization through the order-statistic law of the FPR exceedance, now implemented as the integrated Beta floor (G.5a): on the problem-domain strata it lifts coverage to 0.95-0.99 and eliminates >5pp misses.

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

**Restricting parametric corrections to where they're needed (from envelope_boot).** The insight that the Wilson floor should *not* be applied everywhere is important: it preserves the bootstrap's advantages in the interior while patching its weakness at the boundary. The refinement from D.5/D.9: the floor's success region is specifically the TPR-plateau corner, where binomial variance is the complete uncertainty model. The architecture lesson generalizes -- each region should get the correction whose variance model is actually complete there -- which for the steep corner means a slope-aware or order-statistic correction, not a wider binomial one.

### F.2 Ideas that could be developed further

**Exact order-statistic (horizontal) tail bounds.** The failure points are the first few grid points, where the operating threshold is an extreme order statistic of the negatives. For any continuous score distribution, the true FPR exceedance at the k-th largest negative score is *exactly* Beta(k, n_0+1-k) -- distribution-free, no asymptotics, no density estimation. This gives an exact horizontal (FPR-direction) uncertainty for the first grid points, which the local ROC slope converts into precisely the vertical uncertainty the Wilson floor misses. See the revised G.5 for the construction. This idea has graduated from proposal to production: it is now integrated into `envelope_boot.py` and validated on the problem-domain strata (G.5a), confirming that it targets the measured failure mechanism (threshold-location error plus bias at k=1-10) with an exact, assumption-free tool.

**Hsieh-Turnbull variance as a variance floor near the tails.** The HT variance captures both components but requires estimating the density ratio, equivalently the ROC slope. D.9 showed the HT formula with the true slope is essentially exact at every probed FPR, so the only obstacle is slope estimation. However, the microscopy also showed the binding constraint at the failure points is one-sided support collapse and center bias, not symmetric variance shortfall -- a symmetric HT floor is therefore second choice behind the order-statistic bound, and it must estimate the slope exactly where that is hardest.

**Bootstrap-calibrated Wilson bands.** The Wilson Rectangle method with Sidak correction achieves 0.911 coverage and is the tightest method with >= 90% coverage. Its main weakness is the lack of true simultaneity control. What if you used bootstrap calibration (as in HT-autocalib) to determine the critical value z, but applied it to Wilson intervals instead of HT intervals? This would combine Wilson's always-positive-width property with bootstrap-calibrated simultaneity. **Caveat from G.1:** The Wilson variance is a smooth function of t (it depends only on the empirical TPR and n_pos, not on pointwise bootstrap samples), so it would *not* suffer from the noisy-variance problem that sank the variance-model band. This idea remains viable and may be the most promising path to improved alpha tunability, since it pairs a smooth variance with bootstrap-calibrated simultaneity.

**Envelope with HT as the base instead of the bootstrap.** Rather than enveloping bootstrap curves (which inherit the boundary problem), envelope curves generated from the HT variance model. At each bootstrap replicate, compute R_b(t) = R_hat(t) + sigma_HT(t) * z_b(t) where z_b(t) is drawn from the bootstrap distribution of the studentized process. This would combine the envelope's adaptation properties with HT's variance model, potentially getting the best of both worlds. The risk is that HT's density estimation errors would propagate into the envelope.

### F.3 The hybrid insight

The most important lesson from this project is that **no single uncertainty quantification strategy works everywhere on the ROC curve**. The bootstrap works in the interior but fails at the boundary. Wilson works at the boundary but is too simple for the interior. HT captures both variance components but requires density estimation that fails in the tails and under model misspecification.

The envelope_wilson method succeeds because it is a hybrid: bootstrap in the interior, Wilson at the boundary. The next generation of improvement should keep the hybrid architecture but match each corner to the uncertainty that actually dominates there: bootstrap in the interior, Wilson at the TPR-plateau corner (where it is provably the complete model), and an order-statistic/slope-aware correction at the steep corner's first grid points (where binomial variance is the wrong yardstick by 3-6x in sd).

The G.1 experiment (variance-model band) tested whether the envelope operator could be replaced with a pointwise variance-model approach using bootstrap-calibrated simultaneity. The answer is no -- not with pointwise bootstrap variance, which is too noisy for supremum-based calibration. The envelope operator has an underappreciated robustness property: it tolerates noisy variance estimates because it operates on retained curves directly rather than studentizing against a variance function. Replacing it requires a *smooth* variance estimate (like HT's analytical variance), not just a different band construction.

The "ugly band-aids" feeling is real but misleading. The Wilson floor is not a band-aid -- it is the correct response to a structural limitation of the nonparametric bootstrap. The method is a hybrid *by necessity*, not by accident. The question is not whether to hybrid, but how to do it more gracefully.

---

## Summary Table

| Property | envelope_wilson | KS | HT-autocalib | Wilson Rect (Sidak) | WH |
|---|---|---|---|---|---|
| Coverage at 95%, n=300 | 0.953 | 1.000 | 0.746 | 0.941 | ~0.80* |
| Coverage at 95%, n=1000 (prev 50%) | 0.915 | 1.000 | 0.967† | 0.841 | -- |
| Coverage at 50% | 0.851 | 0.982 | 0.611 | 0.247 | -- |
| Mean band area | 0.397 | 0.469 | 0.536 | 0.331 | -- |
| DGP robustness | Excellent | Perfect | Moderate | Good | Poor |
| Large-n (10k) coverage | 0.830 | 1.000 | 0.926 | 0.839 | -- |
| Max violation (P99) | 0.037 | 0.000 | 0.067 | 0.267 | -- |
| Tunability (50% CI) | Poor | Poor | Good | Fair | -- |
| Distributional assumptions | None | None | Log-concave | None | Binormal |

*WH coverage varies wildly by DGP, from <20% to >95%.
†Prevalence-pooled; the HT-autocalib n=1000 figure was not recomputed for prevalence 50% alone.
‡The envelope_wilson column predates the integrated Beta order-statistic floor (G.5a), which raises coverage on the previously failing problem-domain strata to 0.95-0.99 at an area cost of 0.3-11%.

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
  (FPR < 0.05), where the Wilson floor is *not* active -- the first-k
  fragile zone identified in Section B.8.
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

### G.5 Tail corrections that carry threshold uncertainty (Priority 1)

The failure mechanism is now measured (B.8, D.9): at the first grid points,
threshold-location uncertainty -- not binomial TPR uncertainty -- dominates,
and it manifests as upward bias plus a one-sided lower-arm collapse. Two
constructions address it directly; the first is now the project's most
promising open direction.

**G.5a Exact order-statistic (Beta) horizontal bounds -- most promising.**
For any continuous score distribution, the probability-integral transform
gives an exact, distribution-free law for the true FPR at the k-th largest
observed negative score:

    F_bar(X_(k)) ~ Beta(k, n_0 + 1 - k)

For small k this distribution is wide and right-skewed: at k=1 with
n_0=5000, the 97.5th percentile of the true FPR is ~3.7/n_0 -- nearly 4 grid
points to the right of the nominal t=1/n_0.

**The exact construction** (correcting an earlier, loosely stated version):
let q_j be the (1-a_os) upper Beta quantile for order statistic j. On the
event {U_j <= q_j} (probability >= 1-a_os), monotonicity of R_true gives, for
every evaluation point t >= q_j:

    R_true(t) >= R_true(U_j) = G(X_(j)) >= WilsonLower_{a_wil}(R_hat(j/n_0))

So the lower band at evaluation point t is the Wilson-lowered empirical TPR
at the *largest j whose Beta upper quantile is <= t* -- the bound looks
*backward* to a smaller-FPR operating point whose true FPR provably (whp)
sits at or below t. For t < q_1 (~3.7/n_0 at a_os=0.025; ~6.9/n_0 at
a_os=0.001) no j qualifies and the exact bound is vacuous: the honest
nonparametric lower band at the first few grid points is ~0. Applied as a
one-sided floor, L <- min(L_envelope, L_beta), on grid points within the
floor's jurisdiction (k <= ceil(q_J * n_0), with j = 1..J; J=25 in the
prototype). A mirrored construction would handle the upper-right corner,
though the Wilson floor already covers it.

**Prototype results -- positive (June 2026; `scratch_beta_floor_experiment.py`).**
Run on the same 1,400 paired-comparison cases (identical data and bootstrap
matrices), with two alpha budgets: "strict" (extra alpha 0.05 total,
Bonferroni over the 2J=50 one-sided events: a_os = a_wil = 0.001) and
"generous" (0.01 per event, uncorrected; the order-statistic events are
nested, so Bonferroni is far too conservative and generous brackets the
achievable end):

| Stratum | n cases | Envelope | + floor (strict) | + floor (generous) |
|---|---|---|---|---|
| n=1000, AUC <= 0.9 | 400 | 0.9375 | 0.9875 | 0.9875 |
| n=1000, AUC > 0.9 | 400 | 0.8550 | 0.9875 | 0.9825 |
| n=10000, AUC <= 0.9 | 300 | 0.8200 | 0.9567 | 0.9467 |
| n=10000, AUC > 0.9 | 300 | 0.7667 | 0.9733 | 0.9733 |

- **Fix anatomy.** The floor fixed 73/83 envelope violations at n=1000 and
  103/124 at n=10000, breaking zero covered cases (it is a pure widening).
  Fixed cases had their worst violation at median grid index k=1 -- the
  measured failure set. The remaining violations sit at median k=86
  (n=1000) and k=305 (n=10000), *outside* any tail jurisdiction (zone ends
  ~k=44): a separate, milder interior-calibration residue.
- **Catastrophic misses eliminated.** The >5pp violation rate in the
  AUC > 0.9 stratum drops from 7.25% (n=1000) and 3.0% (n=10000) to 0.0%
  under both budgets.
- **Area cost negligible-to-modest.** +0.0005 on a 0.023 mean area at
  n=10000 high-AUC; +0.0086 on 0.080 (~11%) at n=1000 high-AUC.
- **Red-team A (ties / discrete scores) -- passed**
  (`scratch_beta_floor_discretization.py`). The Beta law assumes continuous
  scores (A2). Quantizing scores into 100 or even 20 bins -- the top bin
  collapsing the entire upper tail into a single atom, exactly where the
  order statistics live; truth recomputed by 2M-draw Monte Carlo on the
  discretized scores -- produced **zero** floor-zone violations across 90
  high-AUC n=10000 cases per level. With atoms, the exceedance
  F_bar(X_(j)) is stochastically *smaller* than the Beta law, so
  discreteness errs conservative rather than breaking the bound.
- **Red-team B (informativeness at operational FPRs) -- a real, fundamental
  cost.** The floor zeroes the lower band below q_1 and weakens it for
  several grid points beyond. At n=10000 the cost is mild (high-AUC median
  lower band at FPR=0.002: 0.112 floored vs 0.168 unfloored; identical from
  FPR=0.01 on). At n=1000 it is stark: the strict variant's lower band is 0
  for FPR <= 0.01 (vs ~0.20-0.28 unfloored) and 0.30 vs 0.44 at FPR=0.02;
  the generous variant recovers most of FPR >= 0.01. This is not a defect of
  the fix -- it is the fix exposing that the unfloored band's claim of
  L ~ 0.2 at FPR=0.002 with n_0=500 (k=1!) was never supportable, which is
  exactly why 14.5% of high-AUC n=1000 envelope cases violated. The
  practical consequence: a valid nonparametric lower band below
  FPR ~ 4-7/n_0 does not exist, and applications needing low-FPR guarantees
  must size n_0 accordingly.

**Integrated into the production method (June 2026;
`scratch_beta_floor_integrated.py`).** The floor now runs inside
`envelope_bootstrap_band` as part of `boundary_method="wilson"`, applied to
the lower band after the Wilson Rectangle floor: J = 25 order statistics,
per-event alpha = α/(2J) (= 0.001 at α = 0.05, the strict budget),
jurisdiction t <= q_J ~ 43/n_0, vacuous below q_1 ~ 6.9/n_0. The integrated
implementation reproduces the prototype to float32 precision. Re-running the
1,400 problem-domain cases (fresh bootstrap matrices) against the captured
pre-floor band:

| Stratum | n cases | Old (Wilson floor only) | New (+ Beta floor) | Area ratio |
|---|---|---|---|---|
| n=1000, AUC <= 0.9 | 400 | 0.948 | 0.988 | 1.020 |
| n=1000, AUC > 0.9 | 400 | 0.843 | 0.990 | 1.106 |
| n=10000, AUC <= 0.9 | 300 | 0.813 | 0.953 | 1.003 |
| n=10000, AUC > 0.9 | 300 | 0.767 | 0.977 | 1.020 |

75/84 old violations fixed at n=1000 and 105/126 at n=10000, zero covered
cases broken; remaining violations sit at median k=103/k=222, outside any
tail jurisdiction. The >5pp violation rate in the high-AUC stratum drops
from 7.25%/3.67% to 0.0%.

**Band attribution.** Fraction of x-axis grid points where each mechanism
sets the final band (a point is "set by" the mechanism that strictly
determined its bound):

| Stratum | Lower: Beta | Lower: Wilson rect | Lower: bootstrap | Upper: Wilson rect |
|---|---|---|---|---|
| n=1000, AUC <= 0.9 | 7.5% | 14.0% | 78.5% | 0.3% |
| n=1000, AUC > 0.9 | 8.1% | 67.2% | 24.7% | 0.5% |
| n=10000, AUC <= 0.9 | 0.8% | 6.8% | 92.4% | 0.8% |
| n=10000, AUC > 0.9 | 0.8% | 45.0% | 54.2% | 5.4% |

The geography is the two-corner model made quantitative: the Beta floor
governs only its extreme-FPR jurisdiction, the Wilson rectangle owns the TPR
plateau (two-thirds of the lower band at high-AUC n=1000), the bootstrap
envelope sets the interior, and the upper band is almost entirely bootstrap.
The three mechanisms barely overlap, which is why the combination works:
each carries the uncertainty channel the others cannot see.

**Refinement opportunities:** (i) joint (non-Bonferroni) calibration of the
nested {q_j} events should recover width -- generous matched strict on
coverage with visibly better informativeness; (ii) fold the floor's alpha
into the band's overall budget instead of bolting it on (the integrated
band overshoots nominal: 0.95-0.99 where the target is 0.95); (iii) the
residual k ~ 50-500 violations are untouched by construction and bound the
achievable gain.

Properties that make this attractive:
- **Exact and distribution-free.** The Beta law holds for every continuous F
  at every n; no density estimation, no asymptotics, no tuning beyond the
  alpha split (a Sidak correction over the ~10-25 fragile points, not over
  the whole grid).
- **One-sided by construction.** It widens only the lower arm at the steep
  corner -- exactly the measured deficit -- instead of symmetrically
  inflating the band.
- **Bias-aware.** The empirical ROC's upward bias at k=1-3 comes from the
  same order-statistic geometry; mapping through the empirical curve at the
  Beta-upper FPR absorbs it. Equivalently, R_hat(t_hi) is a much
  better-estimated quantity than R_hat(t) for tiny t.
- **Self-deactivating.** For flat ROC regions (low AUC), R_hat(t_hi) is
  close to R_hat(t) and the correction costs almost nothing; for steep
  regions it widens the band precisely in proportion to the realized slope.
  No gate is needed -- the geometry is the gate.
- **It explains an existing observation -- and fixes its resolution problem.**
  The Wilson Rectangle band is relatively robust at low FPR because its
  FPR-direction margin is a normal-approximation version of exactly this
  correction. But the rectangle's corner geometry has no resolution at the
  scale that matters: its leftmost lower-right corner sits at
  fpr_upper(0) ~ z^2/(n_0+z^2), with the degenerate (FPR=0, TPR=0) operating
  point below it, so a rectangle lower bound evaluated at k <= 5 collapses
  toward 0 (measured: ~0.000 at k=1-5 on violating n=10,000 cases) --
  protective but vacuous, jumping from "no correction" to "no information"
  across a single z^2/n_0-wide step. The Beta law replaces that one coarse
  step with a calibrated per-k margin (k=1: 97.5th percentile at ~3.7/n_0;
  k=3: ~2.4x the nominal FPR), which is exactly the resolution needed at
  the measured failure points.

**G.5b Finite-difference slope variance floor -- secondary.** The earlier
proposal: estimate the ROC slope by finite differences and use the
approximate HT variance R(1-R)/n_1 + slope² t(1-t)/n_0 as a floor. D.9
strengthened the case for the HT *formula* (exact to 1-3% with the true
slope) but weakened the case for this *intervention*: the failure points
need an asymmetric, bias-aware correction, and a symmetric variance floor
must estimate the slope exactly where finite differences of a step function
are noisiest. Retain as a fallback or as a smooth variance source for
bootstrap-calibrated bands (F.2), not as the primary tail fix.

### G.6 Priority and independence

These directions are largely independent and can be explored in parallel:

| Direction | Addresses | Effort | Status | Independent of |
|---|---|---|---|---|
| G.1 Variance-model band | 50% CI, tunability | Low | **Done — negative result** | G.2, G.3 |
| G.1a Calibration threshold | HT-autocalib robustness | Low | Characterized, not implemented | G.2, G.3 |
| Variance-ratio Wilson floor | Hard cutoff artifact | Low-moderate | **Done — equivalent to hard cutoffs (D.8)** | G.2, G.3, G.5 |
| G.2 Multiplier bootstrap | Interior calibration bias (50% CI) | Moderate | Open; cannot touch the tail problem | G.1, G.3, G.5 |
| G.3 Band depth ranking | Band tightness | Moderate | Open | G.1, G.2 |
| G.4 Split-sample calibration | Future empirical ROC calibration | Moderate-high | Open | All others |
| G.5a Exact Beta horizontal tail bounds | First-k fragile zone (the 95% coverage gap) | Low | **Integrated into `envelope_boot.py` and validated on problem domains; full-suite eval + joint alpha calibration open** | G.1, G.2, G.3 |
| G.5b Finite-difference slope floor | Smooth variance source | Low | Open, demoted | G.1, G.2, G.3 |

Two threads are now closed by experiment. G.1's negative result eliminated
the simplest path to improved alpha tunability and established that bootstrap
calibration requires a smooth variance estimate. The paired validation (D.8)
established that gate redesign cannot move coverage, because every
Wilson-referenced gate is inert at the failure points.

The measured failure anatomy (D.9) set the agenda, and G.5a has now delivered
on it end to end: the Beta floor is integrated into `envelope_bootstrap_band`,
coverage in the worst stratum (AUC > 0.9, n=10000) rises from 0.767 to 0.977,
>5pp misses vanish, the ties red-team passes, and the cost is concentrated
where the honest answer is "unknowable" (FPR < ~4-7/n_0). The remaining work
is refinement: joint calibration of the nested Beta events, folding the
floor's alpha into the band's overall budget (the integrated band overshoots
nominal), and a full-suite evaluation across all 7 DGPs and sample sizes
(including the 50% level, where the floor should be essentially inert since
those failures are global, not boundary-local).

Separately and orthogonally, the 50% CI over-coverage is a *global*
calibration problem (D.2), for which the candidates remain bootstrap-calibrated
Wilson bands (F.2) and the multiplier bootstrap (G.2). G.1a is a small,
self-contained improvement to the existing HT-autocalib method.

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

*Report generated 2026-04-28, updated 2026-05-01, 2026-06-09, and 2026-06-10. Based on code review of 5 method implementations, simulation specification, 2,254,000 evaluations across 7 DGPs, 6 sample sizes, and 23 method variants; targeted simulations of the variance-model band (G.1) across 4 scenarios at 500 simulations each; and the June 2026 follow-up experiments: a 1,400-case paired old/new implementation comparison (`scratch_paired_comparison.py`), Monte Carlo variance decomposition at low-FPR grid points (`scratch_variance_decomposition.py`), empirical-ROC bias measurement (`scratch_bias_check.py`), envelope arm-width microscopy at the first grid points (`scratch_width_check.py`), the G.5a Beta-floor prototype with discretization red-team (`scratch_beta_floor_experiment.py`, `scratch_beta_floor_discretization.py`), and the integrated-floor comparison with band attribution (`scratch_beta_floor_integrated.py`). Trial-level cross-tabs of the stored suite are reproduced by `scratch_profile_analysis.py`.*
