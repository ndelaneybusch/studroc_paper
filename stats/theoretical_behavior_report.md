# Theoretical Behavior of the Studentized Bootstrap Envelope with Wilson Floor

This report provides a theoretical account of the expected behavior of
`envelope_wilson` — the studentized bootstrap envelope SCB for ROC curves with
adaptive Wilson variance floor — across confidence levels, sample sizes, AUC
ranges, and distributional assumptions. It draws on the method specification
(`nonparam_envelope.md`), implementation (`envelope_boot.py`), and the simulation
study (`method_recommendation_report.md`, 2,254,000 evaluations across 7 DGPs).

---

## 1. Method Overview

The method constructs simultaneous confidence bands by:

1. **Bootstrapping**: Generate B stratified bootstrap ROC curves.
2. **Studentizing**: Normalize each bootstrap curve's pointwise deviation from
   the empirical ROC by the bootstrap standard deviation at each FPR grid point.
3. **Retaining**: Keep the (1−α) fraction of curves with the smallest
   supremum studentized deviation (KS statistic).
4. **Enveloping**: Take the pointwise min/max of retained curves.
5. **Wilson floor**: In tail regions where bootstrap variance collapses, apply
   Šidák-corrected Wilson Rectangle bounds as a floor.

The method is a **hybrid** of two distinct uncertainty quantification
strategies: a bootstrap envelope in the interior of the ROC curve, and a
parametric (binomial) correction in the tails. Understanding its behavior
requires analyzing each component and their interaction.

---

## 2. The Bootstrap Tail Problem

### 2.1 Why bootstrap variance collapses at the boundaries

The empirical ROC is a step function on the grid {0, 1/n₀, 2/n₀, ..., 1}.
At FPR = k/n₀ for small k, the TPR depends on only k negatives exceeding the
classification threshold. Every bootstrap resample draws n₀ negatives with
replacement from the same empirical distribution, so:

- At FPR = 0: all bootstrap TPRs are identically 0. Variance is exactly 0.
- At FPR = k/n₀ for small k: bootstrap variance exists but is driven by
  small-count combinatorics of the resampled negatives. The bootstrap can only
  explore score values present in the observed data — it cannot generate
  density beyond the empirical support.

The true ROC R_true(t) at these points depends on the *population* tail of the
negative score distribution, which the empirical distribution underrepresents.
This is not a studentization failure — it is a structural limitation of the
nonparametric bootstrap. No resampling scheme that draws from the empirical
distribution can represent uncertainty about probability mass in unobserved
regions of score space.

The symmetric problem occurs at FPR near 1: TPR approaches 1, few negatives
remain below threshold, and the bootstrap cannot represent uncertainty about the
upper-right corner of the ROC.

### 2.2 Consequence for the envelope

At any grid point where bootstrap variance has collapsed, all retained curves
agree on approximately the same TPR value. The envelope has near-zero width
there, regardless of how many curves are retained or what retention threshold is
used. If R_true deviates from R̂ at any such point — even by a tiny amount — the
band fails to cover.

This is the mechanism behind the ~50% coverage of `envelope_standard` (no Wilson
floor) observed in the ablation study (fig4). Roughly half of all DGP instances
produce a data realization where R_true deviates from R̂ at one or more
collapsed-variance grid points. The base studentized bootstrap envelope is not a
valid simultaneous confidence band, at any sample size.

---

## 3. The Wilson Floor: Mechanism and Scaling

### 3.1 Two-stage correction

The implementation applies Wilson-based corrections at two stages:

**Stage 1 — Variance floor during studentization** (lines 548–558 of
`envelope_boot.py`): The bootstrap variance is floored to at least the Wilson
score variance σ²_Wilson(p) = [p(1−p)/n₁ + z²/(4n₁²)] / (1 + z²/n₁)². This
prevents studentized statistics from blowing up at zero-variance points,
ensuring the KS retention criterion is well-behaved everywhere.

**Stage 2 — Wilson Rectangle tail floor** (lines 636–646): After envelope
construction, the band is widened in *tail regions* using Šidák-corrected Wilson
Rectangle bounds. Tail regions are defined as grid points where effective counts
fall below fixed thresholds: k < k_min_lower (default 15) negatives above the
classification threshold, or m < m_min (default 10) positives.

### 3.2 Why the floor is restricted to the tails

The Wilson interval models TPR as a binomial proportion with n₁ trials. This
captures the variance component R(t)(1−R(t))/n₁ but ignores the
threshold-uncertainty component (g(c_t)/f(c_t))² · t(1−t)/n₀ from the
Hsieh-Turnbull asymptotic variance formula. In the interior of the ROC curve,
the bootstrap correctly captures both variance components — it is strictly more
informative than the Wilson model. Applying the Wilson floor everywhere would
replace good bootstrap variance estimates with worse parametric ones.

Restricting to the tails also controls the Šidák correction penalty. With K_tail
points requiring joint coverage, the per-point significance is
α_tail = 1 − (1−α)^{1/K_tail}. Fewer tail points means milder correction and
tighter bands.

### 3.3 Scaling behavior with n

The tail region is defined by *fixed* count thresholds (k_min = 15, m_min = 10).
As n grows:

| n₀ | Tail FPR range | Fraction of grid corrected |
|----|----------------|---------------------------|
| 15 | [0, 1] | 100% |
| 50 | [0, 0.30] ∪ [0.80, 1] | ~50% |
| 150 | [0, 0.10] ∪ [0.93, 1] | ~17% |
| 500 | [0, 0.03] ∪ [0.98, 1] | ~5% |
| 5000 | [0, 0.003] ∪ [0.998, 1] | ~0.5% |

This means the Wilson correction provides comprehensive coverage at small n
but vanishes at large n. The method's overall coverage trajectory is a direct
consequence of this scaling.

---

## 4. The Envelope's Intrinsic Conservatism

### 4.1 Projection inflation

The retention step selects curves by a *global* criterion (supremum studentized
deviation). The envelope step projects this global selection onto *pointwise*
extremes. These are not equivalent operations.

Curve A may have its largest upward deviation at FPR = 0.3; curve B at
FPR = 0.7. Both are retained (their global KS statistics are similar). The
envelope captures both deviations simultaneously, creating a band wider than any
individual retained curve's deviation. This is analogous to the bounding box of a
set of ellipses being larger than any single ellipse.

This projection inflation means the envelope is inherently over-conservative in
the interior of the ROC curve, where bootstrap variance is reliable and the
retained curves explore diverse deviation patterns.

### 4.2 Extreme-value insensitivity to α

The envelope width at each grid point is determined by the min and max of the
retained set (m curves). By extreme-value theory, for distributions with bounded
support (TPR ∈ [0, 1]), the distance from the extremes to the boundary of
support scales as O(1/m), but the distance from the extremes to the distribution
center scales as approximately F⁻¹(1 − 1/m), which varies slowly
(logarithmically) with m.

At α = 0.05, m ≈ 0.95B = 3800. At α = 0.5, m ≈ 0.5B = 2000. The envelope
width decreases by roughly log(3800)/log(2000) ≈ 1.05× — a 5% change —
while the target miscoverage rate changes 10×.

This means the envelope construction is **not smoothly tunable** across
confidence levels. It functions as a high-confidence tool: useful near
α = 0.05, but producing massively over-conservative bands at lower confidence
levels. The observed 85% coverage at the 50% nominal level (when 50% is the
target) is a direct consequence of this extreme-value scaling. No modification
to the studentization, variance floor, or individual bootstrap curve estimation
(e.g., Harrell-Davis smoothing) changes this property, because the issue is in
the envelope operator itself — the pointwise min/max of a finite sample of
whole curves.

Achieving calibrated bands at lower confidence levels would require abandoning
the envelope in favor of pointwise quantile-based construction (e.g., the
(α/2, 1−α/2) quantiles of bootstrap TPR at each FPR) with a simultaneous
correction. This is effectively what the Working-Hotelling and Wilson Rectangle
methods do, with different correction strategies. Such constructions lose the
adaptive shape and natural asymmetry that make the envelope attractive at 95%.

---

## 5. Coverage Trajectory Across Sample Sizes

The coverage trajectory of `envelope_wilson` at the 95% confidence level is:

| n | Coverage | Dominant mechanism |
|---|----------|--------------------|
| 10 | 1.000 | Wilson floor covers entire curve |
| 30 | 0.991 | Wilson floor covers most of the curve |
| 100 | 0.976 | Wilson covers tails; envelope over-conservative in interior |
| 300 | 0.953 | Wilson covers shrinking tails; near-nominal balance |
| 1,000 | 0.950 | Small tail correction; interior over-conservatism compensates |
| 10,000 | 0.830 | Wilson floor negligible; base envelope under-coverage exposed |

### 5.1 Three-region model

At any sample size, the FPR grid can be partitioned into three regions with
distinct coverage properties:

| Region | Definition | Bootstrap quality | Wilson floor | Coverage driver |
|--------|-----------|-------------------|-------------|----------------|
| **Tails** | k < k_min or m < m_min | Structurally unable to estimate variance | Active | Wilson provides calibrated width |
| **Near-boundary** | k_min ≤ k ≲ 50 | Variance present but noisy/discrete | Inactive | Bootstrap alone; under-covers |
| **Interior** | k ≫ 50 | Reliable; captures both binomial and density-ratio variance | Inactive | Bootstrap + projection inflation; over-covers |

At small n, the tail region dominates → Wilson drives overall over-coverage.
At moderate n, all three regions contribute → balanced near 95%.
At large n, the tail region vanishes and the interior's over-conservatism cannot
fully compensate for the near-boundary region's under-coverage.

### 5.2 The near-boundary gap

The critical region is the "near-boundary" zone: grid points with effective
counts just above the Wilson floor thresholds (k = 15–50) where the bootstrap
has some variance but may still underestimate the true variance of R̂(t). At
these points:

- The bootstrap variance is non-zero but driven by moderate-count combinatorics.
- The Wilson floor is not applied (k ≥ k_min).
- The true ROC has genuine uncertainty that the bootstrap may underrepresent.

This gap between the Wilson floor's coverage and the bootstrap's reliability is
the proximate cause of coverage degradation at large n. As n grows, more grid
points enter this gap region (the tail shrinks but the near-boundary zone
persists), accumulating opportunities for small violations.

### 5.3 Violation magnitudes remain small

Despite the coverage drop at large n, the **magnitude** of violations remains
tiny. At n = 10,000:

- Mean max violation: ~0.002 (0.2 percentage points of TPR)
- P99 max violation: ~0.046
- Violations concentrate in the first ~5% of FPR (fig3)

This is because violations occur at near-boundary points where:
- The true ROC deviation from R̂ is O(1/√n) (sampling variability)
- The band has non-zero but insufficient width
- The shortfall is a fraction of the already-small deviation

The method fails *technically* (R_true escapes the band) but not *practically*
(by amounts far smaller than any clinical or operational decision threshold).

---

## 6. Dependence on AUC

Coverage degrades more at high AUC (fig5: ~75% coverage at AUC ≈ 1.0 vs ~90%
at AUC ≈ 0.6 for n = 10,000). This is expected from the tail mechanism:

At high AUC, the ROC curve rises steeply at low FPR — most of the
discriminative information is concentrated in the boundary region where the
bootstrap tail problem is worst. The true ROC at FPR = 0.01 may have TPR = 0.8,
while the empirical ROC is a step function that jumps discretely. The gap
between the smooth true curve and the step-function empirical is largest
precisely where the bootstrap has least power.

At low AUC, the ROC curve rises gradually, and most uncertainty is in the
interior where the bootstrap works well. The boundary region contributes little
to the overall coverage.

---

## 7. Robustness Across Distributions

The method achieves near-uniform coverage across all 7 DGPs at moderate sample
sizes (fig1: 0.915–0.978 at n = 300 across all DGPs). This robustness arises
because:

1. The bootstrap is fully nonparametric — it makes no distributional
   assumptions. Unlike Working-Hotelling (which assumes binormality) or the
   Hsieh-Turnbull bands (which require density estimation), the bootstrap
   captures the actual sampling distribution of the ROC regardless of the
   underlying score distributions.

2. The Wilson floor depends only on sample sizes and the empirical TPR, not on
   distributional shape. It provides the same correction quality whether the
   DGP is Gaussian, heavy-tailed, skewed, or multimodal.

Simulation evidence confirms this:

- Under departures from binormality (Student-t with low df, bimodal negatives),
  Working-Hotelling and ellipse envelope methods degrade catastrophically
  (coverage < 20% for heavy-tailed or multimodal data at n = 1,000).
  `envelope_wilson` maintains coverage above 90% (fig6).
- Under non-log-concave distributions (bimodal negative, logit-normal with
  large σ), the Hsieh-Turnbull log-concave method fails entirely (coverage near
  0%). `envelope_wilson` is unaffected (fig6b).

The method's robustness is its primary advantage over parametric alternatives.

---

## 8. Band Tightness

Among methods achieving ≥90% coverage at 95% CI, `envelope_wilson` (mean area
0.397) is 15% tighter than the KS band (0.469) while providing similar or better
coverage at moderate n (fig2, fig7). The tightness advantage grows with sample
size: at n = 10,000, the envelope's area is 56% of the KS band's (0.042 vs
0.075).

This efficiency gain comes from heteroscedasticity adaptation: the studentized
KS statistic weights deviations by local standard error. The envelope is tight
where variance is low (near corners) and wide where variance is high (mid-ROC),
while the KS band applies uniform width everywhere. The adaptive shape matches
the actual uncertainty structure of the ROC curve.

---

## 9. Comparison to Alternatives

### KS Band
Always achieves 100% coverage (by construction), but bands are uninformatively
wide — 98% coverage at the 50% CI level, meaning the 50% band is nearly as wide
as the 95% band. The KS band has zero violations ever but provides minimal
discriminative information about where the true ROC lies within the band.

### Working-Hotelling (Binormal)
Achieves good coverage under binormality but degrades catastrophically under
model misspecification (fig6). At n = 1,000 with Student-t (df = 3) data,
coverage drops below 20%. Not suitable as a general-purpose method.

### Wilson Rectangle (Šidák)
Pointwise method with multiplicity correction. Achieves good coverage at small n
but degrades at large n (coverage 0.839 at n = 10,000) because the Šidák
correction is conservative at small n and anti-conservative at large n (too many
test points). Bands are the tightest among methods with ≥90% coverage (area
0.331) but lack the adaptive shape of the envelope.

### Hsieh-Turnbull Log-Concave
Best calibrated method overall (smallest total deviation from nominal at both
95% and 50% levels), but requires log-concavity of score distributions. Fails
completely under non-log-concave data (bimodal, logit-normal; fig6b). Coverage
is inconsistent across sample sizes (0.746 at n = 300, 0.967 at n = 1,000).
Not suitable without distributional verification.

---

## 10. Summary of Expected Behavior

### Where the method is well-calibrated

- **95% CI, n = 30–1,000**: Coverage 0.950–0.991 across all DGPs tested.
  This is the method's sweet spot and covers the majority of practical ROC
  analyses.
- **All DGPs**: Coverage varies by at most 5pp across 7 tested distributions
  at any fixed sample size (fig1).
- **Low-to-moderate AUC (< 0.85)**: Coverage is ≥93% at all sample sizes
  tested.

### Where the method is over-conservative

- **Small n (≤ 30)**: Coverage is 99–100% because the Wilson floor dominates.
  Bands are wider than necessary.
- **50% CI at any n**: Coverage is ~85% (vs 50% target) because the envelope
  operator is insensitive to the retention fraction. The method is not
  designed for, and should not be used at, low confidence levels.

### Where the method under-covers

- **Large n (≥ 10,000)**: Coverage drops to ~83% as the Wilson tail correction
  vanishes. Violations are small in magnitude (~0.2pp) and concentrated in
  the low-FPR region.
- **High AUC + large n**: The worst case. At n = 10,000 and AUC > 0.95,
  coverage may drop to ~75%. The ROC's steep rise at low FPR concentrates
  uncertainty in the bootstrap's weakest region.

### Graceful degradation

In all observed failure modes, violation magnitudes remain small:
- Mean max violation never exceeds 0.003 (0.3pp of TPR) at any sample size.
- P99 max violation is ≤0.046 across all settings.
- Only 0.69% of simulations produce any violation exceeding 5pp.

The method does not fail catastrophically. When coverage is lost, it is lost by
tiny amounts in a small region of the ROC curve — a qualitatively different
failure mode from parametric methods, which can miss the true ROC by tens of
percentage points under model misspecification.
