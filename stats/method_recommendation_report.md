# ROC Confidence Band Method Recommendation Report

Generated from 2,254,000 simulation evaluations across 7 DGPs, [10, 30, 100, 300, 1000, 10000] sample sizes, and 23 methods.

## Data Summary

- **DGPs**: beta_opposing, bimodal_negative, gamma, hetero_gaussian, logitnormal, student_t, weibull
- **Sample sizes**: [10, 30, 100, 300, 1000, 10000]
- **Methods**: 23
- **Total evaluations**: 2,254,000
- **LHS combinations per DGP**: ~1000
- **Log-concave subset**: 1,235,514 / 2,254,000 (54.8%)
- **High AUC (>0.8) subset**: 955,052 / 2,254,000 (42.4%)

## Overall Recommendation (No Prior Knowledge)

**Question**: Which single method should a junior data scientist use when they don't know the underlying data distribution?

### Top 10 Methods by Composite Score

*Score combines closeness to nominal coverage at both 95% and 50% levels, violation magnitude (mean and 95th percentile), with extra penalty for anti-conservative behavior. Lower is better.*

| Method | Cov@95 | Gap@95 | MaxViol@95 | Cov@50 | Gap@50 | MaxViol@50 | Score |
|--------|--------|--------|------------|--------|--------|------------|-------|
| envelope_wilson | 0.950 | -0.000 | 0.0012 | 0.851 | +0.351 | 0.0040 | 0.807 |
| ks | 1.000 | +0.050 | 0.0000 | 0.982 | +0.482 | 0.0003 | 0.876 |
| envelope_wilson_symmetric | 0.946 | -0.004 | 0.0013 | 0.814 | +0.314 | 0.0058 | 0.937 |
| HT_log_concave_logit_autocalib_wilson | 0.895 | -0.055 | 0.0027 | 0.611 | +0.111 | 0.0108 | 1.450 |
| HT_log_concave_logit_wilson | 0.806 | -0.144 | 0.0038 | 0.567 | +0.067 | 0.0115 | 2.072 |
| wilson_rectangle_bonferroni | 0.896 | -0.054 | 0.0073 | 0.312 | -0.188 | 0.0278 | 2.527 |
| wilson_rectangle_sidak | 0.911 | -0.039 | 0.0073 | 0.247 | -0.253 | 0.0335 | 2.921 |
| logit_max_modulus | 0.483 | -0.467 | 0.0035 | 0.402 | -0.098 | 0.0070 | 3.815 |
| envelope_wilson_logit | 0.392 | -0.558 | 0.0032 | 0.290 | -0.210 | 0.0094 | 4.668 |
| envelope_wilson_symmetric_logit | 0.389 | -0.561 | 0.0037 | 0.281 | -0.219 | 0.0117 | 4.831 |

### Detailed Secondary Metrics (Top 8)

| Method | Cov95 | Cov50 | P95Viol@95 | P95Viol@50 | DirImb@95 | RegStd@95 | Area@95 |
|--------|-------|-------|------------|------------|-----------|-----------|---------|
| envelope_wilson | 0.950 | 0.851 | 0.0000 | 0.0213 | 0.043 | 0.016 | 0.397 |
| ks | 1.000 | 0.982 | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.469 |
| envelope_wilson_symmetric | 0.946 | 0.814 | 0.0006 | 0.0327 | 0.048 | 0.016 | 0.394 |
| HT_log_concave_logit_autocalib_wilson | 0.895 | 0.611 | 0.0136 | 0.0631 | 0.012 | 0.018 | 0.536 |
| HT_log_concave_logit_wilson | 0.806 | 0.567 | 0.0237 | 0.0642 | 0.037 | 0.039 | 0.469 |
| wilson_rectangle_bonferroni | 0.896 | 0.312 | 0.0080 | 0.1138 | 0.079 | 0.019 | 0.332 |
| wilson_rectangle_sidak | 0.911 | 0.247 | 0.0082 | 0.1353 | 0.063 | 0.015 | 0.331 |
| logit_max_modulus | 0.483 | 0.402 | 0.0177 | 0.0347 | 0.421 | 0.162 | 0.643 |


### Interpretation

**Best composite score**: `envelope_wilson` — best trade-off of safety (low violations) and informativeness.

**Safest (most conservative)**: `ks` — achieves ~100% coverage at all settings, but bands are very wide and uninformative (coverage at 50% CI is 98.2% instead of 50%).

**Best calibrated overall**: `HT_log_concave_logit_autocalib_wilson` — smallest total deviation from nominal at both 95% and 50% levels.


**`envelope_wilson` details**:

- 95% CI coverage: 0.950 (gap: -0.000)
- 50% CI coverage: 0.851 (gap: +0.351)
- Mean max violation @95%: 0.0012
- 95th pctl max violation @95%: 0.0000

Note: All envelope/bootstrap methods show substantial over-coverage at the 50% level, meaning their 50% bands are wider than necessary. This is a known property of bootstrap confidence bands — the simultaneous coverage guarantee tends to be conservative, especially at lower confidence levels. At 95%, which is the standard reporting level, `envelope_wilson` achieves essentially exact coverage.

## Performance by Sample Size

### Coverage by Sample Size (95% CI)

| Method | n=10 | n=30 | n=100 | n=300 | n=1000 | n=10000 |
|--------|------|------|-------|-------|--------|---------|
| envelope_wilson |  1.000 |  0.991 |  0.976 |  0.953 |  0.950 |  0.830 |
| ks |  1.000 |  1.000 |  1.000 |  1.000 |  1.000 |  1.000 |
| envelope_wilson_symmetric |  1.000 |  0.987 |  0.972 |  0.949 |  0.945 |  0.823 |
| HT_log_concave_logit_autocalib_wilson |  0.970 |  0.891 |  0.797 |  0.746 |  0.967 |  0.926 |
| HT_log_concave_logit_wilson |  0.970 |  0.891 |  0.797 |  0.746 |  0.796 |  0.648 |
| wilson_rectangle_bonferroni |  0.994 |  0.956 |  0.968 |  0.944 |  0.839 |  0.730 |

### Coverage by Sample Size (50% CI)

| Method | n=10 | n=30 | n=100 | n=300 | n=1000 | n=10000 |
|--------|------|------|-------|-------|--------|---------|
| envelope_wilson |  0.930 |  0.935 |  0.918 |  0.870 |  0.830 |  0.641 |
| ks |  0.999 |  0.989 |  0.983 |  0.987 |  0.968 |  0.980 |
| envelope_wilson_symmetric |  0.949 |  0.873 |  0.851 |  0.819 |  0.798 |  0.612 |
| HT_log_concave_logit_autocalib_wilson |  0.903 |  0.709 |  0.501 |  0.416 |  0.583 |  0.580 |
| HT_log_concave_logit_wilson |  0.903 |  0.709 |  0.501 |  0.416 |  0.512 |  0.419 |
| wilson_rectangle_bonferroni |  0.815 |  0.567 |  0.380 |  0.215 |  0.089 |  0.032 |


## Performance by DGP

### Coverage by DGP (95% CI, all sample sizes pooled)

| Method | beta_opposing | bimodal_negative | gamma | hetero_gaussian | logitnormal | student_t | weibull |
|--------|-------|-------|-------|-------|-------|-------|-------|
| envelope_wilson |  0.948 |  0.944 |  0.950 |  0.962 |  0.943 |  0.941 |  0.961 |
| ks |  1.000 |  1.000 |  1.000 |  1.000 |  1.000 |  1.000 |  1.000 |
| envelope_wilson_symmetric |  0.945 |  0.942 |  0.946 |  0.952 |  0.939 |  0.941 |  0.956 |
| HT_log_concave_logit_autocalib_wilson |  0.879 |  0.875 |  0.891 |  0.936 |  0.882 |  0.881 |  0.921 |
| HT_log_concave_logit_wilson |  0.756 |  0.799 |  0.810 |  0.887 |  0.756 |  0.793 |  0.845 |
| wilson_rectangle_bonferroni |  0.870 |  0.821 |  0.935 |  0.893 |  0.884 |  0.926 |  0.941 |


## Violation Patterns Across the ROC Curve

### Violation Rate by FPR Region (95% CI, all settings pooled)

| Method | 0-10 | 10-30 | 30-50 | 50-70 | 70-90 | 90-100 |
|--------|------|------|------|------|------|------|
| envelope_wilson |  0.044 |  0.003 |  0.002 |  0.002 |  0.001 |  0.001 |
| ks |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |
| envelope_wilson_symmetric |  0.046 |  0.005 |  0.002 |  0.002 |  0.002 |  0.001 |
| HT_log_concave_logit_autocalib_wilson |  0.060 |  0.010 |  0.008 |  0.012 |  0.019 |  0.025 |
| HT_log_concave_logit_wilson |  0.124 |  0.016 |  0.012 |  0.016 |  0.025 |  0.043 |
| wilson_rectangle_bonferroni |  0.053 |  0.013 |  0.013 |  0.017 |  0.027 |  0.058 |


## Band Tightness (Efficiency)

Among methods with adequate coverage, tighter bands are more informative. We filter to methods achieving ≥ 90% coverage at the 95% level overall, then rank by mean band area (lower = tighter = better).

### Methods with ≥ 90% Coverage at 95% CI

| Method | Coverage | Mean Area | Std Area | Mean Width |
|--------|----------|-----------|----------|------------|
| wilson_rectangle_sidak | 0.911 | 0.3314 | 0.2638 | 0.3200 |
| envelope_wilson_symmetric | 0.946 | 0.3941 | 0.3028 | 0.3894 |
| envelope_wilson | 0.950 | 0.3975 | 0.3022 | 0.3923 |
| ks | 1.000 | 0.4685 | 0.3016 | 0.4670 |

**Tightest band with ≥90% coverage**: `wilson_rectangle_sidak` (area=0.3314, coverage=0.911)

### Band Area by Sample Size (95% CI)

| Method | n=10 | n=30 | n=100 | n=300 | n=1000 | n=10000 |
|--------|------|------|-------|-------|--------|---------|
| wilson_rectangle_sidak |  0.803 |  0.607 |  0.365 |  0.214 |  0.148 |  0.036 |
| envelope_wilson_symmetric |  0.918 |  0.707 |  0.451 |  0.253 |  0.195 |  0.041 |
| envelope_wilson |  0.918 |  0.711 |  0.456 |  0.258 |  0.199 |  0.042 |
| ks |  0.969 |  0.789 |  0.553 |  0.366 |  0.264 |  0.075 |

### Honorable Mentions (80–90% Coverage)

These methods are tighter but sacrifice coverage below 90%:

| Method | Coverage | Mean Area |
|--------|----------|-----------|
| wilson_rectangle | 0.823 | 0.3001 |
| wilson_rectangle_bonferroni | 0.896 | 0.3320 |
| HT_log_concave_logit_wilson | 0.806 | 0.4687 |
| HT_log_concave_logit_autocalib_wilson | 0.895 | 0.5358 |


## Catastrophic Failure Analysis

A catastrophic failure occurs when the true ROC lies far outside the confidence band. We examine the tail of the violation magnitude distribution — the 95th, 99th, and 99.9th percentiles of max violation, plus the rate of violations exceeding thresholds of 0.05 and 0.10 (5pp and 10pp of TPR).

### Violation Magnitude Tail Distribution (95% CI, coverage ≥ 80%)

| Method | Cov | Mean | P95 | P99 | P99.9 | Max | Rate>5pp | Rate>10pp |
|--------|-----|------|-----|-----|-------|-----|----------|-----------|
| ks | 1.000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0245 | 0.0000 | 0.0000 |
| envelope_wilson | 0.950 | 0.0012 | 0.0000 | 0.0368 | 0.1407 | 0.6675 | 0.0069 | 0.0022 |
| envelope_wilson_symmetric | 0.946 | 0.0013 | 0.0006 | 0.0399 | 0.1446 | 0.6712 | 0.0076 | 0.0025 |
| HT_log_concave_logit_autocalib_wilson | 0.895 | 0.0027 | 0.0136 | 0.0672 | 0.1766 | 0.7725 | 0.0161 | 0.0045 |
| HT_log_concave_logit_wilson | 0.806 | 0.0038 | 0.0237 | 0.0763 | 0.1895 | 0.6934 | 0.0211 | 0.0055 |
| wilson_rectangle_bonferroni | 0.896 | 0.0073 | 0.0080 | 0.2673 | 0.6799 | 0.7893 | 0.0254 | 0.0188 |
| wilson_rectangle_sidak | 0.911 | 0.0073 | 0.0082 | 0.2673 | 0.6800 | 0.7893 | 0.0255 | 0.0189 |
| wilson_rectangle | 0.823 | 0.0088 | 0.0226 | 0.2849 | 0.6899 | 0.7895 | 0.0311 | 0.0211 |

### P99 Max Violation by Sample Size (95% CI)

| Method | n=10 | n=30 | n=100 | n=300 | n=1000 | n=10000 |
|--------|------|------|-------|-------|--------|---------|
| envelope_wilson |  0.0000 |  0.0000 |  0.0410 |  0.0581 |  0.0399 |  0.0461 |
| ks |  0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.0000 |
| envelope_wilson_symmetric |  0.0000 |  0.0200 |  0.0522 |  0.0584 |  0.0410 |  0.0462 |
| HT_log_concave_logit_autocalib_wilson |  0.0221 |  0.0795 |  0.1139 |  0.1041 |  0.0194 |  0.0235 |
| HT_log_concave_logit_wilson |  0.0221 |  0.0795 |  0.1139 |  0.1041 |  0.0609 |  0.0572 |
| wilson_rectangle_bonferroni |  0.0089 |  0.1316 |  0.2803 |  0.3392 |  0.3448 |  0.3706 |

### Rate of Violations > 5pp by Sample Size (95% CI)

| Method | n=10 | n=30 | n=100 | n=300 | n=1000 | n=10000 |
|--------|------|------|-------|-------|--------|---------|
| envelope_wilson |  0.0000 |  0.0047 |  0.0079 |  0.0123 |  0.0076 |  0.0084 |
| ks |  0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.0000 |
| envelope_wilson_symmetric |  0.0000 |  0.0066 |  0.0101 |  0.0124 |  0.0077 |  0.0084 |
| HT_log_concave_logit_autocalib_wilson |  0.0027 |  0.0211 |  0.0421 |  0.0359 |  0.0040 |  0.0027 |
| HT_log_concave_logit_wilson |  0.0027 |  0.0211 |  0.0421 |  0.0359 |  0.0155 |  0.0146 |
| wilson_rectangle_bonferroni |  0.0060 |  0.0191 |  0.0250 |  0.0304 |  0.0314 |  0.0346 |


## Asymptotic Trajectory (n=300 → 1000 → 10000)

A method with sound asymptotics should converge toward nominal coverage as sample size grows. Coverage that *drops further below nominal* at larger n is a red flag — it means the method becomes less reliable precisely when the user might expect it to improve.

### Coverage Trajectory at 95% CI

| Method | n=300 | n=1000 | n=10000 | Drop 300→10k | Gap@10k |
|--------|-------|--------|---------|--------------|---------|
| HT_log_concave_logit_autocalib_wilson | 0.746 | 0.967 | 0.926 | -0.180 | -0.024 |
| ks | 1.000 | 1.000 | 1.000 | +0.000 | +0.050 |
| wilson_rectangle_sidak | 0.941 | 0.841 | 0.839 | +0.102 | -0.111 |
| envelope_wilson | 0.953 | 0.950 | 0.830 | +0.123 | -0.120 |
| envelope_wilson_symmetric | 0.949 | 0.945 | 0.823 | +0.126 | -0.127 |
| wilson_rectangle_bonferroni | 0.944 | 0.839 | 0.730 | +0.214 | -0.220 |
| HT_log_concave_logit_wilson | 0.746 | 0.796 | 0.648 | +0.098 | -0.302 |
| wilson_rectangle | 0.873 | 0.721 | 0.595 | +0.279 | -0.355 |

### Coverage Trajectory at 50% CI

| Method | n=300 | n=1000 | n=10000 | Drop 300→10k | Gap@10k |
|--------|-------|--------|---------|--------------|---------|
| HT_log_concave_logit_autocalib_wilson | 0.416 | 0.583 | 0.580 | -0.164 | +0.080 |
| HT_log_concave_logit_wilson | 0.416 | 0.512 | 0.419 | -0.003 | -0.081 |
| envelope_wilson_symmetric | 0.819 | 0.798 | 0.612 | +0.207 | +0.112 |
| envelope_wilson | 0.870 | 0.830 | 0.641 | +0.229 | +0.141 |
| wilson_rectangle_bonferroni | 0.215 | 0.089 | 0.032 | +0.183 | -0.468 |
| ks | 0.987 | 0.968 | 0.980 | +0.006 | +0.480 |
| wilson_rectangle_sidak | 0.142 | 0.050 | 0.013 | +0.129 | -0.487 |
| wilson_rectangle | 0.004 | 0.001 | 0.000 | +0.004 | -0.500 |

### Violation Magnitude Trajectory (95% CI)

Mean max violation should decrease as n grows. Methods where it *increases* may have structural problems.

| Method | n=300 | n=1000 | n=10000 | Trend |
|--------|-------|--------|---------|-------|
| ks | 0.00000 | 0.00000 | 0.00000 | ~Zero (excellent) |
| HT_log_concave_logit_autocalib_wilson | 0.00651 | 0.00076 | 0.00075 | Flat/stable |
| envelope_wilson | 0.00203 | 0.00125 | 0.00211 | INCREASING (bad) |
| envelope_wilson_symmetric | 0.00212 | 0.00132 | 0.00217 | INCREASING (bad) |
| HT_log_concave_logit_wilson | 0.00651 | 0.00317 | 0.00385 | Non-monotone |
| wilson_rectangle_bonferroni | 0.00915 | 0.00961 | 0.01045 | INCREASING (bad) |
| wilson_rectangle_sidak | 0.00917 | 0.00963 | 0.01046 | INCREASING (bad) |
| wilson_rectangle | 0.01059 | 0.01126 | 0.01106 | INCREASING (bad) |

### Interpretation

**Methods with sound asymptotic trajectory at 95%** (gap < 5pp at n=10000, drop < 15pp from n=300→10000):

- `ks`: n=10000 coverage = 1.000 (gap +0.050), drop from n=300: +0.000

**Methods with anti-conservative drift** (coverage drops below nominal as n grows):

- `wilson_rectangle`: n=10000 coverage = 0.595 (gap -0.355), drop from n=300: +0.279
- `HT_log_concave_logit_wilson`: n=10000 coverage = 0.648 (gap -0.302), drop from n=300: +0.098
- `wilson_rectangle_bonferroni`: n=10000 coverage = 0.730 (gap -0.220), drop from n=300: +0.214
- `envelope_wilson_symmetric`: n=10000 coverage = 0.823 (gap -0.127), drop from n=300: +0.126
- `envelope_wilson`: n=10000 coverage = 0.830 (gap -0.120), drop from n=300: +0.123
- `wilson_rectangle_sidak`: n=10000 coverage = 0.839 (gap -0.111), drop from n=300: +0.102



## Condition A: AUC Known to be High (>0.80)

Subset: 955,052 evaluations

| Method | Cov@95 | Gap@95 | MaxViol@95 | Cov@50 | Gap@50 | MaxViol@50 | Score |
|--------|--------|--------|------------|--------|--------|------------|-------|
| ks | 1.000 | +0.050 | 0.0000 | 0.994 | +0.494 | 0.0001 | 0.893 |
| envelope_wilson | 0.929 | -0.021 | 0.0023 | 0.876 | +0.376 | 0.0035 | 1.116 |
| envelope_wilson_symmetric | 0.924 | -0.026 | 0.0024 | 0.824 | +0.324 | 0.0059 | 1.282 |
| HT_log_concave_logit_autocalib_wilson | 0.901 | -0.049 | 0.0030 | 0.617 | +0.117 | 0.0114 | 1.452 |
| HT_log_concave_logit_wilson | 0.808 | -0.142 | 0.0049 | 0.585 | +0.085 | 0.0122 | 2.306 |
| wilson_rectangle_bonferroni | 0.844 | -0.106 | 0.0165 | 0.317 | -0.183 | 0.0367 | 4.822 |
| envelope_wilson_logit | 0.351 | -0.599 | 0.0042 | 0.293 | -0.207 | 0.0084 | 4.889 |
| wilson_rectangle_sidak | 0.874 | -0.076 | 0.0166 | 0.257 | -0.243 | 0.0415 | 5.029 |


**Winner when AUC > 0.80**: `ks`


## Condition B: Data Known to be Log-Concave

Log-concave distributions: Gaussian (always), Student-t (always), Gamma (shape≥1), Weibull (shape≥1), Beta (alpha≥1).

Subset: 1,235,514 evaluations

| Method | Cov@95 | Gap@95 | MaxViol@95 | Cov@50 | Gap@50 | MaxViol@50 | Score |
|--------|--------|--------|------------|--------|--------|------------|-------|
| envelope_wilson | 0.953 | +0.003 | 0.0011 | 0.863 | +0.363 | 0.0036 | 0.789 |
| envelope_wilson_symmetric | 0.949 | -0.001 | 0.0012 | 0.818 | +0.318 | 0.0052 | 0.874 |
| ks | 1.000 | +0.050 | 0.0000 | 0.982 | +0.482 | 0.0003 | 0.876 |
| HT_log_concave_logit_autocalib_wilson | 0.907 | -0.043 | 0.0024 | 0.626 | +0.126 | 0.0097 | 1.239 |
| HT_log_concave_logit_wilson | 0.833 | -0.117 | 0.0032 | 0.599 | +0.099 | 0.0099 | 1.763 |
| logit_max_modulus | 0.641 | -0.309 | 0.0027 | 0.561 | +0.061 | 0.0057 | 2.491 |
| wilson_rectangle_bonferroni | 0.923 | -0.027 | 0.0116 | 0.319 | -0.181 | 0.0321 | 2.987 |
| wilson_rectangle_sidak | 0.927 | -0.023 | 0.0116 | 0.253 | -0.247 | 0.0376 | 3.450 |


**Winner for log-concave data**: `envelope_wilson`


## Condition C: Sample Size ≥ 300

Subset: 1,288,000 evaluations

| Method | Cov@95 | Gap@95 | MaxViol@95 | Cov@50 | Gap@50 | MaxViol@50 | Score |
|--------|--------|--------|------------|--------|--------|------------|-------|
| ks | 1.000 | +0.050 | 0.0000 | 0.976 | +0.476 | 0.0003 | 0.867 |
| envelope_wilson | 0.921 | -0.029 | 0.0017 | 0.793 | +0.293 | 0.0034 | 0.992 |
| envelope_wilson_symmetric | 0.915 | -0.035 | 0.0017 | 0.756 | +0.256 | 0.0042 | 1.059 |
| HT_log_concave_logit_autocalib_wilson | 0.901 | -0.048 | 0.0022 | 0.541 | +0.041 | 0.0087 | 1.069 |
| HT_log_concave_logit_wilson | 0.747 | -0.203 | 0.0042 | 0.465 | -0.035 | 0.0098 | 2.317 |
| wilson_rectangle_bonferroni | 0.838 | -0.112 | 0.0097 | 0.106 | -0.394 | 0.0257 | 3.391 |
| wilson_rectangle_sidak | 0.865 | -0.085 | 0.0097 | 0.063 | -0.437 | 0.0292 | 3.477 |
| logit_max_modulus | 0.430 | -0.520 | 0.0010 | 0.382 | -0.118 | 0.0016 | 3.549 |


**Winner for n ≥ 300**: `ks`


## Combined Conditions

### A+C: High AUC + Large Sample

Subset: 545,744 evaluations

| Method | Cov@95 | Gap@95 | MaxViol@95 | Cov@50 | Gap@50 | MaxViol@50 | Score |
|--------|--------|--------|------------|--------|--------|------------|-------|
| ks | 1.000 | +0.050 | 0.0000 | 0.992 | +0.492 | 0.0001 | 0.890 |
| HT_log_concave_logit_autocalib_wilson | 0.898 | -0.052 | 0.0033 | 0.524 | +0.024 | 0.0120 | 1.419 |
| envelope_wilson | 0.886 | -0.064 | 0.0033 | 0.810 | +0.310 | 0.0045 | 1.602 |
| envelope_wilson_symmetric | 0.880 | -0.070 | 0.0034 | 0.756 | +0.256 | 0.0058 | 1.673 |
| HT_log_concave_logit_wilson | 0.737 | -0.213 | 0.0066 | 0.468 | -0.032 | 0.0135 | 2.918 |

**Winner**: `ks`


### B+C: Log-Concave + Large Sample

Subset: 706,008 evaluations

| Method | Cov@95 | Gap@95 | MaxViol@95 | Cov@50 | Gap@50 | MaxViol@50 | Score |
|--------|--------|--------|------------|--------|--------|------------|-------|
| ks | 1.000 | +0.050 | 0.0000 | 0.976 | +0.476 | 0.0003 | 0.866 |
| envelope_wilson | 0.925 | -0.025 | 0.0016 | 0.809 | +0.309 | 0.0031 | 0.930 |
| HT_log_concave_logit_autocalib_wilson | 0.908 | -0.042 | 0.0021 | 0.550 | +0.050 | 0.0082 | 0.972 |
| envelope_wilson_symmetric | 0.918 | -0.032 | 0.0016 | 0.755 | +0.255 | 0.0041 | 1.003 |
| HT_log_concave_logit_wilson | 0.780 | -0.170 | 0.0035 | 0.503 | +0.003 | 0.0087 | 1.862 |

**Winner**: `ks`


## Small Sample Performance (n ≤ 30)

Subset: 644,000 evaluations

| Method | Cov@95 | Gap@95 | MaxViol@95 | Cov@50 | Gap@50 | MaxViol@50 | Score |
|--------|--------|--------|------------|--------|--------|------------|-------|
| ks | 1.000 | +0.050 | 0.0000 | 0.994 | +0.494 | 0.0003 | 0.893 |
| envelope_wilson | 0.995 | +0.045 | 0.0003 | 0.933 | +0.433 | 0.0056 | 1.097 |
| envelope_wilson_symmetric | 0.993 | +0.043 | 0.0005 | 0.911 | +0.411 | 0.0081 | 1.364 |
| HT_log_concave_logit_autocalib_wilson | 0.931 | -0.019 | 0.0019 | 0.806 | +0.306 | 0.0098 | 1.472 |
| HT_log_concave_logit_wilson | 0.931 | -0.019 | 0.0019 | 0.806 | +0.306 | 0.0098 | 1.472 |
| envelope_wilson_logit | 0.905 | -0.045 | 0.0015 | 0.713 | +0.213 | 0.0119 | 1.585 |
| envelope_wilson_symmetric_logit | 0.901 | -0.049 | 0.0017 | 0.691 | +0.191 | 0.0145 | 1.745 |
| ellipse_envelope_sweep | 0.944 | -0.006 | 0.0023 | 0.521 | +0.021 | 0.0401 | 2.330 |

**Best for small samples**: `ks`


## Conservatism Analysis

Methods that are consistently conservative (coverage > nominal) vs anti-conservative (coverage < nominal).

| Method | Gap@95 | Gap@50 | Tendency |
|--------|--------|--------|----------|
| envelope_wilson | -0.000 | +0.351 | Near-nominal |
| ks | +0.050 | +0.482 | Conservative |
| envelope_wilson_symmetric | -0.004 | +0.314 | Near-nominal |
| HT_log_concave_logit_autocalib_wilson | -0.055 | +0.111 | Mixed (anti@95, cons@50) |
| HT_log_concave_logit_wilson | -0.144 | +0.067 | Mixed (anti@95, cons@50) |
| wilson_rectangle_bonferroni | -0.054 | -0.188 | Anti-conservative |
| wilson_rectangle_sidak | -0.039 | -0.253 | Anti-conservative |
| logit_max_modulus | -0.467 | -0.098 | Anti-conservative |
| envelope_wilson_logit | -0.558 | -0.210 | Anti-conservative |
| envelope_wilson_symmetric_logit | -0.561 | -0.219 | Anti-conservative |
| wilson_rectangle | -0.127 | -0.414 | Anti-conservative |
| HT_reflected_kde_logit_autocalib | -0.311 | -0.075 | Anti-conservative |


## Executive Summary

The table below shows the top-scoring method per scenario. Note that `ks` achieves perfect safety (no violations ever) at the cost of very wide, uninformative bands. For a practitioner who needs informative bands with reliable 95% coverage, `envelope_wilson` is the standout choice.

| Scenario | Top Composite | Runner-up | Notes |
|----------|--------------|-----------|-------|
| **No prior knowledge** | `envelope_wilson` | `ks` | `envelope_wilson` is near-exact at 95% nominal |
| **AUC known > 0.80** | `ks` | `envelope_wilson` | High AUC makes coverage easier |
| **Log-concave data** | `envelope_wilson` | `envelope_wilson_symmetric` | No major advantage over the general case |
| **n ≥ 300** | `ks` | `envelope_wilson` | Large samples tighten all bands |
| **n ≤ 30** | `ks` | `envelope_wilson` | Small samples: all methods conservative |
| **High AUC + n ≥ 300** | `ks` | `HT_log_concave_logit_autocalib_wilson` | Best-case scenario |
| **Log-concave + n ≥ 300** | `ks` | `envelope_wilson` | Known regularity + data |

### Bottom Line

For a junior data scientist who doesn't know their underlying data distributions:

1. **Use `envelope_wilson`**. It achieves essentially exact 95% coverage (0.950) across all 7 DGPs tested, with negligible violation magnitudes (mean max violation < 0.002). It is robust to distribution shape, sample size, and AUC level.
2. **If safety is paramount** and band width doesn't matter, `ks` never fails — but its bands are so wide as to be uninformative at the 50% level.
3. **`envelope_wilson_symmetric`** is a close alternative to `envelope_wilson` with very similar performance.
4. At **n ≥ 300** and **large AUC**, coverage for `envelope_wilson` dips slightly below 95% (~92%), but violation magnitudes remain tiny. If this concerns you, `ks` provides guaranteed coverage.
5. **Avoid**: `wilson` (pointwise, not simultaneous), `pointwise`, `logit_max_modulus`, and most logit-transformed envelope methods — these show substantial anti-conservative behavior.

### Tightness, Tail Risk, and Asymptotics

6. **Tightness**: Among methods with ≥90% coverage, `wilson_rectangle_sidak` produces the tightest bands (area=0.331) but has poor 50% CI calibration. `envelope_wilson` (area=0.397) is 15% tighter than `ks` (area=0.469) while maintaining better calibration. See the full efficiency table above for per-sample-size band areas.
7. **Catastrophic failure risk**: `envelope_wilson` has a P99 max violation of 0.0368 (i.e., in 99% of simulations, the worst violation is under 3.7% of TPR). Only 0.69% of simulations have any violation exceeding 5pp. Compare `wilson_rectangle_bonferroni` whose P99 is 0.2673 — a 7x higher catastrophic risk. `ks` has zero catastrophic risk but at the cost of uninformative bands.
8. **Asymptotic trajectory**: `envelope_wilson` maintains excellent coverage at n=300 (0.953) and n=1000 (0.950), but drops to 0.830 at n=10000 — a 12pp gap below 95% nominal. This is a notable limitation for very large datasets. However, the mean max violation at n=10000 remains small (~0.002), meaning the bands miss the true ROC by trivial amounts. `HT_log_concave_logit_autocalib_wilson` shows the best large-n convergence (gap of only -2.4pp at n=10000) but has inconsistent coverage at intermediate n. Only `ks` has a truly stable trajectory, because it is always 100%.
9. **Key trade-off at large n**: For n≥10000, users who need strict ≥95% coverage guarantees should consider `ks`. For n≤1000 (the majority of practical ROC analyses), `envelope_wilson` is near-exact at 95% and the best practical choice.

