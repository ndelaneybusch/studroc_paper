# Theoretical Behavior of the Studentized Bootstrap Envelope with Wilson Floor

This report provides a theoretical account of the expected behavior of
`envelope_wilson` — the studentized bootstrap envelope SCB for ROC curves with
adaptive Wilson variance floor and exact Beta order-statistic floor — across
confidence levels, sample sizes, AUC ranges, and distributional assumptions. It draws on the method specification
(`nonparam_envelope.md`), implementation (`envelope_boot.py`), and the simulation
study (`method_recommendation_report.md`, 2,254,000 evaluations across 7 DGPs).

**Revision note (2026-06-09):** Updated to reflect the June 2026 follow-up
experiments documented in `project_evaluation_report.md` (sections D.8, D.9):
the paired old/new implementation comparison, the Monte Carlo variance
decomposition at low-FPR grid points, and the failure-point microscopy. The
main corrections: the failure zone is the first few grid points (k = 1–10),
not k = 15–50; the binding mechanism there is one-sided bootstrap support
collapse plus upward bias of the empirical ROC, not symmetric variance
underestimation; the n=1000 coverage figure of 0.950 was a prevalence-pooling
artifact (0.915 at prevalence 50%); and the Wilson floor's success region (the
TPR plateau) is disjoint from the failure region (the steep low-FPR corner).

**Revision note (2026-06-10):** The exact Beta order-statistic floor is now
integrated into `envelope_boot.py` as part of `boundary_method="wilson"`. It
repairs the first-k fragile zone described in §5.2: on the problem-domain
strata that previously under-covered, coverage rises to 0.95–0.99 and >5pp
misses vanish (§3.4, §5.4). Coverage figures quoted from the stored 2.25M
simulation suite describe the band *without* this floor.

---

## 1. Method Overview

The method constructs simultaneous confidence bands by:

1. **Bootstrapping**: Generate B stratified bootstrap ROC curves.
2. **Studentizing**: Normalize each bootstrap curve's pointwise deviation from
   the empirical ROC by the bootstrap standard deviation at each FPR grid point
   (floored by the Wilson variance to keep studentization stable).
3. **Retaining**: Keep the (1−α) fraction of curves with the smallest
   supremum studentized deviation (KS statistic).
4. **Enveloping**: Take the pointwise min/max of retained curves.
5. **Wilson floor**: Where the bootstrap variance falls below the Wilson
   variance (deficiency `max(0, 1 − σ²_boot/σ²_Wilson) > 0`), apply
   Šidák-corrected Wilson Rectangle bounds as a floor on both band sides,
   with the Šidák strength set by the continuous effective dimensionality
   K_eff = Σ deficiency; then enforce band monotonicity. (The earlier
   implementation used fixed count cutoffs k < 15, m < 10 and applied the
   floor one-sided per tail; a 1,400-case paired comparison shows the two
   are behaviorally equivalent — see §3.1 and `project_evaluation_report.md`
   D.8.)
6. **Beta order-statistic floor**: Floor the lower band at extreme FPR using
   the exact, distribution-free law of the true FPR at the j-th largest
   negative score, F̄(X₍ⱼ₎) ~ Beta(j, n₀+1−j), for j = 1..25, each event
   paired with a one-sided Wilson bound on the empirical TPR at that order
   statistic (per-event alpha = α/50). Vacuous below the first Beta quantile
   (≈ 7/n₀); jurisdiction ends ≈ 43/n₀ (§3.4).

The method is a **hybrid** of three distinct uncertainty quantification
strategies: a bootstrap envelope in the interior of the ROC curve, a
parametric (binomial) correction at the TPR plateau, and an exact
order-statistic bound at the steep low-FPR corner. Understanding its behavior
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

**The collapse is one-sided at the first grid points, not total.** Measured
against Monte Carlo ground truth on violating high-AUC cases at n = 10,000
(`project_evaluation_report.md` D.9): at k = 1–3 the pointwise bootstrap sd is
roughly correct (0.6–1.0× truth), but the bootstrap deviation distribution is
asymmetric. The resampled maximum negative can never exceed the observed
maximum, so bootstrap curves deviate freely *upward* (when extreme negatives
are dropped from the resample) but barely *downward*. The lower envelope arm
reaches only ~0.8–1.4 sd below the empirical curve at k = 1–3, versus ~3.5 sd
in the interior, while the upper arm stays at ~2.6–2.8 sd. A symmetric
variance summary cannot detect this — the variance looks healthy while the
lower tail of the deviation distribution is missing.

A formally analogous problem occurs at FPR near 1, but its *consequence* is
different: there the ROC sits on its TPR plateau (TPR ≈ 1 for both estimate
and truth over a wide FPR range), the slope is ~0 so threshold error converts
to ~zero vertical error, and binomial TPR uncertainty is the complete
uncertainty model. That corner is fully repairable with a binomial (Wilson)
floor; the steep low-FPR corner is not (§3.2) — it requires the
order-statistic floor of §3.4.

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

The geography of the bare envelope's failures is informative: at n = 10,000,
`envelope_standard` violates in the FPR 90–100% region in 55% of simulations
and in 70–90% in 9%, versus 17% in 0–10%. The TPR plateau — where every
bootstrap resample gives TPR ≈ 1 and the envelope width is exactly zero — is
the dominant failure of the *uncorrected* method. The Wilson floor eliminates
it almost entirely (90–100% rate drops to 0.6%), which is why the floored
method's residual failures are concentrated at the opposite (steep) corner.

---

## 3. The Wilson Floor: Mechanism and Scaling

### 3.1 Two-stage correction

The implementation applies Wilson-based corrections at two stages:

**Stage 1 — Variance floor during studentization**: The bootstrap variance is
floored to at least the Wilson score variance
σ²_Wilson(p) = [p(1−p)/n₁ + z²/(4n₁²)] / (1 + z²/n₁)². This prevents
studentized statistics from blowing up at zero-variance points, ensuring the KS
retention criterion is well-behaved everywhere.

**Stage 2 — Wilson Rectangle floor on the envelope**: After envelope
construction, Šidák-corrected Wilson Rectangle bounds are applied as a floor.
The two generations of the implementation differ here, in instructive ways:

- *Hard-cutoff version* (behind the stored simulation suite): tail regions
  defined by fixed effective-count thresholds (k < 15 negatives above
  threshold, or m < 10 positives), with the floor applied **one-sided per
  tail** — in the lower tail (FPR ≈ 0) only the *upper* bound is raised; in
  the upper tail (FPR ≈ 1) only the *lower* bound is dropped. The implicit
  assumption is mirror symmetry: that the (0,0) corner pin protects the lower
  band near the origin the way the (1,1) pin protects the upper band near the
  end.
- *Variance-ratio version* (current code): deficiency weights
  `max(0, 1 − σ²_boot/σ²_Wilson)` decide where the floor applies, the Šidák
  correction strength uses the continuous effective dimensionality
  K_eff = Σ deficiency, and the floor is applied to **both** band sides at
  deficient points, followed by monotonicity enforcement.

**The two versions are behaviorally equivalent** (1,400-case paired comparison
on identical data and bootstrap matrices: coverage within 0.3pp in every
n × AUC stratum, 4/1400 discordant outcomes). The equivalence has a precise
mechanism: at the points where coverage actually fails (k = 1–3, mid-range TPR
on a steep curve), the bootstrap variance *exceeds* the Wilson variance by a
factor of 10–60, so the variance-ratio gate never fires there; and the
hard-cutoff version's gate did fire (k < 15) but only raised the upper bound.
Either way, the lower band at the failure points is the bare envelope arm —
verified bit-identical between versions at k = 1–25 on violating cases.

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

**The yardstick caveat.** "Where the binomial model undercuts the bootstrap"
and "where coverage fails" turn out to be different places. Measured against
Monte Carlo truth at the failure points (steep curve, k = 1–5): the Wilson sd
is only 0.1–0.3 of the true sd (the slope term dominates and Wilson omits it),
while the bootstrap sd is 0.5–1.3× truth. The binomial floor is therefore the
*complete* variance model at the TPR plateau (slope ≈ 0) and the wrong
yardstick — by a factor of 3–6 in sd — at the steep corner. This single fact
explains both why the floor works so well at the upper-right corner and why no
Wilson-referenced gate can engage where the method actually fails.

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
consequence of this scaling — with one refinement from the follow-up
experiments: what disengages at large n is not just the floor's *extent* but
its *relevance*. The corrected region (where the floor binds) and the failure
region (the first few grid points of a steep curve) become disjoint: the floor
keeps firing at the TPR plateau, where it is no longer needed, while the
fragile first-k points fall outside every Wilson-referenced criterion.

### 3.4 The Beta order-statistic floor: an exact bound on the channel the others cannot see

The Wilson corrections measure *vertical* (binomial TPR) uncertainty. At the
first grid points of a steep curve the dominant uncertainty is *horizontal* —
the true FPR of the operating threshold, which is an extreme order statistic
of the negatives — and no vertical-variance yardstick can detect or repair
it (§3.2). The integrated Beta floor carries this channel directly, using
the one exact, finite-sample, distribution-free law available in the
problem: for continuous scores, the true FPR exceedance at the j-th largest
negative score is F̄(X₍ⱼ₎) ~ Beta(j, n₀+1−j), regardless of the score
distribution (probability integral transform).

The construction: let q_j be the (1−α_e) upper Beta quantile, with
α_e = α/(2J) Bonferroni over the 2J one-sided events (J = 25). On the event
{F̄(X₍ⱼ₎) ≤ q_j}, monotonicity of the true ROC gives, for every evaluation
point t ≥ q_j: R_true(t) ≥ G(X₍ⱼ₎) ≥ WilsonLower(R̂(j/n₀)). The lower band
at t is floored by the bound from the largest j with q_j ≤ t — a
backward-looking anchor at a smaller-FPR operating point whose true FPR
provably (whp) sits at or below t. Below q₁ (≈ 6.9/n₀ at α_e = 0.001) the
floor is vacuous: no distribution-free lower bound exists there, and the
band honestly reports ~0 rather than the unsupportable claims that produced
the pre-floor violations.

Theoretical properties worth noting:

1. **Exactness where asymptotics fail.** The Beta law is finite-sample and
   assumption-free (beyond continuity), so the floor's guarantee holds at
   precisely the moving boundary points t = k/n₀ where the empirical-process
   and bootstrap-consistency arguments break down.
2. **One-sided by construction.** It widens only the lower arm at the steep
   corner — the measured deficit — rather than symmetrically inflating the
   band.
3. **Bias-aware.** The empirical ROC's upward bias at k = 1–3 is the same
   order-statistic geometry; anchoring at R̂(j/n₀) for larger j (a far
   better-estimated quantity) absorbs it.
4. **Self-deactivating; no gate.** On flat ROC segments R̂(j/n₀) ≈ R̂(t) and
   the floor costs nothing; on steep segments it widens in proportion to the
   realized slope. The geometry is the gate — avoiding the Wilson floor's
   gate problem (§3.2) entirely.
5. **Conservative under ties.** With score atoms the exceedance is
   stochastically smaller than the Beta law, so discreteness errs safe
   (verified down to 20 score levels).
6. **Fixed-k jurisdiction.** Like the count-based Wilson tail, the
   jurisdiction (k ≤ ~43) is a vanishing *fraction* of the grid as n grows —
   but unlike the Wilson floor, that is exactly where the failures stay
   (§5.2), so the floor and the failure set remain aligned at every n.

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
levels.

Three refinements to this account, from the follow-up experiments
(`project_evaluation_report.md` B.4a, D.2, G.1):

1. **The weak α-sensitivity is a property of sup-norm simultaneity, not of
   the envelope operator specifically.** The critical-value ratio
   c₀.₉₅/c₀.₅₀ for a supremum statistic over many correlated grid points is
   inherently modest (~1.3–1.5×). The G.1 experiment replaced the envelope
   with a variance-model band (R̂ ± c·σ̂, bootstrap-calibrated c) and 50%
   calibration did *not* improve — in most scenarios it worsened. Abandoning
   the envelope does not buy tunability unless the variance estimate is
   smooth (HT-autocalib achieves tunability via a smooth analytical variance
   and a single scalable critical value).

2. **The 50% over-coverage diminishes with n** (coverage at the 50% level:
   0.93 at n = 10 → 0.64 at n = 10,000 at prevalence 50%). It is a
   finite-sample compound of the Wilson floor (dominant at small n),
   bootstrap step-function conservatism, and the sup-norm effect — not a
   fixed property.

3. **The 50% failures are globally distributed**, unlike the 95% failures: at
   α = 0.5 violations occur in every FPR region (4–6% in each interior
   region at n = 10,000, 24% at 0–10%), and the AUC gradient reverses at
   moderate n. The 50% problem is a global calibration problem; the 95%
   problem is a localized boundary problem. They need different fixes.

---

## 5. Coverage Trajectory Across Sample Sizes

The coverage trajectory of `envelope_wilson` at the 95% confidence level,
at prevalence 50% (balanced classes), *before* the integrated Beta floor
(§5.4), is:

| n | Coverage (prev 50%) | Dominant mechanism |
|---|----------|--------------------|
| 10 | 1.000 | Wilson floor covers entire curve |
| 30 | 0.991 | Wilson floor covers most of the curve |
| 100 | 0.976 | Wilson covers tails; envelope over-conservative in interior |
| 300 | 0.953 | Wilson covers shrinking tails; near-nominal balance |
| 1,000 | 0.915 | Already below nominal; first-k failures emerging |
| 10,000 | 0.830 | Floor disengaged at failure points; one-sided tail collapse exposed |

**Prevalence caveat.** Earlier summaries reported 0.950 at n = 1,000; that
figure pooled the prevalence-10% configuration (coverage 0.985 — only 100
positives, hence a wide Wilson floor) with the prevalence-50% configuration
(0.915), and the average happened to land on nominal. Coverage crosses nominal
between n = 300 and n = 1,000; there is no n = 1,000 sweet spot. A corollary:
class imbalance makes the method *more* conservative (the floor scales with
1/n₁), so balanced classes at large n are the risk configuration, not
imbalance.

### 5.1 Two-corner model (replacing the earlier three-region model)

An earlier version of this analysis partitioned the grid into tails (floor
active), a near-boundary zone (k = 15–50, hypothesized under-coverage), and
interior. Direct measurement falsified the near-boundary localization: the
failures live at **k = 1–10**, with the bulk at k = 1–3 (median violation FPR
~0.00015 at n = 10,000; 87.5% of violations below FPR 0.001). The corrected
partition:

| Region | Definition | What dominates uncertainty | Wilson floor | Outcome |
|--------|-----------|---------------------------|-------------|---------|
| **TPR plateau** (upper-right) | empirical TPR ≈ 1 (and the FPR ≈ 1 corner) | Binomial TPR variance (slope ≈ 0) | Active — and the *complete* model here | Fully repaired (90–100% region violation rate 0.6%) |
| **Steep corner, first-k points** | k = 1–10 negatives above threshold on a steep curve | Threshold location (extreme order statistics) × slope | Inert — bootstrap var ≫ Wilson var | Residual failures live here |
| **Interior** | everything else | Both variance components, captured by bootstrap | Inactive | Over-covers slightly (projection inflation) |

At small n the floor is wide relative to local sd and shields the first-k
points; at large n the floor and the failure set decouple.

### 5.2 The first-k fragile zone

At the first few grid points of a steep curve, three measured effects compound
(medians across violating high-AUC cases at n = 10,000):

| k | Upward bias of R̂ (sd units) | Lower envelope arm (sd) | Upper envelope arm (sd) |
|---|---|---|---|
| 1 | +0.66 | 0.80 | 2.78 |
| 3 | +0.38 | 1.40 | 2.60 |
| 10 | +0.18 | 3.16 | 3.25 |
| 25 | +0.10 | 3.50 | 3.40 |

1. **One-sided support collapse**: the lower envelope arm is ~1 sd where the
   interior reaches ~3.5 sd (the bootstrap cannot deviate below the observed
   extreme order statistics), while pointwise variance looks healthy.
2. **Upward bias of the empirical ROC**: extreme-quantile geometry biases R̂
   up by ~0.4–0.7 sd at k = 1–3; the bootstrap, centered on R̂, cannot see it.
   Violating draws are *optimistic* draws (their empirical AUC exceeds truth).
3. **No floor engagement**: bootstrap variance exceeds Wilson variance by
   10–60×, so the deficiency gate reads "healthy" exactly here.

Because effects (1) and (2) are order-statistic geometry at fixed k, they do
not shrink in sd units as n grows, while the floor's protection withdraws.
Violations therefore become more frequent but smaller in absolute TPR with n:
the 0–10% FPR region violation rate rises 0.8% → 15.1% from n = 30 to
n = 10,000 while the median violation magnitude falls ~5pp → ~0.5pp.

The Beta order-statistic floor (§3.4) targets exactly this zone: all three
effects are manifestations of extreme-order-statistic geometry, which the
floor bounds exactly rather than approximately.

### 5.3 Violation magnitudes are small on average — with a high-AUC caveat

At n = 10,000:

- Mean max violation: ~0.002 (0.2 percentage points of TPR); median among
  violators ~0.5pp, with 66% of violations under 1pp
- P99 max violation: ~0.046
- Violations concentrate below FPR ≈ 0.001 — the first one to five grid
  points (fig3 shows the coarser 0–10% binning)

The "technical failure" framing — R_true escapes by amounts too small to
matter — is accurate *on average* but must be qualified by AUC. In the
AUC > 0.95 stratum, violations exceeding 5pp of TPR occur in 4.0% of cases at
n = 300, 5.3% at n = 1,000, and 7.1% at n = 10,000, and the worst stored miss
is 0.668. For high-AUC applications where decisions ride on low-FPR operating
points, these are precisely the cases that matter.

### 5.4 Coverage with the integrated Beta floor

Measured on 1,400 problem-domain cases (4 DGP families × n ∈ {1000, 10000}
× high/low AUC, prevalence 50% — the strata where the pre-floor band fails),
comparing the band with and without the integrated floor on identical data
and bootstrap matrices:

| Stratum | Without Beta floor | With Beta floor | Area ratio |
|---|---|---|---|
| n=1000, AUC ≤ 0.9 | 0.948 | 0.988 | 1.020 |
| n=1000, AUC > 0.9 | 0.843 | 0.990 | 1.106 |
| n=10000, AUC ≤ 0.9 | 0.813 | 0.953 | 1.003 |
| n=10000, AUC > 0.9 | 0.767 | 0.977 | 1.020 |

The floor fixes ~85% of violations (75/84 at n=1000, 105/126 at n=10000)
while breaking zero covered cases; the >5pp violation rate in the high-AUC
stratum drops from 7.25%/3.67% to 0.0%. Remaining violations sit at median
grid index k ≈ 100–220, outside any tail jurisdiction — the separate, milder
interior-calibration residue. Two qualifications: (i) the floored band
slightly overshoots nominal (0.95–0.99 vs 0.95) because the floor's alpha is
bolted on rather than folded into the band's budget; (ii) the informativeness
cost is real and concentrated below FPR ≈ 43/n₀ on the lower side only — in
particular the lower band is honestly ~0 below FPR ≈ 7/n₀, where the
pre-floor band's nonzero claims were never supportable.

**Band attribution** (mean % of x-axis grid points whose final bound each
mechanism strictly set): the Beta floor governs 7.5–8.1% of the lower band
at n=1000 and 0.8% at n=10000 (its fixed-k jurisdiction as a fraction of the
grid); the Wilson rectangle owns the TPR plateau (67% of the lower band at
high-AUC n=1000, 45% at high-AUC n=10000); the bootstrap envelope sets the
rest of the lower band and ≥95% of the upper band. The three mechanisms
barely overlap — each carries the uncertainty channel the others cannot see,
which is the quantitative form of the hybrid argument in §1.

---

## 6. Dependence on AUC

Coverage degrades monotonically in AUC: at n = 10,000, coverage by AUC bin is
0.907 / 0.838 / 0.772 / 0.759 / 0.742 across (0.5–0.7 / 0.7–0.8 / 0.8–0.9 /
0.9–0.95 / 0.95–1.0). This is expected from the first-k mechanism:

At high AUC, the ROC curve rises steeply at low FPR, so the slope term
(g/f)² · t(1−t)/n₀ dominates the local variance and the threshold-location
error at the first grid points converts into large vertical misses. At low
AUC, the curve is flat near the origin, R̂(t_hi) ≈ R̂(t) for nearby FPR values,
and the first-k effects cost little.

The precise risk factor is the **early slope**, not the AUC number itself.
Within-DGP point-biserial correlations of violation with true AUC at
n = 10,000 are +0.19 to +0.27 for five of seven DGP families; the exceptions
(hetero_gaussian +0.02, weibull +0.05) are families whose high-AUC
parameterizations produce comparatively shallow early slopes. A related
curiosity: within student_t, heavier tails (lower df) are slightly *safer*,
plausibly because heavy-tailed negatives place observations deeper into the
extreme tail, extending the bootstrap's support exactly where it matters.

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

- **95% CI, n ≈ 100–500 per class**: Coverage 0.953–0.976 at prevalence 50%.
  This is the method's sweet spot. At n = 1,000 (prevalence 50%) coverage is
  already 0.915; the previously reported 0.950 at n = 1,000 was a
  prevalence-pooling artifact.
- **All DGPs**: Coverage varies by at most ~6pp across 7 tested distributions
  at any fixed sample size (fig1). Distribution family is second-order; ROC
  geometry (early slope) is first-order.
- **Class imbalance**: Coverage improves as prevalence departs from 50%
  (fewer positives → wider Wilson floor). Imbalance is not a risk factor.

### Where the method is over-conservative

- **Small n (≤ 30)**: Coverage is 99–100% because the Wilson floor dominates.
  Bands are wider than necessary.
- **50% CI**: Coverage is far above the 50% target (0.93 at n = 10,
  diminishing to 0.64 at n = 10,000). The cause is a compound of the Wilson
  floor, bootstrap step-function conservatism, and the sup-norm's weak
  sensitivity to α — the last of which affects any sup-norm-calibrated
  simultaneous band, not just the envelope (confirmed by the G.1
  variance-model experiment). The method is not designed for, and should not
  be used at, low confidence levels.

### Where the method under-covered before the Beta floor

- **Moderate-to-large n with balanced classes**: 0.915 at n = 1,000 and 0.830
  at n = 10,000 (prevalence 50%). Violations were lower-bound (~10:1),
  concentrated at the first few grid points (k = 1–10 negatives above
  threshold), and small on average (~0.5pp median at n = 10,000).
- **High AUC**: Monotone degradation, visible from n = 100 up. At n = 10,000
  and AUC > 0.95, coverage was ~0.74. The steep early slope makes the
  threshold-location uncertainty at the first grid points large in TPR units,
  and neither the bootstrap (one-sided support collapse, upward-biased
  center) nor the Wilson floor (wrong yardstick: binomial-only variance,
  3–6× too small in sd there) addresses it.

The integrated Beta order-statistic floor (§3.4) repairs this regime: on the
problem-domain strata, coverage rises to 0.95–0.99 and the >5pp violation
rate drops to zero (§5.4). The price is paid where it is owed — the lower
band is vacuous below FPR ≈ 7/n₀ (no distribution-free bound exists there)
and weakened up to FPR ≈ 43/n₀ — plus a mild overshoot of nominal pending
joint alpha calibration. The misses that remain even with the floor (2–5% of
cases at n = 10,000) live at interior grid points k ≈ 50–500 and are the
milder global-calibration phenomenon, not the boundary mechanism.

### Degradation profile

Before the Beta floor, typical failures were small and localized: mean max
violation ≤ 0.003 at every sample size, P99 ≤ 0.046, and only 0.69% of
simulations exceeded 5pp overall. But the average concealed the high-AUC
stratum: for AUC > 0.95, the >5pp rate was 4.0% / 5.3% / 7.1% at
n = 300 / 1,000 / 10,000, with a worst stored miss of 0.668 of TPR.

With the floor integrated, the conditional summary collapses to a simpler
one: violations that remain are tiny and interior (median worst point at
k ≈ 100–220), the catastrophic high-AUC low-FPR misses are eliminated
(0% > 5pp on the problem domains), and the method's honest limitation is
informativeness rather than validity — it declines to certify a lower bound
below FPR ≈ 7/n₀, because no nonparametric method can. Applications needing
guarantees at lower FPR must size n₀ accordingly. Remaining qualifications:
the floored band has been validated on the problem-domain strata (4 DGP
families, n ∈ {1000, 10000}); the full 7-DGP suite evaluation and folding
the floor's alpha into the band's overall budget are outstanding.
