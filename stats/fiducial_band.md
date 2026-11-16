# Rank-Space Fiducial Simultaneous Confidence Bands for ROC Curves

## Abstract

We present a nonparametric method for constructing simultaneous confidence bands (SCB) for the **true population** ROC curve from the rank structure of the two-sample problem. Under the transform $`s \mapsto 1 - F(s)`$, with $`F`$ the negative-class CDF, the negatives become uniform and the positives acquire CDF exactly $`R`$, so the merged class labels in score order carry everything the data say about the ROC. Each class's CDF evaluated at its own order statistics has the exact law of uniform order statistics, $`\text{Dirichlet}(1, \ldots, 1)`$ spacings, at every sample size. The method draws a cloud of fiducial ROC curves from independent spacing vectors for the two classes, trims it by minimum pointwise rank depth so that every grid point carries the same local tail level, returns the pointwise order-statistic tube at that depth, and widens the two degenerate corners with binomial allowances. The band is rank-invariant, adapts to local variance, needs no bootstrap, density, or slope estimate, and has one tuning input, the Monte Carlo budget. On a regular interior it is asymptotically calibrated. It has **no finite-sample coverage guarantee**: we give an exact construction on which its coverage stays near $`1 - e^{-1}`$ at every sample size and level, and we locate the mechanism in the completion of the two end gaps. In the paper's main simulation suite (seven score families, 6,000 parameter draws each, $`n`$ = 10 to 10,000, three levels) it covers above nominal in every cell at balanced prevalence, at about half the area of the Kolmogorov–Smirnov band; under class imbalance a few heteroscedastic cells fall below nominal.

This document specifies the unfloored band. The exact composition band that serves as its corner floor is specified in [`ell_band.md`](ell_band.md). Proofs and the full mechanism analysis are in [`fiducial_band_theory.md`](fiducial_band_theory.md) (cited below as "theory"); measurements outside the main suite are in [`research_record.md`](research_record.md).

---

## 1. Setup and Assumptions

**Data:** $`\mathcal{D} = \{(y_i, s_i)\}_{i=1}^N`$ where $`y_i \in \{0,1\}`$ is the class label and $`s_i \in \mathbb{R}`$ is the score. Let $`n_0`$ and $`n_1`$ denote the number of negatives and positives, $`N = n_0 + n_1`$.

**Assumptions:**
- A1: Independent sampling within each class and between classes, with class sizes fixed by the design (or conditioned on)
- A2: Continuous class score distributions $`F`$ (negatives) and $`G`$ (positives); no densities, moments, or common support are required. Tied scores are handled by randomized tie-breaking (§2.9), which redefines the estimand
- A3: Higher scores indicate the positive class
- A4: The scoring rule is fixed independently of the evaluation sample

**Estimand.** With $`Q_F(u) = \inf\{x : F(x) \ge u\}`$, $`Q_F(0) = -\infty`$,

```math
R(t) = 1 - G\big(Q_F(1-t)\big), \qquad t \in [0,1].
```

$`R`$ is nondecreasing with $`R(1) = 1`$. It need not be continuous, and $`R(0) > 0`$ is possible when the positive support extends above the negative support. Neither property is assumed.

**Target:** Construct $`\mathcal{B}_\alpha(t) = [L(t), U(t)]`$ such that:

```math
P\left(\forall t \in [0,1]: R(t) \in \mathcal{B}_\alpha(t)\right) \geq 1 - \alpha
```

The probability includes the band's own Monte Carlo randomization. Section 3 states which parts of this target the method attains, and in what sense.

---

## 2. Algorithm

### 2.1 Merged Label Sequence and Empirical ROC

Sort the pooled sample by descending score and record the class labels, $`\lambda \in \{0,1\}^N`$. For $`k = 0, \ldots, n_0 - 1`$, let $`\hat{k}_k`$ count the positives ranked above the $`(k+1)`$-th largest negative, and set $`\hat{k}_{n_0} = n_1`$. The empirical ROC on the native grid is $`\hat{R}(t_k) = \hat{k}_k / n_1`$, the upper staircase convention: the operating point just before the $`(k+1)`$-th negative is admitted.

**Key Insight (rank reduction).** Put $`U_i = 1 - F(X_i)`$ for negatives and $`W_j = 1 - F(Y_j)`$ for positives. Then $`U_i \sim \text{Unif}(0,1)`$ and $`P(W_j \le t) = R(t)`$ exactly (theory Prop. 1). The merged label sequence therefore has the same law as the sequence obtained by merging $`n_0`$ uniforms with $`n_1`$ draws from $`R`$. It is a maximal invariant under strictly increasing score transforms (theory Prop. 2), so any procedure that reads only $`\lambda`$ has coverage $`\mathcal{C}(R, n_0, n_1, \alpha)`$: a function of the ROC curve and the class sizes alone. This does not make coverage the same for different curves, including curves with the same AUC.

### 2.2 Evaluation Grid

The band is computed on the native grid

```math
\mathcal{T} = \{t_k = k/n_0 : k = 0, 1, \ldots, n_0\}.
```

Every empirical ROC has its FPR jumps on $`\mathcal{T}`$. A band on $`\mathcal{T}`$ extends to the continuum without loss of coverage by holding the lower edge from the grid point on the left and the upper edge from the grid point on the right: for $`t \in (t_k, t_{k+1})`$, $`L(t) = L(t_k)`$ and $`U(t) = U(t_{k+1})`$. For a monotone truth, coverage at every grid point then implies coverage everywhere, including when $`R`$ has jumps (theory §1). Linear interpolation of the edges does not have this property.

If a coarser output grid $`\text{linspace}(0,1,K)`$ is requested, the edges are resampled by the same conservative rule (lower edge from $`\lfloor t n_0 \rfloor`$, upper edge from $`\lceil t n_0 \rceil`$).

### 2.3 Fiducial Cloud

For $`b = 1, \ldots, M`$, conditional on $`\lambda`$:

1. Draw independent spacing vectors $`S^0 \sim \text{Dirichlet}(1^{n_0+1})`$ and $`S^1 \sim \text{Dirichlet}(1^{n_1+1})`$, as normalized cumulative sums of iid standard exponentials.
2. Place the $`i`$-th negative at FPR coordinate $`u_i = \sum_{r \le i} S^0_r`$ and the $`j`$-th positive at TPR coordinate $`v_j = \sum_{r \le j} S^1_r`$.
3. On each axis separately, give every maximal run of the *other* class's observations sorted iid uniform fractions of the gap that contains it.
4. Join the resulting ordered coordinate pairs, with $`(0,0)`$ and $`(1,1)`$, by straight lines, and evaluate the polyline $`c_b`$ on $`\mathcal{T}`$.

The two axis completions are independent. Step 1 uses an exact law (theory Prop. 3): for a continuous CDF $`H`$, the values $`H(Z_{(1)}), \ldots, H(Z_{(n)})`$ at its own ordered sample are distributed as uniform order statistics, with $`H(Z_{(i)}) \sim \text{Beta}(i, n+1-i)`$. Steps 3–4 are a modeling choice. Nothing in the data determines where the other class's mass lies inside a gap, and sorted-uniform spreading acts like local linearity of the ROC there (§3.2).

### 2.4 Min-p Depth

At each trim column $`k \in \mathcal{J}`$, give every draw its tie-inclusive two-sided rank in the cloud,

```math
d_{bk} = \min\Big\{\#\{a : c_a(t_k) \le c_b(t_k)\},\; \#\{a : c_a(t_k) \ge c_b(t_k)\}\Big\},
```

and score the draw by its minimum over the trim columns,

```math
D_b = \min_{k \in \mathcal{J}} d_{bk}.
```

$`D_b`$ is small when the draw is extreme at *some* grid point, measured on the rank scale of that point. Because ranks are scale-free, the depth studentizes each column by its own marginal distribution, with no variance estimate.

**Trim columns.** $`\mathcal{J} = \mathcal{T}`$ when $`|\mathcal{T}| \le 2001`$. Larger grids are thinned to every $`\lceil |\mathcal{T}|/1000 \rceil`$-th point plus the first and last 50 points; the band is still built on the full grid. Thinning removes depth constraints, so it can only raise the trim depth and narrow the band (theory Prop. 10b and §6.2).

### 2.5 Trim Depth

Let $`a = \alpha_{\text{eff}} = 1 - (1-\alpha)^C`$ with trim exponent $`C = 1`$ (the identity, $`a = \alpha`$). The trim depth is the $`(\lfloor aM \rfloor + 1)`$-th smallest depth, clipped:

```math
j = \max\Big\{1,\; \min\big\{D_{(\lfloor aM \rfloor + 1)},\; \lfloor M/2 \rfloor\big\}\Big\}, \qquad \ell = \frac{j}{M+1}.
```

$`\ell`$ is the realized **local level**: the one-sided tail probability, on the cloud's rank scale, that the band leaves at every grid point. It is selected from the data and the cloud jointly.

**Lemma (rank–tube equivalence; theory Lemma 5).** A draw lies inside the tube of §2.6 at every trim column iff $`D_b \ge j`$. Consequently at most $`\lfloor aM \rfloor`$ draws leave the tube on $`\mathcal{J}`$, so its in-sample cloud content is at least $`1 - a`$ (theory Prop. 6). This is a statement about the cloud, not about the population ROC.

The exponent $`C`$ is retained as a parameter for ablation only. Under interior calibration a fixed $`C > 1`$ has limiting coverage $`(1-\alpha)^C`$, below nominal (theory §6.2), and every attempt to choose $`C`$ from data or from a library of shapes failed (`research_record.md` Q4).

### 2.6 Order-Statistic Tube

At every grid column, including columns omitted from $`\mathcal{J}`$,

```math
L_{\text{raw}}(t_k) = c_{(j)}(t_k), \qquad U_{\text{raw}}(t_k) = c_{(M-j+1)}(t_k),
```

the $`j`$-th smallest and $`j`$-th largest cloud values. The tube is pointwise, not the envelope of a retained set of whole curves; it contains the envelope of the curves with $`D_b \ge j`$, and the reverse inclusion fails in general (theory §5).

### 2.7 Corner Allowances

A credible band built from the cloud cannot reach the corners where the truth can sit. Every polyline has $`c_b(0) = 0`$, so the raw upper edge is 0 at FPR 0 for any $`M`$. Where every positive outranks the operating threshold, every draw sits strictly below 1, while the truth can equal 1 there. Two allowances repair this:

**Upper allowance (Clopper–Pearson form).** At the band's own local level $`\ell`$,

```math
U_{\text{CP}}(t_k) = \begin{cases} \text{Beta}^{-1}\big(1-\ell;\; \hat{k}_k + 1,\; n_1 - \hat{k}_k\big), & \hat{k}_k < n_1, \\ 1, & \hat{k}_k = n_1, \end{cases}
```

and $`U(t_k) = \max_{i \le k} \max\{U_{\text{raw}}(t_i), U_{\text{CP}}(t_i)\}`$ (running maximum).

**Lower allowance.** $`L(t_k) = 0`$ wherever $`\hat{k}_k = 0`$.

Both operations only widen the tube. Neither is an independent Clopper–Pearson guarantee: the operating threshold is a random negative order statistic whose true FPR is $`U_{(k+1)}`$, not $`t_k`$, and $`\ell`$ was selected using the same data (theory §5). The upper allowance is load-bearing. Where the truth reaches TPR 1 its cloud depth is 0 in essentially every replicate and the allowance carries all of the coverage (`research_record.md` Q1). It was designed after observing that failure. The allowance also makes $`U(0) > 0`$ possible, so the band does not exclude $`R(0) > 0`$. A mirrored Clopper–Pearson lower bound was measured to cost about 15% area for no coverage and is not used.

### 2.8 Boundary Handling

**Clipping:** Clip all values to $`[0,1]`$.

**Boundary Enforcement:**
- $`L(0) = 0`$
- $`U(1) = 1`$

The upper edge at $`t = 0`$ is **not** pinned to 0: separated supports give $`R(0) > 0`$ (theory §1, Cor. 9b).

### 2.9 Ties

The default breaks tied scores uniformly at random, independently of class. Equivalently, attach an independent continuous uniform to every observation and order lexicographically. The augmented scores are continuous, so every statement in §3 applies, and the estimand becomes the ROC of the threshold rule that randomizes within tied scores: linear tie segments and AUC $`P(Y > X) + \tfrac{1}{2} P(Y = X)`$ (theory Prop. 2a). A deterministic alternative (`"even"`) spreads each class evenly within tied blocks; it has no theorem. Breaking ties by class changes the estimand and is not offered. At 20 and 100 score levels, randomized tie-breaking was indistinguishable from continuous scores (`research_record.md` Q1).

---

## 3. Properties

### 3.1 Coverage Guarantees

**Definition (Population Coverage).** The probability that the true population ROC curve falls entirely within the band, over both the data and the band's Monte Carlo draws:
```math
P\left(\forall t: R(t) \in [L(t), U(t)]\right)
```

**Note on Future Samples:** This method constructs a *confidence band* for $`R`$. It is **not** a *prediction band* for a future empirical ROC $`\hat{R}_{\text{new}}`$. A fresh empirical curve adds its own sampling error, approximately doubling the covariance (theory §6.5), so its coverage is far below $`(1-\alpha)`$.

**Proposition 1 (Exact properties).** For every $`\lambda`$, $`\alpha`$, and cloud:
1. *Rank invariance.* The band depends on the scores only through $`\lambda`$ (and the tie-breaking draws). Its coverage depends on the data-generating process only through $`(R, n_0, n_1)`$.
2. *Monotone edges.* $`L`$ and $`U`$ are nondecreasing in $`t`$: pointwise order statistics of nondecreasing curves are nondecreasing, the zeroed lower prefix $`\{k : \hat{k}_k = 0\}`$ is an initial segment, and the upper edge is a running maximum.
3. *Nesting in $`\alpha`$.* On a common cloud, $`j`$ is nondecreasing in $`\alpha`$, so the raw tube narrows and the local level $`\ell`$ rises, which lowers $`U_{\text{CP}}`$. Bands at larger $`\alpha`$ are therefore contained in bands at smaller $`\alpha`$. The production implementation trims one shared cloud at every requested level.
4. *In-sample content.* At most $`\lfloor \alpha M \rfloor`$ draws leave the raw tube on the trim grid (§2.5).

---

**Theorem 1 (Asymptotic Interior Calibration; theory Thm. 7).** Fix $`I = [\varepsilon, 1-\varepsilon]`$. Suppose $`R`$ is continuously differentiable on a neighborhood of $`I`$ with $`0 < R' < \infty`$ there, and $`n_1/n_0 \to \lambda \in (0,\infty)`$. Then in $`\ell^\infty(I)`$,

```math
\sqrt{n_1}\,(\hat{R} - R) \Rightarrow Z, \qquad \mathcal{L}_*\big\{\sqrt{n_1}\,(c_b - \hat{R})\big\} \Rightarrow_P \mathcal{L}(Z),
```

where, for independent standard Brownian bridges $`B_0, B_1`$,

```math
Z(t) = B_1\big(R(t)\big) - \sqrt{\lambda}\, R'(t)\, B_0(t), \qquad \mathrm{Var}\, Z(t) = \sigma^2(t) = R(t)\{1-R(t)\} + \lambda R'(t)^2\, t(1-t).
```

If, in addition, the tube is trimmed on $`I`$ alone, the trim mesh tends to zero, $`M_n / \log K_n \to \infty`$ (with $`K_n`$ the number of trim columns), and the limiting depth cutoff is a continuity and local strict-increase point, then

```math
P\left(L_{\text{raw}}(t) \le R(t) \le U_{\text{raw}}(t)\ \ \forall t \in I\right) \to 1 - a.
```

*Proof sketch:*

1. *One-class bridge.* Writing Dirichlet spacings as normalized exponentials gives exchangeable weights with bounded moments of every order. The conditional weighted-bootstrap theorem of Praestgaard and Wellner (1993, Thms. 2.1–2.2) gives a conditional Brownian-bridge limit for each class's fiducial CDF.

2. *Completion and endpoint conventions.* Moving between $`n`$ and $`n+1`$ spacings, between step conventions, and filling gaps with sorted uniforms costs at most a maximal spacing. On a regular interior the completion diameter is $`O_p(\log n / n) = o_p(n^{-1/2})`$, conditionally in probability (theory §3.1).

3. *Delta method.* The ROC map $`(A, B) \mapsto B \circ A^{-1}`$ has derivative $`\dot\Psi(h_0, h_1)(t) = h_1(t) - R'(t)\, h_0(t)`$ at the population pair. Independent class bridges give the stated limit for both the empirical and the fiducial process.

4. *Trim transfer.* The Gaussian pointwise tail depth of a path $`z`$ is $`\Phi(-|z(t)|/\sigma(t))`$, so the limiting equal-local-level central set is $`\{\|z/\sigma\|_{\infty, I} \le c_a\}`$ with $`P(\|Z/\sigma\|_{\infty,I} \le c_a) = 1 - a`$. Column-wise DKW bounds and cutoff regularity transfer this set to the finite cloud. Since $`-Z \overset{d}{=} Z`$, the truth has the same limiting inclusion probability as a centered draw. $`\square`$

The steps are verified at the level of a proof outline. A full appendix would spell out uniform conditional quantile and depth convergence and the completion comparison.

**Remark (scope of Theorem 1 for the shipped algorithm).** Three features of the implementation lie outside the theorem (theory §6.2). (i) The trim runs over the whole grid, including the moving end columns $`t = k/n_0`$ at fixed $`k`$, where the relevant statistics are extreme order statistics and no Gaussian limit applies. Full-grid trimming is not interior calibration, and at the ends the cloud can be anti-conservative (§3.2). (ii) $`M = 20{,}000`$ is fixed, so $`M_n/\log K_n \to \infty`$ does not hold along an unbounded sequence. (iii) Above 2,001 grid points the trim grid is thinned to about 1,100 columns, and its interior mesh does not tend to zero. Theorem 1 is therefore the asymptotic rationale for the construction, not a coverage statement for it. Directly measured: on binormal AUC .95, kink, and $`t_2`$ shapes at $`n = 50{,}000`$ per class, the full-grid $`C = 1`$ band covered .951–.960 at $`\alpha = .05`$, against .914–.925 for $`C = 2`$ (`research_record.md` Q4, Stage S).

---

**Proposition 2 (The band is not honest at any level).** *[Exact.]* Fix $`n_0 \ge 1`$, $`n_1 \ge 1`$, $`\alpha \in (0,1)`$, $`M`$, and $`C`$. For every $`\epsilon \in (1/n_0, 1)`$ there is a pair of continuous score distributions under which the band's coverage is at most $`1 - (1-\epsilon)^{n_0}`$. Hence

```math
\inf_{F,\,G} P\left(\forall t: R(t) \in \mathcal{B}_\alpha(t)\right) \;\le\; 1 - \left(1 - \frac{1}{n_0}\right)^{n_0} \;\le\; 0.652 \quad (n_0 \ge 10), \qquad \to 1 - e^{-1} \approx 0.632.
```

*Proof.* Let positives be $`\text{Unif}[2,3]`$ and negatives the mixture $`(1-\epsilon)\,\text{Unif}[0,1] + \epsilon\,\text{Unif}[4,5]`$. Then $`R(t) = 0`$ for $`t < \epsilon`$, in particular at $`t_1 = 1/n_0`$. With probability $`(1-\epsilon)^{n_0}`$ no negative falls in $`[4,5]`$, and the label path is perfectly separated: every positive outranks every negative. On that path $`\hat{k}_k = n_1 > 0`$ for every $`k`$, so the lower allowance never applies. Every fiducial polyline rises from $`(0,0)`$ through positive points with strictly positive coordinates, so $`c_b(t_1) > 0`$ almost surely for every draw, and $`L(t_1) = c_{(j)}(t_1) > 0 = R(t_1)`$. $`\square`$

The bound does not depend on $`\alpha`$, $`M`$, or $`C`$: no choice of level or budget repairs it. This is the left-corner instance of the missing-mass lower bounds (theory Cor. 9b), which require any deterministic honest rank band to have $`L(t) = 0`$ for $`t < 1 - \alpha^{1/n_0}`$ on the separated path. The fiducial lower edge at $`t_1`$ is instead about $`\hat{k}_0 / \{n_1 \log(1/\ell)\}`$ (theory eq. 7.2), roughly $`0.14`$ at $`\ell = 10^{-3}`$. In 300 replicates of this construction with $`\epsilon = 1.05/n_0`$ and $`n_0 = n_1 \in \{20, 100, 1000\}`$, the 95% band covered .17–.20. Every separated replicate missed, and the band also missed in most replicates where a high-scoring negative was observed, because the truth jumps from 0 to 1 within the first few grid points. The exact band of [`ell_band.md`](ell_band.md) covered every replicate.

The construction is an extreme case of a general one. Positive or negative mass at the $`1/n`$ scale goes unobserved with probability bounded away from zero, and on that event the data law is exactly the law without it (theory §9, §12.3). Any honest band must leave room for such mass; sorted-uniform completion does not. Constructed slivers of this kind at AUC .60–.95 and $`n`$ up to 2,000 per class reduced coverage to .505–.613, and a step ROC to .108–.238, worsening with $`n`$ (`research_record.md` Q2). No summary of AUC and class counts identifies these cases from the data (theory §12.3).

---

### 3.2 Finite-Sample Behavior

Two departures from Theorem 1 act in opposite directions and in different places.

**Interior over-coverage (second order).** Each draw reads the data's error process $`E = \hat{R} - R`$ at a randomly shifted FPR $`t + \beta_b(t)`$, with $`\beta_b(t)`$ approximately $`N(0, h_t^2)`$, $`h_t^2 = t(1-t)/(n_0+2)`$. To second order the cloud centre is a kernel smooth of the empirical ROC at bandwidth $`h_t \asymp n_0^{-1/2}`$ (theory §6.6):

```math
m(t) - R(t) = (\mathsf{K}_t E)(t), \qquad \mathsf{K}_t = \text{convolution with } N(0, h_t^2).
```

The truth's deviation from the centre is therefore a *smoothed* error process, while the depth cutoff is calibrated on draws that are rough down to the grid. The cutoff exceeds what the truth needs by an amount of order $`n^{-1/4}`$ in normal-score units, the same at every interior FPR. *[Asymptotic; heuristic derivation, checked numerically.]* The consequence is uniform interior over-coverage that decays slowly. On binormal AUC .8 and the diagonal at 500 per class, trimmed and checked on FPR $`(0.1, 0.9)`$ only, the band missed at .020–.022, .090–.110, and .273–.285 for $`\alpha = .05, .2, .5`$. On binormal AUC .8 at 5,000 per class it still covered .963, .883, and .667. A depth computed from kernel-smoothed draws returned interior misses to nominal at 8–24% less width; it is a research arm, not part of the method (theory §6.6).

**Corner under-coverage (first order, local).** At the two ends of the grid the uncertainty is set by a few extreme order statistics. For fixed $`k`$, $`n_0 U_{(k)} \Rightarrow \text{Gamma}(k,1)`$, and end spacings are exponential on the $`1/n`$ scale. Sorted-uniform completion spreads the first negative gap, and the run of $`K`$ negatives below the smallest positive, linearly. If the truth is convex there, it sags below the chord the cloud draws, and the lower edge claims a TPR deficit of order $`\log(1/\ell)/(n_1 K)`$ where the truth's is order $`1/n_1`$ (theory §7.4). Heavy tails make the ROC convex at both ends. Positives more dispersed than negatives make it convex near FPR 1. Scarce positives lengthen the saturated run $`K`$: its median at $`n = 1{,}000`$ is 38, 69, and 170 negatives at 50%, 30%, and 10% prevalence. Every main-suite cell below nominal (§3.7) is heteroscedastic Gaussian at 30% or 10% prevalence, the configuration in which misses concentrate on the TPR plateau and fall below the band (§3.3). Proposition 2 is the limiting case of the mechanism.

### 3.3 Direction of Misses

**Proposition 3 (Near-balance in the Gaussian limit; theory Prop. 7a).** *[Exact for the limiting Gaussian band.]* Let $`X = Z/\sigma`$ on $`I`$, choose $`c`$ with $`P(\sup_I |X| > c) = a`$, and let $`p`$ be the common probability of an upper and of a lower excursion and $`b`$ the probability of both. Then

```math
\frac{a}{2} \le p \le 1 - \sqrt{1-a}, \qquad 0 \le b \le \big(1 - \sqrt{1-a}\big)^2.
```

*Proof.* Gaussian sign symmetry equalizes the two directions, and $`a = 2p - b`$. The covariance of $`Z`$ is nonnegative, so by Pitt's theorem increasing and decreasing events are negatively correlated in the sense $`b \le p^2`$. $`\square`$

At $`a = .05`$ each directional probability lies in $`[.0250, .0254]`$. Systematic directional imbalance is therefore not a first-order interior phenomenon; it points to the corners, to completion, or to the corner allowances. Pooled over the main suite the misses are close to balanced, with a slight upper tilt: at 50% prevalence and $`\alpha = .05`$, 0.84% of datasets miss below and 0.99% above. In the stress strata the corner mechanism makes them strongly one-sided. For Student-$`t`$ with df $`< 4`$ and the heteroscedastic Gaussian with $`\sigma_+/\sigma_- > 2`$, at AUC $`> 0.9`$ and $`n = 100`$–$`3{,}000`$, the truth fell below the 95% band in 5.6%, 9.3%, and 18.4% of datasets at 50%, 30%, and 10% prevalence, and above it in 1.1–1.3% (README §5).

### 3.4 Asymmetry and Heteroscedasticity Adaptation

The tube is asymmetric wherever the cloud's marginals are skewed, which happens near both corners: draws cannot exceed 1, so near TPR 1 the cloud spreads downward and the tube follows. No separate machinery is needed.

The rank depth studentizes every column by its own marginal law. To first order on a regular interior, the tube's half-width is

```math
\frac{U_{\text{raw}}(t) - L_{\text{raw}}(t)}{2} = \frac{c_a\, \sigma(t)}{\sqrt{n_1}} + o_p(n_1^{-1/2}),
```

with $`\sigma^2(t)\,/\,n_1 = R(1-R)/n_1 + R'^2\, t(1-t)/n_0`$ (theory eq. 11.1). This is the Hsieh–Turnbull variance: both classes' sampling errors enter in quadrature, and the slope term is carried by the cloud's horizontal spacings rather than by a density-ratio estimate. Bands that protect each class's CDF separately pay $`a_t + b_t`$ in place of $`\sqrt{a_t^2 + b_t^2}`$, with $`a_t`$, $`b_t`$ the two square-root terms, which is up to $`\sqrt{2}`$ times more at equal multipliers (theory eq. 11.2). This is why the fiducial band is narrower than composition bands.

### 3.5 Step-Function Structure

On the native grid the edges are evaluated at the jump points of $`\hat{R}`$, and the conservative continuum extension of §2.2 makes $`L`$ and $`U`$ step functions. Unlike the empirical staircase, the raw edges are pointwise order statistics of continuous piecewise-linear curves, so they are continuous in $`t`$ and change gradually from one grid point to the next. Discontinuities arise only where an allowance switches on or off: at the end of the zeroed lower prefix, and where the Clopper–Pearson upper bound overtakes the raw edge.

---

### 3.6 Summary of Guarantees

| Property | Finite Sample | Asymptotic |
|----------|---------------|------------|
| Population coverage, unrestricted continuous class | ✗. Infimum $`\le 1 - (1-1/n_0)^{n_0} \approx 0.63`$ at every $`n`$ and $`\alpha`$ (Prop. 2) | ✗ at the moving boundary points $`t = k/n_0`$, $`k`$ fixed |
| Population coverage, main simulation suite | $`\ge`$ nominal in every cell at 50% prevalence; 1 of 126 cell–level combinations below nominal at 30% and 5 at 10% (§3.7) | — |
| Interior calibration | Conservative; surplus of order $`n^{-1/4}`$ in the cutoff (§3.2) | $`\to 1 - \alpha`$ on a regular interior under interior trimming and growing $`M`$ (Thm. 1) |
| Directional balance | Balanced on benign shapes; lower-edge misses dominate on convex corners | Near-balanced, sharp bounds (Prop. 3) |
| Rank invariance | ✓ exact | ✓ |
| Width adapts to local variance | ✓ through the rank depth | ✓ Half-width $`c_a \sigma(t)/\sqrt{n_1}`$ |
| Asymmetric at boundaries | ✓ | ✓ |
| Distribution-free validity | ✗ | ✗ |
| Future-curve coverage | $`\ll 1-\alpha`$ (not covered) | $`\ll 1-\alpha`$ |

### 3.7 Simulation Evidence

The main suite ([`simulation_spec.md`](simulation_spec.md)) draws 6,000 Latin-hypercube parameter points per family for seven score families (binormal, heteroscedastic Gaussian, logit-normal, opposing-skew Beta, Student-$`t`$, bimodal negatives, Weibull), with AUC sampled uniformly on the probit scale over $`[0.55, 0.99]`$. One dataset is drawn per point at each total sample size $`n \in \{10, 30, 100, 300, 1{,}000, 3{,}000, 10{,}000\}`$, at 50%, 30%, and 10% prevalence ($`n = 10`$ at 50% only), and bands are computed at $`\alpha \in \{.05, .2, .5\}`$ with $`M = 20{,}000`$. A cell is one family at one $`n`$ (about 6,000 datasets). Coverage is checked on the native grid excluding the pinned endpoints $`t = 0, 1`$. "Cells below" counts cells whose 95% Wilson interval lies entirely below nominal. Area is relative to the KS band (Campbell 1994) on the same datasets. All rows use $`n \ge 30`$ (42 cells).

| Prevalence | Level | Mean coverage | Worst cell | Cells below | Miss below | Miss above | Area ÷ KS |
|---|---|---|---|---|---|---|---|
| 50% | 95% | 0.982 | 0.961 | 0 | 0.008 | 0.010 | 0.51 |
| 50% | 80% | 0.916 | 0.870 | 0 | 0.036 | 0.049 | 0.49 |
| 50% | 50% | 0.736 | 0.647 | 0 | 0.113 | 0.156 | 0.48 |
| 30% | 95% | 0.980 | 0.939 | 1 | 0.010 | 0.010 | 0.53 |
| 30% | 80% | 0.913 | 0.850 | 0 | 0.038 | 0.049 | 0.51 |
| 30% | 50% | 0.737 | 0.645 | 0 | 0.114 | 0.155 | 0.49 |
| 10% | 95% | 0.975 | 0.893 | 4 | 0.015 | 0.010 | 0.59 |
| 10% | 80% | 0.909 | 0.788 | 1 | 0.042 | 0.050 | 0.56 |
| 10% | 50% | 0.735 | 0.585 | 0 | 0.116 | 0.155 | 0.54 |

Coverage against $`n`$ at 50% prevalence, mean over the seven families:

| Level | $`n=10`$ | 30 | 100 | 300 | 1,000 | 3,000 | 10,000 |
|---|---|---|---|---|---|---|---|
| 95% | 0.989 | 0.989 | 0.988 | 0.983 | 0.980 | 0.978 | 0.971 |
| 80% | 0.949 | 0.946 | 0.933 | 0.922 | 0.912 | 0.903 | 0.882 |
| 50% | 0.847 | 0.808 | 0.771 | 0.744 | 0.725 | 0.708 | 0.662 |
| Area ÷ KS, 95% | 0.66 | 0.62 | 0.55 | 0.51 | 0.48 | 0.46 | 0.45 |

The worst cell at 50% prevalence is the heteroscedastic Gaussian at $`n = 300`$ (0.961). Every cell below nominal is heteroscedastic Gaussian: $`n = 300`$ at 30% prevalence (95% level), $`n = 100`$–$`3{,}000`$ at 10% (95% level), and $`n = 300`$ at 10% (80% level). These are the strata of §3.2, and the deficit closes by $`n = 10{,}000`$ (0.946 at 10% prevalence). Two caveats bound these numbers. The suite's families are smooth and contain no slivers or jumps, so they cannot speak to the cases of Proposition 2. And the per-cell average hides concentration: in the stress strata of §3.3 coverage falls as low as 0.71 (README §7).

---

## 4. Computational Considerations

### 4.1 Complexity

| Operation | Cost |
| :--- | :--- |
| Sort and tie-break | $`O(N \log N)`$ |
| Spacings and within-gap completion | $`O(M N)`$ expected (bucketed sort of uniform keys) |
| Polyline evaluation on the grid | $`O(M (N + G))`$ |
| Min-p depths | $`O(G_J \cdot M \log M)`$ |
| Trim depth (selection) | $`O(M)`$ |
| Pointwise order statistics | $`O(G \cdot M)`$ |
| Corner allowances | $`O(G)`$ Beta quantiles |

**Total:** $`O(M N + G_J\, M \log M)`$, where $`G = n_0 + 1`$ is the grid size and $`G_J = |\mathcal{J}| \le \min(G, \approx 1{,}100)`$ the trim-grid size.

The Monte Carlo core runs in Rust (`fiducial_core`), parallel over draws, with a per-draw RNG stream that is a pure function of (seed, draw index); output is bit-identical across thread counts. Median time per dataset in the main suite, on one core under the suite's concurrent load, was 0.02 s at $`n = 10`$, 0.16 s at $`n = 100`$, 1.3 s at $`n = 1{,}000`$, and 9.6 s at $`n = 10{,}000`$.

### 4.2 Memory

Primary storage: the $`M \times G`$ cloud, held in float32 (the order-statistic resolution $`1/M \ge 5 \times 10^{-5}`$ is three orders of magnitude above float32 granularity). Memory $`= 4MG`$ bytes: 80 MB at $`n_0 = 1{,}000`$ and 800 MB at $`n_0 = 10{,}000`$ for $`M = 20{,}000`$. All levels are trimmed from the same stored cloud.

### 4.3 Monte Carlo Budget

$`M`$ controls resolution, not validity in any proved sense. At a fixed column and a continuous-law tail probability $`\ell`$, the number of draws beyond the edge is $`\text{Binomial}(M, \ell)`$, so $`M\ell`$ governs tail resolution. The realized local level falls as the grid grows, approximately $`\ell \approx 9.7 \times 10^{-4} (\alpha/.05)^{1.2} (K/500)^{-0.27}`$ (round-two fit, `research_record.md` Q1), so a fixed $`M`$ yields a shrinking depth. At $`M = 20{,}000`$ and $`\alpha = .05`$ the depth is 21–23 at $`n_0 = 150`$, 13–15 at 500, 9–11 at 1,500, and 8–9 at 5,000. The seed-to-seed standard deviation of mean width is 0.45–0.66% (theory §11.2).

Two effects remain tied to $`j`$. Integer depths make the realized trimmed fraction fall short of $`a`$, which is conservative. A tube built from the same draws that calibrate it has fresh-draw content below its in-sample content, by more when $`j`$ is small. The implementation warns when $`j < 3`$. At that point the band approaches the full cloud envelope and nearby levels become indistinguishable.

**Parameters:**
- $`M`$: number of fiducial draws (default 20,000).
- $`C`$: trim exponent (default 1; values above 1 are anti-conservative, §2.5).

---

## 5. Complete Pseudocode

```
FUNCTION fiducial_scb(labels, scores, alpha, M=20000, C=1, tie_break="random"):

    # === Step 1: Rank reduction ===
    lab = labels sorted by descending score, ties broken uniformly at random
    n0 = count(lab == 0), n1 = count(lab == 1)
    T = [k / n0 for k in 0..n0]
    khat[k] = #positives before the (k+1)-th negative in lab,  k < n0
    khat[n0] = n1

    # === Step 2: Fiducial cloud ===
    FOR b = 1 TO M:
        u = cumsum(Exp(1)^(n0+1)) / sum      # negative FPR anchors u_1..u_n0
        v = cumsum(Exp(1)^(n1+1)) / sum      # positive TPR anchors v_1..v_n1
        FOR each maximal run of positives inside negative gap (u_i, u_{i+1}):
            x-coordinates = u_i + sort(Unif^r) * (u_{i+1} - u_i)
        FOR each maximal run of negatives inside positive gap (v_j, v_{j+1}):
            y-coordinates = v_j + sort(Unif^r) * (v_{j+1} - v_j)
        polyline = (0,0), merged (x, y) points in label order, (1,1)
        cloud[b, :] = linear_interpolate(polyline, T)

    # === Step 3: Min-p depth ===
    J = T if |T| <= 2001 ELSE every ceil(|T|/1000)-th point + first/last 50
    FOR b = 1 TO M:
        D[b] = min over k in J of
               min( #{a : cloud[a,k] <= cloud[b,k]}, #{a : cloud[a,k] >= cloud[b,k]} )

    # === Step 4: Trim depth ===
    a = 1 - (1 - alpha)^C
    j = sorted(D)[floor(a * M)]          # (floor(aM)+1)-th smallest, 0-indexed
    j = clip(j, 1, floor(M / 2))
    ell = j / (M + 1)
    IF j < 3: WARN "increase M"

    # === Step 5: Order-statistic tube ===
    FOR k = 0 TO n0:
        L[k] = j-th smallest of cloud[:, k]
        U[k] = j-th largest of cloud[:, k]
    L = clip(L, 0, 1), U = clip(U, 0, 1)

    # === Step 6: Corner allowances ===
    FOR k = 0 TO n0:
        U_cp[k] = 1 IF khat[k] == n1 ELSE beta_quantile(1 - ell, khat[k] + 1, n1 - khat[k])
    U = cummax(maximum(U, U_cp))           # left to right
    L[khat == 0] = 0

    # === Step 7: Boundary anchors ===
    L[0] = 0, U[n0] = 1

    RETURN T, L, U
```

---

## 6. Formal Specification

### 6.1 Algorithm 1: Rank-Space Fiducial SCB

**Input:** Scores $`\mathcal{S} = \{(y_i, s_i)\}_{i=1}^N`$, draws $`M`$, significance $`\alpha`$, trim exponent $`C = 1`$

---

#### 1. Rank Reduction
* Break ties by independent uniform keys; sort by descending (score, key) to obtain the label sequence $`\lambda`$.
* Compute $`\hat{k}_k`$ for $`k = 0, \ldots, n_0`$ and the grid $`\mathcal{T} = \{k/n_0\}_{k=0}^{n_0}`$.

#### 2. Fiducial Cloud
* **For** $`b = 1`$ to $`M`$, independently given $`\lambda`$:
    * Draw $`S^0 \sim \text{Dirichlet}(1^{n_0+1})`$, $`S^1 \sim \text{Dirichlet}(1^{n_1+1})`$; set $`u_i = \sum_{r \le i} S^0_r`$, $`v_j = \sum_{r \le j} S^1_r`$.
    * Within each gap of either axis, place the other class's observations at sorted iid uniform fractions of the gap.
    * Evaluate the polyline through $`(0,0)`$, the merged coordinate pairs, and $`(1,1)`$ on $`\mathcal{T}`$: $`c_b(t_k)`$.

#### 3. Depth and Trim
* Select trim columns $`\mathcal{J}`$ ($`\mathcal{J} = \mathcal{T}`$ if $`|\mathcal{T}| \le 2001`$; otherwise the thinned grid of §2.4).
* $`d_{bk} \leftarrow \min\{\#\{a : c_a(t_k) \le c_b(t_k)\}, \#\{a : c_a(t_k) \ge c_b(t_k)\}\}`$; $`D_b \leftarrow \min_{k \in \mathcal{J}} d_{bk}`$.
* $`a \leftarrow 1 - (1-\alpha)^C`$; $`j \leftarrow \max\{1, \min\{D_{(\lfloor aM \rfloor + 1)}, \lfloor M/2 \rfloor\}\}`$; $`\ell \leftarrow j/(M+1)`$.

#### 4. Tube
* **For each** $`t_k \in \mathcal{T}`$: $`L(t_k) \leftarrow c_{(j)}(t_k)`$, $`U(t_k) \leftarrow c_{(M-j+1)}(t_k)`$.
* Clip $`L, U`$ to $`[0,1]`$.

#### 5. Corner Allowances
* $`U_{\text{CP}}(t_k) \leftarrow \text{Beta}^{-1}(1-\ell;\, \hat{k}_k + 1,\, n_1 - \hat{k}_k)`$ if $`\hat{k}_k < n_1`$, else 1.
* $`U(t_k) \leftarrow \max_{i \le k} \max\{U(t_i), U_{\text{CP}}(t_i)\}`$.
* $`L(t_k) \leftarrow 0`$ for every $`k`$ with $`\hat{k}_k = 0`$.

#### 6. Boundary Handling
* Enforce fixed anchor points: $`L(0) = 0`$, $`U(1) = 1`$.
* For continuum or coarser-grid output, extend $`L`$ from the left grid point and $`U`$ from the right grid point.

**Return:** Band $`[L(t), U(t)]`$ over $`\mathcal{T}`$

---

## References

- Campbell, G. (1994). Advances in statistical methodology for the evaluation of diagnostic and laboratory tests. *Statistics in Medicine* 13, 499–508.
- Cui, Y. and Hannig, J. (2019). Nonparametric generalized fiducial inference for survival functions under censoring. *Biometrika* 106, 501–518.
- Hsieh, F. and Turnbull, B. W. (1996). Nonparametric and semiparametric estimation of the receiver operating characteristic curve. *Annals of Statistics* 24, 25–40.
- Pitt, L. D. (1982). Positively correlated normal variables are associated. *Annals of Probability* 10, 496–499.
- Praestgaard, J. and Wellner, J. A. (1993). Exchangeably weighted bootstraps of the general empirical process. *Annals of Probability* 21, 2053–2086.
