# Exact Equal-Local-Levels Composition Bands for ROC Curves

## Abstract

We present a distribution-free simultaneous confidence band (SCB) for the **true population** ROC curve with a finite-sample coverage theorem. For each class, the true survival probabilities at the class's own order statistics are distributed exactly as uniform order statistics, whatever the score distribution. The method bounds every order statistic by the $`\gamma`$ and $`1-\gamma`$ quantiles of its own Beta law, using one local level $`\gamma`$ for all order statistics of a class (equal local levels, ELL). It calibrates $`\gamma`$ by an exact non-crossing probability computed by dynamic programming, with no Monte Carlo. It then composes the two class bands into an ROC band through monotonicity and the observed merged ranks. For every pair of continuous score distributions and every pair of sample sizes, the band contains the entire true ROC with probability at least $`1-\alpha`$, and each one-sided miss has probability at most $`1-\sqrt{1-\alpha}`$. The band is deterministic given the ranks and costs one sort and $`O(N)`$ index gathers per dataset after a cached calibration. It tightens in the class tails, where the constant margin of the Kolmogorov–Smirnov (KS) band is most wasteful. In a simulation suite over seven score families its mean area is 0.63–0.99 times the KS area in every cell. Its price is conservatism, which grows with $`\alpha`$: its 50% band covers 0.95–0.99.

The ELL band is the exact reference method of this project and the corner floor of the fiducial hybrid. The code calls it `m3`. The fiducial band it floors is specified in [`fiducial_band.md`](fiducial_band.md). Proofs and related constructions are in [`fiducial_band_theory.md`](fiducial_band_theory.md) (cited below as "theory"), especially §§9 and 12.

---

## 1. Setup and Assumptions

**Data:** $`\mathcal{D} = \{(y_i, s_i)\}_{i=1}^N`$ where $`y_i \in \{0,1\}`$ is the class label and $`s_i \in \mathbb{R}`$ is the score. Let $`n_0`$ and $`n_1`$ denote the number of negatives and positives, $`N = n_0 + n_1`$.

**Assumptions:**
- A1: Independent sampling within each class and between classes, with class sizes fixed by the design. Under prevalence sampling, where the class sizes are random, the theorem holds conditionally on them and therefore unconditionally
- A2: Continuous class score distributions $`F`$ (negatives) and $`G`$ (positives). No densities, moments, common support, or continuity of the ROC are required. Tied scores are handled by randomized tie-breaking (§2.8), which redefines the estimand
- A3: Higher scores indicate the positive class
- A4: The scoring rule is fixed independently of the evaluation sample

**Notation.** Write $`A(x) = 1 - F(x)`$ and $`B(x) = 1 - G(x)`$ for the class survival functions. Order each class by *descending* score: $`X_{[1]} > X_{[2]} > \cdots > X_{[n_0]}`$ and $`Y_{[1]} > \cdots > Y_{[n_1]}`$.

**Estimand.** With $`Q_F(u) = \inf\{x : F(x) \ge u\}`$ and $`Q_F(0) = -\infty`$,

```math
R(t) = B\big(Q_F(1-t)\big), \qquad t \in [0,1].
```

$`R`$ is nondecreasing with $`R(1) = 1`$. It can jump, and $`R(0) > 0`$ when positives have mass above the negative support.

**Target:** Construct $`\mathcal{B}_\alpha(t) = [L(t), U(t)]`$ such that, for **every** $`F`$, $`G`$ satisfying A2 and every $`n_0, n_1 \ge 1`$:

```math
P\left(\forall t \in [0,1]: R(t) \in \mathcal{B}_\alpha(t)\right) \geq 1 - \alpha
```

---

## 2. Algorithm

### 2.1 Merged Label Sequence

Sort the pooled sample by descending score and record the labels. For $`i = 1, \ldots, n_0`$, let $`p_i`$ count the positives ranked above $`X_{[i]}`$, and set $`p_0 = 0`$. The vector $`(p_i)`$ is the only data the band reads.

**Key Insight (exact pivots).** For a continuous CDF, the survival probabilities at a sample's own descending order statistics are ascending uniform order statistics:

```math
\big(A(X_{[1]}), \ldots, A(X_{[n_0]})\big) \overset{d}{=} \big(U_{(1)}, \ldots, U_{(n_0)}\big), \qquad A(X_{[i]}) \sim \text{Beta}(i,\, n_0 + 1 - i),
```

and likewise for $`B(Y_{[j]})`$ with $`n_1`$. This holds exactly at every sample size and for every continuous law (theory Prop. 3). The two vectors are independent by A1. They are pivots: their joint law is known, and the unknown $`F`$, $`G`$ enter only through the values they are evaluated at.

### 2.2 Evaluation Grid

The edge formulas of §2.6 are defined at every $`t \in [0,1]`$. The implementation evaluates them on the native grid $`\mathcal{T} = \{k/n_0 : k = 0, \ldots, n_0\}`$. Both edges are nondecreasing step functions whose jumps occur at Beta quantiles of the negative class, so they generally fall between grid points. The grid band extends to the continuum without loss of coverage by holding $`L`$ from the grid point on the left and $`U`$ from the grid point on the right (theory §1). The exact theorem is stated for the continuum edges, and a grid evaluation with this extension contains them.

If a coarser output grid $`\text{linspace}(0,1,K)`$ is requested, the edges are resampled by the same rule (lower from $`\lfloor t n_0 \rfloor`$, upper from $`\lceil t n_0 \rceil`$).

### 2.3 One-Sample ELL Bounds

For a class of size $`n`$ and a local level $`\gamma \in (0, 1/2)`$, bound the $`i`$-th pivot by the quantiles of its own Beta law:

```math
b^-_i(\gamma) = \text{Beta}^{-1}(\gamma;\; i,\; n+1-i), \qquad b^+_i(\gamma) = \text{Beta}^{-1}(1-\gamma;\; i,\; n+1-i), \qquad i = 1, \ldots, n.
```

Both sequences are nondecreasing in $`i`$. The simultaneous event is

```math
E(\gamma) = \Big\{ b^-_i(\gamma) \le U_{(i)} \le b^+_i(\gamma) \ \text{ for all } i \Big\},
```

and its probability $`P_n(\gamma) = P(E(\gamma))`$ depends only on $`n`$ and $`\gamma`$. Each order statistic has the same marginal two-sided error $`2\gamma`$: the band is uniform on the scale of the pivots' own laws, not on the probability scale. This is the equal-local-levels principle of Berk and Jones (1979), Owen (1995), and Aldor-Noiman et al. (2013).

### 2.4 Exact Calibration

**Crossing probability.** $`P_n(\gamma)`$ is a boundary non-crossing probability for uniform order statistics. It is computed exactly by sweeping the sorted union of the $`2n`$ bound values with the counting process $`N(c) = \#\{j : U_j \le c\}`$ (Noé 1972; Moscovich and Nadler 2017):

- The constraints are $`N(b^-_i) \le i - 1`$ (from $`U_{(i)} \ge b^-_i`$) and $`N(b^+_i) \ge i`$ (from $`U_{(i)} \le b^+_i`$); ties have probability zero.
- Conditional on $`N(c) = m`$, the remaining $`n - m`$ points are iid uniform on $`(c, 1)`$, so between consecutive breakpoints $`c < c'`$ the count increases by $`\text{Binomial}\big(n-m,\; (c'-c)/(1-c)\big)`$.
- Because both bound sequences are nondecreasing, the live states at each breakpoint form a contiguous window of counts. Mass that would violate the next lower-bound constraint is dropped; states that violated a passed upper-bound constraint are removed.

Every intermediate quantity is a probability in $`[0,1]`$ formed from nonnegative terms, so there is no cancellation.

**Calibration of $`\gamma`$.** For a class level $`\alpha_c`$, choose the largest $`\gamma`$ with $`P_n(\gamma) \ge 1 - \alpha_c`$. $`P_n`$ is nonincreasing in $`\gamma`$, since the bounds nest. Bisection runs on $`\log \gamma`$ over the bracket

```math
\gamma_{\text{lo}} = \frac{\alpha_c}{2n}, \qquad \gamma_{\text{hi}} = \frac{\alpha_c}{2}.
```

$`\gamma_{\text{lo}}`$ is valid by the union bound. Any $`\gamma > \gamma_{\text{hi}}`$ fails at a single order statistic, since $`P_n(\gamma) \le 1 - 2\gamma`$. The target is $`1 - \alpha_c + 10^{-9}`$; the margin absorbs floating-point roundoff. Bisection stops when $`\gamma_{\text{hi}}/\gamma_{\text{lo}} < 1 + 10^{-4}`$ and returns the **conservative endpoint** $`\gamma_{\text{lo}}`$, so the computed coverage of each class band is at least its target. If even the Bonferroni level misses the margin-raised target (possible only when the union bound is tight, e.g. $`n = 1`$), the Bonferroni level is returned; its validity needs no numerics. Results are cached per $`(n, \alpha_c)`$.

Calibrated levels for balanced classes ($`\alpha_c = 1 - \sqrt{1-\alpha}`$):

| $`\alpha`$ | $`n`$ | $`\gamma`$ | $`\Phi^{-1}(1-\gamma)`$ | $`\gamma \,/\, (\alpha_c/2n)`$ |
|---|---|---|---|---|
| 0.05 | 30 | $`8.9 \times 10^{-4}`$ | 3.12 | 2.1 |
| 0.05 | 100 | $`5.1 \times 10^{-4}`$ | 3.29 | 4.0 |
| 0.05 | 1,000 | $`2.4 \times 10^{-4}`$ | 3.49 | 19 |
| 0.05 | 10,000 | $`1.5 \times 10^{-4}`$ | 3.61 | 120 |
| 0.5 | 100 | $`9.7 \times 10^{-3}`$ | 2.34 | 6.6 |
| 0.5 | 1,000 | $`5.0 \times 10^{-3}`$ | 2.58 | 34 |
| 0.5 | 10,000 | $`3.2 \times 10^{-3}`$ | 2.73 | 217 |

The last column is the gain over Bonferroni, which the dependence among order statistics makes large. The local level decays slowly in $`n`$, roughly like $`\alpha_c / \log n`$ rather than $`\alpha_c / n`$.

### 2.5 Class Split

The two class levels must satisfy $`(1-\alpha_0)(1-\alpha_1) = 1 - \alpha`$. With split ratio $`\rho \in (0,1)`$,

```math
1 - \alpha_0 = (1-\alpha)^\rho, \qquad 1 - \alpha_1 = (1-\alpha)^{1-\rho}.
```

The default is the sample-size split

```math
\rho(n_0, n_1) = \frac{n_0^{-1/2}}{n_0^{-1/2} + n_1^{-1/2}},
```

which gives the smaller class the larger share of the error ($`\rho = 0.75`$ at 100/900) and reduces to the Šidák split $`\rho = 1/2`$ when $`n_0 = n_1`$. A class band's half-width scales as $`n^{-1/2}`$, so this spends error where it buys the most width. Any $`\rho`$ that depends only on the class sizes preserves the theorem, because the design fixes them; a $`\rho`$ chosen from the ranks does not. In a screen of 12 shapes and 13 class-size configurations it was narrower than Šidák in nearly every imbalanced cell (1–5% at 1:9 to 1:30, 5–10% at 1:100) and at most 0.85% wider elsewhere (`next_method_ideas.md` §4.3).

Write $`b^\pm_{0,i}`$ for the negative-class bounds at level $`\gamma_0 = \gamma(n_0, \alpha_0)`$ and $`b^\pm_{1,j}`$ for the positive-class bounds at $`\gamma_1 = \gamma(n_1, \alpha_1)`$, with sentinels $`b^-_{1,0} = 0`$ and $`b^+_{1,n_1+1} = 1`$.

### 2.6 Composition

For $`t \in [0,1]`$ define the two negative indices

```math
i_U(t) = \min\{i \ge 1 : b^-_{0,i} \ge t\}, \qquad i_L(t) = \#\{i : b^+_{0,i} < t\} = \max\big(\{i : b^+_{0,i} < t\} \cup \{0\}\big),
```

and the edges

```math
U(t) = \begin{cases} b^+_{1,\; p_{i_U(t)} + 1}, & \text{if } i_U(t) \text{ exists}, \\ 1, & \text{otherwise}, \end{cases} \qquad\qquad L(t) = b^-_{1,\; p_{i_L(t)}}.
```

In words: $`i_U(t)`$ is the highest-ranked negative whose true FPR is certified to be at least $`t`$, so the operating threshold for FPR $`t`$ lies at or above it, and the true TPR there is at most the certified upper bound for the next positive below it. $`i_L(t)`$ is the lowest-ranked negative whose true FPR is certified to be below $`t`$, so the threshold lies at or below it, and the true TPR is at least the certified lower bound for the last positive above it. Both edges are index gathers on precomputed arrays.

**Equivalently** (theory §12.1), the band is the pointwise outer envelope of the ROC curves of *all* pairs of monotone class CDFs consistent with the two class bands and the observed ranks. It is also the hull of the full monotone-completion brackets of every spacing pair in an auxiliary set fixed before the labels are seen, a product of two order-statistic rectangles (theory Prop. 12g). The fiducial band uses the same brackets but lets the labels choose the set; that difference is what removes its guarantee.

### 2.7 Boundary Handling

**Boundary Enforcement:**
- $`L(0) = 0`$
- $`U(1) = L(1) = 1`$, valid because $`R(1) = 1`$ for every continuous pair

The upper edge at $`t = 0`$ is **not** pinned. It is the composition's own value $`U(0) = b^+_{1,\, p_1 + 1}`$, which is positive and allows $`R(0) > 0`$. Pinning $`U(0) = 0`$ is invalid without a support assumption (theory Cor. 9b). A user who can assert $`R(0) = 0`$ may opt into the pin (`assume_r0_zero=True`); the theorem then holds under that additional assumption only.

No clipping is needed: every edge value is a Beta quantile or a sentinel in $`[0,1]`$.

### 2.8 Ties

The default breaks tied scores uniformly at random, independently of class, which is equivalent to attaching an independent continuous uniform to every observation and ordering lexicographically. The augmented class distributions are continuous, so the theorem applies exactly. Its estimand is the ROC of the threshold rule that randomizes within tied scores: linear tie segments and AUC $`P(Y > X) + \tfrac{1}{2} P(Y = X)`$ (theory Prop. 2a). The deterministic alternative (`"even"`) spreads each class evenly within tied blocks and has no theorem. Breaking ties by class changes the estimand and is not offered. The tie-breaking draws are the band's only randomness.

---

## 3. Properties

### 3.1 Coverage Guarantees

**Definition (Population Coverage).** The probability that the true population ROC curve falls entirely within the band:
```math
P\left(\forall t: R(t) \in [L(t), U(t)]\right)
```

**Note on Future Samples:** This method constructs a *confidence band* for $`R`$. It is **not** a *prediction band* for a future empirical ROC $`\hat{R}_{\text{new}}`$. A fresh empirical curve adds its own sampling error, approximately doubling the covariance, so its coverage is far below $`(1-\alpha)`$.

---

**Lemma 1 (Composition).** Let $`E_0^- = \{A(X_{[i]}) \ge b^-_{0,i}\ \forall i\}`$, $`E_0^+ = \{A(X_{[i]}) \le b^+_{0,i}\ \forall i\}`$, and define $`E_1^\pm`$ in the same way for $`B(Y_{[j]})`$. Almost surely:
1. On $`E_0^- \cap E_1^+`$, $`R(t) \le U(t)`$ for every $`t \in [0,1]`$.
2. On $`E_0^+ \cap E_1^-`$, $`R(t) \ge L(t)`$ for every $`t \in [0,1]`$.

*Proof.* (1) Fix $`t`$. If $`i_U(t)`$ does not exist, $`U(t) = 1`$. Otherwise put $`i = i_U(t)`$ and $`p = p_i`$. On $`E_0^-`$, $`A(X_{[i]}) \ge b^-_{0,i} \ge t`$. If $`A(X_{[i]}) > t`$, every $`x`$ with $`A(x) \le t`$ exceeds $`X_{[i]}`$, so $`Q_F(1-t) = \inf\{x : A(x) \le t\} \ge X_{[i]}`$. Equality $`A(X_{[i]}) = t`$ with $`Q_F(1-t) < X_{[i]}`$ would place $`X_{[i]}`$ at the right end of an interval of zero $`F`$-mass, which has probability zero. Since $`B`$ is nonincreasing, $`R(t) = B(Q_F(1-t)) \le B(X_{[i]})`$. Exactly $`p`$ positives outrank $`X_{[i]}`$, so $`Y_{[p+1]} < X_{[i]}`$ when $`p < n_1`$, and $`B(X_{[i]}) \le B(Y_{[p+1]}) \le b^+_{1,p+1}`$ on $`E_1^+`$. When $`p = n_1`$ the sentinel gives $`U(t) = 1`$.

(2) Fix $`t`$ and put $`i = i_L(t)`$. If $`i = 0`$, $`L(t) = 0`$. Otherwise, on $`E_0^+`$, $`A(X_{[i]}) \le b^+_{0,i} < t`$, so $`X_{[i]} \in \{x : A(x) \le t\}`$ and $`Q_F(1-t) \le X_{[i]}`$. Hence $`R(t) \ge B(X_{[i]})`$. The $`p = p_i`$ positives above $`X_{[i]}`$ include $`Y_{[p]}`$, so if $`p \ge 1`$, $`B(X_{[i]}) \ge B(Y_{[p]}) \ge b^-_{1,p}`$ on $`E_1^-`$. When $`p = 0`$ the sentinel gives $`L(t) = 0`$. $`\square`$

The lemma is deterministic given the pivot events. It uses no continuity of $`R`$, holds at every real $`t`$, and covers jumps of $`R`$ and $`R(0) > 0`$.

---

**Theorem 1 (Finite-Sample Coverage).** Under A1–A2, for every $`F`$, $`G`$ and every $`n_0, n_1 \ge 1`$:

```math
P\left(\forall t \in [0,1]: L(t) \le R(t) \le U(t)\right) \;\ge\; P_{n_0}(\gamma_0)\, P_{n_1}(\gamma_1) \;\ge\; (1-\alpha_0)(1-\alpha_1) \;=\; 1 - \alpha.
```

*Proof.* $`E_0 = E_0^- \cap E_0^+`$ and $`E_1 = E_1^- \cap E_1^+`$ are the class ELL events of §2.3 evaluated at the true pivots, so $`P(E_c) = P_{n_c}(\gamma_c) \ge 1 - \alpha_c`$ by the calibration. On $`E_0 \cap E_1`$ both parts of Lemma 1 hold. The pivots of the two classes are independent, so $`P(E_0 \cap E_1) = P(E_0) P(E_1)`$. $`\square`$

Nothing in the argument is asymptotic or simulated. It needs no density, slope, variance estimate, or smoothness of $`R`$. The pivot laws do not depend on $`F`$ or $`G`$, so the bound is uniform over the entire continuous class.

**Remark (numerical exactness).** The theorem concerns the bounds actually used. The calibration and the band compute the bound arrays by the same deterministic double-precision Beta-quantile routine, and the dynamic program evaluates the crossing probability of exactly those arrays. The only numerical error is roundoff inside the DP, which sums and multiplies nonnegative terms. Its worst-case relative error grows like the operation count times machine epsilon, about $`10^{-8}`$ at $`n = 10^4`$. The $`10^{-9}`$ margin covers typical accumulation, and any residual shortfall in the class coverage is far below $`10^{-7}`$. Tie-breaking (§2.8) is the only randomness in the band.

---

**Proposition 2 (One-sided guarantee).** *[Exact; derived here.]* Under A1–A2, for every $`F`$, $`G`$, $`n_0`$, $`n_1`$:

```math
P\big(\exists t: R(t) > U(t)\big) \le 1 - \sqrt{1-\alpha}, \qquad P\big(\exists t: R(t) < L(t)\big) \le 1 - \sqrt{1-\alpha}.
```

At $`\alpha = .05`$ both bounds equal $`0.0253`$.

*Proof.* By Lemma 1, $`P(\exists t: R > U) \le 1 - P(E_0^-)\,P(E_1^+)`$. The vector of uniform order statistics is a nondecreasing function of independent uniforms and is therefore associated (Esary, Proschan and Walkup 1967). $`E_c^-`$ is an increasing event and $`E_c^+`$ a decreasing one, so $`P(E_c) = P(E_c^- \cap E_c^+) \le P(E_c^-) P(E_c^+)`$. The reflection $`U \mapsto 1 - U`$ maps $`U_{(i)}`$ to $`1 - U_{(n+1-i)}`$, and the ELL bounds satisfy $`b^+_i = 1 - b^-_{n+1-i}`$, so $`P(E_c^-) = P(E_c^+)`$. Hence each is at least $`\sqrt{P(E_c)} \ge \sqrt{1-\alpha_c}`$, and $`P(E_0^-) P(E_1^+) \ge \sqrt{(1-\alpha_0)(1-\alpha_1)} = \sqrt{1-\alpha}`$. The lower edge is symmetric. $`\square`$

The theorem therefore does more than bound total error. Neither edge can absorb much more than $`\alpha/2`$, whatever the ROC shape. It does not make the two directions equal; in practice both are far below their bounds (§3.7).

---

### 3.2 Conservatism

The band's realized coverage is well above nominal, and more so at large $`\alpha`$. In the main suite the 95% band covers 0.999–1.000 and the 50% band 0.95–0.99 (§3.7). The nominal level at which the band realizes coverage 0.95 is 0.50–0.85 across 14 smooth cells (`research_record.md` Q5; Šidák split, measured with an earlier Monte Carlo calibration of $`\gamma`$). There are three sources, and none of them is Monte Carlo error or loose calibration: the class events are calibrated exactly.

1. **Projection.** Covering both class CDFs is sufficient but not necessary for covering their ROC composition. A class-band failure at an order statistic far from where the other class's uncertainty matters causes no ROC miss. Where the curve is steep, the negative class dominates and the positive class's error budget is largely unused; where it is flat, the reverse.
2. **Unpaired corners.** The upper edge reads only $`(E_0^-, E_1^+)`$ and the lower edge only $`(E_0^+, E_1^-)`$ (Lemma 1). The two other combinations move both CDFs the same way, and their effects on the ROC largely cancel. The rectangle $`E_0 \cap E_1`$ still spends probability on them (theory §12.1a).
3. **Level insensitivity.** The class calibration must protect $`n`$ order statistics jointly, so $`\gamma`$ responds weakly to $`\alpha`$. At $`n = 1{,}000`$ the normal score of $`\gamma`$ is 3.49 at $`\alpha = .05`$ and 2.58 at $`\alpha = .5`$, so the 50% band is only about 26% narrower than the 95% band in the interior. Sources 1 and 2 then add coverage at every level.

**Limits on recovering it inside a theorem.** M3 is the square $`[0, c]^2`$ in the plane of the two one-sided class excursion statistics. Replacing the square by any lower set of the same probability preserves the theorem (theory Prop. 12h). In the Gaussian interior the half-width of such a band at a point where the two classes contribute equally cannot fall below a computable ceiling (theory eq. 12.1c). In units of the local standard deviation:

| $`n_0 = n_1`$, $`\alpha`$ | ELL (square) | Best possible region | Fiducial tube |
|---|---|---|---|
| 50, .05 | 4.53 | 3.57 | 2.90 |
| 500, .05 | 4.86 | 3.96 | 3.16 |
| 500, .5 | 3.56 | 2.98 | 2.18 |

No composition of two separately protected class CDFs can be more than about 16–21% narrower than ELL there, and the fiducial tube lies 20–27% below that ceiling. Where one class dominates, the square is already the best region. The remaining gap is the price of calibrating on two class maxima that do not know which negative rank faces which positive rank. Closing it requires inference on the ROC directly, by test inversion (theory §12.2). Within deterministic boundary families (ELL–KS interpolations, tail modifiers, class-split exponents), each calibrated exactly, no member beat ELL on every training shape without widening a tail (`research_record.md` Q5).

### 3.3 Width and Tail Adaptivity

The KS band (Campbell 1994) is the same composition with a constant class margin $`c/\sqrt{n}`$, $`c = \sqrt{\log(2/\alpha_c)/2}`$ (Dvoretzky–Kiefer–Wolfowitz with Massart's constant). ELL bounds follow the Beta law of each order statistic, with half-width approximately $`\Phi^{-1}(1-\gamma)\sqrt{u(1-u)/n}`$ at quantile $`u`$ in the body and of order $`\log(1/\gamma)/n`$ at the ends. They are much narrower than KS in the tails and somewhat wider in the middle. Class-band half-widths at $`\alpha = .05`$, $`n = 1{,}000`$:

| Order statistic $`i`$ | 1 | 10 | 100 | 500 |
|---|---|---|---|---|
| ELL | 0.0041 | 0.011 | 0.033 | 0.055 |
| KS | 0.047 | 0.047 | 0.047 | 0.047 |

At the median the ELL bound is 1.08–1.22 times the KS margin for $`n = 100`$–$`10{,}000`$ at $`\alpha = .05`$, and 1.17–1.39 times at $`\alpha = .5`$. ELL therefore has no universal width dominance over KS. Its advantage is largest at high AUC, where the ROC's uncertainty is concentrated in the class tails, and smallest near the diagonal and at small $`n`$. In the main suite its mean area is 0.63–0.99 times KS's in every cell, but it is wider than KS on 2.4%, 2.1%, and 0.1% of individual datasets at 50%, 30%, and 10% prevalence ($`\alpha = .05`$).

**Proposition 3 (Separated sample).** *[Exact; derived here.]* On the perfectly separated rank path, where every positive outranks every negative,

```math
U(t) = 1 \ \ \forall t, \qquad L(t) = \begin{cases} 0, & t \le 1 - \gamma_0^{1/n_0}, \\ \gamma_1^{1/n_1}, & 1 - \gamma_0^{1/n_0} < t < 1, \end{cases} \qquad \int_0^1 (U - L)\,dt = 1 - \gamma_0^{1/n_0}\gamma_1^{1/n_1}.
```

*Proof.* Here $`p_i = n_1`$ for all $`i \ge 1`$, so $`U(t) = b^+_{1,n_1+1} = 1`$. For $`L`$, $`i_L(t) = 0`$ iff $`t \le b^+_{0,1} = 1 - \gamma_0^{1/n_0}`$, and otherwise $`L(t) = b^-_{1,n_1} = \gamma_1^{1/n_1}`$, the $`\text{Beta}(n_1, 1)`$ quantile. $`\square`$

Every deterministic rank-only band honest over all continuous distributions must satisfy $`U = 1`$, $`L(t) \le \alpha^{1/n_1}`$ for $`t < 1`$, and $`L(t) = 0`$ for $`t < 1 - \alpha^{1/n_0}`$ on this path, so its area is at least $`1 - \alpha^{1/n_0 + 1/n_1}`$ (theory Cor. 9a–9b). ELL meets all three conditions, since $`\gamma_c < \alpha`$. For balanced classes, its area exceeds the lower bound by a factor approaching $`\log(1/\gamma)/\log(1/\alpha)`$: 2.1–2.9 at $`\alpha = .05`$ for $`n = 10`$–$`5{,}000`$ per class. At 100 per class the area is 0.141 against the bound of 0.058. The unfloored fiducial band violates the third condition (`fiducial_band.md` Prop. 2).

### 3.4 Asymmetry and Directional Behavior

The class bounds are asymmetric around each pivot's mean wherever its Beta law is skewed, at both ends of each class. The composed band inherits this. Near TPR 1 the upper edge is pinned by the sentinel while the lower edge extends down by about $`\log(1/\gamma_1)/n_1`$. Near FPR 0 the lower edge is held at 0 until $`t`$ exceeds $`b^+_{0,1} \approx \log(1/\gamma_0)/n_0`$. These are exactly the regions where an honest band must leave room for unobserved mass (theory §9).

Proposition 2 caps each direction at $`1 - \sqrt{1-\alpha}`$, but the realized rates need not be equal. In the main suite misses are rare in both directions at $`\alpha = .05`$, and above-band misses are more frequent than below-band misses at $`\alpha = .5`$, increasingly so at low prevalence (§3.7).

### 3.5 Structural Properties

- *Deterministic and rank-invariant.* Given the tie-breaking draws, the band is a function of $`(p_i)`$, $`\alpha`$, and the class sizes. Its coverage depends on the data-generating process only through $`(R, n_0, n_1)`$.
- *Monotone and ordered.* $`i_U`$, $`i_L`$, $`(p_i)`$, and both bound sequences are nondecreasing, so $`L`$ and $`U`$ are nondecreasing. Since $`b^+_{0,i} \ge b^-_{0,i}`$, $`i_L(t) \le i_U(t) - 1`$, which gives $`L(t) \le U(t)`$.
- *Nested in $`\alpha`$.* $`P_n(\gamma)`$ is nonincreasing in $`\gamma`$ and the class levels increase with $`\alpha`$, so bands at larger $`\alpha`$ lie inside bands at smaller $`\alpha`$, up to the $`10^{-4}`$ relative bisection tolerance.
- *Step-function structure.* $`U`$ jumps where $`t`$ crosses a lower negative bound $`b^-_{0,i}`$ at which $`p_i`$ changes; $`L`$ jumps where $`t`$ crosses an upper bound $`b^+_{0,i}`$. Between jumps both edges are constant.

---

### 3.6 Summary of Guarantees

| Property | Finite Sample | Asymptotic |
|----------|---------------|------------|
| Population coverage, every continuous $`F`$, $`G`$ | ✓ $`\ge 1-\alpha`$, exact (Thm. 1) | ✓ |
| One-sided miss probability | ✓ $`\le 1-\sqrt{1-\alpha}`$ each (Prop. 2) | ✓ |
| Jumps in $`R`$, $`R(0) > 0`$, support gaps | ✓ covered by the same theorem | ✓ |
| Ties | ✓ for the randomized (trapezoidal) ROC | ✓ |
| Calibration | ✗ Conservative; 0.95–0.99 at nominal 0.50 | ✗ Conservative (projection and unpaired corners persist) |
| Width adapts to local variance | Partly: tails adapt through the Beta laws; the interior pays both classes' errors additively | Half-width $`\approx z_{\gamma_1} a_t + z_{\gamma_0} b_t`$, with $`a_t`$, $`b_t`$ the vertical and horizontal standard errors and $`z_\gamma = \Phi^{-1}(1-\gamma)`$, against $`c_\alpha\sqrt{a_t^2 + b_t^2}`$ for the direct ROC process |
| Narrower than KS | Usually, not uniformly (§3.3) | ✓ at high AUC |
| Monte Carlo error | None (deterministic given ranks) | None |
| Future-curve coverage | $`\ll 1-\alpha`$ (not covered) | $`\ll 1-\alpha`$ |

### 3.7 Simulation Evidence

The main suite ([`simulation_spec.md`](simulation_spec.md)) draws 6,000 Latin-hypercube parameter points per family for seven score families (binormal, heteroscedastic Gaussian, logit-normal, opposing-skew Beta, Student-$`t`$, bimodal negatives, Weibull), with AUC uniform on the probit scale over $`[0.55, 0.99]`$. One dataset is drawn per point at each total sample size $`n \in \{10, 30, 100, 300, 1{,}000, 3{,}000, 10{,}000\}`$, at 50%, 30%, and 10% prevalence ($`n = 10`$ at 50% only), and bands are computed at $`\alpha \in \{.05, .2, .5\}`$. A cell is one family at one $`n`$ (about 6,000 datasets). Coverage is checked on the native grid excluding the pinned endpoints $`t = 0, 1`$. Area is relative to the KS band on the same datasets. These are confirmations of an exact theorem, not evidence for it; their use is to measure conservatism and width. All rows use $`n \ge 30`$ (42 cells).

| Prevalence | Level | Mean coverage | Worst cell | Miss below | Miss above | Area ÷ KS |
|---|---|---|---|---|---|---|
| 50% | 95% | 1.000 | 0.999 | 0.0001 | 0.0001 | 0.78 |
| 50% | 80% | 0.998 | 0.996 | 0.0007 | 0.0012 | 0.80 |
| 50% | 50% | 0.987 | 0.973 | 0.0055 | 0.0081 | 0.83 |
| 30% | 95% | 1.000 | 0.999 | 0.0001 | 0.0002 | 0.80 |
| 30% | 80% | 0.997 | 0.992 | 0.0011 | 0.0024 | 0.80 |
| 30% | 50% | 0.979 | 0.966 | 0.0074 | 0.0136 | 0.82 |
| 10% | 95% | 0.999 | 0.997 | 0.0003 | 0.0009 | 0.82 |
| 10% | 80% | 0.991 | 0.983 | 0.0026 | 0.0068 | 0.83 |
| 10% | 50% | 0.952 | 0.929 | 0.0144 | 0.0334 | 0.83 |

Against $`n`$ at 50% prevalence, mean over the seven families:

| | $`n=10`$ | 30 | 100 | 300 | 1,000 | 3,000 | 10,000 |
|---|---|---|---|---|---|---|---|
| Coverage, 50% band | 0.998 | 0.994 | 0.993 | 0.989 | 0.984 | 0.981 | 0.978 |
| Area ÷ KS, 95% band | 0.91 | 0.98 | 0.88 | 0.79 | 0.73 | 0.69 | 0.66 |

At $`n \ge 30`$ and 50% prevalence the ELL band has about 1.56 times the mean area of the unfloored fiducial band at $`\alpha = .05`$ (0.280 against 0.179). The ratio is largest at small $`n`$, where the ELL band is close to KS, and falls with $`n`$. Below about 50 per class the fiducial hybrid's floor region covers essentially the whole grid, and the hybrid coincides with the ELL band (`research_record.md` Q3).

---

## 4. Computational Considerations

### 4.1 Complexity

| Operation | Cost |
| :--- | :--- |
| Sort and tie-break | $`O(N \log N)`$ |
| Beta quantile vectors, per class | $`O(n_c)`$ |
| One crossing-probability evaluation | $`O(\sum_s w_s^2)`$ over $`2n`$ breakpoints, with live-window width $`w_s`$; about $`O(n^2)`$ |
| Calibration of $`\gamma`$ | about 15–17 bisection steps ($`\log_2(\ln n / 10^{-4})`$), each one evaluation |
| Composition | $`O(n_0 + n_1)`$ index gathers |

**Total per dataset after calibration:** $`O(N \log N)`$.

Calibration depends only on $`(n_c, \alpha_c)`$ and is cached in-process; a parent process can calibrate once and pass the cache to workers. Measured calibration time per class size and level is about 0.01–0.02 s at $`n \le 100`$, 0.3–0.6 s at $`n = 1{,}000`$, and 30–50 s at $`n = 10{,}000`$. A calibrated band on 10,000 observations takes about 2.5 ms, and the median time per dataset in the main suite was at most 4 ms at every size.

At large $`n`$ the calibration dominates the first call. §4.4 describes how a precomputed table of levels reduces it to about one crossing-probability evaluation.

### 4.2 Memory

$`O(n_0 + n_1)`$: two bound vectors per class and the positive-count vector. The DP holds one window of at most $`n + 1`$ probabilities.

### 4.3 Parameters

- $`\alpha`$: significance level.
- $`\rho`$: class split (default `"sample_size"`; any value fixed by the class sizes preserves the theorem).
- `assume_r0_zero`: optional pin $`U(0) = 0`$, valid only under the stated support assumption.

There are no Monte Carlo budgets, smoothing parameters, or shape parameters.

### 4.4 Warm-Started Calibration from a Table of Levels (Proposed)

*This section specifies an acceleration that is not implemented. The current implementation runs the full bisection of §2.4 on every new $`(n, \alpha_c)`$.*

**Validity does not depend on the estimate.** Theorem 1 needs only a level $`\gamma`$ whose class event is certified: $`P_n(\gamma) \ge 1 - \alpha_c`$, evaluated by the exact recursion. How a candidate $`\gamma`$ was proposed is irrelevant to that check. An estimate of the calibrated level $`\gamma^*(n, \alpha_c)`$ therefore affects only the cost of calibration and, slightly, the width. It cannot affect coverage, provided every returned level passes the exact check.

**The table.** Precompute $`\gamma^*`$ offline by the full calibration of §2.4 on a grid of $`(n, \alpha_c)`$. Index by the class level $`\alpha_c`$, not by $`\alpha`$: with the sample-size split, $`\alpha_c = 1 - (1-\alpha)^\rho`$ takes a continuum of values for each $`\alpha`$. Store and interpolate in coordinates where $`\gamma^*`$ is nearly flat:

```math
x = \log n, \qquad y = \log \alpha_c, \qquad g(x, y) = \log\frac{\gamma^*\, \log n\, \log\log n}{\alpha_c}.
```

On the levels of §2.4, $`\gamma^* \log n \log\log n / \alpha_c`$ is about 0.14, 0.13, and 0.12 at $`n = 100`$, $`1{,}000`$, and $`10{,}000`$ ($`\alpha = .05`$), so $`g`$ varies slowly in both coordinates. Use log-spaced $`n`$ and $`\alpha_c`$ nodes, and interpolate $`g`$ (bilinear or bicubic) to obtain the prediction $`\hat\gamma`$. Below a small size $`n_{\min}`$ (a few hundred), direct calibration costs under 0.1 s and no table is needed.

**Choosing the grid density.** Let $`\varepsilon`$ bound the interpolation error on the table's domain $`\mathcal{G}`$:

```math
\big|\log\hat\gamma(n, \alpha_c) - \log\gamma^*(n, \alpha_c)\big| \le \varepsilon \qquad \text{for } (n, \alpha_c) \in \mathcal{G}.
```

Estimate $`\varepsilon`$ by calibrating exactly at held-out points, such as cell midpoints, and comparing them with the interpolant. Increase the density until the largest held-out error is comfortably below the shading margin $`s`$ used below. Held-out checks sample $`\mathcal{G}`$ and do not prove the bound everywhere. A point where it fails costs extra evaluations, not validity.

**Warm-started bisection, with a step bound.** For $`(n, \alpha_c) \in \mathcal{G}`$ and a shading margin $`s > \varepsilon`$, take the bracket

```math
\gamma_{\text{lo}} = \hat\gamma\, e^{-s}, \qquad \gamma_{\text{hi}} = \hat\gamma\, e^{s}.
```

If the error bound holds at this point, the bracket contains $`\gamma^*`$. Check the invariant $`P_n(\gamma_{\text{lo}}) \ge \tau > P_n(\gamma_{\text{hi}})`$, with $`\tau = 1 - \alpha_c + 10^{-9}`$ as in §2.4. Then bisect on $`\log\gamma`$ to the same relative tolerance $`\gamma_{\text{hi}}/\gamma_{\text{lo}} < 1 + 10^{-4}`$. Bisection halves the log-width $`2s`$ at every step, so it stops after at most

```math
k(s) = \left\lceil \log_2 \frac{2s}{\log(1 + 10^{-4})} \right\rceil
```

further evaluations, in addition to the two bracket checks. This count is a worst case, not an expectation: $`k = 8`$ at $`s = 0.01`$ and $`k = 5`$ at $`s = 0.001`$, against about 15–17 steps for the full bracket $`[\alpha_c/2n, \alpha_c/2]`$. A secant step usually converges faster but has no comparable worst-case bound.

**Accepting the shaded prediction after one check.** The bisection refines a level whose width effect is already negligible. The half-width of a class bound at local level $`\gamma`$ is proportional to $`z = \Phi^{-1}(1-\gamma)`$, and by the Mills ratio $`d z / d\log\gamma \approx -1/z`$. Using $`\gamma^* e^{-s}`$ in place of $`\gamma^*`$ therefore widens each class bound by a relative amount of about

```math
\frac{\Delta z}{z} \approx \frac{s}{z^2}.
```

At $`\alpha = .05`$ ($`z \approx 3.5`$) with $`s = 0.01`$, that is under 0.1%; at $`\alpha = .5`$ ($`z \approx 2.6`$) it is about 0.15%. The cheapest valid rule evaluates $`P_n`$ once, at $`\gamma_{\text{lo}} = \hat\gamma e^{-s}`$, and returns that level if it passes. One evaluation is about $`1/16`$ of the present calibration cost, roughly 2–3 s at $`n = 10{,}000`$ with the current recursion.

**Failure and out-of-domain handling.** If $`P_n(\gamma_{\text{lo}}) < \tau`$, the prediction was too large by more than $`s`$. Step the lower end down geometrically, $`\gamma_{\text{lo}} \leftarrow \gamma_{\text{lo}} e^{-2^m s}`$ for $`m = 1, 2, \ldots`$, until it passes, or fall back to the full bracket; stop the descent at the Bonferroni level $`\alpha_c/2n`$, which is valid without numerics. If $`P_n(\gamma_{\text{hi}}) \ge \tau`$, extend the upper end in the same way, capped at $`\alpha_c/2`$. Outside $`\mathcal{G}`$, run the full calibration, or extrapolate $`g`$ with a wider margin and accept a larger step count. In every branch the returned level is one whose class event passed the exact check, so Theorem 1 holds unchanged.

```
FUNCTION ell_class_level_warm(n, alpha_c, table, s=0.01):
    tau = 1 - alpha_c + 1e-9
    IF n < n_min OR (n, alpha_c) outside table domain:
        RETURN ell_class_level(n, alpha_c)                  # full bisection, §2.4
    g_hat = interpolate(table, log n, log alpha_c)
    gamma_hat = alpha_c * exp(g_hat) / (log n * log log n)
    lo = gamma_hat * exp(-s)
    IF crossing_prob(n, lo) < tau:                           # prediction too large
        RETURN ell_class_level(n, alpha_c)                  # or geometric descent to alpha_c/(2n)
    RETURN lo                                                # one-check rule
    # Refinement variant: set hi = gamma_hat * exp(s), confirm crossing_prob(n, hi) < tau,
    # then bisect on log gamma for at most k(s) steps and return the conservative endpoint.
```

**Building the table.** The table is a pure function of the recursion and the Beta-quantile routine, so it can be versioned with the code and regenerated. The offline cost is one full calibration per node. With the current $`O(n^2)`$ recursion that is about a minute per node at $`n = 10^4`$ and grows quadratically beyond it. Truncating the binomial transitions of the recursion where their remaining mass is negligible keeps the computed probability a certified lower bound, since all terms are nonnegative and dropped mass can only lower it. That makes nodes at larger $`n`$ affordable; its speedup has not been measured.

---

## 5. Complete Pseudocode

```
FUNCTION ell_class_level(n, alpha_c):
    # Largest gamma with exact crossing probability >= 1 - alpha_c
    target = 1 - alpha_c + 1e-9
    lo = alpha_c / (2 n)          # Bonferroni: valid by the union bound
    hi = alpha_c / 2              # single order statistic: any larger gamma fails
    IF crossing_prob(n, hi) >= target: RETURN hi
    IF crossing_prob(n, lo) < target: RETURN lo
    REPEAT up to 40 times:
        mid = sqrt(lo * hi)
        IF crossing_prob(n, mid) >= target: lo = mid ELSE hi = mid
        IF hi / lo < 1 + 1e-4: BREAK
    RETURN lo                     # conservative endpoint

FUNCTION crossing_prob(n, gamma):
    # Exact P(b_lo[i] <= U_(i) <= b_hi[i] for all i), U iid Uniform(0,1)
    b_lo[i] = beta_quantile(gamma, i, n + 1 - i),     i = 1..n
    b_hi[i] = beta_quantile(1 - gamma, i, n + 1 - i)
    state = {count 0: probability 1}, c_prev = 0
    FOR each breakpoint c in sorted(b_lo ∪ b_hi):
        move mass from count m to m + Binomial(n - m, (c - c_prev)/(1 - c_prev)),
            dropping counts above #(b_lo values < c)         # would violate a lower bound
        drop counts below #(b_hi values <= c)                  # violated an upper bound
        c_prev = c
    RETURN total remaining probability

FUNCTION ell_scb(labels, scores, alpha, split="sample_size"):

    # === Step 1: Rank reduction ===
    lab = labels sorted by descending score, ties broken uniformly at random
    n0 = count(lab == 0), n1 = count(lab == 1)
    p[0] = 0
    p[i] = #positives before the i-th negative in lab,   i = 1..n0

    # === Step 2: Class levels ===
    rho = n0^-0.5 / (n0^-0.5 + n1^-0.5)   IF split == "sample_size" ELSE split
    alpha_0 = 1 - (1 - alpha)^rho
    alpha_1 = 1 - (1 - alpha)^(1 - rho)
    g0 = ell_class_level(n0, alpha_0)      # cached per (n, alpha_c)
    g1 = ell_class_level(n1, alpha_1)

    # === Step 3: Class bounds ===
    b0_lo[i], b0_hi[i] = beta_quantile(g0, i, n0+1-i), beta_quantile(1-g0, i, n0+1-i)
    b1_lo[j], b1_hi[j] = beta_quantile(g1, j, n1+1-j), beta_quantile(1-g1, j, n1+1-j)
    b1_lo[0] = 0, b1_hi[n1 + 1] = 1        # sentinels

    # === Step 4: Composition on the grid ===
    T = [k / n0 for k in 0..n0]
    FOR each t in T:
        iU = 1 + #{i : b0_lo[i] < t}       # first negative certified at FPR >= t
        U[t] = 1 IF iU > n0 ELSE b1_hi[p[iU] + 1]
        iL = #{i : b0_hi[i] < t}           # last negative certified at FPR < t
        L[t] = b1_lo[p[iL]]

    # === Step 5: Boundary anchors ===
    L[0] = 0, L[n0] = 1, U[n0] = 1         # U[0] is not pinned

    RETURN T, L, U
```

---

## 6. Formal Specification

### 6.1 Algorithm 1: Exact ELL Class Calibration

**Input:** Class size $`n`$, class significance $`\alpha_c`$

1. Define $`b^\pm_i(\gamma)`$ as in §2.3 and $`P_n(\gamma) = P\{b^-_i(\gamma) \le U_{(i)} \le b^+_i(\gamma)\ \forall i\}`$.
2. Evaluate $`P_n(\gamma)`$ by the counting-process recursion: process the sorted breakpoints $`c_1 \le c_2 \le \cdots`$; between breakpoints propagate $`P(N(c) = m,\ \text{no violation})`$ by $`\text{Binomial}(n - m, (c' - c)/(1 - c))`$ increments; impose $`N(b^-_i) \le i - 1`$ and $`N(b^+_i) \ge i`$ at each breakpoint.
3. Set $`\tau \leftarrow 1 - \alpha_c + 10^{-9}`$, $`\gamma_{\text{lo}} \leftarrow \alpha_c/(2n)`$, $`\gamma_{\text{hi}} \leftarrow \alpha_c/2`$.
4. If $`P_n(\gamma_{\text{hi}}) \ge \tau`$, return $`\gamma_{\text{hi}}`$. If $`P_n(\gamma_{\text{lo}}) < \tau`$, return $`\gamma_{\text{lo}}`$.
5. While $`\gamma_{\text{hi}}/\gamma_{\text{lo}} \ge 1 + 10^{-4}`$ (at most 40 steps): $`\gamma_m \leftarrow \sqrt{\gamma_{\text{lo}}\gamma_{\text{hi}}}`$; if $`P_n(\gamma_m) \ge \tau`$ then $`\gamma_{\text{lo}} \leftarrow \gamma_m`$, else $`\gamma_{\text{hi}} \leftarrow \gamma_m`$.

**Return:** $`\gamma_{\text{lo}}`$

### 6.2 Algorithm 2: ELL Composition SCB

**Input:** Scores $`\mathcal{S} = \{(y_i, s_i)\}_{i=1}^N`$, significance $`\alpha`$, split ratio $`\rho`$ (default: sample-size rule)

---

#### 1. Rank Reduction
* Break ties by independent uniform keys; sort by descending (score, key).
* $`p_0 \leftarrow 0`$; $`p_i \leftarrow \#\{\text{positives ranked above } X_{[i]}\}`$ for $`i = 1, \ldots, n_0`$.

#### 2. Class Levels
* $`1 - \alpha_0 \leftarrow (1-\alpha)^\rho`$, $`1 - \alpha_1 \leftarrow (1-\alpha)^{1-\rho}`$.
* $`\gamma_0 \leftarrow`$ Algorithm 1$`(n_0, \alpha_0)`$; $`\gamma_1 \leftarrow`$ Algorithm 1$`(n_1, \alpha_1)`$.

#### 3. Class Bounds
* $`b^\pm_{0,i} \leftarrow \text{Beta}^{-1}(\gamma_0 \text{ or } 1-\gamma_0;\ i,\ n_0+1-i)`$ for $`i = 1, \ldots, n_0`$.
* $`b^\pm_{1,j} \leftarrow \text{Beta}^{-1}(\gamma_1 \text{ or } 1-\gamma_1;\ j,\ n_1+1-j)`$ for $`j = 1, \ldots, n_1`$; $`b^-_{1,0} \leftarrow 0`$, $`b^+_{1,n_1+1} \leftarrow 1`$.

#### 4. Composition
* **For each** $`t \in \mathcal{T} = \{k/n_0\}_{k=0}^{n_0}`$:
    * $`i_U \leftarrow \min\{i : b^-_{0,i} \ge t\}`$; $`U(t) \leftarrow b^+_{1,\, p_{i_U}+1}`$, or $`1`$ if no such $`i`$.
    * $`i_L \leftarrow \#\{i : b^+_{0,i} < t\}`$; $`L(t) \leftarrow b^-_{1,\, p_{i_L}}`$.

#### 5. Boundary Handling
* Enforce fixed anchor points: $`L(0) = 0`$, $`L(1) = U(1) = 1`$. Leave $`U(0)`$ at its composed value unless $`R(0) = 0`$ is asserted.
* For continuum or coarser-grid output, extend $`L`$ from the left grid point and $`U`$ from the right grid point.

**Return:** Band $`[L(t), U(t)]`$ over $`\mathcal{T}`$

---

## References

- Aldor-Noiman, S., Brown, L. D., Buja, A., Rolke, W. and Stine, R. A. (2013). The power to see: a new graphical test of normality. *The American Statistician* 67, 249–260.
- Berk, R. H. and Jones, D. H. (1979). Goodness-of-fit test statistics that dominate the Kolmogorov statistics. *Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete* 47, 47–59.
- Campbell, G. (1994). Advances in statistical methodology for the evaluation of diagnostic and laboratory tests. *Statistics in Medicine* 13, 499–508.
- Esary, J. D., Proschan, F. and Walkup, D. W. (1967). Association of random variables, with applications. *Annals of Mathematical Statistics* 38, 1466–1474.
- Massart, P. (1990). The tight constant in the Dvoretzky–Kiefer–Wolfowitz inequality. *Annals of Probability* 18, 1269–1283.
- Moscovich, A. and Nadler, B. (2017). Fast calculation of boundary crossing probabilities for Poisson processes. *Statistics & Probability Letters* 123, 177–182.
- Noé, M. (1972). The calculation of distributions of two-sided Kolmogorov–Smirnov type statistics. *Annals of Mathematical Statistics* 43, 58–64.
- Owen, A. B. (1995). Nonparametric likelihood confidence bands for a distribution function. *Journal of the American Statistical Association* 90, 516–521.
