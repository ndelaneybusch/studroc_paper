# Frey Optimization: Characterizing Our Bands and Optimizing the Hybrid Floor

**Status:** derivations worked through and checked numerically at small class sizes (§3). Two probes have run (§3 and §6). The derivation and validation work of §§8–9 has not been started.

## Summary

Frey (2008) showed that, for one sample, the set of order-statistic bands with exact coverage at least a given level is convex, so the band minimizing any linear width objective is a global optimum that can be computed. We use this in two ways.

**To characterize our bands (§§4 and 10).** Every exactly calibrated band is optimal for some linear objective. ELL's is far from ROC area:

- In the interior ELL is close to optimal for width measured in units of each order statistic's standard deviation.
- At the ends it puts about 2,000 times more weight (per standard deviation, at 100 per class) on the innermost bound of each extreme order statistic than on a central bound. Those bounds carry almost no ROC area.
- Against the band optimized for a reference ROC, ELL's expected area is 2.5–4% larger at the diagonal, 6–7% larger at AUC 0.8, and 18–23% larger at AUC 0.95. These are lower bounds on ELL's gap within the composition class.

**To improve one thing: the exact floor of the hybrid, in the part of `auto` mode that routes to the hybrid (§§5–9).** This is where the gain is largest and the cost smallest:

- The floor matters only on the corner regions the hybrid selects, so nearly all interior bounds can be released.
- The datasets that reach the hybrid are those the admission rule rejects, mostly small samples and high AUC, which is where an optimized floor gains most.
- In a first probe at 30, 100, and 300 total observations, an optimized floor made the hybrid 6–12% narrower overall and 13–23% narrower at AUC above 0.95, with no change in coverage in 2,160 paired datasets (§6).
- The theorem behind the floor is unaffected, provided there is one pre-fixed floor per $`(n_0, n_1, \alpha)`$ (§5.2).

Improving the standalone ELL band and tailoring a band to the empirical AUC at run time are set aside (§11).

Sources: Frey's method as recorded in [`literature_references.md`](literature_references.md) §B.12; the bands in [`ell_band.md`](ell_band.md), [`fiducial_band.md`](fiducial_band.md), and [`hybrid_band.md`](hybrid_band.md). Probes: [`experiments/frey_roc_probe.py`](experiments/frey_roc_probe.py) and [`experiments/frey_floor_probe.py`](experiments/frey_floor_probe.py), with outputs in `experiments/res_frey_*.json`.

---

## 1. One Class: Convexity and Its Consequences

### 1.1 Setup

For a class of size $`n`$, a band is a pair of nondecreasing sequences $`0 \le l_i \le h_i \le 1`$, and its coverage is

```math
H(l, h) = P\big(l_i \le U_{(i)} \le h_i,\ i = 1, \ldots, n\big) = n!\ \mathrm{Vol}\,K(l,h), \qquad K(l,h) = \{x : l_i \le x_i \le h_i,\ 0 \le x_1 \le \cdots \le x_n \le 1\}.
```

These are the class bounds $`b^\mp_{c,i}`$ of [`ell_band.md` §2.3](ell_band.md#23-one-sample-ell-bounds). Frey's bounds $`a_i, b_i`$ on the CDF map to them as $`l_i = a_i`$, $`h_i = b_{i-1}`$.

### 1.2 Convexity

**Claim.** $`H^{1/n}`$ is concave in $`(l, h)`$. Hence $`\log H`$ is concave and $`\{(l,h) : H \ge 1 - \alpha_c\}`$ is convex.

*Proof.* The set $`\{(x, l, h) : l_i \le x_i \le h_i,\ 0 \le x_1 \le \cdots \le x_n \le 1\}`$ is defined by linear inequalities in $`(x, l, h)`$ jointly, so it is convex. $`K(l,h)`$ is its section at $`(l,h)`$. By Brunn's theorem (the Brunn–Minkowski inequality applied to parallel sections of a convex body), the $`n`$-th root of the section volume is concave on the set of $`(l,h)`$ with nonempty section. $`\square`$

This is Frey's argument (his eq. 12 and Theorem 7), stated here as a property of the constraint so that it can be reused for any objective.

### 1.3 Gradient of the coverage

Conditional on $`U_{(i)} = c`$, the smaller order statistics are those of $`i-1`$ uniforms on $`(0,c)`$ and the larger ones those of $`n-i`$ uniforms on $`(c,1)`$, independently. Therefore

```math
\frac{\partial H}{\partial l_i} = -f_i(l_i)\, P^{\text{left}}_i(l_i)\, P^{\text{right}}_i(l_i), \qquad \frac{\partial H}{\partial h_i} = +f_i(h_i)\, P^{\text{left}}_i(h_i)\, P^{\text{right}}_i(h_i),
```

where $`f_i`$ is the $`\text{Beta}(i, n+1-i)`$ density and

- $`P^{\text{left}}_i(c)`$ is the non-crossing probability of $`i-1`$ uniform order statistics with bounds $`l_k/c`$ and $`\min(h_k, c)/c`$, $`k < i`$;
- $`P^{\text{right}}_i(c)`$ is that of $`n-i`$ uniform order statistics with bounds $`(\max(l_k, c) - c)/(1-c)`$ and $`(h_k - c)/(1-c)`$, $`k > i`$.

Each factor is one call to the existing crossing-probability kernel on a shorter band. The magnitude of the derivative is the rate at which probability leaves the acceptance region through that bound.

### 1.4 Optimal bands and Frey's condition

Take any weights $`w^-_i, w^+_i \ge 0`$ and minimize the linear objective

```math
\sum_i w^+_i h_i - \sum_i w^-_i l_i \qquad \text{subject to } H(l,h) \ge 1 - \alpha_c .
```

By §1.2 this is a convex program. Its first-order conditions are, for a constant $`C > 0`$ and every bound not at 0 or 1,

```math
\Big|\frac{\partial H}{\partial l_i}\Big| = C\, w^-_i, \qquad \frac{\partial H}{\partial h_i} = C\, w^+_i ,
```

and they are sufficient. These equalities are the conditions at bounds where no other constraint binds. Where a monotonicity constraint ($`l_i \le l_{i+1}`$, $`h_i \le h_{i+1}`$), an ordering constraint ($`l_i \le h_i`$), or a box constraint is active, the full conditions carry a nonnegative multiplier for it, and the gradient ratios of the tied bounds need not be equal. Frey shows the monotonicity constraints are strict at his optimum, but that relies on strictly positive weights; with zero weights or released bounds, as in §§2 and 8, active constraints are the normal case and a solver or check must include them. Frey's problem is the case $`w^-_i = w_i`$, $`w^+_i = w_{i-1}`$ (one weight per interval of the CDF). Nothing in the convexity argument uses that tie between the weights, so the lower and upper bounds may be weighted separately. That freedom is what the ROC objective needs (§2).

### 1.5 Every exact band is optimal for its own gradient

The converse reading of §1.4: a band with $`H = 1 - \alpha_c`$ and nonzero gradient minimizes the linear objective whose weights are its own coverage gradient. This is the supporting hyperplane of the convex feasible set at that band. So "which band is best" is the same question as "which linear objective do we hold", and any band's implied objective can be computed from §1.3.

**ELL in the interior.** Where the Beta law of $`U_{(i)}`$ is close to normal with standard deviation $`\sigma_i`$ and the bound sits $`z_i`$ standard deviations out, $`f_i \approx \varphi(z_i)/\sigma_i`$, so the condition reads

```math
\varphi(z_i) \approx C\, \frac{w_i\, \sigma_i}{\rho_i}, \qquad \rho_i = P^{\text{left}}_i P^{\text{right}}_i .
```

ELL has $`z_i`$ constant, so its implied weights are $`w_i \propto \rho_i/\sigma_i`$: width counted in standard-deviation units, modulated by the conditional factor $`\rho_i`$. Uniform weights instead give $`\varphi(z_i) \propto \sigma_i/\rho_i`$, which sends $`z_i`$ to infinity in the tails; this is why Frey's uniform-weight band has $`l_1 = 0`$ and $`h_n = 1`$.

The same formula shows why reweighting moves a band slowly: $`z_i`$ depends on the logarithm of the weight. Large gains need weights that differ by orders of magnitude, which is the case for ROC area (§2.3).

---

## 2. Two Classes: The ROC Objective

### 2.1 Area of the composition band

Write $`l^c, h^c`$ for the class-$`c`$ bounds, with sentinels $`l^0_0 = 0`$, $`h^0_{n_0+1} = 1`$, $`l^1_0 = 0`$, $`h^1_{n_1+1} = 1`$, and $`p_i`$ for the number of positives ranked above the $`i`$-th negative ($`p_0 = 0`$). From [`ell_band.md` §2.6](ell_band.md#26-composition), the upper edge equals $`h^1_{p_i+1}`$ on $`(l^0_{i-1}, l^0_i]`$ and 1 beyond $`l^0_{n_0}`$; the lower edge equals $`l^1_{p_i}`$ on $`(h^0_i, h^0_{i+1}]`$. So the continuum area is

```math
A = \underbrace{\sum_{i=1}^{n_0} (l^0_i - l^0_{i-1})\, h^1_{p_i+1} + (1 - l^0_{n_0})}_{\int U}\ -\ \underbrace{\sum_{i=1}^{n_0} (h^0_{i+1} - h^0_i)\, l^1_{p_i}}_{\int L}.
```

Two structural facts follow.

- **The edges separate.** The upper edge uses only the pair $`(l^0, h^1)`$ and the lower edge only $`(h^0, l^1)`$. This is Lemma 1 of `ell_band.md` seen from the width side.
- **The area is bilinear.** It is linear in the negative-class bounds for fixed positive-class bounds and ranks, and vice versa.

### 2.2 Expected area under a reference ROC

Let $`M_{ij} = P_{R^*}(p_i = j)`$ under a reference ROC $`R^*`$. Since $`A(X_{[i]}) \sim \text{Beta}(i, n_0+1-i)`$ and, given that value $`u`$, $`p_i \sim \text{Binomial}(n_1, R^*(u))`$,

```math
M_{ij} = \int_0^1 f_i(u)\, \binom{n_1}{j} R^*(u)^j \{1 - R^*(u)\}^{n_1 - j}\, du .
```

Only these marginals are needed, because the area is a sum of terms each involving one $`p_i`$. With $`q^U_i = \sum_j M_{ij} h^1_{j+1}`$ and $`q^L_i = \sum_j M_{ij} l^1_j`$ (and $`q^U_{n_0+1} = 1`$, $`q^L_0 = 0`$),

```math
E_{R^*}[A] = \sum_{i=1}^{n_0} (l^0_i - l^0_{i-1})\, q^U_i + 1 - l^0_{n_0} - \sum_{i=1}^{n_0} (h^0_{i+1} - h^0_i)\, q^L_i .
```

A prior over reference curves enters only through the average of their matrices $`M`$.

### 2.3 The weights

Collecting terms gives the per-class linear objectives of §1.4:

| Bound | Weight | Meaning |
|---|---|---|
| Negative lower $`l^0_i`$ | $`q^U_{i+1} - q^U_i`$ | Expected rise of the certified upper TPR between consecutive negatives |
| Negative upper $`h^0_i`$ | $`q^L_i - q^L_{i-1}`$ | Expected rise of the certified lower TPR |
| Positive upper $`h^1_j`$ | $`\sum_i (l^0_i - l^0_{i-1}) M_{i,j-1}`$ | Expected FPR length on which the upper edge reads this bound |
| Positive lower $`l^1_j`$ | $`\sum_i (h^0_{i+1} - h^0_i) M_{ij}`$ | Expected FPR length on which the lower edge reads this bound |

All are nonnegative. For a reference with high AUC many of them are essentially zero: the upper edge is pinned at 1 over most of the FPR range whatever the bounds are, so the pair $`(l^0, h^1)`$ has little area to buy, while the lower edge spends most of the range reading the lower bounds of the largest positive order statistics.

### 2.4 The optimization problem

```math
\min_{l^0, h^0, l^1, h^1}\ E_{R^*}[A] \qquad \text{subject to}\qquad \log H_0(l^0, h^0) + \log H_1(l^1, h^1) \ge \log(1-\alpha), \quad \text{monotone bounds in } [0,1].
```

- **The constraint is jointly convex**, since each $`\log H_c`$ is concave (§1.2). The class split of [`ell_band.md` §2.5](ell_band.md#25-class-split) is therefore a variable of the joint problem. Plain alternation does not optimize it: with one class held fixed and the product constraint active, the other class cannot lower its own coverage, so no budget moves between classes. Both probes fixed the class targets. Optimizing the split needs an outer search over it with the class problems inside, or a joint update.
- **The objective is bilinear, not convex.** With one class fixed, the other class's problem is exactly §1.4 and has a global solution. Alternating between the classes decreases the objective monotonically and stops at a point that neither class can improve alone. That point need not be the joint global minimum.
- **Other objectives.** Area on $`[0, \tau]`$ or a weighted area $`\int \omega(t)(U - L)\,dt`$ stays linear in the positive-class bounds but is not linear in the negative-class bounds (they enter through $`\Omega(l^0_i)`$ with $`\Omega' = \omega`$). The feasible set is still convex, but the negative-class step loses its global guarantee, and the objective can be nonsmooth (a weight function with jumps, or a sum over grid points, has kinks where a bound crosses a breakpoint).

### 2.5 Validity

Theorem 1 of `ell_band.md` requires only bound sequences fixed before the ranks are seen, with $`P(E_0)P(E_1) \ge 1-\alpha`$. Optimized bounds satisfy this if:

1. they are a function of $`(n_0, n_1, \alpha)`$ and a reference declared in advance, and
2. the final joint coverage is evaluated by the exact recursion and found to be at least $`1-\alpha`$.

The optimizer is then outside the proof, like the bisection in the ELL calibration: a poor solution costs width, not coverage. **Choosing the reference from the evaluation ranks, for example from the empirical AUC, breaks condition 1 and voids the theorem.**

Proposition 2 of `ell_band.md` (each one-sided miss at most $`1 - \sqrt{1-\alpha}`$) uses the reflection symmetry $`h_i = 1 - l_{n+1-i}`$ of the ELL bounds and does **not** carry over. An optimized band has total miss probability at most $`\alpha`$ with no separate cap per direction, unless one is imposed. Imposing one keeps the problem convex (§8.4).

---

## 3. Numerical Check

All at $`\alpha = .05`$ with an equal class split; reference curves are equal-variance binormal. Optimization is SLSQP with the analytic gradient of §1.3; every reported band was re-evaluated by the exact kernel and has joint coverage 0.950000.

### 3.1 The derivations

| Check | Result |
|---|---|
| Gradient of §1.3 against central differences, ELL band, $`n = 25`$ | Largest relative error $`1.0 \times 10^{-8}`$ |
| Uniform-weight optimum, $`n = 20`$, 95%: total width $`\sum (h_i - l_i)`$ | 8.994, against 9.322 for exact KS; Frey's figure digitizes to 8.995 |
| Same band against the bounds digitized from Frey's figure | Largest difference $`1.0 \times 10^{-4}`$, the digitization resolution |
| Expected area of §2.2 against the mean over 4,000 sampled rank vectors, all bands and references | Largest difference 0.0015 |

### 3.2 Expected area

Expected continuum area. "Optimized for" bands come from alternating the class problems of §2.4 from an ELL start. "Mix" is the equal-weight average of the three references.

| Band | 30/30: AUC .5 | .8 | .95 | 100/100: .5 | .8 | .95 | 150/30: .5 | .8 | .95 |
|---|---|---|---|---|---|---|---|---|---|
| KS, exact | .763 | .644 | .507 | .496 | .430 | .321 | .612 | .525 | .401 |
| ELL | .743 | .634 | .470 | .489 | .392 | .239 | .603 | .500 | .344 |
| Frey, uniform weights | .732 | .621 | .464 | .473 | .386 | .249 | .587 | .490 | .346 |
| ELL without 3 end bounds per side | .736 | .626 | .462 | .485 | .388 | .237 | .596 | .494 | .338 |
| Optimized for AUC .5 | .725 | .641 | .529 | .471 | .395 | .280 | .584 | .502 | .382 |
| Optimized for AUC .8 | .775 | .591 | .414 | .502 | .369 | .219 | .623 | .466 | .302 |
| Optimized for AUC .95 | .859 | .620 | .385 | .587 | .393 | .202 | .716 | .494 | .280 |
| Optimized for the mix | .751 | .597 | .406 | .484 | .372 | .216 | .603 | .471 | .298 |
| Asymmetric ELL fitted to the mix | .754 | .617 | .445 | .493 | .387 | .231 | .606 | .485 | .322 |

Ratios to ELL for the two candidates that matter:

| Band | 30/30: AUC .5 | .8 | .95 | 100/100: .5 | .8 | .95 | 150/30: .5 | .8 | .95 |
|---|---|---|---|---|---|---|---|---|---|
| Optimized for the mix | 1.010 | .942 | .863 | .991 | .950 | .904 | 1.000 | .941 | .866 |
| Asymmetric ELL fitted to the mix | 1.014 | .974 | .947 | 1.008 | .987 | .967 | 1.005 | .970 | .935 |

Low-FPR area on $`[0, 0.1]`$ for the mix-optimized band, as a ratio to ELL at 100/100: 1.076, 1.003, and 0.947 at AUC .5, .8, and .95. The total-area objective does not protect the low-FPR region at low AUC.

Caveats on this table:

- ELL and KS use the equal split here, including at 150/30, where production ELL uses the sample-size split. The asymmetric ELL fitted its own split (0.25 at 150/30, against 0.31 from the sample-size rule), so part of its 150/30 gain is the split.
- References are binormal with equal variances only.
- Alternation started from ELL only; other starts were not tried.

### 3.3 Where the optimized band puts its error budget

Marginal tail probability cut off by each bound (the "local level"), 100 per class. ELL uses $`5.1 \times 10^{-4}`$ everywhere.

| Bound set | Edge it feeds | Rank 1 | Rank 5 | Rank 25 | Rank 50 | Rank 75 | Rank 96 | Rank 100 |
|---|---|---|---|---|---|---|---|---|
| Mix-optimized, negative upper $`h^0`$ | Lower | $`1.9 \times 10^{-3}`$ | $`1.5 \times 10^{-3}`$ | $`1.5 \times 10^{-3}`$ | $`1.1 \times 10^{-3}`$ | $`3.8 \times 10^{-4}`$ | $`6 \times 10^{-6}`$ | 0 |
| Mix-optimized, positive lower $`l^1`$ | Lower | 0 | $`6 \times 10^{-6}`$ | $`3.5 \times 10^{-4}`$ | $`1.1 \times 10^{-3}`$ | $`1.5 \times 10^{-3}`$ | $`1.5 \times 10^{-3}`$ | $`1.9 \times 10^{-3}`$ |
| Mix-optimized, negative lower $`l^0`$ | Upper | 0 | $`4.6 \times 10^{-5}`$ | $`4.7 \times 10^{-4}`$ | $`6.1 \times 10^{-4}`$ | $`3.7 \times 10^{-4}`$ | $`3.7 \times 10^{-5}`$ | $`2 \times 10^{-6}`$ |
| Mix-optimized, positive upper $`h^1`$ | Upper | $`2 \times 10^{-6}`$ | $`3.7 \times 10^{-5}`$ | $`3.5 \times 10^{-4}`$ | $`6.0 \times 10^{-4}`$ | $`4.8 \times 10^{-4}`$ | $`4.6 \times 10^{-5}`$ | 0 |
| Frey uniform, lower bounds | Both | 0 | $`9 \times 10^{-6}`$ | $`5.0 \times 10^{-4}`$ | $`1.2 \times 10^{-3}`$ | $`1.1 \times 10^{-3}`$ | $`3.4 \times 10^{-4}`$ | $`1.3 \times 10^{-4}`$ |

Reading:

- **The budget moves to the lower edge.** The two bound sets that form the ROC band's lower edge get local levels about three times ELL's over the ranks the lower edge actually reads, and the two that form the upper edge get about ELL's level in the middle and far less at the ends.
- **Each bound set is tight at one end only**: the negative upper bounds at the top-ranked negatives, the positive lower bounds at the bottom-ranked positives. ELL treats both ends of both classes alike.
- **The band optimized for AUC .95 goes much further** (levels up to $`6 \times 10^{-3}`$ on the lower-edge sets and below $`5 \times 10^{-5}`$ on the upper-edge sets), which is why it is 20% wider than ELL at the diagonal.

The three-parameter asymmetric ELL captures the first effect only: its fitted skews are $`-0.53`$ for negatives and $`+0.58`$ for positives at 100/100, a ratio of about 3 between the lower-edge and upper-edge local levels. It recovers roughly a third of the full optimization's gain.

### 3.4 ELL's implied objective

At $`n = 100`$, class level .05, the coverage gradient of the ELL lower bounds times the order statistic's standard deviation is:

| Rank | 1 | 2 | 5 | 10 | 25 | 50 | 75 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| $`\lvert\partial H/\partial l_i\rvert\, \sigma_i`$ | 0.87 | 0.047 | 0.0064 | 0.0023 | 0.00076 | 0.00042 | 0.00039 | 0.00050 | 0.00090 |

- From rank 25 to rank 100 the values are within a factor of about 2.3, consistent with §1.5: in the interior ELL is close to the optimum for standardized width.
- At rank 1 the value is about 2,000 times the central one. ELL is optimal only for an objective that values raising $`l_1`$ above zero (and, by symmetry, lowering $`h_n`$ below one) enormously. In the ROC band those two bounds control an FPR interval of length about $`\gamma/n`$.
- Dropping those bounds outright recovers little, about 1% of area with three removed per side (§3.2), because ELL's level responds weakly to the budget released. The larger gains need the budget redistributed, not just freed.

---

## 4. What This Says About ELL

1. **ELL is a good default for a shape-agnostic objective and a poor one for ROC area at high AUC.** Its gap to the reference-optimal composition is at least 2.5–4% at the diagonal and 18–23% at AUC .95 at the sizes tried. The claim in the paper that ELL's advantage is largest at high AUC is true relative to KS and should not be read as near-optimality there.
2. **Frey's uniform-weight band is not the right competitor either.** It beats ELL by 1–3% at AUC .5–.8 and loses by 4% at AUC .95 at 100 per class, because rank-space width is not ROC area. This is useful for the state-of-the-art comparison: the meaningful alternative to ELL is an ROC-weighted optimum, not Frey's own band.
3. **The exact-KS band is 5–59% above the reference optimum** at 100 per class, so the ordering KS, then ELL, then optimized is consistent across references except that an optimum tuned to a very different AUC can be worse than both.
4. **These are gaps within rectangular compositions.** They sit alongside, and do not replace, the projection slack discussed in [`ell_band.md` §3.2](ell_band.md#32-conservatism): the optimized band still protects two CDFs separately.

---

## 5. Focus: The Exact Floor in the `auto` Slice

### 5.1 Why here

`auto` mode sends a dataset to the unfloored fiducial band when an admission rule on the class counts and the AUC passes, and to the hybrid otherwise ([`experiments/auto_rule_surface.py`](experiments/auto_rule_surface.py); the rule has the form $`n_{\min} \ge \max\{\text{floor},\ C\,\text{ratio}^b/(1-\text{AUC})^a\}`$ and is not yet fitted). In the hybrid the exact band is used only on the floor region: a left prefix and the expanded right saturated run ([`hybrid_band.md` §§2.2–2.4](hybrid_band.md#22-left-prefix)).

Three facts make this the right place to apply §2:

1. **The objective is narrow.** Only the grid points in the floor region count, so the weights of §2.3 vanish for most bounds and the optimizer can release them.
2. **The slice is where optimization pays.** §3.2 showed gains growing with AUC, and the routed datasets are the high-AUC, small-minority ones.
3. **The reference is not a free choice.** For a given $`(n_0, n_1)`$ the routed AUC range follows from the admission rule, so the routed distribution is determined by the rule and the shape library (§8.2). No table indexed by AUC is needed.

### 5.2 Validity under routing

The hybrid's bound ([`hybrid_band.md` Prop. 1](hybrid_band.md#31-coverage-guarantees)) needs an exact band $`B_M`$ with a full-curve guarantee at a prespecified level, and allows any data-selected region. It therefore holds with an optimized floor in place of ELL if the floor's bounds are fixed sequences with exact joint coverage at least $`1 - \alpha_M`$.

Routing on the empirical AUC does not disturb this, **as long as the data only decide whether the one exact band is used, not which exact band is used**:

```math
P(\text{routed to hybrid},\ B_M \text{ misses}) \le P(B_M \text{ misses}) \le \alpha_M .
```

Consequences for the design:

- **One floor per $`(n_0, n_1, \alpha)`$.** Different floors for different empirical AUC ranges would be a selection among exact bands; the union bound then gives $`K\alpha_M`$ for $`K`$ floors, and correcting for it costs more than tailoring gains (§11).
- **Any second exact band in `auto`** (for example plain ELL for some sizes) must be selected by the class sizes alone.
- **The guarantee is unconditional.** Given that a dataset was routed, the floor's miss rate can exceed $`\alpha_M`$, as [`hybrid_band.md` §3.4](hybrid_band.md#34-full-grid-floors-and-remaining-theory) already notes for the full-grid floor.
- **Proposition 2 of `ell_band.md` is lost** for the floor: no separate cap per direction (§2.5).

### 5.3 The risk particular to the hybrid

The hybrid has no proved split of its error. Its measured coverage is the fiducial parent's exterior misses plus the floor's misses, and ELL contributes almost none because it covers about 0.999. An optimized floor is still conservative as a composition, but it spends its level where the corners need it, so its realized miss rate could rise and take the hybrid's total with it. The first probe found no such effect (§6); the validation of §9 is sized to detect it.

---

## 6. First Probe on the Slice

[`experiments/frey_floor_probe.py`](experiments/frey_floor_probe.py), $`\alpha = .05`$. For each design: 30 reference curves from the binormal, heteroscedastic Gaussian, and Student-$`t`$ families; the weight function $`\omega(t)`$ is the share of training datasets whose floor region contains $`t`$; the floor minimizes expected $`\omega`$-weighted area by the alternation of §2.4 at the sample-size split. Test: 360 fresh datasets per design from six families (three of them not used as references), each evaluated with the default hybrid and with the optimized floor on the same fiducial cloud.

Area of the hybrid with the optimized floor divided by the default hybrid's:

| $`n`$ (prevalence) | Classes (neg/pos) | Grid floored | All AUC | AUC .55–.8 | AUC .8–.95 | AUC .95–.99 | Held-out families |
|---|---|---|---|---|---|---|---|
| 30 (50%) | 15/15 | 94% | 0.880 | 0.980 | 0.888 | 0.775 | 0.885 |
| 30 (10%) | 27/3 | 91% | 0.921 | 0.965 | 0.925 | 0.869 | 0.920 |
| 100 (50%) | 50/50 | 62% | 0.928 | 0.995 | 0.949 | 0.852 | 0.929 |
| 100 (10%) | 90/10 | 66% | 0.891 | 0.972 | 0.903 | 0.788 | 0.886 |
| 300 (50%) | 150/150 | 34% | 0.945 | 0.996 | 0.963 | 0.867 | 0.943 |
| 300 (10%) | 270/30 | 50% | 0.917 | 0.984 | 0.938 | 0.844 | 0.913 |

- **Width inside the floor region** fell by 8–13% overall in every design. The area gain shrinks with $`n`$ because the floored share shrinks.
- **Coverage, paired.** Of the 2,160 datasets, 2,135 were covered by both hybrids and 25 missed by both; none was covered by one and missed by the other (per-dataset outcomes are saved in the result files). With no harmful discordance, the one-sided 95% upper bound on the coverage lost is 0.83 points within a design of 360 and 0.14 points pooled, so this probe does not by itself meet the 0.5-point per-design criterion of §9.2. Coverage by design was 0.972–0.997. The optimized exact band alone covered 0.997–1.000, as did ELL.
- **A floor optimized on references with AUC at least 0.8** gained a further 2–4 points at AUC above 0.95 in the two larger imbalanced designs and cost up to 2% of floor-region width below 0.8. Overall area was about the same.
- **Never wider on average**: the full-range floor was at or below the default hybrid's area in every AUC band of every design.

Limits: 360 datasets per design (a standard error of 0.5–0.9 points on coverage); one level; the routed slice approximated by AUC bands because the rule is not fitted; miss direction not recorded. The floor was optimized for a surrogate (the window objective of §8.1), and the table reports geometric means of trapezoidal-area ratios, not the ratio of mean step-extension areas that §9.3 makes primary.

---

## 7. Where to Optimize

### 7.1 Where the unfloored band fails

From the shape-by-AUC study at 50% prevalence, 95% level (`figures/shape_auc_study/`; values read from the figures, 6,000 draws per point; 10% prevalence is reported to be similar):

| Family and shape | AUC band | Sample sizes below nominal | Lowest coverage |
|---|---|---|---|
| Student-$`t`$, df 1–2 | 0.90–0.95 | 100–300 | about 0.84 |
| Student-$`t`$, df 1–2 | 0.95–1 | 100–1,000 | about 0.67 at 300 |
| Student-$`t`$, df 2–4 | 0.95–1 | 300–3,000 | about 0.83 at 1,000 |
| Student-$`t`$, df 4–10 | 0.95–1 | 3,000–10,000 | about 0.94 |
| Heteroscedastic, $`\sigma_+/\sigma_-`$ 2–5 | 0.90–0.95 | 100 | about 0.93 |
| Heteroscedastic, $`\sigma_+/\sigma_-`$ 2–5 | 0.95–1 | 300–1,000 | about 0.89 at 300 |
| Heteroscedastic, $`\sigma_+/\sigma_-`$ 0.2–0.5 | 0.50–0.90 | 30–100 | about 0.89 at 30 |
| Heteroscedastic, $`\sigma_+/\sigma_-`$ 0.2–0.5 | 0.90–1 | 100–1,000 | about 0.945 |

The hybrid is at or above about 0.97 in every panel. Two features drive the design:

- **The failure is a ridge that moves to larger $`n`$ as AUC rises.** At AUC above 0.95 it runs from 100 to several thousand; at 0.90–0.95 from 100 to 300.
- **Small samples fail at low AUC too** when positives are much less dispersed than negatives, and the rule cannot see shape. So at small $`n_{\min}`$ every AUC is routed, through the rule's floor term.

### 7.2 Zones

The boundaries below are provisional and will be redrawn from the fitted rule. "Floored share" is the mean fraction of the grid in the floor region among routed datasets.

| Zone | Class sizes | AUC routed to the hybrid | Floored share | Measured or expected gain in hybrid area | Decision |
|---|---|---|---|---|---|
| Z0 | $`n_{\min} \lesssim 15`$ | All | About 0.9–1 | 8–12% (measured at 15/15 and 27/3) | Optimize. The floor is nearly the whole band, so this is an optimized exact band |
| Z1 | $`n_{\min}`$ from about 15 to 150, total up to about 300 | Most; all at the smallest sizes, high AUC at the largest | 0.3–0.7 overall, 0.6–0.9 at AUC above 0.95 | 6–11% overall, 13–21% at AUC above 0.95 (measured) | Optimize |
| Z2 | Total about 300 to a few thousand | High only: above about 0.9 at 300–1,000, above about 0.95 beyond | Not measured; expected 0.1–0.4 | Floored share times the 10–20% region gain, so perhaps 2–6% | Measure first (§9.4). Optimize only if the gain is at least 3% and a low-dimensional profile delivers it |
| Z3 | Admitted by the rule, or total above the ridge | None, or a negligible floor | — | — | Keep ELL. Nothing to optimize |

### 7.3 How the optimization differs across $`n_0`$, $`n_1`$, AUC, and $`\alpha`$

- **By $`(n_0, n_1)`$: a separate floor for each pair.** Both the weights (through the rank-count law and the floor region) and the feasible set depend on the pair.
- **By AUC: through the reference weights only, never through separate floors.** For each pair the training distribution is the shape library as filtered by the dispatcher (§8.2). At small sizes that is the whole AUC range; at larger sizes it concentrates on high AUC. This is where "optimize differently by AUC" happens, and it is fixed before any data are seen. The probe supports it: the full-range floor never lost, and the high-AUC floor gained most exactly where routing would send the data.
- **By imbalance: the split is part of the solution.** The sample-size split is the starting point; whether freeing it helps is a derivation question (§8.4).
- **By $`\alpha`$: a separate floor for each level**, since the floor region depends on the trim depth and the feasible set on the level. The sweep is $`\alpha \in \{.05, .1, .2, .5\}`$, and nesting across levels has to be checked or imposed (§8.5).

---

## 8. Derivation Design

What has to be worked out before a floor can ship, in order.

### 8.1 Objective

**Primary objective: the area of the hybrid itself.** The floor is a means to a narrower hybrid, so candidates are judged by

```math
J(\theta) = E\big[\operatorname{Area}\{B_H(\theta)\} \mid \text{routed to the hybrid}\big],
```

where $`\theta`$ is the floor's bounds and $`B_H(\theta)`$ is the band the production construction returns: the hull of the recalibrated fiducial parent with the floor on the region, followed by the tightening closure ([`hybrid_band.md` §§2.5–2.7](hybrid_band.md#25-complement-trimming)). Area is that of the band as claimed, the conservative step extension

```math
\operatorname{Area}(B) = \sum_k (t_{k+1} - t_k)\{U(t_{k+1}) - L(t_k)\},
```

not the grid-width sum or the trapezoidal area. For a fixed dataset and cloud the parent and the region do not depend on $`\theta`$, so minimizing $`J`$ is the same as minimizing the area the floor adds to the parent.

Narrowing the exact band is not the same as narrowing the hybrid:

- Where the parent's edge already lies outside the floor's, the hull is the parent and the floor's width there is irrelevant.
- The closure carries edge values across grid points, so which bounds matter depends on the whole edge, not on the region alone.

**Surrogates, for generating candidates only.** $`J`$ is evaluated through the full construction and is not a linear function of the bounds, so the convex machinery of §§1–2 does not apply to it directly. Candidates come from objectives that are linear in each class's bounds:

- **Window surrogate (used in §6).** The expected $`\omega`$-weighted area of the exact band, with $`\omega(t) = P(t \in A)`$. It ignores the parent, the closure, and the dependence between the region and the ranks.
- **Joint-weight surrogate.** Weights estimated from training draws as $`E[\mathbf{1}\{k \in A\}\,\mathbf{1}\{p_i = j\}]`$, so that the right region is long exactly when many negatives rank below every positive.
- **Hinge surrogate.** The expected area by which the floor exceeds the parent on the region, $`E[\sum_{k \in A} \{(U_M - U_F)_+ + (L_F - L_M)_+\}(t_k)]`$, before closure. It is convex in the positive-class bounds for fixed negative-class bounds, and it stops rewarding the floor for tightening where the parent already decides the hull.

**Selection.** Every candidate (each surrogate, several starts, ELL itself) is scored on $`J`$ with one shared set of training datasets and clouds, so the comparison is paired. The winner is then scored once on separate validation draws that played no part in generating or choosing it. A surrogate's own improvement is never reported as the hybrid's improvement.

To do: implement $`J`$ on stored training tuples (labels, cloud, region, parent); derive the joint and hinge weights; rank the three surrogates by the $`J`$ of what they produce on the six probe designs.

### 8.2 The routed distribution

Conditioning on routing changes two things, and both must enter the objective.

- **Which curves are routed.** A reference ROC with a higher AUC is routed more often.
- **Which samples are routed, given the curve.** The router acts on the empirical AUC, so under one fixed ROC it selects the rank sequences that happen to look favourable. The rank law that matters is

```math
P(p_i = j \mid D) = \frac{\sum_R \pi(R)\, P_R(p_i = j,\ D)}{\sum_R \pi(R)\, P_R(D)},
```

with $`D`$ the event of being routed to the hybrid. Averaging the unconditional laws $`M^R`$ with weights $`\pi(R \mid D)`$ gets the first effect and misses the second, and is generally a different law.

**Procedure.** Estimate the objective from training datasets that have passed through the frozen dispatcher:

1. Draw curves from the shape library and datasets from each, at the pair $`(n_0, n_1)`$.
2. Run the dispatcher. Keep the routed datasets, each with its label sequence, fiducial cloud, floor region, and recalibrated parent.
3. Evaluate $`J`$ of §8.1, and the surrogate weights, as averages over the kept tuples.

This handles the routing selection and the dependence between the region and the ranks together, with no closed-form rank law needed.

Consequences:

- **The floor cannot be finalized before the admission rule is frozen.** A change to the rule changes $`D`$ and requires retraining. Until then the AUC-band approximation of §6 is exploratory.
- **The library fixes the shape prior $`\pi`$.** It is the seven suite families over their ranges, with some families held out for §9.3.
- **Training and validation draws are separate** (§8.1), and neither is reused as confirmatory data.

### 8.3 Constraint and solver

- **Coverage constraint.** Exact joint coverage at $`\alpha_M = \alpha`$, as in the current default. A floor at a declared smaller level is the fallback if §9.2 finds the hybrid's slack eroded.
- **Class split.** An outer one-dimensional search over the split $`\rho`$, with the two class problems solved inside at targets $`(1-\alpha)^\rho`$ and $`(1-\alpha)^{1-\rho}`$, started at the sample-size rule. Several starting splits help but do not certify a joint optimum.
- **Solver.** The alternation of §2.4 inside the split search, keeping the best iterate, from several starting bands (ELL, exact KS, Frey's uniform band). The solver must treat monotonicity, ordering, and box constraints as possibly active (§1.4). The one-class cutting-plane bound of §10 is a certificate for each half-step only.
- **Final check.** Every floor is re-evaluated by the exact kernel and widened minimally if short.

### 8.4 Structural requirements

To verify analytically, then by test:

1. Monotone bounds give monotone, ordered ROC edges, so the tightening closure of the hybrid applies unchanged.
2. On the perfectly separated path the floor is the whole grid and the hybrid is the hull with the floor. The floor's upper edge is 1 there by the sentinel, and its lower edge obeys the necessary conditions of [`ell_band.md` §3.3](ell_band.md#33-width-and-tail-adaptivity) because it is exact. So the corner constructions of [`fiducial_band.md` Prop. 2](fiducial_band.md#31-coverage-guarantees) remain covered whenever the floor covers.
3. **Directional constraints are available and convex.** The lower edge reads only $`(h^0, l^1)`$ and the upper edge only $`(l^0, h^1)`$ (§2.1), so

```math
H_0(0, h^0)\, H_1(l^1, 1) \ge 1 - \eta_L, \qquad H_0(l^0, 1)\, H_1(0, h^1) \ge 1 - \eta_U
```

bound the probability that the truth falls below the band by $`\eta_L`$ and above it by $`\eta_U`$. Each factor is a coverage with the other side's bounds released, so each logarithm is concave by §1.2 and each constraint is convex. These are sufficient conditions, bounds on the directional miss probabilities and not their exact values.
4. **A second certificate.** By the union bound the two directional constraints with $`\eta_L + \eta_U = \alpha`$ also give total coverage $`1-\alpha`$, without the two-sided product constraint. Neither certificate implies the other. To do: compare the floors each yields; $`\eta_L = \eta_U = \alpha/2`$ would give the optimized floor a per-direction cap of $`\alpha/2`$, close to the $`1 - \sqrt{1-\alpha}`$ of ELL's Proposition 2.

### 8.5 Levels and nesting

Two different properties, not to be confused:

- **Nesting of the floors.** Floors optimized separately at each level need not nest. Options: optimize the smallest $`\alpha`$ first and constrain each larger level's bounds to lie inside the previous one (linear constraints, so convexity is kept); or accept non-nesting and document it.
- **Nesting of the hybrid.** Nested floors do not make the hybrid nested. Its floor region and its recalibrated parent both change with the level, and [`hybrid_band.md` §3.3](hybrid_band.md#33-recalibration-and-nesting) already states that shared-cloud nesting across levels is not established for the adaptive hybrid. This is measured on the complete hybrid (§9.7), whatever is done about the floors.

To do: measure how often separately optimized floors fail to nest, what the constraint costs, and how often the complete hybrid fails to nest with either floor.

### 8.6 From tables to something shippable

A floor is $`2(n_0 + n_1)`$ numbers per $`(n_0, n_1, \alpha)`$. With totals up to 300 there are about 45,000 pairs per level, and one optimization takes on the order of a minute, so tabulating every pair is not practical. The plan:

1. Optimize on a lattice of $`(n_{\min}, \text{ratio})`$ covering Z0 and Z1.
2. Express each optimum as four local-level profiles (one per bound set) against rank fraction. §3.3 suggests a simple structure: a plateau on the ranks the edge reads and a taper to zero at the far end.
3. Fit a low-dimensional family to those profiles and a smooth surface for its parameters over $`(\log n_0, \log n_1, \log \alpha)`$.
4. At run time, read the shape from the surface and find the overall level by a certified search, as in [`ell_band.md` §4.4](ell_band.md#44-warm-started-calibration-from-a-table-of-levels-proposed). One exact coverage evaluation can accept a proposed level; it cannot find the calibrated one. The procedure is therefore: a predicted level shaded conservatively; acceptance only if its exact coverage meets the target; otherwise a bracketed bisection on the level; and, if the search fails or the pair is outside the surface's domain, a deterministic fallback to the ELL floor, which needs no surface. The shape needs no certification; only the level does. An error in the surface costs width, never coverage.

To do: choose the profile family from the lattice optima; measure the width lost by the profile and by interpolation against the full optimum (target: under one point of the gain).

---

## 9. Validation Design

### 9.1 Exactness

For every shipped $`(n_0, n_1, \alpha)`$: joint coverage from the exact kernel at least $`1-\alpha`$; bounds monotone and in $`[0,1]`$; the composed band monotone and ordered. Independent of any simulation. Monte Carlo over $`10^6`$ uniform samples per class confirms the kernel on a sample of floors.

### 9.2 Coverage of the hybrid with the new floor

The question is whether the hybrid loses coverage when its floor is tightened (§5.3). Two criteria, both required.

- **Paired loss.** Both hybrids on one fiducial cloud per dataset. Count datasets covered by the default hybrid and missed by the new one, and the reverse. Criterion: the one-sided 95% upper confidence limit for the net coverage lost is below 0.5 points in every stratum.
- **Absolute coverage.** Passing the paired test against the current hybrid does not show adequate coverage; both could fail on the same shapes. Criterion: the one-sided 95% lower confidence limit for the new hybrid's coverage is at least nominal minus one point in every stratum, with that shortfall fixed before the run.

Design:

- **Through the actual dispatcher.** Datasets are generated, routed by the frozen rule, and analysed in the branch they reach. Coverage is reported conditional on the hybrid branch and unconditionally for `auto` as a whole (§9.6).
- **Strata, fixed in advance.** Totals 30, 100, 300 at 50%, 30%, and 10% prevalence, each crossed with: the four AUC bands of the shape study; and a difficult-shape stratum (Student-$`t`$ with df below 4, heteroscedastic Gaussian with $`\sigma_+/\sigma_-`$ above 2 or below 0.5). Levels $`\alpha \in \{.05, .1, .2, .5\}`$. Results are not pooled across AUC bands, since the optimized floor acts mostly at high AUC.
- **Size.** Set by the number of routed datasets and the expected discordance, not by the number generated. With no harmful discordance the upper limit is about $`3/N`$, so the 0.5-point criterion needs at least 600 routed datasets per stratum, and more if any discordance is expected (about 1,300 at a true rate of 0.1 points). The absolute criterion needs about 3,000 per stratum to separate nominal from one point below it at the 95% level ([`sota_plan.md` §4.4](sota_plan.md#44-power)). Plan for 3,000 routed datasets per stratum.
- **Decomposition.** Misses inside and outside the floor region, and by direction, for both hybrids.
- **Saved.** Per-dataset outcomes for both hybrids, so that every paired table can be recomputed.

### 9.3 Width

- Area of the complete hybrid as claimed (the step-extension area of §8.1), as the mean paired difference and the ratio of mean areas, by design and by AUC band (the four bands of the shape study). Geometric-mean ratios and floor-region width are secondary.
- **Adoption rule.** Adopt in a zone if the new hybrid is (i) not wider than the default by more than 1% in any AUC band of any design, and (ii) at least 5% narrower on average over the routed slice.
- **Held-out shapes.** Families that supplied no reference curves are reported separately and must meet (i) on their own.
- **Reference sensitivity.** Repeat with the reference library and the held-out set swapped.

### 9.4 Zone Z2

Before any derivation effort at larger sizes: at totals 1,000 and 3,000 (50% and 10% prevalence), restricted to AUC above 0.9, measure the floored share and the floor-region gain of a fully optimized floor at the largest size where that is affordable, and of the asymmetric ELL of §3.2 at all sizes. Decision as in §7.2.

### 9.5 Adversarial curves

The floor exists to protect the corners, so the hybrid with the new floor is run on the constructions the old one was designed against: the two corner constructions of `fiducial_band.md` Prop. 2, corner and interior slivers, the step ROC, and perfect separation. Criterion: coverage not below the default hybrid's by the paired test of §9.2. Interior jumps are not protected by either floor and are reported, not tested.

### 9.6 End to end

Once the admission rule is fitted, `auto` is evaluated as one method: overall coverage, coverage by branch, and width, on the main suite and the adversarial curves. It is the only test of the complete procedure on data that played no part in training the floor.

### 9.7 Shipping form

The profile-and-surface floors of §8.6 are re-run through §§9.1–9.3 on sizes off the lattice, including the search and its fallback. The exactness check of §9.1 applies to every size a user can request, since it is performed at run time. Nesting of the complete hybrid across levels (§8.5) is measured here for both floors.

---

## 10. Characterizing Our Bands

Independent of the floor work, and kept.

1. **Efficiency curve.** ELL's expected area divided by the reference-optimal area, against reference AUC, for a grid of class sizes and prevalences. This replaces the Gaussian-approximation region comparison in [`ell_band.md` §3.2](ell_band.md#32-conservatism) (a 16–21% bound at one location) with a finite-sample statement about the whole band.
2. **Implied-objective figure.** ELL's and KS's coverage gradients per standard deviation against rank, next to the ROC-area weights for two references. It shows in one picture why ELL beats KS in the tails and what it over-protects.
3. **A certified bound for one class.** For a linear objective, linearizing $`\log H`$ at any feasible band gives a half-space containing the feasible set (concavity). Minimizing the objective over a few such cuts and the monotonicity constraints is a linear program whose value is a lower bound on the optimum. Applied at ELL with the other class fixed, it certifies how close ELL is to that class's optimum without trusting a nonlinear solver. This bound does not extend to the joint two-class optimum.
4. **The same three items for the alternatives** in the state-of-the-art matrix of [`sota_plan.md`](sota_plan.md) (Berk–Jones, Dümbgen–Wellner, Eicker, Frey's uniform band), so the paper can say which objective each published band serves.
5. **The floor as a band.** In zone Z0 the optimized floor is nearly a whole-curve band. Its efficiency curve belongs in the same figure as ELL's, and it is a candidate row for the method matrix of `sota_plan.md` §3.1 at those sizes.
6. **Wording.** The measured gaps are lower bounds on ELL's distance from the best composition, since alternation may not find the global optimum. They are not upper bounds.

---

## 11. Set Aside

| Idea | Why |
|---|---|
| Tailoring a band to the empirical AUC of the evaluation data | The bounds would depend on the ranks, which voids the theorem. The selection is also adverse: a high empirical AUC picks the band with the tightest lower edge when the truth is most likely below it |
| A menu of exact bands with a multiplicity correction | For three bands the correction widens each by roughly 9%, more than tailoring adds over a mixed reference |
| Calibrating one shared event for a menu | The intersected band is narrower than any member, so selection adds nothing |
| A user-declared design AUC for the standalone band | Valid only with independent data; a wrong declaration costs 16–20% at the diagonal. Could return as an option; not a default |
| Replacing standalone ELL at large $`n`$ | The three-parameter asymmetric ELL recovers 1–7% at AUC of 0.8 or more and loses up to 1.4% at the diagonal. Not enough to justify a second default |
| Dropping ELL's end bounds | About 1% |
| Frey's uniform weights as a default | Loses to ELL at high AUC |

---

## 12. Implementation Notes and Open Items

**Implementation.**

- **Kernel.** The crossing-probability kernel returned zero for bounds with a long gap between breakpoints (for example 2,000 observations with the first 960 upper bounds at 0.5), because each binomial transition started from its zero-success term, which underflowed. It now starts from the largest in-window term. Values for ELL and constant-margin bounds are unchanged to about $`10^{-15}`$, and closed-form binomial cases at sizes up to 20,000 are in the test suite. The probes of §§3 and 6 ran on the earlier kernel at sizes of 300 or fewer per class, below the sizes where the failure was reproduced.
- **Gradient cost.** §1.3 needs two shorter crossing-probability evaluations per bound. A forward–backward version of the recursion should give every derivative from two sweeps, since each derivative is a density times a forward and a backward non-crossing probability. That needs to be derived and checked against §1.3.
- **Swapping the floor.** The probe replaces the exact band inside the hybrid by patching the function that builds it. A production version needs the floor's bound source to be a parameter of `M3Floor`, in the boundary-family abstraction of [`sota_plan.md` §5.2](sota_plan.md#52-rust-boundary-families-and-calibration).
- **Cost so far.** All six designs of §6, optimization and testing, ran in under ten minutes on six processes. The three designs of §3.2 took about 14 minutes on 11 cores.

**Open.**

1. The admission rule is not fitted, so the routed distribution of §8.2 cannot be sampled yet and the floors of §6 are exploratory.
2. Alternation was started only from ELL, at fixed class targets.
3. Miss direction was not recorded in §6, and the width cost of the directional constraints of §8.4 is unknown.
4. No candidate has yet been scored on the primary objective of §8.1; the probe scored a surrogate's output on a secondary area metric.
5. Nothing is measured at levels other than .05, at 30% prevalence, or in zone Z2.
6. Frey's theorems were taken from an agent's reading of his preprint and thesis. §1.2 is proved here independently and §3.1 reproduces his band, but his theorem numbering and any differences in the published version have not been checked by us.
