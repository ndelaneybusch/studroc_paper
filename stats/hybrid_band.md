# Fiducial Simultaneous Confidence Bands with a Localized ELL Floor

## Abstract

We combine the rank-space fiducial ROC band with the exact equal-local-levels (ELL) composition band. The fiducial band supplies local adaptation and relatively small width; the ELL band supplies protection against the endpoint-completion failures that dominate the fiducial band's observed under-coverage. The method selects a left prefix and an expanded right saturated run, recomputes the fiducial trim depth on the remaining production trim columns, takes the pointwise hull with ELL on the selected region, and tightens the result to a monotone band. The region is fixed before recalibration, and all fiducial operations use one cloud. The resulting band contains its recalibrated fiducial parent, is nested inside the corresponding hybrid without recalibration, and covers a monotone truth exactly when the unclosed hull does. Its whole-curve miss probability is at most the ELL level plus the parent's exterior miss probability. The exterior term has no distribution-free bound, so the hybrid has **no established unrestricted finite-sample coverage guarantee**. Tightening preserves this whole-band inequality but does not preserve ELL containment or an ELL-only miss bound within the selected region. In the main simulation suite, the default hybrid has no cell significantly below nominal at any of the three tested levels or prevalences, with mean area about 0.59–0.68 times KS at the 95% level.

This document specifies the production hybrid, called `fiducial_hybrid` in the simulation suite and enabled by `m3_floor=True` or `M3Floor()` in the fiducial API. Its components are specified in [`fiducial_band.md`](fiducial_band.md) and [`ell_band.md`](ell_band.md); the code calls ELL `m3`. We use those specifications for cloud generation, rank depth, corner allowances, and ELL calibration rather than repeating them. [`fiducial_band_theory.md`](fiducial_band_theory.md), especially §§8 and 10, is cited below as “theory.”

---

## 1. Setup and Assumptions

Use the data, sampling assumptions, score direction, and population ROC estimand of [`fiducial_band.md` §1](fiducial_band.md#1-setup-and-assumptions). In particular, the two samples are independent, their sizes $`n_0,n_1\ge1`$ are fixed or conditioned on, and the scoring rule is independent of the evaluation sample. The target is simultaneous coverage of the **true population** ROC $`R`$, including its endpoints. The probability includes the fiducial Monte Carlo randomization and any randomized tie-breaking.

No common support, density, or smoothness assumption is needed for the deterministic containment results or for the ELL component's guarantee. Smoothness would be needed for an exterior approximation theorem; no such theorem for the default hybrid is established here. Ties use the randomized ROC estimand and shared tie resolution described in the component documents. The deterministic `tie_break="even"` option does not inherit the randomized-tie theorem.

**Notation.** Let $`\mathcal T=\{t_k=k/n_0:k=0,\ldots,n_0\}`$ be the native grid, $`\mathcal J`$ the production trim indices, $`M`$ the cloud size, and $`a=1-(1-\alpha)^C`$ the fiducial trim level. The default is $`C=1`$, so $`a=\alpha`$. We identify a region $`A`$ with its native-grid index mask. Write $`B_F=[L_F,U_F]`$ for the fiducial **parent actually used in the hull**, after its corner allowances. With default recalibration this is not the unfloored band computed on $`\mathcal J`$. Write $`B_M=[L_M,U_M]`$ for the ELL band at level $`\alpha_M`$.

**Defaults.** The fiducial API itself defaults to `m3_floor=False`. Once enabled, `M3Floor()` selects:

| Setting | Default | Meaning |
|---|---|---|
| `rule` | `"exact"` | Binomial-tail left criterion and binomial-inversion right margin |
| `eps` | $`10^{-3}`$ | Threshold in the left-cut numerical criterion |
| `delta` | $`0.025`$ | One-sided right-boundary error budget |
| `alpha` | `None` | Use the requested band level as $`\alpha_M`$ |
| `split_ratio` | `"sample_size"` | ELL class allocation from the class sizes |
| `closure` | `"tighten"` | Forward maximum of the lower hull; reverse minimum of the upper hull |
| `recalibrate_trim` | `True` | Recompute depth on $`\mathcal J\setminus A`$ |

These are separate controls, not a decomposition of $`\alpha`$ into a proved total error budget. In particular, the name `"exact"` refers to the laws used in the region rule; it does not certify the adaptive left cut or the hybrid's full-curve coverage.

---

## 2. Algorithm

### 2.1 Shared Rank Reduction and Cloud

Resolve ties once and form the descending-score label sequence $`\lambda`$. Compute the empirical positive counts $`\hat k_k`$ as in [`fiducial_band.md` §2.1](fiducial_band.md#21-merged-label-sequence-and-empirical-roc), including the sentinel $`\hat k_{n_0}=n_1`$. Generate one fiducial cloud and its inclusive pointwise ranks by §§2.3–2.4 of that document.

Compute the original trim depth $`j_0=j(\mathcal J,a)`$ using its §2.5. Here “original” means the production trim grid: all columns when $`n_0+1\le2001`$, otherwise the thinned grid returned by `production_trim_rows`. It does not mean every native column at large sample sizes. The tube is always evaluated at every native column.

### 2.2 Left Prefix

For a fixed native index $`k`$, the probability that one cloud draw's first negative anchor exceeds $`t_k`$ is

```math
p_k=(1-k/n_0)^{n_0},\qquad p_{n_0}=0.
```

The exact rule selects the smallest index satisfying the numerical binomial-tail criterion

```math
k_L=\min\left\{k\in\{0,\ldots,n_0\}:
  P\{\operatorname{Binomial}(M,p_k)\ge j_0\}\le\epsilon\right\}.
```

The **cutoff itself is included**: the left mask is $`k\le k_L`$. The implementation computes $`p_k`$ with `exp(n0 * log1p(-k / n0))` and the tail with `binom.sf(j0 - 1, M, p_k)`, scanning upward from zero. The criterion always terminates because $`p_{n_0}=0`$.

For fixed $`k`$, the count of end gaps reaching $`t_k`$ is binomial conditional on the labels. However, $`j_0`$ and hence $`k_L`$ are selected from the same cloud. The plug-in criterion is not an exact conditional or unconditional probability statement about the selected cutoff. Nor does an end-gap count alone determine which draws supply the lower order statistic. It must not be implemented or interpreted as “the lower edge is a chord with probability at most $`\epsilon`$.” Theory §10.6, Corollary 10d gives the valid monotonicity statement for this criterion.

### 2.3 Right Saturated Run

Find the first saturated empirical column and the run length:

```math
k_{\mathrm{sat}}=\min\{k:\hat k_k=n_1\},\qquad K=n_0-k_{\mathrm{sat}}.
```

Thus $`K`$ counts negatives below the smallest positive in the shared label sequence. It is not the number of saturated grid columns, which includes the terminal sentinel.

The exact rule uses

```math
s_U(K;\delta)=
\begin{cases}
\operatorname{Beta}^{-1}_{1-\delta}(K+1,n_0-K),&K<n_0,\\
1,&K=n_0,
\end{cases}
\qquad
k_R=\max\{0,n_0-\lceil n_0s_U(K;\delta)\rceil\}.
```

The right mask is $`k\ge k_R`$, including its start. In floating-point code the expanded length is `ceil(n0 * s_upper - 1e-9)`; the subtraction is a rank-scale rounding tolerance, not an additional statistical allowance. When $`K=n_0`$, the code returns $`k_R=0`$ directly. When $`K=0`$, the Beta bound still gives a nonzero margin.

For continuous scores, conditional on the positive sample, $`K\sim\operatorname{Binomial}(n_0,S)`$ with $`S=F(Y_{\min})`$. For a prespecified $`\delta`$, binomial inversion gives $`P(S>s_U)\le\delta`$. Ideal outward rounding therefore encloses the true terminal interval $`[1-S,1]`$ with probability at least $`1-\delta`$. Theory Proposition 10a gives a separate auxiliary-coordinate interpretation valid for every within-gap completion. Neither result controls the fiducial band outside the selected region. The numerical implementation inherits the usual Beta-quantile and rounding qualifications of the component algorithms.

### 2.4 Region and Comparison Rule

Freeze the native-grid mask

```math
A=\{k:k\le k_L\}\cup\{k:k\ge k_R\}.
```

The two pieces may overlap or meet with no intervening grid point; in that case every column is floored. Do not rebuild the mask after recalibration.

The alternative `rule="stage_f"` uses

```math
k_L=\min\{n_0,\lceil\log(M+1)\rceil\},\qquad
k_R=\max\{0,n_0-K-\lceil2\sqrt{\max(K,1)}\rceil\}.
```

The $`\max(K,1)`$ is part of the implementation: at $`K=0`$ its right margin is two ranks. This rule ignores `eps` and `delta` in computing the mask, although `M3Floor` still validates them. Selecting `"stage_f"` changes only the cut rule; it does not automatically restore the historical split, closure, or trim policy. To reproduce the frozen Stage F combination, also set `split_ratio=0.5`, `closure="widen"`, and `recalibrate_trim=False`, with the study's draw budget.

### 2.5 Complement Trimming

If recalibration is enabled, let $`\mathcal J'=\mathcal J\setminus A`$. When $`\mathcal J'`$ is nonempty, compute $`j_*=j(\mathcal J',a)`$ on the **same cloud**, with the same inclusive ranks, draw count, and effective level. If recalibration is disabled or $`\mathcal J'`$ is empty, use $`j_*=j_0`$. This is a fallback to the original depth, not a median-depth convention and not an instruction to return ELL alone.

Removing trim columns can only increase each draw's minimum rank depth, so $`j_*\ge j_0`$ (theory Proposition 10b). The complement must filter the existing production trim indices. Constructing a new thinning schedule on the complement would change the algorithm and could invalidate this nesting argument.

Build the pointwise order-statistic tube at $`j_*`$ on **all** native columns. Apply the fiducial corner allowances using $`\ell_*=j_*/(M+1)`$, as in [`fiducial_band.md` §§2.6–2.8](fiducial_band.md#26-order-statistic-tube). The allowance must be recomputed at the selected depth. Denote the resulting monotone band by $`B_F`$.

### 2.6 ELL Component and Regional Hull

Use $`\alpha_M=\alpha`$ when `floor.alpha is None`, otherwise the supplied `floor.alpha`. Build the ELL band directly from the **same tie-resolved labels**, using [`ell_band.md` §§2.3–2.7](ell_band.md#23-one-sample-ell-bounds) and the configured class split. The default is

```math
\rho=\frac{n_0^{-1/2}}{n_0^{-1/2}+n_1^{-1/2}}.
```

The ELL level and allocation must be specified without inspecting the ranks or cloud; the sample-size allocation is allowed because it depends only on the fixed class sizes. The hybrid does not enable `assume_r0_zero`, so the ELL upper edge at FPR zero remains unpinned.

Take the hull on $`A`$:

```math
\widetilde L_k=
\begin{cases}\min(L_{F,k},L_{M,k}),&k\in A,\\L_{F,k},&k\notin A,\end{cases}
\qquad
\widetilde U_k=
\begin{cases}\max(U_{F,k},U_{M,k}),&k\in A,\\U_{F,k},&k\notin A.\end{cases}
```

“Floor” means a lower limit on the band's room: it lowers the lower edge and raises the upper edge before closure. It is not replacement by ELL, intersection with ELL, or flooring only the lower edge.

### 2.7 Monotone Closure

The default `closure="tighten"` returns

```math
L_{T,k}=\max_{i\le k}\widetilde L_i,\qquad
U_{T,k}=\min_{i\ge k}\widetilde U_i.
```

The comparison `closure="widen"` returns

```math
L_{W,k}=\min_{i\ge k}\widetilde L_i,\qquad
U_{W,k}=\max_{i\le k}\widetilde U_i.
```

Both edges are clipped to $`[0,1]`$. Tightening requires monotone fiducial parent edges; `stitch_m3_floor` rejects a downward difference below $`-10^{-9}`$. The parent construction supplies monotone edges, so tightening yields a nonempty band. Both closures preserve $`L(0)=0`$ and $`U(1)=1`$. Do not pin $`U(0)`$ or $`L(1)`$ after closure.

### 2.8 Output Grid and Ties

Close the band on the native grid **before** any output resampling. Return `(fpr_grid, lower, upper)` with the conservative lower-left/upper-right step convention of [`fiducial_band.md` §2.2](fiducial_band.md#22-evaluation-grid). For an output point $`t`$, gather the lower edge at $`\lfloor tn_0\rfloor`$ and the upper edge at $`\min(\lceil tn_0\rceil,n_0)`$. Linear interpolation is not the coverage-preserving extension.

Tie-breaking is shared by the cloud, empirical counts, region, and ELL composition. Calling the public ELL routine independently on tied scores could select different labels and would not reproduce this algorithm. `fiducial_band_suite_rs` returns floor masks on the **native grid even when the returned band grid is resampled**; a mask cannot be indexed against a different grid without an explicit mapping.

---

## 3. Properties

### 3.1 Coverage Guarantees

Let $`V(B;D)`$ mean a strict miss at some native-grid index in $`D`$, and $`q_R(B)=P_R\{V(B;\mathcal T)\}`$. For a monotone truth, coverage on the native grid is equivalent to coverage of its conservative continuum extension. The following statements concern that extension, not a linearly interpolated plot.

**Proposition 1 (Whole-band coverage transfer).** *[Exact; theory §8 and Proposition 10e.]* Assume $`B_M`$ has its stated full-curve guarantee at the prespecified level $`\alpha_M`$. For either closure, any data- or cloud-selected region $`A`$, and the monotone parent $`B_F`$ actually used,

```math
q_R(B_H)\le\min\left\{q_R(B_F),\;
\alpha_M+P_R\{V(B_F;A^c)\}\right\}.
```

*Proof.* The unclosed hull contains $`B_F`$ everywhere. If ELL covers globally and $`B_F`$ covers on $`A^c`$, the hull covers globally. Widening preserves this event. Tightening covers a monotone truth exactly when the hull does (Proposition 2 below). The two event inclusions and a union bound prove the result. No independence between the region, cloud, and bands is needed. $`\square`$

The default's exterior term concerns the **recalibrated parent**, not the original raw band. Neither $`\epsilon`$ nor $`\delta`$ bounds it. Setting $`\alpha_M=\alpha/2`$ and trimming at $`\alpha/2`$ does not produce an honest $`1-\alpha`$ hybrid without a separate exterior theorem. ELL calibration and its numerical qualifications are those of [`ell_band.md`](ell_band.md).

**Regional coverage depends on closure.** The unclosed hull and widened band contain ELL on $`A`$, so their in-region miss probability is at most $`\alpha_M`$. For either direction separately, the sharper ELL bound is $`1-\sqrt{1-\alpha_M}`$ (theory Proposition 12i). These regional claims do **not** generally hold after tightening: an exterior lower edge can raise the lower edge inside the right region, and an exterior upper edge can lower it inside the left region.

For example, take five grid columns, $`A=\{0,3,4\}`$, parent edges $`L_F=(0,.2,.6,.6,1)`$, $`U_F=(.8,.9,.95,1,1)`$, and a hull partner $`B_M=[0,1]`$ everywhere. The monotone truth $`R=(0,.2,.4,.5,1)`$ lies in $`B_M`$ globally. Tightening returns $`L_T=(0,.2,.6,.6,.6)`$, which misses at index 3 **inside** $`A`$. The unclosed hull already missed at exterior index 2; tightening has moved that failure into the region without changing simultaneous coverage. This is a deterministic counterexample to a general regional-containment argument, not a measured frequency claim about ELL.

Directionally, Proposition 1 still holds for either closure with $`q_R(B_F)`$ and the exterior probability replaced by their corresponding one-sided versions and $`\alpha_M`$ replaced by $`1-\sqrt{1-\alpha_M}`$. Tightening preserves each global one-sided coverage event. An ELL-only regional cap can be recovered on the joint event that the corresponding parent exterior miss is absent; that is not a conditional calibration claim.

### 3.2 Closure and Exterior Identity

**Proposition 2 (Tightening preserves simultaneous coverage and leaves the exterior unchanged).** *[Exact; theory Proposition 10e, with the exterior identity derived here.]* For a monotone parent and any region,

```math
B_F\subseteq B_T\subseteq\widetilde B\subseteq B_W,
\qquad B_T(t_k)=B_F(t_k)\quad(k\notin A).
```

Moreover, $`B_T`$ and $`\widetilde B`$ cover exactly the same monotone truths on the grid.

*Proof.* Since $`\widetilde L_i\le L_{F,i}\le L_{F,k}`$ for $`i\le k`$, $`L_{T,k}\le L_{F,k}`$. The upper-edge argument is symmetric, proving parent containment. Tightening lies inside the hull by construction, and widening contains it. At an exterior column the hull equals the parent, forcing equality there. Finally, if a monotone $`R`$ covers every hull lower edge, then $`\widetilde L_i\le R_i\le R_k`$ for $`i\le k`$, so $`L_{T,k}\le R_k`$; the upper argument is symmetric. The converse follows from containment. $`\square`$

Consequently, the default hybrid's exterior miss event is **exactly** its recalibrated parent's exterior miss event on the native grid. Tightening does not add room there. Widening can add room outside $`A`$ and can cover truths that the hull misses. The precise coverage difference is

```math
q_R(B_T)-q_R(B_W)
=P_R\{B_W\text{ covers},\ \widetilde B\text{ misses}\}\ge0.
```

This identity measures a tradeoff; it does not bound its size. Equality of the default band and parent outside the mask need not carry over to a coarser resampling whose adjacent native indices straddle the mask.

### 3.3 Recalibration and Nesting

**Proposition 3 (Same-cloud recalibration narrows the complete hybrid).** *[Exact; theory Proposition 10b and Corollary 10c, extended to either closure.]* Hold the cloud, level, empirical counts, floor mask selected from $`j_0`$, ELL band, and closure fixed. Then

```math
j_*\ge j_0,\qquad
B_{H,\mathrm{recal}}\subseteq B_{H,\mathrm{original}}.
```

*Proof.* Restricting the trim domain increases each draw's minimum depth and hence cannot decrease its selected order statistic. A larger trim depth raises the lower tube edge and lowers the upper tube edge at every column. The CP-form upper allowance also decreases as $`j/(M+1)`$ increases; the fixed zero-lower prefix and monotone repair preserve nesting. Hulls with the same ELL band and both kinds of cumulative closure preserve these edge orderings. The empty-complement fallback gives equality. $`\square`$

Recalibration therefore cannot improve coverage relative to its otherwise identical original-depth hybrid on a coupled realization. It can still repair misses of the original **unfloored** band, since adding ELL and recalibrating act in opposite directions. There is no general containment between the default hybrid and the unfloored band.

For a fixed parent and region, decreasing $`\alpha_M`$ widens the ELL component and hence the hybrid under either closure, subject to the component's numerical calibration. Enlarging the region also widens a fixed-parent hull. Under default recalibration, changing region budgets changes the parent too, so the latter fact does not imply monotonicity of the complete procedure in `eps` or `delta`. Shared-cloud nesting across requested $`\alpha`$ levels is likewise not established for this adaptive hybrid.

### 3.4 Full-Grid Floors and Remaining Theory

If $`A=\mathcal T`$, the complement is empty and the algorithm takes the full pointwise hull of the original fiducial band and ELL. Both parent bands are monotone, so the hull already is monotone and either closure leaves it unchanged. It contains ELL everywhere; it equals ELL only if ELL contains the fiducial band. Perfect separation gives $`K=n_0`$ and therefore this case under either cut rule. This removes the forced separated-path failures used in [`fiducial_band.md` Proposition 2](fiducial_band.md#31-coverage-guarantees) whenever ELL covers; it does not establish conditional $`1-\alpha_M`$ coverage given separation.

More generally,

```math
P_R\{V(B_H;\mathcal T),\ A=\mathcal T\}\le\alpha_M.
```

If the floor is the full grid for every realization, the hybrid has global coverage at least $`1-\alpha_M`$. For a random full-grid event, the displayed joint bound does not imply the same conditional bound after selecting that event.

The interior theorem for the fiducial band cannot simply be applied to $`\mathcal J\setminus A`$. That domain is random, its cuts move toward boundaries in some sampling regimes, and the production trim grid is thinned. A fixed regular-interior approximation does not control this exterior domain uniformly. Theory §10.4 identifies a sufficient extension: an event $`E`$ enclosing the relevant endpoint-completion regions with $`P(E^c)\le\delta_{\mathrm{cut}}`$, together with

```math
P_R\{V(B_F;A^c)\cap E\}\le\alpha_I
\quad\Longrightarrow\quad
q_R(B_H)\le\alpha_M+\delta_{\mathrm{cut}}+\alpha_I.
```

The exterior bound remains open. The default adaptive left criterion does not by itself supply a declared $`\delta_{\mathrm{cut}}`$. The exact right-boundary certificate controls a location, not the band's height or all missing features. The tail-height certificates in theory §10.3 and smooth-truth depth in §6.6 are possible ingredients for separately calibrated methods, not operations performed by this hybrid.

### 3.5 Width and Structural Properties

The hybrid is rank-invariant with shared tie resolution and randomization. Its edges are monotone, lie in $`[0,1]`$, and satisfy $`L\le U`$. The reported continuum band consists of opposite-side steps, as specified in the component documents.

Relative to the selected parent, either closure adds nonnegative width. Relative to the original unfloored band, default recalibration can make the net area change positive or negative. Recalibration and tightening each reduce area in their respective coupled comparisons; neither establishes a universal advantage over ELL or KS.

For reported grid points $`t_k`$, the conservative extension's area is

```math
W_{\mathrm{step}}(B)=\sum_k(t_{k+1}-t_k)\,[U(t_{k+1})-L(t_k)].
```

Simulation tables instead use trapezoidal integration of $`U(t_k)-L(t_k)`$; see [`fiducial_band.md` §2.2](fiducial_band.md#22-evaluation-grid) and [`ell_band.md` §3.3](ell_band.md#33-width-and-tail-adaptivity). Report post-closure area using an identified representation. The selected fraction of grid points is not the area cost: widening can propagate beyond the mask, tightening can remove part of the hull's room, and conservative step area couples adjacent columns.

### 3.6 Summary of Guarantees

| Property | Default hybrid | Scope |
|---|---|---|
| Unrestricted finite-sample population coverage | Not established | ELL level plus an unbounded parent exterior term |
| Contains its selected fiducial parent | Yes | The recalibrated parent, not generally the original raw band |
| Contains ELL on the floor region | Not generally after tightening | True before closure and under widening |
| In-region miss probability bounded by ELL level alone | Not established after tightening | True for the hull and widened band |
| Same coverage event as the unclosed hull | Yes | Monotone truths, including each global one-sided event |
| Equals its parent outside the mask | Yes | Native grid, tightened closure |
| Recalibrated hybrid nests inside original-depth hybrid | Yes | Same cloud, region, ELL component, level, and closure |
| Full-grid floor contains ELL | Yes | Equality with ELL is not automatic |
| Rank invariance and monotone valid edges | Yes | Component assumptions and shared tie resolution |
| Asymptotic calibration of the default hybrid | Not established | Requires control on the random exterior domain |
| Main-suite coverage | No significantly under-covering cell | Tested design, tolerance, and configurations in §3.7 |

### 3.7 Simulation Evidence

**Main suite, current defaults. [Empirical.]** The design is specified in [`simulation_spec.md`](simulation_spec.md); the component documents give its seven families, Latin-hypercube sampling, and parameter ranges. The runs in [`nov11`](../data/results/nov11/), [`nov11_p30`](../data/results/nov11_p30/), and [`nov11_p10`](../data/results/nov11_p10/) record `M3Floor()` settings in their metadata and use $`M=20{,}000`$, $`C=1`$. Select the matching prevalence within each directory; `nov11` also contains an additional 10% prevalence run at $`n=1{,}000`$.

The following summaries use $`n=30,100,300,1{,}000,3{,}000,10{,}000`$: 42 family-by-size cells per prevalence, with 6,000 datasets per cell (5,995 for Student-$`t`$). Coverage is checked on the native grid excluding **both** endpoints, counting only TPR exceedances greater than $`10^{-6}`$. These diagnostics do not check strict full-curve coverage at zero tolerance. “Cells below” counts 95% Wilson intervals entirely below nominal across all 126 cell–level combinations. Coverage columns are equal-weight cell means; the area column is the mean of cell mean-area ratios to KS at $`\alpha=.05`$, using trapezoidal grid area.

| Prevalence | Coverage, 95% | Coverage, 80% | Coverage, 50% | Worst cell, 95% | Cells below | Area ÷ KS, 95% |
|---|---|---|---|---|---|---|
| 50% | 0.982 | 0.910 | 0.723 | 0.969 | 0 | 0.59 |
| 30% | 0.981 | 0.907 | 0.717 | 0.969 | 0 | 0.60 |
| 10% | 0.980 | 0.905 | 0.716 | 0.967 | 0 | 0.68 |

The lack of detected under-coverage is evidence for these tested mixtures of shapes, not a distribution-free theorem. At nominal 50% coverage the hybrid is still substantially conservative. The [repair](../figures/readme/hybrid_repair.png), [miss-location](../figures/readme/hybrid_misses.png), and [floor-cost](../figures/readme/floor_cost.png) figures show which strata benefit and where the added width lies. Observed in-region miss rates remain useful diagnostics under tightening, but they should not be interpreted as validation of an ELL-only regional bound.

**Ablations and earlier variants. [Empirical.]** The [complement-trim report](experiments/complement_trim_report.md) isolates recalibration with widening and the earlier ELL allocation on 50 deliberately adversarial cells. Macro coverage at $`\alpha=.05`$ changes from .980 to .976 while mean paired area falls by 1.9%; at $`\alpha=.5`$, coverage changes from .729 to .680 and area falls by 3.5%. These are measurements of that configuration, not the current-default table above.

The [closure/split report](experiments/closure_split_report.md) separates closure from ELL class allocation. In its 46 typical closure cells, tightening the recalibrated hybrid changes coverage from .980 to .978 at $`\alpha=.05`$, with paired area ratio .944; at $`\alpha=.5`$, coverage changes from .716 to .686 with ratio .925. The split study separately supports using the sample-size allocation under imbalance. Historical statements that a full-region floor “is M3” describe observed equality in those experiments, not a general identity of the algorithm (§3.4).

---

## 4. Computational Considerations

### 4.1 Complexity and Shared Computation

Cloud generation, pointwise ranking, and tube extraction dominate, with the costs in [`fiducial_band.md` §4](fiducial_band.md#4-computational-considerations). The hybrid does not require a second cloud. With cached pointwise depths, complement trimming adds a minimum reduction over at most $`M|\mathcal J|`$ integers and an order-statistic selection. The left-cut scan takes at most $`n_0+1`$ binomial-tail evaluations, the right cut one Beta quantile, and the hull and closure $`O(n_0)`$ operations. ELL calibration is cached by class size and class level; its first-call cost can dominate, as described in [`ell_band.md` §4](ell_band.md#4-computational-considerations).

The Rust suite computes all original and recalibrated depths before extracting tubes. It reuses a tube for variants that select the same depth, then applies each region, ELL component, and closure. Multiple levels and floor configurations therefore share the generated cloud and rank cache. It does not redraw or reduce $`M`$ at larger $`\alpha`$.

### 4.2 Memory and Reproducibility

The main storage remains the float32 cloud and, when needed, the integer depth cache described in [`fiducial_band.md` §§4.2–4.3](fiducial_band.md#42-memory). Each region mask and pair of output edges requires $`O(n_0)`$ additional storage. Monte Carlo resolution and float32 ties have the same qualifications as the fiducial component; the floor does not eliminate them.

`fiducial_band_suite_rs` returns a `FiducialSuite` with `grid`, `bands[alpha][name]`, and `floor_regions[alpha][name]`. With the same integer seed and remaining arguments, each band's values match its separate `fiducial_band_rs` call. Reusing an already-advanced `numpy.random.Generator` is not the same initialization. Per-draw Rust RNG streams make the result independent of thread count. The Python/Torch reference path follows the same hybrid sequence but uses a different cloud RNG and is not bit-identical. Its floor still calls the Rust ELL implementation, so enabling the floor requires the extension there too.

### 4.3 Parameters and Porting Requirements

The inherited API is specified in [`fiducial_band.md`](fiducial_band.md). Floor-specific validation requires `rule` in `{"exact", "stage_f"}`, $`0<\epsilon<1`$, $`0<\delta<1/2`$, an optional $`0<\alpha_M<1`$, a valid fixed ELL split or `"sample_size"`, and `closure` in `{"widen", "tighten"}`. Use a Boolean for `recalibrate_trim`. `m3_floor=False` disables the method; `True` resolves to `M3Floor()`.

For a production port, preserve the shared labels and cloud, inclusive cut indices, original production trim subset, frozen region, empty-complement fallback, allowances at the selected depth, and closure-before-resampling order. The native mask is the **hull application region**, not a certificate that the final tightened band contains ELL there. Diagnostics should distinguish $`j_0`$, $`j_*`$, the mask, and the closure. The current suite exposes masks and bands but does not return the depths as public result fields.

For paired native-grid outputs:

```python
from studroc_paper.methods import M3Floor, fiducial_band_suite_rs

suite = fiducial_band_suite_rs(
    y_true=y_true,
    y_score=y_score,
    alphas=[0.05],
    m3_floors={"raw": False, "hybrid": M3Floor()},
    n_draws=20_000,
    random_state=0,
)
lower, upper = suite.bands[0.05]["hybrid"]
region = suite.floor_regions[0.05]["hybrid"]
```

The implementation sources are [`hybrid_floor.py`](../src/studroc_paper/methods/hybrid_floor.py), [`fiducial_band_rs.py`](../src/studroc_paper/methods/fiducial_band_rs.py), and the reference path in [`fiducial_band.py`](../src/studroc_paper/methods/fiducial_band.py). [`test_hybrid_floor.py`](../tests/test_hybrid_floor.py) covers cuts, fallback behavior, closure, nesting, resampling, and the two execution paths; [`test_fiducial_band_rs.py`](../tests/test_fiducial_band_rs.py) covers shared-cloud behavior.

---

## 5. Complete Pseudocode

The named component operations below are the algorithms specified in the linked fiducial and ELL documents. Indices are zero-based except for the one-based trim depths.

```text
HYBRID(y, score, alpha=.05, M=20000, C=1, floor=M3Floor(), output_size=None):
    validate the inherited arguments and floor settings
    labels = merged labels after one shared tie-resolution step
    n0, n1 = class sizes
    grid[k] = k / n0, k = 0,...,n0
    khat = empirical positive counts, with khat[n0] = n1
    J = production_trim_rows(n0 + 1), or all columns if it returns None
    cloud = FIDUCIAL_CLOUD(labels, M, rng)
    a = 1 - (1 - alpha)^C
    j0 = TRIM_DEPTH(cloud, J, a)

    k_sat = first k with khat[k] == n1
    K = n0 - k_sat
    if floor.rule == "exact":
        kL = first k with BinomialSF(j0 - 1; M, (1 - k/n0)^n0) <= floor.eps
        if K == n0:
            kR = 0
        else:
            sU = BetaQuantile(1 - floor.delta; K + 1, n0 - K)
            kR = max(0, n0 - ceil(n0 * sU - 1e-9))
    else:
        kL = min(n0, ceil(log(M + 1)))
        kR = max(0, n0 - K - ceil(2 * sqrt(max(K, 1))))
    A[k] = (k <= kL) or (k >= kR)

    j_star = j0
    if floor.recalibrate_trim:
        J_prime = [k in J where not A[k]]
        if J_prime is not empty:
            j_star = TRIM_DEPTH(cloud, J_prime, a)
    assert j_star >= j0

    LF, UF = POINTWISE_TUBE(cloud, j_star, all native columns)
    LF, UF = FIDUCIAL_CORNER_ALLOWANCES(LF, UF, khat, n1, j_star, M)
    alphaM = alpha if floor.alpha is None else floor.alpha
    LM, UM = ELL_FROM_LABELS(labels, alphaM, floor.split_ratio,
                            assume_r0_zero=False, native_grid=True)
    hullL = where(A, min(LF, LM), LF)
    hullU = where(A, max(UF, UM), UF)
    if floor.closure == "tighten":
        L = forward_cumulative_max(hullL)
        U = reverse_cumulative_min(hullU)
    else:
        L = reverse_cumulative_min(hullL)
        U = forward_cumulative_max(hullU)
    L, U = clip(L, 0, 1), clip(U, 0, 1)
    if output_size is not None:
        output_grid = linspace(0, 1, output_size)
        L = L[floor(output_grid * n0)]
        U = U[min(ceil(output_grid * n0), n0)]
        grid = output_grid
    return grid, L, U
```

The suite retains `A` on the native grid as additional output. Neither the region nor the cloud is regenerated after selecting `j_star`.

---

## 6. Formal Specification

### 6.1 Algorithm 1: Localized ELL–Fiducial Hybrid

**Input:** Binary labels and real scores satisfying §1; $`\alpha\in(0,1)`$; $`M\ge100`$; $`C>0`$; the floor settings in §1; optional output grid size at least two; shared tie-breaking and RNG state.

**Output:** A grid and monotone edges $`0\le L\le U\le1`$, with $`L(0)=0`$, $`U(1)=1`$, interpreted by conservative step extension. The target is population simultaneous coverage; the proved bound is Proposition 1.

#### 1. Original Depth

Reduce the sample to $`\lambda`$ and $`\hat k`$. Generate the cloud and compute $`j_0=j(\mathcal J,1-(1-\alpha)^C)`$ using the fiducial specification.

#### 2. Frozen Region

Compute $`K=n_0-\min\{k:\hat k_k=n_1\}`$. Use §2.2 and §2.3, or the Stage F formulas of §2.4, to construct $`A=\{0,\ldots,k_L\}\cup\{k_R,\ldots,n_0\}`$.

#### 3. Selected Parent

Use $`j_*=j(\mathcal J\setminus A,a)`$ if recalibration is enabled and the complement is nonempty; otherwise $`j_*=j_0`$. Build the full native-grid tube and apply all fiducial allowances at $`j_*`$.

#### 4. ELL Hull

Build ELL at $`\alpha_M`$ from the same labels and the specified class split. Take the pointwise hull on $`A`$, leaving the selected parent elsewhere.

#### 5. Closure and Reporting

Apply the configured cumulative closure, clip to $`[0,1]`$, and only then resample conservatively if requested. Preserve the native-grid mask in any region diagnostics.

---

## References

- [`fiducial_band.md`](fiducial_band.md): fiducial cloud, rank-depth trimming, allowances, and inherited computational conventions.
- [`ell_band.md`](ell_band.md): ELL calibration, class allocation, composition, and its finite-sample guarantee.
- [`fiducial_band_theory.md`](fiducial_band_theory.md), §§8, 10–12: hybrid coverage transfer, boundary certificates, closure, nesting, and ELL directional bounds.
- [`complement_trim_report.md`](experiments/complement_trim_report.md) and [`closure_split_report.md`](experiments/closure_split_report.md): paired ablations and their configurations.
- [`simulation_spec.md`](simulation_spec.md): current-default simulation design and evaluation metrics.
