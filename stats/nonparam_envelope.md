# Studentized Bootstrap Envelope Simultaneous Confidence Bands for ROC Curves

## Abstract

We present a nonparametric method for constructing simultaneous confidence bands (SCB) for the **true population** ROC curve using a studentized bootstrap envelope. The method retains the $(1-\alpha)$ fraction of bootstrap curves most consistent with the empirical ROC (using either studentized Kolmogorov-Smirnov statistics or symmetric tail trimming) and returns their pointwise envelope. The resulting band is asymmetric, adapts to local variance (incorporating a variance floor for stability), and inherits the step-function structure of the empirical ROC. At extreme FPR — where the dominant uncertainty is the location of the operating threshold rather than binomial TPR noise — the lower band is additionally floored by an exact, distribution-free Beta order-statistic bound.

---

## 1. Setup and Assumptions

**Data:** $\mathcal{D} = \{(y_i, s_i)\}_{i=1}^N$ where $y_i \in \{0,1\}$ is the class label and $s_i \in \mathbb{R}$ is the score. Let $n_0$ and $n_1$ denote the number of negatives and positives.

**Assumptions:**
- A1: Independent sampling within each class
- A2: Continuous score distributions (no ties)
- A3: Higher scores indicate positive class
- A4: Finite variance of TPR at any fixed FPR

**Target:** Construct $\mathcal{B}_\alpha(t) = [L(t), U(t)]$ such that:

$$P\left(\forall t \in [0,1]: R_{\text{true}}(t) \in \mathcal{B}_\alpha(t)\right) \geq 1 - \alpha$$

---

## 2. Algorithm

### 2.1 Empirical ROC

Compute the empirical ROC curve $\hat{R}(t)$ and its FPR jump points $\mathcal{J}_0 = \{0, 1/n_0, 2/n_0, \ldots, 1\}$.

### 2.2 Evaluation Grid

Construct a common grid $\mathcal{T}$ for curve evaluation:


**Key Insight:** Both the empirical ROC $\hat{R}(t)$ and any bootstrap ROC $R_b(t)$ are piecewise constant with jumps *only* at FPR values $\{0, 1/n_0, 2/n_0, \ldots, 1\}$. This is because the bootstrap resamples $n_0$ negatives with replacement, preserving the original grid of potential false positive rates.

Therefore, evaluating on the set of all possible jump points is sufficient to capture the exact supremum distance.

| Strategy | Grid $\mathcal{T}$ | Discretization Error $\delta_K$ | Memory |
|----------|-------------------|--------------------------------|--------|
| **Full** | $\{0, \frac{1}{n_0}, \frac{2}{n_0}, \ldots, 1\}$ | **0** (exact) | $B \times (n_0 + 1)$ |
| **Uniform** | $\text{linspace}(0, 1, K)$ | $D / K$ | $B \times K$ |

The **Full** grid provides exact evaluation but scales with sample size $n_0$. The **Uniform** grid allows controlling memory usage ($K$) independent of sample size, introducing a controlled discretization error.

### 2.3 Stratified Bootstrap

For $b = 1, \ldots, B$:
1. Resample $n_0$ negatives with replacement
2. Resample $n_1$ positives with replacement
3. Compute bootstrap ROC $R_b(t)$
4. Evaluate $R_b$ on grid $\mathcal{T}$

### 2.4 Variance Estimation

For each $t \in \mathcal{T}$:

$$\hat{\sigma}_{\text{boot}}^2(t) = \frac{1}{B-1} \sum_{b=1}^{B} \left(R_b(t) - \bar{R}(t)\right)^2$$

where $\bar{R}(t) = \frac{1}{B}\sum_b R_b(t)$.

**Variance Floor:**
At boundaries (FPR near 0 or 1), the bootstrap distribution often collapses to a single value, yielding $\hat{\sigma}_{boot}(t) \approx 0$. This causes instability in studentization. We support multiple methods for imposing a minimum variance floor:

**Wilson Score Variance Floor (`boundary_method="wilson"`)**

Based on the Wilson score interval for a binomial proportion $p = \hat{R}(t)$ with sample size $n_1$:

$$\sigma^2_{wilson}(p) = \frac{1}{(1 + z^2/n_1)^2} \cdot \left(\frac{p(1-p)}{n_1} + \frac{z^2}{4n_1^2}\right)$$

where $z = \Phi^{-1}(1-\alpha/2)$ is the normal quantile.

The implemented `boundary_method` options are `"none"`, `"wilson"`, and `"ks"` (the last affects only post-envelope boundary extension, §2.8, not the variance floor). A Hsieh-Turnbull variance floor — replacing $\sigma^2_{wilson}$ with the full asymptotic variance $R(1-R)/n_1 + (g(c_t)/f(c_t))^2\, t(1-t)/n_0$ estimated via density estimation — was considered but is not part of this implementation; the HT variance is used in the separate `hsieh_turnbull_band` method. Note that follow-up experiments (`project_evaluation_report.md` D.9) show the HT formula with the true slope matches Monte Carlo truth to 1–3% at every probed FPR, while the Wilson (binomial-only) variance is 3–6× too small in sd at the steep low-FPR corner; this gap is the central limitation of the Wilson floor.

**Variance Floor Application:**

The final variance used for studentization is:

$$\hat{\sigma}^2(t) = \max\left(\hat{\sigma}^2_{boot}(t), \sigma^2_{floor}(t)\right)$$

### 2.5 Studentized KS Statistics

To measure the "strangeness" of each bootstrap curve, we compute its maximum studentized deviation from the empirical curve.

**Epsilon Regularizer:**
We define a regularization parameter $\epsilon = \min(1/N, 10^{-6})$, where $N = n_0 + n_1$. This serves as a lower bound on meaningful deviations, ensuring we do not amplify numerical noise or irrelevant micro-fluctuations when variance is extremely low. It is derived from the smallest possible probability mass in the sample space.

**Studentization with Low-Variance Handling:**
For each bootstrap curve $b$, we compute the pointwise studentized deviation $z_b(t)$.
1. Calculate signed deviation: $\delta_b(t) = R_b(t) - \hat{R}(t)$.
2. **Normal Case:** If $\hat{\sigma}(t) \geq \epsilon$:

   $$z_b(t) = \frac{\delta_b(t)}{\hat{\sigma}(t)}$$

3. **Low-Variance Case:** If $\hat{\sigma}(t) < \epsilon$ (variance is effectively zero):

$$
z_b(t) = \begin{cases} 
0 & \text{if } |\delta_b(t)| < \epsilon \text{ (noise)} \\ 
\frac{\delta_b(t)}{\epsilon} & \text{if } |\delta_b(t)| \geq \epsilon \text{ (significant shift)} 
\end{cases}
$$

For each curve, the global statistic is $Z_b = \sup_{t \in \mathcal{T}} |z_b(t)|$.


### 2.6 Curve Retention

We support two methods for determining which curves to retain.

**Option A: Original KS Retention (`retention_method="ks"`)**
Retain the $(1-\alpha)$ fraction of curves with the smallest maximum absolute studentized deviation $Z_b$.

$$\mathcal{R}_\alpha = \left\lbrace R_b : Z_b \leq Z_{(\lceil (1-\alpha)B \rceil)} \right\rbrace$$

where $Z_{(k)}$ is the $k$-th order statistic. This creates a band of "most typical" curves in terms of global shape deviation.

**Option B: Symmetric Retention (`retention_method="symmetric"`)**
The standard KS method can be asymmetric at boundaries (e.g., at high AUC, curves can't deviate upward past 1, but can deviate downward). To ensure balanced tail coverage:
1. Compute *signed* studentized deviations: $s_b(t)$ using the signed difference $(R_b(t) - \hat{R}(t))$ in the numerator.
2. For each curve $b$, find its max upward deviation $M^+_b = \sup_t s_b(t)$ and max downward deviation $M^-_b = \inf_t s_b(t)$.
3. Determine thresholds $q_{up}$ and $q_{down}$ as the $(1-\alpha/2)$ and $(\alpha/2)$ quantiles of $M^+$ and $M^-$ respectively.
4. Retain curves that satisfy:

$$
M^{-}_{b} \geq q_{\text{down}} \quad \text{AND} \quad M^{+}_{b} \leq q_{\text{up}}
$$

This method explicitly trims the most extreme $\alpha/2$ upward excursions and $\alpha/2$ downward excursions.

### 2.7 Envelope Construction and Tail Floors

Compute the pointwise min and max of the retained curves $\mathcal{R}_\alpha$:

$$L(t) = \min_{R_b \in \mathcal{R}_\alpha} R_b(t)$$
$$U(t) = \max_{R_b \in \mathcal{R}_\alpha} R_b(t)$$

**Wilson Rectangle Floor (variance-ratio gated; `boundary_method="wilson"`):**
After envelope construction and clipping, the band is widened where the bootstrap variance has collapsed relative to the binomial model. (An earlier draft of this spec described a symmetric $\hat{R}(t) \pm \sigma_{floor}(t)$ width extension; that is **not** what is implemented.) The implemented procedure:

1. Compute the pointwise deficiency in probability space (regardless of whether the envelope was built in logit space):
   $$d(t) = \max\left(0,\; 1 - \frac{\hat{\sigma}^2_{boot}(t)}{\sigma^2_{wilson}(t)}\right)$$
2. Compute the continuous effective dimensionality $K_{\text{eff}} = \sum_{t \in \mathcal{T}} d(t)$ and the Šidák-corrected level $\alpha_w = 1 - (1-\alpha)^{1/K_{\text{eff}}}$ (when $K_{\text{eff}} > 1$).
3. Compute the Wilson Rectangle band (2D Wilson score rectangles at each operating point, Šidák-corrected across the rectangle's two margins) at level $\alpha_w$.
4. At every grid point with $d(t) > 0$, take the union with the rectangle bounds:
   $$L(t) \leftarrow \min(L(t), L_{\text{rect}}(t)), \qquad U(t) \leftarrow \max(U(t), U_{\text{rect}}(t))$$
5. Enforce band monotonicity: cumulative max left-to-right on $U$, cumulative min right-to-left on $L$.

**Where the gate actually fires, and a known limitation.** Since $\sigma^2_{wilson}$ is a binomial (vertical) variance, $d(t) > 0$ in practice only where the empirical TPR sits on its plateau near 1 (or at the literal corners), where the bootstrap variance is genuinely ~0 and the binomial model is complete. At the first grid points of a steep curve ($k = 1\text{–}10$ negatives above threshold, mid-range TPR), the bootstrap variance *exceeds* the Wilson variance by 10–60×, so the gate never fires — yet that is precisely where coverage fails, via one-sided bootstrap support collapse and upward bias of $\hat{R}$ (see `project_evaluation_report.md` B.8/D.9). A previous implementation used hard count cutoffs ($k < 15$ or $m < 10$) and applied the rectangle one-sided per tail (raising only $U$ in the lower tail, lowering only $L$ in the upper tail); a 1,400-case paired comparison shows the two designs are behaviorally equivalent, because neither ever floors the *lower* band at the low-FPR failure points. The Beta order-statistic floor below closes exactly this gap.

**Exact Beta Order-Statistic Floor (lower band only; `boundary_method="wilson"`):**
The variance-ratio gate measures vertical (binomial) uncertainty and is therefore blind to the horizontal threshold-location uncertainty that dominates at the first grid points. The lower band receives an additional floor built on an exact, finite-sample, distribution-free law: under A1–A2, the true FPR exceedance at the $j$-th largest negative score satisfies

$$\bar{F}(X_{(j)}) \sim \text{Beta}(j,\; n_0 + 1 - j)$$

Let $q_j$ be the $(1-\alpha_e)$ upper quantile of this law, with per-event level $\alpha_e = \alpha / (2J)$ — Bonferroni over the $2J$ one-sided events ($J$ Beta quantile events plus $J$ one-sided Wilson bounds), $J = 25$, so $\alpha_e = 0.001$ at $\alpha = 0.05$. On the event $\{\bar{F}(X_{(j)}) \le q_j\}$, monotonicity of $R_{\text{true}}$ gives, for every evaluation point $t \ge q_j$:

$$R_{\text{true}}(t) \;\ge\; R_{\text{true}}\big(\bar{F}(X_{(j)})\big) \;=\; G(X_{(j)}) \;\ge\; \text{WilsonLower}_{\alpha_e}\big(\hat{R}(j/n_0)\big)$$

The floor at $t$ is the bound from the *largest* $j$ with $q_j \le t$ — a backward-looking bound anchored at a smaller-FPR operating point whose true FPR sits at or below $t$ with high probability. For $t < q_1$ ($\approx 6.9/n_0$ at $\alpha_e = 0.001$) no order statistic qualifies and the floor is vacuous (0): no distribution-free lower bound exists there. Application:

$$L(t) \leftarrow \min\big(L(t),\; L_\beta(t)\big) \qquad \text{for } 0 < t \le q_J$$

Properties: it is a pure widening of the lower band, restricted to the floor's jurisdiction ($t \le q_J \approx 43/n_0$ at $\alpha_e = 0.001$); it preserves band monotonicity (the floor is non-decreasing in $t$ and the pointwise minimum of monotone functions is monotone); it needs no gate — on flat ROC segments $\hat{R}(j/n_0) \approx \hat{R}(t)$ and the floor costs almost nothing, while on steep segments it widens the lower band in proportion to the realized slope. Under ties (violating A2) the exceedance $\bar{F}(X_{(j)})$ is stochastically *smaller* than the Beta law, so discrete scores err conservative rather than breaking the bound (verified empirically with scores quantized to 20 levels).

### 2.8 Boundary Handling

**Clipping:**
Clip all values to $[0, 1]$.

**Boundary Enforcement:**
We explicitly enforce that the confidence band respects logical ROC constraints:
- $L(0) = 0$
- $U(1) = 1$

**KS-Style Boundary Extension (Optional, `boundary_method="ks"`):**
In regions where bootstrap variance collapses completely (near corners), we can optionally extend the band using fixed width margins derived from the analytical Kolmogorov-Smirnov distribution (Campbell 1994). This connects the computed bootstrap envelope to the corners (0,0) and (1,1) with a theoretical worst-case slope.

---

## 2.9 Logit Space Construction (Optional)

As an alternative to probability space, the entire procedure can be performed in logit space (`use_logit=True`). This stabilizes variance across the ROC curve, particularly at boundaries where TPR is near 0 or 1.

**Haldane-Anscombe Correction:**
To handle boundary values, we apply a continuity correction before the logit transform:
$$\text{logit}_{H}(p) = \log\left(\frac{k + 0.5}{n_1 - k + 0.5}\right)$$
where $k = p \cdot n_1$ is the count of true positives.

**Procedure:**
1. Transform empirical and bootstrap TPR values to logit space using the Haldane correction
2. Compute bootstrap standard deviation in logit space
3. Studentize deviations in logit space
4. Apply retention rule (KS or symmetric)
5. Construct envelope in logit space
6. Back-transform to probability space via sigmoid: $p = \sigma(\text{logit}) = 1/(1 + e^{-\text{logit}})$

**Note:** The Wilson corrections *are* applied on the logit path when `boundary_method="wilson"`: the studentization variance floor is transformed into logit space via the Jacobian $\big(d\,\text{logit}(p)/dp\big)^2 = 1/(p(1-p))^2$, and the post-envelope tail floors (Wilson Rectangle and Beta order-statistic, §2.7) are applied in probability space after back-transforming the envelope. The variance-ratio deficiency is always computed in probability space so the gate is consistent across paths.

**Empirical caution:** the logit-space variants perform much worse than the probability-space construction in the simulation suite (35–40% coverage at the 95% level). The Haldane-Anscombe correction stretches the boundary region in a way that widens the interior without repairing the boundary; the logit path is retained for completeness, not recommended.

---

## 3. Properties

### 3.1 Coverage Guarantees

**Definition (Population Coverage).** The probability that the true population ROC curve falls entirely within the band:
$$P\left(\forall t: R_{\text{true}}(t) \in [L(t), U(t)]\right)$$

**Note on Future Samples:** This method constructs a *confidence band* for the underlying population curve $R_{true}$. It is **not** a *prediction band* for future empirical ROC curves $\hat{R}_{new}$. A future empirical curve has additional sampling variability relative to the current empirical curve (variance approximately doubles), so the coverage of future samples will be significantly lower than $(1-\alpha)$.


---

**Theorem 1 (Asymptotic Population Coverage).** Under A1–A4, as $n = \min(n_0, n_1) \to \infty$:
$$P\left(\forall t: R_{\text{true}}(t) \in [L(t), U(t)]\right) \to 1 - \alpha$$

*Proof sketch:* 

1. The empirical ROC process $\sqrt{n}(\hat{R} - R_{true})$ converges weakly to a Gaussian process $\mathbb{G}$ (Hsieh & Turnbull, 1996).

2. By bootstrap consistency for the empirical process, the conditional distribution of $\sqrt{n}(R_b - \hat{R})$ given the data converges to the same limit $\mathbb{G}$.

3. The finite-sample bias $E[\hat{R}(t)] - R_{true}(t) = O(n^{-1})$ vanishes faster than the $O(n^{-1/2})$ standard deviation, so it becomes negligible in the standardized process.

4. The retention rule keeps curves whose studentized supremum deviation is below the $(1-\alpha)$ quantile. By the bootstrap principle and (3), $R_{true}$ falls within the envelope with probability approaching $1-\alpha$. $\square$

**Remark (where Theorem 1's argument is weakest in practice).** The weak-convergence and bootstrap-consistency steps hold uniformly on compact subintervals of $(0,1)$, but the supremum in the method runs over the *full* grid, including the moving boundary points $t = k/n_0$ for small $k$. At those points the relevant statistics are extreme order statistics, not empirical-process averages: the Gaussian approximation does not apply, the bootstrap deviation distribution is one-sided (resampled extremes cannot exceed observed extremes), and the bias of $\hat{R}$ is a non-vanishing fraction of the local standard deviation (see §3.2). Empirically, this is exactly where finite-sample coverage was lost at large $n$ before the Beta order-statistic floor was added (coverage 0.915 at $n = 1{,}000$ and 0.830 at $n = 10{,}000$ at prevalence 50%, with violations concentrated at $k = 1\text{–}10$). Theorem 1 should therefore be read as governing the interior of the curve; the boundary grid points are handled outside the bootstrap argument entirely — the TPR plateau by the Wilson Rectangle floor and the steep low-FPR corner by the exact Beta order-statistic floor of §2.7, whose guarantee is finite-sample and distribution-free precisely where Theorem 1's asymptotic argument fails.

**Remark (the envelope is a projection, not the studentized tube).** Every retained curve satisfies $Z_b \le c$, so the envelope is *contained in* the rectangular tube $\hat{R}(t) \pm c\,\hat\sigma(t)$, but it need not reach the tube boundary at any given $t$. Measured at $n = 10{,}000$ on high-AUC cases, the envelope arms are $\approx 3.5$ local sd in the interior but the *lower* arm shrinks to $\approx 0.8\text{–}1.4$ sd at $k = 1\text{–}3$ — the envelope inherits the bootstrap's one-sided support limitation at the boundary. Coverage is therefore not a direct inversion of the studentized KS test; the projection step matters in finite samples.

---

### 3.2 Finite-Sample Bias

The empirical ROC curve exhibits upward bias in finite samples:
$$E[\hat{R}(t)] > R_{\text{true}}(t) \quad \text{for } t \in (0,1)$$

This arises from the composition of two empirical distribution functions: $\hat{R}(t) = \hat{G}(\hat{F}^{-1}(1-t))$. The bias is $O(n^{-1})$ at fixed $t \in (0,1)$ and increases with ROC curvature (higher AUC implies larger bias).

**Impact on Coverage.** Since the confidence band is centered on $\hat{R}$, the true ROC tends to fall near or below the lower boundary, reducing population coverage in finite samples. Violating realizations are systematically *optimistic* draws: among lower-bound violations in the simulation suite, the empirical AUC exceeds the true AUC by +0.16 on average at $n = 30$, decaying to +0.003 at $n = 10{,}000$, versus ~+0.001 for covered cases.

**Asymptotic Negligibility — at fixed $t$ only.** At fixed $t$, the bias is $O(n^{-1})$ while the band width is $O(n^{-1/2})$, so the relative contribution of bias vanishes as $n \to \infty$. This argument does **not** extend to the moving boundary grid points $t = k/n_0$ at fixed small $k$: there the bias is order-statistic geometry and remains a roughly constant fraction of the local standard deviation as $n$ grows (measured at $n = 10{,}000$ on high-AUC cases: bias $\approx +0.66$, $+0.46$, $+0.38$ sd at $k = 1, 2, 3$, decaying to $\approx 0$ by FPR $= 0.05$). Combined with the one-sided bootstrap support limitation at the same points, this is the dominant finite-sample failure mode of the method at large $n$.

---

### 3.3 Asymmetry

The envelope is naturally asymmetric. Near boundaries (e.g., $\hat{R}(t) \approx 1$):
- Bootstrap curves can deviate substantially downward
- Bootstrap curves cannot exceed 1
- Retained curves cluster near $\hat{R}$ above, spread out below
- Envelope reflects this: $U(t) - \hat{R}(t) < \hat{R}(t) - L(t)$

No separate machinery needed—asymmetry emerges from the bootstrap distribution.

### 3.4 Heteroscedasticity Adaptation

The studentized KS statistic weights deviations by local standard error. A curve with large deviation where $\hat{\sigma}(t)$ is large may be retained, while the same absolute deviation where $\hat{\sigma}(t)$ is small causes rejection.

This yields tighter envelopes in low-variance regions (near corners) and wider envelopes in high-variance regions (mid-ROC).

### 3.5 Step-Function Structure

The envelope boundaries $L(t)$ and $U(t)$ are step functions with jumps at a subset of $\mathcal{T}$. This matches the step-function nature of $\hat{R}(t)$ and reflects genuine uncertainty about threshold placement.

---

### 3.6 Summary of Guarantees

| Property | Finite Sample | Asymptotic |
|----------|---------------|------------|
| Population coverage | $> 1-\alpha$ at small $n$ (Wilson floor dominates). Without the Beta floor: crosses nominal near $n \approx 300\text{–}500$/class, $< 1-\alpha$ beyond (0.915 at $n=1{,}000$, 0.830 at $n=10{,}000$, prev 50%). With the Beta floor: 0.95–0.99 on the previously failing problem-domain strata at $n = 10^3$–$10^4$ (slightly above nominal; the floor's alpha is added to, not folded into, the band's budget). Residual violations sit at interior grid points $k \sim 50\text{–}500$, outside any tail jurisdiction | $\to 1-\alpha$ on the interior; the moving boundary points are covered by the finite-sample Beta floor rather than the asymptotic argument |
| Future-curve coverage | $\ll 1-\alpha$ (not covered) | $\ll 1-\alpha$ |
| Band width adapts to local variance | ✓ | ✓ |
| Asymmetric at boundaries | ✓ (the bootstrap's lower arm is *too short* at $k = 1\text{–}3$ — see §3.1 remark — which the Beta floor repairs) | ✓ |
| Distribution-free | ✓ | ✓ |
| Informative lower band below FPR $\approx q_1 \approx 7/n_0$ | ✗ (vacuous by construction; no distribution-free bound exists there) | ✗ at the moving points $t = k/n_0$, $k$ fixed |

---

## 4. Computational Considerations

### 4.1 Complexity

| Operation | Cost |
| :--- | :--- |
| Empirical ROC | $O(N \log N)$ |
| Bootstrap ROCs | $O(B \cdot N \log N)$ |
| Grid evaluation | $O(B \cdot G)$ |
| Variance estimation | $O(B \cdot G)$ |
| KS statistics | $O(B \cdot G)$ |
| Sorting $Z_b$ | $O(B \log B)$ |
| Envelope | $O(B \cdot G)$ |

**Total:** $O(B \cdot N \log N + B \cdot G)$

Where $G = |\mathcal{T}|$ is the number of grid points.

### 4.2 MemoryS

Primary storage: $B \times G$ matrix of curve evaluations.

For FP32: Memory $= 4 \cdot B \cdot G$ bytes.

### 4.3 Budget Allocation

We allocate a fixed memory budget $C = B \times K$ to minimize the total error $E = \sqrt{\delta_B^2 + \delta_K^2}$, where:
- $\delta_B = \beta / \sqrt{B}$ (Monte Carlo error)
- $\delta_K = D / K$ (Discretization error, 0 for Full grid)

**Optimization Strategy:**

1.  **Full Grid Analysis:**
    - Set $K = n_0 + 1$.
    - Maximize $B = \lfloor C / (n_0 + 1) \rfloor$.
    - Feasible if $B \ge B_{\min}$. Error is just $\delta_B$.

2.  **Uniform Grid Optimization:**
    - Minimize joint error subject to $B \times K = C$.
    - Optimal allocation:
      $$B_{\text{opt}} = \left(\frac{\beta^2 C^2}{2 D^2}\right)^{1/3}, \quad K_{\text{opt}} = \left(\frac{2 D^2 C}{\beta^2}\right)^{1/3}$$
    - Error involves both $\delta_B$ and $\delta_K$.

**Decision Rule:**
Use the **Full Grid** (exact evaluation) if it provides lower error than the optimized uniform grid. This occurs when the sample size $n_0$ is small relative to the budget:

$$(n_0 + 1)^3 < \frac{27 D^2 C}{4 \beta^2}$$

Otherwise, use the **Uniform Grid** with $B_{\text{opt}}$ and $K_{\text{opt}}$ to balance finite-sample efficiency with grid resolution.

**Parameters:**
- $C$: Memory budget (total float entries).
- $D \approx 2n_0\sqrt{2n_0/(n_1(n_0 + n_1))}$: Discretization sensitivity (or estimated from data).
- $\beta = \sqrt{\alpha(1-\alpha)}/\phi(\Phi^{-1}(1-\alpha))$: Bootstrap error coefficient.

---

## 5. Complete Pseudocode

```
FUNCTION envelope_scb(scores_neg, scores_pos, B, alpha, 
                      grid="full", boundary_method="wilson", 
                      retention_method="ks", use_logit=False):
    
    # ... [Steps 1-4: Empirical ROC, Bootstrap, Grid Evaluation as before] ...
    
    # === Step 5: Variance Estimation ===
    bootstrap_var = var([R_boot_T[b] for b in 1:B], axis=0)
    bootstrap_var_raw = bootstrap_var  # kept un-floored for the Step 7b gate
    
    IF boundary_method == "wilson":
        z = normal_quantile(1 - alpha/2)
        denom = 1 + z^2 / n_pos
        variance_floor = (1/denom^2) * (R_hat_T * (1 - R_hat_T) / n_pos + z^2 / (4 * n_pos^2))
        bootstrap_var = maximum(bootstrap_var, variance_floor)
    ELSE:
        variance_floor = zeros(|T|)
    
    sigma_T = sqrt(bootstrap_var)
    
    # === Step 6: Studentized Statistics ===
    epsilon = min(1 / (n0 + n1), 1e-6)
    
    IF retention_method == "symmetric":
        # Compute signed deviations
        M_up = [], M_down = []
        FOR b = 1 TO B:
            max_pos = -inf, min_neg = inf
            FOR i = 1 TO |T|:
                diff = R_boot_T[b][i] - R_hat_T[i]
                z_val = studentize(diff, sigma_T[i], epsilon)
                max_pos = max(max_pos, z_val)
                min_neg = min(min_neg, z_val)
            M_up.append(max_pos)
            M_down.append(min_neg)
            
        q_up = quantile(M_up, 1 - alpha/2)
        q_down = quantile(M_down, alpha/2)
        retained = [b for b in 1:B if M_down[b] >= q_down AND M_up[b] <= q_up]
        
    ELSE: # "ks"
        Z = []
        FOR b = 1 TO B:
            # calculate max absolute studentized dev
            max_dev = max_over_t(abs(studentize(diff, sigma_T[i], epsilon)))
            Z.append(max_dev)
        
        n_retain = ceil((1 - alpha) * B)
        threshold = sorted(Z)[n_retain - 1]
        retained = [b for b in 1:B if Z[b] <= threshold]
    
    # === Step 7: Envelope ===
    L = [], U = []
    FOR i = 1 TO |T|:
        vals = [R_boot_T[b][i] for b in retained]
        L.append(min(vals))
        U.append(max(vals))
    
    # === Step 7b/7c: Tail floors ===
    L = clip(L, 0, 1), U = clip(U, 0, 1)
    
    IF boundary_method == "wilson":
        deficiency = maximum(0, 1 - bootstrap_var_raw / variance_floor)
        K_eff = sum(deficiency)
        alpha_w = 1 - (1 - alpha)^(1 / K_eff) IF K_eff > 1 ELSE alpha
        L_rect, U_rect = wilson_rectangle_band(scores, T, alpha=alpha_w,
                                               correction="sidak")
        FOR i WHERE deficiency[i] > 0:
            L[i] = min(L[i], L_rect[i])
            U[i] = max(U[i], U_rect[i])
        # Enforce monotonicity
        U = cummax(U)                  # left to right
        L = reverse(cummin(reverse(L)))  # right to left
        
        # === Step 7c: Exact Beta order-statistic floor (lower band) ===
        J = 25
        alpha_e = alpha / (2 * J)
        neg_desc = sort_descending(scores_neg)
        FOR j = 1 TO J:
            q[j] = beta_quantile(1 - alpha_e, j, n0 + 1 - j)
            p_hat = mean(scores_pos > neg_desc[j])
            bound[j] = wilson_lower_one_sided(p_hat, n_pos, alpha_e)
        bound[0] = 0  # vacuous: no order statistic qualifies
        FOR i WHERE 0 < T[i] <= q[J]:
            j_star = max({j : q[j] <= T[i]} OR {0})
            L[i] = min(L[i], bound[j_star])
    ELSE IF boundary_method == "ks":
        L, U = extend_boundary_ks(L, U, n_pos, alpha)
    
    # === Step 8: Boundary anchors ===
    L[0] = 0, U[-1] = 1
        
    RETURN T, L, U
```

---

## 6. Formal Specification

### 6.1 Algorithm 1: Logit-Space Studentized Bootstrap Envelope SCB

**Input:** Scores $\mathcal{S} = \{(y_i, s_i)\}_{i=1}^N$, Replicates $B$, Significance $\alpha$

1.  **Preprocessing:**
    * Compute empirical ROC $\hat{R}(t)$ and grid $\mathcal{T} = \{0, 1/n_0, \dots, 1\}$.
    * Define Haldane transform: $H(p) = \log\left(\frac{p \cdot n_1 + 0.5}{n_1 - p \cdot n_1 + 0.5}\right)$.
    * Transform empirical curve: $\hat{L}(t) \leftarrow H(\hat{R}(t))$ for all $t \in \mathcal{T}$.

2.  **Bootstrap Resampling:**
    * **For** $b = 1$ to $B$:
        * Resample $\mathcal{S}$ to obtain bootstrap ROC $R_b(t)$.
        * Transform bootstrap curve: $L_b(t) \leftarrow H(R_b(t))$ for all $t \in \mathcal{T}$.

3.  **Compute Variance in Logit Space:**
    * **For each** $t \in \mathcal{T}$:
        * $\bar{L}(t) \leftarrow \frac{1}{B} \sum_{b=1}^B L_b(t)$
        * $\hat{\sigma}_L^2(t) \leftarrow \frac{1}{B-1} \sum_{b=1}^B (L_b(t) - \bar{L}(t))^2$

4.  **Studentization & Retention:**
    * Set $\epsilon \leftarrow \min(1/N, 10^{-6})$.
    * **For** $b = 1$ to $B$:
        * Compute deviation vector: $\delta_b(t) \leftarrow L_b(t) - \hat{L}(t)$.
        * **For each** $t \in \mathcal{T}$:
            * **If** $\hat{\sigma}_L(t) \geq \epsilon$: $z_b(t) \leftarrow \delta_b(t) / \hat{\sigma}_L(t)$
            * **Else**: $z_b(t) \leftarrow \mathbb{I}(|\delta_b(t)| \geq \epsilon) \cdot (\delta_b(t) / \epsilon)$
        * $Z_b \leftarrow \max_{t \in \mathcal{T}} |z_b(t)|$.

5.  **Thresholding:**
    * Determine threshold $C_{crit} \leftarrow (1-\alpha)$-quantile of $\{Z_1, \dots, Z_B\}$.
    * Identify retained curves: $\mathcal{R} \leftarrow \{b : Z_b \leq C_{crit}\}$.

6.  **Envelope Construction:**
    * **For each** $t \in \mathcal{T}$:
        * $L_{logit}(t) \leftarrow \min_{b \in \mathcal{R}} L_b(t)$; $U_{logit}(t) \leftarrow \max_{b \in \mathcal{R}} L_b(t)$.
        * Back-transform: $L(t) \leftarrow \sigma(L_{logit}(t))$, $U(t) \leftarrow \sigma(U_{logit}(t))$.

**Return:** Envelope $[L(t), U(t)]$ over $\mathcal{T}$

*Note:* when `boundary_method="wilson"` is combined with the logit path, the Wilson variance floor of Algorithm 2 is applied to $\hat{\sigma}_L^2(t)$ in Step 3 after transformation by the logit Jacobian, $\sigma^2_{floor,L}(t) = \sigma^2_{floor}(t) / \big(p(1-p)\big)^2$ with $p = \hat{R}(t)$ clamped away from $\{0,1\}$, and the tail floors of Algorithm 2 (Steps 5b–5c) are applied to the back-transformed envelope in probability space.

### 6.2 Algorithm 2: Studentized Bootstrap SCB with Wilson Variance Floor

**Input:** Scores $\mathcal{S}$, Replicates $B$, Significance $\alpha$

---

#### 1. Initialization
* Compute empirical ROC $\hat{R}(t)$ and grid $\mathcal{T} = \{0, 1/n_0, \dots, 1\}$.
* Calculate critical value from the standard normal distribution: $z_{\alpha/2} \leftarrow \Phi^{-1}(1-\alpha/2)$.

#### 2. Bootstrap Resampling
* **For** $b = 1$ to $B$:
    * Generate bootstrap ROC $R_b(t)$ on grid $\mathcal{T}$ via resampling from $\mathcal{S}$.

#### 3. Variance Estimation with Floor
* **For each** $t \in \mathcal{T}$:
    * Compute bootstrap variance: $\hat{\sigma}\_{boot}^2(t) \leftarrow \text{Var}(\{R\_b(t)\}\_{b=1}^B)$. Retain the un-floored value for Step 5b.
    * Calculate Wilson variance floor for $p = \hat{R}(t)$:

      $$\sigma_{floor}^2(t) \leftarrow \frac{1}{(1 + z_{\alpha/2}^2/n_1)^2} \left(\frac{p(1-p)}{n_1} + \frac{z_{\alpha/2}^2}{4n_1^2}\right)$$
      
    * Select effective variance for studentization: $\hat{\sigma}^2(t) \leftarrow \max(\hat{\sigma}\_{boot}^2(t), \sigma\_{floor}^2(t))$.

#### 4. Studentization & Retention
* Set $\epsilon \leftarrow \min(1/N, 10^{-6})$.
* **For** $b = 1$ to $B$:
    * Compute the maximum studentized deviation: $Z_b = \sup_{t \in \mathcal{T}} |z_b(t)|$, where $z_b(t)$ is the studentized score using effective variance $\hat{\sigma}(t)$ and $\epsilon$-logic for stability.
* Determine threshold $C_{crit} \leftarrow (1-\alpha)$-quantile of $\{Z_1, \dots, Z_B\}$.
* Identify the retained set of bootstrap curves: $\mathcal{R} \leftarrow \{b : Z_b \leq C_{crit}\}$.

#### 5. Envelope Construction
* **For each** $t \in \mathcal{T}$:
    * Bounds from retained replicates: 
        * $L(t) \leftarrow \min_{b \in \mathcal{R}} R_b(t)$
        * $U(t) \leftarrow \max_{b \in \mathcal{R}} R_b(t)$
* Clip $L(t), U(t)$ to the range $[0, 1]$.

#### 5b. Wilson Rectangle Floor (variance-ratio gated)
* Compute pointwise deficiency: $d(t) \leftarrow \max\left(0,\, 1 - \hat{\sigma}_{boot}^2(t) / \sigma_{floor}^2(t)\right)$ (un-floored bootstrap variance).
* Compute $K_{\text{eff}} \leftarrow \sum_t d(t)$ and $\alpha_w \leftarrow 1 - (1-\alpha)^{1/K_{\text{eff}}}$ (if $K_{\text{eff}} > 1$, else $\alpha_w = \alpha$).
* Compute the Wilson Rectangle band $[L_{\text{rect}}, U_{\text{rect}}]$ at level $\alpha_w$ with Šidák correction across the rectangle margins.
* **For each** $t$ with $d(t) > 0$: $L(t) \leftarrow \min(L(t), L_{\text{rect}}(t))$, $U(t) \leftarrow \max(U(t), U_{\text{rect}}(t))$.
* Enforce monotonicity: $U(t) \leftarrow \max_{s \le t} U(s)$; $L(t) \leftarrow \min_{s \ge t} L(s)$.

#### 5c. Exact Beta Order-Statistic Floor (lower band)
* Set $J \leftarrow 25$ (capped at $n_0$) and $\alpha_e \leftarrow \alpha / (2J)$.
* Sort negative scores descending: $X_{(1)} \ge X_{(2)} \ge \dots$
* **For** $j = 1$ to $J$:
    * Beta upper quantile: $q_j \leftarrow F^{-1}_{\text{Beta}(j,\, n_0+1-j)}(1 - \alpha_e)$.
    * Empirical TPR at the $j$-th largest negative: $\hat{p}_j \leftarrow \frac{1}{n_1} \#\{i : s_i > X_{(j)},\, y_i = 1\}$.
    * One-sided Wilson lower bound: $b_j \leftarrow \text{WilsonLower}_{\alpha_e}(\hat{p}_j, n_1)$; set $b_0 \leftarrow 0$.
* **For each** $t \in \mathcal{T}$ with $0 < t \le q_J$:
    * $j^\*(t) \leftarrow \max\big(\{j : q_j \le t\} \cup \{0\}\big)$
    * $L(t) \leftarrow \min\big(L(t),\, b_{j^\*(t)}\big)$

#### 6. Boundary Handling
* Enforce fixed anchor points: $L(0)=0, U(1)=1$.

**Return:** Envelope $[L(t), U(t)]$ over $\mathcal{T}$