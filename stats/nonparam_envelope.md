# Studentized Bootstrap Envelope Simultaneous Confidence Bands for ROC Curves

## Abstract

We present a nonparametric method for constructing simultaneous confidence bands (SCB) for ROC curves using a studentized bootstrap envelope. The method retains the $(1-\alpha)$ fraction of bootstrap curves most consistent with the empirical ROC (using either studentized Kolmogorov-Smirnov statistics or symmetric tail trimming) and returns their pointwise envelope. The resulting band is asymmetric, adapts to local variance (incorporating a Wilson-score floor for stability), and inherits the step-function structure of the empirical ROC.

---

## 1. Setup and Assumptions

**Data:** $\mathcal{D} = \{(y_i, s_i)\}_{i=1}^N$ where $y_i \in \{0,1\}$ is the class label and $s_i \in \mathbb{R}$ is the score. Let $n_0$ and $n_1$ denote the number of negatives and positives.

**Assumptions:**
- A1: Independent sampling within each class
- A2: Continuous score distributions (no ties)
- A3: Higher scores indicate positive class
- A4: Finite variance of TPR at any fixed FPR

**Target:** Construct $\mathcal{B}_\alpha(t) = [L(t), U(t)]$ such that:
$$P\left(\forall t \in [0,1]: R_{true}(t) \in \mathcal{B}_\alpha(t)\right) \geq 1 - \alpha$$

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
$$\hat{\sigma}_{boot}(t) = \sqrt{\frac{1}{B-1} \sum_{b=1}^{B}\left(R_b(t) - \bar{R}(t)\right)^2}$$

where $\bar{R}(t) = \frac{1}{B}\sum_b R_b(t)$.

**Wilson Score Variance Floor:**
At boundaries (FPR near 0 or 1), the bootstrap distribution often collapses to a single value, yielding $\hat{\sigma}_{boot}(t) \approx 0$. This causes instability in studentization. To address this, we impose a minimum variance floor based on the Wilson score interval for a binomial proportion $p = \hat{R}(t)$ (the empirical TPR) with sample size $n_{pos}$:

$$\sigma^2_{wilson}(p) = \left( \frac{z \sqrt{p(1-p)/n_{pos} + z^2/(4n_{pos}^2)}}{1 + z^2/n_{pos}} \right)^2$$

where $z$ is the normal quantile for $(1-\alpha/2)$. The final standard deviation used for studentization is:
$$\hat{\sigma}(t) = \sqrt{\max\left(\hat{\sigma}^2_{boot}(t), \sigma^2_{wilson}(\hat{R}(t))\right)}$$

### 2.5 Studentized KS Statistics

To measure the "strangeness" of each bootstrap curve, we compute its maximum studentized deviation from the empirical curve.

**Epsilon Regularizer:**
We define a regularization parameter $\epsilon = \min(1/N, 10^{-6})$, where $N = n_0 + n_1$. This serves as a lower bound on meaningful deviations, ensuring we do not amplify numerical noise or irrelevant micro-fluctuations when variance is extremely low. It is derived from the smallest possible probability mass in the sample space.

**Studentization with Low-Variance Handling:**
For each bootstrap curve $b$, we compute the pointwise studentized deviation $z_b(t)$.
1. Calculate absolute deviation: $\delta_b(t) = |R_b(t) - \hat{R}(t)|$.
2. **Normal Case:** If $\hat{\sigma}(t) \geq \epsilon$:
   $$z_b(t) = \frac{\delta_b(t)}{\hat{\sigma}(t)}$$
3. **Low-Variance Case:** If $\hat{\sigma}(t) < \epsilon$ (variance is effectively zero):
   $$z_b(t) = \begin{cases} 
   0 & \text{if } \delta_b(t) < \epsilon \text{ (noise)} \\
   \frac{\delta_b(t)}{\epsilon} & \text{if } \delta_b(t) \geq \epsilon \text{ (significant shift)}
   \end{cases}$$

For each curve, the global statistic is $Z_b = \sup_{t \in \mathcal{T}} z_b(t)$.


### 2.6 Curve Retention

We support two methods for determining which curves to retain.

**Option A: Original KS Retention (`retention_method="ks"`)**
Retain the $(1-\alpha)$ fraction of curves with the smallest maximum absolute studentized deviation $Z_b$.
$$\mathcal{R}_\alpha = \left\{R_b : Z_b \leq Z_{(\lceil(1-\alpha)B\rceil)}\right\}$$
where $Z_{(k)}$ is the $k$-th order statistic. This creates a band of "most typical" curves in terms of global shape deviation.

**Option B: Symmetric Retention (`retention_method="symmetric"`)**
The standard KS method can be asymmetric at boundaries (e.g., at high AUC, curves can't deviate upward past 1, but can deviate downward). To ensure balanced tail coverage:
1. Compute *signed* studentized deviations: $s_b(t)$ using the signed difference $(R_b(t) - \hat{R}(t))$ in the numerator.
2. For each curve $b$, find its max upward deviation $M^+_b = \sup_t s_b(t)$ and max downward deviation $M^-_b = \inf_t s_b(t)$.
3. Determine thresholds $q_{up}$ and $q_{down}$ such that $\alpha/2$ of curves exceed $q_{up}$ and $\alpha/2$ fall below $q_{down}$.
4. Retain curves that satisfy:
   $$M^-_b \geq q_{down} \quad \text{AND} \quad M^+_b \leq q_{up}$$

This method explicitly trims the most extreme $\alpha/2$ upward excursions and $\alpha/2$ downward excursions.

### 2.7 Envelope Construction & Boundary Handling

Compute the pointwise min and max of the retained curves $\mathcal{R}_\alpha$:
$$L(t) = \min_{R_b \in \mathcal{R}_\alpha} R_b(t)$$
$$U(t) = \max_{R_b \in \mathcal{R}_\alpha} R_b(t)$$

**Boundary Enforcement:**
We explicitly enforce that the confidence band respects logical ROC constraints:
- $L(0) = 0$
- $U(1) = 1$
- Clip all values to $[0, 1]$.

**KS-Style Boundary Extension (Optional, `boundary_method="ks"`):**
In regions where bootstrap variance collapses completely (near corners), we can optionally extend the band using fixed width margins derived from the analytical Kolmogorov-Smirnov distribution (Campbell 1994). This connects the computed bootstrap envelope to the corners (0,0) and (1,1) with a theoretical worst-case slope.

---

## 3. Properties

### 3.1 Coverage Guarantees

We distinguish two coverage targets with different finite-sample behavior.

**Definition (Future-Curve Coverage).** The probability that an independent empirical ROC curve, computed from a new sample of the same size $(n_0, n_1)$ from the same population, falls entirely within the band:
$$P\left(\forall t: \hat{R}_{new}(t) \in [L(t), U(t)]\right)$$

**Definition (Population Coverage).** The probability that the true population ROC curve falls entirely within the band:
$$P\left(\forall t: R_{true}(t) \in [L(t), U(t)]\right)$$

---

**Theorem 1 (Future-Curve Coverage).** Under A1–A4, the envelope band achieves:
$$P\left(\forall t: \hat{R}_{new}(t) \in [L(t), U(t)]\right) = 1 - \alpha + O(n^{-1/2})$$

where $n = \min(n_0, n_1)$.

*Proof sketch:* The bootstrap directly estimates the sampling distribution of empirical ROC curves. By construction, the envelope contains the $(1-\alpha)$ fraction of bootstrap curves closest to $\hat{R}$ under the studentized supremum metric. The bootstrap distribution of $R_b$ around $\hat{R}$ consistently estimates the sampling distribution of $\hat{R}_{new}$ around its expectation. Since both $\hat{R}$ and $\hat{R}_{new}$ share the same bias structure (both are empirical ROCs at the same sample size), the band correctly captures the variability of future empirical curves. The $O(n^{-1/2})$ error arises from bootstrap approximation error. $\square$

---

**Theorem 2 (Asymptotic Population Coverage).** Under A1–A4, as $n = \min(n_0, n_1) \to \infty$:
$$P\left(\forall t: R_{true}(t) \in [L(t), U(t)]\right) \to 1 - \alpha$$

*Proof sketch:* 

1. The empirical ROC process $\sqrt{n}(\hat{R} - R_{true})$ converges weakly to a Gaussian process $\mathbb{G}$ (Hsieh & Turnbull, 1996).

2. By bootstrap consistency for the empirical process, the conditional distribution of $\sqrt{n}(R_b - \hat{R})$ given the data converges to the same limit $\mathbb{G}$.

3. The finite-sample bias $E[\hat{R}(t)] - R_{true}(t) = O(n^{-1})$ vanishes faster than the $O(n^{-1/2})$ standard deviation, so it becomes negligible in the standardized process.

4. The retention rule keeps curves whose studentized supremum deviation is below the $(1-\alpha)$ quantile. By the bootstrap principle and (3), $R_{true}$ falls within the envelope with probability approaching $1-\alpha$. $\square$

---

### 3.2 Finite-Sample Bias

The empirical ROC curve exhibits upward bias in finite samples:
$$E[\hat{R}(t)] > R_{true}(t) \quad \text{for } t \in (0,1)$$

This arises from the composition of two empirical distribution functions: $\hat{R}(t) = \hat{G}(\hat{F}^{-1}(1-t))$. The bias is $O(n^{-1})$ and increases with ROC curvature (higher AUC implies larger bias).

**Impact on Coverage.** Since the confidence band is centered on $\hat{R}$, the true ROC tends to fall near or below the lower boundary, reducing population coverage in finite samples. This effect is most pronounced at high AUC and small sample sizes. Future-curve coverage is unaffected because both the band and future empirical curves share the same bias structure.

**Asymptotic Negligibility.** Because the bias is $O(n^{-1})$ while the band width is $O(n^{-1/2})$, the relative contribution of bias vanishes as $n \to \infty$, ensuring the asymptotic coverage guarantee of Theorem 2.

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
| Future-curve coverage | $\approx 1-\alpha$ | $= 1-\alpha$ |
| Population coverage | $< 1-\alpha$ (biased low) | $\to 1-\alpha$ |
| Band width adapts to local variance | ✓ | ✓ |
| Asymmetric at boundaries | ✓ | ✓ |
| Distribution-free | ✓ | ✓ |

---

## 4. Computational Considerations

### 4.1 Complexity

| Operation | Cost |
|-----------|------|
| Empirical ROC | $O(N \log N)$ |
| Bootstrap ROCs | $O(B \cdot N \log N)$ |
| Grid evaluation | $O(B \cdot |\mathcal{T}|)$ |
| Variance estimation | $O(B \cdot |\mathcal{T}|)$ |
| KS statistics | $O(B \cdot |\mathcal{T}|)$ |
| Sorting $Z_b$ | $O(B \log B)$ |
| Envelope | $O((1-\alpha)B \cdot |\mathcal{T}|)$ |

**Total:** $O(B \cdot N \log N + B \cdot |\mathcal{T}|)$

### 4.2 Memory

Primary storage: $B \times |\mathcal{T}|$ matrix of curve evaluations.

For FP32: Memory $= 4 \cdot B \cdot |\mathcal{T}|$ bytes.

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
                      retention_method="ks"):
    
    # ... [Steps 1-4: Empirical ROC, Bootstrap, Grid Evaluation as before] ...
    
    # === Step 5: Variance Estimation ===
    sigma_T = []
    FOR i = 1 TO |T|:
        sd_boot = std([R_boot_T[b][i] for b in 1:B])
        IF boundary_method == "wilson":
            p = R_hat_T[i]
            sd_wilson = wilson_score_sd(p, n_pos, alpha)
            sigma_T.append(max(sd_boot, sd_wilson))
        ELSE:
            sigma_T.append(sd_boot)
    
    # === Step 6: Studentized Statistics ===
    epsilon = min(1 / (n0 + n1), 1e-6)
    
    IF retention_method == "symmetric":
        # Compute signed deviations
        M_up = [], M_down = []
        FOR b = 1 TO B:
            max_pos = -inf, min_neg = inf
            FOR i = 1 TO |T|:
                diff = R_boot_T[b][i] - R_hat_T[i]
                # Apply epsilon/sigma logic to diff...
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
        
        threshold = quantile(Z, 1 - alpha)
        retained = [b for b in 1:B if Z[b] <= threshold]
    
    # === Step 7: Envelope ===
    L = [], U = []
    FOR i = 1 TO |T|:
        vals = [R_boot_T[b][i] for b in retained]
        L.append(min(vals))
        U.append(max(vals))
    
    # === Step 8: Boundaries ===
    L = clip(L, 0, 1), U = clip(U, 0, 1)
    L[0] = 0, U[-1] = 1
    
    IF boundary_method == "ks":
        L, U = extend_boundary_ks(L, U, n_pos, alpha)
        
    RETURN T, L, U
```

