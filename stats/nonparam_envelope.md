# Studentized Bootstrap Envelope Simultaneous Confidence Bands for ROC Curves

## Abstract

We present a nonparametric method for constructing simultaneous confidence bands (SCB) for the **true population** ROC curve using a studentized bootstrap envelope. The method retains the $(1-\alpha)$ fraction of bootstrap curves most consistent with the empirical ROC (using either studentized Kolmogorov-Smirnov statistics or symmetric tail trimming) and returns their pointwise envelope. The resulting band is asymmetric, adapts to local variance (incorporating a variance floor for stability), and inherits the step-function structure of the empirical ROC.

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

For each $t \in T$: 
$$\hat{\sigma}_{\text{boot}}^2(t) = \frac{1}{B-1} \sum_{b=1}^{B} \left(R_b(t) - \bar{R}(t)\right)^2$$

where $\bar{R}(t) = \frac{1}{B}\sum_b R_b(t)$.

**Variance Floor:**
At boundaries (FPR near 0 or 1), the bootstrap distribution often collapses to a single value, yielding $\hat{\sigma}_{boot}(t) \approx 0$. This causes instability in studentization. We support multiple methods for imposing a minimum variance floor:

**Option 1: Wilson Score Variance Floor (`boundary_method="wilson"`)**

Based on the Wilson score interval for a binomial proportion $p = \hat{R}(t)$ with sample size $n_1$:

$$\sigma^2_{wilson}(p) = \frac{1}{(1 + z^2/n_1)^2} \cdot \left(\frac{p(1-p)}{n_1} + \frac{z^2}{4n_1^2}\right)$$

where $z = \Phi^{-1}(1-\alpha/2)$ is the normal quantile.

**Option 2: Hsieh-Turnbull Asymptotic Variance (`boundary_method="reflected_kde"` or `"log_concave"`)**

Uses the asymptotic variance formula from Hsieh & Turnbull (1996):

$$\text{Var}(R(t)) = \frac{R(t)(1-R(t))}{n_1} + \left(\frac{g(c_t)}{f(c_t)}\right)^2 \cdot \frac{t(1-t)}{n_0}$$

where $c_t$ is the threshold corresponding to FPR $= t$, and $f$, $g$ are the score densities for negatives and positives respectively. Densities are estimated via:
- `reflected_kde`: Reflected kernel density estimation with ISJ bandwidth selection
- `log_concave`: Log-concave MLE via convex optimization

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
    $$z_b(t) = \begin{cases} 
    0 & \text{if } |\delta_b(t)| < \epsilon \text{ (noise)} \\ 
    \frac{\delta_b(t)}{\epsilon} & \text{if } |\delta_b(t)| \geq \epsilon \text{ (significant shift)} 
    \end{cases}$$

For each curve, the global statistic is $Z_b = \sup_{t \in \mathcal{T}} |z_b(t)|$.


### 2.6 Curve Retention

We support two methods for determining which curves to retain.

**Option A: Original KS Retention (`retention_method="ks"`)**
Retain the $(1-\alpha)$ fraction of curves with the smallest maximum absolute studentized deviation $Z_b$.
$$\mathcal{R}_\alpha = \left\{ R_b : Z_b \leq Z_{(\lceil (1-\alpha)B \rceil)} \right\}$$
where $Z_{(k)}$ is the $k$-th order statistic. This creates a band of "most typical" curves in terms of global shape deviation.

**Option B: Symmetric Retention (`retention_method="symmetric"`)**
The standard KS method can be asymmetric at boundaries (e.g., at high AUC, curves can't deviate upward past 1, but can deviate downward). To ensure balanced tail coverage:
1. Compute *signed* studentized deviations: $s_b(t)$ using the signed difference $(R_b(t) - \hat{R}(t))$ in the numerator.
2. For each curve $b$, find its max upward deviation $M^+_b = \sup_t s_b(t)$ and max downward deviation $M^-_b = \inf_t s_b(t)$.
3. Determine thresholds $q_{up}$ and $q_{down}$ as the $(1-\alpha/2)$ and $(\alpha/2)$ quantiles of $M^+$ and $M^-$ respectively.
4. Retain curves that satisfy:
   $$M^{-}_{b} \geq q_{\text{down}} \quad \text{AND} \quad M^{+}_{b} \leq q_{\text{up}}$$

This method explicitly trims the most extreme $\alpha/2$ upward excursions and $\alpha/2$ downward excursions.

### 2.7 Envelope Construction

Compute the pointwise min and max of the retained curves $\mathcal{R}_\alpha$:
$$L(t) = \min_{R_b \in \mathcal{R}_\alpha} R_b(t)$$
$$U(t) = \max_{R_b \in \mathcal{R}_\alpha} R_b(t)$$

**Envelope Width Extension:**
When using a variance floor (i.e., `boundary_method` is not `"none"` or `"ks"`), the envelope is extended to ensure minimum width based on the variance floor:
$$U(t) \leftarrow \max\left(U(t), \hat{R}(t) + \sigma_{floor}(t)\right)$$
$$L(t) \leftarrow \min\left(L(t), \hat{R}(t) - \sigma_{floor}(t)\right)$$

This guarantees the band is at least as wide as the theoretical minimum uncertainty even when retained bootstrap curves happen to cluster tightly.

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

**Note:** When using logit space, boundary methods `"wilson"`, `"reflected_kde"`, and `"log_concave"` are not applied (variance floors are computed in probability space). Use `boundary_method="none"` or `"ks"` with the logit path.

---

## 3. Properties

### 3.1 Coverage Guarantees

**Definition (Population Coverage).** The probability that the true population ROC curve falls entirely within the band:
$$P\left(\forall t: R_{true}(t) \in [L(t), U(t)]\right)$$

**Note on Future Samples:** This method constructs a *confidence band* for the underlying population curve $R_{true}$. It is **not** a *prediction band* for future empirical ROC curves $\hat{R}_{new}$. A future empirical curve has additional sampling variability relative to the current empirical curve (variance approximately doubles), so the coverage of future samples will be significantly lower than $(1-\alpha)$.


---

**Theorem 1 (Asymptotic Population Coverage).** Under A1–A4, as $n = \min(n_0, n_1) \to \infty$:
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

**Impact on Coverage.** Since the confidence band is centered on $\hat{R}$, the true ROC tends to fall near or below the lower boundary, reducing population coverage in finite samples. This effect is most pronounced at high AUC and small sample sizes. 

**Asymptotic Negligibility.** Because the bias is $O(n^{-1})$ while the band width is $O(n^{-1/2})$, the relative contribution of bias vanishes as $n \to \infty$, ensuring the asymptotic coverage guarantee of Theorem 1.

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
| Population coverage | $< 1-\alpha$ (biased low) | $\to 1-\alpha$ |
| Future-curve coverage | $\ll 1-\alpha$ (not covered) | $\ll 1-\alpha$ |
| Band width adapts to local variance | ✓ | ✓ |
| Asymmetric at boundaries | ✓ | ✓ |
| Distribution-free | ✓ | ✓ |

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
    
    IF boundary_method == "wilson":
        z = normal_quantile(1 - alpha/2)
        denom = 1 + z^2 / n_pos
        variance_floor = (1/denom^2) * (R_hat_T * (1 - R_hat_T) / n_pos + z^2 / (4 * n_pos^2))
    ELSE IF boundary_method in ("reflected_kde", "log_concave"):
        variance_floor = hsieh_turnbull_variance(scores_neg, scores_pos, T, method=boundary_method)
    ELSE:
        variance_floor = zeros(|T|)
    
    IF boundary_method NOT IN ("none", "ks"):
        bootstrap_var = maximum(bootstrap_var, variance_floor)
    
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
    
    # === Step 7b: Envelope Width Extension ===
    IF boundary_method NOT IN ("none", "ks"):
        sigma_floor = sqrt(variance_floor)
        U = maximum(U, R_hat_T + sigma_floor)
        L = minimum(L, R_hat_T - sigma_floor)
    
    # === Step 8: Boundaries ===
    L = clip(L, 0, 1), U = clip(U, 0, 1)
    L[0] = 0, U[-1] = 1
    
    IF boundary_method == "ks":
        L, U = extend_boundary_ks(L, U, n_pos, alpha)
        
    RETURN T, L, U
```

---

## 6. Formal Specification

### 6.1 Logit-space Studentized Bootstrap Envelope SCB

\begin{algorithm}
\caption{Logit-Space Studentized Bootstrap Envelope SCB}
\begin{algorithmic}[1]
\REQUIRE Scores $\mathcal{S} = \{(y_i, s_i)\}_{i=1}^N$, Replicates $B$, Significance $\alpha$
\STATE Compute empirical ROC $\hat{R}(t)$ and grid $\mathcal{T} = \{0, 1/n_0, \dots, 1\}$
\STATE Define Haldane transform: $H(p) = \log\left(\frac{p \cdot n_1 + 0.5}{n_1 - p \cdot n_1 + 0.5}\right)$
\STATE Transform empirical curve: $\hat{L}(t) \leftarrow H(\hat{R}(t))$ for all $t \in \mathcal{T}$

\FOR{$b = 1$ to $B$}
    \STATE Resample $\mathcal{S}$ to obtain bootstrap ROC $R_b(t)$
    \STATE Transform bootstrap curve: $L_b(t) \leftarrow H(R_b(t))$ for all $t \in \mathcal{T}$
\ENDFOR

\STATE \textbf{Compute Variance in Logit Space:}
\FOR{each $t \in \mathcal{T}$}
    \STATE $\bar{L}(t) \leftarrow \frac{1}{B} \sum_{b=1}^B L_b(t)$
    \STATE $\hat{\sigma}_L^2(t) \leftarrow \frac{1}{B-1} \sum_{b=1}^B (L_b(t) - \bar{L}(t))^2$
\ENDFOR

\STATE \textbf{Studentization \& Retention:}
\STATE Set $\epsilon \leftarrow \min(1/N, 10^{-6})$
\FOR{$b = 1$ to $B$}
    \STATE Compute deviation vector: $\delta_b(t) \leftarrow L_b(t) - \hat{L}(t)$
    \STATE Compute studentized statistic (handling low variance):
    \FOR{each $t \in \mathcal{T}$}
        \IF{$\hat{\sigma}_L(t) \geq \epsilon$}
            \STATE $z_b(t) \leftarrow \delta_b(t) / \hat{\sigma}_L(t)$
        \ELSE
            \STATE $z_b(t) \leftarrow \mathbb{I}(|\delta_b(t)| \geq \epsilon) \cdot (\delta_b(t) / \epsilon)$
        \ENDIF
    \ENDFOR
    \STATE $Z_b \leftarrow \max_{t \in \mathcal{T}} |z_b(t)|$
\ENDFOR

\STATE Determine threshold $C_{crit} \leftarrow$ $(1-\alpha)$-quantile of $\{Z_1, \dots, Z_B\}$
\STATE Identify retained curves: $\mathcal{R} \leftarrow \{b : Z_b \leq C_{crit}\}$

\STATE \textbf{Envelope Construction:}
\FOR{each $t \in \mathcal{T}$}
    \STATE $L_{logit}(t) \leftarrow \min_{b \in \mathcal{R}} L_b(t)$
    \STATE $U_{logit}(t) \leftarrow \max_{b \in \mathcal{R}} L_b(t)$
    \STATE \COMMENT{Back-transform to probability space}
    \STATE $L(t) \leftarrow \sigma(L_{logit}(t))$, \quad $U(t) \leftarrow \sigma(U_{logit}(t))$
\ENDFOR

\STATE \textbf{return} Envelope $[L(t), U(t)]$ over $\mathcal{T}$
\end{algorithmic}
\end{algorithm}

### 6.2 Studentized Bootstrap SCB with Wilson Variance Floor

\begin{algorithm}
\caption{Studentized Bootstrap SCB with Wilson Variance Floor}
\begin{algorithmic}[1]
\REQUIRE Scores $\mathcal{S}$, Replicates $B$, Significance $\alpha$
\STATE Compute empirical ROC $\hat{R}(t)$ and grid $\mathcal{T} = \{0, 1/n_0, \dots, 1\}$
\STATE $z_{\alpha/2} \leftarrow \Phi^{-1}(1-\alpha/2)$

\FOR{$b = 1$ to $B$}
    \STATE Generate bootstrap ROC $R_b(t)$ on grid $\mathcal{T}$
\ENDFOR

\STATE \textbf{Variance Estimation with Floor:}
\FOR{each $t \in \mathcal{T}$}
    \STATE $\hat{\sigma}_{boot}^2(t) \leftarrow \text{Var}(\{R_b(t)\}_{b=1}^B)$
    \STATE \COMMENT{Calculate Wilson variance floor for $p = \hat{R}(t)$}
    \STATE $\sigma_{floor}^2(t) \leftarrow \frac{1}{(1 + z_{\alpha/2}^2/n_1)^2} \left(\frac{p(1-p)}{n_1} + \frac{z_{\alpha/2}^2}{4n_1^2}\right)$
    \STATE \COMMENT{Select max variance and convert to std dev}
    \STATE $\hat{\sigma}^2(t) \leftarrow \max(\hat{\sigma}_{boot}^2(t), \sigma_{floor}^2(t))$
    \STATE $\sigma_{floor}(t) \leftarrow \sqrt{\sigma_{floor}^2(t)}$
\ENDFOR

\STATE \textbf{Studentization:}
\STATE Set $\epsilon \leftarrow \min(1/N, 10^{-6})$
\FOR{$b = 1$ to $B$}
    \STATE Calculate $Z_b = \sup_{t \in \mathcal{T}} |z_b(t)|$ using effective variance $\hat{\sigma}(t)$ and $\epsilon$-logic
\ENDFOR

\STATE Threshold $C_{crit} \leftarrow$ $(1-\alpha)$-quantile of $\{Z_b\}$; Retain set $\mathcal{R}$

\STATE \textbf{Envelope with Width Extension:}
\FOR{each $t \in \mathcal{T}$}
    \STATE $L(t) \leftarrow \min_{b \in \mathcal{R}} R_b(t)$
    \STATE $U(t) \leftarrow \max_{b \in \mathcal{R}} R_b(t)$
    
    \STATE \COMMENT{Ensure band width respects theoretical minimum uncertainty}
    \STATE $L(t) \leftarrow \min(L(t), \hat{R}(t) - \sigma_{floor}(t))$
    \STATE $U(t) \leftarrow \max(U(t), \hat{R}(t) + \sigma_{floor}(t))$
\ENDFOR

\STATE \textbf{Boundary Handling:}
\STATE Clip $L(t), U(t)$ to $[0, 1]$; Enforce $L(0)=0, U(1)=1$
\STATE \textbf{return} Envelope $[L(t), U(t)]$
\end{algorithmic}
\end{algorithm}
