# Studentized Bootstrap Envelope Simultaneous Confidence Bands for ROC Curves

## Abstract

We present a nonparametric method for constructing simultaneous confidence bands (SCB) for ROC curves using a studentized bootstrap envelope. The method retains the $(1-\alpha)$ fraction of bootstrap curves most consistent with the empirical ROC (ranked by studentized Kolmogorov-Smirnov statistic) and returns their pointwise envelope. The resulting band is asymmetric, adapts to local variance, and inherits the step-function structure of the empirical ROC.

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

| Method | Grid $\mathcal{T}$ | Properties |
|--------|-------------------|------------|
| Exact | $\mathcal{J}_0 \cup \bigcup_b \mathcal{J}_b$ | Exact supremum; $O(B \cdot n_0)$ points |
| Hybrid | $\mathcal{J}_0 \cup \text{linspace}(0, 1, K)$ | Exact at $\hat{R}$ jumps; $O(n_0 + K)$ points |
| Uniform | $\text{linspace}(0, 1, K)$ | Fast; $O(D/K)$ discretization error |

### 2.3 Stratified Bootstrap

For $b = 1, \ldots, B$:
1. Resample $n_0$ negatives with replacement
2. Resample $n_1$ positives with replacement
3. Compute bootstrap ROC $R_b(t)$
4. Evaluate $R_b$ on grid $\mathcal{T}$

### 2.4 Variance Estimation

For each $t \in \mathcal{T}$:
$$\hat{\sigma}(t) = \sqrt{\frac{1}{B-1} \sum_{b=1}^{B}\left(R_b(t) - \bar{R}(t)\right)^2}$$

where $\bar{R}(t) = \frac{1}{B}\sum_b R_b(t)$.

### 2.5 Studentized KS Statistics

Set regularization $\epsilon = \min(1/n, 10^{-6})$ where $n = n_0 + n_1$.

For each bootstrap curve, compute:
$$Z_b = \sup_{t \in \mathcal{T}} z(t)$$

where the pointwise studentized deviation handles corners explicitly:

$$z(t) = \begin{cases} 
0 & \text{if } \hat{\sigma}(t) < \epsilon \text{ and } |R_b(t) - \hat{R}(t)| < \epsilon \\
\frac{|R_b(t) - \hat{R}(t)|}{\epsilon} & \text{if } \hat{\sigma}(t) < \epsilon \text{ and } |R_b(t) - \hat{R}(t)| \geq \epsilon \\
\frac{|R_b(t) - \hat{R}(t)|}{\hat{\sigma}(t)} & \text{otherwise}
\end{cases}$$

This measures the worst-case standardized deviation from the empirical ROC, with stable behavior at the corners where $\hat{\sigma}(t) \to 0$.

### 2.6 Curve Retention

Retain the $(1-\alpha)B$ curves with smallest $Z_b$:
$$\mathcal{R}_\alpha = \left\{R_b : Z_b \leq Z_{(\lceil(1-\alpha)B\rceil)}\right\}$$

where $Z_{(k)}$ is the $k$-th order statistic.

### 2.7 Envelope Construction

$$L(t) = \min_{R_b \in \mathcal{R}_\alpha} R_b(t)$$
$$U(t) = \max_{R_b \in \mathcal{R}_\alpha} R_b(t)$$

Clip to $[0,1]$: $L(t) \leftarrow \max(0, L(t))$, $U(t) \leftarrow \min(1, U(t))$.

---

## 3. Properties

### 3.1 Asymptotic Coverage

**Theorem.** Under A1–A4, as $\min(n_0, n_1) \to \infty$:
$$P\left(\forall t: R_{true}(t) \in [L(t), U(t)]\right) \to 1 - \alpha$$

**Proof sketch:**

1. The empirical ROC process $\sqrt{n}(\hat{R} - R_{true})$ converges weakly to a Gaussian process $\mathbb{G}$.

2. By bootstrap consistency, the distribution of $R_b$ around $\hat{R}$ approximates the distribution of $\hat{R}$ around $R_{true}$.

3. The retention rule keeps curves whose studentized supremum deviation is below the $(1-\alpha)$ quantile. By the bootstrap principle, $R_{true}$ falls within the envelope of such curves with probability approaching $1-\alpha$.

4. The studentization ensures the retention criterion adapts to local variance, providing uniform (not just pointwise) coverage. $\square$

### 3.2 Asymmetry

The envelope is naturally asymmetric. Near boundaries (e.g., $\hat{R}(t) \approx 1$):
- Bootstrap curves can deviate substantially downward
- Bootstrap curves cannot exceed 1
- Retained curves cluster near $\hat{R}$ above, spread out below
- Envelope reflects this: $U(t) - \hat{R}(t) < \hat{R}(t) - L(t)$

No separate machinery needed—asymmetry emerges from the bootstrap distribution.

### 3.3 Heteroscedasticity Adaptation

The studentized KS statistic weights deviations by local standard error. A curve with large deviation where $\hat{\sigma}(t)$ is large may be retained, while the same absolute deviation where $\hat{\sigma}(t)$ is small causes rejection.

This yields tighter envelopes in low-variance regions (near corners) and wider envelopes in high-variance regions (mid-ROC).

### 3.4 Step-Function Structure

The envelope boundaries $L(t)$ and $U(t)$ are step functions with jumps at a subset of $\mathcal{T}$. This matches the step-function nature of $\hat{R}(t)$ and reflects genuine uncertainty about threshold placement.

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

The optimal $B$ vs $K$ tradeoff follows the same analysis as the parametric band method. The error decomposition:

$$\delta_K = \frac{D}{K} \quad \text{(discretization)}, \quad \delta_B = \frac{\beta}{\sqrt{B}} \quad \text{(Monte Carlo)}$$

yields optimal allocation:

$$B_{opt} = \left(\frac{\beta^2}{2}\right)^{1/3} \cdot \frac{C^{2/3}}{D^{2/3}}, \quad K_{opt} = \left(\frac{2}{\beta^2}\right)^{1/3} \cdot C^{1/3} \cdot D^{2/3}$$

where:
- $C = B \times K$ is the computational budget
- $D = 2n_0\sqrt{2n_0/(n_1(n_0 + n_1))}$ is the discretization sensitivity
- $\beta = \sqrt{\alpha(1-\alpha)}/f(c_\alpha)$ is the bootstrap error coefficient

**Change from parametric method:** Discretization error now affects both **curve ranking** (which curves are retained) and **envelope resolution** (where jumps can occur). For the exact grid, the envelope representation is exact. For uniform grids, ensure $K \geq 2n_0$ so envelope resolution matches empirical ROC resolution.

---

## 5. Complete Pseudocode

```
FUNCTION envelope_scb(scores_neg, scores_pos, B, alpha, grid="hybrid", K=1001):
    
    n0 = length(scores_neg)
    n1 = length(scores_pos)
    
    # === Step 1: Empirical ROC ===
    R_hat, J_hat = compute_roc(scores_neg, scores_pos)
    
    # === Step 2: Bootstrap ===
    R_boot = []          # list of B bootstrap curves
    J_all = set(J_hat)   # collect jump points if exact grid
    
    FOR b = 1 TO B:
        neg_b = resample_with_replacement(scores_neg, n0)
        pos_b = resample_with_replacement(scores_pos, n1)
        R_b, J_b = compute_roc(neg_b, pos_b)
        R_boot.append(R_b)
        IF grid == "exact":
            J_all = J_all ∪ J_b
    
    # === Step 3: Evaluation grid ===
    IF grid == "exact":
        T = sorted(J_all)
    ELIF grid == "hybrid":
        T = sorted(J_hat ∪ linspace(0, 1, K))
    ELSE:
        T = linspace(0, 1, K)
    
    # === Step 4: Evaluate curves on grid ===
    R_hat_T = [R_hat(t) for t in T]
    R_boot_T = [[R_b(t) for t in T] for R_b in R_boot]
    
    # === Step 5: Variance estimation ===
    sigma_T = []
    FOR i = 1 TO |T|:
        values = [R_boot_T[b][i] for b in 1:B]
        sigma_T.append(std(values))
    
    # === Step 6: Studentized KS statistics ===
    epsilon = min(1 / (n0 + n1), 1e-6)
    Z = []
    FOR b = 1 TO B:
        max_dev = 0
        FOR i = 1 TO |T|:
            diff = |R_boot_T[b][i] - R_hat_T[i]|
            IF sigma_T[i] < epsilon:
                IF diff < epsilon:
                    z_val = 0
                ELSE:
                    z_val = diff / epsilon
            ELSE:
                z_val = diff / sigma_T[i]
            max_dev = max(max_dev, z_val)
        Z.append(max_dev)
    
    # === Step 7: Retain lowest-Z curves ===
    n_retain = floor((1 - alpha) * B)
    threshold = sorted(Z)[n_retain]
    retained = [b for b in 1:B if Z[b] <= threshold]
    
    # === Step 8: Compute envelope ===
    L = []
    U = []
    FOR i = 1 TO |T|:
        vals = [R_boot_T[b][i] for b in retained]
        L.append(max(0, min(vals)))
        U.append(min(1, max(vals)))
    
    RETURN T, L, U, R_hat_T
```
