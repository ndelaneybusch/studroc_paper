## 1. Computational Resource Allocation

This guide details how to allocate computational resources (Budget $C$) between the number of bootstrap replicates ($B$) and the evaluation grid resolution ($K$) for the **Studentized Bootstrap Envelope** method described in `nonparam_envelope.md`.

### 1.1 Goal

Minimize the total error in the confidence band boundaries subject to a computational budget $C$ (total curve evaluations).

$$C = B \times K_{eff}$$

where $K_{eff}$ is the effective number of grid points.

### 1.2 Grid Strategies and Error

The choice of grid $\mathcal{T}$ fundamentally alters the error structure:

**1. Exact or Hybrid Grid ($\mathcal{T} \supseteq \mathcal{J}_0$)**
- **Definition:** Includes all $n_0$ jump points of the empirical ROC.
- **Discretization Error ($\delta_K$):** **Zero**. Since $R_b(t)$ and $\hat{R}(t)$ are step functions jumping only at observed negative scores, evaluating at these jumps captures the exact supremum of the difference.
- **Cost:** $K_{eff} \approx n_0$.
- **Constraint:** Requires $C \ge B_{min} \times n_0$.

**2. Uniform Grid**
- **Definition:** $K$ equally spaced points, not necessarily aligned with jumps.
- **Discretization Error ($\delta_K$):** Positive. We may miss the peak deviation if it occurs between grid points.
- **Approximation:** $\delta_K \approx \frac{D}{K}$.
- **Cost:** $K_{eff} = K$.
- **Use Case:** When $n_0$ is too large to afford the Exact grid given the budget.

### 1.3 Error Decomposition

The total error $E$ combines bootstrap variance and discretization bias:

$$E = \sqrt{\delta_B^2 + \delta_K^2}$$

#### Bootstrap Error ($\delta_B$)
Standard error of the $(1-\alpha)$ quantile estimator:
$$\delta_B = \frac{\beta}{\sqrt{B}}, \quad \text{where } \beta \approx \frac{\sqrt{\alpha(1-\alpha)}}{f(c_\alpha)}$$

#### Discretization Error ($\delta_K$)
(Only for Uniform Grid)
$$\delta_K \approx \frac{D}{K}$$
where $D$ is the **discretization sensitivity**.
- **Theoretical Bound:** $D \approx 2n_0 \sqrt{\frac{2n_0}{n_1(n_0+n_1)}}$ (assuming worst-case alignment).
- **Empirical Estimate:** See Section 1.5.

### 1.4 Optimization Strategy

**Priority 1: Use Exact/Hybrid Grid**
If the budget allows for at least $B_{min}$ replicates using the full grid ($n_0$ points), this is optimal because $\delta_K = 0$.
- Set $K = n_0$ (or $n_0 + 100$ for Hybrid).
- Set $B = \lfloor C / K \rfloor$.

**Priority 2: Optimize Uniform Grid**
If $n_0$ is too large ($C / n_0 < B_{min}$), we must subsample the grid. We minimize $E(B, K)$ for the Uniform case:
$$E^2 = \frac{\beta^2}{B} + \frac{D^2}{K^2}, \quad \text{s.t. } B \cdot K = C$$

**Optimal Uniform Allocation:**
$$B_{opt} = \left(\frac{\beta^2 C^2}{2D^2}\right)^{1/3}, \quad K_{opt} = \left(\frac{2D^2 C}{\beta^2}\right)^{1/3}$$

### 1.5 Parameter Estimation

Avoid relying on assumptions by estimating parameters from the data:

**Discretization Constant ($D$):**
Estimate using the empirical ROC properties (jump heights and local variance).
```python
FUNCTION estimate_D(fpr, tpr, n0, n1):
    # Max jump height (typically 1/n1)
    max_jump = max(diff(tpr))
    
    # Interior variance proxy (roughly 0.5 * standard error at R=0.5)
    # Fallback to balanced assumption if interior is sparse
    sigma_int = 0.5 / sqrt(2 * n0 * n1 / (n0 + n1))
    
    # D scales with density of jumps (n0) * jump_height / sigma
    D = n0 * max_jump / sigma_int
    RETURN D
```

**Bootstrap Coefficient ($\beta$):**
Default conservative estimate: $\beta \approx 1.5$ (for $\alpha=0.05$).
For high precision, run a small pilot bootstrap to estimate $f(c_\alpha)$.

### 1.6 Complete Allocation Algorithm

```python
FUNCTION allocate_budget(n0, n1, C, alpha=0.05):
    
    # 1. Constants
    B_min = max(1000, ceil(100 / alpha))  # Min samples for stable quantile
    beta = 1.5                            # Default coefficient
    
    # 2. Check Feasibility of Exact Grid
    K_exact = n0
    B_feasible_exact = floor(C / K_exact)
    
    IF B_feasible_exact >= B_min:
        # === STRATEGY: EXACT ===
        RETURN {
            "method": "hybrid",
            "B": B_feasible_exact,
            "K": K_exact,
            "desc": "Exact grid feasible. Discretization error is zero."
        }
    
    # 3. Fallback to Uniform Grid Optimization
    # Estimate D (using theoretical proxy if data not available)
    D = 2 * n0 * sqrt(2 * n0 / (n1 * (n0 + n1)))
    
    # Calculate Optimal Unconstrained B
    B_opt = ( (beta^2 * C^2) / (2 * D^2) )^(1/3)
    
    # 4. Apply Constraints
    B_final = max(B_opt, B_min)
    K_final = floor(C / B_final)
    
    # Ensure K is not absurdly small
    K_min = 200
    IF K_final < K_min:
        K_final = K_min
        B_final = floor(C / K_final)
        # Warning: Budget is very low, errors will be high
        
    RETURN {
        "method": "uniform",
        "B": round(B_final),
        "K": K_final,
        "desc": "Exact grid too expensive. Using optimized uniform grid."
    }
```

### 1.7 Memory-Constrained Environments

If the budget $C$ is limited by RAM rather than time:

1.  **Calculate Max Budget:** $C_{mem} = \text{AvailableRAM} / 4$ (for float32).
2.  **Run Allocation:** `allocate_budget(n0, n1, C_mem)`.
3.  **Two-Pass Override:**
    If the resulting $B$ or $K$ is unsatisfactory (e.g., high error), you can trade **Time for Memory** using the **Two-Pass Algorithm** (see `nonparam_envelope.md`).
    - In Two-Pass mode, memory is $O(B + K)$ instead of $O(B \times K)$.
    - You can effectively set $C = \infty$ (or time-bound) for the allocation logic, get the optimal large $B$ and $K$, and execute it using the Two-Pass method.

### 1.8 Summary of Recommendations

| Scenario | $n_0$ | Recommendation |
| :--- | :--- | :--- |
| **Small/Medium Data** | $< 5,000$ | **Hybrid Grid.** Use all data points. $K \approx n_0$. Maximize $B$. |
| **Large Data** | $> 50,000$ | **Uniform Grid.** Optimize $B$ vs $K$. Typically $K \in [2k, 10k]$. |
| **Low Memory** | Any | **Two-Pass Algorithm.** Decouple $B$ and $K$ from RAM. |

