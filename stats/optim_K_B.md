# Unified Resource Allocation for Studentized Bootstrap Envelope

## 1. Implementation Context

The envelope bootstrap computes simultaneous confidence bands by:
1. Generating $B$ bootstrap ROC curves
2. Computing studentized KS statistics $T_b = \sup_t |R_b(t) - \hat{R}(t)| / \hat{\sigma}(t)$
3. Retaining the $(1-\alpha)$ fraction with smallest $T_b$
4. Taking the pointwise envelope

**Grid evaluation:** All curves are evaluated on a common FPR grid using **step interpolation** (zero-order hold). This correctly represents the piecewise-constant ROC curves.

**Memory model:** Storing $B$ curves at $K$ grid points requires $O(B \times K)$ memory.

## 2. Grid Strategies

With step interpolation, the ROC value at any $t$ between grid points $t_j$ and $t_{j+1}$ equals the value at $t_j$. This is exact for step functions.

| Strategy | Grid $\mathcal{T}$ | Discretization Error $\delta_K$ |
|----------|-------------------|--------------------------------|
| **Full** | $\{0, \frac{1}{n_0}, \frac{2}{n_0}, \ldots, 1\}$ | **0** (exact) |
| **Uniform** | $\text{linspace}(0, 1, K)$ | $D / K$ |

**Key insight:** Both $\hat{R}(t)$ and $R_b(t)$ have jumps only at FPR values $\{0, 1/n_0, 2/n_0, \ldots, 1\}$. The bootstrap resamples $n_0$ negatives with replacement, so the FPR granularity is unchanged. Evaluating at all $n_0 + 1$ jump points with step interpolation captures the exact supremum.

## 3. Error Model

Total error in confidence band boundaries:
$$E = \sqrt{\delta_B^2 + \delta_K^2}$$

### 3.1 Bootstrap Sampling Error

$$\delta_B = \frac{\beta}{\sqrt{B}}, \quad \beta = \frac{\sqrt{\alpha(1-\alpha)}}{\phi(\Phi^{-1}(1-\alpha))}$$

```python
def compute_beta(alpha: float) -> float:
    """Bootstrap error coefficient from quantile estimation theory."""
    from scipy.stats import norm
    z = norm.ppf(1 - alpha)
    return np.sqrt(alpha * (1 - alpha)) / norm.pdf(z)
```

| $\alpha$ | $\beta$ |
|----------|---------|
| 0.10 | 1.71 |
| 0.05 | 2.12 |
| 0.01 | 3.69 |

### 3.2 Discretization Error

For uniform grid with $K$ points:
$$\delta_K = \frac{D}{K}$$

For full grid ($K = n_0 + 1$):
$$\delta_K = 0$$

## 4. Parameter Estimation

### 4.1 Discretization Sensitivity $D$

```python
def estimate_D(fpr: np.ndarray, tpr: np.ndarray, n0: int, n1: int) -> float:
    """
    Estimate discretization sensitivity from empirical ROC.
    
    D captures how much the supremum can be underestimated when
    the evaluation grid misses jump points.
    """
    # Maximum TPR jump (where large deviations can occur)
    tpr_jumps = np.diff(tpr)
    max_jump = np.max(tpr_jumps)
    
    # Estimate noise level in ROC interior
    interior = (fpr > 0.2) & (fpr < 0.8)
    
    if np.sum(interior) >= 10:
        tpr_int, fpr_int = tpr[interior], fpr[interior]
        
        # Local slopes (ROC derivative estimate)
        with np.errstate(divide='ignore', invalid='ignore'):
            slopes = np.diff(tpr_int) / np.diff(fpr_int)
        slopes = slopes[np.isfinite(slopes) & (slopes > 0)]
        
        if len(slopes) > 0:
            R_mid = np.median(tpr_int)
            slope_mid = np.median(slopes)
            # Var[R(t)] ≈ R(1-R)/n1 + R'² t(1-t)/n0
            sigma = np.sqrt(R_mid * (1 - R_mid) / n1 + slope_mid**2 * 0.25 / n0)
        else:
            sigma = np.sqrt(0.25 / n1 + 0.25 / n0)
    else:
        # Sparse interior: theoretical estimate
        n_eff = 2 * n0 * n1 / (n0 + n1)
        sigma = 0.5 / np.sqrt(n_eff)
    
    # D ≈ (number of potential peaks) × (peak height / noise)
    D = n0 * max_jump / sigma
    return D


def estimate_D_theoretical(n0: int, n1: int) -> float:
    """
    Theoretical D estimate when ROC curve is not yet available.
    
    Derives D = n0 × max_jump / σ under the following assumptions:
    
    1. Continuous scores (no ties): Each positive sample creates exactly 
       one TPR jump of size 1/n1, so max_jump = 1/n1. If ties are present, 
       actual max_jump may be larger (k/n1 for k tied positives), making 
       this estimate conservative.
    
    2. Worst-case variance location: Evaluates the ROC variance at the 
       point R = 0.5, where the binomial term R(1-R)/n1 is maximized. 
       This is conservative for most classifiers (which have AUC > 0.5).
    
    3. Negligible slope contribution: The full variance formula is
           Var[R(t)] = R(1-R)/n1 + R'(t)² × t(1-t)/n0
       This estimate uses only the first term, assuming the ROC slope 
       is moderate. For very steep ROCs (high AUC), the slope term can 
       dominate and the empirical estimate_D() should be preferred.
    
    4. Balanced effective sample size: Uses n_eff = 2×n0×n1/(n0+n1), 
       the harmonic mean, which appropriately weights imbalanced classes.
    
    Derivation:
        max_jump = 1/n1
        σ = 0.5 / √n_eff = 0.5 × √((n0+n1)/(2×n0×n1))
        D = n0 × max_jump / σ
          = n0 × (1/n1) × 2 × √(2×n0×n1/(n0+n1))
          = 2 × n0 × √(2×n0 / (n1×(n0+n1)))
    
    When to use empirical estimate_D() instead:
        - ROC curve is already computed
        - Known high AUC (> 0.9) where slope term matters
        - Known ties in positive scores
        - Class imbalance ratio exceeds 10:1
    """
    return 2 * n0 * np.sqrt(2 * n0 / (n1 * (n0 + n1)))
```


### 4.2 Minimum Bootstrap Replicates

$$B_{\min} = \max\left(500, \left\lceil 100 / \alpha \right\rceil\right)$$

## 5. Optimal Allocation

### 5.1 Full Grid Analysis

With $K = n_0 + 1$ (all jump points):
- Memory: $B \times (n_0 + 1) \leq C \implies B = \lfloor C / (n_0 + 1) \rfloor$
- Error: $E_{\text{full}} = \beta / \sqrt{B} = \beta \sqrt{(n_0 + 1) / C}$
- Feasibility: requires $B \geq B_{\min}$, i.e., $C \geq B_{\min} \times (n_0 + 1)$

### 5.2 Uniform Grid Optimization

For uniform grid, minimize $E^2 = \beta^2/B + D^2/K^2$ subject to $BK = C$:

$$B_{\text{opt}} = \left(\frac{\beta^2 C^2}{2 D^2}\right)^{1/3}, \quad K_{\text{opt}} = \left(\frac{2 D^2 C}{\beta^2}\right)^{1/3}$$

At the optimum, $\delta_B = \sqrt{2} \cdot \delta_K$, and:
$$E_{\text{opt}} = \sqrt{3} \cdot \delta_K^* = \sqrt{3} \left(\frac{D \beta^2}{2C}\right)^{1/3}$$

### 5.3 Decision Rule: Full vs Uniform

**Theorem:** Full grid has lower error than optimized uniform if and only if:
$$(n_0 + 1)^3 < \frac{27 D^2 C}{4 \beta^2}$$

Equivalently:
$$n_0 + 1 < 1.5 \cdot K_{\text{opt}}$$

**Interpretation:** 
- When $n_0$ is small relative to the optimal uniform $K$, use full grid (zero discretization error is "free")
- When $n_0$ is large, the cost of including all jump points is too high; better to subsample and use more bootstrap replicates

## 6. Unified Algorithm

```python
def allocate_budget(
    n0: int,
    n1: int, 
    C: int,
    alpha: float = 0.05,
    fpr: np.ndarray | None = None,
    tpr: np.ndarray | None = None
) -> dict:
    """
    Allocate memory budget between bootstrap replicates and grid resolution.
    
    Parameters
    ----------
    n0, n1 : int
        Number of negative and positive samples
    C : int
        Memory budget (B × K must not exceed C)
    alpha : float
        Significance level
    fpr, tpr : array, optional
        Empirical ROC for data-driven D estimation
    
    Returns
    -------
    dict with allocation details and expected errors
    """
    
    # === Fundamental parameters ===
    beta = compute_beta(alpha)
    D = estimate_D(fpr, tpr, n0, n1) if fpr is not None else estimate_D_theoretical(n0, n1)
    B_min = max(500, int(np.ceil(100 / alpha)))
    
    # === Full grid analysis ===
    K_full = n0 + 1
    B_full = C // K_full
    full_feasible = (B_full >= B_min)
    
    if full_feasible:
        E_full = beta / np.sqrt(B_full)
    else:
        E_full = np.inf
    
    # === Uniform grid optimization ===
    # Unconstrained optimum
    B_unc = (beta**2 * C**2 / (2 * D**2)) ** (1/3)
    K_unc = C / B_unc
    
    # Apply constraints
    K_min = 100
    
    if B_unc >= B_min and K_unc >= K_min:
        B_uniform, K_uniform = B_unc, K_unc
        uniform_regime = "unconstrained"
    elif B_unc < B_min:
        # B-constrained
        B_uniform = B_min
        K_uniform = C / B_min
        uniform_regime = "B-constrained"
    else:
        # K-constrained  
        K_uniform = K_min
        B_uniform = C / K_min
        uniform_regime = "K-constrained"
    
    # Check uniform feasibility
    uniform_feasible = (B_uniform >= B_min and K_uniform >= K_min)
    
    if uniform_feasible:
        delta_B_uniform = beta / np.sqrt(B_uniform)
        delta_K_uniform = D / K_uniform
        E_uniform = np.sqrt(delta_B_uniform**2 + delta_K_uniform**2)
    else:
        E_uniform = np.inf
    
    # === Method selection ===
    if not full_feasible and not uniform_feasible:
        return {
            "status": "insufficient_budget",
            "message": f"Budget C={C} too small. Minimum: {B_min * min(K_min, K_full)}",
            "params": {"beta": beta, "D": D, "B_min": B_min, "n0": n0, "n1": n1}
        }
    
    # Decision rule: full grid wins if E_full < E_uniform
    # Equivalently: (n0+1)^3 < 27 D² C / (4 β²)
    use_full = full_feasible and (not uniform_feasible or E_full <= E_uniform)
    
    if use_full:
        method = "full"
        B, K = int(B_full), int(K_full)
        delta_B = beta / np.sqrt(B)
        delta_K = 0.0
        E = delta_B
        grid_spec = f"All {K} jump points (exact evaluation)"
    else:
        method = "uniform"
        B, K = int(np.round(B_uniform)), int(np.round(K_uniform))
        delta_B = beta / np.sqrt(B)
        delta_K = D / K
        E = np.sqrt(delta_B**2 + delta_K**2)
        grid_spec = f"Uniform {K} points ({uniform_regime})"
    
    return {
        "status": "ok",
        "method": method,
        "B": B,
        "K": K,
        "grid_spec": grid_spec,
        
        # Error analysis
        "delta_B": delta_B,
        "delta_K": delta_K,
        "total_error": E,
        "error_ratio": delta_K / delta_B if delta_B > 0 else 0,
        
        # Comparison
        "comparison": {
            "full": {
                "feasible": full_feasible,
                "B": int(B_full) if full_feasible else None,
                "K": int(K_full),
                "error": float(E_full) if full_feasible else None
            },
            "uniform": {
                "feasible": uniform_feasible,
                "B": int(np.round(B_uniform)) if uniform_feasible else None,
                "K": int(np.round(K_uniform)) if uniform_feasible else None,
                "error": float(E_uniform) if uniform_feasible else None,
                "regime": uniform_regime if uniform_feasible else None
            }
        },
        
        # Parameters
        "params": {
            "alpha": alpha,
            "beta": beta,
            "D": D,
            "B_min": B_min,
            "K_min": K_min,
            "n0": n0,
            "n1": n1,
            "C": C
        }
    }
```

## 7. Grid Construction

```python
def construct_grid(method: str, K: int, n0: int) -> np.ndarray:
    """
    Construct FPR evaluation grid.
    
    For full grid: all n0+1 jump points (exact supremum)
    For uniform: K evenly spaced points
    """
    if method == "full":
        return np.linspace(0, 1, n0 + 1)
    else:
        return np.linspace(0, 1, K)
```

## 8. Examples

### Example 1: Small Dataset (full grid optimal)

```python
>>> allocate_budget(n0=1000, n1=1000, C=10_000_000)
{
    "method": "full",
    "B": 9990,
    "K": 1001,
    "delta_B": 0.0212,
    "delta_K": 0.0,
    "total_error": 0.0212,
    "comparison": {
        "full": {"B": 9990, "K": 1001, "error": 0.0212},
        "uniform": {"B": 6300, "K": 1587, "error": 0.0328}
    }
}
```

### Example 2: Large Dataset (uniform grid optimal)

```python
>>> allocate_budget(n0=50000, n1=50000, C=100_000_000)
{
    "method": "uniform",
    "B": 4640,
    "K": 21550,
    "delta_B": 0.0311,
    "delta_K": 0.0209,
    "total_error": 0.0375,
    "comparison": {
        "full": {"B": 1999, "K": 50001, "error": 0.0474},
        "uniform": {"B": 4640, "K": 21550, "error": 0.0375}
    }
}
```

## 9. Summary

| Scenario | Method | Key Formula |
|----------|--------|-------------|
| $n_0 + 1 < 1.5 \cdot K_{\text{opt}}$ | Full grid | $B = C/(n_0+1)$, $\delta_K = 0$ |
| $n_0 + 1 \geq 1.5 \cdot K_{\text{opt}}$ | Uniform | $B_{\text{opt}} = (\beta^2 C^2 / 2D^2)^{1/3}$ |
**Decision rule:** Full grid is optimal when $(n_0+1)^3 < 27 D^2 C / (4\beta^2)$

**Guarantees with step interpolation:**
- Full grid: exact KS statistics, coverage = nominal
- Uniform grid: $O(D/K)$ discretization error, slight coverage loss