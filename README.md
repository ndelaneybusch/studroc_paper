# ROC Confidence Band Methods: A Comprehensive Simulation Study

This repository contains simulation code and analysis for evaluating methods for constructing simultaneous confidence bands for Receiver Operating Characteristic (ROC) curves under diverse distributional conditions.

There is a development library containing working, semi-optimized code for the methods implemented in this paper inside the `src` directory.

## Overview

The fundamental challenge addressed by this research is constructing valid confidence bands for ROC curves when the **binormal assumption** (scores follow Normal distributions for both classes) cannot be guaranteed. While classical methods like Working-Hotelling bands assume binormal scores, real-world applications often involve heavy-tailed, skewed, or multimodal distributions that violate this assumption (for example, deep learning and large language model-based classifier scores are often heavy-tailed).

This project implements:
- **9 confidence band methods** (literature references and novel bootstrap approaches)
- **18+ data-generating processes** spanning 8 distribution families
- **Rigorous evaluation framework** using Latin Hypercube Sampling
- **a simulator for conducting millions of method evaluations** per confidence level across diverse scenarios

## Key Research Questions

1. How do classical methods (Working-Hotelling, Ellipse-Envelope) perform when binormality is violated? How overconservative is Kolmogorov-Smirnov in such cases?
2. Can bootstrap-based methods achieve valid coverage across heavy-tailed, skewed, and multimodal distributions?
3. What is the cost of distribution-free methods in terms of band width?
4. How do methods handle boundary behavior (near FPR=0 or FPR=1) and class imbalance?

---

## Installation

### Requirements

- Python 3.12+
- UV package manager (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ndelaneybusch/studroc_paper.git
cd studroc_paper
```

2. Install dependencies using UV:
```bash
uv sync
```

This will install all required packages including:
- Core: `numpy`, `scipy`, `pandas`, `scikit-learn`
- Visualization: `matplotlib`, `seaborn`, `colorspacious`
- Machine learning: `torch`, `pygam`
- Density estimation: `KDEpy`
- Optimization: `cvxpy`

---

## Data Generation Approach

### Core Testing Philosophy

The simulation framework uses a **generate-and-verify** approach:

1. **Generate data** from parametric distributions with known parameters
2. **Derive the true population ROC curve** using closed-form or numerical methods
3. **Compute confidence bands** from finite samples
4. **Evaluate coverage** by checking if the true ROC lies within the band

This approach provides **ground truth coverage assessment** without relying on asymptotic approximations.

### True ROC Derivation Methods

For each distribution family, the true population ROC is computed via:

#### Method 1: Closed-Form Analytical Solutions

For distributions with tractable quantile functions:

**Gaussian (Equal Variance):**
```
TPR(FPR) = Φ(d' + Φ⁻¹(FPR))
where d' = (μ₁ - μ₀)/σ
```

**Exponential:**
```
TPR = FPR^(λ_pos/λ_neg)  [Power function]
```

**Heteroskedastic Gaussian:**
```
TPR(FPR) = Φ(Δμ/σ_pos + (σ_neg/σ_pos)·Φ⁻¹(FPR))
```

#### Method 2: Numerical Inversion via Root-Finding

For distributions with tractable CDFs but intractable quantile functions:

**Beta, Gamma, Student's t:** Use Brent's method to solve:
```
For each FPR value f:
  1. Find threshold: t such that P(X_neg > t) = f
  2. Compute TPR: P(X_pos > t)
```

**Gaussian Mixtures:** Numerically invert mixture survival functions.

#### Method 3: Monte Carlo Estimation (Fallback)

When neither analytic nor numerical solutions are feasible:
```python
# Generate very large sample (n=100,000)
scores_pos, scores_neg = dgp.sample(100000, 100000, rng)
thresholds = np.quantile(scores_neg, 1 - fpr_grid)
tpr = [(scores_pos >= t).mean() for t in thresholds]
# Standard error ≈ sqrt(TPR·(1-TPR)/n) < 0.002
```

### Available Data Generating Processes

#### 1. **Gaussian (Equal Variance)**
- **Distribution:** N(0, σ²) vs N(Δμ, σ²)
- **Use case:** Baseline comparison, simplest case with symmetric ROC

#### 2. **Heteroskedastic Gaussian**
- **Distribution:** N(0, 1) vs N(Δμ, σ_ratio)
- **Parameters:** AUC ∈ [0.55, 0.99], σ_ratio ∈ [0.2, 5.0]
- **Use case:** Tests robustness to unequal variances (violates Working-Hotelling assumption)

#### 3. **Log-Normal**
- **Distribution:** N(0, 1) vs LogNormal(μ_pos, σ)
- **Parameters:** AUC ∈ [0.55, 0.99], σ ∈ [0.1, 3.0]
- **Use case:** Right-skewed positive scores (common in diagnostic tests, credit scoring)
- **Note:** Has same ROC as Gaussian after log-transform - best treatment is to transform and then use binormal methods (because the assumptions are met). But all users of ROC curves may not be aware of this, and may use raw scores suboptimally (so e.g. performance on log-normal data may be valuable to know).

#### 4. **Beta (Opposing Skewness)**
- **Distribution:** Beta(α, β) vs Beta(β, α)
- **Parameters:** AUC ∈ [0.55, 0.99], α ∈ [0.5, 10.0]
- **Use case:** Bounded support [0,1], opposing skewness patterns
- **True ROC:** Numerical inversion via incomplete beta functions

#### 5. **Student's t**
- **Distribution:** t_ν(loc=0) vs t_ν(loc=Δ)
- **Parameters:** AUC ∈ [0.55, 0.99], df ∈ [1.1, 30.0]
- **Use case:** **Heavy tails**, outliers (df→∞ converges to Gaussian, df=1 is Cauchy), scores from neural network methods.
- **Critical test:** Binormal methods fail catastrophically for low df

#### 6. **Gamma**
- **Distribution:** Gamma(k_neg, θ_neg) vs Gamma(k_pos, θ_pos)
- **Use case:** Right-skewed, flexible shapes
- **True ROC:** Numerical solution (no closed form)

#### 7. **Bimodal Negative**
- **Distribution:** Gaussian mixture (2 modes) vs N(μ_pos, 1)
- **Parameters:** AUC, mixture_weight ∈ [0.1, 0.9], mode_separation ∈ [0.1, 4.0]
- **Use case:** Models "easy" and "hard" negatives, creates ROC inflection points
- **True ROC:** Numerical inversion of mixture survival functions

#### 8. **Exponential**
- **Distribution:** Exp(λ_neg) vs Exp(λ_pos)
- **Parameters:** AUC ∈ [0.55, 0.99], λ_neg ∈ [0.1, 10.0]
- **Use case:** Memoryless property, bounded support at zero
- **True ROC:** Power function (closed form)

#### 9. **Uniform (Partial/Full/No Overlap)**
- **Distribution:** U(a,b) vs U(c,d) with varying overlap
- **True ROC:** Piecewise linear

#### 10. **Pareto, Cauchy, Weibull** (stress tests)
- Test extreme tail behavior and pathological cases

---

## Implemented Confidence Band Methods

### Literature Reference Methods

#### 1. **Working-Hotelling Band** (`working_hotelling_band`)

**Assumptions:**
- **Binormal scores** (Normal distributions for both classes)
- Approximately equal variances

**Approach:**
- Estimates ROC parameters (a, b) via method of moments
- Constructs bands in probit space (Φ⁻¹-transformed FPR/TPR)
- Uses chi-squared critical value (df=2): W = √(χ²₁₋ₐ,₂)

**Expected Performance:**
- ✅ **Excellent for binormal data:** Near-optimal coverage and width
- ❌ **Poor for studentized/heavy-tailed distributions:** Severely undercovers as n increases
  - With Student-t scores, probit transformation breaks down
  - Converges to the **wrong curve** (binormal approximation ≠ true ROC)
  - Pathologies increase with sample size when assumptions are violated.
- ❌ **Degrades with skewness or multimodality**

#### 2. **Ellipse-Envelope Band** (`ellipse_envelope_band`)

**Assumptions:**
- **Binormal scores** (same as Working-Hotelling)
- Asymptotic normality of parameter estimates

**Approach:**
- Extends Working-Hotelling by accounting for variance estimation uncertainty
- Constructs confidence ellipses around (a, b) parameter estimates
- Two computation methods: sweep (vectorized) and quartic (exact polynomial)

**Expected Performance:**
- ✅ **Superior to Working-Hotelling for binormal data:** Better coverage by accounting for variance uncertainty
- ❌ **Similar failure mode with studentized distributions:** Still relies on binormal model
- ⚠️ **More conservative:** Wider bands when binormal assumption holds

#### 3. **Fixed-Width KS Band** (`fixed_width_ks_band`)

**Assumptions:**
- **None** (distribution-free)
- Treats FPR and TPR as independent (ignores correlation)

**Approach:**
- Based on Kolmogorov-Smirnov two-sample test theory
- Rectangular bands with fixed margins: vertical d = c_α/√n₁, horizontal e = c_α/√n₀
- Bonferroni-style correction: α_adj = 1 - √(1-α)

**Expected Performance:**
- ✅ **Conservative (overcovers) across all data types:** 97-99% coverage for nominal 95%
- ✅ **Distribution-free:** Works for any continuous score distribution
- ⚠️ **Uniform width:** Doesn't adapt to local ROC curvature
- ✅ **Good for small samples:** Reliable when n < 50 per class

#### 4. **Pointwise Bootstrap Band** (`pointwise_bootstrap_band`)

**Assumptions:**
- **Independence across FPR grid** (treats each point separately)

**Approach:**
- Computes pointwise quantiles at each FPR: [Q_{α/2}, Q_{1-α/2}]
- No adjustment for simultaneous inference

**Expected Performance:**
- ❌ **Severely anticonservative for simultaneous inference**
  - Pointwise coverage ≈ 1-α, but simultaneous coverage ≪ 1-α
  - Can drop to 60-70% coverage for nominal 95% with k=1000 grid points
- ✅ **Tight bands:** Narrower than any valid simultaneous method
- ⚠️ **Useful as lower bound:** Shows minimum achievable width

---

### Novel Bootstrap Methods

#### 5. **BP-Smoothed Bootstrap Band** (`bp_smoothed_bootstrap_band`)

**Approach:**
- Uses **Bernstein polynomial smoothing** (degree m = max(10, n^0.4))
- BP-smoothed CDF: F̃ₘ(u) = Σₖ F̂(k/m) · Bₖ,ₘ(u)
- Exact ROC computation via numerical CDF/quantile evaluation
- Studentized retention with Wilson variance floor
- Two retention methods: KS (sup|Z|) or symmetric (separate tail trimming)

**Key Characteristics:**
- ✅ **Distribution-agnostic:** No parametric assumptions
- ✅ **Smoothing reduces noise:** BP dampens extreme bootstrap replicates
- ✅ **Handles bounded support:** Support extension prevents boundary bias
- ✅ **GPU-accelerated:** Batched sampling via PyTorch

**Expected Performance:**
- ✅ **Robust across distributions:** Should maintain coverage for normal, heavy-tailed, skewed, multimodal
- ⚠️ **Bias-variance tradeoff:** Degree parameter controls smoothing

#### 6. **Envelope Bootstrap Band** (`envelope_bootstrap_band`)

**Approach:**
- Studentized bootstrap with multiple boundary correction methods:
  - `wilson`: Wilson score variance floor (binomial-based)
  - `reflected_kde`: Hsieh-Turnbull variance with reflected KDE
  - `log_concave`: Hsieh-Turnbull variance with log-concave MLE
  - `ks`: KS-style margin extension
- Retention methods: `ks` (sup|Z|) or `symmetric` (separate tails)
- Optional logit-space construction for variance stabilization

**Key Characteristics:**
- ✅ **Modular design:** Mix-and-match boundary methods and retention strategies
- ✅ **Variance floor prevents collapse:** Wilson/KS ensure minimum band width
- ✅ **Logit stabilization:** Prevents "pinching" at corners
- ✅ **Harrell-Davis TPR option:** Reduces finite-sample bias

**Expected Performance:**
- ✅ **Best with `wilson` or `ks` boundary:** Principled minimum widths
- ✅ **Symmetric retention:** Better for high-AUC classifiers with skewed distributions

#### 7. **Max-Modulus Bootstrap Band** (`logit_bootstrap_band`)

**Approach:**
- Constructs bands entirely in **logit space** with Haldane-Anscombe correction
- Haldane transform: logit(TPR) = log[(k+0.5)/(n-k+0.5)]
- Asymptotic SE in logit space: σ_logit = 1/√[n·p̂·(1-p̂)]
- Single critical value from sup|Z_b| quantile
- Back-transform via sigmoid

**Key Characteristics:**
- ✅ **Pure logit-space construction:** Most aggressive variance stabilization
- ✅ **Boundary-safe:** Haldane correction ensures finite logits at TPR=0/1
- ✅ **Simpler:** Single critical value (less conservative than envelope)

**Expected Performance:**
- ✅ **Excellent boundary behavior:** Naturally handles corners
- ⚠️ **Assumes asymptotic normality in logit space:** May undercover for very small samples
- ✅ **Tighter than envelope methods**

#### 8. **Hsieh-Turnbull Band** (`hsieh_turnbull_band`)

**Approach:**
- Asymptotic variance formula: Var[R̂(t)] = R(t)(1-R(t))/n₁ + [R'(t)]² · t(1-t)/n₀
- Density estimation for R'(t) = g(c)/f(c):
  - `log_concave`: Convex optimization with concavity constraint (cvxpy)
  - `reflected_kde`: Boundary-reflected kernel density estimation
- Optional bootstrap calibration for critical value
- Optional logit-space construction and Wilson variance floor

**Key Characteristics:**
- ✅ **Theoretically grounded:** Based on Hsieh & Turnbull (1996)
- ✅ **Log-concave MLE robust:** Enforces shape constraint, prevents density ratio explosions
- ✅ **Bootstrap calibration adapts:** To actual correlation structure

**Expected Performance:**
- ✅ **Excellent for log-concave distributions:** Near-optimal coverage and width
- ❌ **Fails for heavy-tailed distributions:** Density ratio unstable in tails
- ⚠️ **Bootstrap calibration critical:** Without it, uses conservative √k heuristic
- ⚠️ **Kurtosis check:** Built-in warning if excess kurtosis > 2.0

---

## Evaluation Framework

### Metrics Computed Per Confidence Band

#### Coverage Metrics
- `covers_entirely`: Whether band fully contains true ROC (target: 1-α)
- `violation_above/below`: Directional violation indicators
- `pointwise_covered`: Coverage at each FPR grid point

#### Violation Location Metrics
- `violation_fpr_above/below`: FPR values where violations occur
- `violation_by_region`: Violations in FPR ranges (0-10, 10-30, ..., 90-100)
- `violation_fpr_mean/median/min/max`: Summary statistics

#### Violation Magnitude Metrics
- `max_violation_above/below`: Maximum distance from true ROC

#### Band Tightness Metrics
- `band_area`: Total area between upper and lower bands
- `band_widths`: Width at each FPR point
- `proportion_grid_points_violated`: Fraction of grid violated

### Aggregated Metrics (Across Simulations)

#### Overall Coverage
- `coverage_rate`: Proportion achieving full coverage (compare to 1-α)
- `coverage_se` and `coverage_ci_lower/upper`: Uncertainty quantification

#### Directional Patterns
- `violation_rate_above/below`: Proportion with violations in each direction
- `direction_test_pvalue`: Binomial test for symmetry (H₀: P(above) = P(below))

#### Band Tightness
- `mean_band_area` and `std_band_area`: Average and variability of width
- `width_by_fpr_region`: Mean width in each FPR region

#### Pointwise Analysis
- `pointwise_coverage_rates`: Coverage at each FPR point
- `violation_rate_by_region`: Regional violation patterns

### Interpretation: Good vs. Bad Performance

#### GOOD Performance Criteria:
1. **Valid coverage:** `coverage_rate` ≈ (1-α) within statistical uncertainty
2. **No systematic bias:** `direction_test_pvalue` > 0.05 (symmetric violations)
3. **Tight bands:** Low `mean_band_width` conditional on valid coverage
4. **Small violations:** Low `mean_max_violation` when coverage fails

#### BAD Performance Patterns:
1. **Undercoverage:** Coverage far below nominal (method too aggressive)
2. **Systematic bias:** Violations predominantly in one direction or region
3. **Overcoverage with excessive width:** Valid but wasteful

### Example: Student's t (n=10000, df∈[1.1,30], AUC∈[0.55,0.99])

**KS Method (α=0.05) - EXCELLENT:**
- Coverage: 100% (CI: [99.2%, 100%])
- Violations: 0% above, 0% below
- Mean width: 0.077
- **Verdict:** Valid, unbiased, reasonably tight

**Pointwise (α=0.05) - CATASTROPHIC:**
- Coverage: 14.4% (CI: [11.6%, 17.8%])
- Violations: 34.8% above, 73.2% below
- Mean width: 0.024
- **Verdict:** Ignores multiplicity, artificially narrow

---

## Simulation Design: Latin Hypercube Sampling

### Why Latin Hypercube Sampling?

The simulation uses **maximin Latin Hypercube Sampling (LHS)** to systematically explore the parameter space of ROC curves with three key properties:

1. **Space-filling:** Comprehensive coverage with minimal redundancy
2. **Maximin optimization:** Maximizes minimum distance between design points (avoids clustering)
3. **Orthogonality:** Minimizes correlation between dimensions (enables main effect estimation)

### Maximin LHS Algorithm

**Sequential Build Procedure:**
1. Divide each dimension into n equally-probable strata
2. Randomly place first design point
3. For each subsequent point:
   - Generate 5 × remaining candidates
   - Select candidate that maximizes minimum distance to existing points
4. Transform grid positions to [0,1]^k via uniform jittering

**Result:** 1000 samples provide coverage comparable to 3000-5000 random samples.

### Six Data Generating Processes

The simulation implements six DGPs, each targeting different distributional characteristics:

| DGP | Parameters | Bounds | What It Tests |
|-----|------------|--------|---------------|
| **Lognormal** | AUC, sigma | [0.55,0.99], [0.1,3.0] | Right-skewness, common in diagnostics |
| **Heteroskedastic** | AUC, σ_ratio | [0.55,0.99], [0.2,5.0] | Unequal variances (violates W-H) |
| **Beta Opposing** | AUC, alpha | [0.55,0.99], [0.5,10.0] | Bounded support, opposing skewness |
| **Student's t** | AUC, df | [0.55,0.99], [1.1,30.0] | Heavy tails, outliers (critical test) |
| **Bimodal Negative** | AUC, mixture_wt, separation | [0.55,0.99], [0.1,0.9], [0.1,4.0] | Multimodality, inflection points |
| **Exponential** | AUC, neg_rate | [0.55,0.99], [0.1,10.0] | Bounded at zero, exponential decay |

**Total experimental design:**
- 6 DGPs × 8 sample configs × 1000 LHS combinations × 24 methods
- **1,152,000 method evaluations** per confidence level

### Sample Size Configurations

**Balanced designs:** n_pos = n_neg ∈ {10, 30, 100, 300, 1000, 10000}

**Imbalanced designs (n=1000 only):**
- Prevalence 1%: n_pos=10, n_neg=990
- Prevalence 10%: n_pos=100, n_neg=900
- Prevalence 50%: n_pos=500, n_neg=500

### Why This Design is Rigorous

1. **Orthogonal dimensions:** AUC sampled independently from shape parameters
   - Enables clean attribution: "How does sigma affect coverage?"
   - Detects interactions: "Does sigma's effect depend on AUC?"

2. **Space-filling ensures coverage of:**
   - **Corners:** Extreme combinations (AUC=0.99 + heavy tails)
   - **Interior:** Typical combinations (AUC=0.75 + moderate skewness)
   - **Boundaries:** Edge cases (AUC=0.55 near random guessing)

3. **DGP diversity tests:**
   - Skewness: Lognormal (right), Beta (opposing), Bimodal (multimodal)
   - Tails: Student's t (heavy), Gaussian (light), Exponential (bounded)
   - Variance: Heteroskedastic (unequal)
   - Support: Beta [0,1], Exponential [0,∞), others unbounded

---

## Repository Structure

```
studroc_paper/
├── src/studroc_paper/
│   ├── datagen/           # Data generating processes and true ROCs
│   │   ├── true_rocs.py   # 18+ DGP implementations with true ROC functions
│   │   └── roc_to_dgp.py  # LHS parameter mapping and numerical solvers
│   ├── methods/           # Confidence band methods
│   │   ├── working_hotelling.py
│   │   ├── ellipse_envelope.py
│   │   ├── ks_band.py
│   │   ├── pointwise_boot.py
│   │   ├── bp_smoothed_boot.py
│   │   ├── envelope_boot.py
│   │   ├── max_modulus_boot.py
│   │   └── hsieh_turnbull_band.py
│   ├── eval/              # Evaluation framework
│   │   ├── eval.py        # Coverage and width metrics
│   │   └── build_data_from_jsons.py  # Result aggregation
│   └── sampling/
│       └── lhs.py         # Maximin Latin Hypercube Sampling
├── scripts/
│   └── run_simulation.py  # Main simulation driver
├── data/
│   └── results/           # Aggregated JSON results
├── figures/               # Visualization outputs
├── tests/                 # Unit tests
└── pyproject.toml         # Dependencies and config
```

---

## Running Simulations

### Basic Usage

Run a simulation for a specific DGP and sample size:

```bash
uv run python scripts/run_simulation.py \
  --dgp student_t \
  --n_total 1000 \
  --n_lhs 1000 \
  --output_dir data/results \
  --seed 42
```

### Output Format

Results are saved as JSON files with structure:
```json
{
  "method_name": {
    "coverage_rate": 0.95,
    "coverage_ci_lower": 0.93,
    "coverage_ci_upper": 0.97,
    "mean_band_width": 0.08,
    "violation_rate_above": 0.02,
    "violation_rate_below": 0.03,
    "direction_test_pvalue": 0.45,
    ...
  }
}
```

---

## Development

### Code Style

This project follows modern Python 3.12+ conventions:
- Type hints using `|` instead of `Union[]`
- Prefer `list` over `List`, `dict` over `Dict`
- Use `pathlib.Path` for file operations
- Google-style docstrings
- Black formatting with line length 88

### Running Tests

```bash
uv run pytest
```

### Linting and Formatting

```bash
# Check code style
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### Type Checking

```bash
uv run mypy src/studroc_paper
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{studroc2025,
  author = {Delaney-Busch, Nathaniel},
  title = {ROC Confidence Band Methods: A Comprehensive Simulation Study},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/ndelaneybusch/studroc}
}
```

---

## Acknowledgments

This work implements and extends methods from:
- Working-Hotelling (1929): Joint confidence regions
- Campbell (1994): Kolmogorov-Smirnov confidence bands for ROC
- Hsieh & Turnbull (1996): Asymptotic variance of ROC curves
- Demidenko (2012): Ellipse-envelope confidence bands

The Bernstein polynomial smoothing approach draws from nonparametric statistics literature, and the logit-space methods build on variance stabilizing transformations for binomial proportions.
