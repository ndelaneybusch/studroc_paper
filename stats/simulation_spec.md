# Simulation Specification: Confidence Band Evaluation (Final Run)

This document specifies the final paper run comparing the studentized bootstrap
envelope band (`envelope_boot.py`) against baselines and its own ablations
across diverse Data Generating Processes (DGPs).

## 1. Objectives

-   Evaluate coverage and tightness of confidence bands under diverse data conditions.
-   Attribute the envelope method's behavior to its components via ablations
    (Beta order-statistic floor, Wilson floor, bootstrap) and symmetric-tail variants.
-   Assess the robustness of the parametric baseline (Working-Hotelling) when
    binormal assumptions are violated.

## 2. Data Generating Processes (DGPs)

We use the DGPs defined in `src/studroc_paper/datagen/true_rocs.py`. For each DGP,
a "Sampling Space" of key properties (AUC, shape) is sampled by Latin Hypercube
Sampling (LHS) and mapped to DGP parameters by `src/studroc_paper/datagen/roc_to_dgp.py`.

### AUC sampling: probit scale

The AUC dimension is sampled **uniformly in z = Φ⁻¹(AUC)** (equivalently, uniformly
in binormal d′ = √2·z), then pushed through Φ. Bounds are stated in AUC units;
z spans [Φ⁻¹(0.55), Φ⁻¹(0.99)] ≈ [0.126, 2.326]. Relative to uniform-in-AUC
sampling, the design density is ∝ 1/φ(z), about 15× higher at AUC = 0.99 than at
0.55 — concentrating the design in the high-AUC regime where tail behavior
dominates. (Rank-biserial correlation, r = 2·AUC − 1, is a linear transform and
would not change the sampling density.)

### Active DGP list and sampling spaces

1.  **Binormal (strictly equal variances)** — `make_heteroskedastic_gaussian_dgp(σ_pos = σ_neg = 1)`
    -   `AUC`: $[0.55, 0.99]$ (1-D design)
2.  **Heteroskedastic Gaussian** — `make_heteroskedastic_gaussian_dgp`
    -   `AUC`: $[0.55, 0.99]$; `sigma_ratio` ($\sigma_{pos}/\sigma_{neg}$): $[0.2, 5.0]$
3.  **Logit-Normal** — `make_logitnormal_dgp`
    -   `AUC`: $[0.55, 0.99]$; `sigma`: $[0.1, 3.0]$
4.  **Beta (Opposing Skew)** — `make_beta_opposing_skew_dgp`
    -   `AUC`: $[0.55, 0.99]$; `alpha`: $[0.5, 10.0]$
5.  **Student's t (heavy tails)** — `make_student_t_dgp`
    -   `AUC`: $[0.55, 0.99]$; `df`: $[1.1, 30.0]$
6.  **Bimodal Negative (mixture)** — `make_bimodal_negative_dgp` (unit component SDs)
    -   `AUC`: $[0.55, 0.99]$; `mixture_weight`: $[0.1, 0.9]$; `mode_separation`: $[0.1, 4.0]$
7.  **Weibull** — `make_weibull_dgp`
    -   `AUC`: $[0.55, 0.99]$; `shape`: $[0.5, 5.0]$

Known redundancies across this list (identical-ROC slices, inert parameter axes
for rank-based methods) are documented in `stats/dgp_redundancy.md`; they are
retained deliberately (replication / placebo checks / WH robustness probes).

## 3. Sampling Strategy (LHS)

For each DGP:
1.  Generate $N_{LHS} = 1000$ maximin LHS points (`src/studroc_paper/sampling/lhs.py`),
    decorrelated by the Iman–Conover transform (skipped for 1-D designs).
2.  Scale to bounds; the AUC column goes through the probit transform above.
3.  Map to DGP parameters via `map_lhs_to_dgp`; combinations with unachievable
    AUC are filtered out.

## 4. Simulation Parameters

-   **Sample Sizes ($n$)**: $\{10, 30, 100, 300, 1000, 10000\}$
-   **Prevalence**: balanced for all $n \neq 1000$; for $n = 1000$ both 10%
    ($n_{pos}=100$) and 50% ($n_{pos}=500$).
-   **Confidence Levels**: $\alpha \in \{0.5, 0.05\}$.
-   **Simulation repeats**: $N_{sim} = 1$ per LHS combination.
-   **Bootstrap Replicates**: $B = 4000$ (shared by all bootstrap-based methods).
-   **Eval grid**: $K = n_0 + 1$ uniform FPR points (exact for step-function bands).
-   **Reproducibility**: a single seed drives a hierarchy of numpy generators;
    torch's RNG is re-seeded from the numpy stream per simulation. A run-level
    metadata JSON records the CLI args, git hash, package versions, and device.

## 5. Method Roster

### Baselines
| Name | Description |
|---|---|
| `ks` | Fixed-width simultaneous KS band (Campbell 1994) |
| `working_hotelling` | Parametric binormal simultaneous band |
| `pointwise` | Pointwise bootstrap percentile band (uncorrected) |
| `pointwise_sidak` | Pointwise bootstrap band, Šidák-corrected across interior grid points |
| `wilson_rectangle_sidak` | 2D Wilson rectangles, Šidák across each rectangle's two margins |
| `wilson_rectangle_bonferroni` | 2D Wilson rectangles, Bonferroni across margins |

### Envelope family (computed jointly by `envelope_band_suite`)
| Name | Configuration |
|---|---|
| `envelope` | **Final method**: Wilson variance floor during studentization + variance-ratio gated Wilson Rectangle floor + exact Beta order-statistic floor (lower band, low FPR). Probability space, KS retention, empirical TPR. |
| `envelope_no_beta_floor` | Ablation: final method without the Beta floor |
| `envelope_no_wilson_floor` | Ablation: raw bootstrap variance, no rectangle; Beta floor only |
| `envelope_no_bootstrap` | Ablation: no bootstrap. Wilson rectangles Šidák-corrected across all interior grid points, plus the Beta floor extended to all $n_0$ order statistics (per-event level $\alpha/(2n_0)$). Implemented as `wilson_beta_band`. |
| `envelope_beta_both_tails` | Symmetric-tail variant: no Wilson machinery; Beta floor at low FPR plus its mirror at high FPR (positive-class order statistics: true TPR at the $j$-th smallest positive $\sim$ exceeds $1-\rho_j$ with $\rho_j$ the Beta$(j, n_1{+}1{-}j)$ upper quantile, anchored at a one-sided Wilson upper bound on its FPR) |
| `envelope_wilson_both_tails` | Symmetric-tail variant: Wilson machinery only, with the rectangle floor forced onto both FPR-tail jurisdictions (matching the Beta floor's reach $q_J$) in addition to the variance-ratio gate; no Beta floor |
| `envelope_no_floors` | Bare studentized envelope (completes the component build-up) |

Dropped from earlier runs: all `harrell_davis`, `logit`, and `symmetric`-retention
variants; Hsieh–Turnbull and ellipse-envelope methods; uncorrected Wilson rectangle;
Wilson pointwise band; logit max-modulus bands.

## 6. Evaluation Metrics

Per band (`src/studroc_paper/eval/eval.py`), saved long-format per CI to feather:

-   Identifiers: DGP, n0/n1/n_total, prevalence, alpha, LHS parameters, DGP
    parameters, lhs_idx/sim_idx.
-   AUC: `true_auc` (dense-grid quadrature of the analytic ROC, corner-refined),
    `empirical_auc`.
-   Coverage: `covers_entirely`, directional violations, violations by FPR region.
-   Violation magnitude: max and **integrated area** above/below; proportion of
    grid points violated; direction-specific min/max violation FPR.
-   Tightness: band area, mean width, width at landmark FPRs
    (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9), mean width per FPR region.
-   Cost: per-method runtime (`runtime_seconds`, with `runtime_is_shared=True`
    for envelope-suite variants whose computation is amortized) and bootstrap
    generation time.

Aggregates per (method, alpha) are saved as JSON: coverage rate with Wilson CI,
directional violation rates and symmetry test, width/area summaries by region,
mean violation areas, and run metadata.

## 7. Order of Operations

1.  Loop over DGPs → sample-size configs → LHS combinations → repeats.
2.  Per repeat: generate data once; generate one $B \times K$ bootstrap matrix;
    all bootstrap-based methods consume it. Envelope-family variants share
    studentization work across variants and alpha levels via
    `envelope_band_suite`. The true ROC is computed once per
    (LHS combination, sample size), not per repeat.
3.  After each (DGP, n0/n1) config: save the long-format feather and the
    aggregated JSON (filenames include DGP, n, prevalence, and date).

Progress bars over LHS combinations; intermediates in RAM.

### Directory Structure
-   `scripts/run_simulation.py`: Main driver script.
-   `data/results/`: Output directory.
