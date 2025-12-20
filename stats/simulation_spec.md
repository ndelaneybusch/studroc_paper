# Simulation Specification: Confidence Band Evaluation

This document specifies the simulation approach to compare `envelope_boot.py` (Non-Parametric Envelope) and `working_hotelling.py` (Parametric Binormal) across various Data Generating Processes (DGPs).

## 1. Objectives

-   Evaluate the coverage and tightness of confidence bands under diverse data conditions.
-   Assess the robustness of the parametric method (Working-Hotelling) when binormal assumptions are violated.
-   Assess the performance of the non-parametric method (Envelope Bootstrap) across different sample sizes and distributions.

## 2. Data Generating Processes (DGPs)

We will use the DGPs defined in `src/datagen/true_rocs.py`. For each DGP, we define a "Sampling Space" of key properties (e.g., AUC, Shape) which will be sampled using Latin Hypercube Sampling (LHS). These properties are then mapped back to the specific DGP parameters.

### Linking Function: AUC to Separation
For many DGPs, we vary a "Shape" parameter and an "AUC" parameter. To achieve a target AUC given a shape, we numerically solve for the "Separation" parameter (e.g., `delta_mu`, `pos_mu`, `beta`) using `scipy.optimize.brentq`.

### DGP List and Sampling Spaces

1.  **Lognormal (Skewed)**
    -   **DGP**: `make_lognormal_dgp(neg_mu=0, pos_mu=?, sigma=?)`
    -   **Sampling Space**:
        -   `AUC`: $[0.55, 0.99]$
        -   `sigma`: $[0.1, 3.0]$ (Controls skewness; higher = more skewed)
    -   **Mapping**: Given `sigma` and target `AUC`, solve for `pos_mu`.

2.  **Heteroskedastic Gaussian (Binormal)**
    -   **DGP**: `make_heteroskedastic_gaussian_dgp(delta_mu=?, sigma_neg=1, sigma_pos=?)`
    -   **Sampling Space**:
        -   `AUC`: $[0.55, 0.99]$
        -   `sigma_ratio` ($ \sigma_{pos} / \sigma_{neg} $): $[0.2, 5.0]$
    -   **Mapping**: Set `sigma_pos = sigma_ratio`. Solve for `delta_mu` to match `AUC`.

3.  **Beta (Opposing Skew)**
    -   **DGP**: `make_beta_opposing_skew_dgp(alpha=?, beta=?)`
    -   **Sampling Space**:
        -   `AUC`: $[0.55, 0.99]$
        -   `alpha`: $[1.5, 10.0]$ (Controls shape/kurtosis)
    -   **Mapping**: Given `alpha`, solve for `beta` to match `AUC`.
    -   *Note*: `alpha=beta` implies AUC=0.5. We assume `beta > alpha` for AUC > 0.5.

4.  **Student's t (Heavy Tails)**
    -   **DGP**: `make_student_t_dgp(df=?, delta_loc=?, scale=1)`
    -   **Sampling Space**:
        -   `AUC`: $[0.55, 0.99]$
        -   `df`: $[1.1, 30.0]$ (Controls tail heaviness; lower = heavier)
    -   **Mapping**: Given `df`, solve for `delta_loc` to match `AUC`.

5.  **Bimodal Negative (Mixture)**
    -   **DGP**: `make_bimodal_negative_dgp`
        -   Neg: $w \cdot N(0, 1) + (1-w) \cdot N(\Delta_{neg}, 1)$
        -   Pos: $N(\Delta_{pos}, 1)$
    -   **Sampling Space**:
        -   `AUC`: $[0.55, 0.99]$
        -   `mixture_weight` ($w$): $[0.1, 0.9]$
        -   `mode_separation` ($\Delta_{neg}$): $[1.0, 4.0]$
    -   **Mapping**: Fix `neg_means=[0, mode_separation]`, `neg_weights=[w, 1-w]`. Solve for `pos_mean` ($\Delta_{pos}$) to match `AUC`.

## 3. Sampling Strategy (LHS)

For each DGP:
1.  Define the bounds for the sampling space (as above).
2.  Use `src/sampling/lhs.py` (`maximin_lhs`) to generate $N_{LHS} = 1000$ parameter combinations.
3.  Map each combination to the actual DGP parameters.

## 4. Simulation Parameters

For each parameter combination (DGP instance), we run simulations across a range of sample sizes.

-   **Sample Sizes ($n$)**: $\{10, 30, 100, 300, 1000, 10000\}$
-   **Prevalence**:
    -   For all $n \neq 1000$: Balanced ($n_{pos} = n_{neg} = n/2$).
    -   For $n = 1000$: Run three prevalence scenarios:
        1.  1% ($n_{pos}=10, n_{neg}=990$)
        2.  10% ($n_{pos}=100, n_{neg}=900$)
        3.  50% ($n_{pos}=500, n_{neg}=500$)
-   **Confidence Levels**: $\alpha \in \{0.5, 0.05\}$ (50% and 95% confidence).
-   **Number of Simulations**: $N_{sim} = 1$ per configuration.
-   **Bootstrap Replicates**: $B = 4000$ (for Envelope method).

## 5. Evaluation Metrics

Using `src/eval/eval.py`, we collect:
-   **Coverage Rate**: Proportion of bands containing the true ROC entirely.
-   **Mean Band Area**: Average area between upper and lower bands (tightness).
-   **Mean Band Width**: Average width across the FPR grid.
-   **Violation Rates**: Frequency of true ROC exiting the band (above/below).

### Directory Structure
-   `stats/run_simulation.py`: Main driver script.
-   `stats/config.py`: Configuration definitions (ranges, DGPs).
-   `data/results/`: Output directory.
