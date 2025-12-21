# Simulation Specification: Confidence Band Evaluation

This document specifies the simulation approach to compare `envelope_boot.py` (Non-Parametric Envelope) and `working_hotelling.py` (Parametric Binormal) across various Data Generating Processes (DGPs).

## 1. Objectives

-   Evaluate the coverage and tightness of confidence bands under diverse data conditions.
-   Assess the robustness of the parametric method (Working-Hotelling) when binormal assumptions are violated.
-   Assess the performance of the non-parametric method (Envelope Bootstrap) across different sample sizes and distributions.

## 2. Data Generating Processes (DGPs)

We will use the DGPs defined in `src/datagen/true_rocs.py`. For each DGP, we define a "Sampling Space" of key properties (e.g., AUC, Shape) which will be sampled using Latin Hypercube Sampling (LHS). These properties are then mapped back to the specific DGP parameters.

### Linking Function: AUC to Separation
For many DGPs, we vary a "Shape" parameter and an "AUC" parameter. Mapping between these ROC properties and DGP parameters is handled by `src/datagen/roc_to_dgp.py`.

### DGP List and Sampling Spaces

1.  **Lognormal (Skewed)**
    -   **DGP**: `make_lognormal_dgp(neg_mu=0, pos_mu=?, sigma=?)`
    -   **Sampling Space**:
        -   `AUC`: $[0.55, 0.99]$
        -   `sigma`: $[0.1, 3.0]$ (Controls skewness; higher = more skewed)

2.  **Heteroskedastic Gaussian (Binormal)**
    -   **DGP**: `make_heteroskedastic_gaussian_dgp(delta_mu=?, sigma_neg=1, sigma_pos=?)`
    -   **Sampling Space**:
        -   `AUC`: $[0.55, 0.99]$
        -   `sigma_ratio` ($ \sigma_{pos} / \sigma_{neg} $): $[0.2, 5.0]$

3.  **Beta (Opposing Skew)**
    -   **DGP**: `make_beta_opposing_skew_dgp(alpha=?, beta=?)`
    -   **Sampling Space**:
        -   `AUC`: $[0.55, 0.99]$
        -   `alpha`: $[0.5, 10.0]$ (Controls shape/kurtosis)
    -   *Note*: `alpha=beta` implies AUC=0.5. We assume `beta > alpha` for AUC > 0.5.

4.  **Student's t (Heavy Tails)**
    -   **DGP**: `make_student_t_dgp(df=?, delta_loc=?, scale=1)`
    -   **Sampling Space**:
        -   `AUC`: $[0.55, 0.99]$
        -   `df`: $[1.1, 30.0]$ (Controls tail heaviness; lower = heavier)

5.  **Bimodal Negative (Mixture)**
    -   **DGP**: `make_bimodal_negative_dgp`
        -   Neg: $w \cdot N(0, 1) + (1-w) \cdot N(\Delta_{neg}, 1)$
        -   Pos: $N(\Delta_{pos}, 1)$
    -   **Sampling Space**:
        -   `AUC`: $[0.55, 0.99]$
        -   `mixture_weight` ($w$): $[0.1, 0.9]$
        -   `mode_separation` ($\Delta_{neg}$): $[0.1, 4.0]$

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
-   **K (eval grid size)**: $K = n0+1$ (1 + number of negative samples).

## 5. Evaluation Metrics

Produce evaluations of each CI using `src/eval/eval.py`. Add identifiers for e.g. DGP, n0/n1 and n_total, confidence level, DGP method, LHS parameters, corresponding DGP parameters, simulation repeat number, and empirical AUC.


## 6. Order of Operations

A complete design for each DGP is specified by a set of LHS-derived parameter combinations, the n0/n1 combinations, the nuber of simulation repeats (n_sim), and the confidence levels for evaluation. Simulation order should be:
1.  Loop over each DGP
2.  Loop over each n0/n1 combination
3.  Loop over each LHS parameter combination
4.  Loop over each simulation repeat
5.  Generate the data and bootstrap samples. Loop over methods, providing these data.
6.  For each confidence level for each method, collect the evaluation metrics.
7.  After each n0/n1 combination, save the individual evaluation metrics from each CI as a table (e.g. parquet), in long format. Save the aggregated metrics as a json keyed by method, separately for each confidence level. Have a special metadata key detailing the DGP, n0/n1, and confidence level, and other run information for attributes common to each CI. The filenames should include DGP, total n, and the date.

There should be a progress bar for LHS combinations.

In-memory intermediate results should be stored in RAM, not the GPU.

### Directory Structure
-   `stats/run_simulation.py`: Main driver script.
-   `stats/config.py`: Configuration definitions (ranges, DGPs).
-   `data/results/`: Output directory.
