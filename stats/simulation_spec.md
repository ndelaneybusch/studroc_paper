# Simulation Specification: Confidence Band Evaluation (Final Run)

This document specifies the final paper run comparing the rank-space fiducial
band and its M3-floored hybrid against the exact M3 band, the studentized
bootstrap envelope, and classical baselines across diverse Data Generating
Processes (DGPs).

## 1. Objectives

-   Evaluate coverage and tightness of confidence bands under diverse data conditions.
-   Measure what the localized M3 floor buys the C = 1 fiducial band, paired on
    one fiducial cloud per replicate.
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

-   **Sample Sizes ($n$)**: $\{10, 30, 100, 300, 1000, 3000, 10000\}$
-   **Prevalence**: balanced for all $n \neq 1000$; for $n = 1000$ both 10%
    ($n_{pos}=100$) and 50% ($n_{pos}=500$).
-   **Confidence Levels**: $\alpha \in \{0.5, 0.2, 0.05\}$.
-   **Simulation repeats**: $N_{sim} = 1$ per LHS combination.
-   **Bootstrap Replicates**: $B = 4000$ (envelope only).
-   **Eval grid**: $K = n_0 + 1$ uniform FPR points (exact for step-function bands).
    The rank-space methods return this native grid directly, without resampling.
-   **Reproducibility**: a single seed drives a hierarchy of numpy generators.
    Each (DGP, sample size) cell has a root seed, and each LHS combination draws
    from its own `SeedSequence.spawn` child, so any process can regenerate any
    combination; within it, each repeat draws its data, a rank-space method seed,
    and the seed of the bootstrap's own torch generator. Results are therefore
    independent of worker count and scheduling. A run-level metadata JSON records
    the CLI args, git hash, package versions, hybrid floor settings, and GPU.

## 5. Method Roster

| Name | Description |
|---|---|
| `ks` | Fixed-width simultaneous KS band (Campbell 1994) |
| `working_hotelling` | Parametric binormal simultaneous band |
| `envelope` | Studentized bootstrap envelope: Wilson variance floor during studentization + variance-ratio gated Wilson Rectangle floor + exact Beta order-statistic floor (lower band, low FPR). Probability space, KS retention, empirical TPR. |
| `m3` | Exact composition band (finite-sample coverage theorem), sample-size class split |
| `fiducial` | Rank-space fiducial band at C = 1, no floor |
| `fiducial_hybrid` | C = 1 fiducial band with the localized exact M3 floor (`M3Floor()` defaults: exact region rule, tails-recalibrated trim, tightened closure, sample-size split; see `stats/experiments/closure_split_report.md`) |

`fiducial` and `fiducial_hybrid` are computed together by
`fiducial_band_suite_rs` from one fiducial cloud per replicate, sized for the
largest per-level draw budget; each band is bit-identical to a separate
`fiducial_band_rs` call with the same seed.

Dropped from earlier runs: the pointwise bootstrap and Wilson rectangle
baselines; the envelope ablations and symmetric-tail variants; all
`harrell_davis`, `logit`, and `symmetric`-retention variants; Hsieh–Turnbull and
ellipse-envelope methods; Wilson pointwise band; logit max-modulus bands.

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
    for the two fiducial arms, whose single suite call is amortized), the
    `device` it ran on, and bootstrap generation time. Runtimes are measured
    under the run's full concurrent load (see §7), so they rank methods within
    a run but are not single-method benchmarks.

Aggregates per (method, alpha) are saved as JSON: coverage rate with Wilson CI,
directional violation rates and symmetry test, width/area summaries by region,
mean violation areas, and run metadata.

## 7. Order of Operations

1.  Draw every DGP's LHS design, then calibrate M3 once for every
    (class sizes, alpha) in the run, in parallel; the calibration is handed to
    every worker.
2.  Submit one task per LHS combination to a pool of single-threaded CPU
    worker processes (GPU hidden), in a fixed order that runs across cell
    boundaries so the pool never drains between cells. A worker regenerates
    the combination's data from its seed, computes the true ROC (once per
    combination) and every rank-space band, and evaluates them.
3.  The envelope runs where it is cheapest: for cells with at least
    `--gpu-min-neg` negatives, the main process regenerates the same data and
    computes the bootstrap matrix and envelope on the GPU while the workers
    run; below that, GPU launch overhead dominates and the workers compute it
    on the CPU. The two lanes' copies of each dataset are checked against each
    other (empirical AUCs) before their results are merged.
4.  When a (DGP, n0/n1) cell's last result arrives: save the long-format
    feather and the aggregated JSON (filenames include DGP, n, prevalence,
    and date).

### Directory Structure
-   `scripts/run_simulation.py`: Main driver script.
-   `data/results/`: Output directory.
