# State-of-the-Art Demonstration Plan

**Status:** draft for review. Nothing in §§3–5 has been run or implemented except the pilot in §4.3.

**Scope.** The demonstration has three phases, one per band, because the three bands are valid in different senses and need different comparison classes.

| Phase | Band | Comparison class | Status in this document |
|---|---|---|---|
| A | ELL composition ([`ell_band.md`](ell_band.md)) | Bands with a finite-sample distribution-free coverage theorem | Full plan, §§1–5 |
| B | Fiducial ([`fiducial_band.md`](fiducial_band.md)) | Nonparametric bands with asymptotic or empirical validity | Outline, §6 |
| C | Hybrid and auto dispatcher ([`hybrid_band.md`](hybrid_band.md)) | All bands that pass an empirical validity gate | Outline, §6 |

Sources for the competitor inventory are in [`literature_references.md`](literature_references.md) (cited as "refs"), especially §§A.14–A.20, B.12–B.14, D, and G.

**What reviewers are asked to check.**
1. Is the class in §1.1 the right one, and is anything in §2.3 excluded that should compete?
2. Are the endpoints and decision rule in §1.3 a fair definition of "state of the art"?
3. Are the comparisons in §4.1 the claims the paper should make, and are the margins sensible?
4. Is the design in §3 adequate for those claims without waste?
5. Are the validation gates in §5.4 enough to trust a Rust reimplementation of other authors' bands?

---

## 1. The Class and the Claim

### 1.1 Class definition

Data, assumptions A1–A4, and the estimand $`R(t) = 1 - G(Q_F(1-t))`$ are those of [`ell_band.md` §1](ell_band.md#1-setup-and-assumptions).

A procedure belongs to the **exact class** $`\mathcal{C}`$ if it maps the two samples and a level $`\alpha`$ to a band $`[L(t), U(t)]`$, $`t \in [0,1]`$, in (FPR, TPR) coordinates, and there is a proof that

```math
P\big(\forall t \in [0,1]: L(t) \le R(t) \le U(t)\big) \ge 1 - \alpha
```

for every pair of continuous $`F`$, $`G`$ and every $`n_0, n_1 \ge 1`$. Four consequences of this definition decide membership:

- **Simultaneous.** Exact bounds at one threshold or one operating point are not members.
- **ROC coordinates.** Exact bands for the Q-Q function on the score scale are not members (refs A.14).
- **Unrestricted.** A guarantee that needs smoothness, a parametric model, or a restricted class of rejection regions is not a member.
- **Proved.** Membership is established by proof. Simulation checks the implementation; it cannot admit a method.

### 1.2 How members are built

Every member known to us is a **composition**: one simultaneous band for each class's survival function at its own order statistics, composed into an ROC band through monotonicity and the merged ranks ([`ell_band.md` §2.6](ell_band.md#26-composition), Lemma 1). A composition band is fixed by three choices:

| Choice | Options in this plan |
|---|---|
| Boundary family: the shape of the class bounds $`b^-_i(\theta) \le U_{(i)} \le b^+_i(\theta)`$ | Constant margin (KS), Beta quantiles at equal local levels (ELL), and the alternatives of §2.2 |
| Calibration: how $`\theta`$ is set for class level $`\alpha_c`$ | An analytic bound (Massart), or the exact non-crossing probability |
| Class split: how $`\alpha`$ divides into $`\alpha_0`$, $`\alpha_1`$ | Equal (Šidák) or the sample-size rule of [`ell_band.md` §2.5](ell_band.md#25-class-split) |

Lemma 1 and Theorem 1 of `ell_band.md` use only that the class events hold with probability at least $`1-\alpha_c`$ and that the bound sequences are nondecreasing. They therefore cover every row of the method matrix in §3.1 without a new proof. This is what makes the class small enough to compare exhaustively: one composition routine and one calibration recursion implement all members.

### 1.3 Operationalizing "state of the art"

Validity is settled by the theorem, so the claim is about efficiency.

**Endpoints.** All are computed on the continuum band, which is a pair of step functions with known breakpoints, so areas are exact and involve no grid.

| Endpoint | Definition | Role |
|---|---|---|
| Area | $`A = \int_0^1 (U - L)\,dt`$ | Primary |
| Low-FPR area | $`A_{\text{low}} = \int_0^{0.1} (U - L)\,dt`$ | Primary |
| Mid-curve area | $`\int_{0.25}^{0.75} (U - L)\,dt`$ | Secondary; ELL is expected to lose here (H8) |
| Widths at FPR 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9 | $`U(t) - L(t)`$ | Descriptive |
| Dominance fraction | Share of datasets on which one band has the smaller area | Descriptive |
| Realized coverage | Strict whole-curve coverage, endpoints included, zero tolerance | Implementation diagnostic (§4.5) and the frontier below |
| Gap to the lower bound | Area on the separated path against $`1 - \alpha^{1/n_0 + 1/n_1}`$ ([`ell_band.md` §3.3](ell_band.md#33-width-and-tail-adaptivity)) | Descriptive |

The existing tables use trapezoidal area on the native grid. That metric stays as a secondary column for continuity. It is not suitable as a primary endpoint for this comparison: on the native grid a constant horizontal margin is rounded to a whole number of grid steps, which changed the pilot's exact-versus-Massart KS ratio by up to 2.8% where the margins differ by 1% (§4.3). For bands that exist only on a grid (the fiducial and hybrid bands of Phases B and C), the area of the band as claimed is that of the conservative step extension, $`\sum_k (t_{k+1} - t_k)\{U(t_{k+1}) - L(t_k)\}`$.

**Estimand and comparison scale.** The scientific quantity is expected area, which is also what the optimized comparators of §2.2 target. For an endpoint $`E`$, a comparator $`c`$, and a cell:

- **Primary:** the mean paired difference $`\Delta = \mathbb{E}[E_{\text{ELL}} - E_c]`$, reported with the **ratio of mean areas** $`\rho = \mathbb{E}[E_{\text{ELL}}] / \mathbb{E}[E_c]`$. Margins are stated on $`\rho`$.
- **Secondary:** the geometric mean of the dataset-level ratios, $`\exp \mathbb{E}[\log(E_{\text{ELL}}/E_c)]`$. It describes typical relative performance across datasets, and can rank methods differently from $`\rho`$ when wide and narrow bands occur in the same cell.

**Four outcomes.** Each confirmatory comparison is classified from its simultaneous confidence interval for $`\rho`$ (§4.2) and a declared margin $`m`$ (2% unless stated):

| Outcome | Interval for $`\rho`$ |
|---|---|
| ELL superior | Entirely below 1 |
| ELL non-inferior | Entirely below $`1 + m`$, but not entirely below 1 |
| ELL inferior | Entirely above 1 |
| Inconclusive | Anything else |

A failure to show non-inferiority is inconclusive, not evidence that the comparator is better. An absence of significant differences is likewise inconclusive, not evidence that ELL is competitive.

**Decision rule.** The claim is about the methods in the matrix of §3.1 and the design distribution of §3.2–3.3. It is not a claim about every exact band, or even every rectangular composition; the matrix does not exhaust either. Within that scope, ELL (sample-size split) is called state of the art on a design region $`\mathcal{D}`$ if all three hold:

1. *Superiority to the published baseline.* ELL is superior to the exact KS band (`ks_x_ss`) on both primary endpoints in every cell of $`\mathcal{D}`$.
2. *Not beaten in any cell.* Against every other member, ELL is superior or non-inferior on both primary endpoints in every cell of $`\mathcal{D}`$. One inferior result on either endpoint fails the rule; an inconclusive result fails it too and is reported as such.
3. *Not beaten overall.* The same, for the design-average comparison.

If a member is superior to ELL on both primary endpoints over $`\mathcal{D}`$, the paper reports that member as the best evaluated. The contribution then shifts from "ELL" to "exactly calibrated tail-sensitive composition", or to the ROC-optimized composition if that is the member. This outcome is pre-committed, not treated as a failed study. Mixed results (ELL better on one endpoint, a comparator on the other) are reported as a trade-off, with no single winner named.

**The region $`\mathcal{D}`$.** The pilot shows ELL and exact KS within 1% of each other when the smaller class has about 10 or fewer observations. We split the design in advance:

- **Core region:** $`\min(n_0, n_1) \ge 30`$. Superiority to KS is claimed here.
- **Small-sample region:** $`\min(n_0, n_1) < 30`$. Only non-inferiority to KS is claimed (H3).

**Coverage–area frontier.** For each member we plot realized worst-cell coverage against mean area at $`\alpha \in \{.05, .1, .2, .5\}`$. All members are valid at every level, so the frontier is descriptive here. It becomes the main cross-class comparison in Phases B and C, and Phase A supplies its exact-class points.

---

## 2. Methods in the Class

### 2.1 Ours

**ELL composition** (`m3`). Beta-quantile class bounds at one local level $`\gamma`$ per class, calibrated by the exact non-crossing probability, with the sample-size split. Specification and proofs: [`ell_band.md`](ell_band.md). Theory: [`fiducial_band_theory.md`](fiducial_band_theory.md) §§9 and 12. An equal-split variant is run as an ablation.

### 2.2 Competitors, published or constructible

| ID | Boundary family | Calibration | Source | Why it is in the comparison |
|---|---|---|---|---|
| KS-M | Constant margin | Massart bound $`\sqrt{\log(2/\alpha_c)/(2n)}`$ | Campbell 1994; refs A.2, B.5 | The published band and the current `ks` baseline |
| KS-X | Constant margin | Exact Kolmogorov critical value | Same band, exact level | Removes the slack in Massart's constant so that KS is not handicapped |
| BONF | Beta quantiles at $`\gamma = \alpha_c/(2n)`$ | Union bound | Simultaneous extension of Hilgers 1991 and of NP-ROC (refs A.12, A.18) | What a Bonferroni grid of Clopper–Pearson bounds gives; isolates the value of exact calibration |
| BJ | Likelihood-ratio (Berk–Jones) bounds | Exact | Owen 1995; refs B.12 | The original tail-sensitive band |
| DW | Dümbgen–Wellner bounds | Exact | Dümbgen & Wellner 2023; refs B.12 | Designed to improve Berk–Jones and Owen in the centre |
| EK | Studentized margin $`c\sqrt{u(1-u)}`$ | Exact | Eicker-type weighting; the Doksum–Sievers W-band weight (refs A.14) | Classical variance weighting, the natural midpoint between KS and ELL |
| FREY | Minimum total width, uniform weights | Exact | Frey 2008; refs B.12 | Frey's published criterion; narrower than KS in rank space |
| ROPT | Bounds minimizing expected ROC-band area under a mixture of reference curves | Exact | Ours, built on Frey's convexity result; [`frey_ell_plan.md`](frey_ell_plan.md) | Our own probe shows it 5–14% narrower than ELL at AUC 0.8–0.95, so it cannot be left out |

Each family runs with the sample-size split. KS and ELL also run with the equal split, so the effect of the split is separated from the effect of the family.

**Frey's band.** The method is recovered from Frey's preprint and thesis, and our probe reproduces his $`n = 20`$ band from the exact kernel (refs B.12; `frey_ell_plan.md` §3.1). It is implemented as specified there, with his first-order conditions as the optimality check.

**The ROC-optimized comparator.** ROPT is trained once per $`(n_0, n_1, \alpha)`$ on a reference library that shares no family and no random draw with the evaluation data, and is then a fixed member like any other. It is run at the designs where full optimization is affordable (totals up to 300). A band optimized separately for each evaluation dataset's true ROC is not a method: it is computed as an **oracle benchmark**, reported descriptively as the best any rectangular composition could do with knowledge of the truth, and excluded from the decision rule.

**Gate on DW.** The exact form of the bound (the additive correction terms and their constants) is taken from the paper, not from memory, and checked against the critical values tabulated in its supplement.

### 2.3 Excluded, with reasons

| Method | Reason | Reference |
|---|---|---|
| Switzer's Smirnov-inversion band; Doksum–Sievers S- and W-bands | Exact, but for the Q-Q function on the score scale. They give no band in ROC coordinates. Our reading of this point must be confirmed against Switzer's text before the paper asserts it | refs A.14 |
| Hilgers 1991; Schäfer 1994; Tilbury et al. 2000; Clopper–Pearson rectangles; NP-ROC | Pointwise. Their simultaneous extension by a union bound is BONF | refs A.12, A.18 |
| Time-uniform DKW (Howard & Ramdas 2022) | A member, but its margin is about twice the fixed-$`n`$ margin, so KS-M dominates it by construction. Stated, not run | refs B.12 |
| Bates et al. 2023 | Uniform finite-sample bound for one class only | refs A.18 |
| Shi, Wang & Ma 2026 | **Not yet classified; pending a full read.** Their target appears to be the optimal (Neyman–Pearson) trade-off frontier between two distributions, which is a different estimand from the ROC of a fixed score, and their finite-VC condition concerns attaining that frontier. Citing the VC restriction is therefore not, by itself, a reason to exclude them from a fixed-score comparison. If the paper yields a finite-sample band for a fixed score's ROC, it joins the matrix | refs A.19 |
| Goldman & Kaplan 2018 | Two-sample Beta-quantile bands calibrated under $`F = G`$ for testing; the authors state they lack coverage | refs A.20 |
| Asymptotic, bootstrap, Bayesian, and parametric bands | No finite-sample theorem. They are the Phase B and C comparison set | refs A.3–A.11, A.15–A.17 |

---

## 3. Simulation Specification

### 3.1 Method matrix

| ID | Family | Calibration | Split | New code? |
|---|---|---|---|---|
| `ks_m_eq` | KS | Massart | Equal | Port of the current `ks` |
| `ks_m_ss` | KS | Massart | Sample-size | New |
| `ks_x_eq` | KS | Exact | Equal | New |
| `ks_x_ss` | KS | Exact | Sample-size | New. **Reference comparator for H1–H3** |
| `ell_eq` | ELL | Exact | Equal | Existing option |
| `ell_ss` | ELL | Exact | Sample-size | Existing `m3`. **The candidate** |
| `bonf_ss` | ELL bounds | Union bound | Sample-size | New |
| `bj_ss` | Berk–Jones | Exact | Sample-size | New |
| `dw_ss` | Dümbgen–Wellner | Exact | Sample-size | New |
| `ek_ss` | Eicker | Exact | Sample-size | New |
| `frey_ss` | Frey, uniform weights | Exact | Sample-size | New |
| `ropt_ss` | ROC-area-optimized, reference mixture | Exact | Part of the solution | New; designs with total up to 300 |
| `ropt_oracle` | ROC-area-optimized for each dataset's true ROC | Exact | Part of the solution | New; oracle benchmark, not a competitor |

Levels: $`\alpha \in \{.05, .1, .2, .5\}`$. The 90% level is added because Macskassy et al. and Sewak & Hothorn report at it.

All bands are computed from one tie-resolved label sequence per dataset, in one call.

### 3.2 Data-generating processes

Every member reads only the ranks, so its coverage and width depend on the data only through $`(R, n_0, n_1)`$. Families matter only as a source of ROC shapes. Three strata:

**Stratum S (smooth).** The seven families and parameter ranges of [`simulation_spec.md`](simulation_spec.md) §2, sampled afresh for this study: within each family and AUC stratum (§3.3), the AUC is drawn uniformly on the probit scale and each shape parameter uniformly over its range, independently for every dataset. The existing Latin-hypercube designs and seeds are **not** reused, for two reasons. Latin-hypercube points are not independent draws, so cell means from them do not have the simple sampling distribution the tests of §4.2 assume. And the existing datasets have already been examined while choosing methods, regions, and margins, which weakens any confirmatory reading of them.

Pairing is preserved inside the new run instead: every Phase A method sees every dataset, and the expensive methods of Phases B and C are run on a prespecified subsample of the same datasets (the first $`N_B`$ of each cell, with $`N_B`$ set from the coverage table of §4.4).

**Stratum H (home turf of other papers).** The DGPs a referee will look for (refs A.10, A.17): the twelve control and case pairs of Martínez-Camblor et al. (normal, lognormal, gamma, and normal-mixture), Demidenko's binormal grid, and the probit, logit, and cloglog transformation families of Sewak & Hothorn. These are fixed curves, not LHS designs. They matter mainly in Phases B and C; they run in Phase A so that the exact-class columns exist for every later table.

**Stratum X (adversarial).** Fixed curves on which an exact band must still cover:

| Curve | What it stresses |
|---|---|
| Perfect separation | $`U = 1`$ everywhere; the lower-bound gap |
| Corner slivers, constructions (a) and (b) of [`fiducial_band.md` Prop. 2](fiducial_band.md#31-coverage-guarantees) | Unobserved mass at the $`1/n`$ scale |
| Interior sliver; step ROC with a jump at FPR 0.5; kink | Jumps in $`R`$ away from the corners |
| Partial separation with $`R(0) > 0`$ | The unpinned $`U(0)`$ |
| Diagonal, AUC 0.5 | Where ELL's advantage should be smallest |
| AUC 0.999 | Near-separation |
| Binormal scores rounded to 20 levels | Ties under randomized tie-breaking, with the randomized-ROC estimand |

### 3.3 Sample sizes, prevalences, and cells

| Total $`n`$ | 50% | 30% | 10% | 2% |
|---|---|---|---|---|
| 10 | ✓ | | | |
| 30 | ✓ | ✓ | ✓ | |
| 100 | ✓ | ✓ | ✓ | |
| 300 | ✓ | ✓ | ✓ | |
| 1,000 | ✓ | ✓ | ✓ | ✓ |
| 3,000 | ✓ | ✓ | ✓ | ✓ |
| 10,000 | ✓ | ✓ | ✓ | ✓ |

That is 22 **designs** (a total size at a prevalence). The 2% column reflects rare-positive evaluation in machine learning; it gives 20, 60, and 200 positives.

**AUC strata.** Four strata with equal probability under probit-uniform sampling on [0.55, 0.99]: AUC 0.550–0.750, 0.750–0.890, 0.890–0.962, and 0.962–0.990. Each receives the same number of datasets.

**Analysis cell.** A cell is a design crossed with an AUC stratum, pooled over families with equal allocation. Stratifying is needed because the ELL-to-KS ratio depends strongly on AUC: in the pilot the standard deviation of the log ratio within a design was up to 0.35, almost all of it variation of the mean with AUC. That gives 88 cells in stratum S, of which 64 (16 designs) are in the core region.

Within a cell the datasets are independent and identically distributed draws from the cell's sampling distribution, so the cell estimand is the mean over that distribution and ordinary one-sample inference applies. Family-level results are reported descriptively. They are not tested, because for rank-based methods a family is a set of ROC shapes, not a separate population.

### 3.4 Replication and cost

| Stratum | Datasets | Basis |
|---|---|---|
| S | 1,500 per family per AUC stratum per design: 10,500 per cell, 42,000 per design, 924,000 in all | Fresh independent draws; exceeds every width requirement in §4.4 |
| H | 3,000 per curve per design, at $`n \in \{100, 300, 1{,}000\}`$ and 50% and 10% prevalence | Detects coverage of 0.93 against 0.95 (§4.4) |
| X | 3,000 per curve per design, same sizes as H plus $`n = 30`$ | Same |

Cost is not the binding constraint in Phase A. A calibrated band costs a few milliseconds ([`ell_band.md` §4.1](ell_band.md#41-complexity)), and all bands at four levels share one sort. We estimate a few core-hours for band construction and truth evaluation together. The one-time calibrations are the larger item: about 40 distinct class sizes, four levels, two splits, and six exactly calibrated families, with the largest sizes at about a minute per bisection today. With the warm-start table and the parallel sweep of §5.2 this is under an hour on the 14-core machine. These estimates are to be replaced by measurements from the dry run (§5.5).

Replication is rationed in Phases B and C, where one band costs seconds. §4.4 is written so that the same table sizes those runs.

### 3.5 Evaluation

- **Truth.** The analytic ROC of each DGP, evaluated at the breakpoints of each band, with left and right limits where $`R`$ jumps.
- **Coverage.** Strict: the band covers if $`L(t) \le R(t) \le U(t)`$ at every $`t \in [0,1]`$, endpoints included, with no tolerance beyond a few units of double-precision roundoff in the comparison. The existing $`10^{-6}`$ tolerance and endpoint exclusion are also reported, for continuity with earlier tables.
- **Recorded per band:** the endpoints of §1.3, miss direction, miss location, and the calibrated parameter.
- **Seeds and metadata:** as in `simulation_spec.md` §4, plus the calibration table's version hash.

---

## 4. Pre-Registered Tests

The comparisons, margins, regions, and analysis script are frozen by commit before the main run (§5.5). The pilot of §4.3 was used to set regions and margins. It is not part of the confirmatory data, and neither is any dataset generated before the freeze.

### 4.1 Confirmatory comparisons

Each row is a set of comparisons of ELL (`ell_ss`) with a comparator, classified into the four outcomes of §1.3 from a simultaneous interval for the ratio of mean areas $`\rho`$. "Core" and "small" are the regions of §1.3. All confirmatory comparisons are at the $`\alpha = .05`$ band level; the other levels are reported with the same intervals as secondary and are not part of the family.

| ID | Claim | Comparator | Endpoint | Cells | Margin and expected outcome | Comparisons |
|---|---|---|---|---|---|---|
| H1 | ELL is narrower than exact KS | `ks_x_ss` | Area | 64 core | Superior | 64 |
| H2 | ELL is narrower at low FPR, by a wide margin at scale | `ks_x_ss` | Low-FPR area | 64 core | Superior; in cells with $`n \ge 1{,}000`$ the interval should lie below 0.80 | 64 |
| H3 | ELL is not materially wider in small samples | `ks_x_ss` | Area | 24 small | Non-inferior at margin 3% | 24 |
| H4 | ELL is not beaten by another exactly calibrated family | `bj_ss`, `dw_ss`, `ek_ss`, `frey_ss` | Area and low-FPR area | 64 core | Superior or non-inferior at 2%; any of the four outcomes is reported | 512 |
| H4r | ELL against the ROC-optimized composition | `ropt_ss` | Area and low-FPR area | 20 core cells with total up to 300 | No prediction; the probe suggests ELL inferior on area at AUC of 0.8 or more | 40 |
| H5 | Exact calibration beats the union bound | `bonf_ss` | Area | 64 core | Superior | 64 |
| H6 | The sample-size split helps under imbalance | `ell_eq` | Area | 24 core cells at 10% and 2% prevalence | Superior | 24 |
| H8 | KS is narrower mid-curve | `ks_x_ss` | Mid-curve area | 64 core | Inferior (ELL wider) | 64 |
| HA | Design averages | Each of `ks_x_ss`, `bj_ss`, `dw_ss`, `ek_ss`, `frey_ss`, `bonf_ss` | Area and low-FPR area | Equal-weight mean over core cells | As the matching row | 12 |

The confirmatory family has **868 comparisons**.

Descriptive only: Massart against exact KS; the effect of the split on KS; the oracle benchmark; dominance fractions; geometric-mean ratios; the frontier; the gap to the separated-path lower bound; family-level tables; all results at levels other than .05.

H8 is registered so that the paper reports where ELL loses with the same rigour as where it wins.

### 4.2 Inference

- **Statistic.** Within a cell the datasets are independent draws (§3.3). The ratio of mean areas $`\rho`$ is estimated by the ratio of sample means, with an interval from the paired data by the delta method (equivalently, an interval for the mean of $`E_{\text{ELL}} - \rho_0 E_c`$ inverted in $`\rho_0`$). The mean paired difference is reported with the same critical value.
- **Multiplicity.** One family, the 868 comparisons of §4.1, with Bonferroni-adjusted two-sided intervals at simultaneous level 95%: critical value $`z = 4.02`$. Simultaneous intervals are used instead of a stepwise test procedure because the four-outcome classification of §1.3 needs an interval for every comparison. There is no gatekeeping or testing order: every comparison is classified, and the decision rule of §1.3 is read off the classifications.
- **Design averages.** The equal-weight mean of cell means over the core region, with a variance that is the sum of the independent cell variances.
- **Coverage is not in the family.** It is an implementation diagnostic with its own rule (§4.5).

### 4.3 Pilot

300 datasets per design at $`\alpha = .05`$; heteroscedastic Gaussian scores with probit-uniform AUC in [0.55, 0.99] and log-uniform variance ratio in [0.2, 5]; ELL from the production code and the four KS variants from a reference implementation checked against the current `ks` band. Area is trapezoidal on the native grid, and the summaries below are **geometric means of dataset-level ratios**, which is the secondary scale of §1.3. They are indicative only; the exact-area endpoint and the ratio-of-means scale may each move them by a few percent.

| $`n`$ (prevalence) | ELL ÷ `ks_m_eq` | ELL ÷ `ks_x_ss` | SD of log ratio | Share with ELL narrower than `ks_x_ss` | Low-FPR ratio to `ks_x_ss` |
|---|---|---|---|---|---|
| 10 (50%) | 0.910 | 0.919 | 0.031 | 1.000 | 0.928 |
| 30 (50%) | 0.974 | 0.986 | 0.024 | 0.633 | 0.878 |
| 100 (50%) | 0.846 | 0.852 | 0.108 | 1.000 | 0.878 |
| 300 (50%) | 0.747 | 0.768 | 0.204 | 0.997 | 0.698 |
| 1,000 (50%) | 0.664 | 0.674 | 0.291 | 1.000 | 0.532 |
| 3,000 (50%) | 0.609 | 0.610 | 0.352 | 1.000 | 0.463 |
| 30 (10%) | 0.950 | 1.006 | 0.030 | 0.323 | 0.972 |
| 100 (10%) | 0.944 | 0.996 | 0.023 | 0.440 | 0.920 |
| 300 (10%) | 0.845 | 0.876 | 0.101 | 0.990 | 0.748 |
| 1,000 (10%) | 0.722 | 0.739 | 0.207 | 0.980 | 0.624 |
| 3,000 (10%) | 0.640 | 0.651 | 0.289 | 0.960 | 0.558 |

Three findings shaped the plan:

1. **The exact, sample-size-split KS band is a materially stronger baseline in small or imbalanced samples.** At 27 negatives and 3 positives ELL goes from 5% narrower than the current baseline to 0.6% wider than the fair one. This is why the core region starts at 30 in the smaller class and why H3 is a non-inferiority claim.
2. **The sample-size split helps KS too**, by about 1.5% under imbalance. It helps ELL by 1.3–2.2%.
3. **The log-ratio standard deviation grows with $`n`$** because the ratio depends on AUC. Within-stratum standard deviations will be smaller; the dry run measures them.

### 4.4 Power

Datasets per cell for 90% power at the critical value of §4.2.

**Paired width comparisons.** $`N = \{(4.02 + 1.28)\,s/\delta\}^2`$, where $`\delta`$ is the distance between the true value and the margin and $`s`$ the standard deviation of the paired quantity, both on the same scale. For the primary endpoint that scale is $`E_{\text{ELL}} - \rho_0 E_c`$ divided by the comparator's mean area, so $`\delta`$ is a difference in $`\rho`$. The pilot recorded only the log-ratio scale, so the table is entered with the pilot's log-ratio standard deviations as a guide; the dry run of §5.5 measures $`s`$ on the primary scale and this section is updated before the freeze.

| Relative SD $`s`$ | $`\delta = 0.005`$ | 0.01 | 0.02 | 0.05 |
|---|---|---|---|---|
| 0.025 | 704 | 176 | 44 | 8 |
| 0.05 | 2,814 | 704 | 176 | 29 |
| 0.10 | 11,253 | 2,814 | 704 | 113 |
| 0.20 | 45,012 | 11,253 | 2,814 | 451 |
| 0.30 | 101,276 | 25,319 | 6,330 | 1,013 |

**Coverage diagnostics.** Datasets needed to flag a cell whose true coverage is below nominal, at a per-cell one-sided critical value of 3.54.

| Nominal | True coverage | $`N`$ |
|---|---|---|
| 0.95 | 0.94 | 11,576 |
| 0.95 | 0.93 | 3,017 |
| 0.95 | 0.92 | 1,392 |
| 0.95 | 0.90 | 535 |
| 0.80 | 0.78 | 9,477 |
| 0.80 | 0.77 | 4,249 |
| 0.80 | 0.75 | 1,554 |
| 0.50 | 0.48 | 14,524 |
| 0.50 | 0.47 | 6,452 |
| 0.50 | 0.45 | 2,319 |

**Reading the tables for Phase A.** With 10,500 datasets per cell:

- H1, H2, H5, H6: pilot effects are 2–50% with $`s \le 0.35`$. Power is essentially 1 in the core region.
- H3: a true ratio of 1.006 against a margin of 1.03 is a distance of 0.024 at $`s \approx 0.03`$. About 45 datasets suffice.
- H4 and H4r: the effect sizes are unknown. At $`s = 0.10`$ the design resolves a difference of 0.5%; at $`s = 0.30`$, 1.6%. The 2% non-inferiority margin is therefore resolvable in every cell, so few comparisons should end inconclusive.
- Coverage: a shortfall of one percentage point at the 95% level is flagged with about 86% power, and two points with near certainty.

**Reading the tables for Phases B and C.** Where one band costs seconds, the coverage table sets the floor: about 3,000 datasets per cell to detect a two-point coverage shortfall at 95%, and about 1,400 for three points. Width comparisons need far fewer. Those phases should therefore size $`N_B`$ by the coverage table.

### 4.5 Coverage diagnostics

Coverage is checked for every member, cell, stratum (S, H, X), and level, with strict whole-curve coverage. A cell is flagged if the upper confidence limit for its coverage is below $`1-\alpha`$.

This is a check on the implementation and the proofs, not a hypothesis whose success licenses the width claims. Finding no flag does not establish validity, which rests on the theorem. Finding a flag stops the analysis: it means a bug or a broken proof, and no width result for that member is reported until it is resolved.

---

## 5. Implementation Plan

All band construction moves into the Rust crate (`rust/`, exposed as `fiducial_core`), with thin Python wrappers. Reference implementations in R, Python, and C++ are used only to produce validation fixtures before the run.

### 5.1 Corrections to existing code and documents

| Item | State |
|---|---|
| `ks_band.py` described its margin as a "Smirnov approximation" with a "Bonferroni correction". It is the DKW margin with Massart's constant and an equal Šidák split | Docstring and comments corrected |
| `hsieh_turnbull_band.py` docstring stated three defaults that the code does not have and described bootstrap calibration as the default | Docstring corrected; it now also records the measured failure of the default variance estimate |
| `literature_references.md` | Updated with the November 2026 searches |
| `hsieh_turnbull_band.py` itself: the default band covered 0–37% of binormal truths on FPR [0.05, 0.95] at a nominal 95% | Not fixed. It is replaced in Phase B (§6); it must not be used as a baseline meanwhile |
| The crossing-probability kernel returned 0 for valid bounds with a long gap between breakpoints (2,000 observations, first 960 upper bounds at 0.5: true value 0.965). Each binomial transition started from its zero-success term, which underflowed | Fixed: transitions start from the largest in-window term, computed by a saddle-point formula. ELL and constant-margin values are unchanged to about $`10^{-15}`$; closed-form binomial cases up to 20,000 observations are in the Rust and Python tests. The error was conservative (lost mass only), so earlier ELL calibrations are not invalidated |
| ELL roundoff margin is a fixed $`10^{-9}`$, and the documents describe it as covering accumulated roundoff | To do in §5.2, with the error sources stated separately |
| Coverage evaluation excludes both endpoints and allows $`10^{-6}`$ | Strict mode to be added in §5.3; the old mode stays as a reported column |
| Area ratios in `ell_band.md` §3.7 and the README are against the Massart, equal-split KS band on the trapezoidal metric | To be restated once Phase A has run; the documents should name the baseline variant |

### 5.2 Rust: boundary families and calibration

One abstraction serves every member: a **boundary family** maps $`(n, \theta)`$ to nondecreasing bound sequences $`b^-_i \le b^+_i`$ in $`[0,1]`$ for the uniform order statistics, with the bands nested in $`\theta`$.

| Family | Parameter | Bounds for $`U_{(i)}`$ |
|---|---|---|
| ELL | local level $`\gamma`$ | Beta$`(i, n+1-i)`$ quantiles at $`\gamma`$ and $`1-\gamma`$ (existing) |
| KS | margin $`d`$ | $`[\,i/n - d,\ (i-1)/n + d\,]`$, clipped to $`[0,1]`$ |
| Berk–Jones | threshold $`c`$ | The interval of $`u`$ with $`n K(i/n, u) \le c`$ and $`n K((i-1)/n, u) \le c`$, with $`K`$ the Bernoulli Kullback–Leibler divergence; roots by bracketed Newton |
| Dümbgen–Wellner | threshold $`\kappa`$ | As in the paper; form to be transcribed from it (§2.2 gate) |
| Eicker | constant $`c`$ | Roots of $`\lvert i/n - u\rvert = c\sqrt{u(1-u)}`$ and the same at $`(i-1)/n`$ |
| Frey, uniform weights | the bounds themselves | Minimizer of $`\sum_i (h_i - l_i)`$ at exact coverage; a convex program ([`frey_ell_plan.md` §1](frey_ell_plan.md#1-one-class-convexity-and-its-consequences)) |
| ROC-optimized | the bounds of both classes | Alternating convex programs on expected ROC-band area ([`frey_ell_plan.md` §2](frey_ell_plan.md#2-two-classes-the-roc-objective)) |

Work items:

1. **Generic calibration.** Bisection on $`\theta`$ against `ell::crossing_prob`, returning the conservative endpoint, for any family. This replaces the ELL-specific bisection in `m3_band_rs.py`. KS-X is the KS family under this routine; KS-M and BONF bypass it with their closed-form parameters.
2. **Numerical error budget.** A margin proportional to the operation count times machine epsilon is a heuristic, not a certificate. State and bound three sources separately: (i) mass deliberately dropped (transition rows whose largest term underflows, and any truncation added in item 3), which only lowers the result; (ii) floating-point error in the sums and products of nonnegative terms, for which a rigorous relative bound follows from the operation count; (iii) error in the bound values themselves, that is, in the Beta quantiles and root-finders that produce $`l_i, h_i`$. Source (iii) does not affect coverage as long as the band is built from the same computed bounds whose crossing probability was evaluated. The calibration target is then raised by the bound on (ii), and the documents say what is and is not bounded.
3. **Faster recursion at large $`n`$.** Truncate binomial transitions whose remaining mass is negligible, keeping the result a lower bound ([`ell_band.md` §4.4](ell_band.md#44-warm-started-calibration-from-a-table-of-levels-proposed)). Measure the speedup; adopt the Moscovich–Nadler FFT recursion only if truncation is not enough.
4. **Calibration table.** A versioned file of calibrated parameters keyed by (family, $`n`$, $`\alpha_c`$), built in parallel over the run's class sizes before the simulation and loaded by workers. Every stored value has passed the exact check, so the table affects speed only.
5. **Check that the windowed recursion's assumptions hold for each family.** `crossing_prob` requires nondecreasing bounds in $`[0,1]`$; the clipped KS and Eicker bounds satisfy this but the tests must confirm it at the extremes. Fixtures with closed-form answers are required for released bounds, repeated bounds, and long gaps at every size the run uses.

### 5.3 Rust: composition, areas, coverage, and the batch interface

1. **Generic composition.** From the label sequence and four bound arrays, return the continuum band as breakpoints and edge values, following [`ell_band.md` §2.6](ell_band.md#26-composition). The native-grid and coarser-grid samplers with the conservative extension are derived from it. The existing ELL path becomes a caller of this routine.
2. **Exact areas.** Total and on any FPR interval, from the breakpoints.
3. **Strict coverage.** Rust returns the breakpoints; Python evaluates the true ROC and its one-sided limits there; Rust returns the verdict, direction, location, and largest exceedance.
4. **Suite call.** One function takes one dataset's labels, the list of method identifiers, and the levels, and returns every band. A batch form takes many label sequences and runs them in parallel with the interpreter lock released.
5. **Python wrappers.** Keyword-only functions with a pydantic model for the method configuration, in a new `exact_bands_rs.py`. `m3_band_rs` and `fixed_width_ks_band` keep their signatures and delegate to it, so existing callers and tests are unaffected.

### 5.4 Validation before the run

Each row is a gate. Fixtures from R and C++ tools are generated once and committed under `tests/fixtures/`, so the test suite does not need those tools.

| What | Checked against | Criterion |
|---|---|---|
| Exact KS margin from the generic calibration | `scipy.stats.kstwo`; R `ks.test(exact = TRUE)` | Agreement to roundoff where the reference is exact. SciPy's values matched the kernel to $`10^{-15}`$ at $`n \le 100`$ and differed by up to $`2 \times 10^{-7}`$ at $`n = 400`$, consistent with SciPy switching to an approximation; at larger $`n`$ the criterion is agreement within the reference's stated accuracy, and the kernel, not SciPy, defines the exact margin |
| `crossing_prob` on arbitrary bounds | `crossprob` (Moscovich–Nadler C++); brute-force enumeration for $`n \le 8`$ | Relative error below $`10^{-9}`$ |
| ELL local levels | R `qqconf`; the current `m3_band_rs`; exact p-values of Moscovich, Nadler & Spiegelman (2016) for their minimum-Beta-tail statistic | Levels within tolerance; bands from the new path equal to the current ones to roundoff |
| Berk–Jones (likelihood-ratio) band | Owen's 1995 table; R `SMIR::NPL.bands` at 0.95 and 0.99; any other source that uses the same likelihood-ratio statistic | Critical values within tolerance. The "exact Berk–Jones" statistic of Moscovich, Nadler & Spiegelman (2016) is the minimum Beta-tail probability, whose inverted band is ELL, not this band; its values are not expected to agree and are used for ELL instead |
| Dümbgen–Wellner band | Critical values in the paper's supplement | Within the supplement's stated precision |
| Class coverage of every family | Monte Carlo with $`10^6`$ uniform samples per $`(n, \alpha_c)`$ | Statistical agreement with the kernel's value: the difference within a prespecified multiple of the Monte Carlo standard error, adjusted for the number of checks. The observed rate is not required to be at or above the target, since an exactly calibrated band falls below it about half the time |
| Composition | An independent index-counting implementation in Python | Equal on random label sequences, including ties, separation, and single-observation classes |
| `ks_m_eq` | The current `fixed_width_ks_band` | Equal to roundoff on the native grid |
| Exact area and strict coverage | Analytic fixtures: step bands and step truths with known areas and known covered or missed status, including violations confined to one breakpoint, at the endpoints $`t = 0`$ and $`t = 1`$, and on the wrong side of a one-sided limit. Dense-grid integration as a secondary check only, since a grid cannot detect an arbitrarily narrow violation | Exact agreement on the analytic fixtures |
| Structural properties of every band | Property tests: monotone edges, $`L \le U`$, nesting in $`\alpha`$, invariance under increasing score transforms, determinism given the seed | All hold |

Tests follow the project's conventions: parametrized over families and class sizes, with realistic label sequences from `conftest.py`.

### 5.5 Order of work

| Step | Content | Exit condition |
|---|---|---|
| 1 | Review of this plan | Changes agreed |
| 2 | Reading gates: Frey 2008, Dümbgen & Wellner 2023 and supplement, Switzer 1976, Shi, Wang & Ma 2026 | §2.2 and §2.3 finalized |
| 3 | Implementation of §§5.2–5.3 | Test suite passes |
| 4 | Validation of §5.4 | Every gate met |
| 5 | Dry run: 300 datasets per design, all methods and strata, on seeds that the main run will not use | Timings and within-stratum standard deviations on the primary scale measured; §3.4 and §4.4 updated |
| 6 | Analysis script written and run on the dry-run data | Produces every table and test of §4 |
| 7 | Freeze: this document, the script, the calibration table, and the trained `ropt_ss` bounds committed; hash recorded. No confirmatory dataset exists before this point | — |
| 8 | Main run and analysis | Results reported against §4.1 without changes; any deviation listed |

---

## 6. Phases B and C (Outline)

These need their own plans. The points already decided:

**Phase B, fiducial band.**
- Two claims: (i) width close to the oracle Hsieh–Turnbull band (true slope, limit-simulated critical value) on regular interiors, with no bandwidth or density estimate; (ii) width within a stated factor of the correctly specified parametric band on its home turf, while that band fails off-model.
- Validity is claimed only on a declared regularity region, the one the auto-rule surface estimates.
- Comparators, all to be reimplemented in Rust and validated against the reference code: the PSN smoothed-bootstrap band and the JMS limit-process band (`nsROC`), a studentized bootstrap sup-band, a sup-calibrated Bayesian bootstrap band, the smoothed empirical-likelihood band (`EL`), Working–Hotelling, the ellipse envelope, and pointwise bands as negative controls.
- The current `hsieh_turnbull_band.py` is retired in favour of the JMS form and the oracle form.
- The Bayesian bootstrap papers must be read first, since they are the closest prior art (refs A.7).

**Phase C, hybrid and auto dispatcher.**
- Claim: the narrowest band with no detected under-coverage on strata S and H.
- The hybrid's behaviour on stratum X must be measured before the claim is scoped; the floor protects the corners, not interior jumps.
- The dispatcher is evaluated end to end as its own method, and its admission rule is validated on families held out from the surface fit.
- Cells are sized from the coverage table of §4.4.

---

## 7. Risks and Open Questions

1. **An alternative may beat ELL on total area.** Frey's and Dümbgen–Wellner's bands are built to be narrower in the centre, and our own ROC-optimized composition was 5–14% narrower than ELL at AUC 0.8–0.95 in a probe. §1.3 commits to reporting that outcome. The low-FPR endpoint is where ELL is most likely to remain best.
2. **The fair baseline shrinks the small-sample result.** The paper's current statement that ELL's mean area is 0.63–0.99 of KS in every cell rests on the Massart, equal-split baseline and will change.
3. **Score-scale exact bands.** If a referee reads Switzer's band as a competitor, the answer is the coordinate argument of refs A.14, which we have not yet confirmed against his text.
4. **Choice of region.** The threshold of 30 in the smaller class was set from a pilot on one family. The dry run could suggest moving it; any change happens before the freeze and is recorded.
5. **Fresh data cost pairing with the existing runs.** Phases B and C must rerun their methods on a subsample of the new datasets; the existing results remain as exploratory evidence.
6. **Should Phase A report at $`\alpha = .5`$ at all?** Exact bands are very conservative there, and the comparison is of little practical use; it is kept for the frontier.
