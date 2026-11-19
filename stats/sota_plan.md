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
3. Are the hypotheses in §4.1 the claims the paper should make, and are the margins sensible?
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
| Realized coverage | Strict whole-curve coverage, endpoints included, zero tolerance | Validity check (H7) and the frontier below |
| Gap to the lower bound | Area on the separated path against $`1 - \alpha^{1/n_0 + 1/n_1}`$ ([`ell_band.md` §3.3](ell_band.md#33-width-and-tail-adaptivity)) | Descriptive |

The existing tables use trapezoidal area on the native grid. That metric stays as a secondary column for continuity. It is not suitable as a primary endpoint for this comparison: on the native grid a constant horizontal margin is rounded to a whole number of grid steps, which changed the pilot's exact-versus-Massart KS ratio by up to 2.8% where the margins differ by 1% (§4.3).

**Comparison scale.** Each comparison is the paired log ratio $`\log(E_{\text{ELL}} / E_c)`$ of an endpoint $`E`$ on the same dataset. Cell means are reported as ratios.

**Decision rule.** ELL (sample-size split) is called state of the art in $`\mathcal{C}`$ on a design region $`\mathcal{D}`$ if all three hold:

1. *Superiority to the published baseline.* H1 and H2 of §4.1 hold in every cell of $`\mathcal{D}`$.
2. *No dominated cell.* No member of the method matrix is significantly narrower than ELL on both primary endpoints in any cell of $`\mathcal{D}`$.
3. *No better alternative overall.* No member has a smaller design-average area with a simultaneous confidence limit excluding a ratio of 1.

If an alternative boundary family fails rule 2 or 3 for ELL, the paper reports that family as the best member. The contribution then shifts from "ELL" to "exactly calibrated tail-sensitive composition", and we say so. This outcome is pre-committed, not treated as a failed study.

**The region $`\mathcal{D}`$.** The pilot shows ELL and exact KS within 1% of each other when the smaller class has about 10 or fewer observations. We split the design in advance:

- **Core region:** $`\min(n_0, n_1) \ge 30`$. Superiority is claimed here.
- **Small-sample region:** $`\min(n_0, n_1) < 30`$. Only non-inferiority is claimed (H3).

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
| MA | Minimum-area bounds | Exact | Frey 2008; refs B.12 | The likeliest to beat ELL on total area. **Gated**, see below |

Each family runs with the sample-size split. KS and ELL also run with the equal split, so the effect of the split is separated from the effect of the family.

**Gate on MA.** Frey's paper has not been read by us and no code was found. Before implementation we read it and decide between two options: implement his bands, or, if they cannot be reproduced reliably, substitute a numerically optimized member of the ELL–KS interpolation family already screened in [`research_record.md`](research_record.md) Q5. The substitution would be reported as such.

**Gate on DW.** The exact form of the bound (the additive correction terms and their constants) is taken from the paper, not from memory, and checked against the critical values tabulated in its supplement.

### 2.3 Excluded, with reasons

| Method | Reason | Reference |
|---|---|---|
| Switzer's Smirnov-inversion band; Doksum–Sievers S- and W-bands | Exact, but for the Q-Q function on the score scale. They give no band in ROC coordinates. Our reading of this point must be confirmed against Switzer's text before the paper asserts it | refs A.14 |
| Hilgers 1991; Schäfer 1994; Tilbury et al. 2000; Clopper–Pearson rectangles; NP-ROC | Pointwise. Their simultaneous extension by a union bound is BONF | refs A.12, A.18 |
| Time-uniform DKW (Howard & Ramdas 2022) | A member, but its margin is about twice the fixed-$`n`$ margin, so KS-M dominates it by construction. Stated, not run | refs B.12 |
| Bates et al. 2023 | Uniform finite-sample bound for one class only | refs A.18 |
| Shi, Wang & Ma 2026 | Guarantee restricted to rejection regions of finite VC dimension. **Pending a full read**; if it contains an unrestricted band it joins the matrix | refs A.19 |
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
| `ma_ss` | Minimum area | Exact | Sample-size | New, gated |

Levels: $`\alpha \in \{.05, .1, .2, .5\}`$. The 90% level is added because Macskassy et al. and Sewak & Hothorn report at it.

All eleven bands are computed from one tie-resolved label sequence per dataset, in one call.

### 3.2 Data-generating processes

Every member reads only the ranks, so its coverage and width depend on the data only through $`(R, n_0, n_1)`$. Families matter only as a source of ROC shapes. Three strata:

**Stratum S (smooth).** The seven families and Latin-hypercube designs of [`simulation_spec.md`](simulation_spec.md) §§2–3, with the same seeds, so that every dataset is the one already used for the fiducial and hybrid results. This pairs Phase A with Phases B and C at no cost.

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

That is 22 **designs** (a total size at a prevalence). The 2% column is new and reflects rare-positive evaluation in machine learning; it gives 20, 60, and 200 positives. The first three columns reuse the existing runs' seeds.

**Analysis cell.** A cell is a design crossed with an **AUC band**, pooled over families. The four bands are quartiles of the design on the probit scale: AUC 0.55–0.71, 0.71–0.86, 0.86–0.95, 0.95–0.99. Banding is needed because the ELL-to-KS ratio depends strongly on AUC: in the pilot the standard deviation of the log ratio within a design was up to 0.35, almost all of it variation of the mean with AUC. That gives 88 cells in stratum S, of which 64 (16 designs) are in the core region.

Family-level results are reported descriptively. They are not tested, because for rank-based methods a family is a set of ROC shapes, not a separate population.

### 3.4 Replication and cost

| Stratum | Datasets | Basis |
|---|---|---|
| S | 6,000 LHS points per family per design: 42,000 per design, about 10,500 per cell, 924,000 in all | Reuses existing seeds; exceeds every requirement in §4.4 |
| H | 3,000 per curve per design, at $`n \in \{100, 300, 1{,}000\}`$ and 50% and 10% prevalence | Detects coverage of 0.93 against 0.95 (§4.4) |
| X | 3,000 per curve per design, same sizes as H plus $`n = 30`$ | Same |

Cost is not the binding constraint in Phase A. A calibrated band costs a few milliseconds ([`ell_band.md` §4.1](ell_band.md#41-complexity)), and all eleven bands at four levels share one sort. We estimate a few core-hours for band construction and truth evaluation together. The one-time calibrations are the larger item: about 40 distinct class sizes, four levels, two splits, and six exactly calibrated families, with the largest sizes at about a minute per bisection today. With the warm-start table and the parallel sweep of §5.2 this is under an hour on the 14-core machine. These estimates are to be replaced by measurements from the dry run (§5.5).

Replication is rationed in Phases B and C, where one band costs seconds. §4.4 is written so that the same table sizes those runs.

### 3.5 Evaluation

- **Truth.** The analytic ROC of each DGP, evaluated at the breakpoints of each band, with left and right limits where $`R`$ jumps.
- **Coverage.** Strict: the band covers if $`L(t) \le R(t) \le U(t)`$ at every $`t \in [0,1]`$, endpoints included, with no tolerance beyond a few units of double-precision roundoff in the comparison. The existing $`10^{-6}`$ tolerance and endpoint exclusion are also reported, for continuity with earlier tables.
- **Recorded per band:** the endpoints of §1.3, miss direction, miss location, and the calibrated parameter.
- **Seeds and metadata:** as in `simulation_spec.md` §4, plus the calibration table's version hash.

---

## 4. Pre-Registered Tests

The hypotheses, margins, regions, and analysis script are frozen by commit before the main run (§5.5). The pilot of §4.3 was used to set regions and margins and is not part of the confirmatory data.

### 4.1 Hypotheses

Ratios are of ELL (`ell_ss`) to the comparator, so values below 1 favour ELL. "Core" and "small" are the regions of §1.3. Unless stated, tests are at $`\alpha = .05`$ band level; the other levels are reported with the same tests as secondary.

| ID | Claim | Comparator | Endpoint | Cells | Test |
|---|---|---|---|---|---|
| H7 | Every member covers at least nominally | — | Strict coverage | All cells, strata S, H, X, all four levels | Flag a cell if the upper confidence limit for coverage is below $`1-\alpha`$. Expect no flags |
| H1 | ELL is narrower than exact KS | `ks_x_ss` | Area | Core | Superiority: mean log ratio $`< 0`$ |
| H2a | ELL is narrower at low FPR | `ks_x_ss` | Low-FPR area | Core | Superiority |
| H2b | The low-FPR gain is large at scale | `ks_x_ss` | Low-FPR area | Core cells with $`n \ge 1{,}000`$ | Superiority with margin: ratio $`< 0.80`$ |
| H3 | ELL is not materially wider in small samples | `ks_x_ss` | Area | Small | Non-inferiority: ratio $`< 1.03`$ |
| H4 | ELL is the best exactly calibrated family | `bj_ss`, `dw_ss`, `ek_ss`, `ma_ss`, each in turn | Area, then low-FPR area | Core | Superiority; where it fails, non-inferiority at ratio $`< 1.02`$; where that fails, the comparator is reported as better |
| H5 | Exact calibration beats the union bound | `bonf_ss` | Area | Core | Superiority |
| H6 | The sample-size split helps under imbalance | `ell_eq` | Area | Core cells at 10% and 2% prevalence | Superiority |
| H8 | KS is narrower mid-curve | `ks_x_ss` | Mid-curve area | Core | Two-sided; the expected sign is ratio $`> 1`$ |

Descriptive only, no test: Massart against exact KS; the effect of the split on KS; dominance fractions; the frontier; the gap to the separated-path lower bound; family-level tables.

H8 is registered so that the paper reports where ELL loses with the same rigour as where it wins.

### 4.2 Test procedure

- **Statistic.** The cell mean of the paired log ratio, with a one-sample $`t`$ test. Datasets within a cell are independent (one per LHS point). A bootstrap interval is reported alongside as a check on the normal approximation.
- **Estimand.** The mean over the cell's design distribution: probit-uniform AUC within the band, the LHS shape ranges, families pooled with equal weight.
- **Multiplicity.** Holm correction across cells within each hypothesis. Across hypotheses, a fixed sequence: H7, then H1, H2a, H3, H2b, H4, H5, H6, H8. A hypothesis is tested only if all earlier ones were not rejected in the wrong direction; H7 gates everything, since a coverage flag means a bug or a broken proof.
- **Level.** Family-wise 0.05 per hypothesis. The power calculations below use a conservative per-test level of $`0.05/250`$ (one-sided $`z = 3.54`$), which approximately covers the largest family (H4: four comparators in 64 cells, 256 tests).
- **Design-average tests** (rule 3 of §1.3) use the equal-weight mean of cell means with a simultaneous interval across comparators.

### 4.3 Pilot

300 datasets per design at $`\alpha = .05`$; heteroscedastic Gaussian scores with probit-uniform AUC in [0.55, 0.99] and log-uniform variance ratio in [0.2, 5]; ELL from the production code and the four KS variants from a reference implementation checked against the current `ks` band. Area is trapezoidal on the native grid, so these numbers are indicative, and the exact-area endpoint may move them by a few percent (§1.3).

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
3. **The log-ratio standard deviation grows with $`n`$** because the ratio depends on AUC. Within-band standard deviations will be smaller; the dry run measures them.

### 4.4 Power

Datasets per cell for 90% power at the per-test level of §4.2.

**Paired width comparisons.** $`N = \{(3.54 + 1.28)\,s/\delta\}^2`$ for a true mean log ratio $`\delta`$ and standard deviation $`s`$.

| SD of log ratio $`s`$ | $`\delta = 0.005`$ | 0.01 | 0.02 | 0.05 |
|---|---|---|---|---|
| 0.025 | 582 | 146 | 37 | 6 |
| 0.05 | 2,325 | 582 | 146 | 24 |
| 0.10 | 9,300 | 2,325 | 582 | 93 |
| 0.20 | 37,198 | 9,300 | 2,325 | 372 |
| 0.30 | 83,694 | 20,924 | 5,231 | 837 |

**Coverage flags.** Datasets needed to flag a cell whose true coverage is below nominal.

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

**Reading the tables for Phase A.** With about 10,500 datasets per cell:

- H1, H2, H5, H6: pilot effects are 2–50% with $`s \le 0.35`$. Power is essentially 1 in the core region.
- H3: a true ratio of 1.006 against a margin of 1.03 is a distance of 0.024 at $`s \approx 0.03`$. About 40 datasets suffice.
- H4: the effect sizes are unknown. At $`s = 0.10`$ the design detects a 1% difference; at $`s = 0.30`$, a 1.4% difference. The 2% non-inferiority margin is therefore resolvable in every cell.
- H7: a shortfall of one percentage point at the 95% level is detected with about 88% power, and two points with near certainty. Exact bands are expected to cover far above nominal, so H7 is a bug detector, and that sensitivity is adequate.

**Reading the tables for Phases B and C.** Where one band costs seconds, the second table sets the floor: about 3,000 datasets per cell to detect a two-point coverage shortfall at 95%, and about 1,400 for three points. Width comparisons need far fewer. Those phases should therefore size cells by the coverage table and can run their width tests on a subsample.

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
| ELL roundoff margin is a fixed $`10^{-9}`$; [`ell_band.md` §3.1](ell_band.md#31-coverage-guarantees) says a production margin should grow with the operation count | To do in §5.2 |
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
| Minimum area | — | Gated (§2.2) |

Work items:

1. **Generic calibration.** Bisection on $`\theta`$ against `ell::crossing_prob`, returning the conservative endpoint, for any family. This replaces the ELL-specific bisection in `m3_band_rs.py`. KS-X is the KS family under this routine; KS-M and BONF bypass it with their closed-form parameters.
2. **Roundoff margin.** Replace the fixed $`10^{-9}`$ target margin with one proportional to the recursion's operation count times machine epsilon.
3. **Faster recursion at large $`n`$.** Truncate binomial transitions whose remaining mass is negligible, keeping the result a lower bound ([`ell_band.md` §4.4](ell_band.md#44-warm-started-calibration-from-a-table-of-levels-proposed)). Measure the speedup; adopt the Moscovich–Nadler FFT recursion only if truncation is not enough.
4. **Calibration table.** A versioned file of calibrated parameters keyed by (family, $`n`$, $`\alpha_c`$), built in parallel over the run's class sizes before the simulation and loaded by workers. Every stored value has passed the exact check, so the table affects speed only.
5. **Check that the windowed recursion's assumptions hold for each family.** `crossing_prob` requires nondecreasing bounds in $`[0,1]`$; the clipped KS and Eicker bounds satisfy this but the tests must confirm it at the extremes.

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
| Exact KS margin from the generic calibration | `scipy.stats.kstwo` quantiles; R `ks.test(exact = TRUE)` | Agreement within the bisection tolerance, for $`n`$ from 1 to 10,000 |
| `crossing_prob` on arbitrary bounds | `crossprob` (Moscovich–Nadler C++); brute-force enumeration for $`n \le 8`$ | Relative error below $`10^{-9}`$ |
| ELL local levels | R `qqconf`; the current `m3_band_rs` | Levels within tolerance; bands from the new path equal to the current ones to roundoff |
| Berk–Jones band | Owen's 1995 table; R `SMIR::NPL.bands` at 0.95 and 0.99; exact p-values of Moscovich, Nadler & Spiegelman 2016 | Critical values within tolerance |
| Dümbgen–Wellner band | Critical values in the paper's supplement | Within the supplement's stated precision |
| Class coverage of every family | Monte Carlo with $`10^6`$ uniform samples per $`(n, \alpha_c)`$ | Coverage at or above target, and within Monte Carlo error of it for exactly calibrated families |
| Composition | An independent index-counting implementation in Python | Equal on random label sequences, including ties, separation, and single-observation classes |
| `ks_m_eq` | The current `fixed_width_ks_band` | Equal to roundoff on the native grid |
| Exact area and strict coverage | Dense-grid numerical integration and evaluation, on curves with and without jumps | Agreement to the dense grid's resolution |
| Structural properties of every band | Property tests: monotone edges, $`L \le U`$, nesting in $`\alpha`$, invariance under increasing score transforms, determinism given the seed | All hold |

Tests follow the project's conventions: parametrized over families and class sizes, with realistic label sequences from `conftest.py`.

### 5.5 Order of work

| Step | Content | Exit condition |
|---|---|---|
| 1 | Review of this plan | Changes agreed |
| 2 | Reading gates: Frey 2008, Dümbgen & Wellner 2023 and supplement, Switzer 1976, Shi, Wang & Ma 2026 | §2.2 and §2.3 finalized |
| 3 | Implementation of §§5.2–5.3 | Test suite passes |
| 4 | Validation of §5.4 | Every gate met |
| 5 | Dry run: 300 datasets per design, all methods and strata | Timings and within-band standard deviations measured; §3.4 and §4.4 updated |
| 6 | Analysis script written and run on the dry-run data | Produces every table and test of §4 |
| 7 | Freeze: this document, the script, and the calibration table committed; hash recorded | — |
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

1. **An alternative family may beat ELL on total area.** Minimum-area and Dümbgen–Wellner bands are built to be narrower in the centre. §1.3 commits to reporting that outcome. The low-FPR endpoint is where ELL is most likely to remain best.
2. **The fair baseline shrinks the small-sample result.** The paper's current statement that ELL's mean area is 0.63–0.99 of KS in every cell rests on the Massart, equal-split baseline and will change.
3. **Score-scale exact bands.** If a referee reads Switzer's band as a competitor, the answer is the coordinate argument of refs A.14, which we have not yet confirmed against his text.
4. **Choice of region.** The threshold of 30 in the smaller class was set from a pilot on one family. The dry run could suggest moving it; any change happens before the freeze and is recorded.
5. **Is the 2% prevalence column worth a new set of seeds?** It breaks pairing with the existing runs for those designs only.
6. **Should Phase A report at $`\alpha = .5`$ at all?** Exact bands are very conservative there, and the comparison is of little practical use; it is kept for the frontier.
