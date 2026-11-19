# Literature References for the Studentized Bootstrap Envelope ROC Band

Annotated bibliography assembled for publication. It covers (A) the prior art
on confidence bands for ROC curves and (B) the statistical techniques the
method is built from. Each entry gives a full citation, a link, and a one-
paragraph summary keyed to the claims it supports in `nonparam_envelope.md`,
`project_evaluation_report.md`, `simulation_spec.md`,
`theoretical_behavior_report.md`, and `notebook_paper.py`.

Citations that could not be verified to a specific page/volume from the search
results are flagged **[verify]** — confirm the bibliographic details before
submission.

---

## A. Prior art: confidence bands for ROC curves

### A.1 Hsieh & Turnbull (1996) — asymptotic theory of the empirical ROC

> Hsieh, F. and Turnbull, B. W. (1996). "Nonparametric and Semiparametric
> Estimation of the Receiver Operating Characteristic Curve." *The Annals of
> Statistics*, 24(1), 25–40.
> https://projecteuclid.org/journals/annals-of-statistics/volume-24/issue-1/Nonparametric-and-semiparametric-estimation-of-the-receiver-operating-characteristic-curve/10.1214/aos/1033066197.full

The foundational large-sample theory underpinning the method's asymptotic
coverage argument (Theorem 1 in `nonparam_envelope.md` §3.1). Hsieh & Turnbull
show that the empirical ROC process √n(R̂ − R_true) converges weakly to a
Gaussian process expressible via two independent Brownian bridges (one per
class), and derive the asymptotic variance
Var(R(t)) = R(t)(1−R(t))/n₁ + [g(c_t)/f(c_t)]²·t(1−t)/n₀. This two-component
variance is the analytical backbone of the entire project: the first
(binomial) term is what the Wilson floor models, and the second
(threshold-location / slope) term is exactly what the Wilson floor *misses* at
the steep low-FPR corner (`project_evaluation_report.md` B.6, B.8, D.9). The
report's claim that "HT with the true slope matches Monte Carlo truth to 1–3%"
is a direct empirical confirmation of this formula. Also the source of the
binormal/semiparametric framing used by the Working–Hotelling baseline.

### A.2 Campbell (1994) — KS-based and bootstrap ROC bands

> Campbell, G. (1994). "Advances in Statistical Methodology for the Evaluation
> of Diagnostic and Laboratory Tests." *Statistics in Medicine*, 13(5–7),
> 499–508. https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.4780130513
> (PubMed: https://pubmed.ncbi.nlm.nih.gov/8023031/)

The direct citation for the `ks` baseline and the optional KS-style boundary
extension (`nonparam_envelope.md` §2.8, `boundary_method="ks"`). Campbell
presents two simultaneous confidence regions for the continuous-data ROC: one
built from Kolmogorov–Smirnov confidence bands on the two class CDFs, and one
from bootstrapping. This is the canonical fixed-width distribution-free
simultaneous band the paper benchmarks against — the method that "always
achieves 100% coverage but is uninformatively wide" (`project_evaluation_report.md`
C.1). Pages 499–508 and DOI 10.1002/sim.4780130513 are confirmed (191–197 is
Ma & Hall, A.6). Macskassy & Provost (A.3) call the KS construction "SJR" and
the bootstrap one "FWB". The idea of composing two one-sample bands predates
Campbell: Doksum (1974) did it for the shift function, and Switzer (1976)
showed how conservative the product-level guarantee is (A.14). The project's
`ks` baseline uses the Dvoretzky–Kiefer–Wolfowitz margin with Massart's
constant (B.5), not exact finite-sample Kolmogorov critical values.

### A.3 Macskassy, Provost & Rosset — ROC confidence bands, empirical evaluation

> Macskassy, S. A. and Provost, F. (2004). "Confidence Bands for ROC Curves:
> Methods and an Empirical Study." *Proc. 1st Workshop on ROC Analysis in AI
> (ROCAI-2004)*. https://archive.nyu.edu/bitstream/2451/27802/2/CPP-07-04.pdf
>
> Macskassy, S. A., Provost, F. and Rosset, S. (2005). "ROC Confidence Bands:
> An Empirical Evaluation." *Proc. 22nd International Conference on Machine
> Learning (ICML)*, 537–544.
> https://icml.cc/Conferences/2005/proceedings/papers/068_ROC_MacskassyEtAl.pdf
> (ACM: https://dl.acm.org/doi/10.1145/1102351.1102419)

The most directly comparable prior work from the ML community and the closest
methodological neighbor: it surveys and empirically compares ways of putting
simultaneous bands around an empirical ROC, including a Working–Hotelling
adaptation, fixed-width (KS/VWB) bands, and a "radial sweep" construction, and
introduces the distinction between *pointwise* and *band* (simultaneous)
coverage that the paper's whole framing rests on. Crucially, Macskassy &
Provost frame the target as the band containing the entire curve — but several
of their methods target *future empirical curves* rather than the *population*
curve, which is exactly the confidence-band-vs-prediction-band distinction made
explicit in `nonparam_envelope.md` §3.1 ("Note on Future Samples"). Position the
present method as the studentized, variance-adaptive, boundary-corrected
successor to this line.

### A.4 Bertail, Clémençon & Vayatis (2008) — bootstrapping the ROC; smoothed bootstrap

> Bertail, P., Clémençon, S. and Vayatis, N. (2008). "On Bootstrapping the ROC
> Curve." *Advances in Neural Information Processing Systems (NeurIPS) 21*.
> https://proceedings.neurips.cc/paper/3404-on-bootstrapping-the-roc-curve.pdf
>
> Bertail, P., Clémençon, S. and Vayatis, N. (2008). "On Constructing Accurate
> Confidence Bands for ROC Curves Through Smooth Resampling." HAL preprint
> hal-00335232. https://hal.science/hal-00335232v1

The key prior result on bootstrap validity for the ROC and a direct warning
relevant to the method's boundary problem. Bertail et al. prove that the
*naive* nonparametric bootstrap of the ROC is deficient and that a *smoothed*
bootstrap (resampling from a kernel-smoothed empirical distribution) is needed
for accurate confidence-band coverage, because the ROC depends on the score
densities the naive bootstrap cannot reproduce. **Verified rates (read from the
paper):** the empirical ROC is strongly consistent in sup-norm with a strong
approximation at rate n^{-1/2}; the *naive* bootstrap approximation of a given
ROC quantile converges only at O_P(n^{-1/4}), whereas the *smoothed* bootstrap
(bandwidth h_n ∼ n^{-1/5}) improves this to n^{-2/5} — still slower than the
Gaussian approximation but a large practical gain. Their own simulation (n =
100, nominal 95%) shows the deficiency concretely: for the sup-deviation
statistic d_n the naive bootstrap covers 90.3% vs. the smoothed 93.1%, and at
the single point r_n(0.2), 89.7% vs. 92.5%. This is the theoretical mirror of
the project's central empirical finding — that the naive stratified bootstrap
collapses one-sidedly at extreme order statistics and cannot represent
probability mass beyond the observed support (`project_evaluation_report.md`
B.5, B.8). The paper should cite this as prior recognition of the problem, then
argue that the exact Beta order-statistic floor is an alternative
(assumption-free, **bandwidth-free**) repair to smoothing at precisely the
corner where bandwidth selection is hardest to tune — a low-FPR-tail
order-statistic guarantee that does not exist in this line of work.

### A.5 Demidenko (2012) — pointwise CIs and the ellipse-envelope binormal band

> Demidenko, E. (2012). "Confidence Intervals and Bands for the Binormal ROC
> Curve Revisited." *Journal of Applied Statistics*, 39(1), 67–79.
> https://www.tandfonline.com/doi/full/10.1080/02664763.2011.578616
> (PubMed: https://pubmed.ncbi.nlm.nih.gov/22523442/)

The ellipse-envelope method named in `project_evaluation_report.md` C.3 and the
source of the "envelope of pointwise confidence regions" idea. Demidenko
constructs confidence ellipses at each binormal operating point and takes their
envelope, improving on Working–Hotelling by accounting for variance estimation;
he shows it covers near-nominal at small/moderate n where Ma & Hall under-cover.
It shares the fatal binormal dependence (catastrophic off-model failure), so it
serves the paper as the "best parametric simultaneous band" foil — and as a
conceptual antecedent of an envelope-of-regions construction, distinct from the
present *bootstrap* envelope-of-curves.

### A.6 Ma & Hall (1993) — Working–Hotelling bands for the ROC

> Ma, G. and Hall, W. J. (1993). "Confidence Bands for Receiver Operating
> Characteristic Curves." *Medical Decision Making*, 13(3), 191–197.
> https://pubmed.ncbi.nlm.nih.gov/8412547/

The original Working–Hotelling-type simultaneous band for the ROC, i.e. the
`working_hotelling` baseline in `simulation_spec.md` §5 and the comparator
throughout `notebook_paper.py` Figs 5–7. Ma & Hall parametrize the binormal ROC
as TPR = Φ(a + b·Φ⁻¹(FPR)) and apply Working–Hotelling hyperbolic bands for
simple linear regression in probit–probit space. This is the method whose
coverage "collapses catastrophically under misspecification (<20% for
Student-t)" in the simulation suite, and the one Demidenko (A.5) set out to
improve. Cite for the precise construction of the parametric baseline.

### A.7 Gu, Ghosal & Roy (2008) — Bayesian bootstrap ROC bands

> Gu, J., Ghosal, S. and Roy, A. (2008). "Bayesian Bootstrap Estimation of ROC
> Curve." *Statistics in Medicine*, 27(26), 5407–5420.
> https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.3366

A nonparametric Bayesian alternative for ROC uncertainty: the Bayesian
bootstrap gives a bandwidth-free smoothing of the empirical ROC and credible
bands for the curve and AUC. Worth citing in the related-work survey as the
Bayesian counterpart to the frequentist resampling approach, and as another
method that produces *credible* (not guaranteed-coverage simultaneous) bounds —
useful contrast for the paper's emphasis on calibrated frequentist coverage.

**Nearest prior art for the fiducial band.** The Bayesian bootstrap puts
Dirichlet(1, …, 1) weights on each class's observations with the observed
merged order held fixed, which is close to drawing uniform spacings per class
conditional on the rank sequence. The companion theory paper is Gu, J. and
Ghosal, S. (2008), "Strong approximations for resample quantile processes and
application to ROC methodology," *Journal of Nonparametric Statistics* 20(3),
229–240, which claims asymptotically accurate coverage for bands. Candidate
differences of the fiducial band, to be checked against the full text: n + 1
spacings per class including both end gaps, within-gap completion, a
simultaneous trim by minimum pointwise rank depth, and the corner allowances.
Whether Gu and Ghosal's bands are simultaneous is **[verify]**; the shipped
implementation (`ROCnReg::pooledROC.BB`) returns a pointwise credible band,
and MATLAB PRT's `prtScoreRocBayesianBootstrap` advertises an undefined
"uniform credible band". A sup-calibrated Bayesian bootstrap band is a
baseline a referee will expect next to the fiducial band.

### A.8 Zheng, Yang & Song (2024) — conformal ROC bands

> Zheng, Z., Yang, B. and Song, P. X. K. (2024). "Quantifying Uncertainty in
> Classification Performance: ROC Confidence Bands Using Conformal Prediction."
> arXiv:2405.12953 (submitted to *The Annals of Statistics*). Dept. of
> Biostatistics, University of Michigan. https://arxiv.org/abs/2405.12953

The most recent competing paradigm and a natural "compare and contrast." It
builds *individualized* conformal confidence intervals for each test
observation's TPR/FPR contribution and aggregates them into an ROC band, with
separate algorithms for iid and non-iid test data. **What it actually targets
(read from the theorems), and why it is not an apples-to-apples competitor:**
the estimand is the uncertainty of classification *on the test data* — the
covered objects are the random variables p^(k)_new = p_n(x_new | y_new = k),
i.e. the operating points generated by random test instances through the
trained score function (Thm 3.2 covers "the ROC curve given by p_n(x_new),
x_new ∈ D_tst"). That is an uncertainty statement about a finite-test-set ROC,
much closer to a *prediction* band for an empirical ROC than to a confidence
band for the fixed population curve R_true defined by the two class-conditional
score distributions — the very distinction drawn in `nonparam_envelope.md` §3.1
("Note on Future Samples"). Two further qualifications matter for the related-
work framing: (i) the guarantee is **asymptotic** (lim over calibration-set
size, with a finite-sample correction term ~2Cφ_n), not the clean exact
finite-sample conformal guarantee, because the covered target is a continuous
random variable whose distribution is estimated; and (ii) coverage is
established **per operating point (marginal)** and the band is formed by
*combining* individualized intervals — there is no sup-norm / joint-multiplicity
argument, so simultaneous coverage over the whole curve in the strong sense this
paper guarantees is not shown. Cite to situate the studentized-envelope-plus-
exact-floor approach against the conformal route, noting the estimand and
simultaneity differences (see §E for the extensibility discussion). As of
November 2026 the arXiv entry is still v1 with no journal reference and no
released code. A follow-up applies the idea to graph classifiers: Wu, Yang,
Chen, Chen and Zheng, "Conditional Prediction ROC Bands for Graph
Classification," arXiv:2410.15239 (abstract only read).

### A.9 nsROC — software reference for ROC bands

> Pérez Fernández, S., Martínez Camblor, P., Filzmoser, P. and Corral, N.
> (2018). "nsROC: An R Package for Non-Standard ROC Curve Analysis." *The R
> Journal*, 10(2), 55–77. https://journal.r-project.org/articles/RJ-2018-043/

Useful as the implementation reference point for existing ROC confidence-band
methods. Its `ROCbands` function packages exactly three simultaneous bands —
**JMS** (A.11, Jensen et al. 2000), **PSN** (A.10, Martínez-Camblor et al.), and
**DEK** (A.5, Demidenko 2012) — which is a tidy map of the practitioner-available
landscape and a useful "compare against what ships today" set. Cite when noting
why a calibrated, assumption-free, density-estimation-free band fills a gap
(two of the three shipped methods require kernel density estimation).

### A.10 Martínez-Camblor, Pérez-Fernández & Corral — PSN smoothed-bootstrap band

> Martínez-Camblor, P., Pérez-Fernández, S. and Corral, N. (2018). "Efficient
> Nonparametric Confidence Bands for Receiver Operating-Characteristic Curves."
> *Statistical Methods in Medical Research*, 27(6), 1892–1908 (online 2016).
> https://doi.org/10.1177/0962280216672490

The `PSN` band in nsROC and the most flexible existing nonparametric option: a
smoothed-bootstrap approximation to the distribution of a pivotal sup-type
statistic, with a tunable width-allocation parameter (`alpha1`) and
applicability to general (non-monotone) ROC curves. It is the closest existing
*nonparametric simultaneous* competitor to the present method, and it shares the
same two dependencies the paper argues against — a kernel **bandwidth/scale
parameter** (`s`, via h = s·σ̂·min(n₊,n₋)^{1/5}) and a smoothing step whose
behavior near the corners is exactly where the order-statistic problem bites.
Position the studentized envelope as achieving comparable interior adaptivity
without a bandwidth, plus an exact tail floor. **Boundary handling — a blunt
clamp (exact text, nsROC `ROCbands` docs):** "if the lower band is lower than 0
or higher than 0.95 it is forced to be 0 or 0.95, respectively; on the other
hand, if the upper band is higher than 1 or lower than 0.05 it is forced to be 1
or 0.05, respectively." This caps the *band values* (sensitivity, y-axis) — the
lower band may never claim TPR > 0.95 and the upper never < 0.05 — an
undocumented stabilization that keeps the bands off the TPR = 0/1 boundaries
where the pivotal statistic's variance degenerates. It is the same corner
instability this paper treats, handled by a non-adaptive hard cap rather than an
exact floor.

**Published simulation protocol and results (read from the preprint).**
Controls and cases from normal, lognormal, gamma, and normal-mixture pairs in
three scenarios; (n, m) ∈ {(50,50), (50,100), (100,100), (100,200)}; 2,000
replicates at 95% with B = 500; coverage of the whole true curve on the nsROC
grid; width as the area between the bands. PSN covered 94–99.6% at s = 1 and
fell to 70.8% in one cell at s = 2. In the same study JMS (A.11) covered
29–61% in the first scenario, DEK (A.5) 56–63% on a binormal case and near 0%
off-model, and pointwise-joined bootstrap bands peaked at 78.9%. The DEK
figures conflict with Demidenko's own 0.89–0.94, so the nsROC DEK
implementation or the coverage check should be validated before DEK is used
as a baseline. The paper's bandwidth is h = s·min(n, m)^{-1/5}·σ̂; the nsROC
R Journal text omits the exponent, and the package source includes it. PSN
picks its α split between the edges to minimise band area. This is the main
published competitor for any "narrowest band that keeps nominal coverage"
claim.

### A.11 Jensen, Müller & Schäfer (2000) — JMS Brownian-bridge band

> Jensen, K., Müller, H.-H. and Schäfer, H. (2000). "Regional Confidence Bands
> for ROC Curves." *Statistics in Medicine*, 19(4), 493–509.
> https://pubmed.ncbi.nlm.nih.gov/10694732/

A genuine prior **nonparametric simultaneous** ROC band that the bibliography
was previously missing, and a direct instantiation of the Hsieh–Turnbull (A.1)
limit theory. JMS build symmetric bands from the asymptotic Brownian-bridge
distribution of the empirical ROC process, approximating the limit process'
distribution by Monte Carlo and estimating the score densities by kernel
smoothing; they also introduce *regional* (sub-interval) bands. This is the
prior method most similar in *goal* to the present one — assumption-light,
nonparametric, simultaneous — so it belongs in the head-to-head framing. Its
Achilles' heel is the one the project repeatedly identifies: it needs the
density ratio g(c)/f(c) (the Hsieh–Turnbull slope term), whose tail estimation
is unstable — `project_evaluation_report.md` D.9 shows the slope term is exactly
what no binomial yardstick recovers and what density estimation must supply.

**Reference implementation (read from the nsROC source).**
`nsROC::ROCbands(method="JMS")` works on the open region (a, b), by default
(1/N, 1 − 1/N). It estimates both densities with R's default Gaussian kernel
(`density()`), forms the Hsieh–Turnbull variance, simulates the limit process
ψ(p) = λ^{-1/2}·(g/f)·B₁(1 − p) + (1 − λ)^{-1/2}·B₂(Ĝ(c_p)) from two
independent Brownian bridges, and takes a quantile of sup|ψ|/√Var as the
critical value. It centres the band on a midpoint-smoothed empirical ROC. Its
authors say the code follows their reading of the paper and was not checked
with Jensen et al. The source takes the 1 − α/2 quantile of the two-sided
supremum, which looks conservative relative to the 1 − α quantile that
two-sided coverage needs; compare with the original paper before treating
either as the definition. This is the literature form of a "Hsieh–Turnbull
band": the critical value comes from the estimated Gaussian limit, not from a
bootstrap or a Bonferroni count.

### A.12 Hilgers (1991) — exact distribution-free ROC bounds at a chosen cut-off

> Hilgers, R. A. (1991). "Distribution-free Confidence Bounds for ROC Curves."
> *Methods of Information in Medicine*, 30(2), 96–101. PMID 1857255.
> https://doi.org/10.1055/s-0038-1634822 **[verify first name]**
>
> Schäfer, H. (1994). "Efficient Confidence Bounds for ROC Curves."
> *Statistics in Medicine*, 13(15), 1551–1561.
> https://doi.org/10.1002/sim.4780131506

The earliest exact, distribution-free ROC inference found, and the nearest
ancestor of the exact composition band's *per-class* logic. Per the abstract,
the bounds "are constructed from minimum and maximum coverages which at a given
value [of the cut-off] can be guaranteed with a confidence," are "strongly
related to two-sided distribution-free tolerance regions," and give a rectangle
that holds the true sensitivity and specificity at a chosen cut-off with a
product-form confidence. That is order-statistic (tolerance-limit) exactness,
but **pointwise**: the abstract claims nothing simultaneous over the curve.
Cite as prior art for exact per-class bounds composed into an ROC statement,
contrasting its single cut-off with the M3 band's sup-norm guarantee. Only the
abstract was read; check the full text for any simultaneous variant before
asserting there is none. A second search (November 2026) could not reach the
full text either; secondary sources describe pointwise rectangles only.
MedCalc's "95% Confidence Bounds for the ROC curve" are Hilgers's. Schäfer
(1994) shortens the intervals at a specified point of the curve by up to 40%
using the Greenhouse–Mantel test; per its abstract it is also pointwise.
Related exact single-point regions: Tilbury et al. (2000), *IEEE Transactions
on Biomedical Engineering* 47, 952–963; Kerekes (2008), *IEEE Geoscience and
Remote Sensing Letters* 5(2); and the Clopper–Pearson rectangles at level
1 − √(1 − α) per axis drawn by R's `plotROC::geom_rocci`.

### A.13 Ledwina & Zagdański (2024) — ODC/ROC comparison curves (not a band)

> Ledwina, T. and Zagdański, A. (2024). "ODC and ROC curves, comparison curves,
> and stochastic dominance." arXiv:2401.14094 [stat.ME].
> https://arxiv.org/abs/2401.14094 **[check for a published version]**

The most recent rank-process work on the ODC/ROC found, read to rule it in or
out as M3 prior art. It is out: it builds no confidence band for the ROC or
ODC. Its "one-sided simultaneous 1−α level acceptance regions" are barriers on
a 127-point dyadic grid for a variance-standardized comparison-curve process,
calibrated by Monte Carlo under F = G, with test-size control under the
composite null but no coverage claim under alternatives. It neither composes
per-class bands nor uses Beta order-statistic local levels, and it cites
neither Campbell (A.2) nor any Berk–Jones or tail-sensitive band. Useful only
for its pointers to the two-sample P-P literature (Aly, Csörgő & Horváth 1987;
Doksum & Sievers 1976; Li, Tiwari & Wells 1996).

### A.14 Doksum (1974), Switzer (1976), Doksum & Sievers (1976) — exact two-sample Q-Q bands

> Doksum, K. (1974). "Empirical Probability Plots and Statistical Inference
> for Nonlinear Models in the Two-Sample Case." *The Annals of Statistics*,
> 2(2), 267–277. https://doi.org/10.1214/aos/1176342662
>
> Switzer, P. (1976). "Confidence Procedures for Two-Sample Problems."
> *Biometrika*, 63(1), 13–25. https://doi.org/10.1093/biomet/63.1.13
>
> Doksum, K. A. and Sievers, G. L. (1976). "Plotting with Confidence:
> Graphical Comparisons of Two Populations." *Biometrika*, 63(3), 421–434.
> https://doi.org/10.2307/2335720

The exact two-sample literature a referee of the exact composition band is
most likely to raise. All three target the Q-Q (treatment or shift) function
G⁻¹∘F on the score scale. Switzer's argument (full text read): transform one
sample by a candidate function; at the true function the two samples are
exchangeable, so inverting any two-sample rank test gives an exact
distribution-free confidence set. Tests whose acceptance region is a box on
the rank vector give bands whose edges are order statistics of the other
sample, with coverage equal to the fraction of the C(m+n, m) rank
arrangements inside the box. Inverting the Smirnov test gives the exact band
that R's `stats::qqplot(conf.level=)` draws. Doksum (1974) built nearly the
same band but set its level by combining two one-sample procedures, which is
the composition idea Campbell (A.2) later applied to the ROC; Switzer's
example at m = n = 40 has a guaranteed level of about 50% against a true
level of about 94.5%. Doksum and Sievers add a variance-weighted "W-band",
implemented with exact lattice-path levels in Wilcox's WRS functions `sband`,
`wband`, and `kswsig` (paper not read; description from the code).

**Relation to the ROC band.** These bands are stated on the score scale,
where coverage does not depend on (F, G). A band for the ROC in (FPR, TPR)
coordinates needs the true false-positive rate at each observed threshold,
which the Q-Q band does not supply, and the composition band's coverage does
depend on the ROC. The Q-Q bands are therefore related work and the source of
the conservativeness argument, not drop-in competitors on ROC axes. This
reading is the project's own and should be checked against Switzer's text
before it is asserted in print. Cite Doksum for priority on composition and
Switzer for the gap between the product-level guarantee and the exact level.

### A.15 Horváth, Horváth & Zhou (2008) — bootstrap bands without density estimation

> Horváth, L., Horváth, Z. and Zhou, W. (2008). "Confidence Bands for ROC
> Curves." *Journal of Statistical Planning and Inference*, 138(6),
> 1894–1904. **[verify issue; abstract not read]**

Bootstrap bands for √n(R̂ − R) that avoid estimating the densities, with
asymptotic validity proved for a smoothed bootstrap (as described by
Martínez-Camblor et al., A.10). The bands are unstudentized and hence of
constant width, which A.10 criticises as inefficient at the ends. No software
found. Together with A.4 this is the "plain bootstrap sup-band" baseline.

### A.16 Smoothed empirical-likelihood ROC intervals and bands

> Claeskens, G., Jing, B.-Y., Peng, L. and Zhou, W. (2003). "Empirical
> Likelihood Confidence Regions for Comparison Distributions and ROC Curves."
> *The Canadian Journal of Statistics*, 31(2), 173–190.
> https://doi.org/10.2307/3316066
>
> Valeinis, J. and Cers, E. "Extending the Two-Sample Empirical Likelihood
> Method." Preprint; R package `EL`. **[no journal version confirmed]**

Smoothed empirical likelihood with a Wilks limit gives pointwise ROC
intervals; whether Claeskens et al. also give simultaneous bands is
**[verify]**. `EL::EL.plot(method="roc", simultaneous=TRUE)` bootstraps a
simultaneous critical value on 50 points of an interval trimmed by 5% at each
end, so the corners are excluded. It needs a kernel bandwidth (default
`bw.nrd0`). Apart from nsROC (A.9) it is the only shipped nonparametric
simultaneous ROC band found. Pointwise only, and not band methods: Gong, Peng
& Qi (2010), *Journal of Multivariate Analysis* 101(6), 1520–1531 (smoothed
jackknife empirical likelihood); Hall, Hyndman & Fan (2004), *Biometrika*
91(3), 743–750 (kernel ROC "intervals and bands" with undersmoothing).

### A.17 Further parametric and semiparametric bands

> Yin, J. and Tian, L. (2016). "Generalized Inference Confidence Band for
> Binormal ROC Curve." *Statistics in Biopharmaceutical Research*, 8(1).
> https://doi.org/10.1080/19466315.2015.1093957
>
> Sewak, A. and Hothorn, T. (2022). "Estimating Transformations for
> Evaluating Diagnostic Tests with Covariate Adjustment." arXiv:2208.03679.
> R package `tram`. **[verify final title and venue]**

Yin and Tian build a binormal band from generalized pivotal quantities and
report satisfactory small-sample coverage where Working–Hotelling (A.6) and
the ellipse envelope (A.5) are liberal; it is the only "generalized
inference" ROC band found and it is parametric. Sewak and Hothorn give
score-inverted uniform bands in a transformation model, valid when the link
is correctly specified. Their simulation (probit, logit, and cloglog links;
AUC 0.5–0.95; 25–100 per class; 90% level) compares against pROC, ROCit, and
the three nsROC bands and finds only the correctly specified model accurate,
with PSN covering at the price of a much larger area. Both are home-turf
foils for a "price of nonparametric inference" comparison. Stata's `rocfit`
followed by `rocplot, confband` ships the Ma–Hall band (A.6).

### A.18 Pointwise and single-operating-point methods from the ML literature

> Tong, X., Feng, Y. and Li, J. J. (2018). "Neyman–Pearson Classification
> Algorithms and NP Receiver Operating Characteristics." *Science Advances*,
> 4(2), eaao1659. arXiv:1608.03109. Package `nproc`.
>
> Urlus, R., Baak, M., Collot, S. and Fridman Rojas, I. (2023). "Pointwise
> Sampling Uncertainties on the Precision-Recall Curve." *AISTATS*, PMLR 206,
> 8211–8232. https://proceedings.mlr.press/v206/urlus23a.html. Package `mmu`.
>
> Bates, S., Candès, E., Lei, L., Romano, Y. and Sesia, M. (2023). "Testing
> for Outliers with Conformal p-values." *The Annals of Statistics*, 51(1),
> 149–178. arXiv:2104.08279.
>
> Dugas, C. and Gadoury, D. (2010). "Pointwise Exact Bootstrap Distributions
> of ROC Curves." *Machine Learning*, 78, 103–136.

None of these is a simultaneous band for the population ROC. Tong et al.
bound type I and type II error at each order-statistic threshold with exact
binomial tails: finite-sample and distribution-free, but one classifier and
one threshold at a time. It is the ML-side precedent for exact
order-statistic bounds on operating points, so the exact band should say how
it goes beyond it. Urlus et al. give asymptotic profile-likelihood regions
per threshold for PR and ROC curves and a "curve uncertainty" union with no
simultaneous guarantee; the method backs an open scikit-learn proposal (issue
25856). Bates et al. give a finite-sample bound on the false-positive rate
uniform over thresholds, which is the one-sample (negative-class) half of the
problem. Dugas and Gadoury compute pointwise bootstrap laws exactly. Related
practice: membership-inference and privacy-auditing papers report
Clopper–Pearson intervals for the TPR at a fixed low FPR, ignoring that the
threshold is estimated from the negatives (Carlini et al. 2022,
arXiv:2112.03570, sets the low-FPR reporting norm); biometrics uses bootstrap
intervals (Bolle, Ratha & Pankanti 2004, *Computer Vision and Image
Understanding* 93, 1–33; Wu, Martin & Kacker 2011, *Journal of Research of
NIST* 116(1)). The learning-theory bounds of Clémençon and Vayatis
(TreeRank and related work) concern the gap between a learned scorer's ROC
and the optimal ROC, not uncertainty about a fixed scorer's curve.

### A.19 Shi, Wang & Ma (2026) — finite-sample bands for trade-off functions

> Shi, Wang and Ma (2026). arXiv:2605.10774. **[verify title, author first
> names; abstract only read]**

Simultaneous bands for the trade-off (ROC) curve between two distributions by
test inversion with non-asymptotic guarantees, under the restriction that
rejection regions come from a class of finite VC dimension, together with a
statement that uniform inference over unrestricted nonparametric classes is
impossible and matching lower bounds under monotone likelihood ratio. This is
the newest neighbour of the exact composition band and a possible source for
an optimality-gap discussion. Read in full before the related-work section is
written; its estimand and constants are not yet known to this project.

### A.20 Goldman & Kaplan (2018) — two-sample Dirichlet bands for testing

> Goldman, M. and Kaplan, D. M. (2018). "Comparing Distributions by Multiple
> Testing across Quantiles or CDF Values." *Journal of Econometrics*, 206(1),
> 143–166. https://doi.org/10.1016/j.jeconom.2018.04.003

The closest published construction to an equal-local-levels two-sample band
(full text read by the search). Each sample gets Beta-quantile bands at a
common local level, but the level is calibrated under F = G for testing, and
the authors state that the bands do not have 1 − α coverage. Stata `distcomp`
implements the test. Also test-only: Buja and Rolke, "Calibration for
Simultaneity" (unpublished manuscript; permutation null bands for Q-Q plots);
Konstantinou, Mrkvička & Myllymäki (2025), *Computational Statistics* 40,
2553–2582 (global envelope n-sample tests).

---

## B. Statistical techniques used by the method

### B.1 Wilson (1927) — score interval for a binomial proportion

> Wilson, E. B. (1927). "Probable Inference, the Law of Succession, and
> Statistical Inference." *Journal of the American Statistical Association*,
> 22(158), 209–212.

The exact source of the variance floor and the Wilson rectangle floor
(`nonparam_envelope.md` §2.4, §2.7; `wilson_rectangle_*` baselines). The Wilson
score interval inverts the normal-approximation z-test for a proportion p with
n trials, yielding the asymmetric, always-positive-width interval whose
implied variance σ²_Wilson = [p(1−p)/n + z²/(4n²)]/(1+z²/n)² is the method's
boundary variance floor. Its defining virtue — non-degenerate width even at
p = 0 or p = 1 — is precisely why it is chosen to stabilize studentization and
to floor the band on the TPR plateau where the bootstrap variance is exactly
zero. Cite alongside Agresti & Coull (B.2) for the modern justification of its
small-sample superiority over the Wald interval.

### B.2 Agresti & Coull (1998); Newcombe (1998) — modern case for the score interval

> Agresti, A. and Coull, B. A. (1998). "Approximate Is Better than 'Exact' for
> Interval Estimation of Binomial Proportions." *The American Statistician*,
> 52(2), 119–126.
>
> Newcombe, R. G. (1998). "Two-Sided Confidence Intervals for the Single
> Proportion: Comparison of Seven Methods." *Statistics in Medicine*, 17(8),
> 857–872.

Supporting citations establishing that the Wilson score interval has near-
nominal coverage and good behavior near 0/1 and at small n, where the Wald
interval fails. These justify the choice of Wilson (over Wald or "exact"
Clopper–Pearson) as the binomial component of the floor and rectangle bands.

### B.3 Hall & Horowitz (2013); Horowitz (2001) — bootstrap supremum confidence bands

> Hall, P. and Horowitz, J. (2013). "A Simple Bootstrap Method for Constructing
> Nonparametric Confidence Bands for Functions." *The Annals of Statistics*,
> 41(4), 1892–1921. https://www.jstor.org/stable/23566534
> (preprint: https://arxiv.org/pdf/1309.4864)
>
> Horowitz, J. L. (2001). "The Bootstrap." In *Handbook of Econometrics*, Vol.
> 5, 3159–3228.
>
> Hall, P. (1993). "On Edgeworth Expansion and Bootstrap Confidence Bands in
> Nonparametric Curve Estimation." *JRSS-B*, 55(1), 291–304.

The theoretical license for studentized bootstrap simultaneous bands, cited in
`project_evaluation_report.md` B.3. Hall & Horowitz formalize building a band by
inverting a bootstrap distribution of a *supremum* studentized statistic over a
grid — exactly the retention rule in `nonparam_envelope.md` §2.5–2.6 (keep the
(1−α) fraction with smallest sup_t|z_b(t)|). Hall (1993) is the classic result
that studentization/bootstrap-t yields higher-order-accurate bands. These
support the asymptotic coverage claim while the report is careful to note the
argument holds on compact subintervals of (0,1) and breaks at the moving
boundary points — the gap the exact floors fill.

### B.4 Efron & Tibshirani (1993) — the bootstrap and the bootstrap-t

> Efron, B. and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
> Chapman & Hall/CRC.
>
> Davison, A. C. and Hinkley, D. V. (1997). *Bootstrap Methods and Their
> Application*. Cambridge University Press.

Standard references for the stratified resampling scheme (resample n₀ negatives
and n₁ positives with replacement; `nonparam_envelope.md` §2.3) and the
studentized "bootstrap-t" idea (normalizing each replicate's deviation by a
variance estimate). Cite for the general validity and mechanics of the
percentile, studentized, and stratified bootstrap that the method and the
`pointwise`/`pointwise_sidak` baselines rest on.

### B.5 Dvoretzky–Kiefer–Wolfowitz (1956); Massart (1990) — the KS band constant

> Dvoretzky, A., Kiefer, J. and Wolfowitz, J. (1956). "Asymptotic Minimax
> Character of the Sample Distribution Function and of the Classical Multinomial
> Estimator." *Annals of Mathematical Statistics*, 27(3), 642–669.
>
> Massart, P. (1990). "The Tight Constant in the Dvoretzky–Kiefer–Wolfowitz
> Inequality." *The Annals of Probability*, 18(3), 1269–1283.

The finite-sample, distribution-free inequality behind the fixed-width KS band
(`ks` baseline, labeled "KS fixed-width (DKW)" in `notebook_paper.py`). DKW
bounds the sup-distance between empirical and true CDF; Massart's sharp constant
C = 2 gives the usable band half-width ε = √(ln(2/α)/(2n)). This is the source
of the KS band's guaranteed coverage (and its uninformative width) that the
paper uses as the safety benchmark. Note the band is applied to each class CDF
and propagated to the ROC.

### B.6 Order statistics, the probability integral transform, and the Beta law

> David, H. A. and Nagaraja, H. N. (2003). *Order Statistics*, 3rd ed. Wiley.
>
> (Standard result) For a continuous F, F(X) ~ Uniform(0,1), and the j-th
> order statistic of n uniforms is Beta(j, n+1−j).

The exact, distribution-free foundation of the Beta order-statistic floor
(`nonparam_envelope.md` §2.7, §6.2 step 5c; the central innovation in
`theoretical_behavior_report.md` §3.4). By the probability integral transform,
the true FPR exceedance at the j-th largest negative score,
F̄(X₍ⱼ₎), is exactly Beta(j, n₀+1−j) regardless of the score distribution.
This is the one finite-sample, assumption-free law available at the steep
corner where the empirical-process/bootstrap arguments fail, and it is what lets
the lower-band floor carry the horizontal (threshold-location) uncertainty the
Wilson floor is structurally blind to. Cite David & Nagaraja (or any standard
order-statistics text) for the Beta result and for the ties argument (with
atoms the exceedance is stochastically smaller, so the bound errs conservative).

### B.7 Šidák (1967) and Bonferroni — multiplicity corrections

> Šidák, Z. (1967). "Rectangular Confidence Regions for the Means of
> Multivariate Normal Distributions." *Journal of the American Statistical
> Association*, 62(318), 626–633.
>
> (Bonferroni inequality — standard; see e.g. Miller, R. G. (1981).
> *Simultaneous Statistical Inference*, 2nd ed. Springer.)

The corrections used to make the rectangle and Beta floors simultaneous across
their grid points / order-statistic events (`nonparam_envelope.md` §2.7:
Šidák level α_w = 1−(1−α)^{1/K_eff}; per-event Bonferroni α_e = α/(2J) for the
Beta floor; `wilson_rectangle_sidak`/`wilson_rectangle_bonferroni` baselines).
Šidák gives the exact correction under independence (1−(1−α)^{1/K}); Bonferroni
the conservative union bound valid under arbitrary dependence. Cite to justify
the multiplicity budgeting and to flag the known caveat that Šidák's
independence assumption is only approximate across correlated grid points
(`project_evaluation_report.md` C.5: rectangles "degrade at large n" as the
correction mishandles many dependent test points).

### B.8 McKay, Beckman & Conover (1979); Iman & Conover (1982) — LHS with rank correlation

> McKay, M. D., Beckman, R. J. and Conover, W. J. (1979). "A Comparison of Three
> Methods for Selecting Values of Input Variables in the Analysis of Output from
> a Computer Code." *Technometrics*, 21(2), 239–245.
>
> Iman, R. L. and Conover, W. J. (1982). "A Distribution-Free Approach to
> Inducing Rank Correlation Among Input Variables." *Communications in
> Statistics – Simulation and Computation*, 11(3), 311–334.
>
> Stein, M. (1987). "Large Sample Properties of Simulations Using Latin
> Hypercube Sampling." *Technometrics*, 29(2), 143–151.

The simulation design references for `simulation_spec.md` §3 and
`src/studroc_paper/sampling/lhs.py`. McKay et al. introduce Latin Hypercube
Sampling as a space-filling variance-reduction design over the (AUC, shape)
sampling space; Iman & Conover give the rank-reordering transform used to
decorrelate / impose dependence among the sampled DGP parameters; Stein
establishes LHS's variance-reduction properties. Cite for the
maximin-LHS-plus-Iman–Conover construction of the 1000-point-per-DGP design.

### B.9 Working & Hotelling (1929) — the original simultaneous regression band

> Working, H. and Hotelling, H. (1929). "Applications of the Theory of Error to
> the Interpretation of Trends." *Journal of the American Statistical
> Association*, 24(165A), 73–85.

The origin of the hyperbolic simultaneous confidence band for a regression line
that Ma & Hall (A.6) transplant onto the probit–probit ROC. Cite for the
genealogy of the `working_hotelling` baseline.

### B.10 Log-concave density estimation — the Hsieh–Turnbull comparator's engine

> Dümbgen, L. and Rufibach, K. (2009). "Maximum Likelihood Estimation of a
> Log-Concave Density and Its Distribution Function." *Bernoulli*, 15(1), 40–68.
>
> Walther, G. (2002). "Detecting the Presence of Mixing with Multiscale
> Maximum Likelihood." *JASA*, 97(458), 508–513.
>
> Rufibach, K. (2007). "Computing Maximum Likelihood Estimators of a
> Log-Concave Density Function." *J. Statistical Computation and Simulation*,
> 77(7), 561–574. (R package `logcondens`.)
>
> Application to ROC: "A Smooth ROC Curve Estimator Based on Log-Concave
> Density Estimates." arXiv:1103.1787 (Rufibach, 2012, *Int. J. Biostatistics*).
> https://arxiv.org/pdf/1103.1787

The semiparametric density-estimation machinery behind the Hsieh–Turnbull
log-concave band discussed as the best-calibrated-but-fragile comparator
(`project_evaluation_report.md` C.4, D.7; `theoretical_behavior_report.md` §9).
Dümbgen & Rufibach establish existence/uniqueness and theory for the univariate
log-concave MLE; the ROC application plugs these density estimates into the
Hsieh–Turnbull variance formula (A.1) to estimate the slope term g(c)/f(c).
Cite to explain both why HT-log-concave is principled (it targets the exact
asymptotic variance) and why it fails on bimodal/heavy-tailed DGPs (log-
concavity is violated), motivating the assumption-free bootstrap route.

### B.11 Haldane (1956) / Anscombe (1956) — the continuity correction for logits

> Haldane, J. B. S. (1956). "The Estimation and Significance of the Logarithm of
> a Ratio of Frequencies." *Annals of Human Genetics*, 20(4), 309–311.
>
> Anscombe, F. J. (1956). "On Estimating Binomial Response Relations."
> *Biometrika*, 43(3–4), 461–464.

The +0.5 cell correction used in the optional logit-space construction
(`nonparam_envelope.md` §2.9, §6.1): logit_H(p) = log((k+0.5)/(n₁−k+0.5)).
Cite for the standard device that keeps the logit finite at TPR = 0 or 1. The
paper's own finding is negative — the logit path under-covers badly
(`project_evaluation_report.md` D.6) — so this is cited to document the variant,
not to endorse it.

### B.12 Equal-local-level (tail-sensitive) one-sample bands — the M3 class band

> Berk, R. H. and Jones, D. H. (1979). "Goodness-of-Fit Test Statistics that
> Dominate the Kolmogorov Statistics." *Zeitschrift für
> Wahrscheinlichkeitstheorie und verwandte Gebiete*, 47, 47–59.
> **[verify pages]**
>
> Owen, A. B. (1995). "Nonparametric Likelihood Confidence Bands for a
> Distribution Function." *Journal of the American Statistical Association*,
> 90(430), 516–521. https://doi.org/10.1080/01621459.1995.10476543
>
> Aldor-Noiman, S., Brown, L. D., Buja, A., Rolke, W. and Stine, R. A. (2013).
> "The Power to See: A New Graphical Test of Normality." *The American
> Statistician*, 67(4), 249–260. https://doi.org/10.1080/00031305.2013.847865
> **[verify author order — sources differ]**
>
> Weine, E., McPeek, M. S. and Abney, M. (2023). "Application of Equal Local
> Levels to Improve Q-Q Plot Testing Bands with R Package qqconf." *Journal of
> Statistical Software*, 106(10). https://doi.org/10.18637/jss.v106.i10
> **[verify author list]**
>
> Dümbgen, L. and Wellner, J. A. (2023). "A New Approach to Tests and
> Confidence Bands for Distribution Functions." *The Annals of Statistics*,
> 51(1), 260–289. https://doi.org/10.1214/22-AOS2249
>
> Learned-Miller, E. and DeStefano, J. (2008). "A Probabilistic Upper Bound on
> Differential Entropy." *IEEE Transactions on Information Theory*, 54(11),
> 5223–5230. **[verify]**
>
> Lourie, N., Cho, K. and He, H. (2024). "Show Your Work with Confidence:
> Confidence Bands for Tuning Curves." NAACL 2024; arXiv:2311.09480.
> https://arxiv.org/abs/2311.09480 **[verify proceedings pages]**

The one-sample band each class receives inside the M3 band (`m3_band_rs.py`):
the same local level γ at every order statistic, bounds
BetaInv(γ; i, n+1−i) and BetaInv(1−γ; i, n+1−i), and γ calibrated so the
simultaneous level is exact. Berk & Jones introduced the statistic; Owen
inverted it into a confidence band for a distribution function ("narrower in
the tails and wider in the center" than KS); Aldor-Noiman et al. used the
equal-local-level construction as "tail-sensitive" Q-Q bands; Weine, McPeek &
Abney named it "equal local levels" and give fast, accurate calibration in
`qqconf`. Dümbgen & Wellner refine the Berk–Jones/Owen bands in the centre.
Learned-Miller & DeStefano and Lourie et al. build the closely related
equal-violation-rate CDF band from Beta(i, n+1−i) intervals, calibrated by
Monte Carlo rather than an exact recursion. None of these papers concerns ROC
curves; the class band is therefore cited as established, and only its use
inside an ROC composition is claimed as new (§D).

**Other exact one-sample bands that could replace ELL in the composition.**
These define the comparison class for a "narrowest exact band" claim, since
each can be calibrated by the same crossing-probability recursion (B.13):

> Frey, J. (2008). "Optimal Distribution-Free Confidence Bands for a
> Distribution Function." *Journal of Statistical Planning and Inference*,
> 138(10), 3086–3098. https://doi.org/10.1016/j.jspi.2007.12.001
>
> Xu, X., Ding, X. and Zhao, S. (2009). "The Reduction of the Average Width
> of Confidence Bands for an Unknown Continuous Distribution Function."
> *Journal of Statistical Computation and Simulation*, 79, 335–347; and "A
> New Confidence Band for Continuous Cumulative Distribution Functions,"
> *Australian & New Zealand Journal of Statistics*, 51, 305–318.
> **[abstracts not read]**
>
> Dufour, J.-M. and Diouf (2026). CIRANO Working Paper 2026s-15. **[verify
> title and first name; summary only read]**

Frey gives bands that are optimal under narrowness or area criteria (no
public code found) and is the likeliest to be narrower than ELL in the middle
of the ROC. Dümbgen & Wellner (above) tabulate critical values in their
supplement and could be calibrated exactly, but no package was found. Dufour
and Diouf study regularized Kolmogorov, Anderson–Darling, Eicker, and
Berk–Jones families with finite-sample Monte Carlo calibration. Owen's band
ships in R as `SMIR::NPL.bands` at the 0.95 and 0.99 levels only. The
time-uniform Dvoretzky–Kiefer–Wolfowitz band of Howard & Ramdas (2022,
*Bernoulli* 28(3), 1704–1728) is about twice as wide as the fixed-n band and
is not a competitor at a fixed sample size. No published width comparison of
ELL against these was found beyond Owen's qualitative statement.

### B.13 Exact boundary-crossing probabilities — the M3 calibration kernel

> Noé, M. (1972). "The Calculation of Distributions of Two-Sided
> Kolmogorov–Smirnov Type Statistics." *The Annals of Mathematical
> Statistics*, 43(1), 58–64. **[verify]**
>
> Steck, G. P. (1971). "Rectangle Probabilities for Uniform Order Statistics
> and the Probability That the Empirical Distribution Function Lies Between Two
> Distribution Functions." *The Annals of Mathematical Statistics*, 42(1),
> 1–11. **[verify]**
>
> Moscovich, A., Nadler, B. and Spiegelman, C. (2016). "On the Exact Berk–Jones
> Statistics and Their p-Value Calculation." *Electronic Journal of
> Statistics*, 10(2), 2329–2354. https://doi.org/10.1214/16-EJS1172
>
> Moscovich, A. and Nadler, B. (2017). "Fast Calculation of Boundary Crossing
> Probabilities for Poisson Processes." *Statistics & Probability Letters*,
> 123, 177–182. https://doi.org/10.1016/j.spl.2016.11.027

The exact non-crossing probability of uniform order statistics between two
boundaries, which the `fiducial_core` kernel computes to set γ without Monte
Carlo. Noé's recursion and Steck's determinant are the classical exact
methods (Owen 1995 used Noé's); Moscovich, Nadler & Spiegelman give the exact
Berk–Jones p-values and Moscovich & Nadler an O(n² log n) two-sided algorithm.
Cite for the claim that M3's simultaneous level is exact to roundoff. Public
code: `crossprob` (C++ with a Python module,
github.com/mosco/crossing-probability).

### B.14 Fiducial bands and rank-depth trimming — antecedents of the fiducial band

> Cui, Y. and Hannig, J. (2019). "Nonparametric Generalized Fiducial
> Inference for Survival Functions under Censoring." *Biometrika*, 106(3),
> 501–518. arXiv:1707.05034.
>
> Myllymäki, M., Mrkvička, T., Grabarnik, P., Seijo, H. and Hahn, U. (2017).
> "Global Envelope Tests for Spatial Processes." *Journal of the Royal
> Statistical Society: Series B*, 79(2), 381–404. R package `GET`.
> **[verify volume and pages]**
>
> Montiel Olea, J. L. and Plagborg-Møller, M. (2019). "Simultaneous
> Confidence Bands: Theory, Implementation, and an Application to SVARs."
> *Journal of Applied Econometrics*, 34(1), 1–17.
> https://doi.org/10.1002/jae.2656

Cui and Hannig is the closest nonparametric fiducial band (one-sample
survival curves, with a functional Bernstein–von Mises theorem); no
generalized fiducial ROC band was found, and the only "generalized
inference" ROC band is binormal (A.17). The fiducial band's trim by minimum
pointwise rank is the rank-envelope idea of Myllymäki et al., whose `GET`
package provides general functional envelopes; no application to ROC curves
was found. Montiel Olea and Plagborg-Møller argue the sup-t band is the
narrowest valid default in linear models and is the generic calibration a
referee may ask to see applied to the ROC; PSN (A.10) with a naive bootstrap
and an equal split is essentially that band. See A.7 for the Bayesian
bootstrap, the nearest resampling relative.

---

## C. Suggested citation map (claim → reference)

| Claim / component (source doc) | Primary references |
|---|---|
| Asymptotic ROC process, two-component variance (Thm 1; B.6, D.9) | Hsieh & Turnbull 1996 (A.1) |
| Studentized bootstrap sup-band validity (B.3) | Hall & Horowitz 2013; Hall 1993; Horowitz 2001 (B.3) |
| Naive bootstrap deficiency at the ROC tail / smoothing (B.5) | Bertail, Clémençon & Vayatis 2008 (A.4) |
| Wilson variance floor & rectangle band (§2.4, §2.7) | Wilson 1927; Agresti & Coull 1998; Newcombe 1998 (B.1–B.2) |
| Exact Beta order-statistic floor (§2.7, §3.4) | Order statistics / PIT: David & Nagaraja 2003 (B.6) |
| KS fixed-width band & boundary extension (§2.8, `ks`) | Campbell 1994 (A.2); DKW 1956 / Massart 1990 (B.5) |
| Working–Hotelling baseline (`working_hotelling`) | Ma & Hall 1993 (A.6); Working & Hotelling 1929 (B.9) |
| Ellipse-envelope comparator (C.3) | Demidenko 2012 (A.5) |
| Hsieh–Turnbull log-concave comparator (C.4, D.7) | Dümbgen & Rufibach 2009; Rufibach 2012 (B.10) |
| Šidák / Bonferroni multiplicity (§2.7) | Šidák 1967; Miller 1981 (B.7) |
| LHS + Iman–Conover design (`simulation_spec.md` §3) | McKay et al. 1979; Iman & Conover 1982; Stein 1987 (B.8) |
| Logit-space Haldane–Anscombe correction (§2.9) | Haldane 1956; Anscombe 1956 (B.11) |
| Prior ROC-band survey / pointwise-vs-simultaneous framing | Macskassy & Provost 2004; Macskassy et al. 2005 (A.3) |
| Prior nonparametric *simultaneous* ROC bands (closest competitors) | Jensen, Müller & Schäfer 2000 (A.11); Martínez-Camblor et al. 2018 (A.10) |
| Conformal distribution-free alternative (different estimand) | Zheng, Yang & Song 2024 (A.8) |
| Bayesian-bootstrap alternative | Gu, Ghosal & Roy 2008 (A.7) |
| Exact composition (M3) band: per-class bands composed by monotonicity | Campbell 1994 (A.2); Hilgers 1991 (A.12, pointwise) |
| M3 class band: equal local levels / tail-sensitive Beta bounds | Berk & Jones 1979; Owen 1995; Aldor-Noiman et al. 2013; Weine et al. 2023 (B.12) |
| M3 exact simultaneous-level calibration | Noé 1972; Steck 1971; Moscovich et al. 2016; Moscovich & Nadler 2017 (B.13) |
| Priority for composing one-sample bands; conservativeness of the product level | Doksum 1974; Switzer 1976 (A.14) |
| Exact two-sample bands on the score (Q-Q) scale | Switzer 1976; Doksum & Sievers 1976 (A.14) |
| Alternative exact class bands for the composition | Frey 2008; Xu, Ding & Zhao 2009; Dümbgen & Wellner 2023 (B.12) |
| Exact but pointwise operating-point bounds | Hilgers 1991; Schäfer 1994 (A.12); Tong, Feng & Li 2018 (A.18) |
| Fiducial band: nearest resampling and fiducial antecedents | Gu, Ghosal & Roy 2008 (A.7); Cui & Hannig 2019; Myllymäki et al. 2017 (B.14) |
| Shipped simultaneous ROC bands | nsROC (A.9–A.11); `EL` (A.16); Stata `rocplot, confband` (A.17) |
| Mainstream software gives pointwise bands only | §G |
| Finite-sample ROC bands under a VC restriction; impossibility | Shi, Wang & Ma 2026 (A.19) |

---

## D. Prior art for the exact composition (M3) band

A targeted search (November 2026) for any published ROC, ODC, or two-sample
P-P band built from equal-local-level, Berk–Jones, or tail-sensitive per-class
bands found none. What is published:

- **The composition.** Campbell (A.2) composes two one-sample KS bands into an
  ROC band by monotonicity; Hilgers (A.12) composes exact per-class
  tolerance-type bounds into an ROC rectangle at one cut-off.
- **The class band.** Equal-local-level / tail-sensitive Beta order-statistic
  bands with exact simultaneous level (B.12), computed by exact
  boundary-crossing recursions (B.13), all for one-sample CDF or Q-Q problems.
- **Other ROC bands.** Every other simultaneous ROC band found is asymptotic,
  bootstrap, Monte Carlo, or parametric (A.1, A.3–A.11), and the nearest recent
  rank-process work (A.13) builds no band at all.

**Defensible claim.** The M3 band is Campbell's composition with
equal-local-level bands in place of KS bands, which no published work found
does. Also not found in print: the α split across classes (Šidák or
sample-size ratio, B.7), exact pivots that remain valid when the ROC jumps,
exact tie handling by random tie-breaking, the refusal to pin U(0) = 0, and its
use as a floor under the fiducial band (the hybrid). Present the class band and
its calibration as cited machinery and these as the contribution.

**Second search (19 November 2026).** Three wider searches (exact
distribution-free bands; asymptotic, bootstrap, and Bayesian bands and
shipped software; machine-learning and conformal work) again found no ROC,
ODC, or P-P band built from equal-local-level class bands. They did find
prior art that changes the framing:

- Composition itself is older than Campbell: Doksum 1974, with Switzer 1976
  showing how loose the product-level guarantee is (A.14). The contribution
  should be stated as the choice of class band, its exact calibration, the
  class split, and the treatment of jumps, ties, and U(0), not as the
  composition.
- Exact two-sample bands exist on the score scale (Switzer; Doksum–Sievers).
  The paper needs a paragraph on why they are not ROC-coordinate bands
  (A.14).
- Goldman & Kaplan 2018 (A.20) use two-sample Beta-quantile bands at a common
  local level, for testing only.
- Shi, Wang & Ma 2026 (A.19) claim finite-sample trade-off-function bands
  under a VC restriction; unread beyond the abstract.
- For the fiducial band, the Bayesian bootstrap ROC (A.7) is close prior art.

**Limits of the search.** Web search over titles and abstracts, plus full
reads of A.13, Lourie et al. (B.12), Switzer 1976, Goldman & Kaplan 2018,
the PSN preprint, Bertail et al., and Zheng et al.; Hilgers (A.12) was read
from its abstract only. Entries added on 19 November 2026 were verified by
search agents, not re-read by the authors; details marked **[verify]** were
not confirmed at all. Before claiming novelty in print, check the citing
literature of Campbell 1994 and Owen 1995 (Google Scholar "cited by") for an
ROC or two-sample application, and the full texts of Hilgers 1991, Gu &
Ghosal 2008, and Shi, Wang & Ma 2026.

---

## E. Is the conformal band (A.8) extensible to a population confidence band?

This is the right question to ask of Zheng, Yang & Song (2024), and the honest
answer has three layers — useful as a related-work paragraph because each layer
strengthens the case for the present method rather than dismissing the conformal
one.

1. **A valid prediction-flavored band can be *used* as a conservative confidence
   band — but only loosely.** A band sized to cover a future/finite-test ROC
   carries the sampling variability of a *second* empirical curve, which is
   roughly twice that of R̂ − R_true (two independent samples;
   `nonparam_envelope.md` §3.1). So such a band, reinterpreted as a band for
   R_true, **over-covers** — it is conservative, hence valid but inefficient.
   You cannot run the implication the other way (tighten a confidence band into
   a prediction band) without modeling the variance split. So "could they build
   a confidence band?" — in this weak sense their construction already yields
   one, just an inefficient one, and it is not what they claim to target.

2. **Re-deriving an *efficient, exact* conformal confidence band for R_true is
   not a free parameter flip, and it hits the same wall this paper is built to
   solve.** Conformal validity rests on exchangeability of observed data and
   covers a quantity tied to those exchangeable draws. The population ROC at low
   FPR depends on the **extreme tail of the negative score distribution**, where
   no finite sample observes beyond its maximum. No exchangeability/resampling
   device — bootstrap or conformal — manufactures information about population
   mass outside the observed support (`project_evaluation_report.md` B.5). A
   conformal confidence band would therefore be vacuous or extrapolating at
   exactly the first few order statistics where this method's Beta floor does
   its work. Conformal has no special escape from the boundary problem.

3. **The one device that *does* give a finite-sample distribution-free guarantee
   at that corner is the order-statistic argument — which is conformal in
   spirit.** The Beta order-statistic floor rests on the probability integral
   transform and the exact law F̄(X₍ⱼ₎) ∼ Beta(j, n₀+1−j) (B.6) — the same
   exchangeability/rank foundation conformal prediction uses. It can fairly be
   described as delivering the "conformal-grade," finite-sample, distribution-
   free guarantee at the low-FPR tail that the Zheng construction (asymptotic,
   marginal, prediction-targeted) does not. That is the sharp contrast to make:
   not "conformal is the wrong tool," but "the method already carries a
   conformal-strength guarantee exactly where the conformal ROC paper is
   weakest, and a calibrated simultaneous envelope everywhere else."

**Bottom line:** the method is extensible in *spirit* but not in its specific,
published construction; redirecting it at R_true forfeits the clean
finite-sample guarantee (target becomes a fixed parameter, not an exchangeable
draw) and runs into the universal extreme-tail obstruction. Treat it as an
adjacent-estimand contemporary, not a benchmark competitor.

---

## F. Open verification items before submission

- ~~Campbell 1994 page range~~ **Resolved:** *Statistics in Medicine* 13,
  499–508 (A.2).
- ~~PSN paper volume and year~~ **Resolved:** *SMMR* 27(6), 1892–1908, 2018,
  online 2016 (A.10).
- Read in full: Gu, Ghosal & Roy 2008 and Gu & Ghosal 2008 (are their bands
  simultaneous; how do the weights differ from n + 1 spacings), Switzer 1976
  (confirm the score-scale versus ROC-scale argument of A.14), Shi, Wang & Ma
  2026 (A.19), and Jensen et al. 2000 (which quantile defines the critical
  value, A.11).
- Validate the nsROC DEK implementation against Demidenko's reported coverage
  before using it as a baseline (A.10).
- Do not copy the reference list of Bertail et al. 2008 for "ROC confidence
  bands: an empirical evaluation"; it misattributes the paper to Girling
  (2000).
- Bibliographic confirmations from the second search: Zou, Hall & Shapiro
  1997 is *Statistics in Medicine* 16(19), 2143–2156; Owen's band is 1995
  (the `SMIR` documentation says 1997); Aly, Csörgő & Horváth 1987 is a
  chapter in *New Perspectives in Theoretical and Applied Statistics*
  (Wiley), pp. 135–156; the smoothed empirical likelihood paper for censored
  ROC data is Yang & Zhao (2012), *Journal of Multivariate Analysis* 109,
  254–263; Macskassy & Provost 2004 is the workshop paper and the ICML 2005
  paper adds Rosset.
- Not found, so do not cite without a source: a kernel-smoothed simultaneous
  ROC band by Lijian Yang and coauthors; "Branscum, Johnson, Hanson &
  Gardner 2008, *Statistics in Medicine*"; a "Stepanova & Wang" band paper;
  any time-uniform ROC or AUC confidence sequence.
- The project's repositories and the `rocci` PyPI page are publicly indexed,
  which matters for double-blind review.
- ~~Authors of arXiv:2405.12953~~ **Resolved:** Zheng, Yang & Song (2024),
  Univ. Michigan Biostatistics (A.8).
- **[verify]** Martínez-Camblor et al. PSN paper: 2016 online-first vs. 2018
  print, and *Statistical Methods in Medical Research* volume/pages (A.10).
- **[verify]** Rufibach log-concave ROC paper year/venue (arXiv 1103.1787 →
  *International Journal of Biostatistics*).
- Confirm whether the paper cites the multidimensional log-concave MLE (Cule,
  Samworth & Stewart 2010) — only the univariate estimator is used here.
- **[verify]** The **[verify]**-flagged details in A.12, B.12, and B.13 (Hilgers
  first name and DOI; Berk & Jones pages; Aldor-Noiman et al. author order;
  Weine et al. author list; Learned-Miller & DeStefano; Noé 1972 and Steck 1971
  issue and pages).
- Run the citing-literature check of §D (Campbell 1994, Owen 1995, Hilgers
  1991) before stating that the exact composition band is new.
- Consider adding Pepe (2003), *The Statistical Evaluation of Medical Tests*
  (Oxford), as the standard textbook anchor for ROC inference if a general
  reference is wanted.

---

## G. What released software computes (searched 19 November 2026)

Entries were read from package documentation or source by a search agent
unless marked otherwise. "Pointwise" means a separate interval at each FPR or
threshold.

| Software | Function | Output | Simultaneous band for the population ROC? |
|---|---|---|---|
| R `nsROC` | `ROCbands(method = "PSN" / "JMS" / "DEK")` | Bands on the grid {0, 1/N, …, 1} | Yes (A.5, A.10, A.11) |
| R `EL` | `EL.plot(method = "roc", simultaneous = TRUE)` | Smoothed empirical-likelihood band | Yes, on an interval trimmed 5% at each end (A.16) |
| Stata | `rocfit` then `rocplot, confband` | Ma–Hall Working–Hotelling band | Yes, binormal (A.6) |
| R `tram` | Sewak–Hothorn score bands | Transformation-model band | Yes, model-based (A.17) |
| MATLAB PRT | `prtScoreRocBayesianBootstrap` | "Uniform credible band", undefined | Claimed (A.7) |
| R `stats` | `qqplot(conf.level = )` | Exact Smirnov band for the Q-Q function | Score scale, not ROC coordinates (A.14) |
| R `pROC` | `ci.se`, `ci.sp`, `ci.coords` | Stratified percentile bootstrap | No, pointwise |
| R `fbroc` | `conf` | Percentile bootstrap TPR on an FPR grid | No, pointwise |
| R `ROCit` | `ciROC` | Delta method or binormal bootstrap per threshold | No, pointwise |
| R `plotROC` | `geom_rocci` | Clopper–Pearson rectangles at chosen cutoffs | No, pointwise |
| R `ROCnReg` | `pooledROC.BB`, `pooledROC.dpm` | Pointwise credible bands | No |
| R `precrec`, `mlr3viz` | `evalmod(cb_alpha = )` | Bounds across multiple test sets | No, pointwise |
| R `verification` | `roc.plot(CI = TRUE)` | Bootstrap boxes at thresholds | No |
| R `logcondens` | `confIntBootLogConROC_t0` | Bootstrap interval at one FPR | No |
| R, PyPI `nproc` | NP-ROC bands | Exact order-statistic bounds per threshold | No, pointwise (A.18) |
| MATLAB | `perfcurve` (`NBoot`, `XVals`, `TVals`) | Bootstrap bounds; the documentation says they are pointwise and not a simultaneous band | No |
| Stata | `rocreg` | Bootstrap intervals at listed FPRs | No |
| SAS | `PROC LOGISTIC` | AUC intervals | No curve band found |
| MedCalc | ROC "confidence bounds" | Hilgers rectangles | No, pointwise (A.12) |
| scikit-learn | `RocCurveDisplay`; `plot_roc_crossval` example | No interval; the example shades ±1 SD across folds | No |
| Python `mmu` | Profile-likelihood regions | Per-threshold regions and their union | No, pointwise (A.18) |
| Python `roc-utils` | `plot_roc_bootstrap`, `plot_mean_roc` | Percentile ribbon; a "CI of the mean curve" that shrinks with the number of bootstrap draws | No |
| Python `confidenceinterval`, `MLstatkit`, `roc_comparison`; HF `evaluate`; `torchmetrics` | AUC or scalar-metric intervals | — | No |

Not opened, and no band known: `ROCR`, `cutpointr`, `OptimalCutpoints`,
`yardstick`, `WeightedROC`, `sROC`, `statsmodels`, `scikit-plot`, TorchEval,
Keras, `mlxtend`, ROCKIT/LABROC.

Evidence on reporting practice, for the motivation section: Király, Mateen &
Sonabend (arXiv:1812.07519) found that 32% of 121 NeurIPS 2017 papers
reported confidence intervals and none said how they were computed;
Christodoulou et al. (MICCAI 2024, arXiv:2409.17763) found intervals in 0.5%
of 221 MICCAI 2023 segmentation papers; Agarwal et al. (NeurIPS 2021,
`rliable`) popularised pointwise bootstrap bands on performance profiles;
Lourie et al. (B.12) show bootstrap bands under-cover for tuning curves.
