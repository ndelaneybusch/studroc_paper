# Next ROC bands: the strongest candidates

*Working assessment, 2026-09-05. Incorporates the completed Stage F study,
the fiducial counterexamples, the production M3 construction, and a new
paired experiment. Recommendations are judgments; guarantees, measurements,
and conjectures are identified separately.*

The best near-term empirical band is the **C = 1 fiducial band with a
localized exact floor**. The best established finite-sample guarantee is
**M3**. Keep both. Neither yet satisfies the full objective: the hybrid has
no uniform bound on its remaining interior failures, while M3 spends much
more width than its realized ROC coverage appears to need.

The largest research opportunity is **direct inference on the ROC rank
experiment**. M3 protects two entire CDFs before projecting them onto one
curve. Fiducial composition approximately combines the errors directly, but
completes unobserved gaps too confidently. A stronger method should combine
the two sampling errors without making that completion assumption.

## 1. What we are optimizing

The primary requirement is simultaneous population coverage,
$P_R\{L(t)\le R(t)\le U(t)\ \forall t\}\ge1-\alpha$.
There are two separate goals here: **honesty**, meaning that lower bound
holds throughout the stated distribution class, and **calibration**, meaning
coverage is also reasonably close to $1-\alpha$. M3 meets the first and
misses the second badly at large alpha.

“Distribution-free” should mean validity over arbitrary class score laws,
with independent iid observations within each class and a scoring rule fixed
independently of the evaluation sample. No finite-moment, Gaussian-tail,
positive-density, concavity, or smoothness assumption should be implicit.
Dependence and reuse of training observations are different sampling problems.

Several distinctions determine whether a candidate actually meets the goals:

- **Rank invariance is not uniform coverage.** Coverage depends on the ROC
  shape rather than on its score representation. It can still vary sharply
  across shapes, including shapes with the same AUC.
- **The whole curve includes difficult endpoints and support gaps.**
  Continuous score CDFs need not produce a continuous ROC. Positive mass
  beyond the negative support can give $R(0)>0$; never impose $U(0)=0$
  without a support assumption. Use the existing $R(1)=1$ convention.
- **Ties require an estimand.** The natural general-data target is the ROC
  of a threshold rule that randomizes within tied scores, giving linear
  tie segments and the Mann–Whitney AUC. Independent continuous auxiliary
  tie breakers provide a construction to analyze. Deterministic class-ordered
  tie breaking and arbitrary numerical jitter are not interchangeable with it.
  New proofs and experiments must explicitly cover this extension.
- **Grid coverage can imply continuum coverage.** For a monotone truth,
  extend the lower edge from the grid point to the left and the upper edge
  from the point to the right. Straight-line interpolation of band edges
  does not provide that implication for arbitrary ROC shapes.
- **WH and KS are comparators, not mathematical width bounds.** Report
  absolute integrated width and paired ratios to both. WH is an efficiency
  reference only where its model is appropriate; an oracle-calibrated
  experimental arm is a benchmark for its construction, not a universal
  lower bound on attainable width.
- **Balance is a design objective, not an automatic consequence of
  two-sidedness.** Measure $P(\exists t:R(t)<L(t))$ and
  $P(\exists t:R(t)>U(t))$, allowing a replicate to contribute to both.
  Across FPR, measure both pointwise miss rates and the probability of any
  miss in each fixed FPR interval. Equal local levels do not equalize
  intervalwise failures when correlation lengths differ.

Literal equal error rates at every FPR are incompatible with known endpoints:
an honest band can have zero error at $t=1$. The useful target is approximate
balance on the nondegenerate interior, with separate tail diagnostics.
Minimum area and equal regional error rates can also conflict. Prefer a
Pareto comparison to hiding that choice inside one score.

## 2. The evidence that should drive the next method

The original [envelope specification](nonparam_envelope.md) and
[evaluation report](project_evaluation_report.md) establish a recurring
failure mechanism: a bootstrap or interpolation convention cannot recover
unseen tail mass merely by improving its estimated variance. The report's
pre-Beta-floor balanced-class coverage fell to .830 at total $n=10{,}000$;
the subsequent floor repaired the tested problem strata. These are different
versions of the band and should not be pooled into a single coverage claim.
The README's broad assertions are not a substitute for this provenance.

The [fiducial theory](fiducial_band_theory.md), especially §§3.1 and 7.4,
identifies the analogous problem in the Dirichlet construction: exact
class-specific spacings do not specify the other class's locations inside
those gaps. Sorted-uniform completion behaves like local ROC linearity.
Convex tail hooks and unobserved slivers can invalidate the resulting band.

The [boundary follow-up](c_calibration_followup_report.md) located a curved,
nonmonotone failure wedge in AUC and sample size. More importantly, the sliver
construction and its subsequent prospective replication show that **the
problem is not confined to high AUC or small samples**. AUC and class counts
alone cannot certify that the fiducial band is safe.

The completed [Stage F report](hybrid_floor_report.md) changes the practical
assessment:

| Study, at alpha = .05 | Cells | C = 1 macro coverage | Frontier hybrid | Hybrid minimum | Hybrid width cost versus C = 1 |
|---|---:|---:|---:|---:|---:|
| A: enriched replay and stress | 116 | .9256 | .9843 | .940 | +11.3% |
| B: prospective adversarial transfer | 30 | .8294 | .9823 | .965 | +9.1% |
| C: seven shapes at two sizes | 14 | .9679 | .9804 | .970 | +15.5% |

These are cell-macro summaries of different, mostly adversarial designs,
not population averages over “all ROC curves.” There were 42,000 paired
replicates. No cell was significantly below nominal by the report's
interval check, but the worst cell had only 200 replicates; this neither
establishes uniform honesty nor excludes a material deficit there.

The mechanism evidence is stronger than the pooled coverage:

- On six fresh sliver cells, raw fiducial coverage was .505–.613; the hybrid
  covered .978–.988. Conditional on missing the sliver, raw coverage was
  zero observed in five cells. The saturated-run trigger reacted on the
  predicted realizations.
- Lower-edge failing replicates fell from 1,413 to 130 in A and from 1,990
  to 98 in B. Hybrid upper-edge counts were 234 and 137. The correction
  largely removes the severe directional imbalance, with a residual tilt
  toward upper-edge failure.
- Residual violations are mostly far inside the unprotected region,
  with median distances of .16–.31 of the FPR range from the floor.
  Enlarging a small boundary margin is therefore poorly targeted.
- At alpha = .5, **Study A** moves from .584 coverage for C = 1 to .781 for
  the hybrid, at +19.7% width. Tail repair has not solved calibration.
- At small samples, already-safe shapes can pay heavily: one
  beta-opposing cell pays +48.8% width for .993 → .995 coverage.

A post-hoc left-cutoff sweep suggests that about seven grid points might
retain most protection at lower cost than the budget-dependent nine or ten.
That is a candidate for confirmation, not a validated replacement.
The $2\sqrt K$ right margin changed one coverage outcome in 23,200 A
replicates relative to no margin. That makes it a low-priority tuning lever,
not proof that it can be removed.

## 3. Shortlist and priorities

| Candidate | Status of guarantee | Expected width opportunity | Balance and alpha calibration | Speed / priority |
|---|---|---|---|---|
| **M3 with optimized deterministic marginal bands and class split** | Existing finite-sample proof carries through | Modest, credible gains; especially imbalance and near-diagonal ROCs | Limited ability to cure projection overcoverage | Fast after calibration; first exact implementation experiment |
| **Direct rank-test inversion, with certified outer projection** | Exact in principle; general computation unresolved | Largest plausible improvement over M3 | Can target both signs and FPR regions directly | Main research investment |
| **Fiducial band with localized exact protection and bracket completion** | Empirically strong; no general whole-band theorem | Best demonstrated practical compromise | Tail balance much improved; central-alpha surplus remains | Retain as empirical incumbent; targeted refinement |
| **Rank-likelihood e-value inversion** | A direct finite-sample argument is available | Uncertain; likelihood/prediction penalties may be costly | One global budget; calibration may still be conservative | Bounded exploratory project |
| **Direct ROC-process calibration with exact tails** | Interior asymptotics under explicit regularity | Plausible width and central-alpha gains | Most direct approximate control of local error shape | Secondary route if approximate validity is acceptable |

M3 itself remains the exact reference. Unmodified C = 1 remains a useful
component and ablation, but not a universally valid standalone choice.
Joint pivotal regions are a principled M3 extension; the first experiment
below lowers their priority as a general replacement.

## 4. M3: keep it, then optimize the part we can certify

### 4.1 Current construction and guarantee

For each class of size $n$, the true class survival probabilities at its
descending order statistics have the joint law of uniform order statistics.
M3 places bounds

$$
b_i^-=\operatorname{Beta}^{-1}(\gamma;i,n+1-i),\qquad
b_i^+=\operatorname{Beta}^{-1}(1-\gamma;i,n+1-i)
$$

around them. The Rust non-crossing dynamic program calibrates $\gamma$
against the **joint** two-sided event. The independent class coverages obey

$$
(1-\alpha_0)(1-\alpha_1)=1-\alpha,\qquad
1-\alpha_0=(1-\alpha)^\rho.
$$

Monotonicity and the observed merged ranks then compose the class bounds
into ROC edges. This is the production
[m3_band_rs.py](../src/studroc_paper/methods/m3_band_rs.py) and Proposition 12
of the theory document. Its class calibration is numerical exact
non-crossing probability, with conservative bisection; it is not the
older Monte Carlo calibration in the round-three harness.

The product of the class coverages is exact under independence. The major
inequality is that **covering both CDFs is sufficient, but not necessary,
to cover their ROC composition**. Choosing a rectangular class region and
then projecting it allows unfavorable horizontal and vertical errors to
combine. Correcting Šidák to another elementary multiplicity formula cannot
eliminate that slack.

Historical M3 experiments covered .978–.998 at nominal .50; Stage F full M3
was about 41–43% wider than C = 1 in A/B at alpha = .05.
M3 often substantially improves on KS, especially at high AUC, but there is
no universal width dominance over KS, particularly near AUC = .5.

### 4.2 Optimize deterministic local levels for ROC width

ELL is an attractive default, not an optimality theorem for integrated ROC
width. The existing exact kernel accepts general monotone lower and upper
order-statistic boundaries. That is a much larger design space:

$$
b_{c,i}^-=\operatorname{Beta}^{-1}(\gamma_c w_{c,i}^-;i,n_c+1-i),\quad
b_{c,i}^+=\operatorname{Beta}^{-1}(1-\gamma_c w_{c,i}^+;i,n_c+1-i).
$$

Fix positive weights from $(n_0,n_1,\alpha)$ and a declared design objective,
then calibrate the actual joint crossing probability. Preserve ordered
boundaries and valid probability levels. The same M3 proof applies.

There are two useful sources of design ideas. Dümbgen and Wellner's
[2023 CDF bands](https://faculty.washington.edu/jonw/JAW-papers/jaw-duembgen-aos.2023.pdf)
refine Berk–Jones/Owen to improve central precision while retaining tail
accuracy, and discuss validity for discontinuous CDFs. Frey's
[optimal-width CDF bands](https://doi.org/10.1016/j.jspi.2007.12.001)
optimize a narrowness criterion directly. Neither result establishes
optimal ROC width after composition; that objective must be evaluated here.

**Proposal.** Optimize a small, fixed family of boundary shapes against
paired *ROC* area, with constraints on tail width and sign/region error
imbalance. Include both KS-shaped and ELL-shaped boundaries and a
Dümbgen–Wellner-inspired alternative. Use the exact probability as a hard
constraint, so a simulation library guides efficiency but is never the
source of validity.

This distinction makes offline optimization much safer than the failed
offline C-remap: a poor design library can give inefficient boundaries,
but cannot invalidate an independently checked universal pivot event.
Do not select the narrowest boundary on the evaluation sample without
accounting for that selection.

### 4.3 Make the class split depend on sample sizes

A useful initial rule is

$$
\rho(n_0,n_1)=
\frac{n_0^{-1/2}}{n_0^{-1/2}+n_1^{-1/2}}.
$$

It assigns more error allowance to the smaller class: $\rho=.75$ for
100/900 and .25 for 900/100. It is fixed conditional on the class counts,
so the theorem is unchanged under the current sampling design.

The new screen found 0.9–5.8% lower paired area at alpha = .05 across
eight imbalanced cells, and 1.9–5.8% at alpha = .5. This is a useful
low-cost candidate. It is not a proof of narrower bands everywhere:
ROC slope can amplify the larger class's horizontal uncertainty.

### 4.4 Joint class regions: valid, but the first screen is mixed

A more substantial change replaces the rectangle by a curved acceptance
region for the two class pivot statistics. Let $p_0,p_1$ be their exact
one-sample tail p-values at the candidate CDFs. Under the true continuous
class laws they are independent uniforms. For example, retain

$$
p_0p_1\ge k_\alpha,\qquad
k_\alpha(1-\log k_\alpha)=\alpha.
$$

This has probability $1-\alpha$. Project the union of compatible class
pairs into a band. Equivalently, take an envelope over compositions at all
class levels $(a,k_\alpha/a)$, $k_\alpha\le a\le1$.

This is an **envelope over allocations**, not the narrowest allocation.
The latter would discard admissible parameter pairs and lose the argument.
A finite staircase that contains the curved region gives a conservative,
computable version; an inner grid approximation has no such guarantee.

The idea can recover some rectangular slack but remains a projection of
two globally protected CDFs. It also permits a relatively extreme error
in either class alone. At steep or flat ROC sections that can cost more
than it saves. The screen below demonstrates precisely this tradeoff.
Keep it as a geometry experiment, not the leading universal replacement.

### 4.5 New paired screen: what is actually supported

The [script](experiments/ideation_exact_20260905.py) and
[results](experiments/res_ideation_exact_20260905.json) use 400 shared
replicates per shape/count combination: 12 combinations, two alpha levels,
five arms, 4,800 generated datasets. Runtime was about 20 seconds.
Shapes are the diagonal, binormal AUC .95, a shifted $t_2$ pair with shift
10, and a piecewise-linear sliver ROC. Sizes are 500/500, 100/900, 900/100.
This is a design screen, not a replication of the full Stage F library.

Paired area ratios to M3 at alpha = .05:

| Shape / counts | Rank weighting | Joint region | Sample-size split |
|---|---:|---:|---:|
| Diagonal, 500/500 | .981 | .927 | 1.000 |
| Binormal .95, 500/500 | 1.009 | .978 | 1.000 |
| Shifted $t_2$, 500/500 | 1.052 | 1.079 | 1.000 |
| Sliver, 500/500 | 1.016 | 1.039 | 1.000 |
| Shifted $t_2$, 100/900 | 1.008 | 1.103 | .942 |
| Shifted $t_2$, 900/100 | 1.052 | 1.116 | .968 |

The weighting used $w_i=[4u_i(1-u_i)]^{1/4}$, $u_i=i/(n+1)$.
It was chosen before the screen, not optimized. Its computed joint pivot
coverage is .950000002 or .500000001. The joint-region arm uses 20 outer
rectangles; its pivot coverage is .954532 or .512786, so the tabulation
itself adds some conservatism.

At alpha = .5, M3 covered .9575–.995 across these cells. Weighted M3
covered .9525–.9975 and the joint arm .9725–1.000. Neither is remotely a
calibration solution. The weight also increased area by 10.9% on the
100/900 heavy-tail cell at this level. These are reasons to optimize
the projected objective carefully and to pursue a more direct construction.

The scratch composition was checked against production M3 and was
bit-identical on 72 randomized cases, including very small and imbalanced
samples. Width comparisons are paired; per-cell coverage with 400
replicates is still too imprecise to certify a 95% method.

## 5. Direct rank-test inversion: the main research candidate

### 5.1 The exact statistical object already exists

For any hypothesized ROC $R_0$, simulate independent negatives
$U_i\sim\mathrm{Uniform}(0,1)$ and positives with CDF $R_0$, and retain
their merged labels. This gives the exact sampling law $P_{R_0}$ of the
observed rank path. Score distributions have disappeared as nuisance
parameters; ROC shape has not.

Given a statistic $T_{R_0}$, a Monte Carlo rank test can use

$$
p_{R_0}(\Lambda)=
\frac{1+\sum_{b=1}^B
  1\{T_{R_0}(\Lambda_b)\ge T_{R_0}(\Lambda)\}}{B+1}.
$$

With iid null paths and a fixed, consistently applied statistic this is
super-uniform, including conservative handling of ties. A jointly
exchangeable global-envelope construction is another valid implementation.
This is the principle behind the existing named-curve test and the
[global-envelope literature](https://arxiv.org/abs/1307.0239).

Invert the tests:

$$
\mathcal C_\alpha(\Lambda)=\{R:p_R(\Lambda)>\alpha\},\qquad
L(t)=\inf_{R\in\mathcal C_\alpha}R(t),\quad
U(t)=\sup_{R\in\mathcal C_\alpha}R(t).
$$

The true curve is excluded only when its own level-alpha test rejects.
**There is no Bonferroni payment for the number of candidate curves or
the number of projected FPR coordinates.** This statement assumes the
test family and the numerical projection are actually valid.

This has the clearest route to finite-sample validity and substantially
better alpha calibration together. It tests the actual two-sample ROC
error, including cancellation, rather than requiring each CDF to be
well estimated on its own.

### 5.2 What to change from the previous M4 work

The failed M4b method calibrated a fiducial trim over a few curves in a
bracket, relying on behavior that was not monotone in the assumed shape
coordinate. That does not refute direct inversion. It rules out treating
two endpoint curves or a finite library as the worst case.

The computational target should be **certified outer projection**:

1. Parameterize boxes of monotone ROC values on a coarse grid, with
   interval-valued monotone completions between grid points. Include
   endpoint mass and support-gap jumps.
2. For a proposed bound at one FPR, search over **all compatible
   completions**. Use the fiducial cloud to propose promising candidates,
   not to limit the parameter space.
3. Exclude a box only if a bound certifies rejection of every curve it
   contains. Otherwise subdivide it or retain its full coordinate range.
4. Stop at a chosen numerical budget and report the outer envelope of
   everything not certified excluded. Incomplete search then costs width.

A finite spline family alone is a distributional restriction. A handful of
accepted simulated curves gives an inner approximation to the confidence
set and is unsafe as a band. The full-domain bounding step, rather than
the named-curve p-value, is the hard new work.

M3 can accelerate the search, but a data-dependent M3 search envelope is
not free. If it has error $\delta$, and rank-test inversion has error
$\alpha-\delta$, their intersection has total error at most $\alpha$.
Alternatively search the full monotone class and avoid that extra budget.

### 5.3 Choose a statistic for the desired band

Start with a signed, locally scaled maximum ROC discrepancy and a small
number of fixed FPR regions. Calibrate its **joint** null distribution at
each candidate $R_0$. A null-specific multiscale or regional correction can
be computed from $R_0$; that uses the hypothesis, not a fitted surrogate
truth. If simulations train the correction, use a separate null training
cloud or an exchangeable construction for the final test.

Compare this with a rank likelihood-ratio statistic. The latter may use more
of the observed interleaving information but could leave a confidence set
whose coordinate envelope is unnecessarily wide. Optimize the reported
band, not merely power against AUC shifts.

Even exact confidence-set coverage need not give exact **band** coverage:
a rejected curve can still lie inside the envelope of accepted curves.
Direct inversion removes a major source of slack, not necessarily all
projection conservatism. Near-diagonal and high-alpha screens should
measure this explicitly.

### 5.4 First bounded experiment

At small class sizes, enumerate all $\binom{n_0+n_1}{n_0}$ label paths.
Use a few-knot ROC representation with interval completions and compare
exact-path tests against M3 on smooth, sliver, jump, and diagonal truths.
First establish that the bounding algorithm encloses the full set in
small problems; then ask whether its width gain survives conservative
completion.

Do not scale this project because named-curve tests work: that part is
already known. Scale it if **certified projection** is tractable and
meaningfully narrower. Direct inversion is the strongest conceptual
candidate, with a substantial computational risk.

## 6. Fiducial bands: retain the empirical strength, repair the uncertainty

### 6.1 Current C = 1 band

Sort scores into one merged label sequence. Draw independent
Dirichlet$(1,\ldots,1)$ spacings, with $n_c+1$ gaps for each class.
Complete the other class's within-gap positions using sorted uniforms,
compose the resulting CDFs into ROC draws, and use a two-sided
equal-local-level/minimum-rank envelope of that cloud.

The nominal trimming map is
$\alpha_{\rm eff}=1-(1-\alpha)^C$, with **C = 1**.
The construction includes its CP-form upper allowance and zero lower
edge where the empirical positive count is zero. Those allowances widen
the band; their data-selected local level does not make them an
independent whole-band certificate.

This gives a compact, rank-invariant uncertainty construction and much
better empirical width than M3 on many shapes. Exact spacings and exact
within-cloud content do **not** imply exact frequentist coverage for the
composed band. A globally fixed C > 1 is not a distribution-free fix for
the central-alpha conservatism.

### 6.2 Retain the localized M3 hybrid

For a rank-selected region $A$, use the interval hull of the fiducial and
M3 bands on $A$ and widening-only monotone closure. The Stage F rule
protects a short left prefix and the empirical-TPR-1 run extended inward.

For any such selection,

$$
P(\text{hybrid miss})
\le \alpha_{\rm tail}
 +P(\text{fiducial miss somewhere outside }A).
$$

The regional term is valid because a hybrid miss inside $A$ implies a
full M3 miss, even though $A$ was selected from the data. The second term
has no uniform bound. Taking $\alpha_{\rm tail}=\alpha/2$ does not create
a theorem unless that exterior term is also at most $\alpha/2$.

**Best short refinements.** Confirm a theory-linked left cutoff near the
observed six-to-seven-point transition, and separate alpha-dependent
statistical protection from the Monte Carlo budget M. Quantify the closure
as part of the band: it supplies roughly a fifth to a quarter of the width
charge and some of the coverage gain. Test a one-sided lower-edge floor
as a new arm, since both severe corner channels are lower-edge; it retains
only the corresponding one-sided regional certificate and needs its own
validation. Never infer its coverage from the two-sided arm.

Keep a small complete reference set at alpha = .05, .2 and .5. The
current alpha-independent region can plausibly waste width at .5, but
its actual required extent must be derived or measured. Simply shrinking
the region until coverage looks right would repeat the calibration problem.

### 6.3 Stronger construction: carry gaps as brackets

The most principled fiducial extension is to stop choosing the unknowable
within-gap positions. For each pair of spacing draws, retain the **entire
interval of ROC values compatible with monotone class CDF completions**.
The existing interpolation sandwich gives the needed brackets, including
both the negative gap and the positive spacings at its endpoints.

This is closely related to the interval-valued inverse construction of
[Cui and Hannig](https://arxiv.org/abs/1707.05034), whose one-sample
functional Bernstein–von Mises result provides asymptotic guidance,
not a general finite-sample ROC-band theorem.

Two distinct versions are worth separating:

- **Empirical version:** trim draws by a fixed score, then envelope the
  full brackets of retained draws. With the same retained indices, this
  contains the envelope of their selected representatives. It removes
  a completion assumption but still needs frequentist calibration.
  Re-trimming after bracket expansion is a different method.
- **Exact version:** choose a data-independent joint acceptance set of
  auxiliary class pivots with probability at least $1-\alpha$, then
  project **every** compatible CDF pair. Truth is included on the
  pivot event. M3 is the rectangular special case; §4.4 is one
  nonrectangular example. Sampling a finite cloud inside that set does
  not compute its guaranteed outer envelope.

This connection is valuable: the Dirichlet machinery can propose efficient
geometry while the auxiliary-space construction supplies an honest
coverage argument. The unresolved issue is whether a useful projection is
narrower than M3, not whether Dirichlet spacings are exact.

Bracket completion must handle empty stretches in the **interior** as
well as the current end runs. Arbitrary distributions can move an unseen
mass problem away from either endpoint. Test translated slivers,
two separated rare components, narrow positive spikes in negative gaps,
and reflected low-AUC examples before claiming that a tail-only fix is
universal.

### 6.4 A router remains useful, with a precise contract

A practical router could use the fiducial band for empirically benign ROC
shapes and M3 or the hybrid for wedge-like shapes. Keep this option, but
make its scope explicit:

- An AUC/count threshold is a **shape-library rule**, not a
  distribution-free certificate. The slivers fail at AUC .60 and .80 too.
- Rank features such as saturation, empty stretches and local
  interleaving runs are more mechanistically relevant than AUC.
  They still cannot certify that unseen mass is absent.
- Selecting between two marginally valid bands on the same data is not
  automatically valid at their original level. Either prove the selected
  procedure directly, calibrate the joint selection event, or select on
  an independent pilot and analyze a fresh inference sample.
  Splitting alone does not make an invalid fiducial branch valid.
- Declared concavity or a tail-shape class can support a class-relative
  router. Passing a goodness-of-fit test is not a certificate that the
  true curve belongs to that class.

There is one exact same-data routing opportunity: construct several bands
from a **common, already-calibrated pivot event**. If truth lies in every
branch on that event, selecting a branch or intersecting them preserves
its guarantee. The joint calibration may consume the apparent width
gain, but it is the right way to attempt adaptive M3/KS geometry.

## 7. Rank-likelihood e-values: a second exact route

The rank experiment has a finite sample space. Let
$p_R(\lambda)=P_R(\Lambda=\lambda)$ and let $q(\lambda)$ be any fixed
normalized probability model on valid merged label sequences. Then

$$
e_R(\Lambda)=\frac{q(\Lambda)}{p_R(\Lambda)},\qquad
E_R[e_R]\le1.
$$

Thus $\{R:e_R(\Lambda)<1/\alpha\}$ is a finite-sample confidence set;
its outer coordinate projection is an honest band. Ratios on impossible
null paths are interpreted as infinity when the numerator is positive.
The proof is the sum
$\sum_{\lambda:p_R(\lambda)>0}q(\lambda)\le1$ and Markov's inequality.

**Why this is interesting.** It targets the ROC directly and requires no
simulation quantile or shape-specific level remap. A misspecified predictor
q hurts power, not validity. A fixed mixture over ROC rank laws is one
choice. Another is a normalized sequential predictor for labels that uses
only the preceding labels and respects the remaining class counts.
This latter construction can learn along the path without treating a
full-data fitted likelihood as an independent numerator.

This is an application of the likelihood-ratio/e-value principle used in
[universal inference](https://pmc.ncbi.nlm.nih.gov/articles/PMC7382245/),
not a claim that an existing ROC implementation or width theorem is
available. Sample splitting is another way to train q; an unrestricted
fit to the same complete path is not justified by the argument.

The rank likelihood is concrete. If $k_j$ positives occur in gap j between
the $n_0$ ordered negative placements, then

$$
p_R(\lambda)=
\frac{n_0!\,n_1!}{\prod_{j=0}^{n_0}k_j!}
\int_{0<u_1<\cdots<u_{n_0}<1}
\prod_{j=0}^{n_0}
  [R(u_{j+1})-R(u_j)]^{k_j}\,du_1\cdots du_{n_0}.
$$

Here the boundary CDF values are $R(u_0)=R(0-)=0$ and
$R(u_{n_0+1})=R(1)=1$, allowing an atom at placement zero.
This is the multinomial gap probability integrated against uniform
order statistics. It suggests dynamic programming for piecewise
representations rather than numerical integration over all dimensions.

**Main risk.** A predictive likelihood penalty can widen the set, and
projection can widen it further. Finite-sample e-value validity alone
does not promise widths between WH and KS or good 50% calibration.
Benchmark q's predictive loss and projected width in the same small
enumerable problems as §5 before building a large inference engine.
For safe numerical exclusion, an upper bound on $p_R$ supplies a lower
bound on $e_R$; an underestimated denominator must not certify rejection.

## 8. Direct ROC-process calibration: an approximate route worth keeping

For regular interior ROC points, the first-order error is the sum of two
independent bridge terms, schematically

$$
\widehat R(t)-R(t)\approx
n_1^{-1/2}B_1(R(t))
 -R'(t)n_0^{-1/2}B_0(t),
$$

with variance
$R(t)(1-R(t))/n_1+R'(t)^2t(1-t)/n_0$.
The signs and covariance matter. Calibrating this combined process can
recover the cancellation that classwise projection discards.

**Proposal.** Use a multiplier or bridge simulation for a direct signed
studentized ROC tube on a declared regular interior, then add exact
order-statistic protection where counts are low. Compare regional
critical values with equal-local-level trimming; measure the full
maximum distribution at .05, .2 and .5. Keep the tube itself as an arm,
so a retained-cloud envelope is not confused with test inversion.

The distinctive opportunity is to target excursion probabilities over
FPR intervals, not just pointwise scales.
[Liebl and Reimherr's fast and fair bands](https://academic.oup.com/jrsssb/article/85/3/842/7133768)
develop varying critical values to balance errors across domain
partitions. Their random-process assumptions do not automatically hold
for the rough ROC bridge, so the transferable idea is regional
calibration, not an off-the-shelf theorem or smooth-process formula.

This route needs derivative/local-modulus estimation and enough observations
in both classes. Jumps, singular score laws, moving extreme quantiles and
unseen slivers defeat a blanket interior Gaussian claim. Calling the method
nonparametric does not remove these assumptions. It is a reasonable secondary
project if excellent broad-library behavior is acceptable, and a poor
substitute for §5 if unrestricted finite-sample honesty is mandatory.

ERL or smoothed-depth trimming is a smaller version of this project:
potentially useful for Monte Carlo resolution and the truth-versus-cloud
roughness discrepancy, but it neither restores missing cloud support nor
turns credible content into population coverage.

## 9. Experiments that would actually change the ranking

The next experiments should separate three questions:
**Can we certify it? Does it use that certificate efficiently?
Where does its remaining error occur?**

1. **Exact M3 design screen.** Expand §4 to modest deterministic boundary
   families, the sample-size split, and both KS/ELL-like shapes.
   Include alpha .2 and more AUC .5–.7 curves, where M3's current advantage
   is weakest. Validate each boundary's non-crossing probability before
   evaluating width. Retain an improvement only if gains survive new
   shapes without a large tail-width penalty.
2. **Small exact inversion benchmark.** Enumerate small rank experiments,
   certify outer projection over interval-completed monotone curves, and
   compare direct-test and e-value versions. This measures whether
   avoiding CDF projection offers enough gain to justify computation.
3. **Hybrid/bracket probe.** Pair the frozen hybrid with a seven-point
   left prefix, a lower-edge-only variant, and full gap brackets.
   Include translated slivers and support gaps, not only the known t
   wedge. Use fresh seeds and retain the frozen arm throughout.
4. **Central-alpha probe after support repair.** On the same datasets,
   compare raw min-p, ERL and a direct-process interior. Estimate where
   excess coverage comes from before fitting another exponent.
   Any finite-range empirical recalibration must be labeled as such.
5. **Adversarial confirmation.** Hold entire construction families out:
   atoms/ties, separated supports, cusps, mixtures with rare mass,
   extreme imbalance, both score orientations, and increasing n along
   fixed shapes. Also use $n$-dependent rare-mass sequences to test
   uniform validity rather than only pointwise asymptotics.

For every arm record whole-curve coverage; upper, lower and both-side
failures; intervalwise and pointwise miss rates; conditional miss depth;
integrated and tail-local width; paired KS/M3 ratios; and cold versus
cached runtime. At alpha = .05, 400 replicates give roughly a one-percentage-
point standard error near nominal. Use those runs to reject weak ideas,
then allocate substantially more replication to binding cells.
Do not call .94 “nominal 95% coverage” or use failure to reject a deficit
as evidence that the deficit is absent.

A rank-only adversarial search can optimize knot locations, slope changes,
jump masses and empty intervals directly. This is more targeted than
another broad score-family sweep. It can falsify uniform guarantees but
cannot prove them. Freeze adversaries and candidate rules before
confirmation.

## 10. Directions to set aside

- **Global C or M3 alpha remaps learned from a finite library.**
  They improve apparent calibration by spending validity the library
  cannot certify. Small tail masses can evade every smooth design cell.
- **Larger boundary margins as the answer to hybrid residuals.**
  Stage F locates most residuals far from those margins.
- **A universal cutoff in AUC or n.** The wedge is nonmonotone in n;
  sliver failures are not confined to the wedge.
- **Ordinary conformal prediction of a future empirical ROC.**
  That is not a confidence band for the fixed population ROC.
- **Ordinary two-sample label permutation for arbitrary ROC nulls.**
  Labels are exchangeable under the diagonal null, not under a general
  named ROC. Simulate the candidate's rank law instead.
- **Automatic concavification.** It changes a nonconcave score ROC into
  a different target. Concavity is useful only as a declared restriction.
- **Intersecting fiducial and M3 at the same nominal level to obtain
  M3's guarantee.** Containing an exact band transfers its guarantee;
  being contained in one does not.
- **Randomly returning vacuous or empty bands to hit the average alpha.**
  Unconditional coverage can be manipulated this way without improving
  the information conveyed by a realized dataset.

The recommended allocation of effort is therefore: keep M3 and the frozen
hybrid as anchors, take the inexpensive exact class-split improvement
through a broader screen, and put the main methodological effort into
certified direct rank inversion. Pursue bracket completion alongside it
because it addresses the measured fiducial failure at its source.
Further scalar calibration should follow an improved construction, not
stand in for one.
