# Theory of the Rank-Space Fiducial ROC Band

*Companion to `stats/next_method_ideas.md` (working model and evidence) and
`src/studroc_paper/methods/fiducial_band.py` (implementation). This document
develops the probabilistic structure behind the method: what is exactly true,
what is asymptotically true, what is finite-sample heuristic, and what is
open. Last substantive revision 2026-09-02, folding in the follow-up
boundary study (§7.3: the validity failure is a curved (AUC, n) *wedge*,
coverage is **not monotone in n**, misses concentrate at the upper-FPR
end, and a localized M3 floor with a pointwise-domination property is the
lead repair); previous revisions 2026-08-30 (§7.2, the Stage S screen and
the C=1 default) and 2026-08-23 (the §14 literature pass).*

**Status tags.** Every claim carries one of:
- **[Exact]** — proved here or a one-step consequence of a classical result;
  holds at every finite n with no conditions beyond those stated.
- **[Lit]** — proved in the cited literature; we state it and use it.
- **[Sketch]** — follows from standard machinery (empirical processes,
  exchangeable bootstraps); conditions stated, proof sketched, details not
  written out.
- **[Empirical]** — measured in `stats/experiments/` (laptop cells, 120–400
  reps); cited, not proved.
- **[Heuristic]** / **[Conjecture]** — a proposed mechanism or statement we
  believe but cannot yet prove; falsifiable, with the discriminating
  experiment named where possible.

---

## 1. Setup and notation

Negatives $X_1,\dots,X_{n_0} \sim F$ iid, positives $Y_1,\dots,Y_{n_1} \sim G$
iid, classes independent, $F, G$ continuous. The ROC curve is
$R(t) = 1 - G\big(Q_F(1-t)\big) = S_G\big(S_F^{-1}(t)\big)$, a
non-decreasing map $[0,1]\to[0,1]$, where
$Q_F(u)=\inf\{x:F(x)\ge u\}$ for $u\in(0,1]$ and only the missing lower
endpoint is completed by $Q_F(0)=-\infty$. Thus $Q_F(1)$ is the finite
upper support endpoint when $F$ attains 1 at a finite point, and is
$+\infty$ otherwise. This convention makes $R(1)=1$ for every $(F,G)$
while retaining $R(0)=1-G(Q_F(1))$, which can be positive when the positive
distribution extends beyond a bounded negative support (Corollary 9.3).
Extending $Q_F(1)$ artificially to $+\infty$ would instead force $R(0)=0$
and would contradict both the placement-value identity and that
separated-support case. The target is a
band $[L, U]$ with $P\big(\forall t: L(t) \le R(t) \le U(t)\big) \ge 1-\alpha$.

The construction (recipe in `next_method_ideas.md` §1): merged label sequence
$\Lambda$ (ties broken at random); $M$ fiducial curves $\tilde R_1,\dots,
\tilde R_M$ from Dirichlet spacings per class with within-gap spreading,
evaluated on the grid $t_k = k/n_0$; equal-local-levels (min-p) trim at level
$\alpha_{\mathrm{eff}} = 1-(1-\alpha)^C$ (default $C=2$) with realized depth
$j^\*$; pointwise $[j^\*\text{-th smallest},\, j^\*\text{-th largest}]$ tube;
Clopper–Pearson upper allowance at local level $\ell = j^\*/(M+1)$; lower
edge zeroed where the empirical TPR count is zero.

**Between grid points (grid-to-continuum, free). [Exact]** The band is
defined at the grid points and extended by the conservative step rule
$L(t) = L(\lfloor n_0 t\rfloor / n_0)$, $U(t) = U(\lceil n_0 t\rceil / n_0)$
— exactly the implementation's output-grid resampling convention. Since $R$
is non-decreasing, coverage at the grid points implies coverage everywhere
under this extension: for $t \in [t_k, t_{k+1}]$,
$R(t) \le R(t_{k+1}) \le U(t_{k+1}) = U(t)$ and
$R(t) \ge R(t_k) \ge L(t_k) = L(t)$. Every grid-level coverage statement
below therefore carries to all of $[0,1]$ with no loss.

**One sentence of positioning** (details in §14): the construction is a
*global rank envelope* (Myllymäki, Mrkvička, Grabarnik, Seijo & Hahn 2017,
JRSS-B) applied to a *fiducial cloud of ROC curves* (the spacings-GFD of
Cui & Hannig 2019 per class, composed through the ROC map; adjacent to,
but distinct from, the Bayesian-bootstrap ROC cloud of Gu, Ghosal & Roy
2008 — see §3), with a finite-sample credible-to-confidence calibration
analysis (§7) and corner devices at the forced scale (§8) that have no
antecedent we have found.

---

## 2. The rank reduction: why the problem is exactly distribution-free

**Proposition 1 (rank space). [Exact]** Let $U_i = 1 - F(X_i)$ and
$W_j = 1 - F(Y_j)$. Then $U_i \sim \mathrm{Uniform}(0,1)$ iid, and
$P(W_j \le t) = 1 - G(Q_F(1-t)) = R(t)$: the positives' transformed values
have CDF exactly the ROC curve, including at both endpoints. The transform
$s \mapsto 1-F(s)$ is non-increasing and reverses the relevant order:
descending score order becomes ascending placement-value order. If $F$ has
a flat interval, no negative observation falls in its interior with positive
probability; positives in that interval form a placement-value tie block
between the same neighboring negatives. Refining that block arbitrarily
therefore leaves the merged *class-label* sequence unchanged.

*Proof.* Probability integral transform for $U_i$; direct computation for
$W_j$. $\square$

*Terminology.* The $W_j$ are the **placement values** of the ROC literature
(Pepe 2003); Gu–Ghosal–Roy (2008) build their Bayesian-bootstrap ROC
estimator on exactly this representation. Proposition 1 is standard; what
the method does with it is not.

**Proposition 2 (maximal invariance and its consequences). [Exact]** The
merged label sequence $\Lambda$ is a maximal invariant of the data under the
group of strictly increasing continuous score transforms. Consequently, for
any band procedure that is a (possibly randomized) function of $\Lambda$
alone:

1. **Invariance.** The coverage probability is a functional
   $\mathcal{C}(R, n_0, n_1)$ of the true curve and sample sizes only. Two
   DGPs with the same ROC shape — Student-$t$ at any df, logit-normal at any
   $\sigma$, anything — have *identical* coverage if their $R$ coincides.
2. **Simulability.** For any hypothesized curve $R_0$, the exact joint law of
   $\Lambda$ (hence of the band, hence of coverage) is obtained by drawing
   $n_0$ uniforms and $n_1$ draws from $R_0$. No asymptotics are involved.

*In words:* "insensitivity to data properties" is not an empirical property
of this method to be checked family-by-family — it is a theorem. The only
axis along which coverage can vary is the *shape* of the true curve (and the
sample sizes). Everything measured in the experiments, and everything the
full simulation can reveal, is a statement about shape. Simulability is also
a *design principle*: it makes every hypothesis about the method exactly
testable (per shape) and powers both the exact test of Proposition 11 and
the offline calibration of `c_calibration_spec.md`.

The implementation is a function of $\Lambda$ plus independent randomization
(tie-breaking, Monte Carlo draws), so Proposition 2 applies to it verbatim.

---

## 3. The fiducial distribution: what is exact about the cloud

**Proposition 3 (Dirichlet spacings; exact marginals). [Exact]** For a single
sample $Z_1,\dots,Z_n \sim H$ continuous, the vector
$\big(H(Z_{(1)}),\dots,H(Z_{(n)})\big)$ is distributed as uniform order
statistics; equivalently its successive spacings (including the two end
gaps) are Dirichlet$(1,\dots,1)$. In particular
$H(Z_{(j)}) \sim \mathrm{Beta}(j,\, n+1-j)$ exactly, for every $j$, $n$, and
continuous $H$.

The fiducial step inverts this: given the data, treat the unknown vector
$\big(H(Z_{(1)}),\dots,H(Z_{(n)})\big)$ as Dirichlet-distributed. Three facts
anchor this choice:

- **Exact pivotal matching at the anchors. [Exact]** $H(Z_{(j)})$ is an
  exact pivot — its Beta$(j, n+1-j)$ law holds for every continuous $H$ at
  every $n$ — and the fiducial marginal at each order statistic reproduces
  that pivotal law exactly. Care with the reading: $H(Z_{(j)})$ is a
  random, data-indexed quantity, not a fixed parameter, so "fiducial
  probability = coverage" is a statement about the pivot, not about a
  parameter's confidence set. The parameter-level consequence is the
  classical one: inverting the pivot gives exact distribution-free
  confidence bounds for the *quantiles* of $H$ at the order statistics —
  the same Beta law that powered the envelope method's Beta floor.
- **GFD reading — and the distinction from the Bayesian bootstrap.
  [Lit + Exact]** The $(n{+}1)$-spacings law is precisely the
  *generalized fiducial distribution* (GFD) of a nonparametric CDF (Hannig
  et al. 2016; the uncensored case of Cui & Hannig 2019, whose inverse
  image of $Y_i = F^{-1}(U_i)$ is the set of CDFs pinched between step
  functions $F^L \le F \le F^U$ carrying exactly this law at the order
  statistics; classically, Hill's $A_{(n)}$ / Dempster's direct
  probabilities for quantiles). It is **not** Rubin's (1981) Bayesian
  bootstrap, and the difference is load-bearing: the BB puts $n$
  Dirichlet$(1,\dots,1)$ weights on the observed atoms, so
  $F_{\mathrm{BB}}(Z_{(j)}) \sim \mathrm{Beta}(j,\, n-j)$ for
  $j=1,\dots,n-1$, with $F_{\mathrm{BB}}(Z_{(n)}) = 1$ identically — no
  mass between or beyond observations — and it is the BB, not the spacings
  law, that arises as
  the noninformative limit of the Dirichlet-process posterior. The two are
  coupled to $O_p(1/n)$ uniformly: writing the spacings partial sums as
  normalized exponentials $S_j = T_j / T_{n+1}$ and the BB partial sums as
  $S'_j = T_j / T_n$, $\sup_j |S_j - S'_j| \le E_{n+1}/T_n = O_p(1/n)$.
  They therefore share the $n^{-1/2}$-scale conditional limit (used in
  Theorem 7), while differing at exactly the $1/n$ scale of the corner
  channel (next bullet) and of the §7 roughness story.
- **Mass beyond the extremes. [Exact]** $1 - H(Z_{(n)}) \sim
  \mathrm{Beta}(1, n)$: the fiducial CDF places a correctly-distributed
  amount of probability beyond the largest observation. This is the exact
  channel whose *absence* broke the resampling bootstrap (its resampled
  extremes cannot exceed observed extremes; measured lower-arm collapse to
  0.8–1.4 sd at $k=1$–3), and the Bayesian bootstrap shares that collapse
  ($F_{\mathrm{BB}}(Z_{(n)}) = 1$ identically). The fiducial cloud does not
  patch this failure — it never has it. This is one half of why the cloud
  is the spacings-GFD and not the BB.

**Proposition 4 (unification of the old floors). [Exact]** The envelope
method's exact Beta order-statistic floor used the law
$\bar F(X_{(j)}) \sim \mathrm{Beta}(j, n_0+1-j)$; by Proposition 3 this is
precisely the fiducial marginal of the cloud's FPR coordinate at the $j$-th
negative. The Wilson/binomial floor modeled TPR at a fixed threshold as
Binomial$(n_1, \cdot)$; the fiducial G-side marginal at a positive order
statistic is the corresponding exact Beta. Both bolt-on floors of the old
method are *marginals of this single object*. The hybrid became a primitive.

### 3.1 The one approximation: the within-gap convention, now bounded

The fiducial argument determines the joint law of each class's CDF *at its
own order statistics*. The positions of the *other* class's points within a
gap, and the curve between anchors, are not determined by it: they depend on
the unknown local shape of $R$. The construction places within-gap points at
sorted-uniform fractions of the gap and interpolates linearly — the natural
exchangeable convention. Cui & Hannig handle the same indeterminacy by
carrying the *interval* $[F^L, F^U]$ (their "conservative option") or by
selecting a log-linear representative; ours is a third selection rule. The
following proposition bounds what any such choice can matter.

**Proposition 3b (interpolation sandwich).** Fix one fiducial draw of both
classes' Dirichlet spacings, and consider the family of composed ROC draws
obtainable under *any* within-gap placement and any monotone interpolation
(all conventions agree on the classes' CDF values at their own order
statistics).

1. **[Exact]** Per class, all conventions lie between the lower and upper
   step completions $\tilde H^L \le \tilde H \le \tilde H^U$, and
   $\sup_x (\tilde H^U - \tilde H^L)(x) \le \Delta_{\max}$, the maximal
   Dirichlet spacing, with $\Delta_{\max} = O_p(\log n / n)$.
2. **[Exact]** Both axes contribute, and the correct bound counts both.
   Let the grid point $t_k$ lie in the F-gap between consecutive negatives
   $X_{(i)}, X_{(i+1)}$ (whose fiducial F-coordinates $x_i < x_{i+1}$ are
   convention-free), let $j$ = the number of positives ranked below
   $X_{(i)}$ and $j'$ = the number ranked below $X_{(i+1)}$. Then *every*
   convention's composed curve value at $t_k$ lies in
   $\big[\tilde G(Y_{(j)}),\ \tilde G(Y_{(j'+1)})\big]$ (with the obvious
   degenerate endpoints $0$ and $1$), so two conventions differ at $t_k$
   by at most the fiducial G-mass spanned by the enclosing F-gap *plus one
   G-spacing at each end*. The extra spacings are not slack in the proof:
   a curve's height *at an F-anchor* is itself a G-side within-gap
   completion — the F-anchor is a negative observation, whose G-coordinate
   the G-convention is free to move within its own gap — so the anchor
   heights are convention-dependent and only their brackets
   $[\tilde G(Y_{(j)}), \tilde G(Y_{(j+1)})]$ are convention-free.
3. **[Sketch, with an explicit modulus rate]** Let
   $I=[\varepsilon,1-\varepsilon]$, enlarge it by a fixed small margin,
   and write

   $$\omega_R(\delta;I)
     =\sup\{|R(s)-R(t)|:s,t\text{ in the enlargement of }I,
                         |s-t|\le\delta\}.$$

   If $n_1/n_0$ stays bounded above and below, the diameter over all
   conventions, uniformly at grid points in $I$, is

   $$O_p\!\left[
       \omega_R\!\left(C\frac{\log n_0}{n_0};I\right)
       +\frac{\log(n_0+n_1)}{n_1}\right]$$

   for any fixed sufficiently large $C$. In particular, if $R$ is locally
   Hölder-$\beta$ on $I$,
   $|R(s)-R(t)|\le H|s-t|^\beta$, the diameter is

   $$O_p\!\left[
       \left(\frac{\log n_0}{n_0}\right)^\beta
       +\frac{\log(n_0+n_1)}{n_1}\right].$$

   It is therefore $o_p(n^{-1/2})$ for comparable sample sizes whenever
   $\beta>1/2$. Theorem 7 assumes $C^1$ smoothness, hence $\beta=1$ and
   recovers the earlier $O_p(\log n/n)$ statement.
4. **The actual boundary.** Continuity alone makes the convention diameter
   vanish on a fixed compact, but need not make it first-order negligible.
   A Hölder-$1/2$ cusp gives the borderline
   $O_p(\sqrt{\log n/n})$ rate, and rougher cusps can be larger. If $R$
   has a jump, which is possible even for continuous score CDFs when $F$
   has a support gap containing positive mass, one negative placement gap
   can carry asymptotically the whole jump and the diameter can be
   $O_p(1)$. Thus “steep” is not the mathematical dividing line:
   the dividing line for first-order equivalence is a local modulus
   $o(\sqrt\delta)$, up to logarithms.

*Proof of (1)–(2).* (1) is the inverse-image structure (Cui & Hannig 2019,
eq. 2.3): consecutive completions differ by at most the spacing they
re-attach; the max of $n$ uniform spacings is $(\log n + O_p(1))/n$. For
(2): any convention's curve value at $x_i$ is its fiducial G-CDF evaluated
at the threshold $X_{(i)}$, which sits between the positive order
statistics $Y_{(j)}$ and $Y_{(j+1)}$; monotonicity of the G-completion
places that value in $[\tilde G(Y_{(j)}), \tilde G(Y_{(j+1)})]$ regardless
of convention, and likewise at $x_{i+1}$ the value lies in
$[\tilde G(Y_{(j')}), \tilde G(Y_{(j'+1)})]$. Monotonicity of the composed
curve between the anchors then confines its value at $t_k$ to
$[\tilde G(Y_{(j)}), \tilde G(Y_{(j'+1)})]$, whose fiducial mass is the
stated bound. $\square$

*Proof of the rate in (3).* In rank space the negative placements are
uniform, whose maximal spacing over the enlarged compact is
$D_{0,n}=O_p(\log n_0/n_0)$. Conditional on those gaps, the positive counts
form a multinomial vector with cell probabilities at most
$\omega_R(D_{0,n};I)$ (apart from the two irrelevant boundary cells).
A union bound over the $O(n_0)$ cells plus the binomial Bernstein inequality
gives

$$\max_g \frac{N_{1,g}}{n_1}
  =O_p\!\left(\omega_R(D_{0,n};I)
               +\frac{\log(n_0+n_1)}{n_1}\right);$$

the square-root Bernstein term is absorbed by
$2\sqrt{ab}\le a+b$. A block spanning $m$ consecutive positive anchors
has fiducial Dirichlet mass equal to a normalized sum of $m+O(1)$ unit
exponentials. Uniform exponential concentration over the same cells adds
only $O_p(\log(n_0+n_1)/n_1)$. Proposition 3b(2) and the two endpoint
spacings then give the display. Under the Hölder condition substitute
$\omega_R(\delta;I)\le H\delta^\beta$. $\square$

**Lemma 3c (the implemented completion is measurable). [Exact]** Fix a
deterministic rule for zero-probability ordering ties. Given the merged
label sequence and the finite vectors of auxiliary exponential and uniform
draws, the construction obtains spacing partial sums, sorted within-gap
fractions, and polyline values by finitely many comparisons and arithmetic
operations. It is therefore a Borel-measurable random curve in
$C[0,1]$. Per class it agrees with the sampled GFD anchor values and lies
between the two GFD step completions, so it is a measurable selection from
the product of the two classes' inverse images. This closes the measurable-
selection bookkeeping in Theorem 7; it does not assert that this particular
selection is uniquely fiducial.

**Measured corroboration. [Empirical]** Coverage is insensitive to the
convention (random vs even spreading indistinguishable; ties red-team), as
(3) predicts on the interior.

**A conservative interval-valued variant (untried).** Following Cui &
Hannig's conservative option: compose, per draw, the *bracket*
$[\tilde R^L, \tilde R^U]$ (lower/upper completions on both axes), score a
draw as inside the tube only when its whole bracket is inside, and take the
band as the envelope of retained brackets. By Proposition 3b(1)–(2) this
band contains the band of every selection convention (that is what is "by
construction" here — containment of the conventions, not frequentist
validity, which rests on the same §6–§7 calibration story as everything
else), inherits content control from Lemma 6b below, and costs at most the
Proposition 3b(2) bracket in width — $O_p(\log n/n)$ under the smooth
interior assumptions, the explicit modulus rate otherwise, and possibly
material at rough or discontinuous regions where one might *want* the extra
width (§10). A cheap derisk candidate.

---

## 4. What uncertainty the cloud carries (and the variance it reproduces)

In rank space the sampling fluctuation of the empirical ROC has the
Hsieh–Turnbull (1996) structure: at interior $t$,

$$\mathrm{Var}\,\hat R(t) \approx \frac{R(t)(1-R(t))}{n_1}
 + R'(t)^2\,\frac{t(1-t)}{n_0},$$

the binomial (vertical) channel plus the threshold-location (horizontal)
channel scaled by the slope. **[Sketch]** The fiducial cloud reproduces both:
the G-side Dirichlet drives the first term, the F-side Dirichlet drives the
second (the slope enters through the composition, not through any estimate
of it). No density or slope estimation occurs anywhere — the quantity that
made HT-type bands fragile is never formed. The cloud is also automatically
monotone, $[0,1]$-valued, asymmetric near the boundaries, and correlated
across $t$ exactly as a CDF composition must be, because every draw *is* a
monotone curve rather than a pointwise interval.

---

## 5. The trim: a global rank envelope of the fiducial cloud

The trim is not a new statistic. The depth $S$ below is the **extreme rank**
of the global rank envelope test (Myllymäki et al. 2017), in the
maximum-rank treatment of pointwise ties; the same functional is the
finite-sample core of **extremal depth** (Narisetty & Nair 2016, JASA), and
the $[j$-th smallest, $j$-th largest$]$ tube over sampled curves goes back at
least to the simultaneous credible bands of Besag, Green, Higdon & Mengersen
(1995, p. 30). Equivalently it is the Westfall–Young (1993) min-p
construction applied to the cloud, and an equal-local-levels band in the
sense of Berk–Jones (1979) / Nair (1984) / Aldor-Noiman et al. (2013). What
is *not* in those literatures is the calibration question of §7 — there the
cloud's own content is taken as the level, which is exactly the $C=1$ arm of
this method.

**Lemma 5 (depth–tube duality). [Exact; essentially Myllymäki et al. 2017,
Thm. 4.2, restated for an arbitrary curve against the cloud rather than an
ensemble member]**
For curves $c$ evaluated on the grid, define
$a_k(c) = \#\{m: \tilde R_m(t_k) \le c(t_k)\}$,
$b_k(c) = \#\{m: \tilde R_m(t_k) \ge c(t_k)\}$ and the depth
$S(c) = \min_k \min(a_k(c), b_k(c))$ (tie-inclusive). Then $c$ lies inside
the pointwise $[j\text{-th smallest}, j\text{-th largest}]$ tube at every
grid point **iff** $S(c) \ge j$. Tubes are nested in $j$.

**Lemma 6 (finite-M content control). [Exact]** With
$j^\* = $ the $(\lfloor \alpha_{\mathrm{eff}} M\rfloor + 1)$-th smallest
draw depth, the fraction of draws inside the tube is $\ge 1 -
\alpha_{\mathrm{eff}}$, for every $M$. Monte Carlo error therefore acts only
through the location of the tube boundary, not through the content
guarantee; and when $M$ is too small the depth saturates at $j^\* = 1$,
which yields the *widest* tube the cloud supports — an underpowered Monte
Carlo budget therefore weakly *widens* the band (by tube nesting, its
coverage dominates that of any deeper trim) while destroying
$\alpha$-resolution. Content control is a statement about the cloud, not by
itself a frequentist guarantee; what the trim level buys in coverage is
§6–§7's subject. **[Empirical]** measured: coverage
$.967 \to .942$ under deliberate saturation at $K=5001$, while the
$\alpha=.05$ and $\alpha=.10$ bands become identical.

**Lemma 6b (content control for arbitrary trim scores). [Exact]** Let
$s_1,\dots,s_M$ be *any* per-draw scores (smaller = more extreme), computed
from the cloud and the data by any rule. Retain the draws whose score is
$\ge$ the $(\lfloor \alpha_{\mathrm{eff}} M\rfloor+1)$-th smallest score
(ties retained), and let the band be the pointwise envelope of the retained
draws (or any superset of it). Then the retained fraction is
$\ge 1-\alpha_{\mathrm{eff}}$ for every $M$.

*Proof.* At most $\lfloor \alpha_{\mathrm{eff}} M\rfloor$ draws have score
strictly below the threshold order statistic. $\square$

*Consequence.* Content control is a property of trimming-by-quantile, not of
the min-p depth. Every member of the family — min-p (the production tube,
via Lemma 5), the ERL refinement below, the smoothed-depth trim of §12, the
interval-valued variant of §3.1 — has identical finite-$M$ content
guarantees. What distinguishes them is the *shape* of the retained set and
the second-order calibration of §7.

### 5.1 The ERL refinement: a known fix for depth saturation

The global-envelope literature hit the $j^\*$-saturation problem (extreme
ranks tie heavily at small $M$) and solved it by refining the ordering
(Myllymäki et al. 2017, §6.1; the population version is Narisetty & Nair's
extremal depth; implemented as `'erl'` in the GET R package, Myllymäki &
Mrkvička 2024, JSS).

**Definition (extreme rank length, ERL).** For draw $m$ let
$R^*_m(t_k) = \min(a_k, b_k)$ be its two-sided pointwise rank, and let
$\rho_m$ be the vector $\big(R^*_m(t_1),\dots,R^*_m(t_K)\big)$ sorted
ascending. Order draws by lexicographic comparison of the $\rho_m$
(smaller = more extreme).

**Properties. [Exact, elementary]**

1. **Refinement.** $S_m < S_{m'}$ implies $\rho_m \prec \rho_{m'}$ (compare
   first components), so ERL refines the min-p ordering. Ties under ERL
   require two draws to share their entire *sorted rank vector* — much
   rarer than sharing a minimum, but of **positive** probability at every
   finite $(M, K)$ (rank vectors are discrete objects; fully-tied columns
   such as the shared endpoints $\tilde R(0) = 0$, $\tilde R(1) = 1$
   contribute identical entries to every draw). A strict ordering requires
   residual ties broken by independent randomization — Myllymäki et al.'s
   lexicographic $(T_i, M_i)$ device, which their Lemma 3.1 covers exactly.
2. **No saturation.** With randomized tie-breaking the ordering is strict,
   so the ERL trim at content $1-a$ retains exactly $\lceil (1-a)M \rceil$
   draws at every $M$: the $\alpha$-resolution is $1/M$ rather than being
   destroyed by the atom at depth $j^\*=1$. Content control is Lemma 6b.
3. **Nesting.** The ERL-retained set contains every draw of depth
   $\ge j^\*+1$ and is contained in the set of depth $\ge j^\*$ (for the
   appropriate $j^\*$), so the ERL envelope is sandwiched between the
   adjacent min-p tubes $j^\*$ and $j^\*+1$. The exact depth–tube duality
   (Lemma 5) is replaced by this two-sided sandwich; GET treats the ERL
   envelope as having the same graphical interpretation in practice.

**Why it matters here.** The M-budget rule of §9 exists to keep $j^\*$
comfortably above saturation; ERL removes the failure mode at its source and
is the GET authors' recommendation precisely when simulations are expensive.
A drop-in ERL trim could cut the required $M$ substantially at large $n_0$
(where $M \approx 10^4$ is the current cost driver), at the price of the
exact Lemma 5 duality. Untried here; flagged in §12.

### 5.2 Structure the trim buys

- **Self-studentization without a variance estimate.** Local ranks adapt the
  tube to the local spread of the cloud — the role studentization played in
  the envelope method — with no pointwise variance in any denominator. The
  earlier negative result (noisy pointwise variance × supremum inflates
  critical values; `project_evaluation_report.md` G.1) is bypassed
  structurally, not repaired. This is not merely an analogy: Narisetty &
  Nair (2016, Cor. 1) prove that for a Gaussian process the extremal-depth
  central regions are exactly the bands $\{\pm w\,\sigma(t)\}$ — the
  equal-local-levels region *is* the correctly-studentized band in the
  Gaussian limit, with the width constant $w$ set by the process's own
  correlation structure rather than by an estimated variance.
- **Directional balance by construction.** The two sides receive identical
  local levels (rank-from-bottom vs rank-from-top is an exact symmetry of
  the statistic). In the Gaussian limit (§6) the deviation law is symmetric,
  so the probabilities of exiting above and below are asymptotically equal;
  spatially, equal local levels spread the exit intensity across the grid
  rather than concentrating it at a corner. **[Sketch]** for the limit
  statement; **[Empirical]** at finite n (e.g. $.017/.017$ at $n=5000$;
  median miss FPR spread over $0.08$–$0.79$ across cells; contrast the
  envelope's measured $\sim$10:1 downward skew).
- **Graze-type misses. [Heuristic]** A coverage failure means the truth's
  depth fell just below $j^\*$ somewhere; the exceedance has no atom and the
  tube boundary is an order statistic of a continuous cloud, so the
  miss-depth distribution has mass concentrated near $0^+$. This is the
  structural reading of the measured "p95 miss depth = 0 everywhere; max a
  few pp".

---

## 6. First-order theory: asymptotic calibration on the interior

**Lemma 7a (Gaussian form of the ELL region). [Exact]** Let $Z$ be a
centered, sample-continuous Gaussian process on a compact interval $I$, with
$\sigma^2(t)=\operatorname{Var}Z(t)$ continuous and bounded away from zero.
For a deterministic path $z$, its population two-sided pointwise depth and
minimum depth are

$$d_t(z)=\min\{P(Z(t)\le z(t)),P(Z(t)\ge z(t))\}
          =\Phi\!\left(-\frac{|z(t)|}{\sigma(t)}\right),$$
$$D(z)=\inf_{t\in I}d_t(z)
      =\Phi\!\left(-\left\|z/\sigma\right\|_\infty\right).$$

Consequently every ELL central region of this law is exactly
$\{z:\|z/\sigma\|_\infty\le w\}$ for some $w$. If
$T=\|Z/\sigma\|_\infty$ and $w_a$ is its $(1-a)$ quantile, the region
has probability $1-a$. The distribution of $T$ is continuous by the
standard anti-concentration theorem for suprema of separable Gaussian
processes; hence $w_a$ is unique whenever its CDF is strictly increasing at
level $1-a$. This last local strictness, weaker than global strict
monotonicity, is the only cutoff regularity used below.

**Lemma 7b (finite-cloud approximation on a growing grid). [Exact
conditional bound + asymptotic consequence]** Let $Q_n$ be the conditional
law of a cloud draw and $F_{n,k}$ its marginal CDF at grid point $t_k$.
For $M$ conditionally iid draws, let $\widehat F_{M,k}$ be the corresponding
empirical marginal CDF. Then, conditionally on the data,

$$P\!\left(\max_{k\le K}\sup_x
  |\widehat F_{M,k}(x)-F_{n,k}(x)|>\eta\ \middle|\ \mathcal D_n\right)
  \le 2K e^{-2M\eta^2}.$$

This is Dvoretzky–Kiefer–Wolfowitz at each column plus a union bound; using
each draw in its own rank changes its empirical tail probabilities by at
most $1/M$. Thus all pointwise ranks and all minimum depths differ from
their $Q_n$-population versions by
$O_p(\sqrt{\log K/M}+1/M)$. If $Q_n\Rightarrow Q$ in probability in
$C(I)$, the marginal laws converge uniformly, the grid mesh tends to zero,
and the limiting depth cutoff has the regularity in Lemma 7a, the empirical
ELL cutoff and tube converge to their $Q$-population counterparts whenever

$$\frac{M_n}{\log K_n}\longrightarrow\infty.$$

Within this fixed-interior regime, no stronger coupling of $M_n$ to the
statistical sample size is needed. For the native ROC grid restricted to
$I$, $K_{n,I}\le n_0+1$, so $M_n/\log n_0\to\infty$ is enough. This
condition is interior-specific: Lemma 7a gives the non-vanishing limiting
local level $\ell_I=\Phi(-w_a)>0$, whereas §9's practical budget rule is
governed by corner strips outside Theorem 7, where the local level continues
to shrink and absolute DKW accuracy alone does not characterize tail
resolution.

**Theorem 7 (conditional equivalence and asymptotic coverage). [Sketch;
the construction-specific steps made explicit]** Let
$I=[\varepsilon,1-\varepsilon]$. Assume $R$ is continuously differentiable
with $0<R'<\infty$ on a neighborhood of $I$, and
$n_0,n_1\to\infty$ with $n_1/n_0\to\lambda\in(0,\infty)$; put
$n=n_1$. Let the grid mesh tend to zero with $K_n=O(n_0)$, and take
$M_n/\log K_n\to\infty$. Then:

1. $\sqrt n(\hat R-R)\rightsquigarrow Z$ in $C(I)$, where

   $$Z(t)=B_G(R(t))-\sqrt\lambda\,R'(t)B_F(t),$$

   with independent Brownian bridges $B_F,B_G$. Replacing $B_F$ by
   $-B_F$ gives the equivalent plus-sign form commonly used for the
   Hsieh–Turnbull limit.
2. Conditionally on the data,
   $\sqrt n(\tilde R-\hat R)\rightsquigarrow Z$ in probability in $C(I)$.
3. For the ELL tube with cloud content $1-a$, any fixed $a\in(0,1)$ whose
   Gaussian cutoff satisfies Lemma 7a's local strictness,

   $$P\big(\forall t\in I:L(t)\le R(t)\le U(t)\big)
     \longrightarrow 1-a.$$

*Proof of (2), conditional composition step.* Work in rank coordinates and
write $\Psi(A,B)=B\circ A^{-1}$. At $(A,B)=(\mathrm{id},R)$, for continuous
tangent directions on $I$,

$$\dot\Psi_{(\mathrm{id},R)}(h_A,h_B)(t)
  =h_B(t)-R'(t)h_A(t).$$

Indeed
$(\mathrm{id}+s h_A)^{-1}(t)=t-s h_A(t)+o(s)$ uniformly, followed by a
first-order expansion of $R+s h_B$. This is the needed Hadamard derivative,
not merely an analogy to the one-sample result. Per class, Dirichlet
Bayesian-bootstrap weights satisfy the Praestgaard–Wellner exchangeable-
weight conditions (exchangeability, unit mean, asymptotic unit variance,
and maximal weight $o_p(\sqrt n)$), yielding the conditional empirical
bridge. Proposition 3's exponential coupling transfers that limit to the
$(n+1)$-spacing GFD. Independence of the two spacing vectors gives
independent bridges. Applying the conditional functional delta method to
the displayed derivative gives
$B_G(R)-\sqrt\lambda R'B_F$. Proposition 3b with $\beta=1$ makes every
within-gap selection differ by $O_p(\log n/n)=o_p(n^{-1/2})$ on $I$, and
Lemma 3c supplies measurability. The same bound absorbs the difference
between the empirical-ROC and cloud centerings. This proves (2) from the
published one-sample weighted-bootstrap/GFD limit.

*Proof of (3).* Here

$$\sigma^2(t)=R(t)(1-R(t))
       +\lambda R'(t)^2t(1-t),$$

which is continuous and bounded away from zero on $I$. Lemma 7a identifies
the limiting content-$1-a$ tube as
$\{z:\|z/\sigma\|_\infty\le w_a\}$. Lemma 7b transfers the
conditional cloud tube to this region under the stated finite-$M$ regime.
The truth is in the data-centered tube exactly when
$\sqrt n(R-\hat R)$ is in its centered version. Its limit is $-Z$, and
$-Z\stackrel d=Z$ by Gaussian symmetry, so the limiting coverage is the
region's probability $1-a$. The monotone step extension in §1 transfers
native-grid coverage to the continuum. $\square$

*In words:* on any interior sub-interval, the raw fiducial credible band
(trim exponent $C=1$) is asymptotically **calibrated**: its limiting
coverage equals $1-a$ exactly, rather than merely exceeding it. (No
finite-sample validity claim is made for $C=1$; the finite-sample evidence
is §7's.) The scope limitations are real and stated: the theorem is
on compacts; the corner zones are governed instead by the
finite-sample devices of §8, and the moving-boundary strip between them
($t \sim k/n_0$, small fixed $k$) is covered by neither argument and rests
on the exact Beta structure of the cloud's tail marginals (Prop. 3–4) plus
simulation evidence.

**What remains before calling this a full proof.** The former construction-
specific gaps are now reduced: Lemma 3c handles measurable selection, the
displayed derivative and conditional delta argument handle composition,
Lemma 7a handles the Gaussian ELL functional, Lemma 7b gives the joint
$M_n,K_n$ regime, and §1 handles grid-to-continuum extension. What still
deserves line-by-line appendix treatment is verification that the chosen
function-space versions of the cited Praestgaard–Wellner or Cui–Hannig
one-sample theorem deliver the stated *conditional-in-probability*
$C(I)$ convergence, including the routine uniform-marginal consequence
used in Lemma 7b. The result remains tagged [Sketch] because that external-
theorem bookkeeping has not been written, not because a method-specific
probabilistic step is still missing.

**A consequence worth stating plainly.** Theorem 7 fixes the asymptotic
meaning of the trim level: coverage tends to $1-\alpha_{\mathrm{eff}}$, i.e.
to $(1-\alpha)^C$. Any *fixed* exponent $C > 1$ therefore drives asymptotic
coverage *below* nominal at fixed $\alpha$: with $C = 2$ and
$\alpha = 0.05$, the limit is $0.9025$, not $0.95$. The $C=2$ remap is a
finite-sample calibration whose justification must be finite-sample — and
whose validity must eventually taper if the theorem's regime is reached.
This is §7's subject.

---

## 7. Second-order behavior: the credible-to-confidence gap and the meaning of C

The fiducial content of the tube is not its frequentist coverage at finite
$n$. This section quantifies the gap. We found nothing analogous in the
antecedent literatures: the global-envelope and functional-depth literatures
use central regions *of the cloud itself* (content = level by definition),
and the fiducial literature proves first-order correctness (Theorem 7's
regime) — the finite-sample calibration function below appears to be new.

Define the calibration function as a generalized inverse,
$a^\*(\alpha; R, n_0, n_1) = \sup\{a : \text{realized coverage at trim
level } a \ \ge\ 1-\alpha\}$ (realized coverage is a nonincreasing, not
necessarily continuous, function of the trim level, so exact equality need
not occur), and the implied exponent
$C^\* = \log(1-a^\*)/\log(1-\alpha)$. Both are computable to Monte Carlo
precision per shape by Proposition 2 (simulability).

**What is measured. [Empirical]** $C^\*$ is not a universal constant. From
the 14-cell table (each entry an independent optimization; 120–400 reps),
plus the large-$n$ ladder (M = 2500, central $\alpha$):

| regime | $C^\*$ at $\alpha=.5$ | $C^\*$ at $\alpha=.05$ |
|---|---|---|
| $n=25$ | 3.1 | 4.6 |
| $n=500$ (binormal/bimodal/kink) | 1.9–2.2 | 2.3–2.6 |
| $n=500$, t(2) shape | 1.66 | 1.84 → 1.17 ± 0.21 (Stage S re-measurement, 2,000 reps; §7.2) |
| $n=2000$–$5000$ | 1.71–1.79 | 1.95–2.38 |
| $n=10{,}000$ | $1.49 \pm 0.17$ | (unmeasured) |
| $n=20{,}000$ | $1.32 \pm 0.16$ | (unmeasured) |

Regularities: $C^\*$ *decreases with n* (fixed-shape fit:
$C^\*(n) - 1 \approx 1.26\,(n/500)^{-0.32}$ — a measured decay, well fit by
a power law over $n = 500$–$2\cdot10^4$ at $\alpha \in \{.5, .2\}$; see
§7.1 for what that fit does and does not license); it is largest at tiny $n$;
it is smallest for the least-smooth truth (t(2), whose curve has a
near-corner). $C=2$ is a serviceable compression of the middle rows — which
is why the fixed map transferred well across held-out cells — but it is a
fit, not a law.

**The standing mechanism: roughness mismatch, vanishing with n.
[Heuristic; multiply fingerprinted]** The min-p depth penalizes *rough*
curves: a rough curve's local ranks decorrelate quickly across the grid, so
its minimum over $K$ points behaves like a minimum over many effective
looks; a smooth curve's rank path moves slowly and its minimum is much less
extreme. The fiducial draws carry interpolation/spacings roughness at scale
$1/n$ *on top of* the $n^{-1/2}$ signal; the truth is smooth. Hence the
truth's depth stochastically dominates a draw's depth, the credible tube at
content $1-\alpha$ over-covers the truth, and deeper trimming ($C>1$)
compensates — by an amount that shrinks as the $1/n$ roughness shrinks
relative to the $n^{-1/2}$ signal, consistent with Theorem 7
($C^\*(n) \to 1$). Fingerprints, all measured: the truth sits $\approx$3×
deeper in its cloud than a draw does at the 5% depth quantile
(binormal .95, $n=500$), with the contrast collapsing toward parity at
$n \ge 10^4$; $C^\*$ is largest at $n = 25$ and smallest for the roughest
truth; the calibration ceiling is flat along a five-fold early-slope ladder
while moving 9–13pp along a roughness-like axis; smoothing a calibration
target at fixed shape moves its calibrated depth 30×; a fiducial *candidate
curve* sits at depth 0 in $\ge 5\%$ of cases (rougher than a Hazen plug-in
curve, which is rougher than the truth), and that roughness ordering
exactly predicts the measured conservatism ordering of the calibration
routes (predictive 1.7–2.3× > plug-in 1.27× > fixed C, in trim-depth
terms).

**Ruled out (do not revisit without new ideas):**

- **The Šidák / "one budget per class" account** ($C^\* \to 2$): excluded
  by the ladder — $C^\* = 1.32 \pm 0.16$ at $n = 2\cdot10^4$ is 4.2 SE
  below a plateau at 2, decaying in step at $\alpha = .2$ (3.89 → 1.23); it
  conflicts with Theorem 7; and a parametric toy (difference of two means,
  fiducial per sample) gives $C = 1$ exactly.
- **Every data-driven calibration whose target carries $1/n$-scale
  roughness**: plug-in calibration (1.27× conservative in depth), bracketed
  worst-case over an M3 confidence set (9–37×; worst-casing *selects* the
  roughest member), fiducial-predictive calibration (1.7–2.3×; integrating
  over the predictive law amplifies the pathology). The corollary for
  design: a valid data-driven route must either de-roughen the target (a
  smoothing scale enters) or change the depth functional itself (§12, open
  problem 3).
- **Rank-computable shape functionals as a level rule**: a 32-functional,
  5-family, pre-split search found in-sample correlations of 0.7–0.9 that
  do not survive out of sample, with $\alpha=.05$ validity degrading under
  every rule that moves anything. Two identifications also eliminated: the
  operative axis is *not* a concavity defect (bimodal .90 is exactly
  concave yet sits below the binormal ladder), and *not* a rank-path
  crossing count (the natural "effective looks" proxy correlates $+0.69$
  with $C^\*$ across shapes at fixed $n$ but $-0.31$ across all cells —
  wrong sign along $n$). One clean positive: functional-driven *level*
  rules do not reintroduce the Wald co-movement bias
  ($|\rho| \le 0.25$) — the failure is lack of signal, not mechanism.
  Identification of the axis is a theory problem (the tail-excursion
  second-order analysis of open problem 1), not a search problem.

### 7.1 A working erosion law for fixed C

*(Convenient-assumptions model; validated at α=.05.)*

**Assumptions.** (A1) Theorem 7 regime, adequate M (the Monte Carlo layer
invisible). (A2) *Independent effective looks*: at one grid point a curve's
two-sided local level p within the cloud satisfies $P(p \le x) = 2x$, and
the min over the grid behaves as a min over $K_c$ independent looks, where
$K_c$ depends on the curve's roughness. (A3) The truth (smooth) has
$K_t \le K_d$ (a rough draw); write $r_n = K_t/K_d \le 1$.

**Derivation.** The trim threshold at fiducial content $1-a$ solves
$(1-2\ell^\*)^{K_d} = 1-a$; coverage is
$(1-2\ell^\*)^{K_t} = (1-a)^{r_n}$. With $a = 1-(1-\alpha)^C$ and
$C^\*(n) \equiv 1/r_n$:

$$\boxed{\ \mathrm{Coverage}_n(C) \;=\; (1-\alpha)^{\,C/C^\*(n)}\ }$$

Consequences: coverage is log-linear in $C$; the entire coverage-vs-$C$
profile at a given (shape, $n_0$, $n_1$) is one number $C^\*(n)$;
$C=1$ gives $(1-\alpha)^{r_n} \ge 1-\alpha$ — *within the model*, coverage
at least $1-\alpha$ at every $n$, converging from above (a model
consequence, not a finite-sample proof — and now measurably false outside
the model's premises: §7.2(a) exhibits a heavy-tail small-$n$ cell where
the truth exits the cloud's support and $C=1$ covers .802); $C>1$ covers at least $1-\alpha$ iff
$C \le C^\*(n)$; the asymptotic deficit under the roughness account
($r_n \to 1$) is $1-(1-\alpha)^C$. It also explains why the Šidák-form
parametrization fit the experiments so cleanly: the exponent scale is the
model's natural scale, and the fitted "2" is $1/r_n$, not the number of
classes.

**Validation. [Empirical]** Using each cell's *measured* $C^\*$ (fitted
only at the exact-95 crossing) to predict coverage at $C=1$ and $C=2$
(α=.05): binormal .95 n=500: predicted .978/.956 vs measured .975/~.963;
bimodal: .983/.966 vs .985/~.967; t(2): .973/.946 vs .968/~.943;
n=2000: .974/.949 vs .970/~.942–.95; n=5000: .979/.958 vs .967/.958. All
within ~1pp (one 1.2pp miss at 120 reps). This validates the functional
form, not the $C^\*$ values themselves (which come from the same cells).

**Rate over n. [Measured (central α) + extrapolated (tail).]** The model
does not deliver the rate of $C^\*(n) \to 1$; it is measured at central
$\alpha$ on the fixed-shape ladder to $n = 20{,}000$:
$\delta(n) = C^\*(n) - 1 \approx 1.26\,(n/500)^{-0.32}$. Precision honesty:
the fit rests on five ladder points at 120–400 reps each, each $C^\*$ an
optimized crossing (binomial error, optimization bias, and functional-form
selection all enter), so the defensible statement is "consistent with a
power law over the measured range"; the exponent is not pinned tightly, and
everything beyond $n = 2\cdot10^4$ below — including the parenthetical
$\approx 1.23$ at $n = 10^5$, $\approx 1.11$ at $10^6$ — is *model
extrapolation, not measurement*. Projected coverage for $C=2$,
$\alpha=.05$, using $(\delta_0, \gamma) = (1.26, 0.32)$:

| n/class | 500 | 2000 | 5000 | $2\cdot10^4$ | $5\cdot10^4$ | $5\cdot10^5$ | $\infty$ |
|---|---|---|---|---|---|---|---|
| coverage | .956 | .945 | .938 | .929 | .923 | .914 | .9025 |

*(Post-hoc confirmation: the Stage S screen — §7.2 — later measured $C=2$
at $n = 5\cdot10^4$, α = .05, 1,500–2,000 reps: .917/.914/.924 on
binormal .95 / kink / t(2). The $5\cdot10^4$ projection of .923 was made
before those cells ran.)*

The crossover below nominal sits near $n \approx$ 1000–2000; beyond it the
erosion is $\approx$ 1pp per decade of $n$ (slow because coverage depends
on the *ratio* $C/C^\*$ and the surplus decays as a small power).
Symmetrically, $C=1$'s coverage *surplus* above $1-\alpha$ decays at the
same pace toward 0.
Caveat on the projection: $\gamma$ is measured at $\alpha = .5/.2$; the
noisy $\alpha=.05$ $C^\*$ series (2.27 → 1.95 → 2.38 at n = 500/2000/5000)
is consistent with it but does not pin it, and the measured $.958$ at
$n=5000$, $C=2$ sits above the projected $.938$ (120 reps, SE ≈ 2pp — the
truth plausibly lies between).

**Known imperfections.** (i) The model predicts $C^\*$ independent of
$\alpha$; measured $C^\*$ rises as $\alpha$ falls (2.18 → 2.83), a
clustering effect (effective looks depend on depth) the independence
assumption omits — treat the law as validated at $\alpha = .05$ and
qualitative at central $\alpha$. A direct central-$\alpha$ check confirms
the sign of the imperfection: at $n = 10^4/2\cdot10^4$, $C=2.2$,
$\alpha=.5$ the law predicts coverage $.359/.315$ vs measured $.407/.383$ —
it overshoots the miscoverage by 1–1.5 SE there. (ii) The $\alpha=.05$
ladder at $n \ge 10^4$ (needs $M \approx 10$–12k by the §9 rule) is the one
arm that would close the tail question directly; it remains unmeasured and
is owned by `c_calibration_spec.md` D3. (iii) The plateau alternative
($\delta_\infty > 0$) is excluded down to $\delta \approx 0.32$ at
$n = 2\cdot10^4$; a plateau below that is not formally excluded but has no
support. (iv) **The mechanistic reading of (A2) is weaker than the law.**
The natural operationalization of the effective-looks ratio — median
crossing counts of a curve's local-rank path — fails even in oracle form
(§7 "ruled out" list). Min-p is an extreme-value functional: whatever
second-order analysis eventually derives $r_n$ must characterize the rate
of *deep tail excursions* of the rank path, not its median-scale
oscillation.

**Production guidance derived from this section — revised by §7.2.**
$C=1$ is the production default (asymptotically calibrated, but *not*
universally conservative — see §7.2(a) and, decisively, §7.3: the
under-coverage region is a curved (AUC, n) wedge, not a small-$n$ hole,
and M3 or the §7.3 localized M3 floor is the indicated repair inside it). Fixed $C=2$ — the former
default — is refuted: it measurably over-trims at central $\alpha$ for
$n \ge 10^4$ (coverage .41/.38 at $n = 10^4/2\cdot10^4$ against nominal
.50, at $C=2.2$, while $C=1$ gives .633/.583) *and* under-covers at
$\alpha = .05$ on heavy-tailed shapes at every measured $n$ (§7.2(b)).
The once-indicated tapered
$C(n) = 1 + \delta_0 (n_{\mathrm{eff}}/500)^{-\gamma}$ is withdrawn: the
Stage S screen found the shape envelope pinned at ~1 by t(2) at n = 500
and the t(2) taper non-monotone, so no taper of this family is both safe
and useful (§7.2(d)). The surviving deep-trim opportunity is
interior-only (§7.2(e)).

**Residual shape spread. [Empirical]** After any *level-only* remap, a
$\pm$10–15pp spread across shapes remains at central $\alpha$ (the spread
exists before recalibration too; the remap removes bias, not dispersion).
Under the roughness account the spread is the shape-dependence of the
roughness contrast; §12 lists the corresponding open problem.

### 7.2 The Stage S screen (2026-08-29): a second failure channel at the tails, and where the C-headroom actually lives

*All claims in this subsection are* **[Empirical]** *: the 27-cell Stage S
screen of `c_calibration_spec.md` (10-shape library at n = 500; taper arm
n = 100–50,000 on three shapes; imbalance arm at minority size 500;
500–2,000 reps per cell via the ladder machinery, α = .05 primary), plus
two follow-up analyses re-using its stored per-rep profiles. Data:
`data/results/c_calibration_20260829/`; verdict:
`stats/c_calibration_screening_report_stage_s.md`. The screen's
pre-registered verdict was STOP: no shape-blind level map worth fitting.*

**(a) $C = 1$ is not universally safe at small n — a validity failure the
$C$ coordinate cannot express.** At the t(2)-shape, AUC .95, n = 100/100
cell, the $C=1$ band covers **.802** at nominal .95 (.688 at nominal .80,
.742 at nominal .90). (The cell's $C^\*(.05) = 0.084$ is a
boundary-pinned artifact — the crossing lands one rung off the ladder
top, where the coverage profile never reaches .95 without the trim
collapsing entirely — so the D5 floor conjecture *as posed* is not
falsified by it; see below for the trapezoid, which is what D5 actually
asked about. The operative finding is the coverage deficit itself.)
Mechanism, read from the per-rep records: the truth
falls *outside the entire untrimmed cloud* in 1–2% of reps, and within the
deepest 2 of ~6,700 draws in 5%; misses concentrate at the two grid
corners (42% at FPR ≤ .02, 55% at FPR ≥ .90); the CP allowance contributes
nothing (allowance attribution 0). This is a channel §7's roughness
account does not describe: not a depth-law mismatch between smooth truth
and rough draws, but *unseen tail mass* — with 100 samples per class, a
heavy-tailed score distribution puts curve mass at the corners that no
draw from the observed ranks reaches. The erosion law's "coverage
$\ge 1-\alpha$ at $C=1$ for every $n$" consequence fails here because
(A2)–(A3) fail: the truth's depth is not merely stochastically deeper, it
exits the cloud's support. The effect is shape-specific at fixed n
(binormal .95 covers .992 and the kink .985 at the same n = 100) and
gone by n = 500 for t(2) (.958). Notably the *trapezoid* truth — the
designed rough adversary, the shape D5's floor conjecture was actually
about — sits comfortably at $C^\* = 2.01$: legitimate roughness is
harmless; tail mass is not, and it breaks the band in a way no trim
level can repair or express.

**(b) Fixed $C=2$ is refuted as a default.** On t(2) cells it measures
.932/.940/.924 at n = 500/5,000/50,000 and .918–.928 on the imbalance
cells (all α = .05, 2,000 reps), and .917 on plain binormal .95 at
n = 50,000. The §7.1 projection (.923 at $n = 5\cdot10^4$) is confirmed
almost exactly — measured .914–.924 across the three taper shapes — which
validates the erosion law's tail but retires the old default. **Production
default changed to $C=1$** in both implementations (same commit as this
revision).

**(c) Status of $C=1$ — SUPERSEDED 2026-09-01 by §7.3; retained as the
Stage S record.** Measured $\ge .950$ at *every* screen cell with
$\min(n_0,n_1) \ge 500$ (range .950–.981 over 10 shapes, 8 imbalance
cells, and n up to 50,000), with the surplus shrinking to ~0 at
n = 50,000 (.951/.954/.960) — the Theorem 7 approach-from-above with a
vanishing cushion, now observed. The former claim "never measured below
.967" is retired: .950–.958 at the large-n and imbalance cells. **The
"safe above $\min(n_0,n_1) \approx 500$" reading of this paragraph was
falsified by the follow-up boundary study** — the Stage S library simply
never entered the failing region (its heavy-tail members stop at AUC
.95): see §7.3 for the wedge. Inside the wedge the exact M3 band
(Prop. 12) remains the indicated fallback.

**(d) No shape-blind level map survives.** Per-shape $C^\*(.05)$ at
n = 500 spans 1.17 (t(2); 2,000 reps — superseding the noisy 1.84 of the
14-cell table) to 3.0 (binormal .60); the library lower envelope minus one
bootstrap SE is 0.97. The t(2) taper is *non-monotone*
(0.08 → 1.17 → 1.49 → 1.07 over n = 100 → 50,000), so the tapered-$C(n)$
family cannot represent the envelope. Re-evaluating arbitrary exponents on
the stored profiles: the largest C holding ≥ .95 point coverage on all
three taper shapes is ≈ 1.5 at n = 5,000 (worth ~4% area) and 1.0 at
n = 50,000 and at every minority-500 imbalance cell. The width a level
map can recover shape-blind is a few percent in a mid-n window — far
short of the per-shape oracle (9.5% mean at n = 500).

**(e) The interior headroom survives; the binding constraint is
corner-local.** Restricting attention to FPR ∈ (.05, .90): the fraction
of reps whose *worst* miss under $C=2$ lies in that interior is ≤ 3.5% at
every one of the 27 cells (nominal budget 5%) — including 1.0% at the
catastrophic t(2) n = 100 cell — and the large shape spread of (d)
compresses correspondingly. A union-bound composite (corners untrimmed,
interior at $C=2$) clears .95 at every cell (worst ≈ .953). Caveat: the
records log only each rep's worst-miss location, so interior miss rates
are optimistic by an estimated ~0.1–0.3pp (independence heuristic); the
definitive measurement builds the actual stitched band per rep (the
follow-up plan in `c_calibration_spec.md`). Two limits bound the claim.
First, **the Theorem 7 erosion applies to any fixed interior $C > 1$
exactly as it did to the full-curve $C = 2$**: interior coverage tends to
$(1-\alpha)^C$ (.926/.903/.880 at $C = 1.5/2/2.5$, $\alpha = .05$), so
the headroom measured here is *finite-range* — flat over
$n = 100$–$50{,}000$ because the erosion is slow (~1pp/decade), not
absent — and a shipping composite must taper $C_{\mathrm{int}} \to 1$ or
clamp to 1 above a declared range. Second, the corner treatment measured
is the *untrimmed cloud envelope + allowances* — an empirical widening,
not an exact distribution-free bound; a theorem-capable composite needs
an exact (M3/Beta-style) corner arm or explicitly library-relative
claims. With those bounds stated: **the deep-trim gains (~7–8% area at
$C=2$ vs $C=1$) appear recoverable on the interior over a declared
finite range if the corners are widened to carry the tail uncertainty**
— a construction change (a composite band), not a level map; see
`next_method_ideas.md` §5/§7.

**(f) A hybrid router (M3 below n = 500, $C>1$ above) was tested and
rejected on economics.** M3 measures 1.26–1.69× the $C=1$ band's area on
the n ≤ 500 screen cells (realized coverage ~1.000 — the theorem holds
with slack), while (d) caps the n > 500 side's gain at ~2–6% in a mid-n
window: the hybrid pays its width where bands are widest and harvests
where the headroom has tapered out. M3's correct role remains the
small-$n_{\mathrm{eff}}$ / guarantee-demanding regime of (c).

### 7.3 The follow-up boundary study (2026-09-01): the wedge, non-monotone coverage, and the localized M3 floor

*All claims* **[Empirical]** *unless marked: 257 student-t cells /
64,625 reps (anchors with sequential replication, a 95-cell LHS sweep,
and four active-learning rounds), plus a five-cell hybrid probe. Full
report: `stats/c_calibration_followup_report.md`; data:
`data/results/c_calibration_followup_20260830/`.*

**(a) The unsafe set is a curved wedge in (AUC, n), and coverage is NOT
monotone in n.** Worst-cased over df, $C=1$ failures at $\alpha=.05$ span
n = 102 to n = 6,656; the failing n-range widens with AUC, and at
AUC $\ge .975$ *no tested n up to 6,656 is safe*. Worst cells: t(2)/.99
covers **.645** at n = 250 and .690 at n = 500 (M3 covers .998–1.000 on
the same seeds; ladder unpinned — this is not the boundary-pinned
artifact of §7.2(a)). At fixed shape t(4.69)/.986 coverage runs
.993 → .947 → .903 → .823 → .847 over n = 150…2,000 (disjoint Wilson
intervals at the ends), so **no `n ≥ threshold` rule and no
monotone-in-n surface can express the boundary** — the sign constraint
the §7.2-era calibration surface was built on is false. A trim-grid
(K > 2001 thinning) artifact was tested and rejected. Cross-family spot
checks at the Weibull/gamma/beta-opposing achievable corners all pass,
so the t-family is binding at the corners tested; the wedge interior is
unmeasured outside the t-family.

**(b) A partial mechanism: the $m$-window.** With $t_q$ the FPR where the
true ROC reaches TPR = q and $m = n_0 t_q$ (at q = .5: expected negatives
scoring above the median positive), failures on the first 194 cells all
fell in $m \in [0.89, 11.2]$, and the window held on 28 out-of-sample
cells — the non-monotonicity in n is n carrying a shape *through* the
window in either direction. Extending n falsified it as a universal rule
(failures at m = 17.6 and 30.2 at n ≈ 5–6.7k): the window's upper edge
grows with AUC, so m compresses the boundary without linearizing it. It
is runtime-estimable without the truth, which makes it the candidate
mechanistic routing/region statistic once its AUC drift is
characterized.

**(c) Miss geometry.** Replayed from seeds: misses are overwhelmingly
lower-edge and concentrate at the *upper* FPR end (peak pointwise miss
rate at $1-\mathrm{FPR} \approx .002$–$.04$; ~70% of missing reps in
large-n cells miss only above FPR = .9), plus a secondary cluster at the
extreme left corner (FPR ≲ .005). Mechanically: heavy-tailed positives
make the true ROC approach 1 slowly, while the band's monotone lower
edge, pinned to reach 1, overshoots it — the §7.2(a) unseen-tail-mass
channel, localized.

**(d) Repairs, in preference order.** *(i) The localized M3 floor:*
pointwise union with M3 on FPR ∈ [0, .005] ∪ [.5, 1], C = 1 elsewhere,
lifts the five probed failing cells from .645–.940 to **.955–.990 at
+6.4% mean width** (full M3: +28–46%). Two structural properties:
the upper region is nearly free (both bands are compressed against
TPR = 1 there), and the union **dominates C = 1 pointwise by
construction** — it is never narrower, and monotone closure preserves the
ordering, so hybrid coverage $\ge$ C = 1 coverage identically
**[Exact]**. It is also theorem-capable: M3 at level $\alpha_2$ misses
somewhere in *any* region with probability $\le \alpha_2$ (a sub-event
of missing anywhere), so the floored region carries an exact miss cap
and only the interior claim is empirical — the two-piece statement the
composite band lacked. Caveats: five cells, 100–200 reps, region chosen
in-sample, left cutoff mis-parameterized in FPR units (it should be grid
points), width unpriced where C = 1 was already valid. *(ii) The
conservative routing rule* (AUC upper bound × n; report §5): zero
failures over all 257 cells with min coverage .944, but 65% of its
M3-routals were unnecessary and the thresholds are read from the data
that validates them. Both await fresh-seed confirmation; the floor study
is specced separately.

---

## 8. Exact finite-sample devices and what is minimax-forced at the corners

**Lemma 8 (monotone widening). [Exact]** The Clopper–Pearson upper
allowance, the zero lower allowance, monotonization, and clipping only
enlarge the band. Coverage of the final band is $\ge$ coverage of the
trimmed tube, for every realization. Validity analysis therefore
modularizes: nothing added after the trim can break it.

**Lemma 9 (corner indistinguishability; what no band can do). [Sketch]**
Let $B$ be *any* rank-based band procedure, valid at level $1-\alpha$ for
all continuous DGPs.

1. **Plateau (upper-right).** At an *exact* plateau point the forcing is
   immediate and needs no argument: $R(t_0) = 1$ and validity give
   $P_R(U(t_0) = 1) \ge 1-\alpha$ directly. The two-point content is the
   *neighborhood* version — the forcing persists at nearly-saturated
   truths. Fix $R_\delta$ with $R_\delta(t_0) = 1-\delta$ obtained from a
   saturated $R$ by moving $\delta$ of the positive class's mass above
   $t_0$; coupling each positive independently, the two joint laws differ
   in total variation by at most $1-(1-\delta)^{n_1} \le n_1\delta$.
   Validity under $R_\delta$ forces
   $P_{R_\delta}\big(U(t_0) \ge 1-\delta\big) \ge 1-\alpha$, hence
   $P_{R}\big(U(t_0) \ge 1-\delta\big) \ge 1-\alpha - n_1\delta$, and the
   same inequality read at $R_\delta$ against its neighbors gives: **no
   rank-based band valid over the class can certify an upper bound more
   than $O(1/n_1)$ below 1 anywhere within $O(1/n_1)$ of a plateau**
   (take $\delta = \beta/n_1$; probability at least $1-\alpha-\beta$).
2. **Origin (lower-left).** Moving the negative class's top-$t_0$ quantile
   mass costs at most $n_0 t_0$ in total variation and can drive
   $R$ on $(0, t_0)$ anywhere in $[0, R(t_0^+)]$. With $t_0 = c/n_0$:
   curves that differ arbitrarily below $c/n_0$ are
   $(n_0 t_0 = c)$-indistinguishable, so **no rank-based band can have a
   nonvacuous lower bound below $\sim c/n_0$** except with probability
   $\lesssim \alpha + c$. (The sharp constants of this scale were computed
   exactly via the Beta law in the envelope-era work — the
   "$\approx 7/n_0$" honesty frontier; this lemma shows the scale is
   minimax-forced, not a limitation of any particular construction.)

*In words:* the two corner behaviors of the method — an upper edge that
saturates at 1 near the plateau, and a lower edge that declines to certify
anything below a few $1/n_0$ — are not design compromises. They are the
boundary of what any distribution-free method can claim, and the method
tracks that boundary at the forced scale. Scope honesty: these are
two-point (Le Cam-style) lower-bound *sketches* that fix scales. A finished
minimax statement needs explicit continuous $(F, G)$ pairs realizing the
perturbations, the exact TV computations, and matching upper procedures
with constants — open problem 5.

**Corollary 9.3 (the point $t = 0$; why the $k=0$ allowance must stay).
[Sketch]** The same F-side move applies *at* $t = 0$: relocating the top
$c/n_0$ of the negative mass below the positives' upper tail is a
TV-$c$ modification under which $R(0)$ becomes $R(c/n_0)$ — note that
$R(0) > 0$ is perfectly possible for continuous scores (bounded negative
support with positive mass above it), so the premise "$R(0) = 0$ for every
continuous DGP" is false in general. Hence any valid rank-based band must
satisfy $U(0) \ge R(c/n_0)$ with probability $\ge 1-\alpha-c$: **pinning
$U(0) = 0$ is forbidden distribution-free.** The fiducial cloud alone has
$\tilde R(0) = 0$ identically (its interpolation places no atom at 0), so
on a separated-support truth the un-allowed band would miss at $t=0$ with
probability one — the $k=0$ CP allowance, whose value
$\approx \hat k_0/n_1 \approx R(1/n_0)$ sits exactly at the forced scale,
is what carries that case. The measured 0.1–13% of area it costs at
$t = 0$ on overlapping-support truths is therefore the minimax-forced
price of assuming nothing about supports; the pin is admissible only
under an explicit assumption $R(0) = 0$, which every suite DGP happens to
satisfy but the shipped method must not presume. Production
(`fiducial_band.py`) applies the allowance at $k = 0$ and should continue
to.

**Why the CP-form upper allowance is necessary and what it is.
[Sketch necessity + Exact widening + Empirical]** The raw fiducial upper
edge sits $O(1/n_1)$ *below* 1 near the
plateau (a credible edge reflects that the truth is *probably* not that
close to 1). Lemma 9.1 says the frequentist edge *must* reach
$1 - O(1/n_1)$-to-$1$ there. The measured consequence of omitting it:
100%-rate micro-misses (median depth $10^{-4}$) on plateau-touching shapes,
and the diagnostic $S(\text{truth}) = 0$ *in every replicate* on the
bimodal-.90 and binormal-.99 cells — the impossibility lemma biting, not an
edge case. The allowance takes the Clopper–Pearson *form* at the band's own
realized local level $\ell = j^\*/(M+1)$: no new tuning parameter. One
precision: because $\ell$ is selected from the same merged labels, the
fixed-level CP exactness theorem does not certify the allowance standalone —
it is a CP-form *widening*, not an exact confidence device in its own
right. Its two guarantees are structural: by Lemma 8 it can only widen, and
it reaches exactly the scale Lemma 9.1 forces. Its mirror on the lower
side is forced only in the
degenerate case (empirical TPR $= 0 \Rightarrow L = 0$); the full mirrored
bound is measured to cost $+15\%$ area for zero coverage — the two corners
are *not* symmetric in cost because the lower-left honesty is already
delivered by clipping and vacuousness, while the upper-right requires an
active allowance.

**Theorem 10 (ties; exactness of random tie-breaking). [Exact]** Let scores
have an arbitrary distribution (atoms allowed). Formally: augment each
score to $(S, V)$ with $V \sim \mathrm{Unif}(0,1)$ independent, ordered
lexicographically. The augmented score is atom-free in the lexicographic
order; conditional on landing in an atom of $S$, both classes' $V$'s are
iid uniform, so the population ROC of the augmented score traverses that
atom's operating segment linearly. The ROC of the refined scores is
therefore exactly the
**trapezoidal completion** of the discrete ROC (linear interpolation across
each atom's operating segment, the Mann–Whitney convention whose area is
$P(Y > X) + \tfrac12 P(Y = X)$); and Propositions 1–3 apply verbatim with
estimand $R_{\mathrm{trap}}$. Random tie-breaking is therefore not a
conservative approximation — it is exact for the trapezoidal estimand.
**[Empirical]** quantization to 20 levels is indistinguishable from
continuous scores at every $\alpha$. Conversely, ranking one class above
the other inside tie blocks changes the estimand to a staircase with
vertical cliffs at *deterministic* FPR values, which lies outside the reach
of any band (measured coverage 0.000, correctly); the implementation
refuses that convention.

---

## 9. The Monte Carlo layer

**[Exact]** Content control at finite $M$ is Lemma 6 (Lemma 6b for any other
trim score); saturation ($j^\*=1$) errs wide. **[Empirical]** The realized
local level is stable in $M$ once unsaturated and follows
$\ell(K, \alpha) \approx 9.7\cdot10^{-4}\,(\alpha/.05)^{1.2}(K/500)^{-0.27}$,
giving the budget rule $M \gtrsim 5/\ell$ ($\approx 10^4$ at
$n_0 = 5000$, $\alpha = .05$). An external anchor: Myllymäki et al.
(2017, §4.4) recommend $s \ge 2500$ simulations for rank-envelope testing at
$\alpha = .05$ (to keep the p-interval narrow) — the same order as this rule
at moderate $K$.

**Two regimes behind the $\ell(K)$ decay. [Heuristic]** The fitted
$K^{-0.27}$ conflates two effects that the Gaussian limit separates. On a
*fixed interior compact*, the limiting standardized process is continuous
with variance bounded away from zero, so as the grid refines the discrete
supremum converges to a finite continuum supremum and the required local
level converges to the **positive constant**
$\ell_I=\Phi(-w_a)>0$ from Lemma 7a — it cannot keep shrinking. Lemma 7b's
$M_n/\log K_n\to\infty$ condition governs this interior discretization,
not the full-grid practical budget. Any continuing decay must come from the
*corner strips*, where the process degenerates and the effective number of
looks keeps growing with $n$ — the regime of equal-precision band constants
(Nair 1984's $\sqrt{\log\log}$ critical-value growth;
Gontscharuk–Landwehr–Finner's one-sample ELL local-level asymptotics). In the
experiments $K = n_0 + 1$ is tied to the sample size and the grid includes
the corners, so both effects are mixed in the fit. Treat $K^{-0.27}$ as an
empirical compression over the tested range ($K \le 5001$), not a law, and
do not extrapolate it. The budget rule is self-diagnosing regardless:
$j^\*$ is computed anyway and the implementation warns at $j^\* < 3$.

**The ERL alternative.** §5.1: trimming by extreme rank length removes the
saturation failure mode (strict ordering at any $M$, given randomized
residual tie-breaking), at the cost of the exact depth–tube duality. It is the literature's standard answer to
exactly this budget problem and is untested here; if the $M \cdot K$ compute
cost at $n_0 = 10^4$ becomes binding in the suite, this is the first lever
to try.

---

## 10. Width: rates, and where the method is loose

**Interior. [Sketch]** On compacts, the tube half-width is
$q_{\ell^\*}(t)\, n^{-1/2}\sigma_R(t)\,(1+o_p(1))$ with $\sigma_R$ the
Hsieh–Turnbull standard deviation and $q_{\ell^\*}$ the Gaussian quantile at
the equalized local level; the simultaneous-band factor over pointwise
intervals is *bounded* on a fixed interior compact (the limiting local
level is a positive constant there, §9) and grows with $n$ only through
the corner strips. Measured calibration of this
picture: the band runs 1.05–1.35× the oracle rank-space ceiling on most
cells, and the ceiling itself is $\approx 1.6\times$ Working–Hotelling area
under binormal truth — the price of assuming nothing, quantified.

**Corners. [Forced scales; Sketch]** The lower band is vacuous below
$O(1/n_0)$ and
the upper band saturates within $O(1/n_1)$ of 1 near the plateau — both at
the scales forced by Lemma 9's two-point sketches, so (modulo the constants
open problem 5 would pin) no distribution-free competitor
can do more than constant-factor better there.

**The known slack.** At steep corners with small $n$ (AUC $.99$,
$n = 150$) the band runs 2–3× the oracle ceiling. Validity is unaffected.
One candidate repair is excluded: intersecting with the exact-Beta corner
edges of M3 never binds — M3's edges are 1.7–4.6× *wider* than the fiducial
band's on $k = 1..25$ at any level carrying a guarantee. The slack is
internal to the cloud/trim; the two live mechanisms are (i) the within-gap
interpolation convention — Proposition 3b's modulus bound permits a large
finite-$n$ effect where the local rise across a negative spacing is large,
even though $C^1$ smoothness makes it first-order negligible eventually —
and (ii) the global ELL budget interacting with the huge local dispersion
of the cloud where the curve is nearly vertical. Unseparated; this is the
main open *width* problem (§12), distinct from all coverage questions.

---

## 11. Behavior over n, AUC, and shape — assembled

- **Broad stability in $n$ has structural support, but is not a theorem —
  and has one measured exception.** The cloud pivots are exact at every $n$
  (Prop. 3), the corner devices are finite-sample constructions at the
  forced scale (§8), and Theorem 7 anchors the raw $C=1$ band at large $n$.
  Measured: $\alpha=.05$ coverage at $C=1$ of $.967$–$.993$ from $n=25$ to
  $5000$ on the round 1–4 cells, and $.950$–$.981$ on every Stage S cell
  with $\min(n_0,n_1) \ge 500$ up to $n = 50{,}000$. The exception —
  larger than those sweeps could see — is the §7.3 wedge: heavy-tailed
  high-AUC shapes under-cover at n from 102 to at least 6,656, with
  coverage *non-monotone in n* (the truth exits the cloud's support at
  the FPR ends; worst .645 at t(2)/.99, n = 250). The
  envelope's drift ($1.00 \to 0.83$) came precisely from asymptotics-based
  components whose regimes shifted with $n$. The important counterweight
  is §7: $C^\*(n)$ demonstrably drifts, so central-$\alpha$ calibration
  under a fixed $C>1$ is itself sample-size dependent and cannot be
  inferred from the exact marginal pivots.
- **AUC / early slope.** The envelope's AUC-degradation channel (threshold
  location at the first grid points × steep slope) is carried here by the
  F-side fiducial tail, whose marginals are the exact Beta laws
  (Prop. 4) — the same mathematics as the fix that repaired the envelope,
  now intrinsic. Measured: coverage $.978$–$.995$ at AUC $.99$ with misses
  *not* at the corner. What remains AUC-sensitive is width (§10), not
  coverage.
- **Shape.** By Proposition 2, shape is the *only* axis of sensitivity.
  Shape enters through: (i) corner degeneracies (handled exactly);
  (ii) the smoothness contrast driving $a^\*$ (§7, the residual
  central-$\alpha$ spread); (iii) the slope profile driving width
  allocation. The full simulation's LHS sweep is, in this framing, a sweep
  over shape space — the right test, and the one thing the hand-picked
  cells cannot substitute for.

---

## 12. Regimes with stronger guarantees, and open problems (ranked)

**Stronger-guarantee regimes:**

1. **One class known.** If the negative-score distribution is known (e.g. a
   calibrated null), the problem reduces to a one-sample band for
   $R = $ CDF of $W$; the F-side fiducial is replaced by the identity, the
   remaining construction is a one-sample Dirichlet band whose asymptotic
   calibration follows from Cui & Hannig's (2019) uncensored theorem
   together with Lemmas 7a–7b, the exact marginals sharpen, and exact
   test-inversion is tractable (the
   one-sample ELL machinery of `qqconf`, Weine et al. 2023, computes exact
   simultaneous levels here). **[Sketch + Lit]**
2. **Pointwise, one-sided, at anchors.** At the order-statistic operating
   points, the fiducial marginals reproduce the exact Beta pivots (Prop. 3),
   which invert to exact distribution-free confidence bounds for quantiles —
   usable when a single operating point matters. **[Exact]**

2b. **Proposition 11 (exact Monte Carlo test at a named curve). [Exact;
   = the global rank envelope test of Myllymäki et al. 2017, instantiated
   for ROC curves.]** For any fixed continuous $R_0$,
   $H_0: R = R_0$ is *simple* in rank space (Prop. 2), so the rank envelope
   test applies off the shelf: simulate $s$ datasets from $R_0$, compute
   the empirical ROC of each on the grid, rank the observed empirical ROC
   among the $s+1$ curves by min-p depth (or ERL), and report the Barnard
   p-value — exact by exchangeability (their Lemma 3.1; with the min-p
   depth's ties, the p-interval $(p_-, p_+]$ of their eq. 10 and
   Prop. 4.1; the ERL ordering with residual ties broken by independent
   randomization — their lexicographic $(T_i, M_i)$ device — collapses the
   interval to a unique exact p-value; without the randomization a narrow
   p-interval remains, since rank vectors are discrete and can tie).
   Their Theorem 4.2 adds the graphical interpretation for free:
   the test rejects iff the observed curve exits the $k_\alpha$-th rank
   envelope of the simulations, so the plot shows *which FPR regions*
   drive the rejection. The implemented two-cloud variant (fixed reference
   cloud A for the depth, second independent sample for the null law of
   the depth) is exact by the same conditioning argument and was the form
   measured: size .190–.202 at $\alpha=.2$ and .046–.051 at $\alpha=.05$
   on five (shape, n) cells including two where WH has 0.000 coverage;
   power .23–.26 at $|\Delta AUC| = .01$ and .71–.85 at $.02$
   ($n = 500$). Structural power profile: the equal-local-levels statistic
   spreads its budget across the grid, so it pays a measured ≈2.5× worse
   sup-norm exchange rate against *localized* early-FPR alternatives than
   against global shifts, and detects a corner pushed down more easily
   than pushed up. Three consequences: (i) exact confidence sets over any
   *finite family* of named curves by inversion (one null simulation per
   curve); (ii) the gap between this and the band is precisely stated —
   the band is the pointwise projection of the inversion over *all*
   curves, and that projection, not exactness, is the obstacle; (iii) the
   band itself is *not* the inversion of this test family (the band's
   cloud conditions on the data; the test's cloud conditions on $R_0$),
   which is why the band needs the $C$-remap while the test needs nothing.
   **Composite/fitted nulls** (e.g. a binormal fit, as a WH diagnostic)
   break simplicity; the options are sample splitting (fit on one half,
   test exactly on the other) or the adjusted-envelope two-stage method of
   Dao & Genton (2014) as adapted to global envelopes by Myllymäki et al.
   (2017, §7) — approximate but well studied; neither is covered by the
   exactness claim as stated.
3. **A certified outer band — M3, now a production method**
   (`src/studroc_paper/methods/m3_band_rs.py`; exact-calibration kernel
   `ell::crossing_prob` in the `fiducial-core` Rust crate; the experimental
   harness `m3_experiments.py` retains the earlier MC-calibrated variant).
   M3 is the theorem-carrying layer next to the tighter fiducial band.

   **Proposition 12 (M3 coverage theorem). [Exact]** Write the negative and
   positive scores in descending order as $X_{[1]}\ge\cdots\ge X_{[n_0]}$
   and $Y_{[1]}\ge\cdots\ge Y_{[n_1]}$, and let
   $A(x)=1-F(x)$ and $B(x)=1-G(x)$ be their survival functions. Let
   $b^{lo}_0, b^{hi}_0$ be the two-sided equal-local-levels bounds for the
   pivots $A(X_{[i]})$ at local level $\gamma_0$ ($b^{lo}_0[i] =
   \mathrm{BetaInv}(\gamma_0; i, n_0{+}1{-}i)$, $b^{hi}_0[i] =
   \mathrm{BetaInv}(1{-}\gamma_0; \cdot)$), and $b^{lo}_1, b^{hi}_1$ the
   corresponding bounds for $B(Y_{[j]})$ at $\gamma_1$. Let $p_i$ be the
   number of positives ranked above the $i$-th ranked negative (descending
   scores), and define on the grid $t \in \{k/n_0\}$:

   $$U(t) = b^{hi}_1\big[\,p_{i_{up}(t)} + 1\,\big],\quad
     i_{up}(t) = \min\{i : b^{lo}_0[i] \ge t\}\ \ (U = 1 \text{ if none});$$
   $$L(t) = b^{lo}_1\big[\,p_{\,i_{lo}(t) - 1}\,\big],\quad
     i_{lo}(t) = \min\{i : b^{hi}_0[i] \ge t\},$$

   with $b^{lo}_1[0] = 0$, $b^{hi}_1[n_1{+}1] = 1$, and $L(1) = U(1) = 1$.
   If the upper-edge minimum does not exist set $U=1$; if the lower-edge
   minimum does not exist set $i_{lo}=n_0+1$. Let $E_A$ be the
   event that all
   $b^{lo}_0[i]\le A(X_{[i]})\le b^{hi}_0[i]$, and $E_B$ the analogous
   event for $B(Y_{[j]})$. Then on $E_A\cap E_B$,
   $L(t)\le R(t)\le U(t)$ for all $t$, hence

   $$P(\forall t: L \le R \le U) \ \ge\ P(E_A)\,P(E_B)
     \ \ge\ (1-\alpha_F)(1-\alpha_G) \ =\ 1-\alpha$$

   with the split $(1-\alpha_F) = (1-\alpha)^{\rho}$, $(1-\alpha_G) =
   (1-\alpha)^{1-\rho}$, any fixed $\rho \in (0,1)$ (default
   $\rho = \tfrac12$, the Šidák split; the classes are independent, so the
   product step is exact, and the calibration below guarantees
   $P(E) \ge 1-\alpha_c$ per class — the middle inequality is an equality
   up to the calibration's bisection tolerance).

   *Proof.* The probability-integral transform gives
   $A(X_{[i]})\sim\mathrm{Beta}(i,n_0+1-i)$ jointly as uniform order
   statistics, and likewise for $B(Y_{[j]})$. This use of the original
   continuous class CDFs is essential: the placement CDF $R$ itself can
   have jumps when $F$ has support gaps, so applying a continuous-CDF pivot
   directly to $R(W_{(j)})$ would be invalid.

   Put $q_t=Q_F(1-t)$, so $R(t)=B(q_t)$. For the upper edge, on $E_A$,
   $A(X_{[i_{up}]})\ge b^{lo}_0[i_{up}]\ge t$, hence
   $q_t\ge X_{[i_{up}]}$ almost surely and
   $R(t)\le B(X_{[i_{up}]})$. If
   $m=p_{i_{up}}$ positives exceed $X_{[i_{up}]}$, then
   $B(X_{[i_{up}]})\le B(Y_{[m+1]})\le b^{hi}_1[m+1]$ on $E_B$,
   with value 1 when $m=n_1$ or the negative index does not exist.
   For the lower edge, $i_{lo}=1$ gives the vacuous bound 0. If
   $i_{lo}\ge2$, minimality and $E_A$ give
   $A(X_{[i_{lo}-1]})\le b^{hi}_0[i_{lo}-1]<t$, hence
   $q_t\le X_{[i_{lo}-1]}$ and
   $R(t)\ge B(X_{[i_{lo}-1]})$. With
   $m'=p_{i_{lo}-1}$ positives above that score,
   if $m'=0$ the lower bound is vacuous, while otherwise
   $B(X_{[i_{lo}-1]})\ge B(Y_{[m']})\ge b^{lo}_1[m']$ on $E_B$.
   The statements hold almost surely; equality ambiguities have probability
   zero under continuity and the generalized inverses give the same weak
   inequalities. Independence of the samples makes $E_A,E_B$ independent.
   Finally $L(1)=U(1)=1$ because $Q_F(0)=-\infty$ and $R(1)=1$; under
   ties with random tie-breaking, Theorem 10 applies verbatim with the
   trapezoidal estimand. $\square$

   **Exact calibration. [Exact]** $P(E)$ for a one-sample two-sided ELL
   band is a non-crossing probability of uniform order statistics,
   $P(\forall i:\ l_i \le U_{(i)} \le h_i)$, computed *exactly* by a
   counting-process dynamic program over the sorted bounds: conditionally
   on $N(c) = j$, increments are Binomial, and (both bound sequences being
   monotone) the alive states form the contiguous window
   [#upper bounds passed, #lower bounds pending], giving
   $O(\sum \text{width}^2)$ cost and no underflow (exact to
   floating-point roundoff; verified against closed forms and Monte
   Carlo). $\gamma_c$ is then set by bisection, returning the conservative
   bracket endpoint, so each class band has simultaneous level *at least*
   $1-\alpha_c$ and within the bisection tolerance of it — the MC
   calibration and its 2-SE safety shading are gone; the remaining
   conservatism in M3 is the product split, the worst-case composition,
   and the (negligible, one-sided) bisection tolerance. (This is the same
   computation as the `qqconf` ELL bands, Weine et al. 2023; measured
   cost: 0.06s at $n = 500$, 47s at $n = 10^4$, one-time per
   $(n, \alpha_c)$ and cached; ≈1.5ms per band thereafter.)

   **The $t = 0$ corner, done right. [Exact]** The production band does
   *not* pin $U(0) = 0$: by Corollary 9.3 that pin is invalid
   distribution-free ($R(0) > 0$ is possible). The composition's own value
   at $t = 0$ is $b^{hi}_1[p_1 + 1]$ — the exact Beta bound at the count
   of positives ranked above the top negative — which sits precisely at
   the minimax-forced scale and carries the separated-support case
   ($p_1 = n_1 \Rightarrow U(0) = 1$, correctly). `assume_r0_zero=True`
   opts into the pin under a user-asserted $R(0) = 0$. (The round-3
   experimental harness pinned by convention; its measured coverages are
   unaffected because every suite DGP satisfies $R(0)=0$.)

   **The split ratio $\rho$ as a lever. [Untested beyond default]** Any
   fixed $\rho(n_0, n_1)$ preserves the theorem (data-independent). The
   round-4 finding that M3's worst-case remap is bound by the 9:1
   *imbalance* cell, not by shape, makes an imbalance-aware $\rho$ the
   obvious candidate tightening — spend less of the budget on the class
   whose band is already tight. Cheap to sweep on the existing cells.

   **Measured profile (unchanged by the exact calibration, which only
   removes the shading slack):** coverage 1.000 at $\alpha=.05$ on all 8
   cells; area 1.45–2.19× the production fiducial band (0.37–0.88× KS —
   it strictly dominates the provable baseline); production spot-check
   1.58× at binormal ~.875, $n=500$. The whole penalty is *level
   accounting*, not geometry: at the nominal level whose realized coverage
   is .95, its area is 0.93–1.05× the fiducial band's. The
   worst-case-remap ceiling over the 14-cell library: a fixed nominal
   α′ = 0.5 covers ≥ .95 everywhere at mean 1.21× the production band's
   area — but with one ladder step of margin, a required-α′ drift of
   ≈ 0.13 per decade of n at fixed shape (the composition slack grows with
   n: the same liability class as fixed C = 2), and the binding cell set
   by 9:1 class imbalance rather than any shape. The accounting factor
   (12–16× in α) cannot be recovered by a fixed remap without forfeiting
   the theorem; a *provable* tightening of the split (the $\rho$ lever) or
   of the worst-case composition is the only route. Two hoped-for roles
   are closed: the miss cap (fiducial ∩ M3(α/10)) is free but inert (never
   bound in 10,400 checks; certificate weak — proves miss depth
   ≤ 0.10–0.90 where the observed worst case is 0.01–0.06, near-vacuous
   exactly where Lemma 9.2 says it must be); and the **domination route to
   a finite-sample theorem is dead** (M3(α′) ⊆ fiducial(.05) essentially
   never holds, at any α′ up to .999, on any cell, interior-restricted or
   not). What survives: M3 as the certification layer when a theorem is
   required, at its honest width. **[Exact validity; measured width]**

**Open problems, ranked by importance:**

1. **Prove the $C^\*(n)$ taper (the roughness-mismatch mechanism).** The
   taper is real and measured ($C^\*(n)-1 \approx 1.26(n/500)^{-0.32}$ to
   $n = 20{,}000$, central $\alpha$). Remaining: the $\alpha=.05$ ladder at
   $n \ge 10^4$ (owned by `c_calibration_spec.md` D3), and a proof via
   second-order analysis of the min-p functional under a rough-vs-smooth
   contrast (Edgeworth/strong-approximation for the two-sample
   Dirichlet-weighted ROC process). The local-level asymptotics of
   one-sample ELL bands (Gontscharuk–Landwehr–Finner 2015/2016) are the
   nearest existing theory and characterize exactly the quantity — the
   effective-looks growth of an equal-local-levels minimum — that the
   proof needs; the crossing-count operationalization is already excluded
   (§7), so the analysis must target deep tail excursions of the rank
   path.
2. **A finite-sample coverage theorem** for the trimmed tube plus
   allowances. §7.2(a) sharpens what any such theorem must confront: at
   small $n$ under heavy tails the truth exits the cloud's support at the
   grid corners, so no theorem can hold for the current construction
   without either a corner widening (the localized-M3-floor direction of
   §7.3(d), which is theorem-capable because M3's guarantee restricts to
   any region, or the composite direction of §7.2(e)) or a domain
   restriction — and §7.3 shows the excluded domain is a curved (AUC, n)
   wedge, so "the current band above some $n_{\mathrm{eff}}$" is not a
   provable object either. The full-band domination-by-M3
   route is ruled out empirically; the
   exchangeability/conformal embedding route remains (the construction
   conditions on $\Lambda$, which is exactly what breaks exchangeability;
   quantifying the gap is the problem). A constructive relaxation worth
   recording: exact inversion of the Prop. 11 test over a *finite or sieve
   family* of curves is certified as-is (one null simulation per member),
   and its pointwise projection is a certified band *for truths in the
   family*; quantifying the approximation cost of a sieve over shape space
   — how fine a family buys validity over a smoothness class — is a
   concrete, attackable version of this problem.
3. **Fix the central-$\alpha$ shape spread at the source: change the depth
   functional.** Every level-side fix is dead (§7). The measured mechanism
   (draws rougher than truth, contrast concentrated in the lower depth
   tail) points at the functional: (a) **smoothed-depth trimming** — rank
   each draw by the min-p depth of its *smoothed* version, trim by that
   score, band = envelope of the retained raw draws; content control is
   Lemma 6b, the depth–tube duality is sacrificed, and a smoothing scale
   enters. (b) **ERL trimming** (§5.1) — worth testing in the same
   derisk: it re-weights exactly the deep-tail excursions that drive the
   contrast, is parameter-free, and is the literature's standard
   refinement. A 3-cell derisk (C2/C5/C4 at α ∈ {.5,.2,.05}) on the Rust
   core would show whether either equalizes the truth-vs-draw depth laws.
4. **Steep-corner width** — whether a shape-aware within-gap interpolation,
   the conservative interval-valued variant of §3.1 (which is *widest*
   exactly at the corner, possibly closing the honesty gap from the safe
   side), or a locally reweighted trim can close the 2–3× gap to the
   oracle ceiling without touching validity. A corner-only experiment
   separating mechanisms (i) and (ii) of §10 is cheap; Proposition 3b's
   modulus rate supplies the natural design coordinate, the true rise
   across a typical maximal negative spacing.
5. **Sharper corner constants** — Lemma 9 gives scales via two-point
   sketches; the exact Beta computations give constants for specific
   events; a clean minimax theorem (explicit continuous perturbation pairs,
   exact TV, matching upper procedures) would tie the corner story shut.

**Scope caveat.** Proposition 2 — and with it everything downstream — is
specific to the two-independent-samples, single-marker design. Paired
designs, covariate-adjusted ROC curves, censoring, and multiple correlated
markers all break the maximal-invariant reduction; none of the guarantees
here transfer automatically, and each needs its own treatment. Conversely,
two extensions are nearly free within the design: a *partial-ROC* band on a
prespecified FPR window (run the ELL trim over the sub-grid only, spending
the whole budget there), and band-inherited simultaneous confidence
statements for monotone functionals (AUC, partial AUC, TPR-at-fixed-FPR:
the interval $[\phi(L), \phi(U)]$ is simultaneously valid whenever the band
is, at the price of conservatism).

---

## 13. Design → property map (summary)

| Design element | Property it purchases | Status |
|---|---|---|
| Rank-only inputs | Coverage depends on (shape, $n_0$, $n_1$) only; family invariance is a theorem; exact simulability per shape | [Exact] |
| Dirichlet spacings per class | Exact Beta pivots at every order statistic, every $n$; = the nonparametric spacings-GFD ($(n{+}1)$ spacings — distinct from the $n$-weight Bayesian bootstrap, which pins the extremes) | [Exact + Lit] |
| Fiducial mass beyond extremes | Two-sided corner uncertainty; the bootstrap's one-sided support collapse cannot occur; old Beta floor subsumed | [Exact] |
| Within-gap convention | Bounded by one gap mass (Prop. 3b); first-order irrelevant under local Hölder exponent $>1/2$ (up to logs); potentially material at rough regions or jumps | [Exact bound; Sketch rate] |
| Curve-valued draws (not pointwise intervals) | Monotone, $[0,1]$-respecting, correctly correlated bands; both HT variance channels carried without density estimation | [Sketch] |
| Min-p / equal-local-levels trim | Simultaneity without a variance estimate (= the correctly-studentized region in the Gaussian limit, Narisetty–Nair Cor. 1); balanced miss directions; spread miss locations; graze-type misses | [Exact structure; balance asymptotic] |
| Trim level from the cloud's own quantile | Finite-$M$ content control for *any* trim score (Lemma 6b); conservative failure mode under saturation; self-diagnosing budget ($j^\*$) | [Exact] |
| $C$-remap of the trim level | Finite-sample centring at central $\alpha$; asymptotically the trim level is the coverage (Thm 7), so fixed $C>1$ is a finite-sample device with a known asymptotic liability; erosion law §7.1 | [Empirical + Sketch] |
| CP-form upper allowance at level $j^\*/(M+1)$ | Matches the plateau scale forced by Lemma 9.1; pure widening (Lemma 8); parameter-free. Not a standalone exact device — the level is data-selected | [Sketch necessity; Exact widening] |
| Degenerate lower allowance | The forced lower-left mirror, at zero width cost; full mirror provably not worth it | [Exact + Empirical] |
| Random tie-breaking | Exact reduction under ties with trapezoidal estimand; no conservatism needed | [Exact] |

**The method space in one taxonomy.** The GET package's envelope types map
this project's whole history onto one axis (what is ordered) times one
choice (what cloud): `'unscaled'` (constant-width MAD) = the KS-type band;
`'st'` (studentized MAD) = the old envelope method's interior statistic;
`'rank'` (extreme rank / min-p) = this method's trim; `'erl'` = the §5.1
refinement. The envelope era applied a `'st'`-type ordering to a *resampling
bootstrap* cloud and needed two exact floors to patch that cloud's corner
failures; the current method applies a `'rank'` ordering to the *fiducial*
cloud, whose corners are exact by construction. The cloud, not the
ordering, was the load-bearing change.

---

## 14. Position in the literature

*(First web pass 2026-08-23; full texts of the four key papers read the
same day: Myllymäki et al. 2017 [arXiv:1307.0239v4], the GET JSS paper
[arXiv:1911.06583], Cui & Hannig 2019 [arXiv:1707.05034v3], Narisetty &
Nair 2016 [arXiv:1511.00128]. Gu–Ghosal–Roy 2008 is paywalled; its
construction is confirmed from the ROCnReg package and citing papers, and
its pointwise-only scope was verified directly against the ROCnReg
implementation.)*

### 14.1 Components with direct prior art (cite, do not re-derive)

- **The trim and tube = global rank envelope.** Myllymäki, Mrkvička,
  Grabarnik, Seijo & Hahn (2017), "Global envelope tests for spatial
  processes," JRSS-B 79:381–404: extreme rank = min-p depth (Lemma 5's
  $S$, in their maximum-rank tie variant); the $[k,k]$-order-statistic
  envelope; Theorem 4.2 = the depth–tube duality with the p-interval
  $(p_-, p_+]$; Lemma 3.1 = the Barnard/Besag–Clifford exchangeability
  argument behind Proposition 11; §6.1 = the ERL refinement of §5.1;
  §4.4 = the $s \ge 2500$ budget guidance; §7 = the Dao–Genton
  composite-null adjustment. The GET R package (Myllymäki & Mrkvička,
  J. Stat. Software 111(3), 2024) implements all of it, including "global
  confidence bands" from bootstrap/posterior curve clouds — with the
  level always taken as the cloud's own content (the $C=1$ arm; their
  eq. 2 defines coverage *under the sampling scheme's distribution*).
- **The depth as a population object = extremal depth.** Narisetty & Nair
  (2016), JASA 111:1705–1714: the d-CDF left-tail-stochastic ordering
  (their finite-sample comparison at the smallest depth level is min-p;
  the full lexicographic ordering is ERL's population version);
  Prop. 4 = exact within-cloud content (our Lemma 6 in their setting);
  Cor. 1 = ED central regions of a Gaussian process are exactly
  $\pm w\,\sigma(t)$ — the equal-local-levels/studentized identification
  used in §5.2 and Theorem 7(3).
- **The tube from sampled curves.** Besag, Green, Higdon & Mengersen
  (1995), Statist. Sci. 10:3–66 (p. 30): simultaneous credible bands from
  MCMC curve draws via the $k$-th order-statistic envelope — the earliest
  antecedent found for "trim a cloud of curves by rank, take the pointwise
  envelope."
- **The nearest existing cloud: Bayesian-bootstrap ROC.** Gu, Ghosal &
  Roy (2008), Statistics in Medicine 27:5407–5420: $n$-weight Dirichlet
  (Bayesian bootstrap) draws per class composed through the
  placement-value representation into ROC draws; used for point
  estimation, *pointwise* credible intervals (verified pointwise in the
  `ROCnReg` implementation, `pooledROC.BB`), AUC, and a binormality
  diagnostic. Adjacent to, but **not identical with**, this method's
  cloud: the BB pins each class CDF at the observed extremes
  (Beta$(j, n-j)$ marginals, no mass between or beyond observations),
  where the spacings-GFD carries the corner channel (§3) — an $O_p(1/n)$
  difference, invisible at first order and load-bearing exactly at the
  corners and in the §7 roughness story. No simultaneous band, no
  frequentist calibration of a band, no corner analysis. Follow-ups: Gu &
  Ghosal (2009, rank-likelihood binormal); Inácio de Carvalho et al.
  (Bayesian-bootstrap ROC surfaces); the `ROCnReg` R package
  (Rodríguez-Álvarez & Inácio 2021).
- **The fiducial theory of the marginal clouds.** Cui & Hannig (2019),
  Biometrika 106:501–518 (with discussion): the nonparametric GFD for a
  CDF/survival function — interval-valued inverse image $[F^L, F^U]$
  (uncensored case: exactly Prop. 3's Dirichlet law at the order
  statistics, with the between-observation mass carried as an interval
  rather than interpolated); a functional Bernstein–von Mises theorem
  (their Thm. 3.2; convergence in distribution *almost surely* of the
  centered, scaled GFD to the Kaplan–Meier Gaussian limit) and its
  content-to-coverage corollary (Cor. 3.1) — the one-sample engine of
  Theorem 7; a concentration inequality (their Thm. 3.1) that may serve a
  finite-sample analysis (open problem 2); and a sup-norm curvewise band +
  two-sample difference test built on the same cloud. Their curvewise band
  is a *constant-shape* sup-norm band around the pointwise median — GET's
  `'unscaled'` type, not equal-local-levels — and its calibration is the
  cloud content.
- **Equal-local-levels bands for distribution functions.** Berk & Jones
  (1979); Nair (1984) equal-precision bands; Owen (1995); Jager & Wellner
  (2007); Aldor-Noiman et al. (2013) ELL QQ bands (whose theoretical
  envelope GET's authors credit as an advantage over simulation);
  Gontscharuk, Landwehr & Finner (2015, 2016) — asymptotics of ELL local
  levels (the one-sample theory of §9's slow $\ell(K)$ decay and a
  candidate door for open problem 1); Finner & Gontscharuk (two-sample
  KS-type tests in terms of local levels — the nearest two-sample ELL
  theory, for the difference of EDFs, not the composition); Weine, McPeek
  et al. (2023), the `qqconf` package — fast exact one-sample ELL levels
  (usable by M3's components). Westfall & Young (1993) — the min-p
  resampling frame.

### 14.2 The ROC-band comparison set

Campbell (1994) — bootstrap and fixed-width ROC bands; Hsieh & Turnbull
(1996) — the weak-convergence limit used throughout; Hall, Hyndman & Fan
(2004), Horváth, Horváth & Zhou (2008), Bertail, Clémençon & Vayatis
(2009) — bootstrap ROC band validity theory (asymptotic, interior);
Macskassy, Provost & Rosset (2005) — the empirical evaluation whose finding
(only fixed-width KS-type bands attain containment out of the box) is the
published statement of the gap this method fills; Claeskens et al. (2003) —
empirical-likelihood bands; Ma & Hall (1993), Working–Hotelling — the
parametric incumbent; a generalized-inference (GPQ) band for the *binormal*
ROC (Statistics in Biopharmaceutical Research 8(1), 2016) — fiducial-
flavored but parametric, a WH competitor rather than an antecedent. None of
these are rank-exact, none address the corners at the $1/n$ scale, and the
distribution-free members are constant-width.

### 14.3 What appears to be novel here

The honest overall assessment: **moderate, combination-driven novelty**.
Every individual component exists somewhere; what appears new is the
assembly, plus one empirical phenomenon. Specifically:

1. **The two-sample composition of one-sample spacings-GFDs, used as a
   simultaneous confidence band.** Gu–Ghosal–Roy compose the *adjacent*
   ($n$-weight BB) cloud but use it pointwise; Cui–Hannig have the
   one-sample GFD with band theory but no composition and a sup-norm
   (unscaled) band shape; the global-envelope literature has the trim but
   applies it to null-simulated or bootstrap clouds at face-value content.
2. **The credible-to-confidence calibration study** (§7): the calibration
   function $a^\*/C^\*$, the roughness-mismatch mechanism and its
   fingerprints, the measured $C^\*(n) \to 1$ taper, and the erosion law
   $(1-\alpha)^{C/C^\*(n)}$. The antecedent literatures take cloud content
   as the level; the second-order gap is not analyzed anywhere we have
   found. Current status honestly stated: an empirical phenomenon plus a
   toy effective-looks model — it becomes the intellectually novel core
   only if the second-order analysis (open problem 1) is completed.
3. **The corner-necessity sketches tied to the CP-form allowance**
   (Lemma 9, Cor. 9.3): two-point lower bounds at the $1/n$ scale matched
   by a parameter-free widening device inside the band — an application of
   standard Le Cam ideas whose value is the pairing with the construction,
   not the technique.
4. **Exact simulability as a design principle** (Prop. 2 used to make
   shape the only calibration coordinate, powering the offline $C$-map and
   the named-curve test's practicality for ROC) — likewise an application
   of standard rank-experiment ideas, valuable for what it organizes.

Two accounting notes. First, claim (1) needs one further check before
publication: whether any applied global-envelope or fiducial paper composes
two independent clouds through a nonlinear map into a band (Cui–Hannig's
two-sample work uses the *difference*, not a composition; nothing found for
compositions) — and absence from a targeted search is evidence, not proof.
Second, the method is currently more novel than any single theorem proved
about it. Theorem 7's construction-specific gaps are now reduced to
explicit lemmas; the strongest available upgrade is a line-by-line
verification of the cited conditional process theorem in the chosen
function space, with the $C^\*$ study presented as its empirical companion
rather than as a result.

### 14.4 Still to verify at full text

Praestgaard & Wellner (1993) (exact conditions for Dirichlet weights); Lo
(1987) and Weng (1989) (large-sample/second-order BB theory — potentially
relevant to open problem 1); Gontscharuk–Landwehr–Finner (the precise
local-level rates); Gu–Ghosal–Roy (paywalled; the pointwise scope is
verified against the ROCnReg implementation, but the paper's own text
remains unread); Frey (2008) (optimal-width distribution-free
bands, for the width discussion of §10).
