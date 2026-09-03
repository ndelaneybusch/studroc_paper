# Theory of the Rank-Space Fiducial ROC Band

*Companion to `stats/next_method_ideas.md` (working model and evidence) and
`src/studroc_paper/methods/fiducial_band.py` (implementation). This document
develops the probabilistic structure behind the method: what is exactly true,
what is asymptotically true, what is finite-sample heuristic, and what is
open. Last substantive revision 2026-09-02: §7.4 derives the convex-corner
failure mechanism, a finite-grid risk score, a shape-class router, and
principled M3 floor regions; §7.4(h) separates the rank-only floor from the
declared-class case. Section 7.3 records the 257-cell empirical wedge and
localized-floor probe. Previous revisions: 2026-08-30 (§7.2, Stage S and
the $C=1$ default) and 2026-08-23 (§14 literature pass).*

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
$\alpha_{\mathrm{eff}} = 1-(1-\alpha)^C$ (default $C=1$ since 2026-08-30, §7.2) with realized depth
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
Every calibration cell since round 1 is simulated in rank space, so the
theorem is exercised rather than tested; consistent with it, the 257
student-t cells of §7.3 organize by (AUC, tail index, n) alone.

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
5. **Where it actually bites: the end gaps at a convex corner (§7.4).
   [Exact mechanism; Sketch rates]** The sorted-uniform placement is the
   fiducial implementation of "the likelihood ratio is constant across the
   gap", i.e. the true ROC is *linear* between consecutive same-class
   operating points. In the end gaps — below the lowest positive, above the
   highest negative — that gap carries $O(1)$ of the local curve deficit, and
   a heavy-tailed truth is exactly there most nonlinear (the ROC is *convex*,
   the classical hook). Lemma 13 shows the convention is calibrated iff the
   ROC is linear across the end gap and anti-conservative iff it is convex
   there, with endpoint miss approximations. (The finite-grid score screens
   the measured wedge without a false negative at its conservative cutoff;
   §7.4(g).)

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
(3) predicts on the interior. Both tested conventions retain the same
endpoint-linearity premise, so this does not test §7.4's failure mechanism.

**A conservative interval-valued variant (untried as a full band; its
tail restriction is, up to the level, the M3 floor — §7.4(e)).** Following
Cui & Hannig's conservative option: compose, per draw, the *bracket*
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
ensemble member. Code parity checked 2026-09-03: on one exported cloud
(t(2)/.99, $n_0=n_1=500$, $M=2000$, $K=501$) GET 1.0.9's
`central_region(type="rank", coverage=.95)` returns edges identical to the
production C = 1 tube at every grid point, max difference 0, retained
fraction .9510; `type="erl"` is strictly narrower at 498/501 points as the
§5.1 sandwich predicts — `scripts/c_calibration/rocnreg_bb_check/`]**
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
  envelope's measured $\sim$10:1 downward skew). Inside the §7.3 wedge the
  endpoint mechanism breaks this balance downward (.301/.013 at
  t(2)/.99, $n=500$).
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
is §7's.) At $n=50{,}000$, Stage S measured .951–.960 coverage on its three
taper shapes. The scope limitations are real and stated: the theorem is
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
level can repair or express. **Sharpened 2026-09-02:** the channel is now
derived (§7.4). "Unseen tail mass" is the right picture, but the
quantitative driver is the within-gap convention's implicit *linearity*
assumption failing at a convex corner: what matters is the ratio of the
average likelihood ratio across the end gap to that at the outermost grid
point (the hook ratio), not the tail mass itself.

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
characterized. **Explained 2026-09-02 (§7.4(c)):** for heavy tails
$t_{.5} = P(X > \delta) \approx (1-\mathrm{AUC})/2$, so
$m \approx N_0 := n_0(1-\mathrm{AUC})/2$ — the coordinate in which the
t-family's corner mechanism depends on $(N_0, N_1, \nu)$ alone. The
window's "AUC drift" was the tail index: at fixed $\nu$ the failing set is
a window in $N$ whose upper edge grows steeply with $\nu$, and the large-$n$
failures are the moderate-df shapes.

**(c) Miss geometry.** Replayed from seeds: misses are overwhelmingly
lower-edge and concentrate at the *upper* FPR end (peak pointwise miss
rate at $1-\mathrm{FPR} \approx .002$–$.04$; ~70% of missing reps in
large-n cells miss only above FPR = .9), plus a secondary cluster at the
extreme left corner (FPR ≲ .005). Mechanically (derived in §7.4): in
the zone where the empirical TPR is 1, the cloud spreads the positive mass
below the lowest *observed* positive uniformly over the $k_{\mathrm{sat}}$
negatives there — the "ROC is linear here" assumption — so the lower edge
claims a TPR deficit of order $\ln(1/\ell)/(n_1 k_{\mathrm{sat}})$, while a
heavy-tailed truth's deficit at the outermost grid points is of order
$1/n_1$ (its likelihood ratio is largest at the extreme end). The left
cluster is the mirror image on the F-axis at the first grid point.

**(d) Repairs, in preference order.** *(i) Historical localized-floor
probe:* pointwise union with M3 on FPR ∈ [0, .005] ∪ [.5, 1], C = 1
elsewhere, lifts the five probed failing cells from .645–.940 to
**.955–.990 at +6.4% mean width** (full M3: +28–46%). Those measurements
are development evidence because the region was chosen on the same cells.
The revised closure only widens, so the hybrid dominates C = 1 pointwise
and contains M3 on its random region **[Exact]**. Consequently the region
has the exact M3 miss cap and the whole-curve bound splits into that cap
plus an empirical exterior term.

*(ii) Frontier floor after §7.4:* the Stage F primary rule is fixed before
outcomes. On the left it uses a budget-derived grid schedule that
upper-bounds $\lceil Q\rceil$; on the right it uses the complete empirical
TPR-1 run plus a predeclared $2\sqrt K$ inward margin. This rule is a
function of class sizes and ranks only. Stage F prospectively tests its
capture, price, sliver behavior, imbalance, and geometry-class transfer;
it does not fit an AUC-conditioned region.

*(iii) Router:* the measured AUC-upper-bound × n rule had zero failures
over the 257 development cells (minimum .944) but routed 65% of its M3
cases unnecessarily. Proposition 14 now classifies it correctly: it is a
student-t/library-relative diagnostic, not a distribution-free rule. Any
router successor must declare a curvature/shape class and maximize the
finite-grid risk over its complete AUC uncertainty set.

### 7.4 Endpoint curvature, a finite-grid risk score, and principled floors

*Status. The constructional claim in (a) and the Student-t hook locations
in (b) are exact. Lemma 13 is a Poisson/Dirichlet endpoint approximation,
not a coverage theorem. Its finite-grid refinement in (c) is
parameter-free; only the map from its risk score to whole-band coverage is
empirically calibrated. The floor containment statements are exact, while
the proposed floor extents remain to be externally tested in Stage F.
Reproduction: `scripts/c_calibration/corner_mechanism.py`.*

**(a) What the convention assumes. [Exact]** On each axis, own-class
elements sit at their Dirichlet cumulative masses and the other class's
elements inside a gap are placed at sorted-uniform fractions of that gap's
mass (§3.1; `_axis_coords`). On the G-axis the $k$ negatives ranked between
consecutive positives $Y_{(j)} < Y_{(j+1)}$ receive fiducial G-values
$\tilde G(Y_{(j)}) + S_{j+1} V_{(r)}$, $r = 1,\dots,k$, with $S_{j+1}$ the
gap's spacing and $V_{(1)} < \dots < V_{(k)}$ sorted uniforms. Averaged over
the uniforms, this allocates the gap's positive mass to the $k{+}1$
sub-intervals cut by the negatives in equal expected shares — in proportion
to the negatives' own F-mass there. Since $R' = dG/dF$ is the likelihood
ratio, this is the fiducial rendering of "the likelihood ratio is constant
across the gap": **the true ROC is taken to be linear between consecutive
positives' operating points**, and symmetrically (F-axis) between
consecutive negatives'. The *randomness* of the placement is not the issue
— the exchangeable law of the fractions is the right one *given* a constant
LR; the issue is which mass is being spread, and over what.

**(b) Where it can matter: the end gaps at a convex corner.** In the
interior every gap holds $O(1/n)$ mass and the curve is smooth, so the
assumption costs second order (Prop. 3b(3)). The two *end gaps* differ in
kind:

- G-axis: below the lowest observed positive $Y_{(1)}$ — mass
  $S_1 \sim \mathrm{Beta}(1, n_1) \approx E/(n_1{+}1)$ — spread over the
  $k_{\mathrm{sat}} := \#\{i : X_i < Y_{(1)}\}$ negatives ranked below it,
  i.e. over the whole FPR range on which the empirical TPR equals 1;
- F-axis: above the top negative $X_{(1)}$ — mass $\approx E'/(n_0{+}1)$ —
  spread over the $p_1 := \#\{j : Y_j > X_{(1)}\}$ positives ranked above
  it, i.e. over $[0, t_1)$ at grid resolution.

In both regions the *local curve deficit being estimated* is itself
$O(1/n)$, so the end-gap allocation is $O(1)$ of the quantity of interest;
and a heavy-tailed truth is exactly there most nonlinear. Write
$\tau(s) := 1 - R(1-s)$ (positive mass below the bottom-$s$ negative
quantile) and $\rho(s) := \tau(s)/s$ (the average LR over that region); at
the origin $\rho_0(t) := R(t)/t$. $R$ is concave near $(1,1)$ iff $\rho$ is
nondecreasing. A positive class whose lower tail is as heavy as the
negatives' (LR $\to c > 0$ far out) gives $\rho$ *decreasing*: the ROC is
**convex** near $(1,1)$ — the classical hook of "improper" ROC curves — and
symmetrically $\rho_0$ increasing is a convex hook at the origin. For a
location family with common tail index $\nu$ (Student-$t$ against shifted
Student-$t$, shift $\delta$) the LR has its minimum at
$x_- = (\delta - \sqrt{\delta^2 + 4\nu})/2$ and its maximum at
$x_+ = \delta - x_-$: the ROC is convex for thresholds below $x_-$ (FPR
above $P(X > x_-) \approx 1/2$ for large $\delta$) and above $x_+$ (FPR
below $\approx (1 - \mathrm{AUC})/2$), concave between. For t(2)/.99 these
regions are FPR $<.005$ and FPR $>.53$, matching the probe's floor.

*Corner scales.* $Q := \ln(1/\ell)$, $\ell$ the band's local level ($Q
\approx 6$–$7$ at $\alpha = .05$; §9). Right end: $s_* := \tau^{-1}(1/n_1)$
(expected depth of the lowest positive), $k_* := n_0 s_*$ (expected width of
the saturated zone in negatives), $p_* := n_1 \tau(1/n_0)$ (expected
positives below the bottom negative), and the **hook ratio**
$h := \rho(s_*)/\rho(1/n_0) = 1/(k_* p_*)$. Left corner:
$h_0 := (Q/n_0)\big/R^{-1}\!\big(Q\,R(1/n_0)\big)$. Both equal 1 for a
linear corner; $h < 1$ and $h_0 > 1$ at convex corners.

**Lemma 13 (leading-order end-gap calibration and hook inflation).
[Sketch]** Approximations: (i)
$\mathrm{Beta}(1, n)\cdot(n{+}1)$ and Dirichlet partial sums are replaced by
Exp(1) and Gamma variables; (ii) counts are replaced by their expectations
where they are large ($p_1 \gg 1$ at the left; grid index $k \gg 1$ at the
right). Both are relative errors of order $1/n$ or $1/\sqrt{\text{count}}$.

*Left corner, grid point $t_1 = 1/n_0$.* Let $\Gamma_1 \sim$ Exp(1) be the
top negative's placement in units of $1/(n_0{+}1)$ and $p_1$ the positives
above it. A cloud draw puts the top negative at fiducial depth
$E/(n_0{+}1)$, $E \sim$ Exp(1) independent of the data, and the $p_1$
positives at sorted-uniform fractions of $[0, E/(n_0{+}1)]$. If $E > 1$ the
draw's TPR at $t_1$ is $\mathrm{Bin}(p_1, 1/E)/n_1 \approx p_1/(n_1 E)$;
draws with $E < 1$ have TPR $\ge p_1/n_1$. Hence the tube's lower edge is
$L(t_1) \approx p_1/(n_1 Q)$. The truth is
$R(1/n_0) \approx (p_1/n_1)\, R(1/n_0)/R(\Gamma_1/n_0)$. Therefore

$$r_L := P(\text{lower-edge miss at } t_1)
 \approx P\big(R(\Gamma_1/n_0) > Q\,R(1/n_0)\big)
 = \exp\!\big(-n_0\,R^{-1}(Q\,R(1/n_0))\big) = \ell^{\,1/h_0}.$$

If $R$ is linear on $[0, Q/n_0]$ then $h_0 = 1$ and $r_L = \ell$: the
end-gap spread transports the top negative's exact Beta$(1, n_0)$ pivot to
the TPR axis without loss. Convex there: $h_0 > 1$, $r_L > \ell$.
Concave: $r_L < \ell$. (If $Q\,R(1/n_0) \ge 1$ the channel is closed —
the curve is already too high at $t_1$ for the top negative's placement to
matter — which is why the left channel shuts off at high AUC.)

*Right end, the saturated zone.* Let $s_1 := 1 - W_{(n_1)}$ be the depth of
the lowest positive, $\Lambda := n_1 \tau(s_1) \sim$ Exp(1), and
$k_{\mathrm{sat}} \approx n_0 s_1$. The cloud gives the zone's negatives TPR
deficits $S V_{(1)} < \dots < S V_{(k_{\mathrm{sat}})}$ with
$S \approx E/(n_1{+}1)$. At the grid point $k$ negatives from the top, with
$k \gg 1$ so that the fiducial negatives' own F-jitter averages out, the
fiducial deficit is $\approx S\,k/k_{\mathrm{sat}}$ with upper-$\ell$
quantile $Q\,k/(k_{\mathrm{sat}}(n_1{+}1))$, while the truth's deficit is
$\tau(k/n_0)$, deterministic. The band misses there iff
$n_1 \tau(k/n_0)\,k_{\mathrm{sat}}/k > Q$, i.e.

$$\Lambda > Q\,h_k,\qquad h_k := \rho(s_1)/\rho(k/n_0).$$

Under linearity $h_k \equiv 1$ and **the whole zone misses together** —
one effective look, probability $e^{-Q} = \ell$, however wide the zone.
Under a convex hook $h_k < 1$, smallest at the outermost grid points, and
the probability is inflated. Replacing both the random $s_1$ by $s_*$ and
the order statistic $V_{(k)}$ by $k/k_{\mathrm{sat}}$ gives the
**large-$k$** zone criterion

$$k_{\mathrm{sat}} > k_{\mathrm{crit}} := Q/p_*
 \ \Longleftrightarrow\ \text{miss},\qquad
 r_R^{(\infty)} := P(k_{\mathrm{sat}} > k_{\mathrm{crit}})
 = \exp\!\big(-n_1\,\tau(Q/(n_0 p_*))\big),$$

again $\ell$ when $\tau$ is linear. This last replacement is poor when the
first few grid points determine the miss and is corrected next. $\square$

Two remarks. The same computation in an interior gap with $j$ positives
below gives a relative excess of order $1/j$ (the anchor
$\tilde G(Y_{(j)}) \sim \Gamma_j/n_1$ is at the right scale; only a fraction
of one spacing is misallocated), so the end gap dominates and the effect
decays into the interior like $1/j$ — the *count of positives below the
threshold* is the natural coordinate for its extent. And both channels
move the truth *below* the cloud while the upper edge additionally carries
the CP allowance: this is the directional asymmetry of the failures
(viol_low $\gg$ viol_high in every failing cell).

**Finite-grid correction. [Sketch]** Conditional on a saturated-zone size
$K$, retain the randomness discarded above. For the point $k$ steps from
the endpoint,

$$n_1(1-\widetilde R_k)\ \dot\sim\
 E\,{k\over K}Z_k,\qquad E\sim\operatorname{Exp}(1),\quad
 Z_k\sim\operatorname{Gamma}(k,\text{rate }k),$$

with $E$ and $Z_k$ independent. Let $q_{k,\ell}$ solve
$P(EZ_k>q_{k,\ell})=\ell$. Its survival function is available without
simulation,

$$P(EZ_k>q)=
 {2(kq)^{k/2}\over\Gamma(k)}K_k(2\sqrt{kq}),$$

where $K_k$ is the modified Bessel function. Thus $q_{1,\ell}>Q$ and
$q_{k,\ell}\downarrow Q$; the cloud is substantially more variable than
the mean-gap approximation at the first few grid points. Define

$$K_{\rm crit}:=
 \left\lceil\min_{1\le k\le n_0}
 \max\left\{k,\ {q_{k,\ell}k\over n_1\tau(k/n_0)}\right\}\right\rceil ,
\qquad
r_R:=\exp\{-n_1\tau(K_{\rm crit}/n_0)\},$$

with $r_R=0$ if $K_{\rm crit}\ge n_0$, and combine the ends as

$$\boxed{\quad r_{\rm corner}=1-(1-r_L)(1-r_R).\quad}$$

This is a *risk score*, not a literal whole-band miss probability: it still
Poissonizes the endpoint spacings, treats the F-axis grid jitter only to
first order, and omits interior gaps. Unlike $r_R^{(\infty)}$, however, it
keeps the grid-resolution effect responsible for the worst false alarms.

**(c) Consequences.**

1. *A fast screen and a slower magnitude predictor.* The analytic score
   uses only $R$ near the two endpoints and costs milliseconds. On the 257
   follow-up cells its correlation with measured miscoverage is .86, versus
   .53 for the large-$k$ formula; $r_{\rm corner}\le .05$ clears 122 cells
   with no observed coverage below .94. The Poissonized endpoint simulator
   in (g) is slower but predicts magnitude better (RMSE .025). These are
   in-sample diagnostics, not new validation.
2. *The t-family reduces to $(N_0, N_1, \nu)$.* With the power-tail
   approximation $P(T_\nu < -y) \approx c_\nu y^{-\nu}$,
   $1 - \mathrm{AUC} \approx 2 c_\nu \delta^{-\nu}$ and

   $$\rho(s) = \big(1 + (2s/(1{-}\mathrm{AUC}))^{1/\nu}\big)^{-\nu},\qquad
     \rho_0(t) = \big(1 - (2t/(1{-}\mathrm{AUC}))^{1/\nu}\big)^{-\nu}
     \ \ (2t < 1{-}\mathrm{AUC}),$$

   so the tail approximation reduces the problem to
   $N_i := n_i (1 - \mathrm{AUC})/2$, $\nu$, and $Q$ (plus the integer
   grid index in the finite-$k$ correction). At fixed $\nu$ the dangerous
   set is generally a **window in $N$**: it opens when the hook becomes
   resolvable and closes only when the sampled tail reaches the nearly
   linear far-tail regime. Its location and width depend strongly on
   $\nu$; there is no universal safety claim at $N_0<1$. Read along $n$ at
   fixed AUC and worst-cased over $\nu$, the maximizing tail index drifts
   upward with $n$, producing the observed wedge. The
   $m$-window of §7.3(b) is this coordinate: for heavy tails
   $t_{.5} = P(X > \delta) \approx (1 - \mathrm{AUC})/2$, so $m \approx N_0$.
3. *Imbalance.* Among the two sample sizes, $r_L$ depends only on $n_0$.
   $r_R$ generally worsens as
   $n_0/n_1$ grows: $h = 1/(n_1 s_* \rho(1/n_0))$, and more negatives at
   fixed positives probe deeper into the far tail where $\rho$ is larger,
   while the saturated zone widens in proportion. **Prediction:**
   negative-majority is the dangerous direction for the right-end channel
   at high AUC; positive-majority is protective (resolution-corrected
   worst case at AUC .975: $n_0 \times n_1 = 5000 \times 500$ gives .25,
   $500 \times 5000$ gives .09). Untested; Stage F's imbalance LHS will
   see it.
4. *Frontier reading.* The failures are
   the Lemma 9 frontiers being violated by way of the convention.
5. *Why the right end dominates, and how deep the misses are.* $h_0$ is
   bounded because $Q\,R(1/n_0) \ge 1$ closes the left channel at high AUC,
   so $r_L$ is 0 for most high-AUC cells and rarely exceeds .1–.2; the
   right score can be much larger. Left misses are *deep* —
   the truth at $t_1$ may be .01–.1 with the edge several times higher
   (measured max depth .10) — right misses are *shallow*, $\lesssim$ a
   few$/n_1$ (measured $10^{-4}$–$.02$).

**(d) Predictions for untested cells.** The following is
$r_{\rm corner}$ from the exact Student-t ROC, worst-cased over the
numerically attainable members of a fixed grid
$\nu\in[1.1,30]$; it is *prediction, not measurement*. The §9 law supplies
$\ell$. The empirically natural screening line is .05.

| AUC | $n=100$ | 500 | 2,000 | 8,000 | 50,000 |
|---|---:|---:|---:|---:|---:|
| .85 | .051 | .019 | .010 | .006 | .003 |
| .90 | .101 | .034 | .018 | .011 | .006 |
| .95 | .261 | .083 | .042 | .025 | .014 |
| .975 | .508 | .178 | .092 | .056 | .032 |
| .99 | .591 | .359 | .207 | .143 | .094 |

For approximate severity, the smallest useful empirical map is

$$P(\text{whole-band miss})\approx q_0+\lambda r_{\rm corner},
\qquad q_0\approx .019,\quad\lambda\approx .62,$$

where $q_0$ is ordinary non-corner miscoverage estimated from cells with
negligible score. An unconstrained weighted fit gives
$.003+.72r_{\rm corner}$; the spread between these maps is a more honest
uncertainty indication than extra digits. Both give roughly 2.5–2.8
percentage-point RMSE on the same 257 cells, with larger errors on the 11
original anchors.

Concrete prospective predictions: at AUC .985–.99 and balanced
$n=8{,}000$–12,000, $\nu=2$ should pass, $\nu\approx4.7$ is borderline,
$\nu\approx6$–8 should fail (predicted coverage about .89–.93 at AUC .99),
and $\nu=10$ returns toward the boundary. At AUC .975, changing
$n_0\times n_1$ from $500\times5000$ to $5000\times500$ raises worst-case
risk from about .09 to .25. A one-heavy-tail family should fail only at
the corresponding endpoint; a concave-corner family should not exhibit
this channel (the follow-up's Weibull, gamma and beta-opposing corner spot
checks, all .977–.992, are consistent with this). Stage F is the external
test of all three predictions.

**(e) The floor from first principles.** Where the convention is wrong,
the honest replacement is the *bracket* completion of Prop. 3b(1): use the
whole enclosing gap. On the G-axis, at a negative with $j$ positives below,
the bracket's lower completion is $1 - \tilde G(Y_{(j+1)})$, whose fiducial
law is Beta$(n_1 - j,\, j{+}1)$ — exactly the pivot of $B(Y_{(j+1)})$, and
$R$ at that negative is $\ge B(Y_{(j+1)})$ deterministically. This uses
the same Beta order-statistic ingredient as M3. M3 additionally makes the
two one-sample bands simultaneous and composes uncertainty from both axes;
it is therefore a certified implementation of the *bracketing principle*,
not literally the bracket cloud at level $\ell$.

The endpoint calculation gives a minimal, observable base region:

- **Left: $k\le\lceil Q\rceil$.** At $t_k=k/n_0$, the top-negative end gap
  can affect the lower $\ell$-quantile only on the event $E>k$, whose
  probability is $e^{-k}$. Once $k>Q$, that event has probability below
  $\ell$ and cannot determine that quantile at leading order. Thus the
  first $\lceil Q\rceil$ grid points are a conservative end-gap floor
  ($Q\approx6$–7 here); fewer points are an empirical economy.
- **Right: $j(t)=0$.** Here
  $j(t)=n_1\{1-\widehat R(t)\}$ is the observed number of positives below
  the threshold. The set $j=0$ is exactly the empirical-TPR-1 saturated
  run over which the final positive spacing is spread. Flooring this
  entire random run is the smallest endpoint-connected, data-measurable
  region that removes the dominant right end-gap mechanism. Extending to
  $j\le j_{\max}$ protects the first interior gaps, whose relative
  interpolation effect is $O(1/(j+1))$; theory fixes the unmargined base
  at $j_{\max}=0$. One margin is not optional: the cloud re-randomizes
  the negatives' F-positions, so the *fiducial* saturated zone ends
  $\pm\sqrt{K}$ grid points from the observed run's end and the end-gap
  spread leaks that far past $j=0$. Either $j_{\max}=1$ or a
  $2\sqrt{K}$-grid-point extension of the run absorbs it. The revised
  Stage F primary rule upper-bounds the realized $Q$ by
  $\log(M+1)$, predeclares the square-root extension, and leaves $j=0$ and
  $j\le1$ as mechanism/price comparators rather than fitted choices.

There is also a larger **curvature-complete** floor when a shape class is
declared. For the shifted common-scale Student-t family the likelihood
ratio turns at

$$x_\pm={\delta\pm\sqrt{\delta^2+4\nu}\over2},\qquad
t_L=P(T_\nu>x_+),\quad t_R=P(T_\nu>x_-),$$

and the ROC is convex exactly on $[0,t_L]\cup[t_R,1]$. Flooring that set
removes every convex within-gap segment, not only the two end gaps. Since
$x_-<0$, $t_R>1/2$ for every $\delta>0$; as $\delta\to\infty$,
$t_R\downarrow1/2$. Hence $[.5,1]$ is the smallest fixed right-hand
interval covering the hook uniformly over the whole shifted-t class—the
probe's cutoff has a first-principles justification. A narrower
shape-restricted rule uses the infimum of $t_R$ over a prespecified
$(\mathrm{AUC},\nu)$ uncertainty set.

Inside these regions the M3-versus-fiducial difference is only
$O(1/n_1)$ per right-tail grid point, explaining why a long right floor can
be cheap. Three choices remain, and should not be conflated:

1. $j_{\max}\ge0$, the sole fitted *extent* beyond the derived end-gap
   region (the left default is $\lceil Q\rceil$);
2. $\alpha_2$, where $\alpha_2=\alpha$ gives the sharpest regional M3 band
   but leaves no formal budget for exterior misses, while
   $\alpha_2=\alpha/2$ leaves half the union-bound budget; and
3. M3's class split $\rho$. Any split fixed independently of the observed
   ranks preserves Proposition 12. A first-order width calculation **[Heuristic; untested]** gives
   $\rho^*\approx A_F/(A_F+A_G)$, where

   $$A_F={1\over\sqrt{n_0}}\int_{\mathcal R}
       R'(t)\sqrt{t(1-t)}\,dt,\qquad
     A_G={1\over\sqrt{n_1}}\int_{\mathcal R}
       \sqrt{R(t)\{1-R(t)\}}\,dt.$$

   Rather than fit another surface, evaluate the exact M3 width over the
   one-dimensional $\rho$ grid for the design sizes and a prespecified
   reference or worst-case shape. Choosing $\rho$ from the same observed
   ranks would forfeit the simple exact-coverage proof unless selection is
   separately accounted for.

**(f) What an optimal router boundary can—and cannot—use.**

**Proposition 14 (AUC and sample sizes do not identify endpoint risk).
[Exact construction; coverage consequence measured — Corollary 14.1.]**
Fix $A\in(0,1)$ and $(n_0,n_1)$. The class of continuous increasing ROC
curves with integral $A$ contains curves with arbitrarily different
$\rho_0$ and $\rho$ near the endpoints, at negligible cost in area:
the perturbation that drives Lemma 13 is a *sliver* of positive mass
$\pi \asymp 1/n_1$ in the extreme lower tail together with an interval
of width $s_1 \gtrsim Q/(n_0 p_*)$ carrying no positive mass, whose joint
area cost is $O(\pi) + O(\pi s_1)$ and is compensated by an arbitrarily
small interior change. (The narrowness is in the sliver, not in the hook
as a whole: a hook confined to one grid cell is invisible to the band, and
a useful counterexample needs the wide empty stretch.) Therefore AUC and
the two sample sizes cannot bound either hook ratio, and no nontrivial
distribution-free router boundary in those three variables exists.
Uniform validity over all continuous ROCs routes everything to M3. Any
useful *router* must declare a shape/curvature class or use additional
tail information; the distribution-free alternative is not a router at
all but the conservative floor of (h).

**Corollary 14.1 (the sliver DGP; the depth–probability frontier of the
C = 1 band). [Sketch + Empirical]** Take $\tau(s) = c\,s$ on
$[0, 1/n_0]$, $\tau(s) = c/n_0$ on $[1/n_0, s_1]$, and any concave body
on $[0, 1-s_1]$ scaled to the target AUC. The sliver is unsampled with
probability $(1 - c/n_0)^{n_1} \approx e^{-c\,n_1/n_0}$; on that event
the empirical-TPR-1 run has $K \approx n_0 s_1$ grid points, the truth's
deficit there is $\pi = c/n_0$, and by Lemma 13 the band misses at every
grid point $k$ with $K > q_{k,\ell}\,k/(n_1 \pi)$ — for any $s_1 > 0$ once
$n_1 \pi \gtrsim q\,k\,/(n_0 s_1)$. Writing the sliver mass in frontier
units, $\pi = d/n_1$: **for every AUC and every $(n_0, n_1)$ with
$\min(n_0,n_1) \gtrsim 100$ there is a continuous DGP on which the C = 1
band misses by depth $\approx d/n_1$ with probability $\ge e^{-d}(1-o(1))$.**
This is the Lemma 9.1-mirror frontier read as a coverage statement: at
depth $1/n_1$ the forced miss probability is $\ge .37$, at $3/n_1$ it is
$\ge .05$ — so the band is not valid at level .95 against misses of the
frontier scale at any sample size — while M3's deficit in the same run is
$\approx \ln(1/\gamma)/n_1 \ge 6/n_1$ and it cannot be forced there.
Measured (production band, $n_0 = n_1 = 500$, 100 replicates each,
`corner_mechanism.py sliver`): C = 1 coverage **.54** at AUC .80 and
**.56** at AUC .95 (both $c = .8$, $s_1 = .25$; predicted saturation
$e^{-.8} = .45$) and **.64** at AUC .60 ($c = 1$, $s_1 = .12$; predicted
$e^{-1} = .37$), against M3 coverage 1.000 on the same data — the
saturation probability accounts for the whole deficit. The construction
scales: with $n_1\pi$ held fixed the
miss probability is $n$-independent, so the C = 1 band's distribution-free
worst case does not improve with $n$.

For a declared class $\mathcal C$ and an AUC uncertainty set $\mathcal A$,
the hard-coverage router furnished by the approximation is

$$r^*(n_0,n_1,\mathcal A,\mathcal C)
 =\sup_{A\in\mathcal A,\ R\in\mathcal C(A)}r_{\rm corner}(R),$$
$$\boxed{\quad\text{use fiducial only if}\quad
q_0+\lambda r^*\le\alpha-\eta,\quad}$$

where $\eta\ge0$ is a validation margin. At $\alpha=.05$, the current
$q_0,\lambda$ estimates make $r^*\lesssim.05$ the natural zero-margin
boundary. For the shifted-t class the only structural input beyond
$(A,n_0,n_1)$ is the allowed $\nu$ range; no fitted AUC-by-$n$ surface is
needed. Because the risk is nonmonotone in both AUC and $n$, a one-sided
AUC upper bound is not safely plugged into this formula: maximize over the
entire confidence set.

If “optimal” means minimizing width plus a miss penalty rather than
enforcing a hard bar, let $\Delta W$ be M3's extra expected width and
$q_F,q_M$ the predicted miss probabilities. The pointwise Bayes/minimax
decision is M3 exactly when

$$\kappa(q_F-q_M)>\Delta W,$$

for the declared penalty $\kappa$ (using suprema over the class for a
minimax rule). This makes explicit why no unique optimal boundary exists
without a loss or coverage constraint. The localized floor usually has a
better measured width tradeoff than global routing, but it provides only
the regional cap of §7.3(d); it is not a full-band theorem unless the
exterior term is also controlled.

**(g) Evidence, briefly. [Empirical]** On the 257 follow-up cells, the
finite-grid analytic score correlates .86 with measured miscoverage and
clears 122 cells at the .05 cutoff with no observed sub-.94 coverage. The
Poissonized endpoint simulator correlates .90 with coverage (RMSE .025);
its $\le.01$ screen clears 103 cells with no observed failure. A 40-rep
production-band spot check at t(6.62)/.988, $n=6656$, directly reproduced
the predicted first-point, mid-zone, and whole-saturated-zone miss modes.

**(h) The distribution-free case: what a heuristic may rely on.** Prop. 14
closes the door on routers that read $(\widehat{\mathrm{AUC}}, n_0, n_1)$
without a declared class. It does not close the door on adaptivity. Two
kinds survive, and they are different objects.

*Declared class: the natural declaration is curvature, not a tail index.*
**Corollary 13.1 (corner concavity suffices). [Sketch]** If $R$ is concave
on $[0, Q/n_0]$ and on $[1-s_*, 1]$ — in particular if the likelihood
ratio is monotone, the classical "proper ROC" assumption — then $h_0 \le 1$
and $h_k \ge 1$, so both corner channels of Lemma 13 are *conservative*:
$r_L, r_R \le \ell$. Together with the interior roughness surplus (§7.1)
this is the leading-order **candidate** for the class-relative validity
statement that has been missing: the corner mechanism is conservative on
the proper-ROC class and anti-conservative off it, and every wedge cell is
off it. The declaration "corners concave" is weaker than a parametric tail
class and is what binormal, bimodal-negative and the tested Weibull/gamma
corners satisfy; the shifted-$t$ family violates it at both ends. Within
the declaration the corner mechanism predicts that no router is needed,
but full-band finite-sample safety still requires proof or prospective
validation. A practical router would therefore be a *class declaration or
class test*, and Prop. 14 says such a test cannot be run from three summary
numbers.

*No class: the conservative floor, and the properties that keep its rule
library-independent.* Absent a declaration the only admissible adaptivity
is widening, and the following properties give the floor exact domination
and a distribution-free regional cap. They do not, without control of the
exterior term, prove full-band $1-\alpha$ coverage:

1. **Widening-only.** The rule may enlarge the C = 1 band and never narrow
   it. Then coverage $\ge$ C = 1 coverage for every DGP and every
   replicate (§7.3(d), exact). Anything that narrows re-enters the
   calibrated-map territory of §7 and inherits the plug-in pathology.
2. **Rank-measurable.** The region is a function of the merged label
   sequence $\Lambda$ alone (Prop. 2). Its law under any DGP is then the
   law of the rank experiment; nothing about the score scale, the family,
   or an estimated AUC enters.
3. **An exact cap inside.** The widening component carries its own
   distribution-free guarantee (M3, Prop. 12), so $P(\text{miss inside the
   region}) \le \alpha_2$ whatever the region's selection rule (the
   sub-event argument of the Stage F spec §1.3).
4. **Frontier honesty as the trigger — not curvature estimation.** The
   conservative design principle is to contain every grid point at which
   the C = 1 edge claims more than the Lemma 9 frontier permits of *any*
   distribution-free band. Both identified places are rank-observable:
   the first $\lceil Q \rceil$ grid points (the band's
   $L(t_k) > 0$ there is a frontier violation whenever $p_1 > 0$), and the
   empirical-TPR-1 run of length $K$ (the band's deficit $\approx Q k/(K n_1)$
   is below the $c/n_1$ frontier for $k < cK/Q$, i.e. over most of the run
   whenever $K > Q/c$). This is exactly the region Lemma 13 produces, but
   derived without any statement about the truth: the rule compares the
   band's own claim against the minimax frontier, and Prop. 2 makes that
   comparison a function of the ranks. Curvature signatures (the growing
   inter-positive gaps and $p_2 - p_1 \gg p_1$ of (e)) may *narrow* the
   region as conservative-only economies; they must not be needed to
   *find* it.
5. **An a-priori width scale.** The *unmargined base* frontier region costs
   at most
   $\lceil Q\rceil/n_0$ (left; heights $\le 1$) plus $K \cdot Q/(n_0 n_1)
   \le Q/n_1$ (right; both deficits are $\le Q/n_1$ on the run) in area:
   $\Delta\text{area} \le Q\,(1/n_0 + 1/n_1)$, with the typical cost far
   below the bound (the probe's +5–7%; the right run is cheap because both
   bands are within $O(Q/n_1)$ of 1 there). The required
   $2\sqrt K$ margin adds at most its grid fraction in the vacuous
   height-one bound; its sharper price remains to be derived and is
   measured separately in Stage F. No fitted price surface is required.
6. **No fitted $(\mathrm{AUC}, n)$ surface in the distribution-free
   floor.** Any such surface encodes the library's shape class (Prop. 14),
   and a rule that depends on $\widehat{\mathrm{AUC}}$ through it is
   class-relative however it is labeled. AUC may enter only as a
   declared-class input — a supremum over the class, per (f) — never as an
   estimated trigger; and because risk is non-monotone in AUC, not merely
   through a one-sided bound.
7. **The residual is the ordinary interior claim.** With 1–4 in place,
   $P(\text{miss}) \le \alpha_2 + P(E_{\mathrm{out}})$ and the corner part
   of $E_{\mathrm{out}}$ is $\ell$-level at leading order (the $j \ge 1$
   gaps are calibrated: their anchors $\Gamma_j$ carry the exact law and one
   misallocated spacing cannot reach the $\ell$-quantile of $\Gamma_j + SV$
   for $j \ge 1$), so what remains empirical is the same interior surplus
   that Theorem 7 and §7.1 already describe. Making that residual a theorem
   is open problem 6, not a routing question.

The two objects should be named separately in guidance: **the floor is the
rank-only, library-independent heuristic with an exact regional cap; the
router is the declared-class heuristic.**
A router that reads $(\widehat{\mathrm{AUC}}, n_0, n_1)$ and nothing else
is not a weaker distribution-free rule — by Prop. 14 it is a class
assumption written in three numbers.

**(i) The defect is pointwise, not a simultaneity artifact — and why the
pointwise literature did not meet it. [Exact mechanism + Empirical]**
Everything in Lemma 13 is a statement about the cloud's *marginal* law at
one grid point: the misallocated end-gap mass changes the fiducial
distribution of $\tilde R(t)$ at a fixed $t$ inside the run. The band only
changes the level at which that marginal is probed ($\ell$ in place of
$\alpha/2$), so the pointwise 95% credible interval — the cloud's .025 and
.975 quantiles, no trim — inherits the same criterion with
$\ln(1/(\alpha/2)) \approx 3.7$ replacing $Q$. Measured
(`corner_mechanism.py pointwise`; $n_0 = n_1 = 500$, $M = 3000$, 200
replicates): on the sliver DGP of Cor. 14.1 (AUC .80) the pointwise
interval at fixed FPR $\ge .98$ covers **.59–.61** (lower-edge misses
.395, i.e. the sliver's unsampled probability); on t(2)/.99 it covers
**.78–.83** at FPR $\ge .98$ and .87–.91 at FPR .80–.95, recovering to
.94–.96 only in the interior; on the concave-corner reference t(30)/.95 it
covers .96–.995 at every point with *zero* lower-edge misses at the
corner — the conservative direction Corollary 13.1 predicts. The
Bayesian-bootstrap cloud of Gu–Ghosal–Roy has the defect in its extreme
form: it places no mass beyond the lowest observed positive (§3), so on
the empirical-TPR-1 run its pointwise interval is the degenerate point
$\{1\}$ and misses whenever the truth is below 1 there — with probability
one, at a depth equal to the truth's deficit. **Verified with the published
code** (`ROCnReg` 1.0.9, `pooledROC.BB`, $B = 2000$ Bayesian-bootstrap
draws, 200 replicates, $n_0 = n_1 = 500$;
`scripts/c_calibration/rocnreg_bb_check/`): its 95% pointwise band covers
**.46, .59, .63, .72** at FPR .998, .994, .990, .980 on t(2)/.99, still
only .79–.87 at FPR .80–.95 and .91 at FPR .50; **.50** at every FPR
$\ge .80$ on the sliver DGP (the sliver's unsampled probability, predicted
$e^{-.8} = .45$); and — the sharpest confirmation of the mechanism —
**.00, .01, .02, .04** at FPR .998–.980 on the *concave-corner* t(30)/.95,
where the truth's deficit is $3\times10^{-6}$–$4\times10^{-5}$: in every
case the lower-edge miss rate equals the probability that the interval
collapses to a point (width 0), i.e. the BB's corner defect is universal
for continuous scores and merely invisible in depth when the tails are
light. Against the same data the spacings-GFD pointwise interval covers
.78–.83 (t(2)/.99), .59–.61 (sliver) and .97–.995 (t(30)/.95) at the same
points: the spacings-GFD repairs the *anchor* (the lowest positive's mass
has the exact Beta$(1,n_1)$ law) and then spends it along the run by the
linear convention; the bracket, or M3, declines to spend it at all.

Three things kept this out of view. (1) *Where coverage was checked.*
Pointwise studies evaluate at fixed interior operating points (FPR .1,
.3, .5) where the count of positives below the threshold is large and the
misallocation is $O(1/j)$ of an $O(1/n_1)$ quantity; the run sits at the
far corner and its location is data-dependent, so a fixed interior $t$
almost never lands in it. A simultaneous band evaluated on the native grid
integrates over the run in every replicate, which is why the defect
surfaced here as a whole-band failure rather than as a pointwise one.
(2) *Which shapes were simulated.* Binormal-type truths have concave
corners; there the convention is conservative (Lemma 13) and the BB's
degenerate corner interval misses by $\sim 10^{-6}$, an invisible
technical miss. Heavy tails at high AUC are what make the hook resolvable.
(3) *What the one-sample theory chose to do at the gaps.* Cui & Hannig's
GFD is exact at the order statistics and carries the between-observation
uncertainty as an *interval* $[F^L, F^U]$ (their conservative option);
their band theory (Thm. 3.2) is on compacts. The ROC composition forces a
*selection* inside that interval because each draw must be a curve, and
the negatives' grid then makes the selection visible at every point of an
end gap. The corner problem is therefore the price of the selection, not
of the fiducial argument or of simultaneity — which is also why its cure
(bracketing, in exact form M3) comes from the same one-sample literature.
One consequence is checkable and was checked: the C = 1 band is *exactly*
GET's rank envelope of the fiducial cloud (Lemma 5, code parity), so any
global-envelope user who fed GET a fiducial or Bayesian-bootstrap ROC
cloud would inherit precisely this corner behavior — the trim has no way
to see, let alone repair, a defect in the cloud's marginals.


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
with constants — open problem 5. The §7.3 wedge is these
frontiers being *violated*: through the within-gap convention the fiducial
lower edge is nonvacuous on the first grid points ($L(t_1) \approx
p_1/(n_1 \ln(1/\ell))$ whenever any positive outranks the top negative) and
tighter than $c/n_1$ in the empirical-TPR-1 zone whenever that zone is
wider than $\sim\ln(1/\ell)$ negatives. Both claims exceed what any
distribution-free band may make, and a convex corner is the truth that
calls the bluff. The bracket completion (Prop. 3b(1)), or a certified M3
floor based on the same bracketing principle, restores both frontiers.

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
Empirically, $j^\*$ ran 11–22 across the Stage S and follow-up cells.

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
  *not* at the corner — on the binormal and bimodal cells, i.e. the
  *concave*-corner case. With a convex (heavy-tail) corner, AUC is the
  second coordinate of the wedge: the §7.4 mechanism is a function of
  $n_i(1-\mathrm{AUC})$ and the tail index, and at AUC $\ge .96$ no tested
  $n \le 6{,}656$ is safe in the t-family. Width is AUC-sensitive
  everywhere (§10); coverage is AUC-sensitive exactly where the corner is
  convex.
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
   cells and .998–1.000 on the probed wedge cells; area 1.45–2.19× the production fiducial band (0.37–0.88× KS —
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
   concrete, attackable version of this problem. **Update 2026-09-02:**
   the finite-sample obstruction is identified (§7.4) — the within-gap
   convention, not the trim or the pivots. The natural theorem target is
   therefore the band with the *bracket completion* (§3.1) on the tails,
   or the M3-floored band implementing the same no-interpolation principle
   with simultaneous Beta bands; the interior claim stays
   empirical/asymptotic (Theorem 7).
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
6. **Turn Lemma 13 into bounds (new, 2026-09-02).** The corner miss rates
   of §7.4 are leading-order: Beta$(1,n)\cdot(n{+}1) \approx$ Exp(1), the
   large-$p_1$ approximation at the left, the deep-zone ($k \gg 1$)
   approximation at the right, the pointwise level standing in for the
   global trim, and the first interior gaps ($j \ge 1$, relative effect
   $\sim 1/j$) omitted. The finite-$k$ product correction in §7.4 removes
   the most consequential right-end approximation; the remainder is a
   finite computation with the exact Beta
   and Dirichlet laws. A clean statement would be: for any $R$ convex on
   $[1-s_*, 1]$, the C=1 band's lower-edge miss probability is bounded in
   terms of $r_R(R,n_0,n_1,\ell)$; and for $R$ concave on both corners, at most
   $\ell$ per corner. The second half is the finite-sample validity
   statement that has been missing for concave truths.

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
| Within-gap convention | Bounded by one gap mass (Prop. 3b); first-order irrelevant on the interior (local Hölder exponent $>1/2$); calibrated to local ROC linearity and materially anti-conservative at convex heavy-tail corners; repaired by bracketing or an M3 floor (§7.4) | [Exact bound; Sketch rate; Empirical failure] |
| Curve-valued draws (not pointwise intervals) | Monotone, $[0,1]$-respecting, correctly correlated bands; both HT variance channels carried without density estimation | [Sketch] |
| Min-p / equal-local-levels trim | Simultaneity without a variance estimate (= the correctly-studentized region in the Gaussian limit, Narisetty–Nair Cor. 1); balanced miss directions; spread miss locations; graze-type misses | [Exact structure; balance asymptotic] |
| Trim level from the cloud's own quantile | Finite-$M$ content control for *any* trim score (Lemma 6b); conservative failure mode under saturation; self-diagnosing budget ($j^\*$) | [Exact] |
| $C$-remap of the trim level | Finite-sample centring at central $\alpha$; asymptotically the trim level is the coverage (Thm 7), so fixed $C>1$ is a finite-sample device with a known asymptotic liability; erosion law §7.1 | [Empirical + Sketch] |
| CP-form upper allowance at level $j^\*/(M+1)$ | Matches the plateau scale forced by Lemma 9.1; pure widening (Lemma 8); parameter-free. Not a standalone exact device — the level is data-selected | [Sketch necessity; Exact widening] |
| Degenerate lower allowance | The forced lower-left mirror, at zero width cost; full mirror provably not worth it | [Exact + Empirical] |
| Rank-only frontier M3 floor | Restores the left and saturated-run honesty frontiers with a predeclared margin; dominates C = 1 and has an exact M3 cap inside the random region | [Exact containment/cap; Sketch extent; external test pending] |
| Random tie-breaking | Exact reduction under ties with trapezoidal estimand; no conservatism needed | [Exact] |

**The method space in one taxonomy.** The GET package's envelope types map
this project's whole history onto one axis (what is ordered) times one
choice (what cloud): `'unscaled'` (constant-width MAD) = the KS-type band;
`'st'` (studentized MAD) = the old envelope method's interior statistic;
`'rank'` (extreme rank / min-p) = this method's trim; `'erl'` = the §5.1
refinement. The envelope era applied a `'st'`-type ordering to a *resampling
bootstrap* cloud and needed two exact floors to patch that cloud's support
collapse; the current method applies a `'rank'` ordering to the *fiducial*
cloud. Its own-class spacings and mass beyond the extremes are exact, but
§7.4 shows that transporting the other class through an end gap is
anti-conservative at convex corners. The cloud was the load-bearing
improvement; the frontier M3 floor repairs the remaining within-gap
convention.

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
  Verified 2026-09-03: GET 1.0.9's rank central region of our fiducial
  cloud coincides with the production C = 1 tube to the last digit
  (Lemma 5 note).
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
  frequentist calibration of a band, no corner analysis. Measured
  2026-09-03 with `ROCnReg` 1.0.9 (§7.4(i)): the pointwise BB band's
  interval collapses to the point $\{1\}$ on the empirical-TPR-1 run and
  its coverage at the corner grid points is .46–.72 on t(2)/.99, .50 on
  the sliver DGP, and $\approx 0$ (at microscopic depth) even on a
  near-binormal truth. Follow-ups: Gu &
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
   only if the second-order analysis (open problem 1) is completed. The
   2026-09-02 corner analysis (§7.4) adds a finite-sample second-order
   statement of a different kind — the end-gap calibration lemma and its
   hook inflation — that we have not found in either literature, both of
   which take the cloud's between-observation completion as given.
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
