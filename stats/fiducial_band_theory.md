# Rank-space fiducial ROC bands: construction, guarantees, and next theory

*Working theory, 2026-09-05. Companion to the
[method assessment](next_method_ideas.md), the
[Stage F measurements](hybrid_floor_report.md), and the
[Python implementation](../src/studroc_paper/methods/fiducial_band.py).
This document describes the current construction and separates what we can
prove from what still needs calibration. It does not change the implementation.*

The method has a sound first-order rationale: independent Dirichlet
spacings reproduce both classes' empirical-process uncertainty, and the
rank trim becomes a studentized Gaussian simultaneous band on a regular
interior interval. Its unresolved issue is finite-sample honesty over
arbitrary ROC shapes. Exact marginal spacings do not make an arbitrary
completion of unobserved gaps exact.

The most useful additions developed here are:

- a distribution-free integrated-width bound for full gap bracketing (§3.1);
- exact empirical-ROC moment identities, and a sharp asymptotic bound on
  directional imbalance (§§4, 6);
- precise coverage-transfer and hybrid inequalities (§8);
- missing-mass lower bounds showing what an honest band must leave open (§9);
- exact, interpretable alternatives to the hybrid's empirical cut-point
  margins, including an optimality statement with a specified scope (§10).

**Status.** “Exact” means a finite-sample or deterministic statement under
its explicit assumptions. “Asymptotic” means a limit under the stated
regularity conditions; proof outlines are identified. “Heuristic” means a
mechanism or design proposal, not a coverage guarantee. “Empirical” means a
measurement on the cited experiments. “Honest” means coverage at least
$1-\alpha$ throughout a stated distribution class; “calibrated” additionally
means coverage is close to that level.

## 1. Target, conventions, and error events

There are independent samples
$X_1,\ldots,X_{n_0}\sim F$ (negatives) and
$Y_1,\ldots,Y_{n_1}\sim G$ (positives), iid within each class. Larger scores
indicate positives. The scoring rule is fixed independently of these
evaluation observations. Unless stated otherwise, $F,G$ are continuous;
they need not have densities, finite moments, or common support.

Define
$$
 Q_F(u)=\inf\{x:F(x)\ge u\},\qquad
 R(t)=1-G(Q_F(1-t)).
$$
Use $Q_F(0)=-\infty$; at $u=1$ retain the actual upper support endpoint,
possibly $+\infty$. Thus $R(1)=1$, but $R(0)$ can be positive. Even
continuous score distributions can produce jumps in $R$ when $F$ has
support gaps. A full-curve theorem cannot silently impose continuity of
$R$, or force the upper band to zero at FPR zero.

For a band $B=[L,U]$, write
$$
 \begin{aligned}
 V^-(B;A)&=\{\exists t\in A:R(t)<L(t)\},\\
 V^+(B;A)&=\{\exists t\in A:R(t)>U(t)\},\\
 V(B;A)&=V^-(B;A)\cup V^+(B;A).
 \end{aligned}
$$
The minus sign means **truth below the lower edge**, not a downward
estimation error. Omit $A$ for $[0,1]$. Coverage is
$1-q_R(B)$, where $q_R(B)=P_R(V(B))$ includes the procedure's independent
Monte Carlo randomization. Both directional events can occur in one
replicate.

The width criterion is
$$
 W(B)=\int_0^1\{U(t)-L(t)\}\,dt.
$$
Pointwise and maximum width remain useful, but are different objectives.

**Grid-to-continuum coverage. [Exact]** On an increasing grid
$0=t_0<\cdots<t_K=1$, extend the lower edge from the grid point to the left
and the upper edge from the point to the right. For
$t\in(t_k,t_{k+1})$ this gives
$L(t)=L(t_k)$ and $U(t)=U(t_{k+1})$. Coverage at every grid point then
implies coverage everywhere, including for a discontinuous $R$.
Linear interpolation of the edges does not have this implication.
Reported numerical integrals must specify which extension they integrate.

## 2. What rank invariance does—and does not—give

**Proposition 1 (the rank experiment). [Exact]** Put
$U_i=1-F(X_i)$ and $W_j=1-F(Y_j)$. Then
$$
 U_i\sim\operatorname{Unif}(0,1),\qquad P(W_j\le t)=R(t).
$$
Consequently the merged class labels in descending score order have the
same law as labels obtained by merging ascending draws from
$\operatorname{Unif}(0,1)$ and $R$.

*Proof.* Apply the continuous probability integral transform to $X_i$.
For $Y_j$, the event $F(Y_j)\ge1-t$ starts at $Q_F(1-t)$, giving the
displayed identity. A flat interval of $F$ can collapse several positives
to one placement value, but almost surely contains no negative observation;
the collapsed positive block does not alter the class-label ordering.
The endpoint conventions in §1 give the same identity at zero and one. $\square$

**Proposition 2 (invariance, not a coverage pivot). [Exact]** For untied
samples considered modulo permutations within each class, the merged
label sequence $\Lambda$ is a maximal invariant under strictly increasing
score transformations. A rank-only procedure therefore has coverage
$\mathcal C(R,n_0,n_1)$ independent of which score representation generates
$R$. It need not have the same coverage for two different curves.

This reduction removes irrelevant score geometry. It does **not** remove
the unknown ROC shape. “Works for heavy tails” must mean works for the ROC
shapes those tails induce, not that score-scale invariance proves honesty.
In particular, equal AUC does not imply equal sampling or coverage laws.

**Proposition 2a (ties, with an explicit estimand). [Exact reduction]** Attach
independent continuous uniform auxiliary variables to observations and
order lexicographically by score and auxiliary variable. Thresholding this
augmented score randomizes within every original score atom. Its ROC has
the linear tie segments of randomized threshold rules, and its AUC is
$$
 P(Y>X)+\tfrac12P(Y=X).
$$
This produces a continuous ordering experiment to which the rank
construction and exact continuous-score arguments apply. Its validity
claims concern this randomized ROC.

The production random tie option implements this idea. Deterministic
“even” tie placement has no theorem here. Breaking ties by class changes
the estimand; it is not an interchangeable computational shortcut.

## 3. The cloud, specified without hidden distributional assumptions

Sort the merged sample in descending score order. For one auxiliary draw:

1. Independently generate
   $S^0\sim\operatorname{Dirichlet}(1^{n_0+1})$ and
   $S^1\sim\operatorname{Dirichlet}(1^{n_1+1})$.
2. Let $u_i=\sum_{r=1}^iS^0_r$ and
   $v_j=\sum_{r=1}^jS^1_r$, with sentinels
   $u_0=v_0=0$, $u_{n_0+1}=v_{n_1+1}=1$.
   The $i$th negative owns FPR coordinate $u_i$; the $j$th positive owns
   TPR coordinate $v_j$.
3. On each axis separately, locate other-class observations in the gaps
   determined by their merged ranks. If $r$ such observations occupy a
   gap, give them sorted iid uniform fractions of its probability mass.
4. Join all resulting ordered coordinate pairs, together with $(0,0)$
   and $(1,1)$, by straight lines. Evaluate this curve $\widetilde R$
   on the requested grid.

Repeat independently $M$ times conditional on the observed labels.
The two axis completions are independent. They are not draws from a
uniquely identified joint conditional law of the two population CDFs.

**Raw endpoint limitation. [Exact]** Every polyline has
$\widetilde R(0)=0$, so its raw pointwise upper envelope also has value
zero there, for any $M$. It cannot cover a truth with $R(0)>0$.
Corner allowances or unrestricted endpoint brackets are therefore
essential for the full distribution class, not cosmetic corrections.

**Proposition 3 (what the Dirichlet law establishes). [Exact]** For a
continuous CDF $H$, its values at its own ordered sample have the joint law
of uniform order statistics:
$$
 (H(Z_{(1)}),\ldots,H(Z_{(n)}))
 \overset d=(A_1,\ldots,A_n),\quad
 A_i=\sum_{r=1}^i S_r,\quad S\sim\operatorname{Dirichlet}(1^{n+1}).
$$
Writing $m=n+1$, for $i\le j$,
$$
 A_i\sim\operatorname{Beta}(i,m-i),\quad
 EA_i=i/m,\quad
 \operatorname{Cov}(A_i,A_j)=\frac{i(m-j)}{m^2(m+1)}.
$$
The two classes' true spacing vectors are independent **unconditionally**.

These are exact sampling pivots. They are not the conditional distribution
of the unknown truth given the merged labels. Observing cross-class ranks
constrains the true spacings in an $R$-dependent way. A fresh independent
Dirichlet pair is a fiducial device, not an exact conditional resample of
that unknown pair.

Nor does this proposition determine the CDF between consecutive
observations, especially in either end gap. Uniform completion is an
additional choice. Its exchangeability allocates equal *expected* shares
to rank subintervals; it can behave like local linearity in probability
coordinates, but it does not impose constant likelihood ratio in each
realized draw.

**Proposition 3a (spacing size and endpoint limits). [Exact and asymptotic]**
For $D_n=\max_{1\le r\le n+1}S_r$,
$$
 ED_n=\frac{H_{n+1}}{n+1},\qquad
 P(D_n>d)\le(n+1)(1-d)^n,\quad 0\le d\le1,
$$
where $H_m=\sum_{r=1}^m1/r$. The exact CDF is
$$
 P(D_n\le d)=\sum_{j=0}^{n+1}(-1)^j
       {n+1\choose j}(1-jd)_+^n.
$$
The expectation follows by integrating this inclusion–exclusion formula;
the probability bound is a union bound over Beta$(1,n)$ spacings.

Writing $S_r=E_r/\sum E_s$ with iid Exp$(1)$ variables gives
$$
 D_n=O_p(\log n/n),\qquad
 (n+1)D_n-\log(n+1)\ \Rightarrow\ \text{Gumbel},
 \qquad nS_{\rm end}\Rightarrow\operatorname{Exp}(1).
$$
The Gumbel CDF is $\exp(-e^{-x})$. The sum normalization changes the
centered maximum by $o_p(1)$. Any fixed collection of distinct rescaled
spacings becomes independent exponentials. At finite $n$, the two end
spacings are not independent:
$$
 P(S_{\rm first}>x,S_{\rm last}>y)=(1-x-y)_+^n.
$$

For $M$ auxiliary draws, a simultaneous bound within one class is
$$
 P_*\!\left\{\max_{b,r}S_{b,r}>
  1-\left(\frac{\eta}{M(n+1)}\right)^{1/n}\right\}\le\eta.
 \tag{3.1}
$$
Here and below $P_*$ denotes probability conditional on the data.

### 3.1 Full gap bracketing: remove the completion choice at small area cost

Fix a spacing pair. Let $p_i$ be the number of positives preceding the
$i$th negative in descending score order, and set
$p_0=0$, $p_{n_0+1}=n_1$. For $t\in(u_i,u_{i+1})$ define
$$
 r^-(t)=v_{p_i},\qquad r^+(t)=v_{p_{i+1}+1}.                 \tag{3.2}
$$
These are the extremal bounds allowed by monotonicity and the two sets of
own-class anchors. At an anchor use the corresponding one-sided limits
to bracket both conventions; at $t=1$ set both edges to one.

Every monotone completion consistent with these anchors, including the
production polyline, lies in this bracket. The brackets allow jumps;
they need not be realizable by a single continuous completion attaining
both edges everywhere. This is an outer set, which is what coverage needs.

**Proposition 3b (universal area bound). [Exact; derived here]** Let
$d_0=\max S^0_i$ and $d_1=\max S^1_j$. Then, for every merged label path,
$$
 \boxed{\quad
 \int_0^1(r^+-r^-)\,dt\le d_0+d_1.
 \quad}                                                    \tag{3.3}
$$
No regularity of $R$, density ratio, or score tails is required.

*Proof.* The area equals
$$
 \sum_{i=0}^{n_0}(u_{i+1}-u_i)
       (v_{p_{i+1}+1}-v_{p_i}).
$$
View the merged labels as a lattice path through the probability
rectangles of widths $S^0_i$ and heights $S^1_j$. The displayed sum is
exactly the area of the visited rectangles. Charge the starting rectangle
and every rectangle entered by a negative step to its width: their
total is at most $d_1\sum_iS^0_i=d_1$. Charge each rectangle entered by a
positive step to its height: their total is at most
$d_0\sum_jS^1_j=d_0$. These charges exhaust the path. $\square$

For uniform spacings with $n_0=n_1=n$, the ratio of the two sides is
$(2n+1)/(2n+2)$, so the constant cannot be uniformly improved.
Conditional on any observed labels,
$$
 E_*\int(r^+-r^-)
 \le \frac{H_{n_0+1}}{n_0+1}+\frac{H_{n_1+1}}{n_1+1}.       \tag{3.4}
$$
A bracket can have order-one height near a jump and still have
$O(\log n/n)$ expected area. Supremum width and integrated width tell
different stories here.

**Corollary 3c (cost of bracketing an existing cloud envelope). [Exact]**
Retain the **same draw indices** as a curve-based envelope, and replace
each retained curve by its bracket. Let $d_c$ now be the largest class-$c$
spacing among those draws. If $B_{\rm curve}$ and $B_{\rm bracket}$ denote
the respective pointwise hulls, then
$$
 B_{\rm curve}\subseteq B_{\rm bracket},\qquad
 0\le W(B_{\rm bracket})-W(B_{\rm curve})\le2(d_0+d_1).        \tag{3.5}
$$
The same statement holds for a fixed pointwise order-statistic tube:
use the same order index for curves and bracket edges.

*Proof.* Equation (3.2) gives
$r_b^+(t)\le r_b^-(t+d_0)+d_1$, extending monotone curves by zero below
zero and one above one. Hence
$$
 r_b^+(t)\le c_b(t+d_0)+d_1,\qquad
 r_b^-(t)\ge c_b(t-d_0)-d_1.
$$
Taking maxima, minima, or the same order statistic preserves the
inequalities. The integral of a bounded monotone function changes by at
most $d_0$ under this shift. Each edge therefore costs at most $d_0+d_1$.
$\square$

Together with (3.1), this yields a high-probability added-area bound of
order
$$
 O\!\left(\frac{\log(Mn_0/\eta)}{n_0}
          +\frac{\log(Mn_1/\eta)}{n_1}\right).
$$
These area bounds concern the underlying continuous-index envelopes.
Conservative reporting-grid extension with maximum mesh $h$ adds at most
$2h$ to the area of a monotone-edged band: at most $h$ for each edge's
one-sided step approximation. Thus (3.3) and the added-area bound (3.5)
remain valid for the respective reported grid bands with an extra $2h$
on the right. On the native grid, $h=1/n_0$.

This is an upper bound, often loose. It is not a uniform coverage theorem
for the bracketed fiducial tube. Re-trimming the bracket cloud can change
the retained set or rank index and loses this particular containment
guarantee.

**Regular-interior comparison. [Asymptotic; proof outline]** If $R$ is
Lipschitz on a neighborhood of a fixed interior interval and the class
sizes are comparable, the completion diameter there is
$O_p(\log n/n)=o_p(n^{-1/2})$. Indeed, the largest true negative probability
gap is $O_p(\log n/n)$; bounded ROC slope bounds its positive probability
mass at the same order. A binomial tail bound controls how many positives
it contains, and exponential-sum bounds control their total auxiliary
positive spacing. The end pieces are treated on the enlarged interval.
This also gives conditional negligibility in probability.

The argument must count *all* positive spacings crossed by a negative gap.
A negative gap can contain many positives; a one-positive-spacing
supremum bound is false. For a Hölder-$\beta$ ROC the corresponding route
gives a leading $O_p((\log n/n)^\beta+\log n/n)$ bound; $\beta>1/2$
makes completion negligible at the root-$n$ scale. Jumps invalidate that
supremum conclusion, but not (3.3).

The finite-grid bracket, sorted-uniform completion, and interpolation are
measurable functions of labels and continuous auxiliary variables.
Rational evaluation points determine suprema for their monotone
one-sided extensions. These observations avoid a selection-measurability
gap in the process arguments.

## 4. Exact empirical moments and what they say about direction

Use the production native grid $t_k=k/n_0$. For $k<n_0$, let
$\widehat k_k$ count positives above the $(k+1)$th largest negative, and
set $\widehat R(t_k)=\widehat k_k/n_1$. At $k=n_0$, set
$\widehat R(1)=1$. This is an upper empirical staircase convention.

**Proposition 4 (Beta–binomial mixture, not generally Beta–binomial).
[Exact]** With $j=k+1$,
$$
 U_{(j)}\sim\operatorname{Beta}(j,n_0+1-j),\qquad
 \widehat k_k\mid U_{(j)}
       \sim\operatorname{Binomial}(n_1,R(U_{(j)})).
$$
Thus, writing $\bar p=E R(U_{(j)})$,
$$
 E\widehat R(t_k)=\bar p,\qquad
 \operatorname{Var}\widehat R(t_k)
 =\frac{\bar p(1-\bar p)}{n_1}
   +\left(1-\frac1{n_1}\right)\operatorname{Var}R(U_{(j)}).
 \tag{4.1}
$$
This is an ordinary Beta–binomial distribution only when the transformed
success probability has the appropriate Beta law, for example $R(t)=t$.

If $R$ is twice continuously differentiable near an interior $t=t_k$,
$$
 E\widehat R(t)-R(t)
 =R'(t)\frac{1-t}{n_0+1}
  +\frac12R''(t)\frac{t(1-t)}{n_0}
  +o(n_0^{-1}).                                             \tag{4.2}
$$
The first term is the native-grid convention, since
$EU_{(k+1)}=(k+1)/(n_0+1)$; it should not be mistaken for curvature bias.
The second is the usual quantile-jitter curvature effect.

Jensen's inequality also gives
$ER(U)\ge R(EU)$ for a globally convex $R$, with the reverse for concavity.
These are statements about the estimator's mean, not signs of a
simultaneous confidence-band violation. Tail quantiles, completion bias,
and asymmetric allowances still matter.

## 5. The rank trim and its precise finite-cloud content

Let $c_b(t_k)$, $b=1,\ldots,M$, $M\ge2$, be the cloud. Let $J$ be the columns
actually used for trimming. Define inclusive two-sided pointwise ranks
$$
 d_{bk}=\min\{\#a:c_a(t_k)\le c_b(t_k),
               \#a:c_a(t_k)\ge c_b(t_k)\},\qquad
 D_b=\min_{k\in J}d_{bk}.
$$
Ties receive the inclusive count on each side. Sort the $D_b$ increasingly
and take the $(\lfloor aM\rfloor+1)$th value, with the implementation's
clipping to at most $\lfloor M/2\rfloor$; call the result $j$.
Here $a=\alpha_{\rm eff}=1-(1-\alpha)^C$, with default $C=1$.

At every output-grid column form
$$
 L_{\rm raw}(t_k)=c_{(j)}(t_k),\qquad
 U_{\rm raw}(t_k)=c_{(M-j+1)}(t_k),\qquad
 \ell=\frac{j}{M+1}.                                       \tag{5.1}
$$
The raw band is a pointwise order-statistic tube, not necessarily the
pointwise envelope of a selected set of whole curves.

**Lemma 5 (rank/tube equivalence). [Exact]** A draw lies in the tube at
every trim column iff $D_b\ge j$. This remains true with the stated
inclusive ranks.

**Proposition 6 (in-sample content only). [Exact]** At most
$\lfloor aM\rfloor$ cloud draws fail the tube on the trim grid, so its
empirical content there is at least $1-a$.

This does not establish content for a fresh conditional draw, much less
coverage of the population ROC. Also, if $J$ is a thinned grid, the claim
does not extend to all output columns.

**Depth saturation.** For a fixed cloud, $j=1$ is its widest possible
order-statistic tube. If more than $aM$ draws have depth one, changing $C$
over a range can leave the band unchanged. Reducing $M$ is not a monotone
widening operation: it changes the cloud itself. Nor does a finite cloud
missing the truth imply that the truth is outside the support of the
infinite conditional law.

An extreme-rank-length refinement orders the sorted vectors of pointwise
depths lexicographically, breaking minimum-depth ties. It can give finer
content control, but does not solve the credible-to-confidence problem.
An envelope of curves with $D_b\ge j$ is contained in the $j$ tube.
There is no general reverse inclusion of a narrower rectangular tube
inside that curve envelope: different coordinates can be extremized by
different rejected curves.

**The production corner allowances.** The upper edge is widened by
$$
 U_{\rm CP}(t_k)=
 \begin{cases}
 \operatorname{Beta}^{-1}_{1-\ell}
       (\widehat k_k+1,n_1-\widehat k_k),&\widehat k_k<n_1,\\
 1,&\widehat k_k=n_1.
 \end{cases}
$$
Then take the running maximum of the larger of this and the raw upper
edge. The lower edge is zeroed where $\widehat k_k=0$, with $L(0)=0$;
the upper endpoint is one.

These operations widen the raw tube and cannot decrease its coverage.
They do **not** independently establish a Clopper–Pearson guarantee for
$R(t_k)$: the threshold is a random negative order statistic, its true FPR
is $U_{(k+1)}$, and $\ell$ was selected using the same data and cloud.
For a fixed threshold selected independently of the positive sample,
and a fixed local error budget, the ordinary binomial argument is exact.
Section 10 uses that distinction constructively.

## 6. Interior asymptotics, including directional balance

### 6.1 The two empirical processes

**Theorem 7 (regular-interior equivalence and calibration).
[Asymptotic; proof outline with the bootstrap conditions verified]**

Fix $I=[\varepsilon,1-\varepsilon]$. Assume $R$ is continuously
differentiable on a neighborhood of $I$, with $0<R'<\infty$ there,
and $n_1/n_0\to\lambda\in(0,\infty)$. Then, in $\ell^\infty(I)$,
$$
 \sqrt{n_1}(\widehat R-R)\Rightarrow Z,\qquad
 \mathcal L_*\{\sqrt{n_1}(\widetilde R-\widehat R)\}
       \Rightarrow_{\!P}\mathcal L(Z),                     \tag{6.1}
$$
where independent standard Brownian bridges give
$$
 Z(t)=B_1(R(t))-\sqrt\lambda\,R'(t)B_0(t).                  \tag{6.2}
$$
The limit is continuous. The empirical staircase itself is not an element
of $C(I)$, so convergence is stated in $\ell^\infty(I)$.

For the raw ELL tube **trimmed on $I$**, also require that the trim-grid
mesh tends to zero, $M_n/\log K_n\to\infty$, and the limiting depth cutoff
is a continuity and local strict-increase point. Then
$$
 P_R\{L_{\rm raw}(t)\le R(t)\le U_{\rm raw}(t)
              \text{ for every }t\in I\}\longrightarrow1-a. \tag{6.3}
$$
Here $K_n$ is the number of evaluated interior columns. Native-grid
spacing is sufficiently fine under the stated sample-size ratio.

*Proof outline.* There are four distinct steps.

1. **One-class weighted bridge.** Let
   $W_{ni}=nE_i/\sum_{r=1}^nE_r$, with iid Exp$(1)$ variables.
   These are exchangeable, nonnegative weights summing to $n$.
   Their moments of every fixed order are uniformly bounded, and
   $n^{-1}\sum(W_{ni}-1)^2\to1$ in probability.
   Thus conditions A1–A5 of Praestgaard–Wellner hold: the moment bound
   supplies the required $L_{2,1}$ and tail conditions. The class of
   half-line indicators is measurable and Donsker with bounded envelope.
   Their Theorem 2.1 gives the conditional weighted empirical bridge;
   Theorem 2.2 also supplies the weaker in-probability version sufficient
   here. [Praestgaard–Wellner (1993), pp. 2056–2057](https://sites.stat.washington.edu/jaw/JAW-papers/jaw-praest-93AP.pdf).
2. **Transfer to $n+1$ spacings.** Use the same exponentials and compare
   the partial sums normalized by $\sum_{1}^{n}E_i$ and by
   $\sum_{1}^{n+1}E_i$. Their maximum anchor discrepancy is at most
   $E_{n+1}/\sum_{1}^{n}E_i=O_p(n^{-1})$. Moving between step conventions
   costs at most a maximal spacing. For two-class ROC completion, use the
   regular-interior bound in §3.1. All these errors are
   $o_p(n^{-1/2})$, including conditionally in probability.
3. **Differentiate the ROC map.** In placement coordinates,
   $\Psi(A,B)=B\circ A^{-1}$ and the population pair is
   $(\operatorname{id},R)$. For continuous tangents,
   $$
    \dot\Psi(h_0,h_1)(t)=h_1(t)-R'(t)h_0(t).
   $$
   Expand the inverse and then $R$ uniformly on the enlarged interior
   interval. Independent class bridges give (6.1)–(6.2).
4. **Transfer the trim.** The Gaussian pointwise tail depth of a path $z$
   is $\Phi(-|z(t)|/\sigma(t))$, where
   $$
    \sigma^2(t)=R(t)(1-R(t))+\lambda R'(t)^2t(1-t).
   $$
   Hence the limiting ELL central set is exactly
   $\{\|z/\sigma\|_{\infty,I}\le c_a\}$, where
   $$
    P\{\|Z/\sigma\|_{\infty,I}\le c_a\}=1-a.
   $$
   Marginal empirical-CDF control (§11) and cutoff regularity transfer
   this set to the finite cloud. Finally $-Z\overset d=Z$, so the truth
   has the same limiting inclusion probability as a centered cloud draw.

A publication appendix should spell out the uniform conditional
quantile/depth convergence and completion comparison. This is a
construction-specific proof outline, not a fully formal appendix proof.

### 6.2 The scope matters for the shipped algorithm

Three distinctions prevent overinterpreting Theorem 7.

- **Interior coverage versus interior calibration.** Trimming over the
  whole curve and then inspecting $I$ does not give equality in (6.3).
  Other locations determine a different cutoff. Full-grid calibration
  can be more conservative on $I$, and moving tails are outside the
  theorem altogether.
- **Capped simulation budget.** Production's automatic $M$ is clipped to
  2,000–20,000. This does not satisfy $M_n/\log K_n\to\infty$ along an
  unbounded sequence.
- **Thinned trim grid.** Above 2,001 columns, production keeps about
  1,000 interior trim points plus the first and last 50. Its interior
  mesh does not tend to zero. Building the tube on the full grid does
  not restore the omitted depth constraints.

Thus Theorem 7 concerns a well-defined asymptotic version, not full-curve
asymptotic validity for all production settings. These are specifications
to test or alter in a future implementation, not changes made here.

In the **interior-calibrated** regime, fixed $C>1$ has limiting coverage
$(1-\alpha)^C$, below nominal. An exponent remap is not asymptotically
free. This is not a theorem assigning that exact limit to the capped,
thinned, full-curve implementation.

### 6.3 Which side should fail?

**Proposition 7a (sharp near-balance in the Gaussian limit).
[Exact for the limiting Gaussian band; asymptotic for Theorem 7]**

For $s,t\in I$ the covariance is
$$
 \begin{aligned}
 K(s,t)
 &=R(\min(s,t))-R(s)R(t)\\
 &\quad+\lambda R'(s)R'(t)\{\min(s,t)-st\}\ge0.
 \end{aligned}                                           \tag{6.4}
$$
Let $X(t)=Z(t)/\sigma(t)$ and choose $c$ with
$P(\sup_I|X|>c)=a$. Define
$$
 p=P(\sup_I X>c)=P(\inf_I X<-c),\qquad
 b=P(\sup_I X>c,\ \inf_I X<-c).
$$
Then
$$
 \boxed{\quad
 \frac a2\le p\le1-\sqrt{1-a},\qquad
 0\le b\le(1-\sqrt{1-a})^2.
 \quad}                                                   \tag{6.5}
$$

*Proof.* Gaussian sign symmetry gives equal directional probabilities,
and inclusion–exclusion gives $a=2p-b$. Nonnegative Gaussian covariances
imply association: increasing events are positively correlated.
The upper excursion is increasing and the lower excursion is decreasing,
so $b\le p^2$. Therefore $a\ge2p-p^2$; combine this with $b\ge0$.
Apply the finite-dimensional result on increasing finite grids and pass
to the continuous-path limit. The association result is
[Pitt's theorem (1982), author-provided paper](https://www.researchgate.net/publication/38361783_Positively_Correlated_Normal_Variables_Are_Associated).
$\square$

At $a=.05$, each directional whole-interval probability is between
.025000 and .025321, and both-sided failure is less than .000642.
At $a=.5$, the corresponding upper limit is approximately .29289322.
This is substantially stronger than the generic range $[a/2,a]$.

For the data-centered band, truth below the lower edge corresponds to a
positive empirical excursion; truth above the upper edge corresponds to
a negative one. Large systematic lower-edge imbalance cannot be explained
by the symmetric first-order interior law alone. It directs attention to
tails, completion, finite-sample bias, or asymmetric postprocessing.
It does not identify which one without experiments.

### 6.4 Equal local levels are not equal regional errors

In the same limit, the pointwise one-sided failure probability is
$\Phi(-c)$ at every nondegenerate interior FPR. This is a genuine reason
to like studentization.

But $P(\sup_{t\in A}X(t)>c)$ also depends on correlations within $A$.
Equal-length FPR intervals can contain different numbers of effectively
distinct excursions. Equal local levels therefore do not equalize
regional failure rates, first-crossing locations, or the location of
maximum violation. Exact endpoints cannot have the same error rate as
nondegenerate interior points.

A regional-level design can assign thresholds $c(t)$, calibrated against
the *joint* process, to redistribute errors. For the Gaussian surrogate,
the explicit area optimization is
$$
 \min_{c(t)\ge0}\int_I \sigma(t)c(t)\,dt
 \quad\text{subject to}\quad
 P\{|Z(t)|\le\sigma(t)c(t)\ \forall t\in I\}\ge1-\alpha.
 \tag{6.6}
$$
Regional balance constraints make the efficiency tradeoff explicit.
Estimating this design from the same sample requires a further
asymptotic argument; it is not an exact distribution-free band.

### 6.5 What an unconditional fiducial draw looks like

Conditional convergence in (6.1) to a nonrandom law gives
$$
 \left(\sqrt{n_1}(\widehat R-R),
       \sqrt{n_1}(\widetilde R-\widehat R)\right)
       \Rightarrow (Z,Z'),
$$
where $Z'$ is an independent copy of $Z$. Consequently
$$
 \sqrt{n_1}(\widetilde R-R)\Rightarrow Z+Z',
$$
whose covariance is $2K$.

The conditional cloud must reproduce estimation uncertainty *around its
random data center*. An unconditional draw includes both center error
and auxiliary error. Dividing the conditional spread by $\sqrt2$ would
destroy the calibration argument.

## 7. Why the tails are a different experiment

### 7.1 Interior Gaussian scale versus endpoint spacing scale

At fixed interior FPR, fluctuations are root-$n$ Gaussian. For fixed $k$,
the negative order statistic satisfies
$n_0U_{(k)}\Rightarrow\operatorname{Gamma}(k,1)$.
Endpoint positive spacings have the exponential scale $1/n_1$.
Replacing these by their means erases order-one *relative* uncertainty.

The maximum-gap result does not repair this automatically. A gap of
absolute mass $O(1/n)$ can be the entire mass of a tail feature. For a
sliver whose expected observed count stays bounded, the chance of seeing
none remains bounded away from zero.

### 7.2 What the completed experiments establish

The [boundary follow-up](c_calibration_followup_report.md) found a curved,
nonmonotone failure wedge for shifted heavy-tailed families. The
[Stage F report](hybrid_floor_report.md) subsequently tested 160 cells
with 42,000 paired replicates. At $\alpha=.05$:

| Design | Cells | Raw $C=1$ coverage, cell macro | Frontier hybrid |
|---|---:|---:|---:|
| A: enriched replay and stress | 116 | .9256 | .9843 |
| B: prospective adversarial transfer | 30 | .8294 | .9823 |
| C: seven shapes at two sizes | 14 | .9679 | .9804 |

These are design-specific averages, not coverage averaged over a natural
population of ROC curves. The measured hybrid minimum was .940 in A.
The report's interval check did not find a significantly undercovered
cell; that does not rule out deficits at its replication counts.

Fresh sliver cells gave raw coverage .505–.613 and hybrid coverage
.978–.988. In five cells, raw conditional coverage was zero observed
when the sliver went unobserved. Lower-edge failing replicates fell
from 1,413 to 130 in A and from 1,990 to 98 in B; hybrid upper-edge
counts were 234 and 137. The floor repairs the dominant directional
defect but leaves a residual tilt toward upper-edge misses.

Residual hybrid violations were mostly in the unprotected interior;
halving the M3 error level barely changed coverage. A wider boundary
margin is not automatically the best next expenditure of width.
See the report for region localization and paired evidence.

### 7.3 Calibration constants are diagnostics, not laws

A fit of coverage versus $C$, sample size, or AUC is a calibration model.
It can fail when trim depth is discrete, when depth saturates, when a new
tail feature becomes visible, or when a sliver remains unseen.

There is no proved monotonicity of coverage in sample size, AUC, or
tail index for this band. Apparent erosion laws, global powers
$(1-\alpha)^\gamma$, and universal numerical wedges should not be
promoted to distribution-free statements.

### 7.4 Convex hooks: exact geometry, approximate risk

Let $\tau(s)=1-R(1-s)$ and $\rho(s)=\tau(s)/s$. When densities exist,
$R'(t)=g(x)/f(x)$ at $x=Q_F(1-t)$.
Concavity of $R$ on a right-tail interval implies convexity of $\tau$
there and nondecreasing $\rho$ when the interval starts at zero
with $\tau(0)=0$. Convexity of $R$ gives the reverse.

The converse from monotonicity of $\rho$ to curvature is false:
a monotone secant-slope ratio is a star-shapedness condition, weaker than
convexity. A tail likelihood ratio tending to a positive constant
does not determine which direction it approaches that constant.

**Exact shifted-Student-$t$ geometry.** For negatives $T_\nu$ and positives
$T_\nu+\delta$, $\delta>0$, differentiation of the log likelihood ratio gives
turning thresholds
$$
 x_\pm=\frac{\delta\pm\sqrt{\delta^2+4\nu}}2.
$$
The ROC is convex for thresholds $x>x_+$ and $x<x_-$, and concave between.
Its hook intervals are therefore
$$
 [0,t_L],\quad[t_R,1],\qquad
 t_L=P(T_\nu>x_+),\quad t_R=P(T_\nu>x_-).                   \tag{7.1}
$$
Since $x_-<0$ and $x_-\to0$ as $\delta\to\infty$,
$t_R>1/2$ and $t_R\to1/2$. Within this family, $[1/2,1]$ is the smallest
fixed right-terminal interval containing every right hook as the
positive shift varies. This is geometric optimality, not optimality
of a confidence-band floor.

**Endpoint mechanism model. [Heuristic, not an inequality]**
Uniform completion spreads an end spacing over a run that may contain
no evidence about the true within-run shape. At the left, with a fixed
local level $\ell$ and $Q=\log(1/\ell)$, a large-positive-run approximation
gives
$$
 L(1/n_0)\approx\frac{p_1}{n_1Q},\qquad
 r_L\approx
 \exp\{-n_0R^{-1}(Q R(1/n_0))\},                           \tag{7.2}
$$
where $p_1$ counts positives above the largest negative. Use the formula
only when $QR(1/n_0)<1$ and the count approximation is adequate.
For a linear corner it gives $\ell$; convexity can inflate this
lower-edge risk.

At the right, let $K$ count negatives below the smallest positive.
For $K>0$ and $1\le k\le K$, a Poissonized finite-grid approximation
for the deficit $k$ negative ranks from FPR one is
$$
 n_1(1-\widetilde R(1-k/n_0))
       \ \dot\sim\ E\,\frac{k}{K} Z_k,\qquad
 E\sim\operatorname{Exp}(1),\quad
 Z_k\sim\operatorname{Gamma}(k,\text{rate }k),               \tag{7.3}
$$
with independent factors in this approximation. If
$P(EZ_k>q_{k,\ell})=\ell$, then
$$
 P(EZ_k>q)=
 \frac{2(kq)^{k/2}}{\Gamma(k)}
 \mathcal K_k(2\sqrt{kq}),                                 \tag{7.4}
$$
where $\mathcal K_k$ is the modified Bessel function. This identity is
exact for the *product model*: integrate $e^{-q/z}$ against the Gamma
density. It does not make (7.3) exact for the production cloud.
As $k\to\infty$, $Z_k\to1$ and $q_{k,\ell}\to Q$; no monotonicity is needed.

Comparing (7.3) with the true deficit suggests the risk condition
$$
 \frac{n_1\tau(k/n_0)K}{k}>q_{k,\ell}.
$$
This preserves more endpoint randomness than replacing every gap by its
mean and helps explain the observed failure wedge.

The limitations are material: $K$ is data-dependent, selected $\ell$
is correlated with the data, small counts and axis interpolation matter,
and the two corner failures are not independent. Neither (7.2) nor
$1-(1-r_L)(1-r_R)$ is a proved miss probability. These belong in
diagnostic scores with prospective validation, not in a claimed error budget.

## 8. Coverage inequalities we can actually use

**Lemma 8 (widening and direction). [Exact]** If
$L_2\le L_1$ and $U_2\ge U_1$ pointwise, including after any data-dependent
selection, then
$$
 V^\pm(B_2;A)\subseteq V^\pm(B_1;A)
$$
for every region $A$. In particular $q_R(B_2)\le q_R(B_1)$.

If only the lower edge is lowered, upper-edge failures are unchanged.
Writing $V^-_i=V^-(B_i)$ and $V^+=V^+(B_1)=V^+(B_2)$, the exact gain is
$$
 q_R(B_1)-q_R(B_2)
   =P_R\{V^-_1\setminus(V^-_2\cup V^+)\}.                  \tag{8.1}
$$
Repairing a lower miss on a replicate that still misses above does not
improve simultaneous coverage. A one-sided floor can overshoot directional
balance even while improving coverage.

**Hybrid inequality. [Exact]** Let $B_F$ be any fiducial band and $B_M$
a full-curve honest band at error $\alpha_M$. Choose an arbitrary
data-dependent region $A$, take their pointwise hull on $A$, leave $B_F$
outside, and apply widening-only closure. Call the result $B_H$. Then
$$
 \boxed{\quad
 q_R(B_H)\le
 \min\left\{q_R(B_F),\
       \alpha_M+P_R(V(B_F;A^c))\right\}.
 \quad}                                                   \tag{8.2}
$$
No independence between $A$, either band, and the data is needed.
On the event that $B_M$ covers globally, the hybrid cannot miss in $A$.
A miss outside $A$ implies a fiducial exterior miss. This proves (8.2).

Directionally, replace $\alpha_M$ by
$P_R(V^\pm(B_M))$ and the exterior event by $V^\pm(B_F;A^c)$.
M3's global guarantee alone bounds each of these M3 probabilities by
$\alpha_M$; it does not allocate $\alpha_M/2$ to each side.

This is why rank-only, AUC-based, or cloud-based selection of a **floor
region** is allowed in (8.2). It is also why the regional M3 cap is not a
whole-band guarantee: the exterior term is still unknown.

**Closure, intersection, and selection. [Exact]**

- Stage F uses lower reverse cumulative minima and upper forward
  cumulative maxima. This widens the hull and preserves all preceding
  containments. It can propagate area cost outside the chosen region.
- The usual shape tightening—lower forward cumulative maxima and upper
  reverse cumulative minima—does not widen a band, but still preserves
  any event of global coverage of a monotone truth. If tightening yields
  an empty interval, the original band could not have covered a
  monotone truth globally.
- The full pointwise hull of two bands has error at most the smaller
  marginal error. Their intersection has error at most the sum of their
  marginal errors. Neither statement requires independence.
- If several bands are all valid on the **same common pivot event** of
  probability at least $1-\alpha$, their intersection or any selected
  one remains valid on that event. Separate marginal guarantees of
  $1-\alpha$ are not the same thing.

**Proposition 8a (quantitative coverage transfer).
[Exact implication; approximation terms require separate bounds]**

Let $T_n=\|\sqrt{n_1}(\widehat R-R)/\sigma\|_{\infty,I}$ and
$T=\|Z/\sigma\|_{\infty,I}$. Suppose
$$
 \sup_x|P(T_n\le x)-P(T\le x)|\le\kappa
$$
and, except on an event of probability $\delta$, the scaled lower and
upper band displacements from $\widehat R$ are uniformly within $r$
of $-c$ and $c$, respectively, after division by $\sigma(t)$.
Equivalently, both
$\|\sqrt{n_1}(L-\widehat R)/\sigma+c\|_{\infty,I}$ and
$\|\sqrt{n_1}(U-\widehat R)/\sigma-c\|_{\infty,I}$ are at most $r$.
Then, with $F_T$ the CDF of $T$,
$$
 F_T(c-r)-\kappa-\delta
 \le P\{R\in B\text{ on }I\}
 \le F_T(c+r)+\kappa+\delta.                               \tag{8.3}
$$
*Proof.* On the good approximation event,
$\{T_n\le c-r\}\subseteq\{R\in B\}\subseteq\{T_n\le c+r\}$.
Apply the two stated bounds. $\square$

This separates sampling approximation ($\kappa$), band/Monte Carlo
approximation ($r,\delta$), and Gaussian anti-concentration
$F_T(c+r)-F_T(c-r)$. We do not yet have useful uniform finite-sample
constants for these terms over arbitrary ROC shapes.

## 9. Missing-mass lower bounds: what cannot be made narrow

These arguments use continuous score laws and require no asymptotics.
They address unseen features directly rather than inferring impossibility
from a finite cloud.

**Lemma 9 (unseen positive mass). [Exact; derived here]** Let the baseline
$P_0$ have negatives uniform on $[0,1]$ and positives uniform on $[2,3]$.
Its ROC is $R_0(t)=1$ everywhere. In the alternative, positives have law
$$
 (1-\pi)\operatorname{Unif}[2,3]
       +\pi\operatorname{Unif}[-2,-1],
$$
so $R_\pi(t)=1-\pi$ for $0\le t<1$, and $R_\pi(1)=1$.

With probability $w=(1-\pi)^{n_1}$ no rare positive is observed.
Conditional on that event, the entire observed data law, including
independent algorithm randomization, is exactly the baseline law.
Every band honest at the alternative must therefore satisfy
$$
 w\,P_0\{\exists t<1:L(t)>1-\pi\}\le\alpha.                 \tag{9.1}
$$
If it is also honest at the baseline,
$$
 P_0\{W(B)\ge\pi\}\ge
       [\,1-\alpha-\alpha/w\,]_+.                         \tag{9.2}
$$
*Proof.* Under the alternative, the event in (9.1) is a lower-edge failure.
On the baseline event that $U(t)\ge1$ everywhere and
$L(t)\le1-\pi$ for every $t<1$, the area is at least $\pi$.
A union bound gives (9.2). $\square$

This is a probability statement for arbitrary, possibly randomized
procedures. It does not say every realized honest band must have a
particular pointwise edge.

**Corollary 9a (deterministic rank-only bands). [Exact]**
On the perfectly separated rank path, a deterministic rank-only band,
with edges in $[0,1]$ and honest over all continuous score distributions,
must satisfy
$$
 U(t)=1,\qquad L(t)\le\alpha^{1/n_1}\quad(0\le t<1).        \tag{9.3}
$$
The upper assertion follows from baseline validity: this path occurs
with probability one. For the lower edge, if
$(1-\pi)^{n_1}>\alpha$, failing anywhere on that path would already
violate alternative coverage. Let $\pi\uparrow1-\alpha^{1/n_1}$.

**Corollary 9b (unseen high-scoring negatives). [Exact]** For the same
separated rank path, put a fraction $\epsilon$ of negative mass above
the positive support. Its ROC is zero for $0\le t<\epsilon$, while
the chance of seeing no such negative is $(1-\epsilon)^{n_0}$.
Consequently the same deterministic honest band must satisfy
$$
 L(t)=0\quad\text{for }0\le t<1-\alpha^{1/n_0}.             \tag{9.4}
$$
For a randomized procedure the corresponding statement is
$(1-\epsilon)^{n_0}P_0\{\exists t<\epsilon:L(t)>0\}\le\alpha$.

Combining (9.3) and (9.4) gives an area lower bound on the separated path:
$$
 \boxed{\quad
 W(B)\ge1-\alpha^{\,1/n_0+1/n_1}
 \sim\log(1/\alpha)(1/n_0+1/n_1).
 \quad}                                                   \tag{9.5}
$$
The deterministic rank-only qualification is essential; (9.1)–(9.2)
are the appropriate version when Monte Carlo randomization remains.

These explain an unavoidable $1/n$ tail-width cost even for apparently
perfect separation. They do not prove M3 is optimal, or that every
dataset requires zero lower edge over a universal seven-rank strip.
The constants follow from a specified indistinguishability argument
and error level.

## 10. Hybrid cut points with exact meanings

The Stage F frontier uses roughly $\lceil\log(M+1)\rceil$ leftmost ranks
and a right saturated run expanded by $\lceil2\sqrt K\rceil$ ranks.
Those are useful empirical choices. We can replace their interpretation
with explicit probability budgets without claiming the entire hybrid
has thereby been calibrated.

### 10.1 Left end: the smallest cut controlling end-gap reach

The auxiliary negative end spacing at FPR zero is
$D\sim\operatorname{Beta}(1,n_0)$, conditional on any observed labels.
Its probability of reaching $k/n_0$ is exactly
$$
 P_*(D>k/n_0)=(1-k/n_0)^{n_0}.
$$
For a fixed $\epsilon\in(0,1)$, the smallest integer cut with reach
probability at most $\epsilon$ is
$$
 \boxed{\quad
 k_L(\epsilon)=
 \left\lceil n_0\{1-\epsilon^{1/n_0}\}\right\rceil
 \le\lceil\log(1/\epsilon)\rceil .
 \quad}                                                   \tag{10.1}
$$
The same Beta law describes the *true* FPR of the largest negative
unconditionally. Thus (10.1) has both a sampling interpretation and an
auxiliary-gap interpretation. Their events are not identical.

A fixed-grid count of auxiliary draws whose end gap reaches a point is
Binomial$(M,p)$ with this $p$. To ensure that *none* of the $M$ end gaps
reaches the unprotected region with probability at least $1-\eta$, one
sufficient choice is $\epsilon=\eta/M$. This explains a $\log M$ scale,
but adds an explicit error budget missing from “take $\log M$.”

Alternatively, $\epsilon=\ell$ controls the reach probability of one draw
at the local-level scale. It does not make the gap irrelevant to the
$\ell$ quantile. If two coupled completion laws differ only on an event
of probability $p$, their CDFs differ by at most $p$, and their quantiles
are bracketed by the other law's levels $q-p$ and $q+p$ (when in $(0,1)$).
A small-probability event can still move a tail quantile.

One may choose (10.1) conditional on a chosen numerical $\ell$ for an
auxiliary diagnostic. Substituting a data-selected $\ell$ into a
fixed-budget sampling claim is not justified without additional accounting.

### 10.2 Right end: an exact margin around the saturated-run boundary

Let $Y_{\min}$ be the smallest observed positive, and define
$$
 S=F(Y_{\min}),\qquad K=\#\{i:X_i<Y_{\min}\}.
$$
The unknown boundary FPR is $1-S$; the observed run length estimates
$n_0S$. Conditional on the positive sample,
$$
 K\mid Y_1,\ldots,Y_{n_1}\sim\operatorname{Binomial}(n_0,S).
 \tag{10.2}
$$
This conditional binomial statement is exact because the threshold uses
only the independent positive sample.

For a **fixed** $\delta\in(0,1/2)$, define the one-sided limits
$$
 s_U(K;\delta)=
 \begin{cases}
 \operatorname{Beta}^{-1}_{1-\delta}(K+1,n_0-K),&K<n_0,\\
 1,&K=n_0,
 \end{cases}
$$
$$
 s_L(K;\delta)=
 \begin{cases}
 \operatorname{Beta}^{-1}_{\delta}(K,n_0-K+1),&K>0,\\
 0,&K=0.
 \end{cases}                                             \tag{10.3}
$$
Then
$$
 P(S>s_U)\le\delta,\qquad P(S<s_L)\le\delta.                \tag{10.4}
$$

**Proposition 10a (the upper boundary bound has a second, fiducial meaning).
[Exact]** Conditional on the merged labels, the auxiliary F-coordinate
$\widetilde S$ of $Y_{\min}$ lies between the $K$th and $(K+1)$th
ascending negative CDF anchors. These have Beta$(K,n_0+1-K)$ and
Beta$(K+1,n_0-K)$ marginals, respectively. Therefore
$$
 P_*(\widetilde S>s_U(K;\delta))\le\delta,\qquad
 P_*(\widetilde S<s_L(K;\delta))\le\delta.                 \tag{10.5}
$$
Use the sentinel values at $K=0,n_0$. This holds for every within-gap
completion, not only uniform spreading.

Thus
$$
 A_R=[\,1-s_U(K;\delta),\,1\,]                             \tag{10.6}
$$
contains the entire true terminal interval $[1-S,1]$ with probability
at least $1-\delta$, and contains the analogous auxiliary interval with
conditional probability at least $1-\delta$. These are two separate
guarantees; their simultaneous intersection can be bounded by $1-2\delta$
without additional information.

Rounding (10.6) outward to the native grid gives an expanded right length
$$
 K_{\rm expanded}=\lceil n_0s_U(K;\delta)\rceil,\qquad
 d_K=K_{\rm expanded}-K.                                  \tag{10.7}
$$
This handles $K=0$ correctly and incorporates the binomial finite-population
factor. For $K/n_0$ bounded away from zero and one,
$$
 d_K=z_{1-\delta}\sqrt{K(1-K/n_0)}+O(1).                  \tag{10.8}
$$
Use the Beta expression for short runs, not this approximation.

For example, at $n_0=500$ and $\delta=.025$:

| Observed $K$ | Exact extra ranks $d_K$ | $\lceil2\sqrt K\rceil$ |
|---:|---:|---:|
| 0 | 4 | 0 |
| 1 | 5 | 2 |
| 5 | 7 | 5 |
| 25 | 12 | 10 |
| 100 | 19 | 20 |
| 250 | 23 | 32 |

**What is optimal here? [Exact, restricted scope]** The upper limit
$s_U$ is pointwise smallest among nonrandomized nondecreasing upper
bounds $b(K)$ with binomial coverage at least $1-\delta$ for every $S$.
If $b(k)<s_U(k)$, choose $S$ strictly between them. Monotonicity implies
failure whenever $K\le k$, but
$P_S(K\le k)>\delta$ by the defining binomial inversion. This contradicts
coverage. For $k=n_0$, validity at $S=1$ forces $b(n_0)=1$.

This is optimality for enclosing a **single unknown boundary**, not for
minimizing expected ROC-band area or calibrating the hybrid. Randomized
bounds, nonmonotone confidence rules, and a different loss are outside
the assertion.

### 10.3 An exact tail-height certificate, using the other boundary limit

Put $D=G(Y_{\min})$. Then $D\sim\operatorname{Beta}(1,n_1)$, so
$$
 d_G=1-\delta_G^{1/n_1},\qquad P(D\le d_G)=1-\delta_G.
$$
On $\{S\ge s_L(K;\delta_F),\,D\le d_G\}$, monotonicity gives
$$
 R(t)\ge1-d_G
       \quad\text{for every }t\ge1-s_L(K;\delta_F).          \tag{10.9}
$$
The joint event has probability at least
$(1-\delta_F)(1-\delta_G)$: conditional on the positive sample, the
negative-sample limit fails with probability at most $\delta_F$;
then integrate over the positive spacing event.

The distinction matters: **$s_U$ encloses the whole uncertain end-gap
region; $s_L$ identifies a region safely inside it.** The lower-height
claim does not extend to all of (10.6). The certificate in (10.9) can
support a confidence-set construction with its own error accounting;
it is not permission to tighten an arbitrary band for free.

There is a dual left-side anchor. Let $X_{\max}$ be the largest negative
and $P_1$ count positives above it. For fixed budgets $\delta_0,\delta_1$,
take
$$
 a_U=1-\delta_0^{1/n_0},\qquad
 b_L=\operatorname{CP}_{\rm lower}(P_1;n_1,\delta_1).
$$
With probability at least $(1-\delta_0)(1-\delta_1)$,
$$
 R(t)\ge b_L\quad\text{for every }t\ge a_U.                 \tag{10.10}
$$
The proof conditions on the negative sample and applies the independent
positive binomial bound. Using many anchors or choosing the most favorable
one requires simultaneous calibration, not repeated pointwise claims.

### 10.4 From boundary uncertainty to an honest hybrid theorem

Let $E$ be an event on which the chosen floor region encloses the relevant
endpoint-completion region, and suppose $P(E^c)\le\delta_{\rm cut}$.
If one could additionally prove
$$
 P_R\{V(B_F;A^c)\cap E\}\le\alpha_I
       \quad\text{uniformly over the intended class},
$$
then (8.2) would give
$$
 q_R(B_H)\le\alpha_M+\delta_{\rm cut}+\alpha_I.             \tag{10.11}
$$
Equations (10.1)–(10.7) supply principled boundary events. The missing
ingredient is the exterior coverage bound, not another interpretation
of a Beta quantile.

For a class with common smoothness bounds, a possible route is to combine
(8.3) on a shrinking interior with exact endpoint brackets. The
root-$n$ approximation must then be uniform down to those moving cuts.
A fixed-interior theorem cannot simply be reused there. For unrestricted
ROC shapes, an unobserved interior feature can defeat a tails-only argument.

### 10.5 What “optimal hybrid cut points” should mean

For a chosen family
$A_{l,u}=[0,l/n_0]\cup[u/n_0,1]$, define the **post-closure** cost
$$
 \Delta W_{l,u}=W(B_{H,l,u})-W(B_F).
$$
It is computable exactly from the observed edges. The corresponding
oracle design problem is
$$
 \min_{l,u}E_R\Delta W_{l,u}
 \quad\text{subject to}\quad
 \alpha_M+P_R\{V(B_F;A_{l,u}^c)\}\le\alpha,                \tag{10.12}
$$
or its supremum-over-shapes version for a stated class. A design using
the exact hybrid risk may improve on this sufficient union-bound constraint.

For a data-dependent policy, put that entire policy inside the
probability in (10.12); fixed-cut marginal risks are not enough.
There is no distribution-free numerical optimizer until the exterior risk
is bounded. The exact boundary cuts are defensible candidates, not the
solution to that larger problem.

Within a declared shifted-$t$ parameter set, (7.1) supplies a different
geometric candidate: use the largest $t_L$ and smallest $t_R$ over that
set. A data-derived parameter set needs its own coverage allowance, and
covering the hook geometry still does not bound all non-hook failures.

## 11. Width and the Monte Carlo layer

### 11.1 First-order width and its limits

Under the interior-calibrated regime of Theorem 7,
$$
 U_{\rm raw}(t)-L_{\rm raw}(t)
   =\frac{2c_a}{\sqrt{n_1}}\sigma(t)+o_p(n_1^{-1/2})
$$
uniformly on $I$, and
$$
 \int_I(U_{\rm raw}-L_{\rm raw})
   =\frac{2c_a}{\sqrt{n_1}}\int_I\sigma(t)\,dt
       +o_p(n_1^{-1/2}).                                  \tag{11.1}
$$
Both sampling errors enter
$$
 v(t)=\frac{R(t)(1-R(t))}{n_1}
         +\frac{R'(t)^2t(1-t)}{n_0}.
$$
This quadratic combination helps explain why direct ROC inference can
be slimmer than projecting two separately protected CDFs. For
$$
 a_t=\sqrt{R(t)(1-R(t))/n_1},\qquad
 b_t=R'(t)\sqrt{t(1-t)/n_0},
$$
$$
 \sqrt{a_t^2+b_t^2}\le a_t+b_t
                \le\sqrt2\sqrt{a_t^2+b_t^2}.              \tag{11.2}
$$
This compares local uncertainty geometries **with equal multipliers**.
M3 and fiducial bands use different simultaneous cutoffs and finite-sample
constructions; (11.2) is not a universal factor bound between their widths.

For full bracketing, (3.5) bounds the completion cost without smoothness.
For a hybrid hull *before closure*, the added width is exactly
$$
 \int_A\big[
   \max(U_F,U_M)-\min(L_F,L_M)-(U_F-L_F)
 \big]\,dt.                                               \tag{11.3}
$$
Widening closure can add more outside $A$. The area cost is not generally
bounded by “tail height times selected strip length,” and a putative
global $Q(1/n_0+1/n_1)$ hybrid-cost inequality ignores F-axis uncertainty
and closure propagation.

WH and KS remain useful width comparators. Neither is a universal
lower or upper mathematical bound on this method, and M3's observed
width advantage over KS is not pointwise dominance for every dataset.

### 11.2 Finite-cloud approximation

Let $Q_n$ be the conditional cloud law and $F_{n,k}$ its marginal CDF at
column $k$. For the empirical marginal CDF $\widehat F_{M,k}$,
$$
 P_*\left\{\max_{k\le K}\sup_x
       |\widehat F_{M,k}(x)-F_{n,k}(x)|>\eta\right\}
       \le2K e^{-2M\eta^2}.                               \tag{11.4}
$$
This is DKW at each column followed by a union bound; no independence
across columns is assumed. In the regular-interior regime, it makes
marginal rank error vanish when $M/\log K\to\infty$. Convergence of the
depth cutoff additionally uses its continuity and the empirical
distribution of draw depths, not just marginal ranks.

At a fixed column and a fixed continuous-law lower-tail probability
$\ell$, the number of draws in that tail is Binomial$(M,\ell)$.
For example,
$$
 P_*(\text{no tail draw})=(1-\ell)^M.
$$
Thus $M\ell$ controls tail resolution, while (11.4) controls absolute
marginal-CDF error. The production rule targeting about five tail draws
is an *expected-count* rule, not a high-probability precision guarantee.
Its upper cap can prevent that target from being met.

Fresh conditional cloud content can be assessed without in-sample
optimism: build a tube on one auxiliary batch, evaluate whole-curve
inclusion on an independent batch, and use a binomial confidence bound
for its $Q_n$-miss probability. Choosing among several candidate tubes
requires simultaneous validation bounds or another independent batch.
This certifies conditional cloud content only, not population coverage.

### 11.3 Short verification experiment

The reproducible
[theory-check script](experiments/theory_checks_20260905.py) and
[results](experiments/res_theory_checks_20260905.json) check:

- the bracket-area formula by two independent calculations, its
  largest-spacing bound, and the shift inequality on 10,726 cases;
- every small merged path for class sizes one through four, strongly
  uneven spacings, and large imbalanced paths;
- a uniform-spacing example attaining ratio $201/202=.99505$ of the bound;
- binomial upper-limit coverage on parameter grids augmented just above
  the confidence-limit jumps, at 12 sample-size/error-level pairs;
- minimality of the integer left cut and the right-margin table in §10;
- positivity of Gaussian covariance matrices for concave, linear,
  and convex ROC examples under three imbalance ratios.

All assertions passed. These are formula and implementation checks,
not a coverage certification of a new fiducial method.
The proofs, rather than the experiment, establish the exact inequalities.

## 12. Exact alternatives and a selection-aware router

### 12.1 M3: the finite-sample anchor

**Proposition 12 (independent marginal pivot composition). [Exact]**
For each class $c$, choose simultaneous bounds on all its own order
statistics,
$$
 b^L_c[i]\le H_c(Z_{c,(i)})\le b^U_c[i],
$$
whose joint event $E_c$ has probability at least $1-\alpha_c$ for every
continuous $H_c$. ELL uses Beta marginal quantiles at a common local level,
calibrated by the uniform-order-statistic crossing probability, not
by treating the ranks as independent.

Take **all** monotone class-CDF completions consistent with these bounds
and the observed sample. Project their Cartesian product through the ROC
map, and use its pointwise outer envelope. On $E_0\cap E_1$, the true
CDF pair belongs to that set, so the envelope covers the full true ROC.
Class independence gives
$$
 P_R(R\in B_{\rm M3})\ge(1-\alpha_0)(1-\alpha_1).           \tag{12.1}
$$
Choose a fixed $\rho\in(0,1)$ and
$$
 1-\alpha_0=(1-\alpha)^\rho,\qquad
 1-\alpha_1=(1-\alpha)^{1-\rho}
$$
to obtain coverage at least $1-\alpha$. Dependence of $\rho$ on fixed
class sizes is allowed; choosing it from observed ranks is not justified
by this product argument alone.

Numerical calibration must preserve the target crossing probability
within a conservative numerical tolerance; the theorem is about the
calibrated pivot event, not unchecked floating-point exactness.

The [M3 implementation](../src/studroc_paper/methods/m3_band_rs.py)
performs this order-statistic composition with conservative quantile and
endpoint conventions. The theorem uses continuous *original class
CDFs*. Do not apply a continuous-CDF pivot directly to $R(W)$, because
the placement CDF $R$ may have atoms. Randomized ties are handled
through §2's continuous augmentation.

M3 can overcover because projection discards information and a failed
marginal-CDF event need not cause any ROC miss. At large alpha this
overcoverage is substantial in measured cases. The exact theorem does
not say it is close to nominal, minimal-width, directionally balanced,
or uniformly slimmer than KS.

**Known negative distribution.** If $F$ is known, the placement values
$W_j=1-F(Y_j)$ are observed iid draws from $R$. The problem reduces to
a one-sample CDF band, and the negative-class variance term vanishes.
An honest one-sample construction that allows atoms then gives an honest
ROC band directly. This is a useful stronger-guarantee special case,
not a justification for treating an estimated $F$ as known.

### 12.2 Direct test inversion: exactness without two-CDF rectangles

**Proposition 12a (testing one named ROC). [Exact]** For a fixed candidate
$R_0$, Proposition 1 specifies the complete null rank experiment.
Generate independent rank datasets from that law. Any statistic applied
symmetrically to the observed and simulated datasets gives an exchangeable
Monte Carlo rank test. With a fixed statistic and $B$ simulations, a
conservative upper-tail p-value is
$$
 p_{R_0}=
 \frac{1+\#\{b:T(\Lambda_b)\ge T(\Lambda_{\rm obs})\}}{B+1}.
 \tag{12.2}
$$
If the statistic itself is estimated from the simulation collection,
the whole operation must be permutation-equivariant; give the observation
the same role as each simulated dataset. Independent nested randomization
can be included symmetrically.

This is the exact-test route behind
[global rank envelopes](https://arxiv.org/abs/1307.0239).
It is not the ordinary fiducial experiment: there only the auxiliary
cloud is conditionally iid, and the truth is not another exchangeable draw.

Invert valid tests over the entire intended ROC class:
$$
 \mathcal C(\Lambda)=\{R:p_R(\Lambda)>\alpha\}.
$$
Then $P_R\{R\in\mathcal C\}\ge1-\alpha$, and any measurable outer band
containing every curve in $\mathcal C$ is honest. There is no multiplicity
penalty for the number of candidate curves: coverage only requires that
the one true curve not be rejected.

The hard part is computing a certified **outer** envelope over an infinite
shape class. A finite library, local optimization, or inner approximation
can omit the true curve and lose the theorem. A useful next method would
use the fiducial band as a search proposal or discrepancy geometry,
with exact rank tests supplying the confidence guarantee.

Passing a fitted-curve goodness-of-fit test does not certify a composite
shape class or establish noninferiority. Sample splitting alone does not
convert that logical implication into a theorem.

### 12.3 Router validity depends on selected errors

A router returns either $B_F$ or $B_M$, not their hull. If
$r(\Lambda)\in\{F,M\}$,
$$
 q_R(B_{\rm route})
 =P_R(r=F,V(B_F))+P_R(r=M,V(B_M))
 \le P_R(r=F,V(B_F))+\alpha_M.                             \tag{12.3}
$$
The selected fiducial term needs control. Marginal coverage estimates
for the two methods do not establish it. In contrast, a localized hull
inherits the direct containment argument (8.2).

AUC is $\int R=1-EW$. Fixing this single mean leaves endpoint mass,
curvature, gap geometry, and tail run distributions largely unrestricted.
It cannot identify the failure mechanism. Experiments also place bad
slivers outside the original high-AUC wedge.

The missing-mass construction gives a selection-aware warning. For its
alternative and baseline, with $w=(1-\pi)^{n_1}$,
$$
 q_{R_\pi}(B_{\rm route})
 \ge w\,P_0\{r=F,\ \exists t<1:L_F(t)>1-\pi\}.              \tag{12.4}
$$
On an unseen sliver, the router sees exactly the baseline data law.
A rank diagnostic can be useful, but cannot reveal mass that was not
observed. This does not prove every adaptive router is impossible;
it states the risk any honest one must accommodate.

### 12.4 A precise wedge definition and a precise optimization

For a stated class $\mathcal R$ and true AUC $A$, define
$$
 \mathcal W_{\mathcal R}
 =\left\{(A,n_0,n_1):
   \sup_{\substack{R\in\mathcal R\\\int R=A}}
       q_R(B_F)>\alpha\right\}.                           \tag{12.5}
$$
This worst-case unsafe set is useful even if it has no simple wedge
shape. Replacing true AUC by its estimate creates a selection problem;
it is not just a noisier lookup of the same theorem.

For a practical empirical router, useful features include both class
sizes, $K$, the top-positive run, realized depth $j$, and a bracket or
local-completion sensitivity measure. A declared-family hook score may
be added as a model-specific feature. None is presently a uniform safety
certificate.

To define “optimal,” specify a distribution over shapes, routing
features $Z_{\rm route}$, and a loss. For expected width plus $\kappa$
times failure, define conditional quantities
$$
 w_j(z)=E\{W(B_j)\mid Z_{\rm route}=z\},\qquad
 q_j(z)=P\{V(B_j)\mid Z_{\rm route}=z\}.
$$
The Bayes-optimal switch to M3 is exactly
$$
 \kappa\{q_F(z)-q_M(z)\}>w_M(z)-w_F(z).                    \tag{12.6}
$$
These are conditional risks under the chosen design distribution, not
differences of separately maximized worst-case risks. Uniform honesty
requires $\sup_{R\in\mathcal R}q_R(B_{\rm route})\le\alpha$ with the
entire router inside the probability.

A split-sample path could work as follows: an independent pilot produces
a shape-class confidence set containing the truth with probability
$1-\delta$, and the inference-sample branch is valid at error
$\alpha-\delta$ uniformly over every curve in that selected set,
conditional on the pilot. Then total error is at most $\alpha$.
The present fiducial theory does not yet supply those finite-sample
class-uniform branch guarantees.

## 13. The strongest next theory and experiment targets

The highest-value sequence is:

1. **Full-bracket fiducial variant with frozen trimming.** Keep the
   production cloud and trim index, then use bracket edges instead of
   one completion. Equations (3.3)–(3.5) prove containment and quantify
   area cost. Compare against $C=1$, the frontier hybrid, and M3 on
   prospective slivers, hooked curves, jumps, ties, and imbalance.
   Can it remove lower-edge failures without M3's full projection cost?
   It may still lack uniform honesty.
2. **Exact-cut hybrid as a diagnostic experiment.** Compare (10.1) and
   (10.7), with prespecified budgets, against the empirical frontier.
   Separate coverage gained inside the new margin from failures left
   in the interior. Record post-closure area and both directional events.
   Stage F suggests the residual interior may now be the binding problem.
3. **Finish a publishable interior theorem.** Formalize conditional
   completion negligibility, uniform quantile/depth convergence, and
   the Gaussian sign result. Explicitly use increasing trim resolution
   and $M$. Establish a moving-boundary extension only under stated,
   uniform regularity conditions; it would supply the missing link in
   (10.11), not a guarantee for arbitrary discontinuous curves.
4. **Certify a direct rank confidence set.** Start with a finite FPR grid
   and optimize over the *full* set of monotone probability allocations
   consistent with that grid, including between-grid mass. Use exact
   tests, conservative nuisance maximization, or certified outer
   optimization. Extend by the monotone band rule. Fiducial geometry
   can accelerate search, but must not restrict the accepted set.
5. **Optimize error placement only after controlling risk.** Separate
   lower/upper and regional constraints, using (6.5) as an interior
   benchmark. Retune for multiple alpha values, including .5.
   A width reduction at alpha .05 is not evidence that M3's high-alpha
   projection conservatism has been solved.

For every candidate, report simultaneous coverage, the two directional
failure probabilities (allowing overlap), regional any-failure rates,
conditional-on-miss magnitude, integrated width, and paired WH/KS/M3
ratios. An unconditional 95th percentile of violation magnitude can
equal zero simply because coverage exceeds 95%; it is not evidence that
the remaining misses are small.

The central unresolved question is precise: **can we preserve direct ROC
error cancellation while protecting unobserved probability allocations,
without paying for two entire marginal-CDF rectangles?**
The bracket-area result makes that a credible efficiency target.
The missing-mass result states the protection that cannot be removed.

## 14. Sources and the boundary of the claims

- The sampling ROC limit and its two-class variance decomposition are
  classical; see
  [Hsieh–Turnbull (1996), ROC estimation](https://doi.org/10.1214/aos/1033066197).
  Section 6 gives the derivative needed for this construction.
- [Praestgaard–Wellner (1993)](https://sites.stat.washington.edu/jaw/JAW-papers/jaw-praest-93AP.pdf)
  supplies the weighted empirical-process theorem used in §6.
  Theorem conditions and conclusions on pp. 2056–2057 were checked
  directly; the proof here uses normalized exponential weights.
- [Cui–Hannig, generalized fiducial survival inference](https://arxiv.org/abs/1707.05034)
  provides the related fiducial framework and a functional
  Bernstein–von Mises theorem. It does not prove finite-sample coverage
  of this two-class ROC completion and trim.
- [Myllymäki et al., global envelope tests](https://arxiv.org/abs/1307.0239)
  provides the exchangeable Monte Carlo envelope-test framework.
  Its exactness depends on exchangeability with the observation.
- [Pitt (1982)](https://doi.org/10.1214/aop/1176993872)
  supplies Gaussian association for nonnegative covariances;
  §6.3 derives the ROC directional bound from it.
- Empirical claims come from the
  [boundary follow-up](c_calibration_followup_report.md),
  [Stage F report](hybrid_floor_report.md), and linked experiment results,
  not from these literature theorems.

The gap-area, coverage-algebra, boundary-inversion, and missing-mass
arguments are derived in this document. “Derived here” identifies the
proof's location, not a claim of priority over all existing literature.
