# Simulation Spec: Offline Calibration of the Fiducial Band's Trim Exponent ("auto mode")

*2026-08-22. Companion to `stats/fiducial_band_theory.md` (§7, §7.1) and
`stats/next_method_ideas.md` (§5.1–5.2). Purpose: design, done right, of the
one-time study that fixes `trim_exponent="auto"` in
`src/studroc_paper/methods/fiducial_band.py` — and, where the right
parametrization is not yet known, collects enough to decide it rather than
assume it.*

**Decision-first amendment (2026-08-24).** The full Stage A/Stage B design
below is a maximum design, not the first experiment to run. A 27-cell Stage S
screen (500 reps initially, adaptive alpha=.05 top-up to 2,000) first asks the
narrower questions that determine whether an auto map is worth fitting at all:

1. At alpha=.05, is the one-SE lower shape envelope at n=500 at least
   C*=1.15, with at least 4% approximate oracle area gain over C=1?
2. Is a taper toward C=1 visible on three mechanism-distinct shapes over
   n in {100, 500, 5000, 50000}?
3. At fixed minority-class size 500, how strongly do direction and majority
   size change C* on two representative shapes?

The screen shares one cloud across alpha in {.50, .20, .10, .05}, but alpha=.05
is its primary estimand. A negative verdict ends the auto-map effort and keeps
a documented fixed/default rule. A positive verdict justifies a *reduced*
Stage A tailored to the observed failure modes; it is not evidence of coverage.

**OUTCOME (2026-08-30, wording revised 2026-08-31). Stage S ran 2026-08-29
(27 cells, 500–2,000 reps each, parity gate passed; no D6-degenerate
points, and no ladder-saturation flags fired — though the flag only
detects j* = 1, and the t(2)/n=100 crossing at j* = 2 is equally
boundary-pinned; the screening report §5 recommends widening the flag to
j* ≤ 3) and returned its pre-registered STOP: the auto-map effort is
ended.** The body of this spec below is retained unchanged as the
pre-registered record; Stage A and Stage B as designed will not run.
Verdict and data: `stats/c_calibration_screening_report_stage_s.md`,
`data/results/c_calibration_20260829/`. Findings folded into
`stats/fiducial_band_theory.md` §7.2 and `stats/next_method_ideas.md`
(§5 items 1–3, §7 lead entry). Disposition of the open questions and of
what remains worth running:

- **Headline findings.** (i) A genuine small-n validity failure that the C
  coordinate cannot express: t(2)/.95 at n = 100 covers .802 at C = 1,
  α = .05, with misses at the FPR corners and the truth exiting the entire
  untrimmed cloud in 1–2% of reps (*unseen tail mass*, not roughness — the
  designed rough adversary passed at C* = 2.01, so D5's floor conjecture
  as posed stands; the cell's sub-1 C* = 0.084 is a boundary-pinned
  artifact one rung off the ladder top, not a shippable level — screening
  report §2). **C = 1 shipped as the production default 2026-08-30**
  (it measured ≥ .950 at every cell with min(n0, n1) ≥ 500). (ii) The §8
  fallback as written ("keep C = 2.0 for n_eff ≤ 1000") is itself
  falsified: C = 2 measured .917–.940 at α = .05 on t(2) at every n ≥ 500
  and on binormal .95 at n = 50,000. (iii) The envelope over shapes is
  pinned at ~1 by t(2) at n = 500 (lower envelope − 1 SE = 0.967 < C*_min
  = 1.15) and t(2)'s taper is non-monotone (0.08 → 1.17 → 1.49 → 1.07), so
  no member of the §2 shipping family is both safe and useful. The mean
  oracle gain (9.5% at n = 500) is real but reachable only per-shape.
- **Decision dispositions.** D1: moot (no map ships; C remains the manual
  coordinate). D2: the screen's directional probe confirmed round 4 —
  imbalance direction is real on binormal .90 (majority-negative 4500×500
  cuts C* to 1.69 vs 2.2–2.7) — but with no map to fit, no reduction is
  frozen; the finding transfers to the composite-band successor. D3:
  resolved — the α = .05 large-n arm is measured to n = 50,000; the
  smooth-shape taper reaches and crosses 1 (binormal .95: C* = 0.87 ±
  0.12 at n = 50,000, C = 1 coverage .951), the *envelope* is not a taper.
  D4: moot. D5: stands as posed (the trapezoid did not push C* below 1);
  superseded in importance by the validity failure of (i), which the C
  coordinate cannot express. D6: never triggered (allowance
  attribution ≈ 0 everywhere; no unconstrained cells).
- **What remains worth running on this infrastructure** (details, cell
  grids, and decision rules in the FOLLOW-UP RUN PLAN below; ordered by
  value to the final `simulation_spec.md` run):
  1. **Locate the small-n validity boundary** of the C = 1 default and
     produce a conservative, library-relative routing threshold.
  2. **Held-out validation of C = 1** — the designer-bias guard for the
     shipped default, with M3 width economics alongside.
  3. **The composite-band derisk** — posed as a *finite-range* question
     (Theorem 7 forces interior coverage → (1−α)^C for any fixed
     interior C > 1, so no fixed-C composite can be an unrestricted
     method; the corner treatment tested is the untrimmed cloud envelope,
     an empirical widening, not an exact bound).
  4. *(Deferred)* Imbalance with min(n0, n1) > 500 — Stage S already
     found C = 1 at nominal under more severe imbalance (minority 500);
     run only if the final-run guidance turns out to need it.
- **Everything else below is obsolete:** the Stage A fitting grid and
  protocol (§6), Stage B as confirmation of a frozen map (§7), the
  acceptance criteria A1–A3 (§8; A4's escalation clause was exercised),
  the large-n dense arm, and the §10 frozen-map artifact deliverable
  (deliverable 2's report exists as the screening report; no map artifact
  will exist).

**FOLLOW-UP OUTCOME (2026-09-01). Items 1-3 ran; item 1 fired its own
escalation clause and invalidated the surface's mechanism.** Full results:
`stats/c_calibration_followup_report.md`. Headlines:

- **Item 1 escalated as pre-registered.** `t2_99` (df 2, AUC .99) covers
  **.690 at n = 500** and .842 at n = 1000, passing only at n = 2500. The
  spec's own trigger — "if an anchor fails at n = 500, the
  min(n0,n1) >= 500 safety claim is library-limited and theory doc §7.2(c)
  must be amended" — is met. Not an artifact: M3 covers .998-1.000 on the
  same seeds, the ladder is unpinned (min_j 16-17), and misses are
  lower-edge. Cross-family spot checks pass, so the t-family is binding.
- **The (1b) surface is misspecified, not merely imprecise.** Its
  sign constraint `b1 >= 0` (coverage nondecreasing in n) is **false**: at
  fixed shape t(4.69)/.986, coverage runs .993, .947, .903, .823, .847 at
  n = 150..2000. The unsafe set is a curved wedge in (AUC, n) reaching past
  n = 6,000, not a small-n half-space, so no contour `n*(df, AUC)` and no
  n-only routing threshold can express it. Refitting with thin-plate and GP
  smooths improves holdout deviance 46%/59% but none is usable for routing
  (all remain 15-27 points optimistic at the t2_99 anchors): of 95 LHS cells
  exactly one sat in the cliff region, so the design, not the smoother, was
  binding. The trim-grid thinning at K > 2001 was tested as an artifact
  explanation and **rejected** (no discontinuity across the switch).
- **Item 2 passed** (10/10 cells, worst .967) **with a scope limit that
  must travel with the result**: the held-out library tops out at AUC .90
  and never enters the failing region, so it validates C = 1 for
  AUC <= .90 at n = 500, not at n = 500 generally.
- **Item 3 has a finite-range candidate.** Parity holds on all 9 core
  cells. The generated report's "no survivor" is driven entirely by the
  n = 100 cell; on the declared range n >= 500, `b0.02-0.95_C2.5` PASSes
  every cell at -6.8% pooled width. The n = 20,000 sentinels show the
  saving inverting, supporting the Theorem-7 clamp. The implemented
  decision rule pools all core cells and so cannot express a range lower
  bound — a code limitation, not a finding.
- **A localized M3 floor is the strongest new lead.** Replay shows the
  misses concentrate at the *upper* FPR end (peak at 1-FPR ~ .002-.04) plus
  a small left-corner cluster, and M3 covers at 100% of the miss points.
  Unioning with M3 on `FPR in [0, .005] u [.5, 1]` lifts coverage to
  .955-.990 on five failing cells at **+6.4% mean width against +28-46%
  for full M3**, and provably cannot do worse than the C = 1 band. Five
  cells, region selected in-sample — the validation run is the priority.
- **Item 5 is unchanged in necessity and changed in target**: the cutoffs
  to confirm are now the (AUC, n) wedge rule of the follow-up report §5,
  not a contour read off the smooth.
- **Successor plan (2026-09-02):** the next round is specced in
  `stats/hybrid_floor_spec.md` ("Stage F") — an information-gathering
  study, not an arbiter (the final suite and its authors decide methods).
  Study A learns the M3-floor region geometry as a function of
  (n_pos, n_neg, AUC) and records a fixed rule; Study B measures the
  fixed rule's out-of-sample repair, price, and residual-miss geometry
  (the improvement loop's raw material); Study C measures cross-family
  transfer of the repair, the mechanism, and the geometry. Item 5's
  router-cutoff confirmation stays on the books for whenever routing
  language enters guidance. T0's doc amendments (theory doc §7.3, both
  band docstrings, the roster in `next_method_ideas.md` §7) landed
  2026-09-02.

**FOLLOW-UP RUN PLAN (2026-08-30; revised 2026-08-31 after external
review — the revision narrows the composite question to a finite range,
replaces the top-up gate and acceptance rules with coverage-driven
noninferiority inference, fixes the stitched band's monotone closure,
slims the held-out grid, and defers the imbalance item).** Implemented in
`scripts/c_calibration/followup_runs.py` (subcommands `boundary`,
`heldout`, `composite`, `imbalance`, `report`; `all` = boundary, heldout,
composite, report), tests in `tests/test_followup_runs.py`, writing to
`data/results/c_calibration_followup_20260830/<item>/`. All runner items
keep the Stage S machinery: same study seed and per-(stage, cell name,
rep) seeding (new cell names ⇒ fresh streams), same ladder kernel,
production trim-grid rule, M budget `m_budget(n0, .05)` (α grid
{.50, .20, .10, .05}), same evaluation conventions (native grid
t_k = k/n0, pointwise truth check, area = mean band width over grid
points). Cells resume and extend; dry-run budget ≈ 8 idealized
core-saturated hours plus top-ups (boundary ≈ 0.5h classification +
0.5h LHS sweep; heldout ≈ 2.3h; composite ≈ 4.5h incl. the n = 20,000
sentinels).

**Predeclared inference rules (all items).** Per cell, the estimand is
the C = 1 arm's coverage at α = .05. Verdict PASS iff point ≥ .94 AND
Wilson-95% lower bound ≥ .925 (the A1-letter noninferiority bar);
MARGINAL if the Wilson CI still straddles .94 at the replication cap;
FAIL otherwise; the strict ≥ .95 point bar is reported alongside.
**Sequential replication replaces the runner's SE(C*) gate** (that gate
targets the retired auto-map estimand): cells top up in batches while
the Wilson CI straddles .94 — boundary/imbalance 1,000 → 3,000, heldout
2,000 → 4,000, composite 500 → 2,000. All claims are library-relative.

1. **`boundary` — locate the C = 1 small-n validity boundary** (hybrid
   design, revised 2026-08-31 a second time on review feedback: a
   handful of hand-picked probes is neither a surface sample nor an
   envelope bound; the redesign separates the two jobs).
   **(1a) Corner anchors, classification-grade** (11 cells, sequential
   replication): the *achievable-frontier* corners of the suite's
   student_t design space — the literal box corner (df = 1.1, AUC .99)
   does not exist: the DGP mapper caps the location shift at 20, so
   t(1.1) tops out at AUC ≈ .976 and the paper's LHS filters such
   combinations. Anchors: `t11_97` (df 1.1, AUC .97 — heaviest tail at
   its cap) and `t2_99` (heaviest tail reaching AUC .99), both at
   n ∈ {250, 500, 1000, 2500}; plus t2_95 at n ∈
   {150, 250, 350} (Stage S consistency + holdout points for the smooth,
   below). These carry the **decision**: per shape the smallest tested n
   from which all larger tested n PASS, and the global routing threshold
   = the worst such n — assumption-free, conservative, and explicitly
   library-relative (routing at runtime sees only sample sizes, so the
   operational threshold must be worst-case over shapes). **If an
   anchor fails at n = 500, the min(n0,n1) ≥ 500 safety claim of the
   OUTCOME entry is library-limited and theory doc §7.2(c) must be
   amended** — the run's most important possible finding.
   **(1b) LHS surface sweep, estimation-grade** (≈95 cells × 125 reps,
   fixed, no top-up — deliberately many cells / few replicates: the
   information lives at the cell level of the sampling hierarchy, and
   the smooth pools replicates across cells): maximin-free Latin
   hypercube (frozen seed 20260831) over log df ∈ [1.1, 30] ×
   probit-AUC ∈ [.55, .99] × log n ∈ [100, 2500] within the student-t
   family (whose shape space is exactly 2-D by rank invariance),
   unachievable (df, AUC) combinations dropped exactly as the paper's
   LHS drops them. Fit: sign-constrained logistic smooth
   logit(cov) = b0 + b1·log n + b2·log df + b3·probit(AUC) with
   b1, b2 ≥ 0, b3 ≤ 0 (the monotone tail-mass mechanism), bootstrap
   over cells. Products: the **provisional boundary contour n\*(df,
   AUC)** at the .94 bar (point fit plus a conservative 90% bootstrap
   quantile — the quotable number comes from the conservative band,
   never the mean fit, since a mean smooth is anti-conservative for an
   envelope), holdout diagnostics against the (1a) anchors, and
   **per-stratum coverage predictions for the final suite's student_t
   strata** (the suite samples LHS in the same coordinates, so the
   surface maps onto it directly).
   **(1c) Cross-family spot checks, classification-grade** (6 cells):
   the achievable corners of the suite's other corner-curved families —
   `wb05_99` (Weibull shape .5, AUC .99), `gm05_93` (gamma shape .5 at
   its ≈.936 cap), `bo05_99` (beta-opposing α .5, AUC .99) at n ∈
   {100, 250} — verifying the t-family is the binding family before the
   threshold is quoted suite-wide.
   An **M3 arm** runs on every (1a)/(1c) cell (same seeds and data,
   α ∈ {.5, .05}, rep count paired to the fiducial arm's final
   post-top-up count) for the routing target's width economics; its
   coverage column is a regression check (Prop. 12 already guarantees
   it).
2. **`heldout` — designer-bias guard for the shipped C = 1 default**
   (10 cells): all six §5.1 held-out shapes at n = 500; a
   mechanism-diverse sentinel subset (t3_90, bimodal_80_sep15, the LHS
   Weibull member) at n = 5,000; the §7 ties cell (binormal_90 at Q = 20,
   random tie-break, n = 1,000). The n = 1,000 shape rows of the original
   plan were dropped on review (little information between the routing
   boundary and the sentinels). Fresh seed streams. α = .5 reported as
   the conservative-dispersion check; M3 width economics on the n ≤ 1,000
   rows. A FAIL traceable to a held-out shape is a library gap in the
   §7.2(c) claim and escalates as in item 1.
3. **`composite` — derisk a *finite-range* composite band** (9 core
   cells at 500 reps + 2 large-n sentinels, no kernel change). Framing
   forced by theory: Theorem 7 drives interior coverage to (1−α)^C for
   any fixed interior C > 1 (.926/.903/.880 at C = 1.5/2/2.5, α = .05),
   so **no fixed-C composite is a candidate for an unrestricted method**
   — the declared question is whether a candidate exists on a declared
   finite range, with the production form clamping C_int to 1 (or
   tapering) above it. The corner treatment tested is the **untrimmed
   cloud envelope + allowances — an empirical widening, not an exact
   distribution-free bound**; the theorem-capable variant (exact
   M3/Beta-style corner arm) is deferred to the full spec. Per rep, ONE
   fiducial cloud (same kernel seed) is trimmed at the corner exponent
   (1e-4), at C = 1 (full-curve reference/parity arm), and at each
   interior C ∈ {1.5, 2.0, 2.5}; each corner cut ∈ {(.02, .95),
   (.05, .90), (.10, .85)} stitches corner-wide with interior-trimmed
   under **monotone closure of both edges** (running max; the lower-edge
   closure is a valid tightening that leaves the per-rep coverage event
   unchanged — tested) and is scored exactly, removing the
   worst-only-logging bias of the Stage S mid-band estimates. Core
   cells: t2_95 at n ∈ {100, 500, 5000} and 500×4,500; binormal_95 at
   {500, 5000}; kink_80, trapezoid_q10_90, binormal_60 at 500.
   Sentinels: t2_95 and binormal_95 at n = 20,000 (250 reps, reduced
   configs: cut (.05,.90) × C ∈ {1.5, 2}) — outside the candidate range,
   they measure the Theorem-7 erosion direction. Per-rep covered/area
   records are retained (common-random-number pairing) and reuse
   validates the design constants (refuse-to-mix). **Decision rule: a
   config is a candidate iff every core cell PASSes the coverage bar
   above AND its pooled paired width change vs the full-curve C = 1 arm
   is negative** (paired per-rep differences, pooled SE reported); among
   candidates prefer the largest saving, breaking ties toward the
   narrower cut. The parity arm must reproduce the Stage S C = 1
   coverages within 3 combined SEs or the item is void. A surviving
   candidate justifies the full composite-band spec (held-out + large-n
   confirmation, taper/range decision, corner-treatment refinement,
   possibly the exact corner arm); **"no survivor" is evidence against
   this coarse (cut × C) family only**, not against the composite idea.
4. **`imbalance` — DEFERRED** (4 cells specced: binormal_90 and t2_95 at
   5,000×1,500 and 1,500×5,000). Stage S found C = 1 at nominal under
   more severe imbalance (minority 500), so this is unlikely to change
   guidance; excluded from `all`, runnable on demand if the final-run
   guidance turns out to need it.
5. **Cutoff confirmation (FOLLOW-UP, gated on item 1's results — not yet
   implemented).** The (1b) smooth *proposes* routing cutoffs; it must
   not *set* them (a smooth over a 2-D family slice, with smoothing bias
   largest exactly at the contour). Once item 1 is in: read the
   candidate global cutoff(s) from the conservative contour band and the
   (1a) anchors, then run **fresh classification-grade confirmation
   cells** (new cell names ⇒ fresh seed streams; the sequential
   replication rule as in item 1a) at the proposed cutoff and one step
   either side of it, on the worst shapes the surface identifies —
   including any shape the (1c) spot checks flag as binding outside the
   t-family. Routing guidance (the theory-doc §7.2(c) threshold and the
   `fiducial_band` docstring language) is frozen only from these
   confirmation cells; the surface's role in the paper is per-stratum
   prediction, not certification. If the confirmation cells contradict
   the surface (a cutoff cell FAILs where the smooth predicted PASS),
   the smooth's functional form is wrong in the tail — report both,
   keep the anchor-based conservative threshold, and treat the
   discrepancy as a finding about the boundary's shape.
   **Empirical-AUC-conditional cutoffs (added 2026-08-31).** The
   candidate family evaluated here is not restricted to n-only rules:
   an (n, AUC-hat) cutoff — route to M3 iff n_eff < N\*(AUC-hat) — is
   strictly more efficient, since an n-only threshold must be
   worst-cased over the whole library while the boundary contour rises
   steeply in AUC. Ground rules, learned the hard way in rounds 2–4:
   (i) *routing is admissible per-dataset adaptivity only because it
   has a conservative direction* — mis-routing to M3 costs width, never
   coverage — so the rule must be calibrated so mis-routing toward the
   fiducial band is controlled: route on an **upper confidence bound of
   the AUC**, not the point estimate (AUC-hat has SE ≈ .01–.03 at the
   small n where routing matters, and the noise must err toward M3);
   (ii) *selection effects are measured, not assumed away* — a router
   conditions on the data, so fiducial coverage among routed-to-fiducial
   reps is not the unconditional coverage. Evaluation is a **per-rep
   replay join**: every rep is deterministically seeded, so its
   empirical AUC (or any later statistic) is recomputable post hoc
   (`followup_runs.replay_empirical_aucs`) and joins to the same rep's
   stored C = 1 coverage indicator across the item-1 anchors, the ~95
   LHS cells, and the held-out cells — no re-simulation, no schema
   change. Candidate routers are scored by their realized composite
   coverage per cell (routed fraction × conditional coverages),
   worst-cased over df at each (n, AUC) region, since tail weight — the
   stronger driver — is not observable at routing time;
   (iii) *shape-functional triggers* (estimating tail weight from the
   ranks to condition the cutoff further) are optional appendix work:
   conservative-only triggers cannot break validity, but round 4's
   32-functional search found no rank functional that survives out of
   sample — expect no signal, and require held-out confirmation before
   any such trigger enters guidance.

Items 1–2 gate the method-usage guidance the paper must state (where
C = 1 is claimed measured-safe, where M3 is the routed recommendation);
item 3 decides whether the roster's fiducial entry is the plain C = 1
band or a range-limited composite; item 5 converts item 1's provisional
contour into frozen routing guidance. Findings fold into
`fiducial_band_theory.md` §7.2 and `next_method_ideas.md` §5/§7 as
amendments to the same entries the OUTCOME block names.

---

## 1. Objective and non-objectives

**Objective.** Produce a frozen, provenance-stamped map from
$(n_0, n_1, \alpha)$ to the trim level of the fiducial band, such that:

- (Validity) realized coverage $\ge 1-\alpha$ (within stated Monte Carlo
  allowance) for **every** shape in an adversarial library, at every grid
  point of $(n, \alpha)$, including held-out shapes not used in fitting;
- (Efficiency) it recovers a meaningful fraction of the width the identity
  map ($C=1$) leaves on the table (target: ≥ half of the 9–13% area gap at
  $\alpha=.05$ where the gap exists);
- (Posterity) the study sharpens the open scientific questions entangled
  with the map: the finite-range taper rate, imbalance reduction, and
  $\alpha$-drift. The production coordinate and asymptote are fixed by
  Theorem 7 rather than selected by simulation.

**Non-objectives.** Per-dataset (shape-adaptive) calibration — ruled out by
the measured plug-in bias (`m2_report.md` §1c); fixing the residual
central-$\alpha$ shape spread (a level map cannot; theory doc §7); anything
about the CP allowances or the fiducial construction itself (frozen as
shipped).

**Why offline calibration is legitimate.** Coverage of the band depends on
the DGP only through (curve shape, $n_0$, $n_1$) — rank invariance,
`fiducial_band_theory.md` Prop. 2. A map calibrated over a shape library
therefore transfers exactly to any real dataset whose shape is inside (or
dominated by) the library. The only transfer risk is shape coverage of the
library, which the design addresses with adversarial + held-out shapes and
a lower-envelope aggregation.

---

## 2. The auto-mode plan (what we intend to ship)

Recorded here for posterity, *before* the study — deviations must be
justified in the final report.

**Intended behavior.** `fiducial_band(..., trim_exponent="auto")` resolves,
at call time, to a value of the trim coordinate from the frozen map:

```
C_auto(n0, n1, alpha) = clamp( 1 + delta0(alpha) * (n_eff / 500)^(-gamma),
                               1.0,  C_max(alpha) )
```

with constants `(delta0(alpha) on the alpha grid, gamma, C_max)` and the
definition of `n_eff` fixed by this study, stored alongside the code with
the study's git hash. Behavior contracts:

1. **Floor at C = 1** always (conjectured universally safe; the study tests
   the conjecture on deliberately rough truths — see D5).
2. **Extrapolation beyond the calibrated n-grid** follows the fitted taper
   (monotone decreasing toward 1), which errs in the safe direction;
   extrapolation in $\alpha$ outside $[.01, .5]$ falls back to C = 1 with a
   warning.
3. Manual `trim_exponent=<float>` remains available and documented;
   library default switches from `2.0` to `"auto"` only if the acceptance
   criteria of §8 are met.
4. The map is calibrated **under exactly the production trim-grid rule**
   (§5.3) — we calibrate what ships, never a variant of it.

**Open parametrization questions the study must decide (not assume):**

- **D1 (coordinate, resolved by design).** Ship and fit $C$. It is the
  production control, and Theorem 7 fixes its asymptote at 1. Continue to
  report $\alpha_{\mathrm{eff}}$ and realized local level $\ell$ as
  diagnostics. Ranking these coordinates by relative dispersion is not a
  legitimate selection rule because that ranking changes under nonlinear
  reparameterization; moreover $\ell$ depends on the finite-cloud budget.
- **D2 (imbalance reduction).** Is `n_eff` $\min(n_0,n_1)$, the harmonic
  mean, or is a genuinely 2-D map needed? Decision rule: fit each candidate
  on the prevalence arm (§5.2). Compare the exponent predicted from balanced
  cells directly with each imbalance cell's measured $C^*$ threshold; accept
  a 1-D reduction only if it does not overpredict any threshold after a
  one-bootstrap-SE margin. Otherwise a 2-D interpolation table is an explicit
  blocker, not an artifact the current resolver can silently ship.
- **D3 (finite-range taper) — asymptote resolved before this study ran**
  (round 3,
  `m3m4_report.md` §6): at central $\alpha$ the decay is real and
  power-like, $C^\*(n)-1 \approx 1.26\,(n/500)^{-0.32}$ measured to
  $n = 20{,}000$ at fixed shape; H1's plateau at 2 is excluded by 4.2 SE.
  What this study must still settle: the **$\alpha = .05$ arm at
  $n \ge 10^4$** (never measured; needs $M \approx 10$–12k), whether
  $\gamma$ is shared across $\alpha$ and shape, and how pure-power,
  power-plus-plateau, and log-decay diagnostics compare on the fuller grid.
  The large-n extension of §5.2 is therefore re-prioritized: the
  $\alpha=.05$ rows are its main payload, the central-$\alpha$ rows are
  confirmation. The shipping family must tend to C=1. Plateau and log-decay
  fits are misspecification diagnostics, not candidates with incompatible
  asymptotes; compare their finite-range prediction errors without allowing
  that comparison to override the limit.
- **D4 ($\alpha$-drift).** Is a separable form
  $\delta_0(\alpha)\cdot f(n)$ adequate, or does the $(n,\alpha)$ surface
  need a joint fit? Decision rule: separable is accepted if its RMS residual
  is no larger than the RMS bootstrap SE of the shape setting the envelope.
  Rejection blocks freezing until a constrained joint surface is specified or
  the conservative loss is explicitly accepted.
- **D5 (shape aggregation and the floor conjecture).** Which quantile of
  the shape library defines the envelope (min, or 10th percentile minus an
  SE margin), and does any legitimately rough truth (trapezoid estimand)
  push $C^\* < 1$? If yes, the floor moves below 1 and §2.1 changes —
  this would be a major finding; the design must be able to detect it.
- **D6 (degenerate shapes).** For plateau-dominated shapes whose coverage
  is carried by the CP allowance, $C^\*$ may be effectively unbounded
  (coverage never dips below $1-\alpha$ on the ladder). Rule: such cells
  impose **no constraint** on the envelope and are reported separately;
  they must not inflate the fitted map.

---

## 3. Estimands and estimators

For each cell (shape $R$, $n_0$, $n_1$) and each rep, compute the full
**coverage-vs-depth profile** using the ladder identity
(`m2_report.md` §0): one pass over the cloud yields, for every trim depth
$j$, whether the allowance-augmented band at depth $j$ covers the truth.
(The CP allowance depends on $j$ through its level $j/(M+1)$, so the
profile is evaluated explicitly per ladder point, as in `rep_profile`.)
Reps are **shared across the whole $\alpha$ grid** — $\alpha$ only selects
a point on the ladder — which is what makes this study affordable.

Per cell, estimate:

- $\mathrm{cov}(j)$ = mean over reps of the depth-$j$ coverage indicator;
- $j^\*(\alpha) = \max\{j : \mathrm{cov}(j) \ge 1-\alpha\}$;
- the three candidate coordinates at calibration:
  $\alpha^\*_{\mathrm{eff}}$ = mean fraction of draws with depth $< j^\*$;
  $C^\* = \log(1-\alpha^\*_{\mathrm{eff}})/\log(1-\alpha)$;
  $\ell^\* = j^\*/(M+1)$;
- uncertainty: nonparametric bootstrap over reps (1,000 resamples) for all
  three, reported as 95% intervals;
- diagnostics: realized $j^\*$ range (saturation flag if the ladder is
  pinned at $j=1$ anywhere relevant — such $(cell, \alpha)$ points are
  **excluded from fitting** and the cell re-run at larger $M$), the
  allowance-attribution (fraction of reps where the CP allowance is what
  covers, for D6), miss direction/location/depth at the calibrated point.

Also record, per cell, coverage and mean area under the three **reference
maps** — $C=1$, $C=2$, and the *provisional* auto formula (updated to the
round-3 fixed-shape fit: $\gamma = 0.32$; envelope $\delta_0(.05) \approx
0.8$, pending this study's shape sweep — note the round-3 family
experiment's noisy $\alpha=.05$ column puts t(2)'s $C^\*$ anywhere in
1.2–1.8, which widens the envelope uncertainty and raises the value of
this study's §4 precision machinery) — so the final report can state the
realized width recovery and regret of the shipped map against these
baselines without re-running anything.

**Convention note (added after round 3):** the CP upper allowance at
$k = 0$ is part of the calibrated procedure and **stays** — pinning
$U(0) = 0$ is invalid distribution-free (`fiducial_band_theory.md`
Corollary 9.3). The map is calibrated with the allowance active at every
grid point, exactly as production behaves.

---

## 4. Precision targets (and why)

Coverage sensitivity to the trim coordinate is
$\partial \mathrm{cov}/\partial C \approx -\alpha/C^\*$ (theory doc §7.1):
$\approx$ 2.2pp per unit $C$ at $\alpha=.05$, $\approx$ 25pp per unit at
$\alpha=.5$. Map errors of $\pm 0.15$ in $C$ therefore cost $\le$ 0.4pp at
$\alpha=.05$ and $\le$ 4pp at $\alpha=.5$. Targets:

- SE($C^\*$) $\le 0.15$ per fitted cell **at $\alpha \le .2$** — achieved
  not by brute reps at $\alpha=.05$ (where the coverage curve is flat in
  $j$ and direct crossing estimation is noisy) but through the ladder: the
  crossing is the $\alpha$-quantile of the truth-depth distribution, whose
  order-statistic SE at $R$ reps is well-behaved; 1,000 reps suffices
  (verified in-flight by the bootstrap CIs; cells failing the target get
  topped up to 2,000 reps before fitting).
- Confirmation arm (frozen map): coverage SE $\le$ 0.7pp $\Rightarrow$
  2,000 reps on the confirmation cells at $\alpha=.05$; 1,000 elsewhere.

---

## 5. Design

### 5.1 Shape library

By rank invariance, cells are curve shapes. Split **fitting** vs
**held-out** *before* running (held-out shapes touch nothing until §7
Stage B).

Fitting set (10):
1–5. binormal, AUC ∈ {.60, .75, .90, .95, .99}
6. heteroscedastic Gaussian shape, AUC .90, σ-ratio 3 (asymmetric curve)
7. t(2) shape, AUC .95 (least-smooth tested; historically envelope-setting)
8. kink truth (vertical to TPR .6 by FPR $2/n_0$, then linear; AUC ≈ .80)
9. bimodal-negative, AUC .90 (mid-curve inflection + plateau)
10. **trapezoid truth**: the exact estimand of scores quantized to Q = 10
    equal-negative-mass levels at AUC .90 — a *legitimately rough* truth,
    included specifically to test the $C^\* \ge 1$ floor conjecture (D5).

Held-out set (6): binormal .85; t(3) shape .90; bimodal .80 with different
separation; logit-normal-type shape; **two LHS-sampled shapes** drawn from
the paper's existing DGP mapper with a fixed seed (guards against designer
bias in the hand-picked library).

### 5.2 Sample-size and imbalance grid

- Balanced ladder (core): $n$ per class ∈
  {25, 50, 100, 250, 500, 1000, 2500, 5000} × all 10 fitting shapes.
- Large-$n$ extension (for D3, the asymptote): $n$ per class ∈
  {12500, 25000, 50000} × 3 envelope-relevant shapes only
  (binormal .95, t(2) .95, kink), $\alpha \ge .05$ arm only.
- Imbalance arm (for D2): $n_0{:}n_1$ ∈ {9:1, 3:1, 1:3, 1:9} at
  $n_{\mathrm{total}}$ ∈ {1000, 5000, 20000} × 3 shapes (binormal .90,
  t(2) .95, bimodal .90).

### 5.3 Grid, M, and the production trim-grid rule

- Evaluation grid: native $t_k = k/n_0$, staircase-upper convention —
  identical to production and to the eval framework.
- **Production trim-grid rule (adopted here and in the shipped code
  simultaneously):** for $K = n_0+1 > 2001$, the min-p trim is computed on
  the thinned grid {every $\lceil K/1000 \rceil$-th point} ∪ {first and
  last 50 points}, with the band evaluated on the full grid. (Validated
  leak-free in `m2_report.md` P4; adopted to keep $M$ affordable at large
  $n$.) The calibration is run under this exact rule — never calibrate a
  procedure other than the one that ships.
- $M$ per cell: the budget rule $M \ge 5/\ell(K_{\mathrm{trim}},
  \alpha_{\min})$ with a ×2 safety factor, where $\alpha_{\min}$ is the
  smallest $\alpha$ fitted at that cell ($.01$ for $n \le 5000$, $.05$
  above). Realized $j^\*$ logged; saturated points excluded and re-run.
- dtype float32 for clouds above $4\times10^7$ entries (as in production);
  the ladder ranks are tie-inclusive, so float32 rounding errs inclusive.

### 5.4 $\alpha$ grid

{.50, .30, .20, .10, .05, .02, .01} for $n \le 5000$;
{.50, .20, .10, .05} for the large-$n$ extension. All from the same reps.

### 5.5 Randomness and reproducibility

Deterministic seed per (cell name, stage) as in the existing harnesses;
package versions and git hash recorded in every output JSON; raw per-rep
profiles retained (not just aggregates) so the fit can be redone under a
different aggregation without re-simulation. New files only; nothing
existing overwritten. Runner: `uv run --project . python`; ≤ 2 concurrent
processes on laptop.

### 5.6 Implementation: the Rust core (added 2026-08-23)

The study runs on the **`fiducial_core` Rust extension** (rayon-threaded,
xoshiro256++ per-draw seeding; measured 12–17× faster than the
Python/torch band at n = 500–5,000, ≈ 0.56 s for a full band at
n₀ = 5,000, M = 10,000), wrapped by
`src/studroc_paper/methods/fiducial_band_rs.py`, which keeps the
statistical envelope (tie-breaking, khat counts, CP allowances, output
grid) in Python and identical to `fiducial_band`.

Prerequisites before any Stage S or Stage A calibration cell runs:

1. **Ladder export.** `fiducial_core` currently exposes only the trimmed
   tube. The study needs a `ladder_profile` entry point: given the label
   sequence, M, seed, a reference-curve vector on the grid, and the khat
   counts, return (a) the per-depth coverage indicators of the reference
   *with the CP allowance evaluated at each ladder depth's own level
   j/(M+1)*, (b) the draw-depth distribution, and (c) pointwise order
   statistics at requested depths — computed streaming in Rust (no
   M × K cloud crossing the FFI boundary; at n = 50,000, M ≈ 15k the
   cloud is ~3 GB in f32 and must not be materialized in Python).
2. **Parity gate.** The Rust and Python paths use different RNGs, so
   equivalence is *statistical, not bitwise*: before Stage S, reproduce
   the round-2 validation cell (binormal .95, n = 500/500, 150+ reps)
   with the Rust path at both trim exponents and both α ∈ {.05, .5};
   coverage and area must agree with the published Python-path numbers
   within Monte Carlo error, and the ladder-profile path must agree with
   `fiducial_band_rs` itself exactly (same seed, same j → same band).
   Record the gate's results in the study report.
3. **The map is calibrated under, and ships with, the Rust
   implementation** — "calibrate what ships" (§5.3) now binds to
   `fiducial_band_rs` as the production entry point for the suite.

---

## 6. Fitting protocol (Stage A)

1. Estimate $C^*(n_{\mathrm{eff}}, \alpha)$ on the fitting shapes, with the
   imbalance reduction checked by D2. Other coordinate summaries are
   descriptive only.
2. Fit the power-to-C=1 taper to the per-shape surpluses
   $\delta(n) = C^*(n) - 1$. Report plateau and log-decay fits only as
   finite-range misspecification diagnostics.
3. **Aggregate across shapes to the envelope** (D5): default = the
   pointwise minimum over fitting shapes of the calibrated coordinate,
   minus one bootstrap SE; D6-degenerate cells excluded. If the minimum is
   dominated everywhere by a single shape (expected: t(2) or trapezoid),
   report that and consider whether the library needs an even rougher
   member before freezing.
4. Fit amplitudes so the shipping curve lies at or below every observed
   min-minus-SE envelope point, including the large-n arm. Then freeze the
   constants, reduction, and
   fallback boundaries ($C_{\max}(\alpha)$ from the small-$n$ end of the
   envelope; $\alpha$-range guards). D2 or D4 rejection and any $C^*<1$
   finding are blockers carried in the candidate artifact and fit report.

## 7. Confirmation protocol (Stage B — frozen map, fresh seeds)

Run the frozen auto map, plus $C=1$ and $C=2$ reference arms, on:

- all 6 held-out shapes × $n \in \{100, 1000, 5000\}$ × full $\alpha$ grid,
  2,000 reps at $\alpha=.05$ (1,000 elsewhere);
- the 3 large-$n$ shapes at $n = 25000$;
- two imbalance cells not used in fitting (e.g. 1:3 at $n_{\mathrm{total}}
  = 2000$);
- one ties cell (Q = 20, random tie-break) as a regression check.

## 8. Acceptance criteria (pre-registered)

Auto mode ships as the default iff, on the confirmation runs:

- **A1 (validity):** coverage $\ge 1-\alpha - 1.0$pp point estimate AND
  $\ge 1-\alpha - 2.5$pp at the lower 95% CI bound, for every confirmation
  cell at $\alpha \le .10$.
- **A2 (efficiency):** mean area $\le$ the $C=1$ arm's area in every cell,
  and $\ge$ 4% below it (i.e. at least half the known 9–13% gap) averaged
  over cells with $n \le 1000$ where the gap lives.
- **A3 (no regret):** no confirmation cell where auto is *both* wider and
  lower-coverage than $C=2$ (would indicate a fitting artifact).
- **A4 (sanity of the science):** the D1–D6 decisions are internally
  consistent and the candidate has no unresolved D2/D4 blocker. If D5 finds
  $C^\* < 1$ anywhere, ship C=1 as default and escalate — the floor conjecture
  is falsified and the theory doc needs revision.

If A1 fails only at cells traceable to a specific library gap, extend the
library and refit (one iteration allowed, documented); if A2 fails, auto
mode is not worth shipping — keep `C=2.0` for $n_{\mathrm{eff}} \le 1000$
and `C=1.0` above as the documented default, which is the crude but
measured-safe fallback.

## 9. Compute budget and stop rules

The executable dry-run is the budget authority. As of 2026-08-24 it reports:
Stage S, 27 cells / 13,500 initial reps / about 6 idealized core-saturated
hours; full Stage A, 125 cells / 250,000 baseline reps / about 181 such hours;
Stage B, 24 cells / 88,000 reps / about 89 such hours. Adaptive top-ups can
increase these totals. The former 4–8 laptop-hour estimate was not consistent
with the checked-in 2,000-rep design or the $\alpha=.01$ cloud budgets and is
retired.

Therefore Stage S is mandatory and the full Stage A grid is only a maximum
design. After a positive screen, retain only arms tied to unresolved routing
decisions in `screening_report.md`. If the large-n screen finds no useful
margin, validate a C=1 clamp rather than estimating a dense tail surface. If
imbalance has a resolved directional effect, retain that arm and plan a 2-D
rule; otherwise test the min-size reduction first. Expand the full $\alpha$
grid only after the primary $\alpha=.05$ usefulness gate passes.

**Round-4 note on D2 (imbalance).** M3's worst-case-level probe
(`r4_report.md` §4) found class imbalance, not shape, to be the binding
coordinate of its level map (the 9:1 cell needs a 1.5–1.7× smaller
nominal level than any balanced cell), and the one shape-functional rule
that fit well exploded on the same cell. Treat this as prior evidence
*against* a 1-D `n_eff` reduction: the imbalance arm of §5.2 is upgraded
to sweep $n_{\mathrm{total}} \in \{1000, 5000, 20000\}$ at ratios
{9:1, 3:1, 1:3, 1:9}, and D2's acceptance rule stands as written.

## 10. Deliverables

1. `data/results/c_calibration/` — per-cell raw profile JSONs + the frozen
   map artifact (constants, coordinate, provenance, git hash).
2. `stats/c_calibration_report.md` — the D1–D6 decisions with evidence,
   the fitted surfaces with CIs, confirmation tables against A1–A4,
   and the final shipped formula stated in one box.
3. Code: the `fiducial_core::ladder_profile` Rust entry point (§5.6.1)
   with its parity gate; the study runner in `stats/experiments/` driving
   it; the frozen map wired into **both** `fiducial_band` and
   `fiducial_band_rs` (`trim_exponent="auto"`), with unit tests: correct
   constant lookup, clamping, warning behavior, and a golden-value test
   against the frozen artifact.
4. Updates: `fiducial_band_theory.md` §7/§7.1 (the finite-range D3 evidence
   and measured $\gamma$); `next_method_ideas.md` §5 (uncertainty items 1–2
   resolved or sharpened).

## 11. Risks and their mitigations (dotted Is)

| Risk | Mitigation |
|---|---|
| Saturated ladders contaminating $C^\*$ at small $\alpha$/large $K$ | $j^\*$ logged per point; saturated points excluded and re-run at larger M (budget rule ×2 upfront) |
| Flat coverage-in-$j$ at $\alpha=.05$ makes crossings noisy | Estimate via truth-depth quantiles on the ladder, not via direct coverage crossings; bootstrap CIs gate the fit (top-up rule §4) |
| Shape library misses a lower-envelope shape | Adversarial members (t(2), kink, trapezoid) + LHS held-outs + one documented refit iteration allowed |
| Calibrating a different procedure than ships | Production trim-grid rule fixed in §5.3 and adopted in code in the same commit as the map |
| Degenerate plateau shapes distorting the envelope | D6 exclusion rule; allowance-attribution logged |
| Misspecifying the finite-range taper | Leave-one-$n$-out diagnostics; fit constrained below observed envelope; held-out-shape confirmation; A3 no-regret check |
| Study results misread as guarantees | Report language fixed in advance: the map is *empirically calibrated over a shape library*; the provable fallback remains C=1 (asymptotic) / M3 (finite-sample), per the theory doc |
