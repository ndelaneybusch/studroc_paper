# Bounded method exploration: interior schedules and certified rank geometry

Date: 2026-09-07. This is an information-gathering screen, not the decision-gating
simulation or a production method change. Run from the repository root with
`uv run --no-sync python -m scripts.methods_exploration.run --profile screen --out <directory>`.
Use `--profile pilot` for an end-to-end implementation check. The screen has a
three-hour wall-clock cap, enforced by a supervising process; unfinished work is
reported as incomplete. Each track checkpoints observations and its decisions.

## Context and interpretation

Read alongside [the theory](fiducial_band_theory.md), especially Theorem 7 and
§§12.1, 12.2, 12.6–12.7; [next methods](next_method_ideas.md);
[Stage S](c_calibration_screening_report_stage_s.md);
[the calibration spec](c_calibration_spec.md);
[the follow-up](c_calibration_followup_report.md); and
[Stage F](hybrid_floor_report.md). Existing uncommitted production/theory work is
not part of this exploration's commit. The exploratory exact-floor formulas are
specified here and do not depend on those uncommitted modules.

The floor's regional guarantee is not whole-band honesty. Theorem 7 applies to
regular fixed interiors with increasing cloud size and a refining trim grid,
not to a fixed cloud cap, a thinned grid, or arbitrary slivers. M3 is the exact
reference. Its numerical certificate is the existing conservative floating-point
non-crossing DP, not an exact-rational certificate.

Two inversion sets must never be confused: inversion over a finite library is an
**inner, optimistic efficiency diagnostic**; an outer box enclosure begins with
the entire ordered knot cube and keeps every unresolved box. At unreported FPRs,
lower edges extend from the left and upper edges from the right. All reported
integrated widths use this conservative step extension, including the comparators.

## Shared protocol

Use deterministic, independently keyed streams for training, evaluation, clouds,
and Monte Carlo nulls. Pair all bands on each label path. Save the config, source
hashes, environment versions, seeds, per-replicate records, elapsed times, and
termination status. Resume only an identical configuration and source fingerprint.
Training never uses held-out observations. A deadline never converts an incomplete
screen into a pass. Pilot runs never authorize promotion.

Report whole-grid coverage (which implies coverage of the conservative continuum
extension), lower/upper/both failures, fixed-interval failures, maximum miss depth,
area, left/right tail widths, paired ratios and standard errors. Pointwise failures
are stored as run-length intervals on the reporting grid so they can be aggregated
without losing location information. Exact enumerations use probability weights;
Monte Carlo summaries use replicate uncertainty. Tail windows are [0,.02] and
[.95,1]. Width selection must not trade away either tail unnoticed.

No screen claims to establish library noninferiority from failure to reject a
deficit. Coverage intervals and censored exponent crossings are explicit.

## 4. Interior exponent after the exact floor

Balanced-equivalent size n is 100, 500, 5,000, 50,000. Class fractions are 1:1,
1:9 and 9:1 at total size 2n, so both directions have the same total sample
budget. Fit the schedule in n_eff = 2 n0 n1 / (n0+n1), equal to n under balance.
Alpha is .05 and .5. Shapes are the Stage S trio (binormal .95, t2 .95, kink .80),
a boundary sliver, and an interior-translated sliver with 20% positive mass
remaining beyond FPR .85 to prevent the saturated-run floor from automatically
covering the interior adversary. Rare masses scale with n1;
keep the exact piecewise-linear inverse, including plateaus. These are deliberate
uniformity stress sequences, not regular fixed-shape asymptotic examples.

Reuse the ladder kernel and follow-up's shared-cloud/edge/allowance construction.
The corner arm is full-grid C=1 plus a localized M3 hull. Left cutoff is the
smallest inclusive k with Binomial(M,(1-k/n0)^n0) survival at j-1 <= .001;
right start is n0-ceil(n0 BetaInv(.975,K+1,n0-K)), with K the trailing negative
run. The floor uses alpha and equal class split. Its region is frozen from the
full-grid C=1 depth within the replicate for every interior C. This removes a
second moving tuning knob and preserves nesting across the exponent ladder.

Trim the interior using **all native columns in the fixed window [.02,.95]**.
Use that same cloud seed for both profiles. Outside the window retain the corner
arm; inside use the interior tube with its depth-specific corner allowances;
then reapply the frozen M3 hull and widen to monotonicity. Include raw full-grid
C=1, exact-floored full-grid C=1, and fixed-interior C=1 as distinct baselines.
The last isolates the change of trim domain from exponent effects.

C ladder: 1, 1.25, 1.5, 2, 2.5, 3.5, 5, 8. Primary statistic: max-minus-min
shape C* at each size, alpha and class ratio; C* is the largest nested-grid C
whose estimated simultaneous coverage reaches nominal. Report interval-inverted
crossings and right/left censoring; never fit a value at the ladder endpoint as
an observed crossing. Use paired bootstrap shape-spread uncertainty, including the full C-grid
bracket width in the upper spread bound.

Only if every crossing is resolved and the 95% upper spread bound is <= .5,
fit C(n_eff,alpha)=max(1,1+a_alpha (n_eff/100)^(-b_alpha)), b in [.1,1].
Fit below the conservative per-cell crossing envelope; maximize mean proposed C
on the design sizes. Freeze the coefficients to JSON before any independent
confirmation. Otherwise propose a fixed C on 500<=n_eff<=5,000 and C=1 outside;
choose only values with conservative coverage support in all eligible cells.
An unsupported fallback is C=1, not permission to interpolate across failures.
A realized depth below three blocks freezing until a larger-cloud confirmation.
Small screens will often be inconclusive; that is useful precision information.
The fixed and decaying candidates still require fresh coverage confirmation in
the week-long gate, including interior slivers. No universal claim follows.

Default replication is 64 per cell, with 24 at n>=5,000, M=2,000 (4,000 at large
n); these are rejection screens. In particular, 24 replicates cannot establish a
95% lower coverage bound of .95 even with no misses, so a successful-looking
large-n sentinel demands further replication before a schedule can be supported.
The bounded run must distinguish this precision limitation from heterogeneity. A same-data doubled-M audit on the first balanced binormal replicate at every size
reports depth resolution and cloud sensitivity. A hard 75-minute allocation
prevents large-n clouds from swallowing all the other questions. Interleave sizes,
ratios, shapes and alphas to make partial output interpretable.

## 5. Likelihood inversion: measure losses before building a solver

At n0=n1 in {5,10,20}, compare three normalized predictors: uniform over paths;
a fixed equal-weight mixture of piecewise-linear approximations to binormal rank
laws on AUC {.5,.6,.7,.8,.9,.95,.99}; and a sequential count-respecting label
predictor fitted on independent mixture draws. Its next-label probability is a
Beta-smoothed table indexed by prefix counts, with forced labels when a class is
exhausted. Fitting the complete evaluation path is forbidden. The mixture masses
are computed by the cell likelihood DP, never by fitting the observed path.

The candidate efficiency library is a monotone lattice on three fixed interior
knots, augmented by smooth and jump/sliver truths. Refinement uses 4,8,16,32 cells
plus every candidate breakpoint and atom, so nested partitions describe the same
candidate. Exact within-cell PL likelihood uses denominator (a+b)!; upper/lower
bounds use a!b! and pure transitions respectively. Probability computations in
this diagnostic use float64 and are checked against independent Fraction DPs.
They do not perform certified exclusions of the unrestricted domain.

Report separately:

1. Predictive penalty: log(p_true/q), its mean/SE, and rejection under each q;
   include oracle q=p_true as a diagnostic, never a deployable numerator.
2. Cell loss: actual upper-minus-lower likelihood gap, its ratio to alpha q,
   the capped union bound n0 n1 sum a_h b_h, and the extra accepted candidates
   and width relative to exact PL likelihood inversion.
3. Projection slack: the fraction of rejected library curves lying completely
   inside the hull of accepted curves, and the hull width. This has no canonical
   additive width decomposition; do not pretend these three losses add in area.

Compare optimistic inner-hull width to M3 and full-grid C=1 on the same reporting
grid. An empty finite set is recorded as empty (zero inner width), never mistaken
for an honest narrow band. Promotion requires nonempty sets on >=99% of smooth
n=20 observations, a paired 95% upper width ratio < .97 versus M3, and upper-cell
hull width at 32 cells also < M3. If it passes, enable a bounded rational box
prototype; otherwise stop. A pass only justifies solver work: a finite library
can miss an accepted curve and understates the true width.

## 6. Certified outer projection of rank tests

Use a statistic family that really is monotone under quantile coupling. For a
label path let K_i be positives preceding negative i. Three fixed nonnegative
weight vectors measure early, whole-path and late positive counts. For each
statistic use both inclusive tails with total Bonferroni budget alpha/6. Also
report a whole-path-only alpha/2 ablation. This sacrifices some power but gives
a tractable, geometry-sensitive certificate; no claim that fiducial trim is
monotone is required.

For a knot box [ell,u], construct the smallest and largest step CDF completions:
lower uses the preceding ell, upper the following u, with possible endpoint
atoms. Every admissible truth lies between these CDFs. Shared uniforms then
order the positive placements and hence every weighted K statistic. Therefore
an upper-tail p-value is bounded above by its value at the upper CDF, and a
lower-tail p-value by its value at the lower CDF. Reject a box only when one of
these **upper bounds** is <= its allocated level. Monotonicity of the statistic,
not a claim about arbitrary bracket endpoints, justifies this step.

At total sizes 4,6,8 enumerate every rank path, calculate exact rational step-law
probabilities by §12.7, and use exact rational tail comparisons. Verify containment
of the exact confidence set for a dense PL lattice and diagonal, binormal
(rational PL approximation), jump and sliver cases, for every observed path.
Also compute exactly weighted coverage and M3 width. The outer search starts with
the full ordered cube at knots .25,.5,.75; a midpoint split propagates monotone
constraints. Stop after the declared box budget and retain all unresolved boxes.
Report the staircase projection-resolution cost explicitly: coarse knots can
make a sound certificate useless for width.

Then at n0=n1=25 use the same bounding argument with common random numbers and
p=(1+#inclusive extremes)/(B+1), B=199, fresh per observation and fixed through
its entire box search. No Monte Carlo CI is needed to certify this *randomized
Monte Carlo test's* inversion: each coupled simulated statistic is bounded
pathwise. This outer band contains the Monte Carlo confidence set, not necessarily
the distinct exact-null confidence set. Report rejection resolution, number of
boxes, retained volume proxy, widths, and compute. Stopping/adaptive subdivision
changes width only. No null simulation seed is selected for narrowness.

## 7. M3 deterministic boundary optimization

Use a three-parameter family: interpolation eta between ELL and KS boundaries;
a symmetric iterated-log tail modifier theta; and a class allocation exponent s,
rho=n0^(-s)/(n0^(-s)+n1^(-s)). For each marginal, vary a common width scale and
calibrate actual joint non-crossing probability to its class target plus 1e-9.
ELL/KS interpolation occurs in boundary values before monotone tightening, and
calibration applies to that final event. Include the unmodified M3 boundary and
sample-size split as anchors. No sample-dependent selection occurs at deployment.

The iterated-log modifier is an exploratory family inspired by the tail/center
tradeoff in [Dümbgen–Wellner](https://sites.stat.washington.edu/jaw/JAW-papers/jaw-duembgen-aos.2023.pdf),
not an implementation of their theorem. The criterion of directly optimizing
width follows [Frey](https://doi.org/10.1016/j.jspi.2007.12.001); we optimize ROC
projection width within this small family, not Frey's unrestricted CDF optimum.

Fit per (n0,n1,alpha), sizes 50/50,100/100,50/450,450/50,500/500;
alpha .05,.2,.5. Training smooth shapes: binormal AUC .55,.65,.8,.95. Held out:
diagonal, binormal .6/.7/.9, heteroscedastic .7, t2 .7/.95, and slivers.
Use paired training mean ROC area with each training tail mean constrained to
<=1.00 times M3; retain M3 when no candidate qualifies. Freeze boundaries and
parameters before generating held-out observations. Held-out eligibility requires
>=3% mean paired area saving with a 95% upper ratio <=.97, and each shape/direction's
95% upper tail ratio <=1.01. Gate separately per count/alpha design, averaging
area over fixed held-out shapes within replication blocks for its uncertainty.
Use baseline M3 in unsuccessful or unlisted regimes; one unsuccessful size must
not erase a useful independently certified boundary at another size. Report
individual cells, never hide a tail loss in
a pooled mean. A numerical coverage failure invalidates the boundary immediately.
Central-alpha overcoverage is measured, not fitted away.

## Deliverables and decision discipline

The implementation has separate `interior`, `likelihood`, `projection`, and `m3`
tracks plus one budgeted runner and summary report. Output records contain enough
information to re-evaluate all gates without regenerating clouds. The executable
screen may stop early; the report lists requested/completed work and does not
freeze an eligible candidate from a partial design. Pilot artifacts are retained
as engineering evidence only. Only the frozen candidate files from complete
screens are inputs to the eventual independent decision-gating simulation.
