# Exploration implementation validation

Updated 2026-09-08. The [specification](methods_exploration_spec.md) and
[runner instructions](../scripts/methods_exploration/README.md) define the revised
five-track study. The full six-hour screen has **not** been run. No method or
exponent has been selected from the pilot.

The [revised pilot](../data/results/methods_exploration_pilot_20260908/report.md)
completed all five tracks in about 21 seconds, including process startup. It ran
from an archive of the staged tree, excluding pre-existing uncommitted production
floor and theory changes. Source hashes and dependency versions are in its manifest.
The [earlier four-track pilot](../data/results/methods_exploration_pilot_20260907/report.md)
remains a historical artifact and cannot be resumed under the revised source hashes.

| Track | Revised pilot execution |
|---|---|
| Small n | 27 paired datasets, both alphas at production automatic M, six arms, four doubled-M audits |
| Interior | 12 paired datasets at balanced n=100 and 1,000, both alphas, eight C values, two doubled-M audits |
| Likelihood | Four observations with all predictors and all four requested cell resolutions |
| Projection | 936 exact path/candidate containment checks; 32 Monte Carlo outer projections |
| M3 | Boundary calibration, separate training/freeze/evaluation, and 36 held-out observations |

Every small-n pilot dataset had an ineligible fixed window. All n=100 interior
pilot cells were also unexposed and reported `no_exposure`, with no artificial
C* estimate. The n=1,000 interior-sliver observations exercised the active trim
branch at both alphas. These are routing checks, not estimates of population
eligibility, comparative coverage or efficiency. All method-promotion flags are false.

Validation passed **90 tests** in the clean staged-tree archive. The working-tree
check also included the pre-existing production floor tests: **184 tests passed**.
The additional checks target overlap at either endpoint, off-grid window guards,
unchanged fallback, alpha-specific ladder routing, nested eligible trims, absent
interior semantics, production cloud budgets, endpoint atoms, unseen-sliver
sampling probability, and calibration that cannot treat inactive samples as C*
evidence. The existing likelihood, projection, ladder, follow-up, M3, checkpoint
and deadline checks remain in the run. Ruff and staged whitespace checks pass.
The existing thinned-grid parity test emits its expected low-trim-depth warning.

Launch the research screen from the repository root:

```sh
uv run --no-sync python -m scripts.methods_exploration.run \
  --profile screen --out data/results/methods_exploration_screen_20260908 \
  --threads 2
```

The supervisor enforces a six-hour resumable allocation. The interior study targets
2,000 datasets per smaller cell and 1,000 per larger cell, with at least 400 eligible
observations required for a conditional calibration cell. The small-n study targets
500 per size/shape cell. A wall-clock deadline never lowers those targets or
promotes partial results. Completion may require additional run segments, especially
at n=50,000. Fresh decision-gating data remain necessary for any selected exponent.
