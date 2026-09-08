# Exploration implementation validation

The [specification](methods_exploration_spec.md) and
[runner instructions](../scripts/methods_exploration/README.md) are ready for the
bounded research run. The full three-hour screen has **not** been run. No method
or exponent has been selected from the pilot.

The [committed pilot](../data/results/methods_exploration_pilot_20260907/report.md)
completed all four tracks in about 12 seconds, including process startup. It was
run from an archive of the staged tree, excluding pre-existing uncommitted
production-floor and theory changes. Its source hashes and dependency versions
are recorded in the manifest.

| Track | Pilot execution |
|---|---|
| Interior | Four paired observations, both alphas and eight C values, plus a doubled-cloud audit |
| Likelihood | Four observations with all predictors and all four requested cell resolutions |
| Projection | 936 exact path/candidate containment checks; 32 Monte Carlo outer projections |
| M3 | Boundary calibration, separate training/freeze/evaluation, and 36 held-out observations |

These are execution and certificate checks, not evidence about comparative power
or coverage on the full design. In particular, the likelihood hull is an optimistic
finite-library diagnostic, and the small outer projection's resolution can itself
make the band vacuous. Every pilot promotion gate remains disabled.

Validation passed **74 tests**, including the existing ladder, follow-up and M3
suites, both in the working tree and in the isolated staged-tree archive. The
new tests cover rational/float cell-DP agreement with atoms and gaps, normalized
predictors, exact-set containment, pathwise quantile coupling, retention of
unresolved boxes, plateau inversion, native-grid composition parity in both
imbalance directions, floor nesting, conservative area after resampling, schedule
clamps, per-design M3 tail vetoes, checkpoint recovery and deadline handling.
Ruff and staged whitespace checks pass. The existing thinned-grid parity test
emits its expected low-trim-depth warning.

Launch the planned research screen from the repository root:

```sh
uv run --no-sync python -m scripts.methods_exploration.run \
  --profile screen --out data/results/methods_exploration_screen_20260907 \
  --threads 2
```

The supervisor enforces the three-hour cap and records any unfinished work as
incomplete. The small replication budget targets information for subsequent
allocation: large-n coverage uncertainty may prevent freezing even when point
estimates look promising. Only a completed, supported proposal is eligible for
independent decision-gating simulation.
