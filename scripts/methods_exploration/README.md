# Bounded method exploration

The [specification](../../stats/methods_exploration_spec.md) defines the methods,
estimands, seed separation, limits, and gates. This package changes no production
method. It uses the installed Rust extension and existing NumPy/SciPy environment;
no new dependencies or uncommitted floor modules are required.

From the repository root:

```sh
uv run --no-sync python -m scripts.methods_exploration.run \
  --profile pilot --out data/results/methods_exploration_pilot --threads 2

uv run --no-sync python -m scripts.methods_exploration.run \
  --profile screen --out data/results/methods_exploration_screen --threads 2
```

`screen` is capped at six hours, split 60/195/45/30/30 minutes among small-n,
interior, likelihood, projection and M3. The small-n study runs first.
`--minutes` changes the total cap. `--track m3`
(or `small_n`, `interior`, `likelihood`, `projection`) runs a track in isolation;
its default cap is that track's allocation, and `--minutes` overrides it. The
screen runs one worker at a time. Large
interior clouds need several GB of RAM, particularly the doubled-M audit.

Reusing the identical command resumes completed observations and skips completed
tracks. A changed source hash, environment version, profile or thread count requires
a new output directory. Killed partial JSON lines are discarded; complete lines are
retained. Unit deadlines can stop a Rust call through worker termination. Read
`status.json` as well as `summary.json`: a budget-exhausted or failed worker does not
pass a gate, even if a stage summary was written before it stopped.

Each directory contains `manifest.json`, `report.md`, per-track `config.json`,
`records.jsonl`, `run.log`, and completion status. Completed tracks also write:

- **small_n:** production-budget studies at class sizes 10–50, with exact/Stage F
  tails, overlap, ablations, unseen-sliver conditioning and pointwise miss rates.
- **interior:** operational and eligible-only coverage crossings, censoring, bootstrap spread bounds, paired width
  uncertainty, doubled-cloud observations, and a `candidate.json` proposal. Frozen
  values require the observed eligibility flag in `interior.schedule_exponent`.
  Any tail overlap with the fixed window returns the unchanged hybrid. The plan
  uses 2,000 replicates per smaller cell and 1,000 per larger cell; calibration
  requires at least 400 eligible samples. An unfinished design remains incomplete.
- **likelihood:** frozen prefix-predictor tables, per-predictor likelihood penalties,
  actual cell counts and gaps, optimistic inner-hull widths and projection slack.
  A completed positive smooth n=20 gate enables the rational box prototype.
- **projection:** exact small-n confidence-set containment checks and probability-
  weighted results, followed by paired common-random-number outer bands at total
  n=50. The exact and Monte Carlo confidence sets are different targets.
- **m3:** frozen boundaries and calibration probabilities written before held-out
  sampling, per-shape tail constraints, and paired evaluation uncertainty.

Coverage location intervals index each band's reporting grid. Regional miss flags
are grid diagnostics. Integrated width uses the monotone continuum extension, not
linear edge interpolation. M3 and C=1 remain paired comparators. A finite candidate
library can understate likelihood-inversion width, so its screen never certifies a
band. Coarse outer knot grids can add substantial projection width.

A pilot checks routing and numerics with deliberately tiny replication. The small-n
track retains production cloud sizes; the other tracks use reduced budgets.
Method-promotion flags are always false; dataset window-eligibility flags are measured. The full screen is also a research
screen: a frozen empirical exponent requires independent decision-gating simulation.

Validation:

```sh
uv run --no-sync pytest tests/test_methods_exploration.py tests/test_hybrid_exploration.py \
  tests/test_fiducial_ladder.py tests/test_followup_runs.py tests/test_m3_band_rs.py
uv run --no-sync ruff check scripts/methods_exploration tests/test_methods_exploration.py tests/test_hybrid_exploration.py
```
