# Stage S: Decision-First Screen of the Trim-Exponent Auto Map

*Run 2026-08-29. Spec: `stats/c_calibration_spec.md` (decision-first amendment,
2026-08-24). Theory companion: `stats/fiducial_band_theory.md` §7/§7.1.
Data: `data/results/c_calibration_20260829/`. Study git hash `5e27609acbb0`.*

**Verdict: STOP.** The screen's own gate returns
*"retain a documented fixed/default rule; full auto-map study is not
justified."* All three evidence arms are complete (27/27 cells, 52,000
replicates, 7.0 h of cell runtime). The blocking failure is the α=.05 shape
lower envelope: **0.967**, against a pre-registered requirement of ≥ 1.15.

This is a resource-allocation verdict, not a coverage guarantee. But the
screen also produced two findings that outrun its remit and that the theory
and working-model documents must absorb regardless of what happens to the
auto map.

---

## 0. Provenance and gates passed before any cell ran

| Check | Result |
|---|---|
| `tests/test_fiducial_ladder.py`, `test_c_calibration_design.py`, `test_fiducial_band_rs.py` | 63/63 pass |
| Parity gate part 1 — statistical, vs. published round-2 numbers | PASS on all 4 arms |
| Parity gate part 2 — exact same-seed vs. `fiducial_band_rs`, full grid | max area diff 7.2e-16, 0 coverage mismatches |
| Parity gate part 3 — exact same-seed, thinned trim grid (K > 2001) | max area diff 3.1e-15, 0 coverage mismatches |

Statistical parity detail (400 reps, M = 3000, binormal .95, n = 500/500):

| arm | α | this run | published | area (this run / published) |
|---|---|---|---|---|
| C=1 | .05 | 0.975 | 0.973 | 0.0633 / 0.0640 |
| C=2 | .05 | 0.955 | 0.960 | 0.0579 / 0.0584 |
| C=1 | .50 | 0.710 | 0.760 | 0.0415 / 0.0420 |
| C=2 | .50 | 0.5025 | 0.553 | 0.0349 / 0.0350 |

All within tolerance. Artifact: `data/results/c_calibration_20260829/parity_gate.json`.

Run configuration: 6 concurrent replicates × 4 rayon threads on 24 logical
cores, 45 GB cloud budget. Every cell except two was topped up to the
2,000-rep ceiling by the α=.05 SE gate, i.e. the gate's SE(C\*) ≤ 0.15 target
was **not** met at the ceiling in most cells — see §5.

---

## 1. The three screening questions, answered

### Q1 — Is the α=.05 one-SE lower shape envelope at n=500 at least C\* = 1.15, with ≥ 4% oracle area gain? **NO on the envelope; yes on the gain.**

| shape (n=500) | C\* | SE | C\* − 1 SE | oracle area gain vs C=1 |
|---|---|---|---|---|
| t2_95 | 1.173 | 0.206 | **0.967** | 2.2% |
| kink_80 | 1.563 | 0.219 | 1.344 | 5.4% |
| trapezoid_q10_90 | 2.010 | 0.238 | 1.772 | 8.0% |
| binormal_95 | 2.228 | 0.224 | 2.004 | 9.8% |
| binormal_90 | 2.280 | 0.266 | 2.014 | 9.7% |
| hetero_90_r3 | 2.360 | 0.257 | 2.102 | 10.8% |
| binormal_99 | 2.598 | 0.251 | 2.347 | 13.4% |
| bimodal_90 | 2.664 | 0.168 | 2.496 | 11.7% |
| binormal_75 | 2.814 | 0.279 | 2.534 | 11.6% |
| binormal_60 | 2.994 | 0.294 | 2.701 | 12.1% |

Mean oracle area gain 9.5% — the 4% efficiency gate passes comfortably, and
the 9–13% gap quoted in the spec is confirmed on 8 of 10 shapes. But the
envelope is set at **0.967 by t(2)**, and the whole point of the envelope
aggregation (D5) is that the shipped map must sit at or below it. A map
clamped at C ≈ 1 recovers essentially none of that 9.5%.

The dispersion is the finding: C\* ranges 1.17 → 2.99 across ten shapes at a
single (n, α). That is a 1.8-unit spread against a per-cell SE of ~0.22 —
roughly 8 SE, so it is structure, not noise. It restates in the C coordinate
the ±10–15pp central-α shape spread that `next_method_ideas.md` §5.1 already
records as unfixable by any level-only remap, and confirms that conclusion
now holds at α=.05 as well, not just at central α.

**The trapezoid did not falsify the floor conjecture.** Its C\* = 2.010 ±
0.238 sits mid-library — the deliberately rough legitimate estimand is not
the envelope-setter. t(2) is. (The one C\* < 1 flag raised is a boundary
artifact; see §2.)

### Q2 — Is a taper toward C = 1 visible on three mechanism-distinct shapes over n ∈ {100, 500, 5000, 50000}? **Yes at the endpoint; no as a common law.**

α=.05, C\* with bootstrap SE:

| shape | n=100 | n=500 | n=5,000 | n=50,000 |
|---|---|---|---|---|
| binormal_95 | 3.052 ± .329 | 2.228 ± .224 | 1.777 ± .212 | 0.867 ± .118 |
| kink_80 | 2.453 ± .219 | 1.563 ± .219 | 1.508 ± .218 | 1.030 ± .153 |
| t2_95 | 0.084 ± .040 † | 1.173 ± .206 | 1.489 ± .155 | 1.071 ± .145 |

† boundary-pinned at j\* = 2; not a usable calibration point (§2).

**This is the study's main scientific payload and it lands cleanly.** All
three shapes converge to C\* ≈ 1 at n = 50,000 — 0.867, 1.030, 1.071, mutually
indistinguishable and each consistent with exactly 1. This is the first
direct α=.05 measurement above n = 10⁴, which spec D3 named as the arm that
had never been run, and it **confirms Theorem 7**: the trim level is
asymptotically the coverage, so C\* → 1. Round 3's central-α extrapolation
was correct.

The corresponding C=1 coverages decline monotonically toward nominal on every
shape — binormal_95 0.9915 → 0.9820 → 0.9715 → 0.9513; kink_80 0.9850 →
0.9725 → 0.9670 → 0.9535; t2_95 (excl. n=100) 0.9575 → 0.9680 → 0.9595.
C=1's finite-sample conservatism is real at small n and **exhausted by
n = 50,000**.

But the approach to the limit is not a shared law. The screen's own endpoint
test resolves a decrease for binormal_95 (2.185, 95% lower bound 1.500) and
kink_80 (1.424, lower bound 0.901) and **fails to resolve one for t2_95**
(−0.987, i.e. the point estimate rises). kink_80 is additionally flat across
two decades (1.563 → 1.508 over n = 500 → 5,000). A single
δ₀·(n_eff/500)^(−γ) family fits binormal at γ ≈ 0.3, kink at γ ≈ 0 over its
middle range, and t(2) not at all. **D3's "is γ shared across shape?" is
answered: no.**

### Q3 — At fixed minority-class size 500, how strongly do direction and majority size move C\*? **Enough to reject a min(n₀,n₁) reduction on one of two shapes.**

All eight cells have min(n₀, n₁) = 500 by construction, so under the 1-D
`n_eff = min` reduction of D2 the C\* values should be constant within shape.

| shape | n₀×n₁ = 1500×500 | 4500×500 | 500×1500 | 500×4500 | spread |
|---|---|---|---|---|---|
| binormal_90 | 2.689 ± .210 | 1.688 ± .189 | 2.424 ± .203 | 2.238 ± .173 | **1.001** |
| t2_95 | 1.296 ± .163 | 1.180 ± .205 | 0.990 ± .174 | 0.905 ± .153 | 0.390 |

For binormal_90 the spread is 1.001 against a paired SE of 0.283 — **3.5 SE**.
The min() reduction is rejected on that shape. For t2_95 the spread is 0.390
against 0.224, i.e. 1.7 SE — not resolved.

**A caveat on the mechanical recommendation.** `check_screen.py` emits
*"test min(n0,n1) first; omit a broad 2-D sweep"* because it asks whether the
spread exceeds 0.15 under a Bonferroni-simultaneous z = 3.1, and binormal_90
scores 0.124 — just under the 0.15 line. That is a threshold miss by 0.026,
not evidence of reducibility. The honest reading is that binormal_90's spread
**is** resolved as nonzero and the round-4 prior against a 1-D `n_eff`
(`r4_report.md` §4) is corroborated, not relieved. Do not read the emitted
recommendation as clearing the min() reduction.

Direction also matters at the larger majority: for binormal_90 at
majority = 4500, negative-majority minus positive-majority C\* is
−0.550, 95% CI [−1.053, −0.047] — resolved at 95%, with negative-majority
cells needing the *smaller* exponent. t2_95's contrasts are both positive
(+0.305, +0.275) and neither is resolved, so the direction is not even
consistent in sign across the two shapes.

---

## 2. The t(2) × n=100 cell: a genuine coverage failure, and what it falsifies

This cell is the single `strong_c_below_1` flag, and it is **not** a floor-
conjecture falsification. It is a corner-coverage failure of the band, which
the C coordinate cannot express.

Measured at α=.05, C=1, M = 6694, 500 reps:

- **Coverage 0.802** (vs nominal 0.95).
- The coverage-vs-depth ladder tops out at 0.978 at j = 1 — the full cloud
  envelope plus CP allowance never reaches 0.95. The crossing therefore lands
  at j\* = 2, one rung off the top, where C\* = 0.084 is an artifact of the
  ladder boundary rather than a trim level anyone could ship.
- 95% of misses are the truth falling **below** the band's lower edge.
- Miss locations are bimodal: 42% at k ≤ 3 and ~55% at k ∈ [94, 99]. Nothing
  in between — this is the two-corner effect (cf. `figA5_two_corner_variance_bias`).
- Miss depths: median 0.0036 on missing reps, p95 over all reps 0.024, max
  0.102. **2.2% of reps miss by more than 5pp**, all at k ∈ {1, 2, 4}.
- M is adequate: mean realized trim depth 17.1, far above the j < 3 warning
  threshold. This is not a Monte Carlo artifact.

**Mechanism.** On the n₀ = 100 grid the t(2) truth is unresolvable at both ends:

| | R(0.01) | R(0.02) | R(0.03) | 1 − R(0.96) |
|---|---|---|---|---|
| t2_95 | 0.0693 | 0.4017 | 0.7138 | 7.9e-3 |
| binormal_95 | 0.4999 | 0.6074 | 0.6720 | 2.3e-5 |

At the upper corner, 1 − R(0.96) = 0.0079 lies *below* the 1/n₁ = 0.01
resolution: with 100 positives the empirical TPR is 1.0 and no rank-based
band can separate R = 0.992 from R = 1. At the lower corner the truth climbs
0.07 → 0.40 → 0.71 across three grid points — vertical at grid resolution.
binormal_95 faces neither (it is at 1.0 to within 2e-5 above, and at 0.50 by
the first grid point below). This is exactly the identifiability frontier of
theory doc §9 and open question 3 ("no band can certify a nonvacuous lower
bound below ~c/n₀"), appearing for the first time as a coverage number rather
than a width caveat.

### What this falsifies

Nothing in `fiducial_band_theory.md`. Theorem 7 is asymptotic and interior
and states plainly: *"No finite-sample validity claim is made for C=1"*, with
corner zones explicitly out of scope. Consistent with that, the theory doc
carries no coverage theorem for the fiducial band.

Three claims in `next_method_ideas.md` **are** falsified, all of them
empirical extrapolations from a library that never contained this cell:

1. §2 headline table: *"Coverage @ α=.05, identity map (C=1) | 0.967–0.995"*.
2. §5.2, echoed verbatim in the theory doc's production guidance
   (`fiducial_band_theory.md:803`): *"C=1 is the conservative, asymptotically
   calibrated fallback (never measured below .967 at α=.05)"*.
3. Prediction **P-A**, whose stated falsifier is "a stratum below ~0.93 at
   adequate M". Measured 0.802.

Prediction **P-D** also fails: its falsifier is ">5pp misses at a rate above
~1%", and this cell gives 2.2%.

**Why it was never caught.** The C=2 fit used t(2) at n = 500 only; round 3's
C\*(n) ladder was fixed-shape. t(2) × n = 100 — rough shape crossed with small
n — is the one corner of the design space that had never been visited. t(2)
at n = 500 / 5,000 / 50,000 gives C=1 coverage 0.9575 / 0.9680 / 0.9595, all
sound. The failure is specific to rough-shape × small-n, not to t(2).

**Recommended documentation change:** "C=1 is the safe fallback" must now
carry an explicit n ≳ 500 caveat, and P-A/P-D should be restated with a
small-n rough-shape exclusion until the boundary is mapped.

---

## 3. What the reference arms say about the shipped default

C=2 is the current library default. At α=.05 across all 27 cells:

- C=2 coverage ranges 0.750 (t2 n=100) / 0.9140 – 0.9730 elsewhere.
- **At n = 50,000, C=2 undercovers on every shape**: 0.9167 (binormal_95),
  0.9140 (kink_80), 0.9245 (t2_95). This is Theorem 7's predicted liability
  ((1−α)^C = 0.9025 in the limit) arriving in the measured range at α=.05,
  where previously it had only been demonstrated at central α.
- The provisional auto formula (γ = 0.32, δ₀ = 0.8) does better at large n —
  0.9420 / 0.9465 / 0.9505 — but undercovers on t(2) at n=500 (0.9365,
  because it prescribes C = 1.8 where t(2)'s C\* is 1.173).

So the existing default has a real large-n problem that the screen documents
quantitatively, and the provisional map fixes that end while breaking the
rough-shape end.

---

## 4. Decisions and recommended routing

**Primary.** Do not launch Stage A as specified. Per spec §8's fallback
clause, keep a documented fixed rule. But the screen's evidence does not
support the spec's suggested fallback of "C = 2.0 for n_eff ≤ 1000 and C = 1.0
above" without amendment, because C = 2.0 at n = 500 already undercovers t(2)
(0.9325) and the n=100 rough-shape regime is not safe at any C.

**Per-decision status:**

- **D1 (coordinate).** Unaffected — C remains the shipped control. The screen
  adds that C is a poor coordinate near the ladder boundary, where it
  compresses toward 0 without signalling that it has stopped being meaningful.
- **D2 (imbalance reduction).** Prior evidence against a 1-D `n_eff` is
  corroborated: binormal_90 rejects min(n₀,n₁) at 3.5 SE at fixed minority
  size, with a resolved directional contrast at majority = 4500. Treat the
  emitted "test min() first" as a threshold artifact, not a clearance.
- **D3 (finite-range taper).** Resolved in the most important respect:
  C\* → 1 at n = 50,000 at α=.05 on all three shapes, confirming Theorem 7 and
  closing the spec's stated gap. Simultaneously, γ is **not** shared across
  shapes, so the single-family form of the auto map is dead.
- **D4 (α-drift).** Not evaluated — the screen ran α ∈ {.50, .20, .10, .05}
  from one cloud but the analysis gate is α=.05 only. Deferred; moot unless a
  map is revived.
- **D5 (floor conjecture).** The trapezoid truth did **not** push C\* below 1
  (2.010 ± 0.238). The one sub-1 flag is the boundary artifact of §2. The
  floor conjecture stands, but the *reason* it matters has changed: the
  binding constraint is no longer "is C\* < 1?" but "is the band valid at
  all?" in the rough-shape small-n corner.
- **D6 (degenerate shapes).** No cell was flagged `unconstrained` or
  `infeasible`. The exclusion rule was never exercised.

**Recommended next work, in priority order:**

1. **Map the rough-shape small-n validity boundary.** t(2) at
   n ∈ {150, 200, 300, 500} × {AUC .90, .95} would locate where C=1 coverage
   recovers, and whether other rough shapes (bimodal, hetero) fail there too.
   This is a validity question and outranks everything about the auto map.
2. **Amend the documentation** per §2 — three specific claims in
   `next_method_ideas.md` plus the identical sentence in the theory doc.
3. **Validate a C=1 clamp at large n** rather than fitting a tail surface, as
   the screen's own routing recommends. The n = 50,000 evidence is that C=1 is
   the right answer there and C=2 is measurably wrong.
4. Only then reconsider whether any restricted map (e.g. shape-agnostic C(n)
   clamped below the t(2) envelope) buys enough width to be worth freezing.
   On present evidence it does not: the envelope is 0.967 at n=500.

---

## 5. Precision and limitations

- **The SE gate was not met.** Spec §4 targets SE(C\*) ≤ 0.15 at α ≤ .2.
  25 of 27 cells were topped to the 2,000-rep ceiling and most still sit at
  SE 0.15–0.33. The screen's conclusions rest on contrasts that are large
  relative to that noise (the shape spread is ~8 SE, the binormal imbalance
  spread 3.5 SE, the taper endpoints 1.5–2.2 units); the individual C\*
  values are not precise to the spec's target and should not be quoted as if
  they were. A Stage A fit at these SEs would have been underpowered.
- **`bootstrap_coordinates` flags `saturated` only when `j_star == ladder[0]`.**
  A crossing at j = 2 or 3 is equally boundary-pinned but passes through as a
  usable estimate — which is how the t(2) n=100 artifact reached the
  envelope calculation and drove the floor flag. Recommend widening the flag
  to `j_star <= 3` (matching the band's own j < 3 low-M warning) before any
  further stage.
- **`check_screen.evaluate` conflates two failure modes.** `proceed` requires
  `not floor_flags`, so a corner-validity failure and an absent calibration
  margin produce the same STOP string. Here the verdict happens to be right
  for the envelope reason independently, but the reported cause would have
  been wrong.
- Screen α grid is {.50, .20, .10, .05}; only α=.05 was analyzed. The other
  three columns are on disk and re-analyzable without re-simulation.
- Shape library is the 10 fitting shapes only. Held-out shapes are untouched,
  as designed.

---

## 6. Artifacts

All under `data/results/c_calibration_20260829/`:

- `parity_gate.json` — the §0 gate results.
- `stageS/<cell>.json.gz` — raw per-rep ladder profiles, 27 cells (re-fittable
  under any aggregation without re-simulation).
- `stageS/<cell>.summary.json` — per-cell aggregates: cov(j), per-α
  j\*/C\*/α_eff\*/ℓ\* with bootstrap CIs, flags, allowance attribution,
  reference-map coverage and area.
- `stageS_run.log` — full run log.
- `screening_check.json` / `screening_report.md` — the mechanical verdict.

Reproduce with:

```bash
uv run python scripts/c_calibration/parity_gate.py \
    --out data/results/c_calibration_20260829
uv run python scripts/c_calibration/run.py --stage S \
    --out data/results/c_calibration_20260829 --workers 6 --threads 4 --mem-gb 45
uv run python scripts/c_calibration/check_screen.py \
    --in data/results/c_calibration_20260829/stageS \
    --out data/results/c_calibration_20260829
```
