# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Simultaneous ROC Bands Under Class Imbalance — 50%, 30%, and 10% Prevalence
#
# This notebook compares three simulation runs that differ only in the class
# split:
#
# | Run | Prevalence | Sample sizes | Folder |
# |---|---|---|---|
# | Standard | 50% | 10 – 10,000 (+ 10% at n = 1,000) | `data/results/nov11` |
# | Moderate imbalance | 30% | 30 – 10,000 | `data/results/nov11_p30` |
# | Strong imbalance | 10% | 30 – 10,000 | `data/results/nov11_p10` |
#
# All three use the same seed, so the 6,000 maximin-LHS parameter draws per
# DGP are identical; only the datasets drawn from them differ. The methods,
# levels (50/80/95%), and evaluation are those of the standard run (see
# `notebook_paper_nov11.py` for the method descriptions).
#
# **Comparability rules.**
#
# - n = 10 exists only in the standard run, so every comparison here starts at
#   n = 30.
# - The standard run's own 10% cell (n = 1,000) is held out of the 10% series
#   and used as a replication check (Section 1): the 10% run redraws that
#   exact cell independently.
# - Averages over DGPs use a prevalence at a sample size only when all seven
#   DGPs are present there, so a run still in progress never shifts an
#   average through its composition. Per-DGP figures show whatever exists.
#
# **How the figures are organized.**
#
# - Every faceted figure of the standard analysis is redrawn once per
#   prevalence (file stems ending `_p50`, `_p30`, `_p10`), and the panels of
#   each set share identical axis limits, so the three versions compare
#   directly.
# - Figures whose key datum was one point per method now show three
#   line-connected points, one per prevalence: filled = 50%, half-filled =
#   30%, hollow = 10%.
# - New views that need several prevalences: a method × metric overview,
#   coverage shifts against the balanced run, a heteroscedastic-Gaussian
#   spotlight, and the replication check.
#
# The figure code lives in `stats/paper_figures_prevalence.py` (built on
# `stats/paper_figures_nov11.py`); figures go to `figures/paper_prevalence/`.
# The single-run analysis of the 10% run alone is in
# `figures/paper_nov11_p10/`.
#
# **Regenerating** (figures only, no commentary):
# ```
# uv run python stats/paper_figures_prevalence.py
# ```
#
# Calibration is the success criterion throughout: coverage closer to
# nominal is better, and coverage above nominal is conservative, not "good".

# %%
import sys
import warnings
from pathlib import Path

import pandas as pd

for _path in ("src", "stats"):
    if str(Path(_path).resolve()) not in sys.path:
        sys.path.append(str(Path(_path).resolve()))

import paper_figures_nov11 as pf  # noqa: E402
import paper_figures_prevalence as pp  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.width", 220)
pf.set_style()

data = pp.load_prevalence_data()
cells = data.cells
frames = data.frames
print(f"Prevalences loaded: {[pf.prevalence_label(p) for p in data.prevalences]}")
print(f"Trials: {sum(len(f) for f in frames.values()):,}")


def at(prevalence: float) -> pd.DataFrame:
    """Return the cell summaries at one prevalence."""
    return cells[cells["prevalence"] == prevalence]


# %% [markdown]
# ### Data completeness
#
# Replicates per DGP × n (hybrid, 95%). Empty entries are cells a run has
# not produced yet.
#
# **Status when the commentary was written:** the 50% and 10% runs were
# complete. The 30% run had finished only the binormal DGP at n = 30–3,000.
# Its points appear in per-DGP figures where present, its per-prevalence
# figures show binormal only, and macro averages leave it out. The
# commentary therefore compares 50% with 10%; rerun the notebook once the
# 30% run finishes to fill in the middle series.

# %%
pp.completeness(data)

# %% [markdown]
# ## 1. Replication check: the n = 1,000, 10% cell drawn twice
#
# **Motivation.** The standard run already contains one 10% cell (n =
# 1,000). The 10% run draws the same cell again with fresh data. If both
# runs are sound, the coverage differences across every method × level ×
# DGP should look like Monte Carlo noise: z-scores near standard normal.

# %%
replication = pp.replication_check(data)
pp.figP26_replication(replication)
replication.groupby(level="alpha")["z"].describe().round(2)

# %% [markdown]
# **Reading.** The two draws of the cell agree.
#
# - At 95% the 42 z-scores (6 methods × 7 DGPs) have mean −0.11 and
#   standard deviation 0.98, almost exactly standard normal.
# - Across all 126 comparisons, 10 have |z| > 2 against about 6 expected.
#   The excess sits at the 80% and 50% levels, mostly Student-t, with the
#   10% run slightly higher.
# - The z-scores are not independent: the methods and levels within a DGP
#   share their datasets, so one lucky draw moves several at once.
#
# Nothing here suggests a difference between the runs beyond sampling noise,
# so the standard run's 10% cell and the 10% run can be read as the same
# experiment.

# %% [markdown]
# ## 2. Overview: how each method responds to imbalance
#
# ### Fig P3 — Method × metric overview
#
# Four summary metrics against n for every method, one line per prevalence:
# mean and worst-cell coverage at 95%, calibration error averaged over the
# three levels, and band area relative to KS.

# %%
pp.figP03_method_overview(cells)

# %% [markdown]
# **Reading.** Going from 50% to 10% prevalence:
#
# - **Hybrid.** Calibration is unchanged. Mean 95% coverage is 0.970–0.993
#   at 10% against 0.970–0.992 at 50%, and the worst cell is 0.967 against
#   0.969. Calibration error averaged over the three levels is slightly
#   *lower* at 10% from n = 300 (0.101 vs. 0.114 at n = 1,000).
# - **Raw fiducial band.** Its mean barely moves, but its worst cell falls
#   to 0.893 (n = 1,000) and stays below 0.95 from n = 100 to 3,000. The
#   floor exists for exactly this case.
# - **Envelope.** It turns *more* conservative under imbalance: coverage
#   rises (0.990 vs. 0.984 at n = 1,000), its one sub-nominal cell from the
#   balanced run disappears, and calibration error and area relative to KS
#   both worsen (area ÷ KS 0.82 vs. 0.56 at n = 1,000).
# - **M3 and KS.** Coverage stays at about 1.00 at every prevalence.
# - **Working–Hotelling.** It covers *more* at 10% (0.62 vs. 0.54 mean at
#   n = 1,000) because its band widens, but it still collapses with n.
#
# Every method's band is wider at 10% (bottom row). This follows from the
# per-class sample sizes, n₀ = 0.9n and n₁ = 0.1n, not from any method's
# design.

# %% [markdown]
# ### Fig P2 — Coverage shift against the balanced run
#
# Coverage at 30% and 10% prevalence minus coverage at 50%, per DGP and n,
# at 95%. The runs are independent, so each shift carries the Monte Carlo
# error of both cells: about ±0.006 (1.96 SE) at coverage 0.97.

# %%
pp.figP02_coverage_shift(cells)

# %% [markdown]
# **Reading (10% − 50%).**
#
# - **Hybrid.** Every shift lies within ±0.009; the mean is −0.002, inside
#   the Monte Carlo band.
# - **M3.** Shifts lie within −0.003 to 0.000.
# - **Raw fiducial band.** The heteroscedastic-Gaussian row is the only
#   structure: −0.026 at n = 30, −0.046 at 100, −0.064 at 300, −0.073 at
#   1,000, −0.051 at 3,000, and −0.025 at 10,000. Every other cell lies
#   within about ±0.006.
# - **Envelope.** It shifts upward, most at n = 10,000 (+0.029 in the
#   heteroscedastic Gaussian, about +0.015 elsewhere).

# %% [markdown]
# ### Table 2 — Method summary at each prevalence
#
# The standard run's Table 1, computed separately per prevalence over the
# cells each run has (n ≥ 30).

# %%
table2 = pp.summary_tables(data)
table2.round(3)

# %% [markdown]
# **Reading.**
#
# - **Cells significantly below nominal.** The hybrid has none at any level
#   at either prevalence. The raw fiducial band has none at 50% but five at
#   10%, all heteroscedastic Gaussian: at 95% for n = 100, 300, 1,000, and
#   3,000 (worst 0.893), and at 80% for n = 300. Working–Hotelling has
#   25–29 per level at both prevalences.
# - **Area.** The hybrid's area ratio to KS rises from 0.50 to 0.59 at
#   n = 1,000. In absolute terms its area at 10% is 1.67–1.69× its balanced
#   area for every n ≥ 300.
# - **Effective sample size.** That ratio is the effective-sample-size
#   prediction: band width scales as 1/√(n·p(1−p)), and
#   √(0.25 / 0.09) = 1.667. Both fiducial arms sit at 1.66–1.67 for
#   n ≥ 1,000. KS is less sensitive (1.41–1.45 there) and
#   the envelope more (up to 2.07). Imbalance costs the rank-space bands
#   exactly what the information loss predicts, and no more.

# %% [markdown]
# ## 3. Calibration
#
# ### Fig P1 — Coverage heatmaps by level and prevalence
#
# One figure per method: rows are levels, columns are prevalences, so a
# column-to-column comparison is a pure imbalance effect.

# %%
for method in pf.METHOD_ORDER:
    pp.figP01_coverage_heatmaps(cells=cells, method=method)

# %% [markdown]
# **Reading.**
#
# - **Hybrid.** Its 10% column is a near copy of its 50% column at every
#   level: 0.967–0.996 at 95%, 0.862–0.969 at 80%, and 0.624–0.893 at 50%.
#   The sup-norm conservatism at the lower levels shrinks with n the same
#   way at both prevalences.
# - **Raw fiducial band.** Its heteroscedastic-Gaussian row is the only one
#   that turns red at 10%.
# - **Envelope.** It gets bluer (more conservative) at 10% for n ≥ 1,000.
# - **Working–Hotelling.** Its off-model rows are red at every prevalence.

# %% [markdown]
# ### Fig 2 — Coverage vs. true AUC, every method, per prevalence
#
# The standard analysis's Fig 2 for each method, drawn once per prevalence
# with matched axes.

# %%
pp.per_prevalence(
    data=data,
    make=lambda p: [
        pf.fig02_coverage_vs_auc(frames[p], method=m) for m in pf.METHOD_ORDER
    ],
)
# %% [markdown]
# **Reading.**
#
# - **Hybrid.** At 10% it is as flat in AUC as it is balanced: every DGP
#   and n stays within 0.955–0.99, with no high-AUC droop.
# - **Raw fiducial band, heteroscedastic Gaussian.** At 10% the high-AUC
#   end gives way. Coverage falls to about 0.73 at AUC 0.98 for n = 1,000,
#   with a dip to about 0.84 around AUC 0.87 at n = 100 and to 0.87 at the
#   top for n = 10,000.
# - **Raw fiducial band, Student-t.** It slips to 0.94 at the highest AUCs.
# - **Raw fiducial band, other DGPs.** They are unchanged from the balanced
#   run.
# - **Envelope, M3, KS, and Working–Hotelling.** They show the same
#   AUC-profiles as balanced, shifted by their prevalence-level offsets
#   (Fig P3).

# %% [markdown]
# ### Fig 3 — Coverage vs. DGP shape parameters, every method, per prevalence

# %%
pp.per_prevalence(
    data=data,
    make=lambda p: [
        pf.fig03_coverage_vs_shape(frames[p], method=m) for m in pf.METHOD_ORDER
    ],
)

# %% [markdown]
# **Reading.** At 10% prevalence the raw fiducial band's balanced pockets
# deepen.
#
# - **Heteroscedastic Gaussian.** Once σ₊/σ₋ exceeds about 2, coverage falls
#   to about 0.80 at n = 100 and 0.81 at n = 1,000 (about 0.91 and 0.95
#   balanced).
# - **Student-t.** The df ≈ 2 dip at n = 1,000 goes from about 0.90 to 0.83.
#   This second, more local failure is also in the tails, and the per-cell
#   tables average it away.
# - **Hybrid.** It stays flat at or above 0.95 along every shape axis at
#   both prevalences. The one touch is the heteroscedastic Gaussian at
#   σ₊/σ₋ ≈ 0.2 and n = 100, at 0.948 with a wide ribbon.
# - **Envelope.** It loses its balanced dip below nominal at large σ₊/σ₋
#   and n = 10,000 (≈ 0.935 → 0.97), consistent with its general drift
#   toward over-coverage under imbalance.
# - **M3, KS, and Working–Hotelling.** They look the same at both
#   prevalences: flat at 1.00 for the first two, collapsing off-model for
#   Working–Hotelling.

# %% [markdown]
# ### Fig 4 — Calibration across confidence levels, per prevalence

# %%
pp.per_prevalence(data=data, make=lambda p: pf.fig04_level_calibration(at(p)))
# %% [markdown]
# ### Fig P24 — Per-cell calibration, traced across prevalences
#
# Each DGP × n cell is a three-point trace within its method's slot (50%,
# 30%, 10% from left to right); filled points are cells whose Wilson
# interval lies entirely below nominal. The counts above each slot are those
# cells per prevalence.

# %%
pp.figP24_calibration_distribution(cells)

# %% [markdown]
# **Reading.**
#
# - **Hybrid.** Its traces stay in a tight band above zero at every level,
#   with no filled points at any prevalence.
# - **Raw fiducial band.** At 95% four traces dive from about +0.03 at 50%
#   to −0.03 to −0.06 at 10%, and one more crosses below zero at 80%. All
#   five are heteroscedastic-Gaussian cells.
# - **Envelope.** Its traces tilt upward (more conservative) at 10%.
# - **Working–Hotelling.** It has a fan of filled points at every
#   prevalence.

# %% [markdown]
# ## 4. Robustness
#
# ### Fig 6 — Coverage vs. n per DGP, every method, per prevalence

# %%
pp.per_prevalence(data=data, make=lambda p: pf.fig06_coverage_vs_n(at(p)))
# %% [markdown]
# ### Fig 7 — Working–Hotelling fragility, per prevalence

# %%
pp.per_prevalence(data=data, make=lambda p: pf.fig07_wh_fragility(frames[p]))
# %% [markdown]
# **Reading.** Working–Hotelling degrades with departure from binormality at
# every prevalence. At 10% its curves start higher because its band is
# wider, but they fall the same way as the tails get heavier or the
# negatives more bimodal. The hybrid's curves stay flat near the top in
# every panel.

# %% [markdown]
# ### Fig P25 — Spotlight: the heteroscedastic Gaussian at high AUC
#
# **Motivation.** The one cell where the raw fiducial band failed in the
# standard run was the heteroscedastic Gaussian at n = 1,000 and 10%
# prevalence. The mechanism is a convex hook at an end of the ROC: when the
# positives are more spread out than the negatives (σ₊/σ₋ > 1), the ROC
# sags below the chord near FPR = 1, and when they are tighter it bulges at
# FPR = 0. The fiducial cloud completes a run of one class's observations
# with no evidence about the other class's shape there, and with many
# negatives per positive the grid resolves deep into the hook. This figure
# splits the high-AUC draws (true AUC > 0.9) by σ₊/σ₋ and tracks coverage
# and the violation rate in each end region.

# %%
pp.figP25_hetero_spotlight(frames)

# %% [markdown]
# **Reading.** The failure is confined to one direction of one shape.
#
# - **Positives wider (σ₊/σ₋ > 1.5).** The raw fiducial band covers 0.786
#   at n = 300 and 0.747 at n = 1,000 at 10% prevalence, against 0.924 and
#   0.937 balanced. Its plateau-region (FPR 90–100%) violation rate is
#   0.21–0.24 at 10% against 0.05–0.07 balanced. Even balanced it was
#   already below nominal in this corner; imbalance multiplies the defect.
# - **Recovery at large n.** Coverage recovers as n grows (0.896 at
#   n = 10,000) as the per-class counts outgrow the hook.
# - **Positives tighter or similar.** There is no failure at either
#   prevalence. The bulge at FPR = 0 is resolved by the positives above
#   the largest negative, and imbalance leaves fewer positives, not more
#   unresolved negatives, at that end.
# - **Hybrid.** It holds 0.96–0.99 in every column at both prevalences,
#   and its plateau violation rate stays at or below 0.011.
#
# This confirms the mechanism: with n₀ ≫ n₁, many negatives fall below the
# smallest positive, the fiducial completion draws a straight chord through
# a region with no positives, and the true curve sags below that chord.

# %% [markdown]
# ### Fig P27 — Coverage by AUC and shape bin, pooled over n
#
# Student-t and the heteroscedastic Gaussian, the two families whose shape
# parameter can put a convex hook at an end of the ROC. Rows are
# shape-parameter bins and columns true-AUC bins. Methods run along x, with
# prevalence falling left to right inside each method. Each point pools all
# sample sizes n = 30–10,000. A prevalence appears only once its run has
# every n for that DGP, so the pooled n mix is identical across points. The
# per-n view, with n inside each method, is Fig 25 of the single-run
# analyses (`figures/paper_nov11/`, `figures/paper_nov11_p10/`).

# %%
for dgp in pf.SHAPE_BINS:
    pp.figP27_shape_auc_coverage(frames, dgp=dgp)

# %% [markdown]
# **Reading.** Imbalance widens the raw band's failure region in both
# families and leaves the hybrid where it was.
#
# - **Heteroscedastic Gaussian, σ₊/σ₋ = 2–5.** The raw band falls in every
#   AUC bin at 10%, not only the highest: 0.981 → 0.931 (AUC 0.50–0.75),
#   0.972 → 0.922, 0.967 → 0.878, and 0.945 → 0.829 (AUC > 0.95), pooled
#   over n. Per n, the worst points at 10% are 0.64 (AUC > 0.95,
#   n = 1,000) and 0.75 (AUC 0.90–0.95, n = 300). With σ₊/σ₋ ≤ 2 there is no
#   shift.
# - **Student-t, heavy tails at high AUC.** At df 1–2 and AUC > 0.95 the raw
#   band goes from 0.872 to 0.795 pooled, reaching 0.48 at n = 300 and 0.58
#   at n = 1,000 for 10%. At df 2–4 and AUC > 0.95 it goes from 0.916 to
#   0.857, with 0.67 at n = 1,000 and 0.82 even at n = 10,000. At df ≥ 4
#   nothing moves.
# - **Hybrid.** It stays at 0.97–0.99 pooled in every panel at both
#   prevalences. Per n, none of its points has a Wilson upper bound below
#   0.95.

# %% [markdown]
# ## 5. Tightness
#
# ### Fig P8 — Coverage vs. area, one point per prevalence

# %%
pp.figP08_pareto(cells)

# %% [markdown]
# **Reading.**
#
# - **Shape of each trace.** Every method's trace runs mostly *rightward*:
#   imbalance costs width, not coverage.
# - **Hybrid and raw fiducial.** At n ≥ 1,000 their traces lie on top of
#   each other, and both stay the tightest calibrated methods at 10%.
# - **Envelope.** Its trace is long and flat: at n = 100–1,000 it ends up
#   about as wide as M3 at 10% while still over-covering.
# - **Working–Hotelling.** Its trace climbs and widens, but stays far
#   below nominal.

# %% [markdown]
# ### Fig P8b — Per-DGP coverage–tightness with 95% CIs

# %%
for family in pf.DGP_FAMILIES:
    pp.figP08b_pareto_ci(cells=cells, family=family)

# %% [markdown]
# ### Fig P22 — Coverage and area per DGP at n = 100, 1,000, and 10,000

# %%
for n in (100, 1000, 10000):
    pp.figP22_prevalence_by_dgp(cells=cells, n=n)

# %% [markdown]
# **Reading.**
#
# - **Coverage.** The coverage traces are short for every method and DGP
#   except the raw fiducial band in the heteroscedastic Gaussian (0.966 →
#   0.893 at n = 1,000).
# - **Area.** The area traces are long and nearly identical across DGPs:
#   the width cost of imbalance depends on the class counts, not the score
#   distribution.

# %% [markdown]
# ### Figs 8c, 9, 10 — Trajectory, width scale, and width profile, per prevalence

# %%
pp.per_prevalence(data=data, make=lambda p: pf.fig08c_trajectory(frames[p]))
# %%
pp.per_prevalence(data=data, make=lambda p: pf.fig09_width_scale_and_calibration(at(p)))
# %%
pp.per_prevalence(data=data, make=lambda p: pf.fig10_width_profile(frames[p]))
# %% [markdown]
# **Reading.**
#
# - **Width scale (Fig 9).** The panels shift up as a whole at 10%: every
#   method is wider relative to WH and KS at small n.
# - **Width profile (Fig 10).** At 10% the hybrid's profile flattens on the
#   right. At n = 100 its width is 0.45 from FPR 0.5 to 0.9, identical to
#   M3, because the plateau floor covers most of the curve there. At
#   n = 1,000 it is 0.072 at FPR 0.9, against 0.047 for the raw band and
#   0.084 for M3.
# - **Large n.** At n = 10,000 the hybrid and the raw band agree to within
#   0.002 at every landmark.

# %% [markdown]
# ## 6. Ablations of the hybrid, per prevalence
#
# The raw fiducial band is the hybrid without its floor; M3 is the floor
# applied everywhere.

# %%
pp.per_prevalence(
    data=data, make=lambda p: pf.fig11_no_floor_ablation(df=frames[p], cells=at(p))
)
# %%
pp.per_prevalence(data=data, make=lambda p: pf.fig12_corner_ablation(frames[p]))
# %%
pp.per_prevalence(data=data, make=lambda p: pf.fig13_floor_only_ablation(at(p)))
# %%
pp.per_prevalence(data=data, make=lambda p: pf.fig14_single_component_summary(at(p)))
# %% [markdown]
# **Reading.** Imbalance turns the ablation from a corner detail into the
# main argument for the floor.
#
# - **Balanced (Fig 12).** The two fiducial arms have nearly the same mean
#   coverage, and the floor's work shows only in the plateau-corner
#   violation rate (0.002–0.005 raw vs. 0.000–0.002 hybrid).
# - **At 10%, plateau corner.** The raw band's plateau-corner rate rises to
#   0.010–0.017 for n = 100–3,000 while the hybrid's stays at
#   0.001–0.003.
# - **At 10%, mean coverage.** The raw band's mean coverage drops below the
#   hybrid's from n = 300 (0.968 vs. 0.978 at n = 1,000; 0.955 vs. 0.977
#   at AUC > 0.9).
# - **Worst cell (Figs 11, 14).** The raw band reaches 0.893; the hybrid
#   stays at 0.967 or above.
# - **M3 alone (Fig 13).** It stays valid and wide at every prevalence. Its
#   50% band still covers 0.95 at 10%, against 0.99 balanced, so it remains
#   nearly blind to the nominal level.

# %% [markdown]
# ## 7. What the floor buys, per prevalence

# %%
pp.per_prevalence(
    data=data, make=lambda p: pf.fig15_paired_floor_effect(data.paired[p])
)
# %%
pp.per_prevalence(data=data, make=lambda p: pf.fig16_floor_region_geometry(frames[p]))
# %%
pp.per_prevalence(data=data, make=lambda p: pf.fig17_misses_by_floor_region(frames[p]))
# %%
pp.per_prevalence(
    data=data, make=lambda p: pf.fig18_floor_tradeoff_by_region(frames[p])
)
# %%
pp.per_prevalence(data=data, make=lambda p: pf.fig19d_floor_occupancy(frames[p]))
# %% [markdown]
# **Reading.**
#
# - **Paired effect (Fig 15).** At 10%, pooled over DGPs, the floor raises
#   95% coverage by +0.007 at n = 300 and +0.010 at n = 1,000 (95% CIs
#   +0.006 to +0.009 and +0.008 to +0.011). Balanced, the pooled difference
#   stays within ±0.003. The pooled number hides two opposite effects:
#   - In the heteroscedastic Gaussian the floor adds +0.04 to +0.08 at 95%
#     for n = 30–3,000, and +0.08 to +0.11 at 80%.
#   - In the other six DGPs the hybrid covers about 0.005 *less* at 95% and
#     0.02–0.07 less at 80% and 50%. Where the raw band over-covers, the
#     recalibrated trim moves it toward nominal.
#
#   So under imbalance the floor both repairs the one failing shape and
#   tightens calibration everywhere else.
# - **Floor geometry (Fig 16).** The left floor is set by the negatives: it
#   ends after 7–9 negative order statistics at every prevalence. The right
#   floor follows the plateau, the run of negatives below the smallest
#   positive, which lengthens as positives become scarce. At n = 1,000 the
#   suffix holds a median 38 negatives at 50%, 108 at 30% (binormal only so
#   far), and 170 at 10%.
# - **Share of the curve floored.** The floored share of the interior grows
#   accordingly, from 0.20 to 0.30 at n = 1,000.
# - **Where the misses move (Figs 17–18).** At 10% the floor removes the
#   raw band's below-band misses inside its region (0.016 → under 0.001 at
#   n = 300–1,000). The misses that remain sit outside it, as they do
#   balanced. The price is in the interior: the hybrid's above-band misses
#   outside the region run about 0.014 for n = 100–3,000, against
#   0.006–0.010 for the raw band, because the recalibrated trim spends the
#   slack the floor frees.

# %% [markdown]
# ### Fig 19 — Example bands at n = 300, per prevalence
#
# Seeded datasets at each prevalence (same seeds and DGPs across
# prevalences), with the hybrid's floor region shaded.

# %%
pp.per_prevalence(
    data=data,
    make=lambda p: [pf.fig19_example_bands(family=f) for f in pf.DGP_FAMILIES],
)
# %% [markdown]
# ## 8. Violation profile, per prevalence

# %%
pp.per_prevalence(
    data=data, make=lambda p: pf.fig20_violation_direction_location(frames[p])
)
# %%
pp.per_prevalence(data=data, make=lambda p: pf.fig21_violation_magnitude(frames[p]))
# %% [markdown]
# **High-AUC audit.** Rate of misses deeper than 5 points of TPR at true
# AUC > 0.9, 95% bands, per prevalence.

# %%
audit = {}
for p in data.prevalences:
    previous = pf.configure(prevalence=p)
    audit[pf.prevalence_label(p)] = pf.high_auc_audit(frames[p])
    pf.CONFIG = previous
pd.concat(audit, names=["prevalence", "method"]).round(4)

# %% [markdown]
# **Reading.**
#
# - **Direction.** Imbalance tilts the raw band's misses toward the truth
#   lying *below* the band: at n = 1,000 its below-band miss rate is 0.021
#   at 10% against 0.010 balanced. Its above-band rate barely moves (0.011
#   vs. 0.010).
# - **Hybrid.** The floor brings the below-band rate back to 0.008, close
#   to balanced.
# - **Depth.** The hybrid's P99 depth rises from 0.002–0.004 to
#   0.003–0.008 at 10%. It stays under one point of TPR, and deep misses
#   (> 5 points at AUC > 0.9) remain at or below 0.0015.
# - **Working–Hotelling.** Its deep-miss rate is unchanged by prevalence
#   (0.42 at n = 10,000).

# %% [markdown]
# ## 9. Cost

# %%
pp.figP23_runtime(frames)

# %% [markdown]
# **Reading.**
#
# - **Fiducial suite.** It runs about 1.2–1.5× slower at 10% (2.1 s vs.
#   1.3 s per replicate at n = 1,000; 11.6 s vs. 9.6 s at n = 10,000)
#   because its grid has n₀ + 1 points and n₀ is 0.9n instead of 0.5n.
# - **Other methods.** The closed-form methods stay in milliseconds at both
#   prevalences.

# %% [markdown]
# ## Conclusions
#
# 1. **The hybrid's calibration does not depend on prevalence.** Over
#    n = 30–10,000 it has no cell significantly below nominal at any level,
#    at 50% or at 10% prevalence. Its worst 95% cell is 0.967 at 10% against
#    0.969 balanced, and its calibration error across levels is the same or
#    slightly lower at 10%.
# 2. **Imbalance is where the floor earns its place.** Without the floor,
#    the fiducial band fails in the heteroscedastic Gaussian when the
#    positives are the more dispersed class: 0.747 coverage at n = 1,000
#    and AUC > 0.9, with most misses in the plateau corner. Imbalance leaves
#    long runs of negatives below the smallest positive, and the true curve
#    sags below the band's straight-line completion there. The floor repairs
#    it: paired on one cloud, it adds about +0.08 coverage in that DGP at
#    n = 300–1,000. Elsewhere it moves the over-covering raw band toward
#    nominal at every level. A second, more local imbalance-amplified pocket,
#    heavy-tailed Student-t (df ≈ 2) at n = 1,000, falls from about 0.90 to
#    0.83 for the raw band (Fig 3) and is also absent from the hybrid.
# 3. **The width cost of imbalance is the information cost.** Both fiducial
#    arms are 1.66–1.69× wider at 10% than balanced for n ≥ 300. That is
#    the effective-sample-size ratio √(0.25/0.09) = 1.667, so the rank-space
#    bands lose nothing beyond what fewer positives must cost.
# 4. **The envelope over-covers more under imbalance.** It stays valid,
#    loses its one sub-nominal balanced cell, and widens to 0.82× KS at
#    n = 1,000 (0.56× balanced), so its calibration error rises.
# 5. **The comparators keep their character.** M3 and KS cover about 1.00 at
#    every prevalence and level. Working–Hotelling is broken at every
#    prevalence (25+ cells significantly below nominal per level).
# 6. **The runs agree where they overlap.** The n = 1,000, 10% cell drawn in
#    both runs differs only by Monte Carlo noise (z-scores at 95%: mean
#    −0.11, SD 0.98).
# 7. **Pending.** The 30% run was partial (binormal only) when this was
#    written. Rerunning the notebook fills in the middle series, and the
#    readings above should be rechecked for monotonicity in prevalence.
