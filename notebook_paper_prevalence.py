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
# The single-run analyses of the 30% and 10% runs alone are in
# `figures/paper_nov11_p30/` and `figures/paper_nov11_p10/`.
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
# All three runs are complete: 6,000 replicates per DGP × n from n = 30
# (5,995 for Student-t, whose few unachievable LHS draws are dropped
# identically in every run), so every average uses the same cells at every
# prevalence.

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
# **Reading.** Every metric moves monotonically from 50% through 30% to
# 10%, and most of the movement happens between 30% and 10%.
#
# - **Hybrid.** Calibration is unchanged. Mean 95% coverage is 0.970–0.993
#   at 10% against 0.970–0.992 at 50% (0.971–0.991 at 30%), and the worst
#   cell is 0.967, 0.969, and 0.969 at 10%, 30%, and 50%. Calibration error
#   averaged over the three levels is slightly *lower* under imbalance from
#   n = 300 (0.114, 0.111, and 0.101 at n = 1,000 for 50%, 30%, and 10%).
# - **Raw fiducial band.** Its mean barely moves, but its worst cell falls:
#   at n = 1,000 it is 0.966 balanced, 0.950 at 30%, and 0.893 at 10%. At
#   10% it stays below 0.95 from n = 100 to 3,000; at 30% only n = 300
#   (0.939) is below. The floor exists for exactly this case.
# - **Envelope.** It turns *more* conservative under imbalance: coverage
#   rises (0.984, 0.986, 0.990 at n = 1,000), its one sub-nominal cell from
#   the balanced run is already gone at 30%, and calibration error and area
#   relative to KS both worsen (area ÷ KS 0.56, 0.62, 0.82 at n = 1,000).
# - **M3 and KS.** Coverage stays at about 1.00 at every prevalence.
# - **Working–Hotelling.** It covers *more* under imbalance (0.54, 0.55,
#   0.62 mean at n = 1,000) because its band widens, but it still collapses
#   with n.
#
# Every method's band is wider under imbalance (bottom row). This follows
# from the per-class sample sizes, n₀ = (1 − p)n and n₁ = pn, not from any
# method's design.

# %% [markdown]
# ### Fig P2 — Coverage shift against the balanced run
#
# Coverage at 30% and 10% prevalence minus coverage at 50%, per DGP and n,
# at 95%. The runs are independent, so each shift carries the Monte Carlo
# error of both cells: about ±0.006 (1.96 SE) at coverage 0.97.

# %%
pp.figP02_coverage_shift(cells)

# %% [markdown]
# **Reading.**
#
# - **Hybrid.** Every shift lies within ±0.009 at 10% and ±0.005 at 30%;
#   the mean is −0.002 at 10%, inside the Monte Carlo band.
# - **M3.** Shifts lie within −0.003 to 0.000.
# - **Raw fiducial band.** The heteroscedastic-Gaussian row is the only
#   structure. At 10% it is −0.026 at n = 30, −0.046 at 100, −0.064 at 300,
#   −0.073 at 1,000, −0.051 at 3,000, and −0.025 at 10,000. At 30% the
#   same row shows the same shape at a fifth to a third of the depth:
#   −0.006, −0.013, −0.022, −0.015, −0.012, −0.003. Every other cell lies
#   within about ±0.006 at both prevalences.
# - **Envelope.** It shifts upward, most at n = 10,000: +0.029 in the
#   heteroscedastic Gaussian and +0.007 to +0.016 elsewhere at 10%, and
#   +0.020 and up to +0.009 at 30%.

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
#   at any prevalence. The raw fiducial band has none at 50%, one at 30%
#   (heteroscedastic Gaussian, n = 300, 95%: 0.939), and five at 10%, all
#   heteroscedastic Gaussian: at 95% for n = 100, 300, 1,000, and 3,000
#   (worst 0.893), and at 80% for n = 300. Working–Hotelling has 25–29 per
#   level at every prevalence.
# - **Area.** The hybrid's area ratio to KS is 0.50, 0.51, and 0.59 at
#   n = 1,000 for 50%, 30%, and 10%. In absolute terms its mean area is
#   1.09–1.10× its balanced area at 30% and 1.67–1.69× at 10% for every
#   n ≥ 300.
# - **Effective sample size.** Those ratios are the effective-sample-size
#   prediction: band width scales as 1/√(n·p(1−p)), giving √(0.25 / 0.21) =
#   1.091 at 30% and √(0.25 / 0.09) = 1.667 at 10%. Both fiducial arms sit
#   at 1.09–1.10 and 1.66–1.67 for n ≥ 1,000. KS is less sensitive (1.41–1.45
#   at 10%) and the envelope more (up to 2.07). Imbalance costs the
#   rank-space bands exactly what the information loss predicts, and no
#   more.
# - **30% is close to balanced in information terms.** It costs only 9% in
#   width, yet the raw band already shows its first significant failure
#   there. The raw band's defect tracks the length of the run of negatives
#   below the smallest positive (Fig 16), which grows faster than the
#   information loss.

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
# - **Hybrid.** Its three columns are near copies at every level. At 10% it
#   spans 0.967–0.996 at 95%, 0.862–0.968 at 80%, and 0.624–0.893 at 50%;
#   at 30%, 0.969–0.993, 0.871–0.953, and 0.635–0.860. The sup-norm
#   conservatism at the lower levels shrinks with n the same way at every
#   prevalence.
# - **Raw fiducial band.** Its heteroscedastic-Gaussian row is the only one
#   that changes: pale at 30%, red at 10%.
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
# - **Hybrid.** Under imbalance it is as flat in AUC as it is balanced:
#   every DGP and n stays within 0.955–0.99 at every prevalence, with no
#   high-AUC droop.
# - **Raw fiducial band, heteroscedastic Gaussian.** At 10% the high-AUC
#   end gives way. Coverage falls to about 0.73 at AUC 0.98 for n = 1,000,
#   with a dip to about 0.84 around AUC 0.87 at n = 100 and to 0.87 at the
#   top for n = 10,000. At 30% the droop is already visible but shallower:
#   about 0.86–0.88 above AUC 0.95 at n = 300–1,000, against 0.91–0.95
#   balanced.
# - **Raw fiducial band, Student-t.** It slips to 0.94 at the highest AUCs
#   at 10%; at 30% it matches the balanced run.
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
# **Reading.** Under imbalance the raw fiducial band's balanced pockets
# deepen.
#
# - **Heteroscedastic Gaussian.** Once σ₊/σ₋ exceeds about 2, coverage falls
#   to about 0.80 at n = 100 and 0.81 at n = 1,000 at 10% (about 0.91 and
#   0.95 balanced). At 30% the pocket is halfway there: binned over
#   σ₊/σ₋ = 3–5 it is 0.95, 0.92, and 0.86 at n = 100 and 0.95, 0.92, and
#   0.82 at n = 1,000 for 50%, 30%, and 10%.
# - **Student-t.** The df ≈ 2 dip at n = 1,000 goes from about 0.90 to 0.83
#   at 10%. This second, more local failure is also in the tails, and the
#   per-cell tables average it away. At 30% it is no deeper than balanced,
#   so this pocket needs stronger imbalance than the heteroscedastic one.
# - **Hybrid.** It stays flat at or above 0.95 along every shape axis at
#   every prevalence. The one touch is the heteroscedastic Gaussian at
#   σ₊/σ₋ ≈ 0.2 and n = 100, at 0.948 with a wide ribbon.
# - **Envelope.** It loses its balanced dip below nominal at large σ₊/σ₋
#   and n = 10,000 (≈ 0.935 → 0.96 at 30% → 0.97 at 10%), consistent with
#   its general drift toward over-coverage under imbalance.
# - **M3, KS, and Working–Hotelling.** They look the same at every
#   prevalence: flat at 1.00 for the first two, collapsing off-model for
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
#   five are heteroscedastic-Gaussian cells. Their 30% midpoints sit near
#   zero; one (n = 300, 0.939) is already filled.
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
#   0.937 balanced and 0.865 and 0.896 at 30%. Its plateau-region
#   (FPR 90–100%) violation rate is 0.21–0.24 at 10%, 0.09–0.13 at 30%, and
#   0.05–0.07 balanced. Even balanced it was already below nominal in this
#   corner; imbalance multiplies the defect.
# - **Recovery at large n.** Coverage recovers as n grows (0.896 at 10% and
#   0.952 at 30% for n = 10,000) as the per-class counts outgrow the hook.
# - **Positives tighter or similar.** There is no failure at either
#   prevalence. The bulge at FPR = 0 is resolved by the positives above
#   the largest negative, and imbalance leaves fewer positives, not more
#   unresolved negatives, at that end.
# - **Hybrid.** It holds 0.96–0.99 in every column at every prevalence,
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
# analyses (`figures/paper_nov11/`, `figures/paper_nov11_p30/`,
# `figures/paper_nov11_p10/`).

# %%
for dgp in pf.SHAPE_BINS:
    pp.figP27_shape_auc_coverage(frames, dgp=dgp)

# %% [markdown]
# **Reading.** Imbalance widens the raw band's failure region in both
# families and leaves the hybrid where it was. The two families respond
# differently to moderate imbalance.
#
# - **Heteroscedastic Gaussian, σ₊/σ₋ = 2–5.** The raw band falls step by
#   step with prevalence, pooled over n (50% → 30% → 10%): 0.981 → 0.976 →
#   0.931 (AUC 0.50–0.75), 0.972 → 0.961 → 0.922, 0.967 → 0.947 → 0.878,
#   and 0.945 → 0.905 → 0.829 (AUC > 0.95). At 10% it falls in every AUC
#   bin, not only the highest. Per n, the worst points are 0.80 at 30%
#   (AUC > 0.95, n = 300) and 0.64 at 10% (AUC > 0.95, n = 1,000). With
#   σ₊/σ₋ ≤ 2 there is no shift.
# - **Student-t, heavy tails at high AUC.** At df 1–2 and AUC > 0.95 the raw
#   band is 0.872 balanced, 0.888 at 30%, and 0.795 at 10% pooled. Per n it
#   reaches 0.63, 0.56, and 0.48 at n = 300, and 0.58 at n = 1,000 for 10%.
#   At df 2–4 and AUC > 0.95 it is 0.916, 0.911, and 0.857, with 0.67 at
#   n = 1,000 and 0.82 even at n = 10,000 for 10%. At df ≥ 4 nothing moves.
#   The heavy-tail failure is already present balanced and changes little
#   at 30%; only strong imbalance deepens it.
# - **Hybrid.** It stays at 0.97–0.99 pooled in every panel at every
#   prevalence. Per n, none of its points has a Wilson upper bound below
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
#   imbalance costs width, not coverage. The 50% → 30% step is short and
#   the 30% → 10% step long, as the effective-sample-size ratios (1.09 and
#   1.67) predict.
# - **Hybrid and raw fiducial.** At n ≥ 1,000 their traces lie on top of
#   each other, and both stay the tightest calibrated methods at every
#   prevalence.
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
#   0.950 → 0.893 at n = 1,000).
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
# - **Width scale (Fig 9).** The panels shift up as a whole under
#   imbalance, slightly at 30% and clearly at 10%: every method is wider
#   relative to WH and KS at small n.
# - **Width profile (Fig 10).** Under imbalance the hybrid's profile
#   flattens on the right. At n = 100 and 10% its width is 0.45 from FPR 0.5
#   to 0.9, matching M3 at FPR 0.9, because the plateau floor covers most of
#   the curve there; at 30% it is 0.27 at FPR 0.5 and 0.20 at 0.9. At
#   n = 1,000 and FPR 0.9 it is 0.072 at 10%, against 0.047 for the raw band
#   and 0.084 for M3 (0.032, 0.027, and 0.042 at 30%).
# - **Large n.** At n = 10,000 the hybrid and the raw band agree to within
#   0.002 at every landmark and prevalence.

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
# - **Plateau corner under imbalance.** For n = 100–3,000 the raw band's
#   plateau-corner rate rises to 0.005–0.009 at 30% and 0.010–0.017 at 10%,
#   while the hybrid's stays at 0.001–0.003.
# - **Mean coverage under imbalance.** The raw band's mean coverage drops
#   below the hybrid's from n = 300. At n = 1,000 it is 0.977 vs. 0.979 at
#   30% and 0.968 vs. 0.978 at 10%; at AUC > 0.9, 0.971 vs. 0.977 and 0.955
#   vs. 0.977.
# - **Worst cell (Figs 11, 14).** The raw band reaches 0.939 at 30% and
#   0.893 at 10%; the hybrid stays at 0.967 or above.
# - **M3 alone (Fig 13).** It stays valid and wide at every prevalence. Its
#   50% band covers 0.99 balanced, 0.98 at 30%, and still 0.95 at 10%, so it
#   remains nearly blind to the nominal level.

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
#   +0.006 to +0.009 and +0.008 to +0.011). At 30% the pooled gain is
#   +0.002 to +0.004 for n = 300–1,000, and balanced it stays within
#   ±0.003. The pooled number hides two opposite effects:
#   - In the heteroscedastic Gaussian the floor adds +0.04 to +0.08 at 95%
#     for n = 30–3,000 at 10%, and +0.05 to +0.11 at 80%. At 30% it adds
#     +0.01 to +0.04 at 95% (largest at n = 300) and up to +0.04 at 80%.
#   - In the other six DGPs the hybrid covers about 0.005 *less* at 95% and
#     0.01–0.07 less at 80% and 50%, at every prevalence. Where the raw band
#     over-covers, the recalibrated trim moves it toward nominal.
#
#   So under imbalance the floor both repairs the one failing shape and
#   tightens calibration everywhere else.
# - **Floor geometry (Fig 16).** The left floor is set by the negatives: it
#   ends after 7–9 negative order statistics at every prevalence. The right
#   floor follows the plateau, the run of negatives below the smallest
#   positive, which lengthens as positives become scarce. At n = 1,000 the
#   suffix holds a median 38 negatives at 50%, 69 at 30%, and 170 at 10%:
#   about 1.8× and 4.5× the balanced run, against width ratios of 1.09 and
#   1.67. This faster growth is why the raw band's corner defect outpaces
#   the information loss.
# - **Share of the curve floored.** The floored share of the interior grows
#   accordingly: 0.20, 0.22, and 0.30 at n = 1,000.
# - **Where the misses move (Figs 17–18).** Under imbalance the floor
#   removes the raw band's below-band misses inside its region (0.007–0.009
#   at 30% and 0.016 at 10% → under 0.001 at n = 300–1,000). The misses that
#   remain sit outside it, as they do balanced. The price is in the
#   interior: the hybrid's above-band misses outside the region run about
#   0.011–0.013 at 30% and 0.014 at 10% for n = 100–3,000, against
#   0.006–0.011 for the raw band, because the recalibrated trim spends the
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
#   lying *below* the band: at n = 1,000 its below-band miss rate is 0.010
#   balanced, 0.012 at 30%, and 0.021 at 10%. Its above-band rate barely
#   moves (0.010–0.011).
# - **Hybrid.** The floor holds the below-band rate at 0.008–0.009 at every
#   prevalence.
# - **Depth.** The hybrid's P99 depth is 0.002–0.004 balanced and at 30%,
#   and 0.003–0.008 at 10%. It stays under one point of TPR, and deep misses
#   (> 5 points at AUC > 0.9) remain at or below 0.0015.
# - **Working–Hotelling.** Its deep-miss rate is unchanged by prevalence
#   (0.42–0.43 at n = 10,000).

# %% [markdown]
# ## 9. Cost

# %%
pp.figP23_runtime(frames)

# %% [markdown]
# **Reading.**
#
# - **Fiducial suite.** It runs about 1.2–1.5× slower at 10% (2.1 s vs.
#   1.3 s per replicate at n = 1,000; 11.6 s vs. 9.6 s at n = 10,000)
#   because its grid has n₀ + 1 points and n₀ is 0.9n instead of 0.5n. At
#   30% it sits in between (1.7 s and 10.3 s).
# - **Other methods.** The closed-form methods stay in milliseconds at every
#   prevalence.

# %% [markdown]
# ## Conclusions
#
# 1. **The hybrid's calibration does not depend on prevalence.** Over
#    n = 30–10,000 it has no cell significantly below nominal at any level
#    at 50%, 30%, or 10% prevalence. Its worst 95% cell is 0.969, 0.969, and
#    0.967, and its calibration error across levels is the same or slightly
#    lower under imbalance.
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
#    0.83 for the raw band at 10% (Fig 3) and is also absent from the
#    hybrid.
# 3. **The raw band's defect grows faster than the imbalance.** At 30% the
#    heteroscedastic-Gaussian failure is a fifth to a third as deep as at
#    10% and already yields one significant cell (0.939); the Student-t
#    pocket is no deeper than balanced. The defect tracks the plateau run of
#    negatives below the smallest positive, which grows 1.8× at 30% and
#    4.5× at 10% while the width grows only 1.09× and 1.67×.
# 4. **The width cost of imbalance is the information cost.** Both fiducial
#    arms are 1.09–1.10× wider at 30% and 1.66–1.69× wider at 10% than
#    balanced for n ≥ 300. Those are the effective-sample-size ratios
#    √(0.25/0.21) = 1.091 and √(0.25/0.09) = 1.667, so the rank-space bands
#    lose nothing beyond what fewer positives must cost.
# 5. **The envelope over-covers more under imbalance.** It stays valid,
#    loses its one sub-nominal balanced cell already at 30%, and widens to
#    0.62× KS at 30% and 0.82× at 10% for n = 1,000 (0.56× balanced), so its
#    calibration error rises.
# 6. **The comparators keep their character.** M3 and KS cover about 1.00 at
#    every prevalence and level. Working–Hotelling is broken at every
#    prevalence (25+ cells significantly below nominal per level).
# 7. **The runs agree where they overlap.** The n = 1,000, 10% cell drawn in
#    both runs differs only by Monte Carlo noise (z-scores at 95%: mean
#    −0.11, SD 0.98).
