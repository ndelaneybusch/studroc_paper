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
# # Simultaneous Confidence Bands for ROC Curves — Paper Analysis (nov11 run)
#
# This notebook builds every figure and table for the paper from the
# `data/results/nov11` simulation run: 6,000 maximin-LHS parameter draws per
# DGP, seven DGPs, n = 10 to 10,000 (balanced, plus 10% prevalence at
# n = 1,000), 50/80/95% bands, and six methods — about 6.05M band
# evaluations. See `stats/simulation_spec.md` for the design.
#
# **The method under study** is the rank-space **fiducial band with the
# localized exact M3 floor** (`fiducial_hybrid`): the C = 1 fiducial band,
# with the exact M3 band's hull taken on a data-determined region at the
# low-FPR corner and on the TPR plateau, and the trim recalibrated so the
# floor does not double-spend the level. It is compared with
#
# - the **raw fiducial band** (`fiducial`), trimmed from *the same fiducial
#   cloud* in every replicate, so the two are paired exactly;
# - the **M3 exact band** (`m3`), which carries a finite-sample,
#   distribution-free coverage theorem;
# - the **studentized bootstrap envelope** (`envelope`), the previous run's
#   method under study;
# - the **KS fixed-width** and **Working–Hotelling** baselines.
#
# **What this run adds over the June run**, and what the new analyses use:
#
# 1. *Pairing.* The two fiducial arms share one cloud per replicate, so the
#    effect of the floor is measured replicate by replicate (Fig 15) with
#    McNemar-type intervals rather than as a difference of two noisy rates.
# 2. *Floor-region records.* Every replicate stores where the floor operated
#    and splits each method's misses into inside and outside that region,
#    above and below — Figs 16–18 and 19d, all new.
# 3. *An exact-guarantee comparator.* M3 is the first method in the study
#    with a finite-sample theorem; Fig 24 checks every cell against it.
# 4. *Native-grid evaluation* (K = n₀ + 1, read at the top of each jump), so
#    rank-space bands are evaluated exactly rather than after resampling.
# 5. *Per-replicate timing and device*, for a cost comparison (Fig 23).
# 6. *6,000 LHS draws per DGP* (1.5× the June run), which tightens the GAM
#    ribbons in Figs 2, 3, and 7.
#
# The figure code lives in `stats/paper_figures_nov11.py` (one function per
# figure); this notebook calls it and interprets the results. Figures are
# written to `figures/paper_nov11/` as PNG, PDF, and SVG.
#
# **Regenerating** (figures only, no commentary):
# ```
# uv run python stats/paper_figures_nov11.py
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

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200)
pf.set_style()

# %% [markdown]
# ## Data loading and inventory

# %%
meta = pf.load_run_metadata()
print("Run:", meta["timestamp"], "| git", meta["git_hash"][:10])
print("Methods:", ", ".join(meta["methods"]))
print("Hybrid floor:", meta["hybrid_floor"])

df = pf.load_results()
cells = pf.cell_summary(df)
paired = pf.paired_fiducial(df)
print(f"\n{len(df):,} trial rows, {df['method'].nunique()} methods")

inventory = (
    df[(df["method"] == pf.HERO) & (df["alpha"] == 0.05)]
    .groupby(["dgp_type", "n_total", "prevalence"], observed=True)
    .size()
    .unstack(["n_total", "prevalence"])
)
print("\nReplicates per DGP × (n, prevalence):")
print(inventory.to_string())

# %% [markdown]
# Every DGP × configuration cell holds 6,000 replicates, except Student-t,
# where 5 of the 6,000 LHS draws were filtered for an unachievable AUC
# (5,995 per cell). Pooled numbers therefore weight the DGPs essentially
# equally without reweighting, and the macro-averages used throughout weight
# them exactly equally.

# %% [markdown]
# ---
# # 1. Calibration of the hybrid
#
# **Motivation.** The headline claim is that the hybrid is calibrated across
# distribution families, sample sizes, AUC levels, and shape parameters,
# without distributional assumptions.
#
# **Expectations.** At small n the floor region covers the whole curve, so
# the hybrid *is* M3 and inherits its conservatism (coverage ≈ 1.0 at
# n = 10). As n grows the region shrinks to the two corners and the hybrid
# approaches the raw fiducial band, which over-covers by a few points at 95%
# and by more at lower levels (sup-norm bands are insensitive to α).

# %% [markdown]
# ### Fig 1 — Coverage by DGP × configuration

# %%
pf.fig01_coverage_heatmaps(
    cells=cells, alphas=[0.05], name="fig01a_hybrid_coverage_heatmap_95"
)

# %%
pf.fig01_coverage_heatmaps(
    cells=cells, alphas=[0.2, 0.5], name="fig01b_hybrid_coverage_heatmap_50_80"
)

# %% [markdown]
# **Reading Fig 1.** At 95% every cell lies between 0.969 and 1.000, and the
# pattern is the same in every DGP: 1.000 at n = 10 (the hybrid is M3
# there), sliding to ≈ 0.97 at n = 10,000. No cell is red. The 80% and 50%
# panels show the expected sup-norm conservatism — 0.87–0.92 at 80% and
# 0.63–0.74 at 50% for n ≥ 100 — which shrinks with n. The family barely
# matters: at a given n the seven rows agree to within about a point.

# %% [markdown]
# ### Fig 2 — Coverage vs. true AUC
#
# **Motivation.** High AUC creates a steep low-FPR corner where order-
# statistic effects dominate; it is the regime that broke the bare bootstrap
# and that the floor's left region is built for. Ribbons are pointwise 95%
# logistic-GAM bands.

# %%
pf.fig02_coverage_vs_auc(df)

# %% [markdown]
# **Fig 2b–f — The same view for every comparator.** One figure per
# method, same layout and ribbons; the y-axis extends below 0.88 only when a
# method's ribbons do.

# %%
for method in pf.METHOD_ORDER:
    if method != pf.HERO:
        pf.fig02_coverage_vs_auc(df, method=method)

# %% [markdown]
# **Reading Fig 2b–f.** The plain fiducial band tracks the hybrid except
# in the heteroscedastic Gaussian at high AUC, where it dips to ≈ 0.92 at
# n = 1,000 and to ≈ 0.955 at n = 100. That is the low-FPR corner the floor
# exists for. M3 and KS sit at ≈ 1.00 everywhere: valid but uninformative
# about level. The studentized envelope drifts down with n, falling below
# nominal at low AUC by n = 10,000 (≈ 0.87 in the heteroscedastic Gaussian).
# Working–Hotelling holds nominal only in its own model family (binormal and
# heteroscedastic Gaussian). Elsewhere it collapses with both n and AUC.

# %% [markdown]
# ### Fig 3 — Coverage vs. DGP shape parameters
#
# Within each family the LHS design sweeps a shape parameter (tail weight,
# skew, heteroscedasticity, multimodality). A calibrated method is flat in
# these nuisance directions.

# %%
pf.fig03_coverage_vs_shape(df)

# %% [markdown]
# **Fig 3b–f — The same view for every comparator.** One figure per method;
# the y-axis extends below 0.85 only when a method's curves do.

# %%
for method in pf.METHOD_ORDER:
    if method != pf.HERO:
        pf.fig03_coverage_vs_shape(df, method=method)

# %% [markdown]
# **Reading Fig 3b–f.** The shape sweeps locate what the per-cell tables
# average away.
#
# - **Raw fiducial band.** It has two pockets, both in the tails the floor
#   protects. Very heavy-tailed Student-t (df near 1) falls to about 0.76
#   at n = 100, and df ≈ 2–3 dips to about 0.90 at n = 1,000. The extreme σ
#   ratios of the heteroscedastic Gaussian fall to about 0.91 at n = 100.
# - **Envelope.** It is flat at small n but drifts down toward nominal by
#   n = 10,000, and dips below it (≈ 0.935) in the heteroscedastic Gaussian
#   when σ₊/σ₋ > 2.
# - **M3 and KS.** Both sit at ≈ 1.00 across every shape parameter.
# - **Working–Hotelling.** It is flat only along the σ ratio, its own model
#   family. It collapses along every other axis: tail weight, latent σ,
#   skew, mode separation, and Weibull shape.

# %% [markdown]
# ### Fig 4 — Calibration across confidence levels
#
# Empirical vs. nominal coverage, averaged over DGPs, for every method. The
# identity line is perfect calibration.

# %%
pf.fig04_level_calibration(cells)

# %% [markdown]
# **Reading Fig 4.** M3 and KS sit near 1.0 at every requested level: they
# are valid but level-blind. Working–Hotelling sits below the identity on
# average because it fails off-model. The two fiducial arms and the envelope
# track the identity best, with the gap at 50% narrowing as n grows. At
# n = 10 the hybrid coincides with M3.

# %% [markdown]
# ---
# # 2. Robustness: the hybrid vs. Working–Hotelling
#
# **Expectations.** WH ≈ nominal on the two DGPs where the binormal ROC
# model holds (binormal, heteroscedastic Gaussian) and collapses elsewhere,
# worse with n because its band tightens around a biased curve. The
# rank-based methods are invariant to monotone score transforms, so their
# behavior should be indistinguishable across families.

# %% [markdown]
# ### Fig 5 — Side-by-side 95% coverage heatmaps

# %%
pf.fig05_hybrid_vs_wh_heatmap(cells)

# %% [markdown]
# ### Fig 6 — Coverage vs. n, every method

# %%
pf.fig06_coverage_vs_n(cells)

# %% [markdown]
# **Reading Figs 5–6.** WH holds 0.954–0.977 on the two Gaussian DGPs at
# every n, and falls to 0.011 (Weibull), 0.021 (Student-t), 0.126
# (logit-normal), 0.228 (bimodal), and 0.401 (beta) at n = 10,000. Every
# rank-based method is flat across families.

# %% [markdown]
# ### Fig 7 — How far from binormal before WH breaks?

# %%
pf.fig07_wh_fragility(df)

# %% [markdown]
# ### Fig 25 — Coverage by AUC and shape bin: Student-t and heteroscedastic Gaussian
#
# **Motivation.** The experimental studies found the raw fiducial band fails
# where the ROC has a convex hook at an end, which heavy tails (Student-t,
# low df) and unequal class spreads (heteroscedastic Gaussian) both produce.
# Within a DGP, those draws are a minority of the LHS design, so per-cell
# averages dilute them. Here each panel is one shape-parameter bin (rows) ×
# one true-AUC bin (columns); methods run along x, with n increasing left to
# right inside each method. Bars are 95% Wilson intervals. Points below 0.70
# are pinned to the floor with their value printed. The panel's footnote
# gives the LHS draws behind each point: heavy tails (df < 2) are only about
# 3% of the Student-t design, so those panels are noisy.

# %%
for dgp in pf.SHAPE_BINS:
    pf.fig25_shape_auc_coverage(df, dgp=dgp)

# %% [markdown]
# **Reading Fig 25.** The raw fiducial band fails exactly where the
# experiments said it would, and nowhere else in these two families.
#
# - **Student-t.** Its failures sit at df < 4 and AUC > 0.9. At df 1–2 and
#   AUC > 0.95 it covers 0.81 at n = 100, 0.64 at n = 300, and 0.87 at
#   n = 1,000. At df 2–4 and AUC > 0.95 it covers 0.83 at n = 300 and 0.78
#   at n = 1,000. At df ≥ 4 it is within noise of the hybrid.
# - **Heteroscedastic Gaussian.** The failures sit at σ₊/σ₋ > 2 and
#   AUC > 0.95: 0.89 at n = 300 and 0.91 at n = 1,000.
# - **Hybrid.** It covers 0.92–1.00 in every panel. Its lowest point, 0.92
#   at df 1–2, AUC > 0.95, n = 10,000, rests on 52 draws, and no hybrid
#   point has a Wilson upper bound below 0.95.
# - **Envelope.** It drifts toward or below nominal at n = 10,000 in the
#   heteroscedastic Gaussian with σ₊/σ₋ > 2 and AUC < 0.9 (0.89–0.93).
# - **Working–Hotelling.** It is near zero across Student-t at n ≥ 300.
# %% [markdown]
# ---
# # 3. Tightness
#
# **Motivation.** Coverage is cheap if the band is vacuous. The claim: the
# hybrid is far tighter than KS and M3 while staying calibrated, and pays
# for distribution-freeness only a moderate premium over WH.

# %% [markdown]
# ### Fig 8 — Coverage vs. mean band area (macro over DGPs)
#
# Pooled over DGPs, so the spread hides real heterogeneity (WH runs from
# ≈ 0.95 to ≈ 0.01). Fig 8b gives per-cell points with honest CIs.

# %%
pf.fig08_pareto(cells)

# %% [markdown]
# ### Fig 8b — Per-DGP Pareto with 95% CIs

# %%
for family in pf.DGP_FAMILIES:
    pf.fig08b_pareto_ci(cells=cells, family=family)

# %% [markdown]
# ### Fig 8c — Tightening trajectory over n, hybrid vs. WH

# %%
pf.fig08c_trajectory(df)

# %% [markdown]
# ### Fig 9 — Width on the WH→KS scale and calibration error

# %%
pf.fig09_width_scale_and_calibration(cells)

# %% [markdown]
# **Reading Fig 9.** On the WH→KS scale the hybrid starts with M3 at small
# n and joins the raw fiducial band by n ≈ 1,000; the envelope gets there by
# n ≈ 10,000. As plain area ratios (headline numbers below), the hybrid's
# 95% band is 0.91× KS at n = 10, 0.50× at n = 1,000, and 0.45× at
# n = 10,000, and 0.68× M3 at n ≥ 1,000 — M3's finite-sample theorem costs
# about 1.5× the area. On calibration error at 95% the two fiducial arms
# and the envelope are close; the envelope is best at n ≥ 3,000, where its
# coverage (0.944–0.962) sits nearest 0.95. KS and M3 overlap at 0.05
# because both cover ≈ 1.0.

# %% [markdown]
# ### Fig 10 — Width across the FPR axis

# %%
pf.fig10_width_profile(df)

# %% [markdown]
# ---
# # 4. Ablations: the hybrid without its floor, and the floor alone
#
# **Motivation.** The hybrid has two components, and the run contains each
# on its own: the **raw fiducial band** is the hybrid with the floor
# removed, and **M3** is the floor's band applied over the whole curve
# instead of a localized region. This section asks what each component
# contributes, in the same four views as the June envelope ablations.
#
# **Expectations.** Unlike the bare bootstrap envelope, the raw fiducial
# band is a valid-looking band on its own: no collapse at the corners, so
# its average coverage should be close to the hybrid's. What the floor
# should buy is protection where the fiducial calibration is weakest —
# the corners at small n, and the worst DGP × configuration cell. M3 alone
# should be valid everywhere (it has a theorem) but wide, and its width
# should barely respond to the requested level.

# %% [markdown]
# ### Fig 11 — Without the floor
#
# Left: 95% coverage of the three arms vs. n, averaged over DGPs (solid) and
# in the worst DGP × configuration cell, including 10% prevalence (dotted).
# Right: violation rate by FPR region × n for the raw band and the hybrid.

# %%
pf.fig11_no_floor_ablation(df=df, cells=cells)

# %% [markdown]
# **Reading Fig 11.** On average the two fiducial arms are indistinguishable
# at 95% (0.97–1.00 at every n) — the raw band is not broken the way the
# bare bootstrap was. The difference is in the tails of the distribution of
# cells: the raw band's worst cell falls to 0.961 at n = 300 and 0.900 at
# n = 1,000 (heteroscedastic Gaussian, 10% prevalence), while the hybrid's
# worst cell never drops below 0.969. The region maps show where: without
# the floor, the 90–100% (plateau) region carries 0.002–0.005 of misses at
# n = 30–1,000, which the hybrid reduces to 0.000–0.002, and at n ≤ 100 the
# 0–10% region is likewise cleared. The maps also show the other side of
# the trade: from n = 100 the hybrid misses more often in FPR 10–30%
# (0.007–0.008 vs. 0.003–0.005), where the recalibrated trim spends the
# level the floor frees (Fig 18).

# %% [markdown]
# ### Fig 12 — Corner ablation
#
# Same color family for the two fiducial arms, since they are variants of
# one band. The corner panels attribute the floor's effect by region.

# %%
pf.fig12_corner_ablation(df)

# %% [markdown]
# **Reading Fig 12.** The floor's clearest effect is at the TPR plateau:
# without it, 0.4–0.5% of replicates miss in FPR 90–100% at n = 100–1,000
# (0.6–1.0% at AUC > 0.9); with it, 0.01–0.2%. At the steep corner the
# floor clears the 0–10% region at n ≤ 100, but for n ≥ 300 the two arms
# miss there equally (≈ 0.4–1.1%): the left floor now ends within the first
# few order statistics (Fig 16), so most of the 0–10% region lies outside
# it. M3 misses essentially nowhere.

# %% [markdown]
# ### Fig 13 — The floor alone is not the method
#
# M3 is what the hybrid would be if the floor region were the whole curve.

# %%
pf.fig13_floor_only_ablation(cells)

# %% [markdown]
# **Reading Fig 13.** M3 alone is safe at every n and level — ≈ 1.0 at 95%
# and ≈ 0.98 at 50% — but that is the problem: a 50% band that covers
# 98% of the time carries almost no information about the level. Its area
# is 1.4–1.5× the hybrid's at n ≥ 300, and its 50% band is still
# 0.73–0.75× the area of its 95% band, vs. 0.62–0.69× for the hybrid at
# n ≥ 100. At n ≤ 30 the hybrid *is* M3, because the floor region covers
# the whole curve.

# %% [markdown]
# ### Fig 14 — Neither component alone
#
# The three arms on the three criteria that separate them: worst-cell
# coverage (leakiness), area relative to M3 (width), and calibration error
# averaged over the three levels (tunability).

# %%
pf.fig14_single_component_summary(cells)

# %% [markdown]
# **Reading Fig 14.** Each single component fails one criterion. The raw
# fiducial band is the narrowest and well calibrated on average, but its
# worst cell is significantly below nominal. M3 never leaks but is
# 1.3–1.5× wider than the hybrid for n ≥ 100 and has a flat calibration
# error of ≈ 0.24 because it ignores the requested level. The hybrid
# matches M3's leak-resistance at 95% (worst cell ≥ 0.969) and the raw
# band's calibration from n ≈ 300 (error 0.08–0.12 vs. 0.09–0.13 for the
# raw band) and its width from n ≈ 1,000. The price is paid at small n: for
# n ≤ 100 the hybrid takes M3's width at low FPR (0.85 vs. 0.61 at
# FPR = 0.05, n = 100).

# %% [markdown]
# ---
# # 5. What the floor buys — paired on one fiducial cloud (new)
#
# **Motivation.** The hybrid differs from the raw fiducial band only by the
# floor. Because both arms are trimmed from the same cloud in every
# replicate, the effect of the floor can be measured on each replicate
# directly: the coverage difference is driven only by the *discordant*
# replicates (one arm covers, the other misses), and its standard error is
# the McNemar one, far smaller than for two independent rates.

# %% [markdown]
# ### Fig 15 — Paired coverage difference, hybrid − raw fiducial

# %%
pf.fig15_paired_floor_effect(paired)

# %%
pooled = []
for (alpha, n, prev), g in paired.groupby(
    ["alpha", "n_total", "prevalence"], observed=True
):
    diff, lo, hi, disc = pf.paired_difference(
        a=g["covers_entirely_hybrid"].to_numpy(),
        b=g["covers_entirely_fiducial"].to_numpy(),
    )
    pooled.append(
        dict(
            alpha=alpha, n=n, prevalence=prev, diff=diff, lo=lo, hi=hi, discordant=disc
        )
    )
print("Hybrid − raw fiducial coverage, pooled over DGPs (paired 95% CI):")
print(pd.DataFrame(pooled).round(4).to_string(index=False))

# %% [markdown]
# **Reading Fig 15.** Three regimes.
#
# - **n ≤ 30:** the floor covers (nearly) the whole curve, so the hybrid is
#   M3 and covers more: +1.1pp at 95%, +15pp at 50% (n = 10).
# - **Balanced n ≥ 100:** at 95% the arms differ by under ±0.3pp; at 80% and
#   50% the hybrid covers *less* (−0.5 to −4pp), i.e. closer to nominal. The
#   recalibrated trim hands back the slack the floor creates (Fig 18).
# - **n = 1,000 at 10% prevalence:** the one cell where the floor is a
#   rescue rather than a refinement. With only 100 positives the raw band's
#   lower edge misses on heteroscedastic Gaussian data (0.900 at 95%, the
#   only fiducial cell significantly below nominal); the hybrid restores it
#   to 0.974.

# %% [markdown]
# ### Fig 16 — Geometry of the floor region
#
# **Motivation.** The floor's left region ends at a cutoff fixed by n₀, the
# cloud size, and the realized trim depth; the right region spans the TPR
# plateau. How much of the curve does each own, and how does that scale?

# %%
pf.fig16_floor_region_geometry(df)

# %%
hero = pf.primary(df)[lambda d: d["method"] == pf.HERO]
left_k = (
    hero.assign(k=hero["floor_region_left_end_fpr"] * hero["n_neg"])
    .groupby(["alpha", "n_neg"])["k"]
    .median()
    .unstack()
)
print("Median grid index where the left floor ends (k_left):")
print(left_k.round(1).to_string())
print("\nMean share of interior grid points inside the floor region:")
print(
    hero.groupby(["alpha", "n_total"])["floor_region_interior_fraction"]
    .mean()
    .unstack()
    .round(3)
    .to_string()
)

# %% [markdown]
# **Reading Fig 16.** Once n₀ ≥ 50 the left floor ends at an almost constant
# number of order statistics — k ≈ 7–9 at 95%, 5–7 at 80%, 4–5 at 50% — so
# in FPR units it shrinks like 1/n₀. The right floor's length is set by the
# plateau, so it grows with AUC and shrinks with n. Together the floor owns
# the whole curve at n = 10, 59% of it at n = 100, 20% at n = 1,000, and 8%
# at n = 10,000: the hybrid hands the curve from M3 to the fiducial band as
# data accumulate.

# %% [markdown]
# ### Fig 17 — Misses inside vs. outside the floor region

# %%
pf.fig17_misses_by_floor_region(df)

# %% [markdown]
# ### Fig 18 — The trade, by FPR region

# %%
pf.fig18_floor_tradeoff_by_region(df)

# %% [markdown]
# **Reading Figs 17–18.** Inside its region the floor nearly eliminates
# misses (at 50% and n = 100, from ≈ 0.15 to ≈ 0.04). Outside it, misses
# rise, mostly with the truth above the band in FPR 10–50%: the
# recalibrated trim spends there the level the floor freed. In width the
# floor shows up as extra width at the two ends; outside the region the
# hybrid is at most slightly narrower than the raw band (≈ 0.01–0.02 TPR
# at 50% for n = 100–300, ≈ 0.01 at 95% for n = 300). At n ≤ 100 the region
# covers most of the curve, so the hybrid is wider almost everywhere. The
# net effect at 50% is coverage *closer* to nominal at a higher mean width,
# and all of it fades by n ≈ 1,000 as the floor region shrinks.

# %% [markdown]
# ---
# # 6. What the band looks like

# %% [markdown]
# ### Fig 19 — Example 95% bands, with the floor region shaded
#
# One seeded n = 300 dataset per DGP × target AUC. The shaded region is
# where the hybrid took the M3 hull; outside it the hybrid is the
# (recalibrated) fiducial band. The raw fiducial and M3 bands are overlaid,
# so the reader can see the hybrid follow M3 in the shade and track the raw
# band elsewhere. Fresh draws, not simulation evidence.

# %%
for family in pf.DGP_FAMILIES:
    pf.fig19_example_bands(family=family)

# %% [markdown]
# ### Fig 19d — Where the floor operates, across every replicate (new)
#
# Instead of single draws, this uses the stored floor region of every
# replicate: the color at (FPR, n) is the share of replicates whose floor
# region contains that FPR. The left region's shrinkage with n and the
# right region's growth with AUC are both visible directly.

# %%
pf.fig19d_floor_occupancy(df)

# %% [markdown]
# ---
# # 7. Violation profile

# %% [markdown]
# ### Fig 20 — Direction and location of misses

# %%
pf.fig20_violation_direction_location(df)

# %% [markdown]
# ### Fig 21 — Magnitude of misses

# %%
pf.fig21_violation_magnitude(df)

# %%
print("Rate of >5pp violations, true AUC > 0.9, 95% bands:")
print(pf.high_auc_audit(df).round(4).to_string())

# %% [markdown]
# **Reading Figs 20–21.** The fiducial arms lean toward misses above the
# curve; the envelope's misses move from above to below as n grows. Misses
# are shallow for every rank-based method: the 99th percentile of the
# worst violation is at most 0.4pp at every n for the hybrid, raw fiducial,
# and envelope, and exactly 0 for M3 and KS, against 0.75–0.83 for WH at
# n ≥ 300. In the high-AUC stratum the hybrid's rate of >5pp misses is at
# most 0.03%.

# %% [markdown]
# ---
# # 8. Prevalence, cost, and guarantees

# %% [markdown]
# ### Fig 22 — Class imbalance

# %%
pf.fig22_prevalence(df)

# %% [markdown]
# ### Fig 23 — Computational cost (new)
#
# Median wall time to produce all three levels for one replicate. The two
# fiducial arms come from one call and are counted once; the envelope
# includes its bootstrap generation and runs on the GPU for n₀ ≥ 500.
# Measured under the run's concurrent load.

# %%
pf.fig23_runtime(df)

# %% [markdown]
# **Reading Fig 23.** The fiducial suite is the expensive method: its
# 20,000-draw cloud costs ≈ 0.4 s per replicate at n = 300 and ≈ 10 s at
# n = 10,000. M3, KS, and WH cost milliseconds. The envelope on the GPU
# costs ≈ 0.05–0.13 s for n ≥ 1,000. For one dataset in practice all of
# these are fast; the cost matters for simulation and for repeated use.

# %% [markdown]
# ### Fig 24 — Calibration in every cell (new)
#
# Each point is one DGP × configuration cell (56 per method and level).
# Filled points are cells whose 95% Wilson interval lies entirely below
# nominal — significant undercoverage beyond Monte Carlo error.

# %%
pf.fig24_calibration_distribution(cells)

# %%
below = cells.assign(below=cells["cov_hi"] < 1 - cells["alpha"])
print("Cells significantly below nominal (of 56):")
print(below.groupby(["method", "alpha"], observed=True)["below"].sum().unstack())
print(
    below.loc[
        below["below"] & (below["method"] != "working_hotelling"),
        ["method", "alpha", "dgp_type", "n_total", "prevalence", "coverage"],
    ].to_string(index=False)
)

# %% [markdown]
# **Reading Fig 24.** M3 has no cell below nominal, as its theorem requires
# — a direct check of the implementation. The hybrid has none either, at
# any level. The raw fiducial band has one (heteroscedastic Gaussian,
# n = 1,000, 10% prevalence, 95%) and the envelope one (heteroscedastic
# Gaussian, n = 10,000, 95%: 0.944). Working–Hotelling has 34–36 of 56.

# %% [markdown]
# ### Table 1 — Method summary

# %%
summary = pf.method_summary_table(df=df, cells=cells)
summary.round(3)

# %% [markdown]
# ### Headline numbers

# %%
c95 = cells[(cells["alpha"] == 0.05) & (cells["prevalence"] == 0.5)]
hero95 = c95[c95["method"] == pf.HERO]
print("Hybrid coverage at 95%, range over DGPs, by n:")
print(hero95.groupby("n_total")["coverage"].agg(["min", "max"]).round(3).T.to_string())
area = c95.pivot_table(
    index=["dgp_type", "n_total"], columns="method", values="area", observed=True
)
print("\nBand area ÷ KS (mean over DGPs) at 95%:")
print(area.div(area["ks"], axis=0).groupby("n_total").mean().round(3).to_string())
print("\nBand area ÷ M3 (mean over DGPs) at 95%:")
print(area.div(area["m3"], axis=0).groupby("n_total").mean().round(3).to_string())
wh = c95[c95["method"] == "working_hotelling"].pivot_table(
    index="dgp_type", columns="n_total", values="coverage", observed=True
)
print("\nWorking–Hotelling coverage at 95%:")
print(wh.round(3).to_string())

# %% [markdown]
# ---
# # Conclusions
#
# 1. **Calibration (Figs 1–4, 24).** The hybrid never covers significantly
#    below nominal in any of 168 DGP × configuration × level cells. At 95%
#    it runs 0.97–1.00, conservative at small n where it is M3 and
#    approaching nominal with n. Like every sup-norm band it over-covers at
#    50%, by less as n grows.
# 2. **Robustness (Figs 5–7).** Working–Hotelling is calibrated only where
#    the binormal ROC model holds and collapses with n elsewhere (to 0.01 on
#    Weibull at n = 10,000). Every rank-based method is indistinguishable
#    across families.
# 3. **Tightness (Figs 8–10).** At n ≥ 1,000 the hybrid is about half the
#    KS area and two thirds of the M3 area, matching the raw fiducial band
#    and the envelope.
# 4. **Ablations (Figs 11–14).** Neither component is the method on its
#    own. Without the floor, the fiducial band is calibrated on average but
#    leaks at the plateau at small n and in its worst cell (0.900); the floor
#    alone (M3) never leaks but is 1.3–1.5× wider and nearly ignores the
#    requested level. The hybrid keeps M3's worst-case protection at 95% and
#    the raw band's width and calibration for n ≳ 1,000, paying M3's width
#    at low FPR for n ≤ 100.
# 5. **What the floor buys (Figs 15–18).** At 95% the floor changes coverage
#    by less than a third of a point for balanced n ≥ 100, while repairing
#    the one case where the raw fiducial band fails (low prevalence,
#    heteroscedastic Gaussian). Inside its region it nearly eliminates
#    misses; the recalibrated trim spends the freed level in the middle of
#    the curve, moving 50% coverage toward nominal. The region shrinks from
#    the whole curve at n = 10 to 8% of it at n = 10,000.
# 6. **Misses (Figs 20–21).** Rare, shallow, and mildly above-dominant; the
#    99th-percentile miss depth is at most 0.4pp at every n.
# 7. **Cost (Fig 23).** The fiducial suite is the slowest method (≈ 10 s per
#    replicate at n = 10,000), though fast enough for any single analysis.
