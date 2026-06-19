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
# # Simultaneous Confidence Bands for ROC Curves — Paper Analysis
#
# This notebook builds every figure and table for the paper from the
# `data/results/final_20260611` simulation run (4,000 LHS parameter draws per
# DGP, sample sizes 10–10,000, confidence levels 50/80/95%, 13 methods,
# B = 4,000 bootstrap replicates; see `run_metadata_20260611_203233.json`).
#
# **The method under study** ("envelope") is the studentized bootstrap envelope
# with three tail repairs: a Wilson variance floor during studentization, a
# variance-ratio-gated Wilson rectangle floor at the TPR plateau, and an exact
# Beta order-statistic floor on the lower band at extreme low FPR. The run
# includes every ablation needed to attribute coverage to each component.
#
# **Run status:** some DGPs may still be mid-run (gamma and weibull were
# pending when this notebook was first written). All analysis cells are
# data-driven — they pick up whatever DGP × n cells exist on disk — so the
# notebook can simply be re-executed when the remaining results land.
#
# **Outputs:** every figure is saved to `figures/paper/` as high-resolution
# PNG, PDF, and SVG. Summary tables are saved as CSV and Markdown.
#
# **Regenerating:**
# ```
# uv run jupytext --to ipynb notebook_paper.py        # convert to .ipynb
# uv run jupyter nbconvert --to html --execute notebook_paper.ipynb
# ```

# %%
import sys
import warnings
from pathlib import Path

if str(Path("src").resolve()) not in sys.path:
    sys.path.append(str(Path("src").resolve()))

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import t as t_dist

from studroc_paper.eval.build_data_from_jsons import load_individual_results

warnings.filterwarnings("ignore", category=FutureWarning)

RESULTS_DIR = Path("data/results/final_20260611")
FIG_DIR = Path("figures/paper")
FIG_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [0.05, 0.2, 0.5]  # nominal alpha levels in the run (95/80/50% CIs)
N_ORDER = [10, 30, 100, 300, 1000, 3000, 10000]


def set_style() -> None:
    """Set the global matplotlib style for publication figures."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "legend.title_fontsize": 9,
            "figure.titlesize": 13,
            "figure.titleweight": "bold",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "figure.dpi": 110,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "legend.frameon": False,
        }
    )


set_style()


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save a figure as high-res PNG, PDF, and SVG, then display it.

    Args:
        fig: The matplotlib figure to save.
        name: Stem of the output file (no extension).
    """
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    print(f"Saved {name}.{{png,pdf,svg}} to {FIG_DIR}/")
    plt.show()


# %% [markdown]
# ## Configuration: method roster, DGP roster, and visual identity
#
# Visual conventions used consistently throughout:
#
# - **Methods get fixed colors** from the Okabe–Ito colorblind-safe palette.
#   The envelope method is always vermillion; Working–Hotelling is always
#   blue; KS is always black.
# - **Envelope-family ablations share the envelope's color** and are
#   distinguished by *line style and marker shape*, since they are variants
#   of one method rather than independent competitors.
# - **DGPs get fixed colors grouped by family hue**: blues for Gaussian-like,
#   warm colors for heavy-tailed/skewed, green/purple for non-standard shapes.
# - Nominal coverage is always a dashed gray reference line; coverage axes
#   are interpreted as *calibration* (closer to nominal is better — above
#   nominal is conservative, not "good").

# %%
# --- Methods -----------------------------------------------------------------
# label, color, linestyle, marker
METHOD_META: dict[str, dict] = {
    "envelope": dict(
        label="Studentized envelope (full)", color="#D55E00", ls="-", marker="o"
    ),
    "working_hotelling": dict(
        label="Working–Hotelling (binormal)", color="#0072B2", ls="-", marker="s"
    ),
    "ks": dict(label="KS fixed-width (DKW)", color="#000000", ls="-", marker="D"),
    "wilson_rectangle_sidak": dict(
        label="Wilson rectangles (Šidák)", color="#009E73", ls="-", marker="^"
    ),
    "wilson_rectangle_bonferroni": dict(
        label="Wilson rectangles (Bonferroni)", color="#009E73", ls="--", marker="v"
    ),
    "pointwise": dict(
        label="Pointwise bootstrap", color="#999999", ls="-", marker="x"
    ),
    "pointwise_sidak": dict(
        label="Pointwise bootstrap (Šidák)", color="#56B4E9", ls="-", marker="P"
    ),
    # Envelope-family ablations: same hue, distinguished by style.
    "envelope_no_beta_floor": dict(
        label="Envelope without Beta floor", color="#D55E00", ls="--", marker="s"
    ),
    "envelope_no_wilson_floor": dict(
        label="Envelope without Wilson floor", color="#D55E00", ls="-.", marker="^"
    ),
    "envelope_no_floors": dict(
        label="Bare bootstrap envelope (no floors)",
        color="#D55E00",
        ls=":",
        marker="x",
    ),
    "envelope_beta_both_tails": dict(
        label="Beta floors on both tails (no Wilson)",
        color="#E69F00",
        ls="--",
        marker="P",
    ),
    "envelope_wilson_both_tails": dict(
        label="Wilson floors on both tails (no Beta)",
        color="#E69F00",
        ls="-.",
        marker="X",
    ),
    "envelope_no_bootstrap": dict(
        label="Wilson + Beta floors only (no bootstrap)",
        color="#CC79A7",
        ls=":",
        marker="h",
    ),
}


def mlabel(m: str) -> str:
    return METHOD_META.get(m, {}).get("label", m)


def mcolor(m: str) -> str:
    return METHOD_META.get(m, {}).get("color", "#777777")


def mstyle(m: str) -> dict:
    meta = METHOD_META.get(m, {})
    return dict(
        color=meta.get("color", "#777777"),
        linestyle=meta.get("ls", "-"),
        marker=meta.get("marker", "o"),
    )


# --- DGPs --------------------------------------------------------------------
DGP_META: dict[str, dict] = {
    "binormal": dict(label="Binormal", color="#0072B2", family="Gaussian-like"),
    "hetero_gaussian": dict(
        label="Heterosc. Gaussian", color="#56B4E9", family="Gaussian-like"
    ),
    "logitnormal": dict(label="Logit-normal", color="#7BB6DD", family="Gaussian-like"),
    "student_t": dict(
        label="Student-t", color="#D55E00", family="Heavy-tailed / skewed"
    ),
    "gamma": dict(label="Gamma", color="#E69F00", family="Heavy-tailed / skewed"),
    "weibull": dict(label="Weibull", color="#CC6677", family="Heavy-tailed / skewed"),
    "beta_opposing": dict(
        label="Beta (opposing skew)", color="#009E73", family="Non-standard shape"
    ),
    "bimodal_negative": dict(
        label="Bimodal negatives", color="#AA4499", family="Non-standard shape"
    ),
}
DGP_ORDER = list(DGP_META)
DGP_FAMILIES: dict[str, list[str]] = {}
for d, meta in DGP_META.items():
    DGP_FAMILIES.setdefault(meta["family"], []).append(d)


def dlabel(d: str) -> str:
    return DGP_META.get(d, {}).get("label", d)


def dcolor(d: str) -> str:
    return DGP_META.get(d, {}).get("color", "#777777")


# --- Regions and width landmarks ----------------------------------------------
REGIONS = ["0-10", "10-30", "30-50", "50-70", "70-90", "90-100"]
REGION_COLS = [f"violation_{r}" for r in REGIONS]
WIDTH_LANDMARKS = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
WIDTH_COLS = {x: f"width_at_fpr_{x}" for x in WIDTH_LANDMARKS}

# Sequential color scale for sample size (log-spaced viridis)
_N_NORM = mpl.colors.LogNorm(vmin=10, vmax=10000)


def n_color(n: int) -> tuple:
    return mpl.colormaps["viridis"](_N_NORM(n) * 0.92)


# --- Small statistical helpers -------------------------------------------------
def wilson_ci(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple:
    """Wilson score interval for binomial proportions.

    Args:
        k: Number of successes.
        n: Number of trials.
        z: Normal quantile for the confidence level.

    Returns:
        Tuple (lower, upper) arrays.
    """
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def mean_ci_across(values: np.ndarray) -> tuple[float, float, float]:
    """Mean and 95% t-CI of a small sample of per-DGP summary statistics.

    Used for aggregate line plots whose points average a statistic across
    DGPs: the interval is the standard error of that across-DGP mean (Student
    t, D-1 df), so it reflects both Monte Carlo error within cells and
    heterogeneity between DGPs — not the raw min–max spread.

    Args:
        values: Per-DGP statistic values (NaNs ignored).

    Returns:
        Tuple (mean, ci_lower, ci_upper); degenerate to the point itself for
        a single value and (nan, nan, nan) when empty.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if v.size == 0:
        return np.nan, np.nan, np.nan
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0])
    m = float(v.mean())
    half = float(t_dist.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size))
    return m, m - half, m + half


def mstyle_noshape(method: str, marker: str = "o") -> dict:
    """Line style for a method using color (and linestyle) but a fixed marker.

    Color carries method identity; marker shape is uniform to reduce clutter
    on plots where every method already has a distinct color.
    """
    meta = METHOD_META.get(method, {})
    return dict(
        color=meta.get("color", "#777777"),
        linestyle=meta.get("ls", "-"),
        marker=marker,
    )


def pareto_marker(method: str) -> str:
    """Marker for Pareto scatter points: uniform circle except the Wilson
    rectangles, which share a color and need shapes to tell Šidák from
    Bonferroni apart."""
    return {
        "wilson_rectangle_sidak": "^",
        "wilson_rectangle_bonferroni": "v",
    }.get(method, "o")


def coverage_by(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Aggregate coverage with Wilson CIs over grouping columns."""
    g = (
        df.groupby(by, observed=True)["covers_entirely"]
        .agg(coverage="mean", n_trials="size")
        .reset_index()
    )
    lo, hi = wilson_ci(g["coverage"] * g["n_trials"], g["n_trials"])
    g["ci_lo"], g["ci_hi"] = lo, hi
    return g


def macro_coverage(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Coverage macro-averaged over (dgp_type, n_total) cells.

    Equal weight per DGP × n cell, so partially complete DGPs (e.g. a
    mid-run weibull) do not skew pooled numbers.
    """
    cells = (
        df.groupby([*by, "dgp_type", "n_total"], observed=True)["covers_entirely"]
        .mean()
        .reset_index()
    )
    return (
        cells.groupby(by, observed=True)["covers_entirely"]
        .mean()
        .rename("coverage")
        .reset_index()
    )


def fit_rate_curve(
    x: np.ndarray, y: np.ndarray, grid: np.ndarray, n_splines: int = 8
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Smooth P(y=1 | x) with a logistic GAM, with a pointwise 95% band.

    Falls back to binned rates with Wilson intervals if the GAM cannot be
    fit (e.g., perfect separation when every outcome is 1).

    Args:
        x: Predictor values.
        y: Binary outcomes.
        grid: Points at which to evaluate the smooth.
        n_splines: Spline basis size for the GAM.

    Returns:
        Tuple (rate, ci_lower, ci_upper) on the grid, or None if there is
        too little data.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 100:
        return None
    try:
        from pygam import LogisticGAM, s

        gam = LogisticGAM(s(0, n_splines=n_splines)).fit(x.reshape(-1, 1), y)
        pred = gam.predict_proba(grid.reshape(-1, 1))
        lo, hi = gam.confidence_intervals(grid.reshape(-1, 1), width=0.95).T
        return pred, np.clip(lo, 0, 1), np.clip(hi, 0, 1)
    except Exception:
        bins = np.quantile(x, np.linspace(0, 1, 13))
        idx = np.clip(np.searchsorted(bins, x) - 1, 0, 11)
        centers, rates, ks, ns = [], [], [], []
        for b in range(12):
            sel = idx == b
            if sel.sum() >= 20:
                centers.append(x[sel].mean())
                rates.append(y[sel].mean())
                ks.append(y[sel].sum())
                ns.append(sel.sum())
        if len(centers) < 3:
            return None
        centers, rates = np.array(centers), np.array(rates)
        lo_b, hi_b = wilson_ci(np.array(ks), np.array(ns))
        return (
            np.interp(grid, centers, rates),
            np.interp(grid, centers, lo_b),
            np.interp(grid, centers, hi_b),
        )


def add_nominal_line(ax: plt.Axes, alpha: float) -> None:
    ax.axhline(1 - alpha, color="0.35", ls="--", lw=1.0, zorder=1)


def contiguous_spans(
    mask: np.ndarray, x: np.ndarray, min_len: int = 1
) -> list[tuple[float, float]]:
    """Return (x_start, x_end) intervals for runs of True in a boolean mask.

    Args:
        mask: Boolean array over grid points.
        x: Grid coordinates aligned with the mask.
        min_len: Minimum run length (in grid points) to report; shorter runs
            are dropped to suppress single-point speckle in shaded regions.
    """
    spans = []
    in_run = False
    start_idx = 0
    for i, m in enumerate(mask):
        if m and not in_run:
            in_run, start_idx = True, i
        elif not m and in_run:
            in_run = False
            if i - start_idx >= min_len:
                spans.append((x[start_idx], x[i]))
    if in_run and len(mask) - start_idx >= min_len:
        spans.append((x[start_idx], x[-1]))
    return spans


# %% [markdown]
# ## Data loading and inventory
#
# Trial-level results (one row per method × alpha × LHS draw × repeat) are the
# primary substrate — every aggregate in this notebook is recomputed from
# trials so that filters (AUC strata, prevalence, regions) compose freely.
# Strings load as categoricals and floats as float32 to keep the ~8M-row
# frame comfortably in memory.
#
# **Hairline-violation correction.** The simulation pipeline runs in float32,
# and the run's evaluation used a 1e-10 coverage tolerance — far below the
# float32 spacing near TPR = 1 (~6e-8). Methods whose bands on the TPR
# plateau are tighter than one float32 ulp (Working–Hotelling above all, and
# the Wilson rectangles) accumulated phantom "violations" of magnitude
# ~1e-8, which masqueraded as a coverage collapse that *grew with n* (more
# plateau grid points with sub-ulp band width). The coverage flags are
# therefore recomputed here from the stored violation magnitudes with a
# 1e-6 TPR tolerance — far below any statistically meaningful width, far
# above representation noise. `evaluate_single_band` now applies the same
# tolerance at source, so future runs make this recomputation a no-op.
# Envelope results are identical to four decimals either way. The per-region
# violation flags cannot be magnitude-filtered retroactively; they are used
# only for the envelope-family anatomy figures, where the artifact is
# negligible.

# %%
run_meta_files = sorted(RESULTS_DIR.glob("run_metadata_*.json"))
if run_meta_files:
    run_meta = json.loads(run_meta_files[-1].read_text())
    print("Run:", run_meta["timestamp"], "| git", run_meta["git_hash"][:10])
    print("Methods:", ", ".join(run_meta["methods"]))

df = load_individual_results(RESULTS_DIR)

# float32 stores 0.05 inexactly; restore exact float64 keys for filtering
for c in ["alpha", "confidence_level", "prevalence"]:
    df[c] = df[c].astype(float).round(4)
df["n_total"] = df["n_total"].astype(int)

# Recompute coverage flags with the 1e-6 TPR tolerance (see markdown above);
# crossings at float32 representation scale do not count as violations
VIOLATION_EPS = 1e-6
df["violation_above"] = df["max_violation_above"] > VIOLATION_EPS
df["violation_below"] = df["max_violation_below"] > VIOLATION_EPS
df["covers_entirely"] = ~(df["violation_above"] | df["violation_below"])
df["dgp_type"] = pd.Categorical(
    df["dgp_type"].astype(str),
    categories=[d for d in DGP_ORDER if d in set(df["dgp_type"].astype(str))],
    ordered=True,
)
df["method"] = df["method"].astype(str)

print(f"\n{len(df):,} trial rows, {df['method'].nunique()} methods, "
      f"alphas={sorted(df['alpha'].unique())}")

# Inventory: simulations per DGP × n (envelope rows at alpha=0.05, prev=0.5)
inv = (
    df[(df.method == "envelope") & (df.alpha == 0.05) & (df.prevalence == 0.5)]
    .groupby(["dgp_type", "n_total"], observed=True)
    .size()
    .unstack(fill_value=0)
)
print("\nSimulations per DGP × n (prev=0.5):")
print(inv.to_string())
missing = [d for d in DGP_ORDER if d not in inv.index or (inv.loc[d] == 0).any()]
if missing:
    print(f"\nNOTE: incomplete or missing DGPs at this run stage: {missing}."
          " Re-run this notebook when the simulation finishes.")

# %%
# Working subsets used throughout: balanced prevalence; high-AUC stratum.
dfb = df[df.prevalence == 0.5]
PRESENT_DGPS = [d for d in DGP_ORDER if d in set(df["dgp_type"].astype(str))]
PRESENT_NS = sorted(dfb["n_total"].unique())
HIGH_AUC = 0.9  # threshold for the "steep early ROC" stratum


# %% [markdown]
# ---
# # 1. Coverage of the envelope method
#
# **Motivation.** The headline claim is that the studentized bootstrap
# envelope with its two tail floors is a *generically calibrated*
# simultaneous band: close to nominal coverage across distribution families,
# sample sizes, AUC levels, and DGP shape parameters, with no distributional
# assumptions.
#
# **Expectations** (from the project evaluation report and the integrated
# Beta-floor validation): at 95%, coverage should sit in roughly 0.95–0.99
# everywhere — slightly conservative at small n (the Wilson floor dominates
# when the grid is short) and slightly conservative at large n (the strict
# Bonferroni budget of the Beta floor overshoots). The known residual
# weakness is mild over-coverage at the 50% level (a property of sup-norm
# calibrated bands, diminishing with n), which we display rather than hide:
# *calibration, not maximal coverage, is the success criterion*.

# %% [markdown]
# ### Fig 1 — Coverage by DGP × n at each confidence level
#
# Annotated heatmaps; color encodes *deviation from nominal* (red = below
# nominal / anticonservative, blue = above nominal / conservative). The 95%
# band (Fig 1a) is the headline result and gets its own figure; the 50% and
# 80% bands (Fig 1b) are split off with their own, much larger, deviation
# scales (±5pp is enormous at 95% but trivial at 50%).

# %%
DEV_LIMS = {0.05: 0.05, 0.2: 0.12, 0.5: 0.35}
env = dfb[dfb.method == "envelope"]


def plot_coverage_heatmaps(alphas: list[float], fig_name: str, title: str) -> None:
    """Annotated coverage heatmaps (DGP × n) for one or more alpha levels."""
    width = 5.6 if len(alphas) == 1 else 4.6 * len(alphas)
    fig, axes = plt.subplots(
        1, len(alphas), figsize=(width, 0.62 * len(PRESENT_DGPS) + 1.6),
        constrained_layout=True, squeeze=False,
    )
    axes = axes[0]
    se_notes = []
    for ax, alpha in zip(axes, alphas, strict=False):
        sub = env[env.alpha == alpha]
        cov = (
            sub.groupby(["dgp_type", "n_total"], observed=True)["covers_entirely"]
            .mean()
            .unstack()
            .reindex(index=PRESENT_DGPS, columns=PRESENT_NS)
        )
        n_cells = (
            sub.groupby(["dgp_type", "n_total"], observed=True)
            .size()
            .unstack()
            .reindex(index=PRESENT_DGPS, columns=PRESENT_NS)
        )
        max_se = np.nanmax(np.sqrt(cov * (1 - cov) / n_cells).values)
        se_notes.append(f"{(1 - alpha) * 100:.0f}% level ≤ {max_se:.3f}")
        sns.heatmap(
            cov - (1 - alpha),
            annot=cov,
            fmt=".3f",
            cmap="RdBu",
            center=0,
            vmin=-DEV_LIMS[alpha],
            vmax=DEV_LIMS[alpha],
            linewidths=0.8,
            linecolor="white",
            cbar_kws={"label": "Coverage − nominal", "shrink": 0.85},
            annot_kws={"fontsize": 7.5},
            ax=ax,
        )
        ax.set_title(f"{(1 - alpha) * 100:.0f}% band (nominal {1 - alpha:.2f})")
        ax.set_xlabel("Total sample size n")
        ax.set_ylabel("")
        ax.set_yticklabels(
            [dlabel(d) for d in PRESENT_DGPS] if ax is axes[0] else [],
            rotation=0,
        )
        ax.grid(False)
    fig.suptitle(title)
    fig.text(
        0.01, -0.025,
        "Each cell pools ~4,000 simulations; Monte Carlo SE per cell: "
        + ", ".join(se_notes) + ".",
        fontsize=8, color="0.4",
    )
    save_figure(fig, fig_name)


# %%
plot_coverage_heatmaps(
    [0.05],
    "fig01a_envelope_coverage_heatmap_95",
    "Envelope coverage at the 95% band, across DGPs and sample sizes",
)

# %%
plot_coverage_heatmaps(
    [0.2, 0.5],
    "fig01b_envelope_coverage_heatmap_50_80",
    "Envelope coverage at the 80% and 50% bands",
)

# %% [markdown]
# **Reading Fig 1.** Fig 1a (95%) is the headline: cells should be pale
# (within ~2pp of nominal) or lightly blue (a few pp conservative); any red
# cell deserves a targeted look in Section 6 (violation profile). Fig 1b
# documents the known over-coverage of sup-norm bands at lower confidence
# levels — the method is honest about being most useful for high-confidence
# bands. Note the deviation scales differ between the two figures.

# %% [markdown]
# ### Fig 2 — Coverage vs. true AUC within each DGP
#
# **Motivation.** The evaluation report established that ROC *geometry* —
# not distribution family — is the first-order risk factor: high AUC creates
# a steep low-FPR segment where the bootstrap support collapses. The Beta
# floor was built precisely for that regime, so the money plot is coverage
# as a smooth function of true AUC, per DGP, per n.
#
# **Expectation.** Flat profiles near 0.95. The pre-Beta-floor method showed
# a monotone slide toward ~0.74 at AUC > 0.95 for large n; that slide should
# now be gone. Mild conservatism at the highest AUCs is acceptable.
#
# Shaded ribbons are pointwise 95% confidence bands from the logistic-GAM
# fit: where a ribbon sits clear of the nominal line, the deviation is
# resolved by the simulation budget rather than smoothing noise.

# %%
AUC_NS = [100, 1000, 10000]
auc_grid = np.linspace(0.56, 0.985, 160)
# Heteroscedastic Gaussian is dropped here only, so the grid tiles without a
# blank panel (its AUC behaviour is identical to binormal anyway).
FIG2_DGPS = [d for d in PRESENT_DGPS if d != "hetero_gaussian"]
n_dgps = len(FIG2_DGPS)
# Choose ncols (4 or 3) that leaves the fewest blank panels.
ncols = min((4, 3), key=lambda c: (int(np.ceil(n_dgps / c)) * c - n_dgps, -c))
nrows = int(np.ceil(n_dgps / ncols))

fig, axes = plt.subplots(
    nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows), sharex=True, sharey=True,
    squeeze=False,
)
sub95 = env[env.alpha == 0.05]
for i, dgp in enumerate(FIG2_DGPS):
    ax = axes.flat[i]
    for n in AUC_NS:
        ms = sub95[(sub95.dgp_type == dgp) & (sub95.n_total == n)]
        if len(ms) < 200:
            continue
        fit = fit_rate_curve(
            ms["true_auc"].values, ms["covers_entirely"].values, auc_grid
        )
        if fit is not None:
            pred, lo, hi = fit
            ax.plot(auc_grid, pred, color=n_color(n), lw=1.8, label=f"n = {n:,}")
            ax.fill_between(auc_grid, lo, hi, color=n_color(n), alpha=0.14, lw=0)
    add_nominal_line(ax, 0.05)
    ax.set_title(dlabel(dgp), fontsize=10)
    ax.set_ylim(0.88, 1.005)
    ax.set_xlim(0.55, 1.0)
for j in range(n_dgps, nrows * ncols):
    axes.flat[j].set_visible(False)
for ax in axes[-1, :]:
    ax.set_xlabel("True AUC")
for ax in axes[:, 0]:
    ax.set_ylabel("Coverage (95% band)")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(AUC_NS),
           bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Coverage is stable over AUC")
fig.tight_layout()
save_figure(fig, "fig02_envelope_coverage_vs_auc")

# %% [markdown]
# ### Fig 3 — Coverage vs. DGP shape parameters
#
# **Motivation.** Within each family, the LHS design sweeps a shape
# parameter that controls *how hard* the problem is (tail weight, skew,
# heteroscedasticity, multimodality). A calibrated method should be flat in
# these nuisance directions too — this is the within-family robustness claim.
# Ribbons: pointwise 95% GAM confidence bands.

# %%
SHAPE_SPECS = [
    # (dgp, column or derived, x-label, log-x)
    ("student_t", "dgp_df", "Degrees of freedom (lower = heavier tails)", True),
    ("logitnormal", "dgp_sigma", "Latent σ", False),
    ("hetero_gaussian", "sigma_ratio", "σ ratio (positive / negative class)", True),
    ("beta_opposing", "dgp_alpha", "Beta α (skew strength)", False),
    ("bimodal_negative", "mode_separation", "Negative-class mode separation Δ", False),
    ("bimodal_negative", "mixture_weight", "Negative-class mixture weight", False),
    ("weibull", "dgp_neg_shape", "Weibull shape (lower = heavier tail)", False),
    ("gamma", "dgp_neg_shape", "Gamma shape", False),
]


def derive_shape_columns(sub: pd.DataFrame) -> pd.DataFrame:
    """Add derived shape-parameter columns to a filtered trial frame."""
    sub = sub.copy()
    if {"dgp_sigma_pos", "dgp_sigma_neg"}.issubset(sub.columns):
        sub["sigma_ratio"] = sub["dgp_sigma_pos"] / sub["dgp_sigma_neg"]
    if "dgp_neg_means" in sub.columns:
        means = sub["dgp_neg_means"].dropna()
        if len(means):
            sep = means.map(lambda a: float(np.ptp(np.asarray(a))))
            sub.loc[sep.index, "mode_separation"] = sep
    if "dgp_neg_weights" in sub.columns:
        w = sub["dgp_neg_weights"].dropna()
        if len(w):
            sub.loc[w.index, "mixture_weight"] = w.map(
                lambda a: float(np.asarray(a)[0])
            )
    return sub


specs = [
    (d, c, x, lg)
    for (d, c, x, lg) in SHAPE_SPECS
    if d in PRESENT_DGPS
]
ncols = 4
nrows = int(np.ceil(len(specs) / ncols))
fig, axes = plt.subplots(
    nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows), sharey=True, squeeze=False
)
SHAPE_NS = [100, 1000, 10000]
for i, (dgp, col, xlabel, logx) in enumerate(specs):
    ax = axes.flat[i]
    base = derive_shape_columns(
        sub95[(sub95.dgp_type == dgp) & (sub95.n_total.isin(SHAPE_NS))]
    )
    if col not in base.columns:
        ax.set_visible(False)
        continue
    drew_any = False
    for n in SHAPE_NS:
        ms = base[base.n_total == n].dropna(subset=[col])
        if len(ms) < 200:
            continue
        x = ms[col].values.astype(float)
        gx = np.log10(x) if logx else x
        grid = np.linspace(gx.min(), gx.max(), 120)
        fit = fit_rate_curve(gx, ms["covers_entirely"].values, grid)
        if fit is not None:
            pred, lo, hi = fit
            xs = 10**grid if logx else grid
            ax.plot(xs, pred, color=n_color(n), lw=1.8, label=f"n = {n:,}")
            ax.fill_between(xs, lo, hi, color=n_color(n), alpha=0.14, lw=0)
            drew_any = True
    if not drew_any:
        ax.set_visible(False)
        continue
    if logx:
        ax.set_xscale("log")
    add_nominal_line(ax, 0.05)
    ax.set_title(dlabel(dgp), fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylim(0.80, 1.02)
for j in range(len(specs), nrows * ncols):
    axes.flat[j].set_visible(False)
for ax in axes[:, 0]:
    ax.set_ylabel("Coverage (95% band)")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
fig.suptitle("Coverage is flat across DGP shape parameters")
fig.tight_layout()
save_figure(fig, "fig03_envelope_coverage_vs_shape")

# %% [markdown]
# ### Fig 4 — Calibration across confidence levels, by sample size
#
# **Motivation.** A band method should track its nominal level, not merely
# clear it. Each panel plots empirical vs. nominal coverage (macro-averaged
# over DGPs) for all methods at one sample size — the identity line is
# perfect calibration. Showing n = 10 → 10,000 makes the honest case for
# claim 3: KS "wins" every coverage comparison only by being pinned at ~1.0
# regardless of the requested level, while the envelope tracks the identity.
#
# **Expectation.** The envelope hugs the identity at 95%, drifting above at
# 80% and 50% (the sup-norm alpha-insensitivity wall) — a gap that shrinks
# as n grows. KS sits at ~1.0 everywhere; Wilson rectangles fall below at
# stricter levels; Working–Hotelling sits low and worsens with n.

# %%
CAL_METHODS = ["envelope", "ks", "wilson_rectangle_sidak", "working_hotelling",
               "pointwise_sidak"]
CAL_NS = [10, 100, 1000, 10000]
fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.0), sharex=True, sharey=True)
for ax, n in zip(axes.flat, CAL_NS, strict=False):
    for m in CAL_METHODS:
        sub = dfb[(dfb.method == m) & (dfb.n_total == n)]
        rows = []
        for alpha in ALPHAS:
            mc = macro_coverage(sub[sub.alpha == alpha], by=["method"])
            if len(mc):
                rows.append((1 - alpha, mc["coverage"].iloc[0]))
        if rows:
            nom, cov = zip(*sorted(rows), strict=False)
            ax.plot(nom, cov, lw=1.8, ms=5.5, label=mlabel(m),
                    **mstyle_noshape(m))
    ax.plot([0.4, 1.0], [0.4, 1.0], color="0.35", ls="--", lw=1.0, zorder=1)
    ax.set_title(f"n = {n:,}")
    ax.set_xticks([0.5, 0.8, 0.95])
    ax.set_ylim(0.0, 1.05)
for ax in axes[-1, :]:
    ax.set_xlabel("Nominal coverage")
for ax in axes[:, 0]:
    ax.set_ylabel("Empirical coverage")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(CAL_METHODS),
           bbox_to_anchor=(0.5, -0.03))
fig.suptitle("Calibration across confidence levels (identity = perfect)")
fig.tight_layout()
save_figure(fig, "fig04_alpha_calibration")

# %% [markdown]
# ---
# # 2. Robustness: the envelope vs. Working–Hotelling
#
# **Motivation.** Working–Hotelling is the classical tight band — *if* the
# binormal assumption holds. The practical pitch for the envelope is: you
# pay a small width premium over WH on binormal data, and in exchange you
# keep your nominal coverage on every other distribution, where WH can be
# catastrophically wrong (the report measured < 20% coverage on Student-t).
#
# **Expectations.** WH ≈ nominal wherever the binormal model actually holds
# — the binormal DGP *and* the heteroscedastic Gaussian (unequal variances
# are still the binormal ROC model) — at every n. Off the model (heavy
# tails, skew, multimodality), collapse that *worsens* with n: the
# parametric bias does not shrink, while the bands do. The envelope should
# be indistinguishable between the two regimes. This clean dichotomy —
# exactly calibrated under the assumption, catastrophic off it — only
# emerged after the hairline-violation correction (see Data loading): with
# the strict tolerance, sub-ulp phantom violations made WH look like it
# degraded with n even on its home turf.

# %% [markdown]
# ### Fig 5 — Side-by-side coverage heatmaps at 95%

# %%
fig, axes = plt.subplots(
    1, 2, figsize=(4.9 * 2, 0.62 * len(PRESENT_DGPS) + 1.6), constrained_layout=True
)
for ax, m in zip(axes, ["envelope", "working_hotelling"], strict=False):
    sub = dfb[(dfb.method == m) & (dfb.alpha == 0.05)]
    cov = (
        sub.groupby(["dgp_type", "n_total"], observed=True)["covers_entirely"]
        .mean()
        .unstack()
        .reindex(index=PRESENT_DGPS, columns=PRESENT_NS)
    )
    sns.heatmap(
        cov - 0.95,
        annot=cov,
        fmt=".3f",
        cmap="RdBu",
        center=0,
        vmin=-0.5,
        vmax=0.5,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Coverage − 0.95", "shrink": 0.85},
        annot_kws={"fontsize": 7.5},
        ax=ax,
    )
    ax.set_title(mlabel(m))
    ax.set_xlabel("Total sample size n")
    ax.set_ylabel("")
    ax.set_yticklabels(
        [dlabel(d) for d in PRESENT_DGPS] if ax is axes[0] else [], rotation=0
    )
    ax.grid(False)
fig.suptitle("95% bands: assumption-free calibration vs. parametric fragility")
fig.text(
    0.01, -0.025,
    "Each cell pools ~4,000 simulations; Monte Carlo SE per cell ≤ 0.008.",
    fontsize=8, color="0.4",
)
save_figure(fig, "fig05_envelope_vs_wh_heatmap")

# %% [markdown]
# ### Fig 6 — Coverage vs. n, per DGP, core method set
#
# Four methods tell the whole story: the envelope (calibrated everywhere),
# WH (calibrated only on binormal), KS (always 1.0 — safe but level-blind),
# and Wilson rectangles with Šidák (the pragmatic pointwise competitor that
# decays at large n).

# %%
CORE4 = ["envelope", "working_hotelling", "ks", "wilson_rectangle_sidak"]
ncols = min(4, len(PRESENT_DGPS))
nrows = int(np.ceil(len(PRESENT_DGPS) / ncols))
fig, axes = plt.subplots(
    nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows), sharex=True, sharey=True,
    squeeze=False,
)
for i, dgp in enumerate(PRESENT_DGPS):
    ax = axes.flat[i]
    for m in CORE4:
        g = coverage_by(
            dfb[(dfb.method == m) & (dfb.alpha == 0.05) & (dfb.dgp_type == dgp)],
            ["n_total"],
        ).sort_values("n_total")
        if g.empty:
            continue
        ax.plot(g["n_total"], g["coverage"], lw=1.7, ms=4, label=mlabel(m),
                **mstyle_noshape(m))
        ax.fill_between(g["n_total"], g["ci_lo"], g["ci_hi"],
                        color=mcolor(m), alpha=0.15, lw=0)
    add_nominal_line(ax, 0.05)
    ax.set_xscale("log")
    ax.set_title(dlabel(dgp), fontsize=10)
    ax.set_ylim(0, 1.04)
for j in range(len(PRESENT_DGPS), nrows * ncols):
    axes.flat[j].set_visible(False)
for ax in axes[-1, :]:
    ax.set_xlabel("Total sample size n")
for ax in axes[:, 0]:
    ax.set_ylabel("Coverage (95% band)")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
fig.suptitle(
    "Coverage vs. sample size: only the "
    "envelope is calibrated on every family"
)
fig.tight_layout()
save_figure(fig, "fig06_coverage_vs_n_by_dgp")

# %% [markdown]
# ### Fig 7 — How far from binormal before WH breaks?
#
# Coverage as a smooth function of the *departure* parameter: Student-t
# degrees of freedom (binormal as df → ∞) and bimodal mode separation
# (binormal at Δ = 0). This shows WH's failure is not an exotic corner case
# — it begins as soon as the departure is measurable, and grows with n.
# Ribbons (pointwise 95% GAM confidence bands) matter here: they pin down
# *where* WH's curve detaches from nominal, which is the quotable number.

# %%
FRAGILITY = [
    ("student_t", "dgp_df", "Degrees of freedom (heavier tails to the right)",
     True, True),
    ("bimodal_negative", "mode_separation", "Mode separation Δ", False, False),
]
FRAG_NS = [300, 3000]
frag_specs = [f for f in FRAGILITY if f[0] in PRESENT_DGPS]

fig, axes = plt.subplots(
    len(frag_specs), len(FRAG_NS),
    figsize=(4.4 * len(FRAG_NS), 3.2 * len(frag_specs)),
    sharey=True, squeeze=False,
)
for r, (dgp, col, xlabel, logx, invert) in enumerate(frag_specs):
    for c, n in enumerate(FRAG_NS):
        ax = axes[r, c]
        for m in ["envelope", "working_hotelling", "ks"]:
            ms = derive_shape_columns(
                dfb[(dfb.method == m) & (dfb.alpha == 0.05)
                    & (dfb.dgp_type == dgp) & (dfb.n_total == n)]
            ).dropna(subset=[col])
            if len(ms) < 200:
                continue
            x = ms[col].values.astype(float)
            gx = np.log10(x) if logx else x
            grid = np.linspace(gx.min(), gx.max(), 120)
            fit = fit_rate_curve(gx, ms["covers_entirely"].values, grid)
            if fit is not None:
                pred, lo, hi = fit
                st = mstyle(m)
                st.pop("marker")
                xs = 10**grid if logx else grid
                ax.plot(xs, pred, lw=2.0, label=mlabel(m), **st)
                ax.fill_between(xs, lo, hi, color=mcolor(m), alpha=0.14, lw=0)
        if logx:
            ax.set_xscale("log")
        if invert:
            ax.invert_xaxis()  # heavier tails to the right
        add_nominal_line(ax, 0.05)
        if r == 0:
            ax.set_title(f"n = {n:,}")
        ax.set_xlabel(xlabel)
        if c == 0:
            ax.set_ylabel(f"{dlabel(dgp)}\nCoverage (95% band)")
        ax.set_ylim(0, 1.04)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Working–Hotelling degrades continuously with departure from binormality")
fig.tight_layout()
save_figure(fig, "fig07_wh_fragility")

# %% [markdown]
# ---
# # 3. Tightness: coverage without paying the KS price
#
# **Motivation.** Valid coverage is cheap if you are willing to be vacuous —
# KS proves it. The claim to establish: the envelope is only modestly wider
# than WH (the parametric lower bound on width), dramatically tighter than
# KS, and — unlike KS — actually calibrated to its stated level.
#
# **Expectations.** On the area-vs-coverage plane the envelope should sit on
# or near the Pareto frontier: nominal-coverage column, left of KS. The
# area ratio vs. KS should be well below 1 and fall with n (the envelope
# width adapts to local variance; KS width is uniform).

# %% [markdown]
# ### Fig 8 — Coverage vs. mean band area (the Pareto view)
#
# Each point is the macro-average over DGPs. **No uncertainty is drawn here
# on purpose:** within a single DGP × n cell the Monte Carlo error is tiny
# (~0.5–0.8pp on coverage), but *pooled across DGPs* a method's coverage
# reflects genuine heterogeneity, not sampling noise — Working–Hotelling
# alone runs from ~0.95 on Gaussians to ~0.15 on heavy tails. A whisker on
# this aggregate would conflate the two and read as an implausibly wide
# "confidence interval." The honest, disaggregated view with real 95% CIs
# is Fig 8b (one panel per DGP × n); this panel is the bird's-eye summary.

# %%
PARETO_METHODS = [
    "envelope", "ks", "working_hotelling", "wilson_rectangle_sidak",
    "wilson_rectangle_bonferroni", "pointwise_sidak", "pointwise",
]
PARETO_NS = [100, 300, 1000, 10000]
fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.0), sharey=True)
for ax, n in zip(axes.flat, PARETO_NS, strict=False):
    sub = dfb[(dfb.alpha == 0.05) & (dfb.n_total == n)]
    for m in PARETO_METHODS:
        ms = sub[sub.method == m]
        if ms.empty:
            continue
        # macro-average per DGP first so partial DGPs don't skew the mean
        cells = ms.groupby("dgp_type", observed=True).agg(
            cov=("covers_entirely", "mean"), area=("band_area", "mean")
        )
        is_hero = m == "envelope"
        ax.scatter(
            cells["area"].mean(), cells["cov"].mean(),
            s=240 if is_hero else 90,
            marker=pareto_marker(m),
            color=mcolor(m), zorder=10 if is_hero else 5,
            edgecolors="black", linewidths=1.0 if is_hero else 0.4,
            label=mlabel(m),
        )
    add_nominal_line(ax, 0.05)
    ax.set_title(f"n = {n:,}")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
for ax in axes[-1, :]:
    ax.set_xlabel("Mean band area (smaller = tighter)")
for ax in axes[:, 0]:
    ax.set_ylabel("Coverage (95% band)")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
fig.suptitle(
    "Tightness vs. coverage: the envelope holds "
    "nominal coverage at a fraction of the KS area"
)
fig.tight_layout()
save_figure(fig, "fig08_pareto_area_coverage")

# %% [markdown]
# ### Fig 8b — Per-DGP Pareto with honest 95% confidence intervals
#
# **Motivation.** Fig 8 pools over DGPs, so its only "uncertainty" is
# cross-DGP heterogeneity. Here every panel is a single DGP × n cell, where
# the spread *is* Monte Carlo error and confidence intervals are meaningful.
# Each method is one point with a horizontal 95% CI on mean band area
# (normal, ±1.96·SE) and a vertical 95% Wilson CI on coverage. Columns are
# DGPs (grouped by family, matching Fig 15); rows are n = 100, 1k, 10k.
#
# **What to look for.** The CIs are small — coverage SE ~0.3–0.8pp at
# n_sims ≈ 4,000, band-area SE smaller still — which is the point: the
# envelope's near-nominal coverage and the competitors' shortfalls are
# resolved well beyond sampling noise within every individual cell. The
# wide-looking story in Fig 8 was heterogeneity across these panels, not
# imprecision within them.

# %%
PARETO_CI_METHODS = [
    "envelope", "ks", "working_hotelling", "wilson_rectangle_sidak",
    "pointwise_sidak",
]
PARETO_CI_NS = [100, 1000, 10000]


def pareto_points_with_ci(sub_cell: pd.DataFrame, method: str) -> dict | None:
    """Coverage (Wilson) and band-area (normal) point estimates with 95% CIs.

    Args:
        sub_cell: Trial rows already restricted to one DGP × n × alpha cell.
        method: Method name to summarize.

    Returns:
        Dict of point estimates and CI half-extents, or None if absent.
    """
    ms = sub_cell[sub_cell.method == method]
    if ms.empty:
        return None
    n_sims = len(ms)
    cov = ms["covers_entirely"].mean()
    cov_lo, cov_hi = wilson_ci(cov * n_sims, n_sims)
    area = ms["band_area"].mean()
    area_se = ms["band_area"].std(ddof=1) / np.sqrt(n_sims)
    return dict(
        cov=cov, cov_lo=float(cov_lo), cov_hi=float(cov_hi),
        area=area, area_err=1.96 * area_se,
    )


def plot_pareto_ci_family(family: str, dgps: list[str], fig_name: str) -> None:
    """Draw the per-DGP Pareto-with-CI grid for one DGP family."""
    present = [d for d in dgps if d in PRESENT_DGPS]
    fig, axes = plt.subplots(
        len(PARETO_CI_NS), len(present),
        figsize=(3.4 * len(present), 3.0 * len(PARETO_CI_NS)),
        sharex="col", squeeze=False,
    )
    for r, n in enumerate(PARETO_CI_NS):
        for c, dgp in enumerate(present):
            ax = axes[r, c]
            cell = dfb[(dfb.alpha == 0.05) & (dfb.n_total == n)
                       & (dfb.dgp_type == dgp)]
            for m in PARETO_CI_METHODS:
                pt = pareto_points_with_ci(cell, m)
                if pt is None:
                    continue
                is_hero = m == "envelope"
                # Wilson interval is asymmetric and, at p = 1, its upper end
                # sits below 1; clip offsets so errorbar never sees a
                # negative extent.
                yerr_lo = max(0.0, pt["cov"] - pt["cov_lo"])
                yerr_hi = max(0.0, pt["cov_hi"] - pt["cov"])
                ax.errorbar(
                    pt["area"], pt["cov"],
                    xerr=pt["area_err"],
                    yerr=[[yerr_lo], [yerr_hi]],
                    fmt="none", ecolor=mcolor(m), elinewidth=1.0,
                    capsize=2, alpha=0.7, zorder=4,
                )
                ax.scatter(
                    pt["area"], pt["cov"],
                    s=150 if is_hero else 60, marker=pareto_marker(m),
                    color=mcolor(m), edgecolors="black",
                    linewidths=0.9 if is_hero else 0.4,
                    zorder=10 if is_hero else 5, label=mlabel(m),
                )
            add_nominal_line(ax, 0.05)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(left=0)
            if r == 0:
                ax.set_title(dlabel(dgp), fontsize=10)
            if r == len(PARETO_CI_NS) - 1:
                ax.set_xlabel("Mean band area")
            if c == 0:
                ax.set_ylabel(f"n = {n:,}\nCoverage (95% band)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(PARETO_CI_METHODS),
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Per-DGP coverage–tightness with 95% CIs — {family}")
    fig.tight_layout()
    save_figure(fig, fig_name)


# %%
plot_pareto_ci_family(
    "Gaussian-like DGPs",
    DGP_FAMILIES["Gaussian-like"],
    "fig08b_pareto_ci_gaussian_like",
)

# %%
plot_pareto_ci_family(
    "Heavy-tailed / skewed DGPs",
    DGP_FAMILIES["Heavy-tailed / skewed"],
    "fig08b_pareto_ci_heavy_tailed",
)

# %%
plot_pareto_ci_family(
    "Non-standard shapes",
    DGP_FAMILIES["Non-standard shape"],
    "fig08b_pareto_ci_nonstandard",
)

# %% [markdown]
# ### Fig 8c — Envelope vs. Working–Hotelling: the coverage cost of tightening
#
# **Motivation.** The two figures above are snapshots at fixed n. This one
# traces the *trajectory* as n grows, for the two methods whose contrast is
# the paper's thesis: the envelope (assumption-free) and Working–Hotelling
# (the tight parametric benchmark). Each line is one method; each marker is
# a sample size, colored by n on a viridis scale; line style distinguishes
# the methods. As n grows both bands tighten (markers march leftward toward
# zero area); the question is what happens to coverage on the way.
#
# Columns group DGPs by *departure from binormality* — the axis this figure
# is about — which here coincides with distributional quality and with the
# severity of WH's failure (verified against WH coverage at n = 10,000):
#
# - **Gaussian** (binormal, hetero. Gaussian): WH's model holds; coverage
#   ≈ 0.56 / 0.81 even at n = 10k.
# - **Heavy-tailed / skewed** (Student-t, gamma, Weibull): unbounded heavy
#   or skewed tails WH cannot fit; coverage collapses to ≈ 0–0.02.
# - **Bounded / multimodal** (logit-normal, beta-opposing, bimodal): bounded
#   support or mixtures, no heavy tails; WH fails moderately (≈ 0.11–0.28).
#   Logit-normal sits here, not with the Gaussians: although its ROC is
#   binormal-shaped, WH estimates the binormal parameters from the skewed
#   bounded *scores*, so its behavior matches this tier, not column one.
#
# Rows are AUC tertiles (low / mid / high true AUC), since ROC geometry is
# the first-order risk factor. Data are macro-averaged over DGPs in a column.
#
# **Expectation.** The envelope's trajectory stays pinned near the 0.95 line
# as it slides left. Working–Hotelling slides left too, but on the
# heavy-tailed and high-AUC panels its trajectory *plunges* — tightening
# while shedding coverage, the worst quadrant of this plane.

# %%
# Figure-local grouping by departure-from-binormality (see markdown). Distinct
# from the ROC-shape DGP_FAMILIES used elsewhere: logit-normal moves to the
# bounded group, and the Gaussian column is the two genuinely normal-score DGPs.
TRAJ_FAMILIES = {
    "Gaussian": ["binormal", "hetero_gaussian"],
    "Heavy-tailed / skewed": ["student_t", "gamma", "weibull"],
    "Bounded / multimodal": ["logitnormal", "beta_opposing", "bimodal_negative"],
}

# Global AUC tertiles, so rows mean the same thing across all panels. Labels
# carry the actual AUC range (2 d.p.) so readers can see where the cuts fall.
auc_edges = dfb["true_auc"].quantile([0, 1 / 3, 2 / 3, 1.0]).values
auc_disp = auc_edges.copy()  # display edges before the inclusivity nudge
auc_edges[-1] += 1e-6  # make the top tertile's upper bound inclusive
AUC_TERTILES = [
    (auc_edges[0], auc_edges[1], f"Low AUC\n[{auc_disp[0]:.2f}, {auc_disp[1]:.2f})"),
    (auc_edges[1], auc_edges[2], f"Mid AUC\n[{auc_disp[1]:.2f}, {auc_disp[2]:.2f})"),
    (auc_edges[2], auc_edges[3], f"High AUC\n[{auc_disp[2]:.2f}, {auc_disp[3]:.2f}]"),
]
TRAJ_METHODS = [("envelope", "-"), ("working_hotelling", "--")]
TRAJ_NS = N_ORDER

fig, axes = plt.subplots(
    len(AUC_TERTILES), len(TRAJ_FAMILIES),
    figsize=(4.0 * len(TRAJ_FAMILIES), 3.2 * len(AUC_TERTILES)),
    sharex="col", sharey=True, squeeze=False, constrained_layout=True,
)
for r, (auc_lo, auc_hi, auc_label) in enumerate(AUC_TERTILES):
    for c, (family, fam_dgps) in enumerate(TRAJ_FAMILIES.items()):
        ax = axes[r, c]
        fam_present = [d for d in fam_dgps if d in PRESENT_DGPS]
        band = dfb[(dfb.alpha == 0.05) & (dfb.true_auc >= auc_lo)
                   & (dfb.true_auc < auc_hi) & (dfb.dgp_type.isin(fam_present))]
        for method, ls in TRAJ_METHODS:
            xs, ys, ns = [], [], []
            for n in TRAJ_NS:
                ms = band[(band.method == method) & (band.n_total == n)]
                if ms.empty:
                    continue
                # macro-average over DGPs within the family
                per_dgp = ms.groupby("dgp_type", observed=True).agg(
                    cov=("covers_entirely", "mean"), area=("band_area", "mean")
                )
                xs.append(per_dgp["area"].mean())
                ys.append(per_dgp["cov"].mean())
                ns.append(n)
            if not xs:
                continue
            ax.plot(xs, ys, ls=ls, color="0.55", lw=1.3, zorder=3)
            ax.scatter(
                xs, ys, c=[n_color(n) for n in ns], s=55,
                edgecolors="black", linewidths=0.5, zorder=5,
            )
        add_nominal_line(ax, 0.05)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(left=0)
        if r == 0:
            ax.set_title(family, fontsize=10)
        if r == len(AUC_TERTILES) - 1:
            ax.set_xlabel("Mean band area")
        if c == 0:
            ax.set_ylabel(f"{auc_label}\nCoverage (95% band)")

# Legends: method line style, and an n colorbar
style_handles = [
    Line2D([], [], color="0.4", ls=ls, lw=1.5, label=mlabel(m))
    for m, ls in TRAJ_METHODS
]
fig.legend(handles=style_handles, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.04))
sm = mpl.cm.ScalarMappable(norm=_N_NORM, cmap="viridis")
cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01)
cbar.set_label("Sample size n")
cbar.set_ticks(TRAJ_NS)
cbar.set_ticklabels([f"{n:,}" for n in TRAJ_NS])
fig.suptitle(
    "Envelope vs. Working–Hotelling: band-tightening trajectory over n"
)
save_figure(fig, "fig08c_trajectory_envelope_vs_wh")

# %% [markdown]
# ### Fig 9 — Width premium and calibration, head-to-head with KS and WH
#
# Left: where each method's band area sits on the WH→KS width axis,
# normalized per DGP × n as (area − area_WH) / (area_KS − area_WH). So 0 is
# the tight parametric floor (Working–Hotelling), 1 is the conservative
# distribution-free ceiling (KS), and 0.6 means "60% of the way from WH to
# KS." This shows the width *cost of robustness* on an interpretable scale.
# Vertical bars on the left are 95% CIs of the across-DGP mean at each n
# (Student t, D−1 df), carrying both Monte Carlo error and DGP heterogeneity.
# Right: absolute calibration error |coverage − nominal| at 95%,
# macro-averaged across DGPs — the metric on which the envelope should
# dominate both anchors.


def errorbar_across_dgps(ax, ns, per_n_values, *, method, floor0=True):
    """Plot a method's across-DGP mean over n with 95% t-CI whiskers.

    Args:
        ax: Target axes.
        ns: Sample sizes (x).
        per_n_values: List aligned with ns; each entry is the array of
            per-DGP statistic values at that n.
        method: Method key for color/style.
        floor0: Clip the lower whisker at zero for non-negative quantities.
    """
    means, lo_err, hi_err = [], [], []
    for vals in per_n_values:
        m, lo, hi = mean_ci_across(vals)
        means.append(m)
        lo_floor = max(lo, 0.0) if floor0 else lo
        lo_err.append(m - lo_floor)
        hi_err.append(hi - m)
    ax.errorbar(
        ns, means, yerr=[lo_err, hi_err], capsize=2.5, lw=1.8, ms=5,
        label=mlabel(method), **mstyle_noshape(method),
    )


# %%
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))

ax = axes[0]
sub = dfb[dfb.alpha == 0.05]
for m in ["envelope", "wilson_rectangle_sidak"]:
    per_n = []
    for n in PRESENT_NS:
        cells = []
        for dgp in PRESENT_DGPS:
            a_wh = sub[(sub.method == "working_hotelling") & (sub.dgp_type == dgp)
                       & (sub.n_total == n)]["band_area"].mean()
            a_ks = sub[(sub.method == "ks") & (sub.dgp_type == dgp)
                       & (sub.n_total == n)]["band_area"].mean()
            a_m = sub[(sub.method == m) & (sub.dgp_type == dgp)
                      & (sub.n_total == n)]["band_area"].mean()
            span = a_ks - a_wh
            if np.isfinite(a_m) and np.isfinite(span) and span > 0:
                cells.append((a_m - a_wh) / span)
        per_n.append(np.array(cells))
    errorbar_across_dgps(ax, PRESENT_NS, per_n, method=m, floor0=False)
ax.axhline(0.0, color=mcolor("working_hotelling"), ls=":", lw=1.2)
ax.axhline(1.0, color=mcolor("ks"), ls=":", lw=1.2)
ax.text(PRESENT_NS[0], 0.0, " WH (tightest)", va="bottom", ha="left",
        fontsize=8, color=mcolor("working_hotelling"))
ax.text(PRESENT_NS[0], 1.0, " KS (widest)", va="bottom", ha="left",
        fontsize=8, color=mcolor("ks"))
ax.set_xscale("log")
ax.set_xlabel("Total sample size n")
ax.set_ylabel("Band area, fraction of WH→KS range")
ax.set_ylim(-0.05, 1.12)
ax.set_title("Width on the WH→KS scale")
ax.legend(loc="upper right")

ax = axes[1]
for m in ["envelope", "ks", "working_hotelling", "wilson_rectangle_sidak"]:
    errs = []
    for n in PRESENT_NS:
        cells = (
            sub[(sub.method == m) & (sub.n_total == n)]
            .groupby("dgp_type", observed=True)["covers_entirely"].mean()
        )
        errs.append(np.abs(cells.values - 0.95).mean() if len(cells) else np.nan)
    ax.plot(PRESENT_NS, errs, lw=1.8, ms=5, label=mlabel(m), **mstyle_noshape(m))
ax.set_xscale("log")
ax.set_xlabel("Total sample size n")
ax.set_ylabel("Mean |coverage − 0.95| across DGPs")
ax.set_title("Calibration error (lower is better)")
ax.legend(fontsize=8)

fig.suptitle(
    "Band width and calibration relative to the KS and "
    "Working–Hotelling references, by sample size"
)
fig.tight_layout()
save_figure(fig, "fig09_tightness_vs_ks")

# %% [markdown]
# ### Fig 10 — Width anatomy across the FPR axis
#
# Per-DGP median band width at fixed FPR landmarks (n = 300 and n = 1,000,
# 95% bands), averaged across DGPs with 95% CIs (across-DGP t-interval).
# This is where the envelope's *adaptive* width shows: wide where the ROC is
# uncertain (low FPR on steep curves), narrow where it is pinned — versus
# the uniform-width KS band and the shape-blind rectangles.

# %%
WIDTH_METHODS = ["envelope", "ks", "working_hotelling", "wilson_rectangle_sidak"]
WIDTH_NS = [300, 1000]
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
for ax, n in zip(axes, WIDTH_NS, strict=False):
    sub = dfb[(dfb.alpha == 0.05) & (dfb.n_total == n)]
    for m in WIDTH_METHODS:
        ms = sub[sub.method == m]
        if ms.empty:
            continue
        per_lm = [
            ms.groupby("dgp_type", observed=True)[WIDTH_COLS[x]].median().values
            for x in WIDTH_LANDMARKS
        ]
        errorbar_across_dgps(ax, WIDTH_LANDMARKS, per_lm, method=m)
    ax.set_xscale("log")
    ax.set_xlabel("FPR")
    ax.set_title(f"n = {n:,}")
axes[0].set_ylabel("Median band width (TPR units)")
axes[0].legend(fontsize=8)
fig.suptitle(
    "Band width by FPR: the envelope spends "
    "width where uncertainty actually lives"
)
fig.tight_layout()
save_figure(fig, "fig10_width_profile")

# %% [markdown]
# ---
# # 4. Anatomy: why the naive bootstrap fails and what each repair fixes
#
# **Motivation.** This section earns the method's complexity. The argument
# has four steps, each with its own ablation evidence in the run:
#
# 1. **The bare bootstrap envelope is not a confidence band** — boundary
#    variance collapse destroys it at *both* corners
#    (`envelope_no_floors`).
# 2. **Each floor repairs a different corner.** The Wilson rectangle floor
#    owns the TPR plateau (upper-right); the Beta order-statistic floor owns
#    the steep low-FPR corner (`envelope_no_wilson_floor`,
#    `envelope_no_beta_floor`).
# 3. **Neither mechanism alone gives calibration *and* tightness.**
#    Mirroring a single mechanism onto both tails
#    (`envelope_beta_both_tails`, `envelope_wilson_both_tails`) leaves a
#    characteristic cost: Beta-only is leaky in the low-FPR region beyond
#    its jurisdiction (it also forfeits the Wilson variance floor inside
#    the studentization), while Wilson-only keeps coverage only through
#    the rectangle's vacuous corner geometry — a large width premium at
#    operational FPRs.
# 4. **The floors alone are not enough either.** Without the bootstrap
#    interior (`envelope_no_bootstrap`), simultaneity is handled by Šidák
#    over the grid instead of by the resampled correlation structure, and
#    calibration across levels collapses.

# %% [markdown]
# ### Fig 11 — The two-corner failure of the bare bootstrap envelope
#
# Left: coverage vs. n. Right: where the bare envelope's violations live —
# violation rate per FPR region (rows) × n (columns), versus the same map
# for the full method. **Expectation:** the bare envelope fails massively
# at the 90–100% region (bootstrap variance pinned to zero at the plateau)
# and substantially at 0–10% (one-sided support collapse), and *more* so as
# n grows; the full method's map should be near-blank.

# %%
def region_violation_matrix(method: str, alpha: float = 0.05) -> pd.DataFrame:
    """Violation rate per FPR region × n for one method (prev = 0.5)."""
    sub = dfb[(dfb.method == method) & (dfb.alpha == alpha)]
    rates = sub.groupby("n_total", observed=True)[REGION_COLS].mean().T
    rates.index = [f"{r}%" for r in REGIONS]
    return rates.reindex(columns=PRESENT_NS)


fig = plt.figure(figsize=(12.6, 4.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.32)

ax = fig.add_subplot(gs[0, 0])
for m in ["envelope", "envelope_no_floors"]:
    g = coverage_by(dfb[(dfb.method == m) & (dfb.alpha == 0.05)], ["n_total"])
    g = g.sort_values("n_total")
    ax.plot(g["n_total"], g["coverage"], lw=1.9, ms=4.5, label=mlabel(m), **mstyle(m))
add_nominal_line(ax, 0.05)
ax.set_xscale("log")
ax.set_xlabel("Total sample size n")
ax.set_ylabel("Coverage (95% band)")
ax.set_title("Coverage")
ax.set_ylim(0, 1.04)
ax.legend(fontsize=8, loc="center right")

for k, m in enumerate(["envelope_no_floors", "envelope"]):
    ax = fig.add_subplot(gs[0, k + 1])
    mat = region_violation_matrix(m)
    sns.heatmap(
        mat, annot=True, fmt=".2f", cmap="Reds", vmin=0, vmax=0.6,
        linewidths=0.6, linecolor="white", cbar=k == 1,
        cbar_kws={"label": "Violation rate", "shrink": 0.85},
        annot_kws={"fontsize": 7}, ax=ax,
    )
    ax.set_title(mlabel(m), fontsize=10)
    ax.set_xlabel("n")
    ax.set_ylabel("FPR region" if k == 0 else "")
    ax.grid(False)

fig.suptitle(
    "The bare bootstrap envelope fails at both "
    "corners — the full method repairs both"
)
save_figure(fig, "fig11_bare_bootstrap_failure")

# %% [markdown]
# ### Fig 12 — Floor ablation: each floor is load-bearing at its own corner
#
# Same color, different line styles — these are variants of one method.
# Panels (c) and (d) split violations by corner region, which is where the
# attribution becomes unambiguous: removing the Beta floor should move the
# 0–10% curve and leave 90–100% alone; removing the Wilson floor should do
# the reverse.

# %%
ABLATION = ["envelope", "envelope_no_beta_floor", "envelope_no_wilson_floor",
            "envelope_no_floors"]
fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.6))

panel_specs = [
    ("Coverage — all AUC", None, "covers_entirely", True),
    (f"Coverage — true AUC > {HIGH_AUC}", HIGH_AUC, "covers_entirely", True),
    ("Violation rate, FPR 0–10% (steep corner)", None, "violation_0-10", False),
    ("Violation rate, FPR 90–100% (plateau corner)", None, "violation_90-100", False),
]
for ax, (title, auc_min, col, is_cov) in zip(axes.flat, panel_specs, strict=False):
    sub = dfb[dfb.alpha == 0.05]
    if auc_min is not None:
        sub = sub[sub.true_auc > auc_min]
    for m in ABLATION:
        g = (
            sub[sub.method == m]
            .groupby("n_total", observed=True)[col]
            .mean()
            .reindex(PRESENT_NS)
        )
        ax.plot(g.index, g.values, lw=1.8, ms=4.5, label=mlabel(m), **mstyle(m))
    if is_cov:
        add_nominal_line(ax, 0.05)
        ax.set_ylim(0, 1.04)
        ax.set_ylabel("Coverage (95% band)")
    else:
        ax.set_ylabel("Violation rate")
        ax.set_ylim(bottom=0)
    ax.set_xscale("log")
    ax.set_xlabel("Total sample size n")
    ax.set_title(title, fontsize=10)
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))
fig.suptitle(
    "Floor ablation: the Beta floor owns the steep "
    "corner, the Wilson floor owns the plateau"
)
fig.tight_layout()
save_figure(fig, "fig12_floor_ablation")

# %% [markdown]
# ### Fig 13 — One mechanism cannot serve both tails (without paying for it)
#
# The symmetric-tail variants stretch a *single* repair mechanism across
# both corners. The run shows that each can be made "safe enough" — but
# each pays a characteristic price that the hybrid avoids:
#
# - **Beta floors on both tails (no Wilson machinery)** also loses the
#   Wilson *variance floor inside the studentization*, and its low-FPR
#   protection ends at the Beta jurisdiction (k ≲ 43). Result: roughly
#   double the full method's violation rate in the FPR 0–10% region at
#   every n ≥ 100, and consistently lower overall coverage.
# - **Wilson floors on both tails (no Beta floor)** maintains coverage —
#   but only because the rectangle's corner geometry is *protective by
#   vacuity*: forced onto the steep corner it surrenders the lower bound.
#   The price is width exactly where practitioners operate: ~30–45% wider
#   at FPR = 0.05 for n = 300–1,000.
#
# (A nuance worth a sentence in the paper: the mirrored Beta floor *does*
# handle the TPR plateau's lower bound — plateau failures are also a
# support problem — so the Wilson rectangle's irreplaceable contribution is
# the variance floor plus the plateau repair at much lower width cost.)

# %%
SYMM = ["envelope", "envelope_beta_both_tails", "envelope_wilson_both_tails"]
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))

ax = axes[0]
for m in SYMM:
    g = coverage_by(dfb[(dfb.method == m) & (dfb.alpha == 0.05)], ["n_total"])
    g = g.sort_values("n_total")
    ax.plot(g["n_total"], g["coverage"], lw=1.8, ms=4.5, label=mlabel(m), **mstyle(m))
add_nominal_line(ax, 0.05)
ax.set_xscale("log")
ax.set_ylim(0.90, 1.005)
ax.set_xlabel("Total sample size n")
ax.set_ylabel("Coverage (95% band)")
ax.set_title("Overall coverage")
ax.legend(fontsize=7.5, loc="lower left")

ax = axes[1]
for m in SYMM:
    g = (
        dfb[(dfb.method == m) & (dfb.alpha == 0.05)]
        .groupby("n_total", observed=True)["violation_0-10"]
        .mean()
        .reindex(PRESENT_NS)
    )
    ax.plot(g.index, g.values, lw=1.8, ms=4.5, label=mlabel(m), **mstyle(m))
ax.set_xscale("log")
ax.set_xlabel("Total sample size n")
ax.set_ylabel("Violation rate, FPR 0–10%")
ax.set_title("Beta-only: unprotected gap\nbeyond the Beta jurisdiction", fontsize=10)
ax.set_ylim(bottom=0)

ax = axes[2]
for m in SYMM:
    g = (
        dfb[(dfb.method == m) & (dfb.alpha == 0.05)]
        .groupby("n_total", observed=True)["width_at_fpr_0.05"]
        .mean()
        .reindex(PRESENT_NS)
    )
    ax.plot(g.index, g.values, lw=1.8, ms=4.5, label=mlabel(m), **mstyle(m))
ax.set_xscale("log")
ax.set_xlabel("Total sample size n")
ax.set_ylabel("Mean band width at FPR = 0.05")
ax.set_title("Wilson-only: coverage bought\nwith vacuous low-FPR width", fontsize=10)

fig.suptitle(
    "Symmetric-tail ablation: a single mechanism is "
    "either leaky or vacuous — the hybrid is neither"
)
fig.tight_layout()
save_figure(fig, "fig13_symmetric_tail_ablation")

# %% [markdown]
# ### Fig 14 — The floors alone are not a method
#
# `envelope_no_bootstrap` keeps the Wilson rectangle + Beta floors and drops
# the bootstrap interior entirely. **Expectation:** acceptable-looking
# coverage at 95% (the floors are conservative), but (a) badly miscalibrated
# across confidence levels — there is no resampled correlation structure to
# scale with alpha — and (b) wider than the full method wherever the
# bootstrap interior would have adapted. The bootstrap is what makes the
# band *informative*; the floors only make it *safe*.

# %%
NOBOOT = ["envelope", "envelope_no_bootstrap", "wilson_rectangle_sidak"]
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))

for ax, alpha in zip(axes[:2], [0.05, 0.5], strict=False):
    for m in NOBOOT:
        g = coverage_by(dfb[(dfb.method == m) & (dfb.alpha == alpha)], ["n_total"])
        g = g.sort_values("n_total")
        ax.plot(g["n_total"], g["coverage"], lw=1.8, ms=4.5, label=mlabel(m),
                **mstyle(m))
    add_nominal_line(ax, alpha)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("Total sample size n")
    ax.set_ylabel("Coverage")
    ax.set_title(f"{(1 - alpha) * 100:.0f}% band")
axes[0].legend(fontsize=7.5, loc="lower left")

ax = axes[2]
for m in NOBOOT:
    g = (
        dfb[(dfb.method == m) & (dfb.alpha == 0.05)]
        .groupby("n_total", observed=True)["band_area"]
        .mean()
        .reindex(PRESENT_NS)
    )
    ax.plot(g.index, g.values, lw=1.8, ms=4.5, label=mlabel(m), **mstyle(m))
ax.set_xscale("log")
ax.set_xlabel("Total sample size n")
ax.set_ylabel("Mean band area")
ax.set_title("Width (95% band)")
fig.suptitle(
    "Without the bootstrap interior the band "
    "is safe but neither tight nor tunable"
)
fig.tight_layout()
save_figure(fig, "fig14_no_bootstrap_ablation")

# %% [markdown]
# ---
# # 5. What the band looks like: individual examples with mechanism shading
#
# **Motivation.** Readers should be able to *see* the hybrid architecture:
# the bootstrap envelope sets the band in the interior, the Wilson rectangle
# floor takes over on the TPR plateau, and the Beta order-statistic floor
# governs the extreme low-FPR points. We draw one n = 300 example per
# DGP × target AUC: empirical ROC, true ROC, the 95% band, and light
# background shading marking which mechanism *determines the lower bound*
# at each FPR (the upper bound is bootstrap essentially everywhere).
#
# Mechanism attribution is exact, following the full method's own pipeline
# (variance-floored retention envelope -> Wilson rectangle -> Beta floor),
# using the suite's `envelope_pre_floor` diagnostic — the bootstrap arm of
# the full method before either floor is applied. A grid point is
# "Beta-driven" where the Beta floor strictly lowered the post-rectangle
# bound, "Wilson-driven" where the rectangle strictly lowered the bootstrap
# arm, and bootstrap otherwise. Ties go to the bootstrap: a floor that
# merely matches the arm did not set the bound. The one exception is the
# zone below the first Beta quantile q_1 ≈ 6.9/n₀, where the floor enforces
# the vacuous bound of 0 regardless of what the arm claims — that zone is
# credited to the Beta floor even when the arm coincides at 0. Note the
# Wilson *variance* floor inside the studentization is counted as part of
# the bootstrap envelope: the resulting arm is still the envelope of
# retained bootstrap curves, not an external bound.
#
# These panels use fresh draws (fixed seeds) — they are illustrations, not
# part of the simulation evidence.

# %%
import torch  # noqa: E402
from scipy.stats import beta as beta_dist  # noqa: E402

from studroc_paper.datagen.roc_to_dgp import map_lhs_to_dgp  # noqa: E402
from studroc_paper.datagen.true_rocs import (  # noqa: E402
    make_beta_opposing_skew_dgp,
    make_bimodal_negative_dgp,
    make_gamma_dgp,
    make_heteroskedastic_gaussian_dgp,
    make_logitnormal_dgp,
    make_student_t_dgp,
    make_weibull_dgp,
)
from studroc_paper.methods.envelope_boot import envelope_band_suite  # noqa: E402
from studroc_paper.methods.method_utils import (  # noqa: E402
    compute_empirical_roc_from_scores,
)
from studroc_paper.sampling.bootstrap_grid import generate_bootstrap_grid  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DGP_FACTORY = {
    "binormal": make_heteroskedastic_gaussian_dgp,
    "logitnormal": make_logitnormal_dgp,
    "hetero_gaussian": make_heteroskedastic_gaussian_dgp,
    "beta_opposing": make_beta_opposing_skew_dgp,
    "student_t": make_student_t_dgp,
    "bimodal_negative": make_bimodal_negative_dgp,
    "weibull": make_weibull_dgp,
    "gamma": make_gamma_dgp,
}

# Representative (interesting, not adversarial) shape parameters per DGP
EXAMPLE_SHAPE: dict[str, dict] = {
    "binormal": {},
    "hetero_gaussian": {"sigma_ratio": 2.0},
    "logitnormal": {"sigma": 1.5},
    "student_t": {"df": 3.0},
    "gamma": {"shape": 2.0},
    "weibull": {"shape": 1.5},
    "beta_opposing": {"alpha": 2.0},
    "bimodal_negative": {"mixture_weight": 0.35, "mode_separation": 2.5},
}
EXAMPLE_AUCS = [0.70, 0.85, 0.95]
EXAMPLE_N = 300
EXAMPLE_B = 4000
EXAMPLE_ALPHA = 0.05


def make_example_dgp(dgp_type: str, auc: float):
    """Build a DGP instance at a target AUC with representative shape params."""
    lhs = {"auc": np.array([auc])} | {
        k: np.array([v]) for k, v in EXAMPLE_SHAPE[dgp_type].items()
    }
    params = map_lhs_to_dgp(dgp_type, lhs)
    scalar_params = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray) and v.ndim > 0:
            scalar_params[k] = float(v[0])
        elif isinstance(v, list):
            scalar_params[k] = v[0]
        else:
            scalar_params[k] = v
    return DGP_FACTORY[dgp_type](**scalar_params)


def compute_example_band(
    dgp_type: str, auc: float, seed: int, n: int = EXAMPLE_N
) -> dict:
    """Sample one dataset and compute the envelope band suite on it.

    Args:
        dgp_type: DGP family key into DGP_FACTORY / EXAMPLE_SHAPE.
        auc: Target true AUC for the DGP parameterization.
        seed: Seed for the data draw.
        n: Total sample size (split evenly between classes).

    Returns:
        Dict with the FPR grid, empirical/true TPR, the full band, and
        boolean masks attributing the lower bound to the Beta floor and the
        Wilson rectangle floor.
    """
    dgp = make_example_dgp(dgp_type, auc)
    rng = np.random.default_rng(seed)
    n_pos = n_neg = n // 2
    pos, neg = dgp.sample(n_pos, n_neg, rng)
    y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    y_score = np.concatenate([pos, neg]).astype(np.float64)
    fpr_grid = np.linspace(0, 1, n_neg + 1)

    boot = generate_bootstrap_grid(
        y_true=torch.as_tensor(y_true, dtype=torch.float32, device=DEVICE),
        y_score=torch.as_tensor(y_score, dtype=torch.float32, device=DEVICE),
        B=EXAMPLE_B,
        grid=torch.as_tensor(fpr_grid, dtype=torch.float32, device=DEVICE),
        device=DEVICE,
    )
    suite = envelope_band_suite(
        boot_tpr_matrix=boot,
        fpr_grid=fpr_grid,
        y_true=y_true,
        y_score=y_score,
        alphas=[EXAMPLE_ALPHA],
        include_pre_floor_arm=True,
    )[EXAMPLE_ALPHA]

    lower, upper = suite["envelope"]
    lower_no_beta = suite["envelope_no_beta_floor"][0]
    lower_arm = suite["envelope_pre_floor"][0]

    # Exact attribution along the full method's own pipeline:
    #   arm (variance-floored retention envelope) -> rectangle -> Beta.
    # A point is Beta-driven where the Beta floor strictly lowered the
    # post-rectangle bound, Wilson-driven where the rectangle strictly
    # lowered the bootstrap arm, and bootstrap otherwise. Ties go to the
    # bootstrap — a floor that merely matches the arm did not set the bound.
    # Sole exception: below the first Beta quantile q_1 the floor enforces
    # the vacuous bound of 0 regardless of the arm, so that zone is credited
    # to the Beta floor even when the arm is also 0.
    eps = 1e-6
    q_1 = beta_dist.ppf(1 - EXAMPLE_ALPHA / 50, 1, n_neg)
    vacuous_zone = (fpr_grid > 0) & (fpr_grid < q_1)
    beta_mask = (lower < lower_no_beta - eps) | vacuous_zone
    wilson_mask = (~beta_mask) & (lower_no_beta < lower_arm - eps)

    emp_tpr = (
        compute_empirical_roc_from_scores(
            neg_scores=torch.as_tensor(neg, dtype=torch.float64),
            pos_scores=torch.as_tensor(pos, dtype=torch.float64),
            fpr_grid=torch.as_tensor(fpr_grid, dtype=torch.float64),
        )
        .cpu()
        .numpy()
    )
    return dict(
        fpr=fpr_grid,
        lower=lower,
        upper=upper,
        emp=emp_tpr,
        true=dgp.get_true_roc(fpr_grid),
        beta=beta_mask,
        wilson=wilson_mask,
    )


BETA_SHADE = dict(color="#F0E442", alpha=0.45)
WILSON_SHADE = dict(color="#009E73", alpha=0.16)


def plot_example_family(family: str, dgps: list[str], fig_name: str) -> None:
    """Draw the example-band panel grid for one DGP family."""
    fig, axes = plt.subplots(
        len(EXAMPLE_AUCS), len(dgps),
        figsize=(3.3 * len(dgps), 3.15 * len(EXAMPLE_AUCS)),
        sharex=True, sharey=True, squeeze=False,
    )
    for c, dgp_type in enumerate(dgps):
        for r, auc in enumerate(EXAMPLE_AUCS):
            ax = axes[r, c]
            ex = compute_example_band(dgp_type, auc, seed=20260612 + 13 * c + r)
            for x0, x1 in contiguous_spans(ex["beta"], ex["fpr"], min_len=2):
                ax.axvspan(x0, x1, **BETA_SHADE, lw=0, zorder=0)
            for x0, x1 in contiguous_spans(ex["wilson"], ex["fpr"], min_len=2):
                ax.axvspan(x0, x1, **WILSON_SHADE, lw=0, zorder=0)
            ax.fill_between(ex["fpr"], ex["lower"], ex["upper"],
                            color="#D55E00", alpha=0.18, lw=0, zorder=2)
            ax.plot(ex["fpr"], ex["lower"], color="#D55E00", lw=1.1, zorder=3)
            ax.plot(ex["fpr"], ex["upper"], color="#D55E00", lw=1.1, zorder=3)
            ax.plot(ex["fpr"], ex["emp"], color="0.25", lw=1.3, zorder=4)
            ax.plot(ex["fpr"], ex["true"], color="black", ls="--", lw=1.5, zorder=5)
            ax.plot([0, 1], [0, 1], color="0.7", ls=":", lw=0.8, zorder=1)
            ax.set_xlim(-0.02, 1.0)
            ax.set_ylim(0, 1.02)
            ax.set_aspect("equal")
            ax.grid(False)
            if r == 0:
                shape_txt = ", ".join(
                    f"{k}={v:g}" for k, v in EXAMPLE_SHAPE[dgp_type].items()
                )
                ax.set_title(
                    dlabel(dgp_type) + (f"\n({shape_txt})" if shape_txt else ""),
                    fontsize=9.5,
                )
            if c == 0:
                ax.set_ylabel(f"target AUC = {auc:.2f}\nTPR")
            if r == len(EXAMPLE_AUCS) - 1:
                ax.set_xlabel("FPR")
    legend = [
        Line2D([], [], color="black", ls="--", lw=1.5, label="True ROC"),
        Line2D([], [], color="0.25", lw=1.3, label="Empirical ROC"),
        Patch(facecolor="#D55E00", alpha=0.25, label="95% simultaneous band"),
        Patch(facecolor=BETA_SHADE["color"], alpha=BETA_SHADE["alpha"],
              label="Lower bound set by Beta order-statistic floor"),
        Patch(facecolor=WILSON_SHADE["color"], alpha=WILSON_SHADE["alpha"] + 0.15,
              label="Lower bound set by Wilson rectangle floor"),
        Patch(facecolor="white", edgecolor="0.7",
              label="Unshaded: bootstrap envelope"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        f"Example 95% bands at n = {EXAMPLE_N} — {family}", y=1.005
    )
    fig.tight_layout()
    save_figure(fig, fig_name)


# %%
plot_example_family(
    "Gaussian-like DGPs",
    ["binormal", "hetero_gaussian", "logitnormal"],
    "fig15a_example_bands_gaussian_like",
)

# %%
plot_example_family(
    "Heavy-tailed / skewed DGPs",
    ["student_t", "gamma", "weibull"],
    "fig15b_example_bands_heavy_tailed",
)

# %%
plot_example_family(
    "Non-standard shapes",
    ["beta_opposing", "bimodal_negative"],
    "fig15c_example_bands_nonstandard",
)

# %% [markdown]
# **Reading Fig 15.** The gold Beta-floor region occupies (most of) its
# fixed jurisdiction — at n = 300 that is FPR ≲ 43/n₀ ≈ 0.29, which is why
# it is visible at every AUC: with only 150 negatives, the order-statistic
# law has something to say about a wide swath of the curve. The green
# Wilson region is the part that scales with geometry: nearly absent at
# AUC ≈ 0.70, expanding across the whole TPR plateau by AUC ≈ 0.95 — the
# bootstrap's resampling variance collapses exactly where the empirical
# TPR pins to 1. The unshaded interior is the studentized bootstrap doing
# the adaptive work. Note how the lower bound drops to ~0 at the extreme
# left inside the gold region: that is the floor being honest — below
# FPR ≈ 4–7/n₀ a nonparametric lower bound is provably vacuous.

# %% [markdown]
# ### Fig 15d — Who sets the lower bound, by AUC and sample size
#
# **Motivation.** At AUC = 0.95 and n = 300, the panels above show the
# bootstrap setting almost none of the lower bound — the floors own the
# curve. Is that the steady state, or a small-n regime? The floors'
# jurisdictions both shrink with n: the Beta zone ends at ~43/n₀ and the
# Wilson rectangle's plateau region recedes as the band tightens, so the
# bootstrap should reclaim the interior from the middle outward. (This is
# the evaluation report's band-attribution table — bootstrap share of the
# high-AUC lower band rising from ~25% at n = 1,000 to ~54% at n = 10,000 —
# rendered as a picture, here across the full AUC × n × DGP grid.)
#
# Each cell is one seeded draw: a strip across FPR colored by the mechanism
# that sets the 95% *lower* bound there (the upper bound is bootstrap
# essentially everywhere). Single draws, so cell boundaries wobble — read
# the trend, not the pixel edges.
#
# **Two distribution-free landmarks** explain the structure that repeats
# identically across DGP columns. First, the gold/white geography at small
# n is dominated by Beta-law constants: the vacuity threshold q₁ (the
# 99.9% quantile of Beta(1, n₀)) sits at FPR ≈ 0.37 for n₀ = 15 and shrinks
# like 6.9/n₀, and the jurisdiction edge q₂₅ shrinks like 43/n₀. Second,
# the teal sliver just past q₁ in the n = 30 rows is the *handoff seam*:
# the Beta bound steps discontinuously from the vacuous 0 to the j = 1
# order-statistic bound, momentarily overshooting the smoothly-rising
# Wilson rectangle bound, which takes the minimum for a grid point or two.
# Both landmarks depend only on n₀ and the alpha budget — which is why
# every DGP column shows them at the same FPR.

# %%
MECH_NS = [30, 100, 300, 1000, 3000, 10000]
MECH_DGPS = list(EXAMPLE_SHAPE)
mech_cmap = mpl.colors.ListedColormap(["#FFFFFF", "#F2E97E", "#A8D5C6"])

fig, axes = plt.subplots(
    len(EXAMPLE_AUCS), len(MECH_DGPS),
    figsize=(1.75 * len(MECH_DGPS), 2.7 * len(EXAMPLE_AUCS)),
    sharex=True, sharey=True, squeeze=False,
)
for a, auc in enumerate(EXAMPLE_AUCS):
    for c, dgp_type in enumerate(MECH_DGPS):
        ax = axes[a, c]
        for r, n in enumerate(MECH_NS):
            ex = compute_example_band(
                dgp_type, auc, seed=20260700 + 997 * a + 31 * c + r, n=n
            )
            mech = np.zeros(len(ex["fpr"]))
            mech[ex["wilson"]] = 2
            mech[ex["beta"]] = 1
            x = ex["fpr"]
            x_edges = np.concatenate([[0.0], (x[1:] + x[:-1]) / 2, [1.0]])
            ax.pcolormesh(
                x_edges, [r, r + 1], mech[None, :],
                cmap=mech_cmap, vmin=0, vmax=2,
            )
        for r in range(1, len(MECH_NS)):
            ax.axhline(r, color="0.75", lw=0.8)
        if a == 0:
            ax.set_title(dlabel(dgp_type), fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(len(MECH_NS), 0)
        ax.set_yticks(
            np.arange(len(MECH_NS)) + 0.5, [f"{n:,}" for n in MECH_NS]
        )
        ax.set_xticks([0, 0.5, 1])
        if a == len(EXAMPLE_AUCS) - 1:
            ax.set_xlabel("FPR")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("0.6")
    axes[a, 0].set_ylabel(f"target AUC = {auc:.2f}\nTotal sample size n")
legend = [
    Patch(facecolor="#F2E97E", label="Beta order-statistic floor"),
    Patch(facecolor="#A8D5C6", label="Wilson rectangle floor"),
    Patch(facecolor="white", edgecolor="0.6", label="Bootstrap envelope"),
]
fig.legend(handles=legend, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, -0.03))
fig.suptitle(
    "Mechanism setting the 95% lower bound, by AUC and sample size"
)
fig.tight_layout()
save_figure(fig, "fig15d_mechanism_occupancy_by_n")

# %% [markdown]
# ---
# # 6. Violation profile: when the band misses, how does it miss?
#
# **Motivation.** Coverage rates hide the loss function. Two methods with
# 93% coverage can be very different products if one misses by 0.5pp at one
# grid point and the other misses by 25pp across half the curve. Here we
# characterize the envelope's residual 5% (and its competitors' misses) by
# direction, location, and magnitude.
#
# **Expectations.** Residual envelope violations should be (a) mildly
# lower-bound dominant, (b) located in the *interior* low-to-mid FPR range
# (k ≈ 50–500 grid points — the floors' jurisdictions end around k ≈ 44, and
# pre-floor failures at k = 1–10 are repaired), and (c) small: the >5pp
# violation rate in the high-AUC stratum should be ~0. At the 50% level
# violations should spread across the whole curve — that is the global
# sup-norm calibration story, not a tail defect.

# %% [markdown]
# ### Fig 16 — Direction and location of violations
#
# Error bars on the direction panel are 95% Wilson intervals — load-bearing
# here because the rates are fractions of a percent and the small-n cells
# would otherwise invite over-reading.

# %%
fig = plt.figure(figsize=(12.8, 4.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1.2], wspace=0.3)

# (a) Direction: above vs below violation rates by n
ax = fig.add_subplot(gs[0, 0])
sub = dfb[(dfb.method == "envelope") & (dfb.alpha == 0.05)]
g = sub.groupby("n_total", observed=True)[
    ["violation_below", "violation_above"]
].mean().reindex(PRESENT_NS)
g_n = sub.groupby("n_total", observed=True).size().reindex(PRESENT_NS)
x = np.arange(len(g))
for off, col, color, label in [
    (-0.2, "violation_below", "#D55E00",
     "Truth escapes below (band too optimistic)"),
    (0.2, "violation_above", "#0072B2", "Truth escapes above"),
]:
    lo, hi = wilson_ci(g[col] * g_n, g_n)
    ax.bar(
        x + off, g[col], width=0.38, color=color, label=label,
        yerr=np.vstack([g[col] - lo, hi - g[col]]),
        error_kw=dict(lw=0.9, ecolor="0.3", capsize=2),
    )
ax.set_xticks(x, [f"{n:,}" for n in g.index], rotation=45)
ax.set_xlabel("Total sample size n")
ax.set_ylabel("Violation rate")
ax.set_title("Direction (95% band)")
ax.legend(fontsize=7.5)

# (b) Location: ECDF of first violating FPR (below-violations)
ax = fig.add_subplot(gs[0, 1])
for n in [300, 1000, 3000, 10000]:
    ms = sub[(sub.n_total == n) & sub.violation_below]
    v = ms["violation_fpr_below_min"].dropna().values.astype(float)
    if len(v) < 10:
        continue
    v = np.maximum(v, 1.0 / n)  # grid floor for the log axis
    v = np.sort(v)
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=n_color(n), lw=1.8,
            label=f"n = {n:,}  ({len(v)} violations)")
ax.set_xscale("log")
ax.set_xlabel("FPR of first lower-bound violation")
ax.set_ylabel("Cumulative proportion of violations")
ax.set_title("Location of misses")
ax.legend(fontsize=7.5, loc="upper left")

# (c) Violations by region: 95% vs 50% band — local vs global failure modes.
# Region rates are normalized to sum to 100% per band, so each bar set shows
# *where* violations concentrate (the spatial distribution), not their
# absolute frequency.
ax = fig.add_subplot(gs[0, 2])
width = 0.38
for off, alpha, color in [(-0.2, 0.05, "#D55E00"), (0.2, 0.5, "#999999")]:
    sub_a = dfb[(dfb.method == "envelope") & (dfb.alpha == alpha)
                & (dfb.n_total >= 1000)]
    rates = np.array([sub_a[c].mean() for c in REGION_COLS])
    total = rates.sum()
    shares = 100 * rates / total if total > 0 else rates
    ax.bar(np.arange(len(REGIONS)) + off, shares, width=width, color=color,
           label=f"{(1 - alpha) * 100:.0f}% band")
ax.set_xticks(np.arange(len(REGIONS)), [f"{r}%" for r in REGIONS], rotation=45)
ax.set_xlabel("FPR region")
ax.set_ylabel("Share of region-violations (%, n ≥ 1,000)")
ax.set_title("95% misses are residual and spread;\n50% misses are global calibration")
ax.legend(fontsize=8)

fig.suptitle("Anatomy of the envelope's residual violations")
save_figure(fig, "fig16_violation_direction_location")

# %% [markdown]
# ### Fig 17 — Magnitude: are the misses ever large?
#
# Left: ECDF of the worst violation among violating trials (the conditional
# loss given a miss). Right: the tail-risk comparison across methods — P99
# of max violation over *all* trials, the number that distinguishes "misses
# by a hair" from "misses by a catastrophe". The 5pp reference line marks
# what we consider a practically meaningful miss.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))

ax = axes[0]
sub = dfb[(dfb.method == "envelope") & (dfb.alpha == 0.05)]
for n in [300, 1000, 3000, 10000]:
    ms = sub[(sub.n_total == n) & ~sub.covers_entirely]
    v = np.sort(ms["max_violation"].values.astype(float))
    if len(v) < 10:
        continue
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=n_color(n), lw=1.8,
            label=f"n = {n:,}")
ax.axvline(0.05, color="0.35", ls=":", lw=1.0)
ax.text(0.052, 0.05, "5pp", fontsize=8, color="0.35")
ax.set_xlim(0, 0.12)
ax.set_xlabel("Max violation depth (TPR units), given a miss")
ax.set_ylabel("Cumulative proportion of misses")
ax.set_title("Envelope: conditional miss size")
ax.legend(fontsize=8, loc="lower right")

ax = axes[1]
P99_METHODS = ["envelope", "ks", "wilson_rectangle_sidak", "working_hotelling",
               "pointwise_sidak"]
for m in P99_METHODS:
    p99 = []
    for n in PRESENT_NS:
        ms = dfb[(dfb.method == m) & (dfb.alpha == 0.05) & (dfb.n_total == n)]
        p99.append(
            np.percentile(ms["max_violation"], 99) if len(ms) else np.nan
        )
    ax.plot(PRESENT_NS, p99, lw=1.8, ms=4.5, label=mlabel(m), **mstyle_noshape(m))
ax.axhline(0.05, color="0.35", ls=":", lw=1.0)
ax.set_xscale("log")
ax.set_xlabel("Total sample size n")
ax.set_ylabel("P99 of max violation (all trials)")
ax.set_title("Tail risk across methods")
ax.legend(fontsize=8)

fig.suptitle("When the envelope misses, it misses small")
fig.tight_layout()
save_figure(fig, "fig17_violation_magnitude")

# %% [markdown]
# High-AUC large-miss audit — the historically dangerous stratum
# (true AUC > 0.9): rate of misses deeper than 5pp, by method and n.

# %%
audit = []
for m in ["envelope", "envelope_no_beta_floor", "wilson_rectangle_sidak",
          "working_hotelling", "ks"]:
    sub = dfb[(dfb.method == m) & (dfb.alpha == 0.05) & (dfb.true_auc > HIGH_AUC)]
    g = sub.groupby("n_total", observed=True)["max_violation"].agg(
        big_miss_rate=lambda v: (v > 0.05).mean()
    )
    audit.append(g["big_miss_rate"].rename(mlabel(m)))
audit_df = pd.concat(audit, axis=1).reindex(PRESENT_NS).T
print(f"Rate of >5pp violations, true AUC > {HIGH_AUC}, 95% bands:")
print(audit_df.round(4).to_string())

# %% [markdown]
# ---
# # 7. Additional summaries
#
# ### Fig 18 — Class imbalance is not a risk factor
#
# The run includes n = 1,000 at 10% prevalence (100 positives / 900
# negatives) alongside the balanced configuration. The report's earlier
# side-finding was that imbalance is, if anything, mildly *protective* (the
# Wilson floor scales with 1/n₁). Here we verify that with the final method:
# coverage and width per DGP, balanced vs. imbalanced. Coverage dots carry
# 95% Wilson intervals — the claim is "no difference", so the reader needs
# to see that the intervals overlap rather than trust two bare dots.

# %%
prev_sub = df[(df.method == "envelope") & (df.alpha == 0.05) & (df.n_total == 1000)]
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))

for ax, col, xlabel, fmt in [
    (axes[0], "covers_entirely", "Coverage (95% band)", "cov"),
    (axes[1], "band_area", "Mean band area", "area"),
]:
    g = (
        prev_sub.groupby(["dgp_type", "prevalence"], observed=True)[col]
        .mean()
        .unstack()
        .reindex(index=PRESENT_DGPS)
        .dropna(how="all")
    )
    y = np.arange(len(g))
    counts = (
        prev_sub.groupby(["dgp_type", "prevalence"], observed=True)
        .size()
        .unstack()
        .reindex(index=g.index)
        if fmt == "cov"
        else None
    )
    for prev, marker, color in [(0.5, "o", "#D55E00"), (0.1, "s", "#0072B2")]:
        if prev in g.columns:
            if counts is not None:
                lo, hi = wilson_ci(g[prev] * counts[prev], counts[prev])
                ax.errorbar(
                    g[prev], y, xerr=np.vstack([g[prev] - lo, hi - g[prev]]),
                    fmt="none", ecolor=color, elinewidth=1.0, capsize=2,
                    alpha=0.7, zorder=4,
                )
            ax.scatter(g[prev], y, s=55, marker=marker, color=color, zorder=5,
                       label=f"prevalence {prev:.0%}")
    for yi, (_, row) in zip(y, g.iterrows(), strict=False):
        if row.notna().all():
            ax.plot([row.min(), row.max()], [yi, yi], color="0.6", lw=1.2, zorder=2)
    if fmt == "cov":
        ax.axvline(0.95, color="0.35", ls="--", lw=1.0)
    ax.set_yticks(y, [dlabel(d) for d in g.index])
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()
axes[0].legend(loc="lower left", fontsize=8)
fig.suptitle(
    "n = 1,000: balanced vs. 10% prevalence "
    "— imbalance does not break the band"
)
fig.tight_layout()
save_figure(fig, "fig18_prevalence")

# %% [markdown]
# ### Table 1 — Method summary
#
# One row per method: macro-averaged coverage at each level, calibration
# error, width, width relative to KS, and tail risk. This is the table form
# of the paper's overall argument. Saved as CSV and Markdown.

# %%
SUMMARY_METHODS = [
    "envelope", "ks", "working_hotelling", "wilson_rectangle_sidak",
    "wilson_rectangle_bonferroni", "pointwise_sidak", "pointwise",
    "envelope_no_beta_floor", "envelope_no_wilson_floor", "envelope_no_floors",
    "envelope_beta_both_tails", "envelope_wilson_both_tails",
    "envelope_no_bootstrap",
]

rows = []
ks_area = (
    dfb[(dfb.method == "ks") & (dfb.alpha == 0.05)]
    .groupby(["dgp_type", "n_total"], observed=True)["band_area"]
    .mean()
)
for m in SUMMARY_METHODS:
    sub = dfb[dfb.method == m]
    if sub.empty:
        continue
    row = {"method": mlabel(m)}
    for alpha in ALPHAS:
        mc = macro_coverage(sub[sub.alpha == alpha], by=["method"])
        row[f"cov@{(1 - alpha) * 100:.0f}%"] = (
            mc["coverage"].iloc[0] if len(mc) else np.nan
        )
    row["calib err"] = np.mean(
        [abs(row[f"cov@{(1 - a) * 100:.0f}%"] - (1 - a)) for a in ALPHAS]
    )
    s95 = sub[sub.alpha == 0.05]
    row["area@95%"] = (
        s95.groupby(["dgp_type", "n_total"], observed=True)["band_area"]
        .mean()
        .mean()
    )
    cell_area = s95.groupby(["dgp_type", "n_total"], observed=True)["band_area"].mean()
    ratio = (cell_area / ks_area).dropna()
    row["area ÷ KS"] = ratio.mean() if len(ratio) else np.nan
    row["P99 viol@95%"] = (
        np.percentile(s95["max_violation"], 99) if len(s95) else np.nan
    )
    row[">5pp, AUC>0.9"] = (
        (s95[s95.true_auc > HIGH_AUC]["max_violation"] > 0.05).mean()
    )
    rows.append(row)

summary = pd.DataFrame(rows).set_index("method")
summary_path = FIG_DIR / "table01_method_summary"
summary.round(4).to_csv(f"{summary_path}.csv")
try:
    Path(f"{summary_path}.md").write_text(summary.round(3).to_markdown())
    print(f"Saved {summary_path}.csv / .md\n")
except ImportError:
    print(f"Saved {summary_path}.csv (markdown export needs 'tabulate')\n")
print(summary.round(3).to_string())

# %% [markdown]
# ### Headline numbers for the paper text
#
# A compact printout of the quantities most likely to be quoted in the
# abstract and results sections — regenerated automatically on rerun.

# %%
print("=" * 72)
print("HEADLINE NUMBERS (prevalence 0.5 unless stated)")
print("=" * 72)

env95 = dfb[(dfb.method == "envelope") & (dfb.alpha == 0.05)]
cov_by_n = env95.groupby("n_total", observed=True)["covers_entirely"].mean()
print("\nEnvelope coverage at 95% by n:")
print("  " + "   ".join(f"n={n:,}: {c:.3f}" for n, c in cov_by_n.items()))

hi = env95[env95.true_auc > HIGH_AUC]
print(f"\nEnvelope, high-AUC stratum (true AUC > {HIGH_AUC}):")
print(f"  coverage: {hi['covers_entirely'].mean():.3f}"
      f" | >5pp miss rate: {(hi['max_violation'] > 0.05).mean():.4f}")

wh95 = dfb[(dfb.method == "working_hotelling") & (dfb.alpha == 0.05)]
wh_by_dgp = wh95.groupby("dgp_type", observed=True)["covers_entirely"].mean()
print("\nWorking–Hotelling coverage at 95% by DGP:")
for d, c in wh_by_dgp.items():
    print(f"  {dlabel(d):24s} {c:.3f}")

ks_cells = (
    dfb[(dfb.method == "ks") & (dfb.alpha == 0.05)]
    .groupby(["dgp_type", "n_total"], observed=True)["band_area"].mean()
)
env_cells = env95.groupby(["dgp_type", "n_total"], observed=True)["band_area"].mean()
print(f"\nEnvelope band area ÷ KS band area (mean across cells): "
      f"{(env_cells / ks_cells).dropna().mean():.3f}")

bare = dfb[(dfb.method == "envelope_no_floors") & (dfb.alpha == 0.05)]
print(f"\nBare bootstrap envelope coverage at 95% (pooled): "
      f"{bare['covers_entirely'].mean():.3f}  <- why the floors exist")

env50 = dfb[(dfb.method == "envelope") & (dfb.alpha == 0.5)]
print(f"Envelope coverage at 50% (pooled): {env50['covers_entirely'].mean():.3f}"
      "  <- known sup-norm over-coverage, disclosed in limitations")

# %% [markdown]
# ---
# # Conclusions (updated on rerun)
#
# 1. **Calibration (Figs 1–4).** The envelope holds ~nominal 95% coverage
#    across every DGP family, sample size, AUC level, and shape parameter in
#    the design — the high-AUC, large-n failure mode of the pre-Beta-floor
#    method is gone. Residual imperfection: conservative at the 50% level
#    (sup-norm alpha-insensitivity; disclosed, diminishes with n).
# 2. **Robustness (Figs 5–7).** Working–Hotelling is exactly calibrated on
#    the two DGPs where its model truly holds (binormal, heteroscedastic
#    Gaussian) and collapses continuously — and badly — as tails or modes
#    depart from it, with the damage *growing* with n. The envelope is
#    indistinguishable across all these regimes.
# 3. **Tightness (Figs 8–10).** The envelope pays a modest width premium
#    over WH and undercuts KS substantially, while being better calibrated
#    than both; its width is spent adaptively where the curve is uncertain.
# 4. **Anatomy (Figs 11–14).** Every component earns its place: the bare
#    bootstrap fails at both corners; each floor repairs exactly one corner;
#    a single mechanism mirrored onto both tails is either leaky (Beta-only:
#    ~2x the low-FPR violation rate) or vacuous (Wilson-only: +30–45% width
#    at FPR 0.05); floors without the bootstrap are safe but wide and
#    untunable.
# 5. **Failure profile (Figs 16–17).** Residual misses are rare, small,
#    interior, and mildly lower-bound dominant; the >5pp catastrophic misses
#    of the high-AUC stratum are eliminated.
#
# **Pending before submission:** rerun this notebook once the gamma and
# weibull DGPs finish; confirm Fig 1's 95% panel stays within its color
# scale, and refresh the quoted headline numbers.
