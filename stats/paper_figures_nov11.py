"""Paper figures and tables for the nov11 family of simulation runs.

The runs compare the rank-space fiducial band with its localized ELL floor
(``fiducial_hybrid``, the method under study) against the raw fiducial band
(paired on the same fiducial cloud), the exact equal-local-level (ELL) band
(``m3``), the studentized
bootstrap envelope, and the KS and Working-Hotelling baselines: 6,000 LHS
draws per DGP, seven DGPs, and 50/80/95% bands. The standard run
(``data/results/nov11``) is balanced at n = 10 to 10,000 plus 10% prevalence
at n = 1,000; the imbalanced runs hold one prevalence at every n.

Every figure is one function that takes the trial-level frame from
:func:`load_results`, saves PNG/PDF/SVG to the configured figure directory,
and returns the figure. :func:`configure` selects the run, the output
directory, and the run's primary prevalence (the one the single-prevalence
figures draw). ``notebook_paper_nov11.py`` walks through the standard run
with commentary; running this module directly writes every figure and table.

Usage:
    uv run python stats/paper_figures_nov11.py
    uv run python stats/paper_figures_nov11.py --results-dir data/results/nov11_p10 \
        --fig-dir figures/paper_nov11_p10 --prevalence 0.1
"""

import argparse
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.ipc as ipc
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.typing import FillStyleType
from scipy.stats import t as t_dist

ALPHAS = [0.05, 0.2, 0.5]
N_ORDER = [10, 30, 100, 300, 1000, 3000, 10000]
HERO = "fiducial_hybrid"
HIGH_AUC = 0.9
REGIONS = ["0-10", "10-30", "30-50", "50-70", "70-90", "90-100"]
WIDTH_LANDMARKS = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]

# Columns whose list-valued cells would load as Python objects; the scalar
# LHS coordinates carry the same design information
_LIST_COLUMNS = {"dgp_neg_means", "dgp_neg_stds", "dgp_neg_weights"}


# =============================================================================
# Visual identity
# =============================================================================


@dataclass(frozen=True)
class MethodStyle:
    """Plot identity of one method.

    Attributes:
        label: Legend label.
        color: Line and marker color.
        ls: Line style.
        marker: Marker shape.
    """

    label: str
    color: str
    ls: str = "-"
    marker: str = "o"


# Okabe-Ito palette. The two fiducial arms share a hue family (vermillion /
# orange) because they are one band with and without the floor.
METHODS: dict[str, MethodStyle] = {
    "fiducial_hybrid": MethodStyle("Fiducial + ELL floor (hybrid)", "#D55E00"),
    "fiducial": MethodStyle("Fiducial, no floor", "#E69F00", ls="--", marker="s"),
    "m3": MethodStyle("ELL band (equal local levels)", "#CC79A7", marker="^"),
    "envelope": MethodStyle("Studentized envelope", "#009E73", marker="v"),
    "working_hotelling": MethodStyle(
        "Working–Hotelling (binormal)", "#0072B2", marker="D"
    ),
    "ks": MethodStyle("KS fixed-width", "#000000", marker="X"),
}
METHOD_ORDER = list(METHODS)


@dataclass(frozen=True)
class DgpStyle:
    """Plot identity of one DGP.

    Attributes:
        label: Display name.
        color: Color, grouped by family hue.
        family: Shape family used to group panels.
    """

    label: str
    color: str
    family: str


DGPS: dict[str, DgpStyle] = {
    "binormal": DgpStyle("Binormal", "#0072B2", "Gaussian-like"),
    "hetero_gaussian": DgpStyle("Heterosc. Gaussian", "#56B4E9", "Gaussian-like"),
    "logitnormal": DgpStyle("Logit-normal", "#7BB6DD", "Gaussian-like"),
    "student_t": DgpStyle("Student-t", "#D55E00", "Heavy-tailed / skewed"),
    "weibull": DgpStyle("Weibull", "#CC6677", "Heavy-tailed / skewed"),
    "beta_opposing": DgpStyle("Beta (opposing skew)", "#009E73", "Non-standard shape"),
    "bimodal_negative": DgpStyle("Bimodal negatives", "#AA4499", "Non-standard shape"),
}
DGP_ORDER = list(DGPS)
DGP_FAMILIES: dict[str, list[str]] = {}
for _dgp, _style in DGPS.items():
    DGP_FAMILIES.setdefault(_style.family, []).append(_dgp)

# Grouping by departure from binormality (Fig 8c): logit-normal joins the
# bounded group because WH fits the binormal model to its skewed scores
TRAJ_FAMILIES = {
    "Gaussian": ["binormal", "hetero_gaussian"],
    "Heavy-tailed / skewed": ["student_t", "weibull"],
    "Bounded / multimodal": ["logitnormal", "beta_opposing", "bimodal_negative"],
}

# (dgp, LHS column, x-label, log-x)
SHAPE_SPECS = [
    ("student_t", "lhs_df", "Degrees of freedom (lower = heavier tails)", True),
    ("logitnormal", "lhs_sigma", "Latent σ", False),
    ("hetero_gaussian", "lhs_sigma_ratio", "σ ratio (positive / negative)", True),
    ("beta_opposing", "lhs_alpha", "Beta α (skew strength)", False),
    ("bimodal_negative", "lhs_mode_separation", "Negative mode separation Δ", False),
    ("bimodal_negative", "lhs_mixture_weight", "Negative mixture weight", False),
    ("weibull", "lhs_shape", "Weibull shape (lower = heavier tail)", False),
]

_N_NORM = mpl.colors.LogNorm(vmin=10, vmax=10000)
FLOOR_COLOR = "#F0E442"
FLOOR_ALPHA = 0.45


@dataclass
class FigureConfig:
    """Which run the figures draw and where they are written.

    Attributes:
        results_dir: Run directory holding the trial-level feathers.
        fig_dir: Output directory for figures and tables.
        prevalence: The run's primary prevalence; single-prevalence figures
            draw the trials at this prevalence.
        suffix: Appended to every saved file stem.
        tag: Appended to every figure's title, e.g. "10% prevalence".
        deferred: When a list, :func:`save_figure` collects (figure, stem)
            pairs here instead of writing, so a caller can align axes across
            figures before writing them with :func:`write_figure`.
    """

    results_dir: Path = Path("data/results/nov11")
    fig_dir: Path = Path("figures/paper_nov11")
    prevalence: float = 0.5
    suffix: str = ""
    tag: str = ""
    deferred: list[tuple[plt.Figure, str]] | None = field(default=None)


CONFIG = FigureConfig()


def configure(**changes) -> FigureConfig:
    """Replace fields of the module's figure configuration.

    Args:
        **changes: :class:`FigureConfig` fields to replace.

    Returns:
        The previous configuration, for restoring.
    """
    global CONFIG
    previous = CONFIG
    CONFIG = replace(CONFIG, **changes)
    return previous


def prevalence_label(prevalence: float) -> str:
    """Return a prevalence as a percentage label, e.g. "10%"."""
    return f"{prevalence * 100:.0f}%"


def mlabel(method: str) -> str:
    """Return a method's legend label."""
    return METHODS[method].label


def mcolor(method: str) -> str:
    """Return a method's color."""
    return METHODS[method].color


def mline(method: str, *, marker: bool = True) -> dict:
    """Return matplotlib line keywords for a method.

    Args:
        method: Method name.
        marker: Whether to include the method's marker.

    Returns:
        Keyword arguments for ``Axes.plot``.
    """
    style = METHODS[method]
    kwargs = {"color": style.color, "linestyle": style.ls}
    if marker:
        kwargs["marker"] = style.marker
    return kwargs


def dlabel(dgp: str) -> str:
    """Return a DGP's display name."""
    return DGPS[dgp].label


def n_color(n: int) -> tuple:
    """Return the viridis color of a sample size on a log scale."""
    return mpl.colormaps["viridis"](_N_NORM(n) * 0.92)


def level_label(alpha: float) -> str:
    """Return a band's confidence level as a percentage label."""
    return f"{(1 - alpha) * 100:.0f}%"


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
            "figure.dpi": 110,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> plt.Figure:
    """Tag a figure's title and save it, or defer the save.

    Args:
        fig: Figure to save.
        name: File stem, before the configured suffix.

    Returns:
        The figure.
    """
    # Editing the title artist in place keeps its placement (e.g. a raised y)
    title = getattr(fig, "_suptitle", None)
    if CONFIG.tag and title is not None:
        title.set_text(f"{title.get_text()} — {CONFIG.tag}")
    stem = f"{name}{CONFIG.suffix}"
    if CONFIG.deferred is not None:
        CONFIG.deferred.append((fig, stem))
        return fig
    return write_figure(fig=fig, stem=stem)


def write_figure(*, fig: plt.Figure, stem: str) -> plt.Figure:
    """Write a figure as PNG, PDF, and SVG to the configured directory.

    Args:
        fig: Figure to write.
        stem: File stem.

    Returns:
        The figure.
    """
    CONFIG.fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(CONFIG.fig_dir / f"{stem}.{ext}")
    return fig


def add_nominal_line(ax: plt.Axes, alpha: float) -> None:
    """Draw the nominal coverage reference line."""
    ax.axhline(1 - alpha, color="0.35", ls="--", lw=1.0, zorder=1)


def hide_unused(axes: np.ndarray, used: int) -> None:
    """Hide the panels of a grid past the first ``used``."""
    for ax in axes.flat[used:]:
        ax.set_visible(False)


def grid_shape(n_panels: int, max_cols: int = 4) -> tuple[int, int]:
    """Return (rows, cols) for a panel grid, preferring few blank panels."""
    cols = min(
        (c for c in range(2, max_cols + 1)),
        key=lambda c: (int(np.ceil(n_panels / c)) * c - n_panels, -c),
    )
    cols = min(cols, n_panels)
    return int(np.ceil(n_panels / cols)), cols


# =============================================================================
# Statistics helpers
# =============================================================================


def wilson_ci(
    *, k: np.ndarray | float, n: np.ndarray | float, z: float = 1.96
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Wilson score interval for binomial proportions.

    Args:
        k: Successes.
        n: Trials.
        z: Normal quantile.

    Returns:
        Lower and upper bounds.
    """
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def mean_ci_across(values: Iterable[float]) -> tuple[float, float, float]:
    """Return the mean and 95% t-interval of per-DGP statistics.

    Args:
        values: One statistic per DGP; NaNs are ignored.

    Returns:
        Mean, lower, and upper; the point itself for one value and NaNs for
        none.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if v.size == 0:
        return np.nan, np.nan, np.nan
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0])
    m = float(v.mean())
    half = float(t_dist.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size))
    return m, m - half, m + half


def fit_rate_curve(
    *,
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    n_splines: int = 8,
    min_events: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Smooth P(y = 1 | x) with a logistic GAM and a pointwise 95% band.

    Falls back to binned rates with Wilson intervals when the rarer outcome
    has fewer than ``min_events`` occurrences or the GAM fit fails. Near
    separation (a method that almost never misses) otherwise drives the GAM's
    standard errors to infinity and its band to [0, 1].

    Args:
        x: Predictor.
        y: Binary outcome.
        grid: Evaluation points.
        n_splines: Spline basis size.
        min_events: Fewest occurrences of the rarer outcome for the GAM.

    Returns:
        Rate, lower, and upper on ``grid``, or None with fewer than 100
        observations.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 100:
        return None
    if min(y.sum(), len(y) - y.sum()) >= min_events:
        try:
            from pygam import LogisticGAM, s

            gam = LogisticGAM(s(0, n_splines=n_splines)).fit(x.reshape(-1, 1), y)
            pred = gam.predict_proba(grid.reshape(-1, 1))
            lo, hi = gam.confidence_intervals(grid.reshape(-1, 1), width=0.95).T
            if np.isfinite(pred).all() and np.isfinite(lo).all():
                return pred, np.clip(lo, 0, 1), np.clip(hi, 0, 1)
        except Exception:
            pass
    edges = np.quantile(x, np.linspace(0, 1, 13))
    idx = np.clip(np.searchsorted(edges, x) - 1, 0, 11)
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
    lo_b, hi_b = wilson_ci(k=np.array(ks), n=np.array(ns))
    return (
        np.interp(grid, centers, rates),
        np.interp(grid, centers, lo_b),
        np.interp(grid, centers, hi_b),
    )


def paired_difference(
    *, a: np.ndarray, b: np.ndarray
) -> tuple[float, float, float, float]:
    """Return the paired difference in proportions with a 95% interval.

    Uses the discordant-pair (McNemar) variance, which is exact for the
    paired design and far tighter than comparing two independent rates.

    Args:
        a: Binary outcomes of the first arm.
        b: Binary outcomes of the second arm, aligned with ``a``.

    Returns:
        Difference ``mean(a) - mean(b)``, its lower and upper 95% bounds,
        and the discordance rate.
    """
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n = len(a)
    only_a = float(np.sum(a & ~b))
    only_b = float(np.sum(~a & b))
    diff = (only_a - only_b) / n
    se = np.sqrt(max(only_a + only_b - (only_a - only_b) ** 2 / n, 0.0)) / n
    return diff, diff - 1.96 * se, diff + 1.96 * se, (only_a + only_b) / n


# =============================================================================
# Data
# =============================================================================


def load_results(results_dir: Path | None = None) -> pd.DataFrame:
    """Load every trial-level feather of the run into one frame.

    Strings load as categoricals and floats as float32; ``alpha`` and
    ``prevalence`` are rounded so they compare exactly, and ``max_violation``
    is added. ``dgp_type`` carries every DGP as a category, present or not,
    so frames from different runs concatenate cleanly.

    Args:
        results_dir: Run directory; defaults to the configured one.

    Returns:
        One row per (method, alpha, DGP, sample configuration, LHS draw,
        repeat).

    Raises:
        FileNotFoundError: If the directory holds no trial files.
    """
    results_dir = CONFIG.results_dir if results_dir is None else results_dir
    files = sorted(results_dir.glob("*_individual.feather"))
    if not files:
        raise FileNotFoundError(f"No *_individual.feather files in {results_dir}")
    frames = []
    for path in files:
        names = [c for c in ipc.open_file(path).schema.names if c not in _LIST_COLUMNS]
        frame = pd.read_feather(path, columns=names)
        floats = frame.select_dtypes("float64").columns
        frame[floats] = frame[floats].astype("float32")
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    for col in ("alpha", "prevalence"):
        df[col] = df[col].astype(float).round(4)
    for col in ("n_total", "n_neg", "n_pos"):
        df[col] = df[col].astype(int)
    df["max_violation"] = np.maximum(
        df["max_violation_above"], df["max_violation_below"]
    )
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER)
    df["dgp_type"] = pd.Categorical(df["dgp_type"], categories=DGP_ORDER, ordered=True)
    df["device"] = df["device"].astype("category")
    return df


def load_run_metadata(results_dir: Path | None = None) -> dict:
    """Return the run's most recent metadata record."""
    results_dir = CONFIG.results_dir if results_dir is None else results_dir
    path = sorted(results_dir.glob("run_metadata_*.json"))[-1]
    return json.loads(path.read_text())


def primary(df: pd.DataFrame) -> pd.DataFrame:
    """Return the rows at the run's primary prevalence."""
    return df[df["prevalence"] == CONFIG.prevalence]


def present_dgps(df: pd.DataFrame) -> list[str]:
    """Return the DGPs in the frame, in display order."""
    return [d for d in df["dgp_type"].cat.categories if (df["dgp_type"] == d).any()]


def cell_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trials into one row per (method, alpha, DGP, configuration).

    Args:
        df: Trial-level frame.

    Returns:
        Coverage with Wilson bounds, trial count, mean band area with its
        standard error, and mean width.
    """
    keys = ["method", "alpha", "dgp_type", "n_total", "prevalence"]
    g = df.groupby(keys, observed=True)
    out = g.agg(
        coverage=("covers_entirely", "mean"),
        n_trials=("covers_entirely", "size"),
        area=("band_area", "mean"),
        area_sd=("band_area", "std"),
        width=("mean_band_width", "mean"),
    ).reset_index()
    out["cov_lo"], out["cov_hi"] = wilson_ci(
        k=out["coverage"] * out["n_trials"], n=out["n_trials"]
    )
    out["area_se"] = out["area_sd"] / np.sqrt(out["n_trials"])
    return out


def paired_fiducial(df: pd.DataFrame) -> pd.DataFrame:
    """Align the two fiducial arms replicate by replicate.

    Both arms are trimmed from one fiducial cloud per replicate, so the
    pairing is exact.

    Args:
        df: Trial-level frame.

    Returns:
        One row per replicate and alpha with each arm's coverage and the
        replicate's floor region and true AUC.
    """
    keys = ["dgp_type", "n_total", "prevalence", "lhs_idx", "sim_idx", "alpha"]
    cols = [*keys, "covers_entirely", "true_auc"]
    raw = df.loc[df["method"] == "fiducial", cols]
    hyb = df.loc[df["method"] == HERO, [*keys, "covers_entirely"]]
    out = raw.merge(hyb, on=keys, suffixes=("_fiducial", "_hybrid"), validate="1:1")
    return out


# =============================================================================
# 1. Calibration of the hybrid
# =============================================================================

DEV_LIMS = {0.05: 0.05, 0.2: 0.15, 0.5: 0.45}


def _coverage_heatmap(
    *, ax: plt.Axes, cells: pd.DataFrame, alpha: float, lim: float, ylabels: bool
) -> float:
    """Draw one annotated coverage-minus-nominal heatmap and return max SE."""
    sub = cells[cells["alpha"] == alpha]
    cov = sub.pivot(index="dgp_type", columns="col", values="coverage")
    cov = cov.reindex(columns=_config_order(sub))
    n = sub.pivot(index="dgp_type", columns="col", values="n_trials").reindex_like(cov)
    sns.heatmap(
        cov - (1 - alpha),
        annot=cov,
        fmt=".3f",
        cmap="RdBu",
        center=0,
        vmin=-lim,
        vmax=lim,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": f"Coverage − {1 - alpha:.2f}", "shrink": 0.85},
        annot_kws={"fontsize": 7.5},
        ax=ax,
    )
    ax.set_xlabel("Total sample size n (prevalence)")
    ax.set_ylabel("")
    ax.set_yticklabels([dlabel(d) for d in cov.index] if ylabels else [], rotation=0)
    ax.grid(False)
    return float(np.nanmax(np.sqrt(cov * (1 - cov) / n).to_numpy()))


def _config_label(*, n: int, prevalence: float) -> str:
    """Label a configuration, marking prevalences other than the primary."""
    if prevalence == CONFIG.prevalence:
        return f"{n:,}"
    return f"{n:,} ({prevalence_label(prevalence)})"


def _config_order(frame: pd.DataFrame) -> list[str]:
    """Return a frame's configuration labels by n, secondary prevalence first."""
    configs = sorted(
        set(zip(frame["n_total"], frame["prevalence"], strict=True)),
        key=lambda c: (c[0], c[1] == CONFIG.prevalence, c[1]),
    )
    return [_config_label(n=n, prevalence=p) for n, p in configs]


def _with_config_column(cells: pd.DataFrame) -> pd.DataFrame:
    """Label each configuration, marking secondary-prevalence cells."""
    cells = cells.copy()
    cells["col"] = [
        _config_label(n=n, prevalence=p)
        for n, p in zip(cells["n_total"], cells["prevalence"], strict=True)
    ]
    return cells


def fig01_coverage_heatmaps(
    *, cells: pd.DataFrame, alphas: list[float], name: str, method: str = HERO
) -> plt.Figure:
    """Coverage of one method by DGP and configuration, one panel per level.

    Args:
        cells: Output of :func:`cell_summary`.
        alphas: Levels to draw, one panel each.
        name: Output file stem.
        method: Method to draw.

    Returns:
        The figure.
    """
    sub = _with_config_column(cells[cells["method"] == method])
    n_dgps = sub["dgp_type"].nunique()
    fig, axes = plt.subplots(
        1,
        len(alphas),
        figsize=(6.6 * len(alphas), 0.62 * n_dgps + 1.8),
        constrained_layout=True,
        squeeze=False,
    )
    notes = []
    for i, (ax, alpha) in enumerate(zip(axes[0], alphas, strict=True)):
        se = _coverage_heatmap(
            ax=ax, cells=sub, alpha=alpha, lim=DEV_LIMS[alpha], ylabels=i == 0
        )
        ax.set_title(f"{level_label(alpha)} band (nominal {1 - alpha:.2f})")
        notes.append(f"{level_label(alpha)} ≤ {se:.3f}")
    fig.suptitle(f"{mlabel(method)}: coverage across DGPs and sample sizes")
    fig.text(
        0.01,
        -0.03,
        "Each cell pools 6,000 simulations; Monte Carlo SE per cell: "
        + ", ".join(notes)
        + ".",
        fontsize=8,
        color="0.4",
    )
    return save_figure(fig, name)


def fig02_coverage_vs_auc(
    df: pd.DataFrame, *, method: str = HERO, alpha: float = 0.05
) -> plt.Figure:
    """Coverage as a smooth function of true AUC, per DGP and sample size.

    Args:
        df: Trial-level frame.
        method: Method to draw.
        alpha: Band level.

    Returns:
        The figure.
    """
    ns = [100, 1000, 10000]
    sub = primary(df)
    sub = sub[(sub["method"] == method) & (sub["alpha"] == alpha)]
    dgps = present_dgps(sub)
    rows, cols = grid_shape(len(dgps))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.4 * cols, 2.9 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    grid = np.linspace(0.56, 0.985, 160)
    lowest = 1.0
    for ax, dgp in zip(axes.flat, dgps, strict=False):
        for n in ns:
            ms = sub[(sub["dgp_type"] == dgp) & (sub["n_total"] == n)]
            fit = fit_rate_curve(
                x=ms["true_auc"].to_numpy(),
                y=ms["covers_entirely"].to_numpy(),
                grid=grid,
            )
            if fit is None:
                continue
            pred, lo, hi = fit
            lowest = min(lowest, float(pred.min()))
            ax.plot(grid, pred, color=n_color(n), lw=1.8, label=f"n = {n:,}")
            ax.fill_between(grid, lo, hi, color=n_color(n), alpha=0.14, lw=0)
        add_nominal_line(ax, alpha)
        ax.set_title(dlabel(dgp), fontsize=10)
        ax.set_xlim(0.55, 1.0)
    hide_unused(axes, len(dgps))
    default_lo = 0.88 if alpha == 0.05 else max(0.0, 1 - alpha - 0.25)
    lo_lim = min(default_lo, np.floor((lowest - 0.02) * 20) / 20)
    axes.flat[0].set_ylim(lo_lim, 1.005)
    for ax in axes[-1, :]:
        ax.set_xlabel("True AUC")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Coverage ({level_label(alpha)} band)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(ns), bbox_to_anchor=(0.5, -0.04)
    )
    fig.suptitle(f"{mlabel(method)}: coverage vs. true AUC")
    fig.tight_layout()
    return save_figure(fig, f"fig02_coverage_vs_auc_{method}_{level_label(alpha)[:-1]}")


def fig03_coverage_vs_shape(
    df: pd.DataFrame, *, method: str = HERO, alpha: float = 0.05
) -> plt.Figure:
    """Coverage as a smooth function of each DGP's shape parameter.

    Args:
        df: Trial-level frame.
        method: Method to draw.
        alpha: Band level.

    Returns:
        The figure.
    """
    ns = [100, 1000, 10000]
    stem = f"fig03_coverage_vs_shape_{method}_{level_label(alpha)[:-1]}"
    sub = primary(df)
    sub = sub[(sub["method"] == method) & (sub["alpha"] == alpha)]
    specs = [s for s in SHAPE_SPECS if (sub["dgp_type"] == s[0]).any()]
    if not specs:
        fig, ax = plt.subplots(figsize=(6.0, 1.6))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No DGP with a shape parameter in this run", ha="center")
        fig.suptitle(f"{mlabel(method)}: coverage vs. DGP shape parameters")
        return save_figure(fig, stem)
    rows, cols = grid_shape(len(specs))
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.4 * cols, 2.9 * rows), sharey=True, squeeze=False
    )
    lowest = 1.0
    for ax, (dgp, col, xlabel, logx) in zip(axes.flat, specs, strict=False):
        for n in ns:
            ms = sub[(sub["dgp_type"] == dgp) & (sub["n_total"] == n)]
            if ms.empty:
                continue
            x = ms[col].to_numpy(float)
            gx = np.log10(x) if logx else x
            grid = np.linspace(gx.min(), gx.max(), 120)
            fit = fit_rate_curve(x=gx, y=ms["covers_entirely"].to_numpy(), grid=grid)
            if fit is None:
                continue
            pred, lo, hi = fit
            lowest = min(lowest, float(pred.min()))
            xs = 10**grid if logx else grid
            ax.plot(xs, pred, color=n_color(n), lw=1.8, label=f"n = {n:,}")
            ax.fill_between(xs, lo, hi, color=n_color(n), alpha=0.14, lw=0)
        if logx:
            ax.set_xscale("log")
        add_nominal_line(ax, alpha)
        ax.set_title(dlabel(dgp), fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8.5)
    hide_unused(axes, len(specs))
    default_lo = 0.85 if alpha == 0.05 else 0.0
    axes.flat[0].set_ylim(min(default_lo, np.floor((lowest - 0.02) * 20) / 20), 1.01)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Coverage ({level_label(alpha)} band)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(ns), bbox_to_anchor=(0.5, -0.03)
    )
    fig.suptitle(f"{mlabel(method)}: coverage vs. DGP shape parameters")
    fig.tight_layout()
    return save_figure(fig, stem)


def fig04_level_calibration(cells: pd.DataFrame) -> plt.Figure:
    """Empirical vs. nominal coverage, macro-averaged over DGPs, by n.

    Args:
        cells: Output of :func:`cell_summary`.

    Returns:
        The figure.
    """
    sub = primary(cells)
    ns = [int(sub["n_total"].min()), 100, 1000, 10000]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.0), sharex=True, sharey=True)
    for ax, n in zip(axes.flat, ns, strict=True):
        at_n = sub[sub["n_total"] == n]
        macro = at_n.groupby(["method", "alpha"], observed=True)["coverage"].mean()
        for m in METHOD_ORDER:
            if m not in macro.index.get_level_values(0):
                continue
            series = macro.loc[m].sort_index(ascending=False)
            ax.plot(
                1 - series.index,
                series.to_numpy(),
                lw=1.8,
                ms=5.5,
                label=mlabel(m),
                **mline(m),
            )
        ax.plot([0.4, 1.0], [0.4, 1.0], color="0.35", ls="--", lw=1.0, zorder=1)
        ax.set_title(f"n = {n:,}")
        ax.set_xticks([0.5, 0.8, 0.95])
        ax.set_ylim(0.0, 1.05)
    for ax in axes[-1, :]:
        ax.set_xlabel("Nominal coverage")
    for ax in axes[:, 0]:
        ax.set_ylabel("Empirical coverage (mean over DGPs)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Calibration across confidence levels (identity = perfect)")
    fig.tight_layout()
    return save_figure(fig, "fig04_level_calibration")


# =============================================================================
# 2. Robustness
# =============================================================================


def fig05_hybrid_vs_wh_heatmap(cells: pd.DataFrame) -> plt.Figure:
    """95% coverage heatmaps of the hybrid and Working-Hotelling side by side.

    Args:
        cells: Output of :func:`cell_summary`.

    Returns:
        The figure.
    """
    n_dgps = cells["dgp_type"].nunique()
    fig, axes = plt.subplots(
        1, 2, figsize=(6.6 * 2, 0.62 * n_dgps + 1.8), constrained_layout=True
    )
    for i, (ax, m) in enumerate(zip(axes, [HERO, "working_hotelling"], strict=True)):
        sub = _with_config_column(cells[cells["method"] == m])
        _coverage_heatmap(ax=ax, cells=sub, alpha=0.05, lim=0.5, ylabels=i == 0)
        ax.set_title(mlabel(m))
    fig.suptitle("95% bands: distribution-free calibration vs. parametric fragility")
    return save_figure(fig, "fig05_hybrid_vs_wh_heatmap")


def fig06_coverage_vs_n(cells: pd.DataFrame, *, alpha: float = 0.05) -> plt.Figure:
    """Coverage vs. sample size per DGP, every method, with Wilson bands.

    Args:
        cells: Output of :func:`cell_summary`.
        alpha: Band level.

    Returns:
        The figure.
    """
    sub = primary(cells)
    sub = sub[sub["alpha"] == alpha]
    dgps = [d for d in DGP_ORDER if (sub["dgp_type"] == d).any()]
    rows, cols = grid_shape(len(dgps))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.4 * cols, 2.9 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for ax, dgp in zip(axes.flat, dgps, strict=False):
        for m in METHOD_ORDER:
            g = sub[(sub["method"] == m) & (sub["dgp_type"] == dgp)].sort_values(
                "n_total"
            )
            if g.empty:
                continue
            ax.plot(
                g["n_total"], g["coverage"], lw=1.6, ms=3.5, label=mlabel(m), **mline(m)
            )
            ax.fill_between(
                g["n_total"],
                g["cov_lo"],
                g["cov_hi"],
                color=mcolor(m),
                alpha=0.15,
                lw=0,
            )
        add_nominal_line(ax, alpha)
        ax.set_xscale("log")
        ax.set_title(dlabel(dgp), fontsize=10)
        ax.set_ylim(0, 1.04)
    hide_unused(axes, len(dgps))
    for ax in axes[-1, :]:
        ax.set_xlabel("Total sample size n")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Coverage ({level_label(alpha)} band)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Coverage vs. sample size, {level_label(alpha)} bands")
    fig.tight_layout()
    return save_figure(fig, f"fig06_coverage_vs_n_{level_label(alpha)[:-1]}")


def fig07_wh_fragility(df: pd.DataFrame) -> plt.Figure:
    """Coverage as a function of the departure from binormality.

    Args:
        df: Trial-level frame.

    Returns:
        The figure.
    """
    specs = [
        (
            "student_t",
            "lhs_df",
            "Degrees of freedom (heavier tails to the right)",
            True,
            True,
        ),
        ("bimodal_negative", "lhs_mode_separation", "Mode separation Δ", False, False),
    ]
    ns = [300, 3000]
    methods = [HERO, "m3", "envelope", "working_hotelling"]
    sub = primary(df)
    sub = sub[(sub["alpha"] == 0.05) & sub["method"].isin(methods)]
    fig, axes = plt.subplots(
        len(specs),
        len(ns),
        figsize=(4.4 * len(ns), 3.2 * len(specs)),
        sharey=True,
        squeeze=False,
    )
    for r, (dgp, col, xlabel, logx, invert) in enumerate(specs):
        for c, n in enumerate(ns):
            ax = axes[r, c]
            for m in methods:
                ms = sub[
                    (sub["method"] == m)
                    & (sub["dgp_type"] == dgp)
                    & (sub["n_total"] == n)
                ]
                if ms.empty:
                    continue
                x = ms[col].to_numpy(float)
                gx = np.log10(x) if logx else x
                grid = np.linspace(gx.min(), gx.max(), 120)
                fit = fit_rate_curve(
                    x=gx, y=ms["covers_entirely"].to_numpy(), grid=grid
                )
                if fit is None:
                    continue
                pred, lo, hi = fit
                xs = 10**grid if logx else grid
                ax.plot(xs, pred, lw=2.0, label=mlabel(m), **mline(m, marker=False))
                ax.fill_between(xs, lo, hi, color=mcolor(m), alpha=0.14, lw=0)
            if logx:
                ax.set_xscale("log")
            if invert:
                ax.invert_xaxis()
            add_nominal_line(ax, 0.05)
            if r == 0:
                ax.set_title(f"n = {n:,}")
            ax.set_xlabel(xlabel)
            if c == 0:
                ax.set_ylabel(f"{dlabel(dgp)}\nCoverage (95% band)")
            ax.set_ylim(0, 1.04)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "Working–Hotelling degrades continuously with departure from binormality"
    )
    fig.tight_layout()
    return save_figure(fig, "fig07_wh_fragility")


# =============================================================================
# 3. Tightness
# =============================================================================


def fig08_pareto(cells: pd.DataFrame) -> plt.Figure:
    """Coverage vs. mean band area, macro-averaged over DGPs, by n.

    Args:
        cells: Output of :func:`cell_summary`.

    Returns:
        The figure.
    """
    ns = [100, 300, 1000, 10000]
    sub = primary(cells)
    sub = sub[sub["alpha"] == 0.05]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.0), sharey=True)
    for ax, n in zip(axes.flat, ns, strict=True):
        at_n = sub[sub["n_total"] == n]
        macro = at_n.groupby("method", observed=True)[["coverage", "area"]].mean()
        for m in METHOD_ORDER:
            if m not in macro.index:
                continue
            hero = m == HERO
            ax.scatter(
                macro.loc[m, "area"],
                macro.loc[m, "coverage"],
                s=240 if hero else 90,
                marker=METHODS[m].marker,
                color=mcolor(m),
                edgecolors="black",
                linewidths=1.0 if hero else 0.4,
                zorder=10 if hero else 5,
                label=mlabel(m),
            )
        add_nominal_line(ax, 0.05)
        ax.set_title(f"n = {n:,}")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(left=0)
    for ax in axes[-1, :]:
        ax.set_xlabel("Mean band area (smaller = tighter)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Coverage (95% band, mean over DGPs)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Tightness vs. coverage at the 95% level")
    fig.tight_layout()
    return save_figure(fig, "fig08_pareto_area_coverage")


def fig08b_pareto_ci(*, cells: pd.DataFrame, family: str) -> plt.Figure:
    """Per-DGP coverage vs. area with 95% CIs for one DGP family.

    Args:
        cells: Output of :func:`cell_summary`.
        family: Key of ``DGP_FAMILIES``.

    Returns:
        The figure.
    """
    ns = [100, 1000, 10000]
    sub = primary(cells)
    sub = sub[sub["alpha"] == 0.05]
    dgps = [d for d in DGP_FAMILIES[family] if (sub["dgp_type"] == d).any()]
    fig, axes = plt.subplots(
        len(ns),
        len(dgps),
        figsize=(3.6 * len(dgps), 3.0 * len(ns)),
        sharex="row",
        squeeze=False,
    )
    for r, n in enumerate(ns):
        for c, dgp in enumerate(dgps):
            ax = axes[r, c]
            cell = sub[(sub["n_total"] == n) & (sub["dgp_type"] == dgp)].set_index(
                "method"
            )
            for m in METHOD_ORDER:
                if m not in cell.index:
                    continue
                row = cell.loc[m]
                hero = m == HERO
                ax.errorbar(
                    row["area"],
                    row["coverage"],
                    xerr=1.96 * row["area_se"],
                    yerr=[
                        [max(0.0, row["coverage"] - row["cov_lo"])],
                        [max(0.0, row["cov_hi"] - row["coverage"])],
                    ],
                    fmt="none",
                    ecolor=mcolor(m),
                    elinewidth=1.0,
                    capsize=2,
                    alpha=0.7,
                    zorder=4,
                )
                ax.scatter(
                    row["area"],
                    row["coverage"],
                    s=150 if hero else 60,
                    marker=METHODS[m].marker,
                    color=mcolor(m),
                    edgecolors="black",
                    linewidths=0.9 if hero else 0.4,
                    zorder=10 if hero else 5,
                    label=mlabel(m),
                )
            add_nominal_line(ax, 0.05)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(left=0)
            if r == 0:
                ax.set_title(dlabel(dgp), fontsize=10)
            if r == len(ns) - 1:
                ax.set_xlabel("Mean band area")
            if c == 0:
                ax.set_ylabel(f"n = {n:,}\nCoverage (95% band)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(f"Per-DGP coverage–tightness with 95% CIs — {family}")
    fig.tight_layout()
    slug = family.split(" ")[0].lower().replace("-", "_")
    return save_figure(fig, f"fig08b_pareto_ci_{slug}")


def fig08c_trajectory(df: pd.DataFrame) -> plt.Figure:
    """Coverage vs. area as n grows, hybrid vs. WH, by family and AUC tertile.

    Args:
        df: Trial-level frame.

    Returns:
        The figure.
    """
    methods = [(HERO, "-"), ("working_hotelling", "--")]
    sub = primary(df)
    sub = sub[(sub["alpha"] == 0.05) & sub["method"].isin([m for m, _ in methods])]
    edges = sub["true_auc"].quantile([0, 1 / 3, 2 / 3, 1.0]).to_numpy(float)
    labels = [
        f"Low AUC\n[{edges[0]:.2f}, {edges[1]:.2f})",
        f"Mid AUC\n[{edges[1]:.2f}, {edges[2]:.2f})",
        f"High AUC\n[{edges[2]:.2f}, {edges[3]:.2f}]",
    ]
    edges[-1] += 1e-6
    sub = sub.assign(tertile=pd.cut(sub["true_auc"], edges, labels=False, right=False))
    per_dgp = (
        sub.groupby(["tertile", "method", "dgp_type", "n_total"], observed=True)
        .agg(cov=("covers_entirely", "mean"), area=("band_area", "mean"))
        .reset_index()
    )
    fig, axes = plt.subplots(
        3,
        len(TRAJ_FAMILIES),
        figsize=(4.0 * len(TRAJ_FAMILIES), 9.6),
        sharex="col",
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    for r in range(3):
        for c, (family, dgps) in enumerate(TRAJ_FAMILIES.items()):
            ax = axes[r, c]
            band = per_dgp[(per_dgp["tertile"] == r) & per_dgp["dgp_type"].isin(dgps)]
            for m, ls in methods:
                traj = (
                    band[band["method"] == m]
                    .groupby("n_total")[["cov", "area"]]
                    .mean()
                    .sort_index()
                )
                if traj.empty:
                    continue
                ax.plot(
                    traj["area"], traj["cov"], ls=ls, color="0.55", lw=1.3, zorder=3
                )
                ax.scatter(
                    traj["area"],
                    traj["cov"],
                    c=[n_color(n) for n in traj.index],
                    s=55,
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=5,
                )
            add_nominal_line(ax, 0.05)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(left=0)
            if r == 0:
                ax.set_title(family, fontsize=10)
            if r == 2:
                ax.set_xlabel("Mean band area")
            if c == 0:
                ax.set_ylabel(f"{labels[r]}\nCoverage (95% band)")
    fig.legend(
        handles=[
            Line2D([], [], color="0.4", ls=ls, lw=1.5, label=mlabel(m))
            for m, ls in methods
        ],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.04),
    )
    sm = mpl.cm.ScalarMappable(norm=_N_NORM, cmap="viridis")
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01)
    cbar.set_label("Sample size n")
    cbar.set_ticks(N_ORDER)
    cbar.set_ticklabels([f"{n:,}" for n in N_ORDER])
    fig.suptitle("Hybrid vs. Working–Hotelling: band-tightening trajectory over n")
    return save_figure(fig, "fig08c_trajectory_hybrid_vs_wh")


def fig09_width_scale_and_calibration(cells: pd.DataFrame) -> plt.Figure:
    """Band area on the WH-to-KS scale and calibration error, by n.

    Args:
        cells: Output of :func:`cell_summary`.

    Returns:
        The figure.
    """
    sub = primary(cells)
    sub = sub[sub["alpha"] == 0.05]
    area = sub.pivot_table(
        index=["dgp_type", "n_total"], columns="method", values="area", observed=True
    )
    scaled = area.sub(area["working_hotelling"], axis=0).div(
        area["ks"] - area["working_hotelling"], axis=0
    )
    ns = sorted(sub["n_total"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    ax = axes[0]
    for m in [HERO, "fiducial", "envelope", "m3"]:
        means, lo_err, hi_err = [], [], []
        for n in ns:
            mean, lo, hi = mean_ci_across(scaled.xs(n, level="n_total")[m].to_numpy())
            means.append(mean)
            lo_err.append(mean - lo)
            hi_err.append(hi - mean)
        ax.errorbar(
            ns,
            means,
            yerr=[lo_err, hi_err],
            capsize=2.5,
            lw=1.8,
            ms=5,
            label=mlabel(m),
            **mline(m),
        )
    ax.axhline(0.0, color=mcolor("working_hotelling"), ls=":", lw=1.2)
    ax.axhline(1.0, color=mcolor("ks"), ls=":", lw=1.2)
    ax.text(
        ns[0],
        0.0,
        " WH (tightest)",
        va="bottom",
        fontsize=8,
        color=mcolor("working_hotelling"),
    )
    ax.text(ns[0], 1.0, " KS (widest)", va="bottom", fontsize=8, color=mcolor("ks"))
    ax.set_xscale("log")
    ax.set_xlabel("Total sample size n")
    ax.set_ylabel("Band area, fraction of WH→KS range")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("Width on the WH→KS scale")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    err = (
        sub.assign(err=(sub["coverage"] - 0.95).abs())
        .groupby(["method", "n_total"], observed=True)["err"]
        .mean()
    )
    for m in METHOD_ORDER:
        series = err.loc[m]
        ax.plot(
            series.index, series.to_numpy(), lw=1.8, ms=5, label=mlabel(m), **mline(m)
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total sample size n")
    ax.set_ylabel("Mean |coverage − 0.95| across DGPs")
    ax.set_title("Calibration error (lower is better)")
    ax.legend(fontsize=7.5)
    fig.suptitle("Band width and calibration against the KS and WH references")
    fig.tight_layout()
    return save_figure(fig, "fig09_tightness_vs_ks")


def fig10_width_profile(df: pd.DataFrame) -> plt.Figure:
    """Median band width at fixed FPR landmarks, mean over DGPs with 95% CIs.

    Args:
        df: Trial-level frame.

    Returns:
        The figure.
    """
    ns = [100, 1000, 10000]
    cols = [f"width_at_fpr_{x}" for x in WIDTH_LANDMARKS]
    sub = primary(df)
    sub = sub[(sub["alpha"] == 0.05) & sub["n_total"].isin(ns)]
    med = sub.groupby(["n_total", "method", "dgp_type"], observed=True)[cols].median()
    fig, axes = plt.subplots(1, len(ns), figsize=(4.2 * len(ns), 4.2), sharey=True)
    for ax, n in zip(axes, ns, strict=True):
        for m in METHOD_ORDER:
            if (n, m) not in med.index:
                continue
            per = med.loc[(n, m)]
            stats = [mean_ci_across(per[c].to_numpy()) for c in cols]
            means = np.array([s[0] for s in stats])
            ax.errorbar(
                WIDTH_LANDMARKS,
                means,
                yerr=[
                    means - np.array([s[1] for s in stats]),
                    np.array([s[2] for s in stats]) - means,
                ],
                capsize=2,
                lw=1.6,
                ms=4,
                label=mlabel(m),
                **mline(m),
            )
        ax.set_xscale("log")
        ax.set_xlabel("FPR")
        ax.set_title(f"n = {n:,}")
    axes[0].set_ylabel("Median band width (TPR units)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Band width across the FPR axis (95% bands)")
    fig.tight_layout()
    return save_figure(fig, "fig10_width_profile")


# =============================================================================
# 4. Ablations: the hybrid without its floor (raw fiducial) and the floor
#    alone (the ELL band)
# =============================================================================

ABLATIONS = [HERO, "fiducial", "m3"]


def _macro_by_n(
    *, cells: pd.DataFrame, value: str, alpha: float, agg: str = "mean"
) -> pd.DataFrame:
    """Aggregate a per-cell statistic over DGPs for each ablation arm and n.

    Args:
        cells: Output of :func:`cell_summary`, already filtered to the
            configurations of interest.
        value: Column of ``cells`` to aggregate.
        alpha: Band level.
        agg: Aggregation over DGPs (and prevalences), e.g. "mean" or "min".

    Returns:
        Sample sizes by ablation arm.
    """
    sub = cells[(cells["alpha"] == alpha) & cells["method"].isin(ABLATIONS)]
    out = sub.groupby(["n_total", "method"], observed=True)[value].agg(agg).unstack()
    return out[ABLATIONS]


def fig11_no_floor_ablation(*, df: pd.DataFrame, cells: pd.DataFrame) -> plt.Figure:
    """Coverage and the FPR-region map of misses with and without the floor.

    Args:
        df: Trial-level frame.
        cells: Output of :func:`cell_summary`.

    Returns:
        The figure.
    """
    regions = [f"violation_{r}" for r in REGIONS]
    sub = primary(df)
    sub = sub[sub["alpha"] == 0.05]
    fig = plt.figure(figsize=(13.6, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1.15], wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    mean = _macro_by_n(cells=primary(cells), value="coverage", alpha=0.05)
    worst = _macro_by_n(cells=cells, value="coverage", alpha=0.05, agg="min")
    for m in ABLATIONS:
        ax.plot(mean.index, mean[m], lw=1.9, ms=4.5, label=mlabel(m), **mline(m))
        ax.plot(
            worst.index,
            worst[m],
            lw=1.0,
            ms=4.5,
            color=mcolor(m),
            ls=":",
            marker=METHODS[m].marker,
            markerfacecolor="white",
        )
    add_nominal_line(ax, 0.05)
    ax.set_xscale("log")
    ax.set_ylim(0.88, 1.005)
    ax.set_xlabel("Total sample size n")
    ax.set_ylabel("Coverage (95% band)")
    ax.set_title("Mean over DGPs (solid) and worst cell (dotted)", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower left")

    for k, m in enumerate(["fiducial", HERO]):
        ax = fig.add_subplot(gs[0, k + 1])
        rates = sub[sub["method"] == m].groupby("n_total")[regions].mean().T
        rates.index = [f"{r}%" for r in REGIONS]
        sns.heatmap(
            rates,
            annot=True,
            fmt=".3f",
            cmap="Reds",
            vmin=0,
            vmax=0.012,
            linewidths=0.6,
            linecolor="white",
            cbar=k == 1,
            cbar_kws={"label": "Violation rate", "shrink": 0.85},
            annot_kws={"fontsize": 6.5},
            ax=ax,
        )
        ax.set_title(mlabel(m), fontsize=10)
        ax.set_xlabel("n")
        ax.set_ylabel("FPR region" if k == 0 else "")
        ax.grid(False)
    fig.suptitle(
        "Without the floor: calibrated on average, leaky at the corners and "
        "in the worst cell"
    )
    return save_figure(fig, "fig11_no_floor_ablation")


def fig12_corner_ablation(df: pd.DataFrame) -> plt.Figure:
    """Coverage and corner violation rates for the three ablation arms.

    Args:
        df: Trial-level frame.

    Returns:
        The figure.
    """
    sub = primary(df)
    sub = sub[(sub["alpha"] == 0.05) & sub["method"].isin(ABLATIONS)]
    specs = [
        ("Coverage — all AUC", None, "covers_entirely"),
        (f"Coverage — true AUC > {HIGH_AUC}", HIGH_AUC, "covers_entirely"),
        ("Violation rate, FPR 0–10% (steep corner)", None, "violation_0-10"),
        ("Violation rate, FPR 90–100% (plateau corner)", None, "violation_90-100"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.6))
    for ax, (title, auc_min, col) in zip(axes.flat, specs, strict=True):
        part = sub if auc_min is None else sub[sub["true_auc"] > auc_min]
        rates = part.groupby(["method", "n_total"], observed=True)[col].mean()
        for m in ABLATIONS:
            series = rates.loc[m]
            ax.plot(
                series.index,
                series.to_numpy(),
                lw=1.8,
                ms=4.5,
                label=mlabel(m),
                **mline(m),
            )
        if col == "covers_entirely":
            add_nominal_line(ax, 0.05)
            ax.set_ylim(0.94, 1.003)
            ax.set_ylabel("Coverage (95% band)")
        else:
            ax.set_ylim(bottom=0)
            ax.set_ylabel("Violation rate")
        ax.set_xscale("log")
        ax.set_xlabel("Total sample size n")
        ax.set_title(title, fontsize=10)
    top = max(ax.get_ylim()[1] for ax in axes[1])
    axes[1, 1].sharey(axes[1, 0])
    axes[1, 0].set_ylim(0, top)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Corner ablation: the floor removes the plateau misses at small n")
    fig.tight_layout()
    return save_figure(fig, "fig12_corner_ablation")


def fig13_floor_only_ablation(cells: pd.DataFrame) -> plt.Figure:
    """Coverage at 95% and 50% and band area for the floor alone (the ELL band).

    Args:
        cells: Output of :func:`cell_summary`.

    Returns:
        The figure.
    """
    bal = primary(cells)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    for ax, alpha in zip(axes[:2], [0.05, 0.5], strict=True):
        cov = _macro_by_n(cells=bal, value="coverage", alpha=alpha)
        for m in ABLATIONS:
            ax.plot(cov.index, cov[m], lw=1.8, ms=4.5, label=mlabel(m), **mline(m))
        add_nominal_line(ax, alpha)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.04)
        ax.set_xlabel("Total sample size n")
        ax.set_ylabel("Coverage (mean over DGPs)")
        ax.set_title(f"{level_label(alpha)} band")
    axes[0].legend(fontsize=7.5, loc="lower left")
    ax = axes[2]
    for alpha, ls in [(0.05, "-"), (0.5, "--")]:
        area = _macro_by_n(cells=bal, value="area", alpha=alpha)
        for m in ABLATIONS:
            ax.plot(
                area.index,
                area[m],
                lw=1.6,
                ms=4,
                color=mcolor(m),
                ls=ls,
                marker=METHODS[m].marker,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total sample size n")
    ax.set_ylabel("Mean band area")
    ax.set_title("Width: 95% (solid) and 50% (dashed)")
    fig.suptitle(
        "The floor alone (the ELL band) is valid but wide and nearly level-blind"
    )
    fig.text(
        0.01,
        -0.03,
        "ELL = equal local levels: each class's band gives every order statistic "
        "the same exact Beta tail probability, and the two compose into an ROC band.",
        fontsize=8,
        color="0.4",
    )
    fig.tight_layout()
    return save_figure(fig, "fig13_floor_only_ablation")


def fig14_single_component_summary(cells: pd.DataFrame) -> plt.Figure:
    """Worst-cell coverage, area relative to the ELL band, and calibration error by n.

    Args:
        cells: Output of :func:`cell_summary`.

    Returns:
        The figure.
    """
    bal = primary(cells)
    worst = _macro_by_n(cells=cells, value="coverage", alpha=0.05, agg="min")
    area = bal[(bal["alpha"] == 0.05) & bal["method"].isin(ABLATIONS)].pivot_table(
        index=["dgp_type", "n_total"], columns="method", values="area", observed=True
    )
    ratio = area.div(area["m3"], axis=0).groupby("n_total").mean()[ABLATIONS]
    err = (
        bal[bal["method"].isin(ABLATIONS)]
        .assign(err=lambda d: (d["coverage"] - (1 - d["alpha"])).abs())
        .groupby(["n_total", "method"], observed=True)["err"]
        .mean()
        .unstack()[ABLATIONS]
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    panels = [
        (
            worst,
            "Worst-cell coverage (95% band)",
            "Leakiness: worst DGP × configuration",
        ),
        (
            ratio,
            "Band area ÷ ELL-band area (95% band)",
            "Width relative to the floor alone",
        ),
        (err, "Mean |coverage − nominal|, 3 levels", "Calibration across levels"),
    ]
    for ax, (data, ylabel, title) in zip(axes, panels, strict=True):
        for m in ABLATIONS:
            ax.plot(data.index, data[m], lw=1.8, ms=4.5, label=mlabel(m), **mline(m))
        ax.set_xscale("log")
        ax.set_xlabel("Total sample size n")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
    add_nominal_line(axes[0], 0.05)
    axes[0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle(
        "Neither component alone: the raw band is leaky in its worst cell, "
        "the ELL band is wide and level-blind"
    )
    fig.tight_layout()
    return save_figure(fig, "fig14_single_component_summary")


# =============================================================================
# 5. What the floor buys (paired on one fiducial cloud)
# =============================================================================


def fig15_paired_floor_effect(paired: pd.DataFrame) -> plt.Figure:
    """Paired coverage difference, hybrid minus raw fiducial, by DGP and n.

    Args:
        paired: Output of :func:`paired_fiducial`.

    Returns:
        The figure.
    """
    configs = sorted(
        set(zip(paired["n_total"], paired["prevalence"], strict=True)),
        key=lambda c: (c[0], c[1] == CONFIG.prevalence, c[1]),
    )
    xlabels = [_config_label(n=n, prevalence=p) for n, p in configs]
    rows = []
    for key, g in paired.groupby(
        ["dgp_type", "n_total", "prevalence", "alpha"], observed=True
    ):
        dgp, n, prev, alpha = cast(tuple, key)
        diff, lo, hi, disc = paired_difference(
            a=g["covers_entirely_hybrid"].to_numpy(),
            b=g["covers_entirely_fiducial"].to_numpy(),
        )
        rows.append(
            dict(
                dgp_type=dgp,
                n_total=n,
                prevalence=prev,
                alpha=alpha,
                diff=diff,
                lo=lo,
                hi=hi,
                discordance=disc,
            )
        )
    res = pd.DataFrame(rows)
    res["x"] = [
        configs.index((n, p))
        for n, p in zip(res["n_total"], res["prevalence"], strict=True)
    ]
    dgps = [d for d in DGP_ORDER if (res["dgp_type"] == d).any()]
    offsets = np.linspace(-0.3, 0.3, len(dgps))
    fig, axes = plt.subplots(
        2,
        len(ALPHAS),
        figsize=(4.6 * len(ALPHAS), 7.2),
        sharex=True,
        sharey="row",
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    for c, alpha in enumerate(ALPHAS):
        ax, ax_d = axes[0, c], axes[1, c]
        for off, dgp in zip(offsets, dgps, strict=True):
            g = res[(res["alpha"] == alpha) & (res["dgp_type"] == dgp)].sort_values("x")
            ax.errorbar(
                g["x"] + off,
                g["diff"],
                yerr=[g["diff"] - g["lo"], g["hi"] - g["diff"]],
                fmt="o",
                ms=3.5,
                lw=1.0,
                capsize=1.5,
                color=DGPS[dgp].color,
                label=dlabel(dgp),
            )
            ax_d.plot(
                g["x"] + off, g["discordance"], "o", ms=3.5, color=DGPS[dgp].color
            )
        ax.axhline(0, color="0.35", lw=1.0)
        ax.set_title(f"{level_label(alpha)} band")
        ax_d.set_xticks(range(len(configs)), xlabels, rotation=45, ha="right")
        ax_d.set_xlabel("Total sample size n (prevalence)")
        ax_d.set_yscale("log")
    axes[0, 0].set_ylabel("Coverage: hybrid − raw fiducial\n(paired, 95% CI)")
    axes[1, 0].set_ylabel("Discordant replicates")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("What the ELL floor changes, replicate by replicate on one cloud")
    fig.tight_layout()
    return save_figure(fig, "fig15_paired_floor_effect")


def fig16_floor_region_geometry(df: pd.DataFrame) -> plt.Figure:
    """Extent of the floor region: left cutoff, plateau suffix, interior share.

    Args:
        df: Trial-level frame.

    Returns:
        The figure.
    """
    sub = primary(df)
    sub = sub[sub["method"] == HERO]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))

    ax = axes[0]
    for alpha, color in zip(ALPHAS, ["#D55E00", "#E69F00", "#999999"], strict=True):
        g = sub[sub["alpha"] == alpha].groupby("n_neg")["floor_region_left_end_fpr"]
        q = g.quantile([0.1, 0.5, 0.9]).unstack()
        ax.plot(
            q.index,
            q[0.5],
            "o-",
            color=color,
            lw=1.6,
            ms=4,
            label=f"{level_label(alpha)} band",
        )
        ax.fill_between(q.index, q[0.1], q[0.9], color=color, alpha=0.18, lw=0)
    ns0 = np.array(sorted(sub["n_neg"].unique()))
    ax.plot(ns0, 1 / ns0, color="0.5", ls=":", lw=1.0, label="one grid step, $1/n_0$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Negatives $n_0$")
    ax.set_ylabel("FPR where the left floor ends")
    ax.set_title("Left (low-FPR) floor: median, 10–90%")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    at = sub[sub["alpha"] == 0.05]
    bins = np.linspace(0.55, 0.99, 12)
    centers = (bins[1:] + bins[:-1]) / 2
    for n in [100, 1000, 10000]:
        ms = at[at["n_total"] == n]
        idx = np.digitize(ms["true_auc"], bins) - 1
        suffix = 1 - ms["floor_region_right_start_fpr"].to_numpy(float)
        med = [
            np.median(suffix[idx == b]) if np.any(idx == b) else np.nan
            for b in range(len(centers))
        ]
        ax.plot(centers, med, "o-", color=n_color(n), lw=1.6, ms=4, label=f"n = {n:,}")
    ax.set_yscale("log")
    ax.set_xlabel("True AUC")
    ax.set_ylabel("FPR length of the right floor (1 − start)")
    ax.set_title("Right (plateau) floor, 95% band")
    ax.legend(fontsize=7.5)

    ax = axes[2]
    for alpha, color in zip(ALPHAS, ["#D55E00", "#E69F00", "#999999"], strict=True):
        g = sub[sub["alpha"] == alpha].groupby("n_total")[
            "floor_region_interior_fraction"
        ]
        mean = g.mean()
        ax.plot(
            mean.index,
            mean.to_numpy(),
            "o-",
            color=color,
            lw=1.6,
            ms=4,
            label=f"{level_label(alpha)} band",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Total sample size n")
    ax.set_ylabel("Share of interior grid points floored")
    ax.set_title("How much of the curve the floor owns")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5)
    fig.suptitle("Geometry of the localized ELL floor region")
    fig.tight_layout()
    return save_figure(fig, "fig16_floor_region_geometry")


def fig17_misses_by_floor_region(df: pd.DataFrame) -> plt.Figure:
    """Miss rates inside vs. outside the floor region, by direction and n.

    Args:
        df: Trial-level frame.

    Returns:
        The figure.
    """
    cols = {
        ("in", "above"): "miss_above_in_floor_region",
        ("in", "below"): "miss_below_in_floor_region",
        ("out", "above"): "miss_above_outside_floor_region",
        ("out", "below"): "miss_below_outside_floor_region",
    }
    methods = ["fiducial", HERO, "envelope"]
    sub = primary(df)
    sub = sub[sub["method"].isin(methods) & sub["alpha"].isin([0.05, 0.5])]
    per_dgp = sub.groupby(["alpha", "method", "n_total", "dgp_type"], observed=True)[
        list(cols.values())
    ].mean()
    macro = per_dgp.groupby(["alpha", "method", "n_total"], observed=True).mean()
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6), sharex=True, sharey="row")
    for r, alpha in enumerate([0.05, 0.5]):
        for c, where in enumerate(["in", "out"]):
            ax = axes[r, c]
            for m in methods:
                g = macro.loc[(alpha, m)]
                for direction, marker in [("above", "^"), ("below", "v")]:
                    ax.plot(
                        g.index,
                        g[cols[(where, direction)]],
                        lw=1.5,
                        ms=5,
                        color=mcolor(m),
                        linestyle=METHODS[m].ls,
                        marker=marker,
                    )
            ax.set_xscale("log")
            region = "inside" if where == "in" else "outside"
            ax.set_title(f"{level_label(alpha)} band, misses {region} the floor region")
            if c == 0:
                ax.set_ylabel("Miss rate (mean over DGPs)")
            if r == 1:
                ax.set_xlabel("Total sample size n")
    handles = [
        Line2D([], [], lw=1.5, label=mlabel(m), **mline(m, marker=False))
        for m in methods
    ] + [
        Line2D([], [], color="0.3", marker="^", ls="none", label="truth above band"),
        Line2D([], [], color="0.3", marker="v", ls="none", label="truth below band"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Where the misses happen relative to the floor's jurisdiction")
    fig.tight_layout()
    return save_figure(fig, "fig17_misses_by_floor_region")


def fig18_floor_tradeoff_by_region(df: pd.DataFrame) -> plt.Figure:
    """Hybrid minus raw fiducial, in width and in violations, by FPR region.

    Args:
        df: Trial-level frame.

    Returns:
        The figure.
    """
    ns = [30, 100, 300, 1000, 10000]
    wcols = [f"width_region_{r}" for r in REGIONS]
    vcols = [f"violation_{r}" for r in REGIONS]
    sub = primary(df)
    sub = sub[sub["method"].isin(["fiducial", HERO]) & sub["n_total"].isin(ns)]
    per_dgp = sub.groupby(["alpha", "n_total", "method", "dgp_type"], observed=True)[
        wcols + vcols
    ].mean()
    macro = per_dgp.groupby(["alpha", "n_total", "method"], observed=True).mean()
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4), sharex=True, sharey="col")
    x = np.arange(len(REGIONS))
    for r, alpha in enumerate([0.05, 0.5]):
        for c, (cols, ylabel) in enumerate(
            [(wcols, "Δ mean width (TPR units)"), (vcols, "Δ violation rate")]
        ):
            ax = axes[r, c]
            for n in ns:
                if (alpha, n, HERO) not in macro.index:
                    continue
                delta = (
                    macro.loc[(alpha, n, HERO), cols].to_numpy()
                    - macro.loc[(alpha, n, "fiducial"), cols].to_numpy()
                )
                ax.plot(
                    x, delta, "o-", color=n_color(n), lw=1.6, ms=4, label=f"n = {n:,}"
                )
            ax.axhline(0, color="0.35", lw=1.0)
            ax.set_title(f"{level_label(alpha)} band: {ylabel.split(' ', 1)[1]}")
            if c == 0:
                ax.set_ylabel(f"Hybrid − raw fiducial\n{ylabel}")
            if r == 1:
                ax.set_xticks(x, [f"{reg}%" for reg in REGIONS], rotation=45)
                ax.set_xlabel("FPR region")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(ns), bbox_to_anchor=(0.5, -0.04)
    )
    fig.suptitle(
        "The floor widens the ends of the curve; the recalibrated trim "
        "tightens the middle"
    )
    fig.tight_layout()
    return save_figure(fig, "fig18_floor_tradeoff_by_region")


# =============================================================================
# 6. Example bands
# =============================================================================

EXAMPLE_SHAPE: dict[str, dict[str, float]] = {
    "binormal": {},
    "hetero_gaussian": {"sigma_ratio": 2.0},
    "logitnormal": {"sigma": 1.5},
    "student_t": {"df": 3.0},
    "weibull": {"shape": 1.5},
    "beta_opposing": {"alpha": 2.0},
    "bimodal_negative": {"mixture_weight": 0.35, "mode_separation": 2.5},
}
EXAMPLE_AUCS = [0.70, 0.85, 0.95]


def make_example_dgp(
    *, dgp_type: str, auc: float, shape: dict[str, float] | None = None
):
    """Build a DGP at a target AUC with representative shape parameters.

    Args:
        dgp_type: DGP name.
        auc: Target AUC.
        shape: Shape parameters; defaults to the DGP's ``EXAMPLE_SHAPE``.

    Returns:
        The DGP instance.
    """
    from studroc_paper.datagen.roc_to_dgp import map_lhs_to_dgp
    from studroc_paper.datagen.true_rocs import (
        make_beta_opposing_skew_dgp,
        make_bimodal_negative_dgp,
        make_heteroskedastic_gaussian_dgp,
        make_logitnormal_dgp,
        make_student_t_dgp,
        make_weibull_dgp,
    )

    factory = {
        "binormal": make_heteroskedastic_gaussian_dgp,
        "hetero_gaussian": make_heteroskedastic_gaussian_dgp,
        "logitnormal": make_logitnormal_dgp,
        "beta_opposing": make_beta_opposing_skew_dgp,
        "student_t": make_student_t_dgp,
        "bimodal_negative": make_bimodal_negative_dgp,
        "weibull": make_weibull_dgp,
    }
    lhs = {"auc": np.array([auc])} | {
        k: np.array([v])
        for k, v in (EXAMPLE_SHAPE[dgp_type] if shape is None else shape).items()
    }
    params = {}
    for key, value in map_lhs_to_dgp(dgp_type, lhs).items():
        if isinstance(value, np.ndarray) and value.ndim > 0:
            params[key] = float(value[0])
        elif isinstance(value, list):
            params[key] = value[0]
        else:
            params[key] = value
    return factory[dgp_type](**params)


@dataclass(frozen=True)
class ExampleBand:
    """One seeded dataset with its fiducial, hybrid, and ELL bands.

    Attributes:
        fpr: Native FPR grid.
        empirical: Empirical TPR at the top of each jump.
        true: True TPR.
        bands: Method name to (lower, upper).
        floor: Grid mask on which the hybrid took the ELL hull.
    """

    fpr: np.ndarray
    empirical: np.ndarray
    true: np.ndarray
    bands: dict[str, tuple[np.ndarray, np.ndarray]]
    floor: np.ndarray


def compute_example_band(
    *, dgp_type: str, auc: float, seed: int, n: int = 300, alpha: float = 0.05
) -> ExampleBand:
    """Draw one dataset at the primary prevalence and compute its bands.

    Args:
        dgp_type: DGP name.
        auc: Target AUC.
        seed: Seed for the data and the fiducial cloud.
        n: Total sample size.
        alpha: Band level.

    Returns:
        The dataset's bands and the hybrid's floor region.
    """
    from studroc_paper.methods import M3Floor, fiducial_band_suite_rs, m3_band_rs

    dgp = make_example_dgp(dgp_type=dgp_type, auc=auc)
    rng = np.random.default_rng(seed)
    n_pos = round(n * CONFIG.prevalence)
    n_neg = n - n_pos
    pos, neg = dgp.sample(n_pos, n_neg, rng)
    y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    y_score = np.concatenate([pos, neg]).astype(np.float64)
    suite = fiducial_band_suite_rs(
        y_true=y_true,
        y_score=y_score,
        alphas=[alpha],
        m3_floors={"fiducial": False, HERO: M3Floor()},
        random_state=seed,
    )
    _, m3_lower, m3_upper = m3_band_rs(y_true, y_score, alpha=alpha, random_state=seed)
    neg_desc = np.sort(neg)[::-1]
    pos_sorted = np.sort(pos)
    above = n_pos - np.searchsorted(pos_sorted, neg_desc, side="right")
    empirical = np.concatenate([above, [n_pos]]) / n_pos
    return ExampleBand(
        fpr=suite.grid,
        empirical=empirical,
        true=dgp.get_true_roc(suite.grid),
        bands={**suite.bands[alpha], "m3": (m3_lower, m3_upper)},
        floor=suite.floor_regions[alpha][HERO],
    )


def _spans(mask: np.ndarray, x: np.ndarray) -> list[tuple[float, float]]:
    """Return the x-extents of runs of True, padded half a grid step."""
    step = x[1] - x[0]
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return [
        (max(x[a] - step / 2, 0.0), min(x[b - 1] + step / 2, 1.0))
        for a, b in zip(edges[::2], edges[1::2], strict=True)
    ]


def fig19_example_bands(*, family: str, n: int = 300) -> plt.Figure:
    """Seeded example bands for one DGP family with the floor region shaded.

    Args:
        family: Key of ``DGP_FAMILIES``.
        n: Total sample size.

    Returns:
        The figure.
    """
    dgps = DGP_FAMILIES[family]
    fig, axes = plt.subplots(
        len(EXAMPLE_AUCS),
        len(dgps),
        figsize=(3.4 * len(dgps), 3.2 * len(EXAMPLE_AUCS)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for c, dgp in enumerate(dgps):
        for r, auc in enumerate(EXAMPLE_AUCS):
            ax = axes[r, c]
            ex = compute_example_band(
                dgp_type=dgp, auc=auc, seed=20261111 + 13 * c + r, n=n
            )
            for x0, x1 in _spans(ex.floor, ex.fpr):
                ax.axvspan(x0, x1, color=FLOOR_COLOR, alpha=FLOOR_ALPHA, lw=0, zorder=0)
            lo, hi = ex.bands[HERO]
            ax.fill_between(
                ex.fpr,
                lo,
                hi,
                step="post",
                color=mcolor(HERO),
                alpha=0.2,
                lw=0,
                zorder=2,
            )
            for edge in (lo, hi):
                ax.step(
                    ex.fpr, edge, where="post", color=mcolor(HERO), lw=1.1, zorder=3
                )
            for edge in ex.bands["fiducial"]:
                ax.step(
                    ex.fpr,
                    edge,
                    where="post",
                    color=mcolor("fiducial"),
                    lw=1.0,
                    ls="--",
                    zorder=3,
                )
            for edge in ex.bands["m3"]:
                ax.step(
                    ex.fpr,
                    edge,
                    where="post",
                    color=mcolor("m3"),
                    lw=0.8,
                    ls=":",
                    zorder=3,
                )
            ax.step(ex.fpr, ex.empirical, where="post", color="0.25", lw=1.2, zorder=4)
            ax.plot(ex.fpr, ex.true, color="black", ls="--", lw=1.4, zorder=5)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.set_aspect("equal")
            ax.grid(False)
            if r == 0:
                shape = ", ".join(f"{k}={v:g}" for k, v in EXAMPLE_SHAPE[dgp].items())
                ax.set_title(
                    dlabel(dgp) + (f"\n({shape})" if shape else ""), fontsize=9.5
                )
            if c == 0:
                ax.set_ylabel(f"target AUC = {auc:.2f}\nTPR")
            if r == len(EXAMPLE_AUCS) - 1:
                ax.set_xlabel("FPR")
    legend = [
        Line2D([], [], color="black", ls="--", lw=1.4, label="True ROC"),
        Line2D([], [], color="0.25", lw=1.2, label="Empirical ROC"),
        Patch(facecolor=mcolor(HERO), alpha=0.3, label=f"95% {mlabel(HERO)}"),
        Line2D([], [], color=mcolor("fiducial"), ls="--", label=mlabel("fiducial")),
        Line2D([], [], color=mcolor("m3"), ls=":", label=mlabel("m3")),
        Patch(
            facecolor=FLOOR_COLOR,
            alpha=FLOOR_ALPHA,
            label="Floor region (ELL hull taken)",
        ),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Example 95% bands at n = {n} — {family}", y=1.005)
    fig.text(
        0.01,
        -0.09,
        "ELL band: the exact equal-local-level band. Shaded: where the hybrid "
        "takes its hull instead of the fiducial band.",
        fontsize=8,
        color="0.4",
    )
    fig.tight_layout()
    slug = family.split(" ")[0].lower().replace("-", "_")
    return save_figure(fig, f"fig19_example_bands_{slug}")


def fig19d_floor_occupancy(df: pd.DataFrame, *, alpha: float = 0.05) -> plt.Figure:
    """Probability that each FPR lies in the floor region, by AUC and n.

    Computed from every replicate of the run rather than single draws.

    Args:
        df: Trial-level frame.
        alpha: Band level.

    Returns:
        The figure.
    """
    sub = primary(df)
    sub = sub[(sub["method"] == HERO) & (sub["alpha"] == alpha)]
    ns = [30, 100, 300, 1000, 3000, 10000]
    aucs = [
        (0.55, 0.75, "AUC 0.55–0.75"),
        (0.75, 0.9, "AUC 0.75–0.90"),
        (0.9, 1.0, "AUC 0.90–0.99"),
    ]
    x = np.linspace(0, 1, 401)
    x_edges = np.concatenate([[0.0], (x[1:] + x[:-1]) / 2, [1.0]])
    fig, axes = plt.subplots(1, len(aucs), figsize=(4.4 * len(aucs), 3.4), sharey=True)
    for ax, (lo, hi, label) in zip(axes, aucs, strict=True):
        occ = np.zeros((len(ns), len(x)))
        for i, n in enumerate(ns):
            ms = sub[
                (sub["n_total"] == n) & (sub["true_auc"] >= lo) & (sub["true_auc"] < hi)
            ]
            left = ms["floor_region_left_end_fpr"].to_numpy(float)
            right = ms["floor_region_right_start_fpr"].to_numpy(float)
            inside = (x[None, :] <= left[:, None]) | (x[None, :] >= right[:, None])
            occ[i] = inside.mean(axis=0)
        mesh = ax.pcolormesh(
            x_edges, np.arange(len(ns) + 1), occ, cmap="YlOrBr", vmin=0, vmax=1
        )
        ax.set_yticks(np.arange(len(ns)) + 0.5, [f"{n:,}" for n in ns])
        ax.set_ylim(len(ns), 0)
        ax.set_title(label)
        ax.set_xlabel("FPR")
        ax.grid(False)
    axes[0].set_ylabel("Total sample size n")
    cbar = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.01)
    cbar.set_label("P(FPR in floor region)")
    fig.suptitle(
        f"Where the ELL floor operates ({level_label(alpha)} band, all replicates)",
        y=1.06,
    )
    return save_figure(fig, "fig19d_floor_occupancy")


# =============================================================================
# 7. Violation profile
# =============================================================================


def fig20_violation_direction_location(df: pd.DataFrame) -> plt.Figure:
    """Miss direction by n, location of first misses, and misses by region.

    Args:
        df: Trial-level frame.

    Returns:
        The figure.
    """
    methods = [HERO, "fiducial", "envelope"]
    sub = primary(df)
    s95 = sub[(sub["alpha"] == 0.05) & sub["method"].isin(methods)]
    fig = plt.figure(figsize=(14.0, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.1, 1.1], wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    rates = s95.groupby(["method", "n_total"], observed=True).agg(
        above=("violation_above", "mean"),
        below=("violation_below", "mean"),
        n=("violation_above", "size"),
    )
    for m in methods:
        g = rates.loc[m]
        for col, marker, ls in [("above", "^", "-"), ("below", "v", ":")]:
            lo, hi = wilson_ci(k=g[col] * g["n"], n=g["n"])
            ax.errorbar(
                g.index,
                g[col],
                yerr=[np.clip(g[col] - lo, 0, None), np.clip(hi - g[col], 0, None)],
                marker=marker,
                ls=ls,
                color=mcolor(m),
                ms=4.5,
                lw=1.4,
                capsize=1.5,
            )
    ax.set_xscale("log")
    ax.set_xlabel("Total sample size n")
    ax.set_ylabel("Violation rate (95% band)")
    ax.set_title("Direction: truth above (▲) vs. below (▼)")
    ax.legend(
        handles=[
            Line2D([], [], color=mcolor(m), lw=1.5, label=mlabel(m)) for m in methods
        ],
        fontsize=7.5,
    )

    ax = fig.add_subplot(gs[0, 1])
    hero = s95[(s95["method"] == HERO) & s95["n_total"].isin([300, 1000, 3000, 10000])]
    for n, g in hero.groupby("n_total"):
        for col, ls in [
            ("violation_fpr_below_min", ":"),
            ("violation_fpr_above_min", "-"),
        ]:
            v = np.sort(g[col].dropna().to_numpy(float))
            if len(v) < 10:
                continue
            v = np.maximum(v, 1.0 / n)
            ax.plot(
                v,
                np.arange(1, len(v) + 1) / len(v),
                color=n_color(n),
                ls=ls,
                lw=1.6,
                label=f"n = {n:,}" if ls == "-" else None,
            )
    ax.set_xscale("log")
    ax.set_xlabel("FPR of the first violation")
    ax.set_ylabel("Cumulative share of misses")
    ax.set_title("Hybrid: location (solid above, dotted below)")
    ax.legend(fontsize=7.5)

    ax = fig.add_subplot(gs[0, 2])
    cols = [f"violation_{r}" for r in REGIONS]
    for off, alpha, color in [(-0.2, 0.05, mcolor(HERO)), (0.2, 0.5, "#999999")]:
        sa = sub[
            (sub["method"] == HERO) & (sub["alpha"] == alpha) & (sub["n_total"] >= 1000)
        ]
        rates_r = sa[cols].mean().to_numpy()
        ax.bar(
            np.arange(len(REGIONS)) + off,
            100 * rates_r / rates_r.sum(),
            width=0.38,
            color=color,
            label=f"{level_label(alpha)} band",
        )
    ax.set_xticks(np.arange(len(REGIONS)), [f"{r}%" for r in REGIONS], rotation=45)
    ax.set_xlabel("FPR region")
    ax.set_ylabel("Share of region violations (%, n ≥ 1,000)")
    ax.set_title("Hybrid: where misses concentrate")
    ax.legend(fontsize=8)
    fig.suptitle("Anatomy of residual violations")
    return save_figure(fig, "fig20_violation_direction_location")


def fig21_violation_magnitude(df: pd.DataFrame) -> plt.Figure:
    """Conditional miss depth for the hybrid and P99 miss depth for all methods.

    Args:
        df: Trial-level frame.

    Returns:
        The figure.
    """
    sub = primary(df)
    s95 = sub[sub["alpha"] == 0.05]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    ax = axes[0]
    hero = s95[(s95["method"] == HERO) & ~s95["covers_entirely"]]
    for n in [300, 1000, 3000, 10000]:
        v = np.sort(hero.loc[hero["n_total"] == n, "max_violation"].to_numpy(float))
        if len(v) < 10:
            continue
        ax.plot(
            v,
            np.arange(1, len(v) + 1) / len(v),
            color=n_color(n),
            lw=1.8,
            label=f"n = {n:,}",
        )
    ax.axvline(0.05, color="0.35", ls=":", lw=1.0)
    ax.text(0.052, 0.05, "5pp", fontsize=8, color="0.35")
    ax.set_xlim(0, 0.12)
    ax.set_xlabel("Max violation depth (TPR units), given a miss")
    ax.set_ylabel("Cumulative share of misses")
    ax.set_title(f"{mlabel(HERO)}: conditional miss size", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    p99 = s95.groupby(["method", "n_total"], observed=True)["max_violation"].quantile(
        0.99
    )
    for m in METHOD_ORDER:
        series = p99.loc[m]
        ax.plot(
            series.index, series.to_numpy(), lw=1.8, ms=4.5, label=mlabel(m), **mline(m)
        )
    ax.axhline(0.05, color="0.35", ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Total sample size n")
    ax.set_ylabel("P99 of max violation (all trials)")
    ax.set_title("Tail risk across methods")
    ax.legend(fontsize=7.5)
    fig.suptitle("Miss magnitude at the 95% level")
    fig.tight_layout()
    return save_figure(fig, "fig21_violation_magnitude")


def high_auc_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Rate of misses deeper than 5pp at true AUC > 0.9, 95% bands.

    Args:
        df: Trial-level frame.

    Returns:
        Methods by sample size.
    """
    sub = primary(df)
    sub = sub[(sub["alpha"] == 0.05) & (sub["true_auc"] > HIGH_AUC)]
    rate = sub.groupby(["method", "n_total"], observed=True)["max_violation"].agg(
        lambda v: (v > 0.05).mean()
    )
    out = rate.unstack()
    out.index = [mlabel(m) for m in out.index]
    return out


# =============================================================================
# 8. Prevalence, cost, guarantees, summary
# =============================================================================


def fig22_prevalence(df: pd.DataFrame) -> plt.Figure:
    """Coverage and area at n = 1,000, balanced vs. 10% prevalence.

    Args:
        df: Trial-level frame.

    Returns:
        The figure.
    """
    methods = [HERO, "fiducial", "envelope", "m3"]
    sub = df[
        (df["n_total"] == 1000) & (df["alpha"] == 0.05) & df["method"].isin(methods)
    ]
    g = sub.groupby(["method", "dgp_type", "prevalence"], observed=True).agg(
        cov=("covers_entirely", "mean"),
        n=("covers_entirely", "size"),
        area=("band_area", "mean"),
    )
    dgps = [d for d in DGP_ORDER if d in g.index.get_level_values("dgp_type")]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    offsets = np.linspace(-0.27, 0.27, len(methods))
    for off, m in zip(offsets, methods, strict=True):
        for prev, filled in [(0.5, True), (0.1, False)]:
            rows = g.loc[m].xs(prev, level="prevalence").reindex(dgps)
            y = np.arange(len(dgps)) + off
            lo, hi = wilson_ci(k=rows["cov"] * rows["n"], n=rows["n"])
            face = mcolor(m) if filled else "white"
            axes[0].errorbar(
                rows["cov"],
                y,
                xerr=[rows["cov"] - lo, hi - rows["cov"]],
                fmt="none",
                ecolor=mcolor(m),
                elinewidth=1.0,
                capsize=1.5,
            )
            axes[0].scatter(
                rows["cov"],
                y,
                s=36,
                marker=METHODS[m].marker,
                facecolor=face,
                edgecolor=mcolor(m),
                zorder=5,
            )
            axes[1].scatter(
                rows["area"],
                y,
                s=36,
                marker=METHODS[m].marker,
                facecolor=face,
                edgecolor=mcolor(m),
                zorder=5,
            )
    axes[0].axvline(0.95, color="0.35", ls="--", lw=1.0)
    axes[0].set_xlabel("Coverage (95% band)")
    axes[1].set_xlabel("Mean band area")
    axes[0].set_yticks(np.arange(len(dgps)), [dlabel(d) for d in dgps])
    axes[0].invert_yaxis()
    handles = [
        Line2D(
            [],
            [],
            marker=METHODS[m].marker,
            ls="none",
            color=mcolor(m),
            label=mlabel(m),
        )
        for m in methods
    ] + [
        Line2D([], [], marker="o", ls="none", color="0.3", label="50% prevalence"),
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            markerfacecolor="white",
            color="0.3",
            label="10% prevalence",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("n = 1,000: balanced vs. 10% prevalence")
    fig.tight_layout()
    return save_figure(fig, "fig22_prevalence")


def replicate_cost(df: pd.DataFrame) -> pd.DataFrame:
    """Wall time to produce all three levels of each method for one replicate.

    The envelope's cost is its bootstrap generation plus its per-level
    envelopes; the two fiducial arms share one call, reported once.

    Args:
        df: Trial-level frame.

    Returns:
        One row per replicate and method family, with the device used.
    """
    keys = ["dgp_type", "n_total", "prevalence", "lhs_idx", "sim_idx"]
    sub = df[df["method"] != "fiducial"]
    per = (
        sub.groupby([*keys, "method"], observed=True)
        .agg(
            level_sum=("runtime_seconds", "sum"),
            shared=("runtime_seconds", "first"),
            is_shared=("runtime_is_shared", "first"),
            boot=("bootstrap_gen_seconds", "first"),
            device=("device", "first"),
        )
        .reset_index()
    )
    per["seconds"] = np.where(per["is_shared"], per["shared"], per["level_sum"])
    is_env = per["method"] == "envelope"
    per.loc[is_env, "seconds"] += per.loc[is_env, "boot"]
    per["method"] = (
        per["method"].astype(str).replace({HERO: "fiducial suite (both arms)"})
    )
    return per[[*keys, "method", "seconds", "device"]]


def fig23_runtime(df: pd.DataFrame) -> plt.Figure:
    """Median wall time per replicate (all three levels) by method and n.

    Args:
        df: Trial-level frame.

    Returns:
        The figure.
    """
    cost = replicate_cost(primary(df))
    med = cost.groupby(["method", "n_total", "device"], observed=True)[
        "seconds"
    ].median()
    style = {
        "fiducial suite (both arms)": ("Fiducial suite (both arms)", HERO),
        "m3": (mlabel("m3"), "m3"),
        "envelope": (mlabel("envelope"), "envelope"),
        "working_hotelling": (mlabel("working_hotelling"), "working_hotelling"),
        "ks": (mlabel("ks"), "ks"),
    }
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for method, (label, key) in style.items():
        color, marker = mcolor(key), METHODS[key].marker
        series = med.loc[method]
        by_n = series.groupby(level="n_total").max()
        ax.plot(by_n.index, by_n.to_numpy(), color=color, lw=1.6, label=label)
        for (n, device), value in series.items():
            ax.scatter(
                n,
                value,
                color=color if device == "cpu" else "white",
                edgecolor=color,
                marker=marker,
                s=40,
                zorder=5,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total sample size n")
    ax.set_ylabel("Median seconds per replicate (3 levels)")
    ax.set_title("Computational cost (hollow = GPU)")
    ax.legend(fontsize=8)
    fig.text(
        0.01,
        -0.03,
        "Measured under the run's concurrent CPU/GPU load; "
        "not single-method benchmarks.",
        fontsize=8,
        color="0.4",
    )
    fig.tight_layout()
    return save_figure(fig, "fig23_runtime")


def fig24_calibration_distribution(cells: pd.DataFrame) -> plt.Figure:
    """Per-cell coverage minus nominal for every method and level.

    Each point is one DGP × configuration cell; filled points are cells whose
    95% Wilson interval lies entirely below nominal.

    Args:
        cells: Output of :func:`cell_summary`.

    Returns:
        The figure.
    """
    fig, axes = plt.subplots(
        1, len(ALPHAS), figsize=(5.0 * len(ALPHAS), 4.6), sharey=True
    )
    rng = np.random.default_rng(0)
    for ax, alpha in zip(axes, ALPHAS, strict=True):
        sub = cells[cells["alpha"] == alpha]
        for i, m in enumerate(METHOD_ORDER):
            g = sub[sub["method"] == m]
            dev = g["coverage"] - (1 - alpha)
            below = g["cov_hi"] < (1 - alpha)
            x = i + rng.uniform(-0.25, 0.25, len(g))
            ax.scatter(
                x[~below],
                dev[~below],
                s=14,
                facecolor="white",
                edgecolor=mcolor(m),
                lw=0.8,
            )
            ax.scatter(x[below], dev[below], s=16, color=mcolor(m))
            ax.text(
                i,
                1.02,
                f"{int(below.sum())}/{len(g)}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=mcolor(m),
                transform=ax.get_xaxis_transform(),
            )
        ax.axhline(0, color="0.35", ls="--", lw=1.0)
        ax.set_yscale("symlog", linthresh=0.02)
        ax.set_xticks(
            range(len(METHOD_ORDER)),
            [mlabel(m).split(" (")[0] for m in METHOD_ORDER],
            rotation=40,
            ha="right",
        )
        ax.set_title(f"{level_label(alpha)} band", pad=16)
    axes[0].set_ylabel("Coverage − nominal (per DGP × n cell)")
    fig.suptitle(
        "Calibration in every cell; counts are cells significantly below nominal",
        y=1.04,
    )
    fig.tight_layout()
    return save_figure(fig, "fig24_calibration_distribution")


# =============================================================================
# 9. Coverage by AUC and shape-parameter bins
# =============================================================================

AUC_BINS = [0.5, 0.75, 0.9, 0.95, 1.0]


@dataclass(frozen=True)
class ShapeBins:
    """Bins of one DGP's shape parameter for the AUC × shape coverage grid.

    Attributes:
        column: LHS column holding the parameter.
        edges: Bin edges, spanning the LHS range.
        label: Parameter name for panel labels.
    """

    column: str
    edges: list[float]
    label: str


SHAPE_BINS: dict[str, ShapeBins] = {
    "student_t": ShapeBins("lhs_df", [1.0, 2.0, 4.0, 10.0, 30.0], "df"),
    "hetero_gaussian": ShapeBins(
        "lhs_sigma_ratio", [0.2, 0.5, 1.0, 2.0, 5.0], r"$\sigma_+/\sigma_-$"
    ),
}
COVERAGE_FLOOR = 0.7


def binned_rates(
    *, df: pd.DataFrame, dgp: str, alpha: float, inner: str
) -> pd.DataFrame:
    """Coverage per shape bin, AUC bin, method, and inner grouping variable.

    Args:
        df: Trial-level frame.
        dgp: Key of ``SHAPE_BINS``.
        alpha: Band level.
        inner: Column grouped within each method, e.g. "n_total".

    Returns:
        Coverage and replicate count indexed by (shape_bin, auc_bin, method,
        inner); bins are integer positions into the edge lists. Frames from
        the stratified study (``scripts/run_shape_auc_study.py``) carry the
        design's own ``shape_bin`` and ``auc_bin``, which are used as drawn,
        so every point keeps its full replicate count.
    """
    bins = SHAPE_BINS[dgp]
    sub = df[(df["dgp_type"] == dgp) & (df["alpha"] == alpha)]
    if sub.empty:
        index = pd.MultiIndex.from_tuples(
            [], names=["shape_bin", "auc_bin", "method", inner]
        )
        return pd.DataFrame({"coverage": [], "n": []}, index=index)
    if {"shape_bin", "auc_bin"} <= set(sub.columns):
        return sub.groupby(["shape_bin", "auc_bin", "method", inner], observed=True)[
            "covers_entirely"
        ].agg(coverage="mean", n="size")
    sub = sub.assign(
        shape_bin=pd.cut(
            sub[bins.column], bins.edges, labels=False, include_lowest=True
        ),
        auc_bin=pd.cut(sub["true_auc"], AUC_BINS, labels=False, include_lowest=True),
    )
    return sub.groupby(["shape_bin", "auc_bin", "method", inner], observed=True)[
        "covers_entirely"
    ].agg(coverage="mean", n="size")


def draw_method_groups(
    *,
    ax: plt.Axes,
    rates: pd.DataFrame,
    methods: list[str],
    inner: list,
    color_of: Callable[..., object],
    fill_of: Callable[..., FillStyleType],
) -> None:
    """Draw coverage with Wilson intervals, methods outer and ``inner`` within.

    Points of one method are joined in ``inner`` order. Points below
    ``COVERAGE_FLOOR`` are pinned to the floor as a downward triangle with
    their value printed.

    Args:
        ax: Axes to draw on.
        rates: Coverage and count indexed by (method, inner value).
        methods: Methods in x order.
        inner: Inner values in x order.
        color_of: Marker color for an inner value.
        fill_of: Marker fill style for an inner value.
    """
    width = len(inner) + 1
    for g, m in enumerate(methods):
        xs, ys = [], []
        for i, key in enumerate(inner):
            if (m, key) not in rates.index:
                continue
            row = rates.loc[(m, key)]
            x, y = g * width + i, float(row["coverage"])
            lo, hi = wilson_ci(k=y * row["n"], n=row["n"])
            xs.append(x)
            ys.append(max(y, COVERAGE_FLOOR))
            if y < COVERAGE_FLOOR:
                ax.plot(x, COVERAGE_FLOOR + 0.006, "v", ms=4.5, color=mcolor(m))
                ax.text(
                    x,
                    COVERAGE_FLOOR + 0.02,
                    f"{y:.2f}",
                    fontsize=5,
                    rotation=90,
                    ha="center",
                    va="bottom",
                    color=mcolor(m),
                )
                continue
            ax.plot([x, x], [lo, hi], color=color_of(key), lw=0.9, zorder=3)
            ax.plot(
                x,
                y,
                marker=METHODS[m].marker,
                ms=4.0,
                ls="none",
                color=color_of(key),
                fillstyle=fill_of(key),
                markerfacecoloralt="white",
                markeredgewidth=0.7,
                zorder=4,
            )
        ax.plot(xs, ys, color=mcolor(m), ls=METHODS[m].ls, lw=0.7, alpha=0.6, zorder=2)


def shape_auc_grid(
    *,
    rates: pd.DataFrame,
    dgp: str,
    inner: str,
    inner_values: list,
    color_of: Callable[..., object],
    fill_of: Callable[..., FillStyleType],
    alpha: float,
) -> tuple[plt.Figure, np.ndarray]:
    """Lay out shape bins by row and AUC bins by column, methods along x.

    Only the methods present in ``rates`` take an x slot.

    Args:
        rates: Output of :func:`binned_rates` with the same ``inner``.
        dgp: Key of ``SHAPE_BINS``.
        inner: The inner grouping column.
        inner_values: Inner values in x order.
        color_of: Marker color for an inner value.
        fill_of: Marker fill style for an inner value.
        alpha: Band level.

    Returns:
        The figure and its axes grid.
    """
    bins = SHAPE_BINS[dgp]
    present = set(rates.index.get_level_values("method"))
    methods = [m for m in METHOD_ORDER if m in present]
    n_rows, n_cols = len(bins.edges) - 1, len(AUC_BINS) - 1
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=((0.7 + 0.62 * len(methods)) * n_cols, 2.6 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    head = (HERO, inner_values[0])
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            if (r, c) in rates.index.droplevel(["method", inner]):
                panel = rates.loc[(r, c)]
                draw_method_groups(
                    ax=ax,
                    rates=panel,
                    methods=methods,
                    inner=inner_values,
                    color_of=color_of,
                    fill_of=fill_of,
                )
                if head in panel.index:
                    ax.text(
                        0.01,
                        0.03,
                        f"{int(panel.loc[head, 'n']):,} draws per point",
                        transform=ax.transAxes,
                        fontsize=6.5,
                        color="0.4",
                    )
            add_nominal_line(ax, alpha)
            if r == 0:
                ax.set_title(
                    f"AUC {AUC_BINS[c]:.2f}–{AUC_BINS[c + 1]:.2f}", fontsize=10
                )
            if c == 0:
                ax.set_ylabel(
                    f"{bins.label} {bins.edges[r]:g}–{bins.edges[r + 1]:g}\nCoverage"
                )
    width = len(inner_values) + 1
    centers = [g * width + (len(inner_values) - 1) / 2 for g in range(len(methods))]
    axes[0, 0].set_ylim(COVERAGE_FLOOR, 1.005)
    for ax in axes[-1, :]:
        ax.set_xticks(
            centers,
            [mlabel(m).split(" (")[0] for m in methods],
            rotation=40,
            ha="right",
            fontsize=7.5,
        )
    for ax in axes.flat:
        ax.grid(axis="x", visible=False)
    return fig, axes


def fig25_shape_auc_coverage(
    df: pd.DataFrame, *, dgp: str, alpha: float = 0.05
) -> plt.Figure:
    """Coverage by shape-parameter and AUC bin, methods across x with n inside.

    Args:
        df: Trial-level frame.
        dgp: Key of ``SHAPE_BINS``.
        alpha: Band level.

    Returns:
        The figure.
    """
    stem = f"fig25_shape_auc_coverage_{dgp}_{level_label(alpha)[:-1]}"
    sub = primary(df)
    title = f"{dlabel(dgp)}: coverage by shape and AUC ({level_label(alpha)} bands)"
    if not (sub["dgp_type"] == dgp).any():
        fig, ax = plt.subplots(figsize=(6.0, 1.6))
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"No {dlabel(dgp)} trials in this run", ha="center")
        fig.suptitle(title)
        return save_figure(fig, stem)
    rates = binned_rates(df=sub, dgp=dgp, alpha=alpha, inner="n_total")
    ns = sorted(rates.index.get_level_values("n_total").unique())
    fig, _ = shape_auc_grid(
        rates=rates,
        dgp=dgp,
        inner="n_total",
        inner_values=ns,
        color_of=n_color,
        fill_of=lambda _: "full",
        alpha=alpha,
    )
    fig.legend(
        handles=[
            Line2D([], [], marker="o", ls="none", color=n_color(n), label=f"n = {n:,}")
            for n in ns
        ],
        loc="lower center",
        ncol=len(ns),
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle(title + " — n increases left to right within each method")
    fig.tight_layout()
    return save_figure(fig, stem)


def to_markdown(table: pd.DataFrame) -> str:
    """Render a frame as a GitHub Markdown table, index first.

    Args:
        table: Frame to render.

    Returns:
        The Markdown text.
    """
    header = [str(table.index.name or ""), *map(str, table.columns)]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for index, row in table.iterrows():
        lines.append("| " + " | ".join([str(index), *map(str, row.tolist())]) + " |")
    return "\n".join(lines) + "\n"


def method_summary_table(*, df: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """One row per method summarizing calibration, width, tail risk, and cost.

    Args:
        df: Trial-level frame.
        cells: Output of :func:`cell_summary`.

    Returns:
        The table, also written to the figure directory as CSV and Markdown.
    """
    rows = []
    s95 = df[df["alpha"] == 0.05]
    ks_area = cells[(cells["method"] == "ks") & (cells["alpha"] == 0.05)].set_index(
        ["dgp_type", "n_total", "prevalence"]
    )["area"]
    cost = replicate_cost(df).groupby("method")["seconds"].median()
    cost_key = {
        HERO: "fiducial suite (both arms)",
        "fiducial": "fiducial suite (both arms)",
    }
    for m in METHOD_ORDER:
        mc = cells[cells["method"] == m]
        row: dict[str, object] = {"method": mlabel(m)}
        for alpha in ALPHAS:
            row[f"cov@{level_label(alpha)}"] = mc.loc[
                mc["alpha"] == alpha, "coverage"
            ].mean()
        row["calib err"] = np.mean(
            [
                (mc.loc[mc["alpha"] == a, "coverage"] - (1 - a)).abs().mean()
                for a in ALPHAS
            ]
        )
        row["worst cov@95%"] = mc.loc[mc["alpha"] == 0.05, "coverage"].min()
        row["cells sig. below"] = int((mc["cov_hi"] < 1 - mc["alpha"]).sum())
        m95 = mc[mc["alpha"] == 0.05].set_index(["dgp_type", "n_total", "prevalence"])
        row["area@95%"] = m95["area"].mean()
        row["area ÷ KS"] = (m95["area"] / ks_area).mean()
        sm = s95[s95["method"] == m]
        row["P99 viol@95%"] = sm["max_violation"].quantile(0.99)
        row[">5pp, AUC>0.9"] = (
            sm.loc[sm["true_auc"] > HIGH_AUC, "max_violation"] > 0.05
        ).mean()
        row["median s/replicate"] = cost.get(cost_key.get(m, m), np.nan)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("method").astype(float)
    CONFIG.fig_dir.mkdir(parents=True, exist_ok=True)
    stem = f"table01_method_summary{CONFIG.suffix}"
    table.round(4).to_csv(CONFIG.fig_dir / f"{stem}.csv")
    (CONFIG.fig_dir / f"{stem}.md").write_text(
        to_markdown(table.round(3)), encoding="utf-8"
    )
    return table


# =============================================================================
# Driver
# =============================================================================


def main() -> None:
    """Write every figure and table of one run to its figure directory."""
    parser = argparse.ArgumentParser(description="Write one run's paper figures")
    parser.add_argument("--results-dir", type=Path, default=CONFIG.results_dir)
    parser.add_argument("--fig-dir", type=Path, default=CONFIG.fig_dir)
    parser.add_argument(
        "--prevalence",
        type=float,
        default=CONFIG.prevalence,
        help="The run's primary prevalence",
    )
    args = parser.parse_args()
    tag = "" if args.prevalence == 0.5 else f"{prevalence_label(args.prevalence)}"
    configure(
        results_dir=args.results_dir,
        fig_dir=args.fig_dir,
        prevalence=args.prevalence,
        tag=f"{tag} prevalence" if tag else "",
    )
    set_style()
    df = load_results()
    cells = cell_summary(df)
    paired = paired_fiducial(df)
    figures = [
        lambda: fig01_coverage_heatmaps(
            cells=cells, alphas=[0.05], name="fig01a_hybrid_coverage_heatmap_95"
        ),
        lambda: fig01_coverage_heatmaps(
            cells=cells, alphas=[0.2, 0.5], name="fig01b_hybrid_coverage_heatmap_50_80"
        ),
        *[lambda m=m: fig02_coverage_vs_auc(df, method=m) for m in METHOD_ORDER],
        *[lambda m=m: fig03_coverage_vs_shape(df, method=m) for m in METHOD_ORDER],
        lambda: fig04_level_calibration(cells),
        lambda: fig05_hybrid_vs_wh_heatmap(cells),
        lambda: fig06_coverage_vs_n(cells),
        lambda: fig07_wh_fragility(df),
        lambda: fig08_pareto(cells),
        *[lambda f=f: fig08b_pareto_ci(cells=cells, family=f) for f in DGP_FAMILIES],
        lambda: fig08c_trajectory(df),
        lambda: fig09_width_scale_and_calibration(cells),
        lambda: fig10_width_profile(df),
        lambda: fig11_no_floor_ablation(df=df, cells=cells),
        lambda: fig12_corner_ablation(df),
        lambda: fig13_floor_only_ablation(cells),
        lambda: fig14_single_component_summary(cells),
        lambda: fig15_paired_floor_effect(paired),
        lambda: fig16_floor_region_geometry(df),
        lambda: fig17_misses_by_floor_region(df),
        lambda: fig18_floor_tradeoff_by_region(df),
        *[lambda f=f: fig19_example_bands(family=f) for f in DGP_FAMILIES],
        lambda: fig19d_floor_occupancy(df),
        lambda: fig20_violation_direction_location(df),
        lambda: fig21_violation_magnitude(df),
        *([lambda: fig22_prevalence(df)] if df["prevalence"].nunique() > 1 else []),
        lambda: fig23_runtime(df),
        lambda: fig24_calibration_distribution(cells),
        *[lambda d=d: fig25_shape_auc_coverage(df, dgp=d) for d in SHAPE_BINS],
    ]
    for make in figures:
        plt.close(make())
    print(method_summary_table(df=df, cells=cells).round(3).to_string())
    print(f"\nWrote figures and tables to {CONFIG.fig_dir}/")


if __name__ == "__main__":
    main()
