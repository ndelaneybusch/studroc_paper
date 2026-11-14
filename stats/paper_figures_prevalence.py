"""Cross-prevalence figures for the nov11 runs at 50%, 30%, and 10% prevalence.

The three runs share the method roster, the LHS design (same seed), and the
sample sizes n = 30 to 10,000; they differ only in the class split. Figures
come in two kinds:

- **Per-prevalence figures.** Every faceted figure of
  :mod:`paper_figures_nov11` is redrawn once per prevalence, and each set's
  panels are given identical axis limits so the figures compare directly
  (file stems end in ``_p50``, ``_p30``, ``_p10``).
- **Cross-prevalence figures.** Figures whose key datum was one point per
  method draw it as three line-connected points, one per prevalence
  (filled = 50%, half-filled = 30%, hollow = 10%), plus new views that only
  exist with several prevalences: a method × metric overview, coverage
  shifts against the balanced run, a heteroscedastic-Gaussian spotlight, and
  a replication check of the one cell the standard run and the 10% run share.

Macro averages over DGPs use a prevalence at a sample size only when all
seven DGPs are present there, so a run still in progress never shifts an
average by its composition.

Usage:
    uv run python stats/paper_figures_prevalence.py
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import paper_figures_nov11 as pf
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.typing import FillStyleType

RUNS: dict[float, Path] = {
    0.5: Path("data/results/nov11"),
    0.3: Path("data/results/nov11_p30"),
    0.1: Path("data/results/nov11_p10"),
}
FIG_DIR = Path("figures/paper_prevalence")
N_MIN = 30
NS = [30, 100, 300, 1000, 3000, 10000]

PREV_COLORS = {0.5: "#3B0F70", 0.3: "#B73779", 0.1: "#FC8961"}
PREV_FILL: dict[float, FillStyleType] = {0.5: "full", 0.3: "bottom", 0.1: "none"}


def plabel(prevalence: float) -> str:
    """Return a prevalence's legend label."""
    return f"{pf.prevalence_label(prevalence)} prevalence"


def pstem(prevalence: float) -> str:
    """Return a prevalence's file-stem suffix, e.g. "_p10"."""
    return f"_p{round(prevalence * 100)}"


def prevalence_handles() -> list[Line2D]:
    """Return legend handles for the prevalence fill encoding."""
    return [
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            color="0.3",
            fillstyle=PREV_FILL[p],
            markerfacecoloralt="white",
            ms=7,
            label=plabel(p),
        )
        for p in RUNS
    ]


def method_handles(methods: list[str]) -> list[Line2D]:
    """Return legend handles for methods drawn as connected markers."""
    return [
        Line2D([], [], lw=1.4, ms=7, label=pf.mlabel(m), **pf.mline(m)) for m in methods
    ]


# =============================================================================
# Data
# =============================================================================


@dataclass
class PrevalenceData:
    """Trials and cell summaries of every available prevalence run.

    Attributes:
        frames: Trial-level frame per prevalence, restricted to n >= 30.
        cells: Cell summaries of every frame, concatenated.
        paired: Paired fiducial arms per prevalence.
        replicate_cells: Cell summaries of the standard run's own 10% cell
            (n = 1,000), an independent replicate of that cell of the 10% run.
    """

    frames: dict[float, pd.DataFrame]
    cells: pd.DataFrame
    paired: dict[float, pd.DataFrame]
    replicate_cells: pd.DataFrame

    @property
    def prevalences(self) -> list[float]:
        """Prevalences with data, balanced first."""
        return list(self.frames)


def load_prevalence_data() -> PrevalenceData:
    """Load every prevalence run that has results on disk.

    The standard run's 10% cell is kept apart as a replication check rather
    than mixed into the 10% run, and its n = 10 cells are dropped because the
    imbalanced runs start at n = 30.

    Returns:
        The loaded runs.
    """
    frames, paired, replicate = {}, {}, pd.DataFrame()
    for prevalence, path in RUNS.items():
        if not any(path.glob("*_individual.feather")):
            continue
        df = pf.load_results(path)
        if prevalence == 0.5:
            replicate = pf.cell_summary(df[df["prevalence"] == 0.1])
        df = df[(df["prevalence"] == prevalence) & (df["n_total"] >= N_MIN)]
        frames[prevalence] = df.reset_index(drop=True)
        paired[prevalence] = pf.paired_fiducial(frames[prevalence])
    cells = pd.concat([pf.cell_summary(f) for f in frames.values()], ignore_index=True)
    return PrevalenceData(
        frames=frames, cells=cells, paired=paired, replicate_cells=replicate
    )


def completeness(data: PrevalenceData) -> pd.DataFrame:
    """Replicates per DGP and sample size at the 95% level, by prevalence.

    Args:
        data: Loaded runs.

    Returns:
        DGP × (prevalence, n) replicate counts of the hybrid.
    """
    sub = data.cells[(data.cells["method"] == pf.HERO) & (data.cells["alpha"] == 0.05)]
    table = sub.pivot_table(
        index="dgp_type",
        columns=["prevalence", "n_total"],
        values="n_trials",
        observed=True,
    )
    order = [(p, n) for p in RUNS for n in NS if (p, n) in table.columns]
    return table.reindex(columns=order).astype("Int64")


def complete_prevalences(cells: pd.DataFrame, *, n: int) -> list[float]:
    """Return the prevalences that have every DGP of the balanced run at n.

    Args:
        cells: Cell summaries.
        n: Total sample size.

    Returns:
        Prevalences whose macro averages at n are comparable.
    """
    at = cells[cells["n_total"] == n]
    ref = set(at.loc[at["prevalence"] == 0.5, "dgp_type"])
    return [
        p for p in RUNS if ref and ref <= set(at.loc[at["prevalence"] == p, "dgp_type"])
    ]


def comparable_cells(cells: pd.DataFrame) -> pd.DataFrame:
    """Keep the (prevalence, n) groups whose DGP set is complete.

    Args:
        cells: Cell summaries.

    Returns:
        The rows of complete groups.
    """
    keep = [(p, n) for n in NS for p in complete_prevalences(cells, n=n)]
    index = pd.MultiIndex.from_frame(cells[["prevalence", "n_total"]])
    return cells[index.isin(keep)]


# =============================================================================
# Per-prevalence figures with aligned axes
# =============================================================================


def align_axes(figures: list[plt.Figure]) -> None:
    """Give corresponding panels of figures the union of their axis limits.

    Figures whose panel count differs from the most common one (a run with
    missing DGPs) are left as drawn.

    Args:
        figures: Figures drawn by the same function at different prevalences.
    """
    counts = [len(f.axes) for f in figures]
    modal = max(set(counts), key=counts.count)
    matching = [f for f in figures if len(f.axes) == modal]
    for group in zip(*(f.axes for f in matching), strict=True):
        for get, set_ in (("get_xlim", "set_xlim"), ("get_ylim", "set_ylim")):
            lims = [getattr(ax, get)() for ax in group]
            if lims[0][0] > lims[0][1]:
                new = (max(lo for lo, _ in lims), min(hi for _, hi in lims))
            else:
                new = (min(lo for lo, _ in lims), max(hi for _, hi in lims))
            for ax in group:
                getattr(ax, set_)(new)


def per_prevalence(
    *, data: PrevalenceData, make: Callable[[float], object]
) -> list[plt.Figure]:
    """Draw a figure function once per prevalence and align each set's axes.

    Args:
        data: Loaded runs.
        make: Draws the figure(s) for one prevalence through
            :mod:`paper_figures_nov11`, whose saves are collected rather than
            written.

    Returns:
        Every figure drawn, written to ``FIG_DIR``.
    """
    groups: dict[str, list[tuple[plt.Figure, str]]] = {}
    for prevalence in data.prevalences:
        previous = pf.configure(
            prevalence=prevalence,
            fig_dir=FIG_DIR,
            suffix=pstem(prevalence),
            tag=plabel(prevalence),
            deferred=[],
        )
        try:
            make(prevalence)
            for fig, stem in pf.CONFIG.deferred or []:
                base = stem.removesuffix(pstem(prevalence))
                groups.setdefault(base, []).append((fig, stem))
        finally:
            pf.CONFIG = previous
    figures = []
    previous = pf.configure(fig_dir=FIG_DIR)
    try:
        for members in groups.values():
            align_axes([fig for fig, _ in members])
            for fig, stem in members:
                figures.append(pf.write_figure(fig=fig, stem=stem))
    finally:
        pf.CONFIG = previous
    return figures


def save(fig: plt.Figure, name: str) -> plt.Figure:
    """Write a cross-prevalence figure to ``FIG_DIR``."""
    previous = pf.configure(fig_dir=FIG_DIR, suffix="", tag="", deferred=None)
    try:
        return pf.write_figure(fig=fig, stem=name)
    finally:
        pf.CONFIG = previous


# =============================================================================
# Overview: every method, every prevalence
# =============================================================================


def figP01_coverage_heatmaps(*, cells: pd.DataFrame, method: str) -> plt.Figure:
    """One method's coverage by DGP and n, levels by row, prevalences by column.

    Args:
        cells: Cell summaries.
        method: Method to draw.

    Returns:
        The figure.
    """
    sub = cells[cells["method"] == method]
    prevs = [p for p in RUNS if (sub["prevalence"] == p).any()]
    fig, axes = plt.subplots(
        len(pf.ALPHAS),
        len(prevs),
        figsize=(4.6 * len(prevs), 3.0 * len(pf.ALPHAS)),
        squeeze=False,
        constrained_layout=True,
    )
    for r, alpha in enumerate(pf.ALPHAS):
        lim = pf.DEV_LIMS[alpha] if method != "working_hotelling" else 0.5
        for c, prevalence in enumerate(prevs):
            ax = axes[r, c]
            at = sub[(sub["alpha"] == alpha) & (sub["prevalence"] == prevalence)]
            cov = at.pivot(index="dgp_type", columns="n_total", values="coverage")
            cov = cov.reindex(index=pf.DGP_ORDER, columns=NS)
            sns.heatmap(
                cov - (1 - alpha),
                annot=cov,
                fmt=".3f",
                cmap="RdBu",
                center=0,
                vmin=-lim,
                vmax=lim,
                linewidths=0.6,
                linecolor="white",
                cbar=c == len(prevs) - 1,
                cbar_kws={"label": f"Coverage − {1 - alpha:.2f}", "shrink": 0.85},
                annot_kws={"fontsize": 6.5},
                ax=ax,
            )
            ax.set_xticklabels([f"{n:,}" for n in NS], rotation=0, fontsize=7.5)
            ax.set_yticklabels(
                [pf.dlabel(d) for d in pf.DGP_ORDER] if c == 0 else [], rotation=0
            )
            ax.set_xlabel("n" if r == len(pf.ALPHAS) - 1 else "")
            ax.set_ylabel(f"{pf.level_label(alpha)} band" if c == 0 else "")
            if r == 0:
                ax.set_title(plabel(prevalence))
            ax.grid(False)
    fig.suptitle(f"{pf.mlabel(method)}: coverage by DGP, n, level, and prevalence")
    return save(fig, f"figP01_coverage_heatmaps_{method}")


def figP02_coverage_shift(cells: pd.DataFrame) -> plt.Figure:
    """Coverage at each imbalanced prevalence minus the balanced run, 95%.

    The runs are independent, so a cell's shift carries the Monte Carlo
    error of both (about ±0.006 at 95% coverage and 6,000 replicates).

    Args:
        cells: Cell summaries.

    Returns:
        The figure.
    """
    methods = [pf.HERO, "fiducial", "m3", "envelope"]
    s95 = cells[(cells["alpha"] == 0.05) & cells["method"].isin(methods)]
    cov = s95.pivot_table(
        index=["method", "dgp_type", "n_total"],
        columns="prevalence",
        values="coverage",
        observed=True,
    )
    targets = [p for p in RUNS if p != 0.5 and p in cov.columns]
    fig, axes = plt.subplots(
        len(methods),
        len(targets),
        figsize=(5.4 * len(targets), 2.9 * len(methods)),
        squeeze=False,
        constrained_layout=True,
    )
    for r, m in enumerate(methods):
        for c, prevalence in enumerate(targets):
            ax = axes[r, c]
            delta = (cov[prevalence] - cov[0.5]).loc[m].unstack("n_total")
            delta = delta.reindex(index=pf.DGP_ORDER, columns=NS)
            sns.heatmap(
                delta,
                annot=True,
                fmt="+.3f",
                cmap="RdBu",
                center=0,
                vmin=-0.05,
                vmax=0.05,
                linewidths=0.6,
                linecolor="white",
                cbar=c == len(targets) - 1,
                cbar_kws={"label": "Δ coverage", "shrink": 0.85},
                annot_kws={"fontsize": 6.5},
                ax=ax,
            )
            ax.set_xticklabels([f"{n:,}" for n in NS], rotation=0, fontsize=7.5)
            ax.set_yticklabels(
                [pf.dlabel(d) for d in pf.DGP_ORDER] if c == 0 else [], rotation=0
            )
            ax.set_xlabel("n" if r == len(methods) - 1 else "")
            ax.set_ylabel(pf.mlabel(m) if c == 0 else "", fontsize=9)
            if r == 0:
                ax.set_title(f"{plabel(prevalence)} − 50% prevalence")
            ax.grid(False)
    fig.suptitle("Coverage shift under class imbalance (95% bands)")
    return save(fig, "figP02_coverage_shift_95")


def prevalence_metrics(cells: pd.DataFrame) -> pd.DataFrame:
    """Per-method summary metrics by prevalence and n over complete DGP sets.

    Args:
        cells: Cell summaries.

    Returns:
        Mean and worst-cell 95% coverage, calibration error over the three
        levels, and mean area relative to KS, indexed by (method, prevalence,
        n).
    """
    comp = comparable_cells(cells)
    keys = ["method", "prevalence", "n_total"]
    s95 = comp[comp["alpha"] == 0.05]
    ks = s95[s95["method"] == "ks"].set_index(["dgp_type", "prevalence", "n_total"])[
        "area"
    ]
    s95 = s95.assign(
        area_vs_ks=s95["area"].to_numpy()
        / ks.reindex(
            pd.MultiIndex.from_frame(s95[["dgp_type", "prevalence", "n_total"]])
        ).to_numpy()
    )
    out = s95.groupby(keys, observed=True).agg(
        mean_cov=("coverage", "mean"),
        worst_cov=("coverage", "min"),
        area_vs_ks=("area_vs_ks", "mean"),
    )
    err = (
        comp.assign(err=(comp["coverage"] - (1 - comp["alpha"])).abs())
        .groupby(keys, observed=True)["err"]
        .mean()
    )
    return out.join(err.rename("calib_err"))


def figP03_method_overview(cells: pd.DataFrame) -> plt.Figure:
    """Every method's response to imbalance on four summary metrics.

    Args:
        cells: Cell summaries.

    Returns:
        The figure.
    """
    metrics = prevalence_metrics(cells)
    rows = [
        ("mean_cov", "Mean coverage\n(95%, over DGPs)", (0.0, 1.03)),
        ("worst_cov", "Worst-cell coverage\n(95%)", (0.0, 1.03)),
        ("calib_err", "Mean |coverage − nominal|\n(3 levels, log)", (None, None)),
        ("area_vs_ks", "Band area ÷ KS\n(95%)", (0.0, None)),
    ]
    span = metrics.agg(["min", "max"])
    fig, axes = plt.subplots(
        len(rows),
        len(pf.METHOD_ORDER),
        figsize=(2.75 * len(pf.METHOD_ORDER), 2.4 * len(rows)),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for c, m in enumerate(pf.METHOD_ORDER):
        for r, (col, ylabel, (lo, hi)) in enumerate(rows):
            ax = axes[r, c]
            for prevalence in RUNS:
                key = (m, prevalence)
                if key not in metrics.index.droplevel("n_total"):
                    continue
                series = metrics.loc[key, col]
                ax.plot(
                    series.index,
                    series.to_numpy(),
                    "o-",
                    color=PREV_COLORS[prevalence],
                    lw=1.5,
                    ms=3.5,
                    label=plabel(prevalence),
                )
            if col in ("mean_cov", "worst_cov"):
                pf.add_nominal_line(ax, 0.05)
            if col == "calib_err":
                ax.set_yscale("log")
                ax.set_ylim(span.loc["min", col] / 1.3, span.loc["max", col] * 1.3)
            else:
                ax.set_ylim(lo, span.loc["max", col] * 1.05 if hi is None else hi)
            ax.set_xscale("log")
            if r == 0:
                ax.set_title(pf.mlabel(m).split(" (")[0], fontsize=9.5)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if r == len(rows) - 1:
                ax.set_xlabel("n")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("How each method responds to class imbalance")
    fig.tight_layout()
    return save(fig, "figP03_method_overview")


# =============================================================================
# Three connected points per method (from single-point figures)
# =============================================================================


def _connected(
    *,
    ax: plt.Axes,
    method: str,
    xs: list[float],
    ys: list[float],
    prevalences: list[float],
    size: float,
) -> None:
    """Draw one method's per-prevalence points joined in prevalence order."""
    style = pf.METHODS[method]
    hero = method == pf.HERO
    ax.plot(xs, ys, color=style.color, ls=style.ls, lw=1.6 if hero else 1.1, zorder=4)
    for x, y, p in zip(xs, ys, prevalences, strict=True):
        ax.plot(
            x,
            y,
            marker=style.marker,
            ls="none",
            ms=size * (1.35 if hero else 1.0),
            color=style.color,
            fillstyle=PREV_FILL[p],
            markerfacecoloralt="white",
            markeredgecolor="black" if hero else style.color,
            markeredgewidth=0.9,
            zorder=10 if hero else 6,
        )


def figP08_pareto(cells: pd.DataFrame) -> plt.Figure:
    """Coverage vs. mean band area per method, one point per prevalence.

    Args:
        cells: Cell summaries.

    Returns:
        The figure.
    """
    ns = [100, 300, 1000, 10000]
    s95 = cells[cells["alpha"] == 0.05]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.0), sharey=True)
    for ax, n in zip(axes.flat, ns, strict=True):
        at = s95[s95["n_total"] == n]
        prevs = complete_prevalences(at, n=n)
        macro = (
            at[at["prevalence"].isin(prevs)]
            .groupby(["method", "prevalence"], observed=True)[["coverage", "area"]]
            .mean()
        )
        for m in pf.METHOD_ORDER:
            if m not in macro.index.get_level_values(0):
                continue
            pts = macro.loc[m].reindex(prevs)
            _connected(
                ax=ax,
                method=m,
                xs=list(pts["area"]),
                ys=list(pts["coverage"]),
                prevalences=prevs,
                size=8,
            )
        pf.add_nominal_line(ax, 0.05)
        ax.set_title(f"n = {n:,}")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(left=0)
    for ax in axes[-1, :]:
        ax.set_xlabel("Mean band area (smaller = tighter)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Coverage (95% band, mean over DGPs)")
    fig.legend(
        handles=method_handles(pf.METHOD_ORDER) + prevalence_handles(),
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.09),
    )
    fig.suptitle("Tightness vs. coverage at the 95% level, by prevalence")
    fig.tight_layout()
    return save(fig, "figP08_pareto_area_coverage")


def figP08b_pareto_ci(*, cells: pd.DataFrame, family: str) -> plt.Figure:
    """Per-DGP coverage vs. area with 95% CIs, one point per prevalence.

    Args:
        cells: Cell summaries.
        family: Key of ``DGP_FAMILIES``.

    Returns:
        The figure.
    """
    ns = [100, 1000, 10000]
    s95 = cells[cells["alpha"] == 0.05]
    dgps = pf.DGP_FAMILIES[family]
    fig, axes = plt.subplots(
        len(ns),
        len(dgps),
        figsize=(3.6 * len(dgps), 3.0 * len(ns)),
        sharex="row",
        sharey=True,
        squeeze=False,
    )
    for r, n in enumerate(ns):
        for c, dgp in enumerate(dgps):
            ax = axes[r, c]
            cell = s95[(s95["n_total"] == n) & (s95["dgp_type"] == dgp)]
            for m in pf.METHOD_ORDER:
                rows = cell[cell["method"] == m].set_index("prevalence")
                prevs = [p for p in RUNS if p in rows.index]
                if not prevs:
                    continue
                rows = rows.loc[prevs]
                ax.errorbar(
                    rows["area"],
                    rows["coverage"],
                    xerr=1.96 * rows["area_se"],
                    yerr=[
                        (rows["coverage"] - rows["cov_lo"]).clip(lower=0),
                        (rows["cov_hi"] - rows["coverage"]).clip(lower=0),
                    ],
                    fmt="none",
                    ecolor=pf.mcolor(m),
                    elinewidth=0.9,
                    capsize=1.5,
                    alpha=0.6,
                    zorder=3,
                )
                _connected(
                    ax=ax,
                    method=m,
                    xs=list(rows["area"]),
                    ys=list(rows["coverage"]),
                    prevalences=prevs,
                    size=6,
                )
            pf.add_nominal_line(ax, 0.05)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(left=0)
            if r == 0:
                ax.set_title(pf.dlabel(dgp), fontsize=10)
            if r == len(ns) - 1:
                ax.set_xlabel("Mean band area")
            if c == 0:
                ax.set_ylabel(f"n = {n:,}\nCoverage (95% band)")
    fig.legend(
        handles=method_handles(pf.METHOD_ORDER) + prevalence_handles(),
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.1),
    )
    fig.suptitle(f"Per-DGP coverage–tightness with 95% CIs, by prevalence — {family}")
    fig.tight_layout()
    slug = family.split(" ")[0].lower().replace("-", "_")
    return save(fig, f"figP08b_pareto_ci_{slug}")


def figP22_prevalence_by_dgp(*, cells: pd.DataFrame, n: int) -> plt.Figure:
    """Coverage and area per DGP at one n, three connected prevalences each.

    Args:
        cells: Cell summaries.
        n: Total sample size.

    Returns:
        The figure.
    """
    methods = [pf.HERO, "fiducial", "envelope", "m3"]
    sub = cells[
        (cells["n_total"] == n)
        & (cells["alpha"] == 0.05)
        & cells["method"].isin(methods)
    ]
    dgps = pf.DGP_ORDER
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), sharey=True)
    offsets = np.linspace(-0.27, 0.27, len(methods))
    for off, m in zip(offsets, methods, strict=True):
        for i, dgp in enumerate(dgps):
            rows = sub[(sub["method"] == m) & (sub["dgp_type"] == dgp)].set_index(
                "prevalence"
            )
            prevs = [p for p in RUNS if p in rows.index]
            if not prevs:
                continue
            rows = rows.loc[prevs]
            y = [i + off] * len(prevs)
            axes[0].errorbar(
                rows["coverage"],
                y,
                xerr=[
                    (rows["coverage"] - rows["cov_lo"]).clip(lower=0),
                    (rows["cov_hi"] - rows["coverage"]).clip(lower=0),
                ],
                fmt="none",
                ecolor=pf.mcolor(m),
                elinewidth=0.8,
                capsize=1.2,
                alpha=0.6,
            )
            for ax, col in zip(axes, ["coverage", "area"], strict=True):
                _connected(
                    ax=ax,
                    method=m,
                    xs=list(rows[col]),
                    ys=y,
                    prevalences=prevs,
                    size=5.5,
                )
    axes[0].axvline(0.95, color="0.35", ls="--", lw=1.0)
    axes[0].set_xlabel("Coverage (95% band)")
    axes[1].set_xlabel("Mean band area")
    axes[0].set_yticks(np.arange(len(dgps)), [pf.dlabel(d) for d in dgps])
    axes[0].invert_yaxis()
    fig.legend(
        handles=method_handles(methods) + prevalence_handles(),
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.1),
    )
    fig.suptitle(f"n = {n:,}: coverage and width across prevalences")
    fig.tight_layout()
    return save(fig, f"figP22_prevalence_by_dgp_n{n}")


def figP24_calibration_distribution(cells: pd.DataFrame) -> plt.Figure:
    """Per-cell coverage minus nominal, each cell traced across prevalences.

    Each DGP × n cell is a three-point trace (50%, 30%, 10% left to right
    within the method's slot); filled points are cells whose 95% Wilson
    interval lies entirely below nominal at that prevalence.

    Args:
        cells: Cell summaries.

    Returns:
        The figure.
    """
    slot = {0.5: -0.27, 0.3: 0.0, 0.1: 0.27}
    fig, axes = plt.subplots(
        1, len(pf.ALPHAS), figsize=(5.4 * len(pf.ALPHAS), 5.0), sharey=True
    )
    for ax, alpha in zip(axes, pf.ALPHAS, strict=True):
        sub = cells[cells["alpha"] == alpha]
        for i, m in enumerate(pf.METHOD_ORDER):
            g = sub[sub["method"] == m]
            g = g.assign(
                dev=g["coverage"] - (1 - alpha), below=g["cov_hi"] < (1 - alpha)
            )
            for _, trace in g.groupby(["dgp_type", "n_total"], observed=True):
                trace = trace.set_index("prevalence").reindex(
                    [p for p in RUNS if p in set(trace["prevalence"])]
                )
                xs = [i + slot[p] for p in trace.index]
                ax.plot(xs, trace["dev"], color=pf.mcolor(m), lw=0.5, alpha=0.35)
            for p in RUNS:
                at = g[g["prevalence"] == p]
                if at.empty:
                    continue
                x = np.full(len(at), i + slot[p])
                ax.scatter(
                    x[~at["below"]],
                    at.loc[~at["below"], "dev"],
                    s=12,
                    facecolor="white",
                    edgecolor=pf.mcolor(m),
                    lw=0.7,
                    zorder=4,
                )
                ax.scatter(
                    x[at["below"]],
                    at.loc[at["below"], "dev"],
                    s=14,
                    color=pf.mcolor(m),
                    zorder=5,
                )
                ax.text(
                    i + slot[p],
                    1.01,
                    f"{int(at['below'].sum())}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color=PREV_COLORS[p],
                    transform=ax.get_xaxis_transform(),
                )
        ax.axhline(0, color="0.35", ls="--", lw=1.0)
        ax.set_yscale("symlog", linthresh=0.02)
        ax.set_xticks(
            range(len(pf.METHOD_ORDER)),
            [pf.mlabel(m).split(" (")[0] for m in pf.METHOD_ORDER],
            rotation=40,
            ha="right",
        )
        ax.set_title(f"{pf.level_label(alpha)} band", pad=16)
    axes[0].set_ylabel("Coverage − nominal (per DGP × n cell)")
    fig.legend(
        handles=[
            Line2D([], [], color=PREV_COLORS[p], lw=0, marker="$n$", label=plabel(p))
            for p in RUNS
        ],
        title="Counts of cells significantly below nominal "
        "(left to right within each method)",
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.12),
    )
    fig.suptitle(
        "Per-cell calibration traced from 50% to 30% to 10% prevalence", y=1.04
    )
    fig.tight_layout()
    return save(fig, "figP24_calibration_distribution")


def figP23_runtime(frames: dict[float, pd.DataFrame]) -> plt.Figure:
    """Median wall time per replicate by method and n, one panel per prevalence.

    Args:
        frames: Trial-level frame per prevalence.

    Returns:
        The figure.
    """
    style = {
        "fiducial suite (both arms)": ("Fiducial suite (both arms)", pf.HERO),
        "m3": (pf.mlabel("m3"), "m3"),
        "envelope": (pf.mlabel("envelope"), "envelope"),
        "working_hotelling": (pf.mlabel("working_hotelling"), "working_hotelling"),
        "ks": (pf.mlabel("ks"), "ks"),
    }
    fig, axes = plt.subplots(
        1, len(frames), figsize=(4.6 * len(frames), 4.4), sharey=True, squeeze=False
    )
    for ax, (prevalence, frame) in zip(axes[0], frames.items(), strict=True):
        med = (
            pf.replicate_cost(frame)
            .groupby(["method", "n_total", "device"], observed=True)["seconds"]
            .median()
        )
        for method, (label, key) in style.items():
            if method not in med.index.get_level_values(0):
                continue
            color, marker = pf.mcolor(key), pf.METHODS[key].marker
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
                    s=36,
                    zorder=5,
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Total sample size n")
        ax.set_title(plabel(prevalence))
    axes[0, 0].set_ylabel("Median seconds per replicate (3 levels)")
    axes[0, 0].legend(fontsize=7.5)
    fig.suptitle("Computational cost by prevalence (hollow = GPU)")
    fig.text(
        0.01,
        -0.03,
        "Measured under each run's concurrent CPU/GPU load; "
        "not single-method benchmarks.",
        fontsize=8,
        color="0.4",
    )
    fig.tight_layout()
    return save(fig, "figP23_runtime")


# =============================================================================
# Where imbalance bites: the heteroscedastic Gaussian
# =============================================================================

SIGMA_GROUPS = [
    (0.0, 1 / 1.5, "$\\sigma_+/\\sigma_- < 0.67$\n(positives tighter)"),
    (1 / 1.5, 1.5, "$0.67 \\leq \\sigma_+/\\sigma_- \\leq 1.5$"),
    (1.5, np.inf, "$\\sigma_+/\\sigma_- > 1.5$\n(positives wider)"),
]


def figP25_hetero_spotlight(
    frames: dict[float, pd.DataFrame], *, auc_min: float = 0.9
) -> plt.Figure:
    """Raw fiducial vs. hybrid coverage and corner misses in the hetero Gaussian.

    Columns split the LHS draws by the class standard-deviation ratio, the
    parameter that bends the ROC into a convex hook at one end; rows show
    coverage at high AUC and the violation rate in each end region.

    Args:
        frames: Trial-level frame per prevalence.
        auc_min: True-AUC threshold of the high-AUC subset.

    Returns:
        The figure.
    """
    rows = [
        ("covers_entirely", f"Coverage, AUC > {auc_min}"),
        ("violation_0-10", "Violation rate,\nFPR 0–10%"),
        ("violation_90-100", "Violation rate,\nFPR 90–100%"),
    ]
    methods = [(pf.HERO, "-"), ("fiducial", "--")]
    parts = []
    for frame in frames.values():
        sub = frame[
            (frame["dgp_type"] == "hetero_gaussian")
            & (frame["alpha"] == 0.05)
            & frame["method"].isin([m for m, _ in methods])
            & (frame["true_auc"] > auc_min)
        ]
        if not sub.empty:
            parts.append(sub)
    data = pd.concat(parts, ignore_index=True)
    data["group"] = pd.cut(
        data["lhs_sigma_ratio"],
        [lo for lo, _, _ in SIGMA_GROUPS] + [np.inf],
        labels=False,
        right=False,
    )
    rates = data.groupby(
        ["method", "prevalence", "group", "n_total"], observed=True
    ).agg(**{col: (col, "mean") for col, _ in rows}, n=("covers_entirely", "size"))
    fig, axes = plt.subplots(
        len(rows),
        len(SIGMA_GROUPS),
        figsize=(4.0 * len(SIGMA_GROUPS), 2.8 * len(rows)),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for c, (_, _, title) in enumerate(SIGMA_GROUPS):
        for r, (col, ylabel) in enumerate(rows):
            ax = axes[r, c]
            for m, ls in methods:
                for prevalence in RUNS:
                    key = (m, prevalence, c)
                    if key not in rates.index.droplevel("n_total"):
                        continue
                    g = rates.loc[key]
                    ax.plot(
                        g.index,
                        g[col],
                        ls=ls,
                        marker="o",
                        ms=3.5,
                        lw=1.5,
                        color=PREV_COLORS[prevalence],
                    )
                    if col == "covers_entirely":
                        lo, hi = pf.wilson_ci(k=g[col] * g["n"], n=g["n"])
                        ax.fill_between(
                            g.index,
                            lo,
                            hi,
                            color=PREV_COLORS[prevalence],
                            alpha=0.1,
                            lw=0,
                        )
            if col == "covers_entirely":
                pf.add_nominal_line(ax, 0.05)
            else:
                ax.set_ylim(0, float(rates[col].max()) * 1.08)
            ax.set_xscale("log")
            if r == 0:
                ax.set_title(title, fontsize=10)
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == len(rows) - 1:
                ax.set_xlabel("Total sample size n")
    handles = [
        Line2D([], [], color="0.3", ls=ls, lw=1.5, label=pf.mlabel(m))
        for m, ls in methods
    ] + [Line2D([], [], color=PREV_COLORS[p], lw=2.5, label=plabel(p)) for p in RUNS]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        "Heteroscedastic Gaussian, 95% bands: imbalance exposes the hooked ends"
    )
    fig.tight_layout()
    return save(fig, "figP25_hetero_spotlight")


def figP27_shape_auc_coverage(
    frames: dict[float, pd.DataFrame], *, dgp: str, alpha: float = 0.05
) -> plt.Figure:
    """Coverage by shape and AUC bin, pooled over n, prevalence inside method.

    A prevalence enters only when the DGP has every sample size in ``NS`` at
    that prevalence, so the pooled n mix is the same for every point.

    Args:
        frames: Trial-level frame per prevalence.
        dgp: Key of ``pf.SHAPE_BINS``.
        alpha: Band level.

    Returns:
        The figure.
    """
    parts = []
    for frame in frames.values():
        sub = frame[frame["dgp_type"] == dgp]
        if set(NS) <= set(sub["n_total"].unique()):
            parts.append(sub)
    title = (
        f"{pf.dlabel(dgp)}: coverage by shape and AUC "
        f"({pf.level_label(alpha)} bands, pooled over n = 30–10,000)"
    )
    if not parts:
        fig, ax = plt.subplots(figsize=(6.0, 1.6))
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"No complete {pf.dlabel(dgp)} run", ha="center")
        fig.suptitle(title)
        return save(fig, f"figP27_shape_auc_coverage_{dgp}")
    pooled = pd.concat(parts, ignore_index=True)
    rates = pf.binned_rates(df=pooled, dgp=dgp, alpha=alpha, inner="prevalence")
    prevs = [p for p in RUNS if p in set(rates.index.get_level_values("prevalence"))]
    fig, _ = pf.shape_auc_grid(
        rates=rates,
        dgp=dgp,
        inner="prevalence",
        inner_values=prevs,
        color_of=lambda p: PREV_COLORS[p],
        fill_of=lambda p: PREV_FILL[p],
        alpha=alpha,
    )
    fig.legend(
        handles=[
            h for h in prevalence_handles() if h.get_label() in map(plabel, prevs)
        ],
        loc="lower center",
        ncol=len(prevs),
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle(title + " — prevalence falls left to right within each method")
    fig.tight_layout()
    return save(fig, f"figP27_shape_auc_coverage_{dgp}")


# =============================================================================
# Replication check and summary tables
# =============================================================================


def replication_check(data: PrevalenceData) -> pd.DataFrame:
    """Compare the standard run's n = 1,000, 10% cell with the 10% run's.

    The two cells share the design and differ only in their random draws, so
    their coverage differences should look like Monte Carlo noise.

    Args:
        data: Loaded runs.

    Returns:
        One row per method, level, and DGP with both coverages, their
        difference, and its z-score.
    """
    if data.replicate_cells.empty or 0.1 not in data.frames:
        return pd.DataFrame()
    keys = ["method", "alpha", "dgp_type"]
    a = data.replicate_cells.set_index(keys)
    b = data.cells[(data.cells["prevalence"] == 0.1) & (data.cells["n_total"] == 1000)]
    b = b.set_index(keys)
    out = pd.DataFrame(
        {"standard_run": a["coverage"], "p10_run": b["coverage"]}
    ).dropna()
    var = (
        a["coverage"] * (1 - a["coverage"]) / a["n_trials"]
        + b["coverage"] * (1 - b["coverage"]) / b["n_trials"]
    )
    out["diff"] = out["p10_run"] - out["standard_run"]
    out["z"] = out["diff"] / np.sqrt(var.reindex(out.index).clip(lower=1e-8))
    return out


def figP26_replication(table: pd.DataFrame) -> plt.Figure:
    """Coverage of the shared n = 1,000, 10% cell in the two independent runs.

    Args:
        table: Output of :func:`replication_check`.

    Returns:
        The figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    ax = axes[0]
    for m in pf.METHOD_ORDER:
        if m not in table.index.get_level_values("method"):
            continue
        g = table.loc[m]
        ax.scatter(
            g["standard_run"],
            g["p10_run"],
            s=26,
            marker=pf.METHODS[m].marker,
            color=pf.mcolor(m),
            label=pf.mlabel(m),
            alpha=0.85,
        )
    ax.plot([0, 1], [0, 1], color="0.35", ls="--", lw=1.0)
    ax.set_xlabel("Coverage, standard run")
    ax.set_ylabel("Coverage, 10% run")
    ax.set_title("Every method × level × DGP")
    ax.legend(fontsize=7)
    ax = axes[1]
    z = np.sort(table["z"].to_numpy(float))
    expected = pd.Series(np.arange(1, len(z) + 1) / (len(z) + 1))
    from scipy.stats import norm

    ax.plot(norm.ppf(expected), z, "o", ms=3.5, color="0.3")
    lim = max(3.0, float(np.abs(z).max()) + 0.3)
    ax.plot([-lim, lim], [-lim, lim], color="0.35", ls="--", lw=1.0)
    ax.set_xlabel("Normal quantile")
    ax.set_ylabel("z of the coverage difference")
    ax.set_title("Differences vs. pure Monte Carlo noise")
    fig.suptitle("Replication: the n = 1,000, 10% cell drawn twice")
    fig.tight_layout()
    return save(fig, "figP26_replication")


def summary_tables(data: PrevalenceData) -> pd.DataFrame:
    """Write the method summary table at every prevalence and stack them.

    Args:
        data: Loaded runs.

    Returns:
        The stacked table, indexed by (prevalence, method), also written to
        ``FIG_DIR`` as CSV and Markdown.
    """
    tables = {}
    for prevalence, frame in data.frames.items():
        previous = pf.configure(
            prevalence=prevalence, fig_dir=FIG_DIR, suffix=pstem(prevalence)
        )
        try:
            cells = data.cells[data.cells["prevalence"] == prevalence]
            tables[pf.prevalence_label(prevalence)] = pf.method_summary_table(
                df=frame, cells=cells
            )
        finally:
            pf.CONFIG = previous
    stacked = pd.concat(tables, names=["prevalence", "method"])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stacked.round(4).to_csv(FIG_DIR / "table02_method_summary_by_prevalence.csv")
    flat = stacked.round(3).reset_index()
    flat.index = flat.pop("prevalence") + " · " + flat.pop("method")
    (FIG_DIR / "table02_method_summary_by_prevalence.md").write_text(
        pf.to_markdown(flat), encoding="utf-8"
    )
    return stacked


# =============================================================================
# Driver
# =============================================================================


def per_prevalence_figures(
    data: PrevalenceData,
) -> list[Callable[[], plt.Figure | list[plt.Figure]]]:
    """Return the per-prevalence redraws of the faceted figures.

    Args:
        data: Loaded runs.

    Returns:
        Thunks that each draw one aligned set.
    """
    f, c, pr = data.frames, data.cells, data.paired

    def cells_at(p: float) -> pd.DataFrame:
        return c[c["prevalence"] == p]

    makers: list[Callable[[float], object]] = [
        lambda p: [pf.fig02_coverage_vs_auc(f[p], method=m) for m in pf.METHOD_ORDER],
        lambda p: [pf.fig03_coverage_vs_shape(f[p], method=m) for m in pf.METHOD_ORDER],
        lambda p: pf.fig04_level_calibration(cells_at(p)),
        lambda p: pf.fig06_coverage_vs_n(cells_at(p)),
        lambda p: pf.fig07_wh_fragility(f[p]),
        lambda p: pf.fig08c_trajectory(f[p]),
        lambda p: pf.fig09_width_scale_and_calibration(cells_at(p)),
        lambda p: pf.fig10_width_profile(f[p]),
        lambda p: pf.fig11_no_floor_ablation(df=f[p], cells=cells_at(p)),
        lambda p: pf.fig12_corner_ablation(f[p]),
        lambda p: pf.fig13_floor_only_ablation(cells_at(p)),
        lambda p: pf.fig14_single_component_summary(cells_at(p)),
        lambda p: pf.fig15_paired_floor_effect(pr[p]),
        lambda p: pf.fig16_floor_region_geometry(f[p]),
        lambda p: pf.fig17_misses_by_floor_region(f[p]),
        lambda p: pf.fig18_floor_tradeoff_by_region(f[p]),
        lambda p: [pf.fig19_example_bands(family=fam) for fam in pf.DGP_FAMILIES],
        lambda p: pf.fig19d_floor_occupancy(f[p]),
        lambda p: pf.fig20_violation_direction_location(f[p]),
        lambda p: pf.fig21_violation_magnitude(f[p]),
    ]
    return [lambda make=make: per_prevalence(data=data, make=make) for make in makers]


def main() -> None:
    """Write every cross-prevalence figure and table to ``FIG_DIR``."""
    pf.set_style()
    data = load_prevalence_data()
    print(completeness(data).to_string())
    cells = data.cells
    replication = replication_check(data)
    figures: list[Callable[[], plt.Figure | list[plt.Figure]]] = [
        *[
            lambda m=m: figP01_coverage_heatmaps(cells=cells, method=m)
            for m in pf.METHOD_ORDER
        ],
        lambda: figP02_coverage_shift(cells),
        lambda: figP03_method_overview(cells),
        lambda: figP08_pareto(cells),
        *[
            lambda fam=fam: figP08b_pareto_ci(cells=cells, family=fam)
            for fam in pf.DGP_FAMILIES
        ],
        *[lambda n=n: figP22_prevalence_by_dgp(cells=cells, n=n) for n in (100, 1000)],
        lambda: figP22_prevalence_by_dgp(cells=cells, n=10000),
        lambda: figP23_runtime(data.frames),
        lambda: figP24_calibration_distribution(cells),
        lambda: figP25_hetero_spotlight(data.frames),
        *[
            lambda d=d: figP27_shape_auc_coverage(data.frames, dgp=d)
            for d in pf.SHAPE_BINS
        ],
        *([lambda: figP26_replication(replication)] if not replication.empty else []),
        *per_prevalence_figures(data),
    ]
    for make in figures:
        made = make()
        for fig in [made] if isinstance(made, plt.Figure) else made:
            plt.close(fig)
    print(summary_tables(data).round(3).to_string())
    print(f"\nWrote figures and tables to {FIG_DIR}/")


if __name__ == "__main__":
    main()
