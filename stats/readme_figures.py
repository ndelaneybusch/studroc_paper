"""Figures for the README, drawn from the nov11 simulation runs.

The README argues in four steps, and each figure shows only the methods its
step compares:

1. The fiducial band without a floor against Working-Hotelling (coverage) and
   the KS band (width).
2. Where the unfloored fiducial band under-covers, and that its misses sit at
   the two corners of the curve.
3. The exact equal-local-levels (ELL) band against the KS band.
4. The fiducial band with the ELL floor against the unfloored band.

Trials come from the balanced run and the 30% and 10% prevalence runs
(``data/results/nov11``, ``nov11_p30``, ``nov11_p10``). The shape × AUC grid
comes from the stratified study (``data/results/shape_auc_study``, written by
``scripts/run_shape_auc_study.py``). Figures are written to
``figures/readme/`` as PNG.

Usage:
    uv run python stats/readme_figures.py
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import paper_figures_nov11 as pf
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.typing import FillStyleType

FIG_DIR = Path("figures/readme")
RUNS: dict[float, Path] = {
    0.5: Path("data/results/nov11"),
    0.3: Path("data/results/nov11_p30"),
    0.1: Path("data/results/nov11_p10"),
}
SHAPE_AUC_STUDY = Path("data/results/shape_auc_study")
PREV_COLORS = {0.5: "#3B0F70", 0.3: "#B73779", 0.1: "#FC8961"}
PREV_LS = {0.5: "-", 0.3: "--", 0.1: ":"}
METHODS = ["fiducial", "fiducial_hybrid", "m3", "working_hotelling", "ks"]
LABELS = {
    "fiducial": "Fiducial band (no floor)",
    "fiducial_hybrid": "Fiducial band + ELL floor",
    "m3": "ELL band",
    "working_hotelling": "Working–Hotelling",
    "ks": "KS fixed-width",
}
REGION_COLS = [f"violation_{r}" for r in pf.REGIONS]
WIDTH_COLS = [f"width_at_fpr_{x}" for x in pf.WIDTH_LANDMARKS]
PROFILE_NS = [100, 1000, 10000]
STRESS_NS = [100, 300, 1000, 3000]
DPI = 200


def line_style(method: str) -> dict:
    """Return line keywords for a method, with the raw fiducial band solid."""
    style = pf.mline(method)
    if method == "fiducial":
        style["linestyle"] = "-"
    return style


def save(*, fig: plt.Figure, name: str) -> None:
    """Write a figure to the README figure directory and close it."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=DPI)
    plt.close(fig)


def plabel(prevalence: float) -> str:
    """Return a prevalence's legend label."""
    return f"{pf.prevalence_label(prevalence)} prevalence"


# =============================================================================
# Data
# =============================================================================


@dataclass
class Runs:
    """Trials and cell summaries of the three prevalence runs.

    Attributes:
        frames: Trial-level frame per prevalence, restricted to the README's
            methods and to the run's own prevalence.
        cells: Cell summaries of every frame, concatenated.
    """

    frames: dict[float, pd.DataFrame]
    cells: pd.DataFrame

    @property
    def balanced(self) -> pd.DataFrame:
        """Trials of the balanced run."""
        return self.frames[0.5]

    def cells_at(self, prevalence: float) -> pd.DataFrame:
        """Return the cell summaries at one prevalence."""
        return self.cells[self.cells["prevalence"] == prevalence]


def load_runs() -> Runs:
    """Load the three prevalence runs.

    Returns:
        The trials and their cell summaries.
    """
    frames = {}
    for prevalence, path in RUNS.items():
        df = pf.load_results(path)
        keep = (df["prevalence"] == prevalence) & df["method"].isin(METHODS)
        frames[prevalence] = df[keep].reset_index(drop=True)
    cells = pd.concat([pf.cell_summary(f) for f in frames.values()], ignore_index=True)
    return Runs(frames=frames, cells=cells)


def stress_mask(df: pd.DataFrame) -> pd.Series:
    """Flag the draws whose ROC has a convex hook at an end.

    These are Student-t draws with df < 4 and heteroscedastic-Gaussian draws
    whose positives are more than twice as dispersed as the negatives.
    """
    heavy = (df["dgp_type"] == "student_t") & (df["lhs_df"] < 4)
    wide = (df["dgp_type"] == "hetero_gaussian") & (df["lhs_sigma_ratio"] > 2)
    return heavy | wide


STRATA = [
    ("Student-t, df < 4", lambda d: (d["dgp_type"] == "student_t") & (d["lhs_df"] < 4)),
    (
        r"Heterosc. Gaussian, $\sigma_+/\sigma_-$ > 2",
        lambda d: (d["dgp_type"] == "hetero_gaussian") & (d["lhs_sigma_ratio"] > 2),
    ),
    ("All other draws", lambda d: ~stress_mask(d)),
]


# =============================================================================
# Example datasets
# =============================================================================


@dataclass(frozen=True)
class Example:
    """One seeded dataset with its true and empirical ROC and its bands.

    Attributes:
        fpr: Native FPR grid of the rank-space bands.
        true: True TPR on ``fpr``.
        empirical: Empirical TPR on ``fpr``.
        bands: Method name to (grid, lower, upper).
        floor: Grid mask on which the floored band took the ELL hull.
    """

    fpr: np.ndarray
    true: np.ndarray
    empirical: np.ndarray
    bands: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    floor: np.ndarray


def make_example(
    *,
    dgp_type: str,
    auc: float,
    seed: int,
    n: int,
    prevalence: float = 0.5,
    shape: dict[str, float] | None = None,
    alpha: float = 0.05,
) -> Example:
    """Draw one dataset and compute every README band on it.

    Args:
        dgp_type: DGP name.
        auc: Target AUC.
        seed: Seed for the data and the fiducial cloud.
        n: Total sample size.
        prevalence: Share of positives.
        shape: DGP shape parameters; defaults to the DGP's example shape.
        alpha: Band level.

    Returns:
        The dataset's curves and bands.
    """
    from studroc_paper.methods import (
        M3Floor,
        fiducial_band_suite_rs,
        fixed_width_ks_band,
        m3_band_rs,
        working_hotelling_band,
    )

    dgp = pf.make_example_dgp(dgp_type=dgp_type, auc=auc, shape=shape)
    rng = np.random.default_rng(seed)
    n_pos = round(n * prevalence)
    n_neg = n - n_pos
    pos, neg = dgp.sample(n_pos, n_neg, rng)
    y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    y_score = np.concatenate([pos, neg]).astype(np.float64)
    suite = fiducial_band_suite_rs(
        y_true=y_true,
        y_score=y_score,
        alphas=[alpha],
        m3_floors={"fiducial": False, "fiducial_hybrid": M3Floor()},
        random_state=seed,
    )
    bands = {m: (suite.grid, lo, hi) for m, (lo, hi) in suite.bands[alpha].items()}
    bands["m3"] = m3_band_rs(y_true, y_score, alpha=alpha, random_state=seed)
    bands["ks"] = fixed_width_ks_band(y_true, y_score, k=n_neg + 1, alpha=alpha)
    bands["working_hotelling"] = working_hotelling_band(
        y_true, y_score, k=n_neg + 1, alpha=alpha
    )
    above = n_pos - np.searchsorted(np.sort(pos), np.sort(neg)[::-1], side="right")
    return Example(
        fpr=suite.grid,
        true=dgp.get_true_roc(suite.grid),
        empirical=np.concatenate([above, [n_pos]]) / n_pos,
        bands=bands,
        floor=suite.floor_regions[alpha]["fiducial_hybrid"],
    )


def draw_band_edges(
    *, ax: plt.Axes, example: Example, method: str, fill: bool = False, **kwargs
) -> None:
    """Draw one method's band edges, optionally filled.

    Args:
        ax: Target axes.
        example: Dataset holding the band.
        method: Method name.
        fill: Whether to shade between the edges.
        **kwargs: Line keywords overriding the method's defaults.
    """
    grid, lower, upper = example.bands[method]
    style = {"color": pf.mcolor(method), "lw": 1.2} | kwargs
    if fill:
        ax.fill_between(
            grid, lower, upper, step="post", color=style["color"], alpha=0.2, lw=0
        )
    for edge in (lower, upper):
        ax.step(grid, edge, where="post", **style)


def draw_truth(*, ax: plt.Axes, example: Example) -> None:
    """Draw a dataset's empirical and true ROC."""
    ax.step(example.fpr, example.empirical, where="post", color="0.45", lw=1.0)
    ax.plot(example.fpr, example.true, color="black", ls="--", lw=1.4, zorder=6)


CURVE_HANDLES = [
    Line2D([], [], color="black", ls="--", lw=1.4, label="True ROC"),
    Line2D([], [], color="0.45", lw=1.0, label="Empirical ROC"),
]


def fig_example_bands() -> None:
    """Draw 95% fiducial, KS, and Working-Hotelling bands on three datasets."""
    specs = [
        ("binormal", "Binormal"),
        ("student_t", "Student-t (df = 3)"),
        ("bimodal_negative", "Bimodal negatives"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3), sharey=True)
    for i, (ax, (dgp, title)) in enumerate(zip(axes, specs, strict=True)):
        example = make_example(dgp_type=dgp, auc=0.85, seed=20261111 + i, n=300)
        draw_band_edges(ax=ax, example=example, method="ks", lw=1.0)
        draw_band_edges(ax=ax, example=example, method="working_hotelling", lw=1.2)
        draw_band_edges(ax=ax, example=example, method="fiducial", fill=True)
        draw_truth(ax=ax, example=example)
        ax.set(xlim=(0, 1), ylim=(0, 1.02), aspect="equal", xlabel="FPR", title=title)
        ax.grid(False)
    axes[0].set_ylabel("TPR")
    handles = [
        *CURVE_HANDLES,
        Patch(facecolor=pf.mcolor("fiducial"), alpha=0.35, label=LABELS["fiducial"]),
        Line2D([], [], color=pf.mcolor("working_hotelling"), label="Working–Hotelling"),
        Line2D([], [], color=pf.mcolor("ks"), label=LABELS["ks"]),
    ]
    fig.suptitle("95% bands on one dataset per distribution (n = 300, AUC 0.85)")
    fig.tight_layout()
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.01))
    save(fig=fig, name="example_bands")


def fig_corner_examples() -> None:
    """Draw the unfloored and floored bands at a corner the unfloored band misses.

    The seeds are the first, in a search from 1, on which the unfloored band
    misses the true curve at the corner shown and the floored band covers.
    """
    specs = [
        dict(
            title="Student-t (df = 2), AUC 0.97, n = 300, balanced",
            make=dict(
                dgp_type="student_t", auc=0.97, seed=12, n=300, shape={"df": 2.0}
            ),
            zoom=dict(xlim=(0, 0.06), ylim=(0, 1.0)),
            corner="Low-FPR corner",
        ),
        dict(
            title=(
                r"Heterosc. Gaussian ($\sigma_+/\sigma_-$ = 3), AUC 0.95,"
                "\nn = 1,000, 10% prevalence"
            ),
            make=dict(
                dgp_type="hetero_gaussian",
                auc=0.95,
                seed=10,
                n=1000,
                prevalence=0.1,
                shape={"sigma_ratio": 3.0},
            ),
            zoom=dict(xlim=(0.8, 1.0), ylim=(0.9, 1.003)),
            corner="TPR plateau",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.6))
    for row, spec in zip(axes, specs, strict=True):
        example = make_example(**spec["make"])
        grid, lower, _ = example.bands["fiducial"]
        missed = example.true < lower
        missed[[0, -1]] = False
        for ax, zoomed in zip(row, (False, True), strict=True):
            for x0, x1 in pf._spans(example.floor, example.fpr):
                ax.axvspan(x0, x1, color=pf.FLOOR_COLOR, alpha=pf.FLOOR_ALPHA, lw=0)
            draw_band_edges(
                ax=ax, example=example, method="fiducial_hybrid", fill=True, lw=1.3
            )
            draw_band_edges(ax=ax, example=example, method="fiducial", ls="--", lw=1.3)
            draw_truth(ax=ax, example=example)
            ax.plot(
                grid[missed],
                example.true[missed],
                ls="none",
                marker="o",
                ms=3.5 if zoomed else 2.5,
                color="#CC0000",
                zorder=7,
            )
            ax.grid(False)
            ax.set_xlabel("FPR")
            if zoomed:
                ax.set(**spec["zoom"], title=f"{spec['corner']}, enlarged")
            else:
                ax.set(xlim=(0, 1), ylim=(0, 1.02), ylabel="TPR", title=spec["title"])
                (x0, x1), (y0, y1) = spec["zoom"]["xlim"], spec["zoom"]["ylim"]
                ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="0.3", lw=0.8)
    handles = [
        *CURVE_HANDLES,
        Line2D([], [], color=pf.mcolor("fiducial"), ls="--", label=LABELS["fiducial"]),
        Patch(
            facecolor=pf.mcolor("fiducial_hybrid"),
            alpha=0.35,
            label=LABELS["fiducial_hybrid"],
        ),
        Patch(facecolor=pf.FLOOR_COLOR, alpha=pf.FLOOR_ALPHA, label="Floor region"),
        Line2D(
            [],
            [],
            ls="none",
            marker="o",
            ms=4,
            color="#CC0000",
            label="Truth below the unfloored band",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3)
    fig.suptitle("95% bands with and without the floor, at a corner")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig=fig, name="corner_examples")


# =============================================================================
# 1. The fiducial band against Working-Hotelling and KS
# =============================================================================


def draw_coverage_heatmap(
    *,
    ax: plt.Axes,
    cells: pd.DataFrame,
    alpha: float,
    lim: float,
    ylabels: bool,
    cbar: bool = True,
) -> None:
    """Draw one method's coverage minus nominal by DGP and n, annotated.

    Args:
        ax: Target axes.
        cells: Cell summaries of one method at one prevalence.
        alpha: Band level.
        lim: Color limit, as a deviation from nominal.
        ylabels: Whether to label the DGP rows.
        cbar: Whether to draw the color bar.
    """
    cov = cells[cells["alpha"] == alpha].pivot(
        index="dgp_type", columns="n_total", values="coverage"
    )
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
        cbar=cbar,
        cbar_kws={"label": f"Coverage − {1 - alpha:.2f}", "shrink": 0.85},
        annot_kws={"fontsize": 7.5},
        ax=ax,
    )
    ax.set_xticklabels([f"{n:,}" for n in cov.columns], rotation=0)
    ax.set_yticklabels([pf.dlabel(d) for d in cov.index] if ylabels else [], rotation=0)
    ax.set(xlabel="Total sample size n", ylabel="")
    ax.grid(False)


def fig_coverage_heatmap(runs: Runs) -> None:
    """Draw 95% coverage by DGP and n for the fiducial band and WH."""
    cells = runs.cells_at(0.5)
    methods = ["fiducial", "working_hotelling"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6), constrained_layout=True)
    for i, (ax, method) in enumerate(zip(axes, methods, strict=True)):
        draw_coverage_heatmap(
            ax=ax,
            cells=cells[cells["method"] == method],
            alpha=0.05,
            lim=0.5,
            ylabels=i == 0,
        )
        ax.set_title(LABELS[method])
    fig.suptitle("Coverage of 95% bands, balanced classes (6,000 datasets per cell)")
    save(fig=fig, name="coverage_heatmap")


def fig_fiducial_prevalence_heatmap(runs: Runs) -> None:
    """Draw the unfloored band's coverage by DGP, n, level, and prevalence."""
    fig, axes = plt.subplots(
        len(pf.ALPHAS), len(RUNS), figsize=(16.5, 10.5), constrained_layout=True
    )
    for r, alpha in enumerate(pf.ALPHAS):
        for c, prevalence in enumerate(RUNS):
            ax = axes[r, c]
            cells = runs.cells_at(prevalence)
            draw_coverage_heatmap(
                ax=ax,
                cells=cells[(cells["method"] == "fiducial") & (cells["n_total"] >= 30)],
                alpha=alpha,
                lim=pf.DEV_LIMS[alpha],
                ylabels=c == 0,
                cbar=c == len(RUNS) - 1,
            )
            if r == 0:
                ax.set_title(plabel(prevalence))
            if r < len(pf.ALPHAS) - 1:
                ax.set_xlabel("")
            if c == 0:
                ax.set_ylabel(f"{pf.level_label(alpha)} band", fontsize=11)
    fig.suptitle("Fiducial band without a floor: coverage by level and prevalence")
    save(fig=fig, name="fiducial_prevalence_heatmap")


def fig_ell_heatmap(runs: Runs) -> None:
    """Draw the ELL band's coverage by DGP and n at each level."""
    cells = runs.cells_at(0.5)
    cells = cells[cells["method"] == "m3"]
    fig, axes = plt.subplots(
        1, len(pf.ALPHAS), figsize=(18, 4.4), constrained_layout=True
    )
    for i, (ax, alpha) in enumerate(zip(axes, pf.ALPHAS, strict=True)):
        draw_coverage_heatmap(
            ax=ax, cells=cells, alpha=alpha, lim=pf.DEV_LIMS[alpha], ylabels=i == 0
        )
        ax.set_title(f"{pf.level_label(alpha)} band")
    fig.suptitle("ELL band: coverage in every cell, balanced classes")
    save(fig=fig, name="ell_heatmap")


def fig_level_calibration(runs: Runs) -> None:
    """Draw empirical against nominal coverage, averaged over DGPs, by n."""
    cells = runs.cells_at(0.5)
    methods = ["ks", "fiducial", "working_hotelling"]
    ns = [10, 100, 1000, 10000]
    fig, axes = plt.subplots(1, len(ns), figsize=(14.5, 4.0), sharey=True)
    for ax, n in zip(axes, ns, strict=True):
        mean = (
            cells[cells["n_total"] == n]
            .groupby(["method", "alpha"], observed=True)["coverage"]
            .mean()
        )
        for method in methods:
            series = mean.loc[method].sort_index(ascending=False)
            ax.plot(
                1 - series.index,
                series.to_numpy(),
                lw=1.8,
                ms=5.5,
                label=LABELS[method],
                **line_style(method),
            )
        ax.plot([0.4, 1.0], [0.4, 1.0], color="0.35", ls="--", lw=1.0, zorder=1)
        ax.set(title=f"n = {n:,}", xlabel="Nominal coverage", ylim=(0, 1.05))
        ax.set_xticks([0.5, 0.8, 0.95])
    axes[0].set_ylabel("Empirical coverage\n(mean over distributions)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.35", ls="--", lw=1.0))
    labels.append("Perfect calibration")
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle("Calibration across confidence levels (balanced classes)")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig=fig, name="level_calibration")


def draw_rate_curves(
    *, ax: plt.Axes, df: pd.DataFrame, column: str, logx: bool
) -> None:
    """Draw smoothed coverage against one variable, one curve per sample size.

    Each curve is a logistic-GAM smooth with a pointwise 95% ribbon.

    Args:
        ax: Target axes.
        df: Trials of one method, level, and DGP.
        column: Variable on the x-axis.
        logx: Whether to smooth and draw on a log x-axis.
    """
    for n in pf.N_ORDER:
        sub = df[df["n_total"] == n]
        x = sub[column].to_numpy(float)
        gx = np.log10(x) if logx else x
        grid = np.linspace(gx.min(), gx.max(), 120)
        fit = pf.fit_rate_curve(x=gx, y=sub["covers_entirely"].to_numpy(), grid=grid)
        if fit is None:
            continue
        rate, lo, hi = fit
        xs = 10**grid if logx else grid
        ax.plot(xs, rate, color=pf.n_color(n), lw=1.8, label=f"n = {n:,}")
        ax.fill_between(xs, lo, hi, color=pf.n_color(n), alpha=0.14, lw=0)
    if logx:
        ax.set_xscale("log")
    pf.add_nominal_line(ax, 0.05)


def finish_sweep(*, fig: plt.Figure, axes: np.ndarray, used: int) -> None:
    """Label a sweep grid's rows and put the sample-size legend in its last panel.

    Args:
        fig: Figure holding the grid.
        axes: Panel grid whose last panel is unused.
        used: Number of panels holding data.
    """
    for ax in axes.flat[used:]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("Coverage (95% band)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", title="Total sample size")
    fig.tight_layout()


def fig_shape_sweep(
    *, runs: Runs, method: str, ylim: tuple[float, float], title: str, name: str
) -> None:
    """Draw one method's 95% coverage against each DGP's shape parameter.

    Args:
        runs: The loaded runs.
        method: Method to draw.
        ylim: Coverage axis limits.
        title: Figure title.
        name: Output file stem.
    """
    df = runs.balanced
    df = df[(df["alpha"] == 0.05) & (df["method"] == method)]
    rows, cols = pf.grid_shape(len(pf.SHAPE_SPECS))
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.4 * cols, 2.9 * rows), sharey=True, squeeze=False
    )
    for ax, (dgp, col, xlabel, logx) in zip(axes.flat, pf.SHAPE_SPECS, strict=False):
        draw_rate_curves(ax=ax, df=df[df["dgp_type"] == dgp], column=col, logx=logx)
        ax.set_title(pf.dlabel(dgp), fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_ylim(*ylim)
    fig.suptitle(title)
    finish_sweep(fig=fig, axes=axes, used=len(pf.SHAPE_SPECS))
    save(fig=fig, name=name)


def fig_wh_fragility(runs: Runs) -> None:
    """Draw Working-Hotelling coverage against each DGP's shape parameter."""
    fig_shape_sweep(
        runs=runs,
        method="working_hotelling",
        ylim=(-0.03, 1.03),
        title="Working–Hotelling: coverage against each distribution's shape",
        name="wh_fragility",
    )


FIDUCIAL_SWEEP_YLIM = (0.75, 1.005)


def fig_fiducial_fragility(runs: Runs) -> None:
    """Draw the unfloored band's coverage against each DGP's shape parameter."""
    fig_shape_sweep(
        runs=runs,
        method="fiducial",
        ylim=FIDUCIAL_SWEEP_YLIM,
        title="Fiducial band without a floor: coverage against each "
        "distribution's shape",
        name="fiducial_fragility_shape",
    )


def fig_fiducial_auc(runs: Runs) -> None:
    """Draw the unfloored band's 95% coverage against true AUC, per DGP."""
    df = runs.balanced
    df = df[(df["alpha"] == 0.05) & (df["method"] == "fiducial")]
    rows, cols = pf.grid_shape(len(pf.DGP_ORDER))
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.4 * cols, 2.9 * rows), sharey=True, squeeze=False
    )
    for ax, dgp in zip(axes.flat, pf.DGP_ORDER, strict=False):
        draw_rate_curves(
            ax=ax, df=df[df["dgp_type"] == dgp], column="true_auc", logx=False
        )
        ax.set_title(pf.dlabel(dgp), fontsize=10)
        ax.set_xlabel("True AUC", fontsize=8.5)
        ax.set(xlim=(0.55, 1.0), ylim=FIDUCIAL_SWEEP_YLIM)
    fig.suptitle("Fiducial band without a floor: coverage against true AUC")
    finish_sweep(fig=fig, axes=axes, used=len(pf.DGP_ORDER))
    save(fig=fig, name="fiducial_fragility_auc")


def fig_coverage_vs_n(runs: Runs) -> None:
    """Draw 95% coverage against n per DGP with Wilson intervals."""
    cells = runs.cells_at(0.5)
    cells = cells[cells["alpha"] == 0.05]
    methods = ["fiducial", "working_hotelling", "ks"]
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.0), sharex=True, sharey=True)
    for ax, dgp in zip(axes.flat, pf.DGP_ORDER, strict=False):
        for method in methods:
            sub = cells[
                (cells["method"] == method) & (cells["dgp_type"] == dgp)
            ].sort_values("n_total")
            ax.plot(
                sub["n_total"],
                sub["coverage"],
                lw=1.6,
                ms=4.5,
                label=LABELS[method],
                **line_style(method),
            )
            ax.fill_between(
                sub["n_total"],
                sub["cov_lo"],
                sub["cov_hi"],
                color=pf.mcolor(method),
                alpha=0.15,
                lw=0,
            )
        pf.add_nominal_line(ax, 0.05)
        ax.set(xscale="log", ylim=(0, 1.04), title=pf.dlabel(dgp))
    for ax in axes[1]:
        ax.set_xlabel("Total sample size n")
    for ax in axes[:, 0]:
        ax.set_ylabel("Coverage (95% band)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 3].axis("off")
    axes[1, 3].legend(handles, labels, loc="center")
    axes[0, 3].tick_params(labelbottom=True)
    fig.suptitle("Coverage against sample size, by distribution (balanced classes)")
    fig.tight_layout()
    save(fig=fig, name="coverage_vs_n")


def fig_pareto(runs: Runs) -> None:
    """Draw 95% coverage against band area for every DGP at three sample sizes."""
    cells = runs.cells_at(0.5)
    cells = cells[cells["alpha"] == 0.05]
    methods = ["ks", "working_hotelling", "fiducial"]
    fig, axes = plt.subplots(
        len(PROFILE_NS),
        len(pf.DGP_ORDER),
        figsize=(2.3 * len(pf.DGP_ORDER), 2.5 * len(PROFILE_NS)),
        sharex="row",
        sharey=True,
    )
    for r, n in enumerate(PROFILE_NS):
        for c, dgp in enumerate(pf.DGP_ORDER):
            ax = axes[r, c]
            cell = cells[(cells["n_total"] == n) & (cells["dgp_type"] == dgp)]
            for method in methods:
                row = cell[cell["method"] == method].iloc[0]
                ax.errorbar(
                    row["area"],
                    row["coverage"],
                    xerr=1.96 * row["area_se"],
                    yerr=[
                        [max(0.0, row["coverage"] - row["cov_lo"])],
                        [max(0.0, row["cov_hi"] - row["coverage"])],
                    ],
                    fmt="none",
                    ecolor=pf.mcolor(method),
                    elinewidth=1.0,
                    capsize=2,
                )
                ax.scatter(
                    row["area"],
                    row["coverage"],
                    s=70,
                    marker=pf.METHODS[method].marker,
                    color=pf.mcolor(method),
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=5,
                    label=LABELS[method],
                )
            pf.add_nominal_line(ax, 0.05)
            ax.set(ylim=(0, 1.05))
            ax.set_xlim(left=0)
            if r == 0:
                ax.set_title(pf.dlabel(dgp), fontsize=10)
            if r == len(PROFILE_NS) - 1:
                ax.set_xlabel("Mean band area")
            if c == 0:
                ax.set_ylabel(f"n = {n:,}\nCoverage (95% band)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("Coverage against band area, by distribution (balanced classes)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig=fig, name="coverage_vs_area")


def area_ratio(*, cells: pd.DataFrame, numerator: str, denominator: str) -> pd.Series:
    """Return the mean over DGPs of a per-cell area ratio, indexed by n.

    Args:
        cells: Cell summaries at one prevalence and level.
        numerator: Method whose area is on top.
        denominator: Method whose area is underneath.

    Returns:
        The ratio by total sample size.
    """
    area = cells.pivot_table(
        index=["dgp_type", "n_total"], columns="method", values="area", observed=True
    )
    return (area[numerator] / area[denominator]).groupby("n_total").mean()


def draw_width_profiles(
    *, axes: np.ndarray, df: pd.DataFrame, methods: list[str]
) -> None:
    """Draw mean 95% band width at landmark FPRs, one panel per sample size.

    Args:
        axes: One axis per entry of ``PROFILE_NS``.
        df: Trial-level frame at one prevalence.
        methods: Methods to draw.
    """
    sub = df[(df["alpha"] == 0.05) & df["n_total"].isin(PROFILE_NS)]
    widths = sub.groupby(["n_total", "method"], observed=True)[WIDTH_COLS].mean()
    for ax, n in zip(axes, PROFILE_NS, strict=True):
        for method in methods:
            ax.plot(
                pf.WIDTH_LANDMARKS,
                widths.loc[(n, method)].to_numpy(),
                lw=1.8,
                ms=5,
                label=LABELS[method],
                **line_style(method),
            )
        ax.set(xscale="log", xlabel="FPR", title=f"Width by FPR, n = {n:,}")
        ax.set_ylim(0, 1.03)
    axes[0].set_ylabel("Mean band width (TPR)")


def fig_tightness(runs: Runs) -> None:
    """Draw fiducial band area relative to KS and its width along the curve."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.1))
    ax = axes[0]
    for prevalence in RUNS:
        cells = runs.cells_at(prevalence)
        ratio = area_ratio(
            cells=cells[cells["alpha"] == 0.05], numerator="fiducial", denominator="ks"
        )
        ax.plot(
            ratio.index,
            ratio.to_numpy(),
            color=pf.mcolor("fiducial"),
            ls=PREV_LS[prevalence],
            marker="s",
            lw=1.8,
            ms=5,
            label=f"Fiducial, {plabel(prevalence)}",
        )
    cells = runs.cells_at(0.5)
    ratio = area_ratio(
        cells=cells[cells["alpha"] == 0.05],
        numerator="working_hotelling",
        denominator="ks",
    )
    ax.plot(
        ratio.index,
        ratio.to_numpy(),
        lw=1.8,
        ms=5,
        label="Working–Hotelling, 50%",
        **line_style("working_hotelling"),
    )
    ax.axhline(1.0, color=pf.mcolor("ks"), lw=1.2)
    ax.text(10, 0.97, "KS", va="top", fontsize=9)
    ax.set(
        xscale="log",
        ylim=(0, 1.05),
        xlabel="Total sample size n",
        ylabel="Band area ÷ KS band area",
        title="Area relative to KS",
    )
    ax.legend(fontsize=7.5, loc="lower left")
    draw_width_profiles(
        axes=axes[1:], df=runs.balanced, methods=["ks", "fiducial", "working_hotelling"]
    )
    axes[1].legend(fontsize=7.5, loc="lower left")
    fig.suptitle("Width of 95% bands: the fiducial band against KS")
    fig.tight_layout()
    save(fig=fig, name="tightness_vs_ks")


# =============================================================================
# 2. Where the unfloored fiducial band fails
# =============================================================================


@dataclass(frozen=True)
class Series:
    """One coverage line of the stratum figure.

    Attributes:
        method: Method name.
        auc: Half-open true-AUC interval (low, high].
        label: Legend label.
        color: Line color.
        ls: Line style.
        marker: Marker shape.
    """

    method: str
    auc: tuple[float, float]
    label: str
    color: str
    ls: str = "-"
    marker: str = "o"


def fig_stratum_coverage(
    *, runs: Runs, series: list[Series], title: str, name: str
) -> None:
    """Draw 95% coverage against n by shape stratum (rows) and prevalence (columns).

    Args:
        runs: The loaded runs.
        series: Lines to draw in every panel.
        title: Figure title.
        name: Output file stem.
    """
    fig, axes = plt.subplots(
        len(STRATA), len(RUNS), figsize=(11.5, 8.2), sharex=True, sharey=True
    )
    for c, (prevalence, frame) in enumerate(runs.frames.items()):
        df = frame[(frame["alpha"] == 0.05) & (frame["n_total"] >= 30)]
        for r, (stratum, select) in enumerate(STRATA):
            ax = axes[r, c]
            sub = df[select(df)]
            for s in series:
                rows = sub[
                    (sub["method"] == s.method)
                    & (sub["true_auc"] > s.auc[0])
                    & (sub["true_auc"] <= s.auc[1])
                ]
                stats = rows.groupby("n_total")["covers_entirely"].agg(["sum", "size"])
                rate = stats["sum"] / stats["size"]
                lo, hi = pf.wilson_ci(k=stats["sum"], n=stats["size"])
                ax.errorbar(
                    stats.index,
                    rate,
                    yerr=[np.clip(rate - lo, 0, None), np.clip(hi - rate, 0, None)],
                    color=s.color,
                    ls=s.ls,
                    marker=s.marker,
                    lw=1.6,
                    ms=4.5,
                    capsize=2,
                    elinewidth=0.9,
                    label=s.label,
                )
            pf.add_nominal_line(ax, 0.05)
            ax.set(xscale="log", xlim=(20, 15000), ylim=(0.55, 1.01))
            if r == 0:
                ax.set_title(plabel(prevalence))
            if r == len(STRATA) - 1:
                ax.set_xlabel("Total sample size n")
            if c == 0:
                ax.set_ylabel(f"{stratum}\nCoverage (95% band)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(series))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig=fig, name=name)


def fig_hybrid_repair(runs: Runs) -> None:
    """Draw both fiducial arms' coverage at true AUC above 0.9 by stratum."""
    series = [
        Series(
            "fiducial",
            (0.9, 1.0),
            LABELS["fiducial"],
            pf.mcolor("fiducial"),
            ls="--",
            marker="s",
        ),
        Series(
            "fiducial_hybrid",
            (0.9, 1.0),
            LABELS["fiducial_hybrid"],
            pf.mcolor("fiducial_hybrid"),
        ),
    ]
    fig_stratum_coverage(
        runs=runs,
        series=series,
        title="True AUC > 0.9: coverage with and without the floor",
        name="hybrid_repair",
    )


def stress_trials(*, frame: pd.DataFrame, method: str) -> pd.DataFrame:
    """Return one method's 95% trials at n = 100 to 3,000 with stress flags.

    Args:
        frame: Trial-level frame at one prevalence.
        method: Method name.

    Returns:
        The trials with a boolean ``stress`` column marking hook-shaped draws
        (see :func:`stress_mask`) at true AUC above 0.9.
    """
    sub = frame[
        (frame["alpha"] == 0.05)
        & (frame["method"] == method)
        & frame["n_total"].isin(STRESS_NS)
    ]
    return sub.assign(stress=stress_mask(sub) & (sub["true_auc"] > 0.9))


def draw_region_bars(
    *, ax: plt.Axes, rates: dict[str, np.ndarray], colors: dict[str, str]
) -> None:
    """Draw grouped bars of violation rates over the six FPR regions.

    Args:
        ax: Target axes.
        rates: Bar-group label to one rate per FPR region.
        colors: Bar-group label to color.
    """
    width = 0.8 / len(rates)
    x = np.arange(len(pf.REGIONS))
    for i, (label, values) in enumerate(rates.items()):
        offset = (i - (len(rates) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, color=colors[label], label=label)
    ax.set_xticks(x, [f"{r}%" for r in pf.REGIONS], rotation=35)
    ax.set_xlabel("FPR region")
    ax.grid(axis="x", visible=False)


def fig_failure_corners(runs: Runs) -> None:
    """Draw where and in which direction the unfloored band misses."""
    panels = [
        (
            "Student-t, df < 4, AUC > 0.9",
            lambda d: d["stress"] & (d["dgp_type"] == "student_t"),
        ),
        (
            r"Heterosc. Gaussian, $\sigma_+/\sigma_-$ > 2, AUC > 0.9",
            lambda d: d["stress"] & (d["dgp_type"] == "hetero_gaussian"),
        ),
        ("All other draws", lambda d: ~d["stress"]),
    ]
    trials = {
        p: stress_trials(frame=f, method="fiducial") for p, f in runs.frames.items()
    }
    colors = {plabel(p): PREV_COLORS[p] for p in RUNS}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3))
    for ax, (title, select) in zip(axes[:3], panels, strict=True):
        rates = {
            plabel(p): d.loc[select(d), REGION_COLS].mean().to_numpy()
            for p, d in trials.items()
        }
        draw_region_bars(ax=ax, rates=rates, colors=colors)
        ax.set(title=title, ylim=(0, 0.2))
    axes[0].set_ylabel("Share of datasets with a miss in the region")
    axes[0].legend(fontsize=8)
    ax = axes[3]
    directions = [
        ("violation_below", "Truth below the band", "#CC0000"),
        ("violation_above", "Truth above the band", "0.55"),
    ]
    x = np.arange(len(RUNS))
    for i, (col, label, color) in enumerate(directions):
        values = [d.loc[d["stress"], col].mean() for d in trials.values()]
        ax.bar(x + (i - 0.5) * 0.38, values, width=0.38, color=color, label=label)
    ax.set_xticks(x, [pf.prevalence_label(p) for p in RUNS])
    ax.set(
        xlabel="Prevalence",
        ylabel="Share of datasets",
        ylim=(0, 0.2),
        title="Direction of the miss,\nboth stress strata",
    )
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8)
    fig.suptitle(
        "Fiducial band without a floor: where it misses (95% bands, n = 100–3,000)"
    )
    fig.tight_layout()
    save(fig=fig, name="failure_corners")


# =============================================================================
# 3. The ELL band against KS
# =============================================================================


ELL_SHAPE_PANELS = [
    ("student_t", "lhs_df", "Degrees of freedom (lower = heavier tails)"),
    ("hetero_gaussian", "lhs_sigma_ratio", "σ ratio (positive / negative)"),
]
CONTRAST_NS = [300, 1000]


def fig_ell_vs_ks(runs: Runs) -> None:
    """Draw the ELL band's area relative to KS and its coverage against shape.

    The shape panels are the two DGPs on which the unfloored fiducial band
    under-covers; its curves at two sample sizes are drawn for contrast.
    """
    level_colors = {0.05: "#882255", 0.2: "#CC79A7", 0.5: "#E8B5D0"}
    cells = runs.cells_at(0.5)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    ax = axes[0]
    for alpha, color in level_colors.items():
        ratio = area_ratio(
            cells=cells[cells["alpha"] == alpha], numerator="m3", denominator="ks"
        )
        ax.plot(
            ratio.index,
            ratio.to_numpy(),
            color=color,
            marker="^",
            lw=1.8,
            ms=5,
            label=f"{pf.level_label(alpha)} band",
        )
    ax.axhline(1.0, color=pf.mcolor("ks"), lw=1.2)
    ax.text(10, 0.99, "KS", va="top", fontsize=9)
    ax.legend(fontsize=8, loc="lower left")
    ax.set(
        xscale="log",
        ylim=(0.5, 1.03),
        xlabel="Total sample size n",
        ylabel="ELL band area ÷ KS band area",
        title="Area relative to KS",
    )
    df = runs.balanced
    df = df[df["alpha"] == 0.05]
    arms = [("m3", pf.N_ORDER, "-", 1.8), ("fiducial", CONTRAST_NS, "--", 1.2)]
    for ax, (dgp, col, xlabel) in zip(axes[1:], ELL_SHAPE_PANELS, strict=True):
        for method, ns, ls, lw in arms:
            for n in ns:
                sub = df[
                    (df["method"] == method)
                    & (df["dgp_type"] == dgp)
                    & (df["n_total"] == n)
                ]
                gx = np.log10(sub[col].to_numpy(float))
                grid = np.linspace(gx.min(), gx.max(), 120)
                fit = pf.fit_rate_curve(
                    x=gx, y=sub["covers_entirely"].to_numpy(), grid=grid
                )
                if fit is None:
                    continue
                ax.plot(10**grid, fit[0], color=pf.n_color(n), ls=ls, lw=lw)
        pf.add_nominal_line(ax, 0.05)
        ax.set(
            xscale="log",
            ylim=(0.75, 1.01),
            xlabel=xlabel,
            title=f"{pf.dlabel(dgp)}: coverage by shape",
        )
    axes[1].set_ylabel("Coverage (95% band)")
    handles = [
        *[
            Line2D([], [], color=pf.n_color(n), lw=1.8, label=f"ELL, n = {n:,}")
            for n in pf.N_ORDER
        ],
        *[
            Line2D(
                [],
                [],
                color=pf.n_color(n),
                ls="--",
                lw=1.2,
                label=f"Fiducial (no floor), n = {n:,}",
            )
            for n in CONTRAST_NS
        ],
    ]
    axes[2].legend(handles=handles, fontsize=7, loc="lower left", ncol=2)
    fig.suptitle("ELL against KS: less area, and no weak spot (balanced classes)")
    fig.tight_layout()
    save(fig=fig, name="ell_vs_ks")


PREV_FILL: dict[float, FillStyleType] = {0.5: "full", 0.3: "bottom", 0.1: "none"}


def fig_ell_ks_prevalence(runs: Runs, *, n: int = 1000) -> None:
    """Draw ELL and KS coverage and area per DGP, one point per prevalence.

    Args:
        runs: The loaded runs.
        n: Total sample size.
    """
    methods = ["m3", "ks"]
    cells = runs.cells[(runs.cells["n_total"] == n) & (runs.cells["alpha"] == 0.05)]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), sharey=True)
    offsets = dict(zip(methods, (-0.17, 0.17), strict=True))
    for method in methods:
        for i, dgp in enumerate(pf.DGP_ORDER):
            rows = (
                cells[(cells["method"] == method) & (cells["dgp_type"] == dgp)]
                .set_index("prevalence")
                .loc[list(RUNS)]
            )
            y = np.full(len(rows), i + offsets[method])
            axes[0].errorbar(
                rows["coverage"],
                y,
                xerr=[
                    (rows["coverage"] - rows["cov_lo"]).clip(lower=0),
                    (rows["cov_hi"] - rows["coverage"]).clip(lower=0),
                ],
                fmt="none",
                ecolor=pf.mcolor(method),
                elinewidth=0.8,
                capsize=1.5,
                alpha=0.6,
            )
            for ax, col in zip(axes, ["coverage", "area"], strict=True):
                ax.plot(rows[col], y, color=pf.mcolor(method), lw=1.2, zorder=2)
                for x, prevalence in zip(rows[col], rows.index, strict=True):
                    ax.plot(
                        x,
                        y[0],
                        marker=pf.METHODS[method].marker,
                        ms=7,
                        color=pf.mcolor(method),
                        fillstyle=PREV_FILL[prevalence],
                        markerfacecoloralt="white",
                        zorder=3,
                    )
    axes[0].axvline(0.95, color="0.35", ls="--", lw=1.0)
    axes[0].set_xlabel("Coverage (95% band)")
    axes[1].set_xlabel("Mean band area")
    axes[0].set_yticks(
        np.arange(len(pf.DGP_ORDER)), [pf.dlabel(d) for d in pf.DGP_ORDER]
    )
    axes[0].invert_yaxis()
    handles = [
        *[
            Line2D(
                [],
                [],
                color=pf.mcolor(m),
                marker=pf.METHODS[m].marker,
                ms=7,
                label=LABELS[m],
            )
            for m in methods
        ],
        *[
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                ms=7,
                color="0.3",
                fillstyle=PREV_FILL[p],
                markerfacecoloralt="white",
                label=plabel(p),
            )
            for p in RUNS
        ],
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5)
    fig.suptitle(f"ELL and KS at n = {n:,}: coverage and area across prevalences")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig=fig, name="ell_ks_prevalence")


# =============================================================================
# 4. The fiducial band with the ELL floor
# =============================================================================


def fig_worst_cell(runs: Runs) -> None:
    """Draw mean and worst-cell 95% coverage of both fiducial arms by n."""
    arms = ["fiducial", "fiducial_hybrid"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    for ax, prevalence in zip(axes, RUNS, strict=True):
        cells = runs.cells_at(prevalence)
        cells = cells[cells["alpha"] == 0.05]
        for method in arms:
            sub = cells[cells["method"] == method]
            mean = sub.groupby("n_total")["coverage"].mean()
            worst = sub.loc[sub.groupby("n_total")["coverage"].idxmin()].set_index(
                "n_total"
            )
            style = pf.mline(method)
            ax.plot(mean.index, mean.to_numpy(), lw=1.0, ms=3.5, alpha=0.55, **style)
            ax.errorbar(
                worst.index,
                worst["coverage"],
                yerr=[
                    worst["coverage"] - worst["cov_lo"],
                    worst["cov_hi"] - worst["coverage"],
                ],
                lw=2.0,
                ms=5.5,
                capsize=2.5,
                **style,
            )
        pf.add_nominal_line(ax, 0.05)
        ax.set(
            xscale="log",
            ylim=(0.88, 1.003),
            xlabel="Total sample size n",
            title=plabel(prevalence),
        )
    axes[0].set_ylabel("Coverage (95% band)")
    handles = [
        *[
            Line2D(
                [], [], lw=2.0, label=f"{LABELS[m]}: worst distribution", **pf.mline(m)
            )
            for m in arms
        ],
        *[
            Line2D(
                [],
                [],
                lw=1.0,
                alpha=0.55,
                label=f"{LABELS[m]}: mean over distributions",
                **pf.mline(m),
            )
            for m in arms
        ],
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2)
    fig.suptitle("Coverage in the worst distribution, with and without the floor")
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    save(fig=fig, name="worst_cell")


def fig_hybrid_misses(runs: Runs) -> None:
    """Draw where and in which direction both fiducial arms miss."""
    arms = ["fiducial", "fiducial_hybrid"]
    colors = {LABELS[m]: pf.mcolor(m) for m in arms}
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.6), sharey="row")
    for c, (prevalence, frame) in enumerate(runs.frames.items()):
        ax = axes[0, c]
        rates = {}
        for method in arms:
            trials = stress_trials(frame=frame, method=method)
            rates[LABELS[method]] = (
                trials.loc[trials["stress"], REGION_COLS].mean().to_numpy()
            )
        draw_region_bars(ax=ax, rates=rates, colors=colors)
        ax.set_title(f"{plabel(prevalence)}\nStress strata: misses by FPR region")
        ax = axes[1, c]
        df = frame[(frame["alpha"] == 0.05) & (frame["n_total"] >= 30)]
        rate = df.groupby(["method", "n_total"], observed=True)[
            ["violation_below", "violation_above"]
        ].mean()
        for method in arms:
            for col, ls in (("violation_below", "-"), ("violation_above", ":")):
                ax.plot(
                    rate.loc[method].index,
                    rate.loc[method, col].to_numpy(),
                    color=pf.mcolor(method),
                    marker=pf.METHODS[method].marker,
                    ls=ls,
                    lw=1.7,
                    ms=4.5,
                )
        ax.axhline(0.025, color="0.35", ls="--", lw=1.0)
        ax.set(
            xscale="log",
            ylim=(0, 0.03),
            xlabel="Total sample size n",
            title="All draws: miss rate by direction",
        )
    axes[0, 0].set_ylabel("Share of datasets with a miss")
    axes[1, 0].set_ylabel("Share of datasets with a miss")
    axes[0, 0].legend(fontsize=8)
    handles = [
        Line2D([], [], color="0.3", ls="-", label="Truth below the band"),
        Line2D([], [], color="0.3", ls=":", label="Truth above the band"),
        Line2D([], [], color="0.35", ls="--", lw=1.0, label="α/2 = 0.025"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3)
    fig.suptitle("Where and in which direction the 95% bands miss")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig=fig, name="hybrid_misses")


def fig_floor_cost(runs: Runs) -> None:
    """Draw the floored band's area relative to the unfloored band and widths."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.1))
    ax = axes[0]
    for prevalence in RUNS:
        cells = runs.cells_at(prevalence)
        cells = cells[cells["n_total"] >= 30]
        for alpha, marker, width in ((0.05, "o", 1.9), (0.5, "v", 1.0)):
            ratio = area_ratio(
                cells=cells[cells["alpha"] == alpha],
                numerator="fiducial_hybrid",
                denominator="fiducial",
            )
            ax.plot(
                ratio.index,
                ratio.to_numpy(),
                color=PREV_COLORS[prevalence],
                marker=marker,
                lw=width,
                ms=4.5,
            )
    ax.axhline(1.0, color="0.35", ls="--", lw=1.0)
    handles = [
        *[Line2D([], [], color=PREV_COLORS[p], lw=1.9, label=plabel(p)) for p in RUNS],
        Line2D([], [], color="0.3", marker="o", lw=1.9, label="95% band"),
        Line2D([], [], color="0.3", marker="v", lw=1.0, label="50% band"),
    ]
    ax.legend(handles=handles, fontsize=7.5)
    ax.set(
        xscale="log",
        ylim=(0.9, 1.7),
        xlabel="Total sample size n",
        ylabel="Floored area ÷ unfloored area",
        title="Area cost of the floor",
    )
    draw_width_profiles(
        axes=axes[1:], df=runs.balanced, methods=["m3", "fiducial_hybrid", "fiducial"]
    )
    axes[1].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("What the floor costs in width (95% bands unless marked)")
    fig.tight_layout()
    save(fig=fig, name="floor_cost")


def fig_floor_occupancy(runs: Runs) -> None:
    """Draw the share of datasets whose floor region contains each FPR."""
    ns = [30, 100, 300, 1000, 3000, 10000]
    aucs = [
        (0.55, 0.75, "AUC 0.55–0.75"),
        (0.75, 0.9, "AUC 0.75–0.90"),
        (0.9, 1.0, "AUC 0.90–0.99"),
    ]
    prevalences = [0.5, 0.1]
    x = np.linspace(0, 1, 401)
    x_edges = np.concatenate([[0.0], (x[1:] + x[:-1]) / 2, [1.0]])
    fig, axes = plt.subplots(
        len(prevalences), len(aucs), figsize=(13.2, 5.8), sharex=True, sharey=True
    )
    for r, prevalence in enumerate(prevalences):
        frame = runs.frames[prevalence]
        df = frame[(frame["method"] == "fiducial_hybrid") & (frame["alpha"] == 0.05)]
        for c, (lo, hi, label) in enumerate(aucs):
            ax = axes[r, c]
            share = np.zeros((len(ns), len(x)))
            for i, n in enumerate(ns):
                sub = df[
                    (df["n_total"] == n)
                    & (df["true_auc"] >= lo)
                    & (df["true_auc"] < hi)
                ]
                left = sub["floor_region_left_end_fpr"].to_numpy(float)
                right = sub["floor_region_right_start_fpr"].to_numpy(float)
                inside = (x[None, :] <= left[:, None]) | (x[None, :] >= right[:, None])
                share[i] = inside.mean(axis=0)
            mesh = ax.pcolormesh(
                x_edges, np.arange(len(ns) + 1), share, cmap="YlOrBr", vmin=0, vmax=1
            )
            ax.set_yticks(np.arange(len(ns)) + 0.5, [f"{n:,}" for n in ns])
            ax.set_ylim(len(ns), 0)
            ax.grid(False)
            if r == 0:
                ax.set_title(label)
            if r == len(prevalences) - 1:
                ax.set_xlabel("FPR")
            if c == 0:
                ax.set_ylabel(f"{plabel(prevalence)}\nTotal sample size n")
    cbar = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.01)
    cbar.set_label("Share of datasets with the FPR in the floor region")
    fig.suptitle("Where the floor applies (95% band)")
    save(fig=fig, name="floor_occupancy")


def fig_shape_auc_student_t() -> None:
    """Coverage of the three fiducial-family bands by Student-t df and AUC bin.

    Drawn from the stratified study, which gives every (df bin, AUC bin, n)
    point the same number of replicates.
    """
    dgp = "student_t"
    df = pf.load_results(SHAPE_AUC_STUDY)
    rates = pf.binned_rates(df=df, dgp=dgp, alpha=0.05, inner="n_total")
    ns = sorted(rates.index.get_level_values("n_total").unique())
    fig, _ = pf.shape_auc_grid(
        rates=rates,
        dgp=dgp,
        inner="n_total",
        inner_values=ns,
        color_of=pf.n_color,
        fill_of=lambda _: "full",
        alpha=0.05,
    )
    fig.legend(
        handles=[
            Line2D(
                [], [], marker="o", ls="none", color=pf.n_color(n), label=f"n = {n:,}"
            )
            for n in ns
        ],
        loc="lower center",
        ncol=len(ns),
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle(
        "Student-t, 50% prevalence: 95% band coverage by tail weight (df) and "
        "true AUC\n"
        "n increases left to right within each method"
    )
    fig.tight_layout()
    save(fig=fig, name="shape_auc_student_t")


def main() -> None:
    """Write the README figures, or only those named on the command line."""
    parser = argparse.ArgumentParser(description="Write the README figures")
    parser.add_argument(
        "--only", nargs="+", default=None, help="Figure names to write (default: all)"
    )
    only = parser.parse_args().only
    examples: dict[str, Callable[[], None]] = {
        "example_bands": fig_example_bands,
        "corner_examples": fig_corner_examples,
    }
    simulated: dict[str, Callable[[Runs], None]] = {
        "coverage_heatmap": fig_coverage_heatmap,
        "level_calibration": fig_level_calibration,
        "wh_fragility": fig_wh_fragility,
        "coverage_vs_n": fig_coverage_vs_n,
        "coverage_vs_area": fig_pareto,
        "tightness_vs_ks": fig_tightness,
        "fiducial_prevalence_heatmap": fig_fiducial_prevalence_heatmap,
        "fiducial_fragility_shape": fig_fiducial_fragility,
        "fiducial_fragility_auc": fig_fiducial_auc,
        "failure_corners": fig_failure_corners,
        "ell_vs_ks": fig_ell_vs_ks,
        "ell_heatmap": fig_ell_heatmap,
        "ell_ks_prevalence": fig_ell_ks_prevalence,
        "hybrid_repair": fig_hybrid_repair,
        "worst_cell": fig_worst_cell,
        "hybrid_misses": fig_hybrid_misses,
        "floor_cost": fig_floor_cost,
        "floor_occupancy": fig_floor_occupancy,
    }
    stratified: dict[str, Callable[[], None]] = {
        "shape_auc_student_t": fig_shape_auc_student_t
    }
    unknown = set(only or []) - set(examples) - set(simulated) - set(stratified)
    if unknown:
        parser.error(f"unknown figures: {sorted(unknown)}")
    pf.set_style()
    for name, make in examples.items():
        if only is None or name in only:
            make()
    if only is None or set(only) & set(simulated):
        runs = load_runs()
        for name, make in simulated.items():
            if only is None or name in only:
                make(runs)
    for name, make in stratified.items():
        if only is None or name in only:
            make()
    print(f"Wrote README figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
