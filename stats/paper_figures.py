"""Generate publication figures for the envelope_wilson method recommendation.

Produces figures into figures/paper/ demonstrating that envelope_wilson is
the best general-purpose ROC confidence band method.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.feather as pf
import seaborn as sns
from pygam import LogisticGAM, s

from studroc_paper.viz.plot_aggregate import get_method_color, set_publication_style

# ── Config ──────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("data/results/24022026")
OUTPUT_DIR = Path("figures/paper")

DISPLAY_NAMES: dict[str, str] = {
    "envelope_wilson": "Envelope (Wilson)",
    "envelope_wilson_symmetric": "Envelope (Wilson Sym.)",
    "envelope_standard": "Envelope (Standard)",
    "ks": "KS Band",
    "wilson_rectangle_sidak": "Wilson Rect. (Sidak)",
    "wilson_rectangle_bonferroni": "Wilson Rect. (Bonf.)",
    "HT_log_concave_logit_autocalib_wilson": "HT Log-Concave (AutoCalib+Wilson)",
    "HT_log_concave": "HT Log-Concave",
    "HT_log_concave_logit": "HT Log-Concave (logit)",
    "HT_log_concave_logit_autocalib": "HT Log-Concave (AutoCalib)",
    "ellipse_envelope_sweep": "Ellipse Envelope",
    "working_hotelling": "Working-Hotelling",
    "pointwise": "Pointwise",
}

DGP_DISPLAY: dict[str, str] = {
    "beta_opposing": "Beta (Opposing)",
    "bimodal_negative": "Bimodal Negative",
    "gamma": "Gamma",
    "hetero_gaussian": "Hetero. Gaussian",
    "logitnormal": "Logit-Normal",
    "student_t": "Student-t",
    "weibull": "Weibull",
}

COMPETITORS = [
    "envelope_wilson",
    "ellipse_envelope_sweep",
    "working_hotelling",
]

EXTENDED = COMPETITORS + [
    "envelope_wilson_symmetric", "HT_log_concave_logit_autocalib_wilson",
    "wilson_rectangle_bonferroni", "pointwise", "ks", "wilson_rectangle_sidak",
]

ABLATION = ["envelope_wilson", "envelope_standard", "wilson_rectangle_sidak", "ks"]

SAMPLE_SIZES = [10, 30, 100, 300, 1000, 10000]

NOMINAL_95 = 0.95

# Distribution families for fig5
DGP_FAMILIES = {
    "Gaussian-like": ["hetero_gaussian", "logitnormal"],
    "Heavy-tailed / Skewed": ["student_t", "gamma", "weibull"],
    "Non-standard Shape": ["beta_opposing", "bimodal_negative"],
}


def display(method: str) -> str:
    return DISPLAY_NAMES.get(method, method)


def dgp_display(dgp: str) -> str:
    return DGP_DISPLAY.get(dgp, dgp)


def method_color(method: str) -> str:
    return get_method_color(method)


# ── Data Loading ────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load all individual feather results, filtered to prevalence = 0.5."""
    frames = []
    for fp in sorted(RESULTS_DIR.glob("*_individual.feather")):
        frames.append(pf.read_table(fp).to_pandas())
    df = pd.concat(frames, ignore_index=True)
    df["n_total"] = df["n_total"].astype(int)
    df["max_violation"] = np.maximum(df["max_violation_above"], df["max_violation_below"])

    # Filter to balanced prevalence only
    n_before = len(df)
    df = df[df["prevalence"] == 0.5].reset_index(drop=True)
    print(f"Loaded {n_before:,} rows, filtered to prev=0.5: {len(df):,} rows")
    return df


# ── Figure 1a-1e: Coverage Heatmap Panels ─────────────────────────────────

def _make_coverage_heatmap(
    df: pd.DataFrame, method: str, filename: str, title: str,
) -> None:
    """Annotated heatmap: 7 DGPs x 6 sample sizes for one method at 95% CI."""
    set_publication_style()

    sub = df[(df["method"] == method) & (df["alpha"] == 0.05)]
    pivot = sub.groupby(["dgp_type", "n_total"])["covers_entirely"].mean().unstack()
    pivot = pivot.reindex(
        index=sorted(pivot.index, key=lambda x: list(DGP_DISPLAY.keys()).index(x)
                     if x in DGP_DISPLAY else 99),
        columns=SAMPLE_SIZES,
    )
    pivot.index = [dgp_display(d) for d in pivot.index]
    pivot.columns = [f"n={n:,}" for n in pivot.columns]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = sns.diverging_palette(10, 220, s=80, l=55, center="light", as_cmap=True)
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap=cmap, center=0.95,
        vmin=0.80, vmax=1.0, linewidths=0.8, linecolor="white",
        cbar_kws={"label": "Coverage Rate", "shrink": 0.8},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig1_coverage_panels(df: pd.DataFrame) -> None:
    """Generate coverage heatmaps for envelope_wilson and key competitors."""
    panels = [
        ("envelope_wilson", "fig1a_coverage_envelope_wilson.png",
         "Envelope (Wilson) Coverage at 95% CI"),
        ("working_hotelling", "fig1b_coverage_working_hotelling.png",
         "Working-Hotelling Coverage at 95% CI"),
        ("wilson_rectangle_sidak", "fig1c_coverage_wilson_rectangle_sidak.png",
         "Wilson Rect. (Sidak) Coverage at 95% CI"),
        ("ellipse_envelope_sweep", "fig1d_coverage_ellipse_envelope.png",
         "Ellipse Envelope Coverage at 95% CI"),
        ("HT_log_concave_logit_autocalib_wilson", "fig1e_coverage_ht_logconcave.png",
         "HT Log-Concave (AutoCalib+Wilson) Coverage at 95% CI"),
    ]
    for method, filename, title in panels:
        _make_coverage_heatmap(df, method, filename, title)
    print("  Fig 1a-1e: coverage panels done")


# ── Figure 2: Pareto Frontier ──────────────────────────────────────────────

def fig2_pareto_coverage_tightness(df: pd.DataFrame) -> None:
    """Scatter: band area vs coverage for all methods, two panels."""
    set_publication_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, (title, subset) in zip(axes, [
        ("All Settings", df),
        ("High AUC (> 0.85)", df[df["lhs_auc"] > 0.85]),
    ]):
        sub = subset[subset["alpha"] == 0.05]
        records = []
        for method in EXTENDED:
            ms = sub[sub["method"] == method]
            if len(ms) == 0:
                continue
            records.append({
                "method": method,
                "coverage": ms["covers_entirely"].mean(),
                "area": ms["band_area"].mean(),
            })
        tbl = pd.DataFrame(records)

        for _, row in tbl.iterrows():
            m = row["method"]
            marker = "*" if m == "envelope_wilson" else "o"
            size = 200 if m == "envelope_wilson" else 80
            zorder = 10 if m == "envelope_wilson" else 5
            ax.scatter(
                row["area"], row["coverage"],
                c=method_color(m), s=size, marker=marker,
                edgecolors="black" if m == "envelope_wilson" else "none",
                linewidths=1.5 if m == "envelope_wilson" else 0,
                zorder=zorder, label=display(m),
            )

        ax.axhline(NOMINAL_95, color="gray", linestyle="--", alpha=0.6, linewidth=1)
        ax.set_xlabel("Mean Band Area (smaller = tighter)")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0.0, 1.05)
        ax.set_xlim(left=0)

    axes[0].set_ylabel("Coverage Rate (95% CI)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=5,
        bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=9,
    )

    fig.suptitle("Coverage vs. Band Tightness", fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(OUTPUT_DIR / "fig2_pareto_coverage_tightness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Fig 2: Pareto frontier done")


# ── Figure 3: Violation Magnitude Distribution ─────────────────────────────

def fig3_violation_tails(df: pd.DataFrame) -> None:
    """Left: ECDF of max violation. Right: P99 by sample size."""
    set_publication_style()
    methods = ["envelope_wilson", "envelope_wilson_symmetric", "ks",
               "wilson_rectangle_sidak", "HT_log_concave_logit_autocalib_wilson",
               "working_hotelling"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    sub95 = df[df["alpha"] == 0.05]
    for method in methods:
        ms = sub95[sub95["method"] == method]
        viol = np.sort(ms["max_violation"].values)
        ecdf_y = np.arange(1, len(viol) + 1) / len(viol)
        ax1.plot(viol, ecdf_y, color=method_color(method), label=display(method), linewidth=1.5)

    ax1.set_xlim(-0.005, 0.15)
    ax1.set_xlabel("Max Violation Magnitude")
    ax1.set_ylabel("Cumulative Proportion")
    ax1.set_title("Violation Magnitude ECDF (95% CI)", fontsize=11, fontweight="bold")
    ax1.axvline(0.05, color="gray", linestyle=":", alpha=0.5, label="5pp threshold")
    ax1.legend(fontsize=8, frameon=False, loc="lower right")

    for method in methods:
        p99s = []
        for n in SAMPLE_SIZES:
            ms = sub95[(sub95["method"] == method) & (sub95["n_total"] == n)]
            if len(ms) > 0:
                p99s.append(np.percentile(ms["max_violation"], 99))
            else:
                p99s.append(np.nan)
        ax2.plot(
            SAMPLE_SIZES, p99s, color=method_color(method),
            marker="o", markersize=5, linewidth=1.5, label=display(method),
        )

    ax2.set_xscale("log")
    ax2.set_xlabel("Sample Size (n)")
    ax2.set_ylabel("P99 Max Violation")
    ax2.set_title("99th Percentile Violation by n (95% CI)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_violation_tails.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Fig 3: violation tails done")


# ── Figure 4: Component Ablation ───────────────────────────────────────────

def fig4_ablation(df: pd.DataFrame) -> None:
    """Coverage by sample size for envelope_wilson vs its components."""
    set_publication_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, (alpha, title) in zip(
        [ax1, ax2],
        [(0.05, "95% Confidence Level"), (0.5, "50% Confidence Level")],
    ):
        sub = df[df["alpha"] == alpha]
        nominal = 1 - alpha

        for method in ABLATION:
            coverages = []
            for n in SAMPLE_SIZES:
                ms = sub[(sub["method"] == method) & (sub["n_total"] == n)]
                coverages.append(ms["covers_entirely"].mean() if len(ms) > 0 else np.nan)
            ax.plot(
                SAMPLE_SIZES, coverages, color=method_color(method),
                marker="o", markersize=5, linewidth=1.8, label=display(method),
            )

        ax.axhline(nominal, color="gray", linestyle="--", alpha=0.6, linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("Sample Size (n)")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(-0.02, 1.05)

    ax1.set_ylabel("Coverage Rate")
    ax1.legend(fontsize=9, frameon=False)

    fig.suptitle(
        "Component Ablation: Wilson Floor + Envelope Bootstrap Are Both Load-Bearing",
        fontsize=12, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Fig 4: ablation done")


# ── GAM helper ─────────────────────────────────────────────────────────────

def _fit_gam_coverage(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray,
                      n_splines: int = 10) -> np.ndarray | None:
    """Fit a logistic GAM and predict on x_grid. Returns None on failure."""
    try:
        gam = LogisticGAM(s(0, n_splines=n_splines)).fit(x.reshape(-1, 1), y)
        return gam.predict_proba(x_grid.reshape(-1, 1))
    except Exception:
        return None


def _plot_gam_or_binned(ax, x_vals, y_vals, x_grid, color, linewidth, label):
    """Plot GAM fit or fall back to binned means."""
    pred = _fit_gam_coverage(x_vals, y_vals, x_grid)
    if pred is not None:
        ax.plot(x_grid, pred, color=color, linewidth=linewidth, label=label)
    else:
        bins = pd.cut(pd.Series(x_vals), bins=15)
        binned = pd.Series(y_vals).groupby(bins, observed=True).mean()
        ax.plot(
            [iv.mid for iv in binned.index], binned.values,
            color=color, linewidth=linewidth - 0.3, linestyle="--", label=label,
        )


# ── Figure 5a-c: Coverage Trajectories by Distribution Family ─────────────

def fig5_coverage_trajectories(df: pd.DataFrame) -> None:
    """Coverage vs AUC, faceted by distribution family (rows) and n (cols)."""
    set_publication_style()

    n_values = [100, 1000, 10000]
    sub = df[df["alpha"] == 0.05]

    for family_key, (family_name, dgps) in enumerate(DGP_FAMILIES.items()):
        n_rows = len(dgps)
        fig, axes = plt.subplots(
            n_rows, 3, figsize=(14, 3.5 * n_rows), sharey=True, squeeze=False,
        )

        for row, dgp in enumerate(dgps):
            for col, n in enumerate(n_values):
                ax = axes[row, col]
                ns = sub[(sub["n_total"] == n) & (sub["dgp_type"] == dgp)]
                auc_grid = np.linspace(0.55, 0.99, 200)

                for method in COMPETITORS:
                    ms = ns[ns["method"] == method]
                    if len(ms) < 50:
                        continue
                    _plot_gam_or_binned(
                        ax, ms["lhs_auc"].values,
                        ms["covers_entirely"].astype(float).values,
                        auc_grid, method_color(method), 1.8, display(method),
                    )

                ax.axhline(NOMINAL_95, color="gray", linestyle="--", alpha=0.6, linewidth=1)
                ax.set_ylim(0.0, 1.05)

                if row == n_rows - 1:
                    ax.set_xlabel("AUC")
                if col == 0:
                    ax.set_ylabel("Coverage (95% CI)")

                if row == 0:
                    ax.set_title(f"n = {n:,}", fontsize=11, fontweight="bold")

                # Row label
                if col == 0:
                    ax.annotate(
                        dgp_display(dgp), xy=(-0.30, 0.5), xycoords="axes fraction",
                        fontsize=10, fontweight="bold", rotation=90,
                        ha="center", va="center",
                    )

        # Shared legend from first panel
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="lower center", ncol=5,
            bbox_to_anchor=(0.5, -0.06 / n_rows), frameon=False, fontsize=9,
        )
        fig.suptitle(
            f"Coverage vs. AUC: {family_name}",
            fontsize=13, fontweight="bold", y=1.01,
        )
        fig.tight_layout()
        suffix = chr(ord("a") + family_key)
        fig.savefig(
            OUTPUT_DIR / f"fig5{suffix}_coverage_vs_auc_{family_name.lower().replace(' / ', '_').replace(' ', '_')}.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig)

    print("  Fig 5a-c: coverage trajectories done")


# ── Figure 6: Robustness vs Working-Hotelling (2x2) ───────────────────────

def fig6_robustness_vs_binormal(df: pd.DataFrame) -> None:
    """envelope_wilson + ellipse_envelope vs working_hotelling, 2x2 grid."""
    set_publication_style()

    methods = ["envelope_wilson", "ellipse_envelope_sweep", "working_hotelling"]
    n_values = [100, 1000]
    sub = df[df["alpha"] == 0.05]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)

    dgp_configs = [
        {
            "dgp": "student_t",
            "x_col": "lhs_df",
            "xlabel": "Degrees of Freedom",
            "title": "Student-t (Heavy Tails)",
            "invert": True,
        },
        {
            "dgp": "bimodal_negative",
            "x_col": "lhs_mode_separation",
            "xlabel": r"Mode Separation ($\Delta$)",
            "title": "Bimodal Negative",
            "invert": False,
        },
    ]

    for row, cfg in enumerate(dgp_configs):
        for col, n in enumerate(n_values):
            ax = axes[row, col]
            ds = sub[(sub["dgp_type"] == cfg["dgp"]) & (sub["n_total"] == n)]

            for method in methods:
                ms = ds[ds["method"] == method].copy()
                if len(ms) < 30:
                    continue

                x_vals = ms[cfg["x_col"]].values.copy()
                if "transform" in cfg:
                    x_vals = cfg["transform"](x_vals)

                x_grid = np.linspace(x_vals.min(), x_vals.max(), 200)
                _plot_gam_or_binned(
                    ax, x_vals, ms["covers_entirely"].astype(float).values,
                    x_grid, method_color(method), 2, display(method),
                )

            ax.axhline(NOMINAL_95, color="gray", linestyle="--", alpha=0.6)
            if col == 0:
                ax.set_ylabel("Coverage (95% CI)")
            ax.set_xlabel(cfg["xlabel"], fontsize=9)
            if row == 0:
                ax.set_title(f"n = {n:,}", fontsize=11, fontweight="bold")
            ax.set_ylim(0.0, 1.05)

            if col == 0:
                ax.annotate(
                    cfg["title"], xy=(-0.35, 0.5), xycoords="axes fraction",
                    fontsize=10, fontweight="bold", rotation=90,
                    ha="center", va="center",
                )

            if cfg.get("invert"):
                ax.invert_xaxis()

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=10,
    )
    fig.suptitle(
        "Robustness Under Departures from Binormality",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig6_robustness_vs_binormal.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Fig 6: robustness vs binormal done")


# ── Figure 6b: Non-log-concave — envelope_wilson vs HT_log_concave ───────

def fig6b_robustness_non_logconcave(df: pd.DataFrame) -> None:
    """envelope_wilson vs HT_log_concave on non-log-concave distributions."""
    set_publication_style()

    methods = ["envelope_wilson", "HT_log_concave", "HT_log_concave_logit"]
    n_values = [100, 1000]
    sub = df[df["alpha"] == 0.05]

    dgp_configs = [
        {
            "dgp": "bimodal_negative",
            "x_col": "lhs_mode_separation",
            "xlabel": r"Mode Separation ($\Delta$)",
            "title": "Bimodal Negative (mixture)",
        },
        {
            "dgp": "logitnormal",
            "x_col": "lhs_sigma",
            "xlabel": r"$\sigma$",
            "title": "Logit-Normal",
        },
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)

    for row, cfg in enumerate(dgp_configs):
        for col, n in enumerate(n_values):
            ax = axes[row, col]
            ds = sub[(sub["dgp_type"] == cfg["dgp"]) & (sub["n_total"] == n)]

            for method in methods:
                ms = ds[ds["method"] == method].copy()
                if len(ms) < 30:
                    continue

                x_vals = ms[cfg["x_col"]].values.copy()
                x_grid = np.linspace(x_vals.min(), x_vals.max(), 200)
                _plot_gam_or_binned(
                    ax, x_vals, ms["covers_entirely"].astype(float).values,
                    x_grid, method_color(method), 2, display(method),
                )

            ax.axhline(NOMINAL_95, color="gray", linestyle="--", alpha=0.6)
            if col == 0:
                ax.set_ylabel("Coverage (95% CI)")
            ax.set_xlabel(cfg["xlabel"], fontsize=9)
            if row == 0:
                ax.set_title(f"n = {n:,}", fontsize=11, fontweight="bold")
            ax.set_ylim(0.0, 1.05)

            if col == 0:
                ax.annotate(
                    cfg["title"], xy=(-0.35, 0.5), xycoords="axes fraction",
                    fontsize=10, fontweight="bold", rotation=90,
                    ha="center", va="center",
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=10,
    )
    fig.suptitle(
        "Non-Log-Concave Data: Envelope (Wilson) vs. HT Log-Concave",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig6b_robustness_non_logconcave.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Fig 6b: non-log-concave robustness done")


# ── Figure 7: Band Area vs KS ─────────────────────────────────────────────

def fig7_area_vs_ks(df: pd.DataFrame) -> None:
    """Left: area ratio envelope_wilson/ks by DGP and n. Right: coverage comparison."""
    set_publication_style()

    sub = df[df["alpha"] == 0.05]
    dgps = sorted(DGP_DISPLAY.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    cmap = plt.get_cmap("Set2", len(dgps))
    for i, dgp in enumerate(dgps):
        ratios = []
        for n in SAMPLE_SIZES:
            ew = sub[(sub["method"] == "envelope_wilson") & (sub["dgp_type"] == dgp) & (sub["n_total"] == n)]
            ks = sub[(sub["method"] == "ks") & (sub["dgp_type"] == dgp) & (sub["n_total"] == n)]
            if len(ew) > 0 and len(ks) > 0:
                ratios.append(ew["band_area"].mean() / ks["band_area"].mean())
            else:
                ratios.append(np.nan)
        ax1.plot(
            SAMPLE_SIZES, ratios, color=cmap(i), marker="o", markersize=5,
            linewidth=1.5, label=dgp_display(dgp),
        )

    ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xscale("log")
    ax1.set_xlabel("Sample Size (n)")
    ax1.set_ylabel("Band Area Ratio (Envelope Wilson / KS)")
    ax1.set_title("Band Tightness Advantage over KS", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, frameon=False, title="DGP", title_fontsize=9)
    ax1.set_ylim(0.4, 1.05)

    for method in ["envelope_wilson", "ks"]:
        covs = []
        for n in SAMPLE_SIZES:
            ms = sub[(sub["method"] == method) & (sub["n_total"] == n)]
            covs.append(ms["covers_entirely"].mean() if len(ms) > 0 else np.nan)
        ax2.plot(
            SAMPLE_SIZES, covs, color=method_color(method), marker="o",
            markersize=6, linewidth=2, label=display(method),
        )

    ax2.axhline(NOMINAL_95, color="gray", linestyle="--", alpha=0.6)
    ax2.set_xscale("log")
    ax2.set_xlabel("Sample Size (n)")
    ax2.set_ylabel("Coverage Rate (95% CI)")
    ax2.set_title("Coverage Comparison", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=10, frameon=False)
    ax2.set_ylim(0.75, 1.05)

    fig.suptitle(
        "Envelope (Wilson) Produces Tighter Bands Than KS While Maintaining Coverage",
        fontsize=12, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig7_area_vs_ks.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Fig 7: area vs KS done")


# ── Figure 8: Asymptotic Trajectory ───────────────────────────────────────

def fig8_asymptotic_trajectory(df: pd.DataFrame) -> None:
    """Left: coverage vs n. Right: mean max violation vs n."""
    set_publication_style()
    methods = [
        "envelope_wilson", "ks", "wilson_rectangle_sidak",
        "HT_log_concave_logit_autocalib_wilson", "ellipse_envelope_sweep",
        "working_hotelling",
    ]

    sub = df[df["alpha"] == 0.05]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for method in methods:
        covs, viols = [], []
        for n in SAMPLE_SIZES:
            ms = sub[(sub["method"] == method) & (sub["n_total"] == n)]
            if len(ms) > 0:
                covs.append(ms["covers_entirely"].mean())
                viols.append(ms["max_violation"].mean())
            else:
                covs.append(np.nan)
                viols.append(np.nan)

        ax1.plot(SAMPLE_SIZES, covs, color=method_color(method), marker="o",
                 markersize=5, linewidth=1.8, label=display(method))
        ax2.plot(SAMPLE_SIZES, viols, color=method_color(method), marker="o",
                 markersize=5, linewidth=1.8, label=display(method))

    ax1.axhline(NOMINAL_95, color="gray", linestyle="--", alpha=0.6)
    ax1.set_xscale("log")
    ax1.set_xlabel("Sample Size (n)")
    ax1.set_ylabel("Coverage Rate (95% CI)")
    ax1.set_title("Coverage Trajectory", fontsize=11, fontweight="bold")
    ax1.set_ylim(0.0, 1.05)

    ax2.set_xscale("log")
    ax2.set_xlabel("Sample Size (n)")
    ax2.set_ylabel("Mean Max Violation")
    ax2.set_title("Violation Magnitude Trajectory", fontsize=11, fontweight="bold")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=9,
    )

    fig.suptitle("Asymptotic Behavior Across Sample Sizes", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig8_asymptotic_trajectory.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Fig 8: asymptotic trajectory done")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data...")
    df = load_data()

    print("\nGenerating figures...")
    fig1_coverage_panels(df)
    fig2_pareto_coverage_tightness(df)
    fig3_violation_tails(df)
    fig4_ablation(df)
    fig5_coverage_trajectories(df)
    fig6_robustness_vs_binormal(df)
    fig6b_robustness_non_logconcave(df)
    fig7_area_vs_ks(df)
    fig8_asymptotic_trajectory(df)

    print(f"\nAll figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
