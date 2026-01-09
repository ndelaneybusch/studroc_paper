# %% [markdown]
# # Results notebook
#
# This notebook generates the analysis and visualizations for the paper.

# %%
import sys
from pathlib import Path

# Add src to path if running interactively
if str(Path("src").resolve()) not in sys.path:
    sys.path.append(str(Path("src").resolve()))

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

import studroc_paper.viz.plot_aggregate as viz_s
import studroc_paper.viz.plot_aggregate_curve as viz_c
import studroc_paper.viz.plot_indiv as viz_i

# %load_ext autoreload
# %autoreload 2
# from studroc_paper.eval.build_data_from_jsons import process_folder
from studroc_paper.eval.build_data_from_jsons import process_folder


# Helper to save figures
def save_figure(name):
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    clean_name = (
        name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("=", "")
        .replace(",", "")
        .lower()
    )
    path = output_dir / f"{clean_name}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {path}")
    plt.close()


# %% [markdown]
# ## Data Loading

# %%
# Configure results folder
RESULTS_FOLDER = Path("data/results")

# Load dataframes
print(f"Loading results from {RESULTS_FOLDER}...")
try:
    dfs = process_folder(RESULTS_FOLDER)
    print(f"Loaded {len(dfs)} dataframes.")
except FileNotFoundError:
    print(f"Error: Results folder {RESULTS_FOLDER} not found or empty.")
    dfs = {}

# Print available keys
for key in sorted(dfs.keys()):
    print(f"{key}: {len(dfs[key])} rows")

# Process trial-level feathers
print("Loading trial-level results...")
trial_dfs = []
if RESULTS_FOLDER.exists():
    for f in RESULTS_FOLDER.glob("*.feather"):
        try:
            df = pd.read_feather(f)
            # Ensure necessary columns like 'nominal_alpha' exist or are inferred if needed
            # Assuming they are in the dataframe as per `run_experiment.py` output
            trial_dfs.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")

if trial_dfs:
    full_trial_df = pd.concat(trial_dfs, ignore_index=True)
    if "dgp_type" in full_trial_df.columns and "dist" not in full_trial_df.columns:
        full_trial_df["dist"] = full_trial_df["dgp_type"]
    print(f"Loaded {len(full_trial_df)} trial-level rows.")
else:
    print("Warning: No feather files found.")
    full_trial_df = pd.DataFrame()

# %% [markdown]
# ## Subsets

# %%
# Define method subsets
SUBSETS = {
    "core": [
        "envelope_wilson",
        "envelope_wilson_symmetric",
        "envelope_logit_wilson",
        "HT_log_concave",
        "HT_kde_wilson",
        "HT_kde_calib_wilson",
        "ellipse_envelope_sweep",
        "ks",
        "working_hotelling",
    ],
    "envelope": [
        "envelope_standard",
        "envelope_symmetric",
        "envelope_kde",
        "envelope_wilson",
        "envelope_wilson_symmetric",
        "envelope_logit",
        "envelope_wilson_logit",
        "envelope_kde_logit",
    ],
    "HT": [
        "HT_log_concave",
        "HT_log_concave_calib",
        "HT_reflected_kde",
        "HT_kde",
        "HT_kde_calib",
        "HT_kde_wilson",
        "HT_kde_calib_wilson",
    ],
    "reference": [
        "ellipse_envelope_sweep",
        "ellipse_envelope_quartic",
        "working_hotelling",
        "logit_max_modulus",
        "ks",
        "pointwise",
    ],
}


# Helper to filter by subset
def filter_by_subset(df, subset_name):
    if subset_name not in SUBSETS:
        return df
    return df[df["model"].isin(SUBSETS[subset_name])]


# %% [markdown]
# ## Visualizations
#
# ### Overview Panels
#
# A summary of results across all methods.


# %%
# Helper function to get available alphas
def get_alphas(dfs):
    return sorted([k.split("_")[1] for k in dfs.keys() if "_standard" in k])


ALPHAS = get_alphas(dfs)
print(f"Available alphas: {ALPHAS}")

# %% [markdown]
# #### Pareto Frontier by DGP and Alpha
#
# Data filtered for prevalence=50%. Reduced over n_total (mean) for clarity.

# %%
# Collect all standard dataframes
all_dfs = []
for key, df in dfs.items():
    if key.endswith("_standard"):
        alpha = key.split("_")[1]
        df_copy = df.copy()
        df_copy["alpha"] = alpha  # Ensure alpha column exists if not already
        all_dfs.append(df_copy)

if all_dfs:
    full_standard_df = pd.concat(all_dfs, ignore_index=True)
    if (
        "dgp_type" in full_standard_df.columns
        and "dist" not in full_standard_df.columns
    ):
        full_standard_df["dist"] = full_standard_df["dgp_type"]

    # Filter for prevalence 50%
    if "prevalence" in full_standard_df.columns:
        # Check if prevalence is numeric (0.5) or percentage string ("50%")
        # Assuming numeric based on typical data generation
        prev_df = full_standard_df[
            full_standard_df["prevalence"].astype(str).str.contains("0.5|50")
        ]
    else:
        print("Warning: 'prevalence' column not found. Using all data.")
        prev_df = full_standard_df

    # Get DGPs and Alphas for faceting
    if not prev_df.empty:
        dgps = sorted(prev_df["dist"].unique())
        alphas = sorted(
            prev_df["nominal_alpha"].unique()
        )  # Use nominal_alpha for consistency

        # Create grid
        fig, axes = plt.subplots(
            len(dgps),
            len(alphas),
            figsize=(6 * len(alphas), 5 * len(dgps)),
            squeeze=False,
        )

        for i, dist in enumerate(dgps):
            for j, alpha in enumerate(alphas):
                ax = axes[i, j]

                # Filter data
                subset_df = prev_df[
                    (prev_df["dist"] == dist) & (prev_df["nominal_alpha"] == alpha)
                ]

                if not subset_df.empty:
                    # Plot Pareto frontier
                    viz_s.plot_pareto_frontier(
                        subset_df,
                        mode="mean",
                        nominal_alpha=alpha,
                        ax=ax,
                        title=f"{dist} (alpha={alpha})",
                        show_legend=False,
                    )

        # Add shared legend
        handles, labels = [], []
        for ax in axes.flat:
            if ax.get_legend_handles_labels()[0]:
                handles, labels = ax.get_legend_handles_labels()
                break
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.05),
                ncol=min(len(handles), 6),
                bbox_transform=fig.transFigure,
            )
            plt.subplots_adjust(bottom=0.20)

        plt.tight_layout()
        plt.show()
        save_figure(f"pareto_overview")
    else:
        print("No data found for prevalence=50%")

# %% [markdown]
# #### Pareto Frontier (n_path) by Subset
#
# One panel for each method subset ("core", "envelope", "HT", "reference").

# %%
for subset_name in ["core", "envelope", "HT", "reference"]:
    print(f"Generating n_path Pareto frontier for subset: {subset_name}")

    # Re-using the structure above but filtering by subset
    if not prev_df.empty:
        fig, axes = plt.subplots(
            len(dgps),
            len(alphas),
            figsize=(6 * len(alphas), 5 * len(dgps)),
            squeeze=False,
        )
        fig.suptitle(f"Subset: {subset_name} (n_path mode)", fontsize=16, y=1.02)

        for i, dist in enumerate(dgps):
            for j, alpha in enumerate(alphas):
                ax = axes[i, j]

                # Filter data
                subset_df = prev_df[
                    (prev_df["dist"] == dist) & (prev_df["nominal_alpha"] == alpha)
                ]
                subset_df = filter_by_subset(subset_df, subset_name)

                if not subset_df.empty:
                    viz_s.plot_pareto_frontier(
                        subset_df,
                        mode="n_path",
                        nominal_alpha=alpha,
                        ax=ax,
                        title=f"{dist} (alpha={alpha})",
                        show_legend=False,
                    )

        # Add shared legend
        handles, labels = [], []
        for ax in axes.flat:
            if ax.get_legend_handles_labels()[0]:
                handles, labels = ax.get_legend_handles_labels()
                break
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=min(len(handles), 6),
                bbox_transform=fig.transFigure,
            )
            plt.subplots_adjust(bottom=0.15)
        plt.tight_layout()
        plt.show()

# %% [markdown]
# #### Coverage by N_total by Subset
#
# One panel for each method subset. Faceted by DGP and Alpha.

# %%
for subset_name in ["core", "envelope", "HT", "reference"]:
    print(f"Generating Coverage by N_total for subset: {subset_name}")

    if not prev_df.empty:
        fig, axes = plt.subplots(
            len(dgps),
            len(alphas),
            figsize=(6 * len(alphas), 5 * len(dgps)),
            squeeze=False,
        )
        fig.suptitle(f"Subset: {subset_name} (Coverage vs N)", fontsize=16, y=1.02)

        for i, dist in enumerate(dgps):
            for j, alpha in enumerate(alphas):
                ax = axes[i, j]

                # Filter data
                subset_df = prev_df[
                    (prev_df["dist"] == dist) & (prev_df["nominal_alpha"] == alpha)
                ]
                subset_df = filter_by_subset(subset_df, subset_name)

                if not subset_df.empty:
                    viz_s.plot_coverage_by_n_total(
                        subset_df,
                        nominal_alpha=alpha,
                        ax=ax,
                        title=f"{dist} (alpha={alpha})",
                        show_legend=False,
                    )

        # Add shared legend
        handles, labels = [], []
        for ax in axes.flat:
            if ax.get_legend_handles_labels()[0]:
                handles, labels = ax.get_legend_handles_labels()
                break
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0),
                ncol=min(len(handles), 6),
                bbox_transform=fig.transFigure,
            )
            plt.subplots_adjust(bottom=0.15)
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ### Per-distribution Panels
#
# Detailed analysis for each data generating process.

# %%
# Collect all curve dataframes
all_curve_dfs = []
for key, df in dfs.items():
    if key.endswith("_curve"):
        alpha = key.split("_")[1]
        df_copy = df.copy()
        df_copy["alpha"] = alpha
        all_curve_dfs.append(df_copy)

if all_curve_dfs:
    full_curve_df = pd.concat(all_curve_dfs, ignore_index=True)
    if "dgp_type" in full_curve_df.columns and "dist" not in full_curve_df.columns:
        full_curve_df["dist"] = full_curve_df["dgp_type"]
else:
    print("Warning: No curve data found.")
    full_curve_df = pd.DataFrame()


# %%
# Function to create per-distribution panel
def create_distribution_panel(dist, subset_name, standard_df, curve_df):
    """Create a 6x2 panel of plots for a specific distribution and subset."""

    # Setup figure
    fig, axes = plt.subplots(6, 2, figsize=(16, 30))
    fig.suptitle(f"Distribution: {dist} | Subset: {subset_name}", fontsize=20, y=1.00)

    # Common filters
    alpha_05 = 0.05
    alpha_50 = 0.5

    # Filter data for this distribution
    dist_std = standard_df[standard_df["dist"] == dist]
    dist_curve = curve_df[curve_df["dist"] == dist]

    # Filter by subset
    dist_std = filter_by_subset(dist_std, subset_name)
    dist_curve = filter_by_subset(dist_curve, subset_name)

    if dist_std.empty:
        print(f"No data for {dist} in subset {subset_name}")
        plt.close(fig)
        return

    # Helper for prevalence/n_total filtering
    def get_std(alpha, prev=None, n=None):
        d = dist_std[dist_std["nominal_alpha"] == alpha]
        if prev is not None:
            d = d[d["prevalence"].astype(str).str.contains(f"{prev}|{prev * 100:.0f}")]
        if n is not None:
            d = d[d["n_total"] == n]
        return d

    def get_curve(alpha, prev=None, n=None):
        d = dist_curve[dist_curve["nominal_alpha"] == alpha]
        if prev is not None:
            d = d[d["prevalence"].astype(str).str.contains(f"{prev}|{prev * 100:.0f}")]
        if n is not None:
            d = d[d["n_total"] == n]
        return d

    # Get min/max n_total
    n_totals = sorted(dist_std["n_total"].unique())
    min_n = n_totals[0] if n_totals else 0
    max_n = n_totals[-1] if n_totals else 0

    # 1. Pareto Frontier (Prev=50%, Alpha=0.05)
    ax = axes[0, 0]
    d = get_std(alpha_05, prev=0.5)
    if not d.empty:
        viz_s.plot_pareto_frontier(
            d,
            mode="mean",
            nominal_alpha=alpha_05,
            ax=ax,
            title="Pareto (α=0.05)",
            show_legend=False,
        )
    else:
        ax.text(0.5, 0.5, "No Data", ha="center")

    # 2. Pareto Frontier (Prev=50%, Alpha=0.5)
    ax = axes[0, 1]
    d = get_std(alpha_50, prev=0.5)
    if not d.empty:
        viz_s.plot_pareto_frontier(
            d,
            mode="mean",
            nominal_alpha=alpha_50,
            ax=ax,
            title="Pareto (α=0.5)",
            show_legend=False,
        )
    else:
        ax.text(0.5, 0.5, "No Data", ha="center")

    # 3. Coverage by N (Prev=50%, Alpha=0.05)
    ax = axes[1, 0]
    d = get_std(alpha_05, prev=0.5)
    if not d.empty:
        viz_s.plot_coverage_by_n_total(
            d,
            nominal_alpha=alpha_05,
            ax=ax,
            title="Coverage vs N (α=0.05)",
            show_legend=False,
        )
    else:
        ax.text(0.5, 0.5, "No Data", ha="center")

    # 4. Coverage by Prevalence (N=1000, Alpha=0.05)
    ax = axes[1, 1]
    d = get_std(alpha_05, n=1000)
    if not d.empty:
        viz_s.plot_coverage_by_prevalence(
            d,
            nominal_alpha=alpha_05,
            ax=ax,
            title="Coverage vs Prev (n=1k, α=0.05)",
            show_legend=False,
        )
    else:
        ax.text(0.5, 0.5, "No Data (n=1k)", ha="center")

    # 5. Pareto Frontier Low N (Prev=50%, Alpha=0.05)
    ax = axes[2, 0]
    d = get_std(alpha_05, prev=0.5, n=min_n)
    if not d.empty:
        viz_s.plot_pareto_frontier(
            d,
            mode="mean",
            nominal_alpha=alpha_05,
            ax=ax,
            title=f"Pareto (n={min_n}, α=0.05)",
            show_legend=False,
        )
    else:
        ax.text(0.5, 0.5, "No Data", ha="center")

    # 6. Pareto Frontier High N (Prev=50%, Alpha=0.05)
    ax = axes[2, 1]
    d = get_std(alpha_05, prev=0.5, n=max_n)
    if not d.empty:
        viz_s.plot_pareto_frontier(
            d,
            mode="mean",
            nominal_alpha=alpha_05,
            ax=ax,
            title=f"Pareto (n={max_n}, α=0.05)",
            show_legend=False,
        )
    else:
        ax.text(0.5, 0.5, "No Data", ha="center")

    # 7. Regionwise Pareto (Prev=50%, Alpha=0.05) - All N
    ax = axes[3, 0]
    d = get_curve(alpha_05, prev=0.5)
    if not d.empty:
        viz_c.plot_regionwise_pareto_frontier(
            d, ax=ax, title="Region Pareto (All N)", show_legend=False
        )
    else:
        ax.text(0.5, 0.5, "No Data", ha="center")

    # 8. Regionwise Pareto High N (Prev=50%, Alpha=0.05)
    ax = axes[3, 1]
    d = get_curve(alpha_05, prev=0.5, n=max_n)
    if not d.empty:
        viz_c.plot_regionwise_pareto_frontier(
            d, ax=ax, title=f"Region Pareto (n={max_n})", show_legend=False
        )
    else:
        ax.text(0.5, 0.5, "No Data", ha="center")

    # 9. Coverage by Region (Prev=50%, Alpha=0.05) - All N
    ax = axes[4, 0]
    d = get_curve(alpha_05, prev=0.5)
    if not d.empty:
        viz_c.plot_coverage_by_region(
            d,
            nominal_alpha=alpha_05,
            ax=ax,
            title="Coverage by Region (All N)",
            show_legend=False,
        )
    else:
        ax.text(0.5, 0.5, "No Data", ha="center")

    # 10. Coverage by Region High N (Prev=50%, Alpha=0.05)
    ax = axes[4, 1]
    d = get_curve(alpha_05, prev=0.5, n=max_n)
    if not d.empty:
        viz_c.plot_coverage_by_region(
            d,
            nominal_alpha=alpha_05,
            ax=ax,
            title=f"Coverage by Region (n={max_n})",
            show_legend=False,
        )
    else:
        ax.text(0.5, 0.5, "No Data", ha="center")

    # 11. Violation Proximity (Prev=50%, Alpha=0.05)
    ax = axes[5, 0]
    d = get_std(alpha_05, prev=0.5)
    if not d.empty:
        viz_s.plot_violation_proximity(
            d, ax=ax, title="Violation Proximity (α=0.05)", show_legend=False
        )
    else:
        ax.text(0.5, 0.5, "No Data", ha="center")

    # 12. Violation Direction (Prev=50%, Alpha=0.5)
    ax = axes[5, 1]
    d = get_std(alpha_50, prev=0.5)
    if not d.empty:
        viz_s.plot_violation_direction(
            d, ax=ax, title="Violation Direction (α=0.5)", show_legend=False
        )
    else:
        ax.text(0.5, 0.5, "No Data", ha="center")

    # Add shared legend
    # Collect unique methods from this distribution data
    methods = sorted(dist_std["model"].unique())
    color_dict = viz_s.get_method_colors_dict(methods)

    # Method handles (colors)
    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=color_dict[m],
            linestyle="-",
            markersize=6,
            markeredgecolor="white",
            label=m,
        )
        for m in methods
    ]

    # Region handles (shapes)
    marker_dict = viz_c.get_fpr_region_markers()
    regions = sorted(marker_dict.keys(), key=lambda r: int(r.split("-")[0]))
    region_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_dict[r],
            color="gray",
            linestyle="None",
            markersize=6,
            markeredgecolor="white",
            label=f"FPR {r}%",
        )
        for r in regions
    ]

    # Combine
    all_handles = method_handles + [Line2D([], [], linestyle="None")] + region_handles
    all_labels = [h.get_label() for h in all_handles]

    fig.legend(
        all_handles,
        all_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=min(len(methods) + 2 + len(regions), 6),
        bbox_transform=fig.transFigure,
    )
    plt.subplots_adjust(bottom=0.15)

    plt.tight_layout()
    plt.show()


# Run for each distribution and subset
# %%
# CORE
for dist in dgps:
    create_distribution_panel(dist, "core", full_standard_df, full_curve_df)

# %%
# ENVELOPE
for dist in dgps:
    create_distribution_panel(dist, "envelope", full_standard_df, full_curve_df)

# %%
# HT
for dist in dgps:
    create_distribution_panel(dist, "HT", full_standard_df, full_curve_df)

# %%
# REFERENCE
for dist in dgps:
    create_distribution_panel(dist, "reference", full_standard_df, full_curve_df)

# %% [markdown]
# ### Per-distribution Panels by N_total
#
# Detailed analysis for each sample size.


# %%
def create_n_total_panel(dist, n, standard_df, curve_df):
    """Create a 6x2 panel for specific dist and n_total."""

    # Filter for core subset specifically
    subset_name = "core"

    fig, axes = plt.subplots(5, 2, figsize=(16, 30))
    fig.suptitle(f"Dist: {dist} | N: {n} | Subset: {subset_name}", fontsize=20, y=1.00)

    # Filter data
    dist_std = standard_df[
        (standard_df["dist"] == dist) & (standard_df["n_total"] == n)
    ]
    dist_curve = curve_df[(curve_df["dist"] == dist) & (curve_df["n_total"] == n)]

    # Filter prevalence 50%
    dist_std = dist_std[dist_std["prevalence"].astype(str).str.contains("0.5|50")]
    dist_curve = dist_curve[dist_curve["prevalence"].astype(str).str.contains("0.5|50")]

    # Filter subset
    dist_std = filter_by_subset(dist_std, subset_name)
    dist_curve = filter_by_subset(dist_curve, subset_name)

    if dist_std.empty:
        plt.close(fig)
        return

    alpha_05 = 0.05
    alpha_50 = 0.5

    def get_std(alpha):
        return dist_std[dist_std["nominal_alpha"] == alpha]

    def get_curve(alpha):
        return dist_curve[dist_curve["nominal_alpha"] == alpha]

    # 1. Pareto (alpha=0.05)
    ax = axes[0, 0]
    d = get_std(alpha_05)
    if not d.empty:
        viz_s.plot_pareto_frontier(
            d,
            mode="mean",
            nominal_alpha=alpha_05,
            ax=ax,
            title="Pareto (α=0.05)",
            show_legend=False,
        )

    # 2. Pareto (alpha=0.5)
    ax = axes[0, 1]
    d = get_std(alpha_50)
    if not d.empty:
        viz_s.plot_pareto_frontier(
            d,
            mode="mean",
            nominal_alpha=alpha_50,
            ax=ax,
            title="Pareto (α=0.5)",
            show_legend=False,
        )

    # 3-4. Regionwise Pareto
    ax = axes[1, 0]
    d = get_curve(alpha_05)
    if not d.empty:
        viz_c.plot_regionwise_pareto_frontier(
            d, ax=ax, title="Region Pareto (α=0.05)", show_legend=False
        )

    ax = axes[1, 1]
    d = get_curve(alpha_50)
    if not d.empty:
        viz_c.plot_regionwise_pareto_frontier(
            d, ax=ax, title="Region Pareto (α=0.5)", show_legend=False
        )

    # 5-6. Coverage by Region
    ax = axes[2, 0]
    d = get_curve(alpha_05)
    if not d.empty:
        viz_c.plot_coverage_by_region(
            d,
            nominal_alpha=alpha_05,
            ax=ax,
            title="Coverage by Region (α=0.05)",
            show_legend=False,
        )

    ax = axes[2, 1]
    d = get_curve(alpha_50)
    if not d.empty:
        viz_c.plot_coverage_by_region(
            d,
            nominal_alpha=alpha_50,
            ax=ax,
            title="Coverage by Region (α=0.5)",
            show_legend=False,
        )

    # 7-8. Violation Proximity
    ax = axes[3, 0]
    d = get_std(alpha_05)
    if not d.empty:
        viz_s.plot_violation_proximity(
            d, ax=ax, title="Violation Proximity (α=0.05)", show_legend=False
        )

    ax = axes[3, 1]
    d = get_std(alpha_50)
    if not d.empty:
        viz_s.plot_violation_proximity(
            d, ax=ax, title="Violation Proximity (α=0.5)", show_legend=False
        )

    # 9-10. Violation Direction
    ax = axes[4, 0]
    d = get_std(alpha_05)
    if not d.empty:
        viz_s.plot_violation_direction(
            d, ax=ax, title="Violation Direction (α=0.05)", show_legend=False
        )

    ax = axes[4, 1]
    d = get_std(alpha_50)
    if not d.empty:
        viz_s.plot_violation_direction(
            d, ax=ax, title="Violation Direction (α=0.5)", show_legend=False
        )

    # Add shared legend
    # Collect unique methods from this distribution data
    methods = sorted(dist_std["model"].unique())
    color_dict = viz_s.get_method_colors_dict(methods)

    # Method handles (colors)
    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=color_dict[m],
            linestyle="-",
            markersize=6,
            markeredgecolor="white",
            label=m,
        )
        for m in methods
    ]

    # Region handles (shapes)
    marker_dict = viz_c.get_fpr_region_markers()
    regions = sorted(marker_dict.keys(), key=lambda r: int(r.split("-")[0]))
    region_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_dict[r],
            color="gray",
            linestyle="None",
            markersize=6,
            markeredgecolor="white",
            label=f"FPR {r}%",
        )
        for r in regions
    ]

    # Combine
    all_handles = method_handles + [Line2D([], [], linestyle="None")] + region_handles
    all_labels = [h.get_label() for h in all_handles]

    fig.legend(
        all_handles,
        all_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=min(len(methods) + 2 + len(regions), 6),
        bbox_transform=fig.transFigure,
    )
    plt.subplots_adjust(bottom=0.15)

    plt.tight_layout()
    plt.show()
    save_figure(f"n_panel_{dist}_{n}")


# Run per N
if not prev_df.empty:
    for dist in dgps:
        # Get Ns for this dist
        dist_ns = sorted(
            full_standard_df[full_standard_df["dist"] == dist]["n_total"].unique()
        )
        for n in dist_ns:
            try:
                create_n_total_panel(dist, n, full_standard_df, full_curve_df)
            except Exception as e:
                print(f"Error creating N-panel for {dist} - n={n}: {e}")

# %% [markdown]
# ### Per-method Panels
#
# Violation gradient, Coverage by N, Coverage by Prevalence.

# %%
# 1. Violation Location Gradient (Prev=50%)
# Grid: DGP (rows) x Alpha (cols). Panel for EACH Method.

if not full_trial_df.empty:
    for method in sorted(full_standard_df["model"].unique()):
        print(f"Generating Violation Gradient for method: {method}")

        # Determine grid size based on AVAILABLE DGPs and Alphas
        # Use filtered lists to avoid empty subplots if data missing
        current_dgps = sorted(
            full_trial_df[full_trial_df["method"] == method]["dgp_type"].unique()
        )
        current_alphas = sorted(
            full_trial_df[full_trial_df["method"] == method]["alpha"].unique()
        )

        if not current_dgps:
            continue

        fig, axes = plt.subplots(
            len(current_dgps),
            len(current_alphas),
            figsize=(6 * len(current_alphas), 4 * len(current_dgps)),
            squeeze=False,
        )
        fig.suptitle(
            f"Method: {method} - Violation Gradient (by N_total)", fontsize=16, y=1.00
        )

        for i, dist in enumerate(current_dgps):
            for j, alpha in enumerate(current_alphas):
                ax = axes[i, j]

                # Filter trial data: Dist, Method, Alpha
                d = full_trial_df[
                    (full_trial_df["dgp_type"] == dist)
                    & (full_trial_df["method"] == method)
                    & (full_trial_df["alpha"] == alpha)
                ]
                # Filter prev=0.5
                if "prevalence" in d.columns:
                    d = d[d["prevalence"] == 0.5]

                if not d.empty:
                    try:
                        viz_i.plot_violation_location_gradient(
                            d,
                            bin_col="n_total",
                            n_bins=None,
                            ax=ax,
                            title=f"{dist} (alpha={alpha})",
                            show_legend=False,
                        )
                    except Exception as e:
                        print(
                            f"Error plotting violation gradient for {method}/{dist}/{alpha}: {e}"
                        )
                        ax.text(0.5, 0.5, f"Error: {e}", ha="center")

        # Add shared legend for bins
        handles, labels = [], []
        for ax in axes.flat:
            if ax.get_legend_handles_labels()[0]:
                handles, labels = ax.get_legend_handles_labels()
                break
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0),
                ncol=min(len(handles), 6),
                bbox_transform=fig.transFigure,
            )
            plt.subplots_adjust(bottom=0.15)

        plt.tight_layout()
        plt.show()
        save_figure(f"method_violation_gradient_{method}")

# %%
# 2. Coverage by N_total (Prev=50%)
# Grid: DGP (rows) x Alpha (cols). Panel for EACH Method.

for method in sorted(full_standard_df["model"].unique()):
    print(f"Generating Coverage vs N for method: {method}")

    current_dgps = sorted(
        full_standard_df[full_standard_df["model"] == method]["dist"].unique()
    )
    current_alphas = sorted(
        full_standard_df[full_standard_df["model"] == method]["nominal_alpha"].unique()
    )

    if not current_dgps:
        continue

    fig, axes = plt.subplots(
        len(current_dgps),
        len(current_alphas),
        figsize=(6 * len(current_alphas), 4 * len(current_dgps)),
        squeeze=False,
    )
    fig.suptitle(f"Method: {method} - Coverage vs N", fontsize=16, y=1.00)

    for i, dist in enumerate(current_dgps):
        for j, alpha in enumerate(current_alphas):
            ax = axes[i, j]

            d = full_standard_df[
                (full_standard_df["dist"] == dist)
                & (full_standard_df["model"] == method)
                & (full_standard_df["nominal_alpha"] == alpha)
            ]
            d = d[d["prevalence"].astype(str).str.contains("0.5|50")]

            if not d.empty:
                viz_s.plot_coverage_by_n_total(
                    d,
                    nominal_alpha=alpha,
                    ax=ax,
                    title=f"{dist} (alpha={alpha})",
                    show_legend=False,
                )

    plt.tight_layout()
    plt.show()
    save_figure(f"method_coverage_n_{method}")

# %%
# 3. Coverage by Prevalence (N=1000)
# Grid: DGP (rows) x Alpha (cols). Panel for EACH Method.

for method in sorted(full_standard_df["model"].unique()):
    print(f"Generating Coverage vs Prev for method: {method}")

    current_dgps = sorted(
        full_standard_df[full_standard_df["model"] == method]["dist"].unique()
    )
    current_alphas = sorted(
        full_standard_df[full_standard_df["model"] == method]["nominal_alpha"].unique()
    )

    if not current_dgps:
        continue

    fig, axes = plt.subplots(
        len(current_dgps),
        len(current_alphas),
        figsize=(6 * len(current_alphas), 4 * len(current_dgps)),
        squeeze=False,
    )
    fig.suptitle(f"Method: {method} - Coverage vs Prev (N=1k)", fontsize=16, y=1.00)

    for i, dist in enumerate(current_dgps):
        for j, alpha in enumerate(current_alphas):
            ax = axes[i, j]

            d = full_standard_df[
                (full_standard_df["dist"] == dist)
                & (full_standard_df["model"] == method)
                & (full_standard_df["nominal_alpha"] == alpha)
                & (full_standard_df["n_total"] == 1000)
            ]

            if not d.empty:
                viz_s.plot_coverage_by_prevalence(
                    d,
                    nominal_alpha=alpha,
                    ax=ax,
                    title=f"{dist} (alpha={alpha})",
                    show_legend=False,
                )

    plt.tight_layout()
    plt.show()
    save_figure(f"method_coverage_prev_{method}")

# %% [markdown]
# ### Data Trend Panels
#
# Performance as a function of data properties.

# %%
# 1. Violation Location Gradient by lhs_auc (Prev=50%, Alpha=0.05)
n_totals_trend = [300, 1000, 10000]
methods = sorted(full_trial_df["method"].unique())

for method in methods:
    print(f"Data Trend: Violation Gradient by AUC for {method}")

    fig, axes = plt.subplots(
        len(dgps), len(n_totals_trend), figsize=(12, 4 * len(dgps)), squeeze=False
    )
    fig.suptitle(
        f"Violation Gradient by AUC: {method} (Alpha=0.05, Prev=50%)",
        fontsize=16,
        y=1.02,
    )

    for i, dist in enumerate(dgps):
        for j, n in enumerate(n_totals_trend):
            ax = axes[i, j]

            # Prepare data
            # Use full_trial_df as it contains lhs_auc and violation columns
            # Note: full_trial_df uses 'alpha' instead of 'nominal_alpha'
            d = full_trial_df[
                (full_trial_df["dist"] == dist)
                & (full_trial_df["method"] == method)
                & (full_trial_df["n_total"] == n)
                & (full_trial_df["alpha"] == 0.05)
            ]

            # Filter prevalence if column exists
            if "prevalence" in d.columns:
                d = d[d["prevalence"] == 0.5]

            if not d.empty:
                viz_i.plot_violation_location_gradient(
                    d, bin_col="lhs_auc", n_bins=5, ax=ax, title=f"{dist} (n={n})"
                )
    plt.tight_layout()
    plt.show()
    save_figure(f"trend_violation_auc_{method}")


# %%

# 2. Data Property Lines
# Filter by core data subset (each plot will have lines for each of the core methods)
# One figure per Property. Grid: DGP (rows) x N_total (cols)

lhs_properties = ["lhs_auc", "lhs_sigma", "lhs_sigma_ratio"]
n_totals_trend = [300, 1000, 10000]
core_methods = SUBSETS["core"]

for prop in lhs_properties:
    print(f"Generating Data Trend Panel for property: {prop}")

    # Check if property exists in dataframe
    if prop not in full_standard_df.columns:
        print(f"Property {prop} not found in dataframe. Skipping.")
        continue

    # Create figure
    fig, axes = plt.subplots(
        len(dgps),
        len(n_totals_trend),
        figsize=(6 * len(n_totals_trend), 4 * len(dgps)),
        squeeze=False,
    )
    fig.suptitle(f"Coverage vs {prop} (Core Methods)", fontsize=16, y=1.00)

    has_data = False
    for i, dist in enumerate(dgps):
        for j, n in enumerate(n_totals_trend):
            ax = axes[i, j]

            # Filter: Dist, N, Core subset, Alpha=0.05 (usually standard)
            # We focus on alpha=0.05 for trends unless specified otherwise
            d = full_standard_df[
                (full_standard_df["dist"] == dist)
                & (full_standard_df["n_total"] == n)
                & (full_standard_df["nominal_alpha"] == 0.05)
            ]
            d = d[d["prevalence"].astype(str).str.contains("0.5|50")]

            # Filter for core methods
            d = filter_by_subset(d, "core")

            # Ensure property is valid
            d = d[d[prop].notna()]

            if not d.empty:
                has_data = True
                try:
                    viz_i.plot_data_property_lines(
                        d,
                        x_col=prop,
                        y_col="covers_entirely",
                        methods=core_methods,  # Explicitly list to ensure consistent colors/ordering
                        ax=ax,
                        title=f"{dist} (n={n})",
                        nominal_coverage=0.95,
                        show_legend=(
                            i == 0 and j == 0
                        ),  # Legend only on first plot to save space
                    )
                except Exception as e:
                    print(f"Error plotting {prop} for {dist}/{n}: {e}")
                    ax.text(0.5, 0.5, f"Error: {e}", ha="center")

    if has_data:
        plt.tight_layout()
        plt.show()
        save_figure(f"trend_property_{prop}")
    else:
        plt.close(fig)

# %%
