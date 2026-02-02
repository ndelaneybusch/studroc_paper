"""Validate updated Wilson floor method on prior simulation violations.

Re-evaluates ROC confidence bands on cases where the OLD envelope_wilson method
had coverage violations, using the UPDATED method with adaptive tail floor and
Šidák correction.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from studroc_paper.datagen.true_rocs import (
    DGP,
    make_beta_opposing_skew_dgp,
    make_bimodal_negative_dgp,
    make_gamma_dgp,
    make_heteroskedastic_gaussian_dgp,
    make_logitnormal_dgp,
    make_student_t_dgp,
    make_weibull_dgp,
)
from studroc_paper.methods.envelope_boot import envelope_bootstrap_band
from studroc_paper.sampling.bootstrap_grid import generate_bootstrap_grid

FPR_GRID = np.linspace(0, 1, 10001)
N_BOOTSTRAP = 4000
ENVELOPE_PARAMS = {
    "alpha": 0.05,
    "boundary_method": "wilson",
    "retention_method": "ks",
    "use_logit": False,
    "tpr_method": "empirical",
}


def reconstruct_dgp(row: pd.Series) -> DGP:
    """Map row to DGP factory function based on dgp_type."""
    dgp_type = row["dgp_type"]

    if dgp_type == "gamma":
        return make_gamma_dgp(
            neg_shape=row["dgp_neg_shape"],
            pos_shape=row["dgp_pos_shape"],
            neg_scale=row["dgp_neg_scale"],
            pos_scale=row["dgp_pos_scale"],
        )
    elif dgp_type == "beta_opposing":
        return make_beta_opposing_skew_dgp(
            alpha=row["dgp_alpha"],
            beta=row["dgp_beta"],
        )
    elif dgp_type == "logitnormal":
        return make_logitnormal_dgp(
            neg_mu=row["dgp_neg_mu"],
            pos_mu=row["dgp_pos_mu"],
            sigma=row["dgp_sigma"],
        )
    elif dgp_type == "student_t":
        return make_student_t_dgp(
            df=row["dgp_df"],
            delta_loc=row["dgp_delta_loc"],
            scale=row["dgp_scale"],
        )
    elif dgp_type == "bimodal_negative":
        neg_means = row["dgp_neg_means"]
        neg_weights = row["dgp_neg_weights"]
        if isinstance(neg_means, np.ndarray):
            neg_means = tuple(neg_means.tolist())
        if isinstance(neg_weights, np.ndarray):
            neg_weights = tuple(neg_weights.tolist())
        return make_bimodal_negative_dgp(
            neg_means=neg_means,
            neg_weights=neg_weights,
            pos_mean=row["dgp_pos_mean"],
        )
    elif dgp_type == "hetero_gaussian":
        return make_heteroskedastic_gaussian_dgp(
            delta_mu=row["dgp_delta_mu"],
            sigma_neg=row["dgp_sigma_neg"],
            sigma_pos=row["dgp_sigma_pos"],
        )
    elif dgp_type == "weibull":
        return make_weibull_dgp(
            neg_shape=row["dgp_neg_shape"],
            pos_shape=row["dgp_pos_shape"],
            neg_scale=row["dgp_neg_scale"],
            pos_scale=row["dgp_pos_scale"],
        )
    else:
        raise ValueError(f"Unknown dgp_type: {dgp_type}")


def evaluate_coverage(
    true_tpr: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    fpr_grid: np.ndarray,
) -> dict:
    """Find violation points and distances."""
    violation_below = true_tpr < lower
    violation_above = true_tpr > upper
    violated = violation_below | violation_above

    violation_distances = np.zeros_like(true_tpr)
    violation_distances[violation_below] = lower[violation_below] - true_tpr[violation_below]
    violation_distances[violation_above] = true_tpr[violation_above] - upper[violation_above]

    violation_fprs = fpr_grid[violated]
    violation_dists = violation_distances[violated]

    return {
        "new_violated": violated.any(),
        "new_max_violation": float(violation_distances.max()) if violated.any() else 0.0,
        "pct_points_violated": float(violated.mean() * 100),
        "median_violation_fpr": float(np.median(violation_fprs)) if len(violation_fprs) > 0 else np.nan,
        "violation_fprs": violation_fprs.tolist(),
        "violation_distances": violation_dists.tolist(),
    }


def compute_tail_mask(
    fpr_grid: np.ndarray,
    empirical_tpr: np.ndarray,
    n_neg: int,
    n_pos: int,
) -> np.ndarray:
    """Identify tail regions where Wilson floor is active."""
    k = (fpr_grid * n_neg).round().astype(int)
    m = (empirical_tpr * n_pos).round().astype(int)

    k_min_lower, k_min_upper = 15, 10
    m_min = 10

    lower_tail = ((k < k_min_lower) | (m < m_min)) & (fpr_grid <= 0.5)
    upper_tail = (((n_neg - k) < k_min_upper) | ((n_pos - m) < m_min)) & (fpr_grid >= 0.5)

    return lower_tail | upper_tail


def create_visualization(
    rows_data: list[dict],
    output_path: Path,
    dgp_type: str,
) -> None:
    """Create 3x3 grid of ROC plots with violations marked."""
    n_plots = min(9, len(rows_data))
    if n_plots == 0:
        return

    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, data in enumerate(rows_data[:n_plots]):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row_idx, col_idx]

        fpr = data["fpr_grid"]
        true_tpr = data["true_tpr"]
        empirical_tpr = data["empirical_tpr"]
        lower = data["lower"]
        upper = data["upper"]
        tail_mask = data["tail_mask"]

        ax.fill_between(
            fpr[tail_mask],
            0,
            1,
            alpha=0.1,
            color="orange",
            label="Tail region",
        )

        ax.fill_between(fpr, lower, upper, alpha=0.3, color="blue", label="CI band")
        ax.plot(fpr, true_tpr, "g-", linewidth=1.5, label="True ROC")
        ax.plot(fpr, empirical_tpr, "b--", linewidth=1, alpha=0.7, label="Empirical ROC")

        violation_mask = (true_tpr < lower) | (true_tpr > upper)
        if violation_mask.any():
            ax.scatter(
                fpr[violation_mask],
                true_tpr[violation_mask],
                c="red",
                s=5,
                zorder=5,
                label="Violations",
            )

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"lhs_idx={data['lhs_idx']}, old_viol={data['old_max_violation']:.4f}")
        ax.set_aspect("equal")

        if idx == 0:
            ax.legend(loc="lower right", fontsize=8)

    for idx in range(n_plots, n_rows * n_cols):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        axes[row_idx, col_idx].axis("off")

    fig.suptitle(f"{dgp_type}: Wilson Update Validation", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_violated_row(
    row: pd.Series,
    fpr_grid: np.ndarray,
    device: torch.device,
) -> dict | None:
    """Process a single violated row and return results."""
    dgp = reconstruct_dgp(row)
    n_pos = int(row["n_pos"])
    n_neg = int(row["n_neg"])
    lhs_idx = int(row["lhs_idx"])

    rng = np.random.default_rng(seed=lhs_idx)

    scores_pos, scores_neg = dgp.sample(n_pos=n_pos, n_neg=n_neg, rng=rng)

    y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    y_score = np.concatenate([scores_pos, scores_neg])

    y_true_t = torch.from_numpy(y_true).to(device)
    y_score_t = torch.from_numpy(y_score).float().to(device)
    fpr_grid_t = torch.from_numpy(fpr_grid).float().to(device)

    boot_tpr_matrix = generate_bootstrap_grid(
        y_true=y_true_t,
        y_score=y_score_t,
        B=N_BOOTSTRAP,
        grid=fpr_grid_t,
        device=device,
        tpr_method="empirical",
    )

    fpr_out, lower, upper = envelope_bootstrap_band(
        boot_tpr_matrix=boot_tpr_matrix.cpu().numpy(),
        fpr_grid=fpr_grid,
        y_true=y_true,
        y_score=y_score,
        **ENVELOPE_PARAMS,
    )

    true_tpr = dgp.get_true_roc(fpr_grid)

    coverage_results = evaluate_coverage(
        true_tpr=true_tpr,
        lower=lower,
        upper=upper,
        fpr_grid=fpr_grid,
    )

    neg_scores_sorted = np.sort(scores_neg)[::-1]
    k_indices = np.floor(fpr_grid * n_neg).astype(int)
    k_indices = np.clip(k_indices, 0, n_neg - 1)
    k_indices[-1] = n_neg - 1
    thresholds = np.zeros_like(fpr_grid)
    thresholds[:-1] = neg_scores_sorted[k_indices[:-1]]
    thresholds[-1] = neg_scores_sorted[-1] - 1

    empirical_tpr = np.array([(scores_pos > t).mean() for t in thresholds])

    tail_mask = compute_tail_mask(fpr_grid, empirical_tpr, n_neg, n_pos)

    dgp_params = {}
    for col in row.index:
        if col.startswith("dgp_") and col != "dgp_type":
            val = row[col]
            if isinstance(val, np.ndarray):
                val = val.tolist()
            dgp_params[col] = val

    old_max_violation = max(
        row.get("max_violation_above", 0.0),
        row.get("max_violation_below", 0.0),
    )

    result = {
        "distribution": row["dgp_type"],
        "dgp_params": dgp_params,
        "lhs_idx": lhs_idx,
        "old_violated": True,
        "old_max_violation": float(old_max_violation),
        **coverage_results,
        "fpr_grid": fpr_grid,
        "true_tpr": true_tpr,
        "empirical_tpr": empirical_tpr,
        "lower": lower,
        "upper": upper,
        "tail_mask": tail_mask,
    }

    return result


def main() -> None:
    """Run the validation script."""
    results_dir = Path("data/results")
    output_dir = results_dir / "wilson_update_validation"
    output_dir.mkdir(exist_ok=True, parents=True)

    n10000_files = sorted(results_dir.glob("*_n10000_*_individual.feather"))
    print(f"Found {len(n10000_files)} n10000 files to process")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_results = []

    for file_path in n10000_files:
        print(f"\nProcessing: {file_path.name}")

        df = pd.read_feather(file_path)

        violated = df[
            (df["method"] == "envelope_wilson")
            & (df["alpha"] == 0.05)
            & (df["covers_entirely"] == False)  # noqa: E712
        ]

        print(f"  Found {len(violated)} violated rows out of {len(df[df['method'] == 'envelope_wilson'])} envelope_wilson rows")

        if len(violated) == 0:
            continue

        dgp_type = violated["dgp_type"].iloc[0]
        file_results = []
        vis_data = []

        for _, row in tqdm(violated.iterrows(), total=len(violated), desc=f"  {dgp_type}"):
            result = process_violated_row(row, FPR_GRID, device)
            if result is not None:
                vis_data.append(result)
                result_for_df = {
                    k: v
                    for k, v in result.items()
                    if k
                    not in (
                        "fpr_grid",
                        "true_tpr",
                        "empirical_tpr",
                        "lower",
                        "upper",
                        "tail_mask",
                        "violation_fprs",
                        "violation_distances",
                    )
                }
                file_results.append(result_for_df)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_results.extend(file_results)

        if vis_data:
            vis_path = output_dir / f"{dgp_type}_violations.png"
            create_visualization(vis_data, vis_path, dgp_type)
            print(f"  Saved visualization: {vis_path.name}")

        checkpoint_df = pd.DataFrame(all_results)
        checkpoint_path = output_dir / "checkpoint.feather"
        checkpoint_df.to_feather(checkpoint_path)
        print(f"  Checkpoint saved: {len(all_results)} total results")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if all_results:
        results_df = pd.DataFrame(all_results)
        print(f"\nTotal violated rows processed: {len(results_df)}")
        print(f"Still violated with new method: {results_df['new_violated'].sum()} ({results_df['new_violated'].mean() * 100:.1f}%)")
        print(f"\nMean old max violation: {results_df['old_max_violation'].mean():.6f}")
        print(f"Mean new max violation: {results_df['new_max_violation'].mean():.6f}")

        print("\nBreakdown by distribution:")
        for dist in results_df["distribution"].unique():
            dist_df = results_df[results_df["distribution"] == dist]
            still_violated = dist_df["new_violated"].sum()
            print(f"  {dist}: {len(dist_df)} cases, {still_violated} still violated ({still_violated / len(dist_df) * 100:.1f}%)")
    else:
        print("No violated rows found!")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
