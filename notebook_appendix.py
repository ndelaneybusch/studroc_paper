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
# # Appendix: Methods & Simulation Figures
#
# Companion to `notebook_paper.py`. Where the main notebook reports *results*
# (coverage, robustness, tightness, ablations), this appendix makes the
# **ingredients of the simulation** and the **construction of the method**
# legible. Every figure here is *illustrative* — computed fresh from the
# analytic data-generating processes and the method itself on small seeded
# draws (or a modest Monte-Carlo sweep), not from the 2.25M-evaluation results
# run. It therefore does not depend on `data/results/` and runs in a couple of
# minutes on CPU.
#
# **Figures**
#
# - **A1** — DGP atlas: the score distributions that were simulated and the
#   true ROC curves they induce.
# - **A2** — The simulation design space: probit-scale AUC sampling, maximin
#   Latin-hypercube design with Iman–Conover decorrelation, and the nonlinear
#   inversion from (AUC, shape) design coordinates to DGP parameters.
# - **A3** — Anatomy of the studentized bootstrap interior on one dataset:
#   the bootstrap cloud, the studentized tube vs. the projected envelope, and
#   the local standard error with the Wilson variance floor.
# - **A4** — The exact Beta order-statistic law that underpins the low-FPR
#   floor, with a distribution-free Monte-Carlo verification.
# - **A5** — Why one variance model cannot cover both corners: the binomial
#   (Wilson) variance vs. Monte-Carlo truth, and the standardized finite-sample
#   bias of the empirical ROC that vanishes in the interior but not at the
#   moving boundary.
# - **A6** — Lower-band assembly waterfall on one dataset: bare envelope →
#   + Wilson rectangle floor → + Beta floor.
#
# Outputs are written to `figures/paper/` as `figA*` (PNG + PDF + SVG, 400 dpi)
# via the same `save_figure` helper as the paper notebook.
#
# **Regenerating:**
# ```
# uv run python notebook_appendix.py
# # or, for an executed HTML render:
# uv run jupytext --to ipynb notebook_appendix.py
# uv run jupyter nbconvert --to html --execute notebook_appendix.ipynb
# ```

# %%
import sys
import warnings
from pathlib import Path

if str(Path("src").resolve()) not in sys.path:
    sys.path.append(str(Path("src").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats as sps
from scipy.stats import beta as beta_dist
from scipy.stats import norm

from studroc_paper.datagen.roc_to_dgp import map_lhs_to_dgp
from studroc_paper.datagen.true_rocs import (
    make_beta_opposing_skew_dgp,
    make_bimodal_negative_dgp,
    make_gamma_dgp,
    make_heteroskedastic_gaussian_dgp,
    make_logitnormal_dgp,
    make_student_t_dgp,
    make_weibull_dgp,
)
from studroc_paper.methods.envelope_boot import envelope_band_suite
from studroc_paper.methods.method_utils import compute_empirical_roc_from_scores
from studroc_paper.sampling.bootstrap_grid import generate_bootstrap_grid
from studroc_paper.sampling.lhs import iman_conover_transform, maximin_lhs

warnings.filterwarnings("ignore", category=FutureWarning)

FIG_DIR = Path("figures/paper")
FIG_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_style() -> None:
    """Set the global matplotlib style, matching the paper notebook."""
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
# ## Configuration: DGP roster, colors, and shared builders
#
# Visual identity matches the paper notebook: each DGP keeps its family hue,
# the envelope method is vermillion, and nominal references are dashed gray.

# %%
# --- DGP metadata (mirrors notebook_paper.py) --------------------------------
DGP_META: dict[str, dict] = {
    "binormal": dict(label="Binormal", color="#0072B2", family="Gaussian-like"),
    "hetero_gaussian": dict(
        label="Heterosc. Gaussian", color="#56B4E9", family="Gaussian-like"
    ),
    "logitnormal": dict(label="Logit-normal", color="#7BB6DD", family="Gaussian-like"),
    "student_t": dict(label="Student-t", color="#D55E00", family="Heavy-tailed"),
    "gamma": dict(label="Gamma", color="#E69F00", family="Heavy-tailed"),
    "weibull": dict(label="Weibull", color="#CC6677", family="Heavy-tailed"),
    "beta_opposing": dict(
        label="Beta (opposing skew)", color="#009E73", family="Non-standard"
    ),
    "bimodal_negative": dict(
        label="Bimodal negatives", color="#AA4499", family="Non-standard"
    ),
}


def dlabel(d: str) -> str:
    return DGP_META.get(d, {}).get("label", d)


def dcolor(d: str) -> str:
    return DGP_META.get(d, {}).get("color", "#777777")


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

# Representative (interesting, not adversarial) shape parameters per DGP,
# matching notebook_paper.py's example panels.
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

ENVELOPE_COLOR = "#D55E00"
NEG_COLOR = "#888888"
AUC_LEVELS = [0.70, 0.85, 0.95]
AUC_COLORS = {0.70: "#9ECAE1", 0.85: "#3182BD", 0.95: "#08519C"}


def make_params(dgp_type: str, auc: float, shape: dict) -> dict:
    """Map (target AUC, shape) to scalar DGP factory parameters.

    Mirrors the simulation's `map_lhs_to_dgp` call, then scalarizes the
    length-1 arrays it returns (lists/tuples for the mixture DGP are kept).

    Args:
        dgp_type: DGP family key into DGP_FACTORY.
        auc: Target true AUC.
        shape: Family-specific shape parameters (LHS axes other than AUC).

    Returns:
        Keyword arguments ready for the matching ``make_*_dgp`` factory.
    """
    lhs = {"auc": np.array([float(auc)])}
    for key, value in shape.items():
        lhs[key] = np.array([float(value)])
    mapped = map_lhs_to_dgp(dgp_type, lhs)
    out: dict = {}
    for key, value in mapped.items():
        if isinstance(value, np.ndarray) and value.ndim > 0:
            out[key] = float(value[0])
        elif isinstance(value, list):
            out[key] = value[0]
        else:
            out[key] = value
    return out


def make_dgp(dgp_type: str, auc: float, shape: dict):
    """Build a DGP instance and return it with its scalar parameters."""
    params = make_params(dgp_type, auc, shape)
    return DGP_FACTORY[dgp_type](**params), params


def score_densities(
    dgp_type: str, params: dict, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic negative- and positive-class score densities on a grid.

    Args:
        dgp_type: DGP family key.
        params: Scalar DGP parameters from `make_params`.
        x: Evaluation grid for the densities.

    Returns:
        Tuple (pdf_neg, pdf_pos) evaluated on ``x``.
    """
    if dgp_type in ("binormal", "hetero_gaussian"):
        dmu, s_neg, s_pos = params["delta_mu"], params["sigma_neg"], params["sigma_pos"]
        return sps.norm.pdf(x, 0.0, s_neg), sps.norm.pdf(x, dmu, s_pos)
    if dgp_type == "logitnormal":
        mu_n, mu_p, sg = params["neg_mu"], params["pos_mu"], params["sigma"]
        inside = (x > 0) & (x < 1)
        logit = np.zeros_like(x)
        logit[inside] = np.log(x[inside] / (1 - x[inside]))
        jac = np.zeros_like(x)
        jac[inside] = 1.0 / (x[inside] * (1 - x[inside]))
        return sps.norm.pdf(logit, mu_n, sg) * jac, sps.norm.pdf(logit, mu_p, sg) * jac
    if dgp_type == "student_t":
        df, d, s = params["df"], params["delta_loc"], params["scale"]
        return sps.t.pdf(x, df, 0.0, s), sps.t.pdf(x, df, d, s)
    if dgp_type == "gamma":
        return (
            sps.gamma.pdf(x, params["neg_shape"], scale=params["neg_scale"]),
            sps.gamma.pdf(x, params["pos_shape"], scale=params["pos_scale"]),
        )
    if dgp_type == "weibull":
        return (
            sps.weibull_min.pdf(x, params["neg_shape"], scale=params["neg_scale"]),
            sps.weibull_min.pdf(x, params["pos_shape"], scale=params["pos_scale"]),
        )
    if dgp_type == "beta_opposing":
        a, b = params["alpha"], params["beta"]
        return sps.beta.pdf(x, a, b), sps.beta.pdf(x, b, a)
    if dgp_type == "bimodal_negative":
        means, stds, weights = (
            params["neg_means"],
            params["neg_stds"],
            params["neg_weights"],
        )
        neg = sum(
            w * sps.norm.pdf(x, m, s)
            for m, s, w in zip(means, stds, weights, strict=True)
        )
        return neg, sps.norm.pdf(x, params["pos_mean"], params["pos_std"])
    raise ValueError(f"Unknown DGP type: {dgp_type}")


def density_range(dgp, rng: np.random.Generator, pad_pct: float = 0.5) -> np.ndarray:
    """A plotting grid covering both classes' central mass.

    Args:
        dgp: A DGP instance to sample for empirical support.
        rng: Random generator for the support sample.
        pad_pct: Lower/upper percentile to clip heavy tails for the x-range.

    Returns:
        A dense linspace spanning the (clipped) joint support.
    """
    pos, neg = dgp.sample(40_000, 40_000, rng)
    lo = min(np.percentile(neg, pad_pct), np.percentile(pos, pad_pct))
    hi = max(np.percentile(neg, 100 - pad_pct), np.percentile(pos, 100 - pad_pct))
    span = hi - lo
    return np.linspace(lo - 0.02 * span, hi + 0.02 * span, 600)


def true_roc_auc(dgp, n_grid: int = 4000) -> float:
    """Numerically integrate a DGP's analytic ROC to recover its AUC."""
    fpr = np.linspace(0, 1, n_grid)
    return float(np.trapezoid(dgp.get_true_roc(fpr), fpr))


def contiguous_spans(
    mask: np.ndarray, x: np.ndarray, min_len: int = 1
) -> list[tuple[float, float]]:
    """Return (x_start, x_end) intervals for runs of True in a boolean mask.

    Args:
        mask: Boolean array over grid points.
        x: Grid coordinates aligned with the mask.
        min_len: Minimum run length (grid points) to report; shorter runs are
            dropped to suppress single-point speckle in shaded regions.
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
# ---
# # A1 — DGP atlas: what was simulated
#
# **Motivation.** The simulation spans seven DGP families (eight including the
# binormal special case), each named in prose but never drawn. This atlas
# shows, per family, the **score distributions** of the two classes and the
# **true ROC curves** they induce. The top row makes the distributional
# departures from Gaussianity concrete (heavy tails, skew, bounded support,
# multimodality); the bottom row shows how each family's ROC steepens as the
# true AUC rises — the geometry that drives where the band's tail floors do
# their work.
#
# Densities are the exact analytic class densities at a moderate shape setting
# and a reference AUC of 0.85; ROCs are the exact analytic curves at three
# target AUCs.


# %%
def plot_atlas(dgps: list[str], fig_name: str, title: str) -> None:
    """Two-row atlas (densities; true ROCs) across a list of DGP families."""
    rng = np.random.default_rng(20260621)
    ncols = len(dgps)
    fig, axes = plt.subplots(2, ncols, figsize=(2.7 * ncols, 5.4), squeeze=False)
    for c, dgp_type in enumerate(dgps):
        shape = EXAMPLE_SHAPE[dgp_type]
        # --- Top: class score densities at the reference AUC ---------------
        dgp_ref, params_ref = make_dgp(dgp_type, 0.85, shape)
        x = density_range(dgp_ref, rng)
        pdf_neg, pdf_pos = score_densities(dgp_type, params_ref, x)
        ax = axes[0, c]
        ax.fill_between(x, pdf_neg, color=NEG_COLOR, alpha=0.45, lw=0)
        ax.fill_between(x, pdf_pos, color=dcolor(dgp_type), alpha=0.45, lw=0)
        ax.plot(x, pdf_neg, color=NEG_COLOR, lw=1.2)
        ax.plot(x, pdf_pos, color=dcolor(dgp_type), lw=1.4)
        shape_txt = ", ".join(f"{k}={v:g}" for k, v in shape.items())
        ax.set_title(
            dlabel(dgp_type) + (f"\n({shape_txt})" if shape_txt else "\n"), fontsize=9.5
        )
        ax.set_yticks([])
        ax.set_xlabel("Score")
        if c == 0:
            ax.set_ylabel("Density")
            ax.legend(
                handles=[
                    Patch(facecolor=NEG_COLOR, alpha=0.55, label="Negative"),
                    Patch(facecolor=dcolor(dgp_type), alpha=0.55, label="Positive"),
                ],
                loc="upper right",
                fontsize=7,
            )

        # --- Bottom: true ROC at three target AUCs ------------------------
        ax = axes[1, c]
        fpr = np.linspace(0, 1, 600)
        for auc in AUC_LEVELS:
            dgp_auc, _ = make_dgp(dgp_type, auc, shape)
            ax.plot(
                fpr,
                dgp_auc.get_true_roc(fpr),
                color=AUC_COLORS[auc],
                lw=1.8,
                label=f"AUC {auc:.2f}",
            )
        ax.plot([0, 1], [0, 1], color="0.7", ls=":", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.0)
        ax.set_aspect("equal")
        ax.set_xlabel("FPR")
        if c == 0:
            ax.set_ylabel("TPR")
            ax.legend(loc="lower right", fontsize=7.5)
    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, fig_name)


# %%
plot_atlas(
    ["binormal", "hetero_gaussian", "logitnormal", "student_t", "gamma", "weibull"],
    "figA1a_dgp_atlas_gaussian_heavy",
    "DGP atlas I — score distributions and the ROCs they induce",
)

# %% [markdown]
# The non-standard families carry the multimodal / bounded-support story, so
# they get a denser treatment: three shape settings of the score distribution
# (a shape sweep at fixed AUC = 0.85) and a true-ROC panel overlaying those
# three shapes.

# %%
NONSTD_SWEEP = {
    "beta_opposing": ("alpha", [0.7, 2.0, 6.0], r"Beta $\alpha$"),
    "bimodal_negative": ("mode_separation", [0.5, 2.0, 3.5], r"mode sep $\Delta$"),
}

fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.6), squeeze=False)
rng = np.random.default_rng(20260622)
for r, (dgp_type, (sweep_key, sweep_vals, sweep_lab)) in enumerate(
    NONSTD_SWEEP.items()
):
    base_shape = dict(EXAMPLE_SHAPE[dgp_type])
    sweep_colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(sweep_vals)))
    for j, val in enumerate(sweep_vals):
        shape = dict(base_shape, **{sweep_key: val})
        dgp_ref, params_ref = make_dgp(dgp_type, 0.85, shape)
        x = density_range(dgp_ref, rng)
        pdf_neg, pdf_pos = score_densities(dgp_type, params_ref, x)
        ax = axes[r, j]
        ax.fill_between(x, pdf_neg, color=NEG_COLOR, alpha=0.45, lw=0)
        ax.fill_between(x, pdf_pos, color=dcolor(dgp_type), alpha=0.45, lw=0)
        ax.plot(x, pdf_neg, color=NEG_COLOR, lw=1.1)
        ax.plot(x, pdf_pos, color=dcolor(dgp_type), lw=1.3)
        ax.set_yticks([])
        ax.set_xlabel("Score")
        ax.set_title(f"{sweep_lab} = {val:g}", fontsize=9.5)
        if j == 0:
            ax.set_ylabel(f"{dlabel(dgp_type)}\nDensity")
    # Right column: true ROC across the shape sweep at fixed AUC
    ax = axes[r, 3]
    fpr = np.linspace(0, 1, 600)
    for j, val in enumerate(sweep_vals):
        shape = dict(base_shape, **{sweep_key: val})
        dgp_auc, _ = make_dgp(dgp_type, 0.85, shape)
        ax.plot(
            fpr,
            dgp_auc.get_true_roc(fpr),
            color=sweep_colors[j],
            lw=1.8,
            label=f"{sweep_lab}={val:g}",
        )
    ax.plot([0, 1], [0, 1], color="0.7", ls=":", lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("True ROC (AUC = 0.85)", fontsize=9.5)
    ax.legend(loc="lower right", fontsize=7)
fig.suptitle("DGP atlas II — non-standard shapes (shape sweep)")
fig.tight_layout()
save_figure(fig, "figA1b_dgp_atlas_nonstandard")

# %%
# Sanity: the drawn ROCs hit their target AUC.
for dgp_type in DGP_FACTORY:
    dgp_chk, _ = make_dgp(dgp_type, 0.85, EXAMPLE_SHAPE[dgp_type])
    auc_chk = true_roc_auc(dgp_chk)
    print(f"{dgp_type:18s} target AUC 0.850 -> realized {auc_chk:.3f}")

# %% [markdown]
# ---
# # A2 — The simulation design space
#
# **Motivation.** Three design choices govern *which* problems the study
# probes, and all three are described in `simulation_spec.md` but never shown:
# (a) AUC is sampled uniformly on the **probit scale** z = Φ⁻¹(AUC), which
# concentrates draws in the high-AUC regime where the ROC is steep and tail
# behavior dominates; (b) within each DGP the joint (AUC, shape) design is a
# **maximin Latin hypercube** decorrelated by the **Iman–Conover** transform,
# so the axes are explored independently and space-fillingly; (c) the design
# lives in interpretable (AUC, shape) coordinates, but the simulator needs the
# *DGP* parameters, so each point is pushed through a per-family **inversion**
# (`roc_to_dgp`) that solves for the parameter hitting the target AUC — a
# strongly nonlinear map. Combinations with an unachievable AUC return NaN and
# are dropped (rare for the active families and bounds).

# %%
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))

# --- (a) Probit-scale AUC sampling vs. uniform-in-AUC ------------------------
ax = axes[0]
lo, hi = 0.55, 0.99
z_lo, z_hi = norm.ppf(lo), norm.ppf(hi)
rng = np.random.default_rng(7)
u = rng.uniform(0, 1, 200_000)
auc_samples = norm.cdf(z_lo + u * (z_hi - z_lo))
ax.hist(
    auc_samples,
    bins=60,
    density=True,
    color="#3182BD",
    alpha=0.55,
    label="Probit-scale samples",
)
# Analytic design density in AUC units: proportional to 1/phi(Phi^-1(AUC)).
auc_grid = np.linspace(lo, hi, 400)
design_density = 1.0 / norm.pdf(norm.ppf(auc_grid))
design_density /= np.trapezoid(design_density, auc_grid)
ax.plot(auc_grid, design_density, color="#08306B", lw=2.0, label="Design density")
ax.axhline(1.0 / (hi - lo), color="0.4", ls="--", lw=1.2, label="Uniform-in-AUC")
ratio = design_density[-1] / design_density[0]
ax.annotate(
    f"~{ratio:.0f}$\\times$ denser\nat AUC = 0.99",
    xy=(0.97, design_density[-1]),
    xytext=(0.74, design_density[-1] * 0.8),
    fontsize=8.5,
    color="#08306B",
    arrowprops=dict(arrowstyle="->", color="#08306B", lw=1.0),
)
ax.set_xlabel("True AUC")
ax.set_ylabel("Design density")
ax.set_title("(a) Probit-scale AUC sampling")
ax.legend(loc="upper left", fontsize=7.5)

# --- (b) Maximin LHS + Iman-Conover decorrelation ---------------------------
ax = axes[1]
n_lhs = 200
rng = np.random.default_rng(11)
lhs_unit = maximin_lhs(n=n_lhs, k=2, method="build", dup=5, seed=rng.integers(2**31))
lhs_dec = iman_conover_transform(lhs_unit, target_corr=np.eye(2), rng=rng)
# Scale to the student_t design: AUC (probit) x df (linear).
df_lo, df_hi = 1.1, 30.0


def scale_design(unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    auc = norm.cdf(z_lo + unit[:, 0] * (z_hi - z_lo))
    df = df_lo + unit[:, 1] * (df_hi - df_lo)
    return auc, df


auc_raw, df_raw = scale_design(lhs_unit)
auc_dec, df_dec = scale_design(lhs_dec)
corr_raw = np.corrcoef(lhs_unit.T)[0, 1]
corr_dec = np.corrcoef(lhs_dec.T)[0, 1]
ax.scatter(
    auc_raw,
    df_raw,
    s=16,
    color="0.6",
    alpha=0.7,
    label=f"maximin LHS (r = {corr_raw:+.2f})",
)
ax.scatter(
    auc_dec,
    df_dec,
    s=18,
    color="#D55E00",
    alpha=0.85,
    label=f"+ Iman–Conover (r = {corr_dec:+.2f})",
)
ax.set_xlabel("True AUC (probit-sampled)")
ax.set_ylabel("Degrees of freedom")
ax.set_title("(b) Maximin LHS design (Student-t)")
ax.legend(loc="upper right", fontsize=7.5)

# --- (c) Achievable-AUC filtering -------------------------------------------
ax = axes[2]
alpha_grid = np.linspace(0.5, 10.0, 160)
auc_grid_c = np.linspace(0.55, 0.99, 160)
AA, UU = np.meshgrid(alpha_grid, auc_grid_c, indexing="ij")
mapped = map_lhs_to_dgp("beta_opposing", {"auc": UU.ravel(), "alpha": AA.ravel()})
beta_vals = np.asarray(mapped["beta"], dtype=float).reshape(AA.shape)
unachievable = ~np.isfinite(beta_vals)
cs = ax.contourf(alpha_grid, auc_grid_c, beta_vals.T, levels=14, cmap="viridis")
ax.contour(
    alpha_grid,
    auc_grid_c,
    beta_vals.T,
    levels=cs.levels,
    colors="white",
    linewidths=0.4,
    alpha=0.6,
)
cbar = fig.colorbar(cs, ax=ax, shrink=0.85)
cbar.set_label(r"Solved Beta $\beta$ (DGP parameter)")
if unachievable.any():
    ax.contourf(
        alpha_grid,
        auc_grid_c,
        unachievable.T.astype(float),
        levels=[0.5, 1.5],
        colors=["#CC6677"],
        alpha=0.6,
    )
ax.set_xlabel(r"Beta $\alpha$  (design coordinate)")
ax.set_ylabel("Target AUC  (design coordinate)")
n_drop = int(unachievable.sum())
ax.set_title(
    "(c) Design → DGP inversion (Beta opposing)\n"
    + (
        f"{100 * n_drop / unachievable.size:.1f}% unachievable (dropped)"
        if n_drop
        else "all combinations achievable here"
    )
)
fig.suptitle("Simulation design: probit AUC, maximin LHS, DGP-parameter inversion")
fig.tight_layout()
save_figure(fig, "figA2_simulation_design")

# %% [markdown]
# ---
# # A3 — Anatomy of the studentized bootstrap interior
#
# **Motivation.** The example-band figures in the paper (`fig15`) show the
# *assembled* band but not the machinery underneath. Here we open the interior
# on a single high-AUC dataset:
#
# - **(a)** the cloud of bootstrap ROC curves, split into the retained
#   (1 − α) fraction (smallest studentized supremum deviation) and the
#   discarded tail whose excursions set the envelope's reach;
# - **(b)** the studentized tube R̂(t) ± c·σ̂(t) against the actual envelope of
#   retained curves — the envelope is *contained in* the tube but does not
#   reach it, especially the short lower arm at the steep corner (the
#   "projection, not tube" remark in the spec);
# - **(c)** the local standard error: the bootstrap σ̂(t) collapses on the TPR
#   plateau, where the Wilson variance floor takes over as the studentization
#   denominator.

# %%
A3_DGP, A3_AUC, A3_N, A3_B, A3_ALPHA = "student_t", 0.95, 300, 3000, 0.05


def fit_one_dataset(dgp_type: str, auc: float, n: int, b: int, seed: int) -> dict:
    """Sample one dataset and run the envelope suite, keeping diagnostics.

    Returns a dict with the FPR grid, empirical/true TPR, the bootstrap TPR
    matrix (numpy), and the band-suite variants (with the pre-floor arm).
    """
    dgp, _ = make_dgp(dgp_type, auc, EXAMPLE_SHAPE[dgp_type])
    rng = np.random.default_rng(seed)
    n_pos = n_neg = n // 2
    pos, neg = dgp.sample(n_pos, n_neg, rng)
    y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    y_score = np.concatenate([pos, neg]).astype(np.float64)
    fpr_grid = np.linspace(0, 1, n_neg + 1)
    boot = generate_bootstrap_grid(
        y_true=torch.as_tensor(y_true, dtype=torch.float32, device=DEVICE),
        y_score=torch.as_tensor(y_score, dtype=torch.float32, device=DEVICE),
        B=b,
        grid=torch.as_tensor(fpr_grid, dtype=torch.float32, device=DEVICE),
        device=DEVICE,
    )
    suite = envelope_band_suite(
        boot_tpr_matrix=boot,
        fpr_grid=fpr_grid,
        y_true=y_true,
        y_score=y_score,
        alphas=[A3_ALPHA],
        include_pre_floor_arm=True,
    )[A3_ALPHA]
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
        emp=emp_tpr,
        true=dgp.get_true_roc(fpr_grid),
        boot=boot.cpu().numpy().astype(np.float64),
        suite=suite,
        n_pos=n_pos,
    )


def wilson_floor_variance(p: np.ndarray, n: int, z: float) -> np.ndarray:
    """Wilson-score variance floor used inside the studentization."""
    return (1.0 / (1.0 + z**2 / n) ** 2) * (p * (1 - p) / n + z**2 / (4 * n**2))


def studentized_retention(ex: dict, alpha: float) -> dict:
    """Reproduce the suite's variance-floored KS retention to flag curves.

    Mirrors `envelope_band_suite`: deviations are studentized by
    sqrt(max(bootstrap variance, Wilson floor)), each curve is scored by its
    supremum absolute studentized deviation, and the smallest (1 − α) fraction
    is retained.

    Returns a dict with the retained boolean mask, the per-curve statistic,
    the retention threshold c, and the floored standard error.
    """
    boot, emp = ex["boot"], ex["emp"]
    var_raw = boot.var(axis=0, ddof=1)
    z = norm.ppf(1 - alpha / 2)
    wilson_var = wilson_floor_variance(emp, ex["n_pos"], z)
    std_floored = np.sqrt(np.maximum(var_raw, wilson_var))
    eps = 1e-12
    zscores = (boot - emp) / np.maximum(std_floored, eps)
    ks = np.abs(zscores).max(axis=1)
    n_retain = int(np.ceil((1 - alpha) * boot.shape[0]))
    c = np.sort(ks)[n_retain - 1]
    return dict(
        retained=ks <= c,
        ks=ks,
        c=float(c),
        std_floored=std_floored,
        std_boot=np.sqrt(var_raw),
        std_wilson=np.sqrt(wilson_var),
    )


ex = fit_one_dataset(A3_DGP, A3_AUC, A3_N, A3_B, seed=20260623)
ret = studentized_retention(ex, A3_ALPHA)

fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4))

# --- (a) bootstrap cloud, retained vs discarded -----------------------------
ax = axes[0]
rng = np.random.default_rng(3)
show = rng.choice(ex["boot"].shape[0], size=500, replace=False)
for idx in show:
    is_ret = ret["retained"][idx]
    ax.plot(
        ex["fpr"],
        ex["boot"][idx],
        color=("#3182BD" if is_ret else "#D55E00"),
        lw=0.5,
        alpha=(0.06 if is_ret else 0.18),
        zorder=1,
    )
ax.plot(ex["fpr"], ex["emp"], color="black", lw=1.8, zorder=4, label="Empirical ROC")
ax.plot([0, 1], [0, 1], color="0.7", ls=":", lw=0.8)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.0)
ax.set_aspect("equal")
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.set_title("(a) Bootstrap cloud: retained vs. discarded")
ax.legend(
    handles=[
        Line2D([], [], color="black", lw=1.8, label="Empirical ROC"),
        Line2D(
            [],
            [],
            color="#3182BD",
            lw=2,
            label=f"Retained ($1-\\alpha$ = {1 - A3_ALPHA:.2f})",
        ),
        Line2D([], [], color="#D55E00", lw=2, label=r"Discarded ($\alpha$ tail)"),
    ],
    loc="lower right",
    fontsize=7.5,
)

# --- (b) studentized tube vs. projected envelope ----------------------------
ax = axes[1]
lower_arm, upper_arm = ex["suite"]["envelope_pre_floor"]
tube_lo = np.clip(ex["emp"] - ret["c"] * ret["std_floored"], 0, 1)
tube_hi = np.clip(ex["emp"] + ret["c"] * ret["std_floored"], 0, 1)
ax.fill_between(
    ex["fpr"],
    tube_lo,
    tube_hi,
    color="0.75",
    alpha=0.4,
    lw=0,
    label=r"Studentized tube  $\hat{R} \pm c\cdot\hat{\sigma}$",
)
ax.plot(ex["fpr"], ex["emp"], color="black", lw=1.5, label="Empirical ROC")
ax.plot(
    ex["fpr"], lower_arm, color=ENVELOPE_COLOR, lw=1.6, label="Envelope (bootstrap arm)"
)
ax.plot(ex["fpr"], upper_arm, color=ENVELOPE_COLOR, lw=1.6)
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 1.02)
ax.set_xlabel("FPR (low-FPR corner)")
ax.set_ylabel("TPR")
ax.set_title("(b) Tube contains, but envelope under-reaches")
ax.legend(loc="lower right", fontsize=7.5)

# --- (c) local SE: bootstrap vs Wilson floor --------------------------------
ax = axes[2]
ax.plot(
    ex["fpr"],
    ret["std_boot"],
    color="#0072B2",
    lw=1.8,
    label=r"Bootstrap $\hat{\sigma}(t)$",
)
ax.plot(
    ex["fpr"],
    ret["std_wilson"],
    color="#009E73",
    lw=1.8,
    ls="--",
    label=r"Wilson floor $\sigma(t)$",
)
floor_active = ret["std_wilson"] > ret["std_boot"]
ax.fill_between(
    ex["fpr"],
    0,
    np.maximum(ret["std_boot"], ret["std_wilson"]),
    where=floor_active,
    color="#009E73",
    alpha=0.15,
    lw=0,
    label="Floor active (variance collapsed)",
)
ax.set_xlim(0, 1)
ax.set_ylim(bottom=0)
ax.set_xlabel("FPR")
ax.set_ylabel("Local standard error")
ax.set_title("(c) Studentization denominator")
ax.legend(loc="upper right", fontsize=7.5)
fig.suptitle(
    f"Bootstrap interior on one dataset — {dlabel(A3_DGP)}, "
    f"AUC $\\approx$ {A3_AUC}, n = {A3_N}"
)
fig.tight_layout()
save_figure(fig, "figA3_bootstrap_interior_anatomy")

# %% [markdown]
# ---
# # A4 — The exact Beta order-statistic law
#
# **Motivation.** At the first few grid points of a steep ROC the dominant
# uncertainty is *horizontal* — the true FPR of the operating threshold, an
# extreme order statistic of the negatives. The method's repair rests on one
# exact, finite-sample, distribution-free fact: for continuous scores the true
# FPR exceedance at the j-th largest negative score is
# F̄(X₍ⱼ₎) ∼ Beta(j, n₀ + 1 − j), regardless of the score distribution
# (probability integral transform). This figure shows the law (a), verifies it
# by Monte Carlo on a deliberately non-Gaussian DGP (b), and draws the
# resulting stepwise lower floor on an example (c).

# %%
A4_N0 = 150
A4_J_SHOW = [1, 2, 3, 5, 10]
A4_JMAX = 25
A4_ALPHA = 0.05
A4_ALPHA_E = A4_ALPHA / (2 * A4_JMAX)

fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3))

# --- (a) Beta(j, n0+1-j) densities and their upper quantiles -----------------
ax = axes[0]
t = np.linspace(0, 0.45, 1500)
cmap = plt.get_cmap("plasma")(np.linspace(0.1, 0.85, len(A4_J_SHOW)))
for color, j in zip(cmap, A4_J_SHOW, strict=True):
    pdf = beta_dist.pdf(t, j, A4_N0 + 1 - j)
    q_j = beta_dist.ppf(1 - A4_ALPHA_E, j, A4_N0 + 1 - j)
    ax.plot(t, pdf, color=color, lw=1.8, label=f"j = {j}")
    ax.axvline(q_j, color=color, ls=":", lw=1.0, alpha=0.8)
q1 = beta_dist.ppf(1 - A4_ALPHA_E, 1, A4_N0)
qJ = beta_dist.ppf(1 - A4_ALPHA_E, A4_JMAX, A4_N0 + 1 - A4_JMAX)
ax.axvspan(0, q1, color="0.85", alpha=0.5)
ax.annotate(
    f"vacuous below\n$q_1 \\approx$ {q1:.3f} ({q1 * A4_N0:.1f}/$n_0$)",
    xy=(q1, beta_dist.pdf(q1, 1, A4_N0) * 0.5),
    xytext=(q1 + 0.05, 25),
    fontsize=8,
    arrowprops=dict(arrowstyle="->", lw=0.9),
)
ax.set_xlim(0, 0.45)
ax.set_xlabel(r"True FPR exceedance  $\bar{F}(X_{(j)})$")
ax.set_ylabel("Density")
ax.set_title(f"(a) Beta($j$, $n_0+1-j$) law  ($n_0$ = {A4_N0})")
ax.legend(loc="upper right", fontsize=7.5, ncol=2)

# --- (b) distribution-free Monte-Carlo verification -------------------------
ax = axes[1]
mc_dgp, _ = make_dgp("gamma", 0.85, EXAMPLE_SHAPE["gamma"])
mc_rng = np.random.default_rng(20260624)
neg_dist = sps.gamma(EXAMPLE_SHAPE["gamma"]["shape"], scale=1.0)
n_reps = 6000
records: dict[int, list[float]] = {1: [], 5: []}
for _ in range(n_reps):
    _, neg = mc_dgp.sample(2, A4_N0, mc_rng)
    neg_desc = np.sort(neg)[::-1]
    for j in records:
        # True FPR exceedance = survival function of the negative law at X_(j).
        records[j].append(float(neg_dist.sf(neg_desc[j - 1])))
for color, j in zip(["#3182BD", "#D55E00"], [1, 5], strict=True):
    ax.hist(
        records[j], bins=40, density=True, color=color, alpha=0.45, label=f"MC, j = {j}"
    )
    ax.plot(t, beta_dist.pdf(t, j, A4_N0 + 1 - j), color=color, lw=2.0)
ax.set_xlim(0, 0.2)
ax.set_xlabel(r"True FPR exceedance  $\bar{F}(X_{(j)})$")
ax.set_ylabel("Density")
ax.set_title("(b) MC verification (Gamma negatives)")
ax.legend(loc="upper right", fontsize=7.5)

# --- (c) the resulting stepwise lower floor ---------------------------------
ax = axes[2]


def wilson_lower_one_sided(p: np.ndarray, n: int, alpha_e: float) -> np.ndarray:
    """One-sided Wilson lower bound on a binomial proportion."""
    z = norm.ppf(1 - alpha_e)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return np.clip(center - half, 0.0, 1.0)


floor_dgp, _ = make_dgp("student_t", 0.95, EXAMPLE_SHAPE["student_t"])
floor_rng = np.random.default_rng(20260625)
pos_f, neg_f = floor_dgp.sample(A4_N0, A4_N0, floor_rng)
fpr_f = np.linspace(0, 1, A4_N0 + 1)
neg_desc = np.sort(neg_f)[::-1]
js = np.arange(1, A4_JMAX + 1)
q_js = beta_dist.ppf(1 - A4_ALPHA_E, js, A4_N0 + 1 - js)
tpr_hat = np.array([(pos_f > neg_desc[j - 1]).mean() for j in js])
bounds = np.concatenate([[0.0], wilson_lower_one_sided(tpr_hat, A4_N0, A4_ALPHA_E)])
floor = np.zeros_like(fpr_f)
zone = (fpr_f > 0) & (fpr_f <= q_js[-1])
j_star = np.searchsorted(q_js, fpr_f[zone], side="right")
floor[zone] = bounds[j_star]
emp_f = (
    compute_empirical_roc_from_scores(
        neg_scores=torch.as_tensor(neg_f, dtype=torch.float64),
        pos_scores=torch.as_tensor(pos_f, dtype=torch.float64),
        fpr_grid=torch.as_tensor(fpr_f, dtype=torch.float64),
    )
    .cpu()
    .numpy()
)
ax.plot(
    fpr_f,
    floor_dgp.get_true_roc(fpr_f),
    color="black",
    ls="--",
    lw=1.6,
    label="True ROC",
)
ax.plot(fpr_f, emp_f, color="0.35", lw=1.3, label="Empirical ROC")
ax.step(
    fpr_f,
    floor,
    where="post",
    color="#F0A202",
    lw=2.0,
    label=r"Beta floor  $L_\beta(t)$",
)
ax.axvspan(0, q1, color="0.85", alpha=0.5)
ax.axvline(qJ, color="0.5", ls=":", lw=1.0)
ax.annotate(
    f"jurisdiction edge\n$q_{{25}} \\approx$ {qJ:.2f}",
    xy=(qJ, 0.1),
    xytext=(qJ + 0.04, 0.18),
    fontsize=8,
    arrowprops=dict(arrowstyle="->", lw=0.9),
)
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 1.0)
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.set_title("(c) Stepwise lower floor (Student-t, AUC 0.95)")
ax.legend(loc="lower right", fontsize=7.5)
fig.suptitle("The exact Beta order-statistic floor")
fig.tight_layout()
save_figure(fig, "figA4_beta_orderstat_law")

# %% [markdown]
# **Reading A4.** The Beta densities march rightward and broaden with j, so the
# floor's per-event upper quantiles qⱼ tile the low-FPR axis; below q₁ no order
# statistic qualifies and the floor is honestly vacuous (0). Panel (b) confirms
# the law holds for a non-Gaussian (Gamma) negative class — the histograms of
# the *true* exceedance sit on the Beta densities — which is the whole point of
# a distribution-free bound. Panel (c) shows the floor stepping up through its
# jurisdiction and handing back to the bootstrap beyond q₂₅.

# %% [markdown]
# ---
# # A5 — Why one variance model cannot cover both corners
#
# **Motivation.** The Wilson floor models only *vertical* binomial TPR noise.
# That is the complete uncertainty model on the flat TPR plateau, but at the
# steep low-FPR corner the dominant uncertainty is the horizontal threshold
# location, and the binomial variance is several-fold too small. Panel (a)
# compares the Wilson standard error against **Monte-Carlo truth** (the SD of
# the empirical TPR over many fresh datasets) and the single-dataset bootstrap.
#
# Panel (b) makes the asymptotic argument concrete. At a *fixed interior* FPR
# the empirical ROC's bias is O(1/n) while its SD is O(1/√n), so the
# standardized bias |E[R̂] − R| / SD vanishes and the Gaussian/bootstrap
# approximation is sound. At the *moving boundary* t = k/n₀ (fixed small k) it
# does not: there the empirical TPR evaluates the concave true ROC at the
# random exceedance F̄(X₍ₖ₎) ∼ Beta(k, n₀+1−k), so by Jensen it is biased and —
# crucially — the bias stays a roughly constant fraction of the local SD as n
# grows (a downward bias for a steep concave corner). That non-vanishing
# standardized bias is exactly why the bootstrap argument fails at the corner
# and why a finite-sample, order-statistic device (the Beta floor) is needed.

# %%
A5_DGP, A5_AUC = "student_t", 0.95


def emp_roc_fast(pos: np.ndarray, neg: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Empirical ROC TPR on an FPR grid (threshold = upper quantile of neg)."""
    thr = np.quantile(neg, 1 - grid, method="higher")
    pos_sorted = np.sort(pos)
    return 1.0 - np.searchsorted(pos_sorted, thr, side="right") / len(pos)


def mc_roc_stats(dgp_type: str, auc: float, n: int, n_reps: int, seed: int) -> dict:
    """Monte-Carlo mean and SD of the empirical ROC at a DGP/n.

    Returns a dict with the FPR grid, mean and SD of empirical TPR across
    repeats, and the analytic true TPR on the same grid.
    """
    dgp, _ = make_dgp(dgp_type, auc, EXAMPLE_SHAPE[dgp_type])
    rng = np.random.default_rng(seed)
    n_pos = n_neg = n // 2
    grid = np.linspace(0, 1, n_neg + 1)
    acc = np.zeros((n_reps, len(grid)))
    for r in range(n_reps):
        pos, neg = dgp.sample(n_pos, n_neg, rng)
        acc[r] = emp_roc_fast(pos, neg, grid)
    return dict(
        fpr=grid,
        mean=acc.mean(axis=0),
        sd=acc.std(axis=0, ddof=1),
        true=dgp.get_true_roc(grid),
        n_pos=n_pos,
    )


fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

# --- (a) variance comparison at one n ---------------------------------------
ax = axes[0]
mc = mc_roc_stats(A5_DGP, A5_AUC, n=1000, n_reps=1500, seed=20260626)
z = norm.ppf(1 - 0.05 / 2)
wilson_sd = np.sqrt(wilson_floor_variance(mc["mean"], mc["n_pos"], z))
# Single-dataset bootstrap SD for reference.
ex_b = fit_one_dataset(A5_DGP, A5_AUC, 1000, 2000, seed=20260627)
boot_sd = ex_b["boot"].std(axis=0, ddof=1)
ax.plot(mc["fpr"], mc["sd"], color="black", lw=2.0, label="Monte-Carlo truth SD")
ax.plot(mc["fpr"], boot_sd, color="#0072B2", lw=1.6, label="Bootstrap SD (1 dataset)")
ax.plot(
    mc["fpr"], wilson_sd, color="#009E73", lw=1.8, ls="--", label="Wilson (binomial) SD"
)
ax.set_xlim(0, 0.3)
ax.set_ylim(bottom=0)
ax.set_xlabel("FPR (low-FPR corner)")
ax.set_ylabel("Local SD of TPR")
ax.set_title("(a) Wilson variance is too small at the steep corner")
ax.legend(loc="upper right", fontsize=7.5)
# Quote the gap a few grid points in.
k5 = 5
ratio5 = mc["sd"][k5] / max(wilson_sd[k5], 1e-9)
ax.annotate(
    f"truth / Wilson $\\approx$ {ratio5:.1f}$\\times$\nat k = {k5}",
    xy=(mc["fpr"][k5], mc["sd"][k5]),
    xytext=(0.10, mc["sd"][k5] * 1.05),
    fontsize=8,
    arrowprops=dict(arrowstyle="->", lw=0.9),
)

# --- (b) standardized bias: moving boundary vs. fixed interior --------------
ax = axes[1]
A5_NS = [100, 300, 1000, 3000]
INTERIOR_FPR = 0.10
boundary_std_bias, interior_std_bias = [], []
for n in A5_NS:
    mc_n = mc_roc_stats(A5_DGP, A5_AUC, n=n, n_reps=1500, seed=20260628 + n)
    std_bias = np.abs(mc_n["mean"] - mc_n["true"]) / np.maximum(mc_n["sd"], 1e-9)
    # Moving boundary: average over the first few order-statistic grid points.
    boundary_std_bias.append(np.mean(std_bias[[1, 2, 3]]))
    # Fixed interior: the grid point nearest a fixed FPR.
    i_int = int(np.argmin(np.abs(mc_n["fpr"] - INTERIOR_FPR)))
    interior_std_bias.append(std_bias[i_int])
ax.plot(
    A5_NS,
    boundary_std_bias,
    color="#D55E00",
    lw=2.0,
    marker="o",
    ms=6,
    label=r"Moving boundary  $t = k/n_0$  ($k \in \{1,2,3\}$)",
)
ax.plot(
    A5_NS,
    interior_std_bias,
    color="#0072B2",
    lw=2.0,
    marker="s",
    ms=6,
    label=f"Fixed interior  FPR = {INTERIOR_FPR:g}",
)
ax.set_xscale("log")
ax.set_ylim(bottom=0)
ax.set_xlabel("Total sample size n")
ax.set_ylabel(r"Standardized bias  $|E[\hat{R}] - R_{\mathrm{true}}|\,/\,$SD")
ax.set_title("(b) Bias/SD vanishes in the interior, not at the boundary")
ax.legend(loc="center right", fontsize=7.5)
fig.suptitle(f"Two corners, two models — {dlabel(A5_DGP)}, AUC $\\approx$ {A5_AUC}")
fig.tight_layout()
save_figure(fig, "figA5_two_corner_variance_bias")

# %% [markdown]
# ---
# # A6 — Lower-band assembly waterfall
#
# **Motivation.** `fig15d` shows *which* mechanism owns the lower bound across
# the whole AUC × n grid; this is the single-curve complement. On one dataset
# per target AUC we stack the lower bound in the order the method builds it —
# bare bootstrap envelope → + Wilson rectangle floor → + Beta floor — and shade
# the FPR spans where each added floor strictly *lowered* the bound (each floor
# is a pointwise minimum, so it can only widen the band downward to restore
# coverage). The floors' jurisdictions barely overlap, and their share of the
# curve grows with AUC.

# %%
A6_DGP, A6_N, A6_B, A6_ALPHA = "student_t", 300, 3000, 0.05
fig, axes = plt.subplots(
    1, len(AUC_LEVELS), figsize=(4.4 * len(AUC_LEVELS), 4.3), sharey=True
)
for ax, auc in zip(axes, AUC_LEVELS, strict=True):
    ex6 = fit_one_dataset(A6_DGP, auc, A6_N, A6_B, seed=20260629 + int(100 * auc))
    bare = ex6["suite"]["envelope_no_floors"][0]
    plus_wilson = ex6["suite"]["envelope_no_beta_floor"][0]
    plus_beta = ex6["suite"]["envelope"][0]
    fpr = ex6["fpr"]
    n_neg = A6_N // 2
    q1_6 = beta_dist.ppf(1 - A6_ALPHA / (2 * 25), 1, n_neg)
    eps = 1e-6
    wilson_changed = plus_wilson < bare - eps
    beta_changed = (plus_beta < plus_wilson - eps) | ((fpr > 0) & (fpr < q1_6))
    for x0, x1 in contiguous_spans(wilson_changed, fpr, min_len=2):
        ax.axvspan(x0, x1, color="#009E73", alpha=0.16, lw=0)
    for x0, x1 in contiguous_spans(beta_changed, fpr, min_len=2):
        ax.axvspan(x0, x1, color="#F0E442", alpha=0.4, lw=0)
    ax.plot(fpr, ex6["true"], color="black", ls="--", lw=1.5, label="True ROC")
    ax.plot(fpr, bare, color="#999999", lw=1.5, label="Bare envelope")
    ax.plot(fpr, plus_wilson, color="#0072B2", lw=1.5, label="+ Wilson floor")
    ax.plot(fpr, plus_beta, color=ENVELOPE_COLOR, lw=1.9, label="+ Beta floor")
    ax.plot([0, 1], [0, 1], color="0.8", ls=":", lw=0.8)
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("FPR")
    ax.set_title(f"target AUC = {auc:.2f}")
    if ax is axes[0]:
        ax.set_ylabel("Lower bound (TPR)")
        ax.legend(loc="lower right", fontsize=7.5)
fig.suptitle(f"Lower-band assembly waterfall — {dlabel(A6_DGP)}, n = {A6_N}")
fig.tight_layout()
save_figure(fig, "figA6_lower_band_waterfall")

# %% [markdown]
# **Reading A6.** The bare bootstrap envelope (gray) pinches shut at the steep
# low-FPR corner and on the high-AUC plateau — its lower bound rides too high to
# cover, which is why it is not a valid band. The Wilson floor (green spans)
# pushes the bound down across the plateau where the bootstrap variance
# collapsed; the Beta floor (gold spans) pushes it down at the extreme low-FPR
# corner, dropping honestly to 0 below q₁. The final orange bound therefore sits
# below the true ROC throughout, and as AUC rises the floors' combined
# jurisdiction expands — exactly the regime where every variance-only competitor
# fails.
