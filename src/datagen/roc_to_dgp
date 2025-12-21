"""
Efficient LHS-to-DGP parameter mapping with closed-form solutions where available.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import brentq

from ..sampling.lhs import iman_conover_transform, maximin_lhs

# =============================================================================
# Closed-Form Solutions
# =============================================================================


def lognormal_params(auc: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Closed-form: pos_mu = sigma * sqrt(2) * Φ⁻¹(AUC)
    """
    return sigma * np.sqrt(2) * stats.norm.ppf(auc)


def hetero_gaussian_params(auc: np.ndarray, sigma_ratio: np.ndarray) -> np.ndarray:
    """
    Closed-form: delta_mu = sqrt(sigma_ratio² + 1) * Φ⁻¹(AUC)
    """
    return np.sqrt(sigma_ratio**2 + 1) * stats.norm.ppf(auc)


def exponential_params(auc: np.ndarray, neg_rate: float = 1.0) -> np.ndarray:
    """
    Closed-form: pos_rate = neg_rate * (1/AUC - 1)
    """
    return neg_rate * (1.0 / auc - 1.0)


# =============================================================================
# Numerical Solutions with Caching
# =============================================================================


class StudentTSolver:
    """Efficient solver for Student's t DGP parameters."""

    def __init__(self, n_fpr: int = 201):
        self.fpr = np.linspace(1e-10, 1 - 1e-10, n_fpr)
        self._cache = {}

    def _compute_auc(self, df: float, delta: float, scale: float = 1.0) -> float:
        """Compute AUC for given parameters."""
        neg_dist = stats.t(df=df, loc=0, scale=scale)
        pos_dist = stats.t(df=df, loc=delta, scale=scale)
        thresholds = neg_dist.ppf(1 - self.fpr)
        tpr = pos_dist.sf(thresholds)
        return np.trapezoid(tpr, self.fpr)

    def solve(
        self, df: float, target_auc: float, delta_bounds: tuple = (0.001, 20.0)
    ) -> float:
        """Solve for delta_loc given df and target AUC."""

        def objective(delta):
            return self._compute_auc(df, delta) - target_auc

        # Check if solution exists in bounds
        auc_low = self._compute_auc(df, delta_bounds[0])
        auc_high = self._compute_auc(df, delta_bounds[1])

        if target_auc < auc_low or target_auc > auc_high:
            warnings.warn(
                f"Target AUC {target_auc} outside achievable range "
                f"[{auc_low:.3f}, {auc_high:.3f}] for df={df}"
            )
            return np.nan

        return brentq(objective, delta_bounds[0], delta_bounds[1], xtol=1e-6)

    def solve_batch(self, df_array: np.ndarray, auc_array: np.ndarray) -> np.ndarray:
        """Solve for multiple (df, AUC) pairs."""
        return np.array([self.solve(df, auc) for df, auc in zip(df_array, auc_array)])


class BetaOpposingSolver:
    """Efficient solver for Beta opposing skew DGP."""

    def __init__(self, n_fpr: int = 201):
        self.fpr = np.linspace(1e-10, 1 - 1e-10, n_fpr)

    def _compute_auc(self, alpha: float, beta: float) -> float:
        """Compute AUC for Beta(alpha, beta) vs Beta(beta, alpha)."""
        from scipy.special import betainc, betaincinv

        tpr = np.zeros_like(self.fpr)
        for i, f in enumerate(self.fpr):
            t = betaincinv(alpha, beta, 1 - f)
            tpr[i] = 1 - betainc(beta, alpha, t)

        return np.trapezoid(tpr, self.fpr)

    def solve(
        self, alpha: float, target_auc: float, beta_bounds: tuple = (0.1, 50.0)
    ) -> float:
        """Solve for beta given alpha and target AUC."""

        def objective(beta):
            return self._compute_auc(alpha, beta) - target_auc

        return brentq(objective, beta_bounds[0], beta_bounds[1], xtol=1e-6)

    def solve_batch(self, alpha_array: np.ndarray, auc_array: np.ndarray) -> np.ndarray:
        """Solve for multiple (alpha, AUC) pairs."""
        return np.array(
            [self.solve(alpha, auc) for alpha, auc in zip(alpha_array, auc_array)]
        )


class BimodalNegativeSolver:
    """Solver for bimodal negative mixture DGP."""

    def __init__(self, n_fpr: int = 201):
        self.fpr = np.linspace(1e-10, 1 - 1e-10, n_fpr)

    def _mixture_sf(
        self, t: np.ndarray, means: list, stds: list, weights: list
    ) -> np.ndarray:
        """Survival function of Gaussian mixture."""
        return sum(
            w * stats.norm.sf(t, loc=m, scale=s)
            for w, m, s in zip(weights, means, stds)
        )

    def _compute_auc(
        self,
        neg_means: list,
        neg_weights: list,
        pos_mean: float,
        neg_stds: list = None,
        pos_std: float = 1.0,
    ) -> float:
        """Compute AUC for mixture negative vs Gaussian positive."""
        if neg_stds is None:
            neg_stds = [1.0] * len(neg_means)

        # Find thresholds for each FPR
        all_means = neg_means + [pos_mean]
        t_min = min(all_means) - 5
        t_max = max(all_means) + 5

        tpr = np.zeros_like(self.fpr)
        for i, f in enumerate(self.fpr):
            try:
                t = brentq(
                    lambda t: self._mixture_sf(t, neg_means, neg_stds, neg_weights) - f,
                    t_min,
                    t_max,
                )
                tpr[i] = stats.norm.sf(t, loc=pos_mean, scale=pos_std)
            except ValueError:
                tpr[i] = np.nan

        return np.trapezoid(tpr, self.fpr)

    def solve(
        self,
        mixture_weight: float,
        mode_separation: float,
        target_auc: float,
        pos_mean_bounds: tuple = (-2.0, 10.0),
    ) -> float:
        """
        Solve for pos_mean given mixture parameters and target AUC.

        neg_means = [0, mode_separation]
        neg_weights = [mixture_weight, 1 - mixture_weight]
        """
        neg_means = [0.0, mode_separation]
        neg_weights = [mixture_weight, 1 - mixture_weight]

        def objective(pos_mean):
            return self._compute_auc(neg_means, neg_weights, pos_mean) - target_auc

        return brentq(objective, pos_mean_bounds[0], pos_mean_bounds[1], xtol=1e-6)


# =============================================================================
# Unified Interface
# =============================================================================


@dataclass
class LHSSample:
    """Container for LHS samples with DGP parameters."""

    dgp_type: str
    n_samples: int
    lhs_params: dict  # The sampled LHS parameters
    dgp_params: dict  # The solved DGP parameters
    target_auc: np.ndarray  # For verification
    achieved_auc: np.ndarray = None  # Optional verification


def map_lhs_to_dgp(dgp_type: str, lhs_params: dict) -> dict:
    """
    Map LHS parameters to DGP parameters.

    Parameters
    ----------
    dgp_type : str
        One of: 'lognormal', 'hetero_gaussian', 'beta_opposing',
                'student_t', 'bimodal_negative', 'exponential'
    lhs_params : dict
        Must contain 'auc' and DGP-specific shape parameters.

    Returns
    -------
    dict : DGP parameters ready for the make_*_dgp functions
    """
    auc = np.asarray(lhs_params["auc"])

    if dgp_type == "lognormal":
        sigma = np.asarray(lhs_params["sigma"])
        pos_mu = lognormal_params(auc, sigma)
        return {"neg_mu": 0.0, "pos_mu": pos_mu, "sigma": sigma}

    elif dgp_type == "hetero_gaussian":
        sigma_ratio = np.asarray(lhs_params["sigma_ratio"])
        delta_mu = hetero_gaussian_params(auc, sigma_ratio)
        return {"delta_mu": delta_mu, "sigma_neg": 1.0, "sigma_pos": sigma_ratio}

    elif dgp_type == "exponential":
        neg_rate = lhs_params.get("neg_rate", 1.0)
        pos_rate = exponential_params(auc, neg_rate)
        return {"neg_rate": neg_rate, "pos_rate": pos_rate}

    elif dgp_type == "student_t":
        df = np.asarray(lhs_params["df"])
        solver = StudentTSolver()
        delta_loc = solver.solve_batch(df, auc)
        return {"df": df, "delta_loc": delta_loc, "scale": 1.0}

    elif dgp_type == "beta_opposing":
        alpha = np.asarray(lhs_params["alpha"])
        solver = BetaOpposingSolver()
        beta = solver.solve_batch(alpha, auc)
        return {"alpha": alpha, "beta": beta}

    elif dgp_type == "bimodal_negative":
        mixture_weight = np.asarray(lhs_params["mixture_weight"])
        mode_separation = np.asarray(lhs_params["mode_separation"])
        solver = BimodalNegativeSolver()
        pos_mean = np.array(
            [
                solver.solve(w, sep, a)
                for w, sep, a in zip(mixture_weight, mode_separation, auc)
            ]
        )
        return {
            "neg_means": [(0.0, sep) for sep in mode_separation],
            "neg_weights": [(w, 1 - w) for w in mixture_weight],
            "pos_mean": pos_mean,
        }

    else:
        raise ValueError(f"Unknown DGP type: {dgp_type}")


# =============================================================================
# Example: Generate Full Design
# =============================================================================


def generate_simulation_design(n_samples: int = 1000, seed: int = 42):
    """
    Generate LHS designs that are:
    1. Space-filling (maximin)
    2. Orthogonal (uncorrelated columns)
    3. Suitable for main effect + interaction estimation
    """
    rng = np.random.default_rng(seed)
    designs = {}

    specs = [
        ("lognormal", ["auc", "sigma"], [(0.55, 0.99), (0.1, 3.0)]),
        ("hetero_gaussian", ["auc", "sigma_ratio"], [(0.55, 0.99), (0.2, 5.0)]),
        ("student_t", ["auc", "df"], [(0.55, 0.99), (1.1, 30.0)]),
        ("beta_opposing", ["auc", "alpha"], [(0.55, 0.99), (0.5, 10.0)]),
        (
            "bimodal_negative",
            ["auc", "mixture_weight", "mode_separation"],
            [(0.55, 0.99), (0.1, 0.9), (0.1, 4.0)],
        ),
        ("exponential", ["auc", "neg_rate"], [(0.55, 0.99), (0.1, 10.0)]),
    ]

    for dgp_type, param_names, param_bounds in specs:
        print(f"\n=== {dgp_type} ===")
        n_dims = len(param_names)

        # Step 1: Generate initial maximin LHS
        lhs_unit = maximin_lhs(
            n=n_samples, k=n_dims, method="build", dup=5, seed=rng.integers(0, 2**31)
        )

        # Step 2: Apply Iman-Conover to remove correlations
        lhs_orthogonal = iman_conover_transform(
            lhs_unit, target_corr=np.eye(n_dims), rng=rng
        )

        # Step 3: Verify properties
        if n_dims > 1:
            corr = np.corrcoef(lhs_orthogonal.T)
            np.fill_diagonal(corr, 0)
            max_corr = np.max(np.abs(corr))
            print(f"  Max |correlation|: {max_corr:.4f}")

        # Step 4: Scale to parameter bounds
        lower = np.array([b[0] for b in param_bounds])
        upper = np.array([b[1] for b in param_bounds])
        lhs_scaled = lower + lhs_orthogonal * (upper - lower)

        # Create param dict
        lhs_params = {name: lhs_scaled[:, i] for i, name in enumerate(param_names)}

        # Map to DGP params
        dgp_params = map_lhs_to_dgp(dgp_type, lhs_params)

        designs[dgp_type] = {"lhs_params": lhs_params, "dgp_params": dgp_params}

    return designs
