"""Frozen calibration-map artifact: schema, validation, and resolution.

The Stage A fit (`fit_stage_a.py`) freezes its result as a versioned JSON
artifact; Stage B (and, if the acceptance criteria pass, the production
``trim_exponent="auto"`` wiring) resolves the trim coordinate from it. The
schema is fixed here, before the study runs, so the confirmation arm and the
shipped behavior consume the exact same object.

Schema ``c-calibration-map/v1``::

    {
      "schema": "c-calibration-map/v1",
      "coordinate": "C" | "alpha_eff" | "local_level",
      "n_eff": {"reduction": "min" | "harmonic" | "table2d",
                "table": {...} | null},
      "taper": {"family": "power" | "power_plateau" | "log_decay",
                "n_ref": 500,
                "delta0_by_alpha": {"0.05": 0.8, ...},
                "gamma": 0.32,            # power / power_plateau
                "delta_inf_by_alpha": {...},  # power_plateau only
                "b": 1.0},               # log_decay only
      "c_max_by_alpha": {"0.05": 3.2, ...},
      "alpha_range": [0.01, 0.5],
      "n_range": [25, 50000],
      "provenance": {"git_hash": ..., "created": ..., "stage_a_dir": ...,
                     "spec": "stats/c_calibration_spec.md", ...}
    }

Behavior contracts (spec section 2): the resolved coordinate is floored at
C = 1 always; n beyond the calibrated range follows the fitted taper
(monotone toward its own limit, clamped at the small-n end); alpha outside
``alpha_range`` falls back to C = 1 with a warning.
"""

import json
import warnings
from pathlib import Path

import numpy as np

SCHEMA_ID = "c-calibration-map/v1"
COORDINATES = ("C", "alpha_eff", "local_level")
REDUCTIONS = ("min", "harmonic", "table2d")
TAPER_FAMILIES = ("power", "power_plateau", "log_decay")


def validate_artifact(artifact: dict) -> None:
    """Raise ``ValueError`` if the artifact does not satisfy the v1 schema."""
    if artifact.get("schema") != SCHEMA_ID:
        raise ValueError(f"Unknown schema: {artifact.get('schema')!r}")
    if artifact["coordinate"] not in COORDINATES:
        raise ValueError(f"Unknown coordinate: {artifact['coordinate']!r}")
    if artifact["n_eff"]["reduction"] not in REDUCTIONS:
        raise ValueError(f"Unknown n_eff reduction: {artifact['n_eff']!r}")
    taper = artifact["taper"]
    if taper["family"] not in TAPER_FAMILIES:
        raise ValueError(f"Unknown taper family: {taper['family']!r}")
    if not taper["delta0_by_alpha"]:
        raise ValueError("taper.delta0_by_alpha must be non-empty")
    lo, hi = artifact["alpha_range"]
    if not 0.0 < lo < hi < 1.0:
        raise ValueError(f"Bad alpha_range: {artifact['alpha_range']!r}")
    n_lo, n_hi = artifact["n_range"]
    if not 1 <= n_lo < n_hi:
        raise ValueError(f"Bad n_range: {artifact['n_range']!r}")


def load_artifact(path: Path | str) -> dict:
    """Load and validate a frozen-map artifact."""
    artifact = json.loads(Path(path).read_text())
    validate_artifact(artifact)
    return artifact


def _interp_by_alpha(table: dict[str, float], alpha: float) -> float:
    """Interpolate an alpha-keyed constant table linearly in log(alpha)."""
    keys = np.array(sorted(float(k) for k in table))
    vals = np.array([table[k] for k in sorted(table, key=float)])
    return float(np.interp(np.log(alpha), np.log(keys), vals))


def _n_eff(artifact: dict, n0: int, n1: int) -> float:
    reduction = artifact["n_eff"]["reduction"]
    if reduction == "min":
        return float(min(n0, n1))
    if reduction == "harmonic":
        return 2.0 * n0 * n1 / (n0 + n1)
    raise ValueError(
        "table2d n_eff reduction requires resolving through the 2-D table; "
        "use resolve_surplus with explicit table support"
    )


def _taper_surplus(taper: dict, alpha: float, n_eff: float) -> float:
    """The fitted surplus delta(n, alpha) of the winning coordinate above
    its asymptote."""
    n_ref = taper.get("n_ref", 500.0)
    delta0 = _interp_by_alpha(taper["delta0_by_alpha"], alpha)
    x = n_eff / n_ref
    family = taper["family"]
    if family == "power":
        return delta0 * x ** (-taper["gamma"])
    if family == "power_plateau":
        delta_inf = _interp_by_alpha(taper["delta_inf_by_alpha"], alpha)
        return delta_inf + delta0 * x ** (-taper["gamma"])
    if family == "log_decay":
        return delta0 / (1.0 + taper["b"] * max(np.log(x), 0.0))
    raise ValueError(f"Unknown taper family: {family!r}")


def resolve_trim(artifact: dict, *, n0: int, n1: int, alpha: float) -> dict:
    """Resolve the frozen map at one (n0, n1, alpha) call.

    Args:
        artifact: A validated ``c-calibration-map/v1`` artifact.
        n0: Negative-class size.
        n1: Positive-class size.
        alpha: Nominal simultaneous level.

    Returns:
        ``{"mode": "exponent", "C": float}`` for the C / alpha_eff
        coordinates, or ``{"mode": "level", "ell": float}`` for the
        local-level coordinate (the caller converts the level to a fixed
        trim depth ``j = round(ell * (M + 1))``).
    """
    lo, hi = artifact["alpha_range"]
    if not lo <= alpha <= hi:
        warnings.warn(
            f"alpha={alpha} outside the calibrated range [{lo}, {hi}]; "
            "falling back to the conservative C = 1",
            stacklevel=2,
        )
        return {"mode": "exponent", "C": 1.0}

    n_eff = _n_eff(artifact, n0, n1)
    n_lo, _ = artifact["n_range"]
    # Below the calibrated range, clamp to the small-n end (the taper is
    # not validated there and extrapolating it upward is not safe).
    n_eff = max(n_eff, float(n_lo))
    surplus = _taper_surplus(artifact["taper"], alpha, n_eff)

    coordinate = artifact["coordinate"]
    if coordinate == "C":
        c = 1.0 + surplus
    elif coordinate == "alpha_eff":
        # Surplus is parametrized on the alpha_eff scale:
        # alpha_eff = alpha + surplus, converted back to the exponent knob.
        alpha_eff = float(np.clip(alpha + surplus, alpha, 1.0 - 1e-12))
        c = np.log1p(-alpha_eff) / np.log1p(-alpha)
    else:  # local_level
        ell0 = _interp_by_alpha(
            artifact["taper"].get("ell_ref_by_alpha", {"0.05": 0.0}), alpha
        )
        ell = max(ell0 - surplus, 1e-9)
        return {"mode": "level", "ell": ell}

    c_max = _interp_by_alpha(artifact["c_max_by_alpha"], alpha)
    return {"mode": "exponent", "C": float(np.clip(c, 1.0, c_max))}


def resolve_exponent(artifact: dict, *, n0: int, n1: int, alpha: float) -> float:
    """Resolve the map to a trim exponent, for exponent-coordinate maps.

    Raises:
        ValueError: If the map's coordinate is the local level (a fixed
            depth, not expressible as an exponent).
    """
    out = resolve_trim(artifact, n0=n0, n1=n1, alpha=alpha)
    if out["mode"] != "exponent":
        raise ValueError("Frozen map uses the local-level coordinate")
    return out["C"]


def placeholder_artifact() -> dict:
    """A schema-complete artifact with the provisional (round-3) constants.

    For tests and for exercising the Stage B machinery before the real fit
    exists. Never ship this: the constants are the pre-study provisional
    formula, not a calibrated map.
    """
    return {
        "schema": SCHEMA_ID,
        "coordinate": "C",
        "n_eff": {"reduction": "min", "table": None},
        "taper": {
            "family": "power",
            "n_ref": 500,
            "delta0_by_alpha": {"0.05": 0.8},
            "gamma": 0.32,
        },
        "c_max_by_alpha": {"0.05": 6.0},
        "alpha_range": [0.01, 0.5],
        "n_range": [25, 50_000],
        "provenance": {
            "placeholder": True,
            "note": "provisional round-3 constants; not a calibrated map",
        },
    }
