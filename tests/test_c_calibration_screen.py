"""Tests for the decision-first C-calibration screen."""

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts" / "c_calibration")
)

import check_screen  # noqa: E402
import design  # noqa: E402


def _summary(cell: design.Cell, *, c_star: float = 1.5, c_se: float = 0.05) -> dict:
    """Build a schema-faithful screening summary for one designed cell.

    Args:
        cell: Screening cell whose metadata is represented.
        c_star: Synthetic calibrated exponent.
        c_se: Synthetic bootstrap standard error.

    Returns:
        Minimal aggregate payload consumed by ``check_screen.evaluate``.
    """
    return {
        "meta": {
            "cell": {
                "name": cell.name,
                "arm": cell.arm,
                "shape": cell.shape,
                "n0": cell.n0,
                "n1": cell.n1,
            }
        },
        "aggregate": {
            "ladder": [1, 5],
            "area_by_j": [1.0, 0.9],
            "per_alpha": {
                "0.05": {
                    "infeasible": False,
                    "saturated": False,
                    "unconstrained": False,
                    "j_star": 5,
                    "c_star": c_star,
                    "c_star_ci": {"se": c_se},
                }
            },
            "ref_maps": [{"label": "c1", "alpha": 0.05, "area": 1.0}],
        },
    }


def test_complete_useful_screen_proceeds_and_routes_remaining_work():
    """A positive screen should recommend a reduced, question-led Stage A."""
    summaries = [_summary(cell) for cell in design.screening_cells()]
    result = check_screen.evaluate(summaries)
    assert result["verdict"] == "PROCEED to a reduced Stage A map fit"
    assert result["completeness"] == {"shape": True, "taper": True, "imbalance": True}
    assert "focused large-n" in result["recommendations"]["large_n"]
    assert "min(n0,n1)" in result["recommendations"]["imbalance"]


def test_partial_screen_is_inconclusive_instead_of_selection_biased():
    """An excluded or missing limiting cell must prevent a proceed verdict."""
    summaries = [_summary(cell) for cell in design.screening_cells()][1:]
    result = check_screen.evaluate(summaries)
    assert result["verdict"].startswith("INCONCLUSIVE")
    assert not all(result["completeness"].values())


def test_strong_imbalance_signal_preserves_the_directional_arm():
    """A resolved C* spread should route Stage A toward a directional rule."""
    summaries = []
    for cell in design.screening_cells():
        c_star = 1.5
        if cell.arm == "screen_imbalance" and cell.n0 > cell.n1:
            c_star = 1.1
        summaries.append(_summary(cell, c_star=c_star, c_se=0.02))
    result = check_screen.evaluate(summaries)
    assert "directional or 2-D" in result["recommendations"]["imbalance"]
    contrasts = result["imbalance_screen"]["binormal_90"][
        "orientation_contrasts"
    ]
    assert all(contrast["negative_minus_positive_c_star"] < 0 for contrast in contrasts)


def test_taper_diagnostic_requires_a_confidence_resolved_decrease():
    """Endpoint noise must be reflected in the reported taper conclusion."""
    rows = [
        {
            "shape": "t2_95",
            "n0": 100,
            "c_star": 1.8,
            "c_se": 0.05,
            "c_lower_1se": 1.75,
        },
        {
            "shape": "t2_95",
            "n0": 50_000,
            "c_star": 1.3,
            "c_se": 0.05,
            "c_lower_1se": 1.25,
        },
    ]
    diagnostic = check_screen.taper_diagnostics(rows)["t2_95"]
    assert diagnostic["resolved_decrease"]
    assert diagnostic["decrease_lower_95"] > 0.3
