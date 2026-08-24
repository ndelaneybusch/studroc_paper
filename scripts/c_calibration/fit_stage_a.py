"""Stage A fitting protocol: the pre-registered D1-D6 decisions.

Consumes the Stage A cell summaries produced by ``run.py --stage A`` and
mechanically applies the decision rules of ``stats/c_calibration_spec.md``
sections 2 and 6:

- **D1 (coordinate):** among C*, alpha_eff*, and ell*, pick the coordinate
  with the smallest relative dispersion across fitting shapes at fixed
  (n, alpha) and the smoothest n-trend.
- **D2 (imbalance reduction):** min(n0, n1) vs harmonic mean, accepted if
  the implied coverage error at alpha = .05 stays within 1pp across the
  imbalance cells (via the section 7.1 erosion law); otherwise flag a 2-D
  map.
- **D3 (taper family):** pure power vs power+plateau vs log-decay on the
  per-shape surpluses, selected by leave-one-n-out prediction error.
- **D4 (alpha drift):** separable delta0(alpha) * f(n) accepted if the
  joint-fit residuals stay within the bootstrap noise floor.
- **D5 (envelope):** pointwise minimum over fitting shapes minus one
  bootstrap SE (the 10th-percentile variant is reported alongside);
  dominance by a single shape is reported. Any C* < 1 escalates (the floor
  conjecture would be falsified; A4).
- **D6 (degenerate cells):** saturated / infeasible / never-dipping cells
  are excluded from every fit and reported separately.

Outputs a *proposed* frozen map (``frozen_map.json``) and a markdown report
(``stage_a_fit_report.md``). Nothing here launches Stage B: the proposal is
reviewed by a human first (deviations from the mechanical rules must be
justified in ``stats/c_calibration_report.md``).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))

from design import CORE_N, CORE_SHAPES, PROVISIONAL_C_MAX  # noqa: E402
from map_eval import SCHEMA_ID, validate_artifact  # noqa: E402
from runner import provenance  # noqa: E402

D2_COVERAGE_TOL_PP = 1.0  # acceptance rule for a 1-D n_eff reduction
ENVELOPE_ALPHAS = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_summaries(stage_a_dir: Path) -> list[dict]:
    """All Stage A cell summaries, keyed structure preserved."""
    out = []
    for path in sorted(stage_a_dir.glob("*.summary.json")):
        out.append(json.loads(path.read_text()))
    if not out:
        raise FileNotFoundError(f"No cell summaries under {stage_a_dir}")
    return out


def estimate_rows(summaries: list[dict]) -> list[dict]:
    """Flatten summaries to one row per (cell, alpha) with exclusion flags."""
    rows = []
    for summary in summaries:
        cell = summary["meta"]["cell"]
        for alpha_key, est in summary["aggregate"]["per_alpha"].items():
            rows.append(
                {
                    "cell": cell["name"],
                    "arm": cell["arm"],
                    "shape": cell["shape"],
                    "n0": cell["n0"],
                    "n1": cell["n1"],
                    "n_min": min(cell["n0"], cell["n1"]),
                    "n_harm": 2 * cell["n0"] * cell["n1"] / (cell["n0"] + cell["n1"]),
                    "alpha": float(alpha_key),
                    "excluded": bool(
                        est["infeasible"] or est["saturated"] or est["unconstrained"]
                    ),
                    "exclusion": (
                        "infeasible"
                        if est["infeasible"]
                        else "saturated"
                        if est["saturated"]
                        else "unconstrained(D6)"
                        if est["unconstrained"]
                        else ""
                    ),
                    "j_star": est["j_star"],
                    "c_star": est["c_star"],
                    "c_se": est["c_star_ci"]["se"],
                    "aeff_star": est["alpha_eff_star"],
                    "aeff_se": est["alpha_eff_star_ci"]["se"],
                    "ell_star": est["ell_star"],
                    "ell_se": est["ell_star_ci"]["se"],
                    "allowance_attribution": est.get("allowance_attribution"),
                }
            )
    return rows


def _core_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["arm"] == "core" and not r["excluded"]]


# ---------------------------------------------------------------------------
# D1: coordinate choice
# ---------------------------------------------------------------------------


def d1_coordinate(rows: list[dict]) -> dict:
    """Relative dispersion across shapes at fixed (n, alpha), per coordinate,
    plus n-trend smoothness of the shape-median coordinate."""
    core = _core_rows(rows)
    coords = {
        "C": lambda r: r["c_star"] - 1.0,
        "alpha_eff": lambda r: r["aeff_star"] - r["alpha"],
        "local_level": lambda r: r["ell_star"],
    }
    dispersion: dict[str, list[float]] = {k: [] for k in coords}
    medians: dict[str, dict] = {k: {} for k in coords}
    for n in CORE_N:
        for alpha in {r["alpha"] for r in core}:
            group = [r for r in core if r["n0"] == n and r["alpha"] == alpha]
            if len(group) < 3:
                continue
            for name, fn in coords.items():
                vals = np.array([fn(r) for r in group], dtype=np.float64)
                med = float(np.median(vals))
                if abs(med) > 1e-9:
                    dispersion[name].append(
                        float(
                            (np.quantile(vals, 0.9) - np.quantile(vals, 0.1))
                            / abs(med)
                        )
                    )
                medians[name][(n, alpha)] = med

    def smoothness(name: str) -> float:
        # Mean |second difference| of median vs log n, averaged over alphas.
        by_alpha: dict[float, list[tuple[float, float]]] = {}
        for (n, alpha), med in medians[name].items():
            by_alpha.setdefault(alpha, []).append((np.log(n), med))
        curvs = []
        for pts in by_alpha.values():
            pts.sort()
            if len(pts) < 3:
                continue
            y = np.array([p[1] for p in pts])
            scale = max(abs(np.median(y)), 1e-9)
            curvs.append(float(np.mean(np.abs(np.diff(y, 2))) / scale))
        return float(np.mean(curvs)) if curvs else float("nan")

    table = {
        name: {
            "median_rel_dispersion": float(np.median(disp)) if disp else float("nan"),
            "n_trend_curvature": smoothness(name),
        }
        for name, disp in dispersion.items()
    }
    ranked = sorted(
        table,
        key=lambda k: (
            table[k]["median_rel_dispersion"],
            table[k]["n_trend_curvature"],
        ),
    )
    return {"table": table, "winner": ranked[0], "ranking": ranked}


# ---------------------------------------------------------------------------
# surpluses in the winning coordinate
# ---------------------------------------------------------------------------


def surplus_of(row: dict, coordinate: str) -> float | None:
    if row["c_star"] is None:
        return None
    if coordinate == "C":
        return row["c_star"] - 1.0
    if coordinate == "alpha_eff":
        return row["aeff_star"] - row["alpha"]
    return row["ell_star"]  # local_level: fit the level itself


def surplus_se_of(row: dict, coordinate: str) -> float | None:
    key = {"C": "c_se", "alpha_eff": "aeff_se", "local_level": "ell_se"}[coordinate]
    return row[key]


# ---------------------------------------------------------------------------
# D3: taper family by leave-one-n-out
# ---------------------------------------------------------------------------


def _taper_models():
    def power(n, d0, g):
        return d0 * (n / 500.0) ** (-g)

    def power_plateau(n, d_inf, d0, g):
        return d_inf + d0 * (n / 500.0) ** (-g)

    def log_decay(n, d0, b):
        return d0 / (1.0 + b * np.maximum(np.log(n / 500.0), 0.0))

    return {
        "power": (power, ([1e-6, 1e-3], [10.0, 3.0]), [1.0, 0.32]),
        "power_plateau": (
            power_plateau,
            ([0.0, 1e-6, 1e-3], [5.0, 10.0, 3.0]),
            [0.1, 1.0, 0.32],
        ),
        "log_decay": (log_decay, ([1e-6, 1e-3], [10.0, 50.0]), [1.0, 1.0]),
    }


def _fit_taper(family: str, n: np.ndarray, y: np.ndarray, sigma: np.ndarray):
    fn, bounds, p0 = _taper_models()[family]
    popt, _ = curve_fit(
        fn, n, y, p0=p0, sigma=sigma, bounds=bounds, maxfev=20_000, absolute_sigma=True
    )
    return fn, popt


def d3_taper(rows: list[dict], coordinate: str) -> dict:
    """Per-(shape, alpha) leave-one-n-out errors for the three families,
    pooled; includes the large-n arm so the tail-alpha decay is in the fit."""
    fit_rows = [
        r
        for r in rows
        if r["arm"] in ("core", "large_n")
        and not r["excluded"]
        and r["c_star"] is not None
    ]
    loo: dict[str, list[float]] = {f: [] for f in _taper_models()}
    fits_used = 0
    for shape in CORE_SHAPES:
        for alpha in sorted({r["alpha"] for r in fit_rows}):
            grp = sorted(
                (r for r in fit_rows if r["shape"] == shape and r["alpha"] == alpha),
                key=lambda r: r["n_min"],
            )
            if len(grp) < 5:
                continue
            n = np.array([r["n_min"] for r in grp], dtype=np.float64)
            y = np.array([surplus_of(r, coordinate) for r in grp], dtype=np.float64)
            sig = np.array(
                [surplus_se_of(r, coordinate) or 0.1 for r in grp], dtype=np.float64
            )
            sig = np.maximum(sig, 1e-4)
            fits_used += 1
            for family in loo:
                errs = []
                for hold in range(len(grp)):
                    keep = np.arange(len(grp)) != hold
                    try:
                        fn, popt = _fit_taper(family, n[keep], y[keep], sig[keep])
                        pred = fn(np.array([n[hold]]), *popt)[0]
                        errs.append(((pred - y[hold]) / sig[hold]) ** 2)
                    except (RuntimeError, ValueError):
                        errs.append(np.nan)
                if errs:
                    loo[family].append(float(np.nanmean(errs)))
    table = {
        family: {
            "mean_loo_scaled_sqerr": float(np.nanmean(v)) if v else float("nan"),
            "n_series": len(v),
        }
        for family, v in loo.items()
    }
    winner = min(table, key=lambda f: table[f]["mean_loo_scaled_sqerr"])
    return {"table": table, "winner": winner, "series_fitted": fits_used}


# ---------------------------------------------------------------------------
# D5: shape envelope
# ---------------------------------------------------------------------------


def d5_envelope(rows: list[dict], coordinate: str) -> dict:
    """Pointwise envelope over fitting shapes at each (n, alpha)."""
    core = [r for r in _core_rows(rows) if r["c_star"] is not None]
    points = []
    dominance: dict[str, int] = {}
    floor_violations = []
    for n in sorted({r["n0"] for r in core}):
        for alpha in sorted({r["alpha"] for r in core}):
            grp = [r for r in core if r["n0"] == n and r["alpha"] == alpha]
            if len(grp) < 3:
                continue
            vals = np.array([surplus_of(r, coordinate) for r in grp])
            ses = np.array([surplus_se_of(r, coordinate) or 0.0 for r in grp])
            i_min = int(np.argmin(vals))
            env_min_se = float(vals[i_min] - ses[i_min])
            env_q10 = float(np.quantile(vals, 0.10) - np.median(ses))
            dominance[grp[i_min]["shape"]] = dominance.get(grp[i_min]["shape"], 0) + 1
            points.append(
                {
                    "n": n,
                    "alpha": alpha,
                    "envelope_min_minus_se": env_min_se,
                    "envelope_q10_minus_se": env_q10,
                    "min_shape": grp[i_min]["shape"],
                    "n_shapes": len(grp),
                }
            )
            for r in grp:
                if coordinate == "C" and r["c_star"] < 1.0:
                    floor_violations.append(
                        {"cell": r["cell"], "alpha": alpha, "c_star": r["c_star"]}
                    )
    return {
        "points": points,
        "dominance": dominance,
        "floor_violations": floor_violations,
    }


def fit_envelope_taper(
    envelope: dict, family: str
) -> tuple[dict[str, float], float, dict[str, float]]:
    """Fit the selected taper family per alpha on the envelope points with a
    shared decay parameter (D4 separable form), returning
    (delta0_by_alpha, shared_gamma_or_b, c_max_by_alpha)."""
    pts = envelope["points"]
    alphas = sorted({p["alpha"] for p in pts})
    # Stage 1: per-alpha independent fits to seed the shared parameter.
    fn, _, _ = _taper_models()[family]
    per_alpha_fits = {}
    for alpha in alphas:
        sub = [p for p in pts if p["alpha"] == alpha]
        n = np.array([p["n"] for p in sub], dtype=np.float64)
        y = np.maximum(np.array([p["envelope_min_minus_se"] for p in sub]), 1e-4)
        try:
            _, popt = _fit_taper(family, n, y, np.full_like(y, 0.05))
            per_alpha_fits[alpha] = popt
        except (RuntimeError, ValueError):
            continue
    if not per_alpha_fits:
        raise RuntimeError("Envelope taper fit failed at every alpha")
    decay_idx = -1  # gamma (power / power_plateau) or b (log_decay)
    shared_decay = float(np.median([p[decay_idx] for p in per_alpha_fits.values()]))

    # Stage 2: refit the amplitude per alpha at the shared decay.
    delta0_by_alpha = {}
    c_max_by_alpha = {}
    for alpha in alphas:
        sub = [p for p in pts if p["alpha"] == alpha]
        n = np.array([p["n"] for p in sub], dtype=np.float64)
        y = np.maximum(np.array([p["envelope_min_minus_se"] for p in sub]), 1e-4)
        if family == "power":
            basis = (n / 500.0) ** (-shared_decay)
            d0 = float(np.sum(basis * y) / np.sum(basis**2))
            delta0_by_alpha[f"{alpha:g}"] = d0
            small_n = float(np.min(n))
            c_max_by_alpha[f"{alpha:g}"] = float(
                min(1.0 + d0 * (small_n / 500.0) ** (-shared_decay), PROVISIONAL_C_MAX)
            )
        else:
            popt = per_alpha_fits.get(alpha)
            if popt is None:
                continue
            delta0_by_alpha[f"{alpha:g}"] = float(popt[-2])
            c_max_by_alpha[f"{alpha:g}"] = float(
                min(1.0 + fn(np.min(n), *popt), PROVISIONAL_C_MAX)
            )
    return delta0_by_alpha, shared_decay, c_max_by_alpha


# ---------------------------------------------------------------------------
# D2: imbalance reduction
# ---------------------------------------------------------------------------


def d2_reduction(rows: list[dict], coordinate: str) -> dict:
    """Test min vs harmonic n_eff against the imbalance arm via the erosion
    law: predicted coverage error at alpha = .05 must stay within 1pp."""
    core = [r for r in _core_rows(rows) if r["c_star"] is not None]
    imb = [
        r
        for r in rows
        if r["arm"] == "imbalance"
        and not r["excluded"]
        and r["alpha"] == 0.05
        and r["c_star"] is not None
    ]
    results = {}
    for reduction, key in (("min", "n_min"), ("harmonic", "n_harm")):
        errors = []
        for r in imb:
            ref = sorted(
                (c for c in core if c["shape"] == r["shape"] and c["alpha"] == 0.05),
                key=lambda c: c["n_min"],
            )
            if len(ref) < 2:
                continue
            xs = np.log([c["n_min"] for c in ref])
            ys = [surplus_of(c, coordinate) for c in ref]
            pred_surplus = float(np.interp(np.log(r[key]), xs, ys))
            c_pred = 1.0 + pred_surplus if coordinate == "C" else None
            if coordinate == "alpha_eff":
                aeff = r["alpha"] + pred_surplus
                c_pred = float(np.log1p(-aeff) / np.log1p(-r["alpha"]))
            if c_pred is None:
                continue  # local_level: no erosion-law mapping; 2-D check only
            # Erosion law: coverage(C) = (1 - alpha)^(C / C*_cell).
            cov_pred = (1.0 - r["alpha"]) ** (c_pred / r["c_star"])
            errors.append(
                {
                    "cell": r["cell"],
                    "coverage_error_pp": 100.0 * (cov_pred - (1.0 - r["alpha"])),
                }
            )
        if errors:
            worst = max(abs(e["coverage_error_pp"]) for e in errors)
            # Only a coverage *deficit* (negative error) violates validity.
            worst_deficit = max(
                (-e["coverage_error_pp"] for e in errors), default=0.0
            )
            results[reduction] = {
                "cells": errors,
                "worst_abs_error_pp": worst,
                "worst_deficit_pp": worst_deficit,
                "accepted": worst_deficit <= D2_COVERAGE_TOL_PP,
            }
    accepted = [k for k, v in results.items() if v.get("accepted")]
    if accepted:
        winner = min(accepted, key=lambda k: results[k]["worst_deficit_pp"])
    else:
        winner = "table2d"
    return {"table": results, "winner": winner}


# ---------------------------------------------------------------------------
# D4: separability check
# ---------------------------------------------------------------------------


def d4_separability(
    envelope: dict, family: str, delta0_by_alpha: dict, shared_decay: float
) -> dict:
    """Residuals of the separable fit against the envelope points, in units
    of a nominal noise floor (the bootstrap-SE scale used in the envelope)."""
    fn, _, _ = _taper_models()[family]
    resid = []
    for p in envelope["points"]:
        key = f"{p['alpha']:g}"
        if key not in delta0_by_alpha:
            continue
        if family == "power":
            pred = delta0_by_alpha[key] * (p["n"] / 500.0) ** (-shared_decay)
        elif family == "log_decay":
            pred = delta0_by_alpha[key] / (
                1.0 + shared_decay * max(np.log(p["n"] / 500.0), 0.0)
            )
        else:
            continue
        resid.append(p["envelope_min_minus_se"] - pred)
    resid = np.asarray(resid)
    noise_floor = 0.1  # C-scale bootstrap-SE target of spec section 4
    return {
        "n_points": int(resid.size),
        "rms_residual": float(np.sqrt(np.mean(resid**2))) if resid.size else None,
        "max_abs_residual": float(np.max(np.abs(resid))) if resid.size else None,
        "noise_floor": noise_floor,
        "separable_accepted": bool(
            resid.size and np.sqrt(np.mean(resid**2)) <= noise_floor
        ),
    }


# ---------------------------------------------------------------------------
# freezing + report
# ---------------------------------------------------------------------------


def freeze(
    d1: dict, d2: dict, d3: dict, d5: dict, delta0_by_alpha: dict,
    shared_decay: float, c_max_by_alpha: dict, stage_a_dir: Path,
) -> dict:
    coordinate = d1["winner"]
    taper: dict = {
        "family": d3["winner"],
        "n_ref": 500,
        "delta0_by_alpha": delta0_by_alpha,
    }
    if d3["winner"] in ("power", "power_plateau"):
        taper["gamma"] = shared_decay
        if d3["winner"] == "power_plateau":
            taper["delta_inf_by_alpha"] = {k: 0.0 for k in delta0_by_alpha}
    else:
        taper["b"] = shared_decay
    ns = sorted({p["n"] for p in d5["points"]})
    artifact = {
        "schema": SCHEMA_ID,
        "coordinate": coordinate,
        "n_eff": {"reduction": d2["winner"], "table": None},
        "taper": taper,
        "c_max_by_alpha": c_max_by_alpha,
        "alpha_range": [0.01, 0.5],
        "n_range": [min(ns), 50_000],
        "provenance": {
            **provenance(),
            "stage_a_dir": str(stage_a_dir),
            "spec": "stats/c_calibration_spec.md",
            "decisions": {
                "D1": d1["winner"],
                "D2": d2["winner"],
                "D3": d3["winner"],
            },
        },
    }
    blockers = {}
    if d2["winner"] == "table2d":
        blockers["d2_table2d"] = (
            "no 1-D n_eff reduction met the 1pp rule; a 2-D interpolation "
            "table is needed (schema extension + human decision)"
        )
    if d5["floor_violations"]:
        blockers["floor_violations"] = d5["floor_violations"]
    if coordinate == "local_level":
        blockers["local_level_coordinate"] = (
            "D1 selected the local-level coordinate, which resolves to a "
            "fixed trim depth rather than an exponent; Stage B and the "
            "production wiring need the level-mode path (human decision "
            "before freezing)"
        )
    if blockers:
        artifact["provenance"]["blockers"] = blockers
    return artifact


def write_report(path: Path, sections: dict) -> None:
    lines = [
        "# Stage A fit report (mechanical, pre-review)",
        "",
        f"*Generated {time.strftime('%Y-%m-%d %H:%M')} by fit_stage_a.py. "
        "This is the mechanical application of the pre-registered D1-D6 "
        "rules; review before freezing and launching Stage B. The final "
        "report belongs in stats/c_calibration_report.md.*",
        "",
    ]
    for title, payload in sections.items():
        lines.append(f"## {title}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(payload, indent=1, default=float))
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Stage A fitting protocol")
    parser.add_argument(
        "--in",
        dest="stage_a_dir",
        type=Path,
        default=Path("data/results/c_calibration/stageA"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/results/c_calibration")
    )
    args = parser.parse_args(argv)

    summaries = load_summaries(args.stage_a_dir)
    rows = estimate_rows(summaries)
    excluded = [r for r in rows if r["excluded"]]
    print(f"{len(rows)} (cell, alpha) points, {len(excluded)} excluded (D6/saturation)")

    d1 = d1_coordinate(rows)
    print(f"D1 coordinate winner: {d1['winner']}")
    coordinate = d1["winner"]

    d3 = d3_taper(rows, coordinate)
    print(f"D3 taper winner: {d3['winner']}")

    d5 = d5_envelope(rows, coordinate)
    if d5["floor_violations"]:
        print(
            f"D5 FLOOR VIOLATIONS ({len(d5['floor_violations'])} points): "
            "C* < 1 measured — ship C = 1 and escalate (A4)."
        )

    delta0_by_alpha, shared_decay, c_max_by_alpha = fit_envelope_taper(d5, d3["winner"])
    d4 = d4_separability(d5, d3["winner"], delta0_by_alpha, shared_decay)
    print(f"D4 separable accepted: {d4['separable_accepted']}")

    d2 = d2_reduction(rows, coordinate)
    print(f"D2 reduction winner: {d2['winner']}")

    artifact = freeze(
        d1, d2, d3, d5, delta0_by_alpha, shared_decay, c_max_by_alpha, args.stage_a_dir
    )
    validate_artifact(artifact)
    for key, val in artifact["provenance"].get("blockers", {}).items():
        print(f"BLOCKER [{key}]: {val}")
    args.out.mkdir(parents=True, exist_ok=True)
    map_path = args.out / "frozen_map.json"
    map_path.write_text(json.dumps(artifact, indent=1))
    write_report(
        args.out / "stage_a_fit_report.md",
        {
            "Exclusions (D6, saturation, infeasible)": excluded,
            "D1 coordinate": d1,
            "D2 imbalance reduction": d2,
            "D3 taper family": d3,
            "D4 separability": d4,
            "D5 envelope": d5,
            "Proposed frozen map": artifact,
        },
    )
    print(f"Proposed frozen map -> {map_path}")
    print(f"Fit report -> {args.out / 'stage_a_fit_report.md'}")
    print(
        "\nNEXT: review the report, adjust/bless the map, then run\n"
        "  uv run python scripts/c_calibration/run.py --stage B "
        f"--map {map_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
