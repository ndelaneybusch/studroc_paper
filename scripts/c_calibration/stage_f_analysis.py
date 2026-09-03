"""Offline scoring and summaries for the fixed Stage F frontier rules.

All procedures are reconstructed from stored paired parent bands. Analysis
never regenerates a fiducial cloud and has no rule-fitting path.
"""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
from stage_f_core import (
    FrontierRule,
    ObservableSummary,
    decode_array,
    decode_band,
    decode_violations,
    encode_array,
    encode_violations,
    fixed_region_masks,
    frontier_region_masks,
    record_observables,
    stitch_hybrid,
    violation_mask,
)
from stage_f_design import STAGE_F_SEED, load_manifest

RULE_NAMES = (
    "probe_legacy",
    "probe_fpr",
    "count5",
    "frontier_run0",
    "frontier_j1",
    "frontier_floor_v1",
)
ALPHA2_KEYS = {0.05: ("0.05", "0.025"), 0.5: ("0.5", "0.25")}
RuleName = Literal[
    "probe_legacy",
    "probe_fpr",
    "count5",
    "frontier_run0",
    "frontier_j1",
    "frontier_floor_v1",
]
FixedRuleName = Literal["probe_legacy", "probe_fpr", "count5"]


@dataclass(frozen=True)
class CellRecords:
    """One cell's metadata and lossless replicate records."""

    meta: dict
    records: list[dict]

    @property
    def name(self) -> str:
        """Return the stable cell name."""
        return str(self.meta["cell"]["name"])

    @property
    def study(self) -> str:
        """Return the study identifier."""
        return str(self.meta["cell"]["study"])

    @property
    def source(self) -> str:
        """Return the design-source label."""
        return str(self.meta["cell"]["source"])


def load_cell_records(path: Path) -> CellRecords:
    """Load one compressed Stage F cell payload.

    Args:
        path: Compressed cell-record path.

    Returns:
        Validated cell metadata and records.
    """
    with gzip.open(path, "rt") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "stage-f-cell/v2":
        raise ValueError(f"Unknown Stage F cell schema in {path}")
    return CellRecords(meta=payload["meta"], records=payload["records"])


def load_study_records(root: Path, *, study: str) -> list[CellRecords]:
    """Load every completed cell for one study.

    Args:
        root: Stage F output root.
        study: Study identifier.

    Returns:
        Cell payloads sorted by path.
    """
    return [
        load_cell_records(path) for path in sorted((root / study).glob("*.json.gz"))
    ]


def load_complete_study(
    root: Path, *, study: Literal["A", "B", "C"]
) -> list[CellRecords]:
    """Load a study only when every manifest cell is complete.

    Args:
        root: Stage F output root containing manifests and cell directories.
        study: Frozen study identifier.

    Returns:
        Cell records in manifest order.

    Raises:
        RuntimeError: If any output is absent, truncated, duplicated, or
            inconsistent with the current manifest.
    """
    manifest, expected_cells = load_manifest(
        root / "manifests" / f"study_{study.lower()}.json"
    )
    loaded = load_study_records(root, study=study)
    by_name: dict[str, CellRecords] = {}
    for cell in loaded:
        if cell.name in by_name:
            raise RuntimeError(f"Duplicate Stage F cell output: {cell.name}")
        by_name[cell.name] = cell
    expected_names = {cell.name for cell in expected_cells}
    unexpected = sorted(set(by_name) - expected_names)
    if unexpected:
        raise RuntimeError(f"Unexpected Stage F {study} cell outputs: {unexpected}")
    ordered = []
    for expected, expected_payload in zip(
        expected_cells, manifest["cells"], strict=True
    ):
        cell = by_name.get(expected.name)
        if cell is None:
            raise RuntimeError(f"Missing Stage F {study} cell output: {expected.name}")
        if cell.meta.get("cell") != expected_payload:
            raise RuntimeError(
                f"Cell {expected.name} metadata differ from its manifest"
            )
        if len(cell.records) < expected.reps:
            raise RuntimeError(
                f"Cell {expected.name} has {len(cell.records)} records; "
                f"at least {expected.reps} are required"
            )
        ordered.append(cell)
    return ordered


def wilson_interval(
    successes: int, trials: int, *, z: float = 1.96
) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion.

    Args:
        successes: Number of successful trials.
        trials: Total number of trials.
        z: Normal critical value.

    Returns:
        Lower and upper score limits.
    """
    if trials == 0:
        return 0.0, 1.0
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2.0 * trials)) / denominator
    radius = (
        z
        * np.sqrt(proportion * (1.0 - proportion) / trials + z**2 / (4.0 * trials**2))
        / denominator
    )
    return max(0.0, float(center - radius)), min(1.0, float(center + radius))


def _masks_for_rule(
    rule: RuleName, *, observables: ObservableSummary, khat: np.ndarray, m_draws: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return left and right masks for one prespecified rule.

    Args:
        rule: Fixed rule identifier.
        observables: Rank-derived sample summary.
        khat: Empirical positive-count map.
        m_draws: Fiducial cloud size.

    Returns:
        Left and right component masks.
    """
    if rule.startswith("frontier_"):
        frontier_rule = cast(FrontierRule, rule)
        return frontier_region_masks(
            frontier_rule, observables=observables, khat=khat, m_draws=m_draws
        )
    fixed_rule = cast(FixedRuleName, rule)
    return fixed_region_masks(fixed_rule, observables=observables)


def score_record(
    record: dict, *, rule: RuleName, alpha: float, alpha2_key: str, m_draws: int
) -> dict:
    """Score one complete procedure from a stored paired replicate.

    Args:
        record: Lossless paired-parent record.
        rule: Prespecified region rule.
        alpha: Nominal fiducial level.
        alpha2_key: Stored M3 level key.
        m_draws: Fiducial cloud size.

    Returns:
        Replicate-level coverage, location, and width metrics.
    """
    nominal_key = f"{alpha:g}"
    observables = record_observables(record)
    khat = decode_array(record["khat"]).astype(np.int64)
    truth = decode_array(record["truth"]).astype(np.float64)
    fid_lower, fid_upper = decode_band(record["fiducial"][nominal_key])
    m3_lower, m3_upper = decode_band(record["m3"][alpha2_key])
    left, right = _masks_for_rule(
        rule, observables=observables, khat=khat, m_draws=m_draws
    )
    region = left | right
    closure = "legacy" if rule == "probe_legacy" else "widening"
    lower, upper = stitch_hybrid(
        fid_lower, fid_upper, m3_lower, m3_upper, region, closure=closure
    )
    fid_miss = decode_violations(record["fiducial_violations"][nominal_key])
    m3_miss = violation_mask(m3_lower, m3_upper, truth)
    hybrid_miss = violation_mask(lower, upper, truth)
    miss_low = lower > truth + 1e-12
    miss_high = truth > upper + 1e-12
    exterior = fid_miss & ~region
    adjacent = np.zeros_like(region)
    adjacent[1:] |= region[:-1]
    adjacent[:-1] |= region[1:]
    edge_escape = exterior & adjacent
    far_escape = exterior & ~adjacent

    def area_for(mask: np.ndarray) -> float:
        """Return mean width after applying one component mask."""
        component_lower, component_upper = stitch_hybrid(
            fid_lower, fid_upper, m3_lower, m3_upper, mask, closure=closure
        )
        return float(np.mean(component_upper - component_lower))

    fid_area = float(np.mean(fid_upper - fid_lower))
    m3_area = float(np.mean(m3_upper - m3_lower))
    left_area = area_for(left)
    right_area = area_for(right)
    area = float(np.mean(upper - lower))
    raw_lower = np.where(region, np.minimum(fid_lower, m3_lower), fid_lower)
    legacy_propagated = bool(
        rule == "probe_legacy"
        and np.any(region & (lower > truth + 1e-12) & (raw_lower <= truth + 1e-12))
    )
    positive_tail = observables.n1 - khat
    k_sat = int(np.flatnonzero(positive_tail == 0)[0])
    run_length = observables.n0 - k_sat
    depth = np.maximum(lower - truth, truth - upper)
    return {
        "covered": not bool(hybrid_miss.any()),
        "viol_low": bool(miss_low.any()),
        "viol_high": bool(miss_high.any()),
        "max_depth": max(0.0, float(depth.max(initial=0.0))),
        "violations": encode_violations(hybrid_miss),
        "fiducial_failed": bool(fid_miss.any()),
        "fiducial_covered": not bool(fid_miss.any()),
        "m3_covered": not bool(m3_miss.any()),
        "exterior_escape": bool(exterior.any()),
        "conditional_capture": bool(fid_miss.any()) and not bool(exterior.any()),
        "floor_region_failure": bool((hybrid_miss & region).any()),
        "edge_escape": bool(edge_escape.any()),
        "far_escape": bool(far_escape.any()),
        "legacy_propagated_inside": legacy_propagated,
        "area": area,
        "fiducial_area": fid_area,
        "m3_area": m3_area,
        "area_diff_vs_fiducial": area - fid_area,
        "area_ratio_vs_fiducial": area / fid_area if fid_area else 1.0,
        "area_diff_vs_m3": area - m3_area,
        "left_width_cost": left_area - fid_area,
        "right_width_cost": right_area - fid_area,
        "overlap_width_cost": left_area + right_area - area - fid_area,
        "region_fraction": float(region.mean()),
        "left_fraction": float(left.mean()),
        "right_fraction": float(right.mean()),
        "k_sat": k_sat,
        "run_length": run_length,
        "right_margin": math.ceil(2.0 * math.sqrt(max(run_length, 1))),
        "simulation_diagnostics": record.get("simulation_diagnostics", {}),
        "observables": record["observables"],
    }


def summarize_scores(scores: list[dict]) -> dict:
    """Aggregate replicate scores with paired width uncertainty.

    Args:
        scores: Nonempty sequence of replicate score mappings.

    Returns:
        Coverage, failure-location, and width summaries.
    """
    if not scores:
        raise ValueError("cannot summarize an empty score list")
    reps = len(scores)
    successes = sum(row["covered"] for row in scores)
    failures = sum(row["fiducial_failed"] for row in scores)
    captured = sum(row["conditional_capture"] for row in scores)
    numeric = (
        "area",
        "fiducial_area",
        "m3_area",
        "area_diff_vs_fiducial",
        "area_ratio_vs_fiducial",
        "area_diff_vs_m3",
        "left_width_cost",
        "right_width_cost",
        "overlap_width_cost",
        "region_fraction",
        "left_fraction",
        "right_fraction",
        "run_length",
        "right_margin",
    )
    output = {
        "reps": reps,
        "coverage": successes / reps,
        "coverage_wilson95": wilson_interval(successes, reps),
        "exterior_escape": sum(row["exterior_escape"] for row in scores) / reps,
        "conditional_capture": captured / failures if failures else None,
        "floor_region_failure": sum(row["floor_region_failure"] for row in scores)
        / reps,
        "edge_escape": sum(row["edge_escape"] for row in scores) / reps,
        "far_escape": sum(row["far_escape"] for row in scores) / reps,
        "viol_low": sum(row["viol_low"] for row in scores) / reps,
        "viol_high": sum(row["viol_high"] for row in scores) / reps,
        "fiducial_coverage": sum(row["fiducial_covered"] for row in scores) / reps,
        "m3_coverage": sum(row["m3_covered"] for row in scores) / reps,
    }
    for key in numeric:
        values = np.asarray([row[key] for row in scores], dtype=np.float64)
        output[key] = float(values.mean())
        output[f"{key}_se_paired"] = (
            float(values.std(ddof=1) / np.sqrt(reps)) if reps > 1 else None
        )
    return output


def score_composite_floor_record(
    record: dict, *, alpha2_key: str, m_draws: int
) -> dict:
    """Score the declared C2.5-interior plus primary frontier floor.

    Args:
        record: Stored Study B record containing the composite parent.
        alpha2_key: Stored M3 level key.
        m_draws: Fiducial cloud size.

    Returns:
        Replicate score for the composite procedure.
    """
    if "composite_c2.5" not in record:
        raise ValueError("record does not contain the Study B composite arm")
    observables = record_observables(record)
    grid = np.arange(observables.n0 + 1, dtype=np.float64) / observables.n0
    fid_lower, fid_upper = decode_band(record["fiducial"]["0.05"])
    interior_lower, interior_upper = decode_band(record["composite_c2.5"])
    corner = (grid <= 0.02) | (grid >= 0.95)
    base_lower = np.maximum.accumulate(np.where(corner, fid_lower, interior_lower))
    base_upper = np.maximum.accumulate(np.where(corner, fid_upper, interior_upper))
    truth = decode_array(record["truth"]).astype(np.float64)
    modified = dict(record)
    modified["fiducial"] = dict(record["fiducial"])
    modified["fiducial"]["0.05"] = {
        "lower": encode_array(base_lower),
        "upper": encode_array(base_upper),
    }
    modified["fiducial_violations"] = dict(record["fiducial_violations"])
    modified["fiducial_violations"]["0.05"] = encode_violations(
        violation_mask(base_lower, base_upper, truth)
    )
    return score_record(
        modified,
        rule="frontier_floor_v1",
        alpha=0.05,
        alpha2_key=alpha2_key,
        m_draws=m_draws,
    )


def _sliver_strata(scores: list[dict]) -> dict:
    """Return conditional sliver and saturated-run summaries.

    Args:
        scores: Replicate scores for one sliver cell and procedure.

    Returns:
        Summaries by sampled-sliver status and prespecified run-length bins.
    """
    output: dict[str, dict] = {}
    for sampled in (False, True):
        subset = [
            row
            for row in scores
            if row["simulation_diagnostics"].get("sliver_sampled") is sampled
        ]
        if subset:
            output[f"sliver_sampled={str(sampled).lower()}"] = summarize_scores(subset)
    bins = ((0, 0, "0"), (1, 4, "1-4"), (5, 16, "5-16"), (17, math.inf, "17+"))
    for lower, upper, label in bins:
        subset = [row for row in scores if lower <= row["run_length"] <= upper]
        if subset:
            output[f"run_length={label}"] = summarize_scores(subset)
    return output


def evaluate_cell(
    cell: CellRecords, *, rule: RuleName, alpha: float, alpha2_key: str
) -> dict:
    """Score and aggregate one cell under a fixed procedure.

    Args:
        cell: Stored cell records.
        rule: Prespecified rule.
        alpha: Nominal fiducial level.
        alpha2_key: Stored M3 level key.

    Returns:
        Cell-level metrics and conditional sliver summaries when applicable.
    """
    m_draws = int(cell.meta["cell"]["m_draws"])
    scores = [
        score_record(
            record, rule=rule, alpha=alpha, alpha2_key=alpha2_key, m_draws=m_draws
        )
        for record in cell.records
    ]
    output = {
        "cell": cell.name,
        "study": cell.study,
        "source": cell.source,
        **summarize_scores(scores),
    }
    if cell.meta["cell"]["shape_meta"].get("family") == "sliver":
        output["sliver_strata"] = _sliver_strata(scores)
    return output


def write_study_summary(cells: list[CellRecords], *, path: Path) -> None:
    """Write fixed-rule cell and macro summaries for one complete study.

    Args:
        cells: Complete study records.
        path: Destination JSON path.
    """
    rows = []
    for cell in cells:
        for alpha, alpha2_keys in ALPHA2_KEYS.items():
            for alpha2_key in alpha2_keys:
                for rule in RULE_NAMES:
                    rows.append(
                        {
                            "rule": rule,
                            "alpha": alpha,
                            "alpha2": float(alpha2_key),
                            **evaluate_cell(
                                cell, rule=rule, alpha=alpha, alpha2_key=alpha2_key
                            ),
                        }
                    )
                if alpha == 0.05 and all(
                    "composite_c2.5" in record for record in cell.records
                ):
                    scores = [
                        score_composite_floor_record(
                            record,
                            alpha2_key=alpha2_key,
                            m_draws=int(cell.meta["cell"]["m_draws"]),
                        )
                        for record in cell.records
                    ]
                    rows.append(
                        {
                            "rule": "frontier_floor_v1_composite",
                            "alpha": alpha,
                            "alpha2": float(alpha2_key),
                            "cell": cell.name,
                            "study": cell.study,
                            "source": cell.source,
                            **summarize_scores(scores),
                        }
                    )
    grouped: dict[tuple[str, float, float], list[dict]] = {}
    for row in rows:
        key = (str(row["rule"]), float(row["alpha"]), float(row["alpha2"]))
        grouped.setdefault(key, []).append(row)
    rng = np.random.default_rng(STAGE_F_SEED)
    macro = []
    for (rule, alpha, alpha2), group in sorted(grouped.items()):
        coverage = np.asarray([row["coverage"] for row in group])
        width = np.asarray([row["area_diff_vs_fiducial"] for row in group])
        indices = rng.integers(0, len(group), size=(2_000, len(group)))
        macro.append(
            {
                "rule": rule,
                "alpha": alpha,
                "alpha2": alpha2,
                "cells": len(group),
                "coverage_cell_macro": float(coverage.mean()),
                "coverage_cell_bootstrap95": np.quantile(
                    coverage[indices].mean(axis=1), [0.025, 0.975]
                ).tolist(),
                "area_diff_cell_macro": float(width.mean()),
                "area_diff_cell_bootstrap95": np.quantile(
                    width[indices].mean(axis=1), [0.025, 0.975]
                ).tolist(),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"rows": rows, "macro": macro}, indent=2, sort_keys=True) + "\n"
    )
