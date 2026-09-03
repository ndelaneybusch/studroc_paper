"""Offline scoring and frozen-rule fitting for Stage F.

All functions operate on stored parent-band records.  They never regenerate a
fiducial cloud, which makes coordinate selection, price curves, alpha2
comparisons, and residual classification deterministic re-analyses of the
same paired replicates.
"""

from __future__ import annotations

import gzip
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.optimize import linprog
from stage_f_core import (
    Coordinate,
    EdgeRule,
    ModelFamily,
    ObservableSummary,
    RegionArtifact,
    SupportBox,
    artifact_payload,
    component_masks,
    coordinate_values,
    decode_array,
    decode_band,
    decode_violations,
    encode_array,
    encode_violations,
    fixed_region_masks,
    freeze_artifact,
    record_observables,
    stitch_hybrid,
    violation_mask,
    write_artifact,
)
from stage_f_design import DEFAULT_OUT, STAGE_F_SEED, load_manifest

RULE_NAMES = ("probe_legacy", "probe_fpr", "count5", "stage_f_v1")
ALPHA2_KEYS = {0.05: ("0.05", "0.025"), 0.5: ("0.5", "0.25")}
RuleName = Literal["probe_legacy", "probe_fpr", "count5", "stage_f_v1"]
FixedRuleName = Literal["probe_legacy", "probe_fpr", "count5"]
CandidateScore = tuple[float, float, float, float]
Candidate = tuple[CandidateScore, Coordinate, Coordinate, ModelFamily, bool]


def _git_hash() -> str:
    """Return the commit recorded in a frozen Stage F rule artifact."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


@dataclass(frozen=True)
class CellRecords:
    """One cell's metadata and lossless replicate records."""

    meta: dict
    records: list[dict]

    @property
    def name(self) -> str:
        """Stable cell name."""
        return str(self.meta["cell"]["name"])

    @property
    def partition(self) -> str:
        """Frozen Study A split assignment."""
        return str(self.meta["cell"]["partition"])


def load_cell_records(path: Path) -> CellRecords:
    """Load one compressed Stage F cell payload."""
    with gzip.open(path, "rt") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "stage-f-cell/v1":
        raise ValueError(f"Unknown Stage F cell schema in {path}")
    return CellRecords(meta=payload["meta"], records=payload["records"])


def load_study_records(root: Path, *, study: str) -> list[CellRecords]:
    """Load every completed cell for one study."""
    return [
        load_cell_records(path) for path in sorted((root / study).glob("*.json.gz"))
    ]


def load_complete_study(
    root: Path, *, study: Literal["A", "B", "C"]
) -> list[CellRecords]:
    """Load a study only when every frozen manifest cell is complete.

    Args:
        root: Stage F output root containing ``manifests`` and cell directories.
        study: Frozen study identifier.

    Returns:
        Cell records in manifest order.

    Raises:
        RuntimeError: If a cell is absent, truncated, duplicated, or was produced
            from a different manifest.
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
        compatibility = cell.meta.get("compatibility", {})
        if compatibility.get("manifest_hash") != manifest["content_hash"]:
            raise RuntimeError(
                f"Cell {expected.name} was not produced from the frozen "
                f"{study} manifest"
            )
        if compatibility.get("cell") != expected_payload:
            raise RuntimeError(
                f"Cell {expected.name} design constants differ from its manifest"
            )
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
    """Wilson score interval for a binomial proportion."""
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
    rule: RuleName,
    *,
    artifact: RegionArtifact | None,
    observables: ObservableSummary,
    khat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the left and right region masks for one frozen rule."""
    if rule == "stage_f_v1":
        if artifact is None:
            raise ValueError("stage_f_v1 scoring requires a frozen artifact")
        return component_masks(artifact, observables=observables, khat=khat)
    fixed_rule: FixedRuleName = rule
    return fixed_region_masks(fixed_rule, observables=observables)


def score_record(
    record: dict,
    *,
    rule: RuleName,
    alpha: float,
    alpha2_key: str,
    artifact: RegionArtifact | None = None,
) -> dict:
    """Score one complete data-adaptive procedure from a stored replicate."""
    nominal_key = f"{alpha:g}"
    observables = record_observables(record)
    khat = decode_array(record["khat"]).astype(np.int64)
    truth = decode_array(record["truth"]).astype(np.float64)
    fid_lower, fid_upper = decode_band(record["fiducial"][nominal_key])
    m3_lower, m3_upper = decode_band(record["m3"][alpha2_key])
    left, right = _masks_for_rule(
        rule, artifact=artifact, observables=observables, khat=khat
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
        """Return mean band width after applying one component mask."""
        lo, hi = stitch_hybrid(
            fid_lower, fid_upper, m3_lower, m3_upper, mask, closure=closure
        )
        return float(np.mean(hi - lo))

    fid_area = float(np.mean(fid_upper - fid_lower))
    left_area = area_for(left)
    right_area = area_for(right)
    area = float(np.mean(upper - lower))
    raw_lower = np.where(region, np.minimum(fid_lower, m3_lower), fid_lower)
    legacy_propagated = bool(
        rule == "probe_legacy"
        and np.any(region & (lower > truth + 1e-12) & (raw_lower <= truth + 1e-12))
    )
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
        "m3_area": float(np.mean(m3_upper - m3_lower)),
        "area_diff_vs_fiducial": area - fid_area,
        "area_ratio_vs_fiducial": area / fid_area if fid_area else 1.0,
        "area_diff_vs_m3": area - float(np.mean(m3_upper - m3_lower)),
        "left_width_cost": left_area - fid_area,
        "right_width_cost": right_area - fid_area,
        "overlap_width_cost": (left_area - fid_area)
        + (right_area - fid_area)
        - (area - fid_area),
        "region_fraction": float(region.mean()),
        "left_fraction": float(left.mean()),
        "right_fraction": float(right.mean()),
        "observables": record["observables"],
    }


def summarize_scores(scores: list[dict]) -> dict:
    """Aggregate replicate scores while preserving paired width uncertainty."""
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
    )
    out = {
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
        out[key] = float(values.mean())
        out[f"{key}_se_paired"] = (
            float(values.std(ddof=1) / np.sqrt(reps)) if reps > 1 else None
        )
    return out


def score_composite_floor_record(
    record: dict, *, alpha2_key: str, artifact: RegionArtifact
) -> dict:
    """Score the declared C2.5-interior plus stage_f_v1 exploratory arm."""
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
        rule="stage_f_v1",
        alpha=0.05,
        alpha2_key=alpha2_key,
        artifact=artifact,
    )


def evaluate_cell(
    cell: CellRecords,
    *,
    rule: RuleName,
    alpha: float,
    alpha2_key: str,
    artifact: RegionArtifact | None = None,
) -> dict:
    """Score and aggregate one cell under a fixed complete procedure."""
    scores = [
        score_record(
            record, rule=rule, alpha=alpha, alpha2_key=alpha2_key, artifact=artifact
        )
        for record in cell.records
    ]
    return {"cell": cell.name, "partition": cell.partition, **summarize_scores(scores)}


def _required_extent(
    record: dict, *, coordinate: Coordinate, side: Literal["left", "right"]
) -> float:
    """Return the smallest coordinate cutoff capturing observed misses."""
    observables = record_observables(record)
    khat = decode_array(record["khat"]).astype(np.int64)
    misses = decode_violations(record["fiducial_violations"]["0.05"])
    grid = np.arange(observables.n0 + 1) / observables.n0
    relevant = misses & (grid <= 0.5 if side == "left" else grid >= 0.5)
    if not relevant.any():
        return 0.0
    values = coordinate_values(coordinate, observables=observables, khat=khat)
    return float(values[relevant].max())


def _fit_edge(
    records: list[dict],
    *,
    coordinate: Coordinate,
    side: Literal["left", "right"],
    family: ModelFamily,
    include_mq: bool,
) -> EdgeRule:
    """Fit one conservative outer-envelope edge rule."""
    observations = [record_observables(record) for record in records]
    target = np.asarray(
        [
            _required_extent(record, coordinate=coordinate, side=side)
            for record in records
        ]
    )
    if family == "constant":
        return EdgeRule(
            coordinate=coordinate,
            family=family,
            intercept=float(np.quantile(target, 0.995, method="higher")),
        )
    if family == "auc_binned":
        upper = (0.90, 0.94, 0.97, 0.985, 1.0)
        cutoffs = []
        previous = 0.0
        aucs = np.asarray([obs.auc_ub for obs in observations])
        lower = 0.0
        for bound in upper:
            values = target[(aucs > lower) & (aucs <= bound)]
            cutoff = (
                previous
                if not len(values)
                else float(np.quantile(values, 0.995, method="higher"))
            )
            previous = max(previous, cutoff)
            cutoffs.append(previous)
            lower = bound
        return EdgeRule(
            coordinate=coordinate,
            family=family,
            auc_bin_upper=upper,
            auc_bin_cutoffs=tuple(cutoffs),
        )

    knots = (0.94, 0.97, 0.985)
    feature_names = ["log_n0", "log_n1", "auc_ub"]
    if include_mq:
        feature_names += ["m30", "m50", "m70"]
    x_base = np.column_stack(
        [np.ones(len(observations))]
        + [
            np.asarray([obs.features()[name] for obs in observations])
            for name in feature_names
        ]
        + [
            np.maximum(np.asarray([obs.auc_ub for obs in observations]) - knot, 0.0)
            for knot in knots
        ]
    )
    objective = x_base.mean(axis=0)
    bounds = [(0.0, None)]
    for name in feature_names:
        bounds.append((0.0, None) if name == "auc_ub" else (None, None))
    bounds.extend((0.0, None) for _ in knots)
    result = linprog(
        c=objective, A_ub=-x_base, b_ub=-target, bounds=bounds, method="highs"
    )
    if not result.success:
        raise RuntimeError(f"linear outer-envelope fit failed: {result.message}")
    coefficients = dict(
        zip(feature_names, result.x[1 : 1 + len(feature_names)], strict=True)
    )
    return EdgeRule(
        coordinate=coordinate,
        family=family,
        intercept=float(result.x[0]),
        coefficients={key: float(value) for key, value in coefficients.items()},
        auc_knots=knots,
        auc_slopes=tuple(float(value) for value in result.x[-len(knots) :]),
    )


def _support(records: list[dict]) -> SupportBox:
    """Return the rectangular observable support spanned by records."""
    observations = [record_observables(record) for record in records]
    return SupportBox(
        n0=(min(obs.n0 for obs in observations), max(obs.n0 for obs in observations)),
        n1=(min(obs.n1 for obs in observations), max(obs.n1 for obs in observations)),
        auc_ub=(
            min(obs.auc_ub for obs in observations),
            max(obs.auc_ub for obs in observations),
        ),
    )


def _fit_artifact(
    records: list[dict],
    *,
    support_records: list[dict] | None = None,
    left_coordinate: Coordinate,
    right_coordinate: Coordinate,
    family: ModelFamily,
    include_mq: bool,
    training: dict,
) -> RegionArtifact:
    """Fit, annotate, validate, and hash a two-edge rule artifact."""
    artifact = RegionArtifact(
        rule_id="stage_f_v1",
        left=_fit_edge(
            records,
            coordinate=left_coordinate,
            side="left",
            family=family,
            include_mq=include_mq,
        ),
        right=_fit_edge(
            records,
            coordinate=right_coordinate,
            side="right",
            family=family,
            include_mq=include_mq,
        ),
        support=_support(support_records or records),
        training=training,
        provenance={
            "spec": "stats/hybrid_floor_spec.md",
            "fitter": "stage_f_analysis.py",
            "git_hash": _git_hash(),
            "manifest_hashes": sorted(
                {
                    cell_hash
                    for cell_hash in training.get("manifest_hashes", [])
                    if cell_hash
                }
            ),
        },
        study_seed=STAGE_F_SEED,
    )
    return freeze_artifact(artifact)


def _macro_candidate_score(
    cells: list[CellRecords], artifact: RegionArtifact
) -> tuple[float, float, float, float]:
    """Return the prespecified lexicographic validation score."""
    rows = [
        evaluate_cell(
            cell, rule="stage_f_v1", alpha=0.05, alpha2_key="0.05", artifact=artifact
        )
        for cell in cells
    ]
    evaluable = [
        row
        for row, cell in zip(rows, cells, strict=True)
        if sum(
            decode_violations(record["fiducial_violations"]["0.05"]).any()
            for record in cell.records
        )
        >= 10
    ]
    basis = evaluable or rows
    worst_escape = max(row["exterior_escape"] for row in basis)
    macro_escape = float(np.mean([row["exterior_escape"] for row in basis]))
    width = float(np.mean([row["area_diff_vs_fiducial"] for row in rows]))
    family_complexity = {"constant": 0, "auc_binned": 1, "linear_hinge": 2}
    coordinate_complexity = {
        "fpr": 0,
        "negative_count": 1,
        "fpr_distance": 0,
        "negative_distance": 1,
        "positive_tail": 2,
    }
    uses_mq = any(
        name in edge.coefficients
        for edge in (artifact.left, artifact.right)
        for name in ("m30", "m50", "m70")
    )
    complexity = float(
        family_complexity[artifact.left.family]
        + family_complexity[artifact.right.family]
        + coordinate_complexity[artifact.left.coordinate]
        + coordinate_complexity[artifact.right.coordinate]
        + uses_mq
    )
    return worst_escape, macro_escape, width, complexity


def _paired_metric_difference(
    cells: list[CellRecords],
    *,
    baseline: RegionArtifact,
    alternative: RegionArtifact,
    metric: str,
    geometry_only: bool = False,
) -> np.ndarray:
    """Return paired cell-level baseline-minus-alternative differences."""
    differences = []
    for cell in cells:
        if (
            geometry_only
            and sum(
                decode_violations(record["fiducial_violations"]["0.05"]).any()
                for record in cell.records
            )
            < 10
        ):
            continue
        baseline_row = evaluate_cell(
            cell, rule="stage_f_v1", alpha=0.05, alpha2_key="0.05", artifact=baseline
        )
        alternative_row = evaluate_cell(
            cell, rule="stage_f_v1", alpha=0.05, alpha2_key="0.05", artifact=alternative
        )
        differences.append(float(baseline_row[metric]) - float(alternative_row[metric]))
    return np.asarray(differences, dtype=np.float64)


def _bootstrap_mean_interval(values: np.ndarray) -> tuple[float, float]:
    """Return the deterministic paired-cell bootstrap interval for a mean."""
    if not len(values):
        return float("-inf"), float("inf")
    rng = np.random.default_rng(STAGE_F_SEED)
    indices = rng.integers(0, len(values), size=(2_000, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def fit_stage_a(
    cells: list[CellRecords], *, artifact_path: Path, report_path: Path
) -> RegionArtifact:
    """Select on 60%, validate on 40%, refit numeric values, and freeze v1."""
    selection = [cell for cell in cells if cell.partition == "selection"]
    validation = [cell for cell in cells if cell.partition == "internal_validation"]
    refit_cells = [cell for cell in cells if cell.partition != "stress"]
    if not selection or not validation:
        raise ValueError(
            "Stage A requires non-empty selection and validation partitions"
        )
    selection_records = [record for cell in selection for record in cell.records]
    refit_records = [record for cell in refit_cells for record in cell.records]
    manifest_hashes = sorted(
        {
            str(manifest_hash)
            for cell in refit_cells
            if (
                manifest_hash := cell.meta.get("compatibility", {}).get("manifest_hash")
            )
        }
    )
    candidates: list[Candidate] = []
    for left_coordinate in ("fpr", "negative_count"):
        for right_coordinate in ("fpr_distance", "negative_distance", "positive_tail"):
            for family in ("constant", "auc_binned", "linear_hinge"):
                for include_mq in (
                    (False, True) if family == "linear_hinge" else (False,)
                ):
                    artifact = _fit_artifact(
                        selection_records,
                        support_records=refit_records,
                        left_coordinate=left_coordinate,
                        right_coordinate=right_coordinate,
                        family=family,
                        include_mq=include_mq,
                        training={
                            "phase": "selection",
                            "include_mq": include_mq,
                            "manifest_hashes": manifest_hashes,
                        },
                    )
                    score = _macro_candidate_score(validation, artifact)
                    candidates.append(
                        (score, left_coordinate, right_coordinate, family, include_mq)
                    )

    def is_eligible(candidate: Candidate) -> bool:
        """Return whether a candidate clears both validation escape targets."""
        return candidate[0][1] <= 0.005 and candidate[0][0] <= 0.02

    def initial_choice(pool: list[Candidate]) -> Candidate:
        """Apply the deterministic escape gate and lexicographic objective."""
        qualified = [candidate for candidate in pool if is_eligible(candidate)]
        if qualified:
            return min(
                qualified, key=lambda candidate: (candidate[0][2], candidate[0][3])
            )
        return min(pool, key=lambda candidate: candidate[0])

    artifact_cache: dict[tuple[Candidate, str], RegionArtifact] = {}

    def fitted(candidate: Candidate, *, phase: str) -> RegionArtifact:
        """Recreate a validation candidate for paired-cell comparisons."""
        key = (candidate, phase)
        if key in artifact_cache:
            return artifact_cache[key]
        _, left, right, model, candidate_mq = candidate
        artifact = _fit_artifact(
            selection_records,
            support_records=refit_records,
            left_coordinate=left,
            right_coordinate=right,
            family=model,
            include_mq=candidate_mq,
            training={"phase": phase, "manifest_hashes": manifest_hashes},
        )
        artifact_cache[key] = artifact
        return artifact

    def prefer_simpler(current: Candidate, pool: list[Candidate]) -> Candidate:
        """Resolve paired-width statistical ties toward lower complexity."""
        for contender in sorted(pool, key=lambda candidate: candidate[0][3]):
            if contender[0][3] >= current[0][3]:
                continue
            differences = _paired_metric_difference(
                validation,
                baseline=fitted(contender, phase="tie_check"),
                alternative=fitted(current, phase="tie_check"),
                metric="area_diff_vs_fiducial",
            )
            lower, _ = _bootstrap_mean_interval(differences)
            if lower <= 0.0:
                current = contender
        return current

    eligible = [candidate for candidate in candidates if is_eligible(candidate)]
    chosen = initial_choice(candidates)
    if eligible:
        chosen = prefer_simpler(chosen, eligible)
    score, left_coordinate, right_coordinate, family, include_mq = chosen
    if include_mq:
        baseline = next(
            row for row in candidates if row[1:4] == chosen[1:4] and row[4] is False
        )
        metric = (
            "area_diff_vs_fiducial"
            if is_eligible(baseline) and is_eligible(chosen)
            else "exterior_escape"
        )
        paired_improvement = _paired_metric_difference(
            validation,
            baseline=fitted(baseline, phase="mq_check"),
            alternative=fitted(chosen, phase="mq_check"),
            metric=metric,
            geometry_only=metric == "exterior_escape",
        )
        improvement_lower, _ = _bootstrap_mean_interval(paired_improvement)
        if improvement_lower <= 0.0:
            plain_candidates = [
                candidate for candidate in candidates if not candidate[4]
            ]
            chosen = initial_choice(plain_candidates)
            plain_eligible = [
                candidate for candidate in plain_candidates if is_eligible(candidate)
            ]
            if plain_eligible:
                chosen = prefer_simpler(chosen, plain_eligible)
            score, left_coordinate, right_coordinate, family, include_mq = chosen
    artifact = _fit_artifact(
        refit_records,
        support_records=refit_records,
        left_coordinate=left_coordinate,
        right_coordinate=right_coordinate,
        family=family,
        include_mq=include_mq,
        training={
            "phase": "refit",
            "selection_cells": [cell.name for cell in selection],
            "validation_cells": [cell.name for cell in validation],
            "refit_cells": [cell.name for cell in refit_cells],
            "validation_score": list(score),
            "include_mq": include_mq,
            "manifest_hashes": manifest_hashes,
        },
    )
    write_artifact(artifact, artifact_path)
    report = {
        "chosen": {
            "left_coordinate": left_coordinate,
            "right_coordinate": right_coordinate,
            "family": family,
            "include_mq": include_mq,
            "validation_score": score,
        },
        "candidates": [
            {
                "score": candidate_score,
                "left_coordinate": left,
                "right_coordinate": right,
                "family": candidate_family,
                "include_mq": candidate_mq,
            }
            for candidate_score, left, right, candidate_family, candidate_mq in (
                candidates
            )
        ],
        "artifact_hash": artifact.content_hash,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    amendment_path = report_path.with_name("stage_f_v1_spec_amendment.md")
    amendment_path.write_text(
        "# Stage F v1 frozen-rule amendment\n\n"
        f"Artifact: `{artifact_path}`  \n"
        f"SHA-256: `{artifact.content_hash}`  \n"
        f"Git commit: `{artifact.provenance['git_hash']}`\n\n"
        "This file is generated immediately after Study A. Studies B/C must "
        "consume the exact artifact hash above; changing the rule creates a "
        "new version and requires new external data.\n\n"
        "```json\n"
        + json.dumps(
            {
                "left": artifact_payload(artifact)["left"],
                "right": artifact_payload(artifact)["right"],
                "support": artifact_payload(artifact)["support"],
                "m3_split_ratio": artifact.m3_split_ratio,
                "tie_break": artifact.tie_break,
                "outside_support": artifact.outside_support,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n```\n"
    )
    return artifact


def write_external_summary(
    cells: list[CellRecords], *, artifact: RegionArtifact, path: Path
) -> None:
    """Write all fixed-rule and learned-rule cell summaries for B or C."""
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
                                cell,
                                rule=rule,
                                alpha=alpha,
                                alpha2_key=alpha2_key,
                                artifact=artifact,
                            ),
                        }
                    )
                if alpha == 0.05 and all(
                    "composite_c2.5" in record for record in cell.records
                ):
                    composite_scores = [
                        score_composite_floor_record(
                            record, alpha2_key=alpha2_key, artifact=artifact
                        )
                        for record in cell.records
                    ]
                    rows.append(
                        {
                            "rule": "stage_f_v1_composite",
                            "alpha": alpha,
                            "alpha2": float(alpha2_key),
                            "cell": cell.name,
                            "partition": cell.partition,
                            **summarize_scores(composite_scores),
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
        boot_indices = rng.integers(0, len(group), size=(2_000, len(group)))
        boot_coverage = coverage[boot_indices].mean(axis=1)
        boot_width = width[boot_indices].mean(axis=1)
        macro.append(
            {
                "rule": rule,
                "alpha": alpha,
                "alpha2": alpha2,
                "cells": len(group),
                "coverage_cell_macro": float(coverage.mean()),
                "coverage_cell_bootstrap95": np.quantile(
                    boot_coverage, [0.025, 0.975]
                ).tolist(),
                "area_diff_cell_macro": float(width.mean()),
                "area_diff_cell_bootstrap95": np.quantile(
                    boot_width, [0.025, 0.975]
                ).tolist(),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"artifact_hash": artifact.content_hash, "rows": rows, "macro": macro},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def default_stage_a_fit(root: Path = DEFAULT_OUT) -> RegionArtifact:
    """Fit Stage F v1 from completed Study A records and write its artifacts."""
    cells = load_complete_study(root, study="A")
    return fit_stage_a(
        cells,
        artifact_path=root / "rules/stage_f_v1.json",
        report_path=root / "analysis/study_a_selection.json",
    )
