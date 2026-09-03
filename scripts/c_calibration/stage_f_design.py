"""Frozen cell designs, manifests, and seed streams for Stage F.

Manifest creation is intentionally separate from simulation execution.  Study
B/C cells are defined here from pre-Stage-F evidence; Study A's replay corpus
is selected mechanically from existing summaries and combined with a frozen
imbalance LHS and four extent-stress cells.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
from design import STUDY_SEED as LEGACY_STUDY_SEED
from design import m_budget
from scipy.stats import norm, qmc
from shapes import (
    ShapeSpec,
    _lhs_curve,
    get_curve,
    lhs_heldout_specs,
    make_bimodal_negative,
    make_binormal,
    make_hetero_gaussian,
    make_kink,
    make_t_shape,
    make_trapezoid,
    quantize_jitter,
    shape_registry,
)

STAGE_F_SEED = 20260902
MANIFEST_SCHEMA = "stage-f-manifest/v1"
DEFAULT_SOURCE = Path("data/results/c_calibration_followup_20260830")
DEFAULT_OUT = Path("data/results/hybrid_floor_20260902")
PRIMARY_ALPHA = 0.05
TRANSFER_ALPHA = 0.5
BASE_REPS_A = 200
BASE_REPS_EXTERNAL = 400
MAX_REPS_EXTERNAL = 1_200
IMBALANCE_LHS_SEED = 20260902
IMBALANCE_LHS_SIZE = 24

Study = Literal["A", "B", "C"]
Partition = Literal["selection", "internal_validation", "stress", "external"]


@dataclass(frozen=True)
class StageFCell:
    """One frozen Stage F simulation cell."""

    name: str
    study: Study
    source: str
    shape: str
    shape_meta: dict
    n0: int
    n1: int
    reps: int
    reps_max: int
    partition: Partition
    alphas: tuple[float, ...] = (PRIMARY_ALPHA, TRANSFER_ALPHA)
    quantize: int | None = None
    m_draws: int = 0
    seed_mode: Literal["stage_f", "legacy_replay"] = "stage_f"
    source_name: str | None = None
    source_stage: str | None = None
    prior_coverage: float | None = None
    prior_reps: int | None = None
    notes: str = ""

    @property
    def n_grid(self) -> int:
        """Number of points on the native negative grid."""
        return self.n0 + 1

    def with_budget(self) -> StageFCell:
        """Return the cell with the production alpha=.05 cloud budget."""
        if self.m_draws:
            return self
        return replace(self, m_draws=m_budget(self.n0, PRIMARY_ALPHA))


def _stable_hash(text: str) -> int:
    """Map text to a stable unsigned 64-bit integer."""
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def _shape(
    *, name: str, family: str, auc: float, params: dict[str, float] | None = None
) -> tuple[str, dict]:
    """Return a shape registry name and JSON-native reconstruction metadata."""
    return name, {"family": family, "auc": auc, **(params or {})}


def register_cell_shape(cell: StageFCell) -> None:
    """Register a cell's curve from its frozen, JSON-native shape metadata."""
    registry = shape_registry()
    if cell.shape in registry:
        return
    meta = cell.shape_meta
    family = meta["family"]
    auc = float(meta["auc"])
    if family == "student_t":
        build = partial(make_t_shape, auc=auc, df=float(meta["df"]))
    elif family == "binormal":
        build = partial(make_binormal, auc=auc)
    elif family == "hetero_gaussian":
        ratio = float(meta.get("sigma_ratio", meta.get("ratio", 3.0)))
        build = partial(make_hetero_gaussian, auc=auc, sigma_ratio=ratio)
    elif family == "bimodal_negative":
        separation = float(meta.get("mode_separation", meta.get("sep", 3.0)))
        weight = float(meta.get("mixture_weight", meta.get("weight", 0.5)))
        build = partial(make_bimodal_negative, auc=auc, sep=separation, weight=weight)
    elif family == "kink":
        build = partial(
            make_kink,
            t_kink=float(meta.get("t_kink", 0.004)),
            tpr_kink=float(meta.get("tpr_kink", 0.6)),
        )
    elif family in {"weibull", "gamma", "beta_opposing"}:
        excluded = {"family", "auc", "note", "lhs_seed", "lhs_index"}
        params = {
            key: float(value) for key, value in meta.items() if key not in excluded
        }
        build = partial(_lhs_curve, family=family, auc=auc, params=params)
    else:
        raise ValueError(f"Cannot reconstruct Stage F shape family {family!r}")
    registry[cell.shape] = ShapeSpec(
        name=cell.shape, role="stage_f", build=build, meta=dict(meta)
    )


def cell_curve(cell: StageFCell):
    """Return a cell's exact simulation truth, including quantized ties."""
    register_cell_shape(cell)
    base = get_curve(cell.shape)
    return make_trapezoid(base, cell.quantize) if cell.quantize is not None else base


def rep_seed_sequence(cell: StageFCell, rep: int) -> np.random.SeedSequence:
    """Return the frozen per-(study, cell, replicate) seed sequence."""
    if cell.seed_mode == "legacy_replay":
        if cell.source_name is None or cell.source_stage is None:
            raise ValueError("legacy replay cells require source_name and source_stage")
        stage_number = {"S": 0, "A": 1, "B": 2}[cell.source_stage]
        entropy = (LEGACY_STUDY_SEED, stage_number, _stable_hash(cell.source_name), rep)
    else:
        entropy = (STAGE_F_SEED, ord(cell.study), _stable_hash(cell.name), rep)
    return np.random.SeedSequence(entropy=entropy)


def sample_labels(cell: StageFCell, rep: int) -> tuple[np.ndarray, int]:
    """Draw one shared tie-resolved label order and fiducial-cloud seed."""
    register_cell_shape(cell)
    rng = np.random.default_rng(rep_seed_sequence(cell, rep))
    curve = get_curve(cell.shape)
    negative = rng.random(cell.n0)
    positive = curve.inv(rng.random(cell.n1))
    if cell.quantize is not None:
        negative, positive = quantize_jitter(negative, positive, cell.quantize, rng)
    labels = np.concatenate(
        [np.zeros(cell.n0, dtype=np.uint8), np.ones(cell.n1, dtype=np.uint8)]
    )
    order = np.argsort(np.concatenate([negative, positive]), kind="stable")
    seed = int(rng.integers(0, 2**64, dtype=np.uint64))
    return labels[order], seed


def _new_cell(
    *,
    name: str,
    study: Study,
    source: str,
    shape: tuple[str, dict],
    n0: int,
    n1: int,
    partition: Partition,
    reps: int,
    reps_max: int,
    quantize: int | None = None,
    notes: str = "",
) -> StageFCell:
    """Construct a Stage F cell and assign its deterministic cloud budget."""
    shape_name, shape_meta = shape
    return StageFCell(
        name=name,
        study=study,
        source=source,
        shape=shape_name,
        shape_meta=shape_meta,
        n0=n0,
        n1=n1,
        reps=reps,
        reps_max=reps_max,
        partition=partition,
        quantize=quantize,
        notes=notes,
    ).with_budget()


def imbalance_lhs_cells() -> list[StageFCell]:
    """Return the frozen, achievable 24-cell Study A imbalance LHS."""
    from studroc_paper.datagen.roc_to_dgp import StudentTSolver

    sampler = qmc.LatinHypercube(d=4, seed=IMBALANCE_LHS_SEED)
    candidates = sampler.random(IMBALANCE_LHS_SIZE * 4)
    z_lo, z_hi = norm.ppf(0.85), norm.ppf(0.99)
    solver = StudentTSolver()
    feasible = []
    for index, point in enumerate(candidates):
        df = float(np.exp(np.log(1.1) + point[0] * (np.log(30.0) - np.log(1.1))))
        auc = float(norm.cdf(z_lo + point[1] * (z_hi - z_lo)))
        n_total = int(
            round(np.exp(np.log(400) + point[2] * (np.log(10_000) - np.log(400))))
        )
        ratio = float(np.exp(np.log(0.2) + point[3] * (np.log(5.0) - np.log(0.2))))
        if auc > float(solver._compute_auc(df, 20.0)) - 0.002:
            continue
        n0 = max(1, int(round(n_total * ratio / (1.0 + ratio))))
        n1 = n_total - n0
        auc_band = int(np.searchsorted([0.90, 0.95], auc))
        orientation = "n0_gt" if n0 > n1 else "n1_gt"
        feasible.append((index, df, auc, n0, n1, auc_band, orientation))

    chosen = []
    used = set()
    for auc_band in range(3):
        for orientation in ("n0_gt", "n1_gt"):
            match = next(
                row
                for row in feasible
                if row[5] == auc_band and row[6] == orientation and row[0] not in used
            )
            chosen.append(match)
            used.add(match[0])
    for row in feasible:
        if len(chosen) == IMBALANCE_LHS_SIZE:
            break
        if row[0] not in used:
            chosen.append(row)
            used.add(row[0])

    cells = []
    for index, df, auc, n0, n1, _, _ in chosen:
        shape = _shape(
            name=f"sf_a_lhs_t{len(cells):02d}",
            family="student_t",
            auc=round(auc, 6),
            params={"df": round(df, 6), "lhs_index": index},
        )
        cells.append(
            _new_cell(
                name=f"a-imbalance-{len(cells):02d}--n{n0}x{n1}",
                study="A",
                source="imbalance_lhs",
                shape=shape,
                n0=n0,
                n1=n1,
                partition="selection",
                reps=BASE_REPS_A,
                reps_max=BASE_REPS_A,
            )
        )
    if len(cells) != IMBALANCE_LHS_SIZE:
        raise RuntimeError("achievability filter did not yield 24 imbalance cells")
    return cells


def extent_stress_cells() -> list[StageFCell]:
    """Return the four prespecified high-AUC, large-n Study A stress cells."""
    specifications = (
        (2.0, 0.99, 8_000),
        (2.0, 0.99, 12_000),
        (6.62, 0.9883, 8_000),
        (6.62, 0.9883, 12_000),
    )
    return [
        _new_cell(
            name=f"a-stress-t{df:g}-a{auc:.4f}-n{n}",
            study="A",
            source="extent_stress",
            shape=_shape(
                name=f"sf_a_stress_t{index}",
                family="student_t",
                auc=auc,
                params={"df": df},
            ),
            n0=n,
            n1=n,
            partition="stress",
            reps=BASE_REPS_A,
            reps_max=BASE_REPS_A,
        )
        for index, (df, auc, n) in enumerate(specifications)
    ]


def _external_cell(
    *,
    study: Study,
    index: int,
    source: str,
    family: str,
    auc: float,
    n0: int,
    n1: int,
    params: dict[str, float] | None = None,
    quantize: int | None = None,
) -> StageFCell:
    """Construct one external-study cell with the fixed replication limits."""
    shape_name = f"sf_{study.lower()}_{source}_{index:02d}"
    return _new_cell(
        name=f"{study.lower()}-{source}-{index:02d}--n{n0}x{n1}",
        study=study,
        source=source,
        shape=_shape(name=shape_name, family=family, auc=auc, params=params),
        n0=n0,
        n1=n1,
        partition="external",
        reps=BASE_REPS_EXTERNAL,
        reps_max=MAX_REPS_EXTERNAL,
        quantize=quantize,
    )


def study_b_cells() -> list[StageFCell]:
    """Return the completely frozen 24-cell external student-t/safe design."""
    specs: list[
        tuple[str, str, float, int, int, dict[str, float] | None, int | None]
    ] = []
    for df, auc, n in (
        (2.0, 0.99, 250),
        (2.0, 0.99, 500),
        (2.0, 0.99, 1_000),
        (4.69, 0.986, 400),
        (4.69, 0.986, 1_200),
        (4.69, 0.986, 2_000),
        (6.62, 0.9883, 1_900),
        (6.62, 0.9883, 6_656),
        (1.13, 0.926, 130),
        (3.29, 0.9844, 5_131),
    ):
        specs.append(("wedge", "student_t", auc, n, n, {"df": df}, None))
    heldout = lhs_heldout_specs()[1]
    specs.extend(
        [
            ("safe", "binormal", 0.75, 250, 250, None, None),
            ("safe", "binormal", 0.90, 1_000, 1_000, None, None),
            (
                "safe",
                "bimodal_negative",
                0.90,
                250,
                250,
                {"mode_separation": 3.0, "mixture_weight": 0.5},
                None,
            ),
            ("safe", "hetero_gaussian", 0.90, 1_000, 1_000, {"sigma_ratio": 3.0}, None),
            ("safe", "student_t", 0.90, 250, 250, {"df": 3.0}, None),
            (
                "safe",
                "kink",
                0.798,
                1_000,
                1_000,
                {"t_kink": 0.004, "tpr_kink": 0.6},
                None,
            ),
            ("imbalance", "student_t", 0.98, 300, 1_500, {"df": 2.0}, None),
            ("imbalance", "student_t", 0.98, 1_500, 300, {"df": 2.0}, None),
            ("imbalance", "student_t", 0.986, 750, 3_000, {"df": 4.69}, None),
            ("imbalance", "student_t", 0.986, 3_000, 750, {"df": 4.69}, None),
            ("large_n", "student_t", 0.99, 8_000, 8_000, {"df": 2.0}, None),
            ("large_n", "student_t", 0.986, 12_000, 12_000, {"df": 4.69}, None),
            ("regression", "binormal", 0.90, 1_000, 1_000, None, 20),
            (
                "regression",
                str(heldout["family"]),
                float(heldout["auc"]),
                1_000,
                1_000,
                dict(heldout["params"]),
                None,
            ),
        ]
    )
    return [
        _external_cell(
            study="B",
            index=index,
            source=source,
            family=family,
            auc=auc,
            n0=n0,
            n1=n1,
            params=params,
            quantize=quantize,
        )
        for index, (source, family, auc, n0, n1, params, quantize) in enumerate(specs)
    ]


def study_c_cells() -> list[StageFCell]:
    """Return 14 frozen cross-family transfer cells as inside/control pairs."""
    families = [
        ("weibull", 0.985, {"shape": 0.5}),
        ("gamma", 0.925, {"shape": 0.5}),
        ("beta_opposing", 0.985, {"alpha": 0.5}),
        ("bimodal_negative", 0.98, {"mode_separation": 4.0, "mixture_weight": 0.5}),
        ("hetero_gaussian", 0.98, {"sigma_ratio": 3.0}),
    ]
    for specification in lhs_heldout_specs():
        families.append(
            (
                specification["family"],
                max(0.95, float(specification["auc"])),
                dict(specification["params"]),
            )
        )
    cells = []
    for pair, (family, auc, params) in enumerate(families):
        for member, n in enumerate((500, 8_000)):
            cells.append(
                _external_cell(
                    study="C",
                    index=2 * pair + member,
                    source=f"{family}_{'inside' if member == 0 else 'control'}",
                    family=family,
                    auc=auc,
                    n0=n,
                    n1=n,
                    params=params,
                )
            )
    return cells


def _coverage_from_summary(summary: dict) -> float:
    """Extract the primary C=1 coverage from a legacy summary."""
    for row in summary["aggregate"]["ref_maps"]:
        if row["label"] == "c1" and float(row["alpha"]) == PRIMARY_ALPHA:
            return float(row["coverage"])
    raise ValueError("summary has no C=1, alpha=.05 arm")


def replay_corpus_cells(source_root: Path = DEFAULT_SOURCE) -> list[StageFCell]:
    """Select the Study A replay corpus mechanically from prior summaries."""
    candidates = []
    for path in sorted(source_root.rglob("*.summary.json")):
        if path.name.endswith(".m3.summary.json"):
            continue
        try:
            summary = json.loads(path.read_text())
            coverage = _coverage_from_summary(summary)
            prior_reps = int(summary["aggregate"]["reps"])
            meta = summary["meta"]["cell"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        key = str(meta["name"])
        candidates.append((key, coverage, meta))
    unique = {name: (coverage, meta) for name, coverage, meta in candidates}
    failing = [(name, *values) for name, values in unique.items() if values[0] < 0.94]
    near = [
        (name, *values) for name, values in unique.items() if 0.94 <= values[0] < 0.97
    ]
    safe = [(name, *values) for name, values in unique.items() if values[0] >= 0.97]

    def take(rows: list[tuple[str, float, dict]], count: int | None) -> list:
        """Choose a stable hash-ordered prefix from one coverage stratum."""
        ordered = sorted(rows, key=lambda row: _stable_hash(f"stage-f-replay:{row[0]}"))
        return ordered if count is None else ordered[:count]

    selected = take(failing, None) + take(near, 15) + take(safe, 8)
    cells = []
    for name, coverage, meta in selected:
        shape_meta = dict(meta.get("shape_meta", {}))
        if not shape_meta:
            registry_meta = shape_registry().get(meta["shape"])
            shape_meta = dict(registry_meta.meta) if registry_meta else {}
        cell = StageFCell(
            name=f"a-replay--{name}",
            study="A",
            source="replay",
            shape=meta["shape"],
            shape_meta=shape_meta,
            n0=int(meta["n0"]),
            n1=int(meta["n1"]),
            reps=BASE_REPS_A,
            reps_max=BASE_REPS_A,
            partition="selection",
            quantize=meta.get("quantize"),
            seed_mode="legacy_replay",
            source_name=name,
            source_stage=str(meta["stage"]),
            prior_coverage=coverage,
            prior_reps=prior_reps,
            notes=f"prior C=1 coverage={coverage:.6g}",
        ).with_budget()
        cells.append(cell)
    return cells


def _partition_study_a(cells: list[StageFCell]) -> list[StageFCell]:
    """Apply a deterministic approximately 60/40 stratified split."""
    strata: dict[tuple, list[StageFCell]] = {}
    for cell in cells:
        if cell.partition == "stress":
            continue
        auc = float(cell.shape_meta.get("auc", 0.0))
        auc_band = int(np.searchsorted([0.90, 0.94, 0.97, 0.985], auc))
        prior = cell.prior_coverage
        coverage_band = (
            "new"
            if prior is None
            else ("fail" if prior < 0.94 else "near" if prior < 0.97 else "safe")
        )
        orientation = (
            "balanced"
            if cell.n0 == cell.n1
            else ("n0_gt" if cell.n0 > cell.n1 else "n1_gt")
        )
        strata.setdefault(
            (cell.source, auc_band, coverage_band, orientation), []
        ).append(cell)
    assignments = {}
    for rows in strata.values():
        rows = sorted(rows, key=lambda cell: _stable_hash(f"stage-f-split:{cell.name}"))
        n_selection = max(1, int(np.ceil(0.6 * len(rows))))
        assignments.update(
            (cell.name, "selection" if index < n_selection else "internal_validation")
            for index, cell in enumerate(rows)
        )
    return [
        replace(cell, partition=assignments.get(cell.name, cell.partition))
        for cell in cells
    ]


def study_a_cells(source_root: Path = DEFAULT_SOURCE) -> list[StageFCell]:
    """Return the complete replay + imbalance + stress Study A design."""
    return _partition_study_a(
        replay_corpus_cells(source_root) + imbalance_lhs_cells() + extent_stress_cells()
    )


def manifest_payload(*, study: Study, cells: list[StageFCell]) -> dict:
    """Build a stable manifest and its content hash."""
    body = {
        "schema": MANIFEST_SCHEMA,
        "study": study,
        "study_seed": STAGE_F_SEED,
        "design_constants": {
            "primary_alpha": PRIMARY_ALPHA,
            "transfer_alpha": TRANSFER_ALPHA,
            "auc_delta": 0.05,
            "m3_split_ratio": 0.5,
            "tie_break": "random",
            "trim_grid": "production",
        },
        "cells": [asdict(cell) for cell in cells],
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return {**body, "content_hash": hashlib.sha256(canonical.encode()).hexdigest()}


def write_manifests(
    out_dir: Path = DEFAULT_OUT, *, source_root: Path = DEFAULT_SOURCE
) -> dict[str, Path]:
    """Freeze all three Stage F manifests without running a replicate."""
    designs: dict[Study, list[StageFCell]] = {
        "A": study_a_cells(source_root),
        "B": study_b_cells(),
        "C": study_c_cells(),
    }
    manifest_dir = out_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for study, cells in designs.items():
        path = manifest_dir / f"study_{study.lower()}.json"
        proposed = manifest_payload(study=study, cells=cells)
        if path.exists():
            existing, _ = load_manifest(path)
            if existing["content_hash"] != proposed["content_hash"]:
                raise RuntimeError(
                    f"Refusing to replace frozen Stage F manifest at {path}"
                )
        else:
            path.write_text(json.dumps(proposed, indent=2, sort_keys=True) + "\n")
        paths[study] = path
    return paths


def load_manifest(path: Path) -> tuple[dict, list[StageFCell]]:
    """Load a frozen manifest and verify its content hash."""
    payload = json.loads(path.read_text())
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"Unknown Stage F manifest schema: {payload.get('schema')!r}")
    expected = payload.pop("content_hash")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    actual = hashlib.sha256(canonical.encode()).hexdigest()
    if actual != expected:
        raise ValueError("Stage F manifest content hash does not match its payload")
    payload["content_hash"] = expected
    cells = []
    for stored_item in payload["cells"]:
        item = dict(stored_item)
        item["alphas"] = tuple(item["alphas"])
        cells.append(StageFCell(**item))
    return payload, cells
