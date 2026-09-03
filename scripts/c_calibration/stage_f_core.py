"""Statistical and persistence primitives for the Stage F M3-floor study.

This module contains no simulation design and performs no I/O beyond explicit
artifact helpers.  It is the shared contract between the Stage F runner and
offline analysis: observable summaries, region evaluation, both stitch
closures, exact array serialization, and lossless violation-set encoding.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

ARTIFACT_SCHEMA = "stage-f-region/v1"
RECORD_SCHEMA = "stage-f-replicate/v1"
MAX_RLE_INTERVALS = 64
AUC_BOUND_DELTA = 0.05

Coordinate = Literal[
    "fpr", "negative_count", "fpr_distance", "negative_distance", "positive_tail"
]
ModelFamily = Literal["constant", "auc_binned", "linear_hinge"]
Closure = Literal["widening", "legacy"]


@dataclass(frozen=True)
class ObservableSummary:
    """Dataset-only quantities available to a frozen region rule.

    Args:
        n0: Number of negative observations.
        n1: Number of positive observations.
        auc_hat: Empirical Mann-Whitney AUC for the shared tie realization.
        auc_ub: Primary bounded-differences upper confidence bound.
        auc_delong_ub: One-sided normal DeLong sensitivity bound.
        m30: Negative-grid count where empirical TPR first reaches 0.30.
        m50: Negative-grid count where empirical TPR first reaches 0.50.
        m70: Negative-grid count where empirical TPR first reaches 0.70.
    """

    n0: int
    n1: int
    auc_hat: float
    auc_ub: float
    auc_delong_ub: float
    m30: int
    m50: int
    m70: int

    def features(self) -> dict[str, float]:
        """Return the covariates understood by linear edge models."""
        return {
            "log_n0": float(np.log(self.n0)),
            "log_n1": float(np.log(self.n1)),
            "auc_ub": self.auc_ub,
            "m30": float(self.m30),
            "m50": float(self.m50),
            "m70": float(self.m70),
        }


@dataclass(frozen=True)
class SupportBox:
    """Closed observable support of a fitted Stage F rule."""

    n0: tuple[int, int]
    n1: tuple[int, int]
    auc_ub: tuple[float, float]

    def contains(self, observables: ObservableSummary) -> bool:
        """Return whether an observable vector lies inside fitted support."""
        return (
            self.n0[0] <= observables.n0 <= self.n0[1]
            and self.n1[0] <= observables.n1 <= self.n1[1]
            and self.auc_ub[0] <= observables.auc_ub <= self.auc_ub[1]
        )


@dataclass(frozen=True)
class EdgeRule:
    """One endpoint-connected region component.

    Every coordinate is a nonnegative distance from its endpoint, so a larger
    predicted cutoff always enlarges the region.  ``linear_hinge`` models are
    nested outward in ``auc_ub`` because the direct AUC coefficient and every
    hinge slope must be nonnegative.  ``auc_binned`` cutoffs must likewise be
    nondecreasing.
    """

    coordinate: Coordinate
    family: ModelFamily
    intercept: float = 0.0
    coefficients: dict[str, float] = field(default_factory=dict)
    auc_knots: tuple[float, ...] = ()
    auc_slopes: tuple[float, ...] = ()
    auc_bin_upper: tuple[float, ...] = ()
    auc_bin_cutoffs: tuple[float, ...] = ()

    def validate(self, *, side: Literal["left", "right"]) -> None:
        """Validate model shape, coordinate side, and AUC nesting."""
        left_coordinates = {"fpr", "negative_count"}
        right_coordinates = {"fpr_distance", "negative_distance", "positive_tail"}
        allowed = left_coordinates if side == "left" else right_coordinates
        if self.coordinate not in allowed:
            raise ValueError(f"{self.coordinate!r} is not a {side}-edge coordinate")
        if self.intercept < 0.0:
            raise ValueError("edge intercept must be nonnegative")
        if self.family == "constant":
            return
        if self.family == "linear_hinge":
            if len(self.auc_knots) != len(self.auc_slopes):
                raise ValueError("auc_knots and auc_slopes must have equal length")
            if tuple(sorted(self.auc_knots)) != self.auc_knots:
                raise ValueError("auc_knots must be sorted")
            if self.coefficients.get("auc_ub", 0.0) < 0.0 or any(
                slope < 0.0 for slope in self.auc_slopes
            ):
                raise ValueError("AUC effects must be nonnegative")
            unknown = set(self.coefficients) - {
                "log_n0",
                "log_n1",
                "auc_ub",
                "m30",
                "m50",
                "m70",
            }
            if unknown:
                raise ValueError(f"Unknown edge-model coefficients: {sorted(unknown)}")
            return
        if self.family == "auc_binned":
            if not self.auc_bin_upper or len(self.auc_bin_upper) != len(
                self.auc_bin_cutoffs
            ):
                raise ValueError("AUC bins and cutoffs must be non-empty and aligned")
            if tuple(sorted(self.auc_bin_upper)) != self.auc_bin_upper:
                raise ValueError("AUC bin upper bounds must be sorted")
            if any(np.diff(self.auc_bin_cutoffs) < 0.0):
                raise ValueError("AUC-bin cutoffs must be nondecreasing")
            return
        raise ValueError(f"Unknown edge model family: {self.family!r}")

    def cutoff(self, observables: ObservableSummary) -> float:
        """Evaluate the nonnegative edge cutoff for one dataset."""
        if self.family == "constant":
            return self.intercept
        if self.family == "auc_binned":
            index = int(np.searchsorted(self.auc_bin_upper, observables.auc_ub))
            index = min(index, len(self.auc_bin_cutoffs) - 1)
            return self.auc_bin_cutoffs[index]
        features = observables.features()
        value = self.intercept + sum(
            coefficient * features[name]
            for name, coefficient in self.coefficients.items()
        )
        value += sum(
            slope * max(observables.auc_ub - knot, 0.0)
            for knot, slope in zip(self.auc_knots, self.auc_slopes, strict=True)
        )
        return max(float(value), 0.0)


@dataclass(frozen=True)
class RegionArtifact:
    """Frozen observable-only Stage F region rule."""

    rule_id: str
    left: EdgeRule
    right: EdgeRule
    support: SupportBox
    alpha: float = 0.05
    auc_delta: float = AUC_BOUND_DELTA
    tie_break: str = "random"
    m3_split_ratio: float = 0.5
    outside_support: str = "full_region"
    study_seed: int = 20260902
    training: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    schema: str = ARTIFACT_SCHEMA
    content_hash: str = ""

    def validate(self, *, verify_hash: bool = True) -> None:
        """Raise if the artifact violates the frozen Stage F contract."""
        if self.schema != ARTIFACT_SCHEMA:
            raise ValueError(f"Unknown Stage F artifact schema: {self.schema!r}")
        if self.rule_id != "stage_f_v1":
            raise ValueError(f"Unexpected rule_id: {self.rule_id!r}")
        if not 0.0 < self.alpha < 1.0 or not 0.0 < self.auc_delta < 1.0:
            raise ValueError("alpha and auc_delta must lie in (0, 1)")
        if self.tie_break != "random":
            raise ValueError("Stage F v1 requires shared random tie-breaking")
        if not 0.0 < self.m3_split_ratio < 1.0:
            raise ValueError("m3_split_ratio must lie in (0, 1)")
        if self.outside_support != "full_region":
            raise ValueError("Unsupported extrapolation policy")
        self.left.validate(side="left")
        self.right.validate(side="right")
        if verify_hash and self.content_hash != artifact_hash(self):
            raise ValueError("Stage F artifact content hash does not match its payload")


def artifact_payload(artifact: RegionArtifact, *, include_hash: bool = True) -> dict:
    """Convert an artifact to its canonical JSON-native representation."""
    payload = asdict(artifact)
    if not include_hash:
        payload.pop("content_hash", None)
    return payload


def artifact_hash(artifact: RegionArtifact) -> str:
    """Return the SHA-256 hash of an artifact excluding its hash field."""
    canonical = json.dumps(
        artifact_payload(artifact, include_hash=False),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def freeze_artifact(artifact: RegionArtifact) -> RegionArtifact:
    """Return a validated artifact with its canonical content hash filled."""
    candidate = replace(artifact, content_hash=artifact_hash(artifact))
    candidate.validate()
    return candidate


def write_artifact(artifact: RegionArtifact, path: Path) -> None:
    """Validate and write a frozen artifact as stable JSON."""
    artifact.validate()
    if path.exists():
        existing = load_artifact(path)
        if existing.content_hash != artifact.content_hash:
            raise RuntimeError(
                f"Refusing to replace frozen Stage F rule at {path} with a "
                "different artifact"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact_payload(artifact), indent=2, sort_keys=True) + "\n"
    )


def load_artifact(path: Path) -> RegionArtifact:
    """Load and validate a frozen Stage F rule artifact."""
    payload = json.loads(path.read_text())

    def load_edge(values: dict) -> EdgeRule:
        """Normalize JSON lists back to immutable edge-rule tuples."""
        for key in ("auc_knots", "auc_slopes", "auc_bin_upper", "auc_bin_cutoffs"):
            values[key] = tuple(values.get(key, ()))
        return EdgeRule(**values)

    payload["left"] = load_edge(payload["left"])
    payload["right"] = load_edge(payload["right"])
    support = payload["support"]
    payload["support"] = SupportBox(
        n0=tuple(support["n0"]),
        n1=tuple(support["n1"]),
        auc_ub=tuple(support["auc_ub"]),
    )
    artifact = RegionArtifact(**payload)
    artifact.validate()
    return artifact


def khat_from_labels(labels: NDArray) -> NDArray[np.int64]:
    """Return empirical positive counts on the native negative grid."""
    labels = np.asarray(labels, dtype=np.int64)
    positives = np.cumsum(labels)
    negative_indices = np.flatnonzero(labels == 0)
    return np.concatenate([positives[negative_indices], [int(labels.sum())]]).astype(
        np.int64
    )


def empirical_observables(
    labels: NDArray, *, delta: float = AUC_BOUND_DELTA
) -> tuple[ObservableSummary, NDArray[np.int64]]:
    """Compute every rule input from one shared, tie-resolved label ordering.

    Args:
        labels: Labels ordered from highest to lowest score.
        delta: Tail probability in the bounded-differences AUC upper bound.

    Returns:
        Observable summary and empirical positive-count map.

    Raises:
        ValueError: If labels are malformed or either class is absent.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must be a one-dimensional 0/1 array")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    labels = labels.astype(np.int64, copy=False)
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    if n0 == 0 or n1 == 0:
        raise ValueError("both classes must be present")

    positive_before = np.cumsum(labels) - labels
    negative_before = np.cumsum(1 - labels) - (1 - labels)
    positive_placements = (n0 - negative_before[labels == 1]) / n0
    negative_placements = positive_before[labels == 0] / n1
    auc_hat = float(positive_placements.mean())
    variance = 0.0
    if n1 > 1:
        variance += float(positive_placements.var(ddof=1) / n1)
    if n0 > 1:
        variance += float(negative_placements.var(ddof=1) / n0)
    auc_delong_ub = min(1.0, auc_hat + float(norm.ppf(0.95)) * np.sqrt(variance))
    radius = np.sqrt(0.5 * (1.0 / n0 + 1.0 / n1) * np.log(1.0 / delta))
    auc_ub = min(1.0, auc_hat + float(radius))

    khat = khat_from_labels(labels)

    def m_at(q: float) -> int:
        """Return the first negative-grid index reaching a positive fraction."""
        reached = np.flatnonzero(khat >= q * n1)
        return int(reached[0]) if len(reached) else n0

    return (
        ObservableSummary(
            n0=n0,
            n1=n1,
            auc_hat=auc_hat,
            auc_ub=auc_ub,
            auc_delong_ub=auc_delong_ub,
            m30=m_at(0.30),
            m50=m_at(0.50),
            m70=m_at(0.70),
        ),
        khat,
    )


def coordinate_values(
    coordinate: Coordinate, *, observables: ObservableSummary, khat: NDArray
) -> NDArray[np.float64]:
    """Evaluate one endpoint-distance coordinate on the native FPR grid."""
    grid = np.arange(observables.n0 + 1, dtype=np.float64) / observables.n0
    if coordinate == "fpr":
        return grid
    if coordinate == "negative_count":
        return np.arange(observables.n0 + 1, dtype=np.float64)
    if coordinate == "fpr_distance":
        return 1.0 - grid
    if coordinate == "negative_distance":
        return np.arange(observables.n0, -1, -1, dtype=np.float64)
    if coordinate == "positive_tail":
        return observables.n1 - np.asarray(khat, dtype=np.float64)
    raise ValueError(f"Unknown region coordinate: {coordinate!r}")


def component_masks(
    artifact: RegionArtifact, *, observables: ObservableSummary, khat: NDArray
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return left and right endpoint components for a frozen rule."""
    if not artifact.support.contains(observables):
        full = np.ones(observables.n0 + 1, dtype=bool)
        return full, full.copy()
    left = coordinate_values(
        artifact.left.coordinate, observables=observables, khat=khat
    ) <= artifact.left.cutoff(observables)
    right = coordinate_values(
        artifact.right.coordinate, observables=observables, khat=khat
    ) <= artifact.right.cutoff(observables)
    return left, right


def fixed_region_masks(
    rule: Literal["probe_legacy", "probe_fpr", "count5"],
    *,
    observables: ObservableSummary,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return the precommitted fixed benchmark components."""
    grid = np.arange(observables.n0 + 1, dtype=np.float64) / observables.n0
    if rule in {"probe_legacy", "probe_fpr"}:
        return grid <= 0.005, grid >= 0.5
    if rule == "count5":
        return np.arange(observables.n0 + 1) <= 5, grid >= 0.5
    raise ValueError(f"Unknown fixed Stage F rule: {rule!r}")


def stitch_hybrid(
    fid_lower: NDArray,
    fid_upper: NDArray,
    m3_lower: NDArray,
    m3_upper: NDArray,
    region: NDArray,
    *,
    closure: Closure = "widening",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Stitch parent bands and apply the selected monotone closure.

    The widening closure is the theorem-preserving Stage F construction.
    The legacy closure exists solely to reproduce the historical probe.
    """
    fid_lower, fid_upper, m3_lower, m3_upper = (
        np.asarray(value, dtype=np.float64)
        for value in (fid_lower, fid_upper, m3_lower, m3_upper)
    )
    region = np.asarray(region, dtype=bool)
    shapes = {
        value.shape for value in (fid_lower, fid_upper, m3_lower, m3_upper, region)
    }
    if len(shapes) != 1:
        raise ValueError("band edges and region must have the same shape")
    lower = np.where(region, np.minimum(fid_lower, m3_lower), fid_lower)
    upper = np.where(region, np.maximum(fid_upper, m3_upper), fid_upper)
    if closure == "widening":
        lower = np.minimum.accumulate(lower[::-1])[::-1]
    elif closure == "legacy":
        lower = np.maximum.accumulate(lower)
    else:
        raise ValueError(f"Unknown closure: {closure!r}")
    upper = np.maximum.accumulate(upper)
    return np.clip(lower, 0.0, 1.0), np.clip(upper, 0.0, 1.0)


def conservative_resample(
    lower: NDArray, upper: NDArray, *, size: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Resample a native-grid band without narrowing its step envelope.

    Args:
        lower: Lower edge on an equally spaced native grid.
        upper: Upper edge on the same grid.
        size: Number of equally spaced requested output points.

    Returns:
        Conservatively floor-indexed lower and ceil-indexed upper edges.
    """
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if lower.ndim != 1 or lower.shape != upper.shape or len(lower) < 2:
        raise ValueError("native band must contain aligned one-dimensional edges")
    if size < 2:
        raise ValueError("resampled grid size must be at least two")
    native_n = len(lower) - 1
    output = np.linspace(0.0, 1.0, size)
    lower_index = np.floor(output * native_n).astype(int)
    upper_index = np.minimum(np.ceil(output * native_n).astype(int), native_n)
    return lower[lower_index], upper[upper_index]


def violation_mask(
    lower: NDArray, upper: NDArray, truth: NDArray, *, tolerance: float = 1e-12
) -> NDArray[np.bool_]:
    """Return grid points at which a band misses the supplied truth."""
    lower, upper, truth = (np.asarray(value) for value in (lower, upper, truth))
    if lower.shape != upper.shape or lower.shape != truth.shape:
        raise ValueError("lower, upper, and truth must have the same shape")
    return (lower > truth + tolerance) | (truth > upper + tolerance)


def encode_array(values: NDArray) -> dict:
    """Encode an array exactly as portable little-endian base64 bytes."""
    array = np.ascontiguousarray(np.asarray(values))
    dtype = array.dtype.newbyteorder("<")
    little = array.astype(dtype, copy=False)
    return {
        "dtype": dtype.str,
        "shape": list(array.shape),
        "data": base64.b64encode(little.tobytes()).decode("ascii"),
    }


def decode_array(payload: dict) -> NDArray:
    """Decode an array created by :func:`encode_array`."""
    raw = base64.b64decode(payload["data"], validate=True)
    array = np.frombuffer(raw, dtype=np.dtype(payload["dtype"]))
    expected = int(np.prod(payload["shape"], dtype=np.int64))
    if array.size != expected:
        raise ValueError("encoded array byte count does not match its shape")
    return array.reshape(payload["shape"]).copy()


def encode_violations(mask: NDArray, *, max_intervals: int = MAX_RLE_INTERVALS) -> dict:
    """Losslessly encode a violation set as intervals or a packed-bit fallback."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("violation mask must be one-dimensional")
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    intervals = [
        [int(start), int(stop)] for start, stop in zip(starts, stops, strict=True)
    ]
    if len(intervals) <= max_intervals:
        return {
            "encoding": "rle",
            "size": len(mask),
            "intervals": intervals,
            "overflow": False,
        }
    packed = np.packbits(mask, bitorder="little")
    return {
        "encoding": "bitset",
        "size": len(mask),
        "data": base64.b64encode(packed.tobytes()).decode("ascii"),
        "overflow": True,
        "interval_count": len(intervals),
    }


def decode_violations(payload: dict) -> NDArray[np.bool_]:
    """Decode either Stage F violation-set representation."""
    size = int(payload["size"])
    if payload["encoding"] == "rle":
        mask = np.zeros(size, dtype=bool)
        for start, stop in payload["intervals"]:
            if not 0 <= start <= stop <= size:
                raise ValueError("invalid violation interval")
            mask[start:stop] = True
        return mask
    if payload["encoding"] == "bitset":
        packed = np.frombuffer(
            base64.b64decode(payload["data"], validate=True), dtype=np.uint8
        )
        return np.unpackbits(packed, bitorder="little", count=size).astype(bool)
    raise ValueError(f"Unknown violation encoding: {payload['encoding']!r}")


def build_record(
    *,
    observables: ObservableSummary,
    khat: NDArray,
    truth: NDArray,
    fiducial: dict[str, tuple[NDArray, NDArray]],
    m3: dict[str, tuple[NDArray, NDArray]],
) -> dict:
    """Build one lossless offline-scoring record from paired parent bands."""
    truth = np.asarray(truth, dtype=np.float64)
    encoded_fiducial = {
        key: {"lower": encode_array(lo), "upper": encode_array(hi)}
        for key, (lo, hi) in fiducial.items()
    }
    encoded_m3 = {
        key: {"lower": encode_array(lo), "upper": encode_array(hi)}
        for key, (lo, hi) in m3.items()
    }
    violations = {
        key: encode_violations(violation_mask(lo, hi, truth))
        for key, (lo, hi) in fiducial.items()
    }
    cumulative = {}
    for key, (mlo, mhi) in m3.items():
        nominal = "0.05" if key in {"0.05", "0.025"} else "0.5"
        flo, fhi = fiducial[nominal]
        increment = np.maximum(fhi, mhi) - np.minimum(flo, mlo) - (fhi - flo)
        cumulative[key] = {
            "prefix": encode_array(np.cumsum(increment, dtype=np.float64)),
            "suffix": encode_array(np.cumsum(increment[::-1], dtype=np.float64)[::-1]),
        }
    return {
        "schema": RECORD_SCHEMA,
        "observables": asdict(observables),
        "khat": encode_array(np.asarray(khat, dtype=np.int32)),
        "truth": encode_array(truth),
        "fiducial": encoded_fiducial,
        "m3": encoded_m3,
        "fiducial_violations": violations,
        "raw_union_increment_cumulative": cumulative,
    }


def decode_band(payload: dict) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Decode a lower/upper band payload."""
    return decode_array(payload["lower"]).astype(np.float64), decode_array(
        payload["upper"]
    ).astype(np.float64)


def record_observables(record: dict) -> ObservableSummary:
    """Load observable inputs from a Stage F replicate record."""
    if record.get("schema") != RECORD_SCHEMA:
        raise ValueError(f"Unknown record schema: {record.get('schema')!r}")
    return ObservableSummary(**record["observables"])
