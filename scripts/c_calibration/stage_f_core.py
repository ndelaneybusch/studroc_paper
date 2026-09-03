"""Statistical and persistence primitives for the Stage F frontier-floor study.

This module contains no simulation design or I/O. It provides observable
summaries, region evaluation, both stitch closures, exact array serialization,
and lossless violation-set encoding.
"""

from __future__ import annotations

import base64
import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

RECORD_SCHEMA = "stage-f-replicate/v1"
MAX_RLE_INTERVALS = 64
AUC_BOUND_DELTA = 0.05

Closure = Literal["widening", "legacy"]
FrontierRule = Literal["frontier_floor_v1", "frontier_run0", "frontier_j1"]


@dataclass(frozen=True)
class ObservableSummary:
    """Dataset-only quantities retained for diagnostics and rank routing.

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


def frontier_left_cutoff(*, n0: int, m_draws: int) -> int:
    """Return the last left-frontier count resolved by a cloud of size ``M``.

    Args:
        n0: Number of negative observations.
        m_draws: Fiducial Monte Carlo cloud size.

    Returns:
        Inclusive native-grid cutoff ``min(n0, ceil(log(M + 1)))``.
    """
    if n0 < 1 or m_draws < 1:
        raise ValueError("n0 and m_draws must be positive")
    return min(n0, math.ceil(math.log(m_draws + 1)))


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


def component_masks(
    *, observables: ObservableSummary, khat: NDArray, m_draws: int
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return components of the primary frontier rule."""
    return frontier_region_masks(
        "frontier_floor_v1", observables=observables, khat=khat, m_draws=m_draws
    )


def frontier_region_masks(
    rule: FrontierRule, *, observables: ObservableSummary, khat: NDArray, m_draws: int
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return the left frontier and selected saturated-run component.

    Args:
        rule: Primary rule or one of its two prespecified right-edge ablations.
        observables: Rank-derived sample summary; only class sizes are consumed.
        khat: Empirical positive-count map on the native negative grid.
        m_draws: Fiducial cloud size, which fixes the left cutoff.

    Returns:
        Boolean left and right component masks.

    Raises:
        ValueError: If the empirical rank map is malformed.
    """
    khat = np.asarray(khat, dtype=np.int64)
    if (
        khat.shape != (observables.n0 + 1,)
        or khat[0] < 0
        or khat[-1] != observables.n1
        or np.any(np.diff(khat) < 0)
        or np.any(khat > observables.n1)
    ):
        raise ValueError("khat must be a valid empirical count map")
    indices = np.arange(observables.n0 + 1)
    left = indices <= frontier_left_cutoff(n0=observables.n0, m_draws=m_draws)
    positive_tail = observables.n1 - khat
    if rule == "frontier_run0":
        return left, positive_tail == 0
    if rule == "frontier_j1":
        return left, positive_tail <= 1
    if rule != "frontier_floor_v1":
        raise ValueError(f"Unknown frontier rule: {rule!r}")
    saturated = np.flatnonzero(positive_tail == 0)
    if not len(saturated):
        raise ValueError("khat must reach n1 at the final native-grid point")
    k_sat = int(saturated[0])
    run_length = observables.n0 - k_sat
    margin = math.ceil(2.0 * math.sqrt(max(run_length, 1)))
    right = indices >= max(0, k_sat - margin)
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
