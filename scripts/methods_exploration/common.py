"""Shared sampling, continuum metrics, and durable experiment records."""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import beta, norm, t

from studroc_paper.methods.fiducial_band_rs import _apply_corner_allowances
from studroc_paper.methods.fiducial_ladder import khat_from_labels, ladder_profile
from studroc_paper.methods.m3_band_rs import _m3_band_from_labels_rs


@dataclass(frozen=True)
class Curve:
    """A PL placement CDF, allowing repeated knots for positive atoms."""

    name: str
    x: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        """Reject invalid CDF geometry before it can contaminate simulation data."""
        if (
            len(self.x) != len(self.y)
            or len(self.x) < 2
            or not np.all(np.isfinite(self.x))
            or not np.all(np.isfinite(self.y))
            or self.x[0] != 0
            or self.x[-1] != 1
            or self.y[0] != 0
            or self.y[-1] != 1
            or np.any(np.diff(self.x) < 0)
            or np.any(np.diff(self.y) < 0)
        ):
            raise ValueError("Expected an ordered CDF from (0,0) to (1,1)")

    def evaluate(self, *, grid: np.ndarray) -> np.ndarray:
        """Evaluate the right-continuous CDF at the reporting coordinates."""
        return np.interp(x=grid, xp=self.x, fp=self.y)

    def inverse(self, *, uniforms: np.ndarray) -> np.ndarray:
        """Invert each positive-mass segment without bridging CDF plateaus."""
        indices = np.searchsorted(self.y, uniforms, side="left")
        indices = np.clip(indices, 1, len(self.y) - 1)
        left = indices - 1
        mass = self.y[indices] - self.y[left]
        fraction = np.divide(
            uniforms - self.y[left],
            mass,
            out=np.zeros_like(uniforms, dtype=float),
            where=mass > 0,
        )
        return self.x[left] + fraction * (self.x[indices] - self.x[left])


def curve(*, name: str, n0: int, n1: int) -> Curve:
    """Construct fixed smooth truths and sample-size-dependent sliver stresses."""
    grid = np.unique(
        np.r_[
            0,
            np.geomspace(1e-7, 0.01, 48),
            np.linspace(0.01, 0.99, 257),
            1 - np.geomspace(1e-7, 0.01, 48),
            1,
        ]
    )
    if name == "diagonal":
        return Curve(name=name, x=np.array([0.0, 1.0]), y=np.array([0.0, 1.0]))
    if name.startswith("normal_") or name.startswith("hetero_"):
        auc = float(name.split("_")[1])
        scale = 2.0 if name.startswith("hetero_") else 1.0
        shift = np.sqrt(1 + scale**2) * norm.ppf(auc)
        values = norm.sf((norm.isf(grid) - shift) / scale)
    elif name.startswith("t2_"):
        from scripts.c_calibration.shapes import make_t_shape

        source = make_t_shape(auc=float(name.split("_")[1]), df=2)
        return Curve(name=name, x=source.t, y=source.r)
    elif name == "kink":
        return Curve(
            name=name, x=np.array([0.0, 0.004, 1.0]), y=np.array([0.0, 0.6, 1.0])
        )
    elif name in {"sliver", "interior_sliver"}:
        end = 1.0 if name == "sliver" else 0.6
        mass = min(0.1, 0.8 / n1)
        width = min(0.05, 1 / n0)
        if name == "interior_sliver":
            return Curve(
                name=name,
                x=np.array([0.0, 0.1, 0.6 - width, 0.6, 0.85, 1.0]),
                y=np.array([0.0, 0.8 - mass, 0.8 - mass, 0.8, 0.8, 1.0]),
            )
        return Curve(
            name=name,
            x=np.array([0.0, 0.1, end - width, end, 1.0]),
            y=np.array([0.0, 1 - mass, 1 - mass, 1.0, 1.0]),
        )
    elif name == "jump":
        return Curve(
            name=name,
            x=np.array([0.0, 0.5, 0.5, 1.0]),
            y=np.array([0.0, 0.0, 1.0, 1.0]),
        )
    else:
        raise ValueError(f"Unknown curve {name}")
    return Curve(name=name, x=grid, y=values)


def rng_for(*parts: object) -> np.random.Generator:
    """Derive a stable independent stream without Python's randomized hash."""
    digest = hashlib.sha256(json.dumps(parts).encode()).digest()
    return np.random.default_rng(seed=int.from_bytes(digest[:16], "little"))


def sample(*, truth: Curve, n0: int, n1: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ranks from uniform negatives and the exact PL positive inverse."""
    placements = np.r_[rng.random(n0), truth.inverse(uniforms=rng.random(n1))]
    return np.r_[np.zeros(n0, dtype=np.uint8), np.ones(n1, dtype=np.uint8)][
        np.argsort(placements, kind="stable")
    ]


def runs(*, mask: np.ndarray) -> list[list[int]]:
    """Encode every true run as a half-open interval of grid indices."""
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return edges.reshape(-1, 2).tolist()


def metrics(
    *, grid: np.ndarray, lower: np.ndarray, upper: np.ndarray, truth: Curve
) -> dict:
    """Measure conservative continuum area and grid-implied coverage."""
    values = truth.evaluate(grid=grid)
    below, above = values < lower - 1e-12, values > upper + 1e-12
    widths = upper[1:] - lower[:-1]
    result = {
        "covered": not bool(np.any(below | above)),
        "below": bool(np.any(below)),
        "above": bool(np.any(above)),
        "both": bool(np.any(below) and np.any(above)),
        "miss_depth": float(max(0, np.max(lower - values), np.max(values - upper))),
        "low_runs": runs(mask=below),
        "high_runs": runs(mask=above),
        "area": float(np.dot(np.diff(grid), widths)),
    }
    for name, left, right in [
        ("left", 0, 0.02),
        ("interior", 0.02, 0.95),
        ("right", 0.95, 1),
    ]:
        lengths = np.maximum(
            0, np.minimum(grid[1:], right) - np.maximum(grid[:-1], left)
        )
        result[f"{name}_width"] = float(np.dot(lengths, widths) / (right - left))
        result[f"{name}_miss"] = bool(
            np.any((below | above) & (grid >= left) & (grid <= right))
        )
    return result


def interval(*, successes: int, count: int, confidence: float = 0.95) -> list[float]:
    """Return an exact two-sided binomial interval, including all/no successes."""
    tail = (1 - confidence) / 2
    return [
        0.0
        if successes == 0
        else float(beta.ppf(tail, successes, count - successes + 1)),
        1.0
        if successes == count
        else float(beta.ppf(1 - tail, successes + 1, count - successes)),
    ]


def paired_summary(*, values: list[float]) -> dict:
    """Summarize paired replicate ratios with a Student-t uncertainty interval."""
    data = np.asarray(values)
    mean = float(data.mean())
    se = float(data.std(ddof=1) / np.sqrt(len(data))) if len(data) > 1 else None
    radius = float(t.ppf(0.975, len(data) - 1) * se) if se is not None else None
    return {
        "mean": mean,
        "se": se,
        "ci": [mean - radius, mean + radius] if radius is not None else None,
    }


def fiducial_edges(
    *,
    labels: np.ndarray,
    truth: Curve,
    draws: int,
    seed: int,
    alphas: list[float],
    trim_rows: np.ndarray | None,
    threads: int,
) -> tuple[list[tuple], np.ndarray]:
    """Read requested levels directly from one ladder profile and add allowances."""
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    profile = ladder_profile(
        lab_s=labels,
        rtrue=truth.evaluate(grid=np.arange(n0 + 1) / n0),
        n_draws=draws,
        seed=seed,
        ladder=np.array([1]),
        alpha_effs=alphas,
        trim_rows=trim_rows,
        return_edges=True,
        n_threads=threads,
    )
    depths, lower, upper = profile.edges
    khat = khat_from_labels(lab_s=labels)
    result = []
    for depth in profile.ref_j:
        index = int(np.flatnonzero(depths == depth)[0])
        result.append(
            _apply_corner_allowances(
                lower=lower[index].copy(),
                upper=upper[index].copy(),
                khat=khat,
                n1=n1,
                trim_depth=int(depth),
                n_draws=draws,
            )
        )
    return result, profile.ref_j


def m3_edges(*, labels: np.ndarray, alpha: float) -> tuple:
    """Build the production exact reference from the paired rank sequence."""
    return _m3_band_from_labels_rs(lab_s=labels, alpha=alpha)


class Store:
    """Append complete JSON records and resume only identical track inputs."""

    def __init__(self, *, directory: Path, config: dict) -> None:
        """Validate the run identity before loading any resumable observations."""
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        manifest = directory / "config.json"
        if manifest.exists() and json.loads(manifest.read_text()) != config:
            raise ValueError(f"Refusing incompatible resume: {manifest}")
        manifest.write_text(json.dumps(config, indent=2))
        self.path = directory / "records.jsonl"
        self.records = []
        if self.path.exists():
            raw = self.path.read_bytes()
            complete = raw.rsplit(b"\n", 1)[0] + b"\n" if b"\n" in raw else b""
            if not raw.endswith(b"\n"):
                self.path.write_bytes(complete)
            self.records = [json.loads(line) for line in complete.splitlines()]
        self.keys = {row["key"] for row in self.records}
        self.started = time.monotonic()

    def add(self, *, key: str, **record: object) -> None:
        """Flush one indivisible experimental unit before marking it complete."""
        if key in self.keys:
            return
        row = {"key": key, **record}
        with self.path.open("a") as output:
            output.write(json.dumps(row, allow_nan=False) + "\n")
            output.flush()
        self.records.append(row)
        self.keys.add(key)

    def save(self, *, name: str, payload: dict) -> None:
        """Atomically replace a small summary or frozen-candidate artifact."""
        target = self.directory / name
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
        temporary.replace(target)


def resample(
    *, source: np.ndarray, lower: np.ndarray, upper: np.ndarray, target: np.ndarray
) -> tuple:
    """Resample the conservative step band without interpolating its edges."""
    left = np.clip(
        np.searchsorted(source, target, side="right") - 1, 0, len(source) - 1
    )
    right = np.clip(np.searchsorted(source, target, side="left"), 0, len(source) - 1)
    return lower[left], upper[right]
