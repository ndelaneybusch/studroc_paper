"""Per-cell execution engine of the trim-exponent calibration study.

For each cell (shape, n0, n1) the runner simulates replicates in rank space
(negatives iid Uniform, positives iid from the true curve — exact by rank
invariance), computes each replicate's full coverage-vs-depth ladder profile
through the ``fiducial_core`` Rust kernel, and stores the raw per-rep
profiles plus aggregates:

- per ladder depth j: coverage of the allowance-augmented band, areas,
  miss diagnostics;
- the reference maps (C = 1, C = 2, and the auto map — provisional in
  Stage A, frozen in Stage B) evaluated per rep exactly the way production
  selects the depth;
- the draw-depth CDF at the ladder and the truth's own depth (raw-tube
  coverage identity, allowance attribution, D6 diagnostics);
- per-alpha calibration estimands j*, alpha_eff*, C*, ell* with bootstrap
  CIs, the saturation flag, and the SE-gated top-up rule of spec section 4.

Replicates are seeded deterministically per (cell, rep); reps within a cell
run concurrently on a thread pool (the Rust kernel releases the GIL), each
kernel call on its own rayon pool. Raw profiles are written as
``<cell>.json.gz`` and aggregates as ``<cell>.summary.json``; completed
cells are skipped on re-run, partially completed cells are extended.
"""

import gzip
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from design import (  # noqa: E402
    CSTAR_SE_ALPHA_MAX,
    CSTAR_SE_TARGET,
    STUDY_SEED,
    Cell,
    RefArm,
    reference_arms,
    rep_seed_sequence,
)
from shapes import (  # noqa: E402
    get_curve,
    make_trapezoid,
    quantize_jitter,
    shape_registry,
)

from studroc_paper.methods.fiducial_ladder import (  # noqa: E402
    ladder_profile,
    make_ladder,
)

N_BOOT = 1_000
TOPUP_BATCH = 1_000


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def provenance() -> dict:
    """Git hash, package versions, and timestamp for output stamping."""
    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[2],
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = "unknown"
    versions = {}
    for pkg in ("numpy", "scipy", "fiducial-core", "studroc_paper"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "unknown"
    return {
        "git_hash": git_hash,
        "versions": versions,
        "python": sys.version.split()[0],
        "study_seed": STUDY_SEED,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


# ---------------------------------------------------------------------------
# per-rep simulation
# ---------------------------------------------------------------------------


def truth_curve(cell: Cell):
    """The evaluation truth of a cell: the shape's curve, or its trapezoid
    at the cell's quantization level (ties cells; random tie-break makes
    the trapezoid the exact estimand)."""
    base = get_curve(cell.shape)
    if cell.quantize is not None:
        return make_trapezoid(base, cell.quantize)
    return base


def sample_labels(cell: Cell, rep: int) -> tuple[np.ndarray, int]:
    """Simulate one replicate's merged label sequence in rank space.

    Returns the 0/1 label sequence in ascending rank order and the kernel
    seed, both deterministic in (cell, rep).
    """
    rng = np.random.default_rng(rep_seed_sequence(cell, rep))
    curve = get_curve(cell.shape)
    u = rng.random(cell.n0)
    w = curve.inv(rng.random(cell.n1))
    if cell.quantize is not None:
        u, w = quantize_jitter(u, w, cell.quantize, rng)
    lab = np.concatenate(
        [np.zeros(cell.n0, dtype=np.uint8), np.ones(cell.n1, dtype=np.uint8)]
    )
    order = np.argsort(np.concatenate([u, w]), kind="stable")
    seed = int(rng.integers(0, 2**64, dtype=np.uint64))
    return lab[order], seed


def run_rep(
    cell: Cell,
    rep: int,
    *,
    rtrue: np.ndarray,
    ladder: np.ndarray,
    arms: list[RefArm],
    m_draws: int,
    n_threads: int,
) -> dict:
    """One replicate: sample, profile, and compact the result."""
    lab_s, seed = sample_labels(cell, rep)
    prof = ladder_profile(
        lab_s,
        rtrue=rtrue,
        n_draws=m_draws,
        seed=seed,
        ladder=ladder,
        alpha_effs=[arm.alpha_eff for arm in arms],
        trim_rows="production",
        n_threads=n_threads,
    )
    return {
        "covered": prof.covered.astype(np.uint8),
        "viol_low": prof.viol_low.astype(np.uint8),
        "viol_high": prof.viol_high.astype(np.uint8),
        "miss_depth": prof.miss_depth.astype(np.float32),
        "worst_k": prof.worst_k.astype(np.int64),
        "area": prof.area.astype(np.float64),
        "area_raw": prof.area_raw.astype(np.float64),
        "depth_cdf": prof.depth_cdf_at(prof.ladder),
        "truth_depth_low": prof.truth_depth_low,
        "truth_depth_high": prof.truth_depth_high,
        "ref_j": prof.ref_j,
        "ref_covered": prof.ref_covered.astype(np.uint8),
        "ref_viol_low": prof.ref_viol_low.astype(np.uint8),
        "ref_viol_high": prof.ref_viol_high.astype(np.uint8),
        "ref_miss_depth": prof.ref_miss_depth.astype(np.float32),
        "ref_worst_k": prof.ref_worst_k.astype(np.int64),
        "ref_area": prof.ref_area.astype(np.float64),
    }


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def _stack(records: list[dict], key: str) -> np.ndarray:
    return np.stack([r[key] for r in records])


def bootstrap_coordinates(
    cov: np.ndarray,
    depth_cdf: np.ndarray,
    ladder: np.ndarray,
    alpha: float,
    m_draws: int,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> dict:
    """Point estimates and bootstrap intervals of the calibration estimands.

    j* = max ladder depth whose realized coverage is >= 1 - alpha;
    alpha_eff* = mean fraction of draws with depth < j*;
    C* = log(1 - alpha_eff*) / log(1 - alpha); ell* = j* / (M + 1).
    The bootstrap resamples reps (multinomial weights, ``n_boot`` draws).

    Args:
        cov: (reps, J) coverage indicators over the ladder.
        depth_cdf: (reps, J) counts of draws with depth below each ladder j.
        ladder: (J,) trim depths.
        alpha: Nominal level.
        m_draws: Cloud size M.
        rng: Bootstrap randomness.
        n_boot: Number of bootstrap resamples.

    Returns:
        Estimates with 95% bootstrap intervals and SEs, plus flags.
    """
    reps = cov.shape[0]
    covm = cov.mean(axis=0)
    target = 1.0 - alpha

    def coords(cov_mean: np.ndarray, cdf_mean_frac: np.ndarray) -> tuple:
        ok = np.flatnonzero(cov_mean >= target)
        if len(ok) == 0:
            return None, None, None, None
        idx = int(ok.max())
        j_star = int(ladder[idx])
        aeff = float(cdf_mean_frac[idx])
        # Guard the log for degenerate aeff estimates.
        aeff_c = min(max(aeff, 1e-12), 1.0 - 1e-12)
        c_star = float(np.log1p(-aeff_c) / np.log1p(-alpha))
        ell = j_star / (m_draws + 1.0)
        return j_star, aeff, c_star, ell

    cdf_frac = depth_cdf / m_draws
    j_star, aeff_star, c_star, ell_star = coords(covm, cdf_frac.mean(axis=0))

    weights = rng.multinomial(reps, np.full(reps, 1.0 / reps), size=n_boot)
    wf = weights.astype(np.float64) / reps
    cov_b = wf @ cov  # (n_boot, J)
    cdf_b = wf @ cdf_frac
    boot = {"j": [], "aeff": [], "c": [], "ell": []}
    for b in range(n_boot):
        jb, ab, cb, eb = coords(cov_b[b], cdf_b[b])
        if jb is not None:
            boot["j"].append(jb)
            boot["aeff"].append(ab)
            boot["c"].append(cb)
            boot["ell"].append(eb)

    def ci(vals: list) -> dict:
        if not vals:
            return {"se": None, "lo": None, "hi": None}
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "se": float(arr.std(ddof=1)) if len(arr) > 1 else None,
            "lo": float(np.quantile(arr, 0.025)),
            "hi": float(np.quantile(arr, 0.975)),
        }

    saturated = j_star is not None and j_star == int(ladder[0])
    unconstrained = covm[-1] >= target  # never dips below 1 - alpha (D6)
    return {
        "alpha": alpha,
        "j_star": j_star,
        "alpha_eff_star": aeff_star,
        "c_star": c_star,
        "ell_star": ell_star,
        "boot_feasible_frac": len(boot["c"]) / n_boot,
        "c_star_ci": ci(boot["c"]),
        "alpha_eff_star_ci": ci(boot["aeff"]),
        "ell_star_ci": ci(boot["ell"]),
        "j_star_ci": ci(boot["j"]),
        "saturated": bool(saturated),
        "unconstrained": bool(unconstrained),
        "infeasible": j_star is None,
    }


def aggregate(
    cell: Cell, records: list[dict], ladder: np.ndarray, arms: list[RefArm]
) -> dict:
    """Aggregate raw per-rep profiles into the per-cell summary."""
    reps = len(records)
    cov = _stack(records, "covered").astype(np.float64)
    depth_cdf = _stack(records, "depth_cdf").astype(np.float64)
    t_lo = np.array([r["truth_depth_low"] for r in records])
    t_hi = np.array([r["truth_depth_high"] for r in records])
    truth_depth = np.minimum(t_lo, t_hi)

    rng = np.random.default_rng(rep_seed_sequence(cell, 10**9))
    per_alpha = {}
    for alpha in cell.alphas:
        est = bootstrap_coordinates(
            cov, depth_cdf, ladder, alpha, cell.m_draws, rng
        )
        if est["j_star"] is not None:
            idx = int(np.flatnonzero(ladder == est["j_star"])[0])
            raw_cov = truth_depth >= est["j_star"]
            est["allowance_attribution"] = float(
                np.mean(cov[:, idx].astype(bool) & ~raw_cov)
            )
        per_alpha[f"{alpha:g}"] = est

    ref_j = _stack(records, "ref_j")
    ref_cov = _stack(records, "ref_covered").astype(np.float64)
    ref_area = _stack(records, "ref_area")
    ref_vl = _stack(records, "ref_viol_low").astype(np.float64)
    ref_vh = _stack(records, "ref_viol_high").astype(np.float64)
    ref_miss = _stack(records, "ref_miss_depth").astype(np.float64)
    ref_maps = []
    for a, arm in enumerate(arms):
        covm = float(ref_cov[:, a].mean())
        ref_maps.append(
            {
                "label": arm.label,
                "alpha": arm.alpha,
                "exponent": arm.exponent,
                "alpha_eff": arm.alpha_eff,
                "coverage": covm,
                "coverage_se": float(np.sqrt(covm * (1 - covm) / reps)),
                "area": float(ref_area[:, a].mean()),
                "viol_low": float(ref_vl[:, a].mean()),
                "viol_high": float(ref_vh[:, a].mean()),
                "mean_miss_depth_missers": float(ref_miss[ref_miss[:, a] > 0, a].mean())
                if (ref_miss[:, a] > 0).any()
                else 0.0,
                "mean_j": float(ref_j[:, a].mean()),
                "min_j": int(ref_j[:, a].min()),
                "frac_j_below_3": float((ref_j[:, a] < 3).mean()),
            }
        )

    return {
        "reps": reps,
        "cov_by_j": cov.mean(axis=0).tolist(),
        "area_by_j": _stack(records, "area").mean(axis=0).tolist(),
        "area_raw_by_j": _stack(records, "area_raw").mean(axis=0).tolist(),
        "ladder": ladder.tolist(),
        "per_alpha": per_alpha,
        "ref_maps": ref_maps,
        "truth_depth_quantiles": {
            q: float(np.quantile(truth_depth, float(q)))
            for q in ("0.01", "0.05", "0.5")
        },
    }


def se_gate_needs_topup(cell: Cell, agg: dict) -> bool:
    """Apply the stage-specific precision gate for adaptive top-up.

    Stage S tops up only for its primary alpha=.05 decision. Stage A retains
    the full spec-section-4 gate, SE(C*) <= 0.15 at every fitted alpha <=.2.
    """
    for key, est in agg["per_alpha"].items():
        alpha = float(key)
        if (
            (cell.stage == "S" and alpha != 0.05)
            or alpha > CSTAR_SE_ALPHA_MAX
            or est["infeasible"]
            or est["saturated"]
        ):
            continue
        se = est["c_star_ci"]["se"]
        if se is not None and se > CSTAR_SE_TARGET:
            return True
    return False


# ---------------------------------------------------------------------------
# cell driver
# ---------------------------------------------------------------------------


def _record_to_json(record: dict) -> dict:
    out = {}
    for key, val in record.items():
        if isinstance(val, np.ndarray):
            if val.dtype.kind == "f":
                out[key] = [round(float(v), 7) for v in val]
            else:
                out[key] = val.tolist()
        else:
            out[key] = val
    return out


def _record_from_json(record: dict) -> dict:
    return {
        key: (np.asarray(val) if isinstance(val, list) else val)
        for key, val in record.items()
    }


def cell_paths(out_dir: Path, cell: Cell) -> tuple[Path, Path]:
    return out_dir / f"{cell.name}.json.gz", out_dir / f"{cell.name}.summary.json"


def load_existing(out_dir: Path, cell: Cell) -> tuple[list[dict], dict | None]:
    """Load previously computed raw records for a cell, if compatible."""
    raw_path, _ = cell_paths(out_dir, cell)
    if not raw_path.exists():
        return [], None
    with gzip.open(raw_path, "rt") as fh:
        payload = json.load(fh)
    meta = payload["meta"]
    if meta["m_draws"] != cell.m_draws or meta["study_seed"] != STUDY_SEED:
        raise RuntimeError(
            f"{cell.name}: existing output was produced under a different "
            f"design (M={meta['m_draws']} vs {cell.m_draws}); refusing to "
            "mix. Move the old file aside or pass --force."
        )
    return [_record_from_json(r) for r in payload["records"]], meta


def run_cell(
    cell: Cell,
    out_dir: Path,
    *,
    workers: int,
    threads_per_call: int,
    mem_gb: float = 40.0,
    auto_exponent_fn=None,
    m_scale: float = 1.0,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    """Run (or extend) one cell and write its raw + summary outputs.

    Args:
        cell: The cell design.
        out_dir: Output directory (created if missing).
        workers: Concurrent replicates. Automatically reduced so that
            ``workers * cloud_bytes`` stays within ``mem_gb``.
        threads_per_call: Rayon threads per kernel call.
        mem_gb: Memory budget for concurrent clouds.
        auto_exponent_fn: ``f(n0, n1, alpha) -> C`` for the auto reference
            arm (Stage B: the frozen map). ``None`` uses the provisional
            formula.
        m_scale: Multiplier on the budgeted M (saturated-cell re-runs).
        force: Discard any existing incompatible output for this cell.
        verbose: Print progress lines.

    Returns:
        The cell summary dict (also written to ``<cell>.summary.json``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if m_scale != 1.0:
        cell = Cell(**{**cell.__dict__, "m_draws": int(cell.m_draws * m_scale)})
    raw_path, summary_path = cell_paths(out_dir, cell)

    try:
        records, prior_meta = load_existing(out_dir, cell)
    except RuntimeError:
        if not force:
            raise
        records, prior_meta = [], None

    curve = truth_curve(cell)
    rtrue = np.clip(curve.eval(np.arange(cell.n_grid) / cell.n0), 0.0, 1.0)
    ladder = make_ladder(cell.m_draws)
    arms = reference_arms(cell.alphas, cell.n0, cell.n1, auto_exponent_fn)

    cloud_gb = cell.cloud_bytes() / 2**30
    workers_eff = max(1, min(workers, int(mem_gb / max(cloud_gb, 1e-9))))

    def extend_to(target: int) -> None:
        todo = [r for r in range(target) if r >= len(records)]
        if not todo:
            return
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers_eff) as pool:
            futures = [
                pool.submit(
                    run_rep,
                    cell,
                    rep,
                    rtrue=rtrue,
                    ladder=ladder,
                    arms=arms,
                    m_draws=cell.m_draws,
                    n_threads=threads_per_call,
                )
                for rep in todo
            ]
            for i, fut in enumerate(futures):
                records.append(fut.result())
                if verbose and (i + 1) % 100 == 0:
                    rate = (i + 1) / (time.time() - t0)
                    print(
                        f"  [{cell.name}] rep {i + 1}/{len(todo)} "
                        f"({rate:.2f} reps/s)",
                        flush=True,
                    )

    t_start = time.time()
    extend_to(cell.reps)
    agg = aggregate(cell, records, ladder, arms)
    while (
        cell.stage in ("S", "A")
        and se_gate_needs_topup(cell, agg)
        and len(records) < cell.reps_max
    ):
        target = min(len(records) + TOPUP_BATCH, cell.reps_max)
        if verbose:
            print(f"  [{cell.name}] SE gate: topping up to {target} reps", flush=True)
        extend_to(target)
        agg = aggregate(cell, records, ladder, arms)

    meta = {
        "cell": {
            "name": cell.name,
            "stage": cell.stage,
            "arm": cell.arm,
            "shape": cell.shape,
            "shape_meta": shape_registry()[cell.shape].meta,
            "n0": cell.n0,
            "n1": cell.n1,
            "alphas": list(cell.alphas),
            "quantize": cell.quantize,
            "notes": cell.notes,
        },
        "m_draws": cell.m_draws,
        "m_scale": m_scale,
        "study_seed": STUDY_SEED,
        "true_auc": curve.auc(),
        "ladder_size": len(ladder),
        "ref_arms": [
            {"label": a.label, "alpha": a.alpha, "exponent": a.exponent} for a in arms
        ],
        "reps_done": len(records),
        "runtime_s": round(time.time() - t_start, 1)
        + (prior_meta or {}).get("runtime_s", 0.0),
        "provenance": provenance(),
    }

    with gzip.open(raw_path, "wt") as fh:
        json.dump(
            {"meta": meta, "records": [_record_to_json(r) for r in records]}, fh
        )
    summary = {"meta": meta, "aggregate": agg}
    summary_path.write_text(json.dumps(summary, indent=1))
    if verbose:
        flagged = [
            f"alpha={k}: {'saturated' if v['saturated'] else ''}"
            f"{'infeasible' if v['infeasible'] else ''}"
            for k, v in agg["per_alpha"].items()
            if v["saturated"] or v["infeasible"]
        ]
        print(
            f"[{cell.name}] done: {len(records)} reps, M={cell.m_draws}, "
            f"{meta['runtime_s']:.0f}s"
            + (f"; FLAGS: {'; '.join(flagged)}" if flagged else ""),
            flush=True,
        )
    return summary
