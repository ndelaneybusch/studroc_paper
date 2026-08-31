"""Post-Stage-S follow-up runs (spec OUTCOME + follow-up entries, rev. 2026-08-31).

Stage S returned STOP on the auto-map effort; these items are what remains
worth running on this infrastructure (see the dated follow-up entry in
``stats/c_calibration_spec.md``, revised 2026-08-31 after review):

1. ``boundary``   — locate the small-n heavy-tail validity boundary of the
                    C = 1 default (t(2)/.95 broke at n = 100, passed at
                    500; probe n in between, a heavier tail, a higher AUC).
                    Coverage-driven sequential replication; the report
                    produces a conservative, library-relative global
                    routing threshold.
2. ``heldout``    — designer-bias guard: validate C = 1 (plus the exact M3
                    band's width economics) on the six held-out shapes at
                    n = 500, with a mechanism-diverse sentinel subset at
                    n = 5,000 and the ties regression cell.
3. ``composite``  — derisk a *finite-range* composite band (corners widened
                    to the untrimmed cloud envelope — an empirical widening,
                    not an exact bound; interior min-p-trimmed at C > 1) by
                    building the actual stitched band per rep. Theorem 7
                    forces interior coverage -> (1-alpha)^C for fixed C > 1,
                    so no fixed-C composite can be an unrestricted method:
                    the declared question is whether a candidate exists on a
                    declared range, with C_int clamped to 1 (or tapered)
                    above it. A reduced-config large-n sentinel measures the
                    erosion direction.
4. ``imbalance``  — DEFERRED (not part of ``all``): imbalance with
                    min(n0, n1) > 500. Stage S found C = 1 at nominal under
                    more severe imbalance (minority 500); run only if the
                    final-run guidance turns out to need it.

``report`` aggregates whatever has finished into ``followup_report.md``;
``all`` runs boundary, heldout, composite, report. Outputs land under
``data/results/c_calibration_followup_20260830/<item>/``. Runner cells
resume and extend exactly like the Stage S runner (same seeding
discipline: deterministic in (study seed, stage, cell name, rep)).

Decision rules (predeclared; the A1-letter noninferiority bar):

- Per cell, the C = 1 arm at alpha = .05 PASSES iff point coverage >= .94
  AND its Wilson-95% lower bound >= .925; it is MARGINAL if the Wilson CI
  still straddles .94 at the replication cap; FAIL otherwise. The strict
  >= .95 point bar is reported alongside.
- Sequential replication: cells top up in batches while the Wilson CI
  straddles .94 and the cap is not reached (boundary/imbalance:
  1,000 -> 3,000; heldout: 2,000 -> 4,000; composite: 500 -> 2,000).
- A composite config is a candidate iff every cell PASSES the bar above
  AND its pooled paired width change vs the full-curve C = 1 arm is
  negative (paired per-rep differences; pooled SE reported). "No
  survivor" is evidence against this coarse (cut x C) family only, not
  against the composite idea.
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from design import (  # noqa: E402
    ALPHAS_SCREEN,
    Cell,
    m_budget,
    rep_seed_sequence,
    summarize,
)
from runner import provenance, run_cell, truth_curve  # noqa: E402
from shapes import (  # noqa: E402
    ShapeSpec,
    get_curve,
    make_t_shape,
    quantize_jitter,
    shape_registry,
)

from studroc_paper.methods.fiducial_band_rs import fiducial_band_rs  # noqa: E402
from studroc_paper.methods.m3_band_rs import m3_band_rs  # noqa: E402

FOLLOWUP_DATE = "20260830"
DEFAULT_OUT = Path("data/results") / f"c_calibration_followup_{FOLLOWUP_DATE}"

# The A1-letter noninferiority bar (alpha = .05, C = 1 arm).
BAR_POINT = 0.94
BAR_CI_LO = 0.925
BAR_STRICT = 0.95

# Sequential replication (start, batch, cap) per item.
PROBE_REPS = (1_000, 1_000, 3_000)  # boundary + imbalance
HELDOUT_REPS = (2_000, 1_000, 4_000)
COMPOSITE_REPS = (500, 500, 2_000)
SENTINEL_REPS = 250  # large-n composite sentinels: fixed, no top-up

M3_ALPHAS = (0.5, 0.05)

# Composite design knobs (corner cut in FPR units, interior trim exponent).
COMPOSITE_BOUNDARIES = ((0.02, 0.95), (0.05, 0.90), (0.10, 0.85))
COMPOSITE_INTERIOR_C = (1.5, 2.0, 2.5)
SENTINEL_BOUNDARIES = ((0.05, 0.90),)
SENTINEL_INTERIOR_C = (1.5, 2.0)
CORNER_EXPONENT = 1e-4  # alpha_eff << 1/M: the untrimmed cloud envelope
COV_TOL = 1e-9

# Stage S C=1 coverage at alpha=.05 for the composite parity check
# (data/results/c_calibration_20260829/stageS/*.summary.json).
STAGE_S_C1_COV = {
    "composite--t2_95--n100x100": (0.802, 500),
    "composite--t2_95--n500x500": (0.958, 2000),
    "composite--t2_95--n5000x5000": (0.968, 2000),
    "composite--t2_95--n500x4500": (0.950, 2000),
    "composite--binormal_95--n500x500": (0.982, 2000),
    "composite--binormal_95--n5000x5000": (0.972, 2000),
    "composite--kink_80--n500x500": (0.973, 2000),
    "composite--trapezoid_q10_90--n500x500": (0.981, 2000),
    "composite--binormal_60--n500x500": (0.987, 2000),
}


def wilson_ci(p_hat: float, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson 95% score interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 1.0
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def classify(p_hat: float, n: int) -> str:
    """PASS / MARGINAL / FAIL against the A1-letter bar."""
    lo, hi = wilson_ci(p_hat, n)
    if p_hat >= BAR_POINT and lo >= BAR_CI_LO:
        return "PASS"
    if lo < BAR_POINT <= hi:
        return "MARGINAL"
    return "FAIL"


def needs_topup(p_hat: float, n: int) -> bool:
    """Sequential rule: keep replicating while the CI straddles the bar."""
    lo, hi = wilson_ci(p_hat, n)
    return lo < BAR_POINT <= hi


# ---------------------------------------------------------------------------
# extra shapes (boundary probes)
# ---------------------------------------------------------------------------


def register_followup_shapes() -> None:
    """Add the two boundary-probe shapes to the (cached, mutable) registry."""
    registry = shape_registry()
    for spec in (
        ShapeSpec(
            name="t15_95",
            role="followup",
            build=lambda: make_t_shape(0.95, df=1.5),
            meta={
                "family": "student_t",
                "auc": 0.95,
                "df": 1.5,
                "note": "heavier tail than the library's t(2): does the "
                "small-n validity boundary move up?",
            },
        ),
        ShapeSpec(
            name="t2_99",
            role="followup",
            build=lambda: make_t_shape(0.99, df=2.0),
            meta={
                "family": "student_t",
                "auc": 0.99,
                "df": 2.0,
                "note": "higher AUC at t(2) tails: same question via the AUC axis",
            },
        ),
    ):
        registry.setdefault(spec.name, spec)


# ---------------------------------------------------------------------------
# cell lists
# ---------------------------------------------------------------------------


def boundary_cells() -> list[Cell]:
    """Item 1: locate the C = 1 small-n validity boundary."""
    grid = [
        # t(2)/.95: broken at 100 (cov .802), fine at 500 (.958)
        ("t2_95", (150, 250, 350)),
        # milder tail: was never measured below 500
        ("t3_90", (100, 250)),
        # heavier tail / higher AUC: does the boundary move above 500?
        ("t15_95", (250, 500)),
        ("t2_99", (250, 500)),
    ]
    start, _, cap = PROBE_REPS
    cells = []
    for shape, ns in grid:
        for n in ns:
            cells.append(
                _cell(
                    name=f"boundary--{shape}--n{n}",
                    stage="S",
                    arm="followup_boundary",
                    shape=shape,
                    n0=n,
                    n1=n,
                    reps=start,
                    reps_max=cap,
                    notes="C=1 validity-boundary probe (spec follow-up item 1)",
                )
            )
    return cells


def heldout_cells() -> list[Cell]:
    """Item 2: held-out validation of the shipped C = 1 default (+ M3).

    All six held-out shapes at n = 500; a mechanism-diverse sentinel subset
    (heavy tail / mixture-inflection / LHS Weibull) at n = 5,000; the ties
    regression cell at n = 1,000. The n = 1,000 shape rows of the original
    plan were dropped on review (little information between the routing
    boundary and the n = 5,000 sentinels).
    """
    from design import HELDOUT_SHAPES, _resolve_heldout

    start, _, cap = HELDOUT_REPS
    resolved = [_resolve_heldout(s) for s in HELDOUT_SHAPES]
    sentinel = {"t3_90", "bimodal_80_sep15", _resolve_heldout("lhs2")}
    cells = []
    for shape in resolved:
        ns = (500, 5_000) if shape in sentinel else (500,)
        for n in ns:
            cells.append(
                _cell(
                    name=f"heldout--{shape}--n{n}",
                    stage="B",
                    arm="followup_heldout",
                    shape=shape,
                    n0=n,
                    n1=n,
                    reps=start,
                    reps_max=cap,
                    notes="designer-bias guard for the C=1 default "
                    "(spec follow-up item 2)",
                )
            )
    cells.append(
        _cell(
            name="heldout--binormal_90_q20--n1000",
            stage="B",
            arm="followup_ties",
            shape="binormal_90",
            n0=1_000,
            n1=1_000,
            reps=start,
            reps_max=cap,
            quantize=20,
            notes="ties regression check (Q=20, random tie-break)",
        )
    )
    return cells


def imbalance_cells() -> list[Cell]:
    """Item 4 (DEFERRED): imbalance with min(n0, n1) > 500."""
    start, _, cap = PROBE_REPS
    cells = []
    for shape in ("binormal_90", "t2_95"):
        for n0, n1 in ((5_000, 1_500), (1_500, 5_000)):
            cells.append(
                _cell(
                    name=f"imbalance--{shape}--n{n0}x{n1}",
                    stage="S",
                    arm="followup_imbalance",
                    shape=shape,
                    n0=n0,
                    n1=n1,
                    reps=start,
                    reps_max=cap,
                    notes="min(n0,n1)=1500 imbalance gap (deferred; spec "
                    "follow-up item 4)",
                )
            )
    return cells


def _cell(**kwargs) -> Cell:
    kwargs.setdefault("alphas", ALPHAS_SCREEN)
    cell = Cell(**kwargs)
    if cell.m_draws == 0:
        cell = Cell(**{**kwargs, "m_draws": m_budget(cell.n0, cell.alpha_min)})
    return cell


def _with_reps(cell: Cell, reps: int) -> Cell:
    """The same cell pinned to exactly ``reps`` replicates.

    ``reps_max = reps`` disables the runner's built-in SE(C*) gate — that
    gate targets the obsolete C* estimand; these items replicate on the
    coverage classification instead (:func:`run_classified`).
    """
    return Cell(**{**cell.__dict__, "reps": reps, "reps_max": reps})


# ---------------------------------------------------------------------------
# rank-space data (shared by the M3 arm and the composite runs)
# ---------------------------------------------------------------------------


def sample_scores(cell: Cell, rep: int) -> tuple[np.ndarray, np.ndarray, int]:
    """One replicate's (labels, scores) plus a kernel seed.

    Identical draw order to ``runner.sample_labels`` — same (cell, rep) →
    same rank data — but returns scores rather than a merged label sequence,
    for the band entry points that take (y_true, y_score). Sign flip: in
    rank space positives concentrate at *low* values, and the band APIs
    expect higher score = positive.
    """
    rng = np.random.default_rng(rep_seed_sequence(cell, rep))
    curve = get_curve(cell.shape)
    u = rng.random(cell.n0)
    w = curve.inv(rng.random(cell.n1))
    if cell.quantize is not None:
        u, w = quantize_jitter(u, w, cell.quantize, rng)
    seed = int(rng.integers(0, 2**64, dtype=np.uint64))
    y_true = np.concatenate([np.zeros(cell.n0), np.ones(cell.n1)])
    y_score = -np.concatenate([u, w])
    return y_true, y_score, seed


# ---------------------------------------------------------------------------
# runner cells with coverage-driven sequential replication (items 1, 2, 4)
# ---------------------------------------------------------------------------


def _c1_at_05(summary: dict) -> dict:
    for r in summary["aggregate"]["ref_maps"]:
        if r["label"] == "c1" and r["alpha"] == 0.05:
            return r
    raise KeyError("no c1@.05 arm in summary")


def run_classified(
    cell: Cell, out_dir: Path, *, batch: int, cap: int, **runner_kwargs
) -> dict:
    """Run a cell, topping up while the C=1@.05 Wilson CI straddles the bar.

    Replaces the runner's SE(C*) gate (which targets the retired auto-map
    estimand) with the declared coverage classification rule. Resumes and
    extends through the runner's own persistence.
    """
    reps = cell.reps
    while True:
        summary = run_cell(_with_reps(cell, reps), out_dir, **runner_kwargs)
        r = _c1_at_05(summary)
        done = summary["aggregate"]["reps"]
        if not needs_topup(r["coverage"], done) or done >= cap:
            return summary
        reps = min(done + batch, cap)
        print(
            f"  [{cell.name}] coverage gate: CI straddles {BAR_POINT}; "
            f"topping up to {reps} reps",
            flush=True,
        )


# ---------------------------------------------------------------------------
# M3 arm (paired to the fiducial arm's final rep count)
# ---------------------------------------------------------------------------


def run_m3_arm(cell: Cell, out_dir: Path, reps: int, verbose: bool = True) -> dict:
    """M3 coverage/area at the cell over exactly ``reps`` replicates —
    the fiducial arm's final (post-top-up) count, so the width ratio is
    fully paired. Same seeds/data and evaluation conventions (pointwise
    truth check on the native grid, area = mean band width over grid
    points). Reused only if the stored file matches ``reps``."""
    out_path = out_dir / f"{cell.name}.m3.summary.json"
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        if existing.get("reps") == reps:
            return existing
        if verbose:
            print(
                f"[{cell.name}] M3 arm: stored reps={existing.get('reps')} != "
                f"{reps}; recomputing paired",
                flush=True,
            )
    curve = truth_curve(cell)
    rtrue = np.clip(curve.eval(np.arange(cell.n_grid) / cell.n0), 0.0, 1.0)
    t0 = time.time()
    per_alpha = {}
    for alpha in M3_ALPHAS:
        cov = 0
        vlow = 0
        areas = np.empty(reps)
        for rep in range(reps):
            y_true, y_score, _ = sample_scores(cell, rep)
            fpr, lo, hi = m3_band_rs(y_true, y_score, alpha=alpha, random_state=rep)
            ok_low = bool(np.all(lo <= rtrue + COV_TOL))
            ok_high = bool(np.all(rtrue <= hi + COV_TOL))
            cov += ok_low and ok_high
            vlow += not ok_low
            areas[rep] = float(np.mean(hi - lo))
        covm = cov / reps
        per_alpha[f"{alpha:g}"] = {
            "coverage": covm,
            "coverage_se": float(np.sqrt(covm * (1 - covm) / reps)),
            "viol_low": vlow / reps,
            "area": float(areas.mean()),
        }
    out = {
        "cell": cell.name,
        "arm": "m3",
        "reps": reps,
        "per_alpha": per_alpha,
        "runtime_s": round(time.time() - t0, 1),
        "provenance": provenance(),
    }
    out_path.write_text(json.dumps(out, indent=1))
    if verbose:
        a05 = per_alpha["0.05"]
        print(
            f"[{cell.name}] M3 arm: cov(.05) {a05['coverage']:.3f}, "
            f"area {a05['area']:.4f}, {out['runtime_s']:.0f}s ({reps} reps)",
            flush=True,
        )
    return out


# ---------------------------------------------------------------------------
# composite band derisk (item 3)
# ---------------------------------------------------------------------------


def composite_cells() -> list[Cell]:
    grid = [
        # (shape, n0, n1, sentinel)
        ("t2_95", 100, 100, False),  # the broken cell — can the composite fix it?
        ("t2_95", 500, 500, False),
        ("t2_95", 5_000, 5_000, False),
        ("t2_95", 500, 4_500, False),  # imbalance boundary trap
        ("binormal_95", 500, 500, False),
        ("binormal_95", 5_000, 5_000, False),
        ("kink_80", 500, 500, False),
        ("trapezoid_q10_90", 500, 500, False),
        ("binormal_60", 500, 500, False),  # widest oracle gain at n=500
        # Large-n sentinels (reduced configs, fixed reps): measure the
        # Theorem-7 interior erosion direction beyond the candidate range.
        ("t2_95", 20_000, 20_000, True),
        ("binormal_95", 20_000, 20_000, True),
    ]
    start, _, cap = COMPOSITE_REPS
    return [
        _cell(
            name=f"composite--{shape}--n{n0}x{n1}",
            stage="S",
            arm="followup_composite_sentinel" if sentinel else "followup_composite",
            shape=shape,
            n0=n0,
            n1=n1,
            alphas=(0.05,),
            reps=SENTINEL_REPS if sentinel else start,
            reps_max=SENTINEL_REPS if sentinel else cap,
            m_draws=m_budget(n0, 0.05),
            notes="composite-band derisk (spec follow-up item 3)",
        )
        for shape, n0, n1, sentinel in grid
    ]


def _is_sentinel(cell: Cell) -> bool:
    return cell.arm == "followup_composite_sentinel"


def composite_configs(cell: Cell) -> list[tuple[str, tuple | None, float]]:
    """(label, corner cut, interior C) configs for a cell; 'full' = the
    plain full-curve C = 1 band (parity + width baseline)."""
    if _is_sentinel(cell):
        cuts, cs = SENTINEL_BOUNDARIES, SENTINEL_INTERIOR_C
    else:
        cuts, cs = COMPOSITE_BOUNDARIES, COMPOSITE_INTERIOR_C
    return [("full", None, 1.0)] + [
        (f"b{b_lo:g}-{b_hi:g}_C{c:g}", (b_lo, b_hi), c)
        for b_lo, b_hi in cuts
        for c in cs
    ]


def _stitch(
    lo_wide: np.ndarray,
    hi_wide: np.ndarray,
    lo_int: np.ndarray,
    hi_int: np.ndarray,
    grid: np.ndarray,
    b_lo: float,
    b_hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose the corner-wide / interior-trimmed band with monotone closure.

    Both edges get a running max. Upper: the production monotonization
    (widening only). Lower: a valid tightening — max_{s<=t} L(s) <= R(t)
    whenever L <= R pointwise, and the per-rep coverage event is unchanged
    (a raw violation at s stays a violation at s after the max) — which
    also repairs the seam at ``b_hi`` where switching back to the wide
    lower edge would otherwise produce a decreasing ROC boundary.
    """
    corner = (grid <= b_lo) | (grid >= b_hi)
    lo = np.maximum.accumulate(np.where(corner, lo_wide, lo_int))
    hi = np.maximum.accumulate(np.where(corner, hi_wide, hi_int))
    assert lo[0] == 0.0 and hi[-1] >= 1.0 - 1e-12, "endpoint invariant broken"
    assert np.all(np.diff(lo) >= 0) and np.all(np.diff(hi) >= 0), "non-monotone edge"
    assert np.all(lo <= hi + 1e-12), "lower edge crosses upper edge"
    return lo, hi


def _composite_paths(out_dir: Path, cell: Cell) -> Path:
    return out_dir / f"{cell.name}.composite.json"


def _composite_constants(cell: Cell) -> dict:
    # JSON-native values only: this dict is compared against a round-trip
    # through json on reload (tuples would never match).
    return {
        "m_draws": cell.m_draws,
        "corner_exponent": CORNER_EXPONENT,
        "configs": [
            {
                "config": label,
                "boundary": list(bounds) if bounds is not None else None,
                "interior_c": c,
            }
            for label, bounds, c in composite_configs(cell)
        ],
    }


def _load_composite(out_dir: Path, cell: Cell) -> dict | None:
    """Load prior composite records for a cell, validating the design
    constants; a mismatch refuses to mix rather than silently reusing."""
    path = _composite_paths(out_dir, cell)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload.get("constants") != _composite_constants(cell):
        raise RuntimeError(
            f"{cell.name}: existing composite output was produced under "
            "different design constants; move the old file aside."
        )
    return payload


def run_composite_cell(
    cell: Cell,
    out_dir: Path,
    *,
    n_threads: int,
    batch: int = COMPOSITE_REPS[1],
    cap: int = COMPOSITE_REPS[2],
    verbose: bool = True,
) -> dict:
    """Build the actual composite bands per rep and measure exact coverage.

    Per rep, one fiducial cloud (same kernel seed) is trimmed at the corner
    exponent (untrimmed envelope + allowances — an empirical corner
    widening, not an exact bound), at C = 1 (full-curve reference), and at
    each interior C; each (cut, C) config stitches corner-wide with
    interior-trimmed and is scored pointwise against the truth. Alpha = .05.
    Per-rep covered/area records are retained (common-random-number pairing
    for width comparisons); cells resume, and non-sentinel cells top up
    while any width-saving config's coverage CI straddles the bar.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = _composite_paths(out_dir, cell)
    configs = composite_configs(cell)
    labels = [label for label, _, _ in configs]
    payload = _load_composite(out_dir, cell)
    covered = {lb: [] for lb in labels}
    areas = {lb: [] for lb in labels}
    if payload is not None:
        for lb in labels:
            covered[lb] = list(payload["records"]["covered"][lb])
            areas[lb] = list(payload["records"]["area"][lb])

    curve = truth_curve(cell)
    grid = np.arange(cell.n_grid) / cell.n0
    rtrue = np.clip(curve.eval(grid), 0.0, 1.0)
    exponents = sorted({c for _, _, c in configs} | {CORNER_EXPONENT})

    def extend_to(target: int) -> None:
        done = len(covered[labels[0]])
        if done >= target:
            return
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # corner arm trips the j<3 warning
            for rep in range(done, target):
                y_true, y_score, seed = sample_scores(cell, rep)
                bands = {}
                for c_exp in exponents:
                    _, lo, hi = fiducial_band_rs(
                        y_true,
                        y_score,
                        alpha=0.05,
                        n_draws=cell.m_draws,
                        trim_exponent=c_exp,
                        n_threads=n_threads,
                        random_state=seed,
                    )
                    bands[c_exp] = (lo, hi)
                lo_wide, hi_wide = bands[CORNER_EXPONENT]
                for label, bounds, c_exp in configs:
                    if bounds is None:
                        lo, hi = bands[c_exp]
                    else:
                        lo, hi = _stitch(lo_wide, hi_wide, *bands[c_exp], grid, *bounds)
                    ok = bool(
                        np.all(lo <= rtrue + COV_TOL) and np.all(rtrue <= hi + COV_TOL)
                    )
                    covered[label].append(int(ok))
                    areas[label].append(round(float(np.mean(hi - lo)), 7))
                if verbose and (rep + 1 - done) % 100 == 0:
                    rate = (rep + 1 - done) / (time.time() - t0)
                    print(
                        f"  [{cell.name}] rep {rep + 1}/{target} ({rate:.2f} reps/s)",
                        flush=True,
                    )

    t_start = time.time()
    extend_to(cell.reps)
    # Sequential rule: top up while any width-saving config straddles the bar.
    while not _is_sentinel(cell):
        done = len(covered["full"])
        base = float(np.mean(areas["full"]))
        straddling = [
            lb
            for lb in labels
            if lb != "full"
            and float(np.mean(areas[lb])) < base
            and needs_topup(float(np.mean(covered[lb])), done)
        ]
        if not straddling or done >= cap:
            break
        target = min(done + batch, cap)
        print(
            f"  [{cell.name}] coverage gate ({', '.join(straddling)}): "
            f"topping up to {target} reps",
            flush=True,
        )
        extend_to(target)

    reps = len(covered["full"])
    base_areas = np.asarray(areas["full"])
    results = []
    for label, bounds, c_exp in configs:
        cov_arr = np.asarray(covered[label], dtype=float)
        area_arr = np.asarray(areas[label])
        diff = area_arr - base_areas  # paired, common random numbers
        covm = float(cov_arr.mean())
        results.append(
            {
                "config": label,
                "boundary": bounds,
                "interior_c": c_exp,
                "coverage": covm,
                "coverage_wilson95": wilson_ci(covm, reps),
                "verdict_coverage": classify(covm, reps),
                "area": float(area_arr.mean()),
                "area_diff_vs_full_c1": float(diff.mean()),
                "area_diff_se_paired": float(diff.std(ddof=1) / np.sqrt(reps))
                if reps > 1
                else None,
                "area_vs_full_c1": float(area_arr.mean() / base_areas.mean() - 1.0),
            }
        )
    out = {
        "cell": cell.name,
        "shape": cell.shape,
        "n0": cell.n0,
        "n1": cell.n1,
        "sentinel": _is_sentinel(cell),
        "reps": reps,
        "constants": _composite_constants(cell),
        "results": results,
        "records": {"covered": covered, "area": areas},
        "runtime_s": round(time.time() - t_start, 1)
        + ((payload or {}).get("runtime_s", 0.0)),
        "provenance": provenance(),
    }
    path.write_text(json.dumps(out, indent=1))
    if verbose:
        print(
            f"[{cell.name}] composite done: {reps} reps, {out['runtime_s']:.0f}s",
            flush=True,
        )
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _load_summaries(sub_dir: Path) -> list[dict]:
    return [
        json.loads(p.read_text())
        for p in sorted(sub_dir.glob("*.summary.json"))
        if not p.name.endswith(".m3.summary.json")
    ]


def _cell_row(s: dict, sub_dir: Path) -> dict:
    r = _c1_at_05(s)
    reps = s["aggregate"]["reps"]
    lo, hi = wilson_ci(r["coverage"], reps)
    name = s["meta"]["cell"]["name"]
    row = {
        "name": name,
        "shape": s["meta"]["cell"]["shape"],
        "n0": s["meta"]["cell"]["n0"],
        "n1": s["meta"]["cell"]["n1"],
        "reps": reps,
        "cov": r["coverage"],
        "wilson_lo": lo,
        "wilson_hi": hi,
        "verdict": classify(r["coverage"], reps),
        "strict": r["coverage"] >= BAR_STRICT,
        "area": r["area"],
        "cov50": next(
            m["coverage"]
            for m in s["aggregate"]["ref_maps"]
            if m["label"] == "c1" and m["alpha"] == 0.5
        ),
    }
    m3_path = sub_dir / f"{name}.m3.summary.json"
    if m3_path.exists():
        m3 = json.loads(m3_path.read_text())
        row["m3_cov"] = m3["per_alpha"]["0.05"]["coverage"]
        row["m3_ratio"] = m3["per_alpha"]["0.05"]["area"] / r["area"]
        row["m3_paired"] = m3["reps"] == reps
    return row


def _runner_table(rows: list[dict]) -> list[str]:
    lines = [
        "| cell | reps | C=1 cov @ .05 [Wilson 95%] | verdict | >= .95 | "
        "M3 cov | M3/C1 area |",
        "|---|---|---|---|---|---|---|",
    ]
    for w in rows:
        m3c = f"{w['m3_cov']:.3f}" if "m3_cov" in w else "—"
        m3r = (
            f"{w['m3_ratio']:.2f}x" + ("" if w.get("m3_paired") else " (unpaired)")
            if "m3_ratio" in w
            else "—"
        )
        lines.append(
            f"| {w['name']} | {w['reps']} | "
            f"{w['cov']:.3f} [{w['wilson_lo']:.3f}, {w['wilson_hi']:.3f}] | "
            f"{w['verdict']} | {'yes' if w['strict'] else 'no'} | {m3c} | {m3r} |"
        )
    return lines


def _boundary_threshold(rows: list[dict]) -> list[str]:
    """Per-shape smallest passing n, and the conservative global routing
    threshold = the worst (largest) such n over the tested library."""
    lines = [
        "",
        "**Routing threshold (library-relative).** Per shape, the"
        " smallest tested n whose cell and all larger tested n PASS:",
        "",
    ]
    thresholds = {}
    by_shape: dict[str, list[dict]] = {}
    for w in rows:
        by_shape.setdefault(w["shape"], []).append(w)
    for shape, cells in sorted(by_shape.items()):
        cells = sorted(cells, key=lambda w: w["n0"])
        passing_from = None
        for w in reversed(cells):
            if w["verdict"] == "PASS":
                passing_from = w["n0"]
            else:
                break
        if passing_from is None:
            lines.append(
                f"- {shape}: **no tested n passes** (largest tested "
                f"{cells[-1]['n0']}) — boundary above the tested range."
            )
            thresholds[shape] = None
        else:
            lines.append(f"- {shape}: passes from n = {passing_from} up.")
            thresholds[shape] = passing_from
    known = [t for t in thresholds.values() if t is not None]
    if any(t is None for t in thresholds.values()):
        lines.append(
            "\n**Global threshold: UNRESOLVED** — at least one shape never "
            "passes in the tested range; extend the grid before freezing "
            "routing guidance."
        )
    elif known:
        lines.append(
            f"\n**Global routing threshold (worst tested shape): "
            f"min(n0, n1) >= {max(known)}.** This is library-relative — "
            "valid over the tested shapes, not distribution-free; shapes "
            "outside the library can move it."
        )
    lines.append("")
    return lines


def build_report(out_root: Path) -> str:
    """Aggregate whatever has finished into a markdown report."""
    lines = [
        "# C-calibration follow-up runs — report",
        "",
        f"*Generated {time.strftime('%Y-%m-%d %H:%M')} from `{out_root}`. "
        "Design: the dated follow-up entry in `stats/c_calibration_spec.md` "
        "(revised 2026-08-31). Cell verdicts use the A1-letter bar: PASS = "
        f"point >= {BAR_POINT} AND Wilson-95 lower >= {BAR_CI_LO} at "
        "alpha = .05 (C = 1 arm); MARGINAL = CI still straddles the bar at "
        "the replication cap. All claims are library-relative.*",
        "",
    ]

    for item, sub in (
        ("1. Validity boundary", "boundary"),
        ("4. Imbalance min(n0,n1) > 500 (deferred item)", "imbalance"),
    ):
        sub_dir = out_root / sub
        if not sub_dir.exists():
            continue
        rows = [_cell_row(s, sub_dir) for s in _load_summaries(sub_dir)]
        lines.extend([f"## {item}", ""])
        lines.extend(_runner_table(rows))
        if sub == "boundary":
            lines.extend(_boundary_threshold(rows))
        lines.append("")

    sub_dir = out_root / "heldout"
    if sub_dir.exists():
        rows = [_cell_row(s, sub_dir) for s in _load_summaries(sub_dir)]
        lines.extend(["## 2. Held-out validation of C = 1", ""])
        lines.extend(_runner_table(rows))
        worst = min((w["cov"] for w in rows), default=1.0)
        fails = [w["name"] for w in rows if w["verdict"] == "FAIL"]
        marginal = [w["name"] for w in rows if w["verdict"] == "MARGINAL"]
        msg = f"Worst held-out C=1 coverage at alpha=.05: {worst:.3f}."
        if fails:
            msg += (
                f" FAILING cells: {', '.join(fails)} — a library gap in the "
                "theory-doc section 7.2(c) claim; escalate."
            )
        if marginal:
            msg += (
                f" MARGINAL (undecided at the replication cap): "
                f"{', '.join(marginal)}."
            )
        if not fails and not marginal:
            msg += " All cells PASS."
        lines.extend(["", msg, ""])

    sub_dir = out_root / "composite"
    if sub_dir.exists():
        payloads = [
            json.loads(p.read_text()) for p in sorted(sub_dir.glob("*.composite.json"))
        ]
        core = [p for p in payloads if not p.get("sentinel")]
        sentinels = [p for p in payloads if p.get("sentinel")]
        if core:
            lines.extend(["## 3. Composite-band derisk (finite-range)", ""])
            # Parity gate: the 'full' arm must reproduce Stage S.
            lines.append("**Parity (full-curve C=1 arm vs Stage S):**")
            parity_fail = False
            for p in core:
                ref = STAGE_S_C1_COV.get(p["cell"])
                if ref is None:
                    continue
                full = next(r for r in p["results"] if r["config"] == "full")
                ref_cov, ref_n = ref
                se = np.sqrt(
                    full["coverage"] * (1 - full["coverage"]) / max(p["reps"], 1)
                    + ref_cov * (1 - ref_cov) / ref_n
                )
                ok = abs(full["coverage"] - ref_cov) <= 3 * se + 1e-12
                parity_fail |= not ok
                lines.append(
                    f"- {p['cell']}: {full['coverage']:.3f} vs Stage S "
                    f"{ref_cov:.3f} — {'ok' if ok else 'MISMATCH'}"
                )
            if parity_fail:
                lines.append(
                    "\n**PARITY FAILURE — the composite item is void until "
                    "the discrepancy is explained.**"
                )
            lines.append("")
            configs = [r["config"] for r in core[0]["results"] if r["config"] != "full"]
            lines.append(
                "Candidate rule: every cell PASSes the coverage bar AND the "
                "pooled paired width change vs the full-curve C=1 band is "
                "negative. 'No survivor' is evidence against this coarse "
                "(cut x C) family only."
            )
            lines.append("")
            lines.append(
                "| config | worst cell verdict | min cov | pooled dArea "
                "(paired SE) | candidate |"
            )
            lines.append("|---|---|---|---|---|")
            for cfg in configs:
                rows = [x for p in core for x in p["results"] if x["config"] == cfg]
                verdicts = [x["verdict_coverage"] for x in rows]
                worst_v = (
                    "FAIL"
                    if "FAIL" in verdicts
                    else ("MARGINAL" if "MARGINAL" in verdicts else "PASS")
                )
                mincov = min(x["coverage"] for x in rows)
                dmean = float(np.mean([x["area_vs_full_c1"] for x in rows]))
                ses = [
                    x["area_diff_se_paired"]
                    / (x["area"] - x["area_diff_vs_full_c1"])  # the base area
                    for x in rows
                    if x["area_diff_se_paired"] is not None
                    and (x["area"] - x["area_diff_vs_full_c1"]) > 0
                ]
                dse = float(np.sqrt(np.sum(np.square(ses))) / len(rows)) if ses else 0.0
                cand = worst_v == "PASS" and dmean < 0
                lines.append(
                    f"| {cfg} | {worst_v} | {mincov:.3f} | "
                    f"{dmean * 100:+.1f}% ({dse * 100:.2f}pp) | "
                    f"{'YES' if cand else 'no'} |"
                )
            lines.append("")
        if sentinels:
            lines.append(
                "**Large-n sentinels (n = 20,000; Theorem-7 erosion "
                "direction — outside the candidate range):**"
            )
            for p in sentinels:
                for x in p["results"]:
                    if x["config"] == "full":
                        continue
                    lines.append(
                        f"- {p['cell']} {x['config']}: cov {x['coverage']:.3f} "
                        f"[{x['coverage_wilson95'][0]:.3f}, "
                        f"{x['coverage_wilson95'][1]:.3f}], "
                        f"dArea {x['area_vs_full_c1'] * 100:+.1f}%"
                    )
            lines.append("")
        lines.append(
            "Any surviving candidate is a *finite-range* result: the "
            "production form must clamp the interior exponent to 1 (or "
            "taper it) above the calibrated range — Theorem 7 forces "
            "interior coverage toward (1-alpha)^C for fixed C > 1. "
            "Per-cell / per-rep detail is in the `*.composite.json` files."
        )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _scale_cells(cells: list[Cell], scale: float) -> list[Cell]:
    if scale == 1.0:
        return cells
    return [
        Cell(
            **{
                **c.__dict__,
                "reps": max(2, int(c.reps * scale)),
                "reps_max": max(2, int(c.reps_max * scale)),
            }
        )
        for c in cells
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "items",
        nargs="+",
        choices=["boundary", "heldout", "composite", "imbalance", "report", "all"],
        help="'all' = boundary, heldout, composite, report "
        "(imbalance is deferred and must be requested explicitly)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-call", type=int, default=4)
    parser.add_argument("--mem-gb", type=float, default=40.0)
    parser.add_argument("--reps-scale", type=float, default=1.0)
    parser.add_argument("--select", type=str, default="", help="cell-name substring")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    register_followup_shapes()
    items = (
        ["boundary", "heldout", "composite", "report"]
        if "all" in args.items
        else args.items
    )

    runner_items = {
        "boundary": (boundary_cells, PROBE_REPS, True),
        "heldout": (heldout_cells, HELDOUT_REPS, True),
        "imbalance": (imbalance_cells, PROBE_REPS, False),
    }
    for item in items:
        if item in runner_items:
            builder, (_, batch, cap), m3_arm = runner_items[item]
            cells = _scale_cells(builder(), args.reps_scale)
            if args.select:
                cells = [c for c in cells if args.select in c.name]
            if args.dry_run:
                s = summarize(cells)
                print(
                    f"[{item}] {s.n_cells} cells, {s.total_reps} baseline reps "
                    f"(top-up cap x{cap / max(cells[0].reps, 1):.0f}), "
                    f"~{s.total_hours:.1f} idealized core-saturated hours, "
                    f"max cloud {s.max_cloud_gb:.2f} GB"
                )
                for row in s.rows:
                    print(
                        f"  {row['name']}: M={row['M']}, reps={row['reps']}, "
                        f"~{row['est_hours']:.2f}h"
                    )
                continue
            out_dir = args.out / item
            for cell in cells:
                summary = run_classified(
                    cell,
                    out_dir,
                    batch=max(2, int(batch * args.reps_scale)),
                    cap=max(2, int(cap * args.reps_scale)),
                    workers=args.workers,
                    threads_per_call=args.threads_per_call,
                    mem_gb=args.mem_gb,
                )
                # M3 width economics: boundary (routing cells) and the
                # n = 500 held-out rows + ties cell; paired rep counts.
                if m3_arm and (item == "boundary" or cell.n0 <= 1_000):
                    run_m3_arm(cell, out_dir, reps=summary["aggregate"]["reps"])
        elif item == "composite":
            cells = _scale_cells(composite_cells(), args.reps_scale)
            if args.select:
                cells = [c for c in cells if args.select in c.name]
            if args.dry_run:
                from design import rep_cost_seconds

                hours = sum(
                    len({c for _, _, c in composite_configs(cell)} | {CORNER_EXPONENT})
                    * cell.reps
                    * rep_cost_seconds(cell)
                    / 3600.0
                    for cell in cells
                )
                print(
                    f"[composite] {len(cells)} cells, ~{hours:.1f} idealized "
                    "core-saturated hours (one band build per exponent per rep)"
                )
                for c in cells:
                    print(
                        f"  {c.name}: M={c.m_draws}, reps={c.reps}"
                        + (" (sentinel)" if _is_sentinel(c) else "")
                    )
                continue
            out_dir = args.out / item
            _, batch, cap = COMPOSITE_REPS
            for cell in cells:
                run_composite_cell(
                    cell,
                    out_dir,
                    n_threads=args.threads_per_call,
                    batch=max(2, int(batch * args.reps_scale)),
                    cap=max(2, int(cap * args.reps_scale)),
                )
        elif item == "report":
            if args.dry_run:
                continue
            report = build_report(args.out)
            path = args.out / "followup_report.md"
            path.write_text(report)
            print(report)
            print(f"\n[report] written to {path}")


if __name__ == "__main__":
    main()
