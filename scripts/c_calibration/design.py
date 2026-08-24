"""Pre-registered design of the trim-exponent calibration study.

Implements sections 4-5, 7 and 9 of ``stats/c_calibration_spec.md``: the
cell grids of every arm, the Monte Carlo budget rule (with its x2 safety
factor), the alpha grids, the reference maps recorded per cell, and the
deterministic per-(cell, rep) seeding. Rep counts are 2x the spec baseline
(decided at kickoff: single-shot study, mid-desktop compute).

Everything here is deterministic and cheap to construct; the runner consumes
:func:`screening_cells`, :func:`stage_a_cells`, and :func:`stage_b_cells`.
"""

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np

from studroc_paper.methods.fiducial_band import production_trim_rows

# ---------------------------------------------------------------------------
# constants (pre-registered)
# ---------------------------------------------------------------------------

STUDY_SEED = 20260822

REPS_SCREEN = 500
REPS_SCREEN_MAX = 2_000
REPS_FITTING = 2_000  # 2x spec baseline (1,000)
REPS_FITTING_MAX = 4_000  # top-up ceiling for the SE gate (spec section 4)
REPS_CONFIRM = 4_000  # 2x spec baseline (2,000 at alpha = .05)
REPS_CONFIRM_LARGE_N = 2_000  # large-n confirmation rows (cost-bounded)

ALPHAS_CORE = (0.50, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01)
ALPHAS_LARGE = (0.50, 0.20, 0.10, 0.05)
ALPHAS_SCREEN = (0.50, 0.20, 0.10, 0.05)

# Bootstrap SE gate on C* for the top-up rule (spec section 4).
CSTAR_SE_TARGET = 0.15
CSTAR_SE_ALPHA_MAX = 0.20

# Provisional auto map recorded as a reference arm (spec section 3; round-3
# fixed-shape fit gamma = 0.32, envelope delta0(.05) ~ 0.8).
PROVISIONAL_GAMMA = 0.32
PROVISIONAL_DELTA0 = 0.8
PROVISIONAL_C_MAX = 6.0

CORE_N = (25, 50, 100, 250, 500, 1000, 2500, 5000)
CORE_SHAPES = (
    "binormal_60",
    "binormal_75",
    "binormal_90",
    "binormal_95",
    "binormal_99",
    "hetero_90_r3",
    "t2_95",
    "kink_80",
    "bimodal_90",
    "trapezoid_q10_90",
)
LARGE_N = (12_500, 25_000, 50_000)
LARGE_N_SHAPES = ("binormal_95", "t2_95", "kink_80")
IMBALANCE_SHAPES = ("binormal_90", "t2_95", "bimodal_90")
IMBALANCE_N_TOTAL = (1_000, 5_000, 20_000)  # round-4 upgrade (spec section 9)
IMBALANCE_RATIOS = ((9, 1), (3, 1), (1, 3), (1, 9))
HELDOUT_SHAPES = (
    "binormal_85",
    "t3_90",
    "bimodal_80_sep15",
    "heterologit_88_r2",
    "lhs1",  # resolved to the registry names at cell construction
    "lhs2",
)
CONFIRM_N = (100, 1_000, 5_000)

SCREEN_TAPER_SHAPES = ("binormal_95", "t2_95", "kink_80")
SCREEN_TAPER_N = (100, 500, 5_000, 50_000)
SCREEN_IMBALANCE_SHAPES = ("binormal_90", "t2_95")
SCREEN_IMBALANCE_PAIRS = ((4_500, 500), (1_500, 500), (500, 1_500), (500, 4_500))


# ---------------------------------------------------------------------------
# budgets
# ---------------------------------------------------------------------------


def local_level_law(k_trim: int, alpha: float) -> float:
    """Empirical local-level law of the trimmed band (m2_report.md P4).

    ``ell(K, a) = 9.7e-4 * (a / 0.05)**1.2 * (K / 500)**-0.27``, with ``K``
    the number of trim-grid points and ``a`` the effective trim level. The
    binding case for the Monte Carlo budget is the deepest trim fitted at a
    cell, i.e. the C = 1 arm at the cell's smallest alpha.
    """
    return 9.7e-4 * (alpha / 0.05) ** 1.2 * (k_trim / 500.0) ** (-0.27)


def k_trim_of(n0: int) -> int:
    """Number of trim-grid points under the production thinning rule."""
    rows = production_trim_rows(n0 + 1)
    return n0 + 1 if rows is None else len(rows)


def m_budget(n0: int, alpha_min: float) -> int:
    """Monte Carlo cloud size per rep: ``M >= 2 * 5 / ell(K_trim, alpha_min)``.

    The budget rule of spec section 5.3 with its x2 safety factor, floored
    at the production minimum of 2,000 draws.
    """
    ell = local_level_law(k_trim_of(n0), alpha_min)
    return max(2_000, math.ceil(10.0 / ell))


def provisional_auto_exponent(n_eff: int) -> float:
    """The provisional auto trim exponent recorded as a reference arm."""
    c = 1.0 + PROVISIONAL_DELTA0 * (n_eff / 500.0) ** (-PROVISIONAL_GAMMA)
    return float(np.clip(c, 1.0, PROVISIONAL_C_MAX))


# ---------------------------------------------------------------------------
# reference maps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefArm:
    """One reference map evaluated per rep at every alpha of the cell.

    Attributes:
        label: Short arm name (``c1``, ``c2``, ``auto_prov``, ``auto``).
        alpha: Nominal level the arm targets.
        exponent: Trim exponent C; the effective level is
            ``1 - (1 - alpha) ** exponent``.
    """

    label: str
    alpha: float
    exponent: float

    @property
    def alpha_eff(self) -> float:
        return 1.0 - (1.0 - self.alpha) ** self.exponent


def reference_arms(
    alphas: tuple[float, ...], n0: int, n1: int, auto_exponent_fn=None
) -> list[RefArm]:
    """The reference arms recorded per cell: C = 1, C = 2, and the auto map.

    Args:
        alphas: The cell's nominal alpha grid.
        n0: Negative-class size.
        n1: Positive-class size.
        auto_exponent_fn: ``f(n0, n1, alpha) -> C`` for the auto arm. When
            ``None``, the provisional formula is used (Stage A); Stage B
            passes the frozen map's resolver.

    Returns:
        One :class:`RefArm` per (alpha, arm) pair.
    """
    n_eff = min(n0, n1)
    arms = []
    for alpha in alphas:
        if auto_exponent_fn is None:
            c_auto = provisional_auto_exponent(n_eff)
            auto_label = "auto_prov"
        else:
            c_auto = float(auto_exponent_fn(n0, n1, alpha))
            auto_label = "auto"
        arms += [
            RefArm(label="c1", alpha=alpha, exponent=1.0),
            RefArm(label="c2", alpha=alpha, exponent=2.0),
            RefArm(label=auto_label, alpha=alpha, exponent=c_auto),
        ]
    return arms


# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One simulation cell: a (shape, n0, n1) point of one study arm.

    Attributes:
        name: Unique, filesystem-safe cell identifier.
        stage: ``"S"`` (screening), ``"A"`` (fitting), or ``"B"``
            (confirmation).
        arm: Study arm (``core``, ``large_n``, ``imbalance``,
            ``confirm_heldout``, ``confirm_large_n``, ``confirm_imbalance``,
            ``confirm_ties``).
        shape: Shape-registry name of the true curve.
        n0: Negative-class size per rep.
        n1: Positive-class size per rep.
        alphas: Nominal alpha grid fitted/reported at this cell.
        reps: Baseline replicate count.
        reps_max: Ceiling for the SE-gate top-up rule.
        quantize: Quantization levels for the ties cell (``None`` = none);
            when set, the estimand is the trapezoid of the shape at this Q
            and samples are quantized + jittered (random tie-break).
        m_draws: Monte Carlo cloud size per rep (budget rule).
        notes: Free-text provenance.
    """

    name: str
    stage: str
    arm: str
    shape: str
    n0: int
    n1: int
    alphas: tuple[float, ...]
    reps: int
    reps_max: int
    quantize: int | None = None
    m_draws: int = 0
    notes: str = ""

    @property
    def alpha_min(self) -> float:
        return min(self.alphas)

    @property
    def n_grid(self) -> int:
        return self.n0 + 1

    def cloud_bytes(self) -> int:
        """Peak f32 cloud footprint of one rep inside the Rust kernel."""
        return 4 * self.m_draws * self.n_grid


def _mk_cell(**kwargs) -> Cell:
    cell = Cell(**kwargs)
    if cell.m_draws == 0:
        cell = Cell(**{**kwargs, "m_draws": m_budget(cell.n0, cell.alpha_min)})
    return cell


def _alphas_for_total(n_total: int) -> tuple[float, ...]:
    """Alpha grid by total sample size: the .01/.02 rows are fitted only
    where the M budget stays affordable (spec section 5.4)."""
    return ALPHAS_CORE if n_total <= 5_000 else ALPHAS_LARGE


def _resolve_heldout(shape: str) -> str:
    """Map the ``lhs1``/``lhs2`` placeholders to their registry names."""
    if shape in ("lhs1", "lhs2"):
        from shapes import lhs_heldout_specs

        specs = lhs_heldout_specs()
        return specs[0]["name"] if shape == "lhs1" else specs[1]["name"]
    return shape


def stage_a_cells() -> list[Cell]:
    """All Stage A (fitting) cells: core grid, large-n arm, imbalance arm."""
    cells: list[Cell] = []
    for shape in CORE_SHAPES:
        for n in CORE_N:
            cells.append(
                _mk_cell(
                    name=f"core--{shape}--n{n}",
                    stage="A",
                    arm="core",
                    shape=shape,
                    n0=n,
                    n1=n,
                    alphas=ALPHAS_CORE,
                    reps=REPS_FITTING,
                    reps_max=REPS_FITTING_MAX,
                )
            )
    for shape in LARGE_N_SHAPES:
        for n in LARGE_N:
            cells.append(
                _mk_cell(
                    name=f"large_n--{shape}--n{n}",
                    stage="A",
                    arm="large_n",
                    shape=shape,
                    n0=n,
                    n1=n,
                    alphas=ALPHAS_LARGE,
                    reps=REPS_FITTING,
                    reps_max=REPS_FITTING_MAX,
                    notes="D3 asymptote arm: the alpha=.05 rows are the payload",
                )
            )
    for shape in IMBALANCE_SHAPES:
        for n_total in IMBALANCE_N_TOTAL:
            for r0, r1 in IMBALANCE_RATIOS:
                n0 = n_total * r0 // (r0 + r1)
                n1 = n_total - n0
                cells.append(
                    _mk_cell(
                        name=f"imbalance--{shape}--n{n0}x{n1}",
                        stage="A",
                        arm="imbalance",
                        shape=shape,
                        n0=n0,
                        n1=n1,
                        alphas=_alphas_for_total(n_total),
                        reps=REPS_FITTING,
                        reps_max=REPS_FITTING_MAX,
                        notes=f"D2 arm, ratio {r0}:{r1} (round-4 extended sweep)",
                    )
                )
    return cells


def screening_cells() -> list[Cell]:
    """Decision-first cells run before the full Stage A campaign.

    The screen estimates the alpha=.05 taper on three mechanism-distinct
    shapes, the shape spread at n=500, and directional imbalance at fixed
    minority-class size. Its purpose is to decide whether a universal auto
    map is useful and learnable before spending on the full factorial grid.
    """
    keyed: dict[str, Cell] = {}

    def add(cell: Cell) -> None:
        """Add a screen cell, deduplicated by its stable name."""
        keyed[cell.name] = cell

    for shape in SCREEN_TAPER_SHAPES:
        for n in SCREEN_TAPER_N:
            add(
                _mk_cell(
                    name=f"screen_taper--{shape}--n{n}",
                    stage="S",
                    arm="screen_taper",
                    shape=shape,
                    n0=n,
                    n1=n,
                    alphas=ALPHAS_SCREEN,
                    reps=REPS_SCREEN,
                    reps_max=REPS_SCREEN_MAX,
                    notes="stop/go screen: alpha=.05 taper is the primary payload",
                )
            )
    for shape in CORE_SHAPES:
        if shape in SCREEN_TAPER_SHAPES:
            continue
        add(
            _mk_cell(
                name=f"screen_shape--{shape}--n500",
                stage="S",
                arm="screen_shape",
                shape=shape,
                n0=500,
                n1=500,
                alphas=ALPHAS_SCREEN,
                reps=REPS_SCREEN,
                reps_max=REPS_SCREEN_MAX,
                notes="stop/go screen: cross-shape lower envelope",
            )
        )
    for shape in SCREEN_IMBALANCE_SHAPES:
        for n0, n1 in SCREEN_IMBALANCE_PAIRS:
            add(
                _mk_cell(
                    name=f"screen_imbalance--{shape}--n{n0}x{n1}",
                    stage="S",
                    arm="screen_imbalance",
                    shape=shape,
                    n0=n0,
                    n1=n1,
                    alphas=ALPHAS_SCREEN,
                    reps=REPS_SCREEN,
                    reps_max=REPS_SCREEN_MAX,
                    notes=(
                        "stop/go screen: orientation at fixed minority-class size n=500"
                    ),
                )
            )
    return list(keyed.values())


def stage_b_cells() -> list[Cell]:
    """All Stage B (confirmation) cells, run only against a frozen map."""
    cells: list[Cell] = []
    for shape in HELDOUT_SHAPES:
        resolved = _resolve_heldout(shape)
        for n in CONFIRM_N:
            cells.append(
                _mk_cell(
                    name=f"confirm_heldout--{resolved}--n{n}",
                    stage="B",
                    arm="confirm_heldout",
                    shape=resolved,
                    n0=n,
                    n1=n,
                    alphas=ALPHAS_CORE,
                    reps=REPS_CONFIRM,
                    reps_max=REPS_CONFIRM,
                )
            )
    for shape in LARGE_N_SHAPES:
        cells.append(
            _mk_cell(
                name=f"confirm_large_n--{shape}--n25000",
                stage="B",
                arm="confirm_large_n",
                shape=shape,
                n0=25_000,
                n1=25_000,
                alphas=ALPHAS_LARGE,
                reps=REPS_CONFIRM_LARGE_N,
                reps_max=REPS_CONFIRM_LARGE_N,
            )
        )
    for shape, n_total, (r0, r1) in (
        ("binormal_85", 2_000, (1, 3)),
        ("t3_90", 10_000, (9, 1)),
    ):
        n0 = n_total * r0 // (r0 + r1)
        n1 = n_total - n0
        cells.append(
            _mk_cell(
                name=f"confirm_imbalance--{shape}--n{n0}x{n1}",
                stage="B",
                arm="confirm_imbalance",
                shape=shape,
                n0=n0,
                n1=n1,
                alphas=_alphas_for_total(n_total),
                reps=REPS_CONFIRM,
                reps_max=REPS_CONFIRM,
                notes=f"imbalance cell not used in fitting, ratio {r0}:{r1}",
            )
        )
    cells.append(
        _mk_cell(
            name="confirm_ties--binormal_90_q20--n1000",
            stage="B",
            arm="confirm_ties",
            shape="binormal_90",
            n0=1_000,
            n1=1_000,
            alphas=ALPHAS_CORE,
            reps=REPS_CONFIRM_LARGE_N,
            reps_max=REPS_CONFIRM_LARGE_N,
            quantize=20,
            notes="ties red-team regression check (Q=20, random tie-break)",
        )
    )
    return cells


# ---------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------


def _name_hash(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def rep_seed_sequence(cell: Cell, rep: int) -> np.random.SeedSequence:
    """Deterministic per-(cell, rep) seed sequence.

    Stable across runs and machines: a pure function of the study seed, the
    stage, the cell name, and the rep index. Top-up reps continue the same
    indexing, so extending a cell never re-draws earlier reps.
    """
    stage_num = {"S": 0, "A": 1, "B": 2}[cell.stage]
    return np.random.SeedSequence(
        entropy=(STUDY_SEED, stage_num, _name_hash(cell.name), rep)
    )


# ---------------------------------------------------------------------------
# cost model (dry-run estimates only)
# ---------------------------------------------------------------------------

# Calibrated to the measured ~0.56 s/band at n0 = 5000, M = 10,000 with a
# x1.6 allowance for the ladder profile's second column pass and CP table.
_COST_REF_SECONDS = 0.56 * 1.6
_COST_REF_WORK = 10_000 * 5_001 * math.log2(10_000)


def rep_cost_seconds(cell: Cell) -> float:
    """Rough wall-clock per rep with all cores on one rep."""
    work = cell.m_draws * cell.n_grid * math.log2(max(cell.m_draws, 2))
    return _COST_REF_SECONDS * work / _COST_REF_WORK


def cell_cost_hours(cell: Cell) -> float:
    """Rough per-cell wall-clock at full parallel efficiency."""
    return cell.reps * rep_cost_seconds(cell) / 3600.0


@dataclass
class DesignSummary:
    """Aggregate dry-run numbers for a cell list."""

    n_cells: int = 0
    total_reps: int = 0
    total_hours: float = 0.0
    max_cloud_gb: float = 0.0
    rows: list = field(default_factory=list)


def summarize(cells: list[Cell]) -> DesignSummary:
    """Tabulate cost and memory estimates for a cell list."""
    out = DesignSummary()
    for cell in cells:
        hours = cell_cost_hours(cell)
        out.n_cells += 1
        out.total_reps += cell.reps
        out.total_hours += hours
        out.max_cloud_gb = max(out.max_cloud_gb, cell.cloud_bytes() / 2**30)
        out.rows.append(
            {
                "name": cell.name,
                "n0": cell.n0,
                "n1": cell.n1,
                "M": cell.m_draws,
                "reps": cell.reps,
                "alphas": list(cell.alphas),
                "cloud_gb": round(cell.cloud_bytes() / 2**30, 3),
                "est_hours": round(hours, 2),
            }
        )
    return out
