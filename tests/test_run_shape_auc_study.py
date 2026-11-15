"""Checks for resuming the stratified shape × AUC study runner.

In scripts/run_shape_auc_study.py, a long run must survive being stopped and
restarted: the merged result of an interrupted-then-resumed run must equal
that of an uninterrupted one, no chunk may be lost or counted twice, and a
directory holding another design's results must never be resumed into. The
simulation itself is replaced by a deterministic stand-in so these tests
exercise only the orchestration.
"""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("fiducial_core")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]

from scripts import run_shape_auc_study as st  # noqa: E402

REPS = 25  # three chunks per stratum, the last one short
SEED = 7
ALPHAS = (0.05, 0.5)


def fake_records(task: st.ChunkTask) -> list[dict]:
    """Deterministic records standing in for one simulated chunk."""
    return [
        {
            "dgp_type": task.cell.dgp_type,
            "n_total": task.cell.sample_config["n_total"],
            "shape_bin": task.stratum.shape_bin,
            "auc_bin": task.stratum.auc_bin,
            "rep_idx": task.first_rep + i,
            "method": method,
            "alpha": alpha,
            "covers_entirely": hash(
                (
                    task.seed,
                    task.cell.cell_key,
                    task.stratum,
                    task.first_rep + i,
                    method,
                    alpha,
                )
            )
            % 2
            == 0,
        }
        for i in range(task.n_reps)
        for method in st.METHOD_NAMES
        for alpha in task.alphas
    ]


class ChunkRunner:
    """Stand-in for ``run_chunk`` that counts calls and can fail on the k-th."""

    def __init__(self, *, fail_on: int | None = None) -> None:
        self.fail_on = fail_on
        self.calls: list[tuple[str, tuple[int, int, int]]] = []
        self.completed: set[tuple[str, tuple[int, int, int]]] = set()
        self.lock = threading.Lock()

    def __call__(self, task: st.ChunkTask) -> list[dict]:
        chunk = (task.cell.stem, st.chunk_key(task))
        with self.lock:
            self.calls.append(chunk)
            count = len(self.calls)
        if self.fail_on is not None and count >= self.fail_on:
            raise KeyboardInterrupt
        with self.lock:
            self.completed.add(chunk)
        return fake_records(task)


@pytest.fixture
def cells() -> list[st.StudyCell]:
    """Two cells of one DGP at different sample sizes."""
    return [
        st.StudyCell(
            dgp_type="student_t",
            sample_config={
                "n_total": n,
                "n_pos": n // 2,
                "n_neg": n // 2,
                "prevalence": 0.5,
            },
            cell_key=(0, n, 5000),
        )
        for n in (40, 80)
    ]


def run(
    *,
    cells: list[st.StudyCell],
    output_dir: Path,
    runner: ChunkRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume the study as ``main`` does, on threads, with ``runner`` simulating."""
    monkeypatch.setattr(st, "run_chunk", runner)
    pending = st.pending_cells(cells=cells, output_dir=output_dir)
    tasks = st.plan_tasks(cells=pending, reps=REPS, seed=SEED, alphas=ALPHAS)
    with ThreadPoolExecutor(max_workers=2) as pool:
        st.run_study(tasks=tasks, pool=pool, output_dir=output_dir, workers=2)


def results(*, cells: list[st.StudyCell], output_dir: Path) -> dict[str, pd.DataFrame]:
    """Every cell's merged result file."""
    return {
        c.stem: pd.read_feather(output_dir / f"{c.stem}_individual.feather")
        for c in cells
    }


@pytest.mark.parametrize(
    "fail_on", [1, 9, 60], ids=["first-chunk", "first-cell", "second-cell"]
)
def test_interrupted_run_resumes_to_the_uninterrupted_result(
    *,
    cells: list[st.StudyCell],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_on: int,
) -> None:
    reference_dir, resumed_dir = tmp_path / "reference", tmp_path / "resumed"
    run(
        cells=cells,
        output_dir=reference_dir,
        runner=ChunkRunner(),
        monkeypatch=monkeypatch,
    )

    interrupted = ChunkRunner(fail_on=fail_on)
    with pytest.raises(KeyboardInterrupt):
        run(
            cells=cells,
            output_dir=resumed_dir,
            runner=interrupted,
            monkeypatch=monkeypatch,
        )
    all_chunks = {
        (stem, st.chunk_key(t))
        for stem, cell_tasks in st.plan_tasks(
            cells=cells, reps=REPS, seed=SEED, alphas=ALPHAS
        ).items()
        for t in cell_tasks
    }
    saved = {
        chunk
        for chunk in all_chunks
        if (resumed_dir / f"{chunk[0]}_individual.feather").exists()
        or chunk[1] in st.checkpointed_chunks(output_dir=resumed_dir, stem=chunk[0])
    }
    # Only chunks that really finished are on disk, and none collected is lost:
    # chunk k is submitted only after k - 8 earlier results were collected
    assert saved <= interrupted.completed
    assert len(saved) >= fail_on - 4 * 2

    resume = ChunkRunner()
    run(cells=cells, output_dir=resumed_dir, runner=resume, monkeypatch=monkeypatch)

    # The resumed run computes each missing chunk exactly once and nothing else
    assert sorted(resume.calls) == sorted(all_chunks - saved)
    for name, expected in results(cells=cells, output_dir=reference_dir).items():
        got = results(cells=cells, output_dir=resumed_dir)[name]
        pd.testing.assert_frame_equal(got, expected)
        assert len(got) == 16 * REPS * len(st.METHOD_NAMES) * len(ALPHAS)
    assert not (resumed_dir / "partial").exists()


def test_fully_checkpointed_cell_is_merged_without_recomputing(
    *, cells: list[st.StudyCell], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = st.plan_tasks(cells=cells[:1], reps=REPS, seed=SEED, alphas=ALPHAS)
    (cell_tasks,) = tasks.values()
    stem = cells[0].stem
    # Checkpoints of every chunk, as left by an exit just before the merge
    for half in (cell_tasks[::2], cell_tasks[1::2]):
        st.checkpoint(
            stem=stem,
            records=[r for t in half for r in fake_records(t)],
            output_dir=tmp_path,
        )

    runner = ChunkRunner()
    run(cells=cells[:1], output_dir=tmp_path, runner=runner, monkeypatch=monkeypatch)

    assert runner.calls == []
    merged = pd.read_feather(tmp_path / f"{stem}_individual.feather")
    expected = pd.DataFrame.from_records(
        [r for t in cell_tasks for r in fake_records(t)]
    )
    expected = expected.sort_values(
        ["shape_bin", "auc_bin", "rep_idx", "method", "alpha"], ignore_index=True
    )
    pd.testing.assert_frame_equal(merged, expected)
    assert not st.partial_dir(output_dir=tmp_path, stem=stem).exists()


@pytest.mark.parametrize(
    ("reps", "seed", "alphas", "field"),
    [
        (REPS + 1, SEED, ALPHAS, "reps"),
        (REPS, SEED + 1, ALPHAS, "seed"),
        (REPS, SEED, (0.05,), "alphas"),
    ],
    ids=["reps", "seed", "alphas"],
)
def test_output_dir_of_another_design_is_refused(
    *, tmp_path: Path, reps: int, seed: int, alphas: tuple[float, ...], field: str
) -> None:
    original = st.study_design(reps=REPS, seed=SEED, alphas=ALPHAS)
    st.claim_output_dir(output_dir=tmp_path, design=original)
    # Claiming again with the same design is a resume and must pass
    st.claim_output_dir(output_dir=tmp_path, design=original)

    with pytest.raises(ValueError, match=rf"differs in: {field}\)"):
        st.claim_output_dir(
            output_dir=tmp_path,
            design=st.study_design(reps=reps, seed=seed, alphas=alphas),
        )


def test_results_without_a_recorded_design_are_refused(*, tmp_path: Path) -> None:
    pd.DataFrame({"x": [1]}).to_feather(
        tmp_path / "student_t_n30_prev50_individual.feather"
    )

    with pytest.raises(ValueError, match="without a design.json"):
        st.claim_output_dir(
            output_dir=tmp_path,
            design=st.study_design(reps=REPS, seed=SEED, alphas=ALPHAS),
        )
    assert not (tmp_path / "design.json").exists()
