"""Checks for the parallel simulation runner (scripts/run_simulation.py).

The runner's results must be a function of the seed alone: not of worker
count, scheduling order, or which process regenerated a dataset. These tests
pin that contract, the merge of the two compute lanes, the output format, and
the worker isolation the CPU lane depends on.
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("fiducial_core")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_simulation as rs  # noqa: E402

ALPHAS = [0.5, 0.05]
B = 200
TIMING = ["runtime_seconds", "bootstrap_gen_seconds"]


@pytest.fixture(scope="module")
def units() -> list[rs.ConfigUnit]:
    """Two DGPs at a balanced and an imbalanced size, three LHS points each."""
    configs = [{"n_total": 40, "n_pos": 20, "n_neg": 20, "prevalence": 0.5}]
    configs.append({"n_total": 60, "n_pos": 10, "n_neg": 50, "prevalence": 1 / 6})
    return rs.plan_units(
        dgp_types=["student_t", "bimodal_negative"],
        sample_configs=configs,
        n_lhs=3,
        seed=11,
    )


def _run(units, *, threads: int, out: Path, n_sim: int = 2) -> dict[str, pd.DataFrame]:
    out.mkdir()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        rs.run_study(
            units=units,
            n_sim=n_sim,
            alphas=ALPHAS,
            B=B,
            pool=pool,
            gpu=None,
            gpu_min_neg=0,
            max_in_flight=2,
            output_dir=out,
        )
    return {p.name: pd.read_feather(p) for p in sorted(out.glob("*.feather"))}


@pytest.fixture(scope="module")
def run_outputs(units, tmp_path_factory) -> tuple[dict, dict]:
    """The same study run serially and on four threads with a tight queue."""
    serial = _run(units, threads=1, out=tmp_path_factory.mktemp("s") / "o")
    parallel = _run(units, threads=4, out=tmp_path_factory.mktemp("p") / "o")
    return serial, parallel


def test_results_do_not_depend_on_scheduling(run_outputs):
    """Every non-timing column is identical across worker counts."""
    serial, parallel = run_outputs
    assert serial.keys() == parallel.keys() and len(serial) == 4
    for name in serial:
        pd.testing.assert_frame_equal(
            serial[name].drop(columns=TIMING),
            parallel[name].drop(columns=TIMING),
            check_exact=True,
        )


def test_output_has_one_row_per_replicate_method_and_alpha(run_outputs, units):
    serial, _ = run_outputs
    for unit in units:
        frames = [
            df
            for df in serial.values()
            if df.dgp_type.iat[0] == unit.design.dgp_type
            and df.n_neg.iat[0] == unit.sample_config["n_neg"]
        ]
        assert len(frames) == 1
        df = frames[0]
        n_reps = unit.design.n_lhs * 2
        assert len(df) == n_reps * len(rs.METHOD_NAMES) * len(ALPHAS)
        counts = df.groupby(["method", "alpha"]).size()
        assert set(counts) == {n_reps}
        assert set(df.method) == set(rs.METHOD_NAMES)
        # Rows are ordered by LHS point, then repeat.
        keys = df[["lhs_idx", "sim_idx"]].drop_duplicates()
        assert keys.equals(keys.sort_values(["lhs_idx", "sim_idx"]))


def test_cost_columns_identify_shared_calls_and_device(run_outputs):
    df = pd.concat(run_outputs[0].values())
    shared = df.groupby("method").runtime_is_shared.unique()
    assert all(shared[m].tolist() == [True] for m in ("fiducial", "fiducial_hybrid"))
    assert all(
        shared[m].tolist() == [False]
        for m in ("ks", "working_hotelling", "m3", "envelope")
    )
    # Both fiducial arms come from one call per replicate, so they share it.
    fid = df[df.method.isin(["fiducial", "fiducial_hybrid"])]
    assert (
        fid.groupby(
            ["dgp_type", "n_neg", "lhs_idx", "sim_idx", "alpha"]
        ).runtime_seconds.nunique()
        == 1
    ).all()
    assert set(df.device) == {"cpu"}
    assert df.bootstrap_gen_seconds.gt(0).all() and df.runtime_seconds.gt(0).all()


def test_replicates_differ_across_lhs_points_and_repeats(units):
    """Seeds are spawned, not reused: no two replicates share a dataset."""
    for unit in units:
        sims = [rs.simulate_lhs(t) for t in unit.tasks(n_sim=3)]
        reps = [rep for sim in sims for rep in sim.replicates]
        assert len({rep.y_score.tobytes() for rep in reps}) == len(reps)
        assert len({rep.method_seed for rep in reps}) == len(reps)
        assert len({rep.boot_seed for rep in reps}) == len(reps)


def test_regenerating_a_task_in_any_order_reproduces_its_data(units):
    """Any process can rebuild any LHS combination from its task alone."""
    tasks = units[0].tasks(n_sim=3)
    forward = [rs.simulate_lhs(t) for t in tasks]
    backward = [rs.simulate_lhs(t) for t in reversed(tasks)][::-1]
    for a, b in zip(forward, backward, strict=True):
        assert a.empirical_aucs == b.empirical_aucs
        for ra, rb in zip(a.replicates, b.replicates, strict=True):
            assert np.array_equal(ra.y_score, rb.y_score)
            assert (ra.method_seed, ra.boot_seed) == (rb.method_seed, rb.boot_seed)
    # Re-planning from the same seed rebuilds the same tasks.
    replanned = rs.plan_units(
        dgp_types=["student_t"],
        sample_configs=[units[0].sample_config],
        n_lhs=3,
        seed=11,
    )[0].tasks(n_sim=3)
    assert [rs.simulate_lhs(t).empirical_aucs for t in replanned] == [
        s.empirical_aucs for s in forward
    ]


def test_mismatched_gpu_lane_data_is_rejected(units):
    """A worker outcome built from different data than the GPU lane's fails."""
    task = units[0].tasks(n_sim=1)[0]
    sim = rs.simulate_lhs(task)
    outcome = rs.run_lhs_task(task=task, alphas=ALPHAS, B=None)
    envelopes = [
        rs.envelope_bands(rep=rep, alphas=ALPHAS, B=B, device=torch.device("cpu"))
        for rep in sim.replicates
    ]

    results = rs.UnitResults(unit=units[0], n_sim=1)
    results.add_gpu(task.lhs_idx, sim, envelopes)
    results.add_outcome(task.lhs_idx, outcome)

    other = rs.UnitResults(unit=units[0], n_sim=1)
    wrong = replace(sim, empirical_aucs=[sim.empirical_aucs[0] + 1e-9])
    other.add_gpu(task.lhs_idx, wrong, envelopes)
    with pytest.raises(RuntimeError, match="different data"):
        other.add_outcome(task.lhs_idx, outcome)


def _cpu_task_state(task: rs.LhsTask) -> tuple[bool, int, dict[str, str | None]]:
    import os

    rs.run_lhs_task(task=task, alphas=ALPHAS, B=B)
    env = {key: os.environ.get(key) for key in rs.WORKER_THREAD_ENV}
    return torch.cuda.is_initialized(), torch.get_num_threads(), env


def test_cpu_lane_workers_never_initialize_cuda_and_run_single_threaded(units):
    """A full CPU-lane task, envelope included, must leave CUDA untouched.

    Initializing the GPU driver in two dozen workers either opens contexts
    that fight over one card or, with the GPU hidden, crashes the workers;
    and native BLAS pools sized to every core commit gigabytes per worker.
    """
    task = units[0].tasks(n_sim=1)[0]
    with rs.worker_pool(workers=1, m3_cache={}) as pool:
        cuda_initialized, threads, env = pool.submit(_cpu_task_state, task).result()
    assert cuda_initialized is False
    assert threads == 1
    # Inherited at spawn, so native pools were sized before any import.
    assert env == rs.WORKER_THREAD_ENV


def test_cpu_envelope_never_queries_cuda(units, monkeypatch):
    """The envelope on an explicit CPU device must not ask whether CUDA exists."""

    def forbidden() -> bool:
        raise AssertionError("CPU-lane code queried CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    rep = rs.simulate_lhs(units[0].tasks(n_sim=1)[0]).replicates[0]
    result = rs.envelope_bands(rep=rep, alphas=ALPHAS, B=B, device=torch.device("cpu"))
    assert {r["envelope"].device for r in result.bands.values()} == {"cpu"}
