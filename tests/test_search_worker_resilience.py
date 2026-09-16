"""A search worker outlives a trial that fails for a reason other than OOM.

A worker that died on a CUDA fault took every trial still queued behind it,
left nothing in the ledger for any of them, and the parent could only say
"check logs" -- on a machine whose logs were not on the shared store.  Now
the failure is a row and a traceback file in the run directory, and the
worker goes on to its next trial.
"""

import os
import queue
from types import SimpleNamespace

import pytest

pytest.importorskip("lightning")

import src.trainers.tft_trainer as tft_trainer
from src.trainers.search import is_completed_trial


class _Dataset:
    def to_dataloader(self, **kwargs):
        return object()


@pytest.fixture
def worker_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    monkeypatch.setattr(tft_trainer, "get_run_root", lambda run_id: str(tmp_path))
    monkeypatch.setattr(tft_trainer, "get_default_num_workers", lambda: 0)
    monkeypatch.setattr(tft_trainer, "_provenance", lambda run_id: {"host": "h"})
    monkeypatch.setattr(tft_trainer.torch.cuda, "empty_cache", lambda: None)
    return tmp_path


def _trials(n):
    return [
        {
            "params": {"hidden_size": 64 + 8 * i, "lstm_layers": 1, "dropout": 0.1,
                       "learning_rate": 1e-3, "encoder_length": 2},
            "signature": f"sig{i}",
            "trial_id": f"stage1_trial_{i}",
        }
        for i in range(n)
    ]


def _run_worker(trials, result_queue):
    tft_trainer._search_worker(
        0, trials, _Dataset(), _Dataset(), 1, SimpleNamespace(batch_size=4), result_queue, "tft_01", "stage1",
    )
    return [result_queue.get_nowait() for _ in range(len(trials))]


def test_a_crashing_trial_is_recorded_and_the_worker_goes_on(worker_env, monkeypatch):
    calls = []

    def fit(train_dataset, params, n_targets, cfg, log_dir, train_loader, val_loader):
        calls.append(params["hidden_size"])
        if len(calls) == 1:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        return (3, 0.5, 10)

    monkeypatch.setattr(tft_trainer, "_fit_search_trial", fit)

    rows = _run_worker(_trials(2), queue.Queue())

    assert [row["status"] for row in rows] == ["crashed", "completed"]
    assert "illegal memory access" in rows[0]["error"]
    assert rows[0]["wall_seconds"] >= 0
    assert len(calls) == 2, "the second trial must still run"


def test_the_traceback_is_written_into_the_run_directory(worker_env, monkeypatch):
    def fit(*args, **kwargs):
        raise ValueError("dataloader worker (pid 4242) is killed by signal: Bus error")

    monkeypatch.setattr(tft_trainer, "_fit_search_trial", fit)

    _run_worker(_trials(1), queue.Queue())

    crash = worker_env / "search" / "trials" / "stage1_trial_0" / "crash.txt"
    assert crash.exists()
    assert "Bus error" in crash.read_text() and "Traceback" in crash.read_text()


def test_a_crashed_row_never_ranks_but_is_rerun_on_resume(worker_env, monkeypatch):
    from configs.models import TFTSearchSpace
    from src.trainers.search import plan_two_stage_search

    monkeypatch.setattr(tft_trainer, "_fit_search_trial", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    space = TFTSearchSpace(n_trials=2, stage2_top_k=1)
    params = space.sample()[0]
    trial = {"params": params, "signature": tft_trainer._params_signature(params), "trial_id": "stage1_trial_x"}

    [row] = _run_worker([trial], queue.Queue())

    assert not is_completed_trial(row, space.param_keys)
    plan = plan_two_stage_search(space, [row])
    assert params in plan.stage1_pending


def test_an_oom_is_still_told_apart_from_a_crash(worker_env, monkeypatch):
    def fit(*args, **kwargs):
        raise tft_trainer.torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(tft_trainer, "_fit_search_trial", fit)

    [row] = _run_worker(_trials(1), queue.Queue())

    assert row["status"] == "oom" and "error" not in row


def test_a_cuda_cache_that_cannot_be_emptied_does_not_end_the_worker(worker_env, monkeypatch):
    """After a sticky CUDA fault even empty_cache raises."""
    def boom():
        raise RuntimeError("CUDA error: device-side assert triggered")

    monkeypatch.setattr(tft_trainer.torch.cuda, "empty_cache", boom)
    monkeypatch.setattr(tft_trainer, "_fit_search_trial", lambda *a, **k: (3, 0.5, 10))

    rows = _run_worker(_trials(2), queue.Queue())

    assert [row["status"] for row in rows] == ["completed", "completed"]


# ── a worker that dies outright is at least named ─────────────────────────


class _Process:
    pid = 1

    def __init__(self, name, exitcode):
        self.name, self.exitcode = name, exitcode

    def is_alive(self):
        return False

    def join(self, timeout=None):
        pass


def test_a_worker_killed_by_a_signal_is_named_with_its_gpu_and_exit_code(monkeypatch):
    monkeypatch.setattr(tft_trainer, "_record", lambda *a: None)

    with pytest.raises(RuntimeError) as excinfo:
        tft_trainer._collect_search_results(
            [_Process("gpu0", -9), _Process("gpu1", 1), _Process("gpu2", 0)],
            queue.Queue(), "tft_01", "stage1",
        )

    message = str(excinfo.value)
    assert "gpu0 exit=-9" in message and "gpu1 exit=1" in message and "gpu2" not in message
    assert "OOM killer" in message and "crash.txt" in message
