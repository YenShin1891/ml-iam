"""XGB search trials must be logged as they finish, not dumped at stage end.

A stage-end loop stamped all 24 lines of a 40-minute stage with one timestamp,
so the search looked frozen while it ran and instantaneous when it finished.
"""

import logging
import queue

import pytest

from src.trainers.xgb_trainer import _collect_worker_results, _log_trial_result


class FinishedWorker:
    """A worker process that has already queued its results and exited."""

    pid = 1
    exitcode = 0

    def is_alive(self):
        return False

    def join(self):
        pass


class CrashedWorker(FinishedWorker):
    exitcode = 1


def _drain(results, caplog, **kwargs):
    q = queue.Queue()
    for result in results:
        q.put(result)
    options = {
        "stage_name": "Stage 1",
        "expected": len(results),
        "score_key": "val_score",
        **kwargs,
    }
    with caplog.at_level(logging.INFO):
        collected = _collect_worker_results([FinishedWorker()], q, **options)
    return collected, [r.getMessage() for r in caplog.records]


# Scores are negated RMSE, so -0.5 is an RMSE of 0.5 and higher is better.
TRIALS = [
    {"val_score": -0.5, "trial": 0, "gpu": "0", "max_depth": 5, "eta": 0.4},
    {"val_score": -0.3, "trial": 1, "gpu": "1", "max_depth": 9, "eta": 0.4},
    {"val_score": -0.4, "trial": 2, "gpu": "2", "max_depth": 13, "eta": 0.4},
]


def test_each_trial_is_logged_once_as_it_arrives(caplog):
    collected, lines = _drain(TRIALS, caplog)

    assert len(collected) == 3
    assert len(lines) == 3
    assert "Stage 1 1/3" in lines[0]
    assert "Stage 1 3/3" in lines[2]


def test_the_line_reports_the_trial_and_its_rmse(caplog):
    _, lines = _drain(TRIALS, caplog)

    assert "trial=0" in lines[0]
    assert "gpu=0" in lines[0]
    assert "RMSE=0.5000" in lines[0]


def test_the_running_best_only_improves(caplog):
    _, lines = _drain(TRIALS, caplog)

    assert "best=0.5000" in lines[0]
    assert "best=0.3000" in lines[1]
    # The third trial is worse than the second, so the best must not move.
    assert "best=0.3000" in lines[2]


def test_only_the_parameters_the_stage_varies_are_shown(caplog):
    """eta is pinned for every trial in the stage; repeating it is noise."""
    _, lines = _drain(TRIALS, caplog, varied_params=["max_depth"])

    assert "max_depth=5" in lines[0]
    assert "eta" not in lines[0]


def test_a_crashed_worker_still_raises(caplog):
    q = queue.Queue()
    q.put(TRIALS[0])

    with pytest.raises(RuntimeError, match="crashed"):
        _collect_worker_results(
            [CrashedWorker()], q,
            stage_name="Stage 1", expected=1, score_key="val_score",
        )


def test_a_trial_without_gpu_or_params_still_logs(caplog):
    """The sequential path fills in the same keys, but nothing may assume it."""
    with caplog.at_level(logging.INFO):
        _log_trial_result("Stage 2", {"val_score": -0.25}, 0.25, 1, 4, 0.25)

    message = caplog.records[0].getMessage()
    assert "Stage 2 1/4" in message
    assert "trial=-" in message and "gpu=-" in message
    assert message.endswith("best=0.2500")
