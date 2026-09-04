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


# ── what each stage is scored on ──────────────────────────────────────────
#
# XGBoost is the only one of the three models with a feedback loop: its lag
# features are its own past predictions at test time.  So a trial scored on
# one-step predictions from ground-truth lags is not measuring what the test
# phase reports.  Both stages score the rollout; now that every trajectory
# rolls forward in one batch it costs seconds, so there is nothing to save
# by ranking stage 1 on a proxy.

import src.trainers.xgb_trainer as xgb_trainer
from configs.models import XGBSearchSpace


@pytest.fixture
def recorded_stages(tmp_path, monkeypatch):
    """Run the search driver with the trials themselves stubbed out."""
    space = XGBSearchSpace(n_trials=4, stage2_top_k=2)
    monkeypatch.setattr(xgb_trainer, "XGBSearchSpace", lambda **kw: space)
    monkeypatch.setattr(xgb_trainer, "get_run_root", lambda _run_id: str(tmp_path))
    monkeypatch.setattr(xgb_trainer, "_visible_gpu_pool", lambda: ["0"])

    stages = []

    def fake_run_trials(params_list, inputs_by_n_lags, stage, num_boost_round, gpu_pool,
                        use_autoregressive_eval=False):
        stages.append({
            "stage": stage,
            "rounds": num_boost_round,
            "autoregressive": use_autoregressive_eval,
            "n": len(params_list),
            "lag_counts": sorted({p["n_lags"] for p in params_list}),
        })
        return [
            {**params, "val_score": -float(i + 1), "best_iteration": 7,
             "stage": stage, "status": "completed", "trial": i}
            for i, params in enumerate(params_list)
        ]

    monkeypatch.setattr(xgb_trainer, "_run_xgb_trials", fake_run_trials)
    return space, stages


def _splits():
    import numpy as np
    import pandas as pd

    frame = pd.DataFrame({"f": [0.0]})
    return {
        "X_train": frame, "y_train": np.zeros((1, 1)), "X_train_with_index": frame,
        "train_groups": np.zeros(1), "targets": ["A"],
        "X_val": frame, "y_val": np.zeros((1, 1)), "X_val_with_index": frame,
        "obs_train": None, "obs_val": None,
    }


def _run_search():
    from configs.data import CONTEXT_LENGTHS

    return xgb_trainer.hyperparameter_search(
        {n: _splits() for n in CONTEXT_LENGTHS}, "xgb_01", use_cv=False,
    )


def test_stage_one_is_scored_on_the_rollout_at_the_reduced_budget(recorded_stages):
    space, stages = recorded_stages

    _run_search()

    stage1 = next(s for s in stages if s["stage"] == "stage1")
    assert stage1["autoregressive"] is True
    assert stage1["n"] == space.n_trials
    assert stage1["rounds"] == space.stage1_budget["num_boost_round"]


def test_stage_two_is_scored_on_the_rollout_the_test_phase_reports(recorded_stages):
    from configs.models import XGBTrainerConfig
    space, stages = recorded_stages

    _run_search()

    stage2 = next(s for s in stages if s["stage"] == "stage2")
    assert stage2["autoregressive"] is True
    assert stage2["n"] == space.stage2_top_k
    assert stage2["rounds"] == XGBTrainerConfig().num_boost_round


def test_the_rollout_can_be_switched_off_per_stage_without_touching_the_protocol(recorded_stages, monkeypatch):
    from configs.models import XGBTrainerConfig

    cfg = XGBTrainerConfig()
    cfg.search_autoregressive_stage1 = False
    monkeypatch.setattr(xgb_trainer, "XGBTrainerConfig", lambda: cfg)
    _, stages = recorded_stages

    _run_search()

    assert next(s for s in stages if s["stage"] == "stage1")["autoregressive"] is False
    assert next(s for s in stages if s["stage"] == "stage2")["autoregressive"] is True


def test_no_round_count_rides_along_in_best_params(recorded_stages):
    """The round count is the trial's, not a parameter of the final fit.

    It shares a name with a booster argument, so a number left in
    best_params would reach XGBRegressor; the final fit early-stops on the
    validation set instead, the way the LSTM and TFT finals do.
    """
    best, rows_by_stage = _run_search()

    assert "num_boost_round" not in best
    assert "best_iteration" not in best
    # Still recorded per trial, which is where the search report reads it.
    assert all("best_iteration" in row for rows in rows_by_stage.values() for row in rows)


def test_the_winning_lag_count_is_reported_with_the_parameters(recorded_stages):
    """Later phases rebuild the features from it, so it must survive."""
    from configs.data import CONTEXT_LENGTHS

    best, _ = _run_search()

    assert best["n_lags"] in CONTEXT_LENGTHS


def test_both_context_lengths_are_explored_in_stage_one(recorded_stages):
    from configs.data import CONTEXT_LENGTHS
    _, stages = recorded_stages

    _run_search()

    stage1 = next(s for s in stages if s["stage"] == "stage1")
    assert stage1["lag_counts"] == sorted(CONTEXT_LENGTHS)


def test_a_lag_count_with_no_prepared_splits_is_refused(recorded_stages):
    """Silently skipping it would shrink the search without saying so."""
    with pytest.raises(ValueError, match="no splits were prepared"):
        xgb_trainer.hyperparameter_search({2: _splits()}, "xgb_01", use_cv=False)
