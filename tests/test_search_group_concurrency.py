"""Context-length groups must run at the same time, not one after the other.

A worker process is handed one prepared dataset, so trials that disagree
about the context length cannot share one.  That constraint is real; running
the groups in series was the shortcut.  These tests pin the scheduling, not
the training: the workers are stubbed out.
"""

import pytest


# ── XGBoost ───────────────────────────────────────────────────────────────

import src.trainers.xgb_trainer as xgb_trainer


class _Inputs:
    score_key = "val_score"


class _FakeCtx:
    def Queue(self):
        return "queue"


@pytest.fixture
def spawned(monkeypatch):
    """Record every worker launch, and how the trials were collected."""
    launches = []
    order = []

    def fake_spawn(ctx, trials, inputs, stage, gpus, result_queue, use_autoregressive_eval):
        launches.append({"trials": list(trials), "gpus": list(gpus), "queue": result_queue})
        order.append("spawn")
        return [f"worker-{len(launches)}"]

    def fake_collect(processes, result_queue, *, stage_name, expected, score_key,
                     varied_params=(), progress=None):
        order.append("collect")
        return [
            {"trial": trial_id, score_key: -1.0, **params}
            for launch in launches if launch["queue"] == result_queue
            for trial_id, params in launch["trials"]
        ]

    monkeypatch.setattr(xgb_trainer, "_search_context", lambda: _FakeCtx())
    monkeypatch.setattr(xgb_trainer, "_spawn_search_workers", fake_spawn)
    monkeypatch.setattr(xgb_trainer, "_collect_worker_results", fake_collect)
    return launches, order


def _params(n_lags, count):
    return [{"n_lags": n_lags, "max_depth": i} for i in range(count)]


def _run(pool, params, stage="stage2"):
    return xgb_trainer._run_xgb_trials(
        params, {2: _Inputs(), 3: _Inputs()}, stage, 300, pool,
    )


def test_both_lag_groups_start_before_either_is_collected(spawned):
    """The whole point: 79 minutes was two maxima added, not one waited for."""
    launches, order = spawned

    _run([str(i) for i in range(8)], _params(2, 4) + _params(3, 6))

    assert order.count("collect") == 1
    assert order.index("collect") == len(order) - 1
    assert order[:-1] == ["spawn"] * (len(order) - 1)


def test_every_group_shares_one_queue_so_one_drain_serves_the_stage(spawned):
    launches, _ = spawned

    _run([str(i) for i in range(8)], _params(2, 4) + _params(3, 6))

    assert len({launch["queue"] for launch in launches}) == 1


def test_no_gpu_is_handed_to_two_groups(spawned):
    launches, _ = spawned

    _run([str(i) for i in range(8)], _params(2, 4) + _params(3, 6))

    assigned = [gpu for launch in launches for gpu in launch["gpus"]]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) <= {str(i) for i in range(8)}


def test_a_group_only_ever_gets_trials_at_its_own_lag_count(spawned):
    """The constraint the grouping exists for, still true once they overlap."""
    launches, _ = spawned

    _run([str(i) for i in range(8)], _params(2, 4) + _params(3, 6))

    for launch in launches:
        lags = {params["n_lags"] for _, params in launch["trials"]}
        assert len(lags) == 1


def test_trial_ids_number_the_stage_rather_than_restarting_per_group(spawned):
    """Concurrent groups both logging 'trial=0' cannot be told apart."""
    launches, _ = spawned

    _run([str(i) for i in range(8)], _params(2, 4) + _params(3, 6))

    ids = sorted(trial_id for launch in launches for trial_id, _ in launch["trials"])
    assert ids == list(range(10))


def test_the_stage_budget_reaches_every_trial(spawned):
    launches, _ = spawned

    _run([str(i) for i in range(8)], _params(2, 4) + _params(3, 6))

    rounds = {params["num_boost_round"] for launch in launches for _, params in launch["trials"]}
    assert rounds == {300}


def test_one_gpu_still_runs_the_groups_in_series(spawned, monkeypatch):
    """Nothing to divide: the old arrangement is the right one there."""
    launches, order = spawned
    ran = []
    monkeypatch.setattr(
        xgb_trainer, "_run_xgb_trials_sequentially",
        lambda trials, inputs, stage, gpu, expected, varied, autoreg, progress: (
            ran.append((gpu, [t for t, _ in trials]))
            or [{"trial": t, "val_score": -1.0, **p} for t, p in trials]
        ),
    )

    results = _run(["0"], _params(2, 4) + _params(3, 6))

    assert launches == [] and order == []
    assert [gpu for gpu, _ in ran] == ["0", "0"]
    assert len(results) == 10


def test_a_missing_dataset_is_refused_before_anything_is_launched(spawned):
    launches, _ = spawned

    with pytest.raises(KeyError, match="No prepared data for n_lags=5"):
        xgb_trainer._run_xgb_trials(
            _params(5, 2), {2: _Inputs()}, "stage2", 300, [str(i) for i in range(8)],
        )

    assert launches == []


def test_a_short_stage_still_reports_every_trial(spawned):
    """The result count check must span the stage, not one group."""
    launches, _ = spawned

    results = _run([str(i) for i in range(8)], _params(2, 4) + _params(3, 6))

    assert len(results) == 10


# ── TFT ───────────────────────────────────────────────────────────────────

import src.trainers.tft_trainer as tft_trainer


@pytest.fixture
def tft_spawned(monkeypatch):
    launches = []
    order = []

    def fake_spawn(ctx, trials, train_dataset, val_dataset, n_targets,
                   trainer_cfg, result_queue, run_id, stage, gpu_ids):
        launches.append({
            "trials": list(trials), "gpus": list(gpu_ids),
            "queue": result_queue, "dataset": train_dataset,
        })
        order.append("spawn")
        return [f"worker-{len(launches)}"]

    def fake_collect(processes, result_queue, run_id, stage):
        order.append("collect")
        return [
            {"status": "completed", "val_loss": 1.0, **trial["params"]}
            for launch in launches for trial in launch["trials"]
        ]

    monkeypatch.setattr(tft_trainer, "_spawn_search_workers", fake_spawn)
    monkeypatch.setattr(tft_trainer, "_collect_search_results", fake_collect)
    monkeypatch.setattr(tft_trainer, "_get_search_gpu_ids", lambda: list(range(8)))
    return launches, order


def _tft_params(encoder_length, count):
    return [
        {"encoder_length": encoder_length, "hidden_size": 64, "lstm_layers": 1,
         "dropout": 0.1, "learning_rate": 0.001 * (i + 1)}
        for i in range(count)
    ]


def _tft_datasets():
    return {2: ("train-2", "val-2"), 3: ("train-3", "val-3")}


def _tft_run(params, monkeypatch=None):
    from configs.models import TFTTrainerConfig
    return tft_trainer._run_trials_by_encoder_length(
        _tft_datasets(), 1, params, TFTTrainerConfig(), "tft_01", "stage2",
    )


def test_tft_encoder_lengths_start_together(tft_spawned):
    launches, order = tft_spawned

    _tft_run(_tft_params(2, 4) + _tft_params(3, 6))

    assert order.count("collect") == 1
    assert order.index("collect") == len(order) - 1


def test_tft_workers_keep_one_dataset_each(tft_spawned):
    """The reason the groups exist at all; overlapping them must not break it."""
    launches, _ = tft_spawned

    _tft_run(_tft_params(2, 4) + _tft_params(3, 6))

    for launch in launches:
        lengths = {trial["params"]["encoder_length"] for trial in launch["trials"]}
        assert len(lengths) == 1
        assert launch["dataset"] == f"train-{lengths.pop()}"


def test_tft_gpus_are_not_double_booked(tft_spawned):
    launches, _ = tft_spawned

    _tft_run(_tft_params(2, 4) + _tft_params(3, 6))

    assigned = [gpu for launch in launches for gpu in launch["gpus"]]
    assert sorted(assigned) == sorted(set(assigned))


def test_tft_falls_back_to_series_on_one_gpu(tft_spawned, monkeypatch):
    launches, order = tft_spawned
    monkeypatch.setattr(tft_trainer, "_get_search_gpu_ids", lambda: [0])
    calls = []
    monkeypatch.setattr(
        tft_trainer, "_run_trials_once",
        lambda train, val, n, batch, cfg, run_id, stage: calls.append(train) or [],
    )

    _tft_run(_tft_params(2, 4) + _tft_params(3, 6))

    assert calls == ["train-2", "train-3"]
    assert launches == []


def test_a_stage_where_everything_ooms_is_refused(tft_spawned, monkeypatch):
    monkeypatch.setattr(
        tft_trainer, "_collect_search_results",
        lambda processes, queue, run_id, stage: [{"status": "oom"}, {"status": "oom"}],
    )

    with pytest.raises(RuntimeError, match="no completed trials"):
        _tft_run(_tft_params(2, 4) + _tft_params(3, 6))


def test_one_encoder_length_ooming_does_not_end_the_stage(tft_spawned, monkeypatch):
    """It is a finding the ledger records, not a reason to stop the other length."""
    monkeypatch.setattr(
        tft_trainer, "_collect_search_results",
        lambda processes, queue, run_id, stage: [
            {"status": "oom", "encoder_length": 3},
            {"status": "completed", "val_loss": 0.4, "encoder_length": 2},
        ],
    )

    results = _tft_run(_tft_params(2, 4) + _tft_params(3, 6))

    assert len(results) == 2
