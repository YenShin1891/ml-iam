"""The parallel and sequential LSTM searches must agree on everything but scheduling."""

import pandas as pd
import pytest

pytest.importorskip("torch")

import src.trainers.lstm_trainer as lstm_trainer
from src.trainers.lstm_trainer import (
    _report_search_results,
    resolve_feature_columns,
    hyperparameter_search_lstm,
)

TARGETS = ["A", "B"]


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "Model": ["M0"] * 4, "Scenario": ["S0"] * 4, "Region": [0] * 4,
            "Step": [0, 1, 2, 3], "Year": [2020, 2025, 2030, 2035],
            "feat": [1.0, 2.0, 3.0, 4.0], "A": [1.0, 2.0, 3.0, 4.0], "B": [4.0, 3.0, 2.0, 1.0],
        }
    )


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(lstm_trainer, "get_run_root", lambda _run_id: str(tmp_path))
    return "lstm_01", tmp_path


@pytest.fixture
def three_trials(monkeypatch):
    """Shrink the search to three stage-1 trials and one stage-2 refit.

    Also pins execution to the sequential path: the search fans out over every
    visible GPU otherwise, and a test that spawns eight worker processes tests
    the scheduler rather than the search.
    """
    real_space = lstm_trainer.LSTMSearchSpace

    def _space():
        return real_space(n_trials=3, stage2_top_k=1)

    monkeypatch.setattr(lstm_trainer, "LSTMSearchSpace", _space)
    monkeypatch.setattr(lstm_trainer.torch.cuda, "device_count", lambda: 1)


# ── feature resolution ────────────────────────────────────────────────────


def test_explicit_features_are_used_as_given(frame):
    assert resolve_feature_columns(frame, TARGETS, ["feat"]) == ["feat"]


def test_derived_features_exclude_targets_and_index_columns(frame):
    features = resolve_feature_columns(frame, TARGETS, None)

    assert features == ["feat"]
    for excluded in TARGETS + ["Model", "Scenario", "Region", "Step", "Year"]:
        assert excluded not in features


# ── result reporting ──────────────────────────────────────────────────────


def _trial(trial_id, val_loss, sequence_length=1, **extra):
    return {
        "sequence_length": sequence_length, "hidden_size": 64,
        "val_loss": val_loss, "trial_id": trial_id, **extra,
    }


def test_best_params_exclude_bookkeeping_keys(run):
    run_id, _ = run

    best = _report_search_results(
        [_trial(0, 1.5), _trial(1, 0.5), _trial(2, 2.0)], run_id
    )

    assert best == {"sequence_length": 1, "hidden_size": 64}
    assert "val_loss" not in best and "trial_id" not in best


def test_failed_trials_are_written_out_but_never_win(run):
    run_id, root = run

    best = _report_search_results(
        [_trial(0, float("inf"), error="CUDA OOM"), _trial(1, 0.7, sequence_length=2)], run_id
    )

    assert best["sequence_length"] == 2
    saved = pd.read_csv(root / "search_results.csv")
    assert len(saved) == 2
    assert saved["error"].notna().sum() == 1


def test_all_trials_failing_raises(run):
    run_id, _ = run

    with pytest.raises(RuntimeError, match="All LSTM hyperparameter trials failed"):
        _report_search_results([_trial(0, float("inf")), _trial(1, float("inf"))], run_id)


def test_best_per_sequence_length_report_is_written(run):
    run_id, root = run

    _report_search_results(
        [_trial(0, 1.0, sequence_length=1), _trial(1, 0.4, sequence_length=1),
         _trial(2, 0.9, sequence_length=3)],
        run_id,
    )

    by_seq = pd.read_csv(root / "search_best_by_seq_len.csv")
    assert by_seq["sequence_length"].tolist() == [1, 3]
    assert by_seq["val_loss"].tolist() == [0.4, 0.9]


# ── a failing trial does not end the search ───────────────────────────────


def test_one_failing_trial_does_not_abort_the_search(run, frame, monkeypatch, three_trials):
    """Sequential search used to propagate the first exception."""
    run_id, _ = run
    calls = []

    def fake_datasets(*args, **kwargs):
        calls.append(kwargs.get("sequence_length"))
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(lstm_trainer, "create_lstm_datasets", fake_datasets)

    with pytest.raises(RuntimeError, match="All LSTM stage-1 hyperparameter trials failed"):
        hyperparameter_search_lstm(frame, frame, TARGETS, run_id, ["feat"])

    assert len(calls) == 3, "every trial should have been attempted"


def test_a_surviving_trial_wins_over_failures(run, frame, monkeypatch, three_trials):
    run_id, _ = run
    attempts = {"n": 0}

    def fake_run_trial(trial_id, params, *args, stage="stage1", **kwargs):
        attempts["n"] += 1
        row = {**params, "trial_id": trial_id, "stage": stage}
        if trial_id == 1 or stage == "stage2":
            return {**row, "val_loss": 0.25, "status": "completed"}
        return {**row, "val_loss": float("inf"), "status": "failed", "error": "boom"}

    monkeypatch.setattr(lstm_trainer, "_run_lstm_trial", fake_run_trial)

    best = hyperparameter_search_lstm(frame, frame, TARGETS, run_id, ["feat"])

    # Three stage-1 trials, then the single survivor refit in stage 2.
    assert attempts["n"] == 4
    assert "val_loss" not in best and "stage" not in best


# ── shared config construction ────────────────────────────────────────────


def test_default_params_cover_every_tunable():
    from src.trainers.lstm_trainer import LSTM_TUNABLE_PARAMS, default_lstm_params

    assert set(default_lstm_params()) == set(LSTM_TUNABLE_PARAMS)


def test_defaults_round_trip_to_the_config_they_came_from():
    from configs.models import LSTMTrainerConfig
    from src.trainers.lstm_trainer import (
        LSTM_TUNABLE_PARAMS, default_lstm_params, lstm_config_from_params,
    )

    built = lstm_config_from_params(default_lstm_params())
    reference = LSTMTrainerConfig()

    for name in LSTM_TUNABLE_PARAMS:
        assert getattr(built, name) == getattr(reference, name), name


def test_missing_parameters_fall_back_to_the_config_default():
    """They used to fall back to literals that contradicted it (64 vs 128)."""
    from configs.models import LSTMTrainerConfig
    from src.trainers.lstm_trainer import lstm_config_from_params

    built = lstm_config_from_params({"dropout": 0.5})

    assert built.dropout == 0.5
    assert built.hidden_size == LSTMTrainerConfig().hidden_size
    assert built.learning_rate == LSTMTrainerConfig().learning_rate


def test_whole_floats_are_coerced_only_for_integer_settings():
    """best_params comes back through a CSV, which floats the ints."""
    from src.trainers.lstm_trainer import lstm_config_from_params

    built = lstm_config_from_params(
        {"batch_size": 128.0, "hidden_size": 64.0, "dropout": 0.0, "learning_rate": 1.0}
    )

    assert isinstance(built.batch_size, int) and built.batch_size == 128
    assert isinstance(built.hidden_size, int) and built.hidden_size == 64
    assert isinstance(built.dropout, float) and built.dropout == 0.0
    assert isinstance(built.learning_rate, float) and built.learning_rate == 1.0


def test_overrides_beat_both_params_and_defaults():
    from src.trainers.lstm_trainer import lstm_config_from_params

    built = lstm_config_from_params({"batch_size": 64}, max_epochs=20, patience=3, devices=[1])

    assert (built.max_epochs, built.patience, built.devices) == (20, 3, [1])
    assert built.batch_size == 64


def test_the_search_and_the_final_fit_build_the_same_model_config():
    """Same best_params must mean the same architecture in both phases."""
    from src.trainers.lstm_trainer import LSTM_TUNABLE_PARAMS, lstm_config_from_params

    params = {"hidden_size": 96, "num_layers": 3, "dropout": 0.3,
              "batch_size": 64, "sequence_length": 4, "embedding_dim": 16}

    trial = lstm_config_from_params(params, max_epochs=20, patience=3, devices=1)
    final = lstm_config_from_params(params)

    for name in LSTM_TUNABLE_PARAMS:
        assert getattr(trial, name) == getattr(final, name), name
    # ...and they differ only in how long they train.
    assert trial.max_epochs != final.max_epochs


def test_the_per_sequence_length_table_never_mixes_stages(run):
    """Stage-2 rows trained far longer; a mixed table ranks budget, not length."""
    run_id, root = run

    _report_search_results(
        [
            _trial(0, 0.8, sequence_length=1, stage="stage1"),
            _trial(1, 0.7, sequence_length=2, stage="stage1"),
            _trial(2, 0.2, sequence_length=1, stage="stage2"),
            _trial(3, 0.3, sequence_length=2, stage="stage2"),
        ],
        run_id,
    )

    by_seq = pd.read_csv(root / "search_best_by_seq_len.csv")
    assert by_seq["val_loss"].tolist() == [0.2, 0.3]
    assert set(by_seq["stage"]) == {"stage2"}
