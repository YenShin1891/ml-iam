"""The autoregressive rollout must convert predictions before feeding them back.

A prediction is in target units (the y scaler's); the lag column it is
written into is in feature units (the x scaler's), and the two
standardisations differ -- the x scaler saw the lagged column, the y scaler
the target, and for Solar their scales are 15% apart.  The test phase always
converted.  The search's stage-2 rollout did not: it passed the model in and
no scalers, so ten configurations were ranked on a feedback loop the test
phase never runs, and the winner lost to the defaults on the real one.
"""

import numpy as np
import pandas as pd
import pytest

from src.trainers import evaluation


class _Scaler:
    def __init__(self, mean, scale):
        self.mean_ = np.asarray(mean, dtype=float)
        self.scale_ = np.asarray(scale, dtype=float)


class _ConstantModel:
    """Predicts the same y-scaled value every step, and records what it saw."""

    def __init__(self, value):
        self.value = value
        self.inputs = []

    def predict(self, X):
        self.inputs.append(np.array(X, copy=True))
        return np.full((len(X), 1), self.value)


def _rollout(x_scaler, y_scaler, feature_columns=("prev_Primary Energy|Coal", "other")):
    model = _ConstantModel(0.5)
    matrix = np.zeros((3, len(feature_columns)))
    evaluation.autoregressive_predictions(
        model, [0, 1, 2], matrix, 0, y_scaler, x_scaler, list(feature_columns), n_lags=1,
    )
    return model


def test_the_lag_column_receives_the_prediction_in_feature_units():
    # y: mean 10, scale 4 -> a prediction of 0.5 is 12 in raw units.
    # x (lag column): mean 8, scale 2 -> 12 raw is 2.0 in feature units.
    model = _rollout(_Scaler([8.0, 0.0], [2.0, 1.0]), _Scaler([10.0], [4.0]))

    fed_back = model.inputs[1][0, 0]
    assert fed_back == pytest.approx(2.0)
    assert fed_back != pytest.approx(0.5)


def test_scalers_that_do_not_fit_the_features_are_refused_not_worked_around():
    """The old behaviour was a warning and then the wrong-scale loop anyway."""
    with pytest.raises(ValueError, match="unusable"):
        _rollout(_Scaler([8.0], [2.0]), _Scaler([10.0], [4.0]))


def test_the_test_harness_refuses_to_roll_out_without_scalers():
    X = pd.DataFrame({"Model": ["m"], "Scenario": ["s"], "Region": ["r"], "prev_Primary Energy|Coal": [0.0]})

    with pytest.raises(ValueError, match="needs both the x and y scalers"):
        evaluation.test_xgb_autoregressively(X, np.zeros((1, 1)), model=_ConstantModel(0.0))


# ── the search hands its rollout the split's scalers ──────────────────────

import src.trainers.xgb_trainer as xgb_trainer


def test_the_search_rollout_is_the_test_phase_rollout(monkeypatch):
    """Same function, same scalers: stage 2 must score what test reports."""
    from configs.models import XGBTrainerConfig

    seen = {}

    def fake_rollout(X_val_with_index, y_val, **kwargs):
        seen.update(kwargs)
        return np.zeros_like(y_val)

    monkeypatch.setattr(xgb_trainer, "test_xgb_autoregressively", fake_rollout)

    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(40, 2), columns=["a", "prev_Primary Energy|Coal"])
    y = rng.rand(40, 1)
    index = pd.DataFrame({"Model": "m", "Scenario": "s", "Region": "r"}, index=X.index)
    cfg = XGBTrainerConfig()
    cfg.device = "cpu"
    x_scaler, y_scaler = _Scaler([0, 0], [1, 1]), _Scaler([0], [1])

    rollout_rmse, _, one_step_rmse = xgb_trainer._train_single_fold(
        X, y, X, y, pd.concat([X, index], axis=1), ["Primary Energy|Coal"],
        {"max_depth": 2, "eta": 0.3, "num_boost_round": 3}, {}, 1,
        trainer_cfg=cfg, use_autoregressive_eval=True,
        x_scaler=x_scaler, y_scaler=y_scaler,
    )

    assert seen["x_scaler"] is x_scaler
    assert seen["y_scaler"] is y_scaler
    # The one-step error rides along so the proxy can be compared with the
    # objective afterwards; the score itself is the rollout's.
    assert one_step_rmse is not None and one_step_rmse != rollout_rmse


def test_a_fold_scored_one_step_reports_no_rollout_companion(monkeypatch):
    from configs.models import XGBTrainerConfig

    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(40, 2), columns=["a", "prev_Primary Energy|Coal"])
    y = rng.rand(40, 1)
    cfg = XGBTrainerConfig()
    cfg.device = "cpu"

    _, _, one_step_rmse = xgb_trainer._train_single_fold(
        X, y, X, y, X, ["Primary Energy|Coal"],
        {"max_depth": 2, "eta": 0.3, "num_boost_round": 3}, {}, 1,
        trainer_cfg=cfg, use_autoregressive_eval=False,
    )

    assert one_step_rmse is None


def test_the_trial_inputs_carry_the_scalers_from_the_splits(monkeypatch):
    """hyperparameter_search builds the inputs; the scalers must not be dropped there."""
    from configs.data import CONTEXT_LENGTHS
    from configs.models import XGBSearchSpace

    monkeypatch.setattr(xgb_trainer, "XGBSearchSpace", lambda **kw: XGBSearchSpace(n_trials=2, stage2_top_k=1))
    monkeypatch.setattr(xgb_trainer, "_visible_gpu_pool", lambda: ["0"])
    monkeypatch.setattr(xgb_trainer, "write_search_report", lambda *a, **k: None)
    monkeypatch.setattr(xgb_trainer, "get_run_root", lambda _run_id: "/nonexistent")

    captured = {}

    def fake_run(params_list, inputs_by_n_lags, stage, *a, **k):
        captured.update(inputs_by_n_lags)
        return [
            {**p, "val_score": -1.0, "best_iteration": 1, "stage": stage, "status": "completed"}
            for p in params_list
        ]

    monkeypatch.setattr(xgb_trainer, "_run_xgb_trials", fake_run)

    frame = pd.DataFrame({"f": [0.0]})
    scalers = {n: (_Scaler([0], [1]), _Scaler([0], [1])) for n in CONTEXT_LENGTHS}
    splits = {
        n: {
            "X_train": frame, "y_train": np.zeros((1, 1)), "X_train_with_index": frame,
            "train_groups": np.zeros(1), "targets": ["A"],
            "X_val": frame, "y_val": np.zeros((1, 1)), "X_val_with_index": frame,
            "obs_train": None, "obs_val": None,
            "x_scaler": scalers[n][0], "y_scaler": scalers[n][1],
        }
        for n in CONTEXT_LENGTHS
    }

    xgb_trainer.hyperparameter_search(splits, "xgb_01", use_cv=False)

    for n in CONTEXT_LENGTHS:
        assert captured[n].x_scaler is scalers[n][0]
        assert captured[n].y_scaler is scalers[n][1]
