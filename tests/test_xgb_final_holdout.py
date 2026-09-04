"""The final XGB fit must hold val out, the way the LSTM and TFT finals do.

The model used to be fitted on train and val concatenated, which gave
XGBoost strictly more data than the other two models and left it with no
genuinely held-out split to early-stop on.
"""

import numpy as np
import pandas as pd
import pytest

from configs.models.xgb import XGBTrainerConfig
from src.trainers.xgb_trainer import load_final_xgb_model, train_and_save_model

TARGETS = ["A", "B"]
BEST_PARAMS = {"max_depth": 3, "eta": 0.3, "n_lags": 3}


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A fake run directory that get_run_root() resolves to, fitted on CPU."""
    import src.trainers.xgb_trainer as xgb_trainer

    monkeypatch.setattr(xgb_trainer, "get_run_root", lambda _run_id: str(tmp_path))
    monkeypatch.setattr(XGBTrainerConfig, "device", "cpu")
    return "xgb_99", tmp_path


def _splits(n_train=400, n_val=100):
    rng = np.random.RandomState(0)

    def make(n):
        X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
        y = np.column_stack([
            X["f0"] * 2 + rng.normal(size=n) * 0.3,
            X["f1"] - X["f2"] + rng.normal(size=n) * 0.3,
        ])
        return X, y

    return make(n_train), make(n_val)


def test_only_the_training_rows_reach_fit(run, monkeypatch):
    """Val supplies the eval_set, never the boosting updates."""
    import src.trainers.xgb_trainer as xgb_trainer

    run_id, _ = run
    (X_train, y_train), (X_val, y_val) = _splits()
    seen = {}

    class RecordingRegressor(xgb_trainer.XGBRegressor):
        def fit(self, X, y, **kwargs):
            seen["fit_rows"] = len(X)
            seen["eval_rows"] = [len(Xv) for Xv, _ in kwargs.get("eval_set") or []]
            return super().fit(X, y, **kwargs)

    monkeypatch.setattr(xgb_trainer, "XGBRegressor", RecordingRegressor)
    monkeypatch.setattr("configs.data.KEEP_PARTIAL_TARGETS", False)

    train_and_save_model(X_train, y_train, X_val, y_val, TARGETS, BEST_PARAMS, run_id)

    assert seen["fit_rows"] == len(X_train)
    assert seen["eval_rows"] == [len(X_val)]


def test_early_stopping_picks_the_round_count(run, monkeypatch):
    """The search's round count is bookkeeping; val decides where to stop."""
    run_id, _ = run
    (X_train, y_train), (X_val, y_val) = _splits()
    monkeypatch.setattr("configs.data.KEEP_PARTIAL_TARGETS", False)

    train_and_save_model(X_train, y_train, X_val, y_val, TARGETS, BEST_PARAMS, run_id)

    # save_model carries best_iteration into the JSON and predict() honours
    # it, so the test phase scores the best round rather than the last.
    model = load_final_xgb_model(run_id, targets=TARGETS)
    best_iteration = int(model.best_iteration)
    assert 0 < best_iteration < XGBTrainerConfig.num_boost_round - 1


def test_partial_targets_early_stop_per_target(run, monkeypatch):
    """Unobserved targets are NaN, and must be dropped from val as well."""
    run_id, _ = run
    (X_train, y_train), (X_val, y_val) = _splits()
    y_train, y_val = y_train.copy(), y_val.copy()
    y_train[:20, 1] = np.nan
    y_val[:5, 1] = np.nan
    monkeypatch.setattr("configs.data.KEEP_PARTIAL_TARGETS", True)

    train_and_save_model(X_train, y_train, X_val, y_val, TARGETS, BEST_PARAMS, run_id)

    model = load_final_xgb_model(run_id, targets=TARGETS)
    assert all(int(m.best_iteration) > 0 for m in model.models)
    assert model.predict(X_val).shape == (len(X_val), len(TARGETS))


def test_the_callers_best_params_survive(run, monkeypatch):
    """train_xgb logs and returns best_params after the fit pops from it."""
    run_id, _ = run
    (X_train, y_train), (X_val, y_val) = _splits()
    monkeypatch.setattr("configs.data.KEEP_PARTIAL_TARGETS", False)
    best_params = dict(BEST_PARAMS)

    train_and_save_model(X_train, y_train, X_val, y_val, TARGETS, best_params, run_id)

    assert best_params == BEST_PARAMS


def test_a_saved_round_count_cannot_reach_the_booster(run, monkeypatch):
    """Runs searched before the round count left best_params still have one.

    It shares a name with a booster argument, so it has to be stripped rather
    than passed through -- and it must not become the round budget either.
    """
    import src.trainers.xgb_trainer as xgb_trainer

    run_id, _ = run
    (X_train, y_train), (X_val, y_val) = _splits()
    seen = {}

    class RecordingRegressor(xgb_trainer.XGBRegressor):
        def __init__(self, **kwargs):
            seen.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(xgb_trainer, "XGBRegressor", RecordingRegressor)
    monkeypatch.setattr("configs.data.KEEP_PARTIAL_TARGETS", False)
    legacy = {**BEST_PARAMS, "num_boost_round": 700}

    train_and_save_model(X_train, y_train, X_val, y_val, TARGETS, legacy, run_id)

    assert "num_boost_round" not in seen
    assert seen["n_estimators"] == XGBTrainerConfig.num_boost_round
