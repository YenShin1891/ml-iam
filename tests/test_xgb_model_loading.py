"""Both saved model layouts must load, and SHAP must not silently skip either."""

import numpy as np
import pandas as pd
import pytest

from src.trainers.xgb_trainer import (
    PerTargetXGBRegressor,
    count_final_target_models,
    has_final_xgb_model,
    load_final_xgb_model,
)

TARGETS = ["A", "B"]


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A fake run directory that get_run_root() resolves to."""
    import src.trainers.xgb_trainer as xgb_trainer

    run_id = "xgb_99"
    (tmp_path / "checkpoints").mkdir()
    monkeypatch.setattr(xgb_trainer, "get_run_root", lambda _run_id: str(tmp_path))
    return run_id, tmp_path


@pytest.fixture
def xy():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=["f0", "f1", "f2"])
    y = pd.DataFrame({"A": X["f0"] * 2, "B": X["f1"] * -1})
    return X, y


def _save_per_target(run, xy):
    run_id, root = run
    X, y = xy
    model = PerTargetXGBRegressor(targets=TARGETS, n_estimators=4, max_depth=2)
    model.fit(X, y, verbose=False)
    model.save_model(str(root / "checkpoints" / "final_best.json"))
    return run_id


def _save_single(run, xy):
    from xgboost import XGBRegressor

    run_id, root = run
    X, y = xy
    model = XGBRegressor(n_estimators=4, max_depth=2).fit(X, y)
    model.save_model(str(root / "checkpoints" / "final_best.json"))
    return run_id


def test_per_target_layout_is_detected(run, xy):
    run_id = _save_per_target(run, xy)

    assert count_final_target_models(run_id) == 2
    assert has_final_xgb_model(run_id) is True

    model = load_final_xgb_model(run_id, TARGETS)
    assert isinstance(model, PerTargetXGBRegressor)
    assert model.predict(xy[0]).shape == (40, 2)


def test_single_multi_output_layout_still_loads(run, xy):
    run_id = _save_single(run, xy)

    assert count_final_target_models(run_id) == 0
    assert has_final_xgb_model(run_id) is True
    assert load_final_xgb_model(run_id, TARGETS).predict(xy[0]).shape == (40, 2)


def test_missing_model_raises(run):
    run_id, _ = run

    assert has_final_xgb_model(run_id) is False
    with pytest.raises(FileNotFoundError):
        load_final_xgb_model(run_id, TARGETS)


def test_shap_values_are_produced_for_per_target_models(run, xy, monkeypatch):
    """plot_xgb_shap used to skip these runs entirely."""
    import src.visualization.shap_xgb as shap_xgb

    run_id = _save_per_target(run, xy)
    monkeypatch.setattr(shap_xgb, "get_run_root", lambda _run_id: str(run[1]))

    shap_xgb.get_shap_values(run_id, xy[0], targets=TARGETS)

    saved = np.load(str(run[1] / "plots" / "shap_values.npy"))
    assert saved.shape == (40, 3, 2)  # rows, features, targets


def test_shap_shape_matches_between_layouts(run, xy, monkeypatch):
    import src.visualization.shap_xgb as shap_xgb

    monkeypatch.setattr(shap_xgb, "get_run_root", lambda _run_id: str(run[1]))

    run_id = _save_single(run, xy)
    shap_xgb.get_shap_values(run_id, xy[0], targets=TARGETS)
    single = np.load(str(run[1] / "plots" / "shap_values.npy")).shape

    _save_per_target(run, xy)
    shap_xgb.get_shap_values(run_id, xy[0], targets=TARGETS)
    per_target = np.load(str(run[1] / "plots" / "shap_values.npy")).shape

    assert single == per_target
