"""Each booster reads only the lagged outputs that are always there to read."""

import json

import numpy as np
import pandas as pd
import pytest

from src.trainers.xgb_trainer import PerTargetXGBRegressor, hidden_lag_columns

TARGETS = ["A", "B", "C"]
FEATURES = ["feat", "prev_A", "prev_B", "prev_C", "prev2_A", "prev2_B", "prev2_C"]
KWARGS = dict(n_estimators=8, max_depth=3, tree_method="hist", device="cpu", n_jobs=1)


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    y = pd.DataFrame({t: X[f"prev_{t}"] + 0.5 * X["feat"] + rng.normal(scale=0.1, size=n) for t in TARGETS})
    # B is not reported by the last third of the rows: no target, no lag.
    unreported = np.arange(n) >= 2 * n // 3
    y.loc[unreported, "B"] = np.nan
    X.loc[unreported, ["prev_B", "prev2_B"]] = np.nan
    return X, y


def test_scopes_hide_the_right_columns():
    _, y = _data()
    position = FEATURES.index

    assert hidden_lag_columns(FEATURES, TARGETS, y, "all") == [[], [], []]

    own = hidden_lag_columns(FEATURES, TARGETS, y, "own")
    assert own[0] == sorted(position(c) for c in ("prev_B", "prev_C", "prev2_B", "prev2_C"))
    assert position("feat") not in own[0] + own[1] + own[2]

    # A and C are reported on every row, B is not: only B's lags are withheld,
    # and only from the other outputs' boosters.
    mixed = hidden_lag_columns(FEATURES, TARGETS, y, "own_and_complete")
    assert mixed[0] == mixed[2] == [position("prev_B"), position("prev2_B")]
    assert mixed[1] == []

    with pytest.raises(ValueError, match="lag_scope"):
        hidden_lag_columns(FEATURES, TARGETS, y, "some")


def test_a_withheld_lag_cannot_move_the_prediction():
    """What the rollout writes into an unreported output's lag must not reach the others."""
    X, y = _data()
    model = PerTargetXGBRegressor(TARGETS, lag_scope="own_and_complete", **KWARGS)
    model.fit(X, y)

    rows = X.to_numpy()[-20:].copy()          # series that do not report B: its lags are missing
    filled = rows.copy()
    # the rollout's own predictions, written into the empty lag slots
    filled[:, [FEATURES.index("prev_B"), FEATURES.index("prev2_B")]] = np.linspace(-3, 3, 20)[:, None]

    before, after = model.predict(rows), model.predict(filled)
    assert np.array_equal(before[:, [0, 2]], after[:, [0, 2]])
    assert not np.array_equal(before[:, 1], after[:, 1])       # B's own booster does read them

    everything = PerTargetXGBRegressor(TARGETS, lag_scope="all", **KWARGS)
    everything.fit(X, y)
    assert not np.array_equal(everything.predict(rows)[:, [0, 2]], everything.predict(filled)[:, [0, 2]])


def test_the_scope_is_saved_with_the_model(tmp_path):
    X, y = _data()
    model = PerTargetXGBRegressor(TARGETS, lag_scope="own", **KWARGS)
    model.fit(X, y)
    path = str(tmp_path / "final_best.json")
    model.save_model(path)

    saved = json.loads((tmp_path / "final_best_lag_scope.json").read_text())
    assert saved["lag_scope"] == "own" and saved["targets"] == TARGETS

    loaded = PerTargetXGBRegressor.load_model(path, TARGETS)
    assert loaded.hidden_columns == model.hidden_columns
    rows = X.to_numpy()[:30]
    assert np.allclose(loaded.predict(rows), model.predict(rows))


def test_a_model_saved_without_the_record_reads_every_lag(tmp_path):
    X, y = _data()
    model = PerTargetXGBRegressor(TARGETS, **KWARGS)        # the default is the old behaviour
    model.fit(X, y)
    path = str(tmp_path / "final_best.json")
    model.save_model(path)
    (tmp_path / "final_best_lag_scope.json").unlink()        # as runs up to xgb_87 were saved

    loaded = PerTargetXGBRegressor.load_model(path, TARGETS)
    assert loaded.lag_scope == "all" and loaded.hidden_columns == [[], [], []]
    rows = X.to_numpy()[:30]
    assert np.allclose(loaded.predict(rows), model.predict(rows))


# ---------------------------------------------------------------------------
# The scope is a setting of the run: chosen in its YAML, recorded when the run
# is created, and read back by the search and the final fit.
# ---------------------------------------------------------------------------

def _resolved(tmp_path, monkeypatch, obj):
    from scripts import train_from_config as launcher

    monkeypatch.setattr(launcher, "_meta_dir", lambda _run_id: tmp_path)
    config = tmp_path / "run.yaml"
    config.write_text("model: xgb\n")
    cfg = launcher._parse_config(obj)
    launcher._write_run_metadata(cfg, run_id="xgb_01", config_path=config, env={}, cuda_by_phase_resolved={})
    return json.loads((tmp_path / "run_config.resolved.json").read_text())


def test_a_new_run_records_the_scope_its_yaml_names(tmp_path, monkeypatch):
    resolved = _resolved(tmp_path, monkeypatch, {"model": "xgb", "phases": ["preprocess"], "lag_scope": "own"})
    assert resolved["lag_scope"] == "own"


def test_a_new_run_records_the_default_scope_when_its_yaml_names_none(tmp_path, monkeypatch):
    from configs.models.xgb import XGBTrainerConfig

    resolved = _resolved(tmp_path, monkeypatch, {"model": "xgb", "phases": ["preprocess"]})
    assert resolved["lag_scope"] == XGBTrainerConfig().lag_scope


def test_the_launcher_refuses_a_scope_it_cannot_honour():
    from scripts import train_from_config as launcher

    with pytest.raises(ValueError, match="must be one of"):
        launcher._parse_config({"model": "xgb", "phases": ["preprocess"], "lag_scope": "some"})
    with pytest.raises(ValueError, match="runs only"):
        launcher._parse_config({"model": "lstm", "phases": ["preprocess"], "lag_scope": "own"})
    assert launcher._parse_config({"model": "linear", "phases": ["preprocess"], "lag_scope": "own"}).lag_scope == "own"
    resumed = launcher._parse_config({"model": "xgb", "run_id": "xgb_01", "resume": "train", "lag_scope": "own"})
    with pytest.raises(ValueError, match="fixed when a run is created"):
        launcher._validate_model_constraints(resumed)


def test_the_run_store_reads_the_recorded_scope(tmp_path, monkeypatch):
    import src.utils.run_store as run_store_module
    from src.utils.run_store import RunStore

    monkeypatch.setattr(run_store_module, "get_run_root", lambda _run_id: str(tmp_path))
    store = RunStore("xgb_01")
    assert store.lag_scope() == "all"            # no record: a run from before the setting
    store.record_settings(lag_scope="own")
    assert store.lag_scope() == "own"


def test_the_final_fit_uses_the_scope_it_is_handed(tmp_path, monkeypatch):
    from src.trainers import xgb_trainer

    monkeypatch.setattr(xgb_trainer, "get_run_root", lambda _run_id: str(tmp_path))
    monkeypatch.setattr(xgb_trainer, "_first_visible_gpu_token", lambda *a, **k: "")
    monkeypatch.setattr(
        xgb_trainer, "get_xgb_params",
        lambda params, trainer_cfg=None: {**params, "tree_method": "hist", "device": "cpu", "n_jobs": 1},
    )
    X, y = _data()
    xgb_trainer.train_and_save_model(
        X.iloc[:200], y.iloc[:200].to_numpy(), X.iloc[200:].reset_index(drop=True), y.iloc[200:].to_numpy(),
        TARGETS, {"max_depth": 3}, "xgb_01", lag_scope="own",
    )
    saved = json.loads((tmp_path / "checkpoints" / "final_best_lag_scope.json").read_text())
    assert saved["lag_scope"] == "own"
