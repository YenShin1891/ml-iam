"""A run prepares its lag features the way it was trained, whatever the default is today."""

import json

import pytest

import configs.data as data_config
import src.utils.run_store as run_store_module
from src.utils.run_store import RunStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store_module, "get_run_root", lambda _run_id: str(tmp_path))
    return RunStore("xgb_01")


def _record(store, **values):
    meta = store.root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    meta.joinpath("run_config.resolved.json").write_text(json.dumps(values))


def test_a_run_recorded_before_the_setting_existed_used_linear(store):
    _record(store, model="xgb", dataset="v1")
    assert store.lag_fill() == "linear"


def test_the_recorded_setting_wins_over_the_default(store, monkeypatch):
    monkeypatch.setattr(data_config, "LAG_FILL", "linear")
    _record(store, lag_fill="carry_forward")
    assert store.lag_fill() == "carry_forward"


def test_a_run_without_any_record_predates_the_setting(store, monkeypatch):
    monkeypatch.setattr(data_config, "LAG_FILL", "carry_forward")
    assert store.lag_fill() == "linear"


def test_recording_keeps_what_was_already_recorded(store):
    _record(store, dataset="v1")
    store.record_settings(lag_fill="carry_forward")
    assert store.recorded_settings() == {"dataset": "v1", "lag_fill": "carry_forward"}
    assert store.lag_fill() == "carry_forward"


def test_an_unreadable_record_is_no_record(store):
    meta = store.root / "meta"
    meta.mkdir(parents=True)
    meta.joinpath("run_config.resolved.json").write_text("{not json")
    assert store.lag_fill() == "linear"


def test_models_without_lag_features_record_none_and_refuse_the_key(tmp_path, monkeypatch):
    from scripts import train_from_config as launcher

    monkeypatch.setattr(launcher, "_meta_dir", lambda _run_id: tmp_path)
    config = tmp_path / "run.yaml"
    config.write_text("model: tft\n")
    cfg = launcher._parse_config({"model": "tft", "phases": ["preprocess", "train"]})
    launcher._write_run_metadata(cfg, run_id="tft_01", config_path=config, env={}, cuda_by_phase_resolved={})
    assert json.loads((tmp_path / "run_config.resolved.json").read_text())["lag_fill"] is None

    with pytest.raises(ValueError, match="builds no lag features"):
        launcher._parse_config({"model": "tft", "phases": ["preprocess"], "lag_fill": "linear"})


def test_a_new_run_records_a_concrete_setting(tmp_path, monkeypatch):
    from scripts import train_from_config as launcher

    monkeypatch.setattr(launcher, "_meta_dir", lambda _run_id: tmp_path)
    config = tmp_path / "run.yaml"
    config.write_text("model: xgb\n")
    cfg = launcher._parse_config({"model": "xgb", "phases": ["preprocess", "train"]})

    launcher._write_run_metadata(cfg, run_id="xgb_01", config_path=config, env={}, cuda_by_phase_resolved={})

    resolved = json.loads((tmp_path / "run_config.resolved.json").read_text())
    assert resolved["lag_fill"] == data_config.LAG_FILL


def test_the_setting_cannot_be_changed_on_an_existing_run():
    from scripts import train_from_config as launcher

    cfg = launcher._parse_config({"model": "xgb", "run_id": "xgb_01", "resume": "test", "lag_fill": "linear"})
    with pytest.raises(ValueError, match="fixed when a run is created"):
        launcher._validate_model_constraints(cfg)
