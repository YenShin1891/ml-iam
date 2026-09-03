"""Continuing a run must leave what it recorded at creation in place.

train_from_config.py used to rewrite meta/run_config.resolved.json from
whatever YAML launched the phase, so a resume file naming only run_id and the
phase blanked the dataset, two_window and keep_partial_targets the phase was
about to inherit from that very file.
"""

import json

import pytest

import src.utils.utils as utils_module
from scripts import train_from_config


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(utils_module, "get_run_root", lambda run_id: str(tmp_path / run_id))
    return tmp_path


@pytest.fixture
def phases_run(monkeypatch):
    calls = []
    monkeypatch.setattr(train_from_config, "_run_phase", lambda cfg, **kw: calls.append(kw["phase"]))
    return calls


def _config(path, text):
    path.write_text(text)
    return str(path)


def _recorded(results, run_id):
    return json.loads((results / run_id / "meta" / "run_config.resolved.json").read_text())


def test_a_resume_keeps_the_original_record(results, phases_run):
    train_from_config.main(["--run", _config(
        results / "full.yaml",
        "model: tft\nrun_id: tft_01\ndataset: pipeline-x\ntwo_window: true\n"
        "target_normalizer_mode: global\nkeep_partial_targets: false\nphases: [preprocess, train]\n",
    )])
    created = _recorded(results, "tft_01")
    assert created["dataset"] == "pipeline-x" and created["two_window"] is True

    train_from_config.main(["--run", _config(results / "resume.yaml", "model: tft\nrun_id: tft_01\nresume: test\n")])

    assert _recorded(results, "tft_01") == created
    assert len(list((results / "tft_01" / "meta").glob("run_config.resume.*.json"))) == 1
    assert phases_run == ["preprocess", "train", "test"]


def test_later_phases_under_an_existing_run_id_are_a_continuation_too(results, phases_run):
    train_from_config.main(["--run", _config(
        results / "full.yaml", "model: xgb\nrun_id: xgb_01\ndataset: pipeline-x\nphases: [preprocess]\n",
    )])
    created = _recorded(results, "xgb_01")

    train_from_config.main(["--run", _config(results / "more.yaml", "model: xgb\nrun_id: xgb_01\nphases: [test, plot]\n")])

    assert _recorded(results, "xgb_01") == created
    assert phases_run == ["preprocess", "test", "plot"]


def test_resuming_a_run_that_does_not_exist_fails_before_creating_it(results, phases_run):
    with pytest.raises(FileNotFoundError):
        train_from_config.main(["--run", _config(results / "resume.yaml", "model: xgb\nrun_id: xgb_09\nresume: test\n")])

    assert not (results / "xgb_09").exists()
    assert phases_run == []


def test_a_new_run_must_start_with_preprocess(results, phases_run):
    with pytest.raises(ValueError, match="preprocess"):
        train_from_config.main(["--run", _config(results / "bad.yaml", "model: xgb\nphases: [train, test]\n")])

    assert phases_run == []


def test_phases_default_to_the_whole_pipeline():
    cfg = train_from_config._parse_config({"model": "xgb"})

    assert cfg.phases == ("preprocess", "search", "train", "test", "plot")
