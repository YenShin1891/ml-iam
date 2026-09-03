"""A resumed phase must run under the settings its run was created with."""

import argparse
import json

import pytest

import src.utils.utils as utils_module
from scripts.train import _apply_run_settings


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(utils_module, "get_run_root", lambda _run_id: str(tmp_path))
    return tmp_path


def _write_resolved(root, **values):
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    meta.joinpath("run_config.resolved.json").write_text(json.dumps(values))


def _args(**overrides):
    defaults = dict(
        run_id="tft_91",
        keep_partial_targets=None,
        target_normalizer_mode=None,
        two_window=None,
        dataset=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_unspecified_settings_come_from_the_run(run_dir):
    _write_resolved(
        run_dir,
        keep_partial_targets=False,
        target_normalizer_mode="global",
        two_window=True,
        dataset="pipeline-v2",
    )
    args = _args()

    _apply_run_settings(args)

    assert args.keep_partial_targets is False
    assert args.target_normalizer_mode == "global"
    assert args.two_window is True
    assert args.dataset == "pipeline-v2"


def test_command_line_wins_and_says_so(run_dir, caplog):
    _write_resolved(run_dir, target_normalizer_mode="global")
    args = _args(target_normalizer_mode="encoder_floored")

    with caplog.at_level("WARNING"):
        _apply_run_settings(args)

    assert args.target_normalizer_mode == "encoder_floored"
    assert "overrides" in caplog.text


def test_settings_absent_from_the_record_are_left_alone(run_dir):
    _write_resolved(run_dir, two_window=True)
    args = _args()

    _apply_run_settings(args)

    assert args.two_window is True
    assert args.keep_partial_targets is None


def test_runs_without_a_recorded_config_are_untouched(run_dir):
    args = _args()

    _apply_run_settings(args)

    assert args == _args()


def test_an_explicit_no_wins_over_the_record(run_dir, caplog):
    """False is a choice (--no-keep-partial-targets), not an absent setting."""
    _write_resolved(run_dir, keep_partial_targets=True, two_window=True)
    args = _args(keep_partial_targets=False, two_window=False)

    with caplog.at_level("WARNING"):
        _apply_run_settings(args)

    assert args.keep_partial_targets is False
    assert args.two_window is False
    assert caplog.text.count("overrides") == 2


def test_unreadable_config_does_not_crash_the_phase(run_dir, caplog):
    meta = run_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    meta.joinpath("run_config.resolved.json").write_text("{not json")

    with caplog.at_level("WARNING"):
        _apply_run_settings(_args())

    assert "Could not read" in caplog.text
