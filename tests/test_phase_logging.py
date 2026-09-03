"""Each phase runs in its own process and appends to one train.log.

Without a banner the log read as an undifferentiated stream, restating "Logging
is set up" and the run note at every phase while never recording which dataset
the run used.
"""

import logging

import pytest

from scripts import train
from scripts.train import _log_run_header, run_phase


class FakeStore:
    run_id = "xgb_09"


@pytest.fixture
def args():
    class Args:
        dataset = None
        target_normalizer_mode = None
        two_window = None
        keep_partial_targets = None
        note = None

    return Args()


def _messages(caplog):
    return [r.getMessage() for r in caplog.records]


def test_a_phase_is_bracketed_by_its_position_and_duration(monkeypatch, caplog, args):
    monkeypatch.setattr(train, "_phase_function", lambda m, p: lambda store: None)

    with caplog.at_level(logging.INFO):
        train._run("xgb", "search", FakeStore(), args,
                   phases=("preprocess", "search", "train", "test", "plot"))

    started, finished = _messages(caplog)
    assert started == "--- xgb_09 phase 2/5: search ---"
    assert finished.startswith("--- xgb_09 phase 2/5: search done in ")


def test_a_phase_outside_the_recorded_list_still_gets_a_banner(monkeypatch, caplog, args):
    """A hand-resumed phase need not appear in the run's recorded plan."""
    monkeypatch.setattr(train, "_phase_function", lambda m, p: lambda store: None)

    with caplog.at_level(logging.INFO):
        train._run("xgb", "plot", FakeStore(), args, phases=("preprocess",))

    assert _messages(caplog)[0] == "--- xgb_09 phase: plot ---"


def test_the_header_records_what_the_run_was(caplog, args):
    args.dataset = "pipeline-2026-04-13"
    args.note = "mask unobserved targets"
    resolved = {
        "phases": ["preprocess", "search"],
        "cuda_visible_devices_resolved_by_phase": {"default": "5", "search": "5,6,7,8"},
        "keep_partial_targets": True,
    }

    with caplog.at_level(logging.INFO):
        _log_run_header("xgb_09", "xgb", args, resolved)

    header, note = _messages(caplog)
    assert "model=xgb" in header
    assert "dataset=pipeline-2026-04-13" in header
    assert "phases=preprocess, search" in header
    assert "search:5,6,7,8" in header
    assert "keep_partial_targets=True" in header
    assert note == "Run note: mask unobserved targets"


def test_settings_that_were_never_set_stay_out_of_the_header(caplog, args):
    with caplog.at_level(logging.INFO):
        _log_run_header("xgb_09", "xgb", args, {})

    header = _messages(caplog)[0]
    assert "two_window" not in header
    assert "target_normalizer_mode" not in header


def test_an_option_another_phase_consumes_is_not_reported(monkeypatch, caplog):
    """`dataset` is set for the whole run but only preprocess reads it."""
    def only_preprocess_takes_dataset(model, phase):
        if phase == "preprocess":
            return lambda store, dataset=None: None
        return lambda store: None

    monkeypatch.setattr(train, "_phase_function", only_preprocess_takes_dataset)

    with caplog.at_level(logging.INFO):
        run_phase("xgb", "search", FakeStore(), dataset="pipeline-2026-04-13")

    assert "does not support" not in caplog.text


def test_an_option_no_phase_consumes_is_still_reported(monkeypatch, caplog):
    monkeypatch.setattr(train, "_phase_function", lambda m, p: lambda store: None)

    with caplog.at_level(logging.INFO):
        run_phase("xgb", "search", FakeStore(), use_two_window=True)

    assert "does not support use_two_window" in caplog.text
