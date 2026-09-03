"""setup_logging must be re-entrant and carry Lightning's messages to the log."""

import logging

import pytest

import src.utils.utils as utils_module


@pytest.fixture
def root_logger_restored():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    for handler in root.handlers[:]:
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    for handler in handlers:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(level)


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(utils_module, "get_run_root", lambda run_id: str(tmp_path / run_id))
    return tmp_path


def _flush_root():
    for handler in logging.getLogger().handlers:
        handler.flush()


def test_a_second_run_gets_its_own_file(runs, root_logger_restored):
    utils_module.setup_logging("xgb_01", log_file="train.log")
    utils_module.setup_logging("xgb_02", log_file="train.log")

    logging.getLogger().info("second run")
    _flush_root()

    assert "second run" in (runs / "xgb_02" / "logs" / "train.log").read_text()
    assert "second run" not in (runs / "xgb_01" / "logs" / "train.log").read_text()


def test_lightning_messages_reach_the_run_log(runs, root_logger_restored, monkeypatch):
    # What lightning.pytorch does at import when the root has no handlers yet.
    library = logging.getLogger("lightning.pytorch")
    monkeypatch.setattr(library, "propagate", False)
    library.addHandler(logging.NullHandler())

    utils_module.setup_logging("tft_01", log_file="train.log")
    library.info("GPU available: True")
    _flush_root()

    assert "GPU available: True" in (runs / "tft_01" / "logs" / "train.log").read_text()
