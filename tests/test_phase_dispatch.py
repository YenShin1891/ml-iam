"""Phase dispatch resolves by name instead of five if/elif chains."""

import inspect

import pytest

import scripts.train as train
from scripts.train import _ALLOWED_MODELS, _ALLOWED_PHASES, _phase_function, run_phase


class FakeStore:
    run_id = "xgb_01"


# ── resolution ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model", _ALLOWED_MODELS)
@pytest.mark.parametrize("phase", _ALLOWED_PHASES)
def test_every_model_phase_pair_resolves(model, phase):
    function = _phase_function(model, phase)

    assert callable(function)
    assert "store" in inspect.signature(function).parameters


def test_preprocess_is_shared_by_every_model():
    """It was three byte-identical copies."""
    resolved = {_phase_function(model, "preprocess") for model in _ALLOWED_MODELS}

    assert resolved == {train.preprocess}


@pytest.mark.parametrize(
    "model,phase,expected",
    [("xgb", "search", "search_xgb"), ("lstm", "train", "train_lstm"), ("tft", "test", "test_tft")],
)
def test_resolution_follows_the_naming_convention(model, phase, expected):
    assert _phase_function(model, phase).__name__ == expected


def test_unknown_phase_is_an_attribute_error():
    with pytest.raises(AttributeError):
        _phase_function("xgb", "nonexistent")


# ── option filtering ──────────────────────────────────────────────────────


def test_options_the_model_supports_are_passed(monkeypatch):
    seen = {}

    def fake_test_tft(store, use_two_window=False):
        seen["use_two_window"] = use_two_window
        return "ran"

    monkeypatch.setattr(train, "_phase_function", lambda m, p: fake_test_tft)

    assert run_phase("tft", "test", FakeStore(), use_two_window=True) == "ran"
    assert seen == {"use_two_window": True}


def test_options_the_model_does_not_support_are_dropped(monkeypatch):
    """XGB phases take no normalizer mode; they must not receive one."""
    def fake_search_xgb(store):
        return "ran"

    monkeypatch.setattr(train, "_phase_function", lambda m, p: fake_search_xgb)

    assert run_phase(
        "xgb", "search", FakeStore(), target_normalizer_mode="global", use_two_window=True
    ) == "ran"


def test_dropping_a_set_option_is_logged(monkeypatch, caplog):
    monkeypatch.setattr(train, "_phase_function", lambda m, p: lambda store: None)

    with caplog.at_level("INFO"):
        run_phase("xgb", "test", FakeStore(), use_two_window=True, dataset=None)

    assert "does not support use_two_window" in caplog.text
    # An unset option is not worth mentioning.
    assert "does not support dataset" not in caplog.text


def test_the_real_tft_phases_accept_their_options():
    """Guards the filter against a rename silently dropping a real option."""
    assert "target_normalizer_mode" in inspect.signature(_phase_function("tft", "search")).parameters
    assert "target_normalizer_mode" in inspect.signature(_phase_function("tft", "train")).parameters
    assert "use_two_window" in inspect.signature(_phase_function("tft", "test")).parameters
    assert "dataset" in inspect.signature(_phase_function("tft", "preprocess")).parameters
