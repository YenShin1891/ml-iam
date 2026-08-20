"""Two-window TFT predictions must be labelled with the decoder's own steps."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.trainers.tft_two_window_simple import (
    _create_early_window_test_data,
    _create_late_window_test_data,
    _expand_horizon_index,
)

ENCODER_LENGTH, PREDICTION_LENGTH = 3, 12
WINDOW = ENCODER_LENGTH + PREDICTION_LENGTH


@pytest.fixture
def trajectories():
    """Three trajectories of 17 steps, one too short for either window."""
    rows = [
        {"Model": f"M{g}", "Scenario": f"S{g}", "Region": "World", "Step": step}
        for g in range(3)
        for step in range(17 if g < 2 else WINDOW - 1)
    ]
    return pd.DataFrame(rows)


# ── horizon expansion ─────────────────────────────────────────────────────


def test_horizon_steps_follow_the_first_decoder_step():
    """Prediction.index carries the first decoder step, so steps are base + h."""
    idx_df = pd.DataFrame({"Model": ["M0", "M1"], "Step": [3, 5]})
    preds = torch.arange(2 * 4 * 2, dtype=torch.float32).reshape(2, 4, 2)

    expanded, flat = _expand_horizon_index(idx_df, preds, "Step", torch)

    assert expanded["Step"].tolist() == [3, 4, 5, 6, 5, 6, 7, 8]
    assert expanded["Model"].tolist() == ["M0"] * 4 + ["M1"] * 4
    assert flat.shape == (8, 2)
    # Row order must stay (sample, horizon) so preds line up with the index.
    np.testing.assert_array_equal(flat, preds.reshape(8, 2).numpy())


def test_early_window_is_not_labelled_from_the_start_of_the_trajectory():
    """The old mode="early" rule shifted predictions back by encoder_length."""
    idx_df = pd.DataFrame({"Model": ["M0"], "Step": [ENCODER_LENGTH]})
    preds = torch.zeros(1, PREDICTION_LENGTH, 2)

    expanded, _ = _expand_horizon_index(idx_df, preds, "Step", torch)

    steps = expanded["Step"].tolist()
    assert steps == list(range(ENCODER_LENGTH, WINDOW))
    assert steps != list(range(PREDICTION_LENGTH))  # what the old code assigned


def test_non_expandable_predictions_pass_through():
    idx_df = pd.DataFrame({"Model": ["M0", "M1"], "Step": [3, 5]})
    preds = torch.zeros(2, 2)  # already (batch, targets)

    expanded, flat = _expand_horizon_index(idx_df, preds, "Step", torch)

    assert expanded["Step"].tolist() == [3, 5]
    assert flat.shape == (2, 2)


def test_missing_time_index_column_is_an_error():
    idx_df = pd.DataFrame({"Model": ["M0"]})
    preds = torch.zeros(1, 4, 2)

    with pytest.raises(KeyError, match="Step"):
        _expand_horizon_index(idx_df, preds, "Step", torch)


# ── window slicing ────────────────────────────────────────────────────────


def test_early_window_takes_the_first_span_of_long_enough_trajectories(trajectories):
    early = _create_early_window_test_data(trajectories, WINDOW, "Step")

    assert set(early["Model"]) == {"M0", "M1"}  # M2 is too short
    for _, group in early.groupby("Model"):
        assert sorted(group["Step"]) == list(range(WINDOW))


def test_late_window_ends_at_the_last_step(trajectories):
    late = _create_late_window_test_data(trajectories, WINDOW, "Step")

    assert set(late["Model"]) == {"M0", "M1"}
    for _, group in late.groupby("Model"):
        assert sorted(group["Step"]) == list(range(17 - WINDOW, 17))


def test_windows_overlap_where_the_blend_applies(trajectories):
    early = _create_early_window_test_data(trajectories, WINDOW, "Step")
    late = _create_late_window_test_data(trajectories, WINDOW, "Step")

    overlap = set(early["Step"]) & set(late["Step"])
    assert overlap, "no overlap means the weighted blend never runs"
