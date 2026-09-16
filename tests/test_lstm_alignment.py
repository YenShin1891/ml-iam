"""Predictions land on the rows the dataset built its sequences from."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from src.trainers.lstm_trainer import LSTMDataset, align_sequence_predictions

FEATURES = ["feat"]
TARGETS = ["A", "B"]
GROUP_IDS = ["Model", "Scenario"]


def _frame(group_lengths):
    rows = []
    for g, length in enumerate(group_lengths):
        for step in range(length):
            rows.append(
                {
                    "Model": f"M{g}", "Scenario": f"S{g}", "Step": step,
                    "feat": float(g * 10 + step),
                    "A": float(g), "B": float(step),
                    "A__observed": 1.0, "B__observed": 1.0,
                }
            )
    return pd.DataFrame(rows)


def _dataset(frame, sequence_length=1, target_offset=0):
    return LSTMDataset(
        frame, FEATURES, TARGETS,
        group_ids=GROUP_IDS,
        sequence_length=sequence_length,
        target_offset=target_offset,
    )


# ── the dataset records its own mapping ───────────────────────────────────


def test_target_positions_match_the_sequence_count():
    frame = _frame([5, 4])
    dataset = _dataset(frame, sequence_length=2)

    assert len(dataset.target_positions) == len(dataset)


@pytest.mark.parametrize("sequence_length", [1, 2, 3])
@pytest.mark.parametrize("target_offset", [0, 1])
def test_the_first_predicted_row_does_not_move_with_the_context_length(
    sequence_length, target_offset
):
    """Every context length predicts the same rows.

    A shorter context fits more windows into a series, and the extra ones sit
    at its easiest end, so letting them in would make a short context look
    better for being scored on different data.
    """
    from configs.data import MAX_CONTEXT_LENGTH

    dataset = _dataset(_frame([6]), sequence_length, target_offset)

    assert dataset.target_positions[0] == MAX_CONTEXT_LENGTH - 1 + target_offset


def test_a_longer_context_still_starts_where_its_own_window_ends():
    """Beyond the compared range the window itself is the binding constraint."""
    from configs.data import MAX_CONTEXT_LENGTH

    sequence_length = MAX_CONTEXT_LENGTH + 2
    dataset = _dataset(_frame([9]), sequence_length)

    assert dataset.target_positions[0] == sequence_length - 1


def test_groups_too_short_contribute_no_sequences():
    frame = _frame([6, 1])  # the second group cannot fill one sequence
    dataset = _dataset(frame, sequence_length=3)

    assert dataset.target_positions.max() < 6


# ── scattering back ───────────────────────────────────────────────────────


def test_predictions_land_on_their_own_rows():
    frame = _frame([5, 4])
    dataset = _dataset(frame, sequence_length=2)
    predictions = np.arange(len(dataset) * 2, dtype=float).reshape(len(dataset), 2)

    aligned = align_sequence_predictions(dataset, predictions, len(frame))

    assert aligned.shape == (len(frame), 2)
    for i, row in enumerate(dataset.target_positions):
        np.testing.assert_array_equal(aligned[row], predictions[i])


def test_rows_without_a_prediction_stay_nan():
    frame = _frame([5])
    dataset = _dataset(frame, sequence_length=3)
    predictions = np.zeros((len(dataset), 2))

    aligned = align_sequence_predictions(dataset, predictions, len(frame))

    unpredicted = set(range(len(frame))) - set(dataset.target_positions.tolist())
    assert unpredicted, "fixture should leave some rows unpredicted"
    for row in unpredicted:
        assert np.isnan(aligned[row]).all()


def test_alignment_matches_the_re_derived_mapping():
    """The old callers recomputed this from group sizes; results must agree."""
    frame = _frame([6, 5, 4])
    sequence_length, target_offset = 2, 1
    dataset = _dataset(frame, sequence_length, target_offset)
    predictions = np.arange(len(dataset) * 2, dtype=float).reshape(len(dataset), 2)

    aligned = align_sequence_predictions(dataset, predictions, len(frame))

    from configs.data import MAX_CONTEXT_LENGTH

    expected = np.full((len(frame), 2), np.nan)
    pred_idx = 0
    # Mirrors LSTMDataset: windows start at the offset that makes every
    # context length predict the same rows.
    first_start = max(0, MAX_CONTEXT_LENGTH - sequence_length)
    for _, group_data in frame.groupby(GROUP_IDS):
        max_start = len(group_data) - (sequence_length + target_offset) + 1
        for i in range(first_start, max(first_start, max_start)):
            target = group_data.index[i + sequence_length - 1 + target_offset]
            expected[frame.index.get_loc(target)] = predictions[pred_idx]
            pred_idx += 1

    np.testing.assert_array_equal(aligned, expected)


def test_a_prediction_count_mismatch_is_an_error():
    """The old loops silently dropped extras via `if pred_idx < len(...)`."""
    dataset = _dataset(_frame([5]), sequence_length=2)

    with pytest.raises(ValueError, match="predictions for"):
        align_sequence_predictions(dataset, np.zeros((len(dataset) + 3, 2)), 5)
