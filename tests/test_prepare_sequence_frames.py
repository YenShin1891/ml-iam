"""The sequence-model frames are built by one torch-free function the TFT phases share."""

import numpy as np
import pandas as pd
import pytest

import src.data.preprocess as preprocess
from src.data.preprocess import SequenceFrames, assign_group_splits, prepare_sequence_frames


@pytest.fixture
def frame(prepared_frame, monkeypatch):
    """conftest's frame, with the module reading its targets and one group's feature missing."""
    monkeypatch.setattr(preprocess, "OUTPUT_VARIABLES", ["A", "B"])
    frame = prepared_frame.copy()
    frame.loc[frame["Model"] == "M3", "feat"] = np.nan
    return frame


def test_frames_cover_every_row_with_indicators_and_imputed_features(frame):
    frames = prepare_sequence_frames(frame, assign_group_splits(frame))

    assert isinstance(frames, SequenceFrames)
    assert len(frames.train) + len(frames.val) + len(frames.test) == len(frame)
    assert frames.targets == ["A", "B"]
    assert "feat" in frames.features and "feat_is_missing" in frames.features
    every = pd.concat([frames.train, frames.val, frames.test])
    assert every["feat"].notna().all()
    assert every.loc[every["Model"] == "M3", "feat_is_missing"].eq(1.0).all()
    assert {"Step", "DeltaYears"} <= set(every.columns)


def test_unobserved_targets_are_zero_filled_for_the_masked_loss(frame):
    frames = prepare_sequence_frames(frame, assign_group_splits(frame))

    every = pd.concat([frames.train, frames.val, frames.test])
    assert every.loc[every["B__observed"] == 0.0, "B"].eq(0.0).all()


def test_the_saved_assignment_decides_the_split(frame):
    assignment = assign_group_splits(frame)
    assignment["split"] = "test"

    frames = prepare_sequence_frames(frame, assignment)

    assert frames.train.empty and frames.val.empty and len(frames.test) == len(frame)


def test_derive_splits_returns_these_frames(monkeypatch):
    pytest.importorskip("torch")
    from scripts.train_tft import derive_splits

    sentinel = SequenceFrames(
        train=pd.DataFrame({"x": [1]}), val=pd.DataFrame({"x": [2]}), test=pd.DataFrame({"x": [3]}),
        features=["x"], targets=["y"],
    )
    seen = {}

    def fake_prepare(data, assignment=None):
        seen["assignment"] = assignment
        return sentinel

    monkeypatch.setattr(preprocess, "prepare_sequence_frames", fake_prepare)

    splits = derive_splits(pd.DataFrame({"x": [0]}))

    assert splits["train_data"] is sentinel.train and splits["test_data"] is sentinel.test
    assert splits["features"] == ["x"] and splits["targets"] == ["y"]
    assert seen["assignment"] is None
    assert {"tft_target_offset", "tft_min_encoder_length", "tft_max_encoder_length", "tft_time_idx_column"} <= set(splits)
