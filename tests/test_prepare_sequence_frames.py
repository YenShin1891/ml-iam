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


def test_a_resampling_that_the_unobserved_row_drop_undoes_is_reported(prepared_frame, monkeypatch, caplog):
    import configs.data as data_config

    monkeypatch.setattr(preprocess, "OUTPUT_VARIABLES", ["A", "B"])
    monkeypatch.setattr(data_config, "KEEP_PARTIAL_TARGETS", True)
    monkeypatch.setattr(data_config, "IMPUTE_IRREGULAR_INTERVALS", True)
    monkeypatch.setattr(data_config, "INTERPOLATE_TARGETS", False)
    monkeypatch.setattr(data_config, "NORMALIZE_TARGETS_BY_POPULATION", False)
    # M0 reports every ten years; every other group keeps its five-year steps.
    frame = prepared_frame[(prepared_frame["Model"] != "M0") | prepared_frame["Year"].isin([2020, 2030])]

    with caplog.at_level("INFO"):
        prepared, _, _ = preprocess.prepare_features_and_targets_sequence(frame.copy())

    assert "inserted 1 rows with no observed target" in caplog.text
    assert "IMPUTE_IRREGULAR_INTERVALS has no effect: 1 of the 1 rows" in caplog.text
    assert sorted(prepared.loc[prepared["Model"] == "M0", "Year"]) == [2020, 2030]
