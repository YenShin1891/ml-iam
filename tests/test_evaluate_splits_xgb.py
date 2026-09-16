"""The XGB breakdown must score each split the way the test phase does.

The rollout scatters each group's predictions by index label, and split_data
restarts every split's index at 0, so combining splits without renumbering
writes train groups into val rows -- silently, and with plausible metrics.
"""

import numpy as np
import pandas as pd
import pytest

import scripts.evaluate_splits as evaluate_splits

TARGETS = ["A", "B"]


def test_a_split_with_duplicate_labels_is_refused():
    X = pd.DataFrame({"f0": [1.0, 2.0]}, index=[0, 0])

    with pytest.raises(ValueError, match="duplicate index"):
        evaluate_splits._eval_xgb_split(
            {"targets": TARGETS}, "train+val", X, np.zeros((2, 2)), pd.DataFrame(),
        )


@pytest.fixture
def recorded_frames(monkeypatch):
    """Assemble the splits with the rollout itself stubbed out."""
    import scripts.train_xgb as train_xgb
    import src.data.preprocess as preprocess
    import src.trainers.xgb_trainer as xgb_trainer

    def frame(start, n):
        return pd.DataFrame(
            {"f0": np.arange(n, dtype=float), "Population": np.ones(n)},
            index=pd.RangeIndex(n),  # split_data renumbers every split from 0
        ).assign(**{t: 1.0 for t in TARGETS})

    n_train, n_val, n_test = 6, 3, 4
    splits = {
        "n_lags": 2,
        "targets": TARGETS,
        "X_train_with_index": frame(0, n_train),
        "X_val_with_index": frame(0, n_val),
        "X_test_with_index": frame(0, n_test),
        "y_train": np.zeros((n_train, 2)),
        "y_val": np.ones((n_val, 2)),
        "y_test": np.full((n_test, 2), 2.0),
    }

    monkeypatch.setattr(train_xgb, "derive_splits", lambda *a, **k: splits)
    monkeypatch.setattr(preprocess, "prepare_features_and_targets", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(
        preprocess, "split_data",
        lambda *a, **k: (frame(0, n_train), frame(0, n_val), frame(0, n_test)),
    )
    monkeypatch.setattr(xgb_trainer, "load_final_xgb_model", lambda *a, **k: object())

    seen = {}

    def fake_eval(bundle, split_name, X_with_index, y_scaled, df):
        seen[split_name] = (X_with_index, y_scaled, df)
        return np.zeros((len(df), 2)), np.zeros((len(df), 2)), np.ones((len(df), 2), bool), len(df)

    monkeypatch.setattr(evaluate_splits, "_eval_xgb_split", fake_eval)

    class FakeStore:
        run_id = "xgb_99"
        load_processed_data = staticmethod(lambda: None)
        load_best_params = staticmethod(lambda: {"n_lags": 2})
        splits_for = staticmethod(lambda _data: None)
        load_artifact = staticmethod(lambda _name: None)

    evaluate_splits._build_split_results_xgb(FakeStore(), "xgb_99")
    return seen


def test_every_split_is_scored(recorded_frames):
    assert set(recorded_frames) == {"train", "val", "train+val", "test"}


def test_the_combined_split_renumbers_its_rows(recorded_frames):
    """Otherwise val rows reuse train labels and predictions land on the wrong rows."""
    X_combined, y_combined, df_combined = recorded_frames["train+val"]
    n_train = len(recorded_frames["train"][2])
    n_val = len(recorded_frames["val"][2])

    assert X_combined.index.is_unique
    assert len(X_combined) == n_train + n_val
    assert len(df_combined) == n_train + n_val
    # The scaled targets are stacked positionally, so the frame rows have to
    # stay in the same order: train first, then val.
    assert len(y_combined) == len(X_combined)
    np.testing.assert_array_equal(y_combined[:n_train], recorded_frames["train"][1])
    np.testing.assert_array_equal(y_combined[n_train:], recorded_frames["val"][1])


def test_each_split_keeps_the_frame_its_targets_came_from(recorded_frames):
    """Ground truth is read from the raw frame, so lengths must line up."""
    for split_name, (X_with_index, y_scaled, df) in recorded_frames.items():
        assert len(X_with_index) == len(y_scaled) == len(df), split_name
        assert X_with_index.index.is_unique, split_name
