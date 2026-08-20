"""LSTMDataset must fit its target scaler on observed values only."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from src.trainers.lstm_trainer import LSTMDataset


@pytest.fixture
def frame():
    rows = []
    for g in range(4):
        for step in range(5):
            rows.append(
                {
                    "Model": f"M{g}",
                    "Scenario": f"S{g}",
                    "Region": g,
                    "Step": step,
                    "feat": float(g + step),
                    "A": 100.0 + g + step,
                    "A__observed": 0.0 if step < 3 else 1.0,
                }
            )
    frame = pd.DataFrame(rows)
    frame.loc[frame["A__observed"] == 0, "A"] = np.nan
    return frame


def test_scaler_statistics_come_from_observed_targets(frame):
    dataset = LSTMDataset(
        frame,
        features=["feat"],
        targets=["A"],
        group_ids=["Model", "Scenario"],
    )

    observed = frame.loc[frame["A__observed"] == 1, "A"]
    assert dataset.scaler_y.mean_[0] == pytest.approx(observed.mean())
    assert dataset.scaler_y.scale_[0] == pytest.approx(observed.std(ddof=0))
    # The old zero fill would have pulled the mean far below any real value.
    assert dataset.scaler_y.mean_[0] > observed.min() * 0.9


def test_scaled_targets_stay_finite(frame):
    dataset = LSTMDataset(
        frame,
        features=["feat"],
        targets=["A"],
        group_ids=["Model", "Scenario"],
    )

    assert np.isfinite(dataset.y_scaled).all()
    assert dataset.y_sequences.isfinite().all()
