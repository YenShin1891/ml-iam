"""Every model should report metrics per region scale, in readable numbers.

Only XGB passed its frame to save_metrics, so the LSTM and TFT runs recorded a
single overall row -- the per-scale comparison the three models are read
against each other on was missing from both the log and performance.csv.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from src.trainers import evaluation
from src.trainers.evaluation import _metrics_line, save_metrics
from src.utils.regions import scale_of_frame


@pytest.fixture
def run_root(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "get_run_root", lambda _run_id: str(tmp_path))
    return tmp_path


REGIONS = ["World", "R5ASIA", "R10AFRICA", "KOR"]


def _frame(regions=REGIONS, with_scale_column=False):
    frame = pd.DataFrame({"Region": regions, "Year": range(len(regions))})
    if with_scale_column:
        frame["Region_Scale"] = ["World", "R5", "R10", "ISO3"]
    return frame


def _arrays(n):
    y_true = np.arange(n * 2, dtype=float).reshape(n, 2)
    return y_true, y_true + 1.0


def test_metrics_are_broken_down_by_scale_when_the_frame_is_given(run_root, caplog):
    y_true, y_pred = _arrays(len(REGIONS))

    with caplog.at_level(logging.INFO):
        save_metrics("xgb_01", y_true, y_pred, _frame())

    written = pd.read_csv(run_root / "metrics" / "performance.csv")
    assert list(written["Region Type"]) == ["Overall", "World", "R5", "R10", "ISO3"]

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "R10 regions" in logged and "ISO3 regions" in logged


def test_without_a_frame_only_the_overall_row_is_written(run_root):
    y_true, y_pred = _arrays(len(REGIONS))

    save_metrics("lstm_01", y_true, y_pred)

    written = pd.read_csv(run_root / "metrics" / "performance.csv")
    assert list(written["Region Type"]) == ["Overall"]


def test_a_frame_that_does_not_line_up_is_refused(run_root):
    """The sequence models score a horizon subset, not the whole test split."""
    y_true, y_pred = _arrays(len(REGIONS))

    with pytest.raises(ValueError, match="row for row"):
        save_metrics("tft_01", y_true, y_pred, _frame(REGIONS + ["R6WEU"]))


def test_the_overall_line_reports_its_sample_size(run_root, caplog):
    y_true, y_pred = _arrays(len(REGIONS))

    with caplog.at_level(logging.INFO):
        save_metrics("xgb_01", y_true, y_pred, _frame())

    overall = [r.getMessage() for r in caplog.records if "overall metrics" in r.getMessage()]
    assert "(8 samples)" in overall[0]


def test_large_and_small_magnitudes_both_stay_readable():
    """Absolute targets reach 1e16, where %.4f printed seventeen digits."""
    line = _metrics_line({
        "Mean Squared Error": 1.9429068216974588e16,
        "RMSE": 139388192.5307,
        "MAE": 48924432.1364,
        "R2 Score (per-target avg)": 0.9279,
        "R2 Score (pooled)": 0.9590,
        "Pearson Correlation": 0.9795,
    })

    assert "MSE=1.9429e+16" in line
    assert "RMSE=1.3939e+08" in line
    assert "R2_avg=0.9279" in line


def test_a_missing_correlation_does_not_break_the_line():
    line = _metrics_line({
        "Mean Squared Error": 1.0, "RMSE": 1.0, "MAE": 1.0,
        "R2 Score (per-target avg)": float("nan"),
        "R2 Score (pooled)": 0.5,
        "Pearson Correlation": float("nan"),
    })

    assert "R2_avg=nan" in line and "Pearson=nan" in line


def test_integer_coded_regions_are_refused_rather_than_mislabelled(caplog):
    """LSTM encodes Region for its embeddings; prefix rules would call it all ISO3."""
    encoded = pd.DataFrame({"Region": [0, 1, 2, 3]})

    with caplog.at_level(logging.WARNING):
        assert scale_of_frame(encoded) is None

    assert "integer-coded" in caplog.text


def test_a_persisted_scale_column_wins_over_encoded_labels():
    frame = _frame(regions=[0, 1, 2, 3], with_scale_column=True)

    assert list(scale_of_frame(frame)) == ["World", "R5", "R10", "ISO3"]
