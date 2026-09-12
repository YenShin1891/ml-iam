"""performance.csv pools each scale; the per-region list sits beside it."""

import logging

import numpy as np
import pandas as pd
import pytest

import configs.data as data_config
from src.trainers import evaluation
from src.trainers.evaluation import by_region_filename, metrics_by_region, save_metrics


@pytest.fixture
def run_root(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "get_run_root", lambda _run_id: str(tmp_path))
    return tmp_path


# Two rows per region, finest scale first and out of alphabetical order, so
# the file has to put them in order itself.
REGIONS = ["KOR", "KOR", "R10AFRICA", "R10AFRICA", "USA", "USA", "World", "World", "R5ASIA", "R5ASIA"]


def _frame(regions=REGIONS):
    return pd.DataFrame({"Region": regions, "Year": range(len(regions))})


def _arrays(n):
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=(n, 2))
    return y_true, y_true + rng.normal(scale=0.1, size=(n, 2))


def test_one_row_per_region_coarsest_scale_first_then_alphabetical(run_root):
    y_true, y_pred = _arrays(len(REGIONS))

    save_metrics("xgb_01", y_true, y_pred, _frame())

    written = pd.read_csv(run_root / "metrics" / "performance_by_region.csv")
    assert list(written.columns[:3]) == ["Run ID", "Region", "Region Type"]
    assert list(written["Region"]) == ["World", "R5ASIA", "R10AFRICA", "KOR", "USA"]
    assert list(written["Region Type"]) == ["World", "R5", "R10", "ISO3", "ISO3"]
    assert list(written["Sample Size"]) == [4] * 5  # two rows of two targets each


def test_a_scale_with_one_region_scores_as_that_region(run_root):
    y_true, y_pred = _arrays(len(REGIONS))

    save_metrics("xgb_01", y_true, y_pred, _frame())

    by_scale = pd.read_csv(run_root / "metrics" / "performance.csv").set_index("Region Type")
    by_region = pd.read_csv(run_root / "metrics" / "performance_by_region.csv").set_index("Region")
    assert by_region.loc["World", "RMSE"] == pytest.approx(by_scale.loc["World", "RMSE"])
    assert by_region.loc["R5ASIA", "R2 Score (pooled)"] == pytest.approx(by_scale.loc["R5", "R2 Score (pooled)"])


def test_unobserved_elements_are_not_scored(run_root):
    y_true, y_pred = _arrays(len(REGIONS))
    observed = np.ones_like(y_true, dtype=bool)
    observed[:2, 1] = False  # KOR's second target is imputed

    table = metrics_by_region("xgb_01", y_true, y_pred, _frame(), observed_mask=observed)

    assert dict(zip(table["Region"], table["Sample Size"]))["KOR"] == 2


def test_the_file_follows_the_metrics_filename(run_root):
    y_true, y_pred = _arrays(len(REGIONS))

    save_metrics("xgb_01", y_true, y_pred, _frame(), metrics_filename="performance_train.csv")

    assert (run_root / "metrics" / "performance_train_by_region.csv").exists()
    assert by_region_filename("performance.csv") == "performance_by_region.csv"


def test_without_a_frame_no_region_file_is_written(run_root):
    y_true, y_pred = _arrays(4)

    save_metrics("lstm_01", y_true, y_pred)

    assert not (run_root / "metrics" / "performance_by_region.csv").exists()


# ── frames that carry Region as embedding codes ───────────────────────────


class _StoreWithVocabulary:
    def __init__(self, _run_id):
        pass

    def has_categories(self):
        return True

    def load_categories(self):
        return {"Region": ["World", "KOR"]}


def test_codes_are_named_through_the_runs_saved_vocabulary(monkeypatch):
    """LSTM frames carry Region as embedding codes; the list names the regions."""
    monkeypatch.setattr(evaluation, "RunStore", _StoreWithVocabulary)
    frame = pd.DataFrame({"Region": [1, 1, 0, 0], "Region_Scale": ["ISO3", "ISO3", "World", "World"]})
    y_true, y_pred = _arrays(4)

    table = metrics_by_region("lstm_01", y_true, y_pred, frame)

    assert list(table["Region"]) == ["World", "KOR"]
    assert list(table["Region Type"]) == ["World", "ISO3"]


def test_the_in_process_vocabulary_serves_a_run_that_saved_none(run_root):
    data_config.REGION_CODE_TO_LABEL.update({0: "World", 1: "KOR"})
    frame = pd.DataFrame({"Region": [0, 0, 1, 1]})
    y_true, y_pred = _arrays(4)

    table = metrics_by_region("lstm_01", y_true, y_pred, frame)

    assert list(table["Region"]) == ["World", "KOR"]
    assert list(table["Region Type"]) == ["World", "ISO3"]


def test_codes_nothing_can_decode_yield_no_list(run_root, caplog):
    frame = pd.DataFrame({"Region": [0, 0, 1, 1]})
    y_true, y_pred = _arrays(4)

    with caplog.at_level(logging.WARNING):
        assert metrics_by_region("lstm_01", y_true, y_pred, frame) is None

    assert "integer-coded" in caplog.text
    assert not (run_root / "metrics" / "performance_by_region.csv").exists()
