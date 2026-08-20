"""Regression tests for target masking in prepare_data."""

import numpy as np
import pytest

import configs.data as data_config
from src.data.preprocess import prepare_data, sanitize_target_scaler

FEATURES = ["feat", "Region", "Model_Family"]


@pytest.fixture(autouse=True)
def _reset_region_categories():
    data_config.REGION_CATEGORIES.clear()
    data_config.REGION_CODE_TO_LABEL.clear()
    yield
    data_config.REGION_CATEGORIES.clear()
    data_config.REGION_CODE_TO_LABEL.clear()


@pytest.fixture
def keep_partial(monkeypatch):
    monkeypatch.setattr(data_config, "KEEP_PARTIAL_TARGETS", True)


PREPARE_DATA_FIELDS = [
    "X_train", "y_train", "idx_train",
    "X_val", "y_val", "idx_val",
    "X_test", "y_test", "test_data",
    "x_scaler", "y_scaler",
    "train_groups", "val_groups",
    "obs_train", "obs_val", "obs_test",
]


def _run(prepared, targets):
    """Call prepare_data and label its (long) return tuple."""
    result = prepare_data(prepared, targets, FEATURES)
    assert len(result) == len(PREPARE_DATA_FIELDS)
    return dict(zip(PREPARE_DATA_FIELDS, result))


# ── unobserved targets stay unobserved ────────────────────────────────────


def test_unobserved_targets_are_nan_not_zero(prepared_frame, targets, keep_partial):
    """The whole per-target masking design depends on NaN surviving scaling."""
    out = _run(prepared_frame, targets)

    for split in ("train", "val", "test"):
        y, obs = out[f"y_{split}"], out[f"obs_{split}"]
        assert (obs == 0).any(), f"{split} split has nothing unobserved to check"
        assert np.isnan(y[obs == 0]).all(), "unobserved targets must not reach the model"
        assert np.isfinite(y[obs == 1]).all()


def test_target_scaler_ignores_unobserved_values(prepared_frame, targets, keep_partial):
    """Scaler statistics must describe observed values, not the zero fill."""
    out = _run(prepared_frame, targets)
    y_scaler = out["y_scaler"]

    # Recompute the expected statistics straight from the observed rows.
    train_keys = set(map(tuple, out["idx_train"][["Model", "Scenario"]].values))
    train_rows = prepared_frame[
        [tuple(k) in train_keys for k in prepared_frame[["Model", "Scenario"]].values]
    ]
    observed_b = train_rows.loc[train_rows["B__observed"] == 1, "B"]

    assert y_scaler.mean_[1] == pytest.approx(observed_b.mean())
    assert y_scaler.scale_[1] == pytest.approx(observed_b.std(ddof=0))
    # The zero fill would have dragged the mean toward zero.
    assert y_scaler.mean_[1] > observed_b.mean() / 2


def test_complete_targets_are_untouched(prepared_frame, targets, monkeypatch):
    """With KEEP_PARTIAL_TARGETS=False the consumer cannot take NaN."""
    monkeypatch.setattr(data_config, "KEEP_PARTIAL_TARGETS", False)
    complete = prepared_frame.dropna(subset=targets).reset_index(drop=True)

    out = _run(complete, targets)

    for split in ("train", "val", "test"):
        assert np.isfinite(out[f"y_{split}"]).all()


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
def test_sanitize_target_scaler_repairs_all_nan_target():
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(np.array([[1.0, np.nan], [3.0, np.nan]]))

    assert sanitize_target_scaler(scaler, ["A", "B"]) is True
    assert scaler.mean_[1] == 0.0 and scaler.scale_[1] == 1.0
    assert np.isfinite(scaler.transform(np.array([[2.0, 5.0]]))).all()
