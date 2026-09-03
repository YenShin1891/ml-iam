"""Regression tests for target masking and categorical encoding in prepare_data."""

import numpy as np
import pandas as pd
import pytest

import configs.data as data_config
from src.data.preprocess import (
    build_categorical_vocabularies,
    encode_categorical_columns,
    prepare_data,
    sanitize_target_scaler,
)

FEATURES = ["feat", "Region", "Model_Family"]


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
    "categories",
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


# ── categorical codes agree across splits ─────────────────────────────────


def test_categorical_codes_agree_across_splits(prepared_frame, targets, keep_partial):
    """The same label must get the same code in train, val and test."""
    out = _run(prepared_frame, targets)

    label_of = {
        (row.Model, row.Scenario): {"Region": row.Region, "Model_Family": row.Model_Family}
        for row in prepared_frame.itertuples()
    }

    splits = (
        ("train", out["X_train"], out["idx_train"]),
        ("val", out["X_val"], out["idx_val"]),
        ("test", out["X_test"], out["X_test"]),
    )

    # prepare_data returns scaled features, so compare the encoded-then-scaled
    # value each label lands on: one label must mean one value everywhere.
    value_of_label = {}
    for name, X, index_columns in splits:
        keys = list(zip(index_columns["Model"], index_columns["Scenario"]))
        assert keys, f"{name} split is empty"
        for pos, key in enumerate(keys):
            for col in ("Region", "Model_Family"):
                label = label_of[key][col]
                value = float(X[col].iloc[pos])
                previous = value_of_label.setdefault((col, label), (name, value))
                assert previous[1] == pytest.approx(value), (
                    f"{col}={label!r} encodes differently in {previous[0]} and {name}"
                )

    # Distinct labels must not collapse onto the same code.
    for col in ("Region", "Model_Family"):
        values = [v for (c, _), (_, v) in value_of_label.items() if c == col]
        assert len(set(values)) == len(values)


def test_model_family_code_is_stable_when_a_split_misses_a_category(prepared_frame):
    """Per-frame .cat.codes shifts codes; the shared vocabulary must not."""
    full = prepared_frame[["Model_Family"]].copy()
    subset = prepared_frame[prepared_frame["Model_Family"] != "FAM_1"][
        ["Model_Family"]
    ].copy()

    vocab = build_categorical_vocabularies(full, ["Model_Family"])
    encoded_full = encode_categorical_columns(full.copy(), ["Model_Family"], vocab)
    encoded_subset = encode_categorical_columns(subset.copy(), ["Model_Family"], vocab)

    code_of = dict(zip(encoded_full["Model_Family"], full["Model_Family"]))
    for code, label in zip(encoded_subset["Model_Family"], subset["Model_Family"]):
        assert code_of[code] == label

    # Without the shared vocabulary the same label lands on a different code.
    naive = encode_categorical_columns(subset.copy(), ["Model_Family"])
    rare_shared = encoded_subset.loc[subset["Model_Family"] == "FAM_RARE", "Model_Family"].iloc[0]
    rare_naive = naive.loc[subset["Model_Family"] == "FAM_RARE", "Model_Family"].iloc[0]
    assert rare_shared != rare_naive


def test_unknown_category_is_encoded_as_minus_one(prepared_frame, caplog):
    vocab = build_categorical_vocabularies(prepared_frame, ["Model_Family"])
    unseen = pd.DataFrame({"Model_Family": ["FAM_UNSEEN"]})

    encoded = encode_categorical_columns(unseen, ["Model_Family"], vocab)

    assert encoded["Model_Family"].iloc[0] == -1
    assert "outside the training vocabulary" in caplog.text


def test_prepare_data_returns_the_vocabulary(prepared_frame, targets, keep_partial):
    categories = _run(prepared_frame, targets)["categories"]

    assert set(categories) == {"Region", "Model_Family"}
    assert categories["Model_Family"] == sorted(set(prepared_frame["Model_Family"]))
