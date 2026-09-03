"""LSTM SHAP must cover the requested region, like XGB and TFT do.

The LSTM frame carries Region as the integer codes its embeddings were fit on,
so the "R10" prefix matched nothing and the filter fell through: lstm_87's SHAP
plots were computed over every region, while the other two models' were R10.
"""

import logging

import pandas as pd
import pytest

from configs.data import CATEGORICAL_COLUMNS
from src.data.preprocess import (
    build_categorical_vocabularies,
    decode_categorical_column,
    encode_categorical_columns,
)
from src.visualization.helpers import filter_index_frame_by_region
from scripts.train_lstm import _region_labels

REGIONS = ["R10AFRICA", "R10EUROPE", "R5ASIA", "World", "KOR"]


@pytest.fixture
def frame():
    return pd.DataFrame({
        "Region": REGIONS,
        "Model_Family": ["FAM"] * len(REGIONS),
        "Year": range(len(REGIONS)),
    })


def test_decoding_inverts_the_encoding(frame):
    categories = build_categorical_vocabularies(frame)
    encoded = encode_categorical_columns(frame.copy(), CATEGORICAL_COLUMNS, categories)

    assert list(decode_categorical_column(encoded["Region"], categories["Region"])) == REGIONS


def test_codes_outside_the_vocabulary_decode_to_nothing():
    """encode_categorical_columns writes -1 for unseen labels."""
    decoded = decode_categorical_column(pd.Series([-1, 0, 99]), ["World", "KOR"])

    assert list(decoded) == [None, "World", None]


def test_encoded_regions_are_decoded_for_the_filter(frame):
    categories = build_categorical_vocabularies(frame)
    encoded = encode_categorical_columns(frame.copy(), CATEGORICAL_COLUMNS, categories)

    labels = _region_labels(encoded, categories)
    filtered, _, pre, post, matched, mode = filter_index_frame_by_region(
        encoded, "R10", region_series=labels,
    )

    assert mode == "prefix"
    assert (pre, post) == (5, 2)
    assert matched == ["R10AFRICA", "R10EUROPE"]


def test_a_frame_that_kept_its_labels_is_used_as_is(frame):
    assert list(_region_labels(frame, categories=None)) == REGIONS


def test_without_a_vocabulary_no_labels_are_invented(frame, caplog):
    """Guessing here would filter against codes and silently keep every row."""
    categories = build_categorical_vocabularies(frame)
    encoded = encode_categorical_columns(frame.copy(), CATEGORICAL_COLUMNS, categories)

    with caplog.at_level(logging.WARNING):
        assert _region_labels(encoded, categories={}) is None

    assert "no Region vocabulary" in caplog.text


def test_plots_are_skipped_rather_than_drawn_over_the_wrong_regions(tmp_path, monkeypatch, caplog, frame):
    """A filter that matches nothing must not fall through to every region."""
    from src.visualization import shap_nn

    monkeypatch.setattr(shap_nn, "get_run_root", lambda _run_id: str(tmp_path))
    checkpoint = tmp_path / "final"
    checkpoint.mkdir()
    (checkpoint / "best.ckpt").write_bytes(b"")

    drawn = []
    monkeypatch.setattr(shap_nn, "get_lstm_shap_values",
                        lambda *a, **k: drawn.append(a) or (None, None, None, None))

    with caplog.at_level(logging.ERROR):
        shap_nn.plot_lstm_shap("lstm_01", frame, ["Year"], ["A"], region="NOSUCHREGION")

    assert drawn == []
    assert "matched no rows" in caplog.text
