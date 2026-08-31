"""Recovering a pre-splits.parquet run's test groups from its saved test data.

The sequence models save their test frame after encoding categoricals for
their embeddings, so Region comes back as integer codes.  Merging those
against the labels in the processed data raised, which blocked replotting
lstm_87; matching them as strings would have been worse, finding no test
groups at all and silently re-splitting a finished run.
"""

import numpy as np
import pandas as pd
import pytest

import src.utils.run_store as run_store_module
from configs.data import INDEX_COLUMNS
from src.utils.run_store import RunStore

REGIONS = ["World", "R10AFRICA", "R5ASIA", "KOR"]


@pytest.fixture
def data():
    rows = []
    for group in range(20):
        for year in range(4):
            rows.append({
                "Model": f"M{group}",
                "Scenario": f"S{group}",
                "Region": REGIONS[group % len(REGIONS)],
                "Year": 2020 + year,
                "value": float(group),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store_module, "get_run_root", lambda _run_id: str(tmp_path))
    return RunStore("lstm_01")


def _encoded_test_frame(data, vocabulary, groups):
    """A test frame saved the way the sequence models save theirs."""
    frame = data[data["Model"].isin(groups)].copy()
    frame["Region"] = frame["Region"].map(lambda r: vocabulary.index(r)).astype("int64")
    return frame


def _test_groups(assignment):
    rows = assignment[assignment["split"] == "test"][list(INDEX_COLUMNS)]
    return set(map(tuple, rows.itertuples(index=False, name=None)))


def test_encoded_regions_are_decoded_to_recover_the_original_test_groups(store, data):
    vocabulary = sorted(data["Region"].unique())
    store.save_categories({"Region": vocabulary})
    evaluated = ["M3", "M7", "M11"]
    store.save_test_data(_encoded_test_frame(data, vocabulary, evaluated), np.zeros(1))

    assignment = store.splits_for(data)

    assert _test_groups(assignment) == {
        (model, model.replace("M", "S"), REGIONS[int(model[1:]) % len(REGIONS)])
        for model in evaluated
    }


def test_label_frames_are_left_alone(store, data):
    """XGB saves its test frame with regions still spelled out."""
    evaluated = ["M2", "M5"]
    store.save_test_data(data[data["Model"].isin(evaluated)].copy(), np.zeros(1))

    assignment = store.splits_for(data)

    assert {group[0] for group in _test_groups(assignment)} == set(evaluated)


def test_every_group_lands_in_exactly_one_split(store, data):
    vocabulary = sorted(data["Region"].unique())
    store.save_categories({"Region": vocabulary})
    store.save_test_data(_encoded_test_frame(data, vocabulary, ["M1"]), np.zeros(1))

    assignment = store.splits_for(data)

    assert len(assignment) == data[list(INDEX_COLUMNS)].drop_duplicates().shape[0]
    assert not assignment[list(INDEX_COLUMNS)].duplicated().any()


def test_a_run_without_its_own_vocabulary_still_recovers(store, data):
    """The vocabulary is rebuilt from the same data that produced the codes.

    lstm_87 had no categories.json and was recovered exactly this way.
    """
    vocabulary = sorted(data["Region"].unique())
    store.save_test_data(_encoded_test_frame(data, vocabulary, ["M4"]), np.zeros(1))

    assert not store.has_categories()
    assert {group[0] for group in _test_groups(store.splits_for(data))} == {"M4"}


def test_undecodable_codes_are_refused_rather_than_re_split(store, data, monkeypatch):
    """Silently re-splitting would score a run on groups it trained on."""
    vocabulary = sorted(data["Region"].unique())
    store.save_test_data(_encoded_test_frame(data, vocabulary, ["M4"]), np.zeros(1))
    monkeypatch.setattr(type(store), "categories_for", lambda self, data: {})

    with pytest.raises(ValueError, match="cannot be recovered"):
        store.splits_for(data)
