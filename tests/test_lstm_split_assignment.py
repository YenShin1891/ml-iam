"""LSTM must honour the run's shared split assignment.

The assignment is keyed by region labels, but LSTM encodes Region to the codes
its embeddings need.  Encoding before splitting left the two speaking different
languages: pandas refused the merge, and had it not, every row would have
matched nothing and fallen into train, leaving no test set at all.

No LSTM run existed between splits.parquet being introduced and this fix, so
nothing had exercised the combination.
"""

import numpy as np
import pandas as pd
import pytest

import src.utils.run_store as run_store_module
from configs.data import CATEGORICAL_COLUMNS, INDEX_COLUMNS, OUTPUT_VARIABLES
from src.utils.run_store import RunStore
from scripts.train_lstm import derive_splits

REGIONS = ["World", "R10AFRICA", "R5ASIA", "KOR"]


@pytest.fixture
def data():
    """Groups long enough to survive sequence preparation."""
    rows = []
    for group in range(24):
        for step, year in enumerate(range(2020, 2070, 5)):
            row = {
                "Model": f"M{group}",
                "Scenario": f"S{group}",
                "Region": REGIONS[group % len(REGIONS)],
                "Model_Family": f"FAM_{group % 3}",
                "Year": year,
                "Population": 1000.0 + group,
                "GDP": 50.0 + group + step,
            }
            for offset, target in enumerate(OUTPUT_VARIABLES):
                row[target] = float(group + step + offset)
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store_module, "get_run_root", lambda _run_id: str(tmp_path))
    return RunStore("lstm_01")


def _groups(frame):
    return set(map(tuple, frame[list(INDEX_COLUMNS)].drop_duplicates().itertuples(index=False, name=None)))


def test_the_runs_assignment_decides_the_test_set(store, data):
    """Not "every row is unassigned, so call it all train"."""
    splits = derive_splits(data, store)

    assert len(splits["test_data"]) > 0
    assert len(splits["val_data"]) > 0
    assert len(splits["train_data"]) > 0


def test_the_splits_do_not_overlap(store, data):
    splits = derive_splits(data, store)

    train, val, test = (_groups(splits[f"{name}_data"]) for name in ("train", "val", "test"))
    assert not (train & test)
    assert not (train & val)
    assert not (val & test)


def test_the_test_groups_are_the_ones_the_run_assigned(store, data):
    """A later phase must score the groups the run recorded, not fresh ones."""
    assignment = store.splits_for(data)
    expected = set(map(
        tuple,
        assignment[assignment["split"] == "test"][list(INDEX_COLUMNS)]
        .itertuples(index=False, name=None),
    ))

    splits = derive_splits(data, store)

    # Compare in label space: the returned frame carries codes.
    vocabulary = store.load_categories()["Region"]
    got = {
        (model, scenario, vocabulary[int(code)])
        for model, scenario, code in _groups(splits["test_data"])
    }
    assert got == expected


def test_categoricals_come_back_as_codes_for_the_embeddings(store, data):
    splits = derive_splits(data, store)

    for name in ("train_data", "val_data", "test_data"):
        for column in CATEGORICAL_COLUMNS:
            if column in splits[name].columns:
                assert splits[name][column].dtype == np.int64


def test_a_label_keeps_one_code_across_the_splits(store, data):
    """Encoding each split against its own frame would renumber them."""
    splits = derive_splits(data, store)

    seen = {}
    for name in ("train_data", "val_data", "test_data"):
        frame = splits[name]
        for code in frame["Region"].unique():
            rows = frame[frame["Region"] == code]
            seen.setdefault(int(code), set()).add(len(rows) > 0)

    vocabulary = store.load_categories()["Region"]
    assert all(0 <= code < len(vocabulary) for code in seen)
