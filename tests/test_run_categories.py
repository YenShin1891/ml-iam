"""Category vocabularies must survive into every later phase and process."""

import json

import pandas as pd
import pytest

import configs.data as data_config
import src.utils.run_store as run_store_module
from src.data.preprocess import resolve_categorical_vocabularies
from src.utils.run_store import RunStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store_module, "get_run_root", lambda _run_id: str(tmp_path))
    return RunStore("xgb_01"), tmp_path


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "Region": ["World", "R10_A", "R10_B", "World"],
            "Model_Family": ["FAM_B", "FAM_A", "FAM_A", "FAM_C"],
        }
    )


# ── persistence ───────────────────────────────────────────────────────────


def test_first_call_derives_and_saves(store, data):
    run_store, root = store

    categories = run_store.categories_for(data)

    assert categories["Region"] == ["R10_A", "R10_B", "World"]
    assert categories["Model_Family"] == ["FAM_A", "FAM_B", "FAM_C"]
    saved = json.loads((root / "artifacts" / "categories.json").read_text())
    assert saved == categories


def test_later_phases_reuse_the_saved_codes(store, data):
    """A later phase seeing fewer labels must not renumber the survivors."""
    run_store, _ = store
    first = run_store.categories_for(data)

    subset = data[data["Model_Family"] != "FAM_A"]
    second = run_store.categories_for(subset)

    assert second == first


def test_a_new_process_reads_the_saved_vocabulary(store, data, monkeypatch):
    """The dashboard's process never runs preprocessing."""
    run_store, root = store
    expected = run_store.categories_for(data)

    data_config.REGION_CATEGORIES.clear()  # as in a fresh interpreter
    fresh = RunStore("xgb_01")

    assert fresh.has_categories()
    assert fresh.load_categories() == expected


def test_unseen_labels_are_appended_not_inserted(store, data, caplog):
    run_store, _ = store
    run_store.categories_for(data)

    extended = pd.concat(
        [data, pd.DataFrame({"Region": ["AAA"], "Model_Family": ["FAM_A"]})],
        ignore_index=True,
    )
    with caplog.at_level("WARNING"):
        categories = run_store.categories_for(extended)

    # "AAA" sorts first but must not take code 0 away from R10_A.
    assert categories["Region"] == ["R10_A", "R10_B", "World", "AAA"]
    assert "absent from the run's saved vocabulary" in caplog.text


def test_missing_categories_file_is_reported(store):
    run_store, _ = store

    assert run_store.has_categories() is False
    with pytest.raises(FileNotFoundError):
        run_store.load_categories()


# ── reconciliation ────────────────────────────────────────────────────────


def test_resolve_without_persisted_derives_from_data(data):
    assert resolve_categorical_vocabularies(data)["Model_Family"] == [
        "FAM_A", "FAM_B", "FAM_C",
    ]


def test_resolve_preserves_saved_order(data):
    persisted = {"Model_Family": ["FAM_C", "FAM_A", "FAM_B"], "Region": ["World"]}

    resolved = resolve_categorical_vocabularies(data, persisted)

    assert resolved["Model_Family"] == ["FAM_C", "FAM_A", "FAM_B"]
    assert resolved["Region"][0] == "World"


def test_resolve_keeps_columns_absent_from_current_data(data):
    persisted = {"Scenario_Category": ["C1", "C2"]}

    resolved = resolve_categorical_vocabularies(data, persisted)

    assert resolved["Scenario_Category"] == ["C1", "C2"]


# ── migration ─────────────────────────────────────────────────────────────


def test_pre_existing_runs_keep_their_lstm_vocabulary(store, data):
    """An old checkpoint's embeddings are indexed by the codes it trained with."""
    run_store, _ = store
    run_store.save_train_meta({"lstm_model_family_categories": ["FAM_C", "FAM_B", "FAM_A"]})

    categories = run_store.categories_for(data)

    assert categories["Model_Family"] == ["FAM_C", "FAM_B", "FAM_A"]


def test_pre_existing_runs_keep_their_xgb_vocabulary(store, data):
    run_store, _ = store
    run_store.save_train_meta(
        {"xgb_categories": {"Model_Family": ["FAM_B", "FAM_A", "FAM_C"],
                            "Region": ["World", "R10_A", "R10_B"]}}
    )

    categories = run_store.categories_for(data)

    assert categories["Model_Family"] == ["FAM_B", "FAM_A", "FAM_C"]
    assert categories["Region"] == ["World", "R10_A", "R10_B"]
