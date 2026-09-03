"""One split assignment per run, shared by every model."""

import numpy as np
import pandas as pd
import pytest

import src.utils.run_store as run_store_module
from configs.data import INDEX_COLUMNS
from src.data.preprocess import assign_group_splits, observed_group_keys, split_data
from src.utils.run_store import RunStore


@pytest.fixture
def data():
    """40 groups; the short ones are what a lag filter would drop."""
    rows = []
    for g in range(40):
        length = 2 if g % 5 == 0 else 6
        for year in range(length):
            rows.append(
                {
                    "Model": f"M{g}",
                    "Scenario": f"S{g}",
                    "Region": "World" if g % 2 else f"R10_{g}",
                    "Year": 2020 + year,
                    "value": float(g),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store_module, "get_run_root", lambda _run_id: str(tmp_path))
    return RunStore("xgb_01")


def _groups(frame):
    return set(map(tuple, frame[INDEX_COLUMNS].drop_duplicates().to_numpy()))


def _long_groups_only(data):
    """Stand-in for XGBoost's lag filtering."""
    sizes = data.groupby(INDEX_COLUMNS)["Year"].transform("size")
    return data[sizes > 2].reset_index(drop=True)


# ── assignment ────────────────────────────────────────────────────────────


def test_every_group_gets_exactly_one_split(data):
    assignment = assign_group_splits(data)

    assert len(assignment) == len(data[INDEX_COLUMNS].drop_duplicates())
    assert set(assignment["split"]) <= {"train", "val", "test"}
    assert assignment[INDEX_COLUMNS].duplicated().sum() == 0


def test_assignment_is_deterministic(data):
    first = assign_group_splits(data)
    second = assign_group_splits(data.sample(frac=1, random_state=7))

    pd.testing.assert_frame_equal(
        first.sort_values(INDEX_COLUMNS).reset_index(drop=True),
        second.sort_values(INDEX_COLUMNS).reset_index(drop=True),
    )


def test_split_sizes_follow_the_requested_fractions(data):
    assignment = assign_group_splits(data, test_size=0.25, val_size=0.25)
    counts = assignment["split"].value_counts()

    assert counts["test"] == 10 and counts["val"] == 10 and counts["train"] == 20


# ── the bug this fixes ────────────────────────────────────────────────────


def test_models_that_filter_differently_share_a_test_set(data):
    """XGBoost drops short groups; that must not move anyone else's test set."""
    assignment = assign_group_splits(data)

    _, _, seq_test = split_data(data, assignment=assignment)
    _, _, xgb_test = split_data(_long_groups_only(data), assignment=assignment)

    assert _groups(xgb_test) <= _groups(seq_test)
    # No group tests for one model while training another.
    xgb_train, xgb_val, _ = split_data(_long_groups_only(data), assignment=assignment)
    assert not _groups(seq_test) & (_groups(xgb_train) | _groups(xgb_val))


def test_without_a_shared_assignment_the_test_sets_diverge(data):
    """Guards the premise: deriving per model is what caused the drift."""
    _, _, seq_test = split_data(data)
    _, _, xgb_test = split_data(_long_groups_only(data))

    assert _groups(seq_test) != _groups(xgb_test)


def test_splits_are_disjoint_and_cover_every_row(data):
    assignment = assign_group_splits(data)

    train, val, test = split_data(data, assignment=assignment)

    assert len(train) + len(val) + len(test) == len(data)
    assert not _groups(train) & _groups(val)
    assert not _groups(train) & _groups(test)
    assert not _groups(val) & _groups(test)


def test_groups_missing_from_the_assignment_go_to_train(data, caplog):
    """An unknown group must never silently become test data."""
    assignment = assign_group_splits(data)
    extra = pd.DataFrame(
        [{"Model": "M_NEW", "Scenario": "S_NEW", "Region": "World",
          "Year": 2020, "value": 1.0}]
    )

    with caplog.at_level("WARNING"):
        train, val, test = split_data(pd.concat([data, extra], ignore_index=True),
                                      assignment=assignment)

    assert ("M_NEW", "S_NEW", "World") in _groups(train)
    assert "absent from the split assignment" in caplog.text


def test_empty_split_is_reported_not_crashed(data, caplog):
    """The old implementation raised 'No objects to concatenate' here."""
    assignment = assign_group_splits(data)
    assignment["split"] = "train"

    with caplog.at_level("WARNING"):
        train, val, test = split_data(data, assignment=assignment)

    assert len(train) == len(data)
    assert val.empty and test.empty
    assert list(val.columns) == list(data.columns)


# ── persistence ───────────────────────────────────────────────────────────


def test_assignment_is_saved_once_and_reused(store, data):
    first = store.splits_for(data)

    assert store.has_splits()
    # A later phase sees a filtered frame but must reuse the saved assignment.
    second = store.splits_for(_long_groups_only(data))

    pd.testing.assert_frame_equal(first, second)


def test_a_later_process_reads_the_saved_assignment(store, data, tmp_path, monkeypatch):
    saved = store.splits_for(data)

    monkeypatch.setattr(run_store_module, "get_run_root", lambda _run_id: str(tmp_path))
    pd.testing.assert_frame_equal(RunStore("xgb_01").load_splits(), saved)


# ── migration ─────────────────────────────────────────────────────────────


def test_a_pre_existing_run_keeps_the_test_set_it_was_scored_on(store, data, caplog):
    """Recomputing the split could move trained-on groups into test."""
    original_test_groups = {("M1", "S1", "World"), ("M3", "S3", "World")}
    test_rows = data[
        data.apply(lambda r: (r.Model, r.Scenario, r.Region) in original_test_groups, axis=1)
    ]
    store.save_test_data(test_rows, np.zeros((len(test_rows), 1)))

    with caplog.at_level("WARNING"):
        assignment = store.splits_for(data)

    recovered = set(
        map(tuple, assignment.loc[assignment["split"] == "test", INDEX_COLUMNS].to_numpy())
    )
    assert recovered == original_test_groups
    assert "predates splits.parquet" in caplog.text
    assert len(assignment) == len(data[INDEX_COLUMNS].drop_duplicates())


def test_a_fresh_run_uses_the_deterministic_assignment(store, data):
    assignment = store.splits_for(data)

    pd.testing.assert_frame_equal(
        assignment.sort_values(INDEX_COLUMNS).reset_index(drop=True),
        assign_group_splits(data).sort_values(INDEX_COLUMNS).reset_index(drop=True),
    )


# ── canonical group set ───────────────────────────────────────────────────


def test_groups_with_no_observed_target_are_excluded(data):
    """Every prepare_features_* drops them, so they must not take a slot."""
    from configs.data import OUTPUT_VARIABLES

    target = OUTPUT_VARIABLES[0]
    data = data.assign(**{target: 1.0})
    data.loc[data["Model"] == "M7", target] = np.nan

    keys = observed_group_keys(data)

    assert ("M7", "S7", "World") not in set(map(tuple, keys.to_numpy()))
    assert len(keys) == len(data[INDEX_COLUMNS].drop_duplicates()) - 1


def test_row_filtering_does_not_move_the_split(data):
    """The canonical set is why this refactor changes no existing result."""
    from configs.data import OUTPUT_VARIABLES

    data = data.assign(**{OUTPUT_VARIABLES[0]: 1.0})
    filtered = _long_groups_only(data)  # as a lag requirement would filter

    pd.testing.assert_frame_equal(
        assign_group_splits(data), assign_group_splits(pd.concat([filtered, data]))
    )


def test_falls_back_to_all_groups_without_target_columns(data):
    assert len(observed_group_keys(data)) == len(data[INDEX_COLUMNS].drop_duplicates())
