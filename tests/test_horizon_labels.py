"""Saved TFT horizon frames get the labels the dashboard filters read."""

import pandas as pd

from src.visualization.helpers import backfill_group_labels

LABELS = ["Scenario_Category", "Model_Family"]


def stored_test_data():
    return pd.DataFrame({
        "Model": ["A", "A", "B", "B"],
        "Scenario": ["s"] * 4,
        "Region": ["World"] * 4,
        "Year": [2020, 2030, 2020, 2030],
        "Scenario_Category": ["C3", "C3", "C6", "C6"],
        "Model_Family": ["AIM", "AIM", "GCAM", "GCAM"],
    })


def horizon():
    # Other years than test_data, B before A, and a non-default index.
    return pd.DataFrame({
        "Model": ["B", "B", "A"],
        "Scenario": ["s"] * 3,
        "Region": ["World"] * 3,
        "Year": [2030, 2040, 2040],
    }, index=[7, 8, 9])


def test_labels_are_copied_per_group_in_row_order():
    out, filled = backfill_group_labels(horizon(), stored_test_data(), LABELS)

    assert filled == LABELS
    assert list(out.index) == [7, 8, 9]
    assert list(out["Scenario_Category"]) == ["C6", "C6", "C3"]
    assert list(out["Model_Family"]) == ["GCAM", "GCAM", "AIM"]
    assert list(out["Year"]) == [2030, 2040, 2040]


def test_columns_already_present_are_kept():
    out, filled = backfill_group_labels(horizon().assign(Scenario_Category="kept"), stored_test_data(), LABELS)

    assert filled == ["Model_Family"]
    assert list(out["Scenario_Category"]) == ["kept"] * 3


def test_a_label_test_data_cannot_supply_is_left_out():
    out, filled = backfill_group_labels(horizon(), stored_test_data().drop(columns=["Model_Family"]), LABELS)

    assert filled == ["Scenario_Category"]
    assert "Model_Family" not in out.columns


def test_frames_without_the_keys_are_returned_as_they_are():
    frame = horizon().drop(columns=["Region"])

    out, filled = backfill_group_labels(frame, stored_test_data(), LABELS)

    assert filled == []
    assert out is frame
