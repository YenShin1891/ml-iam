"""Scenarios the AR6 metadata cannot categorise keep their rows."""

import pandas as pd

from src.data.process_data import UNCATEGORISED, add_scenario_category


def make_frame(scenarios):
    return pd.DataFrame({
        "Model": ["M"] * len(scenarios),
        "Scenario": scenarios,
        "Region": ["World"] * len(scenarios),
        "Variable": ["V"] * len(scenarios),
        "Unit": ["u"] * len(scenarios),
        "2020": range(len(scenarios)),
    })


# "#N/A" in the metadata CSV reads back as NaN, and a scenario can also be
# absent from the CSV altogether; both leave the merge with no category.
CATEGORIES = pd.DataFrame({
    "Scenario": ["assessed", "unassessed", "lookup_failed"],
    "Scenario_Category": ["C3", UNCATEGORISED, None],
})


def test_a_failed_lookup_is_labelled_rather_than_dropped():
    out = add_scenario_category(make_frame(["lookup_failed"]), CATEGORIES)

    assert list(out["Scenario"]) == ["lookup_failed"]
    assert list(out["Scenario_Category"]) == [UNCATEGORISED]


def test_a_scenario_missing_from_the_metadata_is_labelled_too():
    out = add_scenario_category(make_frame(["absent"]), CATEGORIES)

    assert list(out["Scenario_Category"]) == [UNCATEGORISED]


def test_known_categories_survive_untouched():
    out = add_scenario_category(make_frame(["assessed", "unassessed"]), CATEGORIES)

    assert dict(zip(out["Scenario"], out["Scenario_Category"])) == {
        "assessed": "C3",
        "unassessed": UNCATEGORISED,
    }


def test_no_row_is_lost():
    scenarios = ["assessed", "unassessed", "lookup_failed", "absent"]

    out = add_scenario_category(make_frame(scenarios), CATEGORIES)

    assert len(out) == len(scenarios)
    assert out["Scenario_Category"].notna().all()
