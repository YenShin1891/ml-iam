"""Each model run takes its own AR6 category; runs without one keep their rows."""

import pandas as pd

from src.data.process_data import UNCATEGORISED, add_scenario_category, relabel_scenario_categories


def make_frame(runs):
    """runs: list of (model, scenario)."""
    return pd.DataFrame({
        "Model": [m for m, _ in runs],
        "Scenario": [s for _, s in runs],
        "Region": ["World"] * len(runs),
        "Variable": ["V"] * len(runs),
        "Unit": ["u"] * len(runs),
        "2020": range(len(runs)),
    })


# A run can be absent from the table altogether, which leaves the merge with
# no category.  "shared" is one scenario name run by two models that AR6 put
# in different categories.
CATEGORIES = pd.DataFrame({
    "Model": ["M", "M", "M", "Other"],
    "Scenario": ["assessed", "unassessed", "shared", "shared"],
    "Scenario_Category": ["C3", UNCATEGORISED, "C1", "C6"],
})


def test_a_run_missing_from_the_metadata_is_labelled_rather_than_dropped():
    out = add_scenario_category(make_frame([("M", "absent")]), CATEGORIES)

    assert list(out["Scenario"]) == ["absent"]
    assert list(out["Scenario_Category"]) == [UNCATEGORISED]


def test_known_categories_survive_untouched():
    out = add_scenario_category(make_frame([("M", "assessed"), ("M", "unassessed")]), CATEGORIES)

    assert dict(zip(out["Scenario"], out["Scenario_Category"])) == {
        "assessed": "C3",
        "unassessed": UNCATEGORISED,
    }


def test_the_same_scenario_name_takes_each_models_own_category():
    out = add_scenario_category(make_frame([("M", "shared"), ("Other", "shared")]), CATEGORIES)

    assert dict(zip(out["Model"], out["Scenario_Category"])) == {"M": "C1", "Other": "C6"}


def test_a_name_the_table_lists_only_under_other_models_is_uncategorised():
    out = add_scenario_category(make_frame([("Third", "shared")]), CATEGORIES)

    assert list(out["Scenario_Category"]) == [UNCATEGORISED]


def test_no_row_is_lost():
    runs = [("M", "assessed"), ("M", "unassessed"), ("M", "shared"), ("Other", "shared"), ("M", "absent")]

    out = add_scenario_category(make_frame(runs), CATEGORIES)

    assert len(out) == len(runs)
    assert out["Scenario_Category"].notna().all()


# ── relabelling frames a run saved with per-name labels ──────────────────


def stored_frame():
    # Two rows per run, as a saved test frame has.  "Other"/"shared" carries
    # M's category, as the per-name table gave it; "absent" is unknown to the
    # table.  The index is not the default, as a filtered frame's is not.
    return pd.DataFrame({
        "Model": ["M", "M", "Other", "Other", "M", "M"],
        "Scenario": ["assessed", "assessed", "shared", "shared", "absent", "absent"],
        "Scenario_Category": ["C3", "C3", "C1", "C1", "C8", "C8"],
        "Year": [2020, 2030] * 3,
    }, index=[10, 11, 12, 13, 14, 15])


def test_relabelling_corrects_only_runs_the_table_lists():
    out, n_changed = relabel_scenario_categories(stored_frame(), CATEGORIES)

    assert n_changed == 1
    assert list(out.index) == [10, 11, 12, 13, 14, 15]
    assert list(out["Scenario_Category"]) == ["C3", "C3", "C6", "C6", "C8", "C8"]


def test_relabelling_leaves_the_stored_frame_alone():
    stored = stored_frame()
    relabel_scenario_categories(stored, CATEGORIES)

    assert list(stored["Scenario_Category"]) == ["C3", "C3", "C1", "C1", "C8", "C8"]


def test_relabelling_a_corrected_frame_changes_nothing():
    out, _ = relabel_scenario_categories(stored_frame(), CATEGORIES)
    again, n_again = relabel_scenario_categories(out, CATEGORIES)

    assert n_again == 0
    assert again.equals(out)
