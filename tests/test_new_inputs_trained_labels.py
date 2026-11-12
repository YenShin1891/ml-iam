"""New scenarios may only use labels the run fitted an embedding for.

The vocabulary is wider than the training set: it lists regions whose series
were all dropped before training, and a region can sit wholly in validation.
"""

import pandas as pd
import pytest

from src.inference.new_inputs import InputError, _check_vocabulary, _usable, trained_labels

CATEGORIES = {
    "Region": ["World", "KOR", "DEU", "R6ROWO"],
    "Model_Family": ["MESSAGE", "PyPSA"],
}
ASSIGNMENT = pd.DataFrame(
    {
        "Model": ["MESSAGE-GLOBIOM 1.0", "MESSAGE-GLOBIOM 1.0", "MESSAGEix-GLOBIOM_1.1"],
        "Scenario": ["a", "b", "c"],
        "Region": ["World", "KOR", "R6ROWO"],
        "split": ["train", "train", "val"],
    }
)


def rows(region="World", family="MESSAGE"):
    return pd.DataFrame({"Region": [region], "Model_Family": [family]})


def test_labels_are_read_from_the_split_table():
    seen = trained_labels(ASSIGNMENT)

    assert seen["Region"] == {"any": {"World", "KOR", "R6ROWO"}, "train": {"World", "KOR"}}
    assert seen["Model_Family"]["train"] == {"MESSAGE"}


def test_a_trained_label_passes_silently():
    notes = []
    _check_vocabulary(rows("KOR"), CATEGORIES, ASSIGNMENT, notes)
    assert notes == []


def test_a_region_no_series_carries_is_refused():
    with pytest.raises(InputError, match=r"Region value\(s\) \['DEU'\].*never fitted"):
        _check_vocabulary(rows("DEU"), CATEGORIES, ASSIGNMENT, [])


def test_a_model_family_no_series_carries_is_refused():
    with pytest.raises(InputError, match=r"Model_Family value\(s\) \['PyPSA'\]"):
        _check_vocabulary(rows(family="PyPSA"), CATEGORIES, ASSIGNMENT, [])


def test_a_region_held_out_of_training_is_emulated_with_a_warning():
    notes = []
    _check_vocabulary(rows("R6ROWO"), CATEGORIES, ASSIGNMENT, notes)

    assert len(notes) == 1
    assert notes[0].startswith("WARNING") and "R6ROWO" in notes[0]


def test_a_label_outside_the_vocabulary_is_still_refused():
    with pytest.raises(InputError, match="not in the run's vocabulary"):
        _check_vocabulary(rows("ATL"), CATEGORIES, ASSIGNMENT, [])


def test_without_a_split_table_only_the_vocabulary_is_checked():
    _check_vocabulary(rows("DEU"), CATEGORIES)


def test_the_described_labels_are_the_accepted_ones():
    seen = trained_labels(ASSIGNMENT)

    assert _usable(CATEGORIES, seen, "Region") == ["World", "KOR", "R6ROWO"]
    assert _usable(CATEGORIES, seen, "Model_Family") == ["MESSAGE"]
    assert _usable(CATEGORIES, {}, "Region") == CATEGORIES["Region"]
