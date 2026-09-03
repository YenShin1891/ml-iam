"""Unit normalization: aliases relabel, conversions rescale."""

import pandas as pd
import pytest

from src.data.process_data import UNIT_ALIASES, resolve_units


def make_frame(rows):
    """rows: list of (variable, unit, value_2020)."""
    return pd.DataFrame({
        "Model": ["M"] * len(rows),
        "Scenario": [f"S{i}" for i in range(len(rows))],
        "Region": ["World"] * len(rows),
        "Variable": [r[0] for r in rows],
        "Unit": [r[1] for r in rows],
        "2020": [r[2] for r in rows],
    })


def test_capitalisation_variants_collapse_onto_one_spelling():
    # resolve_units strips Unit from the frame and hands it back in the table.
    _, unit_table = resolve_units(make_frame([
        ("Population", "million", 7000.0),
        ("Population", "Million", 7000.0),
    ]))

    assert list(unit_table["Unit"]) == ["million", "million"]


def test_international_dollars_are_relabelled_not_rescaled():
    df, unit_table = resolve_units(make_frame([
        ("GDP|PPP", "billion US$2010/yr", 111_445.0),
        ("GDP|PPP", "billion Int$2010/yr", 114_770.0),
    ]))

    assert list(unit_table["Unit"]) == ["billion US$2010/yr"] * 2
    # The values are already the same quantity; an alias must not touch them.
    assert list(df["2020"]) == [111_445.0, 114_770.0]


@pytest.mark.parametrize("unit, expected_unit, factor", [
    ("EJ/yr", "PJ/yr", 1000),
    ("Million tkm", "bn tkm/yr", 0.001),
    ("Million pkm", "bn pkm/yr", 0.001),
])
def test_real_conversions_still_rescale(unit, expected_unit, factor):
    df, unit_table = resolve_units(make_frame([("V", unit, 2.0)]))

    assert list(unit_table["Unit"]) == [expected_unit]
    assert list(df["2020"]) == [2.0 * factor]


def test_the_alias_map_does_not_swallow_the_transport_units():
    # "Million tkm" and "Million pkm" begin with the same word as the
    # Population alias; a case fold would break their exact-match conversion.
    assert "Million tkm" not in UNIT_ALIASES
    assert "Million pkm" not in UNIT_ALIASES


def test_a_genuinely_mixed_variable_still_warns(caplog):
    df = make_frame([("V", "PJ/yr", 1.0), ("V", "million", 1.0)])

    with caplog.at_level("WARNING"):
        resolve_units(df)

    assert "more than one unit" in caplog.text


def test_aliased_variables_no_longer_warn(caplog):
    df = make_frame([
        ("GDP|PPP", "billion US$2010/yr", 1.0),
        ("GDP|PPP", "billion Int$2010/yr", 1.0),
        ("Population", "million", 1.0),
        ("Population", "Million", 1.0),
    ])

    with caplog.at_level("WARNING"):
        resolve_units(df)

    assert "more than one unit" not in caplog.text
