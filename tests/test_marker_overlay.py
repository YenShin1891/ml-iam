"""The marker run is drawn in the dataset's units, not the raw file's."""

import pandas as pd

from src.visualization.trajectories import MARKER_MODEL, MARKER_REGION, MARKER_SCENARIO, marker_frame


def raw_rows(rows):
    """rows: list of (model, variable, unit, value_2020, value_2030), as the raw World CSV has them."""
    return pd.DataFrame({
        "Model": [r[0] for r in rows],
        "Scenario": [MARKER_SCENARIO] * len(rows),
        "Region": [MARKER_REGION] * len(rows),
        "Variable": [r[1] for r in rows],
        "Unit": [r[2] for r in rows],
        "2020": [r[3] for r in rows],
        "2030": [r[4] for r in rows],
    })


def test_primary_energy_is_rescaled_from_ej_to_pj():
    frame = marker_frame(raw_rows([
        (MARKER_MODEL, "Primary Energy|Coal", "EJ/yr", 150.0, 100.0),
        (MARKER_MODEL, "Emissions|CO2", "Mt CO2/yr", 40000.0, 30000.0),
    ]))

    assert list(frame["Year"]) == [2020, 2030]
    assert list(frame["Primary Energy|Coal"]) == [150_000.0, 100_000.0]
    assert list(frame["Emissions|CO2"]) == [40000.0, 30000.0]


def test_other_runs_and_empty_years_are_left_out():
    frame = marker_frame(raw_rows([
        (MARKER_MODEL, "Primary Energy|Coal", "EJ/yr", 150.0, None),
        ("Other", "Primary Energy|Coal", "EJ/yr", 1.0, 1.0),
    ]))

    assert list(frame["Year"]) == [2020]
    assert list(frame["Primary Energy|Coal"]) == [150_000.0]


def test_no_marker_rows_gives_none():
    assert marker_frame(raw_rows([("Other", "Primary Energy|Coal", "EJ/yr", 1.0, 1.0)])) is None
