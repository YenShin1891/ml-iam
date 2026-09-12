"""The what-if figures, and where they are saved."""

import json

import matplotlib
import pandas as pd
import pytest
from matplotlib.collections import PolyCollection

matplotlib.use("Agg")

import src.visualization.trajectories as trajectories
import src.visualization.whatif as whatif_plots
from src.inference.whatif import WhatifResult, build_lever_specs, lever_bands, ramp_anchor_values, resolve_anchor_years
from src.visualization.trajectories import get_saved_plots_metadata
from src.visualization.whatif import plot_lever_overlay, plot_whatif_grid, save_whatif_outputs

YEARS = list(range(2005, 2101, 5))
TARGETS = ["A", "B"]
HISTORY = 3


def _rows(scale=1.0):
    frame = pd.DataFrame({
        "Model": "M1", "Scenario": "S1", "Region": "World", "Model_Family": "FAM",
        "Scenario_Category": "C3", "Year": YEARS, "Step": range(len(YEARS)),
    })
    frame["feat_a"] = [scale * (10.0 + i) for i in range(len(YEARS))]
    frame["feat_b"] = [scale * (20.0 + i) for i in range(len(YEARS))]
    frame["feat_a_is_missing"] = 0.0
    frame["feat_b_is_missing"] = 0.0
    for target in TARGETS:
        frame[target] = [float(i) for i in range(len(YEARS))]
        frame[f"{target}__observed"] = 1.0
    return frame


@pytest.fixture
def specs():
    population = pd.concat([_rows(scale=float(s)).assign(Scenario=f"S{s}") for s in range(1, 8)], ignore_index=True)
    bands = lever_bands(population, ["feat_a", "feat_b"])
    rows = _rows()
    anchors = resolve_anchor_years(YEARS, YEARS[:HISTORY])
    return build_lever_specs(rows, bands, ["feat_a", "feat_b"], HISTORY, anchors)


@pytest.fixture
def result():
    rows = _rows()
    years = pd.Index(YEARS, name="Year")
    iam = rows.set_index(years)[TARGETS].astype(float)
    predicted = iam.iloc[HISTORY:]
    return WhatifResult(
        run_id="tft_test", region="World", model="M1", scenario="S1", split="train", category="C3",
        model_family="FAM", years=YEARS, history_years=YEARS[:HISTORY],
        inputs_baseline=rows.set_index(years)[["feat_a", "feat_b"]],
        inputs_edited=rows.set_index(years)[["feat_a", "feat_b"]] * 1.2,
        iam=iam, pred_baseline=predicted + 0.1, pred_edited=predicted + 0.5,
        edits={"feat_a": {2100: 30.0}}, coverage=(1, 1),
        baseline_fit={"Count": 34, "R2": 0.99, "RMSE": 0.1, "MAE": 0.1},
    )


def test_the_grid_draws_iam_and_both_forecasts_per_target(result):
    fig = plot_whatif_grid(result)
    axes = fig.axes

    assert len(axes) == 9  # the dashboard's 3x3 grid, spare panels switched off
    assert [len(ax.lines) for ax in axes[:2]] == [3, 3]
    assert not axes[2].axison
    labels = [text.get_text() for text in axes[0].get_legend().get_texts()]
    assert "Emulator, edited inputs" in labels and "IAM (baseline scenario)" in labels
    matplotlib.pyplot.close(fig)


def test_the_grid_can_shade_the_ar6_range(result):
    band = pd.DataFrame({"target": ["A"] * 3, "Year": [2020, 2050, 2100], "lo": [0.0, 1.0, 2.0], "hi": [5.0, 6.0, 7.0], "n": [9] * 3})

    fig = plot_whatif_grid(result, ar6_band=band)

    assert any(isinstance(c, PolyCollection) for c in fig.axes[0].collections)
    assert not any(isinstance(c, PolyCollection) for c in fig.axes[1].collections)
    matplotlib.pyplot.close(fig)


@pytest.mark.parametrize("mode", ["band", "ratio"])
def test_the_overlay_draws_baseline_and_edited_paths_per_lever(specs, mode):
    feat_a = next(s for s in specs if s.feature == "feat_a")
    edits = {"feat_a": ramp_anchor_values(feat_a, 2.0)}

    fig = plot_lever_overlay(specs, edits, mode=mode, features=["feat_a", "feat_b"])
    ax = fig.axes[0]

    assert len(ax.lines) >= 4  # dotted + solid per lever
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert {"feat_a", "feat_b", "baseline", "edited"} <= set(labels)
    matplotlib.pyplot.close(fig)


def test_the_overlay_without_levers_says_so(specs):
    fig = plot_lever_overlay(specs, {}, features=[])

    assert not fig.axes[0].axison
    matplotlib.pyplot.close(fig)


def test_saved_outputs_appear_in_the_dashboard_listing(tmp_path, monkeypatch, result):
    monkeypatch.setattr(whatif_plots, "get_run_root", lambda _run_id: str(tmp_path))
    monkeypatch.setattr(trajectories, "get_run_root", lambda _run_id: str(tmp_path))
    fig = plot_whatif_grid(result)

    png_path, metadata_path = save_whatif_outputs(
        "tft_test", fig, {"timestamp": "2026-09-12T10:00:00", "regions": ["World"]}, timestamp="20260912_100000"
    )
    matplotlib.pyplot.close(fig)

    assert png_path.endswith("whatif_20260912_100000.png")
    assert json.load(open(metadata_path))["regions"] == ["World"]
    listed = get_saved_plots_metadata("tft_test")
    assert [entry["timestamp"] for entry in listed] == ["20260912_100000"]
    assert listed[0]["plot_path"] == png_path
