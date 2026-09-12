"""The what-if view's data side: regions, baselines, bands, levers and results."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from src.inference.whatif import (
    BaselineCandidate,
    PreparedRun,
    apply_levers,
    ar6_target_bands,
    assemble_result,
    band_at,
    band_coverage,
    band_position,
    baseline_rows,
    build_lever_specs,
    candidate_baselines,
    choose_default_baseline,
    eligible_regions,
    export_frame,
    interpolate_lever_path,
    lever_bands,
    load_prepared_run,
    multiplier_from_anchors,
    preset_edits,
    ramp_anchor_values,
    region_metrics_from_predictions,
    resolve_anchor_years,
    result_metadata,
)

FEATURES = ["feat_a", "feat_b", "feat_c"]
TARGETS = ["A", "B"]
YEARS = list(range(2005, 2101, 5))  # 20 steps
HISTORY = 3


def trajectory(model, scenario, region, category="C3", split="train", *, n_steps=20, imputed=(), scale=1.0):
    rows = []
    for step, year in enumerate(YEARS[:n_steps]):
        row = {
            "Model": model, "Scenario": scenario, "Region": region, "Model_Family": "FAM",
            "Scenario_Category": category, "Region_Scale": "World" if region == "World" else "R5",
            "Year": year, "Step": step, "DeltaYears": 0 if step == 0 else 5, "split": split,
            "Population": 100.0,
        }
        for k, feature in enumerate(FEATURES):
            row[feature] = scale * (10.0 * (k + 1) + step)
            row[f"{feature}_is_missing"] = 1.0 if feature in imputed else 0.0
        for target in TARGETS:
            row[target] = float(step)
            row[f"{target}__observed"] = 1.0
        rows.append(row)
    return rows


def make_prepared(*groups) -> PreparedRun:
    frame = pd.DataFrame([row for group in groups for row in group])
    features = FEATURES + [f"{f}_is_missing" for f in FEATURES] + ["Region", "Model_Family"]
    return PreparedRun("tft_test", frame, features, list(FEATURES), list(TARGETS))


@pytest.fixture
def population():
    """Six World training scenarios whose inputs scale 1x..6x, plus one that imputes feat_a."""
    groups = [trajectory("M1", f"S{i}", "World", scale=float(i)) for i in range(1, 7)]
    groups.append(trajectory("M9", "S9", "World", scale=100.0, imputed=("feat_a",)))
    return make_prepared(*groups)


def specs_for(prepared, model="M1", scenario="S1"):
    rows = baseline_rows(prepared, "World", model, scenario)
    train = prepared.frame[(prepared.frame["Region"] == "World") & (prepared.frame["split"] == "train")]
    bands = lever_bands(train, prepared.raw_features)
    years = rows["Year"].tolist()
    anchors = resolve_anchor_years(years, years[:HISTORY])
    return rows, build_lever_specs(rows, bands, prepared.raw_features, HISTORY, anchors)


# ── regions ───────────────────────────────────────────────────────────────


def metrics_table():
    return pd.DataFrame({
        "Run ID": ["tft_1"] * 4,
        "Region": ["World", "R5ASIA", "R6ROWO", "ZAF"],
        "Region Type": ["World", "R5", "R6", "ISO3"],
        "R2 Score (pooled)": [0.98, 0.96, 0.99, -0.4],
        "Sample Size": [8000, 11000, 126, 2400],
    })


def test_eligible_regions_apply_both_gates_and_keep_file_order():
    gated = eligible_regions(metrics_table(), 0.95, 1000)

    assert gated["Region"].tolist() == ["World", "R5ASIA"]  # R6ROWO too small, ZAF too poor
    assert list(gated.columns) == ["Region", "Region Type", "R2 Score (pooled)", "Sample Size"]


def test_eligible_regions_on_an_empty_table_is_empty():
    assert eligible_regions(metrics_table().iloc[:0], 0.95, 1000).empty


def test_eligible_regions_name_the_missing_column():
    with pytest.raises(ValueError, match="Sample Size"):
        eligible_regions(metrics_table().drop(columns=["Sample Size"]), 0.95, 1000)


def test_region_metrics_can_be_computed_from_a_prediction_bundle():
    horizon = pd.DataFrame({
        "Region": ["World", "World", "R5ASIA", "R5ASIA"],
        "A": [1.0, 2.0, 3.0, 4.0], "B": [2.0, 4.0, 6.0, 8.0],
        "A__observed": [1.0, 1.0, 1.0, 1.0], "B__observed": [1.0, 1.0, 1.0, 0.0],
    })
    y_true = horizon[TARGETS].to_numpy()

    table = region_metrics_from_predictions("tft_1", horizon, y_true, y_true * 1.1, TARGETS)

    assert table["Region"].tolist() == ["World", "R5ASIA"]
    assert table.loc[table["Region"] == "R5ASIA", "Sample Size"].item() == 3  # the unobserved element is skipped


# ── baselines ─────────────────────────────────────────────────────────────


def test_candidates_need_the_category_and_the_window_length():
    prepared = make_prepared(
        trajectory("M1", "S1", "World"),
        trajectory("M3", "S3", "World", n_steps=14),
        trajectory("M4", "S4", "World", category="C1"),
        trajectory("M5", "S5", "R5X"),
    )

    candidates = candidate_baselines(prepared, "World", min_steps=15)

    assert [c.key for c in candidates] == [("M1", "S1")]


def test_candidates_rank_by_reported_inputs_then_coverage_then_name():
    prepared = make_prepared(
        trajectory("M5", "S5", "World", split="val", imputed=("feat_a",)),
        trajectory("M2", "S2", "World", split="test", imputed=("feat_a",)),
        trajectory("M2", "S2", "R5X", split="test", imputed=("feat_a",)),
        trajectory("M1", "S1", "World"),
    )

    candidates = candidate_baselines(prepared, "World")

    assert [c.key for c in candidates] == [("M1", "S1"), ("M2", "S2"), ("M5", "S5")]
    first, second, third = candidates
    assert (first.n_reported, second.n_reported, third.n_reported) == (3, 2, 2)
    assert (first.n_regions, second.n_regions, third.n_regions) == (1, 2, 1)
    assert (second.split, second.n_steps, second.first_year, second.last_year) == ("test", 20, 2005, 2100)
    assert second.label == "M2 / S2"


def test_default_baseline_is_the_preferred_pair_where_offered_else_the_best():
    prepared = make_prepared(trajectory("M1", "S1", "World"), trajectory("M2", "S2", "World", imputed=("feat_b",)))
    candidates = candidate_baselines(prepared, "World")

    assert choose_default_baseline(candidates, ("M2", "S2")).key == ("M2", "S2")
    assert choose_default_baseline(candidates, ("MX", "SX")).key == ("M1", "S1")
    assert choose_default_baseline([], ("M1", "S1")) is None


def test_baseline_rows_come_in_step_order_and_missing_ones_raise():
    prepared = make_prepared(trajectory("M1", "S1", "World"))
    prepared.frame = prepared.frame.iloc[::-1].reset_index(drop=True)

    rows = baseline_rows(prepared, "World", "M1", "S1")

    assert rows["Step"].tolist() == list(range(20))
    with pytest.raises(KeyError):
        baseline_rows(prepared, "World", "M1", "nope")


def test_a_run_without_a_saved_split_is_refused(tmp_path, monkeypatch):
    import src.utils.run_store as run_store

    monkeypatch.setattr(run_store, "get_run_root", lambda _run_id: str(tmp_path))

    with pytest.raises(FileNotFoundError, match="splits.parquet"):
        load_prepared_run(run_store.RunStore("tft_01"))
    assert not (tmp_path / "artifacts").exists()


# ── bands ─────────────────────────────────────────────────────────────────


def test_lever_bands_use_reported_values_only(population):
    bands = lever_bands(population.frame, FEATURES, min_count=5)

    at_2010 = bands[(bands["feature"] == "feat_a") & (bands["Year"] == 2010)].iloc[0]
    # Six reported values 11..66; the imputed 1100 of S9 stays out.
    assert (at_2010["lo"], at_2010["hi"], at_2010["n"]) == (13.75, 63.25, 6)
    at_2010_b = bands[(bands["feature"] == "feat_b") & (bands["Year"] == 2010)].iloc[0]
    assert at_2010_b["n"] == 7  # feat_b is reported by S9


def test_bands_sit_on_the_decade_grid_and_interpolate_between(population):
    bands = lever_bands(population.frame, FEATURES)
    feat_a = bands[bands["feature"] == "feat_a"]

    assert feat_a["Year"].tolist() == list(range(2010, 2101, 10))
    lo, hi = band_at(feat_a, [2010, 2015, 2020])
    assert lo[1] == pytest.approx((lo[0] + lo[2]) / 2) and hi[1] == pytest.approx((hi[0] + hi[2]) / 2)
    assert lever_bands(population.frame, FEATURES, year_step=5)["Year"].min() == 2005


def test_lever_bands_leave_thin_years_unbounded():
    groups = [trajectory("M1", f"S{i}", "World", scale=float(i), n_steps=19) for i in range(1, 7)]
    groups.append(trajectory("M9", "S9", "World", scale=9.0))
    frame = make_prepared(*groups).frame

    bands = lever_bands(frame, FEATURES, min_count=5)
    at_2100 = bands[(bands["feature"] == "feat_a") & (bands["Year"] == 2100)].iloc[0]
    assert at_2100["n"] == 1
    assert math.isnan(at_2100["lo"]) and math.isnan(at_2100["hi"])

    # One scenario of seven is too small a share of the population, whatever the count floor.
    sparse = lever_bands(frame, FEATURES, min_count=1, min_fraction=0.5)
    assert math.isnan(sparse[(sparse["feature"] == "feat_a") & (sparse["Year"] == 2100)]["lo"].iloc[0])
    assert not math.isnan(sparse[(sparse["feature"] == "feat_a") & (sparse["Year"] == 2090)]["lo"].iloc[0])
    every = lever_bands(frame, FEATURES, min_count=1, min_fraction=0.0)
    assert not math.isnan(every[(every["feature"] == "feat_a") & (every["Year"] == 2100)]["lo"].iloc[0])


def test_target_bands_use_observed_values_only(population):
    frame = population.frame.copy()
    frame.loc[frame["Model"] == "M9", "A__observed"] = 0.0
    frame.loc[frame["Model"] == "M9", "A"] = 1e6

    bands = ar6_target_bands(frame, TARGETS, min_count=1)

    at_2010 = bands[(bands["target"] == "A") & (bands["Year"] == 2010)].iloc[0]
    assert at_2010["n"] == 6 and at_2010["hi"] == 1.0
    assert 2005 not in bands["Year"].tolist()


def test_band_position_is_zero_at_lo_and_one_at_hi():
    band = pd.DataFrame({"Year": [2020, 2030, 2040], "lo": [10.0, 20.0, 30.0], "hi": [20.0, 40.0, 30.0]})

    position = band_position(pd.Series([10.0, 25.0, 40.0, 30.0], index=[2020, 2025, 2030, 2040]), band)

    assert position.loc[2020] == 0.0
    assert position.loc[2025] == pytest.approx((25.0 - 15.0) / (30.0 - 15.0))  # band interpolated in Year
    assert position.loc[2030] == 1.0
    assert math.isnan(position.loc[2040])  # degenerate band


# ── anchors and levers ────────────────────────────────────────────────────


def test_anchor_years_follow_history_and_always_end_at_the_last_year():
    assert resolve_anchor_years(YEARS, YEARS[:3]) == [2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]
    assert resolve_anchor_years(range(2005, 2061, 5), [2005, 2010, 2015]) == [2030, 2040, 2050, 2060]
    assert resolve_anchor_years(range(2010, 2101, 5), [2010, 2020, 2035]) == [2040, 2050, 2060, 2070, 2080, 2090, 2100]
    assert resolve_anchor_years(range(2005, 2096, 5), [2005, 2010, 2015])[-1] == 2095


def test_lever_specs_bound_each_anchor_by_the_band_widened_to_the_baseline(population):
    rows, specs = specs_for(population)
    feat_a = next(s for s in specs if s.feature == "feat_a")

    assert feat_a.enabled and feat_a.reported
    assert feat_a.history_years == [2005, 2010, 2015] and feat_a.last_year == 2100
    end = feat_a.anchors[2100]
    # S1 is the smallest scenario: 29 sits below the band's 5th percentile.
    assert (end.base, end.band_lo, end.band_hi) == (29.0, pytest.approx(36.25), pytest.approx(166.75))
    assert (end.lo, end.hi) == (29.0, pytest.approx(166.75))
    assert feat_a.multiplier_bounds == (1.0, pytest.approx(166.75 / 29.0))


def test_imputed_and_degenerate_levers_are_locked(population):
    frame = population.frame
    frame["feat_b"] = 5.0  # every scenario reports the same value
    prepared = make_prepared()
    prepared.frame = frame
    rows, specs = specs_for(prepared, "M9", "S9")
    by_feature = {s.feature: s for s in specs}

    assert not by_feature["feat_a"].enabled and "imputed" in by_feature["feat_a"].reason
    assert by_feature["feat_a"].multiplier_bounds is None
    assert not by_feature["feat_b"].enabled and "one value" in by_feature["feat_b"].reason
    assert by_feature["feat_c"].enabled


def test_a_zero_baseline_keeps_the_anchors_but_not_the_multiplier(population):
    frame = population.frame
    frame.loc[(frame["Model"] == "M1") & (frame["Scenario"] == "S1"), "feat_a"] = 0.0
    prepared = make_prepared()
    prepared.frame = frame
    _, specs = specs_for(prepared)
    feat_a = next(s for s in specs if s.feature == "feat_a")

    assert feat_a.enabled
    assert feat_a.multiplier_bounds is None and "anchors" in feat_a.multiplier_reason
    assert feat_a.anchors[2100].lo == 0.0


def test_the_multiplier_ramps_from_history_and_reads_back(population):
    _, specs = specs_for(population)
    feat_a = next(s for s in specs if s.feature == "feat_a")

    anchors = ramp_anchor_values(feat_a, 2.0)

    assert anchors[2100] == pytest.approx(2.0 * feat_a.anchors[2100].base)
    fraction = (2030 - 2015) / (2100 - 2015)
    assert anchors[2030] == pytest.approx(feat_a.anchors[2030].base * (1.0 + fraction))
    assert multiplier_from_anchors(feat_a, anchors) == pytest.approx(2.0)
    assert multiplier_from_anchors(feat_a, {}) == 1.0


def test_lever_paths_keep_history_and_interpolate_in_year():
    years = [2005, 2010, 2015, 2020, 2030, 2040, 2060, 2080, 2100]
    baseline = pd.Series(1.0, index=years)

    path = interpolate_lever_path(years, [2005, 2010, 2015], baseline, {2040: 3.0, 2100: 5.0, 2010: 9.0})

    assert path[:3].tolist() == [1.0, 1.0, 1.0]  # the anchor inside history is ignored
    assert path[3] == pytest.approx(1.0 + 2.0 * 5 / 25)
    assert path[4] == pytest.approx(1.0 + 2.0 * 15 / 25)
    assert path[5] == 3.0
    assert path[6] == pytest.approx(3.0 + 2.0 * 20 / 60)
    assert path[8] == 5.0
    flat = interpolate_lever_path(years, [2005, 2010, 2015], baseline, {2040: 3.0})
    assert flat[6:].tolist() == [3.0, 3.0, 3.0]
    np.testing.assert_array_equal(interpolate_lever_path(years, [2005, 2010, 2015], baseline, {}), baseline.to_numpy())


def test_apply_levers_changes_only_the_edited_input_after_history(population):
    rows, specs = specs_for(population)
    feat_a = next(s for s in specs if s.feature == "feat_a")

    edited = apply_levers(rows, {"feat_a": ramp_anchor_values(feat_a, 2.0)}, HISTORY)

    unchanged = [c for c in rows.columns if c != "feat_a"]
    pd.testing.assert_frame_equal(edited[unchanged], rows[unchanged])
    assert edited["feat_a"].iloc[:HISTORY].tolist() == rows["feat_a"].iloc[:HISTORY].tolist()
    assert edited["feat_a"].iloc[-1] == pytest.approx(2.0 * rows["feat_a"].iloc[-1])
    assert (edited["feat_a"].iloc[HISTORY:] > rows["feat_a"].iloc[HISTORY:]).all()
    assert edited["feat_a"].dtype == rows["feat_a"].dtype


def test_presets_land_on_the_band_and_skip_locked_levers(population):
    frame = population.frame
    frame.loc[(frame["Model"] == "M1") & (frame["Scenario"] == "S1"), "feat_c_is_missing"] = 1.0
    prepared = make_prepared()
    prepared.frame = frame
    _, specs = specs_for(prepared)

    edits = preset_edits({"feat_a": 1.0, "feat_c": 0.5, "unknown": 0.2}, specs)

    assert list(edits) == ["feat_a"]
    feat_a = next(s for s in specs if s.feature == "feat_a")
    assert edits["feat_a"][2100] == pytest.approx(feat_a.anchors[2100].band_hi)
    assert edits["feat_a"][2030] > feat_a.anchors[2030].base


def test_band_coverage_counts_edited_values_against_the_raw_band(population):
    _, specs = specs_for(population)
    feat_a = next(s for s in specs if s.feature == "feat_a")
    inside = {2050: feat_a.anchors[2050].band_lo + 1.0, 2100: feat_a.anchors[2100].band_hi}
    outside = {2050: feat_a.anchors[2050].band_hi * 10}

    assert band_coverage({"feat_a": inside}, specs) == (2, 2)
    assert band_coverage({"feat_a": outside, "feat_b": {2100: 0.0}}, specs) == (0, 2)
    assert band_coverage({}, specs) == (0, 0)


# ── results ───────────────────────────────────────────────────────────────


def tidy_predictions(rows, offset=0.0):
    future = rows.iloc[HISTORY:]
    out = future[["Model", "Scenario", "Region", "Step", "Year"]].copy()
    for target in TARGETS:
        out[f"{target}_pred"] = future[target].to_numpy() + offset
    return out.reset_index(drop=True)


@pytest.fixture
def result(population):
    rows, specs = specs_for(population)
    feat_a = next(s for s in specs if s.feature == "feat_a")
    edits = {"feat_a": ramp_anchor_values(feat_a, 1.5)}
    edited = apply_levers(rows, edits, HISTORY)
    candidate = candidate_baselines(population, "World")[0]
    return assemble_result(
        "tft_test", "World", candidate, rows, edited, FEATURES, TARGETS,
        tidy_predictions(rows), tidy_predictions(rows, offset=0.5), edits,
        band_coverage(edits, specs), HISTORY,
    )


def test_result_carries_years_inputs_and_a_perfect_baseline_fit(result):
    assert result.history_years == [2005, 2010, 2015]
    assert result.predicted_years == YEARS[HISTORY:]
    assert result.targets == TARGETS
    assert result.baseline_fit["R2"] == 1.0 and result.baseline_fit["Count"] == 2 * 17
    assert result.inputs_edited.loc[2100, "feat_a"] == pytest.approx(1.5 * result.inputs_baseline.loc[2100, "feat_a"])
    assert (result.pred_edited - result.pred_baseline).stack().eq(0.5).all()


def test_export_has_one_row_per_year_with_prefixed_columns(result):
    exported = export_frame(result)

    assert exported["Year"].tolist() == YEARS
    for column in ("input_baseline:feat_a", "input:feat_a", "iam:A", "pred_baseline:A", "pred:B"):
        assert column in exported.columns
    assert exported["pred:A"].isna().sum() == HISTORY  # no forecast for the history years


def test_metadata_is_json_safe_and_carries_the_sidebar_keys(result):
    metadata = result_metadata(result, region_r2=float("nan"), preset="High carbon price")

    text = json.dumps(metadata)
    assert "NaN" not in text
    assert metadata["region_pooled_r2"] is None
    assert metadata["regions"] == ["World"] and metadata["scenario_categories"] == ["C3"]
    assert metadata["model_families"] == ["FAM"]
    assert metadata["num_data_points"] == 17 * 2
    assert metadata["metrics"]["R2"] == 1.0
    assert metadata["edits"]["feat_a"]["2100"] == pytest.approx(1.5 * result.inputs_baseline.loc[2100, "feat_a"])
    assert metadata["band_coverage"] == {"inside": result.coverage[0], "total": result.coverage[1]}
    assert metadata["predictions"]["years"] == YEARS[HISTORY:]
