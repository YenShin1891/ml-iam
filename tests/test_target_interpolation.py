"""The vectorised target interpolation must match the per-group apply it replaced."""

import numpy as np
import pandas as pd

from src.data.preprocess import interpolate_targets, resample_to_uniform_intervals

GROUP_COLS = ["Model", "Scenario", "Region"]
TARGETS = ["A", "B"]


def _reference_interpolate(data):
    """The previous implementation: Series.interpolate(method="index") per group."""
    data = data.sort_values(GROUP_COLS + ["Year"]).copy()
    data["Year"] = pd.to_numeric(data["Year"])

    def _interp_group(grp):
        grp = grp.set_index("Year").sort_index()
        for col in TARGETS:
            if grp[col].notna().any():
                grp[col] = grp[col].interpolate(method="index")
        return grp.reset_index()

    keys = [data[c] for c in GROUP_COLS]
    out = data.groupby(keys, group_keys=False)[data.columns.tolist()].apply(_interp_group)
    return out.reset_index(drop=True)


def _frame(seed=0, n_groups=30):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        years = sorted(rng.choice(np.arange(2010, 2101, 5), size=rng.integers(3, 12), replace=False))
        for year in years:
            rows.append({
                "Model": f"M{g % 4}", "Scenario": f"S{g}", "Region": "World",
                "Year": str(year),  # header strings, as the cached parquet holds them
                "A": rng.normal() if rng.random() > 0.3 else np.nan,
                "B": rng.normal() if rng.random() > 0.3 else np.nan,
                "feat": rng.normal(),
            })
    frame = pd.DataFrame(rows)
    frame.loc[frame["Scenario"] == "S3", "A"] = np.nan  # a target never observed in one series
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # unsorted input


def test_matches_the_per_group_reference_including_leading_and_trailing_gaps():
    frame = _frame()

    got = interpolate_targets(frame, GROUP_COLS, TARGETS)
    expected = _reference_interpolate(frame)

    pd.testing.assert_frame_equal(
        got[GROUP_COLS + ["Year"] + TARGETS].reset_index(drop=True),
        expected[GROUP_COLS + ["Year"] + TARGETS].reset_index(drop=True),
        check_dtype=False,
    )


def test_the_observation_mask_marks_what_was_measured():
    frame = _frame(seed=1)

    got = interpolate_targets(frame, GROUP_COLS, TARGETS)

    original = frame.set_index(GROUP_COLS + ["Year"])["A"].notna()
    marked = got.assign(Year=got["Year"].astype(str)).set_index(GROUP_COLS + ["Year"])["A__observed"].astype(bool)
    assert marked.reindex(original.index).tolist() == original.tolist()


def _reference_resample(data, interval=5):
    """The previous implementation: reindex + interpolate one group at a time."""
    from configs.data import CATEGORICAL_COLUMNS, NON_FEATURE_COLUMNS

    groups = []
    data = data.assign(Year=pd.to_numeric(data["Year"], errors="coerce"))
    for _key, grp in data.groupby(GROUP_COLS, sort=False):
        grp = grp.sort_values("Year")
        years = grp["Year"].values
        diffs = np.diff(years)
        if len(diffs) == 0 or diffs.min() <= interval:
            groups.append(grp)
            continue
        full_years = np.arange(years.min(), years.max() + 1, interval)
        rebuilt = grp.set_index("Year").reindex(full_years)
        fill_cols = set(GROUP_COLS) | {c for c in NON_FEATURE_COLUMNS if c != "Year"} | set(CATEGORICAL_COLUMNS)
        for col in fill_cols:
            if col in rebuilt.columns:
                rebuilt[col] = rebuilt[col].ffill().bfill()
        for col in [c for c in rebuilt.columns if c.endswith("__observed")]:
            rebuilt[col] = rebuilt[col].fillna(0.0)
        numeric = [c for c in rebuilt.select_dtypes(include=[np.number]).columns if not c.endswith("__observed")]
        rebuilt[numeric] = rebuilt[numeric].interpolate(method="index")
        groups.append(rebuilt.reset_index(names="Year"))
    return pd.concat(groups, ignore_index=True)


def test_resampling_matches_the_per_group_reference():
    rng = np.random.default_rng(3)
    rows = []
    for g in range(40):
        step = rng.choice([5, 10, 20])
        years = list(range(2010, 2010 + step * rng.integers(3, 8), step))
        if g == 7:
            years.append(2013)  # an off-grid year, dropped by both implementations
        for year in sorted(years):
            rows.append({
                "Model": f"M{g % 3}", "Scenario": f"S{g}", "Region": "R10_X", "Region_Scale": "R10",
                "Scenario_Category": "C3", "Model_Family": f"M{g % 3}", "Year": str(year),
                "A": rng.normal() if rng.random() > 0.2 else np.nan,
                "B": rng.normal(),
                "feat": rng.normal() if rng.random() > 0.2 else np.nan,
            })
    frame = pd.DataFrame(rows)
    frame["A__observed"] = frame["A"].notna().astype("float32")
    frame["B__observed"] = 1.0

    got = resample_to_uniform_intervals(frame, GROUP_COLS, TARGETS)
    expected = _reference_resample(frame).sort_values(GROUP_COLS + ["Year"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(got[expected.columns], expected, check_dtype=False)


def test_only_coarse_series_are_resampled_and_the_rest_pass_through():
    frame = pd.DataFrame({
        "Model": ["M"] * 6, "Scenario": ["coarse"] * 3 + ["fine"] * 3, "Region": ["World"] * 6,
        "Year": ["2020", "2030", "2040", "2020", "2025", "2030"],
        "A": [1.0, 3.0, 5.0, 1.0, 2.0, 3.0],
        "A__observed": [1.0] * 6,
        "feat": [10.0, 30.0, 50.0, 1.0, 2.0, 3.0],
    })

    out = resample_to_uniform_intervals(frame, GROUP_COLS, ["A"])

    coarse = out[out["Scenario"] == "coarse"].sort_values("Year")
    assert coarse["Year"].tolist() == [2020, 2025, 2030, 2035, 2040]
    assert coarse["A"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert coarse["A__observed"].tolist() == [1.0, 0.0, 1.0, 0.0, 1.0]
    fine = out[out["Scenario"] == "fine"].sort_values("Year")
    assert fine["Year"].tolist() == [2020, 2025, 2030]
    assert fine["A__observed"].tolist() == [1.0, 1.0, 1.0]
