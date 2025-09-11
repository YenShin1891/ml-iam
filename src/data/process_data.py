"""
Refactored data processing pipeline from the original Jupyter notebook into a
reusable module with a small CLI.

Major steps (mirrors the notebook):
- Read AR6 scenario CSVs
- Build variable stats and select variables (inputs + configured outputs)
- Merge scenario categories
- Resolve units (EJ/yr→PJ/yr, Million tkm→bn tkm/yr, Million pkm→bn pkm/yr)
- Melt/pivot into year-indexed wide format
- Apply completeness threshold and compute NA stats
- Produce time-series wide dataset (rows: model/scenario/region/variable; cols: years)
- Save artifacts under configured DATA_PATH and RESULTS_PATH

Config is read from configs/paths.py and configs/data.py.
Metadata CSVs are expected under <repo_root>/metadata/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, cast

import pandas as pd

# Local config (new consolidated modules)
from configs.paths import DATA_PATH, RESULTS_PATH, RAW_DATA_PATH
from configs.data import YEAR_STARTS_AT
from configs import data as dp


# ---------- Paths & constants ----------
REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_DIR = REPO_ROOT / "metadata"

VAR_CLASSIFICATION_CSV = METADATA_DIR / "variable_classification_0327.csv"
SCENARIO_CATEGORY_CSV = METADATA_DIR / "scenario_category.csv"


# ---------- Utilities ----------
def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def get_model_family(model: str) -> str:
    fam = model
    for sep in ["_", "-", "/", " "]:
        if sep in fam:
            fam = fam.split(sep)[0]
    return fam


# ---------- IO ----------
def load_raw_files(raw_dir: Path, filenames: Iterable[str]) -> List[pd.DataFrame]:
    dfs: List[pd.DataFrame] = []
    for name in filenames:
        fp = raw_dir / name
        if not fp.exists():
            raise FileNotFoundError(f"Raw file not found: {fp}")
        df = pd.read_csv(fp, low_memory=False)
        dfs.append(df)
    return dfs


def load_metadata() -> Tuple[pd.DataFrame, pd.DataFrame]:
    var_class = pd.read_csv(VAR_CLASSIFICATION_CSV, dtype=str)
    scenario_cat = pd.read_csv(SCENARIO_CATEGORY_CSV, dtype=str)
    return var_class, scenario_cat


# ---------- Core transforms ----------
def build_stat_table(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat([d[["Model", "Scenario", "Region", "Variable", "Unit"]] for d in df_list], axis=0, ignore_index=True)
    variable_counts = df["Variable"].value_counts().reset_index()
    variable_counts.columns = ["Variable", "Count"]
    return variable_counts.sort_values("Count", ascending=False)


def merge_variable_classification(stat_table: pd.DataFrame, var_class: pd.DataFrame) -> pd.DataFrame:
    subset = var_class[["Variable(All)", "Type", "Model type"]].copy()
    subset.columns = ["Variable", "Type", "Model_Type"]
    return stat_table.merge(subset, on="Variable", how="left")


def select_variables(stat_table: pd.DataFrame, output_variables: Iterable[str], min_count: Optional[int]) -> pd.DataFrame:
    selected = stat_table.copy()
    if min_count is not None:
        selected = selected[selected["Count"] >= min_count]
    mask = (selected["Type"] == "input") | (selected["Variable"].isin(list(output_variables)))
    return cast(pd.DataFrame, selected.loc[mask].reset_index(drop=True))


def filter_by_selected_variables(df_list: List[pd.DataFrame], selected_vars: pd.DataFrame) -> List[pd.DataFrame]:
    keep = set(selected_vars["Variable"].unique())
    return [cast(pd.DataFrame, df.loc[df["Variable"].isin(keep)].copy()) for df in df_list]


def add_scenario_category(df: pd.DataFrame, scenario_cat: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(scenario_cat[["Scenario", "Scenario_Category"]], on="Scenario", how="left")
    # Reorder: Model, Scenario, Scenario_Category, Region, Variable, Unit, years...
    cols = ["Model", "Scenario", "Scenario_Category", "Region"]
    remainder = [c for c in out.columns if c not in cols]
    out = out.loc[:, cols + remainder]
    return cast(pd.DataFrame, out)


def resolve_units(df: pd.DataFrame, year_starts_at: int):
    """Normalize units and return (df_without_unit, unit_table)."""
    df = df.copy(deep=True)

    year_cols = list(df.columns[year_starts_at:])

    # EJ/yr → PJ/yr (×1000)
    mask = df["Unit"] == "EJ/yr"
    if mask.any():
        df.loc[mask, year_cols] = df.loc[mask, year_cols].apply(pd.to_numeric, errors="coerce") * 1000
        df.loc[mask, "Unit"] = "PJ/yr"

    # Million tkm → bn tkm/yr (×0.001)
    mask = df["Unit"] == "Million tkm"
    if mask.any():
        df.loc[mask, year_cols] = df.loc[mask, year_cols].apply(pd.to_numeric, errors="coerce") * 0.001
        df.loc[mask, "Unit"] = "bn tkm/yr"

    # Million pkm → bn pkm/yr (×0.001)
    mask = df["Unit"] == "Million pkm"
    if mask.any():
        df.loc[mask, year_cols] = df.loc[mask, year_cols].apply(pd.to_numeric, errors="coerce") * 0.001
        df.loc[mask, "Unit"] = "bn pkm/yr"

    # Unit consistency check per (Model, Scenario, Region, Variable)
    unit_check = df.groupby(["Model", "Scenario", "Region", "Variable"])['Unit'].nunique()
    if (unit_check > 1).any():
        bad = unit_check[unit_check > 1]
        raise ValueError(f"Unit mismatch found for {len(bad)} combinations. Sample: {bad.head(5)}")

    unit_table = df.loc[:, df.columns[: (year_starts_at + 1)]].copy()
    df = df.drop(columns=["Unit"])  # remove Unit after capturing unit_table
    return df, unit_table


def melt_and_pivot_year(df: pd.DataFrame, year_starts_at: int) -> pd.DataFrame:
    year_columns = list(df.columns[year_starts_at:])
    non_year_columns = list(df.columns[:year_starts_at])
    df_year = df.melt(id_vars=non_year_columns, value_vars=year_columns, var_name="Year", value_name="value")
    out = (
        df_year.pivot_table(
            index=["Model", "Scenario", "Scenario_Category", "Region", "Year"],
            columns="Variable",
            values="value",
        ).reset_index()
    )
    return out


def apply_completeness_threshold(df: pd.DataFrame, selected_vars: pd.DataFrame, ratio: float) -> pd.DataFrame:
    if not (0.0 < ratio <= 1.0):
        raise ValueError("ratio must be in (0, 1]")
    threshold = int(len(selected_vars) * ratio)
    kept = df.dropna(thresh=threshold)
    return kept


def compute_missing_stats(df: pd.DataFrame, original_stat_table: pd.DataFrame) -> pd.DataFrame:
    na_counts = df.isna().sum().reset_index()
    na_counts.columns = ["Variable", "pct_missing"]
    total_rows = df.shape[0]
    na_counts["pct_missing"] = na_counts["pct_missing"] / total_rows * 100.0
    stat_table = original_stat_table.merge(na_counts, on="Variable", how="left")
    return stat_table


def to_series_wide(processed_df_year: pd.DataFrame) -> pd.DataFrame:
    var_melted = processed_df_year.melt(
        id_vars=["Model", "Scenario", "Scenario_Category", "Region", "Year"],
        var_name="Variable",
        value_name="value",
    )
    year_pivoted = (
        var_melted.pivot_table(
            index=["Model", "Scenario", "Scenario_Category", "Region", "Variable"],
            columns="Year",
            values="value",
        ).reset_index()
    )
    # Insert Model_Family as the second column
    year_pivoted.insert(1, "Model_Family", year_pivoted["Model"].apply(get_model_family))
    return year_pivoted


# ---------- Orchestration ----------
def run_pipeline(
    raw_dir: Path,
    data_dir: Path,
    results_dir: Path,
    dataset_name: Optional[str],
    output_variables: Iterable[str],
    min_count: Optional[int] = None,
    completeness_ratio: Optional[float] = None,
    filenames: Iterable[str] | None = None,
) -> Path:
    """Execute the end-to-end processing and return the path to the dataset CSV."""

    # Defaults for AR6 file names (v1.1) matching the original notebook
    if filenames is None:
        filenames = dp.RAW_FILENAMES

    analysis_dir = results_dir / "analysis"
    ensure_dirs(data_dir, results_dir, analysis_dir)

    # Load
    df_list = load_raw_files(raw_dir, filenames)
    var_class, scenario_cat = load_metadata()

    # Stats and selection
    stat_table_raw = build_stat_table(df_list)
    stat_table = merge_variable_classification(stat_table_raw, var_class)
    selected_vars = select_variables(stat_table, output_variables, min_count=min_count or dp.MIN_COUNT)

    # Save selection snapshot
    selected_vars.to_csv(analysis_dir / "var_selected.csv", index=False)

    # Filter variables
    filtered = filter_by_selected_variables(df_list, selected_vars)

    # Merge scenario categories
    filtered = [add_scenario_category(df, scenario_cat) for df in filtered]

    # Unit normalization and capture unit table
    processed_list: List[pd.DataFrame] = []
    unit_tables: List[pd.DataFrame] = []
    for d in filtered:
        proc, unit_tbl = resolve_units(d, YEAR_STARTS_AT)
        processed_list.append(proc)
        unit_tables.append(cast(pd.DataFrame, unit_tbl))

    unit_table = pd.concat(unit_tables, axis=0, ignore_index=True)
    if dp.SAVE_ANALYSIS:
        unit_table.to_csv(analysis_dir / "unit_table.csv", index=False)

    # Year-wise wide frame
    processed_year_frames = [melt_and_pivot_year(d, YEAR_STARTS_AT) for d in processed_list]
    processed_df_year = pd.concat(processed_year_frames, axis=0, ignore_index=True)

    # Completeness filtering
    processed_df_year = apply_completeness_threshold(
        processed_df_year, selected_vars, completeness_ratio or dp.COMPLETENESS_RATIO
    )

    # Missing stats
    value_only = processed_df_year.drop(columns=["Model", "Scenario", "Scenario_Category", "Region", "Year"], errors="ignore")
    stat_table_with_na = compute_missing_stats(value_only, stat_table)
    if dp.SAVE_ANALYSIS:
        stat_table_with_na.to_csv(analysis_dir / "stat_table.csv", index=False)

    # Final time-series wide dataset
    final_series = to_series_wide(processed_df_year)
    # Build versioned dataset name
    if dataset_name is None:
        from datetime import datetime

        parts = [dp.NAME_PREFIX, f"min{min_count or dp.MIN_COUNT}", f"comp{(completeness_ratio or dp.COMPLETENESS_RATIO):.1f}"]
        if dp.TAGS:
            parts.extend(dp.TAGS)
        if dp.INCLUDE_DATE:
            parts.append(datetime.now().strftime(dp.DATE_FMT))
        dataset_name = "-".join(parts) + ".csv"

    out_path = data_dir / dataset_name
    final_series.to_csv(out_path, index=False)

    # Write a small manifest for traceability
    try:
        import json
        manifest = {
            "dataset": str(out_path.name),
            "raw_dir": str(raw_dir),
            "filenames": list(filenames),
            "min_count": int(min_count or dp.MIN_COUNT),
            "completeness_ratio": float(completeness_ratio or dp.COMPLETENESS_RATIO),
            "year_starts_at": int(YEAR_STARTS_AT),
            "output_variables": list(output_variables),
            "tags": list(getattr(dp, "TAGS", [])),
        }
        (results_dir / "analysis" / (out_path.stem + "-manifest.json")).write_text(json.dumps(manifest, indent=2))
    except Exception as e:
        print(f"Warning: failed to write manifest: {e}")

    # Maintain a 'latest' symlink for easy discovery
    try:
        latest_link = data_dir / f"{dp.NAME_PREFIX}-latest.csv"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(out_path.name)
    except Exception as e:
        print(f"Warning: failed to create latest symlink: {e}")

    # Basic logging
    total = final_series.shape[0] * final_series.shape[1]
    missing = final_series.isna().sum().sum()
    pct = (missing / total * 100.0) if total else 0.0
    print(f"Saved dataset: {out_path}")
    print(f"Rows: {final_series.shape[0]}, Cols: {final_series.shape[1]}, Missing: {missing}/{total} ({pct:.2f}%)")

    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process AR6 scenarios into model-ready CSV")
    p.add_argument("--raw-dir", type=Path, default=Path(RAW_DATA_PATH), help="Directory containing AR6 raw CSV files")
    p.add_argument("--data-dir", type=Path, default=Path(DATA_PATH), help="Output data directory")
    p.add_argument("--results-dir", type=Path, default=Path(RESULTS_PATH), help="Analysis/results directory")
    p.add_argument("--dataset-name", type=str, default=None, help="Override output dataset filename (default is versioned)")
    p.add_argument("--min-count", type=int, default=None, help="Minimum count to keep a variable (default from config)")
    p.add_argument("--completeness", type=float, default=None, help="Row completeness ratio (0-1] (default from config)")
    p.add_argument(
        "--filenames",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit list of AR6 CSV filenames inside --raw-dir",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Use OUTPUT_VARIABLES from unified config
    output_vars = dp.OUTPUT_VARIABLES

    out_csv = run_pipeline(
        raw_dir=args.raw_dir,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        dataset_name=args.dataset_name,
        output_variables=output_vars,
        min_count=args.min_count,
        completeness_ratio=args.completeness,
        filenames=args.filenames,
    )
    print(out_csv)


if __name__ == "__main__":
    main()
