"""
Data processing pipeline:
- Read AR6 scenario CSVs
- Build variable stats and select variables (inputs + configured outputs)
- Merge scenario categories
- Resolve units (e.g. EJ/yr→PJ/yr)
- Melt/pivot into year-indexed wide format
- Apply completeness threshold and compute NA stats
- Produce time-series wide dataset (rows: model/scenario/region/variable; cols: years)
- Save artifacts under configured DATA_PATH and RESULTS_PATH

Config is read from configs/paths.py and configs/data.py.
Metadata CSVs are expected under <repo_root>/metadata/.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, cast

import numpy as np
import pandas as pd

# Local config (new consolidated modules)
from configs.paths import DATA_PATH, RESULTS_PATH, RAW_DATA_PATH
from configs.data import YEAR_STARTS_AT, INDEX_COLUMNS
from configs import data as dp
from src.utils.utils import setup_console_logging


# ---------- Paths & constants ----------
REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_DIR = REPO_ROOT / "metadata"

VAR_CLASSIFICATION_CSV = METADATA_DIR / "variable_classification_0327.csv"
SCENARIO_CATEGORY_CSV = METADATA_DIR / "scenario_category.csv"
MODEL_BASE_YEAR_CSV = METADATA_DIR / "unique_models_all_with_base_year.csv"


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


def select_variables(stat_table: pd.DataFrame, output_variables: Iterable[str], min_count: Optional[int], include_intermediate: bool = False) -> pd.DataFrame:
    selected = stat_table.copy()
    if min_count is not None:
        selected = selected[selected["Count"] >= min_count]
    
    # Base mask: input variables and configured output variables
    mask = (selected["Type"] == "input") | (selected["Variable"].isin(list(output_variables)))
    
    if include_intermediate:
        mask = mask | (selected["Type"] == "intermediate")
    
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

    # Keep the leading columns up to and including 'Unit' for the unit table
    leading_cols = list(df.columns[: (year_starts_at + 1)])
    unit_table = df.loc[:, leading_cols].copy()
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


def filter_by_year_interval(processed_df_year: pd.DataFrame) -> pd.DataFrame:
    def most_common_year_interval(years: np.ndarray) -> int:
        if len(years) < 2:
            return -1
        intervals = pd.Series(sorted(years)).diff().dropna().astype(int)
        if intervals.empty:
            return -1
        return int(intervals.value_counts().idxmax())

    keep_idx = []
    for key, group in processed_df_year.groupby(INDEX_COLUMNS):
        years = group["Year"].dropna().astype(int).unique()
        mc_interval = most_common_year_interval(years)
        if mc_interval != -1 and mc_interval < 10:
            keep_idx.extend(group.index.tolist())
    before = len(processed_df_year)
    processed_df_year = processed_df_year.loc[keep_idx].reset_index(drop=True)
    after = len(processed_df_year)
    logging.info(f"no_10years tag: removed {before - after} rows with most common year interval >= 10")
    return processed_df_year


def compute_missing_stats(df: pd.DataFrame, original_stat_table: pd.DataFrame) -> pd.DataFrame:
    na_counts = df.isna().sum().reset_index()
    na_counts.columns = ["Variable", "pct_missing"]
    total_rows = df.shape[0]
    na_counts["pct_missing"] = na_counts["pct_missing"] / total_rows * 100.0
    stat_table = original_stat_table.merge(na_counts, on="Variable", how="left")
    return stat_table


# ---------- Base year filtering ----------
def load_model_base_years(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Model base year CSV not found: {csv_path}; add tag 'apply-base-year' only when the metadata CSV is available, or omit the tag to skip filtering."
        )
    df = pd.read_csv(csv_path, dtype=str)
    assert not df.empty, "Base year metadata is empty"
    # Normalize column names by stripping whitespace
    df.columns = [c.strip() for c in df.columns]
    return df


def resolve_effective_base_year(model: str, meta: pd.DataFrame, available_years: list[int], default_year: int = 2020) -> int:
    """Determine effective base year selecting the ceiling (>=) if exact not present.

    Rules:
      1. Use declared base year if present; else fallback to default_year.
      2. If exact base year exists in available_years, use it.
      3. Otherwise choose the smallest available year that is GREATER than the candidate (ceiling).
      4. If all available years are below the candidate, return 10000 (no valid year).
    """
    years_sorted = sorted(available_years)
    row = meta.loc[meta['Model'] == model]
    base_candidate: int | None = None
    if not row.empty:
        val = row.iloc[0].get('Base year')
        try:
            if val and pd.notna(val):
                base_candidate = int(val)
        except Exception:
            base_candidate = None
    if base_candidate is None:
        base_candidate = default_year
    if not years_sorted:
        return base_candidate
    if base_candidate in years_sorted:
        return base_candidate
    ge_years = [y for y in years_sorted if y > base_candidate]
    if ge_years:
        return ge_years[0]
    else:
        return 10000


def apply_base_year_filter(processed_df_year: pd.DataFrame, base_year_meta: pd.DataFrame) -> pd.DataFrame:
    """Apply per-model base year filtering. Rows earlier than that year are dropped. 
    Returns a new filtered DataFrame.
    """
    model_base_map: dict[str, int] = {}
    for m, g in processed_df_year.groupby('Model'):
        m_str = str(m)
        model_years = sorted(g['Year'].dropna().astype(int).unique())
        model_base_map[m_str] = resolve_effective_base_year(m_str, base_year_meta, model_years)

    df = processed_df_year.copy()
    # Build temporary numeric representations for safe comparison without altering original Year dtype
    year_num = pd.to_numeric(df['Year'], errors='coerce')
    effective = pd.Series(df['Model'].map(model_base_map), index=df.index)
    effective_num = pd.to_numeric(effective, errors='coerce')
    before_rows = len(df)
    mask = year_num.notna() & effective_num.notna() & (year_num >= effective_num)
    dropped = before_rows - mask.sum()
    df = df.loc[mask].copy()
    logging.info(
        "Base-year filter (per-model) removed %d pre-base-year rows (remaining %d)",
        dropped,
        len(df)
    )
    return df


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

    # We'll materialize artifacts into a versioned directory inside DATA_PATH.
    # Create base output dirs; versioned dir is created after version is computed.
    ensure_dirs(data_dir, results_dir)

    # Load
    logging.info("Loading raw CSV files…")
    df_list = load_raw_files(raw_dir, filenames)
    logging.info(f"Loaded {len(df_list)} frames; rows per file: {[len(df) for df in df_list]}")
    var_class, scenario_cat = load_metadata()
    logging.info(f"Loaded metadata: var_class={len(var_class)} rows, scenario_cat={len(scenario_cat)} rows")

    # Stats and selection
    stat_table_raw = build_stat_table(df_list)
    stat_table = merge_variable_classification(stat_table_raw, var_class)
    
    # Tag checks
    tags = list(getattr(dp, "TAGS", []))
    include_intermediate = "include-intermediate" in tags
    apply_base_year = "apply-base-year" in tags
    no_10years = "no-10years" in tags
    
    selected_vars = select_variables(stat_table, output_variables, min_count=min_count or dp.MIN_COUNT, include_intermediate=include_intermediate)
    if include_intermediate:
        logging.info(f"Selected {len(selected_vars)} out of {len(stat_table)} variables (inputs + intermediates + configured outputs)")
    else:
        logging.info(f"Selected {len(selected_vars)} out of {len(stat_table)} variables (inputs + configured outputs)")

    # Filter variables
    filtered = filter_by_selected_variables(df_list, selected_vars)

    # Merge scenario categories
    filtered = [add_scenario_category(df, scenario_cat) for df in filtered]

    # Unit normalization and capture unit table
    processed_list: List[pd.DataFrame] = []
    unit_tables: List[pd.DataFrame] = []
    for i, d in enumerate(filtered, start=1):
        proc, unit_tbl = resolve_units(d, YEAR_STARTS_AT)
        processed_list.append(proc)
        unit_tables.append(cast(pd.DataFrame, unit_tbl))

    unit_table = pd.concat(unit_tables, axis=0, ignore_index=True)
    # Defer writing until versioned directory is known

    # Year-wise wide frame
    logging.info("Melting and pivoting into year-indexed wide frames…")
    processed_year_frames = [melt_and_pivot_year(d, YEAR_STARTS_AT) for d in processed_list]
    processed_df_year = pd.concat(processed_year_frames, axis=0, ignore_index=True)
    logging.info(f"Year-wise concatenated shape: {processed_df_year.shape}")

    # Apply model base-year filtering BEFORE completeness threshold so early padded years don't dilute counts
    if apply_base_year:
        base_year_meta = load_model_base_years(MODEL_BASE_YEAR_CSV)
        processed_df_year = apply_base_year_filter(processed_df_year, base_year_meta)
    else:
        logging.info("Base-year filtering disabled (no 'apply-base-year' tag present).")

    # Apply year interval filtering if the tag is present
    if no_10years:
        processed_df_year = filter_by_year_interval(processed_df_year)
    else:
        logging.info("Year-interval filtering disabled (no 'no-10years' tag present).")

    # Completeness filtering
    before_rows = len(processed_df_year)
    processed_df_year = apply_completeness_threshold(
        processed_df_year, selected_vars, completeness_ratio or dp.COMPLETENESS_RATIO
    )
    logging.info(f"Rows: {before_rows} -> {len(processed_df_year)} after completeness filter")

    # Missing stats
    value_only = processed_df_year.drop(columns=["Model", "Scenario", "Scenario_Category", "Region", "Year"], errors="ignore")
    stat_table_with_na = compute_missing_stats(value_only, stat_table)
    # Defer writing until versioned directory is known

    # Final time-series wide dataset
    logging.info("Creating final time-series wide dataset…")
    final_series = to_series_wide(processed_df_year)
    logging.info(f"Final series shape: {final_series.shape}")

    # Build version label and versioned directory name
    from datetime import datetime
    if dataset_name is None:
        parts = [dp.NAME_PREFIX, f"min{min_count or dp.MIN_COUNT}", f"comp{(completeness_ratio or dp.COMPLETENESS_RATIO):.1f}"]
        if dp.TAGS:
            parts.extend(dp.TAGS)
        if dp.INCLUDE_DATE:
            parts.append(datetime.now().strftime(dp.DATE_FMT))
        version_label = "-".join(parts)  # no extension
    else:
        # Use provided name without extension as the version label
        version_label = Path(dataset_name).stem

    version_dir = data_dir / version_label
    ensure_dirs(version_dir)

    # Write analysis artifacts now into the versioned directory
    if dp.SAVE_ANALYSIS:
        selected_vars.to_csv(version_dir / "var_selected.csv", index=False)
        unit_table.to_csv(version_dir / "unit_table.csv", index=False)
        stat_table_with_na.to_csv(version_dir / "stat_table.csv", index=False)

    # Save processed series into the versioned directory with a stable filename
    out_path = version_dir / "processed_series.csv"
    final_series.to_csv(out_path, index=False)

    # Write a small manifest for traceability
    try:
        import json
        manifest = {
            "dataset": str(out_path.relative_to(version_dir)),
            "raw_dir": str(raw_dir),
            "filenames": list(filenames),
            "min_count": int(min_count or dp.MIN_COUNT),
            "completeness_ratio": float(completeness_ratio or dp.COMPLETENESS_RATIO),
            "year_starts_at": int(YEAR_STARTS_AT),
            "output_variables": list(output_variables),
            "tags": list(getattr(dp, "TAGS", [])),
        }
        # Write manifest alongside the dataset using the version label in the filename
        (version_dir / f"{version_label}-manifest.json").write_text(json.dumps(manifest, indent=2))
    except Exception as e:
        logging.warning(f"Failed to write manifest: {e}")

    # Basic logging
    total = final_series.shape[0] * final_series.shape[1]
    missing = final_series.isna().sum().sum()
    pct = (missing / total * 100.0) if total else 0.0
    logging.info(f"Saved dataset: {out_path}")
    logging.info(f"Rows: {final_series.shape[0]}, Cols: {final_series.shape[1]}, Missing: {missing}/{total} ({pct:.2f}%)")
    
    # Update dataset versions list for easy CLI reference
    update_dataset_versions_list(data_dir, version_label)

    return out_path


def update_dataset_versions_list(data_dir, new_version_name):
    """Update dataset_versions.txt by appending new version names in creation order."""
    versions_file = data_dir / "dataset_versions.txt"
    
    # Read existing versions if file exists
    existing_versions = []
    if versions_file.exists():
        try:
            with open(versions_file, 'r') as f:
                existing_versions = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            existing_versions = []
    
    # Only add if this version doesn't already exist
    if new_version_name not in existing_versions:
        # Append new version to the end
        with open(versions_file, 'a') as f:
            f.write(f"{new_version_name}\n")
        
        logging.info(f"Added {new_version_name} to dataset versions list")
    else:
        logging.info(f"Version {new_version_name} already exists in versions list")


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

    # Configure console logging with shared format (default INFO)
    setup_console_logging(level=logging.INFO)

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

if __name__ == "__main__":
    main()
