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
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, cast

import pandas as pd

# Local config (new consolidated modules)
from configs.paths import DATA_PATH, RESULTS_PATH, RAW_DATA_PATH
from configs import data as dp
from src.utils.utils import setup_console_logging, LocalFormatter


# ---------- Paths & constants ----------
REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_DIR = REPO_ROOT / "metadata"

VAR_CLASSIFICATION_CSV = METADATA_DIR / "variable_classification_1019.csv"
SCENARIO_CATEGORY_CSV = METADATA_DIR / "scenario_category.csv"
MODEL_BASE_YEAR_CSV = METADATA_DIR / "iam_base_years.csv"
SSP_FAMILY_CSV = METADATA_DIR / "ssp_families.csv"


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


def load_ssp_families(csv_path: Path = SSP_FAMILY_CSV) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"SSP families CSV not found: {csv_path}; ensure metadata/ssp_families.csv is available."
        )
    df = pd.read_csv(csv_path, dtype=str)
    return df


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


UNCATEGORISED = "no-climate-assessment"


def add_scenario_category(df: pd.DataFrame, scenario_cat: pd.DataFrame) -> pd.DataFrame:
    """Attach the AR6 category of each scenario, labelling those without one.

    Most of the misses are scenarios the metadata CSV does list but whose
    category cell holds a literal "#N/A" from a failed lookup; only a handful
    per raw file are absent from the CSV outright.  Either way the category is
    simply unknown, which is what UNCATEGORISED already means for the 237
    scenarios the CSV labels that way, so they share the label rather than
    being dropped.  Dropping them used to happen silently downstream, where
    pivot_table discards rows whose index carries a NaN; the column is not a
    model feature (see NON_FEATURE_COLUMNS), so it cost ~3% of the series for
    nothing the models predict from.
    """
    out = df.merge(scenario_cat[["Scenario", "Scenario_Category"]], on="Scenario", how="left")
    uncategorised = out["Scenario_Category"].isna()
    if uncategorised.any():
        logging.info(
            "Labelling %d rows from %d scenarios as %r: no Scenario_Category in %s",
            int(uncategorised.sum()),
            out.loc[uncategorised, "Scenario"].nunique(),
            UNCATEGORISED,
            SCENARIO_CATEGORY_CSV.name,
        )
        out["Scenario_Category"] = out["Scenario_Category"].fillna(UNCATEGORISED)
    # Reorder: Model, Scenario, Scenario_Category, Region, Variable, Unit, years...
    cols = ["Model", "Scenario", "Scenario_Category", "Region"]
    remainder = [c for c in out.columns if c not in cols]
    out = out.loc[:, cols + remainder]
    return cast(pd.DataFrame, out)


def _split_year_and_non_year_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (year_columns, non_year_columns) inferred from column names.

    A year column is any column whose name can be parsed as an integer year
    in a reasonable range (e.g. 1800-2300).
    """
    year_cols: List[str] = []
    for c in df.columns:
        try:
            year = int(str(c))
        except (TypeError, ValueError):
            continue
        if 1800 <= year <= 2300:
            year_cols.append(c)

    if not year_cols:
        raise ValueError("Could not detect any year-like columns in input DataFrame")

    non_year_cols = [c for c in df.columns if c not in year_cols]
    return year_cols, non_year_cols


# Labels that name a unit already spelled another way, so a variable is not
# reported in two "different" units when only the spelling differs.  These
# need relabelling, not rescaling:
#
#   Int$ at PPP is defined as the US dollar's purchasing power in the base
#   year, and the AR6 World medians for GDP|PPP agree under both labels to
#   within scenario spread, so they are the same quantity.
#
# The map is explicit rather than a case fold because folding would also
# rewrite "PJ/yr" or "US$2010/GJ", which reach the unit table as they are,
# and would collide with the exact-match conversions below.
UNIT_ALIASES = {
    "Million": "million",
    "Million ha": "million ha",
    "Million t DM/yr": "million t DM/yr",
    "billion Int$2010/yr": "billion US$2010/yr",
}


def resolve_units(df: pd.DataFrame):
    """Normalize units and return (df_without_unit, unit_table).

    Year columns are inferred from column names rather than a fixed index.
    """
    df = df.copy(deep=True)

    year_cols, non_year_cols = _split_year_and_non_year_columns(df)

    df["Unit"] = df["Unit"].replace(UNIT_ALIASES)

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

    # Every value of a variable ends up in one wide column, so a variable
    # reported in two units mixes magnitudes there.  The AR6 files carry one
    # row per (Model, Scenario, Region, Variable), so checking within those
    # keys could never find anything; check per variable, across the file.
    units_per_variable = df.groupby("Variable")["Unit"].nunique()
    mixed = units_per_variable[units_per_variable > 1]
    if not mixed.empty:
        breakdown = (
            df[df["Variable"].isin(mixed.index)]
            .groupby("Variable")["Unit"].agg(lambda u: dict(u.value_counts()))
        )
        logging.warning(
            "%d variable(s) are reported in more than one unit; their columns mix them: %s",
            len(mixed), breakdown.to_dict(),
        )

    # Keep all non-year columns (including Unit) for the unit table
    unit_table = df.loc[:, non_year_cols].copy()
    df = df.drop(columns=["Unit"])  # remove Unit after capturing unit_table
    return df, unit_table


def melt_and_pivot_year(df: pd.DataFrame) -> pd.DataFrame:
    year_columns, non_year_columns = _split_year_and_non_year_columns(df)
    df_year = df.melt(id_vars=non_year_columns, value_vars=year_columns, var_name="Year", value_name="value")
    out = (
        df_year.pivot_table(
            index=["Model", "Scenario", "Scenario_Category", "Region", "Year"],
            columns="Variable",
            values="value",
        ).reset_index()
    )
    return out


ID_COLUMNS = ["Model", "Scenario", "Scenario_Category", "Region", "Year"]


def apply_completeness_threshold(df: pd.DataFrame, selected_vars: pd.DataFrame, ratio: float) -> pd.DataFrame:
    """Keep the year-rows that report at least *ratio* of the selected variables.

    Only the variable columns count: the identity columns are never missing,
    so counting them let a row through with five fewer variables than the
    ratio asks for.
    """
    if not (0.0 < ratio <= 1.0):
        raise ValueError("ratio must be in (0, 1]")
    threshold = int(len(selected_vars) * ratio)
    value_cols = [c for c in df.columns if c not in ID_COLUMNS]
    return df.dropna(subset=value_cols, thresh=threshold)


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
    if df.empty:
        raise ValueError(f"Base year metadata is empty: {csv_path}")
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
    base_candidate: Optional[int] = None
    if not row.empty:
        val = row.iloc[0].get('Base year')
        try:
            if val and pd.notna(val):
                base_candidate = int(val)
        except (TypeError, ValueError):
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
    # Insert Region_Scale after Region
    from src.utils.regions import region_scales

    region_col_idx = year_pivoted.columns.get_loc("Region")
    year_pivoted.insert(
        region_col_idx + 1, "Region_Scale", region_scales(year_pivoted["Region"]).to_numpy()
    )
    return year_pivoted


def add_ssp_family_column(df: pd.DataFrame, ssp_meta: pd.DataFrame) -> pd.DataFrame:
    """Attach SSP family metadata as a categorical column based on (Model, Scenario).

    Expects ssp_meta to have columns ["Model", "Scenario", "Ssp_family"].
    """
    required_cols = {"Model", "Scenario", "Ssp_family"}
    if not required_cols.issubset(ssp_meta.columns):
        missing = required_cols - set(ssp_meta.columns)
        raise KeyError(f"SSP families metadata missing columns: {sorted(missing)}")

    # if there are any duplicates in ssp_meta, warn and drop duplicates
    dup_mask = ssp_meta.duplicated(subset=["Model", "Scenario"], keep='first')
    if dup_mask.any():
        logging.warning(f"SSP families metadata has {dup_mask.sum()} duplicate (Model, Scenario) entries; keeping first occurrence.")
        ssp_meta = ssp_meta.drop_duplicates(subset=["Model", "Scenario"], keep='first')
    out = df.merge(ssp_meta[["Model", "Scenario", "Ssp_family"]], on=["Model", "Scenario"], how="left")

    # Place Ssp_family immediately after Scenario and log missing count
    if "Ssp_family" in out.columns:
        na_count = int(out["Ssp_family"].isna().sum())
        logging.info(f"SSP family column has {na_count}/{len(out)} missing values after merge.")

        ssp_col = out.pop("Ssp_family")
        if "Scenario" in out.columns:
            insert_pos = list(out.columns).index("Scenario") + 1
        else:
            insert_pos = len(out.columns)
        out.insert(insert_pos, "Ssp_family", ssp_col)

    return out


# ---------- Orchestration ----------
def run_pipeline(
    raw_dir: Path,
    data_dir: Path,
    results_dir: Path,
    dataset_name: Optional[str],
    output_variables: Iterable[str],
    min_count: Optional[int] = None,
    completeness_ratio: Optional[float] = None,
    filenames: Optional[Iterable[str]] = None,
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
    ssp_families = load_ssp_families()
    logging.info(f"Loaded SSP families metadata: {len(ssp_families)} rows")

    # Stats and selection
    stat_table_raw = build_stat_table(df_list)
    stat_table = merge_variable_classification(stat_table_raw, var_class)
    
    # Tag checks
    tags = list(getattr(dp, "TAGS", []))
    if getattr(dp, "INTERPOLATE_TARGETS", False):
        tags.append("interp-targets")
    if getattr(dp, "SCALE_AWARE_IMPUTATION", False):
        tags.append("scale-impute")
    include_intermediate = "include-intermediate" in tags
    apply_base_year = "apply-base-year" in tags
    
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
    for d in filtered:
        proc, unit_tbl = resolve_units(d)
        processed_list.append(proc)
        unit_tables.append(cast(pd.DataFrame, unit_tbl))

    unit_table = pd.concat(unit_tables, axis=0, ignore_index=True)
    # Defer writing until versioned directory is known

    # Year-wise wide frame
    logging.info("Melting and pivoting into year-indexed wide frames…")
    processed_year_frames = [melt_and_pivot_year(d) for d in processed_list]
    processed_df_year = pd.concat(processed_year_frames, axis=0, ignore_index=True)
    logging.info(f"Year-wise concatenated shape: {processed_df_year.shape}")

    # Apply model base-year filtering BEFORE completeness threshold so early padded years don't dilute counts
    if apply_base_year:
        base_year_meta = load_model_base_years(MODEL_BASE_YEAR_CSV)
        processed_df_year = apply_base_year_filter(processed_df_year, base_year_meta)
    else:
        logging.info("Base-year filtering disabled (no 'apply-base-year' tag present).")

    # Completeness filtering
    before_rows = len(processed_df_year)
    processed_df_year = apply_completeness_threshold(
        processed_df_year, selected_vars, completeness_ratio or dp.COMPLETENESS_RATIO
    )
    logging.info(f"Rows: {before_rows} -> {len(processed_df_year)} after completeness filter")

    # Missing stats
    value_only = processed_df_year.drop(columns=ID_COLUMNS, errors="ignore")
    stat_table_with_na = compute_missing_stats(value_only, stat_table)
    # Defer writing until versioned directory is known

    # Final time-series wide dataset
    logging.info("Creating final time-series wide dataset…")
    final_series = to_series_wide(processed_df_year)
    if "with-ssp" in tags:
        final_series = add_ssp_family_column(final_series, ssp_families)
    logging.info(f"Final series shape: {final_series.shape}")

    # Build version label and versioned directory name
    if dataset_name is None:
        parts = [dp.NAME_PREFIX, f"min{min_count or dp.MIN_COUNT}", f"comp{(completeness_ratio or dp.COMPLETENESS_RATIO):.1f}"]
        if tags:
            parts.extend(tags)
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
        manifest = {
            "dataset": str(out_path.relative_to(version_dir)),
            "raw_dir": str(raw_dir),
            "filenames": list(filenames),
            "min_count": int(min_count or dp.MIN_COUNT),
            "completeness_ratio": float(completeness_ratio or dp.COMPLETENESS_RATIO),
            "output_variables": list(output_variables),
            "tags": list(tags),
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
        with open(versions_file, 'r') as f:
            existing_versions = [line.strip() for line in f if line.strip()]
    
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
    logger = setup_console_logging(level=logging.INFO)

    # Also log to a per-run file under data_dir/logs
    log_dir = Path(args.data_dir) / "logs"
    ensure_dirs(log_dir)
    log_file = log_dir / f"process_data_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(LocalFormatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logging.info(f"File logging enabled: {log_file}")

    # Use OUTPUT_VARIABLES from unified config
    output_vars = dp.OUTPUT_VARIABLES

    run_pipeline(
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
