import os
import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Tuple, cast
from sklearn.preprocessing import StandardScaler

from configs.paths import DATA_PATH, RESULTS_PATH
from configs.data import (
    DEFAULT_DATASET,
    N_LAG_FEATURES,
    OUTPUT_VARIABLES,
    INDEX_COLUMNS,
    NON_FEATURE_COLUMNS,
    CATEGORICAL_COLUMNS,
    MAX_YEAR,
    SPLIT_SEED,
    REGION_CATEGORIES,
    INTERPOLATE_TARGETS,
    SCALE_AWARE_IMPUTATION,
    NORMALIZE_TARGETS_BY_POPULATION,
    POPULATION_COLUMN,
    KEEP_PARTIAL_TARGETS,
)

def split_data(
    prepared: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = list(prepared.groupby(INDEX_COLUMNS))
    n_groups = len(groups)
    n_test_groups = int(n_groups * test_size)
    n_val_groups = int(n_groups * val_size)
    n_train_groups = n_groups - n_test_groups - n_val_groups
    
    rng = np.random.RandomState(SPLIT_SEED)
    rng.shuffle(groups)

    train_groups = groups[:n_train_groups]
    val_groups = groups[n_train_groups:n_train_groups + n_val_groups]
    test_groups = groups[n_train_groups + n_val_groups:]

    train_data = pd.concat([group[1] for group in train_groups]).reset_index(drop=True)
    val_data = pd.concat([group[1] for group in val_groups]).reset_index(drop=True)
    test_data = pd.concat([group[1] for group in test_groups]).reset_index(drop=True)
    
    logging.info(f"Train: {len(train_data)} rows, Val: {len(val_data)} rows, Test: {len(test_data)} rows")

    return train_data, val_data, test_data



def encode_categorical_columns(data, columns):
    for col in columns:
        if col in data.columns:
            if col == 'Region':
                if not REGION_CATEGORIES:
                    set_region_categories(data[col])
                data[col] = (
                    pd.Categorical(data[col].astype(str), categories=REGION_CATEGORIES, ordered=True)
                    .codes
                    .astype('float32')
                )
            else:
                data[col] = data[col].astype('category').cat.codes
    return data


def add_missingness_indicators(
    prepared: pd.DataFrame,
    features: list,
    time_known: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
):
    """Add <feature>_is_missing indicators before splitting the dataset.
    """
    if time_known is None:
        time_known = ["Year", "DeltaYears"]
    if categorical_columns is None:
        categorical_columns = CATEGORICAL_COLUMNS

    updated_features = list(features)
    excluded = set(categorical_columns) | set(time_known)

    for col in features:
        if col in excluded or col.endswith("_is_missing"):
            continue
        if col not in prepared.columns:
            continue
        indicator_name = f"{col}_is_missing"
        if indicator_name not in prepared.columns:
            prepared[indicator_name] = (
                prepared[col]
                .isna()
                .map({True: 1, False: 0})
                .astype("float32")
            )
        if indicator_name not in updated_features:
            updated_features.append(indicator_name)

    return prepared, updated_features


def impute_with_train_medians(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list,
    time_known: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
):
    """Impute continuous features with train medians.

    When SCALE_AWARE_IMPUTATION is enabled and a 'Region_Scale' column exists,
    computes medians per Region_Scale group so that ISO3 countries are imputed
    from other ISO3 values, R10 from R10, etc.  Falls back to global train
    median for features that are entirely NaN within a scale group.  Raises
    an error if a Region_Scale present in val/test has zero rows in train.
    """
    if time_known is None:
        time_known = ["Year", "DeltaYears"]
    if categorical_columns is None:
        categorical_columns = CATEGORICAL_COLUMNS

    excluded = set(categorical_columns) | set(time_known)

    use_scale = (
        SCALE_AWARE_IMPUTATION
        and "Region_Scale" in train_df.columns
    )

    if use_scale:
        # Validate: every Region_Scale in val/test must exist in train
        train_scales = set(train_df["Region_Scale"].unique())
        for split_name, split_df in [("val", val_df), ("test", test_df)]:
            split_scales = set(split_df["Region_Scale"].unique()) if "Region_Scale" in split_df.columns else set()
            missing = split_scales - train_scales
            if missing:
                raise ValueError(
                    f"Region_Scale values {missing} found in {split_name} split but not in training data. "
                    f"This means the train/val/test split left an entire geographic scale unrepresented in training. "
                    f"Training scales: {sorted(train_scales)}. "
                    f"Fix: check the split logic or the dataset — every Region_Scale must appear in the training set."
                )

        # Compute per-scale medians from train
        scale_medians = {}
        for scale in sorted(train_scales):
            scale_data = train_df[train_df["Region_Scale"] == scale]
            medians = {}
            for col in features:
                if col in excluded or col.endswith("_is_missing") or col not in scale_data.columns:
                    continue
                med = pd.to_numeric(scale_data[col], errors="coerce").median()
                if pd.notna(med):
                    medians[col] = med
            scale_medians[scale] = medians

        # Also compute global medians as fallback for features missing within a scale
        global_medians = {}
        for col in features:
            if col in excluded or col.endswith("_is_missing") or col not in train_df.columns:
                continue
            med = pd.to_numeric(train_df[col], errors="coerce").median()
            global_medians[col] = med if pd.notna(med) else 0.0

        # Apply imputation per scale
        for frame in (train_df, val_df, test_df):
            if "Region_Scale" not in frame.columns:
                continue
            for scale in frame["Region_Scale"].unique():
                mask = frame["Region_Scale"] == scale
                s_medians = scale_medians.get(scale, {})
                for col in features:
                    if col in excluded or col.endswith("_is_missing") or col not in frame.columns:
                        continue
                    fill_val = s_medians.get(col, global_medians.get(col, 0.0))
                    frame.loc[mask, col] = frame.loc[mask, col].fillna(fill_val)

        logging.info(
            "Scale-aware imputation applied across %d Region_Scale groups: %s",
            len(train_scales), sorted(train_scales),
        )
    else:
        # Original global imputation
        for col in features:
            if col in excluded or col.endswith("_is_missing"):
                continue
            if col not in train_df.columns:
                continue
            median_value = pd.to_numeric(train_df[col], errors="coerce").median()
            if pd.isna(median_value):
                logging.warning("Column '%s' has all-NaN in train; filling with 0.0", col)
                median_value = 0.0
            for frame in (train_df, val_df, test_df):
                if col in frame.columns:
                    frame[col] = frame[col].fillna(median_value)

    return train_df, val_df, test_df


def prepare_data(prepared, targets, features):
    obs_cols = observed_mask_columns(targets)
    has_obs = all(c in prepared.columns for c in obs_cols)

    train_data, val_data, test_data = split_data(prepared)

    X_train = train_data[features].copy()
    X_train_index_columns = train_data[[col for col in INDEX_COLUMNS if col not in features]].copy()
    y_train = train_data[targets].values.copy()
    X_val = val_data[features].copy()
    X_val_index_columns = val_data[[col for col in INDEX_COLUMNS if col not in features]].copy()
    y_val = val_data[targets].values.copy()
    X_test = test_data[features].copy()
    X_test_index_columns = test_data[[col for col in INDEX_COLUMNS if col not in features]].copy()
    y_test = test_data[targets].values.copy()

    # Extract observed masks (float32 arrays matching y shape)
    if has_obs:
        obs_train = train_data[obs_cols].values.copy()
        obs_val = val_data[obs_cols].values.copy()
        obs_test = test_data[obs_cols].values.copy()
    else:
        obs_train = np.ones_like(y_train, dtype=np.float32)
        obs_val = np.ones_like(y_val, dtype=np.float32)
        obs_test = np.ones_like(y_test, dtype=np.float32)

    X_train = encode_categorical_columns(X_train, CATEGORICAL_COLUMNS)
    X_val = encode_categorical_columns(X_val, CATEGORICAL_COLUMNS)
    X_test = encode_categorical_columns(X_test, CATEGORICAL_COLUMNS)

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_val_scaled = x_scaler.transform(X_val)
    X_test_scaled = x_scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    y_scaler = StandardScaler()
    # Fit scaler on observed values only; fill NaN with 0 for transform
    y_train_filled = np.where(np.isnan(y_train), 0.0, y_train)
    y_val_filled = np.where(np.isnan(y_val), 0.0, y_val)
    y_test_filled = np.where(np.isnan(y_test), 0.0, y_test)
    y_train_scaled = y_scaler.fit_transform(y_train_filled)
    y_val_scaled = y_scaler.transform(y_val_filled)
    y_test_scaled = y_scaler.transform(y_test_filled)

    X_test_with_index_scaled = pd.concat(
        [X_test_scaled.reset_index(drop=True), X_test_index_columns.reset_index(drop=True)],
        axis=1
    )

    train_groups = np.asarray(train_data[INDEX_COLUMNS].astype(str).agg('_'.join, axis=1))
    val_groups = np.asarray(val_data[INDEX_COLUMNS].astype(str).agg('_'.join, axis=1))

    return (
        X_train_scaled, y_train_scaled, X_train_index_columns,
        X_val_scaled, y_val_scaled, X_val_index_columns,
        X_test_with_index_scaled, y_test_scaled,
        test_data,
        x_scaler, y_scaler,
        train_groups, val_groups,
        obs_train, obs_val, obs_test,
    )

def load_and_process_data(version=None) -> pd.DataFrame:
    logging.info("Loading and processing data...")
    # Load the dataset specified in configs.data.DEFAULT_DATASET or from version subdirectory.
    # If version is provided, use version/processed_series.csv, otherwise use DEFAULT_DATASET.
    if version:
        dataset_path = os.path.join(DATA_PATH, version, "processed_series.csv")
    else:
        dataset_path = os.path.join(DATA_PATH, DEFAULT_DATASET)

    # Validate dataset path
    if not os.path.isfile(dataset_path):
        versions_file = os.path.join(DATA_PATH, "dataset_versions.txt")
        available_versions = None
        try:
            if os.path.isfile(versions_file) and os.path.getsize(versions_file) > 0:
                with open(versions_file, "r") as f:
                    versions = [ln.strip() for ln in f if ln.strip()]
                if versions:
                    available_versions = versions[-10:]
        except Exception:
            available_versions = None

        abs_path = os.path.abspath(dataset_path)
        if version is None:
            hint = (
                f"DEFAULT_DATASET is configured as '{DEFAULT_DATASET}', but no file was found at: {abs_path}. "
                "Set DATA_PATH in configs/paths.py and run `make process-data`, or pass --dataset <version_name>."
            )
        else:
            hint = (
                f"Processed dataset not found for version '{version}'. Expected file at: {abs_path}. "
                "Set DATA_PATH in configs/paths.py to the directory that contains processed datasets, "
                "or run `make process-data` to generate it."
            )

        if available_versions:
            hint += f" Available versions (last 10): {available_versions}"

        raise FileNotFoundError(hint)
        
    logging.info(f"Reading processed dataset: {dataset_path}")
    processed_series = pd.read_csv(dataset_path)
    # Identify year and non-year columns robustly
    all_cols = list(processed_series.columns)
    year_cols = [c for c in all_cols if str(c).isdigit()]
    try:
        cutoff = int(MAX_YEAR)
        year_cols = [c for c in year_cols if int(c) <= cutoff]
    except Exception:
        logging.warning("MAX_YEAR invalid; skipping year cutoff filter.")
    non_year_cols = [c for c in all_cols if c not in year_cols]

    # Optional: save a heatmap of missing values over variables x years
    try:
        import seaborn as sns  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
        heatmap_df = processed_series.set_index('Variable')[year_cols]
        plt.figure(figsize=(20, 80))
        sns.heatmap(heatmap_df.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.xlabel('Year')
        plt.ylabel('Variable')
        os.makedirs(RESULTS_PATH, exist_ok=True)
        plt.savefig(os.path.join(RESULTS_PATH, 'missing_values_heatmap.png'), bbox_inches='tight')
        plt.close()
    except Exception:
        logging.warning("Could not generate missing values heatmap; continuing.")

    year_melted = processed_series.melt(
        id_vars=non_year_cols, value_vars=year_cols, var_name='Year', value_name='value'
    )
    # Build pivot index — include Region_Scale if present
    pivot_index = ['Model', 'Model_Family', 'Scenario', 'Scenario_Category', 'Region']
    if 'Region_Scale' in non_year_cols:
        pivot_index.append('Region_Scale')
    pivot_index.append('Year')

    var_pivoted = year_melted.pivot_table(
        index=pivot_index,
        columns='Variable', values='value'
    ).reset_index()

    # Derive REGION_CATEGORIES from the actual data so embeddings always
    # cover every region present, regardless of target-filtering changes.
    if 'Region' in var_pivoted.columns:
        set_region_categories(var_pivoted['Region'])

    return var_pivoted


def set_region_categories(regions) -> None:
    """Derive REGION_CATEGORIES from a data column and update module globals.

    Mutates the existing list in-place so that ``from configs.data import
    REGION_CATEGORIES`` bindings see the update.
    """
    from configs.data import REGION_CATEGORIES, REGION_CODE_TO_LABEL

    cats = sorted(set(str(r) for r in regions))
    REGION_CATEGORIES.clear()
    REGION_CATEGORIES.extend(cats)
    REGION_CODE_TO_LABEL.clear()
    REGION_CODE_TO_LABEL.update({idx: r for idx, r in enumerate(cats)})


def observed_mask_columns(output_variables: list) -> list:
    """Return the list of ``{var}__observed`` column names for *output_variables*."""
    return [f"{var}__observed" for var in output_variables]


def interpolate_targets(
    data: pd.DataFrame,
    group_cols: list,
    output_variables: list,
) -> pd.DataFrame:
    """Interpolate target values within each group across years.

    For each (Model, Scenario, Region) group, fills interior NaN target
    values using linear interpolation (weighted by year). This ensures lag
    features are computed from real (interpolated) values instead of NaN.

    Only fills targets that have at least one non-NaN value in the group.

    Before interpolating, captures ``{var}__observed`` boolean columns so
    downstream code can distinguish originally-observed from filled values.
    """
    data = data.sort_values(group_cols + ["Year"]).copy()

    # Capture observation mask *before* any interpolation
    for col in output_variables:
        obs_col = f"{col}__observed"
        if obs_col not in data.columns:
            data[obs_col] = data[col].notna().astype("float32")

    before_nans = data[output_variables].isna().sum().sum()

    data["Year"] = pd.to_numeric(data["Year"], errors="coerce")

    def _interp_group(grp):
        grp = grp.set_index("Year").sort_index()
        for col in output_variables:
            if col in grp.columns and grp[col].notna().any():
                grp[col] = pd.to_numeric(grp[col], errors="coerce")
                grp[col] = grp[col].interpolate(method="index")
        return grp.reset_index()

    data = data.groupby(group_cols, group_keys=False).apply(_interp_group)
    data = data.reset_index(drop=True)

    after_nans = data[output_variables].isna().sum().sum()
    filled = before_nans - after_nans
    logging.info(
        "Target interpolation: filled %d NaN values (%d → %d remaining)",
        filled, before_nans, after_nans,
    )
    return data


def normalize_targets_by_population(
    data: pd.DataFrame,
    targets: list,
    population_col: str = POPULATION_COLUMN,
) -> pd.DataFrame:
    """Divide each target column by population, in place (same column names).

    Rows with missing or non-positive population get NaN targets (they will
    be dropped downstream by the existing dropna(subset=targets) logic, same
    as any other row with a missing target). Callers should interpolate
    `population_col` (e.g. via `interpolate_targets(data, group_cols, [population_col])`)
    before calling this, to avoid unnecessarily dropping otherwise-good rows.
    """
    data = data.copy()

    population = pd.to_numeric(data[population_col], errors="coerce")
    invalid = population.isna() | (population <= 0)
    n_invalid = int(invalid.sum())
    if n_invalid:
        logging.warning(
            "Population normalization: %d/%d rows have missing/non-positive %s; "
            "affected target values set to NaN and will be dropped downstream.",
            n_invalid, len(data), population_col,
        )

    safe_population = population.where(~invalid)
    for col in targets:
        data[col] = pd.to_numeric(data[col], errors="coerce") / safe_population

    return data


def denormalize_by_population(values: np.ndarray, population: np.ndarray) -> np.ndarray:
    """Multiply per-capita values back to absolute units, elementwise by row.

    `values` may be shape (n,) or (n, k); `population` is shape (n,) and is
    broadcast across the k target columns. NaNs propagate naturally.
    """
    values = np.asarray(values, dtype=float)
    population_col_vec = np.asarray(population, dtype=float).reshape(-1, 1)
    if values.ndim == 1:
        return (values.reshape(-1, 1) * population_col_vec).ravel()
    return values * population_col_vec


def add_lag_features(
    data: pd.DataFrame,
    group_cols: list,
    output_variables: list,
    n_lags: int = N_LAG_FEATURES,
    lag_required: bool = True,
) -> pd.DataFrame:
    """Add lagged target features using vectorized groupby.shift().

    ~250x faster than the per-group .apply() approach on 23k groups.
    """
    prepared = data.sort_values(group_cols + ['Year']).copy()

    for lag in range(1, n_lags + 1):
        shifted = prepared.groupby(group_cols, sort=False)[output_variables].shift(lag)
        for col in output_variables:
            prefix = 'prev_' if lag == 1 else f'prev{lag}_'
            prepared[f'{prefix}{col}'] = shifted[col]

    if lag_required:
        row_num = prepared.groupby(group_cols, sort=False).cumcount()
        prepared = prepared[row_num >= n_lags].reset_index(drop=True)

    return cast(pd.DataFrame, prepared)


def prepare_features_and_targets(data: pd.DataFrame, lag_required: bool = True) -> tuple:
    """
    Prepare features and targets for XGBoost model.

    Args:
        data: Input data DataFrame
        lag_required: When True, drop rows without a full history of lag features.
    """
    logging.info(
        "Preparing features and targets for XGBoost (lag_required=%s)...",
        lag_required,
    )

    if NORMALIZE_TARGETS_BY_POPULATION:
        data = interpolate_targets(data, INDEX_COLUMNS, [POPULATION_COLUMN])
        data = normalize_targets_by_population(data, OUTPUT_VARIABLES, POPULATION_COLUMN)

    # Always capture observation mask before interpolation fills NaNs.
    # interpolate_targets creates them internally, but if INTERPOLATE_TARGETS
    # is False we still need them.
    for col in OUTPUT_VARIABLES:
        obs_col = f"{col}__observed"
        if obs_col not in data.columns:
            data[obs_col] = data[col].notna().astype("float32")

    if INTERPOLATE_TARGETS:
        data = interpolate_targets(data, INDEX_COLUMNS, OUTPUT_VARIABLES)

    prepared = add_lag_features(data, INDEX_COLUMNS, OUTPUT_VARIABLES, lag_required=lag_required)
    prepared['Year'] = prepared['Year'].astype(int)

    targets = OUTPUT_VARIABLES
    obs_cols = observed_mask_columns(OUTPUT_VARIABLES)
    features = [
        col for col in prepared.columns
        if col not in NON_FEATURE_COLUMNS
        and col not in targets
        and not col.endswith("__observed")
    ]

    if KEEP_PARTIAL_TARGETS:
        # Keep rows where *at least one* target is observed (element-wise
        # masking handles the rest).  Rows where *all* targets are missing
        # carry zero gradient and waste memory, so drop them.
        any_observed = prepared[obs_cols].any(axis=1)
        n_dropped = int((~any_observed).sum())
        if n_dropped:
            logging.info("Dropped %d rows where all targets are unobserved", n_dropped)
        prepared = prepared[any_observed].reset_index(drop=True)
    else:
        # Original behaviour: require every target to be present.
        prepared = prepared.dropna(subset=targets).reset_index(drop=True)

    if not lag_required:
        missing_lag_rows = prepared[features].isna().any(axis=1).sum()
        if missing_lag_rows:
            logging.info(
                "Lag requirement disabled: retained %d rows with missing lag features",
                missing_lag_rows,
            )

    return prepared, features, targets


def prepare_features_and_targets_sequence(
    data: pd.DataFrame,
    lag_required: bool = True,
    min_context_length: int = 0,
) -> tuple:
    """
    Prepare features and targets for sequence models (TFT, LSTM, etc.).

    Args:
        data: Input data DataFrame
        lag_required: When True, drop rows without a full history of lag features.
        min_context_length: Minimum historical context required when lag features are required.

    This function adds explicit lagged target features for sequence models
    to mirror tree-based preprocessing. It also adds Step and DeltaYears for
    time series indexing.
    """
    logging.info(
        "Preparing features and targets for sequence models (lag_required=%s)...",
        lag_required,
    )

    if NORMALIZE_TARGETS_BY_POPULATION:
        data = interpolate_targets(data, INDEX_COLUMNS, [POPULATION_COLUMN])
        data = normalize_targets_by_population(data, OUTPUT_VARIABLES, POPULATION_COLUMN)

    # Always capture observation mask before interpolation fills NaNs.
    for col in OUTPUT_VARIABLES:
        obs_col = f"{col}__observed"
        if obs_col not in data.columns:
            data[obs_col] = data[col].notna().astype("float32")

    if INTERPOLATE_TARGETS:
        data = interpolate_targets(data, INDEX_COLUMNS, OUTPUT_VARIABLES)

    prepared = add_lag_features(data, INDEX_COLUMNS, OUTPUT_VARIABLES, lag_required=lag_required)
    prepared['Year'] = prepared['Year'].astype(int)

    targets = OUTPUT_VARIABLES
    obs_cols = observed_mask_columns(OUTPUT_VARIABLES)
    features = [
        col for col in prepared.columns
        if col not in NON_FEATURE_COLUMNS
        and col not in targets
        and not col.endswith("__observed")
    ]

    if KEEP_PARTIAL_TARGETS:
        any_observed = prepared[obs_cols].any(axis=1)
        n_dropped = int((~any_observed).sum())
        if n_dropped:
            logging.info("Dropped %d rows where all targets are unobserved", n_dropped)
        prepared = prepared[any_observed].reset_index(drop=True)
    else:
        prepared = prepared.dropna(subset=targets).reset_index(drop=True)

    if not lag_required:
        missing_lag_rows = prepared[features].isna().any(axis=1).sum()
        if missing_lag_rows:
            logging.info(
                "Lag requirement disabled: retained %d rows with missing lag features",
                missing_lag_rows,
            )

    # Make 'Step' and 'DeltaYears' after dropping NaNs
    # Step must align with group_ids used by sequence models
    group_cols = INDEX_COLUMNS
    prepared = prepared.sort_values(group_cols + ['Year'])
    prepared['Step'] = prepared.groupby(group_cols).cumcount().astype('int64')

    # Explicit gap feature: years elapsed since previous observation within each series
    prepared['DeltaYears'] = (
        prepared.groupby(group_cols)['Year'].diff().fillna(0).astype(int)
    )

    # Filter out early steps when a minimum context is required
    if lag_required and min_context_length > 0:
        original_len = len(prepared)
        prepared = prepared[prepared['Step'] >= min_context_length].copy()
        filtered_len = len(prepared)
        logging.info(
            "Lag requirement enforced: filtered out %d early steps, kept %d rows with Step >= %d",
            original_len - filtered_len,
            filtered_len,
            min_context_length,
        )

    return prepared, features, targets


def prepare_features_and_targets_tft(
    data: pd.DataFrame,
    lag_required: bool = True,
    min_context_length: int = 0,
) -> tuple:
    """
    Legacy function for TFT. Now calls prepare_features_and_targets_sequence.
    Kept for backward compatibility.
    """
    logging.info(
        "Preparing features and targets for TFT (using sequence preprocessing, lag_required=%s)...",
        lag_required,
    )
    return prepare_features_and_targets_sequence(
        data,
        lag_required=lag_required,
        min_context_length=min_context_length,
    )


def remove_rows_with_missing_outputs(X, y, X2=None):
    """
    Remove rows with missing outputs from the dataset.
    Args:
        X (pd.DataFrame or np.ndarray): Feature DataFrame.
        y (pd.DataFrame or np.ndarray): Target DataFrame or array.
        X2 (pd.DataFrame or np.ndarray, optional): Additional feature DataFrame.
    """
    mask = ~np.isnan(y).any(axis=1)
    logging.info(f"Removing {(~mask).sum()} rows with missing outputs")

    if isinstance(X, pd.DataFrame):
        X = X[mask].reset_index(drop=True)
    else:
        X = X[mask]

    if isinstance(y, pd.DataFrame):
        y = y[mask].reset_index(drop=True)
    else:
        y = y[mask]

    if X2 is not None:
        if isinstance(X2, pd.DataFrame):
            X2 = X2[mask].reset_index(drop=True)
        else:
            X2 = X2[mask]
        return X, y, X2

    return X, y
