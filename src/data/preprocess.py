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
    """Impute continuous features with train medians to mirror legacy TFT runs."""
    if time_known is None:
        time_known = ["Year", "DeltaYears"]
    if categorical_columns is None:
        categorical_columns = CATEGORICAL_COLUMNS

    excluded = set(categorical_columns) | set(time_known)

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
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)
    y_test_scaled = y_scaler.transform(y_test)
    
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
        train_groups, val_groups
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
    var_pivoted = year_melted.pivot_table(
        index=['Model', 'Model_Family', 'Scenario', 'Scenario_Category', 'Region', 'Year'],
        columns='Variable', values='value'
    ).reset_index()
    return var_pivoted


def add_prev_features(
    group: pd.DataFrame,
    output_variables: list,
    n_lags: int = N_LAG_FEATURES,
    lag_required: bool = True,
) -> pd.DataFrame:
    """Augment a grouped dataframe with lagged target features."""

    # Ensure chronological order within each group before shifting
    group_sorted = group.sort_values('Year')

    combined_df: pd.DataFrame = group_sorted.copy()

    for lag in range(1, n_lags + 1):
        shifted = pd.DataFrame(group_sorted[output_variables].shift(lag))
        rename_map = {}
        for col in shifted.columns:
            prefix = 'prev_' if lag == 1 else f'prev{lag}_'
            rename_map[col] = f"{prefix}{col}"
        shifted = shifted.rename(columns=rename_map)
        for col in shifted.columns:
            combined_df[col] = shifted[col]

    if lag_required:
        # Drop the initial rows without a complete lag history
        combined_df = cast(pd.DataFrame, combined_df.iloc[n_lags:])

    return cast(pd.DataFrame, combined_df)


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

    prepared = data.groupby(INDEX_COLUMNS).apply(
        add_prev_features,
        output_variables=OUTPUT_VARIABLES,
        lag_required=lag_required,
    ).reset_index(drop=True)
    prepared = cast(pd.DataFrame, prepared)
    prepared['Year'] = prepared['Year'].astype(int)

    targets = OUTPUT_VARIABLES
    features = [col for col in prepared.columns if col not in NON_FEATURE_COLUMNS and col not in targets]

    # Only drop rows with missing targets, not lag features
    prepared = prepared.dropna(subset=targets)

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

    prepared = data.groupby(INDEX_COLUMNS).apply(
        add_prev_features,
        output_variables=OUTPUT_VARIABLES,
        lag_required=lag_required,
    ).reset_index(drop=True)
    prepared = cast(pd.DataFrame, prepared)
    prepared['Year'] = prepared['Year'].astype(int)

    targets = OUTPUT_VARIABLES
    features = [col for col in prepared.columns if col not in NON_FEATURE_COLUMNS and col not in targets]

    prepared = prepared.dropna(subset=targets)

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
    logging.info(f"Removing {mask.sum()} rows with missing outputs")

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
