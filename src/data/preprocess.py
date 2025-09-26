import os
import pandas as pd
import numpy as np
import logging

from configs.config import (
    DATA_PATH, RESULTS_PATH,
    DEFAULT_DATASET, MAX_YEAR,
    OUTPUT_VARIABLES, INDEX_COLUMNS, NON_FEATURE_COLUMNS
)


def split_data(prepared):
    groups = list(prepared.groupby(INDEX_COLUMNS))
    n_groups = len(groups)
    # split 8:1:1
    n_train_groups = int(n_groups * 0.8)
    n_val_groups = int(n_groups * 0.1)
    
    np.random.shuffle(groups)

    train_groups = groups[:n_train_groups]
    val_groups = groups[n_train_groups:n_train_groups + n_val_groups]
    test_groups = groups[n_train_groups + n_val_groups:]

    train_data = pd.concat([group[1] for group in train_groups]).reset_index(drop=True)
    val_data = pd.concat([group[1] for group in val_groups]).reset_index(drop=True)
    test_data = pd.concat([group[1] for group in test_groups]).reset_index(drop=True)

    return train_data, val_data, test_data


def encode_categorical_columns(data, columns = ['Region', 'Model_Family']):
    for col in columns:
        if col in data.columns:
            data[col] = data[col].astype('category').cat.codes


def prepare_data(prepared, targets, features):
    train_data, val_data, test_data = split_data(prepared)

    X_train = train_data[features].copy()
    y_train = train_data[targets].values.copy()
    X_val = val_data[features].copy()
    y_val = val_data[targets].values.copy()
    X_test_with_index = test_data[features + [col for col in INDEX_COLUMNS if col not in features]].copy()
    y_test = test_data[targets].values.copy()

    encode_categorical_columns(X_train)
    encode_categorical_columns(X_val)
    encode_categorical_columns(X_test_with_index)

    return X_train, y_train, X_val, y_val, X_test_with_index, y_test, test_data


def load_and_process_data(version=None) -> pd.DataFrame:
    logging.info("Loading and processing data...")
    # Load the dataset specified in configs.data.DEFAULT_DATASET or from version subdirectory.
    # If version is provided, use version/processed_series.csv, otherwise use DEFAULT_DATASET.
    if version:
        dataset_path = os.path.join(DATA_PATH, version, "processed_series.csv")
    else:
        dataset_path = os.path.join(DATA_PATH, DEFAULT_DATASET)
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Processed dataset not found: {dataset_path}. Update configs.data.DEFAULT_DATASET.")
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

    year_melted = processed_series.melt(
        id_vars=non_year_cols, value_vars=year_cols, var_name='Year', value_name='value'
    )
    var_pivoted = year_melted.pivot_table(
        index=['Model', 'Model_Family', 'Scenario', 'Scenario_Category', 'Region', 'Year'],
        columns='Variable', values='value'
    ).reset_index()
    return var_pivoted


def add_prev_outputs_twice(group: pd.DataFrame, output_variables: list) -> pd.DataFrame:
    prev_output_df = group[output_variables].shift(1)
    prev_output_df.columns = ['prev_' + col for col in prev_output_df.columns]
    prev_output_df2 = group[output_variables].shift(2)
    prev_output_df2.columns = ['prev2_' + col for col in prev_output_df2.columns]
    combined = pd.concat([group, prev_output_df, prev_output_df2], axis=1).iloc[2:]
    return combined


def prepare_features_and_targets(data: pd.DataFrame) -> tuple:
    logging.info("Preparing features and targets...")
    prepared = data.groupby(INDEX_COLUMNS).apply(
        add_prev_outputs_twice, output_variables = OUTPUT_VARIABLES
    ).reset_index(drop=True)
    prepared['Year'] = prepared['Year'].astype(int)
    
    targets = OUTPUT_VARIABLES
    features = [col for col in prepared.columns if col not in NON_FEATURE_COLUMNS and col not in targets]
    
    prepared.dropna(subset=targets, inplace=True)
    
    return prepared, features, targets


def prepare_features_and_targets_tft(data: pd.DataFrame) -> tuple:
    logging.info("Preparing features and targets for TFT...")
    prepared = data.copy()
    # Do not need lagging for TFT as it uses autoregressive prediction
    prepared['Year'] = prepared['Year'].astype(int)

    targets = OUTPUT_VARIABLES
    features = [col for col in prepared.columns if col not in NON_FEATURE_COLUMNS and col not in targets]
    
    prepared.dropna(subset=targets, inplace=True)

    # Make 'Step' and 'DeltaYears' after dropping NaNs
    # Step must align with group_ids used by TimeSeriesDataSet
    group_cols = INDEX_COLUMNS
    prepared.sort_values(group_cols + ['Year'], inplace=True)
    prepared['Step'] = prepared.groupby(group_cols).cumcount().astype('int64')

    # Explicit gap feature: years elapsed since previous observation within each series
    prepared['DeltaYears'] = (
        prepared.groupby(group_cols)['Year'].diff().fillna(0).astype(int)
    )
    
    return prepared, features, targets


# def prepare_features_and_targets_tft(data: pd.DataFrame) -> tuple:
#     logging.info("Preparing features and targets for TFT with full 5-year intervals...")
#     df = data.copy()
#     df['Year'] = df['Year'].astype(int)

#     targets = OUTPUT_VARIABLES
#     features = [col for col in df.columns if col not in NON_FEATURE_COLUMNS and col not in targets]
#     df.dropna(subset=targets, inplace=True)

#     group_cols = INDEX_COLUMNS
#     # Expand each group to full 5-year intervals
#     expanded = []
#     for _, group in df.groupby(group_cols):
#         min_year = group['Year'].min()
#         max_year = group['Year'].max()
#         # Generate all 5-year intervals between min and max year (inclusive)
#         all_years = np.arange(min_year, max_year + 1, 5)
#         # Build a DataFrame with all group keys and all years
#         group_keys = {col: group.iloc[0][col] for col in group_cols}
#         full_index = pd.DataFrame({**group_keys, 'Year': all_years})
#         # Merge to get existing data, missing years will be NaN
#         merged = pd.merge(full_index, group, on=group_cols + ['Year'], how='left')
#         expanded.append(merged)
#     prepared = pd.concat(expanded, ignore_index=True)

#     # Step must align with group_ids used by TimeSeriesDataSet
#     prepared['Step'] = prepared.groupby(group_cols).cumcount().astype('int64')

#     return prepared, features, targets


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
    else:
        return X, y


def add_missingness_indicators(
    prepared: pd.DataFrame,
    features: list,
    time_known: list = ["Year", "DeltaYears"],
    categorical_columns: list = None,
):
    """
    Add <col>_is_missing indicators to the full prepared dataframe BEFORE splitting.
    Indicators are stored as string categoricals ("0"/"1") for categorical encoding.
    Returns (prepared_with_indicators, updated_features).
    """
    if categorical_columns is None:
        from configs.config import CATEGORICAL_COLUMNS as CFG_CATS
        categorical_columns = CFG_CATS

    updated_features = list(features)

    cont_cols = [
        c for c in features
        if c not in set((categorical_columns or []) + list(time_known))
        and not c.endswith("_is_missing")
    ]

    for col in cont_cols:
        miss_col = f"{col}_is_missing"
        # create as string categoricals ("0"/"1")
        if miss_col not in prepared.columns:
            prepared[miss_col] = (
                prepared[col].isna().map({True: "1", False: "0"}).astype("category")
            )
        if miss_col not in updated_features:
            updated_features.append(miss_col)

    return prepared, updated_features


def impute_with_train_medians(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    features: list,
    time_known: list = ["Year", "DeltaYears"],
    categorical_columns: list = None,
):
    """
    Impute continuous feature NaNs in train/val/test with the train median of each column.
    Does NOT add indicators (use add_missingness_indicators before splitting).
    Returns (train_data, val_data, test_data).
    """
    if categorical_columns is None:
        from configs.config import CATEGORICAL_COLUMNS as CFG_CATS
        categorical_columns = CFG_CATS

    cont_cols = [
        c for c in features
        if c not in set((categorical_columns or []) + list(time_known))
        and not c.endswith("_is_missing")
    ]

    for col in cont_cols:
        if col not in train_data.columns:
            continue
        fill_value = pd.to_numeric(train_data[col], errors="coerce").median()
        if pd.isna(fill_value):
            logging.warning(f"Column '{col}' has all-NaN in train; filling with 0.0")
            fill_value = 0.0
        for df in (train_data, val_data, test_data):
            if col in df.columns:
                df[col] = df[col].fillna(fill_value)

    return train_data, val_data, test_data