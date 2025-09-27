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


def _ensure_5year_intervals(data: pd.DataFrame, group_cols: list, targets: list) -> pd.DataFrame:
    """
    Ensure every group has data for every 5-year interval by adding missing rows with NaN targets.
    """
    logging.info("Ensuring 5-year intervals with NaN filling...")

    # Get the full range of years across all data
    min_year = data['Year'].min()
    max_year = data['Year'].max()

    # Create 5-year intervals
    year_intervals = list(range(min_year, max_year + 1, 5))
    if year_intervals[-1] < max_year:
        year_intervals.append(max_year)

    # Get unique groups
    unique_groups = data[group_cols].drop_duplicates()

    # Create complete grid of all group × year combinations
    complete_grid = []
    for _, group_row in unique_groups.iterrows():
        for year in year_intervals:
            row_dict = group_row.to_dict()
            row_dict['Year'] = year
            complete_grid.append(row_dict)

    complete_df = pd.DataFrame(complete_grid)

    # Merge with original data to keep existing rows and identify missing ones
    merged = complete_df.merge(data, on=group_cols + ['Year'], how='left', indicator=True)

    # For rows that exist in original data, use original values
    existing_mask = merged['_merge'] == 'both'

    # For missing rows, fill with appropriate values
    missing_mask = merged['_merge'] == 'left_only'

    if missing_mask.any():
        # Get template values for each group (use first available row)
        group_templates = data.groupby(group_cols).first().reset_index()

        # For missing rows, fill non-target columns with group template values
        for _, template in group_templates.iterrows():
            group_filter = True
            for col in group_cols:
                group_filter &= (merged[col] == template[col])

            row_mask = missing_mask & group_filter
            if row_mask.any():
                for col in data.columns:
                    if col not in group_cols + ['Year'] + targets:
                        merged.loc[row_mask, col] = template[col]
                    elif col in targets:
                        merged.loc[row_mask, col] = np.nan

    # Remove merge indicator and sort
    result = merged.drop('_merge', axis=1).sort_values(group_cols + ['Year']).reset_index(drop=True)

    logging.info(f"Added missing rows: {len(result) - len(data)} new rows created")
    return result


def prepare_features_and_targets_tft(data: pd.DataFrame) -> tuple:
    logging.info("Preparing features and targets for TFT...")
    prepared = data.copy()
    # Do not need lagging for TFT as it uses autoregressive prediction
    prepared['Year'] = prepared['Year'].astype(int)

    targets = OUTPUT_VARIABLES

    # Create 5-year intervals with NaN filling for missing years
    group_cols = INDEX_COLUMNS
    prepared = _ensure_5year_intervals(prepared, group_cols, targets)

    features = [col for col in prepared.columns if col not in NON_FEATURE_COLUMNS and col not in targets]

    # Don't drop NaN targets anymore - we'll use a loss mask instead
    # prepared.dropna(subset=targets, inplace=True)

    # Make 'Step' without dropping NaNs first
    # Step must align with group_ids used by TimeSeriesDataSet
    prepared.sort_values(group_cols + ['Year'], inplace=True)
    prepared['Step'] = prepared.groupby(group_cols).cumcount().astype('int64')

    # Remove DeltaYears - no longer needed

    return prepared, features, targets


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
    time_known: list = None,
    categorical_columns: list = None,
):
    """
    Add <col>_is_missing indicators to the full prepared dataframe BEFORE splitting.
    Indicators are stored as string categoricals ("0"/"1") for categorical encoding.
    Returns (prepared_with_indicators, updated_features).
    """
    if time_known is None:
        time_known = ["Year"]
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
    time_known: list = None,
    categorical_columns: list = None,
):
    """
    Impute continuous feature NaNs in train/val/test with the train median of each column.
    Does NOT add indicators (use add_missingness_indicators before splitting).
    Returns (train_data, val_data, test_data).
    """
    if time_known is None:
        time_known = ["Year"]
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