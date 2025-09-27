import os
import pandas as pd
import numpy as np
import logging
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
)

def split_data(prepared, test_size=0.1, val_size=0.1):
    groups = list(prepared.groupby(INDEX_COLUMNS))
    n_groups = len(groups)
    n_test_groups = int(n_groups * test_size)
    n_val_groups = int(n_groups * val_size)
    n_train_groups = n_groups - n_test_groups - n_val_groups
    
    np.random.shuffle(groups)

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
            data[col] = data[col].astype('category').cat.codes
    return data


def add_missingness_indicators(df: pd.DataFrame, features: list):
    """Add binary _is_missing indicators for the given feature columns.

    Returns updated (df, features) where features includes the new indicator columns.
    """
    indicators = df[features].isna().astype(int)
    indicators.columns = [f"{c}_is_missing" for c in indicators.columns]

    # Convert indicators to strings for TFT compatibility
    # TFT expects categorical columns to be strings, not numeric
    indicators = indicators.astype(str)

    df = pd.concat([df, indicators], axis=1)
    return df, features + list(indicators.columns)


def impute_with_train_medians(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, features: list):
    """Impute NaNs in features using medians computed on the training split.

    Returns (train_df_imputed, val_df_imputed, test_df_imputed).
    """
    medians = train_df[features].median(numeric_only=True)
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df[features] = train_df[features].fillna(medians)
    val_df[features] = val_df[features].fillna(medians)
    test_df[features] = test_df[features].fillna(medians)
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

    categorical_columns = ['Region', 'Model_Family']
    X_train = encode_categorical_columns(X_train, categorical_columns)
    X_val = encode_categorical_columns(X_val, categorical_columns)
    X_test = encode_categorical_columns(X_test, categorical_columns)
    
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
    
    train_groups = train_data[INDEX_COLUMNS].astype(str).agg('_'.join, axis=1).values
    val_groups = val_data[INDEX_COLUMNS].astype(str).agg('_'.join, axis=1).values

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


def add_prev_features(group: pd.DataFrame, output_variables: list, n_lags: int = N_LAG_FEATURES) -> pd.DataFrame:
    # Create configurable number of lagged features for output variables
    lagged_dfs = [group]  # Start with original data
    
    for lag in range(1, n_lags + 1):
        lagged_df = group[output_variables].shift(lag)
        if lag == 1:
            lagged_df.columns = ['prev_' + col for col in lagged_df.columns]
        else:
            lagged_df.columns = [f'prev{lag}_' + col for col in lagged_df.columns]
        lagged_dfs.append(lagged_df)
    
    # Combine original data with all lagged features, drop first n_lags rows (NaN due to shifts)
    combined = pd.concat(lagged_dfs, axis=1).iloc[n_lags:]
    return combined


def prepare_features_and_targets(data: pd.DataFrame) -> tuple:
    logging.info("Preparing features and targets...")
    prepared = data.groupby(INDEX_COLUMNS).apply(
        add_prev_features, output_variables = OUTPUT_VARIABLES
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


def prepare_features_and_targets_sequence(data: pd.DataFrame) -> tuple:
    """
    Prepare features and targets for sequence models (TFT, LSTM, etc.).

    This function does NOT add lagged features as sequence models handle
    temporal dependencies internally. It adds Step and DeltaYears for
    time series indexing.
    """
    logging.info("Preparing features and targets for sequence models...")
    prepared = data.copy()
    # Do not need lagging for sequence models as they use autoregressive prediction
    prepared['Year'] = prepared['Year'].astype(int)

    targets = OUTPUT_VARIABLES
    features = [col for col in prepared.columns if col not in NON_FEATURE_COLUMNS and col not in targets]

    prepared.dropna(subset=targets, inplace=True)

    # Make 'Step' and 'DeltaYears' after dropping NaNs
    # Step must align with group_ids used by sequence models
    group_cols = INDEX_COLUMNS
    prepared.sort_values(group_cols + ['Year'], inplace=True)
    prepared['Step'] = prepared.groupby(group_cols).cumcount().astype('int64')

    # Explicit gap feature: years elapsed since previous observation within each series
    prepared['DeltaYears'] = (
        prepared.groupby(group_cols)['Year'].diff().fillna(0).astype(int)
    )

    return prepared, features, targets


def prepare_features_and_targets_tft(data: pd.DataFrame) -> tuple:
    logging.info("Preparing features and targets for TFT...")
    prepared = data.copy()
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
