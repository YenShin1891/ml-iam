import os
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler

from configs.config import (
    DATA_PATH, DATASET_NAME, RESULTS_PATH,
    YEAR_RANGE,
    OUTPUT_VARIABLES, INDEX_COLUMNS, NON_FEATURE_COLUMNS
)

def split_data(prepared, test_size=0.1):
    groups = list(prepared.groupby(INDEX_COLUMNS))
    n_groups = len(groups)
    n_train_groups = int(n_groups * (1 - test_size))
    
    np.random.shuffle(groups)

    train_groups = groups[:n_train_groups]
    test_groups = groups[n_train_groups:]

    train_data = pd.concat([group[1] for group in train_groups]).reset_index(drop=True)
    test_data = pd.concat([group[1] for group in test_groups]).reset_index(drop=True)
    
    logging.info(f"Train: {len(train_data)} rows, Test: {len(test_data)} rows")

    return train_data, test_data



def encode_categorical_columns(data, columns):
    for col in columns:
        if col in data.columns:
            data[col] = data[col].astype('category').cat.codes
    return data


def prepare_data(prepared, targets, features):
    train_data, test_data = split_data(prepared)

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

def load_and_process_data() -> pd.DataFrame:
    logging.info("Loading and processing data...")
    processed_series = pd.read_csv(os.path.join(DATA_PATH, DATASET_NAME))
    processed_series = processed_series.loc[:, :'2100']
    non_year_columns = processed_series.loc[:, :'1990']  # First year column is '1990'
    year_columns = processed_series.loc[:, str(YEAR_RANGE[0]):str(YEAR_RANGE[1])]
    
    # create a heatmap to visualize missing values
    import seaborn as sns
    import matplotlib.pyplot as plt
    # index is the variable names
    year_columns.index = processed_series['Variable']
    plt.figure(figsize=(20, 80))
    sns.heatmap(year_columns.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.xlabel('Year')
    plt.ylabel('Variable')
    plt.show()
    plt.savefig(os.path.join(RESULTS_PATH, 'missing_values_heatmap.png'))

    year_melted = processed_series.melt(
        id_vars=non_year_columns, value_vars=year_columns,
        var_name='Year', value_name='value'
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
            y = y[mask].reset_index(drop=True)
