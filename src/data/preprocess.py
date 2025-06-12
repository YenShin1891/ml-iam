import os
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler

from configs.config import (
    DATA_PATH,
    YEAR_STARTS_AT, DATASET_NAME,
    OUTPUT_VARIABLES, INDEX_COLUMNS, NON_FEATURE_COLUMNS
)

def split_data(prepared):
    groups = list(prepared.groupby(INDEX_COLUMNS))
    n_groups = len(groups)
    # split 6:2:2
    n_train_groups = int(n_groups * 0.6)
    n_val_groups = int(n_groups * 0.2)
    
    np.random.shuffle(groups)

    train_groups = groups[:n_train_groups]
    val_groups = groups[n_train_groups:n_train_groups + n_val_groups]
    test_groups = groups[n_train_groups + n_val_groups:]

    train_data = pd.concat([group[1] for group in train_groups]).reset_index(drop=True)
    val_data = pd.concat([group[1] for group in val_groups]).reset_index(drop=True)
    test_data = pd.concat([group[1] for group in test_groups]).reset_index(drop=True)
    
    logging.info(f"Train: {len(train_data)} rows, Validation: {len(val_data)} rows, Test: {len(test_data)} rows")

    return train_data, val_data, test_data

def encode_categorical_columns(data, columns):
    for col in columns:
        if col in data.columns:
            data[col] = data[col].astype('category').cat.codes
    return data

def prepare_data(prepared, targets, features):
    train_data, val_data, test_data = split_data(prepared)

    X_train = train_data[features].copy()
    y_train = train_data[targets].values.copy()
    X_val = val_data[features].copy()
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

    return (
        X_train_scaled, y_train_scaled,
        X_val_scaled, y_val_scaled,
        X_test_with_index_scaled, y_test_scaled,
        test_data,
        x_scaler, y_scaler
    )

def load_and_process_data() -> pd.DataFrame:
    logging.info("Loading and processing data...")
    processed_series = pd.read_csv(os.path.join(DATA_PATH, DATASET_NAME))
    processed_series = processed_series.loc[:, :'2100']
    year_columns = processed_series.columns[YEAR_STARTS_AT:]
    non_year_columns = processed_series.columns[:YEAR_STARTS_AT]
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