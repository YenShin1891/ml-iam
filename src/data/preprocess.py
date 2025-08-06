import os
import pandas as pd
import numpy as np
import logging

from configs.config import (
    DATA_PATH, DATASET_NAME,
    YEAR_RANGE,
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
    X_test_with_index = test_data[features + [col for col in INDEX_COLUMNS if col not in features]].copy()
    y_test = test_data[targets].values.copy()

    categorical_columns = ['Region', 'Model_Family']
    X_train = encode_categorical_columns(X_train, categorical_columns)
    X_val = encode_categorical_columns(X_val, categorical_columns)
    X_test_with_index = encode_categorical_columns(X_test_with_index, categorical_columns)

    return X_train, y_train, X_val, y_val, X_test_with_index, y_test, test_data

def load_and_process_data() -> pd.DataFrame:
    logging.info("Loading and processing data...")
    processed_series = pd.read_csv(os.path.join(DATA_PATH, DATASET_NAME))
    non_year_columns = processed_series.loc[:, :'1990']  # First year column is '1990'
    year_columns = processed_series.loc[:, str(YEAR_RANGE[0]):str(YEAR_RANGE[1])]
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


def prepare_features_and_targets_tft(data: pd.DataFrame) -> tuple:
    logging.info("Preparing features and targets...")
    prepared = data.groupby(INDEX_COLUMNS).apply(
        add_prev_outputs_twice, output_variables=OUTPUT_VARIABLES
    ).reset_index(drop=True)

    prepared['Year'] = prepared['Year'].astype(int)
    targets = OUTPUT_VARIABLES

    group_id_columns = ['Model', 'Model_Family', 'Scenario', 'Scenario_Category', 'Region']
    features = [col for col in prepared.columns if col not in NON_FEATURE_COLUMNS and col not in targets]
    features = list(set(features) | set(group_id_columns))  # 중복 제거 포함

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
    

def prepare_features_and_targets_mlforecast(data: pd.DataFrame) -> tuple:
    """
    Prepare features and targets for MLForecast (without manual lagging).
    """
    logging.info("Preparing features and targets for MLForecast...")
    
    # Don't add previous outputs - MLForecast will handle this
    prepared = data.copy()
    prepared['Year'] = prepared['Year'].astype(int)
    
    targets = OUTPUT_VARIABLES
    
    # Features exclude targets and non-feature columns, but keep static features
    features = [col for col in prepared.columns if col not in NON_FEATURE_COLUMNS and col not in targets]
    
    return prepared, features, targets


def split_data_mlforecast(prepared, features, targets, index_columns):
    """
    Split data for MLForecast training by groups, keeping time series intact.
    """
    from src.trainers.mlf_trainer import prepare_mlforecast_data
    
    # Prepare data in MLForecast format
    mlf_data = prepare_mlforecast_data(prepared, features, targets, index_columns)
    
    # Extract group_id from unique_id by removing target suffix
    # unique_id format: "index_columns_target", so we remove "_target" part
    mlf_data['group_id'] = mlf_data['unique_id'].str.rsplit('_', n=1).str[0]
    
    # Get unique groups (series) instead of splitting by time
    unique_groups = mlf_data['group_id'].unique()
    n_groups = len(unique_groups)
    
    # Shuffle groups for random split
    np.random.shuffle(unique_groups)
    
    # Split groups: 70% train, 20% validation, 10% test
    n_train_groups = int(n_groups * 0.7)
    n_val_groups = int(n_groups * 0.2)
    
    train_groups = unique_groups[:n_train_groups]
    val_groups = unique_groups[n_train_groups:n_train_groups + n_val_groups]
    test_groups = unique_groups[n_train_groups + n_val_groups:]
    
    # Filter data by group assignments
    train_data = mlf_data[mlf_data['group_id'].isin(train_groups)].reset_index(drop=True)
    val_data = mlf_data[mlf_data['group_id'].isin(val_groups)].reset_index(drop=True)
    test_data = mlf_data[mlf_data['group_id'].isin(test_groups)].reset_index(drop=True)
    
    # Drop the temporary group_id column
    train_data = train_data.drop('group_id', axis=1)
    val_data = val_data.drop('group_id', axis=1)
    test_data = test_data.drop('group_id', axis=1)
    
    logging.info(f"MLForecast split - Train groups: {len(train_groups)}, Val groups: {len(val_groups)}, Test groups: {len(test_groups)}")
    
    return train_data, val_data, test_data