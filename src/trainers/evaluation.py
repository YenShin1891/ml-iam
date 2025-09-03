from sklearn.metrics import mean_squared_error
import os
import logging
import numpy as np
import pandas as pd
import concurrent.futures
from tqdm import tqdm
import xgboost as xgb
from xgboost import DMatrix

from configs.config import INDEX_COLUMNS, NON_FEATURE_COLUMNS, RESULTS_PATH

def group_test_data(X_test_with_index, cache=None):
    """
    Groups the test data by the specified index columns (group each instance)
    Uses caching and more efficient operations for better performance.
    """
    if cache is None:
        cache = {}

    cache_key = (X_test_with_index.shape, tuple(INDEX_COLUMNS), tuple(NON_FEATURE_COLUMNS))
    
    if cache_key in cache:
        return cache[cache_key]
    
    feature_columns = X_test_with_index.drop(columns=NON_FEATURE_COLUMNS, errors='ignore').columns
    if hasattr(feature_columns, 'str'):  # pandas Index/Series
        prev_result = feature_columns.str.startswith('prev_')
        prev2_result = feature_columns.str.startswith('prev2_')
        prev_mask = prev_result.values if hasattr(prev_result, 'values') else prev_result
        prev2_mask = prev2_result.values if hasattr(prev2_result, 'values') else prev2_result
    else:  # already a numpy array
        prev_mask = np.array([str(col).startswith('prev_') for col in feature_columns])
        prev2_mask = np.array([str(col).startswith('prev2_') for col in feature_columns])
    
    grouped = X_test_with_index.groupby(INDEX_COLUMNS, sort=False)
    
    num_groups = grouped.ngroups
    group_indices_list = []
    group_matrices = []
    prev_indices_list = []
    prev2_indices_list = []
    
    group_keys = list(grouped.groups.keys())
    
    for group_key in group_keys:
        group_df = grouped.get_group(group_key)

        group_indices = group_df.index.tolist()

        group_matrix = group_df[feature_columns].to_numpy()
        
        group_indices_list.append(group_indices)
        group_matrices.append(group_matrix)

        prev_indices_list.append(prev_mask)
        prev2_indices_list.append(prev2_mask)
    
    result = (group_indices_list, group_matrices, prev_indices_list, prev2_indices_list)
    cache[cache_key] = result
    
    return result


def autoregressive_predictions(model, group_indices, group_matrix, prev_indices, prev2_indices, start_pos, y_scaler=None, x_scaler=None, feature_columns=None):
    num_targets = model.predict(group_matrix[start_pos:start_pos + 1]).shape[1]
    preds_target = np.full((len(group_indices), num_targets), np.nan, dtype=float)
    
    # Set initial predictions for all targets
    preds_target[start_pos, :] = model.predict(group_matrix[start_pos:start_pos + 1])[0]

    for t in range(start_pos + 1, len(group_indices)):
        X_test_curr = group_matrix[t].copy()
        if t - 1 >= start_pos:
            if y_scaler is not None and x_scaler is not None and feature_columns is not None:
                # Convert scaled predictions to raw values, then scale as features
                raw_preds = y_scaler.inverse_transform(preds_target[t - 1, :].reshape(1, -1))[0]
                # Create dummy data with just the target values to scale properly
                dummy_data = np.full((1, len(feature_columns)), x_scaler.mean_)  # Use feature means as baseline
                
                # Map predictions to their corresponding prev_* columns correctly
                prev_col_indices = [i for i, is_prev in enumerate(prev_indices) if is_prev]
                for pred_idx, raw_pred in enumerate(raw_preds):
                    if pred_idx < len(prev_col_indices):
                        col_idx = prev_col_indices[pred_idx]
                        dummy_data[0, col_idx] = raw_pred
                
                # Transform to get feature-scaled values - create DataFrame to avoid warnings
                dummy_df = pd.DataFrame(dummy_data, columns=feature_columns)
                scaled_data = x_scaler.transform(dummy_df)[0]
                X_test_curr[prev_indices] = scaled_data[prev_indices]
            else:
                X_test_curr[prev_indices] = preds_target[t - 1, :]  # Fallback to original behavior
        if t - 2 >= start_pos:
            if y_scaler is not None and x_scaler is not None and feature_columns is not None:
                # Convert scaled predictions to raw values, then scale as features  
                raw_preds = y_scaler.inverse_transform(preds_target[t - 2, :].reshape(1, -1))[0]
                # Create dummy data with just the target values to scale properly
                dummy_data = np.full((1, len(feature_columns)), x_scaler.mean_)  # Use feature means as baseline
                
                # Map predictions to their corresponding prev2_* columns correctly
                prev2_col_indices = [i for i, is_prev2 in enumerate(prev2_indices) if is_prev2]
                for pred_idx, raw_pred in enumerate(raw_preds):
                    if pred_idx < len(prev2_col_indices):
                        col_idx = prev2_col_indices[pred_idx]
                        dummy_data[0, col_idx] = raw_pred
                
                # Transform to get feature-scaled values - create DataFrame to avoid warnings
                dummy_df = pd.DataFrame(dummy_data, columns=feature_columns)
                scaled_data = x_scaler.transform(dummy_df)[0]
                X_test_curr[prev2_indices] = scaled_data[prev2_indices]
            else:
                X_test_curr[prev2_indices] = preds_target[t - 2, :]  # Fallback to original behavior
        preds_target[t, :] = model.predict(X_test_curr.reshape(1, -1))[0]  # Predict for all targets

    return preds_target


def test_xgb_autoregressively(X_test_with_index, y_test, run_id=None, model=None, disable_progress=False, cache=None, y_scaler=None, x_scaler=None):
    """
    Test the model autoregressively on the test set.
    """
    if cache is None:
        cache = {}
    
    # Load scalers if not provided and run_id is available
    if y_scaler is None and x_scaler is None and run_id is not None:
        try:
            from src.utils.utils import load_session_state
            y_scaler = load_session_state(run_id, "y_scaler.pkl")
            x_scaler = load_session_state(run_id, "x_scaler.pkl")
        except:
            logging.warning("Could not load scalers, falling back to original behavior")
            y_scaler = None
            x_scaler = None
        
    group_indices_list, group_matrices, prev_indices_list, prev2_indices_list = group_test_data(X_test_with_index, cache)
    full_preds = np.full(y_test.shape, np.nan, dtype=float)
    
    if model is None:
        model = xgb.XGBRegressor()
        model.load_model(os.path.join(RESULTS_PATH, run_id, "checkpoints", "final_best.json"))

    # Get feature column names
    feature_columns = [col for col in X_test_with_index.columns if col not in NON_FEATURE_COLUMNS]

    def process_group(args):
        group_indices, group_matrix, prev_indices, prev2_indices = args
        # With no nan values in y_test, we always use the first instance as seed.
        start_pos = 0
        preds_target = autoregressive_predictions(model, group_indices, group_matrix, prev_indices, prev2_indices, start_pos, y_scaler, x_scaler, feature_columns)
        return group_indices, preds_target

    index_to_pos = {idx: pos for pos, idx in enumerate(X_test_with_index.index)}
    groups = list(zip(group_indices_list, group_matrices, prev_indices_list, prev2_indices_list))
    futures = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for group in groups:
            futures.append(executor.submit(process_group, group))
            
        if disable_progress:
            for future in concurrent.futures.as_completed(futures):
                group_indices, preds_target = future.result()
                pos = X_test_with_index.index.get_indexer(group_indices)
                full_preds[pos, :] = preds_target
        else:
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing groups"):
                group_indices, preds_target = future.result()
                pos = [index_to_pos[idx] for idx in group_indices]
                full_preds[pos, :] = preds_target

    if not disable_progress:
        mse = mean_squared_error(y_test, full_preds)
        logging.info(f"Root Mean Squared Error: {np.sqrt(mse)}")

    return full_preds


def save_metrics(run_id, y_true, y_pred):
    """
    Save performance metrics to a CSV file under the specified run directory.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    pearson_corr = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]

    # Store the metrics
    metrics = pd.DataFrame({
        "Run ID": [run_id],
        "Mean Squared Error": [mse],
        "Pearson Correlation": [pearson_corr],
        "R2 Score": [r2],
        "MAE": [mae],
        "RMSE": [rmse],
    })

    metrics_dir = os.path.join(RESULTS_PATH, run_id, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_file = os.path.join(metrics_dir, "performance.csv")
    metrics.to_csv(metrics_file, index=False)
    logging.info("Metrics saved to %s.", metrics_file)
