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


def autoregressive_predictions(model, group_indices, group_matrix, prev_indices, prev2_indices, start_pos):
    num_targets = model.predict(group_matrix[start_pos:start_pos + 1]).shape[1]
    preds_target = np.full((len(group_indices), num_targets), np.nan, dtype=float)
    
    # Set initial predictions for all targets
    preds_target[start_pos, :] = model.predict(group_matrix[start_pos:start_pos + 1])[0]

    for t in range(start_pos + 1, len(group_indices)):
        X_test_curr = group_matrix[t].copy()
        if t - 1 >= start_pos:
            X_test_curr[prev_indices] = preds_target[t - 1, :]  # Use predictions from t-1 for all targets
        if t - 2 >= start_pos:
            X_test_curr[prev2_indices] = preds_target[t - 2, :]  # Use predictions from t-2 for all targets
        preds_target[t, :] = model.predict(X_test_curr.reshape(1, -1))[0]  # Predict for all targets

    return preds_target


def test_xgb_autoregressively(X_test_with_index, y_test, run_id=None, model=None, disable_progress=False, cache=None):
    """
    Test the model autoregressively on the test set.
    """
    if cache is None:
        cache = {}
        
    group_indices_list, group_matrices, prev_indices_list, prev2_indices_list = group_test_data(X_test_with_index, cache)
    full_preds = np.full(y_test.shape, np.nan, dtype=float)
    
    if model is None:
        model = xgb.XGBRegressor()
        model.load_model(os.path.join(RESULTS_PATH, run_id, "checkpoints", "final_best.json"))

    def process_group(args):
        group_indices, group_matrix, prev_indices, prev2_indices = args
        # With no nan values in y_test, we always use the first instance as seed.
        start_pos = 0
        preds_target = autoregressive_predictions(model, group_indices, group_matrix, prev_indices, prev2_indices, start_pos)
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
