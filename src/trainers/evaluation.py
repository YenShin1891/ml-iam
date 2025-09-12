from sklearn.metrics import mean_squared_error
import os
import logging
import numpy as np
import pandas as pd
import concurrent.futures
from tqdm import tqdm
import xgboost as xgb
from xgboost import DMatrix

from configs.paths import RESULTS_PATH
from configs.data import INDEX_COLUMNS, NON_FEATURE_COLUMNS, N_LAG_FEATURES

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
    
    # Create masks for all lag levels
    lag_masks = {}
    for lag in range(1, N_LAG_FEATURES + 1):
        if lag == 1:
            pattern = 'prev_'
        else:
            pattern = f'prev{lag}_'
        
        if hasattr(feature_columns, 'str'):  # pandas Index/Series
            lag_result = feature_columns.str.startswith(pattern)
            lag_masks[lag] = lag_result.values if hasattr(lag_result, 'values') else lag_result
        else:  # already a numpy array
            lag_masks[lag] = np.array([str(col).startswith(pattern) for col in feature_columns])
    
    grouped = X_test_with_index.groupby(INDEX_COLUMNS, sort=False)
    
    num_groups = grouped.ngroups
    group_indices_list = []
    group_matrices = []
    lag_indices_lists = {lag: [] for lag in range(1, N_LAG_FEATURES + 1)}
    
    group_keys = list(grouped.groups.keys())
    
    for group_key in group_keys:
        group_df = grouped.get_group(group_key)

        group_indices = group_df.index.tolist()

        group_matrix = group_df[feature_columns].to_numpy()
        
        group_indices_list.append(group_indices)
        group_matrices.append(group_matrix)

        # Append masks for all lag levels
        for lag in range(1, N_LAG_FEATURES + 1):
            lag_indices_lists[lag].append(lag_masks[lag])
    
    # Return tuple with all lag indices
    result_tuple = [group_indices_list, group_matrices]
    for lag in range(1, N_LAG_FEATURES + 1):
        result_tuple.append(lag_indices_lists[lag])
    result = tuple(result_tuple)
    cache[cache_key] = result
    
    return result


def autoregressive_predictions(model, group_indices, group_matrix, lag_indices_dict, start_pos, y_scaler=None, x_scaler=None, feature_columns=None):
    """
    Generate autoregressive predictions for a single grouped series.

    Notes:
    - Supports arbitrary N_LAG_FEATURES based on configs.data.N_LAG_FEATURES.
    - Does not rely on lag_indices_dict; uses feature column names to locate
      lagged feature columns of the form prev_<var> or prev{lag}_<var>.
    - Assumes model.predict returns a vector of targets aligned with
      OUTPUT_VARIABLES[:num_targets].
    """
    # Infer number of targets from a single prediction
    first_pred = model.predict(group_matrix[start_pos:start_pos + 1])
    # Ensure 2D shape (1, num_targets)
    if first_pred.ndim == 1:
        num_targets = first_pred.shape[0]
    else:
        num_targets = first_pred.shape[1]
    preds_target = np.full((len(group_indices), num_targets), np.nan, dtype=float)

    # Seed with initial prediction
    preds_target[start_pos, :] = first_pred.reshape(1, -1)[0]

    # Precompute lagged feature column indices once
    from configs.data import OUTPUT_VARIABLES
    out_vars = OUTPUT_VARIABLES[:num_targets]
    # Build a dict: lag -> list of column indices (or None) for each output variable
    lag_col_indices = {}
    if feature_columns is None:
        # If not provided, derive from matrix width assuming caller aligned order with X_test_with_index
        feature_columns = []
    for lag in range(1, N_LAG_FEATURES + 1):
        cols_for_lag = []
        for var in out_vars:
            col_name = (f"prev_{var}" if lag == 1 else f"prev{lag}_{var}")
            try:
                col_idx = feature_columns.index(col_name)
            except (ValueError, AttributeError):
                col_idx = None
            cols_for_lag.append(col_idx)
        lag_col_indices[lag] = cols_for_lag

    # If scalers are provided, precompute mean/scale for targets and lagged feature columns
    x_means_attr = getattr(x_scaler, 'mean_', None) if x_scaler is not None else None
    x_scales_attr = getattr(x_scaler, 'scale_', None) if x_scaler is not None else None
    y_means_attr = getattr(y_scaler, 'mean_', None) if y_scaler is not None else None
    y_scales_attr = getattr(y_scaler, 'scale_', None) if y_scaler is not None else None
    def _safe_len(x):
        try:
            return len(x)
        except Exception:
            return -1
    use_scalers = (
        isinstance(x_means_attr, (list, np.ndarray)) and isinstance(x_scales_attr, (list, np.ndarray)) and
        isinstance(y_means_attr, (list, np.ndarray)) and isinstance(y_scales_attr, (list, np.ndarray)) and
        _safe_len(x_means_attr) == _safe_len(feature_columns)
    )
    if use_scalers:
        y_means = np.asarray(y_means_attr)[:num_targets]
        y_scales = np.asarray(y_scales_attr)[:num_targets]
        # For each lag, align x_scaler stats to lag columns
        lag_x_means = {}
        lag_x_scales = {}
        x_means_full = np.asarray(x_means_attr)
        x_scales_full = np.asarray(x_scales_attr)
        for lag, cols in lag_col_indices.items():
            lag_x_means[lag] = np.array([(x_means_full[c] if c is not None else np.nan) for c in cols])
            lag_x_scales[lag] = np.array([(x_scales_full[c] if c is not None else np.nan) for c in cols])

    # Roll forward autoregressively
    for t in range(start_pos + 1, len(group_indices)):
        X_test_curr = group_matrix[t].copy()

        # Update lagged features using previous predictions
        for lag in range(1, N_LAG_FEATURES + 1):
            src_t = t - lag
            if src_t < start_pos:
                continue
            for pred_idx in range(num_targets):
                col_idx = lag_col_indices[lag][pred_idx]
                if col_idx is None:
                    continue
                y_val_scaled = preds_target[src_t, pred_idx]
                if use_scalers:
                    # y_scaled -> raw -> x_scaled(prev)
                    raw_val = y_val_scaled * y_scales[pred_idx] + y_means[pred_idx]
                    x_mean = lag_x_means[lag][pred_idx]
                    x_scale = lag_x_scales[lag][pred_idx]
                    if np.isfinite(x_mean) and np.isfinite(x_scale) and x_scale != 0:
                        x_val_scaled = (raw_val - x_mean) / x_scale
                        X_test_curr[col_idx] = x_val_scaled
                    else:
                        # Fallback: insert y-scaled value if stats are invalid
                        X_test_curr[col_idx] = y_val_scaled
                else:
                    # No scalers: assume spaces match and insert as-is
                    X_test_curr[col_idx] = y_val_scaled

        # Predict for all targets at time t
        next_pred = model.predict(X_test_curr.reshape(1, -1))
        preds_target[t, :] = next_pred.reshape(1, -1)[0]

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
        
    # group_test_data returns (group_indices_list, group_matrices, [optional lag masks...])
    _grouped = group_test_data(X_test_with_index, cache)
    group_indices_list, group_matrices = _grouped[0], _grouped[1]
    full_preds = np.full(y_test.shape, np.nan, dtype=float)
    
    if model is None:
        if run_id is None:
            raise ValueError("Either provide a preloaded `model` or a valid `run_id` to load from disk.")
        model = xgb.XGBRegressor()
        ckpt_path = os.path.join(RESULTS_PATH, run_id, "checkpoints", "final_best.json")
        model.load_model(ckpt_path)

    # Get feature column names
    feature_columns = [col for col in X_test_with_index.columns if col not in NON_FEATURE_COLUMNS]

    def process_group(args):
        group_indices, group_matrix = args
        # With no nan values in y_test, we always use the first instance as seed.
        start_pos = 0
        preds_target = autoregressive_predictions(model, group_indices, group_matrix, None, start_pos, y_scaler, x_scaler, feature_columns)
        return group_indices, preds_target

    index_to_pos = {idx: pos for pos, idx in enumerate(X_test_with_index.index)}
    groups = list(zip(group_indices_list, group_matrices))
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
