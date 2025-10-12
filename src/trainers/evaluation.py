from sklearn.metrics import mean_squared_error
import os
import logging
import numpy as np
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from src.utils.utils import masked_mse
from configs.config import INDEX_COLUMNS, NON_FEATURE_COLUMNS, RESULTS_PATH
from pytorch_forecasting import TimeSeriesDataSet
from configs.models import TFTDatasetConfig


def group_test_data(X_test_with_index):
    """
    Groups the test data by the specified index columns (group each instance)
    """
    grouped = X_test_with_index.groupby(INDEX_COLUMNS)
    group_indices_list = []
    group_matrices = []
    prev_indices_list = []
    prev2_indices_list = []

    for group_key, group_df in grouped:
        group_indices = group_df.index.tolist()
        group_matrix = group_df.drop(columns=NON_FEATURE_COLUMNS, errors='ignore').to_numpy()

        group_indices_list.append(group_indices)
        group_matrices.append(group_matrix)

        reduced_columns = group_df.drop(columns=NON_FEATURE_COLUMNS, errors='ignore').columns
        prev_indices_list.append(reduced_columns.str.startswith('prev_'))
        prev2_indices_list.append(reduced_columns.str.startswith('prev2_'))

    return group_indices_list, group_matrices, prev_indices_list, prev2_indices_list


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


def test_xgb_autoregressively(model, X_test_with_index, y_test):
    group_indices_list, group_matrices, prev_indices_list, prev2_indices_list = group_test_data(X_test_with_index)
    full_preds = np.full(y_test.shape, np.nan, dtype=float)

    def process_group(args):
        group_indices, group_matrix, prev_indices, prev2_indices = args
        # With no nan values in y_test, we always use the first instance as seed.
        start_pos = 0
        preds_target = autoregressive_predictions(model, group_indices, group_matrix, prev_indices, prev2_indices, start_pos)
        return group_indices, preds_target

    groups = list(zip(group_indices_list, group_matrices, prev_indices_list, prev2_indices_list))
    futures = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for group in groups:
            futures.append(executor.submit(process_group, group))
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing groups"):
            group_indices, preds_target = future.result()
            pos = X_test_with_index.index.get_indexer(group_indices)
            full_preds[pos, :] = preds_target

    mse = mean_squared_error(y_test, full_preds)
    logging.info(f"Mean Squared Error: {mse:.2f}")

    return full_preds


def test_rnn(model, X_test, y_test):
    # Reshape data into 3D array for RNN
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test).reshape((X_test.shape[0], 1, X_test.shape[1]))
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)  # Convert to 2D if only one target variable

    # Make predictions using the trained RNN model
    preds = model.predict(X_test_scaled)
    test_auc = masked_mse(y_true=y_test,y_pred=preds)
    print("Mean Square Error:", f"{test_auc:.2f}")
    return preds


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


def save_quantile_metrics(run_id, y_true, y_pred_quantiles, quantiles=None):
    """
    Save quantile-specific metrics including coverage and calibration.

    Args:
        run_id: Experiment run identifier
        y_true: True values [n_samples, n_targets]
        y_pred_quantiles: Quantile predictions [n_samples, n_targets, n_quantiles] or [n_samples, n_quantiles]
        quantiles: List of quantile levels, defaults to [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    """
    if quantiles is None:
        quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]

    # Ensure consistent shapes
    if y_pred_quantiles.ndim == 2 and len(quantiles) == y_pred_quantiles.shape[1]:
        # Single target case: [n_samples, n_quantiles]
        y_pred_quantiles = y_pred_quantiles[:, np.newaxis, :]  # [n_samples, 1, n_quantiles]
        y_true = y_true.reshape(-1, 1) if y_true.ndim == 1 else y_true

    n_samples, n_targets, n_quantiles = y_pred_quantiles.shape

    quantile_metrics = []
    for target_idx in range(n_targets):
        for q_idx, q in enumerate(quantiles):
            q_pred = y_pred_quantiles[:, target_idx, q_idx]
            y_target = y_true[:, target_idx]

            # Coverage: percentage of true values below the quantile prediction
            coverage = np.mean(y_target <= q_pred)

            # Quantile loss
            residual = y_target - q_pred
            q_loss = np.mean(np.maximum(q * residual, (q - 1) * residual))

            quantile_metrics.append({
                "Run ID": run_id,
                "Target": target_idx,
                "Quantile": q,
                "Coverage": coverage,
                "Expected_Coverage": q,
                "Coverage_Error": abs(coverage - q),
                "Quantile_Loss": q_loss
            })

    # Calculate interval coverage for common prediction intervals
    intervals = [
        (0.1, 0.9, "80%"),   # 80% prediction interval
        (0.25, 0.75, "50%"), # 50% prediction interval
        (0.02, 0.98, "96%")  # 96% prediction interval
    ]

    interval_metrics = []
    for lower_q, upper_q, interval_name in intervals:
        if lower_q in quantiles and upper_q in quantiles:
            lower_idx = quantiles.index(lower_q)
            upper_idx = quantiles.index(upper_q)
            expected_coverage = upper_q - lower_q

            for target_idx in range(n_targets):
                lower_pred = y_pred_quantiles[:, target_idx, lower_idx]
                upper_pred = y_pred_quantiles[:, target_idx, upper_idx]
                y_target = y_true[:, target_idx]

                # Interval coverage
                in_interval = (y_target >= lower_pred) & (y_target <= upper_pred)
                coverage = np.mean(in_interval)

                # Average interval width
                avg_width = np.mean(upper_pred - lower_pred)

                interval_metrics.append({
                    "Run ID": run_id,
                    "Target": target_idx,
                    "Interval": interval_name,
                    "Coverage": coverage,
                    "Expected_Coverage": expected_coverage,
                    "Coverage_Error": abs(coverage - expected_coverage),
                    "Average_Width": avg_width
                })

    # Save metrics
    metrics_dir = os.path.join(RESULTS_PATH, run_id, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    quantile_df = pd.DataFrame(quantile_metrics)
    quantile_file = os.path.join(metrics_dir, "quantile_metrics.csv")
    quantile_df.to_csv(quantile_file, index=False)

    interval_df = pd.DataFrame(interval_metrics)
    interval_file = os.path.join(metrics_dir, "interval_metrics.csv")
    interval_df.to_csv(interval_file, index=False)

    logging.info("Quantile metrics saved to %s", quantile_file)
    logging.info("Interval metrics saved to %s", interval_file)
