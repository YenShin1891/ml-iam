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

# def autoregressive_predictions_per_target(models, target, group_indices, group_matrix, prev_indices, prev2_indices, y_test, start_pos):
#     preds_target = np.full(len(group_indices), np.nan, dtype=float)
#     preds_target[start_pos] = models[target].predict(group_matrix[start_pos:start_pos + 1])[0]

#     for t in range(start_pos + 1, len(group_indices)):
#         X_test_curr = group_matrix[t].copy()
#         if t - 1 >= start_pos:
#             X_test_curr[prev_indices] = preds_target[t - 1]
#         if t - 2 >= start_pos:
#             X_test_curr[prev2_indices] = preds_target[t - 2]
#         preds_target[t] = models[target].predict(X_test_curr.reshape(1, -1))[0]

#     return preds_target


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
# def test_xgb_per_target_autoregressively(models, X_test_with_index, y_test, target_names):
#     full_preds = np.full(y_test.shape, np.nan, dtype=float)

#     for i, target in enumerate(target_names):
#         group_indices_list, group_matrices, prev_indices_list, prev2_indices_list = group_test_data(X_test_with_index)

#         for group_indices, group_matrix, prev_indices, prev2_indices in zip(group_indices_list, group_matrices, prev_indices_list, prev2_indices_list):
#             start_pos = next((pos for pos, idx in enumerate(group_indices) if not np.isnan(y_test[idx, i])), None)
#             if start_pos is None:
#                 continue

#             preds_target = autoregressive_predictions_per_target(models, target, group_indices, group_matrix, prev_indices, prev2_indices, y_test, start_pos)
#             full_preds[group_indices, i] = preds_target

#         mask = ~np.isnan(y_test[:, i])
#         test_auc = mean_squared_error(y_pred=full_preds[mask, i], y_true=y_test[mask, i])
#         logging.info(f"Mean Square Error for {target}: {test_auc:.2f}")

#     return full_preds



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


def test_tft_autoregressively(model, X_test_with_index, y_test):
    """
    TFT 모델을 그룹별로 순차 예측 후 전체 결과 반환
    """
    full_preds = np.full(y_test.shape, np.nan, dtype=float)

    grouped = X_test_with_index.groupby(INDEX_COLUMNS)

    dataset_config = TFTDatasetConfig()  # Check if mode has to be 'eval'

    for group_key, group_df in tqdm(grouped, desc="TFT Group Inference"):
        group_df_sorted = group_df.sort_values(["Step"]).copy()

        # Determine targets from model and features from data
        targets = model.hparams.target
        if isinstance(targets, str):
            targets = [targets]
        features = [c for c in group_df_sorted.columns if c not in (NON_FEATURE_COLUMNS + targets)]

        params = dataset_config.build(features, targets, mode="eval")

        dataset = TimeSeriesDataSet(
            group_df_sorted,
            **params,
        )

        dataloader = dataset.to_dataloader(train=False, batch_size=64)
        predictions = model.predict(dataloader, mode="prediction")

        preds_arr = predictions.reshape(-1, model.hparams.output_size)

        # map back to rows for all targets
        group_indices = group_df_sorted.index.tolist()
        pos = X_test_with_index.index.get_indexer(group_indices)
        full_preds[pos, :preds_arr.shape[1]] = preds_arr

    mse = mean_squared_error(y_test, full_preds)
    logging.info(f"TFT Mean Squared Error: {mse:.4f}")

    return full_preds