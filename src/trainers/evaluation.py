from sklearn.metrics import mean_squared_error
import os
import logging
import numpy as np
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from typing import List
from src.utils.utils import masked_mse
from configs.config import SPLIT_POINT, LAGGING, INDEX_COLUMNS, NON_FEATURE_COLUMNS, STATIC_FEATURE_COLUMNS, RESULTS_PATH

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
    
    Args:
        run_id (str): Unique identifier for the run.
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted values from the model.
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

    grouped = X_test_with_index.groupby(["Region", "Scenario"])
    
    for (region, scenario), group_df in tqdm(grouped, desc="TFT Group Inference"):
        
        group_df_sorted = group_df.sort_values("Year")
        
        dataset = TimeSeriesDataSet(
            group_df_sorted,
            time_idx="Year",
            target=model.hparams.target,  # 학습 당시 설정과 동일
            group_ids=["Region", "Scenario"],
            max_encoder_length=30,
            max_prediction_length=10,
            time_varying_known_reals=["Year"],
            time_varying_unknown_reals=[model.hparams.target],
            static_categoricals=["Region", "Model_Family"],
            add_relative_time_idx=True,
            add_target_scales=True,
        )

        dataloader = dataset.to_dataloader(train=False, batch_size=64)
        
        predictions = model.predict(dataloader, mode="prediction")

        # 결과 매핑
        group_indices = group_df_sorted.index.tolist()
        pos = X_test_with_index.index.get_indexer(group_indices)
        full_preds[pos, 0] = predictions.flatten()

    mse = mean_squared_error(y_test[:, 0], full_preds[:, 0])
    logging.info(f"TFT Mean Squared Error: {mse:.4f}")

    return full_preds

def _predict_series_fragment(trained_model, series_data, split_point=SPLIT_POINT):
    """
    Args:
        trained_model: Already trained MLForecast model
        series_data: Single time series data sorted by time
        split_point: Point to split the series into context and target
    
    Returns:
        predictions: DataFrame with predictions and actual values
    """
    if split_point < LAGGING or len(series_data) <= split_point:
        return None
        
    context_data = series_data.iloc[:split_point].copy()
    target_data = series_data.iloc[split_point:].copy()

    # MLForecast will use context to create lag features automatically
    h = len(target_data)
    predictions = trained_model.predict(h=h, df=context_data)
    
    # Merge predictions with actual values
    if len(predictions) > 0:
        series_predictions = predictions.merge(
            target_data[['unique_id', 'ds', 'y', 'target']], 
            on=['unique_id'], 
            how='left'
        )
        return series_predictions
    
    return None

def evaluate_mlforecast_fragment_based(trained_model, eval_data, targets, mode="test", model_type='xgb'):
    """
    Fragment-based evaluation using pre-trained model.
    
    Args:
        trained_model: Pre-trained MLForecast model
        eval_data: Data to evaluate (validation or test)
        targets: List of target variables
        mode: "validation" or "test" for logging purposes
    
    Returns:
        predictions: DataFrame with all predictions
        avg_mse: Average MSE across all series
    """
    logging.info(f"Running MLForecast {mode} with fragment-based forecasting...")
    
    all_predictions = []
    
    # Group data by unique_id to handle each series separately
    for unique_id, series_data in eval_data.groupby('unique_id'):
        series_data = series_data.sort_values('ds').reset_index(drop=True)
        
        try:
            predictions = _predict_series_fragment(trained_model, series_data)
            
            if predictions is not None:
                all_predictions.append(predictions)
                
                # Calculate series-level MSE
                valid_preds = predictions.dropna(subset=['y', model_type])
                    
        except Exception as e:
            logging.warning(f"Failed to predict series {unique_id}: {str(e)}", exc_info=True)
            continue
    
    if all_predictions:
        overall_predictions = pd.concat(all_predictions, ignore_index=True)
        
        # calculate average MSE across all series
        valid_preds = overall_predictions.dropna(subset=['y', model_type])
        if len(valid_preds) > 0:
            avg_mse = mean_squared_error(valid_preds['y'], valid_preds[model_type])
            logging.info(f"Average MSE: {avg_mse:.4f}")
        else:
            logging.warning("No valid predictions found for MSE calculation.")
        
        return overall_predictions, avg_mse
    else:
        logging.warning(f"No predictions generated for {mode}")
        return pd.DataFrame(), float('inf')
    

def test_mlforecast_recursive(model, test_data: pd.DataFrame, targets: List[str]):
    """
    Test MLForecast model with recursive forecasting.
    """
    logging.info("Testing MLForecast model recursively...")
    
    # Get unique series
    unique_series = test_data['unique_id'].unique()
    
    all_predictions = []
    
    for series_id in unique_series:
        series_data = test_data[test_data['unique_id'] == series_id].sort_values('ds')
        
        if len(series_data) == 0:
            continue
            
        # Predict for this series
        h = len(series_data)
        series_preds = model.predict(h=h, X_df=series_data[['unique_id', 'ds']])
        
        # Add actual values for comparison
        series_preds = series_preds.merge(
            series_data[['unique_id', 'ds', 'y', 'target']], 
            on=['unique_id', 'ds'], 
            how='left'
        )
        
        all_predictions.append(series_preds)
    
    if all_predictions:
        full_predictions = pd.concat(all_predictions, ignore_index=True)
        
        # Calculate overall MSE
        valid_preds = full_predictions.dropna(subset=['y', 'xgb'])
        if len(valid_preds) > 0:
            mse = mean_squared_error(valid_preds['y'], valid_preds['xgb'])
            logging.info(f"MLForecast Mean Squared Error: {mse:.4f}")
        
        return full_predictions
    else:
        logging.warning("No predictions generated")
        return pd.DataFrame()