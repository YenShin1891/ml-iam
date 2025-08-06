from sklearn.metrics import mean_squared_error
import os
import logging
import numpy as np
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from typing import List
from src.utils.utils import masked_mse
from configs.config import LAGGING, INDEX_COLUMNS, NON_FEATURE_COLUMNS, STATIC_FEATURE_COLUMNS, RESULTS_PATH

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
#         logging.info(f"Mean Squared Error for {target}: {test_auc:.2f}")

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


class MLForecastContextPredictor:
    """
    Wrapper class to handle lag feature computation for MLForecast predictions
    without retraining the underlying model.
    """

    def __init__(self, trained_model):
        self.trained_model = trained_model
        self.static_features = STATIC_FEATURE_COLUMNS
        
        # Extract lags configuration from trained model
        self.lags = self._extract_lags_config(trained_model)
        self.lag_features = self._extract_lag_features_config(trained_model)
        
        logging.info(f"Extracted lags: {self.lags}")
        if isinstance(self.lag_features, list):
            logging.info(f"Extracted transform features: {self.lag_features}")
        else:
            logging.info(f"Extracted lag_features: {self.lag_features}")
        
    def _extract_lags_config(self, model):
        """Extract lags configuration from trained MLForecast model."""
        # Try different ways to get lags
        lags = getattr(model, 'lags', [])
        if lags:
            return lags
            
        # Check if lags are in lag_transforms keys
        lag_transforms = getattr(model, 'lag_transforms', {})
        if lag_transforms:
            return list(lag_transforms.keys())
            
        # Check ts (time series) object 
        ts = getattr(model, 'ts', None)
        if ts and hasattr(ts, 'lags'):
            return getattr(ts, 'lags', [])
            
        return []
    
    def _extract_lag_features_config(self, model):
        """Extract lag features from trained model."""
        # Get the actual transform names from the trained model
        ts = getattr(model, 'ts', None)
        if ts and hasattr(ts, 'transforms'):
            transforms = getattr(ts, 'transforms', {})
            # Extract transform names that aren't simple target lags
            transform_names = [name for name in transforms.keys() if not name.startswith('lag') or '_' in name]
            if transform_names:
                return transform_names
                        
        # If model uses lag_transforms, get all input features
        lag_transforms = getattr(model, 'lag_transforms', {})
        if lag_transforms:
            return "ALL_INPUT_FEATURES"  # Special marker
                
        # Try traditional lag_features approach
        lag_features = getattr(model, 'lag_features', [])
        if lag_features:
            return lag_features
            
        return []
    
    def _create_future_dataframe_for_new_series(self, unique_id, target_data):
        """
        Create a future dataframe structure for a new series not seen during training.
        """
        h = len(target_data)
        
        # Create the basic structure that MLForecast expects
        future_df = pd.DataFrame({
            'unique_id': [unique_id] * h,
            'ds': target_data['ds'].values
        })
        
        return future_df

    def _compute_lag_features(self, series_data):
        """
        Manually compute lag features for new series, replicating MLForecast's logic.
        This handles both target and feature lags.
        """
        data_with_lags = series_data.copy()

        # Add target lags (MLForecast uses 'lag{i}' format)
        for lag in self.lags:
            lag_col = f'lag{lag}'
            data_with_lags[lag_col] = data_with_lags['y'].shift(lag)

        # Handle feature transforms
        if isinstance(self.lag_features, list) and len(self.lag_features) > 0:
            # Extract input features for transformation
            exclude_cols = ['unique_id', 'ds', 'y', 'target'] + [col for col in self.static_features if col in data_with_lags.columns]
            input_features = [col for col in data_with_lags.columns if col not in exclude_cols]
            
            # Apply transforms to each input feature
            feature_transforms_count = 0
            for transform_name in self.lag_features:
                if transform_name.startswith('lag'):
                    continue  # Skip simple target lags, already handled above
                    
                # Create transform for each input feature
                for feature in input_features:
                    if feature in data_with_lags.columns:
                        transform_col = f'{transform_name}_{feature}'
                        
                        # Extract lag number from transform name (e.g., "rolling_mean_lag1_window_size1" -> 1)
                        if 'lag1' in transform_name:
                            data_with_lags[transform_col] = data_with_lags[feature].shift(1)
                        elif 'lag2' in transform_name:
                            data_with_lags[transform_col] = data_with_lags[feature].shift(2)
                        # Add more lag numbers as needed
                        
                        feature_transforms_count += 1
            
            logging.info(f"Created {len(self.lags)} target lags and {feature_transforms_count} feature transforms")
            
        elif self.lag_features == "ALL_INPUT_FEATURES":
            # Legacy approach for simple lag features
            exclude_cols = ['unique_id', 'ds', 'y', 'target'] + [col for col in self.static_features if col in data_with_lags.columns]
            feature_candidates = [col for col in data_with_lags.columns if col not in exclude_cols]
            
            # Add simple feature lags
            for feature in feature_candidates:
                if feature in data_with_lags.columns:
                    for lag_value in self.lags:
                        lag_col = f'lag{lag_value}_{feature}'
                        data_with_lags[lag_col] = data_with_lags[feature].shift(lag_value)
            
            logging.info(f"Created {len(self.lags)} target lags and {len(feature_candidates) * len(self.lags)} simple feature lags")

        lag_columns = [col for col in data_with_lags.columns if col.startswith('lag') or 'rolling_mean_lag' in col]
        target_lags = [col for col in lag_columns if not '_' in col or col.startswith('lag') and col.count('_') == 0]
        feature_lags = [col for col in lag_columns if col not in target_lags]
        
        logging.info(f"Total: {len(target_lags)} target lags, {len(feature_lags)} feature lags")

        return data_with_lags


    def predict_series(self, series_data):
        """
        Predict for a completely new series using manual lag computation.
        Works for series not seen during training.
        """
        series_data = series_data.sort_values('ds').reset_index(drop=True)
        unique_id = series_data['unique_id'].iloc[0]

        if len(series_data) <= LAGGING:
            return None

        # Manually compute lag features on the full series
        data_with_lags = self._compute_lag_features(series_data)
        
        # Split into context and target periods
        target_data = data_with_lags.iloc[LAGGING:].copy()
        h = len(target_data)
        
        # Create future dataframe for this new series
        future_df = self._create_future_dataframe_for_new_series(unique_id, target_data)
        
        # Collect all columns to add at once to avoid fragmentation
        columns_to_add = {}
        
        # Add all the computed lag features to future_df
        lag_columns = [col for col in target_data.columns if col.startswith('lag') or 'rolling_mean_lag' in col]
        for col in lag_columns:
            columns_to_add[col] = target_data[col].values
            
        # Add non-lag features (but exclude target 'y' and unique_id, ds)
        exclude_cols = ['unique_id', 'ds', 'y', 'target'] + [col for col in self.static_features if col in target_data.columns]
        feature_cols = [col for col in target_data.columns 
                       if col not in exclude_cols and not col.startswith('lag') and 'rolling_mean_lag' not in col]
        
        for col in feature_cols:
            columns_to_add[col] = target_data[col].values
        
        # Add all columns at once to avoid fragmentation
        for col, values in columns_to_add.items():
            future_df[col] = values
            
        logging.info(f"Future_df for new series {unique_id}: shape={future_df.shape}, columns={future_df.columns.tolist()}")
        

        # Check static features specifically
        expected_static = getattr(self.trained_model.ts, 'static_features', [])
        logging.info(f"Model expects static features: {expected_static}")
        for static_feat in expected_static:
            if static_feat in future_df.columns:
                logging.info(f"Static feature '{static_feat}' values: {future_df[static_feat].unique()}")
            else:
                logging.warning(f"Missing static feature: {static_feat}")
        
        # Check time structure
        logging.info(f"Time range in future_df: {future_df['ds'].min()} to {future_df['ds'].max()}")
        logging.info(f"Frequency: {future_df['ds'].diff().iloc[1:]}")
        
        # Detailed debugging - compare with expected structure
        try:
            expected_future = self.trained_model.make_future_dataframe(h)
            expected_cols = set(expected_future.columns)
            actual_cols = set(future_df.columns)
            
            missing_cols = expected_cols - actual_cols
            extra_cols = actual_cols - expected_cols
            
            logging.info(f"Expected future_df has {len(expected_cols)} columns")
            logging.info(f"Our future_df has {len(actual_cols)} columns") 
            
            if missing_cols:
                logging.error(f"Missing {len(missing_cols)} columns: {list(missing_cols)}")
            if extra_cols:
                logging.info(f"Extra {len(extra_cols)} columns (expected for new series): {list(extra_cols)[:3]}...")
                
        except Exception as e:
            logging.warning(f"Could not get expected future structure (expected for new series): {e}")
            
        # Try prediction with more detailed error handling
        try:
            # Try to get the missing combinations specifically  
            missing = self.trained_model.get_missing_future(h, future_df)
            if not missing.empty:
                logging.error(f"Missing combinations shape: {missing.shape}")
                logging.error(f"Missing combinations columns: {missing.columns.tolist()}")
                logging.error(f"Missing combinations sample: {missing.head()}")
            else:
                logging.info("No missing combinations detected by get_missing_future")
        except Exception as debug_e:
            logging.error(f"Could not get missing future info: {debug_e}")
        
        try:
            predictions = self.trained_model.predict(h=h, X_df=future_df)
            
            # Merge with actual values
            target_predictions = predictions.merge(
                target_data[['unique_id', 'ds', 'y'] + 
                           [col for col in self.static_features if col in target_data.columns]], 
                on=['unique_id', 'ds'], 
                how='inner'
            )
            
            return target_predictions
            
        except Exception as e:
            # Try to get missing combinations info
            try:
                missing = self.trained_model.get_missing_future(h, future_df)
                logging.error(f"Missing future combinations: {missing}")
            except Exception as debug_e:
                logging.debug(f"Could not get missing future info: {debug_e}")
            
            logging.error(f"Prediction failed for {unique_id}: {str(e)}")
            raise


def evaluate_mlforecast_with_context(trained_model, eval_data, targets, mode="test", model_type='xgb'):
    """
    Evaluate MLForecast using pre-trained model with context data for lag features.

    Args:
        trained_model: Pre-trained MLForecast model
        eval_data: Data to evaluate (validation or test)
        targets: List of target variables
        mode: "validation" or "test" for logging purposes
        model_type: Type of model used (e.g., 'xgb')

    Returns:
        predictions: DataFrame with all predictions
        avg_mse: Average MSE across all series
    """
    logging.info(f"Running MLForecast {mode} with context-based forecasting...")

    # Create context predictor wrapper
    context_predictor = MLForecastContextPredictor(trained_model)

    all_predictions = []

    # Group data by unique_id to handle each series separately
    for unique_id, series_data in eval_data.groupby('unique_id'):
        try:
            # Let the context predictor handle splitting and prediction
            predictions = context_predictor.predict_series(series_data)

            if predictions is not None and len(predictions) > 0:
                all_predictions.append(predictions)

        except Exception as e:
            logging.warning(f"Failed to predict series {unique_id}: {str(e)}", exc_info=True)
            raise

    if all_predictions:
        overall_predictions = pd.concat(all_predictions, ignore_index=True)

        # Calculate average MSE across all series
        valid_preds = overall_predictions.dropna(subset=['y', model_type])
        if len(valid_preds) > 0:
            avg_mse = mean_squared_error(valid_preds['y'], valid_preds[model_type])
            logging.info(f"Average MSE: {avg_mse:.4f}")
        else:
            logging.warning("No valid predictions found for MSE calculation.")
            avg_mse = float('inf')

        return overall_predictions, avg_mse
    else:
        logging.warning(f"No predictions generated for {mode}")
        return pd.DataFrame(), float('inf')