import logging
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from mlforecast import MLForecast
from mlforecast.utils import PredictionIntervals
from mlforecast.lag_transforms import RollingMean
import os
from configs.config import RESULTS_PATH, LAGGING, STATIC_FEATURE_COLUMNS

PARAM_DIST = {
    'max_depth': [8, 10, 12],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [300, 500, 700],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5],
    'reg_alpha': [0.1, 1, 10],
    'reg_lambda': [1, 10, 100],
}

SEARCH_ITER_N = 10

def sanitize_feature_names(features: List[str]) -> Tuple[dict, dict]:
    """
    Simple sanitization for MLForecast compatibility.
    Handles |, /, and space as _ characters, avoiding double underscores.
    """
    original_to_sanitized = {}
    sanitized_to_original = {}
    
    for feature in features:
        # Replace problematic characters with single underscore
        sanitized = feature.replace('|', '_').replace(' ', '_').replace('/', '_')
        
        # Remove consecutive underscores to avoid __ which MLForecast might interpret specially
        while '__' in sanitized:
            sanitized = sanitized.replace('__', '_')
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        original_to_sanitized[feature] = sanitized
        sanitized_to_original[sanitized] = feature
    
    return original_to_sanitized, sanitized_to_original

def create_mlforecast_model(features: List[str], **xgb_params):
    """
    Create MLForecast model with XGBoost backend and specified lags.
    All features will be treated as lag features.
    """
    xgb_params['tree_method'] = 'hist'
    xgb_params['device'] = 'gpu'
    xgb_params['enable_categorical'] = True
    xgb_model = xgb.XGBRegressor(**xgb_params)
    
    # Sanitize feature names
    original_to_sanitized, sanitized_to_original = sanitize_feature_names(features)
    
    lag_transforms = {
        1: [RollingMean(window_size=1)],  # This creates lag-1 features for all input features
        2: [RollingMean(window_size=1)],  # This creates lag-2 features for all input features
    }
    
    mlf = MLForecast(
        models={'xgb': xgb_model},
        freq='5YE',
        lags=list(range(1, LAGGING + 1)),  # This creates lags for the TARGET (y) only
        lag_transforms=lag_transforms,     # This creates transformed lags for ALL input features
        num_threads=1
    )
    
    # Store mappings for data preparation
    mlf._original_to_sanitized = original_to_sanitized
    mlf._sanitized_to_original = sanitized_to_original
    
    return mlf

def prepare_mlforecast_data(data: pd.DataFrame, features: List[str], targets: List[str], index_columns: List[str]):
    """
    Prepare data for MLForecast format.
    MLForecast expects: unique_id, ds (datetime), y (target)
    """
    original_to_sanitized, _ = sanitize_feature_names(features)
    
    prepared_data = []
    
    for target in targets:
        target_data = data.copy()
        
        target_data['unique_id'] = target_data[index_columns].apply(
            lambda x: '_'.join(x.astype(str)) + f'_{target}', axis=1
        )
        
        target_data['ds'] = pd.to_datetime(target_data['Year'], format='%Y')
        target_data['y'] = target_data[target]
        
        # Rename features to sanitized versions
        for feature in features:
            if feature in target_data.columns:
                sanitized_name = original_to_sanitized[feature]
                target_data = target_data.rename(columns={feature: sanitized_name})
        
        # Keep necessary columns with sanitized names
        sanitized_features = [original_to_sanitized[f] for f in features if f in data.columns]
        keep_cols = ['unique_id', 'ds', 'y'] + sanitized_features
        target_data = target_data[keep_cols].dropna(subset=['y'])
        
        target_data['target'] = target
        
        # Convert object columns to category for XGBoost compatibility
        categorical_columns = ['Model_Family', 'Region', 'target']
        for col in categorical_columns:
            if col in target_data.columns:
                target_data[col] = target_data[col].astype('category').cat.codes
        
        prepared_data.append(target_data)
    
    return pd.concat(prepared_data, ignore_index=True)

def hyperparameter_search_mlforecast(
    train_data: pd.DataFrame, 
    val_data: pd.DataFrame,
    targets: List[str],
    features: List[str],
    run_id: str
) -> Tuple[object, dict, dict]:
    """
    Hyperparameter search for MLForecast with XGBoost.
    """
    from src.trainers.evaluation import evaluate_mlforecast_with_context
    
    search_results = []
    best_score = float('inf')
    best_model = None
    best_params = None
    
    for i, params in enumerate(ParameterSampler(PARAM_DIST, n_iter=SEARCH_ITER_N, random_state=0)):
        logging.info(f"MLForecast Search Iteration {i+1}/{SEARCH_ITER_N} - Params: {params}")
        
        try:
            # Train model on full training data
            mlf = create_mlforecast_model(features, **params)
            
            # Quick check: Look at first unique_id and see if Year varies
            first_unique_id = train_data['unique_id'].iloc[0]
            first_series_data = train_data[train_data['unique_id'] == first_unique_id].head(10)
            logging.info(f"First series ({first_unique_id}) - Year column check:")
            logging.info(f"Years in first series: {first_series_data['Year'].tolist()}")
            logging.info(f"Does Year vary in first series? {first_series_data['Year'].nunique() > 1}")
            
            
            mlf.fit(train_data, static_features=STATIC_FEATURE_COLUMNS)
            
            # Validate using fragment-based evaluation
            _, val_score = evaluate_mlforecast_with_context(
                mlf, val_data, targets, mode="validation", model_type='xgb'
            )
            
            search_results.append({**params, "val_mse": val_score})
            
            if val_score < best_score:
                best_score = val_score
                best_model = mlf
                best_params = params
                
        except Exception as e:
            logging.error(f"Error in iteration {i+1}: {str(e)}", exc_info=True)
            raise e
    
    logging.info(f"Best MLForecast Params: {best_params} with Val MSE: {best_score:.4f}")
    
    # Save best model
    model_path = os.path.join(RESULTS_PATH, run_id, "checkpoints", "best_model.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    best_model.save(model_path)
    logging.info(f"Best model saved to {model_path}")

    return best_model, best_params