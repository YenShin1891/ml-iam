import logging
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from mlforecast import MLForecast
from mlforecast.utils import PredictionIntervals
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

def create_mlforecast_model(**xgb_params):
    """
    Create MLForecast model with XGBoost backend and specified lags.
    """
    xgb_params['tree_method'] = 'hist'
    xgb_params['device'] = 'gpu'
    xgb_params['enable_categorical'] = True
    xgb_model = xgb.XGBRegressor(**xgb_params)
    
    mlf = MLForecast(
        models={'xgb': xgb_model},
        freq='5Y',
        lags=list(range(1, LAGGING + 1)),  # Use lags instead of manually adding prev outputs
        date_features=['year'],
        num_threads=1
    )
    
    return mlf

def prepare_mlforecast_data(data: pd.DataFrame, features: List[str] ,targets: List[str], index_columns: List[str]):
    """
    Prepare data for MLForecast format.
    MLForecast expects: unique_id, ds (datetime), y (target)
    """
    prepared_data = []
    
    for target in targets:
        # Create a copy for each target
        target_data = data.copy()
        
        # Create unique_id by combining index columns
        target_data['unique_id'] = target_data[index_columns].apply(
            lambda x: '_'.join(x.astype(str)) + f'_{target}', axis=1
        )
        
        # Rename Year to ds and target to y
        target_data['ds'] = pd.to_datetime(target_data['Year'], format='%Y')
        target_data['y'] = target_data[target]
        
        # Keep only necessary columns
        keep_cols = ['unique_id', 'ds', 'y'] + [col for col in features]
        target_data = target_data[keep_cols].dropna(subset=['y'])
        
        # Add target identifier
        target_data['target'] = target
        
        # Convert object columns to category for XGBoost compatibility
        categorical_columns = ['Model_Family', 'Region', 'target']
        for col in categorical_columns:
            if col in target_data.columns:
                target_data[col] = target_data[col].astype('category')
        
        prepared_data.append(target_data)
    
    return pd.concat(prepared_data, ignore_index=True)

def hyperparameter_search_mlforecast(
    train_data: pd.DataFrame, 
    val_data: pd.DataFrame,
    targets: List[str],
    run_id: str
) -> Tuple[object, dict, dict]:
    """
    Hyperparameter search for MLForecast with XGBoost.
    """
    from src.trainers.evaluation import evaluate_mlforecast_fragment_based
    
    search_results = []
    best_score = float('inf')
    best_model = None
    best_params = None
    
    for i, params in enumerate(ParameterSampler(PARAM_DIST, n_iter=SEARCH_ITER_N, random_state=0)):
        logging.info(f"MLForecast Search Iteration {i+1}/{SEARCH_ITER_N} - Params: {params}")
        
        try:
            # Train model on full training data
            mlf = create_mlforecast_model(**params)
            mlf.fit(train_data, static_features=STATIC_FEATURE_COLUMNS)
            
            # Validate using fragment-based evaluation
            _, val_score = evaluate_mlforecast_fragment_based(
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