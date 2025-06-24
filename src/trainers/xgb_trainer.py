from sklearn.model_selection import ParameterSampler
from sklearn.metrics import root_mean_squared_error
import os
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple
import gc
import time
import dask.dataframe as dd
from xgboost.dask import DaskDMatrix, train as dask_train, predict as dask_predict
from configs.config import RESULTS_PATH
from configs.dask_config import *

from src.utils.utils import create_dask_client

PARAM_DIST = {
    'max_depth': [8, 10, 12],
    'learning_rate': [0.01, 0.1, 0.2],
    'num_boost_round': [300, 500, 700],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5],
    'reg_alpha': [0.1, 1, 10],
    'reg_lambda': [1, 10, 100],
}

PARAM_PAIRS = [
    ('max_depth', 'learning_rate'),
    ('num_boost_round', 'subsample'),
    ('colsample_bytree', 'gamma'),
    ('reg_alpha', 'reg_lambda')
]

SEARCH_ITER_N = 50
N_FOLDS = 3


def hyperparameter_search(run_id: str, X_train: pd.DataFrame, y_train: np.array, X_val: pd.DataFrame, y_val: np.array, targets: List[str]) -> Tuple[dict, float, dict]:
    """
    Alternative approach using DaskDMatrix since using DaskXGBRegressor has serialization issues.
    Use one GPU at a time, cycling through all available GPUs.
    """
    search_results = []
    best_params = None
    best_score = float('-inf')  # Initialize for neg RMSE (higher is better)
    
    # Store original CUDA_VISIBLE_DEVICES
    original_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    
    for i, params in enumerate(ParameterSampler(PARAM_DIST, n_iter=SEARCH_ITER_N, random_state=0)):
        logging.info(f"Iteration {i+1}/{SEARCH_ITER_N}")
        
        # Select GPU for this iteration
        gpu_id = i % 8
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        try:
            # Create simple client with minimal configuration
            with create_dask_client() as client:
                logging.info(f"Dask Dashboard URL: {client.dashboard_link}")
                logging.info(f"Using GPU {gpu_id} for iteration {i+1}")
                
                # Create Dask DataFrames with smaller partitions
                y_train_df = pd.DataFrame(y_train, columns=targets)
                y_val_df = pd.DataFrame(y_val, columns=targets)
                
                # Use scatter to distribute data to workers
                X_train_future = client.scatter(X_train, broadcast=True)
                y_train_future = client.scatter(y_train_df, broadcast=True)
                X_val_future = client.scatter(X_val, broadcast=True)
                y_val_future = client.scatter(y_val_df, broadcast=True)
                
                # Create DataFrames from futures
                X_train_dask = dd.from_delayed([client.submit(lambda x: x, X_train_future)])
                y_train_dask = dd.from_delayed([client.submit(lambda x: x, y_train_future)])
                X_val_dask = dd.from_delayed([client.submit(lambda x: x, X_val_future)])
                y_val_dask = dd.from_delayed([client.submit(lambda x: x, y_val_future)])
                
                # Create DaskDMatrix
                dtrain = DaskDMatrix(client, X_train_dask, y_train_dask)
                dval = DaskDMatrix(client, X_val_dask, y_val_dask)
                
                # Set XGBoost parameters
                xgb_params = {
                    'tree_method': 'hist',
                    'device': 'cuda',  # Let it use the CUDA_VISIBLE_DEVICES
                    'eval_metric': 'rmse',
                    'verbosity': 1,
                    **params
                }
                num_boost_round = xgb_params.pop('num_boost_round')
                
                # Train model
                model = dask_train(
                    client,
                    xgb_params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=[(dval, 'validation')],
                    early_stopping_rounds=10,
                    verbose_eval=False
                )
                
                try:
                    client.run(gc.collect)  # Force garbage collection
                    predictions = dask_predict(client, model, dval)
                    score = root_mean_squared_error(y_val, predictions)
                    
                except Exception as pred_error:
                    logging.error(f"Prediction error: {str(pred_error)}")
                    
                    # Force garbage collection
                    if 'dval' in locals():
                        del dval
                        dval = DaskDMatrix(client, X_val_dask, y_val_dask)
                        
                    client.run(gc.collect)
                    time.sleep(2)
                
                # Store results
                result = params.copy()
                result['mean_test_score'] = score
                search_results.append(result)
                
                if score > best_score:  # Higher neg RMSE is better
                    best_score = score
                    best_params = params
                    
                    model = dask_train(
                        client,
                        xgb_params,
                        dtrain,
                        num_boost_round=num_boost_round,
                        evals=[(dval, 'validation')],
                        early_stopping_rounds=10,
                        verbose_eval=False
                    )
                    model_path = os.path.join(RESULTS_PATH, run_id, "checkpoints", f"best_model.json")
                    model['booster'].save_model(model_path)
                    logging.info(f"Model saved to {model_path}")
                    
                logging.info(f"Parameters: {params}")
                logging.info(f"Negative RMSE: {score:.4f}")
            
            try:
                del dtrain, dval, model, predictions
                del X_train_future, y_train_future, X_val_future, y_val_future
                client.cancel([X_train_future, y_train_future, X_val_future, y_val_future])
            except:
                pass
                
        except Exception as e:
            logging.error(f"Error in iteration {i+1}: {str(e)}")
            logging.error(f"Full error details:", exc_info=True)
            continue
    
        finally:
            if client:
                try:
                    client.run(gc.collect)
                    client.close()
                except:
                    pass
            gc.collect()
            time.sleep(0.5)
    
    # Restore original CUDA_VISIBLE_DEVICES
    if original_cuda_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_devices
    else:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    
    # Prepare results for visualization
    cv_results_dict = {
        'mean_test_score': [r['mean_test_score'] for r in search_results]
    }
    
    for param in PARAM_DIST.keys():
        cv_results_dict[f'param_{param}'] = [r.get(param) for r in search_results]
    
    logging.info(f"Best parameters: {best_params}")
    logging.info(f"Best Negative RMSE: {best_score:.4f}")
    
    return best_params, cv_results_dict


def visualize_multiple_hyperparam_searches(cv_results_dict, run_id):
    """
    Visualizes the hyperparameter search results using multiple heatmaps for different parameter pairs.
    
    Parameters:
    - cv_results_dict: Dictionary containing the results of the hyperparameter search.
    - run_id: Unique identifier for the current run, used for saving results.
    """
    if cv_results_dict is None:
        logging.error("No random search results provided.")
        return

    results_df = pd.DataFrame(cv_results_dict)
    
    param_search_dir = os.path.join(RESULTS_PATH, run_id, "config")
    os.makedirs(param_search_dir, exist_ok=True)

    for param1, param2 in PARAM_PAIRS:
        # Create pivot table for heatmap
        heatmap_data = results_df.pivot_table(
            index=f'param_{param1}',
            columns=f'param_{param2}',
            values='mean_test_score',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".4f",
            cmap="coolwarm",
            cbar_kws={'label': 'Mean Test Score'},
            linewidths=.5
        )
        plt.title(f"Hyperparameter Search: {param1} vs {param2}")
        plt.xlabel(param2)
        plt.ylabel(param1)
        plt.tight_layout()

        filename = f"hyperparam_search_{param1}_vs_{param2}.png"
        plt.savefig(os.path.join(param_search_dir, filename))
        plt.close()