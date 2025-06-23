from sklearn.model_selection import ParameterSampler
from sklearn.metrics import mean_squared_error
import os
import logging
import pandas as pd
import numpy as np
import json
from typing import List, Tuple, Dict
import time
from dask.distributed import Client
import dask.dataframe as dd
import gc
import tempfile
from dask import delayed
from contextlib import contextmanager
from xgboost import XGBRegressor
from xgboost.dask import DaskDMatrix, train as dask_train

from configs.config import RESULTS_PATH
from configs.dask_config import CLIENT_CONFIGS
from configs.model.xgb import STAGE_1_PARAMS, STAGE_2_PARAMS, STAGE_3_PARAMS
from src.trainers.evaluation import test_xgb_autoregressively

SEARCH_ITER_N_PER_STAGE = 15  # Iterations per stage
N_FOLDS = 5

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'
os.environ['XGB_CUDA_MAX_MEMORY_PERCENT'] = '80'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


def group_k_fold_split(groups: np.array, n_splits: int, shuffle: bool = True, random_state: int = 42):
    """
    Create k-fold splits ensuring each group appears exactly once in test set across all folds.
    """
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    
    if n_groups < n_splits:
        raise ValueError(f"Number of unique groups ({n_groups}) must be at least equal to n_splits ({n_splits})")
    
    if shuffle:
        rng = np.random.RandomState(random_state)
        unique_groups = rng.permutation(unique_groups)
    group_folds = np.array_split(unique_groups, n_splits)
    
    for fold_idx in range(n_splits):
        test_groups = set(group_folds[fold_idx])
        
        test_indices = np.where(np.isin(groups, list(test_groups)))[0]
        train_indices = np.where(~np.isin(groups, list(test_groups)))[0]
        
        yield train_indices, test_indices


@contextmanager
def cuda_device(device_id: str):
    old_cuda = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    try:
        yield
    finally:
        if old_cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda
        else:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            

@contextmanager
def dask_client_context(**kwargs):
    client = Client(**kwargs)
    try:
        yield client
    finally:
        try:
            client.run(gc.collect)
        except Exception:
            pass
        client.close()


def scatter_to_dask(client, X, y):
    scatter_kwargs = {'hash': False, 'broadcast': True}
    X_future = client.scatter(X, **scatter_kwargs)
    y_future = client.scatter(y, **scatter_kwargs)
    X_dask = dd.from_delayed([delayed(lambda x: x)(X_future)], meta=X)
    y_dask = dd.from_delayed([delayed(lambda x: x)(y_future)], meta=y)
    return X_dask, y_dask


def get_xgb_params(base_params: dict) -> dict:
    return {
        'tree_method': 'hist',
        'device': 'cuda',
        'eval_metric': 'rmse',
        'verbosity': 0,
        'max_bin': 256,
        **base_params
    }


def train_and_evaluate_single_config_cv(
    X, y, X_with_index, train_groups, targets, params, gpu_id, client, n_folds=N_FOLDS
) -> Tuple[Dict, float]:
    """
    Train and evaluate a single parameter configuration using k-fold cross-validation.
    Returns a tuple of (params, negative average RMSE).
    """
    with cuda_device(str(gpu_id)):
        try:
            scores = []
            fold_cache = {}
            for fold, (train_idx, val_idx) in enumerate(group_k_fold_split(train_groups, n_splits=n_folds, shuffle=True, random_state=42)):
                logging.info(f"Fold {fold+1}/{n_folds} for params: {params}")
                
                X_train = X.iloc[train_idx]
                X_val_with_index = X_with_index.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                y_train_df = pd.DataFrame(y_train, columns=targets)
                
                X_train_dask, y_train_dask = scatter_to_dask(client, X_train, y_train_df)
                dtrain = DaskDMatrix(client, X_train_dask, y_train_dask)
                
                xgb_params = get_xgb_params(params)
                num_boost_round = xgb_params.pop('num_boost_round')
                model = dask_train(
                    client,
                    xgb_params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=[(dtrain, 'train')],  # Monitor training only in CV setup
                    early_stopping_rounds=50,
                    verbose_eval=False
                )
                with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
                    temp_model_path = tmp_file.name
                try:
                    model['booster'].save_model(temp_model_path)
                    regular_model = XGBRegressor(**xgb_params)
                    regular_model.load_model(temp_model_path)
                    
                    try:
                        client.run(gc.collect)
                    except Exception:
                        pass
                    
                    predictions = test_xgb_autoregressively(
                        X_val_with_index, 
                        y_val, 
                        model=regular_model, 
                        disable_progress=True, 
                        cache=fold_cache.setdefault(f"fold_{fold}", {})
                    )
                finally:
                    try:
                        os.unlink(temp_model_path)
                    except Exception:
                        pass
                
                rmse = np.sqrt(mean_squared_error(y_val, predictions))
                logging.info(f"Fold {fold+1} RMSE: {rmse:.4f}")
                scores.append(rmse)
                
                try:
                    del dtrain, model, regular_model, predictions
                except Exception:
                    logging.warning("Error cleaning up objects", exc_info=True)
                    
            avg_rmse = np.mean(scores)
            score = -avg_rmse

            return params, score
            
        except Exception as e:
            logging.error(f"Error training config {params}: {str(e)}", exc_info=True)
            raise
        finally:
            try:
                if 'dtrain' in locals():
                    del dtrain
                if 'deval' in locals():
                    del deval
                if 'model' in locals():
                    del model
                if 'regular_model' in locals():
                    del regular_model
                client.run(gc.collect)
                gc.collect()
            except:
                pass


def build_param_dist(stage_params: dict, best_params: dict) -> dict:
    """
    Merge current stage params with best params from previous stages.
    If a param is None in stage_params, use the value from best_params.
    """
    param_dist = {}
    for param, values in stage_params.items():
        if values is None:
            if param in best_params:
                param_dist[param] = [best_params[param]]
            else:
                raise ValueError(f"Parameter '{param}' required from previous stage but not found in best_params.")
        else:
            param_dist[param] = values
    return param_dist


def hyperparameter_search(
    X_train: pd.DataFrame, 
    y_train: np.array, 
    X_train_with_index: pd.DataFrame,
    train_groups: np.array,
    targets: List[str], 
    run_id: str, 
    start_stage: int = 1
) -> Tuple[Dict, Dict]:
    """
    Perform staged hyperparameter search for XGBoost using CV on groups.
    Stages:
      1. Tree structure
      2. Learning rate & trees
      3. Regularization
    """
    os.makedirs(os.path.join(RESULTS_PATH, run_id, "checkpoints"), exist_ok=True)
    checkpoint_dir = os.path.join(RESULTS_PATH, run_id, "checkpoints", "staged_search")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    all_results = {
        'stage_1': [],
        'stage_2': [],
        'stage_3': []
    }
    
    best_params = {}
    overall_best_score = float('-inf')
    overall_best_params = None
    stages = [
        ("Stage 1: Tree Structure", STAGE_1_PARAMS),
        ("Stage 2: Learning Rate & Trees", STAGE_2_PARAMS),
        ("Stage 3: Regularization", STAGE_3_PARAMS)
    ]
    
    # Load checkpoints from completed stages before start_stage
    for stage_num in range(1, start_stage):
        checkpoint_file = os.path.join(checkpoint_dir, f"stage_{stage_num}_best.json")
        if os.path.exists(checkpoint_file):
            logging.info(f"Loading best parameters from stage {stage_num}")
            with open(checkpoint_file, 'r') as f:
                stage_data = json.load(f)
                best_params.update(stage_data['params'])
                if stage_data['score'] > overall_best_score:
                    overall_best_score = stage_data['score']
                    overall_best_params = stage_data['params'].copy()
        else:
            raise FileNotFoundError(f"Cannot start from stage {start_stage} without completing stage {stage_num}")
    
    try:
        # Create one client per stage instead of per iteration
        with dask_client_context(**CLIENT_CONFIGS) as client:
            for stage_idx, (stage_name, stage_params) in enumerate(stages):
                stage_num = stage_idx + 1
                
                if stage_num < start_stage:
                    continue
                
                logging.info(f"{'='*50}")
                logging.info(f"Starting {stage_name}")
                logging.info(f"{'='*50}")
                
                current_param_dist = build_param_dist(stage_params, best_params)
                
                if not current_param_dist:
                    logging.warning(f"No parameters to search in {stage_name}, skipping...")
                    continue
                
                stage_results = []
                stage_best_score = float('-inf')
                stage_best_params = None
                
                param_sampler = ParameterSampler(
                    current_param_dist, 
                    n_iter=SEARCH_ITER_N_PER_STAGE, 
                    random_state=stage_idx
                )
                
                for i, params in enumerate(param_sampler):
                    logging.info(f"{stage_name} - Iteration {i+1}/{SEARCH_ITER_N_PER_STAGE}")
                    
                    gpu_id = i % 5
                    
                    try:
                        params_copy, score = train_and_evaluate_single_config_cv(
                            X_train, y_train, X_train_with_index, train_groups, targets, params, gpu_id, client
                        )
                        
                        result = params_copy.copy()
                        result['mean_test_score'] = score
                        result['stage'] = stage_num
                        stage_results.append(result)
                        
                        if score > stage_best_score:
                            stage_best_score = score
                            stage_best_params = params_copy.copy()
                        
                        if score > overall_best_score:
                            overall_best_score = score
                            overall_best_params = params_copy.copy()
                            
                        logging.info(f"RMSE: {-score:.4f}")
                        
                    except Exception as e:
                        logging.error(f"Error in {stage_name} iteration {i+1}: {str(e)}", exc_info=True)
                        raise
            
                # Update best_params with stage results
                if stage_best_params:
                    for param in current_param_dist.keys():
                        if param in stage_best_params:
                            best_params[param] = stage_best_params[param]
                    
                    checkpoint_file = os.path.join(checkpoint_dir, f"stage_{stage_num}_best.json")
                    with open(checkpoint_file, 'w') as f:
                        json.dump({
                            'stage': stage_num,
                            'params': stage_best_params,
                            'score': stage_best_score,
                            'rmse': -stage_best_score
                        }, f, indent=2)
                
                stage_key = f'stage_{stage_num}'
                all_results[stage_key] = stage_results
                
                logging.info(f"\n{stage_name} Complete")
                logging.info(f"Stage Best RMSE: {-stage_best_score:.4f}")
                logging.info(f"Stage Best Params: {stage_best_params}")
        
        logging.info(f"{'='*50}")
        logging.info("STAGED SEARCH COMPLETE")
        logging.info(f"{'='*50}")
        logging.info(f"Overall Best RMSE: {-overall_best_score:.4f}")
        logging.info(f"Final Best Parameters: {overall_best_params}")
        
        return overall_best_params, all_results
        
    except Exception as e:
        logging.error(f"Error in staged hyperparameter search: {str(e)}", exc_info=True)
        raise


def train_and_save_model(
    X_train: pd.DataFrame,
    y_train: np.array,
    train_groups: np.array,
    targets: List[str],
    best_params: Dict,
    run_id: str
    ) -> None:
    logging.info("Training final model with best parameters...")

    with cuda_device("0"):  # Use GPU 0 for final training
        with dask_client_context(**CLIENT_CONFIGS) as client:
            try:
                train_train_idx, train_eval_idx = next(group_k_fold_split(train_groups, n_splits=5, shuffle=True, random_state=42))
                
                X_train_train = X_train.iloc[train_train_idx]
                X_train_eval = X_train.iloc[train_eval_idx]
                y_train_train = y_train[train_train_idx]
                y_train_eval = y_train[train_eval_idx]
                
                y_train_train_df = pd.DataFrame(y_train_train, columns=targets)
                y_train_eval_df = pd.DataFrame(y_train_eval, columns=targets)
                
                X_train_train_dask, y_train_train_dask = scatter_to_dask(client, X_train_train, y_train_train_df)
                X_train_eval_dask, y_train_eval_dask = scatter_to_dask(client, X_train_eval, y_train_eval_df)
                dtrain = DaskDMatrix(client, X_train_train_dask, y_train_train_dask)
                deval = DaskDMatrix(client, X_train_eval_dask, y_train_eval_dask)

                xgb_params = get_xgb_params(best_params)
                num_boost_round = xgb_params.pop('num_boost_round')
                model = dask_train(
                    client,
                    xgb_params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=[(deval, 'eval')],
                    early_stopping_rounds=15,
                    verbose_eval=25
                )

                model_path = os.path.join(RESULTS_PATH, run_id, "checkpoints", f"final_best.json")
                booster = model['booster']
                booster.save_model(model_path)
                logging.info(f"Model saved to {model_path}")

            except Exception as e:
                logging.error(f"Error during final model training: {str(e)}", exc_info=True)
                raise

from sklearn.model_selection import ParameterSampler
from sklearn.metrics import mean_squared_error
import os
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
from typing import List, Tuple, Dict
import time
from dask.distributed import Client
import dask.dataframe as dd
import gc
import tempfile
from dask import delayed
<<<<<<< HEAD
from xgboost.dask import DaskDMatrix, train as dask_train, predict as dask_predict
from configs.config import RESULTS_PATH
from configs.dask_config import *

from src.utils.utils import create_dask_client

# Stage 1: Tree Structure
STAGE_1_PARAMS = {
    'max_depth': [5, 9, 13, 17],
    'min_child_weight': [10, 12, 15, 20],
    'gamma': [0],  # Keep gamma at 0 initially
    'eta': [0.1],  # Fixed learning rate
    'num_boost_round': [1000],  # Fixed with early stopping
    'subsample': [1.0],  # Fixed initially
    'colsample_bytree': [1.0],  # Fixed initially
    'reg_alpha': [0],  # Fixed initially
    'reg_lambda': [1],  # Fixed initially
}

# Stage 2: Stochastic Parameters
STAGE_2_PARAMS = {
    'max_depth': None,  # Will be set from stage 1 best
    'min_child_weight': None,  # Will be set from stage 1 best
    'gamma': [0],  # Keep at 0
    'eta': [0.1],  # Fixed learning rate
    'num_boost_round': [1000],  # Fixed with early stopping
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0],  # Fixed initially
    'reg_lambda': [1],  # Fixed initially
}
=======
from contextlib import contextmanager
from xgboost import XGBRegressor
from xgboost.dask import DaskDMatrix, train as dask_train
>>>>>>> fix: changed val set evaluation from teacher forcing to free-running

from configs.config import RESULTS_PATH
from configs.dask_config import *
from configs.model.xgb import *
from src.trainers.evaluation import test_xgb_autoregressively

SEARCH_ITER_N_PER_STAGE = 15  # Iterations per stage
N_FOLDS = 5


def group_k_fold_split(groups: np.array, n_splits: int, shuffle: bool = True, random_state: int = 42):
    """
    Create k-fold splits ensuring each group appears exactly once in test set across all folds.
    
    Parameters:
    -----------
    groups : np.array
        Array of group labels for each sample
    n_splits : int, default=5
        Number of folds
    shuffle : bool, default=True
        Whether to shuffle groups before splitting
    random_state : int, default=None
        Random state for reproducibility
        
    Returns:
    --------
    Generator yielding (train_indices, test_indices) tuples
    
    Raises:
    -------
    ValueError: If number of unique groups is less than n_splits
    """
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    
    if n_groups < n_splits:
        raise ValueError(f"Number of unique groups ({n_groups}) must be at least equal to n_splits ({n_splits})")
    
    if shuffle:
        rng = np.random.RandomState(random_state)
        unique_groups = rng.permutation(unique_groups)
    group_folds = np.array_split(unique_groups, n_splits)
    
    for fold_idx in range(n_splits):
        test_groups = set(group_folds[fold_idx])
        
        test_indices = np.where(np.isin(groups, list(test_groups)))[0]
        train_indices = np.where(~np.isin(groups, list(test_groups)))[0]
        
        yield train_indices, test_indices


@contextmanager
def cuda_device(device_id: str):
    old_cuda = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    try:
        yield
    finally:
        if old_cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda
        else:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            

@contextmanager
def dask_client_context(**kwargs):
    client = Client(**kwargs)
    try:
        yield client
    finally:
        client.run(gc.collect)
        client.close()


def scatter_to_dask(client, X, y):
    scatter_kwargs = {'hash': False, 'broadcast': True}
    X_future = client.scatter(X, **scatter_kwargs)
    y_future = client.scatter(y, **scatter_kwargs)
    X_dask = dd.from_delayed([delayed(lambda x: x)(X_future)], meta=X)
    y_dask = dd.from_delayed([delayed(lambda x: x)(y_future)], meta=y)
    return X_dask, y_dask


def get_xgb_params(base_params: dict) -> dict:
    return {
        'tree_method': 'hist',
        'device': 'cuda',
        'eval_metric': 'rmse',
        'verbosity': 0,
        'max_bin': 256,
        **base_params
    }


def train_and_evaluate_single_config_cv(
    X, y, X_with_index, train_groups, targets, params, gpu_id, client, n_folds=N_FOLDS
) -> Tuple[Dict, float]:
    """
    Train and evaluate a single parameter configuration using k-fold cross-validation.
    --------
    Returns:
        Tuple[Dict, float]: A tuple containing the parameters and the average negative RMSE score across folds.
    """
    with cuda_device(str(gpu_id % 5)):
        try:
            scores = []
            fold_cache = {}
            for fold, (train_idx, val_idx) in enumerate(group_k_fold_split(train_groups, n_splits=n_folds, shuffle=True, random_state=42)):
                logging.info(f"Fold {fold+1}/{n_folds} for params: {params}")
                
                X_train = X.iloc[train_idx]
                X_val_with_index = X_with_index.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Split training data further for evaluation monitoring
                train_groups_fold = train_groups[train_idx]
                train_train_idx, train_eval_idx = next(group_k_fold_split(train_groups_fold, n_splits=5, shuffle=True, random_state=42))
                
                X_train_train = X_train.iloc[train_train_idx]
                X_train_eval = X_train.iloc[train_eval_idx]
                y_train_train = y_train[train_train_idx]
                y_train_eval = y_train[train_eval_idx]
                
                y_train_train_df = pd.DataFrame(y_train_train, columns=targets)
                y_train_eval_df = pd.DataFrame(y_train_eval, columns=targets)
                
                cache_key = f"fold_{fold}"
                if cache_key not in fold_cache:
                    fold_cache[cache_key] = {}
                
                X_train_train_dask, y_train_train_dask = scatter_to_dask(client, X_train_train, y_train_train_df)
                X_train_eval_dask, y_train_eval_dask = scatter_to_dask(client, X_train_eval, y_train_eval_df)
                dtrain = DaskDMatrix(client, X_train_train_dask, y_train_train_dask)
                deval = DaskDMatrix(client, X_train_eval_dask, y_train_eval_dask)
                
                xgb_params = get_xgb_params(params)
                num_boost_round = xgb_params.pop('num_boost_round')
                model = dask_train(
                    client,
                    xgb_params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=[(deval, 'eval')],
                    early_stopping_rounds=15,
                    verbose_eval=25
                )
                with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
                    temp_model_path = tmp_file.name
                try:
                    model['booster'].save_model(temp_model_path)
                    regular_model = XGBRegressor(**xgb_params)
                    regular_model.load_model(temp_model_path)
                    
                    client.run(gc.collect)
                    
                    predictions = test_xgb_autoregressively(
                        X_val_with_index, 
                        y_val, 
                        model=regular_model, 
                        disable_progress=True, 
                        cache=fold_cache[cache_key]
                    )
                finally:
                    try:
                        os.unlink(temp_model_path)
                    except:
                        pass
                
                # Calculate negative RMSE (higher is better)
                rmse = np.sqrt(mean_squared_error(y_val, predictions))
                logging.info(f"Fold {fold+1} RMSE: {rmse:.4f}")
                scores.append(rmse)
                
                try:
                    del dtrain, deval, model, regular_model, predictions
                    del X_train_train_dask, y_train_train_dask, X_train_eval_dask, y_train_eval_dask
                except:
                    logging.warning("Error cleaning up objects", exc_info=True)
                    
            avg_rmse = np.mean(scores)
            score = -avg_rmse

            return params, score
            
        except Exception as e:
            logging.error(f"Error training config {params}: {str(e)}", exc_info=True)
            raise


def build_param_dist(stage_params: dict, best_params: dict) -> dict:
    """
    Merge current stage params with best params from previous stages.
    If a param is None in stage_params, use the value from best_params.
    """
    param_dist = {}
    for param, values in stage_params.items():
        if values is None:
            if param in best_params:
                param_dist[param] = [best_params[param]]
            else:
                raise ValueError(f"Parameter '{param}' required from previous stage but not found in best_params.")
        else:
            param_dist[param] = values
    return param_dist


def hyperparameter_search(
    X_train: pd.DataFrame, 
    y_train: np.array, 
    X_train_with_index: pd.DataFrame,
    train_groups: np.array,
    targets: List[str], 
    run_id: str, 
    start_stage: int = 1
) -> Tuple[Dict, Dict]:
    """
    Perform staged hyperparameter search for XGBoost.
    
    This function performs hyperparameter search in 3 stages:
    1. Tree structure (max_depth, min_child_weight)
    2. Learning rate and number of trees (eta, num_boost_round)
    3. Regularization (gamma, reg_alpha, reg_lambda)
    
    Returns:
    --------
    Tuple[Dict, Dict]
        A tuple containing:
        - best_params (Dict): The final best hyperparameter combination
        - all_results (Dict): Results from all stages for visualization
    """
    os.makedirs(os.path.join(RESULTS_PATH, run_id, "checkpoints"), exist_ok=True)
    checkpoint_dir = os.path.join(RESULTS_PATH, run_id, "checkpoints", "staged_search")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    all_results = {
        'stage_1': [],
        'stage_2': [],
        'stage_3': []
    }
    
    best_params = {}
    overall_best_score = float('-inf')
    overall_best_params = None
    stages = [
        ("Stage 1: Tree Structure", STAGE_1_PARAMS),
        ("Stage 2: Learning Rate & Trees", STAGE_2_PARAMS),
        ("Stage 3: Regularization", STAGE_3_PARAMS)
    ]
    
    # Load checkpoints from completed stages before start_stage
    for stage_num in range(1, start_stage):
        checkpoint_file = os.path.join(checkpoint_dir, f"stage_{stage_num}_best.json")
        if os.path.exists(checkpoint_file):
            logging.info(f"Loading best parameters from stage {stage_num}")
            with open(checkpoint_file, 'r') as f:
                stage_data = json.load(f)
                best_params.update(stage_data['params'])
                if stage_data['score'] > overall_best_score:
                    overall_best_score = stage_data['score']
                    overall_best_params = stage_data['params'].copy()
        else:
            logging.error(f"Required checkpoint for Stage {stage_num} not found at {checkpoint_file}", exc_info=True)
            raise FileNotFoundError(f"Cannot start from stage {start_stage} without completing stage {stage_num}")
    
    try:
<<<<<<< HEAD
        for stage_idx, (stage_name, stage_params) in enumerate(stages):
            if stage_idx + 1 < start_stage:
                continue
            
            logging.info(f"{'='*50}")
            logging.info(f"Starting {stage_name}")
            logging.info(f"{'='*50}")
            
            # Prepare parameter distribution for current stage
            current_param_dist = {}
            for param, values in stage_params.items():
                if values is None:
                    # Use best value from previous stage
                    if param in best_params:
                        current_param_dist[param] = [best_params[param]]
                    else:
                        logging.warning(f"Parameter {param} not found in previous stages, skipping...")
                else:
                    current_param_dist[param] = values
            
            # Skip if no parameters to search
            if not current_param_dist:
                logging.warning(f"No parameters to search in {stage_name}, skipping...")
                continue
            
            # Initialize tracking for this stage
            stage_results = []
            stage_best_score = float('-inf')
            stage_best_params = None
            
            # Sample parameters for this stage
            param_sampler = ParameterSampler(
                current_param_dist, 
                n_iter=SEARCH_ITER_N_PER_STAGE, 
                random_state=stage_idx
            )
            
            for i, params in enumerate(param_sampler):
                logging.info(f"{stage_name} - Iteration {i+1}/{SEARCH_ITER_N_PER_STAGE}")
>>>>>>> add: 4-stage parameter search
                client = None
                try:
                    # Create client with optimized memory settings
                    client = Client(
                        n_workers=1,
                        threads_per_worker=2,
                        memory_limit='4GB',
                        silence_logs=logging.WARNING,
                        dashboard_address=None,
                        local_directory='/tmp/dask-worker-space'
                    )
                    
                    # Train and evaluate
                    params_copy, score = train_and_evaluate_single_config(
                        X_train, y_train, X_val, y_val, targets, params, gpu_id, client
                    )
                    
                    # Store results
                    result = params_copy.copy()
                    result['mean_test_score'] = score
                    result['stage'] = stage_idx + 1
                    stage_results.append(result)
=======
        # Create one client per stage instead of per iteration
        with dask_client_context(**CLIENT_CONFIGS) as client:
            for stage_idx, (stage_name, stage_params) in enumerate(stages):
                stage_num = stage_idx + 1
                
                if stage_num < start_stage:
                    continue
                
                logging.info(f"{'='*50}")
                logging.info(f"Starting {stage_name}")
                logging.info(f"{'='*50}")
                
                current_param_dist = build_param_dist(stage_params, best_params)
                
                if not current_param_dist:
                    logging.warning(f"No parameters to search in {stage_name}, skipping...")
                    continue
                
                stage_results = []
                stage_best_score = float('-inf')
                stage_best_params = None
                
                param_sampler = ParameterSampler(
                    current_param_dist, 
                    n_iter=SEARCH_ITER_N_PER_STAGE, 
                    random_state=stage_idx
                )
                
                for i, params in enumerate(param_sampler):
                    logging.info(f"{stage_name} - Iteration {i+1}/{SEARCH_ITER_N_PER_STAGE}")
>>>>>>> fix: changed val set evaluation from teacher forcing to free-running
                    
                    gpu_id = i % 8
                    
                    try:
                        params_copy, score = train_and_evaluate_single_config_cv(
                            X_train, y_train, X_train_with_index, train_groups, targets, params, gpu_id, client
                        )
                        
                        result = params_copy.copy()
                        result['mean_test_score'] = score
                        result['stage'] = stage_num
                        stage_results.append(result)
                        
                        if score > stage_best_score:
                            stage_best_score = score
                            stage_best_params = params_copy.copy()
                        
                        if score > overall_best_score:
                            overall_best_score = score
                            overall_best_params = params_copy.copy()
                            
                        logging.info(f"RMSE: {-score:.4f}")
                        
                    except Exception as e:
                        logging.error(f"Error in {stage_name} iteration {i+1}: {str(e)}", exc_info=True)
                        raise
            
                # Update best_params with stage results
                if stage_best_params:
                    for param in current_param_dist.keys():
                        if param in stage_best_params:
                            best_params[param] = stage_best_params[param]
                    
                    checkpoint_file = os.path.join(checkpoint_dir, f"stage_{stage_num}_best.json")
                    with open(checkpoint_file, 'w') as f:
                        json.dump({
                            'stage': stage_num,
                            'params': stage_best_params,
                            'score': stage_best_score,
                            'rmse': -stage_best_score
                        }, f, indent=2)
                
                stage_key = f'stage_{stage_num}'
                all_results[stage_key] = stage_results
                
                logging.info(f"\n{stage_name} Complete")
                logging.info(f"Stage Best RMSE: {-stage_best_score:.4f}")
                logging.info(f"Stage Best Params: {stage_best_params}")
        
        logging.info(f"{'='*50}")
        logging.info("STAGED SEARCH COMPLETE")
        logging.info(f"{'='*50}")
        logging.info(f"Overall Best RMSE: {-overall_best_score:.4f}")
        logging.info(f"Final Best Parameters: {overall_best_params}")
        
        return overall_best_params, all_results
        
    except Exception as e:
        logging.error(f"Error in staged hyperparameter search: {str(e)}", exc_info=True)
        raise


<<<<<<< HEAD
def visualize_staged_search_results(all_results: Dict, run_id: str):
    """
    Visualize the results from staged hyperparameter search.
    
    Creates visualization for each stage showing parameter exploration and performance.
    """
    if not all_results:
        logging.error("No staged search results provided.")
        return
    
    vis_dir = os.path.join(RESULTS_PATH, run_id, "config", "staged_search")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create overall summary plot
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Performance across stages
    plt.subplot(2, 2, 1)
    stage_best_scores = []
    stage_labels = []
>>>>>>> add: 4-stage parameter search
    
    for stage_num in range(1, 5):
        stage_key = f'stage_{stage_num}'
        if stage_key in all_results and all_results[stage_key]:
            scores = [r['mean_test_score'] for r in all_results[stage_key]]
            if scores:
                stage_best_scores.append(max(scores))
                stage_labels.append(f'Stage {stage_num}')
    if stage_best_scores:
        plt.bar(stage_labels, [-s for s in stage_best_scores])  # Convert back to positive RMSE
        plt.ylabel('Best RMSE')
        plt.title('Best Performance by Stage')
        plt.xticks(rotation=45)
    
    # Plot 2: Parameter evolution
    plt.subplot(2, 2, 2)
    param_evolution = {}
    
    for stage_num in range(1, 5):
        stage_key = f'stage_{stage_num}'
        if stage_key in all_results and all_results[stage_key]:
            for result in all_results[stage_key]:
                if result['mean_test_score'] == max(r['mean_test_score'] for r in all_results[stage_key]):
                    for param, value in result.items():
                        if param not in ['mean_test_score', 'stage']:
                            if param not in param_evolution:
                                param_evolution[param] = []
                            param_evolution[param].append((stage_num, value))
    
    for param, values in param_evolution.items():
        stages, vals = zip(*values)
        plt.plot(stages, vals, marker='o', label=param)
    
    plt.xlabel('Stage')
    plt.ylabel('Parameter Value')
    plt.title('Parameter Evolution Across Stages')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 3: Stage progression
    plt.subplot(2, 2, 3)
    for stage_num in range(1, 5):
        stage_key = f'stage_{stage_num}'
        if stage_key in all_results and all_results[stage_key]:
            scores = [-r['mean_test_score'] for r in all_results[stage_key]]  # Convert to positive RMSE
            plt.plot(scores, alpha=0.7, label=f'Stage {stage_num}')
    
    plt.xlabel('Iteration within Stage')
    plt.ylabel('RMSE')
    plt.title('Search Progress by Stage')
    plt.legend()
    
    # Plot 4: Final parameters heatmap (if possible)
    plt.subplot(2, 2, 4)
    final_stage_results = all_results.get('stage_4', [])
    if final_stage_results and len(final_stage_results) > 1:
        # Create a simple heatmap of final stage parameters
        df = pd.DataFrame(final_stage_results)
        numeric_cols = df.select_dtypes(include=[np.number]).drop('stage', axis=1, errors='ignore')
        
        if not numeric_cols.empty:
            corr_matrix = numeric_cols.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Final Stage Parameter Correlations')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'staged_search_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual stage visualizations
    for stage_num in range(1, 5):
        stage_key = f'stage_{stage_num}'
        if stage_key in all_results and all_results[stage_key]:
            visualize_single_stage(all_results[stage_key], stage_num, vis_dir)
=======
def train_and_save_model(
    X_train: pd.DataFrame,
    y_train: np.array,
    targets: List[str],
    best_params: Dict,
    run_id: str
    ) -> None:
    logging.info("Training final model with best parameters...")
>>>>>>> fix: changed val set evaluation from teacher forcing to free-running

    with cuda_device("0"):  # Use GPU 0 for final training
        with dask_client_context(**CLIENT_CONFIGS) as client:
            try:
                y_train_df = pd.DataFrame(y_train, columns=targets)
                
                X_train_dask, y_train_dask = scatter_to_dask(client, X_train, y_train_df)
                dtrain = DaskDMatrix(client, X_train_dask, y_train_dask)

<<<<<<< HEAD
def visualize_single_stage(stage_results: List[Dict], stage_num: int, vis_dir: str):
    """
    Create visualization for a single stage of hyperparameter search.
    """
    df = pd.DataFrame(stage_results)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Stage {stage_num} Hyperparameter Search Results')
    
    # Plot 1: Parameter distribution
    ax1 = axes[0, 0]
    param_cols = [col for col in df.columns if col.startswith('param_') or (col not in ['mean_test_score', 'stage'])]
    if param_cols:
        df[param_cols].boxplot(ax=ax1)
        ax1.set_title('Parameter Value Distributions')
        ax1.tick_params(axis='x', rotation=90)
    
    # Plot 2: Score distribution
    ax2 = axes[0, 1]
    ax2.hist(-df['mean_test_score'], bins=20, edgecolor='black')  # Convert to positive RMSE
    ax2.set_xlabel('RMSE')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Score Distribution')
    
    # Plot 3: Top 5 configurations
    ax3 = axes[1, 0]
    top_5 = df.nlargest(5, 'mean_test_score')
    top_5_data = []
    
    for _, row in top_5.iterrows():
        config_str = ', '.join([f"{k}: {v}" for k, v in row.items() if k not in ['mean_test_score', 'stage']])
        top_5_data.append((config_str, -row['mean_test_score']))  # Convert to positive RMSE
    
    if top_5_data:
        configs, scores = zip(*top_5_data)
        y_pos = np.arange(len(configs))
        ax3.barh(y_pos, scores)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([c[:50] + '...' if len(c) > 50 else c for c in configs])
        ax3.set_xlabel('RMSE')
        ax3.set_title('Top 5 Configurations')
    
    # Plot 4: Parameter impact (if enough data)
    ax4 = axes[1, 1]
    if len(df) > 10:
        # Calculate parameter importance
        param_importance = {}
        for col in param_cols:
            if df[col].dtype in [np.float64, np.int64] and df[col].nunique() > 1:
                corr = df[[col, 'mean_test_score']].corr().iloc[0, 1]
                param_importance[col] = abs(corr)
        
        if param_importance:
            params, importance = zip(*sorted(param_importance.items(), key=lambda x: x[1], reverse=True))
            ax4.bar(params, importance)
            ax4.set_ylabel('Absolute Correlation with Score')
            ax4.set_title('Parameter Importance')
            ax4.tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'stage_{stage_num}_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
=======
                xgb_params = get_xgb_params(best_params)
                num_boost_round = xgb_params.pop('num_boost_round')
                model = dask_train(
                    client,
                    xgb_params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=[(dtrain, 'train')],
                    early_stopping_rounds=50,
                    verbose_eval=True
                )

                model_path = os.path.join(RESULTS_PATH, run_id, "checkpoints", f"final_best.json")
                booster = model['booster']
                booster.save_model(model_path)
                logging.info(f"Model saved to {model_path}")

            except Exception as e:
                logging.error(f"Error during final model training: {str(e)}", exc_info=True)
                raise
>>>>>>> fix: changed val set evaluation from teacher forcing to free-running
