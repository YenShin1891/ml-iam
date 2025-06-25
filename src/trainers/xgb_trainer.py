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
from contextlib import contextmanager
from xgboost import XGBRegressor
from xgboost.dask import DaskDMatrix, train as dask_train

from configs.config import RESULTS_PATH
from configs.dask_config import *
from configs.model.xgb import *
from src.trainers.evaluation import test_xgb_autoregressively

SEARCH_ITER_N_PER_STAGE = 15  # Iterations per stage
N_FOLDS = 5

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'
os.environ['XGB_CUDA_MAX_MEMORY_PERCENT'] = '80'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


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


def train_and_evaluate_single_config(
    X, y, X_with_index, train_groups, targets, params, gpu_id, client, 
    use_cv=True, X_val=None, y_val=None, X_val_with_index=None, n_folds=N_FOLDS,
    use_dask=True
) -> Tuple[Dict, float]:
    """
    Train and evaluate a single parameter configuration using either k-fold CV or single validation set.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Training features (or all features if use_cv=False)
    y : np.array
        Training targets (or all targets if use_cv=False)
    X_with_index : pd.DataFrame
        Features with index for autoregressive testing
    train_groups : np.array
        Group labels for k-fold splitting (ignored if use_cv=False)
    targets : List[str]
        Target column names
    params : dict
        XGBoost parameters
    gpu_id : int
        GPU device ID
    client : dask.distributed.Client
        Dask client for distributed computing (ignored if use_dask=False)
    use_cv : bool, default=True
        If True, use k-fold cross-validation. If False, use single validation set.
    X_val : pd.DataFrame, optional
        Validation features (required if use_cv=False)
    y_val : np.array, optional
        Validation targets (required if use_cv=False)
    X_val_with_index : pd.DataFrame, optional
        Validation features with index (required if use_cv=False)
    n_folds : int, default=N_FOLDS
        Number of folds for cross-validation (ignored if use_cv=False)
    use_dask : bool, default=True
        If True, use Dask for distributed training. If False, use regular XGBoost.
        
    Returns:
    --------
    Tuple[Dict, float]: A tuple containing the parameters and the negative RMSE score.
    """
    with cuda_device(str(gpu_id % 5)):
        try:
            if use_cv:
                # Cross-validation mode
                scores = []
                fold_cache = {}
                for fold, (train_idx, val_idx) in enumerate(group_k_fold_split(train_groups, n_splits=n_folds, shuffle=True, random_state=42)):
                    logging.info(f"Fold {fold+1}/{n_folds} for params: {params}")
                    
                    X_train, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                    X_val_with_index_fold = X_with_index.iloc[val_idx]
                    y_train, y_val_fold = y[train_idx], y[val_idx]
                    
                    cache_key = f"fold_{fold}"
                    if cache_key not in fold_cache:
                        fold_cache[cache_key] = {}
                    
                    fold_score = _train_single_fold(
                        X_train, y_train, X_val_fold, y_val_fold, X_val_with_index_fold,
                        targets, params, client, fold_cache[cache_key], fold+1, use_dask
                    )
                    scores.append(fold_score)
                    
                avg_rmse = np.mean(scores)
                score = -avg_rmse
                
            else:
                # Single validation set mode
                if X_val is None or y_val is None or X_val_with_index is None:
                    raise ValueError("X_val, y_val, and X_val_with_index must be provided when use_cv=False")
                
                logging.info(f"Training with params: {params}")
                fold_score = _train_single_fold(
                    X, y, X_val, y_val, X_val_with_index,
                    targets, params, client, {}, 1, use_dask
                )
                score = -fold_score

            return params, score
            
        except Exception as e:
            logging.error(f"Error training config {params}: {str(e)}", exc_info=True)
            raise
        finally:
            try:
                if use_dask and client:
                    client.run(gc.collect)
                gc.collect()
            except:
                pass


def _train_single_fold(X_train, y_train, X_val, y_val, X_val_with_index, targets, params, client, cache, fold_num, use_dask=True):
    """
    Helper function to train and evaluate a single fold/validation set.
    
    Parameters:
    -----------
    use_dask : bool, default=True
        If True, use Dask for distributed training. If False, use regular XGBoost.
    
    Returns:
    --------
    float: RMSE score for this fold
    """
    y_train_df = pd.DataFrame(y_train, columns=targets)
    y_val_df = pd.DataFrame(y_val, columns=targets)
    
    xgb_params = get_xgb_params(params)
    num_boost_round = xgb_params.pop('num_boost_round')
    
    if use_dask:
        # Dask training
        X_train_dask, y_train_dask = scatter_to_dask(client, X_train, y_train_df)
        X_val_dask, y_val_dask = scatter_to_dask(client, X_val, y_val_df)
        dtrain = DaskDMatrix(client, X_train_dask, y_train_dask)
        dval = DaskDMatrix(client, X_val_dask, y_val_dask)
        
        model = dask_train(
            client,
            xgb_params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, 'validation')],
            early_stopping_rounds=15,
            verbose_eval=25
        )
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
            temp_model_path = tmp_file.name
        try:
            model['booster'].save_model(temp_model_path)
            regular_model = XGBRegressor(**xgb_params)
            regular_model.load_model(temp_model_path)
            
            if client:
                client.run(gc.collect)
        finally:
            try:
                os.unlink(temp_model_path)
            except:
                pass
        
        # Cleanup Dask objects
        try:
            del dtrain, dval, model
            del X_train_dask, y_train_dask, X_val_dask, y_val_dask
        except:
            logging.warning("Error cleaning up Dask objects", exc_info=True)
    else:
        # Regular XGBoost training
        regular_model = XGBRegressor(
            n_estimators=num_boost_round,
            early_stopping_rounds=15,
            **xgb_params
        )
        
        regular_model.fit(
            X_train, y_train_df,
            eval_set=[(X_val, y_val_df)],
            verbose=25
        )
    
    predictions = test_xgb_autoregressively(
        X_val_with_index, 
        y_val, 
        model=regular_model, 
        disable_progress=True, 
        cache=cache
    )
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    logging.info(f"Fold {fold_num} RMSE: {rmse:.4f}")
    
    try:
        del regular_model, predictions
    except:
        logging.warning("Error cleaning up objects", exc_info=True)
    
    return rmse


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
    start_stage: int = 1,
    use_cv: bool = True,
    X_val: pd.DataFrame = None,
    y_val: np.array = None,
    X_val_with_index: pd.DataFrame = None,
    val_groups: np.array = None,
    use_dask: bool = True
) -> Tuple[Dict, Dict]:
    """
    Perform staged hyperparameter search for XGBoost.
    
    This function performs hyperparameter search in 3 stages:
    1. Tree structure (max_depth, min_child_weight)
    2. Learning rate and number of trees (eta, num_boost_round)
    3. Regularization (gamma, reg_alpha, reg_lambda)
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : np.array
        Training targets
    X_train_with_index : pd.DataFrame
        Training features with index for autoregressive testing
    train_groups : np.array
        Group labels for training set
    targets : List[str]
        Target column names
    run_id : str
        Unique identifier for this run
    start_stage : int, default=1
        Stage to start from (1, 2, or 3)
    use_cv : bool, default=True
        If True, use k-fold cross-validation on merged train+val data.
        If False, use train for training and val for validation.
    X_val : pd.DataFrame, optional
        Validation features (required if use_cv=False)
    y_val : np.array, optional
        Validation targets (required if use_cv=False)
    X_val_with_index : pd.DataFrame, optional
        Validation features with index for autoregressive testing (required if use_cv=False)
    val_groups : np.array, optional
        Group labels for validation set (not used in current implementation)
    use_dask : bool, default=True
        If True, use Dask for distributed training. If False, use regular XGBoost.
    
    Returns:
    --------
    Tuple[Dict, Dict]
        A tuple containing:
        - best_params (Dict): The final best hyperparameter combination
        - all_results (Dict): Results from all stages for visualization
    """
    if not use_cv:
        if X_val is None or y_val is None or X_val_with_index is None:
            raise ValueError("X_val, y_val, and X_val_with_index must be provided when use_cv=False")
    
    os.makedirs(os.path.join(RESULTS_PATH, run_id, "checkpoints"), exist_ok=True)
    checkpoint_subdir = "staged_search" if use_cv else "staged_search_single_val"
    if not use_dask:
        checkpoint_subdir += "_no_dask"
    checkpoint_dir = os.path.join(RESULTS_PATH, run_id, "checkpoints", checkpoint_subdir)
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
        # Create client only if using Dask
        if use_dask:
            client_context = dask_client_context(**CLIENT_CONFIGS)
        else:
            # Create a dummy context manager for non-Dask mode
            from contextlib import nullcontext
            client_context = nullcontext()
            
        with client_context as client:
            for stage_idx, (stage_name, stage_params) in enumerate(stages):
                stage_num = stage_idx + 1
                
                if stage_num < start_stage:
                    continue
                
                training_mode = "Dask" if use_dask else "Direct XGBoost"
                cv_suffix = "" if use_cv else " (Single Validation Set)"
                logging.info(f"{'='*50}")
                logging.info(f"Starting {stage_name}{cv_suffix} - {training_mode}")
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
                        if use_cv:
                            params_copy, score = train_and_evaluate_single_config(
                                X_train, y_train, X_train_with_index, train_groups, targets, params, gpu_id, client,
                                use_cv=use_cv, use_dask=use_dask
                            )
                        else:
                            params_copy, score = train_and_evaluate_single_config(
                                X_train, y_train, X_train_with_index, train_groups, targets, params, gpu_id, client,
                                use_cv=use_cv,
                                X_val=X_val, y_val=y_val, X_val_with_index=X_val_with_index,
                                use_dask=use_dask
                            )
                        result = params_copy.copy()
                        score_key = 'mean_test_score' if use_cv else 'val_score'
                        result[score_key] = score
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
        
        training_mode = "DASK" if use_dask else "DIRECT XGBOOST"
        search_type = f"STAGED SEARCH COMPLETE ({training_mode})" if use_cv else f"STAGED SEARCH COMPLETE (Single Validation Set, {training_mode})"
        logging.info(f"{'='*50}")
        logging.info(search_type)
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
    run_id: str,
    use_dask: bool = True
    ) -> None:
    logging.info("Training final model with best parameters...")

    with cuda_device("0"):  # Use GPU 0 for final training
        try:
            y_train_df = pd.DataFrame(y_train, columns=targets)
            xgb_params = get_xgb_params(best_params)
            num_boost_round = xgb_params.pop('num_boost_round')
            
            if use_dask:
                # Dask training
                with dask_client_context(**CLIENT_CONFIGS) as client:
                    X_train_dask, y_train_dask = scatter_to_dask(client, X_train, y_train_df)
                    dtrain = DaskDMatrix(client, X_train_dask, y_train_dask)

                    model = dask_train(
                        client,
                        xgb_params,
                        dtrain,
                        num_boost_round=num_boost_round,
                        verbose_eval=25
                    )

                    model_path = os.path.join(RESULTS_PATH, run_id, "checkpoints", f"final_best.json")
                    booster = model['booster']
                    booster.save_model(model_path)
                    logging.info(f"Model saved to {model_path}")
            else:
                # Regular XGBoost training
                model = XGBRegressor(
                    n_estimators=num_boost_round,
                    **xgb_params
                )
                
                model.fit(X_train, y_train_df, verbose=25)
                
                model_path = os.path.join(RESULTS_PATH, run_id, "checkpoints", f"final_best.json")
                model.save_model(model_path)
                logging.info(f"Model saved to {model_path}")

        except Exception as e:
            logging.error(f"Error during final model training: {str(e)}", exc_info=True)
            raise