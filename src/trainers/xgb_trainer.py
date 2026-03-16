from sklearn.model_selection import ParameterSampler
from sklearn.metrics import mean_squared_error
import os
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
from typing import List, Tuple, Dict, Optional
import time
import gc
from contextlib import contextmanager
from xgboost import XGBRegressor

from configs.paths import RESULTS_PATH
from src.utils.utils import get_run_root
from configs.models import (
    XGBTrainerConfig,
    XGBSearchSpace,
)
from src.trainers.evaluation import test_xgb_autoregressively

FINAL_MODEL_FILENAME = "final_best.json"


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


def _first_visible_gpu() -> str:
    """Return the first GPU ID from CUDA_VISIBLE_DEVICES, or '0' if not set or invalid."""
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if cuda_visible:
        first = cuda_visible.split(',')[0].strip()
        if first.lstrip('-').isdigit():
            return first
    return '0'


def get_xgb_params(base_params: dict, trainer_cfg: Optional[XGBTrainerConfig] = None) -> dict:
    """Compose final XGBoost params from base hyperparams and trainer defaults."""
    if trainer_cfg is None:
        trainer_cfg = XGBTrainerConfig()
    return {
        'tree_method': trainer_cfg.tree_method,
        'device': trainer_cfg.device,
        'eval_metric': trainer_cfg.eval_metric,
        'verbosity': trainer_cfg.verbosity,
        'max_bin': trainer_cfg.max_bin,
        **base_params,
    }


def train_and_evaluate_single_config(
    X, y, X_with_index, train_groups, targets, params, gpu_id,
    use_cv=True, X_val=None, y_val=None, X_val_with_index=None, n_folds=5,
    early_stopping_rounds: int = 15, trainer_cfg: Optional[XGBTrainerConfig] = None,
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
        
    Returns:
    --------
    Tuple[Dict, float]: A tuple containing the parameters and the negative RMSE score.
    """
    with cuda_device(str(gpu_id)):
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
                        targets, params, fold_cache[cache_key], fold+1,
                        early_stopping_rounds=early_stopping_rounds, trainer_cfg=trainer_cfg,
                        show_autoreg_progress=show_autoreg_progress,
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
                    targets, params, {}, 1,
                    early_stopping_rounds=early_stopping_rounds, trainer_cfg=trainer_cfg,
                    show_autoreg_progress=show_autoreg_progress,
                )
                score = -fold_score

            return params, score
            
        except Exception as e:
            logging.error(f"Error training config {params}: {str(e)}", exc_info=True)
            raise
        finally:
            try:
                gc.collect()
            except:
                pass


def _train_single_fold(
    X_train, y_train, X_val, y_val, X_val_with_index, targets, params, cache, fold_num,
    early_stopping_rounds: int = 15, trainer_cfg: Optional[XGBTrainerConfig] = None,
):
    """
    Helper function to train and evaluate a single fold/validation set.
    
    Returns:
    --------
    float: RMSE score for this fold
    """
    y_train_df = pd.DataFrame(y_train, columns=targets)
    y_val_df = pd.DataFrame(y_val, columns=targets)
    
    xgb_params = get_xgb_params(params, trainer_cfg=trainer_cfg)
    num_boost_round = xgb_params.pop('num_boost_round')

    regular_model = XGBRegressor(
        n_estimators=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
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
        disable_progress=(not show_autoreg_progress),
        cache=cache,
    )
    ar_dt = time.perf_counter() - ar_t0
    logging.info(
        "Fold %d autoregressive validation done in %.2fs (show_progress=%s)",
        fold_num,
        ar_dt,
        str(show_autoreg_progress),
    )
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    logging.info(f"Fold {fold_num} RMSE: {rmse:.4f}")
    
    try:
        del regular_model, predictions
    except:
        logging.warning("Error cleaning up objects", exc_info=True)
    
    return rmse


# build_param_dist is provided by XGBSearchSpace to keep config logic with configs


def hyperparameter_search(
    X_train: pd.DataFrame, 
    y_train: np.array, 
    X_train_with_index: pd.DataFrame,
    train_groups: np.array,
    targets: List[str], 
    run_id: str, 
    start_stage: int = 1,
    use_cv: bool = True,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[np.array] = None,
    X_val_with_index: Optional[pd.DataFrame] = None,
    val_groups: Optional[np.array] = None,
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
    
    os.makedirs(os.path.join(get_run_root(run_id), "checkpoints"), exist_ok=True)
    checkpoint_subdir = "staged_search" if use_cv else "staged_search_single_val"
    checkpoint_dir = os.path.join(RESULTS_PATH, run_id, "checkpoints", checkpoint_subdir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    all_results = {
        'stage_1': [],
        'stage_2': [],
        'stage_3': []
    }
    
    trainer_cfg = XGBTrainerConfig()

    best_params = {}
    overall_best_score = float('-inf')
    overall_best_params = None
    search_space = XGBSearchSpace()
    stages = search_space.stages()
    
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
        for stage_idx, (stage_name, stage_params) in enumerate(stages):
            stage_num = stage_idx + 1
            
            if stage_num < start_stage:
                continue
            
            cv_suffix = "" if use_cv else " (Single Validation Set)"
            logging.info(f"{'='*50}")
            logging.info(f"Starting {stage_name}{cv_suffix}")
            logging.info(f"{'='*50}")
            
            current_param_dist = XGBSearchSpace.build_param_dist(stage_params, best_params)
            
            if not current_param_dist:
                logging.warning(f"No parameters to search in {stage_name}, skipping...")
                continue
            
            stage_results = []
            stage_best_score = float('-inf')
            stage_best_params = None
            
            param_sampler = ParameterSampler(
                current_param_dist,
                n_iter=trainer_cfg.search_iter_n_per_stage,
                random_state=stage_idx,
            )
            
            for i, params in enumerate(param_sampler):
                logging.info(f"{stage_name} - Iteration {i+1}/{trainer_cfg.search_iter_n_per_stage}")
                
                # Select GPU from configured pool round-robin
                gpu_id = trainer_cfg.gpu_ids[i % len(trainer_cfg.gpu_ids)]
                
                try:
                    if use_cv:
                        params_copy, score = train_and_evaluate_single_config(
                            X_train, y_train, X_train_with_index, train_groups, targets, params, gpu_id,
                            use_cv=use_cv,
                            n_folds=trainer_cfg.n_folds,
                            early_stopping_rounds=trainer_cfg.early_stopping_rounds,
                            trainer_cfg=trainer_cfg,
                        )
                    else:
                        params_copy, score = train_and_evaluate_single_config(
                            X_train, y_train, X_train_with_index, train_groups, targets, params, gpu_id,
                            use_cv=use_cv,
                            X_val=X_val, y_val=y_val, X_val_with_index=X_val_with_index,
                            n_folds=trainer_cfg.n_folds,
                            early_stopping_rounds=trainer_cfg.early_stopping_rounds,
                            trainer_cfg=trainer_cfg,
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
        
        search_type = "STAGED SEARCH COMPLETE" if use_cv else "STAGED SEARCH COMPLETE (Single Validation Set)"
        logging.info(f"{'='*50}")
        logging.info(search_type)
        logging.info(f"{'='*50}")
        logging.info(f"Overall Best RMSE: {-overall_best_score:.4f}")
        logging.info(f"Final Best Parameters: {overall_best_params}")

        # Ensure non-None dict for type safety
        safe_best = overall_best_params or best_params or {}
        return safe_best, all_results

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
    ) -> None:
    logging.info("Training final model with best parameters...")

    with cuda_device(_first_visible_gpu()):
        try:
            y_train_df = pd.DataFrame(y_train, columns=targets)
            trainer_cfg = XGBTrainerConfig()
            xgb_params = get_xgb_params(best_params, trainer_cfg=trainer_cfg)
            num_boost_round = xgb_params.pop('num_boost_round')

            model = XGBRegressor(
                n_estimators=num_boost_round,
                **xgb_params
            )

            model.fit(X_train, y_train_df, verbose=25)

            model_path = os.path.join(RESULTS_PATH, run_id, "checkpoints", FINAL_MODEL_FILENAME)
            model.save_model(model_path)
            logging.info(f"Model saved to {model_path}")

        except Exception as e:
            logging.error(f"Error during final model training: {str(e)}", exc_info=True)
            raise