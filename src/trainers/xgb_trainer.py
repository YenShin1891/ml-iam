import gc
import json
import logging
import os
import queue
import subprocess
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import multiprocessing as mp
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterSampler
from xgboost import XGBRegressor

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


def _parse_cuda_visible_devices(value: str) -> List[str]:
    return [tok.strip() for tok in value.split(',') if tok.strip()]


def _visible_gpu_pool() -> List[str]:
    """Return a list of GPU tokens that should be used for multi-GPU search.

    Priority:
    1) Respect CUDA_VISIBLE_DEVICES if set (tokens may be indices or UUIDs).
    2) Otherwise, fall back to enumerating GPUs via `nvidia-smi -L`.

    Returns at least one token (defaults to ['0']).
    """
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible:
        tokens = _parse_cuda_visible_devices(cuda_visible)
        if tokens:
            return tokens

    try:
        out = subprocess.check_output(['nvidia-smi', '-L'], stderr=subprocess.DEVNULL, text=True)
        gpu_lines = [ln for ln in out.splitlines() if ln.strip().startswith('GPU ')]
        if gpu_lines:
            return [str(i) for i in range(len(gpu_lines))]
    except Exception:
        pass

    return ['0']


def _first_visible_gpu_token(gpu_pool: Optional[List[str]] = None) -> str:
    pool = gpu_pool or _visible_gpu_pool()
    return pool[0] if pool else '0'


def _cap_search_cpu_threads() -> None:
    """Cap CPU thread usage inside search workers.

    Parallel search runs multiple GPU workers concurrently; leaving BLAS/OMP and
    XGBoost threading uncapped can easily oversubscribe CPUs.
    """
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'


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
    X,
    y,
    X_with_index,
    train_groups,
    targets,
    params,
    *,
    use_cv: bool = True,
    X_val=None,
    y_val=None,
    X_val_with_index=None,
    n_folds: int = 5,
    early_stopping_rounds: int = 15,
    trainer_cfg: Optional[XGBTrainerConfig] = None,
    show_autoreg_progress: bool = False,
    n_jobs: Optional[int] = None,
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
    try:
        if use_cv:
            scores: List[float] = []
            fold_cache: Dict[str, Dict] = {}
            for fold, (train_idx, val_idx) in enumerate(
                group_k_fold_split(train_groups, n_splits=n_folds, shuffle=True, random_state=42)
            ):
                logging.info("Fold %d/%d for params: %s", fold + 1, n_folds, params)

                X_train, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                X_val_with_index_fold = X_with_index.iloc[val_idx]
                y_train, y_val_fold = y[train_idx], y[val_idx]

                cache_key = f"fold_{fold}"
                if cache_key not in fold_cache:
                    fold_cache[cache_key] = {}

                fold_rmse = _train_single_fold(
                    X_train,
                    y_train,
                    X_val_fold,
                    y_val_fold,
                    X_val_with_index_fold,
                    targets,
                    params,
                    fold_cache[cache_key],
                    fold + 1,
                    early_stopping_rounds=early_stopping_rounds,
                    trainer_cfg=trainer_cfg,
                    show_autoreg_progress=show_autoreg_progress,
                    n_jobs=n_jobs,
                )
                scores.append(fold_rmse)

            avg_rmse = float(np.mean(scores))
            score = -avg_rmse
        else:
            if X_val is None or y_val is None or X_val_with_index is None:
                raise ValueError("X_val, y_val, and X_val_with_index must be provided when use_cv=False")

            logging.info("Training with params: %s", params)
            rmse = _train_single_fold(
                X,
                y,
                X_val,
                y_val,
                X_val_with_index,
                targets,
                params,
                {},
                1,
                early_stopping_rounds=early_stopping_rounds,
                trainer_cfg=trainer_cfg,
                show_autoreg_progress=show_autoreg_progress,
                n_jobs=n_jobs,
            )
            score = -rmse

        return params, float(score)
    except Exception as e:
        logging.error("Error training config %s: %s", params, str(e), exc_info=True)
        raise
    finally:
        try:
            gc.collect()
        except Exception:
            pass


def _train_single_fold(
    X_train, y_train, X_val, y_val, X_val_with_index, targets, params, cache, fold_num,
    early_stopping_rounds: int = 15, trainer_cfg: Optional[XGBTrainerConfig] = None,
    show_autoreg_progress: bool = False,
    n_jobs: Optional[int] = None,
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
    if n_jobs is not None:
        xgb_params['n_jobs'] = int(n_jobs)
    num_boost_round = xgb_params.pop('num_boost_round')

    regular_model = XGBRegressor(
        n_estimators=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        **xgb_params
    )
    fit_t0 = time.perf_counter()
    regular_model.fit(
        X_train,
        y_train_df,
        eval_set=[(X_val, y_val_df)],
        verbose=25,
    )
    fit_dt = time.perf_counter() - fit_t0
    logging.info(
        "Fold %d fit() done in %.2fs (train_rows=%d, val_rows=%d, num_boost_round=%s)",
        fold_num,
        fit_dt,
        int(getattr(X_train, 'shape', [len(X_train)])[0]),
        int(getattr(X_val, 'shape', [len(X_val)])[0]),
        str(params.get('num_boost_round')),
    )

    ar_t0 = time.perf_counter()
    predictions = test_xgb_autoregressively(
        X_val_with_index,
        y_val,
        model=regular_model,
        disable_progress=(not show_autoreg_progress),
        max_workers=1,
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


def _search_worker(
    gpu_token: str,
    assignments: List[Tuple[int, Dict]],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_train_with_index: pd.DataFrame,
    train_groups: np.ndarray,
    targets: List[str],
    *,
    use_cv: bool,
    X_val: Optional[pd.DataFrame],
    y_val: Optional[np.ndarray],
    X_val_with_index: Optional[pd.DataFrame],
    n_folds: int,
    early_stopping_rounds: int,
    trainer_cfg: XGBTrainerConfig,
    stage_num: int,
    score_key: str,
    result_queue,
) -> None:
    """Worker process: pinned to exactly one GPU via CUDA_VISIBLE_DEVICES."""
    with cuda_device(gpu_token):
        _cap_search_cpu_threads()

        for trial_idx, params in assignments:
            params_copy, score = train_and_evaluate_single_config(
                X_train,
                y_train,
                X_train_with_index,
                train_groups,
                targets,
                params,
                use_cv=use_cv,
                X_val=X_val,
                y_val=y_val,
                X_val_with_index=X_val_with_index,
                n_folds=n_folds,
                early_stopping_rounds=early_stopping_rounds,
                trainer_cfg=trainer_cfg,
                show_autoreg_progress=trainer_cfg.search_show_autoreg_progress,
                n_jobs=1,
            )
            result = params_copy.copy()
            result[score_key] = float(score)
            result['stage'] = int(stage_num)
            result['gpu'] = str(gpu_token)
            result['trial'] = int(trial_idx)
            result_queue.put(result)


def _collect_worker_results(processes: List[mp.Process], result_queue) -> List[Dict]:
    results: List[Dict] = []

    # Drain results while workers run (avoid queue backpressure).
    while True:
        alive = any(p.is_alive() for p in processes)
        try:
            results.append(result_queue.get(timeout=0.5 if alive else 0.1))
        except queue.Empty:
            if not alive:
                break

    for p in processes:
        p.join()

    # Drain any remaining items
    while True:
        try:
            results.append(result_queue.get_nowait())
        except queue.Empty:
            break

    bad = [p for p in processes if p.exitcode not in (0, None)]
    if bad:
        raise RuntimeError(
            "One or more XGB search workers crashed: "
            + ", ".join(f"pid={p.pid} exitcode={p.exitcode}" for p in bad)
        )

    return results


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
    
    run_root = get_run_root(run_id)
    os.makedirs(os.path.join(run_root, "checkpoints"), exist_ok=True)
    checkpoint_subdir = "staged_search" if use_cv else "staged_search_single_val"
    checkpoint_dir = os.path.join(run_root, "checkpoints", checkpoint_subdir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    all_results = {
        'stage_1': [],
        'stage_2': [],
        'stage_3': []
    }
    
    trainer_cfg = XGBTrainerConfig()
    gpu_pool = _visible_gpu_pool()
    logging.info("XGB search GPU pool: %s (CUDA_VISIBLE_DEVICES=%s)", gpu_pool, os.environ.get('CUDA_VISIBLE_DEVICES'))

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
            
            params_list = list(
                ParameterSampler(
                    current_param_dist,
                    n_iter=trainer_cfg.search_iter_n_per_stage,
                    random_state=stage_idx,
                )
            )

            score_key = 'mean_test_score' if use_cv else 'val_score'
            expected_results = len(params_list)

            if len(gpu_pool) <= 1:
                # Single GPU visible (or forced): run sequentially, but still cap threads.
                with cuda_device(_first_visible_gpu_token(gpu_pool)):
                    _cap_search_cpu_threads()
                    for i, params in enumerate(params_list):
                        logging.info(
                            "%s - Iteration %d/%d",
                            stage_name,
                            i + 1,
                            expected_results,
                        )
                        params_copy, score = train_and_evaluate_single_config(
                            X_train,
                            y_train,
                            X_train_with_index,
                            train_groups,
                            targets,
                            params,
                            use_cv=use_cv,
                            X_val=X_val,
                            y_val=y_val,
                            X_val_with_index=X_val_with_index,
                            n_folds=trainer_cfg.n_folds,
                            early_stopping_rounds=trainer_cfg.early_stopping_rounds,
                            trainer_cfg=trainer_cfg,
                            show_autoreg_progress=trainer_cfg.search_show_autoreg_progress,
                            n_jobs=1,
                        )
                        result = params_copy.copy()
                        result[score_key] = float(score)
                        result['stage'] = stage_num
                        stage_results.append(result)
            else:
                # Multi-GPU: one worker process per GPU.
                try:
                    ctx = mp.get_context('fork')
                except Exception:
                    ctx = mp.get_context()

                result_queue = ctx.Queue()
                assignments: List[List[Tuple[int, Dict]]] = [[] for _ in gpu_pool]
                for i, params in enumerate(params_list):
                    assignments[i % len(gpu_pool)].append((i, params))

                processes: List[mp.Process] = []
                for gpu_token, trials in zip(gpu_pool, assignments):
                    if not trials:
                        continue
                    p = ctx.Process(
                        target=_search_worker,
                        args=(
                            gpu_token,
                            trials,
                            X_train,
                            y_train,
                            X_train_with_index,
                            train_groups,
                            targets,
                        ),
                        kwargs={
                            'use_cv': use_cv,
                            'X_val': X_val,
                            'y_val': y_val,
                            'X_val_with_index': X_val_with_index,
                            'n_folds': trainer_cfg.n_folds,
                            'early_stopping_rounds': trainer_cfg.early_stopping_rounds,
                            'trainer_cfg': trainer_cfg,
                            'stage_num': stage_num,
                            'score_key': score_key,
                            'result_queue': result_queue,
                        },
                    )
                    p.start()
                    processes.append(p)

                stage_results = _collect_worker_results(processes, result_queue)

            if len(stage_results) != expected_results:
                raise RuntimeError(
                    f"XGB search stage {stage_num} produced {len(stage_results)}/{expected_results} results"
                )

            # Determine best params for this stage and overall.
            for r in stage_results:
                score = float(r[score_key])
                if score > stage_best_score:
                    stage_best_score = score
                    stage_best_params = {k: v for k, v in r.items() if k in current_param_dist}
                if score > overall_best_score:
                    overall_best_score = score
                    overall_best_params = {k: v for k, v in r.items() if k in current_param_dist or k in best_params}

            for r in stage_results:
                logging.info("%s trial=%s gpu=%s RMSE: %.4f", stage_name, r.get('trial'), r.get('gpu'), -float(r[score_key]))
        
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

    with cuda_device(_first_visible_gpu_token()):
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

            run_root = get_run_root(run_id)
            os.makedirs(os.path.join(run_root, "checkpoints"), exist_ok=True)
            model_path = os.path.join(run_root, "checkpoints", FINAL_MODEL_FILENAME)
            model.save_model(model_path)
            logging.info(f"Model saved to {model_path}")

        except Exception as e:
            logging.error(f"Error during final model training: {str(e)}", exc_info=True)
            raise