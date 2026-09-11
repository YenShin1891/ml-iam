import gc
import logging
import os
import queue
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import multiprocessing as mp
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from src.utils.utils import format_duration, get_run_root
from configs.models import (
    XGBTrainerConfig,
    XGBSearchSpace,
)
from configs.data import N_LAG_FEATURES
from src.trainers.evaluation import test_xgb_autoregressively
from .search import (
    allocate_pool,
    best_trial,
    completed_trials,
    params_signature,
    select_top_k_signatures,
    write_search_report,
)

FINAL_MODEL_FILENAME = "final_best.json"


class PerTargetXGBRegressor:
    """Wrapper that trains one XGBRegressor per target, dropping NaN rows
    independently per target.  Exposes the same `predict()` / `save_model()` /
    `load_model()` interface as XGBRegressor so that autoregressive evaluation
    works without modification.
    """

    def __init__(self, targets: List[str], **xgb_kwargs):
        self.targets = list(targets)
        self.models: List[XGBRegressor] = [
            XGBRegressor(**xgb_kwargs) for _ in targets
        ]

    def fit(self, X, y_df, *, eval_set=None, verbose=25):
        """Fit one model per target, dropping rows with NaN for that target.

        *y_df* must be a DataFrame with columns matching ``self.targets``.
        *eval_set* is a list of ``(X_val, y_val_df)`` tuples — each val y is
        also a DataFrame.
        """
        for i, (target, model) in enumerate(zip(self.targets, self.models)):
            y_col = y_df[target]
            valid = y_col.notna()
            X_t = X[valid] if isinstance(X, pd.DataFrame) else X[valid.values]
            y_t = y_col[valid]
            ev = None
            if eval_set:
                ev = []
                for Xv, yv_df in eval_set:
                    yv_col = yv_df[target]
                    vv = yv_col.notna()
                    Xv_t = Xv[vv] if isinstance(Xv, pd.DataFrame) else Xv[vv.values]
                    yv_t = yv_col[vv]
                    ev.append((Xv_t, yv_t))
            logging.info(
                "PerTarget[%s] fitting on %d/%d rows (dropped %d NaN)",
                target, int(valid.sum()), len(y_col), int((~valid).sum()),
            )
            model.fit(X_t, y_t, eval_set=ev, verbose=verbose)

    @property
    def best_iteration(self) -> int:
        """Rounds the slowest target needed under early stopping.

        The final model is fitted once for all targets, so it has to be long
        enough for the one that converged last; taking the max is the only
        choice that does not under-train a target.
        """
        iterations = [
            int(getattr(model, "best_iteration", 0) or 0) for model in self.models
        ]
        return max(iterations) if iterations else 0

    def predict(self, X):
        """Predict all targets; returns (n, n_targets) array."""
        preds = []
        for model in self.models:
            p = model.predict(X)
            if p.ndim == 1:
                p = p.reshape(-1, 1)
            preds.append(p)
        return np.hstack(preds)

    def save_model(self, path):
        """Save each model as ``{stem}_{i}.json``."""
        stem, ext = os.path.splitext(path)
        for i, model in enumerate(self.models):
            model.save_model(f"{stem}_{i}{ext}")

    @classmethod
    def load_model(cls, path, targets: List[str]):
        """Load per-target models from ``{stem}_{i}.json`` files."""
        stem, ext = os.path.splitext(path)
        obj = cls.__new__(cls)
        obj.targets = list(targets)
        obj.models = []
        for i in range(len(targets)):
            m = XGBRegressor()
            m.load_model(f"{stem}_{i}{ext}")
            obj.models.append(m)
        return obj


def final_model_path(run_id: str) -> str:
    """Path of a run's final model (the per-target files add an ``_{i}`` suffix)."""
    return os.path.join(get_run_root(run_id), "checkpoints", FINAL_MODEL_FILENAME)


def count_final_target_models(run_id: str) -> int:
    """How many ``final_best_{i}.json`` files the run wrote (0 if none)."""
    stem, ext = os.path.splitext(final_model_path(run_id))
    n = 0
    while os.path.exists(f"{stem}_{n}{ext}"):
        n += 1
    return n


def has_final_xgb_model(run_id: str) -> bool:
    """True when either model layout is present on disk."""
    return count_final_target_models(run_id) > 0 or os.path.exists(final_model_path(run_id))


def load_final_xgb_model(run_id: str, targets: Optional[List[str]] = None):
    """Load a run's final model in whichever layout it was saved.

    KEEP_PARTIAL_TARGETS writes one booster per target as ``final_best_{i}.json``;
    otherwise a single multi-output ``final_best.json`` is written.  Every caller
    should come through here — checking only for ``final_best.json`` silently
    skips per-target runs.
    """
    path = final_model_path(run_id)
    n_target_models = count_final_target_models(run_id)

    if n_target_models:
        if targets is None:
            from configs.data import OUTPUT_VARIABLES
            targets = OUTPUT_VARIABLES
        if len(targets) != n_target_models:
            logging.warning(
                "Run %s has %d per-target models but %d targets were requested; "
                "using the first %d.",
                run_id, n_target_models, len(targets), n_target_models,
            )
        return PerTargetXGBRegressor.load_model(path, list(targets)[:n_target_models])

    if os.path.exists(path):
        model = XGBRegressor()
        model.load_model(path)
        return model

    raise FileNotFoundError(
        f"No final XGBoost model for run {run_id}: expected {path} or {path[:-5]}_0.json"
    )


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


# Searched settings that select or shape the data rather than the booster.
# They ride along in the params dict so they reach the trial row and
# best_params, but XGBRegressor must never see them.
_NON_BOOSTER_PARAMS = ("n_lags",)


def get_xgb_params(base_params: dict, trainer_cfg: Optional[XGBTrainerConfig] = None) -> dict:
    """Compose final XGBoost params from base hyperparams and trainer defaults."""
    if trainer_cfg is None:
        trainer_cfg = XGBTrainerConfig()
    base_params = {k: v for k, v in base_params.items() if k not in _NON_BOOSTER_PARAMS}
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
    use_autoregressive_eval: bool = True,
    obs_mask: Optional[np.ndarray] = None,
    obs_val_mask: Optional[np.ndarray] = None,
    x_scaler=None,
    y_scaler=None,
) -> Tuple[Dict, float, int]:
    """
    Train and evaluate a single parameter configuration using either k-fold CV or single validation set.

    *x_scaler* and *y_scaler* are the split's fitted scalers.  The rollout
    needs both: it writes each step's prediction, which is in target units,
    into the next step's lag column, which is in feature units, and the two
    standardisations differ (by 15% in scale for Solar).  Without them the
    search scored stage 2 on a rollout the test phase never runs.
    
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
    Tuple[Dict, float, int, Optional[float]]: the parameters, the negative
    RMSE score, the boosting round early stopping settled on (the round count
    the final model needs, in place of searching for one), and the negative
    one-step RMSE when the score came from the rollout, else None.
    """
    try:
        if use_cv:
            scores: List[float] = []
            one_steps: List[float] = []
            best_iterations: List[int] = []
            fold_cache: Dict[str, Dict] = {}
            for fold, (train_idx, val_idx) in enumerate(
                group_k_fold_split(train_groups, n_splits=n_folds, shuffle=True, random_state=42)
            ):
                logging.info("Fold %d/%d for params: %s", fold + 1, n_folds, params)

                X_train, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                X_val_with_index_fold = X_with_index.iloc[val_idx]
                y_train, y_val_fold = y[train_idx], y[val_idx]
                obs_t = obs_mask[train_idx] if obs_mask is not None else None
                obs_v = obs_mask[val_idx] if obs_mask is not None else None

                cache_key = f"fold_{fold}"
                if cache_key not in fold_cache:
                    fold_cache[cache_key] = {}

                fold_rmse, fold_best_iteration, fold_one_step = _train_single_fold(
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
                    use_autoregressive_eval=use_autoregressive_eval,
                    obs_train=obs_t,
                    obs_val=obs_v,
                    x_scaler=x_scaler,
                    y_scaler=y_scaler,
                )
                scores.append(fold_rmse)
                best_iterations.append(fold_best_iteration)
                if fold_one_step is not None:
                    one_steps.append(fold_one_step)

            avg_rmse = float(np.mean(scores))
            score = -avg_rmse
            one_step_score = -float(np.mean(one_steps)) if one_steps else None
        else:
            if X_val is None or y_val is None or X_val_with_index is None:
                raise ValueError("X_val, y_val, and X_val_with_index must be provided when use_cv=False")

            logging.info("Training with params: %s", params)
            rmse, best_iteration, one_step_rmse = _train_single_fold(
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
                use_autoregressive_eval=use_autoregressive_eval,
                obs_train=obs_mask,
                obs_val=obs_val_mask,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
            )
            score = -rmse
            best_iterations = [best_iteration]
            one_step_score = -one_step_rmse if one_step_rmse is not None else None

        return params, float(score), max(best_iterations) if best_iterations else 0, one_step_score
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
    use_autoregressive_eval: bool = True,
    obs_train: Optional[np.ndarray] = None,
    obs_val: Optional[np.ndarray] = None,
    x_scaler=None,
    y_scaler=None,
):
    """
    Helper function to train and evaluate a single fold/validation set.
    
    Returns:
    --------
    Tuple[float, int, Optional[float]]: RMSE for this fold, the boosting
    round early stopping settled on, and -- when the fold is scored on the
    rollout -- the one-step RMSE as well.  One-step error is what stage 1
    used to rank on; keeping it next to the rollout score is what lets the
    paper say how well the cheap proxy predicted the objective.
    """
    from configs.data import KEEP_PARTIAL_TARGETS

    y_train_df = pd.DataFrame(y_train, columns=targets)
    y_val_df = pd.DataFrame(y_val, columns=targets)

    xgb_params = get_xgb_params(params, trainer_cfg=trainer_cfg)
    if n_jobs is not None:
        xgb_params['n_jobs'] = int(n_jobs)
    num_boost_round = xgb_params.pop('num_boost_round')

    if KEEP_PARTIAL_TARGETS:
        regular_model = PerTargetXGBRegressor(
            targets=targets,
            n_estimators=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            **xgb_params,
        )
    else:
        regular_model = XGBRegressor(
            n_estimators=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            **xgb_params,
        )
    fit_t0 = time.perf_counter()
    regular_model.fit(
        X_train,
        y_train_df,
        eval_set=[(X_val, y_val_df)],
        verbose=(trainer_cfg or XGBTrainerConfig()).search_fit_verbose,
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

    # RMSE on observed elements only.  Unobserved targets are NaN (see
    # prepare_data), so fall back to finiteness when no mask is supplied.
    y_flat = np.asarray(y_val, dtype=float).flatten()
    if obs_val is not None:
        mask = obs_val.astype(bool).flatten() & np.isfinite(y_flat)
    else:
        mask = np.isfinite(y_flat)
    if not mask.any():
        raise ValueError("No observed validation targets to score this fold on")

    def observed_rmse(predictions) -> float:
        pred_flat = np.asarray(predictions, dtype=float).flatten()
        return float(np.sqrt(mean_squared_error(y_flat[mask], pred_flat[mask])))

    one_step_rmse: Optional[float] = None
    ar_t0 = time.perf_counter()
    if use_autoregressive_eval:
        one_step_rmse = observed_rmse(regular_model.predict(X_val))
        predictions = test_xgb_autoregressively(
            X_val_with_index,
            y_val,
            model=regular_model,
            disable_progress=(not show_autoreg_progress),
            cache=cache,
            n_lags=int(params.get('n_lags', N_LAG_FEATURES)),
            y_scaler=y_scaler,
            x_scaler=x_scaler,
        )
        ar_dt = time.perf_counter() - ar_t0
        logging.info(
            "Fold %d autoregressive validation done in %.2fs (show_progress=%s)",
            fold_num,
            ar_dt,
            str(show_autoreg_progress),
        )
    else:
        predictions = regular_model.predict(X_val)
        ar_dt = time.perf_counter() - ar_t0
        logging.info("Fold %d standard validation done in %.2fs", fold_num, ar_dt)

    rmse = observed_rmse(predictions)
    best_iteration = int(getattr(regular_model, "best_iteration", 0) or 0)
    logging.info("Fold %d RMSE: %.4f (best iteration %d)", fold_num, rmse, best_iteration)

    del regular_model, predictions

    return rmse, best_iteration, one_step_rmse


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
    stage: str,
    score_key: str,
    result_queue,
    use_autoregressive_eval: bool = False,
    obs_train: Optional[np.ndarray] = None,
    obs_val: Optional[np.ndarray] = None,
    x_scaler=None,
    y_scaler=None,
) -> None:
    """Worker process: pinned to exactly one GPU via CUDA_VISIBLE_DEVICES."""
    with cuda_device(gpu_token):
        _cap_search_cpu_threads()

        for trial_idx, params in assignments:
            params_copy, score, best_iteration, one_step = train_and_evaluate_single_config(
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
                use_autoregressive_eval=use_autoregressive_eval,
                obs_mask=obs_train,
                obs_val_mask=obs_val,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
            )
            result = params_copy.copy()
            result[score_key] = float(score)
            result['best_iteration'] = int(best_iteration)
            result['stage'] = stage
            result['status'] = 'completed'
            result['gpu'] = str(gpu_token)
            result['trial'] = int(trial_idx)
            if one_step is not None:
                result['val_score_one_step'] = float(one_step)
            result_queue.put(result)


def _log_trial_result(
    stage_name: str,
    result: Dict,
    rmse: float,
    completed: int,
    expected: int,
    best: float,
    varied_params: Sequence[str] = (),
) -> None:
    """Log one finished trial, identically from the parallel and sequential paths."""
    settings = ", ".join(f"{name}={result[name]}" for name in varied_params if name in result)
    logging.info(
        "%s %d/%d trial=%s gpu=%s RMSE=%.4f best=%.4f%s",
        stage_name,
        completed,
        expected,
        result.get('trial', '-'),
        result.get('gpu', '-'),
        rmse,
        best,
        f" | {settings}" if settings else "",
    )


def _collect_worker_results(
    processes: List[mp.Process],
    result_queue,
    *,
    stage_name: str,
    expected: int,
    score_key: str,
    varied_params: Sequence[str] = (),
    progress: Optional[Dict] = None,
) -> List[Dict]:
    """Drain worker results, logging each trial the moment it lands.

    Logging here rather than after the stage is what makes a search
    observable: workers finish minutes apart, but a stage-end loop stamped
    every line with the same timestamp, so a 40-minute stage read as 24 trials
    completing at once with no sign of life in between.

    *progress* holds the count and best score for the stage as a whole.  It is
    passed in rather than kept here because a stage's trials do not always
    reach one queue: when the GPU pool cannot be divided, the lag groups run
    back to back, and each call would otherwise restart the count at 1.
    """
    results: List[Dict] = []
    if progress is None:
        progress = {'done': 0, 'best': float('inf')}

    def record(result: Dict) -> None:
        results.append(result)
        rmse = -float(result[score_key])
        progress['done'] += 1
        progress['best'] = min(progress['best'], rmse)
        _log_trial_result(
            stage_name, result, rmse, progress['done'],
            expected, progress['best'], varied_params,
        )

    # Drain results while workers run (avoid queue backpressure).
    while True:
        alive = any(p.is_alive() for p in processes)
        try:
            record(result_queue.get(timeout=0.5 if alive else 0.1))
        except queue.Empty:
            if not alive:
                break

    for p in processes:
        p.join()

    # Drain any remaining items
    while True:
        try:
            record(result_queue.get_nowait())
        except queue.Empty:
            break

    # TODO: a dead worker is only noticed once every worker has finished; see
    # the note on tft_trainer._collect_search_results.
    bad = [p for p in processes if p.exitcode not in (0, None)]
    if bad:
        raise RuntimeError(
            "One or more XGB search workers crashed: "
            + ", ".join(f"pid={p.pid} exitcode={p.exitcode}" for p in bad)
        )

    return results


@dataclass
class _XGBTrialInputs:
    """Data and evaluation settings shared by every trial in a search."""

    X_train: pd.DataFrame
    y_train: np.ndarray
    X_train_with_index: pd.DataFrame
    train_groups: np.ndarray
    targets: List[str]
    use_cv: bool
    X_val: Optional[pd.DataFrame]
    y_val: Optional[np.ndarray]
    X_val_with_index: Optional[pd.DataFrame]
    trainer_cfg: XGBTrainerConfig
    obs_train: Optional[np.ndarray]
    obs_val: Optional[np.ndarray]
    # The split's fitted scalers.  The stage-2 rollout cannot run without
    # them: see train_and_evaluate_single_config.
    x_scaler: Any = None
    y_scaler: Any = None

    @property
    def score_key(self) -> str:
        return 'mean_test_score' if self.use_cv else 'val_score'


def _with_budget(params_list: Sequence[Dict], num_boost_round: int) -> List[Dict]:
    """Attach the stage's boosting budget to each trial's parameters.

    It rides along in the params because that is how it reaches XGBRegressor,
    through get_xgb_params, which is where n_estimators is read from.
    """
    return [{**params, 'num_boost_round': int(num_boost_round)} for params in params_list]


def _spawn_search_workers(
    ctx,
    trials: Sequence[Tuple[int, Dict]],
    inputs: _XGBTrialInputs,
    stage: str,
    gpus: Sequence[str],
    result_queue,
    use_autoregressive_eval: bool,
) -> List[mp.Process]:
    """Start one worker per GPU in *gpus*, all reporting to *result_queue*.

    Spawning is separated from collecting so that several groups of trials --
    which cannot share a worker, because a worker is given one prepared
    feature frame -- can still run at the same time against one queue.
    """
    assignments: List[List[Tuple[int, Dict]]] = [[] for _ in gpus]
    for position, trial in enumerate(trials):
        assignments[position % len(gpus)].append(trial)

    trainer_cfg = inputs.trainer_cfg
    processes: List[mp.Process] = []
    for gpu_token, assigned in zip(gpus, assignments):
        if not assigned:
            continue
        process = ctx.Process(
            target=_search_worker,
            args=(
                gpu_token,
                assigned,
                inputs.X_train,
                inputs.y_train,
                inputs.X_train_with_index,
                inputs.train_groups,
                inputs.targets,
            ),
            kwargs={
                'use_cv': inputs.use_cv,
                'X_val': inputs.X_val,
                'y_val': inputs.y_val,
                'X_val_with_index': inputs.X_val_with_index,
                'n_folds': trainer_cfg.n_folds,
                'early_stopping_rounds': trainer_cfg.early_stopping_rounds,
                'trainer_cfg': trainer_cfg,
                'use_autoregressive_eval': use_autoregressive_eval,
                'stage': stage,
                'score_key': inputs.score_key,
                'result_queue': result_queue,
                'obs_train': inputs.obs_train,
                'obs_val': None if inputs.use_cv else inputs.obs_val,
                'x_scaler': inputs.x_scaler,
                'y_scaler': inputs.y_scaler,
            },
        )
        process.start()
        processes.append(process)
    return processes


def _search_context():
    """A multiprocessing context that does not fork a CUDA-touched parent."""
    try:
        return mp.get_context('forkserver')
    except Exception:
        return mp.get_context('spawn')


def _run_xgb_trials_sequentially(
    trials: Sequence[Tuple[int, Dict]],
    inputs: _XGBTrialInputs,
    stage: str,
    gpu_token: str,
    expected: int,
    varied_params: Sequence[str],
    use_autoregressive_eval: bool,
    progress: Dict,
) -> List[Dict]:
    """Run trials one at a time on one GPU, logging against the whole stage.

    *progress* carries the running count and best score across groups, so the
    log reads as one stage rather than restarting at 1/n for each group.
    """
    results: List[Dict] = []
    with cuda_device(gpu_token):
        _cap_search_cpu_threads()
        for trial_idx, params in trials:
            params_copy, score, best_iteration, one_step = train_and_evaluate_single_config(
                inputs.X_train,
                inputs.y_train,
                inputs.X_train_with_index,
                inputs.train_groups,
                inputs.targets,
                params,
                use_cv=inputs.use_cv,
                X_val=inputs.X_val,
                y_val=inputs.y_val,
                X_val_with_index=inputs.X_val_with_index,
                n_folds=inputs.trainer_cfg.n_folds,
                early_stopping_rounds=inputs.trainer_cfg.early_stopping_rounds,
                trainer_cfg=inputs.trainer_cfg,
                show_autoreg_progress=inputs.trainer_cfg.search_show_autoreg_progress,
                n_jobs=1,
                use_autoregressive_eval=use_autoregressive_eval,
                obs_mask=inputs.obs_train,
                obs_val_mask=None if inputs.use_cv else inputs.obs_val,
                x_scaler=inputs.x_scaler,
                y_scaler=inputs.y_scaler,
            )
            result = params_copy.copy()
            result[inputs.score_key] = float(score)
            result['best_iteration'] = int(best_iteration)
            result['stage'] = stage
            result['status'] = 'completed'
            result['gpu'] = str(gpu_token)
            result['trial'] = int(trial_idx)
            if one_step is not None:
                result['val_score_one_step'] = float(one_step)
            results.append(result)

            rmse = -float(score)
            progress['done'] += 1
            progress['best'] = min(progress['best'], rmse)
            _log_trial_result(
                stage, result, rmse, progress['done'],
                expected, progress['best'], varied_params,
            )
    return results


def _run_xgb_trials(
    params_list: List[Dict],
    inputs_by_n_lags: Dict[int, _XGBTrialInputs],
    stage: str,
    num_boost_round: int,
    gpu_pool: Sequence[str],
    use_autoregressive_eval: bool = False,
) -> List[Dict]:
    """Run one stage, with every lag count under way at once.

    Lag count decides which feature frame a trial trains on, so trials that
    disagree about it cannot share a worker: a worker is handed one set of
    frames.  The groups used to run back to back, which is free only while
    each is at least as wide as the GPU pool.  Stage 2 is not -- ten trials
    splitting 4/6 across eight GPUs ran as two waves, never using more than
    six cards, and cost the sum of two maxima instead of one.  So the pool is
    divided between the groups instead (:func:`allocate_pool`) and they run
    together, each worker still holding exactly one set of frames.

    *num_boost_round* is the stage's budget, not a searched parameter: early
    stopping on the validation set decides how many of those rounds are
    actually used, and the winner's count is what the final model is trained
    for.  This is the same arrangement as ``best_epoch`` for LSTM and TFT.

    *use_autoregressive_eval* selects what the trial is scored on: one-step
    predictions from ground-truth lags (cheap), or the same 15-step rollout
    the test phase runs, where each step's prediction becomes the next step's
    lag feature (expensive, and the objective actually being reported).
    """
    if not params_list:
        return []

    params_list = _with_budget(params_list, num_boost_round)
    expected = len(params_list)
    varied_params = sorted(set(params_list[0]) - {'num_boost_round'})

    def lags_of(params: Dict) -> int:
        return int(params.get("n_lags", N_LAG_FEATURES))

    # Trial ids number the whole stage rather than restarting per group: the
    # groups now run at the same time, and two trials called 0 in one log
    # cannot be told apart.
    groups: List[Tuple[int, _XGBTrialInputs, List[Tuple[int, Dict]]]] = []
    for n_lags in sorted({lags_of(p) for p in params_list}):
        inputs = inputs_by_n_lags.get(n_lags)
        if inputs is None:
            raise KeyError(
                f"No prepared data for n_lags={n_lags}; have {sorted(inputs_by_n_lags)}."
            )
        trials = [(i, p) for i, p in enumerate(params_list) if lags_of(p) == n_lags]
        groups.append((n_lags, inputs, trials))

    score_key = groups[0][1].score_key
    # One running count and one running best across the whole stage, however
    # the groups end up being scheduled.
    progress = {'done': 0, 'best': float('inf')}
    results: List[Dict] = []

    allocation = (
        allocate_pool([len(trials) for _, _, trials in groups], len(gpu_pool))
        if len(gpu_pool) > 1 else []
    )

    if allocation:
        ctx = _search_context()
        result_queue = ctx.Queue()
        processes: List[mp.Process] = []
        offset = 0
        for (n_lags, inputs, trials), count in zip(groups, allocation):
            gpus = list(gpu_pool[offset:offset + count])
            offset += count
            logging.info(
                "[%s] %d trial(s) at n_lags=%d on %d GPU(s): %s",
                stage, len(trials), n_lags, len(gpus), ", ".join(gpus),
            )
            processes.extend(_spawn_search_workers(
                ctx, trials, inputs, stage, gpus, result_queue, use_autoregressive_eval,
            ))
        results = _collect_worker_results(
            processes, result_queue, stage_name=stage, expected=expected,
            score_key=score_key, varied_params=varied_params, progress=progress,
        )
    else:
        # One GPU, or fewer GPUs than groups: nothing to divide, so the groups
        # run back to back with the whole pool each, as they always did.
        for n_lags, inputs, trials in groups:
            logging.info("[%s] %d trial(s) at n_lags=%d", stage, len(trials), n_lags)
            if len(gpu_pool) <= 1:
                results.extend(_run_xgb_trials_sequentially(
                    trials, inputs, stage, _first_visible_gpu_token(gpu_pool),
                    expected, varied_params, use_autoregressive_eval, progress,
                ))
                continue
            ctx = _search_context()
            result_queue = ctx.Queue()
            processes = _spawn_search_workers(
                ctx, trials, inputs, stage, gpu_pool, result_queue, use_autoregressive_eval,
            )
            results.extend(_collect_worker_results(
                processes, result_queue, stage_name=stage, expected=expected,
                score_key=score_key, varied_params=varied_params, progress=progress,
            ))

    if len(results) != expected:
        raise RuntimeError(
            f"XGB search {stage} produced {len(results)}/{expected} results"
        )
    return results


def hyperparameter_search(
    splits_by_n_lags: Dict[int, Dict],
    run_id: str,
    use_cv: bool = True,
) -> Tuple[Dict, Dict]:
    """Two-stage random search, the same protocol the LSTM and TFT searches run.

    Stage 1 ranks every sampled configuration under a shortened boosting
    budget; stage 2 refits the leaders at the full budget.  Both are scored
    on the autoregressive rollout the test phase reports (see
    XGBTrainerConfig).  This replaces three sequential stages that each swept a small
    grid exhaustively while holding the other parameters fixed -- coordinate
    descent, which cannot see interactions between parameters and made "a
    trial" mean something different here than for the other two models.

    *splits_by_n_lags* maps each searched lag count to the splits prepared at
    that lag count -- XGBoost's context length, the counterpart of the LSTM's
    sequence_length and the TFT's encoder length.  Every variant drops the
    same leading rows, so the lag counts are scored on the same elements.

    Returns
    -------
    Tuple[Dict, Dict]
        The winning hyperparameters -- including ``n_lags``, which decides
        the feature set the later phases rebuild -- and the per-stage trial
        rows.  The round count is not among them: it is a property of the
        trial, recorded per row as ``best_iteration``, and the final fit
        early-stops for itself.
    """
    space = XGBSearchSpace()
    trainer_cfg = XGBTrainerConfig()
    gpu_pool = _visible_gpu_pool()
    logging.info("XGB search space: %s", space.summary())
    logging.info(
        "XGB search GPU pool: %s (CUDA_VISIBLE_DEVICES=%s)",
        gpu_pool, os.environ.get('CUDA_VISIBLE_DEVICES'),
    )

    searched_lags = set(getattr(space.distributions.get("n_lags"), "values", (N_LAG_FEATURES,)))
    missing = searched_lags - set(splits_by_n_lags)
    if missing:
        raise ValueError(
            f"The search covers n_lags={sorted(searched_lags)} but no splits were "
            f"prepared for {sorted(missing)}."
        )

    inputs_by_n_lags: Dict[int, _XGBTrialInputs] = {}
    for n_lags, splits in splits_by_n_lags.items():
        if not use_cv:
            missing_val = [k for k in ("X_val", "y_val", "X_val_with_index") if splits.get(k) is None]
            if missing_val:
                raise ValueError(f"n_lags={n_lags} splits lack {missing_val} and use_cv=False")
        inputs_by_n_lags[int(n_lags)] = _XGBTrialInputs(
            X_train=splits["X_train"],
            y_train=splits["y_train"],
            X_train_with_index=splits["X_train_with_index"],
            train_groups=splits["train_groups"],
            targets=splits["targets"],
            use_cv=use_cv,
            X_val=splits.get("X_val"),
            y_val=splits.get("y_val"),
            X_val_with_index=splits.get("X_val_with_index"),
            trainer_cfg=trainer_cfg,
            obs_train=splits.get("obs_train"),
            obs_val=splits.get("obs_val"),
            x_scaler=splits.get("x_scaler"),
            y_scaler=splits.get("y_scaler"),
        )
    score_key = next(iter(inputs_by_n_lags.values())).score_key

    all_params = space.sample()
    logging.info(
        "XGB stage1: %d trials at %d rounds, scored on the %s",
        len(all_params), int(space.stage1_budget["num_boost_round"]),
        "autoregressive rollout" if trainer_cfg.search_autoregressive_stage1 else "one-step predictions",
    )
    stage1_started = time.monotonic()
    stage1_rows = _run_xgb_trials(
        all_params, inputs_by_n_lags, "stage1",
        int(space.stage1_budget["num_boost_round"]), gpu_pool,
        use_autoregressive_eval=trainer_cfg.search_autoregressive_stage1,
    )
    logging.info(
        "XGB stage1 complete in %s (%d trials)",
        format_duration(time.monotonic() - stage1_started), len(stage1_rows),
    )

    stage1_done = completed_trials(stage1_rows, space.param_keys, metric=score_key)
    if not stage1_done:
        raise RuntimeError("XGB stage1 produced no completed trials.")

    signature_to_params = {params_signature(p, space.param_keys): p for p in all_params}
    stage2_params = [
        signature_to_params[sig]
        for sig in select_top_k_signatures(
            stage1_done, space.stage2_top_k, space.param_keys, metric=score_key, mode="max",
            stratify_by=space.stage2_stratify_by,
        )
        if sig in signature_to_params
    ]
    logging.info(
        "XGB stage2: refitting the top %d of %d completed stage-1 trials at full budget "
        "(%d rounds), scored on the %s",
        len(stage2_params), len(stage1_done), trainer_cfg.num_boost_round,
        "autoregressive rollout" if trainer_cfg.search_autoregressive_stage2 else "one-step predictions",
    )
    stage2_started = time.monotonic()
    stage2_rows = _run_xgb_trials(
        stage2_params, inputs_by_n_lags, "stage2", trainer_cfg.num_boost_round, gpu_pool,
        use_autoregressive_eval=trainer_cfg.search_autoregressive_stage2,
    )
    logging.info(
        "XGB stage2 complete in %s (%d trials)",
        format_duration(time.monotonic() - stage2_started), len(stage2_rows),
    )

    all_rows = stage1_rows + stage2_rows
    completed = completed_trials(all_rows, space.param_keys, metric=score_key)
    # Stage 2 measured its candidates under the full boosting budget and, by
    # default, on the autoregressive rollout the test phase reports -- so its
    # scores are not comparable with stage 1's, and it decides whenever it
    # ran.  Stage 1 is a cheap ranking proxy and only a fallback.
    stage2_completed = [row for row in completed if row.get("stage") == "stage2"]
    best = best_trial(stage2_completed or completed, metric=score_key, mode="max")

    best_params = {key: best[key] for key in space.param_keys}

    try:
        write_search_report(
            os.path.join(get_run_root(run_id), "search"), space, completed,
            metric=score_key, mode="max",
        )
    except Exception as exc:  # noqa: BLE001 - a report must not sink a search
        logging.warning("Failed to write XGB search report: %s", exc)

    # The winner's round count belongs to the trial, not to best_params: the
    # final fit sees a different amount of data and early-stops for itself, so
    # carrying a number named after a booster argument would only invite it
    # back into the model.  Report it here, where it is a search result.
    logging.info(
        "XGB search complete (%s) -> best RMSE %.4f at round %d with %s",
        "stage2" if stage2_completed else "stage1",
        -float(best[score_key]),
        int(best.get('best_iteration', 0)) + 1,
        best_params,
    )
    return best_params, {"stage1": stage1_rows, "stage2": stage2_rows}


def train_and_save_model(
    X_train: pd.DataFrame,
    y_train: np.array,
    X_val: pd.DataFrame,
    y_val: np.array,
    targets: List[str],
    best_params: Dict,
    run_id: str,
    ) -> None:
    """Train the final model on the training split alone, holding out val.

    Uses X_train/y_train for the boosting updates and X_val/y_val for early
    stopping and best-round selection, matching the data regime the search
    scored its trials in -- and the one train_final_lstm and train_final_tft
    fit their finals in.  The model used to be fitted on train and val
    concatenated, which left XGBoost with strictly more data than the other
    two models and no genuinely held-out split of its own.
    """
    logging.info("Training final model with best parameters...")

    from configs.data import KEEP_PARTIAL_TARGETS

    with cuda_device(_first_visible_gpu_token()):
        try:
            y_train_df = pd.DataFrame(y_train, columns=targets)
            y_val_df = pd.DataFrame(y_val, columns=targets)
            trainer_cfg = XGBTrainerConfig()
            # Runs searched before the round count stopped being written into
            # best_params still have one saved on disk.  It is not a parameter
            # of this fit -- which sees a different amount of data and early-
            # stops for itself -- and it shares a name with a booster
            # argument, so drop it from a copy before it can reach one.
            best_params = dict(best_params)
            stale_num_boost_round = best_params.pop('num_boost_round', None)
            if stale_num_boost_round is not None:
                logging.info(
                    "Ignoring num_boost_round=%s from saved best_params; early "
                    "stopping picks the round count.", stale_num_boost_round,
                )
            xgb_params = get_xgb_params(best_params, trainer_cfg=trainer_cfg)
            num_boost_round = trainer_cfg.num_boost_round
            early_stopping_rounds = trainer_cfg.final_early_stopping_rounds
            logging.info(
                "XGB final training: train=%d val=%d rows, up to %d rounds "
                "with early stopping (patience=%d)",
                len(X_train), len(X_val), num_boost_round, early_stopping_rounds,
            )

            if KEEP_PARTIAL_TARGETS:
                model = PerTargetXGBRegressor(
                    targets=targets,
                    n_estimators=num_boost_round,
                    early_stopping_rounds=early_stopping_rounds,
                    **xgb_params,
                )
            else:
                model = XGBRegressor(
                    n_estimators=num_boost_round,
                    early_stopping_rounds=early_stopping_rounds,
                    **xgb_params,
                )

            model.fit(X_train, y_train_df, eval_set=[(X_val, y_val_df)], verbose=25)
            # save_model carries best_iteration into the saved JSON, and
            # predict() honours it, so the test phase scores the best round
            # rather than the last -- the counterpart of loading best.ckpt.
            logging.info(
                "XGB final training done -> best iteration %d of %d rounds",
                int(getattr(model, "best_iteration", 0) or 0), num_boost_round,
            )

            run_root = get_run_root(run_id)
            os.makedirs(os.path.join(run_root, "checkpoints"), exist_ok=True)
            model_path = os.path.join(run_root, "checkpoints", FINAL_MODEL_FILENAME)
            model.save_model(model_path)
            logging.info(f"Model saved to {model_path}")

        except Exception:
            # The phase must fail here; otherwise the caller saves the scalers,
            # reports the training complete, and the missing model surfaces
            # only when the test phase looks for it.
            logging.error("Final XGBoost training failed", exc_info=True)
            raise