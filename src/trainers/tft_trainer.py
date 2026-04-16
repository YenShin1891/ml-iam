"""TFT trainer with main orchestration functions."""

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.model_selection import ParameterSampler

from configs.paths import RESULTS_PATH
from configs.models import TFTSearchSpace, TFTTrainerConfig
from src.utils.utils import get_run_root
from .tft_dataset import (
    build_datasets,
    create_combined_dataset,
    from_train_template,
    load_dataset_template,
    save_dataset_template,
)
from .tft_model import (
    create_dataloaders,
    create_final_trainer,
    create_search_trainer,
    create_tft_model,
    load_tft_checkpoint,
)
from .tft_utils import get_default_num_workers, single_gpu_env, teardown_distributed


def _get_search_gpu_ids() -> List[int]:
    """Get physical GPU IDs available for search from CUDA_VISIBLE_DEVICES."""
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_env:
        return [int(x.strip()) for x in cuda_env.split(",") if x.strip()]
    n = torch.cuda.device_count()
    return list(range(n)) if n > 0 else []


def _get_best_epoch(trainer) -> int:
    """Extract the epoch of the best val_loss from an EarlyStopping callback."""
    for cb in trainer.callbacks:
        if isinstance(cb, EarlyStopping):
            # best_score is set at the epoch that had the lowest val_loss;
            # current_epoch minus wait_count gives us that epoch.
            return trainer.current_epoch - cb.wait_count
    # Fallback: if no early stopping, the last epoch is the best we know.
    return trainer.current_epoch


def _search_worker(gpu_id, params_list, train_dataset, val_dataset, n_targets, trainer_cfg, result_queue):
    """Run a batch of search trials on a single GPU (spawned subprocess)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Cap dataloader workers to avoid oversubscription across many GPU workers
    num_workers = min(get_default_num_workers(), 4)
    train_loader = train_dataset.to_dataloader(
        train=True, batch_size=trainer_cfg.batch_size,
        num_workers=num_workers, persistent_workers=num_workers > 0,
    )
    val_loader = val_dataset.to_dataloader(
        train=False, batch_size=trainer_cfg.batch_size,
        num_workers=num_workers, persistent_workers=num_workers > 0,
    )

    for i, params in enumerate(params_list):
        logging.info("GPU %d - Trial %d/%d - Params: %s", gpu_id, i + 1, len(params_list), params)
        tft = create_tft_model(train_dataset, params, n_targets)
        trainer = create_search_trainer(trainer_cfg)
        trainer.fit(model=tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
        val_loss = trainer.callback_metrics["val_loss"].item()
        best_epoch = _get_best_epoch(trainer)
        result_queue.put({**params, "val_loss": val_loss, "best_epoch": best_epoch})
        logging.info("GPU %d - Trial %d/%d - val_loss: %.4f best_epoch: %d", gpu_id, i + 1, len(params_list), val_loss, best_epoch)


def hyperparameter_search_tft(
    train_dataset,
    val_dataset,
    targets: List[str],
    run_id: str,
) -> Dict:
    """Perform hyperparameter search for TFT model.

    When multiple GPUs are available, trials are distributed across GPUs
    in parallel (one trial per GPU at a time, each GPU runs its share
    sequentially).  Falls back to sequential search on a single GPU.
    """
    search_cfg = TFTSearchSpace()
    trainer_cfg = TFTTrainerConfig()
    n_targets = len(targets)

    all_params = list(ParameterSampler(
        search_cfg.param_dist, n_iter=search_cfg.search_iter_n, random_state=0,
    ))

    gpu_ids = _get_search_gpu_ids()

    if len(gpu_ids) <= 1:
        # Single GPU or CPU: run sequentially (original behaviour)
        return _search_sequential(train_dataset, val_dataset, n_targets, all_params, trainer_cfg)

    # --- parallel search across GPUs ---
    import torch.multiprocessing as mp

    logging.info("Parallel search across %d GPUs: %s", len(gpu_ids), gpu_ids)

    # Round-robin trials to GPUs
    gpu_params: List[List[Dict]] = [[] for _ in gpu_ids]
    for i, params in enumerate(all_params):
        gpu_params[i % len(gpu_ids)].append(params)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    processes = []
    for gpu_id, params_list in zip(gpu_ids, gpu_params):
        if not params_list:
            continue
        p = ctx.Process(
            target=_search_worker,
            args=(gpu_id, params_list, train_dataset, val_dataset,
                  n_targets, trainer_cfg, result_queue),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Check for worker failures
    failed = [p for p in processes if p.exitcode != 0]
    if failed:
        raise RuntimeError(
            f"{len(failed)} search worker(s) crashed — check logs for GPU-level errors"
        )

    results = []
    while not result_queue.empty():
        results.append(result_queue.get_nowait())

    if not results:
        raise RuntimeError("TFT hyperparameter search produced no results.")

    best = min(results, key=lambda r: r["val_loss"])
    best_score = best.pop("val_loss")
    best_epoch = best.pop("best_epoch")
    best_params = best
    best_params["best_epoch"] = best_epoch

    logging.info("Best TFT Params: %s with Val Loss: %.4f (best epoch: %d)", best_params, best_score, best_epoch)
    return best_params


def _search_sequential(train_dataset, val_dataset, n_targets, all_params, trainer_cfg) -> Dict:
    """Sequential search fallback for single-GPU / CPU environments."""
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, trainer_cfg.batch_size)

    best_score = float("inf")
    best_params = None
    best_epoch = 0

    for i, params in enumerate(all_params):
        logging.info("TFT Search Iteration %d/%d - Params: %s", i + 1, len(all_params), params)
        tft = create_tft_model(train_dataset, params, n_targets)
        trainer = create_search_trainer(trainer_cfg)
        trainer.fit(model=tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
        val_loss = trainer.callback_metrics["val_loss"].item()

        if val_loss < best_score:
            best_score = val_loss
            best_params = params
            best_epoch = _get_best_epoch(trainer)

    logging.info("Best TFT Params: %s with Val Loss: %.4f (best epoch: %d)", best_params, best_score, best_epoch)
    if best_params is None:
        raise RuntimeError("TFT hyperparameter search did not identify a best parameter set.")
    best_params["best_epoch"] = best_epoch
    return best_params


def _is_primary_rank() -> bool:
    """Check if this is the primary DDP rank (or non-DDP)."""
    rank_vars = [
        os.getenv("LOCAL_RANK"),
        os.getenv("PL_TRAINER_GLOBAL_RANK"),
        os.getenv("GLOBAL_RANK"),
        os.getenv("RANK"),
    ]
    return all(rv in (None, "0") for rv in rank_vars)


def train_final_tft(
    train_dataset,
    val_dataset,
    targets: List[str],
    run_id: str,
    best_params: Dict,
    session_state: Optional[Dict] = None,
) -> None:
    """Train final TFT on combined train+val data.

    Uses the epoch count from hyperparameter search (best_epoch) to
    determine how long to train.  This gives the model all available
    data while avoiding both underfitting and overfitting.

    All DDP ranks must execute the same code path so they all reach
    trainer.fit() together.  Only rank 0 writes logs, summaries, and
    the dataset template.
    """
    trainer_cfg = TFTTrainerConfig()
    final_dir = os.path.join(get_run_root(run_id), "final")
    final_ckpt_path = os.path.join(final_dir, "best.ckpt")

    primary = _is_primary_rank()

    os.makedirs(final_dir, exist_ok=True)

    # Get original DataFrames from session_state
    train_df = None
    val_df = None
    if session_state is not None:
        train_df = session_state.get("train_data")
        val_df = session_state.get("val_data")

    if not isinstance(train_df, pd.DataFrame) or not isinstance(val_df, pd.DataFrame):
        raise RuntimeError(
            "train_final_tft requires session_state to include both train_data and val_data as DataFrames"
        )

    # Create combined dataset
    combined_dataset = create_combined_dataset(train_dataset, train_df, val_df)

    # Determine epoch count from search.  The best_epoch from search tells
    # us when the model peaked on a train/val split.  With more data
    # (combined set) the model can benefit from a few extra epochs.
    search_best_epoch = best_params.pop("best_epoch", None)
    if search_best_epoch is not None and search_best_epoch > 0:
        final_epochs = int(search_best_epoch * 1.2) + 1  # 20% headroom for larger dataset
    else:
        final_epochs = trainer_cfg.final_max_epochs

    if primary:
        train_len = len(train_dataset)
        val_len = len(val_dataset)
        combined_len = len(combined_dataset)
        logging.info(
            "TFT final training dataset sizes -> train=%d val=%d combined=%d",
            train_len,
            val_len,
            combined_len,
        )
        logging.info(
            "TFT final training for %d epochs (search best_epoch=%s)",
            final_epochs,
            search_best_epoch,
        )

        # Save dataset template (used later for prediction)
        save_dataset_template(combined_dataset, run_id)

    # Create model with best params; disable LR scheduler since there is
    # no validation loader and ReduceLROnPlateau cannot monitor val_loss.
    n_targets = len(targets)
    tft_final = create_tft_model(
        train_dataset, best_params, n_targets, disable_lr_scheduler=True,
    )

    combined_loader = combined_dataset.to_dataloader(
        train=True,
        batch_size=trainer_cfg.batch_size,
        num_workers=get_default_num_workers(),
        persistent_workers=True,
    )

    final_trainer = create_final_trainer(trainer_cfg, max_epochs_override=final_epochs)
    final_trainer.fit(model=tft_final, train_dataloaders=combined_loader)

    # Tear down DDP process group immediately so non-primary ranks can exit
    # without NCCL watchdog timeouts on rank 0.
    teardown_distributed()

    if primary:
        final_trainer.save_checkpoint(final_ckpt_path)
        callback_metrics = final_trainer.callback_metrics if hasattr(final_trainer, "callback_metrics") else {}

        def _metric_to_float(value):
            if isinstance(value, torch.Tensor):
                return float(value.detach().cpu().item())
            if isinstance(value, (int, float)):
                return float(value)
            try:
                return float(value)
            except Exception:
                return None

        train_loss = _metric_to_float(callback_metrics.get("train_loss"))
        logging.info(
            "TFT final training loss -> train_loss=%s",
            f"{train_loss:.6f}" if train_loss is not None else "NA",
        )

        summary = {
            "best_params": best_params,
            "train_set_rows": train_len,
            "val_set_rows": val_len,
            "combined_rows": combined_len,
            "final_epochs": final_epochs,
            "search_best_epoch": search_best_epoch,
            "train_loss": train_loss,
        }
        summary_path = os.path.join(final_dir, "training_summary.json")
        try:
            with open(summary_path, "w", encoding="utf-8") as fp:
                json.dump(summary, fp, indent=2)
            logging.info("Saved TFT training summary to %s", summary_path)
        except Exception as exc:
            logging.warning("Failed to write TFT training summary: %s", exc)

    if session_state is not None and "tft_time_idx_column" not in session_state:
        session_state["tft_time_idx_column"] = getattr(train_dataset, "time_idx", "Step")


def predict_tft(session_state: Dict, run_id: str) -> np.ndarray:
    """Make predictions following the exact original tft_trajectory_plotting logic."""
    from src.trainers.evaluation import save_metrics

    test_data = session_state["test_data"]
    targets = session_state["targets"]

    # Follow original pattern exactly
    with single_gpu_env():
        # Best effort teardown if a process group is somehow still alive
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            teardown_distributed()
            if torch.distributed.is_initialized():
                raise RuntimeError("Failed to teardown existing distributed process group before prediction.")

        model = load_tft_checkpoint(run_id)

        logging.info("Building test dataset for TFT prediction using saved template...")
        train_template = load_dataset_template(run_id)

        try:
            test_dataset = from_train_template(
                train_template,
                test_data,
                mode="predict"  # This creates predict=True dataset
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to build test dataset from saved template. Ensure test_data columns and dtypes match the training schema: "
                f"{e}"
            )

        template_time_idx = getattr(train_template, "time_idx", None)
        template_group_ids = getattr(train_template, "group_ids", None)
        if not template_time_idx or not template_group_ids:
            raise ValueError("Saved dataset template is missing time_idx or group_ids; cannot align predictions.")
        logging.info("Loaded dataset template for prediction.")

        from configs.models import TFTTrainerConfig
        trainer_cfg = TFTTrainerConfig()
        test_loader = test_dataset.to_dataloader(
            train=False,
            batch_size=trainer_cfg.batch_size,
            num_workers=get_default_num_workers(),
            persistent_workers=True,
        )

        logging.info("Predicting with TFT model (forecast horizon only)...")
        returns = model.predict(test_loader, return_index=True)

        from pytorch_forecasting.models.base._base_model import Prediction as _PFPrediction
        if not isinstance(returns, _PFPrediction):
            raise RuntimeError(f"Unexpected predict() return type: {type(returns)}; expected pytorch_forecasting Prediction.")

        outputs = returns.output
        if isinstance(outputs, list):
            if len(outputs) == 0:
                raise RuntimeError("Prediction.output list is empty.")
            if not all(torch.is_tensor(o) for o in outputs):
                raise RuntimeError("All elements in Prediction.output list must be tensors.")
            preds_tensor = outputs[0] if len(outputs) == 1 else torch.stack(outputs, dim=-1)
        elif torch.is_tensor(outputs):
            preds_tensor = outputs
        else:
            raise RuntimeError(f"Unsupported Prediction.output type: {type(outputs)}")

        index_attr = getattr(returns, 'index', None)
        if isinstance(index_attr, list):
            dfs = [d for d in index_attr if isinstance(d, pd.DataFrame) and not d.empty]
            if not dfs:
                raise RuntimeError("Prediction.index list is empty or has no valid DataFrames.")
            index_df = pd.concat(dfs, ignore_index=True)
        elif isinstance(index_attr, pd.DataFrame):
            if index_attr.empty:
                raise RuntimeError("Prediction.index DataFrame is empty.")
            index_df = index_attr.copy()
        else:
            raise RuntimeError(f"Unsupported Prediction.index type: {type(index_attr)}")

        preds_flat = None  # will set below after optional expansion
        time_idx_name = template_time_idx
        group_ids = list(template_group_ids)

        # Normalize index dataframe
        if "time_idx" in index_df.columns and time_idx_name not in index_df.columns:
            index_df = index_df.rename(columns={"time_idx": time_idx_name})

        # If predictions are 3D (n_samples, pred_len, out_size) but index_df only has n_samples rows,
        # expand the index so each horizon step has its own row with incremented time index.
        if torch.is_tensor(preds_tensor) and preds_tensor.ndim == 3:
            n_samples, pred_len, out_size = preds_tensor.shape
            if len(index_df) == n_samples and pred_len > 1:
                logging.info(
                    "Expanding index_df for multi-step horizon: samples=%d, pred_len=%d", n_samples, pred_len
                )
                # Build expanded index
                expanded_rows = []
                base_cols = index_df.columns.tolist()
                if time_idx_name not in base_cols:
                    raise KeyError(
                        f"Time index column '{time_idx_name}' not found in prediction index DataFrame columns: {base_cols}"
                    )
                for i in range(n_samples):
                    base_row = index_df.iloc[i]
                    base_time = base_row[time_idx_name]
                    for h in range(pred_len):
                        new_row = base_row.copy()
                        # Assumption: decoder steps are consecutive increments
                        new_row[time_idx_name] = base_time + h
                        expanded_rows.append(new_row)
                index_df = pd.DataFrame(expanded_rows).reset_index(drop=True)
                # Flatten predictions accordingly
                preds_flat = preds_tensor.detach().cpu().numpy().reshape(n_samples * pred_len, out_size)
            else:
                preds_flat = preds_tensor.detach().cpu().numpy()
                if preds_flat.ndim == 3:
                    n_samples, pred_len, out_size = preds_flat.shape
                    preds_flat = preds_flat.reshape(n_samples * pred_len, out_size)
        else:
            preds_flat = preds_tensor.detach().cpu().numpy()
            if preds_flat.ndim == 3:
                n_samples, pred_len, out_size = preds_flat.shape
                preds_flat = preds_flat.reshape(n_samples * pred_len, out_size)

        # Evaluate only the forecast horizon rows returned by predict=True.
        key_cols = group_ids + [time_idx_name]
        # Collect reference columns (ensure presence in test_data)
        ref_cols = [c for c in key_cols + ['Year'] + targets if c in test_data.columns]
        horizon_df = index_df[key_cols].merge(
            test_data[ref_cols].drop_duplicates(key_cols),
            on=key_cols,
            how='left'
        )
        horizon_len = len(horizon_df)
        horizon_groups = horizon_df[group_ids].drop_duplicates().shape[0]
        test_group_total = test_data[group_ids].drop_duplicates().shape[0]
        logging.info(
            "TFT horizon coverage -> rows=%d unique_groups=%d (test_groups=%d)",
            horizon_len,
            horizon_groups,
            test_group_total,
        )
        target_offset = int(session_state.get("tft_target_offset", 0) or 0)
        if target_offset > 0:
            if time_idx_name in horizon_df.columns:
                warm_mask = horizon_df[time_idx_name] >= target_offset
                dropped = int((~warm_mask).sum())
                if dropped > 0:
                    logging.info(
                        "Warm start offset %d: dropping %d horizon rows where %s < offset",
                        target_offset,
                        dropped,
                        time_idx_name,
                    )
                    horizon_df = horizon_df.loc[warm_mask].reset_index(drop=True)
                    preds_flat = preds_flat[warm_mask.to_numpy()]
                    horizon_len = len(horizon_df)
                if horizon_len == 0:
                    logging.warning(
                        "All TFT prediction rows filtered by warm-start offset %d.",
                        target_offset,
                    )
            else:
                logging.warning(
                    "Warm start offset configured (%d) but index column '%s' missing in horizon_df.",
                    target_offset,
                    time_idx_name,
                )
        if preds_flat.shape[0] != len(horizon_df):
            logging.error(
                "After expansion attempt: preds_flat rows=%d, horizon_df rows=%d. First few time_idx in index_df: %s",
                preds_flat.shape[0], len(horizon_df), index_df[time_idx_name].head().tolist()
            )
            raise RuntimeError(
                f"Prediction rows ({preds_flat.shape[0]}) != horizon_df rows ({len(horizon_df)})."
            )

        y_true = horizon_df[targets].values

        # Handle RMSE predictions (standard case)
        y_pred = preds_flat
        valid_mask = (~np.isnan(y_true).any(axis=1)) & (~np.isnan(y_pred).any(axis=1))
        if valid_mask.any():
            save_metrics(run_id, y_true[valid_mask], y_pred[valid_mask])
        else:
            logging.warning("No valid rows to compute metrics (all targets or predictions contain NaNs).")

        removed_groups = None
        try:
            test_groups = set(map(tuple, test_data[group_ids].drop_duplicates().itertuples(index=False, name=None)))
            horizon_groups = set(map(tuple, horizon_df[group_ids].drop_duplicates().itertuples(index=False, name=None)))
            removed_groups = sorted(test_groups - horizon_groups)
            if removed_groups:
                preview = removed_groups[:5]
                logging.warning(
                    "TFT prediction dropped %d groups due to window constraints. Sample: %s",
                    len(removed_groups),
                    preview,
                )
        except Exception as exc:
            logging.warning("Failed to compute removed groups for TFT prediction: %s", exc)

        removed_count = len(removed_groups) if removed_groups else 0
        summary = {
            "horizon_rows": horizon_len,
            "horizon_unique_groups": horizon_df[group_ids].drop_duplicates().shape[0],
            "test_unique_groups": test_group_total,
            "removed_groups_count": removed_count,
            "removed_groups_sample": removed_groups[:5] if removed_groups else [],
        }
        prediction_summary_path = os.path.join(get_run_root(run_id), "final", "prediction_summary.json")
        try:
            os.makedirs(os.path.dirname(prediction_summary_path), exist_ok=True)
            with open(prediction_summary_path, "w", encoding="utf-8") as fp:
                json.dump(summary, fp, indent=2)
            logging.info("Saved TFT prediction summary to %s", prediction_summary_path)
        except Exception as exc:
            logging.warning("Failed to write TFT prediction summary: %s", exc)

        # Expose horizon dataframe and y_true for downstream plotting
        session_state['horizon_df'] = horizon_df
        session_state['horizon_y_true'] = y_true
        session_state['removed_groups'] = removed_groups

        # Return predictions matrix
        return y_pred


# Maintain backward compatibility
build_datasets = build_datasets

__all__ = [
    "build_datasets",
    "hyperparameter_search_tft", 
    "train_final_tft",
    "predict_tft",
]