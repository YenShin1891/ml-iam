"""TFT trainer with main orchestration functions."""

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ParameterSampler

from configs.paths import RESULTS_PATH
from configs.models import TFTSearchSpace, TFTTrainerConfig
from configs.data import INDEX_COLUMNS, MAX_SERIES_LENGTH
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
    create_trial_checkpoint,
    load_tft_checkpoint,
)
from .tft_utils import get_default_num_workers, single_gpu_env, teardown_distributed


def hyperparameter_search_tft(
    train_dataset,
    val_dataset,
    targets: List[str],
    run_id: str,
) -> Dict:
    """Perform hyperparameter search for TFT model."""
    search_cfg = TFTSearchSpace()
    trainer_cfg = TFTTrainerConfig()

    search_results = []
    best_score = float("inf")
    best_params = None

    for i, params in enumerate(ParameterSampler(search_cfg.param_dist, n_iter=search_cfg.search_iter_n, random_state=0)):
        logging.info(f"TFT Search Iteration {i+1}/{search_cfg.search_iter_n} - Params: {params}")

        n_targets = len(targets)
        tft = create_tft_model(train_dataset, params, n_targets)

        checkpoint_callback = create_trial_checkpoint(run_id, i)
        trainer = create_search_trainer(trainer_cfg, checkpoint_callback)

        train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, trainer_cfg.batch_size)
        trainer.fit(model=tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

        val_loss = trainer.callback_metrics["val_loss"].item()
        search_results.append({**params, "val_loss": val_loss})

        if val_loss < best_score:
            best_score = val_loss
            best_params = params

    logging.info(f"Best TFT Params: {best_params} with Val Loss: {best_score:.4f}")
    if best_params is None:
        raise RuntimeError("TFT hyperparameter search did not identify a best parameter set.")
    return best_params


def train_final_tft(
    train_dataset,
    val_dataset,
    targets: List[str],
    run_id: str,
    best_params: Dict,
    session_state: Optional[Dict] = None,
) -> None:
    """Train final TFT on combined train+val and save checkpoint."""
    trainer_cfg = TFTTrainerConfig()

    final_dir = os.path.join(RESULTS_PATH, run_id, "final")
    os.makedirs(final_dir, exist_ok=True)
    final_ckpt_path = os.path.join(final_dir, "best.ckpt")

    # Create model with best params
    n_targets = len(targets)
    tft_final = create_tft_model(train_dataset, best_params, n_targets)

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

    train_len = len(train_dataset)
    val_len = len(val_dataset)
    combined_len = len(combined_dataset)
    train_df_rows = len(train_df)
    val_df_rows = len(val_df)
    logging.info(
        "TFT final training dataset sizes -> train=%d (rows=%d) val=%d (rows=%d) combined=%d",
        train_len,
        train_df_rows,
        val_len,
        val_df_rows,
        combined_len,
    )

    # Save dataset template
    save_dataset_template(combined_dataset, run_id)

    combined_loader = combined_dataset.to_dataloader(
        train=True,
        batch_size=trainer_cfg.batch_size,
        num_workers=get_default_num_workers(),
        persistent_workers=True,
    )

    final_trainer = create_final_trainer(trainer_cfg)
    final_trainer.fit(model=tft_final, train_dataloaders=combined_loader)
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
    val_loss = _metric_to_float(callback_metrics.get("val_loss"))
    logging.info(
        "TFT final training losses -> train_loss=%s val_loss=%s",
        f"{train_loss:.6f}" if train_loss is not None else "NA",
        f"{val_loss:.6f}" if val_loss is not None else "NA",
    )

    summary = {
        "best_params": best_params,
        "train_set_rows": train_len,
        "val_set_rows": val_len,
        "combined_rows": combined_len,
        "train_dataframe_rows": train_df_rows,
        "val_dataframe_rows": val_df_rows,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    # Persist summary alongside checkpoint for post-run inspection
    summary_path = os.path.join(final_dir, "training_summary.json")
    try:
        with open(summary_path, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        logging.info("Saved TFT training summary to %s", summary_path)
    except Exception as exc:
        logging.warning("Failed to write TFT training summary: %s", exc)


def prepare_prediction_test_data(session_state: Dict, dataset_template) -> pd.DataFrame:
    """Return test data trimmed to the maximum usable sequence length for prediction."""
    cache_key = "_trimmed_test_data"
    cache_meta_key = f"{cache_key}_meta"

    test_data = session_state.get("test_data")
    if test_data is None:
        raise ValueError("session_state missing 'test_data' required for prediction")

    template_time_idx = getattr(dataset_template, "time_idx", None)
    if template_time_idx is None:
        raise ValueError("Dataset template missing time_idx; cannot prepare test data for prediction")

    template_group_ids = list(getattr(dataset_template, "group_ids", INDEX_COLUMNS))

    # Validate required columns exist in the raw test data
    required_cols = [template_time_idx] + template_group_ids
    missing_cols = [col for col in required_cols if col not in test_data.columns]
    if missing_cols:
        raise ValueError(
            "Test data missing required columns for prediction truncation: "
            + ", ".join(missing_cols)
        )

    max_encoder_length = int(getattr(dataset_template, "max_encoder_length", 0) or 0)
    max_prediction_length = int(getattr(dataset_template, "max_prediction_length", 0) or 0)
    target_length = max_encoder_length + max_prediction_length

    if target_length <= 0:
        target_length = MAX_SERIES_LENGTH
    elif MAX_SERIES_LENGTH:
        target_length = min(target_length, MAX_SERIES_LENGTH)

    if target_length <= 0:
        raise ValueError("Unable to determine a positive target_length for prediction truncation")

    cache_meta = session_state.get(cache_meta_key, {})
    if (
        cache_key in session_state
        and cache_meta.get("time_idx") == template_time_idx
        and cache_meta.get("max_length") == target_length
    ):
        cached_df = session_state[cache_key]
        if isinstance(cached_df, pd.DataFrame):
            return cached_df

    truncated_groups: List[pd.DataFrame] = []
    truncated_count = 0
    dropped_rows = 0

    for _, group in test_data.groupby(template_group_ids, sort=False):
        group_sorted = group.sort_values(template_time_idx)
        if len(group_sorted) > target_length:
            truncated_groups.append(group_sorted.iloc[-target_length:].copy())
            truncated_count += 1
            dropped_rows += len(group_sorted) - target_length
        else:
            truncated_groups.append(group_sorted.copy())

    if truncated_groups:
        trimmed_df = pd.concat(truncated_groups, ignore_index=True)
    else:
        trimmed_df = test_data.copy()

    if truncated_count > 0:
        logging.info(
            "Trimmed %d test trajectories to last %d steps (removed %d rows in total)",
            truncated_count,
            target_length,
            dropped_rows,
        )
    else:
        logging.info("Test trajectories already within %d-step limit; no truncation applied", target_length)

    session_state[cache_key] = trimmed_df
    session_state[cache_meta_key] = {"time_idx": template_time_idx, "max_length": target_length}
    session_state.setdefault("test_data_trimmed", trimmed_df)

    return trimmed_df


def predict_tft(session_state: Dict, run_id: str) -> np.ndarray:
    """Make predictions following the exact original tft_trajectory_plotting logic."""
    from src.trainers.evaluation import save_metrics

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

        test_data = prepare_prediction_test_data(session_state, train_template)

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
        prediction_summary_path = os.path.join(RESULTS_PATH, run_id, "final", "prediction_summary.json")
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