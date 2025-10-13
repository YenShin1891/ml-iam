"""TFT trainer with main orchestration functions."""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ParameterSampler

from configs.paths import RESULTS_PATH
from configs.models import TFTSearchSpace, TFTTrainerConfig
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
from .tft_utils import single_gpu_env, teardown_distributed


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

    # Save dataset template
    save_dataset_template(combined_dataset, run_id)

    combined_loader = combined_dataset.to_dataloader(
        train=True,
        batch_size=trainer_cfg.batch_size,
        num_workers=4,
        persistent_workers=True,
    )

    final_trainer = create_final_trainer(trainer_cfg)
    final_trainer.fit(model=tft_final, train_dataloaders=combined_loader)
    final_trainer.save_checkpoint(final_ckpt_path)


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
            num_workers=4,
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

        # Expose horizon dataframe and y_true for downstream plotting
        session_state['horizon_df'] = horizon_df
        session_state['horizon_y_true'] = y_true

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