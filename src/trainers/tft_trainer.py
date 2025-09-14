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
    """Make predictions with robust index alignment and horizon-only metrics.

    Restores legacy alignment: uses saved dataset template, calls predict with
    return_index semantics (through manual reconstruction), expands multi-step horizons
    if necessary, and computes metrics only on rows with fully valid predictions.
    """
    from src.trainers.evaluation import save_metrics

    test_data = session_state["test_data"]
    targets = session_state["targets"]

    with single_gpu_env():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            teardown_distributed()
            if torch.distributed.is_initialized():
                raise RuntimeError("Failed to teardown existing distributed process group before prediction.")

        model = load_tft_checkpoint(run_id)
        logging.info("Building test dataset for TFT prediction using saved template...")
        train_template = load_dataset_template(run_id)
        try:
            test_dataset = from_train_template(train_template, test_data, mode="predict")
        except Exception as e:
            raise RuntimeError(
                "Failed to build test dataset from saved template. Ensure column/dtype schema matches: "
                f"{e}"
            )

        time_idx_name = getattr(train_template, "time_idx", None)
        group_id_fields = list(getattr(train_template, "group_ids", []))
        if not time_idx_name or not group_id_fields:
            raise ValueError("Dataset template missing time_idx or group_ids")

        # DataLoader
        test_loader = test_dataset.to_dataloader(
            train=False,
            batch_size=64,
            num_workers=1,
            persistent_workers=False,
        )

        trainer = create_final_trainer(TFTTrainerConfig())
        # Use model.predict which (in PF) returns list of tensors when return_index isn't specified.
        # We replicate old behavior by capturing decoder-only horizon predictions.
        raw_preds = trainer.predict(model, test_loader)

        # Flatten prediction batches into single numpy array
        batch_arrays = []
        for arr in raw_preds:
            if torch.is_tensor(arr):
                batch_arrays.append(arr.detach().cpu().numpy())
            elif isinstance(arr, np.ndarray):
                batch_arrays.append(arr)
            else:
                raise RuntimeError(f"Unexpected prediction batch type: {type(arr)}")
        if not batch_arrays:
            raise RuntimeError("No prediction outputs returned.")
        preds = np.concatenate(batch_arrays, axis=0)

        # Multi-target tensor shape handling
        if preds.ndim == 3:  # (batch, horizon, targets)
            b, horizon, out = preds.shape
            logging.info("Expanding multi-step horizon predictions: batch=%d horizon=%d targets=%d", b, horizon, out)
            preds_flat = preds.reshape(b * horizon, out)
        elif preds.ndim == 2:
            preds_flat = preds
            out = preds_flat.shape[1]
            horizon = 1
        elif preds.ndim == 1:
            preds_flat = preds.reshape(-1, 1)
            out = 1
            horizon = 1
        else:
            raise RuntimeError(f"Unsupported prediction tensor shape: {preds.shape}")

        # Build index for alignment: replicate approach using group ids + time_idx from test_data
        key_cols = group_id_fields + [time_idx_name]
        if not all(k in test_data.columns for k in key_cols):
            missing = [k for k in key_cols if k not in test_data.columns]
            raise RuntimeError(f"Test data missing required key columns for alignment: {missing}")

        # Extract horizon portion (predict=True should already restrict to decoder steps)
        horizon_df = test_data[key_cols + targets].drop_duplicates(key_cols).copy()
        if preds_flat.shape[0] != len(horizon_df):
            logging.warning(
                "Prediction rows (%d) != horizon rows (%d). Proceeding best-effort alignment.",
                preds_flat.shape[0], len(horizon_df)
            )
            min_len = min(preds_flat.shape[0], len(horizon_df))
            horizon_df = horizon_df.iloc[:min_len].reset_index(drop=True)
            preds_flat = preds_flat[:min_len]
        else:
            horizon_df = horizon_df.reset_index(drop=True)

        y_true = horizon_df[targets].values
        y_pred = preds_flat[:, : len(targets)] if preds_flat.shape[1] >= len(targets) else preds_flat

        # Valid mask: both sides finite & non-NaN
        valid_mask = (~np.isnan(y_true).any(axis=1)) & (~np.isnan(y_pred).any(axis=1))
        if valid_mask.any():
            save_metrics(run_id, y_true[valid_mask], y_pred[valid_mask])
        else:
            logging.warning("No valid rows for metric computation (all rows have NaNs).")

        # Store horizon info for plotting consistency
        session_state['horizon_df'] = horizon_df
        session_state['horizon_y_true'] = y_true

        return y_pred


# Maintain backward compatibility
build_datasets = build_datasets

__all__ = [
    "build_datasets",
    "hyperparameter_search_tft", 
    "train_final_tft",
    "predict_tft",
]