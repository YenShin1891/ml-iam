"""TFT trainer with main orchestration functions."""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ParameterSampler

from configs.config import RESULTS_PATH
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
    """Make predictions using trained TFT model."""
    from src.trainers.evaluation import save_metrics

    test_data = session_state["test_data"]
    targets = session_state["targets"]

    with single_gpu_env():
        # Clean up any distributed state
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
                mode="predict"
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to build test dataset from saved template. "
                f"Ensure test_data columns and dtypes match the training schema: {e}"
            )

        template_time_idx = getattr(train_template, "time_idx", None)
        template_group_ids = getattr(train_template, "group_ids", None)
        if not template_time_idx or not template_group_ids:
            raise ValueError("Saved dataset template is missing time_idx or group_ids")

        # Create test dataloader
        test_loader = test_dataset.to_dataloader(
            train=False,
            batch_size=64,
            num_workers=1,
            persistent_workers=False,
        )

        # Make predictions
        trainer = create_final_trainer(TFTTrainerConfig())
        
        raw_predictions = trainer.predict(model, test_loader)
        predictions_list = [p.cpu().numpy() for p in raw_predictions]
        predictions_array = np.vstack(predictions_list)

        # Post-process predictions
        if predictions_array.ndim == 3 and predictions_array.shape[1] == 1:
            predictions_array = predictions_array.squeeze(axis=1)

        # Align predictions with test data
        aligned_preds = np.full((len(test_data), len(targets)), np.nan)
        
        for i, batch in enumerate(test_loader):
            batch_indices = batch["groups"]  # Assuming this contains row indices
            start_idx = i * test_loader.batch_size
            end_idx = min(start_idx + test_loader.batch_size, len(predictions_array))
            batch_preds = predictions_array[start_idx:end_idx]
            
            for j, idx in enumerate(batch_indices):
                if j < len(batch_preds):
                    aligned_preds[idx] = batch_preds[j]

        logging.info("TFT prediction completed. Shape: %s", aligned_preds.shape)
        
        # Save metrics
        y_test = session_state.get("y_test")
        if y_test is not None:
            save_metrics(run_id, test_data, y_test, aligned_preds, targets, "TFT")

        return aligned_preds


# Maintain backward compatibility
build_datasets = build_datasets

__all__ = [
    "build_datasets",
    "hyperparameter_search_tft", 
    "train_final_tft",
    "predict_tft",
]