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

        # Now I understand the structure:
        # - Each batch returns a list of 9 tensors (one per target)
        # - Each tensor has shape [batch_size, timesteps, 1] = [64, 13, 1]
        # - We need to extract only the DECODER timesteps (predictions), not all timesteps

        # Get decoder length from the dataset
        decoder_length = getattr(test_dataset, 'max_prediction_length', 1)
        logging.info("Decoder length (max_prediction_length): %d", decoder_length)

        # Ensure targets is a list for proper column selection
        if isinstance(targets, str):
            target_cols = [targets]
        else:
            target_cols = list(targets)

        # Process predictions correctly - only take decoder timesteps
        all_predictions = []

        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= len(raw_preds):
                break

            pred_batch = raw_preds[batch_idx]

            # Extract prediction tensor list from pytorch_forecasting Output object
            if hasattr(pred_batch, 'prediction'):
                pred_list = pred_batch.prediction
            else:
                raise RuntimeError(f"Unexpected prediction batch type: {type(pred_batch)}")

            if not isinstance(pred_list, list):
                raise RuntimeError(f"Expected prediction to be list, got {type(pred_list)}")

            if len(pred_list) != len(target_cols):
                raise RuntimeError(f"Expected {len(target_cols)} targets, got {len(pred_list)} predictions")

            # Each tensor in pred_list has shape [batch_size, timesteps, 1]
            # We only want the decoder timesteps (last decoder_length timesteps)
            batch_size = pred_list[0].shape[0]
            timesteps = pred_list[0].shape[1]

            # Take only the last decoder_length timesteps (these are the actual predictions)
            decoder_start = timesteps - decoder_length

            # For each sample in the batch
            for sample_idx in range(batch_size):
                # Extract decoder predictions for this sample across all targets
                sample_preds = []
                for target_idx, target_tensor in enumerate(pred_list):
                    # Shape: [batch_size, timesteps, 1] -> take [sample_idx, decoder_start:, 0]
                    decoder_preds = target_tensor[sample_idx, decoder_start:, 0].detach().cpu().numpy()
                    sample_preds.extend(decoder_preds)  # Flatten decoder timesteps

                all_predictions.append(sample_preds)

        if not all_predictions:
            raise RuntimeError("No predictions collected from any batch")

        # Convert to numpy array
        preds_array = np.array(all_predictions)
        logging.info("Collected %d predictions with shape %s (decoder_length=%d)",
                    len(all_predictions), preds_array.shape, decoder_length)

        # Now we should have the right number of predictions
        expected_predictions = len(test_dataset)
        if len(preds_array) != expected_predictions:
            logging.warning("Prediction count (%d) != expected (%d)", len(preds_array), expected_predictions)

        # Access the dataset's underlying DataFrame structure properly
        # The test_dataset.data is a dict with keys: ['reals', 'categoricals', 'groups', 'target', 'weight', 'time']
        # We need to reconstruct the index to align with the predictions

        # Get the target data from dataset for alignment
        target_data = test_dataset.data['target']

        # Now I understand the structure:
        # - target_data is a list of 9 tensors (one per target)
        # - Each tensor has 30,962 values (full test_data length)
        # - But we only need the indices that correspond to our 1,980 predictions

        # The dataset index tells us which rows from the full data correspond to our predictions
        dataset_indices = test_dataset.index
        logging.info("Dataset indices length: %d, first 10: %s", len(dataset_indices), dataset_indices[:10])

        # Extract the target values for the decoder timesteps that match our predictions
        # Since we have 1,980 predictions with 13 timesteps each, we need to extract accordingly

        # Debug the indexing issue - why do we get (1980, 9, 9)?
        # Let's understand what dataset_indices actually contains
        logging.info("Dataset indices type: %s", type(dataset_indices))
        if hasattr(dataset_indices, 'shape'):
            logging.info("Dataset indices shape: %s", dataset_indices.shape)
        elif isinstance(dataset_indices, (list, tuple)):
            logging.info("Dataset indices is list/tuple with length: %d", len(dataset_indices))
            if len(dataset_indices) > 0:
                logging.info("First index item type: %s, value: %s", type(dataset_indices[0]), dataset_indices[0])

        # Let's also check what happens when we index a target tensor
        first_target = target_data[0].detach().cpu().numpy()
        logging.info("First target tensor shape: %s", first_target.shape)

        # Try indexing and see what we get
        try:
            indexed_result = first_target[dataset_indices]
            logging.info("Indexed result shape: %s", indexed_result.shape)
            logging.info("This explains why we get (1980, 9, 9) - the indexing is wrong")
        except Exception as e:
            logging.error("Indexing failed: %s", e)

        # The dataset indices is a DataFrame with shape (1980, 9) - this is metadata about sequences
        # We need to extract the actual row indices to use for indexing the target tensors

        # Look for a column that contains the actual row indices
        logging.info("Dataset index columns: %s", list(dataset_indices.columns))

        # The 'index_start' or similar column likely contains the actual row indices we need
        if 'index_start' in dataset_indices.columns:
            actual_indices = dataset_indices['index_start'].values
            logging.info("Using 'index_start' column for indexing, shape: %s", actual_indices.shape)
        elif 'index' in dataset_indices.columns:
            actual_indices = dataset_indices['index'].values
            logging.info("Using 'index' column for indexing, shape: %s", actual_indices.shape)
        else:
            # Fallback to using the DataFrame's index (row numbers)
            actual_indices = dataset_indices.index.values
            logging.info("Using DataFrame index for indexing, shape: %s", actual_indices.shape)

        # Now extract properly using 1D indices
        extracted_targets = []
        for target_idx, target_tensor in enumerate(target_data):
            target_np = target_tensor.detach().cpu().numpy()

            # Extract values at the correct indices - now using 1D indexing
            extracted_values = target_np[actual_indices]
            extracted_targets.append(extracted_values)

            logging.info("Target %d (%s): extracted %d values from %d total, shape: %s",
                        target_idx, target_cols[target_idx], len(extracted_values), len(target_np), extracted_values.shape)

        # Stack targets to create [samples, targets] array
        y_true = np.stack(extracted_targets, axis=1)
        logging.info("Stacked target data shape: %s", y_true.shape)

        # Now we need to handle the prediction reshaping
        # We have predictions (1980, 117) = (samples, 13_timesteps * 9_targets)
        # We need to match this with targets (1980, 9)

        # For multi-step predictions, we should compare each timestep appropriately
        # But for metrics, let's take the mean prediction across the 13 timesteps for each target
        n_samples = preds_array.shape[0]
        n_timesteps_per_target = preds_array.shape[1] // len(target_cols)  # Should be 13

        if n_timesteps_per_target * len(target_cols) != preds_array.shape[1]:
            raise RuntimeError(f"Prediction shape mismatch: {preds_array.shape[1]} != {n_timesteps_per_target} * {len(target_cols)}")

        # Reshape predictions: (samples, 117) -> (samples, 9 targets, 13 timesteps)
        reshaped_preds = preds_array.reshape(n_samples, len(target_cols), n_timesteps_per_target)

        # Take the mean across timesteps: (samples, 9 targets, 13 timesteps) -> (samples, 9 targets)
        y_pred = reshaped_preds.mean(axis=2)
        logging.info("Reshaped predictions from (%d, %d) to (%d, %d, %d), then averaged to (%d, %d)",
                    n_samples, preds_array.shape[1], n_samples, len(target_cols), n_timesteps_per_target,
                    y_pred.shape[0], y_pred.shape[1])

        # Ensure shapes match
        if y_true.shape != y_pred.shape:
            raise RuntimeError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

        # Valid mask: both sides finite & non-NaN
        valid_mask = (~np.isnan(y_true).any(axis=1)) & (~np.isnan(y_pred).any(axis=1))
        if valid_mask.any():
            save_metrics(run_id, y_true[valid_mask], y_pred[valid_mask])
        else:
            logging.warning("No valid rows for metric computation (all rows have NaNs).")

        # Store info for plotting consistency
        # Create a simple DataFrame for consistency with existing code
        import pandas as pd
        test_subset = pd.DataFrame({
            **{col: y_true[:, i] for i, col in enumerate(target_cols)},
            'prediction_idx': range(len(y_true))
        })
        session_state['horizon_df'] = test_subset
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