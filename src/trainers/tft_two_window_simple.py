"""
Simple Two-Window Prediction for TFT
Only modifies test/prediction phase - training remains unchanged.
"""

import logging
import os
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from configs.models.tft import TFTDatasetConfig
from .tft_trainer import predict_tft
from .tft_dataset import from_train_template, load_dataset_template
from .tft_model import load_tft_checkpoint, create_final_trainer
from .tft_utils import single_gpu_env, teardown_distributed


def create_early_window_test_data(test_data: pd.DataFrame) -> pd.DataFrame:
    """Filter test data to early window for each trajectory."""
    # Get required sequence length from TFT configuration
    config = TFTDatasetConfig()
    required_length = config.max_encoder_length + config.max_prediction_length

    filtered_groups = []

    for (model, scenario), group in test_data.groupby(['Model', 'Scenario']):
        steps = sorted(group['Step'].unique())
        if len(steps) >= required_length:  # Need at least required_length steps for early window
            early_steps = steps[:required_length]  # Steps 0 to (required_length-1)
            early_group = group[group['Step'].isin(early_steps)].copy()
            filtered_groups.append(early_group)

    result = pd.concat(filtered_groups, ignore_index=True) if filtered_groups else pd.DataFrame()
    logging.info(f"Early window test data: {len(result)} rows from {len(filtered_groups)} trajectories")
    return result


def predict_early_window(session_state: Dict, run_id: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """Generate predictions for early window using existing trained model."""
    logging.info("Generating early window predictions using existing model...")

    # Create early window test data
    test_data = session_state["test_data"]
    early_test_data = create_early_window_test_data(test_data)

    if len(early_test_data) == 0:
        logging.warning("No early window test data available")
        return np.array([]), pd.DataFrame()

    # Use existing trained model to predict on early window data
    with single_gpu_env():
        import torch
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            teardown_distributed()

        model = load_tft_checkpoint(run_id)
        train_template = load_dataset_template(run_id)

        # Create test dataset for early window
        try:
            test_dataset = from_train_template(train_template, early_test_data, mode="predict")
        except Exception as e:
            raise RuntimeError(f"Failed to build early window test dataset: {e}")

        # Generate predictions
        test_loader = test_dataset.to_dataloader(
            train=False, batch_size=64, num_workers=1, persistent_workers=False
        )

        trainer = create_final_trainer(TFTDatasetConfig())
        raw_preds = trainer.predict(model, test_loader)

        # Process predictions
        batch_arrays = []
        for arr in raw_preds:
            if torch.is_tensor(arr):
                batch_arrays.append(arr.detach().cpu().numpy())
            elif isinstance(arr, np.ndarray):
                batch_arrays.append(arr)

        if not batch_arrays:
            raise RuntimeError("No prediction outputs returned for early window.")

        preds = np.concatenate(batch_arrays, axis=0)

        # Handle multi-step predictions
        if preds.ndim == 3:  # (batch, horizon, targets)
            b, horizon, out = preds.shape
            logging.info(f"Early window multi-step predictions: batch={b}, horizon={horizon}, targets={out}")
            preds_flat = preds.reshape(b * horizon, out)
        else:
            preds_flat = preds

        # Create horizon dataframe
        targets = session_state["targets"]
        key_cols = ['Model', 'Scenario', 'Region', 'Step']

        # Build horizon index for early window
        horizon_df = early_test_data[key_cols + targets].drop_duplicates(key_cols).copy()
        horizon_df = horizon_df.reset_index(drop=True)

        # Align predictions with horizon dataframe
        min_len = min(len(preds_flat), len(horizon_df))
        preds_flat = preds_flat[:min_len]
        horizon_df = horizon_df.iloc[:min_len]

        logging.info(f"Early window predictions: {preds_flat.shape}, horizon_df: {horizon_df.shape}")
        return preds_flat, horizon_df


def combine_predictions_weighted(
    early_preds: np.ndarray,
    early_horizon: pd.DataFrame,
    late_preds: np.ndarray,
    late_horizon: pd.DataFrame,
    targets: list
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Combine early and late predictions with weighted averaging in overlap."""

    logging.info("Combining predictions with weighted averaging...")

    if len(early_preds) == 0:
        logging.info("No early predictions, using late predictions only")
        return late_preds, late_horizon

    if len(late_preds) == 0:
        logging.info("No late predictions, using early predictions only")
        return early_preds, early_horizon

    # Add predictions to dataframes
    early_df = early_horizon.copy()
    late_df = late_horizon.copy()

    for i, target in enumerate(targets):
        if i < early_preds.shape[1]:
            early_df[f'{target}_pred'] = early_preds[:, i]
        if i < late_preds.shape[1]:
            late_df[f'{target}_pred'] = late_preds[:, i]

    # Combine by trajectory
    combined_data = []

    # Get all unique trajectories
    early_trajectories = set(early_df.groupby(['Model', 'Scenario', 'Region']).groups.keys())
    late_trajectories = set(late_df.groupby(['Model', 'Scenario', 'Region']).groups.keys())
    all_trajectories = early_trajectories.union(late_trajectories)

    for (model, scenario, region) in all_trajectories:
        early_group = early_df[
            (early_df['Model'] == model) &
            (early_df['Scenario'] == scenario) &
            (early_df['Region'] == region)
        ].copy()

        late_group = late_df[
            (late_df['Model'] == model) &
            (late_df['Scenario'] == scenario) &
            (late_df['Region'] == region)
        ].copy()

        # Determine steps for each window
        early_steps = set(early_group['Step'].unique()) if len(early_group) > 0 else set()
        late_steps = set(late_group['Step'].unique()) if len(late_group) > 0 else set()
        overlap_steps = early_steps.intersection(late_steps)

        trajectory_combined = []

        # Process all steps for this trajectory
        all_steps = sorted(early_steps.union(late_steps))

        for step in all_steps:
            early_row = early_group[early_group['Step'] == step]
            late_row = late_group[late_group['Step'] == step]

            if step in overlap_steps and len(early_row) > 0 and len(late_row) > 0:
                # Overlap region: weighted average
                if len(overlap_steps) > 1:
                    min_overlap = min(overlap_steps)
                    max_overlap = max(overlap_steps)
                    # Linear interpolation: early weight decreases across overlap
                    early_weight = 1.0 - (step - min_overlap) / (max_overlap - min_overlap)
                    late_weight = 1.0 - early_weight
                else:
                    # Single overlap step: equal weights
                    early_weight = late_weight = 0.5

                # Create weighted average row
                combined_row = early_row.iloc[0].copy()
                for target in targets:
                    pred_col = f'{target}_pred'
                    if pred_col in early_row.columns and pred_col in late_row.columns:
                        early_pred = early_row[pred_col].iloc[0]
                        late_pred = late_row[pred_col].iloc[0]
                        combined_row[pred_col] = early_weight * early_pred + late_weight * late_pred

                trajectory_combined.append(combined_row)

            elif len(early_row) > 0:
                # Early-only region
                trajectory_combined.append(early_row.iloc[0])

            elif len(late_row) > 0:
                # Late-only region
                trajectory_combined.append(late_row.iloc[0])

        if trajectory_combined:
            traj_df = pd.DataFrame(trajectory_combined)
            combined_data.append(traj_df)

    # Combine all trajectories
    if combined_data:
        final_df = pd.concat(combined_data, ignore_index=True)

        # Extract predictions and clean dataframe
        pred_columns = [f'{target}_pred' for target in targets]
        final_preds = final_df[pred_columns].values
        final_horizon = final_df.drop(columns=pred_columns)

        logging.info(f"Combined predictions: {final_preds.shape}, horizon: {final_horizon.shape}")
        return final_preds, final_horizon
    else:
        logging.warning("No combined data generated")
        return late_preds, late_horizon


def predict_tft_two_window(session_state: Dict, run_id: str) -> np.ndarray:
    """Generate two-window predictions using existing trained model."""

    logging.info("Starting two-window prediction (using existing trained model)...")

    # Generate early window predictions
    early_preds, early_horizon = predict_early_window(session_state, run_id)

    # Generate late window predictions (existing approach)
    logging.info("Generating late window predictions...")
    late_preds = predict_tft(session_state, run_id)
    late_horizon = session_state.get('horizon_df')

    if late_horizon is None:
        raise ValueError("Late window horizon_df not found in session state")

    # Combine predictions with weighted averaging
    combined_preds, combined_horizon = combine_predictions_weighted(
        early_preds, early_horizon, late_preds, late_horizon, session_state["targets"]
    )

    # Update session state with combined results
    session_state['horizon_df'] = combined_horizon
    session_state['horizon_y_true'] = combined_horizon[session_state["targets"]].values

    # Store individual predictions for analysis
    session_state['early_predictions'] = early_preds
    session_state['late_predictions'] = late_preds

    logging.info("Two-window prediction completed successfully!")
    return combined_preds