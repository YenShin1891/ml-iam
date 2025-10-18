"""
Two-Window Prediction for TFT
Only modifies test/prediction phase - training remains unchanged.
"""

import logging
import os
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from configs.models.tft import TFTTrainerConfig
from configs.paths import RESULTS_PATH
from .tft_trainer import predict_tft
from .tft_dataset import from_train_template, load_dataset_template
from .tft_model import load_tft_checkpoint, create_inference_trainer
from .tft_utils import single_gpu_env, teardown_distributed, get_default_num_workers


def _create_early_window_test_data(test_data: pd.DataFrame, window_length: int) -> pd.DataFrame:
    """Filter test data to early window for each trajectory."""
    filtered_groups = []

    for (model, scenario), group in test_data.groupby(['Model', 'Scenario']):
        steps = sorted(group['Step'].unique())
        if len(steps) >= window_length:  # Need at least window_length steps for early window
            early_steps = steps[:window_length]  # First window_length steps
            early_group = group[group['Step'].isin(early_steps)].copy()
            filtered_groups.append(early_group)

    result = pd.concat(filtered_groups, ignore_index=True) if filtered_groups else pd.DataFrame()
    logging.info(f"Early window test data: {len(result)} rows from {len(filtered_groups)} trajectories")
    return result


def _predict_early_window(session_state: dict, run_id: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """Generate predictions for early window using existing trained model."""
    logging.info("Generating early window predictions using existing model...")

    # Use existing trained model to predict on early window data
    with single_gpu_env():
        import torch
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            teardown_distributed()

        model = load_tft_checkpoint(run_id)

        # Load dataset template
        train_template = load_dataset_template(run_id)

        # Get window lengths from template
        max_encoder_length = getattr(train_template, 'max_encoder_length', 2)
        max_prediction_length = getattr(train_template, 'max_prediction_length', 13)
        early_window_length = max_encoder_length + max_prediction_length

        # Create early window test data
        test_data = session_state["test_data"]
        early_test_data = _create_early_window_test_data(test_data, early_window_length)

        if len(early_test_data) == 0:
            logging.warning("No early window test data available")
            return np.array([]), pd.DataFrame()

        # Create test dataset for early window
        try:
            test_dataset = from_train_template(train_template, early_test_data, mode="eval")
        except Exception as e:
            raise RuntimeError(f"Failed to build early window test dataset: {e}")

        # Generate predictions
        trainer_cfg = TFTTrainerConfig()
        test_loader = test_dataset.to_dataloader(
            train=False,
            batch_size=trainer_cfg.batch_size,
            num_workers=get_default_num_workers(),
            persistent_workers=False
        )

        trainer = create_inference_trainer()
        returns = model.predict(test_loader, return_index=True)

        from pytorch_forecasting.models.base._base_model import Prediction as _PFPrediction  # type: ignore

        if not isinstance(returns, _PFPrediction):
            raise RuntimeError(f"Expected Prediction object, got {type(returns)}")

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

        def _flatten_predictions(preds_tensor):
            """Flatten predictions tensor to 2D array."""
            if torch.is_tensor(preds_tensor):
                preds_np = preds_tensor.detach().cpu().numpy()
            else:
                preds_np = np.array(preds_tensor)

            if preds_np.ndim == 3:  # (batch, horizon, targets)
                return preds_np.reshape(-1, preds_np.shape[-1])
            elif preds_np.ndim == 2:  # (batch, targets)
                return preds_np
            else:
                raise ValueError(f"Unexpected prediction tensor shape: {preds_np.shape}")

        preds_flat = _flatten_predictions(preds_tensor)

        # Build horizon dataframe for early window
        targets = session_state["targets"]
        template_time_idx = getattr(train_template, "time_idx", None)
        template_group_ids = getattr(train_template, "group_ids", None)
        if template_time_idx is None or template_group_ids is None:
            raise ValueError("Dataset template is missing time_idx or group_ids; cannot align early window predictions")

        index_attr = returns.index
        if hasattr(index_attr, '__iter__') and not isinstance(index_attr, (str, pd.DataFrame)):
            dfs = [df for df in index_attr if isinstance(df, pd.DataFrame) and not df.empty]
            if not dfs:
                raise RuntimeError("No valid DataFrames found in Prediction.index for early window")
            index_df = pd.concat(dfs, ignore_index=True)
        elif isinstance(index_attr, pd.DataFrame):
            index_df = index_attr.copy()
        else:
            raise RuntimeError(f"Unsupported Prediction.index type for early window: {type(index_attr)}")

        def _normalize_index_df(index_df, template_time_idx):
            """Normalize index dataframe columns."""
            if template_time_idx and template_time_idx in index_df.columns:
                return index_df
            else:
                # Try to find time index column
                time_cols = [col for col in index_df.columns if 'time' in col.lower() or 'step' in col.lower()]
                if time_cols:
                    index_df = index_df.rename(columns={time_cols[0]: template_time_idx or 'Step'})
                return index_df

        idx_df = _normalize_index_df(index_df, template_time_idx)

        # Handle multi-step predictions for early window
        if torch.is_tensor(preds_tensor) and preds_tensor.ndim == 3:
            n_samples, pred_len, out_size = preds_tensor.shape
            if len(idx_df) == n_samples and pred_len > 1:
                # Pre-group test data for efficient lookup - much faster than filtering in loop
                early_test_grouped = early_test_data.groupby(list(template_group_ids))

                expanded_rows = []
                for i in range(n_samples):
                    base_row = idx_df.iloc[i]

                    # Get the actual trajectory data to find the correct step sequence
                    trajectory_key = tuple(base_row[col] for col in template_group_ids)
                    try:
                        traj_data = early_test_grouped.get_group(trajectory_key).sort_values(template_time_idx)

                        if len(traj_data) >= pred_len:
                            # Use the actual steps from the trajectory data (first pred_len steps)
                            actual_steps = sorted(traj_data[template_time_idx].unique())[:pred_len]
                            for step_val in actual_steps:
                                new_row = base_row.copy()
                                new_row[template_time_idx] = step_val
                                expanded_rows.append(new_row)
                        else:
                            # Fallback to increment if trajectory data is insufficient
                            base_time = base_row[template_time_idx]
                            for h in range(pred_len):
                                new_row = base_row.copy()
                                new_row[template_time_idx] = base_time + h
                                expanded_rows.append(new_row)
                    except KeyError:
                        # Trajectory not found, use fallback increment
                        base_time = base_row[template_time_idx]
                        for h in range(pred_len):
                            new_row = base_row.copy()
                            new_row[template_time_idx] = base_time + h
                            expanded_rows.append(new_row)

                idx_df = pd.DataFrame(expanded_rows).reset_index(drop=True)
                preds_flat = preds_tensor.detach().cpu().numpy().reshape(n_samples * pred_len, out_size)

        # Build horizon dataframe
        key_cols = list(template_group_ids) + [template_time_idx]
        ref_cols = [c for c in key_cols + ['Year'] + targets if c in early_test_data.columns]
        horizon_df = idx_df[key_cols].merge(
            early_test_data[ref_cols].drop_duplicates(key_cols),
            on=key_cols,
            how='left'
        )

        logging.info(f"Early window predictions: {preds_flat.shape}, horizon_df: {horizon_df.shape}")
        return preds_flat, horizon_df


def _create_late_window_test_data(test_data: pd.DataFrame, window_length: int, prediction_length: int) -> pd.DataFrame:
    """Create late window test data positioned to end at each trajectory's last step."""
    filtered_groups = []

    for (model, scenario, region), group in test_data.groupby(['Model', 'Scenario', 'Region']):
        steps = sorted(group['Step'].unique())
        if len(steps) >= window_length:
            # Position window to end at last step, but ensure we get exactly prediction_length prediction steps
            last_step = max(steps)
            first_step = last_step - window_length + 1

            # Find the actual steps that fall in this range
            late_steps = [s for s in steps if s >= first_step]
            if len(late_steps) >= window_length:
                # Take exactly window_length steps ending at last_step
                late_steps = late_steps[-window_length:]
                late_group = group[group['Step'].isin(late_steps)].copy()
                filtered_groups.append(late_group)

    result = pd.concat(filtered_groups, ignore_index=True) if filtered_groups else pd.DataFrame()
    logging.info(f"Late window test data: {len(result)} rows from {len(filtered_groups)} trajectories")
    return result


def _predict_late_window(session_state: dict, run_id: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Generate predictions for late window positioned to end at trajectory ends."""
    logging.info("Generating late window predictions using existing model...")

    # Create late window test data
    test_data = session_state["test_data"]

    # Get actual prediction length from template
    train_template = load_dataset_template(run_id)
    max_encoder_length = getattr(train_template, 'max_encoder_length', 2)
    max_prediction_length = getattr(train_template, 'max_prediction_length', 13)
    # Use only prediction length for window - this ensures late window ends at trajectory end
    window_length = max_prediction_length
    prediction_length = max_prediction_length

    late_test_data = _create_late_window_test_data(test_data, window_length, prediction_length)

    if len(late_test_data) == 0:
        logging.warning("No late window test data available")
        return np.array([]), pd.DataFrame()

    # Use existing trained model to predict on late window data
    with single_gpu_env():
        import torch
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            teardown_distributed()

        model = load_tft_checkpoint(run_id)

        # Create test dataset for late window with custom handling for better coverage
        try:
            # First try with predict=True for proper prediction format
            test_dataset = from_train_template(train_template, late_test_data, mode="predict")
        except Exception as e1:
            # If that fails due to length constraints, try with eval mode
            try:
                logging.warning(f"Late window predict mode failed ({e1}), trying eval mode")
                test_dataset = from_train_template(train_template, late_test_data, mode="eval")
            except Exception as e2:
                raise RuntimeError(f"Failed to build late window test dataset with both predict ({e1}) and eval ({e2}) modes")

        # Generate predictions
        trainer_cfg = TFTTrainerConfig()
        test_loader = test_dataset.to_dataloader(
            train=False,
            batch_size=trainer_cfg.batch_size,
            num_workers=get_default_num_workers(),
            persistent_workers=False
        )

        trainer = create_inference_trainer()
        returns = model.predict(test_loader, return_index=True)

        from pytorch_forecasting.models.base._base_model import Prediction as _PFPrediction  # type: ignore

        if not isinstance(returns, _PFPrediction):
            raise RuntimeError(f"Expected Prediction object, got {type(returns)}")

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

        def _flatten_predictions(preds_tensor):
            """Flatten predictions tensor to 2D array."""
            if torch.is_tensor(preds_tensor):
                preds_np = preds_tensor.detach().cpu().numpy()
            else:
                preds_np = np.array(preds_tensor)

            if preds_np.ndim == 3:  # (batch, horizon, targets)
                return preds_np.reshape(-1, preds_np.shape[-1])
            elif preds_np.ndim == 2:  # (batch, targets)
                return preds_np
            else:
                raise ValueError(f"Unexpected prediction tensor shape: {preds_np.shape}")

        preds_flat = _flatten_predictions(preds_tensor)

        # Build horizon dataframe for late window
        targets = session_state["targets"]
        template_time_idx = getattr(train_template, "time_idx", None)
        template_group_ids = getattr(train_template, "group_ids", None)
        if template_time_idx is None or template_group_ids is None:
            raise ValueError("Dataset template is missing time_idx or group_ids; cannot align late window predictions")

        index_attr = returns.index
        if hasattr(index_attr, '__iter__') and not isinstance(index_attr, (str, pd.DataFrame)):
            dfs = [df for df in index_attr if isinstance(df, pd.DataFrame) and not df.empty]
            if not dfs:
                raise RuntimeError("No valid DataFrames found in Prediction.index for late window")
            index_df = pd.concat(dfs, ignore_index=True)
        elif isinstance(index_attr, pd.DataFrame):
            index_df = index_attr.copy()
        else:
            raise RuntimeError(f"Unsupported Prediction.index type for late window: {type(index_attr)}")

        def _normalize_index_df(index_df, template_time_idx):
            """Normalize index dataframe columns."""
            if template_time_idx and template_time_idx in index_df.columns:
                return index_df
            else:
                # Try to find time index column
                time_cols = [col for col in index_df.columns if 'time' in col.lower() or 'step' in col.lower()]
                if time_cols:
                    index_df = index_df.rename(columns={time_cols[0]: template_time_idx or 'Step'})
                return index_df

        idx_df = _normalize_index_df(index_df, template_time_idx)

        # Handle multi-step predictions for late window
        if torch.is_tensor(preds_tensor) and preds_tensor.ndim == 3:
            n_samples, pred_len, out_size = preds_tensor.shape
            if len(idx_df) == n_samples and pred_len > 1:
                # Pre-group test data for efficient lookup - much faster than filtering in loop
                late_test_grouped = late_test_data.groupby(list(template_group_ids))

                expanded_rows = []
                for i in range(n_samples):
                    base_row = idx_df.iloc[i]

                    # Get the actual trajectory data to find the correct step sequence
                    trajectory_key = tuple(base_row[col] for col in template_group_ids)
                    try:
                        traj_data = late_test_grouped.get_group(trajectory_key).sort_values(template_time_idx)

                        if len(traj_data) >= pred_len:
                            # Use the actual steps from the trajectory data
                            actual_steps = sorted(traj_data[template_time_idx].unique())[-pred_len:]
                            for step_val in actual_steps:
                                new_row = base_row.copy()
                                new_row[template_time_idx] = step_val
                                expanded_rows.append(new_row)
                        else:
                            # Fallback to increment if trajectory data is insufficient
                            base_time = base_row[template_time_idx]
                            for h in range(pred_len):
                                new_row = base_row.copy()
                                new_row[template_time_idx] = base_time + h
                                expanded_rows.append(new_row)
                    except KeyError:
                        # Trajectory not found, use fallback increment
                        base_time = base_row[template_time_idx]
                        for h in range(pred_len):
                            new_row = base_row.copy()
                            new_row[template_time_idx] = base_time + h
                            expanded_rows.append(new_row)

                idx_df = pd.DataFrame(expanded_rows).reset_index(drop=True)
                preds_flat = preds_tensor.detach().cpu().numpy().reshape(n_samples * pred_len, out_size)

        # Build horizon dataframe
        key_cols = list(template_group_ids) + [template_time_idx]
        ref_cols = [c for c in key_cols + ['Year'] + targets if c in late_test_data.columns]
        horizon_df = idx_df[key_cols].merge(
            late_test_data[ref_cols].drop_duplicates(key_cols),
            on=key_cols,
            how='left'
        )

        logging.info(f"Late window predictions: {preds_flat.shape}, horizon_df: {horizon_df.shape}")
        return preds_flat, horizon_df


def _combine_predictions_weighted(
    early_preds: np.ndarray,
    early_horizon: pd.DataFrame,
    late_preds: np.ndarray,
    late_horizon: pd.DataFrame,
    targets: list
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Combine early and late predictions with weighted averaging in overlap."""

    logging.info("Combining predictions with weighted averaging...")

    # Debug information
    if len(early_preds) > 0 and 'Year' in early_horizon.columns:
        early_years = early_horizon['Year'].unique()
        logging.info(f"Early window: {len(early_years)} years ({min(early_years)}-{max(early_years)})")
    if len(late_preds) > 0 and 'Year' in late_horizon.columns:
        late_years = late_horizon['Year'].unique()
        logging.info(f"Late window: {len(late_years)} years ({min(late_years)}-{max(late_years)})")

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
    trajectory_keys = ['Model', 'Scenario', 'Region']
    early_trajectories = set(
        map(tuple, early_df[trajectory_keys].drop_duplicates().itertuples(index=False, name=None))
    ) if len(early_df) > 0 else set()
    late_trajectories = set(
        map(tuple, late_df[trajectory_keys].drop_duplicates().itertuples(index=False, name=None))
    ) if len(late_df) > 0 else set()
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

        # Debug final result
        if 'Year' in final_horizon.columns:
            final_years = final_horizon['Year'].unique()
            logging.info(f"Final combined: {len(final_years)} years ({min(final_years)}-{max(final_years)})")

        logging.info(f"Combined predictions: {final_preds.shape}, horizon: {final_horizon.shape}")
        return final_preds, final_horizon
    else:
        logging.warning("No combined data generated")
        return late_preds, late_horizon


def predict_tft_two_window(session_state: Dict, run_id: str) -> np.ndarray:
    """Generate two-window predictions with fallback to standard prediction for missing trajectories."""

    logging.info("Starting two-window prediction (using existing trained model)...")

    # Generate early window predictions
    early_preds, early_horizon = _predict_early_window(session_state, run_id)

    # Generate late window predictions (positioned to end at trajectory ends)
    logging.info("Generating late window predictions...")
    late_preds, late_horizon = _predict_late_window(session_state, run_id)

    if late_horizon is None or len(late_horizon) == 0:
        raise ValueError("Late window predictions failed or returned empty results")

    # Combine predictions with weighted averaging
    combined_preds, combined_horizon = _combine_predictions_weighted(
        early_preds, early_horizon, late_preds, late_horizon, session_state["targets"]
    )

    # Check coverage and add fallback for missing trajectories
    test_data = session_state["test_data"]
    all_test_trajectories = set(test_data.apply(lambda x: f"{x['Model']}|{x['Scenario']}|{x['Region']}", axis=1).unique())
    combined_trajectories = set(combined_horizon.apply(lambda x: f"{x['Model']}|{x['Scenario']}|{x['Region']}", axis=1).unique())
    missing_trajectories = all_test_trajectories - combined_trajectories

    logging.info(f"Two-window coverage: {len(combined_trajectories)}/{len(all_test_trajectories)} trajectories")
    logging.info(f"Missing trajectories: {len(missing_trajectories)}")

    if len(missing_trajectories) > 0:
        logging.info("Adding fallback predictions for missing trajectories using standard prediction...")

        # Try standard prediction for fallback
        try:
            fallback_preds = predict_tft(session_state, run_id)
            fallback_horizon = session_state.get('horizon_df')

            if fallback_horizon is not None and len(fallback_horizon) > 0:
                # Get trajectories that are in fallback but not in combined
                fallback_trajectories = set(fallback_horizon.apply(lambda x: f"{x['Model']}|{x['Scenario']}|{x['Region']}", axis=1).unique())
                additional_trajectories = fallback_trajectories - combined_trajectories

                if len(additional_trajectories) > 0:
                    logging.info(f"Adding {len(additional_trajectories)} trajectories from standard prediction")

                    # Extract the missing trajectories from fallback
                    additional_rows = []
                    for traj in additional_trajectories:
                        model, scenario, region = traj.split('|')
                        traj_rows = fallback_horizon[
                            (fallback_horizon['Model'] == model) &
                            (fallback_horizon['Scenario'] == scenario) &
                            (fallback_horizon['Region'] == region)
                        ]
                        additional_rows.append(traj_rows)

                    if additional_rows:
                        additional_df = pd.concat(additional_rows, ignore_index=True)

                        # Combine with two-window results
                        final_horizon = pd.concat([combined_horizon, additional_df], ignore_index=True)
                        final_preds = np.vstack([combined_preds, fallback_preds[fallback_horizon.index.isin(additional_df.index)]])

                        logging.info(f"Final coverage: {len(final_horizon.groupby(['Model', 'Scenario', 'Region']))} trajectories")

                        # Update session state with final results
                        session_state['horizon_df'] = final_horizon
                        session_state['horizon_y_true'] = final_horizon[session_state["targets"]].values
                        session_state['early_predictions'] = early_preds
                        session_state['late_predictions'] = late_preds

                        logging.info("Two-window prediction with fallback completed successfully!")
                        return final_preds

        except Exception as e:
            logging.warning(f"Fallback prediction failed: {e}")

    # Update session state with combined results (original logic)
    session_state['horizon_df'] = combined_horizon
    session_state['horizon_y_true'] = combined_horizon[session_state["targets"]].values

    # Store individual predictions for analysis
    session_state['early_predictions'] = early_preds
    session_state['late_predictions'] = late_preds

    logging.info("Two-window prediction completed successfully!")
    return combined_preds