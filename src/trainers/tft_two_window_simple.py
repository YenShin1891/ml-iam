"""Two-window prediction for TFT.

This module implements a two-window inference strategy on top of a trained
Temporal Fusion Transformer (TFT):

- An *early* window per trajectory, using the first
    ``encoder_length + prediction_length`` steps, provides predictions focused on
    the beginning of the evaluation period.
- A *late* window per trajectory, using the last
    ``encoder_length + prediction_length`` steps, provides predictions anchored
    at the end of each trajectory while preserving the full encoder+decoder span.
- The two sets of predictions are combined with a time-dependent linear
    weighting over any overlap, fading from early → late.

Training is unchanged; this only affects how predictions are generated and
combined at test time.
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd

from configs.models.tft import TFTTrainerConfig
from .tft_dataset import from_train_template, load_dataset_template
from .tft_model import load_tft_checkpoint, create_inference_trainer
from .tft_utils import single_gpu_env, teardown_distributed, get_default_num_workers


TRAJECTORY_COLS = ['Model', 'Scenario', 'Region']


@dataclass
class WindowConfig:
    """Configuration of encoder/decoder lengths used for window slicing."""

    encoder_length: int
    prediction_length: int

    @property
    def total_length(self) -> int:
        return self.encoder_length + self.prediction_length


@dataclass
class WindowPrediction:
    """Predictions and aligned horizon for a single window (early/late)."""

    preds: np.ndarray
    horizon: pd.DataFrame
    name: str


def _flatten_predictions_tensor(preds_tensor, torch_module) -> np.ndarray:
    """Flatten prediction tensor to 2D array (n_rows, n_targets)."""
    if torch_module.is_tensor(preds_tensor):
        preds_np = preds_tensor.detach().cpu().numpy()
    else:
        preds_np = np.array(preds_tensor)

    if preds_np.ndim == 3:  # (batch, horizon, targets)
        return preds_np.reshape(-1, preds_np.shape[-1])
    if preds_np.ndim == 2:  # (batch, targets)
        return preds_np
    raise ValueError(f"Unexpected prediction tensor shape: {preds_np.shape}")


def _collect_index_dataframe(index_attr) -> pd.DataFrame:
    """Normalize Prediction.index to a single DataFrame.

    Handles both a single DataFrame and iterables of DataFrames.
    """
    if hasattr(index_attr, '__iter__') and not isinstance(index_attr, (str, pd.DataFrame)):
        dfs = [df for df in index_attr if isinstance(df, pd.DataFrame) and not df.empty]
        if not dfs:
            raise RuntimeError("No valid DataFrames found in Prediction.index")
        return pd.concat(dfs, ignore_index=True)
    if isinstance(index_attr, pd.DataFrame):
        return index_attr.copy()
    raise RuntimeError(f"Unsupported Prediction.index type: {type(index_attr)}")


def _normalize_index_df(index_df: pd.DataFrame, template_time_idx: str | None) -> pd.DataFrame:
    """Ensure index_df has a consistent time index column name.

    If template_time_idx is present, it is left as-is. Otherwise, we try to
    infer a suitable time column (containing 'time' or 'step') and rename it
    to template_time_idx or a default 'Step'.
    """
    if template_time_idx and template_time_idx in index_df.columns:
        return index_df

    time_cols = [col for col in index_df.columns if 'time' in col.lower() or 'step' in col.lower()]
    if time_cols:
        index_df = index_df.rename(columns={time_cols[0]: template_time_idx or 'Step'})
    return index_df


def _expand_multistep_index(
    idx_df: pd.DataFrame,
    preds_tensor,
    window_data: pd.DataFrame,
    template_group_ids,
    template_time_idx: str,
    mode: str,
    torch_module,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Expand (batch, horizon, targets) predictions to per-step index rows.

    mode="early"  -> use the first pred_len steps from each trajectory.
    mode="late"   -> use the last  pred_len steps from each trajectory.
    """
    if not (torch_module.is_tensor(preds_tensor) and preds_tensor.ndim == 3):
        # Nothing to expand; return original index and flattened predictions.
        return idx_df.reset_index(drop=True), _flatten_predictions_tensor(preds_tensor, torch_module)

    n_samples, pred_len, out_size = preds_tensor.shape
    if len(idx_df) != n_samples or pred_len <= 1:
        # Shape mismatch or single-step horizon; fall back to simple flatten.
        return idx_df.reset_index(drop=True), _flatten_predictions_tensor(preds_tensor, torch_module)

    grouped = window_data.groupby(list(template_group_ids))

    expanded_rows: list[pd.Series] = []
    for i in range(n_samples):
        base_row = idx_df.iloc[i]
        trajectory_key = tuple(base_row[col] for col in template_group_ids)
        try:
            traj_data = grouped.get_group(trajectory_key).sort_values(template_time_idx)
            unique_steps = sorted(traj_data[template_time_idx].unique())
            if len(unique_steps) >= pred_len:
                if mode == "early":
                    steps_for_window = unique_steps[:pred_len]
                else:  # "late"
                    steps_for_window = unique_steps[-pred_len:]
            else:
                # Not enough steps; fall back to synthetic increments
                base_time = base_row[template_time_idx]
                steps_for_window = [base_time + h for h in range(pred_len)]
        except KeyError:
            # Trajectory not found, use fallback increment
            base_time = base_row[template_time_idx]
            steps_for_window = [base_time + h for h in range(pred_len)]

        for step_val in steps_for_window:
            new_row = base_row.copy()
            new_row[template_time_idx] = step_val
            expanded_rows.append(new_row)

    expanded_idx_df = pd.DataFrame(expanded_rows).reset_index(drop=True)
    preds_flat = preds_tensor.detach().cpu().numpy().reshape(n_samples * pred_len, out_size)
    return expanded_idx_df, preds_flat


def _create_early_window_test_data(test_data: pd.DataFrame, window_length: int, time_idx_col: str) -> pd.DataFrame:
    """Filter test data to early window for each trajectory (Model, Scenario, Region)."""
    filtered_groups = []

    for (model, scenario, region), group in test_data.groupby(['Model', 'Scenario', 'Region']):
        steps = sorted(group[time_idx_col].unique())
        if len(steps) >= window_length:  # Need at least window_length steps for early window
            early_steps = steps[:window_length]  # First window_length steps
            early_group = group[group[time_idx_col].isin(early_steps)].copy()
            filtered_groups.append(early_group)

    result = pd.concat(filtered_groups, ignore_index=True) if filtered_groups else pd.DataFrame()
    logging.info(f"Early window test data: {len(result)} rows from {len(filtered_groups)} trajectories")
    return result


def _predict_early_window(session_state: dict, run_id: str) -> WindowPrediction:
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

        # Determine time index and warm-start offset
        raw_time_idx = getattr(train_template, "time_idx", None)
        time_idx_col = raw_time_idx or session_state.get("tft_time_idx_column", "Step")
        target_offset = int(session_state.get("tft_target_offset", 0) or 0)

        # Get window lengths from template
        cfg = WindowConfig(
            encoder_length=getattr(train_template, 'max_encoder_length', 2),
            prediction_length=getattr(train_template, 'max_prediction_length', 13),
        )
        early_window_length = cfg.total_length

        # Create early window test data (keep early steps for encoder context)
        test_data = session_state["test_data"]
        early_test_data = _create_early_window_test_data(test_data, early_window_length, time_idx_col)

        if len(early_test_data) == 0:
            logging.warning("No early window test data available")
            return WindowPrediction(preds=np.array([]), horizon=pd.DataFrame(), name="early")

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

        preds_flat = _flatten_predictions_tensor(preds_tensor, torch)

        # Build horizon dataframe for early window
        targets = session_state["targets"]
        template_time_idx = getattr(train_template, "time_idx", None)
        template_group_ids = getattr(train_template, "group_ids", None)
        if template_time_idx is None or template_group_ids is None:
            raise ValueError("Dataset template is missing time_idx or group_ids; cannot align early window predictions")

        index_attr = returns.index
        index_df = _collect_index_dataframe(index_attr)
        idx_df = _normalize_index_df(index_df, template_time_idx)

        # Handle multi-step predictions for early window
        idx_df, preds_flat = _expand_multistep_index(
            idx_df,
            preds_tensor,
            early_test_data,
            template_group_ids,
            template_time_idx,
            mode="early",
            torch_module=torch,
        )

        # Build horizon dataframe
        key_cols = list(template_group_ids) + [template_time_idx]
        ref_cols = [c for c in key_cols + ['Year'] + targets if c in early_test_data.columns]
        horizon_df = idx_df[key_cols].merge(
            early_test_data.drop_duplicates(subset=key_cols)[ref_cols],
            on=key_cols,
            how='left'
        )

        if target_offset > 0 and template_time_idx in horizon_df.columns:
            horizon_mask = horizon_df[template_time_idx] >= target_offset
            dropped = int((~horizon_mask).sum())
            if dropped > 0:
                logging.info(
                    "Warm start offset %d: early-window horizon filter removed %d rows",
                    target_offset,
                    dropped,
                )
                horizon_df = horizon_df.loc[horizon_mask].reset_index(drop=True)
                preds_flat = preds_flat[horizon_mask.to_numpy()]

        logging.info(f"Early window predictions: {preds_flat.shape}, horizon_df: {horizon_df.shape}")
        return WindowPrediction(preds=preds_flat, horizon=horizon_df, name="early")


def _create_late_window_test_data(test_data: pd.DataFrame, window_length: int, time_idx_col: str) -> pd.DataFrame:
    """Create late window test data positioned to end at each trajectory's last step."""
    filtered_groups = []

    for (model, scenario, region), group in test_data.groupby(['Model', 'Scenario', 'Region']):
        steps = sorted(group[time_idx_col].unique())
        if len(steps) >= window_length:
            # Position window to end at last step while preserving full encoder+decoder span
            last_step = max(steps)
            first_step = last_step - window_length + 1

            # Find the actual steps that fall in this range
            late_steps = [s for s in steps if s >= first_step]
            if len(late_steps) >= window_length:
                # Take exactly window_length steps ending at last_step
                late_steps = late_steps[-window_length:]
                late_group = group[group[time_idx_col].isin(late_steps)].copy()
                filtered_groups.append(late_group)

    result = pd.concat(filtered_groups, ignore_index=True) if filtered_groups else pd.DataFrame()
    logging.info(f"Late window test data: {len(result)} rows from {len(filtered_groups)} trajectories")
    return result


def _predict_late_window(session_state: dict, run_id: str) -> WindowPrediction:
    """Generate predictions for late window positioned to end at trajectory ends."""
    logging.info("Generating late window predictions using existing model...")

    # Create late window test data
    test_data = session_state["test_data"]

    # Get actual prediction length from template
    train_template = load_dataset_template(run_id)
    raw_time_idx = getattr(train_template, "time_idx", None)
    time_idx_col = raw_time_idx or session_state.get("tft_time_idx_column", "Step")
    target_offset = int(session_state.get("tft_target_offset", 0) or 0)
    cfg = WindowConfig(
        encoder_length=getattr(train_template, 'max_encoder_length', 2),
        prediction_length=getattr(train_template, 'max_prediction_length', 13),
    )
    # Need full encoder+decoder span so model sees proper history while window still ends at last step
    window_length = cfg.total_length

    late_test_data = _create_late_window_test_data(test_data, window_length, time_idx_col)

    if len(late_test_data) == 0:
        logging.warning("No late window test data available")
        return WindowPrediction(preds=np.array([]), horizon=pd.DataFrame(), name="late")

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

        preds_flat = _flatten_predictions_tensor(preds_tensor, torch)

        # Build horizon dataframe for late window
        targets = session_state["targets"]
        template_time_idx = getattr(train_template, "time_idx", None)
        template_group_ids = getattr(train_template, "group_ids", None)
        if template_time_idx is None or template_group_ids is None:
            raise ValueError("Dataset template is missing time_idx or group_ids; cannot align late window predictions")

        index_attr = returns.index
        index_df = _collect_index_dataframe(index_attr)
        idx_df = _normalize_index_df(index_df, template_time_idx)

        # Handle multi-step predictions for late window
        idx_df, preds_flat = _expand_multistep_index(
            idx_df,
            preds_tensor,
            late_test_data,
            template_group_ids,
            template_time_idx,
            mode="late",
            torch_module=torch,
        )

        # Build horizon dataframe
        key_cols = list(template_group_ids) + [template_time_idx]
        ref_cols = [c for c in key_cols + ['Year'] + targets if c in late_test_data.columns]
        horizon_df = idx_df[key_cols].merge(
            late_test_data.drop_duplicates(subset=key_cols)[ref_cols],
            on=key_cols,
            how='left'
        )

        if target_offset > 0 and template_time_idx in horizon_df.columns:
            horizon_mask = horizon_df[template_time_idx] >= target_offset
            dropped = int((~horizon_mask).sum())
            if dropped > 0:
                logging.info(
                    "Warm start offset %d: late-window horizon filter removed %d rows",
                    target_offset,
                    dropped,
                )
                horizon_df = horizon_df.loc[horizon_mask].reset_index(drop=True)
                preds_flat = preds_flat[horizon_mask.to_numpy()]

        logging.info(f"Late window predictions: {preds_flat.shape}, horizon_df: {horizon_df.shape}")
        return WindowPrediction(preds=preds_flat, horizon=horizon_df, name="late")


def _combine_predictions_weighted(
    early: WindowPrediction,
    late: WindowPrediction,
    targets: list,
    time_idx_col: str
) -> WindowPrediction:
    """Combine early and late predictions with weighted averaging in overlap."""

    logging.info("Combining predictions with weighted averaging...")

    # Debug information
    if len(early.preds) > 0 and 'Year' in early.horizon.columns:
        early_years = early.horizon['Year'].unique()
        logging.info(f"Early window: {len(early_years)} years ({min(early_years)}-{max(early_years)})")
    if len(late.preds) > 0 and 'Year' in late.horizon.columns:
        late_years = late.horizon['Year'].unique()
        logging.info(f"Late window: {len(late_years)} years ({min(late_years)}-{max(late_years)})")

    if len(early.preds) == 0:
        logging.info("No early predictions, using late predictions only")
        return late

    if len(late.preds) == 0:
        logging.info("No late predictions, using early predictions only")
        return early

    # Add predictions to dataframes
    early_df = early.horizon.copy()
    late_df = late.horizon.copy()

    for i, target in enumerate(targets):
        if i < early.preds.shape[1]:
            early_df[f'{target}_pred'] = early.preds[:, i]
        if i < late.preds.shape[1]:
            late_df[f'{target}_pred'] = late.preds[:, i]

    # Combine by trajectory
    combined_data = []

    # Get all unique trajectories
    trajectory_keys = TRAJECTORY_COLS
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
        early_steps = set(early_group[time_idx_col].unique()) if len(early_group) > 0 else set()
        late_steps = set(late_group[time_idx_col].unique()) if len(late_group) > 0 else set()
        overlap_steps = early_steps.intersection(late_steps)

        trajectory_combined = []

        # Process all steps for this trajectory
        all_steps = sorted(early_steps.union(late_steps))

        for step in all_steps:
            early_row = early_group[early_group[time_idx_col] == step]
            late_row = late_group[late_group[time_idx_col] == step]

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
        return WindowPrediction(preds=final_preds, horizon=final_horizon, name="combined")
    else:
        logging.warning("No combined data generated")
        # If we could not form any combined data, fall back to late window
        return late


def predict_tft_two_window(session_state: Dict, run_id: str) -> np.ndarray:
    """Generate two-window predictions.

    Uses early and late windows only; if coverage assumptions are violated
    (e.g., missing trajectories), this is logged for debugging but no
    automatic fallback predictions are added.
    """

    logging.info("Starting two-window prediction (using existing trained model)...")

    # Generate early window predictions
    early_window = _predict_early_window(session_state, run_id)

    # Generate late window predictions (positioned to end at trajectory ends)
    logging.info("Generating late window predictions...")
    late_window = _predict_late_window(session_state, run_id)

    if late_window.horizon is None or len(late_window.horizon) == 0:
        raise ValueError("Late window predictions failed or returned empty results")

    # Combine predictions with weighted averaging
    time_idx_col = session_state.get("tft_time_idx_column", "Step")
    combined_window = _combine_predictions_weighted(
        early_window,
        late_window,
        session_state["targets"],
        time_idx_col,
    )

    # Check coverage and log any missing trajectories (assumption violations)
    test_data = session_state["test_data"]
    all_test_trajectories = set(
        map(tuple, test_data[TRAJECTORY_COLS].drop_duplicates().itertuples(index=False, name=None))
    )
    combined_trajectories = set(
        map(tuple, combined_window.horizon[TRAJECTORY_COLS].drop_duplicates().itertuples(index=False, name=None))
    )
    missing_trajectories = all_test_trajectories - combined_trajectories

    logging.info(f"Two-window coverage: {len(combined_trajectories)}/{len(all_test_trajectories)} trajectories")
    logging.info(f"Missing trajectories: {len(missing_trajectories)}")

    if len(missing_trajectories) > 0:
        # This indicates that the two-window construction did not cover all
        # trajectories present in the test data. We intentionally do not
        # auto-fill these with alternative predictions; instead, log for
        # investigation.
        logging.error(
            "Two-window prediction assumption violated: %d/%d trajectories missing from combined horizon.",
            len(missing_trajectories),
            len(all_test_trajectories),
        )
        # Log a small sample of missing trajectories for debugging
        sample_missing = list(missing_trajectories)[:10]
        logging.error("Example missing trajectories (up to 10): %s", sample_missing)

    # Update session state with combined results (original logic)
    session_state['horizon_df'] = combined_window.horizon
    session_state['horizon_y_true'] = combined_window.horizon[session_state["targets"]].values

    # Store individual predictions for analysis
    session_state['early_predictions'] = early_window.preds
    session_state['late_predictions'] = late_window.preds

    logging.info("Two-window prediction completed successfully!")
    return combined_window.preds