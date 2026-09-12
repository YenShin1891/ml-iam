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

Each window is sliced to exactly ``encoder_length + prediction_length`` steps
and predicted with ``predict=True``, so every trajectory yields one sample whose
decoder covers the window's last ``prediction_length`` steps.  ``Prediction.index``
reports the first decoder step, and horizon step ``h`` belongs to ``base + h``.

Training is unchanged; this only affects how predictions are generated and
combined at test time.

Coverage note
-------------
Both windows require ``encoder_length + prediction_length`` steps, read off
the training template rather than assumed.  The horizon is pinned to
``MAX_SERIES_LENGTH - MAX_CONTEXT_LENGTH`` so that a run with a shorter
encoder predicts the same steps, which makes the window shorter by exactly
the difference in context.  Trajectories shorter than the window are excluded
from *both* windows and therefore receive no predictions.

The single-window method (``predict_tft`` in ``tft_trainer.py``) cannot fill
that gap: ``predict=True`` raises ``min_prediction_length`` to
``max_prediction_length``, so it too needs ``min_encoder_length +
prediction_length`` steps per trajectory -- the same window as here.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional
import numpy as np
import pandas as pd

from configs.models.tft import TFTTrainerConfig
from .tft_dataset import from_train_template, load_dataset_template
from .tft_model import load_tft_checkpoint
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


def _normalize_index_df(index_df: pd.DataFrame, template_time_idx: Optional[str]) -> pd.DataFrame:
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


def _expand_horizon_index(
    idx_df: pd.DataFrame,
    preds_tensor,
    template_time_idx: str,
    torch_module,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Expand (batch, horizon, targets) predictions to one row per decoder step.

    ``Prediction.index`` holds one row per sample, whose time index is the
    *first decoder step*; decoder steps are consecutive, so horizon step ``h``
    of sample ``i`` belongs to ``base_time + h``.  Labelling the horizon with
    the window's own first steps instead would shift every early-window
    prediction back by the encoder length.
    """
    if not (torch_module.is_tensor(preds_tensor) and preds_tensor.ndim == 3):
        # Nothing to expand; return original index and flattened predictions.
        return idx_df.reset_index(drop=True), _flatten_predictions_tensor(preds_tensor, torch_module)

    n_samples, pred_len, out_size = preds_tensor.shape
    if len(idx_df) != n_samples or pred_len <= 1:
        # Shape mismatch or single-step horizon; fall back to simple flatten.
        return idx_df.reset_index(drop=True), _flatten_predictions_tensor(preds_tensor, torch_module)

    if template_time_idx not in idx_df.columns:
        raise KeyError(
            f"Time index column '{template_time_idx}' not found in prediction index "
            f"columns: {list(idx_df.columns)}"
        )

    expanded = idx_df.loc[idx_df.index.repeat(pred_len)].reset_index(drop=True)
    offsets = np.tile(np.arange(pred_len), n_samples)
    expanded[template_time_idx] = expanded[template_time_idx].to_numpy() + offsets

    preds_flat = preds_tensor.detach().cpu().numpy().reshape(n_samples * pred_len, out_size)
    return expanded, preds_flat


def _unpack_prediction_output(returns, torch_module):
    """Pull a single (batch, horizon, targets) tensor out of a Prediction."""
    from pytorch_forecasting.models.base._base_model import Prediction as _PFPrediction  # type: ignore

    if not isinstance(returns, _PFPrediction):
        raise RuntimeError(f"Expected Prediction object, got {type(returns)}")

    outputs = returns.output
    if isinstance(outputs, list):
        if len(outputs) == 0:
            raise RuntimeError("Prediction.output list is empty.")
        if not all(torch_module.is_tensor(o) for o in outputs):
            raise RuntimeError("All elements in Prediction.output list must be tensors.")
        return outputs[0] if len(outputs) == 1 else torch_module.stack(outputs, dim=-1)
    if torch_module.is_tensor(outputs):
        return outputs
    raise RuntimeError(f"Unsupported Prediction.output type: {type(outputs)}")


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


def predict_window_frame(
    model,
    template,
    data: pd.DataFrame,
    targets: list,
    name: str,
    slice_window: Callable[[pd.DataFrame, int, str], pd.DataFrame],
    *,
    target_offset: int = 0,
    loader_kwargs: Optional[dict] = None,
    predict_kwargs: Optional[dict] = None,
) -> WindowPrediction:
    """Predict one window (early or late) of every trajectory in *data*.

    Both windows are sliced to exactly ``encoder_length + prediction_length``
    steps, so ``predict=True`` yields one sample per trajectory whose decoder
    covers the window's last ``prediction_length`` steps.

    *model* and *template* are already loaded.  *loader_kwargs* reach
    ``to_dataloader`` and *predict_kwargs* reach ``model.predict`` (for
    instance ``trainer_kwargs`` pinning the Trainer to the CPU), so the test
    phase can run this on a GPU with many loader workers and the dashboard's
    what-if view can run the very same forecast in-process without either.
    """
    import torch

    template_time_idx = getattr(template, "time_idx", None)
    template_group_ids = getattr(template, "group_ids", None)
    if not template_time_idx or not template_group_ids:
        raise ValueError(
            f"Dataset template is missing time_idx or group_ids; "
            f"cannot align {name} window predictions"
        )

    cfg = WindowConfig(
        encoder_length=template.max_encoder_length,
        prediction_length=template.max_prediction_length,
    )

    window_data = slice_window(data, cfg.total_length, template_time_idx)
    if len(window_data) == 0:
        logging.warning("No %s window test data available", name)
        return WindowPrediction(preds=np.array([]), horizon=pd.DataFrame(), name=name)

    # predict=True is what makes the horizon alignable: it emits exactly one
    # sample per trajectory.  Eval mode would enumerate every (start,
    # decoder length) pair instead — dozens of overlapping samples per
    # trajectory, most of them padded — with no way to tell which
    # prediction belongs to which step.
    try:
        test_dataset = from_train_template(template, window_data, mode="predict")
    except Exception as e:
        raise RuntimeError(f"Failed to build {name} window test dataset: {e}") from e

    loader_options = {
        "batch_size": TFTTrainerConfig().batch_size,
        "num_workers": 0,
        "persistent_workers": False,
    }
    loader_options.update(loader_kwargs or {})
    test_loader = test_dataset.to_dataloader(train=False, **loader_options)

    returns = model.predict(test_loader, return_index=True, **(predict_kwargs or {}))
    preds_tensor = _unpack_prediction_output(returns, torch)

    index_df = _collect_index_dataframe(returns.index)
    idx_df = _normalize_index_df(index_df, template_time_idx)
    idx_df, preds_flat = _expand_horizon_index(idx_df, preds_tensor, template_time_idx, torch)

    # Build horizon dataframe.  The __observed masks come along so the
    # metrics can skip zero-filled and interpolated targets, as the LSTM
    # and XGBoost paths do.
    from configs.data import POPULATION_COLUMN
    from src.data.preprocess import observed_mask_columns
    targets = list(targets)
    key_cols = list(template_group_ids) + [template_time_idx]
    ref_cols = [
        c for c in key_cols + ['Year'] + targets + observed_mask_columns(targets) + [POPULATION_COLUMN]
        if c in window_data.columns
    ]
    horizon_df = idx_df[key_cols].merge(
        window_data.drop_duplicates(subset=key_cols)[ref_cols],
        on=key_cols,
        how='left'
    )

    if target_offset > 0 and template_time_idx in horizon_df.columns:
        horizon_mask = horizon_df[template_time_idx] >= target_offset
        dropped = int((~horizon_mask).sum())
        if dropped > 0:
            logging.info(
                "Warm start offset %d: %s-window horizon filter removed %d rows",
                target_offset,
                name,
                dropped,
            )
            horizon_df = horizon_df.loc[horizon_mask].reset_index(drop=True)
            preds_flat = preds_flat[horizon_mask.to_numpy()]

    # Convert per-capita predictions back to absolute units.
    from src.data.preprocess import denormalize_by_population
    preds_flat = denormalize_by_population(preds_flat, horizon_df[POPULATION_COLUMN].values)

    logging.info(f"{name.capitalize()} window predictions: {preds_flat.shape}, horizon_df: {horizon_df.shape}")
    return WindowPrediction(preds=preds_flat, horizon=horizon_df, name=name)


def _predict_window(
    session_state: dict,
    run_id: str,
    name: str,
    slice_window: Callable[[pd.DataFrame, int, str], pd.DataFrame],
) -> WindowPrediction:
    """Predict one window (early or late) with the run's trained model.

    Loads the checkpoint and template, pins inference to one GPU and hands
    the test frame to :func:`predict_window_frame`.
    """
    logging.info("Generating %s window predictions using existing model...", name)

    with single_gpu_env():
        import torch
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            teardown_distributed()

        model = load_tft_checkpoint(run_id)
        train_template = load_dataset_template(run_id)

        return predict_window_frame(
            model,
            train_template,
            session_state["test_data"],
            session_state["targets"],
            name,
            slice_window,
            target_offset=int(session_state.get("tft_target_offset", 0) or 0),
            loader_kwargs={"num_workers": get_default_num_workers()},
        )


def _predict_early_window(session_state: dict, run_id: str) -> WindowPrediction:
    """Predictions for the first encoder+decoder span of each trajectory."""
    return _predict_window(session_state, run_id, "early", _create_early_window_test_data)


def _predict_late_window(session_state: dict, run_id: str) -> WindowPrediction:
    """Predictions for the span ending at each trajectory's last step."""
    return _predict_window(session_state, run_id, "late", _create_late_window_test_data)


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


# The blend, for callers outside the test phase (the what-if view).
combine_windows = _combine_predictions_weighted


def predict_tft_two_window(
    session_state: Dict,
    run_id: str,
    *,
    skip_metrics: bool = False,
    metrics_filename: str = "performance.csv",
) -> np.ndarray:
    """Two-window TFT prediction: early and late windows, blended over the overlap.

    Trajectories shorter than ``encoder_length + prediction_length`` fit
    neither window and receive no prediction (module docstring, "Coverage
    note").

    *skip_metrics* / *metrics_filename* let callers evaluate non-test splits
    (e.g. train/val, for over/underfitting diagnostics) without overwriting
    the canonical test-set "performance.csv".
    """
    early_window = _predict_early_window(session_state, run_id)
    late_window = _predict_late_window(session_state, run_id)

    if late_window.horizon is None or len(late_window.horizon) == 0:
        raise ValueError("Late window predictions failed or returned empty results")

    time_idx_col = session_state.get("tft_time_idx_column", "Step")
    combined_window = _combine_predictions_weighted(
        early_window,
        late_window,
        session_state["targets"],
        time_idx_col,
    )

    test_data = session_state["test_data"]
    targets = session_state["targets"]
    all_test_trajectories = set(
        map(tuple, test_data[TRAJECTORY_COLS].drop_duplicates().itertuples(index=False, name=None))
    )
    combined_trajectories = set(
        map(tuple, combined_window.horizon[TRAJECTORY_COLS].drop_duplicates().itertuples(index=False, name=None))
    )
    logging.info(
        "Two-window coverage: %d/%d trajectories",
        len(combined_trajectories),
        len(all_test_trajectories),
    )
    n_missing = len(all_test_trajectories - combined_trajectories)
    if n_missing:
        logging.warning(
            "%d/%d test trajectories are shorter than the encoder+decoder window "
            "and receive no TFT prediction.",
            n_missing, len(all_test_trajectories),
        )

    final_horizon = combined_window.horizon
    final_preds = combined_window.preds

    # The window predictions are already absolute (denormalized in
    # _predict_window); the ground truth read from final_horizon has to match.
    from src.data.preprocess import denormalize_by_population, observed_mask_from_frame
    from configs.data import POPULATION_COLUMN
    session_state['horizon_df'] = final_horizon
    y_true_combined = denormalize_by_population(
        final_horizon[targets].values, final_horizon[POPULATION_COLUMN].values
    )
    session_state['horizon_y_true'] = y_true_combined
    session_state['early_predictions'] = early_window.preds
    session_state['late_predictions'] = late_window.preds

    if not skip_metrics:
        from src.trainers.evaluation import save_metrics
        save_metrics(
            run_id, y_true_combined, final_preds, final_horizon,
            observed_mask=observed_mask_from_frame(final_horizon, targets),
            metrics_filename=metrics_filename,
        )

    logging.info("Two-window prediction completed successfully!")
    return final_preds
