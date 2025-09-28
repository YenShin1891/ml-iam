import os
import logging
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from sklearn.model_selection import ParameterSampler

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MultiLoss, QuantileLoss

from configs.config import RESULTS_PATH, CATEGORICAL_COLUMNS, INDEX_COLUMNS
from configs.models import TFTSearchSpace, TFTTrainerConfig

__all__ = [
    "build_datasets",
    "hyperparameter_search_tft",
    "train_final_tft",
    "predict_tft",
]

# Consistent num_workers utility
_def_num_workers = None

def _default_num_workers() -> int:
    global _def_num_workers
    if _def_num_workers is not None:
        return _def_num_workers
    # allow override via env var
    env_val = os.getenv("DL_NUM_WORKERS")
    if env_val is not None:
        try:
            _def_num_workers = max(1, int(env_val))
            return _def_num_workers
        except ValueError:
            pass
    try:
        cpu_count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    except Exception:
        cpu_count = os.cpu_count() or 2
    _def_num_workers = max(1, (cpu_count or 2) - 1)
    return _def_num_workers


def _teardown_dist():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            world_size = torch.distributed.get_world_size()
        except Exception:
            world_size = "unknown"
        logging.info(
            f"Destroying existing torch.distributed process group (world_size={world_size}) for single-process TFT prediction."
        )
        try:
            torch.distributed.destroy_process_group()
        except Exception as e:
            logging.warning(f"Failed to destroy process group cleanly: {e}")

_DIST_ENV_VARS = [
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "GLOBAL_RANK",
    "GROUP_RANK",
    "NODE_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
]

# New context manager to force single GPU / single process inference safely
from contextlib import contextmanager

@contextmanager
def _single_gpu_env():
    """Temporarily force single-GPU (device 0) inference.

    Stores and restores CUDA_VISIBLE_DEVICES and distributed env vars. This avoids
    potential DataLoader hangs or repeated loops caused by stale multi-process
    environment when calling predict on a single-process Trainer.
    """
    orig_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    orig_dist = {k: os.environ.get(k) for k in _DIST_ENV_VARS}
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        for k in _DIST_ENV_VARS:
            os.environ.pop(k, None)
        logging.info("Forced single GPU inference with CUDA_VISIBLE_DEVICES=0")
        yield
    finally:
        if orig_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = orig_cuda
        # restore dist vars
        for k, v in orig_dist.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        logging.info("Restored CUDA_VISIBLE_DEVICES and distributed environment variables after inference")


def build_datasets(session_state):
    """Build train/val TimeSeriesDataSet objects using shared template logic."""
    val_data = session_state["val_data"]

    train_dataset, _ = _create_train_dataset(session_state)
    val_dataset = _from_train_template(train_dataset, val_data, mode="eval")

    return train_dataset, val_dataset


def hyperparameter_search_tft(
    train_dataset,
    val_dataset,
    targets: List[str],
    run_id: str,
) -> dict:
    search_cfg = TFTSearchSpace()
    trainer_cfg = TFTTrainerConfig()

    search_results = []
    best_score = float("inf")
    best_params = None

    for i, params in enumerate(ParameterSampler(search_cfg.param_dist, n_iter=search_cfg.search_iter_n, random_state=0)):
        logging.info(f"TFT Search Iteration {i+1}/{search_cfg.search_iter_n} - Params: {params}")

        n_targets = len(targets)
        tft = _init_tft_model(train_dataset, params, n_targets)

        checkpoint_callback = _make_trial_checkpoint(run_id, i)
        trainer = _create_search_trainer(trainer_cfg, checkpoint_callback)

        train_loader, val_loader = _make_dataloaders(train_dataset, val_dataset, trainer_cfg.batch_size)
        # trainer.fit(model=tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    #     val_loss = trainer.callback_metrics["val_loss"].item()
    #     search_results.append({**params, "val_loss": val_loss})

    #     if val_loss < best_score:
    #         best_score = val_loss
        best_params = params

    # logging.info(f"Best TFT Params: {best_params} with Val Loss: {best_score:.4f}")
    return best_params


def train_final_tft(
    train_dataset,
    val_dataset,
    targets: List[str],
    run_id: str,
    best_params: dict,
    session_state: Optional[dict] = None,
):
    """Train final TFT on combined train+val and save checkpoint.

    If session_state is provided, will use session_state['train_data'] and
    session_state['val_data'] (pandas DataFrames) to build a combined dataset.
    Otherwise, will fall back to training on train_dataset only.
    """
    trainer_cfg = TFTTrainerConfig()

    final_dir = os.path.join(RESULTS_PATH, run_id, "final")
    os.makedirs(final_dir, exist_ok=True)
    final_ckpt_path = os.path.join(final_dir, "best.ckpt")
    dataset_tpl_path = os.path.join(final_dir, "dataset_template.pt")

    # re-init model with best params; use train_dataset for normalization metadata
    n_targets = len(targets)
    # Use QuantileLoss with 7 quantiles (default quantiles)
    quantile_loss = QuantileLoss()
    n_quantiles = 7
    if n_targets > 1:
        output_size = [n_quantiles] * n_targets
        loss = MultiLoss([quantile_loss for _ in range(n_targets)])
    else:
        output_size = n_quantiles
        loss = quantile_loss

    tft_final = TemporalFusionTransformer.from_dataset(
        train_dataset,
        hidden_size=best_params["hidden_size"],
        lstm_layers=best_params["lstm_layers"],
        dropout=best_params["dropout"],
        learning_rate=best_params["learning_rate"],
        output_size=output_size,
        loss=loss,
        log_interval=0,
    )

    # Prefer the original DataFrames from session_state
    train_df = None
    val_df = None
    if session_state is not None:
        train_df = session_state.get("train_data")
        val_df = session_state.get("val_data")

    assert train_df is not None or val_df is not None, "train_df or val_df is None; cannot build final dataset."
    
    if isinstance(train_df, pd.DataFrame) and isinstance(val_df, pd.DataFrame):
        # Ensure unique pandas index to satisfy TimeSeriesDataSet validation
        combined_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
        # Reuse train template to freeze feature schema/encoders identically across ranks
        combined_dataset = TimeSeriesDataSet.from_dataset(train_dataset, combined_df)
        # Persist the exact dataset template used for the final model to ensure consistent schema at inference
        try:
            torch.save(combined_dataset, dataset_tpl_path)
            logging.info("Saved TFT dataset template to %s", dataset_tpl_path)
        except Exception as e:
            raise RuntimeError(f"Failed to save dataset template to {dataset_tpl_path}: {e}")
        combined_loader = combined_dataset.to_dataloader(
            train=True,
            batch_size=trainer_cfg.batch_size,
            num_workers=_default_num_workers(),
            persistent_workers=True,
        )
    else:
        raise RuntimeError(
            "train_final_tft requires session_state to include both train_data and val_data as DataFrames to build a consistent dataset template."
        )

    final_trainer = Trainer(
        max_epochs=trainer_cfg.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=trainer_cfg.devices,
        strategy="auto",
        gradient_clip_val=trainer_cfg.gradient_clip_val,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,  # we'll save manually
    )

    final_trainer.fit(model=tft_final, train_dataloaders=combined_loader)
    final_trainer.save_checkpoint(final_ckpt_path)


def predict_tft(session_state: dict, run_id: str) -> np.ndarray:
    from src.trainers.evaluation import save_metrics, save_quantile_metrics

    test_data = session_state["test_data"]
    targets = session_state["targets"]

    # Wrap entire prediction path in single GPU context to avoid infinite loop / DDP leftovers
    with _single_gpu_env():
        # Best effort teardown if a process group is somehow still alive
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            _teardown_dist()
            if torch.distributed.is_initialized():  # pragma: no cover
                raise RuntimeError("Failed to teardown existing distributed process group before prediction.")

        model = _load_tft_checkpoint(run_id)

        logging.info("Building test dataset for TFT prediction using saved template...")
        final_dir = os.path.join(RESULTS_PATH, run_id, "final")
        dataset_tpl_path = os.path.join(final_dir, "dataset_template.pt")
        if not os.path.exists(dataset_tpl_path):
            raise FileNotFoundError(
                f"Dataset template not found at {dataset_tpl_path}. Run train_final_tft to generate it."
            )
        try:
            from pytorch_forecasting.data.timeseries._timeseries import TimeSeriesDataSet as _PFTimeSeriesDataSet  # type: ignore
        except Exception:  # pragma: no cover
            from pytorch_forecasting.data.timeseries import TimeSeriesDataSet as _PFTimeSeriesDataSet  # fallback
        try:
            import torch.serialization as _ts
            if hasattr(_ts, "add_safe_globals"):
                _ts.add_safe_globals([_PFTimeSeriesDataSet])
            try:
                train_template = torch.load(dataset_tpl_path, map_location="cpu")
            except Exception as inner_e:
                try:
                    train_template = torch.load(dataset_tpl_path, map_location="cpu", weights_only=False)
                    logging.info("Loaded dataset template with weights_only=False due to prior failure: %s", inner_e)
                except Exception as retry_e:
                    raise RuntimeError(
                        "Failed to load dataset template even after retry with weights_only=False. "
                        f"Original error: {inner_e}; Retry error: {retry_e}"
                    ) from retry_e
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset template from {dataset_tpl_path}: {e}")
        try:
            test_dataset = TimeSeriesDataSet.from_dataset(
                train_template,
                test_data,
                stop_randomization=True,
                predict=True,  # only decoder (forecast horizon) timesteps retained
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
        logging.info("Loaded dataset template from %s for prediction.", dataset_tpl_path)

        trainer_cfg = TFTTrainerConfig()
        test_loader = test_dataset.to_dataloader(
            train=False,
            batch_size=trainer_cfg.batch_size,
            num_workers=_default_num_workers(),
            persistent_workers=True,
        )
        trainer = _create_infer_trainer()

        logging.info("Predicting with TFT model (forecast horizon only)...")
        returns = model.predict(test_loader, return_index=True)

        from pytorch_forecasting.models.base._base_model import Prediction as _PFPrediction  # type: ignore
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

        idx_df = _normalize_index_df(index_df, time_idx_name)

        # If predictions are 3D (n_samples, pred_len, out_size) but index_df only has n_samples rows,
        # expand the index so each horizon step has its own row with incremented time index.
        if torch.is_tensor(preds_tensor) and preds_tensor.ndim == 3:
            n_samples, pred_len, out_size = preds_tensor.shape
            if len(idx_df) == n_samples and pred_len > 1:
                logging.info(
                    "Expanding index_df for multi-step horizon: samples=%d, pred_len=%d", n_samples, pred_len
                )
                # Build expanded index
                expanded_rows = []
                base_cols = idx_df.columns.tolist()
                if time_idx_name not in base_cols:
                    raise KeyError(
                        f"Time index column '{time_idx_name}' not found in prediction index DataFrame columns: {base_cols}"
                    )
                for i in range(n_samples):
                    base_row = idx_df.iloc[i]
                    base_time = base_row[time_idx_name]
                    for h in range(pred_len):
                        new_row = base_row.copy()
                        # Assumption: decoder steps are consecutive increments
                        new_row[time_idx_name] = base_time + h
                        expanded_rows.append(new_row)
                idx_df = pd.DataFrame(expanded_rows).reset_index(drop=True)
                # Flatten predictions accordingly
                preds_flat = preds_tensor.detach().cpu().numpy().reshape(n_samples * pred_len, out_size)
            else:
                preds_flat = _flatten_predictions(preds_tensor)
        else:
            preds_flat = _flatten_predictions(preds_tensor)

        # Evaluate only the forecast horizon rows returned by predict=True.
        key_cols = group_ids + [time_idx_name]
        # Collect reference columns (ensure presence in test_data)
        ref_cols = [c for c in key_cols + ['Year'] + targets if c in test_data.columns]
        horizon_df = idx_df[key_cols].merge(
            test_data[ref_cols].drop_duplicates(key_cols),
            on=key_cols,
            how='left'
        )
        if preds_flat.shape[0] != len(horizon_df):
            logging.error(
                "After expansion attempt: preds_flat rows=%d, horizon_df rows=%d. First few time_idx in idx_df: %s",
                preds_flat.shape[0], len(horizon_df), idx_df[time_idx_name].head().tolist()
            )
            raise RuntimeError(
                f"Prediction rows ({preds_flat.shape[0]}) != horizon_df rows ({len(horizon_df)})."
            )

        y_true = horizon_df[targets].values
        y_pred_quantiles = preds_flat  # Full quantile predictions (7 quantiles per target)

        # Extract median (50th percentile) predictions for point prediction metrics
        # With 7 quantiles [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98], median is at index 3
        n_targets = len(targets)
        n_quantiles = 7
        median_idx = 3  # 0.5 quantile

        if n_targets > 1:
            # Multi-target: shape is [n_samples, n_targets * n_quantiles]
            # Reshape to [n_samples, n_targets, n_quantiles] and extract median
            y_pred_reshaped = y_pred_quantiles.reshape(-1, n_targets, n_quantiles)
            y_pred_median = y_pred_reshaped[:, :, median_idx]  # [n_samples, n_targets]
        else:
            # Single target: shape is [n_samples, n_quantiles]
            y_pred_median = y_pred_quantiles[:, median_idx:median_idx+1]  # [n_samples, 1]

        valid_mask = (~np.isnan(y_true).any(axis=1)) & (~np.isnan(y_pred_median).any(axis=1))
        if valid_mask.any():
            save_metrics(run_id, y_true[valid_mask], y_pred_median[valid_mask])

            # Also save quantile-specific metrics
            if n_targets > 1:
                # Reshape quantile predictions for multi-target case
                y_pred_quantiles_valid = y_pred_quantiles[valid_mask].reshape(-1, n_targets, n_quantiles)
            else:
                y_pred_quantiles_valid = y_pred_quantiles[valid_mask]

            save_quantile_metrics(run_id, y_true[valid_mask], y_pred_quantiles_valid)
        else:
            logging.warning("No valid rows to compute metrics (all horizon targets or predictions contain NaNs).")

        # Expose horizon dataframe, y_true, and both median and full quantile predictions for downstream use
        session_state['horizon_df'] = horizon_df
        session_state['horizon_y_true'] = y_true
        session_state['horizon_y_pred_median'] = y_pred_median
        session_state['horizon_y_pred_quantiles'] = y_pred_quantiles

        # Return the median predictions for backward compatibility
        return y_pred_median


# ---------- Shared dataset builders ----------

# Build union-fitted NaNLabelEncoder encoders across all splits
from pytorch_forecasting.data.encoders import NaNLabelEncoder


def _ordered_categorical_cols(features: List[str]) -> List[str]:
    """Deterministic column order for encoders: INDEX + CATEGORICALS + indicator cols (if any)."""
    static_cols = list(INDEX_COLUMNS) + list(CATEGORICAL_COLUMNS)
    indicator_cols = [f for f in features if f.endswith("_is_missing")]
    # preserve order, drop duplicates
    ordered = list(dict.fromkeys(static_cols + indicator_cols))
    return ordered


def _build_union_encoders(session_state: dict, categorical_cols: List[str], add_nan: bool = False) -> Dict[str, Any]:
    dfs = [session_state.get("train_data"), session_state.get("val_data"), session_state.get("test_data")]
    df_all = pd.concat([df for df in dfs if df is not None], axis=0, ignore_index=True)
    encoders: Dict[str, Any] = {}
    # ensure deterministic iteration order
    for col in categorical_cols:
        if col in df_all.columns:
            s_raw = df_all[col].astype(str).fillna("__NA__")
            # Explicit, deterministic category order
            categories = sorted(pd.unique(s_raw))
            s = pd.Series(pd.Categorical(s_raw, categories=categories, ordered=True))
        else:
            # if column is entirely missing in some split, still create a closed-vocab encoder
            s = pd.Series(pd.Categorical(["__NA__"], categories=["__NA__"], ordered=True))
        enc = NaNLabelEncoder(add_nan=add_nan)
        enc.fit(s)
        encoders[col] = enc
    return encoders


def _create_train_dataset(session_state):
    """Create train TimeSeriesDataSet from session_state using TFTDatasetConfig."""
    from configs.models import TFTDatasetConfig
    dataset_config = TFTDatasetConfig()

    train_data = session_state["train_data"]
    features = session_state["features"]
    targets = session_state["targets"]

    # Determine columns needing fixed vocabularies: include group_ids and all configured categoricals
    categorical_cols = _ordered_categorical_cols(features)

    pretrained_categorical_encoders = _build_union_encoders(session_state, categorical_cols, add_nan=False)

    # inject encoders into dataset config
    dataset_config.pretrained_categorical_encoders = pretrained_categorical_encoders

    train_params = dataset_config.build(features, targets, mode="train")
    train_dataset = TimeSeriesDataSet(train_data, **train_params)
    return train_dataset, dataset_config


def _from_train_template(train_dataset, new_df, mode):
    """Build a dataset from a train template via TimeSeriesDataSet.from_dataset."""
    kwargs = {}
    if mode in {"eval", "test"}:
        kwargs["stop_randomization"] = True
    if mode == "test":
        # Ensure prediction indices (group_ids + time_idx) are returned by model.predict(..., return_index=True)
        kwargs["predict"] = True
    return TimeSeriesDataSet.from_dataset(train_dataset, new_df, **kwargs)


# ---------- Helpers for hyperparameter search ----------


def _init_tft_model(train_dataset, params: dict, n_targets: int) -> TemporalFusionTransformer:
    # Use QuantileLoss with 7 quantiles (default quantiles)
    quantile_loss = QuantileLoss()
    n_quantiles = 7
    if n_targets > 1:
        output_size = [n_quantiles] * n_targets
        loss = MultiLoss([quantile_loss for _ in range(n_targets)])
    else:
        output_size = n_quantiles
        loss = quantile_loss

    return TemporalFusionTransformer.from_dataset(
        train_dataset,
        hidden_size=params["hidden_size"],
        lstm_layers=params["lstm_layers"],
        dropout=params["dropout"],
        learning_rate=params["learning_rate"],
        output_size=output_size,
        loss=loss,
        log_interval=0,
    )


def _make_dataloaders(train_dataset, val_dataset, batch_size: int):
    num_workers = _default_num_workers()
    train_loader = train_dataset.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_loader = val_dataset.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
    )
    return train_loader, val_loader


def _make_trial_checkpoint(run_id: str, trial_idx: int) -> ModelCheckpoint:
    trial_ckpt_dir = os.path.join(RESULTS_PATH, run_id, "checkpoints", f"trial_{trial_idx:03d}")
    os.makedirs(trial_ckpt_dir, exist_ok=True)
    return ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        dirpath=trial_ckpt_dir,
        filename="best_model_{epoch:02d}-{val_loss:.4f}",
    )


def _create_search_trainer(trainer_cfg: TFTTrainerConfig, checkpoint_cb: ModelCheckpoint) -> Trainer:
    return Trainer(
        max_epochs=trainer_cfg.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=trainer_cfg.devices,
        strategy="auto",
        gradient_clip_val=trainer_cfg.gradient_clip_val,
        callbacks=[EarlyStopping(monitor="val_loss", patience=trainer_cfg.patience), checkpoint_cb],
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=True,
    )


# ---------- Helpers for prediction ----------

def _load_tft_checkpoint(run_id: str) -> TemporalFusionTransformer:
    final_ckpt_path = os.path.join(RESULTS_PATH, run_id, "final", "best.ckpt")
    if not os.path.exists(final_ckpt_path):
        raise FileNotFoundError(
            f"Final checkpoint not found at {final_ckpt_path}. Please run training first."
        )
    return TemporalFusionTransformer.load_from_checkpoint(final_ckpt_path)


def _create_infer_trainer() -> Trainer:
    """Create inference Trainer on a single device to preserve index info in outputs.

    Multi-GPU DDP prediction was dropping/fragmenting index DataFrames. Inference is
    lightweight, so we force devices=1 to ensure model.predict(return_index=True)
    returns a single (preds, index_df) tuple.
    """
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    return Trainer(
        accelerator=accelerator,
        devices=1,  # force single device for stable prediction structure
        strategy="auto",
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )


def _flatten_predictions(preds_tensor) -> np.ndarray:
    """Convert various prediction container types to a 2D numpy array on CPU.

    Handles:
    - torch.Tensor on GPU/CPU
    - list/tuple of tensors/arrays (concatenates along batch axis)
    - dicts with common prediction keys
    """
    def _to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # Unpack common dict structures
    if isinstance(preds_tensor, dict):
        for key in ["prediction", "predictions", "output", "y", "y_hat"]:
            if key in preds_tensor:
                return _flatten_predictions(preds_tensor[key])
        # fallback to first value
        if len(preds_tensor):
            return _flatten_predictions(next(iter(preds_tensor.values())))

    # Concatenate sequences
    if isinstance(preds_tensor, (list, tuple)):
        parts = []
        for p in preds_tensor:
            if p is None:
                continue
            if isinstance(p, (list, tuple, dict)):
                p = _flatten_predictions(p)
            else:
                p = _to_numpy(p)
            parts.append(p)
        if not parts:
            raise RuntimeError("Empty predictions sequence")
        try:
            preds_np = np.concatenate(parts, axis=0)
        except Exception:
            # last resort, stack as objects then try squeeze
            preds_np = np.array(parts, dtype=object)
    elif torch.is_tensor(preds_tensor):
        preds_np = preds_tensor.detach().cpu().numpy()
    else:
        preds_np = np.asarray(preds_tensor)

    # Normalize shapes to (N, out_size)
    if preds_np.ndim == 3:
        n_samples, pred_len, out_size = preds_np.shape
        return preds_np.reshape(n_samples * pred_len, out_size)
    if preds_np.ndim == 2:
        return preds_np
    if preds_np.ndim == 1:
        return preds_np.reshape(-1, 1)
    raise RuntimeError(f"Unexpected prediction shape: {preds_np.shape}")


def _normalize_index_df(index_df, time_idx_name: str) -> pd.DataFrame:
    idx_df = index_df.copy() if isinstance(index_df, pd.DataFrame) else pd.DataFrame(index_df)
    if idx_df is None or not isinstance(idx_df, pd.DataFrame) or idx_df.empty:
        raise ValueError(
            "Prediction returned no index information. Ensure model.predict(..., return_index=True) and dataset template alignment."
        )
    if "time_idx" in idx_df.columns and time_idx_name not in idx_df.columns:
        idx_df.rename(columns={"time_idx": time_idx_name}, inplace=True)
    return idx_df


def _aggregate_prediction_outputs(returns):
    """Robustly extract (preds, index_df) from returns produced by model.predict.

    Handles possibilities:
    - Single tuple: (pred_tensor, index_df)
    - List of tuples per batch
    - List of tensors (no index) -> will raise later
    - Dict or list of dicts containing 'prediction' / 'predictions' and optional 'index'
    Filters out empty / zero-dim entries.
    """
    import pandas as _pd
    import torch as _torch
    pred_parts = []
    index_parts = []

    def _as_tensor(x):
        if _torch.is_tensor(x):
            return x
        arr = np.asarray(x)
        if arr.ndim == 0:
            arr = arr.reshape(-1)
        return _torch.as_tensor(arr)

    # Unpack common dict structures
    if isinstance(returns, dict):
        for key in ["prediction", "predictions", "output", "y", "y_hat"]:
            if key in returns:
                return _aggregate_prediction_outputs(returns[key])
        # fallback to first value
        if len(returns):
            return _aggregate_prediction_outputs(next(iter(returns.values())))

    # Concatenate sequences
    if isinstance(returns, (list, tuple)):
        for r in returns:
            if r is None:
                continue
            if isinstance(r, (list, tuple, dict)):
                _aggregate_prediction_outputs(r)
            else:
                pred_parts.append(_as_tensor(r))
    elif _torch.is_tensor(returns):
        pred_parts.append(_as_tensor(returns))
    else:
        raise RuntimeError(f"Unexpected prediction return type: {type(returns)}")

    if not pred_parts:
        raise RuntimeError("No valid prediction parts found in returns.")

    # Attempt to stack/concatenate predictions
    try:
        if all(p.ndim == 1 for p in pred_parts):
            # 1D case: stack
            return _torch.stack(pred_parts, dim=0)
        if all(p.ndim == 2 for p in pred_parts):
            # 2D case: concatenate along batch dimension
            return _torch.cat(pred_parts, dim=0)
        if all(p.ndim == 3 for p in pred_parts):
            # 3D case: concatenate along batch dimension (B, T, C) -> (B*T, C)
            return _torch.cat(pred_parts, dim=0).view(-1, pred_parts[0].size(-1))
    except Exception as e:
        raise RuntimeError(f"Failed to stack/concatenate prediction parts: {e}")

    raise RuntimeError(
        "Inconsistent prediction part dimensions; unable to stack/concatenate. "
        f"Sample shapes: {[p.shape for p in pred_parts]}"
    )
