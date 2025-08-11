import os
import logging
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from sklearn.model_selection import ParameterSampler

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, RMSE, TimeSeriesDataSet
from pytorch_forecasting.metrics import MultiLoss

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
    best_params: dict,
):
    """Train final TFT on combined train+val and save checkpoint."""
    trainer_cfg = TFTTrainerConfig()

    final_dir = os.path.join(RESULTS_PATH, run_id, "final")
    os.makedirs(final_dir, exist_ok=True)
    final_ckpt_path = os.path.join(final_dir, "best.ckpt")

    # re-init model with best params; use train_dataset for normalization metadata
    n_targets = len(targets)
    if n_targets > 1:
        output_size = [1] * n_targets
        loss = MultiLoss([RMSE() for _ in range(n_targets)])
    else:
        output_size = 1
        loss = RMSE()

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

    # Build combined dataset from underlying dataframes if available
    train_df = getattr(train_dataset, "data", None)
    val_df = getattr(val_dataset, "data", None)
    if train_df is not None and val_df is not None:
        combined_df = pd.concat([train_df, val_df], axis=0, ignore_index=False)
        combined_dataset = TimeSeriesDataSet.from_dataset(train_dataset, combined_df)
        combined_loader = combined_dataset.to_dataloader(
            train=True,
            batch_size=trainer_cfg.batch_size,
            num_workers=_default_num_workers(),
            persistent_workers=True,
        )
    else:
        logging.warning("Could not access underlying dataframes from datasets; falling back to training on train_dataset only for final model.")
        combined_loader = train_dataset.to_dataloader(
            train=True,
            batch_size=trainer_cfg.batch_size,
            num_workers=_default_num_workers(),
            persistent_workers=True,
        )

    final_trainer = Trainer(
        max_epochs=trainer_cfg.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=trainer_cfg.devices,
        gradient_clip_val=trainer_cfg.gradient_clip_val,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,  # we'll save manually
    )

    final_trainer.fit(model=tft_final, train_dataloaders=combined_loader)
    final_trainer.save_checkpoint(final_ckpt_path)


def predict_tft(session_state: dict, run_id: str) -> np.ndarray:
    from src.trainers.evaluation import save_metrics

    test_data = session_state["test_data"]
    targets = session_state["targets"]

    model = _load_tft_checkpoint(run_id)

    logging.info("Building test dataset for TFT prediction...")

    # Use shared template logic to build test dataset
    train_dataset, dataset_config = _create_train_dataset(session_state)
    test_dataset = _from_train_template(
        train_dataset,
        test_data,
        mode="test",
    )

    trainer_cfg = TFTTrainerConfig()
    test_loader = test_dataset.to_dataloader(
        train=False,
        batch_size=trainer_cfg.batch_size,
        num_workers=_default_num_workers(),
        persistent_workers=True,
    )

    trainer = _create_infer_trainer()

    logging.info("Predicting with TFT model...")
    preds_tensor, index_df = model.predict(
        test_loader, trainer=trainer, return_index=True
    )

    preds_flat = _flatten_predictions(preds_tensor)
    idx_df = _normalize_index_df(index_df, dataset_config.time_idx)

    full_preds = _align_predictions(
        preds_flat,
        idx_df,
        test_data,
        dataset_config.group_ids,
        dataset_config.time_idx,
        len(targets),
    )

    # Save metrics on valid rows
    y_true = test_data[targets].values
    valid_mask = (~np.isnan(full_preds).any(axis=1)) & (~np.isnan(y_true).any(axis=1))
    if valid_mask.any():
        save_metrics(run_id, y_true[valid_mask], full_preds[valid_mask])
    else:
        logging.warning("No valid rows to compute metrics (all predictions or targets contain NaNs).")

    return full_preds


# ---------- Shared dataset builders ----------

# Build union-fitted NaNLabelEncoder encoders across all splits
from pytorch_forecasting.data.encoders import NaNLabelEncoder


def _build_union_encoders(session_state: dict, categorical_cols: List[str], add_nan: bool = False) -> Dict[str, Any]:
    dfs = [session_state.get("train_data"), session_state.get("val_data"), session_state.get("test_data")]
    df_all = pd.concat([df for df in dfs if df is not None], axis=0, ignore_index=True)
    encoders: Dict[str, Any] = {}
    for col in categorical_cols:
        if col in df_all.columns:
            s = df_all[col].astype(str).fillna("__NA__")
        else:
            # if column is entirely missing in some split, still create a closed-vocab encoder
            s = pd.Series(["__NA__"])  # minimal placeholder
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
    categorical_cols = list(set(INDEX_COLUMNS + CATEGORICAL_COLUMNS + [f for f in features if f.endswith("_is_missing")]))

    pretrained_categorical_encoders = _build_union_encoders(session_state, categorical_cols, add_nan=False)

    # inject encoders into dataset config
    dataset_config.pretrained_categorical_encoders = pretrained_categorical_encoders

    train_params = dataset_config.build(features, targets, mode="train")
    train_dataset = TimeSeriesDataSet(train_data, **train_params)
    return train_dataset, dataset_config


def _from_train_template(train_dataset, new_df, mode):
    """Build a dataset from a train template via TimeSeriesDataSet.from_dataset."""
    kwargs = {
        # "stop_randomization": True,
        "predict": mode == "test",  # only set predict if mode is 'test'
    }
    return TimeSeriesDataSet.from_dataset(train_dataset, new_df, **kwargs)


# ---------- Helpers for hyperparameter search ----------


def _init_tft_model(train_dataset, params: dict, n_targets: int) -> TemporalFusionTransformer:
    # For multi-target, pytorch-forecasting expects output_size as a list (one per target) and a MultiLoss
    if n_targets > 1:
        output_size = [1] * n_targets
        loss = MultiLoss([RMSE() for _ in range(n_targets)])
    else:
        output_size = 1
        loss = RMSE()

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
    trainer_cfg = TFTTrainerConfig()
    return Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=trainer_cfg.devices,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )


def _flatten_predictions(preds_tensor) -> np.ndarray:
    if hasattr(preds_tensor, "detach"):
        preds_np = preds_tensor.detach().cpu().numpy()
    else:
        preds_np = np.asarray(preds_tensor)

    # preds could be (N, pred_len, out_size) – flatten time dimension if present
    if preds_np.ndim == 3:
        n_samples, pred_len, out_size = preds_np.shape
        return preds_np.reshape(n_samples * pred_len, out_size)
    if preds_np.ndim == 2:
        return preds_np
    raise RuntimeError(f"Unexpected prediction shape: {preds_np.shape}")


def _normalize_index_df(index_df, time_idx_name: str) -> pd.DataFrame:
    idx_df = index_df.copy() if isinstance(index_df, pd.DataFrame) else pd.DataFrame(index_df)
    if "time_idx" in idx_df.columns and time_idx_name not in idx_df.columns:
        idx_df.rename(columns={"time_idx": time_idx_name}, inplace=True)
    return idx_df


def _align_predictions(
    preds_flat: np.ndarray,
    idx_df: pd.DataFrame,
    test_data: pd.DataFrame,
    group_cols: List[str],
    time_col: str,
    n_targets: int,
) -> np.ndarray:
    key_cols = group_cols + [time_col]
    test_keys = list(map(tuple, test_data[key_cols].itertuples(index=False, name=None)))
    pos_map = {key: i for i, key in enumerate(test_keys)}

    full_preds = np.full((len(test_data), n_targets), np.nan, dtype=float)

    pred_keys = list(map(tuple, idx_df[key_cols].itertuples(index=False, name=None)))
    for i, key in enumerate(pred_keys):
        pos = pos_map.get(key)
        if pos is not None:
            full_preds[pos, :] = preds_flat[i]
    return full_preds
