"""TFT model creation and training functions."""

import logging
import os
from typing import Dict, List, Optional

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, RMSE, TimeSeriesDataSet
from lightning.pytorch.core.module import LightningModule
from pytorch_forecasting.metrics import MultiLoss

from configs.paths import RESULTS_PATH
from configs.models.tft import TFTTrainerConfig
from .tft_utils import get_default_num_workers


def create_tft_model(
    train_dataset: TimeSeriesDataSet,
    params: Dict,
    n_targets: int
) -> LightningModule:
    """Create TFT model with given parameters."""
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


def create_dataloaders(
    train_dataset: TimeSeriesDataSet,
    val_dataset: TimeSeriesDataSet,
    batch_size: int
):
    """Create data loaders for training and validation."""
    train_loader = train_dataset.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=get_default_num_workers(),
        persistent_workers=True,
    )
    val_loader = val_dataset.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=get_default_num_workers(),
        persistent_workers=True,
    )
    return train_loader, val_loader


def create_trial_checkpoint(run_id: str, trial_idx: int) -> ModelCheckpoint:
    """Create checkpoint callback for hyperparameter search trial."""
    trial_dir = os.path.join(RESULTS_PATH, run_id, "search", f"trial_{trial_idx}")
    os.makedirs(trial_dir, exist_ok=True)
    
    return ModelCheckpoint(
        dirpath=trial_dir,
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )


def create_search_trainer(
    trainer_cfg: TFTTrainerConfig,
    checkpoint_callback: ModelCheckpoint
) -> Trainer:
    """Create trainer for hyperparameter search."""
    early_stop = EarlyStopping(monitor="val_loss", patience=trainer_cfg.patience, mode="min")
    
    return Trainer(
        max_epochs=trainer_cfg.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=trainer_cfg.devices,
        strategy="auto",
        gradient_clip_val=trainer_cfg.gradient_clip_val,
        callbacks=[early_stop, checkpoint_callback],
        logger=False,
        enable_progress_bar=False,
    )


def create_final_trainer(trainer_cfg: TFTTrainerConfig) -> Trainer:
    """Create trainer for final model training (legacy behavior).

    Restores original implementation: honor `trainer_cfg.devices` (default "auto") so
    Lightning can utilize multiple GPUs if available. Optional override:
      Set env FORCE_SINGLE_TFT_FINAL=1 to coerce a single device.
    """
    devices = trainer_cfg.devices
    import os as _os
    if _os.getenv("FORCE_SINGLE_TFT_FINAL"):
        if devices == "auto" or (isinstance(devices, int) and devices < 1):
            devices = 1
    logger = CSVLogger(save_dir=RESULTS_PATH, name="lightning_logs", version=None)
    return Trainer(
        max_epochs=trainer_cfg.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices,
        strategy="auto",
        gradient_clip_val=trainer_cfg.gradient_clip_val,
        logger=logger,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )


def load_tft_checkpoint(run_id: str) -> TemporalFusionTransformer:
    """Load TFT model from checkpoint."""
    final_dir = os.path.join(RESULTS_PATH, run_id, "final")
    final_ckpt_path = os.path.join(final_dir, "best.ckpt")
    
    if not os.path.exists(final_ckpt_path):
        raise FileNotFoundError(f"Final TFT checkpoint not found at {final_ckpt_path}")
    
    try:
        from pytorch_forecasting.models.base._base_model import Prediction as _PFPrediction
        import torch.serialization as _ts
        if hasattr(_ts, "add_safe_globals"):
            _ts.add_safe_globals([_PFPrediction])
    except Exception:
        pass
    
    try:
        model = TemporalFusionTransformer.load_from_checkpoint(final_ckpt_path)
        model.eval()
        logging.info("Loaded TFT model from %s", final_ckpt_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load TFT checkpoint from {final_ckpt_path}: {e}")