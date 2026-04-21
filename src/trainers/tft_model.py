"""TFT model creation and training functions."""

import logging
import os
from typing import Dict, List, Optional

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TemporalFusionTransformer, RMSE, TimeSeriesDataSet
from pytorch_forecasting.metrics import MultiLoss

from configs.models.tft import TFTTrainerConfig
from .tft_utils import get_default_num_workers
from src.utils.utils import get_run_root


def create_tft_model(
    train_dataset: TimeSeriesDataSet,
    params: Dict,
    n_targets: int,
    disable_lr_scheduler: bool = False,
) -> TemporalFusionTransformer:
    """Create TFT model with given parameters.

    Args:
        disable_lr_scheduler: Set True when no val_loss is available to
            monitor.  Sets an unreachably high patience so the
            ReduceLROnPlateau scheduler never fires.
    """
    if n_targets > 1:
        output_size = [1] * n_targets
        loss = MultiLoss([RMSE() for _ in range(n_targets)])
    else:
        output_size = 1
        loss = RMSE()

    kwargs = dict(
        hidden_size=params["hidden_size"],
        lstm_layers=params["lstm_layers"],
        dropout=params["dropout"],
        learning_rate=params["learning_rate"],
        output_size=output_size,
        loss=loss,
        log_interval=0,
    )
    if disable_lr_scheduler:
        # Setting patience to None causes pytorch-forecasting to return an
        # empty scheduler dict which Lightning rejects.  Instead, set an
        # unreachably high patience so the scheduler never fires.
        kwargs["reduce_on_plateau_patience"] = 999999

    return TemporalFusionTransformer.from_dataset(train_dataset, **kwargs)


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



def create_search_trainer(trainer_cfg: TFTTrainerConfig, log_dir: Optional[str] = None) -> Trainer:
    """Create trainer for hyperparameter search.

    Uses a single device because search runs trials sequentially —
    DDP subprocess launching would re-run the entire search loop
    on every spawned rank, causing recursive launches.
    """
    early_stop = EarlyStopping(monitor="val_loss", patience=trainer_cfg.patience, mode="min")

    logger = False
    if log_dir:
        logger = CSVLogger(save_dir=log_dir, name="", version="")

    return Trainer(
        max_epochs=trainer_cfg.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        strategy="auto",
        gradient_clip_val=trainer_cfg.gradient_clip_val,
        callbacks=[early_stop],
        logger=logger,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )


def create_final_trainer(
    trainer_cfg: TFTTrainerConfig,
    ckpt_path: str,
    log_dir: Optional[str] = None,
) -> Trainer:
    """Create trainer for final model training with early stopping on val_loss."""
    early_stop = EarlyStopping(
        monitor="val_loss", patience=trainer_cfg.final_patience, mode="min",
    )
    checkpoint = ModelCheckpoint(
        dirpath=os.path.dirname(ckpt_path),
        filename=os.path.splitext(os.path.basename(ckpt_path))[0],
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    logger = False
    if log_dir:
        logger = CSVLogger(save_dir=log_dir, name="", version="")

    # Use the devices configuration as-is for final training to allow multi-GPU usage
    # Only override if explicitly set to invalid values
    devices = trainer_cfg.devices
    if isinstance(devices, int) and devices < 1:
        devices = 1
    return Trainer(
        max_epochs=trainer_cfg.final_max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices,
        strategy="auto",
        gradient_clip_val=trainer_cfg.gradient_clip_val,
        callbacks=[early_stop, checkpoint],
        logger=logger,
        enable_progress_bar=False,
    )


def create_inference_trainer() -> Trainer:
    """Create single-device trainer for inference to preserve index ordering."""
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    return Trainer(
        accelerator=accelerator,
        devices=1,
        strategy="auto",
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )


def load_tft_checkpoint(run_id: str) -> TemporalFusionTransformer:
    """Load TFT model from checkpoint."""
    final_dir = os.path.join(get_run_root(run_id), "final")
    final_ckpt_path = os.path.join(final_dir, "best.ckpt")
    
    if not os.path.exists(final_ckpt_path):
        raise FileNotFoundError(f"Final TFT checkpoint not found at {final_ckpt_path}")
    
    try:
        model = TemporalFusionTransformer.load_from_checkpoint(
            final_ckpt_path, weights_only=False
        )
        model.eval()
        logging.info("Loaded TFT model from %s", final_ckpt_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load TFT checkpoint from {final_ckpt_path}: {e}")