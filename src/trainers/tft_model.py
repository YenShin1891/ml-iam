"""TFT model creation and training functions."""

import io
import logging
import os
from typing import Dict, Optional

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TemporalFusionTransformer, RMSE, TimeSeriesDataSet
from pytorch_forecasting.metrics import MultiLoss

from configs.models.tft import TFTTrainerConfig
from .progress import EpochProgressLogger
from .tft_utils import get_default_num_workers
from src.utils.utils import get_run_root


class MaskedRMSE(RMSE):
    """RMSE over the observed target elements only.

    Before each loss computation the parent model sets ``_obs_mask`` to a
    ``(batch, time)`` float tensor (1 = observed, 0 = zero-filled or
    interpolated).  Unobserved elements contribute neither squared error nor
    count, so the value is the root of the mean over observed elements.  With
    no mask set the metric is the standard RMSE.
    """

    def __init__(self):
        super().__init__()
        self._obs_mask: Optional[torch.Tensor] = None

    def _update_losses_and_lengths(self, losses: torch.Tensor, lengths: torch.Tensor):
        mask, self._obs_mask = self._obs_mask, None  # consume
        if mask is None:
            return super()._update_losses_and_lengths(losses, lengths)

        mask = mask.to(losses.dtype)
        if mask.shape[:2] != losses.shape[:2]:
            raise ValueError(
                f"Observed mask {tuple(mask.shape)} does not line up with the "
                f"losses {tuple(losses.shape)}"
            )
        # Steps past a sample's decoder length are padding, not observations.
        steps = torch.arange(losses.size(1), device=losses.device).unsqueeze(0)
        mask = mask * (steps < lengths.to(losses.device).unsqueeze(-1)).to(losses.dtype)
        while mask.dim() < losses.dim():
            mask = mask.unsqueeze(-1)

        total = (losses * mask).sum()
        if not torch.isfinite(total):
            total = torch.tensor(1e9, device=losses.device)
        self.losses = self.losses + total
        self.lengths = self.lengths + mask.sum().round().long()

    def reduce_loss(self, losses, lengths, reduction=None):
        # A batch can hold no observed element of one target.  0/0 would be
        # NaN, and sqrt(0) has an infinite derivative, so return a zero that
        # is still attached to the graph.
        if (reduction or self.reduction) != "none" and int(lengths.sum()) == 0:
            return losses.sum() * 0.0
        return super().reduce_loss(losses, lengths, reduction=reduction)


class SyncedTFT(TemporalFusionTransformer):
    """TFT whose logged metrics are reduced across DDP ranks.

    pytorch_forecasting logs val_loss and its per-target metrics without
    ``sync_dist``, so under DDP each rank kept its own value of the metric
    EarlyStopping and ModelCheckpoint monitor, and Lightning warned about it
    once per metric name -- 75 lines a run.
    """

    def log(self, *args, **kwargs):
        kwargs.setdefault("sync_dist", True)
        super().log(*args, **kwargs)


class MaskedTFT(SyncedTFT):
    """Thin wrapper that injects per-target observed masks into MaskedRMSE
    metrics before each loss computation.

    Expects ``__observed`` columns to be present in ``time_varying_known_reals``
    so they appear in ``x["decoder_cont"]``.  Whether the network also reads
    them is decided when the model is created (see ``create_tft_model``).
    """

    def _set_obs_masks(self, x):
        """Extract per-target observed masks from decoder_cont and push them
        into the corresponding MaskedRMSE instances.

        Raises when a mask cannot be found: a masked loss that silently runs
        unmasked trains on zero-filled targets and reports nothing.
        """
        loss = self.loss
        if isinstance(loss, MultiLoss):
            metrics = loss.metrics
        else:
            metrics = [loss]

        from src.data.preprocess import observed_mask_columns
        # The targets live in the dataset parameters; pytorch-forecasting does
        # not copy them into hparams.
        obs_col_names = observed_mask_columns(self.target_names)

        # x_reals names the columns of the continuous tensors, in order
        x_reals = list(self.hparams.x_reals)
        decoder_cont = x["decoder_cont"]  # (batch, time, n_reals)

        for i, metric in enumerate(metrics):
            if not isinstance(metric, MaskedRMSE):
                continue
            if i >= len(obs_col_names) or obs_col_names[i] not in x_reals:
                raise RuntimeError(
                    f"No observed-mask column for target {i} "
                    f"({obs_col_names[i] if i < len(obs_col_names) else 'unnamed'}) "
                    "among the dataset's reals; the masked loss cannot be applied."
                )
            metric._obs_mask = decoder_cont[:, :, x_reals.index(obs_col_names[i])]

    def training_step(self, batch, batch_idx):
        x, y = batch
        self._set_obs_masks(x)
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self._set_obs_masks(x)
        return super().validation_step(batch, batch_idx)


def create_tft_model(
    train_dataset: TimeSeriesDataSet,
    params: Dict,
    n_targets: int,
    disable_lr_scheduler: bool = False,
) -> MaskedTFT:
    """Create TFT model with given parameters.

    Args:
        disable_lr_scheduler: Set True when no val_loss is available to
            monitor.  Sets an unreachably high patience so the
            ReduceLROnPlateau scheduler never fires.
    """
    from src.data.preprocess import observed_mask_columns

    # Masked exactly when the dataset carries a mask column for every target.
    # Read off the dataset rather than configs.data.KEEP_PARTIAL_TARGETS: a
    # spawned search worker re-imports the config and would not see a run's
    # override, while the dataset it is handed cannot disagree with itself.
    flag_columns = observed_mask_columns(train_dataset.target_names)
    masked = all(name in train_dataset.reals for name in flag_columns)

    if n_targets > 1:
        output_size = [1] * n_targets
        loss = MultiLoss([(MaskedRMSE if masked else RMSE)() for _ in range(n_targets)])
    else:
        output_size = 1
        loss = MaskedRMSE() if masked else RMSE()

    model_cls = MaskedTFT if masked else SyncedTFT

    kwargs = dict(
        hidden_size=params["hidden_size"],
        lstm_layers=params["lstm_layers"],
        dropout=params["dropout"],
        learning_rate=params["learning_rate"],
        output_size=output_size,
        loss=loss,
        log_interval=0,
        causal_attention=True,
    )
    if disable_lr_scheduler:
        # Setting patience to None causes pytorch-forecasting to return an
        # empty scheduler dict which Lightning rejects.  Instead, set an
        # unreachably high patience so the scheduler never fires.
        kwargs["reduce_on_plateau_patience"] = 999999

    from configs.data import TFT_OBSERVED_FLAGS_AS_INPUTS
    if masked and not TFT_OBSERVED_FLAGS_AS_INPUTS:
        # The flags stay in the dataset, so every batch still carries them
        # for the loss mask, but they are left out of the variables the
        # network selects from.  from_dataset would otherwise list every
        # known real as an encoder and decoder input.
        flags = set(flag_columns)
        inputs = [name for name in train_dataset._time_varying_known_reals if name not in flags]
        kwargs["time_varying_reals_encoder"] = inputs + list(train_dataset._time_varying_unknown_reals)
        kwargs["time_varying_reals_decoder"] = inputs

    return model_cls.from_dataset(train_dataset, **kwargs)


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
        # A re-run train phase must overwrite best.ckpt: with the version
        # counter on, Lightning writes best-v1.ckpt instead and the test
        # phase keeps loading the stale weights.
        enable_version_counter=False,
    )

    progress = EpochProgressLogger("TFT final training")

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
        callbacks=[early_stop, checkpoint, progress],
        logger=logger,
        enable_progress_bar=False,
    )


def metrics_to(module: torch.nn.Module, device) -> torch.nn.Module:
    """Point every torchmetrics metric inside *module* at *device*.

    A metric pickled into a checkpoint remembers the device it trained on.
    Lightning's ``model.to(map_location)`` then asks that device for a
    tensor, which a CPU-only torch cannot supply; the state itself is
    already wherever ``torch.load`` mapped it.  pytorch-forecasting's
    MultiLoss keeps its metrics in a plain list, so they are walked too.
    """
    target = torch.device(device)
    pending, seen = [module], set()
    while pending:
        obj = pending.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if hasattr(obj, "_device"):
            obj._device = target
        if isinstance(obj, torch.nn.Module):
            pending.extend(obj.children())
            pending.extend(
                m for m in (getattr(obj, "metrics", None) or []) if isinstance(m, torch.nn.Module)
            )
    return module


def _checkpoint_on(ckpt_path: str, map_location) -> io.BytesIO:
    """The checkpoint with its pickled metrics pointed at *map_location*, as bytes.

    The hyperparameters travel nested under one name, so Lightning's keyword
    overrides cannot replace the pickled loss; repairing the checkpoint
    itself can.
    """
    checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    # pytorch-forecasting also stores the loss under its own key and sets it
    # on the model after construction, so both copies are walked.
    for key in ("hyper_parameters", TemporalFusionTransformer.CHECKPOINT_HYPER_PARAMS_SPECIAL_KEY):
        for value in (checkpoint.get(key) or {}).values():
            if isinstance(value, torch.nn.Module):
                metrics_to(value, map_location)
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    return buffer


def load_tft_checkpoint(run_id: str, map_location=None) -> TemporalFusionTransformer:
    """Load TFT model from checkpoint.

    *map_location* goes to Lightning: "cpu" keeps a model trained on a GPU
    off the accelerator, for a process that must not claim one (the
    dashboard's what-if view).  By default the checkpoint's own devices are
    used where available.
    """
    final_dir = os.path.join(get_run_root(run_id), "final")
    final_ckpt_path = os.path.join(final_dir, "best.ckpt")

    if not os.path.exists(final_ckpt_path):
        raise FileNotFoundError(f"Final TFT checkpoint not found at {final_ckpt_path}")

    from configs.data import KEEP_PARTIAL_TARGETS
    model_cls = MaskedTFT if KEEP_PARTIAL_TARGETS else SyncedTFT

    try:
        source = final_ckpt_path
        if isinstance(map_location, (str, torch.device)):
            source = _checkpoint_on(final_ckpt_path, map_location)
        model = model_cls.load_from_checkpoint(
            source, map_location=map_location, weights_only=False
        )
        model.eval()
        logging.info("Loaded TFT model from %s", final_ckpt_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load TFT checkpoint from {final_ckpt_path}: {e}")