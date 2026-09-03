"""The per-epoch heartbeat both final-training loops log.

Before this callback, a TFT run logged nothing at all between the start of
training and its end ten hours later.
"""

import logging

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from lightning.pytorch import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from src.trainers.progress import EpochProgressLogger
from src.utils.utils import format_duration, format_number


class TinyModel(LightningModule):
    """Fits nothing; logs the losses the callback reads."""

    def __init__(self, val_losses):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
        self.val_losses = val_losses

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        (x,) = batch
        loss = self(x).mean()
        self.log("train_loss", torch.tensor(0.5), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        epoch = min(self.current_epoch, len(self.val_losses) - 1)
        self.log("val_loss", torch.tensor(self.val_losses[epoch]))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0)


def _run(val_losses, max_epochs, min_seconds, caplog):
    loader = DataLoader(TensorDataset(torch.zeros(2, 1)), batch_size=1)
    progress = EpochProgressLogger("Test training", min_seconds=min_seconds)
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[progress],
    )
    with caplog.at_level(logging.INFO):
        trainer.fit(TinyModel(val_losses), train_dataloaders=loader, val_dataloaders=loader)
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("Test training")]


def test_every_epoch_is_logged_when_nothing_is_throttled(caplog):
    lines = _run([0.5, 0.4, 0.3], max_epochs=3, min_seconds=0, caplog=caplog)

    assert len(lines) == 3
    assert "epoch 0/3" in lines[0]
    assert "train_loss=0.5000" in lines[0]
    assert "val_loss=0.5000" in lines[0]


def test_throttling_keeps_the_first_and_last_epoch(caplog):
    """A run whose epochs are far faster than min_seconds still bookends itself."""
    lines = _run([0.5, 0.4, 0.3, 0.2], max_epochs=4, min_seconds=3600, caplog=caplog)

    assert len(lines) == 2
    assert "epoch 0/4" in lines[0]
    assert "epoch 3/4" in lines[1]


def test_the_running_best_survives_a_throttled_epoch(caplog):
    """Epoch 1 is the best and is thrown away; the last line still reports it."""
    lines = _run([0.5, 0.1, 0.4, 0.9], max_epochs=4, min_seconds=3600, caplog=caplog)

    assert "best=0.1000 (epoch 1)" in lines[-1]


def test_numbers_are_formatted_for_both_loss_scales():
    assert format_number(0.0538) == "0.0538"
    assert format_number(20831.964844) == "20831.9648"
    assert format_number(29100416.7991) == "2.9100e+07"
    assert format_number(None) == "NA"
    assert format_number(float("nan")) == "nan"


def test_durations_use_the_unit_that_keeps_them_small():
    assert format_duration(45) == "45s"
    assert format_duration(660) == "11.0m"
    assert format_duration(37800) == "10.5h"
