"""Search bookkeeping that the trainers read back."""

import pytest

pytest.importorskip("lightning")

from lightning.pytorch.callbacks import EarlyStopping

from src.trainers.tft_trainer import _get_best_score


class _Trainer:
    def __init__(self, completed_epochs, callbacks):
        self.current_epoch = completed_epochs
        self.callbacks = callbacks


def test_the_best_epoch_is_reported_zero_based():
    """Four epochs ran (0-3); the best was two before the last: index 1."""
    import torch

    early_stop = EarlyStopping(monitor="val_loss")
    early_stop.wait_count = 2
    early_stop.best_score = torch.tensor(0.25)

    best_epoch, best_val_loss = _get_best_score(_Trainer(4, [early_stop]))

    assert (best_epoch, best_val_loss) == (1, 0.25)
