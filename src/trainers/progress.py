"""Per-epoch progress logging for the Lightning training loops.

Both final-training loops run with the progress bar off and their per-epoch
metrics going only to a CSVLogger under the run's ``final/logs``, which left
train.log silent for the whole of training -- ten and a half hours of it for
tft_93.  This callback puts a paced heartbeat in the run log itself.
"""

import logging
import time
from typing import Optional

from lightning.pytorch.callbacks import Callback

from src.utils.utils import format_duration, format_number


class EpochProgressLogger(Callback):
    """Log train/val loss once an epoch, no more often than *min_seconds*.

    The first and last epochs are always logged, so a short run still says
    something and a long one always records where it stopped.  Every line
    carries the running best, so an improvement that lands in a throttled-away
    epoch is still visible on the next line that survives.
    """

    def __init__(self, label: str, min_seconds: float = 60.0, monitor: str = "val_loss"):
        self.label = label
        self.min_seconds = min_seconds
        self.monitor = monitor
        self._best: Optional[float] = None
        self._best_epoch: Optional[int] = None
        self._started: Optional[float] = None
        self._epoch_started: Optional[float] = None
        self._last_logged: Optional[float] = None

    def on_train_start(self, trainer, pl_module):
        self._started = time.monotonic()

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_started = time.monotonic()

    def on_train_epoch_end(self, trainer, pl_module):
        """Runs after the epoch's validation loop, so both losses are in hand."""
        metrics = trainer.callback_metrics
        monitored = _as_float(metrics.get(self.monitor))
        if monitored is not None and (self._best is None or monitored < self._best):
            self._best = monitored
            self._best_epoch = trainer.current_epoch

        if not trainer.is_global_zero:
            return

        now = time.monotonic()
        # EarlyStopping sets should_stop during validation, which has already
        # run by now, so the stopping epoch is never the one thrown away.
        final_epoch = bool(getattr(trainer, "should_stop", False)) or (
            trainer.max_epochs is not None and trainer.current_epoch + 1 >= trainer.max_epochs
        )
        due = self._last_logged is None or now - self._last_logged >= self.min_seconds
        if not (due or final_epoch):
            return
        self._last_logged = now

        parts = [
            f"train_loss={format_number(_as_float(metrics.get('train_loss')))}",
            f"{self.monitor}={format_number(monitored)}",
        ]
        if self._best is not None:
            parts.append(f"best={format_number(self._best)} (epoch {self._best_epoch})")
        if self._epoch_started is not None:
            parts.append(f"epoch took {format_duration(now - self._epoch_started)}")
        if self._started is not None:
            parts.append(f"elapsed {format_duration(now - self._started)}")

        logging.info(
            "%s epoch %d/%s -> %s",
            self.label,
            trainer.current_epoch,
            trainer.max_epochs if trainer.max_epochs is not None else "?",
            ", ".join(parts),
        )


def _as_float(value) -> Optional[float]:
    """Unwrap a callback metric, which may be a tensor, a number, or missing."""
    if value is None:
        return None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return float(item())
        except (RuntimeError, ValueError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
