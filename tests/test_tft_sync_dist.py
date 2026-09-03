"""TFT metrics must be reduced across DDP ranks.

pytorch_forecasting logs val_loss and its per-target SMAPE/MAE/RMSE/MAPE
without ``sync_dist``, so each rank kept its own copy of the value
EarlyStopping and ModelCheckpoint monitor, and Lightning warned once per
metric name -- 75 lines of a tft run's log.  SyncedTFT injects the flag
into every call on its way to Lightning.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("pytorch_forecasting")

from src.trainers.tft_model import MaskedTFT, SyncedTFT


@pytest.fixture
def logged(monkeypatch):
    """Record what reaches pytorch_forecasting's own log method."""
    calls = []
    monkeypatch.setattr(
        "pytorch_forecasting.models.base._base_model.BaseModel.log",
        lambda self, *args, **kwargs: calls.append((args, kwargs)),
    )
    return calls


def test_metrics_are_synced_across_ranks(logged):
    model = object.__new__(SyncedTFT)

    model.log("val_loss", 1.0, on_epoch=True)

    assert logged[0][1]["sync_dist"] is True


def test_an_explicit_choice_is_respected(logged):
    model = object.__new__(SyncedTFT)

    model.log("val_loss", 1.0, sync_dist=False)

    assert logged[0][1]["sync_dist"] is False


def test_the_masked_model_inherits_the_sync(logged):
    assert issubclass(MaskedTFT, SyncedTFT)

    model = object.__new__(MaskedTFT)
    model.log("Primary Energy|Coal val_SMAPE", 1.0)

    assert logged[0][1]["sync_dist"] is True
