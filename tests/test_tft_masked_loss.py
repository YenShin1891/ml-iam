"""The TFT's loss must skip unobserved targets, and the network must not read the flags."""

import numpy as np
import pandas as pd
import pytest
import torch
from pytorch_forecasting import RMSE, TimeSeriesDataSet
from pytorch_forecasting.metrics import MultiLoss
from torch.nn.utils import rnn

import configs.data as data_config
from src.trainers.tft_model import MaskedRMSE, create_tft_model

TARGETS = ["A", "B"]
PARAMS = {"hidden_size": 8, "lstm_layers": 1, "dropout": 0.0, "learning_rate": 1e-3}


def _dataset():
    rng = np.random.default_rng(0)
    rows = []
    for g in range(6):
        for step in range(8):
            rows.append({
                "Model": f"M{g}", "Scenario": "S", "Region": "World", "Step": step,
                "feat": rng.normal(), "A": rng.normal(), "B": rng.normal(),
                "A__observed": 1.0, "B__observed": 0.0 if g % 2 else 1.0,
            })
    frame = pd.DataFrame(rows)
    frame.loc[frame["B__observed"] == 0, "B"] = 0.0  # zero-filled, as the pipeline does
    return TimeSeriesDataSet(
        frame, time_idx="Step", target=TARGETS, group_ids=["Model", "Scenario", "Region"],
        max_encoder_length=2, min_encoder_length=2, max_prediction_length=4, min_prediction_length=1,
        time_varying_known_reals=["feat", "A__observed", "B__observed"],
        scalers={"A__observed": None, "B__observed": None},
    )


@pytest.fixture
def batch_and_model(monkeypatch):
    monkeypatch.setattr(data_config, "KEEP_PARTIAL_TARGETS", True)
    monkeypatch.setattr(data_config, "TFT_OBSERVED_FLAGS_AS_INPUTS", False)
    dataset = _dataset()
    model = create_tft_model(dataset, PARAMS, n_targets=2)
    x, y = next(iter(dataset.to_dataloader(train=False, batch_size=64, num_workers=0)))
    return x, y, model


def _packed(x, y):
    return ([rnn.pack_padded_sequence(t, lengths=x["decoder_lengths"], batch_first=True, enforce_sorted=False)
             for t in y[0]], None)


def test_every_metric_receives_its_mask(batch_and_model):
    x, _, model = batch_and_model

    model._set_obs_masks(x)

    masks = [metric._obs_mask for metric in model.loss.metrics]
    assert all(mask is not None for mask in masks)
    reals = list(model.hparams.x_reals)
    for target, mask in zip(TARGETS, masks):
        assert torch.equal(mask, x["decoder_cont"][:, :, reals.index(f"{target}__observed")])
    # A is observed wherever the decoder is not padding; B is not.
    valid = torch.arange(masks[0].shape[1]).unsqueeze(0) < x["decoder_lengths"].unsqueeze(-1)
    assert masks[0][valid].min() == 1 and masks[1][valid].min() == 0


def test_the_loss_is_the_rmse_over_observed_elements(batch_and_model):
    x, y, model = batch_and_model
    model.eval()
    with torch.no_grad():
        prediction = model(x)["prediction"]
        model._set_obs_masks(x)
        loss = float(model.loss(prediction, _packed(x, y)))

    steps = torch.arange(prediction[0].shape[1]).unsqueeze(0)
    valid = steps < x["decoder_lengths"].unsqueeze(-1)
    reals = list(model.hparams.x_reals)
    expected = 0.0
    for i, target in enumerate(TARGETS):
        observed = valid & (x["decoder_cont"][:, :, reals.index(f"{target}__observed")] > 0.5)
        error = (prediction[i].squeeze(-1) - y[0][i])[observed]
        expected += float(error.pow(2).mean().sqrt())
    assert loss == pytest.approx(expected, rel=1e-5)

    unmasked = float(MultiLoss([RMSE(), RMSE()])(prediction, _packed(x, y)))
    assert loss != pytest.approx(unmasked, rel=1e-6)


def test_a_target_with_nothing_observed_gives_a_finite_gradient():
    metric = MaskedRMSE()
    prediction = torch.ones(3, 4, 1, requires_grad=True)
    metric._obs_mask = torch.zeros(3, 4)

    loss = metric(prediction, torch.zeros(3, 4))
    loss.backward()

    assert float(loss) == 0.0
    assert torch.isfinite(prediction.grad).all()


def test_a_missing_mask_column_is_an_error(batch_and_model):
    x, _, model = batch_and_model
    model.hparams.x_reals = [name for name in model.hparams.x_reals if name != "B__observed"]

    with pytest.raises(RuntimeError, match="masked loss cannot be applied"):
        model._set_obs_masks(x)


def test_the_network_does_not_read_the_flags(batch_and_model):
    x, _, model = batch_and_model
    assert not [name for name in model.reals if name.endswith("__observed")]
    assert "feat" in model.reals
    assert "B__observed" in model.hparams.x_reals  # still in the batch, for the mask

    flipped = dict(x)
    reals = list(model.hparams.x_reals)
    columns = [reals.index(f"{t}__observed") for t in TARGETS]
    for key in ("encoder_cont", "decoder_cont"):
        flipped[key] = x[key].clone()
        flipped[key][:, :, columns] = 1.0 - flipped[key][:, :, columns]

    model.eval()
    with torch.no_grad():
        before, after = model(x)["prediction"], model(flipped)["prediction"]
    for a, b in zip(before, after):
        assert torch.equal(a, b)


def test_the_flags_can_still_be_inputs(monkeypatch):
    monkeypatch.setattr(data_config, "KEEP_PARTIAL_TARGETS", True)
    monkeypatch.setattr(data_config, "TFT_OBSERVED_FLAGS_AS_INPUTS", True)
    model = create_tft_model(_dataset(), PARAMS, n_targets=2)
    assert "B__observed" in model.reals
