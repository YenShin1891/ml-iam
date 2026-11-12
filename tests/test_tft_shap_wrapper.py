"""The TFT SHAP wrapper must explain predictions, not attention weights --
and each explained window in its own context, over encoder and decoder inputs."""

from collections import namedtuple

import pytest

torch = pytest.importorskip("torch")

from src.visualization.shap_nn import _TFTPredictionWrapper, _slice_batch

BATCH, ENC_STEPS, N_FEATURES = 4, 3, 5
HORIZON, N_TARGETS, N_HEADS = 12, 3, 2
STEPS = ENC_STEPS + HORIZON

# Same field order as pytorch_forecasting's TFT output.
Output = namedtuple(
    "Output",
    ["prediction", "encoder_attention", "decoder_attention", "decoder_lengths"],
)


class StubTFT(torch.nn.Module):
    """Returns known per-target predictions plus a decoy attention tensor."""

    def __init__(self, n_outputs=1):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.n_outputs = n_outputs
        self.seen_batch = None

    def forward(self, batch):
        self.seen_batch = batch
        rows = batch["encoder_cont"].shape[0]
        # Target t, horizon h -> t*100 + h, plus the encoder inputs' sum, the
        # decoder inputs of that step and the row's categorical code, so the
        # output depends on everything the wrapper has to route.
        encoder = batch["encoder_cont"].sum(dim=(1, 2)).reshape(rows, 1, 1)
        decoder = batch["decoder_cont"].sum(dim=2).reshape(rows, HORIZON, 1)
        code = batch["decoder_cat"][:, :, :1].to(torch.float32) * 1000
        base = torch.arange(HORIZON, dtype=torch.float32).reshape(1, HORIZON, 1)
        prediction = [
            (t * 100 + base + encoder + decoder + code).expand(rows, HORIZON, self.n_outputs)
            for t in range(N_TARGETS)
        ]
        attention = torch.full((rows, HORIZON, N_HEADS, ENC_STEPS), 7.0)
        return Output(
            prediction=prediction,
            encoder_attention=attention,
            decoder_attention=attention,
            decoder_lengths=batch.get("decoder_lengths"),
        )


def _row(code=0, decoder_length=HORIZON, scale=1.0):
    """One window's batch (batch size 1), as the dataset collates it."""
    return {
        "encoder_cont": torch.zeros(1, ENC_STEPS, N_FEATURES),
        "decoder_cont": torch.zeros(1, HORIZON, N_FEATURES),
        "encoder_cat": torch.full((1, ENC_STEPS, 1), code, dtype=torch.long),
        "decoder_cat": torch.full((1, HORIZON, 1), code, dtype=torch.long),
        "encoder_lengths": torch.tensor([ENC_STEPS]),
        "decoder_lengths": torch.tensor([decoder_length]),
        "target_scale": [torch.full((1, 2), scale) for _ in range(N_TARGETS)],
    }


def _wrapper(model=None, row=None):
    wrapper = _TFTPredictionWrapper(model or StubTFT(), ENC_STEPS)
    wrapper.set_row(row or _row())
    return wrapper


@pytest.fixture
def x():
    return torch.zeros(BATCH, STEPS, N_FEATURES)


def _horizon_mean(target_idx, horizon=HORIZON):
    return target_idx * 100 + torch.arange(horizon, dtype=torch.float32).mean()


def test_wrapper_returns_every_targets_horizon_mean(x):
    out = _wrapper()(x)

    assert out.shape == (BATCH, N_TARGETS)
    for target_idx in range(N_TARGETS):
        torch.testing.assert_close(out[:, target_idx], _horizon_mean(target_idx).expand(BATCH))


def test_wrapper_does_not_return_attention(x):
    """Scanning the output tuple for the first Tensor lands on attention."""
    out = _wrapper()(x)

    assert not torch.allclose(out, torch.full_like(out, 7.0))


def test_padded_decoder_steps_are_excluded(x):
    """Eval-mode datasets pad the decoder; padding must not enter the mean."""
    short = 4
    out = _wrapper(row=_row(decoder_length=short))(x)

    torch.testing.assert_close(out[:, 1], _horizon_mean(1, horizon=short).expand(BATCH))


def test_quantile_output_uses_the_median(x):
    """A quantile loss adds a trailing axis; the middle quantile is the point."""
    out = _wrapper(StubTFT(n_outputs=3))(x)

    torch.testing.assert_close(out[:, 0], _horizon_mean(0).expand(BATCH))


def test_single_target_tensor_prediction_still_works(x):
    class SingleTargetTFT(StubTFT):
        def forward(self, batch):
            out = super().forward(batch)
            return out._replace(prediction=out.prediction[0])

    out = _wrapper(SingleTargetTFT())(x)

    assert out.shape == (BATCH, 1)
    torch.testing.assert_close(out[:, 0], _horizon_mean(0).expand(BATCH))


def test_the_tensor_is_split_into_encoder_and_decoder_inputs():
    model = StubTFT()
    x = torch.arange(BATCH * STEPS * N_FEATURES, dtype=torch.float32).reshape(BATCH, STEPS, N_FEATURES)

    _wrapper(model)(x)

    torch.testing.assert_close(model.seen_batch["encoder_cont"], x[:, :ENC_STEPS])
    torch.testing.assert_close(model.seen_batch["decoder_cont"], x[:, ENC_STEPS:])
    # and it is the tensor continuous_inputs builds from a batch
    batch = {"encoder_cont": x[:, :ENC_STEPS], "decoder_cont": x[:, ENC_STEPS:]}
    torch.testing.assert_close(_TFTPredictionWrapper.continuous_inputs(batch), x)


def test_decoder_inputs_are_attributed():
    """Gradients must reach the decoder steps, not the encoder's alone."""
    x = torch.zeros(BATCH, STEPS, N_FEATURES, requires_grad=True)

    _wrapper()(x).sum().backward()

    assert torch.isfinite(x.grad).all()
    assert x.grad[:, :ENC_STEPS].abs().sum() > 0
    assert x.grad[:, ENC_STEPS:].abs().sum() > 0


def test_every_row_is_scored_in_its_own_context(x):
    """The explained window's codes, lengths and target scale reach every interpolation."""
    model = StubTFT()
    wrapper = _TFTPredictionWrapper(model, ENC_STEPS)

    wrapper.set_row(_row(code=2, decoder_length=5, scale=3.0))
    first = wrapper(x)
    seen = model.seen_batch
    assert seen["decoder_cat"].shape == (BATCH, HORIZON, 1) and bool((seen["decoder_cat"] == 2).all())
    assert seen["decoder_cat"].dtype == torch.long          # codes are never interpolated floats
    assert seen["decoder_lengths"].tolist() == [5] * BATCH
    assert all(scale.shape == (BATCH, 2) and bool((scale == 3.0).all()) for scale in seen["target_scale"])

    wrapper.set_row(_row(code=7))
    second = wrapper(x)
    assert bool((model.seen_batch["decoder_cat"] == 7).all())
    assert not torch.allclose(first, second)


def test_a_wrapper_without_a_row_refuses_to_guess(x):
    with pytest.raises(RuntimeError, match="set_row"):
        _TFTPredictionWrapper(StubTFT(), ENC_STEPS)(x)


def test_slicing_a_batch_keeps_one_window_with_its_batch_axis():
    batch = {
        "encoder_cat": torch.arange(BATCH).reshape(BATCH, 1, 1).expand(BATCH, ENC_STEPS, 1),
        "decoder_lengths": torch.arange(BATCH) + 1,
        "target_scale": [torch.arange(BATCH, dtype=torch.float32).reshape(BATCH, 1).expand(BATCH, 2)],
    }

    row = _slice_batch(batch, 2)

    assert row["encoder_cat"].shape == (1, ENC_STEPS, 1) and bool((row["encoder_cat"] == 2).all())
    assert row["decoder_lengths"].tolist() == [3]
    assert row["target_scale"][0].shape == (1, 2) and bool((row["target_scale"][0] == 2).all())
