"""The TFT SHAP wrapper must explain predictions, not attention weights."""

from collections import namedtuple

import pytest

torch = pytest.importorskip("torch")

from src.visualization.shap_nn import _TFTPredictionWrapper

BATCH, ENC_STEPS, N_FEATURES = 4, 3, 5
HORIZON, N_TARGETS, N_HEADS = 12, 3, 2

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
        # Target t, sample b, horizon h -> t*100 + b*10 + h, times the input sum
        # so the output actually depends on what SHAP perturbs.
        driver = batch["encoder_cont"].sum(dim=(1, 2)).reshape(rows, 1, 1)
        base = torch.arange(HORIZON, dtype=torch.float32).reshape(1, HORIZON, 1)
        offsets = torch.arange(rows, dtype=torch.float32).reshape(rows, 1, 1) * 10
        prediction = [
            (t * 100 + offsets + base + driver).expand(rows, HORIZON, self.n_outputs)
            for t in range(N_TARGETS)
        ]
        attention = torch.full((rows, HORIZON, N_HEADS, ENC_STEPS), 7.0)
        return Output(
            prediction=prediction,
            encoder_attention=attention,
            decoder_attention=attention,
            decoder_lengths=batch.get("decoder_lengths"),
        )


@pytest.fixture
def sample_batch():
    return {
        "encoder_cont": torch.zeros(1, ENC_STEPS, N_FEATURES),
        "encoder_cat": torch.zeros(1, ENC_STEPS, 0, dtype=torch.long),
        "decoder_lengths": torch.tensor([HORIZON]),
        "target_scale": torch.ones(1, 2),
    }


@pytest.fixture
def x():
    return torch.zeros(BATCH, ENC_STEPS, N_FEATURES)


def _expected_mean(target_idx, rows, horizon=HORIZON):
    base = torch.arange(horizon, dtype=torch.float32).mean()
    offsets = torch.arange(rows, dtype=torch.float32) * 10
    return (target_idx * 100 + offsets + base).reshape(rows, 1)


def test_wrapper_returns_the_targets_horizon_mean(sample_batch, x):
    for target_idx in range(N_TARGETS):
        out = _TFTPredictionWrapper(StubTFT(), target_idx, sample_batch)(x)

        assert out.shape == (BATCH, 1)
        torch.testing.assert_close(out, _expected_mean(target_idx, BATCH))


def test_wrapper_does_not_return_attention(sample_batch, x):
    """Scanning the output tuple for the first Tensor lands on attention."""
    out = _TFTPredictionWrapper(StubTFT(), 0, sample_batch)(x)

    assert not torch.allclose(out, torch.full_like(out, 7.0))


def test_targets_beyond_the_encoder_length_are_distinguished(sample_batch, x):
    """The old fallback gave every target past index 2 the same value."""
    outs = [
        _TFTPredictionWrapper(StubTFT(), t, sample_batch)(x) for t in range(N_TARGETS)
    ]

    for a in range(N_TARGETS):
        for b in range(a + 1, N_TARGETS):
            assert not torch.allclose(outs[a], outs[b]), f"targets {a} and {b} collapsed"


def test_padded_decoder_steps_are_excluded(x):
    """Eval-mode datasets pad the decoder; padding must not enter the mean."""
    short = 4
    batch = {
        "encoder_cont": torch.zeros(1, ENC_STEPS, N_FEATURES),
        "encoder_cat": torch.zeros(1, ENC_STEPS, 0, dtype=torch.long),
        "decoder_lengths": torch.tensor([short]),
    }

    out = _TFTPredictionWrapper(StubTFT(), 1, batch)(x)

    torch.testing.assert_close(out, _expected_mean(1, BATCH, horizon=short))


def test_quantile_output_uses_the_median(sample_batch, x):
    """A quantile loss adds a trailing axis; the middle quantile is the point."""
    out = _TFTPredictionWrapper(StubTFT(n_outputs=3), 0, sample_batch)(x)

    torch.testing.assert_close(out, _expected_mean(0, BATCH))


def test_single_target_tensor_prediction_still_works(sample_batch, x):
    class SingleTargetTFT(StubTFT):
        def forward(self, batch):
            out = super().forward(batch)
            return out._replace(prediction=out.prediction[0])

    out = _TFTPredictionWrapper(SingleTargetTFT(), 0, sample_batch)(x)

    torch.testing.assert_close(out, _expected_mean(0, BATCH))


def test_output_is_differentiable_wrt_the_encoder_input(sample_batch):
    """DeepExplainer needs gradients to flow back to the perturbed input."""
    x = torch.zeros(BATCH, ENC_STEPS, N_FEATURES, requires_grad=True)

    _TFTPredictionWrapper(StubTFT(), 0, sample_batch)(x).sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_target_index_out_of_range_is_an_error(sample_batch, x):
    with pytest.raises(IndexError, match="out of range"):
        _TFTPredictionWrapper(StubTFT(), N_TARGETS, sample_batch)(x)
