"""The sequence models explain a random sample of windows, on one scale, and say how well it adds up."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from src.visualization import shap_nn
from src.visualization.helpers import draw_shap_beeswarm, normalise_shap_values


def test_the_sample_is_drawn_from_every_window_not_the_first_ones():
    explained, background = shap_nn.draw_window_sample(10_000, n_explained=100, n_background=50)

    assert len(explained) == 100 and len(background) == 50
    assert len(set(explained)) == 100 and len(set(background)) == 50
    assert not set(explained) & set(background)
    # the old code explained the head of the frame: windows 0..99
    assert explained.max() > 5_000 and background.max() > 5_000


def test_the_sample_is_seeded():
    first = shap_nn.draw_window_sample(5_000, n_explained=64, n_background=32)
    again = shap_nn.draw_window_sample(5_000, n_explained=64, n_background=32)
    other = shap_nn.draw_window_sample(5_000, n_explained=64, n_background=32, seed=1)

    assert all(np.array_equal(a, b) for a, b in zip(first, again))
    assert not np.array_equal(first[0], other[0])


def test_few_windows_are_all_explained_and_the_background_drawn_from_all_of_them():
    explained, background = shap_nn.draw_window_sample(30, n_explained=100, n_background=50)

    assert sorted(explained) == list(range(30))
    assert sorted(background) == list(range(30))
    assert all(len(part) == 0 for part in shap_nn.draw_window_sample(0))


def test_temporal_shap_is_normalised_like_the_tree_models():
    rng = np.random.default_rng(0)
    temporal = rng.normal(size=(40, 3, 6, 2)) * np.array([1.0, 500.0])   # two targets, very different units

    scaled = shap_nn.normalise_temporal_shap(temporal)

    per_feature = scaled.sum(axis=1)                                      # what the beeswarm shows
    np.testing.assert_allclose(np.abs(per_feature).mean(axis=0).sum(axis=0), [1.0, 1.0])
    np.testing.assert_allclose(per_feature, normalise_shap_values(temporal.sum(axis=1)))
    assert not np.shares_memory(scaled, temporal)
    assert not shap_nn.normalise_temporal_shap(np.zeros((4, 3, 2, 1))).any()


def test_additivity_is_reported_per_target_and_pooled():
    rng = np.random.default_rng(1)
    temporal = rng.normal(size=(200, 3, 5, 2))
    base = rng.normal(size=(200, 2))
    exact = temporal.sum(axis=(1, 2)) + base

    good = shap_nn.additivity_report(temporal, exact, base, ["A", "B"])
    assert good["n_explained"] == 200
    assert good["correlation"] == pytest.approx(1.0) and good["relative_error"] == pytest.approx(0.0, abs=1e-9)

    # break target B only: the pooled figure and B's fall, A's does not
    broken = exact.copy()
    broken[:, 1] = rng.normal(size=200) * 5
    bad = shap_nn.additivity_report(temporal, broken, base, ["A", "B"])
    assert bad["per_target"]["A"]["correlation"] == pytest.approx(1.0)
    assert abs(bad["per_target"]["B"]["correlation"]) < 0.5
    assert bad["correlation"] < 0.9


class _ContextModel(torch.nn.Module):
    """Two outputs of a window's inputs, shifted and scaled by a held-fixed context."""

    def __init__(self):
        super().__init__()
        self.offset = None

    def set_row(self, context):
        self.offset = context

    def forward(self, x):
        total = x.sum(dim=(1, 2))
        return torch.stack([torch.tanh(total) * self.offset, total ** 2 + self.offset], dim=1)


def test_expected_gradients_add_up_to_the_prediction_difference():
    torch.manual_seed(0)
    row, background = torch.randn(1, 2, 3), torch.randn(20, 2, 3)
    model = _ContextModel()
    model.set_row(2.0)

    values = shap_nn._expected_gradients(model, row, background, n_samples=4000)

    assert values.shape == (2, 3, 2)
    expected = (model(row) - model(background).mean(dim=0)).numpy()[0]
    np.testing.assert_allclose(values.sum(axis=(0, 1)), expected, rtol=0.01, atol=0.01)
    # seeded: the same draws, the same values
    np.testing.assert_array_equal(values, shap_nn._expected_gradients(model, row, background, n_samples=4000))


def test_expected_gradients_agree_with_the_shap_library():
    shap = pytest.importorskip("shap")
    torch.manual_seed(1)
    row, background = torch.randn(1, 2, 3), torch.randn(20, 2, 3)
    model = _ContextModel()
    model.set_row(1.5)

    ours = shap_nn._expected_gradients(model, row, background, n_samples=4000)
    theirs = shap.GradientExplainer(model, background).shap_values(row, nsamples=4000, rseed=0)
    theirs = np.asarray(theirs)[0] if not isinstance(theirs, list) else np.stack([t[0] for t in theirs], axis=-1)

    assert np.corrcoef(ours.ravel(), theirs.ravel())[0, 1] > 0.99


def test_each_window_is_explained_under_its_own_context():
    torch.manual_seed(0)
    inputs = torch.randn(6, 2, 3)
    background = torch.randn(20, 2, 3)
    contexts = [float(j + 1) for j in range(len(inputs))]
    wrapper = _ContextModel()

    values, fx, base = shap_nn._explain_windows(wrapper, inputs, background, contexts)

    assert values.shape == (6, 2, 3, 2) and fx.shape == base.shape == (6, 2)
    for j, context in enumerate(contexts):
        wrapper.set_row(context)
        np.testing.assert_allclose(fx[j], wrapper(inputs[j:j + 1])[0].numpy(), rtol=1e-5)
        np.testing.assert_allclose(base[j], wrapper(background).mean(dim=0).numpy(), rtol=1e-5)
    # the attributions add up, each row under its own context
    report = shap_nn.additivity_report(values, fx, base, ["A", "B"])
    assert report["correlation"] > 0.98


def test_step_labels_continue_into_the_decoder():
    assert shap_nn._timestep_labels(5, encoder_length=3) == ["t-2", "t-1", "t", "t+1", "t+2"]
    assert shap_nn._timestep_labels(3, encoder_length=3) == ["t-2", "t-1", "t"]


def test_plots_and_arrays_go_to_the_directory_asked_for(tmp_path, monkeypatch):
    run = tmp_path / "run"
    out = tmp_path / "elsewhere"
    monkeypatch.setattr(shap_nn, "get_run_root", lambda _run_id: str(run))
    rng = np.random.default_rng(0)
    temporal = rng.normal(size=(12, 5, 4, 2))
    inputs = rng.normal(size=(12, 5, 4))
    features, targets = ["a", "b", "c", "d"], ["A", "B"]

    shap_nn._save_shap_arrays(str(out), "tft", {"temporal_shap": temporal, "averaged_shap": temporal.mean(axis=1)})
    shap_nn.draw_shap_all_timesteps_plot("tft_01", temporal, inputs, features, targets, 5, model_type="tft", out_dir=str(out))
    shap_nn.draw_temporal_shap_plot("tft_01", temporal, features, targets, 5, model_type="tft", encoder_length=3, out_dir=str(out))

    for name in (
        "tft_shap_values.npy", "tft_shap_values_temporal.npy", "tft_shap_cache.npz",
        "tft_shap_plot.png", "tft_temporal_shap_heatmap.png", "tft_timestep_importance.png",
    ):
        assert (out / name).exists(), name
    assert (out / "indiv_plots" / "shap" / "A.png").exists()
    assert not run.exists()                       # nothing was written into the run
    assert shap_nn._load_shap_cache(str(out / "tft_shap_cache.npz")) is not None


def test_a_cache_in_an_earlier_layout_is_not_read_back(tmp_path):
    path = tmp_path / "tft_shap_cache.npz"
    np.savez_compressed(path, temporal_shap=np.zeros((2, 3, 4, 1)))

    assert shap_nn._load_shap_cache(str(path)) is None
    assert shap_nn._load_shap_cache(str(tmp_path / "missing.npz")) is None


def test_the_beeswarm_styles_the_axes_shap_draws_on():
    """shap clears the figure, so the Axes handed in is gone by the time it returns."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    try:
        drawn = draw_shap_beeswarm(
            ax, rng.normal(size=(30, 5)), rng.normal(size=(30, 5)), list("abcde"),
            max_display=4, xlim_range=(-0.3, 0.5), point_size=5.0, xlabel="shared scale", title="Impact on A",
        )

        assert drawn in fig.axes
        assert drawn.get_xlim() == (-0.3, 0.5)
        assert drawn.get_title() == "Impact on A"
        assert drawn.get_xlabel() == "shared scale"
        sizes = np.concatenate([c.get_sizes() for c in drawn.collections])
        assert len(sizes) and np.all(sizes == 5.0)
    finally:
        plt.close(fig)
