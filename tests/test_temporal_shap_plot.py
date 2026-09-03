"""The temporal SHAP heatmap must label every panel and every encoder step.

The labelling block sat outside the loop, so only the last panel got its
ticks, title and colorbar, and with fewer than nine targets it raised on the
switched-off axis -- which the plot phase then swallowed.  Step labels came
out as "(current)" for every step.
"""

import numpy as np
import pytest

pytest.importorskip("torch")

from src.visualization import shap_nn


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(shap_nn, "get_run_root", lambda _run_id: str(tmp_path))
    return tmp_path


def test_step_labels_count_back_from_the_latest_step():
    assert shap_nn._timestep_labels(3) == ["t-2", "t-1", "t"]
    assert shap_nn._timestep_labels(1) == ["t"]


def test_two_targets_are_drawn_without_error(run):
    rng = np.random.default_rng(0)
    temporal = rng.normal(size=(5, 3, 4, 2))  # samples, steps, features, targets

    shap_nn.draw_temporal_shap_plot("lstm_01", temporal, ["a", "b", "c", "d"], ["A", "B"], 3, model_type="lstm")

    assert (run / "plots" / "lstm_temporal_shap_heatmap.png").exists()
    assert (run / "plots" / "lstm_timestep_importance.png").exists()


def test_shap_outputs_are_stacked_one_way(run):
    per_output = [np.zeros((5, 3, 4)), np.ones((5, 3, 4))]
    single = np.zeros((5, 3, 4))

    assert shap_nn._stack_shap_outputs(per_output).shape == (5, 3, 4, 2)
    assert shap_nn._stack_shap_outputs(single).shape == (5, 3, 4, 1)
