"""Lag re-attribution must act on the column a ranking names, not on its rank."""

import json

import numpy as np
import pytest

from src.visualization import shap_xgb

FEATURES = ["gdp", "prev_B", "carbon"]
TARGETS = ["A", "B"]


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(shap_xgb, "get_run_root", lambda _run_id: str(tmp_path))
    return tmp_path


def test_the_lagged_column_itself_is_replaced(run):
    shap_values = np.zeros((4, 3, 2))
    # Target A: prev_B (column 1) dominates, gdp (column 0) is second.
    shap_values[:, 1, 0] = [4.0, 4.0, 4.0, 4.0]
    shap_values[:, 0, 0] = [1.0, 1.0, 1.0, 1.0]
    # Target B: carbon (column 2) is its most important non-lag input.
    shap_values[:, 2, 1] = [2.0, 2.0, 2.0, 2.0]
    shap_values[:, 0, 1] = [1.0, 1.0, 1.0, 1.0]
    normalised = shap_values / shap_values.__abs__().mean(axis=0).sum(axis=0)

    out = shap_xgb.transform_outputs_to_former_inputs("xgb_01", shap_values, TARGETS, FEATURES)

    # prev_B's column carries carbon's attribution to B times its own; gdp's is untouched.
    np.testing.assert_allclose(out[:, 1, 0], normalised[:, 2, 1] * normalised[:, 1, 0])
    np.testing.assert_allclose(out[:, 0, 0], normalised[:, 0, 0])
    renaming = json.loads((run / "plots" / "csv" / "feature_renaming.json").read_text())
    assert renaming == {"A": {"prev_B": "prev_carbon"}, "B": {}}


def test_the_callers_array_is_left_alone(run):
    shap_values = np.ones((2, 3, 2))
    before = shap_values.copy()

    shap_xgb.transform_outputs_to_former_inputs("xgb_01", shap_values, TARGETS, FEATURES)

    np.testing.assert_array_equal(shap_values, before)


def test_fewer_features_than_the_ranking_depth_is_fine(run):
    shap_values = np.ones((2, 3, 2))

    out = shap_xgb.transform_outputs_to_former_inputs("xgb_01", shap_values, TARGETS, FEATURES)

    assert out.shape == shap_values.shape
