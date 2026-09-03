"""denormalize_by_population must follow NORMALIZE_TARGETS_BY_POPULATION.

The flag was switched off while the multiplication stayed unconditional, so
every saved prediction, ground truth and metric was in <unit> x persons, and
a row with no population lost an observed target to NaN.
"""

import numpy as np
import pytest

import configs.data as data_config
from src.data.preprocess import denormalize_by_population, observed_mask_from_frame


@pytest.fixture
def absolute_targets(monkeypatch):
    monkeypatch.setattr(data_config, "NORMALIZE_TARGETS_BY_POPULATION", False)


@pytest.fixture
def per_capita_targets(monkeypatch):
    monkeypatch.setattr(data_config, "NORMALIZE_TARGETS_BY_POPULATION", True)


def test_absolute_targets_pass_through_untouched(absolute_targets):
    values = np.array([[1.0, 2.0], [3.0, np.nan]])

    out = denormalize_by_population(values, np.array([10.0, np.nan]))

    np.testing.assert_array_equal(out, values)


def test_per_capita_targets_are_scaled_back(per_capita_targets):
    out = denormalize_by_population(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([10.0, 100.0]))

    np.testing.assert_array_equal(out, [[10.0, 20.0], [300.0, 400.0]])


def test_a_flat_vector_keeps_its_shape(per_capita_targets):
    out = denormalize_by_population(np.array([1.0, 2.0]), np.array([10.0, 100.0]))

    np.testing.assert_array_equal(out, [10.0, 200.0])


def test_the_flag_is_read_when_called_not_when_imported(absolute_targets, monkeypatch):
    values = np.array([[1.0]])
    assert denormalize_by_population(values, np.array([5.0]))[0, 0] == 1.0

    monkeypatch.setattr(data_config, "NORMALIZE_TARGETS_BY_POPULATION", True)
    assert denormalize_by_population(values, np.array([5.0]))[0, 0] == 5.0


def test_observed_mask_comes_from_the_frame_or_not_at_all():
    import pandas as pd

    frame = pd.DataFrame({"A": [1.0, 2.0], "A__observed": [1.0, 0.0], "B": [3.0, 4.0]})

    assert observed_mask_from_frame(frame, ["A", "B"]) is None
    frame["B__observed"] = [0.0, 1.0]
    np.testing.assert_array_equal(observed_mask_from_frame(frame, ["A", "B"]), [[1.0, 0.0], [0.0, 1.0]])
