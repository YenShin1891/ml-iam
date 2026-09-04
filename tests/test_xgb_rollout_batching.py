"""The rollout predicts every trajectory's step t in one call.

autoregressive_predictions rolled each group forward with one single-row
predict per step: 28,000 calls for a 2,346-trajectory test set, which is why
the search could not afford to score stage 1 on it.  XGBoost predicts row by
row, so the groups can advance together -- one call per time step -- and the
per-group function stays as the reference these tests hold the batched one
to.
"""

import numpy as np
import pandas as pd
import pytest

from src.trainers import evaluation

TARGET = "Primary Energy|Coal"
COLUMNS = ["a", f"prev_{TARGET}", f"prev2_{TARGET}"]


class _Scaler:
    def __init__(self, mean, scale):
        self.mean_ = np.asarray(mean, dtype=float)
        self.scale_ = np.asarray(scale, dtype=float)


X_SCALER = _Scaler([0.3, 8.0, 5.0], [1.7, 2.0, 3.0])
Y_SCALER = _Scaler([10.0], [4.0])


class _LagModel:
    """A deterministic function of the row, so feedback is visible."""

    def __init__(self):
        self.calls = []

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        self.calls.append(len(X))
        return (0.5 * X[:, 0] + 0.8 * X[:, 1] - 0.3 * X[:, 2]).reshape(-1, 1)


def _groups(rng, lengths):
    return [rng.rand(length, len(COLUMNS)) for length in lengths]


def _reference(model, matrices, n_lags=2):
    return [
        evaluation.autoregressive_predictions(
            model, list(range(len(m))), m, 0, Y_SCALER, X_SCALER, COLUMNS, n_lags=n_lags,
        )
        for m in matrices
    ]


def test_the_batched_rollout_matches_the_per_group_reference_exactly():
    rng = np.random.RandomState(0)
    matrices = _groups(rng, [1, 3, 5, 5, 2])

    preds, lengths = evaluation.rollout_all_groups(_LagModel(), matrices, COLUMNS, Y_SCALER, X_SCALER, n_lags=2)
    expected = _reference(_LagModel(), matrices)

    for g, exp in enumerate(expected):
        np.testing.assert_array_equal(preds[g, :lengths[g]], exp)


def test_it_matches_a_real_booster_too():
    """Row independence is a property of XGBoost, not of the toy model."""
    xgboost = pytest.importorskip("xgboost")
    rng = np.random.RandomState(1)
    X = rng.rand(200, len(COLUMNS))
    booster = xgboost.XGBRegressor(n_estimators=15, max_depth=3, device="cpu")
    booster.fit(X, X[:, 1] * 2 + rng.rand(200) * 0.1)
    matrices = _groups(rng, [4, 6, 1, 6])

    preds, lengths = evaluation.rollout_all_groups(booster, matrices, COLUMNS, Y_SCALER, X_SCALER, n_lags=2)
    expected = _reference(booster, matrices)

    for g, exp in enumerate(expected):
        np.testing.assert_allclose(preds[g, :lengths[g]], exp, rtol=0, atol=1e-9)


def test_one_predict_call_per_time_step_not_per_row():
    model = _LagModel()
    matrices = _groups(np.random.RandomState(2), [12] * 50 + [7] * 30)

    evaluation.rollout_all_groups(model, matrices, COLUMNS, Y_SCALER, X_SCALER, n_lags=2)

    assert len(model.calls) == 12
    # Step 8 onwards only the 50 long groups are still running.
    assert model.calls[:7] == [80] * 7 and model.calls[7:] == [50] * 5


def test_groups_stop_at_their_own_length():
    preds, lengths = evaluation.rollout_all_groups(
        _LagModel(), _groups(np.random.RandomState(3), [2, 5]), COLUMNS, Y_SCALER, X_SCALER, n_lags=2,
    )

    assert list(lengths) == [2, 5]
    assert np.isfinite(preds[0, :2]).all() and np.isnan(preds[0, 2:]).all()
    assert np.isfinite(preds[1]).all()


def test_the_lag_column_receives_the_prediction_in_feature_units():
    class Constant:
        def __init__(self):
            self.seen = []

        def predict(self, X):
            self.seen.append(np.array(X, copy=True))
            return np.full((len(X), 1), 0.5)

    model = Constant()
    # y: 0.5 scaled -> 12 raw; lag-1 column x stats (8, 2) -> 2.0; lag-2 (5, 3) -> 7/3.
    evaluation.rollout_all_groups(model, [np.zeros((3, 3))], COLUMNS, Y_SCALER, X_SCALER, n_lags=2)

    assert model.seen[1][0, 1] == pytest.approx(2.0)
    assert model.seen[2][0, 1] == pytest.approx(2.0)
    assert model.seen[2][0, 2] == pytest.approx(7 / 3)


def test_ground_truth_lags_survive_where_no_prediction_exists_yet():
    """At step 1 the lag-2 column still holds the frame's own value."""
    model = _LagModel()
    matrix = np.arange(9, dtype=float).reshape(3, 3)

    evaluation.rollout_all_groups(model, [matrix], COLUMNS, Y_SCALER, X_SCALER, n_lags=2)
    reference = _reference(_LagModel(), [matrix])[0]

    preds, _ = evaluation.rollout_all_groups(_LagModel(), [matrix], COLUMNS, Y_SCALER, X_SCALER, n_lags=2)
    np.testing.assert_array_equal(preds[0], reference)


def test_a_scaler_that_does_not_fit_the_frame_is_refused():
    with pytest.raises(ValueError, match="fitted on 2 columns"):
        evaluation.rollout_all_groups(
            _LagModel(), _groups(np.random.RandomState(4), [3]), COLUMNS, Y_SCALER, _Scaler([0, 0], [1, 1]),
        )


def test_the_harness_puts_every_prediction_back_on_its_own_row():
    """Groups interleaved in the frame, index not 0..n: positions must not slip."""
    rng = np.random.RandomState(5)
    frame = pd.DataFrame(rng.rand(7, 3), columns=COLUMNS, index=[30, 10, 31, 11, 32, 12, 33])
    frame["Model"] = "m"
    frame["Scenario"] = ["s1", "s2", "s1", "s2", "s1", "s2", "s1"]
    # Region is an encoded feature column, not an index column, so it is
    # part of the rollout frame and of the scaler.
    frame["Region"] = 0.0
    features = COLUMNS + ["Region"]
    x_scaler = _Scaler([0.3, 8.0, 5.0, 0.0], [1.7, 2.0, 3.0, 1.0])

    out = evaluation.test_xgb_autoregressively(
        frame, np.zeros((7, 1)), model=_LagModel(), y_scaler=Y_SCALER, x_scaler=x_scaler,
        n_lags=2, disable_progress=True,
    )

    s1 = frame[frame["Scenario"] == "s1"][features].to_numpy()
    s2 = frame[frame["Scenario"] == "s2"][features].to_numpy()
    expected = [
        evaluation.autoregressive_predictions(
            _LagModel(), list(range(len(m))), m, 0, Y_SCALER, x_scaler, features, n_lags=2,
        )
        for m in (s1, s2)
    ]
    np.testing.assert_array_equal(out[[0, 2, 4, 6]], expected[0])
    np.testing.assert_array_equal(out[[1, 3, 5]], expected[1])
