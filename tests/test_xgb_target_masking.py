"""The per-target XGB wrapper must actually drop unobserved rows."""

import numpy as np
import pandas as pd
import pytest

from src.trainers.xgb_trainer import PerTargetXGBRegressor

TARGETS = ["A", "B"]


@pytest.fixture
def xy():
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"f": rng.normal(size=200)})
    y = pd.DataFrame({"A": X["f"] * 2.0, "B": X["f"] * -3.0})
    return X, y


def test_nan_rows_are_dropped_per_target(xy, caplog):
    X, y = xy
    y = y.copy()
    y.loc[y.index[:80], "B"] = np.nan

    model = PerTargetXGBRegressor(targets=TARGETS, n_estimators=5, max_depth=2)
    with caplog.at_level("INFO"):
        model.fit(X, y, verbose=False)

    assert "PerTarget[A] fitting on 200/200 rows" in caplog.text
    assert "PerTarget[B] fitting on 120/200 rows" in caplog.text


def test_zero_filled_targets_would_corrupt_the_fit(xy):
    """Guards the reason NaN must survive scaling: zeros are learned as truth."""
    X, y = xy
    unobserved = y.index[:150]

    masked = y.copy()
    masked.loc[unobserved, "B"] = np.nan
    zero_filled = y.copy()
    zero_filled.loc[unobserved, "B"] = 0.0

    def _fit(target_frame):
        model = PerTargetXGBRegressor(targets=TARGETS, n_estimators=40, max_depth=3)
        model.fit(X, target_frame, verbose=False)
        return model.predict(X)[:, 1]

    observed = ~y.index.isin(unobserved)
    truth = y["B"].values[observed]
    err_masked = np.abs(_fit(masked)[observed] - truth).mean()
    err_zero_filled = np.abs(_fit(zero_filled)[observed] - truth).mean()

    assert err_masked < err_zero_filled
