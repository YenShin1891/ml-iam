"""The linear baseline runs on the XGB splits with only what a linear fit needs.

XGBoost takes NaN features and integer-coded categoricals as they are; a
linear fit takes neither.  The baseline fills the first and one-hot encodes the
second, and must do so without moving the lag columns the rollout writes to.
"""

import numpy as np
import pandas as pd
import pytest

import configs.data as data_config
import scripts.train_linear as train_linear
from src.trainers import evaluation

TARGETS = list(data_config.OUTPUT_VARIABLES[:2])
LAGS = [f"prev_{t}" for t in TARGETS] + [f"prev2_{t}" for t in TARGETS]
FEATURES = ["feat", "Region", "Model_Family"] + LAGS


@pytest.fixture
def splits(monkeypatch):
    monkeypatch.setattr(data_config, "KEEP_PARTIAL_TARGETS", True)
    rng = np.random.RandomState(0)
    rows = []
    for g in range(30):
        for year in (2020, 2025, 2030, 2035):
            rows.append({
                "Model": f"M{g}", "Scenario": f"S{g}", "Year": year,
                "Region": "World" if g % 2 else f"R10_{g % 4}",
                "Model_Family": f"FAM_{g % 3}",
                "feat": rng.rand(),
                **{name: rng.rand() for name in TARGETS + LAGS},
            })
    frame = pd.DataFrame(rows)
    frame.loc[frame.index % 3 == 0, "feat"] = np.nan
    frame.loc[frame.index % 5 == 0, LAGS[0]] = np.nan
    # The second target is unobserved half the time.
    frame[f"{TARGETS[0]}__observed"] = 1.0
    frame[f"{TARGETS[1]}__observed"] = (frame.index % 2).astype(float)

    import src.data.preprocess as preprocess
    monkeypatch.setattr(
        preprocess, "prepare_features_and_targets",
        lambda *a, **k: (frame.copy(), list(FEATURES), list(TARGETS)),
    )
    return train_linear.derive_splits(frame, n_lags=2)


def test_no_nan_reaches_the_fit(splits):
    assert np.isfinite(splits["X_train"].values).all()
    assert np.isfinite(splits["X_test_with_index"][splits["features"]].values).all()


def test_one_hot_columns_are_appended_so_the_lag_columns_do_not_move(splits):
    features = splits["features"]

    assert features[:len(FEATURES)] == FEATURES
    appended = features[len(FEATURES):]
    assert "Region=World" in appended and "Model_Family=FAM_0" in appended
    assert len(splits["x_scaler"].mean_) == len(features)


def test_partially_observed_targets_are_fitted_on_their_own_rows(splits):
    assert np.isnan(splits["y_train"][:, 1]).any()

    model = train_linear.PerTargetLinear(splits["skip_columns"])
    model.fit(splits["X_train"].values, splits["y_train"])

    assert all(m is not None for m in model.models)


def test_the_integer_codes_are_not_regressed_on(splits):
    model = train_linear.PerTargetLinear(splits["skip_columns"])
    X = splits["X_train"].values
    model.fit(X, splits["y_train"])

    recoded = X.copy()
    recoded[:, splits["skip_columns"]] += 100.0

    np.testing.assert_array_equal(model.predict(X), model.predict(recoded))


def test_it_rolls_out_through_the_shared_harness(splits):
    model = train_linear.PerTargetLinear(splits["skip_columns"])
    model.fit(splits["X_train"].values, splits["y_train"])

    preds = evaluation.test_xgb_autoregressively(
        splits["X_test_with_index"], splits["y_test"], model=model,
        y_scaler=splits["y_scaler"], x_scaler=splits["x_scaler"], n_lags=2,
        disable_progress=True,
    )

    assert preds.shape == splits["y_test"].shape
    assert np.isfinite(preds).all()


def test_a_dummy_the_target_never_varies_on_gets_no_weight():
    """Least squares put ~1e10 here, and the rollout fed it back as a lag."""
    rng = np.random.RandomState(0)
    X = rng.rand(200, 3)
    X[:, 2] = 0.0
    X[:5, 2] = 1.0  # a region whose rows all lack the target
    y = (2.0 * X[:, 0] - X[:, 1]).reshape(-1, 1)
    y[:5] = np.nan

    model = train_linear.PerTargetLinear().fit(X, y)

    assert model.models[0].coef_[2] == 0.0
    assert np.abs(model.predict(X[:5])).max() < 10.0
