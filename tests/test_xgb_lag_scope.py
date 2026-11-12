"""Each booster reads only the lagged outputs that are always there to read."""

import json

import numpy as np
import pandas as pd
import pytest

from src.trainers.xgb_trainer import PerTargetXGBRegressor, hidden_lag_columns

TARGETS = ["A", "B", "C"]
FEATURES = ["feat", "prev_A", "prev_B", "prev_C", "prev2_A", "prev2_B", "prev2_C"]
KWARGS = dict(n_estimators=8, max_depth=3, tree_method="hist", device="cpu", n_jobs=1)


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    y = pd.DataFrame({t: X[f"prev_{t}"] + 0.5 * X["feat"] + rng.normal(scale=0.1, size=n) for t in TARGETS})
    # B is not reported by the last third of the rows: no target, no lag.
    unreported = np.arange(n) >= 2 * n // 3
    y.loc[unreported, "B"] = np.nan
    X.loc[unreported, ["prev_B", "prev2_B"]] = np.nan
    return X, y


def test_scopes_hide_the_right_columns():
    _, y = _data()
    position = FEATURES.index

    assert hidden_lag_columns(FEATURES, TARGETS, y, "all") == [[], [], []]

    own = hidden_lag_columns(FEATURES, TARGETS, y, "own")
    assert own[0] == sorted(position(c) for c in ("prev_B", "prev_C", "prev2_B", "prev2_C"))
    assert position("feat") not in own[0] + own[1] + own[2]

    # A and C are reported on every row, B is not: only B's lags are withheld,
    # and only from the other outputs' boosters.
    mixed = hidden_lag_columns(FEATURES, TARGETS, y, "own_and_complete")
    assert mixed[0] == mixed[2] == [position("prev_B"), position("prev2_B")]
    assert mixed[1] == []

    with pytest.raises(ValueError, match="lag_scope"):
        hidden_lag_columns(FEATURES, TARGETS, y, "some")


def test_a_withheld_lag_cannot_move_the_prediction():
    """What the rollout writes into an unreported output's lag must not reach the others."""
    X, y = _data()
    model = PerTargetXGBRegressor(TARGETS, lag_scope="own_and_complete", **KWARGS)
    model.fit(X, y)

    rows = X.to_numpy()[-20:].copy()          # series that do not report B: its lags are missing
    filled = rows.copy()
    # the rollout's own predictions, written into the empty lag slots
    filled[:, [FEATURES.index("prev_B"), FEATURES.index("prev2_B")]] = np.linspace(-3, 3, 20)[:, None]

    before, after = model.predict(rows), model.predict(filled)
    assert np.array_equal(before[:, [0, 2]], after[:, [0, 2]])
    assert not np.array_equal(before[:, 1], after[:, 1])       # B's own booster does read them

    everything = PerTargetXGBRegressor(TARGETS, lag_scope="all", **KWARGS)
    everything.fit(X, y)
    assert not np.array_equal(everything.predict(rows)[:, [0, 2]], everything.predict(filled)[:, [0, 2]])


def test_the_scope_is_saved_with_the_model(tmp_path):
    X, y = _data()
    model = PerTargetXGBRegressor(TARGETS, lag_scope="own", **KWARGS)
    model.fit(X, y)
    path = str(tmp_path / "final_best.json")
    model.save_model(path)

    saved = json.loads((tmp_path / "final_best_lag_scope.json").read_text())
    assert saved["lag_scope"] == "own" and saved["targets"] == TARGETS

    loaded = PerTargetXGBRegressor.load_model(path, TARGETS)
    assert loaded.hidden_columns == model.hidden_columns
    rows = X.to_numpy()[:30]
    assert np.allclose(loaded.predict(rows), model.predict(rows))


def test_a_model_saved_without_the_record_reads_every_lag(tmp_path):
    X, y = _data()
    model = PerTargetXGBRegressor(TARGETS, **KWARGS)        # the default is the old behaviour
    model.fit(X, y)
    path = str(tmp_path / "final_best.json")
    model.save_model(path)
    (tmp_path / "final_best_lag_scope.json").unlink()        # as runs up to xgb_87 were saved

    loaded = PerTargetXGBRegressor.load_model(path, TARGETS)
    assert loaded.lag_scope == "all" and loaded.hidden_columns == [[], [], []]
    rows = X.to_numpy()[:30]
    assert np.allclose(loaded.predict(rows), model.predict(rows))
