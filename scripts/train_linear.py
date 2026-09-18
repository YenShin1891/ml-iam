"""Linear-regression baseline phase functions: search, train, test, plot.

These are called by the unified scripts/train.py entrypoint.

The baseline is a linear regression with nothing tuned.  It sees the XGBoost
feature set on the run's shared splits and is scored by the same
autoregressive rollout, with only what a linear fit needs in order to run:
NaN features filled with train medians, Region / Model_Family one-hot encoded
because their integer codes carry no order, and a ridge penalty left at
scikit-learn's default (see RIDGE_ALPHA).
"""

import logging

MODEL_ARTIFACT = "final_linear.pkl"
ONE_STEP_METRICS = "performance_one_step.csv"

# scikit-learn's default, not a tuned value.  Plain least squares leaves some
# coefficients undetermined -- a region that never reports a target has a
# constant dummy in that target's fit -- and lands on ~1e10 for them.  The
# one-step metrics never see it (that element is unobserved), but the rollout
# feeds the prediction back as a lag and the trajectory diverges.  Any penalty
# settles those coefficients at zero; validation rollout R2 is the same to
# three decimals for alpha from 1e-6 to 1e3.
RIDGE_ALPHA = 1.0


class PerTargetLinear:
    """One linear fit per target, each on the rows that observe it.

    The counterpart of PerTargetXGBRegressor.  *skip_columns* are positions
    the fit ignores: the integer-coded categoricals stay in the frame, where
    the rollout groups trajectories by them, but only their one-hot columns
    are regressed on.
    """

    def __init__(self, skip_columns=()):
        self.skip_columns = sorted(int(c) for c in skip_columns)
        self.models = []

    def _design(self, X):
        import numpy as np

        X = np.asarray(X, dtype=float)
        return np.delete(X, self.skip_columns, axis=1) if self.skip_columns else X

    def fit(self, X, y):
        import numpy as np
        from sklearn.linear_model import Ridge

        X = self._design(X)
        y = np.asarray(y, dtype=float)
        self.models = []
        for j in range(y.shape[1]):
            observed = np.isfinite(y[:, j])
            if not observed.any():
                # Nothing to fit; the y scaler standardises such a target as
                # identity, so 0 is its mean.
                self.models.append(None)
                continue
            self.models.append(Ridge(alpha=RIDGE_ALPHA).fit(X[observed], y[observed, j]))
        return self

    def predict(self, X):
        import numpy as np

        X = self._design(X)
        return np.column_stack([
            np.zeros(len(X)) if model is None else model.predict(X)
            for model in self.models
        ])


def one_hot_name(column, category):
    return f"{column}={category}"


def _add_one_hot_columns(prepared, features, categories):
    """Append one column per category after the existing features.

    Appended, not substituted, so every existing feature keeps its position
    and the rollout still finds the lag columns where the scaler expects them.
    """
    import pandas as pd

    from configs.data import CATEGORICAL_COLUMNS

    blocks, names = [], []
    for column in CATEGORICAL_COLUMNS:
        if column not in features:
            continue
        vocabulary = [str(c) for c in categories[column]]
        dummies = pd.get_dummies(
            pd.Categorical(prepared[column].astype(str), categories=vocabulary),
            dtype="float32",
        )
        dummies.columns = [one_hot_name(column, c) for c in vocabulary]
        dummies.index = prepared.index
        blocks.append(dummies)
        names.extend(dummies.columns)
    if not blocks:
        return prepared, list(features)
    return pd.concat([prepared] + blocks, axis=1), list(features) + names


def derive_splits(data, store=None, n_lags=None):
    """The XGB splits, made usable by a linear fit.  Takes seconds."""
    import pandas as pd

    from configs.data import CATEGORICAL_COLUMNS, N_LAG_FEATURES
    from src.data.preprocess import (
        build_categorical_vocabularies,
        impute_with_train_medians,
        prepare_data,
        prepare_features_and_targets,
        split_data,
    )

    n_lags = int(N_LAG_FEATURES if n_lags is None else n_lags)

    categories = (
        store.categories_for(data) if store is not None
        else build_categorical_vocabularies(data)
    )
    split_assignment = store.splits_for(data) if store is not None else None
    prepared, features, targets = prepare_features_and_targets(
        data, lag_required=True, n_lags=n_lags,
    )

    # A linear fit cannot take NaN.  Fill with train medians, the imputation
    # the sequence models use, then hand the rows back to prepare_data in one
    # frame: it re-splits by the same assignment.
    continuous = [f for f in features if f not in CATEGORICAL_COLUMNS]
    prepared = pd.concat(
        impute_with_train_medians(
            *split_data(prepared, assignment=split_assignment), continuous,
        ),
        ignore_index=True,
    )
    prepared, features = _add_one_hot_columns(prepared, features, categories)

    (
        X_train, y_train, X_train_index_columns,
        X_val, y_val, X_val_index_columns,
        X_test_with_index, y_test,
        test_data,
        x_scaler, y_scaler,
        train_groups, val_groups,
        obs_train, obs_val, obs_test,
        categories,
    ) = prepare_data(
        prepared, targets, features,
        categories=categories, split_assignment=split_assignment,
    )

    return {
        "n_lags": n_lags,
        "features": features,
        "targets": targets,
        "skip_columns": [features.index(c) for c in CATEGORICAL_COLUMNS if c in features],
        # With their index columns, as the rollout wants them; see train_xgb.
        "X_train_with_index": pd.concat([X_train, X_train_index_columns], axis=1),
        "X_val_with_index": pd.concat([X_val, X_val_index_columns], axis=1),
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test_with_index": X_test_with_index,
        "y_test": y_test,
        "test_data": test_data,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "categories": categories,
    }


def _splits_for(store):
    data = store.load_processed_data()
    return derive_splits(data, store, n_lags=store.load_best_params().get("n_lags"))


def search_linear(store):
    """Nothing to search: the penalty stays at its default."""
    logging.info("The linear baseline is not tuned; skipping search.")
    return store.load_best_params() if store.has_best_params() else None


def train_linear(store):
    """Fit on the training split alone, the regime the other finals use."""
    logging.info("Fitting the linear baseline...")
    splits = _splits_for(store)

    model = PerTargetLinear(skip_columns=splits["skip_columns"])
    model.fit(splits["X_train"].values, splits["y_train"])
    logging.info(
        "Fitted %d targets on %d rows x %d columns",
        len(model.models), len(splits["X_train"]),
        len(splits["features"]) - len(splits["skip_columns"]),
    )

    store.save_artifact(MODEL_ARTIFACT, model)
    store.save_artifact("x_scaler.pkl", splits["x_scaler"])
    store.save_artifact("y_scaler.pkl", splits["y_scaler"])
    store.save_features(splits["features"], splits["targets"])
    return model


def test_linear(store):
    """Score the autoregressive rollout, and one-step prediction beside it."""
    logging.info("Testing the model...")
    splits = _splits_for(store)

    from configs.data import NON_FEATURE_COLUMNS, POPULATION_COLUMN
    from src.data.preprocess import denormalize_by_population, observed_mask_from_frame
    from src.trainers.evaluation import save_metrics, test_xgb_autoregressively

    X_test_with_index = splits["X_test_with_index"]
    test_data = splits["test_data"]
    targets = splits["targets"]
    y_scaler = splits["y_scaler"]
    model = store.load_artifact(MODEL_ARTIFACT)

    preds_scaled = test_xgb_autoregressively(
        X_test_with_index, splits["y_test"], store.run_id, model=model,
        y_scaler=y_scaler, x_scaler=splits["x_scaler"], n_lags=splits["n_lags"],
    )
    # One-step: every row predicted off its true lags.  The gap to the rollout
    # is how much of the error comes from feeding predictions back.
    feature_columns = [c for c in X_test_with_index.columns if c not in NON_FEATURE_COLUMNS]
    one_step_scaled = model.predict(X_test_with_index[feature_columns].values)

    population = test_data[POPULATION_COLUMN].values
    y_test = denormalize_by_population(test_data[targets].values, population)
    obs_mask = observed_mask_from_frame(test_data, targets)

    def absolute(scaled):
        return denormalize_by_population(y_scaler.inverse_transform(scaled), population)

    preds = absolute(preds_scaled)
    store.save_predictions(preds)
    store.save_test_data(test_data, y_test)
    save_metrics(store.run_id, y_test, preds, test_data, observed_mask=obs_mask)
    logging.info("One-step (true lags) metrics:")
    save_metrics(
        store.run_id, y_test, absolute(one_step_scaled), test_data,
        observed_mask=obs_mask, metrics_filename=ONE_STEP_METRICS,
    )
    return preds


def plot_linear(store):
    """Scatter of predictions against truth."""
    from src.visualization import plot_scatter

    _, targets = store.load_features()
    preds = store.load_predictions()["preds"]
    test_data, y_test = store.load_test_data()
    plot_scatter(store.run_id, test_data, y_test, preds, targets, model_name="Linear")
