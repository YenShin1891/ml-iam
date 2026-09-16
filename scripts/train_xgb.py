"""XGBoost phase functions: preprocess, search, train, test, plot.

These are called by the unified scripts/train.py entrypoint.
All heavy imports are lazy to avoid pulling in unnecessary dependencies.
"""

import logging


def derive_splits(data, store=None, n_lags=None):
    """From cached processed_data, derive all XGB splits. Takes seconds.

    Returns the ephemeral dict that phase functions and trainers expect.
    *store* supplies the run's category vocabularies so codes stay identical
    across phases and models.  *n_lags* is the context length -- how many past
    steps the lag features carry -- and defaults to configs.data's setting.
    """
    import pandas as pd

    from configs.data import N_LAG_FEATURES
    from src.data.preprocess import (
        build_categorical_vocabularies,
        prepare_data,
        prepare_features_and_targets,
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
        # The trainers want features and index columns in one frame; building
        # them here keeps every caller from re-deriving the same concat.
        "X_train_with_index": pd.concat([X_train, X_train_index_columns], axis=1),
        "X_val_with_index": pd.concat([X_val, X_val_index_columns], axis=1),
        "X_train": X_train,
        "y_train": y_train,
        "X_train_index_columns": X_train_index_columns,
        "X_val": X_val,
        "y_val": y_val,
        "X_val_index_columns": X_val_index_columns,
        "X_test_with_index": X_test_with_index,
        "y_test": y_test,
        "test_data": test_data,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "train_groups": train_groups,
        "val_groups": val_groups,
        "obs_train": obs_train,
        "obs_val": obs_val,
        "obs_test": obs_test,
        "categories": categories,
    }


def search_xgb(store):
    """Run hyperparameter search and save best_params."""
    from configs.data import CONTEXT_LENGTHS
    from src.trainers.xgb_trainer import hyperparameter_search

    logging.info("Starting hyperparameter search for XGBoost...")
    data = store.load_processed_data()

    # One set of splits per searched context length.  They differ only in how
    # many prev_ columns the features carry -- every variant drops the same
    # leading rows -- so the lag counts are scored on the same elements.
    splits_by_n_lags = {
        n_lags: derive_splits(data, store, n_lags=n_lags)
        for n_lags in CONTEXT_LENGTHS
    }
    for n_lags, splits in splits_by_n_lags.items():
        logging.info(
            "Prepared n_lags=%d splits: %d train / %d val rows, %d features",
            n_lags, len(splits["X_train"]), len(splits["X_val"]), len(splits["features"]),
        )

    best_params, _ = hyperparameter_search(splits_by_n_lags, store.run_id, use_cv=False)

    store.save_best_params(best_params)
    # The winning lag count decides the feature set the later phases rebuild,
    # so save that variant's features, not an arbitrary one.
    winner = splits_by_n_lags[int(best_params["n_lags"])]
    store.save_features(winner["features"], winner["targets"])
    # The searcher already logged the winning parameters, and the phase banner
    # marks the end of the phase; repeating both here said nothing new.
    return best_params


def train_xgb(store):
    """Final training using best_params."""
    from src.trainers.xgb_trainer import train_and_save_model

    logging.info("Starting final XGBoost training...")
    best_params = store.load_best_params()
    data = store.load_processed_data()
    # The lag count is part of the winning configuration, so the final fit has
    # to see the same feature set the winning trial did.
    splits = derive_splits(data, store, n_lags=best_params.get("n_lags"))

    logging.info("Training with best params: %s", best_params)

    # Train on the training split alone and keep val for early stopping, the
    # regime the search scored in and the one the LSTM and TFT finals use.
    train_and_save_model(
        splits["X_train"], splits["y_train"],
        splits["X_val"], splits["y_val"],
        splits["targets"], best_params, store.run_id,
    )

    # Save scalers for autoregressive test
    store.save_artifact("x_scaler.pkl", splits["x_scaler"])
    store.save_artifact("y_scaler.pkl", splits["y_scaler"])
    store.save_features(splits["features"], splits["targets"])

    logging.info("Final XGBoost training complete.")
    return best_params


def test_xgb(store):
    """Test XGBoost model autoregressively."""
    logging.info("Testing the model...")
    best_params = store.load_best_params()
    n_lags = best_params.get("n_lags")
    data = store.load_processed_data()
    splits = derive_splits(data, store, n_lags=n_lags)

    X_test_with_index = splits["X_test_with_index"]
    y_test_scaled = splits["y_test"]
    test_data = splits["test_data"]
    targets = splits["targets"]
    y_scaler = splits["y_scaler"]

    from src.trainers.evaluation import test_xgb_autoregressively, save_metrics
    from src.data.preprocess import denormalize_by_population, observed_mask_from_frame
    from configs.data import POPULATION_COLUMN

    preds_scaled = test_xgb_autoregressively(
        X_test_with_index, y_test_scaled, store.run_id, n_lags=splits["n_lags"],
    )

    # Undo the target scaling; denormalize_by_population then restores absolute
    # units when the run predicts per-capita targets (and is a no-op otherwise).
    # Ground truth comes straight from test_data rather than the scaled y_test
    # array, matching how LSTM/TFT source theirs.
    population = test_data[POPULATION_COLUMN].values
    preds = denormalize_by_population(y_scaler.inverse_transform(preds_scaled), population)
    y_test = denormalize_by_population(test_data[targets].values, population)

    obs_mask = observed_mask_from_frame(test_data, targets)

    store.save_predictions(preds)
    store.save_test_data(test_data, y_test)
    save_metrics(store.run_id, y_test, preds, test_data, observed_mask=obs_mask)
    return preds


def plot_xgb(store):
    """Plot XGBoost predictions and SHAP analysis."""
    import pandas as pd
    from src.visualization import plot_scatter, plot_xgb_shap

    data = store.load_processed_data()
    # Same lag count as training and test, or the SHAP frame would carry a
    # different feature set than the model was fitted on.
    splits = derive_splits(data, store, n_lags=store.load_best_params().get("n_lags"))
    pred_bundle = store.load_predictions()
    preds = pred_bundle["preds"]

    features = splits["features"]
    targets = splits["targets"]
    X_test_with_index = splits["X_test_with_index"]
    # Load already-absolute test_data/y_test as persisted by test_xgb(), rather
    # than re-deriving the scaled versions from derive_splits().
    test_data, y_test = store.load_test_data()

    plot_scatter(store.run_id, test_data, y_test, preds, targets, model_name="XGBoost")
    index_region = test_data['Region'] if isinstance(test_data, pd.DataFrame) and 'Region' in test_data.columns else None
    plot_xgb_shap(
        store.run_id, X_test_with_index, features, targets,
        index_region=index_region, categories=splits["categories"],
    )
