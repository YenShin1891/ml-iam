"""XGBoost phase functions: preprocess, search, train, test, plot.

These are called by the unified scripts/train.py entrypoint.
All heavy imports are lazy to avoid pulling in unnecessary dependencies.
"""

import logging


def derive_splits(data):
    """From cached processed_data, derive all XGB splits. Takes seconds.

    Returns the ephemeral dict that phase functions and trainers expect.
    """
    import numpy as np
    import pandas as pd
    from src.data.preprocess import prepare_data, prepare_features_and_targets

    prepared, features, targets = prepare_features_and_targets(data, lag_required=True)
    prepared = prepared.dropna(subset=targets)
    (
        X_train, y_train, X_train_index_columns,
        X_val, y_val, X_val_index_columns,
        X_test_with_index, y_test,
        test_data,
        x_scaler, y_scaler,
        train_groups, val_groups,
    ) = prepare_data(prepared, targets, features)

    return {
        "features": features,
        "targets": targets,
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
    }


def preprocess_xgb(store, dataset=None):
    """Run the expensive melt+pivot and cache as parquet."""
    from src.data.preprocess import load_and_process_data

    data = load_and_process_data(version=dataset)
    store.save_processed_data(data)
    return data


def search_xgb(store):
    """Run hyperparameter search and save best_params."""
    import pandas as pd

    logging.info("Starting hyperparameter search for XGBoost...")
    data = store.load_processed_data()
    splits = derive_splits(data)

    from src.trainers.xgb_trainer import hyperparameter_search

    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_train_index_columns = splits["X_train_index_columns"]
    train_groups = splits["train_groups"]
    X_val = splits["X_val"]
    y_val = splits["y_val"]
    X_val_index_columns = splits["X_val_index_columns"]
    val_groups = splits["val_groups"]
    targets = splits["targets"]

    X_train_with_index = pd.concat([X_train, X_train_index_columns], axis=1)
    X_val_with_index = pd.concat([X_val, X_val_index_columns], axis=1)

    best_params, all_results = hyperparameter_search(
        X_train, y_train, X_train_with_index, train_groups,
        targets, store.run_id, start_stage=1, use_cv=False,
        X_val=X_val, y_val=y_val, X_val_with_index=X_val_with_index, val_groups=val_groups,
    )
    store.save_best_params(best_params)
    store.save_features(splits["features"], splits["targets"])
    logging.info("Hyperparameter search complete. Best params: %s", best_params)
    return best_params


def train_xgb(store):
    """Final training using best_params."""
    import numpy as np
    import pandas as pd
    from src.trainers.xgb_trainer import train_and_save_model

    logging.info("Starting final XGBoost training...")
    data = store.load_processed_data()
    splits = derive_splits(data)

    best_params = store.load_best_params()

    logging.info("Training with best params: %s", best_params)

    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_train_index_columns = splits["X_train_index_columns"]
    train_groups = splits["train_groups"]
    X_val = splits["X_val"]
    y_val = splits["y_val"]
    X_val_index_columns = splits["X_val_index_columns"]
    val_groups = splits["val_groups"]
    targets = splits["targets"]

    X_train_with_index = pd.concat([X_train, X_train_index_columns], axis=1)
    X_val_with_index = pd.concat([X_val, X_val_index_columns], axis=1)

    X_combined = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    y_combined = np.concatenate([y_train, y_val], axis=0)
    X_combined_with_index = pd.concat([X_train_with_index, X_val_with_index], axis=0, ignore_index=True)
    combined_groups = np.concatenate([train_groups, val_groups], axis=0)

    train_and_save_model(X_combined, y_combined, combined_groups, targets, best_params, store.run_id)

    # Save scalers for autoregressive test
    store.save_artifact("x_scaler.pkl", splits["x_scaler"])
    store.save_artifact("y_scaler.pkl", splits["y_scaler"])
    store.save_features(splits["features"], splits["targets"])

    logging.info("Final XGBoost training complete.")
    return best_params


def test_xgb(store):
    """Test XGBoost model autoregressively."""
    logging.info("Testing the model...")
    data = store.load_processed_data()
    splits = derive_splits(data)

    X_test_with_index = splits["X_test_with_index"]
    y_test = splits["y_test"]
    test_data = splits["test_data"]

    from src.trainers.evaluation import test_xgb_autoregressively, save_metrics

    preds = test_xgb_autoregressively(X_test_with_index, y_test, store.run_id)
    store.save_predictions(preds)
    save_metrics(store.run_id, y_test, preds, test_data)
    return preds


def plot_xgb(store):
    """Plot XGBoost predictions and SHAP analysis."""
    import pandas as pd
    from src.visualization import plot_scatter, plot_xgb_shap

    data = store.load_processed_data()
    splits = derive_splits(data)
    pred_bundle = store.load_predictions()
    preds = pred_bundle["preds"]

    features = splits["features"]
    targets = splits["targets"]
    X_test_with_index = splits["X_test_with_index"]
    y_test = splits["y_test"]
    test_data = splits["test_data"]

    plot_scatter(store.run_id, test_data, y_test, preds, targets, model_name="XGBoost")
    index_region = test_data['Region'] if isinstance(test_data, pd.DataFrame) and 'Region' in test_data.columns else None
    plot_xgb_shap(store.run_id, X_test_with_index, features, targets, index_region=index_region)
