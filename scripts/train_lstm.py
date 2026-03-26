"""LSTM phase functions: preprocess, search, train, test, plot.

These are called by the unified scripts/train.py entrypoint.
All heavy imports are lazy to avoid pulling in unnecessary dependencies.
"""

import logging


def _default_best_params_from_config() -> dict:
    from configs.models.lstm import LSTMTrainerConfig

    default_config = LSTMTrainerConfig()
    return {
        "hidden_size": default_config.hidden_size,
        "num_layers": default_config.num_layers,
        "dropout": default_config.dropout,
        "bidirectional": default_config.bidirectional,
        "dense_hidden_size": default_config.dense_hidden_size,
        "dense_dropout": default_config.dense_dropout,
        "learning_rate": default_config.learning_rate,
        "batch_size": default_config.batch_size,
        "weight_decay": default_config.weight_decay,
        "sequence_length": default_config.sequence_length,
        "target_offset": default_config.target_offset,
    }


def derive_splits(data, lag_required=True):
    """From cached processed_data, derive all LSTM splits. Takes seconds.

    Returns the ephemeral dict that phase functions and trainers expect.
    """
    import pandas as pd
    from src.data.preprocess import (
        add_missingness_indicators,
        impute_with_train_medians,
        prepare_features_and_targets_sequence,
        split_data,
    )
    from configs.data import CATEGORICAL_COLUMNS, REGION_CATEGORIES

    prepared, features, targets = prepare_features_and_targets_sequence(
        data,
        lag_required=lag_required,
        min_context_length=0,
    )
    prepared, features = add_missingness_indicators(prepared, features)

    for col in CATEGORICAL_COLUMNS:
        if col not in prepared.columns:
            continue
        if col == "Region":
            prepared[col] = (
                pd.Categorical(prepared[col].astype(str), categories=REGION_CATEGORIES, ordered=True)
                .codes
                .astype("float32")
            )
        else:
            prepared[col] = prepared[col].astype("category").cat.codes.astype("float32")

    train_data, val_data, test_data = split_data(prepared)
    train_data, val_data, test_data = impute_with_train_medians(
        train_data, val_data, test_data, features
    )

    return {
        "features": features,
        "targets": targets,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
    }


def preprocess_lstm(store, dataset=None, lag_required=True):
    """Run the expensive melt+pivot and cache as parquet."""
    from src.data.preprocess import load_and_process_data

    data = load_and_process_data(version=dataset)
    store.save_processed_data(data)
    return data


def search_lstm(store, lag_required=True):
    """Run hyperparameter search and save best_params."""
    logging.info("Starting hyperparameter search for LSTM...")
    data = store.load_processed_data()
    splits = derive_splits(data, lag_required=lag_required)

    from src.trainers.lstm_trainer import hyperparameter_search_lstm

    best_params = hyperparameter_search_lstm(
        splits["train_data"], splits["val_data"],
        splits["targets"], store.run_id, splits["features"],
    )
    store.save_best_params(best_params)
    store.save_features(splits["features"], splits["targets"])
    logging.info("Hyperparameter search complete. Best params: %s", best_params)
    return best_params


def train_lstm(store, lag_required=True):
    """Final training using best_params."""
    from src.trainers.lstm_trainer import train_final_lstm as _train_final

    logging.info("Starting final LSTM training...")
    data = store.load_processed_data()
    splits = derive_splits(data, lag_required=lag_required)

    best_params = store.load_best_params()

    logging.info("Training with best params: %s", best_params)

    # Build ephemeral session_state dict for trainer (it writes metadata into it)
    session_state = dict(splits)
    _train_final(
        splits["train_data"], splits["val_data"],
        splits["targets"], store.run_id, best_params,
        session_state=session_state, features=splits["features"],
    )

    # Extract trainer-produced metadata and persist via RunStore
    store.save_features(splits["features"], splits["targets"])
    if "lstm_scaler_X" in session_state:
        store.save_artifact("lstm_scaler_X.pkl", session_state["lstm_scaler_X"])
    if "lstm_scaler_y" in session_state:
        store.save_artifact("lstm_scaler_y.pkl", session_state["lstm_scaler_y"])

    train_meta = {}
    for key in ("lstm_features", "lstm_raw_features", "lstm_non_numeric_features",
                "lstm_sequence_length", "lstm_target_offset"):
        if key in session_state:
            train_meta[key] = session_state[key]
    if "lstm_config" in session_state:
        cfg = session_state["lstm_config"]
        train_meta["lstm_config"] = {
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "dropout": cfg.dropout,
            "bidirectional": cfg.bidirectional,
            "dense_hidden_size": cfg.dense_hidden_size,
            "dense_dropout": cfg.dense_dropout,
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "weight_decay": cfg.weight_decay,
            "sequence_length": cfg.sequence_length,
            "target_offset": cfg.target_offset,
        }
    store.save_train_meta(train_meta)

    logging.info("Final LSTM training complete.")
    return best_params


def _build_predict_state(store, splits):
    """Build the ephemeral dict that predict_lstm expects, injecting saved artifacts."""
    session_state = dict(splits)

    if store.has_train_meta():
        meta = store.load_train_meta()
        for key in ("lstm_features", "lstm_raw_features", "lstm_non_numeric_features",
                    "lstm_sequence_length", "lstm_target_offset"):
            if key in meta:
                session_state[key] = meta[key]
        if "lstm_config" in meta:
            from configs.models.lstm import LSTMTrainerConfig
            session_state["lstm_config"] = LSTMTrainerConfig(**meta["lstm_config"])

    if store.has_artifact("lstm_scaler_X.pkl"):
        session_state["lstm_scaler_X"] = store.load_artifact("lstm_scaler_X.pkl")
    if store.has_artifact("lstm_scaler_y.pkl"):
        session_state["lstm_scaler_y"] = store.load_artifact("lstm_scaler_y.pkl")

    return session_state


def test_lstm(store, lag_required=True):
    """Make predictions using trained LSTM model."""
    from src.trainers.lstm_trainer import predict_lstm as _predict_lstm

    logging.info("Testing LSTM model...")
    data = store.load_processed_data()
    splits = derive_splits(data, lag_required=lag_required)
    session_state = _build_predict_state(store, splits)

    preds = _predict_lstm(session_state, store.run_id)

    # Extract horizon data if the predictor produced it
    horizon_df = session_state.get("horizon_df")
    horizon_y_true = session_state.get("horizon_y_true")
    store.save_predictions(preds, horizon_df=horizon_df, horizon_y_true=horizon_y_true)
    return preds


def plot_lstm(store, lag_required=True):
    """Plot LSTM predictions and SHAP plots."""
    from src.visualization import plot_scatter, plot_lstm_shap

    logging.info("Plotting LSTM predictions...")
    data = store.load_processed_data()
    splits = derive_splits(data, lag_required=lag_required)
    pred_bundle = store.load_predictions()
    preds = pred_bundle["preds"]
    targets = splits["targets"]

    # Determine features (prefer encoded features from train meta)
    features = splits["features"]
    if store.has_train_meta():
        meta = store.load_train_meta()
        features = meta.get("lstm_features", features)

    # Use horizon data if available, otherwise fall back to test data
    horizon_df = pred_bundle.get("horizon_df")

    if horizon_df is not None:
        logging.info("Using forecast horizon subset (%d rows) for plotting.", len(horizon_df))
        y_true_aligned = horizon_df[targets].values
        plot_scatter(store.run_id, horizon_df, y_true_aligned, preds, targets, model_name="LSTM")
        test_data_for_shap = horizon_df
    else:
        test_data = splits["test_data"]
        test_targets = test_data[targets].values
        plot_scatter(store.run_id, test_data, test_targets, preds, targets, model_name="LSTM")
        test_data_for_shap = test_data

    # Generate SHAP plots
    from configs.models.lstm import LSTMTrainerConfig
    sequence_length = LSTMTrainerConfig().sequence_length
    if store.has_train_meta():
        meta = store.load_train_meta()
        sequence_length = meta.get("lstm_sequence_length", sequence_length)
    plot_lstm_shap(store.run_id, test_data_for_shap, features, targets, sequence_length=sequence_length)
