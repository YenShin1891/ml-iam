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
        "embedding_dim": default_config.embedding_dim,
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

    # Encode categoricals as integer codes
    num_model_families = None
    num_regions = None
    model_family_categories = None
    for col in CATEGORICAL_COLUMNS:
        if col not in prepared.columns:
            continue
        if col == "Region":
            cat = pd.Categorical(prepared[col].astype(str), categories=REGION_CATEGORIES, ordered=True)
            prepared[col] = cat.codes.astype("int64")
            num_regions = len(REGION_CATEGORIES)
        else:
            cat = prepared[col].astype("category")
            prepared[col] = cat.cat.codes.astype("int64")
            if col == "Model_Family":
                num_model_families = len(cat.cat.categories)
                model_family_categories = list(cat.cat.categories)

    # Separate categorical features from continuous features
    categorical_features = [c for c in CATEGORICAL_COLUMNS if c in features]
    continuous_features = [f for f in features if f not in categorical_features]

    train_data, val_data, test_data = split_data(prepared)
    # Only impute continuous features (categoricals are already int codes, no NaN)
    train_data, val_data, test_data = impute_with_train_medians(
        train_data, val_data, test_data, continuous_features
    )

    return {
        "features": features,
        "continuous_features": continuous_features,
        "categorical_features": categorical_features,
        "num_model_families": num_model_families,
        "num_regions": num_regions,
        "model_family_categories": model_family_categories,
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
        categorical_features=splits["categorical_features"],
        num_model_families=splits["num_model_families"],
        num_regions=splits["num_regions"],
    )
    store.save_best_params(best_params)
    store.save_features(splits["features"], splits["targets"])
    logging.info("Hyperparameter search complete. Best params: %s", best_params)
    return best_params


def _is_primary_rank():
    import os
    rank_vars = [
        os.getenv("LOCAL_RANK"),
        os.getenv("PL_TRAINER_GLOBAL_RANK"),
        os.getenv("GLOBAL_RANK"),
        os.getenv("RANK"),
    ]
    return all(rv in (None, "0") for rv in rank_vars)


def train_lstm(store, lag_required=True):
    """Final training using best_params.

    Under DDP, non-primary ranks still need derive_splits() (LSTM has no
    saved dataset template), but skip metadata/artifact saving.
    """
    from src.trainers.lstm_trainer import train_final_lstm as _train_final

    primary = _is_primary_rank()

    logging.info("Starting final LSTM training...")
    data = store.load_processed_data()
    splits = derive_splits(data, lag_required=lag_required)

    best_params = store.load_best_params()

    if primary:
        logging.info("Training with best params: %s", best_params)

    # Build ephemeral session_state dict for trainer (it writes metadata into it)
    # Only pass session_state on primary rank so non-primary ranks skip saving
    session_state = dict(splits) if primary else None
    _train_final(
        splits["train_data"], splits["val_data"],
        splits["targets"], store.run_id, best_params,
        session_state=session_state, features=splits["features"],
        categorical_features=splits["categorical_features"],
        num_model_families=splits["num_model_families"],
        num_regions=splits["num_regions"],
    )

    if primary:
        # Extract trainer-produced metadata and persist via RunStore
        store.save_features(splits["features"], splits["targets"])
        if "lstm_scaler_X" in session_state:
            store.save_artifact("lstm_scaler_X.pkl", session_state["lstm_scaler_X"])
        if "lstm_scaler_y" in session_state:
            store.save_artifact("lstm_scaler_y.pkl", session_state["lstm_scaler_y"])

        train_meta = {}
        for key in ("lstm_features", "lstm_raw_features", "lstm_non_numeric_features",
                    "lstm_categorical_features", "lstm_num_model_families", "lstm_num_regions",
                    "lstm_sequence_length", "lstm_target_offset"):
            if key in session_state:
                train_meta[key] = session_state[key]
        # Save category vocab so inference encodes consistently
        if splits.get("model_family_categories"):
            train_meta["lstm_model_family_categories"] = splits["model_family_categories"]
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
                "embedding_dim": cfg.embedding_dim,
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
                    "lstm_categorical_features", "lstm_num_model_families", "lstm_num_regions",
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
