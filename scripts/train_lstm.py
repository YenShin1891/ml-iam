"""LSTM phase functions: preprocess, search, train, test, plot.

These are called by the unified scripts/train.py entrypoint.
All heavy imports are lazy to avoid pulling in unnecessary dependencies.
"""

import logging

from src.utils.utils import is_primary_rank


def _default_best_params_from_config() -> dict:
    """Hyperparameters to train with when the search phase is skipped."""
    from src.trainers.lstm_trainer import default_lstm_params

    return default_lstm_params()


def derive_splits(data, store=None):
    """From cached processed_data, derive all LSTM splits. Takes seconds.

    Returns the ephemeral dict that phase functions and trainers expect.
    *store* supplies the run's category vocabularies so codes stay identical
    across phases and models.
    """
    from src.data.preprocess import (
        build_categorical_vocabularies,
        add_missingness_indicators,
        encode_categorical_columns,
        impute_with_train_medians,
        prepare_features_and_targets_sequence,
        split_data,
    )
    from configs.data import CATEGORICAL_COLUMNS

    categories = (
        store.categories_for(data) if store is not None
        else build_categorical_vocabularies(data)
    )
    split_assignment = store.splits_for(data) if store is not None else None

    prepared, features, targets = prepare_features_and_targets_sequence(data)
    prepared, features = add_missingness_indicators(prepared, features)

    num_regions = len(categories.get("Region", [])) or None
    model_family_categories = categories.get("Model_Family")
    num_model_families = len(model_family_categories) if model_family_categories else None

    # Separate categorical features from continuous features
    categorical_features = [c for c in CATEGORICAL_COLUMNS if c in features]
    continuous_features = [f for f in features if f not in categorical_features]

    # Split while the identity columns still hold region labels: the run's
    # assignment is keyed by them, and the codes assigned below would match
    # none of them.  XGB splits before encoding for the same reason.
    train_data, val_data, test_data = split_data(prepared, assignment=split_assignment)

    # Encode categoricals as integer codes against the run's vocabulary, one
    # vocabulary for all three splits so a label keeps the same code in each;
    # the embedding sizes must cover every code, not just the ones present here.
    train_data, val_data, test_data = (
        encode_categorical_columns(frame, CATEGORICAL_COLUMNS, categories)
        for frame in (train_data, val_data, test_data)
    )
    train_data, val_data, test_data = (
        frame.astype({col: "int64" for col in CATEGORICAL_COLUMNS if col in frame.columns})
        for frame in (train_data, val_data, test_data)
    )
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
        "categories": categories,
        "targets": targets,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
    }


def search_lstm(store):
    """Run hyperparameter search and save best_params."""
    logging.info("Starting hyperparameter search for LSTM...")
    data = store.load_processed_data()
    splits = derive_splits(data, store)

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
    # The searcher already logged the winning parameters, and the phase banner
    # marks the end of the phase; repeating both here said nothing new.
    return best_params


def train_lstm(store):
    """Final training using best_params.

    Under DDP, non-primary ranks still need derive_splits() (LSTM has no
    saved dataset template), but skip metadata/artifact saving.
    """
    from src.trainers.lstm_trainer import train_final_lstm as _train_final

    primary = is_primary_rank()

    logging.info("Starting final LSTM training...")
    data = store.load_processed_data()
    splits = derive_splits(data, store)

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


def test_lstm(store):
    """Make predictions using trained LSTM model."""
    from src.trainers.lstm_trainer import predict_lstm as _predict_lstm

    logging.info("Testing LSTM model...")
    data = store.load_processed_data()
    splits = derive_splits(data, store)
    session_state = _build_predict_state(store, splits)

    preds = _predict_lstm(session_state, store.run_id)

    # Extract horizon data if the predictor produced it
    horizon_df = session_state.get("horizon_df")
    horizon_y_true = session_state.get("horizon_y_true")
    store.save_predictions(preds, horizon_df=horizon_df, horizon_y_true=horizon_y_true)
    # Save test_data for dashboard filters; y_test extracted from target columns
    # and converted from per-capita back to absolute (preds are already absolute).
    from src.data.preprocess import denormalize_by_population
    from configs.data import POPULATION_COLUMN
    test_data = splits["test_data"]
    y_test = denormalize_by_population(
        test_data[splits["targets"]].values, test_data[POPULATION_COLUMN].values
    )
    store.save_test_data(test_data, y_test)
    return preds


def _region_labels(frame, categories):
    """Region labels for *frame*, whichever form its Region column is in.

    Region survives as text in some frames and as embedding codes in others;
    returning None where neither is available lets the caller decide, rather
    than filtering against codes that match no region prefix.
    """
    import pandas as pd

    from src.data.preprocess import decode_categorical_column

    if "Region" not in frame.columns:
        return None
    if not pd.api.types.is_numeric_dtype(frame["Region"]):
        return frame["Region"].astype(str)

    vocabulary = (categories or {}).get("Region")
    if not vocabulary:
        logging.warning(
            "Region is encoded but this run has no Region vocabulary; "
            "SHAP cannot be filtered by region."
        )
        return None
    return decode_categorical_column(frame["Region"], vocabulary)


def plot_lstm(store):
    """Plot LSTM predictions and SHAP plots."""
    from src.visualization import plot_scatter, plot_lstm_shap

    logging.info("Plotting LSTM predictions...")
    data = store.load_processed_data()
    splits = derive_splits(data, store)
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

    from src.data.preprocess import denormalize_by_population
    from configs.data import POPULATION_COLUMN

    if horizon_df is not None:
        logging.info("Using forecast horizon subset (%d rows) for plotting.", len(horizon_df))
        y_true_aligned = denormalize_by_population(
            horizon_df[targets].values, horizon_df[POPULATION_COLUMN].values
        )
        plot_scatter(store.run_id, horizon_df, y_true_aligned, preds, targets, model_name="LSTM")
        test_data_for_shap = horizon_df
    else:
        test_data = splits["test_data"]
        test_targets = denormalize_by_population(
            test_data[targets].values, test_data[POPULATION_COLUMN].values
        )
        plot_scatter(store.run_id, test_data, test_targets, preds, targets, model_name="LSTM")
        test_data_for_shap = test_data

    # Generate SHAP plots
    from configs.models.lstm import LSTMTrainerConfig
    sequence_length = LSTMTrainerConfig().sequence_length
    if store.has_train_meta():
        meta = store.load_train_meta()
        sequence_length = meta.get("lstm_sequence_length", sequence_length)

    # The frame carries Region as the integer codes the embeddings were fit on,
    # which no region prefix can match; hand SHAP the labels behind them.
    plot_lstm_shap(
        store.run_id, test_data_for_shap, features, targets,
        sequence_length=sequence_length,
        region_series=_region_labels(test_data_for_shap, splits["categories"]),
    )
