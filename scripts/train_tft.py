"""TFT phase functions: preprocess, search, train, test, plot.

These are called by the unified scripts/train.py entrypoint.
All heavy imports are lazy to avoid pulling in unnecessary dependencies.
"""

import logging


def derive_splits(data, lag_required=True):
    """From cached processed_data, derive all TFT splits. Takes seconds.

    Returns the ephemeral dict that phase functions and trainers expect.
    """
    from configs.models.tft import TFTDatasetConfig
    from src.data.preprocess import (
        add_missingness_indicators,
        impute_with_train_medians,
        prepare_features_and_targets_tft,
        split_data,
    )

    dataset_cfg = TFTDatasetConfig()
    context_length = max(0, dataset_cfg.target_offset)

    prepared, features, targets = prepare_features_and_targets_tft(
        data,
        lag_required=lag_required,
        min_context_length=0,
    )
    dataset_cfg.resolve_encoder_lengths()
    if context_length > 0:
        logging.info(
            "Warm start enabled for TFT: target_offset=%d (retaining early steps for encoder context).",
            context_length,
        )
    prepared, features = add_missingness_indicators(prepared, features)
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
        "tft_target_offset": context_length,
        "tft_min_encoder_length": dataset_cfg.effective_min_encoder_length,
        "tft_max_encoder_length": dataset_cfg.effective_max_encoder_length,
        "tft_time_idx_column": dataset_cfg.time_idx,
        "lag_required": lag_required,
    }


def preprocess_tft(store, dataset=None, lag_required=True):
    """Run the expensive melt+pivot and cache as parquet."""
    from src.data.preprocess import load_and_process_data

    data = load_and_process_data(version=dataset)
    store.save_processed_data(data)
    return data


def search_tft(store, lag_required=True):
    """Run hyperparameter search and save best_params."""
    logging.info("Starting hyperparameter search for TFT...")
    data = store.load_processed_data()
    splits = derive_splits(data, lag_required=lag_required)

    from src.trainers.tft_dataset import build_datasets

    # build_datasets expects a dict with train/val/test data
    best_params = _search_with_splits(splits, store)
    return best_params


def _search_with_splits(splits, store):
    from src.trainers.tft_dataset import build_datasets
    from src.trainers.tft_trainer import hyperparameter_search_tft

    session_state = dict(splits)
    train_dataset, val_dataset = build_datasets(session_state)

    best_params = hyperparameter_search_tft(
        train_dataset, val_dataset, splits["targets"], store.run_id,
    )
    store.save_best_params(best_params)
    store.save_features(splits["features"], splits["targets"])
    logging.info("Hyperparameter search complete. Best params: %s", best_params)
    return best_params


def train_tft(store, lag_required=True):
    """Final training using best_params."""
    from src.trainers.tft_dataset import build_datasets
    from src.trainers.tft_trainer import train_final_tft as _train_final

    logging.info("Starting final TFT training...")
    data = store.load_processed_data()
    splits = derive_splits(data, lag_required=lag_required)

    best_params = store.load_best_params()

    logging.info("Training with best params: %s", best_params)

    session_state = dict(splits)
    train_dataset, val_dataset = build_datasets(session_state)

    _train_final(
        train_dataset, val_dataset, splits["targets"],
        store.run_id, best_params, session_state=session_state,
    )
    store.save_features(splits["features"], splits["targets"])

    logging.info("Final TFT training complete.")
    return best_params


def test_tft(store, lag_required=True, use_two_window=False):
    """Make predictions using trained TFT model."""
    data = store.load_processed_data()
    splits = derive_splits(data, lag_required=lag_required)
    session_state = dict(splits)

    if use_two_window:
        from src.trainers.tft_two_window_simple import predict_tft_two_window
        logging.info("Using two-window prediction approach...")
        preds = predict_tft_two_window(session_state, store.run_id)
    else:
        from src.trainers.tft_trainer import predict_tft as _predict_tft
        logging.info("Using standard single-window prediction...")
        preds = _predict_tft(session_state, store.run_id)

    # Extract horizon data if the predictor produced it
    horizon_df = session_state.get("horizon_df")
    horizon_y_true = session_state.get("horizon_y_true")
    store.save_predictions(preds, horizon_df=horizon_df, horizon_y_true=horizon_y_true)
    return preds


def plot_tft(store, lag_required=True):
    """Plot TFT predictions and SHAP analysis."""
    from src.visualization import plot_scatter, plot_tft_shap

    logging.info("Plotting TFT predictions...")
    data = store.load_processed_data()
    splits = derive_splits(data, lag_required=lag_required)
    pred_bundle = store.load_predictions()
    preds = pred_bundle["preds"]
    targets = splits["targets"]
    features = splits["features"]

    horizon_df = pred_bundle.get("horizon_df")
    horizon_y_true = pred_bundle.get("horizon_y_true")

    if horizon_df is not None and horizon_y_true is not None:
        logging.info("Using forecast horizon subset (%d rows) for plotting.", len(horizon_df))
        plot_scatter(store.run_id, horizon_df, horizon_y_true, preds, targets, model_name="TFT")
        test_data_for_shap = horizon_df
    else:
        test_data = splits["test_data"]
        test_targets = test_data[targets].values
        plot_scatter(store.run_id, test_data, test_targets, preds, targets, model_name="TFT")
        test_data_for_shap = test_data

    # Use full test data for SHAP (needs sufficient sequence length)
    test_data_for_shap = splits["test_data"]
    if test_data_for_shap is not None:
        from configs.models.tft import TFTDatasetConfig
        max_encoder_length = splits.get("tft_max_encoder_length", TFTDatasetConfig().max_encoder_length)
        try:
            plot_tft_shap(store.run_id, test_data_for_shap, features, targets, max_encoder_length=max_encoder_length)
        except Exception as e:
            logging.warning("TFT SHAP analysis failed: %s", e)
            logging.info("SHAP analysis will be skipped.")
    else:
        logging.warning("No test data available for SHAP analysis")
