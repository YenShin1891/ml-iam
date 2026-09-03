"""TFT phase functions: preprocess, search, train, test, plot.

These are called by the unified scripts/train.py entrypoint.
All heavy imports are lazy to avoid pulling in unnecessary dependencies.
"""

import logging

from src.utils.utils import is_primary_rank


def derive_splits(data, store=None, target_normalizer_mode=None):
    """From cached processed_data, derive all TFT splits. Takes seconds.

    Returns the ephemeral dict that phase functions and trainers expect.
    *store* supplies the run's category vocabularies; TFT encodes categoricals
    with its own NaNLabelEncoders, so these are recorded for the dashboard and
    inference rather than used for encoding here.
    """
    from configs.models.tft import TFTDatasetConfig
    from src.data.preprocess import (
        add_missingness_indicators,
        impute_with_train_medians,
        prepare_features_and_targets_sequence,
        split_data,
    )

    dataset_cfg = TFTDatasetConfig()
    context_length = max(0, dataset_cfg.target_offset)

    split_assignment = None
    if store is not None:
        store.categories_for(data)
        split_assignment = store.splits_for(data)

    prepared, features, targets = prepare_features_and_targets_sequence(data)
    dataset_cfg.resolve_encoder_lengths()
    if context_length > 0:
        logging.info(
            "Warm start enabled for TFT: target_offset=%d (retaining early steps for encoder context).",
            context_length,
        )
    prepared, features = add_missingness_indicators(prepared, features)
    train_data, val_data, test_data = split_data(prepared, assignment=split_assignment)
    train_data, val_data, test_data = impute_with_train_medians(
        train_data, val_data, test_data, features
    )

    # When keeping partial targets, fill NaN with 0 — the __observed mask
    # handles loss weighting so filled values don't contribute to gradients.
    # This avoids NaN propagation in EncoderNormalizer / TimeSeriesDataSet.
    from configs.data import KEEP_PARTIAL_TARGETS
    if KEEP_PARTIAL_TARGETS:
        for df in (train_data, val_data, test_data):
            df[targets] = df[targets].fillna(0.0)

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
        "tft_target_normalizer_mode": target_normalizer_mode,
    }


def search_tft(store, target_normalizer_mode=None):
    """Run hyperparameter search and save best_params."""
    logging.info("Starting hyperparameter search for TFT...")
    data = store.load_processed_data()
    splits = derive_splits(data, store, target_normalizer_mode=target_normalizer_mode)

    best_params = _search_with_splits(splits, store)
    return best_params


def _search_with_splits(splits, store):
    from configs.data import CONTEXT_LENGTHS
    from src.trainers.tft_dataset import build_datasets
    from src.trainers.tft_trainer import hyperparameter_search_tft

    # One dataset pair per searched context length.  build_datasets mutates
    # the state it is given, so each gets its own copy.
    datasets_by_encoder_length = {}
    for encoder_length in CONTEXT_LENGTHS:
        datasets_by_encoder_length[encoder_length] = build_datasets(
            dict(splits), encoder_length=encoder_length,
        )
        logging.info("Built TFT datasets for encoder_length=%d", encoder_length)

    best_params = hyperparameter_search_tft(
        datasets_by_encoder_length, splits["targets"], store.run_id,
    )
    store.save_best_params(best_params)
    store.save_features(splits["features"], splits["targets"])
    # The searcher already logged the winning parameters, and the phase banner
    # marks the end of the phase; repeating both here said nothing new.
    return best_params


def train_tft(store, target_normalizer_mode=None):
    """Final training using best_params."""
    from src.trainers.tft_dataset import build_datasets
    from src.trainers.tft_trainer import train_final_tft as _train_final

    primary = is_primary_rank()

    if primary:
        logging.info("Starting final TFT training...")

    data = store.load_processed_data()
    splits = derive_splits(data, store, target_normalizer_mode=target_normalizer_mode)

    best_params = store.load_best_params()

    if primary:
        logging.info("Training with best params: %s", best_params)

    # The encoder length is part of the winning configuration, so the final
    # fit has to see the same window the winning trial did.
    encoder_length = best_params.get("encoder_length")
    session_state = dict(splits)
    if encoder_length is not None:
        session_state["tft_encoder_length"] = int(encoder_length)
    train_dataset, val_dataset = build_datasets(
        dict(session_state)  # build_datasets needs its own copy since it mutates
    )

    _train_final(
        train_dataset, val_dataset, splits["targets"],
        store.run_id, best_params, session_state=session_state,
    )

    if primary:
        store.save_features(splits["features"], splits["targets"])
        logging.info("Final TFT training complete.")
    return best_params


def test_tft(store, use_two_window=False):
    """Make predictions using trained TFT model."""
    data = store.load_processed_data()
    splits = derive_splits(data, store)
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


def plot_tft(store):
    """Plot TFT predictions and SHAP analysis."""
    from src.visualization import plot_scatter, plot_tft_shap

    logging.info("Plotting TFT predictions...")
    data = store.load_processed_data()
    splits = derive_splits(data, store)
    pred_bundle = store.load_predictions()
    preds = pred_bundle["preds"]
    targets = splits["targets"]
    features = splits["features"]

    horizon_df = pred_bundle.get("horizon_df")
    horizon_y_true = pred_bundle.get("horizon_y_true")

    if horizon_df is not None and horizon_y_true is not None:
        logging.info("Using forecast horizon subset (%d rows) for plotting.", len(horizon_df))
        plot_scatter(store.run_id, horizon_df, horizon_y_true, preds, targets, model_name="TFT")
    else:
        from src.data.preprocess import denormalize_by_population
        from configs.data import POPULATION_COLUMN
        test_data = splits["test_data"]
        test_targets = denormalize_by_population(
            test_data[targets].values, test_data[POPULATION_COLUMN].values
        )
        plot_scatter(store.run_id, test_data, test_targets, preds, targets, model_name="TFT")

    # SHAP explains whole encoder windows, so it reads the full test split
    # rather than the horizon rows; it reports its own failures.
    plot_tft_shap(store.run_id, splits["test_data"], features, targets)
