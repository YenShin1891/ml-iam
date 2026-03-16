import argparse
import logging
import os
import numpy as np
import pandas as pd
import torch

from lightning.pytorch import seed_everything

from configs.models import TFTDatasetConfig
from src.data.preprocess import (
    add_missingness_indicators,
    impute_with_train_medians,
    load_and_process_data,
    prepare_features_and_targets_tft,
    split_data,
)
from src.trainers.tft_trainer import (
    build_datasets,
    hyperparameter_search_tft,
    predict_tft,
    train_final_tft,
)
from src.utils.utils import (
    get_next_run_id,
    load_session_state,
    save_session_state,
    setup_logging,
)
from src.visualization import plot_scatter, plot_tft_shap

np.random.seed(0)
seed_everything(42, workers=True)


def process_data(dataset_version=None, lag_required=True):
    data = load_and_process_data(version=dataset_version)
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


def search_tft(session_state, run_id):
    """Run hyperparameter search and store best_params in session_state."""
    logging.info("Starting hyperparameter search for TFT...")
    # Ensure data is available (unlike LSTM script, original TFT search assumed prior processing)
    if "train_data" not in session_state or "val_data" not in session_state:
        logging.info("No processed data found in session state. Processing data before search...")
        data_state = process_data()
        session_state.update(data_state)
        save_session_state(session_state, run_id)
    train_dataset, val_dataset = build_datasets(session_state)
    targets = session_state["targets"]
    best_params = hyperparameter_search_tft(
        train_dataset, val_dataset, targets, run_id
    )
    session_state["best_params"] = best_params
    logging.info("Hyperparameter search complete. Best params: %s", best_params)
    return best_params


def train_tft(session_state, run_id, dataset_version=None):
    """Final training using best_params already present in session_state."""
    if "best_params" not in session_state:
        from configs.models.tft_search import TFTDefaultParams
        default_config = TFTDefaultParams()
        session_state["best_params"] = default_config.to_dict()
        logging.info("best_params not found in session state. Using default TFT parameters: %s", session_state["best_params"])
    best_params = session_state["best_params"]
    logging.info("Starting final TFT training with best params: %s", best_params)
    train_dataset, val_dataset = build_datasets(session_state)
    targets = session_state["targets"]
    train_final_tft(
        train_dataset, val_dataset, targets, run_id, best_params, session_state=session_state
    )
    logging.info("Final TFT training complete.")
    return best_params


def test_tft(session_state, run_id, use_two_window=False, dataset_version=None):
    if use_two_window:
        from src.trainers.tft_two_window_simple import predict_tft_two_window
        logging.info("Using two-window prediction approach...")
        preds = predict_tft_two_window(session_state, run_id)
    else:
        logging.info("Using standard single-window prediction...")
        preds = predict_tft(session_state, run_id)

    session_state["preds"] = preds
    return preds


def plot_tft(session_state, run_id):
    logging.info("Plotting TFT predictions...")
    preds = session_state.get("preds")
    targets = session_state["targets"]
    features = session_state["features"]

    if preds is None:
        logging.info("Predictions missing from current session_state; attempting to reload saved state.")
        reloaded_state = load_session_state(run_id) or {}
        for key in ("preds", "horizon_df", "horizon_y_true"):
            if key in reloaded_state and key not in session_state:
                session_state[key] = reloaded_state[key]
        preds = session_state.get("preds")
        if preds is None:
            raise ValueError("No predictions found in session state. Please run the test step first.")

    horizon_df = session_state.get('horizon_df')
    horizon_y_true = session_state.get('horizon_y_true')

    if horizon_df is not None and horizon_y_true is not None:
        logging.info("Using forecast horizon subset (%d rows) for plotting.", len(horizon_df))
        plot_scatter(run_id, horizon_df, horizon_y_true, preds, targets, model_name="TFT")
        test_data_for_shap = horizon_df
    else:
        test_data = session_state.get("test_data")
        test_targets = session_state.get("test_data")[targets].values if test_data is not None else None

        if test_data is not None and test_targets is not None:
            plot_scatter(run_id, test_data, test_targets, preds, targets, model_name="TFT")
            test_data_for_shap = test_data
        else:
            raise ValueError("No test data found in session state. Please run the test step with predict=True.")

    # Generate SHAP plots - use full test data instead of horizon subset for sufficient sequence length
    test_data_for_shap = session_state.get("test_data")
    if test_data_for_shap is not None:
        from configs.models.tft import TFTDatasetConfig
        max_encoder_length = session_state.get("tft_max_encoder_length", TFTDatasetConfig().max_encoder_length)
        try:
            plot_tft_shap(run_id, test_data_for_shap, features, targets, max_encoder_length=max_encoder_length)
        except Exception as e:
            logging.warning(f"TFT SHAP analysis failed: {e}")
            logging.info("This is likely due to categorical encoding incompatibilities in older TFT runs.")
            logging.info("SHAP analysis will be skipped. For full SHAP support, consider retraining with newer TFT pipeline.")
    else:
        logging.warning("No test data available for SHAP analysis")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and test TFT model.")
    parser.add_argument("--run_id", type=str, help="Run ID for logging.", required=False)
    parser.add_argument(
        "--resume",
        type=str,
        choices=["search", "train", "test", "plot"],
        help="Resume from a specific step. Requires --run_id to be specified.",
        required=False,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset version to use (subdirectory under data/). If not specified, uses default.",
        required=False,
    )
    parser.add_argument(
        "--lag-required",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require complete lag features (use --no-lag-required to allow missing lag history).",
    )
    parser.add_argument(
        "--two-window",
        action="store_true",
        help="Use two-window prediction approach (early + late windows with weighted averaging).",
    )
    parser.add_argument(
        "--skip_search",
        action="store_true",
        help="Skip hyperparameter search step and use default params.",
        required=False,
    )
    parser.add_argument(
        "--note",
        type=str,
        help="Note describing the run condition/type for later reference.",
        required=False,
    )
    args = parser.parse_args()

    # Validation: if resume is specified, run_id must be provided
    if args.resume and not args.run_id:
        parser.error("--run_id is required when --resume is specified")

    # Validation: if resume is not specified, run_id should not be provided
    if not args.resume and args.run_id:
        parser.error("--run_id should only be specified when using --resume")

    return args.run_id, args.resume, args.dataset, args.two_window, args.lag_required, args.skip_search, args.note


def main():
    run_id, resume, dataset_version, use_two_window, lag_required_arg, skip_search, note = parse_arguments()

    if resume is None:
        # Full pipeline: process -> search -> train -> test -> plot
        run_id = get_next_run_id("tft")
        setup_logging(run_id)

        if use_two_window:
            logging.info("Using two-window prediction approach")

        lag_required = True if lag_required_arg is None else lag_required_arg
        session_state = process_data(dataset_version=dataset_version, lag_required=lag_required)
        if note:
            session_state["note"] = note
            logging.info("Run note: %s", note)
        save_session_state(session_state, run_id)
        if skip_search:
            from configs.models.tft_search import TFTDefaultParams

            default_cfg = TFTDefaultParams()
            session_state["best_params"] = default_cfg.to_dict()
            logging.info(
                "skip_search enabled; using default TFT parameters: %s",
                session_state["best_params"],
            )
            save_session_state(session_state, run_id)
        else:
            search_tft(session_state, run_id)
            save_session_state(session_state, run_id)
        train_tft(session_state, run_id, dataset_version=dataset_version)
        save_session_state(session_state, run_id)
        test_tft(session_state, run_id, use_two_window=use_two_window, dataset_version=dataset_version)
        save_session_state(session_state, run_id)
        plot_tft(session_state, run_id)
        return
    else:
        setup_logging(run_id)
        # Load prior session to preserve artifacts (e.g., predictions) before refreshing data
        session_state = load_session_state(run_id) or {}
        if not session_state:
            logging.info("No existing session state found for %s; creating a new one before resuming.", run_id)
        else:
            logging.info("Loaded existing session state for %s and refreshing dataset-dependent entries.", run_id)
        if lag_required_arg is None:
            lag_required = session_state.get("lag_required", True)
        else:
            lag_required = lag_required_arg
            session_state["lag_required"] = lag_required
            logging.info("Overriding lag_required for resumed run: %s", lag_required)
        data_state = process_data(dataset_version=dataset_version, lag_required=lag_required)
        session_state.update(data_state)
        if note:
            session_state["note"] = note
            logging.info("Run note: %s", note)
        save_session_state(session_state, run_id)

    # Step-wise execution when resuming - each phase runs independently
    if resume == "search":
        if skip_search:
            from configs.models.tft_search import TFTDefaultParams

            default_cfg = TFTDefaultParams()
            session_state["best_params"] = default_cfg.to_dict()
            logging.info(
                "skip_search enabled; using default TFT parameters: %s",
                session_state["best_params"],
            )
        else:
            search_tft(session_state, run_id)
        save_session_state(session_state, run_id)
    elif resume == "train":
        train_tft(session_state, run_id, dataset_version=dataset_version)
        save_session_state(session_state, run_id)
    elif resume == "test":
        test_tft(session_state, run_id, use_two_window=use_two_window, dataset_version=dataset_version)
        save_session_state(session_state, run_id)
    elif resume == "plot":
        if session_state.get("preds") is None:
            logging.info("Predictions not present in session state; rerunning test step before plotting.")
            test_tft(session_state, run_id, use_two_window=use_two_window, dataset_version=dataset_version)
            save_session_state(session_state, run_id)
        plot_tft(session_state, run_id)


if __name__ == "__main__":
    main()