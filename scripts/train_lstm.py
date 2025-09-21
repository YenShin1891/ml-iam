import argparse
import logging
import os
import numpy as np
import pandas as pd
import torch

from lightning.pytorch import seed_everything

from src.data.preprocess import (
    add_missingness_indicators,
    impute_with_train_medians,
    load_and_process_data,
    prepare_features_and_targets_sequence,
    split_data,
)
from src.trainers.lstm_trainer import (
    hyperparameter_search_lstm,
    train_final_lstm,
    predict_lstm,
)
from src.utils.utils import (
    get_next_run_id,
    load_session_state,
    save_session_state,
    setup_logging,
)
from src.visualization import plot_scatter, plot_lstm_shap

np.random.seed(0)
seed_everything(42, workers=True)


def process_data():
    """Load and process data for LSTM training."""
    data = load_and_process_data()
    prepared, features, targets = prepare_features_and_targets_sequence(data)
    prepared, features = add_missingness_indicators(prepared, features)
    train_data, val_data, test_data = split_data(prepared)
    train_data, val_data, test_data = impute_with_train_medians(
        train_data, val_data, test_data, features
    )

    # Keep DataFrames like TFT (not arrays like XGBoost) for proper grouping
    return {
        "features": features,
        "targets": targets,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data
    }


def search_lstm(session_state, run_id):
    """Run hyperparameter search and store best_params in session_state."""
    logging.info("Starting hyperparameter search for LSTM...")
    REQUIRED_KEYS = ["features", "targets", "train_data", "val_data", "test_data"]
    missing = [k for k in REQUIRED_KEYS if k not in session_state]
    if missing:
        logging.info(
            "Session state missing keys %s for search; running preprocessing now...",
            missing,
        )
        data_bundle = process_data()
        # Merge freshly processed data into session state
        session_state.update(data_bundle)
        save_session_state(session_state, run_id)

    train_data = session_state["train_data"]
    val_data = session_state["val_data"]
    targets = session_state["targets"]
    features = session_state["features"]

    best_params = hyperparameter_search_lstm(
        train_data, val_data, targets, run_id, features
    )
    session_state["best_params"] = best_params
    logging.info("Hyperparameter search complete. Best params: %s", best_params)
    return best_params


def train_lstm(session_state, run_id):
    """Final training using best_params already present in session_state."""
    REQUIRED_KEYS = ["features", "targets", "train_data", "val_data", "test_data"]
    missing = [k for k in REQUIRED_KEYS if k not in session_state]
    if missing:
        raise RuntimeError(
            f"Session state missing required data keys for training: {missing}. "
            "Preprocess first (run without --resume) or resume from search after data is saved."
        )
    if "best_params" not in session_state:
        raise ValueError("best_params not found in session state. Run the 'search' step first or inject best_params manually.")

    best_params = session_state["best_params"]
    logging.info("Starting final LSTM training with best params: %s", best_params)

    train_data = session_state["train_data"]
    val_data = session_state["val_data"]
    targets = session_state["targets"]
    features = session_state["features"]

    train_final_lstm(
        train_data, val_data, targets, run_id, best_params, session_state=session_state, features=features
    )
    logging.info("Final LSTM training complete.")
    return best_params


def test_lstm(session_state, run_id):
    """Make predictions using trained LSTM model."""
    logging.info("Testing LSTM model...")
    preds = predict_lstm(session_state, run_id)
    session_state["preds"] = preds
    return preds


def plot_lstm(session_state, run_id):
    """Plot LSTM predictions and SHAP plots."""
    logging.info("Plotting LSTM predictions...")
    preds = session_state.get("preds")
    targets = session_state["targets"]
    features = session_state["features"]

    if preds is None:
        raise ValueError("No predictions found in session state. Please run the test step first.")

    # Use horizon data if available (similar to TFT), otherwise fall back to test data
    horizon_df = session_state.get('horizon_df')
    horizon_y_true = session_state.get('horizon_y_true')

    if horizon_df is not None:
        logging.info("Using forecast horizon subset (%d rows) for plotting.", len(horizon_df))
        # Derive y_true from horizon_df to ensure shape alignment with preds
        y_true_aligned = horizon_df[targets].values
        plot_scatter(run_id, horizon_df, y_true_aligned, preds, targets, model_name="LSTM")
        test_data_for_shap = horizon_df
    else:
        test_data = session_state.get("test_data")
        test_targets = session_state.get("test_data")[targets].values if test_data is not None else None

        if test_data is not None and test_targets is not None:
            plot_scatter(run_id, test_data, test_targets, preds, targets, model_name="LSTM")
            test_data_for_shap = test_data
        else:
            raise ValueError("No test data found in session state. Please run the test step with predict=True.")

    # Generate SHAP plots
    from configs.models.lstm import LSTMTrainerConfig
    sequence_length = session_state.get("lstm_sequence_length", LSTMTrainerConfig().sequence_length)
    plot_lstm_shap(run_id, test_data_for_shap, features, targets, sequence_length=sequence_length)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and test LSTM model.")
    parser.add_argument("--run_id", type=str, help="Run ID for logging.", required=False)
    parser.add_argument(
        "--resume",
        type=str,
        choices=["search", "train", "test", "plot"],
        help="Resume from a specific step. Requires --run_id to be specified.",
        required=False,
    )
    parser.add_argument(
        "--skip_search",
        action="store_true",
        help="Skip hyperparameter search step when running full pipeline.",
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

    return args.run_id, args.resume, args.skip_search, args.note


def main():
    """Main training pipeline."""
    run_id, resume, skip_search, note = parse_arguments()

    if resume is None:
        # New full run still performs preprocessing once, then executes chosen steps.
        run_id = get_next_run_id()
        # Only primary rank initializes logging to avoid duplication
        if os.getenv("PL_TRAINER_GLOBAL_RANK") in (None, "0") and os.getenv("GLOBAL_RANK") in (None, "0") and os.getenv("RANK") in (None, "0"):
            setup_logging(run_id)
        session_state = process_data()
        save_session_state(session_state, run_id)
        resume = "train" if skip_search else "search"
        if skip_search:
            from configs.models.lstm import LSTMTrainerConfig
            default_config = LSTMTrainerConfig()
            session_state["best_params"] = {
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
            }
            logging.info("Using default LSTM parameters (search skipped): %s", session_state["best_params"])
            save_session_state(session_state, run_id)
    else:
        # Only primary rank initializes logging to avoid duplication
        if os.getenv("PL_TRAINER_GLOBAL_RANK") in (None, "0") and os.getenv("GLOBAL_RANK") in (None, "0") and os.getenv("RANK") in (None, "0"):
            setup_logging(run_id)
        # Always load existing session state for any resume phase; preprocessing is never redone implicitly.
        session_state = load_session_state(run_id)

    if note:
        session_state["note"] = note
        logging.info("Run note: %s", note)

    # Step-wise execution when resuming - each phase runs independently
    if resume == "search":
        search_lstm(session_state, run_id)
        save_session_state(session_state, run_id)
    elif resume == "train":
        train_lstm(session_state, run_id)
        save_session_state(session_state, run_id)
    elif resume == "test":
        test_lstm(session_state, run_id)
        save_session_state(session_state, run_id)
    elif resume == "plot":
        plot_lstm(session_state, run_id)


if __name__ == "__main__":
    # Run CLI pipeline only on primary process to avoid duplicate runs under DDP
    rank_vars = [
        os.getenv("PL_TRAINER_GLOBAL_RANK"),
        os.getenv("GLOBAL_RANK"),
        os.getenv("RANK"),
    ]
    is_primary = all(rv in (None, "0") for rv in rank_vars)
    if is_primary:
        main()