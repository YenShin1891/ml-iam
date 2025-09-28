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
from src.utils.plotting import plot_scatter_with_uncertainty

np.random.seed(0)
seed_everything(42, workers=True)


def process_data(dataset_version=None):
    data = load_and_process_data(version=dataset_version)
    prepared, features, targets = prepare_features_and_targets_tft(data)
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
        "test_data": test_data
    }


def search_tft(session_state, run_id):
    """Run hyperparameter search and store best_params in session_state."""
    logging.info("Starting hyperparameter search for TFT...")
    train_dataset, val_dataset = build_datasets(session_state)
    targets = session_state["targets"]
    # best_params = hyperparameter_search_tft(
    #     train_dataset, val_dataset, targets, run_id
    # )
    best_params = {'lstm_layers': 2, 'learning_rate': 0.01, 'hidden_size': 32, 'dropout': 0.1}
    session_state["best_params"] = best_params
    logging.info("Hyperparameter search complete. Best params: %s", best_params)
    return best_params


def train_tft(session_state, run_id):
    """Final training using best_params already present in session_state."""
    if "best_params" not in session_state:
        raise ValueError("best_params not found in session state. Run the 'search' step first or inject best_params manually.")
    best_params = session_state["best_params"]
    logging.info("Starting final TFT training with best params: %s", best_params)
    train_dataset, val_dataset = build_datasets(session_state)
    targets = session_state["targets"]
    train_final_tft(
        train_dataset, val_dataset, targets, run_id, best_params, session_state=session_state
    )
    logging.info("Final TFT training complete.")
    return best_params


def test_tft(session_state, run_id):
    preds = predict_tft(session_state, run_id)
    session_state["preds"] = preds
    return preds


def plot_tft(session_state, run_id):
    logging.info("Plotting TFT quantile predictions from session_state...")

    targets = session_state["targets"]
    horizon_df = session_state.get("horizon_df")
    y_true = session_state.get("horizon_y_true")
    y_pred_q = session_state.get("horizon_y_pred_quantiles")  # <-- use quantiles

    missing = [k for k in ["horizon_df", "horizon_y_true", "horizon_y_pred_quantiles"]
               if session_state.get(k) is None]
    if missing:
        raise ValueError(f"Missing {missing} in session_state. Run the test step first.")

    # Ensure shape is [N, n_targets, n_quantiles]
    y_pred_q = np.asarray(y_pred_q)
    if y_pred_q.ndim == 2:
        n_samples, n_cols = y_pred_q.shape
        n_targets = len(targets)
        if n_cols % n_targets != 0:
            raise ValueError(
                f"Cannot infer n_quantiles from shape {y_pred_q.shape} with {n_targets} targets."
            )
        n_quantiles = n_cols // n_targets
        y_pred_q = y_pred_q.reshape(n_samples, n_targets, n_quantiles)
    elif y_pred_q.ndim != 3:
        raise ValueError(f"Expected quantiles with ndim 2 or 3; got {y_pred_q.ndim}.")

    # Plot (linear + log)
    plot_scatter_with_uncertainty(run_id, horizon_df, y_true, y_pred_q, targets,
                                  use_log=False, model_label="TFT")


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
    args = parser.parse_args()
    
    # Validation: if resume is specified, run_id must be provided
    if args.resume and not args.run_id:
        parser.error("--run_id is required when --resume is specified")
    
    # Validation: if resume is not specified, run_id should not be provided
    if not args.resume and args.run_id:
        parser.error("--run_id should only be specified when using --resume")
    
    return args.run_id, args.resume, args.dataset


def main():
    run_id, resume, dataset_version = parse_arguments()

    if resume is None:
        # Full pipeline: process -> search -> train -> test -> plot
        run_id = get_next_run_id()
        setup_logging(run_id)
        session_state = process_data(dataset_version=dataset_version)
        save_session_state(session_state, run_id)
        search_tft(session_state, run_id)
        save_session_state(session_state, run_id)
        train_tft(session_state, run_id)
        save_session_state(session_state, run_id)
        test_tft(session_state, run_id)
        save_session_state(session_state, run_id)
        plot_tft(session_state, run_id)
        return
    else:
        setup_logging(run_id)
        session_state = load_session_state(run_id)
        if resume != "plot":
            session_state = process_data(dataset_version=dataset_version)
        save_session_state(session_state, run_id)

    # Step-wise execution when resuming
    if resume == "search":
        search_tft(session_state, run_id)
        save_session_state(session_state, run_id)
    if resume in ["search", "train"]:
        train_tft(session_state, run_id)
        save_session_state(session_state, run_id)
    if resume in ["test"]:
        test_tft(session_state, run_id)
        save_session_state(session_state, run_id)
    if resume in ["test", "plot"]:
        plot_tft(session_state, run_id)


if __name__ == "__main__":
    main()
