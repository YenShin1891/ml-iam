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
from src.utils.plotting import plot_scatter

np.random.seed(0)
seed_everything(42, workers=True)


def process_data():
    data = load_and_process_data()
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


def train_tft(session_state, run_id):
    train_dataset, val_dataset = build_datasets(session_state)
    
    targets = session_state["targets"]

    # best_params = hyperparameter_search_tft(
    #     train_dataset, val_dataset, targets, run_id
    # )
    # session_state["best_params"] = best_params
    session_state["best_params"] = {'lstm_layers': 2, 'learning_rate': 0.01, 'hidden_size': 32, 'dropout': 0.1}
    best_params = session_state["best_params"]
    train_final_tft(
        train_dataset, val_dataset, targets, run_id, best_params, session_state=session_state
    )

    return best_params


def test_tft(session_state, run_id):
    preds = predict_tft(session_state, run_id)
    return preds


def plot_tft(session_state, run_id):
    logging.info("Plotting TFT predictions...")
    preds = session_state.get("preds")
    targets = session_state["targets"]
    if preds is None:
        raise ValueError("No predictions found in session state. Please run the test step first.")

    horizon_df = session_state.get('horizon_df')
    horizon_y_true = session_state.get('horizon_y_true')
    
    if horizon_df is not None and horizon_y_true is not None:
        logging.info("Using forecast horizon subset (%d rows) for plotting.", len(horizon_df))
        plot_scatter(run_id, horizon_df, horizon_y_true, preds, targets, use_log=False, model_label="TFT")
        plot_scatter(run_id, horizon_df, horizon_y_true, preds, targets, use_log=True, model_label="TFT")
    else:
        raise ValueError("No forecast horizon data found in session state. Please run the test step with predict=True.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and test TFT model.")
    parser.add_argument("--run_id", type=str, help="Run ID for logging.", required=False)
    parser.add_argument("--resume", type=str, choices=["train", "test", "plot"], 
                      help="Resume from a specific step. Requires --run_id to be specified.", required=False)
    args = parser.parse_args()
    
    # Validation: if resume is specified, run_id must be provided
    if args.resume and not args.run_id:
        parser.error("--run_id is required when --resume is specified")
    
    # Validation: if resume is not specified, run_id should not be provided
    if not args.resume and args.run_id:
        parser.error("--run_id should only be specified when using --resume")
    
    return args.run_id, args.resume


def main():
    run_id, resume = parse_arguments()

    if resume is None:
        run_id = get_next_run_id()
        setup_logging(run_id)
        session_state = process_data()
        save_session_state(session_state, run_id)
    else:
        setup_logging(run_id)
        session_state = load_session_state(run_id)
    
    # Execute steps based on resume point
    if resume is None or resume == "train":
        best_params = train_tft(session_state, run_id)
        session_state["best_params"] = best_params
        save_session_state(session_state, run_id)
    
    if resume is None or resume in ["train", "test"]:
        preds = test_tft(session_state, run_id)
        session_state["preds"] = preds
        save_session_state(session_state, run_id)
    
    if resume is None or resume in ["train", "test", "plot"]:
        plot_tft(session_state, run_id)


if __name__ == "__main__":
    main()
