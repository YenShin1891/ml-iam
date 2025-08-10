import argparse
import logging
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet

from src.data.preprocess import load_and_process_data, split_data, prepare_features_and_targets_tft, encode_categorical_columns
from src.trainers.tft_trainer import hyperparameter_search_tft, visualize_multiple_hyperparam_searches_tft
from src.trainers.evaluation import test_tft_autoregressively, save_metrics
from src.utils.utils import setup_logging, save_session_state, load_session_state, get_next_run_id
from src.utils.plotting import plot_scatter
from configs.models import TFTDatasetConfig

np.random.seed(0)

def process_data():
    data = load_and_process_data()
    prepared, features, targets = prepare_features_and_targets_tft(data)
    train_data, val_data, test_data = split_data(prepared)
    
    encode_categorical_columns(train_data)
    encode_categorical_columns(val_data)
    encode_categorical_columns(test_data)
    
    train_data.dropna(subset=targets, inplace=True)
    val_data.dropna(subset=targets, inplace=True)
    test_data.dropna(subset=targets, inplace=True)
    
    return {
        "features": features,
        "targets": targets,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data
    }

def build_datasets(session_state):
    train_data = session_state["train_data"]
    val_data = session_state["val_data"]
    features = session_state["features"]
    targets = session_state["targets"]

    dataset_config = TFTDatasetConfig()

    train_params = dataset_config.build(features, targets, mode="train")
    val_params = dataset_config.build(features, targets, mode="train")

    train_dataset = TimeSeriesDataSet(train_data, **train_params)
    val_dataset = TimeSeriesDataSet(val_data, **val_params)

    session_state["train_dataset"] = train_dataset
    session_state["val_dataset"] = val_dataset


def train_tft(session_state, run_id):
    build_datasets(session_state)

    train_dataset = session_state["train_dataset"]
    val_dataset = session_state["val_dataset"]
    targets = session_state["targets"]

    best_model, best_params, cv_results_dict = hyperparameter_search_tft(
        train_dataset, val_dataset, targets, run_id
    )
    visualize_multiple_hyperparam_searches_tft(cv_results_dict, run_id)
    
    session_state["model"] = best_model

    return best_params

def test_tft(session_state, run_id):
    X_test_with_index = session_state["X_test_with_index"]
    y_test = session_state["y_test"]
    model = session_state["model"]

    logging.info("Testing TFT model...")
    # TODO: Add test result for when encoder decoder length is given so that the series is split at 2015 (only one possible encoder/decoder length)
    preds = test_tft_autoregressively(model, X_test_with_index, y_test)
    session_state["preds"] = preds
    save_metrics(run_id, y_test, preds)

    return preds

def plot_tft(session_state, run_id):
    X_test_with_index = session_state["X_test_with_index"]
    y_test = session_state["y_test"]
    preds = session_state["preds"]
    test_data = session_state["test_data"]
    targets = session_state["targets"]

    plot_scatter(run_id, test_data, y_test, preds, targets)
    plot_scatter(run_id, test_data, y_test, preds, targets, use_log=True)

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
