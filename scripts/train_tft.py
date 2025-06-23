import argparse
import logging
import numpy as np
import pandas as pd
import torch

from src.data.preprocess import load_and_process_data, prepare_data, prepare_features_and_targets_tft, remove_rows_with_missing_outputs
from src.trainers.tft_trainer import hyperparameter_search_tft, visualize_multiple_hyperparam_searches_tft
from src.trainers.evaluation import test_tft_autoregressively, save_metrics
from src.utils.utils import setup_logging, save_session_state, load_session_state, get_next_run_id
from src.utils.plotting import plot_scatter
from pytorch_forecasting import TimeSeriesDataSet

np.random.seed(0)

def process_data():
    data = load_and_process_data()
    prepared, features, targets = prepare_features_and_targets_tft(data)
    (
        X_train, y_train, 
        X_val, y_val,
        X_test_with_index, y_test, 
        test_data
    ) = prepare_data(prepared, targets, features)

    X_train, y_train = remove_rows_with_missing_outputs(X_train, y_train)
    X_val, y_val = remove_rows_with_missing_outputs(X_val, y_val)
    X_test_with_index, y_test, test_data = remove_rows_with_missing_outputs(X_test_with_index, y_test, test_data)

    return {
        "features": features,
        "targets": targets,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test_with_index": X_test_with_index,
        "y_test": y_test,
        "test_data": test_data
    }

def build_datasets(session_state):
    X_train = session_state["X_train"]
    y_train = session_state["y_train"]
    X_val = session_state["X_val"]
    y_val = session_state["y_val"]
    targets = session_state["targets"]

    y_train_df = pd.DataFrame(y_train, columns=targets, index=X_train.index)
    y_val_df = pd.DataFrame(y_val, columns=targets, index=X_val.index)

    train_data = pd.concat([X_train, y_train_df], axis=1)
    val_data = pd.concat([X_val, y_val_df], axis=1)

    group_ids = [col for col in ['Model', 'Model_Family', 'Scenario', 'Scenario_Category', 'Region'] if col in train_data.columns]

    for col in group_ids:
        if col in train_data.columns:
            train_data[col] = train_data[col].astype(str)
            val_data[col] = val_data[col].astype(str)

    common_params = {
        "time_idx": "Year",
        "target": targets[0],  # 단일 타겟 기준
        "group_ids": ["Region", "Scenario"],  # 필요에 따라 조정
        "max_encoder_length": 5,
        "max_prediction_length": 1,
        "time_varying_known_reals": ["Year"],
        "time_varying_unknown_reals": targets,
        "add_relative_time_idx": True,
        "add_target_scales": True,
        "allow_missing_timesteps": True,
    }

    train_dataset = TimeSeriesDataSet(train_data, **common_params)
    val_dataset = TimeSeriesDataSet(val_data, **common_params)

    session_state["train_dataset"] = train_dataset
    session_state["val_dataset"] = val_dataset

    return train_dataset, val_dataset

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
    args = parser.parse_args()
    return args.run_id

def main():
    full_pipeline = True

    if full_pipeline:
        run_id = get_next_run_id()
        setup_logging(run_id)
        session_state = process_data()
        save_session_state(session_state, run_id)
        best_params = train_tft(session_state, run_id)
        session_state["best_params"] = best_params
        save_session_state(session_state, run_id)
        preds = test_tft(session_state, run_id)
        session_state["preds"] = preds
        save_session_state(session_state, run_id)
        plot_tft(session_state, run_id)
    else:
        run_id = "run_05"
        setup_logging(run_id)
        session_state = load_session_state(run_id)
        preds = test_tft(session_state, run_id)
        session_state["preds"] = preds
        save_session_state(session_state, run_id)
        plot_tft(session_state, run_id)

if __name__ == "__main__":
    main()
