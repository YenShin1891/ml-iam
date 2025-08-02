import os
import pickle
import logging
import numpy as np
import pandas as pd

from src.data.preprocess import (
    load_and_process_data, 
    prepare_features_and_targets_mlforecast,
    split_data_mlforecast
)
from src.trainers.mlf_trainer import hyperparameter_search_mlforecast
from src.trainers.evaluation import evaluate_mlforecast_fragment_based, save_metrics
from src.utils.utils import setup_logging, save_session_state, load_session_state, get_next_run_id, load_mlforecast_model
from src.utils.plotting import plot_scatter
from configs.config import INDEX_COLUMNS

np.random.seed(0)

def process_data():
    data = load_and_process_data()
    prepared, features, targets = prepare_features_and_targets_mlforecast(data)
    train_data, val_data, test_data = split_data_mlforecast(prepared, features, targets, INDEX_COLUMNS)
    
    return {
        "features": features,
        "targets": targets,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data
    }


def train_mlforecast(session_state, run_id):
    train_data = session_state["train_data"]
    val_data = session_state["val_data"]
    targets = session_state["targets"]
    
    best_model, best_params = hyperparameter_search_mlforecast(
        train_data, val_data, targets, run_id
    )
    return best_params


def test_mlforecast(session_state, run_id):
    test_data = session_state["test_data"]
    targets = session_state["targets"]
    mlf = load_mlforecast_model(run_id)
    
    model_type = 'xgb'  # Assuming XGBoost model type
    predictions, avg_mse = evaluate_mlforecast_fragment_based(
        mlf, test_data, targets, mode="test", model_type=model_type
    )
    
    # Calculate metrics
    try:
        valid_preds = predictions.dropna(subset=['y', model_type])
        y_true = valid_preds['y'].values
        y_pred = valid_preds[model_type].values
        save_metrics(run_id, y_true, y_pred)
        
    except Exception as e:
        logging.error(f"Error calculating metrics: {str(e)}", exc_info=True)
    
    return predictions


def plot_mlf(session_state, run_id):
    mlf = load_mlforecast_model(run_id)
    pass


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Train and evaluate MLForecast model.")
    parser.add_argument('--resume', type=int, default=1, 
                       help='Resume from step: 1=process_data, 2=train, 3=test, 4=plot')
    parser.add_argument('--run-id', type=str, default=None,
                       help='Specific run ID to resume from')
    return parser.parse_args()
    

def main():
    args = parse_arguments()
    resume_step = args.resume
    
    if resume_step == 1:
        if args.run_id is None:
            run_id = get_next_run_id()
            setup_logging(run_id)
            logging.info(f"Starting new run with ID: {run_id}")
        else:
            run_id = args.run_id
            setup_logging(run_id)
            logging.info(f"Resuming run with ID: {run_id}")
    else:
        if args.run_id is None:
            raise ValueError(f"--run-id is required when resuming from step {resume_step}")
        run_id = args.run_id
        setup_logging(run_id)
        logging.info(f"Resuming run with ID: {run_id}")
    
    # Load or create session state
    if resume_step <= 1:
        logging.info("Step 1: Processing data...")
        session_state = process_data()
        save_session_state(session_state, run_id)
    else:
        logging.info("Loading session state...")
        session_state = load_session_state(run_id)
    
    # Execute steps based on resume point
    if resume_step <= 2:
        logging.info("Step 2: Training model...")
        session_state["best_params"] = train_mlforecast(session_state, run_id)
        save_session_state(session_state, run_id)
    
    if resume_step <= 3:
        logging.info("Step 3: Testing model...")
        session_state["predictions"] = test_mlforecast(session_state, run_id)
        save_session_state(session_state, run_id)
    
    if resume_step <= 4:
        logging.info("Step 4: Plotting results...")
        plot_mlf(session_state, run_id)

if __name__ == "__main__":
    main()