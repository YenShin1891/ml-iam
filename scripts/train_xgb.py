import argparse
import logging
import numpy as np
import pandas as pd

from src.data.preprocess import prepare_data, load_and_process_data, prepare_features_and_targets
from src.trainers.xgb_trainer import hyperparameter_search, train_and_save_model
from src.trainers.evaluation import test_xgb_autoregressively, save_metrics
from src.utils.utils import setup_logging, save_session_state, load_session_state, load_model, get_next_run_id
from src.utils.plotting import plot_scatter, plot_shap

np.random.seed(0)

def preprocessing(run_id):
    data = load_and_process_data()
    prepared, features, targets = prepare_features_and_targets(data)
    prepared = prepared.dropna(subset=targets)
    (
    X_train, y_train, X_train_index_columns,
    X_val, y_val, X_val_index_columns,
    X_test_with_index, y_test,
    test_data,
    x_scaler, y_scaler,
    train_groups, val_groups
    ) = prepare_data(prepared, targets, features)
    save_session_state(x_scaler, run_id, "x_scaler.pkl")
    save_session_state(y_scaler, run_id, "y_scaler.pkl")
    
    return {
        "features": features,
        "targets": targets,
        "X_train": X_train,
        "y_train": y_train,
        "X_train_index_columns": X_train_index_columns,
    "X_val": X_val,
    "y_val": y_val,
    "X_val_index_columns": X_val_index_columns,
        "X_test_with_index": X_test_with_index,
        "y_test": y_test,
        "test_data": test_data,
    "train_groups": train_groups,
    "val_groups": val_groups
    }


def train_xgb(session_state, run_id, start_stage=1):
    targets = session_state["targets"]
    X_train = session_state["X_train"]
    y_train = session_state["y_train"]
    X_train_index_columns = session_state["X_train_index_columns"]
    train_groups = session_state["train_groups"]
    X_val = session_state["X_val"]
    y_val = session_state["y_val"]
    X_val_index_columns = session_state["X_val_index_columns"]
    val_groups = session_state["val_groups"]
    
    X_train_with_index = pd.concat([X_train, X_train_index_columns], axis=1)
    X_val_with_index = pd.concat([X_val, X_val_index_columns], axis=1)

    # Use single validation set for search (no Dask to keep it simple), then train on train+val
    best_params, all_results = hyperparameter_search(
        X_train, y_train, X_train_with_index, train_groups,
        targets, run_id, start_stage=start_stage, use_cv=False,
        X_val=X_val, y_val=y_val, X_val_with_index=X_val_with_index, val_groups=val_groups,
        use_dask=False
    )

    X_combined = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    y_combined = np.concatenate([y_train, y_val], axis=0)
    train_and_save_model(X_combined, y_combined, targets, best_params, run_id, use_dask=False)

    return best_params

def test_xgb(session_state, run_id):
    X_test_with_index = session_state["X_test_with_index"]
    y_test = session_state["y_test"]

    model = load_model(run_id)
            
    logging.info("Testing the model...")
    preds = test_xgb_autoregressively(X_test_with_index, y_test, run_id)
    session_state["preds"] = preds
    save_metrics(run_id, y_test, preds)

    return preds


def plot_xgb(session_state, run_id):
    features = session_state["features"]
    targets = session_state["targets"]
    X_test_with_index = session_state["X_test_with_index"]
    y_test = session_state["y_test"]
    preds = session_state["preds"]
    test_data = session_state["test_data"]

    plot_scatter(run_id, test_data, y_test, preds, targets)
    x_scaler = load_session_state(run_id, "x_scaler.pkl")
    y_scaler = load_session_state(run_id, "y_scaler.pkl")
    plot_scatter(run_id, test_data, y_scaler.inverse_transform(y_test), y_scaler.inverse_transform(preds), targets, filename="scatter_plot_inversed.png")
    plot_shap(run_id, X_test_with_index, features, targets)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and test XGBoost model.")
    parser.add_argument("--run_id", type=str, help="Run ID for logging.", required=False)
    args = parser.parse_args()
    return args.run_id


def main():
    full_pipeline = False

    if full_pipeline:
        run_id = get_next_run_id()
        setup_logging(run_id)
        session_state = preprocessing(run_id)
        save_session_state(session_state, run_id)
        best_params = train_xgb(session_state, run_id)
        session_state["best_params"] = best_params
        save_session_state(session_state, run_id)
        preds = test_xgb(session_state, run_id)
        session_state["preds"] = preds
        save_session_state(session_state, run_id)
        plot_xgb(session_state, run_id)
    else:
    run_id = "run_05"
        setup_logging(run_id)
        session_state = load_session_state(run_id)
        ### implement ###
    # Example: run training directly using existing session_state
    best_params = train_xgb(session_state, run_id)
    session_state["best_params"] = best_params
        save_session_state(session_state, run_id)
        preds = test_xgb(session_state, run_id)
        session_state["preds"] = preds
        save_session_state(session_state, run_id)
        plot_xgb(session_state, run_id)
        #################
        save_session_state(session_state, run_id)


if __name__ == "__main__":
    main()