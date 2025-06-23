import logging
import numpy as np
import pandas as pd

from src.data.preprocess import prepare_data, load_and_process_data, prepare_features_and_targets
from src.trainers.xgb_trainer import hyperparameter_search, train_and_save_model
from src.trainers.evaluation import test_xgb_autoregressively, save_metrics
from src.utils.utils import setup_logging, save_session_state, load_session_state, get_next_run_id
from src.utils.plotting import plot_scatter, plot_shap

np.random.seed(0)

def preprocessing(run_id):
    data = load_and_process_data()
    prepared, features, targets = prepare_features_and_targets(data)
    prepared = prepared.dropna(subset=targets)
    (
        X_train, y_train, 
        X_train_index_columns,
        X_test_with_index, y_test, 
        test_data,
        x_scaler, y_scaler,
        train_groups
    ) = prepare_data(prepared, targets, features)
    save_session_state(y_scaler, run_id, "y_scaler.pkl")
    
    return {
        "features": features,
        "targets": targets,
        "X_train": X_train,
        "y_train": y_train,
        "X_train_index_columns": X_train_index_columns,
        "X_test_with_index": X_test_with_index,
        "y_test": y_test,
        "test_data": test_data,
        "train_groups": train_groups
    }


def train_xgb(session_state, run_id, start_stage=1):
    targets = session_state["targets"]
    X_train = session_state["X_train"]
    y_train = session_state["y_train"]
    X_train_index_columns = session_state["X_train_index_columns"]
    train_groups = session_state["train_groups"]
    
    X_train_with_index = pd.concat([X_train, X_train_index_columns], axis=1)
    best_params, all_results = hyperparameter_search(X_train, y_train, X_train_with_index, train_groups, targets, run_id, start_stage)
    train_and_save_model(X_train, y_train, targets, best_params, run_id)

    return best_params

def test_xgb(session_state, run_id):
    X_test_with_index = session_state["X_test_with_index"]
    y_test = session_state["y_test"]

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
    y_scaler = load_session_state(run_id, "y_scaler.pkl")
    plot_scatter(run_id, X_test_with_index, y_scaler.inverse_transform(y_test), y_scaler.inverse_transform(preds), targets, filename="scatter_plot_inversed.png")
    plot_shap(run_id, X_test_with_index, features, targets)

def main():
    full_pipeline = True

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
        run_id = "run_01"
        setup_logging(run_id)
        session_state = load_session_state(run_id)
        ### implement ###
        best_params = {'subsample': 0.9, 'scale_pos_weight': 100, 'reg_lambda': 10, 'reg_alpha': 10, 'num_boost_round': 300, 'max_depth': 12, 'eta': 0.01, 'gamma': 1, 'colsample_bytree': 0.6}
        X_train = session_state["X_train"]
        y_train = session_state["y_train"]
        targets = session_state["targets"]
        train_and_save_model(X_train, y_train, targets, best_params, run_id)
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