import argparse
import logging
import numpy as np

from src.data.preprocess import prepare_data, load_and_process_data, prepare_features_and_targets, remove_rows_with_missing_outputs
from src.trainers.xgb_trainer import hyperparameter_search, visualize_multiple_hyperparam_searches
from src.trainers.evaluation import test_xgb_autoregressively, save_metrics
from src.utils.utils import setup_logging, save_session_state, load_session_state, get_next_run_id
from src.utils.plotting import plot_scatter, plot_shap

np.random.seed(0)

def train_xgb(run_id):
    data = load_and_process_data()
    prepared, features, targets = prepare_features_and_targets(data)
    (
        X_train, y_train, 
        X_val, y_val,
        X_test_with_index, y_test, 
        test_data
    ) = prepare_data(prepared, targets, features)

    X_train, y_train = remove_rows_with_missing_outputs(X_train, y_train)
    X_val, y_val = remove_rows_with_missing_outputs(X_val, y_val)
    X_test_with_index, y_test, test_data = remove_rows_with_missing_outputs(X_test_with_index, y_test, test_data)

    best_params, best_score, cv_results_dict = hyperparameter_search(X_train, y_train, X_val, y_val, targets)
    visualize_multiple_hyperparam_searches(cv_results_dict, run_id)

    return {
        "features": features,
        "targets": targets,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test_with_index": X_test_with_index,
        "y_test": y_test,
        "test_data": test_data,
        "best_params": best_params,
        "best_score": best_score,
        "trained": True
    }

def test_xgb(session_state, run_id):
    X_test_with_index = session_state["X_test_with_index"]
    y_test = session_state["y_test"]
    model = session_state["model"]

    logging.info("Testing the model...")
    preds = test_xgb_autoregressively(
        model, X_test_with_index, y_test
    )
    session_state["preds"] = preds
    save_metrics(run_id, y_test, preds)

    return preds


def plot_xgb(session_state, run_id):
    model = session_state["model"]
    features = session_state["features"]
    targets = session_state["targets"]
    X_test_with_index = session_state["X_test_with_index"]
    y_test = session_state["y_test"]
    preds = session_state["preds"]
    test_data = session_state["test_data"]

    plot_scatter(run_id, test_data, y_test, preds, targets)
    plot_scatter(run_id, test_data, y_test, preds, targets, use_log=True)
    plot_shap(run_id, model, X_test_with_index, features, targets)


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
        session_state = train_xgb(run_id)
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
        X_train = session_state["X_train"]
        y_train = session_state["y_train"]
        X_val = session_state["X_val"]
        y_val = session_state["y_val"]
        targets = session_state["targets"]
        
        best_params, best_score, cv_results_dict = hyperparameter_search(X_train, y_train, X_val, y_val, targets)
        visualize_multiple_hyperparam_searches(cv_results_dict, run_id)

        session_state["best_params"] = best_params
        session_state["best_score"] = best_score
        #################
        save_session_state(session_state, run_id)


if __name__ == "__main__":
    main()