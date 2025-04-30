import logging
import numpy as np

from src.data.preprocess import prepare_data, load_and_process_data, prepare_features_and_targets, remove_rows_with_missing_outputs
from src.trainers.xgb_trainer import hyperparameter_search, visualize_multiple_hyperparam_searches
from src.trainers.evaluation import test_xgb_autoregressively, save_metrics
from src.utils.utils import setup_logging, save_session_state, load_session_state, get_next_run_id
from utils.plotting import plot_scatter, plot_shap

np.random.seed(0)

def train_xgb(run_id):
    data = load_and_process_data()
    prepared, features, targets = prepare_features_and_targets(data)
    (
        X_train, y_train, 
        X_test_with_index, y_test, 
        test_data
    ) = prepare_data(prepared, targets, features)

    X_train, y_train = remove_rows_with_missing_outputs(X_train, y_train)
    X_test_with_index, y_test, test_data = remove_rows_with_missing_outputs(X_test_with_index, y_test, test_data)

    assert not np.any(np.isnan(y_train)), "y_train contains NaN values."
    assert not np.any(np.isinf(y_train)), "y_train contains Inf values."

    best_model, cv_results = hyperparameter_search(X_train, y_train)
    visualize_multiple_hyperparam_searches(cv_results, run_id)

    return {
        "features": features,
        "targets": targets,
        "X_train": X_train,
        "y_train": y_train,
        "X_test_with_index": X_test_with_index,
        "y_test": y_test,
        "test_data": test_data,
        "model": best_model,
        # "preds": preds,
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


def plot_xgb(session_state):
    model = session_state["model"]
    features = session_state["features"]
    targets = session_state["targets"]
    X_test_with_index = session_state["X_test_with_index"]
    y_test = session_state["y_test"]
    preds = session_state["preds"]
    test_data = session_state["test_data"]

    plot_scatter(test_data, y_test, preds, targets)
    plot_scatter(test_data, y_test, preds, targets, use_log=True)
    plot_shap(model, X_test_with_index, features, targets)

def main():
    run_id = get_next_run_id()
    setup_logging(run_id)

    full_pipeline = True
    if full_pipeline:
        session_state = train_xgb(run_id)
        save_session_state(session_state, run_id)
        preds = test_xgb(session_state, run_id)
        session_state["preds"] = preds
        save_session_state(session_state, run_id)
        plot_xgb(session_state)
    else:
        session_state = load_session_state("session_state1.pkl")
        ### implement ###
        preds = test_xgb(session_state, "run_debug")
        session_state["preds"] = preds
        save_session_state(session_state, "run_debug")

        plot_xgb(session_state)
        # plot_shap(None, X_test_with_index, features, targets)
        #################
        # save_session_state(session_state, "session_state1.pkl")


if __name__ == "__main__":
    main()