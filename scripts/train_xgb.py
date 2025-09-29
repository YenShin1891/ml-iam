import argparse
import logging
import numpy as np
import pandas as pd

from src.data.preprocess import prepare_data, load_and_process_data, prepare_features_and_targets
from src.trainers.xgb_trainer import hyperparameter_search, train_and_save_model
from src.trainers.evaluation import test_xgb_autoregressively, save_metrics
from src.utils.utils import setup_logging, save_session_state, load_session_state, load_model, get_next_run_id, load_best_params
from src.visualization import plot_scatter, plot_xgb_shap

np.random.seed(0)

def preprocessing(run_id, dataset=None):
    data = load_and_process_data(version=dataset)
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


def search_xgb(session_state, run_id):
    """Run hyperparameter search and store best_params in session_state."""
    logging.info("Starting hyperparameter search for XGBoost...")
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
    
    best_params, all_results = hyperparameter_search(
        X_train, y_train, X_train_with_index, train_groups,
        targets, run_id, start_stage=1, use_cv=False,
        X_val=X_val, y_val=y_val, X_val_with_index=X_val_with_index, val_groups=val_groups,
        use_dask=False
    )
    session_state["best_params"] = best_params
    logging.info("Hyperparameter search complete. Best params: %s", best_params)
    return best_params


def train_xgb(session_state, run_id):
    """Final training using best_params already present in session_state."""
    if "best_params" not in session_state:
        logging.info("best_params not searched from search step. Loading from external best_params.json.")
        session_state = load_best_params(session_state)
    best_params = session_state["best_params"]
    logging.info("Starting final XGBoost training with best params: %s", best_params)
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
    
    X_combined = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    y_combined = np.concatenate([y_train, y_val], axis=0)
    X_combined_with_index = pd.concat([X_train_with_index, X_val_with_index], axis=0, ignore_index=True)
    combined_groups = np.concatenate([train_groups, val_groups], axis=0)
    
    train_and_save_model(X_combined, y_combined, combined_groups, targets, best_params, run_id, use_dask=False)
    logging.info("Final XGBoost training complete.")
    return best_params

def test_xgb(session_state, run_id):
    X_test_with_index = session_state["X_test_with_index"]
    y_test = session_state["y_test"]
    test_data = session_state["test_data"]
    logging.info("Testing the model...")
    preds = test_xgb_autoregressively(X_test_with_index, y_test, run_id)
    session_state["preds"] = preds
    save_metrics(run_id, y_test, preds, test_data)

    return preds


def plot_xgb(session_state, run_id):
    features = session_state["features"]
    targets = session_state["targets"]
    X_test_with_index = session_state["X_test_with_index"]
    y_test = session_state["y_test"]
    preds = session_state["preds"]
    test_data = session_state["test_data"]

    plot_scatter(run_id, test_data, y_test, preds, targets, model_name="XGBoost")
    plot_xgb_shap(run_id, X_test_with_index, features, targets)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and test XGBoost model.")
    parser.add_argument("--run_id", type=str, help="Run ID for logging.", required=False)
    parser.add_argument(
        "--resume",
        type=str,
        choices=["search", "train", "test", "plot"],
        help="Resume from a specific step. Requires --run_id to be specified.",
        required=False,
    )
    parser.add_argument(
        "--note",
        type=str,
        help="Note describing the run condition/type for later reference.",
        required=False,
    )
    parser.add_argument(
        "--skip_search",
        action="store_true",
        help="Skip hyperparameter search step when running full pipeline.",
        required=False,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset subdirectory name to use for processed_series.csv. Falls back to DEFAULT_DATASET if not specified.",
        required=False,
    )
    args = parser.parse_args()
    
    # Validation: if resume is specified, run_id must be provided
    if args.resume and not args.run_id:
        parser.error("--run_id is required when --resume is specified")
    
    # Validation: if resume is not specified, run_id should not be provided
    if not args.resume and args.run_id:
        parser.error("--run_id should only be specified when using --resume")
    
    return args.run_id, args.resume, args.note, args.skip_search, args.dataset


def main():
    run_id, resume, note, skip_search, dataset = parse_arguments()

    if resume is None:
        # Full pipeline: process -> search -> train -> test -> plot
        run_id = get_next_run_id()
        setup_logging(run_id)
        session_state = preprocessing(run_id, dataset)
        save_session_state(session_state, run_id)
        resume = "train" if skip_search else "search"  # Start from train step if skipping search
    else:
        setup_logging(run_id)
        session_state = load_session_state(run_id)

    if note:
        session_state["note"] = note
        logging.info("Run note: %s", note)

    pipeline_steps = ["search", "train", "test", "plot"]
    step_functions = {
        "search": search_xgb,
        "train": train_xgb,
        "test": test_xgb,
        "plot": plot_xgb
    }
    start_index = pipeline_steps.index(resume)
    for step in pipeline_steps[start_index:]:
        step_functions[step](session_state, run_id)
        save_session_state(session_state, run_id)


if __name__ == "__main__":
    main()