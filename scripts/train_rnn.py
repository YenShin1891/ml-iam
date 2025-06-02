import argparse
import logging
import numpy as np
import tensorflow as tf
import os
import sys
import pandas as pd
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))  # /root/ssd/home/ml-iam
sys.path.append(project_root)

from src.data.preprocess import load_and_process_data, prepare_features_and_targets
from src.trainers.rnn_trainer import train_rnn, train_rnn_per_target
from src.trainers.evaluation import test_rnn
from src.utils.utils import setup_logging, save_session_state, load_session_state, get_next_run_id
from src.utils.plotting import plot_scatter, plot_time_series

np.random.seed(0)
tf.random.set_seed(0)


def prepare_data(prepared, targets, features):
    """
    Prepare the data for training and testing.
    """
    # exp_name = "Test with " + model
    # condition = (prepared.Model == model)
    # if region is not None:
    #     exp_name += ", " + region
    #     condition &= (prepared.Region == region)
    # if scenario_category is not None:
    #     exp_name += ", " + scenario_category
    #     condition &= (prepared.Scenario_Category == scenario_category)

    # train_data = prepared[~condition]
    # test_data = prepared[condition]

    # train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42) # TODO: is random split the best choice?
    # display(train_data.head())
    # display(val_data.head())

    # group by Model, Scenario, Region and split train/val/test by 80/10/10
    # groups should be considered as one group and shouldn't be split
    groups = list(prepared.groupby(['Model', 'Scenario', 'Region']))
    n_groups = len(groups)
    n_test_groups = int(n_groups * 0.1)
    n_val_groups = int(n_groups * 0.1)
    n_train_groups = n_groups - n_test_groups - n_val_groups

    # shuffle groups
    np.random.shuffle(groups)

    train_groups = groups[:n_train_groups]
    val_groups = groups[n_train_groups:n_train_groups + n_val_groups]
    test_groups = groups[n_train_groups + n_val_groups:]
    print(f"train_groups: {len(train_groups)}, val_groups: {len(val_groups)}, test_groups: {len(test_groups)}")


    train_data = pd.concat([group[1] for group in train_groups])
    print(f"train_data shape: {train_data.shape}")
    val_data = pd.concat([group[1] for group in val_groups])
    print(f"val_data shape: {val_data.shape}")
    test_data = pd.concat([group[1] for group in test_groups])
    print(f"test_data shape: {test_data.shape}")

    # ungroup
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    test_data = test_data.head(10000) # TODO: remove this line to use all test data

    X_train = train_data[features].copy()
    y_train = train_data[targets].values.copy()
    X_val = val_data[features].copy()
    y_val = val_data[targets].values.copy()
    X_train = X_train.fillna(-1.0)
    X_val = X_val.fillna(-1.0)

    X_test = test_data[features].copy()
    X_test = X_test.fillna(-1.0)
    X_test_with_index = test_data[[col for col in prepared.columns if col not in targets]].copy()

    y_test = test_data[targets].values.copy()

    # convert categorical columns to cat codes
    for col in ['Region', 'Model_Family']:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('category').cat.codes
        if col in X_val.columns:
            X_val[col] = X_val[col].astype('category').cat.codes
        if col in X_test.columns:
            X_test[col] = X_test[col].astype('category').cat.codes
        if col in X_test_with_index.columns:
            X_test_with_index[col] = X_test_with_index[col].astype('category').cat.codes

    return X_train, y_train, X_val, y_val, X_test, X_test_with_index, y_test, test_data

def process_data():
    data = load_and_process_data()
    prepared, features, targets = prepare_features_and_targets(data)
    X_train, y_train, X_val, y_val, X_test, X_test_with_index, y_test, test_data = prepare_data(prepared, targets, features)

    return {
        "features": features,
        "targets": targets,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "X_test_with_index": X_test_with_index,
        "y_test": y_test,
        "test_data": test_data
    }


def train(session_state, run_id):
    X_train = session_state["X_train"]
    y_train = session_state["y_train"]
    X_val = session_state["X_val"]
    y_val = session_state["y_val"]
    targets = session_state["targets"]

    logging.info("Training RNN models...")

    if np.isnan(y_train).any():
        models = train_rnn_per_target(X_train, X_val, y_train, y_val, targets)
    else:
        model, x_scaler, y_scaler = train_rnn(X_train, X_val, y_train, y_val)
        models = {target: model for target in targets}
        session_state["x_scaler"] = x_scaler
        session_state["y_scaler"] = y_scaler

    session_state["models"] = models
    logging.info("Training complete.")
    return models


def test(session_state, run_id):
    logging.info("Testing RNN models autoregressively...")

    preds = test_rnn(
        session_state["models"],
        index_columns=['Model', 'Scenario', 'Region'],
        non_feature_columns=['Model', 'Scenario', 'Scenario_Category'],
        X_test_with_index=session_state["X_test_with_index"],
        y_test=session_state["y_test"],
        target_names=session_state["targets"],
        X_train=session_state["X_train"]
    )

    session_state["preds"] = preds
    return preds


def plot(session_state, run_id):
    plot_scatter(
        test_data=session_state["test_data"],
        y_test=session_state["y_test"],
        preds=session_state["preds"],
        targets=session_state["targets"]
    )

    plot_time_series(
        test_data=session_state["test_data"],
        y_test=session_state["y_test"],
        preds=session_state["preds"],
        targets=session_state["targets"]
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and test RNN model.")
    parser.add_argument("--run_id", type=str, help="Run ID for logging.", required=False)
    args = parser.parse_args()
    return args.run_id


def main():
    run_id_arg = parse_arguments()

    if run_id_arg is None:
        run_id = get_next_run_id()
        setup_logging(run_id)
        session_state = process_data()
        save_session_state(session_state, run_id)

        train(session_state, run_id)
        save_session_state(session_state, run_id)

        test(session_state, run_id)
        save_session_state(session_state, run_id)

        plot(session_state, run_id)
    else:
        run_id = run_id_arg
        setup_logging(run_id)
        session_state = load_session_state(run_id)

        test(session_state, run_id)
        save_session_state(session_state, run_id)

        plot(session_state, run_id)


if __name__ == "__main__":
    main()
