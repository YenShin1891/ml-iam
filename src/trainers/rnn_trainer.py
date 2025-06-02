from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.layers import Masking
import pandas as pd
import numpy as np
np.random.seed(0)
from src.utils.utils import masked_mse
import os
from matplotlib import pyplot as plt
from matplotlib import cm
import streamlit as st

import shap
import time
from IPython.display import display, HTML
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.saving import register_keras_serializable

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Masking, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_rnn(X_train, X_valid, y_train, y_valid):
    # Ensure 2D shape for targets
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    if y_valid.ndim == 1:
        y_valid = y_valid.reshape(-1, 1)

    # Fill NaNs in X and convert to float32
    X_train = X_train.fillna(-1.0).astype(np.float32)
    X_valid = X_valid.fillna(-1.0).astype(np.float32)

    # Standardize X
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train).reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_valid_scaled = x_scaler.transform(X_valid).reshape((X_valid.shape[0], 1, X_valid.shape[1]))

    # Standardize y
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_valid_scaled = y_scaler.transform(y_valid)

    # Build model
    model = Sequential([
        Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
        Masking(mask_value=-1.0),
        LSTM(64, return_sequences=False, activation='tanh', recurrent_activation='sigmoid', implementation=1, unroll=True),
        Dense(64, activation='relu'),
        Dense(y_train.shape[1])
    ])
    model.compile(optimizer='adam', loss=masked_mse)

    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train
    print("Training RNN...")
    model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_valid_scaled, y_valid_scaled),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2
    )

    return model, x_scaler, y_scaler


def train_rnn_per_target(X_train, X_valid, y_train, y_valid, target_names):
    models = {}
    for i, target in enumerate(target_names):
        # Filter rows where the current target is not NaN
        valid_rows = ~np.isnan(y_train[:, i])
        X_train_filtered = X_train[valid_rows]
        y_train_filtered = y_train[valid_rows, i]

        valid_rows_val = ~np.isnan(y_valid[:, i])
        X_valid_filtered = X_valid[valid_rows_val]
        y_valid_filtered = y_valid[valid_rows_val, i]

        print(f"Training model for target: {target}")

        # Train an RNN model for each target
        model,_,_ = train_rnn(X_train_filtered, X_valid_filtered, y_train_filtered, y_valid_filtered)
        models[target] = model
    return models
