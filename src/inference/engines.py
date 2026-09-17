"""CPU inference adapters for the dashboard's saved TFT, LSTM and XGB runs."""

from dataclasses import dataclass
import re
import threading

import numpy as np
import pandas as pd

from configs.data import INDEX_COLUMNS, MAX_CONTEXT_LENGTH
from src.utils.run_store import RunStore

_PREDICT_LOCK = threading.Lock()


@dataclass
class TabularEngine:
    run_id: str
    model: object
    features: list
    targets: list
    categories: dict
    x_scaler: object
    y_scaler: object
    encoder_length: int
    metadata: dict

    @property
    def min_steps(self):
        return self.encoder_length + 1


def load_engine(run_id, *, map_location="cpu"):
    kind = run_id.split("_", 1)[0]
    if kind == "tft":
        from src.inference.tft_predict import load_engine as load_tft
        return load_tft(run_id, map_location=map_location)
    store = RunStore(run_id)
    features, targets = store.load_features()
    categories = store.load_categories()
    if kind == "lstm":
        from src.trainers.lstm_trainer import LSTMModel
        meta = store.load_train_meta()
        model = LSTMModel.load_from_checkpoint(str(store.root / "final" / "best.ckpt"), map_location="cpu")
        model.cpu().eval()
        history = MAX_CONTEXT_LENGTH - 1 + int(meta["lstm_target_offset"])
        return TabularEngine(run_id, model, features, targets, categories,
                             store.load_artifact("lstm_scaler_X.pkl"), store.load_artifact("lstm_scaler_y.pkl"), history, meta)
    if kind == "xgb":
        from src.trainers.xgb_trainer import load_final_xgb_model
        model = load_final_xgb_model(run_id, targets)
        for booster in getattr(model, "models", [model]):
            booster.set_params(device="cpu", n_jobs=2)
        lags = [int(match.group(1) or 1) for f in features if (match := re.match(r"prev(\d*)_", f))]
        if not lags:
            raise ValueError("The XGB checkpoint has no saved lag features.")
        return TabularEngine(run_id, model, features, targets, categories,
                             store.load_artifact("x_scaler.pkl"), store.load_artifact("y_scaler.pkl"),
                             max(MAX_CONTEXT_LENGTH, max(lags)), {"n_lags": max(lags)})
    raise ValueError(f"Unsupported emulator: {kind}")


def check_vocabulary(engine, rows):
    if engine.run_id.startswith("tft_"):
        from src.inference.tft_predict import check_vocabulary as check_tft
        return check_tft(engine, rows)
    return sorted(f"{column}={label}" for column, labels in engine.categories.items()
                  if column in rows for label in rows[column].dropna().unique() if label not in labels)


def predict_windows(engine, frame):
    if engine.run_id.startswith("tft_"):
        from src.inference.tft_predict import predict_windows as predict_tft
        return predict_tft(engine, frame)
    problems = check_vocabulary(engine, frame)
    if problems:
        raise ValueError("Unknown category labels: " + ", ".join(problems))
    rows = frame.sort_values(list(INDEX_COLUMNS) + ["Step"]).reset_index(drop=True)
    if (rows.groupby(list(INDEX_COLUMNS), observed=True).size() < engine.min_steps).any():
        raise ValueError(f"Each trajectory needs at least {engine.min_steps} steps.")
    encoded = rows.copy()
    for column, labels in engine.categories.items():
        if column in encoded:
            encoded[column] = pd.Categorical(encoded[column], categories=labels).codes.astype("int64")
    with _PREDICT_LOCK:
        if engine.run_id.startswith("lstm_"):
            positions, predictions = _predict_lstm(engine, encoded)
        else:
            positions, predictions = _predict_xgb(engine, encoded)
    out = rows.iloc[positions][list(INDEX_COLUMNS) + ["Step", "Year"]].copy()
    out[[f"{target}_pred" for target in engine.targets]] = predictions
    return out.reset_index(drop=True)


def _predict_lstm(engine, encoded):
    import torch
    from src.trainers.lstm_trainer import LSTMDataset

    meta = engine.metadata
    categorical = meta.get("lstm_categorical_features", [])
    features = meta["lstm_features"]
    non_numeric = meta.get("lstm_non_numeric_features", [])
    if non_numeric:
        raw = [f for f in meta["lstm_raw_features"] if f not in categorical]
        values = pd.get_dummies(encoded[raw], columns=non_numeric, dummy_na=True).reindex(columns=features, fill_value=0.)
        encoded = pd.concat([encoded.drop(columns=raw), values], axis=1)
    dataset = LSTMDataset(encoded, features + categorical, engine.targets,
                          sequence_length=int(meta["lstm_sequence_length"]), target_offset=int(meta["lstm_target_offset"]),
                          scaler_X=engine.x_scaler, scaler_y=engine.y_scaler, fit_scalers=False,
                          categorical_features=categorical)
    batches = []
    with torch.inference_mode():
        for start in range(0, len(dataset), 128):
            stop = start + 128
            batches.append(engine.model(dataset.X_sequences[start:stop], dataset.masks[start:stop],
                                         cat_indices=dataset.cat_sequences[start:stop]).cpu().numpy())
    scaled = np.concatenate(batches).reshape(-1, len(engine.targets))
    return dataset.target_positions, engine.y_scaler.inverse_transform(scaled)


def _predict_xgb(engine, encoded):
    from src.trainers.evaluation import rollout_all_groups

    # The leading history supplies observed lags. Later lag slots are replaced
    # by predictions in rollout_all_groups, exactly as in the saved test phase.
    positions = []
    matrices = []
    for _, group in encoded.groupby(list(INDEX_COLUMNS), sort=False, observed=True):
        horizon = group.iloc[engine.encoder_length:]
        positions.extend(horizon.index.tolist())
        matrices.append(engine.x_scaler.transform(horizon[engine.features]))
    predictions, lengths = rollout_all_groups(engine.model, matrices, engine.features,
                                               engine.y_scaler, engine.x_scaler, engine.metadata["n_lags"])
    scaled = np.concatenate([predictions[i, :length] for i, length in enumerate(lengths)])
    return positions, engine.y_scaler.inverse_transform(scaled)
