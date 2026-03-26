"""RunStore: per-run artifact I/O.

Replaces the monolithic session_state.pkl with explicit, typed artifact files.
Only truly expensive or irreproducible outputs are persisted:
  - processed_data.parquet  (3-min melt+pivot — cached)
  - best_params.json        (hours of GPU search)
  - model checkpoints       (hours of GPU training)
  - scalers                 (fitted on train split)
  - predictions             (test output)

Cheap artifacts (splits, encoded features, imputed data) are re-derived each
phase from the cached parquet in seconds.
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.utils.utils import get_run_root


class RunStore:
    """Manages per-run artifact I/O."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.root = Path(get_run_root(run_id))

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------

    def _cache_dir(self) -> Path:
        d = self.root / "cache"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _artifacts_dir(self) -> Path:
        d = self.root / "artifacts"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # Cache: expensive preprocessing (melt + pivot_table)
    # ------------------------------------------------------------------

    def save_processed_data(self, df: pd.DataFrame) -> None:
        path = self._cache_dir() / "processed_data.parquet"
        df.to_parquet(path, index=False)
        logging.info("Saved processed data (%d rows) to %s", len(df), path)

    def load_processed_data(self) -> pd.DataFrame:
        path = self._cache_dir() / "processed_data.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"No cached processed_data found at {path}. Run the preprocess phase first."
            )
        df = pd.read_parquet(path)
        logging.info("Loaded processed data (%d rows) from %s", len(df), path)
        return df

    def has_processed_data(self) -> bool:
        return (self._cache_dir() / "processed_data.parquet").exists()

    # ------------------------------------------------------------------
    # Best params (JSON — human-readable, diffable)
    # ------------------------------------------------------------------

    def save_best_params(self, params: dict) -> None:
        path = self._artifacts_dir() / "best_params.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, default=str)
        logging.info("Saved best_params to %s", path)

    def load_best_params(self) -> dict:
        path = self._artifacts_dir() / "best_params.json"
        if not path.exists():
            raise FileNotFoundError(f"No best_params found at {path}. Run search or train first.")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def has_best_params(self) -> bool:
        return (self._artifacts_dir() / "best_params.json").exists()

    # ------------------------------------------------------------------
    # Features + targets (JSON)
    # ------------------------------------------------------------------

    def save_features(self, features: List[str], targets: List[str]) -> None:
        path = self._artifacts_dir() / "features.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"features": features, "targets": targets}, f, indent=2)
        logging.info("Saved features (%d) and targets (%d) to %s", len(features), len(targets), path)

    def load_features(self) -> Tuple[List[str], List[str]]:
        path = self._artifacts_dir() / "features.json"
        if not path.exists():
            raise FileNotFoundError(f"No features.json found at {path}.")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["features"], data["targets"]

    # ------------------------------------------------------------------
    # Train metadata (JSON — LSTM encoded features, sequence_length, etc.)
    # ------------------------------------------------------------------

    def save_train_meta(self, meta: dict) -> None:
        path = self._artifacts_dir() / "train_meta.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        logging.info("Saved train metadata to %s", path)

    def load_train_meta(self) -> dict:
        path = self._artifacts_dir() / "train_meta.json"
        if not path.exists():
            raise FileNotFoundError(f"No train_meta.json found at {path}. Run the train phase first.")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def has_train_meta(self) -> bool:
        return (self._artifacts_dir() / "train_meta.json").exists()

    # ------------------------------------------------------------------
    # Predictions (pickle — numpy arrays may contain NaN)
    # ------------------------------------------------------------------

    def save_predictions(
        self,
        preds,
        horizon_df: Optional[pd.DataFrame] = None,
        horizon_y_true=None,
    ) -> None:
        path = self._artifacts_dir() / "predictions.pkl"
        payload: Dict[str, Any] = {"preds": preds}
        if horizon_df is not None:
            payload["horizon_df"] = horizon_df
        if horizon_y_true is not None:
            payload["horizon_y_true"] = horizon_y_true
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logging.info("Saved predictions to %s", path)

    def load_predictions(self) -> Dict[str, Any]:
        path = self._artifacts_dir() / "predictions.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No predictions found at {path}. Run the test phase first.")
        with open(path, "rb") as f:
            return pickle.load(f)

    def has_predictions(self) -> bool:
        return (self._artifacts_dir() / "predictions.pkl").exists()

    # ------------------------------------------------------------------
    # Generic artifact (pickle — scalers, etc.)
    # ------------------------------------------------------------------

    def save_artifact(self, name: str, obj: Any) -> None:
        path = self._artifacts_dir() / name
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logging.info("Saved artifact '%s' to %s", name, path)

    def load_artifact(self, name: str) -> Any:
        path = self._artifacts_dir() / name
        if not path.exists():
            raise FileNotFoundError(f"Artifact '{name}' not found at {path}.")
        with open(path, "rb") as f:
            return pickle.load(f)

    def has_artifact(self, name: str) -> bool:
        return (self._artifacts_dir() / name).exists()
