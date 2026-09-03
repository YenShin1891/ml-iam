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
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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
        return self.root / "cache"

    def _artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @staticmethod
    def _writable(path: Path) -> Path:
        """*path* with its directory in place; only the save_* methods create one.

        Probing a run (has_*, load_*) must not create it: get_next_run_id
        numbers runs by the directories that exist, so a mistyped dashboard
        URL would otherwise reserve a run id.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # Cache: expensive preprocessing (melt + pivot_table)
    # ------------------------------------------------------------------

    def save_processed_data(self, df: pd.DataFrame) -> None:
        path = self._writable(self._cache_dir() / "processed_data.parquet")
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
        path = self._writable(self._artifacts_dir() / "best_params.json")
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
        path = self._writable(self._artifacts_dir() / "features.json")
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
    # Category vocabularies (JSON — the integer codes the models were fit with)
    # ------------------------------------------------------------------

    def save_categories(self, categories: Dict[str, List[str]]) -> None:
        path = self._writable(self._artifacts_dir() / "categories.json")
        payload = {col: list(values) for col, values in categories.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logging.info(
            "Saved category vocabularies to %s (%s)",
            path,
            ", ".join(f"{col}={len(v)}" for col, v in payload.items()) or "empty",
        )

    def load_categories(self) -> Dict[str, List[str]]:
        path = self._artifacts_dir() / "categories.json"
        if not path.exists():
            raise FileNotFoundError(f"No categories.json found at {path}.")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def has_categories(self) -> bool:
        return (self._artifacts_dir() / "categories.json").exists()

    def categories_for(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """The vocabularies this run encodes with, saving them on first use.

        Codes have to stay identical across phases, models and processes: the
        dashboard and the inference script decode with them long after training.
        Runs made before categories.json existed derive it from their cached
        data on the next phase.
        """
        from src.data.preprocess import resolve_categorical_vocabularies

        persisted = (
            self.load_categories() if self.has_categories()
            else self._legacy_categories()
        )
        categories = resolve_categorical_vocabularies(data, persisted)
        if categories != persisted:
            self.save_categories(categories)
        return categories

    def _legacy_categories(self) -> Optional[Dict[str, List[str]]]:
        """Vocabularies from runs that predate categories.json.

        A trained model's embedding rows are indexed by the codes it saw, so a
        resumed phase has to keep using them rather than renumber from the
        current data.
        """
        if not self.has_train_meta():
            return None

        meta = self.load_train_meta()
        legacy = dict(meta.get("xgb_categories") or {})
        model_families = meta.get("lstm_model_family_categories")
        if model_families:
            legacy.setdefault("Model_Family", list(model_families))

        if legacy:
            logging.info(
                "Seeding category vocabularies for run %s from train_meta: %s",
                self.run_id, ", ".join(sorted(legacy)),
            )
        return legacy or None

    # ------------------------------------------------------------------
    # Train/val/test assignment (parquet — one row per group)
    # ------------------------------------------------------------------

    def save_splits(self, assignment: pd.DataFrame) -> None:
        path = self._writable(self._artifacts_dir() / "splits.parquet")
        assignment.to_parquet(path, index=False)
        counts = assignment["split"].value_counts().to_dict()
        logging.info("Saved split assignment (%d groups: %s) to %s", len(assignment), counts, path)

    def load_splits(self) -> pd.DataFrame:
        path = self._artifacts_dir() / "splits.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No splits.parquet found at {path}.")
        return pd.read_parquet(path)

    def has_splits(self) -> bool:
        return (self._artifacts_dir() / "splits.parquet").exists()

    def splits_for(self, data: pd.DataFrame) -> pd.DataFrame:
        """This run's group -> split assignment, computing it on first use.

        Shared by every model in the run so their test sets are identical, and
        persisted so a later phase cannot re-derive a different one.
        """
        from src.data.preprocess import assign_group_splits

        if self.has_splits():
            return self.load_splits()

        assignment = self._legacy_splits(data)
        if assignment is None:
            assignment = assign_group_splits(data)
        self.save_splits(assignment)
        return assignment

    def _identity_in_labels(self, test_data: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """The saved test rows' identity columns, in *data*'s own terms.

        The sequence models save their test frame after encoding categoricals
        for their embeddings, so Region comes back as integer codes that cannot
        be merged against the labels in the processed data.  Decoding is the
        only safe repair: matching the codes as strings would find nothing,
        leave the run with no test groups, and silently re-split a run whose
        results are already published.
        """
        from configs.data import INDEX_COLUMNS
        from src.data.preprocess import decode_categorical_column

        keys = test_data[list(INDEX_COLUMNS)].copy()
        encoded = [
            col for col in INDEX_COLUMNS
            if pd.api.types.is_numeric_dtype(keys[col])
            and col in data.columns
            and not pd.api.types.is_numeric_dtype(data[col])
        ]
        if not encoded:
            return keys

        vocabularies = self.categories_for(data)
        for col in encoded:
            vocabulary = vocabularies.get(col)
            if not vocabulary:
                raise ValueError(
                    f"Run {self.run_id} saved {col} as integer codes and has no vocabulary "
                    f"to decode them, so its original test groups cannot be recovered. "
                    "Re-run the preprocess phase to write artifacts/splits.parquet."
                )
            keys[col] = decode_categorical_column(keys[col], vocabulary)
            logging.info(
                "Decoded %s from the saved test data to recover run %s's split.",
                col, self.run_id,
            )
        return keys

    def _legacy_splits(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Recover a pre-existing run's split from the test rows it saved.

        Runs made before splits.parquet derived their split from each model's
        own filtered frame.  Recomputing it here could move a group the run
        trained on into test and quietly inflate its metrics, so pin the test
        groups to the ones the run actually evaluated.  Train and val among the
        remaining groups are reassigned, which is harmless: neither is scored.
        """
        from configs.data import INDEX_COLUMNS
        from src.data.preprocess import assign_group_splits

        if self.has_splits() or not self.has_test_data():
            return None

        try:
            test_data, _ = self.load_test_data()
        except Exception as e:  # noqa: BLE001
            logging.warning("Could not read saved test data to recover the split: %s", e)
            return None

        if not all(col in test_data.columns for col in INDEX_COLUMNS):
            return None

        keys = data[INDEX_COLUMNS].drop_duplicates()
        test_keys = self._identity_in_labels(test_data, data).drop_duplicates()
        is_test = (
            keys.merge(test_keys.assign(_test=True), on=list(INDEX_COLUMNS), how="left")["_test"]
            .notna()
            .to_numpy()
        )

        remaining = assign_group_splits(keys[~is_test], test_size=0.0, val_size=0.1)
        assignment = pd.concat(
            [keys[is_test].assign(split="test"), remaining], ignore_index=True
        )
        logging.warning(
            "Run %s predates splits.parquet: recovered its %d test groups from the "
            "saved test data and reassigned train/val over the rest.",
            self.run_id, int(is_test.sum()),
        )
        return assignment.sort_values(list(INDEX_COLUMNS)).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Train metadata (JSON — LSTM encoded features, sequence_length, etc.)
    # ------------------------------------------------------------------

    def save_train_meta(self, meta: dict) -> None:
        path = self._writable(self._artifacts_dir() / "train_meta.json")
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
    # Test data (for dashboard — avoids expensive re-derivation)
    # ------------------------------------------------------------------

    def save_test_data(self, test_data: pd.DataFrame, y_test) -> None:
        """Save test split and targets for dashboard use."""
        td_path = self._writable(self._cache_dir() / "test_data.parquet")
        test_data.to_parquet(td_path, index=False)
        yt_path = self._cache_dir() / "y_test.npy"
        np.save(yt_path, y_test)
        logging.info("Saved test_data (%d rows) and y_test to %s", len(test_data), self._cache_dir())

    def load_test_data(self):
        """Load cached test split and targets."""
        td_path = self._cache_dir() / "test_data.parquet"
        yt_path = self._cache_dir() / "y_test.npy"
        if not (td_path.exists() and yt_path.exists()):
            raise FileNotFoundError(f"No cached test data under {self._cache_dir()}. Run the test phase first.")
        test_data = pd.read_parquet(td_path)
        y_test = np.load(yt_path)
        return test_data, y_test

    def has_test_data(self) -> bool:
        return (self._cache_dir() / "test_data.parquet").exists() and (self._cache_dir() / "y_test.npy").exists()

    # ------------------------------------------------------------------
    # Predictions (pickle — numpy arrays may contain NaN)
    # ------------------------------------------------------------------

    def save_predictions(
        self,
        preds,
        horizon_df: Optional[pd.DataFrame] = None,
        horizon_y_true=None,
    ) -> None:
        path = self._writable(self._artifacts_dir() / "predictions.pkl")
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
        path = self._writable(self._artifacts_dir() / name)
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
