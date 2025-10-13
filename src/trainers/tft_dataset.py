"""TFT dataset building and management functions."""

import logging
import os
from typing import Dict, Tuple, List, Any

import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet

from configs.paths import RESULTS_PATH
from configs.models.tft import TFTDatasetConfig
from configs.data import CATEGORICAL_COLUMNS, INDEX_COLUMNS


# --- Categorical encoder construction (restored from working logic) ---
try:
    from pytorch_forecasting.data.encoders import NaNLabelEncoder  # type: ignore
except Exception:  # pragma: no cover
    NaNLabelEncoder = None  # type: ignore


def _ordered_categorical_cols(features: List[str]) -> List[str]:
    """Deterministic order: group ids + static categoricals + indicator columns."""
    static_cols = list(INDEX_COLUMNS) + [c for c in CATEGORICAL_COLUMNS if c in features]
    indicator_cols = [f for f in features if f.endswith("_is_missing")]
    ordered = list(dict.fromkeys(static_cols + indicator_cols))
    return ordered


def _build_union_encoders(session_state: Dict, categorical_cols: List[str], add_nan: bool = False) -> Dict[str, Any]:
    """Fit NaNLabelEncoder with a closed vocabulary aggregated across splits."""
    if NaNLabelEncoder is None:
        logging.warning("NaNLabelEncoder unavailable; skipping pretrained categorical encoders.")
        return {}
    dfs = [session_state.get("train_data"), session_state.get("val_data"), session_state.get("test_data")]
    df_all = pd.concat([df for df in dfs if df is not None], axis=0, ignore_index=True)
    encoders: Dict[str, Any] = {}
    # ensure deterministic iteration order
    for col in categorical_cols:
        if col in df_all.columns:
            s_raw = df_all[col].astype(str).fillna("__NA__")
            # Explicit, deterministic category order
            categories = sorted(pd.unique(s_raw))
            s = pd.Series(pd.Categorical(s_raw, categories=categories, ordered=True))
        else:
            # if column is entirely missing in some split, still create a closed-vocab encoder
            s = pd.Series(pd.Categorical(["__NA__"], categories=["__NA__"], ordered=True))
        enc = NaNLabelEncoder(add_nan=add_nan)
        enc.fit(s)
        encoders[col] = enc
    return encoders


def build_datasets(session_state: Dict) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """Build train/val TimeSeriesDataSet objects using shared template logic (encoders handle categoricals)."""
    val_data = session_state["val_data"].copy()
    train_dataset, _ = create_train_dataset(session_state)
    val_dataset = from_train_template(train_dataset, val_data, mode="eval")
    return train_dataset, val_dataset


def create_train_dataset(session_state: Dict) -> Tuple[TimeSeriesDataSet, TFTDatasetConfig]:
    """Create training dataset with configuration, coercing categorical-like columns first."""
    train_data = session_state["train_data"].copy()
    features = session_state["features"]
    targets = session_state["targets"]

    config = TFTDatasetConfig()

    # Build union encoders (include group ids to stabilize mapping) then inject
    categorical_cols = _ordered_categorical_cols(features)
    pretrained_encoders = _build_union_encoders(session_state, categorical_cols, add_nan=False)
    config.pretrained_categorical_encoders = pretrained_encoders

    params = config.build(features, targets, mode="train")

    train_dataset = TimeSeriesDataSet(train_data, **params)
    return train_dataset, config


def from_train_template(
    train_dataset: TimeSeriesDataSet, 
    data: pd.DataFrame, 
    mode: str = "eval"
) -> TimeSeriesDataSet:
    """Create dataset from training template."""
    return TimeSeriesDataSet.from_dataset(
        train_dataset, 
        data, 
        stop_randomization=(mode == "eval"),
        predict=(mode == "predict")
    )


def save_dataset_template(dataset: TimeSeriesDataSet, run_id: str) -> str:
    """Save dataset template for later use."""
    final_dir = os.path.join(RESULTS_PATH, run_id, "final")
    os.makedirs(final_dir, exist_ok=True)
    dataset_tpl_path = os.path.join(final_dir, "dataset_template.pt")
    
    try:
        torch.save(dataset, dataset_tpl_path)
        logging.info("Saved TFT dataset template to %s", dataset_tpl_path)
        return dataset_tpl_path
    except Exception as e:
        raise RuntimeError(f"Failed to save dataset template to {dataset_tpl_path}: {e}")


def load_dataset_template(run_id: str) -> TimeSeriesDataSet:
    """Load saved dataset template."""
    final_dir = os.path.join(RESULTS_PATH, run_id, "final")
    dataset_tpl_path = os.path.join(final_dir, "dataset_template.pt")
    
    if not os.path.exists(dataset_tpl_path):
        raise FileNotFoundError(
            f"Dataset template not found at {dataset_tpl_path}. Run train_final_tft to generate it."
        )
    
    try:
        from pytorch_forecasting.data.timeseries._timeseries import TimeSeriesDataSet as _PFTimeSeriesDataSet
    except Exception:
        from pytorch_forecasting.data.timeseries import TimeSeriesDataSet as _PFTimeSeriesDataSet
    
    try:
        import torch.serialization as _ts
        if hasattr(_ts, "add_safe_globals"):
            _ts.add_safe_globals([_PFTimeSeriesDataSet])
        
        try:
            template = torch.load(dataset_tpl_path, map_location="cpu")
        except Exception as inner_e:
            try:
                template = torch.load(dataset_tpl_path, map_location="cpu", weights_only=False)
                logging.info("Loaded dataset template with weights_only=False due to prior failure: %s", inner_e)
            except Exception as retry_e:
                raise RuntimeError(
                    "Failed to load dataset template even after retry with weights_only=False. "
                    f"Original error: {inner_e}; Retry error: {retry_e}"
                ) from retry_e
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset template from {dataset_tpl_path}: {e}")
    
    return template


def create_combined_dataset(
    train_dataset: TimeSeriesDataSet,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame
) -> TimeSeriesDataSet:
    """Create combined train+val dataset."""
    combined_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    return TimeSeriesDataSet.from_dataset(train_dataset, combined_df)


def create_dataset_with_custom_encoders(
    session_state: Dict,
    custom_encoders: Dict[str, Any]
) -> TimeSeriesDataSet:
    """Create a TFT dataset using custom categorical encoders.

    Args:
        session_state: Dictionary containing training data, features, and targets
        custom_encoders: Dictionary of pre-trained categorical encoders

    Returns:
        TimeSeriesDataSet configured with the custom encoders
    """
    train_data = session_state["train_data"].copy()
    features = session_state["features"]
    targets = session_state["targets"]

    # Create config with custom encoders - no monkey patching needed!
    config = TFTDatasetConfig()
    config.pretrained_categorical_encoders = custom_encoders

    params = config.build(features, targets, mode="train")

    return TimeSeriesDataSet(train_data, **params)