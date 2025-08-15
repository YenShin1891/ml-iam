"""TFT dataset building and management functions."""

import logging
import os
from typing import Dict, Tuple

import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet

from configs.config import RESULTS_PATH
from configs.models import TFTDatasetConfig


def build_datasets(session_state: Dict) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """Build train/val TimeSeriesDataSet objects using shared template logic."""
    val_data = session_state["val_data"]

    train_dataset, _ = create_train_dataset(session_state)
    val_dataset = from_train_template(train_dataset, val_data, mode="eval")

    return train_dataset, val_dataset


def create_train_dataset(session_state: Dict) -> Tuple[TimeSeriesDataSet, TFTDatasetConfig]:
    """Create training dataset with configuration."""
    train_data = session_state["train_data"]
    features = session_state["features"]
    targets = session_state["targets"]
    
    config = TFTDatasetConfig()
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