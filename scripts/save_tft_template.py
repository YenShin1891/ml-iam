"""
Script to retroactively save dataset templates for existing TFT runs.
This enables SHAP analysis on older runs that were trained before template saving was implemented.

Usage:
    python scripts/save_tft_template.py --run_id run_41
"""

import argparse
import logging
import os
import torch
from pathlib import Path

from src.utils.utils import load_session_state, setup_logging
from src.trainers.tft_trainer import build_datasets
from configs.config import RESULTS_PATH


def save_template_for_run(run_id):
    """Save dataset template for an existing TFT run."""
    # Check if run exists
    run_dir = Path(RESULTS_PATH) / run_id
    if not run_dir.exists():
        raise ValueError(f"Run directory {run_dir} does not exist")

    # Check if template already exists
    template_path = run_dir / "checkpoints" / "tft_dataset_template.pt"
    if template_path.exists():
        logging.info("Template already exists at %s", template_path)
        return str(template_path)

    # Load session state
    logging.info("Loading session state for %s", run_id)
    session_state = load_session_state(run_id)

    if not session_state:
        raise ValueError(f"No session state found for run {run_id}")

    # Check required data
    required_keys = ["features", "targets", "train_data", "val_data", "test_data"]
    missing_keys = [key for key in required_keys if key not in session_state]
    if missing_keys:
        raise ValueError(f"Missing required keys in session state: {missing_keys}")

    # Build datasets (this creates the template)
    logging.info("Building datasets to create template...")
    train_dataset, val_dataset = build_datasets(session_state)

    # Create combined dataset for template using the original DataFrames from session_state
    import pandas as pd
    train_df = session_state.get("train_data")
    val_df = session_state.get("val_data")

    if not isinstance(train_df, pd.DataFrame) or not isinstance(val_df, pd.DataFrame):
        raise ValueError("train_data and val_data must be pandas DataFrames")

    # Ensure unique pandas index to satisfy TimeSeriesDataSet validation
    combined_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

    # Reuse train template to freeze feature schema/encoders identically
    from pytorch_forecasting import TimeSeriesDataSet
    combined_dataset = TimeSeriesDataSet.from_dataset(train_dataset, combined_df)

    # Save template
    os.makedirs(template_path.parent, exist_ok=True)
    torch.save(combined_dataset, template_path)
    logging.info("Saved dataset template to %s", template_path)

    return str(template_path)


def main():
    parser = argparse.ArgumentParser(description="Save dataset template for existing TFT run")
    parser.add_argument("--run_id", type=str, required=True, help="Run ID to save template for")
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.run_id, log_file="save_template.log")

    try:
        template_path = save_template_for_run(args.run_id)
        logging.info("Successfully saved template: %s", template_path)
        print(f"Template saved to: {template_path}")
    except Exception as e:
        logging.error("Failed to save template: %s", str(e))
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()