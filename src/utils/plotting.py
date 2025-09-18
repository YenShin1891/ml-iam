import json
import logging
import os
import tempfile
from typing import List, Optional, Dict
import re

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import torch
from PIL import Image
from tqdm import tqdm
import glob
import streamlit as st

from configs.paths import RESULTS_PATH
from configs.data import INDEX_COLUMNS, NON_FEATURE_COLUMNS, OUTPUT_UNITS


def _to_numpy(x):
    try:
        if hasattr(x, 'detach') and callable(getattr(x, 'detach')):
            return x.detach().cpu().numpy()
        if hasattr(x, 'cpu') and callable(getattr(x, 'cpu')) and hasattr(x, 'numpy'):
            return x.cpu().numpy()
        if hasattr(x, 'numpy') and callable(getattr(x, 'numpy')):
            return x.numpy()
    except Exception:
        pass
    return np.array(x)

def preprocess_data(
    test_data: pd.DataFrame, 
    y_test: np.ndarray, 
    preds: np.ndarray, 
    target_index: int
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    # Ensure 2D arrays
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)

    # Align lengths defensively
    n = len(test_data)
    if y_test.shape[0] != n or preds.shape[0] != n:
        m = min(n, y_test.shape[0], preds.shape[0])
        test_data = test_data.iloc[:m].reset_index(drop=True)
        y_test = y_test[:m]
        preds = preds[:m]

    # Build valid mask for the requested target index
    col = min(target_index, y_test.shape[1] - 1, preds.shape[1] - 1)
    mask = (~np.isnan(y_test[:, col])) & (~np.isnan(preds[:, col]))
    test_data_valid = test_data[mask].reset_index(drop=True)
    y_test_valid = y_test[mask, col]
    preds_valid = preds[mask, col]
    return test_data_valid, y_test_valid, preds_valid


def format_large_numbers(x, pos):
    if abs(x) >= 1e6:
        val = x/1e6
        return f'{val:.0f}M' if val == int(val) else f'{val:.1f}M'
    elif abs(x) >= 1e3:
        val = x/1e3
        return f'{val:.0f}k' if val == int(val) else f'{val:.1f}k'
    elif x == 0:
        return '0'
    else:
        return f'{x:.0f}' if x == int(x) else f'{x:.1f}'


def create_single_timeseries_plot(ax, test_data, y_test, preds, target_index, targets, alpha=0.5, linewidth=0.5):
    test_data_valid, y_test_valid, preds_valid = preprocess_data(test_data, y_test, preds, target_index)
    for _, group_df in test_data_valid.groupby(INDEX_COLUMNS):
        group_years = group_df['Year']
        group_indices = group_df.index
        group_y_test = y_test_valid[group_indices]
        group_preds = preds_valid[group_indices]
        ax.plot(group_years, group_y_test, label='IAM', alpha=alpha, linewidth=linewidth)
        ax.plot(group_years, group_preds, label='XGBoost', alpha=alpha, linewidth=linewidth)
        ax.fill_between(group_years, group_y_test, group_preds, alpha=0.1)
    ylabel_with_unit = f"{targets[target_index]} ({OUTPUT_UNITS[target_index]})"
    ax.set_xlabel("Year", fontsize=16)
    ax.set_ylabel(ylabel_with_unit, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    formatter = FuncFormatter(format_large_numbers)
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))


def configure_axes(ax, min_val: float, max_val: float, xlabel: str, ylabel: str) -> None:
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    formatter = FuncFormatter(format_large_numbers)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def plot_scatter(run_id, test_data, y_test, preds, targets, filename: Optional[str] = None, model_name: str = "Model"):
    logging.info("Creating scatter plot...")
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    plt.rcParams.update({'font.size': 14})

    for i, ax in enumerate(axes.flatten()):
        test_data_valid, y_test_valid, preds_valid = preprocess_data(test_data, y_test, preds, i)
        unique_years = sorted(test_data_valid['Year'].unique())
        cmap = cm.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, len(unique_years)))

        for year, color in zip(unique_years, colors):
            group_df = test_data_valid[test_data_valid['Year'] == year]
            group_indices = group_df.index
            group_y_test = y_test_valid[group_indices]
            group_preds = preds_valid[group_indices]
            ax.scatter(group_y_test, group_preds, alpha=0.5, color=color, label=year)

        ax.set_title(targets[i])
        min_val = min(y_test_valid.min(), preds_valid.min())
        max_val = max(y_test_valid.max(), preds_valid.max())

        configure_axes(
            ax,
            min_val,
            max_val,
            "IAM",
            model_name,
        )
        ax.legend(title='Year', loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

    plt.tight_layout()
    if filename is None:
        filename = "scatter_plot.png"
    os.makedirs(os.path.join(RESULTS_PATH, run_id, "plots"), exist_ok=True)
    plt.savefig(os.path.join(RESULTS_PATH, run_id, "plots", filename), bbox_inches='tight')
    plt.close()


def plot_time_series(test_data, y_test, preds, targets, alpha=0.5, linewidth=0.5, run_id=None, filter_metadata=None, save_individual=False, individual_indices=[]):
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    plt.rcParams.update({'font.size': 14})

    # Filter years to 2015-2100 before checking if data is empty
    if test_data is not None and 'Year' in test_data.columns:
        year_mask = (test_data['Year'] >= 2015) & (test_data['Year'] <= 2100)
        test_data = test_data[year_mask].reset_index(drop=True)
        if y_test is not None:
            y_test = y_test[year_mask.values]
        if preds is not None:
            preds = preds[year_mask.values]

    if y_test is None or (hasattr(y_test, 'size') and y_test.size == 0):
        st.warning("y_test is empty. Displaying blank plots.")
        for i, ax in enumerate(axes.flatten()):
            ax.set_title(targets[i])
            ax.set_xlabel("Year")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        return

    for i, ax in enumerate(axes.flatten()):
        create_single_timeseries_plot(ax, test_data, y_test, preds, i, targets, alpha, linewidth)

    plt.tight_layout()
    st.pyplot(plt.gcf())
    
    # Save the full plot with filter metadata if run_id is provided
    if run_id and filter_metadata:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"timeseries_{timestamp}.png"
        metadata_filename = f"timeseries_{timestamp}_metadata.json"
        
        plots_dir = os.path.join(RESULTS_PATH, run_id, "saved_dashboard_plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save the full plot
        plt.savefig(os.path.join(plots_dir, plot_filename), bbox_inches='tight')
        
        # Save metadata
        import json
        with open(os.path.join(plots_dir, metadata_filename), 'w') as f:
            json.dump(filter_metadata, f, indent=2)
    
    # Save individual plots if requested
    if save_individual and run_id and filter_metadata:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_dir = os.path.join(RESULTS_PATH, run_id, "saved_dashboard_plots")
        
        for i in individual_indices:
            if 0 <= i < len(targets):
                individual_fig = plt.figure(figsize=(6, 6))
                individual_ax = individual_fig.add_subplot(111)
                create_single_timeseries_plot(individual_ax, test_data, y_test, preds, i, targets, alpha, linewidth)
                
                individual_filename = f"timeseries_{timestamp}_individual_{i}.png"
                plt.savefig(os.path.join(plots_dir, individual_filename), bbox_inches='tight')
                plt.close(individual_fig)


def get_saved_plots_metadata(run_id):
    """Get metadata for all saved time series plots."""
    plots_dir = os.path.join(RESULTS_PATH, run_id, "saved_dashboard_plots")
    if not os.path.exists(plots_dir):
        return []
    
    metadata_files = glob.glob(os.path.join(plots_dir, "*_metadata.json"))
    saved_plots = []
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Extract timestamp from filename
            filename = os.path.basename(metadata_file)
            timestamp_str = filename.replace('timeseries_', '').replace('_metadata.json', '')
            
            # Get corresponding plot file
            plot_file = metadata_file.replace('_metadata.json', '.png')
            if os.path.exists(plot_file):
                saved_plots.append({
                    'timestamp': timestamp_str,
                    'metadata': metadata,
                    'plot_path': plot_file,
                    'filename': os.path.basename(plot_file)
                })
        except (json.JSONDecodeError, FileNotFoundError):
            continue
    
    # Sort by timestamp (newest first)
    saved_plots.sort(key=lambda x: x['timestamp'], reverse=True)
    return saved_plots


def get_shap_values(run_id, X_test):
    """
    Create SHAP plots for the XGBoost model.
    Args:
        xgb: Trained XGBoost model.
        X_test: Dataframe of data feature columns
        features: List of feature names.
        targets: List of target names.
    """
    logging.info("Loading XGBoost model...")
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(RESULTS_PATH, run_id, "checkpoints", "final_best.json"))
    
    logging.info("Creating SHAP explainer...")
    explainer = shap.TreeExplainer(model, approximate=True)
    
    logging.info("Calculating SHAP values...")
    shap_values = explainer.shap_values(X_test)
    np.save(os.path.join(RESULTS_PATH, run_id, "plots", "shap_values.npy"), shap_values)
    logging.info("SHAP values saved to shap_values.npy")


def transform_outputs_to_former_inputs(
    run_id: str, 
    shap_values: np.ndarray, 
    targets: List[str], 
    features: List[str]
) -> np.ndarray:
    sorted_df_list = []
    for i, target in enumerate(targets):
        target_shap_values = np.abs(shap_values[:, :, i])  # Absolute SHAP values for the target
        mean_shap_values = np.mean(target_shap_values, axis=0)  # Mean SHAP values for each feature
        target_value = np.sum(mean_shap_values)
        importance = mean_shap_values / target_value 
        sorted_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        })
        sorted_df = sorted_df.sort_values(by="Importance", ascending=False)
        sorted_df_list.append(sorted_df)
        os.makedirs(os.path.join(RESULTS_PATH, run_id, "plots", "csv"), exist_ok=True)
        sorted_df.to_csv(os.path.join(RESULTS_PATH, run_id, "plots", "csv", f"shap{i+1}_{target}.csv"), index=False)
        # divide all shap values by target_value
        shap_values[:, :, i] = shap_values[:, :, i] / target_value

    input_only = shap_values.copy()
    feature_renaming = {}
    for i, target in enumerate(targets):
        # transform outputs in top 20 features
        num_features = shap_values.shape[1]
        new_feature_list = features.copy()
        feature_renaming[target] = {}
        for j in range(20):
            output = sorted_df_list[i].iloc[j]["Feature"]   # e.g. output = "prev_Primary Energy|Coal"
            if output.startswith("prev") and not output.endswith(target):
                num, output_name = output.split("_")    # e.g. num = "prev", output = "Primary Energy|Coal"
                num = 1 if num == "prev" else int(num[4:])
                output_index = targets.index(output_name)
                # replace [:, j, i] with new features
                old_feature = shap_values[:, j, i]
                for k in range(10):
                    new_input = sorted_df_list[output_index].iloc[k]["Feature"]
                    if not new_input.startswith("prev"):
                        new_input_index = new_feature_list.index(new_input)
                        new_features = shap_values[:, new_input_index, output_index] * old_feature
                        # replace input_only[:, j, i]
                        input_only[:, j, i] = new_features
                        # feature renamed to new_input
                        if num == 1:
                            feature_renaming[target][output] = "prev_" + new_input
                        else:
                            feature_renaming[target][output] = "prev" + str(num) + "_" + new_input
                        break
                    if k == 9:
                        logging.error(f"Input not found in the top 10 features for target {output_name}.")

        assert input_only.shape[1] == len(new_feature_list), f"input_only: {input_only.shape[1]}" + "\n" + f"new_feature_list: {len(new_feature_list)}"

        new_mean_shap_values = np.mean(input_only[:, :, i], axis=0)  # Mean SHAP values for each feature
        new_target_value = np.sum(new_mean_shap_values)
        new_importance = new_mean_shap_values / new_target_value
        new_sorted_df = pd.DataFrame({
            "Feature": new_feature_list,
            "Importance": new_importance
        })
        new_sorted_df = new_sorted_df.sort_values(by="Importance", ascending=False)
        new_sorted_df.to_csv(os.path.join(RESULTS_PATH, run_id, "plots", "csv", f"shap{i+1}_{target}_input_only.csv"), index=False)

    with open(os.path.join(RESULTS_PATH, run_id, "plots", "csv", "feature_renaming.json"), 'w') as json_file:
        json.dump(feature_renaming, json_file, indent=4)

    return input_only

def _make_display_name(feature: str) -> str:
    """Return a display name based on prev/prevN_ convention.
    - No prev prefix -> add " (current)"
    - prev_foo -> "foo (last 5y)"
    - prevN_foo -> "foo (last {N*5}y)"
    """
    m = re.match(r"^prev(\d*)_(.+)$", feature)
    if m:
        n_str, base = m.group(1), m.group(2)
        years = 5 if n_str == "" else int(n_str) * 5
        return f"{base} (last {years}y)"
    return f"{feature} (current)"


def _build_display_names(features: List[str], feature_name_map: Optional[Dict[str, str]] = None) -> List[str]:
    """Build display names for a list of features using optional overrides."""
    feature_name_map = feature_name_map or {}
    return [feature_name_map.get(feat, _make_display_name(feat)) for feat in features]


def draw_shap_plot(run_id, shap_values, X_test, features, targets, exclude_top=False, feature_name_map: Optional[Dict[str, str]] = None, model_prefix=""):
    n = 8

    # Create a 3*3 grid of subplots
    plt.rcParams.update({'font.size': 12})
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    num_targets = len(targets)
    
    for i, ax in tqdm(enumerate(axes.flatten()), total=num_targets):
        if i < num_targets:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                temp_filename = tmpfile.name
            
            # Create a new figure for the SHAP plot
            fig_shap = plt.figure()
            
            if exclude_top:
                # Find the top feature for this target and exclude it
                target_shap_values = np.abs(shap_values[:, :, i])
                mean_shap_values = np.mean(target_shap_values, axis=0)
                top_feature_idx = np.argmax(mean_shap_values)
                
                # Create masks to exclude the top feature
                feature_mask = np.ones(len(features), dtype=bool)
                feature_mask[top_feature_idx] = False
                
                # Filter data
                filtered_shap_values = shap_values[:, feature_mask, i]
                filtered_X_test = X_test.iloc[:, feature_mask]
                filtered_features = [features[j] for j in range(len(features)) if feature_mask[j]]
                filtered_display_names = _build_display_names(filtered_features, feature_name_map)

                # Apply categorical preprocessing to filtered data too
                filtered_X_test_processed = filtered_X_test.copy()
                from configs.data import CATEGORICAL_COLUMNS
                categorical_cols_in_filtered = [col for col in CATEGORICAL_COLUMNS if col in filtered_X_test_processed.columns]

                for col in categorical_cols_in_filtered:
                    filtered_X_test_processed[col] = filtered_X_test_processed[col].astype('category').cat.codes

                shap.summary_plot(
                    filtered_shap_values,
                    filtered_X_test_processed.values.astype(np.float64),
                    feature_names=filtered_display_names,
                    max_display=n,
                    plot_type="violin",
                    show=False,
                )
            else:
                display_names = _build_display_names(features, feature_name_map)

                # Apply same categorical preprocessing as during SHAP calculation
                X_test_processed_for_plot = X_test.copy()

                # Handle categorical columns the same way as in preprocess_features
                from configs.data import CATEGORICAL_COLUMNS
                categorical_cols_in_features = [col for col in CATEGORICAL_COLUMNS if col in X_test_processed_for_plot.columns]

                for col in categorical_cols_in_features:
                    X_test_processed_for_plot[col] = X_test_processed_for_plot[col].astype('category').cat.codes

                # Convert to numeric array for SHAP plotting
                X_test_values = X_test_processed_for_plot.values.astype(np.float64)

                shap.summary_plot(
                    shap_values[:, :, i],
                    X_test_values,
                    feature_names=display_names,
                    max_display=n,
                    plot_type="violin",
                    show=False,
                )
            fig_shap.tight_layout()
            fig_shap.savefig(temp_filename, format='png', bbox_inches='tight')
            plt.close(fig_shap)  # Close the SHAP figure
            img = Image.open(temp_filename)
            ax.imshow(img)
            ax.axis('off')
            title_suffix = " (excluding top feature)" if exclude_top else ""
            # Use centralized units aligned with OUTPUT_VARIABLES ordering
            ax.set_title("Impact on " + targets[i] + " (" + OUTPUT_UNITS[i] + ")" + title_suffix)
            os.remove(temp_filename)
        else:
            ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.join(RESULTS_PATH, run_id, "plots"), exist_ok=True)
    prefix = f"{model_prefix}_" if model_prefix else ""
    filename = f"{prefix}shap_plot_no_first.png" if exclude_top else f"{prefix}shap_plot.png"
    plt.savefig(os.path.join(RESULTS_PATH, run_id, "plots", filename))
    plt.close()


def plot_shap(run_id, X_test_with_index, features, targets, feature_name_map: Optional[Dict[str, str]] = None):
    logging.info("Creating SHAP plots...")
    # If model checkpoint doesn't exist, skip SHAP plotting gracefully
    ckpt_path = os.path.join(RESULTS_PATH, run_id, "checkpoints", "final_best.json")
    if not os.path.exists(ckpt_path):
        logging.warning("Skipping SHAP plots: model checkpoint not found at %s", ckpt_path)
        return
    # Remove non-feature columns from X_test if they exist
    X_test = X_test_with_index.drop(columns=NON_FEATURE_COLUMNS, errors="ignore")
    X_test = X_test.reset_index(drop=True)

    # Subsample if needed
    if X_test.shape[0] > 100:
        indices = np.random.choice(X_test.shape[0], 100, replace=False)
        X_test = X_test.iloc[indices]

    get_shap_values(run_id, X_test)
    logging.info("Transforming outputs to former inputs...")
    shap_values = np.load(os.path.join(RESULTS_PATH, run_id, "plots", "shap_values.npy"), allow_pickle=True)
    shap_values = transform_outputs_to_former_inputs(run_id, shap_values, targets, features)
    logging.info("Drawing regular SHAP plots...")
    draw_shap_plot(run_id, shap_values, X_test, features, targets, exclude_top=False, feature_name_map=feature_name_map)
    logging.info("Drawing SHAP plots excluding top feature...")
    draw_shap_plot(run_id, shap_values, X_test, features, targets, exclude_top=True, feature_name_map=feature_name_map)


def get_lstm_shap_values(run_id, X_test, sequence_length=1):
    """
    Create SHAP values for the LSTM model using DeepExplainer.
    Args:
        run_id: Run ID to locate model checkpoint
        X_test: Test data features
        sequence_length: Sequence length used in LSTM training
    """
    from src.trainers.lstm_trainer import LSTMModel, LSTMDataset

    logging.info("Loading LSTM model...")
    model_path = os.path.join(RESULTS_PATH, run_id, "final", "best.ckpt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LSTM model checkpoint not found: {model_path}")

    model = LSTMModel.load_from_checkpoint(model_path)
    model.eval()

    # Get session state to retrieve scalers and config
    from src.utils.utils import load_session_state
    session_state = load_session_state(run_id)
    scaler_X = session_state.get("lstm_scaler_X")
    scaler_y = session_state.get("lstm_scaler_y")
    features = session_state.get("features")
    targets = session_state.get("targets")

    if scaler_X is None or scaler_y is None:
        raise ValueError("LSTM scalers not found in session state")

    # Preprocess data directly without dummy datasets
    def preprocess_features(data, features, scaler_X, mask_value=-1.0):
        """Preprocess features the same way LSTMDataset does."""
        X = data[features].copy()

        # Handle categorical columns
        from configs.data import CATEGORICAL_COLUMNS
        categorical_cols_in_features = [col for col in CATEGORICAL_COLUMNS if col in X.columns]

        for col in categorical_cols_in_features:
            X[col] = X[col].astype('category').cat.codes

        # Handle NaN values
        X_filled = X.fillna(mask_value).astype(np.float32)

        # Scale features
        X_scaled = scaler_X.transform(X_filled)

        return X_scaled

    def create_sequences_from_scaled_data(X_scaled, sequence_length):
        """Create sequences from scaled data without grouping (for SHAP)."""
        sequences = []
        n_samples = len(X_scaled)

        # Create overlapping sequences
        for i in range(n_samples - sequence_length + 1):
            seq = X_scaled[i:i + sequence_length]
            sequences.append(torch.FloatTensor(seq))

        return torch.stack(sequences) if sequences else torch.empty(0, sequence_length, X_scaled.shape[1])

    # Background data processing
    background_size = min(200, len(X_test))  # Larger background for better baselines
    background_data_raw = X_test.iloc[:background_size]
    background_scaled = preprocess_features(background_data_raw, features, scaler_X)
    background_sequences = create_sequences_from_scaled_data(background_scaled, sequence_length)

    # Take subset of background sequences for DeepExplainer
    if len(background_sequences) > 50:
        background_indices = np.random.choice(len(background_sequences), 50, replace=False)
        background_data = background_sequences[background_indices]
    else:
        background_data = background_sequences

    # Test data processing
    test_size = min(100, len(X_test))
    test_data_subset = X_test.iloc[:test_size]
    test_scaled = preprocess_features(test_data_subset, features, scaler_X)
    test_inputs = create_sequences_from_scaled_data(test_scaled, sequence_length)

    logging.info("Creating DeepSHAP explainer...")

    # Create a simplified model wrapper that DeepExplainer can handle
    class LSTMWrapperForSHAP(torch.nn.Module):
        def __init__(self, lstm_model, sequence_length):
            super().__init__()
            self.lstm_model = lstm_model
            self.sequence_length = sequence_length

        def forward(self, x):
            # x shape: [batch_size, seq_len, features]
            # Don't use torch.no_grad() here - SHAP needs gradients!

            # Create dummy masks
            batch_size = x.shape[0]
            mask = torch.ones(batch_size, self.sequence_length, dtype=torch.float32, device=x.device)

            # Use autoregressive mode (no teacher forcing for inference)
            outputs = self.lstm_model(x, mask=mask, teacher_forcing=False)
            return outputs

    # Create wrapper model
    wrapper_model = LSTMWrapperForSHAP(model, sequence_length)
    wrapper_model.eval()

    # Ensure tensors require gradients for SHAP
    background_data.requires_grad_(True)
    test_inputs.requires_grad_(True)

    # Create DeepExplainer with the wrapper model
    explainer = shap.DeepExplainer(wrapper_model, background_data)

    logging.info("Calculating SHAP values...")
    # Disable additivity check for complex LSTM models
    shap_values = explainer.shap_values(test_inputs, check_additivity=False)

    # Convert to numpy if needed and handle temporal dimensions properly
    if isinstance(shap_values, list):
        shap_values = [_to_numpy(sv) for sv in shap_values]
        shap_values = np.array(shap_values, dtype=np.float64)  # Shape: [n_outputs, n_samples, seq_len, n_features]
        shap_values = np.transpose(shap_values, (1, 2, 3, 0))  # [n_samples, seq_len, n_features, n_outputs]
    else:
        # Single output case
        shap_values = _to_numpy(shap_values)
        shap_values = np.array(shap_values, dtype=np.float64)
        if shap_values.ndim == 3:  # [n_samples, seq_len, n_features]
            shap_values = np.expand_dims(shap_values, axis=-1)  # [n_samples, seq_len, n_features, 1]

    # DON'T average over sequence - preserve temporal information
    # Shape should be [n_samples, seq_len, n_features, n_outputs]
    original_temporal_shap = shap_values.copy()

    # For compatibility with existing plotting functions, also create averaged version
    averaged_shap_values = np.mean(shap_values, axis=1)  # [n_samples, n_features, n_outputs]


    # Save SHAP values (both temporal and averaged versions)
    os.makedirs(os.path.join(RESULTS_PATH, run_id, "plots"), exist_ok=True)
    np.save(os.path.join(RESULTS_PATH, run_id, "plots", "lstm_shap_values_temporal.npy"), original_temporal_shap)
    np.save(os.path.join(RESULTS_PATH, run_id, "plots", "lstm_shap_values.npy"), averaged_shap_values)
    logging.info("LSTM SHAP values saved (temporal and averaged versions)")

    # Return X_test_processed that matches the number of SHAP samples
    # Since sequences reduce the number of samples, we need to align X_test accordingly
    n_shap_samples = averaged_shap_values.shape[0]
    X_test_processed = test_data_subset[features].iloc[:n_shap_samples]

    # Return also the exact test sequences used for SHAP (numpy)
    test_sequences_np = _to_numpy(test_inputs)

    return original_temporal_shap, averaged_shap_values, X_test_processed, test_sequences_np


def plot_lstm_shap(run_id, X_test_with_index, features, targets, sequence_length=1, feature_name_map: Optional[Dict[str, str]] = None):
    """
    Create SHAP plots for the LSTM model similar to XGBoost plots.
    Args:
        run_id: Run ID to locate model checkpoint
        X_test_with_index: Test data with index columns
        features: List of feature names
        targets: List of target names
        sequence_length: Sequence length used in LSTM training
        feature_name_map: Optional mapping for feature display names
    """
    logging.info("Creating LSTM SHAP plots...")

    # Check if model checkpoint exists
    model_path = os.path.join(RESULTS_PATH, run_id, "final", "best.ckpt")
    if not os.path.exists(model_path):
        logging.warning("Skipping LSTM SHAP plots: model checkpoint not found at %s", model_path)
        return

    # Remove non-feature columns from X_test if they exist
    X_test = X_test_with_index.drop(columns=NON_FEATURE_COLUMNS, errors="ignore")
    X_test = X_test.reset_index(drop=True)

    # Subsample if needed (DeepSHAP is computationally expensive)
    if X_test.shape[0] > 100:
        indices = np.random.choice(X_test.shape[0], 100, replace=False)
        X_test = X_test.iloc[indices]

    try:
        # Get SHAP values (both temporal and averaged) and test sequences
        temporal_shap_values, averaged_shap_values, X_test_processed, test_sequences_np = get_lstm_shap_values(run_id, X_test, sequence_length)

        # Draw the all-timesteps SHAP plot (flatten all timesteps as separate features)
        logging.info("Drawing LSTM SHAP plots with all timesteps flattened...")
        draw_lstm_all_timesteps_shap_plot(
            run_id,
            temporal_shap_values,
            test_sequences_np,
            features,
            targets,
            sequence_length,
            feature_name_map=feature_name_map,
        )

        # Optional: temporal heatmaps and CSVs remain for diagnostics
        logging.info("Drawing temporal LSTM SHAP heatmaps and timestep bars...")
        draw_temporal_shap_plot(
            run_id,
            temporal_shap_values,
            X_test_processed,
            features,
            targets,
            sequence_length,
            feature_name_map=feature_name_map,
        )

    except Exception as e:
        logging.error("Failed to create LSTM SHAP plots: %s", str(e))
        logging.exception("Full error traceback:")



def draw_lstm_all_timesteps_shap_plot(
    run_id: str,
    temporal_shap_values: np.ndarray,
    test_sequences_np: np.ndarray,
    features: List[str],
    targets: List[str],
    sequence_length: int,
    feature_name_map: Optional[Dict[str, str]] = None,
) -> None:
    """
    Draw SHAP plots that include ALL timesteps equally by flattening
    every timestep in the sequence into separate feature copies.

    Saves combined grid as plots/lstm_shap_plot.png
    """
    if sequence_length <= 0:
        logging.warning("Invalid sequence_length=%d; skipping all-timesteps SHAP plot", sequence_length)
        return

    # Build flattened inputs across all timesteps: [N, seq_len * n_features]
    x_flat_parts = [test_sequences_np[:, t, :] for t in range(sequence_length)]
    X_flat = np.concatenate(x_flat_parts, axis=1)

    # Build display names per timestep using current/last 5y semantics
    time_labels = []
    for t in range(sequence_length):
        lag = sequence_length - 1 - t
        label = "current" if lag == 0 else f"last {lag*5}y"
        time_labels.append(label)
    feature_name_map = feature_name_map or {}
    base_names = [feature_name_map.get(f, f) for f in features]
    display_names: List[str] = []
    for t in range(sequence_length):
        label = time_labels[t]
        for bn in base_names:
            display_names.append(f"{bn} ({label})")

    # Plot per target in a 3x3 grid
    plt.rcParams.update({'font.size': 12})
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    num_targets = len(targets)

    for i, ax in enumerate(axes.flatten()):
        if i < num_targets:
            fig_shap = plt.figure()
            # Flatten SHAP values per target: concat across timesteps along feature axis
            shap_flat_parts = [temporal_shap_values[:, t, :, i] for t in range(sequence_length)]
            shap_flat = np.concatenate(shap_flat_parts, axis=1)

            shap.summary_plot(
                shap_flat,
                X_flat,
                feature_names=display_names,
                max_display=8,
                plot_type="violin",
                show=False,
            )
            fig_shap.tight_layout()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpf:
                temp_filename = tmpf.name
            fig_shap.savefig(temp_filename, format='png', bbox_inches='tight')
            plt.close(fig_shap)
            img = Image.open(temp_filename)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"SHAP across all timesteps: {targets[i]} ({OUTPUT_UNITS[i]})")
            os.remove(temp_filename)
        else:
            ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.join(RESULTS_PATH, run_id, "plots"), exist_ok=True)
    plt.savefig(os.path.join(RESULTS_PATH, run_id, "plots", "lstm_shap_plot.png"))
    plt.close()



def draw_temporal_shap_plot(
    run_id: str,
    temporal_shap_values: np.ndarray,
    X_test: pd.DataFrame,
    features: List[str],
    targets: List[str],
    sequence_length: int,
    feature_name_map: Optional[Dict[str, str]] = None
) -> None:
    """
    Draw SHAP plots that show feature importance across time steps for LSTM.
    This is unique to sequential models and shows how feature importance varies across the sequence.

    Args:
        temporal_shap_values: Shape [n_samples, seq_len, n_features, n_outputs]
        X_test: Test data features
        features: List of feature names
        targets: List of target names
        sequence_length: Length of input sequences
        feature_name_map: Optional mapping for feature display names
    """
    # Units derived from configs

    # Create temporal feature importance plots
    plt.rcParams.update({'font.size': 12})
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    num_targets = len(targets)

    for i, ax in tqdm(enumerate(axes.flatten()), total=num_targets, desc="Creating temporal SHAP plots"):
        if i < num_targets:
            # Calculate feature importance at each time step
            target_shap = temporal_shap_values[:, :, :, i]  # [n_samples, seq_len, n_features]

            # Get top features (averaged across time for selection)
            avg_feature_importance = np.mean(np.abs(target_shap), axis=(0, 1))  # [n_features]
            top_feature_indices = np.argsort(avg_feature_importance)[-8:][::-1]  # Top 8 features

            # Create heatmap showing feature importance over time
            time_importance = np.mean(np.abs(target_shap[:, :, top_feature_indices]), axis=0)  # [seq_len, 8]

            # Create heatmap
            im = ax.imshow(time_importance.T, aspect='auto', cmap='viridis', interpolation='nearest')

            # Set labels
            time_labels = ["current" if (sequence_length - 1 - j) == 0 else f"last {(sequence_length - 1 - j)*5}y" for j in range(sequence_length)]
            ax.set_xticks(range(sequence_length))
            ax.set_xticklabels(time_labels)

            # Feature labels
            display_names = _build_display_names([features[idx] for idx in top_feature_indices], feature_name_map)
            ax.set_yticks(range(len(top_feature_indices)))
            ax.set_yticklabels([name[:30] + "..." if len(name) > 30 else name for name in display_names], fontsize=10)

            ax.set_title(f"Temporal SHAP: {targets[i]} ({OUTPUT_UNITS[i]})", fontsize=14)
            ax.set_xlabel("Time Step in Sequence", fontsize=12)
            ax.set_ylabel("Features", fontsize=12)

            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.6)

        else:
            ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.join(RESULTS_PATH, run_id, "plots"), exist_ok=True)
    plt.savefig(os.path.join(RESULTS_PATH, run_id, "plots", "lstm_temporal_shap_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Also create time-step comparison plots
    create_timestep_comparison_plots(run_id, temporal_shap_values, features, targets, sequence_length, feature_name_map)


def create_timestep_comparison_plots(
    run_id: str,
    temporal_shap_values: np.ndarray,
    features: List[str],
    targets: List[str],
    sequence_length: int,
    feature_name_map: Optional[Dict[str, str]] = None
) -> None:
    """
    Create bar plots comparing feature importance at different time steps.
    Shows how LSTM weighs recent vs older information.
    """
    # Units derived from configs

    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))

    for i, ax in enumerate(axes.flatten()):
        if i < len(targets):
            # Get SHAP values for this target
            target_shap = temporal_shap_values[:, :, :, i]  # [n_samples, seq_len, n_features]

            # Calculate mean absolute importance at each time step
            timestep_importance = np.mean(np.abs(target_shap), axis=(0, 2))  # [seq_len]

            # Create bar plot
            time_labels = ["current" if (sequence_length - 1 - j) == 0 else f"last {(sequence_length - 1 - j)*5}y" for j in range(sequence_length)]
            colors = cm.get_cmap('viridis')(np.linspace(0, 1, sequence_length))

            bars = ax.bar(range(sequence_length), timestep_importance, color=colors, alpha=0.8)

            ax.set_title(f"Time Step Importance: {targets[i]} ({OUTPUT_UNITS[i]})", fontsize=14)
            ax.set_xlabel("Time Step in Sequence", fontsize=12)
            ax.set_ylabel("Average |SHAP| Value", fontsize=12)
            ax.set_xticks(range(sequence_length))
            ax.set_xticklabels(time_labels)

            # Add value labels on bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)

        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, run_id, "plots", "lstm_timestep_importance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Temporal SHAP plots saved")
