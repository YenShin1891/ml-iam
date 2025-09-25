# Trajectory and scatter plotting (migrated from utils.plot_trajectories)
from configs.paths import RESULTS_PATH
from configs.data import INDEX_COLUMNS, OUTPUT_UNITS
import os, json, datetime, glob, logging
from typing import Optional, Tuple
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FuncFormatter, MaxNLocator
import streamlit as st

__all__ = [
    'preprocess_data','format_large_numbers','create_single_trajectory_plot','create_single_scatter_plot','configure_axes',
    'plot_scatter','plot_trajectories','get_saved_plots_metadata','apply_inverse_scaling','compute_r2'
]

def get_model_type_from_log(run_id):
    """Detect model type from log file name in RESULTS_PATH/run_id."""
    run_dir = os.path.join(RESULTS_PATH, run_id)
    try:
        for fname in os.listdir(run_dir):
            if fname.startswith("train_") and fname.endswith(".log"):
                # e.g., train_lstm.log or train_xgb.log
                model_name = fname[len("train_"):-len(".log")]
                return model_name.lower()
    except Exception as e:
        logging.info(f"Could not determine model type for run {run_id}: {e}")
    return None

def create_single_scatter_plot(ax, test_data_valid, y_test_valid, preds_valid, target_index, targets, model_name, output_units):
    unique_years = sorted(test_data_valid['Year'].unique()) if len(test_data_valid) else []
    cmap = cm.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(unique_years))) if unique_years else []
    for year, color in zip(unique_years, colors):
        group_df = test_data_valid[test_data_valid['Year'] == year]
        group_indices = group_df.index
        group_y_test = y_test_valid[group_indices]
        group_preds = preds_valid[group_indices]
        ax.scatter(group_y_test, group_preds, alpha=0.5, color=color, label=year)
    unit = output_units[target_index] if target_index < len(output_units) else ""
    if target_index < len(targets):
        ax.set_title(targets[target_index], fontsize=19)
    if len(y_test_valid) and len(preds_valid):
        min_val = float(min(y_test_valid.min(), preds_valid.min()))
        max_val = float(max(y_test_valid.max(), preds_valid.max()))
    else:
        min_val, max_val = 0.0, 1.0
    if unit:
        xlabel = f"IAM ({unit})"
        ylabel = f"{model_name} ({unit})"
    else:
        xlabel = "IAM"
        ylabel = model_name
    configure_axes(ax, min_val, max_val, xlabel, ylabel)
    if unique_years:
        ax.legend(title='Year', loc='upper left', bbox_to_anchor=(1, 1), fontsize=11)
    r2_val = compute_r2(y_test_valid, preds_valid)
    if not np.isnan(r2_val):
        ax.text(0.05, 0.95, f'R² = {r2_val:.3f}', transform=ax.transAxes, fontsize=17,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ... (content copied verbatim from original) ...
import numpy as np
import pandas as pd

# Keeping original function bodies for continuity

def preprocess_data(test_data: pd.DataFrame, y_test: np.ndarray, preds: np.ndarray, target_index: int) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    n = len(test_data)
    if y_test.shape[0] != n or preds.shape[0] != n:
        m = min(n, y_test.shape[0], preds.shape[0])
        test_data = test_data.iloc[:m].reset_index(drop=True)
        y_test = y_test[:m]
        preds = preds[:m]
    col = min(target_index, y_test.shape[1] - 1, preds.shape[1] - 1)
    mask = (~np.isnan(y_test[:, col])) & (~np.isnan(preds[:, col]))
    test_data_valid = test_data[mask].reset_index(drop=True)
    y_test_valid = y_test[mask, col]
    preds_valid = preds[mask, col]
    return test_data_valid, y_test_valid, preds_valid

def format_large_numbers(x, pos):  # noqa: ARG001
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

def create_single_trajectory_plot(ax, test_data, y_test, preds, target_index, targets, alpha=0.5, linewidth=0.5):
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
    ax.set_xlabel("Year", fontsize=19)
    ax.set_ylabel(ylabel_with_unit, fontsize=19)
    ax.tick_params(axis='both', which='major', labelsize=15)
    formatter = FuncFormatter(format_large_numbers)
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute coefficient of determination R^2 with basic safety checks.

    Returns NaN if fewer than 2 valid points or zero variance in y_true.
    """
    try:
        if y_true is None or y_pred is None:
            return float('nan')
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
        if mask.sum() < 2:
            return float('nan')
        y_t = y_true[mask]
        y_p = y_pred[mask]
        denom = np.var(y_t)
        if denom == 0:
            return float('nan')
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - y_t.mean()) ** 2)
        if ss_tot == 0:
            return float('nan')
        return 1 - ss_res/ss_tot
    except Exception:
        return float('nan')

def apply_inverse_scaling(y_values: Optional[np.ndarray], preds_values: Optional[np.ndarray], run_id: Optional[str] = None):
    """Attempt to inverse scale arrays using scaler(s) stored in Streamlit session_state.

    Checks known session_state keys for a target scaler and applies its inverse_transform
    if available. Returns (y_out, preds_out, used_scaler_name).

    Parameters
    ----------
    y_values : np.ndarray | None
        True target values (scaled). Not modified in-place.
    preds_values : np.ndarray | None
        Predicted target values (scaled). Not modified in-place.

    Notes
    -----
    - If no scaler is found, originals are returned unchanged.
    - Emits Streamlit informational/warning messages instead of raising.
    """
    if y_values is None or preds_values is None:
        return y_values, preds_values, None
    y_out = np.array(y_values, copy=True)
    preds_out = np.array(preds_values, copy=True)
    scaler_candidates = ['lstm_scaler_y', 'scaler_y']
    scaler_found = None
    scaler_key_used = None
    try:
        for key in scaler_candidates:
            if key in st.session_state and st.session_state[key] is not None:
                scaler_found = st.session_state[key]
                scaler_key_used = key
                break
        # Attempt to load from disk if not found in session_state
        if scaler_found is None and run_id is not None:
            try:
                from src.utils.utils import load_session_state
                fname = 'y_scaler.pkl'
                scaler_candidate = load_session_state(run_id, fname)
                if hasattr(scaler_candidate, 'inverse_transform'):
                    scaler_found = scaler_candidate
                    scaler_key_used = f'file:{fname}'
                    st.session_state['scaler_y'] = scaler_found
                elif isinstance(scaler_candidate, dict):
                    for v in scaler_candidate.values():
                        if hasattr(v, 'inverse_transform'):
                            scaler_found = v
                            scaler_key_used = f'file:{fname}'
                            st.session_state['scaler_y'] = scaler_found
                            break
            except Exception as disk_e:  # noqa: BLE001
                logging.info(f"No y_scaler.pkl loaded for run {run_id}: {disk_e}")
        if scaler_found is None:
            st.info("No scaler found in session state; displaying scaled values (may be misleading).")
            return y_out, preds_out, None
        # Ensure 2D shape
        y_2d = y_out.reshape(-1, 1) if y_out.ndim == 1 else y_out
        preds_2d = preds_out.reshape(-1, 1) if preds_out.ndim == 1 else preds_out
        if hasattr(scaler_found, 'inverse_transform'):
            y_inv = scaler_found.inverse_transform(y_2d)
            preds_inv = scaler_found.inverse_transform(preds_2d)
            y_out = y_inv.ravel() if y_out.ndim == 1 else y_inv
            preds_out = preds_inv.ravel() if preds_out.ndim == 1 else preds_inv
        else:
            st.warning("Scaler found but missing inverse_transform; using scaled values.")
        return y_out, preds_out, scaler_key_used
    except Exception as e:  # noqa: BLE001
        st.warning(f"Automatic inverse scaling failed; using scaled values. Error: {e}")
        return y_values, preds_values, None

def configure_axes(ax, min_val: float, max_val: float, xlabel: str, ylabel: str) -> None:
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(xlabel, fontsize=19)
    ax.set_ylabel(ylabel, fontsize=19)
    ax.tick_params(axis='both', which='major', labelsize=15)
    formatter = FuncFormatter(format_large_numbers)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

def plot_scatter(run_id, test_data, y_test, preds, targets, filename: Optional[str] = None, model_name: str = "Model"):
    logging.info("Creating scatter plot (model-aware inverse scaling)...")
    model_type = get_model_type_from_log(run_id)
    y_plot = None if y_test is None else np.array(y_test, copy=True)
    preds_plot = None if preds is None else np.array(preds, copy=True)
    scaler_key = None
    # Only apply inverse scaling for xgb (or other non-lstm models)
    if model_type and ("xgb" in model_type or "xgboost" in model_type):
        y_plot, preds_plot, scaler_key = apply_inverse_scaling(y_plot, preds_plot, run_id)
        if scaler_key:
            logging.info(f"Scatter: applied inverse scaling using scaler key '{scaler_key}'")
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    plt.rcParams.update({'font.size': 16})
    for i, ax in enumerate(axes.flatten()):
        test_data_valid, y_test_valid, preds_valid = preprocess_data(test_data, y_plot, preds_plot, i)
        create_single_scatter_plot(ax, test_data_valid, y_test_valid, preds_valid, i, targets, model_name, OUTPUT_UNITS)
        # Save individual scatter plot for each output in indiv_plots dir (single plot per figure)
        indiv_dir = os.path.join(RESULTS_PATH, run_id, "plots", "indiv_plots")
        os.makedirs(indiv_dir, exist_ok=True)
        indiv_filename = f"scatter_{i}_{targets[i] if i < len(targets) else 'unknown'}.png"
        indiv_path = os.path.join(indiv_dir, indiv_filename)
        fig_indiv, ax_indiv = plt.subplots(figsize=(7, 7))
        create_single_scatter_plot(ax_indiv, test_data_valid, y_test_valid, preds_valid, i, targets, model_name, OUTPUT_UNITS)
        fig_indiv.tight_layout()
        fig_indiv.savefig(indiv_path, bbox_inches='tight')
        plt.close(fig_indiv)
    plt.tight_layout()
    if filename is None:
        filename = "scatter_plot.png"
    os.makedirs(os.path.join(RESULTS_PATH, run_id, "plots"), exist_ok=True)
    plt.savefig(os.path.join(RESULTS_PATH, run_id, "plots", filename), bbox_inches='tight')
    plt.close()

def plot_trajectories(
    test_data,
    y_test,
    preds,
    targets,
    alpha=0.5,
    linewidth=0.5,
    run_id=None,
    filter_metadata=None,
    save_individual=False,
    individual_indices=None,
):
    """Plot trajectories for each target.

    Parameters
    ----------
    test_data : pd.DataFrame
        Original (possibly filtered) test dataframe containing a 'Year' column and index columns.
    y_test : np.ndarray
        True target values (scaled or original). Shape (n_samples, n_targets) or (n_samples,).
    preds : np.ndarray
        Predicted target values (scaled or original). Shape like y_test.
    targets : list[str]
        List of target variable names.
    alpha : float, optional
        Line alpha for trajectories.
    linewidth : float, optional
        Line width for trajectories.
    run_id : str, optional
        Run identifier for saving plots.
    filter_metadata : dict, optional
        Metadata to save alongside plots.
    save_individual : bool, optional
        Whether to save individual target plots.
    individual_indices : list[int], optional
        Indices of targets to save individually.
    """
    if individual_indices is None:
        individual_indices = []
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    plt.rcParams.update({'font.size': 16})
    # NOTE: Previously filtered years to 2015-2100. Commented out to retain full historical range.
    # if test_data is not None and 'Year' in test_data.columns:
    #     year_mask = (test_data['Year'] >= 2015) & (test_data['Year'] <= 2100)
    #     test_data = test_data[year_mask].reset_index(drop=True)
    #     if y_test is not None:
    #         y_test = y_test[year_mask.values]
    #     if preds is not None:
    #         preds = preds[year_mask.values]
    if y_test is None or (hasattr(y_test, 'size') and y_test.size == 0):
        st.warning("y_test is empty. Displaying blank plots.")
        for i, ax in enumerate(axes.flatten()):
            ax.set_title(targets[i])
            ax.set_xlabel("Year")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        return
    # Prepare copies for potential inverse scaling so we don't mutate caller data
    y_plot = None if y_test is None else np.array(y_test, copy=True)
    preds_plot = None if preds is None else np.array(preds, copy=True)
    model_type = get_model_type_from_log(run_id)
    scaler_key = None
    # Only apply inverse scaling for xgb (or other non-lstm models)
    if model_type and ("xgb" in model_type or "xgboost" in model_type):
        y_plot, preds_plot, scaler_key = apply_inverse_scaling(y_plot, preds_plot, run_id)
        if scaler_key:
            logging.info(f"Applied inverse scaling using scaler in session_state key: {scaler_key}")
    for i, ax in enumerate(axes.flatten()):
        create_single_trajectory_plot(ax, test_data, y_plot, preds_plot, i, targets, alpha, linewidth)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    if run_id and filter_metadata:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"trajectories_{timestamp}.png"
        metadata_filename = f"trajectories_{timestamp}_metadata.json"
        plots_dir = os.path.join(RESULTS_PATH, run_id, "saved_dashboard_plots")
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, plot_filename), bbox_inches='tight')
        with open(os.path.join(plots_dir, metadata_filename), 'w') as f:
            json.dump(filter_metadata, f, indent=2)
    if save_individual and run_id and filter_metadata:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_dir = os.path.join(RESULTS_PATH, run_id, "saved_dashboard_plots")
        for i in individual_indices:
            if 0 <= i < len(targets):
                individual_fig = plt.figure(figsize=(6, 6))
                individual_ax = individual_fig.add_subplot(111)
                create_single_trajectory_plot(individual_ax, test_data, y_plot, preds_plot, i, targets, alpha, linewidth)
                individual_filename = f"trajectories_{timestamp}_individual_{i}.png"
                plt.savefig(os.path.join(plots_dir, individual_filename), bbox_inches='tight')
                plt.close(individual_fig)

def get_saved_plots_metadata(run_id):
    plots_dir = os.path.join(RESULTS_PATH, run_id, "saved_dashboard_plots")
    if not os.path.exists(plots_dir):
        return []
    metadata_files = glob.glob(os.path.join(plots_dir, "*_metadata.json"))
    saved_plots = []
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            filename = os.path.basename(metadata_file)
            if filename.startswith('trajectories_'):
                timestamp_str = filename[len('trajectories_'):-len('_metadata.json')]
            elif filename.startswith('timeseries_'):
                timestamp_str = filename[len('timeseries_'):-len('_metadata.json')]
            else:
                continue
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
    saved_plots.sort(key=lambda x: x['timestamp'], reverse=True)
    return saved_plots
