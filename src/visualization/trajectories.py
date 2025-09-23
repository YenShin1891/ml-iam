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
    'preprocess_data','format_large_numbers','create_single_trajectory_plot','configure_axes',
    'plot_scatter','plot_trajectories','get_saved_plots_metadata'
]

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
        configure_axes(ax, min_val, max_val, "IAM", model_name)
        ax.legend(title='Year', loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    plt.tight_layout()
    if filename is None:
        filename = "scatter_plot.png"
    os.makedirs(os.path.join(RESULTS_PATH, run_id, "plots"), exist_ok=True)
    plt.savefig(os.path.join(RESULTS_PATH, run_id, "plots", filename), bbox_inches='tight')
    plt.close()

def plot_trajectories(test_data, y_test, preds, targets, alpha=0.5, linewidth=0.5, run_id=None, filter_metadata=None, save_individual=False, individual_indices=None):
    if individual_indices is None:
        individual_indices = []
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    plt.rcParams.update({'font.size': 14})
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
    for i, ax in enumerate(axes.flatten()):
        create_single_trajectory_plot(ax, test_data, y_test, preds, i, targets, alpha, linewidth)
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
                create_single_trajectory_plot(individual_ax, test_data, y_test, preds, i, targets, alpha, linewidth)
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
