import matplotlib.pyplot as plt
from matplotlib import cm
import streamlit as st
import shap
import tempfile
import logging
from tqdm import tqdm
from PIL import Image

import numpy as np
import pandas as pd
import os
import json

from configs.config import INDEX_COLUMNS, NON_FEATURE_COLUMNS, RESULTS_PATH, OUTPUT_UNITS
from matplotlib.ticker import FuncFormatter, MaxNLocator


def format_large_numbers(x, pos):
    """Format large numbers for better readability on axes"""
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

def preprocess_data(test_data, y_test, preds, target_index):
    mask = ~np.isnan(y_test[:, target_index]) & ~np.isnan(preds[:, target_index])
    test_data_valid = test_data[mask].reset_index(drop=True)
    y_test_valid = y_test[mask, target_index]
    preds_valid = preds[mask, target_index]
    
    # Filter years to 2015-2100 (made for legacy xgboost code, remove later)
    year_mask = (test_data_valid['Year'] >= 2015) & (test_data_valid['Year'] <= 2100)
    test_data_valid = test_data_valid[year_mask].reset_index(drop=True)
    y_test_valid = y_test_valid[year_mask.values]
    preds_valid = preds_valid[year_mask.values]
    
    return test_data_valid, y_test_valid, preds_valid


def create_single_scatter_plot(ax, test_data, y_test, preds, target_index, targets, use_log, model_label):
    """Create a single scatter plot for the specified target on the given axes"""
    test_data_valid, y_test_valid, preds_valid = preprocess_data(test_data, y_test, preds, target_index)
    unique_years = sorted(test_data_valid['Year'].unique())
    colors = cm.viridis(np.linspace(0, 1, len(unique_years)))
    
    for year, color in zip(unique_years, colors):
        group_df = test_data_valid[test_data_valid['Year'] == year]
        group_indices = group_df.index
        group_y_test = y_test_valid[group_indices]
        group_preds = preds_valid[group_indices]
        ax.scatter(group_y_test, group_preds, alpha=0.5, color=color, label=year)
        
    ax.set_title(targets[target_index], fontsize=17)
    
    if use_log:
        abs_y_test = np.abs(y_test_valid)
        abs_preds = np.abs(preds_valid)
        min_val = max(min(abs_y_test.min(), abs_preds.min()), 1e-10)
        max_val = max(abs_y_test.max(), abs_preds.max())
    else:
        min_val = min(y_test_valid.min(), preds_valid.min())
        max_val = max(y_test_valid.max(), preds_valid.max())
        
    configure_axes(
        ax,
        use_log,
        min_val,
        max_val,
        "IAM" + (" (log scale)" if use_log else ""),
        f"{model_label}" + (" (log scale)" if use_log else ""),
        x_unit=OUTPUT_UNITS[target_index],
        y_unit=OUTPUT_UNITS[target_index]
    )
    ax.legend(title='Year', loc='upper left', bbox_to_anchor=(1, 1), fontsize=11, title_fontsize=14)


def create_single_timeseries_plot(ax, test_data, y_test, preds, target_index, targets, alpha=0.5, linewidth=0.5):
    """Create a single time series plot for the specified target on the given axes"""
    test_data_valid, y_test_valid, preds_valid = preprocess_data(test_data, y_test, preds, target_index)
    
    for group_key, group_df in test_data_valid.groupby(INDEX_COLUMNS):
        group_years = group_df['Year']
        group_indices = group_df.index
        group_y_test = y_test_valid[group_indices]
        group_preds = preds_valid[group_indices]

        ax.plot(group_years, group_y_test, label='IAM', alpha=alpha, linewidth=linewidth)
        ax.plot(group_years, group_preds, label='XGBoost', alpha=alpha, linewidth=linewidth)
        ax.fill_between(group_years, group_y_test, group_preds, alpha=0.1)

    title_with_unit = f"{targets[target_index]} ({OUTPUT_UNITS[target_index]})"
    ax.set_title(title_with_unit, fontsize=17)
    ax.set_xlabel("Year", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Format y-axis for large numbers and reduce tick crowding
    formatter = FuncFormatter(format_large_numbers)
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))


def configure_axes(ax, use_log, min_val, max_val, xlabel, ylabel, x_unit=None, y_unit=None):
    if use_log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal', adjustable='box')
    
    # Create labels with smaller font units
    if x_unit:
        xlabel_with_unit = f"{xlabel} ({x_unit})"
    else:
        xlabel_with_unit = xlabel
    
    if y_unit:
        ylabel_with_unit = f"{ylabel} ({y_unit})"
    else:
        ylabel_with_unit = ylabel
    
    ax.set_xlabel(xlabel_with_unit, fontsize=16)
    ax.set_ylabel(ylabel_with_unit, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    
    # Format large numbers and prevent overlap
    formatter = FuncFormatter(format_large_numbers)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    
    # Reduce number of ticks to prevent crowding
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def plot_scatter(run_id, test_data, y_test, preds, targets, use_log=False, model_label: str = "XGBoost"):
    logging.info("Creating scatter plot...")
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    plt.rcParams.update({'font.size': 14})

    for i, ax in enumerate(axes.flatten()):
        create_single_scatter_plot(ax, test_data, y_test, preds, i, targets, use_log, model_label)

    plt.tight_layout()
    # Preserve legacy filenames for XGBoost
    if model_label == "XGBoost":
        filename = "scatter_plot_log.png" if use_log else "scatter_plot.png"
    else:
        filename = f"scatter_plot_{model_label.lower()}_log.png" if use_log else f"scatter_plot_{model_label.lower()}.png"
    os.makedirs(os.path.join(RESULTS_PATH, run_id, "plots"), exist_ok=True)
    plt.savefig(os.path.join(RESULTS_PATH, run_id, "plots", filename), bbox_inches='tight')
    plt.close()
    
    # Save separate PNG for first graph only
    first_fig = plt.figure(figsize=(6, 6))
    first_ax = first_fig.add_subplot(111)
    create_single_scatter_plot(first_ax, test_data, y_test, preds, 0, targets, use_log, model_label)
    
    if model_label == "XGBoost":
        first_filename = "scatter_plot_first_log.png" if use_log else "scatter_plot_first.png"
    else:
        first_filename = f"scatter_plot_{model_label.lower()}_first_log.png" if use_log else f"scatter_plot_{model_label.lower()}_first.png"
    plt.savefig(os.path.join(RESULTS_PATH, run_id, "plots", first_filename), bbox_inches='tight')
    plt.close(first_fig)


def plot_time_series(test_data, y_test, preds, targets, alpha=0.5, linewidth=0.5):
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

    if y_test is None or y_test.size == 0:
        st.warning("y_test is empty. Displaying blank plots.")
        for i, ax in enumerate(axes.flatten()):
            ax.set_title(targets[i])
            ax.set_xlabel("Year")
        plt.tight_layout()
        st.pyplot(plt)
        return

    for i, ax in enumerate(axes.flatten()):
        create_single_timeseries_plot(ax, test_data, y_test, preds, i, targets, alpha, linewidth)

    plt.tight_layout()
    st.pyplot(plt)
    
    # Save separate PNG for first graph only
    first_fig = plt.figure(figsize=(6, 6))
    first_ax = first_fig.add_subplot(111)
    create_single_timeseries_plot(first_ax, test_data, y_test, preds, 0, targets, alpha, linewidth)
    
    os.makedirs(os.path.join(RESULTS_PATH, "plots"), exist_ok=True)
    plt.savefig(os.path.join(RESULTS_PATH, "plots", "timeseries_first.png"), bbox_inches='tight')
    plt.close(first_fig)


def get_shap_values(run_id, xgb, X_test):
    """
    Create SHAP plots for the XGBoost model.
    Args:
        xgb: Trained XGBoost model.
        X_test: Dataframe of data feature columns
        features: List of feature names.
        targets: List of target names.
    """
    logging.info("Creating SHAP explainer...")
    explainer = shap.TreeExplainer(xgb, approximate=True)
    logging.info("Calculating SHAP values...")
    shap_values = explainer.shap_values(X_test)
    np.save(os.path.join(RESULTS_PATH, run_id, "plots", "shap_values.npy"), shap_values)
    logging.info("SHAP values saved to shap_values.npy")


def transform_outputs_to_former_inputs(run_id, shap_values, targets, features):
    """
    Convert the output SHAP values to input SHAP values. For output variables in the 
    TOP 10 SHAP values, switch them to the inputs that influenced them, multiplying 
    the importance to its original value. Repeat until there are no more output 
    variables in the TOP 7 SHAP values.
    """
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
        sorted_df.to_csv(os.path.join(RESULTS_PATH, run_id, "plots", f"shap{i+1}_{target}.csv"), index=False)
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
        new_sorted_df.to_csv(os.path.join(RESULTS_PATH, run_id, "plots", f"shap{i+1}_{target}_input_only.csv"), index=False)

    with open(os.path.join(RESULTS_PATH, run_id, "plots", "feature_renaming.json"), 'w') as json_file:
        json.dump(feature_renaming, json_file, indent=4)

    return input_only


def draw_shap_plot(run_id, shap_values, X_test, features, targets):
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
            shap.summary_plot(shap_values[:, :, i], X_test, feature_names=features, max_display=n, plot_type="violin", show=False)
            fig_shap.tight_layout()
            fig_shap.savefig(temp_filename, format='png', bbox_inches='tight')
            plt.close(fig_shap)  # Close the SHAP figure
            img = Image.open(temp_filename)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title("Impact on " + targets[i] + " (" + OUTPUT_UNITS[i] + ")")
            os.remove(temp_filename)
        else:
            ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.join(RESULTS_PATH, run_id,  "plots"), exist_ok=True)
    plt.savefig(os.path.join(RESULTS_PATH, run_id, "plots", "shap_plot.png"))
    plt.close()


def plot_shap(run_id, xgb, X_test_with_index, features, targets):
    logging.info("Creating SHAP plots...")
    # Remove non-feature columns from X_test if they exist
    X_test = X_test_with_index.drop(columns=NON_FEATURE_COLUMNS, errors="ignore")
    X_test = X_test.reset_index(drop=True)

    # Subsample if needed
    if X_test.shape[0] > 100:
        indices = np.random.choice(X_test.shape[0], 100, replace=False)
        X_test = X_test.iloc[indices]

    get_shap_values(run_id, xgb, X_test)
    logging.info("Transforming outputs to former inputs...")
    shap_values = np.load(os.path.join(RESULTS_PATH, run_id, "plots", "shap_values.npy"), allow_pickle=True)
    shap_values = transform_outputs_to_former_inputs(run_id, shap_values, targets, features)
    logging.info("Drawing SHAP plots...")
    draw_shap_plot(run_id, shap_values, X_test, features, targets)


# def shap_plot_per_target(models, X_test, features, targets):
#     # subsample if needed
#     if X_test.shape[0] > 500:
#         indices = np.random.choice(X_test.shape[0], 500, replace=False)
#         X_test = X_test[indices]
#     n = 8
#     plt.rcParams.update({'font.size': 12}) 

#     if not os.path.exists("plots"):
#         os.makedirs("plots")
#     if not os.path.exists("plots/temp"):
#         os.makedirs("plots/temp")

#     plot_paths = []
#     units = ["Mt CO2/yr", "Mt CH4/yr", "Mt N2O/yr", "PJ/yr", "PJ/yr", "PJ/yr", "PJ/yr", "PJ/yr", "PJ/yr"]
#     for i, target in enumerate(targets):
#         explainer = shap.TreeExplainer(models[target])
#         shap_values = explainer.shap_values(X_test)
#         shap.plots.violin(shap_values, X_test, feature_names=features, max_display=n, show=False)
#         plt.gca().yaxis.set_tick_params(labelsize=10)
#         plt.gca().set_xlabel("Impact on " + target +" (" + units[i] + ")")
#         plt.tight_layout()
#         plot_path = os.path.join("plots/temp", f"shap_{i}.png")
#         plt.savefig(plot_path)
#         plot_paths.append(plot_path)
#         plt.close()
    
#     html_content = '<div style="display: grid; grid-template-columns: repeat(2, 1fr); grid-gap: 10px;">'
#     timestamp = int(time.time())
#     for plot_path in plot_paths:
#         html_content += f'<div><img src="{plot_path}?t={timestamp}" style="width: 100%;" /></div>'
#     html_content += '</div>'
#     # save the HTML content to a file
#     with open("plots/shap_plots.html", "w") as f:
#         f.write(html_content)
#     display(HTML(html_content))
