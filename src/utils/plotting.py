import json
import logging
import os
import tempfile
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from PIL import Image
import shap
import streamlit as st
from tqdm import tqdm

from configs.config import INDEX_COLUMNS, NON_FEATURE_COLUMNS, RESULTS_PATH

def preprocess_data(
    test_data: pd.DataFrame, 
    y_test: np.ndarray, 
    preds: np.ndarray, 
    target_index: int
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    mask = ~np.isnan(y_test[:, target_index]) & ~np.isnan(preds[:, target_index])
    test_data_valid = test_data[mask].reset_index(drop=True)
    y_test_valid = y_test[mask, target_index]
    preds_valid = preds[mask, target_index]
    return test_data_valid, y_test_valid, preds_valid


def configure_axes(
    ax, 
    use_log: bool, 
    min_val: float, 
    max_val: float, 
    xlabel: str, 
    ylabel: str
) -> None:
    if use_log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_scatter(
    run_id: str,
    test_data: pd.DataFrame,
    y_test: np.ndarray,
    preds: np.ndarray,
    targets: List[str],
    use_log: bool = False,
    model_label: str = "XGBoost"
) -> None:
    logging.info("Creating scatter plot...")
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))

    for i, ax in enumerate(axes.flatten()):
        test_data_valid, y_test_valid, preds_valid = preprocess_data(test_data, y_test, preds, i)
        unique_years = sorted(test_data_valid['Year'].unique())
        colors = cm.viridis(np.linspace(0, 1, len(unique_years)))

        for year, color in zip(unique_years, colors):
            group_df = test_data_valid[test_data_valid['Year'] == year]
            group_indices = group_df.index
            group_y_test = y_test_valid[group_indices]
            group_preds = preds_valid[group_indices]
            ax.scatter(group_y_test, group_preds, alpha=0.5, color=color, label=year)

        ax.set_title(targets[i])
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
            "IAM (log scale)" if use_log else "IAM",
            f"{model_label} (log scale)" if use_log else model_label,
        )
        ax.legend(title='Year', loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

    plt.tight_layout()
    # Preserve legacy filenames for XGBoost
    if model_label == "XGBoost":
        filename = "scatter_plot_log.png" if use_log else "scatter_plot.png"
    else:
        filename = f"scatter_plot_{model_label.lower()}_log.png" if use_log else f"scatter_plot_{model_label.lower()}.png"
    os.makedirs(os.path.join(RESULTS_PATH, run_id, "plots"), exist_ok=True)
    plt.savefig(os.path.join(RESULTS_PATH, run_id, "plots", filename), bbox_inches='tight')
    plt.close()


def plot_time_series(
    test_data: pd.DataFrame,
    y_test: Optional[np.ndarray],
    preds: np.ndarray,
    targets: List[str],
    alpha: float = 0.5,
    linewidth: float = 0.5
) -> None:
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))

    if y_test is None or y_test.size == 0:
        st.warning("y_test is empty. Displaying blank plots.")
        for i, ax in enumerate(axes.flatten()):
            ax.set_title(targets[i])
            ax.set_xlabel("Year")
        plt.tight_layout()
        st.pyplot(plt)
        return

    for i, ax in enumerate(axes.flatten()):
        test_data_valid, y_test_valid, preds_valid = preprocess_data(test_data, y_test, preds, i)

        for group_key, group_df in test_data_valid.groupby(INDEX_COLUMNS):
            group_years = group_df['Year']
            group_indices = group_df.index
            group_y_test = y_test_valid[group_indices]
            group_preds = preds_valid[group_indices]

            ax.plot(group_years, group_y_test, label='IAM', alpha=alpha, linewidth=linewidth)
            ax.plot(group_years, group_preds, label='XGBoost', alpha=alpha, linewidth=linewidth)
            ax.fill_between(group_years, group_y_test, group_preds, alpha=0.1)

        ax.set_title(targets[i])
        ax.set_xlabel("Year")

    plt.tight_layout()
    st.pyplot(plt)


def get_shap_values(run_id: str, xgb, X_test: pd.DataFrame) -> None:
    logging.info("Creating SHAP explainer...")
    explainer = shap.TreeExplainer(xgb, approximate=True)
    logging.info("Calculating SHAP values...")
    shap_values = explainer.shap_values(X_test)
    np.save(os.path.join(RESULTS_PATH, run_id, "plots", "shap_values.npy"), shap_values)
    logging.info("SHAP values saved to shap_values.npy")


def transform_outputs_to_former_inputs(
    run_id: str, 
    shap_values: np.ndarray, 
    targets: List[str], 
    features: List[str]
) -> None:
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
    units = ["Mt CO2/yr", "Mt CH4/yr", "Mt N2O/yr", "PJ/yr", "PJ/yr", "PJ/yr", "PJ/yr", "PJ/yr", "PJ/yr"]
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
            ax.set_title("Impact on " + targets[i] + " (" + units[i] + ")")
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
