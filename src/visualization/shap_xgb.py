# XGBoost SHAP plotting (migrated from utils.plot_shap_xgb)
import os, logging, numpy as np, pandas as pd, shap, xgboost as xgb
from typing import List, Optional, Dict
from configs.paths import RESULTS_PATH
from configs.data import NON_FEATURE_COLUMNS, OUTPUT_UNITS
from .helpers import make_grid, render_external_plot, build_feature_display_names

__all__ = ['get_shap_values','transform_outputs_to_former_inputs','draw_shap_plot','plot_shap']

# Original content (verbatim, minimal edits only for path):

def get_shap_values(run_id, X_test: pd.DataFrame):
    logging.info("Loading XGBoost model...")
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(RESULTS_PATH, run_id, "checkpoints", "final_best.json"))
    logging.info("Creating SHAP explainer...")
    explainer = shap.TreeExplainer(model, approximate=True)
    logging.info("Calculating SHAP values...")
    shap_values = explainer.shap_values(X_test)
    os.makedirs(os.path.join(RESULTS_PATH, run_id, "plots"), exist_ok=True)
    np.save(os.path.join(RESULTS_PATH, run_id, "plots", "shap_values.npy"), shap_values)
    logging.info("SHAP values saved to shap_values.npy")

def transform_outputs_to_former_inputs(run_id: str, shap_values: np.ndarray, targets: List[str], features: List[str]) -> np.ndarray:
    import pandas as pd, json
    sorted_df_list = []
    for i, target in enumerate(targets):
        target_shap_values = np.abs(shap_values[:, :, i])
        mean_shap_values = np.mean(target_shap_values, axis=0)
        target_value = np.sum(mean_shap_values)
        importance = mean_shap_values / target_value
        sorted_df = pd.DataFrame({"Feature": features, "Importance": importance}).sort_values(by="Importance", ascending=False)
        sorted_df_list.append(sorted_df)
        os.makedirs(os.path.join(RESULTS_PATH, run_id, "plots", "csv"), exist_ok=True)
        sorted_df.to_csv(os.path.join(RESULTS_PATH, run_id, "plots", "csv", f"shap{i+1}_{target}.csv"), index=False)
        shap_values[:, :, i] = shap_values[:, :, i] / target_value
    input_only = shap_values.copy()
    feature_renaming = {}
    for i, target in enumerate(targets):
        feature_renaming[target] = {}
        for j in range(20):
            output = sorted_df_list[i].iloc[j]["Feature"]
            if output.startswith("prev") and not output.endswith(target):
                num, output_name = output.split("_")
                num = 1 if num == "prev" else int(num[4:])
                output_index = targets.index(output_name)
                old_feature = shap_values[:, j, i]
                for k in range(10):
                    new_input = sorted_df_list[output_index].iloc[k]["Feature"]
                    if not new_input.startswith("prev"):
                        new_features = shap_values[:, features.index(new_input), output_index] * old_feature
                        input_only[:, j, i] = new_features
                        feature_renaming[target][output] = ("prev_" if num == 1 else f"prev{num}_") + new_input
                        break
    with open(os.path.join(RESULTS_PATH, run_id, "plots", "csv", "feature_renaming.json"), 'w') as json_file:
        json.dump(feature_renaming, json_file, indent=4)
    return input_only

def draw_shap_plot(run_id, shap_values, X_test, features, targets, exclude_top=False, model_prefix="", xlim_range: Optional[tuple] = None):
    n_display = 8
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 12})
    num_targets = len(targets)
    fig, axes = make_grid(num_targets, base_figsize=(20, 20))
    from configs.data import CATEGORICAL_COLUMNS
    X_proc = X_test.copy()
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in X_proc.columns]
    for c in cat_cols:
        X_proc[c] = X_proc[c].astype('category').cat.codes
    X_values = X_proc.values.astype(np.float64)
    for i, ax in enumerate(axes):
        if i >= num_targets:
            ax.axis('off')
            continue
        def _plot(fig_local):
            if exclude_top:
                target_abs = np.abs(shap_values[:, :, i])
                mean_abs = target_abs.mean(axis=0)
                top_idx = int(np.argmax(mean_abs))
                mask = np.ones(len(features), dtype=bool)
                mask[top_idx] = False
                filtered_features = [f for j, f in enumerate(features) if mask[j]]
                filtered_display = build_feature_display_names(filtered_features)
                shap.summary_plot(
                    shap_values[:, mask, i],
                    X_values[:, mask],
                    feature_names=filtered_display,
                    max_display=n_display,
                    plot_type='violin',
                    show=False,
                )
            else:
                display_names = build_feature_display_names(features)
                shap.summary_plot(
                    shap_values[:, :, i],
                    X_values,
                    feature_names=display_names,
                    max_display=n_display,
                    plot_type='violin',
                    show=False,
                )
            # Set x-axis limits to zoom into configured range if specified
            if xlim_range is not None:
                plt.xlim(xlim_range[0], xlim_range[1])
            fig_local.tight_layout()
        render_external_plot(ax, _plot)
        title_suffix = " (excluding top feature)" if exclude_top else ""
        ax.set_title(f"Impact on {targets[i]} ({OUTPUT_UNITS[i]}){title_suffix}")
    fig.tight_layout()
    os.makedirs(os.path.join(RESULTS_PATH, run_id, 'plots'), exist_ok=True)
    prefix = f"{model_prefix}_" if model_prefix else ""
    filename = f"{prefix}shap_plot_no_first.png" if exclude_top else f"{prefix}shap_plot.png"
    fig.savefig(os.path.join(RESULTS_PATH, run_id, 'plots', filename))
    plt.close(fig)

def plot_xgb_shap(run_id, X_test_with_index, features, targets, xlim_range: Optional[tuple] = None):
    logging.info("Creating SHAP plots...")
    ckpt_path = os.path.join(RESULTS_PATH, run_id, "checkpoints", "final_best.json")
    if not os.path.exists(ckpt_path):
        logging.warning("Skipping SHAP plots: model checkpoint not found at %s", ckpt_path)
        return
    X_test = X_test_with_index.drop(columns=NON_FEATURE_COLUMNS, errors="ignore").reset_index(drop=True)
    if X_test.shape[0] > 100:
        import numpy as np
        indices = np.random.choice(X_test.shape[0], 100, replace=False)
        X_test = X_test.iloc[indices]
    get_shap_values(run_id, X_test)
    shap_values = np.load(os.path.join(RESULTS_PATH, run_id, "plots", "shap_values.npy"), allow_pickle=True)
    shap_values = transform_outputs_to_former_inputs(run_id, shap_values, targets, features)
    draw_shap_plot(run_id, shap_values, X_test, features, targets, exclude_top=False, xlim_range=xlim_range)
    draw_shap_plot(run_id, shap_values, X_test, features, targets, exclude_top=True, xlim_range=xlim_range)
