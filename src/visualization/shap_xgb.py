# XGBoost SHAP plotting (migrated from utils.plot_shap_xgb)
import os, logging, numpy as np, pandas as pd, shap, xgboost as xgb
from typing import List, Optional, Dict
from configs.paths import RESULTS_PATH
from configs.data import NON_FEATURE_COLUMNS, OUTPUT_UNITS, CATEGORICAL_COLUMNS, REGION_CATEGORIES
from configs.visualization import (
    DEFAULT_REGION,
    SHAP_FONT_SIZE,
    SHAP_GRID_FIGSIZE,
    SHAP_INDIVIDUAL_FIGSIZE,
    SHAP_MAX_DISPLAY,
    SHAP_MAX_DISPLAY_EXCLUDE_TOP,
)
from .helpers import (
    make_grid,
    render_external_plot,
    build_feature_display_names,
    draw_shap_beeswarm,
    filter_index_frame_by_region,
    sample_scenario_groups,
)

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
    n_display = SHAP_MAX_DISPLAY_EXCLUDE_TOP if exclude_top else SHAP_MAX_DISPLAY
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': SHAP_FONT_SIZE})
    num_targets = len(targets)
    fig, axes = make_grid(num_targets, base_figsize=SHAP_GRID_FIGSIZE)
    X_proc = X_test.copy()
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in X_proc.columns]
    for c in cat_cols:
        if c == 'Region':
            X_proc[c] = (
                pd.Categorical(X_proc[c].astype(str), categories=REGION_CATEGORIES, ordered=True)
                .codes
            )
        else:
            X_proc[c] = X_proc[c].astype('category').cat.codes
    X_values = X_proc.values.astype(np.float64)

    # Create directory for individual plots
    indiv_plots_dir = os.path.join(RESULTS_PATH, run_id, 'plots', 'indiv_plots', 'shap')
    os.makedirs(indiv_plots_dir, exist_ok=True)

    for i, ax in enumerate(axes):
        if i >= num_targets:
            ax.axis('off')
            continue
        def _plot(fig_local):
            # Prepare SHAP matrix for this target
            target_shap = shap_values[:, :, i]  # [samples, features]

            # Optionally exclude the single top feature by |SHAP|
            indices = np.arange(target_shap.shape[1])
            if exclude_top:
                mean_abs = np.abs(target_shap).mean(axis=0)
                top_idx = int(np.argmax(mean_abs))
                indices = indices[indices != top_idx]

            display_names_all = build_feature_display_names(features)
            display_subset = [display_names_all[int(j)] for j in indices]
            target_subset = target_shap[:, indices]
            X_subset = X_values[:, indices]

            ax_local = fig_local.add_subplot(111)
            draw_shap_beeswarm(
                ax_local,
                target_subset,
                X_subset,
                display_subset,
                max_display=n_display,
                xlim_range=xlim_range,
            )
            fig_local.tight_layout()
        render_external_plot(ax, _plot)
        title_suffix = " (excluding top feature)" if exclude_top else ""
        ax.set_title(f"Impact on {targets[i]} ({OUTPUT_UNITS[i]}){title_suffix}")

        # Save individual plot for this target
        fig_indiv = plt.figure(figsize=SHAP_INDIVIDUAL_FIGSIZE)
        target_shap = shap_values[:, :, i]
        indices = np.arange(target_shap.shape[1])
        if exclude_top:
            mean_abs = np.abs(target_shap).mean(axis=0)
            top_idx = int(np.argmax(mean_abs))
            indices = indices[indices != top_idx]

        display_names_all = build_feature_display_names(features)
        display_subset = [display_names_all[int(j)] for j in indices]
        target_subset = target_shap[:, indices]

        ax_indiv = fig_indiv.add_subplot(111)
        draw_shap_beeswarm(
            ax_indiv,
            target_subset,
            X_values[:, indices],
            display_subset,
            max_display=n_display,
            xlim_range=xlim_range,
        )
        plt.title(f"Impact on {targets[i]} ({OUTPUT_UNITS[i]}){title_suffix}")
        plt.tight_layout()
        indiv_filename = f"{targets[i]}_no_top.png" if exclude_top else f"{targets[i]}.png"
        fig_indiv.savefig(os.path.join(indiv_plots_dir, indiv_filename), dpi=300, bbox_inches='tight')
        plt.close(fig_indiv)

    fig.tight_layout()
    os.makedirs(os.path.join(RESULTS_PATH, run_id, 'plots'), exist_ok=True)
    prefix = f"{model_prefix}_" if model_prefix else ""
    filename = f"{prefix}shap_plot_no_first.png" if exclude_top else f"{prefix}shap_plot.png"
    fig.savefig(os.path.join(RESULTS_PATH, run_id, 'plots', filename))
    plt.close(fig)

def plot_xgb_shap(
    run_id,
    X_test_with_index,
    features,
    targets,
    xlim_range: Optional[tuple] = None,
    region: Optional[str] = DEFAULT_REGION,
    index_region: Optional[pd.Series] = None,
):
    logging.info("Creating SHAP plots...")
    ckpt_path = os.path.join(RESULTS_PATH, run_id, "checkpoints", "final_best.json")
    if not os.path.exists(ckpt_path):
        logging.warning("Skipping SHAP plots: model checkpoint not found at %s", ckpt_path)
        return
    # Optional region filter (robust, supports prefix matching like "R10" -> "R10*").
    # If index_region is provided, it is treated as the raw Region labels aligned to rows.
    if region is not None and (index_region is not None or (isinstance(X_test_with_index, pd.DataFrame) and "Region" in X_test_with_index.columns)):
        try:
            X_test_with_index, idx, pre_rows, post_rows, matched_values, mode = filter_index_frame_by_region(
                X_test_with_index,
                region,
                log_prefix="XGB SHAP region filter",
                region_series=index_region,
            )
            if idx is not None and index_region is not None:
                index_region = index_region.reset_index(drop=True).iloc[idx].reset_index(drop=True)
            if matched_values:
                logging.info(
                    "Applied region filter '%s' (%s): %d -> %d rows",
                    region,
                    mode,
                    pre_rows,
                    post_rows,
                )
        except Exception as e:
            logging.warning("Failed region filter using raw labels due to: %s; proceeding without filter", e)
    # Scenario-based sampling on the full index frame (Model/Scenario as group keys)
    X_filtered = X_test_with_index
    group_keys, total_groups, used_groups, group_cols = sample_scenario_groups(
        X_filtered,
        log_prefix="XGB SHAP",
    )

    if group_cols and not group_keys.empty:
        X_joined = X_filtered.merge(group_keys, on=group_cols, how="inner")
    else:
        X_joined = X_filtered

    X_test = X_joined.drop(columns=NON_FEATURE_COLUMNS, errors="ignore").reset_index(drop=True)

    logging.info(
        "XGB SHAP: %d rows after region+scenario filtering (%d -> %d groups by %s)",
        X_test.shape[0],
        total_groups,
        used_groups,
        ",".join(group_cols) if group_cols else "<none>",
    )
    get_shap_values(run_id, X_test)
    shap_values = np.load(os.path.join(RESULTS_PATH, run_id, "plots", "shap_values.npy"), allow_pickle=True)
    shap_values = transform_outputs_to_former_inputs(run_id, shap_values, targets, features)
    draw_shap_plot(run_id, shap_values, X_test, features, targets, exclude_top=False, xlim_range=xlim_range)
    draw_shap_plot(run_id, shap_values, X_test, features, targets, exclude_top=True, xlim_range=xlim_range)

# Backward-compatible alias
def plot_shap(run_id, X_test_with_index, features, targets, xlim_range: Optional[tuple] = None, region: Optional[str] = DEFAULT_REGION, index_region: Optional[pd.Series] = None):
    return plot_xgb_shap(run_id, X_test_with_index, features, targets, xlim_range, region=region, index_region=index_region)
