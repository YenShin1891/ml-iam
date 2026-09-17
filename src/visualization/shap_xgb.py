"""SHAP plots for the XGBoost models."""
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import shap

from src.utils.utils import get_run_root
from configs.data import NON_FEATURE_COLUMNS, CATEGORICAL_COLUMNS
from src.data.preprocess import encode_categorical_columns
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
    output_unit,
    render_external_plot,
    build_feature_display_names,
    draw_shap_beeswarm,
    filter_index_frame_by_region,
    sample_scenario_groups,
)

__all__ = ['get_shap_values', 'normalise_shap_values', 'write_shap_rankings', 'draw_shap_plot', 'plot_xgb_shap']


def get_shap_values(run_id, X_test: pd.DataFrame, targets: Optional[List[str]] = None) -> np.ndarray:
    """SHAP values shaped (rows, features, targets), also saved to plots/shap_values.npy."""
    from src.trainers.xgb_trainer import load_final_xgb_model

    logging.info("Loading XGBoost model...")
    model = load_final_xgb_model(run_id, targets)
    logging.info("Creating SHAP explainer...")

    per_target_models = getattr(model, "models", None)
    if per_target_models is not None:
        # One booster per target: explain each separately and stack to the
        # (rows, features, targets) layout the multi-output explainer returns.
        logging.info("Calculating SHAP values for %d per-target models...", len(per_target_models))
        shap_values = np.stack(
            [
                shap.TreeExplainer(m, approximate=True).shap_values(X_test)
                for m in per_target_models
            ],
            axis=-1,
        )
    else:
        logging.info("Calculating SHAP values...")
        shap_values = shap.TreeExplainer(model, approximate=True).shap_values(X_test)

    os.makedirs(os.path.join(get_run_root(run_id), "plots"), exist_ok=True)
    np.save(os.path.join(get_run_root(run_id), "plots", "shap_values.npy"), shap_values)
    logging.info("SHAP values saved to shap_values.npy: shape %s", np.shape(shap_values))
    return shap_values


def normalise_shap_values(shap_values: np.ndarray) -> np.ndarray:
    """Scale each target's SHAP values by its sum of mean |SHAP|.

    Targets of different magnitude then share one axis.  Every feature keeps
    its own attribution, lagged columns of other targets included.  Returns a
    new array.
    """
    shap_values = np.array(shap_values, dtype=float)  # a copy: normalised in place below
    for i in range(shap_values.shape[2]):
        total = float(np.sum(np.mean(np.abs(shap_values[:, :, i]), axis=0)))
        if total > 0:
            shap_values[:, :, i] /= total
    return shap_values


def write_shap_rankings(run_id: str, shap_values: np.ndarray, targets: List[str], features: List[str]) -> None:
    """Write each target's mean |SHAP| ranking to plots/csv/shap<i>_<target>.csv."""
    csv_dir = os.path.join(get_run_root(run_id), "plots", "csv")
    os.makedirs(csv_dir, exist_ok=True)
    for i, target in enumerate(targets):
        mean_abs = np.mean(np.abs(shap_values[:, :, i]), axis=0)
        ranking = pd.DataFrame({"Feature": features, "Importance": mean_abs}).sort_values(
            by="Importance", ascending=False
        )
        ranking.to_csv(os.path.join(csv_dir, f"shap{i+1}_{target}.csv"), index=False)


def _feature_subset(target_shap: np.ndarray, exclude_top: bool) -> np.ndarray:
    """Column indices to plot: all, or all but the one with the largest mean |SHAP|."""
    indices = np.arange(target_shap.shape[1])
    if exclude_top:
        top_idx = int(np.argmax(np.abs(target_shap).mean(axis=0)))
        indices = indices[indices != top_idx]
    return indices


def draw_shap_plot(run_id, shap_values, X_test, features, targets, exclude_top=False, model_prefix="", xlim_range: Optional[tuple] = None, categories: Optional[Dict[str, list]] = None):
    n_display = SHAP_MAX_DISPLAY_EXCLUDE_TOP if exclude_top else SHAP_MAX_DISPLAY
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': SHAP_FONT_SIZE})
    num_targets = len(targets)
    fig, axes = make_grid(num_targets, base_figsize=SHAP_GRID_FIGSIZE)
    display_names_all = build_feature_display_names(features)
    X_proc = X_test.copy()
    # Feature values only drive the beeswarm colour axis.  They normally arrive
    # already encoded and scaled, in which case re-encoding them against the
    # label vocabulary would map every row to -1; only touch columns that are
    # still labels.
    label_cols = [
        c for c in CATEGORICAL_COLUMNS
        if c in X_proc.columns and not pd.api.types.is_numeric_dtype(X_proc[c])
    ]
    if label_cols:
        X_proc = encode_categorical_columns(X_proc, label_cols, categories)
    X_values = X_proc.values.astype(np.float64)

    # Create directory for individual plots
    indiv_plots_dir = os.path.join(get_run_root(run_id), 'plots', 'indiv_plots', 'shap')
    os.makedirs(indiv_plots_dir, exist_ok=True)

    title_suffix = " (excluding top feature)" if exclude_top else ""
    for i, ax in enumerate(axes):
        if i >= num_targets:
            ax.axis('off')
            continue
        target_shap = shap_values[:, :, i]  # [samples, features]
        indices = _feature_subset(target_shap, exclude_top)
        display_subset = [display_names_all[int(j)] for j in indices]
        target_subset = target_shap[:, indices]
        X_subset = X_values[:, indices]
        title = f"Impact on {targets[i]} ({output_unit(targets[i])}){title_suffix}"

        def _plot(fig_local):
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
        ax.set_title(title)

        # Save individual plot for this target
        fig_indiv = plt.figure(figsize=SHAP_INDIVIDUAL_FIGSIZE)
        ax_indiv = fig_indiv.add_subplot(111)
        draw_shap_beeswarm(
            ax_indiv,
            target_subset,
            X_subset,
            display_subset,
            max_display=n_display,
            xlim_range=xlim_range,
        )
        ax_indiv.set_title(title)
        fig_indiv.tight_layout()
        indiv_filename = f"{targets[i]}_no_top.png" if exclude_top else f"{targets[i]}.png"
        fig_indiv.savefig(os.path.join(indiv_plots_dir, indiv_filename), dpi=300, bbox_inches='tight')
        plt.close(fig_indiv)

    fig.tight_layout()
    os.makedirs(os.path.join(get_run_root(run_id), 'plots'), exist_ok=True)
    prefix = f"{model_prefix}_" if model_prefix else ""
    filename = f"{prefix}shap_plot_no_first.png" if exclude_top else f"{prefix}shap_plot.png"
    fig.savefig(os.path.join(get_run_root(run_id), 'plots', filename))
    plt.close(fig)

def plot_xgb_shap(
    run_id,
    X_test_with_index,
    features,
    targets,
    xlim_range: Optional[tuple] = None,
    region: Optional[str] = DEFAULT_REGION,
    index_region: Optional[pd.Series] = None,
    categories: Optional[Dict[str, list]] = None,
):
    from src.trainers.xgb_trainer import has_final_xgb_model

    logging.info("Creating SHAP plots...")
    if not has_final_xgb_model(run_id):
        logging.warning(
            "Skipping SHAP plots: no final model found under %s",
            os.path.join(get_run_root(run_id), "checkpoints"),
        )
        return
    # Optional region filter (prefix matching like "R10" -> "R10*").  The
    # frame's Region column is scaled for the model, so callers pass the raw
    # labels aligned to its rows as *index_region*.
    X_filtered, _, pre_rows, post_rows, matched_values, mode = filter_index_frame_by_region(
        X_test_with_index,
        region,
        log_prefix="XGB SHAP region filter",
        region_series=index_region,
    )
    if region is not None and not matched_values:
        # Same guard as the sequence models: plotting every region under a
        # filename that names one is worse than plotting nothing.
        logging.error(
            "Skipping XGB SHAP plots: region filter %r matched no rows of %d.",
            region, pre_rows,
        )
        return
    if matched_values:
        logging.info("Applied region filter '%s' (%s): %d -> %d rows", region, mode, pre_rows, post_rows)
    # Scenario-based sampling on the full index frame (Model/Scenario as group keys)
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
    shap_values = get_shap_values(run_id, X_test, targets=targets)
    shap_values = normalise_shap_values(shap_values)
    write_shap_rankings(run_id, shap_values, targets, features)
    draw_shap_plot(run_id, shap_values, X_test, features, targets, exclude_top=False, xlim_range=xlim_range, categories=categories)
    draw_shap_plot(run_id, shap_values, X_test, features, targets, exclude_top=True, xlim_range=xlim_range, categories=categories)

