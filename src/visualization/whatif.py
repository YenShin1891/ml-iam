"""Figures for the what-if view: the lever overlay and the emulated trajectories."""

import datetime
import json
import os
from typing import Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

from configs.visualization import (
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    PLOT_GRID_COLS,
    PLOT_GRID_ROWS,
    TICK_LABELSIZE,
    TRAJECTORY_GRID_FIGSIZE,
    Y_AXIS_NBINS_TRAJECTORY,
)
from src.inference.whatif import LeverSpec, WhatifResult, band_position, interpolate_lever_path
from src.utils.utils import get_run_root
from .helpers import make_grid, output_unit
from .trajectories import format_large_numbers

HISTORY_COLOR = "#e0e0e0"
IAM_COLOR = "#444444"
BASELINE_PRED_COLOR = "#888888"
EDITED_COLOR = "#d62728"
AR6_BAND_COLOR = "#9ecae1"

# Stem of the files the view saves; the sidebar lists them by it.
PLOT_PREFIX = "whatif_"


def plot_whatif_grid(result: WhatifResult, *, ar6_band: Optional[pd.DataFrame] = None) -> Figure:
    """One panel per target: the IAM's values, and the emulator with and without the edits.

    *ar6_band* (columns target, Year, lo, hi) shades where the region's AR6
    scenarios lie, so a forecast can be read against them.
    """
    targets = result.targets
    fig, axes = make_grid(len(targets), PLOT_GRID_ROWS, PLOT_GRID_COLS, base_figsize=TRAJECTORY_GRID_FIGSIZE)
    history = result.history_years
    band_by_target = (
        {t: g for t, g in ar6_band.groupby("target")}
        if ar6_band is not None and len(ar6_band) else {}
    )

    for i, ax in enumerate(axes):
        if i >= len(targets):
            ax.axis("off")
            continue
        target = targets[i]
        first = i == 0
        if history:
            ax.axvspan(
                history[0], history[-1], color=HISTORY_COLOR, alpha=0.6, zorder=0,
                label="History (inputs fixed)" if first else None,
            )
        band = band_by_target.get(target)
        if band is not None:
            finite = band[np.isfinite(band["lo"]) & np.isfinite(band["hi"])].sort_values("Year")
            if len(finite):
                ax.fill_between(
                    finite["Year"], finite["lo"], finite["hi"],
                    color=AR6_BAND_COLOR, alpha=0.35, zorder=1,
                    label="AR6 scenarios, 5–95%" if first else None,
                )
        iam = result.iam[target].dropna()
        if len(iam):
            ax.plot(
                iam.index, iam.values, color=IAM_COLOR, linestyle="-", linewidth=1.8,
                marker="o", markersize=3, zorder=3,
                label="IAM (baseline scenario)" if first else None,
            )
        baseline = result.pred_baseline[target]
        ax.plot(
            baseline.index, baseline.values, color=BASELINE_PRED_COLOR, linestyle="--",
            linewidth=1.8, zorder=4, label="Emulator, unchanged inputs" if first else None,
        )
        edited = result.pred_edited[target]
        ax.plot(
            edited.index, edited.values, color=EDITED_COLOR, linestyle="-",
            linewidth=2.2, zorder=5, label="Emulator, edited inputs" if first else None,
        )
        ax.set_xlabel("Year", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(f"{target} ({output_unit(target)})", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", which="major", labelsize=TICK_LABELSIZE)
        ax.yaxis.set_major_formatter(FuncFormatter(format_large_numbers))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=Y_AXIS_NBINS_TRAJECTORY))

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, fontsize=LEGEND_FONTSIZE)
    fig.suptitle(
        f"{result.region}: {result.model} / {result.scenario} ({result.category})",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    fig.tight_layout()
    return fig


def plot_lever_overlay(
    specs: Sequence[LeverSpec],
    edits: Mapping[str, Mapping[int, float]],
    *,
    mode: str = "band",
    features: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (11, 4.5),
) -> Figure:
    """Every chosen lever on one axis: baseline dotted, edited path solid.

    ``mode="band"`` plots each value's position within the region's AR6
    range (0 = lower quantile, 1 = upper), which puts inputs of any unit on
    one scale and shows at a glance how far an edit strays from the
    scenarios the model learned from.  ``mode="ratio"`` plots the edited
    path relative to the baseline instead.
    """
    by_feature = {spec.feature: spec for spec in specs}
    if features is None:
        features = [spec.feature for spec in specs if spec.feature in edits]
    chosen = [f for f in features if f in by_feature]

    fig, ax = plt.subplots(figsize=figsize)
    if not chosen:
        ax.text(0.5, 0.5, "No levers selected", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig

    history = by_feature[chosen[0]].history_years
    if history:
        ax.axvspan(history[0], history[-1], color=HISTORY_COLOR, alpha=0.6, zorder=0, label="History (fixed)")
    if mode == "band":
        ax.axhspan(0.0, 1.0, color=AR6_BAND_COLOR, alpha=0.25, zorder=0, label="AR6 scenarios, 5–95%")
        ax.set_ylabel("Position within the AR6 range\n(0 = 5th, 1 = 95th percentile)")
    else:
        ax.axhline(1.0, color="#999999", linewidth=1.0, zorder=0)
        ax.set_ylabel("Value relative to the baseline (×)")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k, feature in enumerate(chosen):
        spec = by_feature[feature]
        color = colors[k % len(colors)]
        years = spec.years
        edited = pd.Series(
            interpolate_lever_path(years, spec.history_years, spec.baseline, edits.get(feature, {})),
            index=years,
        )
        if mode == "band":
            y_base = band_position(spec.baseline, spec.band)
            y_edit = band_position(edited, spec.band)
        else:
            base = spec.baseline.replace(0.0, np.nan)
            y_base = pd.Series(np.ones(len(years)), index=years)
            y_edit = edited / base
        ax.plot(years, y_base.values, color=color, linestyle=":", linewidth=1.4, zorder=2)
        ax.plot(
            years, y_edit.values, color=color, linestyle="-",
            linewidth=2.2 if feature in edits else 1.2, zorder=3, label=feature,
        )

    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([0], [0], color="#333333", linestyle=":", linewidth=1.4),
        Line2D([0], [0], color="#333333", linestyle="-", linewidth=2.2),
    ]
    labels += ["baseline", "edited"]
    ax.legend(handles, labels, fontsize=LEGEND_FONTSIZE, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.set_xlabel("Year")
    fig.tight_layout()
    return fig


def save_whatif_outputs(run_id: str, fig: Figure, metadata: dict, timestamp: Optional[str] = None) -> Tuple[str, str]:
    """Save the figure and its metadata where the dashboard lists saved plots."""
    plots_dir = os.path.join(get_run_root(run_id), "saved_dashboard_plots")
    os.makedirs(plots_dir, exist_ok=True)
    stamp = timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(plots_dir, f"{PLOT_PREFIX}{stamp}.png")
    metadata_path = os.path.join(plots_dir, f"{PLOT_PREFIX}{stamp}_metadata.json")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return png_path, metadata_path
