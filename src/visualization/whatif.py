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
from matplotlib import patheffects
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


def plot_whatif_comparison(ensembles, reference_samples=None, *, year=2050, targets=None):
    """Paired time-series and AR6/generated distribution panels, by baseline group.

    ensembles maps group labels to {original: WhatifResult, predictions: {label: frame}}.
    The same group shades are used on both panels. All generated paths are
    drawn; the shaded envelope is min/max, not a confidence interval.
    """
    if not ensembles:
        raise ValueError("At least one completed ensemble is required.")
    first = next(iter(ensembles.values()))["original"]
    targets = list(first.targets if targets is None else targets)
    if not targets:
        raise ValueError("Choose at least one output.")
    if any(t not in bundle["original"].targets for t in targets for bundle in ensembles.values()):
        raise ValueError("Every selected output must be present in every ensemble.")
    ncols = 3 if len(targets) >= 9 else min(2, len(targets))
    nrows = (len(targets) + ncols - 1) // ncols
    fig = plt.figure(figsize=(9 * ncols, 3.9 * nrows + 1.1))
    outer = fig.add_gridspec(nrows, ncols, left=.065, right=.99, top=.92, bottom=.08, hspace=.4, wspace=.22)
    groups = list(ensembles)
    palettes = ["Blues", "Reds", "Oranges", "Purples", "Greens", "YlOrBr", "PuRd", "Greys", "YlGn"]
    for i, target in enumerate(targets):
        pair = outer[i // ncols, i % ncols].subgridspec(1, 2, width_ratios=[1.4, 1], wspace=.08)
        ax = fig.add_subplot(pair[0])
        dist = fig.add_subplot(pair[1], sharey=ax)
        cmap = plt.get_cmap(palettes[i % len(palettes)])
        colors = [cmap(.42 + .4 * k / max(1, len(groups) - 1)) for k in range(len(groups))]
        for k, (group, bundle) in enumerate(ensembles.items()):
            original = bundle["original"].pred_baseline[target]
            paths = pd.concat([prediction[target].rename(label) for label, prediction in bundle["predictions"].items()], axis=1).sort_index()
            color = colors[k]
            if not paths.empty:
                ax.fill_between(paths.index, paths.min(axis=1), paths.max(axis=1), color=color, alpha=.15, linewidth=0)
                for column in paths:
                    ax.plot(paths.index, paths[column], color=color, linewidth=.55, alpha=.32)
            line, = ax.plot(original.index, original.values, color="white", linestyle=(0, (3, 2)), linewidth=1.5, zorder=5)
            line.set_path_effects([patheffects.Stroke(linewidth=2.6, foreground=color), patheffects.Normal()])
            finite_original = original.dropna()
            if len(finite_original):
                last_year = finite_original.index[-1]
                ax.annotate(group, (last_year, finite_original.iloc[-1]), xytext=(-5, 5),
                            textcoords="offset points", ha="right", color=color, fontsize=9, fontweight="bold",
                            path_effects=[patheffects.withStroke(linewidth=2, foreground="white")])
            reference = np.array([], dtype=float)
            if reference_samples is not None and len(reference_samples):
                matched = reference_samples[(reference_samples["group"] == group) & (reference_samples["target"] == target) & (reference_samples["Year"] == year)]
                reference = pd.to_numeric(matched["value"], errors="coerce").to_numpy()
            synthetic = paths.loc[year].to_numpy(dtype=float) if year in paths.index else np.array([])
            for offset, values, filled in ((-.18, reference, False), (.18, synthetic, True)):
                values = values[np.isfinite(values)]
                position = k + offset
                if len(values):
                    box = dist.boxplot([values], positions=[position], widths=.25, whis=(5, 95),
                                       showfliers=False, patch_artist=True, manage_ticks=False)
                    box["boxes"][0].set(facecolor=color if filled else "white", edgecolor=color, alpha=.75)
                    for item in box["whiskers"] + box["caps"] + box["medians"]:
                        item.set(color=color, linewidth=.9)
                    # Deterministic jitter: every sample is visible, including boxplot outliers.
                    jitter = .09 * np.sin(np.arange(len(values)) * 2.399963)
                    dist.scatter(position + jitter, values, s=3, color=color, alpha=.4, linewidths=0)
                else:
                    dist.text(position, .03, "no data", transform=dist.get_xaxis_transform(),
                              rotation=90, ha="center", va="bottom", fontsize=7, color="#777777")
                dist.text(position, 1.02, f"n={len(values)}", transform=dist.get_xaxis_transform(), ha="center", fontsize=7, color="#666666")
            if year in original.index and np.isfinite(original.loc[year]):
                dist.scatter(k + .18, original.loc[year], marker="D", s=22, color="#202020", zorder=5)
        ax.set_title("Synthetic time series", loc="left", fontsize=10, pad=17)
        dist.set_title(f"{year} distribution", loc="left", fontsize=10, pad=17)
        ax.text(-.17, 1.075, chr(97 + i), transform=ax.transAxes, fontsize=13, fontweight="bold")
        ax.axvline(year, color="#777777", linewidth=.65, linestyle=":", alpha=.6)
        ax.set_ylabel(f"{target.replace('|', ' · ')}\n({output_unit(target)})", fontsize=10)
        ax.set_xlabel("Year", fontsize=10)
        dist.set_xticks(range(len(groups)))
        dist.set_xticklabels(groups, fontsize=9)
        dist.set_xlim(-.55, len(groups) - .45)
        dist.set_xlabel("Baseline category group", fontsize=9)
        dist.tick_params(axis="y", labelleft=False)
        for axis in (ax, dist):
            axis.yaxis.set_major_formatter(FuncFormatter(format_large_numbers))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
            axis.tick_params(labelsize=9)
            for spine in axis.spines.values():
                spine.set_color("#999999")
    handles = [Line2D([], [], color="#65829b", linewidth=.8, label="Every High/Low path (shading: full range)"),
               Line2D([], [], color="#444444", linestyle="--", label="Original emulation"),
               Line2D([], [], marker="s", markerfacecolor="white", markeredgecolor="#65829b", linestyle="", label="AR6 (left box)"),
               Line2D([], [], marker="s", color="#65829b", linestyle="", label="Synthetic (right box)"),
               Line2D([], [], marker="D", color="#202020", linestyle="", label=f"Original at {year}")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .005), ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f"{first.region} · High/Low combinations and AR6 comparison", fontsize=15, y=.99)
    fig.text(.5, .955, "Groups follow the source baseline; generated paths have not been reclassified. Boxes: median / IQR; whiskers: 5–95%.", ha="center", fontsize=9, color="#555555")
    return fig


def plot_whatif_grid(
    result: WhatifResult, *, ar6_band: Optional[pd.DataFrame] = None,
    combinations: Optional[Mapping[str, pd.DataFrame]] = None,
) -> Figure:
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
        baseline_line, = ax.plot(
            baseline.index, baseline.values, color=BASELINE_PRED_COLOR, linestyle="--",
            linewidth=1.8, zorder=4, label="Emulator, unchanged inputs" if first else None,
        )
        if combinations is not None:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            for j, prediction in enumerate(combinations.values()):
                path = prediction[target]
                ax.plot(
                    path.index, path.values, color=colors[i % len(colors)],
                    linewidth=0.8, alpha=0.25, zorder=2,
                    label=f"High/Low combinations ({len(combinations)})" if first and j == 0 else None,
                )
            # Keep the original emulation visible above even a dense ensemble.
            baseline_line.set_color("#111111")
        else:
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


def save_whatif_outputs(run_id: str, fig: Figure, metadata: dict, timestamp: Optional[str] = None, *, output_dir: Optional[str] = None) -> Tuple[str, str]:
    """Save the figure and its metadata where the dashboard lists saved plots."""
    plots_dir = output_dir if output_dir is not None else os.path.join(get_run_root(run_id), "saved_dashboard_plots")
    os.makedirs(plots_dir, exist_ok=True)
    stamp = timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(plots_dir, f"{PLOT_PREFIX}{stamp}.png")
    metadata_path = os.path.join(plots_dir, f"{PLOT_PREFIX}{stamp}_metadata.json")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return png_path, metadata_path
