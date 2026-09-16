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
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import patheffects
from matplotlib.ticker import FuncFormatter, MaxNLocator

from configs.data import OUTPUT_VARIABLES

from configs.visualization import (
    LEGEND_FONTSIZE,
    PLOT_GRID_COLS,
    PLOT_GRID_ROWS,
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

# Keep each output's palette consistent across the two result figures.
OUTPUT_PALETTES = dict(zip(OUTPUT_VARIABLES, (
    "Blues", "Reds", "Oranges", "Purples", "Greens", "YlOrBr", "PuRd", "Greys", "Teal",
)))
N2O_CMAP = LinearSegmentedColormap.from_list("N2O teal", ["#e0f4f4", "#61c2c7", "#00616b"])
CHART_LABEL_SIZE = 20
CHART_TICK_SIZE = 14
CHART_LEGEND_SIZE = 18


def _output_cmap(target):
    palette = OUTPUT_PALETTES.get(target, "Blues")
    return N2O_CMAP if palette == "Teal" else plt.get_cmap(palette)


def _style_result_axis(axis):
    axis.yaxis.set_major_formatter(FuncFormatter(format_large_numbers))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.tick_params(labelsize=CHART_TICK_SIZE, color="#777777")
    for spine in axis.spines.values():
        spine.set_color("#999999")
        spine.set_linewidth(.8)


def _result_legend(fig, handles, labels=None):
    fig.legend(handles=handles, labels=labels, loc="lower center",
               bbox_to_anchor=(.5, .12 / fig.get_figheight()), ncol=3,
               fontsize=CHART_LEGEND_SIZE, frameon=False, columnspacing=2)


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
    ncols = min(2, len(targets))
    nrows = (len(targets) + ncols - 1) // ncols
    fig = plt.figure(figsize=(9 * ncols, 4.0 * nrows + 1.5))
    height = fig.get_figheight()
    outer = fig.add_gridspec(nrows, ncols, left=.07, right=.985, top=1 - 1.5 / height, bottom=1.8 / height, hspace=.28, wspace=.24)
    groups = list(ensembles)
    for i, target in enumerate(targets):
        pair = outer[i // ncols, i % ncols].subgridspec(1, 2, width_ratios=[1.4, 1], wspace=.08)
        ax = fig.add_subplot(pair[0])
        dist = fig.add_subplot(pair[1], sharey=ax)
        cmap = _output_cmap(target)
        colors = [cmap(.42 + .4 * k / max(1, len(groups) - 1)) for k in range(len(groups))]
        group_labels = []
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
                group_labels.append(ax.annotate(group, (last_year, finite_original.iloc[-1]), xytext=(-5, 5),
                            textcoords="offset points", ha="right", color=color, fontsize=14, fontweight="bold",
                            path_effects=[patheffects.withStroke(linewidth=2, foreground="white")]))
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
                              rotation=90, ha="center", va="bottom", fontsize=10, color="#777777")
            counts = (np.isfinite(reference).sum(), np.isfinite(synthetic).sum())
            dist.text(k, .86, f"n={counts[0]}/{counts[1]}", transform=dist.get_xaxis_transform(),
                      ha="center", va="top", fontsize=12, color="#666666")
            if year in original.index and np.isfinite(original.loc[year]):
                dist.scatter(k + .18, original.loc[year], marker="D", s=22, color="#202020", zorder=5)
        # Reserve headroom inside both shared axes for headings and sample counts.
        lower, upper = ax.get_ylim()
        ax.set_ylim(lower, upper + .22 * (upper - lower))
        # Keep endpoint labels readable when unchanged paths finish close together.
        previous_y = -np.inf
        pixels_per_point = fig.dpi / 72
        for label in sorted(group_labels, key=lambda item: item.xy[1]):
            anchor_y = ax.transData.transform(label.xy)[1]
            text_y = max(anchor_y + 5 * pixels_per_point, previous_y + 17 * pixels_per_point)
            label.set_position((-5, (text_y - anchor_y) / pixels_per_point))
            previous_y = text_y
        for axis, heading in ((ax, "Synthetic time series"), (dist, f"{year} distribution")):
            axis.text(.03, .97, heading, transform=axis.transAxes, va="top", fontsize=18,
                      zorder=10, bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=2))
        ax.text(-.17, 1.02, chr(97 + i), transform=ax.transAxes, fontsize=19, fontweight="bold")
        ax.axvline(year, color="#777777", linewidth=.65, linestyle=":", alpha=.6)
        ax.set_ylabel(f"{target.replace('|', ' · ')}\n({output_unit(target)})", fontsize=CHART_LABEL_SIZE)
        ax.set_xlabel("Year", fontsize=CHART_LABEL_SIZE)
        dist.set_xticks(range(len(groups)))
        dist.set_xticklabels(groups, fontsize=CHART_TICK_SIZE)
        dist.set_xlim(-.55, len(groups) - .45)
        dist.set_xlabel("Source group", fontsize=CHART_LABEL_SIZE)
        dist.tick_params(axis="y", labelleft=False)
        for axis in (ax, dist):
            _style_result_axis(axis)
    handles = [Line2D([], [], color="#65829b", linewidth=.8, label="Every High/Low path (shading: full range)"),
               Line2D([], [], color="#444444", linestyle="--", label="Original emulation"),
               Line2D([], [], marker="s", markerfacecolor="white", markeredgecolor="#65829b", linestyle="", label="AR6 (left box)"),
               Line2D([], [], marker="s", color="#65829b", linestyle="", label="Synthetic (right box)"),
               Line2D([], [], marker="D", color="#202020", linestyle="", label=f"Original at {year}")]
    _result_legend(fig, handles)
    fig.suptitle(f"{first.region} · High/Low combinations and AR6 comparison", fontsize=26, y=1 - .12 / height)
    fig.text(.5, 1 - .68 / height, "Groups follow the source baseline; generated paths have not been reclassified.\nBoxes: median / IQR; whiskers: 5–95%; n: AR6 / generated samples.", ha="center", va="top", fontsize=16, color="#555555")
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
    fig, axes = make_grid(len(targets), PLOT_GRID_ROWS, PLOT_GRID_COLS, base_figsize=(18, 13.8))
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
        color = _output_cmap(target)(.72)
        if history:
            ax.axvspan(
                history[0], history[-1], facecolor="none", edgecolor="#777777",
                hatch="////", linewidth=0, zorder=0,
                label="History (inputs fixed)",
            )
        band = band_by_target.get(target)
        if band is not None:
            finite = band[np.isfinite(band["lo"]) & np.isfinite(band["hi"])].sort_values("Year")
            if len(finite):
                ax.fill_between(
                    finite["Year"], finite["lo"], finite["hi"],
                    color=_output_cmap(target)(.42), alpha=0.2, zorder=1,
                    label="AR6 scenarios, 5–95%",
                )
        iam = result.iam[target].dropna()
        if len(iam):
            ax.plot(
                iam.index, iam.values, color=IAM_COLOR, linestyle="-", linewidth=1.8,
                marker="o", markersize=3, zorder=3,
                label="IAM (baseline scenario)",
            )
        baseline = result.pred_baseline[target]
        baseline_line, = ax.plot(
            baseline.index, baseline.values, color=BASELINE_PRED_COLOR, linestyle="--",
            linewidth=1.8, zorder=4, label="Emulator, unchanged inputs",
        )
        if combinations is not None:
            for j, prediction in enumerate(combinations.values()):
                path = prediction[target]
                ax.plot(
                    path.index, path.values, color=color,
                    linewidth=0.8, alpha=0.25, zorder=2,
                    label=f"High/Low combinations ({len(combinations)})" if j == 0 else None,
                )
            # Keep the original emulation visible above even a dense ensemble.
            baseline_line.set_color("#111111")
        else:
            edited = result.pred_edited[target]
            ax.plot(
                edited.index, edited.values, color=color, linestyle="-",
                linewidth=2.2, zorder=5, label="Emulator, edited inputs (panel color)",
            )
        ax.set_xlabel("Year", fontsize=CHART_LABEL_SIZE)
        ax.set_ylabel(f"{target.replace('|', ' · ')}\n({output_unit(target)})", fontsize=CHART_LABEL_SIZE)
        ax.set_title(chr(97 + i), loc="left", fontsize=15, fontweight="bold", pad=12)
        _style_result_axis(ax)

    # Collect once across all panels, including a band absent from the first output.
    entries = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        entries.update(zip(labels, handles))
    if entries:
        if "AR6 scenarios, 5–95%" in entries:
            entries["AR6 scenarios, 5–95%"] = Patch(facecolor="#9ca3af", alpha=.2)
        if "Emulator, edited inputs (panel color)" in entries:
            entries["Emulator, edited inputs (panel color)"] = Line2D([], [], color="#444444", linewidth=2.2)
        _result_legend(fig, list(entries.values()), list(entries))
    fig.suptitle(f"{result.region} · Selected scenario emulation", fontsize=26, y=.985)
    fig.text(.5, .94, f"{result.model} / {result.scenario} ({result.category})",
             ha="center", fontsize=17, color="#555555")
    fig.tight_layout(rect=(.01, .13, .99, .94), h_pad=2.5, w_pad=2)

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
        ax.axvspan(
            history[0], history[-1], facecolor="none", edgecolor="#777777",
            hatch="////", linewidth=0, zorder=0, label="History (fixed)",
        )
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
