"""Manuscript figures regenerated from run artifacts.

Figure 3: IAM vs. emulated CO2 scatter, one panel per model
          (:func:`plot_co2_scatter`).
Figure 4: C1-C3 World CO2 trajectories (IAM solid, emulator dashed, error
          band) with an emulator-minus-IAM error row
          (:func:`plot_trajectories_by_category`).

Data loading (:func:`load_run_predictions`) follows the dashboard exactly:
RunStore test data + features + predictions, categorical codes decoded,
horizon-level frames preferred when present, group labels backfilled and
scenario categories relabelled.  Only the run ids change between models, so
the same code path serves XGBoost, LSTM and TFT runs.

Style constants mirror the manuscript's ``figstyle.py``: Times New Roman
bold panel labels (with a fallback when the font is not installed), a
``k`` thousands formatter and shared font sizes.
"""
import logging
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from configs.data import REGION_CODE_TO_LABEL
from src.data.process_data import SCENARIO_CATEGORY_CSV, relabel_scenario_categories
from src.utils.run_store import RunStore
from src.visualization import trajectories as traj
from src.visualization.helpers import backfill_group_labels

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Style (equivalent of nas_plots/figstyle.py)
# --------------------------------------------------------------------------- #
AXIS_LABEL = 13
TICK = 12
ANNOT = 14
LEGEND = 12
PANEL_LABEL = 17
KFMT = FuncFormatter(lambda x, p: f"{x/1000:.0f}k" if abs(x) >= 1000 else f"{x:.0f}")
RUNS: List[Tuple[str, str]] = [("xgb_85", "XGBoost"), ("lstm_89", "LSTM"), ("tft_95", "TFT")]
META_COLUMNS = ["Model", "Scenario", "Region", "Model_Family", "Scenario_Category", "Year"]

# Category palettes for Figure 4; the manuscript uses 'warm'.
CAT_PALETTES: Dict[str, Dict[str, str]] = {
    "warm": {"C1": "#2A9D8F", "C2": "#E08E2B", "C3": "#6D5BA6"},
    "ar6": {"C1": "#3E9DC2", "C2": "#5E6E45", "C3": "#4F5A80"},
}

TIMES_BOLD_CANDIDATES = [
    "C:/Windows/Fonts/timesbd.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
]


def times_bold(size: int = PANEL_LABEL, candidates: Sequence[str] = TIMES_BOLD_CANDIDATES) -> font_manager.FontProperties:
    """FontProperties for the bold serif panel labels.

    Tries the Times New Roman / Liberation Serif bold files in order and
    falls back to matplotlib's default serif bold (DejaVu Serif on a bare
    server) with a warning when none exists.
    """
    for path in candidates:
        if os.path.exists(path):
            return font_manager.FontProperties(fname=path, size=size)
    logger.warning(
        "No Times New Roman / Liberation Serif bold font found (tried %s); "
        "panel labels fall back to matplotlib's default serif bold.",
        ", ".join(candidates),
    )
    return font_manager.FontProperties(family="serif", weight="bold", size=size)


def apply_style() -> None:
    matplotlib.rcParams.update({
        "font.size": TICK,
        "axes.labelsize": AXIS_LABEL,
        "xtick.labelsize": TICK,
        "ytick.labelsize": TICK,
        "legend.fontsize": LEGEND,
    })


def panel_label(ax, idx: int, name: str, y: float = -0.30, font: Optional[font_manager.FontProperties] = None) -> None:
    """'(a) XGBoost'-style label; y < 0 puts it below the panel."""
    ax.set_title(f"({chr(97 + idx)}) {name}", fontproperties=font or times_bold(), y=y)


# --------------------------------------------------------------------------- #
# Data loading (equivalent of nas_plots/dump_traj.py)
# --------------------------------------------------------------------------- #
def _code_maps(store: RunStore) -> Dict[str, Dict[int, str]]:
    maps: Dict[str, Dict[int, str]] = {}
    if store.has_categories():
        for column, labels in store.load_categories().items():
            maps[column] = {i: label for i, label in enumerate(labels)}
    if "Model_Family" not in maps and store.has_train_meta():
        legacy = store.load_train_meta().get("lstm_model_family_categories")
        if legacy:
            maps["Model_Family"] = {i: label for i, label in enumerate(legacy)}
    if "Region" not in maps and REGION_CODE_TO_LABEL:
        maps["Region"] = dict(REGION_CODE_TO_LABEL)
    return maps


def _decode(df: Optional[pd.DataFrame], maps: Dict[str, Dict[int, str]]) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    for column, code_map in maps.items():
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].map(code_map)
    return df


def load_run_predictions(run_id: str) -> pd.DataFrame:
    """Test rows of *run_id* with IAM values and emulator predictions.

    Returns one row per (Model, Scenario, Region, Year) of the test split
    with the metadata columns present in the run (Model, Scenario, Region,
    Model_Family, Scenario_Category, Year) plus ``y|<target>`` (IAM) and
    ``p|<target>`` (emulated) for every target the run was trained on.
    Horizon-level frames (``horizon_df`` / ``horizon_y_true``) are used when
    the run saved them, as the dashboard does.
    """
    store = RunStore(run_id)
    test_data, y_test = store.load_test_data()
    _, targets = store.load_features()
    bundle = store.load_predictions()
    preds = bundle["preds"]
    horizon_df = bundle.get("horizon_df")
    horizon_y = bundle.get("horizon_y_true")

    maps = _code_maps(store)
    test_data = _decode(test_data, maps)
    horizon_df = _decode(horizon_df, maps)
    if horizon_df is not None:
        horizon_df, _ = backfill_group_labels(horizon_df, test_data, ["Scenario_Category", "Model_Family"])

    scen = pd.read_csv(SCENARIO_CATEGORY_CSV, dtype=str)
    test_data, _ = relabel_scenario_categories(test_data, scen)
    if horizon_df is not None and {"Model", "Scenario", "Scenario_Category"} <= set(horizon_df.columns):
        horizon_df, _ = relabel_scenario_categories(horizon_df, scen)

    if horizon_df is not None and horizon_y is not None:
        data, y = horizon_df, horizon_y
    else:
        data, y = test_data, y_test

    yy = np.asarray(y)
    pp = np.asarray(preds)
    if yy.ndim == 1:
        yy = yy[:, None]
    if pp.ndim == 1:
        pp = pp[:, None]
    if len(yy) != len(data) or len(pp) != len(data):
        raise ValueError(
            f"{run_id}: {len(data)} rows but y has {len(yy)} and preds {len(pp)} rows"
        )

    out = data[[c for c in META_COLUMNS if c in data.columns]].reset_index(drop=True)
    for i, t in enumerate(targets):
        out[f"y|{t}"] = yy[:, i]
        out[f"p|{t}"] = pp[:, i]
    logger.info("%s: %d rows, %d scenarios, targets=%s", run_id, len(out),
                out.groupby(["Model", "Scenario"]).ngroups if {"Model", "Scenario"} <= set(out.columns) else -1,
                targets)
    return out


def r_squared(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


# --------------------------------------------------------------------------- #
# Figure 3
# --------------------------------------------------------------------------- #
def plot_co2_scatter(
    run_ids: Sequence[str] = ("xgb_85", "lstm_89", "tft_95"),
    names: Sequence[str] = ("XGBoost", "LSTM", "TFT"),
    target: str = "Emissions|CO2",
    out_path: Optional[str] = None,
    limits: Tuple[float, float] = (-20000, 105000),
    dpi: int = 200,
    data: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, float]:
    """Figure 3: IAM vs. emulated *target*, one panel per run, coloured by year.

    Panels share both axes, show a dotted 1:1 line, an R^2 box and an
    '(a) XGBoost'-style label below; one discrete year legend (the years
    present with non-NaN values, i.e. the 5-year AR6 steps) sits to the
    right.  Returns ``{name: R^2}``; *data* lets a caller pass frames from
    :func:`load_run_predictions` to avoid reloading them.
    """
    if len(run_ids) != len(names):
        raise ValueError("run_ids and names must have the same length")
    apply_style()
    font = times_bold()
    ycol, pcol = f"y|{target}", f"p|{target}"
    frames = {}
    for r in run_ids:
        d = data[r] if data and r in data else load_run_predictions(r)
        if ycol not in d.columns:
            raise KeyError(f"{r} has no target {target!r}")
        frames[r] = d.dropna(subset=[ycol, pcol])
    years = sorted(set().union(*[set(d["Year"].round().astype(int)) for d in frames.values()]))
    logger.info("years in data: %s", years)
    cmap = plt.get_cmap("viridis", len(years))
    year_colour = {yr: cmap(i) for i, yr in enumerate(years)}

    fig, axes = plt.subplots(1, len(run_ids), figsize=(5 * len(run_ids), 5.0), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    r2s: Dict[str, float] = {}
    lo, hi = limits
    for i, (ax, r, name) in enumerate(zip(axes, run_ids, names)):
        d = frames[r]
        y, p = d[ycol].values, d[pcol].values
        r2 = r_squared(y, p)
        r2s[name] = r2
        yrs = d["Year"].round().astype(int).values
        for yr in years:
            m = yrs == yr
            if m.any():
                ax.scatter(y[m], p[m], color=year_colour[yr], s=18, alpha=0.55, linewidths=0)
        ax.plot([lo, hi], [lo, hi], color="0.3", lw=0.8, ls=":", zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.text(0.05, 0.94, f"$R^2$ = {r2:.3f}", transform=ax.transAxes, va="top", fontsize=ANNOT,
                bbox=dict(boxstyle="round", fc="white", ec="0.3", lw=0.8))
        panel_label(ax, i, name, y=-0.30, font=font)
        ax.set_xlabel(f"IAM {target} (Mt CO2/yr)")
        ax.xaxis.set_major_formatter(KFMT)
        ax.yaxis.set_major_formatter(KFMT)
        ax.grid(alpha=0.2)
        logger.info("%s R2=%.4f n=%d years=%s..%s", name, r2, len(d), yrs.min(), yrs.max())
    axes[0].set_ylabel(f"Emulated {target} (Mt CO2/yr)")
    handles = [Line2D([], [], marker="o", ls="", color=year_colour[yr], alpha=0.8, markersize=6, label=str(yr))
               for yr in years]
    fig.legend(handles=handles, title="Year", loc="center left", bbox_to_anchor=(0.905, 0.5), frameon=True,
               fontsize=LEGEND - 1, title_fontsize=LEGEND, handletextpad=0.4, borderpad=0.6)
    fig.subplots_adjust(right=0.9, wspace=0.12)
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        logger.info("saved %s", out_path)
    plt.close(fig)
    return r2s


# --------------------------------------------------------------------------- #
# Figure 4
# --------------------------------------------------------------------------- #
def plot_trajectories_by_category(
    run_ids: Sequence[str] = ("xgb_85", "lstm_89", "tft_95"),
    names: Sequence[str] = ("XGBoost", "LSTM", "TFT"),
    target: str = "Emissions|CO2",
    categories: Sequence[str] = ("C1", "C2", "C3"),
    region: str = "World",
    palette: str = "warm",
    out_path: Optional[str] = None,
    x_start: int = 2015,
    x_end: int = 2100,
    dpi: int = 200,
    data: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, int]:
    """Figure 4: per-scenario *target* trajectories for *categories* in *region*.

    Two rows x one column per run (height ratios 2.3:1).  Top: IAM (solid),
    emulator (dashed) and the error band between them for every test
    scenario, coloured by AR6 category and drawn C3 -> C2 -> C1 so the
    smaller categories stay visible, plus the SSP1-2.6 marker (IMAGE 3.0.1,
    World) as a thick black line.  Bottom: emulator minus IAM with a zero
    line.  Rows share y limits; the legend sits above the panels.
    Returns ``{name: number of scenarios drawn}``.
    """
    if len(run_ids) != len(names):
        raise ValueError("run_ids and names must have the same length")
    colours = CAT_PALETTES[palette]
    missing = [c for c in categories if c not in colours]
    if missing:
        raise KeyError(f"palette {palette!r} has no colour for {missing}")
    apply_style()
    font = times_bold()
    ycol, pcol = f"y|{target}", f"p|{target}"

    frames = {}
    for r in run_ids:
        d = data[r] if data and r in data else load_run_predictions(r)
        mask = d["Scenario_Category"].isin(categories) & (d["Region"] == region) & (d["Year"] >= x_start)
        frames[r] = d.loc[mask].dropna(subset=[ycol, pcol])

    marker = traj.load_marker_scenario([target])
    if marker is not None and target in marker.columns:
        marker = marker[marker["Year"] >= x_start]
    else:
        logger.warning("Marker scenario unavailable for %s; drawing without it.", target)
        marker = None

    n = len(run_ids)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8.2), sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [2.3, 1], "hspace": 0.10, "wspace": 0.08})
    counts: Dict[str, int] = {}
    draw_order = [c for c in ("C3", "C2", "C1") if c in categories] + [c for c in categories if c not in ("C1", "C2", "C3")]
    for i, (r, name) in enumerate(zip(run_ids, names)):
        top, bot = axes[0, i], axes[1, i]
        d = frames[r]
        counts[name] = 0
        for cat in draw_order:
            c = colours[cat]
            for _, g in d[d["Scenario_Category"] == cat].groupby(["Model", "Scenario"]):
                g = g.sort_values("Year")
                yv, pv = g[ycol].values, g[pcol].values
                top.plot(g["Year"], yv, color=c, lw=1.0, alpha=0.9)
                top.plot(g["Year"], pv, color=c, lw=1.0, alpha=0.9, ls=(0, (3, 2)))
                top.fill_between(g["Year"], yv, pv, color=c, alpha=0.10, lw=0)
                bot.plot(g["Year"], pv - yv, color=c, lw=1.0, alpha=0.85)
                counts[name] += 1
        if marker is not None:
            top.plot(marker["Year"], marker[target], color="black", lw=2.6, zorder=6)
        bot.axhline(0, color="0.15", lw=1.0, zorder=5)
        panel_label(top, i, name, y=1.02, font=font)
        for ax in (top, bot):
            ax.yaxis.set_major_formatter(KFMT)
            ax.set_xlim(x_start, x_end)
            ax.grid(axis="y", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
        bot.set_xlabel("Year")
        logger.info("%s: %d scenarios in %s/%s", name, counts[name], list(categories), region)
    for row in axes:
        lo = min(a.get_ylim()[0] for a in row)
        hi = max(a.get_ylim()[1] for a in row)
        for a in row:
            a.set_ylim(lo, hi)
        for a in row[1:]:
            a.tick_params(labelleft=False)
    axes[0, 0].set_ylabel(f"{target} (Mt CO2/yr)")
    axes[1, 0].set_ylabel("Emulator $-$ IAM (Mt CO2/yr)")
    handles = [Line2D([], [], color=colours[c], lw=2.5, label=c) for c in categories] + [
        Line2D([], [], color="0.3", lw=1.5, label="IAM"),
        Line2D([], [], color="0.3", lw=1.5, ls=(0, (3, 2)), label="Emulator"),
        Patch(color="0.55", alpha=0.35, label="Error band"),
    ]
    if marker is not None:
        handles.append(Line2D([], [], color="black", lw=2.6, label="SSP1-2.6 marker (IMAGE 3.0.1)"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.995),
               fontsize=15, handlelength=2.2, columnspacing=1.6)
    fig.subplots_adjust(top=0.87, bottom=0.09, left=0.07, right=0.99)
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        logger.info("saved %s", out_path)
    plt.close(fig)
    return counts
