"""Manuscript figures regenerated from run artifacts.

Figure 3: IAM vs. emulated CO2 scatter, one panel per model
          (:func:`plot_co2_scatter`).
Figure 4: C1-C3 World CO2 trajectories (IAM solid, emulator dashed, error
          band) with an emulator-minus-IAM error row
          (:func:`plot_trajectories_by_category`).
Figure 6: Emissions|CO2 SHAP beeswarms, one panel per model, redrawn from
          the SHAP arrays each run saved (:func:`reconstruct_co2_shap`,
          :func:`plot_shap_co2_beeswarms`).

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
import tempfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.transforms import Bbox, blended_transform_factory

from configs.data import (
    CATEGORICAL_COLUMNS,
    INDEX_COLUMNS,
    NON_FEATURE_COLUMNS,
    REGION_CODE_TO_LABEL,
    SPLIT_SEED,
)
from configs.visualization import DEFAULT_REGION, SHAP_MAX_DISPLAY
from src.data.preprocess import encode_categorical_columns
from src.data.process_data import SCENARIO_CATEGORY_CSV, relabel_scenario_categories
from src.utils.run_store import RunStore
from src.visualization import shap_xgb
from src.visualization import trajectories as traj
from src.visualization.helpers import (
    backfill_group_labels,
    build_feature_display_names,
    filter_index_frame_by_region,
    sample_scenario_groups,
)
from src.visualization.shap_nn import _align_feature_names

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

# Figure 6 (SHAP beeswarm panels): one panel is drawn per model at
# SHAP_PANEL_FIGSIZE and the three are composed side by side, so the fonts
# are larger than the dashboard's SHAP_FONT_SIZE to survive the reduction.
SHAP_PANEL_FIGSIZE: Tuple[float, float] = (7.5, 5.5)
SHAP_TICK = 19            # x ticks, x label, colourbar ticks and label
SHAP_FEATURE_LABEL = 14   # feature names on the y axis (wrapped at the lag suffix)
SHAP_XTICK_BINS = 3
SHAP_POINT_SIZE = 8.0     # dashboard beeswarms use 5.0; larger to survive the 1x3 reduction

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


# --------------------------------------------------------------------------- #
# Figure 6: SHAP beeswarms for Emissions|CO2
# --------------------------------------------------------------------------- #
def _plots_dir(store: RunStore) -> str:
    return os.path.join(str(store.root), "plots")


def _require_frozen_split(store: RunStore) -> None:
    """Refuse to derive splits for a run that has not persisted them.

    ``derive_splits`` -> ``RunStore.splits_for`` / ``categories_for`` write
    splits.parquet / categories.json on first use.  Regenerating a figure
    must never write into a run directory, so only runs whose split and
    vocabularies are already on disk are reconstructed.
    """
    if not store.has_splits() or not store.has_categories():
        raise FileNotFoundError(
            f"{store.run_id}: splits.parquet / categories.json missing; the SHAP inputs cannot be "
            "reconstructed without deriving (and persisting) a split.  Re-run the preprocess phase."
        )


def _sampled_region_frame(frame: pd.DataFrame, region_series, log_prefix: str) -> pd.DataFrame:
    """Rows of *frame* the SHAP phase scored: region filter, then scenario-group sampling."""
    filtered, _, _, _, matched, _ = filter_index_frame_by_region(
        frame, DEFAULT_REGION, region_series=region_series, log_prefix=log_prefix,
    )
    if not matched:
        raise ValueError(f"{log_prefix}: region filter {DEFAULT_REGION!r} matched no rows")
    keys, _, _, group_cols = sample_scenario_groups(filtered, log_prefix=log_prefix)
    if group_cols and not keys.empty:
        return filtered.merge(keys, on=group_cols, how="inner")
    return filtered


def _reconstruct_xgb(run_id: str, target: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    from scripts.train_xgb import derive_splits

    store = RunStore(run_id)
    _require_frozen_split(store)
    data = store.load_processed_data()
    splits = derive_splits(data, store, n_lags=store.load_best_params().get("n_lags"))
    features, targets = list(splits["features"]), list(splits["targets"])
    test_data, _ = store.load_test_data()
    sampled = _sampled_region_frame(splits["X_test_with_index"], test_data["Region"], f"{run_id} SHAP region filter")
    X_test = sampled.drop(columns=NON_FEATURE_COLUMNS, errors="ignore").reset_index(drop=True)

    shap_values = np.load(os.path.join(_plots_dir(store), "shap_values.npy"))
    if shap_values.shape[:2] != (len(X_test), len(features)) or list(X_test.columns) != features:
        raise ValueError(
            f"{run_id}: saved SHAP values {shap_values.shape} do not match the reconstructed "
            f"test matrix {X_test.shape} / feature order"
        )
    # transform_outputs_to_former_inputs re-attributes lagged other-target
    # columns and, as a side effect, rewrites plots/csv under get_run_root.
    # Point it at a scratch directory so the run's own csv files stay untouched.
    original_root = shap_xgb.get_run_root
    with tempfile.TemporaryDirectory(prefix="paper_fig6_") as tmp:
        shap_xgb.get_run_root = lambda rid: os.path.join(tmp, rid)
        try:
            reattributed = shap_xgb.transform_outputs_to_former_inputs(run_id, shap_values, targets, features)
        finally:
            shap_xgb.get_run_root = original_root

    # Feature values only colour the dots; encode any label columns as the
    # SHAP plotting phase does (shap_xgb.draw_shap_plot).
    label_cols = [c for c in CATEGORICAL_COLUMNS if c in X_test.columns and not pd.api.types.is_numeric_dtype(X_test[c])]
    X_proc = encode_categorical_columns(X_test.copy(), label_cols, splits["categories"]) if label_cols else X_test
    X_values = X_proc.values.astype(np.float64)
    ti = targets.index(target)
    return reattributed[:, :, ti], X_values, build_feature_display_names(features)


def _reconstruct_lstm(run_id: str, target: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    from configs.models.lstm import LSTMTrainerConfig
    from scripts.train_lstm import _region_labels, derive_splits

    store = RunStore(run_id)
    _require_frozen_split(store)
    data = store.load_processed_data()
    splits = derive_splits(data, store)
    targets = list(splits["targets"])
    meta = store.load_train_meta() if store.has_train_meta() else {}
    features = meta.get("lstm_features", splits["features"])
    seq_len = int(meta.get("lstm_sequence_length", LSTMTrainerConfig().sequence_length))

    # The SHAP phase scores the horizon-level frame when the run saved one.
    horizon_df = store.load_predictions().get("horizon_df")
    frame = horizon_df if horizon_df is not None else splits["test_data"]
    sampled = _sampled_region_frame(frame, _region_labels(frame, splits["categories"]), f"{run_id} SHAP")
    series_cols = [c for c in INDEX_COLUMNS if c in sampled.columns]
    sort_cols = series_cols + (["Year"] if "Year" in sampled.columns else [])
    sampled = sampled.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    group_ids = sampled.groupby(series_cols, sort=False).ngroup().to_numpy()
    X_test = sampled.drop(columns=NON_FEATURE_COLUMNS, errors="ignore").reset_index(drop=True)

    # Mirrors the preprocess_features / create_sequences closures inside
    # shap_nn.get_lstm_shap_values (not importable): scale the continuous
    # features of the first 100 rows and cut windows that stay in one series.
    raw_features, _ = store.load_features()
    raw_features = meta.get("lstm_raw_features", raw_features)
    categorical = meta.get("lstm_categorical_features", [])
    continuous = [f for f in raw_features if f not in categorical]
    scaler_X = store.load_artifact("lstm_scaler_X.pkl")
    n = min(100, len(X_test))
    scaled = scaler_X.transform(X_test.iloc[:n][continuous].fillna(-1.0).astype(np.float32))
    g = group_ids[:n]
    windows = [scaled[i:i + seq_len] for i in range(len(scaled) - seq_len + 1) if g[i] == g[i + seq_len - 1]]
    test_inputs = np.stack(windows)

    temporal = np.load(os.path.join(_plots_dir(store), "lstm_shap_values_temporal.npy"))
    if temporal.shape[0] != test_inputs.shape[0] or temporal.shape[2] != test_inputs.shape[2]:
        raise ValueError(
            f"{run_id}: temporal SHAP {temporal.shape} does not match the reconstructed windows {test_inputs.shape}"
        )
    names = build_feature_display_names(_align_feature_names(list(features), temporal.shape[2]))
    ti = targets.index(target)
    return temporal.sum(axis=1)[:, :, ti], test_inputs.mean(axis=1), names


def _reconstruct_tft(run_id: str, target: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    store = RunStore(run_id)
    _, targets = store.load_features()
    cache = np.load(os.path.join(_plots_dir(store), "tft_shap_cache.npz"), allow_pickle=True)
    temporal, test_inputs = cache["temporal_shap"], cache["test_inputs"]
    names = build_feature_display_names(_align_feature_names(cache["feature_names"].tolist(), temporal.shape[2]))
    ti = list(targets).index(target)
    return temporal.sum(axis=1)[:, :, ti], test_inputs.mean(axis=1), names


_RECONSTRUCT = {"xgb": _reconstruct_xgb, "lstm": _reconstruct_lstm, "tft": _reconstruct_tft}


def reconstruct_co2_shap(
    run_id: str,
    model_type: Optional[str] = None,
    target: str = "Emissions|CO2",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """(shap_matrix, X_matrix, display_names) behind *run_id*'s *target* beeswarm.

    The SHAP phase saves the attributions but not the rows it scored, so the
    feature matrix that colours the dots is rebuilt the way that phase built it:

    * ``xgb``: ``scripts.train_xgb.derive_splits`` -> R10 region filter ->
      scenario-group sampling; saved ``plots/shap_values.npy`` is then passed
      through ``shap_xgb.transform_outputs_to_former_inputs`` (csv side
      effects redirected to a temporary directory).
    * ``lstm``: ``predictions.pkl`` horizon frame -> region filter -> sampling
      -> ``lstm_scaler_X`` -> sequence windows of the first 100 rows;
      ``plots/lstm_shap_values_temporal.npy`` summed over timesteps, inputs
      averaged over timesteps.
    * ``tft``: everything comes from ``plots/tft_shap_cache.npz``.

    *model_type* defaults to the run id prefix.  Both matrices are
    (n_samples, n_features); the names are the dashboard's display names.
    """
    model_type = (model_type or run_id.split("_", 1)[0]).lower()
    if model_type not in _RECONSTRUCT:
        raise ValueError(f"model_type must be one of {sorted(_RECONSTRUCT)}, got {model_type!r}")
    shap_matrix, X_matrix, names = _RECONSTRUCT[model_type](run_id, target)
    shap_matrix, X_matrix = np.asarray(shap_matrix), np.asarray(X_matrix)
    if shap_matrix.shape != X_matrix.shape or len(names) != shap_matrix.shape[1]:
        raise ValueError(f"{run_id}: SHAP {shap_matrix.shape}, X {X_matrix.shape}, {len(names)} names")
    logger.info("%s (%s): reconstructed %s SHAP matrix %s", run_id, model_type, target, shap_matrix.shape)
    return shap_matrix, X_matrix, names


def top_shap_features(shap_matrix: np.ndarray, names: Sequence[str], k: int = SHAP_MAX_DISPLAY) -> List[str]:
    """Feature names ranked by mean |SHAP|, as the beeswarm orders them."""
    order = np.argsort(np.abs(np.asarray(shap_matrix)).mean(axis=0))[::-1][:k]
    return [names[i] for i in order]


def _wrap_feature_label(name: str) -> str:
    """Break 'GDP|PPP (current)' into two lines before the '(...)' suffix."""
    i = name.find(" (")
    return name[:i] + "\n" + name[i + 1:] if i > 0 else name


def draw_shap_co2_panel(
    shap_matrix: np.ndarray,
    X_matrix: np.ndarray,
    names: Sequence[str],
    out_path: str,
    figsize: Tuple[float, float] = SHAP_PANEL_FIGSIZE,
    tick_size: float = SHAP_TICK,
    feature_size: float = SHAP_FEATURE_LABEL,
    xtick_bins: int = SHAP_XTICK_BINS,
    point_size: float = SHAP_POINT_SIZE,
    dpi: int = 300,
) -> None:
    """One large-font beeswarm panel (no title), saved tightly cropped to *out_path*.

    Same shap.summary_plot call and jitter seed as
    ``helpers.draw_shap_beeswarm``; the fonts, the two-line feature labels
    (variable name bold, lag suffix regular), the dot size and the 3-bin x
    axis are the manuscript's.
    """
    import shap

    with plt.rc_context({"font.size": tick_size}):
        fig = plt.figure(figsize=figsize)
        shap.summary_plot(
            np.asarray(shap_matrix), np.asarray(X_matrix),
            feature_names=[_wrap_feature_label(n) for n in names],
            max_display=SHAP_MAX_DISPLAY, plot_type="dot", show=False,
            rng=np.random.default_rng(SPLIT_SEED), plot_size=figsize,
        )
        ax = plt.gca()
        for coll in ax.collections:
            sizes = coll.get_sizes()
            coll.set_sizes(np.full_like(sizes, point_size, dtype=float) if sizes is not None and len(sizes) else [point_size])
        # Two-line feature labels with only the variable name in bold.  A tick
        # label cannot mix weights, so blank the ticks and draw two texts per
        # row: x in axes coordinates just left of the axis, y at the tick.
        labels = [t.get_text() for t in ax.get_yticklabels()]
        ax.set_yticklabels([""] * len(labels))
        ax.tick_params(axis="y", length=0)
        row_tr = blended_transform_factory(ax.transAxes, ax.transData)
        for y, text in zip(ax.get_yticks(), labels):
            parts = text.split("\n", 1)
            if len(parts) == 2:
                ax.text(-0.01, y, parts[0], transform=row_tr, ha="right", va="bottom", fontsize=feature_size, fontweight="bold")
                ax.text(-0.01, y, parts[1], transform=row_tr, ha="right", va="top", fontsize=feature_size)
            else:
                ax.text(-0.01, y, text, transform=row_tr, ha="right", va="center", fontsize=feature_size, fontweight="bold")
        ax.tick_params(axis="x", labelsize=tick_size)
        ax.set_xlabel(ax.get_xlabel(), fontsize=tick_size)
        if xtick_bins:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=xtick_bins, steps=[1, 2, 5, 10]))
        for other in fig.axes:
            if other is not ax:  # colourbar
                other.tick_params(labelsize=tick_size, length=0)
                other.set_ylabel(other.get_ylabel(), fontsize=tick_size, labelpad=0)
        # Crop to the artists: shap's own layout leaves a wide margin, and the
        # rotated colourbar label is missed by the plain tight bbox.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.dpi_scale_trans.inverted()
        texts = [t for a in fig.axes for t in (a.xaxis.label, a.yaxis.label, *a.get_xticklabels(), *a.get_yticklabels(), *a.texts)]
        bbox = Bbox.union([fig.get_tightbbox(renderer)] + [t.get_window_extent(renderer).transformed(inv) for t in texts])
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches=bbox.padded(0.05))
        plt.close(fig)


def _compose_row(panel_paths: Sequence[str], labels: Sequence[str], out_path: str, dpi: int = 300) -> None:
    """Paste the panels side by side at equal height with a bold serif label under each."""
    from PIL import Image, ImageDraw, ImageFont

    images = [Image.open(p).convert("RGB") for p in panel_paths]
    height = min(im.height for im in images)
    images = [im.resize((round(im.width * height / im.height), height), Image.LANCZOS) for im in images]
    font_px = int(0.04 * images[0].width)  # label height ~4% of a panel's width
    font = ImageFont.truetype(font_manager.findfont(times_bold()), font_px)
    band, gap = int(font_px * 1.6), int(0.02 * images[0].width)
    canvas = Image.new("RGB", (sum(im.width for im in images) + gap * (len(images) - 1), height + band), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for im, text in zip(images, labels):
        canvas.paste(im, (x, 0))
        w = draw.textlength(text, font=font)
        draw.text((x + (im.width - w) / 2, height + font_px * 0.2), text, fill="black", font=font)
        x += im.width + gap
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    canvas.save(out_path, dpi=(dpi, dpi))


def plot_shap_co2_beeswarms(
    run_ids: Sequence[str] = ("xgb_85", "lstm_89", "tft_95"),
    names: Sequence[str] = ("XGBoost", "LSTM", "TFT"),
    target: str = "Emissions|CO2",
    out_path: Optional[str] = None,
    panel_dir: Optional[str] = None,
    dpi: int = 300,
) -> Dict[str, List[str]]:
    """Figure 6: *target* SHAP beeswarms, one large-font panel per run, in one row.

    Each panel is reconstructed with :func:`reconstruct_co2_shap`, drawn with
    :func:`draw_shap_co2_panel` and labelled '(a) XGBoost' etc. underneath in
    the manuscript's bold serif.  Per-panel PNGs go to *panel_dir* (a
    temporary directory when None).  Returns ``{name: top-8 feature names}``
    in beeswarm order so the ranking can be checked against the run's own
    SHAP plots.
    """
    if len(run_ids) != len(names):
        raise ValueError("run_ids and names must have the same length")
    top: Dict[str, List[str]] = {}
    with tempfile.TemporaryDirectory(prefix="paper_fig6_panels_") as tmp:
        panel_dir = panel_dir or tmp
        os.makedirs(panel_dir, exist_ok=True)
        panel_paths = []
        for run_id, name in zip(run_ids, names):
            shap_matrix, X_matrix, display = reconstruct_co2_shap(run_id, target=target)
            top[name] = top_shap_features(shap_matrix, display)
            logger.info("%s top-%d: %s", name, len(top[name]), top[name])
            path = os.path.join(panel_dir, f"{run_id}_{target.replace('|', '_')}_beeswarm.png")
            draw_shap_co2_panel(shap_matrix, X_matrix, display, path, dpi=dpi)
            panel_paths.append(path)
        if out_path:
            _compose_row(panel_paths, [f"({chr(97 + i)}) {n}" for i, n in enumerate(names)], out_path, dpi=dpi)
            logger.info("saved %s", out_path)
    return top
