"""What-if emulation: the data side of the dashboard's live-inference view.

A visitor picks a region the run emulates well, starts from a real AR6
trajectory's inputs, moves the input levers within the range the training
scenarios span, and runs the trained model.  Everything here is plain
pandas: the frames a run was fitted on, which regions and trajectories are
offered, the lever bounds, the edited input paths, and the result bundle.
Nothing imports torch, so the trajectories view and the CPU-only XGBoost
installation keep working without the deep-learning stack.
"""

import datetime
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from configs.dashboard import (
    WHATIF_ANCHOR_YEARS,
    WHATIF_BAND_QUANTILES,
    WHATIF_BAND_YEAR_STEP,
    WHATIF_BASELINE_CATEGORY,
    WHATIF_MIN_BAND_COUNT,
    WHATIF_MIN_BAND_FRACTION,
    WHATIF_PREFERRED_BASELINE,
)
from configs.data import CATEGORICAL_COLUMNS, INDEX_COLUMNS, MAX_SERIES_LENGTH
from src.data.preprocess import prepare_sequence_frames
from src.data.process_data import SCENARIO_CATEGORY_CSV, relabel_scenario_categories
from src.utils.run_store import RunStore

GROUP_KEYS = list(INDEX_COLUMNS)
SPLIT_COLUMN = "split"
R2_COLUMN = "R2 Score (pooled)"
SAMPLE_COLUMN = "Sample Size"


# ── The run's frames ──────────────────────────────────────────────────────


@dataclass
class PreparedRun:
    """Every trajectory a run was fitted and scored on, as the model saw it."""

    run_id: str
    frame: pd.DataFrame
    features: List[str]
    raw_features: List[str]
    targets: List[str]


def raw_feature_names(features: Sequence[str]) -> List[str]:
    """The numeric inputs a visitor may move: no indicators, no categoricals."""
    return [
        f for f in features
        if not f.endswith("_is_missing") and f not in CATEGORICAL_COLUMNS
    ]


def load_prepared_run(store: RunStore) -> PreparedRun:
    """Rebuild the run's train/val/test frames from its cached data and split.

    The chain is the one the TFT phases run, so a trajectory's rows here are
    the rows the test phase predicted from.  Refuses a run without a saved
    split rather than deriving one: probing must never write into a run.
    """
    if not store.has_splits():
        raise FileNotFoundError(
            f"Run {store.run_id} has no saved split assignment (artifacts/splits.parquet); "
            "re-run its preprocess phase."
        )
    data = store.load_processed_data()
    frames = prepare_sequence_frames(data, store.load_splits())

    parts = []
    for name in ("train", "val", "test"):
        part = getattr(frames, name).copy()
        part[SPLIT_COLUMN] = name
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)

    frame, relabelled = relabel_scenario_categories(
        frame, pd.read_csv(SCENARIO_CATEGORY_CSV, dtype=str)
    )
    if relabelled:
        logging.info("%s: %d runs relabelled from %s", store.run_id, relabelled, SCENARIO_CATEGORY_CSV.name)
    frame["Year"] = frame["Year"].astype(int)
    frame = frame.sort_values(GROUP_KEYS + ["Step"]).reset_index(drop=True)

    return PreparedRun(
        run_id=store.run_id,
        frame=frame,
        features=list(frames.features),
        raw_features=raw_feature_names(frames.features),
        targets=list(frames.targets),
    )


# ── Regions ───────────────────────────────────────────────────────────────


def eligible_regions(metrics: pd.DataFrame, r2_threshold: float, min_samples: float) -> pd.DataFrame:
    """Rows of the per-region table that clear the accuracy gate, in file order."""
    required = {"Region", R2_COLUMN, SAMPLE_COLUMN}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Per-region metrics lack column(s) {sorted(missing)}")
    keep = (metrics[R2_COLUMN] >= r2_threshold) & (metrics[SAMPLE_COLUMN] >= min_samples)
    columns = [c for c in ("Region", "Region Type", R2_COLUMN, SAMPLE_COLUMN) if c in metrics.columns]
    return metrics.loc[keep, columns].reset_index(drop=True)


def region_metrics_from_predictions(run_id, horizon_df, horizon_y_true, preds, targets) -> Optional[pd.DataFrame]:
    """The per-region table computed from a loaded prediction bundle.

    For runs tested before the table was written beside performance.csv.
    """
    from src.data.preprocess import observed_mask_from_frame
    from src.trainers.evaluation import metrics_by_region

    return metrics_by_region(
        run_id,
        np.asarray(horizon_y_true, dtype=float),
        np.asarray(preds, dtype=float),
        horizon_df,
        observed_mask_from_frame(horizon_df, list(targets)),
    )


# ── Baseline trajectories ─────────────────────────────────────────────────


@dataclass(frozen=True)
class BaselineCandidate:
    """A trajectory the levers can start from."""

    model: str
    scenario: str
    split: str
    n_steps: int
    first_year: int
    last_year: int
    n_reported: int
    n_regions: int

    @property
    def key(self) -> Tuple[str, str]:
        return (self.model, self.scenario)

    @property
    def label(self) -> str:
        return f"{self.model} / {self.scenario}"


def _reported_everywhere(rows: pd.DataFrame, feature: str) -> bool:
    """True when the IAM reported *feature* at every step (nothing imputed)."""
    indicator = f"{feature}_is_missing"
    if indicator in rows.columns:
        return bool((pd.to_numeric(rows[indicator], errors="coerce").fillna(1) == 0).all())
    return bool(pd.to_numeric(rows[feature], errors="coerce").notna().all())


def candidate_baselines(
    prepared: PreparedRun,
    region: str,
    *,
    min_steps: int = MAX_SERIES_LENGTH,
    category: str = WHATIF_BASELINE_CATEGORY,
) -> List[BaselineCandidate]:
    """Trajectories of *category* in *region* long enough for the model's window.

    Ordered by how many inputs the IAM reported (most first), then by how
    many regions the run covers, then by name.
    """
    frame = prepared.frame
    in_category = frame[frame["Scenario_Category"] == category]
    if in_category.empty:
        return []
    coverage = in_category.groupby(["Model", "Scenario"])["Region"].nunique()

    candidates = []
    for (model, scenario), rows in in_category[in_category["Region"] == region].groupby(
        ["Model", "Scenario"], sort=True
    ):
        if len(rows) < min_steps:
            continue
        n_reported = sum(
            _reported_everywhere(rows, feature)
            for feature in prepared.raw_features
            if feature in rows.columns
        )
        years = rows["Year"].astype(int)
        candidates.append(
            BaselineCandidate(
                model=str(model),
                scenario=str(scenario),
                split=str(rows[SPLIT_COLUMN].iloc[0]) if SPLIT_COLUMN in rows.columns else "",
                n_steps=int(len(rows)),
                first_year=int(years.min()),
                last_year=int(years.max()),
                n_reported=int(n_reported),
                n_regions=int(coverage.loc[(model, scenario)]),
            )
        )
    candidates.sort(key=lambda c: (-c.n_reported, -c.n_regions, c.model, c.scenario))
    return candidates


def choose_default_baseline(
    candidates: Sequence[BaselineCandidate],
    preferred: Tuple[str, str] = WHATIF_PREFERRED_BASELINE,
) -> Optional[BaselineCandidate]:
    """The preferred (Model, Scenario) where offered, else the best candidate."""
    for candidate in candidates:
        if candidate.key == tuple(preferred):
            return candidate
    return candidates[0] if candidates else None


def baseline_rows(prepared: PreparedRun, region: str, model: str, scenario: str) -> pd.DataFrame:
    """One trajectory's rows, in step order."""
    frame = prepared.frame
    mask = (frame["Model"] == model) & (frame["Scenario"] == scenario) & (frame["Region"] == region)
    rows = frame.loc[mask].sort_values("Step").reset_index(drop=True)
    if rows.empty:
        raise KeyError(f"No rows for {model} / {scenario} in {region}")
    return rows


# ── Bands: where the training scenarios lie ───────────────────────────────


def _quantile_band(
    values_by_year: pd.DataFrame,
    value_column: str,
    quantiles: Tuple[float, float],
    min_count: int,
    year_step: int,
    min_fraction: float,
) -> List[Tuple[int, float, float, int]]:
    """(year, lo, hi, n) on the *year_step* grid.

    Bounds are NaN where fewer than *min_count* values, or fewer than
    *min_fraction* of the best-covered grid year's count, were reported.
    """
    lo_q, hi_q = quantiles
    years = pd.to_numeric(values_by_year["Year"], errors="coerce")
    on_grid = values_by_year[(years % year_step == 0).to_numpy()]
    groups = {
        int(year): pd.to_numeric(values, errors="coerce").dropna()
        for year, values in on_grid.groupby("Year")[value_column]
    }
    largest = max((len(v) for v in groups.values()), default=0)
    out = []
    for year in sorted(groups):
        values = groups[year]
        n = int(len(values))
        if n >= min_count and n >= min_fraction * largest:
            lo, hi = float(values.quantile(lo_q)), float(values.quantile(hi_q))
        else:
            lo = hi = math.nan
        out.append((year, lo, hi, n))
    return out


def lever_bands(
    rows: pd.DataFrame,
    raw_features: Sequence[str],
    *,
    quantiles: Tuple[float, float] = WHATIF_BAND_QUANTILES,
    min_count: int = WHATIF_MIN_BAND_COUNT,
    year_step: int = WHATIF_BAND_YEAR_STEP,
    min_fraction: float = WHATIF_MIN_BAND_FRACTION,
) -> pd.DataFrame:
    """Per feature and grid year, the quantile range of the values the IAMs reported.

    Imputed values are left out: a band drawn from medians would say nothing
    about where scenarios actually lie.  Only years on the *year_step* grid
    are evaluated (every scenario reports the decades; the years in between
    come from a different subset), and :func:`band_at` interpolates.  A grid
    year with fewer than *min_count* reported values, or fewer than
    *min_fraction* of the best-covered year's, gets NaN bounds.  Columns:
    feature, Year, lo, hi, n.
    """
    records = []
    for feature in raw_features:
        if feature not in rows.columns:
            continue
        indicator = f"{feature}_is_missing"
        reported = rows if indicator not in rows.columns else rows[rows[indicator] == 0]
        band = _quantile_band(reported[["Year", feature]], feature, quantiles, min_count, year_step, min_fraction)
        records.extend((feature, year, lo, hi, n) for year, lo, hi, n in band)
    return pd.DataFrame(records, columns=["feature", "Year", "lo", "hi", "n"])


def ar6_target_bands(
    rows: pd.DataFrame,
    targets: Sequence[str],
    quantiles: Tuple[float, float] = WHATIF_BAND_QUANTILES,
    min_count: int = WHATIF_MIN_BAND_COUNT,
    year_step: int = WHATIF_BAND_YEAR_STEP,
    min_fraction: float = WHATIF_MIN_BAND_FRACTION,
) -> pd.DataFrame:
    """Per target and grid year, the quantile range of the observed IAM values.

    Columns: target, Year, lo, hi, n.  Interpolated and zero-filled target
    values (``<target>__observed == 0``) are left out, and the grid and
    coverage rules are those of :func:`lever_bands`.
    """
    records = []
    for target in targets:
        if target not in rows.columns:
            continue
        observed_column = f"{target}__observed"
        observed = rows if observed_column not in rows.columns else rows[rows[observed_column] == 1]
        band = _quantile_band(observed[["Year", target]], target, quantiles, min_count, year_step, min_fraction)
        records.extend((target, year, lo, hi, n) for year, lo, hi, n in band)
    return pd.DataFrame(records, columns=["target", "Year", "lo", "hi", "n"])


def band_at(band: pd.DataFrame, years: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Lower and upper bounds at *years*, interpolated in Year over the band's finite rows."""
    years = np.asarray(list(years), dtype=float)
    finite = band[np.isfinite(band["lo"]) & np.isfinite(band["hi"])].sort_values("Year")
    if finite.empty:
        return np.full(len(years), np.nan), np.full(len(years), np.nan)
    x = finite["Year"].to_numpy(dtype=float)
    lo = np.interp(years, x, finite["lo"].to_numpy(dtype=float), left=np.nan, right=np.nan)
    hi = np.interp(years, x, finite["hi"].to_numpy(dtype=float), left=np.nan, right=np.nan)
    return lo, hi


def band_position(values: pd.Series, band: pd.DataFrame) -> pd.Series:
    """Where *values* (indexed by Year) sit within the band: 0 at ``lo``, 1 at ``hi``."""
    years = [int(y) for y in values.index]
    lo, hi = band_at(band, years)
    width = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        position = (values.to_numpy(dtype=float) - lo) / np.where(width > 0, width, np.nan)
    return pd.Series(position, index=values.index)


# ── Levers ────────────────────────────────────────────────────────────────


def resolve_anchor_years(
    years: Sequence[int],
    history_years: Sequence[int],
    anchors: Sequence[int] = WHATIF_ANCHOR_YEARS,
) -> List[int]:
    """The anchor years a trajectory can be pinned at.

    Anchors inside the fixed history, or past the trajectory's end, are
    dropped, and the trajectory's last year is always an anchor so the end
    point is under the visitor's control whenever the series stops short.
    """
    years = sorted(int(y) for y in years)
    last_history = max(int(y) for y in history_years)
    last_year = years[-1]
    resolved = [int(a) for a in anchors if last_history < int(a) < last_year]
    if last_year > last_history:
        resolved.append(last_year)
    return resolved


@dataclass(frozen=True)
class AnchorBounds:
    """A lever's slider range at one anchor year."""

    lo: float          # slider bounds, widened to include the baseline
    hi: float
    base: float        # the baseline's value in that year
    band_lo: float     # the AR6 range itself
    band_hi: float


@dataclass
class LeverSpec:
    """Everything the controls need to know about one input."""

    feature: str
    years: List[int]
    history_years: List[int]
    baseline: pd.Series                 # Year -> value, whole trajectory
    reported: bool                      # never imputed in this trajectory
    anchors: Dict[int, AnchorBounds]
    band: pd.DataFrame                  # [Year, lo, hi]
    enabled: bool                       # anchors may be moved
    reason: str                         # why not, when they may not
    multiplier_bounds: Optional[Tuple[float, float]]   # the simple control's range
    multiplier_reason: str              # why the simple control is off, when it is

    @property
    def last_year(self) -> int:
        return self.years[-1]

    @property
    def last_history_year(self) -> int:
        return self.history_years[-1]


def _baseline_at(years: Sequence[int], baseline: pd.Series, year: int) -> float:
    values = baseline.reindex(years).to_numpy(dtype=float)
    return float(np.interp(year, np.asarray(years, dtype=float), values))


def build_lever_specs(
    rows: pd.DataFrame,
    bands: pd.DataFrame,
    raw_features: Sequence[str],
    history_steps: int,
    anchor_years: Sequence[int],
) -> List[LeverSpec]:
    """Lever specs for a trajectory: bounds at each anchor year, and what is locked."""
    years = [int(y) for y in rows["Year"]]
    history_years = years[:history_steps]
    band_by_feature = {f: g for f, g in bands.groupby("feature")} if len(bands) else {}
    empty_band = pd.DataFrame(columns=["Year", "lo", "hi"])

    specs = []
    for feature in raw_features:
        if feature not in rows.columns:
            continue
        baseline = pd.Series(pd.to_numeric(rows[feature], errors="coerce").to_numpy(dtype=float), index=years)
        reported = _reported_everywhere(rows, feature)
        band = band_by_feature.get(feature, empty_band)[["Year", "lo", "hi"]].reset_index(drop=True)
        lo_at, hi_at = band_at(band, list(anchor_years))

        anchors: Dict[int, AnchorBounds] = {}
        reason = ""
        for year, band_lo, band_hi in zip(anchor_years, lo_at, hi_at):
            base = _baseline_at(years, baseline, int(year))
            if not (np.isfinite(band_lo) and np.isfinite(band_hi)):
                reason = reason or f"too few AR6 scenarios report it in {year}"
                lo, hi = base, base
            elif band_hi <= band_lo:
                reason = reason or f"the AR6 scenarios all share one value in {year}"
                lo, hi = base, base
            else:
                lo, hi = min(band_lo, base), max(band_hi, base)
            if not np.isfinite(base):
                reason = reason or f"the baseline has no value in {year}"
            anchors[int(year)] = AnchorBounds(float(lo), float(hi), float(base), float(band_lo), float(band_hi))
        if not reported:
            reason = "imputed for this trajectory, not reported by the IAM"
        enabled = reason == ""

        multiplier_bounds: Optional[Tuple[float, float]] = None
        multiplier_reason = reason
        if enabled:
            end = anchors[int(anchor_years[-1])] if len(anchor_years) else None
            if end is None:
                multiplier_reason = "the trajectory has no year to move"
            elif end.base <= 0:
                multiplier_reason = f"the baseline is {end.base:g} in {anchor_years[-1]}; use the anchors"
            else:
                multiplier_bounds = (min(end.lo / end.base, 1.0), max(end.hi / end.base, 1.0))
                multiplier_reason = ""

        specs.append(
            LeverSpec(
                feature=feature,
                years=years,
                history_years=history_years,
                baseline=baseline,
                reported=reported,
                anchors=anchors,
                band=band,
                enabled=enabled,
                reason=reason,
                multiplier_bounds=multiplier_bounds,
                multiplier_reason=multiplier_reason,
            )
        )
    return specs


def _ramp_fraction(spec: LeverSpec, year: int) -> float:
    span = spec.last_year - spec.last_history_year
    return 0.0 if span <= 0 else (year - spec.last_history_year) / span


def ramp_anchor_values(spec: LeverSpec, multiplier: float) -> Dict[int, float]:
    """Anchor values that scale the baseline by 1 at the end of history and by *multiplier* at the last year."""
    return {
        year: bounds.base * (1.0 + (float(multiplier) - 1.0) * _ramp_fraction(spec, year))
        for year, bounds in spec.anchors.items()
    }


def multiplier_from_anchors(spec: LeverSpec, anchors: Mapping[int, float]) -> float:
    """The simple control's reading of a set of anchor values: the last year's ratio."""
    end = spec.anchors.get(spec.last_year)
    if end is None or not end.base > 0:
        return 1.0
    return float(anchors.get(spec.last_year, end.base)) / end.base


def interpolate_lever_path(
    years: Sequence[int],
    history_years: Sequence[int],
    baseline: pd.Series,
    anchors: Mapping[int, float],
) -> np.ndarray:
    """A lever's values at *years*: the baseline through history, then the anchors.

    Linear in Year from the last history year through each anchor, flat
    after the last one, so a trajectory with 10-year steps late on needs
    no special case.  Anchors inside history are ignored.
    """
    years = [int(y) for y in years]
    base = baseline.reindex(years).to_numpy(dtype=float)
    last_history = max(int(y) for y in history_years)
    knots = {int(y): float(v) for y, v in anchors.items() if int(y) > last_history}
    if not knots:
        return base
    knot_years = [last_history] + sorted(knots)
    knot_values = [float(baseline.get(last_history, np.nan))] + [knots[y] for y in sorted(knots)]
    if not np.isfinite(knot_values[0]):
        knot_years, knot_values = knot_years[1:], knot_values[1:]
    out = base.copy()
    for i, year in enumerate(years):
        if year <= last_history:
            continue
        out[i] = float(np.interp(year, knot_years, knot_values))
    return out


def apply_levers(
    rows: pd.DataFrame,
    edits: Mapping[str, Mapping[int, float]],
    history_steps: int,
) -> pd.DataFrame:
    """The trajectory with each edited input replaced by its anchored path.

    Only the named feature columns change, and only after the fixed history;
    ids, targets, masks, indicators and the time index stay as they were.
    """
    out = rows.copy()
    years = [int(y) for y in out["Year"]]
    history_years = years[:history_steps]
    for feature, anchors in edits.items():
        if feature not in out.columns or not anchors:
            continue
        baseline = pd.Series(pd.to_numeric(out[feature], errors="coerce").to_numpy(dtype=float), index=years)
        path = interpolate_lever_path(years, history_years, baseline, anchors)
        column = pd.Series(path, index=out.index)
        if pd.api.types.is_numeric_dtype(out[feature]):
            column = column.astype(out[feature].dtype)
        out[feature] = column
    return out


def preset_edits(preset: Mapping[str, float], specs: Sequence[LeverSpec]) -> Dict[str, Dict[int, float]]:
    """Edits that move each lever of *preset* to a position within its band at the last year.

    Position 0 is the band's lower bound, 1 its upper; the path ramps there
    from the end of history.  Levers the trajectory cannot move are skipped.
    """
    by_feature = {spec.feature: spec for spec in specs}
    edits: Dict[str, Dict[int, float]] = {}
    for feature, position in preset.items():
        spec = by_feature.get(feature)
        if spec is None or not spec.enabled:
            continue
        end = spec.anchors[spec.last_year]
        target = end.band_lo + float(position) * (end.band_hi - end.band_lo)
        if spec.multiplier_bounds is not None:
            edits[feature] = ramp_anchor_values(spec, target / end.base)
        else:
            edits[feature] = {
                year: bounds.base + (target - end.base) * _ramp_fraction(spec, year)
                for year, bounds in spec.anchors.items()
            }
    return edits


def band_coverage(edits: Mapping[str, Mapping[int, float]], specs: Sequence[LeverSpec]) -> Tuple[int, int]:
    """How many edited (lever, anchor year) values lie inside the AR6 range, of how many."""
    by_feature = {spec.feature: spec for spec in specs}
    inside = total = 0
    for feature, anchors in edits.items():
        spec = by_feature.get(feature)
        if spec is None:
            continue
        for year, value in anchors.items():
            bounds = spec.anchors.get(int(year))
            if bounds is None or not (np.isfinite(bounds.band_lo) and np.isfinite(bounds.band_hi)):
                continue
            total += 1
            inside += int(bounds.band_lo <= float(value) <= bounds.band_hi)
    return inside, total


# ── Results ───────────────────────────────────────────────────────────────


@dataclass
class WhatifResult:
    """One emulation: the inputs used, the IAM's values and both forecasts."""

    run_id: str
    region: str
    model: str
    scenario: str
    split: str
    category: str
    model_family: str
    years: List[int]
    history_years: List[int]
    inputs_baseline: pd.DataFrame    # Year x raw features
    inputs_edited: pd.DataFrame
    iam: pd.DataFrame                # Year x targets, NaN where unobserved
    pred_baseline: pd.DataFrame      # predicted Year x targets
    pred_edited: pd.DataFrame
    edits: Dict[str, Dict[int, float]]
    coverage: Tuple[int, int]
    baseline_fit: Dict[str, float]   # how the forecast of the unchanged inputs matches the IAM

    @property
    def targets(self) -> List[str]:
        return list(self.pred_baseline.columns)

    @property
    def predicted_years(self) -> List[int]:
        return [int(y) for y in self.pred_baseline.index]


def prediction_frame(tidy: pd.DataFrame, targets: Sequence[str]) -> pd.DataFrame:
    """Year-indexed forecasts from the tidy ``<target>_pred`` output of one trajectory."""
    columns = {f"{t}_pred": t for t in targets}
    out = tidy.set_index(tidy["Year"].astype(int))[list(columns)].rename(columns=columns)
    out.index.name = "Year"
    return out.astype(float).sort_index()


def iam_frame(rows: pd.DataFrame, targets: Sequence[str]) -> pd.DataFrame:
    """The IAM's own target values by Year, NaN where the value was not observed."""
    out = rows.set_index(rows["Year"].astype(int))[list(targets)].astype(float)
    out.index.name = "Year"
    for target in targets:
        observed = f"{target}__observed"
        if observed in rows.columns:
            out.loc[(rows[observed].to_numpy() == 0), target] = np.nan
    return out


def fit_metrics(iam: pd.DataFrame, pred: pd.DataFrame) -> Dict[str, float]:
    """Pooled R2, RMSE and MAE of *pred* against the observed elements of *iam*."""
    years = pred.index.intersection(iam.index)
    y_true = iam.loc[years, pred.columns].to_numpy(dtype=float)
    y_pred = pred.loc[years].to_numpy(dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if yt.size == 0:
        return {"Count": 0, "R2": math.nan, "RMSE": math.nan, "MAE": math.nan}
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = math.nan if ss_tot == 0 else 1.0 - float(np.sum((yt - yp) ** 2)) / ss_tot
    return {
        "Count": int(yt.size),
        "R2": r2,
        "RMSE": float(np.sqrt(np.mean((yt - yp) ** 2))),
        "MAE": float(np.mean(np.abs(yt - yp))),
    }


def assemble_result(
    run_id: str,
    region: str,
    candidate: BaselineCandidate,
    rows: pd.DataFrame,
    edited_rows: pd.DataFrame,
    raw_features: Sequence[str],
    targets: Sequence[str],
    pred_baseline_tidy: pd.DataFrame,
    pred_edited_tidy: pd.DataFrame,
    edits: Mapping[str, Mapping[int, float]],
    coverage: Tuple[int, int],
    history_steps: int,
) -> WhatifResult:
    years = [int(y) for y in rows["Year"]]
    features = [f for f in raw_features if f in rows.columns]
    iam = iam_frame(rows, targets)
    pred_baseline = prediction_frame(pred_baseline_tidy, targets)
    return WhatifResult(
        run_id=run_id,
        region=region,
        model=candidate.model,
        scenario=candidate.scenario,
        split=candidate.split,
        category=str(rows["Scenario_Category"].iloc[0]) if "Scenario_Category" in rows.columns else "",
        model_family=str(rows["Model_Family"].iloc[0]) if "Model_Family" in rows.columns else "",
        years=years,
        history_years=years[:history_steps],
        inputs_baseline=rows.set_index(pd.Index(years, name="Year"))[features].astype(float),
        inputs_edited=edited_rows.set_index(pd.Index(years, name="Year"))[features].astype(float),
        iam=iam,
        pred_baseline=pred_baseline,
        pred_edited=prediction_frame(pred_edited_tidy, targets),
        edits={f: {int(y): float(v) for y, v in a.items()} for f, a in edits.items()},
        coverage=(int(coverage[0]), int(coverage[1])),
        baseline_fit=fit_metrics(iam, pred_baseline),
    )


def export_frame(result: WhatifResult) -> pd.DataFrame:
    """One row per Year: the inputs used, the IAM values and both forecasts."""
    parts = [
        result.inputs_baseline.add_prefix("input_baseline:"),
        result.inputs_edited.add_prefix("input:"),
        result.iam.add_prefix("iam:"),
        result.pred_baseline.add_prefix("pred_baseline:"),
        result.pred_edited.add_prefix("pred:"),
    ]
    out = pd.concat(parts, axis=1).sort_index()
    out.index.name = "Year"
    return out.reset_index()


def _json_safe(value: Any) -> Any:
    """*value* with numpy scalars as Python ones, NaN/inf as None, keys as str."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def result_metadata(result: WhatifResult, region_r2: Optional[float], preset: Optional[str]) -> dict:
    """What the saved plot records: JSON-safe, with the keys the sidebar reads."""
    predicted = result.predicted_years
    return _json_safe({
        "timestamp": datetime.datetime.now().isoformat(),
        "view": "whatif",
        "run_id": result.run_id,
        "region": result.region,
        "regions": [result.region],
        "scenario_categories": [result.category] if result.category else [],
        "model_families": [result.model_family] if result.model_family else [],
        "baseline": {"model": result.model, "scenario": result.scenario, "split": result.split},
        "num_data_points": len(predicted) * len(result.targets),
        "years": result.years,
        "history_years": result.history_years,
        "region_pooled_r2": region_r2,
        "preset": preset,
        "edits": {f: {str(y): v for y, v in a.items()} for f, a in result.edits.items()},
        "band_coverage": {"inside": result.coverage[0], "total": result.coverage[1]},
        "metrics": dict(result.baseline_fit),
        "predictions": {
            "years": predicted,
            "baseline": {t: result.pred_baseline[t].tolist() for t in result.targets},
            "edited": {t: result.pred_edited[t].tolist() for t in result.targets},
        },
    })
