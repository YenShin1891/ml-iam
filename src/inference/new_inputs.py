"""Run a trained emulator on scenarios it has never seen.

The dashboard's what-if view predicts from trajectories the run already
holds.  This module is the other door: a table of new scenarios in the
layout ``make process-data`` writes (one row per series and Variable, one
column per year) goes through the run's own preparation -- the same
functions its test phase used -- and into the same engines.

Nothing here needs the training data.  What preparation took from the
training set travels with the run as summary statistics: the scalers, the
category vocabularies and the imputation medians (artifacts/).

What each model needs from a new scenario:

* TFT and LSTM read the input variables only.  Target columns may be absent.
* XGBoost is autoregressive in the targets: it needs their values at the
  leading ``history_steps`` timesteps and predicts from there on.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from configs.data import CATEGORICAL_COLUMNS, UNITS_BY_OUTPUT
from src.utils.run_store import RunStore

ID_COLUMNS = ["Model", "Scenario", "Region"]
PLACEHOLDER_CATEGORY = "unknown"


class InputError(ValueError):
    """The scenario table cannot be emulated as given; the message says why."""


@dataclass
class PreparedInputs:
    """New scenarios as the engine wants them, and what was done to get there."""

    frame: pd.DataFrame
    history_steps: int
    notes: List[str] = field(default_factory=list)
    # TFT only: the stand-in group labels, mapped back to the caller's.
    aliases: Optional[pd.DataFrame] = None


def model_kind(run_id: str) -> str:
    return run_id.split("_", 1)[0]


def input_variables(features: List[str]) -> List[str]:
    """The Variables a scenario table should carry for *features*."""
    return [
        f for f in features
        if f not in CATEGORICAL_COLUMNS
        and not f.endswith("_is_missing")
        and not re.match(r"prev\d*_", f)
        and f not in ("Year", "DeltaYears", "Step")
    ]


def describe_requirements(run_id: str) -> Dict[str, object]:
    """What a scenario table for *run_id* must contain, from the run's artifacts."""
    store = RunStore(run_id)
    features, targets = store.load_features()
    categories = store.load_categories()
    kind = model_kind(run_id)
    lags = [int(m.group(1) or 1) for f in features if (m := re.match(r"prev(\d*)_", f))]
    return {
        "run_id": run_id,
        "model": kind,
        "input_variables": input_variables(features),
        "targets": list(targets),
        "target_units": {t: UNITS_BY_OUTPUT.get(t, "") for t in targets},
        "regions": list(categories.get("Region", [])),
        "model_families": list(categories.get("Model_Family", [])),
        "needs_target_history": kind == "xgb",
        "n_lags": max(lags) if lags else 0,
    }


# ── Reading ───────────────────────────────────────────────────────────────


def read_scenario_table(path: str) -> pd.DataFrame:
    """Read a wide scenario table and fill in the columns that can be derived."""
    table = pd.read_csv(path)
    return complete_scenario_table(table)


def complete_scenario_table(table: pd.DataFrame) -> pd.DataFrame:
    from src.data.process_data import get_model_family
    from src.utils.regions import region_scales

    table = table.copy()
    table.columns = [str(c).strip() for c in table.columns]
    missing = [c for c in ID_COLUMNS + ["Variable"] if c not in table.columns]
    if missing:
        raise InputError(
            f"The scenario table lacks column(s) {missing}. Expected: Model, Scenario, Region, "
            "Variable, then one column per year (e.g. 2020, 2025, ...)."
        )
    year_columns = [c for c in table.columns if c.isdigit()]
    if not year_columns:
        raise InputError("The scenario table has no year columns (headers such as 2020, 2025, ...).")

    if "Model_Family" not in table.columns:
        table["Model_Family"] = table["Model"].astype(str).map(get_model_family)
    if "Scenario_Category" not in table.columns:
        table["Scenario_Category"] = PLACEHOLDER_CATEGORY
    table["Scenario_Category"] = table["Scenario_Category"].fillna(PLACEHOLDER_CATEGORY)
    if "Region_Scale" not in table.columns:
        table["Region_Scale"] = region_scales(table["Region"]).to_numpy()

    duplicated = table.duplicated(ID_COLUMNS + ["Variable"])
    if duplicated.any():
        example = table.loc[duplicated, ID_COLUMNS + ["Variable"]].iloc[0].tolist()
        raise InputError(f"{int(duplicated.sum())} duplicated (Model, Scenario, Region, Variable) rows, e.g. {example}.")
    for column in year_columns:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    return table


# ── Preparation ───────────────────────────────────────────────────────────


def _check_vocabulary(long: pd.DataFrame, categories: Dict[str, List[str]]) -> None:
    for column in CATEGORICAL_COLUMNS:
        known = categories.get(column)
        if not known or column not in long.columns:
            continue
        unknown = sorted(set(long[column].astype(str)) - set(known))
        if unknown:
            raise InputError(
                f"{column} value(s) {unknown} are not in the run's vocabulary. The embeddings are "
                f"learned per label, so an unseen one cannot be emulated. Known {column} labels: {known}"
            )


def _long_frame(table: pd.DataFrame, variables: List[str], targets: List[str], notes: List[str]) -> pd.DataFrame:
    """Pivot to one row per (series, Year); every wanted column present."""
    from src.data.preprocess import pivot_processed_series

    wanted = set(variables) | set(targets)
    present = set(table["Variable"].astype(str))
    ignored = sorted(present - wanted)
    if ignored:
        notes.append(f"{len(ignored)} Variable(s) are not model inputs and were ignored, e.g. {ignored[:3]}.")
    table = table[table["Variable"].isin(wanted)]
    if table.empty:
        raise InputError("None of the table's Variables is an input of this run (see --describe).")

    long = pivot_processed_series(table, register_regions=False)
    long.columns.name = None
    absent = [v for v in variables if v not in long.columns]
    if absent:
        notes.append(
            f"{len(absent)} of {len(variables)} input Variable(s) are absent and are treated as "
            f"not reported, e.g. {absent[:3]}."
        )
    for column in absent + [t for t in targets if t not in long.columns]:
        long[column] = np.nan

    # A year where a series reports no input at all is not a timestep.
    reported = long[variables].notna().any(axis=1)
    long = long[reported].reset_index(drop=True)
    if long.empty:
        raise InputError("No (series, Year) row reports any input Variable.")
    return long


def _all_rows_kept(long: pd.DataFrame, targets: List[str]) -> pd.DataFrame:
    """Mark every row observed so preparation does not drop the unknown future.

    Preparation discards rows with no observed target: right for training
    data, wrong for a scenario whose targets are what we are after.
    """
    for target in targets:
        long[f"{target}__observed"] = np.float32(1.0)
    return long


def _drop_short(frame: pd.DataFrame, min_steps: int, notes: List[str]) -> pd.DataFrame:
    sizes = frame.groupby(ID_COLUMNS, observed=True, sort=False)["Step"].transform("size")
    short = frame.loc[sizes < min_steps, ID_COLUMNS].drop_duplicates()
    if len(short):
        notes.append(
            f"{len(short)} series have fewer than the {min_steps} timesteps this model needs and "
            f"were skipped, e.g. {short.iloc[0].tolist()}."
        )
        frame = frame[sizes >= min_steps].reset_index(drop=True)
    if frame.empty:
        raise InputError(f"No series has the {min_steps} timesteps this model needs.")
    return frame


def _alias_group_ids(frame: pd.DataFrame, template) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Swap Model/Scenario for labels the TFT's group-id encoders know.

    The encoders have a closed vocabulary, but group ids only tell series
    apart -- they are not model inputs (Region and Model_Family are, and
    those are checked).  Each (Model, Scenario) pair gets its own known pair.
    """
    def classes(column):
        encoders = template.categorical_encoders
        encoder = encoders.get(f"__group_id__{column}") or encoders.get(column)
        return sorted(encoder.classes_.keys())

    models, scenarios = classes("Model"), classes("Scenario")
    pairs = frame[["Model", "Scenario"]].drop_duplicates().reset_index(drop=True)
    if len(pairs) > len(models) * len(scenarios):
        raise InputError(f"At most {len(models) * len(scenarios)} (Model, Scenario) pairs per call; got {len(pairs)}.")
    pairs["_alias_model"] = [models[i // len(scenarios)] for i in range(len(pairs))]
    pairs["_alias_scenario"] = [scenarios[i % len(scenarios)] for i in range(len(pairs))]
    out = frame.merge(pairs, on=["Model", "Scenario"], how="left")
    out["Model"], out["Scenario"] = out.pop("_alias_model"), out.pop("_alias_scenario")
    return out, pairs


def prepare_new_inputs(run_id: str, table: pd.DataFrame, engine) -> PreparedInputs:
    """*table* (wide, completed) as the frame ``predict_windows`` expects."""
    from src.data import preprocess

    store = RunStore(run_id)
    features, targets = store.load_features()
    targets = list(targets)
    kind = model_kind(run_id)
    notes: List[str] = []
    variables = input_variables(features)
    long = _long_frame(table, variables, targets, notes)
    _check_vocabulary(long, store.load_categories())

    if kind == "xgb":
        n_lags = int(engine.metadata["n_lags"])
        history = int(engine.encoder_length)
        frame, _, _ = preprocess.prepare_features_and_targets(
            _all_rows_kept(long, targets), lag_required=False, n_lags=n_lags
        )
        frame["Step"] = frame.groupby(ID_COLUMNS, observed=True).cumcount()
        frame = _drop_short(frame, engine.min_steps, notes)
        # The first predicted step reads its lags from the last n_lags history steps.
        seed = frame[(frame["Step"] >= history - n_lags) & (frame["Step"] < history)]
        empty = seed[targets].isna().all(axis=1)
        if empty.any():
            example = seed.loc[empty, ID_COLUMNS].iloc[0].tolist()
            raise InputError(
                f"XGBoost predicts each step from the previous {n_lags}: it needs target values at "
                f"timesteps {history - n_lags + 1} to {history} of each series (the last history steps). "
                f"{int(empty.sum())} such row(s) have none, e.g. {example}. Use a TFT or LSTM run when "
                "no target history is available."
            )
        partial = [t for t in targets if seed[t].isna().any()]
        if partial:
            notes.append(
                f"{len(partial)} target(s) lack history values in some series and enter as not "
                f"reported, as in training: {partial[:3]}{' ...' if len(partial) > 3 else ''}. "
                "Their predictions for those series are less reliable."
            )
        return PreparedInputs(frame.reset_index(drop=True), history, notes)

    if kind not in ("tft", "lstm"):
        raise InputError(f"Unsupported emulator: {kind}")

    # Sequence models read inputs only; targets are placeholders.
    long[targets] = 0.0
    frame, seq_features, _ = preprocess.prepare_features_and_targets_sequence(_all_rows_kept(long, targets))
    frame, seq_features = preprocess.add_missingness_indicators(frame, seq_features)
    preprocess.apply_medians(frame, store.load_imputation_medians(), seq_features)
    unexpected = [f for f in features if f not in frame.columns]
    if unexpected:
        raise InputError(f"Preparation did not produce feature(s) the run expects: {unexpected[:5]}")
    frame = _drop_short(frame, engine.min_steps, notes)

    aliases = None
    if kind == "tft":
        frame, aliases = _alias_group_ids(frame, engine.template)
    return PreparedInputs(frame.reset_index(drop=True), int(engine.encoder_length), notes, aliases)


# ── Prediction ────────────────────────────────────────────────────────────


def ensure_imputation_medians(run_id: str) -> None:
    """Write the run's imputation medians if its training data is at hand.

    Runs trained before the medians were saved have them only implicitly,
    in the cached data; a published bundle carries the file instead.
    """
    store = RunStore(run_id)
    if store.has_imputation_medians() or model_kind(run_id) == "xgb":
        return
    if not (store.has_processed_data() and store.has_splits()):
        store.load_imputation_medians()  # raises with the explanation
    from src.data import preprocess

    prepared, features, _ = preprocess.prepare_features_and_targets_sequence(store.load_processed_data())
    prepared, features = preprocess.add_missingness_indicators(prepared, features)
    train, _, _ = preprocess.split_data(prepared, assignment=store.load_splits())
    store.save_imputation_medians(preprocess.compute_train_medians(train, features))


def predict_new_inputs(run_id: str, table: pd.DataFrame, *, device: str = "cpu") -> Tuple[pd.DataFrame, List[str]]:
    """Emulate every series of *table*.

    Returns ``(predictions, notes)``: one row per predicted (series, Year)
    with a column per target in the run's units, and what preparation had
    to skip or assume.  The leading history steps carry no prediction.
    """
    from src.inference.engines import load_engine, predict_windows

    ensure_imputation_medians(run_id)
    engine = load_engine(run_id, map_location=device)
    prepared = prepare_new_inputs(run_id, complete_scenario_table(table), engine)

    if model_kind(run_id) == "tft":
        from src.inference.tft_predict import predict_windows as predict_tft

        raw = predict_tft(engine, prepared.frame, accelerator=device)
    else:
        raw = predict_windows(engine, prepared.frame)

    if prepared.aliases is not None:
        back = prepared.aliases.rename(columns={"Model": "_model", "Scenario": "_scenario",
                                                "_alias_model": "Model", "_alias_scenario": "Scenario"})
        raw = raw.merge(back, on=["Model", "Scenario"], how="left")
        raw["Model"], raw["Scenario"] = raw.pop("_model"), raw.pop("_scenario")

    targets = list(engine.targets)
    out = raw[ID_COLUMNS + ["Year"]].copy()
    for target in targets:
        out[target] = raw[f"{target}_pred"].to_numpy()
    out["Year"] = out["Year"].astype(int)
    return out.sort_values(ID_COLUMNS + ["Year"]).reset_index(drop=True), prepared.notes


def to_iamc(predictions: pd.DataFrame, targets: List[str]) -> pd.DataFrame:
    """Predictions in the wide IAMC layout: a row per Variable, a column per year."""
    long = predictions.melt(id_vars=ID_COLUMNS + ["Year"], value_vars=targets, var_name="Variable", value_name="value")
    wide = long.pivot_table(index=ID_COLUMNS + ["Variable"], columns="Year", values="value").reset_index()
    wide.columns.name = None
    wide.insert(4, "Unit", wide["Variable"].map(UNITS_BY_OUTPUT).fillna(""))
    return wide
