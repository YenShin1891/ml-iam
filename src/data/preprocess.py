import os
import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Tuple, cast
from sklearn.preprocessing import StandardScaler

from configs.paths import DATA_PATH
from configs.data import (
    DEFAULT_DATASET,
    N_LAG_FEATURES,
    OUTPUT_VARIABLES,
    INDEX_COLUMNS,
    NON_FEATURE_COLUMNS,
    CATEGORICAL_COLUMNS,
    MAX_YEAR,
    SPLIT_SEED,
    REGION_CATEGORIES,
    INTERPOLATE_TARGETS,
    SCALE_AWARE_IMPUTATION,
    NORMALIZE_TARGETS_BY_POPULATION,
    POPULATION_COLUMN,
    KEEP_PARTIAL_TARGETS,
    IMPUTE_IRREGULAR_INTERVALS,
)

SPLIT_NAMES = ("train", "val", "test")


def observed_group_keys(data: pd.DataFrame) -> pd.DataFrame:
    """The (Model, Scenario, Region) groups that carry any target observation.

    This is the canonical group set for splitting.  Every prepare_features_*
    function drops rows whose targets are all unobserved, so it is the set the
    models actually see, and unlike a model's prepared frame it does not depend
    on that model's own row filtering (lag history, resampling, and so on).
    """
    obs_cols = [c for c in observed_mask_columns(OUTPUT_VARIABLES) if c in data.columns]
    targets = [c for c in OUTPUT_VARIABLES if c in data.columns]

    if obs_cols:  # written before interpolation, so it marks real observations
        keep = data[obs_cols].to_numpy(dtype=bool).any(axis=1)
    elif targets:
        keep = data[targets].notna().to_numpy().any(axis=1)
    else:
        keep = np.ones(len(data), dtype=bool)

    return (
        data.loc[keep, INDEX_COLUMNS]
        .drop_duplicates()
        .sort_values(INDEX_COLUMNS)
        .reset_index(drop=True)
    )


def assign_group_splits(
    data: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = SPLIT_SEED,
) -> pd.DataFrame:
    """Assign every (Model, Scenario, Region) group to one split.

    Derive this once per dataset, not per model.  Each model filters rows
    differently, and shuffling a different group list produces a different
    partition — not a subset of one — so deriving it per model would let a
    group test for one model while training another.  The group set comes from
    :func:`observed_group_keys` so it does not depend on any model's filtering.

    Returns a frame of INDEX_COLUMNS plus a 'split' column.
    """
    keys = observed_group_keys(data)
    n_groups = len(keys)
    n_test = int(n_groups * test_size)
    n_val = int(n_groups * val_size)
    n_train = n_groups - n_test - n_val

    # permutation() is shuffle() over arange, i.e. the same Fisher-Yates draws
    # the previous implementation applied to its list of groups.
    order = np.random.RandomState(seed).permutation(n_groups)
    split = np.empty(n_groups, dtype=object)
    split[order[:n_train]] = "train"
    split[order[n_train:n_train + n_val]] = "val"
    split[order[n_train + n_val:]] = "test"

    keys["split"] = split
    logging.info(
        "Split assignment over %d groups: %s",
        n_groups,
        ", ".join(f"{name}={int((split == name).sum())}" for name in SPLIT_NAMES),
    )
    return keys


def split_data(
    prepared: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    assignment: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition *prepared* by group into train/val/test.

    *assignment* is the shared per-group assignment (see assign_group_splits);
    pass the run's, so every model is scored on the same test groups.  Without
    one, the assignment is derived from *prepared* alone.
    """
    if assignment is None:
        assignment = assign_group_splits(prepared, test_size, val_size)

    labels = prepared[INDEX_COLUMNS].merge(  # left merge preserves row order
        assignment[list(INDEX_COLUMNS) + ["split"]], on=list(INDEX_COLUMNS), how="left"
    )["split"]

    unassigned = int(labels.isna().sum())
    if unassigned:
        missing = (
            prepared.loc[labels.isna().values, INDEX_COLUMNS].drop_duplicates().shape[0]
        )
        logging.warning(
            "%d rows in %d group(s) are absent from the split assignment; "
            "treating them as train so they cannot leak into test.",
            unassigned, missing,
        )
        labels = labels.fillna("train")

    frames = tuple(
        prepared[(labels == name).to_numpy()].reset_index(drop=True)
        for name in SPLIT_NAMES
    )
    train_data, val_data, test_data = frames
    for name, frame in zip(SPLIT_NAMES, frames):
        if frame.empty:
            logging.warning("Split '%s' is empty after model-specific filtering", name)

    logging.info(f"Train: {len(train_data)} rows, Val: {len(val_data)} rows, Test: {len(test_data)} rows")

    return train_data, val_data, test_data



def build_categorical_vocabularies(data, columns=None) -> dict:
    """Build one stable category vocabulary per categorical column.

    Call this on the *full* frame before splitting, then pass the result to
    :func:`encode_categorical_columns` for each split.  Encoding each split
    against its own vocabulary assigns different integer codes to the same
    label whenever a split is missing a category.
    """
    if columns is None:
        columns = CATEGORICAL_COLUMNS

    vocabularies = {}
    for col in columns:
        if col not in data.columns:
            continue
        vocabularies[col] = sorted(set(data[col].astype(str)))

    if 'Region' in vocabularies and not REGION_CATEGORIES:
        # Keep the legacy module global in step for code that still reads it.
        set_region_categories(data['Region'])
    return vocabularies


def resolve_categorical_vocabularies(data, persisted=None, columns=None) -> dict:
    """Reconcile the vocabularies a run was encoded with against *data*.

    *persisted* is what the run saved earlier (see RunStore.categories_for).
    Saved categories keep their positions so codes never shift; labels that
    only appear in *data* are appended, which is the only safe direction — an
    already-trained model has no embedding row for them, but shifting existing
    codes would silently remap every region.
    """
    derived = build_categorical_vocabularies(data, columns)
    if not persisted:
        return derived

    resolved = {}
    for col, values in derived.items():
        saved = list(persisted.get(col) or [])
        if not saved:
            resolved[col] = values
            continue
        unseen = [v for v in values if v not in set(saved)]
        if unseen:
            logging.warning(
                "Column '%s': %d label(s) absent from the run's saved vocabulary, "
                "appended after the known ones: %s",
                col, len(unseen), unseen[:5],
            )
        resolved[col] = saved + unseen

    # Keep any column the current data no longer has, so old codes still decode.
    for col, values in persisted.items():
        resolved.setdefault(col, list(values))
    return resolved


def encode_categorical_columns(data, columns, vocabularies=None):
    """Encode categorical columns to integer codes.

    *vocabularies* maps column name to an ordered category list (see
    :func:`build_categorical_vocabularies`).  Columns without an entry fall
    back to per-frame ``.cat.codes``, which is only safe when the frame holds
    the whole dataset.
    """
    vocabularies = vocabularies or {}
    for col in columns:
        if col not in data.columns:
            continue

        categories = vocabularies.get(col)
        if categories is None and col == 'Region':
            if not REGION_CATEGORIES:
                set_region_categories(data[col])
            categories = REGION_CATEGORIES

        if categories is None:
            data[col] = data[col].astype('category').cat.codes
            continue

        codes = pd.Categorical(
            data[col].astype(str), categories=list(categories), ordered=True
        ).codes
        n_unknown = int((codes == -1).sum())
        if n_unknown:
            logging.warning(
                "Column '%s': %d value(s) outside the training vocabulary encoded as -1",
                col, n_unknown,
            )
        data[col] = codes.astype('float32') if col == 'Region' else codes
    return data


def decode_categorical_column(codes, categories) -> pd.Series:
    """Recover the labels behind codes from :func:`encode_categorical_columns`.

    The sequence models feed integer codes to their embeddings, so anything
    downstream that reasons about region *names* -- filtering SHAP to R10, say
    -- has to translate back first.  Codes outside the vocabulary (-1, and the
    floats Region is stored as) decode to None rather than silently indexing
    from the end of the list.
    """
    vocabulary = list(categories)
    series = codes if isinstance(codes, pd.Series) else pd.Series(list(codes))

    def label(code):
        try:
            index = int(code)
        except (TypeError, ValueError):
            return None
        return vocabulary[index] if 0 <= index < len(vocabulary) else None

    return series.map(label)


def add_missingness_indicators(
    prepared: pd.DataFrame,
    features: list,
    time_known: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
):
    """Add <feature>_is_missing indicators before splitting the dataset.
    """
    if time_known is None:
        time_known = ["Year", "DeltaYears"]
    if categorical_columns is None:
        categorical_columns = CATEGORICAL_COLUMNS

    updated_features = list(features)
    excluded = set(categorical_columns) | set(time_known)

    for col in features:
        if col in excluded or col.endswith("_is_missing"):
            continue
        if col not in prepared.columns:
            continue
        indicator_name = f"{col}_is_missing"
        if indicator_name not in prepared.columns:
            prepared[indicator_name] = (
                prepared[col]
                .isna()
                .map({True: 1, False: 0})
                .astype("float32")
            )
        if indicator_name not in updated_features:
            updated_features.append(indicator_name)

    return prepared, updated_features


def impute_with_train_medians(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list,
    time_known: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
):
    """Impute continuous features with train medians.

    When SCALE_AWARE_IMPUTATION is enabled and a 'Region_Scale' column exists,
    computes medians per Region_Scale group so that ISO3 countries are imputed
    from other ISO3 values, R10 from R10, etc.  Falls back to global train
    median for features that are entirely NaN within a scale group.  Raises
    an error if a Region_Scale present in val/test has zero rows in train.
    """
    if time_known is None:
        time_known = ["Year", "DeltaYears"]
    if categorical_columns is None:
        categorical_columns = CATEGORICAL_COLUMNS

    excluded = set(categorical_columns) | set(time_known)

    use_scale = (
        SCALE_AWARE_IMPUTATION
        and "Region_Scale" in train_df.columns
    )

    if use_scale:
        # Validate: every Region_Scale in val/test must exist in train
        train_scales = set(train_df["Region_Scale"].unique())
        for split_name, split_df in [("val", val_df), ("test", test_df)]:
            split_scales = set(split_df["Region_Scale"].unique()) if "Region_Scale" in split_df.columns else set()
            missing = split_scales - train_scales
            if missing:
                raise ValueError(
                    f"Region_Scale values {missing} found in {split_name} split but not in training data. "
                    f"This means the train/val/test split left an entire geographic scale unrepresented in training. "
                    f"Training scales: {sorted(train_scales)}. "
                    f"Fix: check the split logic or the dataset — every Region_Scale must appear in the training set."
                )

        # Compute per-scale medians from train
        scale_medians = {}
        for scale in sorted(train_scales):
            scale_data = train_df[train_df["Region_Scale"] == scale]
            medians = {}
            for col in features:
                if col in excluded or col.endswith("_is_missing") or col not in scale_data.columns:
                    continue
                med = pd.to_numeric(scale_data[col], errors="coerce").median()
                if pd.notna(med):
                    medians[col] = med
            scale_medians[scale] = medians

        # Also compute global medians as fallback for features missing within a scale
        global_medians = {}
        for col in features:
            if col in excluded or col.endswith("_is_missing") or col not in train_df.columns:
                continue
            med = pd.to_numeric(train_df[col], errors="coerce").median()
            global_medians[col] = med if pd.notna(med) else 0.0

        # Apply imputation per scale
        for frame in (train_df, val_df, test_df):
            if "Region_Scale" not in frame.columns:
                continue
            for scale in frame["Region_Scale"].unique():
                mask = frame["Region_Scale"] == scale
                s_medians = scale_medians.get(scale, {})
                for col in features:
                    if col in excluded or col.endswith("_is_missing") or col not in frame.columns:
                        continue
                    fill_val = s_medians.get(col, global_medians.get(col, 0.0))
                    frame.loc[mask, col] = frame.loc[mask, col].fillna(fill_val)

        logging.info(
            "Scale-aware imputation applied across %d Region_Scale groups: %s",
            len(train_scales), sorted(train_scales),
        )
    else:
        # Original global imputation
        for col in features:
            if col in excluded or col.endswith("_is_missing"):
                continue
            if col not in train_df.columns:
                continue
            median_value = pd.to_numeric(train_df[col], errors="coerce").median()
            if pd.isna(median_value):
                logging.warning("Column '%s' has all-NaN in train; filling with 0.0", col)
                median_value = 0.0
            for frame in (train_df, val_df, test_df):
                if col in frame.columns:
                    frame[col] = frame[col].fillna(median_value)

    return train_df, val_df, test_df


def _keep_partial_targets() -> bool:
    """Read KEEP_PARTIAL_TARGETS at call time.

    scripts/train.py overrides ``configs.data.KEEP_PARTIAL_TARGETS`` at
    runtime, so the module-level import captured above can be stale.
    """
    import configs.data as data_config

    return bool(getattr(data_config, "KEEP_PARTIAL_TARGETS", KEEP_PARTIAL_TARGETS))


def sanitize_target_scaler(scaler, targets=None) -> bool:
    """Repair NaN statistics left by targets with no observed training value.

    ``StandardScaler`` ignores NaN when fitting, so a target that is entirely
    unobserved yields ``mean_``/``scale_`` of NaN, which would turn every
    prediction for that target into NaN.  Fall back to the identity transform.
    Returns True when anything was repaired.
    """
    degenerate = ~np.isfinite(scaler.mean_) | ~np.isfinite(scaler.scale_)
    if not degenerate.any():
        return False

    names = (
        [targets[i] for i in np.flatnonzero(degenerate)]
        if targets is not None else np.flatnonzero(degenerate).tolist()
    )
    logging.warning(
        "Targets with no observed training values, standardised as identity: %s", names
    )
    scaler.mean_[degenerate] = 0.0
    scaler.scale_[degenerate] = 1.0
    if getattr(scaler, "var_", None) is not None:
        scaler.var_[degenerate] = 1.0
    return True


def _fit_target_scaler(y_train, y_val, y_test, targets):
    """Standardise targets, ignoring NaN (unobserved) elements.

    ``StandardScaler`` computes its statistics from the non-NaN entries and
    propagates NaN through ``transform``, so unobserved elements stay NaN for
    downstream per-target masking instead of being trained on as real zeros.
    """
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    if sanitize_target_scaler(y_scaler, targets):
        y_train_scaled = y_scaler.transform(y_train)

    return y_scaler, y_train_scaled, y_scaler.transform(y_val), y_scaler.transform(y_test)


def prepare_data(prepared, targets, features, categories=None, split_assignment=None):
    obs_cols = observed_mask_columns(targets)
    has_obs = all(c in prepared.columns for c in obs_cols)

    # One vocabulary for every split: encoding each separately would give the
    # same label different codes in train and test.  *categories* should come
    # from the run (RunStore.categories_for) so the codes also match the other
    # models and survive into the dashboard; deriving them from `prepared`
    # alone only sees the rows that survived lag filtering.
    if categories is None:
        categories = build_categorical_vocabularies(prepared, CATEGORICAL_COLUMNS)

    train_data, val_data, test_data = split_data(prepared, assignment=split_assignment)

    X_train = train_data[features].copy()
    X_train_index_columns = train_data[[col for col in INDEX_COLUMNS if col not in features]].copy()
    y_train = train_data[targets].values.copy()
    X_val = val_data[features].copy()
    X_val_index_columns = val_data[[col for col in INDEX_COLUMNS if col not in features]].copy()
    y_val = val_data[targets].values.copy()
    X_test = test_data[features].copy()
    X_test_index_columns = test_data[[col for col in INDEX_COLUMNS if col not in features]].copy()
    y_test = test_data[targets].values.copy()

    # Extract observed masks (float32 arrays matching y shape)
    if has_obs:
        obs_train = train_data[obs_cols].values.copy()
        obs_val = val_data[obs_cols].values.copy()
        obs_test = test_data[obs_cols].values.copy()
    else:
        obs_train = np.ones_like(y_train, dtype=np.float32)
        obs_val = np.ones_like(y_val, dtype=np.float32)
        obs_test = np.ones_like(y_test, dtype=np.float32)

    X_train = encode_categorical_columns(X_train, CATEGORICAL_COLUMNS, categories)
    X_val = encode_categorical_columns(X_val, CATEGORICAL_COLUMNS, categories)
    X_test = encode_categorical_columns(X_test, CATEGORICAL_COLUMNS, categories)

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_val_scaled = x_scaler.transform(X_val)
    X_test_scaled = x_scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    # Blank out unobserved targets so they are neither fitted on nor trained
    # on.  Interpolated values count as unobserved: the LSTM/TFT masked losses
    # already skip them, and XGBoost has to agree or the models see different
    # supervision.  With KEEP_PARTIAL_TARGETS=False every row is complete and
    # the consumer (multi-output XGBRegressor) cannot take NaN, so leave it.
    if _keep_partial_targets():
        y_train = np.where(obs_train.astype(bool), y_train, np.nan)
        y_val = np.where(obs_val.astype(bool), y_val, np.nan)
        y_test = np.where(obs_test.astype(bool), y_test, np.nan)

    y_scaler, y_train_scaled, y_val_scaled, y_test_scaled = _fit_target_scaler(
        y_train, y_val, y_test, targets
    )

    X_test_with_index_scaled = pd.concat(
        [X_test_scaled.reset_index(drop=True), X_test_index_columns.reset_index(drop=True)],
        axis=1
    )

    train_groups = np.asarray(train_data[INDEX_COLUMNS].astype(str).agg('_'.join, axis=1))
    val_groups = np.asarray(val_data[INDEX_COLUMNS].astype(str).agg('_'.join, axis=1))

    return (
        X_train_scaled, y_train_scaled, X_train_index_columns,
        X_val_scaled, y_val_scaled, X_val_index_columns,
        X_test_with_index_scaled, y_test_scaled,
        test_data,
        x_scaler, y_scaler,
        train_groups, val_groups,
        obs_train, obs_val, obs_test,
        categories,
    )

def load_and_process_data(version=None) -> pd.DataFrame:
    logging.info("Loading and processing data...")
    # Load the dataset specified in configs.data.DEFAULT_DATASET or from version subdirectory.
    # If version is provided, use version/processed_series.csv, otherwise use DEFAULT_DATASET.
    if version:
        dataset_path = os.path.join(DATA_PATH, version, "processed_series.csv")
    else:
        dataset_path = os.path.join(DATA_PATH, DEFAULT_DATASET)

    # Validate dataset path
    if not os.path.isfile(dataset_path):
        versions_file = os.path.join(DATA_PATH, "dataset_versions.txt")
        available_versions = None
        try:
            if os.path.isfile(versions_file) and os.path.getsize(versions_file) > 0:
                with open(versions_file, "r") as f:
                    versions = [ln.strip() for ln in f if ln.strip()]
                if versions:
                    available_versions = versions[-10:]
        except Exception:
            available_versions = None

        abs_path = os.path.abspath(dataset_path)
        if version is None:
            hint = (
                f"DEFAULT_DATASET is configured as '{DEFAULT_DATASET}', but no file was found at: {abs_path}. "
                "Set DATA_PATH in configs/paths.py and run `make process-data`, or pass --dataset <version_name>."
            )
        else:
            hint = (
                f"Processed dataset not found for version '{version}'. Expected file at: {abs_path}. "
                "Set DATA_PATH in configs/paths.py to the directory that contains processed datasets, "
                "or run `make process-data` to generate it."
            )

        if available_versions:
            hint += f" Available versions (last 10): {available_versions}"

        raise FileNotFoundError(hint)
        
    logging.info(f"Reading processed dataset: {dataset_path}")
    processed_series = pd.read_csv(dataset_path)
    # Identify year and non-year columns robustly
    all_cols = list(processed_series.columns)
    year_cols = [c for c in all_cols if str(c).isdigit()]
    try:
        cutoff = int(MAX_YEAR)
        year_cols = [c for c in year_cols if int(c) <= cutoff]
    except Exception:
        logging.warning("MAX_YEAR invalid; skipping year cutoff filter.")
    non_year_cols = [c for c in all_cols if c not in year_cols]

    year_melted = processed_series.melt(
        id_vars=non_year_cols, value_vars=year_cols, var_name='Year', value_name='value'
    )
    # Build pivot index — include Region_Scale if present
    pivot_index = ['Model', 'Model_Family', 'Scenario', 'Scenario_Category', 'Region']
    if 'Region_Scale' in non_year_cols:
        pivot_index.append('Region_Scale')
    pivot_index.append('Year')

    var_pivoted = year_melted.pivot_table(
        index=pivot_index,
        columns='Variable', values='value'
    ).reset_index()

    # Derive REGION_CATEGORIES from the actual data so embeddings always
    # cover every region present, regardless of target-filtering changes.
    if 'Region' in var_pivoted.columns:
        set_region_categories(var_pivoted['Region'])

    return var_pivoted


def set_region_categories(regions) -> None:
    """Derive REGION_CATEGORIES from a data column and update module globals.

    Mutates the existing list in-place so that ``from configs.data import
    REGION_CATEGORIES`` bindings see the update.
    """
    from configs.data import REGION_CATEGORIES, REGION_CODE_TO_LABEL

    cats = sorted(set(str(r) for r in regions))
    REGION_CATEGORIES.clear()
    REGION_CATEGORIES.extend(cats)
    REGION_CODE_TO_LABEL.clear()
    REGION_CODE_TO_LABEL.update({idx: r for idx, r in enumerate(cats)})


def observed_mask_columns(output_variables: list) -> list:
    """Return the list of ``{var}__observed`` column names for *output_variables*."""
    return [f"{var}__observed" for var in output_variables]


def interpolate_targets(
    data: pd.DataFrame,
    group_cols: list,
    output_variables: list,
) -> pd.DataFrame:
    """Interpolate target values within each group across years.

    For each (Model, Scenario, Region) group, fills interior NaN target
    values using linear interpolation (weighted by year). This ensures lag
    features are computed from real (interpolated) values instead of NaN.

    Only fills targets that have at least one non-NaN value in the group.

    Before interpolating, captures ``{var}__observed`` boolean columns so
    downstream code can distinguish originally-observed from filled values.
    """
    data = data.sort_values(group_cols + ["Year"]).copy()

    # Capture observation mask *before* any interpolation
    for col in output_variables:
        obs_col = f"{col}__observed"
        if obs_col not in data.columns:
            data[obs_col] = data[col].notna().astype("float32")

    before_nans = data[output_variables].isna().sum().sum()

    data["Year"] = pd.to_numeric(data["Year"], errors="coerce")

    def _interp_group(grp):
        grp = grp.set_index("Year").sort_index()
        for col in output_variables:
            if col in grp.columns and grp[col].notna().any():
                grp[col] = pd.to_numeric(grp[col], errors="coerce")
                grp[col] = grp[col].interpolate(method="index")
        return grp.reset_index()

    data = data.groupby(group_cols, group_keys=False).apply(_interp_group)
    data = data.reset_index(drop=True)

    after_nans = data[output_variables].isna().sum().sum()
    filled = before_nans - after_nans
    logging.info(
        "Target interpolation: filled %d NaN values (%d → %d remaining)",
        filled, before_nans, after_nans,
    )
    return data


def resample_to_uniform_intervals(
    data: pd.DataFrame,
    group_cols: list,
    output_variables: list,
    interval: int = 5,
) -> pd.DataFrame:
    """Resample groups with >interval-year spacing onto a uniform grid.

    For each (Model, Scenario, Region) group whose minimum year-to-year
    difference exceeds *interval*, intermediate rows are inserted at every
    *interval* years.  Numeric columns (targets and features) are linearly
    interpolated; observation masks (``{var}__observed``) are set to 0 for
    the inserted rows so they are masked in loss and metrics.
    """
    obs_cols = observed_mask_columns(output_variables)
    groups: list = []
    n_resampled = 0

    for _key, grp in data.groupby(group_cols, sort=False):
        grp = grp.sort_values("Year")
        years = grp["Year"].values
        diffs = np.diff(years)

        if len(diffs) == 0 or diffs.min() <= interval:
            groups.append(grp)
            continue

        n_resampled += 1
        full_years = np.arange(years.min(), years.max() + 1, interval)

        grp_indexed = grp.set_index("Year")
        grp_reindexed = grp_indexed.reindex(full_years)

        # Metadata / categorical columns are constant within group — fill.
        fill_cols = set(group_cols) | {
            c for c in NON_FEATURE_COLUMNS if c != "Year"
        } | set(CATEGORICAL_COLUMNS)
        for col in fill_cols:
            if col in grp_reindexed.columns:
                grp_reindexed[col] = grp_reindexed[col].ffill().bfill()

        # Observation masks: new rows are unobserved.
        for col in obs_cols:
            if col in grp_reindexed.columns:
                grp_reindexed[col] = grp_reindexed[col].fillna(0.0)

        # Interpolate numeric columns (exclude __observed masks).
        numeric_cols = grp_reindexed.select_dtypes(include=[np.number]).columns
        interp_cols = [c for c in numeric_cols if not c.endswith("__observed")]
        grp_reindexed[interp_cols] = grp_reindexed[interp_cols].interpolate(
            method="index"
        )

        grp_reindexed = grp_reindexed.reset_index(names="Year")
        groups.append(grp_reindexed)

    result = pd.concat(groups, ignore_index=True)
    logging.info(
        "Resampled %d groups to uniform %d-year intervals", n_resampled, interval
    )
    return result


def normalize_targets_by_population(
    data: pd.DataFrame,
    targets: list,
    population_col: str = POPULATION_COLUMN,
) -> pd.DataFrame:
    """Divide each target column by population, in place (same column names).

    Rows with missing or non-positive population get NaN targets (they will
    be dropped downstream by the existing dropna(subset=targets) logic, same
    as any other row with a missing target). Callers should interpolate
    `population_col` (e.g. via `interpolate_targets(data, group_cols, [population_col])`)
    before calling this, to avoid unnecessarily dropping otherwise-good rows.
    """
    data = data.copy()

    population = pd.to_numeric(data[population_col], errors="coerce")
    invalid = population.isna() | (population <= 0)
    n_invalid = int(invalid.sum())
    if n_invalid:
        logging.warning(
            "Population normalization: %d/%d rows have missing/non-positive %s; "
            "affected target values set to NaN and will be dropped downstream.",
            n_invalid, len(data), population_col,
        )

    safe_population = population.where(~invalid)
    for col in targets:
        data[col] = pd.to_numeric(data[col], errors="coerce") / safe_population

    return data


def denormalize_by_population(values: np.ndarray, population: np.ndarray) -> np.ndarray:
    """Multiply per-capita values back to absolute units, elementwise by row.

    `values` may be shape (n,) or (n, k); `population` is shape (n,) and is
    broadcast across the k target columns. NaNs propagate naturally.
    """
    values = np.asarray(values, dtype=float)
    population_col_vec = np.asarray(population, dtype=float).reshape(-1, 1)
    if values.ndim == 1:
        return (values.reshape(-1, 1) * population_col_vec).ravel()
    return values * population_col_vec


def add_lag_features(
    data: pd.DataFrame,
    group_cols: list,
    output_variables: list,
    n_lags: int = N_LAG_FEATURES,
    lag_required: bool = True,
) -> pd.DataFrame:
    """Add lagged target features using vectorized groupby.shift().

    ~250x faster than the per-group .apply() approach on 23k groups.
    """
    prepared = data.sort_values(group_cols + ['Year']).copy()

    for lag in range(1, n_lags + 1):
        shifted = prepared.groupby(group_cols, sort=False)[output_variables].shift(lag)
        for col in output_variables:
            prefix = 'prev_' if lag == 1 else f'prev{lag}_'
            prepared[f'{prefix}{col}'] = shifted[col]

    if lag_required:
        row_num = prepared.groupby(group_cols, sort=False).cumcount()
        prepared = prepared[row_num >= n_lags].reset_index(drop=True)

    return cast(pd.DataFrame, prepared)


def prepare_features_and_targets(data: pd.DataFrame, lag_required: bool = True) -> tuple:
    """
    Prepare features and targets for XGBoost model.

    Args:
        data: Input data DataFrame
        lag_required: When True, drop rows without a full history of lag features.
    """
    logging.info(
        "Preparing features and targets for XGBoost (lag_required=%s)...",
        lag_required,
    )

    if NORMALIZE_TARGETS_BY_POPULATION:
        data = interpolate_targets(data, INDEX_COLUMNS, [POPULATION_COLUMN])
        data = normalize_targets_by_population(data, OUTPUT_VARIABLES, POPULATION_COLUMN)

    # Always capture observation mask before interpolation fills NaNs.
    # interpolate_targets creates them internally, but if INTERPOLATE_TARGETS
    # is False we still need them.
    for col in OUTPUT_VARIABLES:
        obs_col = f"{col}__observed"
        if obs_col not in data.columns:
            data[obs_col] = data[col].notna().astype("float32")

    if INTERPOLATE_TARGETS:
        data = interpolate_targets(data, INDEX_COLUMNS, OUTPUT_VARIABLES)

    prepared = add_lag_features(data, INDEX_COLUMNS, OUTPUT_VARIABLES, lag_required=lag_required)
    prepared['Year'] = prepared['Year'].astype(int)

    targets = OUTPUT_VARIABLES
    obs_cols = observed_mask_columns(OUTPUT_VARIABLES)
    features = [
        col for col in prepared.columns
        if col not in NON_FEATURE_COLUMNS
        and col not in targets
        and not col.endswith("__observed")
    ]

    if KEEP_PARTIAL_TARGETS:
        # Keep rows where *at least one* target is observed (element-wise
        # masking handles the rest).  Rows where *all* targets are missing
        # carry zero gradient and waste memory, so drop them.
        any_observed = prepared[obs_cols].any(axis=1)
        n_dropped = int((~any_observed).sum())
        if n_dropped:
            logging.info("Dropped %d rows where all targets are unobserved", n_dropped)
        prepared = prepared[any_observed].reset_index(drop=True)
    else:
        # Original behaviour: require every target to be present.
        prepared = prepared.dropna(subset=targets).reset_index(drop=True)

    if not lag_required:
        missing_lag_rows = prepared[features].isna().any(axis=1).sum()
        if missing_lag_rows:
            logging.info(
                "Lag requirement disabled: retained %d rows with missing lag features",
                missing_lag_rows,
            )

    return prepared, features, targets


def prepare_features_and_targets_sequence(
    data: pd.DataFrame,
    **_kwargs,
) -> tuple:
    """
    Prepare features and targets for sequence models (TFT, LSTM, etc.).

    Sequence models rely on their temporal architecture (encoder attention,
    LSTM hidden state) to capture history — explicit lag features are not
    used.  This function builds Step and DeltaYears for time-series indexing.
    """
    logging.info("Preparing features and targets for sequence models...")

    if NORMALIZE_TARGETS_BY_POPULATION:
        data = interpolate_targets(data, INDEX_COLUMNS, [POPULATION_COLUMN])
        data = normalize_targets_by_population(data, OUTPUT_VARIABLES, POPULATION_COLUMN)

    # Always capture observation mask before interpolation fills NaNs.
    for col in OUTPUT_VARIABLES:
        obs_col = f"{col}__observed"
        if obs_col not in data.columns:
            data[obs_col] = data[col].notna().astype("float32")

    if INTERPOLATE_TARGETS:
        data = interpolate_targets(data, INDEX_COLUMNS, OUTPUT_VARIABLES)

    if IMPUTE_IRREGULAR_INTERVALS:
        data = resample_to_uniform_intervals(data, INDEX_COLUMNS, OUTPUT_VARIABLES)

    prepared = data.copy()
    prepared['Year'] = prepared['Year'].astype(int)

    targets = OUTPUT_VARIABLES
    obs_cols = observed_mask_columns(OUTPUT_VARIABLES)
    features = [
        col for col in prepared.columns
        if col not in NON_FEATURE_COLUMNS
        and col not in targets
        and not col.endswith("__observed")
    ]

    if KEEP_PARTIAL_TARGETS:
        any_observed = prepared[obs_cols].any(axis=1)
        n_dropped = int((~any_observed).sum())
        if n_dropped:
            logging.info("Dropped %d rows where all targets are unobserved", n_dropped)
        prepared = prepared[any_observed].reset_index(drop=True)
    else:
        prepared = prepared.dropna(subset=targets).reset_index(drop=True)

    # Make 'Step' and 'DeltaYears' after dropping NaNs
    # Step must align with group_ids used by sequence models
    group_cols = INDEX_COLUMNS
    prepared = prepared.sort_values(group_cols + ['Year'])
    prepared['Step'] = prepared.groupby(group_cols).cumcount().astype('int64')

    # Explicit gap feature: years elapsed since previous observation within each series
    prepared['DeltaYears'] = (
        prepared.groupby(group_cols)['Year'].diff().fillna(0).astype(int)
    )

    return prepared, features, targets

