"""
Experiment: Run ML-IAM inference for configured scenarios in a single region.

Step 1: Configure region, then print available models and scenarios.
Step 2: Configure scenario(s), then run inference for each model in the dataset.

For models that have actual IAM data for the configured scenarios in this region,
their ground truth is preserved and overlaid on plots. For all other models,
synthetic input rows are constructed using scenario features from a template model
and model-specific technology features from each model's own data (from the same
region if available, otherwise from global median).

Usage:
    # Explore what's available in a region:
    python experiments/inference_for_one_region.py --run-id xgb_75 --region R10REST_ASIA

    # Filter summary to specific model families:
    python experiments/inference_for_one_region.py --run-id xgb_75 --region IDN \
        --model-family MESSAGEix REMIND GCAM

    # Run inference for specific scenarios:
    python experiments/inference_for_one_region.py --run-id xgb_75 --region R10REST_ASIA \
        --scenarios "NGFS2_Divergent Net Zero Policies" "NGFS2_Net-Zero 2050"

    # Run inference filtered by model, family, and scenario category:
    python experiments/inference_for_one_region.py --run-id xgb_75 --region R10REST_ASIA \
        --scenario-categories C2 C3 --model-family GCAM REMIND --models "GCAM 5.3"

    # Use a different template model:
    python experiments/inference_for_one_region.py --run-id xgb_75 --region R10REST_ASIA \
        --scenarios "EN_NPi2020_600" --template-model "REMIND-MAgPIE 2.1-4.2"
"""

import argparse
import json
import logging
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd
import xgboost as xgb

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.data import (
    INDEX_COLUMNS,
    NON_FEATURE_COLUMNS,
    N_LAG_FEATURES,
    OUTPUT_VARIABLES,
    REGION_CATEGORIES,
)
from src.trainers.evaluation import test_xgb_autoregressively
from src.utils.run_store import RunStore
from src.utils.utils import get_run_root
from src.visualization.trajectories import format_large_numbers
from configs.data import OUTPUT_UNITS
from configs.visualization import (
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    PLOT_FONT_SIZE,
    PLOT_GRID_COLS,
    PLOT_GRID_ROWS,
    TICK_LABELSIZE,
    TRAJECTORY_GRID_FIGSIZE,
    Y_AXIS_NBINS_TRAJECTORY,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Features that represent scenario assumptions (shared across models)
SCENARIO_FEATURES = ["GDP|MER", "GDP|PPP", "Population", "Price|Carbon", "Yield|Cereal"]


# ── Helpers ────────────────────────────────────────────────────────────────

def detect_model_type(run_id: str) -> str:
    """Detect model type from run_id prefix (e.g. 'xgb_75' -> 'xgb')."""
    prefix = run_id.split("_", 1)[0].lower()
    if prefix in ("xgb", "xgboost"):
        return "xgb"
    if prefix in ("lstm",):
        return "lstm"
    if prefix in ("tft",):
        return "tft"
    raise ValueError(f"Cannot detect model type from run_id '{run_id}'. Expected prefix: xgb_, lstm_, tft_")


def load_artifacts(run_id: str):
    """Load model, scalers, and feature list from a trained run.

    Returns (model_type, artifacts_dict) where artifacts_dict contains
    everything needed for inference, varying by model type.
    """
    model_type = detect_model_type(run_id)
    run_root = get_run_root(run_id)
    store = RunStore(run_id)

    with open(os.path.join(run_root, "artifacts", "features.json")) as f:
        feat_meta = json.load(f)

    if model_type == "xgb":
        model = xgb.XGBRegressor()
        model.load_model(os.path.join(run_root, "checkpoints", "final_best.json"))
        return model_type, {
            "model": model,
            "x_scaler": store.load_artifact("x_scaler.pkl"),
            "y_scaler": store.load_artifact("y_scaler.pkl"),
            "features": feat_meta["features"],
            "targets": feat_meta["targets"],
        }

    if model_type == "lstm":
        from src.trainers.lstm_trainer import LSTMModel
        from configs.models.lstm import LSTMTrainerConfig

        model = LSTMModel.load_from_checkpoint(os.path.join(run_root, "final", "best.ckpt"))
        model.eval()

        meta = {}
        if store.has_train_meta():
            meta = store.load_train_meta()

        config = LSTMTrainerConfig(**(meta.get("lstm_config", {})))

        return model_type, {
            "model": model,
            "scaler_X": store.load_artifact("lstm_scaler_X.pkl"),
            "scaler_y": store.load_artifact("lstm_scaler_y.pkl"),
            "features": meta.get("lstm_features", feat_meta["features"]),
            "raw_features": meta.get("lstm_raw_features", feat_meta["features"]),
            "targets": feat_meta["targets"],
            "non_numeric_features": meta.get("lstm_non_numeric_features", []),
            "categorical_features": meta.get("lstm_categorical_features", []),
            "model_family_categories": meta.get("lstm_model_family_categories"),
            "sequence_length": meta.get("lstm_sequence_length", config.sequence_length),
            "target_offset": meta.get("lstm_target_offset", config.target_offset),
            "config": config,
        }

    if model_type == "tft":
        from src.trainers.tft_model import load_tft_checkpoint
        from src.trainers.tft_dataset import load_dataset_template

        model = load_tft_checkpoint(run_id)
        template = load_dataset_template(run_id)

        return model_type, {
            "model": model,
            "dataset_template": template,
            "features": feat_meta["features"],
            "targets": feat_meta["targets"],
        }

    raise ValueError(f"Unsupported model type: {model_type}")


def encode_categorical_columns(data: pd.DataFrame, model_family_categories=None) -> pd.DataFrame:
    """Encode categoricals the same way as preprocess.py.

    Args:
        model_family_categories: Fixed category list from training. When provided,
            ensures Model_Family codes match the training vocabulary.
    """
    data = data.copy()
    if "Region" in data.columns:
        data["Region"] = (
            pd.Categorical(data["Region"].astype(str), categories=REGION_CATEGORIES, ordered=True)
            .codes.astype("int64")
        )
    if "Model_Family" in data.columns:
        if model_family_categories is not None:
            data["Model_Family"] = (
                pd.Categorical(data["Model_Family"].astype(str), categories=model_family_categories)
                .codes.astype("int64")
            )
        else:
            data["Model_Family"] = data["Model_Family"].astype("category").cat.codes.astype("int64")
    return data


# ── Step 1: Region summary ────────────────────────────────────────────────

def print_region_summary(cache: pd.DataFrame, region: str, model_families: list = None):
    """Print available models, families, and scenarios for a region."""
    region_data = cache[cache["Region"] == region]
    if region_data.empty:
        logger.error("No data found for region: %s", region)
        logger.info("Available regions: %s", sorted(cache["Region"].unique()))
        return

    if model_families:
        region_data = region_data[region_data["Model_Family"].isin(model_families)]
        if region_data.empty:
            logger.error("No data for families %s in region '%s'", model_families, region)
            all_fams = sorted(cache[cache["Region"] == region]["Model_Family"].unique())
            logger.info("Available families in %s: %s", region, all_fams)
            return

    print(f"\n{'=' * 70}")
    print(f"  Region: {region}")
    if model_families:
        print(f"  Model Families: {model_families}")
    print(f"  Total rows: {len(region_data)}")
    print(f"{'=' * 70}")

    # Models and families
    model_map = region_data[["Model_Family", "Model"]].drop_duplicates().sort_values(["Model_Family", "Model"])
    print(f"\n  Model families: {model_map['Model_Family'].nunique()}")
    print(f"  Models: {len(model_map)}")
    for fam, grp in model_map.groupby("Model_Family"):
        models = list(grp["Model"])
        print(f"    {fam}: {models}")

    # Scenarios
    scenarios = region_data[["Scenario", "Scenario_Category"]].drop_duplicates().sort_values(["Scenario_Category", "Scenario"])
    print(f"\n  Scenarios: {len(scenarios)}")
    for cat, grp in scenarios.groupby("Scenario_Category"):
        scens = list(grp["Scenario"])
        print(f"    {cat} ({len(scens)}): {scens}")

    # Cross-reference: which models ran which scenarios
    combos = region_data[["Model", "Scenario"]].drop_duplicates()
    print(f"\n  (Model, Scenario) combinations: {len(combos)}")
    print(f"{'=' * 70}\n")


# ── Step 2: Build synthetic data ──────────────────────────────────────────

def find_template_model(cache: pd.DataFrame, region: str, scenarios: list) -> str:
    """Find the model that has data for the most requested scenarios in this region."""
    region_data = cache[
        (cache["Region"] == region) & cache["Scenario"].isin(scenarios)
    ]
    if region_data.empty:
        # No model has these scenarios in this region — pick any model with these scenarios globally
        global_data = cache[cache["Scenario"].isin(scenarios)]
        if global_data.empty:
            raise ValueError(f"No model has data for scenarios {scenarios} in any region.")
        counts = global_data.groupby("Model")["Scenario"].nunique().sort_values(ascending=False)
        best = counts.index[0]
        logger.info("No model has these scenarios in %s. Using %s (global) as template.", region, best)
        return best

    counts = region_data.groupby("Model")["Scenario"].nunique().sort_values(ascending=False)
    best = counts.index[0]
    logger.info("Template model: %s (has %d/%d scenarios in %s)", best, counts.iloc[0], len(scenarios), region)
    return best


def build_model_feature_lookup(
    cache: pd.DataFrame,
    model_map: pd.DataFrame,
    model_specific_cols: list,
    region: str,
) -> dict:
    """Build per-model, per-year median of model-specific features.

    For each model:
      1. If data exists in the target region, use median across scenarios per year.
      2. Otherwise, use median across all regions per year.
      3. Interpolate to fill year gaps, extrapolate at edges.
    Returns {model_name: DataFrame indexed by Year with model_specific_cols}.
    """
    lookup = {}
    for _, row in model_map.iterrows():
        model_name = row["Model"]

        # Try target region first
        region_data = cache[(cache["Model"] == model_name) & (cache["Region"] == region)]
        if len(region_data) > 0 and region_data[model_specific_cols].notna().any().any():
            medians = region_data.groupby("Year")[model_specific_cols].median()
        else:
            # Fall back to all regions
            all_data = cache[cache["Model"] == model_name]
            if len(all_data) == 0:
                continue
            medians = all_data.groupby("Year")[model_specific_cols].median()

        # Interpolate gaps, extrapolate edges (forward/backward fill)
        medians.index = pd.to_numeric(medians.index, errors="coerce")
        medians = medians.sort_index()
        medians = medians.interpolate(method="index", limit_direction="both")
        medians = medians.ffill().bfill()

        lookup[model_name] = medians

    logger.info("Built model feature lookup for %d/%d models", len(lookup), len(model_map))
    return lookup


MIN_FAMILY_TRAINING_ROWS = 200


def build_synthetic_data(
    cache: pd.DataFrame,
    region: str,
    scenarios: list,
    template_model: str,
    known_families: list = None,
) -> pd.DataFrame:
    """Build input rows for every (Model, Scenario, Region) combination.

    For models with actual data for these scenarios in this region, their real
    rows are kept. For all others, synthetic rows are constructed from the
    template model's scenario data + each model's own technology features.

    Families with fewer than MIN_FAMILY_TRAINING_ROWS rows in the cache are
    excluded — their embeddings have too little signal to generalize.

    Args:
        known_families: If provided, restrict to families the model was trained on.
    """
    # Get template rows
    template_mask = (
        (cache["Model"] == template_model)
        & (cache["Region"] == region)
        & cache["Scenario"].isin(scenarios)
    )
    template = cache[template_mask].copy()

    # If template model doesn't have data in this region, try globally
    if template.empty:
        template_mask = (
            (cache["Model"] == template_model)
            & cache["Scenario"].isin(scenarios)
        )
        template = cache[template_mask].copy()
        if template.empty:
            raise ValueError(f"No data for template model {template_model} in scenarios {scenarios}")
        # Use the first available region's data and override Region
        first_region = template["Region"].iloc[0]
        logger.warning("Template model %s has no data in %s. Using %s data as base.", template_model, region, first_region)
        template["Region"] = region

    template_scenarios = sorted(template["Scenario"].unique())
    logger.info("Template: %s with %d rows across %d scenarios: %s",
                template_model, len(template), len(template_scenarios), template_scenarios)

    # One representative Model per family — the embedding only distinguishes families,
    # so multiple variants would produce identical predictions.
    all_models = cache[["Model", "Model_Family"]].drop_duplicates().sort_values(["Model_Family", "Model"])
    if known_families is not None:
        all_models = all_models[all_models["Model_Family"].isin(known_families)]

    # Exclude families with too few training rows for a reliable embedding
    family_counts = cache.groupby("Model_Family").size()
    small_families = set(family_counts[family_counts < MIN_FAMILY_TRAINING_ROWS].index)
    if small_families:
        all_models = all_models[~all_models["Model_Family"].isin(small_families)]
        logger.info("Excluded %d families with <%d training rows: %s",
                     len(small_families), MIN_FAMILY_TRAINING_ROWS, sorted(small_families))

    model_map = all_models.groupby("Model_Family").first().reset_index()
    logger.info("Representative models: %d families (one Model per family)", len(model_map))

    # Identify model-specific feature columns
    model_specific_cols = [
        c for c in template.columns
        if c not in (
            INDEX_COLUMNS
            + ["Model_Family", "Scenario_Category", "Region_Scale", "Ssp_family", "Year"]
            + SCENARIO_FEATURES
            + OUTPUT_VARIABLES
        )
    ]

    # Build feature lookup
    feature_lookup = build_model_feature_lookup(cache, model_map, model_specific_cols, region)

    # Assemble rows
    synthetic_rows = []
    for _, row in model_map.iterrows():
        target_model = row["Model"]
        target_family = row["Model_Family"]

        # Check for actual data in this region for these scenarios
        actual = cache[
            (cache["Model"] == target_model)
            & (cache["Region"] == region)
            & cache["Scenario"].isin(scenarios)
        ]
        if len(actual) > 0:
            for scenario in scenarios:
                s_data = actual[actual["Scenario"] == scenario].copy()
                if not s_data.empty:
                    synthetic_rows.append(s_data)
            # Only create synthetic for missing scenarios
            existing_scens = set(actual["Scenario"].unique())
            scenarios_to_create = [s for s in scenarios if s not in existing_scens]
            if not scenarios_to_create:
                continue
        else:
            scenarios_to_create = list(scenarios)

        for scenario in scenarios_to_create:
            s_template = template[template["Scenario"] == scenario].copy()
            if s_template.empty:
                continue

            s_template["Model"] = target_model
            s_template["Model_Family"] = target_family

            # Fill model-specific features from lookup
            if target_model in feature_lookup:
                lookup = feature_lookup[target_model]
                for _, srow in s_template.iterrows():
                    year = srow["Year"]
                    if year in lookup.index:
                        for col in model_specific_cols:
                            if col in lookup.columns:
                                s_template.loc[srow.name, col] = lookup.loc[year, col]

            # Targets will be predicted
            s_template[OUTPUT_VARIABLES] = np.nan
            synthetic_rows.append(s_template)

    result = pd.concat(synthetic_rows, ignore_index=True)
    n_filled = result[model_specific_cols].notna().mean().mean() * 100
    logger.info(
        "Dataset: %d rows, %d (Model, Scenario) combos, model-specific fill: %.1f%%",
        len(result), result.groupby(["Model", "Scenario"]).ngroups, n_filled,
    )
    return result


# ── Preprocessing ─────────────────────────────────────────────────────────

def add_lag_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add lag columns and drop rows without full lag history."""
    data = data.sort_values(INDEX_COLUMNS + ["Year"]).copy()

    for lag in range(1, N_LAG_FEATURES + 1):
        shifted = data.groupby(INDEX_COLUMNS, sort=False)[OUTPUT_VARIABLES].shift(lag)
        for col in OUTPUT_VARIABLES:
            prefix = "prev_" if lag == 1 else f"prev{lag}_"
            data[f"{prefix}{col}"] = shifted[col]

    row_num = data.groupby(INDEX_COLUMNS, sort=False).cumcount()
    data = data[row_num >= N_LAG_FEATURES].reset_index(drop=True)
    return data


def prepare_for_xgb(data: pd.DataFrame, features: list, targets: list, x_scaler):
    """Encode, scale, and format data for XGB inference."""
    X = data[features].copy()
    X = encode_categorical_columns(X)

    X_scaled = pd.DataFrame(x_scaler.transform(X), columns=X.columns, index=X.index)

    index_cols = [c for c in NON_FEATURE_COLUMNS if c in data.columns and c not in features]
    X_test_with_index = pd.concat(
        [X_scaled.reset_index(drop=True), data[index_cols].reset_index(drop=True)],
        axis=1,
    )

    y_test = data[targets].values.copy()
    return X_test_with_index, y_test, data


# ── Inference dispatch ─────────────────────────────────────────────────────

def run_inference(model_type: str, artifacts: dict, synthetic: pd.DataFrame,
                  features: list, targets: list):
    """Run inference, dispatching by model type.

    Returns (results_df, preds_original) where results_df has metadata +
    pred_* columns and preds_original is the raw numpy array.
    """
    if model_type == "xgb":
        return _infer_xgb(artifacts, synthetic, features, targets)
    if model_type == "lstm":
        return _infer_lstm(artifacts, synthetic, features, targets)
    if model_type == "tft":
        return _infer_tft(artifacts, synthetic, features, targets)
    raise ValueError(f"Unsupported model type: {model_type}")


def _infer_xgb(artifacts, synthetic, features, targets):
    X_test, y_test, test_data = prepare_for_xgb(
        synthetic, features, targets, artifacts["x_scaler"],
    )
    n_groups = X_test.groupby(INDEX_COLUMNS).ngroups
    logger.info("Running XGB autoregressive inference on %d groups...", n_groups)

    preds_scaled = test_xgb_autoregressively(
        X_test, y_test,
        model=artifacts["model"],
        y_scaler=artifacts["y_scaler"],
        x_scaler=artifacts["x_scaler"],
        disable_progress=True,
    )
    preds_original = artifacts["y_scaler"].inverse_transform(preds_scaled)
    return _build_results(test_data, preds_original, targets), preds_original


def _infer_lstm(artifacts, synthetic, features, targets):
    import torch
    from torch.utils.data import DataLoader
    from lightning.pytorch import Trainer
    from src.trainers.lstm_trainer import LSTMDataset
    from src.data.preprocess import add_missingness_indicators, impute_with_train_medians

    data = synthetic.copy()

    # Add missingness indicators (LSTM uses these, XGB doesn't)
    data, features_with_missing = add_missingness_indicators(data, features)

    # Encode categoricals (use training vocab so codes match embeddings)
    data = encode_categorical_columns(data, model_family_categories=artifacts.get("model_family_categories"))

    # Impute NaN with scaler mean (≈ train median) so scaled values are ~0 (neutral).
    # During training, NaN was imputed with train medians before fitting scaler_X.
    scaler_X = artifacts["scaler_X"]
    scaler_feature_names = list(scaler_X.feature_names_in_) if hasattr(scaler_X, "feature_names_in_") else []
    for col in features_with_missing:
        if col in data.columns and col not in ("Region", "Model_Family"):
            if col in scaler_feature_names:
                fill_val = scaler_X.mean_[scaler_feature_names.index(col)]
            else:
                fill_val = 0.0
            data[col] = data[col].fillna(fill_val)

    # Add Step and DeltaYears if not present
    data["Year"] = data["Year"].astype(int)
    if "Step" not in data.columns:
        data = data.sort_values(INDEX_COLUMNS + ["Year"])
        data["Step"] = data.groupby(INDEX_COLUMNS).cumcount().astype("int64")
    if "DeltaYears" not in data.columns:
        data["DeltaYears"] = data.groupby(INDEX_COLUMNS)["Year"].diff().fillna(0).astype(int)

    model = artifacts["model"]
    scaler_X = artifacts["scaler_X"]
    scaler_y = artifacts["scaler_y"]
    seq_len = artifacts["sequence_length"]
    target_offset = artifacts["target_offset"]
    config = artifacts["config"]

    # Use the features the LSTM was trained with
    lstm_features = artifacts["features"]
    categorical_features = artifacts.get("categorical_features", [])

    # Ensure all expected feature columns exist
    all_features = lstm_features + [c for c in categorical_features if c not in lstm_features]
    for col in all_features:
        if col not in data.columns:
            data[col] = 0.0

    dataset = LSTMDataset(
        data, all_features, targets,
        sequence_length=seq_len,
        target_offset=target_offset,
        scaler_X=scaler_X, scaler_y=scaler_y,
        fit_scalers=False,
        categorical_features=categorical_features,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
                        num_workers=1, persistent_workers=False)

    trainer = Trainer(
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        enable_progress_bar=False, logger=False,
    )
    raw_preds = trainer.predict(model, loader)
    preds_array = np.vstack([p.cpu().numpy() for p in raw_preds])
    if preds_array.ndim == 3 and preds_array.shape[1] == 1:
        preds_array = preds_array.squeeze(axis=1)

    preds_unscaled = scaler_y.inverse_transform(preds_array)

    # Align predictions with data rows (sequence alignment)
    aligned = np.full((len(data), len(targets)), np.nan)
    pred_idx = 0
    for _, group_data in data.groupby(INDEX_COLUMNS):
        group_size = len(group_data)
        group_indices = group_data.index
        num_seqs = max(0, group_size - (seq_len + target_offset) + 1)
        for i in range(num_seqs):
            if pred_idx < len(preds_unscaled):
                row_idx = data.index.get_loc(group_indices[i + seq_len - 1 + target_offset])
                aligned[row_idx] = preds_unscaled[pred_idx]
                pred_idx += 1

    logger.info("LSTM predictions aligned: %d/%d rows have values",
                (~np.isnan(aligned[:, 0])).sum(), len(data))

    return _build_results(data, aligned, targets), aligned


def _infer_tft(artifacts, synthetic, features, targets):
    import torch
    from src.trainers.tft_dataset import from_train_template
    from src.data.preprocess import add_missingness_indicators
    from configs.models.tft import TFTTrainerConfig

    data = synthetic.copy()

    # Add missingness indicators
    data, features_with_missing = add_missingness_indicators(data, features)

    # TFT handles categorical encoding internally via NaNLabelEncoder in the
    # saved dataset template — do NOT pre-encode to integer codes here.
    # Just ensure categorical columns are strings so the template's encoders
    # can map them.
    for col in ("Region", "Model_Family"):
        if col in data.columns:
            data[col] = data[col].astype(str)

    # Impute NaN — TFT uses per-group normalization so 0.0 is less harmful here
    for col in features_with_missing:
        if col in data.columns and col not in ("Region", "Model_Family"):
            data[col] = data[col].fillna(0.0)

    # TFT requires non-NaN targets even in predict mode (they're not used as
    # inputs but TimeSeriesDataSet validates them). Fill with 0.0.
    for col in targets:
        if col in data.columns:
            data[col] = data[col].fillna(0.0)

    # Add Step and DeltaYears
    data["Year"] = data["Year"].astype(int)
    if "Step" not in data.columns:
        data = data.sort_values(INDEX_COLUMNS + ["Year"])
        data["Step"] = data.groupby(INDEX_COLUMNS).cumcount().astype("int64")
    if "DeltaYears" not in data.columns:
        data["DeltaYears"] = data.groupby(INDEX_COLUMNS)["Year"].diff().fillna(0).astype(int)

    model = artifacts["model"]
    template = artifacts["dataset_template"]

    # Drop rows with Model names not in the training vocabulary — TFT's
    # NaNLabelEncoder rejects unknown categories, and patching it after the
    # fact breaks the code ↔ index alignment during inverse_transform.
    model_enc = template._categorical_encoders.get("__group_id__Model")
    if model_enc is not None and hasattr(model_enc, "classes_"):
        known_models = set(model_enc.classes_.keys())
        unknown_mask = ~data["Model"].isin(known_models)
        if unknown_mask.any():
            dropped = data.loc[unknown_mask, "Model"].unique().tolist()
            logger.warning(
                "Dropping %d rows with models unseen during training: %s",
                unknown_mask.sum(), dropped,
            )
            data = data[~unknown_mask].reset_index(drop=True)

    try:
        test_dataset = from_train_template(template, data, mode="predict")
    except Exception as e:
        raise RuntimeError(f"Failed to build TFT test dataset from template: {e}")

    trainer_cfg = TFTTrainerConfig()
    test_loader = test_dataset.to_dataloader(
        train=False, batch_size=trainer_cfg.batch_size, num_workers=1,
    )

    # Force single-GPU inference via trainer_kwargs — without this,
    # model.predict() creates a Trainer that auto-detects all GPUs and
    # launches DDP, duplicating the script and failing on tiny datasets.
    logger.info("Running TFT prediction...")
    returns = model.predict(
        test_loader, return_index=True,
        trainer_kwargs={"accelerator": "gpu", "devices": 1},
    )

    from pytorch_forecasting.models.base._base_model import Prediction as _PFPrediction
    outputs = returns.output
    if isinstance(outputs, list):
        preds_tensor = outputs[0] if len(outputs) == 1 else torch.stack(outputs, dim=-1)
    elif torch.is_tensor(outputs):
        preds_tensor = outputs
    else:
        raise RuntimeError(f"Unsupported Prediction.output type: {type(outputs)}")

    preds_flat = preds_tensor.detach().cpu().numpy()
    if preds_flat.ndim == 3:
        n, pred_len, out_size = preds_flat.shape
        preds_flat = preds_flat.reshape(n * pred_len, out_size)

    # Align with data — TFT predictions are for the forecast horizon only
    # For now, return predictions aligned to the last rows of each group
    aligned = np.full((len(data), len(targets)), np.nan)
    aligned[-len(preds_flat):] = preds_flat[:len(aligned)]

    logger.info("TFT predictions: %d rows", len(preds_flat))
    return _build_results(data, aligned, targets), aligned


def _build_results(test_data, preds_original, targets):
    """Build results DataFrame from test_data metadata + predictions."""
    results = test_data[["Model", "Model_Family", "Scenario", "Scenario_Category", "Region", "Year"]].copy()
    results["Year"] = results["Year"].astype(int)
    for i, t in enumerate(targets):
        results[f"pred_{t}"] = preds_original[:, i]
    return results


# ── Plotting ──────────────────────────────────────────────────────────────
# Style matches src/visualization/trajectories.py (same constants, formatters,
# axis layout) but handles the inference case where some groups lack ground truth.

def plot_inference_trajectories(
    results: pd.DataFrame,
    ground_truth: pd.DataFrame,
    targets: list,
    output_path: str,
    title: str = "",
):
    """Plot a 3x3 trajectory grid for inference results.

    One line per (Model_Family, Scenario) group:
      - Has IAM answer: solid actual + dashed pred.
      - Synthetic only: dashed pred line.

    One color per family, single figure legend.
    """
    fig, axes = plt.subplots(PLOT_GRID_ROWS, PLOT_GRID_COLS, figsize=TRAJECTORY_GRID_FIGSIZE)
    plt.rcParams.update({"font.size": PLOT_FONT_SIZE})
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    # Build Model → Model_Family map from results
    family_map = results[["Model", "Model_Family"]].drop_duplicates().set_index("Model")["Model_Family"]

    # Ground truth keys (by Model)
    gt_keys = set()
    if not ground_truth.empty:
        gt_keys = set(ground_truth.groupby(["Model", "Scenario"]).groups.keys())

    # Color by family
    groups = sorted(results.groupby(["Model", "Scenario"]).groups.keys())
    families = sorted(results["Model_Family"].unique())
    prop_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    family_colors = {f: prop_colors[i % len(prop_colors)] for i, f in enumerate(families)}

    for idx, ax in enumerate(axes.flatten()):
        if idx >= len(targets):
            ax.set_visible(False)
            continue
        target = targets[idx]

        for model, scenario in groups:
            family = family_map.get(model, model)
            color = family_colors[family]
            has_gt = (model, scenario) in gt_keys

            pred_rows = results[
                (results["Model"] == model) & (results["Scenario"] == scenario)
            ].sort_values("Year")
            if pred_rows.empty:
                continue

            years = pred_rows["Year"].values
            pred_vals = pred_rows[f"pred_{target}"].values

            if has_gt:
                gt_rows = ground_truth[
                    (ground_truth["Model"] == model) & (ground_truth["Scenario"] == scenario)
                ].sort_values("Year")
                gt_vals = gt_rows[target].values

                ax.plot(years, gt_vals, color=color, linestyle="-", alpha=0.7, linewidth=1.5,
                        label=f"{model} (actual)" if idx == 0 else None)
                ax.plot(years, pred_vals, color=color, linestyle="--", alpha=0.7, linewidth=1.5,
                        label=f"{family} (pred)" if idx == 0 else None)
            else:
                ax.plot(years, pred_vals, color=color, linestyle="--", alpha=0.5, linewidth=1.5,
                        label=f"{family} (pred)" if idx == 0 else None)

        # Axis styling
        unit = OUTPUT_UNITS[idx] if idx < len(OUTPUT_UNITS) else ""
        ax.set_ylabel(f"{target} ({unit})", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_xlabel("Year", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", which="major", labelsize=TICK_LABELSIZE)
        ax.yaxis.set_major_formatter(FuncFormatter(format_large_numbers))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=Y_AXIS_NBINS_TRAJECTORY))

    # Single figure legend
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        fig.legend(
            by_label.values(), by_label.keys(),
            loc="lower center", ncol=min(3, len(by_label)),
            fontsize=LEGEND_FONTSIZE, bbox_to_anchor=(0.5, -0.03),
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved trajectory plot: %s", output_path)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run ML-IAM inference for configured scenarios in a single region.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-id", required=True, help="Trained model run ID (e.g. xgb_75)")
    parser.add_argument("--region", required=True, help="Region code (e.g. R10REST_ASIA, KOR)")
    parser.add_argument("--model-family", nargs="*", default=None,
                        help="Filter to specific model families (e.g. MESSAGEix REMIND GCAM)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Filter to specific models (e.g. 'GCAM 5.3' 'REMIND-MAgPIE 2.1-4.2')")
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="Scenario name(s) to run inference for. If omitted, prints region summary only.")
    parser.add_argument("--scenario-categories", nargs="*", default=None,
                        help="Filter to specific scenario categories (e.g. C1 C2 C3)")
    parser.add_argument("--template-model", default=None,
                        help="Model to use as scenario template (default: auto-detect best)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: auto)")
    args = parser.parse_args()

    run_id = args.run_id
    region = args.region

    # Load data
    logger.info("Loading artifacts from run: %s", run_id)
    model_type, artifacts = load_artifacts(run_id)
    features = artifacts["features"]
    targets = artifacts["targets"]
    logger.info("Model type: %s", model_type)

    store = RunStore(run_id)
    cache = store.load_processed_data()
    logger.info("Loaded processed data: %d rows", len(cache))

    if model_type == "lstm" and not artifacts.get("model_family_categories"):
        raise RuntimeError(
            f"Run {run_id} was trained before Model_Family category vocab was saved. "
            "Retrain to persist lstm_model_family_categories in train_meta."
        )

    # ── Step 1: Print region summary ──
    print_region_summary(cache, region, model_families=args.model_family)

    if not args.scenarios and not args.scenario_categories:
        print("No --scenarios or --scenario-categories specified. Add them to run inference.")
        print("Example:")
        region_scens = cache[cache["Region"] == region]["Scenario"].unique()
        for s in sorted(region_scens)[:5]:
            print(f'  --scenarios "{s}"')
        return

    # ── Step 2: Resolve filters ──
    # Start with all data in this region
    region_data = cache[cache["Region"] == region]

    # Resolve scenarios from --scenarios and/or --scenario-categories
    scenarios = set()
    if args.scenarios:
        all_scenarios = set(cache["Scenario"].unique())
        invalid = [s for s in args.scenarios if s not in all_scenarios]
        if invalid:
            logger.error("Scenarios not found in dataset: %s", invalid)
            return
        scenarios.update(args.scenarios)

    if args.scenario_categories:
        cat_scens = region_data[region_data["Scenario_Category"].isin(args.scenario_categories)]["Scenario"].unique()
        if len(cat_scens) == 0:
            logger.error("No scenarios found for categories %s in %s", args.scenario_categories, region)
            return
        scenarios.update(cat_scens)
        logger.info("Scenario categories %s resolved to %d scenarios", args.scenario_categories, len(cat_scens))

    scenarios = sorted(scenarios)

    # Filter models
    model_map = cache[["Model", "Model_Family"]].drop_duplicates()
    if args.model_family:
        model_map = model_map[model_map["Model_Family"].isin(args.model_family)]
    if args.models:
        model_map = model_map[model_map["Model"].isin(args.models)]

    if model_map.empty:
        logger.error("No models match the filters.")
        return

    filtered_models = sorted(model_map["Model"].unique())
    logger.info("Filters: %d scenarios, %d models (%d families)",
                len(scenarios), len(filtered_models), model_map["Model_Family"].nunique())

    # Find or use template model
    template_model = args.template_model or find_template_model(cache, region, scenarios)

    # Build descriptive output dir name from configs
    safe_region = region.replace("+", "plus")
    name_parts = [safe_region]
    if args.model_family:
        name_parts.append("_".join(args.model_family))
    if args.models:
        name_parts.append("_".join(m.replace(" ", "").replace("/", "-") for m in args.models))
    if args.scenario_categories:
        name_parts.append("_".join(args.scenario_categories))
    if args.scenarios:
        # Use short hashes to keep dir name reasonable
        for s in args.scenarios:
            short = s.replace(" ", "_")[:30]
            name_parts.append(short)
    dir_name = "inference__" + "__".join(name_parts)
    output_dir = args.output_dir or os.path.join(get_run_root(run_id), "experiments", dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 3: Build dataset & run inference ──
    synthetic = build_synthetic_data(
        cache, region, scenarios, template_model,
        known_families=artifacts.get("model_family_categories"),
    )

    # Apply model filter — keep only requested families
    filtered_families = sorted(model_map["Model_Family"].unique())
    synthetic = synthetic[synthetic["Model_Family"].isin(filtered_families)].reset_index(drop=True)
    logger.info("After model filter: %d rows (%d families)", len(synthetic), len(filtered_families))

    # Add lag features
    synthetic = add_lag_features(synthetic)

    # Extract ground truth (models with real target values)
    gt_mask = synthetic[targets[0]].notna()
    ground_truth = synthetic[gt_mask].copy()
    ground_truth["Year"] = ground_truth["Year"].astype(int)
    gt_models = sorted(ground_truth["Model"].unique())
    logger.info("Ground truth models: %s", gt_models)

    # Run inference — dispatch by model type
    results, preds_original = run_inference(model_type, artifacts, synthetic, features, targets)

    # Decode integer-encoded categoricals back to string labels for output/plots
    mf_cats = artifacts.get("model_family_categories")
    if mf_cats and pd.api.types.is_integer_dtype(results["Model_Family"]):
        code_to_name = {i: name for i, name in enumerate(mf_cats)}
        results["Model_Family"] = results["Model_Family"].map(code_to_name)
    if mf_cats and not ground_truth.empty and pd.api.types.is_integer_dtype(ground_truth["Model_Family"]):
        ground_truth["Model_Family"] = ground_truth["Model_Family"].map(code_to_name)
    if pd.api.types.is_integer_dtype(results["Region"]):
        results["Region"] = results["Region"].map({i: r for i, r in enumerate(REGION_CATEGORIES)})
    if not ground_truth.empty and pd.api.types.is_integer_dtype(ground_truth["Region"]):
        ground_truth["Region"] = ground_truth["Region"].map({i: r for i, r in enumerate(REGION_CATEGORIES)})

    # Save predictions
    results_path = os.path.join(output_dir, "predictions.csv")
    results.to_csv(results_path, index=False)
    logger.info("Saved predictions: %s (%d rows)", results_path, len(results))

    # ── Step 4: Plot ──
    # Build title suffix from filters
    filter_parts = []
    if args.model_family:
        filter_parts.append(f"Families: {', '.join(args.model_family)}")
    if args.models:
        filter_parts.append(f"Models: {', '.join(args.models)}")
    if args.scenario_categories:
        filter_parts.append(f"Categories: {', '.join(args.scenario_categories)}")
    title_suffix = " | ".join(filter_parts)

    # ── Step 4: Plot ──
    title_parts = [region]
    if title_suffix:
        title_parts.append(title_suffix)
    title = " — ".join(title_parts)

    # Combined trajectory plot
    traj_path = os.path.join(output_dir, "trajectories.png")
    plot_inference_trajectories(results, ground_truth, targets, traj_path, title=title)

    logger.info("Done. All outputs in: %s", output_dir)


if __name__ == "__main__":
    main()
