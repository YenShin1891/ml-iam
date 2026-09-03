"""TFT dataset building and management functions."""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Any

import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet

from src.utils.utils import get_run_root
# TFTDatasetConfig imported locally in functions to match original pattern
from configs.data import CATEGORICAL_COLUMNS, INDEX_COLUMNS, MAX_CONTEXT_LENGTH


from pytorch_forecasting.data.encoders import NaNLabelEncoder


def _ordered_categorical_cols(features: List[str]) -> List[str]:
    """Deterministic order: group ids + static categoricals + indicator columns."""
    static_cols = list(INDEX_COLUMNS) + [c for c in CATEGORICAL_COLUMNS if c in features]
    indicator_cols = [f for f in features if f.endswith("_is_missing")]
    ordered = list(dict.fromkeys(static_cols + indicator_cols))
    return ordered


def _build_union_encoders(session_state: Dict, categorical_cols: List[str], add_nan: bool = False) -> Dict[str, Any]:
    """Fit NaNLabelEncoder with a closed vocabulary aggregated across splits."""
    dfs = [session_state.get("train_data"), session_state.get("val_data"), session_state.get("test_data")]
    df_all = pd.concat([df for df in dfs if df is not None], axis=0, ignore_index=True)
    encoders: Dict[str, Any] = {}
    # ensure deterministic iteration order
    for col in categorical_cols:
        if col in df_all.columns:
            # fillna before astype: str(NaN) is "nan", not the token below.
            s_raw = df_all[col].fillna("__NA__").astype(str)
            # Explicit, deterministic category order
            categories = sorted(pd.unique(s_raw))
            s = pd.Series(pd.Categorical(s_raw, categories=categories, ordered=True))
        else:
            # if column is entirely missing in some split, still create a closed-vocab encoder
            s = pd.Series(pd.Categorical(["__NA__"], categories=["__NA__"], ordered=True))
        enc = NaNLabelEncoder(add_nan=add_nan)
        enc.fit(s)
        encoders[col] = enc
    return encoders


def build_datasets(session_state: Dict) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """Build train/val TimeSeriesDataSet objects using shared template logic (encoders handle categoricals)."""
    val_data = session_state["val_data"]
    train_dataset, config = create_train_dataset(session_state)
    session_state["tft_target_offset"] = config.target_offset
    session_state["tft_min_encoder_length"] = config.effective_min_encoder_length
    session_state["tft_max_encoder_length"] = config.effective_max_encoder_length
    session_state["tft_time_idx_column"] = config.time_idx
    val_dataset = from_train_template(train_dataset, val_data, mode="eval")
    return train_dataset, val_dataset


def compute_target_scale_floors(
    train_data: pd.DataFrame,
    targets: List[str],
    fraction: float = 0.01,
) -> Dict[str, float]:
    """Compute per-target minimum scale floors from training data.

    Returns ``fraction`` of the global standard deviation per target.  This
    prevents the per-sample EncoderNormalizer from using a near-zero scale
    when the encoder window has little or no variance (e.g. variables that
    start at zero for many regions).
    """
    import numpy as np

    floors: Dict[str, float] = {}
    for target in targets:
        if target in train_data.columns:
            global_std = float(
                pd.to_numeric(train_data[target], errors="coerce").std(skipna=True)
            )
            floors[target] = max(global_std * fraction, np.finfo(np.float32).eps)
        else:
            floors[target] = np.finfo(np.float32).eps
    logging.info("Target scale floors (%.1f%% of global σ): %s", fraction * 100, floors)
    return floors


def drop_underlength_groups(
    data: pd.DataFrame,
    group_ids: List[str],
    time_idx: str,
    required_length: int,
) -> pd.DataFrame:
    """Drop groups with fewer than *required_length* steps, in one log line.

    pytorch_forecasting indexes such groups out itself, but announces it with
    a warning naming every dropped group on every dataset build.
    """
    sizes = data.groupby(list(group_ids), observed=True, sort=False)[time_idx].size()
    short = sizes[sizes < required_length]
    if short.empty:
        return data
    if len(short) == len(sizes):
        raise ValueError(
            f"Every group has fewer than {required_length} steps; no "
            "encoder+prediction window fits. Check the encoder/prediction "
            "lengths against the data."
        )

    logging.info(
        "%d of %d groups have fewer than %d steps and cannot fill one "
        "encoder+prediction window; dropping them.",
        len(short), len(sizes), required_length,
    )
    keep = ~data.set_index(list(group_ids)).index.isin(short.index)
    return data.loc[keep]


def create_train_dataset(session_state: Dict) -> Tuple[TimeSeriesDataSet, Any]:
    """Create training dataset with configuration, coercing categorical-like columns first."""
    from configs.models.tft import TFTDatasetConfig

    train_data = session_state["train_data"]
    features = session_state["features"]
    targets = session_state["targets"]

    config = TFTDatasetConfig()
    target_offset = session_state.get("tft_target_offset")
    if target_offset is not None:
        config.target_offset = int(target_offset)

    normalizer_mode = session_state.get("tft_target_normalizer_mode")
    if normalizer_mode is not None:
        config.target_normalizer_mode = normalizer_mode

    # Compute per-target scale floors from training data (used by encoder_floored mode)
    if config.target_normalizer_mode == "encoder_floored":
        config.target_scale_floors = compute_target_scale_floors(
            train_data, targets, fraction=config.scale_floor_fraction,
        )

    # Build union encoders (include group ids to stabilize mapping) then inject
    categorical_cols = _ordered_categorical_cols(features)
    pretrained_encoders = _build_union_encoders(session_state, categorical_cols, add_nan=False)
    config.pretrained_categorical_encoders = pretrained_encoders

    # The threshold uses the longest context under comparison, not this
    # config's, so a short-encoder run trains on exactly the groups a
    # long-encoder run does.
    config.resolve_encoder_lengths()
    train_data = drop_underlength_groups(
        train_data, config.group_ids, config.time_idx,
        MAX_CONTEXT_LENGTH + config.min_prediction_length,
    )

    params = config.build(features, targets, mode="train")

    train_dataset = TimeSeriesDataSet(train_data, **params)
    return train_dataset, config


# Bumped when the on-disk layout changes; older files still load.
_TEMPLATE_FORMAT = 2


@dataclass
class DatasetTemplate:
    """The fitted configuration of a training TimeSeriesDataSet.

    ``TimeSeriesDataSet.from_dataset`` is only ``from_parameters(
    dataset.get_parameters(), data)``, so the parameters — which carry the
    fitted encoders, scalers and target normalizer — are all a later phase
    needs.  Persisting them instead of the dataset object leaves the training
    DataFrame out of the run directory and makes the file far less brittle
    across pytorch_forecasting versions.

    *derived* holds the few attributes pytorch_forecasting computes during
    ``__init__`` (adding ``relative_time_idx``, target scales and so on), which
    cannot be recovered from the constructor arguments alone.
    """

    parameters: Dict[str, Any]
    derived: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet) -> "DatasetTemplate":
        return cls(
            parameters=dataset.get_parameters(),
            derived={
                "reals": list(dataset.reals),
                "categoricals": list(dataset.categoricals),
                "flat_categoricals": list(dataset.flat_categoricals),
                "target_names": list(dataset.target_names),
            },
        )

    # Attributes callers read off the training dataset.
    @property
    def time_idx(self) -> str:
        return self.parameters["time_idx"]

    @property
    def group_ids(self) -> List[str]:
        return list(self.parameters["group_ids"])

    @property
    def target(self):
        return self.parameters["target"]

    @property
    def max_encoder_length(self) -> int:
        return self.parameters["max_encoder_length"]

    @property
    def min_encoder_length(self) -> int:
        return self.parameters["min_encoder_length"]

    @property
    def max_prediction_length(self) -> int:
        return self.parameters["max_prediction_length"]

    @property
    def min_prediction_length(self) -> int:
        return self.parameters["min_prediction_length"]

    @property
    def categorical_encoders(self) -> Dict[str, Any]:
        """The fitted NaNLabelEncoders, keyed as pytorch_forecasting keys them."""
        return self.parameters.get("categorical_encoders") or {}

    @property
    def reals(self) -> List[str]:
        return list(self.derived.get("reals", []))

    @property
    def categoricals(self) -> List[str]:
        return list(self.derived.get("categoricals", []))

    @property
    def flat_categoricals(self) -> List[str]:
        return list(self.derived.get("flat_categoricals", []))

    @property
    def target_names(self) -> List[str]:
        return list(self.derived.get("target_names", []))

    def build(self, data: pd.DataFrame, mode: str = "eval") -> TimeSeriesDataSet:
        """Rebuild a dataset over *data* with this template's configuration."""
        return TimeSeriesDataSet.from_parameters(
            self.parameters,
            data,
            stop_randomization=(mode in {"eval", "test", "predict"}),
            predict=(mode == "predict"),
        )


def from_train_template(
    train_dataset,
    data: pd.DataFrame,
    mode: str = "eval",
) -> TimeSeriesDataSet:
    """Create a dataset from a training template.

    Accepts a live ``TimeSeriesDataSet`` or a loaded :class:`DatasetTemplate`.
    """
    # predict mode raises min_prediction_length to max_prediction_length
    # (one full-horizon prediction per group), so a group needs that much
    # more room to survive.
    prediction_length = (
        train_dataset.max_prediction_length if mode == "predict"
        else train_dataset.min_prediction_length
    )
    data = drop_underlength_groups(
        data, train_dataset.group_ids, train_dataset.time_idx,
        # Again the longest context, so every encoder length is scored on the
        # same trajectories rather than on whichever ones its own window fits.
        MAX_CONTEXT_LENGTH + prediction_length,
    )
    if isinstance(train_dataset, DatasetTemplate):
        return train_dataset.build(data, mode=mode)

    return TimeSeriesDataSet.from_dataset(
        train_dataset,
        data,
        stop_randomization=(mode in {"eval", "test", "predict"}),
        predict=(mode == "predict")
    )


def _dataset_template_path(run_id: str) -> str:
    return os.path.join(get_run_root(run_id), "final", "dataset_template.pt")


def save_dataset_template(dataset: TimeSeriesDataSet, run_id: str) -> str:
    """Save the training dataset's configuration for later phases."""
    dataset_tpl_path = _dataset_template_path(run_id)
    os.makedirs(os.path.dirname(dataset_tpl_path), exist_ok=True)

    template = DatasetTemplate.from_dataset(dataset)
    payload = {
        "format": _TEMPLATE_FORMAT,
        "parameters": template.parameters,
        "derived": template.derived,
    }
    try:
        torch.save(payload, dataset_tpl_path)
        logging.info(
            "Saved TFT dataset template to %s (%.1f KiB)",
            dataset_tpl_path, os.path.getsize(dataset_tpl_path) / 1024,
        )
        return dataset_tpl_path
    except Exception as e:
        raise RuntimeError(f"Failed to save dataset template to {dataset_tpl_path}: {e}")


def load_dataset_template(run_id: str) -> DatasetTemplate:
    """Load a run's dataset template, in either the current or legacy layout."""
    dataset_tpl_path = _dataset_template_path(run_id)

    if not os.path.exists(dataset_tpl_path):
        raise FileNotFoundError(
            f"Dataset template not found at {dataset_tpl_path}. Run train_final_tft to generate it."
        )

    loaded = torch.load(dataset_tpl_path, map_location="cpu", weights_only=False)

    if isinstance(loaded, dict) and "parameters" in loaded:
        return DatasetTemplate(
            parameters=loaded["parameters"], derived=loaded.get("derived", {})
        )

    if isinstance(loaded, TimeSeriesDataSet):
        # Runs trained before the template stopped pickling the whole dataset.
        logging.info("Read legacy pickled dataset template from %s", dataset_tpl_path)
        return DatasetTemplate.from_dataset(loaded)

    raise RuntimeError(
        f"Unrecognised dataset template at {dataset_tpl_path}: got {type(loaded)}"
    )
