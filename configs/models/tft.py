from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from configs.data import CATEGORICAL_COLUMNS, INDEX_COLUMNS, MAX_SERIES_LENGTH

"""TFT-specific length configuration.

Original (working) implementation fixed:
  max_encoder_length = 2
  max_prediction_length = MAX_SERIES_LENGTH - 2

During refactor we briefly flipped the ratio (encoder>>pred) which could alter
forecast horizon and degrade metrics. We restore original defaults while
allowing optional overrides via environment variables:
  TFT_MAX_ENCODER_LENGTH, TFT_MAX_PRED_LENGTH

Override rules:
  1. If neither env var is set: use original pattern (2, MAX_SERIES_LENGTH-2).
  2. If only encoder is set: prediction = MAX_SERIES_LENGTH - encoder.
  3. If only prediction is set: encoder = MAX_SERIES_LENGTH - prediction (min 1).
  4. If both are set and sum > MAX_SERIES_LENGTH: reduce prediction first, then encoder if needed.
"""
import os as _os

_enc_env = _os.getenv("TFT_MAX_ENCODER_LENGTH")
_pred_env = _os.getenv("TFT_MAX_PRED_LENGTH")

# Initialize with original defaults
TFT_MAX_ENCODER_LENGTH: int = 2
TFT_MAX_PRED_LENGTH: int = max(1, MAX_SERIES_LENGTH - 2)

if _enc_env is not None:
    try:
        TFT_MAX_ENCODER_LENGTH = max(1, int(_enc_env))
    except ValueError:
        pass
if _pred_env is not None:
    try:
        TFT_MAX_PRED_LENGTH = max(1, int(_pred_env))
    except ValueError:
        pass

# Derive missing side if only one provided
if _enc_env is not None and _pred_env is None:
    TFT_MAX_PRED_LENGTH = max(1, MAX_SERIES_LENGTH - TFT_MAX_ENCODER_LENGTH)
elif _pred_env is not None and _enc_env is None:
    TFT_MAX_ENCODER_LENGTH = max(1, MAX_SERIES_LENGTH - TFT_MAX_PRED_LENGTH)

# Enforce global cap if both explicitly set
if TFT_MAX_ENCODER_LENGTH + TFT_MAX_PRED_LENGTH > MAX_SERIES_LENGTH:
    overflow = TFT_MAX_ENCODER_LENGTH + TFT_MAX_PRED_LENGTH - MAX_SERIES_LENGTH
    reducible = TFT_MAX_PRED_LENGTH - 1
    take = min(overflow, reducible)
    TFT_MAX_PRED_LENGTH -= take
    overflow -= take
    if overflow > 0:
        reducible_enc = TFT_MAX_ENCODER_LENGTH - 1
        take_enc = min(overflow, reducible_enc)
        TFT_MAX_ENCODER_LENGTH -= take_enc
        overflow -= take_enc
    if TFT_MAX_ENCODER_LENGTH + TFT_MAX_PRED_LENGTH > MAX_SERIES_LENGTH:
        TFT_MAX_PRED_LENGTH = max(1, MAX_SERIES_LENGTH - TFT_MAX_ENCODER_LENGTH)



@dataclass
class TFTDatasetConfig:
    """Builder for TimeSeriesDataSet parameters used by TFT.

    Usage:
        params = TFTDatasetConfig().build(features, targets, mode="train"|"eval")
    """

    time_idx: str = "Step"
    group_ids: List[str] = field(default_factory=lambda: INDEX_COLUMNS)
    max_encoder_length: int = TFT_MAX_ENCODER_LENGTH
    min_encoder_length: int = TFT_MAX_ENCODER_LENGTH  # keep encoder fixed like original (2)
    max_prediction_length: int = TFT_MAX_PRED_LENGTH
    min_prediction_length: int = 1
    add_relative_time_idx: bool = True
    add_target_scales: bool = True
    allow_missing_timesteps: bool = False
    pretrained_categorical_encoders: Dict[str, Any] = field(default_factory=dict)

    def build(
        self,
        features: List[str],
        targets: List[str],
        mode: str,
    ) -> Dict[str, Any]:
        # Lazy import to avoid heavy deps at module import time
        from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer

        if isinstance(targets, str):
            targets = [targets]
        time_known = ["Year", "DeltaYears"]
        indicator_cols = [f for f in features if f.endswith("_is_missing")]

        # Unknown real-valued features exclude:
        # - categoricals (handled separately)
        # - known time columns (used for teacher forcing)
        # - indicator categoricals (treated as known categoricals)
        # - the time index itself (must never be a model input)
        unknown_reals = [
            f for f in features
            if f not in (CATEGORICAL_COLUMNS + time_known + indicator_cols)
            and f != self.time_idx
        ]

        # Shared normalizer across targets, grouped by group_ids
        # IMPORTANT: create a distinct GroupNormalizer per target
        target_normalizer = MultiNormalizer([
            GroupNormalizer(groups=self.group_ids) for _ in targets
        ])

        params: Dict[str, Any] = {
            "time_idx": self.time_idx,
            "target": targets,
            "group_ids": self.group_ids,
            "max_encoder_length": self.max_encoder_length,
            "min_encoder_length": self.min_encoder_length,
            "max_prediction_length": self.max_prediction_length,
            "min_prediction_length": self.min_prediction_length,
            "target_normalizer": target_normalizer,
            "time_varying_known_reals": time_known,
            "time_varying_unknown_reals": unknown_reals,
            "static_categoricals": CATEGORICAL_COLUMNS,
            "time_varying_known_categoricals": indicator_cols,
            "categorical_encoders": self.pretrained_categorical_encoders,
            "add_relative_time_idx": self.add_relative_time_idx,
            "add_target_scales": self.add_target_scales,
            "allow_missing_timesteps": self.allow_missing_timesteps,
        }

        return params


@dataclass
class TFTTrainerConfig:
    max_epochs: int = 30
    batch_size: int = 64
    gradient_clip_val: float = 0.1
    patience: int = 3
    # Use Union for flexibility: -1 for all, int count, list of device indices, or "auto"
    devices: Union[int, List[int], str] = "auto"
