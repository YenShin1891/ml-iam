from dataclasses import dataclass, field
from typing import Any, Dict, List

from configs.config import INDEX_COLUMNS, MAX_SERIES_LENGTH, CATEGORICAL_COLUMNS


@dataclass
class TFTDatasetConfig:
    """Builder for TimeSeriesDataSet parameters used by TFT.

    Usage:
      params = TFTDatasetConfig().build(features, targets, mode="train"|"eval")
    """

    time_idx: str = "Step"
    group_ids: List[str] = field(default_factory=lambda: INDEX_COLUMNS)
    max_encoder_length: int = 5
    min_encoder_length: int = 2
    max_prediction_length: int = MAX_SERIES_LENGTH
    max_prediction_length_eval: int = MAX_SERIES_LENGTH
    min_prediction_length: int = 1
    add_relative_time_idx: bool = True
    add_target_scales: bool = True
    allow_missing_timesteps: bool = False

    def build(self, features: List[str], targets: List[str], mode: str) -> Dict[str, Any]:
        # Lazy import to avoid heavy deps at module import time
        from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
        from pytorch_forecasting.data.encoders import NaNLabelEncoder

        if isinstance(targets, str):
            targets = [targets]
        time_known = ["Year", "DeltaYears"]
        indicator_cols = [f for f in features if f.endswith("_is_missing")]

        # unknown real-valued features exclude categoricals, known time columns, and indicator categoricals
        unknown_reals = [
            f for f in features
            if f not in (CATEGORICAL_COLUMNS + time_known + indicator_cols)
        ]

        # shared normalizer across targets, grouped by group_ids
        target_normalizer = MultiNormalizer([
            GroupNormalizer(groups=self.group_ids)
        ] * len(targets))

        max_pred_len = self.max_prediction_length_eval if mode == "eval" else self.max_prediction_length

        return {
            "time_idx": self.time_idx,
            "target": targets,
            "group_ids": self.group_ids,
            "max_encoder_length": self.max_encoder_length,
            "min_encoder_length": self.min_encoder_length,
            "max_prediction_length": max_pred_len,
            "min_prediction_length": self.min_prediction_length,
            "target_normalizer": target_normalizer,
            "time_varying_known_reals": time_known,
            "time_varying_unknown_reals": unknown_reals,
            "static_categoricals": CATEGORICAL_COLUMNS,
            "time_varying_known_categoricals": indicator_cols,
            "categorical_encoders": {c: NaNLabelEncoder(add_nan=False) for c in indicator_cols},
            "add_relative_time_idx": self.add_relative_time_idx,
            "add_target_scales": self.add_target_scales,
            "allow_missing_timesteps": self.allow_missing_timesteps,
        }


@dataclass
class TFTTrainerConfig:
    max_epochs: int = 30
    batch_size: int = 64
    gradient_clip_val: float = 0.1
    patience: int = 3
    devices: int = 1
