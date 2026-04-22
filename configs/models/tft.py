from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

from configs.data import CATEGORICAL_COLUMNS, INDEX_COLUMNS, MAX_SERIES_LENGTH


@dataclass
class TFTDatasetConfig:
    """Builder for TimeSeriesDataSet parameters used by TFT.

    Usage:
        params = TFTDatasetConfig().build(features, targets, mode="train"|"eval")
    """

    time_idx: str = "Step"
    group_ids: List[str] = field(default_factory=lambda: INDEX_COLUMNS)
    max_encoder_length: int = 3
    min_encoder_length: int = 3
    max_prediction_length: int = MAX_SERIES_LENGTH - 3
    min_prediction_length: int = 1
    add_relative_time_idx: bool = True
    add_target_scales: bool = True
    allow_missing_timesteps: bool = False
    pretrained_categorical_encoders: Dict[str, Any] = field(default_factory=dict)
    target_offset: int = 0  # Set 1 for warm start: reserves encoder context for future predictions (set 0 for cold start)
    _effective_min_encoder_length: int = field(init=False, default=0)
    _effective_max_encoder_length: int = field(init=False, default=0)

    def build(
        self,
        features: List[str],
        targets: List[str],
        mode: str,
    ) -> Dict[str, Any]:
        # Lazy import to avoid heavy deps at module import time
        from pytorch_forecasting.data import EncoderNormalizer, MultiNormalizer

        if isinstance(targets, str):
            targets = [targets]
        time_known = ["Year", "DeltaYears"]
        indicator_cols = [f for f in features if f.endswith("_is_missing")]
        time_known_reals = time_known + indicator_cols

        # Unknown real-valued features exclude categoricals, known time columns, and indicator categoricals
        unknown_reals = [
            f for f in features
            if f not in (CATEGORICAL_COLUMNS + time_known_reals)
        ]

        # Per-sample normalization from each sample's encoder window.
        # Avoids GroupNormalizer's silent fallback to global median stats
        # for unseen groups at test time (data is split by group).
        target_normalizer = MultiNormalizer([
            EncoderNormalizer() for _ in targets
        ])

        min_encoder_length, max_encoder_length = self.resolve_encoder_lengths()

        params: Dict[str, Any] = {
            "time_idx": self.time_idx,
            "target": targets,
            "group_ids": self.group_ids,
            "max_encoder_length": max_encoder_length,
            "min_encoder_length": min_encoder_length,
            "max_prediction_length": self.max_prediction_length,
            "min_prediction_length": self.min_prediction_length,
            "target_normalizer": target_normalizer,
            "time_varying_known_reals": time_known_reals,
            "time_varying_unknown_reals": unknown_reals,
            "static_categoricals": CATEGORICAL_COLUMNS,
            "time_varying_known_categoricals": [],
            "categorical_encoders": self.pretrained_categorical_encoders,
            "add_relative_time_idx": self.add_relative_time_idx,
            "add_target_scales": self.add_target_scales,
            "allow_missing_timesteps": self.allow_missing_timesteps,
        }

        return params

    def resolve_encoder_lengths(self) -> Tuple[int, int]:
        """Resolve encoder lengths after applying warm-start context requirements."""
        required_context = max(0, self.target_offset)
        min_encoder_length = max(self.min_encoder_length, required_context)
        max_encoder_length = max(self.max_encoder_length, min_encoder_length)
        self._effective_min_encoder_length = min_encoder_length
        self._effective_max_encoder_length = max_encoder_length
        return min_encoder_length, max_encoder_length

    @property
    def effective_min_encoder_length(self) -> int:
        if self._effective_min_encoder_length == 0:
            self.resolve_encoder_lengths()
        return self._effective_min_encoder_length

    @property
    def effective_max_encoder_length(self) -> int:
        if self._effective_max_encoder_length == 0:
            self.resolve_encoder_lengths()
        return self._effective_max_encoder_length


@dataclass
class TFTTrainerConfig:
    # Search phase: enough epochs for slow LRs (e.g. 0.001) to converge
    max_epochs: int = 60
    batch_size: int = 64
    gradient_clip_val: float = 0.1
    patience: int = 8
    # Final training: more room to converge
    final_max_epochs: int = 100
    final_patience: int = 20
    # Use Union for flexibility: -1 for all, int count, list of device indices, or "auto"
    devices: Union[int, List[int], str] = "auto"
