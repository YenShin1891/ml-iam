from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from pytorch_forecasting.data import EncoderNormalizer

from configs.data import CATEGORICAL_COLUMNS, INDEX_COLUMNS, MAX_SERIES_LENGTH


class FlooredEncoderNormalizer(EncoderNormalizer):
    """EncoderNormalizer that clamps scale to a minimum floor.

    Prevents degenerate normalization when the encoder window has near-zero
    variance (e.g. Nuclear/Wind/Solar starting at 0, or constant early values),
    which otherwise produces astronomical normalized targets and noisy gradients.
    """

    def __init__(self, min_scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.min_scale = min_scale

    def fit(self, y):
        super().fit(y)
        if isinstance(self.scale_, torch.Tensor):
            self.scale_ = torch.clamp(self.scale_, min=self.min_scale)
        else:
            self.scale_ = max(self.scale_, self.min_scale)
        return self


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
    target_scale_floors: Dict[str, float] = field(default_factory=dict)
    scale_floor_fraction: float = 0.01  # fraction of global σ used as floor when no explicit floors given
    _effective_min_encoder_length: int = field(init=False, default=0)
    _effective_max_encoder_length: int = field(init=False, default=0)

    def build(
        self,
        features: List[str],
        targets: List[str],
        mode: str,
    ) -> Dict[str, Any]:
        # Lazy import to avoid heavy deps at module import time
        from pytorch_forecasting.data import MultiNormalizer

        if isinstance(targets, str):
            targets = [targets]
        excluded = set(CATEGORICAL_COLUMNS)

        # Lagged target columns (prev_X, prev2_X, …) are unknown at forecast
        # time.  Everything else — scenario drivers, indicators, time columns
        # — is known future by construction.
        import re
        _lag_re = re.compile(r"^prev\d*_")
        unknown_reals = [
            f for f in features
            if f not in excluded and _lag_re.match(f)
        ]
        time_known_reals = [
            f for f in features
            if f not in excluded and not _lag_re.match(f)
        ]

        # Include __observed mask columns as known reals so the masked loss
        # can access them from the batch.  They are boolean-like (0/1) and
        # known at all time steps.
        from configs.data import KEEP_PARTIAL_TARGETS
        if KEEP_PARTIAL_TARGETS:
            from src.data.preprocess import observed_mask_columns
            obs_cols = observed_mask_columns(targets)
            time_known_reals.extend(obs_cols)

        # Per-sample normalization from each sample's encoder window, with a
        # minimum scale floor to prevent degenerate normalization.  The floor
        # bounds the worst-case normalized target magnitude and stabilises
        # loss weighting across samples with different encoder variances.
        target_normalizer = MultiNormalizer([
            FlooredEncoderNormalizer(
                min_scale=self.target_scale_floors.get(t, 1.0),
            )
            for t in targets
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
