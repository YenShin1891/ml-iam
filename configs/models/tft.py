from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import torch
from pytorch_forecasting.data import EncoderNormalizer
from sklearn.preprocessing import StandardScaler

from configs.data import (
    CATEGORICAL_COLUMNS,
    INDEX_COLUMNS,
    MAX_CONTEXT_LENGTH,
    MAX_SERIES_LENGTH,
)


class NamelessStandardScaler(StandardScaler):
    """StandardScaler that never learns feature names.

    TimeSeriesDataSet fits covariate scalers on one-column DataFrames but
    re-applies the per-sample ones (the target center/scale columns) to bare
    numpy slices in ``__getitem__``, and sklearn warns about the mismatch on
    every one of those calls -- 18 lines per sample.  Fitting on the values
    alone keeps both sides nameless.
    """

    @staticmethod
    def _nameless(X):
        # Only pandas carries feature names; torch input must stay torch,
        # because __getitem__ assigns the transformed value back into a tensor.
        return X.to_numpy() if isinstance(X, (pd.DataFrame, pd.Series)) else X

    def fit(self, X, y=None):
        return super().fit(self._nameless(X), y)

    def transform(self, X, copy=None):
        return super().transform(self._nameless(X), copy=copy)

    def inverse_transform(self, X, copy=None):
        return super().inverse_transform(self._nameless(X), copy=copy)


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
    group_ids: List[str] = field(default_factory=lambda: list(INDEX_COLUMNS))
    max_encoder_length: int = MAX_CONTEXT_LENGTH
    min_encoder_length: int = MAX_CONTEXT_LENGTH
    # Pinned to the longest context, not to this instance's: a shorter encoder
    # must predict the same steps, or the encoder lengths are not comparable.
    max_prediction_length: int = MAX_SERIES_LENGTH - MAX_CONTEXT_LENGTH
    min_prediction_length: int = 1
    add_relative_time_idx: bool = True
    add_target_scales: bool = True
    allow_missing_timesteps: bool = False
    pretrained_categorical_encoders: Dict[str, Any] = field(default_factory=dict)
    target_offset: int = 0  # Set 1 for warm start: reserves encoder context for future predictions (set 0 for cold start)
    # "encoder_floored" (default): per-sample encoder normalization with scale floor
    # "global": single global μ/σ per target from training data (TorchNormalizer)
    target_normalizer_mode: str = "encoder_floored"
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

        unknown_reals = []
        time_known_reals = [
            f for f in features
            if f not in excluded
        ]

        # Include __observed mask columns as known reals so the masked loss
        # can access them from the batch.  They are boolean-like (0/1) and
        # known at all time steps.
        # pytorch_forecasting standardises every real it is not told otherwise
        # about.  The __observed columns must reach MaskedRMSE as the 0/1
        # masks they are -- standardised, an unobserved element's weight goes
        # negative and the loss rewards error on it.  The target center/scale
        # columns keep default scaling but need scalers without feature names
        # (see NamelessStandardScaler).
        scalers: Dict[str, Any] = {}

        from configs.data import KEEP_PARTIAL_TARGETS
        if KEEP_PARTIAL_TARGETS:
            from src.data.preprocess import observed_mask_columns
            obs_cols = observed_mask_columns(targets)
            time_known_reals.extend(obs_cols)
            scalers.update({col: None for col in obs_cols})

        if self.add_target_scales:
            for target in targets:
                scalers[f"{target}_center"] = NamelessStandardScaler()
                scalers[f"{target}_scale"] = NamelessStandardScaler()

        if self.target_normalizer_mode == "global":
            # Single global μ/σ per target, fitted once on the full training
            # column at TimeSeriesDataSet init time.  Stable but loses
            # per-sample level information.
            from pytorch_forecasting.data.encoders import TorchNormalizer
            target_normalizer = MultiNormalizer([
                TorchNormalizer(method="standard") for _ in targets
            ])
        else:
            # Per-sample normalization from each sample's encoder window,
            # with a minimum scale floor to prevent degenerate normalization.
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
            "scalers": scalers,
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
