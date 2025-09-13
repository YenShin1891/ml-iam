from dataclasses import dataclass, field
from typing import Any, Dict, List, Union, Optional

from configs.data import CATEGORICAL_COLUMNS


@dataclass
class LSTMDatasetConfig:
    """Configuration for LSTM dataset feature separation."""

    def build_feature_groups(self, features: List[str]) -> Dict[str, List[str]]:
        """All features are exogenous u_t for LSTM with teacher forcing."""
        # ALL features are exogenous - observed at current timestep
        exogenous = ["Year", "DeltaYears"]
        indicator_cols = [f for f in features if f.endswith("_is_missing")]
        economic_indicators = [
            f for f in features
            if f not in (CATEGORICAL_COLUMNS + exogenous + indicator_cols)
        ]

        # ALL features are part of u_t (exogenous input)
        all_exogenous = exogenous + indicator_cols + economic_indicators

        return {
            "exogenous_features": all_exogenous,  # u_t - ALL features from dataset
            "static_categoricals": CATEGORICAL_COLUMNS,  # Static info
        }


@dataclass
class LSTMTrainerConfig:
    """Training configuration for LSTM model, following TFT pattern."""

    # Model architecture
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    bidirectional: bool = False

    # Dense layers after LSTM
    dense_hidden_size: int = 64
    dense_dropout: float = 0.0

    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 5
    gradient_clip_val: float = 1.0

    # Device configuration
    devices: Union[int, List[int], str] = "auto"
    accelerator: str = "auto"

    # Optimizer
    optimizer: str = "adam"
    weight_decay: float = 0.0

    # Scheduler
    scheduler: Optional[str] = None
    scheduler_params: Dict[str, Any] = field(default_factory=dict)

    # Data processing
    sequence_length: int = 2
    mask_value: float = -1.0

    # Early stopping
    monitor: str = "val_loss"
    mode: str = "min"

    # Logging
    log_every_n_steps: int = 10

    def build_optimizer_params(self) -> Dict[str, Any]:
        """Build optimizer parameters."""
        params = {
            "lr": self.learning_rate,
        }

        if self.optimizer.lower() == "adam":
            params["weight_decay"] = self.weight_decay
        elif self.optimizer.lower() == "sgd":
            params["momentum"] = self.scheduler_params.get("momentum", 0.9)
            params["weight_decay"] = self.weight_decay

        return params

    def build_scheduler_params(self) -> Optional[Dict[str, Any]]:
        """Build scheduler parameters."""
        if self.scheduler is None:
            return None

        base_params = {"optimizer": None}  # Will be set by trainer
        base_params.update(self.scheduler_params)

        return base_params


@dataclass
class LSTMSearchSpace:
    """Hyperparameter search space for LSTM model."""

    # Model architecture search space - heavily reduced for better coverage
    hidden_size: List[int] = field(default_factory=lambda: [32, 64])
    num_layers: List[int] = field(default_factory=lambda: [1, 2])
    dropout: List[float] = field(default_factory=lambda: [0.1, 0.2])

    # Dense layers
    dense_hidden_size: List[int] = field(default_factory=lambda: [32, 64])
    dense_dropout: List[float] = field(default_factory=lambda: [0.0, 0.1])

    # Training parameters - focused around optimal TFT values
    learning_rate: List[float] = field(default_factory=lambda: [0.01, 0.02])
    batch_size: List[int] = field(default_factory=lambda: [32, 64])
    weight_decay: List[float] = field(default_factory=lambda: [0.0, 1e-5])

    # Search configuration
    search_iter_n: int = 24

    @property
    def param_dist(self) -> Dict[str, List]:
        """Get parameter distribution for sklearn ParameterSampler."""
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "dense_hidden_size": self.dense_hidden_size,
            "dense_dropout": self.dense_dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
        }


__all__ = [
    "LSTMDatasetConfig",
    "LSTMTrainerConfig",
    "LSTMSearchSpace",
]