from dataclasses import dataclass, field
from typing import Any, Dict, List, Union, Optional


@dataclass
class LSTMTrainerConfig:
    """Training configuration for LSTM model, following TFT pattern."""

    # Model architecture
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = False

    # Categorical embeddings
    embedding_dim: int = 16

    # Dense layers after LSTM
    dense_hidden_size: int = 64
    dense_dropout: float = 0.1

    # Training parameters
    learning_rate: float = 0.01
    batch_size: int = 128
    max_epochs: int = 100
    patience: int = 5
    final_patience: int = 20
    gradient_clip_val: float = 1.0

    # Device configuration
    devices: Union[int, List[int], str] = "auto"
    accelerator: str = "auto"

    # Optimizer
    optimizer: str = "adam"
    weight_decay: float = 0

    # Scheduler
    scheduler: Optional[str] = None
    scheduler_params: Dict[str, Any] = field(default_factory=dict)

    # Data processing
    sequence_length: int = 3  # Number of historical timesteps fed into the model
    target_offset: int = 0  # Set 1 for warm start: reserves encoder context for future predictions (set 0 for cold start)
    mask_value: float = -1.0

    # Early stopping
    monitor: str = "val_loss"
    mode: str = "min"


@dataclass
class LSTMSearchSpace:
    """Hyperparameter search space for LSTM model."""

    # Model architecture search space - heavily reduced for better coverage
    hidden_size: List[int] = field(default_factory=lambda: [64, 128])
    num_layers: List[int] = field(default_factory=lambda: [2, 3])
    dropout: List[float] = field(default_factory=lambda: [0.1, 0.2])

    # Dense layers
    dense_hidden_size: List[int] = field(default_factory=lambda: [64, 128])
    dense_dropout: List[float] = field(default_factory=lambda: [0.0, 0.1])

    # Training parameters - focused around optimal TFT values
    learning_rate: List[float] = field(default_factory=lambda: [0.01, 0.02])
    batch_size: List[int] = field(default_factory=lambda: [64, 128])
    weight_decay: List[float] = field(default_factory=lambda: [0.0, 1e-5])

    # Categorical embedding dimension
    embedding_dim: List[int] = field(default_factory=lambda: [4, 8, 16, 32])

    # Sequence length to search (reintroduced)
    sequence_length: List[int] = field(default_factory=lambda: [1, 2, 3, 4])

    # Search configuration
    search_iter_n: int = 48
    # Trials are cut short: the search ranks configurations, the final fit
    # (LSTMTrainerConfig.max_epochs / final_patience) trains them properly.
    max_epochs: int = 20
    patience: int = 3

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
            "embedding_dim": self.embedding_dim,
            "sequence_length": self.sequence_length,
        }


__all__ = [
    "LSTMTrainerConfig",
    "LSTMSearchSpace",
]
