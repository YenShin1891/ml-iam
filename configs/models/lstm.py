from dataclasses import dataclass, field
from typing import Any, Dict, List, Union, Optional

from src.trainers.search import Choice, IntLogUniform, LogUniform, SearchSpace, Uniform


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


class LSTMSearchSpace(SearchSpace):
    """LSTM search space.

    The widest of the three -- an LSTM trial costs seconds, so there is no
    reason to search fewer knobs than the architecture actually has.  It runs
    the same two-stage protocol as the others: stage-1 trials are cut to 20
    epochs to rank them, the top 10 are refit under
    ``LSTMTrainerConfig.max_epochs`` / ``final_patience``.
    """

    def __init__(self, **overrides):
        defaults = dict(
            distributions={
                "hidden_size": IntLogUniform(32, 512, multiple_of=8),
                "num_layers": Choice([1, 2, 3]),
                "dropout": Uniform(0.0, 0.4),
                "dense_hidden_size": IntLogUniform(32, 256, multiple_of=8),
                "dense_dropout": Uniform(0.0, 0.3),
                "learning_rate": LogUniform(1e-4, 5e-2),
                "batch_size": Choice([32, 64, 128, 256]),
                # 1e-8 stands in for "off": log-uniform cannot reach 0, and at
                # this scale the penalty is numerically indistinguishable from
                # no penalty at all.
                "weight_decay": LogUniform(1e-8, 1e-3),
                "embedding_dim": IntLogUniform(4, 64, multiple_of=4),
                # Context length, held to the two settings the three models
                # are being compared at.  Structural, so genuinely discrete.
                "sequence_length": Choice([2, 3]),
            },
            n_trials=50,
            stage1_budget={"max_epochs": 20, "patience": 3},
            stage2_top_k=8,
            seed=0,
        )
        defaults.update(overrides)
        super().__init__(**defaults)


__all__ = [
    "LSTMTrainerConfig",
    "LSTMSearchSpace",
]
