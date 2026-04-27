from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class TFTDefaultParams:
    """Default TFT parameters for when search is skipped."""
    hidden_size: int = 256
    lstm_layers: int = 2
    dropout: float = 0.3
    learning_rate: float = 0.01
    best_epoch: int = 12

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format expected by trainer."""
        return {
            "hidden_size": self.hidden_size,
            "lstm_layers": self.lstm_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "best_epoch": self.best_epoch,
        }


@dataclass
class TFTSearchSpace:
    param_dist: Dict[str, List[Any]] = field(default_factory=lambda: {
        "hidden_size": [128, 256, 512],
        "lstm_layers": [1, 2, 3],
        "dropout": [0.1, 0.2, 0.3],
        "learning_rate": [0.001, 0.01],
    })
    param_pairs: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("hidden_size", "dropout"),
        ("lstm_layers", "learning_rate"),
    ])
    search_iter_n: int = 50
    # Two-stage search: cheap exploration, then deeper rerank on top candidates.
    stage1_max_epochs: int = 25
    stage2_top_k: int = 10
