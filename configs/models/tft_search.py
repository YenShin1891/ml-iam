from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class TFTSearchSpace:
    param_dist: Dict[str, List[Any]] = field(default_factory=lambda: {
        "hidden_size": [8, 16, 32],
        "lstm_layers": [1, 2],
        "dropout": [0.1, 0.3],
        "learning_rate": [0.001, 0.01],
    })
    param_pairs: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("hidden_size", "dropout"),
        ("lstm_layers", "learning_rate"),
    ])
    search_iter_n: int = 10
