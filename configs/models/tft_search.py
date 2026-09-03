from dataclasses import dataclass
from typing import Any, Dict

from src.trainers.search import Choice, IntLogUniform, LogUniform, SearchSpace, Uniform


@dataclass
class TFTDefaultParams:
    """Default TFT parameters for when search is skipped.

    These are the previous search's winner, so they are only meaningful for a
    run that skips the search; changing the space below does not change them.
    """
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


class TFTSearchSpace(SearchSpace):
    """TFT search space.

    The narrowest budget of the three: a single TFT trial takes hours, which
    is why the shared two-stage protocol exists at all.  Stage 1 caps trials
    at 25 epochs to rank them cheaply; only the top 10 are refit under
    ``TFTTrainerConfig.max_epochs``.
    """

    def __init__(self, **overrides):
        defaults = dict(
            distributions={
                "hidden_size": IntLogUniform(64, 512, multiple_of=8),
                "lstm_layers": Choice([1, 2, 3]),
                "dropout": Uniform(0.0, 0.5),
                "learning_rate": LogUniform(1e-4, 3e-2),
            },
            n_trials=50,
            stage1_budget={"max_epochs": 25},
            stage2_top_k=10,
            seed=0,
        )
        defaults.update(overrides)
        super().__init__(**defaults)


__all__ = ["TFTDefaultParams", "TFTSearchSpace"]
