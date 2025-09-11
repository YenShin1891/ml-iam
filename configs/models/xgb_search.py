from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# Stage 1: Tree Structure
STAGE_1_PARAMS: Dict[str, Any] = {
    'max_depth': [5, 9, 13, 17],
    'min_child_weight': [10, 12, 15, 20],
    'gamma': [0],  # Keep gamma at 0 initially
    'eta': [0.4],  # Fixed learning rate
    'num_boost_round': [1000],  # Fixed with early stopping
    'reg_alpha': [0],  # Fixed initially
    'reg_lambda': [1],  # Fixed initially
}

# Stage 2: Learning Rate and Number of Trees
STAGE_2_PARAMS: Dict[str, Any] = {
    'max_depth': None,  # Will be set from stage 1 best
    'min_child_weight': None,  # Will be set from stage 1 best
    'gamma': [0],  # Keep at 0
    'eta': [0.1, 0.3, 0.4, 0.5],
    'num_boost_round': [300, 500, 700, 1000],
    'reg_alpha': [0],  # Fixed initially
    'reg_lambda': [1],  # Fixed initially
}

# Stage 3: Regularization
STAGE_3_PARAMS: Dict[str, Any] = {
    'max_depth': None,  # Will be set from stage 1 best
    'min_child_weight': None,  # Will be set from stage 1 best
    'gamma': [0, 0.1],
    'eta': None,  # Will be set from stage 3 best
    'num_boost_round': None,  # Will be set from stage 3 best
    'reg_alpha': [0, 1, 5, 10],
    'reg_lambda': [0.1, 1, 10]
}


@dataclass
class XGBSearchSpace:
    """Dataclass wrapper for staged XGB search spaces with helpers."""

    stage_1: Dict[str, Any] = field(default_factory=lambda: dict(STAGE_1_PARAMS))
    stage_2: Dict[str, Any] = field(default_factory=lambda: dict(STAGE_2_PARAMS))
    stage_3: Dict[str, Any] = field(default_factory=lambda: dict(STAGE_3_PARAMS))

    def stages(self) -> List[Tuple[str, Dict[str, Any]]]:
        return [
            ("Stage 1: Tree Structure", self.stage_1),
            ("Stage 2: Learning Rate & Trees", self.stage_2),
            ("Stage 3: Regularization", self.stage_3),
        ]

    @staticmethod
    def build_param_dist(stage_params: Dict[str, Any], best_params: Dict[str, Any]) -> Dict[str, List[Any]]:
        """Merge current stage params with best params from previous stages."""
        param_dist: Dict[str, List[Any]] = {}
        for param, values in stage_params.items():
            if values is None:
                if param in best_params:
                    param_dist[param] = [best_params[param]]
                else:
                    raise ValueError(
                        f"Parameter '{param}' required from previous stage but not found in best_params."
                    )
            else:
                param_dist[param] = values
        return param_dist


__all__ = [
    "XGBSearchSpace",
    "STAGE_1_PARAMS",
    "STAGE_2_PARAMS",
    "STAGE_3_PARAMS",
]
