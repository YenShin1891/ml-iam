from dataclasses import dataclass
from typing import Any, Dict

from configs.data import CONTEXT_LENGTHS
from src.trainers.search import (
    Choice,
    IntLogUniform,
    IntUniform,
    LogUniform,
    SearchSpace,
    Uniform,
)


@dataclass
class XGBDefaultParams:
    """Default XGBoost hyperparameters used when search is skipped.

    The previous search's winner (xgb_85); independent of the space below.
    No round count: like a searched run, the final fit early-stops on the
    validation set rather than training for a number fixed in advance.
    """

    max_depth: int = 17
    min_child_weight: int = 3
    gamma: float = 0.05988567454877808
    eta: float = 0.03323227712717957
    reg_alpha: float = 0.0006118466600411781
    reg_lambda: float = 24.067485967955477
    subsample: float = 0.8145098177471501
    colsample_bytree: float = 0.8968951911726731
    # Context length, not a booster argument; see XGBSearchSpace.
    n_lags: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "eta": self.eta,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "n_lags": self.n_lags,
        }


class XGBSearchSpace(SearchSpace):
    """XGBoost search space.

    Previously three sequential stages that each swept a small grid
    exhaustively while holding the other parameters fixed.  That is
    coordinate descent, not search: it cannot see an interaction between,
    say, tree depth and regularisation, and it made "one trial" mean
    something different here than for the other two models.  This is the
    same one-shot random search they run.

    ``n_lags`` is searched but is not a booster argument -- it selects which
    prepared feature frame a trial trains on, and is stripped before the
    parameters reach XGBRegressor.

    ``num_boost_round`` is deliberately absent.  It is a *budget*, not a
    hyperparameter -- early stopping on the validation set picks the round
    count, exactly as early stopping picks ``best_epoch`` for LSTM and TFT --
    so it lives in ``stage1_budget`` and ``XGBTrainerConfig.num_boost_round``.
    """

    def __init__(self, **overrides):
        defaults = dict(
            distributions={
                "max_depth": IntUniform(4, 18),
                "min_child_weight": IntLogUniform(1, 40),
                "gamma": Uniform(0.0, 0.5),
                "eta": LogUniform(0.01, 0.5),
                # 1e-8 stands in for "off"; see the note in LSTMSearchSpace.
                "reg_alpha": LogUniform(1e-8, 10.0),
                "reg_lambda": LogUniform(1e-2, 100.0),
                "subsample": Uniform(0.6, 1.0),
                "colsample_bytree": Uniform(0.6, 1.0),
                # Context length: how many past steps the lag features carry.
                # The counterpart of the LSTM's sequence_length and the TFT's
                # encoder length, held to the same two settings.
                "n_lags": Choice(CONTEXT_LENGTHS),
            },
            n_trials=50,
            stage1_budget={"num_boost_round": 300},
            stage2_top_k=8,
            seed=0,
        )
        defaults.update(overrides)
        super().__init__(**defaults)


__all__ = ["XGBDefaultParams", "XGBSearchSpace"]
