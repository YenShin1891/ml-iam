from dataclasses import dataclass
from typing import Union


@dataclass
class XGBTrainerConfig:
    """Trainer defaults for XGB, mirroring TFTTrainerConfig style."""

    # Base xgboost params used by trainer's get_xgb_params
    tree_method: str = 'hist'
    device: str = 'cuda'
    eval_metric: str = 'rmse'
    verbosity: int = 0
    max_bin: int = 256

    # Training loop controls
    early_stopping_rounds: int = 15
    n_folds: int = 5

    # Diagnostics
    # When True, show tqdm progress bars inside autoregressive validation during search.
    # This can be very noisy; leave False for normal runs.
    search_show_autoreg_progress: bool = False

    # How often xgboost prints its own eval metric while fitting a search
    # trial, in boosting rounds (False silences it).  Every trial fits one
    # model per target, so at the previous setting of 25 this produced 6855 of
    # the 7107 lines in a run's console log and buried the per-trial results.
    search_fit_verbose: Union[int, bool] = False


__all__ = [
    'XGBTrainerConfig',
]
