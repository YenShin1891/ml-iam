from dataclasses import dataclass


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
    search_iter_n_per_stage: int = 15

    # Diagnostics
    # When True, show tqdm progress bars inside autoregressive validation during search.
    # This can be very noisy; leave False for normal runs.
    search_show_autoreg_progress: bool = False


__all__ = [
    'XGBTrainerConfig',
]
