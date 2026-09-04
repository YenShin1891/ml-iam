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
    # Patience for the final fit, which trains longer than a search trial and
    # so can afford to wait longer before calling a plateau -- the same reason
    # LSTM and TFT raise theirs from patience to final_patience.
    final_early_stopping_rounds: int = 50
    n_folds: int = 5
    # Full-budget ceiling on boosting rounds.  A budget, not a hyperparameter:
    # early stopping on the validation set picks the round count the final
    # model is trained for, the same way it picks best_epoch for LSTM and TFT.
    num_boost_round: int = 1500
    # Score every search trial on the same autoregressive rollout the test
    # phase runs, where each step's prediction becomes the next step's lag
    # feature.  One-step predictions from ground-truth lags measure a
    # different quantity -- a model can track the truth and still compound
    # its own errors -- and XGBoost is the only one of the three models with
    # that gap, since the LSTM and TFT validate on what they report.  The
    # rollout used to cost minutes per trial (28,000 single-row predicts), so
    # it was switched off in e5b18d9 and restored for stage 2 alone in
    # 3169031; rolling every trajectory forward together made it seconds, so
    # both stages now score what the paper reports.  The one-step error is
    # still recorded per trial (val_score_one_step) so how well it predicts
    # the rollout can be reported.
    search_autoregressive_stage1: bool = True
    search_autoregressive_stage2: bool = True

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
