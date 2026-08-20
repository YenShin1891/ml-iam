"""Model configuration objects.

Resolved lazily: ``configs.models.tft`` imports torch and pytorch_forecasting,
and importing any name from this package used to drag both in.  That made
``src.trainers.xgb_trainer`` — and therefore the whole XGBoost path — depend on
the deep-learning stack that requirements.txt does not install.
"""

from typing import TYPE_CHECKING

_SUBMODULE_BY_NAME = {
    "TFTDatasetConfig": "tft",
    "TFTTrainerConfig": "tft",
    "TFTSearchSpace": "tft_search",
    "XGBTrainerConfig": "xgb",
    "XGBSearchSpace": "xgb_search",
    "LSTMDatasetConfig": "lstm",
    "LSTMTrainerConfig": "lstm",
    "LSTMSearchSpace": "lstm",
}

__all__ = sorted(_SUBMODULE_BY_NAME)


def __getattr__(name: str):
    """PEP 562 hook: import the owning submodule on first access."""
    submodule = _SUBMODULE_BY_NAME.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(f".{submodule}", __name__), name)
    globals()[name] = value  # subsequent lookups skip this hook
    return value


def __dir__():
    return sorted(set(globals()) | set(_SUBMODULE_BY_NAME))


if TYPE_CHECKING:  # keep the names visible to type checkers and IDEs
    from .lstm import LSTMDatasetConfig, LSTMSearchSpace, LSTMTrainerConfig
    from .tft import TFTDatasetConfig, TFTTrainerConfig
    from .tft_search import TFTSearchSpace
    from .xgb import XGBTrainerConfig
    from .xgb_search import XGBSearchSpace
