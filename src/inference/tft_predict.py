"""Live TFT inference for the what-if view: a run's model on one frame.

The only module of the view that needs torch.  The dashboard imports it
inside the function that runs the emulator, so its trajectories view, and
the CPU-only XGBoost installation, work without the deep-learning stack.
"""

import logging
import threading
from dataclasses import dataclass
from typing import List

import pandas as pd

from configs.dashboard import WHATIF_ACCELERATOR

# One forecast at a time: every dashboard session shares the cached model.
_PREDICT_LOCK = threading.Lock()


@dataclass
class TFTEngine:
    """A run's trained model and dataset template, ready to predict."""

    run_id: str
    model: object
    template: object
    features: List[str]
    targets: List[str]

    @property
    def encoder_length(self) -> int:
        return int(self.template.max_encoder_length)

    @property
    def prediction_length(self) -> int:
        return int(self.template.max_prediction_length)

    @property
    def window_length(self) -> int:
        return self.encoder_length + self.prediction_length

    @property
    def time_idx(self) -> str:
        return str(self.template.time_idx)

    @property
    def group_ids(self) -> List[str]:
        return list(self.template.group_ids)

    @property
    def min_steps(self) -> int:
        """Steps a trajectory needs to receive a forecast at all."""
        from src.trainers.tft_dataset import required_group_length

        return required_group_length(self.prediction_length)


def load_engine(run_id: str, *, map_location: str = WHATIF_ACCELERATOR) -> TFTEngine:
    """Load a run's checkpoint, template and feature lists.

    *map_location* keeps the model off the GPUs by default: the dashboard
    shares the host with training jobs and must not claim one.
    """
    from src.trainers.tft_dataset import load_dataset_template
    from src.trainers.tft_model import load_tft_checkpoint
    from src.utils.run_store import RunStore

    model = load_tft_checkpoint(run_id, map_location=map_location)
    if map_location:
        model.to(map_location)
    template = load_dataset_template(run_id)
    features, targets = RunStore(run_id).load_features()
    logging.info("What-if engine ready for %s on %s", run_id, map_location or "default device")
    return TFTEngine(run_id, model, template, list(features), list(targets))


def check_vocabulary(engine: TFTEngine, rows: pd.DataFrame) -> List[str]:
    """Labels in *rows* the template's encoders never saw, as "column=label".

    The encoders have a closed vocabulary, so an unknown label would fail
    deep inside pytorch-forecasting; naming it here is kinder.
    """
    encoders = engine.template.categorical_encoders
    columns = list(engine.group_ids) + [
        c for c in ("Region", "Model_Family") if c in rows.columns and c not in engine.group_ids
    ]
    problems = set()
    for column in columns:
        encoder = None
        for key in (f"__group_id__{column}", column):
            candidate = encoders.get(key)
            if candidate is not None and hasattr(candidate, "classes_"):
                encoder = candidate
                break
        if encoder is None or column not in rows.columns:
            continue
        known = set(encoder.classes_.keys())
        for label in pd.unique(rows[column].astype(str)):
            if label not in known:
                problems.add(f"{column}={label}")
    return sorted(problems)


def predict_windows(
    engine: TFTEngine,
    frame: pd.DataFrame,
    *,
    accelerator: str = WHATIF_ACCELERATOR,
    batch_size: int = 64,
) -> pd.DataFrame:
    """The run's two-window forecast of every trajectory in *frame*.

    The same early/late windows and blend the test phase scores, on an
    already-loaded model, with no loader workers and a Trainer pinned to
    *accelerator* that writes no logs or checkpoints.  Returns one row per
    predicted step: the group ids, the time index, ``Year`` and
    ``<target>_pred`` columns.
    """
    from src.trainers.tft_two_window_simple import (
        _create_early_window_test_data,
        _create_late_window_test_data,
        combine_windows,
        predict_window_frame,
    )

    time_idx = engine.time_idx
    group_ids = engine.group_ids
    sizes = frame.groupby(group_ids, observed=True, sort=False)[time_idx].size()
    short = sizes[sizes < engine.min_steps]
    if not short.empty:
        raise ValueError(
            f"{len(short)} trajectory(ies) have fewer than {engine.min_steps} steps and cannot be "
            f"emulated: {short.index.tolist()[:3]}"
        )

    trainer_kwargs = {
        "accelerator": accelerator,
        "devices": 1,
        "logger": False,
        "enable_progress_bar": False,
        "enable_checkpointing": False,
    }
    windows = []
    with _PREDICT_LOCK:
        for name, slicer in (
            ("early", _create_early_window_test_data),
            ("late", _create_late_window_test_data),
        ):
            windows.append(
                predict_window_frame(
                    engine.model,
                    engine.template,
                    frame,
                    engine.targets,
                    name,
                    slicer,
                    loader_kwargs={"num_workers": 0, "batch_size": batch_size},
                    predict_kwargs={"trainer_kwargs": dict(trainer_kwargs)},
                )
            )
    combined = combine_windows(windows[0], windows[1], engine.targets, time_idx)

    horizon = combined.horizon.reset_index(drop=True)
    keep = group_ids + [time_idx] + (["Year"] if "Year" in horizon.columns else [])
    out = horizon[keep].copy()
    for i, target in enumerate(engine.targets):
        out[f"{target}_pred"] = combined.preds[:, i]
    return out.sort_values(group_ids + [time_idx]).reset_index(drop=True)
