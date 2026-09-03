"""The saved template must reproduce the training dataset without pickling it."""

import os

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_forecasting")

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import MultiNormalizer, TorchNormalizer

import src.trainers.tft_dataset as tft_dataset
from src.trainers.tft_dataset import (
    DatasetTemplate,
    from_train_template,
    load_dataset_template,
    save_dataset_template,
)

ENCODER, HORIZON = 3, 12


@pytest.fixture
def frame():
    return pd.DataFrame(
        [
            {"gid": f"G{g}", "Step": s, "A": float(s), "B": float(-s), "feat": float(s)}
            for g in range(30)
            for s in range(ENCODER + HORIZON + 2)
        ]
    )


@pytest.fixture
def dataset(frame):
    return TimeSeriesDataSet(
        frame,
        time_idx="Step",
        target=["A", "B"],
        group_ids=["gid"],
        max_encoder_length=ENCODER,
        min_encoder_length=ENCODER,
        max_prediction_length=HORIZON,
        min_prediction_length=1,
        time_varying_unknown_reals=["A", "B"],
        time_varying_known_reals=["feat"],
        static_categoricals=["gid"],
        add_relative_time_idx=True,
        add_target_scales=True,
        target_normalizer=MultiNormalizer([TorchNormalizer(), TorchNormalizer()]),
    )


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(tft_dataset, "get_run_root", lambda _run_id: str(tmp_path))
    return "tft_01", tmp_path


# ── round trip ────────────────────────────────────────────────────────────


def test_saved_template_rebuilds_an_equivalent_dataset(run, dataset, frame):
    run_id, _ = run
    save_dataset_template(dataset, run_id)

    template = load_dataset_template(run_id)
    rebuilt = from_train_template(template, frame, mode="predict")
    reference = TimeSeriesDataSet.from_dataset(dataset, frame, predict=True, stop_randomization=True)

    assert len(rebuilt) == len(reference)
    assert rebuilt.get_parameters().keys() == reference.get_parameters().keys()
    x_rebuilt, _ = next(iter(rebuilt.to_dataloader(train=False, batch_size=4, num_workers=0)))
    x_reference, _ = next(iter(reference.to_dataloader(train=False, batch_size=4, num_workers=0)))
    torch.testing.assert_close(x_rebuilt["encoder_cont"], x_reference["encoder_cont"])


def test_template_exposes_what_callers_read(run, dataset):
    run_id, _ = run
    save_dataset_template(dataset, run_id)

    template = load_dataset_template(run_id)

    assert template.time_idx == dataset.time_idx
    assert template.group_ids == list(dataset.group_ids)
    assert template.max_encoder_length == dataset.max_encoder_length
    assert template.max_prediction_length == dataset.max_prediction_length
    # Derived during TimeSeriesDataSet.__init__, not recoverable from the
    # constructor arguments — hence stored alongside them.
    assert template.reals == list(dataset.reals)
    assert template.categoricals == list(dataset.categoricals)
    assert "relative_time_idx" in template.reals


def test_template_does_not_carry_the_training_frame(run, dataset, frame):
    run_id, root = run
    save_dataset_template(dataset, run_id)

    path = root / "final" / "dataset_template.pt"
    pickled_dataset = root / "whole_dataset.pt"
    torch.save(dataset, pickled_dataset)

    assert os.path.getsize(path) < os.path.getsize(pickled_dataset) / 5
    blob = path.read_bytes()
    assert b"DataFrame" not in blob


def test_repeated_builds_do_not_mutate_the_template(run, dataset, frame):
    run_id, _ = run
    save_dataset_template(dataset, run_id)
    template = load_dataset_template(run_id)

    before = dict(template.parameters)
    from_train_template(template, frame, mode="predict")
    second = from_train_template(template, frame, mode="eval")

    assert template.parameters == before
    assert second.predict_mode is False


# ── compatibility ─────────────────────────────────────────────────────────


def test_legacy_pickled_dataset_still_loads(run, dataset, frame):
    """Runs trained before this change saved the whole TimeSeriesDataSet."""
    run_id, root = run
    (root / "final").mkdir(parents=True, exist_ok=True)
    torch.save(dataset, root / "final" / "dataset_template.pt")

    template = load_dataset_template(run_id)

    assert isinstance(template, DatasetTemplate)
    assert template.reals == list(dataset.reals)
    assert len(from_train_template(template, frame, mode="predict")) == 30


def test_live_dataset_is_still_accepted(dataset, frame):
    """build_datasets() passes the in-memory training dataset."""
    built = from_train_template(dataset, frame, mode="eval")

    assert isinstance(built, TimeSeriesDataSet)


def test_missing_template_names_the_phase_that_writes_it(run):
    run_id, _ = run

    with pytest.raises(FileNotFoundError, match="train_final_tft"):
        load_dataset_template(run_id)


def test_unrecognised_template_is_rejected(run):
    run_id, root = run
    (root / "final").mkdir(parents=True, exist_ok=True)
    torch.save(["not a template"], root / "final" / "dataset_template.pt")

    with pytest.raises(RuntimeError, match="Unrecognised dataset template"):
        load_dataset_template(run_id)


def test_template_exposes_the_fitted_categorical_encoders(run, dataset):
    """The inference script drops rows whose Model the encoder never saw."""
    run_id, _ = run
    save_dataset_template(dataset, run_id)

    encoders = load_dataset_template(run_id).categorical_encoders

    assert "__group_id__gid" in encoders
    assert set(encoders["__group_id__gid"].classes_) == {f"G{g}" for g in range(30)}


# ── context length must not change the evaluation geometry ────────────────
#
# The whole point of searching the encoder length is to learn whether more
# history helps.  If a shorter encoder also predicted more steps, or admitted
# shorter trajectories, it would win on being scored differently instead.

from configs.data import CONTEXT_LENGTHS, MAX_CONTEXT_LENGTH, MAX_SERIES_LENGTH


def _config(encoder_length):
    from configs.models.tft import TFTDatasetConfig

    config = TFTDatasetConfig()
    config.max_encoder_length = encoder_length
    config.min_encoder_length = encoder_length
    return config


def test_every_encoder_length_predicts_the_same_number_of_steps():
    horizons = {_config(length).max_prediction_length for length in CONTEXT_LENGTHS}

    assert horizons == {MAX_SERIES_LENGTH - MAX_CONTEXT_LENGTH}


def test_the_horizon_does_not_stretch_when_the_encoder_shrinks():
    """MAX_SERIES_LENGTH - encoder_length would give the short one extra steps."""
    short, long = min(CONTEXT_LENGTHS), max(CONTEXT_LENGTHS)

    assert _config(short).max_prediction_length == _config(long).max_prediction_length


def test_a_shorter_encoder_needs_a_shorter_window():
    short, long = min(CONTEXT_LENGTHS), max(CONTEXT_LENGTHS)

    def window(length):
        config = _config(length)
        return config.resolve_encoder_lengths()[1] + config.max_prediction_length

    assert window(short) == window(long) - (long - short)


def test_the_trajectory_length_threshold_ignores_the_context_in_use():
    """So the same trajectories are scored whichever encoder length wins."""
    from src.trainers.tft_dataset import required_group_length

    horizon = _config(max(CONTEXT_LENGTHS)).max_prediction_length

    assert required_group_length(horizon) == MAX_CONTEXT_LENGTH + horizon
    # It is a function of the horizon alone -- no encoder length reaches it.
    assert required_group_length(1) == MAX_CONTEXT_LENGTH + 1


def test_the_threshold_leaves_room_for_the_longest_window():
    """A trajectory that passes must fit the longest encoder plus the horizon."""
    from src.trainers.tft_dataset import required_group_length

    config = _config(max(CONTEXT_LENGTHS))
    window = config.resolve_encoder_lengths()[1] + config.max_prediction_length

    assert required_group_length(config.max_prediction_length) == window
