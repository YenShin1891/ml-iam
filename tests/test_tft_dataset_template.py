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
