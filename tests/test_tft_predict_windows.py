"""The what-if engine runs the two-window forecast on an already-loaded model, on the CPU."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_forecasting")

from configs.data import OUTPUT_VARIABLES
import src.trainers.tft_two_window_simple as two_window
from src.inference.tft_predict import TFTEngine, check_vocabulary, predict_windows
from src.trainers.tft_dataset import DatasetTemplate, create_train_dataset
from src.trainers.tft_model import create_tft_model

FEATURES = ["GDP", "Population", "Region", "Model_Family"]
REGIONS = ["World", "R5ASIA"]
ENCODER, HORIZON = 3, 12
STEPS = 17


def _frame(n_groups=4, steps=STEPS, short_last=False):
    rng = np.random.default_rng(0)
    rows = []
    for g in range(n_groups):
        n_steps = 10 if (short_last and g == n_groups - 1) else steps
        for step in range(n_steps):
            row = {
                "Model": f"M{g % 2}", "Scenario": f"S{g}", "Region": REGIONS[g % 2],
                "Model_Family": f"FAM{g % 2}", "Step": step, "Year": 2005 + 5 * step,
                "GDP": float(rng.normal(50, 10)), "Population": float(rng.normal(1000, 100)),
            }
            for i, target in enumerate(OUTPUT_VARIABLES):
                row[target] = float(rng.normal(i, 1))
                row[f"{target}__observed"] = 1.0
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def engine():
    frame = _frame()
    dataset, _ = create_train_dataset({"train_data": frame, "features": FEATURES, "targets": list(OUTPUT_VARIABLES)})
    model = create_tft_model(
        dataset, {"hidden_size": 4, "lstm_layers": 1, "dropout": 0.0, "learning_rate": 0.01},
        n_targets=len(OUTPUT_VARIABLES),
    )
    model.eval()
    return TFTEngine("tft_test", model, DatasetTemplate.from_dataset(dataset), FEATURES, list(OUTPUT_VARIABLES))


def test_the_engine_reads_its_geometry_from_the_template(engine):
    assert (engine.encoder_length, engine.prediction_length) == (ENCODER, HORIZON)
    assert engine.min_steps == ENCODER + HORIZON
    assert engine.time_idx == "Step" and engine.group_ids == ["Model", "Scenario", "Region"]


def test_predict_windows_covers_every_step_after_the_encoder(engine, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frame = _frame()

    tidy = predict_windows(engine, frame)

    pred_columns = [f"{t}_pred" for t in OUTPUT_VARIABLES]
    assert set(pred_columns) <= set(tidy.columns) and "Year" in tidy.columns
    per_group = tidy.groupby(["Model", "Scenario", "Region"])["Step"].apply(list)
    assert all(steps == list(range(ENCODER, STEPS)) for steps in per_group)
    assert len(per_group) == 4
    assert np.isfinite(tidy[pred_columns].to_numpy()).all()
    assert not (tmp_path / "lightning_logs").exists()  # the Trainer writes nothing


def test_a_trajectory_shorter_than_the_window_is_refused(engine):
    with pytest.raises(ValueError, match="fewer than 15 steps"):
        predict_windows(engine, _frame(short_last=True))


def test_unchanged_inputs_reproduce_the_forecast_and_an_edit_moves_it(engine):
    frame = _frame(n_groups=2)
    pred_columns = [f"{t}_pred" for t in OUTPUT_VARIABLES]

    first = predict_windows(engine, frame)
    second = predict_windows(engine, frame.copy())
    edited = frame.copy()
    edited.loc[edited["Step"] >= ENCODER, "GDP"] *= 3.0
    third = predict_windows(engine, edited)

    np.testing.assert_array_equal(first[pred_columns].to_numpy(), second[pred_columns].to_numpy())
    assert not np.allclose(first[pred_columns].to_numpy(), third[pred_columns].to_numpy())


def test_check_vocabulary_names_labels_the_encoders_never_saw(engine):
    frame = _frame(n_groups=2)

    assert check_vocabulary(engine, frame) == []
    assert check_vocabulary(engine, frame.assign(Region="MARS")) == ["Region=MARS"]
    assert check_vocabulary(engine, frame.assign(Model_Family="FAM9")) == ["Model_Family=FAM9"]


def test_a_checkpoint_from_a_gpu_loads_on_the_cpu(engine, tmp_path, monkeypatch):
    """The pickled metrics remember cuda; a CPU-only torch must still load the run."""
    import lightning
    import src.trainers.tft_model as tft_model
    from src.trainers.tft_model import load_tft_checkpoint, metrics_to

    model = engine.model
    model.loss._device = torch.device("cuda:0")
    for metric in model.loss.metrics:  # MultiLoss keeps these in a plain list
        metric._device = torch.device("cuda:0")
    (tmp_path / "final").mkdir()
    checkpoint = {
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
        "pytorch-lightning_version": lightning.__version__,
    }
    # What Lightning's save adds through the model: the dataset parameters,
    # the hparams name and the loss under pytorch-forecasting's special key.
    model.on_save_checkpoint(checkpoint)
    torch.save(checkpoint, tmp_path / "final" / "best.ckpt")
    metrics_to(model.loss, "cpu")  # the live model must not be left pointing at cuda
    monkeypatch.setattr(tft_model, "get_run_root", lambda _run_id: str(tmp_path))

    real_zeros = torch.zeros

    def cpu_only_zeros(*args, **kwargs):
        """What a CPU-only build does when a metric asks its old device for a tensor."""
        device = kwargs.get("device")
        if device is not None and torch.device(device).type == "cuda":
            raise AssertionError("Torch not compiled with CUDA enabled")
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", cpu_only_zeros)

    loaded = load_tft_checkpoint("tft_x", map_location="cpu")

    assert loaded.loss._device == torch.device("cpu")
    assert {metric._device for metric in loaded.loss.metrics} == {torch.device("cpu")}
    assert next(loaded.parameters()).device.type == "cpu"


def test_the_test_phase_entry_point_delegates_to_the_shared_forecast(engine, monkeypatch):
    """_predict_window loads the run's artifacts, then runs the same function the view runs."""
    seen = {}

    def fake_forecast(model, template, data, targets, name, slice_window, **kwargs):
        seen.update(model=model, template=template, data=data, targets=targets, name=name,
                    slice_window=slice_window, **kwargs)
        return two_window.WindowPrediction(preds=np.zeros((0, 9)), horizon=pd.DataFrame(), name=name)

    monkeypatch.setattr(two_window, "load_tft_checkpoint", lambda run_id: engine.model)
    monkeypatch.setattr(two_window, "load_dataset_template", lambda run_id: engine.template)
    monkeypatch.setattr(two_window, "get_default_num_workers", lambda: 3)
    monkeypatch.setattr(two_window, "predict_window_frame", fake_forecast)
    session_state = {"test_data": _frame(), "targets": list(OUTPUT_VARIABLES), "tft_target_offset": 0}

    result = two_window._predict_window(session_state, "tft_x", "early", two_window._create_early_window_test_data)

    assert result.name == "early"
    assert seen["model"] is engine.model and seen["template"] is engine.template
    assert seen["data"] is session_state["test_data"] and seen["targets"] == list(OUTPUT_VARIABLES)
    assert seen["slice_window"] is two_window._create_early_window_test_data
    assert seen["target_offset"] == 0 and seen["loader_kwargs"] == {"num_workers": 3}
