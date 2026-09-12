"""RunStore reads the per-region metrics table the test phase writes."""

import pandas as pd
import pytest

import src.utils.run_store as run_store
from src.trainers.evaluation import by_region_filename


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "get_run_root", lambda _run_id: str(tmp_path))
    return run_store.RunStore("tft_01"), tmp_path


def test_probing_neither_finds_nor_creates_anything(store):
    run, root = store

    assert not run.has_metrics_by_region()
    with pytest.raises(FileNotFoundError, match="test phase"):
        run.load_metrics_by_region()
    assert not (root / "metrics").exists()


def test_the_table_is_read_from_beside_its_metrics_file(store):
    run, root = store
    (root / "metrics").mkdir()
    table = pd.DataFrame({"Run ID": ["tft_01"], "Region": ["World"], "R2 Score (pooled)": [0.98], "Sample Size": [10]})
    table.to_csv(root / "metrics" / "performance_by_region.csv", index=False)
    table.to_csv(root / "metrics" / "performance_train_by_region.csv", index=False)

    assert run.has_metrics_by_region()
    pd.testing.assert_frame_equal(run.load_metrics_by_region(), table)
    assert run.has_metrics_by_region("performance_train.csv")
    assert not run.has_metrics_by_region("performance_val.csv")


def test_the_filename_rule_matches_the_writer(store):
    run, _ = store

    for name in ("performance.csv", "performance_train.csv"):
        assert run._metrics_by_region_path(name).name == by_region_filename(name)
