"""What a trial row carries, so it survives being run on someone else's machine.

None of this is recoverable after the fact: a row saying val_loss=0.31 has no
trace of the dataset it was measured on or the hours it cost, so the fields go
in as the row is written or not at all.
"""
import pytest

pytest.importorskip("lightning")

import src.trainers.tft_trainer as tft_trainer
from src.trainers import provenance
from src.trainers.search import Shard, is_completed_trial

KEYS = tft_trainer._SEARCH_PARAM_KEYS
PARAMS = {
    "hidden_size": 64,
    "lstm_layers": 1,
    "dropout": 0.1,
    "learning_rate": 0.001,
    "encoder_length": 2,
}


@pytest.fixture(autouse=True)
def stub_provenance(monkeypatch):
    """Pin the environment-dependent fields so the tests are about the shape."""
    tft_trainer._provenance.cache_clear()
    monkeypatch.setattr(provenance, "host_name", lambda: "gpu-a")
    monkeypatch.setattr(provenance, "gpu_name", lambda: "RTX 2080 Ti")
    monkeypatch.setattr(provenance, "git_commit", lambda: "abc1234")
    monkeypatch.setattr(provenance, "dataset_version", lambda run_id: "pipeline-2026-09-03")
    yield
    tft_trainer._provenance.cache_clear()


def make_row(**overrides):
    kwargs = dict(
        run_id="tft_99", trial_id="stage1_t0", signature="sig", stage="stage1",
        status="completed", wall_seconds=612.4,
        val_loss=0.31, best_epoch=9, epochs_run=25,
    )
    kwargs.update(overrides)
    return tft_trainer._trial_row(PARAMS, **kwargs)


# --------------------------------------------------------------------------
# The row
# --------------------------------------------------------------------------

def test_a_row_says_where_and_from_what_it_came():
    row = make_row()
    assert row["host"] == "gpu-a"
    assert row["gpu_name"] == "RTX 2080 Ti"
    assert row["git_commit"] == "abc1234"
    assert row["dataset_version"] == "pipeline-2026-09-03"


def test_a_row_says_what_it_cost_and_what_it_bought():
    """Wall time alone is not comparable across trials: early stopping means
    two trials of the same budget can differ severalfold, so the epochs the
    time actually paid for are recorded beside it."""
    row = make_row()
    assert row["wall_seconds"] == 612.4
    assert row["epochs_run"] == 25


def test_a_finished_row_still_ranks():
    assert is_completed_trial(make_row(), KEYS)


def test_the_searched_parameters_survive_the_round_trip():
    row = make_row()
    assert {key: row[key] for key in KEYS} == PARAMS


# --------------------------------------------------------------------------
# Failed trials
# --------------------------------------------------------------------------

def test_a_trial_that_ran_out_of_memory_is_recorded_not_forgotten():
    """The whole point: without a row, a configuration that OOMed looks
    exactly like one nobody has started, and a search that evaluated 47 of
    its 50 configurations reports as complete."""
    row = make_row(status="oom", val_loss=None, best_epoch=None, epochs_run=None)
    assert row["status"] == "oom"
    assert "val_loss" not in row
    assert row["wall_seconds"] == 612.4, "the GPU time it burned still counts"


def test_a_failed_trial_never_enters_a_ranking():
    row = make_row(status="oom", val_loss=None, best_epoch=None, epochs_run=None)
    assert not is_completed_trial(row, KEYS)


def test_a_failed_trial_is_still_retryable():
    """It must not satisfy the resume plan, or the configuration would be
    written off on the strength of one machine it did not fit on."""
    from src.trainers.search import completed_trials

    row = make_row(status="oom", val_loss=None, best_epoch=None, epochs_run=None)
    assert completed_trials([row], KEYS) == []


# --------------------------------------------------------------------------
# Reading the shard from the environment
# --------------------------------------------------------------------------

def test_no_env_var_means_the_whole_search(monkeypatch):
    monkeypatch.delenv("SEARCH_SHARD", raising=False)
    assert tft_trainer._search_shard() is None


def test_the_shard_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("SEARCH_SHARD", "1/3")
    assert tft_trainer._search_shard() == Shard(1, 3)


def test_a_shard_of_one_is_treated_as_no_shard(monkeypatch):
    """So a single-machine run takes exactly the path it took before."""
    monkeypatch.setenv("SEARCH_SHARD", "0/1")
    assert tft_trainer._search_shard() is None


def test_a_malformed_shard_stops_the_search(monkeypatch):
    monkeypatch.setenv("SEARCH_SHARD", "1/")
    with pytest.raises(ValueError):
        tft_trainer._search_shard()


# --------------------------------------------------------------------------
# Provenance gathering itself
# --------------------------------------------------------------------------

def test_a_missing_run_record_does_not_sink_a_search(monkeypatch, tmp_path):
    """Provenance is worth having, never worth crashing a day of trials for."""
    import src.utils.utils as utils_module

    monkeypatch.undo()  # this one is about the real reader, not the stub
    monkeypatch.setattr(utils_module, "get_run_root", lambda run_id: str(tmp_path / run_id))
    assert provenance.dataset_version("tft_does_not_exist") is None


# ── what counts as a dirty tree ────────────────────────────────────────────


def _repo(tmp_path):
    import subprocess

    def git(*args):
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
            cwd=tmp_path, check=True, capture_output=True,
        )

    (tmp_path / "configs" / "runs").mkdir(parents=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "configs" / "runs" / "tft.yaml").write_text("model: tft\n")
    (tmp_path / "src" / "code.py").write_text("x = 1\n")
    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    return tmp_path


def test_editing_a_run_config_does_not_make_the_code_dirty(tmp_path, monkeypatch):
    """Run configs are per-machine settings and are recorded in meta/ anyway."""
    monkeypatch.undo()  # the autouse stub replaced git_commit; this is about the real one
    root = _repo(tmp_path)
    (root / "configs" / "runs" / "tft.yaml").write_text("model: tft\nsearch_shard: '1/3'\n")
    (root / "configs" / "runs" / "apple.yaml").write_text("model: tft\n")

    assert not provenance.git_commit(str(root)).endswith("-dirty")


def test_editing_code_still_does(tmp_path, monkeypatch):
    monkeypatch.undo()
    root = _repo(tmp_path)
    (root / "src" / "code.py").write_text("x = 2\n")

    assert provenance.git_commit(str(root)).endswith("-dirty")
