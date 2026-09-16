"""The run config's `search_shard` key: how a machine learns which slice it runs.

It travels the route CUDA_VISIBLE_DEVICES already takes -- config to env to
phase subprocess -- because which slice a machine runs is a fact about the
machine, not about the model.
"""
import json

import pytest

import src.utils.utils as utils_module
from scripts import train_from_config


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(utils_module, "get_run_root", lambda run_id: str(tmp_path / run_id))
    return tmp_path


@pytest.fixture
def phase_envs(monkeypatch):
    """The env each phase subprocess would have been launched with."""
    seen = []
    monkeypatch.setattr(
        train_from_config, "_run_phase",
        lambda cfg, **kw: seen.append((kw["phase"], dict(kw["base_env"]))),
    )
    return seen


def _run(tmp_path, body, name="run.yaml"):
    path = tmp_path / name
    path.write_text(body)
    return ["--run", str(path)]


BASE = """
model: tft
run_id: tft_99
phases: [search]
dataset: pipeline-2026-09-03
"""


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def test_both_spellings_mean_the_same_shard():
    assert train_from_config._parse_search_shard("2/4") == "2/4"
    assert train_from_config._parse_search_shard({"index": 2, "count": 4}) == "2/4"


def test_no_key_means_the_whole_search():
    assert train_from_config._parse_search_shard(None) is None


@pytest.mark.parametrize("value", ["4/4", "-1/4", "2", {"index": 2}, {"index": 2, "count": 4, "of": 1}])
def test_a_bad_shard_fails_before_a_machine_spends_a_day_on_it(value):
    """The failure this catches is silent otherwise: an out-of-range index
    would run nothing, and a duplicated one would run someone else's slice."""
    with pytest.raises(ValueError):
        train_from_config._parse_search_shard(value)


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------

def test_the_shard_reaches_the_phase_subprocess(tmp_path, results, phase_envs):
    (tmp_path / "tft_99").mkdir()
    train_from_config.main(_run(tmp_path, BASE + 'search_shard: "1/3"\n'))
    assert phase_envs[0][1]["SEARCH_SHARD"] == "1/3"


def test_without_the_key_no_shard_is_inherited_from_the_shell(tmp_path, results, phase_envs, monkeypatch):
    """A stale export left over from an earlier machine's run would otherwise
    silently restrict a search meant to be run whole."""
    monkeypatch.setenv("SEARCH_SHARD", "1/3")
    (tmp_path / "tft_99").mkdir()
    train_from_config.main(_run(tmp_path, BASE))
    assert "SEARCH_SHARD" not in phase_envs[0][1]


def test_the_shard_is_recorded_with_the_run(tmp_path, results, phase_envs):
    train_from_config.main(_run(tmp_path, """
model: tft
phases: [preprocess, search]
dataset: pipeline-2026-09-03
search_shard: {index: 2, count: 4}
"""))
    run_id = next(p.name for p in results.iterdir() if p.name.startswith("tft_"))
    resolved = json.loads((results / run_id / "meta" / "run_config.resolved.json").read_text())
    assert resolved["search_shard"] == "2/4"

    env = json.loads((results / run_id / "meta" / "env.snapshot.json").read_text())
    assert env["global"]["SEARCH_SHARD"] == "2/4"
