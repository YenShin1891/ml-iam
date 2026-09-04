"""Pooling the ledgers of a search several machines ran.

The merge is easy; the checking is the point.  These pin the two things that
would otherwise fail silently: rows that did not measure the same thing being
averaged together, and a configuration nobody scored being indistinguishable
from one that was.
"""
import json

import pytest

from scripts.merge_search_ledgers import (
    check_provenance,
    coverage,
    merge,
    read_ledger,
    timing_table,
)
from src.trainers.search import Choice, IntUniform, SearchSpace, params_signature


@pytest.fixture
def space():
    return SearchSpace(
        distributions={"depth": IntUniform(1, 64), "width": Choice([8, 16, 32])},
        n_trials=6,
        stage2_top_k=2,
        seed=0,
    )


def row(space, params, **overrides):
    base = {
        **params,
        "signature": params_signature(params, space.param_keys),
        "stage": "stage1",
        "status": "completed",
        "val_loss": 0.5,
        "host": "gpu-a",
        "gpu_name": "RTX 2080 Ti",
        "git_commit": "abc1234",
        "dataset_version": "pipeline-2026-09-03",
        "wall_seconds": 600.0,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# Merging
# --------------------------------------------------------------------------

def test_the_same_trial_run_twice_collapses_to_one(space):
    params = space.sample()[0]
    kept, dropped = merge(
        [row(space, params, host="gpu-a"), row(space, params, host="gpu-b")],
        space.param_keys,
    )
    assert len(kept) == 1
    assert len(dropped) == 1


def test_a_completed_trial_beats_the_machine_it_did_not_fit_on(space):
    """Order must not decide it: a larger GPU's score is not refuted by an OOM."""
    params = space.sample()[0]
    scored = row(space, params, host="big", val_loss=0.3)
    oomed = row(space, params, host="small", status="oom", val_loss=None)

    for ledger in ([oomed, scored], [scored, oomed]):
        kept, _ = merge(ledger, space.param_keys)
        assert [r["status"] for r in kept] == ["completed"]
        assert kept[0]["host"] == "big"


def test_the_two_stages_of_one_configuration_are_kept_apart(space):
    """They are scored under different budgets, so one must not evict the other."""
    params = space.sample()[0]
    kept, _ = merge(
        [row(space, params, stage="stage1"), row(space, params, stage="stage2")],
        space.param_keys,
    )
    assert {r["stage"] for r in kept} == {"stage1", "stage2"}


def test_duplicated_work_is_reported_rather_than_forgotten(space):
    params = space.sample()[0]
    _, dropped = merge(
        [row(space, params), row(space, params, wall_seconds=900.0)], space.param_keys
    )
    assert sum(r["wall_seconds"] for r in dropped) == 900.0


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

def test_ledgers_from_different_datasets_are_not_poolable(space):
    rows = [
        row(space, space.sample()[0], host="gpu-a"),
        row(space, space.sample()[1], host="gpu-b", dataset_version="pipeline-2026-04-13"),
    ]
    problems = check_provenance(rows)
    assert any("dataset_version" in problem for problem in problems)
    assert any("gpu-b" in problem for problem in problems)


def test_ledgers_from_different_commits_are_not_poolable(space):
    rows = [
        row(space, space.sample()[0]),
        row(space, space.sample()[1], git_commit="def5678"),
    ]
    assert any("git_commit" in problem for problem in check_provenance(rows))


def test_agreeing_ledgers_raise_nothing(space):
    rows = [row(space, params, host=f"gpu-{i}") for i, params in enumerate(space.sample())]
    assert check_provenance(rows) == []


def test_trials_run_from_uncommitted_edits_are_called_out(space):
    rows = [row(space, params, git_commit="abc1234-dirty") for params in space.sample()[:2]]
    assert any("uncommitted" in problem for problem in check_provenance(rows))


def test_a_ledger_written_before_provenance_existed_is_flagged(space):
    rows = [
        {k: v for k, v in row(space, params).items() if k != "dataset_version"}
        for params in space.sample()[:2]
    ]
    assert any("dataset_version" in problem for problem in check_provenance(rows))


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------

def test_a_trial_that_failed_is_not_a_trial_nobody_reached(space):
    sampled = space.sample()
    rows = [row(space, params) for params in sampled[:3]]
    rows.append(row(space, sampled[3], status="oom", val_loss=None))

    cover = coverage(space, rows)
    assert len(cover["completed"]) == 3
    assert len(cover["failed"]) == 1
    assert len(cover["not_started"]) == len(sampled) - 4
    assert cover["failed"][0]["status"] == "oom"


def test_full_coverage_leaves_nothing_outstanding(space):
    rows = [row(space, params) for params in space.sample()]
    cover = coverage(space, rows)
    assert len(cover["completed"]) == space.n_trials
    assert not cover["failed"] and not cover["not_started"]


def test_stage_two_rows_do_not_count_as_exploring_the_space(space):
    """Stage 2 refits configurations stage 1 already covered; counting them
    again would report a search as complete on a handful of trials."""
    rows = [row(space, params, stage="stage2") for params in space.sample()]
    assert coverage(space, rows)["completed"] == []


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------

def test_cost_is_reported_per_machine_and_never_summed(space):
    sampled = space.sample()
    rows = [
        row(space, sampled[0], host="a", gpu_name="2080 Ti", wall_seconds=3600.0),
        row(space, sampled[1], host="a", gpu_name="2080 Ti", wall_seconds=1800.0),
        row(space, sampled[2], host="b", gpu_name="A100", wall_seconds=600.0),
    ]
    table = {entry["host"]: entry for entry in timing_table(rows)}
    assert table["a"]["trials"] == 2
    assert table["a"]["wall_hours"] == 1.5
    assert table["b"]["gpu"] == "A100"
    assert set(table) == {"a", "b"}, "no row totals the two together"


def test_an_oom_trial_still_reports_the_time_it_burned(space):
    rows = [row(space, space.sample()[0], status="oom", val_loss=None, wall_seconds=120.0)]
    entry = timing_table(rows)[0]
    assert entry["trials"] == 1 and entry["completed"] == 0
    assert entry["wall_hours"] == round(120 / 3600, 2)


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def test_a_half_written_line_does_not_lose_the_rest_of_the_ledger(space, tmp_path, capsys):
    """A machine killed mid-write truncates its last line; the trials before
    it are still good, and the merge must not throw them away."""
    path = tmp_path / "trials.jsonl"
    good = [row(space, params) for params in space.sample()[:2]]
    path.write_text(
        json.dumps(good[0]) + "\n" + json.dumps(good[1]) + "\n" + '{"depth": 4, "wid',
        encoding="utf-8",
    )
    rows = read_ledger(path)
    assert len(rows) == 2
    assert "not valid JSON" in capsys.readouterr().err


def test_every_row_remembers_which_ledger_it_came_from(space, tmp_path):
    path = tmp_path / "trials.jsonl"
    path.write_text(json.dumps(row(space, space.sample()[0])) + "\n", encoding="utf-8")
    assert read_ledger(path)[0]["_source"] == str(path)
