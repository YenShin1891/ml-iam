"""Splitting one search across machines.

The property everything else rests on: a shard is derived from the seed, so
machines that never talk to each other still agree on who runs what.
"""
import pytest

from src.trainers.search import (
    Choice,
    IntUniform,
    SearchSpace,
    Shard,
    params_signature,
    plan_two_stage_search,
)


def make_space(n_trials=12, top_k=4):
    return SearchSpace(
        distributions={"depth": IntUniform(1, 64), "width": Choice([8, 16, 32])},
        n_trials=n_trials,
        stage2_top_k=top_k,
        seed=0,
    )


def completed(space, params, stage="stage1", val_loss=0.5):
    return {
        **params,
        "signature": params_signature(params, space.param_keys),
        "status": "completed",
        "stage": stage,
        "val_loss": val_loss,
    }


# --------------------------------------------------------------------------
# Shard itself
# --------------------------------------------------------------------------

@pytest.mark.parametrize("index, count", [(-1, 3), (3, 3), (0, 0), (1, -2)])
def test_a_shard_outside_its_own_range_is_rejected(index, count):
    with pytest.raises(ValueError):
        Shard(index, count)


def test_the_shards_of_a_search_partition_it_exactly():
    items = list(range(50))
    shards = [Shard(i, 4).members(items) for i in range(4)]
    covered = [item for shard in shards for item in shard]
    assert sorted(covered) == items, "every configuration must be somebody's"
    assert len(covered) == len(set(covered)), "and nobody's twice"


def test_round_robin_keeps_the_shards_the_same_size():
    sizes = {len(Shard(i, 3).members(range(50))) for i in range(3)}
    # 50 does not divide by 3, so one shard carries the remainder -- but no
    # more than that, which is what contiguous blocks would not guarantee.
    assert max(sizes) - min(sizes) <= 1


def test_a_shard_of_one_is_the_whole_search():
    shard = Shard(0, 1)
    assert shard.is_whole
    assert shard.members(range(10)) == list(range(10))


def test_the_index_over_count_spelling_round_trips():
    assert Shard.parse("2/4") == Shard(2, 4)
    assert str(Shard(2, 4)) == "2/4"


@pytest.mark.parametrize("text", ["2", "2/4/6", "a/4", "2/b", ""])
def test_a_malformed_shard_string_is_rejected(text):
    with pytest.raises(ValueError):
        Shard.parse(text)


# --------------------------------------------------------------------------
# Sharded planning
# --------------------------------------------------------------------------

def test_each_machine_plans_only_its_own_slice():
    space = make_space()
    plans = [plan_two_stage_search(space, [], shard=Shard(i, 3)) for i in range(3)]

    keys = space.param_keys
    assigned = [
        {params_signature(p, keys) for p in plan.stage1_pending} for plan in plans
    ]
    everything = {params_signature(p, keys) for p in space.sample()}

    assert set().union(*assigned) == everything
    for i, mine in enumerate(assigned):
        for j, theirs in enumerate(assigned):
            if i != j:
                assert not mine & theirs, "two machines were handed the same trial"


def test_machines_derive_the_same_split_without_talking():
    space = make_space()
    once = plan_two_stage_search(space, [], shard=Shard(1, 3)).stage1_pending
    again = plan_two_stage_search(space, [], shard=Shard(1, 3)).stage1_pending
    assert once == again


def test_a_shards_assignment_does_not_move_as_other_machines_report():
    """The reason membership is positional in the sampled list, not the pending one.

    Shard 1 must keep the same work whether or not shard 0 has finished; a
    slice computed over what is still pending would slide onto trials another
    machine is already running.
    """
    space = make_space()
    mine_alone = plan_two_stage_search(space, [], shard=Shard(1, 3)).stage1_pending

    others = [
        completed(space, params)
        for params in Shard(0, 3).members(space.sample())
    ]
    mine_after = plan_two_stage_search(space, others, shard=Shard(1, 3)).stage1_pending

    assert mine_after == mine_alone


def test_a_shard_still_skips_what_it_already_finished():
    space = make_space()
    mine = plan_two_stage_search(space, [], shard=Shard(2, 3)).stage1_pending
    ledger = [completed(space, mine[0])]

    plan = plan_two_stage_search(space, ledger, shard=Shard(2, 3))
    assert plan.stage1_pending == mine[1:]
    assert plan.stage1_done == 1


def test_progress_counts_the_whole_search_not_the_slice():
    """A machine reading a merged ledger must not report the search finished.

    stage1_done comes from the ledger, which is everyone's; stage1_mine is
    this machine's share.  Confusing the two is how a quarter-done search
    gets declared complete.
    """
    space = make_space()
    everyone = [completed(space, params) for params in space.sample()]
    plan = plan_two_stage_search(space, everyone, shard=Shard(0, 3))

    assert plan.stage1_done == space.n_trials
    assert plan.stage1_mine == len(Shard(0, 3).members(space.sample()))
    assert plan.stage1_mine < plan.stage1_done


def test_stage_two_is_split_across_machines_too():
    space = make_space(top_k=4)
    sampled = space.sample()
    ledger = [
        completed(space, params, val_loss=0.1 * i) for i, params in enumerate(sampled)
    ]

    shortlists = [
        plan_two_stage_search(space, ledger, shard=Shard(i, 2)).stage2_pending
        for i in range(2)
    ]
    keys = space.param_keys
    signatures = [{params_signature(p, keys) for p in s} for s in shortlists]

    unsharded = plan_two_stage_search(space, ledger).stage2_pending
    assert signatures[0] | signatures[1] == {
        params_signature(p, keys) for p in unsharded
    }
    assert not signatures[0] & signatures[1]


def test_an_unsharded_plan_is_unchanged():
    space = make_space()
    assert plan_two_stage_search(space, []).stage1_pending == space.sample()
    assert plan_two_stage_search(space, []).shard is None
