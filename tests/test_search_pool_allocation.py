"""Dividing the GPU pool between trial groups that cannot share a worker.

XGBoost and the TFT both search a context length, and that choice decides
which prepared dataset a trial trains on -- so a worker, which is handed one
dataset, can only take trials that agree about it.  The groups used to run
back to back.  That is free while every group is at least as wide as the
pool, which stage 1 satisfies and stage 2 does not: a top-10 splitting 4/6
across eight GPUs ran as two waves that never used more than six cards.
"""

import pytest

from src.trainers.search import allocate_pool


def waves(sizes, allocation):
    """Trials the busiest GPU has to run in sequence."""
    return max(-(-size // gpus) for size, gpus in zip(sizes, allocation) if size)


def test_the_stage_two_split_that_prompted_this_now_fits_in_one_pass_of_waves():
    """4 and 6 trials over 8 GPUs: two waves, not one wave then another."""
    sizes = [4, 6]

    allocation = allocate_pool(sizes, 8)

    assert sum(allocation) == 8
    assert waves(sizes, allocation) == 2
    # Back to back, each group waits for the other: two full waves in series.
    assert waves([4], [8]) + waves([6], [8]) == 2


def test_a_pool_wide_enough_gives_every_trial_its_own_gpu():
    sizes = [4, 6]

    allocation = allocate_pool(sizes, 10)

    assert allocation == [4, 6]
    assert waves(sizes, allocation) == 1


def test_no_group_is_given_more_gpus_than_it_has_trials():
    """A card held by an idle worker is one the other group could be using."""
    allocation = allocate_pool([1, 9], 8)

    assert allocation[0] == 1
    assert sum(allocation) == 8


def test_a_group_that_cannot_use_the_extra_cards_leaves_them_unallocated():
    allocation = allocate_pool([2, 2], 8)

    assert allocation == [2, 2]


def test_the_allocation_minimises_the_number_of_waves():
    """Checked against every split of the pool, not against a fixed answer."""
    from itertools import product

    for sizes in ([4, 6], [3, 5], [7, 1], [25, 25], [2, 3, 5], [11, 4]):
        pool = 8
        allocation = allocate_pool(sizes, pool)
        best = min(
            waves(sizes, split)
            for split in product(range(1, pool + 1), repeat=len(sizes))
            if sum(split) <= pool
        )
        assert waves(sizes, allocation) == best, (sizes, allocation)


def test_stage_one_is_unaffected_because_its_groups_already_fill_the_pool():
    """The old back-to-back arrangement was never wrong, only wasteful late on."""
    sizes = [25, 25]

    assert waves(sizes, allocate_pool(sizes, 8)) == 7
    # Back to back: ceil(25/8) each, run in series -- the same 8 waves of work,
    # so stage 1 loses at most the tail of one group.
    assert waves([25], [8]) + waves([25], [8]) == 8


def test_a_pool_too_narrow_to_seat_every_group_declines_to_allocate():
    """Two groups and one GPU: there is nothing to divide, so say so."""
    assert allocate_pool([4, 6], 1) == []
    assert allocate_pool([2, 3, 5], 2) == []


def test_an_empty_group_is_given_nothing():
    allocation = allocate_pool([0, 6], 8)

    assert allocation == [0, 6]


def test_a_stage_with_no_trials_allocates_nothing():
    assert allocate_pool([0, 0], 8) == [0, 0]


def test_a_single_group_takes_the_whole_pool():
    assert allocate_pool([10], 8) == [8]


def test_negative_sizes_are_refused():
    with pytest.raises(ValueError, match="non-negative"):
        allocate_pool([4, -1], 8)
