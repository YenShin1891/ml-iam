"""The shared search machinery: distributions, sampling, ranking, the protocol.

What matters here is that all three models get the *same* procedure, so most
of these assert properties the comparison rests on -- reproducibility, no
duplicate configurations, stage 2 deciding over stage 1 -- rather than the
particular ranges any one model happens to declare.
"""

import csv

import numpy as np
import pytest

from src.trainers.search import (
    Choice,
    IntLogUniform,
    IntUniform,
    LogUniform,
    SearchSpace,
    Uniform,
    best_trial,
    budget_curve,
    canonicalize_params,
    completed_trials,
    params_signature,
    plan_two_stage_search,
    rank_trials,
    select_top_k_signatures,
    write_search_report,
)


def _draws(dist, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    return [dist.rvs(rng) for _ in range(n)]


# ── distributions ─────────────────────────────────────────────────────────


def test_log_uniform_spreads_evenly_across_decades():
    """The point of the change: a linear draw would put ~90% in the top decade."""
    draws = _draws(LogUniform(1e-4, 1e-1))

    per_decade = [sum(1 for d in draws if 10.0 ** -e > d >= 10.0 ** -(e + 1)) for e in (1, 2, 3)]

    assert min(draws) >= 1e-4 and max(draws) <= 1e-1
    for count in per_decade:
        assert abs(count - len(draws) / 3) < len(draws) * 0.1


def test_log_uniform_rejects_a_non_positive_lower_bound():
    with pytest.raises(ValueError, match="low > 0"):
        LogUniform(0.0, 1.0)


def test_uniform_and_int_uniform_cover_their_endpoints():
    assert set(_draws(IntUniform(4, 8))) == {4, 5, 6, 7, 8}
    draws = _draws(Uniform(0.0, 0.4))
    assert 0.0 <= min(draws) and max(draws) <= 0.4


def test_int_uniform_step_lands_only_on_the_grid():
    assert set(_draws(IntUniform(10, 20, step=5))) == {10, 15, 20}


def test_int_log_uniform_reaches_both_ends_and_favours_no_end():
    draws = _draws(IntLogUniform(1, 16))

    assert min(draws) == 1 and max(draws) == 16
    # Log-uniform: as many draws below 4 as above it.
    assert abs(sum(1 for d in draws if d < 4) - sum(1 for d in draws if d >= 4)) < len(draws) * 0.15


def test_int_log_uniform_multiple_of_snaps_to_the_grid():
    draws = _draws(IntLogUniform(32, 512, multiple_of=8))

    assert all(d % 8 == 0 for d in draws)
    assert min(draws) >= 32 and max(draws) <= 512


def test_cardinality_is_none_only_for_continuous_parameters():
    assert Uniform(0, 1).cardinality is None
    assert LogUniform(1e-3, 1).cardinality is None
    assert IntUniform(1, 5).cardinality == 5
    assert IntLogUniform(32, 64, multiple_of=8).cardinality == 5
    assert Choice([1, 2, 3]).cardinality == 3


def test_describe_reads_as_a_range_not_a_repr():
    assert LogUniform(1e-4, 0.05).describe() == "log-uniform[0.0001, 0.05]"
    assert IntLogUniform(32, 512, multiple_of=8).describe() == (
        "int-log-uniform[32, 512] (multiples of 8)"
    )
    assert Choice([1, 2, 3]).describe() == "{1, 2, 3}"


# ── sampling ──────────────────────────────────────────────────────────────


@pytest.fixture
def space():
    return SearchSpace(
        distributions={
            "learning_rate": LogUniform(1e-4, 1e-1),
            "depth": IntUniform(2, 10),
            "layers": Choice([1, 2, 3]),
        },
        n_trials=12,
        stage1_budget={"max_epochs": 5},
        stage2_top_k=3,
    )


def test_the_same_seed_gives_the_same_trials(space):
    assert space.sample() == space.sample()


def test_a_different_seed_gives_different_trials(space):
    assert space.sample(seed=1) != space.sample(seed=0)


def test_no_configuration_is_drawn_twice(space):
    keys = space.param_keys
    signatures = [params_signature(p, keys) for p in space.sample(40)]

    assert len(set(signatures)) == len(signatures)


def test_a_bare_list_is_accepted_as_a_discrete_parameter():
    space = SearchSpace(distributions={"layers": [1, 2, 3]}, n_trials=3)

    assert isinstance(space.distributions["layers"], Choice)
    assert {p["layers"] for p in space.sample()} == {1, 2, 3}


def test_asking_for_more_trials_than_the_space_holds_is_an_error():
    """Silently returning fewer would report a budget the search never ran."""
    space = SearchSpace(distributions={"layers": Choice([1, 2])}, n_trials=5)

    with pytest.raises(ValueError, match="space holds only 2"):
        space.sample()


def test_a_space_with_a_continuous_parameter_is_unbounded(space):
    assert space.cardinality is None


# ── trial bookkeeping ─────────────────────────────────────────────────────

KEYS = ("depth", "layers")


def _row(depth, layers, val_loss, stage="stage1", status="completed", **extra):
    return {
        "depth": depth, "layers": layers, "val_loss": val_loss,
        "stage": stage, "status": status, **extra,
    }


def test_only_rows_marked_completed_with_a_finite_metric_count():
    rows = [
        _row(2, 1, 0.5),
        _row(3, 1, 0.4, status="failed"),
        _row(4, 1, float("inf")),
        _row(5, 1, float("nan")),
        {"depth": 6, "val_loss": 0.1, "status": "completed"},  # missing 'layers'
    ]

    assert [r["depth"] for r in completed_trials(rows, KEYS)] == [2]


def test_signatures_survive_a_float_round_trip():
    """Ledgers go through JSON and CSV; a resumed search must still match."""
    original = {"depth": 3, "rate": 0.1 + 0.2}
    round_tripped = {"depth": 3.0, "rate": float(f"{0.1 + 0.2:.12g}")}

    assert params_signature(original, ("depth", "rate")) == params_signature(
        round_tripped, ("depth", "rate")
    )


def test_a_signature_needs_every_searched_parameter():
    with pytest.raises(ValueError, match="Missing search params"):
        params_signature({"depth": 3}, KEYS)


def test_canonicalize_drops_bookkeeping_and_keeps_only_the_searched_keys():
    canonical = canonicalize_params(_row(3, 2, 0.5, gpu="7"), KEYS)

    assert canonical == {"depth": 3, "layers": 2}


def test_ranking_orders_by_the_metric_in_the_requested_direction():
    rows = [_row(1, 1, 0.5), _row(2, 1, 0.1), _row(3, 1, 0.9)]

    assert [r["depth"] for r in rank_trials(rows)] == [2, 1, 3]
    assert [r["depth"] for r in rank_trials(rows, mode="max")] == [3, 1, 2]


def test_best_trial_raises_rather_than_returning_nothing():
    with pytest.raises(RuntimeError, match="No completed trials"):
        best_trial([])


def test_top_k_signatures_are_distinct_and_best_first():
    rows = [_row(1, 1, 0.5), _row(2, 1, 0.1), _row(1, 1, 0.5), _row(3, 1, 0.3)]

    top = select_top_k_signatures(rows, 2, KEYS)

    assert top == [params_signature(_row(2, 1, 0), KEYS), params_signature(_row(3, 1, 0), KEYS)]


def test_budget_curve_is_the_running_best_in_trial_order():
    rows = [
        _row(1, 1, 0.9, trial=0), _row(2, 1, 0.4, trial=1),
        _row(3, 1, 0.6, trial=2), _row(4, 1, 0.2, trial=3),
    ]

    assert budget_curve(rows) == [0.9, 0.4, 0.4, 0.2]


def test_budget_curve_rises_when_the_metric_is_maximised():
    rows = [_row(1, 1, -0.9, trial=0), _row(2, 1, -0.4, trial=1)]

    assert budget_curve(rows, mode="max") == [-0.9, -0.4]


# ── the two-stage protocol ────────────────────────────────────────────────


def test_a_fresh_search_owes_every_trial_to_stage_one(space):
    plan = plan_two_stage_search(space, [])

    assert len(plan.stage1_pending) == space.n_trials
    assert plan.stage2_pending == []
    assert (plan.stage1_done, plan.stage2_done) == (0, 0)


def test_a_resumed_search_skips_the_trials_already_on_disk(space):
    keys = space.param_keys
    drawn = space.sample()
    ledger = [
        {**params, "val_loss": 0.5 + i, "stage": "stage1", "status": "completed"}
        for i, params in enumerate(drawn[:4])
    ]

    plan = plan_two_stage_search(space, ledger)

    assert plan.stage1_done == 4
    assert len(plan.stage1_pending) == space.n_trials - 4
    pending = {params_signature(p, keys) for p in plan.stage1_pending}
    assert not pending & {params_signature(p, keys) for p in drawn[:4]}


def test_stage_two_is_planned_from_the_stage_one_leaders(space):
    drawn = space.sample()
    ledger = [
        {**params, "val_loss": float(i), "stage": "stage1", "status": "completed"}
        for i, params in enumerate(drawn)
    ]

    plan = plan_two_stage_search(space, ledger)

    assert plan.stage1_pending == []
    assert plan.stage2_pending == drawn[: space.stage2_top_k]


def test_a_trial_already_refit_is_not_refit_again(space):
    drawn = space.sample()
    ledger = [
        {**params, "val_loss": float(i), "stage": "stage1", "status": "completed"}
        for i, params in enumerate(drawn)
    ]
    ledger.append({**drawn[0], "val_loss": 0.01, "stage": "stage2", "status": "completed"})

    plan = plan_two_stage_search(space, ledger)

    assert plan.stage2_done == 1
    assert drawn[0] not in plan.stage2_pending
    assert plan.stage2_pending == drawn[1: space.stage2_top_k]


def test_a_failed_trial_is_retried_rather_than_counted(space):
    """A crashed trial that counted as explored would quietly shrink the budget."""
    drawn = space.sample()
    ledger = [{**drawn[0], "val_loss": float("inf"), "stage": "stage1", "status": "failed"}]

    plan = plan_two_stage_search(space, ledger)

    assert len(plan.stage1_pending) == space.n_trials


# ── the report ────────────────────────────────────────────────────────────


def test_the_report_writes_the_space_the_trials_and_the_budget_curve(space, tmp_path):
    rows = [
        _row(2, 1, 0.9, trial=0), _row(3, 2, 0.4, trial=1), _row(4, 3, 0.6, trial=2),
    ]

    written = write_search_report(str(tmp_path), space, rows)

    assert set(written) == {"search_space", "trials", "budget_curve"}
    with open(written["search_space"]) as handle:
        declared = {row["parameter"]: row["range"] for row in csv.DictReader(handle)}
    assert declared == {
        "depth": "int-uniform[2, 10]",
        "layers": "{1, 2, 3}",
        "learning_rate": "log-uniform[0.0001, 0.1]",
    }
    with open(written["budget_curve"]) as handle:
        curve = [float(row["best_val_loss"]) for row in csv.DictReader(handle)]
    assert curve == [0.9, 0.4, 0.4]


def test_the_trial_table_is_written_best_first(space, tmp_path):
    rows = [_row(2, 1, 0.9, trial=0), _row(3, 2, 0.4, trial=1)]

    written = write_search_report(str(tmp_path), space, rows)

    with open(written["trials"]) as handle:
        losses = [float(row["val_loss"]) for row in csv.DictReader(handle)]
    assert losses == [0.4, 0.9]


def test_a_search_with_no_trials_still_documents_its_space(space, tmp_path):
    written = write_search_report(str(tmp_path), space, [])

    assert set(written) == {"search_space"}


# ── the model spaces ──────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["LSTMSearchSpace", "TFTSearchSpace", "XGBSearchSpace"])
def test_every_model_declares_a_usable_space(name):
    import configs.models as models

    space = getattr(models, name)()

    assert isinstance(space, SearchSpace)
    assert space.n_trials >= space.stage2_top_k
    assert space.stage1_budget, "stage 1 must be cheaper than the full budget"
    assert len(space.sample()) == space.n_trials


def test_the_three_models_run_the_same_protocol():
    """The reviewer's point: the spaces may differ, the procedure may not."""
    import configs.models as models

    spaces = [models.LSTMSearchSpace(), models.TFTSearchSpace(), models.XGBSearchSpace()]

    assert len({s.n_trials for s in spaces}) == 1
    assert len({s.stage2_top_k for s in spaces}) == 1
    assert len({s.seed for s in spaces}) == 1


def test_no_model_searches_its_own_training_length():
    """Training length is fitted by early stopping, not searched, in all three."""
    import configs.models as models

    for space in (models.LSTMSearchSpace(), models.TFTSearchSpace(), models.XGBSearchSpace()):
        assert "num_boost_round" not in space.distributions
        assert "max_epochs" not in space.distributions


def test_xgb_ranges_are_continuous_where_the_parameter_is():
    import configs.models as models

    space = models.XGBSearchSpace()

    for name in ("eta", "reg_alpha", "reg_lambda"):
        assert isinstance(space.distributions[name], LogUniform), name
        assert space.distributions[name].cardinality is None


def test_lstm_keeps_only_structural_parameters_discrete():
    import configs.models as models

    discrete = {
        name for name, dist in models.LSTMSearchSpace().distributions.items()
        if isinstance(dist, Choice)
    }

    assert discrete == {"num_layers", "batch_size", "sequence_length"}


def test_the_budget_curve_covers_stage_one_only(space, tmp_path):
    """Stage 2 trains longer, so its scores would show search finding nothing."""
    rows = [
        _row(2, 1, 0.9, trial=0), _row(3, 2, 0.4, trial=1),
        _row(3, 2, 0.05, stage="stage2", trial=0),
    ]

    written = write_search_report(str(tmp_path), space, rows)

    with open(written["budget_curve"]) as handle:
        curve = [float(row["best_val_loss"]) for row in csv.DictReader(handle)]
    assert curve == [0.9, 0.4]


# ── stratified stage two ──────────────────────────────────────────────────


def _stratified_space():
    return SearchSpace(
        distributions={"depth": IntUniform(2, 10), "layers": Choice([1, 2, 3, 4])},
        n_trials=12,
        stage2_top_k=3,
        stage2_stratify_by="layers",
    )


def test_stratifying_gives_every_value_a_slot_before_the_leaders():
    """Otherwise all the stage-2 budget can land on one corner of the space."""
    # layers=1 sweeps the leaderboard; 2 and 3 appear only further down.
    rows = [
        _row(2, 1, 0.1), _row(3, 1, 0.2), _row(4, 1, 0.3),
        _row(5, 2, 0.8), _row(6, 3, 0.9),
    ]

    top = select_top_k_signatures(rows, 3, KEYS, stratify_by="layers")

    chosen = [r for r in rows if params_signature(r, KEYS) in top]
    assert {r["layers"] for r in chosen} == {1, 2, 3}


def test_stratifying_spends_leftover_slots_on_the_global_ranking():
    rows = [_row(2, 1, 0.1), _row(3, 1, 0.2), _row(4, 2, 0.8)]

    top = select_top_k_signatures(rows, 3, KEYS, stratify_by="layers")

    assert len(top) == 3
    assert top[0] == params_signature(_row(2, 1, 0), KEYS)


def test_stratifying_does_not_enlarge_the_stage_two_budget():
    """The allocation changes; the number of full-budget refits does not."""
    rows = [_row(i, i % 4 + 1, i / 10) for i in range(2, 10)]

    assert len(select_top_k_signatures(rows, 3, KEYS, stratify_by="layers")) == 3
    assert len(select_top_k_signatures(rows, 3, KEYS)) == 3


def test_the_plan_stratifies_stage_two_when_the_space_asks_for_it():
    space = _stratified_space()
    drawn = space.sample()
    # Rank so that one 'layers' value would otherwise take every slot.
    ledger = [
        {**params, "val_loss": float(i) if params["layers"] != 1 else -1.0 - i,
         "stage": "stage1", "status": "completed"}
        for i, params in enumerate(drawn)
    ]

    plan = plan_two_stage_search(space, ledger)

    assert len({p["layers"] for p in plan.stage2_pending}) > 1


def test_stratifying_by_an_unsearched_parameter_is_rejected():
    with pytest.raises(ValueError, match="not a searched parameter"):
        SearchSpace(
            distributions={"depth": IntUniform(2, 10)},
            n_trials=4,
            stage2_stratify_by="layers",
        )


def test_only_the_lstm_stratifies_and_it_does_so_by_context_length():
    import configs.models as models

    assert models.LSTMSearchSpace().stage2_stratify_by == "sequence_length"
    assert models.TFTSearchSpace().stage2_stratify_by is None
    assert models.XGBSearchSpace().stage2_stratify_by is None
