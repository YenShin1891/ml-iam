"""Turning wall-clock from mismatched GPUs into one defensible number.

The arithmetic is trivial; what these pin down is the refusal to do it when
the measurement says a single factor does not describe a machine.
"""
import pytest

from scripts.calibrate_search_hosts import (
    _COST_PROXY,
    calibration_configs,
    config_label,
    host_factors,
    normalised_hours,
)


def calib(host, gpu, timings):
    return [
        {"host": host, "gpu_name": gpu, "config": config, "wall_seconds": seconds}
        for config, seconds in timings.items()
    ]


REFERENCE = calib("ref", "2080 Ti", {"c1": 100.0, "c2": 200.0, "c3": 400.0})


# --------------------------------------------------------------------------
# Choosing the configurations
# --------------------------------------------------------------------------

def test_the_configurations_span_the_cost_range():
    """One configuration would give a factor but no way to tell if it holds."""
    picked = calibration_configs("tft", 3)
    costs = [_COST_PROXY["tft"](params) for params in picked]
    assert costs == sorted(costs)
    assert costs[0] < costs[-1]


def test_the_extremes_of_the_space_are_always_included():
    from scripts.merge_search_ledgers import load_space

    every_cost = sorted(_COST_PROXY["tft"](p) for p in load_space("tft").sample())
    costs = [_COST_PROXY["tft"](p) for p in calibration_configs("tft", 3)]
    assert costs[0] == every_cost[0]
    assert costs[-1] == every_cost[-1]


def test_every_machine_picks_the_same_configurations():
    assert calibration_configs("tft", 3) == calibration_configs("tft", 3)


def test_one_configuration_cannot_check_itself():
    with pytest.raises(ValueError):
        calibration_configs("tft", 1)


def test_a_model_with_no_cost_ordering_is_refused():
    with pytest.raises(ValueError):
        calibration_configs("lstm", 3)


# --------------------------------------------------------------------------
# Labelling
# --------------------------------------------------------------------------

def test_a_configuration_is_named_by_what_it_is_not_where_it_ranked():
    """Two co-authors passing different --configs counts rank the same model
    differently; a ratio between differently-ranked labels would compare two
    different models and look perfectly reasonable doing it."""
    shared = calibration_configs("tft", 3)[0]
    assert config_label("tft", shared) == config_label("tft", calibration_configs("tft", 5)[0])


def test_different_configurations_get_different_names():
    labels = {config_label("tft", params) for params in calibration_configs("tft", 3)}
    assert len(labels) == 3


# --------------------------------------------------------------------------
# Factors
# --------------------------------------------------------------------------

def test_a_machine_is_expressed_relative_to_the_reference():
    rows = REFERENCE + calib("fast", "A100", {"c1": 50.0, "c2": 100.0, "c3": 200.0})
    factors = host_factors(rows, "ref")
    assert factors["ref"]["factor"] == 1.0
    assert factors["fast"]["factor"] == 0.5


def test_a_consistent_machine_gets_a_usable_factor():
    rows = REFERENCE + calib("fast", "A100", {"c1": 55.0, "c2": 108.0, "c3": 224.0})
    entry = host_factors(rows, "ref")["fast"]
    assert entry["scalar_holds"]
    assert entry["spread"] < 0.2


def test_a_machine_whose_ratio_drifts_with_model_size_gets_none():
    """The failure a single calibration config hides entirely: 0.95x on a
    small model and 0.55x on a large one averages to a factor wrong at both
    ends of the space the search actually covers."""
    rows = REFERENCE + calib("drifty", "3090", {"c1": 95.0, "c2": 144.0, "c3": 220.0})
    entry = host_factors(rows, "ref")["drifty"]
    assert not entry["scalar_holds"]
    assert entry["spread"] > 0.2


def test_a_machine_that_skipped_a_configuration_says_so():
    rows = REFERENCE + calib("partial", "A100", {"c1": 50.0, "c2": 100.0})
    entry = host_factors(rows, "ref")["partial"]
    assert entry["missing"] == ["c3"]
    assert len(entry["ratios"]) == 2


def test_a_reference_nobody_measured_is_an_error():
    with pytest.raises(KeyError):
        host_factors(REFERENCE, "gpu-nobody-has")


# --------------------------------------------------------------------------
# Normalising a search
# --------------------------------------------------------------------------

def make_factors(**hosts):
    return {
        host: {"factor": factor, "scalar_holds": holds, "gpu": None, "ratios": {}, "spread": 0.0, "missing": []}
        for host, (factor, holds) in hosts.items()
    }


def test_trials_are_restated_in_reference_hours():
    factors = make_factors(ref=(1.0, True), fast=(0.5, True))
    rows = [
        {"host": "ref", "wall_seconds": 3600.0},
        {"host": "fast", "wall_seconds": 3600.0},  # an hour there is two here
    ]
    hours, excluded = normalised_hours(rows, factors)
    assert hours == 3.0
    assert excluded == {}


def test_a_machine_without_a_valid_factor_is_left_out_not_guessed():
    """It is reported separately in raw hours, so a reader can see exactly
    what the normalised total does and does not cover."""
    factors = make_factors(ref=(1.0, True), drifty=(0.72, False))
    rows = [{"host": "ref", "wall_seconds": 3600.0}, {"host": "drifty", "wall_seconds": 7200.0}]

    hours, excluded = normalised_hours(rows, factors)
    assert hours == 1.0
    assert excluded == {"drifty": 2.0}


def test_a_host_with_no_calibration_at_all_is_left_out_too():
    rows = [{"host": "stranger", "wall_seconds": 3600.0}]
    hours, excluded = normalised_hours(rows, make_factors(ref=(1.0, True)))
    assert hours == 0.0
    assert excluded == {"stranger": 1.0}


def test_a_trial_that_recorded_no_time_is_skipped_silently():
    """Rows from ledgers written before timing existed; they are not zero-cost."""
    factors = make_factors(ref=(1.0, True))
    rows = [{"host": "ref", "wall_seconds": None}, {"host": "ref", "wall_seconds": 3600.0}]
    hours, excluded = normalised_hours(rows, factors)
    assert hours == 1.0
    assert excluded == {}
