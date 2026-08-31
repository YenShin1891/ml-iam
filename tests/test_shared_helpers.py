"""Helpers that were duplicated across modules, now with one definition each."""

import pandas as pd
import pytest

from configs.data import REGION_SCALE_ORDER
from src.utils.regions import (
    SCALE_ORDER_COARSEST_FIRST,
    group_regions_by_scale,
    region_scale,
    region_scales,
    regions_ordered_by_scale,
    scale_of_frame,
)
from src.utils.utils import is_primary_rank

REGIONS = ["World", "R10AFRICA", "R5ASIA", "KOR", "R6WEU", "USA"]


# ── region scales ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "region,expected",
    [
        ("World", "World"),
        ("R10AFRICA", "R10"),
        ("R5ASIA", "R5"),
        ("R6WEU", "R6"),
        ("KOR", "ISO3"),
        ("EU", "ISO3"),
    ],
)
def test_region_scale(region, expected):
    assert region_scale(region) == expected


def test_region_scales_is_vectorised():
    assert region_scales(REGIONS).tolist() == ["World", "R10", "R5", "ISO3", "R6", "ISO3"]


def test_scale_of_frame_prefers_the_persisted_column():
    """Region_Scale is written by data processing; don't re-derive it."""
    frame = pd.DataFrame({"Region": ["KOR"], "Region_Scale": ["CUSTOM"]})

    assert scale_of_frame(frame).tolist() == ["CUSTOM"]
    assert scale_of_frame(frame.drop(columns="Region_Scale")).tolist() == ["ISO3"]
    assert scale_of_frame(pd.DataFrame({"other": [1]})) is None


def test_grouping_uses_the_configured_scale_names():
    """The metrics writer used to emit "ISO" for what everything else calls ISO3."""
    grouped = group_regions_by_scale(REGIONS)

    assert list(grouped) == [s for s in SCALE_ORDER_COARSEST_FIRST if s in grouped]
    assert grouped["ISO3"] == ["KOR", "USA"]
    assert set(grouped) <= set(REGION_SCALE_ORDER)


def test_ordering_puts_countries_first():
    assert regions_ordered_by_scale(REGIONS) == [
        "KOR", "USA", "R10AFRICA", "R6WEU", "R5ASIA", "World",
    ]


def test_ordering_keeps_every_region_exactly_once():
    ordered = regions_ordered_by_scale(REGIONS + ["KOR"])

    assert sorted(ordered) == sorted(set(REGIONS))


# ── DDP rank ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("var", ["LOCAL_RANK", "PL_TRAINER_GLOBAL_RANK", "GLOBAL_RANK", "RANK"])
def test_non_zero_rank_is_not_primary(monkeypatch, var):
    monkeypatch.setenv(var, "1")

    assert is_primary_rank() is False


def test_rank_zero_and_unset_are_primary(monkeypatch):
    for var in ("LOCAL_RANK", "PL_TRAINER_GLOBAL_RANK", "GLOBAL_RANK", "RANK"):
        monkeypatch.delenv(var, raising=False)
    assert is_primary_rank() is True

    monkeypatch.setenv("LOCAL_RANK", "0")
    assert is_primary_rank() is True
