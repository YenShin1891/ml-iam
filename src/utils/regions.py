"""Region aggregation scales (World, R5, R6, R10, ISO3).

The scale of a region is decided by its label prefix, configured once in
``configs.data.REGION_SCALE_PREFIXES``.  Everything that needs to bucket
regions goes through here rather than re-deriving the prefix rules, which had
drifted: the metrics writer labelled country-level rows "ISO" while the
``Region_Scale`` column produced by data processing calls them "ISO3".
"""

import logging
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from configs.data import (
    REGION_SCALE_DEFAULT,
    REGION_SCALE_ORDER,
    REGION_SCALE_PREFIXES,
)

# Coarsest first, as configured; the reverse reads well in filter widgets,
# where the many country entries are the ones users reach for most.
SCALE_ORDER_COARSEST_FIRST: List[str] = list(REGION_SCALE_ORDER)
SCALE_ORDER_FINEST_FIRST: List[str] = list(reversed(REGION_SCALE_ORDER))


def region_scale(region) -> str:
    """The aggregation scale a single region label belongs to."""
    label = str(region)
    for prefix, scale in REGION_SCALE_PREFIXES:
        if label.startswith(prefix):
            return scale
    return REGION_SCALE_DEFAULT


def region_scales(regions: Iterable) -> pd.Series:
    """``region_scale`` over an iterable or Series of labels."""
    series = regions if isinstance(regions, pd.Series) else pd.Series(list(regions))
    return series.astype(str).map(region_scale)


def scale_of_frame(frame: pd.DataFrame) -> Optional[pd.Series]:
    """Per-row scale for *frame*, preferring its persisted Region_Scale column.

    Returns None when the frame carries neither Region_Scale nor usable Region
    labels.  The sequence models encode Region to integer codes for their
    embeddings, and prefix matching on those codes would not fail -- it would
    quietly bucket every row as the default scale -- so numeric labels are
    refused rather than guessed at.
    """
    if "Region_Scale" in frame.columns:
        return frame["Region_Scale"].astype(str)
    if "Region" in frame.columns:
        if pd.api.types.is_numeric_dtype(frame["Region"]):
            logging.warning(
                "Region is integer-coded and Region_Scale is absent, so rows cannot "
                "be bucketed by scale; pass a frame that kept its region labels."
            )
            return None
        return region_scales(frame["Region"])
    return None


def group_regions_by_scale(
    regions: Iterable,
    order: Optional[Sequence[str]] = None,
) -> Dict[str, List[str]]:
    """Unique region labels bucketed by scale, keyed in *order*.

    Scales with no regions are omitted.
    """
    order = list(order) if order is not None else SCALE_ORDER_COARSEST_FIRST

    buckets: Dict[str, List[str]] = {scale: [] for scale in order}
    for label in dict.fromkeys(str(r) for r in regions):
        buckets.setdefault(region_scale(label), []).append(label)

    return {scale: labels for scale, labels in buckets.items() if labels}


def regions_ordered_by_scale(
    regions: Iterable,
    order: Optional[Sequence[str]] = None,
) -> List[str]:
    """Unique region labels sorted by scale, preserving order within a scale."""
    grouped = group_regions_by_scale(
        regions, order if order is not None else SCALE_ORDER_FINEST_FIRST
    )
    return [label for labels in grouped.values() for label in labels]
