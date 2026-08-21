"""Third-party warnings this project deliberately silences.

Defined once so the in-process filters and the ``PYTHONWARNINGS`` value handed
to spawned workers cannot drift apart — sklearn emits the feature-name warning
per ``transform`` call, which floods a training log at hundreds of lines a
second when only some of the copies are installed.
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple

# (message regex, category name as PYTHONWARNINGS spells it)
SUPPRESSED: Tuple[Tuple[str, str], ...] = (
    (
        r"X does not have valid feature names, but StandardScaler was fitted with feature names",
        "UserWarning",
    ),
    (
        r"X has feature names, but StandardScaler was fitted without feature names",
        "UserWarning",
    ),
    # pytorch_forecasting repeats this for every short series in the dataset.
    (r"Min encoder length and/or min_prediction_idx", "UserWarning"),
)

_CATEGORIES = {"UserWarning": UserWarning}


def install() -> None:
    """Apply the filters to the current process."""
    for message, category in SUPPRESSED:
        warnings.filterwarnings(
            "ignore", message=message, category=_CATEGORIES[category]
        )


def as_pythonwarnings() -> List[str]:
    """The filters as ``PYTHONWARNINGS`` entries, for subprocesses.

    Commas separate entries in PYTHONWARNINGS, so a message containing one is
    truncated at the first comma.  Filter messages are matched as a regex
    anchored at the start, so the shortened prefix still catches the warning —
    and unlike passing the comma through, it does not split into one broken
    rule plus an "Invalid -W option" complaint at every worker's startup.
    """
    return [
        f"ignore:{message.split(',')[0]}:{category}"
        for message, category in SUPPRESSED
    ]


def export_to_environ(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Merge the filters into *env*'s PYTHONWARNINGS (default: os.environ)."""
    target = os.environ if env is None else env

    existing = [part for part in target.get("PYTHONWARNINGS", "").split(",") if part]
    for rule in as_pythonwarnings():
        if rule not in existing:
            existing.append(rule)

    target["PYTHONWARNINGS"] = ",".join(existing)
    return target
