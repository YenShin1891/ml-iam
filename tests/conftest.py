"""Shared fixtures for the ml-iam test suite."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def targets():
    return ["A", "B"]


@pytest.fixture
def prepared_frame(targets):
    """A small prepared frame: 20 groups x 4 years, two targets, one partially observed.

    Group 5 is the only one that ever reports Model_Family 'FAM_RARE', which is
    what makes per-split categorical encoding drift.
    """
    rows = []
    for g in range(20):
        for year in (2020, 2025, 2030, 2035):
            rows.append(
                {
                    "Model": f"M{g}",
                    "Scenario": f"S{g}",
                    "Region": "World" if g % 2 else f"R10_{g}",
                    "Model_Family": "FAM_RARE" if g == 5 else f"FAM_{g % 2}",
                    "Year": year,
                    "feat": float(g * 10 + year - 2020),
                    "A": float(g + year - 2020),
                    "B": float(g * 2 + year - 2020),
                }
            )
    frame = pd.DataFrame(rows)

    # B is unobserved for the first two years of every group.
    unobserved = frame["Year"] < 2030
    frame["A__observed"] = 1.0
    frame["B__observed"] = (~unobserved).astype(float)
    frame.loc[unobserved, "B"] = np.nan
    return frame
