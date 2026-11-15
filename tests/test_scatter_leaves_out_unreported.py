"""A value the IAM did not report is a fill value: no point is drawn for it."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.trajectories import create_single_scatter_plot


def test_unreported_elements_are_not_drawn():
    frame = pd.DataFrame({"Year": [2020, 2020, 2030, 2030]})
    y = np.array([1.0, 0.0, 3.0, 0.0])          # 0.0 where nothing was reported
    preds = np.array([1.1, 5.0, 2.9, 7.0])
    observed = np.array([1, 0, 1, 0])

    fig, ax = plt.subplots()
    create_single_scatter_plot(ax, frame, y, preds, 0, ["A"], "Model", ["unit"], observed_mask_col=observed)
    drawn = np.vstack([c.get_offsets() for c in ax.collections])
    plt.close(fig)

    assert sorted(drawn[:, 0]) == [1.0, 3.0]
    assert 5.0 not in drawn[:, 1] and 7.0 not in drawn[:, 1]
