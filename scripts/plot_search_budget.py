#!/usr/bin/env python

"""Best validation score against the number of stage-1 search trials.

One panel per run, read from the budget_curve.csv its search phase wrote
(results/<model>/<run_id>/search/). A marker and its trial number show each
trial at which the best score improved, so a flat tail reads as "more trials
would not have changed the selection".

The panels are not on a common scale: XGBoost is scored by the negative RMSE
of its autoregressive rollout (higher is better), the LSTM by its masked
validation MSE and the TFT by its validation RMSE summed over targets in
native units (lower is better).

Usage:
  python scripts/plot_search_budget.py --runs xgb_01 lstm_01 tft_01 --out search_budget.png
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PANELS = {
    "xgb": ("XGBoost", "Best validation score\n(−RMSE of rollout, standardized)"),
    "lstm": ("LSTM", "Best validation loss\n(masked MSE, standardized)"),
    "tft": ("TFT", "Best validation loss\n(RMSE, native units, summed)"),
}
LINE, MARK = "#1f4e79", "#c0504d"


def load_curve(run_id: str) -> pd.Series:
    """The running best per trial count, indexed from 1."""
    from src.utils.utils import get_run_root

    curve = pd.read_csv(os.path.join(get_run_root(run_id), "search", "budget_curve.csv"))
    return curve.set_index("n_trials").iloc[:, 0]


def improvements(curve: pd.Series) -> pd.Series:
    """The points of *curve* at which its value changed."""
    return curve[curve != curve.shift()]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="run ids, one panel each")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)

    fig, axes = plt.subplots(1, len(args.runs), figsize=(4 * len(args.runs), 3.6), squeeze=False)
    for letter, ax, run_id in zip("abcdefgh", axes[0], args.runs):
        name, ylabel = PANELS[run_id.split("_", 1)[0]]
        curve = load_curve(run_id)
        steps = improvements(curve)
        ax.step(curve.index, curve.values, where="post", color=LINE, linewidth=2)
        ax.plot(steps.index, steps.values, "o", color=MARK, markersize=5)
        previous = None
        for trial, value in steps.iloc[1:].items():
            # Two improvements a trial or two apart: put the second label below its marker.
            crowded = previous is not None and trial - previous <= 2
            ax.annotate(str(trial), (trial, value), xytext=(5, -10 if crowded else 3),
                        textcoords="offset points", color=MARK, fontsize=8)
            previous = None if crowded else trial
        ax.set_title(f"({letter}) {name}", loc="left", fontsize=11)
        ax.set_xlabel("Stage-1 trials")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlim(0, len(curve) + 1)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    print(f"saved {args.out}")
    for run_id in args.runs:
        print(run_id, "improved at trials", list(improvements(load_curve(run_id)).index))


if __name__ == "__main__":
    main()
