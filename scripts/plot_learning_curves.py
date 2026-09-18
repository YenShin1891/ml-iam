#!/usr/bin/env python

"""Training vs. validation loss per epoch for the final LSTM and TFT runs.

One two-panel figure, plots/learning_curves.png: (a) LSTM, (b) TFT, each on
a log loss axis with a dashed line at the epoch whose checkpoint was kept
(lowest validation loss). Epochs are numbered from 1.

Both curves come from the Lightning CSVLogger file the train phase already
wrote, results/<model>/<run_id>/final/logs/metrics.csv -- nothing is
retrained. scripts/plot_learning_curve.py draws the same data for a single
run into that run's own plots/ folder; this one pairs two runs for a
side-by-side figure under <RESULTS_PATH>/_analysis/plots.

The two panels are NOT on a common scale, and the y labels say so:

  LSTM  masked MSE on standardized targets, averaged over targets.
  TFT   masked RMSE in native units, summed over the targets (MultiLoss),
        so it sits in the 1e4 range.

Read each panel for its shape (convergence, train/validation gap), not for
its level against the other.

Training loss per epoch: the TFT logs a true epoch mean
(train_loss_epoch). The LSTM only logs train_loss every 50 steps, so its
epoch value is the mean of those ~5 logged steps -- a noisier estimate,
which is why the LSTM training curve is the jagged one.

Usage:
  python scripts/plot_learning_curves.py --lstm lstm_01 --tft tft_01
  python scripts/plot_learning_curves.py --lstm lstm_01 --tft tft_01 --output_dir figs
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# model -> (panel title, y label)
PANELS = {
    "lstm": ("(a) LSTM", "Masked MSE (standardized)"),
    "tft": ("(b) TFT", "Masked RMSE (native units, summed)"),
}

TRAIN_COLOR = "#1f4e79"
VAL_COLOR = "#c0504d"


def epoch_curve(df: pd.DataFrame, column: str) -> pd.Series:
    """One value per epoch (1-based index) for a sparsely logged column.

    Prefers the ``<column>_epoch`` variant pytorch_forecasting writes for
    the TFT; otherwise averages whatever was logged within each epoch.
    """
    for name in (f"{column}_epoch", column):
        if name in df.columns:
            sub = df[["epoch", name]].dropna(subset=[name])
            if not sub.empty:
                curve = sub.groupby("epoch")[name].mean()
                curve.index = curve.index + 1
                return curve
    raise KeyError(f"{column} not found; columns={list(df.columns)}")


def load_curves(run_id: str):
    from src.utils.utils import get_run_root

    path = os.path.join(get_run_root(run_id), "final", "logs", "metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No CSVLogger metrics at {path}; run the train phase first.")
    df = pd.read_csv(path)
    return epoch_curve(df, "train_loss"), epoch_curve(df, "val_loss")


def draw_panel(ax, model: str, run_id: str):
    title, ylabel = PANELS[model]
    train, val = load_curves(run_id)

    ax.plot(train.index, train.values, color=TRAIN_COLOR, lw=1.5, label="Training")
    ax.plot(val.index, val.values, color=VAL_COLOR, lw=1.5, label="Validation")

    best = int(val.idxmin())
    ax.axvline(best, color="black", ls="--", lw=1)
    # Label sits left of the line in the empty band above the curves' tails.
    ax.annotate(
        f"best epoch {best}", xy=(best, 0.62), xycoords=("data", "axes fraction"),
        xytext=(-4, 0), textcoords="offset points", ha="right", va="center", fontsize=8,
    )

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", frameon=False, fontsize=8)

    print(f"{run_id}: best val {val.min():.4g} at epoch {best} of {int(val.index.max())}; "
          f"train there {train.get(best, float('nan')):.4g}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lstm", required=True, help="Trained LSTM run id, e.g. lstm_01.")
    parser.add_argument("--tft", required=True, help="Trained TFT run id, e.g. tft_01.")
    parser.add_argument("--output_dir", default=None,
                        help="Default: <RESULTS_PATH>/_analysis/plots.")
    args = parser.parse_args(argv)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    draw_panel(axes[0], "lstm", args.lstm)
    draw_panel(axes[1], "tft", args.tft)
    fig.tight_layout()

    from configs.paths import RESULTS_PATH

    output_dir = args.output_dir or os.path.join(RESULTS_PATH, "_analysis", "plots")
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "learning_curves.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
