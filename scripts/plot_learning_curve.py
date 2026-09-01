#!/usr/bin/env python

"""Learning curve (train/val loss vs. epoch) for a trained LSTM/TFT run.

Reads and plots results/<model>/<run_id>/final/logs/metrics.csv, which
Lightning's CSVLogger already writes during final training (see
src/trainers/lstm_trainer.py:train_final_lstm and
src/trainers/tft_trainer.py:train_final_tft) -- no retraining needed.

Note on how the curve is built: Lightning's CSVLogger writes one row per
logged step, with train/val metrics populating different rows sparsely
(val_loss is usually already one value per epoch; train_loss is often
logged per training step). This script collapses each metric to one point
per epoch by taking the last logged value within that epoch -- a standard
approximation for this log format, not a true per-epoch mean for train_loss.

XGBoost not covered here (no per-round eval history is persisted yet).

Usage:
  python scripts/plot_learning_curve.py --model lstm --run_id lstm_01
  python scripts/plot_learning_curve.py --model tft  --run_id tft_01 --log-scale
"""

import argparse
import os

import pandas as pd


def _load_metrics_csv(run_root: str) -> pd.DataFrame:
    path = os.path.join(run_root, "final", "logs", "metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No Lightning CSVLogger metrics found at {path}. Run the train phase first."
        )
    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        raise KeyError(f"'epoch' column not found in {path}; columns={list(df.columns)}")
    return df


def _epoch_curve(df: pd.DataFrame, column: str):
    """Collapse a sparsely-logged Lightning CSV column to one value per epoch."""
    if column not in df.columns:
        return None
    sub = df[["epoch", column]].dropna(subset=[column])
    if sub.empty:
        return None
    return sub.groupby("epoch")[column].last()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=["lstm", "tft"], help="Model type.")
    parser.add_argument("--run_id", required=True, help="Trained run id, e.g. lstm_01 / tft_01.")
    parser.add_argument("--output", type=str, default=None, help="Optional PNG path (default: <run_root>/plots/learning_curve.png).")
    parser.add_argument("--log-scale", action="store_true", help="Plot loss on a log y-axis.")
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.utils.utils import get_run_root
    run_root = get_run_root(args.run_id)

    df = _load_metrics_csv(run_root)
    train_curve = _epoch_curve(df, "train_loss")
    val_curve = _epoch_curve(df, "val_loss")

    if train_curve is None and val_curve is None:
        raise ValueError(
            f"Neither train_loss nor val_loss found as logged columns in metrics.csv "
            f"(columns={list(df.columns)})."
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    if train_curve is not None:
        ax.plot(train_curve.index, train_curve.values, label="train_loss", marker="o", markersize=3)
    if val_curve is not None:
        ax.plot(val_curve.index, val_curve.values, label="val_loss", marker="o", markersize=3)
    if args.log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{args.model.upper()} learning curve ({args.run_id})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_path = args.output or os.path.join(run_root, "plots", "learning_curve.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved learning curve to {output_path}")

    if val_curve is not None and len(val_curve) >= 2:
        best_epoch = val_curve.idxmin()
        last_epoch = val_curve.index.max()
        print(
            f"\nBest val_loss at epoch {best_epoch} ({val_curve.min():.4f}); "
            f"training stopped at epoch {last_epoch} ({val_curve.iloc[-1]:.4f})."
        )
        if best_epoch == last_epoch:
            tail = val_curve.loc[val_curve.index >= max(val_curve.index.min(), last_epoch - 4)]
            if len(tail) >= 2 and tail.is_monotonic_decreasing:
                print("-> val_loss was still improving when training stopped; "
                      "consider a larger max_epochs/patience.")
            else:
                print("-> val_loss plateaued near the end of training -- consistent with convergence.")
        else:
            gap = last_epoch - best_epoch
            print(f"-> Early stopping restored the epoch-{best_epoch} checkpoint after {gap} epoch(s) "
                  "without improvement -- consistent with catching overfitting before it set in.")


if __name__ == "__main__":
    main()
