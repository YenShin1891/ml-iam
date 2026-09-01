#!/usr/bin/env python

"""Train/val/test R2/RMSE/MAE breakdown for a trained LSTM/TFT run.

"train" is the exact split the model's weights were fit on, and "val" is
the split used only for early-stopping decisions (never used for a gradient
update) -- see src/trainers/tft_trainer.py:train_final_tft and
src/trainers/lstm_trainer.py:train_final_lstm. "train+val" is the
concatenation of the two. "test" is the held-out test split.

Each split is scored with the model's normal evaluation procedure --
batched windowed prediction for LSTM, single-window encoder/decoder forecast
for TFT -- same procedure used for the official test-set numbers.

XGBoost not covered here: its production model is fit on train+val merged
(src/trainers/xgb_trainer.py:train_and_save_model), so neither split is
genuinely held out from it.

Usage:
  python scripts/evaluate_splits.py --model tft  --run_id tft_01
  python scripts/evaluate_splits.py --model lstm --run_id lstm_01
  python scripts/evaluate_splits.py --model tft  --run_id tft_01 --two-window
"""

import argparse
import logging
import os

import pandas as pd


# ---------------------------------------------------------------------------
# Per-model split evaluation
# ---------------------------------------------------------------------------

def _eval_tft_split(splits, run_id, split_name, df, use_two_window):
    from src.data.preprocess import observed_mask_columns

    targets = splits["targets"]
    safe_name = split_name.replace("+", "_")
    session_state = dict(splits)
    session_state["test_data"] = df

    if use_two_window:
        # Two-window prediction manages its own metrics internally and does
        # not currently accept filename overrides, so it writes to the
        # default performance.csv/prediction_summary.json each call; we only
        # use its return value + session_state side effects here.
        from src.trainers.tft_two_window_simple import predict_tft_two_window
        y_pred = predict_tft_two_window(session_state, run_id)
    else:
        from src.trainers.tft_trainer import predict_tft
        y_pred = predict_tft(
            session_state,
            run_id,
            skip_metrics=True,
            metrics_filename=f"performance_split_{safe_name}.csv",
            prediction_summary_filename=f"prediction_summary_split_{safe_name}.json",
        )

    y_true = session_state["horizon_y_true"]
    horizon_df = session_state["horizon_df"]
    obs_cols = observed_mask_columns(targets)
    obs_mask = horizon_df[obs_cols].values if all(c in horizon_df.columns for c in obs_cols) else None
    return y_true, y_pred, obs_mask, len(horizon_df)


def _eval_lstm_split(base_session_state, run_id, split_name, df):
    from src.trainers.lstm_trainer import predict_lstm
    from src.data.preprocess import observed_mask_columns

    targets = base_session_state["targets"]
    safe_name = split_name.replace("+", "_")
    session_state = dict(base_session_state)
    session_state["test_data"] = df

    y_pred = predict_lstm(
        session_state, run_id,
        skip_metrics=True,
        metrics_filename=f"performance_split_{safe_name}.csv",
    )
    y_true = session_state["horizon_y_true"]
    horizon_df = session_state["horizon_df"]
    obs_cols = observed_mask_columns(targets)
    obs_mask = horizon_df[obs_cols].values if all(c in horizon_df.columns for c in obs_cols) else None
    return y_true, y_pred, obs_mask, len(horizon_df)


# ---------------------------------------------------------------------------
# Per-model split assembly
# ---------------------------------------------------------------------------

def _build_split_results_tft(store, run_id, use_two_window):
    from scripts.train_tft import derive_splits

    data = store.load_processed_data()
    splits = derive_splits(data, store)
    train_data, val_data, test_data = splits["train_data"], splits["val_data"], splits["test_data"]
    combined_data = pd.concat([train_data, val_data], ignore_index=True)

    frames = {"train": train_data, "val": val_data, "train+val": combined_data, "test": test_data}
    results = {}
    for split_name, df in frames.items():
        logging.info("Evaluating TFT on split: %s (%d rows)", split_name, len(df))
        results[split_name] = _eval_tft_split(splits, run_id, split_name, df, use_two_window)
    return splits["targets"], results


def _build_split_results_lstm(store, run_id):
    from scripts.train_lstm import derive_splits, _build_predict_state

    data = store.load_processed_data()
    splits = derive_splits(data, store)
    base_session_state = _build_predict_state(store, splits)

    train_data, val_data, test_data = splits["train_data"], splits["val_data"], splits["test_data"]
    combined_data = pd.concat([train_data, val_data], ignore_index=True)

    frames = {"train": train_data, "val": val_data, "train+val": combined_data, "test": test_data}
    results = {}
    for split_name, df in frames.items():
        logging.info("Evaluating LSTM on split: %s (%d rows)", split_name, len(df))
        results[split_name] = _eval_lstm_split(base_session_state, run_id, split_name, df)
    return splits["targets"], results


# ---------------------------------------------------------------------------
# Shared reporting
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=["lstm", "tft"], help="Model type.")
    parser.add_argument("--run_id", required=True, help="Trained run id, e.g. tft_01 / lstm_01.")
    parser.add_argument("--two-window", action="store_true", help="TFT only: use the two-window prediction path.")
    parser.add_argument("--output", type=str, default=None, help="Optional CSV path for the breakdown (default: <run_root>/metrics/train_val_test_r2_breakdown.csv).")
    args = parser.parse_args(argv)

    from src.utils.utils import setup_logging, get_run_root
    from src.utils.run_store import RunStore
    from src.trainers.evaluation import compute_r2_summary, per_target_r2_table

    setup_logging(args.run_id, log_file="evaluate_splits.log")
    store = RunStore(args.run_id)

    if args.model == "tft":
        targets, results = _build_split_results_tft(store, args.run_id, args.two_window)
    else:
        targets, results = _build_split_results_lstm(store, args.run_id)

    summaries = []
    per_target_frames = []
    for split_name, (y_true, y_pred, obs_mask, n_rows) in results.items():
        summary = compute_r2_summary(y_true, y_pred, observed_mask=obs_mask)
        summary["Split"] = split_name
        summary["Rows"] = n_rows
        summaries.append(summary)

        per_target = per_target_r2_table(y_true, y_pred, targets, observed_mask=obs_mask)
        per_target.insert(0, "Split", split_name)
        per_target_frames.append(per_target)

    summary_df = pd.DataFrame(summaries)[
        ["Split", "Rows", "Sample Size", "R2 (per-target avg)", "R2 (pooled)", "RMSE", "MAE"]
    ]
    per_target_df = pd.concat(per_target_frames, ignore_index=True)
    per_target_pivot = per_target_df.pivot(index="Output Variable", columns="Split", values="R2")
    ordered_cols = [c for c in ["train", "val", "train+val", "test"] if c in per_target_pivot.columns]
    per_target_pivot = per_target_pivot[ordered_cols]

    pd.set_option("display.width", 120)
    print(f"\n=== {args.model.upper()} R2/RMSE/MAE by split (overall) ===")
    print(summary_df.to_string(index=False))
    print(f"\n=== {args.model.upper()} R2 by split (per output variable) ===")
    print(per_target_pivot.to_string())

    train_r2 = summary_df.loc[summary_df["Split"] == "train", "R2 (per-target avg)"].item()
    val_r2 = summary_df.loc[summary_df["Split"] == "val", "R2 (per-target avg)"].item()
    test_r2 = summary_df.loc[summary_df["Split"] == "test", "R2 (per-target avg)"].item()
    gap_train_val = train_r2 - val_r2
    gap_train_test = train_r2 - test_r2

    print("\n=== Diagnosis ===")
    print(f"train R2={train_r2:.3f}  val R2={val_r2:.3f}  test R2={test_r2:.3f}")
    print(f"train-val gap={gap_train_val:+.3f}  train-test gap={gap_train_test:+.3f}")
    if train_r2 < 0.85 and gap_train_val < 0.05:
        print("-> Train R2 itself is low and close to val/test: consistent with UNDERFITTING "
              "(model lacks capacity/fit even on data it was trained on).")
    elif gap_train_val > 0.1 or gap_train_test > 0.1:
        print("-> Large train-val/train-test gap: consistent with OVERFITTING "
              "(model fits training data well but generalizes poorly).")
    else:
        print("-> No large train-val/train-test gap: performance gap alone does not point to "
              "over/underfitting; a low test R2 (if any) likely reflects a genuine capacity/architecture "
              "limitation rather than a generalization gap.")

    output_path = args.output or os.path.join(get_run_root(args.run_id), "metrics", "train_val_test_r2_breakdown.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged = summary_df.merge(
        per_target_df.pivot(index="Split", columns="Output Variable", values="R2").reset_index(),
        on="Split",
    )
    merged.to_csv(output_path, index=False)
    logging.info("Saved train/val/test R2 breakdown to %s", output_path)
    print(f"\nSaved breakdown to {output_path}")


if __name__ == "__main__":
    main()
