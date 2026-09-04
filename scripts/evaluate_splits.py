#!/usr/bin/env python

"""Train/val/test R2/RMSE/MAE breakdown for a trained XGB/LSTM/TFT run.

"train" is the exact split the model was fit on, and "val" is the split used
only for early-stopping decisions (never for a gradient update or a boosting
round) -- see src/trainers/tft_trainer.py:train_final_tft,
src/trainers/lstm_trainer.py:train_final_lstm and
src/trainers/xgb_trainer.py:train_and_save_model. "train+val" is the
concatenation of the two. "test" is the held-out test split.

Each split is scored with the model's normal evaluation procedure -- batched
windowed prediction for LSTM, single-window encoder/decoder forecast for TFT,
autoregressive rollout for XGB -- the same procedure that produced the run's
official test-set numbers, so a split's row here is comparable with them.

Usage:
  python scripts/evaluate_splits.py --model tft  --run_id tft_01
  python scripts/evaluate_splits.py --model lstm --run_id lstm_01
  python scripts/evaluate_splits.py --model xgb  --run_id xgb_01
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
    """Score one split; metrics stay in memory so the run's own files are untouched."""
    from src.data.preprocess import observed_mask_from_frame

    targets = splits["targets"]
    safe_name = split_name.replace("+", "_")
    session_state = dict(splits)
    session_state["test_data"] = df

    if use_two_window:
        from src.trainers.tft_two_window_simple import predict_tft_two_window
        y_pred = predict_tft_two_window(session_state, run_id, skip_metrics=True)
    else:
        from src.trainers.tft_trainer import predict_tft
        y_pred = predict_tft(
            session_state,
            run_id,
            skip_metrics=True,
            prediction_summary_filename=f"prediction_summary_split_{safe_name}.json",
        )

    y_true = session_state["horizon_y_true"]
    horizon_df = session_state["horizon_df"]
    return y_true, y_pred, observed_mask_from_frame(horizon_df, targets), len(horizon_df)


def _eval_lstm_split(base_session_state, run_id, split_name, df):
    from src.trainers.lstm_trainer import predict_lstm
    from src.data.preprocess import observed_mask_from_frame

    targets = base_session_state["targets"]
    session_state = dict(base_session_state)
    session_state["test_data"] = df

    y_pred = predict_lstm(session_state, run_id, skip_metrics=True)
    y_true = session_state["horizon_y_true"]
    horizon_df = session_state["horizon_df"]
    return y_true, y_pred, observed_mask_from_frame(horizon_df, targets), len(horizon_df)


def _eval_xgb_split(bundle, split_name, X_with_index, y_scaled, frame):
    """Roll one split forward the way the test phase rolls the test set.

    One-step prediction off true lags would answer a different question: the
    reported numbers come from feeding each prediction back in as the next
    step's lag features, and the gap between the two is exactly the error
    compounding this breakdown is meant to expose.
    """
    import numpy as np

    from configs.data import POPULATION_COLUMN
    from src.data.preprocess import denormalize_by_population, observed_mask_from_frame
    from src.trainers.evaluation import test_xgb_autoregressively

    targets = bundle["targets"]
    # The rollout scatters each group's predictions by index label, so a
    # duplicate label would silently overwrite another row's prediction.
    if not X_with_index.index.is_unique:
        raise ValueError(f"Split {split_name!r} has duplicate index labels")

    preds_scaled = test_xgb_autoregressively(
        X_with_index, y_scaled,
        model=bundle["model"],
        cache=bundle["cache"],
        n_lags=bundle["n_lags"],
        y_scaler=bundle["y_scaler"],
        x_scaler=bundle["x_scaler"],
        disable_progress=True,
    )

    # Undo the target scaling, then restore absolute units when the run
    # predicts per-capita targets (a no-op otherwise).  Ground truth comes
    # from the raw frame rather than the scaled array, matching test_xgb.
    population = frame[POPULATION_COLUMN].values
    y_pred = denormalize_by_population(bundle["y_scaler"].inverse_transform(preds_scaled), population)
    y_true = denormalize_by_population(frame[targets].values, population)

    # A group shorter than the lag window is never rolled out, leaving NaN
    # predictions that would poison the metrics; drop those elements the same
    # way an unobserved target is dropped.
    obs_mask = observed_mask_from_frame(frame, targets).astype(bool) & np.isfinite(y_pred)
    logging.info(
        "XGB split %s: %d rows, %d scored elements", split_name, len(frame), int(obs_mask.sum()),
    )
    return y_true, y_pred, obs_mask, len(frame)


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


def _build_split_results_xgb(store, run_id):
    import numpy as np

    from scripts.train_xgb import derive_splits
    from src.data.preprocess import prepare_features_and_targets, split_data
    from src.trainers.xgb_trainer import load_final_xgb_model

    data = store.load_processed_data()
    # The lag count is part of the winning configuration, so the rollout has to
    # see the same feature set the model was fitted on.
    splits = derive_splits(data, store, n_lags=store.load_best_params().get("n_lags"))
    targets = splits["targets"]

    # prepare_data hands back only test_data, but every split needs its raw
    # frame here -- for the absolute-unit ground truth, the population column
    # and the observed mask.  The split is deterministic given the run's
    # assignment, so re-deriving it reproduces the same rows in the same order
    # as the scaled matrices derive_splits just returned.
    prepared, _, _ = prepare_features_and_targets(data, lag_required=True, n_lags=splits["n_lags"])
    train_data, val_data, test_data = split_data(prepared, assignment=store.splits_for(data))

    # Load the model and the run's own scalers once: passing them beats the
    # run_id path, which would reload one booster per target per split.
    bundle = {
        "targets": targets,
        "n_lags": splits["n_lags"],
        "model": load_final_xgb_model(run_id, targets),
        "x_scaler": store.load_artifact("x_scaler.pkl"),
        "y_scaler": store.load_artifact("y_scaler.pkl"),
        # Keyed by frame identity, so one cache serves every split.
        "cache": {},
    }

    frames = {
        "train": (splits["X_train_with_index"], splits["y_train"], train_data),
        "val": (splits["X_val_with_index"], splits["y_val"], val_data),
        # ignore_index throughout: split_data restarts each split's index at
        # 0, so a plain concat would give train and val rows the same labels
        # and the rollout, which scatters predictions by label, would write
        # every train group into a val row.
        "train+val": (
            pd.concat([splits["X_train_with_index"], splits["X_val_with_index"]], ignore_index=True),
            np.concatenate([splits["y_train"], splits["y_val"]], axis=0),
            pd.concat([train_data, val_data], ignore_index=True),
        ),
        "test": (splits["X_test_with_index"], splits["y_test"], test_data),
    }

    results = {}
    for split_name, (X_with_index, y_scaled, frame) in frames.items():
        logging.info("Evaluating XGB on split: %s (%d rows)", split_name, len(frame))
        results[split_name] = _eval_xgb_split(bundle, split_name, X_with_index, y_scaled, frame)
    return targets, results


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
    parser.add_argument("--model", required=True, choices=["lstm", "tft", "xgb"], help="Model type.")
    parser.add_argument("--run_id", required=True, help="Trained run id, e.g. tft_01 / lstm_01 / xgb_01.")
    parser.add_argument("--two-window", action="store_true", help="TFT only: use the two-window prediction path.")
    parser.add_argument("--output", type=str, default=None, help="Optional CSV path for the breakdown (default: <run_root>/metrics/train_val_test_r2_breakdown.csv).")
    args = parser.parse_args(argv)

    from src.utils.utils import setup_logging, get_run_root
    from src.utils.run_store import RunStore
    from src.trainers.evaluation import compute_r2_summary, per_target_r2_table
    from scripts.train import _load_resolved_config

    setup_logging(args.run_id, log_file="evaluate_splits.log")
    store = RunStore(args.run_id)

    # Score the run under the settings it was trained with: keep_partial_targets
    # decides how derive_splits fills targets and which TFT class loads the
    # checkpoint, and a two-window run should be scored two-window by default.
    recorded = _load_resolved_config(args.run_id)
    if recorded.get("keep_partial_targets") is not None:
        import configs.data as data_config
        data_config.KEEP_PARTIAL_TARGETS = recorded["keep_partial_targets"]
    use_two_window = args.two_window or bool(recorded.get("two_window"))

    if args.model == "tft":
        targets, results = _build_split_results_tft(store, args.run_id, use_two_window)
    elif args.model == "xgb":
        targets, results = _build_split_results_xgb(store, args.run_id)
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
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    merged = summary_df.merge(
        per_target_df.pivot(index="Split", columns="Output Variable", values="R2").reset_index(),
        on="Split",
    )
    merged.to_csv(output_path, index=False)
    logging.info("Saved train/val/test R2 breakdown to %s", output_path)
    print(f"\nSaved breakdown to {output_path}")


if __name__ == "__main__":
    main()
