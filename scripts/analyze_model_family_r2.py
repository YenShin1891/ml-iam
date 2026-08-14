#!/usr/bin/env python
"""Per-IAM (Model_Family) R2 breakdown across LSTM / TFT / XGB test results.

Reviewer-response analysis: "It would be valuable to see a performance
breakdown by IAM in the test set." Model_Family (src/data/process_data.py's
get_model_family()) collapses the raw `Model` string's version/config suffix
down to the underlying IAM identity, e.g. "REMIND-MAgPIE 3.0-4.4" -> "REMIND".
That's the right grain for an IAM-level breakdown -- the raw `Model` column
would fragment each IAM into many near-duplicate rows by version.

This is a standalone, read-only analysis script -- it does not touch the
training pipeline or any of its saved artifacts. It only reads what a
completed `test` phase already wrote via RunStore (predictions.pkl,
cached test_data.parquet/y_test.npy), so it has no torch/xgboost/lightning
dependency and runs anywhere pandas+numpy are available. It resolves run
directories the same way the rest of the codebase does -- via
`configs.paths.RESULTS_PATH` -- so as long as that's configured on the
machine you run it on (the same one you trained on), there's nothing else
to "connect": just pass the run_id(s) below.

Usage:
    python scripts/analyze_model_family_r2.py \\
        --lstm-run lstm_04 --tft-run tft_07 --xgb-run xgb_02

    # Any subset of the three is fine, e.g. only two models available yet:
    python scripts/analyze_model_family_r2.py --tft-run tft_07 --xgb-run xgb_02

Outputs (default under RESULTS_PATH/_analysis/model_family_r2/, override
with --output-dir):
    model_family_r2.csv        Wide table: Model_Family x {model}_r2 / {model}_n
    model_family_r2.md         Same, as a markdown table for the paper
    model_family_r2_by_target__<model>.csv   Per-target R2 per family (one per model)

Alignment note (why LSTM/XGB and TFT are loaded differently):
    - TFT's saved predictions cover only the forecast-horizon subset of rows
      (`horizon_df`/`horizon_y_true` in predictions.pkl), already row-aligned
      to `preds` by construction (see predict_tft in tft_trainer.py).
    - LSTM's `predictions.pkl` also carries a `horizon_df`/`horizon_y_true`
      key, but for LSTM those are NOT length-matched to `preds` (`horizon_df`
      is the full test_data, `horizon_y_true` is pre-filtered to valid rows,
      `preds` is full-length with NaN padding) -- an artifact of how
      predict_lstm stores things for plotting, not a bug worth reworking for
      this analysis. So for LSTM (and XGB, which never sets horizon_df),
      this script instead loads the full cached test_data/y_test via
      `RunStore.load_test_data()` and aligns it against the full-length
      `preds` array positionally, then drops NaN rows itself.
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd


def _r2(yt: np.ndarray, yp: np.ndarray) -> float:
    """Pooled R2 for a 1D array pair. Mirrors evaluation.py's save_metrics."""
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def _per_target_r2(yt_2d: np.ndarray, yp_2d: np.ndarray, obs_2d: np.ndarray = None):
    """Per-target R2 list + their mean, matching the "R2 Score (per-target avg)"
    convention already used by evaluation.py's save_metrics."""
    target_r2s = []
    for col in range(yt_2d.shape[1]):
        yt_col, yp_col = yt_2d[:, col], yp_2d[:, col]
        if obs_2d is not None:
            mask = obs_2d[:, col].astype(bool)
            yt_col, yp_col = yt_col[mask], yp_col[mask]
        valid = np.isfinite(yt_col) & np.isfinite(yp_col)
        if valid.any():
            target_r2s.append(_r2(yt_col[valid], yp_col[valid]))
        else:
            target_r2s.append(np.nan)
    finite = [v for v in target_r2s if np.isfinite(v)]
    avg = float(np.mean(finite)) if finite else np.nan
    return avg, target_r2s


def _load_tft(store):
    """TFT: horizon_df / horizon_y_true / preds are already row-aligned."""
    bundle = store.load_predictions()
    horizon_df = bundle.get("horizon_df")
    horizon_y_true = bundle.get("horizon_y_true")
    preds = bundle["preds"]
    if horizon_df is None or horizon_y_true is None:
        raise ValueError(
            "TFT predictions.pkl is missing horizon_df/horizon_y_true; "
            "re-run the test phase for this run_id."
        )
    if not (len(horizon_df) == len(horizon_y_true) == len(preds)):
        raise ValueError(
            f"TFT alignment mismatch: horizon_df={len(horizon_df)} "
            f"horizon_y_true={len(horizon_y_true)} preds={len(preds)}"
        )
    horizon_df = horizon_df.reset_index(drop=True)
    valid = ~(np.isnan(horizon_y_true).any(axis=1) | np.isnan(preds).any(axis=1))
    return horizon_df.loc[valid].reset_index(drop=True), horizon_y_true[valid], preds[valid]


def _load_full_test_aligned(store):
    """LSTM / XGB: preds is full-length and positionally aligned to the
    cached test_data/y_test (see docstring above for why TFT differs)."""
    test_data, y_test = store.load_test_data()
    bundle = store.load_predictions()
    preds = bundle["preds"]
    if len(preds) != len(test_data):
        raise ValueError(
            f"Alignment mismatch: preds={len(preds)} test_data={len(test_data)}. "
            "Was test_data cached by a different run/version than these predictions?"
        )
    test_data = test_data.reset_index(drop=True)
    valid = ~(np.isnan(y_test).any(axis=1) | np.isnan(preds).any(axis=1))
    return test_data.loc[valid].reset_index(drop=True), y_test[valid], preds[valid]


def _load_model_result(model_kind: str, run_id: str, targets):
    """Return (family_series, y_true, y_pred, obs_mask_or_None) for one run."""
    from src.utils.run_store import RunStore
    from src.data.process_data import get_model_family
    from src.data.preprocess import observed_mask_columns

    store = RunStore(run_id)
    if not store.has_predictions():
        raise FileNotFoundError(
            f"No predictions found for run_id='{run_id}'. Run the test phase first: "
            f"python scripts/train.py --model {model_kind} --resume test --run_id {run_id}"
        )

    if model_kind == "tft":
        df, y_true, y_pred = _load_tft(store)
    else:
        df, y_true, y_pred = _load_full_test_aligned(store)

    if "Model" not in df.columns:
        raise ValueError(
            f"'{model_kind}' run '{run_id}': no 'Model' column found in the aligned "
            "dataframe; cannot derive Model_Family."
        )
    family = df["Model"].apply(get_model_family)

    obs_cols = observed_mask_columns(targets)
    obs_mask = df[obs_cols].values if all(c in df.columns for c in obs_cols) else None

    return family, y_true, y_pred, obs_mask


def _family_breakdown(family: pd.Series, y_true, y_pred, obs_mask, min_samples: int):
    """Per-family (per-target-avg R2, n) plus per-target R2, for one model."""
    rows = []
    by_target_rows = []
    excluded = []

    for fam, idx in family.groupby(family).groups.items():
        pos = family.index.get_indexer(idx)
        yt = y_true[pos]
        yp = y_pred[pos]
        obs = obs_mask[pos] if obs_mask is not None else None
        n = len(pos)
        if n < min_samples:
            excluded.append((fam, n))
            continue
        avg_r2, per_target = _per_target_r2(yt, yp, obs)
        rows.append({"Model_Family": fam, "r2": avg_r2, "n": n})
        by_target_rows.append({"Model_Family": fam, "n": n, **{f"r2_{i}": v for i, v in enumerate(per_target)}})

    overall_avg_r2, _ = _per_target_r2(y_true, y_pred, obs_mask)
    rows.append({"Model_Family": "__Overall__", "r2": overall_avg_r2, "n": len(y_true)})

    return pd.DataFrame(rows), pd.DataFrame(by_target_rows), excluded


def _to_markdown_table(df: pd.DataFrame) -> str:
    """Minimal markdown table writer (avoids a hard dependency on `tabulate`,
    which isn't in requirements.txt)."""
    headers = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for idx, row in df.iterrows():
        cells = [str(idx)] + [("" if pd.isna(v) else str(v)) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lstm-run", type=str, default=None, help="LSTM run_id (e.g. lstm_04)")
    parser.add_argument("--tft-run", type=str, default=None, help="TFT run_id (e.g. tft_07)")
    parser.add_argument("--xgb-run", type=str, default=None, help="XGB run_id (e.g. xgb_02)")
    parser.add_argument(
        "--min-samples", type=int, default=5,
        help="Minimum test rows for a Model_Family to be reported (default: 5). "
             "Smaller families are excluded and listed in the log, not silently dropped.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Where to write outputs (default: RESULTS_PATH/_analysis/model_family_r2)",
    )
    args = parser.parse_args(argv)

    runs = {"lstm": args.lstm_run, "tft": args.tft_run, "xgb": args.xgb_run}
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        parser.error("Provide at least one of --lstm-run / --tft-run / --xgb-run")

    from src.utils.run_store import RunStore
    from configs.paths import RESULTS_PATH

    output_dir = args.output_dir or os.path.join(RESULTS_PATH, "_analysis", "model_family_r2")
    os.makedirs(output_dir, exist_ok=True)

    summary_frames = {}
    for model_kind, run_id in runs.items():
        logging.info("Loading %s run '%s'...", model_kind, run_id)
        store = RunStore(run_id)
        _, targets = store.load_features()

        family, y_true, y_pred, obs_mask = _load_model_result(model_kind, run_id, targets)
        summary_df, by_target_df, excluded = _family_breakdown(family, y_true, y_pred, obs_mask, args.min_samples)

        if excluded:
            logging.info(
                "%s (%s): excluded %d Model_Family group(s) below --min-samples=%d: %s",
                model_kind, run_id, len(excluded), args.min_samples, excluded,
            )

        summary_frames[model_kind] = summary_df.set_index("Model_Family")

        by_target_path = os.path.join(output_dir, f"model_family_r2_by_target__{model_kind}_{run_id}.csv")
        by_target_df.to_csv(by_target_path, index=False)
        logging.info("%s (%s): wrote per-target breakdown to %s", model_kind, run_id, by_target_path)

    # Combine into one wide table: Model_Family x {model}_r2 / {model}_n
    combined = None
    for model_kind, df in summary_frames.items():
        df = df.rename(columns={"r2": f"{model_kind}_r2", "n": f"{model_kind}_n"})
        combined = df if combined is None else combined.join(df, how="outer")

    # Sort by total sample size descending, keeping __Overall__ pinned at top.
    n_cols = [c for c in combined.columns if c.endswith("_n")]
    combined["_total_n"] = combined[n_cols].sum(axis=1, skipna=True)
    is_overall = combined.index == "__Overall__"
    combined = pd.concat([
        combined.loc[is_overall].sort_index(),
        combined.loc[~is_overall].sort_values("_total_n", ascending=False),
    ]).drop(columns="_total_n")
    combined.index.name = "Model_Family"

    csv_path = os.path.join(output_dir, "model_family_r2.csv")
    combined.to_csv(csv_path)
    logging.info("Wrote combined table to %s", csv_path)

    md_path = os.path.join(output_dir, "model_family_r2.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_to_markdown_table(combined.round(3)))
    logging.info("Wrote markdown table to %s", md_path)

    print(combined.round(3).to_string())
    return combined


if __name__ == "__main__":
    main()
