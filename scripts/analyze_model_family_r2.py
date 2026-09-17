#!/usr/bin/env python

"""Test-set R2 broken down by IAM (Model_Family) for trained XGB/LSTM/TFT runs.

Model_Family (src/data/process_data.py:get_model_family) strips the version
and configuration suffix from the raw Model string, e.g.
"REMIND-MAgPIE 3.0-4.4" -> "REMIND". That is the grain an IAM-level breakdown
wants: the raw Model column would split each IAM into many near-duplicates.

Read-only. It scores what a completed test phase already saved
(artifacts/predictions.pkl, plus cache/test_data.parquet for XGB) with the
metric code that wrote metrics/performance.csv, so each model's Overall row
here is that file's Overall row; a mismatch is logged as a warning.

  model_family_r2.csv                      Model_Family x {model}_r2 / _trajectories / ...
  model_family_r2.md                       the same, as a markdown table
  model_family_r2_by_target__<run_id>.csv  per-target R2 per family

R2 is the per-target average, the headline number of performance.csv. A
target that is constant within a family (a family that never reports Nuclear,
say) has no R2 and is left out of that family's average; n_targets says how
many went in. _trajectories counts the (Model, Scenario, Region) series the
family has in the scored frame, _rows their rows, and _n the scored elements
(observed and finite). The three models are scored on different rows --
their context lengths and horizons differ -- so compare R2 across a row, not
the counts.

Usage:
  python scripts/analyze_model_family_r2.py --xgb-run xgb_85 --lstm-run lstm_89 --tft-run tft_95
  python scripts/analyze_model_family_r2.py --lstm-run lstm_89
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OVERALL = "Overall"


def _load_scored_rows(run_id: str):
    """(frame, y_true, y_pred, targets) of a run, aligned row for row.

    The sequence models save the horizon frame they predicted on beside the
    predictions; XGB predicts every test row, so its frame is the cached
    test split.
    """
    from src.utils.run_store import RunStore

    store = RunStore(run_id)
    _, targets = store.load_features()
    bundle = store.load_predictions()
    y_pred = np.asarray(bundle["preds"], dtype=float)
    frame, y_true = bundle.get("horizon_df"), bundle.get("horizon_y_true")
    if frame is None or y_true is None:
        frame, y_true = store.load_test_data()
    if not (len(frame) == len(y_true) == len(y_pred)):
        raise ValueError(
            f"{run_id}: frame, y_true and preds have {len(frame)}, {len(y_true)} and "
            f"{len(y_pred)} rows; re-run the test phase so they are saved together."
        )
    return frame.reset_index(drop=True), np.asarray(y_true, dtype=float), y_pred, targets


def _score(y_true, y_pred, observed, targets) -> dict:
    from src.trainers.evaluation import per_target_r2_table

    table = per_target_r2_table(y_true, y_pred, targets, observed_mask=observed)
    r2 = table["R2"].to_numpy(dtype=float)
    finite = np.isfinite(r2)
    return {
        "r2": float(r2[finite].mean()) if finite.any() else np.nan,
        "n_targets": int(finite.sum()),
        "n": int(table["Sample Size"].sum()),
        "rows": int(len(y_true)),
        "by_target": dict(zip(targets, r2)),
    }


def family_breakdown(run_id: str, min_rows: int):
    """(summary, per-target) frames for one run, Overall first."""
    from configs.data import INDEX_COLUMNS
    from src.data.preprocess import observed_mask_from_frame
    from src.data.process_data import get_model_family

    frame, y_true, y_pred, targets = _load_scored_rows(run_id)
    observed = observed_mask_from_frame(frame, targets)
    # Derived from Model rather than read: the LSTM frame carries Model_Family
    # as integer codes and the TFT horizon frame does not carry it at all.
    family = frame["Model"].map(get_model_family)
    trajectory = frame.groupby(INDEX_COLUMNS, sort=False).ngroup().to_numpy()

    groups = [(OVERALL, np.arange(len(frame)))]
    skipped = []
    for name, positions in family.groupby(family).indices.items():
        if len(positions) < min_rows:
            skipped.append((name, len(positions)))
            continue
        groups.append((name, positions))
    if skipped:
        logging.info("%s: left out families under --min-rows=%d: %s", run_id, min_rows, skipped)

    summary, by_target = [], []
    for name, positions in groups:
        scored = _score(
            y_true[positions], y_pred[positions],
            observed[positions] if observed is not None else None, targets,
        )
        per_target = scored.pop("by_target")
        summary.append({
            "Model_Family": name, "r2": scored.pop("r2"),
            "trajectories": int(len(np.unique(trajectory[positions]))), **scored,
        })
        by_target.append({"Model_Family": name, "rows": scored["rows"], **per_target})

    _check_against_performance_csv(run_id, summary[0])
    return pd.DataFrame(summary).set_index("Model_Family"), pd.DataFrame(by_target)


def _check_against_performance_csv(run_id: str, overall: dict) -> None:
    from src.utils.utils import get_run_root

    path = os.path.join(get_run_root(run_id), "metrics", "performance.csv")
    if not os.path.exists(path):
        return
    official = pd.read_csv(path)
    official = official[official["Region Type"] == OVERALL]
    if official.empty or "R2 Score (per-target avg)" not in official:
        return
    expected = float(official["R2 Score (per-target avg)"].iloc[0])
    if not np.isclose(expected, overall["r2"], atol=1e-6):
        logging.warning(
            "%s: Overall R2 here is %.6f but performance.csv says %.6f; the saved "
            "predictions and metrics come from different test runs.",
            run_id, overall["r2"], expected,
        )


def _markdown_table(df: pd.DataFrame) -> str:
    # By hand: DataFrame.to_markdown needs tabulate, which is not a requirement.
    header = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for label, row in df.iterrows():
        cells = ["" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v)) for v in row]
        lines.append("| " + " | ".join([str(label)] + cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xgb-run", default=None, help="XGB run id, e.g. xgb_85.")
    parser.add_argument("--lstm-run", default=None, help="LSTM run id, e.g. lstm_89.")
    parser.add_argument("--tft-run", default=None, help="TFT run id, e.g. tft_95.")
    parser.add_argument("--min-rows", type=int, default=5,
                        help="Families with fewer test rows are left out and logged (default: 5).")
    parser.add_argument("--output_dir", default=None,
                        help="Default: <RESULTS_PATH>/_analysis/model_family_r2.")
    args = parser.parse_args(argv)

    runs = {"xgb": args.xgb_run, "lstm": args.lstm_run, "tft": args.tft_run}
    runs = {model: run_id for model, run_id in runs.items() if run_id}
    if not runs:
        parser.error("Give at least one of --xgb-run / --lstm-run / --tft-run.")

    from configs.paths import RESULTS_PATH

    output_dir = args.output_dir or os.path.join(RESULTS_PATH, "_analysis", "model_family_r2")
    os.makedirs(output_dir, exist_ok=True)

    combined = None
    for model, run_id in runs.items():
        summary, by_target = family_breakdown(run_id, args.min_rows)
        by_target.to_csv(os.path.join(output_dir, f"model_family_r2_by_target__{run_id}.csv"), index=False)
        summary = summary.rename(columns={column: f"{model}_{column}" for column in summary.columns})
        combined = summary if combined is None else combined.join(summary, how="outer")

    # Overall first, then the families by how many test rows they have.
    size = combined[[c for c in combined.columns if c.endswith("_rows")]].sum(axis=1)
    order = size.drop(OVERALL).sort_values(ascending=False).index
    combined = combined.loc[[OVERALL, *order]]
    count_columns = [c for c in combined.columns if not c.endswith("_r2")]
    combined[count_columns] = combined[count_columns].astype("Int64")

    combined.to_csv(os.path.join(output_dir, "model_family_r2.csv"))
    with open(os.path.join(output_dir, "model_family_r2.md"), "w", encoding="utf-8") as f:
        f.write(_markdown_table(combined))

    print(combined.round(3).to_string())
    print(f"\nRuns: {runs}\nSaved Model_Family R2 tables to {output_dir}")
    return combined


if __name__ == "__main__":
    main()
