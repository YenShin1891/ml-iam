#!/usr/bin/env python

"""Recompute the SHAP arrays and plots of a trained LSTM or TFT run.

The plot phase writes a run's SHAP files into the run's own plots directory.
This script computes them again without touching the run: everything goes to
--out_dir, by default RESULTS_PATH/_analysis/shap/<run_id>, and a directory
inside the run itself is refused.  It is how a run that is already published
gets SHAP values from a corrected explainer while its files stay as released.

It also prints the additivity check -- how closely each window's SHAP values
add up to its prediction minus the background mean -- and the top features
of one output.

Usage:
  python scripts/recompute_shap.py --run_id lstm_01
  python scripts/recompute_shap.py --run_id tft_01 --out_dir /some/dir --target "Emissions|CO2"

The paper's Figure 6 reads the result with
  python scripts/plot_paper_figures.py fig6 --shap-dir lstm_01=<out_dir> ...
"""

import argparse
import json
import logging
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def default_out_dir(run_id: str) -> str:
    from configs.paths import RESULTS_PATH

    return os.path.join(RESULTS_PATH, "_analysis", "shap", run_id)


def assert_outside_run(out_dir: str, run_root: str) -> None:
    """Refuse an output directory that lies inside the run being explained."""
    out_real, run_real = os.path.realpath(out_dir), os.path.realpath(run_root)
    if out_real == run_real or out_real.startswith(run_real + os.sep):
        raise ValueError(
            f"--out_dir {out_dir} is inside the run ({run_root}); this script must not write into a run."
        )


def assert_run_is_frozen(store) -> None:
    """Refuse a run whose split or vocabularies deriving the splits would (re)write."""
    from src.data.preprocess import resolve_categorical_vocabularies

    if not store.has_splits() or not store.has_categories():
        raise FileNotFoundError(
            f"{store.run_id}: splits.parquet / categories.json missing; deriving its splits would "
            "write them into the run."
        )
    persisted = store.load_categories()
    if resolve_categorical_vocabularies(store.load_processed_data(), persisted) != persisted:
        raise ValueError(
            f"{store.run_id}: its cached data no longer matches categories.json; deriving its splits "
            "would rewrite that file."
        )


def summarise(out_dir: str, model: str, target: str, top_k: int = 8) -> dict:
    """Sample sizes, additivity and the top features of *target*, from the saved cache."""
    import numpy as np

    from src.visualization.helpers import build_feature_display_names
    from src.visualization.shap_nn import additivity_report

    with np.load(os.path.join(out_dir, f"{model}_shap_cache.npz")) as cache:
        temporal = cache["temporal_shap"]
        targets = cache["targets"].tolist()
        names = build_feature_display_names(cache["feature_names"].tolist())
        report = additivity_report(temporal, cache["fx"], cache["base_values"], targets)
        report.update(
            n_candidate_windows=int(cache["n_candidate_windows"]),
            n_background=int(cache["n_background"]),
            n_steps=int(temporal.shape[1]),
            encoder_length=int(cache["encoder_length"]),
            n_features=int(temporal.shape[2]),
        )
    if target in targets:
        per_feature = np.abs(temporal.sum(axis=1)[:, :, targets.index(target)]).mean(axis=0)
        share = per_feature / per_feature.sum() if per_feature.sum() > 0 else per_feature
        order = np.argsort(per_feature)[::-1][:top_k]
        report["top_features"] = {"target": target, "ranking": [[names[i], float(share[i])] for i in order]}
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_id", required=True, help="an lstm_* or tft_* run id")
    parser.add_argument("--out_dir", default=None, help="where the arrays and plots go (never inside the run)")
    parser.add_argument("--target", default="Emissions|CO2", help="output whose top features are printed")
    args = parser.parse_args(argv)

    model = args.run_id.split("_", 1)[0].lower()
    if model not in ("lstm", "tft"):
        parser.error("--run_id must be an lstm_* or tft_* run; XGBoost SHAP is exact and needs no recomputing")

    import matplotlib

    matplotlib.use("Agg")

    from src.utils.run_store import RunStore
    from src.utils.utils import get_run_root

    run_root = get_run_root(args.run_id)
    if not os.path.isdir(run_root):
        parser.error(f"run {args.run_id} not found")
    out_dir = args.out_dir or default_out_dir(args.run_id)
    assert_outside_run(out_dir, run_root)
    os.makedirs(out_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(out_dir, "recompute_shap.log"), mode="w")],
        force=True,
    )

    store = RunStore(args.run_id)
    assert_run_is_frozen(store)
    from scripts.train import _seed
    _seed(model)

    if model == "lstm":
        from scripts.train_lstm import shap_lstm

        shap_lstm(store, out_dir=out_dir)
    else:
        from scripts.train_tft import shap_tft

        shap_tft(store, out_dir=out_dir, use_cached=False)

    cache_path = os.path.join(out_dir, f"{model}_shap_cache.npz")
    if not os.path.exists(cache_path):
        logging.error("No SHAP cache was written to %s; see the errors above.", out_dir)
        return 1
    report = summarise(out_dir, model, args.target)
    report["run_id"] = args.run_id
    with open(os.path.join(out_dir, f"{model}_shap_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
