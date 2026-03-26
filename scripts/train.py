#!/usr/bin/env python

"""Unified training entrypoint for all models (xgb, lstm, tft).

Usage:
  python scripts/train.py --model xgb --resume search --run_id xgb_01
  python scripts/train.py --model lstm  # full pipeline

Typically invoked by train_from_config.py (via `make train`), not directly.
"""

import argparse
import logging
import os
import sys


_ALLOWED_MODELS = ("xgb", "lstm", "tft")
_ALLOWED_PHASES = ("preprocess", "search", "train", "test", "plot")


def _seed(model: str) -> None:
    """Set reproducibility seeds. Lazy-imports to avoid pulling in torch for XGB."""
    import numpy as np
    np.random.seed(0)
    if model in ("lstm", "tft"):
        from lightning.pytorch import seed_everything
        seed_everything(0, workers=True)


def _is_primary_rank() -> bool:
    """Check if this is the primary DDP rank (or non-DDP)."""
    rank_vars = [
        os.getenv("PL_TRAINER_GLOBAL_RANK"),
        os.getenv("GLOBAL_RANK"),
        os.getenv("RANK"),
    ]
    return all(rv in (None, "0") for rv in rank_vars)


# ---------------------------------------------------------------------------
# Model-specific dispatch helpers (lazy imports)
# ---------------------------------------------------------------------------

def _preprocess(model, store, dataset, lag_required):
    if model == "xgb":
        from scripts.train_xgb import preprocess_xgb
        return preprocess_xgb(store, dataset=dataset)
    elif model == "lstm":
        from scripts.train_lstm import preprocess_lstm
        return preprocess_lstm(store, dataset=dataset, lag_required=lag_required)
    elif model == "tft":
        from scripts.train_tft import preprocess_tft
        return preprocess_tft(store, dataset=dataset, lag_required=lag_required)


def _search(model, store, lag_required=True):
    if model == "xgb":
        from scripts.train_xgb import search_xgb
        return search_xgb(store)
    elif model == "lstm":
        from scripts.train_lstm import search_lstm
        return search_lstm(store, lag_required=lag_required)
    elif model == "tft":
        from scripts.train_tft import search_tft
        return search_tft(store, lag_required=lag_required)


def _train(model, store, lag_required=True):
    if model == "xgb":
        from scripts.train_xgb import train_xgb
        return train_xgb(store)
    elif model == "lstm":
        from scripts.train_lstm import train_lstm
        return train_lstm(store, lag_required=lag_required)
    elif model == "tft":
        from scripts.train_tft import train_tft
        return train_tft(store, lag_required=lag_required)


def _test(model, store, lag_required=True, two_window=False):
    if model == "xgb":
        from scripts.train_xgb import test_xgb
        return test_xgb(store)
    elif model == "lstm":
        from scripts.train_lstm import test_lstm
        return test_lstm(store, lag_required=lag_required)
    elif model == "tft":
        from scripts.train_tft import test_tft
        return test_tft(store, lag_required=lag_required, use_two_window=two_window)


def _plot(model, store, lag_required=True):
    if model == "xgb":
        from scripts.train_xgb import plot_xgb
        return plot_xgb(store)
    elif model == "lstm":
        from scripts.train_lstm import plot_lstm
        return plot_lstm(store, lag_required=lag_required)
    elif model == "tft":
        from scripts.train_tft import plot_tft
        return plot_tft(store, lag_required=lag_required)


def _set_default_params(model, store):
    """Inject default hyperparameters (used when search phase is not requested)."""
    if store.has_best_params():
        return
    if model == "xgb":
        from configs.models.xgb_search import XGBDefaultParams
        store.save_best_params(XGBDefaultParams().to_dict())
    elif model == "lstm":
        from scripts.train_lstm import _default_best_params_from_config
        store.save_best_params(_default_best_params_from_config())
    elif model == "tft":
        from configs.models.tft_search import TFTDefaultParams
        store.save_best_params(TFTDefaultParams().to_dict())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Unified model training script.")
    parser.add_argument("--model", required=True, choices=_ALLOWED_MODELS, help="Model type.")
    parser.add_argument("--run_id", type=str, help="Run ID (required with --resume).")
    parser.add_argument("--resume", type=str, choices=_ALLOWED_PHASES, help="Resume from a specific phase.")
    parser.add_argument("--note", type=str, help="Note describing the run.")
    parser.add_argument("--dataset", type=str, help="Dataset version subdirectory.")
    parser.add_argument(
        "--lag-required",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require complete lag features (LSTM/TFT).",
    )
    parser.add_argument("--two-window", action="store_true", help="Two-window prediction (TFT only).")
    args = parser.parse_args(argv)

    if args.resume and not args.run_id:
        parser.error("--run_id is required when --resume is specified")
    if not args.resume and args.run_id:
        parser.error("--run_id should only be specified when using --resume")

    return args


def main(argv=None):
    args = parse_arguments(argv)
    model = args.model

    _seed(model)

    from src.utils.utils import setup_logging, get_next_run_id
    from src.utils.run_store import RunStore

    lag_required = True if args.lag_required is None else args.lag_required

    if args.resume is None:
        # Full pipeline: preprocess -> search -> train -> test -> plot
        run_id = get_next_run_id(model)
        if _is_primary_rank():
            setup_logging(run_id)

        store = RunStore(run_id)

        if args.note:
            logging.info("Run note: %s", args.note)

        _preprocess(model, store, args.dataset, lag_required)
        _search(model, store, lag_required=lag_required)
        _train(model, store, lag_required=lag_required)
        _test(model, store, lag_required=lag_required, two_window=args.two_window)
        _plot(model, store, lag_required=lag_required)
        return

    # Resume mode: single phase
    run_id = args.run_id
    if _is_primary_rank():
        setup_logging(run_id)

    store = RunStore(run_id)

    if args.note:
        logging.info("Run note: %s", args.note)

    # Dispatch
    phase = args.resume
    if phase == "preprocess":
        _preprocess(model, store, args.dataset, lag_required)
        _set_default_params(model, store)
        logging.info("Preprocessing complete.")
    elif phase == "search":
        _search(model, store, lag_required=lag_required)
    elif phase == "train":
        _train(model, store, lag_required=lag_required)
    elif phase == "test":
        _test(model, store, lag_required=lag_required, two_window=args.two_window)
    elif phase == "plot":
        # TFT auto-runs test if predictions are missing
        if model == "tft" and not store.has_predictions():
            logging.info("Predictions not found; rerunning test step before plotting.")
            _test(model, store, lag_required=lag_required, two_window=args.two_window)
        _plot(model, store, lag_required=lag_required)


if __name__ == "__main__":
    if _is_primary_rank():
        main()
