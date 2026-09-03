#!/usr/bin/env python

"""Unified training entrypoint for all models (xgb, lstm, tft).

Usage:
  python scripts/train.py --model xgb --resume search --run_id xgb_01
  python scripts/train.py --model lstm  # full pipeline

Typically invoked by train_from_config.py (via `make train`), not directly.
"""

import argparse
import logging
from pathlib import Path


_ALLOWED_MODELS = ("xgb", "lstm", "tft")
_ALLOWED_PHASES = ("preprocess", "search", "train", "test", "plot")

def _seed(model: str) -> None:
    """Set reproducibility seeds. Lazy-imports to avoid pulling in torch for XGB."""
    import numpy as np
    np.random.seed(0)
    if model in ("lstm", "tft"):
        from lightning.pytorch import seed_everything
        seed_everything(0, workers=True)


# ---------------------------------------------------------------------------
# Phase dispatch
# ---------------------------------------------------------------------------

def preprocess(store, dataset=None):
    """Cache the processed dataset and the assignments derived from it.

    The only phase with no model-specific behaviour: all three models read the
    same parquet, category vocabularies and split assignment.
    """
    from src.data.preprocess import load_and_process_data

    data = load_and_process_data(version=dataset)
    store.save_processed_data(data)
    store.categories_for(data)
    store.splits_for(data)
    return data


def _phase_function(model: str, phase: str):
    """Resolve e.g. ("tft", "search") to scripts.train_tft.search_tft.

    Imported here rather than at module scope so the XGBoost path never loads
    the deep-learning stack.
    """
    from importlib import import_module

    if phase == "preprocess":
        return preprocess
    return getattr(import_module(f"scripts.train_{model}"), f"{phase}_{model}")


def _accepted_by_any_phase(model: str, name: str) -> bool:
    """Whether *any* phase of *model* takes the option *name*.

    All of a model's phases live in one module, so this costs no extra import.
    """
    import inspect

    for phase in _ALLOWED_PHASES:
        try:
            function = _phase_function(model, phase)
        except Exception:  # noqa: BLE001 - a missing phase is not this check's problem
            continue
        if name in inspect.signature(function).parameters:
            return True
    return False


def run_phase(model: str, phase: str, store, **options):
    """Run one phase, passing only the options its model accepts.

    TFT alone takes target_normalizer_mode and two-window prediction, so the
    options are filtered here instead of every phase function growing
    parameters it ignores.
    """
    import inspect

    function = _phase_function(model, phase)
    accepted = inspect.signature(function).parameters
    supported = {name: value for name, value in options.items() if name in accepted}

    for name in options:
        if name in accepted or options[name] in (None, False):
            continue
        # `dataset` is set for the whole run but only preprocess reads it, and
        # saying so in all four other phases is noise about nothing.  Report
        # only options no phase of this model will ever act on.
        if _accepted_by_any_phase(model, name):
            continue
        logging.info("%s %s does not support %s; ignoring it", model, phase, name)

    return function(store, **supported)


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


_RESUMABLE_SETTINGS = ("keep_partial_targets", "target_normalizer_mode", "two_window", "dataset")


def _load_resolved_config(run_id: str) -> dict:
    """Read meta/run_config.resolved.json, written by train_from_config.py."""
    import json

    from src.utils.utils import get_run_root

    path = Path(get_run_root(run_id)) / "meta" / "run_config.resolved.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        logging.warning("Could not read %s: %s", path, e)
        return {}


def _apply_run_settings(args) -> None:
    """Fill unspecified settings from the run's own recorded config.

    Each phase runs in a fresh process, so a phase resumed by hand
    ("--resume test --run_id tft_91") would otherwise silently fall back to
    whatever configs/data.py currently says and evaluate the run under
    different settings than it was trained with.
    """
    resolved = _load_resolved_config(args.run_id)
    if not resolved:
        return

    for name in _RESUMABLE_SETTINGS:
        recorded = resolved.get(name)
        current = getattr(args, name, None)
        if recorded is None:
            continue
        # Only None means "not given": --no-keep-partial-targets and
        # --no-two-window are explicit choices that must win over the record.
        if current is None:
            if current != recorded:
                logging.info("Using %s=%r recorded for run %s", name, recorded, args.run_id)
                setattr(args, name, recorded)
        elif current != recorded:
            logging.warning(
                "%s=%r on the command line overrides %r recorded for run %s",
                name, current, recorded, args.run_id,
            )


def _log_run_header(run_id: str, model: str, args, resolved: dict) -> None:
    """Record what this run is, once, where its own log will carry it.

    Every phase is a separate process appending to one train.log, and the
    settings lived only in meta/run_config.resolved.json — so tft_93's log
    never said which dataset it read or that its search had been skipped.
    """
    lines = [f"model={model}"]
    dataset = args.dataset or resolved.get("dataset")
    if dataset:
        lines.append(f"dataset={dataset}")
    phases = resolved.get("phases")
    if phases:
        lines.append(f"phases={', '.join(phases)}")

    gpus = resolved.get("cuda_visible_devices_resolved_by_phase") or {}
    if gpus:
        lines.append("gpus=" + ", ".join(f"{phase}:{value}" for phase, value in gpus.items()))

    for name in ("keep_partial_targets", "target_normalizer_mode", "two_window"):
        value = getattr(args, name, None)
        if value is None:
            value = resolved.get(name)
        if value is not None:
            lines.append(f"{name}={value}")

    logging.info("=== Run %s: %s ===", run_id, " | ".join(lines))
    note = args.note or resolved.get("note")
    if note:
        logging.info("Run note: %s", note)


def _run(model: str, phase: str, store, args, phases=(), after=None):
    """Run a phase, bracketed by a banner naming it and its duration.

    *after* runs inside the bracket, so work that belongs to the phase is not
    logged beneath the line announcing the phase finished.
    """
    import time

    from src.utils.utils import format_duration

    position = f" {phases.index(phase) + 1}/{len(phases)}" if phase in phases else ""
    logging.info("--- %s phase%s: %s ---", store.run_id, position, phase)

    started = time.monotonic()
    result = run_phase(
        model, phase, store,
        dataset=args.dataset,
        target_normalizer_mode=args.target_normalizer_mode,
        use_two_window=args.two_window,
    )
    if after is not None:
        after()
    logging.info(
        "--- %s phase%s: %s done in %s ---",
        store.run_id, position, phase, format_duration(time.monotonic() - started),
    )
    return result


def _assert_resume_run_exists(run_id: str) -> None:
    """Fail fast if resume is requested for a non-existent run directory."""
    from src.utils.utils import get_run_root

    run_root = Path(get_run_root(run_id))
    if not run_root.exists() or not run_root.is_dir():
        raise FileNotFoundError(
            f"Cannot resume run '{run_id}': run directory does not exist at {run_root}. "
            "Check run_id or run preprocess/full pipeline first."
        )


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
        "--two-window",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Two-window prediction (TFT only; default from the run's recorded config).",
    )
    parser.add_argument(
        "--target-normalizer-mode",
        type=str,
        choices=["encoder_floored", "global"],
        default=None,
        help="TFT target normalizer: 'encoder_floored' (per-sample with floor) or 'global' (single μ/σ per target).",
    )
    parser.add_argument(
        "--keep-partial-targets",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep rows with partial target coverage (default from configs.data).",
    )
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

    from src.utils.utils import setup_logging, get_next_run_id, is_primary_rank
    from src.utils.run_store import RunStore

    if args.run_id:
        _apply_run_settings(args)

    # Override KEEP_PARTIAL_TARGETS if specified on CLI or recorded for the run
    if args.keep_partial_targets is not None:
        import configs.data as _data_cfg
        _data_cfg.KEEP_PARTIAL_TARGETS = args.keep_partial_targets

    if args.resume is None:
        # Full pipeline: preprocess -> search -> train -> test -> plot
        # NOTE: This path is NOT safe for DDP subprocess re-launching
        # (each rank would allocate a different run_id and re-run all
        # phases).  Use train_from_config.py which invokes per-phase
        # with --resume, or pass --resume explicitly.
        if not is_primary_rank():
            raise RuntimeError(
                "Full-pipeline mode (no --resume) cannot be used under DDP. "
                "Use train_from_config.py or pass --resume <phase> --run_id <id>."
            )
        run_id = get_next_run_id(model)
        setup_logging(run_id)

        store = RunStore(run_id)
        _log_run_header(run_id, model, args, {"phases": list(_ALLOWED_PHASES)})

        for phase in _ALLOWED_PHASES:
            _run(model, phase, store, args, phases=_ALLOWED_PHASES)
        return

    # Resume mode: single phase
    run_id = args.run_id
    _assert_resume_run_exists(run_id)
    if is_primary_rank():
        setup_logging(run_id)

    store = RunStore(run_id)

    resolved = _load_resolved_config(run_id)
    phases = tuple(resolved.get("phases") or ())
    # The run's settings belong at the top of its log, so they are written by
    # whichever phase leads the run rather than repeated by every phase.
    if not phases or phases[0] == args.resume:
        _log_run_header(run_id, model, args, resolved)

    # TFT plots from saved predictions, so make sure they exist.
    if args.resume == "plot" and model == "tft" and not store.has_predictions():
        logging.info("Predictions not found; rerunning test step before plotting.")
        _run(model, "test", store, args)

    # Runs without a search phase still need parameters to train with, so the
    # defaults are written while preprocess is still the phase in progress.
    inject_defaults = (
        (lambda: _set_default_params(model, store)) if args.resume == "preprocess" else None
    )
    _run(model, args.resume, store, args, phases=phases, after=inject_defaults)


if __name__ == "__main__":
    main()
