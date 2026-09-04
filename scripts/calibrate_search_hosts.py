#!/usr/bin/env python

"""Measure how much slower or faster each machine is, so their trial times can be pooled.

A search split across different GPUs produces wall-clock numbers that must not
simply be added: an hour on an A100 and an hour on a 2080 Ti are not the same
hour, and their sum describes no machine that exists.  The honest way to report
one cost figure is to normalise every trial to one reference device and publish
the factors alongside the raw per-machine table.

The factor is measured, not looked up.  TFT is attention- and LSTM-heavy and
does not track peak FLOPS, and a machine's trial time also absorbs its
dataloader, its disk and its driver -- none of which a spec sheet knows.  So
each machine runs the same few configurations, drawn from the search space
itself, under a fixed schedule:

  python scripts/calibrate_search_hosts.py measure --run-id tft_99

and the results are pooled against whichever machine the paper quotes:

  python scripts/calibrate_search_hosts.py report --reference gpu-a *.jsonl \\
      --ledger merged.jsonl

Three configurations rather than one, because the point is not only to get a
factor but to find out whether a single factor is *valid*.  If a machine is
1.8x the reference on a small model and 1.2x on a large one, no scalar
describes it, and the report says so instead of averaging the two into a
number that is wrong for both.  Report raw per-host time in that case.

Conditions the measurement depends on, and which `measure` enforces or checks:
one trial at a time on one GPU with nothing else on it, early stopping off so
every machine runs the same number of epochs, and the same commit, dataset and
batch size everywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# How expensive a configuration is expected to be, used only to pick a spread
# of them.  It does not have to be accurate -- it has to order the space.
_COST_PROXY = {
    "tft": lambda p: p["hidden_size"] * p["encoder_length"] * p["lstm_layers"],
}

# Above this spread between a machine's per-configuration ratios, one scalar
# factor does not describe it.  A fifth is already generous: it is the
# difference between a search costing 10 GPU-days and 12.
_SPREAD_TOLERANCE = 0.20

DEFAULT_EPOCHS = 3


# --------------------------------------------------------------------------
# Choosing what to measure
# --------------------------------------------------------------------------

def calibration_configs(model: str, count: int = 3) -> List[Dict[str, Any]]:
    """A cheap, a middling and an expensive configuration from the real space.

    Drawn from the search space rather than invented, so the factor is
    measured on the kind of work it will be applied to, and spread across the
    cost range so the constancy of the ratio can be checked at all.
    """
    from scripts.merge_search_ledgers import load_space

    if model not in _COST_PROXY:
        raise ValueError(f"No cost proxy defined for {model!r}; add one to _COST_PROXY.")
    if count < 2:
        raise ValueError("Calibrating on fewer than two configurations cannot check itself.")

    ordered = sorted(load_space(model).sample(), key=_COST_PROXY[model])
    # Evenly spaced through the ordering, always including both extremes.
    picks = [round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)]
    return [ordered[i] for i in picks]


def config_label(model: str, params: Dict[str, Any]) -> str:
    """Stable name for a calibration configuration across machines.

    A hash of the parameters rather than their cost rank: two co-authors who
    passed different ``--configs`` counts would rank the same configuration
    differently, and a ratio is only meaningful between timings of identical
    work.  Under a hash a mismatch shows up as a configuration one machine did
    not run, which is true, instead of as a comparison between two different
    models, which would be silently wrong.
    """
    import hashlib

    from scripts.merge_search_ledgers import load_space
    from src.trainers.search import params_signature

    signature = params_signature(params, load_space(model).param_keys)
    return "c" + hashlib.sha1(signature.encode("utf-8")).hexdigest()[:8]


# --------------------------------------------------------------------------
# measure
# --------------------------------------------------------------------------
def _recorded_setting(run_id: str, name: str) -> Optional[Any]:
    """One field of the run's meta/run_config.resolved.json, if it has one."""
    from src.utils.utils import get_run_root

    path = Path(get_run_root(run_id)) / "meta" / "run_config.resolved.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get(name)
    except Exception:  # noqa: BLE001 - an absent record is not an error here
        return None




def measure(args) -> int:
    import logging
    import time

    from src.trainers.provenance import trial_provenance
    from src.trainers.tft_model import create_dataloaders
    from src.trainers.tft_trainer import _fit_search_trial
    from src.utils.run_store import RunStore
    from src.utils.utils import setup_logging
    from configs.models import TFTTrainerConfig
    from scripts.train_tft import derive_splits

    setup_logging(args.run_id)
    store = RunStore(args.run_id)

    configs = calibration_configs(args.model, args.configs)
    logging.info(
        "Calibrating %s on %d configuration(s) at a fixed %d epochs, one at a time on one GPU.",
        args.model, len(configs), args.epochs,
    )

    trainer_cfg = TFTTrainerConfig()
    trainer_cfg.max_epochs = args.epochs
    # Early stopping would end the schedule at a different epoch on different
    # machines, and the ratio would then measure convergence luck rather than
    # speed.  Patience beyond the budget can never fire.
    trainer_cfg.patience = args.epochs + 1

    # The run's own normalizer mode, not today's default: a calibration that
    # builds different datasets than the search did is timing different work.
    normalizer = args.target_normalizer_mode or _recorded_setting(args.run_id, "target_normalizer_mode")
    splits = derive_splits(store.load_processed_data(), store, target_normalizer_mode=normalizer)
    n_targets = len(splits["targets"])

    from src.trainers.tft_dataset import build_datasets

    datasets: Dict[int, Any] = {}
    for encoder_length in sorted({int(p["encoder_length"]) for p in configs}):
        datasets[encoder_length] = build_datasets(dict(splits), encoder_length=encoder_length)

    out_path = Path(args.out or store.root / "search" / "calibration.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    provenance = trial_provenance(args.run_id)

    with open(out_path, "w", encoding="utf-8") as handle:
        for index, params in enumerate(configs, start=1):
            train_dataset, val_dataset = datasets[int(params["encoder_length"])]
            train_loader, val_loader = create_dataloaders(
                train_dataset, val_dataset, trainer_cfg.batch_size,
            )
            label = config_label(args.model, params)
            logging.info("Calibration %d/%d: %s", index, len(configs), label)

            started = time.monotonic()
            _, _, epochs_run = _fit_search_trial(
                train_dataset, params, n_targets, trainer_cfg,
                str(out_path.parent / "calibration" / label), train_loader, val_loader,
            )
            elapsed = time.monotonic() - started

            row = {
                **params,
                "config": label,
                "wall_seconds": round(elapsed, 3),
                "epochs_run": int(epochs_run),
                "epochs_requested": args.epochs,
                "batch_size": trainer_cfg.batch_size,
                **provenance,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            logging.info("  %.0f s over %d epoch(s)", elapsed, epochs_run)

    print(f"wrote {out_path}")
    print("Send this file to whoever is pooling the search, along with your trials.jsonl.")
    return 0


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def host_factors(rows: Sequence[Dict[str, Any]], reference: str) -> Dict[str, Dict[str, Any]]:
    """Per-host scaling factor against *reference*, with the evidence for it.

    Returns, per host, the ratio measured on each configuration, the median of
    those (the factor), and their spread -- which is what says whether the
    factor means anything.
    """
    by_host: Dict[str, Dict[str, float]] = defaultdict(dict)
    gpus: Dict[str, Any] = {}
    for row in rows:
        by_host[row["host"]][row["config"]] = float(row["wall_seconds"])
        gpus[row["host"]] = row.get("gpu_name")

    if reference not in by_host:
        raise KeyError(f"No calibration rows for reference host {reference!r}; have {sorted(by_host)}")
    baseline = by_host[reference]

    out: Dict[str, Dict[str, Any]] = {}
    for host, timings in sorted(by_host.items()):
        shared = sorted(set(timings) & set(baseline))
        ratios = {config: timings[config] / baseline[config] for config in shared}
        factor = statistics.median(ratios.values()) if ratios else None
        spread = (max(ratios.values()) - min(ratios.values())) / factor if factor else None
        out[host] = {
            "gpu": gpus.get(host),
            "ratios": ratios,
            "factor": factor,
            "spread": spread,
            "scalar_holds": spread is not None and spread <= _SPREAD_TOLERANCE,
            "missing": sorted(set(baseline) - set(timings)),
        }
    return out


def normalised_hours(ledger_rows, factors: Dict[str, Dict[str, Any]]):
    """Trial time restated as hours on the reference device.

    A host with no valid scalar factor is excluded rather than approximated:
    the total is meant to be defensible, and one machine's guess would not be
    visible in it afterwards.
    """
    total, unaccounted = 0.0, defaultdict(float)
    for row in ledger_rows:
        seconds = row.get("wall_seconds")
        if seconds is None:
            continue
        entry = factors.get(row.get("host"))
        if entry is None or not entry["scalar_holds"]:
            unaccounted[row.get("host")] += float(seconds)
            continue
        total += float(seconds) / entry["factor"]
    return total / 3600.0, {host: seconds / 3600.0 for host, seconds in unaccounted.items()}


def report(args) -> int:
    from scripts.merge_search_ledgers import read_ledger

    rows: List[Dict[str, Any]] = []
    for path in args.calibrations:
        rows.extend(read_ledger(path))
    if not rows:
        print("No calibration rows found.", file=sys.stderr)
        return 1

    try:
        factors = host_factors(rows, args.reference)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"reference: {args.reference} ({factors[args.reference]['gpu']})\n")
    legend = {row["config"]: row for row in rows}
    for label in sorted(legend):
        row = legend[label]
        described = ", ".join(
            f"{key}={row[key]}" for key in sorted(row)
            if key not in {"config", "wall_seconds", "epochs_run", "epochs_requested",
                           "batch_size", "host", "gpu_name", "git_commit",
                           "dataset_version", "_source"}
        )
        print(f"  {label}: {described}")
    print()
    print(f"{'host':<20} {'gpu':<28} {'factor':>7}  {'spread':>7}  configurations")
    for host, entry in factors.items():
        factor = "-" if entry["factor"] is None else f"{entry['factor']:.2f}x"
        spread = "-" if entry["spread"] is None else f"{entry['spread'] * 100:.0f}%"
        detail = ", ".join(f"{config}: {ratio:.2f}x" for config, ratio in sorted(entry["ratios"].items()))
        flag = "" if entry["scalar_holds"] else "  <- no single factor describes this machine"
        print(f"{host:<20} {str(entry['gpu']):<28} {factor:>7}  {spread:>7}  {len(entry['ratios'])}{flag}")
        if args.verbose and detail:
            print(f"{'':<20} {detail}")
        if entry["missing"]:
            print(f"{'':<20} did not run: {entry['missing']}")

    unstable = [host for host, entry in factors.items() if not entry["scalar_holds"]]
    if unstable:
        print(
            f"\nThe ratio is not constant across the space on {unstable}. "
            f"A scalar factor there would be wrong at both ends, so do not normalise it: "
            f"report that machine's hours as measured, on its own row."
        )

    if args.ledger:
        ledger_rows = [row for row in read_ledger(args.ledger) if row.get("status") == "completed"]
        hours, excluded = normalised_hours(ledger_rows, factors)
        print(f"\nsearch cost normalised to {args.reference}: {hours:.1f} GPU-hours")
        print(f"  from {len(ledger_rows)} completed trial(s)")
        for host, host_hours in sorted(excluded.items()):
            print(f"  excluded: {host_hours:.1f} raw h on {host} (no valid factor)")
        print(
            "\nQuote this beside the raw per-machine table from merge_search_ledgers.py, "
            "not instead of it."
        )

    return 0


# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("measure", help="Time the calibration configurations on this machine")
    run.add_argument("--run-id", required=True, help="A run that has already completed preprocess")
    run.add_argument("--model", default="tft", choices=sorted(_COST_PROXY))
    run.add_argument("--gpu", type=int, help="Which GPU to use; default is the first visible one")
    run.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=f"Fixed budget (default {DEFAULT_EPOCHS})")
    run.add_argument("--configs", type=int, default=3, help="How many configurations to time (default 3)")
    run.add_argument(
        "--target-normalizer-mode", choices=["encoder_floored", "global"],
        help="Default: whatever the run recorded, so the datasets match the search's.",
    )
    run.add_argument("--out", type=Path, help="Default: <run>/search/calibration.jsonl")
    run.set_defaults(func=measure)

    pool = sub.add_parser("report", help="Pool the machines' calibration files into factors")
    pool.add_argument("calibrations", nargs="+", type=Path)
    pool.add_argument("--reference", required=True, help="Host every other machine is expressed relative to")
    pool.add_argument("--ledger", type=Path, help="A merged trials.jsonl to restate in reference GPU-hours")
    pool.add_argument("--verbose", action="store_true", help="Show the per-configuration ratios")
    pool.set_defaults(func=report)

    args = parser.parse_args(argv)

    if args.command == "measure":
        # Before torch is imported anywhere: the measurement is per GPU, and a
        # trial sharing a card with anything else times the contention.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        elif "," in visible:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible.split(",")[0].strip()

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
