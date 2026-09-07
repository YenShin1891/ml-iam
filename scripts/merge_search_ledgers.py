#!/usr/bin/env python

"""Merge the trial ledgers of a search that several machines ran in parallel.

Each machine writes its own append-only ``search/trials.jsonl``.  Because a
trial is identified by the hash of its parameters rather than by its position
in a queue, merging is close to concatenation: duplicates collapse, and a
machine that died mid-slice needs no reconciliation beyond handing its slice
to someone else.

What is not automatic is the checking, which is the point of this script.  Two
ledgers are only comparable if they came from the same commit and the same
processed dataset, and neither fact can be recovered from a row's val_loss
afterwards -- so the rows carry it, and this refuses to merge ledgers that
disagree.  It then says which of the sampled configurations still have no
score, separating the ones nobody reached from the ones that OOMed, since a
search that silently evaluated 47 of 50 configurations otherwise reports as a
complete one.

Usage:
  python scripts/merge_search_ledgers.py --model tft --out merged.jsonl \\
      host_a/trials.jsonl host_b/trials.jsonl host_c/trials.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.trainers.search import SearchSpace, is_completed_trial, params_signature  # noqa: E402

_SPACES = {
    "tft": ("configs.models.tft_search", "TFTSearchSpace"),
    "lstm": ("configs.models.lstm", "LSTMSearchSpace"),
    "xgb": ("configs.models.xgb_search", "XGBSearchSpace"),
}

# The fields that must agree across ledgers for their scores to be comparable.
_PROVENANCE_KEYS = ("git_commit", "dataset_version")
# Settings that shape the validation loss without appearing in a trial row.
# Two ledgers that disagree on the target normaliser are scored on different
# scales, and no ranking across them means anything.
_RUN_SETTINGS = ("dataset", "target_normalizer_mode", "two_window", "keep_partial_targets")


def load_space(model: str) -> SearchSpace:
    from importlib import import_module

    module, name = _SPACES[model]
    return getattr(import_module(module), name)()


def read_ledger(path: Path) -> List[Dict[str, Any]]:
    """Rows of one ledger, each tagged with the file it came from.

    A malformed line is reported rather than skipped silently: it is usually
    a row that was half-written when a machine was killed, and knowing which
    trial that was is how you know to re-run it.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  ! {path}:{number} is not valid JSON ({exc.msg}); skipped", file=sys.stderr)
                continue
            row["_source"] = str(path)
            rows.append(row)
    return rows


def merge(
    rows: Iterable[Dict[str, Any]], keys: Sequence[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collapse rows to one per (stage, configuration); return (kept, dropped).

    A completed row always beats an unfinished or OOMed one for the same
    configuration -- a trial that fits on a larger GPU is not refuted by the
    machine it did not fit on.  Between two completed rows the first wins,
    and the loser is returned rather than discarded so the duplicated GPU
    time can be reported.
    """
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    dropped: List[Dict[str, Any]] = []
    for row in rows:
        signature = row.get("signature")
        if not signature:
            try:
                signature = params_signature(row, keys)
            except ValueError:
                dropped.append(row)
                continue
        key = (str(row.get("stage", "stage1")), signature)
        incumbent = best.get(key)
        if incumbent is None:
            best[key] = row
            continue
        if is_completed_trial(incumbent, keys) or not is_completed_trial(row, keys):
            dropped.append(row)
        else:
            dropped.append(incumbent)
            best[key] = row
    return list(best.values()), dropped


def check_provenance(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Complaints about ledgers that cannot honestly be pooled."""
    problems: List[str] = []
    for key in _PROVENANCE_KEYS:
        by_value: Dict[Any, set] = defaultdict(set)
        for row in rows:
            by_value[row.get(key)].add(row.get("host") or row.get("_source"))
        if len(by_value) > 1:
            detail = "; ".join(
                f"{value!r} on {sorted(hosts)}" for value, hosts in sorted(by_value.items(), key=lambda kv: str(kv[0]))
            )
            problems.append(f"{key} disagrees across ledgers: {detail}")
        elif None in by_value:
            problems.append(f"{key} is missing from every row (ledger predates provenance recording)")
    dirty = sorted({row.get("git_commit") for row in rows if str(row.get("git_commit") or "").endswith("-dirty")})
    if dirty:
        problems.append(f"trials ran from uncommitted edits: {dirty}")
    return problems


def run_settings(ledger: Path) -> Optional[Dict[str, Any]]:
    """What the run that wrote *ledger* was configured with, from its meta/.

    A ledger still in its run directory has meta/run_config.resolved.json
    two levels up.  One that has been copied elsewhere has no record, and
    None says so rather than pretending the settings were checked.
    """
    meta = ledger.resolve().parent.parent / "meta" / "run_config.resolved.json"
    try:
        recorded = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return {key: recorded.get(key) for key in _RUN_SETTINGS}


def check_run_settings(settings_by_ledger: Dict[str, Optional[Dict[str, Any]]]) -> List[str]:
    """Complaints about runs whose recorded settings disagree."""
    problems: List[str] = []
    known = {path: found for path, found in settings_by_ledger.items() if found is not None}
    for key in _RUN_SETTINGS:
        by_value: Dict[str, set] = defaultdict(set)
        for path, found in known.items():
            by_value[json.dumps(found.get(key), sort_keys=True)].add(path)
        if len(by_value) > 1:
            detail = "; ".join(f"{value} in {sorted(paths)}" for value, paths in sorted(by_value.items()))
            problems.append(f"{key} disagrees across runs: {detail}")
    return problems


def coverage(space: SearchSpace, rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Split the sampled configurations by what the merged ledger says of them."""
    keys = space.param_keys
    stage1 = [row for row in rows if row.get("stage") != "stage2"]
    scored = {row.get("signature") for row in stage1 if is_completed_trial(row, keys)}
    attempted = {row.get("signature"): row for row in stage1 if not is_completed_trial(row, keys)}

    done, failed, untouched = [], [], []
    for params in space.sample():
        signature = params_signature(params, keys)
        if signature in scored:
            done.append(params)
        elif signature in attempted:
            failed.append({**params, "status": attempted[signature].get("status", "incomplete")})
        else:
            untouched.append(params)
    return {"completed": done, "failed": failed, "not_started": untouched}


def timing_table(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Raw cost per machine: the appendix table a normalised figure is audited against.

    Deliberately not summed into one number.  Hours from different GPUs
    measure different things, and adding them produces a total that describes
    no machine that exists.
    """
    by_host: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_host[(row.get("host"), row.get("gpu_name"))].append(row)

    table = []
    for (host, gpu), group in sorted(by_host.items(), key=lambda kv: str(kv[0])):
        seconds = [float(row["wall_seconds"]) for row in group if row.get("wall_seconds") is not None]
        table.append({
            "host": host,
            "gpu": gpu,
            "trials": len(group),
            "completed": sum(1 for row in group if row.get("status") == "completed"),
            "wall_hours": round(sum(seconds) / 3600.0, 2) if seconds else None,
            "median_trial_seconds": round(statistics.median(seconds)) if seconds else None,
        })
    return table


def _print_report(space: SearchSpace, kept, dropped, problems, cover) -> None:
    print(f"merged {len(kept)} trial row(s); {len(dropped)} duplicate(s) collapsed")
    wasted = sum(float(row.get("wall_seconds") or 0.0) for row in dropped)
    if wasted:
        print(f"  duplicated GPU time: {wasted / 3600.0:.2f} h")

    print("\nper-machine cost (raw, not comparable across GPUs):")
    for entry in timing_table(kept):
        hours = "-" if entry["wall_hours"] is None else f"{entry['wall_hours']:>7.2f} h"
        median = "-" if entry["median_trial_seconds"] is None else f"{entry['median_trial_seconds']:>6d} s/trial"
        print(
            f"  {str(entry['host']):<20} {str(entry['gpu']):<28} "
            f"{entry['completed']:>3}/{entry['trials']:<3} trials  {hours}  {median}"
        )

    print(f"\nstage-1 coverage of the {space.n_trials} sampled configurations:")
    print(f"  completed   {len(cover['completed'])}")
    print(f"  failed      {len(cover['failed'])}")
    print(f"  not started {len(cover['not_started'])}")
    for params in cover["failed"]:
        print(f"    ! {params.get('status')}: {ic(params)}")
    for params in cover["not_started"]:
        print(f"    · unrun: {ic(params)}")

    if problems:
        print("\nprovenance:")
        for problem in problems:
            print(f"  ! {problem}")


def ic(params: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(params.items()) if k != "status")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ledgers", nargs="+", type=Path, help="Each machine's search/trials.jsonl")
    parser.add_argument("--model", required=True, choices=sorted(_SPACES), help="Whose search space to audit against")
    parser.add_argument("--out", type=Path, help="Write the merged ledger here")
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Merge anyway when ledgers disagree on commit or dataset. The result is not a comparable search.",
    )
    args = parser.parse_args(argv)

    missing = [path for path in args.ledgers if not path.exists()]
    if missing:
        parser.error(f"No such ledger(s): {[str(p) for p in missing]}")

    space = load_space(args.model)
    rows: List[Dict[str, Any]] = []
    for path in args.ledgers:
        found = read_ledger(path)
        print(f"{path}: {len(found)} row(s)")
        rows.extend(found)

    kept, dropped = merge(rows, space.param_keys)
    problems = check_provenance(kept)
    settings = {str(path): run_settings(path) for path in args.ledgers}
    for path, found in settings.items():
        if found is None:
            print(f"  ! {path} has no meta/run_config.resolved.json beside it; its run settings were not checked")
    problems.extend(check_run_settings(settings))
    cover = coverage(space, kept)
    _print_report(space, kept, dropped, problems, cover)

    if problems and not args.allow_mismatch:
        print(
            "\nRefusing to write a merged ledger: the trials above did not measure the same thing. "
            "Re-run the offending machine's slice on the agreed commit and dataset, or pass "
            "--allow-mismatch if you have established the difference is immaterial.",
            file=sys.stderr,
        )
        return 1

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            for row in sorted(kept, key=lambda r: (str(r.get("stage")), str(r.get("signature")))):
                handle.write(json.dumps({k: v for k, v in row.items() if k != "_source"}, sort_keys=True) + "\n")
        print(f"\nwrote {args.out}")
        if cover["not_started"] or cover["failed"]:
            print(
                "Stage 2 ranks whatever this file contains, so run the missing configurations "
                "before letting it choose."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
