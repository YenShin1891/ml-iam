#!/usr/bin/env python
"""Can this machine run one search worker per GPU, all at the same time?

The searches never use DDP: each worker is its own process pinned to one
card through CUDA_VISIBLE_DEVICES, exactly as this script pins its probes.
So a machine on which multi-GPU *training* is broken may still fan a search
out over every card -- or may not, if the fault is in using several cards
at once rather than in NCCL.  Thirty seconds here settles it before a
search finds out an hour in.

    python scripts/probe_gpus.py                # every card nvidia-smi lists
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 python scripts/probe_gpus.py

Each probe allocates a few hundred MB, runs matrix products for a few
seconds, and reports the card's name, memory and throughput.  A card that
hangs is reported as such rather than hanging the report: the parent waits
a bounded time and then gives up on it.
"""

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

SECONDS_PER_PROBE = 5.0
PROBE_TIMEOUT = 90.0


def _visible_gpu_tokens() -> List[str]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [token.strip() for token in env.split(",") if token.strip()]
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"CUDA_VISIBLE_DEVICES is unset and nvidia-smi failed: {exc}")
    return [line.strip() for line in out.splitlines() if line.strip()]


def _probe(token: str, seconds: float, result_queue) -> None:
    """Pin to one card and keep it busy; report what happened."""
    os.environ["CUDA_VISIBLE_DEVICES"] = token
    started = time.monotonic()
    report: Dict[str, object] = {"gpu": token}
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False for this card")
        report["name"] = torch.cuda.get_device_name(0)
        report["memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        x = torch.randn(4096, 4096, device="cuda")
        torch.cuda.synchronize()
        flops, deadline = 0.0, time.monotonic() + seconds
        while time.monotonic() < deadline:
            x = (x @ x) * 1e-4
            flops += 2 * 4096**3
        torch.cuda.synchronize()
        if not torch.isfinite(x).all():
            raise RuntimeError("matmul produced non-finite values")
        report["tflops"] = round(flops / (time.monotonic() - started) / 1e12, 1)
        report["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - the whole point is to report it
        report["status"] = f"FAILED: {type(exc).__name__}: {exc}"
    report["seconds"] = round(time.monotonic() - started, 1)
    result_queue.put(report)


def run_probes(tokens: List[str], concurrent: bool, seconds: float = SECONDS_PER_PROBE) -> List[Dict[str, object]]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    reports: Dict[str, Dict[str, object]] = {}

    def launch(batch: List[str]) -> None:
        processes = [ctx.Process(target=_probe, args=(token, seconds, queue)) for token in batch]
        for process in processes:
            process.start()
        deadline = time.monotonic() + PROBE_TIMEOUT
        while len([t for t in batch if t in reports]) < len(batch) and time.monotonic() < deadline:
            try:
                report = queue.get(timeout=1.0)
                reports[str(report["gpu"])] = report
            except Exception:  # noqa: BLE001 - queue.Empty, keep waiting
                pass
        for token, process in zip(batch, processes):
            if token not in reports:
                process.kill()
                reports[token] = {"gpu": token, "status": f"HUNG: no report within {PROBE_TIMEOUT:.0f}s"}
            process.join(timeout=5)

    if concurrent:
        launch(tokens)
    else:
        for token in tokens:
            launch([token])
    return [reports[token] for token in tokens]


def _print(title: str, reports: List[Dict[str, object]]) -> None:
    print(f"\n== {title}")
    print(f"{'gpu':>4}  {'status':8}  {'name':28} {'mem':>7} {'tflops':>7}  {'took':>5}")
    for r in reports:
        status = str(r.get("status", "?"))
        short = "ok" if status == "ok" else status.split(":")[0]
        print(
            f"{str(r['gpu']):>4}  {short:8}  {str(r.get('name', '-')):28} "
            f"{str(r.get('memory_gb', '-')):>7} {str(r.get('tflops', '-')):>7}  {str(r.get('seconds', '-')):>5}"
        )
        if status not in ("ok",):
            print(f"      {status}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seconds", type=float, default=SECONDS_PER_PROBE, help="busy time per probe")
    parser.add_argument("--skip-serial", action="store_true", help="only run the concurrent probe")
    args = parser.parse_args(argv)

    tokens = _visible_gpu_tokens()
    print(f"probing {len(tokens)} card(s): {', '.join(tokens)}")

    serial = None if args.skip_serial else run_probes(tokens, concurrent=False, seconds=args.seconds)
    if serial is not None:
        _print("one at a time", serial)
    together = run_probes(tokens, concurrent=True, seconds=args.seconds)
    _print("all at once (how a search runs)", together)

    good_alone = {r["gpu"] for r in (serial or together) if r.get("status") == "ok"}
    good_together = {r["gpu"] for r in together if r.get("status") == "ok"}
    print()
    if not good_together:
        print("VERDICT: no card completed a concurrent probe; run the search on one GPU (cuda_visible_devices.search: \"<id>\").")
        return 1
    if good_together == set(tokens):
        print(f"VERDICT: all {len(tokens)} cards work at once; the search can use cuda_visible_devices.search: \"{','.join(tokens)}\".")
        return 0
    lost = sorted(good_alone - good_together, key=int)
    bad = sorted(set(tokens) - good_alone, key=int)
    if bad:
        print(f"VERDICT: card(s) {', '.join(bad)} fail even alone; leave them out.")
    if lost:
        print(f"VERDICT: card(s) {', '.join(lost)} work alone but not alongside the others; the fault is in concurrent use.")
    usable = sorted(good_together, key=int)
    print(f"        usable together: cuda_visible_devices.search: \"{','.join(usable)}\"")
    return 1


if __name__ == "__main__":
    sys.exit(main())
