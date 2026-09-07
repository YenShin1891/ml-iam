"""What a trial row has to carry to survive being run on someone else's machine.

A search split across machines produces one ledger per machine, and the rows
in them are only comparable if they came from the same code and the same
data.  That is not checkable after the fact -- a row saying ``val_loss=0.31``
carries no trace of the dataset it was measured on -- so every row records its
own provenance as it is written, and the merge step verifies the ledgers agree
rather than assuming it.

Timing is recorded for the same reason and cannot be recovered later either.
Wall seconds from mismatched GPUs must not simply be summed; they are kept
per row, tagged with the host and device that produced them, so a cost figure
can be normalised to one reference device and audited against the raw table.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
from functools import lru_cache
from typing import Any, Dict, Optional

__all__ = ["git_commit", "host_name", "gpu_name", "dataset_version", "trial_provenance"]


# Per-machine settings -- which GPUs, which shard, which note -- live here
# and are meant to differ from the committed examples; each run records the
# file it was launched from in meta/.  Edits there do not make the code
# irreproducible, so they do not mark the tree dirty.
_NOT_CODE = ("configs/runs",)


@lru_cache(maxsize=4)
def git_commit(root: Optional[str] = None) -> Optional[str]:
    """The commit the trial ran, or None outside a checkout.

    Dirty working trees are marked, because a search run from uncommitted
    edits is not reproducible from the hash alone and the merge report should
    say so rather than quietly claiming it is.  *root* is the checkout to
    ask; it defaults to this repository.
    """
    if root is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance must never sink a search
        return None
    if not head:
        return None
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", ".", *(f":(exclude){path}" for path in _NOT_CODE)],
            cwd=root, capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return head
    return f"{head}-dirty" if dirty else head


@lru_cache(maxsize=1)
def host_name() -> str:
    """Machine the trial ran on; the key the per-host timing table groups by."""
    return socket.gethostname()


def gpu_name() -> Optional[str]:
    """Marketing name of the device in use, or None on CPU.

    Read per call rather than cached: search workers set
    ``CUDA_VISIBLE_DEVICES`` per process, so device 0 is a different card in
    each of them.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        return None


def dataset_version(run_id: str) -> Optional[str]:
    """The processed-data version this run was told to read.

    Taken from the run's own recorded config rather than from configs.data,
    which says what a run started *today* would read and would therefore
    label an old ledger with a new dataset's name.
    """
    from src.utils.utils import get_run_root

    path = os.path.join(get_run_root(run_id), "meta", "run_config.resolved.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle).get("dataset")
    except Exception as exc:  # noqa: BLE001
        logging.debug("No dataset version recorded for %s: %s", run_id, exc)
        return None


def trial_provenance(run_id: str) -> Dict[str, Any]:
    """The provenance fields shared by every trial row this process writes."""
    return {
        "host": host_name(),
        "gpu_name": gpu_name(),
        "git_commit": git_commit(),
        "dataset_version": dataset_version(run_id),
    }
