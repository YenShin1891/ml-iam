"""Utility functions for TFT training and inference."""

import os
import logging
from contextlib import contextmanager

import torch


_def_num_workers = None

_DIST_ENV_VARS = [
    "WORLD_SIZE",
    "RANK", 
    "LOCAL_RANK",
    "GLOBAL_RANK",
    "GROUP_RANK",
    "NODE_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
]


def get_default_num_workers() -> int:
    """Get optimal number of workers for data loading."""
    global _def_num_workers
    if _def_num_workers is not None:
        return _def_num_workers
        
    # Allow override via env var
    env_val = os.getenv("DL_NUM_WORKERS")
    if env_val is not None:
        try:
            _def_num_workers = max(1, int(env_val))
            return _def_num_workers
        except ValueError:
            pass
            
    try:
        cpu_count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    except Exception:
        cpu_count = os.cpu_count() or 2
    _def_num_workers = max(1, (cpu_count or 2) - 1)
    return _def_num_workers


def teardown_distributed() -> None:
    """Clean up distributed training environment."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            world_size = torch.distributed.get_world_size()
        except Exception:
            world_size = "unknown"
        logging.info(
            f"Destroying existing torch.distributed process group (world_size={world_size}) for single-process TFT prediction."
        )
        try:
            torch.distributed.destroy_process_group()
        except Exception as e:
            logging.warning(f"Failed to destroy process group cleanly: {e}")


@contextmanager
def single_gpu_env():
    """Temporarily restrict inference to one GPU.

    Keeps the first GPU the process was already allowed to see rather than
    forcing physical device 0, which a run pinned to CUDA_VISIBLE_DEVICES=3
    may not be allowed to use.  Stores and restores CUDA_VISIBLE_DEVICES and
    the distributed env vars; clearing the latter avoids DataLoader hangs or
    repeated loops from a stale multi-process environment when predicting
    with a single-process Trainer.
    """
    orig_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    orig_dist = {k: os.environ.get(k) for k in _DIST_ENV_VARS}
    if orig_cuda is None:
        first_visible = "0"
    else:
        # "" stays "": it means no GPU at all, not the first one.
        first_visible = orig_cuda.split(",")[0].strip()
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = first_visible
        for k in _DIST_ENV_VARS:
            os.environ.pop(k, None)
        logging.info("Forced single GPU inference with CUDA_VISIBLE_DEVICES=%s", first_visible)
        yield
    finally:
        if orig_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = orig_cuda
        # Restore dist vars
        for k, v in orig_dist.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v