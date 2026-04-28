"""TFT trainer with main orchestration functions."""

import hashlib
import json
import logging
import math
import os
import queue
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.model_selection import ParameterSampler

from configs.paths import RESULTS_PATH
from configs.models import TFTSearchSpace, TFTTrainerConfig
from src.utils.utils import get_run_root
from .tft_dataset import (
    build_datasets,
    create_combined_dataset,
    from_train_template,
    load_dataset_template,
    save_dataset_template,
)
from .tft_model import (
    create_dataloaders,
    create_final_trainer,
    create_search_trainer,
    create_tft_model,
    load_tft_checkpoint,
)
from .tft_utils import get_default_num_workers, single_gpu_env, teardown_distributed


_SEARCH_PARAM_KEYS = ("hidden_size", "lstm_layers", "dropout", "learning_rate")


def _is_completed_trial_row(row: Dict) -> bool:
    """True only for rows usable for dedup and best-param selection."""
    if not isinstance(row, dict):
        return False
    # Explicit completion marker is required to avoid counting partial trials.
    if row.get("status") != "completed":
        return False
    if not all(k in row for k in _SEARCH_PARAM_KEYS):
        return False
    try:
        val_loss = float(row["val_loss"])
    except Exception:
        return False
    if not math.isfinite(val_loss):
        return False
    return True


def _canonicalize_search_params(params: Dict) -> Dict:
    """Return canonical params dict for stable signatures / comparisons."""
    canonical = {}
    for k in _SEARCH_PARAM_KEYS:
        if k not in params:
            continue
        v = params[k]
        if isinstance(v, float):
            canonical[k] = float(format(v, ".12g"))
        else:
            canonical[k] = v
    return canonical


def _params_signature(params: Dict) -> str:
    """Stable signature for a parameter set (only search-relevant keys)."""
    canonical = _canonicalize_search_params(params)
    # Ensure all expected keys are present for a unique signature
    missing = [k for k in _SEARCH_PARAM_KEYS if k not in canonical]
    if missing:
        raise ValueError(f"Missing search params for signature: {missing}. Got keys={list(params.keys())}")
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def _trial_dirname_from_params(params: Dict) -> str:
    sig = _params_signature(params)
    h = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:10]
    return f"trial_{h}"


def _read_trials_ledger(run_id: str) -> List[Dict]:
    """Read search/trials.jsonl if present."""
    ledger_path = os.path.join(get_run_root(run_id), "search", "trials.jsonl")
    if not os.path.exists(ledger_path):
        return []
    out: List[Dict] = []
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _append_trials_ledger(run_id: str, rows: List[Dict]) -> None:
    """Append rows to search/trials.jsonl (best-effort)."""
    if not rows:
        return
    search_root = os.path.join(get_run_root(run_id), "search")
    os.makedirs(search_root, exist_ok=True)
    ledger_path = os.path.join(search_root, "trials.jsonl")
    with open(ledger_path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def _get_search_gpu_ids() -> List[int]:
    """Get physical GPU IDs available for search from CUDA_VISIBLE_DEVICES."""
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_env:
        return [int(x.strip()) for x in cuda_env.split(",") if x.strip()]
    n = torch.cuda.device_count()
    return list(range(n)) if n > 0 else []


def _get_best_epoch(trainer) -> int:
    """Extract the epoch of the best val_loss from an EarlyStopping callback."""
    best_epoch, _ = _get_best_score(trainer)
    return best_epoch


def _get_best_score(trainer):
    """Return (best_epoch, best_val_loss) from the EarlyStopping callback."""
    for cb in trainer.callbacks:
        if isinstance(cb, EarlyStopping):
            best_epoch = trainer.current_epoch - cb.wait_count
            best_val_loss = cb.best_score.item()
            return best_epoch, best_val_loss
    # Fallback: if no early stopping, use last epoch's metrics.
    return trainer.current_epoch, trainer.callback_metrics["val_loss"].item()


def _search_worker(gpu_id, trials, train_dataset, val_dataset, n_targets, trainer_cfg, result_queue, run_id, stage):
    """Run a batch of search trials on a single GPU (spawned subprocess)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Cap dataloader workers to avoid oversubscription across many GPU workers.
    # persistent_workers=False: reusing persistent workers across sequential trials in a
    # spawned subprocess causes a deadlock where workers spin at 100% CPU waiting for
    # a prefetch queue that the previous trial's training loop has already abandoned.
    num_workers = min(get_default_num_workers(), 4)
    train_loader = train_dataset.to_dataloader(
        train=True, batch_size=trainer_cfg.batch_size,
        num_workers=num_workers, persistent_workers=False,
    )
    val_loader = val_dataset.to_dataloader(
        train=False, batch_size=trainer_cfg.batch_size,
        num_workers=num_workers, persistent_workers=False,
    )

    for i, trial in enumerate(trials):
        params = trial["params"]
        trial_id = trial["trial_id"]
        logging.info("[%s] GPU %d - Trial %d/%d - Params: %s", stage, gpu_id, i + 1, len(trials), params)
        tft = create_tft_model(train_dataset, params, n_targets)
        log_dir = os.path.join(get_run_root(run_id), "search", "trials", trial_id)
        os.makedirs(log_dir, exist_ok=True)
        trainer = create_search_trainer(trainer_cfg, log_dir=log_dir)
        trainer.fit(model=tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
        best_epoch, best_val_loss = _get_best_score(trainer)
        result_queue.put({
            **_canonicalize_search_params(params),
            "val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),
            "trial_id": trial_id,
            "signature": trial["signature"],
            "status": "completed",
            "stage": stage,
        })
        logging.info(
            "[%s] GPU %d - Trial %d/%d - best_val_loss: %.4f best_epoch: %d",
            stage, gpu_id, i + 1, len(trials), best_val_loss, best_epoch,
        )


def _run_trials_once(
    train_dataset,
    val_dataset,
    n_targets: int,
    params_list: List[Dict],
    trainer_cfg: TFTTrainerConfig,
    run_id: str,
    stage: str,
) -> List[Dict]:
    """Run one stage of trials and return completed result rows."""
    if not params_list:
        return []

    trials = []
    for p in params_list:
        sig = _params_signature(p)
        trials.append({
            "params": p,
            "signature": sig,
            "trial_id": f"{stage}_{_trial_dirname_from_params(p)}",
        })

    gpu_ids = _get_search_gpu_ids()
    if len(gpu_ids) <= 1:
        train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, trainer_cfg.batch_size)
        results: List[Dict] = []
        for i, trial in enumerate(trials):
            params = trial["params"]
            logging.info("[%s] Trial %d/%d - Params: %s", stage, i + 1, len(trials), params)
            tft = create_tft_model(train_dataset, params, n_targets)
            log_dir = os.path.join(get_run_root(run_id), "search", "trials", trial["trial_id"])
            os.makedirs(log_dir, exist_ok=True)
            trainer = create_search_trainer(trainer_cfg, log_dir=log_dir)
            trainer.fit(model=tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
            epoch, val_loss = _get_best_score(trainer)
            results.append({
                **_canonicalize_search_params(params),
                "val_loss": float(val_loss),
                "best_epoch": int(epoch),
                "trial_id": trial["trial_id"],
                "signature": trial["signature"],
                "status": "completed",
                "stage": stage,
            })
            # Persist immediately so progress tracking and resume are up-to-date.
            try:
                _append_trials_ledger(run_id, [results[-1]])
            except Exception as exc:
                logging.warning("Failed to write trial to ledger (result kept in memory): %s", exc)
        return results

    # Multi-GPU parallel run
    import torch.multiprocessing as mp

    logging.info("[%s] Parallel search across %d GPUs: %s", stage, len(gpu_ids), gpu_ids)
    gpu_trials: List[List[Dict]] = [[] for _ in gpu_ids]
    for i, trial in enumerate(trials):
        gpu_trials[i % len(gpu_ids)].append(trial)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []
    for gpu_id, assigned in zip(gpu_ids, gpu_trials):
        if not assigned:
            continue
        p = ctx.Process(
            target=_search_worker,
            args=(gpu_id, assigned, train_dataset, val_dataset, n_targets, trainer_cfg, result_queue, run_id, stage),
        )
        p.start()
        processes.append(p)

    results = []

    # Stream completed trial rows from workers and append to ledger in real time.
    while True:
        try:
            row = result_queue.get(timeout=1)
            results.append(row)
            try:
                _append_trials_ledger(run_id, [row])
            except Exception as exc:
                logging.warning("Failed to write trial to ledger (result kept in memory): %s", exc)
        except queue.Empty:
            pass

        if all(not p.is_alive() for p in processes):
            break

    for p in processes:
        p.join(timeout=300)
        if p.is_alive():
            logging.warning("Search worker PID %d did not exit within 300 s; killing it.", p.pid)
            p.kill()
            p.join()

    # Drain any rows that were queued right before workers exited.
    while True:
        try:
            row = result_queue.get_nowait()
            results.append(row)
            try:
                _append_trials_ledger(run_id, [row])
            except Exception as exc:
                logging.warning("Failed to write trial to ledger (result kept in memory): %s", exc)
        except queue.Empty:
            break

    failed = [p for p in processes if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"{len(failed)} search worker(s) crashed during {stage} — check logs.")

    if not results:
        raise RuntimeError(f"TFT {stage} produced no results.")
    return results


def hyperparameter_search_tft(
    train_dataset,
    val_dataset,
    targets: List[str],
    run_id: str,
) -> Dict:
    """Perform hyperparameter search for TFT model.

    When multiple GPUs are available, trials are distributed across GPUs
    in parallel (one trial per GPU at a time, each GPU runs its share
    sequentially).  Falls back to sequential search on a single GPU.
    """
    search_cfg = TFTSearchSpace()
    trainer_cfg = TFTTrainerConfig()
    n_targets = len(targets)

    # Resume/dedup now uses only the append-only trials ledger.
    existing_ledger_rows = _read_trials_ledger(run_id)
    completed_ledger_rows = [r for r in existing_ledger_rows if _is_completed_trial_row(r)]
    skipped_rows = len(existing_ledger_rows) - len(completed_ledger_rows)
    if skipped_rows > 0:
        logging.warning(
            "Ignoring %d row(s) in search/trials.jsonl that are missing completed status or valid metrics.",
            skipped_rows,
        )

    all_params = list(
        ParameterSampler(
            search_cfg.param_dist,
            n_iter=search_cfg.search_iter_n,
            random_state=0,
        )
    )

    sig_to_params = {}
    for p in all_params:
        try:
            sig_to_params[_params_signature(p)] = p
        except Exception:
            continue

    stage1_epochs = int(getattr(search_cfg, "stage1_max_epochs", 40))
    stage2_top_k = int(getattr(search_cfg, "stage2_top_k", 10))

    stage1_sigs = set()
    stage2_sigs = set()
    for r in completed_ledger_rows:
        sig = r.get("signature") or _params_signature(r)
        stage = r.get("stage", "stage1")
        if stage == "stage2":
            stage2_sigs.add(sig)
        else:
            stage1_sigs.add(sig)

    # If stage2 is completed for a signature, treat stage1 as done too.
    explored_sigs = stage1_sigs | stage2_sigs
    stage1_remaining = [p for sig, p in sig_to_params.items() if sig not in explored_sigs]

    if explored_sigs:
        logging.info(
            "TFT resume: %d signature(s) already explored; %d/%d remain for stage1.",
            len(explored_sigs), len(stage1_remaining), len(sig_to_params),
        )
    else:
        logging.info("No prior completed TFT trials detected; starting two-stage search.")

    # Stage 1: cheap exploration
    stage1_cfg = TFTTrainerConfig()
    stage1_cfg.max_epochs = stage1_epochs
    stage1_cfg.batch_size = trainer_cfg.batch_size
    stage1_cfg.gradient_clip_val = trainer_cfg.gradient_clip_val
    stage1_cfg.patience = min(trainer_cfg.patience, max(1, stage1_epochs // 4))
    stage1_cfg.devices = trainer_cfg.devices

    stage1_new_results = _run_trials_once(
        train_dataset, val_dataset, n_targets,
        stage1_remaining, stage1_cfg, run_id, stage="stage1",
    )

    # Build stage1 pool from completed rows (legacy rows without stage count as stage1)
    stage1_pool = [
        r for r in completed_ledger_rows
        if r.get("stage", "stage1") != "stage2"
    ]
    stage1_pool.extend([r for r in stage1_new_results if _is_completed_trial_row(r)])

    if not stage1_pool:
        raise RuntimeError("TFT stage1 produced no completed trials.")

    # Stage 2: deeper rerank on top-K stage1 candidates
    ranked_stage1 = sorted(stage1_pool, key=lambda r: float(r["val_loss"]))
    top_signatures = []
    seen = set()
    for r in ranked_stage1:
        sig = r.get("signature") or _params_signature(r)
        if sig in seen:
            continue
        seen.add(sig)
        top_signatures.append(sig)
        if len(top_signatures) >= max(1, stage2_top_k):
            break

    stage2_remaining = [
        sig_to_params[sig]
        for sig in top_signatures
        if sig in sig_to_params and sig not in stage2_sigs
    ]

    logging.info(
        "TFT stage2 candidate signatures: %d (top_k=%d), remaining to run: %d",
        len(top_signatures), stage2_top_k, len(stage2_remaining),
    )

    stage2_new_results = _run_trials_once(
        train_dataset, val_dataset, n_targets,
        stage2_remaining, trainer_cfg, run_id, stage="stage2",
    )

    # Final best: prefer stage2-completed rows; fallback to stage1-completed rows.
    combined_completed = list(completed_ledger_rows)
    combined_completed.extend([r for r in stage1_new_results if _is_completed_trial_row(r)])
    combined_completed.extend([r for r in stage2_new_results if _is_completed_trial_row(r)])

    stage2_completed = [r for r in combined_completed if r.get("stage") == "stage2"]
    best_pool = stage2_completed if stage2_completed else [
        r for r in combined_completed if r.get("stage", "stage1") != "stage2"
    ]

    if not best_pool:
        raise RuntimeError("No completed TFT trials available to select best parameters.")

    best = min(best_pool, key=lambda r: float(r["val_loss"]))
    best_score = float(best["val_loss"])
    best_epoch = int(best.get("best_epoch", 0))
    best_params = {k: best[k] for k in _SEARCH_PARAM_KEYS}
    best_params["best_epoch"] = best_epoch
    logging.info(
        "Best TFT Params (%s): %s with Val Loss: %.4f (best epoch: %d)",
        "stage2" if stage2_completed else "stage1",
        best_params,
        best_score,
        best_epoch,
    )
    return best_params


def _search_sequential(
    train_dataset,
    val_dataset,
    n_targets,
    all_params,
    trainer_cfg,
    run_id,
    existing_ledger_rows=None,
) -> Dict:
    """Sequential search fallback for single-GPU / CPU environments."""
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, trainer_cfg.batch_size)

    new_results = []
    for i, params in enumerate(all_params):
        logging.info("TFT Search Iteration %d/%d - Params: %s", i + 1, len(all_params), params)
        tft = create_tft_model(train_dataset, params, n_targets)
        trial_id = _trial_dirname_from_params(params)
        log_dir = os.path.join(get_run_root(run_id), "search", "trials", trial_id)
        os.makedirs(log_dir, exist_ok=True)
        trainer = create_search_trainer(trainer_cfg, log_dir=log_dir)
        trainer.fit(model=tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
        epoch, val_loss = _get_best_score(trainer)

        row = {
            **_canonicalize_search_params(params),
            "val_loss": float(val_loss),
            "best_epoch": int(epoch),
            "trial_id": trial_id,
        }
        new_results.append(row)

        # Append each trial to ledger so a killed job can resume without redoing work
        try:
            _append_trials_ledger(run_id, [{
                **{k: row.get(k) for k in _SEARCH_PARAM_KEYS},
                "val_loss": float(row["val_loss"]),
                "best_epoch": int(row["best_epoch"]),
                "signature": _params_signature(row),
                "trial_id": trial_id,
                "status": "completed",
            }])
        except Exception as exc:
            logging.warning("Failed to write trial to ledger (result kept in memory): %s", exc)

    # Choose best across existing ledger + new
    combined = []
    if existing_ledger_rows:
        combined.extend([r for r in existing_ledger_rows if _is_completed_trial_row(r)])
    combined.extend([r for r in new_results if _is_completed_trial_row(r)])

    best = min(combined, key=lambda r: float(r["val_loss"]))
    best_params = {k: best[k] for k in _SEARCH_PARAM_KEYS}
    best_params["best_epoch"] = int(best.get("best_epoch", 0))
    best_score = float(best["val_loss"])
    logging.info("Best TFT Params: %s with Val Loss: %.4f (best epoch: %d)", best_params, best_score, best_params["best_epoch"])
    return best_params


def _is_primary_rank() -> bool:
    """Check if this is the primary DDP rank (or non-DDP)."""
    rank_vars = [
        os.getenv("LOCAL_RANK"),
        os.getenv("PL_TRAINER_GLOBAL_RANK"),
        os.getenv("GLOBAL_RANK"),
        os.getenv("RANK"),
    ]
    return all(rv in (None, "0") for rv in rank_vars)


def _split_by_groups(
    df: pd.DataFrame,
    group_cols: List[str],
    monitor_frac: float = 0.1,
    seed: int = 0,
) -> tuple:
    """Split a DataFrame by group IDs into train and monitor portions.

    Returns (train_df, monitor_df) where monitor_df contains ~monitor_frac
    of the unique groups (all their rows) and train_df contains the rest.
    """
    groups = df[group_cols].drop_duplicates()
    monitor_groups = groups.sample(frac=monitor_frac, random_state=seed)
    monitor_keys = set(map(tuple, monitor_groups.itertuples(index=False, name=None)))
    is_monitor = df[group_cols].apply(lambda r: tuple(r) in monitor_keys, axis=1)
    return df[~is_monitor].reset_index(drop=True), df[is_monitor].reset_index(drop=True)


def train_final_tft(
    train_dataset,
    val_dataset,
    targets: List[str],
    run_id: str,
    best_params: Dict,
    session_state: Optional[Dict] = None,
) -> None:
    """Train final TFT on combined train+val data with early stopping.

    Combines train and val data to maximise training signal, then holds out
    ~10% of groups as a monitoring-only validation set for early stopping.
    Per-epoch metrics are logged via CSVLogger.

    All DDP ranks must execute the same code path so they all reach
    trainer.fit() together.  Only rank 0 writes logs, summaries, and
    the dataset template.
    """
    from configs.data import INDEX_COLUMNS
    from pytorch_forecasting import TimeSeriesDataSet

    trainer_cfg = TFTTrainerConfig()
    final_dir = os.path.join(get_run_root(run_id), "final")
    final_ckpt_path = os.path.join(final_dir, "best.ckpt")

    primary = _is_primary_rank()

    os.makedirs(final_dir, exist_ok=True)

    # Pop search best_epoch — informational only, early stopping decides when to stop
    search_best_epoch = best_params.pop("best_epoch", None)

    # Get original DataFrames from session_state
    train_df = session_state.get("train_data") if session_state else None
    val_df = session_state.get("val_data") if session_state else None
    if not isinstance(train_df, pd.DataFrame) or not isinstance(val_df, pd.DataFrame):
        raise RuntimeError(
            "train_final_tft requires session_state with train_data and val_data DataFrames"
        )

    # Combine train+val, then re-split: ~90% for training, ~10% for monitoring
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    combined_train_df, monitor_df = _split_by_groups(combined_df, INDEX_COLUMNS)

    # Create TimeSeriesDataSets from the splits using the original dataset as template
    combined_train_dataset = TimeSeriesDataSet.from_dataset(train_dataset, combined_train_df)
    monitor_dataset = TimeSeriesDataSet.from_dataset(train_dataset, monitor_df)

    if primary:
        logging.info(
            "TFT final training dataset sizes -> combined=%d (train=%d monitor=%d)",
            len(combined_df), len(combined_train_dataset), len(monitor_dataset),
        )
        logging.info(
            "TFT final training for up to %d epochs with early stopping "
            "(patience=%d, search best_epoch=%s)",
            trainer_cfg.final_max_epochs,
            trainer_cfg.final_patience,
            search_best_epoch,
        )

        # Save combined training dataset template (used later for prediction)
        save_dataset_template(combined_train_dataset, run_id)

    n_targets = len(targets)
    tft_final = create_tft_model(combined_train_dataset, best_params, n_targets)

    num_workers = get_default_num_workers()
    train_loader = combined_train_dataset.to_dataloader(
        train=True,
        batch_size=trainer_cfg.batch_size,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_loader = monitor_dataset.to_dataloader(
        train=False,
        batch_size=trainer_cfg.batch_size,
        num_workers=num_workers,
        persistent_workers=True,
    )

    log_dir = os.path.join(final_dir, "logs")
    final_trainer = create_final_trainer(trainer_cfg, ckpt_path=final_ckpt_path, log_dir=log_dir)
    final_trainer.fit(model=tft_final, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Tear down DDP process group immediately so non-primary ranks can exit
    # without NCCL watchdog timeouts on rank 0.
    teardown_distributed()

    if primary:
        # ModelCheckpoint already saved the best weights to final_ckpt_path.
        # Extract best-epoch info from the checkpoint callback.
        from lightning.pytorch.callbacks import ModelCheckpoint as _MC
        best_ckpt_score = None
        for cb in final_trainer.callbacks:
            if isinstance(cb, _MC) and cb.best_model_path:
                best_ckpt_score = cb.best_model_score
                break

        callback_metrics = final_trainer.callback_metrics if hasattr(final_trainer, "callback_metrics") else {}

        def _metric_to_float(value):
            if isinstance(value, torch.Tensor):
                return float(value.detach().cpu().item())
            if isinstance(value, (int, float)):
                return float(value)
            try:
                return float(value)
            except Exception:
                return None

        train_loss = _metric_to_float(callback_metrics.get("train_loss"))
        val_loss = _metric_to_float(callback_metrics.get("val_loss"))
        best_val_loss = _metric_to_float(best_ckpt_score)
        stopped_epoch = final_trainer.current_epoch
        logging.info(
            "TFT final training done -> train_loss=%s val_loss=%s best_val_loss=%s stopped_epoch=%d",
            f"{train_loss:.6f}" if train_loss is not None else "NA",
            f"{val_loss:.6f}" if val_loss is not None else "NA",
            f"{best_val_loss:.6f}" if best_val_loss is not None else "NA",
            stopped_epoch,
        )

        summary = {
            "best_params": best_params,
            "combined_rows": len(combined_df),
            "train_set_rows": len(combined_train_dataset),
            "monitor_set_rows": len(monitor_dataset),
            "final_max_epochs": trainer_cfg.final_max_epochs,
            "final_patience": trainer_cfg.final_patience,
            "stopped_epoch": stopped_epoch,
            "search_best_epoch": search_best_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
        }
        summary_path = os.path.join(final_dir, "training_summary.json")
        try:
            with open(summary_path, "w", encoding="utf-8") as fp:
                json.dump(summary, fp, indent=2)
            logging.info("Saved TFT training summary to %s", summary_path)
        except Exception as exc:
            logging.warning("Failed to write TFT training summary: %s", exc)

    if session_state is not None and "tft_time_idx_column" not in session_state:
        session_state["tft_time_idx_column"] = getattr(train_dataset, "time_idx", "Step")


def predict_tft(session_state: Dict, run_id: str) -> np.ndarray:
    """Make predictions following the exact original tft_trajectory_plotting logic."""
    from src.trainers.evaluation import save_metrics

    test_data = session_state["test_data"]
    targets = session_state["targets"]

    # Follow original pattern exactly
    with single_gpu_env():
        # Best effort teardown if a process group is somehow still alive
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            teardown_distributed()
            if torch.distributed.is_initialized():
                raise RuntimeError("Failed to teardown existing distributed process group before prediction.")

        model = load_tft_checkpoint(run_id)

        logging.info("Building test dataset for TFT prediction using saved template...")
        train_template = load_dataset_template(run_id)

        try:
            test_dataset = from_train_template(
                train_template,
                test_data,
                mode="predict"  # This creates predict=True dataset
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to build test dataset from saved template. Ensure test_data columns and dtypes match the training schema: "
                f"{e}"
            )

        template_time_idx = getattr(train_template, "time_idx", None)
        template_group_ids = getattr(train_template, "group_ids", None)
        if not template_time_idx or not template_group_ids:
            raise ValueError("Saved dataset template is missing time_idx or group_ids; cannot align predictions.")
        logging.info("Loaded dataset template for prediction.")

        from configs.models import TFTTrainerConfig
        trainer_cfg = TFTTrainerConfig()
        test_loader = test_dataset.to_dataloader(
            train=False,
            batch_size=trainer_cfg.batch_size,
            num_workers=get_default_num_workers(),
            persistent_workers=True,
        )

        logging.info("Predicting with TFT model (forecast horizon only)...")
        returns = model.predict(test_loader, return_index=True)

        from pytorch_forecasting.models.base._base_model import Prediction as _PFPrediction
        if not isinstance(returns, _PFPrediction):
            raise RuntimeError(f"Unexpected predict() return type: {type(returns)}; expected pytorch_forecasting Prediction.")

        outputs = returns.output
        if isinstance(outputs, list):
            if len(outputs) == 0:
                raise RuntimeError("Prediction.output list is empty.")
            if not all(torch.is_tensor(o) for o in outputs):
                raise RuntimeError("All elements in Prediction.output list must be tensors.")
            preds_tensor = outputs[0] if len(outputs) == 1 else torch.stack(outputs, dim=-1)
        elif torch.is_tensor(outputs):
            preds_tensor = outputs
        else:
            raise RuntimeError(f"Unsupported Prediction.output type: {type(outputs)}")

        index_attr = getattr(returns, 'index', None)
        if isinstance(index_attr, list):
            dfs = [d for d in index_attr if isinstance(d, pd.DataFrame) and not d.empty]
            if not dfs:
                raise RuntimeError("Prediction.index list is empty or has no valid DataFrames.")
            index_df = pd.concat(dfs, ignore_index=True)
        elif isinstance(index_attr, pd.DataFrame):
            if index_attr.empty:
                raise RuntimeError("Prediction.index DataFrame is empty.")
            index_df = index_attr.copy()
        else:
            raise RuntimeError(f"Unsupported Prediction.index type: {type(index_attr)}")

        preds_flat = None  # will set below after optional expansion
        time_idx_name = template_time_idx
        group_ids = list(template_group_ids)

        # Normalize index dataframe
        if "time_idx" in index_df.columns and time_idx_name not in index_df.columns:
            index_df = index_df.rename(columns={"time_idx": time_idx_name})

        # If predictions are 3D (n_samples, pred_len, out_size) but index_df only has n_samples rows,
        # expand the index so each horizon step has its own row with incremented time index.
        if torch.is_tensor(preds_tensor) and preds_tensor.ndim == 3:
            n_samples, pred_len, out_size = preds_tensor.shape
            if len(index_df) == n_samples and pred_len > 1:
                logging.info(
                    "Expanding index_df for multi-step horizon: samples=%d, pred_len=%d", n_samples, pred_len
                )
                # Build expanded index
                expanded_rows = []
                base_cols = index_df.columns.tolist()
                if time_idx_name not in base_cols:
                    raise KeyError(
                        f"Time index column '{time_idx_name}' not found in prediction index DataFrame columns: {base_cols}"
                    )
                for i in range(n_samples):
                    base_row = index_df.iloc[i]
                    base_time = base_row[time_idx_name]
                    for h in range(pred_len):
                        new_row = base_row.copy()
                        # Assumption: decoder steps are consecutive increments
                        new_row[time_idx_name] = base_time + h
                        expanded_rows.append(new_row)
                index_df = pd.DataFrame(expanded_rows).reset_index(drop=True)
                # Flatten predictions accordingly
                preds_flat = preds_tensor.detach().cpu().numpy().reshape(n_samples * pred_len, out_size)
            else:
                preds_flat = preds_tensor.detach().cpu().numpy()
                if preds_flat.ndim == 3:
                    n_samples, pred_len, out_size = preds_flat.shape
                    preds_flat = preds_flat.reshape(n_samples * pred_len, out_size)
        else:
            preds_flat = preds_tensor.detach().cpu().numpy()
            if preds_flat.ndim == 3:
                n_samples, pred_len, out_size = preds_flat.shape
                preds_flat = preds_flat.reshape(n_samples * pred_len, out_size)

        # Evaluate only the forecast horizon rows returned by predict=True.
        key_cols = group_ids + [time_idx_name]
        # Collect reference columns (ensure presence in test_data)
        ref_cols = [c for c in key_cols + ['Year'] + targets if c in test_data.columns]
        horizon_df = index_df[key_cols].merge(
            test_data[ref_cols].drop_duplicates(key_cols),
            on=key_cols,
            how='left'
        )
        horizon_len = len(horizon_df)
        horizon_groups = horizon_df[group_ids].drop_duplicates().shape[0]
        test_group_total = test_data[group_ids].drop_duplicates().shape[0]
        logging.info(
            "TFT horizon coverage -> rows=%d unique_groups=%d (test_groups=%d)",
            horizon_len,
            horizon_groups,
            test_group_total,
        )
        target_offset = int(session_state.get("tft_target_offset", 0) or 0)
        if target_offset > 0:
            if time_idx_name in horizon_df.columns:
                warm_mask = horizon_df[time_idx_name] >= target_offset
                dropped = int((~warm_mask).sum())
                if dropped > 0:
                    logging.info(
                        "Warm start offset %d: dropping %d horizon rows where %s < offset",
                        target_offset,
                        dropped,
                        time_idx_name,
                    )
                    horizon_df = horizon_df.loc[warm_mask].reset_index(drop=True)
                    preds_flat = preds_flat[warm_mask.to_numpy()]
                    horizon_len = len(horizon_df)
                if horizon_len == 0:
                    logging.warning(
                        "All TFT prediction rows filtered by warm-start offset %d.",
                        target_offset,
                    )
            else:
                logging.warning(
                    "Warm start offset configured (%d) but index column '%s' missing in horizon_df.",
                    target_offset,
                    time_idx_name,
                )
        if preds_flat.shape[0] != len(horizon_df):
            logging.error(
                "After expansion attempt: preds_flat rows=%d, horizon_df rows=%d. First few time_idx in index_df: %s",
                preds_flat.shape[0], len(horizon_df), index_df[time_idx_name].head().tolist()
            )
            raise RuntimeError(
                f"Prediction rows ({preds_flat.shape[0]}) != horizon_df rows ({len(horizon_df)})."
            )

        y_true = horizon_df[targets].values

        # Handle RMSE predictions (standard case)
        y_pred = preds_flat
        valid_mask = (~np.isnan(y_true).any(axis=1)) & (~np.isnan(y_pred).any(axis=1))
        if valid_mask.any():
            save_metrics(run_id, y_true[valid_mask], y_pred[valid_mask])
        else:
            logging.warning("No valid rows to compute metrics (all targets or predictions contain NaNs).")

        removed_groups = None
        try:
            test_groups = set(map(tuple, test_data[group_ids].drop_duplicates().itertuples(index=False, name=None)))
            horizon_groups = set(map(tuple, horizon_df[group_ids].drop_duplicates().itertuples(index=False, name=None)))
            removed_groups = sorted(test_groups - horizon_groups)
            if removed_groups:
                preview = removed_groups[:5]
                logging.warning(
                    "TFT prediction dropped %d groups due to window constraints. Sample: %s",
                    len(removed_groups),
                    preview,
                )
        except Exception as exc:
            logging.warning("Failed to compute removed groups for TFT prediction: %s", exc)

        removed_count = len(removed_groups) if removed_groups else 0
        summary = {
            "horizon_rows": horizon_len,
            "horizon_unique_groups": horizon_df[group_ids].drop_duplicates().shape[0],
            "test_unique_groups": test_group_total,
            "removed_groups_count": removed_count,
            "removed_groups_sample": removed_groups[:5] if removed_groups else [],
        }
        prediction_summary_path = os.path.join(get_run_root(run_id), "final", "prediction_summary.json")
        try:
            os.makedirs(os.path.dirname(prediction_summary_path), exist_ok=True)
            with open(prediction_summary_path, "w", encoding="utf-8") as fp:
                json.dump(summary, fp, indent=2)
            logging.info("Saved TFT prediction summary to %s", prediction_summary_path)
        except Exception as exc:
            logging.warning("Failed to write TFT prediction summary: %s", exc)

        # Expose horizon dataframe and y_true for downstream plotting
        session_state['horizon_df'] = horizon_df
        session_state['horizon_y_true'] = y_true
        session_state['removed_groups'] = removed_groups

        # Return predictions matrix
        return y_pred


# Maintain backward compatibility
build_datasets = build_datasets

__all__ = [
    "build_datasets",
    "hyperparameter_search_tft", 
    "train_final_tft",
    "predict_tft",
]