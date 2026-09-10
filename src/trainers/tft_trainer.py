"""TFT trainer with main orchestration functions."""

import hashlib
import json
import logging
import os
import queue
import time
import traceback
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping

from configs.models import TFTSearchSpace, TFTTrainerConfig
from .search import (
    allocate_pool,
    best_trial,
    canonicalize_params,
    completed_trials,
    is_completed_trial,
    params_signature,
    plan_two_stage_search,
    select_top_k_signatures,
    Shard,
    write_search_report,
)
from .provenance import trial_provenance
from src.utils.utils import get_run_root, is_primary_rank
from .tft_dataset import (
    build_datasets,
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


_SEARCH_PARAM_KEYS = tuple(TFTSearchSpace().param_keys)


def _is_completed_trial_row(row: Dict) -> bool:
    """True only for rows usable for dedup and best-param selection."""
    return is_completed_trial(row, _SEARCH_PARAM_KEYS)


def _canonicalize_search_params(params: Dict) -> Dict:
    """Return canonical params dict for stable signatures / comparisons."""
    return canonicalize_params(params, _SEARCH_PARAM_KEYS)


def _params_signature(params: Dict) -> str:
    """Stable signature for a parameter set (only search-relevant keys)."""
    return params_signature(params, _SEARCH_PARAM_KEYS)


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


@lru_cache(maxsize=4)
def _provenance(run_id: str) -> Dict[str, Any]:
    """Cached per process: shelling out to git once per trial is wasteful."""
    return trial_provenance(run_id)


def _trial_row(
    params: Dict,
    *,
    run_id: str,
    trial_id: str,
    signature: str,
    stage: str,
    status: str,
    wall_seconds: float,
    val_loss: Optional[float] = None,
    best_epoch: Optional[int] = None,
    epochs_run: Optional[int] = None,
    error: Optional[str] = None,
) -> Dict:
    """One ledger row: the result, what produced it, and what it cost.

    Rows are written for failed trials too, with a status other than
    "completed".  ``is_completed_trial`` rejects those, so they never enter a
    ranking, but they leave a trace of a configuration that was attempted and
    did not yield a score.  Without one, a trial that OOMed is indis-
    tinguishable from a trial nobody has reached yet, and a search that
    silently evaluated 47 of its 50 configurations reports as a complete one.
    """
    row = {
        **_canonicalize_search_params(params),
        "trial_id": trial_id,
        "signature": signature,
        "status": status,
        "stage": stage,
        "wall_seconds": round(float(wall_seconds), 3),
        **_provenance(run_id),
    }
    if val_loss is not None:
        row["val_loss"] = float(val_loss)
    if best_epoch is not None:
        row["best_epoch"] = int(best_epoch)
    if epochs_run is not None:
        row["epochs_run"] = int(epochs_run)
    if error is not None:
        row["error"] = str(error)[:500]
    return row


def _record(run_id: str, row: Dict, results: List[Dict]) -> None:
    """Keep a trial row in memory and persist it immediately.

    Persisting per trial rather than per stage is what makes a search
    resumable at trial granularity, and -- once several machines are running
    slices of the same search -- what lets a ledger be collected from a
    machine that has not finished.
    """
    results.append(row)
    try:
        _append_trials_ledger(run_id, [row])
    except Exception as exc:  # noqa: BLE001 - a lost row must not end the search
        logging.warning("Failed to write trial to ledger (result kept in memory): %s", exc)


def _search_shard() -> Optional[Shard]:
    """This machine's slice of the search, from the SEARCH_SHARD env var.

    Set by scripts/train_from_config.py out of the run config's
    ``search_shard`` key, the same route CUDA_VISIBLE_DEVICES already takes:
    which slice a machine runs is a fact about the machine, not about the
    model, and every phase runs in its own subprocess.
    """
    raw = os.environ.get("SEARCH_SHARD", "").strip()
    if not raw:
        return None
    shard = Shard.parse(raw)
    return None if shard.is_whole else shard


def _get_search_gpu_ids() -> List[int]:
    """Get physical GPU IDs available for search from CUDA_VISIBLE_DEVICES."""
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_env:
        return [int(x.strip()) for x in cuda_env.split(",") if x.strip()]
    n = torch.cuda.device_count()
    return list(range(n)) if n > 0 else []


def _get_best_score(trainer):
    """Return (best_epoch, best_val_loss) from the EarlyStopping callback.

    After fit, ``current_epoch`` counts completed epochs, so the last one ran
    at index ``current_epoch - 1``, and ``wait_count`` epochs have passed since
    the best.  Epochs are 0-based, as the progress log reports them.
    """
    for cb in trainer.callbacks:
        if isinstance(cb, EarlyStopping):
            best_epoch = trainer.current_epoch - 1 - cb.wait_count
            best_val_loss = cb.best_score.item()
            return best_epoch, best_val_loss
    # Fallback: if no early stopping, use last epoch's metrics.
    return trainer.current_epoch - 1, trainer.callback_metrics["val_loss"].item()


def _fit_search_trial(
    train_dataset, params, n_targets, trainer_cfg, log_dir, train_loader, val_loader,
) -> Tuple[int, float, int]:
    """Fit one configuration; return (best_epoch, best_val_loss, epochs_run).

    ``epochs_run`` is what the wall time paid for: early stopping means two
    trials of the same nominal budget can differ severalfold in cost, so a
    per-epoch rate -- the only figure comparable across machines whose
    schedules ended at different points -- cannot be derived without it.

    The model and trainer are locals, so they are released on return; the
    caller empties the CUDA cache before the next trial.
    """
    tft = create_tft_model(train_dataset, params, n_targets)
    os.makedirs(log_dir, exist_ok=True)
    trainer = create_search_trainer(trainer_cfg, log_dir=log_dir)
    trainer.fit(model=tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best_epoch, best_val_loss = _get_best_score(trainer)
    return best_epoch, best_val_loss, int(trainer.current_epoch)


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
        log_dir = os.path.join(get_run_root(run_id), "search", "trials", trial_id)
        started = time.monotonic()
        best, status, error = None, "completed", None
        try:
            best = _fit_search_trial(
                train_dataset, params, n_targets, trainer_cfg, log_dir, train_loader, val_loader,
            )
        except torch.cuda.OutOfMemoryError:
            logging.warning(
                "[%s] GPU %d - Trial %d/%d OOM — skipping trial %s. "
                "Consider reducing batch_size or excluding this GPU if another process holds memory.",
                stage, gpu_id, i + 1, len(trials), trial_id,
            )
            status = "oom"
        except Exception as exc:  # noqa: BLE001 - one trial's fault must not take the queue with it
            # A CUDA fault, a dataloader worker killed, an I/O error on the
            # run directory: whatever it was, a worker that died here took
            # every trial still queued behind it, and the parent could only
            # say "check logs" -- logs on another machine.  Record it where
            # the run directory is, and move on.  A sticky GPU fault fails
            # the remaining trials quickly; each is recorded, and the resume
            # plan reruns them, on this machine or another.
            status, error = "crashed", f"{type(exc).__name__}: {exc}"
            crash_file = os.path.join(log_dir, "crash.txt")
            try:
                os.makedirs(log_dir, exist_ok=True)
                with open(crash_file, "w", encoding="utf-8") as handle:
                    handle.write(traceback.format_exc())
            except OSError as write_exc:
                crash_file = f"(could not write crash.txt: {write_exc})"
            logging.error(
                "[%s] GPU %d - Trial %d/%d crashed: %s — traceback in %s; moving on.",
                stage, gpu_id, i + 1, len(trials), error, crash_file,
            )
        elapsed = time.monotonic() - started
        # The trial's model and trainer were locals of _fit_search_trial and
        # are gone by now; hand their cached blocks back before the next one.
        try:
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001 - after a CUDA fault this raises too
            logging.warning("[%s] GPU %d - could not empty the CUDA cache: %s", stage, gpu_id, exc)
        if best is None:
            result_queue.put(_trial_row(
                params, run_id=run_id, trial_id=trial_id, signature=trial["signature"],
                stage=stage, status=status, wall_seconds=elapsed, error=error,
            ))
            continue
        best_epoch, best_val_loss, epochs_run = best
        result_queue.put(_trial_row(
            params, run_id=run_id, trial_id=trial_id, signature=trial["signature"],
            stage=stage, status="completed", wall_seconds=elapsed,
            val_loss=best_val_loss, best_epoch=best_epoch, epochs_run=epochs_run,
        ))
        logging.info(
            "[%s] GPU %d - Trial %d/%d - best_val_loss: %.4f best_epoch: %d in %.0f s",
            stage, gpu_id, i + 1, len(trials), best_val_loss, best_epoch, elapsed,
        )


def _build_trials(params_list: List[Dict], stage: str) -> List[Dict]:
    """Pair each configuration with the identity its rows and logs are filed under."""
    return [
        {
            "params": params,
            "signature": _params_signature(params),
            "trial_id": f"{stage}_{_trial_dirname_from_params(params)}",
        }
        for params in params_list
    ]


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

    trials = _build_trials(params_list, stage)

    gpu_ids = _get_search_gpu_ids()
    if len(gpu_ids) <= 1:
        train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, trainer_cfg.batch_size)
        results: List[Dict] = []
        for i, trial in enumerate(trials):
            params = trial["params"]
            logging.info("[%s] Trial %d/%d - Params: %s", stage, i + 1, len(trials), params)
            log_dir = os.path.join(get_run_root(run_id), "search", "trials", trial["trial_id"])
            started = time.monotonic()
            try:
                epoch, val_loss, epochs_run = _fit_search_trial(
                    train_dataset, params, n_targets, trainer_cfg, log_dir, train_loader, val_loader,
                )
            except torch.cuda.OutOfMemoryError:
                # Same policy as the multi-GPU workers: one configuration that
                # does not fit must not end the search.  It is still recorded,
                # so the merged ledger shows a configuration that was never
                # scored rather than one nobody has started.
                logging.warning("[%s] Trial %d/%d OOM — skipping trial %s.", stage, i + 1, len(trials), trial["trial_id"])
                torch.cuda.empty_cache()
                _record(run_id, _trial_row(
                    params, run_id=run_id, trial_id=trial["trial_id"], signature=trial["signature"],
                    stage=stage, status="oom", wall_seconds=time.monotonic() - started,
                ), results)
                continue
            elapsed = time.monotonic() - started
            torch.cuda.empty_cache()
            _record(run_id, _trial_row(
                params, run_id=run_id, trial_id=trial["trial_id"], signature=trial["signature"],
                stage=stage, status="completed", wall_seconds=elapsed,
                val_loss=val_loss, best_epoch=epoch, epochs_run=epochs_run,
            ), results)
        return results

    # Multi-GPU parallel run
    import torch.multiprocessing as mp

    logging.info("[%s] Parallel search across %d GPUs: %s", stage, len(gpu_ids), gpu_ids)
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = _spawn_search_workers(
        ctx, trials, train_dataset, val_dataset, n_targets,
        trainer_cfg, result_queue, run_id, stage, gpu_ids,
    )
    results = _collect_search_results(processes, result_queue, run_id, stage)
    _assert_something_was_scored(results, stage)
    return results


def _spawn_search_workers(
    ctx,
    trials: List[Dict],
    train_dataset,
    val_dataset,
    n_targets: int,
    trainer_cfg: TFTTrainerConfig,
    result_queue,
    run_id: str,
    stage: str,
    gpu_ids: Sequence[int],
) -> List[Any]:
    """Start one worker per GPU in *gpu_ids*, all reporting to *result_queue*.

    Spawning is separated from collecting so that several groups of trials --
    which cannot share a worker, because a worker is handed one prebuilt
    TimeSeriesDataSet -- can still run at the same time against one queue.
    """
    per_gpu: List[List[Dict]] = [[] for _ in gpu_ids]
    for position, trial in enumerate(trials):
        per_gpu[position % len(gpu_ids)].append(trial)

    processes = []
    for gpu_id, assigned in zip(gpu_ids, per_gpu):
        if not assigned:
            continue
        p = ctx.Process(
            target=_search_worker,
            name=f"gpu{gpu_id}",
            args=(gpu_id, assigned, train_dataset, val_dataset, n_targets, trainer_cfg, result_queue, run_id, stage),
        )
        p.start()
        processes.append(p)
    return processes


def _collect_search_results(processes, result_queue, run_id: str, stage: str) -> List[Dict]:
    """Drain trial rows as workers produce them, writing each to the ledger."""
    results: List[Dict] = []

    # Stream completed trial rows from workers and append to ledger in real time.
    while True:
        try:
            row = result_queue.get(timeout=1)
            _record(run_id, row, results)
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
            _record(run_id, row, results)
        except queue.Empty:
            break

    failed = [p for p in processes if p.exitcode != 0]
    if failed:
        detail = ", ".join(f"{p.name} exit={p.exitcode}" for p in failed)
        raise RuntimeError(
            f"{len(failed)} search worker(s) died during {stage} without reporting: {detail}.  "
            "A negative code is a signal (-9 killed, usually the kernel's OOM killer; -11 segfault; "
            "-7 bus error, often /dev/shm full); a positive one is a Python error whose traceback is "
            "in the launcher's log on that machine (logs/train_*.log).  A trial that failed inside the "
            "worker is instead recorded in the ledger with status 'crashed' and its traceback in "
            "search/trials/<trial_id>/crash.txt.  Trials the dead workers had not finished are rerun "
            "by resuming the search."
        )
    return results


def _assert_something_was_scored(results: List[Dict], stage: str) -> None:
    """Refuse a stage in which every trial OOMed.

    Every trial having OOMed still leaves rows behind, so "no results" would
    no longer be true; what matters is that nothing was scored.
    """
    if any(row.get("status") == "completed" for row in results):
        return
    raise RuntimeError(
        f"TFT {stage} produced no completed trials ({len(results)} attempted). "
        "Check the ledger's status column — an all-OOM stage means the batch size "
        "or hidden_size range does not fit these GPUs."
    )


def _run_trials_by_encoder_length(
    datasets_by_encoder_length: Dict[int, tuple],
    n_targets: int,
    params_list: List[Dict],
    trainer_cfg: TFTTrainerConfig,
    run_id: str,
    stage: str,
) -> List[Dict]:
    """Run one stage, with every encoder length under way at once.

    Encoder length decides which prebuilt TimeSeriesDataSet a trial trains
    on, so trials that disagree about it cannot share a worker -- a worker is
    handed one dataset pair.  The lengths used to run back to back, which is
    free only while each group is at least as wide as the GPU pool.  Stage 2
    is not: a top-10 splitting 4/6 across eight GPUs runs as two waves that
    never use more than six cards, and costs the sum of two maxima rather
    than one.  So the pool is divided between the groups instead
    (:func:`allocate_pool`) and they run together, each worker still holding
    exactly one dataset pair.
    """
    if not params_list:
        return []

    groups = []
    for encoder_length in sorted({int(p["encoder_length"]) for p in params_list}):
        batch = [p for p in params_list if int(p["encoder_length"]) == encoder_length]
        datasets = datasets_by_encoder_length.get(encoder_length)
        if datasets is None:
            raise KeyError(
                f"No dataset built for encoder_length={encoder_length}; "
                f"have {sorted(datasets_by_encoder_length)}."
            )
        groups.append((encoder_length, datasets, batch))

    gpu_ids = _get_search_gpu_ids()
    allocation = (
        allocate_pool([len(batch) for _, _, batch in groups], len(gpu_ids))
        if len(gpu_ids) > 1 else []
    )

    if not allocation:
        # One GPU, or fewer GPUs than encoder lengths: nothing to divide, so
        # the groups run back to back with the whole pool each, as before.
        results: List[Dict] = []
        for encoder_length, (train_dataset, val_dataset), batch in groups:
            logging.info("[%s] %d trial(s) at encoder_length=%d", stage, len(batch), encoder_length)
            results.extend(_run_trials_once(
                train_dataset, val_dataset, n_targets, batch, trainer_cfg, run_id, stage,
            ))
        return results

    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []
    offset = 0
    for (encoder_length, (train_dataset, val_dataset), batch), count in zip(groups, allocation):
        gpus = list(gpu_ids[offset:offset + count])
        offset += count
        logging.info(
            "[%s] %d trial(s) at encoder_length=%d on %d GPU(s): %s",
            stage, len(batch), encoder_length, len(gpus), gpus,
        )
        processes.extend(_spawn_search_workers(
            ctx, _build_trials(batch, stage), train_dataset, val_dataset, n_targets,
            trainer_cfg, result_queue, run_id, stage, gpus,
        ))

    results = _collect_search_results(processes, result_queue, run_id, stage)
    # Checked over the stage rather than per encoder length: one length whose
    # trials all OOM is a finding the ledger records, not a reason to end a
    # search the other length is scoring fine.
    _assert_something_was_scored(results, stage)
    return results


def hyperparameter_search_tft(
    datasets_by_encoder_length: Dict[int, tuple],
    targets: List[str],
    run_id: str,
) -> Dict:
    """Perform hyperparameter search for TFT model.

    *datasets_by_encoder_length* maps each searched context length to its
    (train, val) TimeSeriesDataSet pair.  Every pair predicts the same steps
    of the same trajectories, so the encoder lengths are comparable.

    Returns the winning parameters, or None on a machine that ran one shard
    of stage 1 and has nothing to rank yet (see SEARCH_SHARD).

    When multiple GPUs are available, trials are distributed across GPUs
    in parallel (one trial per GPU at a time, each GPU runs its share
    sequentially).  Falls back to sequential search on a single GPU.
    """
    space = TFTSearchSpace()
    trainer_cfg = TFTTrainerConfig()
    n_targets = len(targets)
    logging.info("TFT search space: %s", space.summary())

    # Resume/dedup uses only the append-only trials ledger.
    existing_ledger_rows = _read_trials_ledger(run_id)
    skipped_rows = len(existing_ledger_rows) - len(completed_trials(existing_ledger_rows, _SEARCH_PARAM_KEYS))
    if skipped_rows > 0:
        logging.warning(
            "Ignoring %d row(s) in search/trials.jsonl that are missing completed status or valid metrics.",
            skipped_rows,
        )

    shard = _search_shard()
    if shard is not None:
        logging.info(
            "Search shard %s: this machine runs %d of the %d sampled configurations. "
            "Merge every machine's search/trials.jsonl before stage 2 decides anything.",
            shard, len(shard.members(space.sample())), space.n_trials,
        )

    plan = plan_two_stage_search(space, existing_ledger_rows, shard=shard)
    if plan.stage1_done or plan.stage2_done:
        logging.info(
            "TFT resume: %d stage-1 and %d stage-2 trial(s) already on disk; %d of this shard's %d remain for stage 1.",
            plan.stage1_done, plan.stage2_done, len(plan.stage1_pending), plan.stage1_mine,
        )
    else:
        logging.info("No prior completed TFT trials detected; starting two-stage search.")

    # Stage 1: cheap exploration under the reduced budget.
    stage1_cfg = TFTTrainerConfig()
    for attribute, value in space.stage1_budget.items():
        setattr(stage1_cfg, attribute, value)
    # Patience scales with the shortened schedule, or a trial would never stop
    # early inside it.
    stage1_cfg.patience = min(trainer_cfg.patience, max(1, stage1_cfg.max_epochs // 4))

    searched = set(space.distributions["encoder_length"].values)
    missing = searched - set(datasets_by_encoder_length)
    if missing:
        raise ValueError(
            f"The search covers encoder_length={sorted(searched)} but no dataset "
            f"was built for {sorted(missing)}."
        )

    stage1_new = _run_trials_by_encoder_length(
        datasets_by_encoder_length, n_targets,
        plan.stage1_pending, stage1_cfg, run_id, stage="stage1",
    )

    stage1_pool = [
        row for row in completed_trials(list(existing_ledger_rows) + stage1_new, _SEARCH_PARAM_KEYS)
        if row.get("stage") != "stage2"
    ]
    if not stage1_pool:
        raise RuntimeError("TFT stage1 produced no completed trials.")

    # Stage 2: refit the stage-1 leaders under the full budget.
    sig_to_params = plan.signature_to_params
    already_stage2 = {
        row.get("signature") or _params_signature(row)
        for row in completed_trials(existing_ledger_rows, _SEARCH_PARAM_KEYS)
        if row.get("stage") == "stage2"
    }
    stage2_signatures = select_top_k_signatures(
        stage1_pool, space.stage2_top_k, _SEARCH_PARAM_KEYS
    )
    if len(stage1_pool) < space.n_trials:
        if shard is not None:
            # The normal state of a machine that ran its slice: the shortlist
            # can only be ranked once every machine's rows are pooled, and
            # refitting the best of a third of the search at the full budget
            # would cost hours and put stage-2 rows for the wrong candidates
            # in the ledger.  Stop here; there is no winner yet.
            logging.info(
                "Search shard %s: stage 1 finished on this machine (%d of %d stage-1 trials "
                "are in this ledger).  Stage 2 waits for the merged ledger: pool every "
                "machine's search/trials.jsonl with scripts/merge_search_ledgers.py, then "
                "resume the search on one machine without a shard.",
                shard, len(stage1_pool), space.n_trials,
            )
            return None
        logging.warning(
            "Ranking stage 2 on %d completed stage-1 trial(s), fewer than the %d the space samples. "
            "Some trials are missing — check the ledger's status column.",
            len(stage1_pool), space.n_trials,
        )
    stage2_shortlist = shard.members(stage2_signatures) if shard is not None else stage2_signatures
    stage2_pending = [
        sig_to_params[sig] for sig in stage2_shortlist
        if sig in sig_to_params and sig not in already_stage2
    ]
    logging.info(
        "TFT stage2 candidates: %d (top_k=%d), assigned to this shard: %d, remaining to run: %d",
        len(stage2_signatures), space.stage2_top_k, len(stage2_shortlist), len(stage2_pending),
    )

    stage2_new = _run_trials_by_encoder_length(
        datasets_by_encoder_length, n_targets,
        stage2_pending, trainer_cfg, run_id, stage="stage2",
    )

    all_completed = completed_trials(
        list(existing_ledger_rows) + stage1_new + stage2_new, _SEARCH_PARAM_KEYS
    )
    stage2_completed = [row for row in all_completed if row.get("stage") == "stage2"]
    # Stage 2 measures each candidate under the full schedule, so it decides
    # whenever it ran at all; stage 1 is only a fallback for a search that
    # never reached it.
    best_pool = stage2_completed or [row for row in all_completed if row.get("stage") != "stage2"]
    if not best_pool:
        raise RuntimeError("No completed TFT trials available to select best parameters.")

    try:
        write_search_report(os.path.join(get_run_root(run_id), "search"), space, all_completed)
    except Exception as exc:  # noqa: BLE001 - a report must not sink a search
        logging.warning("Failed to write TFT search report: %s", exc)

    best = best_trial(best_pool)
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


def train_final_tft(
    train_dataset,
    val_dataset,
    targets: List[str],
    run_id: str,
    best_params: Dict,
    session_state: Optional[Dict] = None,
) -> None:
    """Train final TFT using the same train/val split as the search.

    Uses train_dataset for training and val_dataset for early stopping,
    matching the exact data regime the hyperparameter search used.
    Per-epoch metrics are logged via CSVLogger.

    All DDP ranks must execute the same code path so they all reach
    trainer.fit() together.  Only rank 0 writes logs, summaries, and
    the dataset template.
    """

    trainer_cfg = TFTTrainerConfig()
    final_dir = os.path.join(get_run_root(run_id), "final")
    final_ckpt_path = os.path.join(final_dir, "best.ckpt")

    primary = is_primary_rank()

    os.makedirs(final_dir, exist_ok=True)

    # best_epoch is search bookkeeping, not a model parameter: early stopping
    # decides when to stop.  Drop it from a copy so the caller's dict survives.
    best_params = dict(best_params)
    search_best_epoch = best_params.pop("best_epoch", None)

    if primary:
        logging.info(
            "TFT final training dataset sizes -> train=%d val=%d",
            len(train_dataset), len(val_dataset),
        )
        logging.info(
            "TFT final training for up to %d epochs with early stopping "
            "(patience=%d, search best_epoch=%s)",
            trainer_cfg.final_max_epochs,
            trainer_cfg.final_patience,
            search_best_epoch,
        )

        # Save training dataset template (used later for prediction)
        save_dataset_template(train_dataset, run_id)

    n_targets = len(targets)
    tft_final = create_tft_model(train_dataset, best_params, n_targets)

    num_workers = get_default_num_workers()
    train_loader = train_dataset.to_dataloader(
        train=True,
        batch_size=trainer_cfg.batch_size,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_loader = val_dataset.to_dataloader(
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
            "train_set_rows": len(train_dataset),
            "val_set_rows": len(val_dataset),
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


def predict_tft(
    session_state: Dict,
    run_id: str,
    *,
    skip_metrics: bool = False,
    metrics_filename: str = "performance.csv",
    prediction_summary_filename: str = "prediction_summary.json",
) -> np.ndarray:
    """Make predictions following the exact original tft_trajectory_plotting logic.

    *metrics_filename* and *prediction_summary_filename* let callers evaluate
    non-test splits (e.g. train/val, for over/underfitting diagnostics) without
    overwriting the canonical test-set artifacts under metrics/ and final/.
    """
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
            persistent_workers=False,
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
        from configs.data import POPULATION_COLUMN
        from src.data.preprocess import observed_mask_columns, observed_mask_from_frame

        key_cols = group_ids + [time_idx_name]
        # Reference columns carried onto the horizon rows.  The __observed
        # masks must come along: without them the metrics scored the
        # zero-filled and interpolated targets as ground truth, while the LSTM
        # and XGBoost paths masked them out.
        ref_cols = [
            c for c in key_cols + ['Year'] + targets + observed_mask_columns(targets) + [POPULATION_COLUMN]
            if c in test_data.columns
        ]
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

        # Convert per-capita target/prediction back to absolute units.
        from src.data.preprocess import denormalize_by_population
        population = horizon_df[POPULATION_COLUMN].values
        y_true = denormalize_by_population(horizon_df[targets].values, population)

        # Handle RMSE predictions (standard case)
        y_pred = denormalize_by_population(preds_flat, population)
        # Save metrics unless caller will compute them on combined data (e.g.
        # two-window prediction calls predict_tft for single-window fallback).
        if not skip_metrics:
            # horizon_df, not test_data: predictions cover the forecast horizon
            # only, and the rows were checked against it just above.
            save_metrics(run_id, y_true, y_pred, horizon_df,
                         observed_mask=observed_mask_from_frame(horizon_df, targets),
                         metrics_filename=metrics_filename)

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
        prediction_summary_path = os.path.join(get_run_root(run_id), "final", prediction_summary_filename)
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

__all__ = [
    "build_datasets",
    "hyperparameter_search_tft", 
    "train_final_tft",
    "predict_tft",
]