#!/usr/bin/env python

"""Run model training from a YAML/JSON run config.

This is the new unified entrypoint used by the Makefile targets.
It avoids brittle per-terminal env var exports by reading everything
from a run config file and applying env/CLI overrides consistently.

Example:
  python scripts/train_from_config.py --run configs/runs/xgb_example.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scripts.train import _ALLOWED_MODELS, _ALLOWED_PHASES, _assert_resume_run_exists


@dataclass(frozen=True)
class RunConfig:
    model: str
    phases: Tuple[str, ...]
    resume: Optional[str] = None  # the phase 'resume' named, when it did
    run_id: Optional[str] = None
    dataset: Optional[str] = None
    cuda_visible_devices: Optional[str] = None
    cuda_visible_devices_by_phase: Dict[str, Optional[str]] = field(default_factory=dict)
    two_window: bool = False
    keep_partial_targets: Optional[bool] = None
    target_normalizer_mode: Optional[str] = None  # TFT: "encoder_floored" or "global"
    search_shard: Optional[str] = None  # "index/count" when several machines split one search
    note: Optional[str] = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_run_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Run config not found: {path}")

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix == ".json":
        obj = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "YAML run config requires PyYAML. Install with: pip install pyyaml"
            ) from e
        obj = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported run config extension '{suffix}'. Use .yaml/.yml or .json")

    if obj is None:
        obj = {}
    if not isinstance(obj, dict):
        raise TypeError("Run config must be a mapping/object at the top level")
    return obj


def _normalize_cuda_visible_devices(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        # Keep empty string as an explicit value (CUDA_VISIBLE_DEVICES="" hides GPUs).
        return value.strip()
    if isinstance(value, (list, tuple)):
        tokens: List[str] = []
        for item in value:
            if item is None:
                continue
            tokens.append(str(item).strip())
        joined = ",".join([t for t in tokens if t])
        return joined
    # Numbers, bools, etc.
    return str(value).strip() or None


def _parse_cuda_by_phase(value: Any) -> Dict[str, Optional[str]]:
    """Parse a per-phase CUDA_VISIBLE_DEVICES mapping.

    Expected form:
      cuda_visible_devices:
        default: "0,1,2,3"
        search: "0,1,2,3"
        train: "0"
        test: "0"
        plot: "0"

    Values:
      - string/list: set env var to that value ("" explicitly hides GPUs)
      - null: explicitly UNSET the env var for that phase
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("per-phase cuda_visible_devices must be a mapping")

    out: Dict[str, Optional[str]] = {}
    for k, v in value.items():
        key = str(k).strip().lower()
        if not key:
            continue
        if key != "default" and key not in _ALLOWED_PHASES:
            raise ValueError(
                f"Unknown cuda_visible_devices phase key '{key}'. Allowed: ['default'] + {sorted(_ALLOWED_PHASES)}"
            )
        if v is None:
            out[key] = None
        else:
            out[key] = _normalize_cuda_visible_devices(v)
    return out


def _parse_search_shard(value: Any) -> Optional[str]:
    """Read the ``search_shard`` key into the canonical "index/count" string.

    Accepts either spelling, because both read naturally in a YAML that one
    co-author edits by hand:

      search_shard: "2/4"
      search_shard: {index: 2, count: 4}

    Validation happens here rather than in the trainer so a typo fails before
    a machine spends a day exploring the wrong slice -- or, worse, the same
    slice someone else is already running.
    """
    if value is None:
        return None

    from src.trainers.search import Shard

    if isinstance(value, dict):
        missing = {"index", "count"} - set(value)
        if missing:
            raise ValueError(f"search_shard mapping needs 'index' and 'count'; missing {sorted(missing)}")
        unknown = set(value) - {"index", "count"}
        if unknown:
            raise ValueError(f"Unknown search_shard key(s) {sorted(unknown)}; expected 'index' and 'count'")
        try:
            shard = Shard(int(value["index"]), int(value["count"]))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid search_shard {value!r}: {e}") from None
    else:
        shard = Shard.parse(str(value))

    return str(shard)


def _parse_config(obj: Dict[str, Any]) -> RunConfig:
    model_raw = obj.get("model")
    if not isinstance(model_raw, str) or not model_raw.strip():
        raise ValueError("run config must include non-empty 'model'")
    model = model_raw.strip().lower()
    if model in {"xgboost", "xgbregressor"}:
        model = "xgb"
    if model not in _ALLOWED_MODELS:
        raise ValueError(f"Unsupported model '{model}'. Allowed: {sorted(_ALLOWED_MODELS)}")

    # phases can come from 'phases' or 'resume'
    phases_obj = obj.get("phases")
    resume_obj = obj.get("resume")
    phases: List[str] = []
    if resume_obj is not None:
        if not isinstance(resume_obj, str):
            raise ValueError("'resume' must be a string phase name")
        resume_obj = resume_obj.strip().lower()
        phases = [resume_obj]
    elif phases_obj is None:
        # A new run needs preprocess first: every later phase reads the
        # cached data it writes.
        phases = list(_ALLOWED_PHASES)
    else:
        if not isinstance(phases_obj, (list, tuple)):
            raise ValueError("'phases' must be a list of phase names")
        phases = [str(p).strip().lower() for p in phases_obj]

    phases = [p for p in phases if p]
    if not phases:
        raise ValueError("No phases specified (empty 'phases'/'resume')")
    unknown = [p for p in phases if p not in _ALLOWED_PHASES]
    if unknown:
        raise ValueError(f"Unknown phase(s) {unknown}. Allowed: {sorted(_ALLOWED_PHASES)}")

    run_id = obj.get("run_id")
    if run_id is not None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("'run_id' must be a non-empty string when provided")
        run_id = run_id.strip()

    dataset = obj.get("dataset")
    if dataset is not None:
        if not isinstance(dataset, str) or not dataset.strip():
            dataset = None
        else:
            dataset = dataset.strip()

    cuda_visible_devices_by_phase: Dict[str, Optional[str]] = {}
    cuda_raw = obj.get("cuda_visible_devices")
    cuda_visible_devices: Optional[str] = None
    if isinstance(cuda_raw, dict):
        cuda_visible_devices_by_phase = _parse_cuda_by_phase(cuda_raw)
        # scalar field stays None; per-phase mapping drives env
    else:
        cuda_visible_devices = _normalize_cuda_visible_devices(cuda_raw)

    two_window = bool(obj.get("two_window", False))

    target_normalizer_mode = obj.get("target_normalizer_mode")
    if target_normalizer_mode is not None:
        allowed = {"encoder_floored", "global"}
        if target_normalizer_mode not in allowed:
            raise ValueError(f"'target_normalizer_mode' must be one of {sorted(allowed)}")

    keep_partial_targets = obj.get("keep_partial_targets")
    if keep_partial_targets is not None and not isinstance(keep_partial_targets, bool):
        raise ValueError("'keep_partial_targets' must be boolean when provided")

    search_shard = _parse_search_shard(obj.get("search_shard"))

    note = obj.get("note")
    if note is not None and not isinstance(note, str):
        raise ValueError("'note' must be a string when provided")

    return RunConfig(
        model=model,
        phases=tuple(phases),
        resume=resume_obj,
        run_id=run_id,
        dataset=dataset,
        cuda_visible_devices=cuda_visible_devices,
        cuda_visible_devices_by_phase=cuda_visible_devices_by_phase,
        two_window=two_window,
        keep_partial_targets=keep_partial_targets,
        target_normalizer_mode=target_normalizer_mode,
        search_shard=search_shard,
        note=note,
    )


def _train_script() -> Path:
    return _repo_root() / "scripts" / "train.py"


def _build_phase_argv(cfg: RunConfig, *, phase: str, run_id: str) -> List[str]:
    argv = [str(_train_script()), "--model", cfg.model, "--resume", phase, "--run_id", run_id]

    if cfg.dataset:
        argv.extend(["--dataset", cfg.dataset])

    if cfg.model == "tft" and cfg.two_window:
        argv.append("--two-window")

    if cfg.model == "tft" and cfg.target_normalizer_mode is not None:
        argv.extend(["--target-normalizer-mode", cfg.target_normalizer_mode])

    if cfg.keep_partial_targets is not None:
        argv.append("--keep-partial-targets" if cfg.keep_partial_targets else "--no-keep-partial-targets")

    if cfg.note:
        argv.extend(["--note", cfg.note])

    return argv


def _meta_dir(run_id: str) -> Path:
    from src.utils.utils import get_run_root

    meta_dir = Path(get_run_root(run_id)) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    return meta_dir


def _write_run_metadata(
    cfg: RunConfig,
    *,
    run_id: str,
    config_path: Path,
    env: Dict[str, str],
    cuda_by_phase_resolved: Dict[str, Optional[str]],
) -> None:
    """Record a new run's settings: what scripts/train.py reads back on resume."""
    meta_dir = _meta_dir(run_id)

    # Copy the original run file for provenance
    try:
        meta_dir.joinpath("run_config.original" + config_path.suffix.lower()).write_text(
            config_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    except Exception as e:
        logging.warning("Could not copy original run config for provenance: %s", e)

    # Write resolved config (JSON for easy parsing)
    resolved = {
        "model": cfg.model,
        "run_id": run_id,
        "dataset": cfg.dataset,
        "cuda_visible_devices": cfg.cuda_visible_devices,
        "cuda_visible_devices_by_phase": dict(cfg.cuda_visible_devices_by_phase),
        "cuda_visible_devices_resolved_by_phase": dict(cuda_by_phase_resolved),
        "two_window": cfg.two_window,
        "target_normalizer_mode": cfg.target_normalizer_mode,
        "search_shard": cfg.search_shard,
        "keep_partial_targets": cfg.keep_partial_targets,
        "note": cfg.note,
        "phases": list(cfg.phases),
    }
    meta_dir.joinpath("run_config.resolved.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    env_snapshot = {
        "global": {
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
            "SEARCH_SHARD": env.get("SEARCH_SHARD"),
            "DL_NUM_WORKERS": env.get("DL_NUM_WORKERS"),
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": env.get("MKL_NUM_THREADS"),
        },
        "cuda_visible_devices_by_phase": dict(cuda_by_phase_resolved),
    }
    meta_dir.joinpath("env.snapshot.json").write_text(
        json.dumps(env_snapshot, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_resume_record(
    cfg: RunConfig,
    *,
    run_id: str,
    config_path: Path,
    cuda_by_phase_resolved: Dict[str, Optional[str]],
) -> None:
    """Record a continuation without touching the run's original settings.

    run_config.resolved.json is what a resumed phase inherits its dataset,
    two_window, target_normalizer_mode and keep_partial_targets from.
    Rewriting it from a resume YAML that only names run_id and the phase
    used to blank those settings before the phase could read them.
    """
    record = {
        "config_path": str(config_path),
        "phases": list(cfg.phases),
        "cuda_visible_devices_resolved_by_phase": dict(cuda_by_phase_resolved),
        "overrides": {
            name: getattr(cfg, name)
            for name in ("dataset", "two_window", "target_normalizer_mode", "search_shard", "keep_partial_targets", "note")
            if getattr(cfg, name) not in (None, False)
        },
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _meta_dir(run_id).joinpath(f"run_config.resume.{stamp}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _continues_existing_run(cfg: RunConfig) -> bool:
    """Whether the config picks up a run that already exists on disk.

    That is any 'resume', and any explicit run_id whose phases do not start
    from preprocess: everything after preprocess reads what an earlier phase
    of that run cached.
    """
    return cfg.resume is not None or (cfg.run_id is not None and cfg.phases[0] != "preprocess")


def _validate_model_constraints(cfg: RunConfig) -> None:
    if cfg.run_id is not None and not cfg.run_id.startswith(f"{cfg.model}_"):
        raise ValueError(
            f"run_id '{cfg.run_id}' does not match model '{cfg.model}'. Expected prefix '{cfg.model}_'."
        )

    if cfg.run_id is None and cfg.phases[0] != "preprocess":
        raise ValueError(
            f"phases start with '{cfg.phases[0]}' but no run_id is set: a new run must begin "
            "with 'preprocess', and any later phase needs the run_id of a run that already ran it."
        )


def _allocate_run_id(cfg: RunConfig) -> str:
    from src.utils.utils import get_next_run_id

    if cfg.run_id:
        return cfg.run_id
    return get_next_run_id(cfg.model)


def _resolve_cuda_for_phase(cfg: RunConfig, phase: str) -> Tuple[bool, Optional[str]]:
    """Return (has_override, value) for CUDA_VISIBLE_DEVICES for a given phase.

    - If per-phase mapping specifies the phase key: use it (value may be None meaning unset).
    - Else if mapping specifies 'default': use it.
    - Else if scalar cuda_visible_devices is set: use it.
    - Else: no override.
    """
    if cfg.cuda_visible_devices_by_phase:
        if phase in cfg.cuda_visible_devices_by_phase:
            return True, cfg.cuda_visible_devices_by_phase[phase]
        if "default" in cfg.cuda_visible_devices_by_phase:
            return True, cfg.cuda_visible_devices_by_phase["default"]

    if cfg.cuda_visible_devices is not None:
        return True, cfg.cuda_visible_devices

    return False, None


def _run_phase(cfg: RunConfig, *, phase: str, run_id: str, base_env: Dict[str, str], repo_root: Path) -> None:
    argv = _build_phase_argv(cfg, phase=phase, run_id=run_id)
    cmd = [sys.executable, *argv]

    subprocess.run(cmd, check=True, cwd=str(repo_root), env=base_env)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run training from YAML/JSON run config")
    p.add_argument("--run", required=True, help="Path to run config (.yaml/.yml/.json)")
    args = p.parse_args(argv)

    repo_root = _repo_root()
    config_path = (repo_root / args.run).resolve() if not os.path.isabs(args.run) else Path(args.run).resolve()

    obj = _load_run_file(config_path)
    cfg = _parse_config(obj)
    _validate_model_constraints(cfg)

    run_id = _allocate_run_id(cfg)
    phases = cfg.phases

    continuing = _continues_existing_run(cfg)
    if continuing:
        # Before anything below creates the directory, which would let the
        # child's own existence check pass on an empty run.
        _assert_resume_run_exists(run_id)

    # Prepare env for subprocesses
    child_env = dict(os.environ)

    # Apply global/default CUDA setting (if provided) so phases that don't override are stable.
    if cfg.cuda_visible_devices_by_phase and "default" in cfg.cuda_visible_devices_by_phase:
        default_cuda = cfg.cuda_visible_devices_by_phase["default"]
        if default_cuda is None:
            child_env.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            child_env["CUDA_VISIBLE_DEVICES"] = default_cuda
    elif cfg.cuda_visible_devices is not None:
        child_env["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices

    # Which slice of a search this machine runs travels the same way
    # CUDA_VISIBLE_DEVICES does: it is a fact about the machine rather than
    # about the model, and every phase runs in its own subprocess.
    if cfg.search_shard is not None:
        child_env["SEARCH_SHARD"] = cfg.search_shard
    else:
        child_env.pop("SEARCH_SHARD", None)

    cuda_resolved: Dict[str, Optional[str]] = {}
    for phase in phases:
        has_override, value = _resolve_cuda_for_phase(cfg, phase)
        if has_override:
            cuda_resolved[phase] = value

    # Persist the run's settings before any phase runs, so a crashed run
    # still records intent.  A continuation keeps the original record.
    if continuing:
        _write_resume_record(
            cfg, run_id=run_id, config_path=config_path, cuda_by_phase_resolved=cuda_resolved,
        )
    else:
        _write_run_metadata(
            cfg,
            run_id=run_id,
            config_path=config_path,
            env=child_env,
            cuda_by_phase_resolved=cuda_resolved,
        )

    print(run_id, flush=True)

    for phase in phases:
        phase_env = dict(child_env)
        has_override, cuda_value = _resolve_cuda_for_phase(cfg, phase)
        if has_override:
            if cuda_value is None:
                phase_env.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                phase_env["CUDA_VISIBLE_DEVICES"] = cuda_value

        _run_phase(cfg, phase=phase, run_id=run_id, base_env=phase_env, repo_root=repo_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
