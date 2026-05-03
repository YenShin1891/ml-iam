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
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


_ALLOWED_MODELS = {"xgb", "lstm", "tft"}
_ALLOWED_PHASES = {"preprocess", "search", "train", "test", "plot"}


def _assert_run_dir_exists(run_id: str) -> None:
    from src.utils.utils import get_run_root

    run_root = Path(get_run_root(run_id))
    if not run_root.exists() or not run_root.is_dir():
        raise FileNotFoundError(
            f"Cannot resume run '{run_id}': run directory does not exist at {run_root}. "
            "Check run_id or run preprocess/full pipeline first."
        )


@dataclass(frozen=True)
class RunConfig:
    model: str
    phases: Tuple[str, ...]
    run_id: Optional[str] = None
    dataset: Optional[str] = None
    cuda_visible_devices: Optional[str] = None
    cuda_visible_devices_by_phase: Dict[str, Optional[str]] = field(default_factory=dict)
    lag_required: Optional[bool] = None
    two_window: bool = False
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


def _parse_config(obj: Dict[str, Any], *, config_path: Path) -> RunConfig:
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
        phases = [resume_obj.strip().lower()]
    elif phases_obj is None:
        phases = ["search", "train", "test", "plot"]
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

    lag_required = obj.get("lag_required")
    if lag_required is not None and not isinstance(lag_required, bool):
        raise ValueError("'lag_required' must be boolean when provided")

    two_window = bool(obj.get("two_window", False))

    note = obj.get("note")
    if note is not None and not isinstance(note, str):
        raise ValueError("'note' must be a string when provided")

    return RunConfig(
        model=model,
        phases=tuple(phases),
        run_id=run_id,
        dataset=dataset,
        cuda_visible_devices=cuda_visible_devices,
        cuda_visible_devices_by_phase=cuda_visible_devices_by_phase,
        lag_required=lag_required,
        two_window=two_window,
        note=note,
    )


def _train_script() -> Path:
    return _repo_root() / "scripts" / "train.py"


def _build_phase_argv(cfg: RunConfig, *, phase: str, run_id: str) -> List[str]:
    argv = [str(_train_script()), "--model", cfg.model, "--resume", phase, "--run_id", run_id]

    if cfg.dataset:
        argv.extend(["--dataset", cfg.dataset])

    if cfg.model in {"lstm", "tft"} and cfg.lag_required is not None:
        argv.append("--lag-required" if cfg.lag_required else "--no-lag-required")

    if cfg.model == "tft" and cfg.two_window:
        argv.append("--two-window")

    if cfg.note:
        argv.extend(["--note", cfg.note])

    return argv


def _write_run_metadata(
    cfg: RunConfig,
    *,
    run_id: str,
    config_path: Path,
    env: Dict[str, str],
    cuda_by_phase_resolved: Dict[str, Optional[str]],
) -> None:
    from src.utils.utils import get_run_root

    run_root = Path(get_run_root(run_id))
    run_root.mkdir(parents=True, exist_ok=True)

    meta_dir = run_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

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
        "lag_required": cfg.lag_required,
        "two_window": cfg.two_window,
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


def _effective_phases(cfg: RunConfig) -> Tuple[str, ...]:
    """Return the phases that will actually execute."""
    return tuple(cfg.phases)


def _validate_model_constraints(cfg: RunConfig) -> None:
    if cfg.run_id is not None and not cfg.run_id.startswith(f"{cfg.model}_"):
        raise ValueError(
            f"run_id '{cfg.run_id}' does not match model '{cfg.model}'. Expected prefix '{cfg.model}_'."
        )

    if cfg.model == "xgb" and cfg.lag_required is False:
        raise ValueError("xgb always requires lag features; set lag_required: true or omit it")

    # When using resume (single phase) without run_id, it cannot work.
    if len(cfg.phases) == 1 and cfg.phases[0] in _ALLOWED_PHASES and cfg.run_id is None and cfg.phases[0] != "search":
        # search could create a new run_id; other phases require an existing run.
        raise ValueError("run_id is required when running a single resume phase other than 'search'")


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
    has_resume_field = obj.get("resume") is not None
    cfg = _parse_config(obj, config_path=config_path)
    _validate_model_constraints(cfg)

    run_id = _allocate_run_id(cfg)

    phases = _effective_phases(cfg)

    if has_resume_field:
        _assert_run_dir_exists(run_id)

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

    cuda_resolved: Dict[str, Optional[str]] = {}
    for phase in phases:
        has_override, value = _resolve_cuda_for_phase(cfg, phase)
        if has_override:
            cuda_resolved[phase] = value

    # Persist resolved config/env snapshot under the run directory
    # (We write this before phases so a crashed run still records intent.)
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
