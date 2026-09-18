#!/usr/bin/env python
"""Pack a trained run for publication: weights and summary statistics, no data.

    python scripts/export_run_bundle.py --run_id tft_95 --out release/

Writes ``<out>/<run_id>.tar.gz`` and adds its SHA-256 to ``<out>/checksums.json``.
Unpacked under RESULTS_PATH/<model>/ (``make fetch-models`` does this), the
run serves ``scripts/predict.py`` as it stands, and the dashboard and
``--resume test`` once the user has rebuilt the processed data themselves.

What goes in is a whitelist:

* the final weights -- Lightning checkpoints reduced to the state dict and
  hyperparameters (no optimizer state, no callback state);
* artifacts/: feature and target lists, category vocabularies, scalers,
  imputation medians, best hyperparameters, and the split assignment
  (series identifiers and a train/val/test label -- no values);
* meta/run_config.resolved.json, reduced to the settings a resumed phase reads;
* metrics/*.csv and a generated MODEL_CARD.md.

What stays out: cache/ (the processed AR6 data), predictions (they embed
test-set values), plots, logs.  Users obtain AR6 from IIASA themselves.

The bundle is then scanned for absolute local paths and the export fails if
any is found.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARTIFACTS = (
    "best_params.json",
    "categories.json",
    "features.json",
    "splits.parquet",
    "imputation_medians.json",
    "train_meta.json",
    "x_scaler.pkl",
    "y_scaler.pkl",
    "lstm_scaler_X.pkl",
    "lstm_scaler_y.pkl",
)
CHECKPOINT_KEYS_DROPPED = ("callbacks", "optimizer_states", "lr_schedulers", "loops")
RESOLVED_KEYS = ("run_id", "model", "dataset", "keep_partial_targets", "target_normalizer_mode", "two_window")
LOCAL_PATH = re.compile(rb"(?<![A-Za-z0-9_./-])/(?:mnt|root|home|Users)/[A-Za-z0-9_./-]{3,}")
LIBRARIES = ("python", "numpy", "pandas", "scikit-learn", "xgboost", "torch", "lightning",
             "pytorch-lightning", "pytorch-forecasting")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _slim_checkpoint(source: Path, target: Path) -> None:
    import torch

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    for key in CHECKPOINT_KEYS_DROPPED:
        checkpoint.pop(key, None)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target)


def _library_versions() -> dict:
    from importlib import metadata

    versions = {"python": ".".join(map(str, sys.version_info[:3]))}
    for name in LIBRARIES[1:]:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _copy_weights(run_root: Path, bundle: Path, kind: str) -> list:
    copied = []
    if kind == "xgb":
        files = sorted((run_root / "checkpoints").glob("final_best*.json"))
        if not files:
            raise FileNotFoundError(f"No final XGBoost model under {run_root / 'checkpoints'}")
        for file in files:
            (bundle / "checkpoints").mkdir(exist_ok=True)
            shutil.copy2(file, bundle / "checkpoints" / file.name)
            copied.append(f"checkpoints/{file.name}")
        return copied
    checkpoint = run_root / "final" / "best.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No final checkpoint at {checkpoint}")
    _slim_checkpoint(checkpoint, bundle / "final" / "best.ckpt")
    copied.append("final/best.ckpt")
    for name in ("dataset_template.pt", "training_summary.json"):
        if (run_root / "final" / name).exists():
            shutil.copy2(run_root / "final" / name, bundle / "final" / name)
            copied.append(f"final/{name}")
    return copied


def _overall_metrics(run_root: Path) -> dict:
    import pandas as pd

    path = run_root / "metrics" / "performance.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    if "Region Type" in table.columns:
        table = table[table["Region Type"] == "Overall"]
    if table.empty:
        return {}
    row = table.iloc[-1]
    wanted = ("R2 Score (per-target avg)", "R2 Score (pooled)", "RMSE", "MAE", "Sample Size")
    return {k: (float(row[k]) if k != "Sample Size" else int(row[k])) for k in wanted if k in row.index}


def _model_card(run_id: str, kind: str, resolved: dict, requirements: dict, params: dict,
                metrics: dict, versions: dict, commit: str, files: list) -> str:
    names = {"xgb": "XGBoost (one booster per target, autoregressive rollout)",
             "lstm": "LSTM (sequence-to-one, learned Region and Model_Family embeddings)",
             "tft": "Temporal Fusion Transformer (two-window forecast)"}
    lines = [
        f"# ML-IAM trained emulator: {run_id}",
        "",
        f"- **Model**: {names.get(kind, kind)}",
        f"- **Code**: https://github.com/YenShin1891/ml-iam (commit `{commit}` exported this bundle)",
        f"- **Training data**: IPCC AR6 Scenarios Database v1.1 (IIASA), processed dataset `{resolved.get('dataset')}`",
        f"- **Exported**: {datetime.date.today().isoformat()}",
        "",
        "This bundle holds trained weights and summary statistics (scalers, category",
        "vocabularies, imputation medians, the train/val/test assignment of series",
        "identifiers). It contains **no AR6 values**; obtain AR6 from",
        "https://data.ece.iiasa.ac.at/ar6/ under its own terms.",
        "",
        "## Use",
        "",
        "```bash",
        "git clone https://github.com/YenShin1891/ml-iam && cd ml-iam",
        "pip install -r requirements.txt            # plus requirements-advanced.txt for LSTM/TFT",
        "cp configs/paths-template.py configs/paths.py",
        f"make fetch-models MODELS={run_id}           # or unpack this archive under RESULTS_PATH/{kind}/",
        f"python scripts/predict.py --run_id {run_id} --describe",
        f"python scripts/predict.py --run_id {run_id} --input my_scenarios.csv --output emulated.csv",
        "```",
        "",
        "The input format, the limits of the emulator and how to reproduce the paper's",
        "test metrics are in `docs/USING_THE_EMULATOR.md` of the repository.",
        "",
        "## Inputs and outputs",
        "",
        f"- {len(requirements['input_variables'])} input Variables (IAMC names; `--describe` lists them). "
        "Absent ones are treated as not reported.",
        f"- {len(requirements['regions'])} regions and {len(requirements['model_families'])} model families; "
        "both are closed vocabularies.",
        "- Target history: " + (
            f"**required** -- values of the targets at the first timesteps (the model reads {requirements['n_lags']} lags)."
            if requirements["needs_target_history"] else "not needed; the model reads the inputs only."),
        "- Outputs:",
    ]
    lines += [f"  - `{t}` ({u})" for t, u in requirements["target_units"].items()]
    lines += ["", "## Hyperparameters", "", "```json", json.dumps(params, indent=2), "```", ""]
    if metrics:
        lines += ["## Test-set performance (autoregressive / full-horizon)", ""]
        lines += [f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in metrics.items()]
        lines += ["", "See `metrics/` for the per-scale and per-split tables.", ""]
    lines += ["## Library versions at export", ""]
    lines += [f"- {k} {v}" for k, v in versions.items()]
    lines += [
        "",
        "The scalers are scikit-learn pickles and the checkpoints PyTorch Lightning files;",
        "load them with the versions above (the repository's requirements files pin them).",
        "XGBoost boosters are JSON and load across versions.",
        "",
        "## Files",
        "",
    ]
    lines += [f"- `{f}`" for f in files]
    lines += ["", "## Citation", "", "See the repository README for the paper and the DOI of this record.", ""]
    return "\n".join(lines)


def _scan_for_local_paths(bundle: Path) -> list:
    hits = []
    for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
        found = sorted({m.group(0).decode("utf-8", "replace") for m in LOCAL_PATH.finditer(path.read_bytes())})
        if found:
            hits.append((str(path.relative_to(bundle)), found[:3]))
    return hits


def export(run_id: str, out_dir: Path) -> Path:
    from src.inference.new_inputs import describe_requirements, ensure_imputation_medians, model_kind
    from src.utils.run_store import RunStore

    store = RunStore(run_id)
    run_root = Path(store.root)
    kind = model_kind(run_id)
    if not run_root.is_dir():
        raise FileNotFoundError(f"Run not found: {run_root}")
    ensure_imputation_medians(run_id)

    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / run_id
        (bundle / "artifacts").mkdir(parents=True)
        (bundle / "meta").mkdir()
        files = _copy_weights(run_root, bundle, kind)

        for name in ARTIFACTS:
            source = run_root / "artifacts" / name
            if source.exists():
                shutil.copy2(source, bundle / "artifacts" / name)
                files.append(f"artifacts/{name}")

        resolved_path = run_root / "meta" / "run_config.resolved.json"
        resolved = json.loads(resolved_path.read_text()) if resolved_path.exists() else {}
        resolved = {k: resolved.get(k) for k in RESOLVED_KEYS if k in resolved}
        resolved["run_id"] = run_id  # a continued run records the id it started from
        (bundle / "meta" / "run_config.resolved.json").write_text(json.dumps(resolved, indent=2) + "\n")
        files.append("meta/run_config.resolved.json")

        for source in sorted((run_root / "metrics").glob("*.csv")):
            (bundle / "metrics").mkdir(exist_ok=True)
            shutil.copy2(source, bundle / "metrics" / source.name)
            files.append(f"metrics/{source.name}")

        params = json.loads((run_root / "artifacts" / "best_params.json").read_text()) \
            if (run_root / "artifacts" / "best_params.json").exists() else {}
        card = _model_card(run_id, kind, resolved, describe_requirements(run_id), params,
                           _overall_metrics(run_root), _library_versions(), _git_commit(), files)
        (bundle / "MODEL_CARD.md").write_text(card)

        hits = _scan_for_local_paths(bundle)
        if hits:
            raise RuntimeError("Local paths found in the bundle; nothing was written:\n" +
                               "\n".join(f"  {name}: {found}" for name, found in hits))

        manifest = {f: _sha256(bundle / f) for f in files + ["MODEL_CARD.md"]}
        (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        archive = out_dir / f"{run_id}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(bundle, arcname=run_id)
        shutil.copy2(bundle / "MODEL_CARD.md", out_dir / f"{run_id}.MODEL_CARD.md")

    checksums_path = out_dir / "checksums.json"
    checksums = json.loads(checksums_path.read_text()) if checksums_path.exists() else {}
    checksums[archive.name] = {"sha256": _sha256(archive), "bytes": archive.stat().st_size}
    checksums_path.write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n")
    return archive


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_id", required=True, nargs="+")
    parser.add_argument("--out", required=True, help="directory for the archives and checksums.json")
    args = parser.parse_args(argv)
    for run_id in args.run_id:
        archive = export(run_id, Path(args.out))
        print(f"{run_id}: {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
