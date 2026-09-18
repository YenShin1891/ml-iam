#!/usr/bin/env python
"""Download a published trained run and unpack it under RESULTS_PATH.

    python scripts/fetch_models.py --list
    python scripts/fetch_models.py --run_id tft_95
    python scripts/fetch_models.py --run_id tft_95 --archive ~/Downloads/tft_95.tar.gz

The published runs, their archive names and SHA-256 sums are listed in
metadata/published_models.json.  Every download (or local ``--archive``) is
checked against that sum before anything is unpacked, and every unpacked
file against the bundle's own manifest afterwards.  An existing run
directory is left alone unless ``--force`` is given.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO))
REGISTRY = REPO / "metadata" / "published_models.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_registry(path: Path = REGISTRY) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def archive_url(registry: dict, archive_name: str) -> str:
    template = registry.get("url_template") or ""
    record = str(registry.get("zenodo_record") or "")
    if not template or ("{record}" in template and not record):
        raise RuntimeError(
            "metadata/published_models.json names no download location yet. Download the archive "
            "from the DOI in the README and pass it with --archive."
        )
    return template.format(record=record, name=archive_name)


def _download(url: str, target: Path) -> None:
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, open(target, "wb") as out:  # noqa: S310 (https URL from the registry)
        shutil.copyfileobj(response, out, length=1 << 20)


def _safe_members(tar: tarfile.TarFile, run_id: str):
    for member in tar.getmembers():
        parts = Path(member.name).parts
        if member.name.startswith("/") or ".." in parts or not parts or parts[0] != run_id:
            raise RuntimeError(f"Refusing archive member outside {run_id}/: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise RuntimeError(f"Refusing archive member that is not a plain file: {member.name}")
        yield member


def install(run_id: str, archive: Path, expected_sha256: str, results_path: Path, force: bool = False) -> Path:
    actual = _sha256(archive)
    if expected_sha256 and actual != expected_sha256:
        raise RuntimeError(f"{archive.name}: SHA-256 {actual} does not match the published {expected_sha256}")

    kind = run_id.split("_", 1)[0]
    target = results_path / kind / run_id
    if target.exists() and not force:
        raise FileExistsError(f"{target} already exists; pass --force to replace the published files in it.")

    with tempfile.TemporaryDirectory(dir=str(results_path) if results_path.exists() else None) as tmp:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(tmp, members=list(_safe_members(tar, run_id)))
        unpacked = Path(tmp) / run_id
        manifest = json.loads((unpacked / "manifest.json").read_text())
        for name, digest in manifest.items():
            if _sha256(unpacked / name) != digest:
                raise RuntimeError(f"{run_id}/{name} does not match the bundle manifest")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            # Replace the published files only; anything the user produced (cache/, plots/) stays.
            shutil.copytree(unpacked, target, dirs_exist_ok=True)
        else:
            shutil.move(str(unpacked), str(target))
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_id", nargs="+", help="published run(s) to fetch; 'all' for every one")
    parser.add_argument("--archive", help="use this local .tar.gz instead of downloading (one run_id only)")
    parser.add_argument("--results-path", help="defaults to RESULTS_PATH in configs/paths.py")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true", help="list the published runs and exit")
    args = parser.parse_args(argv)

    registry = load_registry()
    models = registry.get("models", {})
    if args.list or not args.run_id:
        for run_id, entry in models.items():
            size = f"{entry.get('bytes', 0) / 1e6:.0f} MB" if entry.get("bytes") else "?"
            print(f"{run_id:10s} {size:>8s}  {entry.get('description', '')}")
        if registry.get("doi"):
            print(f"DOI: {registry['doi']}")
        return 0

    run_ids = list(models) if args.run_id == ["all"] else args.run_id
    if args.archive and len(run_ids) != 1:
        parser.error("--archive goes with exactly one --run_id")
    if args.results_path:
        results_path = Path(args.results_path)
    else:
        from configs.paths import RESULTS_PATH
        results_path = Path(RESULTS_PATH)

    for run_id in run_ids:
        entry = models.get(run_id)
        if entry is None:
            print(f"ERROR: {run_id} is not a published run. Published: {', '.join(models)}", file=sys.stderr)
            return 2
        try:
            if args.archive:
                target = install(run_id, Path(args.archive).expanduser(), entry.get("sha256", ""), results_path, args.force)
            else:
                with tempfile.TemporaryDirectory() as tmp:
                    local = Path(tmp) / entry["archive"]
                    _download(archive_url(registry, entry["archive"]), local)
                    target = install(run_id, local, entry.get("sha256", ""), results_path, args.force)
        except (RuntimeError, FileExistsError, FileNotFoundError) as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2
        print(f"{run_id}: installed at {target}")
        print(f"  next: python scripts/predict.py --run_id {run_id} --describe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
