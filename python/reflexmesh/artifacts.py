"""Fetch versioned, hash-verified model releases from the project repository."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def install_models(archive: Path, manifest: dict, destination: Path) -> dict:
    """Verify the entire flat archive before replacing any destination checkpoint."""
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "reflexmesh-models":
        raise ValueError("unsupported model manifest")
    files = manifest.get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= 128:
        raise ValueError("invalid model file manifest")
    entries = {}
    for entry in files:
        name = entry.get("name")
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name)
            or name in entries
        ):
            raise ValueError("model names must be unique flat filenames")
        if (
            type(entry.get("size")) is not int
            or not 0 <= entry["size"] <= 134_217_728
            or not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", ""))
        ):
            raise ValueError("invalid model size or digest")
        entries[name] = entry
    if sum(e["size"] for e in entries.values()) > 536_870_912:
        raise ValueError("model bundle is too large")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("model archive digest mismatch")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".reflexmesh-models-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary)
        with zipfile.ZipFile(archive) as bundle:
            if len(bundle.infolist()) != len(entries) or {
                i.filename for i in bundle.infolist()
            } != set(entries):
                raise ValueError("archive and manifest file sets differ")
            for info in bundle.infolist():
                record = entries[info.filename]
                mode = info.external_attr >> 16
                if info.is_dir() or stat.S_ISLNK(mode) or info.file_size != record["size"]:
                    raise ValueError("archive type or declared size mismatch")
                raw = bundle.read(info)
                if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                    raise ValueError("checkpoint digest mismatch")
                (staged / info.filename).write_bytes(raw)
        for name in entries:
            target = destination / name
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise ValueError("model target must be a regular file")
        for name in entries:
            os.replace(staged / name, destination / name)
        (destination / "release-manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    return {
        "version": manifest["version"],
        "files": len(entries),
        "verified": True,
        "destination": str(destination),
    }


def fetch_models(destination: str | Path, version: str = "v0.01.000") -> dict:
    if not re.fullmatch(r"v\d+\.\d{2}\.\d{3}", version):
        raise ValueError("version must use vX.XX.XXX format")
    base = f"https://github.com/fsantibanezleal/reflexmesh/releases/download/{version}/"

    def download(name, limit):
        with urllib.request.urlopen(base + name, timeout=60) as response:
            if not response.geturl().startswith("https://"):
                raise ValueError("artifact redirect must remain HTTPS")
            raw = response.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("release artifact exceeds its size bound")
        return raw

    manifest = json.loads(download("models-manifest.json", 1_048_576))
    if manifest.get("version") != version:
        raise ValueError("model manifest version mismatch")
    raw = download("models.zip", 536_870_912)
    with tempfile.TemporaryDirectory(prefix="reflexmesh-download-") as temporary:
        archive = Path(temporary) / "models.zip"
        archive.write_bytes(raw)
        return install_models(archive, manifest, Path(destination))
