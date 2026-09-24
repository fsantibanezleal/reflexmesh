"""Acquire exact upstream sources outside the distributable package.

Archives retain their upstream license. This module does not relicense or vendor
their code, and acquisition does not establish benchmark completion.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class Source:
    name: str
    repository: str
    revision: str
    reference: str
    license_note: str


SOURCES = {
    "toolsandbox": Source(
        "toolsandbox",
        "apple-aiml-research/ToolSandbox",
        "c8571d7854316d2e1c5f288e59fe1e34e53f6dd1",
        "main as of 2026-09-20",
        "Apple ToolSandbox License; inspect upstream LICENSE before redistribution",
    ),
    "tau2": Source(
        "tau2",
        "sierra-research/tau2-bench",
        "5ba9e3e56db57c5e4114bf7f901291f09b2c5619",
        "v0.1.3",
        "MIT; retain upstream LICENSE; data and third-party terms remain applicable",
    ),
}


def acquire(name: str, destination: str | Path) -> Path:
    """Fetch an immutable official GitHub archive and record its content digest."""
    source = SOURCES[name]
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{name}-{source.revision}"
    marker = target / "reflexmesh-source.json"
    if marker.exists():
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if recorded.get("revision") != source.revision:
            raise ValueError("source manifest revision mismatch")
        return target
    if target.exists():
        raise FileExistsError(f"incomplete source acquisition: {target}")
    url = f"https://codeload.github.com/{source.repository}/zip/{source.revision}"
    archive = root / f"{name}-{source.revision}.zip"
    request = urllib.request.Request(url, headers={"User-Agent": "reflexmesh-transfer/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, archive.open("xb") as output:
        downloaded = 0
        while chunk := response.read(1024 * 1024):
            downloaded += len(chunk)
            if downloaded > 268_435_456:
                raise ValueError("upstream source download exceeds archive byte bound")
            output.write(chunk)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as zipped:
        entries = zipped.infolist()
        total = sum(entry.file_size for entry in entries)
        if total > 1024 * 1024 * 1024 or len(entries) > 100_000:
            raise ValueError("upstream source archive exceeds extraction bounds")
        for entry in entries:
            parts = PurePosixPath(entry.filename).parts
            if not parts or any(part in {"..", "."} for part in parts) or "\\" in entry.filename:
                raise ValueError("unsafe upstream archive member")
            relative = Path(*parts[1:])
            path = (target / relative).resolve()
            if not path.is_relative_to(target):
                raise ValueError("archive escaped source destination")
            if entry.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(entry) as source_file, path.open("xb") as output:
                    while chunk := source_file.read(1024 * 1024):
                        output.write(chunk)
    marker.write_text(
        json.dumps({**asdict(source), "archive_url": url, "archive_sha256": digest}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return target


def validate_source(source: Path) -> dict:
    """Compare every original extracted file with the pinned archive's bytes.

    Build products may add files but may not replace original source/data. The
    archive digest is a recorded anchor, not a cryptographic publisher signature.
    """
    source = source.resolve()
    marker = json.loads((source / "reflexmesh-source.json").read_text(encoding="utf-8"))
    expected = SOURCES[marker["name"]]
    if marker["revision"] != expected.revision or marker["repository"] != expected.repository:
        raise ValueError("source identity differs from adapter pin")
    archive = source.parent / f"{expected.name}-{expected.revision}.zip"
    archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if archive_digest != marker["archive_sha256"]:
        raise ValueError("source archive digest changed")
    entries, total = [], 0
    with zipfile.ZipFile(archive) as zipped:
        for entry in zipped.infolist():
            if entry.is_dir():
                continue
            parts = PurePosixPath(entry.filename).parts
            relative = Path(*parts[1:])
            actual = (source / relative).resolve()
            if not actual.is_relative_to(source) or not actual.is_file():
                raise ValueError(f"extracted upstream file differs from archive: {relative}")
            with zipped.open(entry) as original:
                original_digest = hashlib.file_digest(original, "sha256").hexdigest()
            with actual.open("rb") as extracted:
                actual_digest = hashlib.file_digest(extracted, "sha256").hexdigest()
            if original_digest != actual_digest:
                raise ValueError(f"extracted upstream file differs from archive: {relative}")
            total += entry.file_size
            entries.append((relative.as_posix(), original_digest))
    return {
        "source": expected.name,
        "revision": expected.revision,
        "archive_sha256": archive_digest,
        "original_files_checked": len(entries),
        "original_bytes_checked": total,
        "tree_manifest_sha256": hashlib.sha256(json.dumps(sorted(entries)).encode()).hexdigest(),
        "all_original_files_match_archive": True,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=tuple(SOURCES))
    parser.add_argument("--destination", type=Path, default=Path(".external/sources"))
    args = parser.parse_args()
    print(acquire(args.name, args.destination))
