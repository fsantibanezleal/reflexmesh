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
        while chunk := response.read(1024 * 1024):
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=tuple(SOURCES))
    parser.add_argument("--destination", type=Path, default=Path(".external/sources"))
    args = parser.parse_args()
    print(acquire(args.name, args.destination))
