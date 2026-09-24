"""Attach exact native source lineage to wheels built from a checkout or sdist."""

import hashlib
import json
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithNativeLineage(build_py):
    def run(self):
        super().run()
        root = Path(__file__).resolve().parent
        sources = sorted([*root.glob("Cargo.*"), *(root / "rust").rglob("*.rs")])
        hashes = {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
            if path.is_file()
        }
        if not hashes or "Cargo.lock" not in hashes:
            raise RuntimeError("native source lineage is incomplete")
        destination = Path(self.build_lib) / "reflexmesh" / "native-source.json"
        destination.write_text(
            json.dumps({"schema_version": 1, "source_sha256": hashes}, indent=2, sort_keys=True),
            encoding="utf-8",
        )


setup(cmdclass={"build_py": BuildWithNativeLineage})
