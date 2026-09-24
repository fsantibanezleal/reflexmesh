"""Capture immutable inputs and exact executed installation at training boundaries."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace


def hashes(directory):
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(directory).glob("*"))
        if p.is_file()
    }


class TrainingCapture:
    def __init__(self, args):
        self.args = args
        self.path = args.artifacts / "training-runs" / (args.stage + ".json")

    def _lineage(self):
        from ..pipeline import lineage

        planner = SimpleNamespace(model_id=self.args.model) if self.args.stage == "gates" else None
        return lineage(self.args.artifacts / "checkpoints", [], planner)

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".pending")
        temporary.write_text(json.dumps(self.record, indent=2, allow_nan=False), encoding="utf8")
        temporary.replace(self.path)

    def __enter__(self):
        if self.path.exists():
            previous = self.path.read_bytes()
            archived = self.path.with_name(
                self.args.stage + ".prior-" + hashlib.sha256(previous).hexdigest()[:16] + ".json"
            )
            archived.write_bytes(previous)
        self.started = time.perf_counter()
        self.record = {
            "schema_version": 1,
            "stage": self.args.stage,
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "capture_timing": "before stage imports and fitting/collection execution",
            "arguments": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(self.args).items()
            },
            "data_sha256_before": hashes(self.args.artifacts / "data"),
            "installation_before": self._lineage(),
        }
        self._write()
        return self

    def __exit__(self, kind, error, traceback):
        after = self._lineage()
        self.record.update(
            finished_at=datetime.now(UTC).isoformat(),
            elapsed_seconds=time.perf_counter() - self.started,
            status="failed" if error else "complete",
            data_sha256_after=hashes(self.args.artifacts / "data"),
            installation_after=after,
            source_unchanged=all(
                self.record["installation_before"][key] == after[key]
                for key in ("source_sha256", "native_source_sha256", "native_binary", "packages")
            ),
        )
        if error:
            self.record["failure"] = {"type": kind.__name__, "message": str(error)}
        elif not self.record["source_unchanged"]:
            self.record["status"] = "invalidated"
            self.record["failure"] = {
                "type": "SourceChanged",
                "message": "Executed installation changed during the stage",
            }
        self._write()
        if not self.record["source_unchanged"] and error is None:
            raise RuntimeError(
                "training installation changed during execution; artifacts require audit"
            )
        return False
