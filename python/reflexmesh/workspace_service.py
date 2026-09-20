"""One local workspace worker, preview preconditions and bounded run history."""

from __future__ import annotations

import json
import time
from pathlib import Path
from uuid import uuid4

from .effectors import WorkspaceFiles
from .server import Run, RunManager
from .workflows import execute_workflow, validate_recipe


class WorkspaceManager(RunManager):
    def __init__(self, workspace: str | Path, templates=()):
        super().__init__(Path(workspace), capacity=1, retention=128)
        self.files = WorkspaceFiles(workspace)
        self.templates = tuple(templates)
        self.previews: dict[str, dict] = {}

    def config(self):
        return {
            "enabled": True,
            "schema_version": 1,
            "workspace_label": self.files.root.name,
            "canonical_root": str(self.files.root),
            "tools": ["file.read", "file.write", *["process." + p.name for p in self.templates]],
        }

    def preview(self, recipe: dict):
        steps = validate_recipe(recipe, self.files, self.templates)
        paths = {self.files.canonical(s["verify"]["path"]) for s in steps}
        paths.update(
            self.files.canonical(s["arguments"]["path"]) for s in steps if "path" in s["arguments"]
        )
        preconditions = {path: self.files.fingerprint("file:" + path) for path in sorted(paths)}
        key = uuid4().hex
        with self._lock:
            self.previews = {
                k: v for k, v in self.previews.items() if v["expires"] > time.monotonic()
            }
            if len(self.previews) >= 128:
                raise RuntimeError("preview capacity exhausted")
            self.previews[key] = {
                "recipe": json.loads(json.dumps(recipe)),
                "preconditions": preconditions,
                "expires": time.monotonic() + 120,
            }
        return {
            "preview_id": key,
            "canonical_root": str(self.files.root),
            "steps": steps,
            "capabilities": sorted({s["tool"] for s in steps}),
            "preconditions": preconditions,
            "expires_in_seconds": 120,
        }

    def launch(self, preview_id: str):
        with self._lock:
            preview = self.previews.get(preview_id)
            if not preview or preview["expires"] <= time.monotonic():
                raise ValueError("preview is missing, consumed or expired")
            for path, fingerprint in preview["preconditions"].items():
                if self.files.fingerprint("file:" + path) != fingerprint:
                    del self.previews[preview_id]
                    raise ValueError("workspace changed since preview; preview the recipe again")
            run = self.start(preview)
            del self.previews[preview_id]
            return run

    def _execute(self, run: Run):
        try:
            with self._lock:
                run.status = "running"
            # Recheck in the worker before opening the journal or dispatching.
            for path, fingerprint in run.request["preconditions"].items():
                if self.files.fingerprint("file:" + path) != fingerprint:
                    raise ValueError("workspace changed before execution")

            def event(value):
                with self._lock:
                    if len(run.events) >= 4096:
                        run.cancellation.set()
                        return
                    run.events.append({**value, "sequence": len(run.events) + 1})

            result = execute_workflow(
                run.request["recipe"],
                self.files.root,
                templates=self.templates,
                cancel_event=run.cancellation,
                on_event=event,
            )
            with self._lock:
                run.result = result
                run.status = (
                    "cancelled"
                    if result["cancelled"]
                    else "succeeded"
                    if result["verified_success"]
                    else "failed"
                )
        except Exception as exc:  # noqa: BLE001 - typed error only across authenticated HTTP boundary
            with self._lock:
                run.error = type(exc).__name__
                run.status = "cancelled" if run.cancellation.is_set() else "failed"
        finally:
            self._slots.release()

    def snapshot(self, run_id, after=0):
        result = super().snapshot(run_id, after)
        result["workflow"] = result.pop("episode")
        return result
