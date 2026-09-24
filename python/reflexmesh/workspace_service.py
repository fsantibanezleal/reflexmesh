"""One local workspace worker, preview preconditions and bounded run history."""

from __future__ import annotations

import hashlib
import json
import re
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from .effectors import WorkspaceFiles, register_process
from .runtime import EffectReceipt, Runtime
from .server import Run, RunManager
from .workflows import execute_workflow, validate_recipe


class WorkspaceManager(RunManager):
    def __init__(self, workspace: str | Path, templates=()):
        super().__init__(Path(workspace), capacity=1, retention=128)
        self.files = WorkspaceFiles(workspace)
        self.templates = tuple(templates)
        self.previews: dict[str, dict] = {}

    @contextmanager
    def _recovery_runtime(self):
        # Recovery must never race this service's workflow worker. Runtime also
        # owns the journal across processes for its whole lifetime.
        if not self._slots.acquire(blocking=False):
            raise RuntimeError("workspace worker busy")
        try:
            capabilities = tuple(self.config()["tools"])
            with Runtime(self.files.root, capabilities) as runtime:
                self.files.register(runtime)
                for template in self.templates:
                    register_process(runtime, template)
                yield runtime
        finally:
            self._slots.release()

    @staticmethod
    def _record_hash(record):
        return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()

    def _recoverable(self, action, evidence):
        if action not in self.config()["tools"]:
            return False
        if not action.startswith("process."):
            return True
        template = next(p for p in self.templates if "process." + p.name == action)
        digest = hashlib.sha256(json.dumps(template.argv).encode()).hexdigest()
        return (
            evidence.get("direct_child_terminated") is True
            and evidence.get("argv_sha256") == digest
        )

    def recovery(self):
        with self._recovery_runtime() as runtime:
            snapshot = runtime.snapshot()
            unknown = []
            for key, record in snapshot["state"]["intents"].items():
                if record["status"] != "effect_unknown":
                    continue
                action = record["intent"]["action_id"]
                evidence = (record.get("outcome") or {}).get("evidence", {})
                recoverable = self._recoverable(action, evidence)
                unknown.append(
                    {
                        "intent_id": key,
                        "action_id": action,
                        "resources": record["intent"]["resources"],
                        "status": record["status"],
                        "prior_evidence": evidence,
                        "record_sha256": self._record_hash(record),
                        "recoverable_in_workbench": recoverable,
                    }
                )
            return {
                "schema_version": 1,
                "unknown": unknown,
                "journal_sequence": snapshot["journal_sequence"],
            }

    def reconcile(self, request):
        required = {
            "intent_id",
            "record_sha256",
            "resolution",
            "checks",
            "reviewed_effects",
            "processes_quiescent",
            "note",
        }
        if (
            set(request) != required
            or not isinstance(request["intent_id"], str)
            or not isinstance(request["record_sha256"], str)
            or request["resolution"] not in {"completed_verified", "failed_verified"}
            or request["reviewed_effects"] is not True
            or request["processes_quiescent"] is not True
            or not isinstance(request["note"], str)
            or not 10 <= len(request["note"].strip()) <= 2000
            or not isinstance(request["checks"], list)
            or not 1 <= len(request["checks"]) <= 32
        ):
            raise ValueError("recovery requires explicit operator review and file evidence")
        checks = {}
        for check in request["checks"]:
            if not isinstance(check, dict) or "path" not in check:
                raise ValueError("invalid recovery check")
            path = self.files.canonical(check["path"])
            if path in checks:
                raise ValueError("duplicate recovery path")
            if set(check) == {"path", "absent"} and check["absent"] is True:
                expected = "absent"
            elif (
                set(check) == {"path", "sha256"}
                and isinstance(check["sha256"], str)
                and re.fullmatch(r"[0-9a-f]{64}", check["sha256"])
            ):
                expected = "sha256:" + check["sha256"]
            else:
                raise ValueError("check requires a SHA256 digest or explicit absence")
            checks[path] = expected
        with self._recovery_runtime() as runtime:
            record = runtime.snapshot()["state"]["intents"].get(request["intent_id"])
            if (
                not record
                or record["status"] != "effect_unknown"
                or self._record_hash(record) != request["record_sha256"]
            ):
                raise ValueError("unknown intent changed; inspect it again")
            action = record["intent"]["action_id"]
            prior = (record.get("outcome") or {}).get("evidence", {})
            if action not in self.config()["tools"]:
                raise ValueError("the original tool is no longer registered")
            if not self._recoverable(action, prior):
                raise ValueError("process crash requires external verification through Runtime API")
            changed = tuple(
                r["resource_id"] for r in record["intent"]["resources"] if r["access"] == "write"
            )
            if any(r.startswith("file:") and r[5:] not in checks for r in changed):
                raise ValueError("all leased files require a recovery check")
            actual = {p: self.files.fingerprint("file:" + p) for p in checks}
            if actual != checks:
                raise ValueError("recovery file evidence does not match actual workspace")
            evidence = {
                "verification": "file fingerprints and explicit operator attestation",
                "verified_checks": actual,
                "operator_review": request["note"].strip(),
                "reviewed_effects": True,
                "processes_quiescent": True,
                "scope": "listed files; operator reviewed remaining effects and process tree",
                "prior_record_sha256": request["record_sha256"],
            }
            result = runtime.reconcile(
                request["intent_id"], EffectReceipt(request["resolution"], evidence, changed)
            )
            with self._lock:
                self.previews.clear()
            return {
                "result": result,
                "verified_checks": actual,
                "journal_sequence": runtime.snapshot()["journal_sequence"],
            }

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
