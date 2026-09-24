"""Real crash/restart fixture used by C20, with no automatic effect replay.

An owned worker enters the native effect boundary, appends one record, flushes
and fsyncs it, then exits without returning a receipt. Reopening the same SQLite
journal converts the running intent to ``effect_unknown``. Reconciliation reads
the actual durable record; neither reopen nor reconciliation calls the writer.

This tests a process crash, not sudden power loss or filesystem fault tolerance.
The application and registered verifier are trusted, as elsewhere in Runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Self

from ..processes import spawn, worker_python
from ..runtime import EffectReceipt, Runtime, ToolSpec, wire

_ACTION = "crash.append"
_RESOURCE = "durable-effect"
_EXIT_CODE = 23
_MAX_EVIDENCE_BYTES = 65_536
_BOOTSTRAP = """
import sys
sys.path.insert(0, sys.argv[1])
from reflexmesh.environments.crash import _worker
_worker(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])
"""


def _parameters(value: int, intent_id: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not -(2**63) <= value < 2**63:
        raise ValueError("crash fixture value must be a signed 64-bit integer")
    if not isinstance(intent_id, str) or not re.fullmatch(r"[A-Za-z0-9:_.-]{1,160}", intent_id):
        raise ValueError("crash fixture requires a bounded explicit intent identity")


def _record(value: int, intent_id: str) -> dict[str, Any]:
    return {"operation": "append", "key": intent_id, "value": value}


def _canonical(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(root: Path) -> str:
    path = root / "effects.jsonl"
    if not path.exists():
        return "absent"
    if path.stat().st_size > _MAX_EVIDENCE_BYTES:
        raise RuntimeError("crash effect evidence exceeds its bound")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _register(runtime: Runtime, value: int, intent_id: str, execute) -> None:
    expected = _record(value, intent_id)

    def validate(arguments: dict[str, Any]) -> dict[str, Any]:
        if _canonical(arguments) != _canonical(expected):
            raise ValueError("crash fixture only admits its immutable effect record")
        return dict(expected)

    runtime.register(
        ToolSpec(
            _ACTION,
            (_ACTION,),
            validate,
            lambda _: {_RESOURCE: "write"},
            execute,
            lambda _: _fingerprint(runtime.workspace),
            allow_write=True,
        )
    )


def _worker(directory: str, value: int, intent_id: str, crash_point: str) -> None:
    """Private worker entry point. Abrupt exit is intentional and never in parent."""
    _parameters(value, intent_id)
    if crash_point not in {"before_effect", "after_effect"}:
        raise ValueError("unsupported crash point")
    root = Path(directory).resolve(strict=True)

    def effect(arguments: dict[str, Any], cancelled: threading.Event) -> EffectReceipt:
        if cancelled.is_set():
            return EffectReceipt("cancelled_verified", {"effect_dispatched": False})
        if crash_point == "before_effect":
            os._exit(_EXIT_CODE)
        # Deliberately non-idempotent: a blind second dispatch would append a
        # second line and fail the exactly-one-record verifier.
        with (root / "effects.jsonl").open("ab") as stream:
            stream.write((_canonical(arguments) + "\n").encode("utf8"))
            stream.flush()
            os.fsync(stream.fileno())
        os._exit(_EXIT_CODE)

    with Runtime(root, (_ACTION,)) as runtime:
        _register(runtime, value, intent_id, effect)
        runtime.execute_candidate(
            runtime.candidate(_ACTION, _record(value, intent_id)),
            intent_id=intent_id,
            idempotency_key=intent_id,
        )
    raise RuntimeError("crash fixture worker unexpectedly returned from its effect")


class CrashRecovery:
    """Owned process-crash evidence and an explicitly reopened native runtime.

    ``create`` requires a fresh directory. ``open`` only reopens an existing
    fixture; it never starts a worker. Call ``close`` before removing its files.
    """

    def __init__(self, directory: str | Path, *, value: int, intent_id: str):
        _parameters(value, intent_id)
        self.root = Path(directory).resolve(strict=True)
        self.effect_path = self.root / "effects.jsonl"
        self.value = value
        self.intent_id = intent_id
        self._expected = (_canonical(_record(value, intent_id)) + "\n").encode("utf8")
        metadata_path = self.root / "worker.json"
        if not metadata_path.is_file() or metadata_path.stat().st_size > 4096:
            raise RuntimeError("crash fixture lacks bounded worker termination evidence")
        self.worker = json.loads(metadata_path.read_text(encoding="utf8"))
        if (
            self.worker.get("schema_version") != 1
            or self.worker.get("exit_code") != _EXIT_CODE
            or self.worker.get("intent_id") != intent_id
            or self.worker.get("value") != value
            or not (self.root / ".reflexmesh" / "journal.sqlite3").is_file()
        ):
            raise RuntimeError("crash fixture identity or termination evidence mismatch")
        self.runtime = Runtime(self.root, (_ACTION,))
        try:
            record = self.runtime.snapshot()["state"]["intents"].get(intent_id)
            if not record or record["intent"]["arguments"] != _record(value, intent_id):
                raise RuntimeError("recovered journal does not contain the expected effect intent")
            if record["status"] not in {"effect_unknown", "completed_verified"}:
                raise RuntimeError("recovered crash intent has an unexpected state")

            def prohibit_reexecution(*_: Any) -> EffectReceipt:
                raise AssertionError("recovered crash effect must never be dispatched again")

            _register(self.runtime, value, intent_id, prohibit_reexecution)
        except BaseException:
            self.runtime.close()
            raise

    @classmethod
    def create(
        cls,
        directory: str | Path,
        *,
        value: int,
        intent_id: str = "crashed-write",
        crash_point: str = "after_effect",
        timeout: float = 15.0,
    ) -> Self:
        """Execute exactly one real worker attempt and reopen its durable journal."""
        _parameters(value, intent_id)
        if crash_point not in {"before_effect", "after_effect"} or not 0 < timeout <= 60:
            raise ValueError("invalid crash point or bounded worker timeout")
        root = Path(directory).resolve()
        root.mkdir(parents=True, exist_ok=False)
        # Use the package that this process actually imported. In a wheel test
        # this must not shadow its extension with an unbuilt repository checkout.
        import reflexmesh

        package_parent = str(Path(reflexmesh.__file__).resolve().parent.parent)
        with (root / "worker-stderr.txt").open("wb") as stderr:
            child = spawn(
                [
                    worker_python(),
                    "-I",
                    "-c",
                    _BOOTSTRAP,
                    package_parent,
                    str(root),
                    str(value),
                    intent_id,
                    crash_point,
                ],
                cwd=root,
                stderr=stderr,
            )
            try:
                exit_code = child.wait(timeout=timeout)
                metadata = {
                    "schema_version": 1,
                    "pid": child.pid,
                    "exit_code": exit_code,
                    "containment": child.containment,
                    "crash_point": crash_point,
                    "intent_id": intent_id,
                    "value": value,
                }
            finally:
                child.close()
        if exit_code != _EXIT_CODE:
            error = (root / "worker-stderr.txt").read_text(encoding="utf8", errors="replace")
            raise RuntimeError(f"crash worker exited {exit_code}: {error[:2000]}")
        with (root / "worker.json").open("w", encoding="utf8") as stream:
            stream.write(wire(metadata))
            stream.flush()
            os.fsync(stream.fileno())
        return cls(root, value=value, intent_id=intent_id)

    @classmethod
    def open(cls, directory: str | Path, *, value: int, intent_id: str = "crashed-write") -> Self:
        """Reopen only; no launch, replayed execution, or implicit reconciliation."""
        return cls(directory, value=value, intent_id=intent_id)

    @property
    def status(self) -> str:
        return self.runtime.snapshot()["state"]["intents"][self.intent_id]["status"]

    def evidence(self) -> dict[str, Any]:
        """Read bounded durable content, independent of the policy's action claim."""
        content = b""
        exists = self.effect_path.is_file()
        bounded = not exists or self.effect_path.stat().st_size <= _MAX_EVIDENCE_BYTES
        if exists and bounded:
            content = self.effect_path.read_bytes()
        try:
            records = [json.loads(line) for line in content.splitlines()] if bounded else None
        except (ValueError, UnicodeDecodeError):
            records = None
        return {
            "intent_id": self.intent_id,
            "worker_exit_code": self.worker["exit_code"],
            "crash_point": self.worker["crash_point"],
            "journal_status": self.status,
            "effect_exists": exists,
            "effect_bytes_bounded": bounded,
            "effect_count": len(records) if records is not None else None,
            "effect_sha256": hashlib.sha256(content).hexdigest() if bounded else None,
            "matches_expected_once": exists and bounded and content == self._expected,
            "effect_dispatched_during_recovery": False,
        }

    def reconcile(self) -> dict[str, Any]:
        """Complete only after independently proving exactly one expected effect."""
        evidence = self.evidence()
        if not evidence["matches_expected_once"]:
            raise RuntimeError("durable effect is missing, changed, malformed, or duplicated")
        if self.status == "completed_verified":
            return self.runtime.snapshot()["state"]["intents"][self.intent_id]
        return self.runtime.reconcile(
            self.intent_id,
            EffectReceipt("completed_verified", evidence, (_RESOURCE,)),
        )

    def verify(self) -> bool:
        """Truth requires both one actual effect and verified native reconciliation."""
        try:
            return self.status == "completed_verified" and self.evidence()["matches_expected_once"]
        except (OSError, ValueError, KeyError):
            return False

    def close(self) -> None:
        self.runtime.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
