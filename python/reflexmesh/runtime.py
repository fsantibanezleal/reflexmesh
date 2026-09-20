"""Durable native admission plus scoped, registered Python effectors.

This boundary controls cooperating registered tools. It is not an operating
system sandbox: untrusted executable code requires an external VM/container.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self
from uuid import uuid4

from .contracts import Candidate, Decision, Observation, validate_decision


def wire(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class AdmissionError(ValueError):
    """A tool cannot be admitted under its concrete local authority."""


class _JournalOwnership:
    """Nonblocking OS ownership for one cooperating runtime per journal.

    The sidecar inode is retained on close. Deleting a live sidecar would allow
    another owner to lock a different inode. Locks are released by the OS after
    process termination. This is a local-filesystem coordination boundary, not
    protection against an actor able to replace journal/lock files.
    """

    def __init__(self, journal: Path):
        self.path = journal.with_name(journal.name + ".owner.lock")
        self._stream = None
        if journal.exists() and journal.stat().st_nlink != 1:
            raise AdmissionError("journal hardlink aliases cannot have independent ownership")
        if self.path.is_symlink():
            raise AdmissionError("journal ownership lock cannot be a symbolic link")
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NOINHERIT", 0)
        descriptor = os.open(self.path, flags, 0o600)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise AdmissionError("journal ownership lock must be an unaliased regular file")
            os.set_inheritable(descriptor, False)
            stream = os.fdopen(descriptor, "r+b", buffering=0)
            descriptor = -1
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                stream.close()
                raise AdmissionError(
                    "journal already owned or ownership lock unavailable"
                ) from error
            self._stream = stream
        finally:
            if descriptor != -1:
                os.close(descriptor)

    def close(self) -> None:
        if self._stream is not None:
            # Closing this noninherited descriptor releases the OS lock, even
            # when construction failed before SQLite finished opening.
            self._stream.close()
            self._stream = None


@dataclass(frozen=True)
class EffectReceipt:
    status: str
    evidence: dict[str, Any]
    changed_resources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.status
            not in {"completed_verified", "failed_verified", "effect_unknown", "cancelled_verified"}
            or not self.evidence
        ):
            raise ValueError("receipt requires a supported status and concrete evidence")


@dataclass(frozen=True)
class ToolSpec:
    """The application registers authority. Model-produced fields cannot expand it."""

    name: str
    capabilities: tuple[str, ...]
    validate: Callable[[dict[str, Any]], dict[str, Any]]
    resources: Callable[[dict[str, Any]], dict[str, str]]
    execute: Callable[[dict[str, Any], threading.Event], EffectReceipt]
    fingerprint: Callable[[str], str]
    allow_write: bool = False
    version: int = 1


class DurableJournal:
    """SQLite FULL commits persist the native hash-checked journal before effects."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ownership = _JournalOwnership(self.path)
        self.connection = None
        try:
            self.connection = sqlite3.connect(self.path, check_same_thread=False)
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS broker (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL, digest TEXT NOT NULL)"
            )
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS receipts (intent_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS archives (tip TEXT PRIMARY KEY, payload TEXT NOT NULL, digest TEXT NOT NULL)"
            )
            self.connection.commit()
        except BaseException:
            self.close()
            raise

    def load(self) -> str | None:
        row = self.connection.execute("SELECT payload,digest FROM broker WHERE id=1").fetchone()
        if not row:
            return None
        if hashlib.sha256(row[0].encode()).hexdigest() != row[1]:
            raise AdmissionError("durable journal checksum mismatch")
        return row[0]

    def save(self, payload: str, intent_id: str | None = None, receipt: dict | None = None) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO broker VALUES (1,?,?)",
                (payload, hashlib.sha256(payload.encode()).hexdigest()),
            )
            if intent_id and receipt:
                self.connection.execute(
                    "INSERT OR REPLACE INTO receipts VALUES (?,?)", (intent_id, wire(receipt))
                )

    def close(self) -> None:
        try:
            if self.connection is not None:
                self.connection.close()
                self.connection = None
        finally:
            self._ownership.close()

    def archive(self, tip: str, payload: str) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO archives VALUES (?,?,?)",
                (tip, payload, hashlib.sha256(payload.encode()).hexdigest()),
            )


class Runtime:
    """Resource-versioned broker; no fallback to an unguarded Python executor.

    Local registrations are authoritative. Call ``candidate`` to bind concrete
    arguments to observed resources, then ``execute`` to re-observe and admit.
    A persisted in-flight intent becomes unknown after restart; explicit external
    evidence is required for reconciliation. Replay never executes an effector.
    """

    def __init__(
        self,
        workspace: str | Path,
        capabilities: tuple[str, ...],
        *,
        journal_path: str | Path | None = None,
        workspace_id: str | None = None,
    ):
        from ._native import Broker

        self.workspace = Path(workspace).resolve(strict=True)
        if not self.workspace.is_dir():
            raise AdmissionError("workspace must be an existing directory")
        self._lock = threading.RLock()
        self._tools: dict[str, ToolSpec] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._closed = False
        self.workspace_id = workspace_id or hashlib.sha256(str(self.workspace).encode()).hexdigest()
        self.store = DurableJournal(
            journal_path or self.workspace / ".reflexmesh" / "journal.sqlite3"
        )
        try:
            journal = self.store.load()
            if journal:
                self.broker = Broker.from_journal(journal)
                snapshot = self.snapshot()
                if snapshot["workspace_id"] != self.workspace_id:
                    raise AdmissionError("journal belongs to a different workspace")
                # Restart cannot grant capabilities absent from durable authority.
                for cap in set(snapshot["state"]["capabilities"]) - set(capabilities):
                    self.broker.revoke(cap)
            else:
                self.broker = Broker(
                    wire(
                        {
                            "workspace_id": self.workspace_id,
                            "capabilities": sorted(set(capabilities)),
                            "trusted_sources": ["runtime", "verifier"],
                        }
                    )
                )
            self._sequence = self.snapshot()["state"]["source_sequences"].get("runtime", 0)
            self._persist()
        except BaseException:
            self.store.close()
            self._closed = True
            raise

    def _persist(self, intent_id: str | None = None, receipt: dict | None = None) -> None:
        payload = self.broker.journal()
        self.store.save(payload, intent_id, receipt)
        if len(json.loads(payload)["entries"]) >= 256:
            tip = self.snapshot()["journal_tip"]
            self.store.archive(tip, payload)
            self.broker.checkpoint(tip)
            self.store.save(self.broker.journal())

    def snapshot(self) -> dict[str, Any]:
        return json.loads(self.broker.snapshot())

    def register(self, spec: ToolSpec) -> None:
        with self._lock:
            if self._closed or spec.name in self._tools:
                raise AdmissionError("runtime closed or tool already registered")
            self.broker.register_action(
                wire(
                    {
                        "action_id": spec.name,
                        "version": spec.version,
                        "required_capabilities": spec.capabilities,
                        "allow_write": spec.allow_write,
                    }
                )
            )
            self._tools[spec.name] = spec
            self._persist()

    def _observe(self, spec: ToolSpec, resources: dict[str, str]) -> dict[str, int]:
        for resource_id in resources:
            self._sequence += 1
            self.broker.observe(
                wire(
                    {
                        "event_id": uuid4().hex,
                        "source_id": "runtime",
                        "source_sequence": self._sequence,
                        "kind": "resource_observed",
                        "resource_id": resource_id,
                        "fingerprint": spec.fingerprint(resource_id),
                    }
                )
            )
        state = self.snapshot()["state"]["resources"]
        return {key: state[key]["revision"] for key in resources}

    def candidate(
        self,
        tool: str,
        arguments: dict[str, Any],
        *,
        action_id: str | None = None,
        features: dict[str, float] | None = None,
    ) -> Candidate:
        with self._lock:
            spec = self._tools[tool]
            arguments = spec.validate(json.loads(wire(arguments)))
            resources = spec.resources(arguments)
            if any(access not in {"read", "write"} for access in resources.values()):
                raise AdmissionError("invalid resource access declaration")
            versions = self._observe(spec, resources)
            self._persist()
            return Candidate(
                action_id or uuid4().hex,
                tool,
                arguments,
                ("write",) if any(access == "write" for access in resources.values()) else (),
                spec.capabilities,
                versions,
                features or {},
            )

    def execute(
        self,
        observation: Observation,
        decision: Decision,
        *,
        intent_id: str | None = None,
        idempotency_key: str | None = None,
        ttl_ms: int = 30_000,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        validate_decision(observation, decision)
        if decision.mode != "act":
            raise AdmissionError("only explicit act decisions enter the effect boundary")
        candidate = observation.candidate(decision.candidate_id)
        return self.execute_candidate(
            candidate,
            intent_id=intent_id,
            idempotency_key=idempotency_key,
            ttl_ms=ttl_ms,
            cancel_event=cancel_event,
        )

    def execute_candidate(
        self,
        candidate: Candidate,
        *,
        intent_id: str | None = None,
        idempotency_key: str | None = None,
        ttl_ms: int = 30_000,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        intent_id = intent_id or uuid4().hex
        with self._lock:
            if self._closed or not candidate.allowed:
                raise AdmissionError("runtime closed or candidate excluded")
            spec = self._tools[candidate.tool]
            arguments = spec.validate(json.loads(wire(candidate.arguments)))
            resources = spec.resources(arguments)
            if set(resources) != set(candidate.resource_versions):
                raise AdmissionError("candidate resource binding is incomplete")
            self._observe(spec, resources)
            submitted = json.loads(
                self.broker.submit(
                    wire(
                        {
                            "intent_id": intent_id,
                            "action_id": spec.name,
                            "action_version": spec.version,
                            "resources": [
                                {
                                    "resource_id": key,
                                    "expected_revision": candidate.resource_versions[key],
                                    "access": access,
                                }
                                for key, access in resources.items()
                            ],
                            "idempotency_key": idempotency_key,
                            "ttl_ms": ttl_ms,
                            "arguments": arguments,
                        }
                    )
                )
            )
            self._persist()
            if submitted["duplicate"]:
                return submitted["record"]
            if cancel_event and cancel_event.is_set():
                result = json.loads(self.broker.cancel(intent_id))
                self._persist()
                return result
            started = json.loads(self.broker.begin(intent_id))
            self._persist()  # effect never starts before its durable running receipt
            cancelled = self._cancel[intent_id] = cancel_event or threading.Event()
        try:
            receipt = spec.execute(started["arguments"], cancelled)
        except Exception as exc:  # noqa: BLE001 - an arbitrary registered effector can fail after an effect
            # An arbitrary effector exception does not establish absence of effects.
            receipt = EffectReceipt(
                "effect_unknown",
                {"error_type": type(exc).__name__, "message": str(exc)[:2000], "verified": False},
            )
        with self._lock:
            fingerprint = {}
            try:
                fingerprint = {
                    resource: spec.fingerprint(resource) for resource in receipt.changed_resources
                }
            except Exception as exc:  # noqa: BLE001 - verifier failures preserve unknown outcome semantics
                receipt = EffectReceipt(
                    "effect_unknown", {"verification_error": type(exc).__name__}
                )
            outcome = {
                "status": receipt.status,
                "changed_resources": receipt.changed_resources,
                "resource_fingerprints": fingerprint,
                "evidence": receipt.evidence,
            }
            try:
                result = json.loads(self.broker.finish(intent_id, wire(outcome)))
                self._persist(intent_id, outcome)
            finally:
                self._cancel.pop(intent_id, None)
            return result

    def cancel(self, intent_id: str) -> dict[str, Any]:
        with self._lock:
            result = json.loads(self.broker.cancel(intent_id))
            self._persist()
            if intent_id in self._cancel:
                self._cancel[intent_id].set()
            return result

    def revoke(self, capability: str) -> None:
        with self._lock:
            result = json.loads(self.broker.revoke(capability))
            self._persist()
            for intent_id in result["affected"]:
                if intent_id in self._cancel:
                    self._cancel[intent_id].set()

    def reconcile(self, intent_id: str, receipt: EffectReceipt) -> dict[str, Any]:
        """Trusted caller supplies independently obtained external evidence."""
        with self._lock:
            record = self.snapshot()["state"]["intents"][intent_id]
            spec = self._tools[record["intent"]["action_id"]]
            outcome = {
                "status": receipt.status,
                "evidence": receipt.evidence,
                "changed_resources": receipt.changed_resources,
                "resource_fingerprints": {
                    r: spec.fingerprint(r) for r in receipt.changed_resources
                },
            }
            result = json.loads(self.broker.reconcile(intent_id, wire(outcome)))
            self._persist(intent_id, outcome)
            return result

    def replay(self) -> dict[str, Any]:
        from ._native import Broker

        return json.loads(Broker.replay(self.broker.journal()))

    def close(self) -> None:
        with self._lock:
            if self._cancel:
                raise RuntimeError("cancel and join running effects before closing the runtime")
            if not self._closed:
                try:
                    self._persist()
                finally:
                    self.store.close()
                    self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
