"""Controlled, observable interleavings around actual file/event transport.

Faults are designed experiments, not captured production events. Producer threads
are independent of controller actions; waiting only observes their artifacts.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class FaultDriver:
    payload_limit = 512

    def __init__(self, root: Path, variant: str, seed: int):
        self.root, self.variant, self.seed = root, variant, seed
        self.directory = root / ".scenario"
        self.directory.mkdir()
        self.thread = None
        self.started = False
        self.conflict_injected = False
        self.cancelled = threading.Event()
        self._write("revision.json", {"sequence": 1, "producer": "initial"})
        event = {"id": f"delivery-{seed}", "sequence": 1, "payload": "current work"}
        self._write("inbox.json", [event, event] if variant == "duplicate_event" else [event])
        self._write("dedup.json", {"accepted": [], "duplicates": 0})
        if variant == "boundary":
            (self.directory / "payload.txt").write_text(
                "x" * (self.payload_limit + 1), encoding="utf8"
            )

    def _write(self, name, value):
        path = self.directory / name
        temporary = path.with_suffix(path.suffix + ".pending")
        temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf8")
        temporary.replace(path)

    def first_observation_delivered(self):
        if self.started:
            return
        self.started = True
        if self.variant not in {"delayed_observation", "unavailable_dependency"}:
            return

        def produce():
            if self.cancelled.wait(0.015):
                return
            if self.variant == "delayed_observation":
                self._write(
                    "delivery.json",
                    {
                        "sequence": 2,
                        "payload": "new source event",
                        "producer": "independent thread",
                    },
                )
            else:
                self._write(
                    "dependency.ready", {"available": True, "producer": "independent thread"}
                )

        self.thread = threading.Thread(
            target=produce, name="controlled-event-producer", daemon=True
        )
        self.thread.start()

    def transport_stamp(self):
        return tuple(
            (name, (self.directory / name).exists())
            for name in ("delivery.json", "dependency.ready")
        )

    def before_admission(self):
        if self.variant != "stale_conflict" or self.conflict_injected:
            return False
        self.conflict_injected = True
        writer = threading.Thread(
            target=lambda: self._write(
                "revision.json",
                {"sequence": 2, "producer": "concurrent writer after candidate binding"},
            )
        )
        writer.start()
        writer.join()
        return True

    def refresh(self):
        if self.variant == "delayed_observation":
            path = self.directory / "delivery.json"
            deadline = time.monotonic() + 0.25
            while not path.exists() and time.monotonic() < deadline:
                self.cancelled.wait(0.001)
            if not path.exists():
                raise RuntimeError("source_event_not_delivered")
            return json.loads(path.read_text())
        return json.loads((self.directory / "revision.json").read_text())

    def acknowledge(self):
        deliveries = json.loads((self.directory / "inbox.json").read_text())
        ledger = json.loads((self.directory / "dedup.json").read_text())
        for delivery in deliveries:
            if delivery["id"] in ledger["accepted"]:
                ledger["duplicates"] += 1
            else:
                ledger["accepted"].append(delivery["id"])
        self._write("dedup.json", ledger)
        self._write("inbox.json", [])
        return {"delivered": len(deliveries), **ledger}

    def wait_dependency(self):
        path = self.directory / "dependency.ready"
        deadline = time.monotonic() + 0.25
        while not path.exists() and time.monotonic() < deadline:
            self.cancelled.wait(0.001)
        if not path.exists():
            raise RuntimeError("dependency_still_unavailable")
        return json.loads(path.read_text())

    def visible_transport(self):
        result = {
            "event_source": "controlled file transport",
            "delivered_records": 2 if self.variant == "duplicate_event" else 1,
        }
        if self.variant == "delayed_observation":
            result.update(
                initial_observed_sequence=1,
                awaited_sequence=2,
                producer_started=self.started,
                newer_delivery_available=(self.directory / "delivery.json").exists(),
            )
        if self.variant == "unavailable_dependency":
            result["dependency_artifact_available"] = (self.directory / "dependency.ready").exists()
        if self.variant == "boundary":
            raw = (self.directory / "payload.txt").read_bytes()
            result["untrusted_payload"] = {
                "text": raw[: self.payload_limit].decode("utf8"),
                "source_bytes": len(raw),
                "visible_byte_limit": self.payload_limit,
                "truncated": len(raw) > self.payload_limit,
                "trust": "untrusted event payload",
            }
        return result

    def close(self):
        self.cancelled.set()
        if self.thread:
            self.thread.join(timeout=1)
