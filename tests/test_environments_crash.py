"""Actual owned-worker exit, native recovery, and external effect verification."""

import json

import pytest
from reflexmesh.environments.crash import CrashRecovery


def test_crash_restart_requires_evidence_and_never_reexecutes(tmp_path):
    root = tmp_path / "crash"
    with CrashRecovery.create(root, value=40000) as fixture:
        assert fixture.worker["pid"] > 0
        assert fixture.worker["exit_code"] == 23
        assert fixture.status == "effect_unknown"
        assert not fixture.verify()
        original = fixture.effect_path.read_bytes()
        evidence = fixture.evidence()
        assert evidence["matches_expected_once"] and evidence["effect_count"] == 1
        assert fixture.runtime.replay()["state"] == fixture.runtime.snapshot()["state"]
        assert fixture.effect_path.read_bytes() == original
    with CrashRecovery.open(root, value=40000) as fixture:
        assert fixture.status == "effect_unknown"
        assert fixture.effect_path.read_bytes() == original
        assert fixture.reconcile()["status"] == "completed_verified"
        assert fixture.verify()
        assert fixture.reconcile()["status"] == "completed_verified"
        assert fixture.effect_path.read_bytes() == original
    with CrashRecovery.open(root, value=40000) as fixture:
        assert fixture.status == "completed_verified" and fixture.verify()
        assert fixture.effect_path.read_bytes() == original


def test_unknown_crashed_intent_retains_write_lease_and_duplicate_is_not_dispatched(tmp_path):
    with CrashRecovery.create(tmp_path / "crash", value=12) as fixture:
        intent = fixture.runtime.snapshot()["state"]["intents"][fixture.intent_id]["intent"]
        before = fixture.effect_path.read_bytes()
        duplicate = json.loads(fixture.runtime.broker.submit(json.dumps(intent)))
        assert duplicate["duplicate"]
        assert duplicate["record"]["status"] == "effect_unknown"
        candidate = fixture.runtime.candidate("crash.append", intent["arguments"])
        with pytest.raises(RuntimeError, match="lease_conflict"):
            fixture.runtime.execute_candidate(candidate, intent_id="unsafe-retry")
        assert fixture.status == "effect_unknown"
        assert fixture.effect_path.read_bytes() == before
        fixture.reconcile()
        assert fixture.verify()


@pytest.mark.parametrize("damage", ["missing", "changed", "malformed", "duplicate", "oversized"])
def test_reconciliation_rejects_unverified_effects_without_releasing_unknown(tmp_path, damage):
    with CrashRecovery.create(tmp_path / "crash", value=7) as fixture:
        if damage == "missing":
            fixture.effect_path.unlink()
        elif damage == "changed":
            fixture.effect_path.write_text('{"value":8}\n', encoding="utf8")
        elif damage == "malformed":
            fixture.effect_path.write_bytes(b"\xff\n")
        elif damage == "duplicate":
            fixture.effect_path.write_bytes(fixture.effect_path.read_bytes() * 2)
        else:
            fixture.effect_path.write_bytes(b"x" * 65_537)
        with pytest.raises(RuntimeError, match="durable effect"):
            fixture.reconcile()
        assert fixture.status == "effect_unknown"
        assert not fixture.verify()


def test_crash_before_effect_cannot_be_claimed_as_success(tmp_path):
    with CrashRecovery.create(tmp_path / "crash", value=3, crash_point="before_effect") as fixture:
        assert fixture.status == "effect_unknown"
        assert not fixture.effect_path.exists()
        assert fixture.evidence()["effect_count"] == 0
        with pytest.raises(RuntimeError, match="durable effect"):
            fixture.reconcile()
        assert not fixture.verify()


def test_fixture_creation_rejects_reuse_and_open_rejects_identity_change(tmp_path):
    root = tmp_path / "crash"
    with CrashRecovery.create(root, value=9), pytest.raises(FileExistsError):
        CrashRecovery.create(root, value=9)
    with pytest.raises(RuntimeError, match="identity"):
        CrashRecovery.open(root, value=10)
    with pytest.raises(RuntimeError, match="identity"):
        CrashRecovery.open(root, value=9, intent_id="another-intent")


def test_reconciled_state_does_not_hide_later_effect_corruption(tmp_path):
    with CrashRecovery.create(tmp_path / "crash", value=5) as fixture:
        fixture.reconcile()
        assert fixture.verify()
        fixture.effect_path.write_bytes(fixture.effect_path.read_bytes() * 2)
        assert not fixture.verify()
        with pytest.raises(RuntimeError, match="durable effect"):
            fixture.reconcile()
