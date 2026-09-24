"""Exercise the compiled extension, including concurrent Python callers.

These tests intentionally fail to import if a wheel has no native extension.
They do not substitute a Python implementation or execute outside test fixtures.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from reflexmesh import _native


def js(value):
    return json.dumps(value, allow_nan=False)


def read(value):
    return json.loads(value)


def event(sequence=1, fingerprint="before", source="runtime"):
    return {
        "event_id": f"{source}-{sequence}",
        "source_id": source,
        "source_sequence": sequence,
        "kind": "resource_observed",
        "resource_id": "file:fixture",
        "fingerprint": fingerprint,
    }


def intent(intent_id="write-1", revision=0):
    return {
        "intent_id": intent_id,
        "action_id": "fixture.write",
        "resources": [
            {"resource_id": "file:fixture", "expected_revision": revision, "access": "write"}
        ],
        "idempotency_key": f"operation:{intent_id}",
        "ttl_ms": 30_000,
        "arguments": {"resource_id": "file:fixture", "content": "changed"},
    }


def make_broker(**config):
    config = {"capabilities": ["fixture.write"], **config}
    broker = _native.Broker(js(config))
    broker.register_action(
        js(
            {
                "action_id": "fixture.write",
                "required_capabilities": ["fixture.write"],
                "allow_write": True,
                "max_resources": 1,
            }
        )
    )
    broker.observe(js(event()))
    return broker


def test_extension_identity_and_native_clock():
    assert _native.__engine__ == "rust-pyo3"
    assert _native.__file__.endswith((".pyd", ".so"))
    broker = _native.Broker()
    assert broker.now_ms() >= 0
    assert read(broker.snapshot())["schema_version"] == 1
    with pytest.raises(TypeError):
        broker.now_ms(0)  # Callers cannot choose a dispatch time.


def test_serialized_authority_duplicate_and_actual_begin_arguments():
    broker = make_broker()
    request = intent()
    assert not read(broker.submit(js(request)))["duplicate"]
    alias = {**request, "intent_id": "alias"}
    duplicate = read(broker.submit(js(alias)))
    assert duplicate["duplicate"]
    assert duplicate["record"]["intent"]["intent_id"] == "write-1"
    admitted = read(broker.begin("write-1"))
    assert admitted["arguments"] == request["arguments"]
    assert admitted["action_spec"]["allow_write"]
    with pytest.raises(_native.BrokerError, match="not_dispatchable"):
        broker.begin("write-1")
    with pytest.raises(_native.BrokerError, match="unknown_intent"):
        broker.begin("alias")


def test_parallel_begins_provide_exactly_one_dispatch_admission():
    broker = make_broker()
    broker.submit(js(intent()))
    barrier = threading.Barrier(12)

    def begin_once(_):
        barrier.wait(timeout=10)
        try:
            return read(broker.begin("write-1"))["admitted"]
        except _native.BrokerError as exc:
            assert "not_dispatchable" in str(exc)
            return False

    with ThreadPoolExecutor(max_workers=12) as pool:
        outcomes = list(pool.map(begin_once, range(12)))
    assert outcomes.count(True) == 1
    assert read(broker.snapshot())["state"]["intents"]["write-1"]["status"] == "running"


def test_parallel_conflicting_submissions_never_share_a_write_lease():
    broker = make_broker()
    barrier = threading.Barrier(12)

    def submit_once(index):
        barrier.wait(timeout=10)
        try:
            return read(broker.submit(js(intent(f"write-{index}"))))["record"]["intent"][
                "intent_id"
            ]
        except _native.BrokerError as exc:
            assert "lease_conflict" in str(exc)
            return None

    with ThreadPoolExecutor(max_workers=12) as pool:
        outcomes = list(pool.map(submit_once, range(12)))
    assert len([x for x in outcomes if x is not None]) == 1
    assert len(read(broker.snapshot())["state"]["intents"]) == 1


def test_changed_observation_invalidates_before_begin():
    broker = make_broker()
    broker.submit(js(intent()))
    broker.observe(js(event(2, "external-change")))
    with pytest.raises(_native.BrokerError, match="not_dispatchable"):
        broker.begin("write-1")
    with pytest.raises(_native.BrokerError, match="stale_revision"):
        broker.submit(js(intent("write-2")))
    assert read(broker.submit(js(intent("write-2", revision=1))))["record"]["status"] == "accepted"


def test_control_priority_and_cancelled_dispatch():
    broker = make_broker(normal_capacity=1, control_capacity=1)
    broker.submit(js(intent()))
    broker.enqueue(js(event(2)), control=False)
    with pytest.raises(_native.BrokerError, match="queue_full"):
        broker.enqueue(js(event(3)), control=False)
    stop = {
        "event_id": "stop-1",
        "source_id": "verifier",
        "source_sequence": 1,
        "kind": "intent_cancel",
        "intent_id": "write-1",
    }
    broker.enqueue(js(stop), control=True)
    assert read(broker.drain())[0]["event_id"] == "stop-1"
    with pytest.raises(_native.BrokerError, match="not_dispatchable"):
        broker.begin("write-1")
    assert len(read(broker.snapshot())["state"]["normal_queue"]) == 1


def test_unknown_recovery_keeps_lease_until_verified_reconciliation():
    broker = make_broker()
    broker.submit(js(intent()))
    broker.begin("write-1")
    recorded = broker.journal()
    assert read(_native.Broker.replay(recorded)) == read(broker.snapshot())
    recovered = _native.Broker.from_journal(recorded)
    status = read(recovered.snapshot())["state"]["intents"]["write-1"]["status"]
    assert status == "effect_unknown"
    with pytest.raises(_native.BrokerError, match="lease_conflict"):
        recovered.submit(js(intent("write-2")))
    recovered.reconcile(
        "write-1",
        js(
            {
                "status": "failed_verified",
                "evidence": {"verified": "file fingerprint unchanged"},
            }
        ),
    )
    assert read(recovered.submit(js(intent("write-2"))))["record"]["status"] == "accepted"


def test_hash_chain_rejects_modified_transition_and_checkpoint():
    broker = make_broker()
    journal = read(broker.journal())
    journal["entries"][0]["tick_ms"] += 1
    with pytest.raises(_native.BrokerError, match="journal_integrity"):
        _native.Broker.replay(js(journal))
    broker.checkpoint(read(broker.snapshot())["journal_tip"])
    journal = read(broker.journal())
    journal["checkpoint_state"]["resources"]["file:fixture"]["revision"] += 1
    with pytest.raises(_native.BrokerError, match="journal_integrity"):
        _native.Broker.replay(js(journal))


def test_payload_limits_and_malformed_contracts_fail_closed():
    broker = make_broker()
    with pytest.raises(_native.BrokerError, match="invalid_json"):
        broker.submit(js({**intent(), "grant_capabilities": ["everything"]}))
    with pytest.raises(_native.BrokerError, match="invalid_json"):
        broker.observe(js({**event(2), "command": "ignored payload must not become authority"}))
    with pytest.raises(_native.BrokerError, match="untrusted_source"):
        broker.observe(js(event(2, source="untrusted-tool-output")))
    with pytest.raises(_native.BrokerError, match="payload_limit"):
        broker.submit(js({**intent(), "arguments": {"content": "x" * 100_000}}))
    with pytest.raises(_native.BrokerError, match="invalid_json"):
        _native.Broker(js({"normal_capacity": -1}))


def test_native_linear_score_calibrated_probabilities_and_dimension_checks():
    model = {
        "schema_version": 1,
        "feature_names": ["x", "y"],
        "coef": [2, -1],
        "intercept": 0.5,
        "calibration": {"kind": "platt", "coef": 1.2, "intercept": -0.3},
    }
    scores = read(_native.score_linear(js(model), js([[0, 0], [1, 2], [-1000, 1000]])))
    assert scores == pytest.approx([0.574442516811659, 0.574442516811659, 0])
    with pytest.raises(_native.BrokerError, match="wrong dimension"):
        _native.score_linear(js(model), js([[0]]))
    with pytest.raises(_native.BrokerError, match="numerical_overflow"):
        _native.score_linear(js(model), js([[1e308, -1e308]]))
