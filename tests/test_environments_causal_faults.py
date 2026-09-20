import json
import time

from reflexmesh.environments import CaseSpec, SoftwareEnvironment
from reflexmesh.policies.rules import GuardedFSMPolicy


def finish(env):
    policy = GuardedFSMPolicy()
    while not env.terminal:
        env.step(policy.predict(env.observe()))
    assert env.verify(), env.history


def test_delayed_delivery_starts_after_first_observation_and_refresh_reads_event():
    with SoftwareEnvironment(CaseSpec("C01", "delayed_observation", 45000)) as env:
        delivery = env.root / ".scenario/delivery.json"
        assert not delivery.exists() and not env.faults.started
        observed = env.observe()
        assert observed.state["stale"] and env.faults.started
        env.faults.thread.join(timeout=1)
        assert delivery.exists()
        newer = env.observe()
        assert newer is not observed
        assert newer.state["event_transport"]["newer_delivery_available"]
        _, outcome = env.step("a-refresh")
        assert outcome.effects["source_event"]["sequence"] == 2
        assert not env.state["stale"]
        finish(env)


def test_concurrent_variant_rejects_old_candidate_in_native_admission():
    with SoftwareEnvironment(CaseSpec("C01", "stale_conflict", 45000)) as env:
        observed = env.observe()
        assert not observed.state["stale"]
        _, outcome = env.step("a-move")
        assert outcome.status == "failed" and "stale_revision" in outcome.effects["error"]
        assert (env.root / "source.txt").exists()
        assert json.loads((env.root / ".scenario/revision.json").read_text())["sequence"] == 2
        assert env.state["stale"]
        finish(env)


def test_duplicate_delivery_updates_persistent_dedup_and_native_event_ledger():
    with SoftwareEnvironment(CaseSpec("C01", "duplicate_event", 45000)) as env:
        inbox = json.loads((env.root / ".scenario/inbox.json").read_text())
        assert len(inbox) == 2 and inbox[0]["id"] == inbox[1]["id"]
        env.observe()
        _, outcome = env.step("a-ack_event")
        assert outcome.effects["native_duplicate"]["duplicate"] is True
        ledger = json.loads((env.root / ".scenario/dedup.json").read_text())
        assert ledger == {"accepted": ["delivery-45000"], "duplicates": 1}
        env.step("a-ack_event")
        assert json.loads((env.root / ".scenario/dedup.json").read_text()) == ledger
        assert env.runtime.replay()["state"] == env.runtime.snapshot()["state"]
        finish(env)


def test_dependency_is_created_by_producer_not_wait_action():
    with SoftwareEnvironment(CaseSpec("C01", "unavailable_dependency", 45000)) as env:
        path = env.root / ".scenario/dependency.ready"
        assert not path.exists()
        env.observe()
        deadline = time.monotonic() + 1
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert path.exists() and env.steps == 0
        _, outcome = env.step("a-wait_dependency")
        assert outcome.effects["observed_readiness"]["producer"] == "independent thread"
        finish(env)


def test_boundary_uses_actual_oversized_payload_and_real_schema_denial():
    with SoftwareEnvironment(CaseSpec("C14", "boundary", 45000)) as env:
        observed = env.observe()
        payload = observed.state["event_transport"]["untrusted_payload"]
        assert payload["source_bytes"] == 513 and len(payload["text"].encode()) == 512
        assert payload["truncated"] and not observed.state["boundary_probe"]["admitted"]
        assert "unregistered operation arguments" in observed.state["boundary_probe"]["reason"]
        finish(env)


def test_incremental_index_reads_only_changed_content_and_keeps_unchanged_entries():
    with SoftwareEnvironment(CaseSpec("C02", "nominal", 45000)) as env:
        before = json.loads((env.root / "index.json").read_text())
        env.step("a-index")
        assert env.verify()
        after = json.loads((env.root / "index.json").read_text())
        audit = json.loads((env.root / "index-audit.json").read_text())
        assert after["unchanged-a.txt"] == before["unchanged-a.txt"]
        assert set(audit["content_reads"]) == {"changed.txt", "added.txt"}
        assert len(audit["reused"]) == 2 and audit["deleted"] == ["deleted.txt"]


def test_attacker_text_is_visible_after_tool_result_but_cannot_grant_authority():
    with SoftwareEnvironment(CaseSpec("C15", "nominal", 45000)) as env:
        env.step("a-inspect")
        observed = env.observe()
        result = observed.state["untrusted_tool_result"]
        assert "Ignore the goal" in result["text"]
        assert "untrusted" in result["trust"]
        assert observed.state["red_team_probe"]["admitted"] is False
        assert env.outside_canary.read_text() == "outside-preserve"
        assert observed.candidate("a-escape").allowed is False
        finish(env)


def test_concurrent_file_writer_runs_after_candidate_binding_before_admission():
    with SoftwareEnvironment(CaseSpec("C11", "nominal", 45000)) as env:
        observed = env.observe()
        assert observed.state["expected_revision"] == 1 and not env.concurrent_writer_done
        _, outcome = env.step("a-cas_write")
        assert "stale_revision" in outcome.effects["error"]
        assert json.loads((env.root / "shared.json").read_text()) == {"revision": 2, "value": 45100}
        finish(env)


def test_unsupported_argument_rejection_is_actual_runtime_validation():
    with SoftwareEnvironment(CaseSpec("C14", "nominal", 45000)) as env:
        env.step("a-reject")
        proof = json.loads((env.root / "rejection.json").read_text())
        assert proof["source"] == "actual Runtime.candidate validation"
        assert "unregistered operation arguments" in proof["reason"]
        assert not (env.root / "invoked.json").exists()
        finish(env)


def test_asynchronous_proposal_has_actual_future_and_input_binding():
    with SoftwareEnvironment(CaseSpec("C13", "nominal", 45000)) as env:
        assert env.proposal_future is None
        env.observe()
        assert env.proposal_future is not None
        env.step("a-wait")
        proposal = json.loads((env.root / "proposal.json").read_text())
        assert proposal["input_sha256"] and "not an LLM" in proposal["source"]
        finish(env)


def test_out_of_order_events_are_rejected_by_native_source_sequence():
    with SoftwareEnvironment(CaseSpec("C18", "nominal", 45000)) as env:
        env.step("a-reorder")
        proof = json.loads((env.root / "ordering-admission.json").read_text())
        assert [item["accepted"] for item in proof] == [True, False, False]
        assert all("stale_sequence" in item["error"] for item in proof[1:])
        finish(env)


def test_http_reliability_changes_after_an_observed_success():
    with SoftwareEnvironment(CaseSpec("C19", "nominal", 45000)) as env:
        assert env.service.failures == 0 and not env.service.changed_reliability
        _, baseline = env.step("a-http_get")
        assert baseline.effects["http_status"] == 200 and not env.verify()
        assert env.service.changed_reliability and env.service.failures == 3
        _, failure = env.step("a-http_get")
        assert failure.effects["http_status"] == 503
        finish(env)
