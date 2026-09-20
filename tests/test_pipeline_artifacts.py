import threading

import pytest
from reflexmesh.environments import CaseSpec
from reflexmesh.evaluation.runner import evaluate, run_episode
from reflexmesh.pipeline import make_policy


def test_actual_trace_has_native_receipts_and_incremental_callback():
    updates = []
    result = run_episode(
        make_policy("M01", "."),
        CaseSpec("C01", "nominal", 42000),
        on_step=lambda event, decision: updates.append((event, decision)),
    )
    assert result["metrics"]["task_success"] == 1
    assert result["metrics"]["native_receipts"] >= 1
    assert len(updates) == result["metrics"]["steps"]
    assert updates[0][1]["resources"][0]["version"] >= 1


def test_cancel_before_effect_and_resume_lineage_rejection(tmp_path):
    cancellation = threading.Event()
    cancellation.set()
    result = run_episode(
        make_policy("M01", "."), CaseSpec("C01", "nominal", 42000), cancel_event=cancellation
    )
    assert result["status"] == "cancelled" and result["metrics"]["native_receipts"] == 0
    policies = [make_policy("M01", ".")]
    specs = [CaseSpec("C01", "nominal", 42000)]
    evaluate(policies, specs, tmp_path, lineage={"model": "v1"})
    with pytest.raises(ValueError, match="lineage"):
        evaluate(policies, specs, tmp_path, lineage={"model": "v2"})
