import subprocess
import sys
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


def test_base_rule_path_does_not_import_training_frameworks():
    script = """
import sys
for name in ('torch','sklearn','xgboost','gymnasium','sb3_contrib','onnxruntime'):
    sys.modules[name]=None
from reflexmesh.pipeline import make_policy
from reflexmesh.environments import SoftwareEnvironment,CaseSpec
for method in ('M01','M02'):
    policy=make_policy(method,'.')
    with SoftwareEnvironment(CaseSpec('C01','nominal',82000)) as env:
        env.step(policy.predict(env.observe()))
        assert env.verify()
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)


def test_invalid_policy_decision_is_recorded_failed_cell_without_effects():
    class InvalidPolicy:
        policy_id = "M09"

        def reset(self, goal_id=None):
            pass

        def predict(self, obs):
            raise ValueError("provider output violates contract")

    result = run_episode(InvalidPolicy(), CaseSpec("C01", "nominal", 82000))
    assert result["metrics"]["task_success"] == 0
    assert result["metrics"]["invalid_policy_output"] == 1
    assert result["metrics"]["native_receipts"] == 0
    assert result["policy_errors"][0]["type"] == "ValueError"
    assert result["decisions"][0]["diagnostics"]["provider_usage"] is None


def test_invalid_unused_background_proposal_preserves_true_task_outcome_and_resets():
    from reflexmesh.policies.planning import AsyncPlannerPolicy

    class InvalidPlanner:
        model_id = "unit-test-invalid-output"

        def plan(self, obs):
            raise ValueError("malformed proposal")

    policy = AsyncPlannerPolicy(InvalidPlanner())
    try:
        for seed in (82000, 82001):
            result = run_episode(policy, CaseSpec("C01", "nominal", seed))
            assert result["metrics"]["task_success"] == 1
            assert result["metrics"]["invalid_policy_output"] == 1
            assert result["planner_work"]["provider_usage"] is None
    finally:
        policy.close()
