from dataclasses import replace

from reflexmesh.environments import CaseSpec, SoftwareEnvironment
from reflexmesh.learning.routing import _branch, eligible_records, pair_signature


def test_pair_signature_detects_readiness_change_and_preserves_candidate_order_invariance():
    with SoftwareEnvironment(CaseSpec("C01", "nominal", 50000, "validation")) as env:
        observation = env.observe()
        signature = pair_signature(observation)
        assert (
            pair_signature(replace(observation, candidates=tuple(reversed(observation.candidates))))
            == signature
        )
        assert (
            pair_signature(replace(observation, state={**observation.state, "job_ready": 1}))
            != signature
        )
        assert pair_signature(replace(observation, history=("spawn",))) != signature


def test_mismatched_reset_branch_retains_actual_outcome_without_becoming_training_pair():
    spec = CaseSpec("C01", "nominal", 50000, "validation")
    result = _branch(spec, [], "a-move", "deliberately-different-pre-action-input")
    assert result["success"] and not result["start_features_match"]
    assert result["native_receipts"] > 0 and result["pairing_exclusion_reason"]
    records = [
        {"fast": {"M11": {"pair_eligible": False, "observed_utility_difference": 100}}},
        {"fast": {"M11": {"pair_eligible": True, "observed_utility_difference": -100}}},
    ]
    assert eligible_records(records, "M11") == records[1:]
