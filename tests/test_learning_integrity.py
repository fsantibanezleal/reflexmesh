from dataclasses import replace

import numpy as np
import pytest
from reflexmesh.environments import CaseSpec, SoftwareEnvironment
from reflexmesh.features import FEATURE_NAMES, feature_matrix
from reflexmesh.learning.dataset import assert_disjoint
from reflexmesh.policies.classical import LinUCBPolicy
from reflexmesh.policies.control import LookaheadPolicy, TransitionNetwork


def test_features_exclude_identity_and_candidate_permutation_is_equivariant():
    with SoftwareEnvironment(CaseSpec("C01", "nominal", 42000)) as env:
        obs = env.observe()
        changed = replace(
            obs,
            goal_id="unseen-seed-family-secret",
            event_id="different",
            revision=999,
            state={**obs.state, "oracle_action": "a-bad", "test_label": False, "seed": 2},
        )
        np.testing.assert_array_equal(feature_matrix(obs), feature_matrix(changed))
        reversed_obs = replace(obs, candidates=tuple(reversed(obs.candidates)))
        np.testing.assert_array_equal(feature_matrix(obs)[::-1], feature_matrix(reversed_obs))
        assert feature_matrix(obs).shape[1] == len(FEATURE_NAMES)


def test_group_and_seed_overlap_rejected_even_different_episode_names():
    a = {"episode_ids": ["a"], "group_ids": ["g"], "environment_seeds": [1]}
    b = {"episode_ids": ["b"], "group_ids": ["h"], "environment_seeds": [1]}
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(a, b)


def test_linucb_inverse_matches_direct_ridge_solution():
    policy = LinUCBPolicy(dimension=4)
    x = np.asarray([[1, 0, 1, 0], [0, 2, 0, 1], [1, 1, 0, 0]], dtype=float)
    reward = np.asarray([1.0, -0.1, 0.2])
    for vector, value in zip(x, reward, strict=True):
        policy.update(vector, value)
    np.testing.assert_allclose(policy.a_inv, np.linalg.inv(np.eye(4) + x.T @ x), atol=1e-12)
    np.testing.assert_allclose(
        policy.a_inv @ policy.b, np.linalg.solve(np.eye(4) + x.T @ x, x.T @ reward), atol=1e-12
    )


def test_transition_search_respects_declared_expansion_budget():
    with SoftwareEnvironment(CaseSpec("C01", "nominal", 42000)) as env:
        decision = LookaheadPolicy(TransitionNetwork(), expansions=7).predict(env.observe())
        assert decision.diagnostics["expansions"] <= 7
        assert decision.confidence is None and decision.diagnostics["scores_are_utility"]


def test_contracts_reject_nonfinite_nested_values_and_negative_duration():
    from reflexmesh.contracts import Candidate, Decision, Outcome

    with pytest.raises(ValueError):
        Candidate("a", "tool", {"nested": [float("nan")]})
    with pytest.raises(ValueError):
        Decision("test", "a", scores={"a": float("inf")})
    with pytest.raises(ValueError):
        Outcome("goal", "a", None, False, 0.0, -1.0, "pending")
    with SoftwareEnvironment(CaseSpec("C01", "nominal", 42000)) as env, pytest.raises(ValueError):
        replace(env.observe(), budget_remaining=float("nan"))
    assert Outcome("goal", "a", None, False, 0.0, 0.0, "pending").success is None
