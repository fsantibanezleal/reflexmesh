import pytest
from reflexmesh.contracts import Decision
from reflexmesh.environments import CaseSpec, SoftwareEnvironment, case_matrix
from reflexmesh.policies.rules import BehaviorTreePolicy, GuardedFSMPolicy


@pytest.mark.parametrize("spec", case_matrix(seeds=1), ids=lambda s: s.episode_id)
def test_all_owned_cases_have_independent_actual_effect_truth(spec):
    policy = GuardedFSMPolicy()
    with SoftwareEnvironment(spec) as env:
        while not env.terminal:
            _, outcome = env.step(policy.predict(env.observe()))
            assert not outcome.contract_violation
        assert env.verify(), env.history
        assert env.runtime.snapshot()["state"]["intents"]
        assert env.runtime.replay()["state"] == env.runtime.snapshot()["state"]


@pytest.mark.parametrize("family", [f"C{i:02d}" for i in range(1, 21)])
def test_behavior_tree_executes_registered_effects(family):
    with SoftwareEnvironment(CaseSpec(family, "nominal", 41001)) as env:
        policy = BehaviorTreePolicy()
        while not env.terminal:
            decision = policy.predict(env.observe())
            assert decision.diagnostics["node_trace"]
            env.step(decision)
        assert env.verify(), env.history


def test_denied_boundary_cannot_write_and_runtime_rechecks_changed_file():
    with SoftwareEnvironment(CaseSpec("C01", "boundary", 41000)) as env:
        obs = env.observe()
        _, outcome = env.step(Decision("test", "a-escape"))
        assert outcome.contract_violation and not env.verify()
    with SoftwareEnvironment(CaseSpec("C01", "nominal", 41000)) as env:
        obs = env.observe()
        (env.root / "source.txt").write_text("externally replaced")
        from reflexmesh._native import BrokerError

        with pytest.raises(BrokerError, match="stale_revision"):
            env.runtime.execute(obs, Decision("test", "a-move"))
        assert not (env.root / "destination/document.txt").exists()


def test_ambiguous_post_only_reconciles_one_effect():
    with SoftwareEnvironment(CaseSpec("C06", "nominal", 41000)) as env:
        env.step("a-http_post")
        assert env.service.writes == 1 and env.state["write_unknown"]
        env.step("a-http_post")
        assert env.service.writes == 1
        env.step("a-http_status")
        assert env.verify() and env.service.writes == 1
