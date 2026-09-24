import time
from dataclasses import replace

import numpy as np
from reflexmesh.environments import CaseSpec, SoftwareEnvironment
from reflexmesh.policies.control import LookaheadPolicy, Metacontroller, TransitionNetwork
from reflexmesh.policies.rules import GuardedFSMPolicy


class Benefit:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, x):
        return np.tile([0.0, 1.0], (len(x), 1))

    def predict_gain(self, x):
        return np.ones(len(x))


class SlowPlanner:
    model_id = "unit-test-delayed-planner"

    def __init__(self, delay=0.17):
        self.delay = delay
        self.calls = 0

    def plan(self, obs):
        self.calls += 1
        time.sleep(self.delay)
        return GuardedFSMPolicy().predict(obs)


def controller(planner, **kw):
    return Metacontroller(
        GuardedFSMPolicy(),
        LookaheadPolicy(TransitionNetwork()),
        Benefit(),
        planner,
        use_effects=False,
        **kw,
    )


def test_deliberation_hold_preserves_steps_and_applies_three_real_arrivals():
    planner = SlowPlanner()
    policy = controller(planner)
    holds = 0
    applied = 0
    try:
        with SoftwareEnvironment(CaseSpec("C10", "nominal", 41000)) as env:
            while not env.terminal:
                before = env.steps
                decision = policy.predict(env.observe())
                env.step(decision)
                if decision.diagnostics.get("deliberation_hold"):
                    holds += 1
                    assert env.steps == before
                applied += int(decision.diagnostics.get("planner_applied", False))
            assert env.verify() and env.steps == 3
        assert planner.calls == 3 and applied == 3 and holds >= 3
        assert policy.finish_episode()["planner_calls"] == 3
    finally:
        policy.close()


def test_changed_binding_discards_proposal_and_ttl_bounds_affected_hold():
    policy = controller(SlowPlanner(0.3), planner_ttl_ms=10, max_planner_calls=1)
    try:
        with SoftwareEnvironment(CaseSpec("C01", "nominal", 41000)) as env:
            obs = env.observe()
            decision = policy.predict(obs)
            assert decision.mode == "wait"
            time.sleep(0.012)
            next_decision = policy.predict(obs)
            assert next_decision.mode == "act" and next_decision.diagnostics["planner_timeout"]
    finally:
        policy.close()


def test_same_id_revision_changed_arguments_invalidate_pending_proposal():
    policy = controller(SlowPlanner(0.3), max_planner_calls=1)
    try:
        with SoftwareEnvironment(CaseSpec("C01", "nominal", 41000)) as env:
            obs = env.observe()
            assert policy.predict(obs).mode == "wait"
            changed = replace(
                obs,
                candidates=tuple(
                    replace(
                        c, arguments={**c.arguments, "description": "changed semantic argument"}
                    )
                    for c in obs.candidates
                ),
            )
            assert changed.revision == obs.revision
            policy.predict(changed)
            assert policy.discards[0]["reason"] == "binding_changed"
    finally:
        policy.close()
    policy = controller(SlowPlanner(0.3), max_planner_calls=1)
    try:
        with SoftwareEnvironment(CaseSpec("C01", "nominal", 41000)) as env:
            assert policy.predict(env.observe()).mode == "wait"
            env.step("a-refresh")
            policy.predict(env.observe())
            assert policy.discards[0]["reason"] == "binding_changed"
    finally:
        policy.close()
