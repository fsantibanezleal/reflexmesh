import threading
import time
from dataclasses import replace

import pytest
from reflexmesh.contracts import Candidate, Decision, Observation
from reflexmesh.planners import AsyncDeliberator, _parse


def observation():
    return Observation("event", "goal", 1, "changed", {}, (Candidate("a", "tool"),))


def test_unregistered_model_action_rejected():
    with pytest.raises(ValueError):
        _parse(
            '{"candidate_id":"unregistered","mode":"act","reason":"x"}', observation(), "model", {}
        )


def test_async_arrival_revalidated_and_old_proposal_dropped():
    entered, release = threading.Event(), threading.Event()

    class Planner:
        def plan(self, obs):
            entered.set()
            assert release.wait(5)
            return Decision("model", "a")

    controller = AsyncDeliberator(Planner(), max_pending=1)
    try:
        controller.submit("one", observation(), 5000)
        assert entered.wait(5)
        assert controller.poll("one", replace(observation(), revision=2)) is None
        assert controller.status("one")["status"] == "stale"
        with pytest.raises(RuntimeError, match="capacity"):
            controller.submit("two", observation(), 5000)
        release.set()
    finally:
        release.set()
        controller.close()


def test_invalid_planner_response_becomes_failed_request():
    class Planner:
        def plan(self, obs):
            return Decision("model", "bad")

    controller = AsyncDeliberator(Planner())
    try:
        controller.submit("one", observation(), 5000)
        for _ in range(100):
            controller.poll("one", observation())
            if controller.status("one")["status"] != "pending":
                break
            time.sleep(0.001)
        assert controller.status("one")["status"] == "failed"
    finally:
        controller.close()


def test_changed_arguments_with_same_revision_invalidate_slow_proposal():
    release = threading.Event()

    class Planner:
        def plan(self, obs):
            assert release.wait(5)
            return Decision("model", "a")

    controller = AsyncDeliberator(Planner())
    try:
        original = observation()
        controller.submit("one", original, 5000)
        changed = replace(original, candidates=(Candidate("a", "tool", {"different": "argument"}),))
        assert controller.poll("one", changed) is None
        assert controller.status("one")["status"] == "stale"
    finally:
        release.set()
        controller.close()
