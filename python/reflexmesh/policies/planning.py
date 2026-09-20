"""Lightweight real planner policies; no training framework dependency."""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from ..contracts import Decision, ModelRequiredError, Observation, validate_decision
from .rules import GuardedFSMPolicy


def proposal_binding(observation):
    payload = {
        "goal_id": observation.goal_id,
        "revision": observation.revision,
        "state": observation.state,
        "capabilities": observation.capabilities,
        "budget": observation.budget_remaining,
        "deadline": observation.deadline_remaining_ms,
        "candidates": [
            {
                "action_id": c.action_id,
                "tool": c.tool,
                "arguments": c.arguments,
                "effects": c.effects,
                "capabilities": c.required_capabilities,
                "resources": c.resource_versions,
                "allowed": c.allowed,
            }
            for c in sorted(observation.candidates, key=lambda c: c.action_id)
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


class DirectPlannerPolicy:
    policy_id = "M09"

    def __init__(self, planner):
        if planner is None:
            raise ModelRequiredError("M09 requires a configured real planner")
        self.planner = planner

    def reset(self, goal_id: str | None = None) -> None:
        pass

    def predict(self, observation: Observation) -> Decision:
        start = time.perf_counter()
        decision = self.planner.plan(observation)
        validate_decision(observation, decision)
        return replace(
            decision,
            policy_id=self.policy_id,
            diagnostics={
                **decision.diagnostics,
                "delegated": True,
                "planner_ms": (time.perf_counter() - start) * 1000,
                "planner_model": self.planner.model_id,
            },
        )


class AsyncPlannerPolicy:
    policy_id = "M10"

    def __init__(self, planner):
        if planner is None:
            raise ModelRequiredError("M10 requires a real asynchronous planner")
        self.planner = planner
        self.fallback = GuardedFSMPolicy()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reflexmesh-M10")
        self.pending = None
        self.origin = None

    def finish_episode(self):
        if self.pending is None:
            return {"planner_calls": 0}
        proposal = self.pending.result()
        return {
            "planner_calls": 1,
            "completed_proposal": proposal.to_dict(),
            "origin": self.origin,
            "background_work_accounted": True,
        }

    def reset(self, goal_id: str | None = None) -> None:
        try:
            if self.pending is not None:
                self.pending.result()  # account for completed work before next independent episode
        finally:
            self.pending, self.origin = None, None

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)

    def predict(self, observation: Observation) -> Decision:
        diagnostics = {"delegated": False, "planner_model": self.planner.model_id}
        if self.pending is None:
            self.origin = proposal_binding(observation)
            self.pending = self.executor.submit(self.planner.plan, observation)
            diagnostics.update(delegated=True, planner_pending=True)
        elif self.pending.done():
            proposal = self.pending.result()
            if self.origin == proposal_binding(observation):
                validate_decision(observation, proposal)
                return replace(
                    proposal,
                    policy_id=self.policy_id,
                    diagnostics={**proposal.diagnostics, "planner_applied": True},
                )
            diagnostics["stale_plan_discarded"] = True
        decision = self.fallback.predict(observation)
        return replace(
            decision,
            policy_id=self.policy_id,
            reason="asynchronous_fsm_planner",
            diagnostics=diagnostics,
        )
