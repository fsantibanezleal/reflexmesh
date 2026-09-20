"""Inspectable deterministic baselines with explicit behavior-tree node states."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..contracts import Decision, Observation


def recommended_operation(observation: Observation) -> str:
    s = observation.state
    ops = {c.arguments.get("operation", c.tool) for c in observation.admissible}
    for key, operation in (
        ("stale", "refresh"),
        ("duplicate", "ack_event"),
        ("dependency_missing", "wait_dependency"),
        ("cancel_requested", "cancel"),
    ):
        if s.get(key) and operation in ops:
            return operation
    if "validate" in ops:
        return (
            "validate"
            if s.get("needs_validation")
            else ("accept" if s.get("valid") else "quarantine")
        )
    if "run_test" in ops:
        return "repair_config" if s.get("config_missing") else "run_test"
    if "http_post" in ops:
        return "http_status" if s.get("write_unknown") else "http_post"
    if "http_get" in ops:
        return "backoff" if s.get("failed") else "http_get"
    if "restart" in ops:
        return "restart"
    if "cleanup" in ops:
        return "cleanup" if s.get("pressure") else "write"
    if "extract" in ops:
        return (
            "extract"
            if not s.get("input_ready")
            else ("transform" if not s.get("transformed") else "report")
        )
    if "cas_write" in ops:
        return "rebase" if s.get("conflict") else "cas_write"
    if "spawn" in ops:
        return "collect" if s.get("job_ready") else "spawn"
    if "collect" in ops:
        return "collect" if s.get("job_ready") else "wait"
    if "reject" in ops:
        return "invoke" if s.get("confirmed") else "reject"
    if "reconcile" in ops:
        return "reconcile"
    if "reorder" in ops:
        return "write" if s.get("confirmed") else "reorder"
    if "invoke" in ops:
        return "inspect" if s.get("uncertain") else "invoke"
    if "checksum" in s or "untrusted" in str(s.get("goal", "")).lower():
        return "report" if s.get("confirmed") else "inspect"
    if s.get("index_stale") or "index" in str(s.get("goal", "")).lower():
        return "index"
    if "move" in ops:
        return "move"
    return "refresh" if "refresh" in ops else "stop"


class GuardedFSMPolicy:
    policy_id = "M01"

    def reset(self, goal_id: str | None = None) -> None:
        pass

    def predict(self, observation: Observation) -> Decision:
        operation = recommended_operation(observation)
        for candidate in observation.admissible:
            if candidate.arguments.get("operation", candidate.tool) == operation:
                return Decision(
                    self.policy_id,
                    candidate.action_id,
                    reason="explicit_state_guard",
                    diagnostics={"operation": operation, "learned": False},
                )
        return Decision(self.policy_id, None, "stop", reason="no_supported_rule")


@dataclass
class Node:
    name: str
    condition: Callable[[Observation], bool] | None = None
    operation: str | None = None
    children: list[Node] = field(default_factory=list)
    kind: str = "action"

    def tick(self, observation: Observation, trace: list[dict]) -> tuple[str, str | None]:
        if self.kind == "condition":
            status = "success" if self.condition(observation) else "failure"
            trace.append({"node": self.name, "status": status})
            return status, None
        if self.kind == "action":
            action = next(
                (
                    c.action_id
                    for c in observation.admissible
                    if c.arguments.get("operation", c.tool) == self.operation
                ),
                None,
            )
            status = "running" if action else "failure"
            trace.append({"node": self.name, "status": status})
            return status, action
        for child in self.children:
            status, action = child.tick(observation, trace)
            if self.kind == "sequence" and status != "success":
                return status, action
            if self.kind == "selector" and status != "failure":
                return status, action
        return ("success" if self.kind == "sequence" else "failure"), None


class BehaviorTreePolicy:
    policy_id = "M02"

    def __init__(self):
        # Independent declarative tree, with guarded subtrees and real node
        # statuses. The FSM's recommendation function is not its interpreter.
        def branch(name, condition, operation):
            return Node(
                name,
                kind="sequence",
                children=[
                    Node(name + "-condition", condition=condition, kind="condition"),
                    Node(name + "-action", operation=operation),
                ],
            )

        def available(obs, operation):
            return any(c.arguments.get("operation", c.tool) == operation for c in obs.admissible)

        def state(key):
            return lambda obs: bool(obs.state.get(key))

        def task(operation, branches, fallback):
            return Node(
                operation + "-task",
                kind="sequence",
                children=[
                    Node(
                        operation + "-available",
                        kind="condition",
                        condition=lambda obs: available(obs, operation),
                    ),
                    Node(
                        operation + "-selector",
                        kind="selector",
                        children=[
                            branch(operation + "-" + op, condition, op)
                            for condition, op in branches
                        ]
                        + [Node("default-" + fallback, operation=fallback)],
                    ),
                ],
            )

        self.root = Node(
            "reactive-root",
            kind="selector",
            children=[
                branch("refresh-stale", state("stale"), "refresh"),
                branch("ack-duplicate", state("duplicate"), "ack_event"),
                branch("resolve-dependency", state("dependency_missing"), "wait_dependency"),
                branch("cancel-running", state("cancel_requested"), "cancel"),
                task(
                    "validate",
                    [(state("needs_validation"), "validate"), (state("valid"), "accept")],
                    "quarantine",
                ),
                task("run_test", [(state("config_missing"), "repair_config")], "run_test"),
                task("http_post", [(state("write_unknown"), "http_status")], "http_post"),
                task("http_get", [(state("failed"), "backoff")], "http_get"),
                task("restart", [], "restart"),
                task("cleanup", [(state("pressure"), "cleanup")], "write"),
                task(
                    "extract",
                    [
                        (lambda obs: not obs.state.get("input_ready"), "extract"),
                        (lambda obs: not obs.state.get("transformed"), "transform"),
                    ],
                    "report",
                ),
                task("cas_write", [(state("conflict"), "rebase")], "cas_write"),
                task("spawn", [(state("job_ready"), "collect")], "spawn"),
                task("collect", [(state("job_ready"), "collect")], "wait"),
                task("reject", [(state("confirmed"), "invoke")], "reject"),
                task("reconcile", [], "reconcile"),
                task("reorder", [(state("confirmed"), "write")], "reorder"),
                task("invoke", [(state("uncertain"), "inspect")], "invoke"),
                Node(
                    "untrusted-content-task",
                    kind="sequence",
                    children=[
                        Node(
                            "untrusted-content",
                            kind="condition",
                            condition=lambda obs: "untrusted" in str(obs.state.get("goal", "")),
                        ),
                        task("inspect", [(state("confirmed"), "report")], "inspect"),
                    ],
                ),
                branch(
                    "index-changed-content",
                    lambda obs: (
                        bool(obs.state.get("index_stale"))
                        or "index" in str(obs.state.get("goal", "")).lower()
                    ),
                    "index",
                ),
                task("move", [], "move"),
            ],
        )

    def reset(self, goal_id: str | None = None) -> None:
        pass

    def predict(self, observation: Observation) -> Decision:
        trace: list[dict] = []
        status, action = self.root.tick(observation, trace)
        return Decision(
            self.policy_id,
            action,
            "act" if action else "stop",
            reason="reactive_behavior_tree",
            diagnostics={"tree_status": status, "node_trace": trace},
        )
