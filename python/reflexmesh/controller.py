"""Persistent-goal coordinator with independent urgent and effect execution lanes."""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .contracts import Observation, Policy
from .planners import AsyncDeliberator
from .runtime import Runtime


@dataclass(frozen=True)
class Event:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid4().hex)
    observed_at: float = field(default_factory=time.monotonic)


@dataclass
class Goal:
    goal_id: str
    observe: Callable[[Runtime, Event, tuple[dict, ...]], Observation]
    verify: Callable[[Runtime], bool]
    dependencies: tuple[str, ...] = ()
    max_steps: int = 128
    timeout_seconds: float = 300
    state: str = "pending"
    steps: int = 0
    submitted_at: float = field(default_factory=time.monotonic)
    history: list[dict[str, Any]] = field(default_factory=list)
    active_intent: str | None = None
    active_plan: str | None = None
    error: str | None = None
    cancellation: threading.Event = field(default_factory=threading.Event)


class Controller:
    """Coordinates goals, tools and an optional bounded slow planner.

    A policy can dispatch, wait, stop or delegate. Models never register tools.
    Control events continue while an effector/planner runs in a worker. A goal's
    success comes exclusively from its independent verifier. Goals are bounded;
    native intents and effects are durable through Runtime. Application-level
    goal factories are explicitly resubmitted after restart.
    """

    def __init__(
        self,
        runtime: Runtime,
        policy: Policy | None = None,
        planner: Any = None,
        *,
        max_goals: int = 128,
        normal_capacity: int = 256,
        control_capacity: int = 32,
        max_effects: int = 4,
        tick_seconds: float = 0.01,
        policy_factory: Callable[[str], Policy] | None = None,
    ):
        if min(max_goals, normal_capacity, control_capacity, max_effects) < 1:
            raise ValueError("controller capacities must be positive")
        if (policy is None) == (policy_factory is None):
            raise ValueError("provide either one goal-aware policy or a per-goal policy_factory")
        self.runtime, self.policy = runtime, policy
        self.policy_factory = policy_factory
        self._goal_policies: dict[str, Policy] = {}
        self._goal_policy_locks: dict[str, threading.Lock] = {}
        self.planner = AsyncDeliberator(planner) if planner else None
        self.goals: dict[str, Goal] = {}
        self.max_goals, self.max_effects, self.tick_seconds = max_goals, max_effects, tick_seconds
        self.normal: asyncio.Queue[Event] = asyncio.Queue(normal_capacity)
        self.control: asyncio.Queue[Event] = asyncio.Queue(control_capacity)
        self.effects: dict[str, asyncio.Task] = {}
        self.inferences: dict[str, tuple[asyncio.Task, Observation]] = {}
        self._policy_lock = threading.Lock()
        self._stop = False
        self._listeners: list[Callable[[dict], Any]] = []
        self._last = Event("start")

    def subscribe(self, listener: Callable[[dict], Any]) -> None:
        self._listeners.append(listener)

    async def _emit(self, value: dict) -> None:
        for listener in self._listeners:
            result = listener(value)
            if inspect.isawaitable(result):
                await result

    def submit(self, goal: Goal) -> None:
        if goal.goal_id in self.goals or len(self.goals) >= self.max_goals:
            raise ValueError("duplicate goal or controller goal capacity exhausted")
        if goal.max_steps < 1 or goal.timeout_seconds <= 0:
            raise ValueError("goals require positive step and time bounds")
        if goal.goal_id in goal.dependencies or any(d not in self.goals for d in goal.dependencies):
            raise ValueError("dependencies must reference previously submitted goals")
        if self.policy_factory:
            policy = self.policy_factory(goal.goal_id)
            if any(policy is previous for previous in self._goal_policies.values()):
                raise ValueError("policy_factory must return an independent policy per goal")
            self._goal_policies[goal.goal_id] = policy
            self._goal_policy_locks[goal.goal_id] = threading.Lock()
        self.goals[goal.goal_id] = goal

    def publish(self, event: Event) -> None:
        urgent = event.kind in {"cancel_goal", "revoke_capability", "shutdown"}
        (self.control if urgent else self.normal).put_nowait(event)

    async def _cancel(self, goal: Goal, reason: str) -> None:
        if goal.state in {"succeeded", "failed", "cancelled", "blocked"}:
            return
        goal.state = "cancelling" if goal.active_intent else "cancelled"
        goal.error = reason
        goal.cancellation.set()
        if goal.active_plan and self.planner:
            self.planner.cancel(goal.active_plan)
            goal.active_plan = None
        if goal.active_intent and goal.active_intent in self.runtime.snapshot()["state"]["intents"]:
            self.runtime.cancel(goal.active_intent)
        await self._emit(
            {
                "kind": "goal_cancel",
                "goal_id": goal.goal_id,
                "reason": reason,
                "effect_rollback_claimed": False,
            }
        )

    async def _control_event(self, event: Event) -> None:
        if event.kind == "cancel_goal":
            if event.payload.get("goal_id") in self.goals:
                await self._cancel(self.goals[event.payload["goal_id"]], "requested")
        elif event.kind == "revoke_capability":
            self.runtime.revoke(event.payload["capability"])
        elif event.kind == "shutdown":
            self._stop = True
            for goal in self.goals.values():
                await self._cancel(goal, "shutdown")

    async def _finish_effects(self) -> None:
        for goal_id, task in list(self.effects.items()):
            if not task.done():
                continue
            goal = self.goals[goal_id]
            del self.effects[goal_id]
            goal.active_intent = None
            try:
                receipt = task.result()
                goal.history.append({"kind": "effect", "receipt": receipt})
                if receipt["status"] == "effect_unknown":
                    goal.state, goal.error = "blocked", "effect_requires_reconciliation"
                elif goal.state == "cancelling":
                    goal.state = "cancelled"
                elif goal.verify(self.runtime):
                    goal.state = "succeeded"
            except Exception as exc:  # noqa: BLE001 - user effect failure becomes an explicit failed goal
                goal.state, goal.error = "failed", type(exc).__name__ + ": " + str(exc)[:1000]
            await self._emit(
                {"kind": "goal_state", "goal_id": goal_id, "state": goal.state, "error": goal.error}
            )

    async def _advance(self, goal: Goal) -> None:
        if goal.state not in {"pending", "running"}:
            return
        if time.monotonic() - goal.submitted_at > goal.timeout_seconds:
            await self._cancel(goal, "deadline")
            return
        if goal.goal_id in self.effects:
            return
        dependencies = [self.goals[d].state for d in goal.dependencies]
        if any(s in {"failed", "cancelled", "blocked"} for s in dependencies):
            goal.state, goal.error = "blocked", "dependency_failed"
            return
        if any(s != "succeeded" for s in dependencies):
            return
        if goal.verify(self.runtime):
            goal.state = "succeeded"
            return
        if goal.steps >= goal.max_steps:
            goal.state, goal.error = "failed", "step_budget_exhausted"
            return
        goal.state = "running"
        observation = goal.observe(self.runtime, self._last, tuple(goal.history[-32:]))
        if observation.goal_id != goal.goal_id:
            raise ValueError("observation factory returned the wrong goal")
        decision = None
        if goal.active_plan and self.planner:
            decision = self.planner.poll(goal.active_plan, observation)
            status = self.planner.status(goal.active_plan)["status"]
            if status != "pending":
                await self._emit(
                    {
                        "kind": "planner_arrival",
                        "goal_id": goal.goal_id,
                        **self.planner.status(goal.active_plan),
                    }
                )
                goal.active_plan = None
        if decision is None:
            if goal.goal_id not in self.inferences:

                def predict():
                    policy = self._goal_policies.get(goal.goal_id, self.policy)
                    lock = self._goal_policy_locks.get(goal.goal_id, self._policy_lock)
                    with lock:
                        return policy.predict(observation)

                self.inferences[goal.goal_id] = (
                    asyncio.create_task(asyncio.to_thread(predict)),
                    observation,
                )
                return
            task, submitted = self.inferences[goal.goal_id]
            if not task.done():
                return
            del self.inferences[goal.goal_id]
            if AsyncDeliberator._binding(submitted) != AsyncDeliberator._binding(observation):
                await self._emit({"kind": "stale_policy_result", "goal_id": goal.goal_id})
                return
            decision = task.result()
        if decision.mode == "delegate":
            if self.planner and not goal.active_plan:
                request = uuid4().hex
                try:
                    self.planner.submit(request, observation, observation.deadline_remaining_ms)
                    goal.active_plan = request
                except RuntimeError:
                    await self._emit({"kind": "planner_backpressure", "goal_id": goal.goal_id})
            elif not self.planner:
                goal.state, goal.error = "blocked", "planner_required"
            return
        if decision.mode == "stop":
            goal.state, goal.error = "failed", "policy_stopped_without_verified_success"
            return
        if decision.mode in {"wait", "observe"}:
            return
        if len(self.effects) >= self.max_effects:
            return
        intent_id = uuid4().hex
        goal.steps += 1
        goal.active_intent = intent_id
        goal.history.append(
            {
                "kind": "decision",
                "decision": decision.to_dict(),
                "event_id": observation.event_id,
                "revision": observation.revision,
            }
        )
        # execute() repeats native authority/version checks immediately before dispatch.
        self.effects[goal.goal_id] = asyncio.create_task(
            asyncio.to_thread(
                self.runtime.execute,
                observation,
                decision,
                intent_id=intent_id,
                ttl_ms=max(1, int(observation.deadline_remaining_ms)),
                cancel_event=goal.cancellation,
            )
        )
        await self._emit(
            {
                "kind": "decision",
                "goal_id": goal.goal_id,
                "intent_id": intent_id,
                "decision": decision.to_dict(),
            }
        )

    async def run(self, *, until_idle: bool = False) -> dict[str, str]:
        try:
            while not self._stop:
                # Separate urgent lane is drained before normal work.
                while not self.control.empty():
                    await self._control_event(self.control.get_nowait())
                if not self.normal.empty():
                    self._last = self.normal.get_nowait()
                    await self._emit(
                        {
                            "kind": "event",
                            "event_id": self._last.event_id,
                            "event_kind": self._last.kind,
                        }
                    )
                await self._finish_effects()
                for goal in self.goals.values():
                    await self._advance(goal)
                if (
                    until_idle
                    and not self.effects
                    and all(
                        g.state not in {"pending", "running", "cancelling"}
                        for g in self.goals.values()
                    )
                ):
                    break
                await asyncio.sleep(self.tick_seconds)
        finally:
            for goal in self.goals.values():
                if goal.active_intent:
                    await self._cancel(goal, "controller_exit")
            if self.effects:
                await asyncio.gather(*self.effects.values(), return_exceptions=True)
                await self._finish_effects()
            if self.inferences:
                await asyncio.gather(
                    *(t for t, _ in self.inferences.values()), return_exceptions=True
                )
            if self.planner:
                await asyncio.to_thread(self.planner.close)
            if self._goal_policies:
                await asyncio.gather(
                    *(
                        asyncio.to_thread(policy.close)
                        for policy in self._goal_policies.values()
                        if hasattr(policy, "close")
                    )
                )
        return {key: goal.state for key, goal in self.goals.items()}


async def watch_files(
    controller: Controller,
    paths: tuple[str, ...],
    *,
    interval: float = 0.25,
    stop: asyncio.Event | None = None,
) -> None:
    """Bounded polling sensor using the same canonical workspace boundary as tools."""
    from .effectors import WorkspaceFiles

    files = WorkspaceFiles(controller.runtime.workspace)
    names = tuple(files.canonical(path) for path in paths)
    if not 0.01 <= interval <= 3600 or len(names) > 4096:
        raise ValueError("watcher interval/path bounds exceeded")
    seen: dict[str, str] = {}
    while not controller._stop and not (stop and stop.is_set()):
        for name in names:
            fingerprint = await asyncio.to_thread(files.fingerprint, "file:" + name)
            if seen.get(name) != fingerprint:
                controller.publish(
                    Event("file_changed", {"path": name, "fingerprint": fingerprint})
                )
                seen[name] = fingerprint
        await asyncio.sleep(interval)
