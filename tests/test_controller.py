import asyncio
import threading

import pytest
from reflexmesh.contracts import Decision, Observation
from reflexmesh.controller import Controller, Event, Goal
from reflexmesh.effectors import WorkspaceFiles
from reflexmesh.runtime import EffectReceipt, Runtime, ToolSpec


class First:
    policy_id = "first"

    def predict(self, observation):
        return Decision("first", observation.admissible[0].action_id)

    def reset(self, goal_id=None):
        pass


def write_goal(name, path, content, dependencies=()):
    def observe(runtime, event, history):
        candidate = runtime.candidate(
            "file.write", {"path": path, "content": content}, action_id="write"
        )
        return Observation(
            event.event_id,
            name,
            sum(candidate.resource_versions.values()),
            event.kind,
            {"goal": name},
            (candidate,),
            ("file.write",),
        )

    return Goal(
        name,
        observe,
        lambda r: (r.workspace / path).exists() and (r.workspace / path).read_text() == content,
        dependencies,
    )


@pytest.mark.asyncio
async def test_dependency_goals_execute_actual_files(tmp_path):
    with Runtime(tmp_path, ("file.write",)) as runtime:
        WorkspaceFiles(tmp_path).register(runtime)
        controller = Controller(runtime, First())
        controller.submit(write_goal("first", "a.txt", "A"))
        controller.submit(write_goal("second", "b.txt", "B", ("first",)))
        result = await controller.run(until_idle=True)
        assert result == {"first": "succeeded", "second": "succeeded"}
        assert (tmp_path / "a.txt").read_text() == "A"
        assert (tmp_path / "b.txt").read_text() == "B"


@pytest.mark.asyncio
async def test_control_event_interrupts_running_effect(tmp_path):
    started = threading.Event()

    def effect(args, cancelled):
        started.set()
        assert cancelled.wait(5)
        return EffectReceipt("cancelled_verified", {"no_effect": True})

    with Runtime(tmp_path, ("work",)) as runtime:
        runtime.register(
            ToolSpec("work", ("work",), lambda a: a, lambda a: {}, effect, lambda _: "x")
        )

        def observe(runtime, event, history):
            candidate = runtime.candidate("work", {}, action_id="work")
            return Observation(event.event_id, "g", 0, event.kind, {}, (candidate,), ("work",))

        controller = Controller(runtime, First())
        controller.submit(Goal("g", observe, lambda _: False))
        task = asyncio.create_task(controller.run(until_idle=True))
        assert await asyncio.to_thread(started.wait, 3)
        controller.publish(Event("cancel_goal", {"goal_id": "g"}))
        assert await asyncio.wait_for(task, 3) == {"g": "cancelled"}


@pytest.mark.asyncio
async def test_urgent_cancel_remains_responsive_during_slow_policy(tmp_path):
    entered, release, cancellation_seen = threading.Event(), threading.Event(), asyncio.Event()

    class Slow(First):
        def predict(self, observation):
            entered.set()
            assert release.wait(5)
            return super().predict(observation)

    with Runtime(tmp_path, ("file.write",)) as runtime:
        WorkspaceFiles(tmp_path).register(runtime)
        controller = Controller(runtime, Slow())
        controller.submit(write_goal("g", "never.txt", "never"))

        def listener(event):
            if event["kind"] == "goal_cancel":
                cancellation_seen.set()

        controller.subscribe(listener)
        task = asyncio.create_task(controller.run(until_idle=True))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            controller.publish(Event("cancel_goal", {"goal_id": "g"}))
            await asyncio.wait_for(cancellation_seen.wait(), 1)
            assert not (tmp_path / "never.txt").exists()
        finally:
            release.set()
            await asyncio.wait_for(task, 3)
