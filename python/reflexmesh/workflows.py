"""Verified dependency workflows in a host-configured workspace.

Recipes select registered tools. A web recipe cannot add process templates or
choose another root; those capabilities are configured by the local operator.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import threading
from pathlib import Path

from .contracts import Decision, Observation
from .controller import Controller, Event, Goal
from .effectors import ProcessTemplate, WorkspaceFiles, register_process
from .runtime import Runtime


def process_templates(value: list) -> tuple[ProcessTemplate, ...]:
    if not isinstance(value, list) or len(value) > 32:
        raise ValueError("up to 32 process registrations are supported")
    result, names = [], set()
    for item in value:
        if not isinstance(item, dict) or set(item) - {"name", "argv", "timeout_seconds"}:
            raise ValueError("invalid process registration")
        name, argv = item.get("name"), item.get("argv")
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name)
            or name in names
        ):
            raise ValueError("process names must be unique portable identifiers")
        if (
            not isinstance(argv, list)
            or not 1 <= len(argv) <= 64
            or any(not isinstance(a, str) or len(a) > 8192 for a in argv)
        ):
            raise ValueError("process argv must be a bounded string list")
        if not Path(argv[0]).is_absolute():
            raise ValueError("process executable must be absolute")
        timeout = item.get("timeout_seconds", 30)
        if (
            type(timeout) not in {float, int}
            or not math.isfinite(timeout)
            or not 0 < timeout <= 3600
        ):
            raise ValueError("process timeout must be finite and in (0,3600]")
        names.add(name)
        result.append(ProcessTemplate(name, tuple(argv), float(timeout)))
    return tuple(result)


def validate_recipe(recipe: dict, files: WorkspaceFiles, templates=()) -> list[dict]:
    # JSON roundtrip both rejects nonfinite data and detaches caller-owned maps.
    recipe = json.loads(json.dumps(recipe, allow_nan=False))
    if (
        not isinstance(recipe, dict)
        or set(recipe) != {"schema_version", "steps"}
        or recipe["schema_version"] != 1
    ):
        raise ValueError("workflow requires only schema_version 1 and steps")
    steps = recipe["steps"]
    if not isinstance(steps, list) or not 1 <= len(steps) <= 128:
        raise ValueError("workflow requires 1..128 steps")
    names, ordered = {}, []
    tools = {"file.read", "file.write", *["process." + p.name for p in templates]}
    for step in steps:
        if not isinstance(step, dict) or set(step) - {
            "id",
            "tool",
            "arguments",
            "depends_on",
            "verify",
            "timeout_seconds",
        }:
            raise ValueError("invalid workflow step fields")
        name = step.get("id")
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name)
            or name in names
        ):
            raise ValueError("step IDs must be unique portable identifiers")
        if step.get("tool") not in tools or not isinstance(step.get("arguments"), dict):
            raise ValueError("workflow selects an unregistered tool")
        args = step["arguments"]
        expected = (
            {"path", "content"}
            if step["tool"] == "file.write"
            else {"path"}
            if step["tool"] == "file.read"
            else set()
        )
        if set(args) != expected:
            raise ValueError("workflow tool arguments do not match its registered schema")
        if "path" in args:
            files.path(args["path"])
        if "content" in args and (
            not isinstance(args["content"], str) or len(args["content"].encode()) > files.max_bytes
        ):
            raise ValueError("workflow content exceeds the file capability")
        check = step.get("verify")
        if (
            not isinstance(check, dict)
            or set(check) != {"path", "sha256"}
            or not isinstance(check["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", check["sha256"])
        ):
            raise ValueError(
                "each step requires an independent relative path and SHA256 postcondition"
            )
        files.path(check["path"])
        dependencies = step.setdefault("depends_on", [])
        if (
            not isinstance(dependencies, list)
            or any(not isinstance(d, str) for d in dependencies)
            or len(set(dependencies)) != len(dependencies)
        ):
            raise ValueError("dependencies must be unique step IDs")
        timeout = step.setdefault("timeout_seconds", 30)
        if (
            type(timeout) not in {float, int}
            or not math.isfinite(timeout)
            or not 0 < timeout <= 3600
        ):
            raise ValueError("step timeout must be finite and in (0,3600]")
        names[name] = step
    remaining = dict(names)
    while remaining:
        ready = [
            s
            for s in remaining.values()
            if all(d in {r["id"] for r in ordered} for d in s["depends_on"])
        ]
        if not ready:
            raise ValueError("workflow has a cycle or an unknown dependency")
        for step in ready:
            ordered.append(step)
            del remaining[step["id"]]
    return ordered


class DependencyPolicy:
    policy_id = "verified-workflow-fsm"

    def predict(self, observation):
        return Decision(
            self.policy_id,
            observation.admissible[0].action_id,
            reason="registered_dependency_ready",
        )

    def reset(self, goal_id=None):
        pass


def execute_workflow(
    recipe: dict, workspace: str | Path, *, templates=(), cancel_event=None, on_event=None
) -> dict:
    files = WorkspaceFiles(workspace)
    steps = validate_recipe(recipe, files, templates)
    capabilities = ("file.read", "file.write", *["process." + p.name for p in templates])
    cancelled = cancel_event if cancel_event is not None else threading.Event()
    with Runtime(workspace, capabilities) as runtime:
        files.register(runtime)
        for template in templates:
            register_process(runtime, template)
        controller = Controller(runtime, DependencyPolicy())
        if on_event:
            controller.subscribe(on_event)
        for step in steps:

            def verify(_runtime, check=step["verify"]):
                return (
                    files.fingerprint("file:" + files.canonical(check["path"]))
                    == "sha256:" + check["sha256"]
                )

            def observe(_runtime, event, history, step=step):
                candidate = runtime.candidate(step["tool"], step["arguments"], action_id=step["id"])
                return Observation(
                    event.event_id,
                    step["id"],
                    sum(candidate.resource_versions.values()),
                    event.kind,
                    {"goal": step["id"]},
                    (candidate,),
                    capabilities,
                    deadline_remaining_ms=step["timeout_seconds"] * 1000,
                )

            controller.submit(
                Goal(
                    step["id"],
                    observe,
                    verify,
                    tuple(step["depends_on"]),
                    max_steps=1,
                    timeout_seconds=step["timeout_seconds"],
                )
            )

        async def run():
            task = asyncio.create_task(controller.run(until_idle=True))
            while not task.done():
                if cancelled.is_set():
                    controller.publish(Event("shutdown"))
                    break
                await asyncio.sleep(0.01)
            return await task

        goals = asyncio.run(run())
        snapshot = runtime.snapshot()
        return {
            "schema_version": 1,
            "lane": "configured-workspace",
            "goals": goals,
            "verified_success": all(state == "succeeded" for state in goals.values()),
            "cancelled": cancelled.is_set(),
            "journal_sequence": snapshot["journal_sequence"],
            "goal_evidence": {
                name: {"state": goal.state, "error": goal.error, "history": goal.history}
                for name, goal in controller.goals.items()
            },
            "recipe_sha256": hashlib.sha256(
                json.dumps(recipe, sort_keys=True).encode()
            ).hexdigest(),
        }
