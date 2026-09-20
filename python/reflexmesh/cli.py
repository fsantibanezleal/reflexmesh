"""Reproducible local entry points, without implicit credentials or remote tools."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import secrets
from pathlib import Path


def run_workflow(recipe_path: str, workspace: str) -> dict:
    from .contracts import Decision, Observation
    from .controller import Controller, Goal
    from .effectors import ProcessTemplate, WorkspaceFiles, register_process
    from .runtime import Runtime

    recipe = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
    if set(recipe) - {"schema_version", "steps", "processes"} or recipe.get("schema_version") != 1:
        raise ValueError("workflow requires schema_version 1 and registered steps")
    steps = recipe.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 128:
        raise ValueError("workflow requires 1..128 steps")
    processes = recipe.get("processes", [])
    if not isinstance(processes, list) or len(processes) > 32:
        raise ValueError("workflow supports up to 32 registered process templates")
    capabilities = ("file.read", "file.write", *["process." + p["name"] for p in processes])

    class WorkflowPolicy:
        policy_id = "workflow-fsm"

        def predict(self, obs):
            return Decision(
                self.policy_id, obs.admissible[0].action_id, reason="registered_dependency_ready"
            )

        def reset(self, goal_id=None):
            pass

    with Runtime(workspace, capabilities) as runtime:
        WorkspaceFiles(workspace).register(runtime)
        for process in processes:
            register_process(
                runtime,
                ProcessTemplate(
                    process["name"],
                    tuple(process["argv"]),
                    timeout_seconds=process.get("timeout_seconds", 30),
                ),
            )
        controller = Controller(runtime, WorkflowPolicy())
        for step in steps:
            if set(step) - {"id", "tool", "arguments", "depends_on", "verify", "timeout_seconds"}:
                raise ValueError("unknown workflow step field")
            check = step["verify"]
            if set(check) != {"path", "sha256"} or len(check["sha256"]) != 64:
                raise ValueError("each step needs an independent path+sha256 postcondition")
            files = WorkspaceFiles(workspace)
            verified_path = files.path(check["path"])

            def verify(r, path=verified_path, digest=check["sha256"]):
                return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest

            def observe(r, event, history, step=step):
                candidate = r.candidate(step["tool"], step["arguments"], action_id=step["id"])
                return Observation(
                    event.event_id,
                    step["id"],
                    sum(candidate.resource_versions.values()),
                    event.kind,
                    {"goal": step["id"], "history": list(history[-4:])},
                    (candidate,),
                    capabilities,
                    deadline_remaining_ms=step.get("timeout_seconds", 30) * 1000,
                )

            controller.submit(
                Goal(
                    step["id"],
                    observe,
                    verify,
                    tuple(step.get("depends_on", [])),
                    max_steps=1,
                    timeout_seconds=step.get("timeout_seconds", 30),
                )
            )
        result = asyncio.run(controller.run(until_idle=True))
        return {
            "goals": result,
            "verified_success": all(s == "succeeded" for s in result.values()),
            "journal_sequence": runtime.snapshot()["journal_sequence"],
        }


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="reflexmesh", description="Typed software control and learned metacontrol"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="report installed engine and runtime versions")
    workflow = commands.add_parser("workflow", help="execute an explicit registered JSON workflow")
    workflow.add_argument("recipe")
    workflow.add_argument("--workspace", required=True)
    serve = commands.add_parser("serve", help="serve the authenticated local workbench API")
    serve.add_argument("--checkpoints", required=True)
    serve.add_argument("--artifacts")
    serve.add_argument("--static")
    serve.add_argument("--port", type=int, default=8151)
    serve.add_argument("--model", help="explicit local Ollama model (no implicit provider)")
    serve.add_argument("--token-env", default="REFLEXMESH_LOCAL_TOKEN")
    pipeline = commands.add_parser(
        "pipeline", help="run the scientific training/evaluation command line"
    )
    pipeline.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        from . import _native

        versions = {}
        for name in (
            "reflexmesh",
            "numpy",
            "torch",
            "scikit-learn",
            "xgboost",
            "sb3-contrib",
            "fastapi",
        ):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        print(
            json.dumps(
                {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "native_engine": _native.__engine__,
                    "native_version": _native.__version__,
                    "packages": versions,
                },
                indent=2,
            )
        )
    elif args.command == "workflow":
        result = run_workflow(args.recipe, args.workspace)
        print(json.dumps(result, indent=2))
        return 0 if result["verified_success"] else 1
    elif args.command == "serve":
        import uvicorn

        from .planners import OllamaPlanner
        from .server import create_app

        if not 1024 <= args.port <= 65535:
            parser.error("port must be 1024..65535")
        token = os.environ.get(args.token_env) or secrets.token_urlsafe(32)
        app = create_app(
            checkpoints=args.checkpoints,
            artifacts=args.artifacts,
            static=args.static,
            token=token,
            planner=OllamaPlanner(args.model) if args.model else None,
        )
        print(f"Local URL: http://127.0.0.1:{args.port}")
        print("Local session token (enter only in this local workbench): " + token)
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")
    elif args.command == "pipeline":
        from .pipeline import main as pipeline_main

        return pipeline_main(args.arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
