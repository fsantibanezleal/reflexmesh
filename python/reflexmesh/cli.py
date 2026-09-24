"""Reproducible local entry points, without implicit credentials or remote tools."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import secrets
from pathlib import Path


def run_workflow(recipe_path: str, workspace: str) -> dict:
    from .workflows import execute_workflow, process_templates

    path = Path(recipe_path)
    if path.stat().st_size > 2_097_152:
        raise ValueError("workflow recipe exceeds 2MiB")
    recipe = json.loads(path.read_text(encoding="utf-8"))
    templates = process_templates(recipe.pop("processes", []))
    return execute_workflow(recipe, workspace, templates=templates)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="reflexmesh", description="Typed software control and learned metacontrol"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="report installed engine and runtime versions")
    models = commands.add_parser(
        "fetch-models", help="install hash-verified checkpoints from a project release"
    )
    models.add_argument("--destination", required=True)
    models.add_argument("--version", default="v0.01.000")
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
    serve.add_argument("--workspace", help="explicit existing directory for user workflow tools")
    serve.add_argument(
        "--process-config", help="local JSON list of immutable process registrations"
    )
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
    elif args.command == "fetch-models":
        from .artifacts import fetch_models

        print(json.dumps(fetch_models(args.destination, args.version), indent=2))
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
        from .workflows import process_templates

        templates = ()
        if args.process_config:
            if not args.workspace:
                parser.error("--process-config requires --workspace")
            config_path = Path(args.process_config)
            if config_path.stat().st_size > 1_048_576:
                parser.error("process config exceeds 1MiB")
            templates = process_templates(json.loads(config_path.read_text(encoding="utf-8")))
        app = create_app(
            checkpoints=args.checkpoints,
            artifacts=args.artifacts,
            static=args.static,
            token=token,
            planner=OllamaPlanner(args.model) if args.model else None,
            workspace=args.workspace,
            process_templates=templates,
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
