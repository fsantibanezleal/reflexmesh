"""Reproducible collection, fitting, frozen evaluation, and artifact export CLI."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict
from pathlib import Path

from .contracts import ModelRequiredError
from .environments import case_matrix
from .policies import METHODS


def make_policy(method_id, checkpoint_dir, planner=None, **options):
    backend = options.pop("backend", "native")
    from .policies.classical import CandidateScorer, LinUCBPolicy
    from .policies.control import (
        AsyncPlannerPolicy,
        DeferralPolicy,
        DirectPlannerPolicy,
        LookaheadPolicy,
        Metacontroller,
    )
    from .policies.recurrent import RecurrentPolicy
    from .policies.rl import MaskedPPOPolicy
    from .policies.rules import BehaviorTreePolicy, GuardedFSMPolicy

    directory = Path(checkpoint_dir)
    if method_id == "M01":
        return GuardedFSMPolicy()
    if method_id == "M02":
        return BehaviorTreePolicy()
    if method_id in {"M03", "M04"}:
        return CandidateScorer.load(directory, method_id, backend=backend)
    if method_id == "M05":
        return RecurrentPolicy.load(directory)
    if method_id == "M06":
        return LinUCBPolicy.load(directory)
    if method_id == "M07":
        return MaskedPPOPolicy.load(directory)
    if method_id == "M08":
        return LookaheadPolicy.load(directory)
    if method_id == "M09":
        return DirectPlannerPolicy(planner)
    if method_id == "M10":
        return AsyncPlannerPolicy(planner)
    if method_id in {"M11", "M12"}:
        import joblib

        payload = joblib.load(directory / (method_id + ".gate.joblib"))
        if method_id == "M11":
            return DeferralPolicy(
                CandidateScorer.load(directory, "M03", backend=backend),
                payload["model"],
                planner,
                payload["threshold"],
            )
        return Metacontroller(
            RecurrentPolicy.load(directory),
            LookaheadPolicy.load(directory),
            payload["model"],
            planner,
            payload["threshold"],
            **options,
        )
    raise ValueError(f"unknown method {method_id}")


def load_policies(checkpoint_dir, planner=None, methods=None):
    return [make_policy(method, checkpoint_dir, planner) for method in (methods or METHODS)]


def lineage(checkpoint_dir, specs, planner=None):
    directory = Path(checkpoint_dir)
    source = Path(__file__).parent
    executed = (
        "contracts.py",
        "features.py",
        "runtime.py",
        "effectors.py",
        "planners.py",
        "processes.py",
        "pipeline.py",
        "policies",
        "learning",
        "environments",
        "evaluation",
    )
    paths = []
    for name in executed:
        item = source / name
        paths.extend(item.rglob("*.py") if item.is_dir() else [item] if item.exists() else [])
    files = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    repository = Path.cwd() if (Path.cwd() / "Cargo.toml").exists() else source.parent.parent
    native = {
        str(p.relative_to(repository)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (list((repository / "rust").rglob("*.rs")) + list(repository.glob("Cargo.*")))
        if p.is_file()
    }
    models = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.glob("*")
        if p.is_file()
    }
    packages = {}
    for name in (
        "numpy",
        "scikit-learn",
        "xgboost",
        "torch",
        "onnxruntime",
        "sb3-contrib",
        "gymnasium",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    planner_info = {"model": getattr(planner, "model_id", None)}
    if planner is not None:
        import httpx

        try:
            tags = httpx.get("http://127.0.0.1:11434/api/tags", timeout=10).json()
            planner_info["registry"] = [
                m for m in tags.get("models", []) if m["name"] == planner.model_id
            ]
        except Exception as exc:
            raise ModelRequiredError("cannot verify configured local planner provenance") from exc
    from . import _native

    native_binary = {
        "name": Path(_native.__file__).name,
        "sha256": hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest(),
    }
    return {
        "source_sha256": files,
        "native_source_sha256": native,
        "native_binary": native_binary,
        "checkpoint_sha256": models,
        "spec_sha256": hashlib.sha256(
            json.dumps([asdict(s) for s in specs], sort_keys=True).encode()
        ).hexdigest(),
        "packages": packages,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "planner": planner_info,
        "inference_device": "CPU for fitted policies; planner device reported by its runtime",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["collect", "train", "ppo", "gates", "evaluate", "ablations", "transfer"]
    )
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--train-seeds", type=int, default=5)
    parser.add_argument("--ppo-steps", type=int, default=16384)
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    data = args.artifacts / "data"
    checkpoints = args.artifacts / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    def progress(value):
        print(json.dumps(value), flush=True)

    if args.stage == "collect":
        from .learning.dataset import assert_disjoint, collect

        manifests = [
            collect(case_matrix(split, count), data, exploration)
            for split, count, exploration in [
                ("train", args.train_seeds, 0.15),
                ("validation", 1, 0.15),
                ("calibration", 1, 0.0),
            ]
        ]
        assert_disjoint(*manifests)
        progress({"stage": "collected", "episodes": [m["episodes"] for m in manifests]})
    elif args.stage == "train":
        from .learning.dataset import assert_disjoint, load_episodes
        from .learning.train import train_classical, train_recurrent, train_transition

        assert_disjoint(
            *[
                json.loads((data / f"{s}.manifest.json").read_text())
                for s in ("train", "validation", "calibration")
            ]
        )
        train = load_episodes(data, "train")
        calibration = load_episodes(data, "calibration")
        validation = load_episodes(data, "validation")
        progress(train_classical(train, calibration, checkpoints))
        progress(train_recurrent(train, calibration, checkpoints))
        progress(train_transition(train, validation, checkpoints))
    elif args.stage == "ppo":
        from .policies.rl import train_ppo

        progress(train_ppo(checkpoints, timesteps=args.ppo_steps))
    else:
        from .planners import OllamaPlanner

        planner = (
            OllamaPlanner(model_id=args.model)
            if args.stage == "gates" or any(m in args.methods for m in ("M09", "M10", "M11", "M12"))
            else None
        )
        if args.stage == "gates":
            from .learning.routing import train_gates

            progress(train_gates(checkpoints, data, planner, progress))
        elif args.stage == "ablations":
            from .evaluation.runner import evaluate

            specs = case_matrix("test", args.seeds)
            for name, options in [
                ("no_memory", {"use_memory": False}),
                ("no_effects", {"use_effects": False}),
                ("no_deferral", {"use_deferral": False}),
                ("no_invalidation", {"use_invalidation": False}),
            ]:
                policy = make_policy("M12", checkpoints, planner, **options)
                try:
                    evaluate(
                        [policy],
                        specs,
                        (args.output or args.artifacts / "ablations") / name,
                        progress,
                        {
                            **lineage(checkpoints, specs, planner),
                            "ablation": options,
                            "gate_calibration_scope": "fixed full-controller gate; not recalibrated after component removal",
                        },
                    )
                finally:
                    policy.close()
        else:
            from .environments import SoftwareEnvironment
            from .evaluation.runner import evaluate

            factory = SoftwareEnvironment
            specs = case_matrix("test", args.seeds)
            if args.stage == "transfer":
                from .environments.structural import StructuralEnvironment, structural_matrix

                specs = structural_matrix()
                factory = StructuralEnvironment
            policies = load_policies(checkpoints, planner, args.methods)
            try:
                result = evaluate(
                    policies,
                    specs,
                    args.output
                    or args.artifacts
                    / ("structural-transfer" if args.stage == "transfer" else "evaluation"),
                    progress,
                    lineage(checkpoints, specs, planner),
                    factory,
                )
                progress(
                    {
                        "stage": "evaluated",
                        "episodes": len(result["episodes"]),
                        "methods": args.methods,
                    }
                )
            finally:
                for policy in policies:
                    if hasattr(policy, "close"):
                        policy.close()


if __name__ == "__main__":
    main()
