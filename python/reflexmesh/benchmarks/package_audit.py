"""Smoke a real installed wheel, native effects and optional trained JSON policies."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path


def audit(checkpoints: Path | None = None, require_base_only: bool = False) -> dict:
    import reflexmesh
    from reflexmesh import _native
    from reflexmesh.effectors import WorkspaceFiles
    from reflexmesh.environments import CaseSpec, SoftwareEnvironment
    from reflexmesh.pipeline import make_policy
    from reflexmesh.runtime import Runtime

    package = Path(reflexmesh.__file__).resolve().parent
    if "site-packages" not in package.parts:
        raise AssertionError("audit must import an installed wheel outside the source tree")
    frameworks = ("torch", "sklearn", "xgboost", "gymnasium", "sb3_contrib", "onnxruntime")
    available = {name: importlib.util.find_spec(name) is not None for name in frameworks}
    if require_base_only and any(available.values()):
        raise AssertionError(
            f"base-only audit environment contains training frameworks: {available}"
        )
    with tempfile.TemporaryDirectory(prefix="reflexmesh-wheel-audit-") as directory:
        root = Path(directory)
        with Runtime(root, ("file.read", "file.write")) as runtime:
            WorkspaceFiles(root).register(runtime)
            receipt = runtime.execute_candidate(
                runtime.candidate(
                    "file.write", {"path": "wheel-proof.txt", "content": "actual native effect"}
                )
            )
            assert receipt["status"] == "completed_verified"
            assert (root / "wheel-proof.txt").read_text() == "actual native effect"
            assert runtime.replay()["state"] == runtime.snapshot()["state"]
    policies = {}
    for method in ("M01", "M02") + (("M03", "M04") if checkpoints else ()):
        policy = make_policy(method, checkpoints or Path("."))
        spec = CaseSpec("C01", "nominal", 82000)
        policy.reset(spec.episode_id)
        decisions = []
        with SoftwareEnvironment(spec) as environment:
            while not environment.terminal and len(decisions) < 10:
                decision = policy.predict(environment.observe())
                decisions.append(
                    {"candidate_id": decision.candidate_id, "diagnostics": decision.diagnostics}
                )
                environment.step(decision)
            success = environment.verify()
            assert success, f"installed {method} did not complete actual C01 document routing"
            if method in {"M03", "M04"}:
                assert all(
                    d["diagnostics"]["inference_backend"] == "native_cached" for d in decisions
                )
        policies[method] = {
            "case": "C01/nominal/82000",
            "actual_outcome_verified": success,
            "decisions": decisions,
        }
    imported_frameworks = [name for name in frameworks if name in sys.modules]
    if imported_frameworks:
        raise AssertionError(f"base serving imported optional frameworks: {imported_frameworks}")
    files = [
        (path.relative_to(package).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(package.rglob("*.py"))
    ]
    extension = Path(_native.__file__)
    return {
        "schema_version": 1,
        "python": sys.version,
        "package_version": reflexmesh.__version__,
        "installed_from_site_packages": True,
        "engine": _native.__engine__,
        "extension": extension.name,
        "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "package_python_manifest_sha256": hashlib.sha256(json.dumps(files).encode()).hexdigest(),
        "optional_frameworks_available": available,
        "optional_frameworks_imported": imported_frameworks,
        "base_only_required": require_base_only,
        "file_effect_and_replay": "passed",
        "policies": policies,
        "limitations": "Installation smoke and one actual routing task; not the release outcome matrix",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--require-base-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.checkpoints, args.require_base_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
