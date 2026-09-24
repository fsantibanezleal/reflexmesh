"""Create deterministic checkpoint assets; does not fit models or change evidence."""

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path


def build_models(checkpoints: Path, output: Path, version: str):
    if not re.fullmatch(r"v\d+\.\d{2}\.\d{3}", version):
        raise ValueError("invalid display release version")
    required = {
        "M03.linear.json",
        "M03.joblib",
        "M04.trees.json",
        "M04.joblib",
        "M04.xgboost.json",
        "M05.onnx",
        "M05.pt",
        "M05.metadata.json",
        "M06.npz",
        "M07.zip",
        "M08.pt",
        "M11.gate.joblib",
        "M12.gate.joblib",
        "routing-training.json",
        "classical-training.json",
        "recurrent-training.json",
        "ppo-training.json",
        "transition-training.json",
    }
    paths = sorted(p for p in checkpoints.iterdir() if p.is_file() and p.name in required)
    if not required <= {p.name for p in paths}:
        raise ValueError("complete fitted checkpoint set is required")
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "models.zip"
    files = []
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in paths:
            raw = path.read_bytes()
            info = zipfile.ZipInfo(path.name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, raw)
            files.append(
                {"name": path.name, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            )
    manifest = {
        "schema_version": 1,
        "kind": "reflexmesh-models",
        "version": version,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files": files,
        "license": "Apache-2.0",
        "training_domain": "owned resettable software integration environments",
        "planner_weights_included": False,
        "limitations": "Fitted research policies, not a guarantee of arbitrary-workflow success or safety. Load trusted project artifacts only.",
    }
    (output / "models-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", default="v0.01.000")
    args = parser.parse_args()
    manifest = build_models(args.checkpoints, args.output, args.version)
    print(
        json.dumps({"files": len(manifest["files"]), "archive_sha256": manifest["archive_sha256"]})
    )
