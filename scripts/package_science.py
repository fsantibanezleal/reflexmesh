"""Bundle only owned corrected evidence, with deterministic members and byte hashes."""

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def main():
    audit = json.loads((ARTIFACTS / "corrected-science-audit.json").read_text())
    if audit["status"] != "validated" or audit["completed_episodes"] != 19488:
        raise ValueError("Complete audited corrected evidence is required")
    directories = ["evaluation", "structural-transfer", "ablations", "data", "training-runs"]
    files = [
        path for name in directories for path in (ARTIFACTS / name).rglob("*") if path.is_file()
    ]
    files.extend(
        [
            ARTIFACTS / "corrected-science-audit.json",
            ARTIFACTS / "native-corrected-wheel-parity.json",
        ]
    )
    if sum(path.parent.name == "episodes" for path in files) != 19488:
        raise ValueError("Unexpected episode count")
    output = ARTIFACTS / "releases" / "v0.01.000"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "kind": "reflexmesh-owned-science",
        "episodes": 19488,
        "external_upstream_raw_data_included": False,
        "files": [],
    }
    archive = output / "corrected-science.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(files):
            if path.is_symlink() or not path.resolve().is_relative_to(ARTIFACTS.resolve()):
                raise ValueError("Unexpected evidence path")
            raw = path.read_bytes()
            name = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, raw)
            manifest["files"].append(
                {"path": name, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            )
    manifest["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output / "science-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print(
        json.dumps(
            {
                "archive": str(archive),
                "bytes": archive.stat().st_size,
                "files": len(files),
                "sha256": manifest["archive_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
