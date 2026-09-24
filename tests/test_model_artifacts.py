import hashlib
import zipfile

import pytest
from reflexmesh.artifacts import install_models


def test_model_install_checks_all_bytes_before_replacing_existing_files(tmp_path):
    archive = tmp_path / "models.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("M03.native.json", b"new")
        bundle.writestr("M04.native.json", b"wrong")
    manifest = {
        "schema_version": 1,
        "kind": "reflexmesh-models",
        "version": "v0.01.000",
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files": [
            {"name": "M03.native.json", "size": 3, "sha256": hashlib.sha256(b"new").hexdigest()},
            {"name": "M04.native.json", "size": 5, "sha256": "0" * 64},
        ],
    }
    destination = tmp_path / "installed"
    destination.mkdir()
    (destination / "M03.native.json").write_bytes(b"old")
    with pytest.raises(ValueError, match="checkpoint digest"):
        install_models(archive, manifest, destination)
    assert (destination / "M03.native.json").read_bytes() == b"old"
    manifest["files"][1]["sha256"] = hashlib.sha256(b"wrong").hexdigest()
    assert install_models(archive, manifest, destination)["verified"]
    assert (destination / "M03.native.json").read_bytes() == b"new"


def test_model_manifest_cannot_extract_outside_destination(tmp_path):
    manifest = {"schema_version": 1, "kind": "reflexmesh-models", "files": [{"name": "../escape"}]}
    with pytest.raises(ValueError, match="flat filenames"):
        install_models(tmp_path / "unused.zip", manifest, tmp_path / "models")
