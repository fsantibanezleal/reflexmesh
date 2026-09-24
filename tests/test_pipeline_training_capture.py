import json
from types import SimpleNamespace

import pytest
from reflexmesh.learning.provenance import TrainingCapture


def installation(source="one"):
    return {
        "source_sha256": {"source.py": source},
        "native_source_sha256": {"rust/lib.rs": "two"},
        "native_binary": {"sha256": "three"},
        "packages": {"numpy": "version"},
        "checkpoint_sha256": {},
    }


def test_training_capture_binds_inputs_and_retains_failed_attempt(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/input.json").write_text("before")
    args = SimpleNamespace(artifacts=tmp_path, stage="train", model="fixture")
    monkeypatch.setattr(TrainingCapture, "_lineage", lambda self: installation())
    with pytest.raises(ValueError, match="fit failed"), TrainingCapture(args):
        raise ValueError("fit failed")
    failed = json.loads((tmp_path / "training-runs/train.json").read_text())
    assert failed["status"] == "failed" and failed["source_unchanged"]
    with TrainingCapture(args):
        (tmp_path / "data/input.json").write_text("after")
    complete = json.loads((tmp_path / "training-runs/train.json").read_text())
    assert complete["status"] == "complete"
    assert complete["data_sha256_before"] != complete["data_sha256_after"]
    assert len(list((tmp_path / "training-runs").glob("train.prior-*.json"))) == 1


def test_training_capture_rejects_source_change_during_fitting(tmp_path, monkeypatch):
    snapshots = iter([installation("before"), installation("after")])
    monkeypatch.setattr(TrainingCapture, "_lineage", lambda self: next(snapshots))
    args = SimpleNamespace(artifacts=tmp_path, stage="ppo", model="fixture")
    with pytest.raises(RuntimeError, match="installation changed"), TrainingCapture(args):
        pass
    record = json.loads((tmp_path / "training-runs/ppo.json").read_text())
    assert record["source_unchanged"] is False
