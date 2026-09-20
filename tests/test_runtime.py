import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from reflexmesh.contracts import Candidate
from reflexmesh.effectors import ProcessTemplate, WorkspaceFiles, register_process
from reflexmesh.processes import worker_python
from reflexmesh.runtime import AdmissionError, EffectReceipt, Runtime, ToolSpec


def runtime_files(tmp_path):
    runtime = Runtime(tmp_path, ("file.read", "file.write"))
    WorkspaceFiles(tmp_path).register(runtime)
    return runtime


def test_write_read_replay_and_recover(tmp_path):
    with runtime_files(tmp_path) as runtime:
        result = runtime.execute_candidate(
            runtime.candidate("file.write", {"path": "report.txt", "content": "verified"}),
            intent_id="write1",
        )
        assert result["status"] == "completed_verified"
        read = runtime.execute_candidate(runtime.candidate("file.read", {"path": "report.txt"}))
        assert read["outcome"]["evidence"]["content"] == "verified"
        assert runtime.replay()["state"] == runtime.snapshot()["state"]
    with runtime_files(tmp_path) as recovered:
        assert recovered.snapshot()["state"]["intents"]["write1"]["status"] == "completed_verified"
        assert (tmp_path / "report.txt").read_text() == "verified"


def test_external_change_invalidates_candidate_before_write(tmp_path):
    path = tmp_path / "shared.txt"
    path.write_text("first")
    with runtime_files(tmp_path) as runtime:
        candidate = runtime.candidate("file.write", {"path": "shared.txt", "content": "overwrite"})
        path.write_text("concurrent")
        with pytest.raises(RuntimeError, match="stale|revision"):
            runtime.execute_candidate(candidate)
        assert path.read_text() == "concurrent"


def test_forged_resources_and_capability_reject(tmp_path):
    with Runtime(tmp_path, ("file.read",)) as runtime:
        WorkspaceFiles(tmp_path).register(runtime)
        forged = Candidate("c", "file.write", {"path": "p.txt", "content": "x"})
        with pytest.raises(AdmissionError, match="incomplete"):
            runtime.execute_candidate(forged)
        valid_binding = runtime.candidate("file.write", {"path": "p.txt", "content": "x"})
        with pytest.raises(RuntimeError, match="capability"):
            runtime.execute_candidate(valid_binding)
        assert not (tmp_path / "p.txt").exists()


@pytest.mark.parametrize(
    "path",
    [
        "../outside",
        "/absolute",
        "C:/Windows/file",
        "folder\\file",
        ".reflexmesh/journal.sqlite3",
        "CON.txt",
        "trailing. ",
        ".",
    ],
)
def test_path_boundaries(tmp_path, path):
    with pytest.raises(AdmissionError):
        WorkspaceFiles(tmp_path).path(path)


def test_unknown_effect_keeps_lease_and_explicit_reconciliation(tmp_path):
    started, release = threading.Event(), threading.Event()
    state = {"value": 0}

    def effect(arguments, cancelled):
        started.set()
        assert release.wait(5)
        state["value"] += 1
        raise TimeoutError("receipt lost after effect")

    spec = ToolSpec(
        "counter",
        ("counter",),
        lambda a: a,
        lambda a: {"counter": "write"},
        effect,
        lambda _: str(state["value"]),
        True,
    )
    with Runtime(tmp_path, ("counter",)) as runtime, ThreadPoolExecutor(1) as pool:
        runtime.register(spec)
        candidate = runtime.candidate("counter", {})
        future = pool.submit(runtime.execute_candidate, candidate, intent_id="i")
        assert started.wait(5)
        runtime.cancel("i")
        release.set()
        assert future.result()["status"] == "effect_unknown"
        with pytest.raises(RuntimeError, match="conflict|lease|locked"):
            runtime.execute_candidate(runtime.candidate("counter", {}))
        reconciled = runtime.reconcile(
            "i", EffectReceipt("completed_verified", {"counter": state["value"]}, ("counter",))
        )
        assert reconciled["status"] == "completed_verified"
        assert state["value"] == 1


def test_process_is_real_and_argv_cannot_be_changed(tmp_path):
    with Runtime(tmp_path, ("process.check",)) as runtime:
        register_process(runtime, ProcessTemplate("check", (worker_python(), "-c", "print(17*23)")))
        with pytest.raises(AdmissionError):
            runtime.candidate("process.check", {"argv": ["something_else"]})
        result = runtime.execute_candidate(runtime.candidate("process.check", {}))
        assert result["status"] == "completed_verified"
        assert result["outcome"]["evidence"]["stdout"].strip() == "391"


def test_corrupt_journal_fails_closed(tmp_path):
    with runtime_files(tmp_path):
        pass
    db = sqlite3.connect(tmp_path / ".reflexmesh" / "journal.sqlite3")
    db.execute("UPDATE broker SET payload='{}'")
    db.commit()
    db.close()
    with pytest.raises(AdmissionError, match="checksum"):
        Runtime(tmp_path, ())


def test_unchanged_observation_preserves_version(tmp_path):
    (tmp_path / "a").write_text("same")
    with runtime_files(tmp_path) as runtime:
        first = runtime.candidate("file.read", {"path": "a"})
        second = runtime.candidate("file.read", {"path": "a"})
        assert first.resource_versions == second.resource_versions


def test_path_aliases_share_resource_and_hardlinks_reject(tmp_path):
    import os

    (tmp_path / "file").write_text("data")
    with runtime_files(tmp_path) as runtime:
        first = runtime.candidate("file.read", {"path": "./file"})
        second = runtime.candidate("file.read", {"path": "file"})
        assert first.resource_versions == second.resource_versions
        if os.name == "nt":
            third = runtime.candidate("file.read", {"path": "FILE"})
            assert first.resource_versions == third.resource_versions
        os.link(tmp_path / "file", tmp_path / "alias")
        with pytest.raises(AdmissionError, match="hardlink"):
            runtime.candidate("file.read", {"path": "alias"})


def test_revoke_signals_running_worker(tmp_path):
    started = threading.Event()

    def work(args, cancelled):
        started.set()
        assert cancelled.wait(5)
        return EffectReceipt("cancelled_verified", {"effect_dispatched": False})

    with Runtime(tmp_path, ("work",)) as runtime, ThreadPoolExecutor(1) as pool:
        runtime.register(
            ToolSpec("work", ("work",), lambda a: a, lambda a: {}, work, lambda _: "0")
        )
        future = pool.submit(runtime.execute_candidate, runtime.candidate("work", {}))
        assert started.wait(5)
        runtime.revoke("work")
        assert future.result()["status"] == "cancelled"


def test_checkpoint_preserves_audit_and_recovers(tmp_path):
    with runtime_files(tmp_path) as runtime:
        for i in range(260):
            runtime.candidate("file.read", {"path": "file"})
        journal = json.loads(runtime.broker.journal())
        assert journal["checkpoint_sequence"] > 0
        assert len(journal["entries"]) < 256
        assert runtime.store.connection.execute("SELECT COUNT(*) FROM archives").fetchone()[0] >= 1
        assert runtime.replay()["state"] == runtime.snapshot()["state"]


def test_real_process_crash_after_effect_requires_reconciliation(tmp_path):
    from pathlib import Path

    from reflexmesh.processes import spawn

    source = str(Path(__file__).resolve().parents[1] / "python")
    code = """
import os,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from reflexmesh.runtime import Runtime,ToolSpec
root=Path(sys.argv[2])
def effect(arguments,cancelled):
    (root/'committed.txt').write_text('one effect')
    os._exit(23)
runtime=Runtime(root,('crash-test',))
runtime.register(ToolSpec('crash-test',('crash-test',),lambda a:a,
    lambda a:{'effect':'write'},effect,
    lambda r:(root/'committed.txt').read_text() if (root/'committed.txt').exists() else 'absent',True))
runtime.execute_candidate(runtime.candidate('crash-test',{}),intent_id='crashed-intent')
"""
    child = spawn([worker_python(), "-c", code, source, str(tmp_path)], cwd=tmp_path)
    try:
        assert child.wait(timeout=10) == 23
    finally:
        child.close()
    assert (tmp_path / "committed.txt").read_text() == "one effect"
    with Runtime(tmp_path, ("crash-test",)) as recovered:
        assert (
            recovered.snapshot()["state"]["intents"]["crashed-intent"]["status"] == "effect_unknown"
        )
        recovered.register(
            ToolSpec(
                "crash-test",
                ("crash-test",),
                lambda a: a,
                lambda a: {"effect": "write"},
                lambda a, c: (_ for _ in ()).throw(AssertionError("must not reexecute")),
                lambda r: (tmp_path / "committed.txt").read_text(),
                True,
            )
        )
        result = recovered.reconcile(
            "crashed-intent",
            EffectReceipt(
                "completed_verified",
                {"actual_content": (tmp_path / "committed.txt").read_text()},
                ("effect",),
            ),
        )
        assert result["status"] == "completed_verified"
