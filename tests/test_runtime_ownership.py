"""One live native state owner per durable SQLite journal across processes."""

import os
import sqlite3
import time
from pathlib import Path

import pytest
import reflexmesh
from reflexmesh.processes import spawn, worker_python
from reflexmesh.runtime import AdmissionError, DurableJournal, Runtime


def test_second_runtime_and_direct_journal_cannot_share_live_owner(tmp_path):
    with Runtime(tmp_path, ("work",)) as owner:
        original = owner.store.load()
        with pytest.raises(AdmissionError, match="already owned"):
            Runtime(tmp_path, ("escalated",))
        with pytest.raises(AdmissionError, match="already owned"):
            DurableJournal(owner.store.path)
        assert owner.store.load() == original
        assert owner.snapshot()["state"]["capabilities"] == ["work"]
    with Runtime(tmp_path, ("work",)) as reopened:
        assert reopened.snapshot()["state"]["capabilities"] == ["work"]


def test_second_process_is_denied_before_reading_or_mutating_journal(tmp_path):
    code = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from reflexmesh.runtime import AdmissionError, Runtime
try:
    runtime = Runtime(sys.argv[2], ('unexpected',))
except AdmissionError as error:
    assert 'already owned' in str(error), str(error)
    sys.exit(17)
runtime.close()
Path(sys.argv[2], 'incorrectly-admitted').touch()
sys.exit(3)
"""
    package = str(Path(reflexmesh.__file__).resolve().parent.parent)
    with Runtime(tmp_path, ("work",)) as owner, (tmp_path / "stderr").open("wb") as stderr:
        before = owner.store.load()
        child = spawn(
            [worker_python(), "-I", "-c", code, package, str(tmp_path)],
            cwd=tmp_path,
            stderr=stderr,
        )
        try:
            status = child.wait(timeout=10)
        finally:
            child.close()
        assert status == 17, (tmp_path / "stderr").read_text(errors="replace")
        assert not (tmp_path / "incorrectly-admitted").exists()
        assert owner.store.load() == before


def test_bad_workspace_identity_constructor_releases_owner_lock(tmp_path):
    with Runtime(tmp_path, (), workspace_id="original"):
        pass
    for _ in range(2):
        with pytest.raises(AdmissionError, match="different workspace"):
            Runtime(tmp_path, (), workspace_id="changed")
    with Runtime(tmp_path, (), workspace_id="original"):
        pass


def test_invalid_checksum_constructor_releases_owner_lock(tmp_path):
    with Runtime(tmp_path, ()) as runtime:
        path = runtime.store.path
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE broker SET digest='corrupted' WHERE id=1")
    for _ in range(2):
        with pytest.raises(AdmissionError, match="checksum"):
            Runtime(tmp_path, ())
    # A direct owner can still acquire the lock to inspect/recover the damage.
    journal = DurableJournal(path)
    journal.close()


def test_sqlite_constructor_failure_releases_owner_lock(tmp_path, monkeypatch):
    import reflexmesh.runtime as module

    with monkeypatch.context() as patch:

        def fail_connect(*args, **kwargs):
            raise sqlite3.OperationalError("injected connect failure")

        patch.setattr(module.sqlite3, "connect", fail_connect)
        with pytest.raises(sqlite3.OperationalError, match="injected"):
            Runtime(tmp_path, ())
    with Runtime(tmp_path, ()):
        pass


def test_close_persistence_failure_releases_owner_lock(tmp_path, monkeypatch):
    runtime = Runtime(tmp_path, ())

    def fail_persist(*args, **kwargs):
        raise OSError("injected persist failure")

    monkeypatch.setattr(runtime, "_persist", fail_persist)
    with pytest.raises(OSError, match="injected"):
        runtime.close()
    assert runtime._closed
    runtime.close()  # Closure is idempotent even after a failed final save.
    with Runtime(tmp_path, ()):
        pass


def test_journal_hardlink_cannot_bypass_canonical_owner(tmp_path):
    with Runtime(tmp_path, ()) as runtime:
        alias = tmp_path / "journal-alias.sqlite3"
        try:
            os.link(runtime.store.path, alias)
        except OSError as error:
            pytest.skip(f"test filesystem does not support hardlinks: {error}")
        with pytest.raises(AdmissionError, match="hardlink"):
            Runtime(tmp_path, (), journal_path=alias)
        alias.unlink()


def test_os_releases_journal_lock_after_abrupt_worker_exit(tmp_path):
    from reflexmesh.environments.crash import CrashRecovery

    with CrashRecovery.create(tmp_path / "crash", value=71) as fixture:
        assert fixture.worker["exit_code"] == 23
        assert fixture.status == "effect_unknown"
        with pytest.raises(AdmissionError, match="already owned"):
            Runtime(fixture.root, ("crash.append",))
        fixture.reconcile()
        assert fixture.verify()


def test_recovery_endpoint_returns_409_while_an_external_process_owns_journal(tmp_path):
    from fastapi.testclient import TestClient
    from reflexmesh.server import create_app

    code = """
import sys,time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from reflexmesh.runtime import Runtime
with Runtime(sys.argv[2], ('file.read', 'file.write')):
    Path(sys.argv[2], 'owner-ready').touch()
    time.sleep(30)
"""
    package = str(Path(reflexmesh.__file__).resolve().parent.parent)
    token = "test-external-owner-token-more-than-24-characters"
    app = create_app(checkpoints=tmp_path, workspace=tmp_path, token=token)
    with (tmp_path / "stderr").open("wb") as stderr:
        child = spawn(
            [worker_python(), "-I", "-c", code, package, str(tmp_path)],
            cwd=tmp_path,
            stderr=stderr,
        )
        try:
            deadline = time.monotonic() + 10
            while not (tmp_path / "owner-ready").exists() and time.monotonic() < deadline:
                assert child.poll() is None, (tmp_path / "stderr").read_text(errors="replace")
                time.sleep(0.01)
            assert (tmp_path / "owner-ready").exists()
            with TestClient(app, base_url="http://127.0.0.1") as client:
                client.headers["Authorization"] = "Bearer " + token
                response = client.get("/api/workspace/recovery")
                assert response.status_code == 409
                assert response.json()["detail"] == "workspace busy or journal unavailable"
                child.kill()
                child.wait(timeout=5)
                reopened = client.get("/api/workspace/recovery")
                assert reopened.status_code == 200, reopened.text
                assert reopened.json()["unknown"] == []
        finally:
            child.close()
