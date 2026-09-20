import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from reflexmesh.effectors import ProcessTemplate, register_process
from reflexmesh.processes import worker_python
from reflexmesh.runtime import Runtime
from reflexmesh.server import create_app
from reflexmesh.workspace_service import WorkspaceManager


def interrupted_process(root):
    template = ProcessTemplate(
        "write_then_wait",
        (
            worker_python(),
            "-c",
            "from pathlib import Path; import time; Path('partial.txt').write_text('partial'); time.sleep(60)",
        ),
    )
    with Runtime(root, ("file.read", "file.write", "process.write_then_wait")) as runtime:
        register_process(runtime, template)
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(
                runtime.execute_candidate,
                runtime.candidate("process.write_then_wait", {}),
                intent_id="interrupted",
            )
            deadline = time.monotonic() + 10
            while not (root / "partial.txt").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            runtime.cancel("interrupted")
            assert future.result(timeout=5)["status"] == "effect_unknown"
    assert (root / "partial.txt").read_text() == "partial"
    return template


def request_for(record):
    return {
        "intent_id": record["intent_id"],
        "record_sha256": record["record_sha256"],
        "resolution": "failed_verified",
        "checks": [{"path": "partial.txt", "sha256": hashlib.sha256(b"partial").hexdigest()}],
        "reviewed_effects": True,
        "processes_quiescent": True,
        "note": "Inspected partial output and verified the worker tree is stopped.",
    }


def test_recovery_requires_review_and_actual_evidence_then_releases_lease(tmp_path):
    template = interrupted_process(tmp_path)
    token = "test-recovery-token-more-than-24-characters"
    app = create_app(
        checkpoints=tmp_path, workspace=tmp_path, process_templates=(template,), token=token
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/workspace/recovery").status_code == 401
        client.headers["Authorization"] = "Bearer " + token
        record = client.get("/api/workspace/recovery").json()["unknown"][0]
        assert record["recoverable_in_workbench"]
        request = request_for(record)
        for invalid in [
            {**request, "reviewed_effects": False},
            {**request, "record_sha256": "0" * 64},
            {**request, "checks": [{"path": "partial.txt", "absent": True}]},
            {**request, "checks": [{"path": "../escape", "absent": True}]},
        ]:
            assert client.post("/api/workspace/reconcile", json=invalid).status_code == 422
            assert len(client.get("/api/workspace/recovery").json()["unknown"]) == 1
        response = client.post("/api/workspace/reconcile", json=request)
        assert response.status_code == 200, response.text
        assert response.json()["result"]["status"] == "failed_verified"
        assert client.get("/api/workspace/recovery").json()["unknown"] == []
        assert client.post("/api/workspace/reconcile", json=request).status_code == 422
    # The old lease has gone. This new intent is admitted then cancelled before
    # dispatch, without redoing the uncertain write.
    with Runtime(tmp_path, ("file.read", "file.write", "process.write_then_wait")) as runtime:
        register_process(runtime, template)
        cancel = threading.Event()
        cancel.set()
        result = runtime.execute_candidate(
            runtime.candidate("process.write_then_wait", {}), cancel_event=cancel
        )
        assert result["status"] == "cancelled" and result["started_ms"] is None
        assert (tmp_path / "partial.txt").read_text() == "partial"


def test_recovery_is_exclusive_and_rejects_changed_process_registration(tmp_path):
    template = interrupted_process(tmp_path)
    manager = WorkspaceManager(tmp_path, (template,))
    assert manager._slots.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="busy"):
            manager.recovery()
    finally:
        manager._slots.release()
    changed = WorkspaceManager(
        tmp_path, (ProcessTemplate(template.name, (worker_python(), "-c", "print('different')")),)
    )
    try:
        record = changed.recovery()["unknown"][0]
        assert not record["recoverable_in_workbench"]
        with pytest.raises(ValueError, match="external verification"):
            changed.reconcile(request_for(record))
    finally:
        manager.close()
        changed.close()
