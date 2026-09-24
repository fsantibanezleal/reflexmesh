import time

from fastapi.testclient import TestClient
from reflexmesh.server import create_app

TOKEN = "test-session-token-32-characters-long"
HEADERS = {"Authorization": "Bearer " + TOKEN}


def app(tmp_path):
    return create_app(checkpoints=tmp_path, token=TOKEN, allowed_hosts=("testserver",))


def test_anonymous_read_contract_and_authenticated_effect_boundary(tmp_path):
    with TestClient(app(tmp_path)) as client:
        assert client.get("/api/config").json()["local_auth_required"] is True
        assert (
            client.post("/api/run", json={"case_id": "C01", "method_id": "M01"}).status_code == 401
        )
        assert client.get("/api/events?run_id=x").status_code == 401
        assert (
            client.post(
                "/api/run", headers={**HEADERS, "Origin": "http://evil.example"}, json={}
            ).status_code
            == 403
        )
        assert client.get("/healthz", headers={"Host": "evil.example"}).status_code == 400


def test_actual_native_run_streams_and_finishes(tmp_path):
    with TestClient(app(tmp_path)) as client:
        response = client.post(
            "/api/run",
            headers=HEADERS,
            json={"case_id": "C01", "method_id": "M01", "variant": "nominal", "seed": 40000},
        )
        assert response.status_code == 200, response.text
        run_id = response.json()["run_id"]
        end = time.monotonic() + 15
        while time.monotonic() < end:
            state = client.get("/api/events", params={"run_id": run_id}, headers=HEADERS).json()
            if state["status"] in {"failed", "succeeded", "cancelled"}:
                break
            time.sleep(0.02)
        assert state["status"] == "succeeded", state
        assert state["metrics"]["native_receipts"] >= 1
        assert state["decisions"]
        assert state["metrics"]["task_success"] == 1


def test_invalid_requests_and_missing_models_are_explicit(tmp_path):
    with TestClient(app(tmp_path)) as client:
        assert (
            client.post(
                "/api/run", headers=HEADERS, json={"case_id": "C01", "method_id": "M09"}
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/api/run",
                headers=HEADERS,
                json={"case_id": "C01", "method_id": "M01", "parameters": {"shell": "unsafe"}},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/run",
                headers=HEADERS,
                json={"case_id": "C01", "method_id": "M01", "seed": True},
            ).status_code
            == 422
        )
        assert (
            client.post("/api/replay", headers=HEADERS, json={"schema_version": 0}).status_code
            == 422
        )
        assert client.post("/api/cancel/missing", headers=HEADERS).status_code == 404
