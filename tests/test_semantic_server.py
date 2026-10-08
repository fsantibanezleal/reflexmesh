"""Exercise actual native semantic effects through the authenticated API."""

import time

import pytest
from fastapi.testclient import TestClient
from reflexmesh.contracts import Decision
from reflexmesh.server import create_app

TOKEN = "semantic-test-session-token-32-characters"
HEADERS = {"Authorization": "Bearer " + TOKEN}


class FirstAvailablePolicy:
    """Effect-path fixture, deliberately unrelated to benchmark performance."""

    policy_id = "M01"

    def reset(self, goal_id=None):
        pass

    def predict(self, observation):
        candidate = next(
            (
                c
                for c in observation.admissible
                if c.features.get("prerequisites") and not c.features.get("completed")
            ),
            None,
        )
        return Decision(
            self.policy_id, candidate.action_id if candidate else None, reason="test_effect_path"
        )


def test_semantic_run_has_native_receipts_and_distinct_provenance(tmp_path):
    with TestClient(
        create_app(
            checkpoints=tmp_path,
            token=TOKEN,
            allowed_hosts=("testserver",),
            policy_factory=lambda *args, **kwargs: FirstAvailablePolicy(),
        )
    ) as client:
        config = client.get("/api/config").json()
        assert config["suites"][1]["cases"] == [f"D{i:02}" for i in range(1, 9)]
        request = {
            "suite": "semantic_transfer",
            "case_id": "D01",
            "method_id": "M01",
            "seed": 82000,
        }
        assert client.post("/api/run", json=request).status_code == 401
        response = client.post("/api/run", headers=HEADERS, json=request)
        assert response.status_code == 200, response.text
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            state = client.get(
                "/api/events", headers=HEADERS, params={"run_id": response.json()["run_id"]}
            ).json()
            if state["status"] in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(0.02)
        assert state["error"] is None, state
        assert state["decisions"]
        assert state["metrics"]["native_receipts"] > 0
        assert state["episode"]["lane"] == "owned-semantic-transfer"
        assert state["provenance"]["execution_origin"] == "fresh-local"
        assert state["provenance"]["suite"] == "semantic_transfer"
        assert any(
            c["arguments"].get("description") for d in state["decisions"] for c in d["candidates"]
        )


@pytest.mark.parametrize(
    "override",
    [
        {"suite": []},
        {"suite": "unknown"},
        {"case_id": "C01"},
        {"variant": "noisy"},
        {"seed": 1000000},
        {"seed": True},
        {"method_id": {}},
    ],
)
def test_invalid_semantic_request_never_starts_worker(tmp_path, override):
    application = create_app(checkpoints=tmp_path, token=TOKEN, allowed_hosts=("testserver",))
    with TestClient(application) as client:
        request = {"suite": "semantic_transfer", "case_id": "D01", "method_id": "M01", **override}
        assert client.post("/api/run", headers=HEADERS, json=request).status_code == 422
        assert application.state.manager._runs == {}
