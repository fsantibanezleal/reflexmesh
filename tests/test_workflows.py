import hashlib
import json
import time

import pytest
from fastapi.testclient import TestClient
from reflexmesh.cli import run_workflow
from reflexmesh.server import create_app
from reflexmesh.workflows import execute_workflow


def step(name, path, content, dependencies=()):
    return {
        "id": name,
        "tool": "file.write",
        "arguments": {"path": path, "content": content},
        "depends_on": list(dependencies),
        "verify": {"path": path, "sha256": hashlib.sha256(content.encode()).hexdigest()},
    }


def test_dependency_recipe_executes_real_files_and_is_verified_on_repeat(tmp_path):
    recipe = {
        "schema_version": 1,
        "steps": [step("b", "b.txt", "B", ["a"]), step("a", "a.txt", "A")],
    }
    result = execute_workflow(recipe, tmp_path)
    assert result["verified_success"] and list(result["goals"]) == ["a", "b"]
    assert (tmp_path / "b.txt").read_text() == "B"
    assert len(result["goal_evidence"]["a"]["history"]) == 2
    repeated = execute_workflow(recipe, tmp_path)
    assert repeated["verified_success"] and not repeated["goal_evidence"]["a"]["history"]


@pytest.mark.parametrize(
    "bad",
    [
        step("escape", "../outside.txt", "bad"),
        step("cycle", "c.txt", "bad", ["cycle"]),
        {**step("shell", "c.txt", "bad"), "tool": "process.unregistered", "arguments": {}},
    ],
)
def test_whole_recipe_is_validated_before_any_effect(tmp_path, bad):
    recipe = {"schema_version": 1, "steps": [step("valid", "never.txt", "never"), bad]}
    with pytest.raises(ValueError):
        execute_workflow(recipe, tmp_path)
    assert not (tmp_path / "never.txt").exists()


def test_cli_and_sdk_share_execution_contract(tmp_path):
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps({"schema_version": 1, "steps": [step("write", "real.txt", "actual")]})
    )
    assert run_workflow(str(recipe), str(tmp_path))["verified_success"]
    assert (tmp_path / "real.txt").read_text() == "actual"


def test_local_workspace_preview_detects_changes_and_is_single_use(tmp_path):
    token = "test-token-at-least-24-characters"
    headers = {"Authorization": "Bearer " + token}
    app = create_app(checkpoints=tmp_path / "checkpoints", workspace=tmp_path, token=token)
    recipe = {"schema_version": 1, "steps": [step("write", "document.txt", "reviewed")]}
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/workspace/config").status_code == 401
        assert "canonical_root" not in client.get("/api/config").json()
        preview = client.post("/api/workspace/preview", json=recipe, headers=headers).json()
        (tmp_path / "document.txt").write_text("concurrent edit")
        assert (
            client.post(
                "/api/workspace/run", json={"preview_id": preview["preview_id"]}, headers=headers
            ).status_code
            == 409
        )
        assert (tmp_path / "document.txt").read_text() == "concurrent edit"
        preview = client.post("/api/workspace/preview", json=recipe, headers=headers).json()
        response = client.post(
            "/api/workspace/run", json={"preview_id": preview["preview_id"]}, headers=headers
        )
        assert response.status_code == 200
        run = response.json()["run_id"]
        assert (
            client.post(
                "/api/workspace/run", json={"preview_id": preview["preview_id"]}, headers=headers
            ).status_code
            == 409
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = client.get(
                "/api/workspace/events", params={"run_id": run}, headers=headers
            ).json()
            if result["status"] in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(0.01)
        assert result["status"] == "succeeded" and result["workflow"]["verified_success"]
        assert (
            result["workflow"]["goal_evidence"]["write"]["history"][-1]["receipt"]["status"]
            == "completed_verified"
        )
        assert (tmp_path / "document.txt").read_text() == "reviewed"


def test_workspace_capabilities_cannot_be_added_through_request(tmp_path):
    token = "test-token-at-least-24-characters"
    app = create_app(checkpoints=tmp_path, workspace=tmp_path, token=token)
    with TestClient(
        app, base_url="http://127.0.0.1", headers={"Authorization": "Bearer " + token}
    ) as client:
        recipe = {
            "schema_version": 1,
            "steps": [step("write", "a.txt", "A")],
            "processes": [{"name": "x", "argv": ["evil"]}],
        }
        assert client.post("/api/workspace/preview", json=recipe).status_code == 422
        assert not (tmp_path / "a.txt").exists()
