from fastapi.testclient import TestClient
from reflexmesh.server import create_app


def test_spa_fallback_never_masks_missing_assets(tmp_path):
    (tmp_path / "index.html").write_text("<html>workbench</html>")
    with TestClient(
        create_app(checkpoints=tmp_path, static=tmp_path, allowed_hosts=("testserver",))
    ) as client:
        assert client.get("/introduction").status_code == 200
        for path in ["/missing.json", "/missing.svg", "/api/missing", "/data/missing"]:
            assert (
                client.get(
                    path, headers={"Authorization": "Bearer " + client.app.state.local_token}
                ).status_code
                == 404
            )
