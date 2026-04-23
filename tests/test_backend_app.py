"""Basic tests for FastAPI backend skeleton.

Verifies health and readiness endpoints respond as expected.
"""

from fastapi.testclient import TestClient

from src.app.app import create_app


def test_health_and_readiness_endpoints():
    app = create_app()
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j.get("status") == "ok"
    assert "timestamp" in j

    r2 = client.get("/readiness")
    assert r2.status_code == 200
    j2 = r2.json()
    assert j2.get("ready") is True


def test_api_root():
    app = create_app()
    client = TestClient(app)

    r = client.get("/api/")
    assert r.status_code == 200
    assert r.json().get("message") == "API root"
