import json
import logging

from fastapi.testclient import TestClient

from src.app.app import create_app
from src.app.schemas.action_event import ActionEvent, EventPayload, EventType
from src.inference.runtime_logging import configure_runtime_logging, log_audit_event


def test_request_id_middleware(tmp_path, monkeypatch):
    """Verify that request correlation IDs are generated and returned in headers."""
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("LOG_DIR", str(log_dir))

    app = create_app()
    client = TestClient(app)

    # 1. Request without X-Request-ID should auto-generate one
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    req_id_1 = response.headers["X-Request-ID"]
    assert len(req_id_1) > 0

    # 2. Request with X-Request-ID should preserve it
    response = client.get("/health", headers={"X-Request-ID": "custom-id-123"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "custom-id-123"


def test_log_audit_event(tmp_path, monkeypatch):
    """Verify that log_audit_event writes EventPayload JSON lines to audit.log."""
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("LOG_DIR", str(log_dir))

    evt = ActionEvent(
        start_frame_index=0,
        end_frame_index=10,
        label="walking",
        confidence=0.8,
    )
    payload = EventPayload(
        camera_id="cam_abc",
        event_type=EventType.DETECTION,
        data=evt,
    )

    # Call the audit log helper
    log_audit_event(payload)

    audit_file = log_dir / "audit.log"
    assert audit_file.exists()

    # Read the file
    content = audit_file.read_text(encoding="utf-8")
    lines = content.strip().split("\n")
    assert len(lines) == 1

    data = json.loads(lines[0])
    assert data["camera_id"] == "cam_abc"
    assert data["event_type"] == "DETECTION"
    assert data["data"]["label"] == "walking"


def test_configure_runtime_logging_creates_file(tmp_path, monkeypatch):
    """Verify configure_runtime_logging sets up a FileHandler writing JSON records."""
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("LOG_DIR", str(log_dir))

    # Configure and log an event
    configure_runtime_logging(log_file="test_backend.log")

    logger = logging.getLogger("hbr.structured")
    logger.info("Test log entry")

    log_file = log_dir / "test_backend.log"
    assert log_file.exists()

    # Verify content is JSON
    content = log_file.read_text(encoding="utf-8")
    assert "Test log entry" in content
    
    # Assert JSON format
    data = json.loads(content.strip().split("\n")[-1])
    assert data["message"] == "Test log entry"

    # Cleanup handlers to avoid interfering with other tests
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
