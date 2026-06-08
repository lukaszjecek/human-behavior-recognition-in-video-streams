"""Automated integration tests for the browser camera WebSocket endpoint."""

import uuid

import cv2
import numpy as np
import pytest
import torch
import yaml
from fastapi.testclient import TestClient

from src.app.app import create_app
from src.app.core.settings import settings
from src.app.db.models import Base
from src.app.db.repository import get_events
from src.app.db.session import get_db


@pytest.fixture(name="test_db")
def fixture_test_db():
    """Set up an isolated in-memory SQLite database for testing."""
    from src.app.db import session

    original_url = settings.database_url
    settings.database_url = "sqlite:///:memory:"
    session._engine = None

    engine = session.get_engine()
    session.init_db()

    db = session.SessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)
        session._engine = None
        settings.database_url = original_url


@pytest.fixture(name="client")
def fixture_client(test_db):
    """Provides a TestClient with overridden get_db dependency."""
    app = create_app()

    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture(name="pipeline_assets")
def fixture_pipeline_assets(tmp_path):
    """Creates a configuration YAML and a model checkpoint for testing."""
    # 1. Create a configuration YAML
    config_path = tmp_path / "config.yml"
    config_data = {
        "pipeline": {"target_resolution": [64, 64], "temporal_window": 4},
        "inference": {"stride": 2, "class_labels": ["walk", "run", "fight"], "device": "cpu"},
        "tracking": {"default_track_id": 1},
        "alert": {"persistence_threshold": 2, "resolve_threshold": 1, "danger_labels": ["fight"]},
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)

    # 2. Create a dummy model checkpoint where the model outputs 'fight' prediction (index 2)
    checkpoint_path = tmp_path / "model.pth"
    num_classes = 3
    state_dict = {
        "fc.weight": torch.zeros((num_classes, 3)),
        "fc.bias": torch.zeros(num_classes),
    }
    # To ensure class index 2 (fight) is predicted, we set its weights high
    state_dict["fc.weight"][2, :] = 10.0
    state_dict["fc.bias"][2] = 5.0

    checkpoint = {
        "model_name": "dummy",
        "model_state_dict": state_dict,
    }
    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path, config_path


def test_camera_ws_initialization_success(client, pipeline_assets):
    """Test successful initialization of the camera stream session."""
    ckpt_path, config_path = pipeline_assets
    with client.websocket_connect("/api/websocket/camera") as ws:
        init_payload = {
            "checkpoint_path": str(ckpt_path),
            "config_path": str(config_path),
            "device": "cpu",
        }
        ws.send_json(init_payload)

        # Expect an initialized status response
        resp = ws.receive_json()
        assert resp["message_type"] == "STATUS"
        assert resp["status"] == "initialized"
        assert "successfully initialized" in resp["message"]
        assert resp["session_id"] is not None


def test_camera_ws_initialization_failure_missing_fields(client):
    """Test initialization failure when required fields are missing."""
    with client.websocket_connect("/api/websocket/camera") as ws:
        init_payload = {
            "checkpoint_path": "some_path.pth",
            # missing config_path
        }
        ws.send_json(init_payload)

        resp = ws.receive_json()
        assert resp["message_type"] == "STATUS"
        assert resp["status"] == "initialization_failed"
        assert "Missing checkpoint_path" in resp["message"]

        # WebSocket should be closed by backend
        with pytest.raises(Exception):
            ws.receive_json()


def test_camera_ws_initialization_failure_invalid_paths(client, tmp_path):
    """Test initialization failure when paths do not exist."""
    with client.websocket_connect("/api/websocket/camera") as ws:
        init_payload = {
            "checkpoint_path": str(tmp_path / "nonexistent.pth"),
            "config_path": str(tmp_path / "nonexistent.yml"),
        }
        ws.send_json(init_payload)

        resp = ws.receive_json()
        assert resp["message_type"] == "STATUS"
        assert resp["status"] == "initialization_failed"
        assert "Pipeline initialization failed" in resp["message"]
        assert resp["error_type"] == "ValueError"
        assert "invalid or restricted" in resp["error"]


def test_camera_ws_initialization_failure_path_traversal(client):
    """Test that path traversal attempts or files outside allowed directories are blocked."""
    with client.websocket_connect("/api/websocket/camera") as ws:
        # A path that resides outside allowed CWD/TEMP/app
        init_payload = {
            "checkpoint_path": "/restricted/model.pth",
            "config_path": "/restricted/config.yml",
        }
        ws.send_json(init_payload)

        resp = ws.receive_json()
        assert resp["message_type"] == "STATUS"
        assert resp["status"] == "initialization_failed"
        assert resp["error_type"] == "ValueError"
        assert "invalid or restricted" in resp["error"]


def test_camera_ws_stop_message(client, pipeline_assets):
    """Test that client can stop the streaming loop with a 'stop' text message."""
    ckpt_path, config_path = pipeline_assets
    with client.websocket_connect("/api/websocket/camera") as ws:
        init_payload = {
            "checkpoint_path": str(ckpt_path),
            "config_path": str(config_path),
        }
        ws.send_json(init_payload)
        ws.receive_json()  # Consume the init status

        # Send stop message
        ws.send_text("stop")
        resp = ws.receive_json()
        assert resp["message_type"] == "STATUS"
        assert resp["status"] == "stopped"

        # Connection should close
        with pytest.raises(Exception):
            ws.receive_json()


def test_camera_ws_invalid_frame_bytes(client, pipeline_assets):
    """Test that invalid frame bytes are handled without closing the websocket."""
    ckpt_path, config_path = pipeline_assets
    with client.websocket_connect("/api/websocket/camera") as ws:
        init_payload = {
            "checkpoint_path": str(ckpt_path),
            "config_path": str(config_path),
        }
        ws.send_json(init_payload)
        ws.receive_json()  # Consume the init status

        # Send invalid binary bytes
        ws.send_bytes(b"invalid-image-data-bytes")
        resp = ws.receive_json()
        assert resp["message_type"] == "STATUS"
        assert resp["status"] == "running"
        assert "Error processing frame" in resp["message"]
        assert resp["error_type"] == "ValueError"
        assert "Failed to decode binary frame" in resp["error"]

        # Send a stop message to verify connection is still active and can be closed cleanly
        ws.send_text("stop")
        resp = ws.receive_json()
        assert resp["status"] == "stopped"


def test_camera_ws_realtime_observable_before_stop(client, pipeline_assets):
    """Test that client can receive detections and alerts without having to send 'stop' first."""
    ckpt_path, config_path = pipeline_assets
    session_id = uuid.uuid4()
    with client.websocket_connect("/api/websocket/camera") as ws:
        init_payload = {
            "checkpoint_path": str(ckpt_path),
            "config_path": str(config_path),
            "session_id": str(session_id),
        }
        ws.send_json(init_payload)
        ws.receive_json()  # Consume the init status

        # Create dummy 64x64 BGR frames
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        _, jpeg_bytes = cv2.imencode(".jpg", frame)
        raw_bytes = jpeg_bytes.tobytes()

        # Window size is 4, stride is 2.
        # Send 4 frames to complete Window 1 -> should trigger DETECTION immediately.
        for _ in range(4):
            ws.send_bytes(raw_bytes)

        # We should be able to receive a DETECTION JSON immediately without sending stop!
        det_msg = ws.receive_json()
        assert det_msg["event_type"] == "DETECTION"
        assert det_msg["session_id"] == str(session_id)
        assert det_msg["data"]["label"] == "fight"

        # Send 2 more frames to complete Window 2 (frames 2-5, total 6 frames).
        # Since both window 1 and window 2 output 'fight' (danger label) and persistence_threshold is 2:
        # Window 2 will trigger another DETECTION followed by an ALERT.
        for _ in range(2):
            ws.send_bytes(raw_bytes)

        # Read next two messages - we expect one DETECTION and one ALERT.
        messages = [ws.receive_json(), ws.receive_json()]
        event_types = [m["event_type"] for m in messages]
        assert "DETECTION" in event_types
        assert "ALERT" in event_types

        # Finally, send stop cleanly
        ws.send_text("stop")
        resp = ws.receive_json()
        assert resp["status"] == "stopped"


def test_camera_ws_processing_flow(client, test_db, pipeline_assets):
    """Test that streaming valid frames triggers detections, alerts, and db persistence."""
    ckpt_path, config_path = pipeline_assets
    session_id = uuid.uuid4()

    with client.websocket_connect("/api/websocket/camera") as ws:
        init_payload = {
            "checkpoint_path": str(ckpt_path),
            "config_path": str(config_path),
            "session_id": str(session_id),
        }
        ws.send_json(init_payload)
        ws.receive_json()  # Consume the init status

        # Create dummy 64x64 BGR frames
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        _, jpeg_bytes = cv2.imencode(".jpg", frame)
        raw_bytes = jpeg_bytes.tobytes()

        # Send frames to trigger window inferences (temporal_window=4, stride=2)
        # To get detections and alerts, we need to complete at least 2 windows
        # (consecutive danger labels)
        # Window 1: frames 0-3 (requires 4 frames).
        # Window 2: frames 2-5 (requires 6 frames total).
        for _ in range(6):
            ws.send_bytes(raw_bytes)

        # Send stop message to close the websocket connection cleanly and send final status message
        ws.send_text("stop")

        received_detections = []
        received_alerts = []
        received_statuses = []

        # Read responses until connection is closed
        while True:
            try:
                msg = ws.receive_json()
                if msg.get("event_type") == "DETECTION":
                    received_detections.append(msg)
                elif msg.get("event_type") == "ALERT":
                    received_alerts.append(msg)
                elif msg.get("message_type") == "STATUS":
                    received_statuses.append(msg)
            except Exception:
                break

        # Assert detections and alerts were received
        assert len(received_detections) >= 2
        assert len(received_alerts) >= 1
        assert any(status["status"] == "stopped" for status in received_statuses)

        # Check detection event fields
        det = received_detections[0]
        assert det["camera_id"] == "browser_camera"
        assert det["session_id"] == str(session_id)
        assert det["data"]["label"] == "fight"
        assert det["data"]["confidence"] > 0.9

        # Check alert event fields
        alert = received_alerts[0]
        assert alert["camera_id"] == "browser_camera"
        assert alert["session_id"] == str(session_id)
        assert alert["data"]["severity"] == "HIGH"
        assert "fight" in alert["data"]["message"]

        # Verify that these events are persisted to the database
        db_events = get_events(test_db, session_id=session_id)
        # At least one DETECTION and one ALERT persisted
        event_types = [e.event_type for e in db_events]
        assert "DETECTION" in event_types
        assert "ALERT" in event_types

        # Verify the database records contain correct metadata
        persisted_detections = [e for e in db_events if e.event_type == "DETECTION"]
        assert persisted_detections[0].camera_id == "browser_camera"
        assert persisted_detections[0].payload["data"]["label"] == "fight"
