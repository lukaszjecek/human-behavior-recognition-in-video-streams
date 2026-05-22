import pytest
from fastapi.testclient import TestClient

from src.app.app import create_app
from src.app.schemas.action_event import (
    ActionEvent,
    AlertData,
    BoundingBox,
    EventPayload,
    EventType,
)
from src.app.services.websocket_manager import websocket_manager


@pytest.fixture(autouse=True)
def reset_websocket_manager_state() -> None:
    websocket_manager.active_connections = []
    websocket_manager.loop = None
    yield
    websocket_manager.active_connections = []
    websocket_manager.loop = None


def test_websocket_echo() -> None:
    """Test that the echo websocket correctly echoes text back."""
    app = create_app()
    client = TestClient(app)
    with client.websocket_connect("/ws/echo") as websocket:
        websocket.send_text("hello")
        data = websocket.receive_text()
        assert data == "echo: hello"


def test_websocket_live_stream() -> None:
    """Test that live notifications are correctly pushed to connected websocket clients."""
    app = create_app()
    client = TestClient(app)
    with client.websocket_connect("/ws/live") as websocket:
        # Create a test event payload
        action_evt = ActionEvent(
            start_frame_index=1,
            end_frame_index=16,
            start_timestamp=0.0,
            end_timestamp=1.0,
            label="jump",
            confidence=0.85,
        )
        payload = EventPayload(
            event_type=EventType.DETECTION,
            camera_id="test_cam.mp4",
            data=action_evt,
        )

        # Broadcast the payload synchronously using the manager
        websocket_manager.broadcast_sync(payload)

        # Retrieve and verify the broadcasted message on the socket client
        data = websocket.receive_json()
        assert data["event_type"] == "DETECTION"
        assert data["camera_id"] == "test_cam.mp4"
        assert data["data"]["label"] == "jump"
        assert data["data"]["confidence"] == 0.85

        # Verify alert payload broadcasting
        alert_data = AlertData(
            severity="HIGH",
            message="Alert triggered for label: jump",
            action_event=action_evt,
        )
        alert_payload = EventPayload(
            event_type=EventType.ALERT,
            camera_id="test_cam.mp4",
            data=alert_data,
        )

        websocket_manager.broadcast_sync(alert_payload)

        data = websocket.receive_json()
        assert data["event_type"] == "ALERT"
        assert data["camera_id"] == "test_cam.mp4"
        assert data["data"]["severity"] == "HIGH"
        assert data["data"]["message"] == "Alert triggered for label: jump"


def test_websocket_live_stream_with_bboxes() -> None:
    """Test that live notifications with bounding boxes are pushed correctly."""
    app = create_app()
    client = TestClient(app)
    with client.websocket_connect("/ws/live") as websocket:
        # Create a test event payload with bboxes
        bbox = BoundingBox(
            x_min=10.0,
            y_min=20.0,
            x_max=100.0,
            y_max=200.0,
            label="car",
            confidence=0.9,
            coordinate_space="source_pixels",
            frame_index=123,
            source_width=1280,
            source_height=720,
        )
        action_evt = ActionEvent(
            start_frame_index=1,
            end_frame_index=16,
            start_timestamp=0.0,
            end_timestamp=1.0,
            label="jump",
            confidence=0.85,
            bboxes=[bbox],
        )
        payload = EventPayload(
            event_type=EventType.DETECTION,
            camera_id="test_cam.mp4",
            data=action_evt,
        )

        # Broadcast the payload synchronously using the manager
        websocket_manager.broadcast_sync(payload)

        # Retrieve and verify the broadcasted message on the socket client
        data = websocket.receive_json()
        assert data["event_type"] == "DETECTION"
        assert data["camera_id"] == "test_cam.mp4"
        assert data["data"]["label"] == "jump"
        assert data["data"]["confidence"] == 0.85
        assert "bboxes" in data["data"]
        assert len(data["data"]["bboxes"]) == 1
        assert data["data"]["bboxes"][0]["label"] == "car"
        assert data["data"]["bboxes"][0]["x_min"] == 10.0
        assert data["data"]["bboxes"][0]["confidence"] == 0.9
        assert data["data"]["bboxes"][0]["box_format"] == "xyxy"
        assert data["data"]["bboxes"][0]["coordinate_space"] == "source_pixels"
        assert data["data"]["bboxes"][0]["frame_index"] == 123
        assert data["data"]["bboxes"][0]["source_width"] == 1280
        assert data["data"]["bboxes"][0]["source_height"] == 720


def test_websocket_api_echo() -> None:
    """Test that the api prefix echo websocket works correctly."""
    app = create_app()
    client = TestClient(app)
    with client.websocket_connect("/api/websocket/echo") as websocket:
        websocket.send_text("hello")
        data = websocket.receive_text()
        assert data == "echo: hello"


def test_websocket_api_live_stream() -> None:
    """Test that live notifications are correctly pushed to /api/websocket/live clients."""
    app = create_app()
    client = TestClient(app)
    with client.websocket_connect("/api/websocket/live") as websocket:
        action_evt = ActionEvent(
            start_frame_index=1,
            end_frame_index=16,
            start_timestamp=0.0,
            end_timestamp=1.0,
            label="jump",
            confidence=0.85,
        )
        payload = EventPayload(
            event_type=EventType.DETECTION,
            camera_id="test_cam.mp4",
            data=action_evt,
        )

        websocket_manager.broadcast_sync(payload)

        data = websocket.receive_json()
        assert data["event_type"] == "DETECTION"
        assert data["camera_id"] == "test_cam.mp4"
        assert data["data"]["label"] == "jump"


def test_websocket_end_to_end_integration(tmp_path) -> None:
    """Test that live detections/alerts from active sessions are sent to websocket."""
    import time

    import cv2
    import numpy as np
    import torch
    import yaml

    # 1. Create a dummy valid MP4 video of 20 frames
    video_path = tmp_path / "integration_video.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (64, 64),
    )
    assert writer.isOpened()
    for _ in range(20):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    # 2. Create a configuration YAML
    config_path = tmp_path / "config.yml"
    config_data = {
        "pipeline": {"target_resolution": [64, 64], "temporal_window": 4},
        "inference": {"stride": 2, "class_labels": ["walk", "run", "fight"], "device": "cpu"},
        "tracking": {"default_track_id": 1},
        "alert": {"persistence_threshold": 2, "resolve_threshold": 1, "danger_labels": ["fight"]},
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)

    # 3. Create a dummy model checkpoint where the model outputs 'fight' prediction (index 2)
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

    # 4. Start the FastAPI app and test client
    app = create_app()
    client = TestClient(app)

    # 5. Connect to the WebSocket stream
    with client.websocket_connect("/api/websocket/live") as websocket:
        # Start the inference session by calling the endpoint
        payload = {
            "video_path": str(video_path),
            "checkpoint_path": str(checkpoint_path),
            "config_path": str(config_path),
        }

        response = client.post("/api/sessions", json=payload)
        assert response.status_code == 201
        session_id = response.json()["id"]

        # Now, wait for websocket messages. We expect:
        # - DETECTION messages as windows are processed.
        # - An ALERT message once the persistence threshold is met.

        received_detections = []
        received_alerts = []

        start_time = time.time()
        while (len(received_detections) < 3 or len(received_alerts) < 1) and (
            time.time() - start_time < 15
        ):
            try:
                msg = websocket.receive_json()
                if msg["event_type"] == "DETECTION":
                    received_detections.append(msg)
                elif msg["event_type"] == "ALERT":
                    received_alerts.append(msg)
            except Exception:
                break

        # Verify that we actually received detections and alerts
        assert len(received_detections) > 0, "No DETECTION events received over WebSocket"
        assert len(received_alerts) > 0, "No ALERT events received over WebSocket"

        # Verify the structure of one of the received detections
        detection = received_detections[0]
        assert detection["camera_id"] == "integration_video.mp4"
        assert detection["data"]["label"] == "fight"
        assert detection["data"]["confidence"] > 0.9

        # Verify the structure of the received alert
        alert = received_alerts[0]
        assert alert["camera_id"] == "integration_video.mp4"
        assert alert["data"]["severity"] == "HIGH"
        assert "fight" in alert["data"]["message"]

        # Clean up the session by stopping it
        client.post(f"/api/sessions/{session_id}/stop")
