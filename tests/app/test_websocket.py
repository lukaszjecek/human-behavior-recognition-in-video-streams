from fastapi.testclient import TestClient
import pytest

from src.app.app import create_app
from src.app.schemas.action_event import ActionEvent, AlertData, EventPayload, EventType
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

