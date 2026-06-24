import uuid
import cv2
import numpy as np
import pytest
import torch
import yaml
import json
from fastapi.testclient import TestClient

from src.app.schemas.action_event import ActionEvent
from src.inference.mp4_cli import InferenceCliRequest, run_mp4_to_json_action_inference

from src.app.app import create_app
from src.app.core.settings import settings
from src.app.db.models import Base
from src.app.db.session import get_db


@pytest.fixture(name="test_db")
def fixture_test_db():
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
    config_path = tmp_path / "config.yml"
    config_data = {
        "pipeline": {"target_resolution": [64, 64], "temporal_window": 4},
        "inference": {"stride": 2, "class_labels": ["walk", "run", "fight"], "device": "cpu"},
        "tracking": {"default_track_id": 1},
        "alert": {"persistence_threshold": 2, "resolve_threshold": 1, "danger_labels": ["fight"]},
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)

    checkpoint_path = tmp_path / "model.pth"
    num_classes = 3
    state_dict = {
        "fc.weight": torch.zeros((num_classes, 3)),
        "fc.bias": torch.zeros(num_classes),
    }
    state_dict["fc.weight"][2, :] = 10.0
    state_dict["fc.bias"][2] = 5.0

    checkpoint = {
        "model_name": "dummy",
        "model_state_dict": state_dict,
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path, config_path


def test_streaming_emits_before_stop(client, pipeline_assets):
    """
    Step 1: Test no expectation of EOF.
    Proves that live camera emits events (including alerts) during an active stream,
    before the client sends 'stop' and closes the connection.
    """
    ckpt_path, config_path = pipeline_assets
    session_id = uuid.uuid4()

    with client.websocket_connect("/api/websocket/camera") as ws:
        init_payload = {
            "checkpoint_path": str(ckpt_path),
            "config_path": str(config_path),
            "session_id": str(session_id),
        }
        ws.send_json(init_payload)
        
        status = ws.receive_json()
        assert status["message_type"] == "STATUS"
        assert status["status"] == "initialized"

        # Create dummy 64x64 BGR frames
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        _, jpeg_bytes = cv2.imencode(".jpg", frame)
        raw_bytes = jpeg_bytes.tobytes()

        # Send exactly enough frames to complete first window (window_size=4)
        for _ in range(4):
            ws.send_bytes(raw_bytes)

        # We assert that we can receive the DETECTION event IMMEDIATELY.
        # If the backend waits for EOF, this call will hang indefinitely (or timeout).
        det_msg = ws.receive_json()
        
        assert det_msg.get("event_type") == "DETECTION", "Should emit DETECTION immediately"
        assert det_msg["data"]["label"] == "fight"

        # Send more frames to trigger an alert (persistence_threshold=2)
        # We need another window to complete. Stride=2 means 2 more frames.
        for _ in range(2):
            ws.send_bytes(raw_bytes)

        # We expect a DETECTION and an ALERT because of the danger label.
        msg1 = ws.receive_json()
        msg2 = ws.receive_json()
        
        event_types = {msg1.get("event_type"), msg2.get("event_type")}
        assert "DETECTION" in event_types, "Second detection should be emitted"
        assert "ALERT" in event_types, "Alert should be emitted during active stream"

        # Explicitly verify we are still connected and CAN send stop
        ws.send_text("stop")
        stop_resp = ws.receive_json()
        assert stop_resp["status"] == "stopped", "Stream should gracefully stop when instructed"


def test_streaming_false_positive_prevention(client, pipeline_assets):
    """
    Step 1 Extension: Ensure that sending fewer frames than the temporal_window
    does not prematurely trigger a DETECTION event.
    """
    ckpt_path, config_path = pipeline_assets
    session_id = uuid.uuid4()

    with client.websocket_connect("/api/websocket/camera") as ws:
        init_payload = {
            "checkpoint_path": str(ckpt_path),
            "config_path": str(config_path),
            "session_id": str(session_id),
        }
        ws.send_json(init_payload)
        ws.receive_json()  # STATUS initialized

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        _, jpeg_bytes = cv2.imencode(".jpg", frame)
        raw_bytes = jpeg_bytes.tobytes()

        # Send only 3 frames (window_size is 4)
        for _ in range(3):
            ws.send_bytes(raw_bytes)

        # Send stop immediately
        ws.send_text("stop")
        
        # The next message MUST be 'stopped' status. If we got a DETECTION here,
        # it means it was emitted prematurely without waiting for full temporal window.
        resp = ws.receive_json()
        assert resp.get("message_type") == "STATUS", "Should be STATUS, not DETECTION"
        assert resp.get("status") == "stopped"


def test_streaming_sudden_disconnect(client, pipeline_assets):
    """
    Step 1 Extension: Ensure the backend handles a sudden client disconnect
    gracefully without crashing.
    """
    ckpt_path, config_path = pipeline_assets
    session_id = uuid.uuid4()

    # Context manager handles the websocket lifecycle
    with client.websocket_connect("/api/websocket/camera") as ws:
        init_payload = {
            "checkpoint_path": str(ckpt_path),
            "config_path": str(config_path),
            "session_id": str(session_id),
        }
        ws.send_json(init_payload)
        ws.receive_json()  # STATUS initialized

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        _, jpeg_bytes = cv2.imencode(".jpg", frame)
        raw_bytes = jpeg_bytes.tobytes()

        # Send frames but immediately disconnect (exit context manager)
        for _ in range(4):
            ws.send_bytes(raw_bytes)
            
    # If the backend crashed due to disconnect, the next client request or app state might be compromised.
    # We can verify the app is still alive by starting a new WS connection.
    with client.websocket_connect("/api/websocket/camera") as ws2:
        ws2.send_json(init_payload)
        status = ws2.receive_json()
        assert status["status"] == "initialized", "Backend survived disconnect and allows new connections"


def test_mp4_and_camera_payload_compatibility(client, pipeline_assets, dummy_video, tmp_path):
    """
    Step 2: MP4 and WebSockets contract compatibility test.
    Runs inference on MP4 (via CLI) and via live camera (WebSockets).
    Extracts the generated event payloads and verifies that both fully
    utilize the same structure declared in ActionEvent.
    """
    ckpt_path, config_path = pipeline_assets
    session_id = uuid.uuid4()
    
    # --- PATH 1: MP4 CLI ---
    output_path = tmp_path / "actions.json"
    request = InferenceCliRequest(
        input_path=dummy_video,
        checkpoint_path=ckpt_path,
        config_path=config_path,
        output_path=output_path,
        device="cpu"
    )
    exit_code = run_mp4_to_json_action_inference(request)
    assert exit_code == 0, "MP4 CLI inference failed"
    
    mp4_data = json.loads(output_path.read_text(encoding="utf-8"))
    assert mp4_data["event_count"] > 0
    mp4_event_raw = mp4_data["events"][0]
    
    # Contract verification on MP4 side
    # Parse to ActionEvent to ensure schema compliance
    mp4_action_event = ActionEvent(**mp4_event_raw)


    # --- PATH 2: CAMERA WEBSOCKET ---
    with client.websocket_connect("/api/websocket/camera") as ws:
        init_payload = {
            "checkpoint_path": str(ckpt_path),
            "config_path": str(config_path),
            "session_id": str(session_id),
        }
        ws.send_json(init_payload)
        ws.receive_json()  # STATUS initialized

        # Push one window of frames
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        _, jpeg_bytes = cv2.imencode(".jpg", frame)
        raw_bytes = jpeg_bytes.tobytes()

        for _ in range(4):
            ws.send_bytes(raw_bytes)

        det_msg = ws.receive_json()
        assert det_msg.get("event_type") == "DETECTION"
        camera_event_raw = det_msg["data"]
        
        # Contract verification on Camera side
        camera_action_event = ActionEvent(**camera_event_raw)
        
        ws.send_text("stop")
        ws.receive_json() # status stopped

    # --- CONTRACT COMPARISON ---
    # Both paths (MP4 and Websocket) were successfully deserialized into ActionEvent.
    # We verify that JSON payloads contain necessary, shared baseline keys.
    mp4_keys = set(mp4_event_raw.keys())
    camera_keys = set(camera_event_raw.keys())
    
    required_keys = {"label", "confidence", "start_frame_index", "end_frame_index"}
    assert required_keys.issubset(mp4_keys), f"MP4 missing keys: {required_keys - mp4_keys}"
    assert required_keys.issubset(camera_keys), f"Camera missing keys: {required_keys - camera_keys}"
    
    # Additional assertions to ensure fields are of correct type and meaning
    assert isinstance(mp4_action_event.label, str)
    assert isinstance(camera_action_event.label, str)
    assert 0.0 <= mp4_action_event.confidence <= 1.0
    assert 0.0 <= camera_action_event.confidence <= 1.0

    # Extended Checks: Tracking IDs
    # Since tracking config has default_track_id: 1, both should have it properly attached.
    assert mp4_action_event.track_id == 1
    assert camera_action_event.track_id == 1


def test_context_fallback_behavior_unknown(client, pipeline_assets, dummy_video, tmp_path):
    """
    Step 3: Test context fallback behavior.
    Verifies that when context is unavailable (no explicit integration), both 
    MP4 and Camera pipelines fallback correctly to 'unknown' scene tag.
    """
    ckpt_path, config_path = pipeline_assets
    session_id = uuid.uuid4()
    
    # --- Check MP4 ---
    output_path = tmp_path / "actions.json"
    request = InferenceCliRequest(
        input_path=dummy_video,
        checkpoint_path=ckpt_path,
        config_path=config_path,
        output_path=output_path,
        device="cpu"
    )
    run_mp4_to_json_action_inference(request)
    mp4_data = json.loads(output_path.read_text(encoding="utf-8"))
    mp4_event = ActionEvent(**mp4_data["events"][0])
    
    assert mp4_event.context is not None, "Context should be attached to MP4 events"
    assert mp4_event.context.scene_tag == "unknown", "MP4 Context should fallback to 'unknown'"
    assert mp4_event.context.confidence == 0.0

    # --- Check Camera ---
    with client.websocket_connect("/api/websocket/camera") as ws:
        init_payload = {
            "checkpoint_path": str(ckpt_path),
            "config_path": str(config_path),
            "session_id": str(session_id),
        }
        ws.send_json(init_payload)
        ws.receive_json()

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        _, jpeg_bytes = cv2.imencode(".jpg", frame)
        raw_bytes = jpeg_bytes.tobytes()

        for _ in range(4):
            ws.send_bytes(raw_bytes)

        det_msg = ws.receive_json()
        camera_event = ActionEvent(**det_msg["data"])
        
        ws.send_text("stop")
        ws.receive_json()

    assert camera_event.context is not None, "Context should be attached to Camera events"
    assert camera_event.context.scene_tag == "unknown", "Camera Context should fallback to 'unknown'"
    assert camera_event.context.confidence == 0.0


def test_bounding_boxes_presence(client, pipeline_assets, dummy_video, tmp_path):
    """
    Step 4: Test bounding boxes field presence.
    Verifies that the bboxes field complies with the ActionEvent contract.
    It should either be None or a valid list of BoundingBox objects, 
    even when using a dummy model that doesn't explicitly emit spatial data.
    """
    ckpt_path, config_path = pipeline_assets
    
    output_path = tmp_path / "actions.json"
    request = InferenceCliRequest(
        input_path=dummy_video,
        checkpoint_path=ckpt_path,
        config_path=config_path,
        output_path=output_path,
        device="cpu"
    )
    run_mp4_to_json_action_inference(request)
    mp4_data = json.loads(output_path.read_text(encoding="utf-8"))
    mp4_event = ActionEvent(**mp4_data["events"][0])
    
    # We verify that bboxes is handled correctly according to schema.
    # It must be None or a list.
    assert mp4_event.bboxes is None or isinstance(mp4_event.bboxes, list), "bboxes field must match ActionEvent schema"
