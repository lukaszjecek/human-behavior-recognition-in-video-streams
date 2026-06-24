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
    Krok 1: Test braku oczekiwania na EOF.
    Udowadnia, że kamera na żywo emituje zdarzenia (w tym alerty) w trakcie
    aktywnego streamu, zanim klient wyśle 'stop' i zakończy połączenie.
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


def test_mp4_and_camera_payload_compatibility(client, pipeline_assets, dummy_video, tmp_path):
    """
    Krok 2: Test zgodności kontraktów MP4 i WebSockets.
    Uruchamia inferencję na MP4 (z CLI) oraz przez kamerę na żywo (WebSockets).
    Wyciąga payloady wygenerowanych zdarzeń i weryfikuje czy obydwa w pełni
    używają tej samej struktury zadeklarowanej w ActionEvent.
    """
    ckpt_path, config_path = pipeline_assets
    session_id = uuid.uuid4()
    
    # --- ŚCIEŻKA 1: MP4 CLI ---
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
    
    # Weryfikacja kontraktu po stronie MP4
    # Parse to ActionEvent to ensure schema compliance
    mp4_action_event = ActionEvent(**mp4_event_raw)


    # --- ŚCIEŻKA 2: CAMERA WEBSOCKET ---
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
        
        # Weryfikacja kontraktu po stronie Camery
        camera_action_event = ActionEvent(**camera_event_raw)
        
        ws.send_text("stop")
        ws.receive_json() # status stopped

    # --- PORÓWNANIE KONTRAKTÓW ---
    # Obie ścieżki (MP4 i Websocket) udało się zdeserializować do ActionEvent.
    # Weryfikujemy, czy payloady JSON zawierają niezbędne, wspólne klucze bazowe.
    mp4_keys = set(mp4_event_raw.keys())
    camera_keys = set(camera_event_raw.keys())
    
    required_keys = {"label", "confidence", "start_frame_index", "end_frame_index"}
    assert required_keys.issubset(mp4_keys), f"MP4 missing keys: {required_keys - mp4_keys}"
    assert required_keys.issubset(camera_keys), f"Camera missing keys: {required_keys - camera_keys}"
    
    # Dodatkowe asercje dla pewności, że pola są poprawnego typu i znaczenia
    assert isinstance(mp4_action_event.label, str)
    assert isinstance(camera_action_event.label, str)
    assert 0.0 <= mp4_action_event.confidence <= 1.0
    assert 0.0 <= camera_action_event.confidence <= 1.0
