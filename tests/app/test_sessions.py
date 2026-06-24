"""Tests for the session lifecycle REST endpoints."""

import threading
import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.app.app import create_app
from src.app.core.settings import Settings
from src.app.schemas.session import SessionStatus
from src.app.services.session_manager import manager


@pytest.fixture
def client(tmp_path):
    """Provides a TestClient with the configured FastAPI app."""
    manager._sessions.clear()
    app = create_app(
        Settings(
            data_dir=tmp_path,
            upload_dir=tmp_path / "uploads",
            database_url="sqlite:///:memory:",
        )
    )
    return TestClient(app)


@patch("src.app.services.session_manager.run_offline_mp4_inference")
def test_session_happy_path(mock_inference, client, tmp_path):
    """Test full lifecycle: start -> status -> stop, mocking the actual inference."""
    pause_event = threading.Event()
    def mock_run(*args, **kwargs):
        pause_event.wait(timeout=2.0)
    mock_inference.side_effect = mock_run
    
    # Setup dummy files for FilePath validation
    video_file = tmp_path / "video.mp4"
    video_file.touch()
    ckpt_file = tmp_path / "model.pth"
    ckpt_file.touch()
    config_file = tmp_path / "config.yml"
    config_file.touch()

    # 1. Start session
    payload = {
        "video_path": str(video_file),
        "checkpoint_path": str(ckpt_file),
        "config_path": str(config_file),
    }
    
    response = client.post("/api/sessions", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["status"] == SessionStatus.RUNNING
    
    session_id = data["id"]
    
    # 2. Get status
    response = client.get(f"/api/sessions/{session_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == session_id
    assert data["status"] == SessionStatus.RUNNING

    # 3. Stop session
    response = client.post(f"/api/sessions/{session_id}/stop")
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == SessionStatus.STOPPED
    
    pause_event.set()


@patch("src.app.services.session_manager.run_offline_mp4_inference")
def test_session_duplicate_video(mock_inference, client, tmp_path):
    """Test starting a session for a video already being processed yields 409."""
    pause_event = threading.Event()
    def mock_run(*args, **kwargs):
        pause_event.wait(timeout=2.0)
    mock_inference.side_effect = mock_run
    
    video_file = tmp_path / "dup_video.mp4"
    video_file.touch()
    ckpt_file = tmp_path / "model.pth"
    ckpt_file.touch()
    config_file = tmp_path / "config.yml"
    config_file.touch()

    payload = {
        "video_path": str(video_file),
        "checkpoint_path": str(ckpt_file),
        "config_path": str(config_file),
    }

    # First start succeeds
    resp1 = client.post("/api/sessions", json=payload)
    assert resp1.status_code == 201

    # Second start with same video fails
    resp2 = client.post("/api/sessions", json=payload)
    assert resp2.status_code == 409
    assert "already being processed" in resp2.json()["detail"]
    
    pause_event.set()


def test_session_not_found(client):
    """Test accessing a non-existent session yields 404."""
    fake_id = str(uuid.uuid4())
    
    response = client.get(f"/api/sessions/{fake_id}")
    assert response.status_code == 404
    
    response = client.post(f"/api/sessions/{fake_id}/stop")
    assert response.status_code == 404


@patch("src.app.services.session_manager.run_offline_mp4_inference")
def test_stop_invalid_state(mock_inference, client, tmp_path):
    """Test stopping a session that is no longer running yields 400."""
    pause_event = threading.Event()
    def mock_run(*args, **kwargs):
        pause_event.wait(timeout=2.0)
    mock_inference.side_effect = mock_run

    # Setup dummy files
    video_file = tmp_path / "video.mp4"
    video_file.touch()
    ckpt_file = tmp_path / "model.pth"
    ckpt_file.touch()
    config_file = tmp_path / "config.yml"
    config_file.touch()

    # 1. Start session
    payload = {
        "video_path": str(video_file),
        "checkpoint_path": str(ckpt_file),
        "config_path": str(config_file),
    }
    
    response = client.post("/api/sessions", json=payload)
    session_id = response.json()["id"]

    # Stop it once
    response1 = client.post(f"/api/sessions/{session_id}/stop")
    assert response1.status_code == 202
    
    # Try stopping again
    response2 = client.post(f"/api/sessions/{session_id}/stop")
    assert response2.status_code == 400
    assert "Cannot stop" in response2.json()["detail"]
    
    pause_event.set()
