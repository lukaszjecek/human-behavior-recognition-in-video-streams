"""Tests for MP4 upload and uploaded-video session startup."""

import threading
import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.app.app import create_app
from src.app.core.settings import Settings
from src.app.services.session_manager import manager


@pytest.fixture(autouse=True)
def clear_sessions():
    """Keep global session manager state isolated between upload tests."""
    manager._sessions.clear()
    yield
    manager._sessions.clear()


@pytest.fixture
def client(tmp_path):
    """Provide a FastAPI test client with uploads stored under tmp_path."""
    app = create_app(
        Settings(
            upload_dir=tmp_path / "uploads",
            database_url="sqlite:///:memory:",
        )
    )
    return TestClient(app)


def _session_paths(tmp_path):
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.touch()
    config_path = tmp_path / "config.yml"
    config_path.write_text("pipeline: {}\n", encoding="utf-8")
    return checkpoint_path, config_path


def test_upload_rejects_non_mp4_files(client):
    response = client.post(
        "/api/videos/upload",
        files={"file": ("clip.txt", b"not a video", "text/plain")},
    )

    assert response.status_code == 400
    assert "Only .mp4" in response.json()["detail"]


def test_upload_stores_mp4_under_uploads_directory(client):
    response = client.post(
        "/api/videos/upload",
        files={"file": ("operator_clip.mp4", b"fake mp4 bytes", "video/mp4")},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["original_filename"] == "operator_clip.mp4"
    assert data["filename"] == f"{data['video_id']}.mp4"
    assert data["size_bytes"] == len(b"fake mp4 bytes")

    upload_dir = client.app.state.settings.upload_dir
    stored_path = upload_dir / data["filename"]
    assert stored_path.is_file()
    assert stored_path.read_bytes() == b"fake mp4 bytes"
    assert stored_path.resolve().is_relative_to(upload_dir.resolve())


@patch("src.app.services.session_manager.run_offline_mp4_inference")
def test_session_can_start_from_uploaded_video_id(mock_inference, client, tmp_path):
    pause_event = threading.Event()

    def mock_run(*args, **kwargs):
        pause_event.wait(timeout=2.0)

    mock_inference.side_effect = mock_run
    checkpoint_path, config_path = _session_paths(tmp_path)

    upload_response = client.post(
        "/api/videos/upload",
        files={"file": ("operator_clip.mp4", b"fake mp4 bytes", "video/mp4")},
    )
    assert upload_response.status_code == 201
    video_id = upload_response.json()["video_id"]

    response = client.post(
        "/api/sessions/",
        json={
            "video_id": video_id,
            "checkpoint_path": str(checkpoint_path),
            "config_path": str(config_path),
        },
    )

    assert response.status_code == 201
    session_id = response.json()["id"]
    assert response.json()["status"] == "running"

    session = manager._sessions[uuid.UUID(session_id)]
    assert session.request.video_path == (
        client.app.state.settings.upload_dir / f"{video_id}.mp4"
    ).resolve()

    pause_event.set()


def test_session_rejects_invalid_or_missing_uploaded_video_id(client, tmp_path):
    checkpoint_path, config_path = _session_paths(tmp_path)
    base_payload = {
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
    }

    traversal_response = client.post(
        "/api/sessions/",
        json={
            **base_payload,
            "video_id": "../operator_clip.mp4",
        },
    )
    assert traversal_response.status_code == 422

    missing_response = client.post(
        "/api/sessions/",
        json={
            **base_payload,
            "video_id": str(uuid.uuid4()),
        },
    )
    assert missing_response.status_code == 404
    assert "Uploaded video not found" in missing_response.json()["detail"]


def test_session_rejects_arbitrary_backend_video_paths(client, tmp_path):
    checkpoint_path, config_path = _session_paths(tmp_path)
    outside_video = tmp_path.parent / f"{uuid.uuid4()}.mp4"
    outside_video.write_bytes(b"outside allowed roots")

    response = client.post(
        "/api/sessions/",
        json={
            "video_path": str(outside_video),
            "checkpoint_path": str(checkpoint_path),
            "config_path": str(config_path),
        },
    )

    assert response.status_code == 400
    assert "configured backend video directory" in response.json()["detail"]

    traversal_response = client.post(
        "/api/sessions/",
        json={
            "video_path": "../operator_clip.mp4",
            "checkpoint_path": str(checkpoint_path),
            "config_path": str(config_path),
        },
    )
    assert traversal_response.status_code == 400
    assert "traversal" in traversal_response.json()["detail"]
