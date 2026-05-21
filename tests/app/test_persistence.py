"""Automated tests for database persistence layer, repositories, and routes."""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.app.app import create_app
from src.app.core.settings import settings
from src.app.db.models import Base
from src.app.db.repository import get_event_by_id, get_events, save_event
from src.app.db.session import get_db
from src.app.schemas.action_event import (
    ActionEvent,
    AlertData,
    EventPayload,
    EventType,
)


@pytest.fixture(name="test_db")
def fixture_test_db():
    """Set up an isolated in-memory SQLite database for testing."""
    from src.app.db import session

    # Force SQLite in-memory URL in settings for lazy initialization
    original_url = settings.database_url
    settings.database_url = "sqlite:///:memory:"

    # Reset global engine in session module to force recreation with SQLite
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


@pytest.fixture(name="sample_detection_payload")
def fixture_sample_detection_payload():
    """Provides a sample DETECTION event payload."""
    evt = ActionEvent(
        start_frame_index=0,
        end_frame_index=15,
        label="running",
        confidence=0.92,
        start_timestamp=0.0,
        end_timestamp=0.5,
    )
    return EventPayload(
        camera_id="cam_01",
        event_type=EventType.DETECTION,
        data=evt,
    )


@pytest.fixture(name="sample_alert_payload")
def fixture_sample_alert_payload(sample_detection_payload):
    """Provides a sample ALERT event payload."""
    alert = AlertData(
        severity="HIGH",
        message="Running behavior detected",
        action_event=sample_detection_payload.data,
    )
    return EventPayload(
        camera_id="cam_01",
        event_type=EventType.ALERT,
        data=alert,
    )


def test_repository_save_and_retrieve_detection(test_db, sample_detection_payload):
    """Verify that a detection event payload is successfully saved and retrieved."""
    # 1. Save event
    db_evt = save_event(test_db, sample_detection_payload)
    assert db_evt.event_id == sample_detection_payload.event_id
    assert db_evt.event_type == "DETECTION"
    assert db_evt.camera_id == "cam_01"

    # 2. Get event by ID
    retrieved = get_event_by_id(test_db, sample_detection_payload.event_id)
    assert retrieved is not None
    assert retrieved.event_id == db_evt.event_id
    assert retrieved.payload["camera_id"] == "cam_01"
    assert retrieved.payload["event_type"] == "DETECTION"


def test_repository_save_and_retrieve_alert(test_db, sample_alert_payload):
    """Verify that an alert event payload is successfully saved and retrieved."""
    # 1. Save event
    db_evt = save_event(test_db, sample_alert_payload)
    assert db_evt.event_id == sample_alert_payload.event_id
    assert db_evt.event_type == "ALERT"

    # 2. Get event by ID
    retrieved = get_event_by_id(test_db, sample_alert_payload.event_id)
    assert retrieved is not None
    assert retrieved.event_type == "ALERT"
    assert retrieved.payload["data"]["severity"] == "HIGH"


def test_repository_query_filtering_and_sorting(
    test_db, sample_detection_payload, sample_alert_payload
):
    """Verify filtering, sorting, and pagination in get_events repository helper."""
    # Modify timestamps to ensure deterministic ordering
    sample_detection_payload.timestamp = datetime(2026, 5, 21, 10, 0, 0, tzinfo=timezone.utc)
    sample_alert_payload.timestamp = datetime(2026, 5, 21, 11, 0, 0, tzinfo=timezone.utc)

    save_event(test_db, sample_detection_payload)
    save_event(test_db, sample_alert_payload)

    # 1. Query all (newest first)
    events = get_events(test_db)
    assert len(events) == 2
    assert events[0].event_id == sample_alert_payload.event_id  # Newest first
    assert events[1].event_id == sample_detection_payload.event_id

    # 2. Filter by type
    detections = get_events(test_db, event_type="DETECTION")
    assert len(detections) == 1
    assert detections[0].event_id == sample_detection_payload.event_id

    alerts = get_events(test_db, event_type="ALERT")
    assert len(alerts) == 1
    assert alerts[0].event_id == sample_alert_payload.event_id

    # 3. Pagination
    paginated = get_events(test_db, limit=1, offset=1)
    assert len(paginated) == 1
    assert paginated[0].event_id == sample_detection_payload.event_id


def test_repository_write_validation_failures(test_db):
    """Verify validation triggers and error handling in save_event."""
    # Test typing error
    with pytest.raises(TypeError):
        save_event(test_db, "not-a-payload")  # type: ignore

    # Test event_id missing (None event_id)
    evt = ActionEvent(
        start_frame_index=0,
        end_frame_index=5,
        label="walking",
        confidence=0.8,
    )
    bad_payload = EventPayload(
        camera_id="cam_01",
        event_type=EventType.DETECTION,
        data=evt,
    )
    bad_payload.event_id = None  # type: ignore
    with pytest.raises(ValueError):
        save_event(test_db, bad_payload)


def test_api_read_history_endpoints(
    client, test_db, sample_detection_payload, sample_alert_payload
):
    """Test the history endpoints GET /api/events/ and GET /api/events/{id}."""
    # 1. Initially empty
    resp = client.get("/api/events/")
    assert resp.status_code == 200
    assert resp.json() == []

    # Save to test_db
    save_event(test_db, sample_detection_payload)
    save_event(test_db, sample_alert_payload)

    # 2. Get history list
    resp = client.get("/api/events/")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    # Returned as list of EventPayload dicts
    assert data[0]["event_id"] == str(sample_alert_payload.event_id)
    assert data[1]["event_id"] == str(sample_detection_payload.event_id)

    # 3. Get with filters
    resp = client.get("/api/events/?event_type=DETECTION")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["event_id"] == str(sample_detection_payload.event_id)

    # 4. Get by ID
    resp = client.get(f"/api/events/{sample_detection_payload.event_id}")
    assert resp.status_code == 200
    assert resp.json()["event_id"] == str(sample_detection_payload.event_id)

    # Get non-existent
    resp = client.get("/api/events/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


@patch("src.app.services.session_manager.run_offline_mp4_inference")
def test_pipeline_integration_saves_events(
    mock_inference, client, test_db, tmp_path, sample_detection_payload
):
    """Test that events produced by the inference pipeline are persisted to database."""
    # Mock the pipeline execution to emit a single detection event
    def mock_run_inference(request, stop_event, on_event):
        if on_event:
            on_event(sample_detection_payload)

    mock_inference.side_effect = mock_run_inference

    # Touch test files
    video_file = tmp_path / "video.mp4"
    video_file.touch()
    ckpt_file = tmp_path / "model.pth"
    ckpt_file.touch()
    config_file = tmp_path / "config.yml"
    config_file.touch()

    # Start session
    payload = {
        "video_path": str(video_file),
        "checkpoint_path": str(ckpt_file),
        "config_path": str(config_file),
    }
    response = client.post("/api/sessions/", json=payload)
    assert response.status_code == 201
    session_id = response.json()["id"]
    assert session_id is not None

    # Retrieve from endpoints and assert event has been saved
    resp = client.get("/api/events/")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["event_id"] == str(sample_detection_payload.event_id)
    assert data[0]["camera_id"] == "cam_01"
