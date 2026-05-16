import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.inference.service import InferenceServiceRequest, InferenceServiceResult
from src.inference.session import InferenceSession, SessionStatus


@pytest.fixture
def mock_request():
    return InferenceServiceRequest(
        checkpoint_path=Path("dummy.pth"),
        config_path=Path("dummy.yml"),
        video_path=Path("dummy.mp4"),
    )


@pytest.fixture
def mock_result():
    # We only need a MagicMock that acts as the result object
    return MagicMock(spec=InferenceServiceResult)


def test_session_initial_state(mock_request):
    session = InferenceSession(mock_request)
    assert session.status() == SessionStatus.IDLE
    assert session.result() is None


@patch("src.inference.session.run_inference")
def test_session_completes_successfully(mock_run_inference, mock_request, mock_result):
    mock_run_inference.return_value = mock_result
    
    session = InferenceSession(mock_request)
    session.start()
    
    # Wait for the thread to finish
    if session._thread:
        session._thread.join(timeout=2.0)
        
    assert session.status() == SessionStatus.FINISHED
    assert session.result() == mock_result
    mock_run_inference.assert_called_once()
    kwargs = mock_run_inference.call_args.kwargs
    assert "stop_event" in kwargs


@patch("src.inference.session.run_inference")
def test_session_stop_transitions_to_stopped(mock_run_inference, mock_request):
    # Make run_inference sleep so we can stop it
    def slow_inference(request, stop_event):
        while not stop_event.is_set():
            time.sleep(0.01)
        return None
        
    mock_run_inference.side_effect = slow_inference
    
    session = InferenceSession(mock_request)
    session.start()
    
    # Give it a moment to enter RUNNING state and start the mock
    time.sleep(0.05)
    assert session.status() == SessionStatus.RUNNING
    
    session.stop()
    assert session.status() == SessionStatus.STOPPED
    assert session.result() is None


@patch("src.inference.session.run_inference")
def test_session_error_transitions_to_error(mock_run_inference, mock_request):
    mock_run_inference.side_effect = RuntimeError("mocked failure")
    
    session = InferenceSession(mock_request)
    session.start()
    
    if session._thread:
        session._thread.join(timeout=2.0)
        
    assert session.status() == SessionStatus.ERROR
    assert session.result() is None


def test_session_stop_before_start_is_safe(mock_request):
    session = InferenceSession(mock_request)
    session.stop()
    assert session.status() == SessionStatus.IDLE


@patch("src.inference.session.run_inference")
def test_session_cannot_start_twice(mock_run_inference, mock_request, mock_result):
    mock_run_inference.return_value = mock_result
    
    session = InferenceSession(mock_request)
    session.start()
    
    with pytest.raises(RuntimeError, match="Cannot start session"):
        session.start()
        
    if session._thread:
        session._thread.join(timeout=2.0)
