import pytest
from src.inference.engine import InferenceEngine


class DummyModel:
    """Mock model to test the return triggering behavior."""
    pass


def test_inference_engine_initialization():
    """Test if the engine initializes correctly with buffer and stride."""
    engine = InferenceEngine(window_size=16, stride=4)

    assert engine.buffer.window_size == 16
    assert engine.stride == 4
    assert engine.frame_count == 0
    assert engine.model is None
    assert engine.get_latest_window() is None


def test_invalid_stride_value():
    """Test validation for stride parameter."""
    with pytest.raises(ValueError):
        InferenceEngine(stride=0)
    with pytest.raises(ValueError):
        InferenceEngine(stride=-2)


def test_stride_trigger_cadence():
    """Test if inference is triggered strictly every N frames after filling the buffer."""
    engine = InferenceEngine(window_size=3, stride=2, model=DummyModel())

    # Fill the buffer (first full window triggers inference)
    assert engine.process_frame("f1") is None
    assert engine.process_frame("f2") is None
    assert engine.process_frame("f3") == "prediction_stub"
    assert engine.get_latest_window() == ["f1", "f2", "f3"]

    # Next frame (stride = 1 since full) -> No trigger
    assert engine.process_frame("f4") is None
    # Still preserves the old window!
    assert engine.get_latest_window() == ["f1", "f2", "f3"]

    # Next frame (stride = 2 since full) -> Trigger!
    assert engine.process_frame("f5") == "prediction_stub"
    assert engine.get_latest_window() == ["f3", "f4", "f5"]

    # Check if frame counter correctly reached 5
    assert engine.frame_count == 5


def test_get_latest_window_determinism():
    """Verify that the retrieved window does not mutate while buffer progresses."""
    engine = InferenceEngine(window_size=2, stride=3, model=DummyModel())

    engine.process_frame("A")
    engine.process_frame("B")  # Triggers inference, window is ["A", "B"]

    saved_window = engine.get_latest_window()
    assert saved_window == ["A", "B"]

    engine.process_frame("C")  # No inference (stride wait)
    engine.process_frame("D")  # No inference (stride wait)

    # Buffer has moved forward ["C", "D"], but the latest inference window should be intact
    assert engine.buffer.get_window() == ["C", "D"]
    assert engine.get_latest_window() == saved_window
