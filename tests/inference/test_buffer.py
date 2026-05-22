import pytest

from src.inference.buffer import FrameBuffer


def test_empty_buffer():
    """Test the behavior of an empty buffer."""
    buf = FrameBuffer(window_size=16)

    assert buf.current_size == 0
    assert buf.is_full() is False
    assert buf.get_window() == []


def test_partial_buffer():
    """Test the behavior of a partially filled buffer."""
    buf = FrameBuffer(window_size=5)
    buf.append("frame_1")
    buf.append("frame_2")

    assert buf.current_size == 2
    assert buf.is_full() is False
    assert buf.get_window() == ["frame_1", "frame_2"]


def test_full_buffer_and_overflow():
    """Test buffer overflow and automatic removal of the oldest frames."""
    buf = FrameBuffer(window_size=3)
    buf.append("f1")
    buf.append("f2")
    buf.append("f3")

    assert buf.is_full() is True
    assert buf.get_window() == ["f1", "f2", "f3"]

    # Adding a new frame after the buffer is full (overflow - removes f1)
    buf.append("f4")
    assert buf.is_full() is True
    assert buf.get_window() == ["f2", "f3", "f4"]

    # Another overflow (removes f2)
    buf.append("f5")
    assert buf.get_window() == ["f3", "f4", "f5"]


def test_configurable_window_size():
    """Test window size configuration and input validation."""
    buf = FrameBuffer(window_size=32)
    assert buf.window_size == 32

    # Check if ValueError is raised for invalid input
    with pytest.raises(ValueError):
        FrameBuffer(window_size=0)

    with pytest.raises(ValueError):
        FrameBuffer(window_size=-5)


def test_clear_buffer():
    """Test clearing the buffer content."""
    buf = FrameBuffer(window_size=5)
    buf.append("frame_1")
    buf.append("frame_2")

    # Verify the buffer has elements before clearing
    assert buf.current_size == 2

    # Clear the buffer
    buf.clear()

    # Verify the state after clearing
    assert buf.current_size == 0
    assert buf.is_full() is False
    assert buf.get_window() == []
