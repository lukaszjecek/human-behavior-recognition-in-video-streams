"""Tests for shared frame preprocessing operations."""
import numpy as np
import pytest

from src.data.frame_ops import normalize_frames, preprocess_single_frame


def create_dummy_bgr_frame(height: int = 480, width: int = 640) -> np.ndarray:
    """Helper to create a dummy BGR frame."""
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


def test_preprocess_single_frame_basic():
    """Test basic frame preprocessing with BGR->RGB and resize."""
    frame = create_dummy_bgr_frame()
    target_resolution = (224, 224)

    result = preprocess_single_frame(frame, target_resolution)

    # Check output shape matches target resolution
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8


def test_preprocess_single_frame_bgr_to_rgb():
    """Test that BGR to RGB conversion is applied."""
    # Create frame with distinct BGR values
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :, 0] = 100  # Blue
    frame[:, :, 1] = 150  # Green
    frame[:, :, 2] = 200  # Red

    result = preprocess_single_frame(frame, (100, 100))

    # After BGR->RGB conversion:
    # result[:, :, 0] should be Red (originally frame[:, :, 2])
    # result[:, :, 1] should be Green (originally frame[:, :, 1])
    # result[:, :, 2] should be Blue (originally frame[:, :, 0])
    assert np.all(result[:, :, 0] == 200)  # Red channel
    assert np.all(result[:, :, 1] == 150)  # Green channel
    assert np.all(result[:, :, 2] == 100)  # Blue channel


def test_preprocess_single_frame_resize():
    """Test resizing to different resolutions."""
    frame = create_dummy_bgr_frame(480, 640)

    # Resize to smaller
    result = preprocess_single_frame(frame, (112, 112))
    assert result.shape == (112, 112, 3)

    # Resize to larger
    result = preprocess_single_frame(frame, (512, 512))
    assert result.shape == (512, 512, 3)

    # Non-square resize - verify width/height ordering
    # target_resolution=(320, 240) means width=320, height=240
    # Output shape should be (H=240, W=320, C=3)
    result = preprocess_single_frame(frame, (320, 240))
    assert result.shape == (240, 320, 3), \
        f"Expected shape (240, 320, 3) for target_resolution=(320, 240), got {result.shape}"
    
    # Test reverse: target_resolution=(240, 320) means width=240, height=320
    # Output shape should be (H=320, W=240, C=3)
    result = preprocess_single_frame(frame, (240, 320))
    assert result.shape == (320, 240, 3), \
        f"Expected shape (320, 240, 3) for target_resolution=(240, 320), got {result.shape}"
    
    # Test common video resolution: 640x480
    result = preprocess_single_frame(frame, (640, 480))
    assert result.shape == (480, 640, 3), \
        f"Expected shape (480, 640, 3) for target_resolution=(640, 480), got {result.shape}"


def test_preprocess_single_frame_dtype_validation():
    """Test dtype validation in preprocessing."""
    # Float frame should fail when validate_dtype=True
    float_frame = np.random.rand(100, 100, 3).astype(np.float32)

    with pytest.raises(ValueError, match="invalid dtype"):
        preprocess_single_frame(float_frame, (224, 224), validate_dtype=True)

    # Should pass when validate_dtype=False
    result = preprocess_single_frame(
        float_frame, (224, 224), validate_dtype=False)
    assert result.shape == (224, 224, 3)


def test_preprocess_single_frame_type_validation():
    """Test that non-array frame inputs raise a clear ValueError."""
    with pytest.raises(ValueError, match="Expected numpy.ndarray"):
        preprocess_single_frame("not-an-array", (224, 224))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_frame",
    [
        np.zeros((100, 100), dtype=np.uint8),
        np.zeros((100, 100, 1), dtype=np.uint8),
        np.zeros((100, 100, 4), dtype=np.uint8),
    ],
)
def test_preprocess_single_frame_shape_validation(invalid_frame: np.ndarray):
    """Test that invalid frame shapes/channels raise a clear ValueError."""
    with pytest.raises(ValueError, match=r"Expected \(H, W, 3\)"):
        preprocess_single_frame(invalid_frame, (224, 224))


@pytest.mark.parametrize(
    "invalid_resolution",
    [
        [224, 224],
        (224,),
        (224, 0),
        (-1, 224),
        (224.0, 224),
        ("224", 224),
    ],
)
def test_preprocess_single_frame_target_resolution_validation(invalid_resolution):
    """Test target resolution validation in preprocessing."""
    frame = create_dummy_bgr_frame()

    with pytest.raises(ValueError, match="tuple of two positive integers"):
        preprocess_single_frame(frame, invalid_resolution)  # type: ignore[arg-type]


def test_normalize_frames_uint8():
    """Test normalization of uint8 frames."""
    # Create frames with known values
    frames = np.array([
        np.zeros((100, 100, 3), dtype=np.uint8),  # All 0
        np.ones((100, 100, 3), dtype=np.uint8) * 255,  # All 255
        np.ones((100, 100, 3), dtype=np.uint8) * 127,  # Mid value
    ])

    result = normalize_frames(frames)

    assert result.dtype == np.float32
    assert result.shape == frames.shape

    # Check normalization
    assert np.allclose(result[0], 0.0)
    assert np.allclose(result[1], 1.0)
    assert np.allclose(result[2], 127.0 / 255.0, atol=1e-3)


def test_normalize_frames_float32():
    """Test that float32 frames are handled correctly."""
    # Already normalized frames
    frames = np.random.rand(5, 100, 100, 3).astype(np.float32)

    result = normalize_frames(frames)

    assert result.dtype == np.float32
    assert np.array_equal(result, frames)  # Should be unchanged


def test_normalize_frames_unsupported_float_dtype():
    """Test that unsupported float dtypes raise an explicit error."""
    frames = np.random.rand(5, 100, 100, 3).astype(np.float64)

    with pytest.raises(ValueError, match="Expected np.uint8 or np.float32"):
        normalize_frames(frames)


def test_normalize_frames_single_frame():
    """Test normalization of a single frame (3D array)."""
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 128

    result = normalize_frames(frame)

    assert result.dtype == np.float32
    assert result.shape == frame.shape
    assert np.allclose(result, 128.0 / 255.0, atol=1e-3)


def test_normalize_frames_range():
    """Test that normalized values are in [0.0, 1.0] range."""
    frames = np.random.randint(0, 256, (10, 50, 50, 3), dtype=np.uint8)

    result = normalize_frames(frames)

    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_preprocessing_pipeline_consistency():
    """Integration test: ensure consistent preprocessing pipeline."""
    frame = create_dummy_bgr_frame()
    target_resolution = (224, 224)

    # Step 1: Preprocess (BGR->RGB, resize)
    processed = preprocess_single_frame(frame, target_resolution)

    # Step 2: Normalize
    normalized = normalize_frames(processed)

    assert processed.shape == (224, 224, 3)
    assert processed.dtype == np.uint8
    assert normalized.shape == (224, 224, 3)
    assert normalized.dtype == np.float32
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
