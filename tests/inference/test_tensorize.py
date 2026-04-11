import numpy as np
import pytest
import torch

from src.inference.tensorize import FrameTensorizer


def create_dummy_bgr_frame(height: int = 480, width: int = 640, channels: int = 3) -> np.ndarray:
    """Helper function to create a dummy BGR frame."""
    return np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)


def test_tensorizer_initialization():
    """Test tensorizer initialization with valid and invalid parameters."""
    # Valid initialization
    tensorizer = FrameTensorizer(target_resolution=(224, 224))
    assert tensorizer.target_resolution == (224, 224)

    # Custom resolution
    tensorizer = FrameTensorizer(target_resolution=(112, 112))
    assert tensorizer.target_resolution == (112, 112)

    # Invalid: not a tuple
    with pytest.raises(ValueError, match="must be a tuple"):
        FrameTensorizer(target_resolution=[224, 224])

    # Invalid: wrong tuple length
    with pytest.raises(ValueError, match="must be a tuple"):
        FrameTensorizer(target_resolution=(224,))

    # Invalid: negative dimensions
    with pytest.raises(ValueError, match="must be positive"):
        FrameTensorizer(target_resolution=(224, -1))

    # Invalid: zero dimensions
    with pytest.raises(ValueError, match="must be positive"):
        FrameTensorizer(target_resolution=(0, 224))


def test_tensorize_empty_frames_list():
    """Test that tensorize raises ValueError for empty frames list."""
    tensorizer = FrameTensorizer()

    with pytest.raises(ValueError, match="cannot be empty"):
        tensorizer.tensorize([])


def test_tensorize_invalid_container_type():
    """Test that tensorize raises ValueError for non-list/tuple containers."""
    tensorizer = FrameTensorizer()
    
    # Passing a numpy array instead of a list should fail with clear error
    frames_array = np.random.randint(0, 256, (3, 480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="must be a list or tuple"):
        tensorizer.tensorize(frames_array)
    
    # Passing a string should also fail
    with pytest.raises(ValueError, match="must be a list or tuple"):
        tensorizer.tensorize("not a list")
    
    # Passing a dict should also fail
    with pytest.raises(ValueError, match="must be a list or tuple"):
        tensorizer.tensorize({})


def test_tensorize_none_frame():
    """Test that tensorize raises ValueError for None frames."""
    tensorizer = FrameTensorizer()
    frames = [create_dummy_bgr_frame(), None, create_dummy_bgr_frame()]

    with pytest.raises(ValueError, match="Frame at index 1 is None"):
        tensorizer.tensorize(frames)


def test_tensorize_invalid_frame_type():
    """Test that tensorize raises ValueError for non-numpy array frames."""
    tensorizer = FrameTensorizer()
    frames = [create_dummy_bgr_frame(), "not a frame",
              create_dummy_bgr_frame()]

    with pytest.raises(ValueError, match="not a numpy array"):
        tensorizer.tensorize(frames)


def test_tensorize_invalid_frame_shape():
    """Test that tensorize raises ValueError for frames with invalid shape."""
    tensorizer = FrameTensorizer()

    # Grayscale frame (H, W) instead of (H, W, C)
    gray_frame = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
    frames = [gray_frame]

    with pytest.raises(ValueError, match="invalid shape"):
        tensorizer.tensorize(frames)

    # Frame with wrong number of channels
    rgba_frame = np.random.randint(0, 256, (480, 640, 4), dtype=np.uint8)
    frames = [rgba_frame]

    with pytest.raises(ValueError, match="invalid shape"):
        tensorizer.tensorize(frames)


def test_tensorize_invalid_dtype():
    """Test that tensorize raises ValueError for non-uint8 frames."""
    tensorizer = FrameTensorizer()
    
    # Float32 frame (already normalized) - should fail
    float_frame = np.random.rand(480, 640, 3).astype(np.float32)
    frames = [float_frame]
    
    with pytest.raises(ValueError, match="invalid dtype.*Expected np.uint8"):
        tensorizer.tensorize(frames)
    
    # Float64 frame - should fail
    float64_frame = np.random.rand(480, 640, 3).astype(np.float64)
    frames = [float64_frame]
    
    with pytest.raises(ValueError, match="invalid dtype.*Expected np.uint8"):
        tensorizer.tensorize(frames)
    
    # Int32 frame - should fail
    int32_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.int32)
    frames = [int32_frame]
    
    with pytest.raises(ValueError, match="invalid dtype.*Expected np.uint8"):
        tensorizer.tensorize(frames)


def test_tensorize_preprocess_error_chains_original_exception(monkeypatch):
    """Test that preprocessing failures preserve original exception context."""
    tensorizer = FrameTensorizer()
    frame = create_dummy_bgr_frame()

    def _raise_preprocess_error(*args, **kwargs):
        raise RuntimeError("boom")

    import src.inference.tensorize as tensorize_module
    monkeypatch.setattr(
        tensorize_module,
        "preprocess_single_frame",
        _raise_preprocess_error,
    )

    with pytest.raises(ValueError, match="Failed to preprocess frame at index 0: boom") as exc_info:
        tensorizer.tensorize([frame])

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "boom"


def test_tensorize_output_shape():
    """Test that tensorize produces correct output shape [B, T, C, H, W]."""
    tensorizer = FrameTensorizer(target_resolution=(224, 224))

    # Single frame
    frames = [create_dummy_bgr_frame()]
    tensor = tensorizer.tensorize(frames)
    assert tensor.shape == (1, 1, 3, 224, 224)  # [B=1, T=1, C=3, H=224, W=224]

    # Multiple frames (16 frames - typical for video models)
    frames = [create_dummy_bgr_frame() for _ in range(16)]
    tensor = tensorizer.tensorize(frames)
    # [B=1, T=16, C=3, H=224, W=224]
    assert tensor.shape == (1, 16, 3, 224, 224)

    # Different number of frames
    frames = [create_dummy_bgr_frame() for _ in range(32)]
    tensor = tensorizer.tensorize(frames)
    assert tensor.shape == (1, 32, 3, 224, 224)


def test_tensorize_output_shape_custom_resolution():
    """Test tensorize with custom target resolution."""
    tensorizer = FrameTensorizer(target_resolution=(112, 112))

    frames = [create_dummy_bgr_frame() for _ in range(8)]
    tensor = tensorizer.tensorize(frames)

    assert tensor.shape == (1, 8, 3, 112, 112)


def test_tensorize_non_square_resolution():
    """
    Test tensorize with non-square resolution to verify width/height ordering.
    
    Critical test: Ensures that target_resolution=(width, height) correctly maps to
    output shape [B, T, C, H, W] where H=height and W=width.
    """
    # Test case 1: 320x240 (landscape)
    tensorizer = FrameTensorizer(target_resolution=(320, 240))
    frames = [create_dummy_bgr_frame() for _ in range(16)]
    tensor = tensorizer.tensorize(frames)
    
    # target_resolution=(320, 240) means width=320, height=240
    # Output should be [B=1, T=16, C=3, H=240, W=320]
    assert tensor.shape == (1, 16, 3, 240, 320), \
        f"Expected shape (1, 16, 3, 240, 320) but got {tensor.shape}"
    
    # Test case 2: 240x320 (portrait) - swapped dimensions
    tensorizer = FrameTensorizer(target_resolution=(240, 320))
    frames = [create_dummy_bgr_frame() for _ in range(8)]
    tensor = tensorizer.tensorize(frames)
    
    # target_resolution=(240, 320) means width=240, height=320
    # Output should be [B=1, T=8, C=3, H=320, W=240]
    assert tensor.shape == (1, 8, 3, 320, 240), \
        f"Expected shape (1, 8, 3, 320, 240) but got {tensor.shape}"
    
    # Test case 3: 640x480 (common video resolution)
    tensorizer = FrameTensorizer(target_resolution=(640, 480))
    frames = [create_dummy_bgr_frame() for _ in range(4)]
    tensor = tensorizer.tensorize(frames)
    
    # target_resolution=(640, 480) means width=640, height=480
    # Output should be [B=1, T=4, C=3, H=480, W=640]
    assert tensor.shape == (1, 4, 3, 480, 640), \
        f"Expected shape (1, 4, 3, 480, 640) but got {tensor.shape}"


def test_tensorize_dtype_and_normalization():
    """Test that tensorize produces float32 dtype and normalizes values to [0.0, 1.0]."""
    tensorizer = FrameTensorizer(target_resolution=(224, 224))

    # Create frames with known values
    frames = [create_dummy_bgr_frame() for _ in range(4)]
    tensor = tensorizer.tensorize(frames)

    # Check dtype
    assert tensor.dtype == torch.float32

    # Check value range (normalized to [0.0, 1.0])
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_tensorize_normalization_boundary_values():
    """Test normalization with known boundary pixel values."""
    tensorizer = FrameTensorizer(target_resolution=(224, 224))

    # Create a frame with all zeros (black)
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tensor = tensorizer.tensorize([black_frame])

    # All values should be 0.0 after normalization
    assert torch.allclose(tensor, torch.zeros_like(tensor), atol=1e-6)

    # Create a frame with all 255 (white)
    white_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    tensor = tensorizer.tensorize([white_frame])

    # All values should be 1.0 after normalization
    assert torch.allclose(tensor, torch.ones_like(tensor), atol=1e-6)


def test_tensorize_bgr_to_rgb_conversion():
    """Test that BGR to RGB conversion is applied correctly."""
    tensorizer = FrameTensorizer(target_resolution=(224, 224))

    # Create a frame with distinct BGR values
    # Blue channel = 100, Green channel = 150, Red channel = 200
    bgr_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bgr_frame[:, :, 0] = 100  # Blue
    bgr_frame[:, :, 1] = 150  # Green
    bgr_frame[:, :, 2] = 200  # Red

    tensor = tensorizer.tensorize([bgr_frame])

    # After BGR->RGB conversion:
    # tensor[0, 0, 0, :, :] should be Red channel (originally bgr_frame[:,:,2])
    # tensor[0, 0, 1, :, :] should be Green channel (originally bgr_frame[:,:,1])
    # tensor[0, 0, 2, :, :] should be Blue channel (originally bgr_frame[:,:,0])

    # Expected normalized values
    expected_red = 200 / 255.0
    expected_green = 150 / 255.0
    expected_blue = 100 / 255.0

    # Check channel values
    assert torch.allclose(tensor[0, 0, 0, :, :], torch.full(
        (224, 224), expected_red), atol=1e-2)
    assert torch.allclose(tensor[0, 0, 1, :, :], torch.full(
        (224, 224), expected_green), atol=1e-2)
    assert torch.allclose(tensor[0, 0, 2, :, :], torch.full(
        (224, 224), expected_blue), atol=1e-2)


def test_tensorize_resizing():
    """Test that frames are correctly resized to target resolution."""
    tensorizer = FrameTensorizer(target_resolution=(112, 112))

    # Create frames with different sizes
    frame1 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    frame2 = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
    frame3 = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)

    tensor = tensorizer.tensorize([frame1, frame2, frame3])

    # All frames should be resized to (112, 112)
    assert tensor.shape == (1, 3, 3, 112, 112)


def test_tensorize_maintains_temporal_order():
    """Test that temporal order of frames is preserved."""
    tensorizer = FrameTensorizer(target_resolution=(64, 64))

    # Create frames with distinct patterns
    frames = []
    for i in range(5):
        frame = np.ones((100, 100, 3), dtype=np.uint8) * (i * 50)
        frames.append(frame)

    tensor = tensorizer.tensorize(frames)

    # Check that temporal dimension preserves order
    # Frame 0 should be darker than frame 4
    assert tensor[0, 0, :, :, :].mean() < tensor[0, 4, :, :, :].mean()


def test_tensorize_multiple_calls_independence():
    """Test that multiple calls to tensorize are independent."""
    tensorizer = FrameTensorizer(target_resolution=(224, 224))

    frames1 = [create_dummy_bgr_frame() for _ in range(8)]
    frames2 = [create_dummy_bgr_frame() for _ in range(16)]

    tensor1 = tensorizer.tensorize(frames1)
    tensor2 = tensorizer.tensorize(frames2)

    assert tensor1.shape == (1, 8, 3, 224, 224)
    assert tensor2.shape == (1, 16, 3, 224, 224)

    # Tensors should be different (different random data)
    assert not torch.allclose(tensor1[0, :8], tensor2[0, :8])


def test_tensorize_integration_with_buffer():
    """Integration test: tensorize frames from a buffer-like structure."""
    from src.inference.buffer import FrameBuffer

    tensorizer = FrameTensorizer(target_resolution=(224, 224))
    buffer = FrameBuffer(window_size=16)

    # Fill buffer with frames
    for _ in range(16):
        buffer.append(create_dummy_bgr_frame())

    # Get frames from buffer and tensorize
    frames = buffer.get_window()
    tensor = tensorizer.tensorize(frames)

    assert tensor.shape == (1, 16, 3, 224, 224)
    assert tensor.dtype == torch.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0
