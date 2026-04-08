import numpy as np
import torch
from typing import List, Tuple, Any

from src.data.frame_ops import preprocess_single_frame, normalize_frames


class FrameTensorizer:
    """
    Converts buffered raw frames into model-ready tensors for real-time inference.

    Pipeline:
    1. Convert OpenCV BGR frames to RGB
    2. Resize frames to model input size
    3. Normalize pixel values from [0, 255] to [0.0, 1.0]
    4. Assemble output tensor in [B, T, C, H, W] format

    Where:
        B = Batch size (1 for single inference window)
        T = Temporal dimension (number of frames)
        C = Channels (3 for RGB)
        H = Height
        W = Width
    """

    def __init__(self, target_resolution: Tuple[int, int] = (224, 224)):
        """
        Initializes the tensorizer.

        Args:
            target_resolution: Target (width, height) for resizing frames.
                             Default is (224, 224) which is standard for many video models.
        """
        if not isinstance(target_resolution, tuple) or len(target_resolution) != 2:
            raise ValueError(
                "target_resolution must be a tuple of (width, height)")

        if target_resolution[0] <= 0 or target_resolution[1] <= 0:
            raise ValueError("target_resolution dimensions must be positive")

        self.target_resolution = target_resolution

    def tensorize(self, frames: List[Any]) -> torch.Tensor:
        """
        Converts a list of raw BGR frames into a model-ready tensor.

        Args:
            frames: List of BGR frames (numpy arrays) from OpenCV.

        Returns:
            torch.Tensor: Shape [B, T, C, H, W] where B=1, T=len(frames), C=3,
                         H=target_resolution[1] (height) and
                         W=target_resolution[0] (width). Note that
                         target_resolution is specified as (width, height)
                         to match OpenCV's resize dsize order.

        Raises:
            ValueError: If frames list is empty or contains invalid frames.
        """
        if not isinstance(frames, (list, tuple)):
            raise ValueError("frames must be a list or tuple of frames")

        if len(frames) == 0:
            raise ValueError("frames list cannot be empty")

        processed_frames = []

        for i, frame in enumerate(frames):
            if frame is None:
                raise ValueError(f"Frame at index {i} is None")

            if not isinstance(frame, np.ndarray):
                raise ValueError(f"Frame at index {i} is not a numpy array")

            if len(frame.shape) != 3 or frame.shape[2] != 3:
                raise ValueError(
                    f"Frame at index {i} has invalid shape {frame.shape}. "
                    "Expected (H, W, 3)"
                )

            if frame.dtype != np.uint8:
                raise ValueError(
                    f"Frame at index {i} has invalid dtype {frame.dtype}. "
                    "Expected np.uint8 for proper normalization"
                )

            # Use shared preprocessing: BGR->RGB, resize
            try:
                frame_processed = preprocess_single_frame(
                    frame,
                    self.target_resolution,
                    validate_dtype=False  # Already validated above
                )
                processed_frames.append(frame_processed)
            except Exception as e:
                raise ValueError(
                    f"Failed to preprocess frame at index {i}: {e}") from e

        # Stack frames and normalize to [0.0, 1.0] using shared utility
        frames_array = np.stack(processed_frames, axis=0)
        frames_array = normalize_frames(frames_array)

        # Convert to torch tensor
        # Current shape: [T, H, W, C]
        # Target shape: [B, T, C, H, W]
        tensor = torch.from_numpy(frames_array)

        # Permute from [T, H, W, C] to [T, C, H, W]
        tensor = tensor.permute(0, 3, 1, 2)

        # Add batch dimension: [T, C, H, W] -> [B, T, C, H, W]
        tensor = tensor.unsqueeze(0)

        return tensor
