import cv2
import numpy as np
import torch
from typing import List, Tuple, Any


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
                         H and W are from target_resolution.

        Raises:
            ValueError: If frames list is empty or contains invalid frames.
        """
        if not frames:
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

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize to target resolution
            frame_resized = cv2.resize(frame_rgb, self.target_resolution)

            processed_frames.append(frame_resized)

        # Stack frames and normalize to [0.0, 1.0]
        frames_array = np.stack(
            processed_frames, axis=0).astype(np.float32) / 255.0

        # Convert to torch tensor
        # Current shape: [T, H, W, C]
        # Target shape: [B, T, C, H, W]
        tensor = torch.from_numpy(frames_array)

        # Permute from [T, H, W, C] to [T, C, H, W]
        tensor = tensor.permute(0, 3, 1, 2)

        # Add batch dimension: [T, C, H, W] -> [B, T, C, H, W]
        tensor = tensor.unsqueeze(0)

        return tensor
