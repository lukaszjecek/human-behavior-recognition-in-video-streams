"""
Shared preprocessing utilities for video frames.

This module contains common preprocessing operations used by both
offline (VideoPreprocessor) and online (FrameTensorizer) pipelines
to ensure consistency.
"""
import cv2
import numpy as np
from typing import Tuple


def preprocess_single_frame(
    frame: np.ndarray,
    target_resolution: Tuple[int, int],
    validate_dtype: bool = True
) -> np.ndarray:
    """
    Preprocess a single BGR frame for model input.

    Pipeline:
    1. Validate frame shape and channel count
    2. Validate target resolution
    3. Validate frame dtype (optional)
    4. Convert BGR to RGB
    5. Resize to target resolution

    Args:
        frame: BGR frame (numpy array) from OpenCV with dtype uint8.
        target_resolution: Target (width, height) for resizing as positive integers.
        validate_dtype: If True, validates that frame is uint8.

    Returns:
        np.ndarray: Preprocessed RGB frame in uint8, shape (H, W, 3).
    
    Raises:
        ValueError: If frame shape/channels are invalid, target_resolution is invalid,
                   or dtype is not uint8 (when validate_dtype=True).
    """
    if not isinstance(frame, np.ndarray):
        raise ValueError(
            f"Frame has invalid type {type(frame).__name__}. "
            "Expected numpy.ndarray"
        )

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"Frame has invalid shape {frame.shape}. "
            "Expected (H, W, 3)"
        )

    if (
        not isinstance(target_resolution, tuple)
        or len(target_resolution) != 2
        or any(
            not isinstance(dim, (int, np.integer))
            or isinstance(dim, bool)
            or dim <= 0
            for dim in target_resolution
        )
    ):
        raise ValueError(
            f"target_resolution has invalid value {target_resolution}. "
            "Expected tuple of two positive integers (width, height)"
        )

    if validate_dtype and frame.dtype != np.uint8:
        raise ValueError(
            f"Frame has invalid dtype {frame.dtype}. "
            "Expected np.uint8 for preprocessing"
        )

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize to target resolution
    frame_resized = cv2.resize(frame_rgb, target_resolution)

    return frame_resized


def normalize_frames(frames: np.ndarray) -> np.ndarray:
    """
    Normalize frame pixel values from [0, 255] to [0.0, 1.0].

    Args:
        frames: Array of frames with dtype uint8 or already float32.
               Shape can be (H, W, C) for single frame or (N, H, W, C) for batch.

    Returns:
        np.ndarray: Normalized frames with dtype float32, values in [0.0, 1.0].

    Raises:
        ValueError: If frames dtype is not np.uint8 or np.float32.
    """
    if frames.dtype == np.uint8:
        return frames.astype(np.float32) / 255.0
    if frames.dtype == np.float32:
        # Already float32, assume normalized or intentionally prepared by caller.
        return frames

    raise ValueError(
        f"Frames have invalid dtype {frames.dtype}. "
        "Expected np.uint8 or np.float32 for normalization"
    )
