import cv2
import numpy as np
import torch
from pathlib import Path


class VideoPreprocessor:
    """
    Module responsible for consistent video preprocessing into tensors.

    Pipeline operations order:
    1. Decode: Video frame decoding via OpenCV (BGR -> RGB conversion).
    2. Resize: Scaling to `target_resolution` (W, H).
    3. Normalize: Casting pixel values from [0, 255] range to [0.0, 1.0].
    4. Windowing: Extracting temporal windows of length `T` frames with a given `stride`.
    5. Tensor shape: PyTorch conversion and axis permutation to [T, C, H, W].
    """
    def __init__(self, target_resolution: tuple = (224, 224), temporal_window: int = 16, stride: int = 16):
        self.target_resolution = target_resolution
        self.T = temporal_window
        self.stride = stride

    def process(self, video_path: Path) -> torch.Tensor:
        """
        Processes the entire video and returns extracted, model-ready windows.

        Returns:
        torch.Tensor: Tensor of shape [num_windows, T, C, H, W].
        A single window (before batching) strictly follows the [T, C, H, W] format.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.target_resolution)
            frames.append(frame)

        cap.release()

        if not frames:
            return torch.zeros((0, self.T, 3, self.target_resolution[1], self.target_resolution[0]))

        frames_np = np.array(frames, dtype=np.float32) / 255.0
        total_frames = len(frames_np)

        windows = []
        for start_idx in range(0, total_frames, self.stride):
            window = frames_np[start_idx: start_idx + self.T]

            if len(window) < self.T:
                if len(window) == 0:
                    continue
                pad_len = self.T - len(window)
                padding = np.repeat(window[-1:], pad_len, axis=0)
                window = np.concatenate([window, padding], axis=0)

            windows.append(window)

        if not windows:
            return torch.zeros((0, self.T, 3, self.target_resolution[1], self.target_resolution[0]))

        windows_np = np.stack(windows)

        tensor_windows = torch.from_numpy(windows_np).permute(0, 1, 4, 2, 3)

        return tensor_windows