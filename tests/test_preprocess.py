import cv2
import numpy as np
import pytest
import torch
from src.data.preprocess import VideoPreprocessor


@pytest.fixture
def dummy_video(tmp_path):
    video_path = tmp_path / "test_video.mp4"
    out = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,
        (320, 240)
    )
    for i in range(40):
        frame = np.full((240, 320, 3), i * 5, dtype=np.uint8)
        out.write(frame)
    out.release()
    return video_path


def test_preprocessor_shape_and_stride(dummy_video):
    preprocessor = VideoPreprocessor(target_resolution=(128, 128), temporal_window=16, stride=16)
    tensor = preprocessor.process(dummy_video)

    assert tensor.shape == (3, 16, 3, 128, 128)

    preprocessor_32 = VideoPreprocessor(target_resolution=(224, 224), temporal_window=32, stride=10)
    tensor_32 = preprocessor_32.process(dummy_video)

    assert tensor_32.shape == (4, 32, 3, 224, 224)


def test_preprocessor_normalization(dummy_video):
    preprocessor = VideoPreprocessor()
    tensor = preprocessor.process(dummy_video)

    assert tensor.dtype == torch.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0