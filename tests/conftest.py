import cv2
import numpy as np
import pytest


@pytest.fixture
def dummy_video(tmp_path):
    """Creates a 40-frame dummy video in a temporary directory."""
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