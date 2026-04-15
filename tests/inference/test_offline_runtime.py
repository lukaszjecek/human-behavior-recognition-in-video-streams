import cv2
import numpy as np

from src.inference.offline_runtime import run_video

def test_run_video_processes_sample_mp4(tmp_path):

    video_path = tmp_path / "video.mp4"

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (64, 64),
    )

    assert writer.isOpened()

    for _ in range(20):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        writer.write(frame)

    writer.release()

    processed_frames, inference_windows = run_video(str(video_path))

    assert processed_frames == 20
    assert inference_windows > 0
