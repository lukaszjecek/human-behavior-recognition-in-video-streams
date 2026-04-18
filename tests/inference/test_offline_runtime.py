import cv2
import numpy as np
import pytest

from src.inference.engine import InferenceEngine
from src.inference.offline_runtime import run_video


class DummyPredictionModel:
    def __call__(self, window):
        return {"label": "action", "confidence": 0.9}

def test_run_video_raises_on_invalid_path_without_hanging():
    with pytest.raises(FileNotFoundError):
        run_video("path/that/does/not/exist.mp4")

def test_run_video_processes_sample_mp4_and_exposes_track_id(tmp_path):
    video_path = tmp_path / "sample.mp4"

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

    engine = InferenceEngine(model=DummyPredictionModel())

    processed_frames, inference_windows, inference_results, action_events = run_video(
        str(video_path),
        engine=engine,
    )

    assert processed_frames == 20
    assert inference_windows > 0
    assert len(inference_results) == inference_windows
    assert len(action_events) == inference_windows

    first_result = inference_results[0]
    assert first_result.start_frame_index >= 1
    assert first_result.end_frame_index >= first_result.start_frame_index
    assert first_result.start_timestamp is not None
    assert first_result.end_timestamp is not None
    assert first_result.end_timestamp >= first_result.start_timestamp

    assert all(event.track_id == 1 for event in action_events)
