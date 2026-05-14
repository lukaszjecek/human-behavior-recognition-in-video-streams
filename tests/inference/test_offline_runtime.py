import cv2
import numpy as np
import pytest

from src.inference.engine import InferenceEngine
from src.inference.offline_runtime import run_source, run_video
from src.inference.source_adapters import RtspSourceAdapter


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


def test_run_source_processes_rtsp_adapter_frames(monkeypatch):
    frames = [np.full((64, 64, 3), fill_value=i, dtype=np.uint8) for i in range(10)]
    captured = {}

    class _FakeCapture:
        def __init__(self, source_ref):
            self._source_ref = source_ref
            self._remaining_frames = list(frames)
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            if not self._remaining_frames:
                return False, None
            return True, self._remaining_frames.pop(0)

        def release(self):
            self.released = True

    def _fake_video_capture(source_ref):
        capture = _FakeCapture(source_ref)
        captured["source_ref"] = source_ref
        captured["capture"] = capture
        return capture

    monkeypatch.setattr("src.inference.source_adapters.cv2.VideoCapture", _fake_video_capture)

    adapter = RtspSourceAdapter(rtsp_uri="rtsp://localhost:8554/live")
    engine = InferenceEngine(window_size=4, stride=2, model=DummyPredictionModel())

    processed_frames, inference_windows, inference_results, action_events = run_source(
        source_adapter=adapter,
        engine=engine,
        emit_runtime_summary=False,
    )

    assert captured["source_ref"] == "rtsp://localhost:8554/live"
    assert captured["capture"].released is True
    assert processed_frames == len(frames)
    assert inference_windows > 0
    assert len(inference_results) == inference_windows
    assert len(action_events) == inference_windows
    assert all(event.track_id == 1 for event in action_events)
