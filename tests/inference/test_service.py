from threading import Event

import numpy as np
import pytest
import torch
import yaml

from src.inference.service import (
    InferenceServiceRequest,
    _build_request_source_adapter,
    run_inference,
    run_offline_mp4_inference,
)
from src.inference.source_adapters import FileSourceAdapter, RtspSourceAdapter
from src.models.dummy import DummyBehaviorModel


def _write_dummy_checkpoint(checkpoint_path):
    torch.manual_seed(1234)
    model = DummyBehaviorModel(num_classes=2)
    checkpoint = {
        "model_name": "dummy",
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, str(checkpoint_path))


def _write_inference_config(config_path):
    config = {
        "pipeline": {
            "target_resolution": [64, 64],
            "temporal_window": 4,
        },
        "inference": {
            "stride": 2,
            "class_labels": ["idle", "moving"],
        },
        "tracking": {
            "default_track_id": 1,
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_run_offline_mp4_inference_returns_typed_result(dummy_video, tmp_path):
    checkpoint_path = tmp_path / "dummy_checkpoint.pth"
    config_path = tmp_path / "inference.yml"

    _write_dummy_checkpoint(checkpoint_path)
    _write_inference_config(config_path)

    request = InferenceServiceRequest(
        video_path=dummy_video,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )

    result = run_offline_mp4_inference(request)

    assert result.frame_count == 40
    assert result.inference_count > 0
    assert len(result.inference_results) > 0
    assert result.event_count > 0
    assert len(result.action_events) == result.event_count
    assert result.action_events[0].track_id == 1


def test_run_inference_processes_rtsp_frames_with_runtime(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "dummy_checkpoint.pth"
    config_path = tmp_path / "inference.yml"
    _write_dummy_checkpoint(checkpoint_path)
    _write_inference_config(config_path)

    frames = [np.full((64, 64, 3), fill_value=i * 10, dtype=np.uint8) for i in range(10)]
    capture_state = {}

    class _FakeCapture:
        def __init__(self, source_ref):
            self._frames = list(frames)
            self.released = False
            self.source_ref = source_ref

        def isOpened(self):
            return True

        def read(self):
            if not self._frames:
                import time
                time.sleep(0.1)
                stop.set()
                return False, None
            return True, self._frames.pop(0)

        def release(self):
            self.released = True

    stop = Event()
    def _fake_video_capture(source_ref):
        capture = _FakeCapture(source_ref)
        capture_state["source_ref"] = source_ref
        capture_state["capture"] = capture
        
        return capture

    monkeypatch.setattr("src.inference.source_adapters.cv2.VideoCapture", _fake_video_capture)

    request = InferenceServiceRequest(
        video_path=None,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        source_type="rtsp",
        source_uri="rtsp://camera.local/stream",
    )
    result = run_inference(request, stop_event=stop)

    assert capture_state["source_ref"] == "rtsp://camera.local/stream"
    assert capture_state["capture"].released is True
    assert result.frame_count == len(frames)
    assert result.inference_count > 0
    assert len(result.inference_results) > 0
    assert result.event_count > 0
    assert all(event.track_id == 1 for event in result.action_events)


def test_build_request_source_adapter_accepts_non_mp4_file_path(tmp_path):
    video_path = tmp_path / "sample.avi"
    video_path.write_bytes(b"")

    request = InferenceServiceRequest(
        video_path=video_path,
        checkpoint_path=tmp_path / "dummy_checkpoint.pth",
        config_path=tmp_path / "inference.yml",
    )

    adapter = _build_request_source_adapter(request)

    assert isinstance(adapter, FileSourceAdapter)
    assert adapter.source_ref == str(video_path)


def test_build_request_source_adapter_creates_rtsp_adapter(tmp_path):
    request = InferenceServiceRequest(
        video_path=None,
        checkpoint_path=tmp_path / "dummy_checkpoint.pth",
        config_path=tmp_path / "inference.yml",
        source_type="rtsp",
        source_uri="rtsp://camera.local/stream",
    )

    adapter = _build_request_source_adapter(request)

    assert isinstance(adapter, RtspSourceAdapter)
    assert adapter.source_ref == "rtsp://camera.local/stream"


def test_run_inference_rejects_rtsp_request_without_uri(tmp_path):
    request = InferenceServiceRequest(
        video_path=None,
        checkpoint_path=tmp_path / "dummy_checkpoint.pth",
        config_path=tmp_path / "inference.yml",
        source_type="rtsp",
    )

    with pytest.raises(ValueError, match="request.source_uri"):
        run_inference(request)


def test_run_offline_mp4_inference_rejects_rtsp_request(tmp_path):
    request = InferenceServiceRequest(
        video_path=None,
        checkpoint_path=tmp_path / "dummy_checkpoint.pth",
        config_path=tmp_path / "inference.yml",
        source_type="rtsp",
        source_uri="rtsp://camera.local/stream",
    )

    with pytest.raises(ValueError, match="supports only file sources"):
        run_offline_mp4_inference(request)


def test_run_offline_mp4_inference_rejects_non_video_input(tmp_path):
    request = InferenceServiceRequest(
        video_path=tmp_path / "sample.txt",
        checkpoint_path=tmp_path / "dummy_checkpoint.pth",
        config_path=tmp_path / "inference.yml",
    )

    with pytest.raises(ValueError, match="supported video extension"):
        run_offline_mp4_inference(request)


def test_run_offline_mp4_inference_rejects_source_uri(tmp_path):
    request = InferenceServiceRequest(
        video_path=tmp_path / "sample.mp4",
        checkpoint_path=tmp_path / "dummy_checkpoint.pth",
        config_path=tmp_path / "inference.yml",
        source_uri="sample.mp4",
    )

    with pytest.raises(ValueError, match="does not accept request.source_uri"):
        run_offline_mp4_inference(request)
