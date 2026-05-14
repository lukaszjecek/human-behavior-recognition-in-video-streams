import pytest

from src.inference.source_adapters import (
    FileSourceAdapter,
    RtspSourceAdapter,
    build_source_adapter,
)


def test_build_source_adapter_creates_file_adapter(tmp_path):
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"")

    adapter = build_source_adapter(source_type="file", source_ref=video_path)

    assert isinstance(adapter, FileSourceAdapter)
    assert adapter.source_ref == str(video_path)


def test_build_source_adapter_creates_rtsp_adapter():
    adapter = build_source_adapter(
        source_type="rtsp",
        source_ref="rtsp://localhost:8554/live",
    )

    assert isinstance(adapter, RtspSourceAdapter)
    assert adapter.source_ref == "rtsp://localhost:8554/live"


def test_file_source_adapter_open_capture_uses_file_path(tmp_path, monkeypatch):
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"")
    captured = {}

    class _FakeCapture:
        pass

    def _fake_video_capture(source_ref):
        captured["source_ref"] = source_ref
        return _FakeCapture()

    monkeypatch.setattr("src.inference.source_adapters.cv2.VideoCapture", _fake_video_capture)

    adapter = FileSourceAdapter(video_path=video_path)
    capture = adapter.open_capture()

    assert isinstance(capture, _FakeCapture)
    assert captured["source_ref"] == str(video_path)


def test_rtsp_source_adapter_open_capture_uses_rtsp_uri(monkeypatch):
    captured = {}

    class _FakeCapture:
        pass

    def _fake_video_capture(source_ref):
        captured["source_ref"] = source_ref
        return _FakeCapture()

    monkeypatch.setattr("src.inference.source_adapters.cv2.VideoCapture", _fake_video_capture)

    adapter = RtspSourceAdapter(rtsp_uri="rtsps://localhost:8554/live")
    capture = adapter.open_capture()

    assert isinstance(capture, _FakeCapture)
    assert captured["source_ref"] == "rtsps://localhost:8554/live"


def test_rtsp_source_adapter_rejects_non_rtsp_uri():
    with pytest.raises(ValueError, match="rtsp_uri must start with rtsp:// or rtsps://"):
        RtspSourceAdapter(rtsp_uri="http://localhost/live")
