"""Tests for CameraStreamSession bbox_hook wiring (issue #119)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import yaml

from src.app.services.camera_stream_manager import CameraStreamSession
from src.inference.pipeline import InferenceEventPipeline
from src.inference.runtime import InferenceRuntimeSettings
from src.models.dummy import DummyBehaviorModel


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


def _make_settings(*, bbox_enabled: bool = False) -> InferenceRuntimeSettings:
    return InferenceRuntimeSettings(
        target_resolution=(64, 64),
        window_size=4,
        stride=2,
        class_labels=["idle", "moving"],
        default_track_id=None,
        device="cpu",
        bbox_enabled=bbox_enabled,
        bbox_model_name="yolov8n.pt",
        bbox_confidence_threshold=0.4,
        bbox_weights_dir=None,
        bbox_frame_selector="middle",
    )


def _make_dummy_model() -> torch.nn.Module:
    return DummyBehaviorModel(num_classes=2)


@pytest.fixture()
def camera_session_setup(monkeypatch, tmp_path):
    """Provide dummy file paths and patch infrastructure so CameraStreamSession can init."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump({"pipeline": {"target_resolution": [64, 64], "temporal_window": 4}}))
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.write_bytes(b"dummy")

    # Allow any path through security check
    monkeypatch.setattr(
        "src.app.services.camera_stream_manager.validate_safe_path",
        lambda _path, _exts: True,
    )
    # Avoid real YAML parsing; return controlled settings
    monkeypatch.setattr(
        "src.app.services.camera_stream_manager.load_runtime_settings",
        lambda _p: _make_settings(bbox_enabled=False),
    )
    # Avoid real checkpoint loading
    monkeypatch.setattr(
        "src.app.services.camera_stream_manager.model_cache.get_model",
        lambda _ckpt, _dev: _make_dummy_model(),
    )
    # Suppress ContextModule init (already try/except in production code, but speeds up tests)
    monkeypatch.setattr(
        "src.inference.context_adapter.ContextModule.__init__",
        lambda self: (_ for _ in ()).throw(RuntimeError("disabled in test")),
        raising=False,
    )

    return checkpoint_path, config_path


def _spy_pipeline_init(captured: dict):
    """Return a patched InferenceEventPipeline.__init__ that records bbox_hook."""
    _real_init = InferenceEventPipeline.__init__

    def _patched(self, *args, **kwargs):
        captured["bbox_hook"] = kwargs.get("bbox_hook")
        _real_init(self, *args, **kwargs)

    return _patched


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_camera_session_bbox_disabled_by_default(monkeypatch, camera_session_setup):
    checkpoint_path, config_path = camera_session_setup

    captured: dict = {}
    monkeypatch.setattr(InferenceEventPipeline, "__init__", _spy_pipeline_init(captured))

    session = CameraStreamSession(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device="cpu",
    )

    assert session.bbox_hook is None
    assert captured.get("bbox_hook") is None


def test_camera_session_bbox_enabled_wires_hook(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yml"
    config_path.write_text("")
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.write_bytes(b"dummy")

    monkeypatch.setattr(
        "src.app.services.camera_stream_manager.validate_safe_path",
        lambda _path, _exts: True,
    )
    monkeypatch.setattr(
        "src.app.services.camera_stream_manager.load_runtime_settings",
        lambda _p: _make_settings(bbox_enabled=True),
    )
    monkeypatch.setattr(
        "src.app.services.camera_stream_manager.model_cache.get_model",
        lambda _ckpt, _dev: _make_dummy_model(),
    )

    fake_enricher = MagicMock()
    monkeypatch.setattr(
        "src.inference.bbox_detector.get_or_create_bbox_enricher",
        lambda **_kw: fake_enricher,
    )

    captured: dict = {}
    monkeypatch.setattr(InferenceEventPipeline, "__init__", _spy_pipeline_init(captured))

    session = CameraStreamSession(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device="cpu",
    )

    assert session.bbox_hook is fake_enricher
    assert captured.get("bbox_hook") is fake_enricher


def test_camera_session_bbox_initialization_failure_falls_back_to_none(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yml"
    config_path.write_text("")
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.write_bytes(b"dummy")

    monkeypatch.setattr(
        "src.app.services.camera_stream_manager.validate_safe_path",
        lambda _path, _exts: True,
    )
    monkeypatch.setattr(
        "src.app.services.camera_stream_manager.load_runtime_settings",
        lambda _p: _make_settings(bbox_enabled=True),
    )
    monkeypatch.setattr(
        "src.app.services.camera_stream_manager.model_cache.get_model",
        lambda _ckpt, _dev: _make_dummy_model(),
    )

    def _raise(**_kw):
        raise RuntimeError("ultralytics not available")

    monkeypatch.setattr("src.inference.bbox_detector.get_or_create_bbox_enricher", _raise)

    captured: dict = {}
    monkeypatch.setattr(InferenceEventPipeline, "__init__", _spy_pipeline_init(captured))

    session = CameraStreamSession(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device="cpu",
    )  # must not raise

    assert session.bbox_hook is None
    assert captured.get("bbox_hook") is None
