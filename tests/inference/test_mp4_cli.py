"""Tests for MP4 CLI inference helpers."""

import json

import numpy as np
import pytest
import torch
import yaml

from src.inference.engine import InferenceResult
from src.inference.mp4_cli import (
    InferenceCliRequest,
    WindowModelAdapter,
    _expand_batched_inference_results,
    build_track_ids,
    load_model_from_checkpoint,
    load_runtime_settings,
    resolve_inference_device,
    run_mp4_to_json_action_inference,
)
from src.inference.tensorize import FrameTensorizer
from src.models.dummy import DummyBehaviorModel


def _write_dummy_checkpoint(checkpoint_path):
    torch.manual_seed(1234)
    model = DummyBehaviorModel(num_classes=2)
    checkpoint = {
        "model_name": "dummy",
        "num_classes": 2,
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, str(checkpoint_path))


def test_run_mp4_to_json_action_inference_writes_output(dummy_video, tmp_path):
    checkpoint_path = tmp_path / "dummy_checkpoint.pth"
    config_path = tmp_path / "inference.yml"
    output_path = tmp_path / "actions.json"

    _write_dummy_checkpoint(checkpoint_path)

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

    request = InferenceCliRequest(
        input_path=dummy_video,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        output_path=output_path,
    )
    exit_code = run_mp4_to_json_action_inference(request)

    assert exit_code == 0
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["event_count"] > 0
    assert len(payload["events"]) == payload["event_count"]
    assert "confidence" in payload["events"][0]
    assert payload["events"][0]["track_id"] == 1


def test_load_runtime_settings_rejects_invalid_target_resolution(tmp_path):
    config_path = tmp_path / "invalid.yml"
    config_path.write_text("pipeline:\n  target_resolution: invalid\n", encoding="utf-8")

    with pytest.raises(TypeError, match="pipeline.target_resolution"):
        load_runtime_settings(config_path)


def test_load_runtime_settings_reads_device_override(tmp_path):
    config_path = tmp_path / "inference.yml"
    config_path.write_text(
        (
            "pipeline:\n"
            "  target_resolution: [64, 64]\n"
            "  temporal_window: 4\n"
            "inference:\n"
            "  stride: 1\n"
            "  device: mps\n"
        ),
        encoding="utf-8",
    )

    settings = load_runtime_settings(config_path)
    assert settings.device == "mps"


def test_load_model_from_checkpoint_rejects_invalid_payload(tmp_path):
    checkpoint_path = tmp_path / "bad_checkpoint.pth"
    torch.save(["not", "a", "dict"], str(checkpoint_path))

    with pytest.raises(TypeError, match="Checkpoint must contain"):
        load_model_from_checkpoint(checkpoint_path, torch.device("cpu"))


def test_load_model_from_checkpoint_requires_num_classes_metadata(tmp_path):
    checkpoint_path = tmp_path / "missing_num_classes.pth"
    model = DummyBehaviorModel(num_classes=2)
    torch.save(
        {
            "model_name": "dummy",
            "model_state_dict": model.state_dict(),
        },
        str(checkpoint_path),
    )

    with pytest.raises(TypeError, match="num_classes in checkpoint"):
        load_model_from_checkpoint(checkpoint_path, torch.device("cpu"))


def test_build_track_ids_rejects_invalid_results_type():
    with pytest.raises(TypeError, match="InferenceResult"):
        build_track_ids([object()], default_track_id=1)


def test_build_track_ids_supports_none_track_id():
    result = InferenceResult(
        window=tuple(),
        start_frame_index=0,
        end_frame_index=3,
        start_timestamp=0.0,
        end_timestamp=0.1,
        prediction=torch.tensor([0.1, 0.9]),
    )
    track_ids = build_track_ids([result], default_track_id=None)
    assert track_ids == [None]


def test_window_model_adapter_returns_full_batch_for_2d_output():
    class _BatchModel(torch.nn.Module):
        def forward(self, _: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[0.1, 0.9], [0.8, 0.2]], dtype=torch.float32)

    adapter = WindowModelAdapter(
        model=_BatchModel(),
        tensorizer=FrameTensorizer(target_resolution=(8, 8)),
        device=torch.device("cpu"),
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    prediction = adapter((frame,))

    assert prediction.shape == (2, 2)
    assert torch.allclose(
        prediction,
        torch.tensor([[0.1, 0.9], [0.8, 0.2]], dtype=torch.float32),
    )


def test_expand_batched_inference_results_splits_batch_prediction():
    result = InferenceResult(
        window=tuple(),
        start_frame_index=0,
        end_frame_index=3,
        start_timestamp=0.0,
        end_timestamp=0.1,
        prediction=torch.tensor([[0.1, 0.9], [0.8, 0.2]], dtype=torch.float32),
    )

    expanded = _expand_batched_inference_results([result])

    assert len(expanded) == 2
    assert torch.allclose(expanded[0].prediction, torch.tensor([0.1, 0.9]))
    assert torch.allclose(expanded[1].prediction, torch.tensor([0.8, 0.2]))


def test_resolve_inference_device_prefers_cli_over_config(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class _MPSBackend:
        @staticmethod
        def is_available():
            return True

    monkeypatch.setattr(torch.backends, "mps", _MPSBackend())
    device = resolve_inference_device(cli_device="cpu", config_device="mps")
    assert device.type == "cpu"


def test_resolve_inference_device_auto_uses_mps_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class _MPSBackend:
        @staticmethod
        def is_available():
            return True

    monkeypatch.setattr(torch.backends, "mps", _MPSBackend())
    device = resolve_inference_device(cli_device=None, config_device=None)
    assert device.type == "mps"
