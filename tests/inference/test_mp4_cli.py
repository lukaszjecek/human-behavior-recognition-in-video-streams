"""Tests for MP4 CLI inference helpers."""

import json

import pytest
import torch
import yaml

from src.inference.engine import InferenceResult
from src.inference.mp4_cli import (
    InferenceCliRequest,
    build_track_ids,
    load_model_from_checkpoint,
    load_runtime_settings,
    run_mp4_to_json_action_inference,
)
from src.models.dummy import DummyBehaviorModel


def _write_dummy_checkpoint(checkpoint_path):
    torch.manual_seed(1234)
    model = DummyBehaviorModel(num_classes=2)
    checkpoint = {
        "model_name": "dummy",
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


def test_load_model_from_checkpoint_rejects_invalid_payload(tmp_path):
    checkpoint_path = tmp_path / "bad_checkpoint.pth"
    torch.save(["not", "a", "dict"], str(checkpoint_path))

    with pytest.raises(TypeError, match="Checkpoint must contain"):
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
