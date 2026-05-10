import pytest
import torch
import yaml

from src.inference.service import (
    InferenceServiceRequest,
    run_offline_mp4_inference,
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


def test_run_offline_mp4_inference_returns_typed_result(dummy_video, tmp_path):
    checkpoint_path = tmp_path / "dummy_checkpoint.pth"
    config_path = tmp_path / "inference.yml"

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


def test_run_offline_mp4_inference_rejects_non_mp4_video_path(tmp_path):
    video_path = tmp_path / "sample.avi"
    checkpoint_path = tmp_path / "dummy_checkpoint.pth"
    config_path = tmp_path / "inference.yml"

    video_path.write_bytes(b"not-an-mp4")
    _write_dummy_checkpoint(checkpoint_path)
    config_path.write_text("pipeline: {}", encoding="utf-8")

    request = InferenceServiceRequest(
        video_path=video_path,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )
    with pytest.raises(ValueError, match=r"\.mp4"):
        run_offline_mp4_inference(request)
