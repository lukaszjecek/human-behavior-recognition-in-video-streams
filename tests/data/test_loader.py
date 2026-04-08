import json
import pytest
import torch
from src.data.loader import get_dataloader


@pytest.fixture
def mock_data(tmp_path, dummy_video):
    """Prepares a mini-manifest and a dummy video for testing."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    video_file = raw_dir / "walking" / "test.mp4"
    video_file.parent.mkdir()
    video_file.write_bytes(dummy_video.read_bytes())

    manifest_path = tmp_path / "manifest.jsonl"
    entry = {
        "video_id": "test_1",
        "path": "walking/test.mp4",
        "label": "walking",
        "split": "train"
    }
    with open(manifest_path, "w") as f:
        f.write(json.dumps(entry) + "\n")

    return manifest_path, raw_dir


def test_loader_batch_shape(mock_data):
    manifest_path, raw_dir = mock_data
    batch_size = 2

    loader = get_dataloader(manifest_path, raw_dir, split="train", batch_size=batch_size)

    for videos, labels in loader:
        assert videos.shape == (1, 16, 3, 224, 224)
        assert labels.shape == (1,)
        assert videos.dtype == torch.float32
        break