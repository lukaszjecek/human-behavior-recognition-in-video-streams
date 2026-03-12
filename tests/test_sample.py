import json
import sys

import yaml

from src.data.sample import main


def test_sampler_is_deterministic_and_valid(tmp_path, monkeypatch):
    config_file = tmp_path / "data_pipeline.yml"
    raw_dir = tmp_path / "raw"
    manifests_dir = tmp_path / "manifests"

    class_a = raw_dir / "walking"
    class_a.mkdir(parents=True)

    for i in range(10):
        (class_a / f"video_{i}.mp4").write_bytes(b"")

    config_data = {
        "directories": {
            "raw": str(raw_dir),
            "manifests": str(manifests_dir)
        },
        "pipeline": {"seed": 123},
        "splits": {"train": 0.6, "val": 0.2, "test": 0.2}
    }
    config_file.write_text(yaml.dump(config_data), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["sample.py", "--config", str(config_file)])

    assert main() == 0
    manifest_path = manifests_dir / "manifest.jsonl"
    assert manifest_path.exists()

    lines_run1 = manifest_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines_run1) == 10

    first_entry = json.loads(lines_run1[0])
    assert all(k in first_entry for k in ["video_id", "path", "label", "split"])
    assert first_entry["label"] == "walking"

    manifest_path.unlink()
    assert main() == 0
    lines_run2 = manifest_path.read_text(encoding="utf-8").strip().split("\n")

    assert lines_run1 == lines_run2