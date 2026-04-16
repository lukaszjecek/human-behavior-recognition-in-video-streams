import json

import src.main as app_main


def test_main_writes_summary(tmp_path, monkeypatch):
    subset_dir = tmp_path / "subset"
    class_dir = subset_dir / "walking"
    class_dir.mkdir(parents=True)
    (class_dir / "sample.mp4").write_bytes(b"")

    log_dir = tmp_path / "logs"

    monkeypatch.setattr(app_main, "DATA_DIR", subset_dir)
    monkeypatch.setattr(app_main, "LOG_DIR", log_dir)

    result = app_main.main([])

    assert result == 0

    summary_path = log_dir / "startup_summary.json"
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["video_count"] == 1
    assert summary["class_count"] == 1


def test_main_requires_input_and_checkpoint_together(tmp_path):
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"")

    result = app_main.main(["--input", str(video_path)])
    assert result == 2


def test_main_dispatches_inference_mode(monkeypatch, tmp_path):
    input_path = tmp_path / "sample.mp4"
    checkpoint_path = tmp_path / "model.pth"
    config_path = tmp_path / "config.yml"
    output_path = tmp_path / "actions.json"

    input_path.write_bytes(b"")
    checkpoint_path.write_bytes(b"")
    config_path.write_text("pipeline: {}", encoding="utf-8")

    captured = {}

    def _fake_runner(request):
        captured["request"] = request
        return 0

    monkeypatch.setattr(app_main, "run_mp4_to_json_action_inference", _fake_runner)

    result = app_main.main(
        [
            "--input",
            str(input_path),
            "--checkpoint",
            str(checkpoint_path),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ]
    )
    assert result == 0
    assert captured["request"].input_path == input_path
    assert captured["request"].checkpoint_path == checkpoint_path
    assert captured["request"].config_path == config_path
    assert captured["request"].output_path == output_path


def test_main_inference_returns_non_zero_on_failure(monkeypatch, tmp_path):
    input_path = tmp_path / "sample.mp4"
    checkpoint_path = tmp_path / "model.pth"

    input_path.write_bytes(b"")
    checkpoint_path.write_bytes(b"")

    def _failing_runner(_request):
        raise ValueError("bad inference config")

    monkeypatch.setattr(app_main, "run_mp4_to_json_action_inference", _failing_runner)

    result = app_main.main(
        [
            "--input",
            str(input_path),
            "--checkpoint",
            str(checkpoint_path),
        ]
    )
    assert result == 1
