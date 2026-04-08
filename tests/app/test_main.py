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

    result = app_main.main()

    assert result == 0

    summary_path = log_dir / "startup_summary.json"
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["video_count"] == 1
    assert summary["class_count"] == 1