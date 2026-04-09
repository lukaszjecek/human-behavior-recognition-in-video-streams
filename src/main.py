import json
import sys
from os import getenv
from pathlib import Path

DATA_DIR = Path(getenv("DATA_DIR", "/app/data/raw"))
LOG_DIR = Path(getenv("LOG_DIR", "/app/data/logs"))

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_DIR.exists():
        print(f"[ERROR] DATA_DIR does not exist: {DATA_DIR}")
        return 1

    video_files = [p for p in DATA_DIR.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    class_dirs = sorted({p.parent.name for p in video_files})

    summary = {
        "data_dir": str(DATA_DIR),
        "video_count": len(video_files),
        "class_count": len(class_dirs),
        "classes_preview": class_dirs[:20],
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    summary_path = LOG_DIR / "startup_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Wrote summary to: {summary_path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())