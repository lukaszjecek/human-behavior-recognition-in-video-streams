import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


def get_video_info(video_path: Path):
    """Probes video file for metadata using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_pipeline.yml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    manifest_path = Path(config["directories"]["manifests"]) / "manifest.jsonl"
    raw_dir = Path(config["directories"]["raw"])
    report_dir = Path("/app/data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading manifest from {manifest_path}...")
    data = []
    with open(manifest_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)

    print("Probing video files for metadata (this may take a while)...")
    meta_list = []
    for path in df["path"]:
        meta = get_video_info(raw_dir / path)
        meta_list.append(meta if meta else {})

    meta_df = pd.DataFrame(meta_list)
    df = pd.concat([df, meta_df], axis=1)

    plt.figure(figsize=(12, 15))
    sns.countplot(data=df, y="label", hue="split")
    plt.title("Class Distribution per Split")
    plt.savefig(report_dir / "class_distribution.png")

    plt.figure(figsize=(10, 6))
    sns.histplot(df["duration"], bins=30, kde=True)
    plt.title("Video Duration Distribution (seconds)")
    plt.xlabel("Duration (s)")
    plt.savefig(report_dir / "duration_distribution.png")

    stats = {
        "total_samples": len(df),
        "avg_duration": df["duration"].mean(),
        "min_duration": df["duration"].min(),
        "max_duration": df["duration"].max(),
        "common_resolutions": df.groupby(["width", "height"]).size().to_dict(),
        "avg_fps": df["fps"].mean()
    }

    with open(report_dir / "eda_report.json", "w") as f:
        json.dump(stats, f, indent=4)

    print("\n--- EDA Complete ---")
    print(f"Artifacts saved to {report_dir}")
    print(f"Average Duration: {stats['avg_duration']:.2f}s")
    print(f"Resolution breakdown: {stats['common_resolutions']}")


if __name__ == "__main__":
    main()