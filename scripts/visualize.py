import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import yaml


def main():
    parser = argparse.ArgumentParser(description="Visual validation tool for dataset annotations.")
    parser.add_argument("--config", default="configs/data_pipeline.yml", help="Path to config file")
    parser.add_argument("--video_id", type=str, default=None,
                        help="Specific video_id to visualize. If empty, a random one is chosen.")
    parser.add_argument("--output", default="/app/data/logs/visualized_output.mp4",
                        help="Path to export the .mp4 result")
    args = parser.parse_args()


    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    manifest_path = Path(config["directories"]["manifests"]) / "manifest.jsonl"
    raw_dir = Path(config["directories"]["raw"])

    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found at {manifest_path}")
        sys.exit(1)

    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))

    if not entries:
        print("[ERROR] Manifest is empty.")
        sys.exit(1)

    if args.video_id:
        selected = next((e for e in entries if e["video_id"] == args.video_id), None)
        if not selected:
            print(f"[ERROR] video_id '{args.video_id}' not found in manifest.")
            sys.exit(1)
    else:
        selected = random.choice(entries)

    video_path = raw_dir / selected["path"]
    if not video_path.exists():
        print(f"[ERROR] Video file does not exist: {video_path}")
        sys.exit(1)

    label = selected.get("label", "Unknown")
    bboxes = selected.get("bboxes", {})

    print("--- Visualizer ---")
    print(f"Selected Video: {selected['video_id']}")
    print(f"Class Label: {label}")
    print(f"Source Path: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        text = f"Label: {label}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        cv2.rectangle(frame, (10, 10), (20 + text_width + 10, 10 + text_height + 20), (0, 0, 0), -1)
        cv2.putText(frame, text, (20, 10 + text_height + 10),
                    font, font_scale, (0, 255, 0), thickness)

        cv2.putText(frame, f"Frame: {frame_idx}", (20, 10 + text_height + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        frame_key = f"frame_{frame_idx}"
        if frame_key in bboxes:
            box = bboxes[frame_key]
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[2]), int(box[3]))

            cv2.rectangle(frame, p1, p2, (0, 0, 255), 3)
            cv2.putText(frame, "Target", (p1[0], p1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[OK] Rendered {frame_idx} frames.")
    print(f"[OK] Exported to: {out_path}")


if __name__ == "__main__":
    main()