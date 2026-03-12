import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def set_seed(seed: int):
    random.seed(seed)


def generate_splits(video_paths: list[Path], splits_config: dict) -> list[dict]:
    by_class = {}
    for p in video_paths:
        label = p.parent.name
        by_class.setdefault(label, []).append(p)

    train_pct = splits_config.get('train', 0.7)
    val_pct = splits_config.get('val', 0.15)

    manifest_entries = []

    for label, paths in by_class.items():
        paths = sorted(paths)
        random.shuffle(paths)

        n = len(paths)
        n_train = int(n * train_pct)
        n_val = int(n * val_pct)

        for i, p in enumerate(paths):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"

            entry = {
                "video_id": p.stem,
                "path": f"{label}/{p.name}",
                "label": label,
                "split": split
            }
            manifest_entries.append(entry)

    return manifest_entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate dataset splits and manifest.")
    parser.add_argument("--config", default="configs/data_pipeline.yml", help="Path to YAML config")
    parser.add_argument("--output", default="manifest.jsonl", help="Output filename")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return 1

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config.get("pipeline", {}).get("seed", 42)
    set_seed(seed)

    raw_dir = Path(config.get("directories", {}).get("raw", "/app/data/raw"))
    manifests_dir = Path(config.get("directories", {}).get("manifests", "/app/data/manifests"))
    manifests_dir.mkdir(parents=True, exist_ok=True)

    output_path = manifests_dir / args.output

    if not raw_dir.exists():
        print(f"[ERROR] Raw directory not found: {raw_dir}")
        return 1

    video_paths = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]

    if not video_paths:
        print(f"[WARNING] No videos found in {raw_dir}. Please place dataset in data/raw.")
        return 0

    splits_config = config.get("splits", {"train": 0.7, "val": 0.15, "test": 0.15})
    manifest_entries = generate_splits(video_paths, splits_config)

    split_counts = Counter()
    class_split_counts = {}

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            split = entry["split"]
            label = entry["label"]

            split_counts[split] += 1
            if label not in class_split_counts:
                class_split_counts[label] = Counter()
            class_split_counts[label][split] += 1

    print(f"\n[OK] Manifest written to: {output_path}")
    print("\n--- Summary Stats ---")
    print(f"Total videos processed: {len(video_paths)}")
    print(f"Global Splits: {dict(split_counts)}")
    print("\nPer-Class Breakdown:")
    for label, counts in sorted(class_split_counts.items()):
        print(f"  - {label}: {dict(counts)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())