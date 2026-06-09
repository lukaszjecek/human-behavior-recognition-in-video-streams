"""Generate dataset split manifests from raw video directories."""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def set_seed(seed: int) -> None:
    """Seed random sampling for reproducible split generation."""
    random.seed(seed)


def validate_splits_config(splits_config: dict[str, Any]) -> None:
    """Validate train/val/test split percentages."""
    for split_name in ("train", "val", "test"):
        if split_name not in splits_config:
            raise ValueError(f"Missing split percentage: {split_name}")

        value = splits_config[split_name]
        if not isinstance(value, (int, float)):
            raise TypeError(f"Split percentage for {split_name} must be numeric")

        if value < 0:
            raise ValueError(f"Split percentage for {split_name} must be >= 0")

    total = sum(float(splits_config[name]) for name in ("train", "val", "test"))
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split percentages must sum to 1.0, got {total}")


def generate_splits(
    video_paths: list[Path],
    raw_dir: Path,
    splits_config: dict[str, Any],
) -> list[dict]:
    """Generate manifest entries grouped into train, validation, and test splits."""
    validate_splits_config(splits_config)

    by_class: dict[str, list[Path]] = {}
    for path in video_paths:
        label = path.parent.name
        by_class.setdefault(label, []).append(path)

    train_pct = float(splits_config.get("train", 0.7))
    val_pct = float(splits_config.get("val", 0.15))

    manifest_entries = []

    for label, paths in sorted(by_class.items()):
        paths = sorted(paths)
        random.shuffle(paths)

        n = len(paths)
        n_train = int(n * train_pct)
        n_val = int(n * val_pct)

        for i, path in enumerate(paths):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"

            relative_path = path.relative_to(raw_dir).as_posix()

            entry = {
                "video_id": path.stem,
                "path": relative_path,
                "label": label,
                "split": split,
            }
            manifest_entries.append(entry)

    return manifest_entries


def print_split_warnings(class_split_counts: dict[str, Counter]) -> None:
    """Print warnings for classes missing a split."""
    for label, counts in sorted(class_split_counts.items()):
        missing_splits = [split for split in ("train", "val", "test") if counts.get(split, 0) == 0]
        if missing_splits:
            print(
                f"[WARNING] Class '{label}' has no samples in split(s): "
                f"{', '.join(missing_splits)}"
            )


def main() -> int:
    """Generate a JSONL manifest from configured raw video directories."""
    parser = argparse.ArgumentParser(description="Generate dataset splits and manifest.")
    parser.add_argument(
        "--config",
        default="configs/data_pipeline.yml",
        help="Path to YAML config",
    )
    parser.add_argument("--output", default="manifest.jsonl", help="Output filename or path")
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Override raw video directory, e.g. /data/raw",
    )
    parser.add_argument(
        "--manifests-dir",
        default=None,
        help="Override manifests output directory, e.g. /data/manifests",
    )
    parser.add_argument(
        "--expected-classes",
        type=int,
        default=None,
        help="Fail if class count differs",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override split seed")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return 1

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = (
        args.seed
        if args.seed is not None
        else config.get("pipeline", {}).get("seed", 42)
    )
    set_seed(seed)

    raw_dir = (
        Path(args.raw_dir)
        if args.raw_dir is not None
        else Path(config.get("directories", {}).get("raw", "/app/data/raw"))
    )
    manifests_dir = (
        Path(args.manifests_dir)
        if args.manifests_dir is not None
        else Path(config.get("directories", {}).get("manifests", "/app/data/manifests"))
    )

    if Path(args.output).is_absolute():
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        manifests_dir.mkdir(parents=True, exist_ok=True)
        output_path = manifests_dir / args.output

    if not raw_dir.exists():
        print(f"[ERROR] Raw directory not found: {raw_dir}")
        return 1

    if not raw_dir.is_dir():
        print(f"[ERROR] Raw path is not a directory: {raw_dir}")
        return 1

    video_paths = [
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    ]

    if not video_paths:
        print(f"[WARNING] No videos found in {raw_dir}. Please place dataset in data/raw.")
        return 0

    labels = sorted({path.parent.name for path in video_paths})
    if args.expected_classes is not None and len(labels) != args.expected_classes:
        print(
            f"[ERROR] Expected {args.expected_classes} classes, "
            f"but found {len(labels)} classes in {raw_dir}"
        )
        print(f"Found labels: {labels}")
        return 1

    splits_config = config.get("splits", {"train": 0.7, "val": 0.15, "test": 0.15})
    manifest_entries = generate_splits(video_paths, raw_dir, splits_config)

    split_counts = Counter()
    class_split_counts: dict[str, Counter] = {}

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
    print(f"Raw directory: {raw_dir}")
    print(f"Seed: {seed}")
    print(f"Total videos processed: {len(video_paths)}")
    print(f"Class count: {len(labels)}")
    print(f"Global Splits: {dict(split_counts)}")
    print("\nPer-Class Breakdown:")
    for label, counts in sorted(class_split_counts.items()):
        print(f"  - {label}: {dict(counts)}")

    print_split_warnings(class_split_counts)

    return 0


if __name__ == "__main__":
    sys.exit(main())
