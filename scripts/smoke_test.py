"""Smoke-test the video dataloader and baseline model forward pass."""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import get_dataloader  # noqa: E402
from src.models.baseline import BaselineBehaviorModel  # noqa: E402


def resolve_device(device_arg: str) -> torch.device:
    """Resolve smoke-test device from CLI argument."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")

    return torch.device(device_arg)


def resolve_data_paths(
    *,
    config: dict[str, Any],
    data_dir: str | None,
    raw_dir: str | None,
    manifests_dir: str | None,
) -> tuple[Path, Path]:
    """Resolve raw data and manifest paths from CLI overrides or YAML config."""
    if data_dir:
        base_data = Path(data_dir)
        resolved_raw_dir = base_data / "raw"
        resolved_manifest_path = base_data / "manifests" / "manifest.jsonl"
    else:
        directories = config.get("directories", {})
        resolved_raw_dir = Path(directories.get("raw", "/app/data/raw"))
        resolved_manifest_path = (
            Path(directories.get("manifests", "/app/data/manifests")) / "manifest.jsonl"
        )

    if raw_dir:
        resolved_raw_dir = Path(raw_dir)

    if manifests_dir:
        resolved_manifest_path = Path(manifests_dir) / "manifest.jsonl"

    return resolved_manifest_path, resolved_raw_dir


def main() -> int:
    """Run a short dataloader/model smoke test."""
    parser = argparse.ArgumentParser(description="Smoke-test dataset loading and model forward pass.")
    parser.add_argument("--config", default="configs/data_pipeline.yml", help="Path to YAML config")
    parser.add_argument("--data-dir", default=None, help="Override root data directory, e.g. /data")
    parser.add_argument("--raw-dir", default=None, help="Override raw video directory")
    parser.add_argument("--manifests-dir", default=None, help="Override manifest directory")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Manifest split")
    parser.add_argument("--batches", type=int, default=3, help="Number of batches to test")
    parser.add_argument("--batch-size", type=int, default=2, help="Smoke-test batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Device")
    args = parser.parse_args()

    if args.batches <= 0:
        raise ValueError("batches must be > 0")

    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")

    if args.num_workers < 0:
        raise ValueError("num-workers must be >= 0")

    if args.prefetch_factor <= 0:
        raise ValueError("prefetch-factor must be > 0")

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    manifest_path, raw_dir = resolve_data_paths(
        config=config,
        data_dir=args.data_dir,
        raw_dir=args.raw_dir,
        manifests_dir=args.manifests_dir,
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run src.data.sample first."
        )

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"
    persistent_workers = args.num_workers > 0

    print("--- Starting Smoke Test ---")
    print(f"Config: {config_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Raw data directory: {raw_dir}")
    print(f"Split: {args.split}")
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Batches requested: {args.batches}")
    print(f"DataLoader workers: {args.num_workers}")

    start_time = time.time()

    loader = get_dataloader(
        manifest_path=manifest_path,
        data_dir=raw_dir,
        split=args.split,
        batch_size=args.batch_size,
        config_path=config_path,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    num_classes = len(loader.dataset.label_to_idx)
    if num_classes == 0:
        raise ValueError(f"No classes detected in split: {args.split}")

    model = BaselineBehaviorModel(num_classes=num_classes)
    model.to(device)
    model.eval()

    print(f"Detected {num_classes} classes in manifest split.")
    print(f"Processing up to {args.batches} batches...")

    processed_batches = 0

    for batch_idx, (videos, labels) in enumerate(loader):
        if batch_idx >= args.batches:
            break

        videos = videos.to(device, non_blocking=pin_memory)
        labels = labels.to(device, non_blocking=pin_memory)

        with torch.no_grad():
            outputs = model(videos)

        print(f"Batch {batch_idx + 1}:")
        print(f"  - Input shape:  {list(videos.shape)} [B, T, C, H, W]")
        print(f"  - Output shape: {list(outputs.shape)} [B, num_classes]")
        print(f"  - Labels:       {labels.detach().cpu().tolist()}")

        processed_batches += 1

    if processed_batches == 0:
        raise ValueError("No batches produced by DataLoader")

    elapsed = time.time() - start_time
    print("\n--- SUCCESS ---")
    print(f"Processed batches: {processed_batches}")
    print(f"Timing: {elapsed:.2f}s total ({elapsed / processed_batches:.2f}s per batch)")

    return 0


if __name__ == "__main__":
    sys.exit(main())