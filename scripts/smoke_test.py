import argparse
import time
import sys
from pathlib import Path
import torch
import yaml

from src.data.loader import get_dataloader
from src.models.dummy import DummyBehaviorModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_pipeline.yml")
    parser.add_argument("--batches", type=int, default=3, help="Number of batches to test")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    manifest_path = Path(config["directories"]["manifests"]) / "manifest.jsonl"
    raw_dir = Path(config["directories"]["raw"])

    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found at {manifest_path}. Run src.data.sample first.")
        sys.exit(1)

    print(f"--- Starting Smoke Test ---")
    start_time = time.time()

    loader = get_dataloader(
        manifest_path=manifest_path,
        data_dir=raw_dir,
        split="train",
        batch_size=2,
        config_path=Path(args.config)
    )

    num_classes = len(loader.dataset.label_to_idx)
    model = DummyBehaviorModel(num_classes=num_classes)
    model.eval()

    print(f"Detected {num_classes} classes in manifest.")
    print(f"Processing {args.batches} batches...")

    try:
        for i, (videos, labels) in enumerate(loader):
            if i >= args.batches:
                break

            with torch.no_grad():
                outputs = model(videos)

            print(f"Batch {i + 1}:")
            print(f"  - Input shape:  {list(videos.shape)} [B, T, C, H, W]")
            print(f"  - Output shape: {list(outputs.shape)} [B, num_classes]")
            print(f"  - Labels:       {labels.tolist()}")

        elapsed = time.time() - start_time
        print(f"\n--- SUCCESS ---")
        print(f"Timing: {elapsed:.2f}s total ({elapsed / args.batches:.2f}s per batch)")
        sys.exit(0)

    except Exception as e:
        print(f"\n--- FAILED ---")
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()