"""Validation module for the baseline behavior recognition model."""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

from src.data.loader import get_dataloader
from src.models.baseline import BaselineBehaviorModel


def main() -> int:
    """
    Main entrypoint for the validation script.
    Loads a checkpoint and evaluates it on the validation split.
    """
    parser = argparse.ArgumentParser(description="Validate a saved baseline checkpoint.")
    parser.add_argument("--config", default="configs/train.yml", help="Path to config")
    parser.add_argument("--checkpoint", required=True, help="Path to the .pth checkpoint file")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    manifest_path = Path(config["directories"]["manifests"]) / "manifest.jsonl"
    raw_dir = Path(config["directories"]["raw"])
    metrics_dir = Path(config["directories"]["logs"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    val_loader = get_dataloader(
        manifest_path=manifest_path,
        data_dir=raw_dir,
        split="val",
        batch_size=config["training"]["batch_size"],
        config_path=Path("configs/data_pipeline.yml")
    )

    classes = list(val_loader.dataset.label_to_idx.keys())
    model = BaselineBehaviorModel(num_classes=len(classes))
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total if total > 0 else 0.0
    
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "classes_evaluated": classes,
        "total_samples": total,
        "accuracy": round(accuracy, 4)
    }

    output_file = metrics_dir / "validation_summary.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    return 0


if __name__ == "__main__":
    sys.exit(main())