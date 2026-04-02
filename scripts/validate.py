import argparse
import json
from pathlib import Path
import yaml
import torch

from src.data.loader import get_dataloader
from src.models.baseline import BaselineBehaviorModel

def main():
    parser = argparse.ArgumentParser(description="Validate saved checkpoint")
    parser.add_argument("--config", default="config/train.yml", help="Path to cfg")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pth file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Validation running at:{device}")

    manifest_path = Path(config["directories"]["manifests"]) / "manifest.jsonl"
    raw_dir = Path(config["directories"]["raw"])
    logs_dir = Path(config["directories"]["logs"])

    val_loader = get_dataloader(
        manifest_path=manifest_path,
        data_dir=raw_dir,
        split="val",
        batch_size=config["training"]["batch_size"],
        config_path=Path("config/data_pipeline.yml")
    )

    classes = list(val_loader.dataset.label_to_idx.keys())
    num_classes = len(classes)

    model = BaselineBehaviorModel(num_classes=num_classes)

    # DoD - loading data from checkpoint
    checkpoint = torch.load(args.chechpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    print("Evaluation time")
    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total if total > 0 else 0.0

    # Savin summary
    summary = {
        "checkpoint_path": args.checkpoint,
        "classes_evaluated": classes,
        "total_validation_samples": total,
        "accuracy": round(accuracy, 4)
    }

    summary_file = logs_dir / "validation_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print('\n VALIDATION COMPLETED')
    print(f'Classes: {classes}')
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Saved to file: {summary_file}')

if __name__ == "__main__":
    main()