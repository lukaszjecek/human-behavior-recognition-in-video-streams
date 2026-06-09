"""Module for training the baseline behavior recognition model."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.amp import GradScaler, autocast

from src.data.loader import get_dataloader
from src.models.baseline import BaselineBehaviorModel


def resolve_device(device_arg: str) -> torch.device:
    """Resolve training device from CLI argument."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")

    return torch.device(device_arg)


def make_json_safe_idx_to_label(idx_to_label: dict[int, str]) -> dict[str, str]:
    """Convert integer keys to strings for JSON output."""
    return {str(idx): label for idx, label in idx_to_label.items()}


def save_checkpoint(
    checkpoint_path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: float,
    num_classes: int,
    class_labels: list[str],
    label_to_idx: dict[str, int],
    idx_to_label: dict[int, str],
    training_metadata: dict,
) -> None:
    """Save a runtime-compatible training checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_name": "baseline",
            "num_classes": num_classes,
            "class_labels": class_labels,
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "training": training_metadata,
        },
        checkpoint_path,
    )


def main() -> int:
    """Main entrypoint for the training script."""
    parser = argparse.ArgumentParser(description="Training entrypoint for the baseline model")
    parser.add_argument("--config", default="configs/train.yml", help="Path to training config file")

    parser.add_argument("--data-dir", default=None, help="Override root data directory, e.g. /data")
    parser.add_argument(
        "--checkpoints-dir",
        default=None,
        help="Override checkpoints directory, e.g. /checkpoints",
    )
    parser.add_argument("--logs-dir", default=None, help="Override logs directory")

    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")

    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Training device")
    parser.add_argument("--expected-classes", type=int, default=None, help="Fail if detected class count differs")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker count")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Use AMP on CUDA")

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.data_dir:
        base_data = Path(args.data_dir)
        manifest_path = base_data / "manifests" / "manifest.jsonl"
        raw_dir = base_data / "raw"
    else:
        manifest_path = Path(config["directories"]["manifests"]) / "manifest.jsonl"
        raw_dir = Path(config["directories"]["raw"])

    if args.checkpoints_dir:
        checkpoints_dir = Path(args.checkpoints_dir)
    else:
        checkpoints_dir = Path(config["directories"]["checkpoints"])

    if args.logs_dir:
        logs_dir = Path(args.logs_dir)
    elif args.checkpoints_dir:
        logs_dir = Path(args.checkpoints_dir)
    else:
        logs_dir = Path(config["directories"]["logs"])

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    epochs = args.epochs if args.epochs is not None else config["training"]["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else config["training"]["batch_size"]
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else config["training"]["learning_rate"]
    )

    if epochs <= 0:
        raise ValueError("epochs must be > 0")

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")

    if args.save_every <= 0:
        raise ValueError("save_every must be > 0")

    if args.num_workers is None:
        num_workers = min(16, os.cpu_count() or 1)
    else:
        num_workers = args.num_workers

    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")

    if args.prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be > 0")

    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0

    print(f"Device selected for training: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"AMP enabled: {use_amp}")
    print(f"Manifest: {manifest_path}")
    print(f"Raw data directory: {raw_dir}")
    print(f"Checkpoints directory: {checkpoints_dir}")
    print(f"Logs directory: {logs_dir}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"DataLoader workers: {num_workers}")
    print(f"Pin memory: {pin_memory}")
    print(f"Persistent workers: {persistent_workers}")
    print(f"Prefetch factor: {args.prefetch_factor}")

    train_loader = get_dataloader(
        manifest_path=manifest_path,
        data_dir=raw_dir,
        split="train",
        batch_size=batch_size,
        config_path=Path("configs/data_pipeline.yml"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    label_to_idx = train_loader.dataset.label_to_idx
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    class_labels = [idx_to_label[idx] for idx in sorted(idx_to_label)]

    num_classes = len(class_labels)
    train_samples = len(train_loader.dataset)

    if num_classes == 0:
        raise ValueError("No classes detected in training split")

    if train_samples == 0:
        raise ValueError("No training samples detected")

    if args.expected_classes is not None and num_classes != args.expected_classes:
        raise ValueError(
            f"Expected {args.expected_classes} classes, but detected {num_classes}. "
            "Check manifest split and dataset structure."
        )

    print(f"Training samples detected: {train_samples}")
    print(f"Number of classes detected: {num_classes}")
    print("Class labels:")
    for idx, label in enumerate(class_labels):
        print(f"  {idx}: {label}")

    classes_path = checkpoints_dir / "classes.json"
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_classes": num_classes,
                "class_labels": class_labels,
                "label_to_idx": label_to_idx,
                "idx_to_label": make_json_safe_idx_to_label(idx_to_label),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    runtime_labels_path = checkpoints_dir / "runtime_class_labels.yml"
    with open(runtime_labels_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "inference": {
                    "class_labels": class_labels,
                }
            },
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"Class mapping saved to: {classes_path}")
    print(f"Runtime class labels snippet saved to: {runtime_labels_path}")

    model = BaselineBehaviorModel(num_classes=num_classes)
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler(device="cuda", enabled=use_amp)

    log_file = logs_dir / "training_log.jsonl"
    print(f"Training log saved to: {log_file}")

    training_metadata = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": str(device),
        "amp": use_amp,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "prefetch_factor": args.prefetch_factor,
        "manifest_path": str(manifest_path),
        "raw_dir": str(raw_dir),
        "classes_path": str(classes_path),
        "runtime_labels_path": str(runtime_labels_path),
        "train_samples": train_samples,
    }

    start_time = time.time()
    avg_loss = None

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        batches = 0

        for videos, labels in train_loader:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=use_amp):
                outputs = model(videos)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            batches += 1

        if batches == 0:
            raise ValueError("No batches produced by train DataLoader")

        avg_loss = epoch_loss / batches

        log_entry = {
            "epoch": epoch,
            "epochs": epochs,
            "average_loss": round(avg_loss, 6),
            "batches": batches,
            "train_samples": train_samples,
            "num_classes": num_classes,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": str(device),
            "amp": use_amp,
            "time_elapsed": round(time.time() - start_time, 2),
        }

        print(f"Epoch [{epoch}/{epochs}]")
        print(f"Loss: {avg_loss:.6f}")
        print(f"Batches: {batches}")

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        if epoch % args.save_every == 0 or epoch == epochs:
            checkpoint_path = checkpoints_dir / f"baseline_{num_classes}classes_epoch_{epoch}.pth"
            save_checkpoint(
                checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                loss=avg_loss,
                num_classes=num_classes,
                class_labels=class_labels,
                label_to_idx=label_to_idx,
                idx_to_label=idx_to_label,
                training_metadata=training_metadata,
            )
            print(f"Checkpoint saved to: {checkpoint_path}")

    final_checkpoint_path = checkpoints_dir / f"baseline_{num_classes}classes_final.pth"
    save_checkpoint(
        final_checkpoint_path,
        epoch=epochs,
        model=model,
        optimizer=optimizer,
        loss=avg_loss,
        num_classes=num_classes,
        class_labels=class_labels,
        label_to_idx=label_to_idx,
        idx_to_label=idx_to_label,
        training_metadata=training_metadata,
    )

    summary_path = checkpoints_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint_path": str(final_checkpoint_path),
                "classes_path": str(classes_path),
                "runtime_labels_path": str(runtime_labels_path),
                "log_file": str(log_file),
                "num_classes": num_classes,
                "class_labels": class_labels,
                "train_samples": train_samples,
                "final_loss": avg_loss,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "device": str(device),
                "amp": use_amp,
                "num_workers": num_workers,
                "time_elapsed": round(time.time() - start_time, 2),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nSUCCESSFULLY TRAINED MODEL")
    print(f"FINAL MODEL SAVED TO: {final_checkpoint_path}")
    print(f"CLASS MAPPING SAVED TO: {classes_path}")
    print(f"RUNTIME LABELS SAVED TO: {runtime_labels_path}")
    print(f"TRAINING SUMMARY SAVED TO: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())