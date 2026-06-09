"""Module for training the baseline behavior recognition model."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

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
    scaler: GradScaler,
    loss: float,
    num_classes: int,
    class_labels: list[str],
    label_to_idx: dict[str, int],
    idx_to_label: dict[int, str],
    training_metadata: dict[str, Any],
    best_val_accuracy: float | None,
    best_val_loss: float | None,
    metrics: dict[str, Any],
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
            "scaler_state_dict": scaler.state_dict(),
            "loss": loss,
            "best_val_accuracy": best_val_accuracy,
            "best_val_loss": best_val_loss,
            "metrics": metrics,
            "training": training_metadata,
        },
        checkpoint_path,
    )


def load_resume_checkpoint(
    checkpoint_path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    expected_num_classes: int,
    expected_class_labels: list[str],
) -> tuple[int, float | None, float | None]:
    """Load model, optimizer and AMP scaler state from a checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError("Resume checkpoint must contain a mapping/object payload")

    checkpoint_num_classes = checkpoint.get("num_classes")
    if checkpoint_num_classes is not None and checkpoint_num_classes != expected_num_classes:
        raise ValueError(
            f"Resume checkpoint has num_classes={checkpoint_num_classes}, "
            f"but current dataset has num_classes={expected_num_classes}"
        )

    checkpoint_class_labels = checkpoint.get("class_labels")
    if checkpoint_class_labels is not None and checkpoint_class_labels != expected_class_labels:
        raise ValueError(
            "Resume checkpoint class_labels do not match the current manifest mapping. "
            "Refusing to resume with incompatible label order."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError("Resume checkpoint missing model_state_dict")

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        print("[WARNING] Resume checkpoint missing optimizer_state_dict; optimizer starts fresh.")

    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    else:
        print("[WARNING] Resume checkpoint missing scaler_state_dict; AMP scaler starts fresh.")

    completed_epoch = int(checkpoint.get("epoch", 0))
    start_epoch = completed_epoch + 1

    best_val_accuracy = checkpoint.get("best_val_accuracy")
    best_val_loss = checkpoint.get("best_val_loss")

    return start_epoch, best_val_accuracy, best_val_loss


def build_class_mapping(train_loader) -> tuple[dict[str, int], dict[int, str], list[str]]:
    """Build stable class mapping from the train split."""
    label_to_idx = train_loader.dataset.label_to_idx
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    class_labels = [idx_to_label[idx] for idx in sorted(idx_to_label)]

    return label_to_idx, idx_to_label, class_labels


def compute_accuracy(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    top_k: int = 1,
) -> int:
    """Return number of correct predictions for top-k accuracy."""
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    k = min(top_k, outputs.shape[1])
    _, predictions = outputs.topk(k, dim=1)
    correct = predictions.eq(labels.view(-1, 1).expand_as(predictions))

    return int(correct.any(dim=1).sum().item())


def train_one_epoch(
    *,
    model: torch.nn.Module,
    train_loader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
) -> dict[str, Any]:
    """Run one training epoch and return train metrics."""
    model.train()

    epoch_loss = 0.0
    batches = 0
    samples = 0
    correct_top1 = 0

    for videos, labels in train_loader:
        videos = videos.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(videos)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        epoch_loss += loss.item()
        batches += 1
        samples += batch_size
        correct_top1 += compute_accuracy(outputs.detach(), labels, top_k=1)

    if batches == 0 or samples == 0:
        raise ValueError("No batches produced by train DataLoader")

    return {
        "loss": epoch_loss / batches,
        "accuracy": correct_top1 / samples,
        "batches": batches,
        "samples": samples,
    }


def evaluate(
    *,
    model: torch.nn.Module,
    data_loader,
    criterion: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
    num_classes: int,
) -> dict[str, Any]:
    """Evaluate model on a validation split."""
    model.eval()

    epoch_loss = 0.0
    batches = 0
    samples = 0
    correct_top1 = 0
    correct_top5 = 0
    use_top5 = num_classes >= 5

    with torch.no_grad():
        for videos, labels in data_loader:
            videos = videos.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")

            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(videos)
                loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            epoch_loss += loss.item()
            batches += 1
            samples += batch_size
            correct_top1 += compute_accuracy(outputs, labels, top_k=1)

            if use_top5:
                correct_top5 += compute_accuracy(outputs, labels, top_k=5)

    if batches == 0 or samples == 0:
        return {
            "loss": None,
            "accuracy": None,
            "top5_accuracy": None,
            "batches": batches,
            "samples": samples,
        }

    return {
        "loss": epoch_loss / batches,
        "accuracy": correct_top1 / samples,
        "top5_accuracy": (correct_top5 / samples) if use_top5 else None,
        "batches": batches,
        "samples": samples,
    }


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
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True, help="Run validation after each epoch")
    parser.add_argument("--resume-from", default=None, help="Path to checkpoint used to resume training")

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
    print(f"Validation enabled: {args.validate}")
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
        window_strategy="random",
    )

    label_to_idx, idx_to_label, class_labels = build_class_mapping(train_loader)

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

    val_loader = None
    val_samples = 0

    if args.validate:
        val_loader = get_dataloader(
            manifest_path=manifest_path,
            data_dir=raw_dir,
            split="val",
            batch_size=batch_size,
            config_path=Path("configs/data_pipeline.yml"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=args.prefetch_factor,
            label_to_idx=label_to_idx,
            window_strategy="middle",
        )
        val_samples = len(val_loader.dataset)

        if val_samples == 0:
            print("[WARNING] Validation split is empty. Validation metrics will be skipped.")
            val_loader = None

    print(f"Training samples detected: {train_samples}")
    print(f"Validation samples detected: {val_samples}")
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

    start_epoch = 1
    best_val_accuracy = None
    best_val_loss = None
    resumed_from = None

    if args.resume_from:
        resume_path = Path(args.resume_from)
        start_epoch, best_val_accuracy, best_val_loss = load_resume_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            expected_num_classes=num_classes,
            expected_class_labels=class_labels,
        )
        resumed_from = str(resume_path)
        print(f"Resumed from checkpoint: {resume_path}")
        print(f"Next epoch: {start_epoch}")
        print(f"Best val accuracy from checkpoint: {best_val_accuracy}")
        print(f"Best val loss from checkpoint: {best_val_loss}")

    if start_epoch > epochs:
        raise ValueError(
            f"Resume checkpoint already completed epoch {start_epoch - 1}, "
            f"but requested epochs={epochs}"
        )

    log_file = logs_dir / "training_log.jsonl"
    print(f"Training log saved to: {log_file}")

    training_metadata = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": str(device),
        "amp": use_amp,
        "validate": args.validate,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "prefetch_factor": args.prefetch_factor,
        "manifest_path": str(manifest_path),
        "raw_dir": str(raw_dir),
        "classes_path": str(classes_path),
        "runtime_labels_path": str(runtime_labels_path),
        "train_samples": train_samples,
        "val_samples": val_samples,
        "resumed_from": resumed_from,
    }

    start_time = time.time()
    final_train_loss = None
    final_train_accuracy = None
    final_val_loss = None
    final_val_accuracy = None
    final_val_top5_accuracy = None

    best_checkpoint_path = checkpoints_dir / f"baseline_{num_classes}classes_best.pth"

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )

        final_train_loss = train_metrics["loss"]
        final_train_accuracy = train_metrics["accuracy"]

        val_metrics = {
            "loss": None,
            "accuracy": None,
            "top5_accuracy": None,
            "batches": 0,
            "samples": 0,
        }

        if val_loader is not None:
            val_metrics = evaluate(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
                num_classes=num_classes,
            )

        final_val_loss = val_metrics["loss"]
        final_val_accuracy = val_metrics["accuracy"]
        final_val_top5_accuracy = val_metrics["top5_accuracy"]

        metrics = {
            "epoch": epoch,
            "epochs": epochs,
            "train_loss": round(final_train_loss, 6),
            "train_accuracy": round(final_train_accuracy, 6),
            "train_batches": train_metrics["batches"],
            "train_samples": train_metrics["samples"],
            "val_loss": round(final_val_loss, 6) if final_val_loss is not None else None,
            "val_accuracy": round(final_val_accuracy, 6) if final_val_accuracy is not None else None,
            "val_top5_accuracy": round(final_val_top5_accuracy, 6) if final_val_top5_accuracy is not None else None,
            "val_batches": val_metrics["batches"],
            "val_samples": val_metrics["samples"],
            "num_classes": num_classes,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": str(device),
            "amp": use_amp,
            "time_elapsed": round(time.time() - start_time, 2),
        }

        print(f"Epoch [{epoch}/{epochs}]")
        print(f"Train loss: {final_train_loss:.6f}")
        print(f"Train accuracy: {final_train_accuracy:.6f}")
        if final_val_loss is not None:
            print(f"Val loss: {final_val_loss:.6f}")
        if final_val_accuracy is not None:
            print(f"Val accuracy: {final_val_accuracy:.6f}")
        if final_val_top5_accuracy is not None:
            print(f"Val top-5 accuracy: {final_val_top5_accuracy:.6f}")

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        current_is_best = False

        if final_val_accuracy is not None:
            if best_val_accuracy is None or final_val_accuracy > best_val_accuracy:
                best_val_accuracy = final_val_accuracy
                best_val_loss = final_val_loss
                current_is_best = True
        elif final_val_loss is not None:
            if best_val_loss is None or final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                current_is_best = True

        if current_is_best:
            save_checkpoint(
                best_checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                loss=final_train_loss,
                num_classes=num_classes,
                class_labels=class_labels,
                label_to_idx=label_to_idx,
                idx_to_label=idx_to_label,
                training_metadata=training_metadata,
                best_val_accuracy=best_val_accuracy,
                best_val_loss=best_val_loss,
                metrics=metrics,
            )
            print(f"Best checkpoint saved to: {best_checkpoint_path}")

        if epoch % args.save_every == 0 or epoch == epochs:
            checkpoint_path = checkpoints_dir / f"baseline_{num_classes}classes_epoch_{epoch}.pth"
            save_checkpoint(
                checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                loss=final_train_loss,
                num_classes=num_classes,
                class_labels=class_labels,
                label_to_idx=label_to_idx,
                idx_to_label=idx_to_label,
                training_metadata=training_metadata,
                best_val_accuracy=best_val_accuracy,
                best_val_loss=best_val_loss,
                metrics=metrics,
            )
            print(f"Checkpoint saved to: {checkpoint_path}")

    final_checkpoint_path = checkpoints_dir / f"baseline_{num_classes}classes_final.pth"
    save_checkpoint(
        final_checkpoint_path,
        epoch=epochs,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        loss=final_train_loss,
        num_classes=num_classes,
        class_labels=class_labels,
        label_to_idx=label_to_idx,
        idx_to_label=idx_to_label,
        training_metadata=training_metadata,
        best_val_accuracy=best_val_accuracy,
        best_val_loss=best_val_loss,
        metrics={
            "train_loss": final_train_loss,
            "train_accuracy": final_train_accuracy,
            "val_loss": final_val_loss,
            "val_accuracy": final_val_accuracy,
            "val_top5_accuracy": final_val_top5_accuracy,
        },
    )

    summary_path = checkpoints_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint_path": str(final_checkpoint_path),
                "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path.exists() else None,
                "classes_path": str(classes_path),
                "runtime_labels_path": str(runtime_labels_path),
                "log_file": str(log_file),
                "num_classes": num_classes,
                "class_labels": class_labels,
                "train_samples": train_samples,
                "val_samples": val_samples,
                "final_train_loss": final_train_loss,
                "final_train_accuracy": final_train_accuracy,
                "final_val_loss": final_val_loss,
                "final_val_accuracy": final_val_accuracy,
                "final_val_top5_accuracy": final_val_top5_accuracy,
                "best_val_accuracy": best_val_accuracy,
                "best_val_loss": best_val_loss,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "device": str(device),
                "amp": use_amp,
                "num_workers": num_workers,
                "resumed_from": resumed_from,
                "time_elapsed": round(time.time() - start_time, 2),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nSUCCESSFULLY TRAINED MODEL")
    print(f"FINAL MODEL SAVED TO: {final_checkpoint_path}")
    if best_checkpoint_path.exists():
        print(f"BEST MODEL SAVED TO: {best_checkpoint_path}")
    print(f"CLASS MAPPING SAVED TO: {classes_path}")
    print(f"RUNTIME LABELS SAVED TO: {runtime_labels_path}")
    print(f"TRAINING SUMMARY SAVED TO: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
