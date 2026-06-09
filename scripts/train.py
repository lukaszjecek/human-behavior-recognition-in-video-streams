"""Module for training the baseline behavior recognition model."""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.data.loader import get_dataloader
from src.models.baseline import BaselineBehaviorModel


def main() -> int:
    """Main entrypoint for the training script."""
    parser = argparse.ArgumentParser(description="Trainin entrypoint for the baseline mdl")
    parser.add_argument("--config", default="configs/train.yml", help="Path to config file")
    # server trainin args
    parser.add_argument("--data-dir", default=None, help="Override root data directory (server use)")
    parser.add_argument("--checkpoints-dir", default=None, help="Override checkpoints directory (server use)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for powerful GPUs")
    args = parser.parse_args()

    # config loadin
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # path settin
    if args.data_dir:
        base_data = Path(args.data_dir)
        manifest_path = base_data / 'manifests' / 'manifest.jsonl'
        raw_dir = base_data / 'raw'
    else:
        manifest_path = Path(config['directories']['manifests']) / 'manifest.jsonl'
        raw_dir = Path(config['directories']['raw'])
    if args.checkpoints_dir:
        checkpoints_dir = Path(args.checkpoints_dir)
        logs_dir = Path(args.checkpoints_dir) # Zapisujemy logi też tam, żeby Łukasz miał wszystko w jednym miejscu
    else:
        checkpoints_dir = Path(config['directories']['checkpoints'])
        logs_dir = Path(config['directories']['logs'])

    # output fleders
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # parameters
    epochs = config['training']['epochs']
    batch_size = args.batch_size if args.batch_size else config['training']['batch_size']
    learning_rate = config['training']['learning_rate']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected for training {device}")
    print(f"Using batch size: {batch_size}")

    train_loader = get_dataloader(
        manifest_path=manifest_path,
        data_dir=raw_dir,
        split='train',
        batch_size=batch_size,
        config_path=Path('configs/data_pipeline.yml') # original data cfg
    )

    num_classes = len(train_loader.dataset.label_to_idx)
    print(f'Number of classes detected: {num_classes}')

    # model init
    model = BaselineBehaviorModel(num_classes=num_classes)
    model.to(device)
    model.train()

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    log_file = logs_dir / "training_log.jsonl"
    print(f"logs saved to: {log_file}")

    # learnin loop begin
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epochs_loss = 0.0
        batches = 0

        for videos, labels in train_loader:
            videos, labels = videos.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # Forward
            outputs = model(videos)
            loss = criterion(outputs, labels)

            # backward
            loss.backward()
            optimizer.step()

            epochs_loss += loss.item()
            batches += 1

        avg_loss = epochs_loss / batches

        # DoD logging - to console and file
        log_entry = {
            "epoch": epoch,
            "average_loss": round(avg_loss, 4),
            "time_elapsed": round(time.time() - start_time, 2)
        }
        print(f"Epoch [{epoch}/{epochs}] \n Loss: {avg_loss:.4f} \n")

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + "\n")


    # DoD checkpoint save
    checkpoints_path = checkpoints_dir / f'baseline_epoch_{epochs}.pth'
    torch.save({
        'epoch': epochs,
        'model_name': 'baseline',
        'num_classes': num_classes,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, checkpoints_path)

    print("\n SUCCESSFULLY TRAINED MODEL")
    print(f"MODEL SAVED TO: {checkpoints_path}")

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
