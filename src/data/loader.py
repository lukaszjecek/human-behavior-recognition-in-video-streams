import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset, DataLoader

from src.data.preprocess import VideoPreprocessor


class VideoDataset(Dataset):
    """
    Dataset class that reads a JSONL manifest and prepares video tensors.
    """

    def __init__(self, manifest_path: Path, data_dir: Path, split: str = "train", config_path: Path = None):
        self.data_dir = data_dir
        self.samples = []

        if config_path and config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        else:
            config = {}

        pipeline_cfg = config.get("pipeline", {})
        res = pipeline_cfg.get("target_resolution", [224, 224])
        t_window = pipeline_cfg.get("temporal_window", 16)

        self.preprocessor = VideoPreprocessor(
            target_resolution=tuple(res),
            temporal_window=t_window,
            stride=t_window
        )

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if entry["split"] == split:
                    self.samples.append(entry)

        unique_labels = sorted(list(set(s["label"] for s in self.samples)))
        self.label_to_idx = {label: i for i, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = self.data_dir / sample["path"]

        windows = self.preprocessor.process(video_path)

        video_tensor = windows[0]
        label_idx = self.label_to_idx[sample["label"]]

        return video_tensor, torch.tensor(label_idx, dtype=torch.long)


def get_dataloader(manifest_path: Path, data_dir: Path, split: str = "train",
                   batch_size: int = 4, config_path: Path = None) -> DataLoader:
    dataset = VideoDataset(manifest_path, data_dir, split, config_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))