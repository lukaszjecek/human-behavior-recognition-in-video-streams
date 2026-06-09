"""Dataset and dataloader helpers for video manifests."""

import json
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.data.preprocess import VideoPreprocessor


class VideoDataset(Dataset):
    """Dataset class that reads a JSONL manifest and prepares video tensors."""

    def __init__(
        self,
        manifest_path: Path,
        data_dir: Path,
        split: str = "train",
        config_path: Path | None = None,
        label_to_idx: dict[str, int] | None = None,
        window_strategy: str = "auto",
    ) -> None:
        """Initialize the dataset from a manifest and optional pipeline config."""
        self.data_dir = data_dir
        self.split = split
        self.samples = []
        self.window_strategy = self._resolve_window_strategy(window_strategy, split)

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
            stride=t_window,
        )

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if entry["split"] == split:
                    self.samples.append(entry)

        if label_to_idx is None:
            unique_labels = sorted(list(set(s["label"] for s in self.samples)))
            self.label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = dict(label_to_idx)

        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    @staticmethod
    def _resolve_window_strategy(window_strategy: str, split: str) -> str:
        """Resolve automatic window strategy from split name."""
        allowed = {"auto", "first", "middle", "random"}
        if window_strategy not in allowed:
            raise ValueError(f"window_strategy must be one of {sorted(allowed)}")

        if window_strategy != "auto":
            return window_strategy

        if split == "train":
            return "random"

        return "middle"

    def __len__(self) -> int:
        """Return the number of samples in the selected split."""
        return len(self.samples)

    def _select_window(self, windows: torch.Tensor) -> torch.Tensor:
        """Select one temporal window from preprocessed video windows."""
        num_windows = len(windows)

        if num_windows == 0:
            raise ValueError("Cannot select a window from an empty tensor")

        if self.window_strategy == "first":
            window_idx = 0
        elif self.window_strategy == "middle":
            window_idx = num_windows // 2
        elif self.window_strategy == "random":
            window_idx = random.randrange(num_windows)
        else:
            raise ValueError(f"Unsupported window_strategy: {self.window_strategy}")

        return windows[window_idx]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the preprocessed tensor and label index for a sample."""
        sample = self.samples[idx]
        video_path = self.data_dir / sample["path"]

        windows = self.preprocessor.process(video_path)

        if len(windows) == 0:
            raise ValueError(f"No frames/windows extracted from video: {video_path}")

        label = sample["label"]
        if label not in self.label_to_idx:
            raise KeyError(
                f"Label '{label}' is not present in label_to_idx. "
                "Check split consistency and class mapping."
            )

        video_tensor = self._select_window(windows)
        label_idx = self.label_to_idx[label]

        return video_tensor, torch.tensor(label_idx, dtype=torch.long)


def get_dataloader(
    manifest_path: Path,
    data_dir: Path,
    split: str = "train",
    batch_size: int = 4,
    config_path: Path | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    label_to_idx: dict[str, int] | None = None,
    window_strategy: str = "auto",
) -> DataLoader:
    """Build a dataloader for the requested manifest split."""
    dataset = VideoDataset(
        manifest_path=manifest_path,
        data_dir=data_dir,
        split=split,
        config_path=config_path,
        label_to_idx=label_to_idx,
        window_strategy=window_strategy,
    )

    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": split == "train",
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        dataloader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**dataloader_kwargs)
