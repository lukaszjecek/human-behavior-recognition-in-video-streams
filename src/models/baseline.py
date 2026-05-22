"""Baseline video behavior classification model."""

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18


class BaselineBehaviorModel(nn.Module):
    """Baseline R3D-18 model for behavior classification."""

    def __init__(self, num_classes: int = 5) -> None:
        """Initialize the baseline model with a class-specific output head."""
        super().__init__()
        self.model = r3d_18(weights=None)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass on a batch of video tensors."""
        x = x.permute(0, 2, 1, 3, 4)
        return self.model(x)
