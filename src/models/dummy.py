"""Dummy video behavior classification model."""

import torch
import torch.nn as nn


class DummyBehaviorModel(nn.Module):
    """Simple 3D model that accepts video tensors.

    Inputs use the shape [B, T, C, H, W], and outputs are class logits.
    """

    def __init__(self, num_classes: int = 65) -> None:
        """Initialize the pooling layer and classifier head."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(3, num_classes)  # Input: 3 channels (RGB)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass on a batch of video tensors."""
        batch_size = x.shape[0]

        x = x.permute(0, 2, 1, 3, 4)

        x = self.pool(x)  # [B, C, 1, 1, 1]
        x = x.view(batch_size, -1)  # [B, C]
        return self.fc(x)
