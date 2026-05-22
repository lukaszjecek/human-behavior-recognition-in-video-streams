import torch.nn as nn


class DummyBehaviorModel(nn.Module):
    """Simple 3D model that accepts [B, T, C, H, W] tensors
    and returns class logits.
    """

    def __init__(self, num_classes=65):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(3, num_classes)  # Input: 3 channels (RGB)

    def forward(self, x):
        batch_size = x.shape[0]

        x = x.permute(0, 2, 1, 3, 4)

        x = self.pool(x)  # [B, C, 1, 1, 1]
        x = x.view(batch_size, -1)  # [B, C]
        return self.fc(x)