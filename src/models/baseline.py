import torch.nn as nn
from torchvision.models.video import r3d_18

class BaselineBehaviorModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = r3d_18(weights=None)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        return self.model(x)