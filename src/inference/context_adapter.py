"""Minimal context module for Sprint 3 context-aware alerting."""

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL.Image import Image


class ContextModule:
    """Extracts scene tags from video frames without retraining."""

    def __init__(self) -> None:
        """Initialize the ContextModule with pre-trained MobileNetV2."""
        self.model = models.mobilenet_v2(weights=models.MobileNetV2_Weights.IMAGENET1K_V2)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.context_map = {
            "outdoor": range(400, 900),
            "indoor": range(0, 400),
            "vehicle_setting": range(900, 1000),
        }

    def get_context(self, frame_tensor: Image) -> dict:
        """Extract context from a given frame.

        Args:
            frame_tensor: PIL Image or Tensor frame.

        Returns:
            dict: Context output contract with scene_tag and confidence.
        """
        with torch.no_grad():
            img = self.transform(frame_tensor).unsqueeze(0)
            output = self.model(img)

            probabilities = F.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

            idx = predicted_idx.item()
            conf_val = round(confidence.item(), 3)

            for context, indexes in self.context_map.items():
                if idx in indexes:
                    return {"scene_tag": context, "confidence": conf_val}
            return {"scene_tag": "unknown", "confidence": conf_val}