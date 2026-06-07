"""Minimal context module for Sprint 3 context-aware alerting.

Imports ``torch`` and ``torchvision`` lazily so that the rest of the inference
package remains importable even when these heavy dependencies are absent
(e.g. in lightweight test environments or CI runners without GPU support).
When the dependencies are unavailable :meth:`ContextModule.get_context` returns
a safe fallback ``{"scene_tag": "unknown", "confidence": 0.0}`` and logs a
one-time WARNING instead of raising an exception.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy dependencies — resolved at *instance creation* time, not at
# module import time.  This lets ``from src.inference.context_adapter import
# ContextModule`` succeed even when torch/torchvision are absent.
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn.functional as F
    import torchvision.models as models
    import torchvision.transforms as T
    from PIL.Image import Image

    _DEPS_AVAILABLE: bool = True
except ImportError as _import_err:  # noqa: BLE001
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    models = None  # type: ignore[assignment]
    T = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    _DEPS_AVAILABLE: bool = False
    logger.warning(
        "ContextModule dependencies unavailable (%s). "
        "get_context() will always return scene_tag='unknown'.",
        _import_err,
    )


# Fallback returned whenever context inference is unavailable or fails.
_FALLBACK_CONTEXT: dict[str, Any] = {"scene_tag": "unknown", "confidence": 0.0}


class ContextModule:
    """Extracts security-relevant scene tags from video frames.

    Uses a pre-trained MobileNetV2 backbone to classify each frame into one of
    three scene categories: ``outdoor``, ``indoor``, or ``vehicle_setting``.
    Falls back to ``scene_tag='unknown'`` when:

    - ``torch`` or ``torchvision`` are not installed,
    - model weights cannot be downloaded or loaded,
    - ``get_context()`` encounters any runtime error.

    This makes the class safe to construct in environments that do not have GPU
    or vision dependencies, and safe to call from :class:`InferenceEventPipeline`
    without guarding every call site.
    """

    CONTEXT_MAP = {
        "outdoor": (
            919, 920,  # Street sign, traffic light
            673, 555,  # Mouse (picket fence), fence
            970, 971,  # Alp, valley (landscape)
            704, 705,  # Parking meter, park bench
        ),
        "indoor": (
            498, 500,  # Cinema, home theater
            508, 603,  # Computer desk, heater
            724, 743,  # Office, prison
            849, 850,  # Store, sliding door
        ),
        "vehicle_setting": (
            817, 511,  # Bus, car
            407, 751,  # Ambulance, racer
            654, 656,  # Minibus, minivan
        ),
    }

    def __init__(self) -> None:
        """Initialize the ContextModule with pre-trained MobileNetV2.

        Never raises.  When ``torch``/``torchvision`` are not installed, or
        when the model weights cannot be loaded, the instance enters a
        *degraded* mode: ``_available`` is set to ``False`` and
        :meth:`get_context` returns
        ``{"scene_tag": "unknown", "confidence": 0.0}`` on every call.
        This guarantees that the inference pipeline can always construct a
        ``ContextModule`` safely and that a missing dependency never propagates
        as an exception to the caller.
        """
        self._available: bool = False
        self.model = None
        self.transform = None

        if not _DEPS_AVAILABLE:
            # Warning already emitted at module-import time; no need to repeat it.
            return

        try:
            self.model = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.IMAGENET1K_V2
            )
            self.model.eval()

            self.transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self._available = True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ContextModule: failed to load MobileNetV2 weights (%s). "
                "get_context() will return fallback.",
                exc,
            )


    def get_context(self, frame_tensor: "Image") -> dict:
        """Extract context from a given frame.

        Args:
            frame_tensor: PIL Image representing one video frame.

        Returns:
            dict: ``{"scene_tag": str, "confidence": float}``.
                  Returns ``{"scene_tag": "unknown", "confidence": 0.0}`` when
                  the model is unavailable or inference fails.
        """
        if not self._available or self.model is None or self.transform is None:
            return dict(_FALLBACK_CONTEXT)

        try:
            with torch.no_grad():
                img = self.transform(frame_tensor).unsqueeze(0)
                output = self.model(img)

                probabilities = F.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

                idx = predicted_idx.item()
                conf_val = round(confidence.item(), 3)

                for context, indexes in self.CONTEXT_MAP.items():
                    if idx in indexes:
                        return {"scene_tag": context, "confidence": conf_val}
                return {"scene_tag": "unknown", "confidence": conf_val}

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ContextModule.get_context() failed, returning fallback: %s", exc
            )
            return dict(_FALLBACK_CONTEXT)