"""Tests for ContextModule — construction safety and fallback behaviour.

Covers:
- Determinism: identical frames produce identical scene_tag / confidence.
- No-raise guarantee: __init__() never raises, even when torch/torchvision
  are unavailable (DoD: context module failures fall back to 'unknown' context
  without failing inference).
- Fallback when deps absent: get_context() returns scene_tag='unknown' and
  confidence=0.0 when _available is False.
"""

from unittest.mock import patch

from PIL import Image

import src.inference.context_adapter as _adapter
from src.inference.context_adapter import ContextModule


def test_context_determinism() -> None:
    """Verify that the module returns deterministic context outputs."""
    module = ContextModule()
    fake_frame = Image.new("RGB", (224, 224), color=(128, 128, 128))

    res1 = module.get_context(fake_frame)
    res2 = module.get_context(fake_frame)

    assert res1["scene_tag"] == res2["scene_tag"]
    assert res1["confidence"] == res2["confidence"]


def test_init_does_not_raise_when_deps_unavailable() -> None:
    """ContextModule.__init__() must never raise, even when deps are absent.

    Simulates the environment where torch/torchvision are not installed by
    patching the module-level _DEPS_AVAILABLE flag.  The constructor must
    return a valid, degraded instance instead of raising RuntimeError.
    """
    with patch.object(_adapter, "_DEPS_AVAILABLE", False):
        # Must not raise — this is the core DoD requirement.
        module = ContextModule()

    assert module._available is False  # noqa: SLF001


def test_get_context_returns_unknown_fallback_when_unavailable() -> None:
    """get_context() must return the unknown-context sentinel when unavailable.

    This covers both the "deps absent" path and the "weight-loading failed"
    path, both of which leave _available=False.
    """
    with patch.object(_adapter, "_DEPS_AVAILABLE", False):
        module = ContextModule()

    fake_frame = Image.new("RGB", (64, 64), color=(0, 0, 0))
    result = module.get_context(fake_frame)

    assert result == {"scene_tag": "unknown", "confidence": 0.0}