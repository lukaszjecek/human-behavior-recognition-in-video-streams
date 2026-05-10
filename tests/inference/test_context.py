"""Local verification for Context Module determinism."""

from PIL import Image

from src.inference.context_adapter import ContextModule


def test_context_determinism() -> None:
    """Verify that the module returns deterministic context outputs."""
    module = ContextModule()
    fake_frame = Image.new('RGB', (224, 224), color=(128, 128, 128))
    
    res1 = module.get_context(fake_frame)
    res2 = module.get_context(fake_frame)
    
    assert res1["scene_tag"] == res2["scene_tag"]
    assert res1["confidence"] == res2["confidence"]