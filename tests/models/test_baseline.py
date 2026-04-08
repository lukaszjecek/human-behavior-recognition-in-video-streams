import torch

from src.models.baseline import BaselineBehaviorModel


def test_baseline_model_synthetic_input():
    batch_size = 2
    time_steps = 16  # Frames
    channels = 3
    height = 224
    width = 224
    num_classes = 5

    model = BaselineBehaviorModel(num_classes=num_classes)
    model.eval()

    dummy_input = torch.randn(batch_size, time_steps, channels, height, width)

    with torch.no_grad():
        output = model(dummy_input)

    expected_shape = (batch_size, num_classes)
    assert output.shape == expected_shape, (f"Oczekiwano kształtu {expected_shape}, "
                                            f"otrzymano {output.shape}")