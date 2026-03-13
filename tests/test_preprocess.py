import torch
from src.data.preprocess import VideoPreprocessor



def test_preprocessor_shape_and_stride(dummy_video):
    preprocessor = VideoPreprocessor(target_resolution=(128, 128), temporal_window=16, stride=16)
    tensor = preprocessor.process(dummy_video)

    assert tensor.shape == (3, 16, 3, 128, 128)

    preprocessor_32 = VideoPreprocessor(target_resolution=(224, 224), temporal_window=32, stride=10)
    tensor_32 = preprocessor_32.process(dummy_video)

    assert tensor_32.shape == (4, 32, 3, 224, 224)


def test_preprocessor_normalization(dummy_video):
    preprocessor = VideoPreprocessor()
    tensor = preprocessor.process(dummy_video)

    assert tensor.dtype == torch.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0