from src.inference.engine import InferenceEngine


def test_inference_engine_initialization():
    """Test if the engine initializes correctly with the frame buffer."""
    engine = InferenceEngine(window_size=16)

    assert engine.buffer.window_size == 16
    assert engine.model is None


def test_inference_engine_process_frame():
    """Test the frame processing logic and stub prediction trigger."""
    class DummyModel:
        pass  # Just a placeholder for testing

    engine = InferenceEngine(window_size=3, model=DummyModel())

    # Buffer is not full yet
    assert engine.process_frame("frame_1") is None
    assert engine.process_frame("frame_2") is None

    # Buffer is now full, it should trigger inference and return the stub prediction
    assert engine.process_frame("frame_3") == "prediction_stub"

    # Next frame shifts the window (overflow), it should still predict
    assert engine.process_frame("frame_4") == "prediction_stub"
