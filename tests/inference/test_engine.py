import pytest

from src.inference.engine import (
    InferenceEngine,
    InferenceResult
)


class DummyModel:
    def __call__(self, window):
        return window


def test_invalid_parameters():

    with pytest.raises(ValueError):
        InferenceEngine(window_size=0)

    with pytest.raises(ValueError):
        InferenceEngine(stride=0)

    with pytest.raises(ValueError):
        InferenceEngine(stride=-1)


def test_first_window_triggers():

    engine = InferenceEngine(
        window_size=3,
        stride=2,
        model=DummyModel()
    )

    assert engine.process_frame("f1") is None
    assert engine.process_frame("f2") is None

    result = engine.process_frame("f3")

    assert isinstance(result, InferenceResult)

    latest = engine.get_latest_result().window

    assert latest == ("f1", "f2", "f3")


def test_window_copy_safety():

    engine = InferenceEngine(
        window_size=2,
        stride=2,
        model=DummyModel()
    )

    engine.process_frame("A")
    engine.process_frame("B")

    window = engine.get_latest_result().window

    with pytest.raises(TypeError):
        window[0] = "CORRUPTED"

    assert engine.get_latest_result().window[0] == "A"


def test_stride_cadence():

    engine = InferenceEngine(
        window_size=3,
        stride=2,
        model=DummyModel()
    )

    triggers = []

    for i in range(10):

        out = engine.process_frame(f"f{i}")

        if out is not None:
            triggers.append(i)

    assert triggers == [2, 4, 6, 8]


def test_metadata_integrity():

    engine = InferenceEngine(
        window_size=3,
        stride=2,
        model=DummyModel()
    )

    engine.process_frame("f1")
    engine.process_frame("f2")
    engine.process_frame("f3")

    result = engine.get_latest_result()

    assert isinstance(result, InferenceResult)

    assert result.start_frame_index == 1
    assert result.end_frame_index == 3


def test_large_stride():

    engine = InferenceEngine(
        window_size=3,
        stride=10,
        model=DummyModel()
    )

    triggers = 0

    for i in range(20):

        if engine.process_frame(i):
            triggers += 1

    assert triggers == 2


def test_reset():

    engine = InferenceEngine(
        window_size=2,
        stride=1,
        model=DummyModel()
    )

    engine.process_frame(1)
    engine.process_frame(2)

    engine.reset()

    assert engine.frame_count == 0

    assert engine.get_latest_result() is None
